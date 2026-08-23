from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading

import pandas as pd
import pytest
import scripts.materialize_shadow_week1_input as week1_materializer

from scripts.materialize_shadow_week1_input import (
    EXPECTED_ROWS,
    EXPECTED_SYNTHETIC_CLOCKS,
    ShadowWeek1MaterializationError,
    _publish_bundle_exclusive,
    _recover_interrupted_bundle,
    _sidecar_path,
    _stage_bundle,
    build_shadow_week1_payloads,
    materialize_shadow_week1_input,
)
from scripts.run_shadow_file_pilot import shadow_clock_input_from_payload
from smc_trader.model import Bar


def _bar(*, synthetic: bool = False) -> Bar:
    return Bar(
        start=pd.Timestamp("2024-06-03T13:30:00Z"),
        open=18_500.0,
        high=18_501.0,
        low=18_499.0,
        close=18_500.25,
        volume=0.0 if synthetic else 100.0,
        symbol="NQM4",
        instrument_id=13743,
        synthetic_no_trade=synthetic,
    )


def _feature(bar: Bar) -> pd.DataFrame:
    observed = bar.end - pd.Timedelta(250, unit="ms")
    return pd.DataFrame(
        [
            {
                "decision_time": bar.end,
                "symbol": bar.symbol,
                "instrument_id": bar.instrument_id,
                "book_observed_at": observed,
                "publisher_id": 1,
                "sequence": 123,
                "bid": 18_500.0,
                "ask": 18_500.25,
                "bid_size": 2.0,
                "ask_size": 6.0,
                "top5_bid_size": 10.0,
                "top5_ask_size": 10.0,
                # Phase-6 stores top-five, while the shadow payload must carry
                # the independently recomputed best-level value.
                "depth_imbalance": 0.0,
                "book_valid": True,
                "book_age_seconds": 0.25,
            }
        ]
    )


def _small_sidecar(output: Path, payload: bytes) -> dict[str, object]:
    return {
        "format_version": 1,
        "output": {
            "path": str(output.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": 1,
            "bytes": len(payload),
        },
    }


def test_payload_recomputes_best_level_age_and_parses_back() -> None:
    bar = _bar()
    payloads = build_shadow_week1_payloads(
        [bar],
        _feature(bar),
        expected_rows=1,
        expected_synthetic_clocks=(),
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["execution"]["depth_imbalance"] == pytest.approx(-0.5)
    assert payload["execution"]["data_age_seconds"] == pytest.approx(0.25)
    assert payload["account"]["equity"] == 100_000.0
    assert payload["account"]["position"] is None
    assert payload["approved_intents"] == payload["execution_events"] == []
    parsed = shadow_clock_input_from_payload(payload)
    assert parsed.bar == bar
    assert parsed.execution.depth_imbalance == pytest.approx(-0.5)
    assert parsed.execution_observed_at == bar.end - pd.Timedelta(250, unit="ms")


def test_payload_rejects_top5_imbalance_mismatch() -> None:
    bar = _bar()
    feature = _feature(bar)
    feature.loc[0, "depth_imbalance"] = 0.1

    with pytest.raises(
        ShadowWeek1MaterializationError,
        match="top-5 depth imbalance",
    ):
        build_shadow_week1_payloads(
            [bar],
            feature,
            expected_rows=1,
            expected_synthetic_clocks=(),
        )


def test_bundle_commit_never_overwrites_late_created_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "week1.jsonl"
    sidecar = _sidecar_path(output)
    payload = b'{"ours":true}\n'
    original_link = os.link

    def late_output_link(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination) == output:
            output.write_bytes(b"late-created-by-other-writer\n")
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(week1_materializer.os, "link", late_output_link)
    with pytest.raises(FileExistsError):
        _publish_bundle_exclusive(
            output,
            payload,
            _small_sidecar(output, payload),
        )

    assert output.read_bytes() == b"late-created-by-other-writer\n"
    assert not sidecar.exists()
    assert not tuple(tmp_path.glob(".*.staging"))


def test_sidecar_publish_failure_leaves_no_orphan_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "week1.jsonl"
    sidecar = _sidecar_path(output)
    payload = b'{"clock":1}\n'
    original_link = os.link

    def fail_sidecar_link(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination) == sidecar:
            raise OSError("simulated sidecar publication failure")
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(week1_materializer.os, "link", fail_sidecar_link)
    with pytest.raises(OSError, match="sidecar publication"):
        _publish_bundle_exclusive(
            output,
            payload,
            _small_sidecar(output, payload),
        )
    assert not output.exists()
    assert not sidecar.exists()
    assert not tuple(tmp_path.glob(".*.staging"))

    monkeypatch.setattr(week1_materializer.os, "link", original_link)
    result = _publish_bundle_exclusive(
        output,
        payload,
        _small_sidecar(output, payload),
    )
    assert output.read_bytes() == payload
    assert json.loads(sidecar.read_text(encoding="utf-8")) == result


def test_stage_collision_never_unlinks_the_other_writers_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "week1.jsonl"
    payload = b'{"clock":1}\n'
    original_write = week1_materializer._write_exclusive
    collided: list[Path] = []

    def collide_on_manifest(path: Path, value: bytes):
        if path.name.endswith(".manifest.staging"):
            path.write_bytes(b"other-writer-stage")
            collided.append(path)
        return original_write(path, value)

    monkeypatch.setattr(
        week1_materializer,
        "_write_exclusive",
        collide_on_manifest,
    )
    with pytest.raises(FileExistsError):
        _stage_bundle(
            output,
            payload,
            _small_sidecar(output, payload),
        )

    assert len(collided) == 1
    assert collided[0].read_bytes() == b"other-writer-stage"
    assert not tuple(tmp_path.glob(".*.output.staging"))


def test_recovery_waits_for_active_sidecar_first_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "week1.jsonl"
    sidecar = _sidecar_path(output)
    payload = b'{"clock":1}\n'
    original_link = os.link
    sidecar_linked = threading.Event()
    recovery_attempted = threading.Event()
    recovery_finished = threading.Event()
    recovered: list[bool] = []

    def recover_during_publish() -> None:
        assert sidecar_linked.wait(timeout=2.0)
        recovery_attempted.set()
        recovered.append(_recover_interrupted_bundle(output))
        recovery_finished.set()

    recovery_thread = threading.Thread(target=recover_during_publish)
    recovery_thread.start()

    def pause_after_sidecar_link(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )
        if Path(destination) == sidecar:
            sidecar_linked.set()
            assert recovery_attempted.wait(timeout=2.0)
            assert not recovery_finished.wait(timeout=0.1)

    monkeypatch.setattr(week1_materializer.os, "link", pause_after_sidecar_link)
    result = _publish_bundle_exclusive(
        output,
        payload,
        _small_sidecar(output, payload),
    )
    recovery_thread.join(timeout=2.0)

    assert not recovery_thread.is_alive()
    assert recovered == [False]
    assert output.read_bytes() == payload
    assert json.loads(sidecar.read_text(encoding="utf-8")) == result
    assert not tuple(tmp_path.glob(".*.staging"))


def test_output_link_success_then_exception_is_still_a_committed_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "week1.jsonl"
    sidecar = _sidecar_path(output)
    payload = b'{"clock":1}\n'
    original_link = os.link

    def raise_after_output_link(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )
        if Path(destination) == output:
            raise OSError("simulated post-link notification failure")

    monkeypatch.setattr(week1_materializer.os, "link", raise_after_output_link)
    result = _publish_bundle_exclusive(
        output,
        payload,
        _small_sidecar(output, payload),
    )

    assert output.read_bytes() == payload
    assert json.loads(sidecar.read_text(encoding="utf-8")) == result
    assert not tuple(tmp_path.glob(".*.staging"))


def test_sidecar_link_success_then_exception_rolls_back_only_owned_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "week1.jsonl"
    sidecar = _sidecar_path(output)
    payload = b'{"clock":1}\n'
    original_link = os.link

    def raise_after_sidecar_link(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )
        if Path(destination) == sidecar:
            raise OSError("simulated post-link notification failure")

    monkeypatch.setattr(week1_materializer.os, "link", raise_after_sidecar_link)
    with pytest.raises(OSError, match="post-link notification"):
        _publish_bundle_exclusive(
            output,
            payload,
            _small_sidecar(output, payload),
        )

    assert not output.exists()
    assert not sidecar.exists()
    assert not tuple(tmp_path.glob(".*.staging"))


def test_interrupted_sidecar_first_publish_is_provably_recovered(
    tmp_path: Path,
) -> None:
    output = tmp_path / "week1.jsonl"
    sidecar = _sidecar_path(output)
    payload = b'{"clock":1}\n'
    _, output_owned, manifest_owned = _stage_bundle(
        output,
        payload,
        _small_sidecar(output, payload),
    )
    os.link(manifest_owned.path, sidecar, follow_symlinks=False)

    assert sidecar.exists() and not output.exists()
    assert _recover_interrupted_bundle(output) is True
    assert not output.exists()
    assert not sidecar.exists()
    assert not output_owned.path.exists()
    assert not manifest_owned.path.exists()

    _publish_bundle_exclusive(
        output,
        payload,
        _small_sidecar(output, payload),
    )
    assert output.exists() and sidecar.exists()


def test_preexisting_sidecar_is_never_treated_as_owned_orphan(
    tmp_path: Path,
) -> None:
    output = tmp_path / "week1.jsonl"
    sidecar = _sidecar_path(output)
    sidecar.write_text('{"user":"owned"}', encoding="utf-8")
    before = sidecar.read_bytes()

    with pytest.raises(FileExistsError, match="overwrite"):
        _publish_bundle_exclusive(
            output,
            b'{"clock":1}\n',
            _small_sidecar(output, b'{"clock":1}\n'),
        )

    assert sidecar.read_bytes() == before
    assert not output.exists()


def test_formal_week1_materialization_is_hash_bound_and_non_overwriting(
    tmp_path: Path,
) -> None:
    output = tmp_path / "week1-shadow-input.jsonl"

    result = materialize_shadow_week1_input(output)

    assert result["status"] == "complete_historical_cold_start_file_input"
    assert result["census"]["rows"] == EXPECTED_ROWS
    assert result["census"]["real_completed"] == EXPECTED_ROWS - 1
    assert result["census"]["synthetic_decision_clocks"] == [
        value.isoformat() for value in EXPECTED_SYNTHETIC_CLOCKS
    ]
    assert result["limits"] == {
        "cold_start": True,
        "warm_state_restored": False,
        "warmup_history_included": False,
        "real_time_live": False,
        "multi_day_live_pilot": False,
        "broker_submission": False,
        "live_account_state": False,
        "raw_mbo_read": False,
        "sealed_holdout_read": False,
        "phase9_gate_closed": True,
    }
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == EXPECTED_ROWS
    first = shadow_clock_input_from_payload(json.loads(lines[0]))
    last = shadow_clock_input_from_payload(json.loads(lines[-1]))
    assert first.bar.symbol == last.bar.symbol == "NQM4"
    assert first.bar.instrument_id == last.bar.instrument_id == 13743
    synthetic = [
        shadow_clock_input_from_payload(json.loads(line))
        for line in lines
        if json.loads(line)["bar"]["synthetic_no_trade"]
    ]
    assert [value.bar.end.tz_convert("UTC") for value in synthetic] == list(
        EXPECTED_SYNTHETIC_CLOCKS
    )
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    assert json.loads(sidecar.read_text(encoding="utf-8")) == result

    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_shadow_week1_input(output)
