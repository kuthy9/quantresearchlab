from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_shadow_file_pilot import (
    COMPLETED,
    INPUT_JOURNAL,
    INPUT_SCHEMA_VERSION,
    PARITY_RECORDS,
    ShadowFilePilotError,
    _read_clock_file,
    preflight_file_pilot_capacity,
    run_file_pilot,
    shadow_clock_input_from_payload,
)
from smc_trader.shadow_live import ShadowLiveError, ShadowLiveRunner


pytestmark = pytest.mark.research_runner


def _payload(index: int) -> dict[str, object]:
    start = pd.Timestamp("2024-06-03T13:30:00Z") + pd.Timedelta(
        index, unit="m"
    )
    end = start + pd.Timedelta(1, unit="m")
    price = 18_500.0 + index * 0.25
    open_ticks = 74_000 + index
    execution_id = f"execution-reality:{end.isoformat()}"
    account_id = f"account-snapshot:{end.isoformat()}"
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "feed_event_id": f"feed:{start.isoformat()}",
        "received_at": (end + pd.Timedelta(5, unit="ms")).isoformat(),
        "bar": {
            "start": start.isoformat(),
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price + 0.25,
            "volume": 100.0 + index,
            "symbol": "NQM4",
            "instrument_id": 13743,
            "synthetic_no_trade": False,
            "data_gap_before_minutes": 0,
            "price_tick_size": 0.25,
            "normalized_ohlc_ticks": [
                open_ticks,
                open_ticks + 2,
                open_ticks - 2,
                open_ticks + 1,
            ],
        },
        "execution": {
            "spread_points": 0.25,
            "expected_slippage_points": 0.0,
            "commission_per_contract_per_side": 2.25,
            "quantity": 1,
            "deadline": (end + pd.Timedelta(1, unit="h")).isoformat(),
            "data_age_seconds": 0.0,
            "size_available": 10.0,
            "source": "shadow_file_fixture_bbo",
            "bid": price + 0.25,
            "ask": price + 0.5,
            "bid_size": 10.0,
            "ask_size": 10.0,
            "depth_imbalance": 0.0,
            "anomalies": [],
        },
        "execution_observed_at": end.isoformat(),
        "execution_known_at": end.isoformat(),
        "execution_source_event_id": execution_id,
        "account": {
            "equity": 100_000.0,
            "open_risk_fraction": 0.0,
            "requested_risk_fraction": 0.005,
            "quantity": 1,
            "point_value": 20.0,
            "position": None,
        },
        "account_observed_at": end.isoformat(),
        "account_known_at": end.isoformat(),
        "account_snapshot_id": account_id,
        "source_event_ids": [
            f"market-feed:{start.isoformat()}",
            execution_id,
            account_id,
        ],
        "approved_intents": [],
        "execution_events": [],
    }


def _write_input(path: Path, rows: int) -> None:
    path.write_text(
        "".join(
            json.dumps(_payload(index), sort_keys=True) + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )


def test_file_pilot_completes_only_after_independent_cold_replay(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clocks.jsonl"
    output = tmp_path / "pilot"
    _write_input(source, 4)

    result = run_file_pilot(
        source,
        output,
        checkpoint_clocks=2,
    )

    assert result["status"] == "complete_engineering_file_replay"
    assert result["gate_pass"] is True
    assert result["real_time_live"] is False
    assert result["multi_day_live_pilot_completed"] is False
    assert result["broker_submission_authorized"] is False
    assert result["source_rows"] == result["parity_records"] == 4
    assert result["approved_intents"] == result["execution_events"] == 0
    assert result["external_submission_attempts"] == 0
    assert result["live_record_fingerprint"] == result[
        "cold_record_fingerprint"
    ]
    assert result["live_journal_fingerprint"] == result[
        "cold_journal_fingerprint"
    ]
    assert (output / COMPLETED).is_file()
    assert (output / INPUT_JOURNAL).read_bytes().endswith(b"\n")
    assert len((output / INPUT_JOURNAL).read_text().splitlines()) == 4
    assert len((output / PARITY_RECORDS).read_text().splitlines()) == 4
    assert (output / "_checkpoint/manifest.json").is_file()

    with pytest.raises(ShadowFilePilotError, match="empty output"):
        run_file_pilot(source, output, checkpoint_clocks=2)
    with pytest.raises(ShadowFilePilotError, match="cannot be resumed"):
        run_file_pilot(source, output, resume=True, checkpoint_clocks=2)


def test_capacity_preflight_is_read_only_and_never_promotes_prefix_evidence(
    tmp_path: Path,
) -> None:
    reference_source = tmp_path / "reference.jsonl"
    target_source = tmp_path / "target.jsonl"
    reference_output = tmp_path / "reference-output"
    _write_input(reference_source, 2)
    _write_input(target_source, 4)
    run_file_pilot(reference_source, reference_output, checkpoint_clocks=2)

    result = preflight_file_pilot_capacity(
        target_source,
        reference_output_directory=reference_output,
        output_parent=tmp_path,
    )

    assert result["status"] == "bounded_capacity_preflight_only"
    assert result["target_rows"] == 4
    assert result["reference_rows"] == 2
    assert result["engine_created"] is False
    assert result["clocks_replayed"] == 0
    assert result["historical_reference_promoted_as_current_result"] is False
    assert result["full_6900_replay_authorized"] is False
    assert result["known_nonlinear_consistency_residual"] is True
    assert result["projected_checkpoint_bytes"] >= 2 * (
        reference_output / "_checkpoint" / json.loads(
            (reference_output / "_checkpoint/manifest.json").read_text()
        )["state_file"]
    ).stat().st_size


def test_capacity_preflight_rejects_checkpoint_not_bound_by_completion(
    tmp_path: Path,
) -> None:
    reference_source = tmp_path / "reference.jsonl"
    target_source = tmp_path / "target.jsonl"
    reference_output = tmp_path / "reference-output"
    _write_input(reference_source, 2)
    _write_input(target_source, 4)
    run_file_pilot(reference_source, reference_output, checkpoint_clocks=2)

    checkpoint_path = reference_output / "_checkpoint/manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    fake_state = (
        reference_output / "_checkpoint" / f"state-{'0' * 64}.pkl"
    )
    fake_state.write_bytes(b"x")
    checkpoint["state_file"] = fake_state.name
    checkpoint["state_sha256"] = "0" * 64
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ShadowFilePilotError, match="completed exact prefix"):
        preflight_file_pilot_capacity(
            target_source,
            reference_output_directory=reference_output,
            output_parent=tmp_path,
        )


def test_file_pilot_checkpoint_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clocks.jsonl"
    resumed_output = tmp_path / "resumed"
    clean_output = tmp_path / "clean"
    _write_input(source, 5)

    paused = run_file_pilot(
        source,
        resumed_output,
        checkpoint_clocks=2,
        stop_after_clocks=2,
    )
    assert paused["status"] == "paused_checkpointed_prefix"
    assert paused["source_rows_consumed"] == 2
    assert not (resumed_output / COMPLETED).exists()
    assert len((resumed_output / INPUT_JOURNAL).read_text().splitlines()) == 2

    resumed = run_file_pilot(
        source,
        resumed_output,
        resume=True,
        checkpoint_clocks=2,
    )
    clean = run_file_pilot(
        source,
        clean_output,
        checkpoint_clocks=2,
    )
    for key in (
        "live_record_fingerprint",
        "cold_record_fingerprint",
        "live_journal_fingerprint",
        "cold_journal_fingerprint",
        "parity_audit_id",
    ):
        assert resumed[key] == clean[key]


def test_file_pilot_narrow_schema_and_resume_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    valid = _payload(0)
    assert not shadow_clock_input_from_payload(valid).approved_intents

    legacy = json.loads(json.dumps(valid))
    legacy["schema_version"] = "phase9_shadow_file_input_v1"
    with pytest.raises(ShadowFilePilotError, match="schema version differs"):
        shadow_clock_input_from_payload(legacy)

    missing_grid = json.loads(json.dumps(valid))
    missing_grid["bar"].pop("price_tick_size")
    missing_grid["bar"].pop("normalized_ohlc_ticks")
    with pytest.raises(ShadowFilePilotError, match="bar fields differ"):
        shadow_clock_input_from_payload(missing_grid)

    approval = dict(valid)
    approval["approved_intents"] = [{"unparsed": "must fail closed"}]
    with pytest.raises(ShadowFilePilotError, match="requires zero"):
        shadow_clock_input_from_payload(approval)

    position = json.loads(json.dumps(valid))
    position["account"]["position"] = {"unparsed": "must fail closed"}
    with pytest.raises(ShadowFilePilotError, match="flat account"):
        shadow_clock_input_from_payload(position)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"schema_version":"phase9_shadow_file_input_v2",'
        '"schema_version":"phase9_shadow_file_input_v2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ShadowFilePilotError, match="duplicate JSON key"):
        _read_clock_file(duplicate)

    source = tmp_path / "clocks.jsonl"
    output = tmp_path / "pilot"
    _write_input(source, 3)
    run_file_pilot(
        source,
        output,
        checkpoint_clocks=1,
        stop_after_clocks=1,
    )
    changed_rows = source.read_text(encoding="utf-8").splitlines()
    changed_clock = json.loads(changed_rows[-1])
    changed_clock["bar"]["close"] += 0.25
    changed_clock["bar"]["normalized_ohlc_ticks"][3] += 1
    changed_rows[-1] = json.dumps(changed_clock, sort_keys=True)
    source.write_text("\n".join(changed_rows) + "\n", encoding="utf-8")
    with pytest.raises(ShadowFilePilotError, match="run manifest differs"):
        run_file_pilot(
            source,
            output,
            resume=True,
            checkpoint_clocks=1,
        )


def test_file_pilot_rejects_durable_journal_tamper_on_resume(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clocks.jsonl"
    output = tmp_path / "pilot"
    _write_input(source, 3)
    run_file_pilot(
        source,
        output,
        checkpoint_clocks=1,
        stop_after_clocks=2,
    )
    journal_path = output / INPUT_JOURNAL
    rows = journal_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["feed_event_id"] = "feed:tampered"
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    journal_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ShadowFilePilotError, match="not a checkpoint prefix"):
        run_file_pilot(
            source,
            output,
            resume=True,
            checkpoint_clocks=1,
        )


def test_file_pilot_duplicate_attempts_are_not_misreported_as_records(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clocks.jsonl"
    output = tmp_path / "pilot"
    rows = (_payload(0), _payload(0), _payload(1))
    source.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )

    result = run_file_pilot(source, output, checkpoint_clocks=1)

    assert result["gate_pass"] is True
    assert result["source_rows"] == result["attempted_clocks"] == 3
    assert result["accepted_clocks"] == result["parity_records"] == 2


def test_file_pilot_runtime_drift_persists_the_attempt_and_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clocks.jsonl"
    output = tmp_path / "pilot"
    _write_input(source, 2)

    def _runtime_drift(_: ShadowLiveRunner) -> None:
        raise ShadowLiveError("simulated runtime binding drift")

    monkeypatch.setattr(
        ShadowLiveRunner,
        "_require_runtime_bindings",
        _runtime_drift,
    )
    with pytest.raises(ShadowLiveError, match="runtime binding drift"):
        run_file_pilot(source, output, checkpoint_clocks=1)

    failure = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
    assert failure["source_rows_consumed"] == 1
    assert failure["attempted_clock_journaled"] is True
    assert failure["checkpointed_failure"] is True
    assert len((output / INPUT_JOURNAL).read_text().splitlines()) == 1


def test_file_pilot_pre_journal_failure_is_terminal_without_false_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clocks.jsonl"
    output = tmp_path / "pilot"
    _write_input(source, 2)

    def _fail_before_journal(
        _: ShadowLiveRunner,
        __: object,
    ) -> None:
        raise ShadowLiveError("simulated pre-journal entry failure")

    monkeypatch.setattr(ShadowLiveRunner, "process", _fail_before_journal)
    with pytest.raises(ShadowLiveError, match="pre-journal"):
        run_file_pilot(source, output, checkpoint_clocks=1)

    failure = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
    assert failure["source_rows_consumed"] == 0
    assert failure["attempted_clock_journaled"] is False
    assert failure["checkpointed_failure"] is False
    assert failure["failure"] is None
    assert not (output / "_checkpoint/manifest.json").exists()
    with pytest.raises(ShadowFilePilotError, match="failed file pilot"):
        run_file_pilot(source, output, resume=True, checkpoint_clocks=1)
