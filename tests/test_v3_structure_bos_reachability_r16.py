from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.diagnose_v3_structure_bos_reachability_r16 as r16


def _target(target_id: str) -> r16.TargetDefinition:
    return next(value for value in r16.TARGETS if value.target_id == target_id)


def _support(target_id: str) -> dict:
    return next(
        value for value in r16.support_rows() if value["target_id"] == target_id
    )


def test_registered_support_is_prebound_and_identity_exact() -> None:
    rows = r16.support_rows()
    assert len(rows) == 10
    assert len({row["target_id"] for row in rows}) == 10
    assert all(row["expected_matching_swing_count"] == 1 for row in rows)
    assert all(row["expected_matching_bos_count"] == 1 for row in rows)
    assert all(row["expected_admission_count"] == 1 for row in rows)
    assert [
        (row["pivot_ordinal"], row["pending_ordinal"])
        for row in rows
    ] == [
        (6, 8),
        (6, 8),
        (8, 10),
        (6, 8),
        (6, 8),
        (8, 10),
        (8, 10),
        (6, 8),
        (6, 8),
        (8, 10),
    ]
    for target, row in zip(r16.TARGETS, rows, strict=True):
        expected_swing = r16.production_identity(
            r16.sha256_file(r16.STRUCTURE_PROTOCOL),
            r16.SYMBOL,
            int(r16.INSTRUMENT_ID),
            target.timeframe.value,
            target.swing_side,
            row["pivot_start"],
            int(row["target_price_ticks"]),
        )
        expected_bos = r16.production_identity(
            r16.sha256_file(r16.STRUCTURE_PROTOCOL),
            target.timeframe.value,
            target.direction.value,
            expected_swing,
            row["pending_clock"],
        )
        assert row["target_swing_id"] == expected_swing
        assert row["target_bos_id"] == expected_bos
    assert len(r16.support_root(rows)) == 64


@pytest.mark.historical_frozen
def test_implementation_manifest_binds_every_registered_dependency() -> None:
    path = (
        r16.ROOT
        / "configs"
        / "experiments"
        / "EXP-SMC-3.0.1-001-R16-IMPLEMENTATION-FREEZE.json"
    )
    expected = r16.sha256_file(path)
    payload, actual = r16._load_manifest(
        path,
        expected_sha256=expected,
    )
    assert actual == expected
    assert payload["support_root_sha256"] == r16.support_root(
        payload["support_rows"]
    )


def test_literal_ohlc_mirror_is_involutive_and_tick_exact() -> None:
    for candle in (
        *r16.CANONICAL_LONG_PREFIX,
        r16.TERMINAL_CONFIRMED_LONG,
        r16.TERMINAL_WICK_LONG,
        r16.TERMINAL_OPPOSED_SHORT,
    ):
        mirrored = r16.mirror_candle(candle)
        assert r16.mirror_candle(mirrored) == candle
        assert mirrored.open_ticks == 2 * r16.REFERENCE_PRICE_TICKS - candle.open_ticks
        assert mirrored.close_ticks == 2 * r16.REFERENCE_PRICE_TICKS - candle.close_ticks
        assert mirrored.high_ticks == 2 * r16.REFERENCE_PRICE_TICKS - candle.low_ticks
        assert mirrored.low_ticks == 2 * r16.REFERENCE_PRICE_TICKS - candle.high_ticks


@pytest.mark.parametrize(
    "target_id",
    (
        "r16-1h-short-confirmed",
        "r16-4h-long-opposed",
    ),
)
def test_minute_expansion_reconstructs_every_registered_target_candle(
    target_id: str,
) -> None:
    target = _target(target_id)
    template = r16.template_for_target(target)
    intervals = r16.target_intervals(target.timeframe)
    assert len(template) == len(intervals) == 12
    for candle, (start, end) in zip(template, intervals, strict=True):
        bars = r16.expand_candle_to_bars(candle, start=start, end=end)
        assert bars[0].start == start
        assert bars[-1].end == end
        assert round(bars[0].open / r16.TICK_SIZE) == candle.open_ticks
        assert round(max(bar.high for bar in bars) / r16.TICK_SIZE) == candle.high_ticks
        assert round(min(bar.low for bar in bars) / r16.TICK_SIZE) == candle.low_ticks
        assert round(bars[-1].close / r16.TICK_SIZE) == candle.close_ticks


def test_future_read_guard_fails_after_boundary_without_consuming_a_bar() -> None:
    target = _target("r16-1h-short-confirmed")
    bars = r16.target_bars(target)
    guard = r16.FutureReadGuard(bars)
    first = next(guard)
    assert first == bars[0]
    guard.mark_boundary_processed()
    with pytest.raises(RuntimeError, match="after registered resolution"):
        next(guard)
    assert guard.future_reads == 1


@pytest.mark.parametrize(
    "target_id",
    (
        "r16-1h-short-confirmed",
        "r16-1h-short-wick",
        "r16-1h-short-opposed",
    ),
)
def test_online_funnel_reaches_exact_prebound_identity_without_future(
    target_id: str,
) -> None:
    result = r16.run_target(_target(target_id), _support(target_id))
    assert result["target_status"] == "pass"
    assert result["matching_swing_count"] == 1
    assert result["matching_bos_count"] == 1
    assert result["admission_count"] == 1
    assert result["future_bars_read_for_resolution"] == 0
    assert result["observed_bucket"] == result["expected_bucket"]
    assert [stage["name"] for stage in result["stages"]] == list(
        r16.STAGE_NAMES
    )
    assert all(stage["reached"] for stage in result["stages"])
    ordinals = [stage["ordinal"] for stage in result["stages"]]
    assert ordinals == sorted(ordinals)


def test_semantic_summary_serialization_is_complete_and_byte_stable(
    monkeypatch,
) -> None:
    rows = r16.support_rows()

    def fixed_result(target, support):
        return {
            "target_id": target.target_id,
            "timeframe": target.timeframe.value,
            "direction": target.direction.value,
            "expected_bucket": target.expected_bucket,
            "target_swing_id": support["target_swing_id"],
            "target_bos_id": support["target_bos_id"],
            "matching_swing_count": 1,
            "matching_bos_count": 1,
            "boundary_ordinal": 11,
            "boundary_clock": "2025-01-06T06:00:00-05:00",
            "stages": [
                {
                    "name": name,
                    "reached": True,
                    "ordinal": 11,
                    "clock": "2025-01-06T06:00:00-05:00",
                    "production_predicate": r16.STAGE_PREDICATES[name],
                }
                for name in r16.STAGE_NAMES
            ],
            "observed_bucket": target.expected_bucket,
            "admission_count": 1,
            "future_bars_read_for_resolution": 0,
            "prefix_commitment_sha256": "a" * 64,
            "rejection_reason": None,
            "target_status": "pass",
        }

    monkeypatch.setattr(r16, "run_target", fixed_result)
    summary = r16.semantic_summary(
        implementation_manifest_sha256="b" * 64,
        registered_support=rows,
    )
    raw = r16.semantic_summary_bytes(summary)
    assert raw.endswith(b"\n")
    assert raw == r16.semantic_summary_bytes(summary)
    assert list(summary) == [
        "format_version",
        "protocol_id",
        "preregistration_sha256",
        "implementation_manifest_sha256",
        "support_root_sha256",
        "target_count",
        "admission_count",
        "extra_admission_count",
        "duplicate_admission_count",
        "future_bars_read_for_resolution",
        "targets",
        "semantic_status",
    ]
    decoded = json.loads(raw)
    assert decoded["target_count"] == decoded["admission_count"] == 10
    assert decoded["semantic_status"] == "pass"


def test_attempt_marker_exists_before_semantic_generator_is_called(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = r16.support_rows()
    run_root = tmp_path / "diagnostic"
    manifest = {
        "protocol_id": r16.PROTOCOL_ID,
        "preregistration_sha256": r16.PREREGISTRATION_SHA256,
        "bindings": {
            "harness_sha256": r16.sha256_file(Path(r16.__file__)),
        },
        "support_rows": list(rows),
        "support_root_sha256": r16.support_root(rows),
        "runs": {
            "diagnostic": {
                "run_id": "test-diagnostic",
                "output_root": "diagnostic",
            },
            "verification": {
                "run_id": "test-verification",
                "output_root": "verification",
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        r16.canonical_json_bytes(manifest, final_lf=True)
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(r16, "ROOT", tmp_path)
    monkeypatch.setattr(
        r16,
        "_load_manifest",
        lambda path, expected_sha256: (manifest, manifest_sha),
    )

    def assert_marker_before_support_derivation():
        assert (run_root / "ATTEMPT.json").is_file()
        return rows

    monkeypatch.setattr(
        r16,
        "support_rows",
        assert_marker_before_support_derivation,
    )

    def assert_marker_then_return(**kwargs):
        assert (run_root / "ATTEMPT.json").is_file()
        return {
            "format_version": 1,
            "protocol_id": r16.PROTOCOL_ID,
            "preregistration_sha256": r16.PREREGISTRATION_SHA256,
            "implementation_manifest_sha256": manifest_sha,
            "support_root_sha256": r16.support_root(rows),
            "target_count": 10,
            "admission_count": 10,
            "extra_admission_count": 0,
            "duplicate_admission_count": 0,
            "future_bars_read_for_resolution": 0,
            "targets": [],
            "semantic_status": "pass",
        }

    monkeypatch.setattr(r16, "semantic_summary", assert_marker_then_return)
    assert (
        r16.execute_run(
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            mode="diagnostic",
        )
        == 0
    )
    assert (run_root / "RESULT.json").is_file()
