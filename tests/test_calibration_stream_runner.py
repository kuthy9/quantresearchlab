from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd
import pytest

from scripts.run_continuous_replay import (
    BRAIN_CALIBRATION_FIELD_TYPES,
    DECISION_FIELD_TYPES,
    FUNNEL_FIELD_TYPES,
    _row,
)
from smc_trader.brain_calibration import BrainCalibrationRecord
from smc_trader.model import (
    AccountState,
    HypothesisSequenceState,
    MarketBelief,
    PlaybookPhase,
    PositionSnapshot,
    SequenceStepState,
)
from smc_trader.validation import FunnelTransition

from .helpers import engine_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_brain_calibration_stream_schema_matches_record_contract() -> None:
    assert tuple(BRAIN_CALIBRATION_FIELD_TYPES) == tuple(
        field.name for field in fields(BrainCalibrationRecord)
    )


def test_legacy_2022_v5_lineage_is_retained_but_incompatible() -> None:
    path = (
        ROOT
        / "outputs/development/scene_graph_v1_lineage"
        / "retained_incompatible_2022_v5.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    legacy = payload["legacy_artifact"]
    legacy_root = ROOT / legacy["path"]

    assert payload["status"] == "retained_incompatible"
    assert legacy["completion_status"] == "complete"
    assert legacy["decision_rows"] == 354_135
    assert legacy["brain_calibration_rows"] == 30_045
    assert _sha256(legacy_root / "COMPLETED.json") == legacy[
        "completed_sha256"
    ]
    assert payload["migration_policy"] == {
        "legacy_directory_files_modified": False,
        "legacy_rows_rewritten": False,
        "legacy_checkpoint_resumable": False,
        "new_capture_required": True,
        "failure_mode": "fail_closed",
    }
    assert "new Brain calibration fitting" in payload["forbidden_uses"]


def test_funnel_stream_schema_matches_record_contract() -> None:
    assert tuple(FUNNEL_FIELD_TYPES) == tuple(
        field.name for field in fields(FunnelTransition)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registered_fixture(tmp_path: Path) -> tuple[Path, Path]:
    index = pd.date_range(
        "2023-01-03 18:00",
        periods=360,
        freq="min",
        tz="America/New_York",
        name="ts",
    )
    base = pd.Series(range(len(index)), index=index, dtype=float)
    raw_close = (
        100.0
        + 0.01 * base
        + 0.20 * ((base % 11) - 5) / 5.0
    )
    close = (raw_close / 0.25).round() * 0.25
    frame = pd.DataFrame(
        {
            "open": close.shift(1, fill_value=close.iloc[0]),
            "close": close,
            "volume": 10.0 + (base % 7),
            "symbol": "NQH3",
            "instrument_id": 1,
        },
        index=index,
    )
    frame["high"] = frame[["open", "close"]].max(axis=1) + 0.50
    frame["low"] = frame[["open", "close"]].min(axis=1) - 0.50
    source = tmp_path / "causal_previous_session_front.parquet"
    frame.to_parquet(source)

    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "protocol_version": "2.3.0-test-stream-resume",
                "causal_front_sha256": _sha256(source),
                "belief_calibration_valid_from": (
                    "2023-01-01T00:00:00-05:00"
                ),
                "ohlcv_windows": {
                    "legacy_revealed_development": {
                        "start": "2017-01-01T00:00:00-05:00",
                        "end_exclusive": "2022-01-01T00:00:00-05:00",
                        "purpose": "test",
                    },
                    "belief_calibration": {
                        "start": "2022-01-01T00:00:00-05:00",
                        "end_exclusive": "2023-01-01T00:00:00-05:00",
                        "purpose": "test",
                    },
                    "action_clock_development": {
                        "start": "2023-01-01T00:00:00-05:00",
                        "end_exclusive": "2024-01-01T00:00:00-05:00",
                        "purpose": "test",
                    },
                    "rolling_validation": {
                        "start": "2024-01-01T00:00:00-05:00",
                        "end_exclusive": "2026-04-01T00:00:00-04:00",
                        "purpose": "test",
                    },
                    "sealed_holdout": {
                        "start": "2026-04-01T00:00:00-04:00",
                        "end_exclusive": "2026-08-01T00:00:00-04:00",
                        "purpose": "test",
                    },
                },
                "mbo_windows": {
                    "revealed_execution_development": {
                        "start": "2024-06-02T00:00:00Z",
                        "end_exclusive": "2024-08-02T00:00:00Z",
                        "purpose": "test",
                    },
                    "sealed_holdout": {
                        "start": "2024-08-02T00:00:00Z",
                        "end_exclusive": "2024-12-02T00:00:00Z",
                        "purpose": "test",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "model.json"
    config.write_text(
        json.dumps(
            {
                "version": "2.3.0-stream-resume-test",
                "calibration_artifact_namespace": "stream_resume_test",
                "playbook_registry": "configs/playbooks_v2.json",
                "calibration_artifact": None,
                "action_equivalence_protocol": (
                    "configs/action_equivalence_v2_2.json"
                ),
                "action_clock_value_protocol": (
                    "configs/action_clock_value_protocol_v2_3.json"
                ),
                "validation_protocol": str(validation),
                "action_clock_value_artifact": None,
                "tick_size": 0.25,
                "point_value": 20.0,
                "observer": {
                    "atr_period": 14,
                    "swing_k": 2,
                    "external_liquidity_lookback": 80,
                    "memory_events": 64,
                    "minimum_bars": {
                        "4H": 2,
                        "1H": 2,
                        "5m": 2,
                        "1m": 2,
                    },
                },
                "brain": {
                    "prior_decay": 0.92,
                    "forming_probability": 0.42,
                    "armed_probability": 0.58,
                    "executable_probability": 0.5,
                    "weakening_probability": 0.46,
                    "invalidation_probability": 0.28,
                },
                "decision": {
                    "minimum_utility_advantage": 0.12,
                    "uncertainty_penalty": 0.35,
                    "deadline_penalty_minutes": 20,
                    "maximum_reward_R": 3.0,
                },
                "risk": {
                    "maximum_trade_risk_fraction": 0.01,
                    "maximum_total_risk_fraction": 0.02,
                    "maximum_spread_ticks": 4,
                    "maximum_cost_R": 0.2,
                    "minimum_fillability": 0.45,
                    "minimum_minutes_to_deadline": 5,
                    "minimum_target_R": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return source, config


def _command(
    source: Path,
    config: Path,
    output: Path,
    *,
    resume: bool = False,
    stop_after: int = 0,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_action_clock_calibration.py",
        "--source",
        str(source),
        "--output",
        str(output),
        "--config",
        str(config),
        "--validation-protocol",
        str(config.parent / "validation.json"),
        "--start",
        "2023-01-03T20:00:00-05:00",
        "--end",
        "2023-01-03T23:00:00-05:00",
        "--warmup-days",
        "1",
        "--shard-rows",
        "31",
        "--checkpoint-bars",
        "50",
    ]
    if resume:
        command.append("--resume")
    if stop_after:
        command.extend(
            ["--diagnostic-stop-after-bars", str(stop_after)]
        )
    return command


def _continuous_command(
    source: Path,
    config: Path,
    output: Path,
    *,
    resume: bool = False,
    stop_after: int = 0,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_continuous_replay.py",
        "--source",
        str(source),
        "--output",
        str(output),
        "--config",
        str(config),
        "--validation-protocol",
        str(config.parent / "validation.json"),
        "--start",
        "2023-01-03T20:00:00-05:00",
        "--end",
        "2023-01-03T23:00:00-05:00",
        "--warmup-days",
        "1",
        "--shard-rows",
        "31",
        "--checkpoint-bars",
        "50",
    ]
    if resume:
        command.append("--resume")
    if stop_after:
        command.extend(["--diagnostic-stop-after-bars", str(stop_after)])
    return command


def _write_managed_fixture(tmp_path: Path, version: str) -> tuple[Path, Path]:
    source, config = _write_registered_fixture(tmp_path)
    validation = tmp_path / "validation.json"
    protocol = json.loads(validation.read_text(encoding="utf-8"))
    protocol["protocol_version"] = f"{version}-test-managed-resume"
    protocol["ohlcv_windows"] = {
        "managed_policy_calibration": {
            "start": "2023-01-03T20:00:00-05:00",
            "end_exclusive": "2023-01-03T23:00:00-05:00",
            "purpose": "bounded synthetic recovery test",
        }
    }
    validation.write_text(json.dumps(protocol), encoding="utf-8")
    model = json.loads(config.read_text(encoding="utf-8"))
    model.update(
        version=version,
        managed_policy_artifact=None,
        managed_net_value_artifact=None,
    )
    config.write_text(json.dumps(model), encoding="utf-8")
    return source, config


def _managed_command(
    script: str,
    source: Path,
    config: Path,
    output: Path,
    *extra: str,
) -> list[str]:
    return [
        sys.executable,
        f"scripts/{script}",
        "--source", str(source),
        "--output", str(output),
        "--config", str(config),
        "--validation-protocol", str(config.parent / "validation.json"),
        "--start", "2023-01-03T20:00:00-05:00",
        "--end", "2023-01-03T23:00:00-05:00",
        "--warmup-days", "1",
        "--decision-shard-rows", "31",
        "--checkpoint-bars", "50",
        *extra,
    ]


def test_legacy_annual_entry_fails_before_source_or_output_access(tmp_path) -> None:
    output = tmp_path / "must_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_continuous_replay.py",
            "--source", str(tmp_path / "missing.parquet"),
            "--output", str(output),
            "--start", "2023-01-03T20:00:00-05:00",
            "--end", "2023-01-03T23:00:00-05:00",
            "--gross-policy-calibration",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "--gross-policy-calibration is retired" in result.stderr
    assert not output.exists()


def test_streamed_history_rejects_full_minute_traces_before_source_access(
    tmp_path,
) -> None:
    output = tmp_path / "must_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_continuous_replay.py",
            "--source", str(tmp_path / "missing.parquet"),
            "--output", str(output),
            "--start", "2023-01-03T20:00:00-05:00",
            "--end", "2023-01-03T23:00:00-05:00",
            "--include-decision-traces",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "no longer emits full minute traces" in result.stderr
    assert not output.exists()


def test_decision_row_freezes_stop_and_all_target_provenance() -> None:
    row = _row(engine_snapshot())
    assert row["trace_schema_version"] is None
    assert row["decision_trace"] is None
    assert row["h4_regime"] == "h4_up"
    assert row["position_open"] is False
    assert row["position_thesis_hash"] is None
    assert row["position_setup_id"] is None
    assert row["position_playbook"] is None
    assert row["position_direction"] is None
    invalidation = json.loads(row["invalidation_source"])
    assert invalidation == {
        "confirmed_at": "2025-01-06T08:00:00-05:00",
        "formed_at": "2025-01-06T06:00:00-05:00",
        "lifecycle": "visible",
        "observed_at": "2025-01-06T08:00:00-05:00",
        "price": 98.0,
        "rationale": "confirmed swing low",
        "side": "below",
        "source_id": "below-level",
        "source_kind": "legacy_liquidity",
        "timeframe": "1H",
    }
    targets = json.loads(row["targets"])
    assert targets == [
        {
            "confirmed_at": "2025-01-06T08:00:00-05:00",
            "formed_at": "2025-01-06T06:00:00-05:00",
            "lifecycle": "visible",
            "price": 103.0,
            "side": "above",
            "source_id": "above-level",
            "source_kind": "legacy_liquidity",
            "timeframe": "1H",
        }
    ]
    traced = _row(
        engine_snapshot(),
        include_decision_trace=True,
    )
    assert traced["trace_schema_version"] == 1
    assert json.loads(traced["decision_trace"])[
        "future_path_included"
    ] is False


def test_light_decision_row_retains_terminal_top_raw_diagnostics() -> None:
    snapshot = engine_snapshot()
    hypothesis = next(iter(snapshot.belief.hypotheses.values()))
    terminal = replace(
        hypothesis,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=snapshot.observation.asof,
        terminal_at=snapshot.observation.asof,
        terminal_reason="delivery_not_ready",
        thesis_strength=0.67,
        sequence_progress=1.0,
        location_quality=0.81,
        entry_readiness=0.66,
        delivery_quality=0.0,
        evidence_group_scores={
            "structure": 1.0,
            "displacement": 1.0,
            "location": 0.81,
            "liquidity": 1.0,
            "trigger": 0.66,
            "execution": 0.0,
        },
        raw_quality_dimensions={
            "thesis_strength": 0.67,
            "sequence_progress": 1.0,
            "location_quality": 0.81,
            "entry_readiness": 0.66,
            "delivery_quality": 0.0,
            "uncertainty": 0.24,
        },
        hard_gate_results={
            "h4_structure_and_draw": True,
            "first_pullback_to_frozen_zone": False,
            "typed_entry_trigger": False,
        },
    )
    terminal_snapshot = replace(
        snapshot,
        belief=MarketBelief(
            snapshot.observation.asof,
            {terminal.key: terminal},
        ),
    )

    row = _row(terminal_snapshot)

    assert row["top_phase"] == "invalidated"
    assert row["top_raw_location_quality"] == pytest.approx(0.81)
    assert row["top_raw_entry_readiness"] == pytest.approx(0.66)
    assert row["top_raw_delivery_quality"] == pytest.approx(0.0)
    assert row["top_raw_uncertainty"] == pytest.approx(0.24)
    assert row["top_terminal_reason"] == "delivery_not_ready"
    assert json.loads(row["top_failed_hard_gate_ids"]) == [
        "first_pullback_to_frozen_zone",
        "typed_entry_trigger",
    ]
    assert tuple(row) == tuple(DECISION_FIELD_TYPES)


def test_light_decision_row_keeps_only_position_identity() -> None:
    snapshot = engine_snapshot()
    plan = snapshot.decision.plan
    assert plan is not None
    position = PositionSnapshot(
        thesis_hash="f" * 64,
        symbol=snapshot.observation.symbol,
        instrument_id=snapshot.observation.instrument_id,
        playbook=plan.playbook,
        direction=plan.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=snapshot.observation.asof - pd.Timedelta(minutes=1),
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=1,
        setup_id="setup-position",
        entry_location_id="location-position",
        entry_path_id="path-position",
    )
    row = _row(snapshot, belief_position_input=position)
    assert row["position_open"] is True
    assert row["position_thesis_hash"] == "f" * 64
    assert row["position_setup_id"] == "setup-position"
    assert row["position_playbook"] == plan.playbook.value
    assert row["position_direction"] == plan.direction.value
    assert row["decision_trace"] is None


def test_light_decision_row_does_not_report_terminal_lifecycle_as_open() -> None:
    snapshot = engine_snapshot()
    plan = snapshot.decision.plan
    assert plan is not None
    terminal = PositionSnapshot(
        thesis_hash="e" * 64,
        symbol=snapshot.observation.symbol,
        instrument_id=snapshot.observation.instrument_id,
        playbook=plan.playbook,
        direction=plan.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=snapshot.observation.asof - pd.Timedelta(minutes=2),
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=-1.0,
        elapsed_minutes=2,
        status="invalidated",
        setup_id="terminal-setup",
        entry_location_id="terminal-location",
        entry_path_id="terminal-path",
    )
    row = _row(
        snapshot,
        account_state=AccountState(equity=100_000.0),
        belief_position_input=terminal,
    )

    assert row["position_open"] is False
    assert row["position_thesis_hash"] is None
    assert row["position_setup_id"] is None
    assert row["position_playbook"] is None
    assert row["position_direction"] is None


def test_light_decision_row_counts_only_eligible_active_setups() -> None:
    snapshot = engine_snapshot()
    hypothesis = next(iter(snapshot.belief.hypotheses.values()))
    started_at = snapshot.observation.asof - pd.Timedelta(minutes=2)
    sequence = HypothesisSequenceState(
        protocol_version="test-active-setup-count",
        protocol_hash="b" * 64,
        setup_id="setup-active",
        steps=(
            SequenceStepState(
                "setup",
                True,
                1.0,
                started_at,
            ),
        ),
        started_at=started_at,
    )
    active = replace(
        hypothesis,
        phase=PlaybookPhase.EXECUTABLE,
        sequence=sequence,
    )
    active_snapshot = replace(
        snapshot,
        belief=MarketBelief(snapshot.observation.asof, {active.key: active}),
    )

    assert active.eligible is True
    assert _row(active_snapshot)["active_setup_count"] == 1

    terminal = replace(
        active,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=snapshot.observation.asof,
        plan=None,
    )
    terminal_snapshot = replace(
        snapshot,
        belief=MarketBelief(
            snapshot.observation.asof,
            {terminal.key: terminal},
        ),
    )

    assert terminal.eligible is False
    assert _row(terminal_snapshot)["active_setup_count"] == 0


def test_continuous_development_replay_is_bounded_resumable_and_deterministic(
    tmp_path,
) -> None:
    source, config = _write_registered_fixture(tmp_path)
    resumed_output = tmp_path / "continuous-resumed"
    interrupted = subprocess.run(
        _continuous_command(
            source,
            config,
            resumed_output,
            stop_after=137,
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert interrupted.returncode != 0
    progress = json.loads(
        (resumed_output / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "failed"
    assert progress["resume_supported"] is True
    assert 0.0 < progress["completed_percent"] < 100.0

    resumed = subprocess.run(
        _continuous_command(
            source,
            config,
            resumed_output,
            resume=True,
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr

    uninterrupted_output = tmp_path / "continuous-uninterrupted"
    uninterrupted = subprocess.run(
        _continuous_command(source, config, uninterrupted_output),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert uninterrupted.returncode == 0, uninterrupted.stderr

    resumed_completed = json.loads(
        (resumed_output / "COMPLETED.json").read_text(encoding="utf-8")
    )
    uninterrupted_completed = json.loads(
        (uninterrupted_output / "COMPLETED.json").read_text(encoding="utf-8")
    )
    assert (
        resumed_completed["stream_manifest_sha256"]
        == uninterrupted_completed["stream_manifest_sha256"]
    )
    resumed_summary = json.loads(
        (resumed_output / "summary.json").read_text(encoding="utf-8")
    )
    uninterrupted_summary = json.loads(
        (uninterrupted_output / "summary.json").read_text(encoding="utf-8")
    )
    assert resumed_summary["rolling_state_commitment"] == (
        uninterrupted_summary["rolling_state_commitment"]
    )
    assert resumed_summary["resume_count"] == 1
    assert uninterrupted_summary["resume_count"] == 0
    assert resumed_summary["full_snapshot_hash_per_minute"] is False
    assert resumed_summary["sequential_execution_evaluated"] is False
    assert resumed_summary["net_R"] is None
    assert resumed_summary["output_contract"] == "manifest_first_shards_v1"
    assert (
        resumed_completed["bindings"]["liquidity_protocol_sha256"]
        is None
    )
    assert json.loads(
        (resumed_output / "ai_primitive_proposals.json").read_text(
            encoding="utf-8"
        )
    ) == []

    decision_manifest = json.loads(
        (resumed_output / "decision_shards.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision_manifest["rows"] == resumed_summary["decision_rows"]
    assert decision_manifest["field_types"]["4H_ready"] == "bool"
    assert decision_manifest["field_types"]["1H_ready"] == "bool"
    assert (
        decision_manifest["field_types"]["invalidation_source_id"]
        == "large_string"
    )
    assert (
        decision_manifest["field_types"]["primary_target_id"]
        == "large_string"
    )
    assert decision_manifest["field_types"]["target_ids"] == "large_string"
    assert (
        decision_manifest["field_types"]["invalidation_source"]
        == "large_string"
    )
    assert decision_manifest["field_types"]["targets"] == "large_string"
    assert (
        decision_manifest["field_types"]["top_raw_location_quality"]
        == "float64"
    )
    assert (
        decision_manifest["field_types"]["top_raw_entry_readiness"]
        == "float64"
    )
    assert (
        decision_manifest["field_types"]["top_raw_delivery_quality"]
        == "float64"
    )
    assert (
        decision_manifest["field_types"]["top_raw_uncertainty"]
        == "float64"
    )
    assert (
        decision_manifest["field_types"]["top_terminal_reason"]
        == "large_string"
    )
    assert (
        decision_manifest["field_types"]["top_failed_hard_gate_ids"]
        == "large_string"
    )
    funnel_manifest = json.loads(
        (
            resumed_output
            / "funnel_transition_shards.manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert funnel_manifest["field_types"]["terminal_at"] == "timestamp_ny"
    assert (
        funnel_manifest["field_types"]["terminal_reason"]
        == "large_string"
    )
    assert (
        funnel_manifest["field_types"]["terminal_source_ids"]
        == "large_string"
    )
    path_manifest = json.loads(
        (resumed_output / "path_test_shards.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        path_manifest["field_types"]["invalidation_source_id"]
        == "large_string"
    )
    assert (
        path_manifest["field_types"]["target_source_id"]
        == "large_string"
    )
    assert max(resumed_summary["peak_buffer_rows"].values()) <= 31


@pytest.mark.parametrize(
    ("script", "version"),
    [
        ("run_managed_policy_calibration.py", "2.1.0"),
        ("run_managed_net_calibration.py", "2.2.0"),
    ],
)
def test_managed_runners_publish_durable_failure_and_resume(
    tmp_path,
    script: str,
    version: str,
) -> None:
    source, config = _write_managed_fixture(tmp_path, version)
    output = tmp_path / "managed"
    interrupted = subprocess.run(
        _managed_command(
            script, source, config, output,
            "--diagnostic-stop-after-bars", "137",
        ),
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert interrupted.returncode != 0
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["resume_supported"] is True
    assert progress["durable_checkpoint_only"] is True

    resumed = subprocess.run(
        _managed_command(script, source, config, output, "--resume"),
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    progress = json.loads((output / "progress.json").read_text())
    completed = output / "COMPLETED.json"
    assert progress["status"] == "complete"
    assert progress["source_rows"] == progress["source_rows_total"]
    assert completed.stat().st_mtime_ns >= (output / "progress.json").stat().st_mtime_ns


def test_arbitrary_checkpoint_resume_matches_uninterrupted_stream_manifests(
    tmp_path,
) -> None:
    source, config = _write_registered_fixture(tmp_path)
    resumed_output = tmp_path / "resumed"
    interrupted = subprocess.run(
        _command(
            source,
            config,
            resumed_output,
            stop_after=137,
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert interrupted.returncode != 0
    progress = json.loads(
        (resumed_output / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "failed"
    assert progress["resume_supported"] is True
    assert 0.0 < progress["complete_percent"] < 100.0

    checkpoint_manifest = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint_state = pickle.loads(
        (
            resumed_output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert any(checkpoint_state["buffers"].values())
    assert all(
        len(rows) < 31 for rows in checkpoint_state["buffers"].values()
    )

    resumed = subprocess.run(
        _command(source, config, resumed_output, resume=True),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (resumed_output / "COMPLETED.json").is_file()
    assert json.loads(
        (resumed_output / "progress.json").read_text(encoding="utf-8")
    )["status"] == "complete"
    assert (resumed_output / "COMPLETED.json").stat().st_mtime_ns >= (
        resumed_output / "progress.json"
    ).stat().st_mtime_ns

    uninterrupted_output = tmp_path / "uninterrupted"
    uninterrupted = subprocess.run(
        _command(source, config, uninterrupted_output),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert uninterrupted.returncode == 0, uninterrupted.stderr

    resumed_completion = json.loads(
        (resumed_output / "COMPLETED.json").read_text(encoding="utf-8")
    )
    uninterrupted_completion = json.loads(
        (uninterrupted_output / "COMPLETED.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        resumed_completion["stream_manifest_sha256"]
        == uninterrupted_completion["stream_manifest_sha256"]
    )
    assert (
        resumed_completion["rolling_state_commitment"]
        == uninterrupted_completion["rolling_state_commitment"]
    )
    resumed_summary = json.loads(
        (resumed_output / "summary.json").read_text(encoding="utf-8")
    )
    uninterrupted_summary = json.loads(
        (uninterrupted_output / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert resumed_summary["resume_count"] == 1
    assert uninterrupted_summary["resume_count"] == 0
    resumed_summary.pop("resume_count")
    uninterrupted_summary.pop("resume_count")
    assert resumed_summary == uninterrupted_summary
