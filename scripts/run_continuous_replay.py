#!/usr/bin/env python3
"""Run the causal model with bounded, resumable development outputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import (  # noqa: E402
    atomic_bytes,
    atomic_parquet,
    canonical_json,
    new_stream_state,
    sha256_file,
    verify_stream_shards,
    write_stream_manifest,
    write_stream_shards_bounded,
)
from smc_trader.calibration_replay import (  # noqa: E402
    CalibrationSequentialReplay,
    HASH_MODE,
    ReplayCheckpointStore,
    iter_after_source_checkpoint,
)
from smc_trader.brain_calibration import (  # noqa: E402
    BrainCalibrationRecord,
    BrainCalibrationRecorder,
    RECORDER_VERSION as BRAIN_CALIBRATION_RECORDER_VERSION,
)
from smc_trader.engine import (  # noqa: E402
    ContinuousSMCEngine,
    _configured_primitive_protocol_hashes,
)
from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.decision_trace import (  # noqa: E402
    TRACE_SCHEMA_VERSION,
    build_decision_trace,
)
from smc_trader.ai_review import (  # noqa: E402
    CausalPrimitiveRegistry,
)
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import (  # noqa: E402
    AccountState,
    Action,
    Timeframe,
    to_primitive,
)
from smc_trader.scene_graph import brain_input_contract_hash  # noqa: E402
from smc_trader.mbo import (  # noqa: E402
    MinuteExecutionRealityStore,
    assert_mbo_source_allowed,
)
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.policy_value import (  # noqa: E402
    build_v2_1_engine,
    managed_policy_code_fingerprint,
    path_structural_score,
)
from smc_trader.simulation import SequentialReplay  # noqa: E402
from smc_trader.visualization import (  # noqa: E402
    DecisionVisualizer,
    SealedVisualAudit,
)
from smc_trader.validation import (  # noqa: E402
    FrozenPathTestRecorder,
    FunnelTransition,
    PathTestResult,
    PlaybookFunnelRecorder,
    load_validation_protocol,
    records_frame,
)


TRADE_COLUMNS = [
    "thesis_hash",
    "playbook",
    "direction",
    "setup_id",
    "entry_location_id",
    "entry_path_id",
    "decision_time",
    "opened_at",
    "closed_at",
    "entry_price",
    "original_invalidation",
    "final_stop",
    "target",
    "exit_price",
    "exit_reason",
    "gross_R",
    "cost_R",
    "net_R",
    "ambiguous_same_bar",
]

ENTRY_ATTEMPT_COLUMNS = [
    "thesis_hash",
    "action_variant_hash",
    "decision_time",
    "playbook",
    "direction",
    "setup_id",
    "entry_location_id",
    "entry_path_id",
    "planned_entry",
    "original_invalidation",
    "invalidation_source_id",
    "invalidation_source",
    "primary_target",
    "primary_target_id",
    "targets",
    "deadline",
    "decision_probability",
    "decision_raw_probability",
    "decision_phase",
    "best_variant_utility_R",
    "path_structural_score_R",
    "best_variant_components",
    "filled_at",
    "outcome",
]

TRADE_FIELD_TYPES = {
    "thesis_hash": "large_string",
    "playbook": "large_string",
    "direction": "large_string",
    "setup_id": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "decision_time": "timestamp_ny",
    "opened_at": "timestamp_ny",
    "closed_at": "timestamp_ny",
    "entry_price": "float64",
    "original_invalidation": "float64",
    "final_stop": "float64",
    "target": "float64",
    "exit_price": "float64",
    "exit_reason": "large_string",
    "gross_R": "float64",
    "cost_R": "float64",
    "net_R": "float64",
    "ambiguous_same_bar": "bool",
}

ENTRY_ATTEMPT_FIELD_TYPES = {
    "thesis_hash": "large_string",
    "action_variant_hash": "large_string",
    "decision_time": "timestamp_ny",
    "playbook": "large_string",
    "direction": "large_string",
    "setup_id": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "planned_entry": "float64",
    "original_invalidation": "float64",
    "invalidation_source_id": "large_string",
    "invalidation_source": "large_string",
    "primary_target": "float64",
    "primary_target_id": "large_string",
    "targets": "large_string",
    "deadline": "timestamp_ny",
    "decision_probability": "float64",
    "decision_raw_probability": "float64",
    "decision_phase": "large_string",
    "best_variant_utility_R": "float64",
    "path_structural_score_R": "float64",
    "best_variant_components": "large_string",
    "filled_at": "timestamp_ny",
    "outcome": "large_string",
}

STREAM_KEYS = {
    "decision_shards": "snapshot_hash",
    "funnel_transition_shards": "snapshot_hash",
    "path_test_shards": "setup_id",
    "brain_calibration_shards": "sample_id",
}

DECISION_FIELD_TYPES = {
    "asof": "timestamp_ny",
    "snapshot_hash": "large_string",
    "model_action": "large_string",
    "risk_action": "large_string",
    "utility_advantage_R": "float64",
    "best_variant_action": "large_string",
    "best_variant_utility_R": "float64",
    "best_variant_hypothesis_key": "large_string",
    "best_variant_components": "large_string",
    "path_structural_score_R": "float64",
    "managed_gross_R": "float64",
    "managed_policy_value_available": "float64",
    "managed_policy_value_in_support": "float64",
    "top_playbook": "large_string",
    "top_direction": "large_string",
    "top_probability": "float64",
    "top_raw_probability": "float64",
    "top_raw_location_quality": "float64",
    "top_raw_entry_readiness": "float64",
    "top_raw_delivery_quality": "float64",
    "top_raw_uncertainty": "float64",
    "calibration_version": "large_string",
    "calibration_hash": "large_string",
    "top_phase": "large_string",
    "top_terminal_reason": "large_string",
    "top_failed_hard_gate_ids": "large_string",
    "uncertainty": "float64",
    "top_setup_id": "large_string",
    "top_protocol_version": "large_string",
    "top_protocol_hash": "large_string",
    "top_sequence_completed_steps": "int64",
    "top_sequence_total_steps": "int64",
    "top_sequence_complete": "bool",
    "decision_hypothesis_key": "large_string",
    "decision_playbook": "large_string",
    "decision_direction": "large_string",
    "decision_probability": "float64",
    "decision_raw_probability": "float64",
    "decision_phase": "large_string",
    "decision_setup_id": "large_string",
    "decision_entry_location_id": "large_string",
    "decision_entry_path_id": "large_string",
    "scale_registry_id": "large_string",
    "scene_revision_id": "large_string",
    "focus_revision_id": "large_string",
    "focus_primary_timeframes": "large_string",
    "focus_supplemental_timeframes": "large_string",
    "focus_resolution_status": "large_string",
    "dominant_scene_hypothesis_id": "large_string",
    "competing_scene_hypothesis_ids": "large_string",
    "h4_regime": "large_string",
    "position_open": "bool",
    "position_thesis_hash": "large_string",
    "position_setup_id": "large_string",
    "position_playbook": "large_string",
    "position_direction": "large_string",
    "hypothesis_phase_counts": "large_string",
    "active_setup_count": "int64",
    "complete_sequence_count": "int64",
    "hypothesis_plan_count": "int64",
    "4H_ready": "bool",
    "1H_ready": "bool",
    "15m_ready": "bool",
    "5m_ready": "bool",
    "1m_ready": "bool",
    "observation_anomalies": "large_string",
    "group3_boundary_transitions": "large_string",
    "group4_state": "large_string",
    "group4_boundary_transitions": "large_string",
    "planned_entry": "float64",
    "invalidation": "float64",
    "invalidation_source_id": "large_string",
    "invalidation_source": "large_string",
    "primary_target": "float64",
    "primary_target_id": "large_string",
    "liquidity_route_id": "large_string",
    "context_draw_id": "large_string",
    "intermediate_liquidity_ids": "large_string",
    "primary_deliverable_target_id": "large_string",
    "terminal_draw_id": "large_string",
    "path_blocker_ids": "large_string",
    "source_path_ids": "large_string",
    "target_ids": "large_string",
    "targets": "large_string",
    "primary_target_R": "float64",
    "spread_points": "float64",
    "cost_points": "float64",
    "fillability": "float64",
    "execution_source": "large_string",
    "book_bid": "float64",
    "book_ask": "float64",
    "book_bid_size": "float64",
    "book_ask_size": "float64",
    "book_depth_imbalance": "float64",
    "vetoes": "large_string",
    "reasons": "large_string",
    "trace_schema_version": "int64",
    "decision_trace": "large_string",
}

FUNNEL_FIELD_TYPES = {
    "setup_id": "large_string",
    "playbook": "large_string",
    "direction": "large_string",
    "phase": "large_string",
    "terminal_at": "timestamp_ny",
    "terminal_reason": "large_string",
    "terminal_source_ids": "large_string",
    "observed_at": "timestamp_ny",
    "probability": "float64",
    "uncertainty": "float64",
    "completed_steps": "int64",
    "total_steps": "int64",
    "protocol_version": "large_string",
    "protocol_hash": "large_string",
    "snapshot_hash": "large_string",
}

PATH_TEST_FIELD_TYPES = {
    "setup_id": "large_string",
    "hypothesis_key": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "playbook": "large_string",
    "direction": "large_string",
    "setup_started_at": "timestamp_ny",
    "sequence_completed_at": "timestamp_ny",
    "decision_time": "timestamp_ny",
    "resolved_at": "timestamp_ny",
    "outcome": "large_string",
    "success": "bool",
    "entry": "float64",
    "invalidation": "float64",
    "invalidation_source_id": "large_string",
    "target": "float64",
    "target_source_id": "large_string",
    "deadline": "timestamp_ny",
    "probability": "float64",
    "raw_probability": "float64",
    "uncertainty": "float64",
    "phase": "large_string",
    "calibration_version": "large_string",
    "calibration_hash": "large_string",
    "mfe_R": "float64",
    "mae_R": "float64",
    "elapsed_minutes": "int64",
    "formation_minutes": "int64",
    "entry_touched": "bool",
    "entry_touched_at": "timestamp_ny",
    "time_to_entry_minutes": "int64",
    "ambiguous_same_bar": "bool",
    "decision_hash": "large_string",
    "protocol_version": "large_string",
    "protocol_hash": "large_string",
    "config_hash": "large_string",
    "code_hash": "large_string",
}

BRAIN_CALIBRATION_FIELD_TYPES = {
    "sample_id": "large_string",
    "hypothesis_key": "large_string",
    "scene_hypothesis_id": "large_string",
    "competing_scene_hypothesis_ids": "large_string",
    "context_root_ids": "large_string",
    "scene_revision_id": "large_string",
    "playbook": "large_string",
    "direction": "large_string",
    "dimension": "large_string",
    "setup_id": "large_string",
    "episode_id": "large_string",
    "context_id": "large_string",
    "evidence_revision_id": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "sampled_at": "timestamp_ny",
    "resolved_at": "timestamp_ny",
    "raw_value": "float64",
    "outcome_value": "float64",
    "resolution": "large_string",
    "censored": "bool",
    "fit_eligible": "bool",
    "phase": "large_string",
    "origin_price": "float64",
    "trigger_bar_high": "float64",
    "trigger_bar_low": "float64",
    "invalidation_price": "float64",
    "invalidation_source_id": "large_string",
    "draw_price": "float64",
    "draw_id": "large_string",
    "liquidity_route_id": "large_string",
    "context_draw_id": "large_string",
    "intermediate_liquidity_ids": "large_string",
    "primary_deliverable_target_id": "large_string",
    "terminal_draw_id": "large_string",
    "path_blocker_ids": "large_string",
    "source_path_ids": "large_string",
    "dfp_structure_id": "large_string",
    "dfp_structure_confirmed_at": "timestamp_ny",
    "dfp_h1_bos_id": "large_string",
    "deadline": "timestamp_ny",
    "symbol": "large_string",
    "instrument_id": "int64",
    "protocol_version": "large_string",
    "protocol_hash": "large_string",
    "registry_hash": "large_string",
    "model_code_hash": "large_string",
    "config_hash": "large_string",
    "primitive_protocol_hashes": "large_string",
    "brain_input_contract_hash": "large_string",
}

STREAM_FIELD_TYPES = {
    "decision_shards": DECISION_FIELD_TYPES,
    "funnel_transition_shards": FUNNEL_FIELD_TYPES,
    "path_test_shards": PATH_TEST_FIELD_TYPES,
    "brain_calibration_shards": BRAIN_CALIBRATION_FIELD_TYPES,
}


def _deadline(timestamp: pd.Timestamp) -> pd.Timestamp:
    local = timestamp.tz_convert("America/New_York")
    day = local.tz_localize(None).normalize()
    if local.hour >= 18:
        day += pd.Timedelta(days=1)
    deadline = (day + pd.Timedelta(hours=17)).tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward"
    )
    return deadline


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _optional_protocol_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = ROOT / source
    return sha256_file(source)


def _enter_structural_score(utility) -> float:
    if utility.action is not Action.ENTER:
        raise ValueError("enter structural score requires an enter variant")
    if "path_structural_score_R" in utility.components:
        return float(utility.components["path_structural_score_R"])
    return path_structural_score(utility)


def _source_provenance(observation, source_id: str) -> dict[str, Any] | None:
    location = next(
        (
            value
            for value in observation.entry_locations
            if value.location_id == source_id
        ),
        None,
    )
    if location is not None:
        source_zone = next(
            (
                value
                for value in (
                    *observation.frame(Timeframe.M5).fair_value_gaps,
                    *observation.frame(Timeframe.M5).order_blocks,
                )
                if getattr(
                    value,
                    "fvg_id",
                    getattr(value, "order_block_id", None),
                )
                == location.source_zone_id
            ),
            None,
        )
        return {
            "source_kind": "entry_location",
            "timeframe": Timeframe.M5.value,
            "formed_at": location.formed_at.isoformat(),
            "confirmed_at": location.formed_at.isoformat(),
            "lifecycle": location.lifecycle.value,
            "source_zone_kind": location.source_zone_kind,
            "source_zone_id": location.source_zone_id,
            "source_zone_lifecycle": (
                None
                if source_zone is None
                else source_zone.lifecycle.value
            ),
            "source_displacement_id": location.source_displacement_id,
            "source_bos_id": location.source_bos_id,
            "lower_bound": float(location.lower_bound),
            "upper_bound": float(location.upper_bound),
            "failure_boundary": float(location.failure_boundary),
        }
    if observation.liquidity_inventory_authoritative:
        item = next(
            (
                value
                for value in observation.liquidity_inventory
                if value.item_id == source_id
            ),
            None,
        )
        if item is not None:
            return {
                "source_kind": item.kind,
                "timeframe": item.timeframe.value,
                "formed_at": item.formed_at.isoformat(),
                "confirmed_at": item.confirmed_at.isoformat(),
                "lifecycle": item.lifecycle.value,
            }
    event = next(
        (
            value
            for value in observation.recent_events
            if value.event_id == source_id
        ),
        None,
    )
    if event is not None:
        return {
            "source_kind": event.kind.value,
            "timeframe": event.timeframe.value,
            "formed_at": (
                None
                if event.formed_at is None
                else event.formed_at.isoformat()
            ),
            "confirmed_at": (
                None
                if event.confirmed_at is None
                else event.confirmed_at.isoformat()
            ),
            "lifecycle": event.lifecycle,
        }
    level = next(
        (
            value
            for timeframe in observation.active_timeframes
            for value in observation.frame(timeframe).liquidity
            if value.level_id == source_id
        ),
        None,
    )
    if level is None:
        return None
    return {
        "source_kind": "legacy_liquidity",
        "timeframe": level.timeframe.value,
        "formed_at": level.formed_at.isoformat(),
        "confirmed_at": level.confirmed_at.isoformat(),
        "lifecycle": "consumed" if level.swept else "visible",
    }


def _invalidation_provenance(observation, plan) -> str | None:
    if plan is None:
        return None
    invalidation = plan.invalidation
    payload: dict[str, Any] = {
        "source_id": invalidation.source_level_id,
        "price": float(invalidation.price),
        "side": invalidation.side,
        "observed_at": invalidation.observed_at.isoformat(),
        "rationale": invalidation.rationale,
    }
    source = _source_provenance(
        observation,
        invalidation.source_level_id,
    )
    if source is not None:
        payload.update(source)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _targets_provenance(observation, plan) -> str | None:
    if plan is None:
        return None
    output: list[dict[str, Any]] = []
    for target in plan.targets:
        payload: dict[str, Any] = {
            "source_id": target.level_id,
            "price": float(target.price),
            "side": target.side,
            "timeframe": target.timeframe.value,
            "formed_at": target.formed_at.isoformat(),
            "confirmed_at": target.confirmed_at.isoformat(),
        }
        source = _source_provenance(observation, target.level_id)
        if source is not None:
            payload.update(source)
        output.append(payload)
    return json.dumps(output, sort_keys=True, ensure_ascii=False)


def _load_mbo_execution(
    path: str | Path,
    *,
    validation,
    start: pd.Timestamp,
    end: pd.Timestamp,
    reveal_sealed_holdout: bool,
) -> tuple[MinuteExecutionRealityStore, dict]:
    source = assert_mbo_source_allowed(
        path,
        allow_sealed_holdout=reveal_sealed_holdout,
    )
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"MBO execution reality requires its materialization manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("format_version", 0)) != 1:
        raise RuntimeError("unsupported MBO execution manifest format")
    if manifest.get("validation_protocol_hash") != validation.fingerprint:
        raise RuntimeError("MBO execution manifest uses a different validation protocol")
    actual_hash = _sha256_file(source)
    if manifest.get("output_sha256") != actual_hash:
        raise RuntimeError("MBO execution parquet does not match its manifest hash")
    requested_window = validation.classify_mbo(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
    )
    if manifest.get("validation_window_role") != requested_window.role:
        raise RuntimeError("MBO execution manifest role does not match replay interval")
    materialized_start = pd.Timestamp(manifest.get("start"))
    materialized_end = pd.Timestamp(manifest.get("end_exclusive"))
    if (
        materialized_start.tzinfo is None
        or materialized_end.tzinfo is None
        or materialized_start > start.tz_convert("UTC")
        or materialized_end < end.tz_convert("UTC")
    ):
        raise RuntimeError("MBO execution file does not cover the replay interval")
    is_sealed = bool(manifest.get("sealed_holdout_read")) or (
        requested_window.role == "sealed_holdout"
    )
    if is_sealed and not reveal_sealed_holdout:
        raise RuntimeError(
            "sealed MBO holdout requires --reveal-sealed-mbo-holdout"
        )
    return (
        MinuteExecutionRealityStore.from_parquet(
            source,
            allow_sealed_holdout=reveal_sealed_holdout,
        ),
        manifest,
    )


def _row(
    snapshot,
    previous_snapshot=None,
    *,
    source_bar=None,
    account_state=None,
    belief_position_input=None,
    include_decision_trace: bool = False,
) -> dict:
    trace = (
        build_decision_trace(
            snapshot,
            previous_snapshot,
            source_bar=source_bar,
            account_state=account_state,
            belief_position_input=belief_position_input,
        )
        if include_decision_trace
        else None
    )
    ranked = snapshot.belief.ranked()
    top = ranked[0] if ranked else None
    decision_hypothesis = (
        None
        if snapshot.decision.best_hypothesis_key is None
        else snapshot.belief.hypotheses.get(
            snapshot.decision.best_hypothesis_key
        )
    )
    decision_sequence = (
        None if decision_hypothesis is None else decision_hypothesis.sequence
    )
    summary_hypothesis = decision_hypothesis or top
    focus = snapshot.belief.focus_state
    plan = snapshot.decision.plan
    liquidity_route = (
        None
        if summary_hypothesis is None
        else summary_hypothesis.liquidity_route
    )
    if liquidity_route is None and plan is not None:
        liquidity_route = plan.liquidity_route
    best_variant = (
        snapshot.decision.utilities[0]
        if snapshot.decision.utilities
        else None
    )
    best_components = (
        {} if best_variant is None else dict(best_variant.components)
    )
    if "path_structural_score_R" in best_components:
        best_structural_score = float(
            best_components["path_structural_score_R"]
        )
    elif best_variant is not None and best_variant.action is Action.ENTER:
        best_structural_score = _enter_structural_score(best_variant)
    else:
        best_structural_score = None
    sequence = None if top is None else top.sequence
    top_raw_dimensions = (
        {} if top is None else dict(top.raw_quality_dimensions)
    )
    top_failed_hard_gate_ids = (
        None
        if top is None
        else json.dumps(
            sorted(
                gate_id
                for gate_id, passed in top.hard_gate_results.items()
                if not passed
            ),
            separators=(",", ":"),
        )
    )
    h4 = snapshot.observation.frame(Timeframe.H4)
    if not h4.ready:
        h4_regime = "h4_unready"
    else:
        h4_direction = float(h4.metrics.get("structure_direction", 0.0))
        if h4_direction > 0.0:
            h4_regime = "h4_up"
        elif h4_direction < 0.0:
            h4_regime = "h4_down"
        else:
            h4_regime = "h4_flat"
    # ``belief_position_input`` may be a one-bar terminal lifecycle record
    # after the portfolio has already closed.  Light replay fields describe
    # actual open exposure, so prefer the account snapshot and never label a
    # terminal acknowledgement as an open position.
    position = (
        getattr(account_state, "position", None)
        if account_state is not None
        else belief_position_input
    )
    if position is not None and getattr(position, "status", "open") != "open":
        position = None
    phase_counts: dict[str, int] = {}
    for hypothesis in snapshot.belief.hypotheses.values():
        phase_counts[hypothesis.phase.value] = (
            phase_counts.get(hypothesis.phase.value, 0) + 1
        )
    return {
        "asof": snapshot.observation.asof,
        "snapshot_hash": snapshot.snapshot_hash,
        "model_action": snapshot.decision.selected_action.value,
        "risk_action": snapshot.risk.final_action.value,
        "utility_advantage_R": snapshot.decision.advantage,
        "best_variant_action": (
            None if best_variant is None else best_variant.action.value
        ),
        "best_variant_utility_R": (
            None if best_variant is None else best_variant.utility
        ),
        "best_variant_hypothesis_key": (
            None if best_variant is None else best_variant.hypothesis_key
        ),
        "best_variant_components": json.dumps(
            best_components,
            sort_keys=True,
        ),
        "path_structural_score_R": best_structural_score,
        "managed_gross_R": best_components.get("managed_gross_R"),
        "managed_policy_value_available": (
            best_components.get("managed_value_available")
        ),
        "managed_policy_value_in_support": (
            best_components.get("managed_value_in_support")
        ),
        "top_playbook": None if top is None else top.playbook.value,
        "top_direction": None if top is None else top.direction.value,
        "top_probability": None if top is None else top.probability,
        "top_raw_probability": (
            None if top is None else top.raw_probability
        ),
        "top_raw_location_quality": top_raw_dimensions.get(
            "location_quality"
        ),
        "top_raw_entry_readiness": top_raw_dimensions.get(
            "entry_readiness"
        ),
        "top_raw_delivery_quality": top_raw_dimensions.get(
            "delivery_quality"
        ),
        "top_raw_uncertainty": top_raw_dimensions.get("uncertainty"),
        "calibration_version": (
            None if top is None else top.calibration_version
        ),
        "calibration_hash": None if top is None else top.calibration_hash,
        "top_phase": None if top is None else top.phase.value,
        "top_terminal_reason": (
            None if top is None else top.terminal_reason
        ),
        "top_failed_hard_gate_ids": top_failed_hard_gate_ids,
        "uncertainty": None if top is None else top.uncertainty,
        "top_setup_id": None if sequence is None else sequence.setup_id,
        "top_protocol_version": (
            None if sequence is None else sequence.protocol_version
        ),
        "top_protocol_hash": None if sequence is None else sequence.protocol_hash,
        "top_sequence_completed_steps": (
            None if sequence is None else sequence.completed_steps
        ),
        "top_sequence_total_steps": (
            None if sequence is None else len(sequence.steps)
        ),
        "top_sequence_complete": None if sequence is None else sequence.complete,
        "decision_hypothesis_key": snapshot.decision.best_hypothesis_key,
        "decision_playbook": (
            None
            if decision_hypothesis is None
            else decision_hypothesis.playbook.value
        ),
        "decision_direction": (
            None
            if decision_hypothesis is None
            else decision_hypothesis.direction.value
        ),
        "decision_probability": (
            None if decision_hypothesis is None else decision_hypothesis.probability
        ),
        "decision_raw_probability": (
            None
            if decision_hypothesis is None
            else decision_hypothesis.raw_probability
        ),
        "decision_phase": (
            None
            if decision_hypothesis is None
            else decision_hypothesis.phase.value
        ),
        "decision_setup_id": (
            None if decision_sequence is None else decision_sequence.setup_id
        ),
        "decision_entry_location_id": (
            None if plan is None else plan.entry_location_id
        ),
        "decision_entry_path_id": (
            None if plan is None else plan.entry_path_id
        ),
        "scale_registry_id": snapshot.observation.scale_registry_id,
        "scene_revision_id": (
            snapshot.belief.scene_revision_id
            or snapshot.observation.scene_revision_id
        ),
        "focus_revision_id": (
            None if focus is None else focus.focus_revision_id
        ),
        "focus_primary_timeframes": (
            None
            if focus is None
            else json.dumps(
                list(focus.primary_timeframes),
                ensure_ascii=False,
            )
        ),
        "focus_supplemental_timeframes": (
            None
            if focus is None
            else json.dumps(
                list(focus.supplemental_timeframes),
                ensure_ascii=False,
            )
        ),
        "focus_resolution_status": (
            None if focus is None else focus.resolution_status.value
        ),
        "dominant_scene_hypothesis_id": (
            snapshot.belief.dominant_hypothesis_id
        ),
        "competing_scene_hypothesis_ids": json.dumps(
            list(snapshot.belief.competing_hypothesis_ids),
            ensure_ascii=False,
        ),
        "h4_regime": h4_regime,
        "position_open": position is not None,
        "position_thesis_hash": (
            None if position is None else position.thesis_hash
        ),
        "position_setup_id": (
            None if position is None else position.setup_id
        ),
        "position_playbook": (
            None if position is None else position.playbook.value
        ),
        "position_direction": (
            None if position is None else position.direction.value
        ),
        "hypothesis_phase_counts": json.dumps(
            phase_counts,
            sort_keys=True,
        ),
        "active_setup_count": sum(
            hypothesis.eligible
            and hypothesis.sequence is not None
            and hypothesis.sequence.setup_id is not None
            for hypothesis in snapshot.belief.hypotheses.values()
        ),
        "complete_sequence_count": sum(
            hypothesis.sequence is not None
            and hypothesis.sequence.complete
            for hypothesis in snapshot.belief.hypotheses.values()
        ),
        "hypothesis_plan_count": sum(
            hypothesis.plan is not None
            for hypothesis in snapshot.belief.hypotheses.values()
        ),
        "4H_ready": snapshot.observation.frame(Timeframe.H4).ready,
        "1H_ready": snapshot.observation.frame(Timeframe.H1).ready,
        "15m_ready": (
            snapshot.observation.frame(Timeframe.M15).ready
            if Timeframe.M15 in snapshot.observation.active_timeframes
            else None
        ),
        "5m_ready": snapshot.observation.frame(Timeframe.M5).ready,
        "1m_ready": snapshot.observation.frame(Timeframe.M1).ready,
        "observation_anomalies": json.dumps(
            list(snapshot.observation.anomalies),
            ensure_ascii=False,
        ),
        "group3_boundary_transitions": (
            None
            if not (
                snapshot.observation.group3_boundary_fvg_transitions
                or snapshot.observation
                .group3_boundary_order_block_transitions
            )
            else json.dumps(
                {
                    "fair_value_gaps": to_primitive(
                        snapshot.observation
                        .group3_boundary_fvg_transitions
                    ),
                    "order_blocks": to_primitive(
                        snapshot.observation
                        .group3_boundary_order_block_transitions
                    ),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        ),
        "group4_state": json.dumps(
            {
                "dealing_ranges": to_primitive(
                    tuple(
                        state
                        for state in snapshot.observation
                        .frame(Timeframe.H1).dealing_ranges
                        if state.lifecycle.value != "broken"
                    )
                    or snapshot.observation
                    .frame(Timeframe.H1).dealing_ranges[-1:]
                ),
                "manipulations": to_primitive(
                    tuple(
                        state
                        for state in snapshot.observation.manipulations
                        if state.lifecycle.value == "swept"
                    )
                    or snapshot.observation.manipulations[-1:]
                ),
                "range_boundary_inventory": to_primitive(
                    tuple(
                        item
                        for item in snapshot.observation.liquidity_inventory
                        if item.kind == "range_boundary"
                    )[-4:]
                ),
                "ambiguous_sweep_item_ids": list(
                    snapshot.observation
                    .group4_ambiguous_sweep_item_ids
                ),
                "atr_unready_sweep_item_ids": list(
                    snapshot.observation
                    .group4_atr_unready_sweep_item_ids
                ),
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
        "group4_boundary_transitions": (
            None
            if not (
                snapshot.observation.group4_boundary_range_transitions
                or snapshot.observation
                .group4_boundary_manipulation_transitions
            )
            else json.dumps(
                {
                    "dealing_ranges": to_primitive(
                        snapshot.observation
                        .group4_boundary_range_transitions
                    ),
                    "manipulations": to_primitive(
                        snapshot.observation
                        .group4_boundary_manipulation_transitions
                    ),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        ),
        "planned_entry": None if plan is None else plan.planned_entry,
        "invalidation": None if plan is None else plan.invalidation.price,
        "invalidation_source_id": (
            None if plan is None else plan.invalidation.source_level_id
        ),
        "invalidation_source": _invalidation_provenance(
            snapshot.observation,
            plan,
        ),
        "primary_target": None if plan is None else plan.targets[0].price,
        "primary_target_id": None if plan is None else plan.targets[0].level_id,
        "liquidity_route_id": (
            None if liquidity_route is None else liquidity_route.route_id
        ),
        "context_draw_id": (
            None if liquidity_route is None else liquidity_route.context_draw_id
        ),
        "intermediate_liquidity_ids": (
            None
            if liquidity_route is None
            else json.dumps(
                list(liquidity_route.intermediate_liquidity_ids),
                ensure_ascii=False,
            )
        ),
        "primary_deliverable_target_id": (
            None
            if liquidity_route is None
            else liquidity_route.primary_deliverable_target_id
        ),
        "terminal_draw_id": (
            None if liquidity_route is None else liquidity_route.terminal_draw_id
        ),
        "path_blocker_ids": (
            None
            if liquidity_route is None
            else json.dumps(
                list(liquidity_route.path_blocker_ids),
                ensure_ascii=False,
            )
        ),
        "source_path_ids": (
            None
            if liquidity_route is None
            else json.dumps(
                list(liquidity_route.source_path_ids),
                ensure_ascii=False,
            )
        ),
        "target_ids": (
            None
            if plan is None
            else json.dumps(
                [target.level_id for target in plan.targets],
                ensure_ascii=False,
            )
        ),
        "targets": _targets_provenance(snapshot.observation, plan),
        "primary_target_R": None if plan is None else plan.primary_target_R,
        "spread_points": snapshot.observation.execution.spread_points,
        "cost_points": snapshot.observation.execution.expected_round_trip_cost_points,
        "fillability": snapshot.observation.execution.fillability,
        "execution_source": snapshot.observation.execution.source,
        "book_bid": snapshot.observation.execution.bid,
        "book_ask": snapshot.observation.execution.ask,
        "book_bid_size": snapshot.observation.execution.bid_size,
        "book_ask_size": snapshot.observation.execution.ask_size,
        "book_depth_imbalance": snapshot.observation.execution.depth_imbalance,
        "vetoes": json.dumps([item.value for item in snapshot.risk.vetoes]),
        "reasons": json.dumps(list(snapshot.risk.reasons), ensure_ascii=False),
        "trace_schema_version": (
            TRACE_SCHEMA_VERSION if trace is not None else None
        ),
        "decision_trace": (
            None
            if trace is None
            else json.dumps(
                trace,
                sort_keys=True,
                ensure_ascii=False,
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="outputs/v2_replay")
    parser.add_argument(
        "--config",
        default="configs/model_v3_development.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2.json",
    )
    parser.add_argument("--warmup-days", type=int, default=45)
    parser.add_argument(
        "--spread-points",
        type=float,
        default=None,
        help=(
            "optional non-authoritative constant research spread; omitted means "
            "OHLCV-only execution is unavailable and entry is risk-vetoed"
        ),
    )
    parser.add_argument(
        "--slippage-points",
        type=float,
        default=None,
        help=(
            "optional constant research slippage; omitted means 0.0 in streamed "
            "development and preserves the legacy visual default in legacy mode"
        ),
    )
    parser.add_argument(
        "--mbo-execution",
        help="pre-materialized causal MBO minute execution-reality parquet",
    )
    parser.add_argument("--chart-every", type=int, default=0)
    parser.add_argument(
        "--audit-path-tests",
        type=int,
        default=0,
        help="seal and later reveal the first N frozen path tests as separate visual audits",
    )
    parser.add_argument(
        "--ai-review-directory",
        help=(
            "optional directory of identity-bound diagnostic review JSON; "
            "use templates emitted by the pre-reveal candidate workflow"
        ),
    )
    parser.add_argument(
        "--simulate-execution",
        action="store_true",
        help=(
            "explicit final-validation opt-in: execute approved actions and "
            "write trade/entry-attempt artifacts"
        ),
    )
    parser.add_argument("--shard-rows", type=int, default=25_000)
    parser.add_argument("--checkpoint-bars", type=int, default=25_000)
    parser.add_argument(
        "--include-decision-traces",
        action="store_true",
        help=(
            "legacy visual-mode compatibility only; streamed history "
            "rejects full minute traces and uses sampled audit replay"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostic-stop-after-bars",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--legacy-visual-replay",
        action="store_true",
        help=(
            "run the former non-resumable inline chart/future-reveal workflow; "
            "daily development should use the default streamed mode"
        ),
    )
    parser.add_argument(
        "--gross-policy-calibration",
        action="store_true",
        help=(
            "retired annual-calibration entry; use "
            "scripts/run_managed_policy_calibration.py"
        ),
    )
    parser.add_argument(
        "--acknowledge-research-roll-lineage",
        action="store_true",
        help="allow a legacy same-day-volume continuous-front file for research replay only",
    )
    parser.add_argument(
        "--reveal-sealed-holdout",
        action="store_true",
        help="one-way acknowledgement that this run reveals the frozen OHLCV holdout",
    )
    parser.add_argument(
        "--reveal-sealed-mbo-holdout",
        action="store_true",
        help="explicitly reveal a manifest-bound MBO execution holdout",
    )
    return parser.parse_args()


def _legacy_visual_main(args: argparse.Namespace) -> None:
    if args.resume:
        raise RuntimeError("--legacy-visual-replay does not support checkpoint resume")
    if args.ai_review_directory:
        raise ValueError(
            "identity-bound AI review requires the two-pass "
            "scripts/run_scenario_visual_audit.py workflow"
        )
    if args.gross_policy_calibration:
        raise RuntimeError(
            "--gross-policy-calibration is retired; use the bounded, resumable "
            "scripts/run_managed_policy_calibration.py runner"
        )
    legacy_spread_points = (
        0.25 if args.spread_points is None else float(args.spread_points)
    )
    legacy_slippage_points = (
        0.25 if args.slippage_points is None else float(args.slippage_points)
    )
    if legacy_spread_points < 0.0 or legacy_slippage_points < 0.0:
        raise ValueError("legacy research spread and slippage cannot be negative")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None:
        start = start.tz_localize("America/New_York")
    if end.tzinfo is None:
        end = end.tz_localize("America/New_York")
    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_ohlcv(start, end)
    if args.gross_policy_calibration:
        if window.role != "managed_policy_calibration":
            raise RuntimeError(
                "--gross-policy-calibration requires the registered "
                "managed_policy_calibration OHLCV window"
            )
        if start != window.start or end != window.end_exclusive:
            raise RuntimeError(
                "managed-policy calibration must consume the complete "
                "registered [start, end) window"
            )
        if args.mbo_execution:
            raise RuntimeError(
                "gross managed-policy calibration cannot consume MBO execution"
            )
        if not args.simulate_execution:
            raise RuntimeError(
                "gross managed-policy calibration requires --simulate-execution"
            )
    if window.role == "sealed_holdout" and not args.reveal_sealed_holdout:
        raise RuntimeError(
            "requested interval is the sealed OHLCV holdout; pass "
            "--reveal-sealed-holdout only after code/config/calibration hashes are frozen"
        )
    source_hash = _sha256_file(args.source)
    if (
        source_hash != validation.causal_front_sha256
        and not args.acknowledge_research_roll_lineage
    ):
        raise RuntimeError(
            "OHLCV source hash does not match the preregistered causal front"
        )
    load_start = start - pd.Timedelta(days=args.warmup_days)
    loaded = load_ohlcv(args.source, start=load_start, end=end)
    if not loaded.contract_selection_causal and not args.acknowledge_research_roll_lineage:
        raise RuntimeError(
            "source uses legacy ex-post same-day roll selection; pass "
            "--acknowledge-research-roll-lineage for research-only replay"
        )
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    config_source = Path(args.config)
    config_payload = json.loads(config_source.read_text(encoding="utf-8"))
    managed_artifact = config_payload.get("managed_policy_artifact")
    if args.gross_policy_calibration and managed_artifact:
        raise RuntimeError(
            "managed-policy calibration must use the frozen policy-base model, "
            "not a fitted managed-policy artifact"
        )
    if managed_artifact:
        engine = build_v2_1_engine(config_source)
        code_hash = managed_policy_code_fingerprint()
    else:
        engine = ContinuousSMCEngine.from_config(config_source)
        code_hash = model_code_fingerprint()
    config_hash = hashlib.sha256(config_source.read_bytes()).hexdigest()
    execution_manifest: dict = {}
    if args.mbo_execution:
        execution_store, execution_manifest = _load_mbo_execution(
            args.mbo_execution,
            validation=validation,
            start=start,
            end=end,
            reveal_sealed_holdout=args.reveal_sealed_mbo_holdout,
        )
    else:
        execution_store = None
    sequential = SequentialReplay(engine=engine) if args.simulate_execution else None
    visualizer = DecisionVisualizer()
    ai_registry = CausalPrimitiveRegistry()
    ai_proposals_by_decision: dict[str, tuple] = {}
    ai_review_files_used: list[str] = []

    def ai_proposals_for(snapshot):
        cached = ai_proposals_by_decision.get(snapshot.snapshot_hash)
        if cached is not None:
            return cached
        proposals = ()
        ai_proposals_by_decision[snapshot.snapshot_hash] = proposals
        return proposals

    funnel = PlaybookFunnelRecorder()
    path_tests = FrozenPathTestRecorder(
        config_hash=config_hash,
        code_hash=code_hash,
    )
    rows: list[dict] = []
    trades = []
    entry_approvals: dict[str, dict] = {}
    filled_entries: dict[str, pd.Timestamp] = {}
    artifacts = []
    visual_audits: dict[str, SealedVisualAudit] = {}
    audit_records: list[str] = []
    sealed_audit_count = 0
    decision_number = 0
    last_asof = start

    def reveal_new_results(results) -> None:
        for result in results:
            audit = visual_audits.pop(result.setup_id, None)
            if audit is None:
                continue
            _, record = audit.reveal(result)
            audit_records.append(str(record))

    for bar in iter_completed_bars(loaded.frame):
        if bar.end >= end:
            break
        previous_snapshot = engine.last_snapshot
        for audit in visual_audits.values():
            audit.on_bar(bar)
        result_count = len(path_tests.results)
        path_tests.on_bar(bar)
        reveal_new_results(path_tests.results[result_count:])
        if bar.end < start:
            execution = ExecutionRealityInput(
                spread_points=0.0,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                deadline=_deadline(bar.end),
                source="constant_warmup_no_execution_authority",
            )
        elif execution_store is not None:
            execution = execution_store.for_bar(
                bar,
                deadline=_deadline(bar.end),
            )
        elif args.gross_policy_calibration:
            execution = ExecutionRealityInput(
                spread_points=0.0,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                deadline=_deadline(bar.end),
                source="gross_policy_calibration_zero_cost",
            )
        else:
            execution = ExecutionRealityInput(
                spread_points=legacy_spread_points,
                expected_slippage_points=legacy_slippage_points,
                deadline=_deadline(bar.end),
                source="constant_cli_research_only",
            )
        if sequential is not None:
            step = sequential.on_bar(bar, execution=execution)
            snapshot = step.snapshot
            in_window_closed = [
                record
                for record in step.closed_trades
                if record.decision_time >= start
            ]
            trades.extend(in_window_closed)
            for record in step.closed_trades:
                filled_entries.setdefault(record.thesis_hash, record.opened_at)
            if step.position is not None:
                filled_entries.setdefault(
                    step.position.thesis_hash,
                    step.position.opened_at,
                )
        else:
            snapshot = engine.on_bar(
                bar,
                execution=execution,
                account=AccountState(equity=100_000.0),
            )
        if snapshot.observation.asof < start:
            continue
        last_asof = snapshot.observation.asof
        if (
            snapshot.risk.final_action is Action.ENTER
            and snapshot.risk.frozen_thesis is not None
        ):
            approval = _approval_row(snapshot)
            thesis_hash = approval["thesis_hash"]
            if thesis_hash in entry_approvals:
                raise AssertionError(
                    "risk-approved thesis hash was emitted more than once"
                )
            entry_approvals[thesis_hash] = approval
        funnel.observe(snapshot)
        open_before = {item.setup_id for item in path_tests.open_tests}
        path_tests.observe(snapshot)
        for frozen in path_tests.open_tests:
            if (
                frozen.setup_id in open_before
                or sealed_audit_count >= max(0, args.audit_path_tests)
            ):
                continue
            audit = SealedVisualAudit.seal(
                visualizer,
                snapshot,
                engine.histories(80),
                destination / "path_audits" / frozen.setup_id,
                ai_proposals=ai_proposals_for(snapshot),
                hypothesis_key=frozen.hypothesis_key,
                previous_snapshot=previous_snapshot,
                source_bar=bar,
                account_state=(
                    step.account_state
                    if sequential is not None
                    else AccountState(equity=100_000.0)
                ),
                belief_position_input=(
                    step.belief_position_input
                    if sequential is not None
                    else None
                ),
            )
            visual_audits[frozen.setup_id] = audit
            sealed_audit_count += 1
        rows.append(
            _row(
                snapshot,
                previous_snapshot,
                source_bar=bar,
                account_state=(
                    step.account_state
                    if sequential is not None
                    else AccountState(equity=100_000.0)
                ),
                belief_position_input=(
                    step.belief_position_input
                    if sequential is not None
                    else None
                ),
                include_decision_trace=args.include_decision_traces,
            )
        )
        decision_number += 1
        if args.chart_every > 0 and decision_number % args.chart_every == 0:
            artifact = visualizer.render_decision(
                snapshot,
                engine.histories(80),
                destination / "decisions" / f"{snapshot.snapshot_hash[:20]}.png",
                ai_proposals=ai_proposals_for(snapshot),
            )
            artifacts.append(artifact)

    decisions = pd.DataFrame(rows)
    if decisions.empty:
        raise ValueError("requested interval produced no completed decision bars")
    if not (
        (pd.to_datetime(decisions["asof"], utc=True) >= start.tz_convert("UTC"))
        & (pd.to_datetime(decisions["asof"], utc=True) < end.tz_convert("UTC"))
    ).all():
        raise AssertionError("decision clocks escaped the registered [start, end) interval")
    decisions.to_parquet(destination / "decisions.parquet", index=False)
    result_count = len(path_tests.results)
    path_tests.close_unresolved(last_asof)
    reveal_new_results(path_tests.results[result_count:])
    records_frame(funnel.rows, record_type=FunnelTransition).to_parquet(
        destination / "funnel_transitions.parquet",
        index=False,
    )
    records_frame(path_tests.results, record_type=PathTestResult).to_parquet(
        destination / "path_tests.parquet",
        index=False,
    )
    trade_rows = [to_primitive(record) for record in trades]
    if args.simulate_execution:
        pd.DataFrame(trade_rows, columns=TRADE_COLUMNS).to_parquet(
            destination / "trades.parquet",
            index=False,
        )
        entry_attempt_rows = []
        for thesis_hash, approval in entry_approvals.items():
            filled_at = filled_entries.get(thesis_hash)
            if filled_at is not None:
                outcome = "filled"
            elif approval["decision_time"] == last_asof:
                outcome = "pending_right_censored"
            else:
                outcome = "not_filled_or_expired_next_bar"
            entry_attempt_rows.append(
                {
                    **approval,
                    "filled_at": filled_at,
                    "outcome": outcome,
                }
            )
        pd.DataFrame(
            entry_attempt_rows,
            columns=ENTRY_ATTEMPT_COLUMNS,
        ).to_parquet(destination / "entry_attempts.parquet", index=False)
    else:
        entry_attempt_rows = []
    if artifacts:
        visualizer.build_index(artifacts, destination / "decisions" / "index.html")
    pending_ai_proposals = ai_registry.pending()
    (destination / "ai_primitive_proposals.json").write_text(
        json.dumps(
            [to_primitive(proposal) for proposal in pending_ai_proposals],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "version": str(config_payload.get("version", "")),
        "source": str(loaded.source),
        "source_sha256": source_hash,
        "source_matches_preregistered_causal_front": (
            source_hash == validation.causal_front_sha256
        ),
        "source_role": loaded.source_role,
        "validation_protocol_version": validation.version,
        "validation_protocol_hash": validation.fingerprint,
        "validation_window_role": window.role,
        "config_hash": config_hash,
        "model_code_hash": code_hash,
        "structure_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "structure_protocol", None)
        ),
        "liquidity_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "liquidity_protocol", None)
        ),
        "displacement_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "displacement_protocol", None)
        ),
        "group3_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "group3_protocol", None)
        ),
        "group4_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "group4_protocol", None)
        ),
        "group5_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "group5_protocol", None)
        ),
        "execution_reality_source": (
            str(args.mbo_execution)
            if execution_store is not None
            else (
                "gross_policy_calibration_zero_cost"
                if args.gross_policy_calibration
                else "constant_cli_research_only"
            )
        ),
        "gross_policy_calibration": bool(args.gross_policy_calibration),
        "execution_authority": bool(execution_store is not None),
        "mbo_execution_manifest_hash": (
            hashlib.sha256(
                json.dumps(
                    execution_manifest,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if execution_manifest
            else None
        ),
        "mbo_execution_window_role": (
            execution_manifest.get("validation_window_role")
            if execution_manifest
            else None
        ),
        "contract_selection_causal": loaded.contract_selection_causal,
        "warnings": loaded.warnings,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "decision_clock_interval": "[start, end)",
        "decision_rows": len(decisions),
        "model_action_counts": (
            decisions["model_action"].value_counts().sort_index().to_dict()
            if not decisions.empty
            else {}
        ),
        "action_counts": (
            decisions["risk_action"].value_counts().sort_index().to_dict()
            if not decisions.empty
            else {}
        ),
        "approved_entry_attempts": len(entry_attempt_rows),
        "filled_entry_attempts": sum(
            row["outcome"] == "filled" for row in entry_attempt_rows
        ),
        "unfilled_or_expired_entry_attempts": sum(
            row["outcome"] == "not_filled_or_expired_next_bar"
            for row in entry_attempt_rows
        ),
        "pending_entry_attempts_at_end": sum(
            row["outcome"] == "pending_right_censored"
            for row in entry_attempt_rows
        ),
        "future_path_loaded": False,
        "future_path_visible_to_model": False,
        "sequential_execution_evaluated": bool(args.simulate_execution),
        "profitability_evaluated": bool(
            args.simulate_execution
            and trade_rows
            and not args.gross_policy_calibration
        ),
        "closed_trades": len(trade_rows),
        "funnel_transitions": len(funnel.rows),
        "path_tests": len(path_tests.results),
        "visual_path_audits_sealed": sealed_audit_count,
        "visual_path_audit_records": audit_records,
        "ai_review_directory": (
            None if ai_review_root is None else str(ai_review_root)
        ),
        "ai_review_files_used": ai_review_files_used,
        "ai_primitive_proposals": len(pending_ai_proposals),
        "ai_primitive_model_authority": False,
        "path_test_outcomes": (
            records_frame(path_tests.results)["outcome"].value_counts().to_dict()
            if path_tests.results
            else {}
        ),
        "net_R": (
            float(sum(float(row["net_R"]) for row in trade_rows))
            if trade_rows
            else 0.0
        ),
        "chart_artifacts": [to_primitive(item) for item in artifacts],
    }
    (destination / "summary.json").write_text(
        json.dumps(to_primitive(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["action_counts"], sort_keys=True))


def _stream_progress(
    state: dict[str, Any],
    *,
    total_source_rows: int,
    session_started: float,
    session_source_start: int,
) -> dict[str, Any]:
    elapsed = max(time.monotonic() - session_started, 1e-9)
    session_rows = max(
        0,
        int(state["source_rows_consumed"]) - int(session_source_start),
    )
    rows_per_second = session_rows / elapsed
    remaining = max(
        0,
        int(total_source_rows) - int(state["source_rows_consumed"]),
    )
    return {
        "completed_percent": (
            100.0
            if total_source_rows == 0
            else 100.0
            * int(state["source_rows_consumed"])
            / int(total_source_rows)
        ),
        "source_rows_processed": int(state["source_rows_consumed"]),
        "source_rows_total": int(total_source_rows),
        "processed_bars": int(state["processed_bars"]),
        "decision_rows": int(state["decision_rows"]),
        "brain_calibration_rows": int(
            state.get("brain_calibration_rows", 0)
        ),
        "rows_per_second": rows_per_second,
        "eta_seconds": (
            None if rows_per_second <= 0.0 else remaining / rows_per_second
        ),
        "resume_count": int(state["resume_count"]),
    }


def _stream_record(
    record: FunnelTransition | PathTestResult | BrainCalibrationRecord,
) -> dict[str, Any]:
    return asdict(record)


def _approval_row(snapshot) -> dict[str, Any]:
    thesis = snapshot.risk.frozen_thesis
    if thesis is None:
        raise ValueError("entry approval requires a frozen thesis")
    best_variant = snapshot.decision.utilities[0]
    if best_variant.action is not Action.ENTER:
        raise AssertionError("approved entry is not the best utility variant")
    plan = snapshot.decision.plan
    if plan is None:
        raise AssertionError("approved entry is missing its structural plan")
    variant_payload = {
        "verb": Action.ENTER.value,
        "playbook": thesis.playbook.value,
        "direction": thesis.direction.value,
        "setup_id": thesis.setup_id,
        "entry_location_id": thesis.entry_location_id,
        "entry_path_id": thesis.entry_path_id,
        "planned_entry": plan.planned_entry,
        "invalidation_source_id": plan.invalidation.source_level_id,
        "primary_target_id": plan.targets[0].level_id,
        "deadline": plan.deadline.isoformat(),
    }
    action_variant_hash = hashlib.sha256(
        json.dumps(
            variant_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selected = snapshot.belief.hypotheses.get(
        best_variant.hypothesis_key or ""
    )
    return {
        "thesis_hash": thesis.thesis_hash,
        "action_variant_hash": action_variant_hash,
        "decision_time": snapshot.observation.asof,
        "playbook": thesis.playbook.value,
        "direction": thesis.direction.value,
        "setup_id": thesis.setup_id,
        "entry_location_id": thesis.entry_location_id,
        "entry_path_id": thesis.entry_path_id,
        "planned_entry": thesis.entry,
        "original_invalidation": thesis.original_invalidation.price,
        "invalidation_source_id": (
            thesis.original_invalidation.source_level_id
        ),
        "invalidation_source": _invalidation_provenance(
            snapshot.observation,
            plan,
        ),
        "primary_target": thesis.original_targets[0].price,
        "primary_target_id": thesis.original_targets[0].level_id,
        "targets": _targets_provenance(snapshot.observation, plan),
        "deadline": thesis.deadline,
        "decision_probability": (
            None if selected is None else selected.probability
        ),
        "decision_raw_probability": (
            None if selected is None else selected.raw_probability
        ),
        "decision_phase": None if selected is None else selected.phase.value,
        "best_variant_utility_R": best_variant.utility,
        "path_structural_score_R": _enter_structural_score(best_variant),
        "best_variant_components": json.dumps(
            dict(best_variant.components),
            sort_keys=True,
        ),
    }


def _streamed_main(args: argparse.Namespace) -> None:
    if args.gross_policy_calibration:
        raise RuntimeError(
            "--gross-policy-calibration is retired; use "
            "scripts/run_managed_policy_calibration.py"
        )
    if args.chart_every or args.audit_path_tests or args.ai_review_directory:
        raise ValueError(
            "streamed development replay keeps visualization and AI review "
            "out of checkpoint state; use scripts/run_scenario_visual_audit.py "
            "or pass --legacy-visual-replay for the former workflow"
        )
    if args.include_decision_traces:
        raise RuntimeError(
            "streamed history replay no longer emits full minute traces; "
            "select a 20-40 case batch from lightweight decision shards and "
            "render sampled traces with scripts/render_blind_decision_batch.py"
        )
    if args.warmup_days < 0:
        raise ValueError("warmup days cannot be negative")
    if args.shard_rows < 1 or args.checkpoint_bars < 1:
        raise ValueError("shard rows and checkpoint bars must be positive")
    effective_shard_rows = int(args.shard_rows)
    if args.diagnostic_stop_after_bars < 0:
        raise ValueError("diagnostic stop cannot be negative")
    if args.spread_points is not None and args.spread_points < 0:
        raise ValueError("constant research spread cannot be negative")
    streamed_slippage_points = (
        0.0 if args.slippage_points is None else float(args.slippage_points)
    )
    if streamed_slippage_points < 0:
        raise ValueError("constant research slippage cannot be negative")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None:
        start = start.tz_localize("America/New_York")
    if end.tzinfo is None:
        end = end.tz_localize("America/New_York")
    if end <= start:
        raise ValueError("replay interval must be positive")

    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_ohlcv(start, end)
    if window.role == "sealed_holdout" and not args.reveal_sealed_holdout:
        raise RuntimeError(
            "requested interval is the sealed OHLCV holdout; reveal is reserved "
            "for a frozen final-validation candidate"
        )
    source = Path(args.source)
    source_hash = _sha256_file(source)
    if (
        source_hash != validation.causal_front_sha256
        and not args.acknowledge_research_roll_lineage
    ):
        raise RuntimeError(
            "OHLCV source hash does not match the registered causal front"
        )

    load_start = start - pd.Timedelta(days=args.warmup_days)
    loaded = load_ohlcv(source, start=load_start, end=end)
    if (
        not loaded.contract_selection_causal
        and not args.acknowledge_research_roll_lineage
    ):
        raise RuntimeError(
            "development replay requires strict previous-session contract selection"
        )
    replay_frame = loaded.frame.loc[
        loaded.frame.index + pd.Timedelta(minutes=1) < end
    ]
    if replay_frame.empty:
        raise ValueError("requested replay interval has no source rows")
    total_source_rows = int(len(replay_frame))

    config_source = Path(args.config)
    config_payload = json.loads(config_source.read_text(encoding="utf-8"))
    managed_artifact = config_payload.get("managed_policy_artifact")
    if managed_artifact:
        engine = build_v2_1_engine(config_source)
        code_hash = managed_policy_code_fingerprint()
    else:
        engine = ContinuousSMCEngine.from_config(config_source)
        code_hash = model_code_fingerprint()
    config_hash = sha256_file(config_source)
    brain_calibration_enabled = bool(
        engine.brain.registry.registry_version.startswith("4.")
        and window.role in {"calibration", "belief_calibration"}
    )
    brain_calibration_protocol_hashes = (
        _configured_primitive_protocol_hashes(
            config_payload.get("observer", {})
        )
        if brain_calibration_enabled
        else {}
    )
    brain_calibration_input_contract_hash = (
        brain_input_contract_hash(engine.reader.scale_specs)
        if brain_calibration_enabled
        else None
    )

    execution_manifest: dict[str, Any] = {}
    if args.mbo_execution:
        execution_store, execution_manifest = _load_mbo_execution(
            args.mbo_execution,
            validation=validation,
            start=start,
            end=end,
            reveal_sealed_holdout=args.reveal_sealed_mbo_holdout,
        )
        execution_mode = "mbo_causal_execution"
    else:
        execution_store = None
        execution_mode = (
            "ohlcv_only_execution_unavailable"
            if args.spread_points is None
            else "constant_cli_research_only"
        )

    destination = Path(args.output)
    completed_path = destination / "COMPLETED.json"
    progress_path = destination / "progress.json"
    if args.resume:
        if completed_path.exists():
            raise FileExistsError("development replay is already complete")
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "refusing non-resume use of a non-empty development output"
        )
    destination.mkdir(parents=True, exist_ok=True)

    bindings = {
        "runner": "continuous_development_stream_v1",
        "source_sha256": source_hash,
        "source_rows": total_source_rows,
        "source_first": replay_frame.index[0].isoformat(),
        "source_last": replay_frame.index[-1].isoformat(),
        "source_role": loaded.source_role,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "warmup_days": int(args.warmup_days),
        "config_sha256": config_hash,
        "model_code_hash": code_hash,
        "playbook_registry_hash": engine.brain.registry.fingerprint,
        "belief_calibration_hash": engine.brain.calibrator.fingerprint,
        "brain_calibration_capture": brain_calibration_enabled,
        "brain_calibration_recorder_version": (
            BRAIN_CALIBRATION_RECORDER_VERSION
            if brain_calibration_enabled
            else None
        ),
        "brain_calibration_recorder_sha256": (
            sha256_file(ROOT / "smc_trader/brain_calibration.py")
            if brain_calibration_enabled
            else None
        ),
        "brain_input_contract_hash": (
            brain_calibration_input_contract_hash
        ),
        "managed_policy_calibration_hash": getattr(
            getattr(engine.decision, "calibrator", None),
            "fingerprint",
            None,
        ),
        "structure_protocol_sha256": _optional_protocol_sha256(
            engine.observer.config.structure_protocol
        ),
        "liquidity_protocol_sha256": _optional_protocol_sha256(
            engine.observer.config.liquidity_protocol
        ),
        "displacement_protocol_sha256": _optional_protocol_sha256(
            engine.observer.config.displacement_protocol
        ),
        "group3_protocol_sha256": _optional_protocol_sha256(
            engine.observer.config.group3_protocol
        ),
        "group4_protocol_sha256": _optional_protocol_sha256(
            engine.observer.config.group4_protocol
        ),
        "group5_protocol_sha256": _optional_protocol_sha256(
            engine.observer.config.group5_protocol
        ),
        "validation_protocol_hash": validation.fingerprint,
        "validation_window_role": window.role,
        "runner_sha256": sha256_file(Path(__file__)),
        "decision_trace_schema_version": (
            TRACE_SCHEMA_VERSION
            if args.include_decision_traces
            else None
        ),
        "decision_trace_sha256": (
            sha256_file(ROOT / "smc_trader/decision_trace.py")
            if args.include_decision_traces
            else None
        ),
        "include_decision_traces": bool(
            args.include_decision_traces
        ),
        "calibration_replay_sha256": sha256_file(
            ROOT / "smc_trader/calibration_replay.py"
        ),
        "artifact_stream_sha256": sha256_file(
            ROOT / "smc_trader/artifact_stream.py"
        ),
        "simulate_execution": bool(args.simulate_execution),
        "execution_mode": execution_mode,
        "constant_spread_points": args.spread_points,
        "constant_slippage_points": streamed_slippage_points,
        "mbo_execution_sha256": (
            None
            if args.mbo_execution is None
            else sha256_file(args.mbo_execution)
        ),
        "mbo_execution_manifest_sha256": (
            None
            if args.mbo_execution is None
            else sha256_file(
                Path(args.mbo_execution).with_suffix(
                    Path(args.mbo_execution).suffix + ".manifest.json"
                )
            )
        ),
        "shard_rows": effective_shard_rows,
        "checkpoint_bars": int(args.checkpoint_bars),
        "hash_mode": HASH_MODE,
        "allow_data_gap_reset": False,
        "visual_state_checkpointed": False,
        "ai_state_checkpointed": False,
    }

    brain_lineage_path = destination / "brain_calibration_lineage.json"
    if brain_calibration_enabled:
        brain_lineage = canonical_json(
            {
                "format_version": 1,
                "lineage_contract": "brain-calibration-input-v1",
                "recorder_version": BRAIN_CALIBRATION_RECORDER_VERSION,
                "source_sha256": source_hash,
                "source_rows": total_source_rows,
                "start": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "brain_input_contract_hash": (
                    brain_calibration_input_contract_hash
                ),
                "playbook_registry_hash": engine.brain.registry.fingerprint,
                "model_code_hash": code_hash,
                "config_hash": config_hash,
                "primitive_protocol_hashes": (
                    brain_calibration_protocol_hashes
                ),
                "legacy_rows_or_checkpoints_accepted": False,
            }
        )
        if brain_lineage_path.exists():
            if brain_lineage_path.read_bytes() != brain_lineage:
                raise ValueError("Brain calibration lineage sidecar is stale")
        elif args.resume:
            raise FileNotFoundError(
                "resume requires the bound Brain calibration lineage sidecar"
            )
        else:
            atomic_bytes(brain_lineage_path, brain_lineage)

    checkpoint = ReplayCheckpointStore(destination / "_checkpoint")
    total_fields = {
        "decision_shards": "decision_rows",
        "funnel_transition_shards": "funnel_rows",
        "path_test_shards": "path_test_rows",
        "brain_calibration_shards": "brain_calibration_rows",
    }

    def sync_checkpoint_aliases(state: dict[str, Any]) -> None:
        decision_stream = state["streams"]["decision_shards"]
        state["next_shard_index"] = int(
            decision_stream["next_shard_index"]
        )
        state["committed_shards"] = decision_stream["committed_shards"]

    def verify_state_streams(state: dict[str, Any]) -> None:
        if set(state.get("streams", {})) != set(STREAM_KEYS):
            raise ValueError("checkpoint stream family changed")
        if set(state.get("buffers", {})) != set(STREAM_KEYS):
            raise ValueError("checkpoint buffer family changed")
        for name in STREAM_KEYS:
            committed = verify_stream_shards(
                destination,
                state["streams"][name],
            )
            buffer = state["buffers"][name]
            if (
                not isinstance(buffer, list)
                or len(buffer) >= effective_shard_rows
            ):
                raise ValueError("checkpoint contains an invalid bounded buffer")
            expected = int(state[total_fields[name]])
            if committed + len(buffer) != expected:
                raise ValueError(f"{name} rows are not conserved by checkpoint")

    if args.resume:
        if not checkpoint.exists:
            raise FileNotFoundError("resume requested without a valid checkpoint")
        state = checkpoint.load(
            expected_bindings=bindings,
            expected_replay_type=CalibrationSequentialReplay,
        )
        verify_state_streams(state)
        state["resume_count"] = int(state["resume_count"]) + 1
        sync_checkpoint_aliases(state)
        checkpoint.save(state, bindings=bindings)
    else:
        streams = {
            name: new_stream_state(STREAM_FIELD_TYPES[name])
            for name in STREAM_KEYS
        }
        state = {
            "replay": CalibrationSequentialReplay(
                engine=engine,
                simulate_execution=bool(args.simulate_execution),
            ),
            "funnel": PlaybookFunnelRecorder(),
            "path_tests": FrozenPathTestRecorder(
                config_hash=config_hash,
                code_hash=code_hash,
            ),
            "brain_calibration": (
                BrainCalibrationRecorder(
                    registry_hash=engine.brain.registry.fingerprint,
                    model_code_hash=code_hash,
                    config_hash=config_hash,
                    primitive_protocol_hashes=(
                        brain_calibration_protocol_hashes
                    ),
                    brain_input_contract_hash=str(
                        brain_calibration_input_contract_hash
                    ),
                )
                if brain_calibration_enabled
                else None
            ),
            "streams": streams,
            "buffers": {name: [] for name in STREAM_KEYS},
            "processed_bars": 0,
            "source_rows_consumed": 0,
            "last_checkpoint_processed_bars": 0,
            "decision_rows": 0,
            "funnel_rows": 0,
            "path_test_rows": 0,
            "brain_calibration_rows": 0,
            "model_action_counts": {},
            "risk_action_counts": {},
            "path_outcome_counts": {},
            "entry_approvals": {},
            "filled_entries": {},
            "last_source_start": None,
            "last_asof": None,
            "resume_count": 0,
            "finalized": False,
            "peak_buffer_rows": {name: 0 for name in STREAM_KEYS},
            "next_shard_index": 0,
            "committed_shards": streams["decision_shards"][
                "committed_shards"
            ],
        }
    args._streamed_output_owned = True

    replay: CalibrationSequentialReplay = state["replay"]
    funnel: PlaybookFunnelRecorder = state["funnel"]
    path_tests: FrozenPathTestRecorder = state["path_tests"]
    brain_calibration: BrainCalibrationRecorder | None = state.get(
        "brain_calibration"
    )
    buffers: dict[str, list[dict[str, Any]]] = state["buffers"]
    if (brain_calibration is not None) != brain_calibration_enabled:
        raise ValueError("checkpoint Brain calibration capture mode changed")
    if (
        brain_calibration is not None
        and not isinstance(brain_calibration, BrainCalibrationRecorder)
    ):
        raise ValueError("checkpoint Brain calibration recorder is invalid")
    if bool(replay.simulate_execution) != bool(args.simulate_execution):
        raise ValueError("checkpoint execution-simulation mode changed")
    if state["finalized"] and int(state["source_rows_consumed"]) != total_source_rows:
        raise ValueError("finalized checkpoint did not consume the bound source")

    session_started = time.monotonic()
    session_source_start = int(state["source_rows_consumed"])
    durable_progress = _stream_progress(
        state,
        total_source_rows=total_source_rows,
        session_started=session_started,
        session_source_start=session_source_start,
    )

    def append_stream(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        offset = 0
        while offset < len(rows):
            capacity = effective_shard_rows - len(buffers[name])
            take = min(capacity, len(rows) - offset)
            buffers[name].extend(rows[offset : offset + take])
            offset += take
            state[total_fields[name]] = (
                int(state[total_fields[name]]) + take
            )
            state["peak_buffer_rows"][name] = max(
                int(state["peak_buffer_rows"][name]),
                len(buffers[name]),
            )
            if len(buffers[name]) >= effective_shard_rows:
                flush_stream(name, final=False)

    def flush_stream(name: str, *, final: bool) -> None:
        buffer = buffers[name]
        while len(buffer) >= effective_shard_rows or (final and buffer):
            take = min(len(buffer), effective_shard_rows)
            chunk = buffer[:take]
            write_stream_shards_bounded(
                destination,
                name,
                chunk,
                state["streams"][name],
                key_column=STREAM_KEYS[name],
                maximum_rows=effective_shard_rows,
                field_types=STREAM_FIELD_TYPES[name],
            )
            del buffer[:take]

    def commit_checkpoint(*, final: bool = False) -> None:
        nonlocal durable_progress, safe_source_checkpoint
        if state["last_source_start"] is None:
            return
        was_safe_source_checkpoint = safe_source_checkpoint
        safe_source_checkpoint = False
        for name in STREAM_KEYS:
            flush_stream(name, final=final)
        state["last_checkpoint_processed_bars"] = int(
            state["processed_bars"]
        )
        sync_checkpoint_aliases(state)
        checkpoint.save(state, bindings=bindings)
        durable_progress = _stream_progress(
            state,
            total_source_rows=total_source_rows,
            session_started=session_started,
            session_source_start=session_source_start,
        )
        atomic_bytes(
            progress_path,
            canonical_json(
                {
                    **durable_progress,
                    "status": "running",
                    "resume_supported": True,
                    "durable_checkpoint_only": True,
                }
            ),
        )
        print(json.dumps(durable_progress, sort_keys=True), flush=True)
        safe_source_checkpoint = was_safe_source_checkpoint

    def execution_for_bar(bar) -> ExecutionRealityInput:
        if execution_store is not None and bar.end >= start:
            return execution_store.for_bar(
                bar,
                deadline=_deadline(bar.end),
            )
        if args.spread_points is None:
            return ExecutionRealityInput(
                spread_points=None,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                deadline=_deadline(bar.end),
                source="ohlcv_only_execution_unavailable",
            )
        return ExecutionRealityInput(
            spread_points=float(args.spread_points),
            expected_slippage_points=streamed_slippage_points,
            deadline=_deadline(bar.end),
            source="constant_cli_research_only",
        )

    iterator = iter_after_source_checkpoint(
        replay_frame,
        state["last_source_start"],
        allow_data_gap_reset=False,
    )
    safe_source_checkpoint = False
    try:
        if not state["finalized"]:
            for bar in iterator:
                safe_source_checkpoint = False
                previous_snapshot = replay.engine.last_snapshot
                if brain_calibration is not None:
                    brain_calibration.on_bar(bar)
                    append_stream(
                        "brain_calibration_shards",
                        [
                            _stream_record(item)
                            for item in brain_calibration.drain_rows()
                        ],
                    )
                path_tests.on_bar(bar)
                resolved = path_tests.drain_results()
                if resolved:
                    append_stream(
                        "path_test_shards",
                        [_stream_record(item) for item in resolved],
                    )
                    for item in resolved:
                        state["path_outcome_counts"][item.outcome] = (
                            int(
                                state["path_outcome_counts"].get(
                                    item.outcome,
                                    0,
                                )
                            )
                            + 1
                        )

                step = replay.on_bar(
                    bar,
                    execution=execution_for_bar(bar),
                )
                snapshot = step.snapshot
                if args.simulate_execution:
                    for record in step.closed_trades:
                        state["filled_entries"].setdefault(
                            record.thesis_hash,
                            record.opened_at,
                        )
                    if step.position is not None:
                        state["filled_entries"].setdefault(
                            step.position.thesis_hash,
                            step.position.opened_at,
                        )

                if snapshot.observation.asof >= start:
                    if brain_calibration is not None:
                        brain_calibration.observe(
                            snapshot,
                            source_bar=bar,
                        )
                        append_stream(
                            "brain_calibration_shards",
                            [
                                _stream_record(item)
                                for item in brain_calibration.drain_rows()
                            ],
                        )
                    funnel.observe(snapshot)
                    append_stream(
                        "funnel_transition_shards",
                        [
                            _stream_record(item)
                            for item in funnel.drain_rows()
                        ],
                    )
                    path_tests.observe(snapshot)
                    append_stream(
                        "decision_shards",
                        [
                            _row(
                                snapshot,
                                previous_snapshot,
                                source_bar=bar,
                                account_state=step.account_state,
                                belief_position_input=(
                                    step.belief_position_input
                                ),
                                include_decision_trace=(
                                    args.include_decision_traces
                                ),
                            )
                        ],
                    )
                    model_action = snapshot.decision.selected_action.value
                    risk_action = snapshot.risk.final_action.value
                    state["model_action_counts"][model_action] = (
                        int(state["model_action_counts"].get(model_action, 0))
                        + 1
                    )
                    state["risk_action_counts"][risk_action] = (
                        int(state["risk_action_counts"].get(risk_action, 0))
                        + 1
                    )
                    state["last_asof"] = snapshot.observation.asof
                    if (
                        args.simulate_execution
                        and snapshot.risk.final_action is Action.ENTER
                        and snapshot.risk.frozen_thesis is not None
                    ):
                        approval = _approval_row(snapshot)
                        thesis_hash = approval["thesis_hash"]
                        if thesis_hash in state["entry_approvals"]:
                            raise AssertionError(
                                "risk-approved thesis was emitted twice"
                            )
                        state["entry_approvals"][thesis_hash] = approval

                state["processed_bars"] = int(state["processed_bars"]) + 1
                if not bar.synthetic_no_trade:
                    state["source_rows_consumed"] = (
                        int(state["source_rows_consumed"]) + 1
                    )
                    state["last_source_start"] = bar.start
                    safe_source_checkpoint = True

                due = (
                    int(state["source_rows_consumed"]) == 1
                    or int(state["processed_bars"])
                    - int(state["last_checkpoint_processed_bars"])
                    >= args.checkpoint_bars
                    or any(
                        len(buffers[name]) >= effective_shard_rows
                        for name in STREAM_KEYS
                    )
                )
                if safe_source_checkpoint and due:
                    commit_checkpoint()
                if (
                    args.diagnostic_stop_after_bars > 0
                    and int(state["processed_bars"])
                    >= args.diagnostic_stop_after_bars
                    and safe_source_checkpoint
                ):
                    commit_checkpoint()
                    raise RuntimeError(
                        "intentional diagnostic interruption after checkpoint"
                    )
    except KeyboardInterrupt as exc:
        if safe_source_checkpoint:
            commit_checkpoint()
        atomic_bytes(
            progress_path,
            canonical_json(
                {
                    **durable_progress,
                    "status": "interrupted",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "resume_supported": checkpoint.exists,
                    "durable_checkpoint_only": True,
                }
            ),
        )
        raise
    except Exception as exc:
        atomic_bytes(
            progress_path,
            canonical_json(
                {
                    **durable_progress,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "resume_supported": checkpoint.exists,
                    "durable_checkpoint_only": True,
                }
            ),
        )
        raise

    if state["last_asof"] is None or int(state["decision_rows"]) == 0:
        raise ValueError("requested interval produced no completed decisions")
    if int(state["source_rows_consumed"]) != total_source_rows:
        raise RuntimeError("source iterator ended before all bound rows were consumed")
    if not state["finalized"]:
        if brain_calibration is not None:
            brain_calibration.close_unresolved(state["last_asof"])
            append_stream(
                "brain_calibration_shards",
                [
                    _stream_record(item)
                    for item in brain_calibration.drain_rows()
                ],
            )
        path_tests.close_unresolved(state["last_asof"])
        resolved = path_tests.drain_results()
        append_stream(
            "path_test_shards",
            [_stream_record(item) for item in resolved],
        )
        for item in resolved:
            state["path_outcome_counts"][item.outcome] = (
                int(state["path_outcome_counts"].get(item.outcome, 0)) + 1
            )
        state["finalized"] = True
    commit_checkpoint(final=True)
    verify_state_streams(state)

    manifest_hashes: dict[str, str] = {}
    for name in STREAM_KEYS:
        manifest_path = write_stream_manifest(
            destination,
            name,
            state["streams"][name],
            artifact=f"continuous_development_{name}",
            bindings=bindings,
        )
        manifest_hashes[name] = sha256_file(manifest_path)

    trade_rows: list[dict[str, Any]] = []
    entry_attempt_rows: list[dict[str, Any]] = []
    if args.simulate_execution:
        if replay.portfolio is None:
            raise AssertionError("simulation mode lost its portfolio")
        trade_rows = [
            asdict(record)
            for record in replay.portfolio.records
            if record.decision_time >= start
        ]
        atomic_parquet(
            pd.DataFrame(trade_rows, columns=TRADE_COLUMNS),
            destination / "trades.parquet",
            field_types=TRADE_FIELD_TYPES,
        )
        for thesis_hash, approval in state["entry_approvals"].items():
            filled_at = state["filled_entries"].get(thesis_hash)
            if filled_at is not None:
                outcome = "filled"
            elif approval["decision_time"] == state["last_asof"]:
                outcome = "pending_right_censored"
            else:
                outcome = "not_filled_or_expired_next_bar"
            entry_attempt_rows.append(
                {
                    **approval,
                    "filled_at": filled_at,
                    "outcome": outcome,
                }
            )
        atomic_parquet(
            pd.DataFrame(
                entry_attempt_rows,
                columns=ENTRY_ATTEMPT_COLUMNS,
            ),
            destination / "entry_attempts.parquet",
            field_types=ENTRY_ATTEMPT_FIELD_TYPES,
        )

    ai_proposals_path = destination / "ai_primitive_proposals.json"
    atomic_bytes(
        ai_proposals_path,
        json.dumps([], separators=(",", ":")).encode("utf-8"),
    )
    summary = {
        "version": str(config_payload.get("version", "")),
        "source": str(loaded.source),
        "source_sha256": source_hash,
        "source_matches_preregistered_causal_front": (
            source_hash == validation.causal_front_sha256
        ),
        "source_role": loaded.source_role,
        "validation_protocol_version": validation.version,
        "validation_protocol_hash": validation.fingerprint,
        "validation_window_role": window.role,
        "config_hash": config_hash,
        "model_code_hash": code_hash,
        "execution_reality_source": execution_mode,
        "execution_authority": bool(execution_store is not None),
        "contract_selection_causal": loaded.contract_selection_causal,
        "warnings": list(loaded.warnings),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "decision_clock_interval": "[start, end)",
        "decision_rows": int(state["decision_rows"]),
        "funnel_transitions": int(state["funnel_rows"]),
        "path_tests": int(state["path_test_rows"]),
        "brain_calibration_capture": brain_calibration_enabled,
        "brain_calibration_rows": int(
            state["brain_calibration_rows"]
        ),
        "brain_input_contract_hash": (
            brain_calibration_input_contract_hash
        ),
        "model_action_counts": dict(
            sorted(state["model_action_counts"].items())
        ),
        "action_counts": dict(sorted(state["risk_action_counts"].items())),
        "path_test_outcomes": dict(
            sorted(state["path_outcome_counts"].items())
        ),
        "approved_entry_attempts": len(entry_attempt_rows),
        "filled_entry_attempts": sum(
            row["outcome"] == "filled" for row in entry_attempt_rows
        ),
        "unfilled_or_expired_entry_attempts": sum(
            row["outcome"] == "not_filled_or_expired_next_bar"
            for row in entry_attempt_rows
        ),
        "pending_entry_attempts_at_end": sum(
            row["outcome"] == "pending_right_censored"
            for row in entry_attempt_rows
        ),
        "snapshot_hash_mode": HASH_MODE,
        "rolling_state_commitment": replay.rolling_commitment,
        "full_snapshot_hash_per_minute": False,
        "checkpoint_resume_supported": True,
        "output_contract": "manifest_first_shards_v1",
        "resume_count": int(state["resume_count"]),
        "shard_rows": effective_shard_rows,
        "checkpoint_bars": int(args.checkpoint_bars),
        "stream_rows": {
            name: int(state["streams"][name]["rows"])
            for name in STREAM_KEYS
        },
        "peak_buffer_rows": dict(state["peak_buffer_rows"]),
        "future_path_visible_to_model": False,
        "future_path_outcomes_separate_stream": True,
        "sequential_execution_evaluated": bool(args.simulate_execution),
        "profitability_evaluated": bool(
            args.simulate_execution and trade_rows
        ),
        "closed_trades": len(trade_rows),
        "net_R": (
            float(sum(float(row["net_R"]) for row in trade_rows))
            if args.simulate_execution
            else None
        ),
        "chart_artifacts": [],
        "ai_primitive_model_authority": False,
    }
    summary_path = destination / "summary.json"
    atomic_bytes(summary_path, canonical_json(to_primitive(summary)))
    final_progress = {
        **_stream_progress(
            state,
            total_source_rows=total_source_rows,
            session_started=session_started,
            session_source_start=session_source_start,
        ),
        "status": "complete",
        "resume_supported": False,
    }
    atomic_bytes(progress_path, canonical_json(final_progress))
    auxiliary_hashes = {
        "ai_primitive_proposals.json": sha256_file(ai_proposals_path),
    }
    if brain_calibration_enabled:
        auxiliary_hashes["brain_calibration_lineage.json"] = sha256_file(
            brain_lineage_path
        )
    if args.simulate_execution:
        auxiliary_hashes.update(
            {
                "trades.parquet": sha256_file(destination / "trades.parquet"),
                "entry_attempts.parquet": sha256_file(
                    destination / "entry_attempts.parquet"
                ),
            }
        )
    atomic_bytes(
        completed_path,
        canonical_json(
            {
                "format_version": 1,
                "status": "complete",
                "bindings": bindings,
                "stream_manifest_sha256": manifest_hashes,
                "summary_sha256": sha256_file(summary_path),
                "progress_sha256": sha256_file(progress_path),
                "auxiliary_artifact_sha256": auxiliary_hashes,
                "checkpoint_manifest_sha256": sha256_file(
                    checkpoint.manifest_path
                ),
            }
        ),
    )
    print(json.dumps(summary["action_counts"], sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.legacy_visual_replay:
        _legacy_visual_main(args)
        return
    try:
        _streamed_main(args)
    except Exception as exc:
        destination = Path(args.output)
        progress_path = destination / "progress.json"
        checkpoint_manifest = destination / "_checkpoint" / "manifest.json"
        if (
            bool(getattr(args, "_streamed_output_owned", False))
            and not (destination / "COMPLETED.json").exists()
            and (progress_path.exists() or checkpoint_manifest.exists())
        ):
            try:
                prior_progress = (
                    json.loads(progress_path.read_text(encoding="utf-8"))
                    if progress_path.is_file()
                    else {}
                )
            except (OSError, ValueError, TypeError):
                prior_progress = {}
            atomic_bytes(
                progress_path,
                canonical_json(
                    {
                        **prior_progress,
                        "status": "failed",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                        "resume_supported": checkpoint_manifest.is_file(),
                        "durable_checkpoint_only": True,
                    }
                ),
            )
        raise


if __name__ == "__main__":
    main()
