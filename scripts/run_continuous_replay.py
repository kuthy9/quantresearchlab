#!/usr/bin/env python3
"""Run the causal model with bounded, resumable development outputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
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
    ReplayCheckpointStore,
    iter_after_source_checkpoint,
)
from smc_trader.brain_calibration import (  # noqa: E402
    BrainCalibrationRecord,
    BrainCalibrationRecorder,
    RECORDER_SCHEMA_VERSION as BRAIN_CALIBRATION_RECORDER_SCHEMA_VERSION,
)
from smc_trader.engine import ContinuousSMCEngine  # noqa: E402
from smc_trader.io import load_ohlcv  # noqa: E402
from smc_trader.model import (  # noqa: E402
    Action,
    Timeframe,
    to_primitive,
)
from smc_trader.mbo import (  # noqa: E402
    MinuteExecutionRealityStore,
    assert_mbo_source_allowed,
)
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.validation import (  # noqa: E402
    load_validation_protocol,
)
from smc_trader.visualization import (  # noqa: E402
    DecisionVisualizer,
    VisualArtifact,
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
    "entry_attempt_id",
    "decision_time",
    "playbook",
    "direction",
    "setup_id",
    "episode_id",
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
    "entry_attempt_id": "large_string",
    "decision_time": "timestamp_ny",
    "playbook": "large_string",
    "direction": "large_string",
    "setup_id": "large_string",
    "episode_id": "large_string",
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
    "best_variant_components": "large_string",
    "filled_at": "timestamp_ny",
    "outcome": "large_string",
}

BASE_STREAM_KEYS = {
    "decision_shards": "decision_id",
}

DECISION_FIELD_TYPES = {
    "decision_id": "large_string",
    "asof": "timestamp_ny",
    "symbol": "large_string",
    "instrument_id": "int64",
    "price": "float64",
    "model_action": "large_string",
    "risk_action": "large_string",
    "utility_advantage_R": "float64",
    "action_utilities": "large_string",
    "best_variant_action": "large_string",
    "best_variant_utility_R": "float64",
    "best_variant_hypothesis_key": "large_string",
    "model_reasons": "large_string",
    "top_playbook": "large_string",
    "top_direction": "large_string",
    "top_probability": "float64",
    "top_raw_probability": "float64",
    "top_thesis_strength": "float64",
    "top_sequence_progress": "float64",
    "top_location_quality": "float64",
    "top_entry_readiness": "float64",
    "top_delivery_quality": "float64",
    "top_uncertainty": "float64",
    "top_phase": "large_string",
    "top_terminal_reason": "large_string",
    "top_failed_hard_gate_ids": "large_string",
    "top_setup_id": "large_string",
    "top_episode_id": "large_string",
    "top_entry_location_id": "large_string",
    "decision_hypothesis_key": "large_string",
    "decision_playbook": "large_string",
    "decision_direction": "large_string",
    "decision_phase": "large_string",
    "decision_setup_id": "large_string",
    "decision_episode_id": "large_string",
    "decision_entry_location_id": "large_string",
    "decision_entry_path_id": "large_string",
    "h4_regime": "large_string",
    "position_open": "bool",
    "position_thesis_hash": "large_string",
    "position_setup_id": "large_string",
    "position_playbook": "large_string",
    "position_direction": "large_string",
    "observation_anomalies": "large_string",
    "planned_entry": "float64",
    "entry_zone_lower": "float64",
    "entry_zone_upper": "float64",
    "invalidation": "float64",
    "invalidation_source_id": "large_string",
    "primary_target": "float64",
    "primary_target_id": "large_string",
    "selected_draw_id": "large_string",
    "liquidity_route_id": "large_string",
    "context_draw_id": "large_string",
    "intermediate_liquidity_ids": "large_string",
    "primary_deliverable_target_id": "large_string",
    "terminal_draw_id": "large_string",
    "path_blocker_ids": "large_string",
    "source_path_ids": "large_string",
    "target_ids": "large_string",
    "primary_target_R": "float64",
    "remaining_path_R": "float64",
    "deadline": "timestamp_ny",
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
    "risk_reasons": "large_string",
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
}

STREAM_FIELD_TYPES = {
    "decision_shards": DECISION_FIELD_TYPES,
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


def _calendar_warmup_start(
    start: pd.Timestamp,
    *,
    days: int,
) -> pd.Timestamp:
    """Subtract local market-calendar days instead of fixed 24-hour blocks."""

    local = start.tz_convert("America/New_York")
    return local - pd.DateOffset(days=days)


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
    return None


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
    registered = None
    for artifact in validation.mbo_identity.development_execution_artifacts:
        registered_path = Path(artifact.path)
        if not registered_path.is_absolute():
            registered_path = ROOT / registered_path
        if registered_path.resolve(strict=False) == source:
            registered = artifact
            break
    if registered is None:
        raise RuntimeError("MBO execution artifact is not registered in data_splits.json")
    registered_manifest_path = Path(registered.manifest_path)
    if not registered_manifest_path.is_absolute():
        registered_manifest_path = ROOT / registered_manifest_path
    if registered_manifest_path.resolve(strict=False) != manifest_path:
        raise RuntimeError("MBO execution manifest path is not the registered manifest")
    if sha256_file(manifest_path) != registered.manifest_sha256:
        raise RuntimeError("MBO execution manifest does not match data_splits.json")
    actual_hash = sha256_file(source)
    if actual_hash != registered.sha256 or manifest.get("output_sha256") != actual_hash:
        raise RuntimeError("MBO execution parquet does not match its manifest hash")
    requested_window = validation.classify_mbo(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
    )
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
    *,
    account_state=None,
    belief_position_input=None,
) -> dict:
    ranked = snapshot.belief.ranked()
    top = ranked[0] if ranked else None
    decision_hypothesis = (
        None
        if snapshot.decision.best_hypothesis_key is None
        else snapshot.belief.hypotheses.get(
            snapshot.decision.best_hypothesis_key
        )
    )
    summary_hypothesis = decision_hypothesis or top
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
    sequence = None if top is None else top.sequence
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
    position = (
        getattr(account_state, "position", None)
        if account_state is not None
        else belief_position_input
    )
    if position is not None and getattr(position, "status", "open") != "open":
        position = None
    action_utilities: dict[str, dict[str, Any]] = {}
    for utility in snapshot.decision.utilities:
        existing = action_utilities.get(utility.action.value)
        if existing is None or float(utility.utility) > float(existing["utility_R"]):
            action_utilities[utility.action.value] = {
                "utility_R": float(utility.utility),
                "hypothesis_key": utility.hypothesis_key,
                "reason": utility.reason,
            }
    asof = snapshot.observation.asof
    return {
        "decision_id": (
            f"{snapshot.observation.symbol}:"
            f"{snapshot.observation.instrument_id}:{asof.isoformat()}"
        ),
        "asof": asof,
        "symbol": snapshot.observation.symbol,
        "instrument_id": snapshot.observation.instrument_id,
        "price": snapshot.observation.price,
        "model_action": snapshot.decision.selected_action.value,
        "risk_action": snapshot.risk.final_action.value,
        "utility_advantage_R": snapshot.decision.advantage,
        "action_utilities": json.dumps(
            action_utilities,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "best_variant_action": (
            None if best_variant is None else best_variant.action.value
        ),
        "best_variant_utility_R": (
            None if best_variant is None else best_variant.utility
        ),
        "best_variant_hypothesis_key": (
            None if best_variant is None else best_variant.hypothesis_key
        ),
        "model_reasons": json.dumps(
            list(snapshot.decision.reasons),
            ensure_ascii=False,
        ),
        "top_playbook": None if top is None else top.playbook.value,
        "top_direction": None if top is None else top.direction.value,
        "top_probability": None if top is None else top.probability,
        "top_raw_probability": (
            None if top is None else top.raw_probability
        ),
        "top_thesis_strength": None if top is None else top.thesis_strength,
        "top_sequence_progress": None if top is None else top.sequence_progress,
        "top_location_quality": None if top is None else top.location_quality,
        "top_entry_readiness": None if top is None else top.entry_readiness,
        "top_delivery_quality": None if top is None else top.delivery_quality,
        "top_uncertainty": None if top is None else top.uncertainty,
        "top_phase": None if top is None else top.phase.value,
        "top_terminal_reason": (
            None if top is None else top.terminal_reason
        ),
        "top_failed_hard_gate_ids": top_failed_hard_gate_ids,
        "top_setup_id": (
            None
            if top is None
            else (None if sequence is None else sequence.setup_id)
            or top.setup_context_id
        ),
        "top_episode_id": None if top is None else top.episode_id,
        "top_entry_location_id": None if top is None else top.entry_location_id,
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
        "decision_phase": (
            None
            if decision_hypothesis is None
            else decision_hypothesis.phase.value
        ),
        "decision_setup_id": (
            None if decision_hypothesis is None else decision_hypothesis.setup_context_id
        ),
        "decision_episode_id": (
            None if decision_hypothesis is None else decision_hypothesis.episode_id
        ),
        "decision_entry_location_id": (
            None if plan is None else plan.entry_location_id
        ),
        "decision_entry_path_id": (
            None if plan is None else plan.entry_path_id
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
        "observation_anomalies": json.dumps(
            list(snapshot.observation.anomalies),
            ensure_ascii=False,
        ),
        "planned_entry": None if plan is None else plan.planned_entry,
        "entry_zone_lower": None if plan is None else plan.entry_zone_lower,
        "entry_zone_upper": None if plan is None else plan.entry_zone_upper,
        "invalidation": None if plan is None else plan.invalidation.price,
        "invalidation_source_id": (
            None if plan is None else plan.invalidation.source_level_id
        ),
        "primary_target": None if plan is None else plan.targets[0].price,
        "primary_target_id": None if plan is None else plan.targets[0].level_id,
        "selected_draw_id": None if plan is None else plan.selected_draw_id,
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
        "primary_target_R": None if plan is None else plan.primary_target_R,
        "remaining_path_R": None if plan is None else plan.remaining_path_R,
        "deadline": None if plan is None else plan.deadline,
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
        "risk_reasons": json.dumps(
            list(snapshot.risk.reasons),
            ensure_ascii=False,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="outputs/replay")
    parser.add_argument(
        "--config",
        default="configs/model.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/data_splits.json",
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
            "development"
        ),
    )
    parser.add_argument(
        "--mbo-execution",
        help="pre-materialized causal MBO minute execution-reality parquet",
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
        "--brain-calibration",
        action="store_true",
        help=(
            "emit the additional Brain calibration stream for a registered "
            "calibration window"
        ),
    )
    parser.add_argument(
        "--visualize-at",
        action="append",
        default=[],
        metavar="AWARE_TIMESTAMP",
        help=(
            "render one completed-data decision view at this exact decision "
            "clock; repeat for at most 40 preselected clocks"
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


def _visualization_clocks(
    values: list[str] | tuple[str, ...],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, ...]:
    if len(values) > 40:
        raise ValueError("at most 40 decision clocks may be visualized")
    clocks: list[pd.Timestamp] = []
    for raw in values:
        clock = pd.Timestamp(raw)
        if clock.tzinfo is None:
            raise ValueError("visualization decision clocks must be timezone-aware")
        clock = clock.tz_convert("UTC")
        if not start.tz_convert("UTC") <= clock < end.tz_convert("UTC"):
            raise ValueError(
                "visualization decision clocks must lie inside the replay interval"
            )
        clocks.append(clock)
    if len(clocks) != len(set(clocks)):
        raise ValueError("visualization decision clocks must be unique")
    return tuple(sorted(clocks))


def _visualization_key(value: pd.Timestamp) -> str:
    clock = pd.Timestamp(value)
    if clock.tzinfo is None:
        raise ValueError("visualization decision clock must be timezone-aware")
    return clock.tz_convert("UTC").isoformat()


def _visualization_filename(value: pd.Timestamp) -> str:
    clock = pd.Timestamp(value)
    if clock.tzinfo is None:
        raise ValueError("visualization decision clock must be timezone-aware")
    return clock.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ.png")


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
    record: BrainCalibrationRecord,
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
    selected = snapshot.belief.hypotheses.get(
        best_variant.hypothesis_key or ""
    )
    entry_attempt_id = "|".join(
        (
            str(thesis.setup_id or ""),
            str(thesis.entry_location_id or ""),
            str(thesis.entry_path_id or ""),
            snapshot.observation.asof.isoformat(),
        )
    )
    return {
        "thesis_hash": thesis.thesis_hash,
        "entry_attempt_id": entry_attempt_id,
        "decision_time": snapshot.observation.asof,
        "playbook": thesis.playbook.value,
        "direction": thesis.direction.value,
        "setup_id": thesis.setup_id,
        "episode_id": None if selected is None else selected.episode_id,
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
        "best_variant_components": json.dumps(
            dict(best_variant.components),
            sort_keys=True,
        ),
    }


def _streamed_main(args: argparse.Namespace) -> None:
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
    visualization_clocks = _visualization_clocks(
        args.visualize_at,
        start=start,
        end=end,
    )
    visualization_clock_keys = frozenset(
        _visualization_key(clock) for clock in visualization_clocks
    )
    visualization_enabled = bool(visualization_clocks)

    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_ohlcv(start, end)
    if window.role == "sealed_holdout" and not args.reveal_sealed_holdout:
        raise RuntimeError(
            "requested interval is the sealed OHLCV holdout; reveal is reserved "
            "for a frozen final-validation candidate"
        )
    source = Path(args.source)
    source_hash = sha256_file(source)
    if (
        source_hash != validation.causal_source.sha256
        and not args.acknowledge_research_roll_lineage
    ):
        raise RuntimeError(
            "OHLCV source hash does not match the registered causal front"
        )

    load_start = _calendar_warmup_start(
        start,
        days=args.warmup_days,
    )
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
    engine = ContinuousSMCEngine.from_config(
        config_source,
        runtime_mode="development",
    )
    config_identity = sha256_file(config_source)
    brain_calibration_enabled = bool(args.brain_calibration)
    if brain_calibration_enabled and window.role not in {
        "calibration",
        "belief_calibration",
    }:
        raise RuntimeError(
            "--brain-calibration requires a registered calibration window"
        )
    if args.mbo_execution:
        execution_store, mbo_manifest = _load_mbo_execution(
            args.mbo_execution,
            validation=validation,
            start=start,
            end=end,
            reveal_sealed_holdout=args.reveal_sealed_mbo_holdout,
        )
        execution_mode = "mbo_causal_execution"
    else:
        execution_store = None
        mbo_manifest = None
        execution_mode = (
            "ohlcv_only_execution_unavailable"
            if args.spread_points is None
            else "constant_cli_research_only"
        )

    destination = Path(args.output)
    completed_path = destination / "COMPLETED.json"
    progress_path = destination / "progress.json"
    run_manifest_path = destination / "run_manifest.json"
    if args.resume:
        if completed_path.exists():
            raise FileExistsError("development replay is already complete")
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "refusing non-resume use of a non-empty development output"
        )
    destination.mkdir(parents=True, exist_ok=True)

    stream_keys = dict(BASE_STREAM_KEYS)
    if brain_calibration_enabled:
        stream_keys["brain_calibration_shards"] = "sample_id"

    run_manifest = {
        "schema_version": 1,
        "runner": "continuous_replay",
        "source": {
            "path": str(source.resolve()),
            "sha256": source_hash,
            "rows": total_source_rows,
            "first": replay_frame.index[0].isoformat(),
            "last": replay_frame.index[-1].isoformat(),
            "role": loaded.source_role,
        },
        "model_config": {
            "path": str(config_source.resolve()),
            "identity": config_identity,
            "schema_version": config_payload.get("schema_version"),
        },
        "window": {
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "role": window.role,
            "warmup_days": int(args.warmup_days),
        },
        "execution": {
            "simulate": bool(args.simulate_execution),
            "mode": execution_mode,
            "constant_spread_points": args.spread_points,
            "constant_slippage_points": streamed_slippage_points,
            "mbo_execution": (
                None
                if args.mbo_execution is None
                else str(Path(args.mbo_execution).resolve())
            ),
            "mbo_source_sha256": (
                None
                if mbo_manifest is None
                else str(mbo_manifest["output_sha256"])
            ),
        },
        "output": {
            "brain_calibration": brain_calibration_enabled,
            "brain_calibration_schema_version": (
                BRAIN_CALIBRATION_RECORDER_SCHEMA_VERSION
                if brain_calibration_enabled
                else None
            ),
            "shard_rows": effective_shard_rows,
            "checkpoint_bars": int(args.checkpoint_bars),
            "stream_families": sorted(stream_keys),
            "visualization": {
                "enabled": visualization_enabled,
                "decision_clocks_utc": sorted(visualization_clock_keys),
                "directory": (
                    "visualizations" if visualization_enabled else None
                ),
                "selection": "explicit_decision_clocks",
            },
        },
    }
    run_manifest_bytes = canonical_json(to_primitive(run_manifest))
    if args.resume:
        if not run_manifest_path.is_file():
            raise FileNotFoundError("resume requires run_manifest.json")
        if run_manifest_path.read_bytes() != run_manifest_bytes:
            raise ValueError("run manifest differs from the requested replay")
    else:
        atomic_bytes(run_manifest_path, run_manifest_bytes)
    bindings = {"run_manifest": run_manifest_path.name}

    checkpoint = ReplayCheckpointStore(destination / "_checkpoint")
    total_fields = {
        "decision_shards": "decision_rows",
        "brain_calibration_shards": "brain_calibration_rows",
    }

    def sync_checkpoint_aliases(state: dict[str, Any]) -> None:
        decision_stream = state["streams"]["decision_shards"]
        state["next_shard_index"] = int(
            decision_stream["next_shard_index"]
        )
        state["committed_shards"] = decision_stream["committed_shards"]

    def verify_state_streams(state: dict[str, Any]) -> None:
        if set(state.get("streams", {})) != set(stream_keys):
            raise ValueError("checkpoint stream family changed")
        if set(state.get("buffers", {})) != set(stream_keys):
            raise ValueError("checkpoint buffer family changed")
        for name in stream_keys:
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
            for name in stream_keys
        }
        state = {
            "replay": CalibrationSequentialReplay(
                engine=engine,
                simulate_execution=bool(args.simulate_execution),
            ),
            "brain_calibration": (
                BrainCalibrationRecorder()
                if brain_calibration_enabled
                else None
            ),
            "streams": streams,
            "buffers": {name: [] for name in stream_keys},
            "processed_bars": 0,
            "source_rows_consumed": 0,
            "last_checkpoint_processed_bars": 0,
            "decision_rows": 0,
            "brain_calibration_rows": 0,
            "model_action_counts": {},
            "risk_action_counts": {},
            "entry_approvals": {},
            "filled_entries": {},
            "last_source_start": None,
            "last_asof": None,
            "resume_count": 0,
            "finalized": False,
            "visual_artifacts": {},
            "peak_buffer_rows": {name: 0 for name in stream_keys},
            "next_shard_index": 0,
            "committed_shards": streams["decision_shards"][
                "committed_shards"
            ],
        }
    args._streamed_output_owned = True

    replay: CalibrationSequentialReplay = state["replay"]
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

    visual_artifacts = state.get("visual_artifacts")
    if not isinstance(visual_artifacts, dict):
        raise ValueError("checkpoint visualization state is invalid")
    if not set(visual_artifacts).issubset(visualization_clock_keys):
        raise ValueError("checkpoint contains an unrequested visualization")
    for clock_key, artifact in visual_artifacts.items():
        if not isinstance(artifact, VisualArtifact):
            raise ValueError("checkpoint visual artifact is invalid")
        if artifact.path.is_symlink() or not artifact.path.is_file():
            raise FileNotFoundError(
                f"checkpoint visual artifact is missing: {artifact.path}"
            )
        if _visualization_key(artifact.maximum_market_time) > clock_key:
            raise ValueError("visual artifact contains future market data")

    visualizer = DecisionVisualizer() if visualization_enabled else None
    visualization_directory = (
        destination / "visualizations" if visualization_enabled else None
    )
    if visualization_directory is not None:
        visualization_directory.mkdir(parents=True, exist_ok=True)

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
                key_column=stream_keys[name],
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
        for name in stream_keys:
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

    def render_requested_visual(snapshot) -> None:
        if visualizer is None or visualization_directory is None:
            return
        clock_key = _visualization_key(snapshot.observation.asof)
        if (
            clock_key not in visualization_clock_keys
            or clock_key in visual_artifacts
        ):
            return
        histories = replay.engine.histories(
            bars=max(DecisionVisualizer.PANEL_BARS.values())
        )
        destination_path = (
            visualization_directory
            / _visualization_filename(snapshot.observation.asof)
        )
        temporary_path = destination_path.with_name(
            f".{destination_path.stem}.tmp.png"
        )
        try:
            artifact = visualizer.render_decision(
                snapshot,
                histories,
                temporary_path,
            )
            if artifact.maximum_market_time > snapshot.observation.asof:
                raise AssertionError("visual artifact revealed future market data")
            os.replace(temporary_path, destination_path)
        except Exception:
            if temporary_path.is_file() and not temporary_path.is_symlink():
                temporary_path.unlink()
            raise
        artifact = replace(artifact, path=destination_path)
        visual_artifacts[clock_key] = artifact

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
                if brain_calibration is not None:
                    brain_calibration.on_bar(bar)
                    append_stream(
                        "brain_calibration_shards",
                        [
                            _stream_record(item)
                            for item in brain_calibration.drain_rows()
                        ],
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
                    render_requested_visual(snapshot)
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
                    append_stream(
                        "decision_shards",
                        [
                            _row(
                                snapshot,
                                account_state=step.account_state,
                                belief_position_input=(
                                    step.belief_position_input
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
                        for name in stream_keys
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
    missing_visualizations = sorted(
        visualization_clock_keys.difference(visual_artifacts)
    )
    if missing_visualizations:
        raise ValueError(
            "requested visualization decision clocks were not observed: "
            + ", ".join(missing_visualizations)
        )
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
        state["finalized"] = True
    commit_checkpoint(final=True)
    verify_state_streams(state)

    stream_manifests: dict[str, str] = {}
    for name in stream_keys:
        manifest_path = write_stream_manifest(
            destination,
            name,
            state["streams"][name],
            artifact=f"continuous_development_{name}",
            bindings=bindings,
        )
        stream_manifests[name] = str(manifest_path.relative_to(destination))

    visualization_index: str | None = None
    if visualization_directory is not None:
        index_path = visualization_directory / "index.html"
        temporary_index_path = visualization_directory / ".index.tmp.html"
        DecisionVisualizer.build_index(
            tuple(
                visual_artifacts[key]
                for key in sorted(visual_artifacts)
            ),
            temporary_index_path,
        )
        os.replace(temporary_index_path, index_path)
        visualization_index = str(index_path.relative_to(destination))

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

    summary = {
        "schema_version": config_payload.get("schema_version"),
        "source": str(loaded.source),
        "source_matches_preregistered_causal_front": (
            source_hash == validation.causal_source.sha256
        ),
        "source_role": loaded.source_role,
        "validation_schema_version": validation.schema_version,
        "validation_window_role": window.role,
        "execution_reality_source": execution_mode,
        "execution_authority": bool(execution_store is not None),
        "contract_selection_causal": loaded.contract_selection_causal,
        "warnings": list(loaded.warnings),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "decision_clock_interval": "[start, end)",
        "decision_rows": int(state["decision_rows"]),
        "brain_calibration_capture": brain_calibration_enabled,
        "brain_calibration_rows": int(
            state["brain_calibration_rows"]
        ),
        "model_action_counts": dict(
            sorted(state["model_action_counts"].items())
        ),
        "action_counts": dict(sorted(state["risk_action_counts"].items())),
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
        "checkpoint_resume_supported": True,
        "output_contract": "lightweight_shards",
        "resume_count": int(state["resume_count"]),
        "shard_rows": effective_shard_rows,
        "checkpoint_bars": int(args.checkpoint_bars),
        "stream_rows": {
            name: int(state["streams"][name]["rows"])
            for name in stream_keys
        },
        "peak_buffer_rows": dict(state["peak_buffer_rows"]),
        "visualization_capture": visualization_enabled,
        "visualization_decisions": len(visual_artifacts),
        "visualization_index": visualization_index,
        "future_path_visible_to_model": False,
        "future_path_output": False,
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
    atomic_bytes(
        completed_path,
        canonical_json(
            {
                "schema_version": 1,
                "status": "complete",
                "run_manifest": run_manifest_path.name,
                "summary": summary_path.name,
                "progress": progress_path.name,
                "stream_manifests": stream_manifests,
                "trades": "trades.parquet" if args.simulate_execution else None,
                "entry_attempts": (
                    "entry_attempts.parquet" if args.simulate_execution else None
                ),
                "visualizations_index": visualization_index,
            }
        ),
    )
    print(json.dumps(summary["action_counts"], sort_keys=True))


def main() -> None:
    args = parse_args()
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
