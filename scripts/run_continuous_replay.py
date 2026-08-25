#!/usr/bin/env python3
"""Run the causal model with bounded, resumable development outputs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping

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
from smc_trader.brain_entry_sequence import (  # noqa: E402
    brain_observation_view,
)
from smc_trader.calibration_replay import (  # noqa: E402
    CalibrationSequentialReplay,
    ReplayCheckpointStore,
    iter_after_source_checkpoint,
)
from smc_trader.causal_cases import (  # noqa: E402
    CAUSAL_CASE_INPUT_FIELD_TYPES,
    CAUSAL_CASE_OUTCOME_FIELD_TYPES,
    CAUSAL_CASE_RECORDER_SCHEMA_VERSION,
    CausalCaseInputRecord,
    CausalCaseOutcomeRecord,
    CausalCaseRecorder,
    expected_causal_case_run_identity,
    write_causal_case_library_manifest,
)
from smc_trader.market_cases import (  # noqa: E402
    MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY,
    MARKET_CASE_INPUT_FIELD_TYPES,
    MARKET_CASE_PROTOCOL,
    MARKET_CASE_RECORDER_SCHEMA_VERSION,
    MarketCaseInputRecord,
    MarketEpisodeCaseRecorder,
    expected_market_case_run_identity,
)
from smc_trader.brain_calibration import (  # noqa: E402
    BrainCalibrationRecord,
    BrainCalibrationRecorder,
    RECORDER_SCHEMA_VERSION as BRAIN_CALIBRATION_RECORDER_SCHEMA_VERSION,
    SUPPORTED_PLAYBOOKS as BRAIN_CALIBRATION_PLAYBOOKS,
)
from smc_trader.engine import (  # noqa: E402
    ContinuousSMCEngine,
    normalize_action_disabled_playbooks,
)
from smc_trader.io import load_ohlcv  # noqa: E402
from smc_trader.model import (  # noqa: E402
    Action,
    Direction,
    NeutralEngineSnapshot,
    Playbook,
    Timeframe,
    to_primitive,
)
from smc_trader.mbo import (  # noqa: E402
    MinuteExecutionRealityStore,
    assert_mbo_source_allowed,
)
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.playbooks import (  # noqa: E402
    _open_thesis_supports_playbook,
)
from smc_trader.shadow_outcome import (  # noqa: E402
    RECORDER_SCHEMA_VERSION as SHADOW_OUTCOME_RECORDER_SCHEMA_VERSION,
    SHADOW_DERIVED_SCHEMA_VERSION,
    SHADOW_EPISODE_OUTCOME_FIELD_TYPES,
    SHADOW_MECHANISM_CHALLENGE_FIELD_TYPES,
    SHADOW_MECHANISM_MOTIF_FIELD_TYPES,
    SHADOW_MOTIF_ROOT_SAMPLE_LIMIT,
    SHADOW_OUTCOME_FIELD_TYPES,
    SHADOW_OUTCOME_PROTOCOL,
    SHADOW_ROOT_EPISODE_FIELD_TYPES,
    SHADOW_ROOT_SEQUENCE_FIELD_TYPES,
    ShadowEpisodeOutcomeRecord,
    ShadowMechanismChallengeRecord,
    ShadowMechanismMotifRecord,
    ShadowRootSequenceRecord,
    ShadowCandidateOutcomeRecord,
    ShadowCandidateOutcomeRecorder,
    derive_shadow_episode_outcomes,
    derive_shadow_mechanism_challenges,
    derive_shadow_root_episode_records,
    derive_shadow_root_sequence_records,
)
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

# Versioned pickled legacy Brain runtime-state contract.
BRAIN_RUNTIME_STATE_SCHEMA_VERSION = 15
# The input-only MarketEpisode mode has a disjoint, deliberately small state
# shape and therefore owns an independent resume schema.
MARKET_CASE_INPUT_RUNTIME_STATE_SCHEMA_VERSION = 8
MARKET_CASE_PROFILE_REGISTRY_SCHEMA_VERSION = 1
DEFAULT_MARKET_CASE_PROFILE_REGISTRY = (
    ROOT / "configs/market_case_input_profiles_v2.json"
)

# Versioned diagnostic-state contract.  Natural episode lifecycles are
# intentionally broader than action candidates: a dormant episode cannot
# authorize an entry, but its explicit terminal update must still be counted.
# Persist this separately from the Brain runtime schema so an action-only
# funnel checkpoint cannot be resumed into the lifecycle-aware reducer.
NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION = 3
NATURAL_FUNNEL_FIELD_TYPES = {
    "record_type": "large_string",
    "record_id": "large_string",
    "payload_json": "large_string",
}
NATURAL_FUNNEL_SCHEMA_FINGERPRINT = hashlib.sha256(
    json.dumps(
        NATURAL_FUNNEL_FIELD_TYPES,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _repository_commit_identity() -> dict[str, str]:
    """Read the exact repository HEAD used by a new neutral input run."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "repository HEAD commit identity is unavailable"
        ) from exc
    stdout = getattr(result, "stdout", None)
    commit = stdout.strip() if isinstance(stdout, str) else ""
    if (
        getattr(result, "returncode", None) != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RuntimeError("repository HEAD commit identity is invalid")
    return {"commit": commit}

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
    "top_context_metadata": "large_string",
    "top_competing_episode_ids": "large_string",
    "market_thesis_id": "large_string",
    "bound_market_thesis_id": "large_string",
    "market_thesis_root_id": "large_string",
    "market_thesis_mechanism": "large_string",
    "market_thesis_authority_relation": "large_string",
    "playbook_match_strength": "float64",
    "market_thesis_binding_required": "bool",
    "market_thesis_action_bound": "bool",
    "market_thesis_match_status": "large_string",
    "playbook_first_failed_hard_gate_id": "large_string",
    "playbook_plan_delivery_valid": "bool",
    "global_market_mode": "large_string",
    "global_authority_timeframe": "large_string",
    "global_authority_direction": "large_string",
    "global_authority_source_ids": "large_string",
    "global_dislocated": "bool",
    "global_scale_relations": "large_string",
    "global_path_blocker_count": "int64",
    "global_nearest_path_blocker_id": "large_string",
    "global_key_path_blocker_ids": "large_string",
    "global_material_conflict_count": "int64",
    "global_key_material_conflict_ids": "large_string",
    "global_unexplained_episode_count": "int64",
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
    "authority_barrier_id": "large_string",
    "authority_barrier_price": "float64",
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


EPISODE_FUNNEL_STAGES = (
    "candidate_root",
    "forming",
    "armed",
    "waiting_location",
    "waiting_trigger",
    "executable",
    "decision_enter",
    "risk_pass",
    "order_filled",
    "position_terminal",
)
_EPISODE_PHASE_PREFIX = {
    "forming": EPISODE_FUNNEL_STAGES[:2],
    "armed": EPISODE_FUNNEL_STAGES[:3],
    "waiting_location": EPISODE_FUNNEL_STAGES[:4],
    "waiting_trigger": EPISODE_FUNNEL_STAGES[:5],
    "executable": EPISODE_FUNNEL_STAGES[:6],
    "entered": EPISODE_FUNNEL_STAGES[:6],
    "delivering": EPISODE_FUNNEL_STAGES[:6],
}
_EPISODE_FUNNEL_STRATA = (
    "playbook",
    "direction",
    "source_timeframe",
    "source_tier",
    "authority_relation",
    "market_mode",
)

OPEN_THESIS_BINDING_STAGES = (
    "open_thesis_created",
    "mechanism_direction_matched",
    "exact_root_bound",
    "causal_gates_complete",
    "plan_delivery_valid",
    "executable",
)
_OPEN_THESIS_BINDING_BITS = {
    stage: 1 << index
    for index, stage in enumerate(OPEN_THESIS_BINDING_STAGES)
}
_OPEN_THESIS_FUNNEL_STRATA = (
    "playbook",
    "direction",
    "thesis_direction",
    "mechanism",
    "authority_relation",
)

BRAIN_CALIBRATION_FIELD_TYPES = {
    "sample_id": "large_string",
    "hypothesis_key": "large_string",
    "playbook": "large_string",
    "direction": "large_string",
    "dimension": "large_string",
    "setup_id": "large_string",
    "calibration_unit_id": "large_string",
    "calibration_unit_kind": "large_string",
    "episode_id": "large_string",
    "context_id": "large_string",
    "context_thesis_id": "large_string",
    "parent_context_thesis_id": "large_string",
    "evidence_revision_id": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "selected_trigger_id": "large_string",
    "selected_trigger_kind": "large_string",
    "selected_trigger_at": "timestamp_ny",
    "available_trigger_kinds": "large_string",
    "market_thesis_id": "large_string",
    "bound_market_thesis_id": "large_string",
    "market_thesis_root_id": "large_string",
    "market_thesis_mechanism": "large_string",
    "market_thesis_authority_relation": "large_string",
    "playbook_match_strength": "float64",
    "market_thesis_binding_required": "bool",
    "market_thesis_action_bound": "bool",
    "market_thesis_match_status": "large_string",
    "playbook_first_failed_hard_gate_id": "large_string",
    "playbook_plan_delivery_valid": "bool",
    "global_market_mode": "large_string",
    "authority_relation": "large_string",
    "authority_rank_gap": "int64",
    "conflict_role": "large_string",
    "conflict_scope": "large_string",
    "acceptance_state": "large_string",
    "obstruction_distance_R": "float64",
    "free_path_R": "float64",
    "soft_obstruction_count": "int64",
    "hard_barrier_before_target": "bool",
    "ambiguity_count": "int64",
    "uncertainty_conflict": "float64",
    "uncertainty_required_evidence_missing": "float64",
    "uncertainty_authority_missing": "float64",
    "uncertainty_graph_ambiguity": "float64",
    "uncertainty_total": "float64",
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
    "target_deadline_kind": "large_string",
    "deadline": "timestamp_ny",
    "symbol": "large_string",
    "instrument_id": "int64",
    "authority_barrier_id": "large_string",
    "authority_barrier_price": "float64",
}

STREAM_FIELD_TYPES = {
    "decision_shards": DECISION_FIELD_TYPES,
    "brain_calibration_shards": BRAIN_CALIBRATION_FIELD_TYPES,
    "shadow_outcome_shards": SHADOW_OUTCOME_FIELD_TYPES,
    "causal_case_input_shards": CAUSAL_CASE_INPUT_FIELD_TYPES,
    "causal_case_outcome_shards": CAUSAL_CASE_OUTCOME_FIELD_TYPES,
    "market_case_input_shards": MARKET_CASE_INPUT_FIELD_TYPES,
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
    observation = brain_observation_view(observation)
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


def _compact_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _action_candidate_items(belief: Any) -> tuple[tuple[str, Any], ...]:
    """Return root-specific action candidates without changing summaries.

    Real ``MarketBelief`` instances own the production fallback contract in
    ``action_candidate_items()``.  The mapping fallback only keeps the small
    replay-statistic unit-test doubles and graph-free test mode usable.
    """

    provider = getattr(belief, "action_candidate_items", None)
    if callable(provider):
        return tuple(provider())
    hypotheses = getattr(belief, "hypotheses", {})
    return tuple(hypotheses.items())


def _lifecycle_candidate_items(belief: Any) -> tuple[tuple[str, Any], ...]:
    """Return every candidate whose causal lifecycle remains observable.

    Production ``MarketBelief`` keeps dormant entry episodes outside the
    action map so they cannot authorize a new entry.  Natural-funnel
    diagnostics must nevertheless observe their explicit terminal update;
    otherwise a root disappearing from the open set looks like a forming
    episode that survived until the replay boundary.  The fallback preserves
    compatibility with the small graph-free replay test doubles.
    """

    provider = getattr(belief, "lifecycle_candidate_items", None)
    if callable(provider):
        return tuple(provider())
    return _action_candidate_items(belief)


def _resolve_hypothesis(belief: Any, identity: str | None) -> Any | None:
    """Resolve a Decision identity across root candidates and summaries."""

    resolver = getattr(belief, "resolve_hypothesis", None)
    if callable(resolver):
        return resolver(identity)
    if identity is None:
        return None
    return getattr(belief, "hypotheses", {}).get(identity)


def _lsr_reach_key(values: tuple[str, ...]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _enum_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    return text or default


def _episode_key(hypothesis: Any) -> str | None:
    episode_id = getattr(hypothesis, "episode_id", None)
    setup_id = getattr(hypothesis, "setup_context_id", None)
    if episode_id is None and setup_id is None:
        return None
    identity = (
        str(episode_id)
        if episode_id is not None
        else f"provisional:{setup_id}"
    )
    return _lsr_reach_key(
        (
            hypothesis.playbook.value,
            hypothesis.direction.value,
            identity,
        )
    )


def _optional_diagnostic_identity(value: Any) -> str | None:
    """Normalize optional identities without turning sentinels into owners."""

    if value is None:
        return None
    identity = str(value).strip()
    if not identity or identity.lower() in {"none", "unknown"}:
        return None
    return identity


def _episode_identity_diagnostics(
    hypothesis: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Project exact Context/zone identities for offline lifecycle census."""

    playbook = hypothesis.playbook.value
    lsr_zone_id = _optional_diagnostic_identity(
        metadata.get("lsr_entry_zone_id")
    )
    return {
        "context_thesis_id": _optional_diagnostic_identity(
            getattr(hypothesis, "context_thesis_id", None)
        ),
        "parent_context_thesis_id": _optional_diagnostic_identity(
            getattr(hypothesis, "parent_context_thesis_id", None)
        ),
        "initiating_event_id": _optional_diagnostic_identity(
            getattr(hypothesis, "initiating_event_id", None)
        ),
        "market_thesis_root_id": _optional_diagnostic_identity(
            getattr(hypothesis, "market_thesis_root_id", None)
        ),
        "entry_location_id": _optional_diagnostic_identity(
            getattr(hypothesis, "entry_location_id", None)
        ),
        "entry_path_id": _optional_diagnostic_identity(
            getattr(hypothesis, "entry_path_id", None)
        ),
        "lsr_manipulation_id": _optional_diagnostic_identity(
            metadata.get("lsr_manipulation_id")
        ),
        "lsr_pool_path_id": _optional_diagnostic_identity(
            metadata.get("lsr_pool_path_id")
        ),
        "lsr_displacement_id": _optional_diagnostic_identity(
            metadata.get("lsr_displacement_id")
        ),
        "lsr_entry_zone_id": lsr_zone_id,
        "eligible_entry_zone": bool(
            playbook == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
            and lsr_zone_id is not None
            and getattr(hypothesis, "episode_id", None) is not None
        ),
    }


def _update_episode_causal_diagnostics(
    record: dict[str, Any],
    hypothesis: Any,
    *,
    asof: pd.Timestamp,
) -> None:
    """Freeze first zone/pullback/trigger/plan clocks without future data."""

    metadata = dict(getattr(hypothesis, "context_metadata", {}) or {})
    record.update(_episode_identity_diagnostics(hypothesis, metadata))
    if record.get("eligible_entry_zone"):
        record.setdefault("eligible_zone_first_observed_at", asof.isoformat())
    trigger = getattr(hypothesis, "selected_trigger", None)
    if trigger is not None:
        record.setdefault("first_trigger_observed_at", asof.isoformat())
        record.setdefault("selected_trigger_id", str(trigger.trigger_id))
        record.setdefault("selected_trigger_kind", str(trigger.trigger_kind))
        record.setdefault(
            "selected_trigger_at",
            pd.Timestamp(trigger.observed_at).isoformat(),
        )
    feasibility = getattr(hypothesis, "plan_feasibility", None)
    if feasibility is not None and bool(getattr(feasibility, "valid", False)):
        record.setdefault("first_plan_valid_at", asof.isoformat())
        record["plan_delivery_valid"] = True
    else:
        record.setdefault("plan_delivery_valid", False)


def _provisional_index_key(
    hypothesis: Any,
    identity: str,
) -> str:
    return _lsr_reach_key(
        (
            hypothesis.playbook.value,
            hypothesis.direction.value,
            identity,
        )
    )


def _episode_record(
    state: dict[str, Any],
    hypothesis: Any,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve a provisional setup record into its later typed episode."""

    key = _episode_key(hypothesis)
    if key is None:
        raise ValueError("episode record requires a setup or episode identity")
    records = state.setdefault("natural_episode_funnel_records", {})
    provisional = state.setdefault("natural_provisional_episode_keys", {})
    aliases = state.setdefault("natural_episode_key_aliases", {})
    if not all(isinstance(value, dict) for value in (records, provisional, aliases)):
        raise ValueError("checkpoint natural episode identity state is invalid")
    record = records.get(key)
    episode_id = getattr(hypothesis, "episode_id", None)
    if record is None and episode_id is not None:
        prior_key = next(
            (
                provisional.get(_provisional_index_key(hypothesis, str(identity)))
                for identity in (
                    getattr(hypothesis, "context_id", None),
                    getattr(hypothesis, "setup_context_id", None),
                )
                if identity is not None
                and provisional.get(
                    _provisional_index_key(hypothesis, str(identity))
                )
                in records
            ),
            None,
        )
        if prior_key is not None and prior_key != key:
            record = records.pop(prior_key)
            records[key] = record
            aliases[prior_key] = key
            for index_key, value in tuple(provisional.items()):
                if value == prior_key:
                    provisional.pop(index_key)
    return key, record


def _hypothesis_source_ids(hypothesis: Any) -> frozenset[str]:
    """Return the compact identity set that can bind a candidate root.

    This mirrors the Brain's public causal identities without copying a
    Scene Graph snapshot into the replay statistics layer.
    """

    # Competing episode IDs are deliberately excluded: being mentioned as a
    # rival does not mean the hypothesis causally owns or binds that root.
    values: list[str | None] = [
        getattr(hypothesis, "required_root_id", None),
        getattr(hypothesis, "market_thesis_root_id", None),
        getattr(hypothesis, "episode_id", None),
        getattr(hypothesis, "setup_context_id", None),
        getattr(hypothesis, "context_id", None),
        getattr(hypothesis, "initiating_event_id", None),
        getattr(hypothesis, "entry_location_id", None),
    ]
    sequence = getattr(hypothesis, "sequence", None)
    if sequence is not None:
        values.extend(
            source_id
            for step in getattr(sequence, "steps", ())
            for source_id in getattr(step, "source_ids", ())
        )
    return frozenset(str(value) for value in values if value)


def _update_candidate_root_cohort(
    state: dict[str, Any],
    snapshot: Any,
) -> None:
    """Record the true pre-hypothesis denominator and causal episode links."""

    records = state.setdefault("natural_candidate_root_records", {})
    if not isinstance(records, dict):
        raise ValueError("checkpoint candidate-root cohort state is invalid")
    context = getattr(snapshot.belief, "global_context", None)
    root_ids = tuple(
        dict.fromkeys(
            getattr(context, "candidate_structured_episode_ids", ())
            if context is not None
            else ()
        )
    )
    asof = pd.Timestamp(snapshot.observation.asof)
    for root_id in root_ids:
        root_id = str(root_id)
        descriptor = _root_descriptor(snapshot.observation, root_id)
        record = records.get(root_id)
        if record is None:
            record = {
                "candidate_root_id": root_id,
                "market_epoch_id": str(
                    getattr(context, "market_epoch_id", "unknown")
                ),
                "first_observed_at": asof.isoformat(),
                "last_observed_at": asof.isoformat(),
                "source_timeframe": descriptor["timeframe"],
                "source_kind": descriptor["source_kind"],
                "direction": descriptor["direction"],
                "structural_scale": descriptor["structural_scale"],
                "bos_scope": descriptor.get("bos_scope"),
                "nearest_playbook": descriptor["nearest_playbook"],
                "source_tier": str(descriptor.get("source_tier", "unknown")),
                "authority_relation": "unbound",
                "market_mode": _enum_text(
                    getattr(context, "market_mode", None)
                ),
                "linked_episode_keys": [],
                "first_linked_at": None,
            }
            records[root_id] = record
        else:
            record["last_observed_at"] = asof.isoformat()

    current_epoch = str(getattr(context, "market_epoch_id", "unknown"))
    hypotheses = tuple(
        hypothesis
        for _, hypothesis in _action_candidate_items(snapshot.belief)
    )
    eligible_by_root: dict[str, list[tuple[int, str, Any]]] = {}
    for hypothesis in hypotheses:
        episode_key = _episode_key(hypothesis)
        if episode_key is None:
            continue
        for source_id in _hypothesis_source_ids(hypothesis):
            record = records.get(source_id)
            if (
                not isinstance(record, dict)
                or record.get("linked_episode_keys")
                or record.get("market_epoch_id") != current_epoch
            ):
                continue
            expected_direction = record.get("direction")
            if expected_direction not in {None, "unknown"} and (
                hypothesis.direction.value != expected_direction
            ):
                continue
            expected_playbook = record.get("nearest_playbook")
            initiating_exact = int(
                getattr(hypothesis, "initiating_event_id", None) == source_id
            )
            playbook_match = int(
                expected_playbook in {None, hypothesis.playbook.value}
            )
            eligible_by_root.setdefault(source_id, []).append(
                (initiating_exact * 2 + playbook_match, episode_key, hypothesis)
            )
    for root_id, eligible in eligible_by_root.items():
        _, episode_key, hypothesis = max(
            eligible,
            key=lambda value: (value[0], value[1]),
        )
        record = records[root_id]
        record["linked_episode_keys"] = [episode_key]
        record["first_linked_at"] = asof.isoformat()
        metadata = dict(getattr(hypothesis, "context_metadata", {}) or {})
        record["authority_relation"] = str(
            metadata.get("authority_relation", "unrelated")
        )
        record["source_tier"] = str(
            metadata.get(
                "source_tier",
                metadata.get(
                    "manipulation_tier",
                    record.get("source_tier", "unknown"),
                ),
            )
        )


def _record_episode_stage(
    record: dict[str, Any],
    stage: str,
    asof: pd.Timestamp,
) -> None:
    first = record.setdefault("first_reached_at", {})
    if stage in first:
        return
    clock = pd.Timestamp(asof).isoformat()
    first[stage] = clock
    record.setdefault("first_reach_sequence", []).append(
        {"stage": stage, "at": clock}
    )


def _new_compact_natural_funnel_state() -> dict[str, Any]:
    """Return bounded-detail state for aggregate Brain diagnostics.

    Identity sets and one integer mask per episode are de-duplication state,
    not diagnostic rows.  Full clocks, durations and causal payloads remain
    exclusive to ``--include-natural-funnel-details``.  Default replay does
    not construct this state unless a Brain diagnostic mode is enabled.
    """

    return {
        # Identity-bearing state is intentionally limited to the current
        # live cohort.  Completed roots/episodes are folded into the counters
        # below instead of remaining in a year-long de-duplication set.
        "active_candidate_root_ids": set(),
        "active_linked_candidate_root_ids": set(),
        "pending_candidate_strata": {},
        "candidate_roots_observed": 0,
        "candidate_roots_linked": 0,
        "candidate_root_counts": {},
        "episode_masks": {},
        "episode_metadata": {},
        "provisional_episode_keys": {},
        "episode_stage_counts": {},
        "terminal_disposition_counts": {},
        "episode_count": 0,
        "candidate_cases": {},
        "episode_cases": {},
    }


def _compact_case_insert(
    cases: dict[str, dict[str, Any]],
    identity: str,
    value: dict[str, Any],
    *,
    limit: int = 20,
) -> None:
    if identity in cases:
        cases[identity].update(value)
    elif len(cases) < limit:
        cases[identity] = value


def _compact_disposition_case_insert(
    cases: dict[str, dict[str, Any]],
    identity: str,
    value: dict[str, Any],
    *,
    limit: int = 40,
) -> None:
    """Keep bounded cases while retaining one example per disposition."""

    if identity in cases:
        cases[identity].update(value)
        return
    if len(cases) < limit:
        cases[identity] = value
        return
    disposition = str(value.get("disposition", "unknown"))
    disposition_counts = Counter(
        str(case.get("disposition", "unknown"))
        for case in cases.values()
    )
    if disposition_counts.get(disposition, 0):
        return
    replaceable = tuple(
        (str(case.get("first_seen_at", "")), str(case_id))
        for case_id, case in cases.items()
        if disposition_counts[
            str(case.get("disposition", "unknown"))
        ]
        > 1
    )
    if not replaceable:
        return
    _, replaced_identity = max(replaceable)
    cases.pop(replaced_identity)
    cases[identity] = value


def _compact_episode_record(
    compact: dict[str, Any],
    hypothesis: Any,
) -> tuple[str, dict[str, Any], bool]:
    """Resolve provisional identities without retaining a full episode row."""

    key = _episode_key(hypothesis)
    if key is None:
        raise ValueError("compact episode requires a setup or episode identity")
    masks = compact["episode_masks"]
    metadata = compact["episode_metadata"]
    provisional = compact["provisional_episode_keys"]
    episode_id = getattr(hypothesis, "episode_id", None)
    migrated = False
    if key not in masks and episode_id is not None:
        prior_key = next(
            (
                provisional.get(_provisional_index_key(hypothesis, str(value)))
                for value in (
                    getattr(hypothesis, "context_id", None),
                    getattr(hypothesis, "setup_context_id", None),
                )
                if value is not None
                and provisional.get(
                    _provisional_index_key(hypothesis, str(value))
                )
                in masks
            ),
            None,
        )
        if prior_key is not None and prior_key != key:
            masks[key] = masks.pop(prior_key)
            metadata[key] = metadata.pop(prior_key)
            for index_key, value in tuple(provisional.items()):
                if value == prior_key:
                    provisional.pop(index_key)
            prior_case = compact["episode_cases"].pop(prior_key, None)
            if prior_case is not None:
                prior_case["record_id"] = key
                compact["episode_cases"][key] = prior_case
            migrated = True
    created = key not in masks
    if created:
        masks[key] = 0
        compact["episode_count"] = int(compact["episode_count"]) + 1
    if episode_id is None:
        for value in (
            getattr(hypothesis, "context_id", None),
            getattr(hypothesis, "setup_context_id", None),
        ):
            if value is not None:
                provisional[_provisional_index_key(hypothesis, str(value))] = key
    return key, metadata.setdefault(key, {}), bool(created and not migrated)


def _update_compact_natural_episode_funnel(
    state: dict[str, Any],
    snapshot: Any,
    *,
    step: Any | None = None,
) -> None:
    """Update aggregate-only state when Brain diagnostics are enabled."""

    compact = state.setdefault(
        "natural_funnel_compact_state",
        _new_compact_natural_funnel_state(),
    )
    if not isinstance(compact, dict):
        raise ValueError("checkpoint compact natural funnel state is invalid")
    required = set(_new_compact_natural_funnel_state())
    if set(compact) != required:
        raise ValueError("checkpoint compact natural funnel contract changed")

    context = getattr(snapshot.belief, "global_context", None)
    market_mode = _enum_text(getattr(context, "market_mode", None))
    asof = pd.Timestamp(snapshot.observation.asof)
    active_roots: set[str] = compact["active_candidate_root_ids"]
    linked_roots: set[str] = compact[
        "active_linked_candidate_root_ids"
    ]
    pending_root_strata: dict[str, str] = compact[
        "pending_candidate_strata"
    ]
    candidate_counts: dict[str, dict[str, int]] = compact[
        "candidate_root_counts"
    ]
    root_ids = tuple(
        dict.fromkeys(
            getattr(context, "candidate_structured_episode_ids", ())
            if context is not None
            else ()
        )
    )
    current_root_ids = {str(value) for value in root_ids}
    # A canonical open-thesis root does not revive after leaving the active
    # context.  Retire its identity while preserving its already-counted
    # cohort disposition.  This is replay-statistics compaction only; the
    # production Scene Graph and EventMemory remain untouched.
    departed_root_ids = set(active_roots.difference(current_root_ids))
    for raw_root_id in root_ids:
        root_id = str(raw_root_id)
        if root_id in active_roots:
            continue
        descriptor = _root_descriptor(snapshot.observation, root_id)
        strata = (
            str(descriptor.get("nearest_playbook") or "unknown"),
            str(descriptor.get("direction") or "unknown"),
            str(descriptor.get("timeframe") or "unknown"),
            str(descriptor.get("source_tier") or "unknown"),
            "unbound",
            market_mode,
        )
        encoded = _lsr_reach_key(strata)
        active_roots.add(root_id)
        compact["candidate_roots_observed"] = (
            int(compact["candidate_roots_observed"]) + 1
        )
        pending_root_strata[root_id] = {
            "encoded": encoded,
            "market_epoch_id": str(
                getattr(context, "market_epoch_id", "unknown")
            ),
            "direction": descriptor.get("direction"),
            "nearest_playbook": descriptor.get("nearest_playbook"),
        }
        counts = candidate_counts.setdefault(
            encoded,
            {"candidate_roots": 0, "linked_to_episode": 0},
        )
        counts["candidate_roots"] += 1
        _compact_case_insert(
            compact["candidate_cases"],
            root_id,
            {
                "record_type": "candidate_root",
                "record_id": root_id,
                "first_observed_at": asof.isoformat(),
                "playbook": strata[0],
                "direction": strata[1],
                "source_timeframe": strata[2],
                "authority_relation": strata[4],
                "market_mode": strata[5],
                "linked_to_episode": False,
            },
        )

    action_candidates = tuple(_action_candidate_items(snapshot.belief))
    lifecycle_candidates = tuple(
        _lifecycle_candidate_items(snapshot.belief)
    )
    eligible_by_root: dict[str, list[tuple[int, str, Any]]] = {}
    current_epoch = str(getattr(context, "market_epoch_id", "unknown"))
    for _, hypothesis in action_candidates:
        episode_key = _episode_key(hypothesis)
        if episode_key is None:
            continue
        for source_id in _hypothesis_source_ids(hypothesis):
            if source_id not in pending_root_strata or source_id in linked_roots:
                continue
            pending = pending_root_strata[source_id]
            if not isinstance(pending, dict):
                raise ValueError("compact candidate-root state is invalid")
            if pending.get("market_epoch_id") != current_epoch:
                continue
            expected_direction = pending.get("direction")
            if expected_direction not in {None, "unknown"} and (
                hypothesis.direction.value != expected_direction
            ):
                continue
            expected_playbook = pending.get("nearest_playbook")
            initiating_exact = int(
                getattr(hypothesis, "initiating_event_id", None) == source_id
            )
            playbook_match = int(
                expected_playbook in {None, hypothesis.playbook.value}
            )
            eligible_by_root.setdefault(source_id, []).append(
                (initiating_exact * 2 + playbook_match, episode_key, hypothesis)
            )
    for source_id, eligible in eligible_by_root.items():
        _, _, hypothesis = max(
            eligible,
            key=lambda value: (value[0], value[1]),
        )
        pending = pending_root_strata.pop(source_id)
        encoded = str(pending["encoded"])
        old_strata = list(json.loads(encoded))
        hypothesis_metadata = dict(
            getattr(hypothesis, "context_metadata", {}) or {}
        )
        new_strata = (
            old_strata[0],
            old_strata[1],
            old_strata[2],
            str(
                hypothesis_metadata.get(
                    "source_tier",
                    hypothesis_metadata.get(
                        "manipulation_tier", old_strata[3]
                    ),
                )
            ),
            str(
                hypothesis_metadata.get(
                    "authority_relation", "unrelated"
                )
            ),
            old_strata[5],
        )
        linked_encoded = _lsr_reach_key(new_strata)
        linked_roots.add(source_id)
        compact["candidate_roots_linked"] = (
            int(compact["candidate_roots_linked"]) + 1
        )
        candidate_counts[encoded]["candidate_roots"] -= 1
        if candidate_counts[encoded]["candidate_roots"] == 0:
            candidate_counts.pop(encoded)
        linked_counts = candidate_counts.setdefault(
            linked_encoded,
            {"candidate_roots": 0, "linked_to_episode": 0},
        )
        linked_counts["candidate_roots"] += 1
        linked_counts["linked_to_episode"] += 1
        case = compact["candidate_cases"].get(source_id)
        if case is not None:
            case["linked_to_episode"] = True
            case["authority_relation"] = new_strata[4]

    # Bind against this minute's candidates before retiring roots that left
    # the open context on the same clock; that transition is the common
    # root→episode hand-off.
    for root_id in departed_root_ids:
        active_roots.discard(root_id)
        linked_roots.discard(root_id)
        pending_root_strata.pop(root_id, None)

    stage_counts: dict[str, int] = compact["episode_stage_counts"]
    masks: dict[str, int] = compact["episode_masks"]
    metadata_by_key: dict[str, dict[str, Any]] = compact["episode_metadata"]

    def mark(hypothesis: Any, stage: str) -> None:
        key, metadata, created = _compact_episode_record(compact, hypothesis)
        if created:
            hypothesis_metadata = dict(
                getattr(hypothesis, "context_metadata", {}) or {}
            )
            metadata.update(
                {
                    "playbook": hypothesis.playbook.value,
                    "direction": hypothesis.direction.value,
                    "setup_id": getattr(hypothesis, "setup_context_id", None),
                    "episode_id": getattr(hypothesis, "episode_id", None),
                    "source_timeframe": str(
                        hypothesis_metadata.get("source_timeframe", "unknown")
                    ),
                    "source_tier": str(
                        hypothesis_metadata.get(
                            "source_tier",
                            hypothesis_metadata.get(
                                "manipulation_tier", "unknown"
                            ),
                        )
                    ),
                    "authority_relation": str(
                        hypothesis_metadata.get(
                            "authority_relation", "unrelated"
                        )
                    ),
                    "market_mode": market_mode,
                    "terminal": False,
                }
            )
            metadata.update(
                _episode_identity_diagnostics(
                    hypothesis,
                    hypothesis_metadata,
                )
            )
            _compact_case_insert(
                compact["episode_cases"],
                key,
                {
                    "record_type": "episode",
                    "record_id": key,
                    "first_observed_at": asof.isoformat(),
                    "playbook": metadata["playbook"],
                    "direction": metadata["direction"],
                    "source_timeframe": metadata["source_timeframe"],
                    "authority_relation": metadata["authority_relation"],
                    "market_mode": metadata["market_mode"],
                    "highest_stage": "candidate_root",
                    "terminal": False,
                    "censored": False,
                    "exit_reason": None,
                },
            )
        elif getattr(hypothesis, "episode_id", None) is not None:
            metadata["episode_id"] = str(hypothesis.episode_id)
        _update_episode_causal_diagnostics(
            metadata,
            hypothesis,
            asof=asof,
        )
        bit = 1 << EPISODE_FUNNEL_STAGES.index(stage)
        mask = int(masks[key])
        if not mask & bit:
            strata = tuple(
                str(metadata.get(field, "unknown"))
                for field in _EPISODE_FUNNEL_STRATA
            )
            encoded = _lsr_reach_key((*strata, stage))
            stage_counts[encoded] = int(stage_counts.get(encoded, 0)) + 1
            masks[key] = mask | bit
            if stage == "waiting_trigger":
                metadata.setdefault("first_pullback_at", asof.isoformat())
        case = compact["episode_cases"].get(key)
        if case is not None:
            for identity_field in (
                "episode_id",
                "context_thesis_id",
                "parent_context_thesis_id",
                "initiating_event_id",
                "entry_location_id",
                "entry_path_id",
                "lsr_manipulation_id",
                "lsr_pool_path_id",
                "lsr_displacement_id",
                "lsr_entry_zone_id",
                "eligible_entry_zone",
            ):
                case[identity_field] = metadata.get(identity_field)
            reached = {
                item
                for index, item in enumerate(EPISODE_FUNNEL_STAGES)
                if int(masks[key]) & (1 << index)
            }
            case["highest_stage"] = next(
                (
                    item
                    for item in reversed(EPISODE_FUNNEL_STAGES)
                    if item in reached
                ),
                "candidate_root",
            )

    def mark_existing(key: str, stage: str) -> None:
        metadata = metadata_by_key[key]
        bit = 1 << EPISODE_FUNNEL_STAGES.index(stage)
        mask = int(masks[key])
        if not mask & bit:
            strata = tuple(
                str(metadata.get(field, "unknown"))
                for field in _EPISODE_FUNNEL_STRATA
            )
            encoded = _lsr_reach_key((*strata, stage))
            stage_counts[encoded] = int(stage_counts.get(encoded, 0)) + 1
            masks[key] = mask | bit
        case = compact["episode_cases"].get(key)
        if case is not None:
            case["highest_stage"] = stage

    for _, hypothesis in lifecycle_candidates:
        if _episode_key(hypothesis) is None:
            continue
        phase = hypothesis.phase.value
        prefix = _EPISODE_PHASE_PREFIX.get(phase)
        if phase == "weakening":
            key = _episode_key(hypothesis)
            prior_mask = 0 if key is None else int(masks.get(key, 0))
            executable_bit = 1 << EPISODE_FUNNEL_STAGES.index("executable")
            prefix = EPISODE_FUNNEL_STAGES[
                : 6 if prior_mask & executable_bit else 5
            ]
        if prefix is None:
            prefix = ("candidate_root",)
        for stage in prefix:
            mark(hypothesis, stage)
        key = _episode_key(hypothesis)
        if key is not None and phase in {"completed", "invalidated"}:
            metadata = metadata_by_key.get(key)
            if metadata is not None:
                metadata["terminal"] = True
                metadata["terminal_reason"] = (
                    getattr(hypothesis, "terminal_reason", None) or phase
                )
            case = compact["episode_cases"].get(key)
            if case is not None:
                case["terminal"] = True
                case["exit_reason"] = (
                    getattr(hypothesis, "terminal_reason", None) or phase
                )

    selected = _resolve_hypothesis(
        snapshot.belief,
        snapshot.decision.best_hypothesis_key,
    )
    if selected is not None and _episode_key(selected) is not None:
        if snapshot.decision.selected_action is Action.ENTER:
            mark(selected, "decision_enter")
        if snapshot.risk.final_action is Action.ENTER:
            mark(selected, "risk_pass")

    position = None if step is None else getattr(step, "position", None)
    if position is not None:
        matches = [
            key
            for key, metadata in metadata_by_key.items()
            if metadata.get("playbook") == position.playbook.value
            and metadata.get("direction") == position.direction.value
            and metadata.get("setup_id") == position.setup_id
        ]
        if len(matches) == 1:
            mark_existing(matches[0], "order_filled")
    approvals = state.get("entry_approvals", {})
    for trade in tuple(
        () if step is None else getattr(step, "closed_trades", ())
    ):
        approval = approvals.get(trade.thesis_hash)
        if not isinstance(approval, Mapping):
            continue
        matches = [
            key
            for key, metadata in metadata_by_key.items()
            if metadata.get("episode_id") == approval.get("episode_id")
            and metadata.get("playbook") == trade.playbook
            and metadata.get("direction") == trade.direction
        ]
        if len(matches) != 1:
            continue
        key = matches[0]
        mark_existing(key, "order_filled")
        mark_existing(key, "position_terminal")
        metadata_by_key[key]["terminal"] = True
        metadata_by_key[key]["terminal_reason"] = trade.exit_reason
        case = compact["episode_cases"].get(key)
        if case is not None:
            case["terminal"] = True
            case["exit_reason"] = trade.exit_reason

    # Terminal candidates are retained only while Brain still exposes their
    # root-specific identity (normally the one-minute terminal/rearm bridge).
    # Run this after position/trade feedback so a disappearing candidate can
    # still receive its final position-terminal stage on the same clock.
    current_episode_keys = {
        key
        for _, hypothesis in lifecycle_candidates
        if (key := _episode_key(hypothesis)) is not None
    }
    for key, metadata in tuple(metadata_by_key.items()):
        if not metadata.get("terminal") or key in current_episode_keys:
            continue
        reason = str(
            (
                compact["episode_cases"].get(key, {}) or {}
            ).get("exit_reason")
            or metadata.get("terminal_reason")
            or "terminal_unspecified"
        )
        terminal_counts = compact["terminal_disposition_counts"]
        terminal_counts[reason] = int(terminal_counts.get(reason, 0)) + 1
        masks.pop(key, None)
        metadata_by_key.pop(key, None)
        for index_key, value in tuple(
            compact["provisional_episode_keys"].items()
        ):
            if value == key:
                compact["provisional_episode_keys"].pop(index_key, None)


def _compact_natural_episode_funnel_summary(
    state: dict[str, Any],
    *,
    end_asof: pd.Timestamp | None = None,
) -> dict[str, Any]:
    compact = state.get("natural_funnel_compact_state")
    if not isinstance(compact, dict):
        raise ValueError("checkpoint compact natural funnel state is invalid")
    rows: list[dict[str, Any]] = []
    stage_rank = {
        stage: index for index, stage in enumerate(EPISODE_FUNNEL_STAGES)
    }
    for encoded, count in compact["episode_stage_counts"].items():
        values = json.loads(encoded)
        row = dict(
            zip(
                (*_EPISODE_FUNNEL_STRATA, "stage"),
                values,
                strict=True,
            )
        )
        row["episodes_first_reached"] = int(count)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            *(str(row[field]) for field in _EPISODE_FUNNEL_STRATA),
            stage_rank[str(row["stage"])],
        )
    )
    candidate_rows: list[dict[str, Any]] = []
    for encoded, counts in compact["candidate_root_counts"].items():
        strata = json.loads(encoded)
        candidate_rows.append(
            {
                **dict(zip(_EPISODE_FUNNEL_STRATA, strata, strict=True)),
                "candidate_roots": int(counts["candidate_roots"]),
                "linked_to_episode": int(counts["linked_to_episode"]),
            }
        )
    candidate_rows.sort(
        key=lambda row: tuple(
            str(row[field]) for field in _EPISODE_FUNNEL_STRATA
        )
    )
    cases = [
        dict(case)
        for case in (
            *compact["candidate_cases"].values(),
            *compact["episode_cases"].values(),
        )
    ][:40]
    if end_asof is not None:
        for case in cases:
            if case.get("record_type") == "episode" and not case.get(
                "terminal", False
            ):
                case["censored"] = True
    linked = int(compact["candidate_roots_linked"])
    observed = int(compact["candidate_roots_observed"])
    terminal_dispositions = Counter(
        {
            str(key): int(value)
            for key, value in compact["terminal_disposition_counts"].items()
        }
    )
    # A run may end on the one-minute terminal bridge before the identity is
    # absent on a later clock.  Include that live terminal metadata in the
    # report without mutating checkpoint state.
    for key, metadata in compact["episode_metadata"].items():
        if not metadata.get("terminal"):
            continue
        reason = str(
            (compact["episode_cases"].get(key, {}) or {}).get("exit_reason")
            or metadata.get("terminal_reason")
            or "terminal_unspecified"
        )
        terminal_dispositions[reason] += 1
    return {
        "diagnostic_schema_version": (
            NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "counting_basis": (
            "candidate_root_cohort_then_unique_episode_first_reach_per_stage"
        ),
        "denominators": {
            "candidate_roots_observed": observed,
            "candidate_roots_linked_to_episode": linked,
            "candidate_roots_unlinked": observed - linked,
            "formed_episodes_observed": int(compact["episode_count"]),
        },
        "candidate_root_counts": candidate_rows,
        "stage_order": list(EPISODE_FUNNEL_STAGES),
        "strata": list(_EPISODE_FUNNEL_STRATA),
        "episodes_observed": int(compact["episode_count"]),
        "rows": rows,
        "terminal_disposition_counts": dict(
            sorted(terminal_dispositions.items())
        ),
        "compact_case_index": {
            "selection": "online_bounded_first_seen_by_record_type",
            "limit": 40,
            "cases": cases,
        },
    }


def _episode_record_for_hypothesis(
    state: dict[str, Any],
    hypothesis: Any,
) -> dict[str, Any] | None:
    if _episode_key(hypothesis) is None:
        return None
    _, value = _episode_record(state, hypothesis)
    return value


def _update_open_thesis_binding_funnel(
    state: dict[str, Any],
    snapshot: Any,
    *,
    action_playbooks: tuple[Playbook, ...] = (
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    ),
) -> None:
    """Count each open-thesis/playbook opportunity once per reached stage."""

    records = state.setdefault("open_thesis_binding_records", {})
    if not isinstance(records, dict):
        raise ValueError("checkpoint open-thesis funnel state is invalid")
    context = getattr(snapshot.belief, "global_context", None)
    if context is None:
        return
    action_candidates = _action_candidate_items(snapshot.belief)
    for thesis in context.open_market_theses:
        playbooks = tuple(
            playbook
            for playbook in action_playbooks
            if _open_thesis_supports_playbook(playbook, thesis)
        )
        directions = (
            tuple(Direction)
            if thesis.direction is None
            else (thesis.direction,)
        )
        opportunities = (
            tuple((playbook, direction) for playbook in playbooks for direction in directions)
            if playbooks and directions
            else ((None, None),)
        )
        for playbook, direction in opportunities:
            hypothesis_key = (
                None
                if playbook is None or direction is None
                else f"{playbook.value}:{direction.value}"
            )
            identity = _lsr_reach_key(
                (
                    context.market_epoch_id,
                    thesis.thesis_id,
                    "unmatched" if hypothesis_key is None else hypothesis_key,
                )
            )
            record = records.setdefault(
                identity,
                {
                    "market_epoch_id": context.market_epoch_id,
                    "thesis_id": thesis.thesis_id,
                    "root_id": thesis.root_id,
                    "playbook": (
                        "unmatched" if playbook is None else playbook.value
                    ),
                    "direction": (
                        "unknown" if direction is None else direction.value
                    ),
                    "thesis_direction": (
                        "unknown"
                        if thesis.direction is None
                        else thesis.direction.value
                    ),
                    "mechanism": thesis.mechanism,
                    "authority_relation": thesis.authority_relation,
                    "first_seen_at": pd.Timestamp(
                        context.updated_at
                    ).isoformat(),
                    "last_seen_at": pd.Timestamp(
                        context.updated_at
                    ).isoformat(),
                    "mask": 0,
                    "stage_strata": {},
                    "selected_hypothesis_match_strength": None,
                    "current_match_status": "unmatched",
                    "last_observed_failed_hard_gate_id": None,
                    "plan_delivery_failure": None,
                    "current_plan_delivery_failure": None,
                    "executable_block_reason": None,
                    "draw_candidate_count": 0,
                    "sample_draw_candidate_ids": (),
                    "selected_draw_id": None,
                },
            )
            record["authority_relation"] = thesis.authority_relation
            draw_candidate_ids = tuple(
                getattr(thesis, "draw_candidate_ids", ())
            )
            record["draw_candidate_count"] = len(draw_candidate_ids)
            record["sample_draw_candidate_ids"] = draw_candidate_ids[:3]
            record["last_seen_at"] = pd.Timestamp(
                context.updated_at
            ).isoformat()
            mask = int(record["mask"])

            def mark(stage: str) -> None:
                nonlocal mask
                bit = _OPEN_THESIS_BINDING_BITS[stage]
                if not mask & bit:
                    record["stage_strata"][stage] = {
                        field: record[field]
                        for field in _OPEN_THESIS_FUNNEL_STRATA
                    }
                mask |= bit

            mark("open_thesis_created")
            if hypothesis_key is None:
                record["mask"] = mask
                continue
            # Compatibility is an analytical fact established before typed
            # evaluation.  The next stage asks whether Brain actually emitted
            # an independent candidate for this exact canonical root.
            mark("mechanism_direction_matched")
            eligible_candidates = [
                (candidate_id, hypothesis)
                for candidate_id, hypothesis in action_candidates
                if hypothesis.playbook is playbook
                and hypothesis.direction is direction
                and (
                    (
                        getattr(hypothesis, "required_root_id", None)
                        == thesis.root_id
                    )
                    or (
                        getattr(hypothesis, "required_root_id", None) is None
                        and thesis.thesis_id
                        in getattr(hypothesis, "market_thesis_ids", ())
                    )
                )
            ]
            if not eligible_candidates:
                record["current_match_status"] = "candidate_missing"
                record["selected_hypothesis_match_strength"] = None
                record["mask"] = mask
                continue
            _, hypothesis = max(
                eligible_candidates,
                key=lambda item: (
                    int(
                        getattr(item[1], "bound_market_thesis_id", None)
                        == thesis.thesis_id
                    ),
                    float(
                        getattr(item[1], "playbook_match_strength", 0.0)
                    ),
                    item[0],
                ),
            )
            required_root_id = getattr(
                hypothesis,
                "required_root_id",
                None,
            )
            exact_bound = (
                required_root_id in {None, thesis.root_id}
                and hypothesis.bound_market_thesis_id == thesis.thesis_id
                and getattr(
                    hypothesis,
                    "market_thesis_root_id",
                    thesis.root_id,
                )
                == thesis.root_id
            )
            record["current_match_status"] = (
                "exact_root_bound"
                if exact_bound
                else "root_identity_unbound"
            )
            record["selected_hypothesis_match_strength"] = (
                float(hypothesis.playbook_match_strength)
                if hypothesis.market_thesis_id == thesis.thesis_id
                else None
            )
            selected_draw = getattr(hypothesis, "selected_draw", None)
            record["selected_draw_id"] = (
                None
                if selected_draw is None
                else str(selected_draw.level_id)
            )
            if not exact_bound:
                record["mask"] = mask
                continue
            mark("exact_root_bound")
            hard_gate_results = dict(hypothesis.hard_gate_results)
            gates_complete = bool(hard_gate_results) and all(
                hard_gate_results.values()
            )
            failed_gate = next(
                (
                    gate_id
                    for gate_id, passed in hard_gate_results.items()
                    if not passed
                ),
                "missing_hard_gate_results" if not hard_gate_results else None,
            )
            record["last_observed_failed_hard_gate_id"] = failed_gate
            if not gates_complete:
                record["mask"] = mask
                continue
            mark("causal_gates_complete")
            metadata = dict(hypothesis.context_metadata)
            feasibility = getattr(hypothesis, "plan_feasibility", None)
            plan_delivery_valid = (
                bool(feasibility.valid)
                if feasibility is not None
                else metadata.get(
                    "playbook_plan_delivery_valid",
                    "false",
                ).lower()
                == "true"
            )
            if not plan_delivery_valid:
                metadata_failure = metadata.get(
                    "plan_feasibility_failure"
                )
                if metadata_failure in {None, "", "none"}:
                    metadata_failure = None
                failure = (
                    None
                    if feasibility is None
                    else feasibility.failure_reason
                ) or metadata_failure or (
                    "playbook_plan_unavailable"
                    if hypothesis.plan is None
                    else "delivery_not_available"
                )
                record["current_plan_delivery_failure"] = failure
                if record.get("plan_delivery_failure") is None:
                    record["plan_delivery_failure"] = failure
                record["mask"] = mask
                continue
            record["current_plan_delivery_failure"] = None
            mark("plan_delivery_valid")
            if hypothesis.phase.value == "executable":
                mark("executable")
                record["executable_block_reason"] = None
            else:
                record["executable_block_reason"] = (
                    f"phase:{hypothesis.phase.value}"
                )
            record["mask"] = mask


def _open_thesis_binding_funnel_summary(
    state: dict[str, Any],
) -> dict[str, Any]:
    records = state.get("open_thesis_binding_records", {})
    if not isinstance(records, dict):
        raise ValueError("checkpoint open-thesis funnel state is invalid")
    aggregate = state.get("open_thesis_binding_aggregate", {})
    if not isinstance(aggregate, dict):
        raise ValueError("checkpoint open-thesis aggregate is invalid")
    match_counts = {
        str(key): int(value)
        for key, value in aggregate.get("match_counts", {}).items()
    }
    root_masks: dict[str, int] = {}
    dispositions = {
        str(key): int(value)
        for key, value in aggregate.get("dispositions", {}).items()
    }
    failed_gates = {
        str(key): int(value)
        for key, value in aggregate.get("failed_gates", {}).items()
    }
    compact_cases = [
        dict(value) for value in aggregate.get("cases", {}).values()
    ]
    for record_identity, record in records.items():
        if not isinstance(record, dict):
            raise ValueError("checkpoint open-thesis funnel record is invalid")
        mask = int(record["mask"])
        root_key = _lsr_reach_key(
            (record["market_epoch_id"], record["thesis_id"])
        )
        root_masks[root_key] = int(root_masks.get(root_key, 0)) | mask
        for stage in OPEN_THESIS_BINDING_STAGES:
            if mask & _OPEN_THESIS_BINDING_BITS[stage]:
                stage_strata = record.get("stage_strata", {}).get(stage)
                if not isinstance(stage_strata, dict):
                    raise ValueError(
                        "open-thesis funnel stage strata are invalid"
                    )
                strata = tuple(
                    str(stage_strata[field])
                    for field in _OPEN_THESIS_FUNNEL_STRATA
                )
                encoded = _lsr_reach_key((*strata, stage))
                match_counts[encoded] = int(match_counts.get(encoded, 0)) + 1
        disposition, failed_gate = _open_thesis_disposition(record, mask)
        dispositions[disposition] = int(dispositions.get(disposition, 0)) + 1
        if failed_gate is not None:
            failed_gates[failed_gate] = int(failed_gates.get(failed_gate, 0)) + 1
        compact_cases.append(
            _open_thesis_case(
                str(record_identity), record, mask, disposition
            )
        )
    rows: list[dict[str, Any]] = []
    stage_rank = {
        stage: index for index, stage in enumerate(OPEN_THESIS_BINDING_STAGES)
    }
    for encoded, count in match_counts.items():
        values = json.loads(encoded)
        row = dict(
            zip(
                (*_OPEN_THESIS_FUNNEL_STRATA, "stage"),
                values,
                strict=True,
            )
        )
        row["opportunities_first_reached"] = int(count)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            *(str(row[field]) for field in _OPEN_THESIS_FUNNEL_STRATA),
            stage_rank[str(row["stage"])],
        )
    )
    root_stage_counts = {
        stage: int(aggregate.get("root_stage_counts", {}).get(stage, 0))
        + sum(
            bool(mask & _OPEN_THESIS_BINDING_BITS[stage])
            for mask in root_masks.values()
        )
        for stage in OPEN_THESIS_BINDING_STAGES
    }
    compact_cases.sort(
        key=lambda row: (
            str(row["disposition"]),
            str(row["first_seen_at"]),
            str(row["thesis_id"]),
            str(row["playbook"]),
            str(row["direction"]),
        )
    )
    selected_cases: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for disposition in sorted(dispositions):
        representative = next(
            (
                row
                for row in compact_cases
                if row["disposition"] == disposition
            ),
            None,
        )
        if representative is not None:
            selected_cases.append(representative)
            selected_ids.add(str(representative["record_id"]))
    for row in compact_cases:
        if len(selected_cases) >= 40:
            break
        if str(row["record_id"]) in selected_ids:
            continue
        selected_cases.append(row)
        selected_ids.add(str(row["record_id"]))
    return {
        "counting_basis": {
            "root": "unique_market_epoch_and_thesis_id",
            "match": "unique_market_epoch_thesis_and_hypothesis_key",
        },
        "stage_order": list(OPEN_THESIS_BINDING_STAGES),
        "root_theses_observed": int(
            aggregate.get("root_theses_observed", 0)
        )
        + len(root_masks),
        "match_opportunities_observed": int(
            aggregate.get("match_opportunities_observed", 0)
        )
        + len(records),
        "root_stage_counts": root_stage_counts,
        "match_rows": rows,
        "highest_stage_observed_dispositions": dict(
            sorted(dispositions.items())
        ),
        "last_observed_failed_hard_gate_counts": dict(
            sorted(failed_gates.items())
        ),
        "compact_case_index": {
            "selection": (
                "deterministic_one_per_disposition_then_first_seen"
            ),
            "limit": 40,
            "cases": selected_cases,
        },
    }


def _compact_open_thesis_binding_records(
    state: dict[str, Any],
    snapshot: Any,
) -> None:
    """Fold closed thesis opportunities into bounded aggregate state.

    The mutable record dictionary is only needed while a thesis remains in
    the current open set.  Once the thesis closes, its mask/disposition can be
    counted exactly once and the identity-bearing row can be released.
    """

    records = state.get("open_thesis_binding_records", {})
    if not isinstance(records, dict):
        raise ValueError("checkpoint open-thesis funnel state is invalid")
    context = getattr(snapshot.belief, "global_context", None)
    if context is None:
        return
    current = {
        (str(context.market_epoch_id), str(thesis.thesis_id))
        for thesis in context.open_market_theses
    }
    aggregate = state.setdefault(
        "open_thesis_binding_aggregate",
        {
            "root_stage_counts": {},
            "root_theses_observed": 0,
            "match_counts": {},
            "dispositions": {},
            "failed_gates": {},
            "match_opportunities_observed": 0,
            "cases": {},
        },
    )
    if not isinstance(aggregate, dict):
        raise ValueError("checkpoint open-thesis aggregate is invalid")
    closed_root_masks: dict[str, int] = {}
    for identity, record in tuple(records.items()):
        key = (str(record["market_epoch_id"]), str(record["thesis_id"]))
        if key in current:
            continue
        mask = int(record["mask"])
        root_key = _lsr_reach_key(key)
        closed_root_masks[root_key] = (
            int(closed_root_masks.get(root_key, 0)) | mask
        )
        for stage in OPEN_THESIS_BINDING_STAGES:
            if not mask & _OPEN_THESIS_BINDING_BITS[stage]:
                continue
            strata = record.get("stage_strata", {}).get(stage)
            if not isinstance(strata, dict):
                raise ValueError("open-thesis funnel stage strata are invalid")
            encoded = _lsr_reach_key(
                tuple(str(strata[field]) for field in _OPEN_THESIS_FUNNEL_STRATA)
                + (stage,)
            )
            aggregate["match_counts"][encoded] = (
                int(aggregate["match_counts"].get(encoded, 0)) + 1
            )
        disposition, failed_gate = _open_thesis_disposition(record, mask)
        aggregate["dispositions"][disposition] = (
            int(aggregate["dispositions"].get(disposition, 0)) + 1
        )
        if failed_gate is not None:
            aggregate["failed_gates"][failed_gate] = (
                int(aggregate["failed_gates"].get(failed_gate, 0)) + 1
            )
        aggregate["match_opportunities_observed"] = (
            int(aggregate["match_opportunities_observed"]) + 1
        )
        _compact_disposition_case_insert(
            aggregate["cases"],
            str(identity),
            _open_thesis_case(str(identity), record, mask, disposition),
            limit=40,
        )
        records.pop(identity)
    for mask in closed_root_masks.values():
        aggregate["root_theses_observed"] = (
            int(aggregate["root_theses_observed"]) + 1
        )
        for stage in OPEN_THESIS_BINDING_STAGES:
            if mask & _OPEN_THESIS_BINDING_BITS[stage]:
                aggregate["root_stage_counts"][stage] = (
                    int(aggregate["root_stage_counts"].get(stage, 0)) + 1
                )


def _open_thesis_disposition(
    record: Mapping[str, Any],
    mask: int,
) -> tuple[str, str | None]:
    if not mask & _OPEN_THESIS_BINDING_BITS["mechanism_direction_matched"]:
        return "no_mechanism_direction_match", None
    if not mask & _OPEN_THESIS_BINDING_BITS["exact_root_bound"]:
        return "matched_root_unbound", None
    if not mask & _OPEN_THESIS_BINDING_BITS["causal_gates_complete"]:
        return "root_bound_causal_gate_incomplete", str(
            record.get("last_observed_failed_hard_gate_id") or "unknown"
        )
    if not mask & _OPEN_THESIS_BINDING_BITS["plan_delivery_valid"]:
        return str(
            record.get("current_plan_delivery_failure")
            or record.get("plan_delivery_failure")
            or "gates_complete_plan_or_delivery_invalid"
        ), None
    if not mask & _OPEN_THESIS_BINDING_BITS["executable"]:
        return "plan_delivery_valid_not_executable", None
    return "executable", None


def _open_thesis_case(
    identity: str,
    record: Mapping[str, Any],
    mask: int,
    disposition: str,
) -> dict[str, Any]:
    highest_stage = next(
        (
            stage
            for stage in reversed(OPEN_THESIS_BINDING_STAGES)
            if mask & _OPEN_THESIS_BINDING_BITS[stage]
        ),
        "none",
    )
    return {
        "record_id": identity,
        "thesis_id": str(record["thesis_id"]),
        "root_id": str(record["root_id"]),
        "mechanism": str(record["mechanism"]),
        "thesis_direction": str(record["thesis_direction"]),
        "playbook": str(record["playbook"]),
        "direction": str(record["direction"]),
        "authority_relation": str(record["authority_relation"]),
        "first_seen_at": str(record["first_seen_at"]),
        "last_seen_at": str(record.get("last_seen_at", record["first_seen_at"])),
        "highest_stage_observed": highest_stage,
        "disposition": disposition,
        "current_match_status": (
            "root_identity_unbound"
            if disposition == "matched_root_unbound"
            else str(record.get("current_match_status", "unknown"))
        ),
        "selected_hypothesis_match_strength": record.get(
            "selected_hypothesis_match_strength"
        ),
        "last_observed_failed_hard_gate_id": record.get(
            "last_observed_failed_hard_gate_id"
        ),
        "plan_delivery_failure": record.get("plan_delivery_failure"),
        "current_plan_delivery_failure": record.get(
            "current_plan_delivery_failure"
        ),
        "executable_block_reason": record.get("executable_block_reason"),
        "draw_candidate_count": int(record.get("draw_candidate_count", 0)),
        "sample_draw_candidate_ids": list(
            record.get("sample_draw_candidate_ids", ())
        ),
        "selected_draw_id": record.get("selected_draw_id"),
    }


def _update_natural_episode_funnel(
    state: dict[str, Any],
    snapshot: Any,
    *,
    step: Any | None = None,
    retain_details: bool = True,
) -> None:
    """Update one compact, checkpoint-safe record per causal episode."""

    if not retain_details:
        _update_compact_natural_episode_funnel(
            state,
            snapshot,
            step=step,
        )
        return

    records = state.setdefault("natural_episode_funnel_records", {})
    if not isinstance(records, dict):
        raise ValueError("checkpoint natural episode funnel state is invalid")
    asof = pd.Timestamp(snapshot.observation.asof)
    context = getattr(snapshot.belief, "global_context", None)
    market_mode = _enum_text(getattr(context, "market_mode", None))

    # Candidate roots exist before a playbook episode.  Capture that cohort
    # independently so no_eligible_root and disconnected-root losses remain
    # visible instead of being manufactured from a later phase prefix.
    _update_candidate_root_cohort(state, snapshot)

    for _, hypothesis in _lifecycle_candidate_items(snapshot.belief):
        key = _episode_key(hypothesis)
        if key is None:
            continue
        metadata = dict(getattr(hypothesis, "context_metadata", {}) or {})
        key, record = _episode_record(state, hypothesis)
        if record is None:
            record = {
                "episode_id": (
                    None
                    if getattr(hypothesis, "episode_id", None) is None
                    else str(hypothesis.episode_id)
                ),
                "setup_id": getattr(hypothesis, "setup_context_id", None),
                "context_id": getattr(hypothesis, "context_id", None),
                "provisional": getattr(hypothesis, "episode_id", None) is None,
                "playbook": hypothesis.playbook.value,
                "direction": hypothesis.direction.value,
                "source_timeframe": str(
                    metadata.get("source_timeframe", "unknown")
                ),
                "source_tier": str(
                    metadata.get(
                        "source_tier",
                        metadata.get("manipulation_tier", "unknown"),
                    )
                ),
                "authority_relation": str(
                    metadata.get("authority_relation", "unrelated")
                ),
                "market_mode": market_mode,
                "first_reached_at": {},
                "first_reach_sequence": [],
                "phase_duration_seconds": {},
                "last_observed_phase": None,
                "last_observed_at": asof.isoformat(),
                "exit_reason": None,
                "terminal": False,
                "censored": False,
                **_episode_identity_diagnostics(hypothesis, metadata),
            }
            records[key] = record
            if record["provisional"]:
                provisional = state.setdefault(
                    "natural_provisional_episode_keys",
                    {},
                )
                for identity in (
                    record["setup_id"],
                    record["context_id"],
                ):
                    if identity is not None:
                        provisional[
                            _provisional_index_key(hypothesis, str(identity))
                        ] = key
        elif getattr(hypothesis, "episode_id", None) is not None:
            record["episode_id"] = str(hypothesis.episode_id)
            record["setup_id"] = getattr(
                hypothesis,
                "setup_context_id",
                record.get("setup_id"),
            )
            record["context_id"] = getattr(
                hypothesis,
                "context_id",
                record.get("context_id"),
            )
            record["provisional"] = False

        _update_episode_causal_diagnostics(
            record,
            hypothesis,
            asof=asof,
        )

        previous_clock = pd.Timestamp(record["last_observed_at"])
        previous_phase = record.get("last_observed_phase")
        if previous_phase is not None and asof > previous_clock:
            durations = record.setdefault("phase_duration_seconds", {})
            durations[previous_phase] = float(
                durations.get(previous_phase, 0.0)
            ) + float((asof - previous_clock).total_seconds())

        phase = hypothesis.phase.value
        prefix = _EPISODE_PHASE_PREFIX.get(phase)
        if phase == "weakening":
            previously_executable = "executable" in record.get(
                "first_reached_at",
                {},
            )
            prefix = EPISODE_FUNNEL_STAGES[
                : 6 if previously_executable else 5
            ]
        if prefix is None:
            prefix = ("candidate_root",)
        for stage in prefix:
            _record_episode_stage(record, stage, asof)
        first_reached_at = record.get("first_reached_at", {})
        if "waiting_trigger" in first_reached_at:
            record.setdefault(
                "first_pullback_at",
                first_reached_at["waiting_trigger"],
            )
        record["last_observed_phase"] = phase
        record["last_observed_at"] = asof.isoformat()
        if phase in {"completed", "invalidated"}:
            record["terminal"] = True
            record["exit_reason"] = (
                getattr(hypothesis, "terminal_reason", None) or phase
            )

    selected = _resolve_hypothesis(
        snapshot.belief,
        snapshot.decision.best_hypothesis_key,
    )
    if selected is not None:
        record = _episode_record_for_hypothesis(state, selected)
        if record is not None and snapshot.decision.selected_action is Action.ENTER:
            _record_episode_stage(record, "decision_enter", asof)
        if record is not None and snapshot.risk.final_action is Action.ENTER:
            _record_episode_stage(record, "risk_pass", asof)

    if step is None:
        return
    position = getattr(step, "position", None)
    if position is not None:
        matches = [
            record
            for record in records.values()
            if isinstance(record, dict)
            and record.get("playbook") == position.playbook.value
            and record.get("direction") == position.direction.value
            and record.get("setup_id") == position.setup_id
        ]
        if len(matches) == 1:
            _record_episode_stage(matches[0], "order_filled", asof)
    approvals = state.get("entry_approvals", {})
    for trade in tuple(getattr(step, "closed_trades", ())):
        approval = approvals.get(trade.thesis_hash)
        if not isinstance(approval, Mapping):
            continue
        matches = [
            record
            for record in records.values()
            if isinstance(record, dict)
            and record.get("episode_id") == approval.get("episode_id")
            and record.get("playbook") == trade.playbook
            and record.get("direction") == trade.direction
        ]
        if len(matches) == 1:
            _record_episode_stage(matches[0], "order_filled", asof)
            _record_episode_stage(matches[0], "position_terminal", asof)
            matches[0]["terminal"] = True
            matches[0]["exit_reason"] = trade.exit_reason


def _natural_episode_funnel_summary(
    state: dict[str, Any],
    *,
    end_asof: pd.Timestamp | None = None,
    include_details: bool = False,
) -> dict[str, Any]:
    if (
        not include_details
        and "natural_funnel_compact_state" in state
    ):
        return _compact_natural_episode_funnel_summary(
            state,
            end_asof=end_asof,
        )
    records = state.get("natural_episode_funnel_records", {})
    candidate_records = state.get("natural_candidate_root_records", {})
    aliases = state.get("natural_episode_key_aliases", {})
    if not isinstance(records, dict):
        raise ValueError("checkpoint natural episode funnel state is invalid")
    if not isinstance(candidate_records, dict):
        raise ValueError("checkpoint candidate-root cohort state is invalid")
    if not isinstance(aliases, dict):
        raise ValueError("checkpoint natural episode aliases are invalid")
    aggregate: dict[str, int] = {}
    episode_rows: list[dict[str, Any]] = []
    for record in records.values():
        if not isinstance(record, dict):
            raise ValueError("checkpoint natural episode record is invalid")
        row = {
            key: value
            for key, value in record.items()
            if key != "last_observed_at"
        }
        if end_asof is not None and not row.get("terminal"):
            last_phase = row.get("last_observed_phase")
            last_clock = record.get("last_observed_at")
            if last_phase is not None and last_clock is not None:
                elapsed = max(
                    0.0,
                    float(
                        (
                            pd.Timestamp(end_asof)
                            - pd.Timestamp(last_clock)
                        ).total_seconds()
                    ),
                )
                durations = dict(row.get("phase_duration_seconds", {}))
                durations[last_phase] = float(
                    durations.get(last_phase, 0.0)
                ) + elapsed
                row["phase_duration_seconds"] = durations
        row["censored"] = bool(
            not row.get("terminal") and end_asof is not None
        )
        episode_rows.append(row)
        for stage in row.get("first_reached_at", {}):
            strata = tuple(str(row.get(field, "unknown")) for field in _EPISODE_FUNNEL_STRATA)
            encoded = _lsr_reach_key((*strata, stage))
            aggregate[encoded] = int(aggregate.get(encoded, 0)) + 1
    rows: list[dict[str, Any]] = []
    for encoded, count in aggregate.items():
        values = json.loads(encoded)
        row = dict(
            zip(
                (*_EPISODE_FUNNEL_STRATA, "stage"),
                values,
                strict=True,
            )
        )
        row["episodes_first_reached"] = int(count)
        rows.append(row)
    stage_rank = {stage: index for index, stage in enumerate(EPISODE_FUNNEL_STAGES)}
    rows.sort(
        key=lambda row: (
            *(str(row[field]) for field in _EPISODE_FUNNEL_STRATA),
            stage_rank[str(row["stage"])],
        )
    )
    episode_rows.sort(
        key=lambda row: (
            str(row["playbook"]),
            str(row["direction"]),
            str(row["episode_id"]),
        )
    )
    def resolved_episode_key(value: str) -> str:
        seen: set[str] = set()
        current = value
        while current in aliases:
            if current in seen:
                raise ValueError("checkpoint natural episode aliases cycle")
            seen.add(current)
            current = str(aliases[current])
        return current

    candidate_rows = sorted(
        (
            {
                key: (
                    list(
                        dict.fromkeys(
                            resolved_episode_key(str(item))
                            for item in value
                        )
                    )
                    if key == "linked_episode_keys"
                    else value
                )
                for key, value in record.items()
                if key != "last_observed_at"
            }
            for record in candidate_records.values()
            if isinstance(record, dict)
        ),
        key=lambda row: str(row["candidate_root_id"]),
    )
    if len(candidate_rows) != len(candidate_records):
        raise ValueError("checkpoint candidate-root record is invalid")
    linked_roots = sum(
        bool(row.get("linked_episode_keys")) for row in candidate_rows
    )
    candidate_counts: dict[tuple[str, ...], dict[str, int]] = {}
    for row in candidate_rows:
        strata = (
            str(row.get("nearest_playbook") or "unknown"),
            str(row.get("direction", "unknown")),
            str(row.get("source_timeframe", "unknown")),
            str(row.get("source_tier", "unknown")),
            str(row.get("authority_relation", "unbound")),
            str(row.get("market_mode", "unknown")),
        )
        counts = candidate_counts.setdefault(
            strata,
            {"candidate_roots": 0, "linked_to_episode": 0},
        )
        counts["candidate_roots"] += 1
        counts["linked_to_episode"] += int(
            bool(row.get("linked_episode_keys"))
        )
    candidate_count_rows = [
        {
            **dict(zip(_EPISODE_FUNNEL_STRATA, strata, strict=True)),
            **counts,
        }
        for strata, counts in sorted(candidate_counts.items())
    ]
    terminal_dispositions = Counter(
        str(row.get("exit_reason") or "terminal_unspecified")
        for row in episode_rows
        if row.get("terminal")
    )
    compact_cases: list[dict[str, Any]] = []
    seen_case_strata: set[tuple[str, ...]] = set()
    candidate_cases = [
        {
            "record_type": "candidate_root",
            "record_id": str(row["candidate_root_id"]),
            "first_observed_at": row.get("first_observed_at"),
            "playbook": str(row.get("nearest_playbook") or "unknown"),
            "direction": str(row.get("direction", "unknown")),
            "source_timeframe": str(row.get("source_timeframe", "unknown")),
            "authority_relation": str(
                row.get("authority_relation", "unbound")
            ),
            "market_mode": str(row.get("market_mode", "unknown")),
            "linked_to_episode": bool(row.get("linked_episode_keys")),
        }
        for row in candidate_rows
    ]
    episode_cases = []
    for row in episode_rows:
        reached = set(row.get("first_reached_at", {}))
        highest_stage = next(
            (
                stage
                for stage in reversed(EPISODE_FUNNEL_STAGES)
                if stage in reached
            ),
            "candidate_root",
        )
        identity = (
            row.get("episode_id")
            or row.get("setup_id")
            or row.get("context_id")
            or "unknown"
        )
        episode_cases.append(
            {
                "record_type": "episode",
                "record_id": _lsr_reach_key(
                    (
                        str(row.get("playbook", "unknown")),
                        str(row.get("direction", "unknown")),
                        str(identity),
                    )
                ),
                "first_observed_at": row.get("first_reached_at", {}).get(
                    "candidate_root"
                ),
                "playbook": str(row.get("playbook", "unknown")),
                "direction": str(row.get("direction", "unknown")),
                "source_timeframe": str(
                    row.get("source_timeframe", "unknown")
                ),
                "authority_relation": str(
                    row.get("authority_relation", "unrelated")
                ),
                "market_mode": str(row.get("market_mode", "unknown")),
                "highest_stage": highest_stage,
                "terminal": bool(row.get("terminal")),
                "censored": bool(row.get("censored")),
                "exit_reason": row.get("exit_reason"),
            }
        )
    case_pool = sorted(
        (*candidate_cases, *episode_cases),
        key=lambda case: (
            str(case.get("first_observed_at") or ""),
            str(case["record_type"]),
            str(case["record_id"]),
        ),
    )
    for case in case_pool:
        stratum = (
            str(case["record_type"]),
            str(case["playbook"]),
            str(case["direction"]),
            str(case.get("source_timeframe", "unknown")),
            str(
                case.get(
                    "highest_stage",
                    "linked" if case.get("linked_to_episode") else "unlinked",
                )
            ),
        )
        if stratum in seen_case_strata:
            continue
        compact_cases.append(case)
        seen_case_strata.add(stratum)
        if len(compact_cases) >= 40:
            break
    if len(compact_cases) < 40:
        selected_ids = {
            (str(item["record_type"]), str(item["record_id"]))
            for item in compact_cases
        }
        for case in case_pool:
            identity = (str(case["record_type"]), str(case["record_id"]))
            if identity in selected_ids:
                continue
            compact_cases.append(case)
            selected_ids.add(identity)
            if len(compact_cases) >= 40:
                break

    result = {
        "diagnostic_schema_version": (
            NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "counting_basis": (
            "candidate_root_cohort_then_unique_episode_first_reach_per_stage"
        ),
        "denominators": {
            "candidate_roots_observed": len(candidate_rows),
            "candidate_roots_linked_to_episode": linked_roots,
            "candidate_roots_unlinked": len(candidate_rows) - linked_roots,
            "formed_episodes_observed": len(episode_rows),
        },
        "candidate_root_counts": candidate_count_rows,
        "stage_order": list(EPISODE_FUNNEL_STAGES),
        "strata": list(_EPISODE_FUNNEL_STRATA),
        "episodes_observed": len(episode_rows),
        "rows": rows,
        "terminal_disposition_counts": dict(
            sorted(terminal_dispositions.items())
        ),
        "compact_case_index": {
            "selection": "deterministic_stratified_then_first_seen",
            "limit": 40,
            "cases": compact_cases,
        },
    }
    if include_details:
        result["candidate_root_rows"] = candidate_rows
        result["episodes"] = episode_rows
    return result


def _write_natural_funnel_diagnostics(
    destination: Path,
    details: Mapping[str, Any],
    *,
    maximum_rows: int,
) -> str:
    """Write opt-in full funnel rows outside the default summary."""

    rows: list[dict[str, str]] = []
    for record_type, records, identity_field in (
        (
            "candidate_root",
            details.get("candidate_root_rows", ()),
            "candidate_root_id",
        ),
        ("episode", details.get("episodes", ()), "episode_id"),
    ):
        for index, record in enumerate(records):
            identity = record.get(identity_field)
            if identity is None:
                identity = (
                    record.get("setup_id")
                    or record.get("context_id")
                    or f"row-{index}"
                )
            if record_type == "episode":
                identity = _lsr_reach_key(
                    (
                        str(record.get("playbook", "unknown")),
                        str(record.get("direction", "unknown")),
                        str(identity),
                    )
                )
            rows.append(
                {
                    "record_type": record_type,
                    "record_id": str(identity),
                    "payload_json": _compact_json(record),
                }
            )
    rows.sort(key=lambda row: (row["record_type"], row["record_id"]))
    row_keys = tuple(
        f'{row["record_type"]}|{row["record_id"]}' for row in rows
    )
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("natural funnel diagnostic identities are not unique")
    relative_directory = Path("natural_funnel_diagnostics")
    shards: list[dict[str, Any]] = []
    for shard_index, offset in enumerate(range(0, len(rows), maximum_rows)):
        chunk = rows[offset : offset + maximum_rows]
        relative = relative_directory / f"part-{shard_index:05d}.parquet"
        shard_path = destination / relative
        atomic_parquet(
            pd.DataFrame(
                chunk,
                columns=("record_type", "record_id", "payload_json"),
            ),
            shard_path,
            field_types=NATURAL_FUNNEL_FIELD_TYPES,
        )
        shard_keys = tuple(
            f'{row["record_type"]}|{row["record_id"]}' for row in chunk
        )
        shards.append(
            {
                "path": str(relative),
                "rows": len(chunk),
                "first_key": shard_keys[0],
                "last_key": shard_keys[-1],
                "sha256": sha256_file(shard_path),
            }
        )
    manifest_relative = relative_directory / "manifest.json"
    atomic_bytes(
        destination / manifest_relative,
        canonical_json(
            {
                "format_version": NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION,
                "status": "complete",
                "schema_fingerprint": NATURAL_FUNNEL_SCHEMA_FINGERPRINT,
                "counting_basis": details.get("counting_basis"),
                "rows": len(rows),
                "candidate_root_rows": len(
                    details.get("candidate_root_rows", ())
                ),
                "episode_rows": len(details.get("episodes", ())),
                "shards": shards,
            }
        ),
    )
    return str(manifest_relative)


def _first_failed_gate(hypothesis: Any | None) -> str | None:
    if hypothesis is None:
        return None
    return next(
        (
            str(name)
            for name, passed in getattr(
                hypothesis,
                "hard_gate_results",
                {},
            ).items()
            if not bool(passed)
        ),
        None,
    )


def _unexplained_reason(first_failed_gate: str | None) -> str:
    if first_failed_gate is None:
        return "no_eligible_root"
    gate = first_failed_gate.lower()
    if "graph" in gate or "identity" in gate or "connected" in gate:
        return "graph_identity_disconnected"
    if "order" in gate or "sequence" in gate:
        return "causal_order_failed"
    if "displacement" in gate:
        return "no_reverse_displacement"
    if any(token in gate for token in ("entry_location", "entry_zone", "fvg", "order_block")):
        return "no_frozen_entry_zone"
    if any(token in gate for token in ("trigger", "micro_bos", "reacceptance")):
        return "no_trigger"
    if any(token in gate for token in ("draw", "liquidity")):
        return "draw_missing"
    if any(token in gate for token in ("authority", "conflict", "opposition")):
        return "authority_conflict"
    if any(token in gate for token in ("delivery", "barrier", "remaining_path", "obstruction")):
        return "delivery_blocked"
    return "causal_order_failed"


def _root_descriptor(observation: Any, root_id: str) -> dict[str, Any]:
    observation = brain_observation_view(observation)
    manipulation = next(
        (
            item
            for item in getattr(observation, "manipulations", ())
            if item.manipulation_id == root_id
        ),
        None,
    )
    if manipulation is not None:
        inventory = next(
            (
                item
                for item in getattr(observation, "liquidity_inventory", ())
                if item.item_id == manipulation.source_inventory_item_id
            ),
            None,
        )
        terminal_at = (
            manipulation.resolved_at or manipulation.censored_at
        )
        return {
            "formed_at": manipulation.formed_at.isoformat(),
            "terminal_at": (
                None if terminal_at is None else terminal_at.isoformat()
            ),
            "timeframe": manipulation.source_timeframe.value,
            "direction": (
                "short" if manipulation.side == "above" else "long"
            ),
            "source_kind": manipulation.source_kind,
            "structural_scale": _enum_text(
                getattr(inventory, "structural_rank", None)
            ),
            "event_order_signature": [
                "liquidity_source",
                "sweep",
                manipulation.lifecycle.value,
            ],
            "nearest_playbook": (
                Playbook.FAILED_AUCTION_VALUE_RETURN.value
                if manipulation.source_kind == "mature_range_boundary"
                else Playbook.LIQUIDITY_SWEEP_REVERSAL.value
            ),
        }
    path = next(
        (
            item
            for item in getattr(observation, "path_sequences", ())
            if item.sequence_id == root_id
        ),
        None,
    )
    if path is not None:
        terminal_at = getattr(path, "ended_at", None)
        return {
            "formed_at": path.formed_at.isoformat(),
            "terminal_at": (
                None if terminal_at is None else terminal_at.isoformat()
            ),
            "timeframe": Timeframe.M1.value,
            "direction": path.direction.value,
            "source_kind": path.context_kind,
            "structural_scale": "internal",
            "event_order_signature": [step.kind for step in path.steps],
            "nearest_playbook": (
                Playbook.LIQUIDITY_SWEEP_REVERSAL.value
                if path.context_kind == "pool_reversal"
                else Playbook.DISPLACEMENT_FIRST_PULLBACK.value
            ),
        }
    bos = next(
        (
            item
            for frame in getattr(observation, "frames", {}).values()
            for item in getattr(frame, "structure_breaks", ())
            if item.bos_id == root_id
        ),
        None,
    )
    if bos is not None:
        target_rank = next(
            (
                getattr(item, "structural_rank", None)
                for item in getattr(observation, "liquidity_inventory", ())
                if (
                    getattr(item, "item_id", None) == bos.target_swing_id
                    or bos.target_swing_id
                    in getattr(item, "source_ids", ())
                )
                and getattr(item, "structural_rank", None)
                in {"internal", "intermediate", "external"}
            ),
            None,
        )
        terminal_at = getattr(bos, "failed_at", None)
        formed_at = (
            getattr(bos, "resolved_at", None)
            or getattr(bos, "pending_at", None)
            or observation.asof
        )
        return {
            "formed_at": formed_at.isoformat(),
            "terminal_at": (
                None if terminal_at is None else terminal_at.isoformat()
            ),
            "timeframe": bos.timeframe.value,
            "direction": bos.direction.value,
            "source_kind": "accepted_continuation_bos",
            "structural_scale": target_rank or "unknown",
            "bos_scope": _enum_text(getattr(bos, "scope", None)),
            "event_order_signature": [
                "confirmed_structure_swing",
                "continuation_bos",
                "accepted",
            ],
            "nearest_playbook": (
                Playbook.DISPLACEMENT_FIRST_PULLBACK.value
            ),
        }
    return {
        "formed_at": observation.asof.isoformat(),
        "terminal_at": None,
        "timeframe": "unknown",
        "direction": "unknown",
        "source_kind": "unknown",
        "structural_scale": "unknown",
        "event_order_signature": [],
        "nearest_playbook": None,
    }


_UNEXPLAINED_AGGREGATE_DIMENSIONS = (
    "nearest_playbook",
    "direction",
    "timeframe",
    "source_kind",
    "structural_scale",
    "authority_relation",
    "unexplained_reason",
    "resolution",
    "terminal",
    "censored",
)


def _record_unexplained_episode_aggregate(
    state: dict[str, Any],
    row: Mapping[str, Any],
) -> None:
    counts = state.setdefault("unexplained_episode_aggregate_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("checkpoint unexplained aggregate state is invalid")
    key = json.dumps(
        [row.get(name) for name in _UNEXPLAINED_AGGREGATE_DIMENSIONS],
        separators=(",", ":"),
    )
    counts[key] = int(counts.get(key, 0)) + 1
    state["unexplained_episode_finalized_count"] = int(
        state.get("unexplained_episode_finalized_count", 0)
    ) + 1


def _sample_unexplained_episode_details(
    rows: list[dict[str, Any]],
    *,
    maximum: int = 40,
) -> list[dict[str, Any]]:
    """Return a deterministic, bounded diagnostic sample by semantic stratum."""

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(
            str(row.get(name))
            for name in (
                "source_kind",
                "timeframe",
                "direction",
                "nearest_playbook",
                "unexplained_reason",
                "resolution",
            )
        )
        groups.setdefault(key, []).append(row)
    for values in groups.values():
        values.sort(
            key=lambda row: (
                str(row.get("first_observed_at")),
                str(row.get("episode_root_id")),
            )
        )
    sampled: list[dict[str, Any]] = []
    ordered_keys = sorted(groups)
    while ordered_keys and len(sampled) < maximum:
        remaining: list[tuple[str, ...]] = []
        for key in ordered_keys:
            values = groups[key]
            if values and len(sampled) < maximum:
                sampled.append(values.pop(0))
            if values:
                remaining.append(key)
        ordered_keys = remaining
    return sampled


def _unexplained_episode_details_payload(
    rows: list[dict[str, Any]],
    *,
    maximum: int = 40,
) -> dict[str, Any]:
    """Return the stable JSON envelope for opt-in unexplained samples."""

    episodes = _sample_unexplained_episode_details(rows, maximum=maximum)
    return {
        "schema_version": 1,
        "counting_basis": (
            "bounded_deterministic_stratified_sample_of_unexplained_episodes"
        ),
        "population_count": len(rows),
        "sample_count": len(episodes),
        "maximum_sample_count": maximum,
        "episodes": episodes,
    }


def _update_unexplained_episode_summaries(
    state: dict[str, Any],
    snapshot: Any,
) -> None:
    summaries = state.setdefault("unexplained_episode_summaries", {})
    active_prior = state.setdefault("active_unexplained_episode_ids", [])
    if not isinstance(summaries, dict) or not isinstance(active_prior, list):
        raise ValueError("checkpoint unexplained episode state is invalid")
    context = getattr(snapshot.belief, "global_context", None)
    active = tuple(
        dict.fromkeys(
            getattr(context, "unexplained_structured_episode_ids", ())
            if context is not None
            else ()
        )
    )
    scene_changed = any(
        getattr(snapshot.observation, name, ())
        for name in (
            "scene_added_node_ids",
            "scene_revised_node_ids",
            "scene_added_edge_ids",
            "scene_revised_edge_ids",
            "scene_resolution_event_ids",
        )
    )
    if tuple(active_prior) == active and not scene_changed:
        return
    asof = pd.Timestamp(snapshot.observation.asof)
    hypotheses = tuple(
        hypothesis
        for _, hypothesis in _action_candidate_items(snapshot.belief)
    )
    for root_id in active:
        descriptor = _root_descriptor(snapshot.observation, str(root_id))
        nearest = descriptor["nearest_playbook"]
        direction = descriptor["direction"]
        matching = next(
            (
                hypothesis
                for hypothesis in hypotheses
                if hypothesis.playbook.value == nearest
                and hypothesis.direction.value == direction
                and getattr(hypothesis, "required_root_id", root_id)
                == root_id
            ),
            None,
        )
        failed_gate = _first_failed_gate(matching)
        relation = (
            "unrelated"
            if matching is None
            else str(
                getattr(matching, "context_metadata", {}).get(
                    "authority_relation",
                    "unrelated",
                )
            )
        )
        if root_id not in summaries:
            draws = (
                ()
                if context is None or direction not in {"long", "short"}
                else tuple(
                    context.external_draw_candidates.get(
                        "above" if direction == "long" else "below",
                        (),
                    )
                )
            )
            obstruction_view = (
                None
                if context is None or direction not in {"long", "short"}
                else getattr(context, "obstruction_views", {}).get(direction)
            )
            blocker_ids = (
                tuple(
                    obstruction.obstruction_id
                    for obstruction in sorted(
                        obstruction_view.hard_barriers,
                        key=lambda item: (
                            abs(
                                item.contact_price(Direction(direction))
                                - snapshot.observation.price
                            ),
                            item.obstruction_id,
                        ),
                    )
                )
                if obstruction_view is not None
                else tuple(
                    () if context is None else context.path_blocker_ids
                )
            )
            summaries[root_id] = {
                "episode_root_id": str(root_id),
                **descriptor,
                "authority_relation": relation,
                "visible_draw_count": len(draws),
                "nearest_draw_id": None if not draws else str(draws[0]),
                "hard_blocker_count": len(tuple(dict.fromkeys(blocker_ids))),
                "nearest_blocker_id": (
                    None if not blocker_ids else str(blocker_ids[0])
                ),
                "first_failed_gate": failed_gate,
                "unexplained_reason": _unexplained_reason(failed_gate),
                "first_observed_at": asof.isoformat(),
                "last_observed_at": asof.isoformat(),
                "terminal": descriptor["terminal_at"] is not None,
                "censored": False,
                "resolution": (
                    "source_terminal"
                    if descriptor["terminal_at"] is not None
                    else None
                ),
            }
        else:
            summaries[root_id]["last_observed_at"] = asof.isoformat()
            if descriptor["terminal_at"] is not None:
                summaries[root_id]["terminal_at"] = descriptor["terminal_at"]
                summaries[root_id]["terminal"] = True
                summaries[root_id]["resolution"] = "source_terminal"

    active_set = set(active)
    for root_id in tuple(active_prior):
        if root_id in active_set or root_id not in summaries:
            continue
        summary = summaries[root_id]
        if not summary.get("terminal"):
            summary["terminal_at"] = asof.isoformat()
            summary["resolution"] = "explained_or_root_resolved"
            summary["terminal"] = True
        if not bool(state.get("unexplained_episode_detail_mode", True)):
            _record_unexplained_episode_aggregate(state, summary)
            del summaries[root_id]
    state["active_unexplained_episode_ids"] = list(active)


def _unexplained_episode_summary(
    state: dict[str, Any],
    *,
    end_asof: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    summaries = state.get("unexplained_episode_summaries", {})
    active = set(state.get("active_unexplained_episode_ids", ()))
    if not isinstance(summaries, dict):
        raise ValueError("checkpoint unexplained episode state is invalid")
    output: list[dict[str, Any]] = []
    for root_id, value in summaries.items():
        if not isinstance(value, dict):
            raise ValueError("checkpoint unexplained episode summary is invalid")
        row = dict(value)
        if root_id in active and not row.get("terminal") and end_asof is not None:
            row["censored"] = True
            row["terminal_at"] = pd.Timestamp(end_asof).isoformat()
            row["resolution"] = "right_censored"
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            str(row["first_observed_at"]),
            str(row["episode_root_id"]),
        ),
    )


def _unexplained_episode_aggregate(
    state: dict[str, Any],
    *,
    end_asof: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Aggregate unexplained roots without embedding episode-level arrays."""

    details = _unexplained_episode_summary(state, end_asof=end_asof)
    counts: dict[tuple[Any, ...], int] = {
        tuple(json.loads(key)): int(value)
        for key, value in state.get(
            "unexplained_episode_aggregate_counts",
            {},
        ).items()
    }
    for detail in details:
        key = tuple(
            detail.get(name) for name in _UNEXPLAINED_AGGREGATE_DIMENSIONS
        )
        counts[key] = int(counts.get(key, 0)) + 1
    rows = [
        {
            **dict(zip(_UNEXPLAINED_AGGREGATE_DIMENSIONS, key)),
            "episodes": count,
        }
        for key, count in sorted(
            counts.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        )
    ]
    return {
        "counting_basis": "unique_episode_root",
        "episodes_observed": int(
            state.get("unexplained_episode_finalized_count", 0)
        )
        + len(details),
        "rows": rows,
    }


def _row(
    snapshot,
    *,
    account_state=None,
    belief_position_input=None,
) -> dict:
    ranked = snapshot.belief.ranked()
    top = ranked[0] if ranked else None
    decision_hypothesis = _resolve_hypothesis(
        snapshot.belief,
        snapshot.decision.best_hypothesis_key,
    )
    summary_hypothesis = decision_hypothesis or top
    summary_metadata = (
        {}
        if summary_hypothesis is None
        else dict(summary_hypothesis.context_metadata)
    )
    first_failed_hard_gate_id = (
        None
        if summary_hypothesis is None
        else next(
            (
                gate_id
                for gate_id, passed in summary_hypothesis.hard_gate_results.items()
                if not passed
            ),
            None,
        )
    )
    plan_delivery_valid = (
        None
        if summary_hypothesis is None
        else summary_metadata.get(
            "playbook_plan_delivery_valid",
            "false",
        ).lower()
        == "true"
    )
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
    global_context = snapshot.belief.global_context
    context_direction = (
        None
        if summary_hypothesis is None
        else summary_hypothesis.direction.value
    )
    obstruction_view = (
        None
        if global_context is None or context_direction is None
        else global_context.obstruction_views.get(context_direction)
    )
    path_obstructions = tuple(
        sorted(
            (
                ()
                if obstruction_view is None
                else obstruction_view.hard_barriers
            ),
            key=lambda obstruction: (
                abs(
                    obstruction.contact_price(
                        summary_hypothesis.direction
                    )
                    - snapshot.observation.price
                ),
                obstruction.obstruction_id,
            ),
        )
    )
    path_blocker_ids = tuple(
        dict.fromkeys(
            obstruction.obstruction_id
            for obstruction in path_obstructions
        )
    )
    material_conflicts = (
        () if global_context is None else global_context.material_conflicts
    )
    focused_conflict_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in getattr(
                snapshot.belief,
                "cross_scale_conflicts",
                (),
            )
            if value
        )
    )
    if not focused_conflict_ids:
        focused_conflict_ids = tuple(
            dict.fromkeys(conflict.conflict_id for conflict in material_conflicts)
        )
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
        "top_context_metadata": (
            None if top is None else _compact_json(top.context_metadata)
        ),
        "top_competing_episode_ids": (
            None
            if top is None
            else _compact_json(top.competing_episode_ids)
        ),
        "market_thesis_id": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.market_thesis_id
        ),
        "bound_market_thesis_id": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.bound_market_thesis_id
        ),
        "market_thesis_root_id": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.market_thesis_root_id
        ),
        "market_thesis_mechanism": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.market_thesis_mechanism
        ),
        "market_thesis_authority_relation": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.market_thesis_authority_relation
        ),
        "playbook_match_strength": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.playbook_match_strength
        ),
        "market_thesis_binding_required": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.market_thesis_binding_required
        ),
        "market_thesis_action_bound": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.market_thesis_action_bound
        ),
        "market_thesis_match_status": (
            None
            if summary_hypothesis is None
            else summary_hypothesis.market_thesis_match_status
        ),
        "playbook_first_failed_hard_gate_id": (
            first_failed_hard_gate_id
        ),
        "playbook_plan_delivery_valid": plan_delivery_valid,
        "global_market_mode": (
            None if global_context is None else global_context.market_mode.value
        ),
        "global_authority_timeframe": (
            None
            if global_context is None
            or global_context.authority_timeframe is None
            else global_context.authority_timeframe.value
        ),
        "global_authority_direction": (
            None
            if global_context is None
            or global_context.authority_direction is None
            else global_context.authority_direction.value
        ),
        "global_authority_source_ids": (
            None
            if global_context is None
            else _compact_json(global_context.authority_source_ids)
        ),
        "global_dislocated": (
            None if global_context is None else global_context.dislocated
        ),
        "global_scale_relations": (
            None
            if global_context is None
            else _compact_json(global_context.scale_relations)
        ),
        "global_path_blocker_count": len(path_blocker_ids),
        "global_nearest_path_blocker_id": (
            None if not path_blocker_ids else path_blocker_ids[0]
        ),
        "global_key_path_blocker_ids": _compact_json(path_blocker_ids[:3]),
        "global_material_conflict_count": len(material_conflicts),
        "global_key_material_conflict_ids": _compact_json(
            focused_conflict_ids[:3]
        ),
        "global_unexplained_episode_count": (
            0
            if global_context is None
            else len(global_context.unexplained_structured_episode_ids)
        ),
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
        "authority_barrier_id": (
            None
            if liquidity_route is None
            else liquidity_route.authority_barrier_id
        ),
        "authority_barrier_price": (
            None
            if liquidity_route is None
            else liquidity_route.authority_barrier_price
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
    parser.add_argument(
        "--market-case-profile-registry",
        default=str(DEFAULT_MARKET_CASE_PROFILE_REGISTRY),
        help=(
            "versioned current MarketCase profile registry; only read by "
            "--market-case-input"
        ),
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
    parser.add_argument(
        "--action-disabled-playbook",
        action="append",
        default=[],
        choices=tuple(playbook.value for playbook in Playbook),
        metavar="PLAYBOOK",
        help=(
            "deny new-entry action candidates from this playbook while "
            "retaining raw Brain lifecycle/semantic state and management of "
            "already-open positions; repeat for multiple playbooks"
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
        "--calibration-only",
        action="store_true",
        help=(
            "with --brain-calibration, omit default decision shards while "
            "retaining decision counts; diagnostics remain independently "
            "controlled by --brain-diagnostics"
        ),
    )
    parser.add_argument(
        "--compact-scene-graph",
        action="store_true",
        help=(
            "opt in to checkpoint-cadence cold Scene Graph compaction; "
            "allowed only with --brain-calibration --calibration-only"
        ),
    )
    parser.add_argument(
        "--shadow-outcomes",
        action="store_true",
        help=(
            "record outcome-blind Eye-event candidates and reveal only their "
            "later structural paths; this diagnostic never feeds model actions"
        ),
    )
    parser.add_argument(
        "--shadow-details",
        action="store_true",
        help=(
            "write per-candidate mechanism challenges and motif diagnostics; "
            "default Shadow output keeps only episode/root shards and counts"
        ),
    )
    parser.add_argument(
        "--causal-case-library",
        action="store_true",
        help=(
            "emit sparse EntryEpisode decision-time revisions and a separate "
            "future-label stream by consuming the existing Shadow recorder; "
            "requires --shadow-outcomes and never starts a second replay"
        ),
    )
    parser.add_argument(
        "--market-case-input",
        action="store_true",
        help=(
            "emit only the playbook-neutral MarketEpisode input stream; this "
            "mode disables all action, diagnostic, Shadow, outcome and legacy "
            "case outputs"
        ),
    )
    parser.add_argument(
        "--brain-diagnostics",
        action="store_true",
        help=(
            "collect aggregate natural-episode, open-thesis binding and "
            "unexplained-episode diagnostics independently of calibration "
            "and Shadow capture"
        ),
    )
    parser.add_argument(
        "--include-unexplained-episode-details",
        action="store_true",
        help=(
            "write a separate stratified unexplained-episode sample capped "
            "at 40 cases; the default summary remains aggregate-only"
        ),
    )
    parser.add_argument(
        "--include-natural-funnel-details",
        action="store_true",
        help=(
            "write full candidate-root and episode rows to separate diagnostic "
            "Parquet shards; the default summary remains aggregate-only"
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
        "--diagnostic-fail-shadow-finalize-after-batches",
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


def _load_shadow_diagnostic_profile(
    path: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup_days: int,
    action_disabled_playbooks: tuple[Playbook, ...] = (),
) -> tuple[str, Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = payload.get("shadow_diagnostic_profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("validation protocol has no shadow diagnostic profiles")
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for name, candidate in profiles.items():
        if not isinstance(name, str) or not isinstance(candidate, Mapping):
            raise ValueError("shadow diagnostic profile registry is invalid")
        profile_start = pd.Timestamp(candidate.get("start"))
        profile_end = pd.Timestamp(candidate.get("end_exclusive"))
        if profile_start.tzinfo is None or profile_end.tzinfo is None:
            raise ValueError("shadow diagnostic profile clocks must be aware")
        if (
            start == profile_start
            and end == profile_end
            and int(warmup_days)
            == int(candidate.get("warmup_calendar_days", -1))
        ):
            matches.append((name, candidate))
    if not matches:
        raise ValueError(
            "--shadow-outcomes requires an exact registered window and warmup"
        )
    if len(matches) != 1:
        raise ValueError("shadow diagnostic window matches multiple profiles")
    profile_name, profile = matches[0]
    required_false = (
        "threshold_search",
        "calibration_fit_allowed",
        "pnl_used",
        "mbo_used",
        "future_path_visible_to_model",
        "shadow_output_affects_action",
    )
    if any(profile.get(name) is not False for name in required_false):
        raise ValueError("shadow diagnostic profile violates its non-action contract")
    if profile.get("playbooks_frozen_during_run") is not True:
        raise ValueError("shadow diagnostic must freeze playbooks during the run")
    if profile.get("model_config") != "configs/model.json":
        raise ValueError("shadow diagnostic profile has no frozen model config")
    profile_action_disabled = profile.get("action_disabled_playbooks", [])
    if (
        not isinstance(profile_action_disabled, list)
        or any(type(value) is not str for value in profile_action_disabled)
    ):
        raise ValueError(
            "shadow diagnostic action_disabled_playbooks must be a string list"
        )
    expected_action_disabled = normalize_action_disabled_playbooks(
        profile_action_disabled
    )
    if tuple(action_disabled_playbooks) != expected_action_disabled:
        raise ValueError(
            "shadow diagnostic runtime action policy does not exactly match "
            "its registered profile"
        )
    profile_identity = {
        "recorder_schema_version": profile.get("recorder_schema_version"),
        "derived_schema_version": profile.get("derived_schema_version"),
        "protocol_version": profile.get("protocol_version"),
    }
    expected_identity = {
        "recorder_schema_version": SHADOW_OUTCOME_RECORDER_SCHEMA_VERSION,
        "derived_schema_version": SHADOW_DERIVED_SCHEMA_VERSION,
        "protocol_version": SHADOW_OUTCOME_PROTOCOL["protocol_version"],
    }
    if profile_identity != expected_identity:
        raise ValueError(
            "shadow diagnostic profile identity is incompatible with the "
            "current Shadow recorder contract"
        )
    if profile.get("observer_transition_delta_transport") is not True:
        raise ValueError(
            "shadow diagnostic profile must enable typed transition transport"
        )
    candidate_sources = profile.get("candidate_sources")
    if (
        not isinstance(candidate_sources, list)
        or tuple(candidate_sources)
        != tuple(SHADOW_OUTCOME_PROTOCOL["candidate_events"])
    ):
        raise ValueError(
            "shadow diagnostic profile candidate_sources disagree with the "
            "current shadow recorder protocol"
        )
    return profile_name, dict(profile)


def _load_market_case_input_profile(
    path: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup_days: int,
) -> tuple[str, Mapping[str, Any]]:
    """Bind input-only capture to one exact preregistered window."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        type(payload) is not dict
        or set(payload)
        != {
            "schema_version",
            "registry",
            "authority",
            "historical_registry",
            "market_case_input_profiles",
        }
        or payload.get("schema_version")
        != MARKET_CASE_PROFILE_REGISTRY_SCHEMA_VERSION
        or payload.get("registry") != "market_case_input_profiles"
        or payload.get("authority") != "current"
        or payload.get("historical_registry") != "configs/data_splits.json"
    ):
        raise ValueError(
            "market-case input profile registry identity is invalid"
        )
    profiles = payload.get("market_case_input_profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError(
            "validation protocol must contain a market-case input profile registry"
        )
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for name, candidate in profiles.items():
        if not isinstance(name, str) or not isinstance(candidate, Mapping):
            raise ValueError("market-case input profile registry is invalid")
        profile_start = pd.Timestamp(candidate.get("start"))
        profile_end = pd.Timestamp(candidate.get("end_exclusive"))
        if profile_start.tzinfo is None or profile_end.tzinfo is None:
            raise ValueError("market-case input profile clocks must be aware")
        if (
            start == profile_start
            and end == profile_end
            and int(warmup_days)
            == int(candidate.get("warmup_calendar_days", -1))
        ):
            matches.append((name, candidate))
    if not matches:
        raise ValueError(
            "--market-case-input requires an exact registered window and "
            "warmup"
        )
    if len(matches) != 1:
        raise ValueError("market-case input window matches multiple profiles")
    profile_name, profile = matches[0]
    required_false = (
        "threshold_search",
        "calibration_fit_allowed",
        "future_path_used",
        "outcome_used",
        "pnl_used",
        "mbo_used",
        "brain_output",
        "decision_output",
        "risk_output",
        "execution_output",
        "shadow_output",
        "legacy_case_output",
    )
    if any(profile.get(name) is not False for name in required_false):
        raise ValueError(
            "market-case input profile violates its input-only contract"
        )
    expected_identity = {
        "recorder_schema_version": MARKET_CASE_RECORDER_SCHEMA_VERSION,
        "protocol_version": MARKET_CASE_PROTOCOL["protocol_version"],
    }
    profile_identity = {
        "recorder_schema_version": profile.get("recorder_schema_version"),
        "protocol_version": profile.get("protocol_version"),
    }
    if (
        profile.get("runner_mode") != "market_episode_input_only"
        or profile.get("observer_transition_delta_transport") is not True
        or profile.get("model_config") != "configs/model.json"
        or profile_identity != expected_identity
    ):
        raise ValueError("market-case input profile identity is incompatible")
    return profile_name, dict(profile)


def _validate_market_input_replay_contract(
    profile: Mapping[str, Any],
    replay_contract: tuple[str, int],
) -> None:
    """Match a preregistered representation window to its sole source contract."""

    expected = profile.get("expected_replay_contract")
    if expected is None:
        return
    if not isinstance(expected, Mapping):
        raise ValueError("market-case input expected_replay_contract must be an object")
    registered = (expected.get("symbol"), expected.get("instrument_id"))
    if replay_contract != registered:
        raise ValueError(
            "--market-case-input replay contract does not match the registered profile"
        )


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
        "decision_rows": int(state.get("decision_rows", 0)),
        "brain_calibration_rows": int(
            state.get("brain_calibration_rows", 0)
        ),
        "shadow_outcome_rows": int(
            state.get("shadow_outcome_rows", 0)
        ),
        **(
            {
                "causal_case_input_rows": int(
                    state["causal_case_input_rows"]
                ),
                "causal_case_outcome_rows": int(
                    state["causal_case_outcome_rows"]
                ),
            }
            if "causal_case_input_rows" in state
            else {}
        ),
        **(
            {
                "market_case_input_rows": int(
                    state["market_case_input_rows"]
                ),
            }
            if "market_case_input_rows" in state
            else {}
        ),
        "rows_per_second": rows_per_second,
        "eta_seconds": (
            None if rows_per_second <= 0.0 else remaining / rows_per_second
        ),
        "resume_count": int(state["resume_count"]),
    }


def _stream_record(
    record: (
        BrainCalibrationRecord
        | ShadowCandidateOutcomeRecord
        | ShadowEpisodeOutcomeRecord
        | ShadowMechanismChallengeRecord
        | CausalCaseInputRecord
        | CausalCaseOutcomeRecord
        | MarketCaseInputRecord
    ),
) -> dict[str, Any]:
    return (
        record.to_dict()
        if isinstance(
            record,
            (
                ShadowCandidateOutcomeRecord,
                ShadowEpisodeOutcomeRecord,
                ShadowMechanismChallengeRecord,
            ),
        )
        else asdict(record)
    )


def _new_shadow_derived_counts() -> dict[str, Any]:
    return {
        "episode_count": 0,
        "episode_quadrants": Counter(),
        "episode_target_buckets": Counter(),
        "episode_risk_qualified": Counter(),
        "raw_challenge_row_count": 0,
        "representative_root_challenge_count": 0,
        "unbound_challenge_row_count": 0,
        "challenge_quadrants": Counter(),
        "representative_playbook_decisions": Counter(),
        "rejected_valid_gates": Counter(),
        "non_evaluable": Counter(),
        "challenge_target_buckets": Counter(),
        "challenge_risk_qualified": Counter(),
        "root_count": 0,
        "eligible_root_count": 0,
        "root_sequence_unit_count": 0,
        "root_sequence_scope_counts": Counter(),
        "root_sequence_depths": Counter(),
        "root_sequence_lifecycle_acceptance": Counter(),
    }


def _count_shadow_episode(counts: dict[str, Any], item: Any) -> None:
    counts["episode_count"] += 1
    bucket = item.target_R_bucket
    counts["episode_target_buckets"][(bucket, "total")] += 1
    if item.outcome_evaluable:
        outcome = "valid" if item.outcome_class == "path_valid" else "failed"
        counts["episode_target_buckets"][(bucket, "evaluable")] += 1
        counts["episode_target_buckets"][(bucket, outcome)] += 1
        counts["episode_quadrants"][
            "accepted_path_valid"
            if outcome == "valid"
            else "accepted_path_failed"
        ] += 1
    if item.risk_qualified_target_R:
        counts["episode_risk_qualified"]["total"] += 1
        if item.outcome_evaluable:
            counts["episode_risk_qualified"]["evaluable"] += 1
            counts["episode_risk_qualified"][
                "valid" if item.outcome_class == "path_valid" else "failed"
            ] += 1


def _count_representative_shadow_challenge(
    counts: dict[str, Any],
    item: Any,
) -> None:
    """Aggregate only the outcome-blind selected revision for one root."""

    counts["representative_root_challenge_count"] += 1
    playbook = item.playbook
    counts["representative_playbook_decisions"][
        (
            playbook,
            "accepted" if item.accepted_at_candidate_clock else "rejected",
        )
    ] += 1
    bucket = item.target_R_bucket
    counts["challenge_target_buckets"][(playbook, bucket, "total")] += 1
    if item.outcome_evaluable:
        outcome = "valid" if item.outcome_class == "path_valid" else "failed"
        counts["challenge_target_buckets"][(playbook, bucket, "evaluable")] += 1
        counts["challenge_target_buckets"][(playbook, bucket, outcome)] += 1
        quadrant = (
            ("accepted" if item.accepted_at_candidate_clock else "rejected")
            + "_path_"
            + outcome
        )
        counts["challenge_quadrants"][(playbook, quadrant)] += 1
        if not item.accepted_at_candidate_clock and outcome == "valid":
            counts["rejected_valid_gates"][(
                playbook,
                item.first_failed_gate or "no_failed_gate_recorded",
            )] += 1
    else:
        counts["non_evaluable"][(playbook, item.outcome_class)] += 1
    if item.risk_qualified_target_R:
        counts["challenge_risk_qualified"][(playbook, "total")] += 1
        if item.outcome_evaluable:
            counts["challenge_risk_qualified"][(playbook, "evaluable")] += 1
            counts["challenge_risk_qualified"][(
                playbook,
                "valid" if item.outcome_class == "path_valid" else "failed",
            )] += 1


def _count_shadow_root_sequence(
    counts: dict[str, Any],
    item: ShadowRootSequenceRecord,
) -> None:
    """Count descriptive candidate-clock lifecycle units only."""

    counts["root_sequence_unit_count"] += 1
    counts["root_sequence_scope_counts"][(item.playbook, item.scope_kind)] += 1
    depth = (
        "complete_playbook_sequence"
        if item.complete_sequence_observed
        else "single"
        if item.event_order_length == 1
        else "binary"
        if item.event_order_length == 2
        else "ternary"
        if item.event_order_length == 3
        else "four_plus_partial"
    )
    counts["root_sequence_depths"][(item.playbook, depth)] += 1
    counts["root_sequence_lifecycle_acceptance"][
        (item.playbook, "ever_accepted" if item.ever_accepted else "never_accepted")
    ] += 1
    counts["root_sequence_lifecycle_acceptance"][
        (
            item.playbook,
            "selected_accepted"
            if item.selected_revision_accepted
            else "selected_rejected",
        )
    ] += 1


def _shadow_derived_summary(
    counts: Mapping[str, Any],
    motifs: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    episode_buckets = counts["episode_target_buckets"]
    challenge_buckets = counts["challenge_target_buckets"]
    challenge_quadrants = counts["challenge_quadrants"]
    representative_decisions = counts["representative_playbook_decisions"]
    rejected = counts["rejected_valid_gates"]
    non_evaluable = counts["non_evaluable"]
    challenge_risk = counts["challenge_risk_qualified"]
    sequence_scopes = counts["root_sequence_scope_counts"]
    sequence_depths = counts["root_sequence_depths"]
    sequence_acceptance = counts["root_sequence_lifecycle_acceptance"]
    playbooks = sorted(
        {
            playbook
            for playbook, _ in (
                *challenge_quadrants.keys(),
                *rejected.keys(),
                *non_evaluable.keys(),
                *representative_decisions.keys(),
                *sequence_scopes.keys(),
                *sequence_depths.keys(),
                *sequence_acceptance.keys(),
            )
        }
    )
    eligible_motifs = sum(
        int(value["eligible_count"]) >= 2
        and len(value["eligible_dates"]) >= 2
        for value in motifs.values()
    )
    return {
        "counting_basis": {
            "raw_challenge_rows": "candidate revision × playbook",
            "representative_root_challenges": (
                "one first-geometry-complete outcome representative per "
                "root × playbook"
            ),
            "root_sequence_units": (
                "strict root × playbook × explicit EntryEpisode/setup/Context "
                "scope; longest candidate-clock signature only, never a "
                "price-outcome representative"
            ),
            "unbound_challenges": (
                "candidate/playbook rows without a canonical market-thesis "
                "root identity"
            ),
        },
        "episode_outcome_count": int(counts["episode_count"]),
        "episode_executable_quadrants": dict(
            sorted(counts["episode_quadrants"].items())
        ),
        "episode_target_R_buckets": {
            bucket: {
                outcome: episode_buckets[(bucket, outcome)]
                for item_bucket, outcome in sorted(episode_buckets)
                if item_bucket == bucket
            }
            for bucket in sorted({key[0] for key in episode_buckets})
        },
        "episode_risk_qualified_target_R_ge_1": dict(
            sorted(counts["episode_risk_qualified"].items())
        ),
        "raw_challenge_row_count": int(counts["raw_challenge_row_count"]),
        "representative_root_challenge_count": int(
            counts["representative_root_challenge_count"]
        ),
        "unbound_challenge_row_count": int(
            counts["unbound_challenge_row_count"]
        ),
        "independent_root_episode_count": int(counts["root_count"]),
        "eligible_root_episode_evidence_count": int(
            counts["eligible_root_count"]
        ),
        "root_sequence_unit_count": int(counts["root_sequence_unit_count"]),
        "root_sequence_scope_counts": {
            playbook: {
                scope: int(sequence_scopes[(playbook, scope)])
                for item_playbook, scope in sorted(sequence_scopes)
                if item_playbook == playbook
            }
            for playbook in playbooks
            if any(key[0] == playbook for key in sequence_scopes)
        },
        "root_sequence_depth_counts": {
            playbook: {
                depth: int(sequence_depths[(playbook, depth)])
                for item_playbook, depth in sorted(sequence_depths)
                if item_playbook == playbook
            }
            for playbook in playbooks
            if any(key[0] == playbook for key in sequence_depths)
        },
        "root_sequence_candidate_clock_acceptance": {
            playbook: {
                outcome: int(sequence_acceptance[(playbook, outcome)])
                for item_playbook, outcome in sorted(sequence_acceptance)
                if item_playbook == playbook
            }
            for playbook in playbooks
            if any(key[0] == playbook for key in sequence_acceptance)
        },
        "mechanism_motif_count": len(motifs),
        "motifs_eligible_for_preregistration_review": eligible_motifs,
        "motif_action_authority": False,
        "mechanism_challenge_four_quadrants": {
            playbook: {
                quadrant: challenge_quadrants[(playbook, quadrant)]
                for item_playbook, quadrant in sorted(challenge_quadrants)
                if item_playbook == playbook
            }
            for playbook in playbooks
        },
        "representative_playbook_acceptance": {
            playbook: {
                "accepted": int(
                    representative_decisions[(playbook, "accepted")]
                ),
                "rejected": int(
                    representative_decisions[(playbook, "rejected")]
                ),
                "total": int(
                    representative_decisions[(playbook, "accepted")]
                    + representative_decisions[(playbook, "rejected")]
                ),
                "acceptance_rate": (
                    float(
                        representative_decisions[(playbook, "accepted")]
                    )
                    / float(
                        representative_decisions[(playbook, "accepted")]
                        + representative_decisions[(playbook, "rejected")]
                    )
                ),
            }
            for playbook in playbooks
            if (
                representative_decisions[(playbook, "accepted")]
                + representative_decisions[(playbook, "rejected")]
            )
            > 0
        },
        "mechanism_rejected_valid_path_first_gates": {
            playbook: {
                gate: rejected[(playbook, gate)]
                for item_playbook, gate in sorted(rejected)
                if item_playbook == playbook
            }
            for playbook in playbooks
        },
        "mechanism_non_evaluable": {
            playbook: {
                reason: non_evaluable[(playbook, reason)]
                for item_playbook, reason in sorted(non_evaluable)
                if item_playbook == playbook
            }
            for playbook in playbooks
        },
        "mechanism_target_R_buckets": {
            playbook: {
                bucket: {
                    outcome: challenge_buckets[(playbook, bucket, outcome)]
                    for item_playbook, item_bucket, outcome in sorted(
                        challenge_buckets
                    )
                    if item_playbook == playbook and item_bucket == bucket
                }
                for bucket in sorted(
                    {
                        item_bucket
                        for item_playbook, item_bucket, _ in challenge_buckets
                        if item_playbook == playbook
                    }
                )
            }
            for playbook in playbooks
        },
        "mechanism_risk_qualified_target_R_ge_1": {
            playbook: {
                outcome: challenge_risk[(playbook, outcome)]
                for item_playbook, outcome in sorted(challenge_risk)
                if item_playbook == playbook
            }
            for playbook in playbooks
        },
    }


def _shadow_root_key(item: ShadowMechanismChallengeRecord) -> str:
    root_id = item.market_thesis_root_id
    if not isinstance(root_id, str) or not root_id:
        raise ValueError("Shadow root challenge lacks a canonical root identity")
    return json.dumps(
        [item.symbol, int(item.instrument_id), item.direction, root_id],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _shadow_timestamp_ns(value: pd.Timestamp) -> int:
    clock = pd.Timestamp(value)
    if clock.tzinfo is None:
        raise ValueError("Shadow root revision timestamp must be timezone-aware")
    return int(clock.tz_convert("UTC").value)


def _merge_shadow_root_selection_sqlite(
    connection: sqlite3.Connection,
    values: tuple[ShadowMechanismChallengeRecord, ...],
) -> None:
    """Persist one outcome-blind revision per root with bounded process RAM."""

    if not values:
        return
    representative = values[0]
    root_key = _shadow_root_key(representative)
    candidate_id = representative.candidate_id
    geometry_incomplete = int(not representative.geometry_complete)
    observed_at_ns = _shadow_timestamp_ns(representative.observed_at)
    for item in values:
        if (
            _shadow_root_key(item) != root_key
            or item.candidate_id != candidate_id
        ):
            raise ValueError("Shadow root selection batch mixes root revisions")
        if (
            int(not item.geometry_complete) != geometry_incomplete
            or _shadow_timestamp_ns(item.observed_at) != observed_at_ns
        ):
            raise ValueError(
                "Shadow root revision playbooks disagree on selection rank"
            )
    for item in values:
        payload = json.dumps(
            to_primitive(item.to_dict()),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        existing = connection.execute(
            "SELECT payload FROM root_sequence_revision "
            "WHERE root_key = ? AND playbook = ? AND candidate_id = ?",
            (root_key, item.playbook, candidate_id),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload:
                raise ValueError(
                    "Shadow sequence revision changed across batches"
                )
            continue
        connection.execute(
            "INSERT INTO root_sequence_revision "
            "(root_key, playbook, candidate_id, observed_at_ns, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                root_key,
                item.playbook,
                candidate_id,
                observed_at_ns,
                payload,
            ),
        )
    prior = connection.execute(
        "SELECT geometry_incomplete, observed_at_ns, candidate_id "
        "FROM root_choice WHERE root_key = ?",
        (root_key,),
    ).fetchone()
    rank = (geometry_incomplete, observed_at_ns, candidate_id)
    prior_rank = (
        None
        if prior is None
        else (int(prior[0]), int(prior[1]), str(prior[2]))
    )
    if prior_rank is not None and prior_rank[2] == candidate_id:
        if rank != prior_rank:
            raise ValueError(
                "Shadow root revision rank changed across batches"
            )
    if prior_rank is None or rank < prior_rank:
        connection.execute(
            "INSERT INTO root_choice "
            "(root_key, geometry_incomplete, observed_at_ns, candidate_id) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(root_key) DO UPDATE SET "
            "geometry_incomplete=excluded.geometry_incomplete, "
            "observed_at_ns=excluded.observed_at_ns, "
            "candidate_id=excluded.candidate_id",
            (root_key, geometry_incomplete, observed_at_ns, candidate_id),
        )
        connection.execute(
            "DELETE FROM root_challenge WHERE root_key = ?",
            (root_key,),
        )
    elif rank != prior_rank:
        return
    for item in values:
        payload = json.dumps(
            to_primitive(item.to_dict()),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        existing = connection.execute(
            "SELECT payload FROM root_challenge "
            "WHERE root_key = ? AND playbook = ?",
            (root_key, item.playbook),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload:
                raise ValueError(
                    "Shadow candidate/playbook challenge changed across batches"
                )
            continue
        connection.execute(
            "INSERT INTO root_challenge (root_key, playbook, payload) "
            "VALUES (?, ?, ?)",
            (root_key, item.playbook, payload),
        )


def _shadow_challenge_from_payload(
    payload: str,
) -> ShadowMechanismChallengeRecord:
    values = json.loads(payload)
    values["observed_at"] = pd.Timestamp(values["observed_at"])
    values["resolved_at"] = pd.Timestamp(values["resolved_at"])
    return ShadowMechanismChallengeRecord(**values)


def _shadow_derived_definitions(
    include_details: bool,
) -> dict[str, tuple[str, Mapping[str, str]]]:
    definitions: dict[str, tuple[str, Mapping[str, str]]] = {
        "episode_outcomes": (
            "episode_key",
            SHADOW_EPISODE_OUTCOME_FIELD_TYPES,
        ),
        "root_episode_challenges": (
            "root_episode_key",
            SHADOW_ROOT_EPISODE_FIELD_TYPES,
        ),
        "root_episode_sequences": (
            "sequence_unit_key",
            SHADOW_ROOT_SEQUENCE_FIELD_TYPES,
        ),
    }
    if include_details:
        definitions.update(
            {
                "mechanism_challenges": (
                    "challenge_id",
                    SHADOW_MECHANISM_CHALLENGE_FIELD_TYPES,
                ),
                "mechanism_motifs": (
                    "motif_id",
                    SHADOW_MECHANISM_MOTIF_FIELD_TYPES,
                ),
            }
        )
    return definitions


def _verify_complete_shadow_derived_outputs(
    destination: Path,
    payload: Mapping[str, Any],
    *,
    source_stream_manifest: str,
    include_details: bool,
) -> dict[str, Any]:
    """Verify every immutable derived shard before finalizer reuse."""

    definitions = _shadow_derived_definitions(include_details)
    streams = payload.get("streams")
    outputs = payload.get("outputs")
    summary = payload.get("summary")
    final = destination / "shadow_derived"
    if (
        not isinstance(streams, Mapping)
        or set(streams) != set(definitions)
        or not isinstance(outputs, Mapping)
        or not isinstance(summary, Mapping)
        or any(outputs.get(key) != value for key, value in summary.items())
        or outputs.get("manifest") != "shadow_derived_manifest.json"
        or final.is_symlink()
        or not final.is_dir()
    ):
        raise ValueError("complete Shadow derived output contract is invalid")

    for name, (_, field_types) in definitions.items():
        entry = streams.get(name)
        entry_field_types = (
            entry.get("field_types")
            if isinstance(entry, Mapping)
            else None
        )
        entry_rows = (
            entry.get("rows") if isinstance(entry, Mapping) else None
        )
        relative_manifest = str(
            Path("shadow_derived") / f"{name}.manifest.json"
        )
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry_field_types, Mapping)
            or not isinstance(entry_rows, int)
            or isinstance(entry_rows, bool)
            or entry_rows < 0
            or entry.get("manifest") != relative_manifest
            or entry.get("action_authority") is not False
            or dict(entry_field_types) != dict(field_types)
            or outputs.get(name) != relative_manifest
        ):
            raise ValueError(
                f"complete Shadow derived stream contract is invalid: {name}"
            )
        stream_manifest_path = destination / relative_manifest
        if stream_manifest_path.is_symlink() or not stream_manifest_path.is_file():
            raise ValueError(
                f"complete Shadow derived stream manifest is missing: {name}"
            )
        stream_manifest = json.loads(
            stream_manifest_path.read_text(encoding="utf-8")
        )
        manifest_field_types = stream_manifest.get("field_types")
        manifest_rows = stream_manifest.get("rows")
        if (
            stream_manifest.get("format_version") != 1
            or stream_manifest.get("artifact")
            != f"shadow_derived_{name}"
            or stream_manifest.get("status") != "complete"
            or stream_manifest.get("stream") != name
            or stream_manifest.get("bindings")
            != {"source_stream_manifest": source_stream_manifest}
            or not isinstance(manifest_field_types, Mapping)
            or dict(manifest_field_types) != dict(field_types)
            or not isinstance(manifest_rows, int)
            or isinstance(manifest_rows, bool)
            or manifest_rows != entry_rows
        ):
            raise ValueError(
                f"complete Shadow derived stream manifest is invalid: {name}"
            )
        shards = stream_manifest.get("shards")
        if not isinstance(shards, list):
            raise ValueError(
                f"complete Shadow derived shard list is invalid: {name}"
            )
        stream_state = {
            "rows": manifest_rows,
            "next_shard_index": len(shards),
            "committed_shards": shards,
            "schema_fingerprint": stream_manifest.get(
                "schema_fingerprint"
            ),
            "field_types": dict(field_types),
        }
        verified_rows = verify_stream_shards(final, stream_state)
        if verified_rows != entry_rows:
            raise ValueError(
                f"complete Shadow derived rows are not conserved: {name}"
            )
    return dict(outputs)


def _finalize_shadow_outputs_impl(
    destination: Path,
    *,
    shadow_stream_state: Mapping[str, Any],
    source_stream_manifest: str,
    maximum_rows: int,
    include_details: bool,
    fail_after_batches: int = 0,
) -> dict[str, Any]:
    """Stream candidate shards into restartable, bounded derived shards."""

    import pyarrow.parquet as pq

    manifest_path = destination / "shadow_derived_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") == SHADOW_DERIVED_SCHEMA_VERSION
            and payload.get("status") == "complete"
            and payload.get("source_stream_manifest") == source_stream_manifest
            and payload.get("details_enabled") is include_details
        ):
            return _verify_complete_shadow_derived_outputs(
                destination,
                payload,
                source_stream_manifest=source_stream_manifest,
                include_details=include_details,
            )
        raise ValueError("existing Shadow derived manifest conflicts with run")

    work = destination / ".shadow_derived.tmp"
    final = destination / "shadow_derived"
    for path in (work, final):
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise ValueError("unsafe Shadow derived output path")
            shutil.rmtree(path)
    work.mkdir(parents=True)
    index_path = work / "shadow_index.sqlite3"

    definitions = _shadow_derived_definitions(include_details)
    streams = {
        name: new_stream_state(field_types)
        for name, (_, field_types) in definitions.items()
    }
    buffers: dict[str, list[dict[str, Any]]] = {
        name: [] for name in definitions
    }

    def emit(name: str, records: Any, *, final_flush: bool = False) -> None:
        if records:
            buffers[name].extend(item.to_dict() for item in records)
        while len(buffers[name]) >= maximum_rows or (
            final_flush and buffers[name]
        ):
            take = min(maximum_rows, len(buffers[name]))
            chunk = buffers[name][:take]
            del buffers[name][:take]
            key, field_types = definitions[name]
            write_stream_shards_bounded(
                work,
                name,
                chunk,
                streams[name],
                key_column=key,
                maximum_rows=maximum_rows,
                field_types=field_types,
            )

    counts = _new_shadow_derived_counts()
    motif_state: dict[tuple[str, ...], dict[str, Any]] = {}
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE seen_episode (
            episode_key TEXT PRIMARY KEY
        );
        CREATE TABLE root_choice (
            root_key TEXT PRIMARY KEY,
            geometry_incomplete INTEGER NOT NULL,
            observed_at_ns INTEGER NOT NULL,
            candidate_id TEXT NOT NULL
        );
        CREATE TABLE root_challenge (
            root_key TEXT NOT NULL,
            playbook TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (root_key, playbook),
            FOREIGN KEY (root_key) REFERENCES root_choice(root_key)
        );
        CREATE TABLE root_sequence_revision (
            root_key TEXT NOT NULL,
            playbook TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            observed_at_ns INTEGER NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (root_key, playbook, candidate_id),
            FOREIGN KEY (root_key) REFERENCES root_choice(root_key)
        );
        """
    )
    batches = 0
    try:
        for shard in shadow_stream_state.get("committed_shards", ()):
            parquet = pq.ParquetFile(destination / str(shard["path"]))
            for batch in parquet.iter_batches(batch_size=maximum_rows):
                batches += 1
                records = batch.to_pandas().to_dict(orient="records")
                episodes = derive_shadow_episode_outcomes(records)
                challenges = derive_shadow_mechanism_challenges(records)
                counts["raw_challenge_row_count"] += len(challenges)
                by_candidate: dict[str, list[ShadowMechanismChallengeRecord]] = {}
                for item in challenges:
                    by_candidate.setdefault(item.candidate_id, []).append(item)
                with connection:
                    for item in episodes:
                        try:
                            connection.execute(
                                "INSERT INTO seen_episode (episode_key) VALUES (?)",
                                (item.episode_key,),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(
                                "duplicate first-executable outcome across "
                                "Shadow shards"
                            ) from exc
                    for values in by_candidate.values():
                        if not values[0].market_thesis_root_id:
                            counts["unbound_challenge_row_count"] += len(values)
                            continue
                        _merge_shadow_root_selection_sqlite(
                            connection,
                            tuple(values),
                        )
                for item in episodes:
                    _count_shadow_episode(counts, item)
                emit("episode_outcomes", episodes)
                if include_details:
                    emit("mechanism_challenges", challenges)
                if fail_after_batches and batches >= fail_after_batches:
                    raise RuntimeError(
                        "intentional Shadow finalizer interruption"
                    )

        cursor = connection.execute(
            "SELECT root_key, playbook, payload FROM root_challenge "
            "ORDER BY root_key, playbook"
        )
        current_root_key: str | None = None
        current_values: list[ShadowMechanismChallengeRecord] = []

        def finalize_root() -> None:
            nonlocal current_values
            if not current_values:
                return
            root_rows = derive_shadow_root_episode_records(current_values)
            if len(root_rows) != 1:
                raise ValueError("Shadow root compaction did not yield one row")
            item = root_rows[0]
            for challenge in current_values:
                _count_representative_shadow_challenge(counts, challenge)
            emit("root_episode_challenges", root_rows)
            counts["root_count"] += 1
            counts["eligible_root_count"] += int(
                item.eligible_episode_evidence
            )
            motif_key = (
                item.market_mechanism,
                item.source_timeframe,
                item.authority_relation,
                item.event_order_signature,
                item.rejected_gates,
                item.target_R_bucket,
            )
            motif = motif_state.setdefault(
                motif_key,
                {
                    "count": 0,
                    "eligible_count": 0,
                    "dates": set(),
                    "eligible_dates": set(),
                    "root_keys": [] if include_details else None,
                },
            )
            motif["count"] += 1
            motif["eligible_count"] += int(item.eligible_episode_evidence)
            motif["dates"].add(item.session_date_ny)
            if item.eligible_episode_evidence:
                motif["eligible_dates"].add(item.session_date_ny)
            if include_details:
                root_keys = motif["root_keys"]
                if len(root_keys) < SHADOW_MOTIF_ROOT_SAMPLE_LIMIT:
                    root_keys.append(item.root_episode_key)
            current_values = []

        for root_key, _, payload in cursor:
            root_key = str(root_key)
            if current_root_key is not None and root_key != current_root_key:
                finalize_root()
            current_root_key = root_key
            current_values.append(_shadow_challenge_from_payload(str(payload)))
        finalize_root()

        sequence_cursor = connection.execute(
            "SELECT root_key, payload FROM root_sequence_revision "
            "ORDER BY root_key, observed_at_ns, candidate_id, playbook"
        )
        current_sequence_root: str | None = None
        current_sequence_values: list[ShadowMechanismChallengeRecord] = []

        def finalize_root_sequences() -> None:
            nonlocal current_sequence_values
            if not current_sequence_values:
                return
            sequence_rows = derive_shadow_root_sequence_records(
                current_sequence_values
            )
            for sequence_row in sequence_rows:
                _count_shadow_root_sequence(counts, sequence_row)
            emit("root_episode_sequences", sequence_rows)
            current_sequence_values = []

        for root_key, payload in sequence_cursor:
            root_key = str(root_key)
            if (
                current_sequence_root is not None
                and root_key != current_sequence_root
            ):
                finalize_root_sequences()
            current_sequence_root = root_key
            current_sequence_values.append(
                _shadow_challenge_from_payload(str(payload))
            )
        finalize_root_sequences()
        indexed_episode_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM seen_episode"
            ).fetchone()[0]
        )
        indexed_root_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM root_choice"
            ).fetchone()[0]
        )
        indexed_challenge_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM root_challenge"
            ).fetchone()[0]
        )
        indexed_sequence_revision_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM root_sequence_revision"
            ).fetchone()[0]
        )
        if indexed_episode_count != int(counts["episode_count"]):
            raise ValueError("Shadow episode SQLite index is not conserved")
        if indexed_root_count != int(counts["root_count"]):
            raise ValueError("Shadow root SQLite index is not conserved")
        if indexed_challenge_count != int(
            counts["representative_root_challenge_count"]
        ):
            raise ValueError("Shadow challenge SQLite index is not conserved")
        if indexed_sequence_revision_count != int(
            counts["raw_challenge_row_count"]
            - counts["unbound_challenge_row_count"]
        ):
            raise ValueError(
                "Shadow root sequence revision index is not conserved"
            )
    finally:
        connection.close()
        index_path.unlink(missing_ok=True)
        if sys.exc_info()[0] is not None and work.exists():
            shutil.rmtree(work)

    if include_details:
        motif_rows = []
        for key, value in sorted(motif_state.items()):
            eligible = (
                int(value["eligible_count"]) >= 2
                and len(value["eligible_dates"]) >= 2
            )
            import hashlib

            motif_rows.append(
                ShadowMechanismMotifRecord(
                    motif_id="shadow-motif:"
                    + hashlib.sha256(
                        json.dumps(key, separators=(",", ":")).encode()
                    ).hexdigest()[:24],
                    market_mechanism=key[0],
                    source_timeframe=key[1],
                    authority_relation=key[2],
                    event_order_signature=key[3],
                    rejected_gates=key[4],
                    target_R_bucket=key[5],
                    episode_count=int(value["count"]),
                    eligible_episode_count=int(value["eligible_count"]),
                    distinct_dates=len(value["dates"]),
                    eligible_distinct_dates=len(value["eligible_dates"]),
                    sample_root_episode_keys=json.dumps(
                        sorted(value["root_keys"]), separators=(",", ":")
                    ),
                    sample_root_episode_count=len(value["root_keys"]),
                    root_episode_keys_truncated=(
                        int(value["count"]) > len(value["root_keys"])
                    ),
                    eligible_for_preregistration_review=eligible,
                    action_authority=False,
                )
            )
        emit("mechanism_motifs", motif_rows)

    manifests: dict[str, str] = {}
    for name in definitions:
        emit(name, (), final_flush=True)
        path = write_stream_manifest(
            work,
            name,
            streams[name],
            artifact=f"shadow_derived_{name}",
            bindings={"source_stream_manifest": source_stream_manifest},
        )
        manifests[name] = str(Path("shadow_derived") / path.name)
    raw_challenges = int(counts["raw_challenge_row_count"])
    unbound_challenges = int(counts["unbound_challenge_row_count"])
    representative_challenges = int(
        counts["representative_root_challenge_count"]
    )
    if not 0 <= unbound_challenges <= raw_challenges:
        raise ValueError("Shadow unbound challenge count is not conserved")
    if not 0 <= representative_challenges <= (
        raw_challenges - unbound_challenges
    ):
        raise ValueError(
            "Shadow representative challenge count is not conserved"
        )
    if include_details and int(streams["mechanism_challenges"]["rows"]) != (
        raw_challenges
    ):
        raise ValueError("Shadow detail challenge rows are not conserved")
    if int(streams["root_episode_sequences"]["rows"]) != int(
        counts["root_sequence_unit_count"]
    ):
        raise ValueError("Shadow root sequence rows are not conserved")
    summary = _shadow_derived_summary(counts, motif_state)
    os.replace(work, final)
    outputs = {
        "manifest": manifest_path.name,
        "episode_outcomes": manifests["episode_outcomes"],
        "root_episode_challenges": manifests["root_episode_challenges"],
        "root_episode_sequences": manifests["root_episode_sequences"],
        "mechanism_challenges": manifests.get("mechanism_challenges"),
        "mechanism_motifs": manifests.get("mechanism_motifs"),
        **summary,
    }
    atomic_bytes(
        manifest_path,
        canonical_json(
            {
                "schema_version": SHADOW_DERIVED_SCHEMA_VERSION,
                "status": "complete",
                "source_stream_manifest": source_stream_manifest,
                "details_enabled": include_details,
                "information_boundaries": {
                    "root_episode_challenges": (
                        "first-geometry-complete representative may carry "
                        "later price outcomes"
                    ),
                    "root_episode_sequences": (
                        "candidate-clock lifecycle only; excludes target, "
                        "invalidation, MFE, MAE, path validity, and terminal "
                        "outcome clocks"
                    ),
                },
                "outputs": outputs,
                "streams": {
                    name: {
                        "manifest": path,
                        "rows": int(streams[name]["rows"]),
                        "field_types": dict(definitions[name][1]),
                        "action_authority": False,
                    }
                    for name, path in manifests.items()
                },
                "summary": summary,
                "model_feedback": "none_shadow_only",
            }
        ),
    )
    return outputs


def _finalize_shadow_outputs(
    destination: Path,
    *,
    shadow_stream_state: Mapping[str, Any],
    source_stream_manifest: str,
    maximum_rows: int,
    include_details: bool,
    fail_after_batches: int = 0,
) -> dict[str, Any]:
    """Finalize Shadow outputs and remove only its private failed workspace."""

    try:
        return _finalize_shadow_outputs_impl(
            destination,
            shadow_stream_state=shadow_stream_state,
            source_stream_manifest=source_stream_manifest,
            maximum_rows=maximum_rows,
            include_details=include_details,
            fail_after_batches=fail_after_batches,
        )
    except Exception:
        work = destination / ".shadow_derived.tmp"
        if work.exists():
            if work.is_symlink() or not work.is_dir():
                raise ValueError("unsafe Shadow derived temporary path")
            shutil.rmtree(work)
        raise


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
    selected = _resolve_hypothesis(
        snapshot.belief,
        best_variant.hypothesis_key,
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
    if args.diagnostic_fail_shadow_finalize_after_batches < 0:
        raise ValueError("Shadow finalizer diagnostic stop cannot be negative")
    market_case_input_enabled = bool(
        getattr(args, "market_case_input", False)
    )
    if market_case_input_enabled and any(
        (
            bool(args.brain_calibration),
            bool(args.calibration_only),
            bool(args.compact_scene_graph),
            bool(args.shadow_outcomes),
            bool(args.shadow_details),
            bool(args.causal_case_library),
            bool(args.brain_diagnostics),
            bool(args.include_unexplained_episode_details),
            bool(args.include_natural_funnel_details),
            bool(args.visualize_at),
            bool(args.simulate_execution),
            args.mbo_execution is not None,
            args.spread_points is not None,
            args.slippage_points is not None,
            bool(args.action_disabled_playbook),
            bool(args.diagnostic_fail_shadow_finalize_after_batches),
        )
    ):
        raise ValueError(
            "--market-case-input is exclusive with Brain calibration, "
            "diagnostics, Shadow, legacy cases, Decision output, "
            "visualization, playbook action policy and execution inputs"
        )
    effective_scene_graph_compaction = bool(
        args.compact_scene_graph or market_case_input_enabled
    )
    brain_diagnostics_enabled = bool(args.brain_diagnostics)
    if (
        args.include_unexplained_episode_details
        or args.include_natural_funnel_details
    ) and not brain_diagnostics_enabled:
        raise ValueError(
            "diagnostic detail output requires --brain-diagnostics"
        )
    if args.shadow_details and not args.shadow_outcomes:
        raise ValueError("--shadow-details requires --shadow-outcomes")
    if args.causal_case_library and not args.shadow_outcomes:
        raise ValueError("--causal-case-library requires --shadow-outcomes")
    if args.calibration_only and not args.brain_calibration:
        raise ValueError("--calibration-only requires --brain-calibration")
    if args.calibration_only and args.simulate_execution:
        raise ValueError(
            "--calibration-only cannot be combined with --simulate-execution"
        )
    if args.compact_scene_graph and not (
        args.brain_calibration and args.calibration_only
    ):
        raise ValueError(
            "--compact-scene-graph requires --brain-calibration "
            "--calibration-only"
        )
    if args.shadow_outcomes and (
        args.simulate_execution
        or args.mbo_execution is not None
        or args.spread_points is not None
    ):
        raise ValueError(
            "--shadow-outcomes is an OHLCV-only diagnostic and cannot use "
            "execution simulation, MBO or a constant spread"
        )
    if args.spread_points is not None and args.spread_points < 0:
        raise ValueError("constant research spread cannot be negative")
    streamed_slippage_points = (
        0.0 if args.slippage_points is None else float(args.slippage_points)
    )
    if streamed_slippage_points < 0:
        raise ValueError("constant research slippage cannot be negative")
    action_disabled_playbooks = normalize_action_disabled_playbooks(
        args.action_disabled_playbook
    )

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

    shadow_profile_match = (
        _load_shadow_diagnostic_profile(
            args.validation_protocol,
            start=start,
            end=end,
            warmup_days=int(args.warmup_days),
            action_disabled_playbooks=action_disabled_playbooks,
        )
        if args.shadow_outcomes
        else None
    )
    shadow_profile_name, shadow_profile = (
        (None, None)
        if shadow_profile_match is None
        else shadow_profile_match
    )
    market_input_profile_match = (
        _load_market_case_input_profile(
            args.market_case_profile_registry,
            start=start,
            end=end,
            warmup_days=int(args.warmup_days),
        )
        if market_case_input_enabled
        else None
    )
    market_input_profile_name, market_input_profile = (
        (None, None)
        if market_input_profile_match is None
        else market_input_profile_match
    )
    if (
        shadow_profile is not None
        and Path(args.config).resolve() != (ROOT / "configs/model.json").resolve()
    ):
        raise ValueError(
            "--shadow-outcomes requires the frozen configs/model.json"
        )
    if (
        market_input_profile is not None
        and Path(args.config).resolve() != (ROOT / "configs/model.json").resolve()
    ):
        raise ValueError(
            "--market-case-input requires the frozen configs/model.json"
        )
    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_ohlcv(start, end)
    if shadow_profile is not None and window.role != shadow_profile.get(
        "allowed_ohlcv_role"
    ):
        raise ValueError("shadow diagnostic window role does not match its profile")
    if market_input_profile is not None and window.role != (
        market_input_profile.get("allowed_ohlcv_role")
    ):
        raise ValueError(
            "market-case input window role does not match its profile"
        )
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
    market_source_contract: tuple[str, int] | None = None
    if market_case_input_enabled:
        contract_pairs = replay_frame.loc[
            :, ["symbol", "instrument_id"]
        ].drop_duplicates()
        if len(contract_pairs) != 1:
            raise ValueError(
                "--market-case-input requires exactly one distinct "
                "(symbol, instrument_id) across the complete replay frame "
                "including warmup"
            )
        pair = contract_pairs.iloc[0]
        market_source_contract = (
            str(pair["symbol"]),
            int(pair["instrument_id"]),
        )
        if market_input_profile is not None:
            _validate_market_input_replay_contract(
                market_input_profile,
                market_source_contract,
            )
    total_source_rows = int(len(replay_frame))
    last_completed_asof = pd.Timestamp(replay_frame.index[-1]) + pd.Timedelta(
        minutes=1
    )
    if last_completed_asof.tzinfo is None:
        raise ValueError("last completed source clock must be timezone-aware")

    config_source = Path(args.config)
    config_payload = json.loads(config_source.read_text(encoding="utf-8"))
    engine = ContinuousSMCEngine.from_config(
        config_source,
        runtime_mode="development",
        action_disabled_playbooks=action_disabled_playbooks,
    )
    if args.shadow_outcomes or market_case_input_enabled:
        # The production observer keeps the complete typed states but omits
        # their transport-only delta tuples.  Shadow candidate capture needs
        # those deltas to register neutral Eye clocks without snapshot scans;
        # enabling this flag does not change reducer, graph, Brain or action
        # semantics.
        engine.observer.config = replace(
            engine.observer.config,
            typed_transition_delta_transport=True,
        )
    config_identity = sha256_file(config_source)
    brain_calibration_enabled = bool(args.brain_calibration)
    decision_stream_enabled = not bool(
        args.calibration_only or market_case_input_enabled
    )
    if brain_calibration_enabled and window.role not in {
        "calibration",
        "brain_validation",
        "brain_calibration_trial",
    }:
        raise RuntimeError(
            "--brain-calibration requires a registered Brain calibration, "
            "validation or fixed diagnostic-trial window"
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

    stream_keys = dict(BASE_STREAM_KEYS) if decision_stream_enabled else {}
    if brain_calibration_enabled:
        stream_keys["brain_calibration_shards"] = "sample_id"
    if args.shadow_outcomes:
        stream_keys["shadow_outcome_shards"] = "candidate_id"
    if args.causal_case_library:
        stream_keys["causal_case_input_shards"] = "revision_id"
        stream_keys["causal_case_outcome_shards"] = "outcome_id"
    if market_case_input_enabled:
        stream_keys["market_case_input_shards"] = "revision_id"

    registry = engine.brain.registry
    runtime_action_policy_identity = dict(
        engine.runtime_action_policy_identity
    )
    brain_runtime_identity = {
        "runtime_state_schema_version": BRAIN_RUNTIME_STATE_SCHEMA_VERSION,
        "registry_fingerprint": registry.fingerprint,
        "registry_schema_version": registry.schema_version,
        "playbook_schema_versions": {
            playbook.value: registry.for_playbook(playbook).schema_version
            for playbook in Playbook
        },
        "runtime_action_policy": runtime_action_policy_identity,
    }

    brain_calibration_identity = None
    if brain_calibration_enabled:
        brain_calibration_identity = {
            "recorder_schema_version": (
                BRAIN_CALIBRATION_RECORDER_SCHEMA_VERSION
            ),
            "registry_fingerprint": registry.fingerprint,
            "registry_schema_version": registry.schema_version,
            "playbook_schema_versions": {
                playbook.value: registry.for_playbook(playbook).schema_version
                for playbook in sorted(
                    BRAIN_CALIBRATION_PLAYBOOKS,
                    key=lambda item: item.value,
                )
            },
        }

    shadow_outcome_identity = None
    if shadow_profile is not None:
        shadow_outcome_identity = {
            "profile_name": shadow_profile_name,
            "recorder_schema_version": shadow_profile[
                "recorder_schema_version"
            ],
            "derived_schema_version": shadow_profile[
                "derived_schema_version"
            ],
            "protocol_version": shadow_profile["protocol_version"],
        }

    causal_case_identity = None
    if args.causal_case_library:
        causal_case_identity = expected_causal_case_run_identity()

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
        "brain_runtime_identity": brain_runtime_identity,
        "runtime_action_policy_identity": runtime_action_policy_identity,
        **(
            {
                "brain_diagnostics_identity": {
                    "natural_funnel_schema_version": (
                        NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
                    )
                }
            }
            if brain_diagnostics_enabled
            else {}
        ),
        "brain_calibration_identity": brain_calibration_identity,
        "brain_calibration_fit_admission": (
            validation.brain_calibration_fit_admission.as_dict()
            if brain_calibration_enabled
            else None
        ),
        "shadow_outcome_identity": shadow_outcome_identity,
        **(
            {"causal_case_identity": causal_case_identity}
            if causal_case_identity is not None
            else {}
        ),
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
            "brain_diagnostics": brain_diagnostics_enabled,
            "shadow_outcomes": bool(args.shadow_outcomes),
            "shadow_derived_outputs": bool(args.shadow_outcomes),
            "shadow_details": bool(args.shadow_details),
            **(
                {"causal_case_library": True}
                if args.causal_case_library
                else {}
            ),
            "calibration_only": bool(args.calibration_only),
            "scene_graph_compaction": {
                "enabled": bool(args.compact_scene_graph),
                "cadence": "checkpoint",
                "terminal_context_grace_minutes": 60,
            },
            "decision_shards": decision_stream_enabled,
            "unexplained_episode_details": bool(
                args.include_unexplained_episode_details
            ),
            "natural_funnel_details": bool(
                args.include_natural_funnel_details
            ),
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
    if market_case_input_enabled:
        if market_input_profile_name is None or market_input_profile is None:
            raise AssertionError("market-case input profile was not bound")
        if market_source_contract is None:
            raise AssertionError("market-case input source contract was not bound")
        source_symbol, source_instrument_id = market_source_contract
        run_manifest = {
            "schema_version": 1,
            "runner": "continuous_replay",
            "mode": "market_case_input",
            "runtime_state_schema_version": (
                MARKET_CASE_INPUT_RUNTIME_STATE_SCHEMA_VERSION
            ),
            "data_continuity": dict(
                MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY
            ),
            "repository": _repository_commit_identity(),
            "profile": {
                "name": market_input_profile_name,
                "identity": hashlib.sha256(
                    canonical_json(to_primitive(market_input_profile))
                ).hexdigest(),
            },
            "profile_registry": {
                "path": str(
                    Path(args.market_case_profile_registry).resolve()
                ),
                "sha256": sha256_file(
                    Path(args.market_case_profile_registry)
                ),
                "schema_version": (
                    MARKET_CASE_PROFILE_REGISTRY_SCHEMA_VERSION
                ),
            },
            "source": {
                "path": str(source.resolve()),
                "sha256": source_hash,
                "rows": total_source_rows,
                "first": replay_frame.index[0].isoformat(),
                "last": replay_frame.index[-1].isoformat(),
                "last_completed_asof": last_completed_asof.isoformat(),
                "role": loaded.source_role,
                "symbol": source_symbol,
                "instrument_id": source_instrument_id,
            },
            "model_config": {
                "path": str(config_source.resolve()),
                "sha256": config_identity,
                "schema_version": config_payload.get("schema_version"),
                "tick_size": config_payload.get("tick_size"),
                "timezone": config_payload.get("timezone"),
            },
            "market_case_input_identity": (
                expected_market_case_run_identity()
            ),
            "window": {
                "start": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "role": window.role,
                "warmup_days": int(args.warmup_days),
                "observation_clock": "completed_1m_bar_end",
                "capture_interval": "[start,end_exclusive)",
            },
            "output": {
                "stream_families": ["market_case_input_shards"],
                "shard_rows": effective_shard_rows,
                "checkpoint_bars": int(args.checkpoint_bars),
            },
        }
    run_manifest_bytes = canonical_json(to_primitive(run_manifest))
    run_manifest_sha256 = hashlib.sha256(run_manifest_bytes).hexdigest()
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
        "shadow_outcome_shards": "shadow_outcome_rows",
        "causal_case_input_shards": "causal_case_input_rows",
        "causal_case_outcome_shards": "causal_case_outcome_rows",
        "market_case_input_shards": "market_case_input_rows",
    }

    def sync_checkpoint_aliases(state: dict[str, Any]) -> None:
        primary_name = next(iter(state["streams"]))
        primary_stream = state["streams"][primary_name]
        state["next_shard_index"] = int(
            primary_stream["next_shard_index"]
        )
        state["committed_shards"] = primary_stream["committed_shards"]

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

    def verify_runtime_action_policy_state(state: Mapping[str, Any]) -> None:
        if state.get("runtime_action_policy_identity") != (
            runtime_action_policy_identity
        ):
            raise ValueError("checkpoint runtime action policy changed")
        replay_state = state.get("replay")
        engine_state = getattr(replay_state, "engine", None)
        if dict(
            getattr(engine_state, "runtime_action_policy_identity", {})
        ) != runtime_action_policy_identity:
            raise ValueError("checkpoint Engine runtime action policy changed")

    market_checkpoint_keys = {
        "replay",
        "market_cases",
        "streams",
        "buffers",
        "processed_bars",
        "source_rows_consumed",
        "last_checkpoint_processed_bars",
        "decision_rows",
        "market_case_input_rows",
        "last_source_start",
        "last_asof",
        "resume_count",
        "finalized",
        "peak_buffer_rows",
        "next_shard_index",
        "committed_shards",
        "scene_graph_compaction",
    }

    def verify_market_input_checkpoint(state: Mapping[str, Any]) -> None:
        recorder = state.get("market_cases")
        if set(state) != market_checkpoint_keys:
            raise ValueError("market-case input checkpoint state changed")
        if not isinstance(recorder, MarketEpisodeCaseRecorder):
            raise ValueError("checkpoint market-case recorder is invalid")
        if getattr(recorder, "recorder_schema_version", None) != (
            MARKET_CASE_RECORDER_SCHEMA_VERSION
        ):
            raise ValueError("checkpoint market-case recorder schema changed")

    def verify_scene_graph_compaction_state(
        state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        compaction = state.get("scene_graph_compaction")
        if not effective_scene_graph_compaction:
            if compaction is not None:
                raise ValueError(
                    "disabled Scene Graph compaction checkpoint contains state"
                )
            return None
        expected_keys = {
            "runs",
            "last_processed_bars",
            "last_result",
            "peak_before_nodes",
            "peak_after_nodes",
        }
        if not isinstance(compaction, dict) or set(compaction) != expected_keys:
            raise ValueError("checkpoint Scene Graph compaction state is invalid")
        for name in (
            "runs",
            "last_processed_bars",
            "peak_before_nodes",
            "peak_after_nodes",
        ):
            if type(compaction[name]) is not int or int(compaction[name]) < 0:
                raise ValueError(
                    "checkpoint Scene Graph compaction state is invalid"
                )
        processed_bars = state.get("processed_bars")
        if (
            type(processed_bars) is not int
            or int(compaction["last_processed_bars"]) != processed_bars
        ):
            raise ValueError(
                "checkpoint Scene Graph compaction processed clock is invalid"
            )
        runs = int(compaction["runs"])
        result = compaction["last_result"]
        if runs == 0:
            if (
                processed_bars != 0
                or result is not None
                or int(compaction["peak_before_nodes"]) != 0
                or int(compaction["peak_after_nodes"]) != 0
            ):
                raise ValueError(
                    "checkpoint Scene Graph compaction state is invalid"
                )
            return compaction
        if not isinstance(result, Mapping) or set(result) != {
            "asof",
            "history_retention_floor",
            "revision_id",
            "before",
            "after",
        }:
            raise ValueError(
                "checkpoint Scene Graph compaction result is invalid"
            )
        count_keys = {
            "nodes",
            "node_revisions",
            "edges",
            "edge_revisions",
            "seen_events",
            "seen_displacement_transitions",
        }
        counts: dict[str, Mapping[str, Any]] = {}
        for phase in ("before", "after"):
            values = result[phase]
            if not isinstance(values, Mapping) or set(values) != count_keys:
                raise ValueError(
                    "checkpoint Scene Graph compaction result is invalid"
                )
            if any(
                type(values[name]) is not int or values[name] < 0
                for name in count_keys
            ):
                raise ValueError(
                    "checkpoint Scene Graph compaction result is invalid"
                )
            counts[phase] = values
        if any(
            int(counts["after"][name]) > int(counts["before"][name])
            for name in count_keys
        ):
            raise ValueError(
                "checkpoint Scene Graph compaction result is invalid"
            )
        asof = pd.Timestamp(result["asof"])
        floor = pd.Timestamp(result["history_retention_floor"])
        if (
            asof.tzinfo is None
            or floor.tzinfo is None
            or floor > asof
            or not isinstance(result["revision_id"], str)
            or not result["revision_id"]
            or int(compaction["peak_before_nodes"])
            < int(counts["before"]["nodes"])
            or int(compaction["peak_after_nodes"])
            < int(counts["after"]["nodes"])
        ):
            raise ValueError(
                "checkpoint Scene Graph compaction result is invalid"
            )
        replay_state = state.get("replay")
        engine_state = getattr(replay_state, "engine", None)
        snapshot = getattr(engine_state, "last_snapshot", None)
        graph = getattr(getattr(engine_state, "observer", None), "scene_graph", None)
        if (
            snapshot is None
            or snapshot.observation.asof != asof
            or (
                market_case_input_enabled
                and not isinstance(snapshot, NeutralEngineSnapshot)
            )
            or graph is None
            or graph.last_asof != asof
            or graph.history_retention_floor != floor
            or graph.revision_id != result["revision_id"]
        ):
            raise ValueError(
                "checkpoint Scene Graph compaction runtime binding is invalid"
            )
        actual_after = {
            "nodes": len(graph._nodes),
            "node_revisions": sum(
                len(values) for values in graph._node_revisions.values()
            ),
            "edges": len(graph._edges),
            "edge_revisions": sum(
                len(values) for values in graph._edge_revisions.values()
            ),
            "seen_events": len(graph._seen_market_event_ids),
            "seen_displacement_transitions": len(
                graph._seen_displacement_transition_ids
            ),
        }
        if actual_after != dict(counts["after"]):
            raise ValueError(
                "checkpoint Scene Graph compaction runtime counts are invalid"
            )
        return compaction

    if args.resume:
        if not checkpoint.exists:
            raise FileNotFoundError("resume requested without a valid checkpoint")
        state = checkpoint.load(
            expected_bindings=bindings,
            expected_replay_type=CalibrationSequentialReplay,
        )
        if market_case_input_enabled:
            verify_market_input_checkpoint(state)
        else:
            verify_runtime_action_policy_state(state)
        verify_state_streams(state)
        verify_scene_graph_compaction_state(state)
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
            "runtime_action_policy_identity": runtime_action_policy_identity,
            "brain_calibration": (
                BrainCalibrationRecorder()
                if brain_calibration_enabled
                else None
            ),
            "shadow_outcomes": (
                ShadowCandidateOutcomeRecorder()
                if args.shadow_outcomes
                else None
            ),
            **(
                {
                    "causal_cases": CausalCaseRecorder(
                        source_path=source,
                        source_sha256=source_hash,
                        source_role=loaded.source_role,
                        split_role=window.role,
                        capture_start=start,
                        model_versions={
                            "model_config_identity": config_identity,
                            "model_config_schema_version": config_payload.get(
                                "schema_version"
                            ),
                            "brain_runtime_identity": brain_runtime_identity,
                            "shadow_outcome_identity": shadow_outcome_identity,
                            "causal_case_recorder_schema_version": (
                                CAUSAL_CASE_RECORDER_SCHEMA_VERSION
                            ),
                        },
                    )
                }
                if args.causal_case_library
                else {}
            ),
            **(
                {
                    "market_cases": MarketEpisodeCaseRecorder(
                        capture_start=start,
                    )
                }
                if market_case_input_enabled
                else {}
            ),
            "streams": streams,
            "buffers": {name: [] for name in stream_keys},
            "processed_bars": 0,
            "source_rows_consumed": 0,
            "last_checkpoint_processed_bars": 0,
            "decision_rows": 0,
            "brain_calibration_rows": 0,
            "shadow_outcome_rows": 0,
            **(
                {
                    "causal_case_input_rows": 0,
                    "causal_case_outcome_rows": 0,
                }
                if args.causal_case_library
                else {}
            ),
            **(
                {
                    "market_case_input_rows": 0,
                }
                if market_case_input_enabled
                else {}
            ),
            "model_action_counts": {},
            "risk_action_counts": {},
            "entry_approvals": {},
            "filled_entries": {},
            "last_source_start": None,
            "last_asof": None,
            "resume_count": 0,
            "finalized": False,
            "visual_artifacts": {},
            "brain_diagnostics_enabled": brain_diagnostics_enabled,
            **(
                {
                    "scene_graph_compaction": {
                        "runs": 0,
                        "last_processed_bars": 0,
                        "last_result": None,
                        "peak_before_nodes": 0,
                        "peak_after_nodes": 0,
                    }
                }
                if effective_scene_graph_compaction
                else {}
            ),
            **(
                {
                    "natural_funnel_detail_mode": bool(
                        args.include_natural_funnel_details
                    ),
                    "natural_funnel_diagnostic_schema_version": (
                        NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
                    ),
                    **(
                        {
                            "natural_episode_funnel_records": {},
                            "natural_candidate_root_records": {},
                            "natural_provisional_episode_keys": {},
                            "natural_episode_key_aliases": {},
                        }
                        if args.include_natural_funnel_details
                        else {
                            "natural_funnel_compact_state": (
                                _new_compact_natural_funnel_state()
                            )
                        }
                    ),
                    "open_thesis_binding_records": {},
                    "open_thesis_binding_aggregate": {
                        "root_stage_counts": {},
                        "root_theses_observed": 0,
                        "match_counts": {},
                        "dispositions": {},
                        "failed_gates": {},
                        "match_opportunities_observed": 0,
                        "cases": {},
                    },
                    "unexplained_episode_detail_mode": bool(
                        args.include_unexplained_episode_details
                    ),
                    "unexplained_episode_summaries": {},
                    "active_unexplained_episode_ids": [],
                    "unexplained_episode_aggregate_counts": {},
                    "unexplained_episode_finalized_count": 0,
                }
                if brain_diagnostics_enabled
                else {}
            ),
            "peak_buffer_rows": {name: 0 for name in stream_keys},
            "next_shard_index": 0,
            "committed_shards": next(iter(streams.values()))[
                "committed_shards"
            ],
        }
        if market_case_input_enabled:
            state = {
                key: value
                for key, value in state.items()
                if key in market_checkpoint_keys
            }
    args._streamed_output_owned = True
    scene_graph_compaction_state = verify_scene_graph_compaction_state(state)
    checkpoint_diagnostics_enabled = state.get(
        "brain_diagnostics_enabled",
        False if market_case_input_enabled else None,
    )
    if checkpoint_diagnostics_enabled is not brain_diagnostics_enabled:
        raise ValueError("checkpoint Brain diagnostics mode changed")
    diagnostic_state_keys = {
        "natural_funnel_detail_mode",
        "natural_funnel_diagnostic_schema_version",
        "natural_episode_funnel_records",
        "natural_candidate_root_records",
        "natural_provisional_episode_keys",
        "natural_episode_key_aliases",
        "natural_funnel_compact_state",
        "open_thesis_binding_records",
        "open_thesis_binding_aggregate",
        "unexplained_episode_detail_mode",
        "unexplained_episode_summaries",
        "active_unexplained_episode_ids",
        "unexplained_episode_aggregate_counts",
        "unexplained_episode_finalized_count",
    }
    if brain_diagnostics_enabled:
        if state.get("natural_funnel_diagnostic_schema_version") != (
            NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
        ):
            raise ValueError(
                "checkpoint natural funnel diagnostic schema changed"
            )
        detail_mode = bool(args.include_natural_funnel_details)
        checkpoint_detail_mode = bool(
            state.setdefault("natural_funnel_detail_mode", detail_mode)
        )
        if checkpoint_detail_mode != detail_mode:
            raise ValueError("checkpoint natural funnel detail mode changed")
        if detail_mode:
            natural_episode_records = state.setdefault(
                "natural_episode_funnel_records",
                {},
            )
            natural_candidate_root_records = state.setdefault(
                "natural_candidate_root_records",
                {},
            )
            natural_provisional_episode_keys = state.setdefault(
                "natural_provisional_episode_keys",
                {},
            )
            natural_episode_key_aliases = state.setdefault(
                "natural_episode_key_aliases",
                {},
            )
            natural_state_valid = all(
                isinstance(value, dict)
                for value in (
                    natural_episode_records,
                    natural_candidate_root_records,
                    natural_provisional_episode_keys,
                    natural_episode_key_aliases,
                )
            )
        else:
            compact_natural_state = state.setdefault(
                "natural_funnel_compact_state",
                _new_compact_natural_funnel_state(),
            )
            natural_state_valid = isinstance(compact_natural_state, dict)
        open_thesis_binding_records = state.setdefault(
            "open_thesis_binding_records",
            {},
        )
        open_thesis_binding_aggregate = state.setdefault(
            "open_thesis_binding_aggregate",
            {
                "root_stage_counts": {},
                "root_theses_observed": 0,
                "match_counts": {},
                "dispositions": {},
                "failed_gates": {},
                "match_opportunities_observed": 0,
                "cases": {},
            },
        )
        unexplained_episode_summaries = state.setdefault(
            "unexplained_episode_summaries",
            {},
        )
        active_unexplained_episode_ids = state.setdefault(
            "active_unexplained_episode_ids",
            [],
        )
        unexplained_detail_mode = bool(
            state.setdefault(
                "unexplained_episode_detail_mode",
                bool(args.include_unexplained_episode_details),
            )
        )
        unexplained_aggregate_counts = state.setdefault(
            "unexplained_episode_aggregate_counts",
            {},
        )
        unexplained_finalized_count = state.setdefault(
            "unexplained_episode_finalized_count",
            0,
        )
        if (
            not natural_state_valid
            or not isinstance(open_thesis_binding_records, dict)
            or not isinstance(open_thesis_binding_aggregate, dict)
            or not isinstance(unexplained_episode_summaries, dict)
            or not isinstance(active_unexplained_episode_ids, list)
            or unexplained_detail_mode
            != bool(args.include_unexplained_episode_details)
            or not isinstance(unexplained_aggregate_counts, dict)
            or not isinstance(unexplained_finalized_count, int)
        ):
            raise ValueError("checkpoint episode diagnostic state is invalid")
    elif diagnostic_state_keys.intersection(state):
        raise ValueError("disabled Brain diagnostics checkpoint contains state")

    if not market_case_input_enabled:
        verify_runtime_action_policy_state(state)
    replay: CalibrationSequentialReplay = state["replay"]
    brain_calibration: BrainCalibrationRecorder | None = state.get(
        "brain_calibration"
    )
    shadow_outcomes: ShadowCandidateOutcomeRecorder | None = state.get(
        "shadow_outcomes"
    )
    causal_cases: CausalCaseRecorder | None = state.get("causal_cases")
    market_cases: MarketEpisodeCaseRecorder | None = state.get("market_cases")
    buffers: dict[str, list[dict[str, Any]]] = state["buffers"]
    if (brain_calibration is not None) != brain_calibration_enabled:
        raise ValueError("checkpoint Brain calibration capture mode changed")
    if (
        brain_calibration is not None
        and not isinstance(brain_calibration, BrainCalibrationRecorder)
    ):
        raise ValueError("checkpoint Brain calibration recorder is invalid")
    if (shadow_outcomes is not None) != bool(args.shadow_outcomes):
        raise ValueError("checkpoint shadow-outcome capture mode changed")
    if (
        shadow_outcomes is not None
        and not isinstance(
            shadow_outcomes,
            ShadowCandidateOutcomeRecorder,
        )
    ):
        raise ValueError("checkpoint shadow-outcome recorder is invalid")
    if (
        shadow_outcomes is not None
        and getattr(shadow_outcomes, "recorder_schema_version", None)
        != SHADOW_OUTCOME_RECORDER_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint shadow-outcome recorder schema changed")
    if ("causal_cases" in state) != bool(args.causal_case_library) or (
        causal_cases is not None
    ) != bool(args.causal_case_library):
        raise ValueError("checkpoint causal-case capture mode changed")
    if causal_cases is not None and not isinstance(causal_cases, CausalCaseRecorder):
        raise ValueError("checkpoint causal-case recorder is invalid")
    if (
        causal_cases is not None
        and getattr(causal_cases, "recorder_schema_version", None)
        != CAUSAL_CASE_RECORDER_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint causal-case recorder schema changed")
    if not args.causal_case_library and {
        "causal_case_input_rows",
        "causal_case_outcome_rows",
    }.intersection(state):
        raise ValueError("disabled causal-case checkpoint contains state")
    if ("market_cases" in state) != market_case_input_enabled or (
        market_cases is not None
    ) != market_case_input_enabled:
        raise ValueError("checkpoint market-case capture mode changed")
    if market_cases is not None and not isinstance(
        market_cases,
        MarketEpisodeCaseRecorder,
    ):
        raise ValueError("checkpoint market-case recorder is invalid")
    if (
        market_cases is not None
        and getattr(market_cases, "recorder_schema_version", None)
        != MARKET_CASE_RECORDER_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint market-case recorder schema changed")
    if not market_case_input_enabled and {
        "market_case_input_rows",
    }.intersection(state):
        raise ValueError("disabled market-case checkpoint contains state")
    if bool(replay.simulate_execution) != bool(args.simulate_execution):
        raise ValueError("checkpoint execution-simulation mode changed")
    if state["finalized"] and int(state["source_rows_consumed"]) != total_source_rows:
        raise ValueError("finalized checkpoint did not consume the bound source")

    visual_artifacts = state.get(
        "visual_artifacts",
        {} if market_case_input_enabled else None,
    )
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
    last_heartbeat_at = time.monotonic()
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
        nonlocal durable_progress, safe_source_checkpoint, last_heartbeat_at
        if state["last_source_start"] is None:
            return
        was_safe_source_checkpoint = safe_source_checkpoint
        safe_source_checkpoint = False
        for name in stream_keys:
            flush_stream(name, final=final)
        if effective_scene_graph_compaction:
            assert isinstance(scene_graph_compaction_state, dict)
            processed_bars = int(state["processed_bars"])
            if processed_bars > int(
                scene_graph_compaction_state["last_processed_bars"]
            ):
                result = replay.engine.compact_scene_graph_runtime()
                scene_graph_compaction_state["runs"] = (
                    int(scene_graph_compaction_state["runs"]) + 1
                )
                scene_graph_compaction_state[
                    "last_processed_bars"
                ] = processed_bars
                scene_graph_compaction_state["last_result"] = dict(result)
                scene_graph_compaction_state["peak_before_nodes"] = max(
                    int(
                        scene_graph_compaction_state[
                            "peak_before_nodes"
                        ]
                    ),
                    int(result["before"]["nodes"]),
                )
                scene_graph_compaction_state["peak_after_nodes"] = max(
                    int(
                        scene_graph_compaction_state[
                            "peak_after_nodes"
                        ]
                    ),
                    int(result["after"]["nodes"]),
                )
        state["last_checkpoint_processed_bars"] = int(
            state["processed_bars"]
        )
        sync_checkpoint_aliases(state)
        verify_scene_graph_compaction_state(state)
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
        last_heartbeat_at = time.monotonic()
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

    def drain_shadow_records() -> None:
        if shadow_outcomes is None:
            return
        records = shadow_outcomes.drain_rows()
        if causal_cases is not None:
            causal_cases.consume_shadow_records(records)
            append_stream(
                "causal_case_outcome_shards",
                [
                    _stream_record(item)
                    for item in causal_cases.drain_outcome_rows()
                ],
            )
        append_stream(
            "shadow_outcome_shards",
            [_stream_record(item) for item in records],
        )

    iterator = iter_after_source_checkpoint(
        replay_frame,
        state["last_source_start"],
        maximum_no_trade_gap_minutes=(
            int(
                MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY[
                    "maximum_no_trade_gap_minutes"
                ]
            )
            if market_case_input_enabled
            else 5
        ),
        allow_data_gap_reset=(
            market_case_input_enabled
            and bool(
                MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY[
                    "allow_same_contract_data_gap_reset"
                ]
            )
        ),
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
                if shadow_outcomes is not None:
                    shadow_outcomes.on_bar(bar)
                if market_case_input_enabled:
                    step = None
                    snapshot = replay.engine.on_bar_neutral_input(bar)
                else:
                    step = replay.on_bar(
                        bar,
                        execution=execution_for_bar(bar),
                    )
                    snapshot = step.snapshot
                if (
                    brain_calibration is not None
                    and snapshot.observation.asof < start
                ):
                    brain_calibration.prime(
                        snapshot,
                        source_bar=bar,
                    )
                if (
                    causal_cases is not None
                    and snapshot.observation.asof < start
                ):
                    causal_cases.prime(
                        snapshot,
                        source_bar=bar,
                        source_row_ordinal=int(state["source_rows_consumed"]),
                        replay_update_ordinal=int(state["processed_bars"]),
                        scene_graph=replay.engine.observer.scene_graph,
                    )
                if (
                    market_cases is not None
                    and snapshot.observation.asof < start
                ):
                    market_cases.prime(
                        snapshot,
                        source_bar=bar,
                        source_row_ordinal=int(state["source_rows_consumed"]),
                        replay_update_ordinal=int(state["processed_bars"]),
                        scene_graph=replay.engine.observer.scene_graph,
                    )
                if (
                    shadow_outcomes is not None
                    and snapshot.observation.asof < start
                ):
                    shadow_outcomes.prime(
                        snapshot,
                        source_bar=bar,
                    )
                if args.simulate_execution:
                    assert not market_case_input_enabled and step is not None
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
                    if brain_diagnostics_enabled:
                        assert not market_case_input_enabled and step is not None
                        _update_natural_episode_funnel(
                            state,
                            snapshot,
                            step=step,
                            retain_details=bool(
                                args.include_natural_funnel_details
                            ),
                        )
                        _update_open_thesis_binding_funnel(
                            state,
                            snapshot,
                            action_playbooks=BRAIN_CALIBRATION_PLAYBOOKS,
                        )
                        _compact_open_thesis_binding_records(
                            state,
                            snapshot,
                        )
                        _update_unexplained_episode_summaries(
                            state,
                            snapshot,
                        )
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
                    if causal_cases is not None:
                        causal_cases.observe(
                            snapshot,
                            source_bar=bar,
                            source_row_ordinal=int(state["source_rows_consumed"]),
                            replay_update_ordinal=int(state["processed_bars"]),
                            scene_graph=replay.engine.observer.scene_graph,
                        )
                        append_stream(
                            "causal_case_input_shards",
                            [
                                _stream_record(item)
                                for item in causal_cases.drain_input_rows()
                            ],
                        )
                    if market_cases is not None:
                        market_cases.observe(
                            snapshot,
                            source_bar=bar,
                            source_row_ordinal=int(
                                state["source_rows_consumed"]
                            ),
                            replay_update_ordinal=int(state["processed_bars"]),
                            scene_graph=replay.engine.observer.scene_graph,
                        )
                        append_stream(
                            "market_case_input_shards",
                            [
                                _stream_record(item)
                                for item in market_cases.drain_input_rows()
                            ],
                        )
                    if shadow_outcomes is not None:
                        shadow_outcomes.observe(
                            snapshot,
                            source_bar=bar,
                        )
                        drain_shadow_records()
                    if decision_stream_enabled:
                        assert not market_case_input_enabled and step is not None
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
                    elif not market_case_input_enabled:
                        state["decision_rows"] = int(
                            state["decision_rows"]
                        ) + 1
                    if not market_case_input_enabled:
                        model_action = snapshot.decision.selected_action.value
                        risk_action = snapshot.risk.final_action.value
                        state["model_action_counts"][model_action] = (
                            int(
                                state["model_action_counts"].get(
                                    model_action,
                                    0,
                                )
                            )
                            + 1
                        )
                        state["risk_action_counts"][risk_action] = (
                            int(
                                state["risk_action_counts"].get(
                                    risk_action,
                                    0,
                                )
                            )
                            + 1
                        )
                    state["last_asof"] = snapshot.observation.asof
                    if (
                        args.simulate_execution
                        and snapshot.risk.final_action is Action.ENTER
                        and snapshot.risk.frozen_thesis is not None
                    ):
                        assert not market_case_input_enabled and step is not None
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
                elif (
                    int(state["processed_bars"]) % 5_000 == 0
                    or time.monotonic() - last_heartbeat_at >= 60.0
                ):
                    heartbeat = _stream_progress(
                        state,
                        total_source_rows=total_source_rows,
                        session_started=session_started,
                        session_source_start=session_source_start,
                    )
                    atomic_bytes(
                        progress_path,
                        canonical_json(
                            {
                                **heartbeat,
                                "status": "running",
                                "resume_supported": checkpoint.exists,
                                "durable_checkpoint_only": False,
                                "last_durable_checkpoint_processed_bars": int(
                                    durable_progress["processed_bars"]
                                ),
                                "last_durable_checkpoint_source_rows": int(
                                    durable_progress[
                                        "source_rows_processed"
                                    ]
                                ),
                            }
                        ),
                    )
                    last_heartbeat_at = time.monotonic()
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

    if state["last_asof"] is None or (
        not market_case_input_enabled
        and int(state["decision_rows"]) == 0
    ):
        raise ValueError("requested interval produced no completed decisions")
    if int(state["source_rows_consumed"]) != total_source_rows:
        raise RuntimeError(
            "source iterator ended before all bound rows were consumed"
        )
    if market_case_input_enabled and pd.Timestamp(state["last_asof"]) != (
        last_completed_asof
    ):
        raise ValueError(
            "market-case input clock differs from the final completed "
            "1m bar"
        )
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
        if shadow_outcomes is not None:
            shadow_outcomes.close_unresolved(state["last_asof"])
            drain_shadow_records()
        if causal_cases is not None:
            causal_cases.close_unresolved(state["last_asof"])
            append_stream(
                "causal_case_input_shards",
                [
                    _stream_record(item)
                    for item in causal_cases.drain_input_rows()
                ],
            )
            append_stream(
                "causal_case_outcome_shards",
                [
                    _stream_record(item)
                    for item in causal_cases.drain_outcome_rows()
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
            bindings=(
                {
                    **bindings,
                    "run_manifest_sha256": run_manifest_sha256,
                }
                if name == "market_case_input_shards"
                else bindings
            ),
        )
        stream_manifests[name] = str(manifest_path.relative_to(destination))

    causal_case_library_manifest: str | None = None
    if causal_cases is not None:
        case_manifest_path = write_causal_case_library_manifest(
            destination,
            input_stream_manifest=stream_manifests[
                "causal_case_input_shards"
            ],
            outcome_stream_manifest=stream_manifests[
                "causal_case_outcome_shards"
            ],
            run_manifest=run_manifest_path.name,
        )
        causal_case_library_manifest = str(
            case_manifest_path.relative_to(destination)
        )

    shadow_derived_outputs: dict[str, Any] | None = None
    if shadow_outcomes is not None:
        shadow_derived_outputs = _finalize_shadow_outputs(
            destination,
            shadow_stream_state=state["streams"]["shadow_outcome_shards"],
            source_stream_manifest=stream_manifests["shadow_outcome_shards"],
            maximum_rows=effective_shard_rows,
            include_details=bool(args.shadow_details),
            fail_after_batches=int(
                args.diagnostic_fail_shadow_finalize_after_batches
            ),
        )

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

    unexplained_episode_details_path: Path | None = None
    if args.include_unexplained_episode_details:
        unexplained_episode_details_path = (
            destination / "unexplained_episodes.json"
        )
        atomic_bytes(
            unexplained_episode_details_path,
            canonical_json(
                to_primitive(
                    _unexplained_episode_details_payload(
                        _unexplained_episode_summary(
                            state,
                            end_asof=state["last_asof"],
                        )
                    )
                )
            ),
        )

    natural_funnel_details_manifest: str | None = None
    natural_funnel_details: dict[str, Any] | None = None
    if args.include_natural_funnel_details:
        natural_funnel_details = _natural_episode_funnel_summary(
            state,
            end_asof=state["last_asof"],
            include_details=True,
        )
        natural_funnel_details_manifest = _write_natural_funnel_diagnostics(
            destination,
            natural_funnel_details,
            maximum_rows=effective_shard_rows,
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
        "decision_rows": int(state.get("decision_rows", 0)),
        "brain_calibration_capture": brain_calibration_enabled,
        "brain_diagnostics": brain_diagnostics_enabled,
        "runtime_action_policy_identity": runtime_action_policy_identity,
        "brain_calibration_rows": int(
            state.get("brain_calibration_rows", 0)
        ),
        "brain_calibration_late_registration": (
            None
            if brain_calibration is None
            else dict(brain_calibration.late_registration_summary)
        ),
        "shadow_outcomes": (
            None
            if shadow_outcomes is None
            else {
                **dict(shadow_outcomes.summary),
                "output_affects_model": False,
                "future_visible_to_model": False,
            }
        ),
        "shadow_derived_outputs": shadow_derived_outputs,
        **(
            {
                "causal_case_library": {
                    **dict(causal_cases.summary),
                    "manifest": causal_case_library_manifest,
                    "input_stream": stream_manifests[
                        "causal_case_input_shards"
                    ],
                    "future_outcome_stream": stream_manifests[
                        "causal_case_outcome_shards"
                    ],
                    "future_visible_to_input": False,
                    "output_affects_model": False,
                }
            }
            if causal_cases is not None
            else {}
        ),
        "model_action_counts": dict(
            sorted(state.get("model_action_counts", {}).items())
        ),
        "action_counts": dict(
            sorted(state.get("risk_action_counts", {}).items())
        ),
        "natural_episode_funnel": (
            _natural_episode_funnel_summary(
                state,
                end_asof=state["last_asof"],
            )
            if brain_diagnostics_enabled
            else None
        ),
        "natural_funnel_details": natural_funnel_details_manifest,
        "open_thesis_binding_funnel": (
            _open_thesis_binding_funnel_summary(state)
            if brain_diagnostics_enabled
            else None
        ),
        "unexplained_episode_aggregate": (
            _unexplained_episode_aggregate(
                state,
                end_asof=state["last_asof"],
            )
            if brain_diagnostics_enabled
            else None
        ),
        "unexplained_episode_details": (
            None
            if unexplained_episode_details_path is None
            else unexplained_episode_details_path.name
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
        "checkpoint_resume_supported": False,
        "output_contract": (
            "entry_episode_case_inputs_and_separate_future_labels"
            if causal_cases is not None
            else "calibration_rows_shadow_outcomes_and_aggregates"
            if args.calibration_only and args.shadow_outcomes
            else "calibration_rows_and_aggregates"
            if args.calibration_only
            else "lightweight_decision_calibration_and_shadow_shards"
            if brain_calibration_enabled and args.shadow_outcomes
            else "lightweight_decision_and_shadow_shards"
            if args.shadow_outcomes
            else "lightweight_decision_and_calibration_shards"
            if brain_calibration_enabled
            else "lightweight_decision_shards"
        ),
        "resume_count": int(state["resume_count"]),
        "shard_rows": effective_shard_rows,
        "checkpoint_bars": int(args.checkpoint_bars),
        "scene_graph_compaction": (
            None
            if scene_graph_compaction_state is None
            else to_primitive(scene_graph_compaction_state)
        ),
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
    if market_case_input_enabled:
        if market_cases is None or market_input_profile_name is None:
            raise AssertionError("market-case input recorder was not bound")
        summary = {
            "schema_version": 1,
            "mode": "market_case_input",
            "run_manifest": run_manifest_path.name,
            "run_manifest_sha256": run_manifest_sha256,
            "source_rows_processed": int(state["source_rows_consumed"]),
            "source_rows_total": total_source_rows,
            "processed_bars": int(state["processed_bars"]),
            "market_case_input": {
                **dict(market_cases.summary),
                "stream_manifest": stream_manifests[
                    "market_case_input_shards"
                ],
                "future_visible": False,
                "outcome_joined": False,
                "output_affects_model": False,
            },
            "resume_count": int(state["resume_count"]),
            "shard_rows": effective_shard_rows,
            "checkpoint_bars": int(args.checkpoint_bars),
            "stream_rows": {
                "market_case_input_shards": int(
                    state["streams"]["market_case_input_shards"]["rows"]
                )
            },
            "peak_buffer_rows": dict(state["peak_buffer_rows"]),
            "scene_graph_compaction": to_primitive(
                scene_graph_compaction_state
            ),
            "checkpoint_resume_supported": False,
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
    if market_case_input_enabled:
        completed_payload = {
            "schema_version": 1,
            "status": "complete",
            "run_manifest": run_manifest_path.name,
            "run_manifest_sha256": run_manifest_sha256,
            "summary": summary_path.name,
            "progress": progress_path.name,
            "market_case_input_shards": stream_manifests[
                "market_case_input_shards"
            ],
        }
    else:
        completed_payload = {
            "schema_version": 1,
            "status": "complete",
            "run_manifest": run_manifest_path.name,
            "summary": summary_path.name,
            "progress": progress_path.name,
            "stream_manifests": stream_manifests,
            **(
                {"causal_case_library": causal_case_library_manifest}
                if causal_case_library_manifest is not None
                else {}
            ),
            **(
                {"shadow_derived_outputs": shadow_derived_outputs}
                if shadow_derived_outputs is not None
                else {}
            ),
            "trades": "trades.parquet" if args.simulate_execution else None,
            "entry_attempts": (
                "entry_attempts.parquet" if args.simulate_execution else None
            ),
            "visualizations_index": visualization_index,
            "unexplained_episode_details": (
                None
                if unexplained_episode_details_path is None
                else unexplained_episode_details_path.name
            ),
            "natural_funnel_details": natural_funnel_details_manifest,
        }
    atomic_bytes(completed_path, canonical_json(completed_payload))
    checkpoint.retire_completed()
    print(json.dumps(summary.get("action_counts", {}), sort_keys=True))


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
