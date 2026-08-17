"""Event-level causal targets for the typed playbook brain.

This module deliberately does not fit a model.  It freezes a small number of
typed belief observations, then resolves them from later completed 1m OHLCV.
It is designed to be called by the existing replay loop::

    recorder.on_bar(bar)        # before the engine consumes this bar
    snapshot = replay.on_bar(...)
    recorder.observe(snapshot)  # after the completed-bar snapshot exists

Only DFP and LSR are observed.  FAVR remains parked.  Sequence progress and
uncertainty are descriptive rows; the other four dimensions carry causal
future outcomes suitable for a separate, simple calibration fitter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping

import pandas as pd

from .market_clock import MARKET_TIMEZONE, special_session_close
from .model import (
    Bar,
    Direction,
    EngineSnapshot,
    EntryLocationLifecycle,
    Playbook,
    PlaybookPhase,
    StructureLifecycle,
    Timeframe,
    aware_timestamp,
)


# Schema 19 admits fitted targets only after Brain has published an explicit
# Context Thesis owner (and, for entry dimensions, an exact child Episode and
# path).  Schema 18 could promote DFP's evaluator-native ``context_id`` or
# crash while priming an identity-incomplete diagnostic root.  Mixing those
# admissions across a resume boundary would change the evidence population,
# so every earlier checkpoint fails closed.
RECORDER_SCHEMA_VERSION = 19

SUPPORTED_PLAYBOOKS = frozenset(
    {
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    }
)
FITTED_DIMENSIONS = frozenset(
    {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
)
DESCRIPTIVE_DIMENSIONS = frozenset(
    {"sequence_progress", "uncertainty"}
)
ALL_DIMENSIONS = FITTED_DIMENSIONS | DESCRIPTIVE_DIMENSIONS
_MARKET_THESIS_MATCH_STATUSES = frozenset(
    {
        "not_required",
        "no_open_thesis",
        "no_direction_match",
        "no_mechanism_match",
        "root_identity_unbound",
        "exact_root_bound",
    }
)

_BOUNDARY_ANOMALIES = frozenset(
    {
        "contract_change_history_reset",
        "data_gap_history_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)

_DFP_THESIS_FAILURE_REASONS = frozenset(
    {
        "opposed_structure",
        "global_authority_invalidated",
    }
)
_DFP_LOCAL_EPISODE_TERMINAL_REASONS = frozenset(
    {
        "frozen_entry_location_missing",
        "entry_zone_left_or_failed",
        "trigger_opposed",
        "entry_window_expired",
        "frozen_invalidation_breached",
        "entry_zone_beyond_frozen_invalidation",
        "primary_target_consumed_or_missing",
        "position_invalidation_breached",
        "position_invalidated",
    }
)
_LSR_CONTEXT_THESIS_FAILURE_REASONS = frozenset(
    {
        "accepted_outside_or_failed",
        "frozen_pool_path_missing",
        "source_manipulation_missing",
        "context_structural_invalidation_breached",
        "context_frozen_source_invalidated",
        "frozen_invalidation_breached",
        "position_invalidation_breached",
        "position_invalidated",
        "global_authority_invalidated",
    }
)
_READINESS_FAILURE_REASONS = frozenset(
    {
        "trigger_opposed",
        "entry_trigger_contradicted",
        "entry_zone_left_or_failed",
        "entry_zone_left",
        "micro_bos_opposed",
        "mss_confirmed_after_first_pullback",
    }
)
_CENSOR_TERMINAL_REASONS = frozenset(
    {
        "entry_path_censored",
        "pool_path_censored",
        "manipulation_resolution_deadline",
        "opposite_displacement_ambiguous_same_clock",
        "micro_bos_ambiguous_same_clock",
        "data_gap_reset",
        "contract_change_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
_INVALID_TRIGGER_TERMINAL_REASONS = (
    _DFP_THESIS_FAILURE_REASONS
    | _LSR_CONTEXT_THESIS_FAILURE_REASONS
    | _READINESS_FAILURE_REASONS
    | _CENSOR_TERMINAL_REASONS
    | frozenset(
        {
            "episode_deadline_elapsed",
            "deadline_elapsed",
            "entry_window_expired",
            "frozen_entry_location_missing",
            "entry_zone_beyond_frozen_invalidation",
            "selected_draw_consumed_or_missing",
        }
    )
)


def _registered_session_deadline(asof: pd.Timestamp) -> pd.Timestamp:
    """Freeze the market-session horizon used by thesis calibration.

    This is deliberately independent of the execution observation.  A DFP
    Context Thesis may remain alive beyond the horizon; the recorder only
    evaluates one preregistered causal window and never writes this clock
    back into Brain state.
    """

    local = aware_timestamp(
        asof,
        name="brain_calibration.session_deadline_asof",
    ).tz_convert(MARKET_TIMEZONE)
    session_day = local.tz_localize(None).normalize()
    if (local.hour, local.minute) >= (18, 0):
        session_day += pd.Timedelta(days=1)
    session_marker = session_day.tz_localize(
        MARKET_TIMEZONE,
        ambiguous=True,
        nonexistent="shift_forward",
    )
    special_close = special_session_close(session_marker)
    if special_close is not None:
        return special_close
    return (session_day + pd.Timedelta(hours=17)).tz_localize(
        MARKET_TIMEZONE,
        ambiguous=True,
        nonexistent="shift_forward",
    )


def _encode_identities(values: Any) -> str:
    identities = tuple(dict.fromkeys(str(item) for item in values))
    if any(not item for item in identities):
        raise ValueError("identity lists cannot contain empty values")
    return json.dumps(list(identities), separators=(",", ":"))


def _validate_identity_list(value: str, *, name: str) -> None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a JSON identity list") from exc
    if (
        not isinstance(parsed, list)
        or any(not isinstance(item, str) or not item for item in parsed)
        or len(parsed) != len(set(parsed))
    ):
        raise ValueError(f"{name} identities are invalid")


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.isoformat()


def _timestamp(value: Any, *, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    return aware_timestamp(value, name=name)


def _action_candidate_items(
    belief: Any,
) -> tuple[tuple[str, Any], ...]:
    """Return stable root-candidate identities, with graph-free compatibility."""

    selector = getattr(belief, "action_candidate_items", None)
    if callable(selector):
        items = tuple(selector())
    else:
        # Compatibility for legacy/global-context-free recorder fixtures.
        # Production MarketBelief always takes the branch above.
        items = tuple(
            (hypothesis.key, hypothesis)
            for hypothesis in belief.candidates()
        )
    candidate_ids = tuple(candidate_id for candidate_id, _ in items)
    if any(
        not isinstance(candidate_id, str) or not candidate_id
        for candidate_id in candidate_ids
    ) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Brain action candidates have invalid identities")
    return items


def _lifecycle_candidate_items(
    belief: Any,
    *,
    action_items: tuple[tuple[str, Any], ...],
) -> tuple[tuple[str, Any], ...]:
    """Return the broader resolution-only candidate projection.

    Older fixtures and graph-free compatibility beliefs do not expose this
    contract, so their action projection remains the only available lifecycle
    view.  Production MarketBelief includes current action candidates plus
    retained/dormant and position-management identities.
    """

    selector = getattr(belief, "lifecycle_candidate_items", None)
    if not callable(selector):
        return action_items
    items = tuple(selector())
    if not items and action_items:
        # Graph-free unit-test helpers may intentionally override only the
        # action interface while inheriting MarketBelief's empty-map
        # lifecycle implementation.  Production MarketBelief derives both
        # views from the same non-empty thesis map, so this compatibility
        # fallback cannot promote a dormant production candidate.
        return action_items
    candidate_ids = tuple(candidate_id for candidate_id, _ in items)
    if any(
        not isinstance(candidate_id, str) or not candidate_id
        for candidate_id in candidate_ids
    ) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Brain lifecycle candidates have invalid identities")
    action_ids = {candidate_id for candidate_id, _ in action_items}
    if not action_ids.issubset(candidate_ids):
        raise ValueError(
            "Brain lifecycle candidates omit a current action candidate"
        )
    return items


def _clamped(value: Any, *, name: str) -> float:
    if type(value) is bool:
        raise ValueError(f"{name} must be numeric, not boolean")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return parsed


def _enum_text(value: Any, default: str) -> str:
    """Return a compact enum/string value without serializing graph payloads."""

    if value is None:
        return default
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    return text or default


def _context_feature(
    metadata: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    value = metadata.get(name, default)
    return default if value in (None, "") else value


def _optional_numeric(value: Any) -> Any | None:
    if value is None:
        return None
    if type(value) is bool:
        raise ValueError("calibration numeric metadata cannot be boolean")
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "unknown",
        "none",
        "null",
        "not_evaluated",
    }:
        return None
    return value


def _strict_metadata_bool(value: Any, *, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ValueError(f"{name} must be a strict boolean")


def _validate_market_thesis_diagnostics(value: Any) -> None:
    identities = (
        value.market_thesis_id,
        value.bound_market_thesis_id,
        value.market_thesis_root_id,
        value.market_thesis_mechanism,
        value.market_thesis_authority_relation,
        value.playbook_first_failed_hard_gate_id,
    )
    if (
        any(
            identity is not None
            and (not isinstance(identity, str) or not identity)
            for identity in identities
        )
        or type(value.market_thesis_binding_required) is not bool
        or type(value.market_thesis_action_bound) is not bool
        or type(value.playbook_plan_delivery_valid) is not bool
        or value.market_thesis_match_status
        not in _MARKET_THESIS_MATCH_STATUSES
        or not math.isfinite(float(value.playbook_match_strength))
        or not 0.0 <= float(value.playbook_match_strength) <= 1.0
        or (
            value.market_thesis_id is None
            and any(identity is not None for identity in identities[1:5])
        )
        or (
            value.market_thesis_id is not None
            and any(identity is None for identity in identities[2:5])
        )
        or (
            value.bound_market_thesis_id is not None
            and value.bound_market_thesis_id != value.market_thesis_id
        )
        or value.market_thesis_action_bound
        != (value.bound_market_thesis_id is not None)
        or value.market_thesis_binding_required
        != (value.market_thesis_match_status != "not_required")
        or value.market_thesis_action_bound
        != (value.market_thesis_match_status == "exact_root_bound")
        or (value.market_thesis_id is not None)
        != (
            value.market_thesis_match_status
            in {"root_identity_unbound", "exact_root_bound"}
        )
        or (
            value.market_thesis_id is None
            and value.playbook_match_strength != 0.0
        )
    ):
        raise ValueError("market thesis calibration diagnostics are invalid")


def _validate_context_episode_identities(value: Any) -> None:
    for name in (
        "context_thesis_id",
        "parent_context_thesis_id",
    ):
        identity = getattr(value, name)
        if identity is not None and (
            not isinstance(identity, str) or not identity
        ):
            raise ValueError(
                f"brain calibration {name} must be non-empty text"
            )
    if (
        value.parent_context_thesis_id is not None
        and value.parent_context_thesis_id != value.context_thesis_id
    ):
        raise ValueError(
            "brain calibration entry episode parent disagrees with context thesis"
        )
    if value.episode_id is not None and value.parent_context_thesis_id is None:
        raise ValueError(
            "brain calibration entry episode lacks its parent context thesis"
        )
    if value.dimension == "thesis_strength":
        expected_kind = (
            "dfp_context_thesis"
            if value.playbook
            == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
            else "lsr_context_thesis"
            if value.playbook
            == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
            else None
        )
        if (
            expected_kind is None
            or value.calibration_unit_kind != expected_kind
            or value.context_thesis_id is None
            or value.calibration_unit_id != value.context_thesis_id
        ):
            raise ValueError(
                "brain calibration thesis unit disagrees with its Context owner"
            )


def _validate_lsr_context_thesis_target_contract(value: Any) -> None:
    """Keep an LSR Context thesis free of child EntryEpisode targets."""

    if not (
        value.dimension == "thesis_strength"
        and value.calibration_unit_kind == "lsr_context_thesis"
    ):
        return
    empty_route_lists = all(
        getattr(value, name) == "[]"
        for name in (
            "intermediate_liquidity_ids",
            "path_blocker_ids",
            "source_path_ids",
        )
    )
    invalidation_price = value.invalidation_price
    if (
        value.draw_id is not None
        or value.draw_price is not None
        or value.liquidity_route_id is not None
        or value.context_draw_id is not None
        or value.primary_deliverable_target_id is not None
        or value.terminal_draw_id is not None
        or value.authority_barrier_id is not None
        or value.authority_barrier_price is not None
        or not empty_route_lists
        or value.obstruction_distance_R is not None
        or value.free_path_R is not None
        or value.soft_obstruction_count != 0
        or value.hard_barrier_before_target
        or invalidation_price is None
        or not math.isfinite(float(invalidation_price))
        or float(invalidation_price) <= 0.0
        or not isinstance(value.invalidation_source_id, str)
        or not value.invalidation_source_id
        or value.deadline is None
        or value.target_deadline_kind != "thesis_deadline"
    ):
        raise ValueError("LSR Context thesis target custody is invalid")


def _uncertainty_total(
    conflict: float,
    required_evidence_missing: float,
    authority_missing: float,
    graph_ambiguity: float,
) -> float:
    components = tuple(
        _clamped(value, name="uncertainty_component")
        for value in (
            conflict,
            required_evidence_missing,
            authority_missing,
            graph_ambiguity,
        )
    )
    return _clamped(
        1.0 - math.prod(1.0 - value for value in components),
        name="uncertainty_total",
    )


def _uncertainty_revision_key(hypothesis: Any) -> str:
    """Identity one contemporaneous uncertainty state, not its minute."""

    metadata = dict(getattr(hypothesis, "context_metadata", {}) or {})
    sequence = getattr(hypothesis, "sequence", None)
    fields = (
        "authority_relation",
        "authority_rank_gap",
        "conflict_role",
        "conflict_scope",
        "acceptance_state",
        "ambiguity_count",
        "uncertainty_conflict",
        "uncertainty_required_evidence_missing",
        "uncertainty_authority_missing",
        "uncertainty_graph_ambiguity",
        "uncertainty_total",
    )
    return json.dumps(
        {
            "evidence_revision_id": getattr(
                hypothesis,
                "evidence_revision_id",
                None,
            ),
            "phase": hypothesis.phase.value,
            "completed_steps": (
                None if sequence is None else sequence.completed_steps
            ),
            "components": {
                name: metadata.get(name)
                for name in fields
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _hypothesis_context_features(
    hypothesis: Any,
    global_context: Any | None,
) -> dict[str, Any]:
    """Freeze only calibration covariates, never a duplicate graph snapshot.

    The Brain publishes the identity-bound, hypothesis-specific
    interpretation in ``context_metadata``.  Global conflicts deliberately
    do not provide a fallback here: ``affected_hypothesis_ids`` describes the
    candidate direction slots a graph fact could affect, not proof that the
    fact is connected to this exact setup/episode.  Reinterpreting that broad
    list in the recorder would contaminate unrelated calibration rows.
    """

    metadata = dict(getattr(hypothesis, "context_metadata", {}) or {})
    global_market_mode = _enum_text(
        None
        if global_context is None
        else getattr(global_context, "market_mode", None),
        "unknown",
    )
    relation = _enum_text(
        _context_feature(metadata, "authority_relation", None),
        "unrelated",
    )
    conflict_role = _enum_text(
        _context_feature(metadata, "conflict_role", None),
        "none",
    )
    conflict_scope = _enum_text(
        _context_feature(metadata, "conflict_scope", None),
        "none",
    )
    acceptance_state = _enum_text(
        _context_feature(metadata, "acceptance_state", None),
        "unknown",
    )

    rank_gap_raw = _optional_numeric(
        _context_feature(metadata, "authority_rank_gap", None)
    )
    obstruction_raw = _optional_numeric(
        _context_feature(
            metadata,
            "obstruction_distance_R",
            None,
        )
    )
    free_path_raw = _optional_numeric(
        _context_feature(metadata, "free_path_R", None)
    )
    hard_barrier_raw = _context_feature(
        metadata,
        "hard_barrier_before_target",
        "false",
    )
    blocker_ids_raw = metadata.get("path_blocker_ids")
    soft_count_raw = _context_feature(
        metadata,
        "soft_obstruction_count",
        0,
    )
    ambiguity_raw = _context_feature(metadata, "ambiguity_count", 0)

    rank_gap = 0 if rank_gap_raw is None else int(rank_gap_raw)
    soft_count = int(soft_count_raw or 0)
    ambiguity_count = int(ambiguity_raw or 0)
    obstruction_distance = (
        None if obstruction_raw is None else float(obstruction_raw)
    )
    free_path = None if free_path_raw is None else float(free_path_raw)
    if rank_gap < 0 or soft_count < 0 or ambiguity_count < 0:
        raise ValueError("calibration context counts cannot be negative")
    if obstruction_distance is not None and (
        not math.isfinite(obstruction_distance)
        or obstruction_distance < 0.0
    ):
        raise ValueError("calibration obstruction distance is invalid")
    if free_path is not None and (
        not math.isfinite(free_path) or free_path < 0.0
    ):
        raise ValueError("calibration free path is invalid")
    hard_barrier = _strict_metadata_bool(
        hard_barrier_raw,
        name="hard_barrier_before_target",
    )
    blocker_ids: tuple[str, ...] | None = None
    if blocker_ids_raw is not None:
        if not isinstance(blocker_ids_raw, str):
            raise ValueError("path_blocker_ids metadata must be JSON text")
        _validate_identity_list(
            blocker_ids_raw,
            name="path_blocker_ids metadata",
        )
        blocker_ids = tuple(json.loads(blocker_ids_raw))
    plan_delivery_valid = _strict_metadata_bool(
        _context_feature(
            metadata,
            "playbook_plan_delivery_valid",
            "false",
        ),
        name="playbook_plan_delivery_valid",
    )
    first_failed_hard_gate_id = next(
        (
            str(gate_id)
            for gate_id, passed in getattr(
                hypothesis,
                "hard_gate_results",
                {},
            ).items()
            if not passed
        ),
        None,
    )
    uncertainty_names = (
        "uncertainty_conflict",
        "uncertainty_required_evidence_missing",
        "uncertainty_authority_missing",
        "uncertainty_graph_ambiguity",
        "uncertainty_total",
    )
    if any(name in metadata for name in uncertainty_names):
        if not all(name in metadata for name in uncertainty_names):
            raise ValueError("uncertainty component metadata is incomplete")
        uncertainty_values = {
            name: _clamped(metadata[name], name=name)
            for name in uncertainty_names
        }
    else:
        raw_dimensions = dict(
            getattr(hypothesis, "raw_quality_dimensions", {}) or {}
        )
        total = _clamped(
            raw_dimensions.get(
                "uncertainty",
                getattr(hypothesis, "uncertainty", 1.0),
            ),
            name="uncertainty_total",
        )
        # Compatibility for hand-built test hypotheses.  Production Brain
        # always publishes all components from schema v5 onward.
        uncertainty_values = {
            "uncertainty_conflict": 0.0,
            "uncertainty_required_evidence_missing": 0.0,
            "uncertainty_authority_missing": total,
            "uncertainty_graph_ambiguity": 0.0,
            "uncertainty_total": total,
        }
    recomputed = _uncertainty_total(
        uncertainty_values["uncertainty_conflict"],
        uncertainty_values["uncertainty_required_evidence_missing"],
        uncertainty_values["uncertainty_authority_missing"],
        uncertainty_values["uncertainty_graph_ambiguity"],
    )
    if not math.isclose(
        recomputed,
        uncertainty_values["uncertainty_total"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("uncertainty total disagrees with its components")
    binding_required = getattr(
        hypothesis,
        "market_thesis_binding_required",
        False,
    )
    action_bound = getattr(
        hypothesis,
        "market_thesis_action_bound",
        False,
    )
    if type(binding_required) is not bool or type(action_bound) is not bool:
        raise ValueError("market thesis binding flags must be boolean")
    return {
        "market_thesis_id": getattr(hypothesis, "market_thesis_id", None),
        "bound_market_thesis_id": getattr(
            hypothesis,
            "bound_market_thesis_id",
            None,
        ),
        "market_thesis_root_id": getattr(
            hypothesis,
            "market_thesis_root_id",
            None,
        ),
        "market_thesis_mechanism": getattr(
            hypothesis,
            "market_thesis_mechanism",
            None,
        ),
        "market_thesis_authority_relation": getattr(
            hypothesis,
            "market_thesis_authority_relation",
            None,
        ),
        "playbook_match_strength": _clamped(
            getattr(hypothesis, "playbook_match_strength", 0.0),
            name="playbook_match_strength",
        ),
        "market_thesis_binding_required": binding_required,
        "market_thesis_action_bound": action_bound,
        "market_thesis_match_status": str(
            getattr(hypothesis, "market_thesis_match_status", "not_required")
        ),
        "playbook_first_failed_hard_gate_id": (
            first_failed_hard_gate_id
        ),
        "playbook_plan_delivery_valid": plan_delivery_valid,
        "global_market_mode": global_market_mode,
        "authority_relation": relation,
        "authority_rank_gap": rank_gap,
        "conflict_role": conflict_role,
        "conflict_scope": conflict_scope,
        "acceptance_state": acceptance_state,
        "obstruction_distance_R": obstruction_distance,
        "free_path_R": free_path,
        "soft_obstruction_count": soft_count,
        "hard_barrier_before_target": hard_barrier,
        "path_blocker_ids": blocker_ids,
        "ambiguity_count": ambiguity_count,
        **uncertainty_values,
    }


@dataclass(frozen=True)
class BrainCalibrationRecord:
    """One resolved calibration observation written to a light shard."""

    sample_id: str
    hypothesis_key: str
    playbook: str
    direction: str
    dimension: str
    setup_id: str
    calibration_unit_id: str
    calibration_unit_kind: str
    episode_id: str | None
    context_id: str | None
    context_thesis_id: str | None
    parent_context_thesis_id: str | None
    evidence_revision_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    selected_trigger_id: str | None
    selected_trigger_kind: str | None
    selected_trigger_at: pd.Timestamp | None
    available_trigger_kinds: str
    market_thesis_id: str | None
    bound_market_thesis_id: str | None
    market_thesis_root_id: str | None
    market_thesis_mechanism: str | None
    market_thesis_authority_relation: str | None
    playbook_match_strength: float
    market_thesis_binding_required: bool
    market_thesis_action_bound: bool
    market_thesis_match_status: str
    playbook_first_failed_hard_gate_id: str | None
    playbook_plan_delivery_valid: bool
    global_market_mode: str
    authority_relation: str
    authority_rank_gap: int
    conflict_role: str
    conflict_scope: str
    acceptance_state: str
    obstruction_distance_R: float | None
    free_path_R: float | None
    soft_obstruction_count: int
    hard_barrier_before_target: bool
    ambiguity_count: int
    uncertainty_conflict: float
    uncertainty_required_evidence_missing: float
    uncertainty_authority_missing: float
    uncertainty_graph_ambiguity: float
    uncertainty_total: float
    sampled_at: pd.Timestamp
    resolved_at: pd.Timestamp
    raw_value: float
    outcome_value: float | None
    resolution: str
    censored: bool
    fit_eligible: bool
    phase: str
    origin_price: float
    trigger_bar_high: float
    trigger_bar_low: float
    invalidation_price: float | None
    invalidation_source_id: str | None
    draw_price: float | None
    draw_id: str | None
    liquidity_route_id: str | None
    context_draw_id: str | None
    intermediate_liquidity_ids: str
    primary_deliverable_target_id: str | None
    terminal_draw_id: str | None
    path_blocker_ids: str
    source_path_ids: str
    dfp_structure_id: str | None
    dfp_structure_confirmed_at: pd.Timestamp | None
    dfp_h1_bos_id: str | None
    target_deadline_kind: str
    deadline: pd.Timestamp | None
    symbol: str
    instrument_id: int
    authority_barrier_id: str | None = None
    authority_barrier_price: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "sampled_at",
            "resolved_at",
            "selected_trigger_at",
            "dfp_structure_confirmed_at",
            "deadline",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"brain_calibration.{name}"),
                )
        if (
            not self.sample_id
            or not self.hypothesis_key
            or self.playbook
            not in {item.value for item in SUPPORTED_PLAYBOOKS}
            or self.direction not in {item.value for item in Direction}
            or self.dimension not in ALL_DIMENSIONS
            or not self.setup_id
            or not self.calibration_unit_id
            or not self.calibration_unit_kind
            or not self.resolution
            or not self.phase
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
        ):
            raise ValueError("brain calibration record identity is invalid")
        _validate_context_episode_identities(self)
        _validate_lsr_context_thesis_target_contract(self)
        _validate_market_thesis_diagnostics(self)
        for name in (
            "available_trigger_kinds",
            "intermediate_liquidity_ids",
            "path_blocker_ids",
            "source_path_ids",
        ):
            _validate_identity_list(getattr(self, name), name=name)
        trigger_kinds = tuple(json.loads(self.available_trigger_kinds))
        trigger_identity = (
            self.selected_trigger_id,
            self.selected_trigger_kind,
            self.selected_trigger_at,
        )
        if (
            any(value is None for value in trigger_identity)
            != all(value is None for value in trigger_identity)
            or (
                self.selected_trigger_id is not None
                and (
                    self.selected_trigger_kind not in trigger_kinds
                    or self.selected_trigger_at > self.sampled_at
                )
            )
            or (self.selected_trigger_id is None and trigger_kinds)
        ):
            raise ValueError("calibration frozen trigger identity is invalid")
        blocker_ids = tuple(json.loads(self.path_blocker_ids))
        for name in (
            "global_market_mode",
            "authority_relation",
            "conflict_role",
            "conflict_scope",
            "acceptance_state",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                self,
                name,
            ):
                raise ValueError(f"{name} is invalid")
        if (
            type(self.authority_rank_gap) is not int
            or self.authority_rank_gap < 0
            or type(self.soft_obstruction_count) is not int
            or self.soft_obstruction_count < 0
            or type(self.hard_barrier_before_target) is not bool
            or type(self.ambiguity_count) is not int
            or self.ambiguity_count < 0
            or (
                self.obstruction_distance_R is not None
                and (
                    type(self.obstruction_distance_R) is bool
                    or
                    not math.isfinite(float(self.obstruction_distance_R))
                    or float(self.obstruction_distance_R) < 0.0
                )
            )
            or (
                self.free_path_R is not None
                and (
                    type(self.free_path_R) is bool
                    or
                    not math.isfinite(float(self.free_path_R))
                    or float(self.free_path_R) < 0.0
                )
            )
        ):
            raise ValueError("brain calibration context metrics are invalid")
        if self.hard_barrier_before_target != bool(blocker_ids) or (
            self.hard_barrier_before_target
            and (
                self.free_path_R is None
                or self.obstruction_distance_R is None
            )
        ):
            raise ValueError(
                "hard barrier state disagrees with its frozen blocker geometry"
            )
        uncertainty_components = (
            self.uncertainty_conflict,
            self.uncertainty_required_evidence_missing,
            self.uncertainty_authority_missing,
            self.uncertainty_graph_ambiguity,
            self.uncertainty_total,
        )
        if any(
            not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in uncertainty_components
        ) or not math.isclose(
            self.uncertainty_total,
            _uncertainty_total(*uncertainty_components[:4]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("brain calibration uncertainty components are invalid")
        if (
            self.dimension == "uncertainty"
            and not math.isclose(
                self.raw_value,
                self.uncertainty_total,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("uncertainty row raw value must equal uncertainty_total")
        if self.target_deadline_kind not in {
            "thesis_deadline",
            "entry_deadline",
            "plan_deadline",
        }:
            raise ValueError("calibration target deadline kind is invalid")
        for name in (
            "liquidity_route_id",
            "context_draw_id",
            "primary_deliverable_target_id",
            "terminal_draw_id",
            "authority_barrier_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} is invalid")
        if (self.authority_barrier_id is None) != (
            self.authority_barrier_price is None
        ) or (
            self.authority_barrier_price is not None
            and (
                not math.isfinite(float(self.authority_barrier_price))
                or float(self.authority_barrier_price) <= 0.0
            )
        ):
            raise ValueError("calibration authority barrier is invalid")
        _clamped(self.raw_value, name="record.raw_value")
        if self.outcome_value is not None:
            _clamped(self.outcome_value, name="record.outcome_value")
        non_future_deadline_is_descriptive = bool(
            self.deadline is not None
            and self.deadline <= self.sampled_at
            and self.dimension in DESCRIPTIVE_DIMENSIONS
            and self.resolved_at == self.sampled_at
        )
        # ``origin_price`` is the close at this calibration revision, while
        # ``trigger_bar_*`` remains the geometry of the first frozen trigger.
        # They describe the same bar only when there is no selected trigger
        # yet, or when that trigger formed at ``sampled_at``.  Later evidence
        # and descriptive revisions are allowed to observe price outside the
        # historical trigger bar without rewriting either frozen value.
        origin_must_match_trigger_bar = bool(
            self.selected_trigger_at is None
            or self.selected_trigger_at == self.sampled_at
        )
        if (
            self.resolved_at < self.sampled_at
            or (
                self.deadline is not None
                and self.deadline <= self.sampled_at
                and not non_future_deadline_is_descriptive
            )
            or not math.isfinite(float(self.origin_price))
            or self.origin_price <= 0.0
            or not math.isfinite(float(self.trigger_bar_high))
            or not math.isfinite(float(self.trigger_bar_low))
            or self.trigger_bar_high < self.trigger_bar_low
            or (
                origin_must_match_trigger_bar
                and not self.trigger_bar_low
                <= self.origin_price
                <= self.trigger_bar_high
            )
            or any(
                value is not None
                and (not math.isfinite(float(value)) or float(value) <= 0.0)
                for value in (self.invalidation_price, self.draw_price)
            )
        ):
            raise ValueError("brain calibration record clock or price is invalid")
        if type(self.censored) is not bool or type(self.fit_eligible) is not bool:
            raise ValueError("brain calibration flags must be boolean")
        if self.censored and self.outcome_value is not None:
            raise ValueError("censored calibration rows cannot carry outcomes")
        expected_fit = bool(
            self.dimension in FITTED_DIMENSIONS
            and not self.censored
            and self.outcome_value is not None
            and not (
                self.dimension == "delivery_quality"
                and self.hard_barrier_before_target
            )
        )
        if self.fit_eligible != expected_fit:
            raise ValueError("fit eligibility disagrees with row resolution")
        if self.dimension in DESCRIPTIVE_DIMENSIONS and (
            self.resolved_at != self.sampled_at
            or self.outcome_value is not None
            or self.censored
        ):
            raise ValueError("descriptive dimensions cannot carry future labels")
    def to_dict(self) -> dict[str, Any]:
        """Return a parquet/jsonl-friendly mapping without changing clocks."""

        return asdict(self)


@dataclass(frozen=True)
class _OpenSample:
    sample_id: str
    hypothesis_key: str
    playbook: str
    direction: str
    dimension: str
    setup_id: str
    calibration_unit_id: str
    calibration_unit_kind: str
    episode_id: str | None
    context_id: str | None
    context_thesis_id: str | None
    parent_context_thesis_id: str | None
    evidence_revision_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    selected_trigger_id: str | None
    selected_trigger_kind: str | None
    selected_trigger_at: pd.Timestamp | None
    available_trigger_kinds: str
    market_thesis_id: str | None
    bound_market_thesis_id: str | None
    market_thesis_root_id: str | None
    market_thesis_mechanism: str | None
    market_thesis_authority_relation: str | None
    playbook_match_strength: float
    market_thesis_binding_required: bool
    market_thesis_action_bound: bool
    market_thesis_match_status: str
    playbook_first_failed_hard_gate_id: str | None
    playbook_plan_delivery_valid: bool
    global_market_mode: str
    authority_relation: str
    authority_rank_gap: int
    conflict_role: str
    conflict_scope: str
    acceptance_state: str
    obstruction_distance_R: float | None
    free_path_R: float | None
    soft_obstruction_count: int
    hard_barrier_before_target: bool
    ambiguity_count: int
    uncertainty_conflict: float
    uncertainty_required_evidence_missing: float
    uncertainty_authority_missing: float
    uncertainty_graph_ambiguity: float
    uncertainty_total: float
    sampled_at: pd.Timestamp
    raw_value: float
    phase: str
    origin_price: float
    trigger_bar_high: float
    trigger_bar_low: float
    invalidation_price: float | None
    invalidation_source_id: str | None
    draw_price: float | None
    draw_id: str | None
    liquidity_route_id: str | None
    context_draw_id: str | None
    intermediate_liquidity_ids: str
    primary_deliverable_target_id: str | None
    terminal_draw_id: str | None
    path_blocker_ids: str
    source_path_ids: str
    dfp_structure_id: str | None
    dfp_structure_confirmed_at: pd.Timestamp | None
    dfp_h1_bos_id: str | None
    target_deadline_kind: str
    deadline: pd.Timestamp | None
    symbol: str
    instrument_id: int
    authority_barrier_id: str | None = None
    authority_barrier_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sampled_at",
            aware_timestamp(self.sampled_at, name="open_sample.sampled_at"),
        )
        if self.selected_trigger_at is not None:
            object.__setattr__(
                self,
                "selected_trigger_at",
                aware_timestamp(
                    self.selected_trigger_at,
                    name="open_sample.selected_trigger_at",
                ),
            )
        if self.deadline is not None:
            object.__setattr__(
                self,
                "deadline",
                aware_timestamp(self.deadline, name="open_sample.deadline"),
            )
        if self.dfp_structure_confirmed_at is not None:
            object.__setattr__(
                self,
                "dfp_structure_confirmed_at",
                aware_timestamp(
                    self.dfp_structure_confirmed_at,
                    name="open_sample.dfp_structure_confirmed_at",
                ),
            )
        _validate_context_episode_identities(self)
        _validate_lsr_context_thesis_target_contract(self)
        _validate_market_thesis_diagnostics(self)
        _validate_identity_list(
            self.available_trigger_kinds,
            name="available_trigger_kinds",
        )
        trigger_kinds = tuple(json.loads(self.available_trigger_kinds))
        trigger_identity = (
            self.selected_trigger_id,
            self.selected_trigger_kind,
            self.selected_trigger_at,
        )
        if (
            any(value is None for value in trigger_identity)
            != all(value is None for value in trigger_identity)
            or (
                self.selected_trigger_id is not None
                and (
                    self.selected_trigger_kind not in trigger_kinds
                    or self.selected_trigger_at > self.sampled_at
                )
            )
            or (self.selected_trigger_id is None and trigger_kinds)
        ):
            raise ValueError("open sample frozen trigger identity is invalid")
        if (self.authority_barrier_id is None) != (
            self.authority_barrier_price is None
        ) or (
            self.authority_barrier_price is not None
            and (
                not math.isfinite(float(self.authority_barrier_price))
                or float(self.authority_barrier_price) <= 0.0
            )
        ):
            raise ValueError("open sample authority barrier is invalid")


class BrainCalibrationRecorder:
    """Incrementally freeze and causally resolve typed Brain dimensions."""

    def __init__(self) -> None:
        self._open: dict[str, _OpenSample] = {}
        self._rows: list[BrainCalibrationRecord] = []
        self._seen: set[str] = set()
        self._seen_location_ids: set[tuple[str, str]] = set()
        self._late_registration_keys: set[
            tuple[str, str, str, str]
        ] = set()
        self._incomplete_registration_keys: set[
            tuple[str, str, str, str, str]
        ] = set()
        self._pending_bar_resolutions: dict[
            str,
            tuple[pd.Timestamp, float | None, str, bool],
        ] = {}
        self._trigger_bar_geometry: dict[
            tuple[str, int, str, pd.Timestamp],
            tuple[float, float],
        ] = {}
        self._terminal_thesis_units: dict[
            tuple[str, str, str, str],
            tuple[pd.Timestamp, float, str, str],
        ] = {}
        self._last_bar_start: pd.Timestamp | None = None
        self._last_observation_asof: pd.Timestamp | None = None

    @property
    def open_samples(self) -> tuple[_OpenSample, ...]:
        return tuple(self._open[key] for key in sorted(self._open))

    @property
    def rows(self) -> tuple[BrainCalibrationRecord, ...]:
        return tuple(self._rows)

    def drain_rows(self) -> tuple[BrainCalibrationRecord, ...]:
        rows = tuple(self._rows)
        self._rows.clear()
        return rows

    @staticmethod
    def _selected_trigger_identity(
        observation: Any,
        hypothesis: Any,
    ) -> tuple[str, int, str, pd.Timestamp] | None:
        selected_trigger = getattr(hypothesis, "selected_trigger", None)
        trigger_id = getattr(
            hypothesis,
            "selected_trigger_id",
            None
            if selected_trigger is None
            else getattr(selected_trigger, "trigger_id", None),
        )
        trigger_at = getattr(
            hypothesis,
            "selected_trigger_at",
            None
            if selected_trigger is None
            else getattr(selected_trigger, "observed_at", None),
        )
        if trigger_id is None and trigger_at is None:
            return None
        if (
            not isinstance(trigger_id, str)
            or not trigger_id
            or trigger_at is None
        ):
            raise ValueError("selected trigger identity or clock is incomplete")
        return (
            str(observation.symbol),
            int(observation.instrument_id),
            trigger_id,
            aware_timestamp(
                trigger_at,
                name="brain_calibration.selected_trigger_at",
            ),
        )

    def _remember_selected_trigger_bar(
        self,
        observation: Any,
        hypothesis: Any,
        *,
        source_bar: Bar,
    ) -> None:
        """Keep only the completed bar that first owns a frozen trigger."""

        entry_path_id = self._entry_path_id(observation, hypothesis)
        if not self._owns_selected_trigger_path(
            hypothesis,
            entry_path_id=entry_path_id,
        ):
            return
        identity = self._selected_trigger_identity(observation, hypothesis)
        if identity is None:
            return
        trigger_at = identity[-1]
        if trigger_at > source_bar.end:
            raise ValueError(
                "selected trigger cannot be known before its bar completes"
            )
        if trigger_at < source_bar.end:
            # The exact bar must already have been observed.  Never substitute
            # the current bar for an older frozen trigger.
            return
        geometry = (float(source_bar.high), float(source_bar.low))
        prior = self._trigger_bar_geometry.get(identity)
        if prior is not None and prior != geometry:
            raise ValueError("frozen trigger-bar geometry was rewritten")
        self._trigger_bar_geometry[identity] = geometry

    def _selected_trigger_bar_geometry(
        self,
        observation: Any,
        hypothesis: Any,
        *,
        source_bar: Bar,
        entry_path_id: str | None = None,
    ) -> tuple[float, float] | None:
        if not self._owns_selected_trigger_path(
            hypothesis,
            entry_path_id=entry_path_id,
        ):
            return None
        identity = self._selected_trigger_identity(observation, hypothesis)
        if identity is None:
            return float(source_bar.high), float(source_bar.low)
        trigger_at = identity[-1]
        if trigger_at == source_bar.end:
            return float(source_bar.high), float(source_bar.low)
        geometry = self._trigger_bar_geometry.get(identity)
        return geometry

    @staticmethod
    def _owns_selected_trigger_path(
        hypothesis: Any,
        *,
        entry_path_id: str | None,
    ) -> bool:
        selected_trigger = getattr(hypothesis, "selected_trigger", None)
        selected_path_id = getattr(
            selected_trigger,
            "entry_path_id",
            None,
        )
        return bool(
            BrainCalibrationRecorder._owns_entry_episode(hypothesis)
            and isinstance(entry_path_id, str)
            and entry_path_id
            and selected_path_id == entry_path_id
        )

    @property
    def late_registration_summary(self) -> Mapping[str, Any]:
        by_playbook: dict[str, int] = {}
        for (
            playbook,
            _direction,
            _unit_kind,
            _unit_id,
        ) in self._late_registration_keys:
            by_playbook[playbook] = by_playbook.get(playbook, 0) + 1
        return {
            "counting_basis": "unique_expired_calibration_unit",
            "late_registration_skipped": len(self._late_registration_keys),
            "by_playbook": dict(sorted(by_playbook.items())),
            "incomplete_registration_skipped": len(
                self._incomplete_registration_keys
            ),
            "incomplete_by_dimension": dict(
                sorted(
                    (
                        dimension,
                        sum(
                            1
                            for item in self._incomplete_registration_keys
                            if item[2] == dimension
                        ),
                    )
                    for dimension in {
                        item[2]
                        for item in self._incomplete_registration_keys
                    }
                )
            ),
        }

    def on_bar(self, bar: Bar) -> None:
        """Resolve existing samples from one later completed 1m bar.

        This must run before the engine creates the snapshot for ``bar``.  A
        sample created from that snapshot therefore cannot see its own bar.
        """

        if self._last_bar_start is not None and bar.start <= self._last_bar_start:
            raise ValueError("calibration bars must be strictly increasing")
        self._last_bar_start = bar.start
        if bar.data_gap_before_minutes:
            self._censor_all(bar.end, "data_gap_boundary")
            return
        if not self._open:
            return
        for sample in tuple(self._open.values()):
            if (bar.symbol, bar.instrument_id) != (
                sample.symbol,
                sample.instrument_id,
            ):
                self._censor_sample(sample, bar.end, "contract_boundary")
                continue
            if bar.synthetic_no_trade:
                # The placeholder has no price path, but causal clock time
                # still advances.  Targets whose frozen horizon ends on this
                # no-trade interval settle at that exact deadline.
                if sample.deadline is not None and sample.deadline <= bar.end:
                    self._queue_deadline(sample, sample.deadline)
                continue
            self._resolve_with_bar(sample, bar)

    def observe(
        self,
        snapshot: EngineSnapshot,
        *,
        source_bar: Bar,
    ) -> None:
        """Resolve typed lifecycles, then freeze newly eligible samples."""

        observation = snapshot.observation
        belief = snapshot.belief
        asof = aware_timestamp(
            observation.asof,
            name="brain_calibration.observation.asof",
        )
        if (
            source_bar.end != asof
            or (source_bar.symbol, source_bar.instrument_id)
            != (observation.symbol, observation.instrument_id)
            or not math.isclose(
                float(source_bar.close),
                float(observation.price),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "source bar must be the exact completed bar behind snapshot"
            )
        if (
            self._last_observation_asof is not None
            and asof <= self._last_observation_asof
        ):
            raise ValueError("calibration observations must be strictly increasing")
        self._last_observation_asof = asof
        if set(observation.anomalies) & _BOUNDARY_ANOMALIES:
            self._censor_all(asof, "observation_boundary")
            return
        if source_bar.synthetic_no_trade:
            self._commit_pending_bar_resolutions(preemptive_only=True)
            return

        action_candidate_items = _action_candidate_items(belief)
        action_hypotheses = {
            candidate_id: hypothesis
            for candidate_id, hypothesis in action_candidate_items
            if hypothesis.playbook in SUPPORTED_PLAYBOOKS
        }
        if len(action_hypotheses) != sum(
            hypothesis.playbook in SUPPORTED_PLAYBOOKS
            for _, hypothesis in action_candidate_items
        ):
            raise ValueError("Brain candidates contain duplicate candidate IDs")
        lifecycle_candidate_items = _lifecycle_candidate_items(
            belief,
            action_items=action_candidate_items,
        )
        resolution_hypotheses = {
            candidate_id: hypothesis
            for candidate_id, hypothesis in lifecycle_candidate_items
            if hypothesis.playbook in SUPPORTED_PLAYBOOKS
        }
        if len(resolution_hypotheses) != sum(
            hypothesis.playbook in SUPPORTED_PLAYBOOKS
            for _, hypothesis in lifecycle_candidate_items
        ):
            raise ValueError(
                "Brain lifecycle candidates contain duplicate candidate IDs"
            )
        for hypothesis in action_hypotheses.values():
            self._remember_selected_trigger_bar(
                observation,
                hypothesis,
                source_bar=source_bar,
            )
        locations = {
            item.location_id: item
            for item in getattr(observation, "entry_locations", ())
        }
        # A completed bar can straddle a non-aligned frozen deadline, or
        # resolve an invalidation before the snapshot is built.  Settle those
        # conservative clock/price outcomes before any snapshot-derived DFP
        # structure interpretation; otherwise post-deadline structure from
        # the same completed bar could rewrite the causal label.
        self._commit_pending_bar_resolutions(preemptive_only=True)
        context_theses = getattr(belief, "context_theses", {})
        self._resolve_lsr_context_theses(asof, context_theses)
        self._promote_horizon_context_terminals(
            asof,
            context_theses,
        )
        self._resolve_dfp_context_theses(asof, observation)
        self._resolve_from_snapshot(
            asof,
            resolution_hypotheses,
            locations,
            observation=observation,
        )
        self._commit_pending_bar_resolutions()
        for sample in tuple(self._open.values()):
            if sample.deadline is not None and sample.deadline <= asof:
                self._deadline(sample, sample.deadline)
        for candidate_id, hypothesis in action_hypotheses.items():
            self._register_hypothesis(
                observation,
                hypothesis,
                locations,
                source_bar,
                candidate_id=candidate_id,
                global_context=getattr(belief, "global_context", None),
            )

    def prime(
        self,
        snapshot: EngineSnapshot,
        *,
        source_bar: Bar,
    ) -> None:
        """Remember warmup identities without opening or emitting targets."""

        if self._open or self._rows or self._pending_bar_resolutions:
            raise ValueError("calibration priming is allowed only before capture")
        observation = snapshot.observation
        asof = aware_timestamp(
            observation.asof,
            name="brain_calibration.prime.asof",
        )
        if (
            source_bar.end != asof
            or (source_bar.symbol, source_bar.instrument_id)
            != (observation.symbol, observation.instrument_id)
            or not math.isclose(
                float(source_bar.close),
                float(observation.price),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "source bar must be the exact completed bar behind snapshot"
            )
        if (
            self._last_observation_asof is not None
            and asof <= self._last_observation_asof
        ):
            raise ValueError("calibration observations must be strictly increasing")
        self._last_observation_asof = asof
        if (
            source_bar.synthetic_no_trade
            or set(observation.anomalies) & _BOUNDARY_ANOMALIES
        ):
            return
        for candidate_id, hypothesis in _action_candidate_items(
            snapshot.belief
        ):
            if hypothesis.playbook not in SUPPORTED_PLAYBOOKS:
                continue
            self._remember_selected_trigger_bar(
                observation,
                hypothesis,
                source_bar=source_bar,
            )
            self._prime_hypothesis(
                observation,
                hypothesis,
                candidate_id=candidate_id,
                source_bar=source_bar,
                global_context=getattr(
                    snapshot.belief,
                    "global_context",
                    None,
                ),
            )

    def _prime_hypothesis(
        self,
        observation: Any,
        hypothesis: Any,
        *,
        candidate_id: str,
        source_bar: Bar,
        global_context: Any | None,
    ) -> None:
        sequence = hypothesis.sequence
        if (
            sequence is None
            or sequence.setup_id is None
            or hypothesis.phase
            in {PlaybookPhase.INACTIVE, PlaybookPhase.COMPLETED}
        ):
            return
        path_id = self._entry_path_id(observation, hypothesis)
        raw = self._raw_dimensions(hypothesis)

        def remember(dimension: str, raw_value: float, revision_key: str) -> None:
            if self._calibration_unit(
                hypothesis,
                dimension,
                entry_path_id=path_id,
            ) is None:
                # Runtime root candidates may expose evaluator-native context
                # and sequence diagnostics before a stable Context/Episode
                # owner exists.  Warmup must not invent an owner merely to
                # left-censor a fitted target.
                return
            sample = self._freeze_sample(
                observation,
                hypothesis,
                candidate_id=candidate_id,
                dimension=dimension,
                raw_value=raw_value,
                revision_key=revision_key,
                entry_path_id=path_id,
                source_bar=source_bar,
                global_context=global_context,
            )
            self._seen.add(sample.sample_id)

        valid_trigger = self._valid_trigger_observation(hypothesis, raw)
        if hypothesis.phase is PlaybookPhase.INVALIDATED:
            if valid_trigger:
                frozen_trigger = getattr(hypothesis, "selected_trigger", None)
                trigger_id = (
                    getattr(
                        hypothesis,
                        "selected_trigger_id",
                        None
                        if frozen_trigger is None
                        else frozen_trigger.trigger_id,
                    )
                    or path_id
                    or hypothesis.evidence_revision_id
                    or sequence.setup_id
                )
                remember(
                    "entry_readiness",
                    raw["entry_readiness"],
                    f"trigger:{trigger_id}",
                )
                if hypothesis.plan is not None:
                    remember(
                        "delivery_quality",
                        raw["delivery_quality"],
                        f"delivery-trigger:{trigger_id}",
                    )
            return

        revision = getattr(hypothesis, "evidence_revision_id", None)
        if revision is not None:
            remember("thesis_strength", raw["thesis_strength"], revision)
            remember(
                "uncertainty",
                raw["uncertainty"],
                _uncertainty_revision_key(hypothesis),
            )
        remember(
            "sequence_progress",
            raw["sequence_progress"],
            (
                f"{sequence.setup_id}:{hypothesis.phase.value}:"
                f"{sequence.completed_steps}/{len(sequence.steps)}"
            ),
        )
        location_id = getattr(hypothesis, "entry_location_id", None)
        if (
            location_id is not None
            and self._calibration_unit(
                hypothesis,
                "location_quality",
                entry_path_id=path_id,
            )
            is not None
            and any(
                item.location_id == location_id
                for item in getattr(observation, "entry_locations", ())
            )
        ):
            self._seen_location_ids.add((candidate_id, location_id))
        if valid_trigger:
            frozen_trigger = getattr(hypothesis, "selected_trigger", None)
            trigger_id = (
                getattr(
                    hypothesis,
                    "selected_trigger_id",
                    None
                    if frozen_trigger is None
                    else frozen_trigger.trigger_id,
                )
                or path_id
                or revision
                or sequence.setup_id
            )
            remember(
                "entry_readiness",
                raw["entry_readiness"],
                f"trigger:{trigger_id}",
            )
            if hypothesis.plan is not None:
                remember(
                    "delivery_quality",
                    raw["delivery_quality"],
                    f"delivery-trigger:{trigger_id}",
                )

    def close_unresolved(
        self,
        asof: pd.Timestamp,
        *,
        reason: str = "window_end",
    ) -> None:
        """Right-censor every unresolved sample at a replay boundary."""

        clock = aware_timestamp(asof, name="brain_calibration.window_end")
        self._censor_all(clock, reason)

    def state_dict(self) -> dict[str, Any]:
        """Serialize enough state for exact checkpoint/resume."""

        return {
            "schema_version": RECORDER_SCHEMA_VERSION,
            "open_samples": [
                self._serialize_open(sample)
                for sample in self.open_samples
            ],
            "queued_rows": [self._serialize_row(row) for row in self._rows],
            "seen_sample_ids": sorted(self._seen),
            "seen_location_ids": [
                list(item) for item in sorted(self._seen_location_ids)
            ],
            "late_registration_keys": [
                list(item) for item in sorted(self._late_registration_keys)
            ],
            "incomplete_registration_keys": [
                list(item)
                for item in sorted(self._incomplete_registration_keys)
            ],
            "pending_bar_resolutions": [
                {
                    "sample_id": sample_id,
                    "resolved_at": _iso(resolved_at),
                    "outcome": outcome,
                    "resolution": resolution,
                    "censored": censored,
                }
                for sample_id, (
                    resolved_at,
                    outcome,
                    resolution,
                    censored,
                ) in sorted(self._pending_bar_resolutions.items())
            ],
            "trigger_bar_geometry": [
                {
                    "symbol": symbol,
                    "instrument_id": instrument_id,
                    "trigger_id": trigger_id,
                    "trigger_at": _iso(trigger_at),
                    "high": high,
                    "low": low,
                }
                for (
                    symbol,
                    instrument_id,
                    trigger_id,
                    trigger_at,
                ), (high, low) in sorted(
                    self._trigger_bar_geometry.items(),
                    key=lambda item: item[0],
                )
            ],
            "terminal_thesis_units": [
                {
                    "playbook": playbook,
                    "direction": direction,
                    "calibration_unit_kind": calibration_unit_kind,
                    "calibration_unit_id": calibration_unit_id,
                    "context_thesis_id": calibration_unit_id,
                    "resolved_at": _iso(resolved_at),
                    "outcome": outcome,
                    "resolution": resolution,
                    "scope": scope,
                }
                for (
                    playbook,
                    direction,
                    calibration_unit_kind,
                    calibration_unit_id,
                ), (resolved_at, outcome, resolution, scope) in sorted(
                    self._terminal_thesis_units.items(),
                    key=lambda item: item[0],
                )
            ],
            "last_bar_start": _iso(self._last_bar_start),
            "last_observation_asof": _iso(self._last_observation_asof),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "BrainCalibrationRecorder":
        """Restore a recorder produced by :meth:`state_dict`."""

        if state.get("schema_version") != RECORDER_SCHEMA_VERSION:
            raise ValueError("unsupported brain calibration recorder state")
        recorder = cls()
        recorder._seen = {str(item) for item in state.get("seen_sample_ids", ())}
        recorder._seen_location_ids = {
            (str(item[0]), str(item[1]))
            for item in state.get("seen_location_ids", ())
        }
        recorder._late_registration_keys = {
            (
                str(item[0]),
                str(item[1]),
                str(item[2]),
                str(item[3]),
            )
            for item in state.get("late_registration_keys", ())
        }
        recorder._incomplete_registration_keys = {
            (
                str(item[0]),
                str(item[1]),
                str(item[2]),
                str(item[3]),
                str(item[4]),
            )
            for item in state.get("incomplete_registration_keys", ())
        }
        recorder._last_bar_start = _timestamp(
            state.get("last_bar_start"),
            name="brain_calibration.last_bar_start",
        )
        recorder._last_observation_asof = _timestamp(
            state.get("last_observation_asof"),
            name="brain_calibration.last_observation_asof",
        )
        terminal_thesis_units = state.get("terminal_thesis_units")
        if not isinstance(terminal_thesis_units, list):
            raise ValueError(
                "brain calibration checkpoint lacks terminal thesis units"
            )
        for payload in terminal_thesis_units:
            if not isinstance(payload, Mapping):
                raise ValueError("terminal thesis checkpoint row is invalid")
            playbook = payload.get("playbook")
            direction = payload.get("direction")
            calibration_unit_kind = payload.get("calibration_unit_kind")
            calibration_unit_id = payload.get("calibration_unit_id")
            context_thesis_id = payload.get("context_thesis_id")
            resolved_at = _timestamp(
                payload.get("resolved_at"),
                name="brain_calibration.terminal_thesis.resolved_at",
            )
            outcome = payload.get("outcome")
            resolution = payload.get("resolution")
            scope = payload.get("scope")
            if (
                playbook not in {
                    Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
                    Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
                }
                or direction not in {Direction.LONG.value, Direction.SHORT.value}
                or calibration_unit_kind
                not in {"dfp_context_thesis", "lsr_context_thesis"}
                or (
                    playbook == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
                    and calibration_unit_kind != "dfp_context_thesis"
                )
                or (
                    playbook == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
                    and calibration_unit_kind != "lsr_context_thesis"
                )
                or not isinstance(calibration_unit_id, str)
                or not calibration_unit_id
                or context_thesis_id != calibration_unit_id
                or resolved_at is None
                or isinstance(outcome, bool)
                or not isinstance(outcome, (int, float))
                or not math.isfinite(float(outcome))
                or not 0.0 <= float(outcome) <= 1.0
                or not isinstance(resolution, str)
                or not resolution
                or scope
                not in {
                    "context_terminal",
                    "thesis_observation_horizon",
                }
            ):
                raise ValueError("terminal thesis checkpoint row is invalid")
            key = (
                str(playbook),
                str(direction),
                str(calibration_unit_kind),
                calibration_unit_id,
            )
            if key in recorder._terminal_thesis_units:
                raise ValueError("duplicate terminal thesis checkpoint owner")
            recorder._terminal_thesis_units[key] = (
                resolved_at,
                float(outcome),
                resolution,
                str(scope),
            )
        trigger_geometry = state.get("trigger_bar_geometry")
        if not isinstance(trigger_geometry, list):
            raise ValueError(
                "brain calibration checkpoint lacks frozen trigger-bar geometry"
            )
        for payload in trigger_geometry:
            if not isinstance(payload, Mapping):
                raise ValueError("frozen trigger-bar checkpoint row is invalid")
            symbol = payload.get("symbol")
            instrument_id = payload.get("instrument_id")
            trigger_id = payload.get("trigger_id")
            trigger_at = _timestamp(
                payload.get("trigger_at"),
                name="brain_calibration.trigger_bar_geometry.trigger_at",
            )
            high = payload.get("high")
            low = payload.get("low")
            if (
                not isinstance(symbol, str)
                or not symbol
                or type(instrument_id) is not int
                or instrument_id <= 0
                or not isinstance(trigger_id, str)
                or not trigger_id
                or trigger_at is None
                or isinstance(high, bool)
                or isinstance(low, bool)
            ):
                raise ValueError("frozen trigger-bar checkpoint row is invalid")
            high = float(high)
            low = float(low)
            if (
                not math.isfinite(high)
                or not math.isfinite(low)
                or high < low
            ):
                raise ValueError("frozen trigger-bar checkpoint geometry is invalid")
            key = (symbol, instrument_id, trigger_id, trigger_at)
            if key in recorder._trigger_bar_geometry:
                raise ValueError("duplicate frozen trigger-bar checkpoint identity")
            recorder._trigger_bar_geometry[key] = (high, low)
        for payload in state.get("open_samples", ()):
            sample = recorder._deserialize_open(payload)
            if sample.sample_id in recorder._open:
                raise ValueError("duplicate open calibration sample in state")
            if (
                sample.dimension == "thesis_strength"
                and recorder._terminal_thesis_key(sample)
                in recorder._terminal_thesis_units
            ):
                raise ValueError(
                    "terminal thesis checkpoint retains an open revision"
                )
            recorder._open[sample.sample_id] = sample
        for payload in state.get("pending_bar_resolutions", ()):
            sample_id = str(payload["sample_id"])
            if sample_id not in recorder._open:
                raise ValueError(
                    "pending calibration outcome lacks its open sample"
                )
            outcome = payload.get("outcome")
            recorder._pending_bar_resolutions[sample_id] = (
                aware_timestamp(
                    payload["resolved_at"],
                    name=(
                        "brain_calibration.pending_bar_resolution."
                        "resolved_at"
                    ),
                ),
                None
                if outcome is None
                else _clamped(
                    outcome,
                    name=(
                        "brain_calibration.pending_bar_resolution.outcome"
                    ),
                ),
                str(payload["resolution"]),
                _strict_metadata_bool(
                    payload["censored"],
                    name="pending_bar_resolution.censored",
                ),
            )
        recorder._rows = [
            recorder._deserialize_row(payload)
            for payload in state.get("queued_rows", ())
        ]
        return recorder

    def _resolve_with_bar(self, sample: _OpenSample, bar: Bar) -> None:
        deadline = sample.deadline
        # The label may use a completed bar only when the entire bar belongs
        # to the frozen horizon.  A bar crossing a non-aligned deadline would
        # otherwise leak post-deadline high/low/close into the target.
        if deadline is not None and bar.end > deadline:
            if bar.start < deadline:
                self._queue_bar_resolution(
                    sample,
                    resolved_at=deadline,
                    outcome=None,
                    resolution="deadline_crossed_inside_completed_bar",
                    censored=True,
                )
            else:
                self._queue_deadline(sample, deadline)
            return
        invalidated = bool(
            sample.invalidation_price is not None
            and (
                (
                    sample.direction == Direction.LONG.value
                    and bar.low <= sample.invalidation_price
                )
                or (
                    sample.direction == Direction.SHORT.value
                    and bar.high >= sample.invalidation_price
                )
            )
        )
        # Conservative same-bar ordering: invalidation always wins.
        if invalidated:
            self._queue_bar_resolution(
                sample,
                resolved_at=bar.end,
                outcome=0.0,
                resolution="invalidation_touched",
            )
            return
        draw_touched = bool(
            sample.draw_price is not None
            and (
                (
                    sample.direction == Direction.LONG.value
                    and bar.high >= sample.draw_price
                )
                or (
                    sample.direction == Direction.SHORT.value
                    and bar.low <= sample.draw_price
                )
            )
        )
        if sample.dimension in {"thesis_strength", "delivery_quality"}:
            # An LSR Context Thesis owns the manipulation/sweep lifecycle,
            # not any one child EntryEpisode's deliverable target.  A zone's
            # draw remains a delivery-quality label and must never settle the
            # shared reversal thesis.
            if (
                sample.dimension == "thesis_strength"
                and sample.calibration_unit_kind == "lsr_context_thesis"
            ):
                return
            if draw_touched:
                self._queue_bar_resolution(
                    sample,
                    resolved_at=bar.end,
                    outcome=1.0,
                    resolution="draw_delivered",
                )
                return
        elif sample.dimension == "entry_readiness":
            if bar.start >= sample.sampled_at:
                advanced = bool(
                    (
                        sample.direction == Direction.LONG.value
                        and bar.close > sample.trigger_bar_high
                    )
                    or (
                        sample.direction == Direction.SHORT.value
                        and bar.close < sample.trigger_bar_low
                    )
                )
                if advanced:
                    self._queue_bar_resolution(
                        sample,
                        resolved_at=bar.end,
                        outcome=1.0,
                        resolution="close_advanced_before_deadline",
                    )
                    return
                # A pause inside the frozen trigger range is unresolved, not
                # negative evidence.  Keep the sample open until a later
                # close advances, invalidation wins, or deadline expires.

    def _queue_bar_resolution(
        self,
        sample: _OpenSample,
        *,
        resolved_at: pd.Timestamp,
        outcome: float | None,
        resolution: str,
        censored: bool = False,
    ) -> None:
        """Hold a completed-bar result until Observation validates the bar."""

        if sample.sample_id in self._pending_bar_resolutions:
            raise ValueError("calibration sample received two bar resolutions")
        self._pending_bar_resolutions[sample.sample_id] = (
            aware_timestamp(
                resolved_at,
                name="brain_calibration.pending_bar_resolution.resolved_at",
            ),
            None
            if outcome is None
            else _clamped(
                outcome,
                name="brain_calibration.pending_bar_resolution.outcome",
            ),
            str(resolution),
            bool(censored),
        )

    def _commit_pending_bar_resolutions(
        self,
        *,
        preemptive_only: bool = False,
    ) -> None:
        """Commit bar results only after same-clock anomaly/semantic checks."""

        for sample_id, (
            resolved_at,
            outcome,
            resolution,
            censored,
        ) in tuple(self._pending_bar_resolutions.items()):
            if preemptive_only and not (
                censored
                or outcome != 1.0
                or resolution
                in {
                    "thesis_intact_at_deadline",
                    "deadline_without_dimension_delivery",
                    "deadline_without_trigger_advance",
                }
            ):
                continue
            sample = self._open.get(sample_id)
            if sample is None:
                self._pending_bar_resolutions.pop(sample_id, None)
                continue
            self._finish(
                sample,
                resolved_at=resolved_at,
                outcome=outcome,
                resolution=resolution,
                censored=censored,
            )

    def _queue_deadline(
        self,
        sample: _OpenSample,
        deadline: pd.Timestamp,
    ) -> None:
        thesis_intact = sample.dimension == "thesis_strength"
        resolution = "deadline_without_dimension_delivery"
        if thesis_intact:
            resolution = "thesis_intact_at_deadline"
        elif sample.dimension == "entry_readiness":
            resolution = "deadline_without_trigger_advance"
        self._queue_bar_resolution(
            sample,
            resolved_at=deadline,
            outcome=1.0 if thesis_intact else 0.0,
            resolution=resolution,
        )

    def _deadline(self, sample: _OpenSample, deadline: pd.Timestamp) -> None:
        thesis_intact = sample.dimension == "thesis_strength"
        resolution = "deadline_without_dimension_delivery"
        if thesis_intact:
            resolution = "thesis_intact_at_deadline"
        elif sample.dimension == "entry_readiness":
            resolution = "deadline_without_trigger_advance"
        self._finish(
            sample,
            resolved_at=deadline,
            outcome=1.0 if thesis_intact else 0.0,
            resolution=resolution,
        )

    @staticmethod
    def _thesis_failure(
        sample: _OpenSample,
        reason: str | None,
        terminal_source_ids: set[str],
    ) -> bool:
        if reason == "global_frozen_source_invalidated":
            if sample.playbook == Playbook.DISPLACEMENT_FIRST_PULLBACK.value:
                thesis_sources = {
                    item
                    for item in (
                        sample.dfp_structure_id,
                        sample.draw_id,
                    )
                    if item is not None
                }
                return bool(thesis_sources & terminal_source_ids)
            # An LSR child owns its zone/path/invalidation projection, while
            # the reversal Context owns the sweep/source thesis.  Requiring
            # the exact Context ID prevents a local zone terminal from being
            # promoted into a shared thesis failure.
            return bool(
                sample.context_thesis_id is not None
                and sample.context_thesis_id in terminal_source_ids
            )
        if sample.playbook == Playbook.DISPLACEMENT_FIRST_PULLBACK.value:
            return reason in _DFP_THESIS_FAILURE_REASONS
        return bool(
            reason in _LSR_CONTEXT_THESIS_FAILURE_REASONS
            and sample.context_thesis_id is not None
            and sample.context_thesis_id in terminal_source_ids
        )

    def _resolve_lsr_context_theses(
        self,
        asof: pd.Timestamp,
        contexts: Mapping[str, Any],
    ) -> None:
        """Resolve LSR thesis units only from their exact Context owner.

        EntryEpisode phase/reason remains authoritative for its zone/path
        dimensions, never for this shared market thesis.  The Context view is
        the durable causal projection that survives a child path closing.
        """

        if not isinstance(contexts, Mapping):
            raise ValueError("Brain Context Thesis projection is invalid")
        for sample in tuple(self._open.values()):
            if (
                sample.playbook
                != Playbook.LIQUIDITY_SWEEP_REVERSAL.value
                or sample.dimension != "thesis_strength"
            ):
                continue
            context_id = sample.context_thesis_id
            if context_id is None or sample.calibration_unit_id != context_id:
                raise ValueError("LSR thesis sample lacks its Context owner")
            context = contexts.get(context_id)
            if context is None:
                # A bounded projection is not a terminal fact.  Frozen price,
                # deadline, an exact later Context tombstone, or the replay
                # boundary remains responsible for settlement.
                continue
            context_direction = getattr(context, "direction", None)
            context_direction = getattr(
                context_direction,
                "value",
                context_direction,
            )
            if (
                getattr(context, "context_thesis_id", None) != context_id
                or context_direction != sample.direction
            ):
                raise ValueError(
                    "Brain LSR Context Thesis projection disagrees with its "
                    "calibration owner"
                )
            lifecycle = getattr(context, "lifecycle", None)
            if lifecycle not in {
                "forming",
                "active",
                "weakening",
                "completed",
                "invalidated",
                "censored",
            }:
                raise ValueError("Brain LSR Context Thesis lifecycle is invalid")
            if lifecycle not in {"completed", "invalidated", "censored"}:
                continue
            terminal_at = _timestamp(
                getattr(context, "terminal_at", None),
                name="brain_calibration.lsr_context_terminal.terminal_at",
            )
            terminal_reason = getattr(context, "terminal_reason", None)
            if (
                terminal_at is None
                or terminal_at < sample.sampled_at
                or terminal_at > asof
                or not isinstance(terminal_reason, str)
                or not terminal_reason
            ):
                raise ValueError("Brain LSR Context Thesis terminal is invalid")
            if terminal_reason == "context_thesis_deadline_elapsed":
                deadline = sample.deadline
                if deadline is None or deadline > terminal_at:
                    raise ValueError(
                        "LSR Context deadline terminal disagrees with its "
                        "frozen calibration horizon"
                    )
                self._deadline(sample, deadline)
            elif lifecycle == "completed":
                self._finish(
                    sample,
                    resolved_at=terminal_at,
                    outcome=1.0,
                    resolution=f"context_completed:{terminal_reason}",
                )
            elif lifecycle == "invalidated":
                self._finish(
                    sample,
                    resolved_at=terminal_at,
                    outcome=0.0,
                    resolution=f"thesis_contradicted:{terminal_reason}",
                )
            else:
                self._finish(
                    sample,
                    resolved_at=terminal_at,
                    outcome=None,
                    resolution=f"context_censored:{terminal_reason}",
                    censored=True,
                )

    def _resolve_dfp_context_theses(
        self,
        asof: pd.Timestamp,
        observation: Any,
    ) -> None:
        """Resolve frozen DFP context authority independently of entry episodes."""

        frame = getattr(observation, "frame", None)
        if not callable(frame):
            return
        h4 = frame(Timeframe.H4)
        for sample in tuple(self._open.values()):
            if (
                sample.playbook
                != Playbook.DISPLACEMENT_FIRST_PULLBACK.value
                or sample.dimension != "thesis_strength"
                or sample.dfp_structure_id is None
            ):
                continue
            frozen_structure = next(
                (
                    item
                    for item in h4.structures
                    if item.structure_id == sample.dfp_structure_id
                ),
                None,
            )
            if frozen_structure is not None and (
                frozen_structure.direction.value != sample.direction
                or frozen_structure.lifecycle
                is not StructureLifecycle.CONFIRMED
                or frozen_structure.confirmed_at is None
            ):
                # A present typed revision of the exact frozen structure is
                # authoritative.  Mere absence from a bounded Observation
                # frame is not equivalent to a confirmed break and must not
                # manufacture a thesis failure.
                self._finish(
                    sample,
                    resolved_at=asof,
                    outcome=0.0,
                    resolution="thesis_contradicted:opposed_structure",
                )
                continue
            structure_clock = (
                sample.dfp_structure_confirmed_at
                or (
                    None
                    if frozen_structure is None
                    else frozen_structure.confirmed_at
                )
            )
            opposed = any(
                item.direction.value != sample.direction
                and item.lifecycle is StructureLifecycle.CONFIRMED
                and item.confirmed_at is not None
                # An opposing structure already visible at the frozen sample
                # clock is part of its contemporaneous evidence, not a future
                # thesis outcome.  Only a strictly later confirmation may
                # resolve this causal observation.  The exact frozen
                # structure revision above remains independently terminal.
                and item.confirmed_at > sample.sampled_at
                and (
                    structure_clock is None
                    or item.confirmed_at >= structure_clock
                )
                for item in h4.structures
            )
            if opposed:
                self._finish(
                    sample,
                    resolved_at=asof,
                    outcome=0.0,
                    resolution="thesis_contradicted:opposed_structure",
                )

    def _resolve_from_snapshot(
        self,
        asof: pd.Timestamp,
        hypotheses: Mapping[str, Any],
        locations: Mapping[str, Any],
        *,
        observation: Any,
    ) -> None:
        for sample in tuple(self._open.values()):
            pending = self._pending_bar_resolutions.get(sample.sample_id)
            if pending is not None and (
                pending[3] or pending[1] != 1.0
            ):
                # Observation-level anomaly checks have already passed.
                # A frozen invalidation or non-aligned deadline is therefore
                # the conservative same-bar result and must precede any
                # lifecycle interpretation built from that completed bar.
                self._finish(
                    sample,
                    resolved_at=pending[0],
                    outcome=pending[1],
                    resolution=pending[2],
                    censored=pending[3],
                )
                continue
            if sample.dimension in {
                "entry_readiness",
                "delivery_quality",
            }:
                # Trigger-clock targets own frozen price geometry.  A later
                # Brain terminal/rearm cannot rewrite their causal path.
                continue
            if sample.dimension == "location_quality":
                location = locations.get(sample.entry_location_id)
                if location is not None:
                    # The frozen EntryLocation owns this target.  Its typed
                    # lifecycle is still authoritative when the associated
                    # root-specific Brain candidate is temporarily absent
                    # from the current open-thesis projection.
                    if location.lifecycle is EntryLocationLifecycle.REJECTED:
                        self._finish(
                            sample,
                            resolved_at=asof,
                            outcome=1.0,
                            resolution="frozen_zone_rejected",
                        )
                        continue
                    if location.lifecycle is EntryLocationLifecycle.LEFT:
                        self._finish(
                            sample,
                            resolved_at=asof,
                            outcome=0.0,
                            resolution="frozen_zone_left",
                        )
                        continue
            hypothesis = hypotheses.get(sample.hypothesis_key)
            if hypothesis is None:
                # Candidate discovery is a bounded current-minute projection,
                # not a lifecycle authority.  Before reaching this branch we
                # have already applied every causal owner available without a
                # current candidate: hard replay/contract boundaries and the
                # frozen price/deadline path in ``on_bar``; DFP H4 structure
                # in ``_resolve_dfp_context_theses``; and the frozen typed
                # EntryLocation above.  Absence from the projection therefore
                # supplies no terminal fact for either a context thesis or an
                # entry episode.  Keep the sample open until one of those
                # frozen owners resolves it, a matching explicit terminal
                # candidate reappears, or the capture boundary censors it.
                continue
            terminal_sources = set(
                getattr(hypothesis, "terminal_source_ids", ())
            )
            current_path_id = (
                None
                if sample.dimension == "thesis_strength"
                else self._entry_path_id(observation, hypothesis)
            )
            current_unit = self._calibration_unit(
                hypothesis,
                sample.dimension,
                entry_path_id=current_path_id,
            )
            current_owner = (
                None if current_unit is None else current_unit[0]
            )
            frozen_terminal_owner_ids = {
                identity
                for identity in (
                    sample.calibration_unit_id,
                    *(
                        (
                            sample.dfp_structure_id,
                            sample.invalidation_source_id,
                            sample.draw_id,
                        )
                        if sample.dimension == "thesis_strength"
                        else (
                            sample.entry_path_id,
                            sample.entry_location_id,
                        )
                        if sample.dimension == "location_quality"
                        else ()
                    ),
                )
                if identity is not None
            }
            owner_matches = bool(
                current_owner == sample.calibration_unit_id
                or frozen_terminal_owner_ids & terminal_sources
            )
            if not owner_matches:
                # A candidate ID may project a newly rearmed entry path while
                # an older frozen path is still awaiting its own typed
                # terminal.  Replacement is no more causal than disappearance:
                # never apply the new episode's phase/reason to the old owner,
                # and never censor the old sample merely because the current
                # map selected a different unit.  The frozen path geometry,
                # location, deadline, explicit terminal_source_ids, or replay
                # boundary remains responsible for settlement.
                continue
            phase = hypothesis.phase
            reason = getattr(hypothesis, "terminal_reason", None)
            matching_terminal = phase in {
                PlaybookPhase.COMPLETED,
                PlaybookPhase.INVALIDATED,
            }
            pending = self._pending_bar_resolutions.get(
                sample.sample_id
            )
            draw_consumed_by_this_bar = bool(
                pending is not None
                and pending[2] == "draw_delivered"
                and sample.draw_id is not None
                and sample.draw_id in terminal_sources
                and reason
                in {
                    "context_draw_consumed_or_missing",
                    "primary_target_consumed_or_missing",
                    "selected_draw_consumed_or_missing",
                    "draw_consumed_or_missing",
                }
            )
            if (
                matching_terminal
                and phase is PlaybookPhase.INVALIDATED
                and draw_consumed_by_this_bar
            ):
                # The exact frozen draw disappearing because this completed
                # bar touched it is delivery, not missing custody.  Keep the
                # pending price outcome; any structural contradiction or
                # unrelated missing source still follows the terminal path.
                continue
            if matching_terminal and phase is PlaybookPhase.INVALIDATED:
                if reason in _CENSOR_TERMINAL_REASONS:
                    if (
                        sample.dimension == "thesis_strength"
                        and sample.playbook
                        == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
                    ):
                        # A child zone/path can become unobservable without
                        # censoring its still-live reversal Context Thesis.
                        pass
                    else:
                        self._finish(
                            sample,
                            resolved_at=asof,
                            outcome=None,
                            resolution=f"terminal_censored:{reason}",
                            censored=True,
                        )
                        continue
                elif reason in {"episode_deadline_elapsed", "deadline_elapsed"}:
                    if (
                        sample.dimension == "thesis_strength"
                        and sample.playbook
                        in {
                            Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
                            Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
                        }
                        and sample.deadline is not None
                        and sample.deadline > asof
                    ):
                        # A local entry episode can expire while its shared
                        # Context Thesis remains open under its own deadline.
                        pass
                    else:
                        self._deadline(
                            sample,
                            sample.deadline or asof,
                        )
                        continue
                elif (
                    sample.dimension == "thesis_strength"
                    and self._thesis_failure(sample, reason, terminal_sources)
                ):
                    self._finish(
                        sample,
                        resolved_at=asof,
                        outcome=0.0,
                        resolution=f"thesis_contradicted:{reason}",
                    )
                    continue
                elif sample.dimension == "thesis_strength":
                    if (
                        (
                            sample.playbook
                            == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
                            and reason in _DFP_LOCAL_EPISODE_TERMINAL_REASONS
                        )
                        or sample.playbook
                        == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
                        or reason == "global_frozen_source_invalidated"
                    ):
                        # A Context Thesis owns a longer lifecycle than any
                        # one entry-zone/trigger attempt.  LSR exact Context
                        # terminals were already consumed above from
                        # belief.context_theses; any remaining LSR terminal is
                        # local to one child and cannot settle this sample.
                        pass
                    else:
                        # A terminal outside the explicitly local DFP set
                        # cannot establish whether the frozen thesis survived.
                        # In particular, a missing draw identity must not be
                        # relabelled as successful merely because its price was
                        # not observed after custody was lost.
                        self._finish(
                            sample,
                            resolved_at=asof,
                            outcome=None,
                            resolution="non_thesis_terminal_unresolved",
                            censored=True,
                        )
                        continue
                elif (
                    sample.dimension == "location_quality"
                    and reason in _READINESS_FAILURE_REASONS
                ):
                    self._finish(
                        sample,
                        resolved_at=asof,
                        outcome=0.0,
                        resolution=f"location_failed:{reason}",
                    )
                    continue


    def _register_hypothesis(
        self,
        observation: Any,
        hypothesis: Any,
        locations: Mapping[str, Any],
        source_bar: Bar,
        *,
        candidate_id: str,
        global_context: Any | None,
    ) -> None:
        if self._terminal_owner_blocks_hypothesis(hypothesis):
            # A genuinely terminal Context cannot be rearmed by a later child
            # candidate projection.  DFP's recorder-local thesis horizon
            # deliberately does not enter this branch.
            return
        sequence = hypothesis.sequence
        if (
            sequence is None
            or sequence.setup_id is None
            or hypothesis.phase
            in {PlaybookPhase.INACTIVE, PlaybookPhase.COMPLETED}
        ):
            return
        path_id = self._entry_path_id(observation, hypothesis)
        raw = self._raw_dimensions(hypothesis)

        valid_trigger = self._valid_trigger_observation(
            hypothesis,
            raw,
        )
        if hypothesis.phase is PlaybookPhase.INVALIDATED:
            if valid_trigger:
                self._register_trigger_targets(
                    observation,
                    hypothesis,
                    raw=raw,
                    path_id=path_id,
                    source_bar=source_bar,
                    candidate_id=candidate_id,
                    global_context=global_context,
                )
            return

        revision = getattr(hypothesis, "evidence_revision_id", None)
        if revision is not None:
            self._register(
                observation,
                hypothesis,
                dimension="thesis_strength",
                raw_value=raw["thesis_strength"],
                revision_key=revision,
                entry_path_id=path_id,
                source_bar=source_bar,
                candidate_id=candidate_id,
                global_context=global_context,
            )
            self._register_descriptive(
                observation,
                hypothesis,
                dimension="uncertainty",
                raw_value=raw["uncertainty"],
                revision_key=_uncertainty_revision_key(hypothesis),
                entry_path_id=path_id,
                resolution="contemporaneous_uncertainty_observed",
                source_bar=source_bar,
                candidate_id=candidate_id,
                global_context=global_context,
            )

        sequence_key = (
            f"{sequence.setup_id}:{hypothesis.phase.value}:"
            f"{sequence.completed_steps}/{len(sequence.steps)}"
        )
        self._register_descriptive(
            observation,
            hypothesis,
            dimension="sequence_progress",
            raw_value=raw["sequence_progress"],
            revision_key=sequence_key,
            entry_path_id=path_id,
            resolution="sequence_observed",
            source_bar=source_bar,
            candidate_id=candidate_id,
            global_context=global_context,
        )

        location_id = getattr(hypothesis, "entry_location_id", None)
        location = locations.get(location_id)
        if location is not None:
            location_seen_key = (candidate_id, location.location_id)
            if location_seen_key not in self._seen_location_ids:
                if location.lifecycle in {
                    EntryLocationLifecycle.APPROACHING,
                    EntryLocationLifecycle.IN_ZONE,
                }:
                    registered = self._register(
                        observation,
                        hypothesis,
                        dimension="location_quality",
                        raw_value=raw["location_quality"],
                        revision_key=f"location:{location.location_id}",
                        entry_path_id=path_id,
                        source_bar=source_bar,
                        candidate_id=candidate_id,
                        global_context=global_context,
                    )
                    if registered:
                        self._seen_location_ids.add(location_seen_key)
                else:
                    # A terminal location first discovered at a replay/window
                    # boundary is audit context, not a calibration target.
                    self._seen_location_ids.add(location_seen_key)

        if valid_trigger:
            self._register_trigger_targets(
                observation,
                hypothesis,
                raw=raw,
                path_id=path_id,
                source_bar=source_bar,
                candidate_id=candidate_id,
                global_context=global_context,
            )

    @staticmethod
    def _valid_trigger_observation(
        hypothesis: Any,
        raw: Mapping[str, float],
    ) -> bool:
        sequence = hypothesis.sequence
        if not sequence.complete or raw["entry_readiness"] <= 0.0:
            return False
        if hypothesis.phase is not PlaybookPhase.INVALIDATED:
            return True
        # A completed, aligned trigger can share its clock with a delivery
        # veto (for example insufficient remaining path).  Explicit trigger,
        # structure, frozen-invalidation, deadline, or custody failures do
        # not describe a valid trigger and must not manufacture targets.
        reason = getattr(hypothesis, "terminal_reason", None)
        return reason not in _INVALID_TRIGGER_TERMINAL_REASONS

    def _register_trigger_targets(
        self,
        observation: Any,
        hypothesis: Any,
        *,
        raw: Mapping[str, float],
        path_id: str | None,
        source_bar: Bar,
        candidate_id: str,
        global_context: Any | None,
    ) -> None:
        sequence = hypothesis.sequence
        revision = getattr(hypothesis, "evidence_revision_id", None)
        frozen_trigger = getattr(hypothesis, "selected_trigger", None)
        trigger_id = (
            getattr(
                hypothesis,
                "selected_trigger_id",
                None
                if frozen_trigger is None
                else frozen_trigger.trigger_id,
            )
            or path_id
            or revision
            or sequence.setup_id
        )
        self._register(
            observation,
            hypothesis,
            dimension="entry_readiness",
            raw_value=raw["entry_readiness"],
            revision_key=f"trigger:{trigger_id}",
            entry_path_id=path_id,
            source_bar=source_bar,
            candidate_id=candidate_id,
            global_context=global_context,
        )
        if hypothesis.plan is not None:
            self._register(
                observation,
                hypothesis,
                dimension="delivery_quality",
                raw_value=raw["delivery_quality"],
                revision_key=f"delivery-trigger:{trigger_id}",
                entry_path_id=path_id,
                source_bar=source_bar,
                candidate_id=candidate_id,
                global_context=global_context,
            )

    def _register(
        self,
        observation: Any,
        hypothesis: Any,
        *,
        dimension: str,
        raw_value: float,
        revision_key: str,
        entry_path_id: str | None,
        source_bar: Bar,
        candidate_id: str,
        global_context: Any | None,
    ) -> bool:
        if self._calibration_unit(
            hypothesis,
            dimension,
            entry_path_id=entry_path_id,
        ) is None:
            # Missing ownership is not a late or geometry-incomplete sample:
            # no causal calibration unit exists yet.  Descriptive dimensions
            # are admitted independently by their own diagnostic identities.
            return False
        if (
            dimension in {"entry_readiness", "delivery_quality"}
            and getattr(hypothesis, "selected_trigger", None) is not None
            and self._selected_trigger_bar_geometry(
                observation,
                hypothesis,
                source_bar=source_bar,
                entry_path_id=entry_path_id,
            )
            is None
        ):
            # A newly materialized Episode cannot claim an older trigger
            # observed by an ownerless warmup root.  Repeated snapshots remain
            # harmless no-ops; only the exact trigger clock may seed geometry.
            return False
        sample = self._freeze_sample(
            observation,
            hypothesis,
            dimension=dimension,
            raw_value=raw_value,
            revision_key=revision_key,
            entry_path_id=entry_path_id,
            source_bar=source_bar,
            candidate_id=candidate_id,
            global_context=global_context,
        )
        if (
            sample.dimension == "thesis_strength"
            and self._terminal_thesis_key(sample)
            in self._terminal_thesis_units
        ):
            # A new evidence revision cannot reopen an already settled causal
            # owner.  Mark its exact sample identity seen so a future replay
            # snapshot cannot backfill it with a later clock.
            self._seen.add(sample.sample_id)
            return False
        if sample.sample_id in self._seen:
            prior = self._open.get(sample.sample_id)
            if (
                prior is not None
                and sample.dimension == "thesis_strength"
                and (
                    not math.isclose(
                        sample.raw_value,
                        prior.raw_value,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    or sample.direction != prior.direction
                    or sample.draw_id != prior.draw_id
                    or sample.draw_price != prior.draw_price
                    or sample.invalidation_source_id
                    != prior.invalidation_source_id
                    or sample.invalidation_price != prior.invalidation_price
                    or sample.deadline != prior.deadline
                )
            ):
                raise ValueError(
                    "Context thesis revision disagrees across child episodes"
                )
            return True
        thesis_required = (
            {
                "invalidation_price": sample.invalidation_price,
                "deadline": sample.deadline,
            }
            if sample.calibration_unit_kind == "lsr_context_thesis"
            else {
                "draw_price": sample.draw_price,
                "deadline": sample.deadline,
            }
        )
        required = {
            "thesis_strength": thesis_required,
            "location_quality": {
                "invalidation_price": sample.invalidation_price,
                "deadline": sample.deadline,
            },
            "entry_readiness": {
                "invalidation_price": sample.invalidation_price,
                "deadline": sample.deadline,
            },
            "delivery_quality": {
                "invalidation_price": sample.invalidation_price,
                "draw_price": sample.draw_price,
                "deadline": sample.deadline,
            },
        }[dimension]
        missing = tuple(
            name for name, value in required.items() if value is None
        )
        if missing:
            # A causal revision is sampled only once.  Rebuilding it later
            # would silently replace its clock/raw value (and for readiness,
            # its trigger extreme) with future information.
            self._seen.add(sample.sample_id)
            self._incomplete_registration_keys.add(
                (
                    sample.playbook,
                    sample.direction,
                    sample.dimension,
                    sample.calibration_unit_kind,
                    sample.calibration_unit_id,
                )
            )
            return False
        if sample.deadline is not None and sample.deadline <= sample.sampled_at:
            self._seen.add(sample.sample_id)
            self._late_registration_keys.add(
                (
                    sample.playbook,
                    sample.direction,
                    sample.calibration_unit_kind,
                    sample.calibration_unit_id,
                )
            )
            return True
        self._seen.add(sample.sample_id)
        self._open[sample.sample_id] = sample
        return True

    def _register_descriptive(
        self,
        observation: Any,
        hypothesis: Any,
        *,
        dimension: str,
        raw_value: float,
        revision_key: str,
        entry_path_id: str | None,
        resolution: str,
        source_bar: Bar,
        candidate_id: str,
        global_context: Any | None,
    ) -> None:
        if self._calibration_unit(
            hypothesis,
            dimension,
            entry_path_id=entry_path_id,
        ) is None:
            return
        sample = self._freeze_sample(
            observation,
            hypothesis,
            dimension=dimension,
            raw_value=raw_value,
            revision_key=revision_key,
            entry_path_id=entry_path_id,
            source_bar=source_bar,
            candidate_id=candidate_id,
            global_context=global_context,
        )
        if sample.sample_id in self._seen:
            return
        self._seen.add(sample.sample_id)
        self._finish(
            sample,
            resolved_at=sample.sampled_at,
            outcome=None,
            resolution=resolution,
        )

    def _freeze_sample(
        self,
        observation: Any,
        hypothesis: Any,
        *,
        dimension: str,
        raw_value: float,
        revision_key: str,
        entry_path_id: str | None,
        source_bar: Bar,
        candidate_id: str,
        global_context: Any | None,
    ) -> _OpenSample:
        sequence = hypothesis.sequence
        invalidation = hypothesis.invalidation
        plan = hypothesis.plan
        if invalidation is None and plan is not None:
            invalidation = plan.invalidation
        draw = getattr(hypothesis, "draw_selection", None)
        thesis_draw = getattr(hypothesis, "thesis_draw", None)
        targets = tuple(getattr(hypothesis, "deliverable_targets", ()))
        target = None
        if draw is not None:
            target = next(
                (item for item in targets if item.level_id == draw.draw_id),
                None,
            )
        if target is None and targets:
            target = targets[0]
        lsr_context_thesis = bool(
            dimension == "thesis_strength"
            and hypothesis.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        )
        if lsr_context_thesis:
            # The shared LSR Context is the frozen manipulation/sweep thesis.
            # Every zone below it owns an independent trade target; copying a
            # child draw here would let Episode A settle the Context and
            # suppress Episodes B/C.
            draw_id = None
            draw_price = None
        elif dimension == "thesis_strength" and thesis_draw is not None:
            draw_id = thesis_draw.level_id
            draw_price = thesis_draw.price
        else:
            draw_id = (
                draw.draw_id
                if draw is not None
                else None if target is None else target.level_id
            )
            draw_price = (
                draw.price
                if draw is not None
                else None if target is None else target.price
            )
        liquidity_route = getattr(hypothesis, "liquidity_route", None)
        if liquidity_route is None and plan is not None:
            liquidity_route = getattr(plan, "liquidity_route", None)
        if lsr_context_thesis:
            liquidity_route = None
        thesis_deadline = getattr(hypothesis, "thesis_deadline", None)
        entry_deadline = getattr(hypothesis, "episode_deadline", None)
        if dimension == "thesis_strength":
            deadline = thesis_deadline
            target_deadline_kind = "thesis_deadline"
        elif dimension == "entry_readiness":
            deadline = entry_deadline
            target_deadline_kind = "entry_deadline"
        elif dimension == "delivery_quality":
            deadline = None if plan is None else plan.deadline
            target_deadline_kind = "plan_deadline"
        elif dimension == "location_quality":
            deadline = entry_deadline
            target_deadline_kind = "entry_deadline"
        else:
            deadline = entry_deadline or thesis_deadline
            target_deadline_kind = (
                "entry_deadline"
                if entry_deadline is not None
                else "thesis_deadline"
            )
        sampled_at = aware_timestamp(
            observation.asof,
            name="brain_calibration.sampled_at",
        )
        if (
            deadline is None
            and dimension == "thesis_strength"
            and hypothesis.playbook
            in {
                Playbook.DISPLACEMENT_FIRST_PULLBACK,
                Playbook.LIQUIDITY_SWEEP_REVERSAL,
            }
        ):
            # A Context Thesis is intentionally not forced to expire with one
            # local entry episode.  Calibration still needs a frozen, finite
            # observation horizon so fitted rows have one comparable causal
            # target.  Use the registered market-session clock—not execution
            # availability—and keep it recorder-local.
            deadline = _registered_session_deadline(sampled_at)
        if deadline is None and dimension in DESCRIPTIVE_DIMENSIONS:
            execution = getattr(observation, "execution", None)
            minutes = (
                None
                if execution is None
                else getattr(execution, "minutes_to_deadline", None)
            )
            if minutes is not None and int(minutes) >= 0:
                deadline = sampled_at + pd.Timedelta(minutes=int(minutes))
        if (
            dimension == "thesis_strength"
            and hypothesis.playbook
            is Playbook.DISPLACEMENT_FIRST_PULLBACK
        ):
            # DFP's entry-zone protection is not its H4/H1 thesis
            # contradiction.  Opposed structure/BOS is resolved from the
            # typed terminal reason instead.
            invalidation = None
        dfp_structure_id = None
        dfp_structure_confirmed_at = None
        dfp_h1_bos_id = None
        if (
            dimension == "thesis_strength"
            and hypothesis.playbook
            is Playbook.DISPLACEMENT_FIRST_PULLBACK
        ):
            steps = {
                getattr(step, "step_id", None): step
                for step in getattr(sequence, "steps", ())
            }
            context_step = steps.get("h4_structure_and_draw")
            context_sources = tuple(
                getattr(context_step, "source_ids", ())
            )
            if context_sources:
                dfp_structure_id = str(context_sources[0])
                frame = getattr(observation, "frame", None)
                if callable(frame):
                    structure = next(
                        (
                            item
                            for item in frame(Timeframe.H4).structures
                            if item.structure_id == dfp_structure_id
                        ),
                        None,
                    )
                    if structure is not None:
                        dfp_structure_confirmed_at = structure.confirmed_at
                if dfp_structure_confirmed_at is None:
                    dfp_structure_confirmed_at = getattr(
                        context_step,
                        "observed_at",
                        None,
                    )
            h1_step = steps.get("h1_continuation_bos")
            h1_sources = tuple(getattr(h1_step, "source_ids", ()))
            if getattr(h1_step, "satisfied", False) and h1_sources:
                dfp_h1_bos_id = str(h1_sources[0])
        setup_id = getattr(hypothesis, "setup_context_id", None)
        if setup_id is None:
            setup_id = getattr(sequence, "setup_id", None)
        calibration_unit = self._calibration_unit(
            hypothesis,
            dimension,
            entry_path_id=entry_path_id,
        )
        if setup_id is None or calibration_unit is None:
            raise ValueError(
                f"{dimension} calibration sample lacks a setup or unit identity"
            )
        calibration_unit_id, calibration_unit_kind = calibration_unit
        selected_trigger = (
            getattr(hypothesis, "selected_trigger", None)
            if self._owns_selected_trigger_path(
                hypothesis,
                entry_path_id=entry_path_id,
            )
            else None
        )
        if selected_trigger is None:
            selected_trigger_id = None
            selected_trigger_kind = None
            selected_trigger_at = None
            available_trigger_kinds = ()
        else:
            selected_trigger_id = getattr(
                hypothesis,
                "selected_trigger_id",
                selected_trigger.trigger_id,
            )
            selected_trigger_kind = getattr(
                hypothesis,
                "selected_trigger_kind",
                selected_trigger.trigger_kind,
            )
            selected_trigger_at = getattr(
                hypothesis,
                "selected_trigger_at",
                selected_trigger.observed_at,
            )
            available_trigger_kinds = tuple(
                getattr(
                    hypothesis,
                    "available_trigger_kinds",
                    selected_trigger.available_trigger_kinds,
                )
            )
        trigger_geometry = self._selected_trigger_bar_geometry(
            observation,
            hypothesis,
            source_bar=source_bar,
            entry_path_id=entry_path_id,
        )
        trigger_bar_high, trigger_bar_low = (
            (float(source_bar.high), float(source_bar.low))
            if trigger_geometry is None
            else trigger_geometry
        )
        identity_payload = {
            # Context thesis revisions are shared across child EntryEpisodes;
            # zone/path dimensions remain owned by the exact action candidate.
            # This prevents multiple zones from manufacturing independent
            # copies of one reversal thesis while retaining child provenance
            # in the emitted ``hypothesis_key`` field.
            "owner_key": (
                calibration_unit_id
                if dimension == "thesis_strength"
                else candidate_id
            ),
            "dimension": dimension,
            "calibration_unit_id": calibration_unit_id,
            "calibration_unit_kind": calibration_unit_kind,
            "revision_key": str(revision_key),
        }
        sample_id = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        context_features = _hypothesis_context_features(
            hypothesis,
            global_context,
        )
        context_blocker_ids = context_features.pop(
            "path_blocker_ids",
            None,
        )
        if lsr_context_thesis:
            # Route geometry belongs to one zone-owned EntryEpisode.  The
            # shared reversal thesis retains only its sweep invalidation and
            # observation horizon, so a sibling can never inherit these
            # target/path diagnostics through a Context-owned row.
            context_blocker_ids = ()
            context_features.update(
                obstruction_distance_R=None,
                free_path_R=None,
                soft_obstruction_count=0,
                hard_barrier_before_target=False,
            )
        frozen_raw_value = _clamped(
            raw_value,
            name=f"raw.{dimension}",
        )
        if dimension == "uncertainty" and not math.isclose(
            frozen_raw_value,
            context_features["uncertainty_total"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "uncertainty raw value disagrees with component total"
            )
        return _OpenSample(
            sample_id=sample_id,
            hypothesis_key=candidate_id,
            playbook=hypothesis.playbook.value,
            direction=hypothesis.direction.value,
            dimension=dimension,
            setup_id=str(setup_id),
            calibration_unit_id=calibration_unit_id,
            calibration_unit_kind=calibration_unit_kind,
            episode_id=getattr(hypothesis, "episode_id", None),
            context_id=getattr(hypothesis, "context_id", None),
            context_thesis_id=getattr(
                hypothesis,
                "context_thesis_id",
                getattr(hypothesis, "context_id", None),
            ),
            parent_context_thesis_id=getattr(
                hypothesis,
                "parent_context_thesis_id",
                (
                    getattr(hypothesis, "context_thesis_id", None)
                    or getattr(hypothesis, "context_id", None)
                )
                if getattr(hypothesis, "episode_id", None) is not None
                else None,
            ),
            evidence_revision_id=getattr(
                hypothesis,
                "evidence_revision_id",
                None,
            ),
            entry_location_id=getattr(
                hypothesis,
                "entry_location_id",
                None,
            ),
            entry_path_id=entry_path_id,
            selected_trigger_id=selected_trigger_id,
            selected_trigger_kind=selected_trigger_kind,
            selected_trigger_at=_timestamp(
                selected_trigger_at,
                name="brain_calibration.selected_trigger_at",
            ),
            available_trigger_kinds=_encode_identities(
                available_trigger_kinds
            ),
            **context_features,
            sampled_at=sampled_at,
            raw_value=frozen_raw_value,
            phase=hypothesis.phase.value,
            origin_price=float(observation.price),
            trigger_bar_high=trigger_bar_high,
            trigger_bar_low=trigger_bar_low,
            invalidation_price=(
                None if invalidation is None else float(invalidation.price)
            ),
            invalidation_source_id=(
                None
                if invalidation is None
                else invalidation.source_level_id
            ),
            draw_price=None if draw_price is None else float(draw_price),
            draw_id=draw_id,
            liquidity_route_id=(
                None
                if liquidity_route is None
                else liquidity_route.route_id
            ),
            context_draw_id=(
                None
                if liquidity_route is None
                else liquidity_route.context_draw_id
            ),
            intermediate_liquidity_ids=_encode_identities(
                ()
                if liquidity_route is None
                else liquidity_route.intermediate_liquidity_ids
            ),
            primary_deliverable_target_id=(
                None
                if liquidity_route is None
                else liquidity_route.primary_deliverable_target_id
            ),
            terminal_draw_id=(
                None
                if liquidity_route is None
                else liquidity_route.terminal_draw_id
            ),
            authority_barrier_id=(
                None
                if liquidity_route is None
                else getattr(liquidity_route, "authority_barrier_id", None)
            ),
            authority_barrier_price=(
                None
                if liquidity_route is None
                else getattr(liquidity_route, "authority_barrier_price", None)
            ),
            path_blocker_ids=_encode_identities(
                (
                    context_blocker_ids
                    if context_blocker_ids is not None
                    else ()
                    if liquidity_route is None
                    else liquidity_route.path_blocker_ids
                )
            ),
            source_path_ids=_encode_identities(
                ()
                if liquidity_route is None
                else liquidity_route.source_path_ids
            ),
            dfp_structure_id=dfp_structure_id,
            dfp_structure_confirmed_at=_timestamp(
                dfp_structure_confirmed_at,
                name="brain_calibration.dfp_structure_confirmed_at",
            ),
            dfp_h1_bos_id=dfp_h1_bos_id,
            target_deadline_kind=target_deadline_kind,
            deadline=_timestamp(deadline, name="brain_calibration.deadline"),
            symbol=str(observation.symbol),
            instrument_id=int(observation.instrument_id),
        )

    @staticmethod
    def _calibration_unit(
        hypothesis: Any,
        dimension: str,
        *,
        entry_path_id: str | None = None,
    ) -> tuple[str, str] | None:
        if dimension == "thesis_strength":
            identity = getattr(hypothesis, "context_thesis_id", None)
            if hypothesis.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
                kind = "dfp_context_thesis"
            else:
                # LSR's sweep/reacceptance/reverse-displacement thesis is a
                # long-lived Context.  Each zone below it owns a separate
                # EntryEpisode, never a separate thesis calibration unit.
                kind = "lsr_context_thesis"
        elif dimension == "location_quality":
            identity = (
                entry_path_id
                if BrainCalibrationRecorder._owns_entry_episode(hypothesis)
                else None
            )
            kind = "entry_path_location"
        elif dimension in {"entry_readiness", "delivery_quality"}:
            identity = (
                entry_path_id
                if BrainCalibrationRecorder._owns_entry_episode(hypothesis)
                else None
            )
            kind = "trigger_entry_path"
        elif dimension == "sequence_progress":
            identity = getattr(hypothesis, "setup_context_id", None)
            kind = "hypothesis_sequence"
        else:
            identity = (
                getattr(hypothesis, "episode_id", None)
                or getattr(hypothesis, "context_id", None)
                or getattr(hypothesis, "setup_context_id", None)
            )
            kind = "decision_hypothesis"
        if not isinstance(identity, str) or not identity:
            return None
        return identity, kind

    @staticmethod
    def _owns_entry_episode(hypothesis: Any) -> bool:
        context_thesis_id = getattr(hypothesis, "context_thesis_id", None)
        parent_context_thesis_id = getattr(
            hypothesis,
            "parent_context_thesis_id",
            None,
        )
        episode_id = getattr(hypothesis, "episode_id", None)
        return bool(
            isinstance(context_thesis_id, str)
            and context_thesis_id
            and isinstance(episode_id, str)
            and episode_id
            and parent_context_thesis_id == context_thesis_id
        )

    @staticmethod
    def _raw_dimensions(hypothesis: Any) -> dict[str, float]:
        supplied = dict(getattr(hypothesis, "raw_quality_dimensions", {}))
        fallback = {
            "thesis_strength": (
                hypothesis.raw_probability
                if getattr(hypothesis, "raw_probability", None) is not None
                else hypothesis.thesis_strength
            ),
            "sequence_progress": hypothesis.sequence_progress,
            "location_quality": hypothesis.location_quality,
            "entry_readiness": hypothesis.entry_readiness,
            "delivery_quality": hypothesis.delivery_quality,
            "uncertainty": hypothesis.uncertainty,
        }
        output: dict[str, float] = {}
        for dimension in ALL_DIMENSIONS:
            value = supplied.get(dimension, fallback[dimension])
            if value is None:
                raise ValueError(
                    f"typed belief lacks raw {dimension} for calibration"
                )
            output[dimension] = _clamped(
                value,
                name=f"raw.{dimension}",
            )
        return output

    @staticmethod
    def _entry_path_id(observation: Any, hypothesis: Any) -> str | None:
        plan = hypothesis.plan
        explicit = getattr(hypothesis, "entry_path_id", None)
        plan_path = (
            None if plan is None else getattr(plan, "entry_path_id", None)
        )
        for name, identity in (
            ("belief", explicit),
            ("plan", plan_path),
        ):
            if identity is not None and (
                not isinstance(identity, str) or not identity
            ):
                raise ValueError(f"{name} entry path identity is invalid")
        if (
            explicit is not None
            and plan_path is not None
            and explicit != plan_path
        ):
            raise ValueError("belief and plan entry path identities disagree")
        paths = tuple(getattr(observation, "path_sequences", ()))
        frozen_path_id = explicit or plan_path
        if frozen_path_id is not None:
            exact = tuple(
                item
                for item in paths
                if getattr(item, "sequence_id", None) == frozen_path_id
            )
            if len(exact) > 1:
                raise ValueError("duplicate frozen entry path identity")
            if exact:
                path = exact[0]
                if (
                    path.context_kind != "zone_return"
                    or path.context_id
                    != getattr(hypothesis, "entry_location_id", None)
                    or path.direction is not hypothesis.direction
                ):
                    raise ValueError(
                        "frozen entry path disagrees with its belief owner"
                    )
            # The frozen path may legitimately be outside a bounded current
            # Scene Graph frame.  Its explicit belief identity remains the
            # owner for a planless WAITING_TRIGGER revision and after dormant
            # episode retention.
            return frozen_path_id
        location_id = getattr(hypothesis, "entry_location_id", None)
        matches = [
            item
            for item in paths
            if item.context_kind == "zone_return"
            and item.context_id == location_id
            and item.direction is hypothesis.direction
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda item: (item.last_updated_at, item.sequence_id),
        ).sequence_id

    @staticmethod
    def _terminal_thesis_key(
        sample: _OpenSample,
    ) -> tuple[str, str, str, str]:
        if sample.dimension != "thesis_strength":
            raise ValueError("terminal thesis owner requires a thesis sample")
        return (
            sample.playbook,
            sample.direction,
            sample.calibration_unit_kind,
            sample.calibration_unit_id,
        )

    @staticmethod
    def _terminal_thesis_scope(
        sample: _OpenSample,
        resolution: str,
    ) -> str:
        if resolution == "thesis_intact_at_deadline":
            # The recorder's market-session horizon closes only this fitted
            # thesis unit.  It is not a runtime Context invalidation and must
            # not suppress a later child episode's path/trigger targets.
            return "thesis_observation_horizon"
        return "context_terminal"

    def _terminal_thesis_for_hypothesis(
        self,
        hypothesis: Any,
    ) -> tuple[pd.Timestamp, float, str, str] | None:
        unit = self._calibration_unit(hypothesis, "thesis_strength")
        if unit is None:
            return None
        return self._terminal_thesis_units.get(
            (
                hypothesis.playbook.value,
                hypothesis.direction.value,
                unit[1],
                unit[0],
            )
        )

    def _terminal_owner_blocks_hypothesis(self, hypothesis: Any) -> bool:
        terminal = self._terminal_thesis_for_hypothesis(hypothesis)
        if terminal is None:
            return False
        scope = terminal[3]
        return scope == "context_terminal"

    def _promote_horizon_context_terminals(
        self,
        asof: pd.Timestamp,
        contexts: Mapping[str, Any],
    ) -> None:
        """Promote only an exact, typed runtime Context terminal.

        A recorder-local observation horizon settles the fitted thesis unit,
        but deliberately leaves later child episodes eligible.  If the
        runtime subsequently publishes that same long-lived Context as
        completed or invalidated, upgrade only the durable tombstone scope.
        The already emitted calibration row remains immutable.
        """

        if not isinstance(contexts, Mapping):
            raise ValueError("Brain Context Thesis projection is invalid")
        clock = aware_timestamp(
            asof,
            name="brain_calibration.context_terminal_observed_at",
        )
        for key, terminal in tuple(self._terminal_thesis_units.items()):
            playbook, direction, unit_kind, context_id = key
            prior_at, outcome, resolution, scope = terminal
            expected_kind = {
                Playbook.DISPLACEMENT_FIRST_PULLBACK.value:
                    "dfp_context_thesis",
                Playbook.LIQUIDITY_SWEEP_REVERSAL.value:
                    "lsr_context_thesis",
            }.get(playbook)
            if (
                unit_kind != expected_kind
                or scope != "thesis_observation_horizon"
            ):
                continue
            context = contexts.get(context_id)
            if context is None:
                continue
            context_direction = getattr(context, "direction", None)
            context_direction = getattr(
                context_direction,
                "value",
                context_direction,
            )
            if (
                getattr(context, "context_thesis_id", None) != context_id
                or context_direction != direction
            ):
                raise ValueError(
                    "Brain Context Thesis projection disagrees with its "
                    "terminal calibration owner"
                )
            lifecycle = getattr(context, "lifecycle", None)
            if lifecycle not in {"completed", "invalidated"}:
                continue
            terminal_at = _timestamp(
                getattr(context, "terminal_at", None),
                name="brain_calibration.context_terminal.terminal_at",
            )
            terminal_reason = getattr(context, "terminal_reason", None)
            if (
                terminal_at is None
                or terminal_at < prior_at
                or terminal_at > clock
                or not isinstance(terminal_reason, str)
                or not terminal_reason
            ):
                raise ValueError(
                    "Brain Context Thesis terminal state is invalid"
                )
            self._terminal_thesis_units[key] = (
                terminal_at,
                outcome,
                f"context_terminal:{terminal_reason}",
                "context_terminal",
            )

    def _finish(
        self,
        sample: _OpenSample,
        *,
        resolved_at: pd.Timestamp,
        outcome: float | None,
        resolution: str,
        censored: bool = False,
    ) -> None:
        if (
            sample.dimension == "thesis_strength"
            and not censored
            and outcome is not None
        ):
            key = self._terminal_thesis_key(sample)
            if key in self._terminal_thesis_units:
                # Iterators may still hold a peer removed by the first
                # terminal broadcast.  A durable owner tombstone makes that
                # stale visit a no-op instead of emitting a second outcome.
                self._open.pop(sample.sample_id, None)
                self._pending_bar_resolutions.pop(sample.sample_id, None)
                return
            scope = self._terminal_thesis_scope(sample, resolution)
            terminal = (
                aware_timestamp(
                    resolved_at,
                    name="brain_calibration.terminal_thesis.resolved_at",
                ),
                _clamped(outcome, name="terminal_thesis.outcome"),
                str(resolution),
                scope,
            )
            peers = tuple(
                peer
                for peer in self._open.values()
                if peer.dimension == "thesis_strength"
                and self._terminal_thesis_key(peer) == key
            )
            if not peers:
                peers = (sample,)
            self._terminal_thesis_units[key] = terminal
            for peer in peers:
                self._finish_one(
                    peer,
                    resolved_at=terminal[0],
                    outcome=terminal[1],
                    resolution=terminal[2],
                    censored=False,
                )
            return
        self._finish_one(
            sample,
            resolved_at=resolved_at,
            outcome=outcome,
            resolution=resolution,
            censored=censored,
        )

    def _finish_one(
        self,
        sample: _OpenSample,
        *,
        resolved_at: pd.Timestamp,
        outcome: float | None,
        resolution: str,
        censored: bool = False,
    ) -> None:
        self._open.pop(sample.sample_id, None)
        self._pending_bar_resolutions.pop(sample.sample_id, None)
        record = BrainCalibrationRecord(
            sample_id=sample.sample_id,
            hypothesis_key=sample.hypothesis_key,
            playbook=sample.playbook,
            direction=sample.direction,
            dimension=sample.dimension,
            setup_id=sample.setup_id,
            calibration_unit_id=sample.calibration_unit_id,
            calibration_unit_kind=sample.calibration_unit_kind,
            episode_id=sample.episode_id,
            context_id=sample.context_id,
            context_thesis_id=sample.context_thesis_id,
            parent_context_thesis_id=sample.parent_context_thesis_id,
            evidence_revision_id=sample.evidence_revision_id,
            entry_location_id=sample.entry_location_id,
            entry_path_id=sample.entry_path_id,
            selected_trigger_id=sample.selected_trigger_id,
            selected_trigger_kind=sample.selected_trigger_kind,
            selected_trigger_at=sample.selected_trigger_at,
            available_trigger_kinds=sample.available_trigger_kinds,
            market_thesis_id=sample.market_thesis_id,
            bound_market_thesis_id=sample.bound_market_thesis_id,
            market_thesis_root_id=sample.market_thesis_root_id,
            market_thesis_mechanism=sample.market_thesis_mechanism,
            market_thesis_authority_relation=(
                sample.market_thesis_authority_relation
            ),
            playbook_match_strength=sample.playbook_match_strength,
            market_thesis_binding_required=(
                sample.market_thesis_binding_required
            ),
            market_thesis_action_bound=(
                sample.market_thesis_action_bound
            ),
            market_thesis_match_status=(
                sample.market_thesis_match_status
            ),
            playbook_first_failed_hard_gate_id=(
                sample.playbook_first_failed_hard_gate_id
            ),
            playbook_plan_delivery_valid=(
                sample.playbook_plan_delivery_valid
            ),
            global_market_mode=sample.global_market_mode,
            authority_relation=sample.authority_relation,
            authority_rank_gap=sample.authority_rank_gap,
            conflict_role=sample.conflict_role,
            conflict_scope=sample.conflict_scope,
            acceptance_state=sample.acceptance_state,
            obstruction_distance_R=sample.obstruction_distance_R,
            free_path_R=sample.free_path_R,
            soft_obstruction_count=sample.soft_obstruction_count,
            hard_barrier_before_target=(
                sample.hard_barrier_before_target
            ),
            ambiguity_count=sample.ambiguity_count,
            uncertainty_conflict=sample.uncertainty_conflict,
            uncertainty_required_evidence_missing=(
                sample.uncertainty_required_evidence_missing
            ),
            uncertainty_authority_missing=(
                sample.uncertainty_authority_missing
            ),
            uncertainty_graph_ambiguity=(
                sample.uncertainty_graph_ambiguity
            ),
            uncertainty_total=sample.uncertainty_total,
            sampled_at=sample.sampled_at,
            resolved_at=resolved_at,
            raw_value=sample.raw_value,
            outcome_value=outcome,
            resolution=resolution,
            censored=censored,
            fit_eligible=bool(
                sample.dimension in FITTED_DIMENSIONS
                and not censored
                and outcome is not None
                and not (
                    sample.dimension == "delivery_quality"
                    and sample.hard_barrier_before_target
                )
            ),
            phase=sample.phase,
            origin_price=sample.origin_price,
            trigger_bar_high=sample.trigger_bar_high,
            trigger_bar_low=sample.trigger_bar_low,
            invalidation_price=sample.invalidation_price,
            invalidation_source_id=sample.invalidation_source_id,
            draw_price=sample.draw_price,
            draw_id=sample.draw_id,
            liquidity_route_id=sample.liquidity_route_id,
            context_draw_id=sample.context_draw_id,
            intermediate_liquidity_ids=(
                sample.intermediate_liquidity_ids
            ),
            primary_deliverable_target_id=(
                sample.primary_deliverable_target_id
            ),
            terminal_draw_id=sample.terminal_draw_id,
            authority_barrier_id=sample.authority_barrier_id,
            authority_barrier_price=sample.authority_barrier_price,
            path_blocker_ids=sample.path_blocker_ids,
            source_path_ids=sample.source_path_ids,
            dfp_structure_id=sample.dfp_structure_id,
            dfp_structure_confirmed_at=(
                sample.dfp_structure_confirmed_at
            ),
            dfp_h1_bos_id=sample.dfp_h1_bos_id,
            target_deadline_kind=sample.target_deadline_kind,
            deadline=sample.deadline,
            symbol=sample.symbol,
            instrument_id=sample.instrument_id,
        )
        self._rows.append(record)

    def _censor_all(self, asof: pd.Timestamp, reason: str) -> None:
        clock = aware_timestamp(asof, name="brain_calibration.censor_at")
        for sample in tuple(self._open.values()):
            self._censor_sample(sample, clock, reason)

    def _censor_sample(
        self,
        sample: _OpenSample,
        asof: pd.Timestamp,
        reason: str,
    ) -> None:
        clock = aware_timestamp(asof, name="brain_calibration.censor_at")
        if sample.deadline is not None:
            clock = min(clock, sample.deadline)
        self._finish(
            sample,
            resolved_at=max(clock, sample.sampled_at),
            outcome=None,
            resolution=reason,
            censored=True,
        )

    @staticmethod
    def _serialize_open(sample: _OpenSample) -> dict[str, Any]:
        payload = asdict(sample)
        payload["sampled_at"] = _iso(sample.sampled_at)
        payload["selected_trigger_at"] = _iso(sample.selected_trigger_at)
        payload["dfp_structure_confirmed_at"] = _iso(
            sample.dfp_structure_confirmed_at
        )
        payload["deadline"] = _iso(sample.deadline)
        return payload

    @staticmethod
    def _deserialize_open(payload: Mapping[str, Any]) -> _OpenSample:
        values = dict(payload)
        values["sampled_at"] = _timestamp(
            values.get("sampled_at"),
            name="brain_calibration.sampled_at",
        )
        values["selected_trigger_at"] = _timestamp(
            values.get("selected_trigger_at"),
            name="brain_calibration.selected_trigger_at",
        )
        values["deadline"] = _timestamp(
            values.get("deadline"),
            name="brain_calibration.deadline",
        )
        values["dfp_structure_confirmed_at"] = _timestamp(
            values.get("dfp_structure_confirmed_at"),
            name="brain_calibration.dfp_structure_confirmed_at",
        )
        return _OpenSample(**values)

    @staticmethod
    def _serialize_row(row: BrainCalibrationRecord) -> dict[str, Any]:
        payload = asdict(row)
        payload["sampled_at"] = _iso(row.sampled_at)
        payload["resolved_at"] = _iso(row.resolved_at)
        payload["selected_trigger_at"] = _iso(row.selected_trigger_at)
        payload["dfp_structure_confirmed_at"] = _iso(
            row.dfp_structure_confirmed_at
        )
        payload["deadline"] = _iso(row.deadline)
        return payload

    @staticmethod
    def _deserialize_row(payload: Mapping[str, Any]) -> BrainCalibrationRecord:
        values = dict(payload)
        for name in (
            "sampled_at",
            "resolved_at",
            "selected_trigger_at",
            "dfp_structure_confirmed_at",
            "deadline",
        ):
            values[name] = _timestamp(
                values.get(name),
                name=f"brain_calibration.{name}",
            )
        return BrainCalibrationRecord(**values)


__all__ = [
    "ALL_DIMENSIONS",
    "BrainCalibrationRecord",
    "BrainCalibrationRecorder",
    "DESCRIPTIVE_DIMENSIONS",
    "FITTED_DIMENSIONS",
    "RECORDER_SCHEMA_VERSION",
    "SUPPORTED_PLAYBOOKS",
]
