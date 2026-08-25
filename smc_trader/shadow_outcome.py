"""Outcome-blind shadow candidates for one frozen replay.

The recorder is intentionally downstream-only: it observes typed Eye deltas,
substantive OpenMarketThesis revisions, and Brain/Decision diagnostics, but
none of its state is fed back into the engine.  Candidate geometry is frozen
at registration and price outcomes are read only from later completed real 1m
bars.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Mapping

import pandas as pd

from .brain_entry_sequence import brain_observation_view
from .model import (
    BOSLifecycle,
    Bar,
    Direction,
    EngineSnapshot,
    EntryLocationLifecycle,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    Playbook,
    PlaybookPhase,
    ReacceptanceLifecycle,
    StructureLifecycle,
    Timeframe,
    aware_timestamp,
)


RECORDER_SCHEMA_VERSION = 8
SHADOW_DERIVED_SCHEMA_VERSION = 9
SHADOW_MOTIF_ROOT_SAMPLE_LIMIT = 40

# This is a run-level contract.  It should be copied once into a run manifest,
# not repeated in every output row.
SHADOW_OUTCOME_PROTOCOL: Mapping[str, Any] = {
    "protocol_version": "shadow-candidate-outcome-1.7.0",
    "candidate_clock": "completed_typed_transition_or_open_thesis_revision",
    "typed_transition_delta_required_for_eye_events": True,
    "open_market_thesis_revision_rule": (
        "one_candidate_per_root_and_substantive_evidence_revision"
    ),
    "lsr_eligible_zone_rule": (
        "formation_clock_approaching_entry_location_under_exact_frozen_"
        "lsr_context_displacement"
    ),
    "lsr_zone_episode_binding_rule": (
        "formation_clock_freezes_unique_active_exact_market_belief_context_"
        "and_entry_episode_custody_by_entry_location_then_entry_path_then_"
        "trigger;zero_or_multiple_active_matches_fail_closed_without_"
        "sibling_or_future_fallback;retained_owner_custody_is_descriptive_"
        "and_never_actionable"
    ),
    "candidate_events": (
        "playbook_executable",
        "open_market_thesis_revision",
        "confirmed_bos",
        "displacement_active",
        "eligible_entry_fvg",
        "eligible_entry_order_block",
        "first_entry_fvg",
        "first_entry_order_block",
        "liquidity_sweep",
        "mature_range_reentry",
        "qualified_micro_bos",
        "qualified_reacceptance",
    ),
    "non_zone_entry_rule": "next_real_1m_open",
    "zone_entry_rule": "next_later_touch_of_frozen_zone_midpoint",
    "playbook_entry_rule": "next_later_touch_of_frozen_planned_entry",
    "outcome_horizon_real_1m_bars": 60,
    "same_bar_priority": "invalidation_before_target",
    "path_validity": "frozen_target_before_frozen_invalidation",
    "frozen_fields": (
        "direction",
        "source_episode_id",
        "source_setup_id",
        "source_context_thesis_id",
        "entry_episode_terminal_at",
        "entry_episode_terminal_reason",
        "entry_location_id",
        "entry_path_id",
        "lsr_displacement_id",
        "lsr_entry_zone_id",
        "entry_episode_binding_status",
        "entry_rule",
        "entry_reference",
        "invalidation",
        "draw",
        "target",
        "deadline_at",
        "deadline_real_1m_bars",
    ),
    "model_feedback": "none_shadow_only",
}

_HORIZON_REAL_BARS = int(
    SHADOW_OUTCOME_PROTOCOL["outcome_horizon_real_1m_bars"]
)
_BOUNDARY_ANOMALIES = frozenset(
    {
        "contract_change_history_reset",
        "data_gap_history_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
_PLAYBOOKS = tuple(Playbook)
SHADOW_SEQUENCE_PLAYBOOKS = (
    Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
    Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
)
_COMPLETE_SEQUENCE_STEPS: Mapping[str, tuple[str, ...]] = {
    Playbook.DISPLACEMENT_FIRST_PULLBACK.value: (
        "h4_structure_and_draw",
        "m5_displacement_zone",
        "first_pullback_to_frozen_zone",
        "typed_entry_trigger",
    ),
    Playbook.LIQUIDITY_SWEEP_REVERSAL.value: (
        "formed_pool_sweep",
        "outside_acceptance_failed",
        "reverse_displacement_zone",
        "reversal_first_pullback",
        "typed_entry_trigger",
    ),
}
_PLAYBOOK_EXECUTABLE_EVENT_KIND = "playbook_executable"
_LSR_ZONE_EVENT_KINDS = frozenset(
    {
        "eligible_entry_fvg",
        "eligible_entry_order_block",
        "first_entry_fvg",
        "first_entry_order_block",
    }
)
_EXACT_ENTRY_EPISODE_BINDINGS = frozenset(
    {
        "exact_entry_location",
        "exact_entry_path",
        "exact_trigger_binding",
    }
)
_NEUTRAL_CANDIDATE_EVENT_KINDS = frozenset(
    str(kind)
    for kind in SHADOW_OUTCOME_PROTOCOL["candidate_events"]
    if kind != _PLAYBOOK_EXECUTABLE_EVENT_KIND
)
_NEUTRAL_CANDIDATE_EVENT_ORDER = {
    kind: index
    for index, kind in enumerate(SHADOW_OUTCOME_PROTOCOL["candidate_events"])
    if kind != _PLAYBOOK_EXECUTABLE_EVENT_KIND
}
_EXPIRED_RESOLUTIONS = frozenset(
    {
        "deadline_no_delivery",
        "entry_unfilled_deadline",
        "deadline_inside_completed_bar_censored",
    }
)


@dataclass(frozen=True)
class ShadowCandidateOutcomeRecord:
    """One compact candidate with root-specific diagnostics encoded as JSON."""

    candidate_id: str
    event_kind: str
    event_id: str
    observed_at: pd.Timestamp
    resolved_at: pd.Timestamp
    symbol: str
    instrument_id: int
    direction: str
    source_timeframe: str
    source_ids: str
    candidate_origin: str
    source_playbook: str | None
    source_episode_id: str | None
    source_setup_id: str | None
    source_context_thesis_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    lsr_displacement_id: str | None
    lsr_entry_zone_id: str | None
    entry_episode_binding_status: str | None
    entry_episode_terminal_at: pd.Timestamp | None
    entry_episode_terminal_reason: str | None
    selected_trigger_kind: str | None
    selected_trigger_id: str | None
    selected_trigger_at: pd.Timestamp | None
    available_trigger_kinds: str
    decision_price: float
    entry_rule: str
    entry_reference_price: float | None
    entry_price: float | None
    entry_at: pd.Timestamp | None
    entry_zone_lower: float | None
    entry_zone_upper: float | None
    invalidation_price: float | None
    invalidation_source_id: str | None
    draw_id: str | None
    target_price: float | None
    deadline_at: pd.Timestamp | None
    deadline_real_1m_bars: int | None
    elapsed_real_1m_bars: int
    geometry_complete: bool
    geometry_incomplete_reason: str | None
    filled: bool
    resolution: str
    censored: bool
    target_before_invalidation: bool | None
    invalidation_before_target: bool | None
    same_bar_collision: bool
    mfe_points: float | None
    mae_points: float | None
    mfe_R: float | None
    mae_R: float | None
    hit_0_5R: bool | None
    hit_1R: bool | None
    hit_2R: bool | None
    time_to_draw_real_bars: int | None
    wait_improvement_points: float | None
    zone_departed_before_terminal: bool | None
    zone_departure_kind: str | None
    structural_thesis_invalidated: bool
    first_changed_evidence_id: str | None
    first_changed_evidence_at: pd.Timestamp | None
    brain_phase: str | None
    thesis_strength: float | None
    sequence_progress: float | None
    location_quality: float | None
    entry_readiness: float | None
    delivery_quality: float | None
    uncertainty: float | None
    decision_action: str
    risk_action: str
    playbook_outcomes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowEpisodeOutcomeRecord:
    """One terminal row for one first-executable playbook episode."""

    episode_key: str
    candidate_id: str
    playbook: str
    direction: str
    episode_id: str
    setup_id: str | None
    context_thesis_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    lsr_displacement_id: str | None
    lsr_entry_zone_id: str | None
    first_executable_at: pd.Timestamp
    observed_at: pd.Timestamp
    resolved_at: pd.Timestamp
    symbol: str
    instrument_id: int
    source_timeframe: str
    market_thesis_id: str | None
    bound_market_thesis_id: str | None
    market_thesis_root_id: str | None
    market_thesis_mechanism: str | None
    market_thesis_authority_relation: str | None
    selected_trigger_kind: str | None
    selected_trigger_id: str | None
    selected_trigger_at: pd.Timestamp | None
    available_trigger_kinds: str
    planned_entry_price: float
    entry_price: float | None
    entry_at: pd.Timestamp | None
    invalidation_price: float | None
    invalidation_source_id: str | None
    draw_id: str | None
    target_price: float | None
    deadline_at: pd.Timestamp | None
    frozen_target_R: float | None
    target_R: float | None
    target_R_bucket: str
    risk_qualified_target_R: bool
    planned_target_before_invalidation_deadline: bool | None
    geometry_complete: bool
    geometry_incomplete_reason: str | None
    filled: bool
    outcome_evaluable: bool
    outcome_class: str
    resolution: str
    censored: bool
    expired: bool
    target_first: bool | None
    invalidation_first: bool | None
    same_bar_collision: bool
    same_bar_stop_first: bool
    mfe_R: float | None
    mae_R: float | None
    hit_0_5R: bool | None
    hit_1R: bool | None
    hit_1_5R: bool | None
    hit_2R: bool | None
    time_to_draw_real_bars: int | None
    structural_thesis_invalidated: bool
    first_changed_evidence_id: str | None
    first_changed_evidence_at: pd.Timestamp | None
    phase: str
    thesis_strength: float | None
    sequence_progress: float | None
    location_quality: float | None
    entry_readiness: float | None
    delivery_quality: float | None
    uncertainty: float | None
    decision_action: str
    risk_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowMechanismChallengeRecord:
    """One neutral Eye candidate challenged against one active playbook."""

    challenge_id: str
    candidate_id: str
    event_kind: str
    event_id: str
    observed_at: pd.Timestamp
    resolved_at: pd.Timestamp
    symbol: str
    instrument_id: int
    direction: str
    source_timeframe: str
    playbook: str
    episode_id: str | None
    setup_id: str | None
    context_thesis_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    lsr_displacement_id: str | None
    lsr_entry_zone_id: str | None
    entry_episode_binding_status: str | None
    entry_episode_terminal_at: pd.Timestamp | None
    entry_episode_terminal_reason: str | None
    accepted_at_candidate_clock: bool
    first_failed_gate: str | None
    failed_gates: str
    event_order_signature: str
    plan_feasibility_valid: bool
    plan_feasibility_failure_reason: str | None
    graph_connected: bool
    exact_root_bound: bool
    market_thesis_match_status: str
    market_thesis_id: str | None
    market_thesis_root_id: str | None
    market_thesis_mechanism: str | None
    market_thesis_authority_relation: str | None
    liquidity_route_id: str | None
    context_draw_id: str | None
    intermediate_liquidity_ids: str
    primary_deliverable_target_id: str | None
    terminal_draw_id: str | None
    authority_barrier_id: str | None
    authority_barrier_price: float | None
    entry_price: float | None
    invalidation_price: float | None
    target_price: float | None
    target_R: float | None
    target_R_bucket: str
    risk_qualified_target_R: bool
    planned_target_before_invalidation_deadline: bool | None
    geometry_complete: bool
    geometry_incomplete_reason: str | None
    filled: bool
    outcome_evaluable: bool
    outcome_class: str
    resolution: str
    censored: bool
    same_bar_collision: bool
    mfe_R: float | None
    mae_R: float | None
    hit_0_5R: bool | None
    hit_1R: bool | None
    hit_1_5R: bool | None
    hit_2R: bool | None
    time_to_draw_real_bars: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowRootEpisodeRecord:
    """One finalization-only row for one independent market-thesis root.

    Multiple substantive revisions of one root remain useful diagnostics, but
    they are not independent market samples.  This record therefore freezes
    the first revision with complete geometry (or the first revision when no
    complete geometry exists) without consulting its later outcome.
    """

    root_episode_key: str
    root_id: str | None
    candidate_id: str
    observed_at: pd.Timestamp
    session_date_ny: str
    symbol: str
    instrument_id: int
    direction: str
    market_mechanism: str
    source_timeframe: str
    authority_relation: str
    event_order_signature: str
    rejected_gates: str
    playbook_liquidity_routes: str
    target_R_bucket: str
    dfp_rejected: bool
    lsr_rejected: bool
    geometry_complete: bool
    filled: bool
    censored: bool
    expired: bool
    outcome_evaluable: bool
    path_valid: bool
    risk_qualified_target_R: bool
    eligible_episode_evidence: bool
    action_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="shadow_root_episode.observed_at",
            ),
        )
        if self.action_authority:
            raise ValueError("shadow root episodes cannot authorize actions")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowRootSequenceRecord:
    """Outcome-blind lifecycle sequence for one strict root/episode scope.

    This is deliberately separate from :class:`ShadowRootEpisodeRecord`.
    The latter freezes an early geometry-complete revision for unbiased path
    comparison.  A sequence census instead needs every candidate-clock
    revision belonging to the same explicit identity, while never consulting
    any later price outcome when selecting its deepest observed sequence.
    """

    sequence_unit_key: str
    root_episode_key: str
    root_id: str
    symbol: str
    instrument_id: int
    direction: str
    playbook: str
    scope_kind: str
    scope_id: str
    episode_id: str | None
    setup_id: str | None
    market_mechanism: str
    source_timeframe: str
    authority_relation: str
    selected_candidate_id: str
    selected_event_kind: str
    selected_event_id: str
    first_observed_at: pd.Timestamp
    selected_at: pd.Timestamp
    last_observed_at: pd.Timestamp
    revision_count: int
    primitive_event_counts: str
    primitive_first_observed_at: str
    candidate_event_order_signature: str
    event_order_signature: str
    observed_sequence_signatures: str
    step_first_observed_at: str
    event_order_length: int
    event_bigrams: str
    event_trigrams: str
    signature_variant_count: int
    complete_sequence_observed: bool
    accepted_revision_count: int
    ever_accepted: bool
    first_accepted_at: pd.Timestamp | None
    first_failed_gate: str | None
    first_failed_gate_at: pd.Timestamp | None
    selected_revision_accepted: bool
    selected_first_failed_gate: str | None
    selected_failed_gates: str
    selected_plan_feasibility_valid: bool
    selected_graph_connected: bool
    selected_exact_root_bound: bool
    lifecycle_terminal_status: str = "unknown_not_recorded"
    selection_basis: str = (
        "longest_candidate_clock_signature_then_first_observed"
    )
    action_authority: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "first_observed_at",
            "selected_at",
            "last_observed_at",
        ):
            object.__setattr__(
                self,
                field_name,
                aware_timestamp(
                    getattr(self, field_name),
                    name=f"shadow_root_sequence.{field_name}",
                ),
            )
        if self.first_accepted_at is not None:
            object.__setattr__(
                self,
                "first_accepted_at",
                aware_timestamp(
                    self.first_accepted_at,
                    name="shadow_root_sequence.first_accepted_at",
                ),
            )
        if self.first_failed_gate_at is not None:
            object.__setattr__(
                self,
                "first_failed_gate_at",
                aware_timestamp(
                    self.first_failed_gate_at,
                    name="shadow_root_sequence.first_failed_gate_at",
                ),
            )
        try:
            primitive_counts = json.loads(self.primitive_event_counts)
            primitive_clocks = json.loads(
                self.primitive_first_observed_at
            )
            candidate_events = json.loads(
                self.candidate_event_order_signature
            )
            selected_signature = json.loads(self.event_order_signature)
            signatures = json.loads(self.observed_sequence_signatures)
            step_clocks = json.loads(self.step_first_observed_at)
            bigrams = json.loads(self.event_bigrams)
            trigrams = json.loads(self.event_trigrams)
            failed_gates = json.loads(self.selected_failed_gates)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("shadow root sequence JSON is invalid") from exc
        valid_scope = self.scope_kind in {
            "context_thesis",
            "setup_context",
            "entry_episode",
        }
        complete_steps = _COMPLETE_SEQUENCE_STEPS.get(self.playbook)
        if (
            self.action_authority
            or not valid_scope
            or self.playbook not in SHADOW_SEQUENCE_PLAYBOOKS
            or not self.selected_candidate_id
            or not self.selected_event_kind
            or not self.selected_event_id
            or self.revision_count < 1
            or self.signature_variant_count < 1
            or self.signature_variant_count != len(signatures)
            or self.event_order_length != len(selected_signature)
            or self.event_order_length < 1
            or not (
                self.first_observed_at
                <= self.selected_at
                <= self.last_observed_at
            )
            or self.accepted_revision_count < 0
            or self.accepted_revision_count > self.revision_count
            or self.ever_accepted != (self.accepted_revision_count > 0)
            or self.ever_accepted != (self.first_accepted_at is not None)
            or (
                self.first_accepted_at is not None
                and not (
                    self.first_observed_at
                    <= self.first_accepted_at
                    <= self.last_observed_at
                )
            )
            or (self.first_failed_gate is None)
            != (self.first_failed_gate_at is None)
            or (
                self.first_failed_gate_at is not None
                and not (
                    self.first_observed_at
                    <= self.first_failed_gate_at
                    <= self.last_observed_at
                )
            )
            or not isinstance(primitive_counts, dict)
            or sum(primitive_counts.values()) != self.revision_count
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, int)
                or value < 1
                for key, value in primitive_counts.items()
            )
            or not isinstance(primitive_clocks, dict)
            or set(primitive_clocks) != set(primitive_counts)
            or any(
                not isinstance(value, str)
                or aware_timestamp(
                    value,
                    name="shadow_root_sequence.primitive_first_observed_at",
                )
                < self.first_observed_at
                or aware_timestamp(
                    value,
                    name="shadow_root_sequence.primitive_first_observed_at",
                )
                > self.last_observed_at
                for value in primitive_clocks.values()
            )
            or not isinstance(candidate_events, list)
            or not candidate_events
            or len(candidate_events) != len(set(candidate_events))
            or any(not isinstance(value, str) or not value for value in candidate_events)
            or not isinstance(selected_signature, list)
            or any(not isinstance(value, str) or not value for value in selected_signature)
            or not isinstance(signatures, list)
            or any(
                not isinstance(signature, list)
                or not signature
                or any(not isinstance(value, str) or not value for value in signature)
                for signature in signatures
            )
            or selected_signature not in signatures
            or not isinstance(step_clocks, dict)
            or set(step_clocks) != {
                value for signature in signatures for value in signature
            }
            or any(
                not isinstance(value, str)
                or aware_timestamp(
                    value,
                    name="shadow_root_sequence.step_first_observed_at",
                )
                < self.first_observed_at
                or aware_timestamp(
                    value,
                    name="shadow_root_sequence.step_first_observed_at",
                )
                > self.last_observed_at
                for value in step_clocks.values()
            )
            or not isinstance(bigrams, list)
            or any(not isinstance(value, list) or len(value) != 2 for value in bigrams)
            or not isinstance(trigrams, list)
            or any(not isinstance(value, list) or len(value) != 3 for value in trigrams)
            or not isinstance(failed_gates, list)
            or any(not isinstance(value, str) or not value for value in failed_gates)
            or self.complete_sequence_observed
            != (complete_steps is not None and tuple(selected_signature) == complete_steps)
            or self.selection_basis
            != "longest_candidate_clock_signature_then_first_observed"
            or self.lifecycle_terminal_status != "unknown_not_recorded"
        ):
            raise ValueError("shadow root sequence contract is inconsistent")
        if self.scope_kind == "entry_episode":
            if self.episode_id != self.scope_id or not self.episode_id:
                raise ValueError("entry sequence scope lacks its episode identity")
        elif self.episode_id is not None:
            raise ValueError("non-entry sequence scope carries an episode identity")
        if self.scope_kind == "setup_context":
            if self.setup_id != self.scope_id or not self.setup_id:
                raise ValueError("setup sequence scope lacks its setup identity")
        elif self.scope_kind == "context_thesis" and self.setup_id is not None:
            raise ValueError("context-only sequence carries a setup identity")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowMechanismMotifRecord:
    """Cross-date motif with only a bounded diagnostic root-ID sample."""

    motif_id: str
    market_mechanism: str
    source_timeframe: str
    authority_relation: str
    event_order_signature: str
    rejected_gates: str
    target_R_bucket: str
    episode_count: int
    eligible_episode_count: int
    distinct_dates: int
    eligible_distinct_dates: int
    sample_root_episode_keys: str
    sample_root_episode_count: int
    root_episode_keys_truncated: bool
    eligible_for_preregistration_review: bool
    action_authority: bool = False

    def __post_init__(self) -> None:
        try:
            root_sample = json.loads(self.sample_root_episode_keys)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "shadow mechanism motif root sample must be JSON"
            ) from exc
        if (
            self.action_authority
            or self.episode_count < 1
            or self.eligible_episode_count < 0
            or self.eligible_episode_count > self.episode_count
            or self.distinct_dates < 1
            or self.eligible_distinct_dates < 0
            or self.eligible_distinct_dates > self.distinct_dates
            or self.eligible_for_preregistration_review
            != (
                self.eligible_episode_count >= 2
                and self.eligible_distinct_dates >= 2
            )
            or not isinstance(root_sample, list)
            or len(root_sample) > SHADOW_MOTIF_ROOT_SAMPLE_LIMIT
            or any(
                not isinstance(value, str) or not value
                for value in root_sample
            )
            or root_sample != sorted(set(root_sample))
            or self.sample_root_episode_count != len(root_sample)
            or self.sample_root_episode_count > self.episode_count
            or self.root_episode_keys_truncated
            != (self.episode_count > self.sample_root_episode_count)
            or (
                self.root_episode_keys_truncated
                and self.sample_root_episode_count
                != SHADOW_MOTIF_ROOT_SAMPLE_LIMIT
            )
        ):
            raise ValueError("shadow mechanism motif contract is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SHADOW_OUTCOME_FIELD_TYPES: Mapping[str, str] = {
    "candidate_id": "large_string",
    "event_kind": "large_string",
    "event_id": "large_string",
    "observed_at": "timestamp_ny",
    "resolved_at": "timestamp_ny",
    "symbol": "large_string",
    "instrument_id": "int64",
    "direction": "large_string",
    "source_timeframe": "large_string",
    "source_ids": "large_string",
    "candidate_origin": "large_string",
    "source_playbook": "large_string",
    "source_episode_id": "large_string",
    "source_setup_id": "large_string",
    "source_context_thesis_id": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "lsr_displacement_id": "large_string",
    "lsr_entry_zone_id": "large_string",
    "entry_episode_binding_status": "large_string",
    "entry_episode_terminal_at": "timestamp_ny",
    "entry_episode_terminal_reason": "large_string",
    "selected_trigger_kind": "large_string",
    "selected_trigger_id": "large_string",
    "selected_trigger_at": "timestamp_ny",
    "available_trigger_kinds": "large_string",
    "decision_price": "float64",
    "entry_rule": "large_string",
    "entry_reference_price": "float64",
    "entry_price": "float64",
    "entry_at": "timestamp_ny",
    "entry_zone_lower": "float64",
    "entry_zone_upper": "float64",
    "invalidation_price": "float64",
    "invalidation_source_id": "large_string",
    "draw_id": "large_string",
    "target_price": "float64",
    "deadline_at": "timestamp_ny",
    "deadline_real_1m_bars": "int64",
    "elapsed_real_1m_bars": "int64",
    "geometry_complete": "bool",
    "geometry_incomplete_reason": "large_string",
    "filled": "bool",
    "resolution": "large_string",
    "censored": "bool",
    "target_before_invalidation": "bool",
    "invalidation_before_target": "bool",
    "same_bar_collision": "bool",
    "mfe_points": "float64",
    "mae_points": "float64",
    "mfe_R": "float64",
    "mae_R": "float64",
    "hit_0_5R": "bool",
    "hit_1R": "bool",
    "hit_2R": "bool",
    "time_to_draw_real_bars": "int64",
    "wait_improvement_points": "float64",
    "zone_departed_before_terminal": "bool",
    "zone_departure_kind": "large_string",
    "structural_thesis_invalidated": "bool",
    "first_changed_evidence_id": "large_string",
    "first_changed_evidence_at": "timestamp_ny",
    "brain_phase": "large_string",
    "thesis_strength": "float64",
    "sequence_progress": "float64",
    "location_quality": "float64",
    "entry_readiness": "float64",
    "delivery_quality": "float64",
    "uncertainty": "float64",
    "decision_action": "large_string",
    "risk_action": "large_string",
    "playbook_outcomes": "large_string",
}


SHADOW_EPISODE_OUTCOME_FIELD_TYPES: Mapping[str, str] = {
    "episode_key": "large_string",
    "candidate_id": "large_string",
    "playbook": "large_string",
    "direction": "large_string",
    "episode_id": "large_string",
    "setup_id": "large_string",
    "context_thesis_id": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "lsr_displacement_id": "large_string",
    "lsr_entry_zone_id": "large_string",
    "first_executable_at": "timestamp_ny",
    "observed_at": "timestamp_ny",
    "resolved_at": "timestamp_ny",
    "symbol": "large_string",
    "instrument_id": "int64",
    "source_timeframe": "large_string",
    "market_thesis_id": "large_string",
    "bound_market_thesis_id": "large_string",
    "market_thesis_root_id": "large_string",
    "market_thesis_mechanism": "large_string",
    "market_thesis_authority_relation": "large_string",
    "selected_trigger_kind": "large_string",
    "selected_trigger_id": "large_string",
    "selected_trigger_at": "timestamp_ny",
    "available_trigger_kinds": "large_string",
    "planned_entry_price": "float64",
    "entry_price": "float64",
    "entry_at": "timestamp_ny",
    "invalidation_price": "float64",
    "invalidation_source_id": "large_string",
    "draw_id": "large_string",
    "target_price": "float64",
    "deadline_at": "timestamp_ny",
    "frozen_target_R": "float64",
    "target_R": "float64",
    "target_R_bucket": "large_string",
    "risk_qualified_target_R": "bool",
    "planned_target_before_invalidation_deadline": "bool",
    "geometry_complete": "bool",
    "geometry_incomplete_reason": "large_string",
    "filled": "bool",
    "outcome_evaluable": "bool",
    "outcome_class": "large_string",
    "resolution": "large_string",
    "censored": "bool",
    "expired": "bool",
    "target_first": "bool",
    "invalidation_first": "bool",
    "same_bar_collision": "bool",
    "same_bar_stop_first": "bool",
    "mfe_R": "float64",
    "mae_R": "float64",
    "hit_0_5R": "bool",
    "hit_1R": "bool",
    "hit_1_5R": "bool",
    "hit_2R": "bool",
    "time_to_draw_real_bars": "int64",
    "structural_thesis_invalidated": "bool",
    "first_changed_evidence_id": "large_string",
    "first_changed_evidence_at": "timestamp_ny",
    "phase": "large_string",
    "thesis_strength": "float64",
    "sequence_progress": "float64",
    "location_quality": "float64",
    "entry_readiness": "float64",
    "delivery_quality": "float64",
    "uncertainty": "float64",
    "decision_action": "large_string",
    "risk_action": "large_string",
}


SHADOW_MECHANISM_CHALLENGE_FIELD_TYPES: Mapping[str, str] = {
    "challenge_id": "large_string",
    "candidate_id": "large_string",
    "event_kind": "large_string",
    "event_id": "large_string",
    "observed_at": "timestamp_ny",
    "resolved_at": "timestamp_ny",
    "symbol": "large_string",
    "instrument_id": "int64",
    "direction": "large_string",
    "source_timeframe": "large_string",
    "playbook": "large_string",
    "episode_id": "large_string",
    "setup_id": "large_string",
    "context_thesis_id": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "lsr_displacement_id": "large_string",
    "lsr_entry_zone_id": "large_string",
    "entry_episode_binding_status": "large_string",
    "entry_episode_terminal_at": "timestamp_ny",
    "entry_episode_terminal_reason": "large_string",
    "accepted_at_candidate_clock": "bool",
    "first_failed_gate": "large_string",
    "failed_gates": "large_string",
    "event_order_signature": "large_string",
    "plan_feasibility_valid": "bool",
    "plan_feasibility_failure_reason": "large_string",
    "graph_connected": "bool",
    "exact_root_bound": "bool",
    "market_thesis_match_status": "large_string",
    "market_thesis_id": "large_string",
    "market_thesis_root_id": "large_string",
    "market_thesis_mechanism": "large_string",
    "market_thesis_authority_relation": "large_string",
    "liquidity_route_id": "large_string",
    "context_draw_id": "large_string",
    "intermediate_liquidity_ids": "large_string",
    "primary_deliverable_target_id": "large_string",
    "terminal_draw_id": "large_string",
    "authority_barrier_id": "large_string",
    "authority_barrier_price": "float64",
    "entry_price": "float64",
    "invalidation_price": "float64",
    "target_price": "float64",
    "target_R": "float64",
    "target_R_bucket": "large_string",
    "risk_qualified_target_R": "bool",
    "planned_target_before_invalidation_deadline": "bool",
    "geometry_complete": "bool",
    "geometry_incomplete_reason": "large_string",
    "filled": "bool",
    "outcome_evaluable": "bool",
    "outcome_class": "large_string",
    "resolution": "large_string",
    "censored": "bool",
    "same_bar_collision": "bool",
    "mfe_R": "float64",
    "mae_R": "float64",
    "hit_0_5R": "bool",
    "hit_1R": "bool",
    "hit_1_5R": "bool",
    "hit_2R": "bool",
    "time_to_draw_real_bars": "int64",
}


SHADOW_ROOT_EPISODE_FIELD_TYPES: Mapping[str, str] = {
    "root_episode_key": "large_string",
    "root_id": "large_string",
    "candidate_id": "large_string",
    "observed_at": "timestamp_ny",
    "session_date_ny": "large_string",
    "symbol": "large_string",
    "instrument_id": "int64",
    "direction": "large_string",
    "market_mechanism": "large_string",
    "source_timeframe": "large_string",
    "authority_relation": "large_string",
    "event_order_signature": "large_string",
    "rejected_gates": "large_string",
    "playbook_liquidity_routes": "large_string",
    "target_R_bucket": "large_string",
    "dfp_rejected": "bool",
    "lsr_rejected": "bool",
    "geometry_complete": "bool",
    "filled": "bool",
    "censored": "bool",
    "expired": "bool",
    "outcome_evaluable": "bool",
    "path_valid": "bool",
    "risk_qualified_target_R": "bool",
    "eligible_episode_evidence": "bool",
    "action_authority": "bool",
}


SHADOW_ROOT_SEQUENCE_FIELD_TYPES: Mapping[str, str] = {
    "sequence_unit_key": "large_string",
    "root_episode_key": "large_string",
    "root_id": "large_string",
    "symbol": "large_string",
    "instrument_id": "int64",
    "direction": "large_string",
    "playbook": "large_string",
    "scope_kind": "large_string",
    "scope_id": "large_string",
    "episode_id": "large_string",
    "setup_id": "large_string",
    "market_mechanism": "large_string",
    "source_timeframe": "large_string",
    "authority_relation": "large_string",
    "selected_candidate_id": "large_string",
    "selected_event_kind": "large_string",
    "selected_event_id": "large_string",
    "first_observed_at": "timestamp_ny",
    "selected_at": "timestamp_ny",
    "last_observed_at": "timestamp_ny",
    "revision_count": "int64",
    "primitive_event_counts": "large_string",
    "primitive_first_observed_at": "large_string",
    "candidate_event_order_signature": "large_string",
    "event_order_signature": "large_string",
    "observed_sequence_signatures": "large_string",
    "step_first_observed_at": "large_string",
    "event_order_length": "int64",
    "event_bigrams": "large_string",
    "event_trigrams": "large_string",
    "signature_variant_count": "int64",
    "complete_sequence_observed": "bool",
    "accepted_revision_count": "int64",
    "ever_accepted": "bool",
    "first_accepted_at": "timestamp_ny",
    "first_failed_gate": "large_string",
    "first_failed_gate_at": "timestamp_ny",
    "selected_revision_accepted": "bool",
    "selected_first_failed_gate": "large_string",
    "selected_failed_gates": "large_string",
    "selected_plan_feasibility_valid": "bool",
    "selected_graph_connected": "bool",
    "selected_exact_root_bound": "bool",
    "lifecycle_terminal_status": "large_string",
    "selection_basis": "large_string",
    "action_authority": "bool",
}


SHADOW_MECHANISM_MOTIF_FIELD_TYPES: Mapping[str, str] = {
    "motif_id": "large_string",
    "market_mechanism": "large_string",
    "source_timeframe": "large_string",
    "authority_relation": "large_string",
    "event_order_signature": "large_string",
    "rejected_gates": "large_string",
    "target_R_bucket": "large_string",
    "episode_count": "int64",
    "eligible_episode_count": "int64",
    "distinct_dates": "int64",
    "eligible_distinct_dates": "int64",
    "sample_root_episode_keys": "large_string",
    "sample_root_episode_count": "int64",
    "root_episode_keys_truncated": "bool",
    "eligible_for_preregistration_review": "bool",
    "action_authority": "bool",
}


@dataclass(frozen=True)
class _CandidateSpec:
    event_kind: str
    event_id: str
    observed_at: pd.Timestamp
    direction: Direction
    source_timeframe: Timeframe
    source_ids: tuple[str, ...]
    entry_zone_lower: float | None = None
    entry_zone_upper: float | None = None
    direct_invalidation: float | None = None
    direct_invalidation_id: str | None = None
    direct_entry: float | None = None
    direct_target: float | None = None
    direct_draw_id: str | None = None
    deadline_at: pd.Timestamp | None = None
    candidate_origin: str = "shadow_eye_candidate"
    source_playbook: Playbook | None = None
    source_episode_id: str | None = None
    source_setup_id: str | None = None
    source_context_thesis_id: str | None = None
    entry_location_id: str | None = None
    entry_path_id: str | None = None
    lsr_displacement_id: str | None = None
    lsr_entry_zone_id: str | None = None
    source_action_candidate_id: str | None = None
    entry_episode_binding_status: str | None = None
    entry_episode_terminal_at: pd.Timestamp | None = None
    entry_episode_terminal_reason: str | None = None
    selected_trigger_kind: str | None = None
    selected_trigger_id: str | None = None
    selected_trigger_at: pd.Timestamp | None = None
    available_trigger_kinds: tuple[str, ...] = ()
    allow_unlinked_draw_fallback: bool = True
    allow_context_link_expansion: bool = True


@dataclass(frozen=True)
class _LSRZoneCustody:
    """Formation-clock ownership frozen for one physical LSR entry zone."""

    source_action_candidate_id: str | None
    source_episode_id: str | None
    source_setup_id: str | None
    source_context_thesis_id: str | None
    entry_location_id: str
    entry_path_id: str | None
    lsr_displacement_id: str
    lsr_entry_zone_id: str
    binding_status: str
    direction: Direction
    formed_at: pd.Timestamp
    owner_terminal_at: pd.Timestamp | None = None
    owner_terminal_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self,
            "formed_at",
            aware_timestamp(
                self.formed_at,
                name="shadow_outcome.lsr_zone_custody.formed_at",
            ),
        )
        if self.owner_terminal_at is not None:
            object.__setattr__(
                self,
                "owner_terminal_at",
                aware_timestamp(
                    self.owner_terminal_at,
                    name="shadow_outcome.lsr_zone_custody.owner_terminal_at",
                ),
            )
        required = (
            self.entry_location_id,
            self.lsr_displacement_id,
            self.lsr_entry_zone_id,
            self.binding_status,
        )
        optional = (
            self.source_action_candidate_id,
            self.source_episode_id,
            self.source_setup_id,
            self.source_context_thesis_id,
            self.entry_path_id,
        )
        if any(not isinstance(value, str) or not value for value in required):
            raise ValueError("LSR zone custody has an invalid required identity")
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in optional
        ):
            raise ValueError("LSR zone custody has an invalid optional identity")
        exact = self.binding_status in _EXACT_ENTRY_EPISODE_BINDINGS
        exact_identities = (
            self.source_action_candidate_id,
            self.source_episode_id,
            self.source_setup_id,
            self.source_context_thesis_id,
        )
        if exact and (
            any(value is None for value in exact_identities)
            or self.source_episode_id != self.source_setup_id
            or self.entry_path_id is None
        ):
            raise ValueError("exact LSR zone custody is identity-incomplete")
        terminal = self.owner_terminal_at is not None
        if (
            terminal != (self.owner_terminal_reason is not None)
            or (
                self.owner_terminal_reason is not None
                and (
                    not isinstance(self.owner_terminal_reason, str)
                    or not self.owner_terminal_reason
                )
            )
            or (
                self.owner_terminal_at is not None
                and self.owner_terminal_at < self.formed_at
            )
            or terminal and not exact
        ):
            raise ValueError("LSR zone custody terminal tombstone is invalid")


@dataclass
class _OpenCandidate:
    candidate_id: str
    event_kind: str
    event_id: str
    observed_at: pd.Timestamp
    symbol: str
    instrument_id: int
    direction: Direction
    source_timeframe: Timeframe
    source_ids: tuple[str, ...]
    candidate_origin: str
    source_playbook: Playbook | None
    source_episode_id: str | None
    source_setup_id: str | None
    source_context_thesis_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    lsr_displacement_id: str | None
    lsr_entry_zone_id: str | None
    entry_episode_binding_status: str | None
    entry_episode_terminal_at: pd.Timestamp | None
    entry_episode_terminal_reason: str | None
    selected_trigger_kind: str | None
    selected_trigger_id: str | None
    selected_trigger_at: pd.Timestamp | None
    available_trigger_kinds: tuple[str, ...]
    decision_price: float
    entry_rule: str
    entry_reference_price: float | None
    entry_zone_lower: float | None
    entry_zone_upper: float | None
    invalidation_price: float
    invalidation_source_id: str
    draw_id: str
    target_price: float
    deadline_at: pd.Timestamp | None
    deadline_real_1m_bars: int | None
    playbook_diagnostics: tuple[dict[str, Any], ...]
    decision_action: str
    risk_action: str
    brain_phase: str | None
    thesis_strength: float | None
    sequence_progress: float | None
    location_quality: float | None
    entry_readiness: float | None
    delivery_quality: float | None
    uncertainty: float | None
    geometry_incomplete_reason: str | None = None
    elapsed_real_1m_bars: int = 0
    entry_price: float | None = None
    entry_at: pd.Timestamp | None = None
    mfe_points: float = 0.0
    mae_points: float = 0.0
    same_bar_collision: bool = False
    target_before_invalidation: bool | None = None
    invalidation_before_target: bool | None = None
    time_to_draw_real_bars: int | None = None
    zone_departed_at: pd.Timestamp | None = None
    zone_departure_kind: str | None = None
    structural_thesis_invalidated: bool = False
    first_changed_evidence_id: str | None = None
    first_changed_evidence_at: pd.Timestamp | None = None

    @property
    def initial_risk(self) -> float:
        if self.entry_price is None:
            return 0.0
        return abs(float(self.entry_price) - float(self.invalidation_price))


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hypothesis_context_identity(
    hypothesis: Any,
    name: str,
) -> str | None:
    direct = getattr(hypothesis, name, None)
    if isinstance(direct, str) and direct and direct not in {"none", "unknown"}:
        return direct
    metadata = getattr(hypothesis, "context_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(name)
    return (
        value
        if isinstance(value, str)
        and value
        and value not in {"none", "unknown"}
        else None
    )


def _plan_feasibility_state(hypothesis: Any) -> tuple[bool, str | None]:
    """Read the Brain-owned common plan contract without re-deriving it."""

    feasibility = getattr(hypothesis, "plan_feasibility", None)
    if feasibility is None:
        return False, "plan_feasibility_missing"
    valid = getattr(feasibility, "valid", None)
    reason = getattr(feasibility, "failure_reason", None)
    if type(valid) is not bool:
        raise ValueError("shadow hypothesis has invalid plan feasibility flag")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError("shadow hypothesis has invalid plan feasibility reason")
    if valid and reason is not None:
        raise ValueError("valid plan feasibility cannot carry a failure reason")
    if not valid and reason is None:
        reason = "plan_feasibility_invalid_unspecified"
    return valid, reason


def _frozen_event_order_signature(
    hypothesis: Any | None,
    *,
    fallback_event_kind: str,
) -> tuple[str, ...]:
    """Freeze the causal steps visible at the candidate clock.

    Sequence steps are historical episode facts.  Sorting their completed-bar
    clocks makes this independent of later revisions while preserving the
    protocol order for simultaneous transitions.
    """

    sequence = None if hypothesis is None else getattr(hypothesis, "sequence", None)
    ordered: list[tuple[int, int, str]] = []
    for index, step in enumerate(
        () if sequence is None else getattr(sequence, "steps", ())
    ):
        if not bool(getattr(step, "satisfied", False)):
            continue
        observed_at = getattr(step, "observed_at", None)
        if observed_at is None:
            raise ValueError("satisfied shadow sequence step lacks its clock")
        clock = aware_timestamp(
            observed_at,
            name="shadow_outcome.sequence_step.observed_at",
        )
        ordered.append((int(clock.value), index, str(step.step_id)))
    if not ordered:
        return (fallback_event_kind,)
    return tuple(
        step_id for _, _, step_id in sorted(ordered)
    )


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _liquidity_route_diagnostics(hypothesis: Any | None) -> dict[str, Any]:
    """Freeze the compact Brain-owned liquidity-route identity at this clock."""

    route = None if hypothesis is None else getattr(
        hypothesis,
        "liquidity_route",
        None,
    )
    if route is None and hypothesis is not None:
        plan = getattr(hypothesis, "plan", None)
        route = None if plan is None else getattr(plan, "liquidity_route", None)
    if route is None:
        return {
            "liquidity_route_id": None,
            "context_draw_id": None,
            "intermediate_liquidity_ids": [],
            "primary_deliverable_target_id": None,
            "terminal_draw_id": None,
            "authority_barrier_id": None,
            "authority_barrier_price": None,
        }
    intermediates = getattr(route, "intermediate_liquidity_ids", ())
    if isinstance(intermediates, str) or not isinstance(
        intermediates,
        (tuple, list),
    ):
        raise ValueError("shadow liquidity route has invalid waypoint identities")
    return {
        "liquidity_route_id": getattr(route, "route_id", None),
        "context_draw_id": getattr(route, "context_draw_id", None),
        "intermediate_liquidity_ids": list(intermediates),
        "primary_deliverable_target_id": getattr(
            route,
            "primary_deliverable_target_id",
            None,
        ),
        "terminal_draw_id": getattr(route, "terminal_draw_id", None),
        "authority_barrier_id": getattr(route, "authority_barrier_id", None),
        "authority_barrier_price": _finite(
            getattr(route, "authority_barrier_price", None)
        ),
    }


def _action_candidate_items(
    belief: Any,
) -> tuple[tuple[str, Any], ...]:
    """Read only the Brain's root-specific action-candidate contract."""

    accessor = getattr(belief, "action_candidate_items", None)
    if not callable(accessor):
        raise TypeError(
            "shadow outcome recording requires MarketBelief."
            "action_candidate_items()"
        )
    items = tuple(accessor())
    if any(
        not isinstance(identity, str)
        or not identity
        or (
            getattr(hypothesis, "candidate_id", None) is not None
            and getattr(hypothesis, "candidate_id") != identity
        )
        for identity, hypothesis in items
    ):
        raise ValueError("invalid root-specific action candidate mapping")
    return items


def _lifecycle_candidate_items(
    belief: Any,
) -> tuple[tuple[str, Any], ...]:
    """Read action plus retained/position lifecycle identities for custody."""

    accessor = getattr(belief, "lifecycle_candidate_items", None)
    if not callable(accessor):
        return _action_candidate_items(belief)
    items = tuple(accessor())
    if any(
        not isinstance(identity, str)
        or not identity
        or getattr(hypothesis, "candidate_id", None) != identity
        for identity, hypothesis in items
    ):
        raise ValueError("invalid lifecycle candidate mapping")
    if len({identity for identity, _ in items}) != len(items):
        raise ValueError("lifecycle candidate mapping contains duplicates")
    return items


def _candidate_entry_path_ids(hypothesis: Any) -> tuple[str, ...]:
    plan = getattr(hypothesis, "plan", None)
    trigger = getattr(hypothesis, "selected_trigger", None)
    return tuple(
        dict.fromkeys(
            value
            for value in (
                getattr(hypothesis, "entry_path_id", None),
                None if plan is None else getattr(plan, "entry_path_id", None),
                (
                    None
                    if trigger is None
                    else getattr(trigger, "entry_path_id", None)
                ),
            )
            if isinstance(value, str) and value
        )
    )


def _candidate_direct_entry_location_ids(hypothesis: Any) -> tuple[str, ...]:
    plan = getattr(hypothesis, "plan", None)
    return tuple(
        dict.fromkeys(
            value
            for value in (
                getattr(hypothesis, "entry_location_id", None),
                (
                    None
                    if plan is None
                    else getattr(plan, "entry_location_id", None)
                ),
            )
            if isinstance(value, str) and value
        )
    )


def _entry_location_path_ids(
    observation: Any,
    location_id: str,
) -> tuple[str, ...]:
    observation = brain_observation_view(observation)
    return tuple(
        sorted(
            {
                str(path.sequence_id)
                for path in observation.path_sequences
                if getattr(path, "context_kind", None) == "zone_return"
                and getattr(path, "context_id", None) == location_id
                and isinstance(getattr(path, "sequence_id", None), str)
                and bool(path.sequence_id)
            }
        )
    )


def _lsr_active_market_belief_projection(
    belief: Any,
    candidate_id: str,
    hypothesis: Any,
) -> tuple[Any, Any] | None:
    """Return the exact live Context/EntryEpisode projection for a root.

    Terminal grace action candidates are intentionally rejected here, before
    location/path/trigger match cardinality is evaluated.  They may remain in
    the action-candidate view for lifecycle settlement, but cannot compete for
    a newly observed physical zone.
    """

    entry_episodes = getattr(belief, "entry_episodes", None)
    context_theses = getattr(belief, "context_theses", None)
    if not isinstance(entry_episodes, Mapping) or not isinstance(
        context_theses,
        Mapping,
    ):
        return None
    if (
        getattr(hypothesis, "candidate_id", None) != candidate_id
        or getattr(hypothesis, "playbook", None)
        is not Playbook.LIQUIDITY_SWEEP_REVERSAL
        or getattr(hypothesis, "phase", None)
        in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
    ):
        return None
    episode_id = getattr(hypothesis, "episode_id", None)
    setup_id = getattr(hypothesis, "setup_context_id", None)
    context_id = getattr(hypothesis, "context_thesis_id", None)
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or setup_id != episode_id
        or not isinstance(context_id, str)
        or not context_id
        or getattr(hypothesis, "parent_context_thesis_id", None)
        != context_id
    ):
        return None
    episode = entry_episodes.get(candidate_id)
    context = context_theses.get(context_id)
    if episode is None or context is None:
        return None
    candidate_location_id = getattr(hypothesis, "entry_location_id", None)
    candidate_path_id = getattr(hypothesis, "entry_path_id", None)
    episode_location_id = getattr(episode, "entry_location_id", None)
    episode_path_id = getattr(episode, "entry_path_id", None)
    if not bool(
        getattr(episode, "candidate_id", None) == candidate_id
        and getattr(episode, "episode_id", None) == episode_id
        and getattr(episode, "parent_context_thesis_id", None) == context_id
        and getattr(episode, "playbook", None)
        is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and getattr(episode, "direction", None)
        is getattr(hypothesis, "direction", None)
        and getattr(episode, "phase", None)
        not in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
        and getattr(episode, "terminal_at", None) is None
        and (
            candidate_location_id is None
            or episode_location_id is None
            or candidate_location_id == episode_location_id
        )
        and (
            candidate_path_id is None
            or episode_path_id is None
            or candidate_path_id == episode_path_id
        )
        and getattr(context, "context_thesis_id", None) == context_id
        and getattr(context, "direction", None)
        is getattr(hypothesis, "direction", None)
        and getattr(context, "lifecycle", None)
        in {"forming", "active", "weakening"}
        and getattr(context, "terminal_at", None) is None
        and episode_id in tuple(getattr(context, "child_episode_ids", ()))
    ):
        return None
    return episode, context


def _lsr_projected_entry_location_ids(
    hypothesis: Any,
    episode: Any,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *_candidate_direct_entry_location_ids(hypothesis),
                *(
                    (episode.entry_location_id,)
                    if isinstance(
                        getattr(episode, "entry_location_id", None),
                        str,
                    )
                    and episode.entry_location_id
                    else ()
                ),
            )
        )
    )


def _lsr_projected_entry_path_ids(
    hypothesis: Any,
    episode: Any,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *_candidate_entry_path_ids(hypothesis),
                *(
                    (episode.entry_path_id,)
                    if isinstance(getattr(episode, "entry_path_id", None), str)
                    and episode.entry_path_id
                    else ()
                ),
            )
        )
    )


def _lsr_projected_triggers(
    hypothesis: Any,
    episode: Any,
) -> tuple[Any, ...]:
    output: list[Any] = []
    for item in (
        getattr(hypothesis, "selected_trigger", None),
        getattr(episode, "selected_trigger", None),
    ):
        if item is not None and all(item is not seen for seen in output):
            output.append(item)
    return tuple(output)


def _lsr_zone_trigger_clock_is_causal(
    hypothesis: Any,
    episode: Any,
    location: Any,
) -> bool:
    """Reject a trigger that predates formation or typed first pullback."""

    triggers = _lsr_projected_triggers(hypothesis, episode)
    if not triggers:
        return True
    trigger_ids = {
        getattr(item, "trigger_id", None)
        for item in triggers
        if isinstance(getattr(item, "trigger_id", None), str)
        and item.trigger_id
    }
    if len(trigger_ids) != 1:
        return False
    first_pullback_at = getattr(episode, "first_pullback_at", None)
    if first_pullback_at is None:
        first_pullback_at = getattr(hypothesis, "first_pullback_at", None)
    if first_pullback_at is None:
        return False
    formed_at = aware_timestamp(
        location.formed_at,
        name="shadow_outcome.lsr_zone.formed_at",
    )
    first_pullback_at = aware_timestamp(
        first_pullback_at,
        name="shadow_outcome.lsr_zone.first_pullback_at",
    )
    if first_pullback_at < formed_at:
        return False
    path_ids = _lsr_projected_entry_path_ids(hypothesis, episode)
    if len(path_ids) != 1:
        return False
    for trigger in triggers:
        trigger_at = getattr(trigger, "observed_at", None)
        if trigger_at is None:
            return False
        trigger_at = aware_timestamp(
            trigger_at,
            name="shadow_outcome.lsr_zone.trigger_at",
        )
        if trigger_at <= first_pullback_at or trigger_at <= formed_at:
            return False
        if getattr(trigger, "setup_id", None) != getattr(
            hypothesis,
            "episode_id",
            None,
        ):
            return False
        if getattr(trigger, "entry_location_id", None) != location.location_id:
            return False
        if getattr(trigger, "entry_path_id", None) != path_ids[0]:
            return False
    return True


def _valid_lsr_zone_hypothesis(
    hypothesis: Any,
    location: Any,
) -> bool:
    context_id = getattr(hypothesis, "context_thesis_id", None)
    episode_id = getattr(hypothesis, "episode_id", None)
    direct_location_ids = _candidate_direct_entry_location_ids(hypothesis)
    return bool(
        getattr(hypothesis, "playbook", None)
        is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and getattr(hypothesis, "direction", None) is location.direction
        and getattr(hypothesis, "phase", None)
        not in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
        and isinstance(context_id, str)
        and bool(context_id)
        and getattr(hypothesis, "parent_context_thesis_id", None)
        == context_id
        and isinstance(episode_id, str)
        and bool(episode_id)
        and (
            not direct_location_ids
            or direct_location_ids == (location.location_id,)
        )
        and isinstance(getattr(hypothesis, "setup_context_id", None), str)
        and bool(hypothesis.setup_context_id)
        and _hypothesis_context_identity(
            hypothesis,
            "lsr_displacement_id",
        )
        == location.source_displacement_id
        and _hypothesis_context_identity(
            hypothesis,
            "lsr_entry_zone_id",
        )
        == location.source_zone_id
        and isinstance(getattr(hypothesis, "required_root_id", None), str)
        and bool(hypothesis.required_root_id)
        and isinstance(
            getattr(hypothesis, "bound_market_thesis_id", None),
            str,
        )
        and bool(hypothesis.bound_market_thesis_id)
        and getattr(hypothesis, "market_thesis_action_bound", False) is True
        and getattr(hypothesis, "market_thesis_match_status", None)
        == "exact_root_bound"
    )


def _lsr_zone_spec_matches(hypothesis: Any, spec: "_CandidateSpec") -> bool:
    trigger = getattr(hypothesis, "selected_trigger", None)
    location_binding = bool(
        spec.entry_location_id
        in _candidate_direct_entry_location_ids(hypothesis)
    )
    path_binding = bool(
        spec.entry_path_id is not None
        and spec.entry_path_id in _candidate_entry_path_ids(hypothesis)
    )
    trigger_binding = bool(
        trigger is not None
        and (
            getattr(trigger, "entry_location_id", None)
            == spec.entry_location_id
            or (
                spec.entry_path_id is not None
                and getattr(trigger, "entry_path_id", None)
                == spec.entry_path_id
            )
        )
    )
    binding_matches = {
        "exact_entry_location": location_binding,
        "exact_entry_path": path_binding,
        "exact_trigger_binding": trigger_binding,
    }.get(str(spec.entry_episode_binding_status), False)
    return bool(
        getattr(hypothesis, "playbook", None)
        is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and getattr(hypothesis, "direction", None) is spec.direction
        and getattr(hypothesis, "episode_id", None) == spec.source_episode_id
        and getattr(hypothesis, "setup_context_id", None)
        == spec.source_setup_id
        and getattr(hypothesis, "context_thesis_id", None)
        == spec.source_context_thesis_id
        and getattr(hypothesis, "parent_context_thesis_id", None)
        == spec.source_context_thesis_id
        and binding_matches
        and (
            spec.entry_path_id is None
            or spec.entry_path_id in _candidate_entry_path_ids(hypothesis)
        )
        and _hypothesis_context_identity(
            hypothesis,
            "lsr_displacement_id",
        )
        == spec.lsr_displacement_id
        and _hypothesis_context_identity(
            hypothesis,
            "lsr_entry_zone_id",
        )
        == spec.lsr_entry_zone_id
        and getattr(hypothesis, "market_thesis_action_bound", False) is True
        and getattr(hypothesis, "market_thesis_match_status", None)
        == "exact_root_bound"
    )


def _lsr_zone_episode_binding(
    snapshot: EngineSnapshot,
    location: Any,
) -> tuple[str | None, Any | None, str, str | None]:
    """Resolve one zone event to one exact LSR EntryEpisode.

    Location identity has priority.  A frozen entry path, then a selected
    trigger explicitly bound to that same location/path, may recover older
    lifecycle projections that no longer expose ``entry_location_id``.
    Zero or multiple matches remain descriptive root events and never borrow
    a sibling via action-candidate ordering.
    """

    candidates = tuple(
        (identity, hypothesis, projection[0])
        for identity, hypothesis in _action_candidate_items(snapshot.belief)
        if getattr(hypothesis, "playbook", None)
        is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and getattr(hypothesis, "direction", None) is location.direction
        and (
            projection := _lsr_active_market_belief_projection(
                snapshot.belief,
                identity,
                hypothesis,
            )
        )
        is not None
    )
    direct = tuple(
        item
        for item in candidates
        if location.location_id
        in _lsr_projected_entry_location_ids(item[1], item[2])
    )
    binding_status = "exact_entry_location"
    matches = direct
    path_ids = _entry_location_path_ids(
        snapshot.observation,
        location.location_id,
    )
    path_id = path_ids[0] if len(path_ids) == 1 else None
    if not matches and path_id is not None:
        matches = tuple(
            item
            for item in candidates
            if path_id in _lsr_projected_entry_path_ids(item[1], item[2])
        )
        binding_status = "exact_entry_path"
    if not matches:
        matches = tuple(
            item
            for item in candidates
            for trigger in _lsr_projected_triggers(item[1], item[2])
            if (
                getattr(trigger, "entry_location_id", None)
                == location.location_id
                or (
                    path_id is not None
                    and getattr(trigger, "entry_path_id", None) == path_id
                )
            )
        )
        binding_status = "exact_trigger_binding"
    if not matches:
        return None, None, "entry_episode_binding_missing", path_id
    if len(matches) != 1:
        return None, None, "entry_episode_binding_ambiguous", path_id
    identity, hypothesis, episode = matches[0]
    if (
        not _valid_lsr_zone_hypothesis(hypothesis, location)
        or not _lsr_zone_trigger_clock_is_causal(
            hypothesis,
            episode,
            location,
        )
    ):
        return identity, hypothesis, "entry_episode_binding_invalid", path_id
    frozen_path_ids = _lsr_projected_entry_path_ids(hypothesis, episode)
    if len(frozen_path_ids) > 1:
        return identity, hypothesis, "entry_episode_binding_invalid", path_id
    if path_id is None:
        path_id = frozen_path_ids[0] if len(frozen_path_ids) == 1 else None
    elif frozen_path_ids and path_id not in frozen_path_ids:
        return identity, hypothesis, "entry_episode_binding_invalid", path_id
    if path_id is None:
        return identity, hypothesis, "entry_episode_binding_invalid", None
    return identity, hypothesis, binding_status, path_id


def _lsr_zone_context_binding(
    snapshot: EngineSnapshot,
    location: Any,
) -> tuple[Any | None, str]:
    """Resolve the parent Context independently of child creation.

    This keeps formation binding failures visible in the denominator: a
    Context-linked zone still produces an eligible-zone diagnostic when the
    Brain failed to materialize exactly one child EntryEpisode.
    """

    context_theses = getattr(snapshot.belief, "context_theses", None)
    if not isinstance(context_theses, Mapping):
        return None, "lsr_context_binding_missing"
    contexts = tuple(
        context
        for context_id, context in context_theses.items()
        if isinstance(context_id, str)
        and context_id
        and getattr(context, "context_thesis_id", None) == context_id
        and getattr(context, "direction", None) is location.direction
        and getattr(context, "lifecycle", None)
        in {"forming", "active", "weakening"}
        and getattr(context, "terminal_at", None) is None
        and location.source_displacement_id
        in tuple(getattr(context, "authority_ids", ()))
    )
    if not contexts:
        return None, "lsr_context_binding_missing"
    if len(contexts) != 1:
        return None, "lsr_context_binding_ambiguous"
    return contexts[0], "exact_lsr_context"


def _lsr_zone_custody_owner_is_terminal(
    belief: Any,
    custody: _LSRZoneCustody,
) -> bool:
    """Read a frozen owner's typed terminal projection without rebinding."""

    if custody.binding_status not in _EXACT_ENTRY_EPISODE_BINDINGS:
        return False
    if custody.owner_terminal_at is not None:
        return True
    entry_episodes = getattr(belief, "entry_episodes", None)
    context_theses = getattr(belief, "context_theses", None)
    if not isinstance(entry_episodes, Mapping) or not isinstance(
        context_theses,
        Mapping,
    ):
        return False
    episode = entry_episodes.get(custody.source_action_candidate_id)
    context = context_theses.get(custody.source_context_thesis_id)
    episode_terminal = bool(
        episode is not None
        and getattr(episode, "episode_id", None) == custody.source_episode_id
        and getattr(episode, "parent_context_thesis_id", None)
        == custody.source_context_thesis_id
        and (
            getattr(episode, "phase", None)
            in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
            or getattr(episode, "terminal_at", None) is not None
        )
    )
    context_terminal = bool(
        context is not None
        and getattr(context, "context_thesis_id", None)
        == custody.source_context_thesis_id
        and (
            getattr(context, "lifecycle", None)
            in {"completed", "invalidated", "censored"}
            or getattr(context, "terminal_at", None) is not None
        )
    )
    return episode_terminal or context_terminal


def _frozen_executable_binding_is_exact(
    belief: Any,
    candidate_id: str,
    hypothesis: Any,
    plan: Any,
    *,
    source_episode_id: str | None,
    source_setup_id: str | None,
) -> bool:
    """Validate an executable whose discovery root is temporarily absent.

    This is deliberately narrower than source-ID graph matching.  The exact
    action candidate, its typed EntryEpisode projection and its live parent
    Context must all agree before Shadow may consume the frozen binding.
    """

    entry_episodes = getattr(belief, "entry_episodes", None)
    context_theses = getattr(belief, "context_theses", None)
    if not isinstance(entry_episodes, Mapping) or not isinstance(
        context_theses,
        Mapping,
    ):
        return False
    episode = entry_episodes.get(candidate_id)
    context_id = getattr(hypothesis, "context_thesis_id", None)
    parent_context_id = getattr(
        hypothesis,
        "parent_context_thesis_id",
        None,
    )
    context = context_theses.get(context_id)
    episode_id = getattr(hypothesis, "episode_id", None)
    setup_id = getattr(hypothesis, "setup_context_id", None)
    entry_location_id = getattr(hypothesis, "entry_location_id", None)
    entry_path_id = getattr(hypothesis, "entry_path_id", None)
    required_root_id = getattr(hypothesis, "required_root_id", None)
    bound_thesis_id = getattr(
        hypothesis,
        "bound_market_thesis_id",
        None,
    )
    episode_plan = None if episode is None else getattr(episode, "plan", None)
    return bool(
        getattr(hypothesis, "candidate_id", None) == candidate_id
        and getattr(hypothesis, "record_kind", None) == "root_candidate"
        and isinstance(required_root_id, str)
        and bool(required_root_id)
        and getattr(hypothesis, "market_thesis_root_id", None)
        == required_root_id
        and isinstance(bound_thesis_id, str)
        and bool(bound_thesis_id)
        and getattr(hypothesis, "market_thesis_id", None)
        == bound_thesis_id
        and bound_thesis_id
        in tuple(getattr(hypothesis, "market_thesis_ids", ()))
        and getattr(hypothesis, "market_thesis_action_bound", False) is True
        and getattr(hypothesis, "market_thesis_match_status", None)
        == "exact_root_bound"
        and isinstance(episode_id, str)
        and bool(episode_id)
        and episode_id == source_episode_id == setup_id == source_setup_id
        and isinstance(context_id, str)
        and bool(context_id)
        and parent_context_id == context_id
        and isinstance(entry_location_id, str)
        and bool(entry_location_id)
        and isinstance(entry_path_id, str)
        and bool(entry_path_id)
        and getattr(plan, "setup_id", None) == setup_id
        and getattr(plan, "entry_location_id", None) == entry_location_id
        and getattr(plan, "entry_path_id", None) == entry_path_id
        and episode is not None
        and getattr(episode, "candidate_id", None) == candidate_id
        and getattr(episode, "episode_id", None) == episode_id
        and getattr(episode, "parent_context_thesis_id", None) == context_id
        and getattr(episode, "playbook", None) is hypothesis.playbook
        and getattr(episode, "direction", None) is hypothesis.direction
        and getattr(episode, "entry_location_id", None) == entry_location_id
        and getattr(episode, "entry_path_id", None) == entry_path_id
        and getattr(episode, "phase", None) is PlaybookPhase.EXECUTABLE
        and episode_plan is not None
        and getattr(episode_plan, "setup_id", None) == setup_id
        and getattr(episode_plan, "entry_location_id", None)
        == entry_location_id
        and getattr(episode_plan, "entry_path_id", None) == entry_path_id
        and context is not None
        and getattr(context, "context_thesis_id", None) == context_id
        and getattr(context, "lifecycle", None)
        not in {"completed", "invalidated", "censored"}
        and getattr(context, "terminal_at", None) is None
        and episode_id in tuple(getattr(context, "child_episode_ids", ()))
    )


def _frozen_lsr_zone_binding_is_exact(
    belief: Any,
    candidate_id: str,
    hypothesis: Any,
    *,
    source_episode_id: str | None,
    source_context_thesis_id: str | None,
    entry_location_id: str | None,
    entry_path_id: str | None,
) -> bool:
    """Validate a root-absent LSR zone against its live parent/child views."""

    entry_episodes = getattr(belief, "entry_episodes", None)
    context_theses = getattr(belief, "context_theses", None)
    if not isinstance(entry_episodes, Mapping) or not isinstance(
        context_theses,
        Mapping,
    ):
        return False
    episode = entry_episodes.get(candidate_id)
    context = context_theses.get(source_context_thesis_id)
    return bool(
        getattr(hypothesis, "candidate_id", None) == candidate_id
        and getattr(hypothesis, "playbook", None)
        is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and getattr(hypothesis, "episode_id", None) == source_episode_id
        and getattr(hypothesis, "context_thesis_id", None)
        == source_context_thesis_id
        and getattr(hypothesis, "parent_context_thesis_id", None)
        == source_context_thesis_id
        and getattr(hypothesis, "entry_location_id", None)
        == entry_location_id
        and getattr(hypothesis, "entry_path_id", None) == entry_path_id
        and getattr(hypothesis, "market_thesis_action_bound", False) is True
        and getattr(hypothesis, "market_thesis_match_status", None)
        == "exact_root_bound"
        and episode is not None
        and getattr(episode, "candidate_id", None) == candidate_id
        and getattr(episode, "episode_id", None) == source_episode_id
        and getattr(episode, "parent_context_thesis_id", None)
        == source_context_thesis_id
        and getattr(episode, "playbook", None)
        is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and getattr(episode, "direction", None) is hypothesis.direction
        and getattr(episode, "entry_location_id", None) == entry_location_id
        and getattr(episode, "entry_path_id", None) == entry_path_id
        and getattr(episode, "phase", None)
        not in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
        and context is not None
        and getattr(context, "context_thesis_id", None)
        == source_context_thesis_id
        and getattr(context, "lifecycle", None)
        not in {"completed", "invalidated", "censored"}
        and getattr(context, "terminal_at", None) is None
        and source_episode_id
        in tuple(getattr(context, "child_episode_ids", ()))
    )


def _target_r(
    direction: Direction,
    entry: float | None,
    invalidation: float | None,
    target: float | None,
) -> float | None:
    if entry is None or invalidation is None or target is None:
        return None
    risk = direction.sign * (float(entry) - float(invalidation))
    reward = direction.sign * (float(target) - float(entry))
    if risk <= 0.0 or reward <= 0.0:
        return None
    return reward / risk


def _target_r_bucket(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value < 0.5:
        return "lt_0_5R"
    if value < 1.0:
        return "0_5_to_lt_1R"
    if value < 2.0:
        return "1_to_lt_2R"
    if value < 3.0:
        return "2_to_lt_3R"
    return "ge_3R"


def _outcome_class(
    *,
    geometry_complete: bool,
    filled: bool,
    censored: bool,
    expired: bool,
    path_valid: bool,
) -> tuple[bool, str]:
    evaluable = bool(
        geometry_complete and filled and not censored and not expired
    )
    if evaluable:
        return True, "path_valid" if path_valid else "path_failed"
    if not geometry_complete:
        return False, "geometry_unavailable"
    if censored:
        return False, "censored"
    if expired:
        return False, "expired"
    return False, "not_filled"


def _is_expired_resolution(resolution: str) -> bool:
    """Return whether a terminal resolution exhausted its frozen deadline."""

    return resolution in _EXPIRED_RESOLUTIONS


def _record_value(
    record: ShadowCandidateOutcomeRecord | Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    value = (
        record.get(name, default)
        if isinstance(record, Mapping)
        else getattr(record, name, default)
    )
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _record_timestamp(
    record: ShadowCandidateOutcomeRecord | Mapping[str, Any],
    name: str,
) -> pd.Timestamp | None:
    value = _record_value(record, name)
    if value is None:
        return None
    return aware_timestamp(
        pd.Timestamp(value),
        name=f"shadow_outcome.derived.{name}",
    )


def _record_diagnostics(
    record: ShadowCandidateOutcomeRecord | Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw = _record_value(record, "playbook_outcomes", "[]")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("shadow playbook outcomes must be a JSON list")
    diagnostics: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, Mapping):
            raise ValueError("shadow playbook diagnostic must be an object")
        diagnostics.append(dict(item))
    return tuple(diagnostics)


def _derived_outcome(
    record: ShadowCandidateOutcomeRecord | Mapping[str, Any],
) -> tuple[bool, str, bool]:
    geometry_complete = bool(_record_value(record, "geometry_complete", False))
    filled = bool(_record_value(record, "filled", False))
    censored = bool(_record_value(record, "censored", False))
    resolution = str(_record_value(record, "resolution", ""))
    path_valid = _record_value(record, "target_before_invalidation") is True
    outcome_evaluable, outcome_class = _outcome_class(
        geometry_complete=geometry_complete,
        filled=filled,
        censored=censored,
        expired=_is_expired_resolution(resolution),
        path_valid=path_valid,
    )
    return outcome_evaluable, outcome_class, path_valid


def derive_shadow_episode_outcomes(
    records: tuple[ShadowCandidateOutcomeRecord | Mapping[str, Any], ...]
    | list[ShadowCandidateOutcomeRecord | Mapping[str, Any]],
) -> tuple[ShadowEpisodeOutcomeRecord, ...]:
    """Derive one strict terminal row per first-executable episode.

    This is deliberately a finalization pass over the existing candidate
    stream.  It has no checkpoint state and cannot feed outcomes back into
    the replay.
    """

    output: list[ShadowEpisodeOutcomeRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        if (
            _record_value(record, "event_kind")
            != _PLAYBOOK_EXECUTABLE_EVENT_KIND
        ):
            continue
        playbook = _record_value(record, "source_playbook")
        direction_value = _record_value(record, "direction")
        episode_id = _record_value(record, "source_episode_id")
        if not (
            isinstance(playbook, str)
            and playbook
            and isinstance(episode_id, str)
            and episode_id
        ):
            # Completed v7 candidate shards predate the explicit source
            # columns.  Their executable event ID was already frozen as the
            # stable playbook:direction:episode identity.  Accept that one
            # exact format only, then cross-check it against the diagnostic
            # below; do not infer identity from arbitrary legacy fields.
            event_parts = str(_record_value(record, "event_id", "")).split(
                ":",
                2,
            )
            if len(event_parts) == 3 and event_parts[1] == direction_value:
                playbook, _, episode_id = event_parts
        if not all(
            isinstance(value, str) and value
            for value in (playbook, direction_value, episode_id)
        ):
            raise ValueError(
                "playbook executable outcome lacks its frozen episode identity"
            )
        episode_identity = (playbook, direction_value, episode_id)
        if episode_identity in seen:
            raise ValueError(
                "duplicate first-executable outcome for episode "
                + "|".join(episode_identity)
            )
        seen.add(episode_identity)
        diagnostics = tuple(
            item
            for item in _record_diagnostics(record)
            if item.get("playbook") == playbook
            and item.get("episode_id") == episode_id
        )
        if len(diagnostics) != 1:
            raise ValueError(
                "playbook executable outcome must bind exactly one matching "
                "episode diagnostic"
            )
        diagnostic = diagnostics[0]
        if not (
            diagnostic.get("accepted") is True
            and diagnostic.get("exact_root_bound") is True
            and diagnostic.get("market_thesis_match_status")
            == "exact_root_bound"
        ):
            raise ValueError(
                "playbook executable outcome is not exactly root-bound"
            )
        direction = Direction(direction_value)
        planned_entry = _finite(_record_value(record, "entry_reference_price"))
        if planned_entry is None:
            raise ValueError("playbook executable outcome lacks planned entry")
        invalidation = _finite(_record_value(record, "invalidation_price"))
        target = _finite(_record_value(record, "target_price"))
        frozen_target_r = _target_r(
            direction,
            planned_entry,
            invalidation,
            target,
        )
        outcome_evaluable, outcome_class, path_valid = _derived_outcome(record)
        target_first = _record_value(record, "target_before_invalidation")
        invalidation_first = _record_value(
            record,
            "invalidation_before_target",
        )
        same_bar_collision = bool(
            _record_value(record, "same_bar_collision", False)
        )
        resolution = str(_record_value(record, "resolution", ""))
        observed_at = _record_timestamp(record, "observed_at")
        resolved_at = _record_timestamp(record, "resolved_at")
        if observed_at is None or resolved_at is None:
            raise ValueError("episode outcome timestamps are required")
        mfe_r = _finite(_record_value(record, "mfe_R"))
        output.append(
            ShadowEpisodeOutcomeRecord(
                episode_key="|".join(episode_identity),
                candidate_id=str(_record_value(record, "candidate_id")),
                playbook=playbook,
                direction=direction_value,
                episode_id=episode_id,
                setup_id=(
                    _record_value(record, "source_setup_id")
                    or diagnostic.get("setup_id")
                ),
                context_thesis_id=(
                    _record_value(record, "source_context_thesis_id")
                    or diagnostic.get("context_thesis_id")
                ),
                entry_location_id=(
                    _record_value(record, "entry_location_id")
                    or diagnostic.get("entry_location_id")
                ),
                entry_path_id=(
                    _record_value(record, "entry_path_id")
                    or diagnostic.get("entry_path_id")
                ),
                lsr_displacement_id=(
                    _record_value(record, "lsr_displacement_id")
                    or diagnostic.get("lsr_displacement_id")
                ),
                lsr_entry_zone_id=(
                    _record_value(record, "lsr_entry_zone_id")
                    or diagnostic.get("lsr_entry_zone_id")
                ),
                first_executable_at=observed_at,
                observed_at=observed_at,
                resolved_at=resolved_at,
                symbol=str(_record_value(record, "symbol")),
                instrument_id=int(_record_value(record, "instrument_id")),
                source_timeframe=str(
                    _record_value(record, "source_timeframe")
                ),
                market_thesis_id=diagnostic.get("market_thesis_id"),
                bound_market_thesis_id=diagnostic.get(
                    "bound_market_thesis_id"
                ),
                market_thesis_root_id=diagnostic.get(
                    "market_thesis_root_id"
                ),
                market_thesis_mechanism=diagnostic.get(
                    "market_thesis_mechanism"
                ),
                market_thesis_authority_relation=diagnostic.get(
                    "market_thesis_authority_relation"
                ),
                selected_trigger_kind=_record_value(
                    record,
                    "selected_trigger_kind",
                ),
                selected_trigger_id=_record_value(
                    record,
                    "selected_trigger_id",
                ),
                selected_trigger_at=_record_timestamp(
                    record,
                    "selected_trigger_at",
                ),
                available_trigger_kinds=str(
                    _record_value(record, "available_trigger_kinds", "[]")
                ),
                planned_entry_price=planned_entry,
                entry_price=_finite(_record_value(record, "entry_price")),
                entry_at=_record_timestamp(record, "entry_at"),
                invalidation_price=invalidation,
                invalidation_source_id=_record_value(
                    record,
                    "invalidation_source_id",
                ),
                draw_id=_record_value(record, "draw_id"),
                target_price=target,
                deadline_at=_record_timestamp(record, "deadline_at"),
                frozen_target_R=frozen_target_r,
                target_R=frozen_target_r,
                target_R_bucket=_target_r_bucket(frozen_target_r),
                risk_qualified_target_R=bool(
                    frozen_target_r is not None and frozen_target_r >= 1.0
                ),
                planned_target_before_invalidation_deadline=(
                    path_valid if outcome_evaluable else None
                ),
                geometry_complete=bool(
                    _record_value(record, "geometry_complete", False)
                ),
                geometry_incomplete_reason=_record_value(
                    record,
                    "geometry_incomplete_reason",
                ),
                filled=bool(_record_value(record, "filled", False)),
                outcome_evaluable=outcome_evaluable,
                outcome_class=outcome_class,
                resolution=resolution,
                censored=bool(_record_value(record, "censored", False)),
                expired=_is_expired_resolution(resolution),
                target_first=(
                    None if target_first is None else bool(target_first)
                ),
                invalidation_first=(
                    None
                    if invalidation_first is None
                    else bool(invalidation_first)
                ),
                same_bar_collision=same_bar_collision,
                same_bar_stop_first=bool(
                    same_bar_collision and invalidation_first is True
                ),
                mfe_R=mfe_r,
                mae_R=_finite(_record_value(record, "mae_R")),
                hit_0_5R=_record_value(record, "hit_0_5R"),
                hit_1R=_record_value(record, "hit_1R"),
                hit_1_5R=None if mfe_r is None else mfe_r >= 1.5,
                hit_2R=_record_value(record, "hit_2R"),
                time_to_draw_real_bars=_record_value(
                    record,
                    "time_to_draw_real_bars",
                ),
                structural_thesis_invalidated=bool(
                    _record_value(
                        record,
                        "structural_thesis_invalidated",
                        False,
                    )
                ),
                first_changed_evidence_id=_record_value(
                    record,
                    "first_changed_evidence_id",
                ),
                first_changed_evidence_at=_record_timestamp(
                    record,
                    "first_changed_evidence_at",
                ),
                phase=str(diagnostic.get("phase") or "executable"),
                thesis_strength=_finite(diagnostic.get("thesis_strength")),
                sequence_progress=_finite(
                    diagnostic.get("sequence_progress")
                ),
                location_quality=_finite(diagnostic.get("location_quality")),
                entry_readiness=_finite(diagnostic.get("entry_readiness")),
                delivery_quality=_finite(diagnostic.get("delivery_quality")),
                uncertainty=_finite(diagnostic.get("uncertainty")),
                decision_action=str(
                    _record_value(record, "decision_action", "")
                ),
                risk_action=str(_record_value(record, "risk_action", "")),
            )
        )
    return tuple(output)


def derive_shadow_mechanism_challenges(
    records: tuple[ShadowCandidateOutcomeRecord | Mapping[str, Any], ...]
    | list[ShadowCandidateOutcomeRecord | Mapping[str, Any]],
    *,
    active_playbooks: tuple[Playbook, ...] = (
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    ),
) -> tuple[ShadowMechanismChallengeRecord, ...]:
    """Challenge each neutral Eye candidate against active mechanisms only."""

    allowed = frozenset(item.value for item in active_playbooks)
    output: list[ShadowMechanismChallengeRecord] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        event_kind = str(_record_value(record, "event_kind", ""))
        if event_kind == _PLAYBOOK_EXECUTABLE_EVENT_KIND:
            continue
        if event_kind not in _NEUTRAL_CANDIDATE_EVENT_KINDS:
            raise ValueError(
                "unknown neutral shadow candidate event kind: " + event_kind
            )
        direction_value = str(_record_value(record, "direction"))
        direction = Direction(direction_value)
        candidate_id = str(_record_value(record, "candidate_id"))
        outcome_evaluable, outcome_class, path_valid = _derived_outcome(record)
        actual_entry = _finite(_record_value(record, "entry_price"))
        invalidation = _finite(_record_value(record, "invalidation_price"))
        target = _finite(_record_value(record, "target_price"))
        target_r = _target_r(direction, actual_entry, invalidation, target)
        mfe_r = _finite(_record_value(record, "mfe_R"))
        observed_at = _record_timestamp(record, "observed_at")
        resolved_at = _record_timestamp(record, "resolved_at")
        if observed_at is None or resolved_at is None:
            raise ValueError("mechanism challenge timestamps are required")
        for diagnostic in _record_diagnostics(record):
            playbook = diagnostic.get("playbook")
            if (
                playbook not in allowed
                or diagnostic.get("runtime_enabled") is not True
            ):
                continue
            identity = (candidate_id, str(playbook))
            if identity in seen:
                raise ValueError(
                    "duplicate mechanism challenge for candidate/playbook"
                )
            seen.add(identity)
            challenge_id = "challenge:" + hashlib.sha256(
                "|".join(identity).encode("utf-8")
            ).hexdigest()[:24]
            failed_gates = diagnostic.get("failed_gates") or []
            output.append(
                ShadowMechanismChallengeRecord(
                    challenge_id=challenge_id,
                    candidate_id=candidate_id,
                    event_kind=event_kind,
                    event_id=str(_record_value(record, "event_id")),
                    observed_at=observed_at,
                    resolved_at=resolved_at,
                    symbol=str(_record_value(record, "symbol")),
                    instrument_id=int(_record_value(record, "instrument_id")),
                    direction=direction_value,
                    source_timeframe=str(
                        _record_value(record, "source_timeframe")
                    ),
                    playbook=str(playbook),
                    episode_id=diagnostic.get("episode_id"),
                    setup_id=diagnostic.get("setup_id"),
                    context_thesis_id=diagnostic.get(
                        "context_thesis_id"
                    ),
                    entry_location_id=(
                        diagnostic.get("entry_location_id")
                        or _record_value(record, "entry_location_id")
                    ),
                    entry_path_id=(
                        diagnostic.get("entry_path_id")
                        or _record_value(record, "entry_path_id")
                    ),
                    lsr_displacement_id=(
                        diagnostic.get("lsr_displacement_id")
                        or _record_value(record, "lsr_displacement_id")
                    ),
                    lsr_entry_zone_id=(
                        diagnostic.get("lsr_entry_zone_id")
                        or _record_value(record, "lsr_entry_zone_id")
                    ),
                    entry_episode_binding_status=(
                        diagnostic.get("entry_episode_binding_status")
                        or _record_value(
                            record,
                            "entry_episode_binding_status",
                        )
                    ),
                    entry_episode_terminal_at=_record_timestamp(
                        record,
                        "entry_episode_terminal_at",
                    ),
                    entry_episode_terminal_reason=_record_value(
                        record,
                        "entry_episode_terminal_reason",
                    ),
                    accepted_at_candidate_clock=bool(
                        diagnostic.get("accepted", False)
                    ),
                    first_failed_gate=diagnostic.get("first_failed_gate"),
                    failed_gates=_json(failed_gates),
                    event_order_signature=_json(
                        diagnostic.get("event_order_signature")
                        or [event_kind]
                    ),
                    plan_feasibility_valid=bool(
                        diagnostic.get("plan_feasibility_valid", False)
                    ),
                    plan_feasibility_failure_reason=diagnostic.get(
                        "plan_feasibility_failure_reason"
                    ),
                    graph_connected=bool(
                        diagnostic.get("graph_connected", False)
                    ),
                    exact_root_bound=bool(
                        diagnostic.get("exact_root_bound", False)
                    ),
                    market_thesis_match_status=str(
                        diagnostic.get("market_thesis_match_status")
                        or "no_open_thesis"
                    ),
                    market_thesis_id=diagnostic.get("market_thesis_id"),
                    market_thesis_root_id=diagnostic.get(
                        "market_thesis_root_id"
                    ),
                    market_thesis_mechanism=diagnostic.get(
                        "market_thesis_mechanism"
                    ),
                    market_thesis_authority_relation=diagnostic.get(
                        "market_thesis_authority_relation"
                    ),
                    liquidity_route_id=diagnostic.get("liquidity_route_id"),
                    context_draw_id=diagnostic.get("context_draw_id"),
                    intermediate_liquidity_ids=_json(
                        diagnostic.get("intermediate_liquidity_ids") or []
                    ),
                    primary_deliverable_target_id=diagnostic.get(
                        "primary_deliverable_target_id"
                    ),
                    terminal_draw_id=diagnostic.get("terminal_draw_id"),
                    authority_barrier_id=diagnostic.get(
                        "authority_barrier_id"
                    ),
                    authority_barrier_price=_finite(
                        diagnostic.get("authority_barrier_price")
                    ),
                    entry_price=actual_entry,
                    invalidation_price=invalidation,
                    target_price=target,
                    target_R=target_r,
                    target_R_bucket=_target_r_bucket(target_r),
                    risk_qualified_target_R=bool(
                        target_r is not None and target_r >= 1.0
                    ),
                    planned_target_before_invalidation_deadline=(
                        path_valid if outcome_evaluable else None
                    ),
                    geometry_complete=bool(
                        _record_value(record, "geometry_complete", False)
                    ),
                    geometry_incomplete_reason=_record_value(
                        record,
                        "geometry_incomplete_reason",
                    ),
                    filled=bool(_record_value(record, "filled", False)),
                    outcome_evaluable=outcome_evaluable,
                    outcome_class=outcome_class,
                    resolution=str(_record_value(record, "resolution", "")),
                    censored=bool(_record_value(record, "censored", False)),
                    same_bar_collision=bool(
                        _record_value(record, "same_bar_collision", False)
                    ),
                    mfe_R=mfe_r,
                    mae_R=_finite(_record_value(record, "mae_R")),
                    hit_0_5R=_record_value(record, "hit_0_5R"),
                    hit_1R=_record_value(record, "hit_1R"),
                    hit_1_5R=None if mfe_r is None else mfe_r >= 1.5,
                    hit_2R=_record_value(record, "hit_2R"),
                    time_to_draw_real_bars=_record_value(
                        record,
                        "time_to_draw_real_bars",
                    ),
                )
            )
    return tuple(output)


_MOTIF_PLAYBOOKS = SHADOW_SEQUENCE_PLAYBOOKS


def _challenge_rejected_gates(item: ShadowMechanismChallengeRecord) -> tuple[str, ...]:
    raw = json.loads(item.failed_gates)
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ValueError("shadow challenge failed_gates must be a JSON list")
    return tuple(
        dict.fromkeys(
            value
            for value in (
                *raw,
                item.plan_feasibility_failure_reason,
                item.first_failed_gate,
            )
            if isinstance(value, str) and value
        )
    )


def _challenge_liquidity_route(
    item: ShadowMechanismChallengeRecord,
) -> dict[str, Any]:
    waypoints = json.loads(item.intermediate_liquidity_ids)
    if not isinstance(waypoints, list) or any(
        not isinstance(value, str) or not value for value in waypoints
    ):
        raise ValueError(
            "shadow challenge intermediate_liquidity_ids must be a JSON list"
        )
    return {
        "liquidity_route_id": item.liquidity_route_id,
        "context_draw_id": item.context_draw_id,
        "intermediate_liquidity_ids": waypoints,
        "primary_deliverable_target_id": (
            item.primary_deliverable_target_id
        ),
        "terminal_draw_id": item.terminal_draw_id,
        "authority_barrier_id": item.authority_barrier_id,
        "authority_barrier_price": item.authority_barrier_price,
    }


def _consistent_challenge_value(
    values: tuple[ShadowMechanismChallengeRecord, ...],
    field_name: str,
) -> Any:
    distinct = {
        json.dumps(getattr(item, field_name), sort_keys=True, default=str)
        for item in values
    }
    if len(distinct) != 1:
        raise ValueError(
            "root episode challenges disagree on frozen " + field_name
        )
    return getattr(values[0], field_name)


def _root_episode_key(root_key: tuple[str, int, str, str]) -> str:
    return "root-episode:" + hashlib.sha256(
        "|".join(str(value) for value in root_key).encode("utf-8")
    ).hexdigest()[:24]


def _sequence_scope(
    item: ShadowMechanismChallengeRecord,
) -> tuple[str, str]:
    if item.episode_id:
        return "entry_episode", item.episode_id
    if item.setup_id:
        return "setup_context", item.setup_id
    thesis_id = item.market_thesis_id
    if not thesis_id:
        raise ValueError(
            "context sequence scope lacks its epoch-aware thesis identity"
        )
    return "context_thesis", thesis_id


def _challenge_event_order(
    item: ShadowMechanismChallengeRecord,
) -> tuple[str, ...]:
    try:
        values = json.loads(item.event_order_signature)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "shadow challenge event_order_signature must be JSON"
        ) from exc
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise ValueError(
            "shadow challenge event_order_signature must be a non-empty list"
        )
    return tuple(values)


def derive_shadow_root_sequence_records(
    challenges: tuple[ShadowMechanismChallengeRecord, ...]
    | list[ShadowMechanismChallengeRecord],
) -> tuple[ShadowRootSequenceRecord, ...]:
    """Aggregate candidate-clock revisions without selecting on outcomes.

    Units are isolated by canonical market root, playbook, and the strongest
    explicit local identity available: EntryEpisode, setup Context, or the
    Context Thesis itself.  The descriptive representative is the first
    revision at the maximum observed causal sequence depth.  Price outcomes,
    target delivery, MFE/MAE, and censor labels are intentionally unread.
    """

    grouped: dict[
        tuple[str, int, str, str, str, str, str],
        list[ShadowMechanismChallengeRecord],
    ] = {}
    for item in challenges:
        if item.playbook not in SHADOW_SEQUENCE_PLAYBOOKS:
            continue
        root_id = item.market_thesis_root_id
        if not isinstance(root_id, str) or not root_id:
            continue
        scope_kind, scope_id = _sequence_scope(item)
        key = (
            item.symbol,
            int(item.instrument_id),
            item.direction,
            root_id,
            item.playbook,
            scope_kind,
            scope_id,
        )
        grouped.setdefault(key, []).append(item)

    output: list[ShadowRootSequenceRecord] = []
    for key, raw_values in sorted(grouped.items()):
        values = tuple(
            sorted(
                raw_values,
                key=lambda item: (
                    item.observed_at,
                    _NEUTRAL_CANDIDATE_EVENT_ORDER.get(
                        item.event_kind,
                        len(_NEUTRAL_CANDIDATE_EVENT_ORDER),
                    ),
                    item.candidate_id,
                ),
            )
        )
        if len({item.candidate_id for item in values}) != len(values):
            raise ValueError(
                "duplicate candidate revision in shadow sequence scope"
            )
        root_key = key[:4]
        playbook = key[4]
        scope_kind = key[5]
        scope_id = key[6]
        setup_ids = {
            item.setup_id for item in values if item.setup_id is not None
        }
        if scope_kind == "entry_episode" and len(setup_ids) > 1:
            raise ValueError(
                "one shadow EntryEpisode carries multiple setup identities"
            )
        mechanisms = {
            item.market_thesis_mechanism
            for item in values
            if item.market_thesis_mechanism
        }
        if len(mechanisms) > 1:
            raise ValueError(
                "one shadow market root carries multiple mechanisms"
            )

        signatures_by_candidate = {
            item.candidate_id: _challenge_event_order(item)
            for item in values
        }
        selected = min(
            values,
            key=lambda item: (
                -len(signatures_by_candidate[item.candidate_id]),
                item.observed_at,
                _NEUTRAL_CANDIDATE_EVENT_ORDER.get(
                    item.event_kind,
                    len(_NEUTRAL_CANDIDATE_EVENT_ORDER),
                ),
                item.candidate_id,
            ),
        )
        selected_signature = signatures_by_candidate[selected.candidate_id]
        signature_variants = tuple(
            sorted(
                set(signatures_by_candidate.values()),
                key=lambda signature: (len(signature), signature),
            )
        )

        def unique_ngrams(size: int) -> list[tuple[str, ...]]:
            seen: set[tuple[str, ...]] = set()
            ordered: list[tuple[str, ...]] = []
            for signature in signature_variants:
                for index in range(len(signature) - size + 1):
                    value = signature[index : index + size]
                    if value not in seen:
                        seen.add(value)
                        ordered.append(value)
            return ordered

        bigrams = unique_ngrams(2)
        trigrams = unique_ngrams(3)

        primitive_counts = Counter(item.event_kind for item in values)
        primitive_first: dict[str, str] = {}
        candidate_events: list[str] = []
        step_first: dict[str, str] = {}
        for item in values:
            clock = item.observed_at.isoformat()
            primitive_first.setdefault(item.event_kind, clock)
            if item.event_kind not in candidate_events:
                candidate_events.append(item.event_kind)
            for step in signatures_by_candidate[item.candidate_id]:
                step_first.setdefault(step, clock)

        accepted = tuple(
            item for item in values if item.accepted_at_candidate_clock
        )
        first_failed = next(
            (
                item
                for item in values
                if isinstance(item.first_failed_gate, str)
                and item.first_failed_gate
            ),
            None,
        )
        root_episode_key = _root_episode_key(root_key)
        unit_identity = (*root_key, playbook, scope_kind, scope_id)
        sequence_unit_key = "root-sequence:" + hashlib.sha256(
            "|".join(str(value) for value in unit_identity).encode("utf-8")
        ).hexdigest()[:24]
        output.append(
            ShadowRootSequenceRecord(
                sequence_unit_key=sequence_unit_key,
                root_episode_key=root_episode_key,
                root_id=root_key[3],
                symbol=root_key[0],
                instrument_id=root_key[1],
                direction=root_key[2],
                playbook=playbook,
                scope_kind=scope_kind,
                scope_id=scope_id,
                episode_id=(
                    scope_id if scope_kind == "entry_episode" else None
                ),
                setup_id=(
                    next(iter(setup_ids))
                    if setup_ids
                    else None
                ),
                market_mechanism=(
                    next(iter(mechanisms)) if mechanisms else "unexplained"
                ),
                source_timeframe=selected.source_timeframe,
                authority_relation=(
                    selected.market_thesis_authority_relation or "unknown"
                ),
                selected_candidate_id=selected.candidate_id,
                selected_event_kind=selected.event_kind,
                selected_event_id=selected.event_id,
                first_observed_at=values[0].observed_at,
                selected_at=selected.observed_at,
                last_observed_at=values[-1].observed_at,
                revision_count=len(values),
                primitive_event_counts=_json(
                    dict(sorted(primitive_counts.items()))
                ),
                primitive_first_observed_at=_json(primitive_first),
                candidate_event_order_signature=_json(candidate_events),
                event_order_signature=_json(selected_signature),
                observed_sequence_signatures=_json(signature_variants),
                step_first_observed_at=_json(step_first),
                event_order_length=len(selected_signature),
                event_bigrams=_json(bigrams),
                event_trigrams=_json(trigrams),
                signature_variant_count=len(signature_variants),
                complete_sequence_observed=(
                    selected_signature
                    == _COMPLETE_SEQUENCE_STEPS[playbook]
                ),
                accepted_revision_count=len(accepted),
                ever_accepted=bool(accepted),
                first_accepted_at=(
                    accepted[0].observed_at if accepted else None
                ),
                first_failed_gate=(
                    None
                    if first_failed is None
                    else first_failed.first_failed_gate
                ),
                first_failed_gate_at=(
                    None if first_failed is None else first_failed.observed_at
                ),
                selected_revision_accepted=(
                    selected.accepted_at_candidate_clock
                ),
                selected_first_failed_gate=selected.first_failed_gate,
                selected_failed_gates=_json(
                    _challenge_rejected_gates(selected)
                ),
                selected_plan_feasibility_valid=(
                    selected.plan_feasibility_valid
                ),
                selected_graph_connected=selected.graph_connected,
                selected_exact_root_bound=selected.exact_root_bound,
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                item.first_observed_at,
                item.sequence_unit_key,
            ),
        )
    )


def derive_shadow_root_episode_records(
    challenges: tuple[ShadowMechanismChallengeRecord, ...]
    | list[ShadowMechanismChallengeRecord],
) -> tuple[ShadowRootEpisodeRecord, ...]:
    """Collapse thesis revisions into independent, outcome-blind root units.

    Candidate selection uses only candidate-clock geometry and clocks: the
    first geometry-complete revision is frozen, or the first revision if the
    root never has complete geometry.  Later path validity is never consulted
    when choosing the representative revision.
    """

    roots: dict[
        tuple[str, int, str, str],
        dict[str, dict[str, ShadowMechanismChallengeRecord]],
    ] = {}
    for item in challenges:
        if item.playbook not in _MOTIF_PLAYBOOKS:
            continue
        root_id = item.market_thesis_root_id
        if not isinstance(root_id, str) or not root_id:
            # Root episodes are independent OpenMarketThesis units.  A raw
            # Eye candidate without a canonical thesis root remains a raw
            # challenge and must not be promoted into a pseudo root.
            continue
        root_identity = root_id
        root_key = (
            item.symbol,
            int(item.instrument_id),
            item.direction,
            root_identity,
        )
        by_playbook = roots.setdefault(root_key, {}).setdefault(
            item.candidate_id,
            {},
        )
        if item.playbook in by_playbook:
            raise ValueError(
                "duplicate shadow root revision/playbook challenge"
            )
        by_playbook[item.playbook] = item

    output: list[ShadowRootEpisodeRecord] = []
    for root_key, revisions in sorted(roots.items()):
        ranked: list[
            tuple[
                int,
                pd.Timestamp,
                str,
                tuple[ShadowMechanismChallengeRecord, ...],
            ]
        ] = []
        for candidate_id, by_playbook in revisions.items():
            values = tuple(
                by_playbook[name]
                for name in _MOTIF_PLAYBOOKS
                if name in by_playbook
            )
            observed_at = min(item.observed_at for item in values)
            geometry_complete = bool(
                _consistent_challenge_value(values, "geometry_complete")
            )
            ranked.append(
                (
                    int(not geometry_complete),
                    observed_at,
                    candidate_id,
                    values,
                )
            )
        _, observed_at, candidate_id, values = min(ranked)
        by_playbook = {item.playbook: item for item in values}
        representative = values[0]
        for field_name in (
            "source_timeframe",
            "market_thesis_root_id",
            "market_thesis_mechanism",
            "market_thesis_authority_relation",
            "target_R_bucket",
            "risk_qualified_target_R",
            "geometry_complete",
            "filled",
            "outcome_evaluable",
            "outcome_class",
            "resolution",
            "censored",
        ):
            _consistent_challenge_value(values, field_name)

        event_order = {
            name: json.loads(by_playbook[name].event_order_signature)
            for name in _MOTIF_PLAYBOOKS
            if name in by_playbook
        }
        rejected_gates = {
            name: list(_challenge_rejected_gates(by_playbook[name]))
            for name in _MOTIF_PLAYBOOKS
            if name in by_playbook
        }
        playbook_liquidity_routes = {
            name: _challenge_liquidity_route(by_playbook[name])
            for name in _MOTIF_PLAYBOOKS
            if name in by_playbook
        }
        dfp = by_playbook.get(_MOTIF_PLAYBOOKS[0])
        lsr = by_playbook.get(_MOTIF_PLAYBOOKS[1])
        dfp_rejected = bool(
            dfp is not None and not dfp.accepted_at_candidate_clock
        )
        lsr_rejected = bool(
            lsr is not None and not lsr.accepted_at_candidate_clock
        )
        expired = _is_expired_resolution(representative.resolution)
        root_id = representative.market_thesis_root_id
        path_valid = bool(
            representative.outcome_evaluable
            and representative.outcome_class == "path_valid"
            and representative.planned_target_before_invalidation_deadline
            is True
        )
        eligible_episode_evidence = bool(
            isinstance(root_id, str)
            and root_id
            and dfp_rejected
            and lsr_rejected
            and representative.geometry_complete
            and representative.filled
            and not representative.censored
            and not expired
            and representative.outcome_evaluable
            and path_valid
            and representative.risk_qualified_target_R
        )
        root_episode_key = _root_episode_key(root_key)
        session_date_ny = observed_at.tz_convert(
            "America/New_York"
        ).date().isoformat()
        output.append(
            ShadowRootEpisodeRecord(
                root_episode_key=root_episode_key,
                root_id=root_id,
                candidate_id=candidate_id,
                observed_at=observed_at,
                session_date_ny=session_date_ny,
                symbol=representative.symbol,
                instrument_id=representative.instrument_id,
                direction=representative.direction,
                market_mechanism=(
                    representative.market_thesis_mechanism
                    or "unexplained"
                ),
                source_timeframe=representative.source_timeframe,
                authority_relation=(
                    representative.market_thesis_authority_relation
                    or "unknown"
                ),
                event_order_signature=_json(event_order),
                rejected_gates=_json(rejected_gates),
                playbook_liquidity_routes=_json(playbook_liquidity_routes),
                target_R_bucket=representative.target_R_bucket,
                dfp_rejected=dfp_rejected,
                lsr_rejected=lsr_rejected,
                geometry_complete=representative.geometry_complete,
                filled=representative.filled,
                censored=representative.censored,
                expired=expired,
                outcome_evaluable=representative.outcome_evaluable,
                path_valid=path_valid,
                risk_qualified_target_R=(
                    representative.risk_qualified_target_R
                ),
                eligible_episode_evidence=eligible_episode_evidence,
                action_authority=False,
            )
        )
    return tuple(
        sorted(output, key=lambda item: (item.observed_at, item.root_episode_key))
    )


def aggregate_shadow_mechanism_motifs(
    episodes: tuple[ShadowRootEpisodeRecord, ...]
    | list[ShadowRootEpisodeRecord],
) -> tuple[ShadowMechanismMotifRecord, ...]:
    """Aggregate repeatable motifs without granting action authority."""

    grouped: dict[tuple[str, ...], list[ShadowRootEpisodeRecord]] = {}
    for item in episodes:
        key = (
            item.market_mechanism,
            item.source_timeframe,
            item.authority_relation,
            item.event_order_signature,
            item.rejected_gates,
            item.target_R_bucket,
        )
        grouped.setdefault(key, []).append(item)

    output: list[ShadowMechanismMotifRecord] = []
    for key, values in sorted(grouped.items()):
        eligible = tuple(
            item for item in values if item.eligible_episode_evidence
        )
        dates = {item.session_date_ny for item in values}
        eligible_dates = {item.session_date_ny for item in eligible}
        motif_id = "shadow-motif:" + hashlib.sha256(
            _json(key).encode("utf-8")
        ).hexdigest()[:24]
        root_keys = sorted({item.root_episode_key for item in values})
        if len(root_keys) != len(values):
            raise ValueError("shadow motif contains duplicate root episodes")
        root_sample = root_keys[:SHADOW_MOTIF_ROOT_SAMPLE_LIMIT]
        output.append(
            ShadowMechanismMotifRecord(
                motif_id=motif_id,
                market_mechanism=key[0],
                source_timeframe=key[1],
                authority_relation=key[2],
                event_order_signature=key[3],
                rejected_gates=key[4],
                target_R_bucket=key[5],
                episode_count=len(values),
                eligible_episode_count=len(eligible),
                distinct_dates=len(dates),
                eligible_distinct_dates=len(eligible_dates),
                sample_root_episode_keys=_json(root_sample),
                sample_root_episode_count=len(root_sample),
                root_episode_keys_truncated=(
                    len(root_keys) > len(root_sample)
                ),
                eligible_for_preregistration_review=bool(
                    len(eligible) >= 2 and len(eligible_dates) >= 2
                ),
                action_authority=False,
            )
        )
    return tuple(output)


class ShadowCandidateOutcomeRecorder:
    """Causal, outcome-blind recorder for neutral Eye event candidates."""

    def __init__(
        self,
        *,
        enabled_playbooks: tuple[Playbook, ...] = (
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
        ),
    ) -> None:
        self._recorder_schema_version = RECORDER_SCHEMA_VERSION
        self._enabled_playbooks = frozenset(
            Playbook(item) for item in enabled_playbooks
        )
        self._open: dict[str, _OpenCandidate] = {}
        self._lsr_zone_custody: dict[
            tuple[str, int, str],
            _LSRZoneCustody,
        ] = {}
        self._rows: list[ShadowCandidateOutcomeRecord] = []
        self._seen_events: set[tuple[str, str, str]] = set()
        self._last_bar_start: pd.Timestamp | None = None
        self._last_observation_asof: pd.Timestamp | None = None
        self._pending_bar: Bar | None = None
        self._candidate_events: Counter[str] = Counter()
        self._entry_episode_bindings: Counter[str] = Counter()
        self._resolutions: Counter[str] = Counter()
        self._quadrants: Counter[tuple[str, str]] = Counter()
        self._fill_counts: Counter[str] = Counter()
        self._case_ids: list[str] = []
        self._case_strata: dict[str, list[str]] = {}
        self._typed_delta_missing = 0
        self._candidate_count = 0

    @property
    def rows(self) -> tuple[ShadowCandidateOutcomeRecord, ...]:
        return tuple(self._rows)

    @property
    def open_candidates(self) -> tuple[_OpenCandidate, ...]:
        return tuple(self._open[key] for key in sorted(self._open))

    @property
    def summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": RECORDER_SCHEMA_VERSION,
            "counting_basis": {
                "candidate": (
                    "unique neutral Eye transition, Context-linked LSR entry-"
                    "zone formation, substantive open-thesis revision, or "
                    "first executable playbook episode"
                ),
                "quadrant": (
                    "source playbook of one unique first-executable episode; "
                    "neutral Eye candidates are reported only by the derived "
                    "mechanism challenge artifact"
                ),
            },
            "candidate_count": self._candidate_count,
            "open_candidate_count": len(self._open),
            "frozen_lsr_zone_custody_count": len(self._lsr_zone_custody),
            "terminal_lsr_zone_custody_count": sum(
                item.owner_terminal_at is not None
                for item in self._lsr_zone_custody.values()
            ),
            "emitted_candidate_count": sum(self._resolutions.values()),
            "typed_delta_missing_observations": self._typed_delta_missing,
            "candidate_events": dict(sorted(self._candidate_events.items())),
            "entry_episode_bindings": dict(
                sorted(self._entry_episode_bindings.items())
            ),
            "resolutions": dict(sorted(self._resolutions.items())),
            "fills": dict(sorted(self._fill_counts.items())),
            "four_quadrants": {
                playbook: {
                    quadrant: count
                    for (item_playbook, quadrant), count in sorted(
                        self._quadrants.items()
                    )
                    if item_playbook == playbook
                }
                for playbook in sorted(
                    {item[0] for item in self._quadrants}
                )
            },
            "case_ids": tuple(self._case_ids),
            "case_strata": {
                candidate_id: tuple(values)
                for candidate_id, values in sorted(
                    self._case_strata.items()
                )
            },
        }

    @property
    def recorder_schema_version(self) -> int:
        return int(self._recorder_schema_version)

    def drain_rows(self) -> tuple[ShadowCandidateOutcomeRecord, ...]:
        rows = tuple(self._rows)
        self._rows.clear()
        return rows

    def _refresh_lsr_zone_terminal_tombstones(
        self,
        snapshot: EngineSnapshot,
    ) -> None:
        """Latch the first typed terminal fact for every exact zone owner.

        The live terminal projection may be retained for only one completed
        bar.  Once observed, its local reason and clock are immutable custody
        facts and survive later Context/EntryEpisode compaction.
        """

        belief = snapshot.belief
        entry_episodes = getattr(belief, "entry_episodes", None)
        context_theses = getattr(belief, "context_theses", None)
        if not isinstance(entry_episodes, Mapping) or not isinstance(
            context_theses,
            Mapping,
        ):
            return
        for key, custody in tuple(self._lsr_zone_custody.items()):
            if (
                custody.binding_status
                not in _EXACT_ENTRY_EPISODE_BINDINGS
                or custody.owner_terminal_at is not None
            ):
                continue
            terminal_at = None
            terminal_reason = None
            episode = entry_episodes.get(
                custody.source_action_candidate_id
            )
            if (
                episode is not None
                and getattr(episode, "episode_id", None)
                == custody.source_episode_id
                and getattr(episode, "parent_context_thesis_id", None)
                == custody.source_context_thesis_id
                and (
                    getattr(episode, "phase", None)
                    in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
                    or getattr(episode, "terminal_at", None) is not None
                )
            ):
                terminal_at = getattr(episode, "terminal_at", None)
                terminal_reason = getattr(episode, "terminal_reason", None)
            if terminal_at is None:
                context = context_theses.get(
                    custody.source_context_thesis_id
                )
                if (
                    context is not None
                    and getattr(context, "context_thesis_id", None)
                    == custody.source_context_thesis_id
                    and (
                        getattr(context, "lifecycle", None)
                        in {"completed", "invalidated", "censored"}
                        or getattr(context, "terminal_at", None) is not None
                    )
                ):
                    terminal_at = getattr(context, "terminal_at", None)
                    terminal_reason = getattr(
                        context,
                        "terminal_reason",
                        None,
                    )
            if terminal_at is None and terminal_reason is None:
                continue
            if (
                terminal_at is None
                or not isinstance(terminal_reason, str)
                or not terminal_reason
            ):
                raise ValueError(
                    "typed LSR owner terminal projection is incomplete"
                )
            terminal_at = aware_timestamp(
                terminal_at,
                name="shadow_outcome.lsr_owner_terminal_at",
            )
            if terminal_at > snapshot.observation.asof:
                raise ValueError(
                    "LSR owner terminal tombstone cannot be future-dated"
                )
            self._lsr_zone_custody[key] = replace(
                custody,
                owner_terminal_at=terminal_at,
                owner_terminal_reason=terminal_reason,
            )

    def on_bar(self, bar: Bar) -> None:
        """Register the bar clock; all terminal handling waits for Engine."""

        if self._last_bar_start is not None and bar.start <= self._last_bar_start:
            raise ValueError("shadow outcome bars must be strictly increasing")
        if self._pending_bar is not None:
            raise RuntimeError(
                "previous shadow bar has not received its Engine snapshot"
            )
        self._last_bar_start = bar.start
        self._pending_bar = bar

    def prime(self, snapshot: EngineSnapshot, *, source_bar: Bar) -> None:
        """Consume warmup identities without opening outcome candidates."""

        observation = self._validate_snapshot(snapshot, source_bar, allow_equal=False)
        self._consume_pending_bar(source_bar)
        if (
            source_bar.synthetic_no_trade
            or set(observation.anomalies) & _BOUNDARY_ANOMALIES
        ):
            return
        self._refresh_lsr_zone_terminal_tombstones(snapshot)
        for spec in self._open_thesis_candidate_specs(snapshot):
            self._seen_events.add(self._event_key(spec))
        for spec in self._playbook_candidate_specs(snapshot):
            self._seen_events.add(self._event_key(spec))
        if not bool(observation.typed_transition_delta_available):
            return
        for spec in self._candidate_specs(snapshot):
            self._seen_events.add(self._event_key(spec))

    def observe(self, snapshot: EngineSnapshot, *, source_bar: Bar) -> None:
        """Freeze candidates from the current typed delta after Engine output."""

        observation = self._validate_snapshot(snapshot, source_bar, allow_equal=False)
        self._consume_pending_bar(source_bar)
        if set(observation.anomalies) & _BOUNDARY_ANOMALIES:
            self._censor_all(observation.asof, "observation_boundary")
            return
        if source_bar.data_gap_before_minutes:
            self._censor_all(observation.asof, "data_gap_boundary")
            return
        self._refresh_lsr_zone_terminal_tombstones(snapshot)
        self._process_existing_candidates(snapshot, source_bar)
        if source_bar.synthetic_no_trade:
            return
        for spec in self._open_thesis_candidate_specs(snapshot):
            key = self._event_key(spec)
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            self._register(snapshot, source_bar, spec)
        for spec in self._playbook_candidate_specs(snapshot):
            key = self._event_key(spec)
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            self._register(snapshot, source_bar, spec)
        if not bool(observation.typed_transition_delta_available):
            self._typed_delta_missing += 1
            return
        for spec in self._candidate_specs(snapshot):
            key = self._event_key(spec)
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            self._register(snapshot, source_bar, spec)

    @classmethod
    def _open_thesis_candidate_specs(
        cls,
        snapshot: EngineSnapshot,
    ) -> tuple[_CandidateSpec, ...]:
        """Freeze one shadow candidate per substantive open-thesis revision.

        A thesis revision is descriptive only.  It never acquires action
        authority here: the recorder merely freezes geometry already owned by
        the exact root-specific Brain candidate, or by identities carried by
        that thesis.  Missing geometry is emitted explicitly rather than
        filled from unrelated current-market levels.
        """

        context = getattr(snapshot.belief, "global_context", None)
        theses = () if context is None else context.open_market_theses
        action_candidates = _action_candidate_items(snapshot.belief)
        observation = brain_observation_view(snapshot.observation)
        specs: list[_CandidateSpec] = []
        for thesis in sorted(theses, key=lambda item: item.thesis_id):
            if getattr(thesis, "lifecycle", "forming") == "invalidated":
                # The terminal revision closes factual state; it is not a new
                # forward-looking shadow opportunity.
                continue
            raw_direction = getattr(thesis, "direction", None)
            if raw_direction is None:
                continue
            direction = Direction(raw_direction)
            root_id = getattr(thesis, "root_id", None)
            thesis_id = getattr(thesis, "thesis_id", None)
            if not isinstance(root_id, str) or not root_id:
                continue
            if not isinstance(thesis_id, str) or not thesis_id:
                continue

            updated_raw = getattr(thesis, "updated_at", None)
            revision = getattr(thesis, "evidence_revision_id", None)
            evidence_state = getattr(thesis, "evidence_state", None)
            if not revision and evidence_state is not None:
                revision = getattr(evidence_state, "revision_id", None)
            # Old mocks and archived objects may not expose the new evidence
            # contract.  They are ignored instead of becoming one candidate on
            # every minute through a clock-based fallback.
            if updated_raw is None or not isinstance(revision, str) or not revision:
                continue
            updated_at = aware_timestamp(
                updated_raw,
                name="shadow_outcome.open_thesis.updated_at",
            )
            if updated_at > observation.asof:
                raise ValueError("open thesis revision cannot be future-dated")
            if updated_at != observation.asof:
                # Persisted theses are not minute snapshots.  Only the clock
                # at which their factual evidence revision changed registers
                # a new shadow candidate.
                continue

            root_candidates = tuple(
                (identity, hypothesis)
                for identity, hypothesis in action_candidates
                if getattr(hypothesis, "direction", None) is direction
                and (
                    getattr(hypothesis, "required_root_id", None) == root_id
                    or getattr(hypothesis, "market_thesis_root_id", None)
                    == root_id
                )
            )
            selected_identity, selected = cls._select_root_candidate(
                root_candidates
            )
            plan = None if selected is None else getattr(selected, "plan", None)
            plan_complete = bool(
                plan is not None
                and all(
                    hasattr(plan, name)
                    for name in (
                        "planned_entry",
                        "invalidation",
                        "targets",
                        "deadline",
                    )
                )
                and bool(getattr(plan, "targets", ()))
            )

            entry_location_ids = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        getattr(selected, "entry_location_id", None),
                        *getattr(thesis, "entry_location_ids", ()),
                    )
                    if isinstance(value, str) and value
                )
            )
            location = next(
                (
                    item
                    for identity in entry_location_ids
                    for item in observation.entry_locations
                    if getattr(item, "location_id", None) == identity
                ),
                None,
            )

            invalidation = None
            invalidation_id = None
            target = None
            draw_id = None
            planned_entry = None
            deadline_at = None
            zone_lower = None
            zone_upper = None
            if plan_complete:
                target_state = plan.targets[0]
                invalidation = _finite(plan.invalidation.price)
                invalidation_id = getattr(
                    plan.invalidation,
                    "source_level_id",
                    None,
                )
                target = _finite(target_state.price)
                draw_id = (
                    getattr(plan, "selected_draw_id", None)
                    or getattr(target_state, "level_id", None)
                )
                planned_entry = _finite(plan.planned_entry)
                deadline_at = aware_timestamp(
                    plan.deadline,
                    name="shadow_outcome.open_thesis.plan_deadline",
                )
                zone_lower = _finite(getattr(plan, "entry_zone_lower", None))
                zone_upper = _finite(getattr(plan, "entry_zone_upper", None))
            else:
                if selected is not None:
                    frozen_invalidation = getattr(selected, "invalidation", None)
                    if frozen_invalidation is not None:
                        invalidation = _finite(
                            getattr(frozen_invalidation, "price", None)
                        )
                        invalidation_id = getattr(
                            frozen_invalidation,
                            "source_level_id",
                            None,
                        )
                    frozen_draw = getattr(selected, "thesis_draw", None)
                    if frozen_draw is None:
                        frozen_draw = next(
                            iter(getattr(selected, "deliverable_targets", ())),
                            None,
                        )
                    if frozen_draw is not None:
                        target = _finite(getattr(frozen_draw, "price", None))
                        draw_id = getattr(frozen_draw, "level_id", None)
                    deadline_raw = getattr(
                        selected,
                        "episode_deadline",
                        None,
                    )
                    if deadline_raw is None:
                        deadline_raw = getattr(
                            selected,
                            "thesis_deadline",
                            None,
                        )
                    if deadline_raw is not None:
                        deadline_at = aware_timestamp(
                            deadline_raw,
                            name="shadow_outcome.open_thesis.deadline",
                        )
                if location is not None:
                    zone_lower = _finite(getattr(location, "lower_bound", None))
                    zone_upper = _finite(getattr(location, "upper_bound", None))

            if target is None or not draw_id:
                draw_reference = (
                    planned_entry
                    if planned_entry is not None
                    else (
                        (zone_lower + zone_upper) / 2.0
                        if zone_lower is not None and zone_upper is not None
                        else float(observation.price)
                    )
                )
                target, draw_id = cls._linked_thesis_draw(
                    observation,
                    thesis,
                    direction,
                    draw_reference,
                )

            selected_trigger = (
                None
                if selected is None
                else getattr(selected, "selected_trigger", None)
            )
            source_ids = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        thesis_id,
                        root_id,
                        revision,
                        selected_identity,
                        getattr(selected, "episode_id", None),
                        getattr(selected, "setup_context_id", None),
                        getattr(selected, "entry_path_id", None),
                        *getattr(thesis, "authority_source_ids", ()),
                        *getattr(thesis, "mechanism_event_ids", ()),
                        *entry_location_ids,
                        *getattr(thesis, "trigger_event_ids", ()),
                        *getattr(thesis, "obstruction_ids", ()),
                        *getattr(thesis, "conflict_ids", ()),
                        *getattr(thesis, "supporting_event_ids", ()),
                        *getattr(thesis, "opposing_event_ids", ()),
                        invalidation_id,
                        draw_id,
                    )
                    if isinstance(value, str) and value
                )
            )
            specs.append(
                _CandidateSpec(
                    event_kind="open_market_thesis_revision",
                    event_id=f"{root_id}|{revision}",
                    observed_at=updated_at,
                    direction=direction,
                    source_timeframe=Timeframe(thesis.source_timeframe),
                    source_ids=source_ids,
                    entry_zone_lower=zone_lower,
                    entry_zone_upper=zone_upper,
                    direct_invalidation=invalidation,
                    direct_invalidation_id=invalidation_id,
                    direct_entry=planned_entry,
                    direct_target=target,
                    direct_draw_id=draw_id,
                    deadline_at=deadline_at,
                    candidate_origin="open_market_thesis_revision",
                    source_playbook=(
                        None if selected is None else selected.playbook
                    ),
                    source_episode_id=(
                        None
                        if selected is None
                        else getattr(selected, "episode_id", None)
                    ),
                    source_setup_id=(
                        None
                        if selected is None
                        else getattr(selected, "setup_context_id", None)
                    ),
                    source_context_thesis_id=(
                        None
                        if selected is None
                        else getattr(selected, "context_thesis_id", None)
                    ),
                    entry_location_id=(
                        None
                        if selected is None
                        else getattr(selected, "entry_location_id", None)
                    ),
                    entry_path_id=(
                        None
                        if selected is None
                        else getattr(selected, "entry_path_id", None)
                    ),
                    lsr_displacement_id=(
                        None
                        if selected is None
                        else _hypothesis_context_identity(
                            selected,
                            "lsr_displacement_id",
                        )
                    ),
                    lsr_entry_zone_id=(
                        None
                        if selected is None
                        else _hypothesis_context_identity(
                            selected,
                            "lsr_entry_zone_id",
                        )
                    ),
                    selected_trigger_kind=(
                        None
                        if selected_trigger is None
                        else _enum_value(selected_trigger.trigger_kind)
                    ),
                    selected_trigger_id=(
                        None
                        if selected_trigger is None
                        else selected_trigger.trigger_id
                    ),
                    selected_trigger_at=(
                        None
                        if selected_trigger is None
                        else aware_timestamp(
                            selected_trigger.observed_at,
                            name="shadow_outcome.open_thesis.trigger_at",
                        )
                    ),
                    available_trigger_kinds=tuple(
                        _enum_value(item)
                        for item in (
                            ()
                            if selected_trigger is None
                            else selected_trigger.available_trigger_kinds
                        )
                    ),
                    allow_unlinked_draw_fallback=False,
                    allow_context_link_expansion=False,
                )
            )
        return tuple(specs)

    @staticmethod
    def _select_root_candidate(
        candidates: tuple[tuple[str, Any], ...],
    ) -> tuple[str | None, Any | None]:
        if not candidates:
            return None, None

        def rank(item: tuple[str, Any]) -> tuple[int, int, int, float, str]:
            identity, hypothesis = item
            feasibility_valid, _ = _plan_feasibility_state(hypothesis)
            return (
                int(not feasibility_valid),
                int(getattr(hypothesis, "plan", None) is None),
                int(getattr(hypothesis, "phase", None) is not PlaybookPhase.EXECUTABLE),
                -float(getattr(hypothesis, "playbook_match_strength", 0.0)),
                identity,
            )

        return min(candidates, key=rank)

    @staticmethod
    def _linked_thesis_draw(
        observation: Any,
        thesis: Any,
        direction: Direction,
        entry: float,
    ) -> tuple[float | None, str | None]:
        allowed = set(getattr(thesis, "draw_candidate_ids", ()))
        if not allowed:
            return None, None
        candidates: list[tuple[float, float, str]] = []
        for item in observation.liquidity_inventory:
            if (
                item.lifecycle is not LiquidityInventoryLifecycle.VISIBLE
                or item.item_id not in allowed
                or item.side != direction.opposing_liquidity_side
            ):
                continue
            contact = (
                float(item.lower_bound)
                if direction is Direction.LONG
                else float(item.upper_bound)
            )
            distance = direction.sign * (contact - entry)
            if distance > 0.0:
                candidates.append((distance, contact, item.item_id))
        if not candidates:
            return None, None
        _, price, item_id = min(candidates)
        return price, item_id

    @staticmethod
    def _playbook_candidate_specs(
        snapshot: EngineSnapshot,
    ) -> tuple[_CandidateSpec, ...]:
        """First executable occurrence for one stable episode/setup identity."""

        asof = snapshot.observation.asof
        specs: list[_CandidateSpec] = []
        context = getattr(snapshot.belief, "global_context", None)
        thesis_by_id = {
            thesis.thesis_id: thesis
            for thesis in (
                () if context is None else context.open_market_theses
            )
        }
        for candidate_id, hypothesis in _action_candidate_items(
            snapshot.belief
        ):
            feasibility_valid, _ = _plan_feasibility_state(hypothesis)
            plan = hypothesis.plan
            if (
                hypothesis.phase is not PlaybookPhase.EXECUTABLE
                or not feasibility_valid
                or plan is None
                or not all(
                    hasattr(plan, name)
                    for name in (
                        "planned_entry",
                        "invalidation",
                        "targets",
                        "deadline",
                    )
                )
                or not plan.targets
            ):
                continue
            episode_identity = next(
                (
                    value
                    for value in (
                        getattr(hypothesis, "episode_id", None),
                        getattr(hypothesis, "setup_context_id", None),
                        getattr(plan, "setup_id", None),
                    )
                    if isinstance(value, str) and value
                ),
                None,
            )
            if episode_identity is None:
                continue
            target = plan.targets[0]
            selected_trigger = getattr(
                hypothesis,
                "selected_trigger",
                None,
            )
            bound_thesis = thesis_by_id.get(
                getattr(hypothesis, "bound_market_thesis_id", None)
            )
            source_ids = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        episode_identity,
                        getattr(hypothesis, "market_thesis_id", None),
                        getattr(hypothesis, "bound_market_thesis_id", None),
                        getattr(hypothesis, "market_thesis_root_id", None),
                        getattr(hypothesis, "required_root_id", None),
                        getattr(plan, "entry_location_id", None),
                        getattr(hypothesis, "entry_path_id", None),
                        getattr(plan, "entry_path_id", None),
                        getattr(plan.invalidation, "source_level_id", None),
                        getattr(target, "level_id", None),
                        (
                            None
                            if bound_thesis is None
                            else bound_thesis.root_id
                        ),
                        *(
                            ()
                            if bound_thesis is None
                            else bound_thesis.mechanism_event_ids
                        ),
                        *(
                            ()
                            if bound_thesis is None
                            else bound_thesis.entry_location_ids
                        ),
                        *(
                            ()
                            if bound_thesis is None
                            else bound_thesis.trigger_event_ids
                        ),
                    )
                    if isinstance(value, str) and value
                )
            )
            specs.append(
                _CandidateSpec(
                    event_kind="playbook_executable",
                    # The action-candidate identity includes its canonical
                    # thesis root, so compatible roots cannot collapse into
                    # one playbook-direction executable event.
                    event_id=candidate_id,
                    observed_at=asof,
                    direction=hypothesis.direction,
                    source_timeframe=(
                        Timeframe.M5
                        if bound_thesis is None
                        else bound_thesis.source_timeframe
                    ),
                    source_ids=source_ids,
                    entry_zone_lower=_finite(
                        getattr(plan, "entry_zone_lower", None)
                    ),
                    entry_zone_upper=_finite(
                        getattr(plan, "entry_zone_upper", None)
                    ),
                    direct_invalidation=float(plan.invalidation.price),
                    direct_invalidation_id=plan.invalidation.source_level_id,
                    direct_entry=float(plan.planned_entry),
                    direct_target=float(target.price),
                    direct_draw_id=(
                        getattr(plan, "selected_draw_id", None)
                        or target.level_id
                    ),
                    deadline_at=aware_timestamp(
                        plan.deadline,
                        name="shadow_outcome.playbook_deadline",
                    ),
                    candidate_origin="playbook_candidate",
                    source_playbook=hypothesis.playbook,
                    source_episode_id=episode_identity,
                    source_setup_id=(
                        getattr(hypothesis, "setup_context_id", None)
                        or getattr(plan, "setup_id", None)
                    ),
                    source_context_thesis_id=getattr(
                        hypothesis,
                        "context_thesis_id",
                        None,
                    ),
                    entry_location_id=(
                        getattr(hypothesis, "entry_location_id", None)
                        or getattr(plan, "entry_location_id", None)
                    ),
                    entry_path_id=(
                        getattr(hypothesis, "entry_path_id", None)
                        or getattr(plan, "entry_path_id", None)
                    ),
                    lsr_displacement_id=_hypothesis_context_identity(
                        hypothesis,
                        "lsr_displacement_id",
                    ),
                    lsr_entry_zone_id=_hypothesis_context_identity(
                        hypothesis,
                        "lsr_entry_zone_id",
                    ),
                    source_action_candidate_id=candidate_id,
                    entry_episode_binding_status="exact_action_candidate",
                    selected_trigger_kind=(
                        None
                        if selected_trigger is None
                        else _enum_value(selected_trigger.trigger_kind)
                    ),
                    selected_trigger_id=(
                        None
                        if selected_trigger is None
                        else selected_trigger.trigger_id
                    ),
                    selected_trigger_at=(
                        None
                        if selected_trigger is None
                        else aware_timestamp(
                            selected_trigger.observed_at,
                            name="shadow_outcome.selected_trigger_at",
                        )
                    ),
                    available_trigger_kinds=tuple(
                        _enum_value(item)
                        for item in (
                            ()
                            if selected_trigger is None
                            else selected_trigger.available_trigger_kinds
                        )
                    ),
                )
            )
        return tuple(specs)

    def close_unresolved(self, asof: pd.Timestamp) -> None:
        if self._pending_bar is not None:
            raise RuntimeError(
                "cannot close shadow outcomes before the last Engine snapshot"
            )
        asof = aware_timestamp(asof, name="shadow_outcome.close_unresolved")
        self._censor_all(asof, "window_right_censored")

    def _consume_pending_bar(self, source_bar: Bar) -> None:
        if self._pending_bar != source_bar:
            raise ValueError(
                "shadow snapshot does not match the pending completed bar"
            )
        self._pending_bar = None

    def _validate_snapshot(
        self,
        snapshot: EngineSnapshot,
        source_bar: Bar,
        *,
        allow_equal: bool,
    ) -> Any:
        observation = brain_observation_view(snapshot.observation)
        asof = aware_timestamp(
            observation.asof,
            name="shadow_outcome.observation.asof",
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
            raise ValueError("source bar must exactly back the shadow snapshot")
        if self._last_observation_asof is not None:
            invalid = (
                asof < self._last_observation_asof
                if allow_equal
                else asof <= self._last_observation_asof
            )
            if invalid:
                raise ValueError("shadow observations must be strictly increasing")
        self._last_observation_asof = asof
        return observation

    @staticmethod
    def _event_key(spec: _CandidateSpec) -> tuple[str, str, str]:
        return (
            spec.event_kind,
            spec.event_id,
            (
                "stable_episode_identity"
                if spec.event_kind
                in {
                    "playbook_executable",
                    "open_market_thesis_revision",
                }
                else spec.observed_at.isoformat()
            ),
        )

    def _candidate_specs(
        self,
        snapshot: EngineSnapshot,
    ) -> tuple[_CandidateSpec, ...]:
        observation = brain_observation_view(snapshot.observation)
        asof = observation.asof
        specs: list[_CandidateSpec] = []
        for frame in observation.frames.values():
            for bos in getattr(frame, "structure_breaks", ()):
                if (
                    bos.lifecycle is BOSLifecycle.CONFIRMED
                    and bos.resolved_at == asof
                ):
                    specs.append(
                        _CandidateSpec(
                            "confirmed_bos",
                            bos.bos_id,
                            asof,
                            bos.direction,
                            bos.timeframe,
                            tuple(
                                value
                                for value in (
                                    bos.bos_id,
                                    bos.target_swing_id,
                                    bos.source_structure_id,
                                    bos.source_displacement_id,
                                )
                                if value
                            ),
                        )
                    )
        displacement = getattr(observation, "displacement", None)
        for transition in (
            ()
            if displacement is None
            else displacement.transitions_this_update
        ):
            if transition.lifecycle == "active" and transition.observed_at == asof:
                specs.append(
                    _CandidateSpec(
                        "displacement_active",
                        transition.entity_id,
                        asof,
                        transition.direction,
                        Timeframe.M5,
                        (transition.entity_id, transition.transition_id),
                    )
                )
        for location in observation.entry_locations:
            (
                binding_id,
                binding,
                binding_status,
                entry_path_id,
            ) = _lsr_zone_episode_binding(snapshot, location)
            context_binding, context_binding_status = (
                _lsr_zone_context_binding(snapshot, location)
            )
            metadata_owner = binding or context_binding
            if (
                binding is None
                and context_binding_status != "exact_lsr_context"
            ):
                binding_status = context_binding_status
            source_episode_id = (
                None if binding is None else getattr(binding, "episode_id", None)
            )
            source_setup_id = (
                None
                if binding is None
                else getattr(binding, "setup_context_id", None)
            )
            source_context_thesis_id = (
                None
                if metadata_owner is None
                else getattr(metadata_owner, "context_thesis_id", None)
            )
            entry_episode_terminal_at = None
            entry_episode_terminal_reason = None
            custody_key = (
                observation.symbol,
                int(observation.instrument_id),
                location.location_id,
            )
            formation_clock = bool(
                location.lifecycle is EntryLocationLifecycle.APPROACHING
                and location.formed_at == asof
            )
            first_entry_clock = bool(
                location.lifecycle is EntryLocationLifecycle.IN_ZONE
                and location.first_entered_at == asof
            )
            custody = self._lsr_zone_custody.get(custody_key)
            if formation_clock:
                frozen = _LSRZoneCustody(
                    source_action_candidate_id=binding_id,
                    source_episode_id=source_episode_id,
                    source_setup_id=source_setup_id,
                    source_context_thesis_id=source_context_thesis_id,
                    entry_location_id=location.location_id,
                    entry_path_id=entry_path_id,
                    lsr_displacement_id=location.source_displacement_id,
                    lsr_entry_zone_id=location.source_zone_id,
                    binding_status=binding_status,
                    direction=location.direction,
                    formed_at=location.formed_at,
                )
                if custody is None:
                    self._lsr_zone_custody[custody_key] = frozen
                    custody = frozen
                elif (
                    custody.entry_location_id != location.location_id
                    or custody.lsr_displacement_id
                    != location.source_displacement_id
                    or custody.lsr_entry_zone_id != location.source_zone_id
                    or custody.direction is not location.direction
                    or custody.formed_at != location.formed_at
                ):
                    raise ValueError(
                        "repeated LSR formation changed physical zone identity"
                    )
            if custody is not None and (formation_clock or first_entry_clock):
                binding_id = custody.source_action_candidate_id
                source_episode_id = custody.source_episode_id
                source_setup_id = custody.source_setup_id
                source_context_thesis_id = (
                    custody.source_context_thesis_id
                )
                entry_path_id = custody.entry_path_id
                binding_status = custody.binding_status
                entry_episode_terminal_at = custody.owner_terminal_at
                entry_episode_terminal_reason = (
                    custody.owner_terminal_reason
                )
                current_matches = tuple(
                    hypothesis
                    for identity, hypothesis in _action_candidate_items(
                        snapshot.belief
                    )
                    if identity == binding_id
                )
                lifecycle_matches = tuple(
                    hypothesis
                    for identity, hypothesis in _lifecycle_candidate_items(
                        snapshot.belief
                    )
                    if identity == binding_id
                )
                binding = (
                    current_matches[0]
                    if len(current_matches) == 1
                    else lifecycle_matches[0]
                    if len(lifecycle_matches) == 1
                    else None
                )
                metadata_owner = binding
                if (
                    first_entry_clock
                    and _lsr_zone_custody_owner_is_terminal(
                        snapshot.belief,
                        custody,
                    )
                ):
                    binding_status = "episode_terminal_before_first_entry"
                    binding = None
                    metadata_owner = None
                elif (
                    first_entry_clock
                    and custody.binding_status
                    in _EXACT_ENTRY_EPISODE_BINDINGS
                ):
                    physical_path_ids = _entry_location_path_ids(
                        snapshot.observation,
                        location.location_id,
                    )
                    projection = (
                        None
                        if binding is None
                        else _lsr_active_market_belief_projection(
                            snapshot.belief,
                            str(binding_id),
                            binding,
                        )
                    )
                    if physical_path_ids != (custody.entry_path_id,):
                        binding_status = "entry_episode_binding_invalid"
                    elif projection is None:
                        binding_status = (
                            "entry_episode_owner_projection_missing_"
                            "before_first_entry"
                        )
                        binding = None
                        metadata_owner = None
                    elif not _lsr_zone_trigger_clock_is_causal(
                        binding,
                        projection[0],
                        location,
                    ):
                        binding_status = "entry_episode_binding_invalid"
                    elif not current_matches:
                        binding_status = (
                            "exact_retained_entry_episode_custody"
                        )
            selected_trigger = (
                None
                if binding is None
                else getattr(binding, "selected_trigger", None)
            )
            source_values = (
                location.location_id,
                location.source_zone_id,
                location.source_displacement_id,
                location.source_bos_id,
                binding_id,
                source_episode_id,
                source_setup_id,
                source_context_thesis_id,
                (
                    None
                    if metadata_owner is None
                    else getattr(metadata_owner, "required_root_id", None)
                ),
                (
                    None
                    if metadata_owner is None
                    else getattr(
                        metadata_owner,
                        "bound_market_thesis_id",
                        None,
                    )
                ),
                entry_path_id,
                (
                    None
                    if selected_trigger is None
                    else getattr(selected_trigger, "trigger_id", None)
                ),
            )
            common = {
                "source_ids": tuple(
                    dict.fromkeys(
                        value
                        for value in source_values
                        if isinstance(value, str) and value
                    )
                ),
                "entry_zone_lower": location.lower_bound,
                "entry_zone_upper": location.upper_bound,
                "direct_invalidation": location.failure_boundary,
                "direct_invalidation_id": location.location_id,
                "source_playbook": (
                    None
                    if source_context_thesis_id is None
                    and source_episode_id is None
                    else Playbook.LIQUIDITY_SWEEP_REVERSAL
                ),
                "source_episode_id": source_episode_id,
                "source_setup_id": source_setup_id,
                "source_context_thesis_id": source_context_thesis_id,
                "entry_location_id": location.location_id,
                "entry_path_id": entry_path_id,
                "lsr_displacement_id": location.source_displacement_id,
                "lsr_entry_zone_id": location.source_zone_id,
                "source_action_candidate_id": binding_id,
                "entry_episode_binding_status": binding_status,
                "entry_episode_terminal_at": (
                    entry_episode_terminal_at
                ),
                "entry_episode_terminal_reason": (
                    entry_episode_terminal_reason
                ),
                "selected_trigger_kind": (
                    None
                    if selected_trigger is None
                    else _enum_value(selected_trigger.trigger_kind)
                ),
                "selected_trigger_id": (
                    None
                    if selected_trigger is None
                    else selected_trigger.trigger_id
                ),
                "selected_trigger_at": (
                    None
                    if selected_trigger is None
                    else aware_timestamp(
                        selected_trigger.observed_at,
                        name="shadow_outcome.zone_event.trigger_at",
                    )
                ),
                "available_trigger_kinds": tuple(
                    _enum_value(item)
                    for item in (
                        ()
                        if selected_trigger is None
                        else selected_trigger.available_trigger_kinds
                    )
                ),
            }
            if (
                formation_clock
                and (
                    context_binding is not None
                    or context_binding_status
                    == "lsr_context_binding_ambiguous"
                )
            ):
                specs.append(
                    _CandidateSpec(
                        event_kind=(
                            f"eligible_entry_{location.source_zone_kind}"
                        ),
                        event_id=location.location_id,
                        observed_at=asof,
                        direction=location.direction,
                        source_timeframe=Timeframe.M5,
                        candidate_origin="shadow_lsr_eligible_entry_zone",
                        **common,
                    )
                )
            if (
                first_entry_clock
            ):
                specs.append(
                    _CandidateSpec(
                        event_kind=f"first_entry_{location.source_zone_kind}",
                        event_id=location.location_id,
                        observed_at=asof,
                        direction=location.direction,
                        source_timeframe=Timeframe.M1,
                        **common,
                    )
                )
        for manipulation in observation.group4_manipulation_transitions_this_update:
            direction = (
                Direction.SHORT
                if manipulation.side == "above"
                else Direction.LONG
            )
            sources = tuple(
                dict.fromkeys(
                    (
                        manipulation.manipulation_id,
                        manipulation.source_id,
                        manipulation.source_inventory_item_id,
                        *manipulation.crossed_source_ids,
                    )
                )
            )
            if (
                manipulation.lifecycle is ManipulationLifecycle.SWEPT
                and manipulation.swept_at == asof
            ):
                specs.append(
                    _CandidateSpec(
                        "liquidity_sweep",
                        manipulation.manipulation_id,
                        asof,
                        direction,
                        manipulation.source_timeframe,
                        sources,
                        direct_invalidation=manipulation.sweep_extreme,
                        direct_invalidation_id=manipulation.manipulation_id,
                    )
                )
            if (
                manipulation.lifecycle is ManipulationLifecycle.REACCEPTED
                and manipulation.source_kind == "mature_range_boundary"
                and manipulation.reaccepted_at == asof
            ):
                specs.append(
                    _CandidateSpec(
                        "mature_range_reentry",
                        manipulation.manipulation_id,
                        asof,
                        direction,
                        manipulation.source_timeframe,
                        sources,
                        direct_invalidation=manipulation.sweep_extreme,
                        direct_invalidation_id=manipulation.manipulation_id,
                    )
                )
        for reference in observation.micro_bos_references:
            if reference.qualified and reference.resolved_at == asof:
                location = self._location(
                    observation,
                    reference.context_id,
                )
                specs.append(
                    _CandidateSpec(
                        "qualified_micro_bos",
                        reference.reference_id,
                        asof,
                        reference.expected_direction,
                        Timeframe.M1,
                        (
                            reference.reference_id,
                            reference.context_id,
                            reference.bos_id,
                            reference.target_swing_id,
                        ),
                        None if location is None else location.lower_bound,
                        None if location is None else location.upper_bound,
                        None if location is None else location.failure_boundary,
                        None if location is None else location.location_id,
                    )
                )
        for reacceptance in observation.qualified_reacceptances:
            if (
                reacceptance.lifecycle
                is ReacceptanceLifecycle.HELD
                and reacceptance.held_at == asof
            ):
                location = self._location(
                    observation,
                    reacceptance.context_id,
                )
                specs.append(
                    _CandidateSpec(
                        "qualified_reacceptance",
                        reacceptance.reacceptance_id,
                        asof,
                        reacceptance.direction,
                        Timeframe.M1,
                        (
                            reacceptance.reacceptance_id,
                            reacceptance.context_id,
                            reacceptance.source_entity_id,
                        ),
                        None if location is None else location.lower_bound,
                        None if location is None else location.upper_bound,
                        reacceptance.failure_boundary,
                        reacceptance.reacceptance_id,
                    )
                )
        return tuple(specs)

    @staticmethod
    def _location(observation: Any, location_id: str) -> Any | None:
        observation = brain_observation_view(observation)
        return next(
            (
                item
                for item in observation.entry_locations
                if item.location_id == location_id
            ),
            None,
        )

    def _register(
        self,
        snapshot: EngineSnapshot,
        source_bar: Bar,
        spec: _CandidateSpec,
    ) -> None:
        observation = snapshot.observation
        decision_price = float(observation.price)
        zone = (
            spec.entry_zone_lower is not None
            and spec.entry_zone_upper is not None
        )
        entry_reference = (
            float(spec.direct_entry)
            if spec.direct_entry is not None
            else (
                (
                    float(spec.entry_zone_lower)
                    + float(spec.entry_zone_upper)
                )
                / 2.0
                if zone
                else None
            )
        )
        geometry_reference = (
            decision_price if entry_reference is None else entry_reference
        )
        invalidation, invalidation_id = self._select_invalidation(
            snapshot,
            spec,
            geometry_reference,
        )
        if spec.direct_target is not None and spec.direct_draw_id:
            target, draw_id = spec.direct_target, spec.direct_draw_id
        elif spec.allow_unlinked_draw_fallback:
            target, draw_id = self._select_draw(
                observation,
                spec.direction,
                geometry_reference,
                excluded_ids=set(spec.source_ids),
            )
        else:
            target, draw_id = None, None
        diagnostics = self._playbook_diagnostics(snapshot, spec)
        candidate_id = self._candidate_id(observation, spec)
        self._candidate_count += 1
        self._candidate_events[spec.event_kind] += 1
        if spec.event_kind in _LSR_ZONE_EVENT_KINDS:
            self._entry_episode_bindings[
                spec.entry_episode_binding_status
                or "entry_episode_binding_unspecified"
            ] += 1
        geometry_incomplete_reason = self._geometry_incomplete_reason(
            spec.direction,
            geometry_reference,
            invalidation,
            invalidation_id,
            target,
            draw_id,
        )
        geometry_complete = geometry_incomplete_reason is None
        if not geometry_complete:
            self._emit_incomplete(
                snapshot,
                spec,
                candidate_id,
                diagnostics,
                entry_reference,
                invalidation,
                invalidation_id,
                target,
                draw_id,
                geometry_incomplete_reason,
            )
            return
        top = self._top_directional_hypothesis(snapshot, spec.direction)
        candidate = _OpenCandidate(
            candidate_id=candidate_id,
            event_kind=spec.event_kind,
            event_id=spec.event_id,
            observed_at=spec.observed_at,
            symbol=observation.symbol,
            instrument_id=observation.instrument_id,
            direction=spec.direction,
            source_timeframe=spec.source_timeframe,
            source_ids=spec.source_ids,
            candidate_origin=spec.candidate_origin,
            source_playbook=spec.source_playbook,
            source_episode_id=spec.source_episode_id,
            source_setup_id=spec.source_setup_id,
            source_context_thesis_id=spec.source_context_thesis_id,
            entry_location_id=spec.entry_location_id,
            entry_path_id=spec.entry_path_id,
            lsr_displacement_id=spec.lsr_displacement_id,
            lsr_entry_zone_id=spec.lsr_entry_zone_id,
            entry_episode_binding_status=(
                spec.entry_episode_binding_status
            ),
            entry_episode_terminal_at=spec.entry_episode_terminal_at,
            entry_episode_terminal_reason=(
                spec.entry_episode_terminal_reason
            ),
            selected_trigger_kind=spec.selected_trigger_kind,
            selected_trigger_id=spec.selected_trigger_id,
            selected_trigger_at=spec.selected_trigger_at,
            available_trigger_kinds=spec.available_trigger_kinds,
            decision_price=decision_price,
            entry_rule=(
                SHADOW_OUTCOME_PROTOCOL["playbook_entry_rule"]
                if spec.event_kind == "playbook_executable"
                else (
                    SHADOW_OUTCOME_PROTOCOL["zone_entry_rule"]
                    if zone
                    else SHADOW_OUTCOME_PROTOCOL["non_zone_entry_rule"]
                )
            ),
            entry_reference_price=entry_reference,
            entry_zone_lower=spec.entry_zone_lower,
            entry_zone_upper=spec.entry_zone_upper,
            invalidation_price=float(invalidation),
            invalidation_source_id=str(invalidation_id),
            draw_id=str(draw_id),
            target_price=float(target),
            deadline_at=spec.deadline_at,
            deadline_real_1m_bars=(
                None
                if spec.event_kind == "playbook_executable"
                else _HORIZON_REAL_BARS
            ),
            playbook_diagnostics=diagnostics,
            decision_action=_enum_value(snapshot.decision.selected_action),
            risk_action=_enum_value(snapshot.risk.final_action),
            brain_phase=None if top is None else _enum_value(top.phase),
            thesis_strength=None if top is None else _finite(top.thesis_strength),
            sequence_progress=None if top is None else _finite(top.sequence_progress),
            location_quality=None if top is None else _finite(top.location_quality),
            entry_readiness=None if top is None else _finite(top.entry_readiness),
            delivery_quality=None if top is None else _finite(top.delivery_quality),
            uncertainty=None if top is None else _finite(top.uncertainty),
            geometry_incomplete_reason=None,
        )
        self._open[candidate_id] = candidate
        if (
            candidate.deadline_at is not None
            and candidate.deadline_at <= candidate.observed_at
        ):
            # A thesis can first become visible to the diagnostic recorder at
            # the formal-window boundary while retaining a plan deadline from
            # warmup.  It is already expired at this candidate clock, so there
            # is no causal future path to observe.  Preserve the diagnostic
            # row, but expire it immediately at observation time rather than
            # emitting a terminal clock in the past.
            self._resolve(
                candidate,
                self._deadline_terminal_at(candidate),
                "entry_unfilled_deadline",
                censored=False,
            )

    @staticmethod
    def _candidate_id(observation: Any, spec: _CandidateSpec) -> str:
        payload = "|".join(
            (
                str(SHADOW_OUTCOME_PROTOCOL["protocol_version"]),
                observation.symbol,
                str(observation.instrument_id),
                spec.event_kind,
                spec.event_id,
                spec.observed_at.isoformat(),
                spec.direction.value,
            )
        )
        return "shadow:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _geometry_incomplete_reason(
        direction: Direction,
        entry: float,
        invalidation: float | None,
        invalidation_id: str | None,
        target: float | None,
        draw_id: str | None,
    ) -> str | None:
        missing: list[str] = []
        if invalidation is None or not invalidation_id:
            missing.append("invalidation")
        if target is None or not draw_id:
            missing.append("draw")
        if missing:
            return "missing_" + "_and_".join(missing)
        valid = (
            direction is Direction.LONG
            and invalidation < entry < target
        ) or (
            direction is Direction.SHORT
            and target < entry < invalidation
        )
        return None if valid else "invalid_directional_geometry"

    @staticmethod
    def _valid_geometry(
        direction: Direction,
        entry: float,
        invalidation: float | None,
        target: float | None,
    ) -> bool:
        """Recheck frozen prices after a later limit fill."""

        if invalidation is None or target is None:
            return False
        return (
            direction is Direction.LONG
            and invalidation < entry < target
        ) or (
            direction is Direction.SHORT
            and target < entry < invalidation
        )

    @staticmethod
    def _select_invalidation(
        snapshot: EngineSnapshot,
        spec: _CandidateSpec,
        entry: float,
    ) -> tuple[float | None, str | None]:
        observation = snapshot.observation
        if spec.direct_invalidation is not None:
            price = float(spec.direct_invalidation)
            if spec.direction.sign * (entry - price) > 0:
                return price, spec.direct_invalidation_id
            return None, None
        timeframe_rank = {
            Timeframe.H4: 4,
            Timeframe.H1: 3,
            Timeframe.M15: 2,
            Timeframe.M5: 1,
            Timeframe.M1: 0,
        }
        minimum_rank = timeframe_rank[spec.source_timeframe]
        linked_ids = set(spec.source_ids) | {spec.event_id}
        if spec.allow_context_link_expansion:
            context = getattr(snapshot.belief, "global_context", None)
            for thesis in (
                () if context is None else context.open_market_theses
            ):
                if (
                    thesis.direction in {None, spec.direction}
                    and (
                        thesis.root_id in linked_ids
                        or linked_ids.intersection(
                            {
                                *thesis.mechanism_event_ids,
                                *getattr(thesis, "entry_location_ids", ()),
                                *getattr(thesis, "trigger_event_ids", ()),
                            }
                        )
                    )
                ):
                    linked_ids.update(
                        getattr(thesis, "authority_source_ids", ())
                    )
        candidates: list[tuple[float, int, float, str]] = []
        for frame in observation.frames.values():
            if timeframe_rank.get(frame.timeframe, -1) < minimum_rank:
                continue
            for structure in getattr(frame, "structures", ()):
                price = _finite(getattr(structure, "protected_price", None))
                structure_ids = {
                    value
                    for value in (
                        getattr(structure, "structure_id", None),
                        getattr(structure, "protected_swing_id", None),
                        getattr(structure, "latest_high_id", None),
                        getattr(structure, "latest_low_id", None),
                    )
                    if value
                }
                if (
                    structure.lifecycle is StructureLifecycle.CONFIRMED
                    and structure.direction is spec.direction
                    and price is not None
                    and spec.direction.sign * (entry - price) > 0
                    and structure.structure_id
                    and linked_ids.intersection(structure_ids)
                ):
                    candidates.append(
                        (
                            abs(entry - price),
                            -timeframe_rank.get(frame.timeframe, 0),
                            price,
                            structure.structure_id,
                        )
                    )
        for item in observation.liquidity_inventory:
            if (
                item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
                and item.side == spec.direction.invalidation_side
                and timeframe_rank.get(item.timeframe, -1) >= minimum_rank
                and linked_ids.intersection(
                    {item.item_id, *item.source_ids}
                )
                and (
                    bool(getattr(item, "is_protected_swing", False))
                    or getattr(item, "structural_rank", "internal")
                    == "external"
                )
            ):
                price = (
                    float(item.lower_bound)
                    if spec.direction is Direction.LONG
                    else float(item.upper_bound)
                )
                if spec.direction.sign * (entry - price) > 0:
                    candidates.append(
                        (
                            abs(entry - price),
                            -timeframe_rank.get(item.timeframe, 0),
                            price,
                            item.item_id,
                        )
                    )
        if not candidates:
            return None, None
        _, _, price, source_id = min(candidates)
        return price, source_id

    @staticmethod
    def _select_draw(
        observation: Any,
        direction: Direction,
        entry: float,
        *,
        excluded_ids: set[str],
    ) -> tuple[float | None, str | None]:
        candidates: list[tuple[float, float, str]] = []
        for item in observation.liquidity_inventory:
            if (
                item.lifecycle is not LiquidityInventoryLifecycle.VISIBLE
                or item.side != direction.opposing_liquidity_side
                or item.item_id in excluded_ids
            ):
                continue
            contact = (
                float(item.lower_bound)
                if direction is Direction.LONG
                else float(item.upper_bound)
            )
            distance = direction.sign * (contact - entry)
            if distance > 0:
                candidates.append((distance, contact, item.item_id))
        if not candidates:
            return None, None
        _, price, item_id = min(candidates)
        return price, item_id

    def _playbook_diagnostics(
        self,
        snapshot: EngineSnapshot,
        spec: _CandidateSpec,
    ) -> tuple[dict[str, Any], ...]:
        action_candidates = _action_candidate_items(snapshot.belief)
        hypotheses: dict[Playbook, list[Any]] = {}
        for _, hypothesis in action_candidates:
            if hypothesis.direction is spec.direction:
                hypotheses.setdefault(hypothesis.playbook, []).append(
                    hypothesis
                )
        context = getattr(snapshot.belief, "global_context", None)
        open_theses = () if context is None else context.open_market_theses
        executable_source = None
        executable_thesis = None
        frozen_executable_binding = False
        zone_binding_source = None
        zone_binding_thesis = None
        frozen_zone_binding = False
        zone_binding_status = spec.entry_episode_binding_status
        zone_event = spec.event_kind in _LSR_ZONE_EVENT_KINDS
        if spec.event_kind == _PLAYBOOK_EXECUTABLE_EVENT_KIND:
            direct_matches = tuple(
                hypothesis
                for identity, hypothesis in action_candidates
                if identity == spec.event_id
            )
            if len(direct_matches) != 1:
                raise ValueError(
                    "playbook executable Shadow candidate must resolve exactly "
                    "one frozen action-candidate identity"
                )
            executable_source = direct_matches[0]
            plan = getattr(executable_source, "plan", None)
            episode_identity = next(
                (
                    value
                    for value in (
                        getattr(executable_source, "episode_id", None),
                        getattr(executable_source, "setup_context_id", None),
                        None if plan is None else getattr(plan, "setup_id", None),
                    )
                    if isinstance(value, str) and value
                ),
                None,
            )
            if (
                spec.source_playbook is None
                or executable_source.playbook is not spec.source_playbook
                or executable_source.direction is not spec.direction
                or episode_identity != spec.source_episode_id
            ):
                raise ValueError(
                    "playbook executable Shadow identity disagrees with its "
                    "frozen action candidate"
                )
            required_root_id = getattr(
                executable_source,
                "required_root_id",
                None,
            )
            bound_thesis_id = getattr(
                executable_source,
                "bound_market_thesis_id",
                None,
            )
            thesis_matches = tuple(
                thesis
                for thesis in open_theses
                if thesis.root_id == required_root_id
                and thesis.thesis_id == bound_thesis_id
            )
            if len(thesis_matches) > 1:
                raise ValueError(
                    "playbook executable Shadow candidate must bind exactly "
                    "one current open-market thesis root"
                )
            if thesis_matches:
                executable_thesis = thesis_matches[0]
            else:
                frozen_executable_binding = (
                    _frozen_executable_binding_is_exact(
                        snapshot.belief,
                        spec.event_id,
                        executable_source,
                        plan,
                        source_episode_id=spec.source_episode_id,
                        source_setup_id=spec.source_setup_id,
                    )
                )
                if not frozen_executable_binding:
                    raise ValueError(
                        "playbook executable Shadow candidate without a "
                        "current root requires one exact live frozen "
                        "Context/EntryEpisode binding"
                    )
        elif zone_event:
            if (
                spec.source_action_candidate_id is not None
                and zone_binding_status in _EXACT_ENTRY_EPISODE_BINDINGS
            ):
                direct_matches = tuple(
                    hypothesis
                    for identity, hypothesis in action_candidates
                    if identity == spec.source_action_candidate_id
                )
                if (
                    len(direct_matches) == 1
                    and _lsr_active_market_belief_projection(
                        snapshot.belief,
                        str(spec.source_action_candidate_id),
                        direct_matches[0],
                    )
                    is not None
                    and _lsr_zone_spec_matches(direct_matches[0], spec)
                ):
                    zone_binding_source = direct_matches[0]
                else:
                    zone_binding_status = (
                        "entry_episode_binding_ambiguous"
                        if len(direct_matches) > 1
                        else "entry_episode_binding_invalid"
                    )
            elif zone_binding_status in _EXACT_ENTRY_EPISODE_BINDINGS:
                zone_binding_status = "entry_episode_binding_invalid"
            if zone_binding_source is not None:
                required_root_id = getattr(
                    zone_binding_source,
                    "required_root_id",
                    None,
                )
                bound_thesis_id = getattr(
                    zone_binding_source,
                    "bound_market_thesis_id",
                    None,
                )
                thesis_matches = tuple(
                    thesis
                    for thesis in open_theses
                    if thesis.root_id == required_root_id
                    and thesis.thesis_id == bound_thesis_id
                )
                if len(thesis_matches) > 1:
                    raise ValueError(
                        "LSR zone Shadow candidate must bind exactly one "
                        "current open-market thesis root"
                    )
                if thesis_matches:
                    zone_binding_thesis = thesis_matches[0]
                else:
                    frozen_zone_binding = _frozen_lsr_zone_binding_is_exact(
                        snapshot.belief,
                        str(spec.source_action_candidate_id),
                        zone_binding_source,
                        source_episode_id=spec.source_episode_id,
                        source_context_thesis_id=(
                            spec.source_context_thesis_id
                        ),
                        entry_location_id=spec.entry_location_id,
                        entry_path_id=spec.entry_path_id,
                    )
                    if not frozen_zone_binding:
                        zone_binding_source = None
                        zone_binding_status = (
                            "entry_episode_parent_context_missing"
                        )
        connected_ids = set(spec.source_ids) | {spec.event_id}
        connected = tuple(
            thesis
            for thesis in open_theses
            if thesis.direction in {None, spec.direction}
            and (
                thesis.root_id in connected_ids
                or bool(
                    connected_ids.intersection(
                        {
                            *thesis.mechanism_event_ids,
                            *getattr(thesis, "entry_location_ids", ()),
                            *getattr(thesis, "trigger_event_ids", ()),
                        }
                    )
                )
            )
        )
        # Resolve the Eye candidate to one canonical graph root before
        # choosing an evaluated Brain candidate.  Selection by playbook alone
        # would collapse compatible roots back into the old six-slot summary.
        direct_root_owners = tuple(
            thesis for thesis in connected if thesis.root_id == spec.event_id
        )
        source_root_owners = tuple(
            thesis for thesis in connected if thesis.root_id in connected_ids
        )
        event_owners = tuple(
            thesis
            for thesis in connected
            if spec.event_id
            in {
                *thesis.mechanism_event_ids,
                *getattr(thesis, "entry_location_ids", ()),
                *getattr(thesis, "trigger_event_ids", ()),
            }
        )
        frozen_root_binding = bool(
            frozen_executable_binding or frozen_zone_binding
        )
        candidate_root_resolved = bool(
            executable_thesis is not None
            or zone_binding_thesis is not None
            or frozen_root_binding
        )
        chosen = executable_thesis or zone_binding_thesis
        if chosen is None and not frozen_root_binding:
            for owners in (
                direct_root_owners,
                source_root_owners,
                event_owners,
                connected,
            ):
                if len(owners) == 1:
                    chosen = owners[0]
                    candidate_root_resolved = True
                    break
        if chosen is None and connected and not frozen_root_binding:
            # Descriptive only: ambiguous graph ownership never grants exact
            # root binding or action acceptance.
            chosen = sorted(connected, key=lambda item: item.thesis_id)[0]
        output: list[dict[str, Any]] = []
        for playbook in _PLAYBOOKS:
            slot_candidates = tuple(hypotheses.get(playbook, ()))
            hypothesis = (
                executable_source
                if executable_source is not None
                and playbook is spec.source_playbook
                else zone_binding_source
                if zone_event
                and playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
                else next(
                    (
                        item
                        for item in slot_candidates
                        if chosen is not None
                        and getattr(item, "required_root_id", None)
                        == chosen.root_id
                    ),
                    None,
                )
            )
            if (
                hypothesis is None
                and not candidate_root_resolved
                and not (
                    zone_event
                    and playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
                )
            ):
                hypothesis = next(
                    iter(
                        sorted(
                            slot_candidates,
                            key=lambda item: item.key,
                        )
                    ),
                    None,
                )
            if hypothesis is None:
                missing_gate = (
                    zone_binding_status
                    if (
                        zone_event
                        and playbook
                        is Playbook.LIQUIDITY_SWEEP_REVERSAL
                        and zone_binding_status
                    )
                    else (
                        "root_specific_hypothesis_missing"
                        if chosen is not None
                        else "hypothesis_missing"
                    )
                )
                output.append(
                    {
                        "playbook": playbook.value,
                        "hypothesis_key": None,
                        "phase": PlaybookPhase.INACTIVE.value,
                        "accepted": False,
                        "runtime_hypothesis_accepted": False,
                        "quadrant_acceptance_basis": (
                            "candidate_specific_accepted"
                        ),
                        "runtime_enabled": playbook in self._enabled_playbooks,
                        "first_failed_gate": missing_gate,
                        "failed_gates": [missing_gate],
                        "market_thesis_id": (
                            None if chosen is None else chosen.thesis_id
                        ),
                        "market_thesis_root_id": (
                            None if chosen is None else chosen.root_id
                        ),
                        "market_thesis_mechanism": (
                            None if chosen is None else chosen.mechanism
                        ),
                        "market_thesis_authority_relation": (
                            None
                            if chosen is None
                            else chosen.authority_relation
                        ),
                        "graph_connected": chosen is not None,
                        "exact_root_bound": False,
                        "market_thesis_match_status": (
                            "no_open_thesis"
                            if chosen is None
                            else "mechanism_direction_unmatched"
                        ),
                        "selected_hypothesis_match_strength": None,
                        "plan_delivery_valid": False,
                        "plan_feasibility_valid": False,
                        "plan_feasibility_failure_reason": (
                            missing_gate
                        ),
                        **_liquidity_route_diagnostics(None),
                        "event_order_signature": [spec.event_kind],
                        "episode_id": (
                            spec.source_episode_id if zone_event else None
                        ),
                        "setup_id": (
                            spec.source_setup_id if zone_event else None
                        ),
                        "context_thesis_id": (
                            spec.source_context_thesis_id
                            if zone_event
                            else None
                        ),
                        "parent_context_thesis_id": (
                            spec.source_context_thesis_id
                            if zone_event
                            else None
                        ),
                        "entry_location_id": spec.entry_location_id,
                        "entry_path_id": spec.entry_path_id,
                        "lsr_displacement_id": spec.lsr_displacement_id,
                        "lsr_entry_zone_id": spec.lsr_entry_zone_id,
                        "entry_episode_binding_status": (
                            zone_binding_status
                            if zone_event
                            else spec.entry_episode_binding_status
                        ),
                        "entry_episode_terminal_at": (
                            None
                            if spec.entry_episode_terminal_at is None
                            else spec.entry_episode_terminal_at.isoformat()
                        ),
                        "entry_episode_terminal_reason": (
                            spec.entry_episode_terminal_reason
                        ),
                    }
                )
                continue
            if (
                frozen_executable_binding
                and hypothesis is executable_source
            ) or (
                frozen_zone_binding
                and hypothesis is zone_binding_source
            ):
                match_status = "exact_root_bound"
                strength = float(hypothesis.playbook_match_strength)
            elif chosen is None:
                match_status = "no_connected_open_thesis"
                strength = None
            elif (
                candidate_root_resolved
                and getattr(hypothesis, "required_root_id", None)
                == chosen.root_id
                and hypothesis.bound_market_thesis_id == chosen.thesis_id
            ):
                match_status = "exact_root_bound"
                strength = (
                    float(hypothesis.playbook_match_strength)
                    if hypothesis.market_thesis_id == chosen.thesis_id
                    else None
                )
            elif chosen.thesis_id in hypothesis.market_thesis_ids:
                match_status = "matched_root_unbound"
                strength = (
                    float(hypothesis.playbook_match_strength)
                    if hypothesis.market_thesis_id == chosen.thesis_id
                    else None
                )
            else:
                match_status = "mechanism_direction_unmatched"
                strength = None
            failed = [
                name
                for name, passed in hypothesis.hard_gate_results.items()
                if not passed
            ]
            (
                plan_feasibility_valid,
                plan_feasibility_failure_reason,
            ) = _plan_feasibility_state(hypothesis)
            event_order_signature = _frozen_event_order_signature(
                hypothesis,
                fallback_event_kind=spec.event_kind,
            )
            exact_binding = bool(
                candidate_root_resolved
                and match_status == "exact_root_bound"
                and (
                    (
                        chosen is not None
                        and getattr(hypothesis, "required_root_id", None)
                        == chosen.root_id
                        and hypothesis.bound_market_thesis_id
                        == chosen.thesis_id
                    )
                    or (
                        frozen_executable_binding
                        and hypothesis is executable_source
                    )
                    or (
                        frozen_zone_binding
                        and hypothesis is zone_binding_source
                    )
                )
            )
            runtime_enabled = playbook in self._enabled_playbooks
            graph_connected = bool(
                chosen is not None
                or (
                    frozen_executable_binding
                    and hypothesis is executable_source
                )
                or (
                    frozen_zone_binding
                    and hypothesis is zone_binding_source
                )
            )
            frozen_metadata_source = (
                hypothesis
                if (
                    (
                        frozen_executable_binding
                        and hypothesis is executable_source
                    )
                    or (
                        frozen_zone_binding
                        and hypothesis is zone_binding_source
                    )
                )
                else None
            )
            first_rejection_reason = next(
                (
                    reason
                    for reason in (
                        (
                            None
                            if runtime_enabled
                            else "runtime_disabled"
                        ),
                        (
                            "graph_identity_disconnected"
                            if not graph_connected
                            else None
                        ),
                        (
                            "market_thesis_exact_root_binding"
                            if hypothesis.market_thesis_binding_required
                            and not exact_binding
                            else None
                        ),
                        (
                            failed[0]
                            if failed
                            else (
                                "hard_gates_missing"
                                if not hypothesis.hard_gate_results
                                else None
                            )
                        ),
                        (
                            None
                            if plan_feasibility_valid
                            else plan_feasibility_failure_reason
                        ),
                        (
                            None
                            if hypothesis.phase
                            is PlaybookPhase.EXECUTABLE
                            else "phase_not_executable"
                        ),
                    )
                    if reason is not None
                ),
                None,
            )
            runtime_hypothesis_accepted = bool(
                runtime_enabled
                and hypothesis.phase is PlaybookPhase.EXECUTABLE
                and plan_feasibility_valid
                and hypothesis.hard_gate_results
                and not failed
                and (
                    not hypothesis.market_thesis_binding_required
                    or hypothesis.market_thesis_action_bound
                )
            )
            accepted = bool(
                runtime_hypothesis_accepted
                and graph_connected
                and (
                    not hypothesis.market_thesis_binding_required
                    or exact_binding
                )
            )
            output.append(
                {
                    "playbook": playbook.value,
                    "hypothesis_key": hypothesis.key,
                    "phase": _enum_value(hypothesis.phase),
                    "accepted": accepted,
                    "runtime_hypothesis_accepted": (
                        runtime_hypothesis_accepted
                    ),
                    "quadrant_acceptance_basis": (
                        "candidate_specific_accepted"
                    ),
                    "runtime_enabled": runtime_enabled,
                    "first_failed_gate": first_rejection_reason,
                    "failed_gates": failed,
                    "market_thesis_id": (
                        chosen.thesis_id
                        if chosen is not None
                        else getattr(
                            frozen_metadata_source,
                            "market_thesis_id",
                            None,
                        )
                    ),
                    "market_thesis_root_id": (
                        chosen.root_id
                        if chosen is not None
                        else getattr(
                            frozen_metadata_source,
                            "market_thesis_root_id",
                            None,
                        )
                    ),
                    "market_thesis_mechanism": (
                        chosen.mechanism
                        if chosen is not None
                        else getattr(
                            frozen_metadata_source,
                            "market_thesis_mechanism",
                            None,
                        )
                    ),
                    "market_thesis_authority_relation": (
                        chosen.authority_relation
                        if chosen is not None
                        else getattr(
                            frozen_metadata_source,
                            "market_thesis_authority_relation",
                            None,
                        )
                    ),
                    "graph_connected": graph_connected,
                    "exact_root_bound": exact_binding,
                    "market_thesis_match_status": match_status,
                    "selected_hypothesis_match_strength": strength,
                    # Compatibility name retained in the compact JSON, but
                    # its value now comes solely from the common Brain-owned
                    # feasibility contract.
                    "plan_delivery_valid": plan_feasibility_valid,
                    "plan_feasibility_valid": plan_feasibility_valid,
                    "plan_feasibility_failure_reason": (
                        plan_feasibility_failure_reason
                    ),
                    **_liquidity_route_diagnostics(hypothesis),
                    "event_order_signature": list(event_order_signature),
                    "hard_gates": dict(hypothesis.hard_gate_results),
                    "episode_id": getattr(hypothesis, "episode_id", None),
                    "setup_id": getattr(
                        hypothesis,
                        "setup_context_id",
                        None,
                    ),
                    "context_thesis_id": getattr(
                        hypothesis,
                        "context_thesis_id",
                        None,
                    ),
                    "parent_context_thesis_id": getattr(
                        hypothesis,
                        "parent_context_thesis_id",
                        None,
                    ),
                    "entry_location_id": getattr(
                        hypothesis,
                        "entry_location_id",
                        spec.entry_location_id,
                    ),
                    "entry_path_id": getattr(
                        hypothesis,
                        "entry_path_id",
                        spec.entry_path_id,
                    ),
                    "lsr_displacement_id": (
                        _hypothesis_context_identity(
                            hypothesis,
                            "lsr_displacement_id",
                        )
                        or spec.lsr_displacement_id
                    ),
                    "lsr_entry_zone_id": (
                        _hypothesis_context_identity(
                            hypothesis,
                            "lsr_entry_zone_id",
                        )
                        or spec.lsr_entry_zone_id
                    ),
                    "entry_episode_binding_status": (
                        zone_binding_status
                        if (
                            zone_event
                            and playbook
                            is Playbook.LIQUIDITY_SWEEP_REVERSAL
                        )
                        else spec.entry_episode_binding_status
                    ),
                    "entry_episode_terminal_at": (
                        None
                        if spec.entry_episode_terminal_at is None
                        else spec.entry_episode_terminal_at.isoformat()
                    ),
                    "entry_episode_terminal_reason": (
                        spec.entry_episode_terminal_reason
                    ),
                    "bound_market_thesis_id": getattr(
                        hypothesis,
                        "bound_market_thesis_id",
                        None,
                    ),
                    "thesis_strength": _finite(hypothesis.thesis_strength),
                    "sequence_progress": _finite(hypothesis.sequence_progress),
                    "location_quality": _finite(hypothesis.location_quality),
                    "entry_readiness": _finite(hypothesis.entry_readiness),
                    "delivery_quality": _finite(hypothesis.delivery_quality),
                    "uncertainty": _finite(hypothesis.uncertainty),
                }
            )
        return tuple(output)

    @staticmethod
    def _top_directional_hypothesis(
        snapshot: EngineSnapshot,
        direction: Direction,
    ) -> Any | None:
        candidates = tuple(
            item
            for _, item in _action_candidate_items(snapshot.belief)
            if item.direction is direction
        )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                float(item.playbook_match_strength),
                float(item.thesis_strength or 0.0),
            ),
        )

    def _process_existing_candidates(
        self,
        snapshot: EngineSnapshot,
        bar: Bar,
    ) -> None:
        for candidate in tuple(self._open.values()):
            if (bar.symbol, bar.instrument_id) != (
                candidate.symbol,
                candidate.instrument_id,
            ):
                self._resolve(
                    candidate,
                    bar.end,
                    "contract_boundary",
                    censored=True,
                )
                continue
            if bar.end <= candidate.observed_at:
                continue
            if (
                candidate.deadline_at is not None
                and candidate.deadline_at <= bar.start
            ):
                self._resolve(
                    candidate,
                    self._deadline_terminal_at(candidate),
                    (
                        "deadline_no_delivery"
                        if candidate.entry_price is not None
                        else "entry_unfilled_deadline"
                    ),
                    censored=False,
                )
                continue
            if (
                candidate.deadline_at is not None
                and bar.start < candidate.deadline_at < bar.end
            ):
                self._resolve(
                    candidate,
                    self._deadline_terminal_at(candidate),
                    "deadline_inside_completed_bar_censored",
                    censored=True,
                )
        precounted_open_entries: set[str] = set()
        if not bar.synthetic_no_trade:
            for candidate in tuple(self._open.values()):
                if (
                    bar.end > candidate.observed_at
                    and candidate.entry_price is None
                    and candidate.entry_reference_price is None
                ):
                    candidate.elapsed_real_1m_bars += 1
                    candidate.entry_price = float(bar.open)
                    candidate.entry_at = bar.start
                    if not self._valid_geometry(
                        candidate.direction,
                        candidate.entry_price,
                        candidate.invalidation_price,
                        candidate.target_price,
                    ):
                        self._resolve(
                            candidate,
                            bar.end,
                            "activation_geometry_invalid",
                            censored=True,
                        )
                        continue
                    self._fill_counts["filled"] += 1
                    precounted_open_entries.add(candidate.candidate_id)
        # Current snapshot invalidation is known before accepting a target
        # inferred from the same completed bar.
        self._update_current_evidence(snapshot)
        for candidate in tuple(self._open.values()):
            if bar.end <= candidate.observed_at:
                continue
            if bar.synthetic_no_trade:
                if (
                    candidate.deadline_at is not None
                    and candidate.deadline_at <= bar.end
                ):
                    self._resolve(
                        candidate,
                        self._deadline_terminal_at(candidate),
                        (
                            "deadline_no_delivery"
                            if candidate.entry_price is not None
                            else "entry_unfilled_deadline"
                        ),
                        censored=False,
                    )
                continue
            self._advance_real_bar(
                candidate,
                bar,
                count_bar=(
                    candidate.candidate_id
                    not in precounted_open_entries
                ),
            )

    def _advance_real_bar(
        self,
        candidate: _OpenCandidate,
        bar: Bar,
        *,
        count_bar: bool = True,
    ) -> None:
        if count_bar:
            candidate.elapsed_real_1m_bars += 1
        invalidated = (
            bar.low <= candidate.invalidation_price
            if candidate.direction is Direction.LONG
            else bar.high >= candidate.invalidation_price
        )
        targeted = (
            bar.high >= candidate.target_price
            if candidate.direction is Direction.LONG
            else bar.low <= candidate.target_price
        )
        filled_at_unknown_touch = False
        if candidate.entry_price is None:
            if candidate.entry_reference_price is None:
                candidate.entry_price = float(bar.open)
                candidate.entry_at = bar.start
            elif (
                bar.low <= candidate.entry_reference_price <= bar.high
            ):
                candidate.entry_price = float(candidate.entry_reference_price)
                candidate.entry_at = bar.end
                filled_at_unknown_touch = True
            else:
                if invalidated:
                    candidate.structural_thesis_invalidated = True
                    self._resolve(
                        candidate,
                        bar.end,
                        "invalidation_before_entry",
                        censored=True,
                    )
                    return
                if targeted:
                    self._resolve(
                        candidate,
                        bar.end,
                        "draw_consumed_before_entry",
                        censored=True,
                    )
                    return
            if candidate.entry_price is not None:
                if not self._valid_geometry(
                    candidate.direction,
                    candidate.entry_price,
                    candidate.invalidation_price,
                    candidate.target_price,
                ):
                    self._resolve(
                        candidate,
                        bar.end,
                        "activation_geometry_invalid",
                        censored=True,
                    )
                    return
                self._fill_counts["filled"] += 1
            else:
                self._resolve_deadline_if_due(candidate, bar)
                return
        if candidate.entry_price is not None:
            if invalidated:
                candidate.same_bar_collision = bool(targeted)
                candidate.invalidation_before_target = True
                candidate.target_before_invalidation = False
                candidate.structural_thesis_invalidated = True
                candidate.mae_points = max(
                    candidate.mae_points,
                    candidate.initial_risk,
                )
                self._resolve(
                    candidate,
                    bar.end,
                    (
                        "same_bar_invalidation_priority"
                        if targeted
                        else "invalidation_first"
                    ),
                    censored=False,
                )
                return
            if filled_at_unknown_touch:
                # Entry/target order inside the completed bar is unknowable.
                # A simultaneous target is censored; otherwise keep the fill
                # but admit no excursion until a later completed bar.
                if targeted:
                    self._resolve(
                        candidate,
                        bar.end,
                        "entry_target_same_bar_order_unknown",
                        censored=True,
                    )
                    return
                self._resolve_deadline_if_due(candidate, bar)
                return
            if targeted:
                candidate.target_before_invalidation = True
                candidate.invalidation_before_target = False
                candidate.time_to_draw_real_bars = (
                    candidate.elapsed_real_1m_bars
                )
                candidate.mfe_points = max(
                    candidate.mfe_points,
                    abs(candidate.target_price - candidate.entry_price),
                )
                self._resolve(
                    candidate,
                    bar.end,
                    "target_first",
                    censored=False,
                )
                return
            # A non-terminal bar may safely contribute its completed envelope.
            self._update_excursions(candidate, bar)
        self._resolve_deadline_if_due(candidate, bar)

    def _resolve_deadline_if_due(
        self,
        candidate: _OpenCandidate,
        bar: Bar,
    ) -> None:
        if (
            candidate.candidate_id in self._open
            and candidate.deadline_at is not None
            and candidate.deadline_at <= bar.end
        ):
            self._resolve(
                candidate,
                self._deadline_terminal_at(candidate),
                (
                    "deadline_no_delivery"
                    if candidate.entry_price is not None
                    else "entry_unfilled_deadline"
                ),
                censored=False,
            )
            return
        if (
            candidate.deadline_real_1m_bars is not None
            and candidate.elapsed_real_1m_bars
            >= candidate.deadline_real_1m_bars
        ):
            self._resolve(
                candidate,
                bar.end,
                (
                    "deadline_no_delivery"
                    if candidate.entry_price is not None
                    else "entry_unfilled_deadline"
                ),
                censored=False,
            )

    @staticmethod
    def _deadline_terminal_at(candidate: _OpenCandidate) -> pd.Timestamp:
        """Return the causal terminal clock while preserving frozen deadline."""

        if candidate.deadline_at is None:
            raise ValueError("deadline terminal clock requires a deadline")
        return max(candidate.deadline_at, candidate.observed_at)

    @staticmethod
    def _update_excursions(candidate: _OpenCandidate, bar: Bar) -> None:
        entry = float(candidate.entry_price)
        if candidate.direction is Direction.LONG:
            favorable = max(0.0, float(bar.high) - entry)
            adverse = max(0.0, entry - float(bar.low))
        else:
            favorable = max(0.0, entry - float(bar.low))
            adverse = max(0.0, float(bar.high) - entry)
        candidate.mfe_points = max(candidate.mfe_points, favorable)
        candidate.mae_points = max(candidate.mae_points, adverse)
        if (
            candidate.entry_zone_lower is not None
            and candidate.entry_zone_upper is not None
            and candidate.zone_departed_at is None
            and not (
                candidate.entry_zone_lower
                <= float(bar.close)
                <= candidate.entry_zone_upper
            )
        ):
            candidate.zone_departed_at = bar.end
            candidate.zone_departure_kind = (
                "favorable"
                if (
                    candidate.direction is Direction.LONG
                    and bar.close > candidate.entry_zone_upper
                )
                or (
                    candidate.direction is Direction.SHORT
                    and bar.close < candidate.entry_zone_lower
                )
                else "adverse"
            )

    def _update_current_evidence(self, snapshot: EngineSnapshot) -> None:
        context = getattr(snapshot.belief, "global_context", None)
        invalidated = (
            set()
            if context is None
            else set(context.invalidated_source_ids)
        )
        for candidate in tuple(self._open.values()):
            if (
                not candidate.structural_thesis_invalidated
                and invalidated.intersection(candidate.source_ids)
            ):
                candidate.structural_thesis_invalidated = True
                candidate.first_changed_evidence_id = (
                    "source_identity_invalidated"
                )
                candidate.first_changed_evidence_at = snapshot.observation.asof
                candidate.invalidation_before_target = True
                candidate.target_before_invalidation = False
                self._resolve(
                    candidate,
                    snapshot.observation.asof,
                    "source_identity_invalidated",
                    censored=False,
                )
                continue
            if candidate.first_changed_evidence_id is not None:
                continue
            for frozen in candidate.playbook_diagnostics:
                identity = frozen.get("hypothesis_key")
                current = snapshot.belief.resolve_hypothesis(identity)
                if current is None:
                    continue
                if getattr(current, "record_kind", None) == "retained_episode":
                    # A bounded Scene-Graph root may be absent while its
                    # frozen causal episode remains unresolved.  The Brain
                    # deliberately clears every current-action gate on that
                    # lifecycle-only projection; this is an authority
                    # boundary, not newly observed adverse evidence.  Actual
                    # source invalidation is handled above and future price
                    # outcomes remain owned by the frozen Shadow candidate.
                    continue
                frozen_identity = (
                    frozen.get("episode_id"),
                    frozen.get("setup_id"),
                    frozen.get("bound_market_thesis_id"),
                )
                current_identity = (
                    getattr(current, "episode_id", None),
                    getattr(current, "setup_context_id", None),
                    getattr(current, "bound_market_thesis_id", None),
                )
                if frozen_identity != current_identity:
                    # A rearmed episode is not later evidence for this freeze.
                    continue
                prior_gates = frozen.get("hard_gates", {})
                changed = next(
                    (
                        name
                        for name, value in prior_gates.items()
                        if current.hard_gate_results.get(name) is not value
                    ),
                    None,
                )
                if changed is not None:
                    candidate.first_changed_evidence_id = (
                        f"{frozen['playbook']}:{changed}"
                    )
                    candidate.first_changed_evidence_at = (
                        snapshot.observation.asof
                    )
                    break

    def _emit_incomplete(
        self,
        snapshot: EngineSnapshot,
        spec: _CandidateSpec,
        candidate_id: str,
        diagnostics: tuple[dict[str, Any], ...],
        entry_reference: float | None,
        invalidation: float | None,
        invalidation_id: str | None,
        target: float | None,
        draw_id: str | None,
        geometry_incomplete_reason: str,
    ) -> None:
        top = self._top_directional_hypothesis(snapshot, spec.direction)
        candidate = _OpenCandidate(
            candidate_id=candidate_id,
            event_kind=spec.event_kind,
            event_id=spec.event_id,
            observed_at=spec.observed_at,
            symbol=snapshot.observation.symbol,
            instrument_id=snapshot.observation.instrument_id,
            direction=spec.direction,
            source_timeframe=spec.source_timeframe,
            source_ids=spec.source_ids,
            candidate_origin=spec.candidate_origin,
            source_playbook=spec.source_playbook,
            source_episode_id=spec.source_episode_id,
            source_setup_id=spec.source_setup_id,
            source_context_thesis_id=spec.source_context_thesis_id,
            entry_location_id=spec.entry_location_id,
            entry_path_id=spec.entry_path_id,
            lsr_displacement_id=spec.lsr_displacement_id,
            lsr_entry_zone_id=spec.lsr_entry_zone_id,
            entry_episode_binding_status=(
                spec.entry_episode_binding_status
            ),
            entry_episode_terminal_at=spec.entry_episode_terminal_at,
            entry_episode_terminal_reason=(
                spec.entry_episode_terminal_reason
            ),
            selected_trigger_kind=spec.selected_trigger_kind,
            selected_trigger_id=spec.selected_trigger_id,
            selected_trigger_at=spec.selected_trigger_at,
            available_trigger_kinds=spec.available_trigger_kinds,
            decision_price=float(snapshot.observation.price),
            entry_rule=(
                SHADOW_OUTCOME_PROTOCOL["playbook_entry_rule"]
                if spec.event_kind == "playbook_executable"
                else (
                    SHADOW_OUTCOME_PROTOCOL["zone_entry_rule"]
                    if entry_reference is not None
                    else SHADOW_OUTCOME_PROTOCOL["non_zone_entry_rule"]
                )
            ),
            entry_reference_price=entry_reference,
            entry_zone_lower=spec.entry_zone_lower,
            entry_zone_upper=spec.entry_zone_upper,
            invalidation_price=float(invalidation or 0.0),
            invalidation_source_id=str(invalidation_id or ""),
            draw_id=str(draw_id or ""),
            target_price=float(target or 0.0),
            deadline_at=spec.deadline_at,
            deadline_real_1m_bars=(
                None
                if spec.event_kind == "playbook_executable"
                else _HORIZON_REAL_BARS
            ),
            playbook_diagnostics=diagnostics,
            decision_action=_enum_value(snapshot.decision.selected_action),
            risk_action=_enum_value(snapshot.risk.final_action),
            brain_phase=None if top is None else _enum_value(top.phase),
            thesis_strength=None if top is None else _finite(top.thesis_strength),
            sequence_progress=None if top is None else _finite(top.sequence_progress),
            location_quality=None if top is None else _finite(top.location_quality),
            entry_readiness=None if top is None else _finite(top.entry_readiness),
            delivery_quality=None if top is None else _finite(top.delivery_quality),
            uncertainty=None if top is None else _finite(top.uncertainty),
            geometry_incomplete_reason=geometry_incomplete_reason,
        )
        self._resolve(
            candidate,
            snapshot.observation.asof,
            "geometry_incomplete",
            censored=False,
            geometry_complete=False,
        )

    def _resolve(
        self,
        candidate: _OpenCandidate,
        resolved_at: pd.Timestamp,
        resolution: str,
        *,
        censored: bool,
        geometry_complete: bool = True,
    ) -> None:
        resolved_at = aware_timestamp(
            resolved_at,
            name="shadow_outcome.resolved_at",
        )
        if resolved_at < candidate.observed_at:
            raise ValueError(
                "shadow candidate resolution cannot precede observation"
            )
        self._open.pop(candidate.candidate_id, None)
        filled = candidate.entry_price is not None
        if not filled:
            self._fill_counts["unfilled"] += 1
        risk = candidate.initial_risk
        mfe_r = candidate.mfe_points / risk if risk > 0 else None
        mae_r = candidate.mae_points / risk if risk > 0 else None
        path_valid = candidate.target_before_invalidation is True
        expired = _is_expired_resolution(resolution)
        quadrant_eligible = bool(
            geometry_complete and filled and not censored and not expired
        )
        outcomes: list[dict[str, Any]] = []
        for diagnostic in candidate.playbook_diagnostics:
            accepted = bool(diagnostic["accepted"])
            quadrant = None
            is_source_episode = bool(
                candidate.event_kind == "playbook_executable"
                and candidate.source_playbook is not None
                and diagnostic["playbook"] == candidate.source_playbook.value
                and diagnostic.get("episode_id")
                == candidate.source_episode_id
            )
            if quadrant_eligible and is_source_episode:
                quadrant = (
                    ("accepted" if accepted else "rejected")
                    + "_path_"
                    + ("valid" if path_valid else "failed")
                )
                key = (diagnostic["playbook"], quadrant)
                self._quadrants[key] += 1
                case_key = f"{diagnostic['playbook']}:{quadrant}"
                if (
                    candidate.candidate_id not in self._case_ids
                    and len(self._case_ids) < 40
                ):
                    self._case_ids.append(candidate.candidate_id)
                if candidate.candidate_id in self._case_ids:
                    strata = self._case_strata.setdefault(
                        candidate.candidate_id,
                        [],
                    )
                    if case_key not in strata:
                        strata.append(case_key)
            outcomes.append(
                {
                    **{
                        key: value
                        for key, value in diagnostic.items()
                        if key != "hard_gates"
                    },
                    "quadrant": quadrant,
                }
            )
        record = ShadowCandidateOutcomeRecord(
            candidate_id=candidate.candidate_id,
            event_kind=candidate.event_kind,
            event_id=candidate.event_id,
            observed_at=candidate.observed_at,
            resolved_at=resolved_at,
            symbol=candidate.symbol,
            instrument_id=candidate.instrument_id,
            direction=candidate.direction.value,
            source_timeframe=candidate.source_timeframe.value,
            source_ids=_json(candidate.source_ids),
            candidate_origin=candidate.candidate_origin,
            source_playbook=(
                None
                if candidate.source_playbook is None
                else candidate.source_playbook.value
            ),
            source_episode_id=candidate.source_episode_id,
            source_setup_id=candidate.source_setup_id,
            source_context_thesis_id=candidate.source_context_thesis_id,
            entry_location_id=candidate.entry_location_id,
            entry_path_id=candidate.entry_path_id,
            lsr_displacement_id=candidate.lsr_displacement_id,
            lsr_entry_zone_id=candidate.lsr_entry_zone_id,
            entry_episode_binding_status=(
                candidate.entry_episode_binding_status
            ),
            entry_episode_terminal_at=candidate.entry_episode_terminal_at,
            entry_episode_terminal_reason=(
                candidate.entry_episode_terminal_reason
            ),
            selected_trigger_kind=candidate.selected_trigger_kind,
            selected_trigger_id=candidate.selected_trigger_id,
            selected_trigger_at=candidate.selected_trigger_at,
            available_trigger_kinds=_json(candidate.available_trigger_kinds),
            decision_price=candidate.decision_price,
            entry_rule=candidate.entry_rule,
            entry_reference_price=candidate.entry_reference_price,
            entry_price=candidate.entry_price,
            entry_at=candidate.entry_at,
            entry_zone_lower=candidate.entry_zone_lower,
            entry_zone_upper=candidate.entry_zone_upper,
            invalidation_price=(
                candidate.invalidation_price
                if candidate.invalidation_source_id
                else None
            ),
            invalidation_source_id=(
                candidate.invalidation_source_id or None
            ),
            draw_id=candidate.draw_id or None,
            target_price=(candidate.target_price if candidate.draw_id else None),
            deadline_at=candidate.deadline_at,
            deadline_real_1m_bars=candidate.deadline_real_1m_bars,
            elapsed_real_1m_bars=candidate.elapsed_real_1m_bars,
            geometry_complete=geometry_complete,
            geometry_incomplete_reason=(
                None
                if geometry_complete
                else candidate.geometry_incomplete_reason
            ),
            filled=filled,
            resolution=resolution,
            censored=censored,
            target_before_invalidation=(
                candidate.target_before_invalidation if filled else None
            ),
            invalidation_before_target=(
                candidate.invalidation_before_target if filled else None
            ),
            same_bar_collision=candidate.same_bar_collision,
            mfe_points=(candidate.mfe_points if filled else None),
            mae_points=(candidate.mae_points if filled else None),
            mfe_R=mfe_r,
            mae_R=mae_r,
            hit_0_5R=None if mfe_r is None else mfe_r >= 0.5,
            hit_1R=None if mfe_r is None else mfe_r >= 1.0,
            hit_2R=None if mfe_r is None else mfe_r >= 2.0,
            time_to_draw_real_bars=candidate.time_to_draw_real_bars,
            wait_improvement_points=(
                None
                if candidate.entry_price is None
                else candidate.direction.sign
                * (candidate.decision_price - candidate.entry_price)
            ),
            zone_departed_before_terminal=(
                None
                if candidate.entry_zone_lower is None
                else (
                    candidate.zone_departed_at is not None
                    and candidate.zone_departed_at < resolved_at
                )
            ),
            zone_departure_kind=candidate.zone_departure_kind,
            structural_thesis_invalidated=(
                candidate.structural_thesis_invalidated
            ),
            first_changed_evidence_id=candidate.first_changed_evidence_id,
            first_changed_evidence_at=candidate.first_changed_evidence_at,
            brain_phase=candidate.brain_phase,
            thesis_strength=candidate.thesis_strength,
            sequence_progress=candidate.sequence_progress,
            location_quality=candidate.location_quality,
            entry_readiness=candidate.entry_readiness,
            delivery_quality=candidate.delivery_quality,
            uncertainty=candidate.uncertainty,
            decision_action=candidate.decision_action,
            risk_action=candidate.risk_action,
            playbook_outcomes=_json(outcomes),
        )
        self._rows.append(record)
        self._resolutions[resolution] += 1

    def _censor_all(self, asof: pd.Timestamp, reason: str) -> None:
        for candidate in tuple(self._open.values()):
            self._resolve(candidate, asof, reason, censored=True)


__all__ = [
    "RECORDER_SCHEMA_VERSION",
    "SHADOW_DERIVED_SCHEMA_VERSION",
    "SHADOW_EPISODE_OUTCOME_FIELD_TYPES",
    "SHADOW_MECHANISM_CHALLENGE_FIELD_TYPES",
    "SHADOW_MECHANISM_MOTIF_FIELD_TYPES",
    "SHADOW_OUTCOME_FIELD_TYPES",
    "SHADOW_OUTCOME_PROTOCOL",
    "SHADOW_MOTIF_ROOT_SAMPLE_LIMIT",
    "SHADOW_ROOT_EPISODE_FIELD_TYPES",
    "SHADOW_ROOT_SEQUENCE_FIELD_TYPES",
    "SHADOW_SEQUENCE_PLAYBOOKS",
    "ShadowCandidateOutcomeRecord",
    "ShadowCandidateOutcomeRecorder",
    "ShadowEpisodeOutcomeRecord",
    "ShadowMechanismChallengeRecord",
    "ShadowMechanismMotifRecord",
    "ShadowRootEpisodeRecord",
    "ShadowRootSequenceRecord",
    "aggregate_shadow_mechanism_motifs",
    "derive_shadow_episode_outcomes",
    "derive_shadow_mechanism_challenges",
    "derive_shadow_root_episode_records",
    "derive_shadow_root_sequence_records",
]
