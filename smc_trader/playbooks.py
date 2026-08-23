"""Small preregistered playbook set with continuous belief/state updates."""
from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Collection, Mapping, Sequence

import pandas as pd

from .calibration import TypedBrainCalibrator
from .dol_ranking import (
    DOLCandidateFact,
    DOLDirection,
    DOLObstructionFact,
    DOLObstructionViewFact,
    DOLRankingProtocol,
    DOLRankingResult,
    load_dol_ranking_protocol,
    rank_dol_candidates,
)
from .dol_probability import (
    DOLProbabilityModelArtifact,
    DOLProbabilityProtocol,
    DOLProbabilityResult,
    load_dol_probability_protocol,
    marginalize_dol_probabilities,
)
from .market_clock import MARKET_TIMEZONE, special_session_close
from .model import (
    AccountState,
    BOSLifecycle,
    BOSPostBreakState,
    BOSScope,
    DealingRangeState,
    DealingRangeLifecycle,
    DeliveryObstruction,
    Direction,
    DrawSelection,
    EntryEpisodeState,
    EntryLocationLifecycle,
    EntryLocationState,
    Evidence,
    EventKind,
    EventOrigin,
    FVGQualification,
    FairValueGapLifecycle,
    FrozenLSRContext,
    FrozenTriggerState,
    FrozenRangeAuctionContext,
    GlobalConflictEvidence,
    GlobalConflictRole,
    GlobalMarketContext,
    HypothesisConflictRelation,
    HypothesisBelief,
    HypothesisSequenceState,
    LiquidityInventoryLifecycle,
    LiquidityInventoryItem,
    LiquidityLevel,
    LiquidityRoute,
    MarketBelief,
    MarketEvent,
    MarketObservation,
    ManipulationLifecycle,
    ManipulationState,
    ScaleRelation,
    MicroBOSReference,
    OpenMarketThesis,
    ContextThesisState,
    OrderBlockLifecycle,
    PathSequenceLifecycle,
    PathSequenceStep,
    PathSequenceState,
    PlanFeasibility,
    Playbook,
    PlaybookPhase,
    PositionSnapshot,
    QualifiedReacceptanceLifecycle,
    SMC_SEMANTIC_VERSION,
    SequenceStepState,
    StructuralLevel,
    StructureLifecycle,
    Timeframe,
    TradePlan,
    aware_timestamp,
    clamp,
    to_primitive,
)
from .path_belief import (
    HypothesisManager,
    PathBeliefProtocol,
    PathBeliefUpdateRecord,
    PathCompetitionSetState,
    PathKind,
    PathOutcomeEvent,
    PathStatus,
    PathTerminalEvent,
    create_path_competition_set,
    load_path_belief_protocol,
)
from .signal_policy import (
    AdmittedDOLCalibrationArtifact,
    AdmittedPathLikelihoodArtifact,
    SignalArtifactPins,
    SignalAssessment,
    SignalEvaluationContext,
    SignalPolicyProtocol,
    TargetBeforeInvalidationArtifact,
    assess_signal,
    load_signal_policy_protocol,
)
from .trade_intent import (
    EntryMethod,
    TradeIntent,
    TradeIntentError,
    build_trade_intent,
)
from .playbook_registry import (
    PlaybookProtocol,
    PlaybookRegistry,
    load_playbook_registry,
)
from .scene_graph import (
    EvidenceStatus,
    SceneEdgeKind,
    SceneGraphDelta,
    TemporalMarketSceneGraph,
    build_hypothesis_states,
    build_open_market_theses,
    select_focus,
    update_global_market_context,
)


# One Brain update can evaluate many root-specific LSR candidates that ask the
# same current-graph connectivity question.  This cache is deliberately scoped
# to the dynamic ``PlaybookBrain.update`` call: it is never checkpointed and is
# reset even when evaluation raises.  Revision/asof remain in the key so a
# nested or otherwise unusual update cannot reuse an answer from another graph
# clock.
_LSR_CONNECTION_MEMO: ContextVar[
    dict[tuple[object, ...], bool] | None
] = ContextVar("lsr_connection_memo", default=None)


@dataclass(frozen=True)
class BrainConfig:
    tick_size: float = 0.25
    minimum_remaining_path_R: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.tick_size))
            or self.tick_size <= 0.0
            or not math.isfinite(float(self.minimum_remaining_path_R))
            or self.minimum_remaining_path_R <= 0.0
        ):
            raise ValueError(
                "brain tick size and minimum remaining path must be positive"
            )


def _market_lifecycle_deadline(asof: pd.Timestamp) -> pd.Timestamp:
    """Return the registered session close without consulting execution data."""

    local = pd.Timestamp(asof).tz_convert(MARKET_TIMEZONE)
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


@dataclass(frozen=True)
class _Evaluation:
    evidence: tuple[Evidence, ...]
    trigger_ready: bool
    setup_clock: pd.Timestamp | None
    sequence_signals: Mapping[str, "_SequenceSignal"]
    setup_identity: str | None = None
    context_identity: str | None = None
    episode_identity: str | None = None
    initiating_event_id: str | None = None
    entry_location_id: str | None = None
    entry_path_id: str | None = None
    plan: TradePlan | None = None
    invalidation: StructuralLevel | None = None
    selected_draw: LiquidityLevel | None = None
    thesis_draw: LiquidityLevel | None = None
    draw_selection: DrawSelection | None = None
    liquidity_route: LiquidityRoute | None = None
    thesis_target: float = 0.0
    location_quality: float = 0.0
    entry_readiness: float = 0.0
    delivery_quality: float = 0.0
    typed_uncertainty: float = 1.0
    uncertainty_conflict: float = 0.0
    uncertainty_required_evidence_missing: float = 0.0
    uncertainty_authority_missing: float = 1.0
    uncertainty_graph_ambiguity: float = 0.0
    evidence_group_scores: Mapping[str, float] = field(
        default_factory=dict
    )
    hard_gate_results: Mapping[str, bool] = field(
        default_factory=dict
    )
    invalidated: bool = False
    entry_window_expired: bool = False
    terminal_reason: str | None = None
    terminal_source_ids: tuple[str, ...] = ()
    authority_tier: str | None = None
    authority_source_timeframe: str | None = None
    authority_structural_rank: str | None = None
    global_context_connected: bool = False
    authority_latched: bool = False
    source_nested: bool = False
    competing_episode_ids: tuple[str, ...] = ()
    material_authority_opposition: bool = False
    authority_relation: HypothesisConflictRelation = (
        HypothesisConflictRelation.UNRELATED
    )
    authority_rank_gap: int | None = None
    conflict_role: GlobalConflictRole | None = None
    conflict_scope: str | None = None
    acceptance_state: str | None = None
    obstruction_distance_R: float | None = None
    free_path_R: float | None = None
    soft_obstruction_count: int = 0
    hard_barrier_before_target: bool = False
    path_blocker_ids: tuple[str, ...] = ()
    ambiguity_count: int = 0
    market_thesis_ids: tuple[str, ...] = ()
    market_thesis_id: str | None = None
    bound_market_thesis_id: str | None = None
    market_thesis_root_id: str | None = None
    market_thesis_mechanism: str | None = None
    market_thesis_authority_relation: str | None = None
    playbook_match_strength: float = 0.0
    market_thesis_binding_required: bool = False
    market_thesis_action_bound: bool = False
    market_thesis_match_status: str = "not_required"
    selected_trigger: FrozenTriggerState | None = None
    target_selection_failure_reason: str | None = None
    lsr_displacement_id: str | None = None
    lsr_entry_zone_id: str | None = None
    lsr_pool_path_id: str | None = None

    def __post_init__(self) -> None:
        components = (
            self.uncertainty_conflict,
            self.uncertainty_required_evidence_missing,
            self.uncertainty_authority_missing,
            self.uncertainty_graph_ambiguity,
        )
        if (
            any(
                not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in (*components, self.typed_uncertainty)
            )
            or not math.isclose(
                float(self.typed_uncertainty),
                _uncertainty_total(
                    conflict=float(components[0]),
                    required_evidence_missing=float(components[1]),
                    authority_missing=float(components[2]),
                    graph_ambiguity=float(components[3]),
                ),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "evaluation uncertainty total disagrees with its components"
            )
        if type(self.ambiguity_count) is not int or self.ambiguity_count < 0:
            raise ValueError("evaluation ambiguity count is invalid")
        thesis_identity_fields = (
            self.market_thesis_id,
            self.bound_market_thesis_id,
            self.market_thesis_root_id,
            self.market_thesis_mechanism,
            self.market_thesis_authority_relation,
        )
        valid_match_statuses = {
            "not_required",
            "no_open_thesis",
            "no_direction_match",
            "no_mechanism_match",
            "root_identity_unbound",
            "exact_root_bound",
        }
        if (
            not math.isfinite(float(self.playbook_match_strength))
            or not 0.0 <= float(self.playbook_match_strength) <= 1.0
            or type(self.market_thesis_binding_required) is not bool
            or type(self.market_thesis_action_bound) is not bool
            or self.market_thesis_match_status not in valid_match_statuses
            or len(self.market_thesis_ids)
            != len(set(self.market_thesis_ids))
            or any(
                not isinstance(identity, str) or not identity
                for identity in self.market_thesis_ids
            )
            or any(
                value is not None
                and (not isinstance(value, str) or not value)
                for value in thesis_identity_fields
            )
            or (
                self.market_thesis_id is None
                and any(value is not None for value in thesis_identity_fields[1:])
            )
            or (
                self.market_thesis_id is not None
                and any(
                    value is None
                    for value in (
                        self.market_thesis_root_id,
                        self.market_thesis_mechanism,
                        self.market_thesis_authority_relation,
                    )
                )
            )
            or (
                self.market_thesis_id is not None
                and (
                    not self.market_thesis_ids
                    or self.market_thesis_id != self.market_thesis_ids[0]
                )
            )
            or bool(self.market_thesis_ids)
            != (self.market_thesis_id is not None)
            or (
                self.bound_market_thesis_id is not None
                and (
                    self.bound_market_thesis_id
                    not in self.market_thesis_ids
                    or self.bound_market_thesis_id
                    != self.market_thesis_id
                )
            )
            or self.market_thesis_action_bound
            != (self.bound_market_thesis_id is not None)
            or self.market_thesis_binding_required
            != (self.market_thesis_match_status != "not_required")
            or self.market_thesis_action_bound
            != (self.market_thesis_match_status == "exact_root_bound")
            or (self.market_thesis_id is not None)
            != (
                self.market_thesis_match_status
                in {"root_identity_unbound", "exact_root_bound"}
            )
            or (
                self.market_thesis_id is None
                and self.playbook_match_strength != 0.0
            )
        ):
            raise ValueError("evaluation market-thesis match is invalid")
        if (
            len(self.path_blocker_ids) != len(set(self.path_blocker_ids))
            or any(
                not isinstance(identity, str) or not identity
                for identity in self.path_blocker_ids
            )
            or self.hard_barrier_before_target
            != bool(self.path_blocker_ids)
        ):
            raise ValueError("evaluation blocker identities are invalid")
        if self.target_selection_failure_reason not in {
            None,
            "visible_draw_missing",
            "primary_structural_draw_missing",
            "all_draws_at_or_beyond_context_draw",
            "authority_barrier_missing",
            "all_draws_beyond_barrier",
            "all_targets_below_minimum_R",
            "remaining_path_below_minimum",
            "context_draw_consumed",
            "frozen_primary_target_unavailable",
            "root_temporarily_absent",
        }:
            raise ValueError("evaluation target-selection failure is invalid")

    @property
    def selected_trigger_kind(self) -> str | None:
        return (
            None
            if self.selected_trigger is None
            else self.selected_trigger.trigger_kind
        )

    @property
    def selected_trigger_id(self) -> str | None:
        return (
            None
            if self.selected_trigger is None
            else self.selected_trigger.trigger_id
        )

    @property
    def selected_trigger_at(self) -> pd.Timestamp | None:
        return (
            None
            if self.selected_trigger is None
            else self.selected_trigger.observed_at
        )

    @property
    def available_trigger_kinds(self) -> tuple[str, ...]:
        return (
            ()
            if self.selected_trigger is None
            else self.selected_trigger.available_trigger_kinds
        )


@dataclass(frozen=True)
class _SequenceSignal:
    value: float
    observed_at: pd.Timestamp | None
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", clamp(self.value))


@dataclass(frozen=True)
class _CountertrendTargetSelection:
    """One LSR delivery choice and its frozen authority cap."""

    target: LiquidityLevel | None
    context_draw: LiquidityLevel | None
    failure_reason: str | None
    authority_barrier_id: str | None
    authority_barrier_price: float | None

    def __post_init__(self) -> None:
        if (self.authority_barrier_id is None) != (
            self.authority_barrier_price is None
        ):
            raise ValueError("countertrend authority barrier is incomplete")
        if self.authority_barrier_price is not None and (
            not math.isfinite(float(self.authority_barrier_price))
            or float(self.authority_barrier_price) <= 0.0
        ):
            raise ValueError("countertrend authority barrier price is invalid")
        if (self.target is None) != (self.failure_reason is not None):
            raise ValueError("countertrend target result is inconsistent")
        if (self.target is None) != (self.context_draw is None):
            raise ValueError("countertrend target/context roles are inconsistent")


@dataclass(frozen=True)
class _DFPTargetSelection:
    """One executable DFP delivery target below its terminal H4 draw."""

    target: LiquidityLevel | None
    failure_reason: str | None
    authority_barrier_id: str | None
    authority_barrier_price: float | None

    def __post_init__(self) -> None:
        if (self.target is None) != (self.failure_reason is not None):
            raise ValueError("DFP target selection result is inconsistent")
        if (self.authority_barrier_id is None) != (
            self.authority_barrier_price is None
        ):
            raise ValueError("DFP authority barrier is incomplete")
        if self.authority_barrier_price is not None and (
            not math.isfinite(float(self.authority_barrier_price))
            or float(self.authority_barrier_price) <= 0.0
        ):
            raise ValueError("DFP authority barrier price is invalid")


def _select_frozen_trigger(
    valid_triggers: Sequence[PathSequenceStep],
    *,
    prior: HypothesisBelief | None,
    setup_id: str | None,
    entry_path_id: str | None,
    entry_location_id: str | None,
    direction: Direction,
    trigger_must_follow: pd.Timestamp | None = None,
    require_strictly_after: bool = False,
) -> tuple[FrozenTriggerState | None, PathSequenceStep | None]:
    """Latch the first qualified trigger for one exact setup episode."""

    ordered = tuple(
        sorted(
            (
                step
                for step in valid_triggers
                if (
                    not require_strictly_after
                    or (
                        trigger_must_follow is not None
                        and step.observed_at > trigger_must_follow
                    )
                )
            ),
            key=lambda step: (
                step.observed_at,
                step.step_id,
            ),
        )
    )
    strongest_current = (
        None
        if not ordered
        else max(
            ordered,
            key=lambda step: (
                step.strength,
                -step.observed_at.value,
            ),
        )
    )
    prior_trigger = None if prior is None else prior.selected_trigger
    if (
        prior_trigger is not None
        and setup_id is not None
        and prior_trigger.setup_id == setup_id
        and prior_trigger.entry_path_id == entry_path_id
        and prior_trigger.entry_location_id == entry_location_id
        and prior_trigger.direction is direction
        and (
            not require_strictly_after
            or (
                trigger_must_follow is not None
                and prior_trigger.observed_at > trigger_must_follow
            )
        )
    ):
        kinds = tuple(
            dict.fromkeys(
                (
                    *prior_trigger.available_trigger_kinds,
                    *(step.kind for step in ordered),
                )
            )
        )
        return replace(
            prior_trigger,
            available_trigger_kinds=kinds,
        ), strongest_current
    if (
        not ordered
        or setup_id is None
        or entry_path_id is None
        or entry_location_id is None
    ):
        return None, None
    selected = ordered[0]
    return FrozenTriggerState(
        trigger_id=selected.step_id,
        trigger_kind=selected.kind,
        observed_at=selected.observed_at,
        setup_id=setup_id,
        entry_path_id=entry_path_id,
        entry_location_id=entry_location_id,
        direction=direction,
        source_entity_id=selected.source_entity_id,
        source_event_id=selected.source_event_id,
        strength=selected.strength,
        available_trigger_kinds=tuple(
            dict.fromkeys(step.kind for step in ordered)
        ),
    ), strongest_current


def _retain_matching_prior_trigger(
    selected_trigger: FrozenTriggerState | None,
    *,
    prior: HypothesisBelief | None,
    setup_id: str | None,
    entry_location_id: str | None,
    entry_path_id: str | None,
    direction: Direction,
    trigger_must_follow: pd.Timestamp | None = None,
    require_strictly_after: bool = False,
) -> FrozenTriggerState | None:
    """Keep a frozen trigger only for the exact effective entry-path episode.

    A setup identity can remain stable while Group5 moves to a different
    entry location or path.  In that case the old trigger is episode history,
    not evidence for the new path.  Validate both the evaluator result and
    the prior fallback because a latched evaluator trigger can also outlive a
    change in the effective sequence or plan identity.
    """

    prior_trigger = None if prior is None else prior.selected_trigger
    for trigger in (selected_trigger, prior_trigger):
        if (
            trigger is not None
            and setup_id is not None
            and entry_location_id is not None
            and entry_path_id is not None
            and trigger.setup_id == setup_id
            and trigger.entry_location_id == entry_location_id
            and trigger.entry_path_id == entry_path_id
            and trigger.direction is direction
            and (
                not require_strictly_after
                or (
                    trigger_must_follow is not None
                    and trigger.observed_at > trigger_must_follow
                )
            )
        ):
            return trigger
    return None


@dataclass(frozen=True)
class _LSRRangeContext:
    """Optional mature-balance context around one core pool reversal."""

    dealing_range: DealingRangeState
    swept_boundary: LiquidityInventoryItem
    opposing_boundary: LiquidityInventoryItem | None


@dataclass(frozen=True)
class _LSRCandidate:
    """One exact pool-reversal path and its downstream authority reading."""

    path: PathSequenceState
    manipulation: ManipulationState
    tier: str
    eligible_root: bool
    connected_to_global: bool
    nested_source: bool
    source_inventory: LiquidityInventoryItem | None
    source_pool: object | None
    structural_rank: str
    stage: int
    authority_latched: bool = False


@dataclass(frozen=True)
class _LSRContextBinding:
    """Compact exact-root authority proof retained with one active Context."""

    thesis: OpenMarketThesis
    playbook_match_strength: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.playbook_match_strength) <= 1.0:
            raise ValueError("LSR Context binding strength is invalid")


@dataclass(frozen=True)
class _LSRExecutionOwner:
    """Frozen action provenance for the first fully executable child."""

    candidate_id: str
    plan: TradePlan
    trigger: FrozenTriggerState
    first_executable_at: pd.Timestamp

    def __post_init__(self) -> None:
        first_executable_at = aware_timestamp(
            self.first_executable_at,
            name="lsr_execution_owner.first_executable_at",
        )
        object.__setattr__(self, "first_executable_at", first_executable_at)
        if (
            not self.candidate_id
            or self.plan.playbook
            is not Playbook.LIQUIDITY_SWEEP_REVERSAL
            or self.plan.lsr_context is None
            or self.plan.setup_id != self.trigger.setup_id
            or self.plan.entry_location_id
            != self.trigger.entry_location_id
            or self.plan.entry_path_id != self.trigger.entry_path_id
            or self.plan.direction is not self.trigger.direction
            or first_executable_at < self.trigger.observed_at
        ):
            raise ValueError("LSR execution-owner provenance is invalid")


_LSR_TIER_RANK = {"A": 0, "B": 1, "C": 2, "ineligible": 3}
_LSR_STRUCTURAL_RANK = {
    "external": 0,
    "intermediate": 1,
    "internal": 2,
    "unknown": 3,
}
_LSR_CONTEXT_RELATIONS = frozenset(
    {
        SceneEdgeKind.ANCHORS.value,
        SceneEdgeKind.LOCATED_AT.value,
        SceneEdgeKind.SOURCED_FROM.value,
        SceneEdgeKind.PROMOTED_FROM.value,
        SceneEdgeKind.CONTAINED_BY.value,
        SceneEdgeKind.ALIGNS_WITH.value,
        SceneEdgeKind.SWEEPS.value,
    }
)


def _identity_tuple(*values: str | None) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(value for value in values if value is not None)
    )


def _frozen_invalidation_breached(
    direction: Direction,
    price: float,
    invalidation: StructuralLevel | None,
) -> bool:
    if invalidation is None:
        return False
    return (
        direction is Direction.LONG
        and price <= invalidation.price
    ) or (
        direction is Direction.SHORT
        and price >= invalidation.price
    )


def _entry_zone_beyond_frozen_invalidation(
    observation: MarketObservation,
    evaluation: "_Evaluation",
    direction: Direction,
    invalidation: StructuralLevel | None,
) -> bool:
    if evaluation.entry_location_id is None or invalidation is None:
        return False
    location = next(
        (
            item
            for item in observation.entry_locations
            if item.location_id == evaluation.entry_location_id
        ),
        None,
    )
    if location is None:
        return False
    planned_entry = (
        evaluation.plan.planned_entry
        if evaluation.plan is not None
        else location.contact_reference_price
        if location.contact_reference_price is not None
        else location.near_edge
    )
    return (
        direction is Direction.LONG
        and invalidation.price >= planned_entry
    ) or (
        direction is Direction.SHORT
        and invalidation.price <= planned_entry
    )


def _evidence(
    primitive: str,
    value: float,
    weight: float,
    supports: bool,
    observation: MarketObservation,
    explanation: str,
) -> Evidence:
    return Evidence(
        primitive=primitive,
        value=clamp(value),
        weight=float(weight),
        supports=bool(supports),
        observed_at=observation.asof,
        explanation=explanation,
    )

_TERMINAL_PHASES = {
    PlaybookPhase.COMPLETED,
    PlaybookPhase.INVALIDATED,
}
_EVIDENCE_GROUPS = (
    "structure",
    "displacement",
    "location",
    "liquidity",
    "trigger",
    "execution",
)


def _prior_step_sources(
    prior: HypothesisBelief | None,
    step_id: str,
) -> tuple[str, ...]:
    if (
        prior is None
        or prior.phase in _TERMINAL_PHASES
        or prior.sequence is None
    ):
        return ()
    return next(
        (
            step.source_ids
            for step in prior.sequence.steps
            if step.step_id == step_id and step.satisfied
        ),
        (),
    )


def _visible_level_map(
    observation: MarketObservation,
) -> dict[str, LiquidityLevel]:
    return {level.level_id: level for level in _visible_levels(observation)}


def _inventory_item_map(
    observation: MarketObservation,
) -> dict[str, object]:
    return {
        item.item_id: item
        for item in observation.liquidity_inventory
        if (
            item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
            and item.confirmed_at <= observation.asof
        )
    }


def _draw_rank(
    observation: MarketObservation,
    level: LiquidityLevel,
    entry: float,
) -> tuple[object, ...]:
    """Deterministic structural tier, then strength, then distance."""

    item = _inventory_item_map(observation).get(level.level_id)
    if item is None:
        return (
            9,
            0.0,
            abs(level.price - entry),
            level.confirmed_at,
            level.level_id,
        )
    higher_timeframe = item.timeframe in {Timeframe.H4, Timeframe.H1}
    pooled = item.kind in {"equal_highs", "equal_lows"}
    external = bool(
        item.structural_rank == "external"
        or item.is_protected_swing
        or higher_timeframe
    )
    reference = item.kind.startswith("previous_")
    tier = (
        0
        if item.is_protected_swing
        else 1
        if external and pooled
        else 2
        if reference
        else 3
        if external and item.kind == "swing"
        else 4
        if item.kind == "range_boundary"
        else 5
        if pooled
        else 6
    )
    return (
        tier,
        -float(item.visibility_strength),
        -float(item.strength),
        abs(level.price - entry),
        level.confirmed_at,
        level.level_id,
    )


def _draw_selection(
    observation: MarketObservation,
    target: LiquidityLevel | None,
    *,
    playbook: Playbook,
    prior: HypothesisBelief | None,
) -> DrawSelection | None:
    if target is None:
        return None
    item = _inventory_item_map(observation).get(target.level_id)
    if item is None:
        return None
    if (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.draw_selection is not None
        and prior.draw_selection.draw_id == item.item_id
        and prior.draw_selection.source_timeframe is item.timeframe
        and prior.draw_selection.source_kind == item.kind
        and prior.draw_selection.side == item.side
        and math.isclose(
            prior.draw_selection.price,
            target.price,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        return prior.draw_selection
    tier = _draw_rank(observation, target, target.price)[0]
    return DrawSelection(
        draw_id=item.item_id,
        selected_at=observation.asof,
        selection_reason=(
            f"{playbook.value}:primary_deliverable_target:"
            f"tier={tier}:"
            f"{item.timeframe.value}:{item.kind}:"
            "strength_then_distance"
        ),
        source_timeframe=item.timeframe,
        source_kind=item.kind,
        side=item.side,
        price=target.price,
        source_confirmed_at=item.confirmed_at,
        strength=item.strength,
    )


def _target_is_deliverable(
    level: LiquidityLevel,
    direction: Direction,
    entry: float,
    tick_size: float,
) -> bool:
    return bool(
        level.side == direction.opposing_liquidity_side
        and (
            (
                direction is Direction.LONG
                and level.price > entry + tick_size
            )
            or (
                direction is Direction.SHORT
                and level.price < entry - tick_size
            )
        )
    )


def _select_target(
    observation: MarketObservation,
    direction: Direction,
    entry: float,
    config: BrainConfig,
    *,
    preferred_id: str | None = None,
    required_timeframe: Timeframe | None = None,
    require_external: bool = False,
    require_preferred: bool = False,
    allowed_ids: frozenset[str] | None = None,
) -> LiquidityLevel | None:
    levels = _visible_level_map(observation)
    inventory = _inventory_item_map(observation)

    def eligible(level: LiquidityLevel) -> bool:
        item = inventory.get(level.level_id)
        return bool(
            (allowed_ids is None or level.level_id in allowed_ids)
            and
            (
                required_timeframe is None
                or level.timeframe is required_timeframe
            )
            and (
                not require_external
                or item is not None
                and (
                    item.structural_rank == "external"
                    or item.is_protected_swing
                    or item.timeframe in {Timeframe.H4, Timeframe.H1}
                )
            )
            and _target_is_deliverable(
                level,
                direction,
                entry,
                config.tick_size,
            )
        )
    if require_preferred and preferred_id is None:
        return None
    if preferred_id is not None:
        preferred = levels.get(preferred_id)
        if (
            preferred is not None
            and eligible(preferred)
        ):
            return preferred
        if require_preferred:
            return None
    candidates = [
        level
        for level in levels.values()
        if (
            eligible(level)
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda level: _draw_rank(observation, level, entry),
    )


def _select_primary_deliverable_target(
    observation: MarketObservation,
    direction: Direction,
    entry: float,
    config: BrainConfig,
    *,
    context_draw: LiquidityLevel | None,
    preferred_id: str | None = None,
    allowed_ids: frozenset[str] | None = None,
) -> LiquidityLevel | None:
    """Choose the nearest visible delivery before the directional draw."""

    levels = _visible_level_map(observation)
    if preferred_id is not None:
        preferred = levels.get(preferred_id)
        if (
            preferred is not None
            and (allowed_ids is None or preferred.level_id in allowed_ids)
            and _target_is_deliverable(
                preferred,
                direction,
                entry,
                config.tick_size,
            )
            and (
                context_draw is None
                or direction.sign
                * (preferred.price - context_draw.price)
                <= 0.0
            )
        ):
            return preferred
        return None
    candidates = [
        level
        for level in levels.values()
        if _target_is_deliverable(
            level,
            direction,
            entry,
            config.tick_size,
        )
        and (allowed_ids is None or level.level_id in allowed_ids)
        and (
            context_draw is None
            or direction.sign * (level.price - context_draw.price) <= 0.0
        )
    ]
    if not candidates:
        return context_draw
    return min(
        candidates,
        key=lambda level: (
            abs(level.price - entry),
            _draw_rank(observation, level, entry),
        ),
    )


def _higher_authority_opposes(
    global_context: GlobalMarketContext | None,
    direction: Direction,
) -> bool:
    return bool(
        global_context is not None
        and global_context.authority_timeframe
        in {Timeframe.H4, Timeframe.H1}
        and global_context.authority_direction is not None
        and global_context.authority_direction is not direction
    )


def _liquidity_contact_price(
    item: LiquidityInventoryItem,
    direction: Direction,
) -> float | None:
    """Return the first conservative contact with one frozen liquidity zone."""

    if item.side != direction.opposing_liquidity_side:
        return None
    return float(
        item.lower_bound
        if direction is Direction.LONG
        else item.upper_bound
    )


def _authority_barrier_distance(
    direction: Direction,
    entry: float,
    global_context: GlobalMarketContext,
    tick_size: float,
    *,
    excluded_ids: frozenset[str] = frozenset(),
) -> float | None:
    """Return the nearest forward hard barrier from the context view."""

    selected = _nearest_authority_barrier(
        direction,
        entry,
        global_context,
        tick_size,
        excluded_ids=excluded_ids,
    )
    return None if selected is None else selected[1]


def _nearest_authority_barrier(
    direction: Direction,
    entry: float,
    global_context: GlobalMarketContext,
    tick_size: float,
    *,
    excluded_ids: frozenset[str] = frozenset(),
) -> tuple[DeliveryObstruction, float] | None:
    """Resolve one nearest forward authority barrier and its distance."""

    view = global_context.obstruction_views.get(direction.value)
    if view is None:
        return None
    candidates: list[tuple[float, str, DeliveryObstruction]] = []
    for obstruction in view.hard_barriers:
        if (
            obstruction.obstruction_id in excluded_ids
            or not excluded_ids.isdisjoint(obstruction.source_ids)
        ):
            continue
        if not _obstruction_can_block(obstruction, direction):
            continue
        contact = obstruction.contact_price(direction)
        distance = direction.sign * (contact - entry)
        if distance > tick_size:
            candidates.append(
                (float(distance), obstruction.obstruction_id, obstruction)
            )
    if not candidates:
        return None
    distance, _, obstruction = min(candidates)
    return obstruction, distance


def _frozen_target_crosses_authority_barrier(
    plan: TradePlan,
    observation: MarketObservation,
    global_context: GlobalMarketContext | None,
    tick_size: float,
) -> bool:
    """Report whether a new authority barrier now precedes a frozen target."""

    if global_context is None or not plan.targets:
        return False
    selected_target = plan.targets[0]
    inventory_target = _inventory_item_map(observation).get(
        selected_target.level_id
    )
    selected_ids = frozenset(
        {
            plan.selected_draw_id,
            selected_target.level_id,
            *(
                ()
                if inventory_target is None
                else inventory_target.source_ids
            ),
        }
    )
    barrier = _nearest_authority_barrier(
        plan.direction,
        plan.planned_entry,
        global_context,
        tick_size,
        excluded_ids=selected_ids,
    )
    if barrier is None:
        return False
    obstruction, barrier_distance = barrier
    if abs(
        obstruction.contact_price(plan.direction) - selected_target.price
    ) <= tick_size:
        return False
    target_distance = plan.direction.sign * (
        selected_target.price - plan.planned_entry
    )
    return bool(
        target_distance > tick_size
        and target_distance + tick_size >= barrier_distance
    )


def _cluster_directional_liquidity(
    observation: MarketObservation,
    direction: Direction,
    entry: float,
    tick_size: float,
    *,
    allowed_ids: frozenset[str] | None = None,
) -> tuple[tuple[LiquidityLevel, tuple[str, ...]], ...]:
    """Cluster visible draw identities by conservative first-contact price."""

    inventory = _inventory_item_map(observation)
    projected: list[LiquidityLevel] = []
    for level in _visible_levels(observation):
        if allowed_ids is not None and level.level_id not in allowed_ids:
            continue
        item = inventory.get(level.level_id)
        if item is None:
            continue
        contact = _liquidity_contact_price(item, direction)
        if contact is None:
            continue
        projected.append(replace(level, price=contact))
    projected.sort(key=lambda value: (value.price, value.level_id))
    clusters: list[list[LiquidityLevel]] = []
    cluster_upper: list[float] = []
    for level in projected:
        if not clusters or level.price > cluster_upper[-1] + tick_size:
            clusters.append([level])
            cluster_upper.append(level.price)
        else:
            clusters[-1].append(level)
            cluster_upper[-1] = max(cluster_upper[-1], level.price)
    output: list[tuple[LiquidityLevel, tuple[str, ...]]] = []
    for cluster in clusters:
        representative = min(
            cluster,
            key=lambda value: _draw_rank(observation, value, entry),
        )
        member_ids = tuple(
            value.level_id
            for value in sorted(
                cluster,
                key=lambda value: (
                    _draw_rank(observation, value, entry),
                    value.level_id,
                ),
            )
        )
        output.append((representative, member_ids))
    return tuple(output)


def _select_countertrend_lsr_target_result(
    observation: MarketObservation,
    direction: Direction,
    entry: float,
    config: BrainConfig,
    global_context: GlobalMarketContext,
    *,
    context_draw: LiquidityLevel | None,
    preferred_id: str | None = None,
    allowed_ids: frozenset[str] | None = None,
    location: EntryLocationState | None = None,
    invalidation: StructuralLevel | None = None,
) -> _CountertrendTargetSelection:
    """Choose visible local delivery strictly before authority obstruction.

    An intact H4/H1 authority does not invalidate the local reversal thesis,
    but it forbids assuming delivery to or through that authority.  Timeframe
    is not authority: an internal H1 level may be a valid local target, while
    an M5 level beyond a protected barrier is not.
    """

    clusters = _cluster_directional_liquidity(
        observation,
        direction,
        entry,
        config.tick_size,
        allowed_ids=allowed_ids,
    )
    if not clusters:
        return _CountertrendTargetSelection(
            target=None,
            context_draw=None,
            failure_reason="visible_draw_missing",
            authority_barrier_id=None,
            authority_barrier_price=None,
        )

    barrier = _nearest_authority_barrier(
        direction,
        entry,
        global_context,
        config.tick_size,
    )
    if barrier is None:
        return _CountertrendTargetSelection(
            target=None,
            context_draw=None,
            failure_reason="authority_barrier_missing",
            authority_barrier_id=None,
            authority_barrier_price=None,
        )
    barrier_obstruction, _ = barrier
    barrier_price = barrier_obstruction.contact_price(direction)

    barrier_distance = direction.sign * (barrier_price - entry)
    candidate_clusters = tuple(
        (level, member_ids)
        for level, member_ids in clusters
        if (
            (distance := direction.sign * (level.price - entry))
            > config.tick_size
            and barrier_distance > config.tick_size
            and distance + config.tick_size
            < barrier_distance
        )
    )
    if not candidate_clusters:
        return _CountertrendTargetSelection(
            target=None,
            context_draw=None,
            failure_reason="all_draws_beyond_barrier",
            authority_barrier_id=barrier_obstruction.obstruction_id,
            authority_barrier_price=barrier_price,
        )

    ordered = tuple(
        sorted(
            candidate_clusters,
            key=lambda item: (
                direction.sign * (item[0].price - entry),
                _draw_rank(observation, item[0], entry),
            ),
        )
    )
    if preferred_id is not None:
        ordered = tuple(
            sorted(
                ordered,
                key=lambda item: preferred_id not in item[1],
            )
        )
    # Compatibility callers without frozen entry geometry may inspect the
    # price-path choice, but the production LSR path always supplies both.
    if location is None or invalidation is None:
        selected = ordered[0][0]
    else:
        risk_qualified: list[tuple[LiquidityLevel, float]] = []
        for level, _ in ordered:
            planned = _select_planned_entry(
                observation,
                direction,
                location,
                invalidation,
                level,
                config,
            )
            if planned is None:
                continue
            risk = abs(planned - invalidation.price)
            target_distance = direction.sign * (level.price - planned)
            barrier_from_plan = direction.sign * (barrier_price - planned)
            if (
                risk >= config.tick_size
                and target_distance / risk >= 1.0
                and barrier_from_plan > config.tick_size
                and target_distance + config.tick_size < barrier_from_plan
            ):
                risk_qualified.append((level, risk))
        if not risk_qualified:
            return _CountertrendTargetSelection(
                target=None,
                context_draw=None,
                failure_reason="all_targets_below_minimum_R",
                authority_barrier_id=barrier_obstruction.obstruction_id,
                authority_barrier_price=barrier_price,
            )
        selected = next(
            (
                level
                for level, risk in risk_qualified
                if max(
                    0.0,
                    direction.sign * (level.price - observation.price),
                )
                / risk
                >= config.minimum_remaining_path_R
            ),
            None,
        )
        if selected is None:
            return _CountertrendTargetSelection(
                target=None,
                context_draw=None,
                failure_reason="remaining_path_below_minimum",
                authority_barrier_id=barrier_obstruction.obstruction_id,
                authority_barrier_price=barrier_price,
            )

    selected_distance = direction.sign * (selected.price - entry)
    terminal_clusters = tuple(
        (level, member_ids)
        for level, member_ids in candidate_clusters
        if direction.sign * (level.price - entry) + config.tick_size
        >= selected_distance
    )
    preferred_context = next(
        (
            level
            for level, member_ids in terminal_clusters
            if context_draw is not None and context_draw.level_id in member_ids
        ),
        None,
    )
    terminal_draw = (
        preferred_context
        if preferred_context is not None
        else min(
            (level for level, _ in terminal_clusters),
            key=lambda level: _draw_rank(observation, level, entry),
        )
    )
    return _CountertrendTargetSelection(
        target=selected,
        context_draw=terminal_draw,
        failure_reason=None,
        authority_barrier_id=barrier_obstruction.obstruction_id,
        authority_barrier_price=barrier_price,
    )


def _select_countertrend_lsr_target(
    observation: MarketObservation,
    direction: Direction,
    entry: float,
    config: BrainConfig,
    global_context: GlobalMarketContext,
    *,
    context_draw: LiquidityLevel | None,
    preferred_id: str | None = None,
    allowed_ids: frozenset[str] | None = None,
    location: EntryLocationState | None = None,
    invalidation: StructuralLevel | None = None,
) -> LiquidityLevel | None:
    """Compatibility view over the richer countertrend selection result."""

    return _select_countertrend_lsr_target_result(
        observation,
        direction,
        entry,
        config,
        global_context,
        context_draw=context_draw,
        preferred_id=preferred_id,
        allowed_ids=allowed_ids,
        location=location,
        invalidation=invalidation,
    ).target


def _liquidity_route(
    observation: MarketObservation,
    direction: Direction,
    entry: float,
    *,
    context_draw: LiquidityLevel | None,
    primary_target: LiquidityLevel | None,
    prior_route: LiquidityRoute | None = None,
    range_context: _LSRRangeContext | None = None,
    conservative_contacts: bool = False,
    authority_barrier_id: str | None = None,
    authority_barrier_price: float | None = None,
    tick_size: float = 0.25,
) -> LiquidityRoute | None:
    if primary_target is None:
        return None
    inventory = _inventory_item_map(observation)
    primary_target_price_basis = (
        "conservative_contact"
        if conservative_contacts
        else "inventory_price"
    )

    def route_level(level: LiquidityLevel) -> LiquidityLevel:
        if not conservative_contacts:
            return level
        item = inventory.get(level.level_id)
        contact = (
            None
            if item is None
            else _liquidity_contact_price(item, direction)
        )
        return level if contact is None else replace(level, price=contact)

    routed_context_draw = (
        None if context_draw is None else route_level(context_draw)
    )
    deliverable = sorted(
        (
            projected
            for representative, _ in _cluster_directional_liquidity(
                observation,
                direction,
                entry,
                tick_size,
            )
            for projected in (route_level(representative),)
            if _target_is_deliverable(
                projected,
                direction,
                entry,
                tick_size,
            )
            and direction.sign * (projected.price - primary_target.price)
            < -tick_size
        ),
        key=lambda level: (
            direction.sign * (level.price - entry),
            _draw_rank(observation, level, entry),
        ),
    )
    intermediates = tuple(
        level.level_id
        for level in deliverable
        if level.level_id != primary_target.level_id
    )
    terminal = (
        routed_context_draw
        if routed_context_draw is not None
        else primary_target
    )
    # Liquidity is a waypoint/draw, not an obstruction.  Hard barriers are
    # attached later from GlobalMarketContext after planned-entry geometry is
    # known; do not turn every nearer visible level into a blocker here.
    blockers: tuple[str, ...] = ()
    source_path = tuple(
        dict.fromkeys(
            value
            for value in (
                *(
                    ()
                    if range_context is None
                    else (
                        range_context.dealing_range.range_id,
                        range_context.swept_boundary.item_id,
                        None
                        if range_context.opposing_boundary is None
                        else range_context.opposing_boundary.item_id,
                    )
                ),
                None if context_draw is None else context_draw.level_id,
                *intermediates,
                primary_target.level_id,
                terminal.level_id,
            )
            if value is not None
        )
    )
    if (
        prior_route is not None
        and prior_route.context_draw_id
        == (None if context_draw is None else context_draw.level_id)
        and prior_route.primary_deliverable_target_id
        == primary_target.level_id
        and prior_route.terminal_draw_id == terminal.level_id
        and prior_route.authority_barrier_id == authority_barrier_id
        and prior_route.authority_barrier_price == authority_barrier_price
        and prior_route.primary_target_price_basis
        == primary_target_price_basis
        and prior_route.intermediate_liquidity_ids == intermediates
        and prior_route.source_path_ids == source_path
        and prior_route.range_context_id
        == (
            None
            if range_context is None
            else range_context.dealing_range.range_id
        )
        and prior_route.range_midpoint
        == (
            None
            if range_context is None
            else range_context.dealing_range.midpoint
        )
        and prior_route.swept_range_boundary_id
        == (
            None
            if range_context is None
            else range_context.swept_boundary.item_id
        )
        and prior_route.opposing_range_boundary_id
        == (
            None
            if (
                range_context is None
                or range_context.opposing_boundary is None
            )
            else range_context.opposing_boundary.item_id
        )
    ):
        return prior_route
    raw = (
        f"{observation.asof.isoformat()}|{direction.value}|"
        f"{None if context_draw is None else context_draw.level_id}|"
        f"{primary_target.level_id}|{terminal.level_id}|{source_path}|"
        f"{primary_target_price_basis}|{authority_barrier_id}|"
        f"{authority_barrier_price}"
    )
    return LiquidityRoute(
        route_id=f"route:{hashlib.sha256(raw.encode()).hexdigest()[:24]}",
        selected_at=observation.asof,
        context_draw_id=(
            None if context_draw is None else context_draw.level_id
        ),
        intermediate_liquidity_ids=intermediates,
        primary_deliverable_target_id=primary_target.level_id,
        terminal_draw_id=terminal.level_id,
        authority_barrier_id=authority_barrier_id,
        authority_barrier_price=authority_barrier_price,
        primary_target_price_basis=primary_target_price_basis,
        path_blocker_ids=blockers,
        source_path_ids=source_path,
        range_context_id=(
            None
            if range_context is None
            else range_context.dealing_range.range_id
        ),
        range_midpoint=(
            None
            if range_context is None
            else range_context.dealing_range.midpoint
        ),
        swept_range_boundary_id=(
            None
            if range_context is None
            else range_context.swept_boundary.item_id
        ),
        opposing_range_boundary_id=(
            None
            if (
                range_context is None
                or range_context.opposing_boundary is None
            )
            else range_context.opposing_boundary.item_id
        ),
    )


def _execution_unavailable(observation: MarketObservation) -> bool:
    names = set(observation.anomalies) | set(
        observation.execution.anomalies
    )
    return bool(
        observation.execution.source in {"missing", "unknown"}
        or observation.execution.source.startswith("constant")
        or names
        & {
            "spread_missing_used_one_tick",
            "execution_constant_assumption",
            "deadline_missing",
            "deadline_elapsed",
            "stale_market_data",
            "insufficient_top_of_book_depth",
        }
    )


def _location_quality(location: EntryLocationState | None) -> float:
    """Describe entry-zone quality without treating first contact as success.

    APPROACHING remains below an actual visit.  IN_ZONE is graded by the
    frozen first penetration, REJECTED by the observed reaction, and LEFT is
    terminally zero.  The empirical mapping is fitted later; these values are
    deliberately descriptive rather than a trade label.
    """

    if location is None or location.lifecycle is EntryLocationLifecycle.LEFT:
        return 0.0
    width = location.upper_bound - location.lower_bound
    if location.lifecycle is EntryLocationLifecycle.APPROACHING:
        return clamp(
            0.5
            / (
                1.0
                + location.distance_to_zone_points
                / max(width, 1e-12)
            )
        )
    if location.lifecycle is EntryLocationLifecycle.IN_ZONE:
        penetration = (
            1.0
            if location.first_penetration_fraction is None
            else float(location.first_penetration_fraction)
        )
        return clamp(0.5 + 0.5 * (1.0 - penetration))
    if location.lifecycle is EntryLocationLifecycle.REJECTED:
        return 1.0
    return 0.0


def _delivery_quality(
    observation: MarketObservation,
    direction: Direction,
    plan: TradePlan | None,
) -> float:
    """Describe remaining causal path before global obstruction routing.

    Liquidity waypoints and descriptive H1 rolling proxies are deliberately
    excluded.  The later GlobalMarketContext router applies price-geometric
    hard barriers and soft friction to this base value.
    """

    if plan is None or not plan.targets:
        return 0.0
    target_distance = abs(plan.targets[0].price - observation.price)
    if target_distance <= 0.0:
        return 1.0
    return clamp(
        plan.remaining_path_R / max(plan.primary_target_R, 1e-12)
    )


def _select_planned_entry(
    observation: MarketObservation,
    direction: Direction,
    location: EntryLocationState | None,
    invalidation: StructuralLevel | None,
    target: LiquidityLevel | None,
    config: BrainConfig,
) -> float | None:
    """Compare frozen zone prices; the eye never substitutes current close."""

    if location is None or invalidation is None or target is None:
        return None
    candidates = [location.near_edge, location.midpoint]
    if location.source_zone_kind == "order_block":
        order_block = next(
            (
                item
                for item in observation.frame(Timeframe.M5).order_blocks
                if item.order_block_id == location.source_zone_id
            ),
            None,
        )
        if order_block is not None:
            candidates.append(
                order_block.body_upper_bound
                if direction is Direction.LONG
                else order_block.body_lower_bound
            )
    valid: list[tuple[float, float, float, float]] = []
    for candidate in dict.fromkeys(float(value) for value in candidates):
        if not location.lower_bound <= candidate <= location.upper_bound:
            continue
        risk_points = abs(candidate - invalidation.price)
        reward_points = direction.sign * (target.price - candidate)
        # This is a market-location comparison.  Spread and execution cost
        # belong to Decision/Risk and must not change the Brain's selected
        # structural entry or its five market-quality dimensions.
        net_space = reward_points
        if (
            risk_points < config.tick_size
            or reward_points <= 0.0
            or net_space <= 0.0
            or (
                direction is Direction.LONG
                and invalidation.price >= candidate
            )
            or (
                direction is Direction.SHORT
                and invalidation.price <= candidate
            )
        ):
            continue
        utility = net_space / risk_points
        valid.append(
            (
                utility,
                net_space,
                -abs(candidate - location.near_edge),
                candidate,
            )
        )
    return None if not valid else max(valid)[-1]


def _typed_plan(
    *,
    playbook: Playbook,
    direction: Direction,
    observation: MarketObservation,
    config: BrainConfig,
    setup_id: str | None,
    location: EntryLocationState | None,
    entry_path: PathSequenceState | None,
    invalidation: StructuralLevel | None,
    target: LiquidityLevel | None,
    draw_selection: DrawSelection | None = None,
    range_auction: FrozenRangeAuctionContext | None = None,
    lsr_context: FrozenLSRContext | None = None,
    liquidity_route: LiquidityRoute | None = None,
    planned_entry: float | None = None,
) -> TradePlan | None:
    if (
        setup_id is None
        or location is None
        or entry_path is None
        or invalidation is None
        or target is None
    ):
        return None
    planned_entry = (
        _select_planned_entry(
            observation,
            direction,
            location,
            invalidation,
            target,
            config,
        )
        if planned_entry is None
        else float(planned_entry)
    )
    if planned_entry is None:
        return None
    risk = abs(planned_entry - invalidation.price)
    if (
        risk < config.tick_size
        or (
            direction is Direction.LONG
            and invalidation.price >= planned_entry
        )
        or (
            direction is Direction.SHORT
            and invalidation.price <= planned_entry
        )
        or not _target_is_deliverable(
            target,
            direction,
            planned_entry,
            config.tick_size,
        )
    ):
        return None
    primary_R = abs(target.price - planned_entry) / risk
    remaining_points = max(
        0.0,
        direction.sign * (target.price - observation.price),
    )
    deadline = observation.asof + pd.Timedelta(
        minutes=observation.execution.minutes_to_deadline
    )
    return TradePlan(
        playbook=playbook,
        direction=direction,
        planned_entry=float(planned_entry),
        invalidation=invalidation,
        targets=(target,),
        risk_points=float(risk),
        primary_target_R=float(primary_R),
        remaining_path_R=float(remaining_points / risk),
        deadline=deadline,
        setup_id=setup_id,
        entry_location_id=location.location_id,
        entry_path_id=entry_path.sequence_id,
        entry_zone_lower=location.lower_bound,
        entry_zone_upper=location.upper_bound,
        selected_draw_id=target.level_id,
        draw_selection=draw_selection,
        range_auction=range_auction,
        lsr_context=lsr_context,
        liquidity_route=liquidity_route,
    )


def _typed_evidence_items(
    protocol: PlaybookProtocol,
    observation: MarketObservation,
    support_values: Mapping[str, float],
    contradict_values: Mapping[str, float],
) -> tuple[Evidence, ...]:
    return tuple(
        _evidence(
            name,
            support_values.get(name, 0.0),
            1.0,
            True,
            observation,
            f"typed causal support: {name}",
        )
        for name in protocol.supporting_evidence
    ) + tuple(
        _evidence(
            name,
            contradict_values.get(name, 0.0),
            1.0,
            False,
            observation,
            f"typed causal contradiction: {name}",
        )
        for name in protocol.contradicting_evidence
    )


def _uncertainty_total(
    *,
    conflict: float,
    required_evidence_missing: float,
    authority_missing: float,
    graph_ambiguity: float,
) -> float:
    """Combine independent epistemic deficits with a bounded noisy-OR."""

    components = tuple(
        clamp(value)
        for value in (
            conflict,
            required_evidence_missing,
            authority_missing,
            graph_ambiguity,
        )
    )
    residual_certainty = math.prod(1.0 - value for value in components)
    return clamp(1.0 - residual_certainty)


def _typed_uncertainty(
    *,
    market_support: Sequence[float],
    market_contradictions: Sequence[float],
    authority_missing: float,
    graph_ambiguity: float = 0.0,
) -> Mapping[str, float]:
    """Market uncertainty only; execution readiness is a separate group.

    Sequence progress remains a deterministic stage field.  Missing causal
    evidence still limits how complete the current market reading is, while
    execution data availability belongs to the execution group and risk veto.
    """

    support_mass = sum(clamp(value) for value in market_support)
    contradiction_mass = sum(
        clamp(value) for value in market_contradictions
    )
    conflict = (
        0.0
        if support_mass <= 0.0 or contradiction_mass <= 0.0
        else clamp(
            2.0
            * min(support_mass, contradiction_mass)
            / (support_mass + contradiction_mass)
        )
    )
    # A zero-valued future causal step is sequence progress, not missing
    # evidence.  Stage-specific required inputs are added only after the
    # latched sequence and current phase have been constructed.
    required_missing = 0.0
    authority = clamp(authority_missing)
    ambiguity = clamp(graph_ambiguity)
    total = _uncertainty_total(
        conflict=conflict,
        required_evidence_missing=required_missing,
        authority_missing=authority,
        graph_ambiguity=ambiguity,
    )
    return {
        "typed_uncertainty": total,
        "uncertainty_conflict": conflict,
        "uncertainty_required_evidence_missing": required_missing,
        "uncertainty_authority_missing": authority,
        "uncertainty_graph_ambiguity": ambiguity,
    }


def _typed_market_uncertainty(
    observation: MarketObservation,
    support: Mapping[str, float],
    contradict: Mapping[str, float],
    *,
    semantic_authority_missing: float = 0.0,
    graph_ambiguity: float = 0.0,
) -> Mapping[str, float]:
    execution_names = {
        "remaining_path_available",
        "execution_fillability",
        "remaining_path_consumed",
        "execution_unavailable",
    }
    authority_missing = max(
        clamp(semantic_authority_missing),
        float(
            any(
                name.startswith("clock_")
                for name in observation.anomalies
            )
        ),
    )
    return _typed_uncertainty(
        market_support=tuple(
            value
            for name, value in support.items()
            if name not in execution_names
        ),
        market_contradictions=tuple(
            value
            for name, value in contradict.items()
            if name not in execution_names
        ),
        authority_missing=authority_missing,
        graph_ambiguity=graph_ambiguity,
    )


def _replace_uncertainty_components(
    evaluation: _Evaluation,
    *,
    conflict: float | None = None,
    required_evidence_missing: float | None = None,
    authority_missing: float | None = None,
    graph_ambiguity: float | None = None,
) -> _Evaluation:
    """Replace named components and keep the published total reproducible."""

    resolved_conflict = clamp(
        evaluation.uncertainty_conflict
        if conflict is None
        else conflict
    )
    resolved_required = clamp(
        evaluation.uncertainty_required_evidence_missing
        if required_evidence_missing is None
        else required_evidence_missing
    )
    resolved_authority = clamp(
        evaluation.uncertainty_authority_missing
        if authority_missing is None
        else authority_missing
    )
    resolved_ambiguity = clamp(
        evaluation.uncertainty_graph_ambiguity
        if graph_ambiguity is None
        else graph_ambiguity
    )
    return replace(
        evaluation,
        typed_uncertainty=_uncertainty_total(
            conflict=resolved_conflict,
            required_evidence_missing=resolved_required,
            authority_missing=resolved_authority,
            graph_ambiguity=resolved_ambiguity,
        ),
        uncertainty_conflict=resolved_conflict,
        uncertainty_required_evidence_missing=resolved_required,
        uncertainty_authority_missing=resolved_authority,
        uncertainty_graph_ambiguity=resolved_ambiguity,
    )


_UNCERTAINTY_STEP_TIMEFRAMES: Mapping[
    Playbook,
    tuple[tuple[Timeframe, ...], ...],
] = {
    Playbook.DISPLACEMENT_FIRST_PULLBACK: (
        (Timeframe.H4,),
        (Timeframe.M5,),
        (Timeframe.M1,),
        (Timeframe.M1,),
    ),
    Playbook.LIQUIDITY_SWEEP_REVERSAL: (
        (),
        (),
        (Timeframe.M5,),
        (Timeframe.M1,),
        (Timeframe.M1,),
    ),
    Playbook.FAILED_AUCTION_VALUE_RETURN: (
        (Timeframe.H1,),
        (Timeframe.M1,),
        (Timeframe.M5,),
        (Timeframe.M1,),
        (Timeframe.M1,),
    ),
}


def _stage_uncertainty(
    evaluation: _Evaluation,
    playbook: Playbook,
    sequence: HypothesisSequenceState,
    observation: MarketObservation,
) -> _Evaluation:
    """Add only inputs required by the causal prefix already completed."""

    completed_steps = tuple(step for step in sequence.steps if step.satisfied)
    # hard_gate=False is an observed market result, not missing evidence.
    # The boolean gate contract cannot represent unavailable separately, so
    # only an unavailable typed contract that is already required by the
    # current causal prefix belongs in required_evidence_missing.
    group5_required_after = (
        2
        if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
        else 1
    )
    typed_contract_missing = float(
        len(completed_steps) >= group5_required_after
        and not observation.group5_typed_available
    )
    required_missing = typed_contract_missing
    timeframe_contract = _UNCERTAINTY_STEP_TIMEFRAMES[playbook]
    required_timeframes = {
        timeframe
        for index in range(min(len(completed_steps), len(timeframe_contract)))
        for timeframe in timeframe_contract[index]
    }
    if (
        playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and completed_steps
        and evaluation.authority_source_timeframe is not None
    ):
        try:
            required_timeframes.add(
                Timeframe(evaluation.authority_source_timeframe)
            )
        except ValueError:
            # The typed authority contract itself is missing/invalid; it is
            # not silently replaced with M1 authority.
            required_missing = 1.0
    frame_missing = (
        0.0
        if not required_timeframes
        else sum(
            not observation.frame(timeframe).ready
            for timeframe in required_timeframes
        )
        / len(required_timeframes)
    )
    authority_missing = max(
        evaluation.uncertainty_authority_missing,
        frame_missing,
    )
    return _replace_uncertainty_components(
        evaluation,
        required_evidence_missing=required_missing,
        authority_missing=authority_missing,
    )


def _path_for_location(
    observation: MarketObservation,
    location: EntryLocationState | None,
) -> PathSequenceState | None:
    if location is None:
        return None
    return next(
        (
            path
            for path in observation.path_sequences
            if (
                path.context_kind == "zone_return"
                and path.context_id == location.location_id
                and path.direction is location.direction
            )
        ),
        None,
    )


def _path_step(
    path: PathSequenceState | None,
    kinds: set[str],
):
    if path is None:
        return None
    return next(
        (step for step in path.steps if step.kind in kinds),
        None,
    )


def _micro_reference_has_exact_source(
    observation: MarketObservation,
    reference: MicroBOSReference,
) -> bool:
    return any(
        state.bos_id == reference.bos_id
        and state.timeframe is Timeframe.M1
        and state.lifecycle is BOSLifecycle.CONFIRMED
        and state.direction is reference.bos_direction
        and state.target_swing_id == reference.target_swing_id
        and state.scope is reference.scope
        and state.pending_at == reference.pending_at
        and state.resolved_at == reference.resolved_at
        for state in observation.frame(Timeframe.M1).structure_breaks
    )


def _select_entry_location(
    observation: MarketObservation,
    direction: Direction,
    *,
    after: pd.Timestamp | None,
    prior: HypothesisBelief | None,
    source_displacement_id: str | None = None,
    source_zone_id: str | None = None,
    required_location_id: str | None = None,
    required_thesis: OpenMarketThesis | None = None,
) -> EntryLocationState | None:
    by_id = {
        location.location_id: location
        for location in observation.entry_locations
    }
    path_by_location = {
        path.context_id: path
        for path in observation.path_sequences
        if (
            path.context_kind == "zone_return"
            and path.direction is direction
        )
    }

    def belongs_to_required_root(
        location: EntryLocationState,
    ) -> bool:
        if required_thesis is None:
            return True
        if required_thesis.entry_location_ids:
            return (
                location.location_id
                in required_thesis.entry_location_ids
            )
        path = path_by_location.get(location.location_id)
        closure_ids = {
            *required_thesis.mechanism_event_ids,
            *required_thesis.entry_location_ids,
            *required_thesis.trigger_event_ids,
        }
        return bool(
            {
                location.location_id,
                location.source_displacement_id,
                location.source_zone_id,
                None if path is None else path.sequence_id,
            }
            & closure_ids
        )
    if (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.entry_location_id is not None
    ):
        bound = by_id.get(prior.entry_location_id)
        return (
            bound
            if (
                bound is not None
                and bound.direction is direction
                and (
                    source_displacement_id is None
                    or bound.source_displacement_id
                    == source_displacement_id
                )
                and (
                    source_zone_id is None
                    or bound.source_zone_id == source_zone_id
                )
                and (
                    required_location_id is None
                    or bound.location_id == required_location_id
                )
                and belongs_to_required_root(bound)
            )
            else None
        )
    terminal_cutoff = (
        prior.phase_started_at
        if prior is not None and prior.phase in _TERMINAL_PHASES
        else None
    )
    candidates = [
        location
        for location in observation.entry_locations
        if (
            location.direction is direction
            and (
                required_location_id is None
                or location.location_id == required_location_id
            )
            and (
                source_displacement_id is None
                or location.source_displacement_id
                == source_displacement_id
            )
            and (
                source_zone_id is None
                or location.source_zone_id == source_zone_id
            )
            and location.lifecycle is not EntryLocationLifecycle.LEFT
            and belongs_to_required_root(location)
            and location.location_id in path_by_location
            and (
                path_by_location[location.location_id].lifecycle
                is PathSequenceLifecycle.ACTIVE
                or path_by_location[location.location_id].ended_at
                == observation.asof
            )
            and (after is None or location.formed_at > after)
            and (
                terminal_cutoff is None
                or location.formed_at > terminal_cutoff
            )
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda location: (
            location.formed_at,
            location.location_id,
        ),
    )


def _typed_dfp(
    observation: MarketObservation,
    direction: Direction,
    protocol: PlaybookProtocol,
    prior: HypothesisBelief | None,
    config: BrainConfig,
    *,
    global_context: GlobalMarketContext | None = None,
    required_thesis: OpenMarketThesis | None = None,
) -> _Evaluation:
    required_draw_ids = (
        None
        if required_thesis is None
        else frozenset(required_thesis.draw_candidate_ids)
    )
    all_structures = {
        item.structure_id: item
        for item in observation.frame(Timeframe.H4).structures
        if item.structure_id is not None
    }
    structures = {
        item.structure_id: item
        for item in all_structures.values()
        if (
            item.direction is direction
            and item.lifecycle is StructureLifecycle.CONFIRMED
            and item.confirmed_at is not None
        )
    }
    frozen_context_sources = _prior_step_sources(
        prior,
        "h4_structure_and_draw",
    )
    frozen_structure_id = (
        frozen_context_sources[0]
        if len(frozen_context_sources) >= 1
        else None
    )
    frozen_draw_id = (
        frozen_context_sources[1]
        if len(frozen_context_sources) >= 2
        else None
    )
    structure = (
        structures.get(frozen_structure_id)
        if frozen_structure_id is not None
        else (
            max(
                structures.values(),
                key=lambda item: (
                    item.confirmed_at,
                    item.structure_id,
                ),
            )
            if structures
            else None
        )
    )
    visible = _visible_level_map(observation)
    reference_entry = float(observation.price)
    prior_plan_same_direction = (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.plan is not None
        and prior.direction is direction
        and prior.plan.playbook
        is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    preferred_context_draw_id = (
        prior.liquidity_route.context_draw_id
        if (
            prior is not None
            and prior.phase not in _TERMINAL_PHASES
            and prior.liquidity_route is not None
        )
        else frozen_draw_id
    )
    preferred_primary_target_id = (
        prior.liquidity_route.primary_deliverable_target_id
        if (
            prior is not None
            and prior.phase not in _TERMINAL_PHASES
            and prior.liquidity_route is not None
        )
        else prior.draw_selection.draw_id
        if (
            prior is not None
            and prior.phase not in _TERMINAL_PHASES
            and prior.draw_selection is not None
        )
        else prior.plan.selected_draw_id
        if prior_plan_same_direction
        else None
    )
    context_draw = _select_target(
        observation,
        direction,
        reference_entry,
        config,
        preferred_id=preferred_context_draw_id,
        required_timeframe=Timeframe.H4,
        require_external=True,
        require_preferred=preferred_context_draw_id is not None,
        allowed_ids=required_draw_ids,
    )
    context_clock = (
        max(structure.confirmed_at, context_draw.confirmed_at)
        if structure is not None and context_draw is not None
        else None
    )
    h1_candidates = {
        item.bos_id: item
        for item in observation.frame(Timeframe.H1).structure_breaks
        if (
            item.direction is direction
            and item.lifecycle is BOSLifecycle.CONFIRMED
            and item.scope is BOSScope.CONTINUATION
            and item.post_break_state is BOSPostBreakState.ACCEPTED
            and item.resolved_at is not None
            and context_clock is not None
            and item.resolved_at > context_clock
        )
    }
    h1_bos = (
        min(
            h1_candidates.values(),
            key=lambda item: (item.resolved_at, item.bos_id),
        )
        if h1_candidates
        else None
    )
    location = _select_entry_location(
        observation,
        direction,
        after=context_clock,
        prior=prior,
        required_thesis=required_thesis,
    )
    entry_path = _path_for_location(observation, location)
    if (
        location is not None
        and (
            context_clock is None
            or location.formed_at <= context_clock
        )
    ):
        location = None
        entry_path = None
    first_pullback = _path_step(entry_path, {"first_pullback"})
    if (
        first_pullback is not None
        and location is not None
        and first_pullback.source_entity_id != location.location_id
    ):
        # A malformed or cross-episode path must not borrow a return from a
        # different frozen zone, even if the enclosing path identity matches.
        first_pullback = None
    wick_trigger = _path_step(entry_path, {"wick_rejection"})
    held_trigger = _path_step(entry_path, {"reacceptance_held"})
    micro_trigger = _path_step(entry_path, {"micro_bos_confirmed"})
    qualified_micro = next(
        (
            reference
            for reference in observation.micro_bos_references
            if (
                location is not None
                and micro_trigger is not None
                and reference.context_kind == "zone_return"
                and reference.context_id == location.location_id
                and reference.expected_direction is direction
                and reference.qualified
                and reference.bos_id == micro_trigger.source_event_id
                and reference.target_swing_id
                == micro_trigger.source_entity_id
                and reference.resolved_at == micro_trigger.observed_at
                and _micro_reference_has_exact_source(
                    observation,
                    reference,
                )
            )
        ),
        None,
    )
    qualified_reacceptance = next(
        (
            state
            for state in observation.qualified_reacceptances
            if (
                location is not None
                and held_trigger is not None
                and state.reacceptance_id
                == held_trigger.source_entity_id
                and state.context_kind == "entry_zone"
                and state.context_id == location.location_id
                and state.direction is direction
                and state.lifecycle
                is QualifiedReacceptanceLifecycle.HELD
                and state.held_at == held_trigger.observed_at
            )
        ),
        None,
    )
    ambiguity_step = _path_step(
        entry_path,
        {"micro_bos_ambiguous"},
    )
    contradiction_step = _path_step(
        entry_path,
        {
            "location_left",
            "reacceptance_failed",
            "micro_bos_opposed",
        },
    )
    valid_triggers = tuple(
        step
        for step, same_clock_allowed, qualified in (
            (
                wick_trigger,
                True,
                wick_trigger is not None
                and location is not None
                and wick_trigger.source_entity_id
                == location.location_id,
            ),
            (
                held_trigger,
                False,
                qualified_reacceptance is not None,
            ),
            (micro_trigger, False, qualified_micro is not None),
        )
        if (
            step is not None
            and qualified
            and first_pullback is not None
            and step.direction is direction
            and (
                step.observed_at >= first_pullback.observed_at
                if same_clock_allowed
                else step.observed_at > first_pullback.observed_at
            )
        )
    )
    candidate_setup_id = (
        None if entry_path is None else entry_path.sequence_id
    )
    selected_trigger, trigger_step = _select_frozen_trigger(
        valid_triggers,
        prior=prior,
        setup_id=candidate_setup_id,
        entry_path_id=candidate_setup_id,
        entry_location_id=(
            None if location is None else location.location_id
        ),
        direction=direction,
    )
    trigger_ready = bool(
        first_pullback is not None
        and trigger_step is not None
        and ambiguity_step is None
        and contradiction_step is None
        and location is not None
        and location.lifecycle is not EntryLocationLifecycle.LEFT
        and entry_path is not None
        and entry_path.lifecycle is not PathSequenceLifecycle.CENSORED
    )
    context_draw_id = (
        frozen_draw_id
        if frozen_draw_id is not None
        else None if context_draw is None else context_draw.level_id
    )
    context_identity = (
        (
            f"dfp-context:{structure.structure_id}:"
            f"{context_draw_id}"
        )
        if structure is not None and context_draw_id is not None
        else None
    )
    episode_identity = (
        None if entry_path is None else entry_path.sequence_id
    )
    setup_identity = episode_identity or context_identity
    initiating_event_id = (
        location.source_displacement_id
        if location is not None
        else None if structure is None else structure.structure_id
    )
    invalidation = (
        StructuralLevel(
            price=location.failure_boundary,
            side=direction.invalidation_side,
            source_level_id=location.location_id,
            observed_at=location.formed_at,
            rationale=(
                "exact frozen Group 5 entry-zone failure boundary"
            ),
        )
        if location is not None
        else None
    )
    if location is not None:
        planned_entry = (
            location.contact_reference_price
            if location.contact_reference_price is not None
            else location.near_edge
        )
        context_draw = _select_target(
            observation,
            direction,
            float(planned_entry),
            config,
            preferred_id=preferred_context_draw_id,
            required_timeframe=Timeframe.H4,
            require_external=True,
            require_preferred=preferred_context_draw_id is not None,
            allowed_ids=required_draw_ids,
        )
    else:
        planned_entry = reference_entry
    target_selection_failure_reason: str | None = None
    authority_barrier_id: str | None = None
    authority_barrier_price: float | None = None
    if location is not None and invalidation is not None:
        if context_draw is None:
            primary_target = None
            target_selection_failure_reason = (
                "context_draw_consumed"
                if preferred_context_draw_id is not None
                else "visible_draw_missing"
            )
        else:
            target_selection = _select_dfp_primary_target_result(
                observation,
                direction,
                float(planned_entry),
                config,
                global_context,
                context_draw=context_draw,
                preferred_id=preferred_primary_target_id,
                allowed_ids=required_draw_ids,
                location=location,
                invalidation=invalidation,
            )
            primary_target = target_selection.target
            target_selection_failure_reason = (
                target_selection.failure_reason
            )
            authority_barrier_id = (
                target_selection.authority_barrier_id
            )
            authority_barrier_price = (
                target_selection.authority_barrier_price
            )
    else:
        primary_target = None
    selected_planned_entry = _select_planned_entry(
        observation,
        direction,
        location,
        invalidation,
        primary_target,
        config,
    )
    if selected_planned_entry is not None:
        planned_entry = selected_planned_entry
    draw_selection = _draw_selection(
        observation,
        primary_target if structure is not None else None,
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        prior=prior,
    )
    liquidity_route = (
        None
        if location is None
        else _liquidity_route(
            observation,
            direction,
            float(planned_entry),
            context_draw=context_draw,
            primary_target=primary_target,
            prior_route=(
                None if prior is None else prior.liquidity_route
            ),
            authority_barrier_id=authority_barrier_id,
            authority_barrier_price=authority_barrier_price,
            tick_size=config.tick_size,
        )
    )
    plan = _typed_plan(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=direction,
        observation=observation,
        config=config,
        setup_id=setup_identity,
        location=location,
        entry_path=entry_path,
        invalidation=invalidation,
        target=primary_target,
        draw_selection=draw_selection,
        liquidity_route=liquidity_route,
        planned_entry=selected_planned_entry,
    )
    remaining_ok = bool(
        plan is not None
        and plan.remaining_path_R >= config.minimum_remaining_path_R
    )
    execution_missing = _execution_unavailable(observation)
    frozen_structure_state = (
        None
        if frozen_structure_id is None
        else all_structures.get(frozen_structure_id)
    )
    frozen_structure_invalidated = bool(
        frozen_structure_state is not None
        and (
            frozen_structure_state.direction is not direction
            or frozen_structure_state.lifecycle
            in {
                StructureLifecycle.BROKEN,
                StructureLifecycle.FORMATION_FAILED,
            }
        )
    )
    frozen_context_observed_at = next(
        (
            step.observed_at
            for step in (() if prior is None or prior.sequence is None else prior.sequence.steps)
            if step.step_id == "h4_structure_and_draw"
            and step.observed_at is not None
        ),
        None if structure is None else structure.confirmed_at,
    )
    opposed_structures = tuple(
        item
        for item in observation.frame(Timeframe.H4).structures
        if (
            item.direction is not direction
            and item.lifecycle is StructureLifecycle.CONFIRMED
            and item.confirmed_at is not None
            and frozen_context_observed_at is not None
            and item.confirmed_at > frozen_context_observed_at
        )
    )
    opposed_structure = frozen_structure_invalidated or bool(
        opposed_structures
    )
    zone_failed = bool(
        location is not None
        and location.lifecycle is EntryLocationLifecycle.LEFT
    )
    stale_trigger = bool(
        entry_path is not None
        and entry_path.lifecycle is PathSequenceLifecycle.CLOSED
        and entry_path.ended_at != observation.asof
    )
    context_draw_consumed = bool(
        preferred_context_draw_id is not None
        and any(
            item.item_id == preferred_context_draw_id
            and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            for item in observation.liquidity_inventory
        )
    )
    trigger_contradiction = contradiction_step is not None
    trigger_ambiguous = ambiguity_step is not None
    hard_gates = {
        "h4_structure_and_draw": bool(
            structure is not None
            and context_draw is not None
            and not opposed_structure
        ),
        "m5_displacement_zone": bool(
            location is not None and entry_path is not None
        ),
        "first_pullback_to_frozen_zone": (
            first_pullback is not None
        ),
        "typed_entry_trigger": trigger_ready,
    }
    support = {
        "h4_structure_direction": float(structure is not None),
        "h1_continuation_bos": float(h1_bos is not None),
        "m5_displacement_zone": float(
            location is not None and entry_path is not None
        ),
        "directional_draw_visible": float(context_draw is not None),
        "first_pullback_to_frozen_zone": float(
            first_pullback is not None
        ),
        "typed_entry_trigger": float(trigger_ready),
        "remaining_path_available": float(remaining_ok),
        "execution_fillability": (
            0.0
            if execution_missing
            else observation.execution.fillability
        ),
    }
    contradict = {
        "opposed_structure": float(opposed_structure),
        "zone_left_or_failed": float(zone_failed),
        "trigger_opposed": float(
            trigger_contradiction
        ),
        "draw_consumed_or_missing": float(
            context_draw_consumed
        ),
        "remaining_path_consumed": float(
            plan is not None and not remaining_ok
        ),
        "execution_unavailable": float(execution_missing),
    }
    group_scores = {
        # The optional H1 acceptance remains an evidence revision, but its
        # absence cannot halve the mandatory H4 structure group.
        "structure": support["h4_structure_direction"]
        * (1.0 - contradict["opposed_structure"]),
        "displacement": support["m5_displacement_zone"],
        "location": _location_quality(location)
        * (1.0 - contradict["zone_left_or_failed"]),
        "liquidity": support["directional_draw_visible"]
        * (1.0 - contradict["draw_consumed_or_missing"]),
        "trigger": support["typed_entry_trigger"]
        * (1.0 - contradict["trigger_opposed"]),
        "execution": support["execution_fillability"]
        * (1.0 - contradict["execution_unavailable"]),
    }
    censored_path = bool(
        entry_path is not None
        and entry_path.lifecycle is PathSequenceLifecycle.CENSORED
    )
    terminal_reason: str | None = None
    terminal_source_ids: tuple[str, ...] = ()
    if opposed_structure:
        terminal_reason = "opposed_structure"
        terminal_source_ids = _identity_tuple(
            frozen_structure_id,
            *(
                item.structure_id
                for item in opposed_structures
            ),
        )
    elif context_draw_consumed:
        terminal_reason = "context_draw_consumed_or_missing"
        terminal_source_ids = _identity_tuple(
            preferred_context_draw_id
        )
    elif zone_failed:
        terminal_reason = "entry_zone_left_or_failed"
        terminal_source_ids = _identity_tuple(
            None if location is None else location.location_id
        )
    elif trigger_contradiction:
        terminal_reason = "trigger_opposed"
        terminal_source_ids = _identity_tuple(
            None
            if contradiction_step is None
            else contradiction_step.source_entity_id,
            None
            if contradiction_step is None
            else contradiction_step.source_event_id,
        )
    elif censored_path:
        terminal_reason = "entry_path_censored"
        terminal_source_ids = _identity_tuple(
            None if entry_path is None else entry_path.sequence_id
        )
    elif stale_trigger:
        terminal_reason = "entry_window_expired"
        terminal_source_ids = _identity_tuple(
            None if entry_path is None else entry_path.sequence_id
        )
    setup_clock = (
        entry_path.formed_at
        if entry_path is not None
        else context_clock
    )
    core_thesis = min(
        (
            0.0
            if structure is None
            else clamp(
                structure.cumulative_magnitude_atr
                / max(1.0, float(structure.sequence_count))
            )
        ),
        (
            0.0
            if context_draw is None
            else clamp(
                getattr(
                    _inventory_item_map(observation).get(
                        context_draw.level_id
                    ),
                    "strength",
                    0.0,
                )
            )
        ),
    )
    # H1 continuation acceptance is a bounded supporting group, not a
    # prerequisite. Its absence must not dilute the H4 structure/draw thesis.
    thesis_target = (
        core_thesis
        if h1_bos is None
        else (core_thesis + clamp(h1_bos.strength)) / 2.0
    )
    return _Evaluation(
        evidence=_typed_evidence_items(
            protocol,
            observation,
            support,
            contradict,
        ),
        trigger_ready=trigger_ready,
        setup_clock=setup_clock,
        sequence_signals={
            "h4_structure_and_draw": _SequenceSignal(
                float(hard_gates["h4_structure_and_draw"]),
                context_clock,
                tuple(
                    value
                    for value in (
                        None
                        if structure is None
                        else structure.structure_id,
                        None
                        if context_draw is None
                        else context_draw.level_id,
                        None
                        if structure is None
                        else structure.latest_high_id,
                        None
                        if structure is None
                        else structure.latest_low_id,
                        None
                        if structure is None
                        else structure.protected_swing_id,
                    )
                    if value is not None
                ),
            ),
            "m5_displacement_zone": _SequenceSignal(
                float(location is not None and entry_path is not None),
                None if location is None else location.formed_at,
                ()
                if location is None
                else (
                    location.location_id,
                    location.source_displacement_id,
                    location.source_zone_id,
                ),
            ),
            "first_pullback_to_frozen_zone": _SequenceSignal(
                float(first_pullback is not None),
                (
                    None
                    if first_pullback is None
                    else first_pullback.observed_at
                ),
                ()
                if first_pullback is None
                else (first_pullback.source_entity_id,),
            ),
            "typed_entry_trigger": _SequenceSignal(
                float(trigger_ready),
                (
                    None
                    if selected_trigger is None
                    else selected_trigger.observed_at
                ),
                ()
                if selected_trigger is None
                else _identity_tuple(
                    selected_trigger.entry_path_id,
                    selected_trigger.trigger_id,
                    selected_trigger.source_entity_id,
                    selected_trigger.source_event_id,
                ),
            ),
        },
        setup_identity=setup_identity,
        context_identity=context_identity,
        episode_identity=episode_identity,
        initiating_event_id=initiating_event_id,
        entry_location_id=(
            None if location is None else location.location_id
        ),
        entry_path_id=(
            None if entry_path is None else entry_path.sequence_id
        ),
        plan=plan,
        invalidation=invalidation,
        selected_draw=primary_target,
        thesis_draw=context_draw,
        draw_selection=draw_selection,
        liquidity_route=liquidity_route,
        thesis_target=thesis_target,
        location_quality=group_scores["location"],
        entry_readiness=(
            0.0
            if not trigger_ready or trigger_step is None
            else clamp(trigger_step.strength)
        ),
        delivery_quality=_delivery_quality(
            observation,
            direction,
            plan,
        ),
        **_typed_market_uncertainty(
            observation,
            support,
            contradict,
            graph_ambiguity=float(trigger_ambiguous),
        ),
        ambiguity_count=int(trigger_ambiguous),
        evidence_group_scores=group_scores,
        hard_gate_results=hard_gates,
        invalidated=bool(
            opposed_structure
            or zone_failed
            or trigger_contradiction
            or censored_path
            or (
                preferred_context_draw_id is not None
                and context_draw_consumed
            )
        ),
        entry_window_expired=stale_trigger,
        terminal_reason=terminal_reason,
        terminal_source_ids=terminal_source_ids,
        selected_trigger=selected_trigger,
        target_selection_failure_reason=target_selection_failure_reason,
    )


def _global_identity_values(context: object | None, name: str) -> tuple[str, ...]:
    if context is None:
        return ()
    raw = getattr(context, name, ())
    if isinstance(raw, Mapping):
        values = tuple(value for group in raw.values() for value in group)
    elif isinstance(raw, str):
        values = (raw,)
    else:
        values = tuple(raw or ())
    return tuple(
        dict.fromkeys(
            identity
            for value in values
            for identity in (
                value
                if isinstance(value, str)
                else getattr(value, "item_id", None)
                or getattr(value, "level_id", None)
                or getattr(value, "draw_id", None),
            )
            if isinstance(identity, str) and identity
        )
    )


def _lsr_context_ids(global_context: object | None) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *_global_identity_values(
                    global_context,
                    "authority_source_ids",
                ),
                *_global_identity_values(
                    global_context,
                    "external_draw_candidates",
                ),
            )
        )
    )


def _whitelisted_global_connection(
    source_ids: Sequence[str],
    context_ids: Sequence[str],
    scene_graph: TemporalMarketSceneGraph | None,
    *,
    asof: pd.Timestamp,
) -> bool:
    """Use only bounded current-epoch semantic adjacency.

    The public historical path query is deliberately not used here: LSR root
    authority is a live Brain concern, so a closed edge or an earlier market
    epoch cannot promote a current M15/M5 sweep.  Relations are traversed in
    either orientation because source identities may sit on either endpoint of
    ``LOCATED_AT``/``SOURCED_FROM`` while the relation itself remains explicit.
    """

    # Connectivity is set-valued; canonical order also lets repeated
    # root-candidate evaluations share one answer during this Brain clock.
    sources = tuple(sorted(set(value for value in source_ids if value)))
    targets = tuple(sorted(set(value for value in context_ids if value)))
    memo = _LSR_CONNECTION_MEMO.get()
    memo_key = (
        None if scene_graph is None else id(scene_graph),
        None if scene_graph is None else scene_graph.revision_id,
        pd.Timestamp(asof).isoformat(),
        sources,
        targets,
    )
    if memo is not None and memo_key in memo:
        return memo[memo_key]

    def resolved(value: bool) -> bool:
        if memo is not None:
            memo[memo_key] = value
        return value

    if not sources or not targets:
        return resolved(False)
    if not set(sources).isdisjoint(targets):
        return resolved(True)
    if scene_graph is None:
        return resolved(False)
    if scene_graph.last_asof is None or asof != scene_graph.last_asof:
        return resolved(False)
    current_epoch = scene_graph._market_epoch_id
    start_nodes = tuple(
        dict.fromkeys(
            node_id
            for source_id in sources
            for node_id in (
                scene_graph._node_id_for_source(source_id, asof=asof),
            )
            if node_id is not None
            and node_id in scene_graph._nodes
            and scene_graph._nodes[node_id].market_epoch_id == current_epoch
        )
    )
    target_nodes = {
        node_id
        for target_id in targets
        for node_id in (
            scene_graph._node_id_for_source(target_id, asof=asof),
        )
        if node_id is not None
        and node_id in scene_graph._nodes
        and scene_graph._nodes[node_id].market_epoch_id == current_epoch
    }
    if not start_nodes or not target_nodes:
        return resolved(False)
    queue = deque((node_id, 0) for node_id in start_nodes)
    seen = set(start_nodes)
    while queue:
        node_id, depth = queue.popleft()
        if node_id in target_nodes:
            return resolved(True)
        if depth >= 5:
            continue
        adjacent_edge_ids = (
            scene_graph._outgoing.get(node_id, set())
            | scene_graph._incoming.get(node_id, set())
        )
        for edge_id in adjacent_edge_ids:
            edge = scene_graph._edges.get(edge_id)
            if (
                edge is None
                or edge.lifecycle != "active"
                or edge.observed_at > asof
                or edge.relation.value not in _LSR_CONTEXT_RELATIONS
            ):
                continue
            neighbor = (
                edge.target_node_id
                if edge.source_node_id == node_id
                else edge.source_node_id
            )
            node = scene_graph._nodes.get(neighbor)
            if (
                node is None
                or node.market_epoch_id != current_epoch
                or neighbor in seen
            ):
                continue
            seen.add(neighbor)
            queue.append((neighbor, depth + 1))
    return resolved(False)


def _lsr_path_stage(path: PathSequenceState) -> int:
    kinds = {step.kind for step in path.steps}
    ordered = (
        {"pool_swept"},
        {"reacceptance_held"},
        {"opposite_displacement"},
        {"micro_bos_confirmed"},
    )
    stage = 0
    for expected in ordered:
        if not kinds.intersection(expected):
            break
        stage += 1
    return stage


def _lsr_candidate(
    observation: MarketObservation,
    path: PathSequenceState,
    *,
    global_context: object | None,
    scene_graph: TemporalMarketSceneGraph | None,
) -> _LSRCandidate | None:
    manipulation = next(
        (
            item
            for item in observation.manipulations
            if (
                item.manipulation_id == path.context_id
                and item.source_kind == "formed_liquidity_pool"
            )
        ),
        None,
    )
    if manipulation is None:
        return None
    inventory = next(
        (
            item
            for item in observation.liquidity_inventory
            if item.item_id == manipulation.source_inventory_item_id
        ),
        None,
    )
    pool = next(
        (
            item
            for item in observation.liquidity_pool_states
            if item.pool_id == manipulation.source_id
        ),
        None,
    )
    context_ids = _lsr_context_ids(global_context)
    source_ids = tuple(
        dict.fromkeys(
            (
                manipulation.manipulation_id,
                manipulation.source_id,
                manipulation.source_inventory_item_id,
                *manipulation.crossed_source_ids,
                *manipulation.coincident_source_ids,
                *(inventory.source_ids if inventory is not None else ()),
            )
        )
    )
    connected = _whitelisted_global_connection(
        source_ids,
        context_ids,
        scene_graph,
        asof=observation.asof,
    )
    structural_rank = str(
        getattr(inventory, "structural_rank", "unknown") or "unknown"
    )
    if structural_rank not in _LSR_STRUCTURAL_RANK:
        structural_rank = "unknown"
    timeframe = manipulation.source_timeframe
    rank_has_independent_authority = bool(
        inventory is not None
        and structural_rank in {"external", "intermediate"}
    )
    if (
        timeframe in {Timeframe.H4, Timeframe.H1}
        or rank_has_independent_authority
    ):
        tier = "A"
        eligible_root = True
    elif timeframe in {Timeframe.M15, Timeframe.M5}:
        tier = "B" if connected else "ineligible"
        eligible_root = connected
    elif timeframe is Timeframe.M1:
        # A one-minute source remains trigger/refinement evidence even when it
        # is nested in a higher-scale graph.  External metadata alone must not
        # promote it into a standalone reversal thesis.
        tier = "C"
        eligible_root = False
    else:
        tier = "ineligible"
        eligible_root = False
    return _LSRCandidate(
        path=path,
        manipulation=manipulation,
        tier=tier,
        eligible_root=eligible_root,
        connected_to_global=connected,
        nested_source=bool(connected),
        source_inventory=inventory,
        source_pool=pool,
        structural_rank=structural_rank,
        stage=_lsr_path_stage(path),
    )


def _lsr_candidate_rank(candidate: _LSRCandidate) -> tuple[object, ...]:
    timeframe_rank = {
        Timeframe.H4: 0,
        Timeframe.H1: 1,
        Timeframe.M15: 2,
        Timeframe.M5: 3,
        Timeframe.M1: 4,
    }.get(candidate.manipulation.source_timeframe, 9)
    touches = int(
        getattr(candidate.source_pool, "total_touch_count", 0)
        or len(getattr(candidate.source_pool, "touch_times", ()) or ())
    )
    pool_strength = float(
        getattr(candidate.source_pool, "strength", 0.0) or 0.0
    )
    source_age = int(
        getattr(candidate.source_pool, "age_bars", 0) or 0
    )
    return (
        _LSR_TIER_RANK[candidate.tier],
        -int(candidate.connected_to_global),
        -candidate.stage,
        timeframe_rank,
        _LSR_STRUCTURAL_RANK[candidate.structural_rank],
        -touches,
        -pool_strength,
        source_age,
        candidate.path.formed_at,
        candidate.path.sequence_id,
    )


def _select_pool_path(
    observation: MarketObservation,
    direction: Direction,
    prior: HypothesisBelief | None,
    *,
    global_context: object | None = None,
    scene_graph: TemporalMarketSceneGraph | None = None,
    required_thesis: OpenMarketThesis | None = None,
    frozen_candidate: _LSRCandidate | None = None,
) -> tuple[_LSRCandidate | None, tuple[str, ...]]:
    prior_pool_path_id = (
        None
        if prior is None
        else getattr(prior, "context_id", None)
        or getattr(prior, "setup_context_id", None)
    )
    paths = {
        path.sequence_id: path
        for path in observation.path_sequences
        if (
            path.context_kind == "pool_reversal"
            and path.direction is direction
            and (
                path.lifecycle is PathSequenceLifecycle.ACTIVE
                or path.ended_at == observation.asof
                or (
                    required_thesis is not None
                    and path.lifecycle is PathSequenceLifecycle.CLOSED
                    and path.transition_reason
                    == "pool_reversal_sequence_observed"
                )
                or (
                    prior is not None
                    and prior.phase not in _TERMINAL_PHASES
                    and prior_pool_path_id == path.sequence_id
                )
            )
        )
    }
    current_candidates = tuple(
        candidate
        for path in paths.values()
        for candidate in (
            _lsr_candidate(
                observation,
                path,
                global_context=global_context,
                scene_graph=scene_graph,
            ),
        )
        if candidate is not None
        and (
            required_thesis is None
            or candidate.manipulation.manipulation_id
            == required_thesis.root_id
        )
    )
    if frozen_candidate is not None:
        # Once reacceptance + reverse displacement established the Context,
        # the manipulation state in that causal prefix is immutable.  A later
        # reducer lifecycle rollover (for example ACCEPTED_OUTSIDE on a newer
        # bar) cannot retrospectively erase the already-observed mechanism.
        # Explicit frozen-source invalidation and sweep-extreme breach remain
        # independent Context terminals downstream.
        current_candidates = tuple(
            replace(
                candidate,
                path=frozen_candidate.path,
                manipulation=frozen_candidate.manipulation,
                authority_latched=True,
            )
            if (
                candidate.path.sequence_id
                == frozen_candidate.path.sequence_id
                and candidate.manipulation.manipulation_id
                == frozen_candidate.manipulation.manipulation_id
            )
            else candidate
            for candidate in current_candidates
        )
    retained_candidate = None
    if (
        frozen_candidate is not None
        and frozen_candidate.path.direction is direction
        and frozen_candidate.path.sequence_id not in paths
        and (
            required_thesis is None
            or frozen_candidate.manipulation.manipulation_id
            == required_thesis.root_id
        )
    ):
        retained_candidate = replace(
            frozen_candidate,
            authority_latched=True,
        )
    all_candidates = (
        *current_candidates,
        *((retained_candidate,) if retained_candidate is not None else ()),
    )
    if (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior_pool_path_id is not None
    ):
        selected = next(
            (
                candidate
                for candidate in all_candidates
                if candidate.path.sequence_id == prior_pool_path_id
            ),
            None,
        )
        authority_latched = bool(
            selected is not None
            and selected.manipulation.source_timeframe
            in {Timeframe.M15, Timeframe.M5}
            and not selected.eligible_root
            and bool(prior.hard_gate_results.get("formed_pool_sweep", False))
        )
        if authority_latched and selected is not None:
            # Root authority belongs to the frozen episode history.  A live
            # graph connection may later affect delivery/uncertainty, but its
            # absence cannot silently replace or terminalize that episode.
            selected = replace(
                selected,
                tier="B",
                eligible_root=True,
                authority_latched=True,
            )
        selected_rank = (
            _LSR_TIER_RANK["ineligible"]
            if selected is None
            else _LSR_TIER_RANK[selected.tier]
        )
        competing = tuple(
            candidate.path.sequence_id
            for candidate in sorted(all_candidates, key=_lsr_candidate_rank)
            if (
                candidate.eligible_root
                and candidate.path.sequence_id != prior_pool_path_id
                and _LSR_TIER_RANK[candidate.tier] < selected_rank
            )
        )
        return selected, competing
    terminal_cutoff = (
        prior.phase_started_at
        if prior is not None and prior.phase in _TERMINAL_PHASES
        else None
    )
    candidates = [
        candidate
        for candidate in all_candidates
        if (
            candidate.eligible_root
            and (
                candidate.path.lifecycle is PathSequenceLifecycle.ACTIVE
                or candidate.path.ended_at == observation.asof
                or (
                    required_thesis is not None
                    and candidate.path.lifecycle
                    is PathSequenceLifecycle.CLOSED
                    and candidate.path.transition_reason
                    == "pool_reversal_sequence_observed"
                )
                or candidate is retained_candidate
            )
            and (
                terminal_cutoff is None
                or candidate.path.formed_at > terminal_cutoff
                or (
                    candidate.path.sequence_id
                    in getattr(prior, "competing_episode_ids", ())
                    and candidate.path.lifecycle
                    is PathSequenceLifecycle.ACTIVE
                )
            )
        )
    ]
    if not candidates:
        return None, ()
    ordered = sorted(candidates, key=_lsr_candidate_rank)
    return ordered[0], tuple(
        candidate.path.sequence_id for candidate in ordered[1:]
    )


def _lsr_entry_episode_identity(
    manipulation_root_id: str,
    displacement_id: str,
    zone_id: str,
    direction: Direction,
) -> str:
    """Stable local Episode identity below one LSR Context Thesis."""

    raw = (
        f"{manipulation_root_id}|{displacement_id}|{zone_id}|"
        f"{direction.value}"
    )
    return (
        "lsr-entry-episode:"
        f"{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
    )


def _lsr_location_connected_to_manipulation(
    observation: MarketObservation,
    scene_graph: TemporalMarketSceneGraph | None,
    *,
    location: EntryLocationState,
    entry_path: PathSequenceState,
    pool_path: PathSequenceState,
    manipulation: ManipulationState,
) -> bool:
    """Require an exact graph closure without changing graph authority."""

    if scene_graph is None:
        # Graph-free construction remains a diagnostic/test compatibility
        # surface.  Graph-backed production updates fail closed below.
        return True
    local_episode_connected = bool(
        scene_graph.find_action_path(
            (
                location.location_id,
                location.source_zone_id,
                location.source_displacement_id,
            ),
            (entry_path.sequence_id,),
            max_depth=6,
            asof=observation.asof,
        )
    )
    pool_path_is_current = any(
        item.sequence_id == pool_path.sequence_id
        for item in observation.path_sequences
    )
    context_connected = bool(
        scene_graph.find_action_path(
            (
                pool_path.sequence_id
                if pool_path_is_current
                else manipulation.manipulation_id,
            ),
            (manipulation.manipulation_id,),
            max_depth=6,
            asof=observation.asof,
        )
    )
    # The frozen opposite-displacement step is the typed bridge between the
    # two current graph components: root -> displacement and displacement ->
    # zone.  Group5 owns that exact identity join; no direction-only or
    # temporal-proximity inference is introduced here.
    return local_episode_connected and context_connected


def _eligible_lsr_entry_locations(
    observation: MarketObservation,
    direction: Direction,
    *,
    pool_path: PathSequenceState | None,
    manipulation: ManipulationState | None,
    scene_graph: TemporalMarketSceneGraph | None,
) -> tuple[EntryLocationState, ...]:
    """Enumerate every causal zone under one frozen reverse displacement.

    Ordering is formation-clock/identity only.  Strength is deliberately not
    consulted, so a later or cosmetically stronger zone cannot displace an
    already-observed sibling Episode.
    """

    reacceptance = _path_step(pool_path, {"reacceptance_held"})
    displacement = _path_step(pool_path, {"opposite_displacement"})
    if (
        pool_path is None
        or manipulation is None
        or reacceptance is None
        or displacement is None
        or displacement.source_entity_id is None
        or displacement.source_event_id
        != manipulation.manipulation_id
    ):
        return ()
    displacement_id = displacement.source_entity_id
    displacement_active_at = getattr(
        displacement,
        "source_active_at",
        None,
    )
    eligible: list[EntryLocationState] = []
    for location in observation.entry_locations:
        if (
            location.direction is not direction
            or location.source_displacement_id != displacement_id
            or location.lifecycle is EntryLocationLifecycle.LEFT
            or location.formed_at <= reacceptance.observed_at
        ):
            continue
        source_zone = _entry_location_source_zone(observation, location)
        entry_path = _path_for_location(observation, location)
        if source_zone is None or entry_path is None:
            continue
        if (
            location.source_zone_kind == "fvg"
            and source_zone.qualification
            is not FVGQualification.DISPLACEMENT_LINKED
        ):
            continue
        # Every typed order block already carries an exact active
        # displacement/BOS qualification in its model contract.
        if (
            source_zone.direction is not direction
            or source_zone.source_displacement_id != displacement_id
            or source_zone.formed_at <= reacceptance.observed_at
            or source_zone.source_displacement_active_at is None
            or source_zone.source_displacement_active_at
            <= reacceptance.observed_at
            or (
                displacement_active_at is not None
                and source_zone.source_displacement_active_at
                != displacement_active_at
            )
            or not _lsr_location_connected_to_manipulation(
                observation,
                scene_graph,
                location=location,
                entry_path=entry_path,
                pool_path=pool_path,
                manipulation=manipulation,
            )
        ):
            continue
        eligible.append(location)
    return tuple(
        sorted(
            eligible,
            key=lambda value: (value.formed_at, value.location_id),
        )
    )


def _opposed_mss_for_displacement(
    observation: MarketObservation,
    direction: Direction,
    displacement_id: str | None,
    *,
    after: pd.Timestamp | None,
):
    """Return the first exact M5 opposed BOS/MSS for one displacement."""

    if displacement_id is None or after is None:
        return None
    candidates = tuple(
        state
        for state in observation.frame(Timeframe.M5).structure_breaks
        if (
            state.direction is direction
            and state.lifecycle is BOSLifecycle.CONFIRMED
            and state.scope is BOSScope.OPPOSED
            and state.mss_qualified
            and state.source_displacement_id == displacement_id
            and state.resolved_at is not None
            and state.resolved_at > after
        )
    )
    return (
        min(candidates, key=lambda state: (state.resolved_at, state.bos_id))
        if candidates
        else None
    )


def _lsr_optional_range_context(
    observation: MarketObservation,
    direction: Direction,
    manipulation: ManipulationState | None,
) -> _LSRRangeContext | None:
    """Join a pool reversal to a crossed mature range without gating LSR.

    Group4 keeps S/R, pool liquidity and mature-range liquidity as separate
    identities.  A pool-sourced manipulation may nevertheless record a
    mature range ID among its exact crossed sources.  This join only exposes
    that already-observed context; it never manufactures a range from the
    pool and never changes the core LSR sequence.
    """

    if (
        manipulation is None
        or manipulation.source_kind != "formed_liquidity_pool"
        or manipulation.side != direction.invalidation_side
    ):
        return None
    crossed_ids = set(manipulation.crossed_source_ids)
    ranges = tuple(
        state
        for state in observation.frame(Timeframe.H1).dealing_ranges
        if (
            state.lifecycle is DealingRangeLifecycle.MATURE
            and state.mature_at is not None
            and state.mature_at <= manipulation.swept_at
            and state.range_id in crossed_ids
        )
    )
    if not ranges:
        return None
    inventory = tuple(observation.liquidity_inventory)
    candidates: list[_LSRRangeContext] = []
    for state in ranges:
        swept_boundary = next(
            (
                item
                for item in inventory
                if (
                    item.kind == "range_boundary"
                    and item.side == manipulation.side
                    and state.range_id in item.source_ids
                    and item.confirmed_at <= manipulation.swept_at
                    and (
                        manipulation.sweep_extreme > item.upper_bound
                        if item.side == "above"
                        else manipulation.sweep_extreme < item.lower_bound
                    )
                )
            ),
            None,
        )
        if swept_boundary is None:
            continue
        opposing = next(
            (
                item
                for item in inventory
                if (
                    item.kind == "range_boundary"
                    and item.side == direction.opposing_liquidity_side
                    and item.lifecycle
                    is LiquidityInventoryLifecycle.VISIBLE
                    and state.range_id in item.source_ids
                    and item.confirmed_at <= observation.asof
                )
            ),
            None,
        )
        candidates.append(
            _LSRRangeContext(
                dealing_range=state,
                swept_boundary=swept_boundary,
                opposing_boundary=opposing,
            )
        )
    if not candidates:
        return None
    source_boundary = (
        manipulation.source_upper_bound
        if manipulation.side == "above"
        else manipulation.source_lower_bound
    )
    return min(
        candidates,
        key=lambda context: (
            abs(
                (
                    context.dealing_range.upper_bound
                    if manipulation.side == "above"
                    else context.dealing_range.lower_bound
                )
                - source_boundary
            ),
            context.dealing_range.mature_at,
            context.dealing_range.range_id,
        ),
    )


def _frozen_lsr_context(
    candidate: _LSRCandidate | None,
    direction: Direction,
) -> FrozenLSRContext | None:
    """Freeze the parent reversal mechanism without a child-zone identity."""

    if candidate is None:
        return None
    path = candidate.path
    manipulation = candidate.manipulation
    reacceptance = _path_step(path, {"reacceptance_held"})
    displacement = _path_step(path, {"opposite_displacement"})
    displacement_active_at = (
        None if displacement is None else displacement.source_active_at
    )
    if (
        reacceptance is None
        or displacement is None
        or displacement.source_event_id != manipulation.manipulation_id
        or displacement.source_entity_id is None
        or displacement_active_at is None
        or manipulation.reaccepted_at is None
        or manipulation.reaccepted_at != reacceptance.observed_at
    ):
        return None
    return FrozenLSRContext(
        manipulation_id=manipulation.manipulation_id,
        manipulation_protocol_hash=manipulation.protocol_hash,
        source_pool_id=manipulation.source_id,
        pool_path_id=path.sequence_id,
        pool_path_protocol_hash=path.protocol_hash,
        displacement_id=displacement.source_entity_id,
        direction=direction,
        swept_at=manipulation.swept_at,
        reaccepted_at=reacceptance.observed_at,
        displacement_active_at=displacement_active_at,
        displacement_observed_at=displacement.observed_at,
        sweep_extreme=manipulation.sweep_extreme,
    )


def _typed_lsr(
    observation: MarketObservation,
    direction: Direction,
    protocol: PlaybookProtocol,
    prior: HypothesisBelief | None,
    config: BrainConfig,
    *,
    global_context: GlobalMarketContext | None = None,
    scene_graph: TemporalMarketSceneGraph | None = None,
    required_thesis: OpenMarketThesis | None = None,
    required_entry_location_id: str | None = None,
    context_only: bool = False,
    frozen_context_candidate: _LSRCandidate | None = None,
) -> _Evaluation:
    required_draw_ids = (
        None
        if required_thesis is None
        else frozenset(required_thesis.draw_candidate_ids)
    )
    selected_candidate, competing_episode_ids = _select_pool_path(
        observation,
        direction,
        prior,
        global_context=global_context,
        scene_graph=scene_graph,
        required_thesis=required_thesis,
        frozen_candidate=frozen_context_candidate,
    )
    pool_path = (
        None if selected_candidate is None else selected_candidate.path
    )
    manipulation = (
        None
        if selected_candidate is None
        else selected_candidate.manipulation
    )
    authority_root_allowed = bool(
        selected_candidate is not None
        and selected_candidate.eligible_root
    )
    range_context = _lsr_optional_range_context(
        observation,
        direction,
        manipulation,
    )
    pool_swept = _path_step(pool_path, {"pool_swept"})
    sweep_return = _path_step(pool_path, {"reacceptance_held"})
    opposite_displacement = _path_step(
        pool_path, {"opposite_displacement"}
    )
    pool_ambiguity = _path_step(
        pool_path,
        {
            "opposite_displacement_ambiguous",
            "micro_bos_ambiguous",
        },
    )
    pool_contradiction = _path_step(
        pool_path,
        {
            "accepted_outside",
            "reacceptance_failed",
            "micro_bos_opposed",
        },
    )
    pool_acceptance_failure = _path_step(
        pool_path,
        {"accepted_outside"},
    )
    return_clock = None if sweep_return is None else sweep_return.observed_at
    displacement_id = (
        None
        if opposite_displacement is None
        else opposite_displacement.source_entity_id
    )
    opposed_mss = _opposed_mss_for_displacement(
        observation,
        direction,
        displacement_id,
        after=return_clock,
    )
    eligible_locations = _eligible_lsr_entry_locations(
        observation,
        direction,
        pool_path=pool_path,
        manipulation=manipulation,
        scene_graph=scene_graph,
    )
    eligible_by_id = {
        item.location_id: item for item in eligible_locations
    }
    frozen_location = (
        None
        if prior is None or prior.entry_location_id is None
        else next(
            (
                item
                for item in observation.entry_locations
                if item.location_id == prior.entry_location_id
                and item.direction is direction
                and item.source_displacement_id == displacement_id
                and (
                    required_entry_location_id is None
                    or item.location_id == required_entry_location_id
                )
            ),
            None,
        )
    )
    location = (
        None
        if context_only
        else frozen_location
        if frozen_location is not None
        else eligible_by_id.get(required_entry_location_id)
        if required_entry_location_id is not None
        else eligible_locations[0]
        if eligible_locations
        else None
    )
    if location is not None and (
        return_clock is None
        or location.formed_at <= return_clock
        or location.source_displacement_id != displacement_id
        or (
            required_entry_location_id is not None
            and location.location_id != required_entry_location_id
        )
    ):
        location = None
    entry_path = _path_for_location(observation, location)
    context_identity = (
        None if pool_path is None else pool_path.sequence_id
    )
    episode_identity = (
        _lsr_entry_episode_identity(
            manipulation.manipulation_id,
            displacement_id,
            location.source_zone_id,
            direction,
        )
        if (
            manipulation is not None
            and displacement_id is not None
            and location is not None
        )
        else prior.episode_id
        if (
            not context_only
            and prior is not None
            and prior.context_id == context_identity
        )
        else None
    )
    setup_identity = episode_identity or context_identity
    first_pullback = _path_step(entry_path, {"first_pullback"})
    wick_trigger = _path_step(entry_path, {"wick_rejection"})
    held_trigger = _path_step(entry_path, {"reacceptance_held"})
    micro_trigger = _path_step(entry_path, {"micro_bos_confirmed"})
    qualified_micro = next(
        (
            reference
            for reference in observation.micro_bos_references
            if (
                location is not None
                and reference.context_kind == "zone_return"
                and reference.context_id == location.location_id
                and reference.expected_direction is direction
                and reference.qualified
                and first_pullback is not None
                and reference.resolved_at
                > first_pullback.observed_at
                and micro_trigger is not None
                and reference.bos_id == micro_trigger.source_event_id
                and reference.target_swing_id
                == micro_trigger.source_entity_id
                and reference.resolved_at == micro_trigger.observed_at
                and _micro_reference_has_exact_source(
                    observation,
                    reference,
                )
            )
        ),
        None,
    )
    qualified_reacceptance = next(
        (
            state
            for state in observation.qualified_reacceptances
            if (
                location is not None
                and held_trigger is not None
                and state.reacceptance_id
                == held_trigger.source_entity_id
                and state.context_kind == "entry_zone"
                and state.context_id == location.location_id
                and state.direction is direction
                and state.lifecycle
                is QualifiedReacceptanceLifecycle.HELD
                and state.held_at == held_trigger.observed_at
            )
        ),
        None,
    )
    entry_ambiguity = _path_step(
        entry_path,
        {"micro_bos_ambiguous"},
    )
    entry_contradiction = _path_step(
        entry_path,
        {
            "location_left",
            "reacceptance_failed",
            "micro_bos_opposed",
        },
    )
    valid_triggers = tuple(
        step
        for step, qualified in (
            (
                wick_trigger,
                wick_trigger is not None
                and location is not None
                and wick_trigger.source_entity_id
                == location.location_id,
            ),
            (
                held_trigger,
                qualified_reacceptance is not None,
            ),
            (micro_trigger, qualified_micro is not None),
        )
        if (
            step is not None
            and qualified
            and first_pullback is not None
            and step.direction is direction
            # A completed one-minute bar cannot prove a same-bar wick came
            # after the first contact.  Every trigger family is strict-after.
            and step.observed_at > first_pullback.observed_at
        )
    )
    candidate_setup_id = setup_identity
    selected_trigger, trigger_step = _select_frozen_trigger(
        valid_triggers,
        prior=prior,
        setup_id=candidate_setup_id,
        entry_path_id=(
            None if entry_path is None else entry_path.sequence_id
        ),
        entry_location_id=(
            None if location is None else location.location_id
        ),
        direction=direction,
        trigger_must_follow=(
            None
            if first_pullback is None
            else first_pullback.observed_at
        ),
        require_strictly_after=True,
    )
    trigger_ready = bool(
        first_pullback is not None
        and trigger_step is not None
        and entry_ambiguity is None
        and entry_contradiction is None
        and location is not None
        and location.lifecycle is not EntryLocationLifecycle.LEFT
        and entry_path is not None
        and entry_path.lifecycle is not PathSequenceLifecycle.CENSORED
    )
    initiating_event_id = (
        None
        if manipulation is None
        else manipulation.manipulation_id
    )
    invalidation = (
        StructuralLevel(
            price=manipulation.sweep_extreme,
            side=direction.invalidation_side,
            source_level_id=manipulation.manipulation_id,
            observed_at=manipulation.swept_at,
            rationale=(
                "exact original formed-pool sweep extreme"
            ),
        )
        if manipulation is not None
        else None
    )
    prior_plan_same_setup = bool(
        prior is not None
        and prior.plan is not None
        and prior.plan.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and prior.plan.setup_id == setup_identity
        and location is not None
        and prior.plan.entry_location_id == location.location_id
    )
    lsr_context = (
        prior.plan.lsr_context
        if (
            prior_plan_same_setup
            and prior is not None
            and prior.plan is not None
            and prior.plan.lsr_context is not None
        )
        else _frozen_lsr_context(selected_candidate, direction)
    )
    prior_route = (
        prior.liquidity_route
        if (
            prior is not None
            and prior.phase not in _TERMINAL_PHASES
            and prior.setup_context_id == setup_identity
        )
        else None
    )
    prior_context_draw_was_optional_range = bool(
        prior_route is not None
        and prior_route.range_context_id is not None
        and prior_route.context_draw_id
        == prior_route.opposing_range_boundary_id
    )
    preferred_context_draw_id = (
        prior.thesis_draw.level_id
        if (
            prior is not None
            and prior.phase not in _TERMINAL_PHASES
            and prior.episode_id == episode_identity
            and prior.thesis_draw is not None
        )
        else None
        if prior_route is None or prior_context_draw_was_optional_range
        else prior_route.context_draw_id
    )
    prior_is_execution_owner = bool(
        prior_plan_same_setup
        and prior is not None
        and prior.plan is not None
        and prior.candidate_id is not None
        and prior.context_metadata.get("lsr_context_execution_owner")
        == prior.candidate_id
    )
    # Target selection remains descriptive until this child becomes the first
    # uniquely fully executable Episode.  Thereafter the exact frozen target
    # is required; a later, stronger draw cannot rewrite the owner plan.
    preferred_primary_target_id = (
        prior.plan.selected_draw_id
        if prior_is_execution_owner
        and prior is not None
        and prior.plan is not None
        else None
    )
    planned_entry = (
        float(observation.price)
        if location is None
        else float(
            location.contact_reference_price
            if location.contact_reference_price is not None
            else location.near_edge
        )
    )
    contextual_draw_id = (
        preferred_context_draw_id
        if preferred_context_draw_id is not None
        else (
            None
            if (
                range_context is None
                or range_context.opposing_boundary is None
            )
            else range_context.opposing_boundary.item_id
        )
    )
    context_draw = _select_target(
        observation,
        direction,
        float(observation.price),
        config,
        preferred_id=contextual_draw_id,
        require_preferred=preferred_context_draw_id is not None,
        allowed_ids=required_draw_ids,
    )
    frozen_context_draw_missing = bool(
        preferred_context_draw_id is not None and context_draw is None
    )
    countertrend_lsr = bool(
        global_context is not None
        and _higher_authority_opposes(global_context, direction)
    )
    target_selection_failure_reason: str | None = None
    authority_barrier_id: str | None = None
    authority_barrier_price: float | None = None
    primary_target = None
    if location is not None:
        if countertrend_lsr:
            countertrend_selection = _select_countertrend_lsr_target_result(
                observation,
                direction,
                planned_entry,
                config,
                global_context,
                context_draw=context_draw,
                preferred_id=preferred_primary_target_id,
                allowed_ids=required_draw_ids,
                location=location,
                invalidation=invalidation,
            )
            primary_target = countertrend_selection.target
            if not frozen_context_draw_missing:
                context_draw = countertrend_selection.context_draw
            target_selection_failure_reason = (
                countertrend_selection.failure_reason
            )
            authority_barrier_id = (
                countertrend_selection.authority_barrier_id
            )
            authority_barrier_price = (
                countertrend_selection.authority_barrier_price
            )
        else:
            primary_target = _select_primary_deliverable_target(
                observation,
                direction,
                planned_entry,
                config,
                context_draw=context_draw,
                preferred_id=preferred_primary_target_id,
                allowed_ids=required_draw_ids,
            )
            if primary_target is None:
                target_selection_failure_reason = "visible_draw_missing"
    if (
        prior_is_execution_owner
        and preferred_primary_target_id is not None
        and (
            primary_target is None
            or primary_target.level_id != preferred_primary_target_id
        )
    ):
        primary_target = None
        target_selection_failure_reason = "frozen_primary_target_unavailable"
    if frozen_context_draw_missing:
        target_selection_failure_reason = "context_draw_consumed"
        primary_target = None
    selected_planned_entry = _select_planned_entry(
        observation,
        direction,
        location,
        invalidation,
        primary_target,
        config,
    )
    if selected_planned_entry is not None:
        planned_entry = selected_planned_entry
    draw_selection = _draw_selection(
        observation,
        primary_target if manipulation is not None else None,
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        prior=prior,
    )
    liquidity_route = (
        None
        if location is None
        else _liquidity_route(
            observation,
            direction,
            planned_entry,
            context_draw=context_draw,
            primary_target=primary_target,
            prior_route=prior_route,
            range_context=range_context,
            conservative_contacts=countertrend_lsr,
            authority_barrier_id=authority_barrier_id,
            authority_barrier_price=authority_barrier_price,
            tick_size=config.tick_size,
        )
    )
    current_plan = _typed_plan(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        direction=direction,
        observation=observation,
        config=config,
        setup_id=setup_identity,
        location=location,
        entry_path=entry_path,
        invalidation=invalidation,
        target=primary_target,
        draw_selection=draw_selection,
        lsr_context=lsr_context,
        liquidity_route=liquidity_route,
        planned_entry=selected_planned_entry,
    )
    # The Context-level execution-owner latch below freezes the complete plan
    # only when trigger, hard gates and feasibility first coexist.  Before
    # that causal clock, a waiting Episode may still observe a changing route.
    plan = current_plan
    remaining_ok = bool(
        plan is not None
        and plan.remaining_path_R >= config.minimum_remaining_path_R
    )
    execution_missing = _execution_unavailable(observation)
    context_established = bool(
        (
            frozen_context_candidate is not None
            and frozen_context_candidate.manipulation.lifecycle
            is ManipulationLifecycle.REACCEPTED
            and _path_step(
                frozen_context_candidate.path,
                {"reacceptance_held"},
            )
            is not None
            and _path_step(
                frozen_context_candidate.path,
                {"opposite_displacement"},
            )
            is not None
        )
        or (
            prior is not None
            and prior.context_thesis_id is not None
            and prior.hard_gate_results.get(
                "outside_acceptance_failed",
                False,
            )
            and prior.context_metadata.get(
                "lsr_displacement_id",
                "none",
            )
            not in {"", "none"}
        )
    )
    current_accepted_outside = bool(
        pool_acceptance_failure is not None
        or (
            manipulation is not None
            and manipulation.lifecycle
            is ManipulationLifecycle.ACCEPTED_OUTSIDE
        )
    )
    # Accepted outside rejects a sweep before the reversal Context exists.
    # It cannot retrospectively erase an already frozen reacceptance and
    # reverse displacement when Group4 later rolls its bounded current view.
    accepted_outside = bool(
        current_accepted_outside and not context_established
    )
    zone_left = bool(
        location is not None
        and location.lifecycle is EntryLocationLifecycle.LEFT
    )
    stale_trigger = bool(
        entry_path is not None
        and entry_path.lifecycle is PathSequenceLifecycle.CLOSED
        and entry_path.ended_at != observation.asof
    )
    frozen_pool_missing = bool(
        observation.group5_typed_available
        and prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.setup_context_id is not None
        and pool_path is None
    )
    frozen_location_missing = bool(
        observation.group5_typed_available
        and prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.entry_location_id is not None
        and location is None
    )
    hard_gates = {
        "formed_pool_sweep": bool(
            pool_path is not None
            and pool_swept is not None
            and manipulation is not None
            and authority_root_allowed
        ),
        "outside_acceptance_failed": bool(
            sweep_return is not None
            and manipulation is not None
            and (
                (
                    manipulation.lifecycle
                    is ManipulationLifecycle.REACCEPTED
                    and manipulation.reaccepted_at
                    == sweep_return.observed_at
                )
                or (
                    context_established
                    and lsr_context is not None
                    and lsr_context.reaccepted_at
                    == sweep_return.observed_at
                )
            )
            and sweep_return.source_event_id
            == manipulation.manipulation_id
            and sweep_return.source_entity_id
            == manipulation.manipulation_id
            and not accepted_outside
            and pool_ambiguity is None
        ),
        "reverse_displacement_zone": bool(
            opposite_displacement is not None
            and location is not None
            and entry_path is not None
            and location.source_displacement_id == displacement_id
        ),
        "reversal_first_pullback": first_pullback is not None,
        "typed_entry_trigger": trigger_ready,
    }
    support = {
        "pool_liquidity_swept": float(
            hard_gates["formed_pool_sweep"]
        ),
        "pool_return_confirmed": float(
            hard_gates["outside_acceptance_failed"]
        ),
        "reverse_displacement_zone": float(
            hard_gates["reverse_displacement_zone"]
        ),
        "opposed_mss_confirmation": float(opposed_mss is not None),
        "reversal_first_pullback": float(
            hard_gates["reversal_first_pullback"]
        ),
        "typed_entry_trigger": float(trigger_ready),
        "mature_range_context_visible": (
            0.0
            if range_context is None
            else clamp(range_context.dealing_range.strength)
        ),
        "opposing_range_boundary_visible": float(
            range_context is not None
            and range_context.opposing_boundary is not None
        ),
        "opposing_draw_visible": float(context_draw is not None),
        "remaining_path_available": float(remaining_ok),
        "execution_fillability": (
            0.0
            if execution_missing
            else observation.execution.fillability
        ),
    }
    contradict = {
        "accepted_outside_or_failed": float(accepted_outside),
        "reverse_displacement_missing": float(
            sweep_return is not None
            and opposite_displacement is None
        ),
        "entry_zone_left": float(zone_left),
        "micro_bos_opposed": float(
            entry_contradiction is not None
        ),
        "draw_consumed_or_missing": float(
            preferred_context_draw_id is not None
            and context_draw is None
        ),
        "remaining_path_consumed": float(
            plan is not None and not remaining_ok
        ),
        "execution_unavailable": float(execution_missing),
    }
    group_scores = {
        "structure": support["pool_return_confirmed"]
        * (1.0 - contradict["accepted_outside_or_failed"]),
        "displacement": support["reverse_displacement_zone"]
        * (1.0 - contradict["reverse_displacement_missing"]),
        "location": _location_quality(location)
        * (1.0 - contradict["entry_zone_left"]),
        "liquidity": min(
            support["pool_liquidity_swept"],
            support["opposing_draw_visible"],
        )
        * (1.0 - contradict["draw_consumed_or_missing"]),
        "trigger": support["typed_entry_trigger"]
        * (1.0 - contradict["micro_bos_opposed"]),
        "execution": support["execution_fillability"]
        * (1.0 - contradict["execution_unavailable"]),
    }
    pool_terminal_failure = bool(
        pool_path is not None
        and (
            pool_path.lifecycle is PathSequenceLifecycle.CENSORED
            or (
                not context_established
                and pool_path.lifecycle is PathSequenceLifecycle.CLOSED
                and pool_path.transition_reason
                in {
                    "accepted_outside",
                    "micro_bos_opposed",
                    "manipulation_resolution_deadline",
                }
            )
        )
    )
    entry_path_censored = bool(
        entry_path is not None
        and entry_path.lifecycle is PathSequenceLifecycle.CENSORED
    )
    terminal_reason: str | None = None
    terminal_source_ids: tuple[str, ...] = ()
    if accepted_outside:
        terminal_reason = "accepted_outside_or_failed"
        terminal_source_ids = _identity_tuple(
            None if pool_path is None else pool_path.sequence_id,
            None
            if pool_acceptance_failure is None
            else pool_acceptance_failure.source_entity_id,
            None
            if pool_acceptance_failure is None
            else pool_acceptance_failure.source_event_id,
        )
    elif zone_left:
        terminal_reason = "entry_zone_left"
        terminal_source_ids = _identity_tuple(
            None if location is None else location.location_id
        )
    elif entry_contradiction is not None:
        terminal_reason = "micro_bos_opposed"
        terminal_source_ids = _identity_tuple(
            entry_contradiction.source_entity_id,
            entry_contradiction.source_event_id,
        )
    elif pool_terminal_failure:
        terminal_reason = (
            "pool_path_censored"
            if pool_path is None
            else pool_path.transition_reason
        )
        terminal_source_ids = _identity_tuple(
            None if pool_path is None else pool_path.sequence_id
        )
    elif entry_path_censored:
        terminal_reason = "entry_path_censored"
        terminal_source_ids = _identity_tuple(
            None if entry_path is None else entry_path.sequence_id
        )
    elif stale_trigger:
        terminal_reason = "entry_window_expired"
        terminal_source_ids = _identity_tuple(
            None if entry_path is None else entry_path.sequence_id
        )
    base_thesis_target = min(
        (
            0.0
            if manipulation is None
            else clamp(manipulation.strength)
        ),
        (
            0.0
            if sweep_return is None
            else clamp(sweep_return.strength)
        ),
    )
    thesis_target = (
        base_thesis_target
        if opposed_mss is None
        else max(
            base_thesis_target,
            (
                base_thesis_target
                + clamp(opposed_mss.strength)
            )
            / 2.0,
        )
    )
    return _Evaluation(
        evidence=_typed_evidence_items(
            protocol,
            observation,
            support,
            contradict,
        ),
        trigger_ready=trigger_ready,
        setup_clock=(
            None if pool_swept is None else pool_swept.observed_at
        ),
        sequence_signals={
            "formed_pool_sweep": _SequenceSignal(
                float(hard_gates["formed_pool_sweep"]),
                None if pool_swept is None else pool_swept.observed_at,
                ()
                if pool_path is None
                else (
                    pool_path.sequence_id,
                    pool_path.context_id,
                ),
            ),
            "outside_acceptance_failed": _SequenceSignal(
                float(hard_gates["outside_acceptance_failed"]),
                return_clock,
                ()
                if sweep_return is None
                else (sweep_return.source_entity_id,),
            ),
            "reverse_displacement_zone": _SequenceSignal(
                float(hard_gates["reverse_displacement_zone"]),
                (
                    None
                    if location is None
                    else location.formed_at
                ),
                ()
                if (
                    location is None
                    or opposite_displacement is None
                )
                else (
                    opposite_displacement.step_id,
                    opposite_displacement.source_event_id,
                    opposite_displacement.source_entity_id,
                    location.location_id,
                    location.source_displacement_id,
                    location.source_zone_id,
                    *(
                        ()
                        if opposed_mss is None
                        else (
                            opposed_mss.bos_id,
                            opposed_mss.target_swing_id,
                        )
                    ),
                ),
            ),
            "reversal_first_pullback": _SequenceSignal(
                float(first_pullback is not None),
                (
                    None
                    if first_pullback is None
                    else first_pullback.observed_at
                ),
                ()
                if first_pullback is None
                else (first_pullback.source_entity_id,),
            ),
            "typed_entry_trigger": _SequenceSignal(
                float(trigger_ready),
                (
                    None
                    if selected_trigger is None
                    else selected_trigger.observed_at
                ),
                ()
                if selected_trigger is None
                else _identity_tuple(
                    selected_trigger.entry_path_id,
                    selected_trigger.trigger_id,
                    selected_trigger.source_entity_id,
                    selected_trigger.source_event_id,
                ),
            ),
        },
        setup_identity=setup_identity,
        context_identity=context_identity,
        episode_identity=episode_identity,
        initiating_event_id=initiating_event_id,
        entry_location_id=(
            None if location is None else location.location_id
        ),
        entry_path_id=(
            None if entry_path is None else entry_path.sequence_id
        ),
        plan=plan,
        invalidation=invalidation,
        selected_draw=primary_target,
        # LSR has no child-dependent Context draw.  The selected delivery
        # target remains on the child plan/route only.
        thesis_draw=None,
        draw_selection=draw_selection,
        liquidity_route=liquidity_route,
        thesis_target=thesis_target,
        location_quality=group_scores["location"],
        entry_readiness=(
            0.0
            if not trigger_ready or trigger_step is None
            else clamp(trigger_step.strength)
        ),
        delivery_quality=_delivery_quality(
            observation,
            direction,
            plan,
        ),
        **_typed_market_uncertainty(
            observation,
            {
                name: value
                for name, value in support.items()
                if name
                not in {
                    "mature_range_context_visible",
                    "opposing_range_boundary_visible",
                }
            },
            contradict,
            graph_ambiguity=float(
                pool_ambiguity is not None
                or entry_ambiguity is not None
            ),
        ),
        ambiguity_count=(
            int(pool_ambiguity is not None)
            + int(entry_ambiguity is not None)
        ),
        evidence_group_scores=group_scores,
        hard_gate_results=hard_gates,
        invalidated=bool(
            accepted_outside
            or zone_left
            or entry_contradiction is not None
            or pool_terminal_failure
            or entry_path_censored
        ),
        entry_window_expired=stale_trigger,
        terminal_reason=terminal_reason,
        terminal_source_ids=terminal_source_ids,
        selected_trigger=selected_trigger,
        target_selection_failure_reason=target_selection_failure_reason,
        lsr_displacement_id=displacement_id,
        lsr_entry_zone_id=(
            None if location is None else location.source_zone_id
        ),
        lsr_pool_path_id=(
            None if pool_path is None else pool_path.sequence_id
        ),
        authority_tier=(
            None if selected_candidate is None else selected_candidate.tier
        ),
        authority_source_timeframe=(
            None
            if selected_candidate is None
            else selected_candidate.manipulation.source_timeframe.value
        ),
        authority_structural_rank=(
            None
            if selected_candidate is None
            else selected_candidate.structural_rank
        ),
        global_context_connected=bool(
            selected_candidate is not None
            and selected_candidate.connected_to_global
        ),
        source_nested=bool(
            selected_candidate is not None
            and selected_candidate.nested_source
        ),
        authority_latched=bool(
            selected_candidate is not None
            and selected_candidate.authority_latched
        ),
        competing_episode_ids=competing_episode_ids,
    )


def _select_favr_context(
    observation: MarketObservation,
    direction: Direction,
    prior: HypothesisBelief | None,
    required_thesis: OpenMarketThesis | None = None,
):
    ranges = {
        state.range_id: state
        for state in observation.frame(Timeframe.H1).dealing_ranges
    }
    range_manipulations = tuple(
        state
        for state in observation.manipulations
        if state.source_kind == "mature_range_boundary"
        and (
            required_thesis is None
            or state.manipulation_id == required_thesis.root_id
        )
    )
    if (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.initiating_event_id is not None
    ):
        manipulation = next(
            (
                state
                for state in range_manipulations
                if state.manipulation_id == prior.initiating_event_id
            ),
            None,
        )
        return (
            None if manipulation is None else ranges.get(manipulation.source_id),
            manipulation,
        )
    expected_side = "below" if direction is Direction.LONG else "above"
    terminal_cutoff = (
        prior.terminal_at
        if prior is not None and prior.phase in _TERMINAL_PHASES
        else None
    )
    candidates = tuple(
        state
        for state in range_manipulations
        if (
            state.side == expected_side
            and state.source_id in ranges
            and (
                terminal_cutoff is None
                or state.swept_at > terminal_cutoff
            )
        )
    )
    if candidates:
        manipulation = min(
            candidates,
            key=lambda state: (state.swept_at, state.manipulation_id),
        )
        return ranges[manipulation.source_id], manipulation
    manipulated_range_ids = {
        state.source_id for state in range_manipulations
    }
    mature = tuple(
        state
        for state in ranges.values()
        if (
            state.lifecycle is DealingRangeLifecycle.MATURE
            and state.range_id not in manipulated_range_ids
        )
    )
    if not mature:
        return None, None
    selected = max(
        mature,
        key=lambda state: (state.mature_at, state.range_id),
    )
    return selected, None


def _entry_location_source_zone(
    observation: MarketObservation,
    location: EntryLocationState,
):
    """Resolve the exact current M5 zone frozen into an entry location."""

    frame = observation.frame(Timeframe.M5)
    candidates = (
        tuple(
            state
            for state in frame.fair_value_gaps
            if state.fvg_id == location.source_zone_id
        )
        if location.source_zone_kind == "fvg"
        else tuple(
            state
            for state in frame.order_blocks
            if state.order_block_id == location.source_zone_id
        )
        if location.source_zone_kind == "order_block"
        else ()
    )
    if len(candidates) != 1:
        return None
    source = candidates[0]
    source_bos_id = (
        None
        if location.source_zone_kind == "fvg"
        else source.source_bos_id
    )
    source_failed = (
        source.lifecycle is FairValueGapLifecycle.INVALIDATED
        if location.source_zone_kind == "fvg"
        else source.lifecycle is OrderBlockLifecycle.FAILED
    )
    if (
        source_failed
        or source.protocol_hash
        != location.source_group3_protocol_hash
        or source.protocol_hash != location.source_zone_protocol_hash
        or source.symbol != observation.symbol
        or source.symbol != location.symbol
        or source.instrument_id != observation.instrument_id
        or source.instrument_id != location.instrument_id
        or source.timeframe is not Timeframe.M5
        or source.source_displacement_id
        != location.source_displacement_id
        or source_bos_id != location.source_bos_id
        or source.direction is not location.direction
        or not math.isclose(
            source.lower_bound,
            location.lower_bound,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            source.upper_bound,
            location.upper_bound,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            source.midpoint,
            location.midpoint,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            source.invalidation_price,
            location.failure_boundary,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or source.confirmed_at != location.formed_at
    ):
        return None
    return source


def _select_favr_location(
    observation: MarketObservation,
    direction: Direction,
    dealing_range,
    manipulation,
    prior: HypothesisBelief | None,
) -> EntryLocationState | None:
    if (
        dealing_range is None
        or manipulation is None
        or manipulation.lifecycle is not ManipulationLifecycle.REACCEPTED
        or manipulation.reaccepted_at is None
    ):
        return None
    by_id = {
        location.location_id: location
        for location in observation.entry_locations
    }

    def eligible(location: EntryLocationState) -> bool:
        source = _entry_location_source_zone(observation, location)
        return bool(
            source is not None
            and source.source_displacement_active_at is not None
            and location.direction is direction
            and location.lifecycle is not EntryLocationLifecycle.LEFT
            and location.formed_at > manipulation.reaccepted_at
            and source.source_displacement_active_at
            > manipulation.reaccepted_at
            and dealing_range.lower_bound <= location.lower_bound
            and location.upper_bound <= dealing_range.upper_bound
            and (
                location.midpoint < dealing_range.midpoint
                if direction is Direction.LONG
                else location.midpoint > dealing_range.midpoint
            )
            and _path_for_location(observation, location) is not None
        )

    if (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.entry_location_id is not None
    ):
        bound = by_id.get(prior.entry_location_id)
        return bound if bound is not None and eligible(bound) else None
    candidates = tuple(
        location
        for location in observation.entry_locations
        if eligible(location)
    )
    return (
        min(
            candidates,
            key=lambda location: (location.formed_at, location.location_id),
        )
        if candidates
        else None
    )


def _favr_opposite_boundary_target(
    observation: MarketObservation,
    dealing_range,
    direction: Direction,
) -> LiquidityLevel | None:
    if dealing_range is None:
        return None
    side = "above" if direction is Direction.LONG else "below"
    expected_price = (
        dealing_range.upper_bound
        if direction is Direction.LONG
        else dealing_range.lower_bound
    )
    matches = tuple(
        item
        for item in observation.liquidity_inventory
        if (
            item.kind == "range_boundary"
            and item.side == side
            and item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
            and dealing_range.range_id in item.source_ids
            and math.isclose(
                item.price,
                expected_price,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and item.confirmed_at <= observation.asof
        )
    )
    if len(matches) != 1:
        return None
    item = matches[0]
    return LiquidityLevel(
        level_id=item.item_id,
        timeframe=item.timeframe,
        side=item.side,
        price=item.price,
        formed_at=item.formed_at,
        confirmed_at=item.confirmed_at,
        touches=max(0, len(item.source_ids) - 1),
        swept=False,
    )


def _typed_favr(
    observation: MarketObservation,
    direction: Direction,
    protocol: PlaybookProtocol,
    prior: HypothesisBelief | None,
    config: BrainConfig,
    *,
    required_thesis: OpenMarketThesis | None = None,
) -> _Evaluation:
    dealing_range, manipulation = _select_favr_context(
        observation,
        direction,
        prior,
        required_thesis,
    )
    prior_episode_live = bool(
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.setup_context_id is not None
    )
    observed_range_ids = {
        state.range_id
        for state in observation.frame(Timeframe.H1).dealing_ranges
    }
    observed_manipulation_ids = {
        state.manipulation_id for state in observation.manipulations
    }
    observed_location_ids = {
        state.location_id for state in observation.entry_locations
    }
    frozen_range_missing = bool(
        observation.frame(Timeframe.H1).ready
        and prior_episode_live
        and prior is not None
        and prior.context_id is not None
        and prior.context_id not in observed_range_ids
    )
    frozen_manipulation_missing = bool(
        observation.frame(Timeframe.H1).ready
        and prior_episode_live
        and prior is not None
        and prior.initiating_event_id is not None
        and prior.initiating_event_id not in observed_manipulation_ids
    )
    mature = bool(
        dealing_range is not None
        and dealing_range.lifecycle is DealingRangeLifecycle.MATURE
        and dealing_range.mature_at is not None
    )
    range_reaccepted = bool(
        mature
        and manipulation is not None
        and manipulation.source_id == dealing_range.range_id
        and manipulation.lifecycle is ManipulationLifecycle.REACCEPTED
        and manipulation.reaccepted_at is not None
        and manipulation.reentry_price is not None
    )
    accepted_outside = bool(
        manipulation is not None
        and manipulation.lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE
    )
    resolution_censored = bool(
        manipulation is not None
        and manipulation.lifecycle is ManipulationLifecycle.SWEPT
        and manipulation.deadline_elapsed
        and manipulation.censored_at is not None
    )
    location = _select_favr_location(
        observation,
        direction,
        dealing_range,
        manipulation,
        prior,
    )
    frozen_location_missing = bool(
        observation.group5_typed_available
        and prior_episode_live
        and prior is not None
        and prior.entry_location_id is not None
        and (
            prior.entry_location_id not in observed_location_ids
            or location is None
        )
    )
    entry_path = _path_for_location(observation, location)
    return_mss = _opposed_mss_for_displacement(
        observation,
        direction,
        None if location is None else location.source_displacement_id,
        after=(
            None if manipulation is None else manipulation.reaccepted_at
        ),
    )
    first_pullback = _path_step(entry_path, {"first_pullback"})
    mss_after_first_pullback = bool(
        first_pullback is not None
        and return_mss is not None
        and return_mss.resolved_at > first_pullback.observed_at
    )
    held_trigger = _path_step(entry_path, {"reacceptance_held"})
    micro_trigger = _path_step(entry_path, {"micro_bos_confirmed"})
    qualified_micro = next(
        (
            reference
            for reference in observation.micro_bos_references
            if (
                location is not None
                and first_pullback is not None
                and micro_trigger is not None
                and reference.context_kind == "zone_return"
                and reference.context_id == location.location_id
                and reference.expected_direction is direction
                and reference.qualified
                and reference.resolved_at > first_pullback.observed_at
                and reference.bos_id == micro_trigger.source_event_id
                and reference.target_swing_id
                == micro_trigger.source_entity_id
                and reference.resolved_at == micro_trigger.observed_at
                and _micro_reference_has_exact_source(
                    observation,
                    reference,
                )
            )
        ),
        None,
    )
    trigger_step = (
        micro_trigger if qualified_micro is not None else held_trigger
    )
    entry_ambiguity = _path_step(
        entry_path,
        {"micro_bos_ambiguous"},
    )
    entry_contradiction = _path_step(
        entry_path,
        {
            "location_left",
            "reacceptance_failed",
            "micro_bos_opposed",
        },
    )
    trigger_ready = bool(
        first_pullback is not None
        and trigger_step is not None
        and entry_ambiguity is None
        and entry_contradiction is None
    )
    setup_identity = (
        manipulation.manipulation_id
        if manipulation is not None
        else None if dealing_range is None else dealing_range.range_id
    )
    setup_clock = (
        manipulation.swept_at
        if manipulation is not None
        else None if dealing_range is None else dealing_range.mature_at
    )
    episode_identity = (
        None if manipulation is None else manipulation.manipulation_id
    )
    invalidation = (
        StructuralLevel(
            price=manipulation.sweep_extreme,
            side=direction.invalidation_side,
            source_level_id=manipulation.manipulation_id,
            observed_at=manipulation.swept_at,
            rationale="exact original mature-range sweep extreme",
        )
        if manipulation is not None
        else None
    )
    target = (
        _favr_opposite_boundary_target(
            observation,
            dealing_range,
            direction,
        )
        if range_reaccepted
        else None
    )
    draw_selection = _draw_selection(
        observation,
        target,
        playbook=Playbook.FAILED_AUCTION_VALUE_RETURN,
        prior=prior,
    )
    if (
        draw_selection is not None
        and (
            prior is None
            or prior.draw_selection != draw_selection
        )
    ):
        draw_selection = replace(
            draw_selection,
            selection_reason=(
                "failed_auction_value_return:"
                "frozen_opposite_range_boundary"
            ),
        )
    planned_entry = (
        None
        if location is None
        else location.contact_reference_price
        if location.contact_reference_price is not None
        else location.near_edge
    )
    selected_planned_entry = _select_planned_entry(
        observation,
        direction,
        location,
        invalidation,
        target,
        config,
    )
    if selected_planned_entry is not None:
        planned_entry = selected_planned_entry
    midpoint_chase = bool(
        dealing_range is not None
        and planned_entry is not None
        and (
            planned_entry >= dealing_range.midpoint
            if direction is Direction.LONG
            else planned_entry <= dealing_range.midpoint
        )
    )
    range_auction = (
        FrozenRangeAuctionContext(
            range_id=dealing_range.range_id,
            manipulation_id=manipulation.manipulation_id,
            lower_bound=dealing_range.lower_bound,
            upper_bound=dealing_range.upper_bound,
            midpoint=dealing_range.midpoint,
            value_price=dealing_range.value_price,
            mature_at=dealing_range.mature_at,
            manipulation_side=manipulation.side,
            swept_at=manipulation.swept_at,
            manipulation_extreme=manipulation.sweep_extreme,
            reentry_candidate_at=manipulation.reentry_candidate_at,
            reentered_at=manipulation.reaccepted_at,
            reentry_price=manipulation.reentry_price,
            opposite_liquidity_id=target.level_id,
        )
        if (
            range_reaccepted
            and target is not None
            and draw_selection is not None
        )
        else None
    )
    liquidity_route = (
        None
        if planned_entry is None
        else _liquidity_route(
            observation,
            direction,
            float(planned_entry),
            context_draw=target,
            primary_target=target,
            prior_route=(
                None if prior is None else prior.liquidity_route
            ),
            tick_size=config.tick_size,
        )
    )
    plan = (
        None
        if midpoint_chase
        else _typed_plan(
            playbook=Playbook.FAILED_AUCTION_VALUE_RETURN,
            direction=direction,
            observation=observation,
            config=config,
            setup_id=episode_identity,
            location=location,
            entry_path=entry_path,
            invalidation=invalidation,
            target=target,
            draw_selection=draw_selection,
            range_auction=range_auction,
            liquidity_route=liquidity_route,
            planned_entry=selected_planned_entry,
        )
    )
    remaining_ok = bool(
        plan is not None
        and plan.remaining_path_R >= config.minimum_remaining_path_R
    )
    execution_missing = _execution_unavailable(observation)
    zone_left = bool(
        location is not None
        and location.lifecycle is EntryLocationLifecycle.LEFT
    )
    stale_trigger = bool(
        entry_path is not None
        and entry_path.lifecycle is PathSequenceLifecycle.CLOSED
        and entry_path.ended_at != observation.asof
    )
    entry_path_censored = bool(
        entry_path is not None
        and entry_path.lifecycle is PathSequenceLifecycle.CENSORED
    )
    hard_gates = {
        "mature_dealing_range": mature,
        "range_boundary_sweep_reaccepted": range_reaccepted,
        "opposite_displacement_zone_inside_range": bool(
            location is not None
            and entry_path is not None
            and return_mss is not None
            and not mss_after_first_pullback
        ),
        "first_pullback_to_frozen_zone": first_pullback is not None,
        "aligned_entry_trigger": trigger_ready,
    }
    support = {
        "mature_range_visible": float(mature),
        "range_boundary_sweep_reaccepted": float(range_reaccepted),
        "return_displacement_zone": float(
            hard_gates["opposite_displacement_zone_inside_range"]
        ),
        "exact_first_pullback": float(first_pullback is not None),
        "aligned_entry_trigger": float(trigger_ready),
        "opposing_range_liquidity_visible": float(target is not None),
        "remaining_path_available": float(remaining_ok),
        "execution_fillability": (
            0.0
            if execution_missing
            else observation.execution.fillability
        ),
    }
    contradict = {
        "range_broken_or_missing": float(
            frozen_range_missing
            or (dealing_range is not None and not mature)
        ),
        "accepted_outside": float(accepted_outside),
        "return_zone_missing_or_outside_range": float(
            frozen_manipulation_missing
            or frozen_location_missing
            or (
                location is not None
                and mss_after_first_pullback
            )
        ),
        "entry_zone_left": float(zone_left),
        "trigger_opposed": float(
            entry_contradiction is not None
        ),
        "draw_consumed_or_missing": float(
            range_reaccepted
            and target is None
        ),
        "midpoint_chase": float(midpoint_chase),
        "remaining_path_consumed": float(
            plan is not None and not remaining_ok
        ),
        "execution_unavailable": float(execution_missing),
    }
    group_scores = {
        "structure": support["mature_range_visible"]
        * (1.0 - contradict["range_broken_or_missing"]),
        "displacement": support["return_displacement_zone"]
        * (1.0 - contradict["return_zone_missing_or_outside_range"]),
        "location": _location_quality(location)
        * (1.0 - max(
            contradict["entry_zone_left"],
            contradict["midpoint_chase"],
        )),
        "liquidity": min(
            support["range_boundary_sweep_reaccepted"],
            support["opposing_range_liquidity_visible"],
        )
        * (1.0 - max(
            contradict["accepted_outside"],
            contradict["draw_consumed_or_missing"],
        )),
        "trigger": support["aligned_entry_trigger"]
        * (1.0 - contradict["trigger_opposed"]),
        "execution": support["execution_fillability"]
        * (1.0 - contradict["execution_unavailable"]),
    }
    terminal_reason: str | None = None
    terminal_source_ids: tuple[str, ...] = ()
    if accepted_outside:
        terminal_reason = "range_auction_accepted_outside"
    elif resolution_censored:
        terminal_reason = "range_auction_resolution_deadline"
    elif frozen_range_missing:
        terminal_reason = "frozen_dealing_range_missing"
    elif dealing_range is not None and not mature:
        terminal_reason = "frozen_dealing_range_broken"
    elif frozen_manipulation_missing:
        terminal_reason = "frozen_manipulation_missing"
    elif frozen_location_missing:
        terminal_reason = "frozen_entry_zone_missing"
    elif zone_left:
        terminal_reason = "entry_zone_left"
    elif entry_contradiction is not None:
        terminal_reason = "entry_trigger_contradicted"
    elif entry_path_censored:
        terminal_reason = "entry_path_censored"
    elif mss_after_first_pullback:
        terminal_reason = "mss_confirmed_after_first_pullback"
    elif midpoint_chase:
        terminal_reason = "planned_entry_chased_past_range_value"
    elif (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.draw_selection is not None
        and target is None
    ):
        terminal_reason = "selected_draw_consumed_or_missing"
    if terminal_reason is not None:
        terminal_source_ids = _identity_tuple(
            (
                prior.context_id
                if dealing_range is None and prior is not None
                else None if dealing_range is None else dealing_range.range_id
            ),
            (
                prior.initiating_event_id
                if manipulation is None and prior is not None
                else None
                if manipulation is None
                else manipulation.manipulation_id
            ),
            (
                prior.entry_location_id
                if location is None and prior is not None
                else None if location is None else location.location_id
            ),
            None
            if prior is None or prior.draw_selection is None
            else prior.draw_selection.draw_id,
        )
    return _Evaluation(
        evidence=_typed_evidence_items(
            protocol,
            observation,
            support,
            contradict,
        ),
        trigger_ready=trigger_ready,
        setup_clock=setup_clock,
        sequence_signals={
            "mature_dealing_range": _SequenceSignal(
                float(mature),
                None if dealing_range is None else dealing_range.mature_at,
                ()
                if dealing_range is None
                else (dealing_range.range_id,),
            ),
            "range_boundary_sweep_reaccepted": _SequenceSignal(
                float(range_reaccepted),
                None
                if manipulation is None
                else manipulation.reaccepted_at,
                ()
                if manipulation is None
                else (
                    manipulation.manipulation_id,
                    manipulation.source_inventory_item_id,
                ),
            ),
            "opposite_displacement_zone_inside_range": _SequenceSignal(
                float(
                    location is not None
                    and entry_path is not None
                    and return_mss is not None
                ),
                None if return_mss is None else return_mss.resolved_at,
                ()
                if location is None or return_mss is None
                else (
                    location.location_id,
                    location.source_displacement_id,
                    location.source_zone_id,
                    return_mss.bos_id,
                    return_mss.target_swing_id,
                ),
            ),
            "first_pullback_to_frozen_zone": _SequenceSignal(
                float(first_pullback is not None),
                None
                if first_pullback is None
                else first_pullback.observed_at,
                ()
                if first_pullback is None or entry_path is None
                else (
                    entry_path.sequence_id,
                    first_pullback.step_id,
                    first_pullback.source_entity_id,
                ),
            ),
            "aligned_entry_trigger": _SequenceSignal(
                float(trigger_ready),
                None if trigger_step is None else trigger_step.observed_at,
                ()
                if trigger_step is None or entry_path is None
                else tuple(
                    value
                    for value in (
                        entry_path.sequence_id,
                        trigger_step.step_id,
                        trigger_step.source_entity_id,
                        trigger_step.source_event_id,
                    )
                    if value is not None
                ),
            ),
        },
        setup_identity=setup_identity,
        context_identity=(
            None if dealing_range is None else dealing_range.range_id
        ),
        episode_identity=episode_identity,
        initiating_event_id=(
            None if manipulation is None else manipulation.manipulation_id
        ),
        entry_location_id=(
            None if location is None else location.location_id
        ),
        entry_path_id=(
            None if entry_path is None else entry_path.sequence_id
        ),
        plan=plan,
        invalidation=invalidation,
        selected_draw=target,
        draw_selection=draw_selection,
        liquidity_route=liquidity_route,
        thesis_target=min(
            group_scores["structure"],
            group_scores["liquidity"],
        ),
        location_quality=group_scores["location"],
        entry_readiness=(
            0.0
            if not trigger_ready or trigger_step is None
            else clamp(trigger_step.strength)
        ),
        delivery_quality=_delivery_quality(
            observation,
            direction,
            plan,
        ),
        **_typed_market_uncertainty(
            observation,
            support,
            contradict,
            semantic_authority_missing=max(
                float("parked" in protocol.status),
                float(
                    "group4_atr_unready_sweep"
                    in observation.anomalies
                ),
                float(
                    bool(
                        observation.group4_atr_unready_sweep_item_ids
                    )
                ),
            ),
            graph_ambiguity=float(entry_ambiguity is not None),
        ),
        ambiguity_count=int(entry_ambiguity is not None),
        evidence_group_scores=group_scores,
        hard_gate_results=hard_gates,
        invalidated=bool(
            accepted_outside
            or resolution_censored
            or frozen_range_missing
            or frozen_manipulation_missing
            or frozen_location_missing
            or (dealing_range is not None and not mature)
            or zone_left
            or entry_contradiction is not None
            or entry_path_censored
            or mss_after_first_pullback
            or midpoint_chase
            or terminal_reason == "selected_draw_consumed_or_missing"
        ),
        entry_window_expired=stale_trigger,
        terminal_reason=terminal_reason,
        terminal_source_ids=terminal_source_ids,
    )


def _typed_parked(
    observation: MarketObservation,
    protocol: PlaybookProtocol,
) -> _Evaluation:
    support = {"parked_no_authority": 0.0}
    contradict = {"mature_range_authority_missing": 1.0}
    hard_gates = {
        step.step_id: False for step in protocol.required_sequence
    }
    return _Evaluation(
        evidence=_typed_evidence_items(
            protocol,
            observation,
            support,
            contradict,
        ),
        trigger_ready=False,
        setup_clock=None,
        sequence_signals={
            step.step_id: _SequenceSignal(0.0, None)
            for step in protocol.required_sequence
        },
        thesis_target=0.0,
        location_quality=0.0,
        entry_readiness=0.0,
        delivery_quality=0.0,
        typed_uncertainty=1.0,
        evidence_group_scores={
            group: 0.0 for group in _EVIDENCE_GROUPS
        },
        hard_gate_results=hard_gates,
        invalidated=False,
    )


def _typed_evaluate(
    playbook: Playbook,
    observation: MarketObservation,
    direction: Direction,
    protocol: PlaybookProtocol,
    prior: HypothesisBelief | None,
    config: BrainConfig,
    *,
    global_context: GlobalMarketContext | None = None,
    scene_graph: TemporalMarketSceneGraph | None = None,
    required_thesis: OpenMarketThesis | None = None,
    required_entry_location_id: str | None = None,
    lsr_context_only: bool = False,
    frozen_lsr_context_candidate: _LSRCandidate | None = None,
) -> _Evaluation:
    if playbook is Playbook.FAILED_AUCTION_VALUE_RETURN:
        return _typed_favr(
            observation,
            direction,
            protocol,
            prior,
            config,
            required_thesis=required_thesis,
        )
    if "parked" in protocol.status:
        return _typed_parked(observation, protocol)
    if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        return _typed_dfp(
            observation,
            direction,
            protocol,
            prior,
            config,
            global_context=global_context,
            required_thesis=required_thesis,
        )
    return _typed_lsr(
        observation,
        direction,
        protocol,
        prior,
        config,
        global_context=global_context,
        scene_graph=scene_graph,
        required_thesis=required_thesis,
        required_entry_location_id=required_entry_location_id,
        context_only=lsr_context_only,
        frozen_context_candidate=frozen_lsr_context_candidate,
    )


def _evaluation_source_ids(
    prior: HypothesisBelief | None,
    evaluation: _Evaluation,
) -> frozenset[str]:
    """Collect exact frozen identities; never infer relevance by direction."""

    values: list[str | None] = [
        evaluation.setup_identity,
        evaluation.context_identity,
        evaluation.episode_identity,
        evaluation.initiating_event_id,
        evaluation.entry_location_id,
        evaluation.entry_path_id,
        None
        if evaluation.invalidation is None
        else evaluation.invalidation.source_level_id,
        None
        if evaluation.selected_draw is None
        else evaluation.selected_draw.level_id,
        *evaluation.terminal_source_ids,
    ]
    values.extend(
        source_id
        for signal in evaluation.sequence_signals.values()
        for source_id in signal.source_ids
    )
    if evaluation.selected_trigger is not None:
        trigger = evaluation.selected_trigger
        values.extend(
            (
                trigger.trigger_id,
                trigger.entry_path_id,
                trigger.entry_location_id,
                trigger.source_entity_id,
                trigger.source_event_id,
            )
        )
    if evaluation.liquidity_route is not None:
        route = evaluation.liquidity_route
        values.extend(
            (
                route.context_draw_id,
                route.primary_deliverable_target_id,
                route.range_context_id,
                route.swept_range_boundary_id,
                route.opposing_range_boundary_id,
                *route.path_blocker_ids,
            )
        )
    if prior is not None:
        values.extend(
            (
                prior.key,
                prior.setup_context_id,
                prior.context_id,
                prior.episode_id,
                prior.initiating_event_id,
                prior.entry_location_id,
                prior.entry_path_id,
                None
                if prior.invalidation is None
                else prior.invalidation.source_level_id,
                *prior.terminal_source_ids,
            )
        )
        if prior.selected_trigger is not None:
            trigger = prior.selected_trigger
            values.extend(
                (
                    trigger.trigger_id,
                    trigger.entry_path_id,
                    trigger.entry_location_id,
                    trigger.source_entity_id,
                    trigger.source_event_id,
                )
            )
        if prior.sequence is not None:
            values.extend(
                source_id
                for step in prior.sequence.steps
                for source_id in step.source_ids
            )
    return frozenset(value for value in values if value)


def _dfp_context_terminal_source_ids(
    source: _Evaluation | HypothesisBelief,
) -> tuple[str, ...]:
    """Return only frozen DFP identities that can terminate its Context.

    The DFP ``h4_structure_and_draw`` signal deliberately retains the current
    latest-high/latest-low identities so a new confirmed leg produces one
    evidence revision.  Those mutable legs are supporting evidence, however,
    not structural invalidation authority.  The signal contract is ordered as
    ``structure, context draw, latest high, latest low, protected swing``;
    confirmed structures always retain the protected swing as the final item.
    """

    if isinstance(source, _Evaluation):
        signal = source.sequence_signals.get("h4_structure_and_draw")
        signal_source_ids = () if signal is None else signal.source_ids
        context_draw_id = (
            source.thesis_draw.level_id
            if source.thesis_draw is not None
            else source.liquidity_route.context_draw_id
            if source.liquidity_route is not None
            else None
        )
    else:
        if source.playbook is not Playbook.DISPLACEMENT_FIRST_PULLBACK:
            return ()
        step = next(
            (
                item
                for item in (
                    () if source.sequence is None else source.sequence.steps
                )
                if item.step_id == "h4_structure_and_draw"
            ),
            None,
        )
        signal_source_ids = () if step is None else step.source_ids
        context_draw_id = (
            None
            if source.thesis_draw is None
            else source.thesis_draw.level_id
        )
    authority_structure_id = (
        None if not signal_source_ids else signal_source_ids[0]
    )
    protected_swing_id = (
        None if len(signal_source_ids) < 5 else signal_source_ids[-1]
    )
    if (
        authority_structure_id is None
        or context_draw_id is None
        or protected_swing_id is None
        or signal_source_ids[1] != context_draw_id
    ):
        return ()
    return _identity_tuple(
        authority_structure_id,
        context_draw_id,
        protected_swing_id,
    )


def _open_thesis_supports_playbook(
    playbook: Playbook,
    thesis: OpenMarketThesis,
) -> bool:
    if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        return thesis.mechanism in {
            "directional_displacement",
            "frozen_zone_return",
        }
    if playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
        return thesis.mechanism in {
            "liquidity_sweep",
            "range_failed_auction",
        }
    return thesis.mechanism == "range_failed_auction"


def _thesis_candidate_id(
    market_epoch_id: str,
    root_id: str,
    playbook: Playbook,
    direction: Direction,
) -> str:
    """Stable identity for one independently evaluated canonical root."""

    raw = (
        f"{market_epoch_id}|{root_id}|{playbook.value}|{direction.value}"
    )
    return (
        "thesis-candidate:"
        f"{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
    )


def _lsr_zone_candidate_id(
    market_epoch_id: str,
    manipulation_root_id: str,
    displacement_id: str,
    zone_id: str,
    direction: Direction,
) -> str:
    """Action identity for exactly one LSR entry-zone Episode."""

    raw = (
        f"{market_epoch_id}|{manipulation_root_id}|{displacement_id}|"
        f"{zone_id}|{direction.value}"
    )
    return (
        "lsr-zone-candidate:"
        f"{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
    )


def _stable_context_thesis_id(
    market_epoch_id: str,
    playbook: Playbook,
    direction: Direction,
    evaluation: _Evaluation,
    prior: HypothesisBelief | None,
    *,
    known_context_ids: Collection[str] = (),
) -> str | None:
    """Freeze the long-horizon identity independently of an entry root.

    For DFP, the first registered sequence signal contains the H4 authority
    structure and context draw.  Including both explicitly in the hash keeps
    M5 displacement A/B/C as distinct candidates under one parent context.
    Other playbooks use their native context and draw identities without
    weakening the same epoch/direction boundary.
    """

    context_id = evaluation.context_identity
    prior_parent_is_known = bool(
        prior is not None
        and prior.context_thesis_id is not None
        and prior.context_thesis_id in known_context_ids
    )
    if context_id is None:
        # A cached root candidate can outlive the evaluator-native Context
        # that originally produced it.  Do not let that root resurrect a
        # long-lived thesis from a parent identity alone: only a prior that
        # still owns its native Context may bridge a transient evidence gap.
        return (
            prior.context_thesis_id
            if (
                prior is not None
                and prior.context_id is not None
                and prior.context_thesis_id is not None
                and prior_parent_is_known
            )
            else None
        )
    if (
        prior is not None
        and prior.context_id == context_id
        and prior.context_thesis_id is not None
        and prior_parent_is_known
    ):
        return prior.context_thesis_id
    if playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
        # LSR draw/target selection is child-entry dependent.  The parent
        # reversal Context is instead the exact manipulation/pool mechanism;
        # every zone under its frozen displacement must hash to one parent.
        if (
            "formed_pool_sweep" not in evaluation.sequence_signals
            or not evaluation.initiating_event_id
            or not evaluation.lsr_displacement_id
        ):
            return None
        raw = (
            f"{market_epoch_id}|{evaluation.initiating_event_id}|"
            f"{direction.value}|{context_id}|lsr-context"
        )
        return (
            "context-thesis:"
            f"{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
        )
    if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        # DFP may observe a directional root and a candidate draw before it
        # has the complete frozen H4 authority contract.  Those facts are
        # useful diagnostics, but they are not a Context Thesis identity.
        # Requiring the exact terminal roles prevents ``None`` authority from
        # being hashed into a parent that cannot later resolve causally.
        terminal_role_ids = _dfp_context_terminal_source_ids(evaluation)
        if len(terminal_role_ids) != 3:
            return None
        authority_structure_id, context_draw_id, _ = terminal_role_ids
    else:
        if playbook is not Playbook.FAILED_AUCTION_VALUE_RETURN:
            return None
        context_signal = next(
            iter(evaluation.sequence_signals.values()),
            None,
        )
        authority_structure_id = (
            None
            if context_signal is None or not context_signal.source_ids
            else context_signal.source_ids[0]
        )
        context_draw_id = (
            evaluation.thesis_draw.level_id
            if evaluation.thesis_draw is not None
            else evaluation.liquidity_route.context_draw_id
            if evaluation.liquidity_route is not None
            else None
        )
    raw = (
        f"{market_epoch_id}|{authority_structure_id}|{direction.value}|"
        f"{context_draw_id}|{context_id}"
    )
    return (
        "context-thesis:"
        f"{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
    )


def _bind_open_market_theses(
    playbook: Playbook,
    direction: Direction,
    prior: HypothesisBelief | None,
    evaluation: _Evaluation,
    global_context: GlobalMarketContext | None,
    *,
    required_thesis: OpenMarketThesis | None = None,
) -> _Evaluation:
    """Match graph-first theses without granting them action authority."""

    if global_context is None:
        return evaluation
    if not global_context.open_market_theses:
        return replace(
            evaluation,
            market_thesis_binding_required=True,
            market_thesis_match_status="no_open_thesis",
        )
    evaluation_ids = _evaluation_source_ids(prior, evaluation)
    direction_matches = tuple(
        thesis
        for thesis in global_context.open_market_theses
        if (
            required_thesis is None
            or thesis.thesis_id == required_thesis.thesis_id
        )
        if thesis.direction is None or thesis.direction is direction
    )
    if not direction_matches:
        return replace(
            evaluation,
            market_thesis_binding_required=True,
            market_thesis_match_status="no_direction_match",
        )
    matches: list[tuple[OpenMarketThesis, float, bool]] = []
    for thesis in direction_matches:
        if not _open_thesis_supports_playbook(playbook, thesis):
            continue
        # Shared authority and draw identities may raise descriptive
        # similarity.  Only the canonical episode root can grant the
        # action-facing graph binding; closure members such as a shared BOS,
        # FVG or OB remain supporting evidence and cannot join two episodes.
        if required_thesis is None:
            action_bound = thesis.root_id in evaluation_ids
        elif playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
            action_bound = bool(
                evaluation.initiating_event_id is not None
                and evaluation.initiating_event_id
                in thesis.mechanism_event_ids
            )
        else:
            # LSR/FAVR canonical roots are the frozen manipulation identity.
            action_bound = bool(
                evaluation.initiating_event_id == thesis.root_id
                and evaluation.setup_identity is not None
            )
        authority_ids = set(thesis.authority_source_ids)
        draw_ids = set(thesis.draw_candidate_ids)
        location_ids = set(thesis.entry_location_ids)
        trigger_ids = set(thesis.trigger_event_ids)
        components = (
            True,
            not evaluation_ids.isdisjoint(authority_ids),
            not evaluation_ids.isdisjoint(draw_ids),
            not evaluation_ids.isdisjoint(location_ids),
            not evaluation_ids.isdisjoint(trigger_ids),
        )
        matches.append(
            (thesis, sum(components) / len(components), action_bound)
        )
    if not matches:
        return replace(
            evaluation,
            market_thesis_binding_required=True,
            market_thesis_match_status="no_mechanism_match",
        )
    matches.sort(
        key=lambda value: (
            -int(value[2]),
            -value[1],
            value[0].formed_at,
            value[0].thesis_id,
        )
    )
    primary = matches[0][0]
    bound = next(
        (thesis for thesis, _, action_bound in matches if action_bound),
        None,
    )
    return replace(
        evaluation,
        market_thesis_ids=tuple(
            thesis.thesis_id for thesis, _, _ in matches
        ),
        market_thesis_id=primary.thesis_id,
        bound_market_thesis_id=(
            None if bound is None else bound.thesis_id
        ),
        market_thesis_root_id=primary.root_id,
        market_thesis_mechanism=primary.mechanism,
        market_thesis_authority_relation=primary.authority_relation,
        playbook_match_strength=matches[0][1],
        market_thesis_binding_required=True,
        market_thesis_action_bound=bound is not None,
        market_thesis_match_status=(
            "exact_root_bound"
            if bound is not None
            else "root_identity_unbound"
        ),
    )


def _position_matches_root_candidate(
    position: PositionSnapshot,
    candidate: HypothesisBelief,
) -> bool:
    """Match one filled position to its exact frozen root projection."""

    plan = candidate.plan
    return bool(
        position.status in {"open", "completed", "invalidated"}
        and candidate.candidate_id is not None
        and candidate.required_root_id is not None
        and candidate.playbook is position.playbook
        and candidate.direction is position.direction
        and candidate.setup_context_id == position.setup_id
        and candidate.entry_location_id == position.entry_location_id
        and plan is not None
        and plan.setup_id == position.setup_id
        and plan.entry_location_id == position.entry_location_id
        and plan.entry_path_id == position.entry_path_id
        and plan.invalidation == position.original_invalidation
        and bool(plan.targets)
        and plan.targets[0] == position.primary_target
        and plan.deadline == position.deadline
        and candidate.market_thesis_binding_required
        and candidate.market_thesis_action_bound
        and candidate.market_thesis_match_status == "exact_root_bound"
        and candidate.market_thesis_root_id
        == candidate.required_root_id
        and candidate.bound_market_thesis_id
        == candidate.market_thesis_id
    )


def _retain_position_market_thesis_binding(
    prior: HypothesisBelief,
    evaluation: _Evaluation,
    required_thesis: OpenMarketThesis,
) -> _Evaluation:
    """Carry an exact frozen graph binding after analytical root closure.

    A filled position and an active LSR parent Context can both outlive the
    bounded current-root view.  They may retain only their previously proven
    exact-root binding; no identity similarity can establish a new binding.
    """

    if (
        prior.required_root_id != required_thesis.root_id
        or prior.market_thesis_root_id != required_thesis.root_id
        or prior.market_thesis_id != required_thesis.thesis_id
        or prior.bound_market_thesis_id != required_thesis.thesis_id
        or not prior.market_thesis_action_bound
        or prior.market_thesis_match_status != "exact_root_bound"
    ):
        raise ValueError(
            "position-management candidate lost its frozen thesis binding"
        )
    return replace(
        evaluation,
        market_thesis_ids=prior.market_thesis_ids,
        market_thesis_id=prior.market_thesis_id,
        bound_market_thesis_id=prior.bound_market_thesis_id,
        market_thesis_root_id=prior.market_thesis_root_id,
        market_thesis_mechanism=prior.market_thesis_mechanism,
        market_thesis_authority_relation=(
            prior.market_thesis_authority_relation
        ),
        playbook_match_strength=prior.playbook_match_strength,
        market_thesis_binding_required=True,
        market_thesis_action_bound=True,
        market_thesis_match_status="exact_root_bound",
    )


def _retain_lsr_context_market_thesis_binding(
    binding: _LSRContextBinding,
    evaluation: _Evaluation,
    required_thesis: OpenMarketThesis,
) -> _Evaluation:
    """Bind a late child only to its parent's compact exact-root proof."""

    thesis = binding.thesis
    if (
        thesis.thesis_id != required_thesis.thesis_id
        or thesis.root_id != required_thesis.root_id
        or evaluation.initiating_event_id != thesis.root_id
        or evaluation.setup_identity is None
    ):
        raise ValueError(
            "late LSR EntryEpisode lost its frozen Context thesis binding"
        )
    return replace(
        evaluation,
        market_thesis_ids=(thesis.thesis_id,),
        market_thesis_id=thesis.thesis_id,
        bound_market_thesis_id=thesis.thesis_id,
        market_thesis_root_id=thesis.root_id,
        market_thesis_mechanism=thesis.mechanism,
        market_thesis_authority_relation=thesis.authority_relation,
        playbook_match_strength=binding.playbook_match_strength,
        market_thesis_binding_required=True,
        market_thesis_action_bound=True,
        market_thesis_match_status="exact_root_bound",
    )


def _retain_unresolved_episode_evaluation(
    prior: HypothesisBelief,
    evaluation: _Evaluation,
    global_context: GlobalMarketContext,
    observation: MarketObservation,
) -> _Evaluation:
    """Turn root disappearance into dormant evidence, never a terminal fact.

    Reducer retention and Scene-Graph current-root visibility are bounded
    views.  Their absence cannot stand in for an observed invalidation.  We
    preserve frozen causal identities and sequence clocks, fail every current
    action gate, and wait for structure/draw/deadline/explicit terminal data.
    """

    prior_steps = {
        step.step_id: step
        for step in (() if prior.sequence is None else prior.sequence.steps)
    }
    signals = dict(evaluation.sequence_signals)
    for step_id, step in prior_steps.items():
        if step.satisfied:
            signals[step_id] = _SequenceSignal(
                value=step.value,
                observed_at=step.observed_at,
                source_ids=step.source_ids,
            )
    # The frozen path exists before a trigger or plan.  Read it from the
    # Episode belief itself so a WAITING_TRIGGER candidate can become dormant
    # without losing the identity needed for a later exact-path resolution.
    entry_path_id = prior.entry_path_id
    selected_draw = (
        prior.plan.targets[0]
        if prior.plan is not None
        else prior.deliverable_targets[0]
        if prior.deliverable_targets
        else None
    )
    prior_source_ids = {
        identity
        for identity in (
            prior.required_root_id,
            prior.context_id,
            prior.episode_id,
            prior.setup_context_id,
            prior.entry_location_id,
            prior.initiating_event_id,
            entry_path_id,
            None
            if prior.invalidation is None
            else prior.invalidation.source_level_id,
            None if prior.thesis_draw is None else prior.thesis_draw.level_id,
            *(target.level_id for target in prior.deliverable_targets),
            *(
                source_id
                for step in prior_steps.values()
                for source_id in step.source_ids
            ),
        )
        if identity is not None
    }
    reason = evaluation.terminal_reason
    source_ids = set(evaluation.terminal_source_ids)

    # Bounded inventories may omit the frozen objects entirely.  Only an
    # exact typed terminal state may close a dormant episode; mere absence
    # always retains its frozen plan and trigger.
    frozen_location = next(
        (
            location
            for location in observation.entry_locations
            if location.location_id == prior.entry_location_id
        ),
        None,
    )
    frozen_path = next(
        (
            path
            for path in observation.path_sequences
            if path.sequence_id == entry_path_id
        ),
        None,
    )
    if (
        frozen_location is not None
        and frozen_location.lifecycle is EntryLocationLifecycle.LEFT
    ):
        reason = "entry_zone_left_or_failed"
        source_ids = {frozen_location.location_id}
    elif (
        frozen_path is not None
        and frozen_path.lifecycle is PathSequenceLifecycle.CENSORED
    ):
        reason = "entry_path_censored"
        source_ids = {frozen_path.sequence_id}
    elif (
        frozen_path is not None
        and frozen_path.lifecycle is PathSequenceLifecycle.CLOSED
    ):
        reason = "entry_window_expired"
        source_ids = {frozen_path.sequence_id}

    context_draw_consumed = bool(
        prior.thesis_draw is not None
        and any(
            item.item_id == prior.thesis_draw.level_id
            and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            for item in observation.liquidity_inventory
        )
    )
    selected_draw_consumed = bool(
        selected_draw is not None
        and any(
            item.item_id == selected_draw.level_id
            and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            for item in observation.liquidity_inventory
        )
    )
    if context_draw_consumed:
        reason = "context_draw_consumed_or_missing"
        source_ids = {prior.thesis_draw.level_id}  # type: ignore[union-attr]
    elif selected_draw_consumed:
        reason = "selected_draw_consumed_or_missing"
        source_ids = {selected_draw.level_id}  # type: ignore[union-attr]

    explicit_global_terminal = bool(
        not source_ids.isdisjoint(global_context.invalidated_source_ids)
        and not source_ids.isdisjoint(prior_source_ids)
    )
    related_terminal = bool(
        reason is not None
        and (
            not source_ids.isdisjoint(prior_source_ids)
            or evaluation.setup_identity == prior.setup_context_id
            or explicit_global_terminal
            or context_draw_consumed
            or selected_draw_consumed
        )
    )
    retained = replace(
        evaluation,
        trigger_ready=False,
        setup_clock=(
            None if prior.sequence is None else prior.sequence.started_at
        ),
        sequence_signals=signals,
        setup_identity=prior.setup_context_id,
        context_identity=prior.context_id,
        episode_identity=prior.episode_id,
        initiating_event_id=prior.initiating_event_id,
        entry_location_id=prior.entry_location_id,
        entry_path_id=entry_path_id,
        plan=prior.plan,
        invalidation=prior.invalidation,
        selected_draw=selected_draw,
        thesis_draw=prior.thesis_draw,
        draw_selection=prior.draw_selection,
        liquidity_route=prior.liquidity_route,
        thesis_target=float(
            prior.raw_quality_dimensions.get(
                "thesis_strength",
                prior.raw_probability or 0.0,
            )
        ),
        entry_readiness=0.0,
        delivery_quality=0.0,
        hard_gate_results={
            gate_id: False for gate_id in evaluation.hard_gate_results
        },
        invalidated=related_terminal,
        entry_window_expired=(
            related_terminal and reason == "entry_window_expired"
        ),
        terminal_reason=reason if related_terminal else None,
        terminal_source_ids=(
            _identity_tuple(*sorted(source_ids))
            if related_terminal
            else ()
        ),
        selected_trigger=prior.selected_trigger,
        target_selection_failure_reason="root_temporarily_absent",
        ambiguity_count=evaluation.ambiguity_count + 1,
    )
    return _replace_uncertainty_components(
        retained,
        graph_ambiguity=max(
            retained.uncertainty_graph_ambiguity,
            0.5,
        ),
    )


def _retained_episode_has_live_frozen_path(
    prior: HypothesisBelief,
    evaluation: _Evaluation,
    observation: MarketObservation,
) -> bool:
    """Return whether an absent discovery root's exact Episode can advance.

    Root discovery and EntryEpisode lifetime are separate clocks.  Losing the
    current open-thesis projection therefore cannot by itself freeze an exact
    episode whose setup, location and path are still present and live.  These
    equality checks intentionally admit only the frozen episode; they cannot
    borrow a location, path or trigger from another root.
    """

    if (
        prior.phase in _TERMINAL_PHASES
        or prior.setup_context_id is None
        or prior.episode_id is None
        or prior.entry_location_id is None
        or evaluation.invalidated
        or evaluation.entry_window_expired
        or evaluation.terminal_reason is not None
        or evaluation.setup_identity != prior.setup_context_id
        or evaluation.episode_identity != prior.episode_id
        or evaluation.entry_location_id != prior.entry_location_id
    ):
        return False
    frozen_path_id = prior.entry_path_id
    if frozen_path_id is None:
        return False
    if evaluation.entry_path_id != frozen_path_id:
        return False
    locations = tuple(
        location
        for location in observation.entry_locations
        if location.location_id == prior.entry_location_id
    )
    paths = tuple(
        path
        for path in observation.path_sequences
        if path.sequence_id == frozen_path_id
    )
    return bool(
        len(locations) == 1
        and locations[0].lifecycle is not EntryLocationLifecycle.LEFT
        and len(paths) == 1
        and (
            paths[0].lifecycle is PathSequenceLifecycle.ACTIVE
            or paths[0].ended_at == observation.asof
        )
    )


def _conflict_is_relevant(
    conflict: GlobalConflictEvidence,
    hypothesis_key: str,
    source_ids: frozenset[str],
) -> bool:
    affected = set(conflict.affected_hypothesis_ids)
    if hypothesis_key not in affected:
        return False
    conflict_sources = (
        conflict.event_id,
        conflict.source_node_id,
        conflict.target_node_id,
    )
    return any(
        conflict_id == source_id
        or conflict_id.endswith(f":{source_id}")
        or source_id.endswith(f":{conflict_id}")
        for conflict_id in conflict_sources
        for source_id in source_ids
    )


def _conflict_role(
    conflict: GlobalConflictEvidence,
) -> GlobalConflictRole:
    """Return the single typed Scene-Graph conflict interpretation."""

    return GlobalConflictRole(conflict.role)


def _hypothesis_conflict_relation(
    conflict: GlobalConflictEvidence,
    hypothesis_key: str,
    direction: Direction,
    source_ids: frozenset[str],
) -> HypothesisConflictRelation:
    """Interpret one graph conflict relative to one bound hypothesis.

    Scene Graph owns the market fact (source opposes target); Brain owns its
    hypothesis-specific meaning.  Direction alone is insufficient: an edge
    that is not identity-bound to this setup must remain unrelated.
    """

    if not _conflict_is_relevant(conflict, hypothesis_key, source_ids):
        return HypothesisConflictRelation.UNRELATED
    source_direction = conflict.source_direction
    target_direction = conflict.target_direction
    if source_direction is None or target_direction is None:
        return HypothesisConflictRelation.UNRELATED
    role = _conflict_role(conflict)
    if role is GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY:
        return (
            HypothesisConflictRelation.LOCAL_COUNTERTREND
            if source_direction is direction
            else HypothesisConflictRelation.UNRELATED
        )
    if source_direction is direction and target_direction is not direction:
        return HypothesisConflictRelation.SUPPORTS_CHALLENGER
    if target_direction is direction and source_direction is not direction:
        return HypothesisConflictRelation.CHALLENGES_INCUMBENT
    return HypothesisConflictRelation.UNRELATED


_TIMEFRAME_AUTHORITY_RANK: Mapping[Timeframe, int] = {
    Timeframe.H4: 5,
    Timeframe.H1: 4,
    Timeframe.M15: 3,
    Timeframe.M5: 2,
    Timeframe.M1: 1,
}


def _conflict_rank_gap(conflict: GlobalConflictEvidence) -> int:
    return abs(
        _TIMEFRAME_AUTHORITY_RANK[conflict.target_timeframe]
        - _TIMEFRAME_AUTHORITY_RANK[conflict.source_timeframe]
    )


def _primary_conflict_diagnostic(
    relations: Sequence[
        tuple[GlobalConflictEvidence, HypothesisConflictRelation]
    ],
) -> tuple[
    HypothesisConflictRelation,
    int | None,
    GlobalConflictRole | None,
    str | None,
    Timeframe | None,
]:
    """Select one deterministic diagnostic without changing hard routing."""

    priority = {
        HypothesisConflictRelation.CHALLENGES_INCUMBENT: 0,
        HypothesisConflictRelation.SUPPORTS_CHALLENGER: 1,
        HypothesisConflictRelation.LOCAL_COUNTERTREND: 2,
        HypothesisConflictRelation.UNRELATED: 3,
    }
    relevant = tuple(
        pair
        for pair in relations
        if pair[1] is not HypothesisConflictRelation.UNRELATED
    )
    if not relevant:
        return HypothesisConflictRelation.UNRELATED, None, None, None, None
    conflict, relation = min(
        relevant,
        key=lambda pair: (
            priority[pair[1]],
            -_conflict_rank_gap(pair[0]),
            -int(pair[0].observed_at.value),
            pair[0].conflict_id,
        ),
    )
    return (
        relation,
        _conflict_rank_gap(conflict),
        _conflict_role(conflict),
        conflict.structural_scale,
        conflict.source_timeframe,
    )


def _obstruction_matches_selected_draw(
    obstruction: DeliveryObstruction,
    selected_ids: frozenset[str],
) -> bool:
    return bool(
        obstruction.obstruction_id in selected_ids
        or not selected_ids.isdisjoint(obstruction.source_ids)
    )


_DIRECTIONAL_OBSTRUCTION_KINDS = frozenset(
    {
        "structure",
        "accepted_bos",
        "bos",
        "displacement",
        "fvg",
        "order_block",
    }
)

_DELIVERY_TIMEFRAME_RANK: Mapping[Timeframe, int] = {
    Timeframe.H4: 5,
    Timeframe.H1: 4,
    Timeframe.M15: 3,
    Timeframe.M5: 2,
    Timeframe.M1: 1,
}
_DELIVERY_SCOPE_RANK: Mapping[str, int] = {
    "internal": 1,
    "intermediate": 2,
    "external": 3,
}


def _delivery_setup_authority(
    evaluation: _Evaluation,
) -> tuple[int, int]:
    """Return the minimum structural authority for a hard path gate.

    DFP/FAVR delivery is a 5m setup even when its thesis is H4/H1.  LSR owns
    an explicit sweep-source scale/rank, so its frozen episode supplies the
    comparison.  Unknown ranks fail to the least-authoritative valid scope;
    they never manufacture a stronger blocker.
    """

    timeframe = Timeframe.M5
    if evaluation.authority_source_timeframe is not None:
        try:
            timeframe = Timeframe(evaluation.authority_source_timeframe)
        except ValueError:
            timeframe = Timeframe.M5
    scope = evaluation.authority_structural_rank or "internal"
    return (
        _DELIVERY_TIMEFRAME_RANK[timeframe],
        _DELIVERY_SCOPE_RANK.get(scope, 1),
    )


def _hard_obstruction_matches_setup(
    obstruction: DeliveryObstruction,
    evaluation: _Evaluation,
) -> bool:
    """Require confirmed and equal-or-higher structural authority."""

    if obstruction.acceptance_state not in {"accepted", "confirmed"}:
        return False
    required_timeframe, required_scope = _delivery_setup_authority(evaluation)
    candidate_timeframe = _DELIVERY_TIMEFRAME_RANK[obstruction.timeframe]
    if candidate_timeframe != required_timeframe:
        return candidate_timeframe > required_timeframe
    return (
        _DELIVERY_SCOPE_RANK[obstruction.structural_scope]
        >= required_scope
    )


_DFP_STRUCTURAL_LIQUIDITY_KINDS = frozenset(
    {
        "swing",
        "equal_highs",
        "equal_lows",
        "previous_session_high",
        "previous_session_low",
        "previous_day_high",
        "previous_day_low",
        "previous_week_high",
        "previous_week_low",
        "range_boundary",
    }
)


def _select_dfp_primary_target_result(
    observation: MarketObservation,
    direction: Direction,
    entry: float,
    config: BrainConfig,
    global_context: GlobalMarketContext | None,
    *,
    context_draw: LiquidityLevel,
    preferred_id: str | None,
    allowed_ids: frozenset[str] | None,
    location: EntryLocationState,
    invalidation: StructuralLevel,
) -> _DFPTargetSelection:
    """Choose a risk-qualified structural delivery before the H4 thesis draw.

    The H4 external draw supplies directional context and the terminal route;
    it is deliberately excluded from executable primary-target candidates.
    Candidate identities are clustered by conservative first contact, while
    the executable target retains its registered liquidity price.  The latter
    keeps DFP on the standard inventory-price risk contract; conservative
    contact targets remain an LSR-only convention.
    """

    inventory = _inventory_item_map(observation)
    levels = _visible_level_map(observation)
    structural_clusters = tuple(
        (levels[level.level_id], member_ids)
        for level, member_ids in _cluster_directional_liquidity(
            observation,
            direction,
            entry,
            config.tick_size,
            allowed_ids=allowed_ids,
        )
        if any(
            getattr(inventory.get(member_id), "kind", None)
            in _DFP_STRUCTURAL_LIQUIDITY_KINDS
            for member_id in member_ids
        )
    )
    if not structural_clusters:
        return _DFPTargetSelection(
            target=None,
            failure_reason="primary_structural_draw_missing",
            authority_barrier_id=None,
            authority_barrier_price=None,
        )

    context_distance = direction.sign * (context_draw.price - entry)
    before_context = tuple(
        (level, member_ids)
        for level, member_ids in structural_clusters
        if (
            context_draw.level_id not in member_ids
            and (distance := direction.sign * (level.price - entry))
            > config.tick_size
            and distance + config.tick_size < context_distance
        )
    )
    if not before_context:
        return _DFPTargetSelection(
            target=None,
            failure_reason="all_draws_at_or_beyond_context_draw",
            authority_barrier_id=None,
            authority_barrier_price=None,
        )

    candidate_barriers: tuple[tuple[DeliveryObstruction, float], ...] = ()
    if global_context is not None:
        view = global_context.obstruction_views.get(direction.value)
        candidate_barriers = tuple(
            (obstruction, distance)
            for obstruction in (() if view is None else view.hard_barriers)
            for distance in (
                _obstruction_distance(obstruction, direction, entry),
            )
            if (
                obstruction.hard
                and obstruction.acceptance_state
                in {"accepted", "confirmed"}
                and distance is not None
                and distance + config.tick_size < context_distance
                and not _obstruction_matches_selected_draw(
                    obstruction,
                    frozenset({context_draw.level_id}),
                )
                and _DELIVERY_TIMEFRAME_RANK[obstruction.timeframe]
                >= _DELIVERY_TIMEFRAME_RANK[Timeframe.M5]
            )
        )

    # A liquidity target can also be represented by a protected-boundary or
    # accepted-structure obstruction.  Compare each clustered candidate with
    # barriers independently so that its own identity, source identity, or
    # price cluster cannot block delivery to itself.
    capped: list[
        tuple[
            LiquidityLevel,
            tuple[str, ...],
            tuple[DeliveryObstruction, float] | None,
        ]
    ] = []
    blocked: list[tuple[DeliveryObstruction, float]] = []
    for level, member_ids in before_context:
        member_identity_ids = {
            *member_ids,
            *(
                source_id
                for member_id in member_ids
                for source_id in getattr(
                    inventory.get(member_id),
                    "source_ids",
                    (),
                )
            ),
        }
        member_contact_prices = tuple(
            contact
            for member_id in member_ids
            for item in (inventory.get(member_id),)
            if item is not None
            for contact in (_liquidity_contact_price(item, direction),)
            if contact is not None
        )
        relevant_barriers = tuple(
            (obstruction, distance)
            for obstruction, distance in candidate_barriers
            if (
                not _obstruction_matches_selected_draw(
                    obstruction,
                    frozenset(member_identity_ids),
                )
                and all(
                    abs(
                        obstruction.contact_price(direction) - contact
                    ) > config.tick_size
                    for contact in member_contact_prices
                )
            )
        )
        nearest = (
            None
            if not relevant_barriers
            else min(
                relevant_barriers,
                key=lambda item: (item[1], item[0].obstruction_id),
            )
        )
        target_distance = direction.sign * (level.price - entry)
        if (
            nearest is not None
            and target_distance + config.tick_size >= nearest[1]
        ):
            blocked.append(nearest)
            continue
        capped.append((level, member_ids, nearest))
    if not capped:
        barrier_obstruction, _ = min(
            blocked,
            key=lambda item: (item[1], item[0].obstruction_id),
        )
        return _DFPTargetSelection(
            target=None,
            failure_reason="all_draws_beyond_barrier",
            authority_barrier_id=barrier_obstruction.obstruction_id,
            authority_barrier_price=barrier_obstruction.contact_price(
                direction
            ),
        )

    ordered = tuple(
        sorted(
            capped,
            key=lambda item: (
                preferred_id not in item[1]
                if preferred_id is not None
                else False,
                direction.sign * (item[0].price - entry),
                _draw_rank(observation, item[0], entry),
            ),
        )
    )
    risk_qualified: list[
        tuple[
            LiquidityLevel,
            float,
            tuple[DeliveryObstruction, float] | None,
        ]
    ] = []
    for level, _, barrier in ordered:
        planned = _select_planned_entry(
            observation,
            direction,
            location,
            invalidation,
            level,
            config,
        )
        if planned is None:
            continue
        risk = abs(planned - invalidation.price)
        target_distance = direction.sign * (level.price - planned)
        if risk >= config.tick_size and target_distance / risk >= 1.0:
            risk_qualified.append((level, risk, barrier))
    if not risk_qualified:
        return _DFPTargetSelection(
            target=None,
            failure_reason="all_targets_below_minimum_R",
            authority_barrier_id=None,
            authority_barrier_price=None,
        )

    selected = next(
        (
            (level, barrier)
            for level, risk, barrier in risk_qualified
            if max(
                0.0,
                direction.sign * (level.price - observation.price),
            )
            / risk
            >= config.minimum_remaining_path_R
        ),
        None,
    )
    if selected is None:
        return _DFPTargetSelection(
            target=None,
            failure_reason="remaining_path_below_minimum",
            authority_barrier_id=None,
            authority_barrier_price=None,
        )
    selected_level, selected_barrier = selected
    return _DFPTargetSelection(
        target=selected_level,
        failure_reason=None,
        authority_barrier_id=(
            None
            if selected_barrier is None
            else selected_barrier[0].obstruction_id
        ),
        authority_barrier_price=(
            None
            if selected_barrier is None
            else selected_barrier[0].contact_price(direction)
        ),
    )


def _obstruction_can_block(
    obstruction: DeliveryObstruction,
    direction: Direction,
) -> bool:
    """Directional zones obstruct only the direction they oppose."""

    return not (
        obstruction.source_kind in _DIRECTIONAL_OBSTRUCTION_KINDS
        and obstruction.direction is direction
    )


def _obstruction_distance(
    obstruction: DeliveryObstruction,
    direction: Direction,
    entry: float,
) -> float | None:
    if not _obstruction_can_block(obstruction, direction):
        return None
    distance = direction.sign * (
        obstruction.contact_price(direction) - entry
    )
    return float(distance) if distance > 0.0 else None


def _route_with_hard_barriers(
    route: LiquidityRoute | None,
    barrier_ids: tuple[str, ...],
) -> LiquidityRoute | None:
    if route is None or route.path_blocker_ids == barrier_ids:
        return route
    raw = f"{route.route_id}|hard_barriers|{'|'.join(barrier_ids)}"
    return replace(
        route,
        route_id=f"route:{hashlib.sha256(raw.encode()).hexdigest()[:24]}",
        path_blocker_ids=barrier_ids,
    )


def _apply_delivery_obstruction_view(
    evaluation: _Evaluation,
    direction: Direction,
    global_context: GlobalMarketContext,
    tick_size: float,
    observation: MarketObservation | None = None,
) -> _Evaluation:
    """Apply current entry-to-target obstruction geometry to one thesis.

    This is the Brain's interpretation of descriptive context.  A selected
    draw is never reclassified as its own blocker; hard barriers are semantic
    gates, while soft friction only lowers delivery monotonically.
    """

    plan = evaluation.plan
    if not isinstance(plan, TradePlan) or not plan.targets:
        return evaluation
    entry = float(plan.planned_entry)
    target_distance = direction.sign * (
        plan.targets[0].price - entry
    )
    if target_distance <= tick_size:
        return replace(
            evaluation,
            delivery_quality=0.0,
            free_path_R=0.0,
        )
    inventory_target = (
        None
        if observation is None
        else _inventory_item_map(observation).get(
            plan.targets[0].level_id
        )
    )
    selected_ids = frozenset(
        {
            plan.selected_draw_id,
            plan.targets[0].level_id,
            *(
                ()
                if inventory_target is None
                else inventory_target.source_ids
            ),
        }
    )
    view = global_context.obstruction_views[direction.value]

    def distance(
        obstruction: DeliveryObstruction,
    ) -> float | None:
        if (
            _obstruction_matches_selected_draw(obstruction, selected_ids)
            or abs(
                obstruction.contact_price(direction)
                - plan.targets[0].price
            )
            <= tick_size
        ):
            return None
        return _obstruction_distance(obstruction, direction, entry)

    eligible_hard = tuple(
        (obstruction, candidate_distance)
        for obstruction in view.hard_barriers
        for candidate_distance in (distance(obstruction),)
        if candidate_distance is not None
        and candidate_distance + tick_size < target_distance
        and _hard_obstruction_matches_setup(obstruction, evaluation)
    )
    # A lower-authority candidate remains visible as friction; it cannot veto
    # a higher-authority setup.  Scene Graph has already collapsed co-located
    # and nested facts, so this count is per semantic price cluster.
    soft = tuple(
        (obstruction, candidate_distance)
        for obstruction in (*view.soft_frictions, *view.hard_barriers)
        for candidate_distance in (distance(obstruction),)
        if candidate_distance is not None
        and candidate_distance + tick_size < target_distance
        and (
            not obstruction.hard
            or not _hard_obstruction_matches_setup(obstruction, evaluation)
        )
    )
    # Once the nearest qualified hard cluster is encountered, farther hard
    # objects cannot add information to the current frozen path.
    hard = tuple(
        sorted(
            eligible_hard,
            key=lambda item: (item[1], item[0].obstruction_id),
        )[:1]
    )
    ordered_hard_ids = tuple(
        obstruction.obstruction_id
        for obstruction, _ in sorted(
            hard,
            key=lambda item: (item[1], item[0].obstruction_id),
        )
    )
    first_obstruction = min(
        (candidate_distance for _, candidate_distance in (*hard, *soft)),
        default=None,
    )
    first_hard = min(
        (candidate_distance for _, candidate_distance in hard),
        default=None,
    )
    risk_points = max(float(plan.risk_points), tick_size)
    free_path_points = min(
        target_distance,
        target_distance if first_hard is None else first_hard,
    )
    route = _route_with_hard_barriers(
        plan.liquidity_route,
        ordered_hard_ids,
    )
    routed_plan = (
        plan
        if route is plan.liquidity_route
        else replace(plan, liquidity_route=route)
    )
    soft_count = len(soft)
    delivery = (
        0.0
        if hard
        else evaluation.delivery_quality / (1.0 + soft_count)
    )
    return replace(
        evaluation,
        plan=routed_plan,
        liquidity_route=route,
        delivery_quality=clamp(delivery),
        obstruction_distance_R=(
            None
            if first_obstruction is None
            else first_obstruction / risk_points
        ),
        free_path_R=free_path_points / risk_points,
        soft_obstruction_count=soft_count,
        hard_barrier_before_target=bool(hard),
        path_blocker_ids=ordered_hard_ids,
    )


def _route_global_context(
    playbook: Playbook,
    direction: Direction,
    prior: HypothesisBelief | None,
    evaluation: _Evaluation,
    global_context: GlobalMarketContext | None,
    *,
    tick_size: float = 0.25,
    observation: MarketObservation | None = None,
    context_thesis: ContextThesisState | None = None,
) -> _Evaluation:
    """Route global facts to independent dimensions deterministically.

    Historical sequence signals are intentionally untouched.  This function
    changes only current validity/readiness/delivery, and it never consumes
    spread, depth, cost or execution deadline fields.
    """

    if global_context is None:
        return evaluation
    key = f"{playbook.value}:{direction.value}"
    source_ids = _evaluation_source_ids(prior, evaluation)
    terminal_source_ids = source_ids
    if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        # A newly confirmed H4 leg revises the long-horizon thesis but does
        # not break its frozen structure.  Keep those mutable source IDs in
        # ``source_ids`` for graph binding/conflict diagnostics while denying
        # them terminal authority over either the child EntryEpisode or its
        # parent Context Thesis.
        same_context_prior = bool(
            prior is not None
            and prior.context_id == evaluation.context_identity
        )
        dfp_sources = (
            (evaluation, prior)
            if same_context_prior and prior is not None
            else (evaluation,)
        )
        context_step_source_ids: set[str] = set()
        for source in dfp_sources:
            step = (
                source.sequence_signals.get("h4_structure_and_draw")
                if isinstance(source, _Evaluation)
                else next(
                    (
                        item
                        for item in (
                            ()
                            if source.sequence is None
                            else source.sequence.steps
                        )
                        if item.step_id == "h4_structure_and_draw"
                    ),
                    None,
                )
            )
            if step is not None:
                context_step_source_ids.update(step.source_ids)
        frozen_context_sources = (
            {
                *context_thesis.authority_ids,
                *(
                    ()
                    if context_thesis.context_draw is None
                    else (context_thesis.context_draw.level_id,)
                ),
                *(
                    ()
                    if context_thesis.structural_invalidation is None
                    else (
                        context_thesis.structural_invalidation.source_level_id,
                    )
                ),
            }
            if context_thesis is not None
            else set(
                _dfp_context_terminal_source_ids(
                    prior
                    if same_context_prior and prior is not None
                    else evaluation
                )
            )
        )
        mutable_context_sources = (
            context_step_source_ids - frozen_context_sources
        )
        terminal_source_ids = frozenset(
            source_ids - mutable_context_sources
        )
    conflict_relations = tuple(
        (
            conflict,
            _hypothesis_conflict_relation(
                conflict,
                key,
                direction,
                source_ids,
            ),
        )
        for conflict in global_context.material_conflicts
    )
    (
        authority_relation,
        authority_rank_gap,
        diagnostic_role,
        diagnostic_scope,
        diagnostic_timeframe,
    ) = _primary_conflict_diagnostic(conflict_relations)
    acceptance_state = (
        None
        if diagnostic_timeframe is None
        else global_context.scale_relation_details[
            diagnostic_timeframe.value
        ].acceptance_state
    )
    routed = replace(
        evaluation,
        authority_relation=authority_relation,
        authority_rank_gap=authority_rank_gap,
        conflict_role=diagnostic_role,
        conflict_scope=diagnostic_scope,
        acceptance_state=acceptance_state,
    )
    authority_invalidations = tuple(
        conflict
        for conflict, relation in conflict_relations
        if (
            relation is HypothesisConflictRelation.CHALLENGES_INCUMBENT
            and _conflict_role(conflict)
            is GlobalConflictRole.AUTHORITY_INVALIDATION
        )
    )
    if authority_invalidations:
        terminal_sources = _identity_tuple(
            *(
                identity
                for conflict in authority_invalidations
                for identity in (
                    conflict.conflict_id,
                    conflict.event_id,
                    conflict.source_node_id,
                    conflict.target_node_id,
                )
            )
        )
        return replace(
            routed,
            trigger_ready=False,
            plan=None,
            thesis_target=0.0,
            location_quality=0.0,
            entry_readiness=0.0,
            delivery_quality=0.0,
            hard_gate_results={
                gate_id: False
                for gate_id in evaluation.hard_gate_results
            },
            invalidated=True,
            terminal_reason="global_authority_invalidated",
            terminal_source_ids=terminal_sources,
            material_authority_opposition=True,
        )
    invalidated = tuple(
        source_id
        for source_id in global_context.invalidated_source_ids
        if source_id in terminal_source_ids
    )
    if invalidated:
        return replace(
            routed,
            trigger_ready=False,
            plan=None,
            thesis_target=0.0,
            location_quality=0.0,
            entry_readiness=0.0,
            delivery_quality=0.0,
            hard_gate_results={
                gate_id: False
                for gate_id in evaluation.hard_gate_results
            },
            invalidated=True,
            terminal_reason="global_frozen_source_invalidated",
            terminal_source_ids=invalidated,
        )
    ambiguous_ids = tuple(
        source_id
        for source_id in global_context.ambiguous_evidence
        if source_id in source_ids
    )
    unknown_ids = tuple(
        source_id
        for source_id in global_context.unknown_evidence
        if source_id in source_ids
    )
    if ambiguous_ids:
        # Ambiguity is absence of a directional conclusion, not evidence for
        # the opposite thesis.  Keep episode history and wait for resolution.
        routed = replace(
            routed,
            trigger_ready=False,
            entry_readiness=0.0,
            ambiguity_count=(
                routed.ambiguity_count + len(ambiguous_ids)
            ),
        )
        routed = _replace_uncertainty_components(
            routed,
            graph_ambiguity=max(
                routed.uncertainty_graph_ambiguity,
                len(ambiguous_ids) / (len(ambiguous_ids) + 1.0),
            ),
        )
    if unknown_ids:
        routed = _replace_uncertainty_components(
            routed,
            authority_missing=max(
                routed.uncertainty_authority_missing,
                len(unknown_ids) / (len(unknown_ids) + 1.0),
            ),
        )

    authority_opposes = _higher_authority_opposes(
        global_context,
        direction,
    )
    if authority_opposes and playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        # DFP requires alignment with the dominant intact authority.  Preserve
        # thesis history, but fail the current delivery/trigger permission;
        # only an exact frozen-source failure is terminal.
        routed = replace(
            routed,
            trigger_ready=False,
            delivery_quality=0.0,
            material_authority_opposition=True,
        )
    elif authority_opposes and playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
        # LSR may express a local countertrend auction.  Its target selector
        # has already capped delivery before the nearest authority barrier.
        routed = replace(
            routed,
            delivery_quality=(
                0.0 if routed.plan is None else routed.delivery_quality
            ),
        )

    challenging_transition = any(
        relation is HypothesisConflictRelation.CHALLENGES_INCUMBENT
        and _conflict_role(conflict)
        is GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE
        for conflict, relation in conflict_relations
    )
    if challenging_transition:
        # A connected challenger opposing this hypothesis suspends current
        # execution permission without erasing its thesis, plan identity or
        # historical causal sequence.  The same edge supports the opposite
        # LSR direction and therefore must not weaken that hypothesis.
        routed = replace(
            routed,
            trigger_ready=False,
            delivery_quality=0.0,
            material_authority_opposition=True,
        )
        transition_count = sum(
            relation is HypothesisConflictRelation.CHALLENGES_INCUMBENT
            and _conflict_role(conflict)
            is GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE
            for conflict, relation in conflict_relations
        )
        routed = _replace_uncertainty_components(
            routed,
            conflict=max(
                routed.uncertainty_conflict,
                transition_count / (transition_count + 1.0),
            ),
        )

    routed = _apply_delivery_obstruction_view(
        routed,
        direction,
        global_context,
        tick_size,
        observation,
    )

    return routed


def _evaluation_context_metadata(
    evaluation: _Evaluation,
    *,
    plan_feasibility: PlanFeasibility,
) -> Mapping[str, str]:
    """Project compact routing diagnostics for calibration and replay rows."""

    metadata = {
        "authority_relation": evaluation.authority_relation.value,
        "authority_rank_gap": (
            "unknown"
            if evaluation.authority_rank_gap is None
            else str(evaluation.authority_rank_gap)
        ),
        "conflict_role": (
            "none"
            if evaluation.conflict_role is None
            else evaluation.conflict_role.value
        ),
        "conflict_scope": evaluation.conflict_scope or "unknown",
        "acceptance_state": evaluation.acceptance_state or "unknown",
        "obstruction_distance_R": (
            "unknown"
            if evaluation.obstruction_distance_R is None
            else f"{evaluation.obstruction_distance_R:.12g}"
        ),
        "free_path_R": (
            "unknown"
            if evaluation.free_path_R is None
            else f"{evaluation.free_path_R:.12g}"
        ),
        "soft_obstruction_count": str(
            evaluation.soft_obstruction_count
        ),
        "hard_barrier_before_target": str(
            evaluation.hard_barrier_before_target
        ).lower(),
        "path_blocker_ids": json.dumps(
            list(evaluation.path_blocker_ids),
            separators=(",", ":"),
        ),
        "ambiguity_count": str(evaluation.ambiguity_count),
        "playbook_plan_delivery_valid": str(
            plan_feasibility.valid
        ).lower(),
        "plan_feasibility_failure": (
            "none"
            if plan_feasibility.failure_reason is None
            else plan_feasibility.failure_reason
        ),
        "uncertainty_conflict": f"{evaluation.uncertainty_conflict:.12g}",
        "uncertainty_required_evidence_missing": (
            f"{evaluation.uncertainty_required_evidence_missing:.12g}"
        ),
        "uncertainty_authority_missing": (
            f"{evaluation.uncertainty_authority_missing:.12g}"
        ),
        "uncertainty_graph_ambiguity": (
            f"{evaluation.uncertainty_graph_ambiguity:.12g}"
        ),
        "uncertainty_total": f"{evaluation.typed_uncertainty:.12g}",
    }
    if evaluation.authority_tier is not None:
        metadata.update(
            {
                "manipulation_tier": evaluation.authority_tier,
                "source_timeframe": (
                    evaluation.authority_source_timeframe or "unknown"
                ),
                "structural_rank": (
                    evaluation.authority_structural_rank or "unknown"
                ),
                "global_context_connected": str(
                    evaluation.global_context_connected
                ).lower(),
                "nesting": (
                    "nested" if evaluation.source_nested else "isolated"
                ),
                "authority_latched": str(
                    evaluation.authority_latched
                ).lower(),
            }
        )
    if (
        evaluation.lsr_pool_path_id is not None
        or evaluation.lsr_displacement_id is not None
        or evaluation.lsr_entry_zone_id is not None
    ):
        metadata.update(
            {
                "lsr_manipulation_id": (
                    evaluation.initiating_event_id or "unknown"
                ),
                "lsr_pool_path_id": (
                    evaluation.lsr_pool_path_id or "unknown"
                ),
                "lsr_displacement_id": (
                    evaluation.lsr_displacement_id or "unknown"
                ),
                "lsr_entry_zone_id": (
                    evaluation.lsr_entry_zone_id or "none"
                ),
            }
        )
    return metadata


def _favr_graph_broken_index(
    evaluation: _Evaluation,
    scene_graph: TemporalMarketSceneGraph,
) -> int | None:
    """Return the first disconnected FAVR stage using exact identities."""

    signals = evaluation.sequence_signals
    range_signal = signals["mature_dealing_range"]
    sweep_signal = signals["range_boundary_sweep_reaccepted"]
    zone_signal = signals[
        "opposite_displacement_zone_inside_range"
    ]
    pullback_signal = signals["first_pullback_to_frozen_zone"]
    trigger_signal = signals["aligned_entry_trigger"]

    if sweep_signal.value > 0.0:
        if (
            len(range_signal.source_ids) < 1
            or len(sweep_signal.source_ids) < 2
        ):
            return 1
        range_id = range_signal.source_ids[0]
        manipulation_id, boundary_id = sweep_signal.source_ids[:2]
        if not (
            evaluation.context_identity == range_id
            and evaluation.initiating_event_id == manipulation_id
            and scene_graph.has_direct_relation(
                (boundary_id,),
                SceneEdgeKind.SOURCED_FROM,
                (range_id,),
                source_kind="liquidity",
                target_kind="range",
            )
            and scene_graph.has_direct_relation(
                (manipulation_id,),
                SceneEdgeKind.SWEEPS,
                (boundary_id,),
                source_kind="manipulation",
                target_kind="liquidity",
            )
        ):
            return 1

    if zone_signal.value > 0.0:
        if (
            len(sweep_signal.source_ids) < 1
            or len(zone_signal.source_ids) < 3
        ):
            return 2
        manipulation_id = sweep_signal.source_ids[0]
        location_id, displacement_id, zone_id = (
            zone_signal.source_ids[:3]
        )
        zone_created = any(
            scene_graph.has_direct_relation(
                (displacement_id,),
                SceneEdgeKind.CREATES,
                (zone_id,),
                source_kind="displacement",
                target_kind=zone_kind,
            )
            for zone_kind in ("fvg", "order_block")
        )
        zone_return = any(
            scene_graph.has_direct_relation(
                (location_id,),
                SceneEdgeKind.RETURNS_TO,
                (zone_id,),
                source_kind="entry_location",
                target_kind=zone_kind,
            )
            for zone_kind in ("fvg", "order_block")
        )
        if not (
            evaluation.entry_location_id == location_id
            and scene_graph.find_causal_path(
                (manipulation_id,),
                (displacement_id,),
                # One direct admitted semantic/provenance edge is required.
                # The diagnostic PRECEDES edge is intentionally excluded.
                max_depth=1,
            )
            and zone_created
            and zone_return
        ):
            return 2

    if pullback_signal.value > 0.0:
        if (
            len(zone_signal.source_ids) < 1
            or len(pullback_signal.source_ids) < 3
        ):
            return 3
        location_id = zone_signal.source_ids[0]
        path_id, pullback_step_id, pullback_location_id = (
            pullback_signal.source_ids[:3]
        )
        if not (
            evaluation.entry_path_id == path_id
            and pullback_location_id == location_id
            and scene_graph.has_direct_relation(
                (pullback_step_id,),
                SceneEdgeKind.SOURCED_FROM,
                (path_id,),
                source_kind="path_step",
                target_kind="path_sequence",
            )
            and scene_graph.has_direct_relation(
                (pullback_step_id,),
                SceneEdgeKind.SOURCED_FROM,
                (location_id,),
                source_kind="path_step",
                target_kind="entry_location",
            )
        ):
            return 3

    if trigger_signal.value > 0.0:
        if (
            len(pullback_signal.source_ids) < 1
            or len(trigger_signal.source_ids) < 2
        ):
            return 4
        path_id = pullback_signal.source_ids[0]
        trigger_path_id, trigger_step_id = trigger_signal.source_ids[:2]
        if not (
            evaluation.entry_path_id == path_id == trigger_path_id
            and scene_graph.has_direct_relation(
                (trigger_step_id,),
                SceneEdgeKind.SOURCED_FROM,
                (path_id,),
                source_kind="path_step",
                target_kind="path_sequence",
            )
        ):
            return 4
    return None


def _require_connected_graph_sequence(
    playbook: Playbook,
    protocol: PlaybookProtocol,
    evaluation: _Evaluation,
    scene_graph: TemporalMarketSceneGraph | None,
    prior: HypothesisBelief | None,
) -> _Evaluation:
    """Fail closed on disconnected typed facts without inventing a veto.

    Snapshot detectors still describe each fact.  In the graph-backed engine,
    however, a later fact may advance a hypothesis only when the graph can
    connect it to the preceding context.  A missing relation is therefore an
    unresolved causal gap: it blocks the current and downstream hard gates,
    raises uncertainty, and withholds the plan, but it does not invalidate the
    thesis or create contradictory evidence.
    """

    if scene_graph is None:
        return evaluation
    ordered_ids = tuple(
        step.step_id for step in protocol.required_sequence
    )
    signals = dict(evaluation.sequence_signals)
    prior_steps = {
        step.step_id: step
        for step in (
            ()
            if prior is None or prior.sequence is None
            else prior.sequence.steps
        )
    }
    broken_index: int | None
    if playbook is Playbook.FAILED_AUCTION_VALUE_RETURN:
        broken_index = _favr_graph_broken_index(
            evaluation,
            scene_graph,
        )
    elif playbook in {
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    }:
        broken_index = None
        for index, (left_id, right_id) in enumerate(
            zip(ordered_ids[:-1], ordered_ids[1:]),
            start=1,
        ):
            left = signals[left_id]
            right = signals[right_id]
            prior_left = prior_steps.get(left_id)
            left_sources = (
                left.source_ids
                if left.value > 0.0
                else ()
                if prior_left is None or not prior_left.satisfied
                else prior_left.source_ids
            )
            if not left_sources or right.value <= 0.0:
                continue
            if not scene_graph.find_action_path(
                left_sources,
                right.source_ids,
                max_depth=6,
            ):
                broken_index = index
                break
    else:
        return evaluation
    if broken_index is None:
        return evaluation
    gates = dict(evaluation.hard_gate_results)
    for step_id in ordered_ids[broken_index:]:
        signal = signals[step_id]
        signals[step_id] = replace(signal, value=0.0)
        if step_id in gates:
            gates[step_id] = False
    disconnected_steps = max(1, len(ordered_ids) - broken_index)
    routed = replace(
        evaluation,
        trigger_ready=False,
        plan=None,
        entry_readiness=0.0,
        delivery_quality=0.0,
        sequence_signals=signals,
        hard_gate_results=gates,
        ambiguity_count=(
            evaluation.ambiguity_count + disconnected_steps
        ),
    )
    return _replace_uncertainty_components(
        routed,
        graph_ambiguity=max(
            routed.uncertainty_graph_ambiguity,
            disconnected_steps / (disconnected_steps + 1.0),
        ),
    )


def _terminal_candidate_is_new(
    prior: HypothesisBelief,
    candidate_id: str | None,
    candidate_clock: pd.Timestamp | None,
) -> bool:
    if (
        prior.phase not in _TERMINAL_PHASES
        or candidate_id is None
        or candidate_clock is None
    ):
        return False
    prior_identity = (
        prior.episode_id
        or prior.setup_context_id
        or (
            None
            if prior.sequence is None
            else prior.sequence.setup_id
        )
    )
    terminal_at = prior.terminal_at or prior.phase_started_at
    different_identity = bool(
        prior_identity is None or candidate_id != prior_identity
    )
    if not different_identity:
        return False
    if candidate_id in prior.competing_episode_ids:
        # This candidate formed while the prior episode owned the slot.  It
        # may acquire authority after terminal without rewriting its original
        # causal clock or requiring the same event to be emitted again.
        return True
    return candidate_clock > terminal_at


def _sequence_state(
    protocol: PlaybookProtocol,
    registry: PlaybookRegistry,
    evaluation: _Evaluation,
    prior: HypothesisBelief | None,
) -> HypothesisSequenceState:
    candidate_id = None
    first_definition = protocol.required_sequence[0]
    first_signal = evaluation.sequence_signals.get(first_definition.step_id)
    if (
        evaluation.setup_clock is not None
        and first_signal is not None
        and first_signal.observed_at is not None
        and first_signal.observed_at <= evaluation.setup_clock
        and first_signal.value >= first_definition.minimum_value
    ):
        candidate_id = evaluation.setup_identity
    prior_sequence = None if prior is None else prior.sequence
    if (
        candidate_id is None
        and prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior_sequence is not None
        and prior_sequence.setup_id is not None
        and not evaluation.invalidated
        and not evaluation.entry_window_expired
        and evaluation.terminal_reason is None
    ):
        # A bounded current view may omit the root/zone/path without emitting
        # a causal terminal.  Preserve the frozen episode sequence; current
        # gates still fail closed and therefore cannot create a stale ENTER.
        candidate_id = prior_sequence.setup_id
    prior_is_terminal = bool(
        prior is not None
        and prior.phase
        in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
    )
    terminal_candidate_is_new = bool(
        prior_is_terminal
        and prior is not None
        and _terminal_candidate_is_new(
            prior,
            candidate_id,
            evaluation.setup_clock,
        )
    )
    if prior_is_terminal and not terminal_candidate_is_new:
        candidate_id = None
        prior_sequence = None
    reuse_prior = bool(
        prior_sequence is not None
        and prior_sequence.setup_id is not None
        and candidate_id == prior_sequence.setup_id
    )
    if reuse_prior:
        assert prior_sequence is not None
        setup_id = prior_sequence.setup_id
        setup_clock = prior_sequence.started_at
        prior_steps = {
            step.step_id: step for step in prior_sequence.steps
        }
    else:
        setup_id = candidate_id
        setup_clock = evaluation.setup_clock
        prior_steps = {}

    steps: list[SequenceStepState] = []
    prior_clock: pd.Timestamp | None = None
    sequence_open = setup_id is not None
    for definition in protocol.required_sequence:
        signal = evaluation.sequence_signals.get(definition.step_id)
        if signal is None:
            raise ValueError(
                f"{protocol.playbook.value} evaluator omitted registered sequence "
                f"step {definition.step_id}"
            )
        latched = prior_steps.get(definition.step_id)
        if latched is not None and latched.satisfied and sequence_open:
            state = latched
        else:
            raw_satisfied = bool(
                signal.observed_at is not None
                and signal.value >= definition.minimum_value
            )
            ordered = bool(
                raw_satisfied
                and sequence_open
                and (
                    prior_clock is None
                    or signal.observed_at is not None
                    and signal.observed_at >= prior_clock
                )
            )
            state = SequenceStepState(
                step_id=definition.step_id,
                satisfied=ordered,
                value=signal.value,
                observed_at=signal.observed_at,
                source_ids=signal.source_ids,
            )
        steps.append(state)
        if state.satisfied:
            prior_clock = state.observed_at
        else:
            sequence_open = False
    if not steps[0].satisfied:
        setup_id = None
        setup_clock = None
    return HypothesisSequenceState(
        protocol_version=str(protocol.schema_version),
        protocol_hash=registry.fingerprint,
        setup_id=setup_id,
        steps=tuple(steps),
        started_at=setup_clock,
    )


def _validate_evidence_contract(
    protocol: PlaybookProtocol,
    evidence: Sequence[Evidence],
) -> None:
    produced = {item.primitive for item in evidence}
    registered = set(protocol.supporting_evidence) | set(
        protocol.contradicting_evidence
    )
    if produced != registered:
        missing = sorted(registered - produced)
        extra = sorted(produced - registered)
        raise ValueError(
            f"{protocol.playbook.value} evidence violates frozen registry; "
            f"missing={missing}, extra={extra}"
        )




def _visible_levels(observation: MarketObservation) -> list[LiquidityLevel]:
    return [
        LiquidityLevel(
            level_id=item.item_id,
            timeframe=item.timeframe,
            side=item.side,
            price=item.price,
            formed_at=item.formed_at,
            confirmed_at=item.confirmed_at,
            touches=max(0, len(item.source_ids) - 1),
            swept=False,
        )
        for item in observation.liquidity_inventory
        if (
            item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
            and item.confirmed_at <= observation.asof
        )
    ]




def _terminal_has_new_setup(
    prior: HypothesisBelief,
    sequence: HypothesisSequenceState,
) -> bool:
    return _terminal_candidate_is_new(
        prior,
        sequence.setup_id,
        sequence.started_at,
    )




def _typed_evidence_revision(
    hypothesis_key: str,
    protocol: PlaybookProtocol,
    evaluation: _Evaluation,
) -> str:
    """Hash only evidence that can change the typed thesis dimension.

    Evidence observation clocks are deliberately excluded: every completed
    minute describes the same retained evidence again.  Frozen source clocks,
    source identities, registered values and gate results remain included.
    """

    thesis_primitives = set(
        protocol.evidence_groups.get("structure", ())
    ) | set(protocol.evidence_groups.get("liquidity", ()))
    evidence = tuple(
        sorted(
            (
                item.primitive,
                float(item.value),
                float(item.weight),
                bool(item.supports),
            )
            for item in evaluation.evidence
            if item.primitive in thesis_primitives
        )
    )
    thesis_step_ids = {
        step.step_id for step in protocol.required_sequence[:2]
    }
    sequence_sources = tuple(
        sorted(
            (
                step_id,
                float(signal.value),
                None
                if signal.observed_at is None
                else signal.observed_at.isoformat(),
                tuple(signal.source_ids),
            )
            for step_id, signal in evaluation.sequence_signals.items()
            if step_id in thesis_step_ids
        )
    )
    revision_draw = (
        None
        if protocol.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        else evaluation.selected_draw
    )
    payload = (
        hypothesis_key,
        evaluation.context_identity,
        None if revision_draw is None else revision_draw.level_id,
        float(evaluation.thesis_target),
        evidence,
        sequence_sources,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _typed_thesis_strength(
    prior: HypothesisBelief | None,
    context_id: str | None,
    target: float,
    evidence_revision_id: str,
) -> float:
    target = clamp(target)
    prior_raw = (
        None
        if prior is None
        else prior.raw_quality_dimensions.get(
            "thesis_strength",
            prior.raw_probability
            if prior.raw_probability is not None
            else prior.thesis_strength,
        )
    )
    if (
        prior is None
        or context_id is None
        or prior.context_id != context_id
        or prior_raw is None
    ):
        return target
    if prior.evidence_revision_id == evidence_revision_id:
        return float(prior_raw)
    # A semantic revision is one new observation, not another minute of the
    # same evidence.  Replace the descriptive raw score once; the fitted
    # reliability map is applied later and never feeds this reducer.
    return target


def _plan_feasibility(
    plan: TradePlan | None,
    evaluation: _Evaluation,
    config: BrainConfig,
) -> PlanFeasibility:
    """Validate geometry without inventing playbook stop/draw semantics."""

    if plan is None:
        failure_reason = next(
            reason
            for reason, missing in (
                (
                    "market_thesis_root_unbound",
                    evaluation.market_thesis_binding_required
                    and not evaluation.market_thesis_action_bound,
                ),
                ("frozen_entry_location_missing", evaluation.entry_location_id is None),
                ("entry_path_missing", evaluation.entry_path_id is None),
                ("frozen_invalidation_missing", evaluation.invalidation is None),
                (
                    evaluation.target_selection_failure_reason
                    or "visible_draw_missing",
                    evaluation.selected_draw is None,
                ),
                ("entry_deadline_expired", evaluation.entry_window_expired),
                (
                    "hard_obstruction_before_target",
                    evaluation.hard_barrier_before_target,
                ),
                ("qualified_trigger_missing", evaluation.selected_trigger is None),
                ("playbook_plan_unavailable", True),
            )
            if missing
        )
        return PlanFeasibility(
            valid=False,
            planned_entry=None,
            invalidation=evaluation.invalidation,
            target=evaluation.selected_draw,
            remaining_path_R=None,
            deadline=None,
            failure_reason=failure_reason,
        )
    failure_reason = (
        "thesis_invalidated"
        if evaluation.invalidated
        else "entry_deadline_expired"
        if evaluation.entry_window_expired
        else "frozen_entry_location_missing"
        if evaluation.entry_location_id is None
        else "entry_path_missing"
        if evaluation.entry_path_id is None
        else "frozen_invalidation_missing"
        if evaluation.invalidation is None
        else "selected_draw_consumed_or_missing"
        if evaluation.selected_draw is None
        else "hard_obstruction_before_target"
        if evaluation.hard_barrier_before_target
        else "remaining_path_below_minimum"
        if plan.remaining_path_R < config.minimum_remaining_path_R
        else "delivery_not_available"
        if evaluation.delivery_quality <= 0.0
        else None
    )
    return PlanFeasibility(
        valid=failure_reason is None,
        planned_entry=plan.planned_entry,
        invalidation=plan.invalidation,
        target=plan.targets[0],
        remaining_path_R=plan.remaining_path_R,
        deadline=plan.deadline,
        failure_reason=failure_reason,
    )


def _typed_plan_delivery_valid(
    plan: TradePlan | None,
    evaluation: _Evaluation,
    config: BrainConfig,
) -> bool:
    """Compatibility boolean backed by the common feasibility contract."""

    return _plan_feasibility(plan, evaluation, config).valid


def _position_context_terminal_phase(
    prior: HypothesisBelief | None,
    evaluation: _Evaluation,
    observation: MarketObservation,
) -> PlaybookPhase | None:
    """Resolve only parent-Context terminals for an existing position."""

    if prior is None or prior.context_thesis_id is None:
        return None
    context_draw_id = (
        None if prior.thesis_draw is None else prior.thesis_draw.level_id
    )
    if context_draw_id is not None and any(
        item.item_id == context_draw_id
        and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
        for item in observation.liquidity_inventory
    ):
        return PlaybookPhase.COMPLETED
    if (
        prior.thesis_deadline is not None
        and observation.asof >= prior.thesis_deadline
    ):
        return PlaybookPhase.INVALIDATED
    reason = evaluation.terminal_reason
    while reason is not None and reason.startswith("parent_context_"):
        reason = reason.removeprefix("parent_context_")
    if reason in {
        "context_draw_consumed_or_missing",
        "context_draw_consumed",
    }:
        return PlaybookPhase.COMPLETED
    if reason in {
        "opposed_structure",
        "global_authority_invalidated",
        "context_thesis_deadline_elapsed",
        "context_structural_invalidation_breached",
        "context_frozen_source_invalidated",
    }:
        return PlaybookPhase.INVALIDATED
    if reason != "global_frozen_source_invalidated":
        return None
    terminal_sources = set(evaluation.terminal_source_ids)
    if context_draw_id is not None and context_draw_id in terminal_sources:
        return PlaybookPhase.COMPLETED
    context_sources = (
        {
            *(
                source_id
                for source_id in _dfp_context_terminal_source_ids(prior)
                if source_id != context_draw_id
            ),
        }
        if prior.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
        else {
            prior.context_id,
            *(
                source_id
                for step in (
                    () if prior.sequence is None else prior.sequence.steps
                )
                for source_id in step.source_ids
                if source_id != context_draw_id
            ),
        }
    )
    context_sources.discard(None)
    return (
        PlaybookPhase.INVALIDATED
        if not terminal_sources.isdisjoint(context_sources)
        else None
    )


def _frozen_primary_target_consumed(
    prior: HypothesisBelief | None,
    evaluation: _Evaluation,
    observation: MarketObservation,
) -> bool:
    target_id = (
        prior.plan.targets[0].level_id
        if (
            prior is not None
            and prior.plan is not None
            and prior.phase not in _TERMINAL_PHASES
        )
        else prior.deliverable_targets[0].level_id
        if (
            prior is not None
            and prior.deliverable_targets
            and prior.phase not in _TERMINAL_PHASES
        )
        else evaluation.selected_draw.level_id
        if evaluation.selected_draw is not None
        else None
    )
    return bool(
        target_id is not None
        and any(
            item.item_id == target_id
            and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            for item in observation.liquidity_inventory
        )
    )


def _lsr_has_frozen_execution_owner_plan(
    prior: HypothesisBelief | None,
) -> bool:
    """Return whether an LSR child owns the Context's frozen trade plan.

    Before execution ownership is selected, an Entry Episode may carry a
    provisional route target while it is still waiting for its own pullback
    or trigger.  Delivery of that descriptive target is not an Episode
    terminal.  The target becomes completion authority only after this exact
    child wins the first fully executable plan freeze.
    """

    return bool(
        prior is not None
        and prior.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and prior.plan is not None
        and prior.candidate_id is not None
        and prior.context_metadata.get("lsr_context_execution_owner")
        == prior.candidate_id
    )


def _typed_phase(
    playbook: Playbook,
    direction: Direction,
    prior: HypothesisBelief | None,
    thesis_strength: float,
    evaluation: _Evaluation,
    sequence: HypothesisSequenceState,
    plan: TradePlan | None,
    frozen_invalidation: StructuralLevel | None,
    episode_deadline: pd.Timestamp | None,
    observation: MarketObservation,
    position: PositionSnapshot | None,
    config: BrainConfig,
    *,
    parked: bool,
) -> PlaybookPhase:
    if parked:
        return PlaybookPhase.INACTIVE
    if position is not None and (
        position.playbook is playbook
        and position.direction is direction
    ):
        if position.status == "completed":
            return PlaybookPhase.COMPLETED
        if position.status == "invalidated":
            return PlaybookPhase.INVALIDATED
        stop_breached = (
            position.direction is Direction.LONG
            and observation.price
            <= position.original_invalidation.price
        ) or (
            position.direction is Direction.SHORT
            and observation.price
            >= position.original_invalidation.price
        )
        if stop_breached:
            return PlaybookPhase.INVALIDATED
        if _frozen_primary_target_consumed(
            prior,
            evaluation,
            observation,
        ):
            return PlaybookPhase.COMPLETED
        context_terminal_phase = _position_context_terminal_phase(
            prior,
            evaluation,
            observation,
        )
        if context_terminal_phase is not None:
            return context_terminal_phase
        setup_mismatch = bool(
            position.setup_id is not None
            and (
                sequence.setup_id != position.setup_id
                or evaluation.entry_location_id
                != position.entry_location_id
                or evaluation.entry_path_id != position.entry_path_id
            )
        )
        if (
            setup_mismatch
            or evaluation.invalidated
            or evaluation.delivery_quality <= 0.0
            or evaluation.material_authority_opposition
        ):
            return PlaybookPhase.WEAKENING
        if (
            position.unrealized_R > 0.0
            and evaluation.delivery_quality > 0.0
            and thesis_strength > 0.0
        ):
            return PlaybookPhase.DELIVERING
        return PlaybookPhase.ENTERED

    if (
        prior is not None
        and prior.phase in _TERMINAL_PHASES
        and not _terminal_has_new_setup(prior, sequence)
    ):
        return prior.phase
    prior_had_setup = bool(
        prior is not None
        and prior.phase
        not in {
            PlaybookPhase.INACTIVE,
            PlaybookPhase.COMPLETED,
            PlaybookPhase.INVALIDATED,
        }
        and prior.setup_context_id is not None
    )
    current_has_setup = bool(
        sequence.setup_id is not None and sequence.completed_steps > 0
    )
    setup_is_live = prior_had_setup or current_has_setup
    if setup_is_live and _frozen_invalidation_breached(
        direction,
        observation.price,
        frozen_invalidation,
    ):
        return PlaybookPhase.INVALIDATED
    if (
        setup_is_live
        and position is None
        and episode_deadline is not None
        and observation.asof >= episode_deadline
    ):
        return PlaybookPhase.INVALIDATED
    if setup_is_live and _entry_zone_beyond_frozen_invalidation(
        observation,
        evaluation,
        direction,
        frozen_invalidation,
    ):
        return PlaybookPhase.INVALIDATED
    if (
        _frozen_primary_target_consumed(prior, evaluation, observation)
        and (
            playbook is not Playbook.LIQUIDITY_SWEEP_REVERSAL
            or _lsr_has_frozen_execution_owner_plan(prior)
        )
    ):
        return PlaybookPhase.COMPLETED
    if (
        evaluation.invalidated
        or evaluation.entry_window_expired
    ) and (
        prior_had_setup or current_has_setup
    ):
        return PlaybookPhase.INVALIDATED
    if not current_has_setup:
        return PlaybookPhase.INACTIVE
    step_seen = {
        step.step_id: step.satisfied for step in sequence.steps
    }
    if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        mechanism_step = "m5_displacement_zone"
        location_step = "first_pullback_to_frozen_zone"
        trigger_step = "typed_entry_trigger"
        optional_confirmation = any(
            item.primitive == "h1_continuation_bos"
            and item.supports
            and item.value > 0.0
            for item in evaluation.evidence
        )
    elif playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
        mechanism_step = "reverse_displacement_zone"
        location_step = "reversal_first_pullback"
        trigger_step = "typed_entry_trigger"
        optional_confirmation = step_seen.get(
            "outside_acceptance_failed",
            False,
        )
    else:
        mechanism_step = "opposite_displacement_zone_inside_range"
        location_step = "first_pullback_to_frozen_zone"
        trigger_step = "aligned_entry_trigger"
        optional_confirmation = step_seen.get(
            "range_boundary_sweep_reaccepted",
            False,
        )
    all_market_gates = bool(
        evaluation.hard_gate_results
        and all(evaluation.hard_gate_results.values())
    )
    delivery_ready = _typed_plan_delivery_valid(plan, evaluation, config)
    market_thesis_bound = bool(
        not evaluation.market_thesis_binding_required
        or evaluation.market_thesis_action_bound
    )
    if (
        evaluation.trigger_ready
        and all_market_gates
        and delivery_ready
        and market_thesis_bound
        and thesis_strength > 0.0
    ):
        return PlaybookPhase.EXECUTABLE
    if not step_seen.get(mechanism_step, False):
        return (
            PlaybookPhase.ARMED
            if optional_confirmation
            else PlaybookPhase.FORMING
        )
    if not step_seen.get(location_step, False):
        return PlaybookPhase.WAITING_LOCATION
    trigger_was_observed = step_seen.get(trigger_step, False)
    if not trigger_was_observed:
        return PlaybookPhase.WAITING_TRIGGER
    if (
        evaluation.material_authority_opposition
        or plan is None
        or evaluation.delivery_quality <= 0.0
        or not delivery_ready
        or not evaluation.trigger_ready
        or not all_market_gates
        or not market_thesis_bound
        or thesis_strength <= 0.0
    ):
        # The trigger is part of episode history, but current delivery or
        # execution permission has weakened.  Calling this waiting_trigger
        # would falsely claim the confirmation never occurred.
        return PlaybookPhase.WEAKENING
    return PlaybookPhase.WAITING_TRIGGER


def _typed_terminal_closure(
    phase: PlaybookPhase,
    direction: Direction,
    evaluation: _Evaluation,
    sequence: HypothesisSequenceState,
    plan: TradePlan | None,
    frozen_invalidation: StructuralLevel | None,
    episode_deadline: pd.Timestamp | None,
    observation: MarketObservation,
    position: PositionSnapshot | None,
) -> tuple[pd.Timestamp | None, str | None, tuple[str, ...]]:
    if phase not in _TERMINAL_PHASES:
        return None, None, ()
    if position is not None and position.status in {
        "completed",
        "invalidated",
    }:
        reason = f"position_{position.status}"
        sources = _identity_tuple(
            position.thesis_hash,
            position.setup_id,
            position.entry_location_id,
            position.entry_path_id,
            position.original_invalidation.source_level_id,
        )
    elif (
        position is not None
        and phase is PlaybookPhase.INVALIDATED
        and (
            (
                position.direction is Direction.LONG
                and observation.price
                <= position.original_invalidation.price
            )
            or (
                position.direction is Direction.SHORT
                and observation.price
                >= position.original_invalidation.price
            )
        )
    ):
        reason = "position_invalidation_breached"
        sources = _identity_tuple(
            position.thesis_hash,
            position.setup_id,
            position.entry_location_id,
            position.entry_path_id,
            position.original_invalidation.source_level_id,
        )
    elif phase is PlaybookPhase.INVALIDATED and (
        _frozen_invalidation_breached(
            direction,
            observation.price,
            frozen_invalidation,
        )
    ):
        reason = "frozen_invalidation_breached"
        sources = _identity_tuple(
            sequence.setup_id,
            evaluation.entry_location_id,
            evaluation.entry_path_id,
            frozen_invalidation.source_level_id,
        )
    elif (
        phase is PlaybookPhase.INVALIDATED
        and _entry_zone_beyond_frozen_invalidation(
            observation,
            evaluation,
            direction,
            frozen_invalidation,
        )
    ):
        reason = "entry_zone_beyond_frozen_invalidation"
        sources = _identity_tuple(
            sequence.setup_id,
            evaluation.entry_location_id,
            None
            if frozen_invalidation is None
            else frozen_invalidation.source_level_id,
        )
    elif evaluation.terminal_reason is not None:
        # A confirmed semantic invalidation/censor on the deadline clock owns
        # the terminal reason.  Deadline is only the fallback when nothing
        # more specific resolved the episode on that completed bar.
        reason = evaluation.terminal_reason
        sources = evaluation.terminal_source_ids
    elif (
        phase is PlaybookPhase.INVALIDATED
        and position is None
        and episode_deadline is not None
        and observation.asof >= episode_deadline
    ):
        reason = "episode_deadline_elapsed"
        sources = _identity_tuple(
            sequence.setup_id,
            evaluation.episode_identity,
            evaluation.context_identity,
        )
    else:
        reason = (
            "completed"
            if phase is PlaybookPhase.COMPLETED
            else "invalidated"
        )
        sources = ()
    structural_source = (
        None
        if frozen_invalidation is None
        else frozen_invalidation.source_level_id
    )
    return (
        observation.asof,
        reason,
        _identity_tuple(
            *sources,
            evaluation.episode_identity,
            evaluation.context_identity,
            evaluation.initiating_event_id,
            sequence.setup_id,
            structural_source,
        ),
    )


def _belief_bound_source_ids(
    hypotheses: Mapping[str, HypothesisBelief],
) -> frozenset[str]:
    values: list[str | None] = []
    for hypothesis in hypotheses.values():
        values.extend(
            (
                hypothesis.setup_context_id,
                hypothesis.context_id,
                hypothesis.episode_id,
                hypothesis.initiating_event_id,
                hypothesis.entry_location_id,
                None
                if hypothesis.invalidation is None
                else hypothesis.invalidation.source_level_id,
            )
        )
        values.extend(hypothesis.competing_episode_ids)
        if hypothesis.selected_trigger is not None:
            trigger = hypothesis.selected_trigger
            values.extend(
                (
                    trigger.trigger_id,
                    trigger.entry_path_id,
                    trigger.entry_location_id,
                    trigger.source_entity_id,
                    trigger.source_event_id,
                )
            )
        if hypothesis.sequence is not None:
            values.extend(
                source_id
                for step in hypothesis.sequence.steps
                for source_id in step.source_ids
            )
        if hypothesis.liquidity_route is not None:
            route = hypothesis.liquidity_route
            values.extend(
                (
                    route.context_draw_id,
                    route.primary_deliverable_target_id,
                    route.range_context_id,
                    *route.path_blocker_ids,
                )
            )
    return frozenset(value for value in values if value)


def _identity_is_bound(identity: str, bound_ids: frozenset[str]) -> bool:
    return any(
        identity == bound
        or identity.endswith(f":{bound}")
        or bound.endswith(f":{identity}")
        for bound in bound_ids
    )


_THESIS_INVALIDATING_NODE_LIFECYCLES: Mapping[str, frozenset[str]] = {
    "swing": frozenset({"broken", "formation_failed"}),
    "structure": frozenset({"broken", "formation_failed"}),
    "bos": frozenset({"failed"}),
    "support_resistance": frozenset({"retired"}),
    "liquidity": frozenset({"consumed", "retired"}),
    "fvg": frozenset({"invalidated"}),
    "order_block": frozenset({"failed"}),
    "range": frozenset({"broken"}),
    "manipulation": frozenset({"accepted_outside", "censored"}),
    "entry_location": frozenset({"left"}),
    "reacceptance": frozenset({"failed", "censored"}),
    "path_sequence": frozenset({"censored"}),
    "swing_projection": frozenset({"broken", "censored"}),
}


def _hypothesis_terminal_role_ids(
    hypothesis: HypothesisBelief,
    node_kind: str,
    node_lifecycle: str | None = None,
) -> tuple[str, ...]:
    """Return only identities whose terminal state invalidates this thesis.

    Scene Graph ``INVALIDATED`` also describes normal event completion such as
    displacement exhaustion and consumed path blockers.  Those facts must not
    be promoted into a frozen-thesis failure merely because they remain in the
    episode's historical sequence.
    """

    invalidation_id = (
        None
        if hypothesis.invalidation is None
        else hypothesis.invalidation.source_level_id
    )
    sequence_ids = tuple(
        source_id
        for step in (() if hypothesis.sequence is None else hypothesis.sequence.steps)
        for source_id in step.source_ids
    )
    core_ids = _identity_tuple(
        hypothesis.setup_context_id,
        hypothesis.context_id,
        hypothesis.episode_id,
        hypothesis.initiating_event_id,
    )
    entry_ids = _identity_tuple(
        hypothesis.entry_location_id,
        invalidation_id,
    )
    route = hypothesis.liquidity_route
    if hypothesis.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
        zone_id = hypothesis.context_metadata.get("lsr_entry_zone_id")
        trigger_ids = _identity_tuple(
            None
            if hypothesis.selected_trigger is None
            else hypothesis.selected_trigger.trigger_id,
            None
            if hypothesis.selected_trigger is None
            else hypothesis.selected_trigger.source_entity_id,
            None
            if hypothesis.selected_trigger is None
            else hypothesis.selected_trigger.source_event_id,
        )
        if node_kind == "liquidity":
            return _identity_tuple(
                None
                if hypothesis.draw_selection is None
                else hypothesis.draw_selection.draw_id,
                None
                if hypothesis.plan is None
                else hypothesis.plan.selected_draw_id,
                None if route is None else route.context_draw_id,
                None
                if route is None
                else route.primary_deliverable_target_id,
                None if route is None else route.terminal_draw_id,
            )
        if node_kind in {"fvg", "order_block"}:
            return _identity_tuple(
                hypothesis.entry_location_id,
                None if zone_id in {None, "none"} else zone_id,
            )
        if node_kind == "entry_location":
            return _identity_tuple(hypothesis.entry_location_id)
        if node_kind in {"bos", "reacceptance"}:
            return trigger_ids
        if node_kind == "path_sequence":
            return _identity_tuple(
                hypothesis.entry_path_id,
                hypothesis.context_id,
            )
        if node_kind == "manipulation":
            if (
                node_lifecycle == "accepted_outside"
                and hypothesis.context_thesis_id is not None
                and hypothesis.hard_gate_results.get(
                    "outside_acceptance_failed",
                    False,
                )
                and hypothesis.context_metadata.get(
                    "lsr_displacement_id",
                    "none",
                )
                not in {"", "none"}
            ):
                # This lifecycle can reject a not-yet-established reversal,
                # but cannot retroactively invalidate a frozen reacceptance +
                # reverse-displacement Context.  CENSOR/source invalidation
                # and sweep-extreme breach retain terminal authority.
                return ()
            return _identity_tuple(
                hypothesis.initiating_event_id,
                invalidation_id,
            )
        if node_kind in {
            "swing",
            "structure",
            "support_resistance",
            "swing_projection",
        }:
            return _identity_tuple(invalidation_id)
        return ()
    if hypothesis.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        context_ids = _dfp_context_terminal_source_ids(hypothesis)
        authority_structure_id = (
            None if not context_ids else context_ids[0]
        )
        context_draw_id = (
            None if len(context_ids) < 2 else context_ids[1]
        )
        protected_swing_id = (
            None if len(context_ids) < 3 else context_ids[-1]
        )
        episode_zone_id = next(
            (
                step.source_ids[-1]
                for step in (
                    ()
                    if hypothesis.sequence is None
                    else hypothesis.sequence.steps
                )
                if step.step_id == "m5_displacement_zone"
                and len(step.source_ids) >= 3
            ),
            None,
        )
        trigger_ids = _identity_tuple(
            None
            if hypothesis.selected_trigger is None
            else hypothesis.selected_trigger.trigger_id,
            None
            if hypothesis.selected_trigger is None
            else hypothesis.selected_trigger.source_entity_id,
            None
            if hypothesis.selected_trigger is None
            else hypothesis.selected_trigger.source_event_id,
        )
        if node_kind == "liquidity":
            return _identity_tuple(
                context_draw_id,
                None
                if hypothesis.draw_selection is None
                else hypothesis.draw_selection.draw_id,
                None
                if hypothesis.plan is None
                else hypothesis.plan.selected_draw_id,
                None
                if route is None
                else route.primary_deliverable_target_id,
                None if route is None else route.terminal_draw_id,
            )
        if node_kind == "structure":
            return _identity_tuple(authority_structure_id)
        if node_kind == "swing":
            return _identity_tuple(protected_swing_id, invalidation_id)
        if node_kind == "swing_projection":
            # Projection rollover is descriptive scale metadata.  The raw
            # structure/protected swing owns the typed terminal lifecycle.
            return ()
        if node_kind == "bos":
            return trigger_ids
        if node_kind in {"fvg", "order_block"}:
            return _identity_tuple(*entry_ids, episode_zone_id)
        if node_kind == "entry_location":
            return entry_ids
        if node_kind == "reacceptance":
            return trigger_ids
        if node_kind == "path_sequence":
            return _identity_tuple(
                hypothesis.entry_path_id,
                hypothesis.setup_context_id,
                hypothesis.episode_id,
            )
        return _identity_tuple(invalidation_id)
    if node_kind == "liquidity":
        return _identity_tuple(
            None
            if hypothesis.draw_selection is None
            else hypothesis.draw_selection.draw_id,
            None if hypothesis.plan is None else hypothesis.plan.selected_draw_id,
            None if route is None else route.context_draw_id,
            None if route is None else route.primary_deliverable_target_id,
            None if route is None else route.terminal_draw_id,
        )
    if node_kind in {"swing", "support_resistance"}:
        return _identity_tuple(invalidation_id)
    if node_kind in {"structure", "bos", "swing_projection"}:
        return _identity_tuple(invalidation_id, *sequence_ids)
    if node_kind in {"fvg", "order_block"}:
        return _identity_tuple(*entry_ids, *sequence_ids)
    if node_kind == "range":
        if hypothesis.playbook is not Playbook.FAILED_AUCTION_VALUE_RETURN:
            return ()
        return _identity_tuple(
            None if route is None else route.range_context_id,
            *core_ids,
            *sequence_ids,
        )
    if node_kind == "manipulation":
        return _identity_tuple(
            invalidation_id,
            *core_ids,
            *sequence_ids,
        )
    if node_kind == "entry_location":
        return _identity_tuple(*entry_ids, *core_ids, *sequence_ids)
    if node_kind == "reacceptance":
        return _identity_tuple(*core_ids, *sequence_ids)
    if node_kind == "path_sequence":
        return _identity_tuple(
            hypothesis.setup_context_id,
            hypothesis.episode_id,
        )
    return _identity_tuple(invalidation_id)


def _route_terminal_delta_to_global_context(
    context: GlobalMarketContext,
    previous_belief: MarketBelief | None,
    scene_delta: SceneGraphDelta,
    scene_graph: TemporalMarketSceneGraph,
) -> GlobalMarketContext:
    """Bind terminal graph changes to prior frozen hypothesis identities."""

    if previous_belief is None:
        return context
    prior_candidates = {
        **previous_belief.thesis_candidates,
        **previous_belief.retained_episode_candidates,
        **previous_belief.position_management_candidates,
    }
    if not prior_candidates and previous_belief.global_context is None:
        prior_candidates = dict(previous_belief.hypotheses)
    context_structure_ids = frozenset(
        value
        for context_thesis in previous_belief.context_theses.values()
        for value in context_thesis.authority_ids
        if (
            context_thesis.structural_invalidation is None
            or value
            != context_thesis.structural_invalidation.source_level_id
        )
        and (
            context_thesis.context_draw is None
            or value != context_thesis.context_draw.level_id
        )
    )
    context_protected_ids = frozenset(
        context_thesis.structural_invalidation.source_level_id
        for context_thesis in previous_belief.context_theses.values()
        if context_thesis.structural_invalidation is not None
    )
    context_draw_ids = frozenset(
        context_thesis.context_draw.level_id
        for context_thesis in previous_belief.context_theses.values()
        if context_thesis.context_draw is not None
    )
    context_bound_ids = frozenset(
        value
        for value in (
            *context_structure_ids,
            *context_protected_ids,
            *context_draw_ids,
        )
    )
    bound_ids = frozenset(
        (*_belief_bound_source_ids(prior_candidates), *context_bound_ids)
    )
    if not bound_ids:
        return context
    previous_context = previous_belief.global_context
    if (
        previous_context is not None
        and previous_context.market_epoch_id != context.market_epoch_id
    ):
        invalidated = bound_ids
    else:
        invalidated_values: list[str] = []
        for node_id in dict.fromkeys(
            (*scene_delta.added_node_ids, *scene_delta.revised_node_ids)
        ):
            node = scene_graph._nodes.get(node_id)
            if (
                node is None
                or node.market_epoch_id != context.market_epoch_id
                or node.ambiguity_state is not EvidenceStatus.INVALIDATED
            ):
                continue
            invalidating_lifecycles = _THESIS_INVALIDATING_NODE_LIFECYCLES.get(
                node.kind,
                frozenset(),
            )
            if node.lifecycle not in invalidating_lifecycles:
                continue
            terminal_ids = frozenset(
                value
                for value in (
                    node.node_id,
                    node.entity_id,
                    *node.source_ids,
                )
                if value
            )
            terminal_entity_ids = frozenset(
                value
                for value in (node.node_id, node.entity_id)
                if value
            )
            for hypothesis in prior_candidates.values():
                invalidated_values.extend(
                    role_id
                    for role_id in _hypothesis_terminal_role_ids(
                        hypothesis,
                        node.kind,
                        node.lifecycle,
                    )
                    if (
                        role_id in terminal_entity_ids
                        if hypothesis.playbook
                        is Playbook.DISPLACEMENT_FIRST_PULLBACK
                        else _identity_is_bound(role_id, terminal_ids)
                    )
                )
            context_role_ids = (
                context_structure_ids
                if node.kind == "structure"
                else context_protected_ids
                if node.kind == "swing"
                else context_draw_ids
                if node.kind == "liquidity"
                else frozenset()
            )
            invalidated_values.extend(
                role_id
                for role_id in context_role_ids
                if role_id in terminal_entity_ids
            )
        invalidated = frozenset(invalidated_values)
    # The Scene-Graph global projection may mention a bound authority source
    # through a derived node's shared provenance (notably swing projections).
    # For Brain-owned frozen identities, only the typed role routing above is
    # authoritative.  Keep unrelated global facts and replace bound facts by
    # the exact role-resolved set for this delta.
    routed_invalidated = tuple(
        dict.fromkeys(
            (
                *(
                    source_id
                    for source_id in context.invalidated_source_ids
                    if source_id not in bound_ids
                ),
                *sorted(invalidated),
            )
        )
    )
    return (
        context
        if routed_invalidated == context.invalidated_source_ids
        else replace(
            context,
            invalidated_source_ids=routed_invalidated,
        )
    )


def _candidate_requires_explanation(
    candidate_id: str,
    observation: MarketObservation,
    global_context: GlobalMarketContext,
    scene_graph: TemporalMarketSceneGraph,
) -> bool:
    """Keep only high-authority structured roots, never isolated M1 noise."""

    manipulation = next(
        (
            item
            for item in observation.manipulations
            if item.manipulation_id == candidate_id
        ),
        None,
    )
    if manipulation is not None:
        if manipulation.lifecycle is not ManipulationLifecycle.REACCEPTED:
            return False
        if manipulation.source_timeframe in {Timeframe.H4, Timeframe.H1}:
            return True
        if manipulation.source_timeframe is Timeframe.M1:
            return False
        path = next(
            (
                item
                for item in observation.path_sequences
                if item.context_kind == "pool_reversal"
                and item.context_id == manipulation.manipulation_id
            ),
            None,
        )
        return bool(
            path is not None
            and (
                candidate := _lsr_candidate(
                    observation,
                    path,
                    global_context=global_context,
                    scene_graph=scene_graph,
                )
            )
            is not None
            and candidate.eligible_root
        )
    candidate_node_id = scene_graph._node_id_for_source(
        candidate_id,
        asof=observation.asof,
    )
    node = (
        None
        if candidate_node_id is None
        else scene_graph._nodes.get(candidate_node_id)
    )
    if (
        node is not None
        and node.kind == "displacement"
        and node.timeframe
        in {
            Timeframe.H4.value,
            Timeframe.H1.value,
            Timeframe.M15.value,
            Timeframe.M5.value,
        }
        and node.lifecycle in {"started", "active"}
    ):
        # A graph-connected directional episode is worth lightweight shadow
        # attribution even before it creates a qualified entry zone.  It still
        # has no plan or action authority.
        return True
    path = next(
        (
            item
            for item in observation.path_sequences
            if item.sequence_id == candidate_id
        ),
        None,
    )
    if path is None:
        return False
    if path.context_kind != "pool_reversal":
        return True
    candidate = _lsr_candidate(
        observation,
        path,
        global_context=global_context,
        scene_graph=scene_graph,
    )
    return bool(candidate is not None and candidate.eligible_root)


def _finalize_global_context(
    context: GlobalMarketContext,
    observation: MarketObservation,
    scene_graph: TemporalMarketSceneGraph,
    hypotheses: Mapping[str, HypothesisBelief],
) -> GlobalMarketContext:
    """Resolve changed structured roots against the six fixed playbooks."""

    bound_ids = _belief_bound_source_ids(hypotheses)
    retained = tuple(
        dict.fromkeys(
            (
                *context.unexplained_structured_episode_ids,
                *context.candidate_structured_episode_ids,
            )
        )
    )
    unexplained: list[str] = []
    for candidate_id in retained:
        if _identity_is_bound(candidate_id, bound_ids):
            continue
        node_id = scene_graph._node_id_for_source(
            candidate_id,
            asof=observation.asof,
        )
        node = (
            None
            if node_id is None
            else scene_graph._nodes.get(node_id)
        )
        if (
            node is None
            or node.market_epoch_id != context.market_epoch_id
            or node.ambiguity_state is EvidenceStatus.INVALIDATED
        ):
            continue
        if _candidate_requires_explanation(
            candidate_id,
            observation,
            context,
            scene_graph,
        ):
            unexplained.append(candidate_id)
    return replace(
        context,
        unexplained_structured_episode_ids=tuple(unexplained),
    )


def _inactive_summary_belief(
    playbook: Playbook,
    direction: Direction,
    protocol: PlaybookProtocol,
    registry: PlaybookRegistry,
    *,
    asof: pd.Timestamp,
    calibration_version: str,
) -> HypothesisBelief:
    """Blank six-slot projection when no current root candidate exists."""

    sequence = HypothesisSequenceState(
        protocol_version=str(protocol.schema_version),
        protocol_hash=registry.fingerprint,
        setup_id=None,
        steps=tuple(
            SequenceStepState(
                step_id=definition.step_id,
                satisfied=False,
                value=0.0,
                observed_at=None,
                source_ids=(),
            )
            for definition in protocol.required_sequence
        ),
        started_at=None,
    )
    feasibility = PlanFeasibility(
        valid=False,
        planned_entry=None,
        invalidation=None,
        target=None,
        remaining_path_R=None,
        deadline=None,
        failure_reason="no_root_candidate",
    )
    return HypothesisBelief(
        playbook=playbook,
        direction=direction,
        probability=0.0,
        phase=PlaybookPhase.INACTIVE,
        phase_started_at=asof,
        supporting=(),
        contradicting=(),
        invalidation=None,
        deliverable_targets=(),
        remaining_path_R=None,
        uncertainty=1.0,
        plan=None,
        sequence=sequence,
        raw_probability=0.0,
        calibration_version=calibration_version,
        thesis_strength=0.0,
        sequence_progress=0.0,
        location_quality=0.0,
        entry_readiness=0.0,
        delivery_quality=0.0,
        evidence_group_scores={
            name: 0.0
            for name in (
                "structure",
                "displacement",
                "location",
                "liquidity",
                "trigger",
                "execution",
            )
        },
        hard_gate_results={gate_id: False for gate_id in protocol.hard_gates},
        raw_quality_dimensions={
            "thesis_strength": 0.0,
            "sequence_progress": 0.0,
            "location_quality": 0.0,
            "entry_readiness": 0.0,
            "delivery_quality": 0.0,
            "uncertainty": 1.0,
        },
        context_metadata={
            "playbook_plan_delivery_valid": "false",
            "plan_feasibility_failure": "no_root_candidate",
        },
        record_kind="summary",
        plan_feasibility=feasibility,
    )


_CONTEXT_COMPLETION_REASONS = frozenset(
    {
        "context_draw_consumed_or_missing",
        "context_draw_consumed",
    }
)
_CONTEXT_INVALIDATION_REASONS = frozenset(
    {
        "opposed_structure",
        "global_authority_invalidated",
        "context_thesis_deadline_elapsed",
        "context_structural_invalidation_breached",
        "context_frozen_source_invalidated",
    }
)


@dataclass(frozen=True)
class _ContextClosure:
    """Compact, deadline-bounded tombstone for a terminal Context Thesis."""

    phase: PlaybookPhase
    reason: str
    source_ids: tuple[str, ...]
    terminal_at: pd.Timestamp
    retain_until: pd.Timestamp


def _normalized_context_reason(reason: str) -> str:
    while reason.startswith("parent_context_"):
        reason = reason.removeprefix("parent_context_")
    return reason


def _context_level_terminal(
    candidate: HypothesisBelief,
    thesis: OpenMarketThesis | None,
    context_thesis: ContextThesisState | None = None,
) -> tuple[str, str] | None:
    reason = candidate.terminal_reason
    if reason is not None and reason.startswith("parent_context_"):
        parent_reason = _normalized_context_reason(reason)
        if (
            candidate.playbook
            is Playbook.LIQUIDITY_SWEEP_REVERSAL
            and parent_reason in _CONTEXT_COMPLETION_REASONS
        ):
            return None
        return (
            (
                "completed"
                if parent_reason in _CONTEXT_COMPLETION_REASONS
                else "invalidated"
            ),
            parent_reason,
        )
    if (
        reason in _CONTEXT_COMPLETION_REASONS
        and candidate.playbook
        is not Playbook.LIQUIDITY_SWEEP_REVERSAL
    ):
        return "completed", str(reason)
    if reason in _CONTEXT_INVALIDATION_REASONS:
        return "invalidated", str(reason)
    if (
        candidate.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and reason
        in {
            "frozen_invalidation_breached",
            "position_invalidation_breached",
        }
    ):
        # LSR's sweep extreme and manipulation/reacceptance are Context
        # facts, not zone-local stops.  Their failure closes every child.
        return "invalidated", str(reason)
    if reason != "global_frozen_source_invalidated":
        return None
    context_sources = (
        {
            *context_thesis.authority_ids,
            *(
                ()
                if context_thesis.context_draw is None
                else (context_thesis.context_draw.level_id,)
            ),
            *(
                ()
                if context_thesis.structural_invalidation is None
                else (
                    context_thesis.structural_invalidation.source_level_id,
                )
            ),
        }
        if context_thesis is not None
        else set(_dfp_context_terminal_source_ids(candidate))
        if candidate.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
        else {
            source_id
            for source_id in (
                candidate.initiating_event_id,
                candidate.context_metadata.get("lsr_displacement_id"),
                None
                if candidate.invalidation is None
                else candidate.invalidation.source_level_id,
            )
            if source_id
        }
        if candidate.playbook
        is Playbook.LIQUIDITY_SWEEP_REVERSAL
        else {
            *(
                () if thesis is None else thesis.authority_source_ids
            ),
            *(
                () if candidate.thesis_draw is None else (
                    candidate.thesis_draw.level_id,
                )
            ),
            *(
                ()
                if candidate.sequence is None
                or not candidate.sequence.steps
                else candidate.sequence.steps[0].source_ids
            ),
        }
    )
    return (
        ("invalidated", reason)
        if not context_sources.isdisjoint(candidate.terminal_source_ids)
        else None
    )


def _cascade_context_terminals(
    observation: MarketObservation,
    candidates: Mapping[str, HypothesisBelief],
    candidate_theses: Mapping[str, OpenMarketThesis],
    closed_contexts: Mapping[str, _ContextClosure] | None = None,
    context_theses: Mapping[str, ContextThesisState] | None = None,
) -> dict[str, HypothesisBelief]:
    """Close all child episodes when their shared context closes."""

    output = dict(candidates)
    terminals = dict(closed_contexts or {})
    for candidate_id, candidate in output.items():
        context_id = candidate.context_thesis_id
        if context_id is None:
            continue
        # An existing closure is immutable.  A later child under the same
        # authority/draw identity cannot move the parent's terminal clock.
        if context_id in terminals:
            continue
        context_draw_consumed = bool(
            candidate.playbook
            is not Playbook.LIQUIDITY_SWEEP_REVERSAL
            and
            candidate.thesis_draw is not None
            and any(
                item.item_id == candidate.thesis_draw.level_id
                and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
                for item in observation.liquidity_inventory
            )
        )
        if context_draw_consumed:
            terminals[context_id] = _ContextClosure(
                phase=PlaybookPhase.COMPLETED,
                reason="context_draw_consumed",
                source_ids=_identity_tuple(
                    context_id,
                    candidate.thesis_draw.level_id,
                ),
                terminal_at=observation.asof,
                retain_until=max(
                    observation.asof + pd.Timedelta(minutes=1),
                    candidate.thesis_deadline
                    or _market_lifecycle_deadline(observation.asof),
                ),
            )
            continue
        if (
            candidate.phase not in _TERMINAL_PHASES
            and candidate.thesis_deadline is not None
            and observation.asof >= candidate.thesis_deadline
        ):
            terminals[context_id] = _ContextClosure(
                phase=PlaybookPhase.INVALIDATED,
                reason="context_thesis_deadline_elapsed",
                source_ids=_identity_tuple(
                    context_id,
                    candidate.context_id,
                    None
                    if candidate.thesis_draw is None
                    else candidate.thesis_draw.level_id,
                ),
                terminal_at=observation.asof,
                retain_until=observation.asof + pd.Timedelta(minutes=1),
            )
            continue
        terminal = _context_level_terminal(
            candidate,
            candidate_theses.get(candidate_id),
            (context_theses or {}).get(context_id),
        )
        if terminal is None:
            continue
        lifecycle, reason = terminal
        terminals[context_id] = _ContextClosure(
            phase=(
                PlaybookPhase.COMPLETED
                if lifecycle == "completed"
                else PlaybookPhase.INVALIDATED
            ),
            reason=_normalized_context_reason(reason),
            source_ids=_identity_tuple(
                *candidate.terminal_source_ids,
                context_id,
                candidate.context_id,
            ),
            terminal_at=candidate.terminal_at or observation.asof,
            retain_until=max(
                observation.asof + pd.Timedelta(minutes=1),
                candidate.thesis_deadline
                or _market_lifecycle_deadline(observation.asof),
            ),
        )
    for candidate_id, candidate in tuple(output.items()):
        terminal = terminals.get(candidate.context_thesis_id or "")
        if terminal is None:
            continue
        if candidate.phase in _TERMINAL_PHASES:
            # Reclassify a same-clock Context draw delivery from the generic
            # invalidation emitted by the entry evaluator.  Older local child
            # terminals remain immutable episode history.
            if (
                candidate.terminal_at == observation.asof
                and (
                    candidate.phase is not terminal.phase
                    or _normalized_context_reason(
                        candidate.terminal_reason or ""
                    )
                    != terminal.reason
                )
                and _normalized_context_reason(
                    candidate.terminal_reason or ""
                )
                in {
                    "global_frozen_source_invalidated",
                    "context_draw_consumed_or_missing",
                    "context_draw_consumed",
                    "opposed_structure",
                    "global_authority_invalidated",
                    "context_thesis_deadline_elapsed",
                    "context_structural_invalidation_breached",
                    "context_frozen_source_invalidated",
                }
            ):
                output[candidate_id] = replace(
                    candidate,
                    phase=terminal.phase,
                    phase_started_at=observation.asof,
                    terminal_at=observation.asof,
                    terminal_reason=terminal.reason,
                    terminal_source_ids=_identity_tuple(
                        *terminal.source_ids,
                        candidate.episode_id,
                        candidate.setup_context_id,
                    ),
                )
            continue
        output[candidate_id] = replace(
            candidate,
            phase=terminal.phase,
            phase_started_at=terminal.terminal_at,
            terminal_at=terminal.terminal_at,
            terminal_reason=(
                terminal.reason
                if _normalized_context_reason(
                    candidate.terminal_reason or ""
                )
                == terminal.reason
                else f"parent_context_{terminal.reason}"
            ),
            terminal_source_ids=_identity_tuple(
                *terminal.source_ids,
                candidate.episode_id,
                candidate.setup_context_id,
            ),
        )
    return output


def _freeze_lsr_context_execution_owner(
    observation: MarketObservation,
    candidates: Mapping[str, HypothesisBelief],
    winners: dict[str, str],
    owner_snapshots: dict[str, _LSRExecutionOwner],
    global_context: GlobalMarketContext | None,
    config: BrainConfig,
) -> dict[str, HypothesisBelief]:
    """Latch the first causal trigger under each LSR Context Thesis.

    Every zone keeps its own analytical Episode.  Only the first Episode for
    which its frozen trigger and all plan-validity gates coexist receives
    action ownership, and that identity remains frozen for the Context
    lifetime.  Strength, target quality and later zones never participate in
    this selection.
    """

    output = dict(candidates)
    grouped: dict[str, list[tuple[str, HypothesisBelief]]] = {}
    for candidate_id, candidate in output.items():
        if (
            candidate.playbook
            is not Playbook.LIQUIDITY_SWEEP_REVERSAL
            or candidate.context_thesis_id is None
            or candidate.episode_id is None
        ):
            continue
        grouped.setdefault(candidate.context_thesis_id, []).append(
            (candidate_id, candidate)
        )
    for context_id, values in grouped.items():
        winner = winners.get(context_id)
        if winner is None:
            triggered = tuple(
                (candidate_id, candidate)
                for candidate_id, candidate in values
                if (
                    candidate.phase is PlaybookPhase.EXECUTABLE
                    and candidate.plan is not None
                    and candidate.selected_trigger is not None
                    and candidate.plan_feasibility is not None
                    and candidate.plan_feasibility.valid
                )
            )
            if triggered:
                # Ownership begins only when the trigger and every current
                # hard/plan gate first coexist.  A historically earlier
                # trigger that lacked a target or feasible plan at that time
                # cannot win by hindsight once multiple siblings become
                # executable together.
                first_clock = min(
                    item[1].phase_started_at for item in triggered
                )
                first = tuple(
                    item
                    for item in triggered
                    if item[1].phase_started_at == first_clock
                )
                winner = (
                    first[0][0]
                    if len(first) == 1
                    else f"lsr-ambiguous-same-clock:{context_id}"
                )
                winners[context_id] = winner
                if len(first) == 1:
                    owner_candidate_id, owner_candidate = first[0]
                    assert owner_candidate.plan is not None
                    assert owner_candidate.selected_trigger is not None
                    owner_snapshots[context_id] = _LSRExecutionOwner(
                        candidate_id=owner_candidate_id,
                        plan=owner_candidate.plan,
                        trigger=owner_candidate.selected_trigger,
                        first_executable_at=first_clock,
                    )
        if winner is None:
            continue
        owner_snapshot = owner_snapshots.get(context_id)
        for candidate_id, candidate in values:
            if candidate_id == winner:
                if (
                    owner_snapshot is not None
                    and candidate.phase in _TERMINAL_PHASES
                ):
                    # A terminal fact changes only the owner's lifecycle.  It
                    # cannot erase the plan/trigger or expose a route, draw or
                    # deadline re-evaluated on that terminal bar.  The frozen
                    # execution owner remains the sole plan source even after
                    # target delivery or invalidation; terminal phase keeps
                    # that historical plan non-actionable.
                    frozen_plan = owner_snapshot.plan
                    frozen_target = frozen_plan.targets[0]
                    remaining_path_R = max(
                        0.0,
                        frozen_plan.direction.sign
                        * (frozen_target.price - observation.price),
                    ) / frozen_plan.risk_points
                    projected_plan = replace(
                        frozen_plan,
                        remaining_path_R=float(remaining_path_R),
                    )
                    current_trigger_kinds = (
                        ()
                        if candidate.selected_trigger is None
                        else candidate.selected_trigger.available_trigger_kinds
                    )
                    projected_trigger = replace(
                        owner_snapshot.trigger,
                        available_trigger_kinds=tuple(
                            dict.fromkeys(
                                (
                                    *owner_snapshot.trigger
                                    .available_trigger_kinds,
                                    *current_trigger_kinds,
                                )
                            )
                        ),
                    )
                    feasibility = candidate.plan_feasibility
                    if feasibility is not None:
                        feasibility = replace(
                            feasibility,
                            planned_entry=projected_plan.planned_entry,
                            invalidation=projected_plan.invalidation,
                            target=frozen_target,
                            remaining_path_R=remaining_path_R,
                            deadline=projected_plan.deadline,
                        )
                    hard_barrier = _frozen_target_crosses_authority_barrier(
                        projected_plan,
                        observation,
                        global_context,
                        config.tick_size,
                    )
                    metadata = dict(candidate.context_metadata)
                    metadata.update(
                        {
                            "lsr_context_execution_owner": winner,
                            "lsr_context_execution_selection": (
                                "first_fully_executable_plan"
                            ),
                            "lsr_first_executable_at": (
                                owner_snapshot.first_executable_at.isoformat()
                            ),
                            "hard_barrier_before_target": str(
                                hard_barrier
                            ).lower(),
                            "playbook_plan_delivery_valid": str(
                                bool(feasibility is not None and feasibility.valid)
                            ).lower(),
                            "plan_feasibility_failure": (
                                "none"
                                if (
                                    feasibility is None
                                    or feasibility.failure_reason is None
                                )
                                else feasibility.failure_reason
                            ),
                        }
                    )
                    output[candidate_id] = replace(
                        candidate,
                        plan=projected_plan,
                        invalidation=frozen_plan.invalidation,
                        deliverable_targets=frozen_plan.targets,
                        remaining_path_R=remaining_path_R,
                        draw_selection=frozen_plan.draw_selection,
                        liquidity_route=frozen_plan.liquidity_route,
                        selected_trigger=projected_trigger,
                        plan_feasibility=feasibility,
                        context_metadata=metadata,
                    )
                    continue
                if (
                    owner_snapshot is not None
                    and candidate.phase not in _TERMINAL_PHASES
                ):
                    frozen_plan = owner_snapshot.plan
                    frozen_target = frozen_plan.targets[0]
                    remaining_path_R = max(
                        0.0,
                        frozen_plan.direction.sign
                        * (frozen_target.price - observation.price),
                    ) / frozen_plan.risk_points
                    projected_plan = replace(
                        frozen_plan,
                        remaining_path_R=float(remaining_path_R),
                    )
                    visible = _visible_level_map(observation)
                    route = projected_plan.liquidity_route
                    target_visible = frozen_target.level_id in visible
                    route_draws_visible = bool(
                        route is None
                        or (
                            (
                                route.context_draw_id is None
                                or route.context_draw_id in visible
                            )
                            and (
                                route.terminal_draw_id is None
                                or route.terminal_draw_id in visible
                            )
                        )
                    )
                    hard_barrier = _frozen_target_crosses_authority_barrier(
                        projected_plan,
                        observation,
                        global_context,
                        config.tick_size,
                    )
                    hard_deadline = observation.asof + pd.Timedelta(
                        minutes=observation.execution.minutes_to_deadline
                    )
                    deadline_valid = bool(
                        observation.asof < projected_plan.deadline
                        <= hard_deadline
                    )
                    frozen_failure = (
                        "selected_draw_consumed_or_missing"
                        if not target_visible or not route_draws_visible
                        else "hard_obstruction_before_target"
                        if hard_barrier
                        else "remaining_path_below_minimum"
                        if remaining_path_R < config.minimum_remaining_path_R
                        else "entry_deadline_expired"
                        if not deadline_valid
                        else None
                    )
                    current_failure = (
                        None
                        if candidate.plan_feasibility is None
                        else candidate.plan_feasibility.failure_reason
                    )
                    geometry_failures = {
                        "all_draws_beyond_barrier",
                        "context_draw_consumed",
                        "frozen_primary_target_unavailable",
                        "hard_obstruction_before_target",
                        "remaining_path_below_minimum",
                        "selected_draw_consumed_or_missing",
                        "visible_draw_missing",
                    }
                    if current_failure in geometry_failures:
                        current_failure = None
                    feasibility_failure = frozen_failure or current_failure
                    if (
                        feasibility_failure is None
                        and candidate.phase is PlaybookPhase.WEAKENING
                    ):
                        feasibility_failure = "delivery_not_available"
                    feasibility_valid = feasibility_failure is None
                    metadata = dict(candidate.context_metadata)
                    metadata.update(
                        {
                            "lsr_context_execution_owner": winner,
                            "lsr_context_execution_selection": (
                                "first_fully_executable_plan"
                            ),
                            "lsr_first_executable_at": (
                                owner_snapshot.first_executable_at.isoformat()
                            ),
                            "hard_barrier_before_target": str(
                                hard_barrier
                            ).lower(),
                            "playbook_plan_delivery_valid": str(
                                feasibility_valid
                            ).lower(),
                            "plan_feasibility_failure": (
                                "none"
                                if feasibility_failure is None
                                else feasibility_failure
                            ),
                        }
                    )
                    feasibility = PlanFeasibility(
                        valid=feasibility_valid,
                        planned_entry=projected_plan.planned_entry,
                        invalidation=projected_plan.invalidation,
                        target=frozen_target,
                        remaining_path_R=remaining_path_R,
                        deadline=projected_plan.deadline,
                        failure_reason=feasibility_failure,
                    )
                    phase = candidate.phase
                    phase_started_at = candidate.phase_started_at
                    if (
                        phase is PlaybookPhase.EXECUTABLE
                        and not feasibility_valid
                    ):
                        phase = PlaybookPhase.WEAKENING
                        phase_started_at = observation.asof
                    elif phase is PlaybookPhase.EXECUTABLE:
                        phase_started_at = (
                            owner_snapshot.first_executable_at
                        )
                    output[candidate_id] = replace(
                        candidate,
                        phase=phase,
                        phase_started_at=phase_started_at,
                        plan=projected_plan,
                        invalidation=projected_plan.invalidation,
                        deliverable_targets=projected_plan.targets,
                        remaining_path_R=remaining_path_R,
                        draw_selection=projected_plan.draw_selection,
                        liquidity_route=projected_plan.liquidity_route,
                        selected_trigger=owner_snapshot.trigger,
                        plan_feasibility=feasibility,
                        context_metadata=metadata,
                    )
                continue
            if (
                candidate.phase in _TERMINAL_PHASES
                or candidate.plan is None
            ):
                continue
            metadata = dict(candidate.context_metadata)
            metadata.update(
                {
                    "lsr_context_execution_owner": winner,
                    "lsr_context_execution_selection": (
                        "ambiguous_same_clock_fail_closed"
                        if winner.startswith(
                            "lsr-ambiguous-same-clock:"
                        )
                        else "first_causal_trigger"
                    ),
                }
            )
            output[candidate_id] = replace(
                candidate,
                phase=PlaybookPhase.WEAKENING,
                phase_started_at=observation.asof,
                plan=None,
                remaining_path_R=None,
                plan_feasibility=PlanFeasibility(
                    valid=False,
                    planned_entry=None,
                    invalidation=candidate.invalidation,
                    target=(
                        candidate.deliverable_targets[0]
                        if candidate.deliverable_targets
                        else None
                    ),
                    remaining_path_R=None,
                    deadline=None,
                    failure_reason=(
                        "lsr_context_execution_trigger_ambiguous_same_clock"
                        if winner.startswith(
                            "lsr-ambiguous-same-clock:"
                        )
                        else (
                            "lsr_context_execution_episode_already_frozen"
                        )
                    ),
                ),
                context_metadata=metadata,
            )
    return output


def _fail_closed_shared_episode_ownership(
    observation: MarketObservation,
    candidates: Mapping[str, HypothesisBelief],
    reserved_candidates: Mapping[str, HypothesisBelief],
    candidate_theses: Mapping[str, OpenMarketThesis] | None = None,
    previous_contexts: Mapping[str, ContextThesisState] | None = None,
    closed_contexts: Mapping[str, _ContextClosure] | None = None,
) -> dict[str, HypothesisBelief]:
    """Remove action identity when multiple roots claim one entry episode.

    A graph join may legitimately expose the same zone/path/trigger to more
    than one analytical Context root.  That is an unresolved ownership
    ambiguity, not a process-fatal invariant and never a reason to choose an
    arbitrary winner.
    """

    theses = candidate_theses or {}
    contexts = previous_contexts or {}
    closures = closed_contexts or {}
    owners: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for identity, candidate, position_owner in (
        *(
            (identity, candidate, False)
            for identity, candidate in candidates.items()
        ),
        *(
            (identity, candidate, True)
            for identity, candidate in reserved_candidates.items()
        ),
    ):
        context_id = candidate.context_thesis_id
        previous_context = (
            None if context_id is None else contexts.get(context_id)
        )
        if (
            context_id is not None
            and (
                context_id in closures
                or (
                    previous_context is not None
                    and (
                        previous_context.lifecycle
                        in {"completed", "invalidated", "censored"}
                        or (
                            previous_context.thesis_deadline is not None
                            and observation.asof
                            >= previous_context.thesis_deadline
                        )
                    )
                )
                or _context_level_terminal(
                    candidate,
                    theses.get(identity),
                    previous_context,
                )
                is not None
            )
        ):
            # A terminal parent no longer claims new physical entry evidence.
            # A locally terminal child below an otherwise-live Context still
            # owns its path history and prevents another live Context from
            # reusing that same zone/path as a new Episode.
            continue
        # Entry-path ownership is an explicit Episode identity.  In
        # particular, a WAITING_TRIGGER episode has no plan/trigger yet and
        # must not fall back to its setup id (the pool/displacement root is a
        # different causal object from the frozen zone-return path).
        path_id = candidate.entry_path_id
        identities = (
            ("entry_location", candidate.entry_location_id),
            ("entry_path", path_id),
            (
                "trigger",
                None
                if candidate.selected_trigger is None
                else candidate.selected_trigger.trigger_id,
            ),
        )
        for kind, source_id in identities:
            if source_id is not None:
                owners.setdefault((kind, source_id), []).append(
                    (identity, position_owner)
                )
    conflicted = {
        identity
        for values in owners.values()
        if len({owner for owner, _ in values}) > 1
        for identity, position_owner in values
        if not position_owner
    }
    # The open theses remain in GlobalMarketContext as analysis candidates.
    # Their projections are omitted from the realtime action mapping until a
    # unique owner exists; no arbitrary root receives the shared episode.
    return {
        identity: candidate
        for identity, candidate in candidates.items()
        if identity not in conflicted
    }


def _lsr_context_can_spawn_child(
    observation: MarketObservation,
    context_thesis_id: str | None,
    previous_contexts: Mapping[str, ContextThesisState],
    closed_contexts: Mapping[str, _ContextClosure],
) -> bool:
    """Allow zone discovery only below a live, known LSR Context.

    ``None`` means the manipulation root has never established a Context and
    may create its first child.  Once a root has a stable Context identity,
    absence from the bounded current view is not rearm authority: the exact
    prior Context must still be live and outside every causal terminal guard.
    Existing children are settled by the retained-episode path independently
    of this discovery permission.
    """

    if context_thesis_id is None:
        return True
    if context_thesis_id in closed_contexts:
        return False
    context = previous_contexts.get(context_thesis_id)
    if context is None or context.lifecycle in {
        "completed",
        "invalidated",
        "censored",
    }:
        return False
    return bool(
        context.thesis_deadline is None
        or observation.asof < context.thesis_deadline
    )


def _context_structural_invalidation(
    observation: MarketObservation,
    candidate: HypothesisBelief,
    previous: ContextThesisState | None,
) -> StructuralLevel | None:
    if previous is not None and previous.structural_invalidation is not None:
        return previous.structural_invalidation
    if candidate.playbook is not Playbook.DISPLACEMENT_FIRST_PULLBACK:
        return candidate.invalidation
    structure_id = None
    if candidate.sequence is not None:
        structure_step = next(
            (
                step
                for step in candidate.sequence.steps
                if step.step_id == "h4_structure_and_draw"
            ),
            None,
        )
        if structure_step is not None and structure_step.source_ids:
            structure_id = structure_step.source_ids[0]
    structure = next(
        (
            item
            for item in observation.frame(Timeframe.H4).structures
            if item.structure_id == structure_id
            and item.protected_swing_id is not None
            and item.protected_price is not None
            and item.confirmed_at is not None
        ),
        None,
    )
    if structure is None:
        return None
    return StructuralLevel(
        price=float(structure.protected_price),
        side=candidate.direction.invalidation_side,
        source_level_id=structure.protected_swing_id,
        observed_at=structure.confirmed_at,
        rationale="frozen Context Thesis protected structure",
    )


def _first_pullback_at(
    candidate: HypothesisBelief,
) -> pd.Timestamp | None:
    if candidate.sequence is None:
        return None
    return next(
        (
            step.observed_at
            for step in candidate.sequence.steps
            if step.step_id
            in {
                "first_pullback_to_frozen_zone",
                "reversal_first_pullback",
            }
            and step.satisfied
        ),
        None,
    )


def _retained_context_terminal(
    observation: MarketObservation,
    global_context: GlobalMarketContext,
    context: ContextThesisState,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Resolve a childless context from frozen causal facts only."""

    draw = context.context_draw
    if draw is not None:
        consumed = next(
            (
                item
                for item in observation.liquidity_inventory
                if item.item_id == draw.level_id
                and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            ),
            None,
        )
        if consumed is not None:
            return (
                "completed",
                "context_draw_consumed",
                _identity_tuple(
                    context.context_thesis_id,
                    draw.level_id,
                    *consumed.source_ids,
                ),
            )
    if (
        context.thesis_deadline is not None
        and observation.asof >= context.thesis_deadline
    ):
        return (
            "invalidated",
            "context_thesis_deadline_elapsed",
            _identity_tuple(context.context_thesis_id),
        )
    invalidation = context.structural_invalidation
    invalidated_ids = set(global_context.invalidated_source_ids)
    context_source_ids = {
        *(
            source_id
            for source_id in context.authority_ids
            if draw is None or source_id != draw.level_id
        ),
        *(
            ()
            if invalidation is None
            else (invalidation.source_level_id,)
        ),
    }
    if not invalidated_ids.isdisjoint(context_source_ids):
        return (
            "invalidated",
            "context_frozen_source_invalidated",
            _identity_tuple(
                context.context_thesis_id,
                *sorted(invalidated_ids & context_source_ids),
            ),
        )
    return None


def _build_context_episode_views(
    observation: MarketObservation,
    global_context: GlobalMarketContext | None,
    candidates: Mapping[str, HypothesisBelief],
    candidate_theses: Mapping[str, OpenMarketThesis],
    previous_contexts: Mapping[str, ContextThesisState],
    closed_contexts: Mapping[str, _ContextClosure] | None = None,
    episode_candidates: Mapping[str, HypothesisBelief] | None = None,
) -> tuple[
    dict[str, ContextThesisState],
    dict[str, EntryEpisodeState],
]:
    if global_context is None:
        return {}, {}
    grouped: dict[str, list[tuple[str, HypothesisBelief]]] = {}
    episodes: dict[str, EntryEpisodeState] = {}
    for candidate_id, candidate in candidates.items():
        context_id = candidate.context_thesis_id
        if context_id is None:
            continue
        grouped.setdefault(context_id, []).append((candidate_id, candidate))
    for candidate_id, candidate in (
        candidates
        if episode_candidates is None
        else episode_candidates
    ).items():
        context_id = candidate.context_thesis_id
        if context_id is None:
            continue
        if candidate.episode_id is None or candidate.episode_deadline is None:
            continue
        formed_at = (
            candidate.sequence.started_at
            if candidate.sequence is not None
            and candidate.sequence.started_at is not None
            else candidate.phase_started_at
        )
        episodes[candidate_id] = EntryEpisodeState(
            episode_id=candidate.episode_id,
            parent_context_thesis_id=context_id,
            candidate_id=candidate_id,
            playbook=candidate.playbook,
            direction=candidate.direction,
            initiating_event_id=candidate.initiating_event_id,
            entry_location_id=candidate.entry_location_id,
            entry_path_id=candidate.entry_path_id,
            first_pullback_at=_first_pullback_at(candidate),
            selected_trigger=candidate.selected_trigger,
            plan=candidate.plan,
            invalidation=candidate.invalidation,
            deadline=candidate.episode_deadline,
            phase=candidate.phase,
            formed_at=formed_at,
            updated_at=observation.asof,
            terminal_at=candidate.terminal_at,
            terminal_reason=candidate.terminal_reason,
        )
    contexts: dict[str, ContextThesisState] = {}
    closures = closed_contexts or {}
    for context_id, values in grouped.items():
        previous = previous_contexts.get(context_id)
        _, representative = min(
            values,
            key=lambda item: (
                item[1].phase_started_at,
                item[0],
            ),
        )
        theses = tuple(
            thesis
            for candidate_id, _ in values
            for thesis in (candidate_theses.get(candidate_id),)
            if thesis is not None
        )
        terminal_candidates = tuple(
            candidate
            for _, candidate in values
            if _context_level_terminal(
                candidate,
                candidate_theses.get(candidate.candidate_id or ""),
                previous,
            )
            is not None
            or candidate.terminal_reason
            in {
                "context_thesis_deadline_elapsed",
                "parent_context_context_thesis_deadline_elapsed",
            }
        )
        closure = closures.get(context_id)
        if closure is not None:
            lifecycle = (
                "completed"
                if closure.phase is PlaybookPhase.COMPLETED
                else "invalidated"
            )
            terminal_at = closure.terminal_at
            terminal_reason = closure.reason
        elif terminal_candidates:
            owner = min(
                terminal_candidates,
                key=lambda candidate: (
                    candidate.terminal_at or observation.asof,
                    candidate.candidate_id or "",
                ),
            )
            lifecycle = (
                "completed"
                if owner.phase is PlaybookPhase.COMPLETED
                else "invalidated"
            )
            terminal_at = owner.terminal_at or observation.asof
            terminal_reason = _normalized_context_reason(
                owner.terminal_reason or lifecycle
            )
        else:
            lifecycle = (
                "active"
                if any(candidate.context_id is not None for _, candidate in values)
                else "forming"
            )
            terminal_at = None
            terminal_reason = None
        authority_ids = (
            previous.authority_ids
            if previous is not None
            else _identity_tuple(
                representative.initiating_event_id,
                representative.context_metadata.get(
                    "lsr_displacement_id"
                ),
            )
            if representative.playbook
            is Playbook.LIQUIDITY_SWEEP_REVERSAL
            else tuple(
                source_id
                for source_id in _dfp_context_terminal_source_ids(
                    representative
                )
                if (
                    representative.thesis_draw is None
                    or source_id != representative.thesis_draw.level_id
                )
            )
            if representative.playbook
            is Playbook.DISPLACEMENT_FIRST_PULLBACK
            else tuple(
                dict.fromkeys(
                    (
                        *(
                            source_id
                            for thesis in theses
                            for source_id in thesis.authority_source_ids
                        ),
                        representative.context_id
                        or representative.required_root_id
                        or context_id,
                    )
                )
            )
        )
        supporting = tuple(
            dict.fromkeys(
                (
                    *(
                        ()
                        if previous is None
                        else previous.supporting_event_ids
                    ),
                    *(
                        identity
                        for thesis in theses
                        for identity in thesis.authority_source_ids
                    ),
                    *(
                        source_id
                        for step in (
                            ()
                            if representative.sequence is None
                            else representative.sequence.steps
                        )
                        if step.step_id == "h4_structure_and_draw"
                        for source_id in step.source_ids
                    ),
                )
            )
        )
        opposing = tuple(
            dict.fromkeys(
                (
                    *(
                        ()
                        if previous is None
                        else previous.opposing_event_ids
                    ),
                    *(
                        () if closure is None else closure.source_ids
                    ),
                )
            )
        )
        # This is the bounded current child view, not an append-only history.
        # Closed children remain visible through their one-bar candidate grace;
        # longer-run counts belong in diagnostics rather than live Brain state.
        child_ids = tuple(
            dict.fromkeys(
                episode.episode_id
                for episode in episodes.values()
                if episode.parent_context_thesis_id == context_id
            )
        )
        formed_at = (
            previous.formed_at
            if previous is not None
            else min(representative.phase_started_at, closure.terminal_at)
            if closure is not None
            else min(
                (
                    thesis.formed_at
                    for thesis in theses
                ),
                default=representative.phase_started_at,
            )
        )
        contexts[context_id] = ContextThesisState(
            context_thesis_id=context_id,
            market_epoch_id=global_context.market_epoch_id,
            direction=representative.direction,
            authority_ids=authority_ids,
            context_draw=(
                None
                if representative.playbook
                is Playbook.LIQUIDITY_SWEEP_REVERSAL
                else representative.thesis_draw
                if representative.thesis_draw is not None
                else None if previous is None else previous.context_draw
            ),
            structural_invalidation=_context_structural_invalidation(
                observation,
                representative,
                previous,
            ),
            supporting_event_ids=supporting,
            opposing_event_ids=opposing,
            lifecycle=lifecycle,
            formed_at=formed_at,
            updated_at=observation.asof,
            thesis_deadline=(
                representative.thesis_deadline
                if representative.thesis_deadline is not None
                else None if previous is None else previous.thesis_deadline
            ),
            child_episode_ids=child_ids,
            terminal_at=terminal_at,
            terminal_reason=terminal_reason,
        )
    # Context lifecycle is independent of the current bounded entry-root
    # discovery set.  An active thesis with zero children remains visible and
    # keeps its evidence until a frozen causal terminal is observed.
    for context_id, previous in previous_contexts.items():
        if (
            context_id in contexts
            or previous.market_epoch_id != global_context.market_epoch_id
            or previous.lifecycle
            in {"completed", "invalidated", "censored"}
        ):
            continue
        terminal = _retained_context_terminal(
            observation,
            global_context,
            previous,
        )
        if terminal is None:
            contexts[context_id] = replace(
                previous,
                updated_at=observation.asof,
                child_episode_ids=(),
            )
            continue
        lifecycle, reason, sources = terminal
        contexts[context_id] = replace(
            previous,
            lifecycle=lifecycle,
            updated_at=observation.asof,
            opposing_event_ids=tuple(
                dict.fromkeys((*previous.opposing_event_ids, *sources))
            ),
            child_episode_ids=(),
            terminal_at=observation.asof,
            terminal_reason=reason,
        )
    return contexts, episodes


def _shadow_path_for_thesis(
    thesis: OpenMarketThesis,
) -> PathKind | None:
    """Map only an explicit existing thesis relation to one registered path."""

    if thesis.mechanism == "range_failed_auction":
        return PathKind.FAILED_BREAKOUT
    return {
        "aligned": PathKind.CONTINUATION,
        "local_countertrend": PathKind.DEEPER_RETRACEMENT,
        "challenges_incumbent": PathKind.REVERSAL,
    }.get(thesis.authority_relation)


def _shadow_real_completed_clock(observation: MarketObservation) -> bool:
    frame = observation.frames.get(Timeframe.M1)
    geometry = None if frame is None else frame.candle_structure
    return bool(geometry is not None and geometry.real_completed)


def _shadow_dol_facts(
    observation: MarketObservation,
    context: GlobalMarketContext,
) -> tuple[
    Mapping[str, tuple[DOLCandidateFact, ...]],
    Mapping[str, DOLObstructionViewFact],
    Mapping[str, tuple[tuple[str, str], ...]],
]:
    """Adapt exact current facts without scanning or price-based fallback."""

    path_candidates: dict[str, set[PathKind]] = {}
    for thesis in context.open_market_theses:
        path = _shadow_path_for_thesis(thesis)
        if path is None or thesis.lifecycle == "invalidated":
            continue
        for candidate_id in thesis.draw_candidate_ids:
            path_candidates.setdefault(candidate_id, set()).add(path)

    inventory = _inventory_item_map(observation)
    by_direction: dict[str, tuple[DOLCandidateFact, ...]] = {}
    exclusions: dict[str, tuple[tuple[str, str], ...]] = {}
    obstruction_facts: dict[str, DOLObstructionViewFact] = {}
    for direction in Direction:
        direction_key = direction.value
        side = direction.opposing_liquidity_side
        resolved: list[DOLCandidateFact] = []
        unresolved: list[tuple[str, str]] = []
        for candidate_id in context.external_draw_candidates.get(side, ()):
            item = inventory.get(candidate_id)
            if item is None:
                unresolved.append((candidate_id, "exact_inventory_join_missing"))
                continue
            paths = path_candidates.get(candidate_id, set())
            if not paths:
                unresolved.append((candidate_id, "path_association_missing"))
                continue
            if len(paths) != 1:
                unresolved.append((candidate_id, "path_association_ambiguous"))
                continue
            if item.side != side:
                unresolved.append((candidate_id, "inventory_side_mismatch"))
                continue
            if not item.source_ids:
                unresolved.append((candidate_id, "candidate_source_ids_missing"))
                continue
            if item.structural_rank not in {"internal", "external"}:
                unresolved.append(
                    (candidate_id, "candidate_structural_rank_unregistered")
                )
                continue
            try:
                fact = DOLCandidateFact(
                    candidate_id=item.item_id,
                    timeframe=item.timeframe.value,
                    side=item.side,
                    target_price=float(item.price),
                    source_kind=item.kind,
                    source_ids=tuple(item.source_ids),
                    structural_rank=item.structural_rank,
                    strength=float(item.strength),
                    age_real_completed_bars=item.age_bars,
                    path=next(iter(paths)),
                )
            except (TypeError, ValueError):
                unresolved.append((candidate_id, "candidate_fact_contract_invalid"))
                continue
            resolved.append(fact)
        view = context.obstruction_views[direction_key]
        hard = tuple(
            DOLObstructionFact(
                obstruction_id=item.obstruction_id,
                lower_bound=float(item.lower_bound),
                upper_bound=float(item.upper_bound),
                hard=True,
                source_kind=item.source_kind,
                source_ids=tuple(item.source_ids),
            )
            for item in view.hard_barriers
        )
        soft = tuple(
            DOLObstructionFact(
                obstruction_id=item.obstruction_id,
                lower_bound=float(item.lower_bound),
                upper_bound=float(item.upper_bound),
                hard=False,
                source_kind=item.source_kind,
                source_ids=tuple(item.source_ids),
            )
            for item in view.soft_frictions
        )
        by_direction[direction_key] = tuple(resolved)
        exclusions[direction_key] = tuple(sorted(unresolved))
        obstruction_facts[direction_key] = DOLObstructionViewFact(
            direction=DOLDirection(direction_key),
            hard_barriers=hard,
            soft_frictions=soft,
        )
    return by_direction, obstruction_facts, exclusions


def _shadow_canonical_event_map(
    observation: MarketObservation,
) -> Mapping[str, MarketEvent]:
    """Return exact canonical semantic events visible to this Brain clock."""

    events_by_id: dict[str, MarketEvent] = {}
    for event in (
        *observation.recent_events,
        *observation.semantic_events_this_update,
        *(
            event
            for timeline in observation.retained_entity_timelines.values()
            for event in timeline
        ),
    ):
        if (
            event.origin is not EventOrigin.SEMANTIC_ATOMIC
            or event.semantic_version != SMC_SEMANTIC_VERSION
            or event.known_at > observation.asof
        ):
            continue
        identity_payload = {
            "semantic_version": event.semantic_version,
            "semantic_type": event.kind.value,
            "event_time": event.event_time,
            "known_at": event.known_at,
            "timeframe": event.timeframe.value,
            "side": event.side,
            "price": event.price,
            "direction": (
                None if event.direction is None else event.direction.value
            ),
            "source_event_ids": event.source_event_ids,
            "source_data_ids": event.source_data_ids,
            "source_entity_ids": event.source_entity_ids,
            "context_event_ids": event.context_event_ids,
            "origin": event.origin.value,
            "evidence": event.evidence,
            "zone": event.zone,
        }
        canonical_event_id = hashlib.sha256(
            json.dumps(
                to_primitive(identity_payload),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        if event.event_id != canonical_event_id:
            continue
        prior = events_by_id.get(event.event_id)
        if prior is not None and prior != event:
            raise ValueError("one canonical semantic event has conflicting payloads")
        events_by_id[event.event_id] = event
    return events_by_id


_PATH_RUNTIME_WINNER_RULES: Mapping[PathKind, str] = {
    PathKind.CONTINUATION: (
        "accepted_continuation_bos_bound_to_frozen_authority"
    ),
    PathKind.DEEPER_RETRACEMENT: (
        "accepted_opposed_bos_on_connected_lower_scale_while_"
        "frozen_authority_intact"
    ),
    PathKind.REVERSAL: "accepted_opposed_bos_bound_to_frozen_authority",
    PathKind.BALANCE: "authoritative_balance_range_activation",
    PathKind.FAILED_BREAKOUT: (
        "rejected_bos_crossing_bound_to_frozen_authority"
    ),
    PathKind.RESIDUAL_UNKNOWN: "never_realized",
}
_PATH_RUNTIME_FALSIFICATION_RULES: Mapping[PathKind, str] = {
    PathKind.CONTINUATION: "frozen_authority_opposed_mss_core_confirmed",
    PathKind.DEEPER_RETRACEMENT: (
        "frozen_authority_continuation_bos_confirmed"
    ),
    PathKind.REVERSAL: "frozen_authority_continuation_bos_confirmed",
    PathKind.BALANCE: "frozen_authority_directional_bos_confirmed",
    PathKind.FAILED_BREAKOUT: "realized_competitor_only",
    PathKind.RESIDUAL_UNKNOWN: "never_before_common_horizon",
}


def _validate_shadow_path_resolution_protocol(
    protocol: PathBeliefProtocol,
) -> None:
    runtime = protocol.runtime_resolution
    if (
        runtime.protocol_version != "path_runtime_resolution_phase7_v1.0"
        or runtime.entry_episode_terminal_authority is not False
        or any(
            runtime.winner_rule(path) != rule
            for path, rule in _PATH_RUNTIME_WINNER_RULES.items()
        )
        or any(
            runtime.falsification_rule(path) != rule
            for path, rule in _PATH_RUNTIME_FALSIFICATION_RULES.items()
        )
        or any(
            runtime.expiry_rule(path) != "shared_common_horizon"
            for path in PathKind
        )
    ):
        raise ValueError("unsupported path runtime resolution protocol")


def _shadow_path_resolution_events(
    observation: MarketObservation,
    context: GlobalMarketContext,
    state: PathCompetitionSetState,
    protocol: PathBeliefProtocol,
) -> tuple[tuple[PathTerminalEvent, ...], tuple[PathOutcomeEvent, ...]]:
    """Map exact global market facts into the existing pure reducer API.

    Local setup/EntryEpisode objects are intentionally absent.  A structural
    fact must bind either the competition set's frozen authority structure or
    an unambiguous connected lower-scale relation to that authority.
    """

    _validate_shadow_path_resolution_protocol(protocol)
    if (
        state.status is not PathStatus.ACTIVE
        or observation.asof <= state.asof
        or observation.asof >= state.common_expires_at
    ):
        return (), ()
    authority = next(
        (
            layer
            for layer in context.authority_stack
            if layer.structure_id == state.authority_structure_id
        ),
        None,
    )
    if authority is None:
        return (), ()

    events = _shadow_canonical_event_map(observation)
    current = tuple(
        sorted(
            (
                event
                for event in events.values()
                if event.known_at == observation.asof
            ),
            key=lambda event: event.event_id,
        )
    )
    raw_breaks = {
        event.event_id: event
        for event in events.values()
        if event.kind is EventKind.RAW_BOUNDARY_BREAK
    }

    def raw_source_structure_id(raw_break: MarketEvent) -> str | None:
        bos_id = raw_break.evidence.get("bos_id")
        target_swing_id = raw_break.evidence.get("target_swing_id")
        break_bar_id = raw_break.evidence.get("break_bar_id")
        displacement_id = raw_break.evidence.get("source_displacement_id")
        entity_ids = raw_break.source_entity_ids
        valid_entities = bool(
            isinstance(bos_id, str)
            and bos_id
            and isinstance(target_swing_id, str)
            and target_swing_id
            and len(entity_ids) in {3, 4}
            and entity_ids[0] == bos_id
            and entity_ids[1] == target_swing_id
            and (
                (displacement_id is None and len(entity_ids) == 3)
                or (
                    isinstance(displacement_id, str)
                    and displacement_id
                    and len(entity_ids) == 4
                    and entity_ids[3] == displacement_id
                )
            )
        )
        if (
            not valid_entities
            or raw_break.direction is None
            or raw_break.evidence.get("scope")
            not in {"continuation", "opposed"}
            or raw_break.evidence.get("break_standard")
            != "close_beyond_confirmed_boundary"
            or not isinstance(break_bar_id, str)
            or not break_bar_id
            or raw_break.source_data_ids != (break_bar_id,)
            or len(raw_break.source_event_ids) != 2
            or not raw_break.context_event_ids
            or raw_break.event_time != raw_break.known_at
        ):
            return None
        return entity_ids[2]

    def structural_raw_parent(
        event: MarketEvent,
    ) -> tuple[MarketEvent, str] | None:
        bos_id = event.evidence.get("bos_id")
        candidates = tuple(
            raw_breaks[event_id]
            for event_id in event.source_event_ids
            if event_id in raw_breaks
            and raw_breaks[event_id].evidence.get("bos_id") == bos_id
        )
        if len(candidates) != 1:
            return None
        raw_break = candidates[0]
        source_structure_id = raw_source_structure_id(raw_break)
        if (
            source_structure_id is None
            or raw_break.known_at != event.known_at
            or raw_break.timeframe is not event.timeframe
            or raw_break.side != event.side
            or raw_break.price != event.price
            or raw_break.direction is not event.direction
            or raw_break.evidence.get("scope") != event.evidence.get("scope")
            or len(event.source_event_ids) != 2
            or event.source_entity_ids != (bos_id, source_structure_id)
        ):
            return None
        return raw_break, source_structure_id

    terminal_sources: dict[PathKind, tuple[str, set[str]]] = {}

    def add_terminal(path: PathKind, reason: str, source_id: str) -> None:
        if state.member(path).status is not PathStatus.ACTIVE:
            return
        prior = terminal_sources.get(path)
        if prior is not None and prior[0] != reason:
            raise ValueError("one path has conflicting global falsification facts")
        sources = {source_id} if prior is None else prior[1]
        sources.add(source_id)
        terminal_sources[path] = (reason, sources)

    for event in current:
        raw_parent = structural_raw_parent(event)
        bound_to_authority = bool(
            raw_parent is not None
            and raw_parent[1] == state.authority_structure_id
        )
        if not bound_to_authority:
            continue
        if (
            event.kind is EventKind.MSS_CORE_CONFIRMED
            and event.evidence.get("scope") == "opposed"
        ):
            add_terminal(
                PathKind.CONTINUATION,
                "frozen_authority_opposed_mss_core_confirmed",
                event.event_id,
            )
            add_terminal(
                PathKind.BALANCE,
                "frozen_authority_directional_bos_confirmed",
                event.event_id,
            )
        elif (
            event.kind is EventKind.QUALIFIED_BOS
            and event.evidence.get("scope") == "continuation"
        ):
            for path in (PathKind.DEEPER_RETRACEMENT, PathKind.REVERSAL):
                add_terminal(
                    path,
                    "frozen_authority_continuation_bos_confirmed",
                    event.event_id,
                )
            add_terminal(
                PathKind.BALANCE,
                "frozen_authority_directional_bos_confirmed",
                event.event_id,
            )

    def raw_break_for(
        event: MarketEvent,
    ) -> tuple[MarketEvent, str] | None:
        """Resolve only the exact BOS crossing shape emitted by Observation.

        A BOS terminal crossing owns penetration/bar parents; the earlier raw
        break is contextual ancestry.  Requiring that production shape keeps
        an unrelated acceptance/sweep (or a canonical-looking forged event)
        from deciding the global path competition.
        """

        bos_id = event.evidence.get("bos_id")
        if not isinstance(bos_id, str) or not bos_id:
            return None
        candidates = tuple(
            raw_breaks[event_id]
            for event_id in event.context_event_ids
            if event_id in raw_breaks
            and raw_breaks[event_id].evidence.get("bos_id") == bos_id
        )
        if len(candidates) != 1:
            return None
        raw_break = candidates[0]
        target_swing_id = raw_break.evidence.get("target_swing_id")
        level_id = (
            None
            if not isinstance(target_swing_id, str) or not target_swing_id
            else f"swing:{target_swing_id}"
        )
        source_structure_id = raw_source_structure_id(raw_break)
        crossing_generation = (
            None
            if level_id is None
            else hashlib.sha256(
                (
                    f"{SMC_SEMANTIC_VERSION}|crossing-v1|"
                    f"{event.timeframe.value}|{level_id}|"
                    f"{raw_break.known_at.isoformat()}"
                ).encode("utf-8")
            ).hexdigest()[:24]
        )
        expected_resolution = (
            "held_outside"
            if event.kind is EventKind.ACCEPTANCE_CONFIRMED
            else "returned_inside"
        )
        direction_matches = (
            event.direction is raw_break.direction
            if event.kind is EventKind.ACCEPTANCE_CONFIRMED
            else event.direction is not None
            and raw_break.direction is not None
            and event.direction is not raw_break.direction
        )
        if (
            source_structure_id is None
            or raw_break.known_at >= event.known_at
            or event.event_time != raw_break.known_at
            or event.timeframe is not raw_break.timeframe
            or event.side != raw_break.side
            or event.price != raw_break.price
            or not direction_matches
            or len(event.source_event_ids) != 2
            or len(event.context_event_ids) < 2
            or event.evidence.get("level_id") != level_id
            or event.evidence.get("target_swing_id") != target_swing_id
            or event.source_entity_ids != (level_id,)
            or event.evidence.get("crossing_generation_id")
            != crossing_generation
            or event.evidence.get("crossed_at")
            != raw_break.known_at.isoformat()
            or event.evidence.get("resolved_at")
            != event.known_at.isoformat()
            or event.evidence.get("resolution_bars") != 1
            or event.evidence.get("resolution") != expected_resolution
        ):
            return None
        return raw_break, source_structure_id

    winners: list[tuple[PathKind, str, tuple[str, ...]]] = []
    timeframe_rank = {
        Timeframe.M1: 0,
        Timeframe.M5: 1,
        Timeframe.M15: 2,
        Timeframe.H1: 3,
        Timeframe.H4: 4,
    }
    for event in current:
        if event.kind not in {
            EventKind.ACCEPTANCE_CONFIRMED,
            EventKind.SWEEP_CONFIRMED,
        }:
            continue
        raw_parent = raw_break_for(event)
        if raw_parent is None:
            continue
        raw_break, source_structure_id = raw_parent
        source_ids = tuple(sorted((event.event_id, raw_break.event_id)))
        bound_to_authority = (
            source_structure_id == state.authority_structure_id
        )
        scope = raw_break.evidence.get("scope")
        if event.kind is EventKind.SWEEP_CONFIRMED:
            if (
                bound_to_authority
                and event.evidence.get("resolution") == "returned_inside"
            ):
                winners.append(
                    (
                        PathKind.FAILED_BREAKOUT,
                        "rejected_bos_crossing_bound_to_frozen_authority",
                        source_ids,
                    )
                )
            continue
        if event.evidence.get("resolution") != "held_outside":
            continue
        if bound_to_authority and scope == "continuation":
            winners.append(
                (
                    PathKind.CONTINUATION,
                    "accepted_continuation_bos_bound_to_frozen_authority",
                    source_ids,
                )
            )
            continue
        if bound_to_authority and scope == "opposed":
            winners.append(
                (
                    PathKind.REVERSAL,
                    "accepted_opposed_bos_bound_to_frozen_authority",
                    source_ids,
                )
            )
            continue
        relation = context.scale_relation_details.get(event.timeframe.value)
        if (
            scope == "opposed"
            and event.direction is not None
            and event.direction is not authority.direction
            and authority.status == "intact"
            and context.dominant_authority_layer is not None
            and context.dominant_authority_layer.structure_id
            == state.authority_structure_id
            and timeframe_rank[event.timeframe]
            < timeframe_rank[authority.timeframe]
            and relation is not None
            and relation.authority_layer_id == state.authority_structure_id
            and relation.direction is event.direction
            and not set(relation.evidence_ids).isdisjoint(
                {
                    str(raw_break.evidence["bos_id"]),
                    raw_break.event_id,
                }
            )
            and relation.graph_connected
            and not relation.ambiguous
            and relation.relation
            in {
                ScaleRelation.NORMAL_PULLBACK,
                ScaleRelation.MATERIAL_OPPOSITION,
            }
        ):
            winners.append(
                (
                    PathKind.DEEPER_RETRACEMENT,
                    (
                        "accepted_opposed_bos_on_connected_lower_scale_"
                        "while_frozen_authority_intact"
                    ),
                    source_ids,
                )
            )

    balance = context.balance_context
    if balance is not None and balance.status == "authoritative":
        for event in current:
            if (
                event.kind is EventKind.DEALING_RANGE_ACTIVATED
                and event.timeframe is balance.timeframe
                and (
                    event.evidence.get("range_id") == balance.context_id
                    or balance.context_id in event.source_entity_ids
                )
            ):
                winners.append(
                    (
                        PathKind.BALANCE,
                        "authoritative_balance_range_activation",
                        (event.event_id,),
                    )
                )

    distinct_winners = {path for path, _, _ in winners}
    if len(distinct_winners) > 1:
        raise ValueError("same clock has ambiguous path realized-winner facts")
    outcomes: tuple[PathOutcomeEvent, ...] = ()
    if winners:
        winner = winners[0][0]
        reason = winners[0][1]
        if any(item[1] != reason for item in winners):
            raise ValueError("one path has conflicting realized-winner rules")
        if state.member(winner).status is not PathStatus.ACTIVE:
            raise ValueError("a falsified path cannot become the realized winner")
        outcome_sources = tuple(
            sorted({source for _, _, sources in winners for source in sources})
        )
        outcomes = (
            protocol.make_outcome_event(
                competition_set_id=state.competition_set_id,
                winner_path=winner,
                reason=reason,
                source_event_ids=outcome_sources,
                known_at=observation.asof,
            ),
        )
    # The frozen reducer contract applies same-clock path terminals before
    # descriptive evidence or an outcome.  A path cannot be realized on the
    # same clock that an exact global fact falsifies that path.
    if outcomes and outcomes[0].winner_path in terminal_sources:
        outcomes = ()

    terminals = tuple(
        protocol.make_terminal_event(
            competition_set_id=state.competition_set_id,
            path=path,
            rule_id="registered_path_invalidation",
            reason=reason,
            source_event_ids=tuple(sorted(sources)),
            known_at=observation.asof,
        )
        for path, (reason, sources) in sorted(
            terminal_sources.items(),
            key=lambda item: item[0].value,
        )
        if not outcomes or outcomes[0].winner_path is not path
    )
    return terminals, outcomes


def _shadow_path_contribution_specs(
    observation: MarketObservation,
    context: GlobalMarketContext,
    *,
    seen_tokens: Collection[tuple[str, ...]],
    unresolved_dependency_cluster: str | None = None,
) -> tuple[
    tuple[tuple[str, ...], str, tuple[str, ...], str], ...
]:
    """Admit only exact Phase 6 mechanism events into the path ledger.

    DFP/LSR/FAVR candidates, open-thesis projections, DOL obstacles and local
    EntryEpisode lifecycles are deliberately absent.  Phase 6 admitted the two
    mechanism families as evidence sources; it did not estimate path
    likelihoods, so the production protocol assigns neutral increments.  A
    synthetic/admitted likelihood protocol without a registered dependency
    resolver receives one conservative competition-set cluster: it cannot
    multiply repeated or cross-family facts by default.
    """

    del context  # Scope ownership remains with the caller, not a playbook.
    events_by_id = {
        event_id: event
        for event_id, event in _shadow_canonical_event_map(observation).items()
        if event.known_at == observation.asof
    }

    output: list[tuple[tuple[str, ...], str, tuple[str, ...], str]] = []
    for event in sorted(events_by_id.values(), key=lambda item: item.event_id):
        if event.timeframe is not Timeframe.M5:
            continue
        if event.kind is EventKind.ACCEPTANCE_CONFIRMED:
            family_instance = event.evidence.get("crossing_generation_id")
            rule_id = "acceptance_continuation"
        elif (
            event.kind is EventKind.DISPLACEMENT_OBSERVED
            and event.evidence.get("lifecycle") == "active"
        ):
            family_instance = event.evidence.get("displacement_id")
            rule_id = "displacement_impact"
        else:
            continue
        if not isinstance(family_instance, str) or not family_instance:
            continue
        source_ids = (event.event_id,)
        # The correlation namespace is global.  Rule/family labels describe
        # provenance; they must not make one upstream fact independent merely
        # by changing the evidence family.
        correlation_key = (
            f"semantic-entity:{family_instance}"
            if unresolved_dependency_cluster is None
            else unresolved_dependency_cluster
        )
        token = (
            "phase6_admitted",
            rule_id,
            correlation_key,
            "source_event_ids",
            event.event_id,
        )
        if token not in seen_tokens:
            output.append((token, rule_id, source_ids, correlation_key))
    return tuple(output)


class PlaybookBrain:
    """Updates six playbook-direction beliefs; it never chooses an action.

    Omitting Scene Graph inputs is an isolated evaluator/unit-test mode.  The
    runtime Engine always supplies both graph and delta, where exact open-
    thesis root binding is required before a hypothesis can be executable.
    """

    def __init__(
        self,
        config: BrainConfig | None = None,
        registry: PlaybookRegistry | None = None,
        calibrator: TypedBrainCalibrator | None = None,
        path_hypotheses_protocol: str | Path = "configs/path_hypotheses.json",
        expected_path_protocol_fingerprint: str | None = None,
        expected_dol_protocol_fingerprint: str | None = None,
        dol_probability_protocol: str | Path = "configs/dol_probability.json",
        expected_dol_probability_protocol_fingerprint: str | None = None,
        dol_probability_model_artifact: DOLProbabilityModelArtifact | None = None,
        signal_policy_protocol: str | Path = "configs/signal_policy.json",
        expected_signal_policy_fingerprint: str | None = None,
        path_likelihood_artifact: AdmittedPathLikelihoodArtifact | None = None,
        dol_calibration_artifact: AdmittedDOLCalibrationArtifact | None = None,
        outcome_model_artifact: TargetBeforeInvalidationArtifact | None = None,
        signal_artifact_pins: SignalArtifactPins | None = None,
    ) -> None:
        self.config = config or BrainConfig()
        self.registry = registry or load_playbook_registry()
        self.calibrator = calibrator or TypedBrainCalibrator.identity()
        if not isinstance(
            self.calibrator,
            TypedBrainCalibrator,
        ):
            raise TypeError("brain requires TypedBrainCalibrator")
        if (
            self.calibrator.registry_hash is not None
            and self.calibrator.registry_hash != self.registry.fingerprint
        ):
            raise ValueError("brain calibration does not match playbook registry")
        self.path_protocol: PathBeliefProtocol = load_path_belief_protocol(
            path_hypotheses_protocol
        )
        self.dol_protocol: DOLRankingProtocol = load_dol_ranking_protocol(
            path_hypotheses_protocol
        )
        if (
            expected_path_protocol_fingerprint is not None
            and expected_path_protocol_fingerprint
            != self.path_protocol.fingerprint
        ):
            raise ValueError("brain path protocol fingerprint is stale")
        if (
            expected_dol_protocol_fingerprint is not None
            and expected_dol_protocol_fingerprint
            != self.dol_protocol.fingerprint
        ):
            raise ValueError("brain DOL protocol fingerprint is stale")
        self.dol_probability_protocol: DOLProbabilityProtocol = (
            load_dol_probability_protocol(dol_probability_protocol)
        )
        if (
            self.dol_probability_protocol.ranking_protocol_fingerprint
            != self.dol_protocol.fingerprint
        ):
            raise ValueError(
                "brain DOL probability and ranking protocols disagree"
            )
        if (
            expected_dol_probability_protocol_fingerprint is not None
            and expected_dol_probability_protocol_fingerprint
            != self.dol_probability_protocol.fingerprint
        ):
            raise ValueError("brain DOL probability protocol fingerprint is stale")
        if (
            dol_probability_model_artifact is not None
            and (
                not isinstance(
                    dol_probability_model_artifact,
                    DOLProbabilityModelArtifact,
                )
                or dol_probability_model_artifact.protocol_fingerprint
                != self.dol_probability_protocol.fingerprint
                or dol_probability_model_artifact.ranking_protocol_fingerprint
                != self.dol_protocol.fingerprint
            )
        ):
            raise ValueError("brain DOL probability artifact binding is stale")
        self.dol_probability_model_artifact = dol_probability_model_artifact
        self.signal_policy: SignalPolicyProtocol = load_signal_policy_protocol(
            signal_policy_protocol
        )
        if (
            expected_signal_policy_fingerprint is not None
            and expected_signal_policy_fingerprint != self.signal_policy.fingerprint
        ):
            raise ValueError("brain Signal Policy fingerprint is stale")
        admission_types = (
            (path_likelihood_artifact, AdmittedPathLikelihoodArtifact),
            (dol_calibration_artifact, AdmittedDOLCalibrationArtifact),
            (outcome_model_artifact, TargetBeforeInvalidationArtifact),
        )
        if any(
            artifact is not None and not isinstance(artifact, expected)
            for artifact, expected in admission_types
        ):
            raise TypeError("brain shadow admissions must use exact artifact types")
        supplied_admission = any(
            artifact is not None
            for artifact, _ in admission_types
        ) or signal_artifact_pins is not None
        if self.path_protocol.can_apply_bayesian_update and not supplied_admission:
            raise ValueError(
                "Bayesian path updates require a separately pinned artifact set"
            )
        if supplied_admission and (
            path_likelihood_artifact is None
            or dol_calibration_artifact is None
            or outcome_model_artifact is None
            or not isinstance(signal_artifact_pins, SignalArtifactPins)
            or dol_probability_model_artifact is None
            or not self.path_protocol.can_apply_bayesian_update
            or path_likelihood_artifact.source_path_protocol_fingerprint
            != self.path_protocol.fingerprint
            or path_likelihood_artifact.source_path_model_version
            != self.path_protocol.model_version
            or dol_probability_model_artifact.source_path_protocol_fingerprint
            != self.path_protocol.fingerprint
            or dol_probability_model_artifact.source_path_model_version
            != self.path_protocol.model_version
            or outcome_model_artifact.path_likelihood_artifact_id
            != path_likelihood_artifact.artifact_id
            or outcome_model_artifact.dol_calibration_artifact_id
            != dol_calibration_artifact.artifact_id
            or outcome_model_artifact.signal_policy_fingerprint
            != self.signal_policy.fingerprint
            or signal_artifact_pins.signal_policy_fingerprint
            != self.signal_policy.fingerprint
            or signal_artifact_pins.path_protocol_fingerprint
            != self.path_protocol.fingerprint
            or signal_artifact_pins.path_likelihood_artifact_id
            != path_likelihood_artifact.artifact_id
            or signal_artifact_pins.dol_calibration_artifact_id
            != dol_calibration_artifact.artifact_id
            or signal_artifact_pins.outcome_model_artifact_id
            != outcome_model_artifact.artifact_id
            or signal_artifact_pins.dol_probability_model_fingerprint
            != dol_probability_model_artifact.fingerprint
        ):
            raise ValueError("brain shadow artifact admission binding is stale")
        self.path_likelihood_artifact = path_likelihood_artifact
        self.dol_calibration_artifact = dol_calibration_artifact
        self.outcome_model_artifact = outcome_model_artifact
        self.signal_artifact_pins = signal_artifact_pins
        self._belief: MarketBelief | None = None
        self.hypothesis_manager = HypothesisManager(self.path_protocol)
        self._path_competition_state: PathCompetitionSetState | None = None
        self._path_scope_key: tuple[str, ...] | None = None
        self._path_seen_evidence_tokens: set[tuple[str, ...]] = set()
        self._dol_probability_results: dict[str, DOLProbabilityResult] = {}
        self._signal_real_completed_bar_anchors: dict[
            str,
            tuple[str, str, pd.Timestamp, int],
        ] = {}
        self._candidate_priors: dict[str, HypothesisBelief] = {}
        self._candidate_theses: dict[str, OpenMarketThesis] = {}
        self._candidate_epoch_id: str | None = None
        self._closed_contexts: dict[str, _ContextClosure] = {}
        self._lsr_execution_winners: dict[str, str] = {}
        self._lsr_execution_owner_snapshots: dict[
            str, _LSRExecutionOwner
        ] = {}
        self._lsr_context_seeds: dict[
            tuple[str, str, Direction], _LSRCandidate
        ] = {}
        self._lsr_context_seed_ids: dict[
            tuple[str, str, Direction], str
        ] = {}
        self._lsr_retired_seed_keys: set[
            tuple[str, str, Direction]
        ] = set()
        self._lsr_context_bindings: dict[
            tuple[str, str, Direction], _LSRContextBinding
        ] = {}
        self._lsr_terminal_episode_ids: dict[str, set[str]] = {}

    @property
    def current(self) -> MarketBelief | None:
        return self._belief

    def reset(self) -> None:
        self._belief = None
        self.hypothesis_manager.reset()
        self._path_competition_state = None
        self._path_scope_key = None
        self._path_seen_evidence_tokens.clear()
        self._dol_probability_results.clear()
        self._signal_real_completed_bar_anchors.clear()
        self._candidate_priors.clear()
        self._candidate_theses.clear()
        self._candidate_epoch_id = None
        self._closed_contexts.clear()
        self._lsr_execution_winners.clear()
        self._lsr_execution_owner_snapshots.clear()
        self._lsr_context_seeds.clear()
        self._lsr_context_seed_ids.clear()
        self._lsr_retired_seed_keys.clear()
        self._lsr_context_bindings.clear()
        self._lsr_terminal_episode_ids.clear()

    def _update_shadow_path_diagnostics(
        self,
        observation: MarketObservation,
        global_context: GlobalMarketContext | None,
        context_theses: Mapping[str, ContextThesisState],
        lifecycle_candidates: Mapping[str, HypothesisBelief],
        previous_belief: MarketBelief | None,
    ) -> tuple[
        PathCompetitionSetState | None,
        tuple[PathBeliefUpdateRecord, ...],
        Mapping[str, DOLRankingResult],
        Mapping[str, tuple[tuple[str, str], ...]],
    ]:
        """Advance the existing Brain's path diagnostics exactly once."""

        # Typed setup/entry lifecycles remain available to the legacy Brain,
        # but cannot update or terminalize a market-path competition set.
        del context_theses, lifecycle_candidates, previous_belief

        authority = (
            None
            if global_context is None
            else global_context.dominant_authority_layer
        )
        if global_context is None or authority is None:
            self.hypothesis_manager.reset()
            self._path_competition_state = None
            self._path_scope_key = None
            self._path_seen_evidence_tokens.clear()
            self._dol_probability_results.clear()
            self._signal_real_completed_bar_anchors.clear()
            return None, (), {}, {}

        instrument_id = f"{observation.symbol}:{observation.instrument_id}"
        state = self._path_competition_state
        horizon = _market_lifecycle_deadline(observation.asof)
        if horizon <= observation.asof:
            if (
                state is not None
                and state.status is PathStatus.ACTIVE
                and observation.asof >= state.common_expires_at
            ):
                if observation.asof <= state.asof:
                    raise ValueError("path horizon clock did not advance")
                if self.hypothesis_manager.state != state:
                    raise ValueError("hypothesis manager state is out of sync")
                state, record = self.hypothesis_manager.advance(
                    asof=observation.asof,
                    real_completed_bar=_shadow_real_completed_clock(
                        observation
                    ),
                )
                self._path_competition_state = state
                self._dol_probability_results.clear()
                self._signal_real_completed_bar_anchors.clear()
                return state, (record,), {}, {}
            self.hypothesis_manager.reset()
            self._path_competition_state = None
            self._path_scope_key = None
            self._path_seen_evidence_tokens.clear()
            self._dol_probability_results.clear()
            self._signal_real_completed_bar_anchors.clear()
            return None, (), {}, {}
        horizon_id = f"market-session:{horizon.isoformat()}"
        scope_key = (
            instrument_id,
            global_context.market_epoch_id,
            authority.structure_id,
            horizon_id,
            self.path_protocol.fingerprint,
        )
        new_scope = state is None or self._path_scope_key != scope_key
        if (
            not new_scope
            and state is not None
            and state.status is not PathStatus.ACTIVE
        ):
            if observation.asof < state.asof:
                raise ValueError("path diagnostic clock moved backwards")
            carried = (
                state
                if observation.asof == state.asof
                else replace(state, asof=observation.asof)
            )
            self.hypothesis_manager.state = carried
            self._path_competition_state = carried
            self._dol_probability_results.clear()
            self._signal_real_completed_bar_anchors.clear()
            return carried, (), {}, {}
        working_manager = self.hypothesis_manager
        working_seen_tokens = set(self._path_seen_evidence_tokens)
        if new_scope:
            working_manager = HypothesisManager(self.path_protocol)
            working_seen_tokens = set()
            state = create_path_competition_set(
                self.path_protocol,
                instrument_id=instrument_id,
                market_epoch_id=global_context.market_epoch_id,
                authority_structure_id=authority.structure_id,
                horizon_id=horizon_id,
                formed_at=observation.asof,
                common_expires_at=horizon,
            )
        else:
            working_manager = self.hypothesis_manager.fork()
        assert state is not None

        candidate_facts, obstruction_facts, exclusions = _shadow_dol_facts(
            observation,
            global_context,
        )

        def rankings_for(
            current: PathCompetitionSetState,
        ) -> dict[str, DOLRankingResult]:
            return {
                direction.value: rank_dol_candidates(
                    self.dol_protocol,
                    direction=DOLDirection(direction.value),
                    current_price=float(observation.price),
                    external_draw_candidates=candidate_facts[direction.value],
                    obstruction_view=obstruction_facts[direction.value],
                    path_state=current,
                )
                for direction in Direction
            }

        def probabilities_for(
            current: PathCompetitionSetState,
        ) -> dict[str, DOLProbabilityResult]:
            # Candidate ranking remains available as a development feature,
            # but an unfitted heuristic must never be published by the Brain
            # as a probability distribution.  The standalone marginalizer
            # also requires an exact fitted/admitted model artifact; runtime
            # projection opens under that same fail-closed boundary.
            if (
                current.status is not PathStatus.ACTIVE
                or self.dol_probability_model_artifact is None
            ):
                return {}
            return {
                direction.value: marginalize_dol_probabilities(
                    self.dol_probability_protocol,
                    ranking_protocol=self.dol_protocol,
                    direction=DOLDirection(direction.value),
                    current_price=float(observation.price),
                    external_draw_candidates=candidate_facts[direction.value],
                    obstruction_view=obstruction_facts[direction.value],
                    path_state=current,
                    model_artifact=self.dol_probability_model_artifact,
                )
                for direction in Direction
            }

        specs = list(
            _shadow_path_contribution_specs(
                observation,
                global_context,
                seen_tokens=working_seen_tokens,
                unresolved_dependency_cluster=(
                    None
                    if not self.path_protocol.can_apply_bayesian_update
                    else (
                        "unresolved-dependency-cluster:"
                        f"{state.competition_set_id}"
                    )
                ),
            )
        )
        # DOL obstruction, setup and EntryEpisode facts are consumers of the
        # path state, never evidence that changes market-path probability.
        specs = sorted(specs, key=lambda value: value[0])
        contributions = tuple(
            self.path_protocol.make_contribution(
                competition_set_id=state.competition_set_id,
                rule_id=rule_id,
                source_event_ids=source_ids,
                known_at=observation.asof,
                correlation_key=correlation_key,
                require_admitted=True,
            )
            for _, rule_id, source_ids, correlation_key in specs
        )
        real_completed = _shadow_real_completed_clock(observation)
        records: tuple[PathBeliefUpdateRecord, ...]
        if new_scope:
            state, record = working_manager.initialize_state(
                state,
                contributions=contributions,
                real_completed_bar=real_completed,
            )
            records = (record,)
        else:
            terminal_events, outcome_events = _shadow_path_resolution_events(
                observation,
                global_context,
                state,
                self.path_protocol,
            )
            if observation.asof < state.asof:
                raise ValueError("path diagnostic clock moved backwards")
            if observation.asof == state.asof:
                if contributions or terminal_events or outcome_events:
                    raise ValueError(
                        "same-clock path evidence changed after initialization"
                    )
                records = ()
            else:
                if working_manager.state != state:
                    raise ValueError("hypothesis manager state is out of sync")
                state, record = working_manager.advance(
                    asof=observation.asof,
                    contributions=contributions,
                    terminal_events=terminal_events,
                    outcome_events=outcome_events,
                    real_completed_bar=real_completed,
                )
                records = (record,)
        active = state.status is PathStatus.ACTIVE
        rankings = rankings_for(state) if active else {}
        probabilities = probabilities_for(state) if active else {}
        self.hypothesis_manager = working_manager
        self._path_scope_key = scope_key
        self._path_seen_evidence_tokens = set(working_seen_tokens)
        self._path_seen_evidence_tokens.update(
            token for token, _, _, _ in specs
        )
        self._path_competition_state = state
        self._dol_probability_results = probabilities
        if new_scope:
            self._signal_real_completed_bar_anchors.clear()
        return state, records, rankings, (exclusions if active else {})

    def _shadow_admission_diagnostics(self) -> tuple[str, ...]:
        diagnostics: list[str] = []
        if self.dol_probability_model_artifact is None:
            diagnostics.append("dol_probability_model_artifact_missing")
        if self.path_likelihood_artifact is None:
            diagnostics.append("path_likelihood_artifact_missing")
        if self.dol_calibration_artifact is None:
            diagnostics.append("dol_calibration_artifact_missing")
        if self.outcome_model_artifact is None:
            diagnostics.append("outcome_model_artifact_missing")
        if self.signal_artifact_pins is None:
            diagnostics.append("signal_artifact_pins_missing")
        return tuple(diagnostics) or ("all_shadow_artifacts_admitted",)

    def _project_shadow_signal_assessments(
        self,
        observation: MarketObservation,
        belief: MarketBelief,
    ) -> MarketBelief:
        """Evaluate current typed setups without touching action candidates."""

        if belief.asof != observation.asof:
            raise ValueError("shadow signal projection requires current belief")
        diagnostics: dict[str, tuple[str, ...]] = {
            "__admission__": self._shadow_admission_diagnostics(),
        }
        assessments: dict[str, SignalAssessment] = {}
        state = belief.path_competition_state
        if state is None or state.status is not PathStatus.ACTIVE:
            self._signal_real_completed_bar_anchors.clear()
            diagnostics["__path__"] = ("active_path_competition_set_missing",)
            return replace(
                belief,
                signal_assessments=assessments,
                trade_intents={},
                shadow_signal_rejections=diagnostics,
            )

        live_anchor_ids = {
            candidate_id
            for candidate_id, candidate in belief.thesis_candidates.items()
            for episode in (belief.entry_episodes.get(candidate_id),)
            if (
                candidate.phase is PlaybookPhase.EXECUTABLE
                and episode is not None
                and episode.phase is PlaybookPhase.EXECUTABLE
                and candidate.candidate_id == episode.candidate_id
                and candidate.episode_id == episode.episode_id
                and candidate.plan == episode.plan
            )
        }
        self._signal_real_completed_bar_anchors = {
            candidate_id: anchor
            for candidate_id, anchor in (
                self._signal_real_completed_bar_anchors.items()
            )
            if candidate_id in live_anchor_ids
            and anchor[0] == state.competition_set_id
        }
        anomaly_ids = tuple(
            sorted(
                set(
                    (*observation.anomalies, *observation.execution.anomalies)
                )
            )
        )
        for candidate_id, candidate in sorted(belief.thesis_candidates.items()):
            if candidate.phase is not PlaybookPhase.EXECUTABLE:
                continue
            episode = belief.entry_episodes.get(candidate_id)
            if episode is None:
                diagnostics[candidate_id] = ("typed_entry_episode_missing",)
                continue
            exact_lifecycle = bool(
                candidate.candidate_id == episode.candidate_id
                and candidate.episode_id == episode.episode_id
                and candidate.plan == episode.plan
                and episode.phase is PlaybookPhase.EXECUTABLE
            )
            real_completed_age: int | None = None
            if exact_lifecycle:
                anchor_identity = (
                    state.competition_set_id,
                    episode.episode_id,
                    candidate.phase_started_at,
                )
                anchor = self._signal_real_completed_bar_anchors.get(candidate_id)
                if (
                    anchor is None
                    or anchor[:3] != anchor_identity
                    or anchor[3] > state.real_completed_bar_count
                ):
                    anchor = (*anchor_identity, state.real_completed_bar_count)
                    self._signal_real_completed_bar_anchors[candidate_id] = anchor
                real_completed_age = state.real_completed_bar_count - anchor[3]
            probability = belief.dol_probabilities.get(candidate.direction.value)
            if probability is None:
                diagnostics[candidate_id] = (
                    "current_dol_probability_result_missing",
                )
                continue
            plan = candidate.plan
            if plan is None or not plan.selected_draw_id:
                diagnostics[candidate_id] = ("trade_plan_draw_missing",)
                continue
            candidate_draw_id = plan.selected_draw_id
            source_ids = tuple(
                sorted(
                    {
                        value
                        for value in (
                            candidate.initiating_event_id,
                            episode.initiating_event_id,
                        )
                        if value
                    }
                )
            )
            if not source_ids:
                diagnostics[candidate_id] = (
                    "initiating_event_source_missing",
                )
                continue
            risk_points = None if plan is None else float(plan.risk_points)
            estimated_cost_R = (
                None
                if risk_points is None or risk_points <= 0.0
                else float(
                    observation.execution.expected_round_trip_cost_points
                )
                / risk_points
            )
            evaluation_payload = {
                "asof": observation.asof.isoformat(),
                "candidate_id": candidate_id,
                "execution_source": observation.execution.source,
                "execution_data_age_seconds": (
                    observation.execution.data_age_seconds
                ),
                "anomalies": anomaly_ids,
                "cost_R": estimated_cost_R,
            }
            evaluation_id = hashlib.sha256(
                json.dumps(
                    evaluation_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24]
            try:
                assessment = assess_signal(
                    self.signal_policy,
                    asof=observation.asof,
                    symbol=observation.symbol,
                    instrument_id=state.instrument_id,
                    path_state=state,
                    dol_ranking=probability,
                    dol_candidate_id=candidate_draw_id,
                    setup_candidate=candidate,
                    entry_episode=episode,
                    evaluation=SignalEvaluationContext(
                        coverage_id=f"signal-coverage:{evaluation_id}",
                        coverage_fraction=0.0 if anomaly_ids else 1.0,
                        ood_assessment_id=f"signal-ood:{evaluation_id}",
                        out_of_distribution=bool(anomaly_ids),
                        cost_estimate_id=(
                            None
                            if estimated_cost_R is None
                            else f"signal-cost:{evaluation_id}"
                        ),
                        estimated_cost_R=estimated_cost_R,
                        source_event_ids=source_ids,
                        real_completed_bars_since_setup=real_completed_age,
                    ),
                    path_likelihood_artifact=self.path_likelihood_artifact,
                    dol_calibration_artifact=self.dol_calibration_artifact,
                    outcome_model_artifact=self.outcome_model_artifact,
                    dol_probability_model_artifact=(
                        self.dol_probability_model_artifact
                    ),
                    path_protocol=self.path_protocol,
                    artifact_pins=self.signal_artifact_pins,
                )
            except (TypeError, ValueError) as error:
                # This append-only component cannot take down the established
                # Brain/Decision path.  Contract mismatches remain explicit
                # shadow diagnostics and produce no intent.
                diagnostics[candidate_id] = (
                    f"signal_assessment_not_built:{error}",
                )
                continue
            assessments[candidate_id] = assessment
            if assessment.rejection_reasons:
                diagnostics[candidate_id] = tuple(
                    reason.value for reason in assessment.rejection_reasons
                )
        return replace(
            belief,
            signal_assessments=assessments,
            trade_intents={},
            shadow_signal_rejections=diagnostics,
        )

    def project_shadow_trade_intents(
        self,
        belief: MarketBelief,
        account: AccountState,
    ) -> MarketBelief:
        """Freeze eligible shadow intents; never call Decision/Risk/Execution."""

        if not isinstance(account, AccountState):
            raise TypeError("shadow Trade Intent projection requires AccountState")
        if self._belief is None or belief != self._belief:
            raise ValueError("shadow Trade Intent projection requires current belief")
        intents: dict[str, TradeIntent] = {}
        diagnostics = {
            identity: tuple(values)
            for identity, values in belief.shadow_signal_rejections.items()
        }
        account_payload = {
            "asof": belief.asof.isoformat(),
            "equity": account.equity,
            "open_risk_fraction": account.open_risk_fraction,
            "requested_risk_fraction": account.requested_risk_fraction,
            "quantity": account.quantity,
            "point_value": account.point_value,
            "position": to_primitive(account.position),
        }
        account_id = hashlib.sha256(
            json.dumps(
                account_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        methods = (
            EntryMethod.FVG_50_LIMIT,
            EntryMethod.OB_50_LIMIT,
            EntryMethod.RECLAIM_ENTRY,
            EntryMethod.MARKET_ENTRY,
        )
        for candidate_id, assessment in belief.signal_assessments.items():
            if not assessment.eligible:
                continue
            candidate = belief.thesis_candidates.get(candidate_id)
            episode = belief.entry_episodes.get(candidate_id)
            if candidate is None or episode is None:
                diagnostics[candidate_id] = ("typed_setup_missing_at_intent",)
                continue
            try:
                intents[candidate_id] = build_trade_intent(
                    assessment,
                    asof=belief.asof,
                    setup_candidate=candidate,
                    entry_episode=episode,
                    account=account,
                    account_snapshot_id=f"account-snapshot:{account_id}",
                    risk_budget_id=f"risk-budget:{account_id}",
                    entry_method_preferences=methods,
                )
            except (TradeIntentError, TypeError, ValueError) as error:
                diagnostics[candidate_id] = (
                    f"trade_intent_not_built:{error}",
                )
        projected = replace(
            belief,
            trade_intents=intents,
            shadow_signal_rejections=diagnostics,
        )
        self._belief = projected
        return projected

    def update(
        self,
        observation: MarketObservation,
        *,
        position: PositionSnapshot | None = None,
        scene_graph: TemporalMarketSceneGraph | None = None,
        scene_delta: SceneGraphDelta | None = None,
        precomputed_global_context: GlobalMarketContext | None = None,
    ) -> MarketBelief:
        """Update one completed clock with an update-local graph-query memo.

        Runtime Engine supplies the raw, once-reduced global context.  Direct
        Brain callers retain the historical fallback that computes that raw
        base here before applying terminal routing and typed thesis building.
        """

        previous_belief = self._belief
        previous_hypothesis_manager = self.hypothesis_manager.fork()
        previous_path_competition_state = self._path_competition_state
        previous_path_scope_key = self._path_scope_key
        previous_path_seen_evidence_tokens = set(
            self._path_seen_evidence_tokens
        )
        previous_dol_probability_results = dict(
            self._dol_probability_results
        )
        previous_signal_real_completed_bar_anchors = dict(
            self._signal_real_completed_bar_anchors
        )
        token = _LSR_CONNECTION_MEMO.set({})
        try:
            return self._update_without_connection_memo(
                observation,
                position=position,
                scene_graph=scene_graph,
                scene_delta=scene_delta,
                precomputed_global_context=precomputed_global_context,
            )
        except Exception:
            # Path/DOL state is an append-only projection of a completed Brain
            # clock.  A later Focus, MarketBelief or shadow-signal failure must
            # not leave that projection ahead of the last published belief.
            self._belief = previous_belief
            self.hypothesis_manager = previous_hypothesis_manager
            self._path_competition_state = previous_path_competition_state
            self._path_scope_key = previous_path_scope_key
            self._path_seen_evidence_tokens = (
                previous_path_seen_evidence_tokens
            )
            self._dol_probability_results = (
                previous_dol_probability_results
            )
            self._signal_real_completed_bar_anchors = (
                previous_signal_real_completed_bar_anchors
            )
            raise
        finally:
            _LSR_CONNECTION_MEMO.reset(token)

    def _update_without_connection_memo(
        self,
        observation: MarketObservation,
        *,
        position: PositionSnapshot | None = None,
        scene_graph: TemporalMarketSceneGraph | None = None,
        scene_delta: SceneGraphDelta | None = None,
        precomputed_global_context: GlobalMarketContext | None = None,
    ) -> MarketBelief:
        hypotheses: dict[str, HypothesisBelief] = {}
        thesis_candidates: dict[str, HypothesisBelief] = {}
        retained_episode_candidates: dict[str, HypothesisBelief] = {}
        position_management_candidates: dict[str, HypothesisBelief] = {}
        evaluated_by_slot: dict[str, list[HypothesisBelief]] = {}
        previous_belief = self._belief
        market_lifecycle_deadline = _market_lifecycle_deadline(
            observation.asof
        )
        prior_hypotheses = (
            previous_belief.hypotheses
            if previous_belief is not None
            else {}
        )
        prior_thesis_candidates = (
            previous_belief.thesis_candidates
            if previous_belief is not None
            else {}
        )
        prior_retained_candidates = (
            previous_belief.retained_episode_candidates
            if previous_belief is not None
            else {}
        )
        prior_position_candidates = (
            previous_belief.position_management_candidates
            if previous_belief is not None
            else {}
        )
        candidate_epoch_changed = False
        global_context = None
        if precomputed_global_context is not None and (
            scene_graph is None or scene_delta is None
        ):
            raise ValueError(
                "precomputed global context requires its graph and delta"
            )
        if scene_graph is not None and scene_delta is not None:
            global_context = precomputed_global_context
            if global_context is None:
                global_context = update_global_market_context(
                    (
                        None
                        if previous_belief is None
                        else previous_belief.global_context
                    ),
                    observation,
                    scene_delta,
                    scene_graph,
                )
            elif (
                global_context.updated_at != observation.asof
                or global_context.scene_revision_id
                != scene_delta.revision_id
                or global_context.market_epoch_id
                != scene_graph._market_epoch_id
                or global_context.open_market_theses
            ):
                raise ValueError(
                    "precomputed global context is not the raw current base"
                )
            global_context = _route_terminal_delta_to_global_context(
                global_context,
                previous_belief,
                scene_delta,
                scene_graph,
            )
            global_context = replace(
                global_context,
                open_market_theses=build_open_market_theses(
                    (
                        ()
                        if previous_belief is None
                        or previous_belief.global_context is None
                        else previous_belief.global_context.open_market_theses
                    ),
                    observation,
                    scene_delta,
                    scene_graph,
                    global_context,
                ),
            )
            if (
                self._candidate_epoch_id is not None
                and self._candidate_epoch_id
                != global_context.market_epoch_id
            ):
                candidate_epoch_changed = True
                self._candidate_priors.clear()
                self._candidate_theses.clear()
                self._closed_contexts.clear()
                self._lsr_execution_winners.clear()
                self._lsr_execution_owner_snapshots.clear()
                self._lsr_context_seeds.clear()
                self._lsr_context_seed_ids.clear()
                self._lsr_retired_seed_keys.clear()
                self._lsr_context_bindings.clear()
                self._lsr_terminal_episode_ids.clear()
            self._candidate_epoch_id = global_context.market_epoch_id
            # Context tombstones are compact causal guards, not an archive.
            # They survive through the frozen thesis horizon (plus one
            # completed-bar rearm grace) and are otherwise discarded.
            self._closed_contexts = {
                identity: closure
                for identity, closure in self._closed_contexts.items()
                if observation.asof <= closure.retain_until
            }
        focus_state = None
        work_items: list[
            tuple[
                Playbook,
                Direction,
                PlaybookProtocol,
                OpenMarketThesis | None,
                str | None,
                HypothesisBelief | None,
                bool,
                bool,
                str | None,
                bool,
                _LSRCandidate | None,
            ]
        ] = []
        frozen_lsr_bindings: dict[str, _LSRContextBinding] = {}
        current_candidate_ids: set[str] = set()
        known_context_ids = frozenset(
            (
                *(
                    ()
                    if previous_belief is None
                    else previous_belief.context_theses
                ),
                *self._closed_contexts,
            )
        )
        previous_contexts = (
            {}
            if previous_belief is None
            else previous_belief.context_theses
        )
        for playbook in Playbook:
            protocol = self.registry.for_playbook(playbook)
            for direction in Direction:
                key = f"{playbook.value}:{direction.value}"
                matches = tuple(
                    thesis
                    for thesis in (
                        ()
                        if global_context is None
                        or playbook
                        is Playbook.FAILED_AUCTION_VALUE_RETURN
                        else global_context.open_market_theses
                    )
                    if (
                        thesis.lifecycle != "invalidated"
                    )
                    and (
                        thesis.direction is None
                        or thesis.direction is direction
                    )
                    and _open_thesis_supports_playbook(
                        playbook,
                        thesis,
                    )
                )
                if not matches:
                    if global_context is None:
                        work_items.append(
                            (
                                playbook,
                                direction,
                                protocol,
                                None,
                                None,
                                prior_hypotheses.get(key),
                                False,
                                False,
                                None,
                                False,
                                None,
                            )
                        )
                    continue
                assert global_context is not None
                for thesis in matches:
                    lsr_context_candidate: _LSRCandidate | None = None
                    location_items: tuple[
                        tuple[str | None, str | None, bool], ...
                    ] = ((None, None, False),)
                    if playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
                        seed_key = (
                            global_context.market_epoch_id,
                            thesis.root_id,
                            direction,
                        )
                        known_lsr_context_id = (
                            self._lsr_context_seed_ids.get(seed_key)
                        )
                        context_can_spawn_child = (
                            seed_key not in self._lsr_retired_seed_keys
                            and _lsr_context_can_spawn_child(
                                observation,
                                known_lsr_context_id,
                                previous_contexts,
                                self._closed_contexts,
                            )
                        )
                        lsr_root, _ = _select_pool_path(
                            observation,
                            direction,
                            None,
                            global_context=global_context,
                            scene_graph=scene_graph,
                            required_thesis=thesis,
                            frozen_candidate=(
                                self._lsr_context_seeds.get(seed_key)
                            ),
                        )
                        lsr_context_candidate = lsr_root
                        if (
                            lsr_root is not None
                            and _path_step(
                                lsr_root.path,
                                {"reacceptance_held"},
                            )
                            is not None
                            and _path_step(
                                lsr_root.path,
                                {"opposite_displacement"},
                            )
                            is not None
                        ):
                            self._lsr_context_seeds[seed_key] = lsr_root
                        lsr_locations = (
                            _eligible_lsr_entry_locations(
                                observation,
                                direction,
                                pool_path=(
                                    None
                                    if lsr_root is None
                                    else lsr_root.path
                                ),
                                manipulation=(
                                    None
                                    if lsr_root is None
                                    else lsr_root.manipulation
                                ),
                                scene_graph=scene_graph,
                            )
                            if context_can_spawn_child
                            else ()
                        )
                        location_items = (
                            tuple(
                                (
                                    location.location_id,
                                    location.source_displacement_id,
                                    False,
                                )
                                for location in lsr_locations
                            )
                            or ((None, None, True),)
                        )
                    for (
                        required_location_id,
                        required_displacement_id,
                        context_only,
                    ) in location_items:
                        location = (
                            None
                            if required_location_id is None
                            else next(
                                item
                                for item in observation.entry_locations
                                if item.location_id == required_location_id
                            )
                        )
                        candidate_id = (
                            _lsr_zone_candidate_id(
                                global_context.market_epoch_id,
                                thesis.root_id,
                                required_displacement_id,
                                location.source_zone_id,
                                direction,
                            )
                            if (
                                playbook
                                is Playbook.LIQUIDITY_SWEEP_REVERSAL
                                and required_location_id is not None
                                and required_displacement_id is not None
                                and location is not None
                            )
                            else _thesis_candidate_id(
                                global_context.market_epoch_id,
                                thesis.root_id,
                                playbook,
                                direction,
                            )
                        )
                        known_lsr_context_id = (
                            self._lsr_context_seed_ids.get(
                                (
                                    global_context.market_epoch_id,
                                    thesis.root_id,
                                    direction,
                                )
                            )
                            if (
                                playbook
                                is Playbook.LIQUIDITY_SWEEP_REVERSAL
                                and required_location_id is not None
                            )
                            else None
                        )
                        if (
                            known_lsr_context_id is not None
                            and candidate_id
                            in self._lsr_terminal_episode_ids.get(
                                known_lsr_context_id,
                                set(),
                            )
                        ):
                            continue
                        candidate_prior = self._candidate_priors.get(
                            candidate_id,
                            prior_thesis_candidates.get(
                                candidate_id,
                                prior_retained_candidates.get(candidate_id),
                            ),
                        )
                        current_candidate_ids.add(candidate_id)
                        self._candidate_theses[candidate_id] = thesis
                        work_items.append(
                            (
                                playbook,
                                direction,
                                protocol,
                                thesis,
                                candidate_id,
                                candidate_prior,
                                False,
                                False,
                                required_location_id,
                                context_only,
                                lsr_context_candidate,
                            )
                        )

        # Group5 retains only a bounded path archive.  Once a completed pool
        # path is evicted, an otherwise-live LSR Context must still be able to
        # discover a later zone from its exact frozen manipulation and reverse
        # displacement.  This branch activates only when the canonical root is
        # absent from the current analytical thesis view, and carries forward
        # the prior exact-root binding rather than inferring fresh authority.
        if (
            global_context is not None
            and previous_belief is not None
            and not candidate_epoch_changed
        ):
            live_root_keys = {
                (
                    work_thesis.root_id,
                    work_direction,
                )
                for (
                    work_playbook,
                    work_direction,
                    _work_protocol,
                    work_thesis,
                    _work_candidate_id,
                    _work_prior,
                    _work_position_only,
                    _work_retained_only,
                    _work_location_id,
                    _work_context_only,
                    _work_lsr_context_candidate,
                ) in work_items
                if (
                    work_playbook
                    is Playbook.LIQUIDITY_SWEEP_REVERSAL
                    and work_thesis is not None
                )
            }
            for seed_key, frozen_seed in sorted(
                self._lsr_context_seeds.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2].value,
                ),
            ):
                epoch_id, root_id, seed_direction = seed_key
                if (
                    epoch_id != global_context.market_epoch_id
                    or (root_id, seed_direction) in live_root_keys
                ):
                    continue
                frozen_context_id = self._lsr_context_seed_ids.get(
                    seed_key
                )
                previous_context = (
                    None
                    if frozen_context_id is None
                    else previous_contexts.get(frozen_context_id)
                )
                if (
                    frozen_context_id is None
                    or previous_context is None
                    or not _lsr_context_can_spawn_child(
                        observation,
                        frozen_context_id,
                        previous_contexts,
                        self._closed_contexts,
                    )
                ):
                    continue
                binding = self._lsr_context_bindings.get(seed_key)
                if binding is None:
                    continue
                frozen_thesis = binding.thesis
                if (
                    frozen_thesis.market_epoch_id != epoch_id
                    or frozen_thesis.root_id != root_id
                ):
                    continue
                selected_seed, _ = _select_pool_path(
                    observation,
                    seed_direction,
                    None,
                    global_context=global_context,
                    scene_graph=scene_graph,
                    required_thesis=frozen_thesis,
                    frozen_candidate=frozen_seed,
                )
                if selected_seed is None:
                    continue
                late_locations = _eligible_lsr_entry_locations(
                    observation,
                    seed_direction,
                    pool_path=selected_seed.path,
                    manipulation=selected_seed.manipulation,
                    scene_graph=scene_graph,
                )
                for location in late_locations:
                    candidate_id = _lsr_zone_candidate_id(
                        epoch_id,
                        root_id,
                        location.source_displacement_id,
                        location.source_zone_id,
                        seed_direction,
                    )
                    if candidate_id in current_candidate_ids:
                        continue
                    if candidate_id in self._lsr_terminal_episode_ids.get(
                        frozen_context_id,
                        set(),
                    ):
                        continue
                    candidate_prior = self._candidate_priors.get(
                        candidate_id,
                    )
                    current_candidate_ids.add(candidate_id)
                    self._candidate_theses[candidate_id] = frozen_thesis
                    frozen_lsr_bindings[candidate_id] = binding
                    work_items.append(
                        (
                            Playbook.LIQUIDITY_SWEEP_REVERSAL,
                            seed_direction,
                            self.registry.for_playbook(
                                Playbook.LIQUIDITY_SWEEP_REVERSAL
                            ),
                            frozen_thesis,
                            candidate_id,
                            candidate_prior,
                            False,
                            False,
                            location.location_id,
                            False,
                            selected_seed,
                        )
                    )

        # The analytical root set is a live discovery view.  A root closing
        # after a fill must not erase the unique frozen candidate that owns
        # the position.  Retain exactly one prior root projection for
        # management only; zero or ambiguous matches remain fail-closed.
        if position is not None and global_context is not None:
            previous_root_candidates = {
                **prior_thesis_candidates,
                **prior_retained_candidates,
                **prior_position_candidates,
            }
            position_matches = tuple(
                (identity, candidate)
                for identity, candidate in previous_root_candidates.items()
                if _position_matches_root_candidate(position, candidate)
            )
            if len(position_matches) == 1:
                retained_id, retained_prior = position_matches[0]
                retained_thesis = self._candidate_theses.get(retained_id)
                if (
                    retained_id not in current_candidate_ids
                    and retained_thesis is not None
                    and retained_thesis.root_id
                    == retained_prior.required_root_id
                    and retained_thesis.thesis_id
                    == retained_prior.market_thesis_id
                ):
                    work_items.append(
                        (
                            retained_prior.playbook,
                            retained_prior.direction,
                            self.registry.for_playbook(
                                retained_prior.playbook
                            ),
                            retained_thesis,
                            retained_id,
                            retained_prior,
                            True,
                            False,
                            retained_prior.entry_location_id,
                            False,
                            (
                                self._lsr_context_seeds.get(
                                    (
                                        global_context.market_epoch_id,
                                        retained_prior.required_root_id or "",
                                        retained_prior.direction,
                                    )
                                )
                                if retained_prior.playbook
                                is Playbook.LIQUIDITY_SWEEP_REVERSAL
                                else None
                            ),
                        )
                    )

        # A bounded discovery root may disappear while its frozen causal
        # episode remains unresolved.  Re-evaluate only the exact frozen
        # setup/location/path: a uniquely live match keeps the same
        # root-specific action identity, while a missing or ambiguous path is
        # retained dormant until explicit terminal evidence closes it.
        if global_context is not None and not candidate_epoch_changed:
            position_retained_ids = {
                identity
                for identity, candidate in {
                    **prior_thesis_candidates,
                    **prior_retained_candidates,
                    **prior_position_candidates,
                }.items()
                if position is not None
                and _position_matches_root_candidate(position, candidate)
            }
            prior_lifecycle_candidates = {
                **prior_thesis_candidates,
                **prior_retained_candidates,
            }
            current_theses_by_root = {
                thesis.root_id: thesis
                for thesis in global_context.open_market_theses
            }
            current_lsr_episode_roots = {
                thesis.root_id
                for identity in current_candidate_ids
                for thesis in (self._candidate_theses.get(identity),)
                if thesis is not None
                and any(
                    work_candidate_id == identity
                    and work_playbook
                    is Playbook.LIQUIDITY_SWEEP_REVERSAL
                    and work_location_id is not None
                    for (
                        work_playbook,
                        _work_direction,
                        _work_protocol,
                        _work_thesis,
                        work_candidate_id,
                        _work_prior,
                        _work_position_only,
                        _work_retained_only,
                        work_location_id,
                        _work_context_only,
                        _work_lsr_context_candidate,
                    ) in work_items
                )
            }
            for retained_id, retained_prior in sorted(
                prior_lifecycle_candidates.items()
            ):
                if (
                    retained_id in current_candidate_ids
                    or retained_id in position_retained_ids
                    or retained_prior.phase in _TERMINAL_PHASES
                    or not retained_prior.market_thesis_action_bound
                    or retained_prior.market_thesis_match_status
                    != "exact_root_bound"
                    or (
                        retained_prior.playbook
                        is Playbook.LIQUIDITY_SWEEP_REVERSAL
                        and retained_prior.entry_location_id is None
                        and retained_prior.required_root_id
                        in current_lsr_episode_roots
                    )
                ):
                    continue
                retained_thesis = current_theses_by_root.get(
                    retained_prior.required_root_id or "",
                    self._candidate_theses.get(retained_id),
                )
                if (
                    retained_thesis is None
                    or retained_thesis.market_epoch_id
                    != global_context.market_epoch_id
                    or retained_thesis.root_id
                    != retained_prior.required_root_id
                ):
                    continue
                self._candidate_theses[retained_id] = retained_thesis
                work_items.append(
                    (
                        retained_prior.playbook,
                        retained_prior.direction,
                        self.registry.for_playbook(retained_prior.playbook),
                        retained_thesis,
                        retained_id,
                        retained_prior,
                        False,
                        True,
                        retained_prior.entry_location_id,
                        False,
                        (
                            self._lsr_context_seeds.get(
                                (
                                    global_context.market_epoch_id,
                                    retained_prior.required_root_id or "",
                                    retained_prior.direction,
                                )
                            )
                            if retained_prior.playbook
                            is Playbook.LIQUIDITY_SWEEP_REVERSAL
                            else None
                        ),
                    )
                )

        for (
            playbook,
            direction,
            protocol,
            required_thesis,
            candidate_id,
            prior,
            position_management_only,
            retained_episode_only,
            required_entry_location_id,
            lsr_context_only,
            frozen_lsr_context_candidate,
        ) in work_items:
                key = f"{playbook.value}:{direction.value}"
                context_id: str | None = None
                episode_id: str | None = None
                context_thesis_id: str | None = None
                thesis_deadline: pd.Timestamp | None = None
                episode_deadline: pd.Timestamp | None = None
                initiating_event_id: str | None = None
                frozen_invalidation: StructuralLevel | None = None
                evidence_revision_id: str | None = None
                terminal_at: pd.Timestamp | None = None
                terminal_reason: str | None = None
                terminal_source_ids: tuple[str, ...] = ()
                raw_quality_dimensions: Mapping[str, float] = {}
                retained_episode_progressing = False
                evaluation = _typed_evaluate(
                    playbook,
                    observation,
                    direction,
                    protocol,
                    prior,
                    self.config,
                    global_context=global_context,
                    scene_graph=scene_graph,
                    required_thesis=required_thesis,
                    required_entry_location_id=(
                        required_entry_location_id
                        if playbook
                        is Playbook.LIQUIDITY_SWEEP_REVERSAL
                        else None
                    ),
                    lsr_context_only=bool(
                        playbook
                        is Playbook.LIQUIDITY_SWEEP_REVERSAL
                        and lsr_context_only
                    ),
                    frozen_lsr_context_candidate=(
                        frozen_lsr_context_candidate
                        if playbook
                        is Playbook.LIQUIDITY_SWEEP_REVERSAL
                        else None
                    ),
                )
                evaluation = _bind_open_market_theses(
                    playbook,
                    direction,
                    prior,
                    evaluation,
                    global_context,
                    required_thesis=required_thesis,
                )
                frozen_lsr_binding = (
                    None
                    if candidate_id is None
                    else frozen_lsr_bindings.get(candidate_id)
                )
                if frozen_lsr_binding is not None:
                    if required_thesis is None:
                        raise ValueError(
                            "frozen LSR Context discovery requires its exact "
                            "market thesis"
                        )
                    evaluation = _retain_lsr_context_market_thesis_binding(
                        frozen_lsr_binding,
                        evaluation,
                        required_thesis,
                    )
                if position_management_only:
                    if prior is None or required_thesis is None:
                        raise ValueError(
                            "position-management work item requires its "
                            "frozen prior and thesis"
                        )
                    evaluation = _retain_position_market_thesis_binding(
                        prior,
                        evaluation,
                        required_thesis,
                    )
                elif retained_episode_only:
                    if prior is None or required_thesis is None:
                        raise ValueError(
                            "retained episode work item requires its frozen "
                            "prior and thesis"
                        )
                    evaluation = _retain_position_market_thesis_binding(
                        prior,
                        evaluation,
                        required_thesis,
                )
                evaluation = _require_connected_graph_sequence(
                    playbook,
                    protocol,
                    evaluation,
                    scene_graph,
                    prior,
                )
                routing_context_thesis = None
                routing_context_thesis_id: str | None = None
                if global_context is not None:
                    routing_context_thesis_id = _stable_context_thesis_id(
                        global_context.market_epoch_id,
                        playbook,
                        direction,
                        evaluation,
                        prior,
                        known_context_ids=known_context_ids,
                    )
                    if (
                        routing_context_thesis_id is not None
                        and previous_belief is not None
                    ):
                        routing_context_thesis = (
                            previous_belief.context_theses.get(
                                routing_context_thesis_id
                            )
                        )
                evaluation = _route_global_context(
                    playbook,
                    direction,
                    prior,
                    evaluation,
                    global_context,
                    tick_size=self.config.tick_size,
                    observation=observation,
                    context_thesis=routing_context_thesis,
                )
                if retained_episode_only:
                    assert prior is not None
                    assert global_context is not None
                    retained_episode_progressing = (
                        _retained_episode_has_live_frozen_path(
                            prior,
                            evaluation,
                            observation,
                        )
                    )
                    if not retained_episode_progressing:
                        evaluation = _retain_unresolved_episode_evaluation(
                            prior,
                            evaluation,
                            global_context,
                            observation,
                        )
                _validate_evidence_contract(protocol, evaluation.evidence)
                sequence = _sequence_state(
                    protocol,
                    self.registry,
                    evaluation,
                    prior,
                )
                if set(evaluation.hard_gate_results) != set(
                    protocol.hard_gates
                ):
                    raise ValueError(
                        f"{playbook.value} typed evaluator hard gates "
                        "disagree with the registry"
                    )
                current_hard_gates = {
                    gate_id: bool(
                        evaluation.hard_gate_results[gate_id]
                    )
                    for gate_id in protocol.hard_gates
                }
                if (
                    prior is not None
                    and prior.phase in _TERMINAL_PHASES
                    and not _terminal_has_new_setup(prior, sequence)
                ):
                    # The terminal belief is the immutable closure of the
                    # last episode.  It remains visible until a different,
                    # causally newer candidate supersedes it.
                    if candidate_id is not None:
                        destination = (
                            position_management_candidates
                            if position_management_only
                            else retained_episode_candidates
                            if retained_episode_only
                            else thesis_candidates
                        )
                        retained_prior = replace(
                            prior,
                            record_kind=(
                                "position_management"
                                if position_management_only
                                else "retained_episode"
                                if retained_episode_only
                                else "root_candidate"
                            ),
                        )
                        destination[candidate_id] = retained_prior
                        self._candidate_priors[candidate_id] = retained_prior
                    if not retained_episode_only:
                        evaluated_by_slot.setdefault(key, []).append(
                            retained_prior
                            if candidate_id is not None
                            else prior
                        )
                    continue
                setup_id = sequence.setup_id
                context_id = evaluation.context_identity
                if (
                    context_id is None
                    and prior is not None
                    and prior.setup_context_id == setup_id
                ):
                    context_id = prior.context_id
                episode_id = (
                    evaluation.episode_identity
                    if evaluation.episode_identity == setup_id
                    and (
                        candidate_id is None
                        or evaluation.market_thesis_action_bound
                        or position_management_only
                        or retained_episode_only
                    )
                    else None
                )
                entry_location_id = evaluation.entry_location_id
                if (
                    episode_id is None
                    and prior is not None
                    and prior.setup_context_id == setup_id
                ):
                    episode_id = prior.episode_id
                initiating_event_id = (
                    evaluation.initiating_event_id
                    or (
                        prior.initiating_event_id
                        if (
                            prior is not None
                            and prior.setup_context_id == setup_id
                        )
                        else None
                    )
                )
                position_key_matches = bool(
                    position is not None
                    and position.playbook is playbook
                    and position.direction is direction
                )
                position_matches_current = bool(
                    position_key_matches
                    and position is not None
                    and position.setup_id is not None
                    and position.setup_id == setup_id
                    and position.entry_location_id
                    == evaluation.entry_location_id
                    and position.entry_path_id
                    == evaluation.entry_path_id
                )
                position_matches_prior = bool(
                    position_key_matches
                    and position is not None
                    and position.setup_id is not None
                    and prior is not None
                    and prior.setup_context_id == position.setup_id
                    and prior.entry_location_id
                    == position.entry_location_id
                    and prior.plan is not None
                    and prior.entry_path_id == position.entry_path_id
                )
                position_matches = bool(
                    position_matches_current
                    or position_matches_prior
                )
                if (
                    position_matches_prior
                    and not position_matches_current
                    and prior is not None
                    and prior.sequence is not None
                ):
                    # An open position owns the exact typed episode that
                    # created it.  If current source evidence disappears
                    # or a newer same-direction candidate appears, retain
                    # the entered episode identity and let current evidence
                    # move it to weakening instead of attaching the old
                    # position to the new candidate.
                    sequence = prior.sequence
                    setup_id = prior.setup_context_id
                    context_id = prior.context_id
                    episode_id = prior.episode_id
                    entry_location_id = prior.entry_location_id
                    initiating_event_id = prior.initiating_event_id
                same_episode = bool(
                    prior is not None
                    and episode_id is not None
                    and prior.episode_id == episode_id
                    and prior.setup_context_id == setup_id
                )
                if (
                    position_matches
                    and position is not None
                    and same_episode
                    and prior is not None
                    and prior.invalidation is not None
                    and position.original_invalidation
                    != prior.invalidation
                ):
                    raise ValueError(
                        "position invalidation differs from its frozen "
                        "belief episode"
                    )
                frozen_invalidation = (
                    prior.invalidation
                    if same_episode
                    and prior is not None
                    and prior.invalidation is not None
                    else position.original_invalidation
                    if position_matches and position is not None
                    else evaluation.invalidation
                )
                if episode_id is not None:
                    if (
                        same_episode
                        and prior is not None
                        and prior.episode_deadline is not None
                    ):
                        episode_deadline = prior.episode_deadline
                    else:
                        episode_deadline = market_lifecycle_deadline
                if context_id is not None:
                    if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
                        # DFP Context lifetime is owned by H4/H1 structure,
                        # its external draw and the market epoch.  The short
                        # entry Episode still owns the session/plan deadline.
                        thesis_deadline = None
                    elif playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
                        # The reversal Context outlives any one zone-return
                        # window.  Its registered session thesis horizon is
                        # shared by all child EntryEpisodes; a plan-level
                        # deadline may shorten only the local child below.
                        thesis_deadline = (
                            prior.thesis_deadline
                            if (
                                prior is not None
                                and prior.context_id == context_id
                                and prior.thesis_deadline is not None
                            )
                            else market_lifecycle_deadline
                        )
                    elif episode_deadline is not None:
                        thesis_deadline = episode_deadline
                evidence_revision_id = _typed_evidence_revision(
                    (
                        routing_context_thesis_id
                        if (
                            playbook
                            is Playbook.LIQUIDITY_SWEEP_REVERSAL
                            and routing_context_thesis_id is not None
                        )
                        else candidate_id or key
                    ),
                    protocol,
                    evaluation,
                )
                context_thesis_id = (
                    _stable_context_thesis_id(
                        global_context.market_epoch_id,
                        playbook,
                        direction,
                        evaluation,
                        prior,
                        known_context_ids=known_context_ids,
                    )
                    if global_context is not None
                    else context_id
                )
                if context_thesis_id is None:
                    # Entry Episodes are children, never substitutes for a
                    # missing causal parent.  Preserve the evaluator-native
                    # Context for diagnostics while withholding the episode
                    # identity and its lifetime from action state.
                    episode_id = None
                    episode_deadline = None
                raw_thesis_strength = _typed_thesis_strength(
                    prior,
                    context_id,
                    evaluation.thesis_target,
                    evidence_revision_id,
                )
                plan = (
                    evaluation.plan
                    if (
                        evaluation.plan is not None
                        and evaluation.plan.setup_id == setup_id
                    )
                    else None
                )
                if (
                    position_matches
                    and prior is not None
                    and prior.plan is not None
                ):
                    # Preserve the exact entry thesis while the matching
                    # position is open.  Current evidence may weaken the
                    # episode, but cannot erase its typed identity.
                    plan = prior.plan
                if (
                    plan is not None
                    and frozen_invalidation is not None
                    and plan.invalidation != frozen_invalidation
                ):
                    plan = None
                if (
                    plan is not None
                    and episode_id is not None
                    and episode_deadline is not None
                    and (
                        not same_episode
                        or prior is None
                        or prior.plan is None
                    )
                ):
                    episode_deadline = min(
                        episode_deadline,
                        plan.deadline,
                    )
                    if (
                        playbook
                        not in {
                            Playbook.DISPLACEMENT_FIRST_PULLBACK,
                            Playbook.LIQUIDITY_SWEEP_REVERSAL,
                        }
                    ):
                        thesis_deadline = episode_deadline
                if (
                    plan is not None
                    and not position_matches
                    and episode_deadline is not None
                    and observation.asof >= episode_deadline
                ):
                    # The market episode may remain observable after its
                    # frozen entry deadline, but it can no longer publish a
                    # newly constructed executable plan.  Failing closed here
                    # also prevents a current draw selection from being
                    # back-dated by the deadline clamp below.
                    plan = None
                if (
                    plan is not None
                    and same_episode
                    and prior is not None
                    and prior.plan is not None
                    and (
                        prior.plan.setup_id,
                        prior.plan.entry_location_id,
                        prior.plan.entry_path_id,
                    )
                    == (
                        plan.setup_id,
                        plan.entry_location_id,
                        plan.entry_path_id,
                    )
                ):
                    frozen_plan_deadline = min(
                        plan.deadline,
                        prior.plan.deadline,
                        episode_deadline or plan.deadline,
                    )
                    if (
                        not position_matches
                        and observation.asof >= frozen_plan_deadline
                    ):
                        plan = None
                    else:
                        plan = replace(
                            plan,
                            deadline=frozen_plan_deadline,
                        )
                elif (
                    plan is not None
                    and episode_deadline is not None
                    and plan.deadline > episode_deadline
                ):
                    plan = replace(plan, deadline=episode_deadline)
                if (
                    position_matches
                    and plan is not None
                    and _frozen_target_crosses_authority_barrier(
                        plan,
                        observation,
                        global_context,
                        self.config.tick_size,
                    )
                ):
                    # The entered thesis keeps its original target.  A newly
                    # visible authority barrier changes delivery prospects,
                    # not the frozen plan geometry.
                    evaluation = replace(
                        evaluation,
                        delivery_quality=0.0,
                    )
                parked = "parked" in protocol.status
                if parked:
                    # Parked playbooks remain fully observable for causal
                    # diagnostics but cannot expose an executable plan.
                    plan = None
                plan_feasibility = _plan_feasibility(
                    plan,
                    evaluation,
                    self.config,
                )
                plan_delivery_valid = plan_feasibility.valid
                phase = _typed_phase(
                    playbook,
                    direction,
                    prior,
                    raw_thesis_strength,
                    evaluation,
                    sequence,
                    plan,
                    frozen_invalidation,
                    episode_deadline,
                    observation,
                    position if position_matches else None,
                    self.config,
                    parked=parked,
                )
                if (
                    retained_episode_only
                    and not retained_episode_progressing
                    and phase is PlaybookPhase.EXECUTABLE
                ):
                    # An absent current root remains observable for lifecycle
                    # resolution only.  It must be independently rediscovered
                    # before its episode can re-enter the action map.
                    phase = PlaybookPhase.WEAKENING
                evaluation = _stage_uncertainty(
                    evaluation,
                    playbook,
                    sequence,
                    observation,
                )
                terminal_prior_sequence_restored = False
                if (
                    phase in _TERMINAL_PHASES
                    and prior is not None
                    and prior.phase not in _TERMINAL_PHASES
                    and prior.sequence is not None
                    and (
                        sequence.setup_id is None
                        or sequence.setup_id
                        != prior.sequence.setup_id
                    )
                ):
                    # Closing evidence may remove the frozen source from
                    # the current observation.  Preserve the episode that
                    # is being closed; the current evidence and terminal
                    # reason still describe why it closed.
                    terminal_prior_sequence_restored = True
                    sequence = prior.sequence
                    setup_id = sequence.setup_id
                    context_id = prior.context_id
                    episode_id = prior.episode_id
                    thesis_deadline = prior.thesis_deadline
                    episode_deadline = prior.episode_deadline
                    entry_location_id = prior.entry_location_id
                    initiating_event_id = prior.initiating_event_id
                    frozen_invalidation = prior.invalidation
                if (
                    terminal_prior_sequence_restored
                    and context_thesis_id is not None
                    and context_thesis_id not in known_context_ids
                ):
                    # A newly derived parent belongs to the current complete
                    # authority sequence.  Restoring a different prior local
                    # episode must not attach that older sequence to the new
                    # parent or let the Context builder infer empty authority.
                    context_thesis_id = None
                if context_thesis_id is None:
                    # Terminal evidence can restore a prior local sequence so
                    # the root explains its closure.  It still cannot restore
                    # an EntryEpisode after parent identity validation failed.
                    episode_id = None
                    episode_deadline = None
                raw_sequence_progress = (
                    sequence.completed_steps
                    / max(1, len(sequence.steps))
                )
                raw_quality_dimensions = {
                    "thesis_strength": clamp(raw_thesis_strength),
                    "sequence_progress": clamp(raw_sequence_progress),
                    "location_quality": clamp(
                        evaluation.location_quality
                    ),
                    "entry_readiness": clamp(
                        evaluation.entry_readiness
                    ),
                    "delivery_quality": clamp(
                        evaluation.delivery_quality
                    ),
                    "uncertainty": clamp(
                        evaluation.typed_uncertainty
                    ),
                }
                if parked:
                    calibrated_dimensions = dict(
                        raw_quality_dimensions
                    )
                else:
                    calibrated_dimensions = {
                        name: (
                            value
                            if name in {"sequence_progress", "uncertainty"}
                            else self.calibrator.apply(
                                playbook,
                                name,
                                value,
                            )
                        )
                        for name, value in raw_quality_dimensions.items()
                    }
                thesis_strength = calibrated_dimensions[
                    "thesis_strength"
                ]
                sequence_progress = calibrated_dimensions[
                    "sequence_progress"
                ]
                uncertainty = calibrated_dimensions["uncertainty"]
                probability = thesis_strength
                raw_probability = raw_thesis_strength
                (
                    terminal_at,
                    terminal_reason,
                    terminal_source_ids,
                ) = _typed_terminal_closure(
                    phase,
                    direction,
                    evaluation,
                    sequence,
                    plan,
                    frozen_invalidation,
                    episode_deadline,
                    observation,
                    (
                        position if position_matches else None
                    ),
                )
                if phase in _TERMINAL_PHASES:
                    terminal_source_ids = _identity_tuple(
                        *terminal_source_ids,
                        episode_id,
                        context_id,
                        initiating_event_id,
                    )
                phase_continues = bool(
                    prior is not None and prior.phase is phase
                )
                if (
                    prior is not None
                    and prior.phase in _TERMINAL_PHASES
                ):
                    # A constructed typed belief after a terminal prior has
                    # already passed the new-episode test above.
                    phase_continues = False
                phase_started = (
                    prior.phase_started_at
                    if phase_continues and prior is not None
                    else observation.asof
                )
                supporting = tuple(item for item in evaluation.evidence if item.supports and item.value > 0)
                contradicting = tuple(item for item in evaluation.evidence if not item.supports and item.value > 0)
                belief_invalidation = frozen_invalidation
                if (
                    belief_invalidation is None
                    and position_matches
                    and position is not None
                ):
                    belief_invalidation = position.original_invalidation
                if (
                    belief_invalidation is None
                    and phase in _TERMINAL_PHASES
                    and prior is not None
                    and prior.setup_context_id == sequence.setup_id
                ):
                    belief_invalidation = prior.invalidation
                effective_entry_path_id = (
                    plan.entry_path_id
                    if plan is not None
                    else evaluation.entry_path_id
                    if evaluation.entry_path_id is not None
                    else prior.entry_path_id
                    if (
                        prior is not None
                        and prior.setup_context_id == sequence.setup_id
                        and prior.entry_location_id == entry_location_id
                    )
                    else None
                )
                belief_thesis_draw = (
                    None
                    if playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
                    else evaluation.thesis_draw
                    if evaluation.thesis_draw is not None
                    else prior.thesis_draw
                    if (
                        prior is not None
                        and prior.context_id == context_id
                    )
                    else None
                )
                belief_liquidity_route = (
                    plan.liquidity_route
                    if plan is not None
                    else evaluation.liquidity_route
                    if evaluation.liquidity_route is not None
                    else prior.liquidity_route
                    if (
                        prior is not None
                        and prior.context_id == context_id
                    )
                    else None
                )
                belief_selected_trigger = _retain_matching_prior_trigger(
                    evaluation.selected_trigger,
                    prior=prior,
                    setup_id=sequence.setup_id,
                    entry_location_id=entry_location_id,
                    entry_path_id=effective_entry_path_id,
                    direction=direction,
                    trigger_must_follow=(
                        next(
                            (
                                step.observed_at
                                for step in sequence.steps
                                if (
                                    step.step_id
                                    == "reversal_first_pullback"
                                    and step.satisfied
                                )
                            ),
                            None,
                        )
                        if playbook
                        is Playbook.LIQUIDITY_SWEEP_REVERSAL
                        else None
                    ),
                    require_strictly_after=(
                        playbook
                        is Playbook.LIQUIDITY_SWEEP_REVERSAL
                    ),
                )
                evaluated_belief = HypothesisBelief(
                    playbook=playbook,
                    direction=direction,
                    probability=probability,
                    phase=phase,
                    phase_started_at=phase_started,
                    supporting=supporting,
                    contradicting=contradicting,
                    invalidation=belief_invalidation,
                    deliverable_targets=(
                        plan.targets
                        if plan is not None
                        else ()
                        if evaluation.selected_draw is None
                        else (evaluation.selected_draw,)
                    ),
                    remaining_path_R=None if plan is None else plan.remaining_path_R,
                    uncertainty=uncertainty,
                    plan=plan,
                    sequence=sequence,
                    raw_probability=raw_probability,
                    calibration_version=self.calibrator.version,
                    thesis_strength=thesis_strength,
                    sequence_progress=sequence_progress,
                    location_quality=calibrated_dimensions[
                        "location_quality"
                    ],
                    entry_readiness=calibrated_dimensions[
                        "entry_readiness"
                    ],
                    delivery_quality=calibrated_dimensions[
                        "delivery_quality"
                    ],
                    evidence_group_scores=evaluation.evidence_group_scores,
                    hard_gate_results=current_hard_gates,
                    setup_context_id=sequence.setup_id,
                    entry_location_id=(
                        entry_location_id
                        if sequence.setup_id is not None
                        else None
                    ),
                    entry_path_id=(
                        effective_entry_path_id
                        if sequence.setup_id is not None
                        and entry_location_id is not None
                        else None
                    ),
                    context_id=context_id,
                    episode_id=episode_id,
                    context_thesis_id=context_thesis_id,
                    parent_context_thesis_id=(
                        context_thesis_id
                        if episode_id is not None
                        else None
                    ),
                    thesis_deadline=thesis_deadline,
                    episode_deadline=episode_deadline,
                    initiating_event_id=initiating_event_id,
                    evidence_revision_id=evidence_revision_id,
                    terminal_at=terminal_at,
                    terminal_reason=terminal_reason,
                    terminal_source_ids=terminal_source_ids,
                    thesis_draw=belief_thesis_draw,
                    draw_selection=(
                        plan.draw_selection
                        if plan is not None
                        else evaluation.draw_selection
                    ),
                    liquidity_route=belief_liquidity_route,
                    raw_quality_dimensions=raw_quality_dimensions,
                    context_metadata=_evaluation_context_metadata(
                        evaluation,
                        plan_feasibility=plan_feasibility,
                    ),
                    competing_episode_ids=(
                        evaluation.competing_episode_ids
                    ),
                    market_thesis_ids=evaluation.market_thesis_ids,
                    market_thesis_id=evaluation.market_thesis_id,
                    bound_market_thesis_id=(
                        evaluation.bound_market_thesis_id
                    ),
                    market_thesis_root_id=(
                        evaluation.market_thesis_root_id
                    ),
                    market_thesis_mechanism=(
                        evaluation.market_thesis_mechanism
                    ),
                    market_thesis_authority_relation=(
                        evaluation.market_thesis_authority_relation
                    ),
                    playbook_match_strength=(
                        evaluation.playbook_match_strength
                    ),
                    market_thesis_binding_required=(
                        evaluation.market_thesis_binding_required
                    ),
                    market_thesis_action_bound=(
                        evaluation.market_thesis_action_bound
                    ),
                    market_thesis_match_status=(
                        evaluation.market_thesis_match_status
                    ),
                    selected_trigger=belief_selected_trigger,
                    candidate_id=candidate_id,
                    required_root_id=(
                        None
                        if required_thesis is None
                        else required_thesis.root_id
                    ),
                    record_kind=(
                        "summary"
                        if candidate_id is None
                        else "position_management"
                        if position_management_only
                        else "retained_episode"
                        if (
                            retained_episode_only
                            and not retained_episode_progressing
                        )
                        else "root_candidate"
                    ),
                    plan_feasibility=plan_feasibility,
                )
                if (
                    playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
                    and global_context is not None
                    and required_thesis is not None
                    and context_thesis_id is not None
                ):
                    seed_key = (
                        global_context.market_epoch_id,
                        required_thesis.root_id,
                        direction,
                    )
                    if seed_key in self._lsr_context_seeds:
                        self._lsr_context_seed_ids[seed_key] = (
                            context_thesis_id
                        )
                        if (
                            evaluated_belief.market_thesis_action_bound
                            and evaluated_belief.market_thesis_match_status
                            == "exact_root_bound"
                            and evaluated_belief.market_thesis_id
                            == required_thesis.thesis_id
                        ):
                            self._lsr_context_bindings[seed_key] = (
                                _LSRContextBinding(
                                    thesis=required_thesis,
                                    playbook_match_strength=(
                                        evaluated_belief.playbook_match_strength
                                    ),
                                )
                            )
                if candidate_id is not None:
                    destination = (
                        position_management_candidates
                        if position_management_only
                        else retained_episode_candidates
                        if (
                            retained_episode_only
                            and not retained_episode_progressing
                        )
                        else thesis_candidates
                    )
                    destination[candidate_id] = evaluated_belief
                    self._candidate_priors[candidate_id] = evaluated_belief
                if (
                    not retained_episode_only
                    or retained_episode_progressing
                ):
                    evaluated_by_slot.setdefault(key, []).append(
                        evaluated_belief
                    )
        context_diagnostic_candidates = dict(thesis_candidates)
        thesis_candidates = _fail_closed_shared_episode_ownership(
            observation,
            thesis_candidates,
            {
                **retained_episode_candidates,
                **position_management_candidates,
            },
            self._candidate_theses,
            previous_contexts,
            self._closed_contexts,
        )
        conflicted_candidate_ids = (
            set(context_diagnostic_candidates) - set(thesis_candidates)
        )
        for identity in conflicted_candidate_ids:
            self._candidate_priors.pop(identity, None)
        lifecycle_candidates = {
            **thesis_candidates,
            **retained_episode_candidates,
            **position_management_candidates,
        }
        context_lifecycle_candidates = _cascade_context_terminals(
            observation,
            {
                **context_diagnostic_candidates,
                **retained_episode_candidates,
                **position_management_candidates,
            },
            self._candidate_theses,
            self._closed_contexts,
            previous_contexts,
        )
        lifecycle_candidates = _cascade_context_terminals(
            observation,
            lifecycle_candidates,
            self._candidate_theses,
            self._closed_contexts,
            previous_contexts,
        )
        lifecycle_candidates = _freeze_lsr_context_execution_owner(
            observation,
            lifecycle_candidates,
            self._lsr_execution_winners,
            self._lsr_execution_owner_snapshots,
            global_context,
            self.config,
        )
        for identity, candidate in lifecycle_candidates.items():
            if (
                candidate.playbook
                is Playbook.LIQUIDITY_SWEEP_REVERSAL
                and candidate.context_thesis_id is not None
                and candidate.episode_id is not None
                and candidate.phase in _TERMINAL_PHASES
            ):
                self._lsr_terminal_episode_ids.setdefault(
                    candidate.context_thesis_id,
                    set(),
                ).add(identity)
        thesis_candidates = {
            identity: lifecycle_candidates[identity]
            for identity in thesis_candidates
        }
        retained_episode_candidates = {
            identity: lifecycle_candidates[identity]
            for identity in retained_episode_candidates
        }
        position_management_candidates = {
            identity: lifecycle_candidates[identity]
            for identity in position_management_candidates
        }
        for identity, candidate in lifecycle_candidates.items():
            self._candidate_priors[identity] = candidate
        evaluated_by_slot = {
            key: [
                lifecycle_candidates.get(
                    candidate.candidate_id or "",
                    candidate,
                )
                for candidate in values
                if candidate.candidate_id is None
                or candidate.candidate_id in lifecycle_candidates
            ]
            for key, values in evaluated_by_slot.items()
        }
        context_theses, entry_episodes = _build_context_episode_views(
            observation,
            global_context,
            context_lifecycle_candidates,
            self._candidate_theses,
            previous_contexts,
            self._closed_contexts,
            lifecycle_candidates,
        )
        for context_id, context in context_theses.items():
            if context.lifecycle not in {"completed", "invalidated"}:
                continue
            self._lsr_retired_seed_keys.update(
                seed_key
                for seed_key, seeded_context_id
                in self._lsr_context_seed_ids.items()
                if seeded_context_id == context_id
            )
            existing_closure = self._closed_contexts.get(context_id)
            if existing_closure is not None:
                continue
            terminal_at = context.terminal_at or observation.asof
            self._closed_contexts[context_id] = _ContextClosure(
                phase=(
                    PlaybookPhase.COMPLETED
                    if context.lifecycle == "completed"
                    else PlaybookPhase.INVALIDATED
                ),
                reason=_normalized_context_reason(
                    context.terminal_reason or context.lifecycle
                ),
                source_ids=_identity_tuple(
                    context_id,
                    *context.opposing_event_ids,
                    None
                    if context.context_draw is None
                    else context.context_draw.level_id,
                    None
                    if context.structural_invalidation is None
                    else context.structural_invalidation.source_level_id,
                ),
                terminal_at=terminal_at,
                retain_until=max(
                    terminal_at + pd.Timedelta(minutes=1),
                    context.thesis_deadline
                    or _market_lifecycle_deadline(terminal_at),
                ),
            )
        retained_lsr_context_ids = {
            *context_theses,
            *self._closed_contexts,
        }
        self._lsr_execution_winners = {
            context_id: candidate_id
            for context_id, candidate_id
            in self._lsr_execution_winners.items()
            if context_id in retained_lsr_context_ids
        }
        self._lsr_execution_owner_snapshots = {
            context_id: owner
            for context_id, owner
            in self._lsr_execution_owner_snapshots.items()
            if context_id in retained_lsr_context_ids
        }
        retained_seed_keys = {
            seed_key
            for seed_key, context_id
            in self._lsr_context_seed_ids.items()
            if context_id in retained_lsr_context_ids
        }
        self._lsr_context_seeds = {
            seed_key: seed
            for seed_key, seed in self._lsr_context_seeds.items()
            if seed_key in retained_seed_keys
        }
        self._lsr_context_seed_ids = {
            seed_key: context_id
            for seed_key, context_id
            in self._lsr_context_seed_ids.items()
            if seed_key in retained_seed_keys
        }
        self._lsr_context_bindings = {
            seed_key: binding
            for seed_key, binding in self._lsr_context_bindings.items()
            if seed_key in retained_seed_keys
        }
        self._lsr_terminal_episode_ids = {
            context_id: set(candidate_ids)
            for context_id, candidate_ids
            in self._lsr_terminal_episode_ids.items()
            if context_id in retained_lsr_context_ids
        }
        compact_lsr_terminal_ids = {
            candidate_id
            for candidate_ids in self._lsr_terminal_episode_ids.values()
            for candidate_id in candidate_ids
        }
        self._candidate_priors = {
            identity: candidate
            for identity, candidate in self._candidate_priors.items()
            if identity not in compact_lsr_terminal_ids
        }
        self._candidate_theses = {
            identity: thesis
            for identity, thesis in self._candidate_theses.items()
            if identity not in compact_lsr_terminal_ids
        }
        phase_rank = {
            PlaybookPhase.INVALIDATED: 0,
            PlaybookPhase.COMPLETED: 0,
            PlaybookPhase.INACTIVE: 1,
            PlaybookPhase.FORMING: 2,
            PlaybookPhase.ARMED: 3,
            PlaybookPhase.WAITING_LOCATION: 4,
            PlaybookPhase.WAITING_TRIGGER: 5,
            PlaybookPhase.EXECUTABLE: 6,
            PlaybookPhase.WEAKENING: 7,
            PlaybookPhase.DELIVERING: 8,
            PlaybookPhase.ENTERED: 9,
        }
        for playbook in Playbook:
            protocol = self.registry.for_playbook(playbook)
            for direction in Direction:
                key = f"{playbook.value}:{direction.value}"
                values = evaluated_by_slot.get(key, ())
                if not values:
                    hypotheses[key] = _inactive_summary_belief(
                        playbook,
                        direction,
                        protocol,
                        self.registry,
                        asof=observation.asof,
                        calibration_version=self.calibrator.version,
                    )
                    continue
                execution_owner_values = tuple(
                    value
                    for value in values
                    if (
                        playbook
                        is Playbook.LIQUIDITY_SWEEP_REVERSAL
                        and value.context_thesis_id is not None
                        and self._lsr_execution_winners.get(
                            value.context_thesis_id
                        )
                        == value.candidate_id
                    )
                )
                selected = max(
                    execution_owner_values or values,
                    key=lambda value: (
                        phase_rank[value.phase],
                        int(value.market_thesis_action_bound),
                        float(value.entry_readiness or 0.0),
                        float(value.thesis_strength or 0.0),
                        -value.uncertainty,
                        value.required_root_id or "",
                    ),
                )
                # Six slots are read-only projections.  They carry no prior or
                # action identity of their own and identify the exact root
                # candidate from which their display state was copied.
                hypotheses[key] = (
                    selected
                    if selected.candidate_id is None
                    else replace(
                        selected,
                        candidate_id=None,
                        required_root_id=None,
                        record_kind="summary",
                        summary_source_candidate_id=selected.candidate_id,
                    )
                )
        if global_context is not None and scene_graph is not None:
            global_context = _finalize_global_context(
                global_context,
                observation,
                scene_graph,
                thesis_candidates,
            )
        (
            path_competition_state,
            path_update_records,
            dol_rankings,
            dol_candidate_exclusions,
        ) = self._update_shadow_path_diagnostics(
            observation,
            global_context,
            context_theses,
            lifecycle_candidates,
            previous_belief,
        )
        base_belief = MarketBelief(
            asof=observation.asof,
            hypotheses=hypotheses,
            global_context=global_context,
            thesis_candidates=thesis_candidates,
            retained_episode_candidates=(
                retained_episode_candidates
            ),
            position_management_candidates=(
                position_management_candidates
            ),
            context_theses=context_theses,
            entry_episodes=entry_episodes,
            path_competition_state=path_competition_state,
            path_update_records_this_clock=path_update_records,
            dol_rankings=dol_rankings,
            dol_probabilities=self._dol_probability_results,
            dol_candidate_exclusions=dol_candidate_exclusions,
            path_protocol_status=self.path_protocol.status,
            path_authority=self.path_protocol.authority,
            dol_probability_protocol_fingerprint=(
                self.dol_probability_protocol.fingerprint
            ),
            signal_policy_protocol_fingerprint=self.signal_policy.fingerprint,
        )
        if scene_graph is not None:
            ranked_candidates = base_belief.ranked()
            current_top = next(
                (
                    candidate
                    for candidate in ranked_candidates
                    for owner in (
                        self._lsr_execution_winners.get(
                            candidate.context_thesis_id or ""
                        ),
                    )
                    if (
                        candidate.playbook
                        is not Playbook.LIQUIDITY_SWEEP_REVERSAL
                        or owner is None
                        or (
                            not owner.startswith(
                                "lsr-ambiguous-same-clock:"
                            )
                            and candidate.candidate_id == owner
                        )
                    )
                ),
                None,
            )
            focus_state = select_focus(
                # Losing the last realtime root is itself a candidate change.
                # Do not let select_focus fall back to yesterday's dormant
                # ranked candidate and carry an executable/delivering question
                # into a clock with no action identity.
                previous_belief if current_top is not None else None,
                observation,
                scene_delta,
                scene_graph,
                global_context,
                preferred_candidate=current_top,
            )
            # Unexplained roots remain an offline diagnostic.  They may shape
            # the question only when no registered root candidate exists;
            # they never become a second action veto or Focus identity.
            if current_top is None and global_context is not None and (
                global_context.unexplained_structured_episode_ids
            ):
                focus_state = replace(
                    focus_state,
                    reason_codes=tuple(
                        dict.fromkeys(
                            (*focus_state.reason_codes, "unexplained_roots_present")
                        )
                    ),
                    question="which registered mechanism explains the active market root",
                    resolution_status=EvidenceStatus.UNKNOWN,
                )
            focused_scene = scene_graph.query(
                focus_state,
                context_ids=tuple(
                    dict.fromkeys(
                        value
                        for hypothesis in (
                            *thesis_candidates.values(),
                            *position_management_candidates.values(),
                        )
                        for value in (
                            hypothesis.required_root_id,
                            hypothesis.context_id,
                            hypothesis.setup_context_id,
                            hypothesis.initiating_event_id,
                        )
                        if value is not None
                    )
                ),
                completed_only=True,
                ready_timeframes=tuple(
                    timeframe.value
                    for timeframe in observation.active_timeframes
                    if observation.frame(timeframe).ready
                ),
            )
            context_candidates = {
                **thesis_candidates,
                **position_management_candidates,
            }
            context_hypotheses = build_hypothesis_states(
                context_candidates,
                scene_graph,
                focused_scene,
            )
            dominant_id = (
                None if current_top is None else current_top.candidate_id
            )
            competing_ids = tuple(
                identity for identity, candidate in context_candidates.items()
                if identity != dominant_id
                and candidate.phase
                not in {
                    PlaybookPhase.INACTIVE,
                    PlaybookPhase.COMPLETED,
                    PlaybookPhase.INVALIDATED,
                }
            )
            dominant_context = (
                None
                if dominant_id is None
                else context_hypotheses[dominant_id]
            )
            focused_conflict_ids = set(
                focused_scene.cross_scale_conflicts
            )
            dominant_conflict_ids = tuple(
                conflict_id
                for conflict_id in (
                    ()
                    if dominant_context is None
                    else dominant_context.material_conflict_ids
                )
                if conflict_id in focused_conflict_ids
            )
            if dominant_conflict_ids:
                focus_resolution = EvidenceStatus.CONFLICTING
            elif dominant_context is None:
                focus_resolution = EvidenceStatus.UNKNOWN
            elif EvidenceStatus.CONFLICTING in (
                dominant_context.ambiguous_evidence.values()
            ):
                focus_resolution = EvidenceStatus.CONFLICTING
            elif EvidenceStatus.AMBIGUOUS in (
                dominant_context.ambiguous_evidence.values()
            ):
                focus_resolution = EvidenceStatus.AMBIGUOUS
            elif EvidenceStatus.UNKNOWN in (
                dominant_context.ambiguous_evidence.values()
            ):
                focus_resolution = EvidenceStatus.UNKNOWN
            elif EvidenceStatus.UNKNOWN in (
                dominant_context.missing_evidence.values()
            ):
                focus_resolution = EvidenceStatus.UNKNOWN
            elif dominant_context.missing_evidence:
                focus_resolution = EvidenceStatus.NOT_OBSERVED
            else:
                focus_resolution = EvidenceStatus.CONFIRMED
            focus_reasons = list(focus_state.reason_codes)
            focus_question = focus_state.question
            if dominant_conflict_ids:
                focus_reasons.append("root_related_cross_scale_conflict")
                focus_question = (
                    "which scale owns the active structural conflict"
                )
            focus_state = replace(
                focus_state,
                resolution_status=focus_resolution,
                reason_codes=tuple(dict.fromkeys(focus_reasons)),
                question=focus_question,
                hypothesis_id=dominant_id,
            )
            unresolved = tuple(
                dict.fromkeys(
                    (
                        *focused_scene.unresolved_ambiguities,
                        *(
                            f"{identity}:{name}"
                            for identity, context in context_hypotheses.items()
                            for name in context.ambiguous_evidence
                        ),
                        *(
                            f"{identity}:{name}"
                            for identity, context in context_hypotheses.items()
                            for name, status in context.missing_evidence.items()
                            if status is EvidenceStatus.UNKNOWN
                        ),
                    )
                )
            )
            self._belief = MarketBelief(
                asof=observation.asof,
                hypotheses=hypotheses,
                context_hypotheses=context_hypotheses,
                dominant_hypothesis_id=dominant_id,
                competing_hypothesis_ids=competing_ids,
                focus_state=focus_state,
                cross_scale_conflicts=(
                    dominant_conflict_ids
                ),
                unresolved_ambiguities=unresolved,
                scene_revision_id=scene_graph.revision_id,
                global_context=global_context,
                thesis_candidates=thesis_candidates,
                retained_episode_candidates=(
                    retained_episode_candidates
                ),
                position_management_candidates=(
                    position_management_candidates
                ),
                context_theses=context_theses,
                entry_episodes=entry_episodes,
                path_competition_state=path_competition_state,
                path_update_records_this_clock=path_update_records,
                dol_rankings=dol_rankings,
                dol_probabilities=self._dol_probability_results,
                dol_candidate_exclusions=dol_candidate_exclusions,
                path_protocol_status=self.path_protocol.status,
                path_authority=self.path_protocol.authority,
                dol_probability_protocol_fingerprint=(
                    self.dol_probability_protocol.fingerprint
                ),
                signal_policy_protocol_fingerprint=self.signal_policy.fingerprint,
            )
        else:
            self._belief = base_belief

        self._belief = self._project_shadow_signal_assessments(
            observation,
            self._belief,
        )

        # Root-specific priors are live working state, not an archive.  Keep
        # current analytical roots, the one frozen root that may own a
        # position, and at most one completed-bar grace period for a terminal
        # action candidate to prove a causally newer rearm.  A released
        # position and candidates from a closed market epoch must not remain
        # latent in a long replay or live process.
        terminal_grace: dict[str, HypothesisBelief] = {}
        if not candidate_epoch_changed:
            for identity, previous_candidate in (
                {
                    **prior_thesis_candidates,
                    **prior_retained_candidates,
                }.items()
            ):
                if (
                    identity in thesis_candidates
                    or identity in retained_episode_candidates
                    or identity in position_management_candidates
                    or identity in compact_lsr_terminal_ids
                ):
                    continue
                cached = self._candidate_priors.get(
                    identity,
                    previous_candidate,
                )
                if cached.phase in _TERMINAL_PHASES:
                    terminal_grace[identity] = cached
        retained_priors = {
            identity: candidate
            for identity, candidate in {
                **terminal_grace,
                **thesis_candidates,
                **retained_episode_candidates,
                **position_management_candidates,
            }.items()
            if identity not in compact_lsr_terminal_ids
        }
        self._candidate_priors = retained_priors
        self._candidate_theses = {
            identity: thesis
            for identity, thesis in self._candidate_theses.items()
            if identity in retained_priors
        }
        return self._belief


__all__ = ["BrainConfig", "PlaybookBrain"]
