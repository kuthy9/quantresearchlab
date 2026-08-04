"""Small preregistered playbook set with continuous belief/state updates."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Mapping, Sequence

import pandas as pd

from .calibration import TypedBrainCalibrator
from .model import (
    BOSLifecycle,
    BOSScope,
    DealingRangeLifecycle,
    Direction,
    DrawSelection,
    EntryLocationLifecycle,
    EntryLocationState,
    Evidence,
    EventKind,
    FrozenRangeAuctionContext,
    HypothesisBelief,
    HypothesisSequenceState,
    LiquidityInventoryLifecycle,
    LiquidityLevel,
    LiquidityRoute,
    MarketBelief,
    MarketObservation,
    ManipulationLifecycle,
    MicroBOSReference,
    PathSequenceLifecycle,
    PathSequenceState,
    Playbook,
    PlaybookPhase,
    PositionSnapshot,
    SequenceStepState,
    StructuralLevel,
    StructureLifecycle,
    Timeframe,
    TradePlan,
    clamp,
)
from .playbook_registry import (
    PlaybookProtocol,
    PlaybookRegistry,
    load_playbook_registry,
)
from .scene_graph import (
    EvidenceStatus,
    SceneGraphDelta,
    TemporalMarketSceneGraph,
    build_hypothesis_states,
    select_focus,
    supplement_focus_once,
)


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
    draw_selection: DrawSelection | None = None
    liquidity_route: LiquidityRoute | None = None
    thesis_target: float = 0.0
    location_quality: float = 0.0
    entry_readiness: float = 0.0
    delivery_quality: float = 0.0
    typed_uncertainty: float = 1.0
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


@dataclass(frozen=True)
class _SequenceSignal:
    value: float
    observed_at: pd.Timestamp | None
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", clamp(self.value))


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
        location.contact_reference_price
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
    tier = (
        0
        if higher_timeframe and pooled
        else 1
        if item.kind == "range_boundary"
        else 2
        if higher_timeframe and item.kind == "swing"
        else 3
        if pooled
        else 4
    )
    return (
        tier,
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
            item.price,
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
        price=item.price,
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
    require_preferred: bool = False,
) -> LiquidityLevel | None:
    levels = _visible_level_map(observation)
    if require_preferred and preferred_id is None:
        return None
    if preferred_id is not None:
        preferred = levels.get(preferred_id)
        if (
            preferred is not None
            and (
                required_timeframe is None
                or preferred.timeframe is required_timeframe
            )
            and _target_is_deliverable(
                preferred,
                direction,
                entry,
                config.tick_size,
            )
        ):
            return preferred
        if require_preferred:
            return None
    candidates = [
        level
        for level in levels.values()
        if (
            (
                required_timeframe is None
                or level.timeframe is required_timeframe
            )
            and _target_is_deliverable(
                level,
                direction,
                entry,
                config.tick_size,
            )
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
) -> LiquidityLevel | None:
    """Choose the nearest visible delivery before the directional draw."""

    levels = _visible_level_map(observation)
    if preferred_id is not None:
        preferred = levels.get(preferred_id)
        if (
            preferred is not None
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


def _liquidity_route(
    observation: MarketObservation,
    direction: Direction,
    entry: float,
    *,
    context_draw: LiquidityLevel | None,
    primary_target: LiquidityLevel | None,
    prior_route: LiquidityRoute | None = None,
) -> LiquidityRoute | None:
    if primary_target is None:
        return None
    deliverable = sorted(
        (
            level
            for level in _visible_levels(observation)
            if _target_is_deliverable(
                level,
                direction,
                entry,
                0.0,
            )
            and (
                context_draw is None
                or direction.sign * (level.price - context_draw.price) <= 0.0
            )
        ),
        key=lambda level: abs(level.price - entry),
    )
    intermediates = tuple(
        level.level_id
        for level in deliverable
        if level.level_id
        not in {
            primary_target.level_id,
            None if context_draw is None else context_draw.level_id,
        }
    )
    terminal = (
        context_draw
        if context_draw is not None
        else max(
            deliverable,
            key=lambda level: abs(level.price - entry),
            default=primary_target,
        )
    )
    blockers = tuple(
        level.level_id
        for level in deliverable
        if abs(level.price - entry)
        < abs(primary_target.price - entry)
    )
    source_path = tuple(
        dict.fromkeys(
            value
            for value in (
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
        and prior_route.intermediate_liquidity_ids == intermediates
        and prior_route.path_blocker_ids == blockers
        and prior_route.source_path_ids == source_path
    ):
        return prior_route
    raw = (
        f"{observation.asof.isoformat()}|{direction.value}|"
        f"{None if context_draw is None else context_draw.level_id}|"
        f"{primary_target.level_id}|{terminal.level_id}"
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
        path_blocker_ids=blockers,
        source_path_ids=source_path,
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
    """Describe remaining causal path after accounting for H1 blockage.

    Reward magnitude remains a separate decision input.  The raw value keeps
    the preregistered remaining-path fraction and caps it when a confirmed H1
    obstruction lies before the frozen draw.  It never inspects future bars or
    execution fields; empirical delivery reliability is fitted later.
    """

    if plan is None or not plan.targets:
        return 0.0
    if (
        plan.liquidity_route is not None
        and plan.liquidity_route.path_blocker_ids
    ):
        return 0.0
    primary = plan.targets[0]
    if any(
        level.level_id != primary.level_id
        and _target_is_deliverable(
            level,
            direction,
            float(plan.planned_entry),
            0.0,
        )
        and abs(level.price - plan.planned_entry)
        < abs(primary.price - plan.planned_entry)
        for level in _visible_levels(observation)
    ):
        return 0.0
    target_distance = abs(plan.targets[0].price - observation.price)
    if target_distance <= 0.0:
        return 1.0
    remaining_fraction = clamp(
        plan.remaining_path_R / max(plan.primary_target_R, 1e-12)
    )
    h1 = observation.frame(Timeframe.H1)
    h1_atr = max(float(h1.metrics.get("atr", 0.0)), 1e-12)
    obstruction_atr = float(
        h1.metrics.get(
            (
                "up_path_obstruction_atr"
                if direction is Direction.LONG
                else "down_path_obstruction_atr"
            ),
            0.0,
        )
    )
    target_distance_atr = target_distance / h1_atr
    clearance = clamp(
        obstruction_atr / max(target_distance_atr, 1e-12)
    )
    return min(remaining_fraction, clearance)


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
    liquidity_route: LiquidityRoute | None = None,
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
        location.contact_reference_price
        if location.contact_reference_price is not None
        else location.near_edge
    )
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


def _typed_uncertainty(
    *,
    market_support: Sequence[float],
    market_contradictions: Sequence[float],
    authority_missing: float,
) -> float:
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
    evidence_slots = max(1.0, float(len(market_support)))
    evidence_coverage = clamp(
        (support_mass + contradiction_mass) / evidence_slots
    )
    missing = max(
        clamp(authority_missing),
        1.0 - evidence_coverage,
    )
    return clamp(1.0 - (1.0 - conflict) * (1.0 - missing))


def _typed_market_uncertainty(
    observation: MarketObservation,
    support: Mapping[str, float],
    contradict: Mapping[str, float],
    *,
    semantic_authority_missing: float = 0.0,
) -> float:
    execution_names = {
        "remaining_path_available",
        "execution_fillability",
        "remaining_path_consumed",
        "execution_unavailable",
    }
    missing_frames = sum(
        not frame.ready for frame in observation.frames.values()
    ) / max(1.0, float(len(observation.frames)))
    authority_missing = max(
        missing_frames,
        float(not observation.liquidity_inventory_authoritative),
        float(not observation.group5_authoritative),
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
) -> EntryLocationState | None:
    by_id = {
        location.location_id: location
        for location in observation.entry_locations
    }
    if (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.entry_location_id is not None
    ):
        bound = by_id.get(prior.entry_location_id)
        return (
            bound
            if bound is not None and bound.direction is direction
            else None
        )
    terminal_cutoff = (
        prior.phase_started_at
        if prior is not None and prior.phase in _TERMINAL_PHASES
        else None
    )
    path_by_location = {
        path.context_id: path
        for path in observation.path_sequences
        if (
            path.context_kind == "zone_return"
            and path.direction is direction
        )
    }
    candidates = [
        location
        for location in observation.entry_locations
        if (
            location.direction is direction
            and location.lifecycle is not EntryLocationLifecycle.LEFT
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
) -> _Evaluation:
    structures = {
        item.structure_id: item
        for item in observation.frame(Timeframe.H4).structures
        if (
            item.structure_id is not None
            and item.direction is direction
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
        require_preferred=preferred_context_draw_id is not None,
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
            and item.resolved_at is not None
            and context_clock is not None
            and item.resolved_at > context_clock
        )
    }
    frozen_h1_sources = _prior_step_sources(
        prior,
        "h1_continuation_bos",
    )
    frozen_h1_bos_id = (
        frozen_h1_sources[0] if frozen_h1_sources else None
    )
    h1_bos = (
        h1_candidates.get(frozen_h1_bos_id)
        if frozen_h1_bos_id is not None
        else (
            min(
                h1_candidates.values(),
                key=lambda item: (item.resolved_at, item.bos_id),
            )
            if h1_candidates
            else None
        )
    )
    location = _select_entry_location(
        observation,
        direction,
        after=None if h1_bos is None else h1_bos.resolved_at,
        prior=prior,
    )
    entry_path = _path_for_location(observation, location)
    if (
        location is not None
        and (
            h1_bos is None
            or location.formed_at <= h1_bos.resolved_at
        )
    ):
        location = None
        entry_path = None
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
    trigger_step = {
        "zone_rejection_observed": wick_trigger,
        "qualified_reacceptance_held": held_trigger,
        "micro_bos_aligned": (
            micro_trigger if qualified_micro is not None else None
        ),
    }.get(
        None if entry_path is None else entry_path.transition_reason
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
    trigger_ready = bool(
        first_pullback is not None
        and trigger_step is not None
        and ambiguity_step is None
        and contradiction_step is None
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
            require_preferred=preferred_context_draw_id is not None,
        )
    else:
        planned_entry = reference_entry
    primary_target = (
        None
        if location is None or context_draw is None
        else _select_primary_deliverable_target(
            observation,
            direction,
            float(planned_entry),
            config,
            context_draw=context_draw,
            preferred_id=preferred_primary_target_id,
        )
    )
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
    )
    remaining_ok = bool(
        plan is not None
        and plan.remaining_path_R >= config.minimum_remaining_path_R
    )
    execution_missing = _execution_unavailable(observation)
    frozen_structure_missing = bool(
        observation.frame(Timeframe.H4).ready
        and frozen_structure_id is not None
        and structure is None
    )
    frozen_h1_bos_missing = bool(
        observation.frame(Timeframe.H1).ready
        and frozen_h1_bos_id is not None
        and h1_bos is None
    )
    frozen_location_missing = bool(
        observation.group5_authoritative
        and prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.entry_location_id is not None
        and location is None
    )
    opposed_structures = tuple(
        item
        for item in observation.frame(Timeframe.H4).structures
        if (
            item.direction is not direction
            and item.lifecycle is StructureLifecycle.CONFIRMED
            and item.confirmed_at is not None
            and (
                structure is None
                or item.confirmed_at >= structure.confirmed_at
            )
        )
    )
    opposed_structure = frozen_structure_missing or bool(
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
    context_draw_missing = bool(
        observation.liquidity_inventory_authoritative
        and preferred_context_draw_id is not None
        and context_draw is None
    )
    primary_target_missing = bool(
        observation.liquidity_inventory_authoritative
        and preferred_primary_target_id is not None
        and primary_target is None
    )
    trigger_contradiction = contradiction_step is not None
    trigger_ambiguous = ambiguity_step is not None
    hard_gates = {
        "h4_structure_and_draw": bool(
            structure is not None
            and context_draw is not None
            and not opposed_structure
        ),
        "h1_continuation_bos": h1_bos is not None,
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
        "trigger_opposed_or_ambiguous": float(
            trigger_contradiction
        ),
        "draw_consumed_or_missing": float(
            context_draw_missing or primary_target_missing
        ),
        "remaining_path_consumed": float(
            plan is not None and not remaining_ok
        ),
        "execution_unavailable": float(execution_missing),
    }
    group_scores = {
        "structure": min(
            support["h4_structure_direction"],
            support["h1_continuation_bos"],
        )
        * (1.0 - contradict["opposed_structure"]),
        "displacement": support["m5_displacement_zone"],
        "location": _location_quality(location)
        * (1.0 - contradict["zone_left_or_failed"]),
        "liquidity": support["directional_draw_visible"]
        * (1.0 - contradict["draw_consumed_or_missing"]),
        "trigger": support["typed_entry_trigger"]
        * (1.0 - contradict["trigger_opposed_or_ambiguous"]),
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
    elif context_draw_missing:
        terminal_reason = "context_draw_consumed_or_missing"
        terminal_source_ids = _identity_tuple(
            preferred_context_draw_id
        )
    elif primary_target_missing:
        terminal_reason = "primary_target_consumed_or_missing"
        terminal_source_ids = _identity_tuple(
            preferred_primary_target_id
        )
    elif frozen_h1_bos_missing:
        terminal_reason = "frozen_h1_bos_missing"
        terminal_source_ids = _identity_tuple(frozen_h1_bos_id)
    elif frozen_location_missing:
        terminal_reason = "frozen_entry_location_missing"
        terminal_source_ids = _identity_tuple(
            None if prior is None else prior.entry_location_id
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
            "h1_continuation_bos": _SequenceSignal(
                float(h1_bos is not None),
                None if h1_bos is None else h1_bos.resolved_at,
                ()
                if h1_bos is None
                else (
                    h1_bos.bos_id,
                    h1_bos.target_swing_id,
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
                    if trigger_step is None
                    else trigger_step.observed_at
                ),
                ()
                if trigger_step is None
                else tuple(
                    value
                    for value in (
                        trigger_step.source_entity_id,
                        trigger_step.source_event_id,
                    )
                    if value is not None
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
        draw_selection=draw_selection,
        liquidity_route=liquidity_route,
        thesis_target=min(
            (
                0.0
                if structure is None
                else clamp(
                    structure.cumulative_magnitude_atr
                    / max(1.0, float(structure.sequence_count))
                )
            ),
            0.0 if h1_bos is None else clamp(h1_bos.strength),
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
        typed_uncertainty=_typed_market_uncertainty(
            observation,
            support,
            contradict,
            semantic_authority_missing=float(trigger_ambiguous),
        ),
        evidence_group_scores=group_scores,
        hard_gate_results=hard_gates,
        invalidated=bool(
            opposed_structure
            or frozen_h1_bos_missing
            or frozen_location_missing
            or zone_failed
            or trigger_contradiction
            or censored_path
            or (
                observation.liquidity_inventory_authoritative
                and preferred_context_draw_id is not None
                and preferred_context_draw_id not in visible
            )
            or (
                observation.liquidity_inventory_authoritative
                and preferred_primary_target_id is not None
                and preferred_primary_target_id not in visible
            )
        ),
        entry_window_expired=stale_trigger,
        terminal_reason=terminal_reason,
        terminal_source_ids=terminal_source_ids,
    )


def _select_pool_path(
    observation: MarketObservation,
    direction: Direction,
    prior: HypothesisBelief | None,
) -> PathSequenceState | None:
    paths = {
        path.sequence_id: path
        for path in observation.path_sequences
        if (
            path.context_kind == "pool_reversal"
            and path.direction is direction
        )
    }
    if (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.setup_context_id is not None
    ):
        return paths.get(prior.setup_context_id)
    terminal_cutoff = (
        prior.phase_started_at
        if prior is not None and prior.phase in _TERMINAL_PHASES
        else None
    )
    candidates = [
        path
        for path in paths.values()
        if (
            terminal_cutoff is None
            or path.formed_at > terminal_cutoff
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda path: (path.formed_at, path.sequence_id),
    )


def _typed_lsr(
    observation: MarketObservation,
    direction: Direction,
    protocol: PlaybookProtocol,
    prior: HypothesisBelief | None,
    config: BrainConfig,
) -> _Evaluation:
    pool_path = _select_pool_path(observation, direction, prior)
    manipulation = (
        next(
            (
                item
                for item in observation.manipulations
                if (
                    pool_path is not None
                    and item.manipulation_id == pool_path.context_id
                    and item.source_kind == "formed_liquidity_pool"
                )
            ),
            None,
        )
        if pool_path is not None
        else None
    )
    pool_swept = _path_step(pool_path, {"pool_swept"})
    sweep_return = _path_step(
        pool_path,
        {"sweep_rejection", "reacceptance_held"},
    )
    pool_ambiguity = _path_step(
        pool_path,
        {"micro_bos_ambiguous"},
    )
    pool_contradiction = _path_step(
        pool_path,
        {
            "accepted_outside",
            "reacceptance_failed",
            "micro_bos_opposed",
        },
    )
    return_clock = (
        None if sweep_return is None else sweep_return.observed_at
    )
    location = _select_entry_location(
        observation,
        direction,
        after=return_clock,
        prior=prior,
    )
    entry_path = _path_for_location(observation, location)
    if (
        location is not None
        and (
            return_clock is None
            or location.formed_at <= return_clock
        )
    ):
        location = None
        entry_path = None
    first_pullback = _path_step(entry_path, {"first_pullback"})
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
        and micro_trigger is not None
        and qualified_micro is not None
        and entry_ambiguity is None
        and entry_contradiction is None
    )
    context_identity = (
        None if pool_path is None else pool_path.sequence_id
    )
    episode_identity = context_identity
    setup_identity = episode_identity
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
    preferred_context_draw_id = (
        prior.liquidity_route.context_draw_id
        if (
            prior is not None
            and prior.phase not in _TERMINAL_PHASES
            and prior.setup_context_id == setup_identity
            and prior.liquidity_route is not None
        )
        else None
    )
    preferred_primary_target_id = (
        prior.liquidity_route.primary_deliverable_target_id
        if (
            prior is not None
            and prior.phase not in _TERMINAL_PHASES
            and prior.setup_context_id == setup_identity
            and prior.liquidity_route is not None
        )
        else prior.draw_selection.draw_id
        if (
            prior is not None
            and prior.phase not in _TERMINAL_PHASES
            and prior.setup_context_id == setup_identity
            and prior.draw_selection is not None
        )
        else prior.plan.selected_draw_id
        if prior_plan_same_setup
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
    context_draw = _select_target(
        observation,
        direction,
        planned_entry,
        config,
        preferred_id=preferred_context_draw_id,
        require_preferred=preferred_context_draw_id is not None,
    )
    primary_target = (
        None
        if location is None
        else _select_primary_deliverable_target(
            observation,
            direction,
            planned_entry,
            config,
            context_draw=context_draw,
            preferred_id=preferred_primary_target_id,
        )
    )
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
            prior_route=(
                None if prior is None else prior.liquidity_route
            ),
        )
    )
    plan = _typed_plan(
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
        liquidity_route=liquidity_route,
    )
    remaining_ok = bool(
        plan is not None
        and plan.remaining_path_R >= config.minimum_remaining_path_R
    )
    execution_missing = _execution_unavailable(observation)
    accepted_outside = bool(
        pool_contradiction is not None
        or (
            manipulation is not None
            and manipulation.lifecycle
            is ManipulationLifecycle.ACCEPTED_OUTSIDE
        )
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
        observation.group5_authoritative
        and prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.setup_context_id is not None
        and pool_path is None
    )
    frozen_location_missing = bool(
        observation.group5_authoritative
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
        ),
        "outside_acceptance_failed": bool(
            sweep_return is not None
            and not accepted_outside
            and pool_ambiguity is None
        ),
        "reverse_displacement_zone": bool(
            location is not None and entry_path is not None
        ),
        "reversal_first_pullback": first_pullback is not None,
        "aligned_micro_bos_trigger": trigger_ready,
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
        "reversal_first_pullback": float(
            hard_gates["reversal_first_pullback"]
        ),
        "aligned_micro_bos_trigger": float(trigger_ready),
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
        "reverse_displacement_missing": 0.0,
        "entry_zone_left": float(zone_left),
        "micro_bos_opposed_or_ambiguous": float(
            entry_contradiction is not None
        ),
        "draw_consumed_or_missing": float(
            observation.liquidity_inventory_authoritative
            and (
                (
                    preferred_context_draw_id is not None
                    and context_draw is None
                )
                or (
                    preferred_primary_target_id is not None
                    and primary_target is None
                )
            )
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
        "trigger": support["aligned_micro_bos_trigger"]
        * (1.0 - contradict["micro_bos_opposed_or_ambiguous"]),
        "execution": support["execution_fillability"]
        * (1.0 - contradict["execution_unavailable"]),
    }
    pool_censored = bool(
        pool_path is not None
        and pool_path.lifecycle is PathSequenceLifecycle.CENSORED
    )
    entry_path_censored = bool(
        entry_path is not None
        and entry_path.lifecycle is PathSequenceLifecycle.CENSORED
    )
    selected_draw_missing = bool(
        observation.liquidity_inventory_authoritative
        and prior_plan_same_setup
        and prior is not None
        and prior.plan is not None
        and prior.plan.selected_draw_id
        not in _visible_level_map(observation)
    )
    terminal_reason: str | None = None
    terminal_source_ids: tuple[str, ...] = ()
    if accepted_outside:
        terminal_reason = "accepted_outside_or_failed"
        terminal_source_ids = _identity_tuple(
            None if pool_path is None else pool_path.sequence_id,
            None
            if pool_contradiction is None
            else pool_contradiction.source_entity_id,
            None
            if pool_contradiction is None
            else pool_contradiction.source_event_id,
        )
    elif frozen_pool_missing:
        terminal_reason = "frozen_pool_path_missing"
        terminal_source_ids = _identity_tuple(
            None if prior is None else prior.setup_context_id
        )
    elif frozen_location_missing:
        terminal_reason = "frozen_entry_location_missing"
        terminal_source_ids = _identity_tuple(
            None if prior is None else prior.entry_location_id
        )
    elif (
        observation.group5_authoritative
        and pool_path is not None
        and manipulation is None
    ):
        terminal_reason = "source_manipulation_missing"
        terminal_source_ids = _identity_tuple(pool_path.context_id)
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
    elif pool_censored:
        terminal_reason = "pool_path_censored"
        terminal_source_ids = _identity_tuple(
            None if pool_path is None else pool_path.sequence_id
        )
    elif entry_path_censored:
        terminal_reason = "entry_path_censored"
        terminal_source_ids = _identity_tuple(
            None if entry_path is None else entry_path.sequence_id
        )
    elif selected_draw_missing:
        terminal_reason = "selected_draw_consumed_or_missing"
        terminal_source_ids = _identity_tuple(
            None
            if prior is None or prior.plan is None
            else prior.plan.selected_draw_id
        )
    elif stale_trigger:
        terminal_reason = "entry_window_expired"
        terminal_source_ids = _identity_tuple(
            None if entry_path is None else entry_path.sequence_id
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
                None if location is None else location.formed_at,
                ()
                if location is None
                else (
                    location.location_id,
                    location.source_displacement_id,
                    location.source_zone_id,
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
            "aligned_micro_bos_trigger": _SequenceSignal(
                float(trigger_ready),
                (
                    None
                    if qualified_micro is None
                    else qualified_micro.resolved_at
                ),
                ()
                if qualified_micro is None
                else (
                    qualified_micro.reference_id,
                    qualified_micro.bos_id,
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
        draw_selection=draw_selection,
        liquidity_route=liquidity_route,
        thesis_target=min(
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
        ),
        location_quality=group_scores["location"],
        entry_readiness=(
            0.0
            if not trigger_ready or micro_trigger is None
            else clamp(micro_trigger.strength)
        ),
        delivery_quality=_delivery_quality(
            observation,
            direction,
            plan,
        ),
        typed_uncertainty=_typed_market_uncertainty(
            observation,
            support,
            contradict,
            semantic_authority_missing=float(
                pool_ambiguity is not None
                or entry_ambiguity is not None
            ),
        ),
        evidence_group_scores=group_scores,
        hard_gate_results=hard_gates,
        invalidated=bool(
            accepted_outside
            or frozen_pool_missing
            or frozen_location_missing
            or (
                observation.group5_authoritative
                and pool_path is not None
                and manipulation is None
            )
            or zone_left
            or entry_contradiction is not None
            or pool_censored
            or entry_path_censored
            or selected_draw_missing
        ),
        entry_window_expired=stale_trigger,
        terminal_reason=terminal_reason,
        terminal_source_ids=terminal_source_ids,
    )


def _select_favr_context(
    observation: MarketObservation,
    direction: Direction,
    prior: HypothesisBelief | None,
):
    ranges = {
        state.range_id: state
        for state in observation.frame(Timeframe.H1).dealing_ranges
    }
    range_manipulations = tuple(
        state
        for state in observation.manipulations
        if state.source_kind == "mature_range_boundary"
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
        return bool(
            location.direction is direction
            and location.lifecycle is not EntryLocationLifecycle.LEFT
            and location.formed_at > manipulation.reaccepted_at
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
) -> _Evaluation:
    dealing_range, manipulation = _select_favr_context(
        observation,
        direction,
        prior,
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
        and observation.liquidity_inventory_authoritative
        and prior_episode_live
        and prior is not None
        and prior.initiating_event_id is not None
        and prior.initiating_event_id not in observed_manipulation_ids
    )
    frozen_location_missing = bool(
        observation.group5_authoritative
        and prior_episode_live
        and prior is not None
        and prior.entry_location_id is not None
        and prior.entry_location_id not in observed_location_ids
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
    location = _select_favr_location(
        observation,
        direction,
        dealing_range,
        manipulation,
        prior,
    )
    entry_path = _path_for_location(observation, location)
    first_pullback = _path_step(entry_path, {"first_pullback"})
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
    hard_gates = {
        "mature_dealing_range": mature,
        "range_boundary_sweep_reaccepted": range_reaccepted,
        "opposite_displacement_zone_inside_range": bool(
            location is not None and entry_path is not None
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
        ),
        "entry_zone_left": float(zone_left),
        "trigger_opposed_or_ambiguous": float(
            entry_contradiction is not None
        ),
        "draw_consumed_or_missing": float(
            observation.liquidity_inventory_authoritative
            and range_reaccepted
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
        * (1.0 - contradict["trigger_opposed_or_ambiguous"]),
        "execution": support["execution_fillability"]
        * (1.0 - contradict["execution_unavailable"]),
    }
    terminal_reason: str | None = None
    terminal_source_ids: tuple[str, ...] = ()
    if accepted_outside:
        terminal_reason = "range_auction_accepted_outside"
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
    elif midpoint_chase:
        terminal_reason = "planned_entry_chased_past_range_value"
    elif (
        prior is not None
        and prior.phase not in _TERMINAL_PHASES
        and prior.draw_selection is not None
        and observation.liquidity_inventory_authoritative
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
                None
                if first_pullback is None
                else first_pullback.observed_at,
                ()
                if first_pullback is None
                else (first_pullback.source_entity_id,),
            ),
            "aligned_entry_trigger": _SequenceSignal(
                float(trigger_ready),
                None if trigger_step is None else trigger_step.observed_at,
                ()
                if trigger_step is None
                else tuple(
                    value
                    for value in (
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
        typed_uncertainty=_typed_market_uncertainty(
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
                float(entry_ambiguity is not None),
            ),
        ),
        evidence_group_scores=group_scores,
        hard_gate_results=hard_gates,
        invalidated=bool(
            accepted_outside
            or frozen_range_missing
            or frozen_manipulation_missing
            or frozen_location_missing
            or (dealing_range is not None and not mature)
            or zone_left
            or entry_contradiction is not None
            or midpoint_chase
            or (
                entry_path is not None
                and entry_path.lifecycle is PathSequenceLifecycle.CENSORED
            )
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
) -> _Evaluation:
    if playbook is Playbook.FAILED_AUCTION_VALUE_RETURN:
        return _typed_favr(
            observation,
            direction,
            protocol,
            prior,
            config,
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
        )
    return _typed_lsr(
        observation,
        direction,
        protocol,
        prior,
        config,
    )


def _require_connected_graph_sequence(
    playbook: Playbook,
    protocol: PlaybookProtocol,
    evaluation: _Evaluation,
    scene_graph: TemporalMarketSceneGraph | None,
    prior: HypothesisBelief | None,
) -> _Evaluation:
    """Fail closed on disconnected DFP/LSR facts without inventing a veto.

    Snapshot detectors still describe each fact.  In the graph-backed engine,
    however, a later fact may advance a hypothesis only when the graph can
    connect it to the preceding context.  A missing relation is therefore an
    unresolved causal gap: it blocks the current and downstream hard gates,
    raises uncertainty, and withholds the plan, but it does not invalidate the
    thesis or create contradictory evidence.
    """

    if (
        scene_graph is None
        or playbook
        not in {
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
        }
    ):
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
    broken_index: int | None = None
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
        if not scene_graph.find_path(
            left_sources,
            right.source_ids,
            max_depth=6,
        ):
            broken_index = index
            break
    if broken_index is None:
        return evaluation
    gates = dict(evaluation.hard_gate_results)
    for step_id in ordered_ids[broken_index:]:
        signal = signals[step_id]
        signals[step_id] = replace(signal, value=0.0)
        if step_id in gates:
            gates[step_id] = False
    return replace(
        evaluation,
        trigger_ready=False,
        plan=None,
        entry_readiness=0.0,
        delivery_quality=0.0,
        typed_uncertainty=max(0.75, evaluation.typed_uncertainty),
        sequence_signals=signals,
        hard_gate_results=gates,
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
    return bool(
        candidate_clock > terminal_at
        and (
            prior_identity is None
            or candidate_id != prior_identity
        )
    )


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
        prior_steps: dict[str, SequenceStepState] = {}
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
    if observation.liquidity_inventory_authoritative:
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
                item.lifecycle
                is LiquidityInventoryLifecycle.VISIBLE
                and item.confirmed_at <= observation.asof
            )
        ]
    consumed_ids = {
        source_id
        for event in observation.recent_events
        if event.kind
        in {
            EventKind.LIQUIDITY_SWEEP,
            EventKind.LIQUIDITY_CONSUMED,
            EventKind.STRUCTURE_BREAK,
        }
        for source_id in event.source_ids
    }
    output: list[LiquidityLevel] = []
    for timeframe in observation.active_timeframes:
        output.extend(
            level
            for level in observation.frame(timeframe).liquidity
            if (
                not level.swept
                and level.level_id not in consumed_ids
                and level.confirmed_at <= observation.asof
            )
        )
    return output




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
    payload = (
        hypothesis_key,
        evaluation.context_identity,
        None
        if evaluation.selected_draw is None
        else evaluation.selected_draw.level_id,
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
        setup_mismatch = bool(
            position.setup_id is not None
            and (
                sequence.setup_id != position.setup_id
                or evaluation.entry_location_id
                != position.entry_location_id
                or evaluation.entry_path_id != position.entry_path_id
            )
        )
        if setup_mismatch or evaluation.invalidated:
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
        evaluation.invalidated
        or evaluation.entry_window_expired
    ) and (
        prior_had_setup or current_has_setup
    ):
        return PlaybookPhase.INVALIDATED
    if plan is not None and observation.asof >= plan.deadline:
        return PlaybookPhase.INVALIDATED
    if not current_has_setup:
        return PlaybookPhase.INACTIVE
    if sequence.completed_steps == 1:
        return PlaybookPhase.FORMING
    if sequence.completed_steps == 2:
        return PlaybookPhase.ARMED
    if sequence.completed_steps == 3:
        return PlaybookPhase.WAITING_LOCATION
    if sequence.completed_steps < len(sequence.steps):
        return PlaybookPhase.WAITING_TRIGGER
    all_market_gates = bool(
        evaluation.hard_gate_results
        and all(evaluation.hard_gate_results.values())
    )
    delivery_ready = bool(
        plan is not None
        and plan.remaining_path_R
        >= config.minimum_remaining_path_R
        and evaluation.delivery_quality > 0.0
    )
    if (
        sequence.complete
        and evaluation.trigger_ready
        and all_market_gates
        and delivery_ready
        and thesis_strength > 0.0
    ):
        return PlaybookPhase.EXECUTABLE
    return PlaybookPhase.INVALIDATED


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
        and episode_deadline is not None
        and observation.asof >= episode_deadline
    ):
        reason = "episode_deadline_elapsed"
        sources = _identity_tuple(
            sequence.setup_id,
            evaluation.entry_location_id,
            evaluation.entry_path_id,
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
        reason = evaluation.terminal_reason
        sources = evaluation.terminal_source_ids
    elif plan is not None and observation.asof >= plan.deadline:
        reason = "deadline_elapsed"
        sources = _identity_tuple(
            plan.setup_id,
            plan.entry_location_id,
            plan.entry_path_id,
            plan.selected_draw_id,
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


class PlaybookBrain:
    """Updates six playbook-direction beliefs; it never chooses an action."""

    def __init__(
        self,
        config: BrainConfig | None = None,
        registry: PlaybookRegistry | None = None,
        calibrator: TypedBrainCalibrator | None = None,
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
        self._belief: MarketBelief | None = None

    @property
    def current(self) -> MarketBelief | None:
        return self._belief

    def reset(self) -> None:
        self._belief = None

    def update(
        self,
        observation: MarketObservation,
        *,
        position: PositionSnapshot | None = None,
        scene_graph: TemporalMarketSceneGraph | None = None,
        scene_delta: SceneGraphDelta | None = None,
    ) -> MarketBelief:
        hypotheses: dict[str, HypothesisBelief] = {}
        previous_belief = self._belief
        prior_hypotheses = (
            previous_belief.hypotheses
            if previous_belief is not None
            else {}
        )
        focus_state = (
            None
            if scene_graph is None
            else select_focus(
                previous_belief,
                observation,
                scene_delta,
                scene_graph,
            )
        )
        focused_scene = (
            None
            if scene_graph is None or focus_state is None
            else scene_graph.query(
                focus_state,
                context_ids=(
                    ()
                    if previous_belief is None
                    else tuple(
                        dict.fromkeys(
                            value
                            for hypothesis in previous_belief.hypotheses.values()
                            for value in (
                                hypothesis.context_id,
                                hypothesis.setup_context_id,
                                hypothesis.initiating_event_id,
                            )
                            if value is not None
                        )
                    )
                ),
                completed_only=True,
                ready_timeframes=tuple(
                    timeframe.value
                    for timeframe in observation.active_timeframes
                    if observation.frame(timeframe).ready
                ),
            )
        )
        for playbook in Playbook:
            protocol = self.registry.for_playbook(playbook)
            for direction in Direction:
                key = f"{playbook.value}:{direction.value}"
                prior = prior_hypotheses.get(key)
                context_id: str | None = None
                episode_id: str | None = None
                episode_deadline: pd.Timestamp | None = None
                initiating_event_id: str | None = None
                frozen_invalidation: StructuralLevel | None = None
                evidence_revision_id: str | None = None
                terminal_at: pd.Timestamp | None = None
                terminal_reason: str | None = None
                terminal_source_ids: tuple[str, ...] = ()
                raw_quality_dimensions: Mapping[str, float] = {}
                evaluation = _typed_evaluate(
                    playbook,
                    observation,
                    direction,
                    protocol,
                    prior,
                    self.config,
                )
                evaluation = _require_connected_graph_sequence(
                    playbook,
                    protocol,
                    evaluation,
                    scene_graph,
                    prior,
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
                ordered_gates = {
                    step.step_id: step.satisfied
                    for step in sequence.steps
                    if step.step_id in protocol.hard_gates
                }
                if set(ordered_gates) != set(protocol.hard_gates):
                    raise ValueError(
                        f"{playbook.value} sequence omitted a hard gate"
                    )
                if (
                    prior is not None
                    and prior.phase in _TERMINAL_PHASES
                    and not _terminal_has_new_setup(prior, sequence)
                ):
                    # The terminal belief is the immutable closure of the
                    # last episode.  It remains visible until a different,
                    # causally newer candidate supersedes it.
                    hypotheses[key] = prior
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
                    and prior.plan.entry_path_id
                    == position.entry_path_id
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
                    ordered_gates = {
                        step.step_id: step.satisfied
                        for step in sequence.steps
                        if step.step_id in protocol.hard_gates
                    }
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
                    elif position_matches and position is not None:
                        episode_deadline = position.deadline
                    else:
                        episode_deadline = observation.asof + pd.Timedelta(
                            minutes=max(
                                0,
                                observation.execution.minutes_to_deadline,
                            )
                        )
                evidence_revision_id = _typed_evidence_revision(
                    key,
                    protocol,
                    evaluation,
                )
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
                    position_matches_prior
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
                    plan = replace(
                        plan,
                        deadline=min(
                            plan.deadline,
                            prior.plan.deadline,
                            episode_deadline or plan.deadline,
                        ),
                    )
                elif (
                    plan is not None
                    and episode_deadline is not None
                    and plan.deadline > episode_deadline
                ):
                    plan = replace(plan, deadline=episode_deadline)
                parked = "parked" in protocol.status
                if parked:
                    # Parked playbooks remain fully observable for causal
                    # diagnostics but cannot expose an executable plan.
                    plan = None
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
                    sequence = prior.sequence
                    setup_id = sequence.setup_id
                    context_id = prior.context_id
                    episode_id = prior.episode_id
                    episode_deadline = prior.episode_deadline
                    entry_location_id = prior.entry_location_id
                    initiating_event_id = prior.initiating_event_id
                    frozen_invalidation = prior.invalidation
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
                        name: self.calibrator.apply(
                            playbook,
                            name,
                            value,
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
                hypotheses[key] = HypothesisBelief(
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
                    hard_gate_results=ordered_gates,
                    setup_context_id=sequence.setup_id,
                    entry_location_id=(
                        entry_location_id
                        if sequence.setup_id is not None
                        else None
                    ),
                    context_id=context_id,
                    episode_id=episode_id,
                    episode_deadline=episode_deadline,
                    initiating_event_id=initiating_event_id,
                    evidence_revision_id=evidence_revision_id,
                    terminal_at=terminal_at,
                    terminal_reason=terminal_reason,
                    terminal_source_ids=terminal_source_ids,
                    draw_selection=(
                        plan.draw_selection
                        if plan is not None
                        else evaluation.draw_selection
                    ),
                    liquidity_route=(
                        plan.liquidity_route
                        if plan is not None
                        else evaluation.liquidity_route
                    ),
                    raw_quality_dimensions=raw_quality_dimensions,
                )
        base_belief = MarketBelief(
            asof=observation.asof,
            hypotheses=hypotheses,
        )
        if scene_graph is not None and focus_state is not None:
            current_top = (
                base_belief.ranked()[0]
                if base_belief.hypotheses
                else None
            )
            prior_top = (
                previous_belief.ranked()[0]
                if previous_belief is not None
                and previous_belief.hypotheses
                else None
            )
            focus_state = supplement_focus_once(
                focus_state,
                prior_top,
                current_top,
                observation.active_timeframes,
            )
            focused_scene = scene_graph.query(
                focus_state,
                context_ids=tuple(
                    dict.fromkeys(
                        value
                        for hypothesis in hypotheses.values()
                        for value in (
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
            context_hypotheses = build_hypothesis_states(
                hypotheses,
                scene_graph,
                focused_scene,
            )
            dominant_id = None
            if current_top is not None:
                dominant_id = next(
                    (
                        identity
                        for identity, context in context_hypotheses.items()
                        if context.playbook is current_top.playbook
                        and context.direction is current_top.direction
                    ),
                    None,
                )
            competing_ids = tuple(
                identity
                for identity, context in context_hypotheses.items()
                if identity != dominant_id
                and hypotheses[
                    f"{context.playbook.value}:{context.direction.value}"
                ].phase
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
            dominant_conflict_paths = (
                set()
                if dominant_context is None
                else set(dominant_context.contradicting_graph_paths)
            )
            focused_conflict_ids = set(
                focused_scene.cross_scale_conflicts
            )
            dominant_conflict_ids = tuple(
                edge.edge_id
                for edge in focused_scene.edges
                if edge.edge_id in focused_conflict_ids
                and (
                    edge.source_node_id,
                    edge.relation.value,
                    edge.target_node_id,
                )
                in dominant_conflict_paths
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
            focus_triggers = list(focus_state.trigger_event_ids)
            focus_question = focus_state.question
            if dominant_conflict_ids:
                focus_reasons.append("cross_scale_conflict")
                focus_triggers.extend(dominant_conflict_ids)
                focus_question = (
                    "which scale owns the active structural conflict"
                )
            focus_state = replace(
                focus_state,
                resolution_status=focus_resolution,
                reason_codes=tuple(dict.fromkeys(focus_reasons)),
                trigger_event_ids=tuple(dict.fromkeys(focus_triggers)),
                question=focus_question,
                hypothesis_id=dominant_id,
                phase_at_selection=(
                    None if current_top is None else current_top.phase.value
                ),
                focus_revision_id="",
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
            )
        else:
            self._belief = base_belief
        return self._belief


__all__ = ["BrainConfig", "PlaybookBrain"]
