"""Independent structural risk engine and conservative bar execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .model import (
    AccountState,
    Action,
    Bar,
    BOSLifecycle,
    BOSScope,
    Decision,
    DealingRangeLifecycle,
    Direction,
    EntryLocationLifecycle,
    EntryLocationState,
    EventKind,
    FairValueGapLifecycle,
    FVGQualification,
    FrozenLSRContext,
    FrozenThesis,
    LiquidityInventoryLifecycle,
    LiquidityLevel,
    ManipulationLifecycle,
    MarketObservation,
    MicroBOSReference,
    OrderBlockLifecycle,
    PathSequenceLifecycle,
    Playbook,
    PositionSnapshot,
    QualifiedReacceptanceLifecycle,
    RiskAssessment,
    StructuralLevel,
    Timeframe,
    TradePlan,
    VetoCode,
    content_hash,
)


@dataclass(frozen=True)
class RiskLimits:
    maximum_trade_risk_fraction: float = 0.01
    maximum_total_risk_fraction: float = 0.02
    maximum_spread_ticks: float = 4.0
    maximum_cost_R: float = 0.20
    minimum_fillability: float = 0.45
    minimum_minutes_to_deadline: int = 5
    minimum_target_R: float = 1.0
    tick_size: float = 0.25


@dataclass(frozen=True)
class BarExecution:
    filled: bool
    closed: bool
    reason: str
    entry_price: float | None
    exit_price: float | None
    stop_touched: bool
    target_touched: bool
    ambiguous_same_bar: bool


def _freeze(plan: TradePlan, created_at) -> FrozenThesis:
    payload: dict[str, Any] = {
        "created_at": created_at,
        "playbook": plan.playbook,
        "direction": plan.direction,
        "entry": plan.planned_entry,
        "original_invalidation": plan.invalidation,
        "original_targets": plan.targets,
        "deadline": plan.deadline,
        "draw_selection": plan.draw_selection,
        "range_auction": plan.range_auction,
        "lsr_context": plan.lsr_context,
        "liquidity_route": plan.liquidity_route,
    }
    if plan.setup_id is not None:
        payload.update(
            {
                "setup_id": plan.setup_id,
                "entry_location_id": plan.entry_location_id,
                "entry_path_id": plan.entry_path_id,
            }
        )
    thesis_hash = content_hash(payload)
    return FrozenThesis(
        thesis_hash=thesis_hash,
        created_at=created_at,
        playbook=plan.playbook,
        direction=plan.direction,
        entry=plan.planned_entry,
        original_invalidation=plan.invalidation,
        original_targets=plan.targets,
        deadline=plan.deadline,
        setup_id=plan.setup_id,
        entry_location_id=plan.entry_location_id,
        entry_path_id=plan.entry_path_id,
        draw_selection=plan.draw_selection,
        range_auction=plan.range_auction,
        lsr_context=plan.lsr_context,
        liquidity_route=plan.liquidity_route,
    )


def _same_price(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def _entry_location_has_exact_zone_source(
    location: EntryLocationState,
    observation: MarketObservation,
) -> bool:
    """Prove that Group 5 retained the exact frozen M5 FVG/OB source."""

    frame = observation.frame(Timeframe.M5)
    if location.source_zone_kind == "fvg":
        matches = tuple(
            state
            for state in frame.fair_value_gaps
            if state.fvg_id == location.source_zone_id
        )
        if len(matches) != 1:
            return False
        source = matches[0]
        source_bos_id = None
        source_failed = (
            source.lifecycle is FairValueGapLifecycle.INVALIDATED
        )
    elif location.source_zone_kind == "order_block":
        matches = tuple(
            state
            for state in frame.order_blocks
            if state.order_block_id == location.source_zone_id
        )
        if len(matches) != 1:
            return False
        source = matches[0]
        source_bos_id = source.source_bos_id
        source_failed = source.lifecycle is OrderBlockLifecycle.FAILED
    else:
        return False
    return bool(
        not source_failed
        and source.protocol_hash
        == location.source_group3_protocol_hash
        == location.source_zone_protocol_hash
        and source.symbol == observation.symbol == location.symbol
        and source.instrument_id
        == observation.instrument_id
        == location.instrument_id
        and source.timeframe is Timeframe.M5
        and source.direction is location.direction
        and source.source_displacement_id
        == location.source_displacement_id
        and source_bos_id == location.source_bos_id
        and _same_price(source.lower_bound, location.lower_bound)
        and _same_price(source.upper_bound, location.upper_bound)
        and _same_price(source.midpoint, location.midpoint)
        and _same_price(
            source.invalidation_price,
            location.failure_boundary,
        )
        and source.confirmed_at == location.formed_at
    )


def _exact_entry_zone_source(
    location: EntryLocationState,
    observation: MarketObservation,
):
    if not _entry_location_has_exact_zone_source(location, observation):
        return None
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
    )
    return candidates[0] if len(candidates) == 1 else None


def _has_opposed_mss(
    observation: MarketObservation,
    direction: Direction,
    displacement_id: str,
    *,
    after: pd.Timestamp,
    before: pd.Timestamp,
) -> bool:
    return any(
        state.timeframe is Timeframe.M5
        and state.direction is direction
        and state.lifecycle is BOSLifecycle.CONFIRMED
        and state.scope is BOSScope.OPPOSED
        and state.mss_qualified
        and state.source_displacement_id == displacement_id
        and state.resolved_at is not None
        and state.resolved_at > after
        and state.resolved_at <= before
        for state in observation.frame(Timeframe.M5).structure_breaks
    )


def _micro_reference_has_exact_source(
    reference: MicroBOSReference,
    observation: MarketObservation,
) -> bool:
    """Prove that a Group 5 micro reference is the exact upstream M1 BOS."""

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


def _canonical_visible_levels(
    observation: MarketObservation,
) -> dict[str, LiquidityLevel]:
    candidates = [
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

    canonical: dict[str, LiquidityLevel] = {}
    ambiguous: set[str] = set()
    for level in candidates:
        if level.level_id in ambiguous:
            continue
        prior = canonical.get(level.level_id)
        if prior is None:
            canonical[level.level_id] = level
            continue
        if (
            prior.timeframe is not level.timeframe
            or prior.side != level.side
            or not _same_price(prior.price, level.price)
            or prior.formed_at != level.formed_at
            or prior.confirmed_at != level.confirmed_at
        ):
            canonical.pop(level.level_id, None)
            ambiguous.add(level.level_id)
    return canonical


def _visible_level_ids(observation: MarketObservation) -> set[str]:
    return set(_canonical_visible_levels(observation))


@dataclass(frozen=True)
class _CausalStopSource:
    price: float
    side: str
    timeframe: Timeframe
    observed_at: Any
    source_kind: str


def _canonical_stop_sources(
    observation: MarketObservation,
) -> dict[str, _CausalStopSource]:
    sources: dict[str, _CausalStopSource] = {}
    ambiguous: set[str] = set()

    def register(source_id: str, source: _CausalStopSource) -> None:
        if source_id in ambiguous:
            return
        prior = sources.get(source_id)
        if prior is None:
            sources[source_id] = source
            return
        if (
            prior.side != source.side
            or prior.timeframe is not source.timeframe
            or not _same_price(prior.price, source.price)
            or prior.observed_at != source.observed_at
            or prior.source_kind != source.source_kind
        ):
            sources.pop(source_id, None)
            ambiguous.add(source_id)

    for item in observation.liquidity_inventory:
        if (
            item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
            and item.kind == "swing"
            and item.confirmed_at <= observation.asof
        ):
            register(
                item.item_id,
                _CausalStopSource(
                    price=item.price,
                    side=item.side,
                    timeframe=item.timeframe,
                    observed_at=item.confirmed_at,
                    source_kind="swing",
                ),
            )

    for event in observation.recent_events:
        if (
            event.kind is EventKind.LIQUIDITY_SWEEP
            and event.observed_at <= observation.asof
            and event.side in {"above", "below"}
            and event.price is not None
        ):
            register(
                event.event_id,
                _CausalStopSource(
                    price=float(event.price),
                    side=event.side,
                    timeframe=event.timeframe,
                    observed_at=event.observed_at,
                    source_kind=event.kind.value,
                ),
            )
    for location in observation.entry_locations:
        register(
            location.location_id,
            _CausalStopSource(
                price=location.failure_boundary,
                side=location.direction.invalidation_side,
                timeframe=Timeframe.M5,
                observed_at=location.formed_at,
                source_kind="entry_zone_failure",
            ),
        )
    for state in observation.manipulations:
        if state.source_kind not in {
            "formed_liquidity_pool",
            "mature_range_boundary",
        }:
            continue
        direction = (
            Direction.SHORT
            if state.side == "above"
            else Direction.LONG
        )
        register(
            state.manipulation_id,
            _CausalStopSource(
                price=state.sweep_extreme,
                side=direction.invalidation_side,
                timeframe=Timeframe.M1,
                observed_at=state.swept_at,
                source_kind=(
                    "range_sweep_extreme"
                    if state.source_kind == "mature_range_boundary"
                    else "pool_sweep_extreme"
                ),
            ),
        )
    return sources


def _current_lsr_manipulation_matches(
    context: FrozenLSRContext,
    observation: MarketObservation,
) -> bool:
    """Validate a materialized parent without requiring bounded retention.

    The frozen Context is the decision-time custody record.  If Group5/Group4
    still exposes the exact manipulation, it must agree byte-for-byte on the
    causal fields; its absence alone is not a reason to reject a late child
    zone after the bounded diagnostic path was compacted.
    """

    matches = tuple(
        item
        for item in observation.manipulations
        if item.manipulation_id == context.manipulation_id
    )
    if not matches:
        return True
    if len(matches) != 1:
        return False
    manipulation = matches[0]
    lifecycle_clock_matches = bool(
        (
            manipulation.lifecycle is ManipulationLifecycle.REACCEPTED
            and manipulation.reaccepted_at == context.reaccepted_at
        )
        or (
            manipulation.lifecycle
            is ManipulationLifecycle.ACCEPTED_OUTSIDE
            and manipulation.accepted_outside_at is not None
            and manipulation.accepted_outside_at
            > context.displacement_observed_at
        )
    )
    return bool(
        manipulation.source_kind == "formed_liquidity_pool"
        and lifecycle_clock_matches
        and manipulation.protocol_hash == context.manipulation_protocol_hash
        and manipulation.source_id == context.source_pool_id
        and manipulation.swept_at == context.swept_at
        and _same_price(
            manipulation.sweep_extreme,
            context.sweep_extreme,
        )
        and manipulation.side
        == (
            "below"
            if context.direction is Direction.LONG
            else "above"
        )
    )


def _materialized_lsr_pool_path_matches(
    context: FrozenLSRContext,
    observation: MarketObservation,
) -> bool:
    """Validate an exact current pool path when it is still materialized."""

    matches = tuple(
        item
        for item in observation.path_sequences
        if item.sequence_id == context.pool_path_id
    )
    if not matches:
        return True
    if len(matches) != 1:
        return False
    path = matches[0]
    return_steps = tuple(
        step for step in path.steps if step.kind == "reacceptance_held"
    )
    displacement_steps = tuple(
        step for step in path.steps if step.kind == "opposite_displacement"
    )
    return bool(
        path.context_kind == "pool_reversal"
        and path.context_id == context.manipulation_id
        and path.direction is context.direction
        and path.protocol_hash == context.pool_path_protocol_hash
        and path.symbol == observation.symbol
        and path.instrument_id == observation.instrument_id
        # The Context freezes the causal reacceptance/displacement prefix.
        # Later diagnostic resolution steps cannot retrospectively erase it;
        # only a censored/source-invalidated path loses provenance custody.
        and path.lifecycle is not PathSequenceLifecycle.CENSORED
        and len(return_steps) == 1
        and len(displacement_steps) == 1
        and return_steps[0].source_event_id == context.manipulation_id
        and return_steps[0].source_entity_id == context.manipulation_id
        and return_steps[0].observed_at == context.reaccepted_at
        and displacement_steps[0].source_event_id
        == context.manipulation_id
        and displacement_steps[0].source_entity_id == context.displacement_id
        and displacement_steps[0].source_active_at
        == context.displacement_active_at
        and displacement_steps[0].observed_at
        == context.displacement_observed_at
    )


def _valid_lsr_trigger_family(
    path,
    location: EntryLocationState,
    first_pullback,
    plan: TradePlan,
    observation: MarketObservation,
) -> bool:
    """Validate the registered zone-bound LSR trigger OR family."""

    expected_trigger = {
        "zone_rejection_observed": "wick_rejection",
        "qualified_reacceptance_held": "reacceptance_held",
        "micro_bos_aligned": "micro_bos_confirmed",
    }.get(path.transition_reason)
    if expected_trigger is None:
        return False
    trigger_steps = tuple(
        step for step in path.steps if step.kind == expected_trigger
    )
    if not trigger_steps:
        return False
    trigger = trigger_steps[-1]
    if expected_trigger == "wick_rejection":
        return bool(
            path.lifecycle is PathSequenceLifecycle.ACTIVE
            and trigger.source_entity_id == location.location_id
            and trigger.observed_at > first_pullback.observed_at
            and location.rejected_at == trigger.observed_at
        )
    if expected_trigger == "reacceptance_held":
        return bool(
            path.lifecycle is PathSequenceLifecycle.ACTIVE
            and trigger.observed_at > first_pullback.observed_at
            and any(
                state.context_kind == "entry_zone"
                and state.context_id == location.location_id
                and state.reacceptance_id == trigger.source_entity_id
                and state.direction is plan.direction
                and state.lifecycle is QualifiedReacceptanceLifecycle.HELD
                and state.held_at == trigger.observed_at
                for state in observation.qualified_reacceptances
            )
        )
    return bool(
        path.lifecycle is PathSequenceLifecycle.CLOSED
        and path.ended_at == observation.asof
        and trigger.observed_at > first_pullback.observed_at
        and any(
            reference.context_kind == "zone_return"
            and reference.context_id == location.location_id
            and reference.expected_direction is plan.direction
            and reference.qualified
            and reference.bos_id == trigger.source_event_id
            and reference.target_swing_id == trigger.source_entity_id
            and reference.resolved_at == trigger.observed_at
            and _micro_reference_has_exact_source(reference, observation)
            for reference in observation.micro_bos_references
        )
    )


def _valid_typed_entry_location(
    plan: TradePlan,
    observation: MarketObservation,
) -> bool:
    typed = (
        plan.setup_id,
        plan.entry_location_id,
        plan.entry_path_id,
        plan.entry_zone_lower,
        plan.entry_zone_upper,
        plan.selected_draw_id,
    )
    if any(value is None for value in typed):
        return False
    location = next(
        (
            item
            for item in observation.entry_locations
            if item.location_id == plan.entry_location_id
        ),
        None,
    )
    path = next(
        (
            item
            for item in observation.path_sequences
            if item.sequence_id == plan.entry_path_id
        ),
        None,
    )
    origin = None if path is None else path.steps[0]
    pullback_steps = (
        ()
        if path is None
        else tuple(
            step for step in path.steps if step.kind == "first_pullback"
        )
    )
    if (
        location is None
        or path is None
        or not _entry_location_has_exact_zone_source(
            location,
            observation,
        )
        or path.context_kind != "zone_return"
        or path.context_id != location.location_id
        or path.direction is not plan.direction
        or path.protocol_hash != location.protocol_hash
        or path.symbol != observation.symbol
        or path.instrument_id != observation.instrument_id
        or path.formed_at != location.formed_at
        or location.direction is not plan.direction
        or location.lifecycle is EntryLocationLifecycle.LEFT
        or origin is None
        or origin.kind != "zone_visible"
        or origin.observed_at != location.formed_at
        or origin.source_event_id != location.source_zone_id
        or origin.source_entity_id != location.source_zone_id
        or len(pullback_steps) != 1
        or location.first_entered_at is None
        or pullback_steps[0].observed_at
        != location.first_entered_at
        or pullback_steps[0].source_event_id is not None
        or pullback_steps[0].source_entity_id
        != location.location_id
        or not _same_price(
            location.lower_bound,
            float(plan.entry_zone_lower),
        )
        or not _same_price(
            location.upper_bound,
            float(plan.entry_zone_upper),
        )
        or not (
            location.lower_bound
            <= plan.planned_entry
            <= location.upper_bound
        )
        or plan.selected_draw_id != plan.targets[0].level_id
    ):
        return False
    first_pullback = pullback_steps[0]
    step_kinds = {step.kind for step in path.steps}
    if plan.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        expected_trigger = {
            "zone_rejection_observed": "wick_rejection",
            "qualified_reacceptance_held": "reacceptance_held",
            "micro_bos_aligned": "micro_bos_confirmed",
        }.get(path.transition_reason)
        trigger_steps = [
            step
            for step in path.steps
            if step.kind == expected_trigger
        ]
        if (
            plan.setup_id != plan.entry_path_id
            or expected_trigger is None
            or not trigger_steps
        ):
            return False
        trigger_step = trigger_steps[-1]
        if expected_trigger == "wick_rejection":
            return bool(
                path.lifecycle is PathSequenceLifecycle.ACTIVE
                and trigger_step.source_entity_id == location.location_id
                and location.rejected_at == trigger_step.observed_at
            )
        if expected_trigger == "reacceptance_held":
            return bool(
                path.lifecycle is PathSequenceLifecycle.ACTIVE
                and any(
                    state.context_kind == "entry_zone"
                    and state.context_id == location.location_id
                    and state.reacceptance_id
                    == trigger_step.source_entity_id
                    and state.lifecycle
                    is QualifiedReacceptanceLifecycle.HELD
                    and state.held_at == trigger_step.observed_at
                    for state in observation.qualified_reacceptances
                )
            )
        return bool(
            path.lifecycle is PathSequenceLifecycle.CLOSED
            and path.ended_at == observation.asof
            and any(
                reference.context_kind == "zone_return"
                and reference.context_id == location.location_id
                and reference.expected_direction is plan.direction
                and reference.qualified
                and reference.bos_id == trigger_step.source_event_id
                and reference.target_swing_id
                == trigger_step.source_entity_id
                and reference.resolved_at == trigger_step.observed_at
                and _micro_reference_has_exact_source(
                    reference,
                    observation,
                )
                for reference in observation.micro_bos_references
            )
        )
    if plan.playbook is Playbook.FAILED_AUCTION_VALUE_RETURN:
        context = plan.range_auction
        if context is None or plan.setup_id != context.manipulation_id:
            return False
        dealing_range = next(
            (
                state
                for state in observation.frame(Timeframe.H1).dealing_ranges
                if state.range_id == context.range_id
            ),
            None,
        )
        manipulation = next(
            (
                state
                for state in observation.manipulations
                if state.manipulation_id == context.manipulation_id
            ),
            None,
        )
        zone_source = _exact_entry_zone_source(location, observation)
        if (
            dealing_range is None
            or dealing_range.lifecycle is not DealingRangeLifecycle.MATURE
            or manipulation is None
            or manipulation.source_kind != "mature_range_boundary"
            or manipulation.source_id != dealing_range.range_id
            or manipulation.lifecycle is not ManipulationLifecycle.REACCEPTED
            or manipulation.reaccepted_at is None
            or manipulation.reentry_price is None
            or manipulation.reentry_candidate_at is None
            or zone_source is None
            or zone_source.source_displacement_active_at is None
            or zone_source.source_displacement_active_at
            <= manipulation.reaccepted_at
            or not _has_opposed_mss(
                observation,
                plan.direction,
                location.source_displacement_id,
                after=manipulation.reaccepted_at,
                before=first_pullback.observed_at,
            )
            or dealing_range.mature_at != context.mature_at
            or not _same_price(
                dealing_range.lower_bound,
                context.lower_bound,
            )
            or not _same_price(
                dealing_range.upper_bound,
                context.upper_bound,
            )
            or not _same_price(dealing_range.midpoint, context.midpoint)
            or not _same_price(
                dealing_range.value_price,
                context.value_price,
            )
            or manipulation.side != context.manipulation_side
            or manipulation.swept_at != context.swept_at
            or manipulation.reaccepted_at != context.reentered_at
            or manipulation.reentry_candidate_at
            != context.reentry_candidate_at
            or not _same_price(
                manipulation.sweep_extreme,
                context.manipulation_extreme,
            )
            or not _same_price(
                manipulation.reentry_price,
                context.reentry_price,
            )
            or location.formed_at <= context.reentered_at
            or location.lower_bound < context.lower_bound
            or location.upper_bound > context.upper_bound
            or (
                plan.direction is Direction.LONG
                and (
                    location.midpoint >= context.midpoint
                    or plan.planned_entry >= context.midpoint
                )
            )
            or (
                plan.direction is Direction.SHORT
                and (
                    location.midpoint <= context.midpoint
                    or plan.planned_entry <= context.midpoint
                )
            )
        ):
            return False
        if path.transition_reason == "qualified_reacceptance_held":
            held_steps = tuple(
                step
                for step in path.steps
                if step.kind == "reacceptance_held"
            )
            return bool(
                path.lifecycle is PathSequenceLifecycle.ACTIVE
                and held_steps
                and any(
                    state.context_kind == "entry_zone"
                    and state.context_id == location.location_id
                    and state.lifecycle
                    is QualifiedReacceptanceLifecycle.HELD
                    and any(
                        step.source_entity_id == state.reacceptance_id
                        and step.observed_at == state.held_at
                        for step in held_steps
                    )
                    for state in observation.qualified_reacceptances
                )
            )
        if path.transition_reason != "micro_bos_aligned":
            return False
        micro_steps = tuple(
            step
            for step in path.steps
            if step.kind == "micro_bos_confirmed"
        )
        return bool(
            path.lifecycle is PathSequenceLifecycle.CLOSED
            and path.ended_at == observation.asof
            and any(
                reference.context_kind == "zone_return"
                and reference.context_id == location.location_id
                and reference.expected_direction is plan.direction
                and reference.qualified
                and reference.resolved_at > first_pullback.observed_at
                and any(
                    step.source_event_id == reference.bos_id
                    and step.source_entity_id == reference.target_swing_id
                    and step.observed_at == reference.resolved_at
                    for step in micro_steps
                )
                and _micro_reference_has_exact_source(
                    reference,
                    observation,
                )
                for reference in observation.micro_bos_references
            )
        )
    if plan.playbook is not Playbook.LIQUIDITY_SWEEP_REVERSAL:
        return False
    context = plan.lsr_context
    zone_source = _exact_entry_zone_source(location, observation)
    if (
        context is None
        or zone_source is None
        or plan.setup_id
        != context.entry_episode_id(location.source_zone_id)
        or context.direction is not plan.direction
        or location.source_displacement_id != context.displacement_id
        or zone_source.source_displacement_id != context.displacement_id
        or zone_source.source_displacement_active_at
        != context.displacement_active_at
        or location.formed_at < context.displacement_observed_at
        or first_pullback.observed_at < location.formed_at
        or path.protocol_hash != context.pool_path_protocol_hash
        or (
            location.source_zone_kind == "fvg"
            and zone_source.qualification
            is not FVGQualification.DISPLACEMENT_LINKED
        )
        or not _current_lsr_manipulation_matches(context, observation)
        or not _materialized_lsr_pool_path_matches(context, observation)
    ):
        return False
    return _valid_lsr_trigger_family(
        path,
        location,
        first_pullback,
        plan,
        observation,
    )


def _valid_plan_arithmetic(
    plan: TradePlan,
    observation: MarketObservation,
) -> bool:
    risk_points = abs(plan.planned_entry - plan.invalidation.price)
    if risk_points <= 0.0 or not _same_price(
        risk_points,
        plan.risk_points,
    ):
        return False
    primary_R = (
        abs(plan.targets[0].price - plan.planned_entry) / risk_points
    )
    if not _same_price(primary_R, plan.primary_target_R):
        return False
    remaining_R = (
        max(
            0.0,
            plan.direction.sign
            * (plan.targets[0].price - observation.price),
        )
        / risk_points
    )
    if not _same_price(remaining_R, plan.remaining_path_R):
        return False
    hard_deadline = observation.asof + pd.Timedelta(
        minutes=observation.execution.minutes_to_deadline
    )
    return observation.asof < plan.deadline <= hard_deadline


def _valid_stop(
    plan: TradePlan,
    observation: MarketObservation,
) -> bool:
    stop = plan.invalidation
    if stop.observed_at > observation.asof or not stop.source_level_id:
        return False
    # LSR freezes the exact parent manipulation and sweep extreme in the
    # plan.  Bounded Scene-Graph/Group5 retention may legitimately remove the
    # materialized manipulation before a later child becomes executable, so
    # absence alone cannot erase that already causal stop.  When the parent is
    # still materialized, the helpers below continue to require an exact,
    # unambiguous match; no fallback stop or relaxed geometry is introduced.
    if plan.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
        context = plan.lsr_context
        if (
            context is None
            or plan.entry_location_id is None
            or stop.source_level_id != context.manipulation_id
            or stop.observed_at != context.swept_at
            or not _same_price(stop.price, context.sweep_extreme)
            or not _current_lsr_manipulation_matches(context, observation)
            or not _materialized_lsr_pool_path_matches(context, observation)
        ):
            return False
        if plan.direction is Direction.LONG:
            return stop.side == "below" and stop.price < plan.planned_entry
        return stop.side == "above" and stop.price > plan.planned_entry
    source = _canonical_stop_sources(observation).get(stop.source_level_id)
    if source is None:
        return False
    if (
        stop.side != source.side
        or not _same_price(stop.price, source.price)
        or stop.observed_at != source.observed_at
    ):
        return False
    if plan.entry_location_id is not None:
        if plan.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
            if (
                plan.setup_id != plan.entry_path_id
                or stop.source_level_id != plan.entry_location_id
            ):
                return False
        elif plan.playbook is Playbook.FAILED_AUCTION_VALUE_RETURN:
            context = plan.range_auction
            manipulation = next(
                (
                    item
                    for item in observation.manipulations
                    if (
                        context is not None
                        and item.manipulation_id
                        == context.manipulation_id
                        and item.source_kind
                        == "mature_range_boundary"
                    )
                ),
                None,
            )
            if (
                context is None
                or manipulation is None
                or plan.setup_id != manipulation.manipulation_id
                or stop.source_level_id != manipulation.manipulation_id
                or stop.observed_at != manipulation.swept_at
                or not _same_price(
                    stop.price,
                    manipulation.sweep_extreme,
                )
                or not _same_price(
                    stop.price,
                    context.manipulation_extreme,
                )
            ):
                return False
    if plan.direction is Direction.LONG:
        return stop.side == "below" and stop.price < plan.planned_entry
    return stop.side == "above" and stop.price > plan.planned_entry


def _valid_targets(
    plan: TradePlan,
    observation: MarketObservation,
    *,
    tick_size: float = 0.25,
) -> bool:
    route = plan.liquidity_route
    visible = _canonical_visible_levels(observation)
    conservative_contact_basis = bool(
        route is not None
        and route.primary_target_price_basis == "conservative_contact"
    )
    if (
        conservative_contact_basis
        and plan.playbook is not Playbook.LIQUIDITY_SWEEP_REVERSAL
    ):
        return False
    contact_prices: dict[str, float] = {}
    ambiguous_contact_ids: set[str] = set()
    if conservative_contact_basis:
        for item in observation.liquidity_inventory:
            if (
                item.lifecycle is not LiquidityInventoryLifecycle.VISIBLE
                or item.confirmed_at > observation.asof
                or item.side != plan.direction.opposing_liquidity_side
                or item.item_id in ambiguous_contact_ids
            ):
                continue
            contact = float(
                item.lower_bound
                if plan.direction is Direction.LONG
                else item.upper_bound
            )
            prior = contact_prices.get(item.item_id)
            if prior is None:
                contact_prices[item.item_id] = contact
            elif not _same_price(prior, contact):
                contact_prices.pop(item.item_id, None)
                ambiguous_contact_ids.add(item.item_id)

    def target_price(level_id: str, inventory_price: float) -> float | None:
        if not conservative_contact_basis:
            return inventory_price
        return contact_prices.get(level_id)

    if route is not None:
        target = plan.targets[0]
        barrier_price = route.authority_barrier_price
        if (
            route.primary_deliverable_target_id != target.level_id
            or route.context_draw_id is None
            or route.terminal_draw_id is None
            or target.level_id not in route.source_path_ids
            or route.context_draw_id not in route.source_path_ids
            or route.terminal_draw_id not in route.source_path_ids
            # Brain freezes only price-geometric hard barriers here.  A plan
            # crossing one cannot be approved; ordinary nearer liquidity is
            # a waypoint/draw and is intentionally not inferred as a blocker.
            or route.path_blocker_ids
            or route.context_draw_id not in visible
            or route.terminal_draw_id not in visible
            or (
                barrier_price is not None
                and (
                    (
                        plan.direction is Direction.LONG
                        and not (
                            plan.planned_entry
                            < target.price
                            and target.price + tick_size
                            < float(barrier_price)
                        )
                    )
                    or (
                        plan.direction is Direction.SHORT
                        and not (
                            plan.planned_entry
                            > target.price
                            and target.price - tick_size
                            > float(barrier_price)
                        )
                    )
                )
            )
        ):
            return False
    if plan.playbook is Playbook.FAILED_AUCTION_VALUE_RETURN:
        context = plan.range_auction
        if (
            context is None
            or len(plan.targets) != 1
            or plan.selected_draw_id
            != context.opposite_liquidity_id
            or plan.draw_selection is None
            or plan.draw_selection.draw_id != plan.selected_draw_id
        ):
            return False
        item = next(
            (
                candidate
                for candidate in observation.liquidity_inventory
                if candidate.item_id == context.opposite_liquidity_id
            ),
            None,
        )
        target = plan.targets[0]
        expected_side = (
            "above" if plan.direction is Direction.LONG else "below"
        )
        expected_price = (
            context.upper_bound
            if plan.direction is Direction.LONG
            else context.lower_bound
        )
        return bool(
            item is not None
            and item.kind == "range_boundary"
            and item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
            and item.side == expected_side
            and context.range_id in item.source_ids
            and item.confirmed_at <= observation.asof
            and target.level_id == item.item_id
            and target.timeframe is item.timeframe
            and target.side == item.side
            and _same_price(target.price, item.price)
            and _same_price(target.price, expected_price)
            and target.formed_at == item.formed_at
            and target.confirmed_at == item.confirmed_at
            and not target.swept
            and (
                target.price > plan.planned_entry
                if plan.direction is Direction.LONG
                else target.price < plan.planned_entry
            )
        )
    if (
        route is None
        and
        plan.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
        and plan.setup_id is not None
        and plan.targets[0].timeframe is not Timeframe.H4
    ):
        return False
    for target in plan.targets:
        source = visible.get(target.level_id)
        if source is None:
            return False
        expected_price = (
            target_price(target.level_id, source.price)
        )
        if (
            expected_price is None
            or target.timeframe is not source.timeframe
            or target.side != source.side
            or not _same_price(target.price, expected_price)
            or target.formed_at != source.formed_at
            or target.confirmed_at != source.confirmed_at
            or target.confirmed_at > observation.asof
            or target.swept
        ):
            return False
        if plan.direction is Direction.LONG and target.price <= plan.planned_entry:
            return False
        if plan.direction is Direction.SHORT and target.price >= plan.planned_entry:
            return False
    return bool(plan.targets)


def protection_tightens(
    direction: Direction,
    current_stop: float,
    candidate: StructuralLevel,
    current_price: float,
) -> bool:
    if direction is Direction.LONG:
        return current_stop < candidate.price < current_price
    return current_price < candidate.price < current_stop


def causal_protection_candidate(
    position: PositionSnapshot,
    observation: MarketObservation,
) -> StructuralLevel | None:
    """Return the tightest currently visible post-entry structural stop."""

    consumed = {
        source_id
        for event in observation.recent_events
        for source_id in event.source_ids
    }
    inventory_levels = [
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
            and item.kind == "swing"
        )
    ]
    candidates = [
        level
        for level in inventory_levels
        if (
            not level.swept
            and level.level_id not in consumed
            and position.opened_at < level.confirmed_at <= observation.asof
            and level.side == position.direction.invalidation_side
            and protection_tightens(
                position.direction,
                position.current_stop,
                StructuralLevel(
                    price=level.price,
                    side=level.side,
                    source_level_id=level.level_id,
                    observed_at=level.confirmed_at,
                    rationale="post-entry confirmed structure",
                ),
                observation.price,
            )
        )
    ]
    if not candidates:
        return None
    source = (
        max(candidates, key=lambda level: level.price)
        if position.direction is Direction.LONG
        else min(candidates, key=lambda level: level.price)
    )
    return StructuralLevel(
        price=source.price,
        side=source.side,
        source_level_id=source.level_id,
        observed_at=source.confirmed_at,
        rationale="current causal post-entry structure eligible for protection",
    )


class StructuralRiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def _forced_position_exit(
        self,
        decision: Decision,
        observation: MarketObservation,
        account: AccountState,
    ) -> RiskAssessment | None:
        position = account.position
        if position is None:
            return None
        frozen_deadline = position.deadline
        current_execution_deadline = observation.asof + pd.Timedelta(
            minutes=observation.execution.minutes_to_deadline
        )
        execution_deadline_conflict = bool(
            current_execution_deadline < frozen_deadline
        )
        stop_breached = (
            position.direction is Direction.LONG and observation.price <= position.current_stop
        ) or (
            position.direction is Direction.SHORT and observation.price >= position.current_stop
        )
        deadline_elapsed = observation.asof >= frozen_deadline
        severe_data = (
            "stale_market_data" in observation.anomalies
            or "contract_change_history_reset" in observation.anomalies
            or "data_gap_history_reset" in observation.anomalies
            or "data_anomaly" in observation.anomalies
            or "tick_size_mismatch" in observation.anomalies
            or position.symbol != observation.symbol
            or position.instrument_id != observation.instrument_id
            or any(name.startswith("clock_") for name in observation.anomalies)
        )
        if not (
            stop_breached
            or execution_deadline_conflict
            or deadline_elapsed
            or severe_data
        ):
            return None
        reasons: list[str] = []
        vetoes: list[VetoCode] = []
        if stop_breached:
            reasons.append("current structural stop is breached")
            vetoes.append(VetoCode.INVALID_STOP)
        if execution_deadline_conflict:
            reasons.append(
                "current execution deadline precedes the frozen "
                "position deadline"
            )
            vetoes.append(VetoCode.DEADLINE)
            vetoes.append(VetoCode.DATA_ANOMALY)
        elif deadline_elapsed:
            reasons.append("hard position deadline elapsed")
            vetoes.append(VetoCode.DEADLINE)
        if severe_data:
            reasons.append("market data is stale or clock-invalid")
            vetoes.append(VetoCode.DATA_ANOMALY)
        return RiskAssessment(
            requested_action=decision.selected_action,
            final_action=Action.EXIT,
            passed=False,
            vetoes=tuple(dict.fromkeys(vetoes)),
            reasons=tuple(reasons),
        )

    def review(
        self,
        decision: Decision,
        observation: MarketObservation,
        account: AccountState | None = None,
    ) -> RiskAssessment:
        account = account or AccountState(equity=100_000.0)
        forced = self._forced_position_exit(decision, observation, account)
        if forced is not None:
            return forced
        action = decision.selected_action
        if action is Action.ABSTAIN:
            return RiskAssessment(action, action, True, (), tuple(decision.reasons))
        if action in {Action.WAIT, Action.HOLD, Action.EXIT}:
            return RiskAssessment(action, action, True, (), tuple(decision.reasons))
        if action is Action.PROTECT:
            position = account.position
            candidate = (
                None
                if position is None
                else causal_protection_candidate(position, observation)
            )
            if (
                position is None
                or candidate is None
                or candidate.observed_at > observation.asof
                or candidate.side != position.direction.invalidation_side
                or candidate.source_level_id
                not in _canonical_stop_sources(observation)
                or not protection_tightens(
                    position.direction,
                    position.current_stop,
                    candidate,
                    observation.price,
                )
            ):
                return RiskAssessment(
                    action,
                    Action.ABSTAIN,
                    False,
                    (VetoCode.PROTECTION_NOT_TIGHTER,),
                    ("protection must use confirmed structure and can only tighten risk",),
                )
            return RiskAssessment(
                action,
                action,
                True,
                (),
                ("protection accepted without changing original invalidation",),
                protected_stop=candidate.price,
            )

        plan = decision.plan
        vetoes: list[VetoCode] = []
        reasons: list[str] = []
        if plan is None:
            vetoes.append(VetoCode.NO_PLAN)
            reasons.append("enter has no complete structural plan")
        else:
            execution = observation.execution
            structural_risk = abs(
                plan.planned_entry - plan.invalidation.price
            )
            spread_ticks = execution.spread_points / self.limits.tick_size
            cost_R = (
                execution.expected_round_trip_cost_points
                / max(structural_risk, 1e-12)
            )
            critical_anomalies = {
                "stale_market_data",
                "spread_missing_used_one_tick",
                "execution_constant_assumption",
                "deadline_missing",
                "deadline_elapsed",
                "contract_change_history_reset",
                "data_gap_history_reset",
                "data_anomaly",
                "tick_size_mismatch",
                "insufficient_top_of_book_depth",
                "group4_atr_unready_sweep",
            }
            combined_anomalies = (
                tuple(observation.anomalies)
                + tuple(execution.anomalies)
            )
            if any(
                anomaly.startswith("clock_") or anomaly in critical_anomalies
                for anomaly in combined_anomalies
            ) or (
                execution.source in {"missing", "unknown"}
                or execution.source.startswith("constant")
            ):
                vetoes.append(VetoCode.DATA_ANOMALY)
                reasons.append(
                    "data anomaly or non-observed execution input has "
                    "hard veto authority"
                )
            if execution.data_age_seconds > 60:
                vetoes.append(VetoCode.STALE_DATA)
                reasons.append("market data is stale")
            if spread_ticks > self.limits.maximum_spread_ticks:
                vetoes.append(VetoCode.SPREAD)
                reasons.append(
                    f"spread {spread_ticks:.1f} ticks exceeds "
                    f"{self.limits.maximum_spread_ticks:.1f}"
                )
            if cost_R > self.limits.maximum_cost_R:
                vetoes.append(VetoCode.COST)
                reasons.append(
                    f"round-trip cost {cost_R:.3f}R exceeds {self.limits.maximum_cost_R:.3f}R"
                )
            if execution.minutes_to_deadline < self.limits.minimum_minutes_to_deadline:
                vetoes.append(VetoCode.DEADLINE)
                reasons.append("insufficient time remains before hard deadline")
            frozen_minutes_to_deadline = (
                plan.deadline - observation.asof
            ).total_seconds() / 60.0
            if (
                frozen_minutes_to_deadline <= 1.0
                or frozen_minutes_to_deadline
                < self.limits.minimum_minutes_to_deadline
            ):
                if VetoCode.DEADLINE not in vetoes:
                    vetoes.append(VetoCode.DEADLINE)
                reasons.append(
                    "frozen thesis deadline leaves insufficient time for "
                    "a causally completed entry bar"
                )
            if "deadline_missing" in observation.anomalies:
                vetoes.append(VetoCode.DEADLINE)
                reasons.append("execution deadline is missing")
            if execution.fillability < self.limits.minimum_fillability:
                vetoes.append(VetoCode.FILLABILITY)
                reasons.append("estimated fillability is below the hard floor")
            actual_trade_risk = (
                structural_risk
                * max(1, int(account.quantity))
                * float(account.point_value)
                / max(float(account.equity), 1.0)
            )
            effective_trade_risk = max(
                float(account.requested_risk_fraction),
                float(actual_trade_risk),
            )
            if (
                effective_trade_risk > self.limits.maximum_trade_risk_fraction
                or account.open_risk_fraction + effective_trade_risk
                > self.limits.maximum_total_risk_fraction
            ):
                vetoes.append(VetoCode.ACCOUNT_RISK)
                reasons.append(
                    "requested, actual contract, or aggregate account risk exceeds its hard cap"
                )
            if not _valid_stop(plan, observation):
                vetoes.append(VetoCode.INVALID_STOP)
                reasons.append("stop is not a causally observed structural invalidation")
            if not _valid_plan_arithmetic(plan, observation):
                vetoes.append(VetoCode.NO_PLAN)
                reasons.append(
                    "plan risk, reward or deadline disagrees with frozen "
                    "structural prices and execution time"
                )
            if not _valid_typed_entry_location(plan, observation):
                vetoes.append(VetoCode.NO_PLAN)
                reasons.append(
                    "planned entry is not bound to its frozen Group 5 zone"
                )
            if not _valid_targets(
                plan,
                observation,
                tick_size=self.limits.tick_size,
            ):
                vetoes.append(VetoCode.INVALID_TARGET)
                reasons.append("target is not in current causal liquidity inventory")
            actual_primary_R = (
                abs(plan.targets[0].price - plan.planned_entry)
                / max(structural_risk, 1e-12)
            )
            if actual_primary_R < self.limits.minimum_target_R:
                vetoes.append(VetoCode.REWARD_RISK)
                reasons.append(
                    f"visible target offers {actual_primary_R:.3f}R, below "
                    f"{self.limits.minimum_target_R:.3f}R"
                )
        if vetoes:
            return RiskAssessment(
                requested_action=action,
                final_action=Action.ABSTAIN,
                passed=False,
                vetoes=tuple(dict.fromkeys(vetoes)),
                reasons=tuple(reasons),
            )
        assert plan is not None
        thesis = _freeze(plan, observation.asof)
        return RiskAssessment(
            requested_action=action,
            final_action=Action.ENTER,
            passed=True,
            vetoes=(),
            reasons=("entry passed every independent hard-risk gate",),
            frozen_thesis=thesis,
        )


def conservative_position_bar(
    direction: Direction,
    bar: Bar,
    *,
    current_stop: float,
    target: float,
) -> BarExecution:
    if direction is Direction.LONG:
        stop_touched = bar.low <= current_stop
        target_touched = bar.high >= target
        stop_fill = min(float(current_stop), float(bar.open))
    else:
        stop_touched = bar.high >= current_stop
        target_touched = bar.low <= target
        stop_fill = max(float(current_stop), float(bar.open))
    ambiguous = bool(stop_touched and target_touched)
    if stop_touched:
        return BarExecution(
            filled=True,
            closed=True,
            reason="same_bar_ambiguous_stop_first" if ambiguous else "stop",
            entry_price=None,
            exit_price=stop_fill,
            stop_touched=True,
            target_touched=target_touched,
            ambiguous_same_bar=ambiguous,
        )
    if target_touched:
        return BarExecution(
            filled=True,
            closed=True,
            reason="target",
            entry_price=None,
            exit_price=float(target),
            stop_touched=False,
            target_touched=True,
            ambiguous_same_bar=False,
        )
    return BarExecution(True, False, "open", None, None, False, False, False)


def conservative_entry_bar(plan: TradePlan, bar: Bar) -> BarExecution:
    """Conservative limit-style fill and stop/target precedence in one bar."""

    if plan.direction is Direction.LONG:
        filled = bar.low <= plan.planned_entry
    else:
        filled = bar.high >= plan.planned_entry
    if not filled:
        return BarExecution(False, False, "not_filled", None, None, False, False, False)
    position = conservative_position_bar(
        plan.direction,
        bar,
        current_stop=plan.invalidation.price,
        target=plan.targets[0].price,
    )
    # The order between entry and a favorable target touch is unknowable from
    # OHLC. A same-bar target is therefore never credited. An adverse stop
    # remains chargeable under the conservative execution rule.
    if position.target_touched and not position.stop_touched:
        return BarExecution(
            filled=True,
            closed=False,
            reason="filled_target_touch_not_credited_same_bar",
            entry_price=float(plan.planned_entry),
            exit_price=None,
            stop_touched=False,
            target_touched=True,
            ambiguous_same_bar=False,
        )
    return BarExecution(
        filled=True,
        closed=position.closed,
        reason=position.reason,
        entry_price=float(plan.planned_entry),
        exit_price=position.exit_price,
        stop_touched=position.stop_touched,
        target_touched=position.target_touched,
        ambiguous_same_bar=position.ambiguous_same_bar,
    )


__all__ = [
    "BarExecution",
    "RiskLimits",
    "StructuralRiskEngine",
    "causal_protection_candidate",
    "conservative_entry_bar",
    "conservative_position_bar",
    "protection_tightens",
]
