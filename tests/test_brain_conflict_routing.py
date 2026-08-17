from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_trader.model import (
    Action,
    ActionUtility,
    AuthorityLayer,
    Decision,
    DeliveryObstruction,
    Direction,
    DirectionalObstructionView,
    GlobalConflictEvidence,
    GlobalConflictRole,
    GlobalMarketContext,
    HypothesisConflictRelation,
    HypothesisSequenceState,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    MarketMode,
    OpenMarketThesis,
    Playbook,
    PlaybookPhase,
    PositionSnapshot,
    ScaleRelation,
    ScaleRelationState,
    SequenceStepState,
    StructuralLevel,
    Timeframe,
)
import smc_trader.playbooks as playbooks_module
from smc_trader.playbooks import (
    BrainConfig,
    _Evaluation,
    _SequenceSignal,
    _bind_open_market_theses,
    _route_global_context,
    _select_countertrend_lsr_target,
    _select_countertrend_lsr_target_result,
    _liquidity_route,
    _plan_feasibility,
    _typed_lsr,
    _typed_phase,
)
from smc_trader.risk import StructuralRiskEngine, _valid_targets
from smc_trader.scene_graph import (
    TemporalMarketSceneGraph,
    update_global_market_context,
)

from .helpers import long_plan, market_observation
from .test_v4_typed_vertical import (
    _advance_observation,
    _brain,
    _h4_structure,
    _lsr_observation,
    _mapped_brain,
    _ready_decision_layer,
)


BASE = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")


def _context(
    *,
    direction: Direction = Direction.LONG,
    mode: MarketMode = MarketMode.DIRECTIONAL,
    authority_source_ids: tuple[str, ...] = ("h4-authority",),
    path_blocker_ids: tuple[str, ...] = (),
    conflicts: tuple[GlobalConflictEvidence, ...] = (),
) -> GlobalMarketContext:
    authority = AuthorityLayer(
        timeframe=Timeframe.H4,
        direction=direction,
        structure_id="h4-authority",
        confirmed_at=BASE - pd.Timedelta(hours=4),
        protected_level_id=None,
        structural_scope="external",
        acceptance_state="confirmed",
        status="intact",
        source_ids=authority_source_ids,
    )
    return GlobalMarketContext(
        updated_at=BASE,
        scene_revision_id="scene:r000000000001",
        market_epoch_id="epoch:test",
        authority_stack=(authority,),
        market_mode=mode,
        scale_relation_details={
            timeframe.value: ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.ALIGNED,
                direction=direction,
                authority_layer_id=authority.structure_id,
                evidence_ids=(authority.structure_id,),
                evidence_kind="structure",
                structural_scope="external",
                acceptance_state="confirmed",
                since=authority.confirmed_at,
                age_bars=1,
                graph_connected=True,
                ambiguous=False,
            )
            for timeframe in Timeframe
        },
        external_draw_candidates={"above": (), "below": ()},
        obstruction_views={
            direction_.value: DirectionalObstructionView(
                direction=direction_,
                nearest_draw_id=None,
                nearest_draw_price=None,
                hard_barriers=tuple(
                    DeliveryObstruction(
                        obstruction_id=identity,
                        timeframe=Timeframe.H4,
                        direction=None,
                        side=(
                            "above"
                            if direction_ is Direction.LONG
                            else "below"
                        ),
                        lower_bound=(
                            105.0
                            if direction_ is Direction.LONG
                            else 95.0
                        ),
                        upper_bound=(
                            105.0
                            if direction_ is Direction.LONG
                            else 95.0
                        ),
                        hard=True,
                        source_kind="protected_swing",
                        source_ids=(identity,),
                        structural_scope="external",
                        acceptance_state="confirmed",
                    )
                    for identity in path_blocker_ids
                ),
                soft_frictions=(),
            )
            for direction_ in Direction
        },
        material_conflicts=conflicts,
        unknown_evidence=(),
        ambiguous_evidence=(),
        dislocations_by_scale={
            timeframe.value: () for timeframe in Timeframe
        },
    )


def _conflict(
    *,
    role: GlobalConflictRole,
    hypothesis_key: str,
    source_timeframe: Timeframe,
    target_timeframe: Timeframe = Timeframe.H4,
    structural_scale: str,
    source_direction: Direction = Direction.SHORT,
    target_direction: Direction = Direction.LONG,
    event_id: str = "frozen-source",
) -> GlobalConflictEvidence:
    return GlobalConflictEvidence(
        conflict_id=(
            f"conflict:{role.value}:{source_direction.value}:"
            f"{target_direction.value}:{event_id}"
        ),
        event_id=event_id,
        observed_at=BASE,
        source_node_id=f"node:{event_id}",
        target_node_id=f"node:{event_id}:authority",
        source_timeframe=source_timeframe,
        target_timeframe=target_timeframe,
        source_direction=source_direction,
        target_direction=target_direction,
        structural_scale=structural_scale,
        role=role,
        reason=role.value,
        affected_hypothesis_ids=(hypothesis_key,),
    )


def _evaluation(*, plan: object | None = None) -> _Evaluation:
    return _Evaluation(
        evidence=(),
        trigger_ready=True,
        setup_clock=BASE,
        sequence_signals={
            "step": _SequenceSignal(1.0, BASE, ("frozen-source",))
        },
        setup_identity="setup",
        context_identity="context",
        episode_identity="setup",
        entry_location_id="location",
        entry_path_id="path",
        plan=plan,
        thesis_target=0.7,
        location_quality=0.8,
        entry_readiness=0.8,
        delivery_quality=1.0,
        typed_uncertainty=0.1,
        uncertainty_authority_missing=0.1,
        hard_gate_results={"gate": True},
    )


def _completed_sequence(
    step_ids: tuple[str, ...],
) -> HypothesisSequenceState:
    return HypothesisSequenceState(
        protocol_version="test",
        protocol_hash="test-protocol",
        setup_id="setup",
        started_at=BASE,
        steps=tuple(
            SequenceStepState(
                step_id=step_id,
                satisfied=True,
                value=1.0,
                observed_at=BASE,
                source_ids=("frozen-source",),
            )
            for step_id in step_ids
        ),
    )


def _inventory(
    identity: str,
    *,
    timeframe: Timeframe,
    side: str,
    lower: float,
    upper: float,
    rank: str = "internal",
    protected: bool = False,
    source_ids: tuple[str, ...] | None = None,
) -> LiquidityInventoryItem:
    return LiquidityInventoryItem(
        item_id=identity,
        timeframe=timeframe,
        side=side,
        kind=(
            "previous_day_high"
            if side == "above"
            else "previous_day_low"
        ),
        price=lower if side == "below" else upper,
        lower_bound=lower,
        upper_bound=upper,
        formed_at=BASE - pd.Timedelta(hours=2),
        confirmed_at=BASE - pd.Timedelta(hours=1),
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=source_ids or (f"source:{identity}",),
        age_bars=1,
        strength=0.7,
        structural_rank=rank,
        is_protected_swing=protected,
        visibility_strength=0.7,
    )


def _select(
    direction: Direction,
    entry: float,
    items: tuple[LiquidityInventoryItem, ...],
    context: GlobalMarketContext,
):
    observation = replace(
        market_observation(asof=BASE, price=entry),
        liquidity_inventory=items,
    )
    inventory = {item.item_id: item for item in items}
    views = dict(context.obstruction_views)
    current_view = views[direction.value]
    views[direction.value] = replace(
        current_view,
        hard_barriers=tuple(
            DeliveryObstruction(
                obstruction_id=identity,
                timeframe=inventory[identity].timeframe,
                direction=None,
                side=inventory[identity].side,
                lower_bound=inventory[identity].lower_bound,
                upper_bound=inventory[identity].upper_bound,
                hard=True,
                source_kind=inventory[identity].kind,
                source_ids=inventory[identity].source_ids,
                structural_scope=inventory[identity].structural_rank,
                acceptance_state="confirmed",
            )
            for identity in context.path_blocker_ids
            if identity in inventory
        ),
    )
    context = replace(context, obstruction_views=views)
    return _select_countertrend_lsr_target(
        observation,
        direction,
        entry,
        BrainConfig(),
        context,
        context_draw=None,
    )


def _with_obstructions(
    context: GlobalMarketContext,
    direction: Direction,
    *,
    hard: tuple[DeliveryObstruction, ...] = (),
    soft: tuple[DeliveryObstruction, ...] = (),
) -> GlobalMarketContext:
    views = dict(context.obstruction_views)
    views[direction.value] = DirectionalObstructionView(
        direction=direction,
        nearest_draw_id=None,
        nearest_draw_price=None,
        hard_barriers=hard,
        soft_frictions=soft,
    )
    return replace(context, obstruction_views=views)


def _obstruction(
    identity: str,
    *,
    direction: Direction,
    price: float,
    hard: bool,
    source_ids: tuple[str, ...] | None = None,
    zone_direction: Direction | None = None,
    source_kind: str | None = None,
    timeframe: Timeframe | None = None,
    structural_scope: str | None = None,
    acceptance_state: str = "confirmed",
) -> DeliveryObstruction:
    return DeliveryObstruction(
        obstruction_id=identity,
        timeframe=(
            timeframe
            or (Timeframe.H1 if hard else Timeframe.M15)
        ),
        direction=zone_direction,
        side="above" if direction is Direction.LONG else "below",
        lower_bound=price,
        upper_bound=price,
        hard=hard,
        source_kind=(
            source_kind
            or ("protected_swing" if hard else "support_resistance")
        ),
        source_ids=source_ids or (f"source:{identity}",),
        structural_scope=(
            structural_scope
            or ("external" if hard else "internal")
        ),
        acceptance_state=acceptance_state,
    )


def test_local_countertrend_conflict_does_not_force_global_uncertainty() -> None:
    key = "liquidity_sweep_reversal:short"
    conflict = _conflict(
        role=GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY,
        hypothesis_key=key,
        source_timeframe=Timeframe.M5,
        structural_scale="internal",
    )
    routed = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        _evaluation(plan=object()),
        _context(conflicts=(conflict,)),
    )

    assert routed.thesis_target == pytest.approx(0.7)
    assert routed.delivery_quality == pytest.approx(1.0)
    assert routed.entry_readiness == pytest.approx(0.8)
    assert routed.typed_uncertainty == pytest.approx(0.1)
    assert not routed.invalidated


@pytest.mark.parametrize(
    ("direction", "source_direction", "target_direction"),
    (
        (Direction.LONG, Direction.SHORT, Direction.LONG),
        (Direction.SHORT, Direction.LONG, Direction.SHORT),
    ),
)
def test_authority_transition_challenge_is_long_short_symmetric(
    direction: Direction,
    source_direction: Direction,
    target_direction: Direction,
) -> None:
    key = f"{Playbook.LIQUIDITY_SWEEP_REVERSAL.value}:{direction.value}"
    conflict = _conflict(
        role=GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE,
        hypothesis_key=key,
        source_timeframe=Timeframe.H1,
        structural_scale="external",
        source_direction=source_direction,
        target_direction=target_direction,
    )
    evaluation = _evaluation(plan=object())
    routed = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        direction,
        None,
        evaluation,
        _context(
            direction=target_direction,
            mode=MarketMode.TRANSITION,
            conflicts=(conflict,),
        ),
    )

    assert routed.authority_relation is (
        HypothesisConflictRelation.CHALLENGES_INCUMBENT
    )
    assert routed.thesis_target == evaluation.thesis_target
    assert routed.sequence_signals == evaluation.sequence_signals
    assert not routed.trigger_ready
    assert routed.delivery_quality == 0.0
    assert routed.material_authority_opposition


def test_authority_transition_supports_lsr_challenger() -> None:
    key = "liquidity_sweep_reversal:short"
    conflict = _conflict(
        role=GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE,
        hypothesis_key=key,
        source_timeframe=Timeframe.H1,
        structural_scale="external",
    )
    evaluation = _evaluation(plan=object())
    routed = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        evaluation,
        _context(mode=MarketMode.TRANSITION, conflicts=(conflict,)),
    )

    assert routed.authority_relation is (
        HypothesisConflictRelation.SUPPORTS_CHALLENGER
    )
    assert routed.trigger_ready
    assert routed.delivery_quality == evaluation.delivery_quality
    assert not routed.material_authority_opposition
    assert routed.typed_uncertainty == evaluation.typed_uncertainty


def test_authority_transition_opposes_lsr_incumbent_direction() -> None:
    key = "liquidity_sweep_reversal:long"
    conflict = _conflict(
        role=GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE,
        hypothesis_key=key,
        source_timeframe=Timeframe.H1,
        structural_scale="external",
    )
    routed = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.LONG,
        None,
        _evaluation(plan=object()),
        _context(mode=MarketMode.TRANSITION, conflicts=(conflict,)),
    )

    assert routed.authority_relation is (
        HypothesisConflictRelation.CHALLENGES_INCUMBENT
    )
    assert not routed.trigger_ready
    assert routed.delivery_quality == 0.0
    assert routed.material_authority_opposition


def test_identity_disconnected_transition_is_unrelated() -> None:
    key = "liquidity_sweep_reversal:long"
    conflict = _conflict(
        role=GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE,
        hypothesis_key=key,
        source_timeframe=Timeframe.H1,
        structural_scale="external",
        event_id="other-episode",
    )
    evaluation = _evaluation(plan=object())
    routed = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.LONG,
        None,
        evaluation,
        _context(mode=MarketMode.TRANSITION, conflicts=(conflict,)),
    )

    assert routed.authority_relation is HypothesisConflictRelation.UNRELATED
    assert routed.trigger_ready == evaluation.trigger_ready
    assert routed.delivery_quality == evaluation.delivery_quality
    assert routed.typed_uncertainty == evaluation.typed_uncertainty
    assert not routed.material_authority_opposition


def test_authority_invalidation_only_terminalizes_incumbent_direction() -> None:
    conflict = _conflict(
        role=GlobalConflictRole.AUTHORITY_INVALIDATION,
        hypothesis_key="liquidity_sweep_reversal:long",
        source_timeframe=Timeframe.H1,
        structural_scale="external",
    )
    # Scene facts normally list both affected directions.  Preserve the same
    # identity binding while routing each hypothesis by source/target side.
    conflict = replace(
        conflict,
        affected_hypothesis_ids=(
            "liquidity_sweep_reversal:long",
            "liquidity_sweep_reversal:short",
        ),
    )
    context = _context(mode=MarketMode.TRANSITION, conflicts=(conflict,))
    incumbent = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.LONG,
        None,
        _evaluation(plan=object()),
        context,
    )
    challenger = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        _evaluation(plan=object()),
        context,
    )

    assert incumbent.invalidated
    assert incumbent.terminal_reason == "global_authority_invalidated"
    assert not challenger.invalidated
    assert challenger.authority_relation is (
        HypothesisConflictRelation.SUPPORTS_CHALLENGER
    )
    assert challenger.trigger_ready


def test_hard_obstruction_before_target_blocks_delivery_not_thesis() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    barrier_price = (plan.planned_entry + plan.targets[0].price) / 2.0
    context = _with_obstructions(
        _context(),
        Direction.LONG,
        hard=(
            _obstruction(
                "h1-hard-barrier",
                direction=Direction.LONG,
                price=barrier_price,
                hard=True,
            ),
        ),
    )
    evaluation = _evaluation(plan=plan)
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        context,
    )

    assert routed.thesis_target == evaluation.thesis_target
    assert not routed.invalidated
    assert routed.delivery_quality == 0.0
    assert routed.hard_barrier_before_target
    assert routed.path_blocker_ids == ("h1-hard-barrier",)
    assert routed.obstruction_distance_R is not None
    assert routed.free_path_R == pytest.approx(
        (barrier_price - plan.planned_entry) / plan.risk_points
    )


def test_only_nearest_authority_matched_hard_cluster_is_frozen() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    target_distance = plan.targets[0].price - plan.planned_entry
    context = _with_obstructions(
        _context(),
        Direction.LONG,
        hard=(
            _obstruction(
                "near-h1-barrier",
                direction=Direction.LONG,
                price=plan.planned_entry + target_distance * 0.25,
                hard=True,
            ),
            _obstruction(
                "far-h4-barrier",
                direction=Direction.LONG,
                price=plan.planned_entry + target_distance * 0.75,
                hard=True,
                timeframe=Timeframe.H4,
            ),
        ),
    )
    evaluation = _evaluation(plan=plan)
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        context,
    )

    assert routed.delivery_quality == 0.0
    assert routed.path_blocker_ids == ("near-h1-barrier",)
    assert routed.thesis_target == evaluation.thesis_target
    assert routed.sequence_signals == evaluation.sequence_signals
    assert routed.location_quality == evaluation.location_quality
    assert routed.entry_readiness == evaluation.entry_readiness


def test_lower_authority_hard_candidate_is_only_soft_friction() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    barrier_price = (plan.planned_entry + plan.targets[0].price) / 2.0
    context = _with_obstructions(
        _context(),
        Direction.LONG,
        hard=(
            _obstruction(
                "m1-local-candidate",
                direction=Direction.LONG,
                price=barrier_price,
                hard=True,
                timeframe=Timeframe.M1,
                structural_scope="internal",
            ),
        ),
    )
    evaluation = _evaluation(plan=plan)
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        context,
    )

    assert not routed.hard_barrier_before_target
    assert routed.path_blocker_ids == ()
    assert routed.soft_obstruction_count == 1
    assert 0.0 < routed.delivery_quality < evaluation.delivery_quality


def test_selected_draw_alias_inside_cluster_is_not_a_blocker() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    target_distance = plan.targets[0].price - plan.planned_entry
    context = _with_obstructions(
        _context(),
        Direction.LONG,
        hard=(
            _obstruction(
                "cluster-representative",
                direction=Direction.LONG,
                price=plan.planned_entry + target_distance * 0.5,
                hard=True,
                source_ids=(
                    "cluster-representative",
                    plan.targets[0].level_id,
                ),
            ),
        ),
    )
    evaluation = _evaluation(plan=plan)
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        context,
    )

    assert routed.delivery_quality == evaluation.delivery_quality
    assert routed.path_blocker_ids == ()
    assert routed.soft_obstruction_count == 0


def test_barrier_at_or_beyond_target_is_not_inside_delivery_path() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    context = _with_obstructions(
        _context(),
        Direction.LONG,
        hard=(
            _obstruction(
                "at-target",
                direction=Direction.LONG,
                price=plan.targets[0].price,
                hard=True,
            ),
        ),
    )
    evaluation = _evaluation(plan=plan)
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        context,
    )

    assert routed.delivery_quality == evaluation.delivery_quality
    assert not routed.hard_barrier_before_target


def test_soft_friction_only_reduces_delivery_monotonically() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    target_distance = plan.targets[0].price - plan.planned_entry
    context = _with_obstructions(
        _context(),
        Direction.LONG,
        soft=(
            _obstruction(
                "soft-1",
                direction=Direction.LONG,
                price=plan.planned_entry + target_distance * 0.25,
                hard=False,
            ),
            _obstruction(
                "soft-2",
                direction=Direction.LONG,
                price=plan.planned_entry + target_distance * 0.5,
                hard=False,
            ),
        ),
    )
    evaluation = _evaluation(plan=plan)
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        context,
    )

    assert routed.delivery_quality == pytest.approx(
        evaluation.delivery_quality / 3.0
    )
    assert routed.soft_obstruction_count == 2
    assert not routed.hard_barrier_before_target
    assert routed.trigger_ready
    assert not routed.invalidated


def test_selected_draw_is_not_reclassified_as_hard_obstruction() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    context = _with_obstructions(
        _context(),
        Direction.LONG,
        hard=(
            _obstruction(
                plan.targets[0].level_id,
                direction=Direction.LONG,
                price=plan.targets[0].price,
                hard=True,
            ),
        ),
    )
    evaluation = _evaluation(plan=plan)
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        context,
    )

    assert routed.delivery_quality == evaluation.delivery_quality
    assert not routed.hard_barrier_before_target
    assert routed.soft_obstruction_count == 0


def test_only_opposing_directional_zone_blocks_delivery() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    target_distance = plan.targets[0].price - plan.planned_entry
    context = _with_obstructions(
        _context(),
        Direction.LONG,
        hard=(
            _obstruction(
                "same-direction-fvg",
                direction=Direction.LONG,
                price=plan.planned_entry + target_distance * 0.25,
                hard=True,
                zone_direction=Direction.LONG,
                source_kind="fvg",
            ),
            _obstruction(
                "opposing-fvg",
                direction=Direction.LONG,
                price=plan.planned_entry + target_distance * 0.5,
                hard=True,
                zone_direction=Direction.SHORT,
                source_kind="fvg",
            ),
        ),
    )
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        _evaluation(plan=plan),
        context,
    )

    assert routed.hard_barrier_before_target
    assert routed.delivery_quality == 0.0
    assert routed.obstruction_distance_R == pytest.approx(
        target_distance * 0.5 / plan.risk_points
    )


def test_one_identity_bound_unknown_contributes_uncertainty_once() -> None:
    evaluation = _evaluation()
    context = replace(
        _context(),
        unknown_evidence=("frozen-source",),
    )

    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        context,
    )

    # The same authority-missing component is updated to its strongest
    # identity-bound value; it is not counted again as a separate conflict.
    assert routed.typed_uncertainty == pytest.approx(0.5)
    assert routed.trigger_ready is evaluation.trigger_ready
    assert routed.delivery_quality == evaluation.delivery_quality


def test_h1_authority_transition_pauses_h4_aligned_dfp_without_erasing_plan() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    key = "displacement_first_pullback:long"
    conflict = _conflict(
        role=GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE,
        hypothesis_key=key,
        source_timeframe=Timeframe.H1,
        structural_scale="intermediate",
    )
    evaluation = _evaluation(plan=plan)
    routed = _route_global_context(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation,
        _context(mode=MarketMode.TRANSITION, conflicts=(conflict,)),
    )

    assert routed.plan is plan
    assert routed.thesis_target == pytest.approx(evaluation.thesis_target)
    assert routed.sequence_signals == evaluation.sequence_signals
    assert not routed.trigger_ready
    assert routed.delivery_quality == 0.0
    # Existing 0.1 authority uncertainty combines with the 0.5 transition
    # conflict by noisy-OR: 1 - (0.9 * 0.5) = 0.55.
    assert routed.typed_uncertainty == pytest.approx(0.55)
    assert routed.hard_gate_results == evaluation.hard_gate_results
    assert routed.authority_relation is (
        HypothesisConflictRelation.CHALLENGES_INCUMBENT
    )
    assert routed.authority_rank_gap == 1
    assert routed.conflict_role is (
        GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE
    )
    assert routed.conflict_scope == "intermediate"
    assert routed.acceptance_state == "confirmed"

    sequence = _completed_sequence(
        (
            "h4_structure_and_draw",
            "m5_displacement_zone",
            "first_pullback_to_frozen_zone",
            "typed_entry_trigger",
        )
    )
    # Complete sequence + retained plan cannot remain executable while its
    # current trigger/delivery permission is paused.
    phase = _typed_phase(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        routed.thesis_target,
        routed,
        sequence,
        plan,
        plan.invalidation,
        plan.deadline,
        observation,
        None,
        BrainConfig(),
        parked=False,
    )
    assert phase is PlaybookPhase.WEAKENING


def test_shared_global_authority_cannot_bind_an_unrelated_open_episode() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    thesis = OpenMarketThesis(
        thesis_id="market-thesis:unrelated",
        root_id="different-displacement",
        market_epoch_id="epoch:test",
        formed_at=BASE,
        updated_at=BASE,
        direction=Direction.LONG,
        source_timeframe=Timeframe.M5,
        structural_scale="intermediate",
        mechanism="directional_displacement",
        authority_relation="aligned",
        authority_source_ids=("frozen-source",),
        # A closure member may be shared with this evaluation.  It remains
        # descriptive support and cannot substitute for the canonical root.
        mechanism_event_ids=("different-displacement", "frozen-source"),
    )
    context = replace(
        _context(authority_source_ids=("frozen-source",)),
        open_market_theses=(thesis,),
    )
    evaluation = _bind_open_market_theses(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        _evaluation(plan=plan),
        context,
    )

    assert evaluation.market_thesis_ids == (thesis.thesis_id,)
    assert evaluation.market_thesis_binding_required
    assert not evaluation.market_thesis_action_bound
    assert evaluation.market_thesis_id == thesis.thesis_id
    assert evaluation.bound_market_thesis_id is None
    assert evaluation.market_thesis_match_status == "root_identity_unbound"

    phase = _typed_phase(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        evaluation.thesis_target,
        evaluation,
        _completed_sequence(
            (
                "h4_structure_and_draw",
                "m5_displacement_zone",
                "first_pullback_to_frozen_zone",
                "typed_entry_trigger",
            )
        ),
        plan,
        plan.invalidation,
        plan.deadline,
        observation,
        None,
        BrainConfig(),
        parked=False,
    )
    assert phase is PlaybookPhase.WEAKENING


@pytest.mark.parametrize(
    ("thesis_direction", "mechanism", "expected_status"),
    (
        (None, None, "no_open_thesis"),
        (Direction.SHORT, "directional_displacement", "no_direction_match"),
        (Direction.LONG, "liquidity_sweep", "no_mechanism_match"),
    ),
)
def test_open_thesis_unmatched_reason_is_explicit(
    thesis_direction: Direction | None,
    mechanism: str | None,
    expected_status: str,
) -> None:
    theses = ()
    if mechanism is not None:
        thesis = OpenMarketThesis(
            thesis_id=f"market-thesis:{expected_status}",
            root_id=f"root:{expected_status}",
            market_epoch_id="epoch:test",
            formed_at=BASE,
            updated_at=BASE,
            direction=thesis_direction,
            source_timeframe=Timeframe.M5,
            structural_scale="intermediate",
            mechanism=mechanism,
            authority_relation="unrelated",
            mechanism_event_ids=(f"root:{expected_status}",),
        )
        theses = (thesis,)
    evaluation = _bind_open_market_theses(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        None,
        _evaluation(plan=long_plan(market_observation(asof=BASE))),
        replace(_context(), open_market_theses=theses),
    )

    assert evaluation.market_thesis_binding_required
    assert not evaluation.market_thesis_action_bound
    assert evaluation.market_thesis_ids == ()
    assert evaluation.market_thesis_match_status == expected_status


def test_position_weakens_on_zero_delivery_or_material_authority_opposition() -> None:
    observation = market_observation(asof=BASE)
    plan = long_plan(observation)
    position = PositionSnapshot(
        thesis_hash="position:test",
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=BASE - pd.Timedelta(minutes=1),
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.5,
        elapsed_minutes=1,
    )
    sequence = _completed_sequence(("h4_structure_and_draw",))
    for evaluation in (
        replace(_evaluation(plan=plan), delivery_quality=0.0),
        replace(
            _evaluation(plan=plan),
            material_authority_opposition=True,
        ),
    ):
        phase = _typed_phase(
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Direction.LONG,
            None,
            0.7,
            evaluation,
            sequence,
            plan,
            plan.invalidation,
            plan.deadline,
            observation,
            position,
            BrainConfig(),
            parked=False,
        )
        assert phase is PlaybookPhase.WEAKENING


def test_h1_internal_target_before_h4_protected_barrier_is_selectable() -> None:
    target = _inventory(
        "h1-internal-target",
        timeframe=Timeframe.H1,
        side="below",
        lower=99.25,
        upper=99.5,
    )
    barrier = _inventory(
        "h4-protected-barrier",
        timeframe=Timeframe.H4,
        side="below",
        lower=98.0,
        upper=98.0,
        rank="external",
        protected=True,
        source_ids=("h4-authority",),
    )
    selected = _select(
        Direction.SHORT,
        101.0,
        (target, barrier),
        _context(path_blocker_ids=(barrier.item_id,)),
    )

    assert selected is not None
    assert selected.level_id == target.item_id
    assert selected.timeframe is Timeframe.H1
    assert selected.price == pytest.approx(target.upper_bound)


def test_m5_target_beyond_authority_barrier_is_rejected() -> None:
    barrier = _inventory(
        "h1-authority-barrier",
        timeframe=Timeframe.H1,
        side="below",
        lower=100.0,
        upper=100.0,
        rank="external",
        protected=True,
        source_ids=("h4-authority",),
    )
    target = _inventory(
        "m5-target-behind",
        timeframe=Timeframe.M5,
        side="below",
        lower=99.25,
        upper=99.5,
    )
    assert _select(
        Direction.SHORT,
        101.0,
        (target, barrier),
        _context(path_blocker_ids=(barrier.item_id,)),
    ) is None


def test_opposite_side_blocker_is_ignored_when_forward_barrier_exists() -> None:
    reverse = _inventory(
        "reverse-side-barrier",
        timeframe=Timeframe.H4,
        side="below",
        lower=99.0,
        upper=99.0,
        rank="external",
        protected=True,
    )
    target = _inventory(
        "long-target",
        timeframe=Timeframe.H1,
        side="above",
        lower=101.5,
        upper=102.0,
    )
    forward = _inventory(
        "forward-barrier",
        timeframe=Timeframe.H4,
        side="above",
        lower=103.0,
        upper=103.0,
        rank="external",
        protected=True,
        source_ids=("h4-authority",),
    )
    selected = _select(
        Direction.LONG,
        100.0,
        (reverse, target, forward),
        _context(
            path_blocker_ids=(reverse.item_id, forward.item_id),
        ),
    )
    assert selected is not None
    assert selected.level_id == target.item_id


@pytest.mark.parametrize(
    ("direction", "entry", "side", "target_bounds", "barrier_price", "expected"),
    (
        (Direction.LONG, 100.0, "above", (101.5, 102.0), 103.0, 101.5),
        (Direction.SHORT, 100.0, "below", (98.0, 98.5), 97.0, 98.5),
    ),
)
def test_countertrend_contact_geometry_is_long_short_symmetric(
    direction: Direction,
    entry: float,
    side: str,
    target_bounds: tuple[float, float],
    barrier_price: float,
    expected: float,
) -> None:
    target = _inventory(
        "target",
        timeframe=Timeframe.M5,
        side=side,
        lower=target_bounds[0],
        upper=target_bounds[1],
    )
    barrier = _inventory(
        "barrier",
        timeframe=Timeframe.H4,
        side=side,
        lower=barrier_price,
        upper=barrier_price,
        rank="external",
        protected=True,
        source_ids=("h4-authority",),
    )
    selected = _select(
        direction,
        entry,
        (target, barrier),
        _context(path_blocker_ids=(barrier.item_id,)),
    )
    assert selected is not None
    assert selected.price == pytest.approx(expected)


def test_countertrend_target_fails_closed_without_authority_barrier() -> None:
    target = _inventory(
        "unbounded-target",
        timeframe=Timeframe.M5,
        side="below",
        lower=98.0,
        upper=98.5,
    )
    assert _select(
        Direction.SHORT,
        100.0,
        (target,),
        _context(path_blocker_ids=()),
    ) is None


def test_countertrend_target_skips_near_waypoint_below_one_R() -> None:
    base = _lsr_observation()
    near = _inventory(
        "near-waypoint",
        timeframe=Timeframe.M5,
        side="below",
        lower=100.5,
        upper=100.5,
    )
    far = _inventory(
        "risk-qualified-primary",
        timeframe=Timeframe.H1,
        side="below",
        lower=99.0,
        upper=99.0,
    )
    observation = replace(
        base,
        price=101.0,
        liquidity_inventory=(
            near,
            far,
            *(
                item
                for item in base.liquidity_inventory
                if item.kind in {"equal_highs", "equal_lows"}
            ),
        ),
    )
    context = _with_obstructions(
        _context(),
        Direction.SHORT,
        hard=(
            _obstruction(
                "h4-authority-cap",
                direction=Direction.SHORT,
                price=98.0,
                hard=True,
                timeframe=Timeframe.H4,
            ),
        ),
    )
    result = _select_countertrend_lsr_target_result(
        observation,
        Direction.SHORT,
        101.0,
        BrainConfig(),
        context,
        context_draw=None,
        location=base.entry_locations[0],
        invalidation=StructuralLevel(
            price=102.25,
            side="above",
            source_level_id="sweep-extreme",
            observed_at=base.asof,
            rationale="frozen sweep extreme",
        ),
    )

    assert result.failure_reason is None
    assert result.target is not None
    assert result.target.level_id == far.item_id
    assert result.context_draw is not None
    assert result.context_draw.level_id == far.item_id
    assert result.authority_barrier_id == "h4-authority-cap"
    assert result.authority_barrier_price == pytest.approx(98.0)


def test_typed_lsr_does_not_let_near_waypoint_cap_risk_qualified_target() -> None:
    base = _lsr_observation()
    near = _inventory(
        "near-waypoint",
        timeframe=Timeframe.M5,
        side="below",
        lower=100.5,
        upper=100.5,
    )
    far = _inventory(
        "risk-qualified-primary",
        timeframe=Timeframe.H1,
        side="below",
        lower=99.0,
        upper=99.0,
    )
    pool_source = next(
        item
        for item in base.liquidity_inventory
        if item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
    )
    observation = replace(
        base,
        price=101.0,
        liquidity_inventory=(near, far, pool_source),
        manipulations=(
            replace(base.manipulations[0], sweep_extreme=102.25),
        ),
    )
    context = _with_obstructions(
        _context(),
        Direction.SHORT,
        hard=(
            _obstruction(
                "h4-authority-cap",
                direction=Direction.SHORT,
                price=98.0,
                hard=True,
                timeframe=Timeframe.H4,
            ),
        ),
    )
    brain = _brain()

    evaluation = _typed_lsr(
        observation,
        Direction.SHORT,
        brain.registry.for_playbook(
            Playbook.LIQUIDITY_SWEEP_REVERSAL
        ),
        None,
        BrainConfig(),
        global_context=context,
        scene_graph=None,
    )

    assert evaluation.plan is not None
    assert evaluation.plan.targets[0].level_id == far.item_id
    assert evaluation.plan.primary_target_R >= 1.0
    assert evaluation.plan.remaining_path_R >= 1.0
    route = evaluation.plan.liquidity_route
    assert route is not None
    assert route.intermediate_liquidity_ids == (near.item_id,)
    assert route.primary_deliverable_target_id == far.item_id
    assert route.context_draw_id == far.item_id
    assert route.terminal_draw_id == far.item_id
    assert route.authority_barrier_id == "h4-authority-cap"


@pytest.mark.parametrize(
    ("items", "barrier_price", "invalidation_price", "expected"),
    (
        ((), 98.0, 102.25, "visible_draw_missing"),
        (
            (
                _inventory(
                    "draw-without-cap",
                    timeframe=Timeframe.M5,
                    side="below",
                    lower=99.0,
                    upper=99.0,
                ),
            ),
            None,
            102.25,
            "authority_barrier_missing",
        ),
        (
            (
                _inventory(
                    "draw-behind-cap",
                    timeframe=Timeframe.M5,
                    side="below",
                    lower=97.0,
                    upper=97.0,
                ),
            ),
            98.0,
            102.25,
            "all_draws_beyond_barrier",
        ),
        (
            (
                _inventory(
                    "sub-one-R-only",
                    timeframe=Timeframe.M5,
                    side="below",
                    lower=100.5,
                    upper=100.5,
                ),
            ),
            98.0,
            103.0,
            "all_targets_below_minimum_R",
        ),
    ),
)
def test_countertrend_target_failure_reasons_are_specific(
    items: tuple[LiquidityInventoryItem, ...],
    barrier_price: float | None,
    invalidation_price: float,
    expected: str,
) -> None:
    base = _lsr_observation()
    context = _context()
    if barrier_price is not None:
        context = _with_obstructions(
            context,
            Direction.SHORT,
            hard=(
                _obstruction(
                    "authority-cap",
                    direction=Direction.SHORT,
                    price=barrier_price,
                    hard=True,
                    timeframe=Timeframe.H4,
                ),
            ),
        )
    result = _select_countertrend_lsr_target_result(
        replace(
            base,
            price=101.0,
            liquidity_inventory=(
                *items,
                *(
                    item
                    for item in base.liquidity_inventory
                    if item.kind in {"equal_highs", "equal_lows"}
                ),
            ),
        ),
        Direction.SHORT,
        101.0,
        BrainConfig(),
        context,
        context_draw=None,
        location=base.entry_locations[0],
        invalidation=StructuralLevel(
            price=invalidation_price,
            side="above",
            source_level_id="sweep-extreme",
            observed_at=base.asof,
            rationale="frozen sweep extreme",
        ),
    )
    assert result.target is None
    assert result.failure_reason == expected


def test_countertrend_target_distinguishes_exhausted_remaining_path() -> None:
    base = _lsr_observation()
    target = _inventory(
        "risk-qualified-but-already-delivered",
        timeframe=Timeframe.H1,
        side="below",
        lower=99.0,
        upper=99.0,
    )
    observation = replace(
        base,
        price=99.5,
        liquidity_inventory=(
            target,
            *(
                item
                for item in base.liquidity_inventory
                if item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            ),
        ),
    )
    context = _with_obstructions(
        _context(),
        Direction.SHORT,
        hard=(
            _obstruction(
                "authority-cap",
                direction=Direction.SHORT,
                price=98.0,
                hard=True,
                timeframe=Timeframe.H4,
            ),
        ),
    )
    result = _select_countertrend_lsr_target_result(
        observation,
        Direction.SHORT,
        101.0,
        BrainConfig(),
        context,
        context_draw=next(
            level
            for level in playbooks_module._visible_levels(observation)
            if level.level_id == target.item_id
        ),
        location=base.entry_locations[0],
        invalidation=StructuralLevel(
            price=102.25,
            side="above",
            source_level_id="sweep-extreme",
            observed_at=base.asof,
            rationale="frozen sweep extreme",
        ),
    )

    assert result.target is None
    assert result.failure_reason == "remaining_path_below_minimum"
    assert result.authority_barrier_id == "authority-cap"


def test_liquidity_route_deduplicates_waypoint_cluster_and_stops_at_primary() -> None:
    base = _lsr_observation()
    near_a = _inventory(
        "near-equal-low",
        timeframe=Timeframe.M5,
        side="below",
        lower=100.5,
        upper=100.5,
    )
    near_b = _inventory(
        "near-swing-low",
        timeframe=Timeframe.H1,
        side="below",
        lower=100.5,
        upper=100.5,
    )
    primary_item = _inventory(
        "primary",
        timeframe=Timeframe.H1,
        side="below",
        lower=99.0,
        upper=99.0,
    )
    beyond = _inventory(
        "beyond-primary",
        timeframe=Timeframe.M5,
        side="below",
        lower=98.5,
        upper=98.5,
    )
    observation = replace(
        base,
        liquidity_inventory=(
            near_a,
            near_b,
            primary_item,
            beyond,
            *(
                item
                for item in base.liquidity_inventory
                if item.kind in {"equal_highs", "equal_lows"}
            ),
        ),
    )
    primary = next(
        level
        for level in playbooks_module._visible_levels(observation)
        if level.level_id == primary_item.item_id
    )
    route = _liquidity_route(
        observation,
        Direction.SHORT,
        101.0,
        context_draw=primary,
        primary_target=primary,
        conservative_contacts=True,
        authority_barrier_id="h4-cap",
        authority_barrier_price=98.0,
        tick_size=0.25,
    )

    assert route is not None
    assert len(route.intermediate_liquidity_ids) == 1
    assert route.intermediate_liquidity_ids[0] in {
        near_a.item_id,
        near_b.item_id,
    }
    assert beyond.item_id not in route.intermediate_liquidity_ids
    assert route.primary_deliverable_target_id == primary_item.item_id
    assert route.terminal_draw_id == primary_item.item_id
    assert route.authority_barrier_id == "h4-cap"
    assert route.authority_barrier_price == pytest.approx(98.0)


def test_consumed_waypoint_does_not_invalidate_another_candidate_route() -> None:
    base = _lsr_observation()
    waypoint = _inventory(
        "shared-waypoint",
        timeframe=Timeframe.M5,
        side="below",
        lower=100.5,
        upper=100.5,
    )
    primary_item = _inventory(
        "candidate-b-primary",
        timeframe=Timeframe.H1,
        side="below",
        lower=99.0,
        upper=99.0,
    )
    observation = replace(
        base,
        liquidity_inventory=(
            waypoint,
            primary_item,
            *(
                item
                for item in base.liquidity_inventory
                if item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            ),
        ),
    )
    primary = next(
        level
        for level in playbooks_module._visible_levels(observation)
        if level.level_id == primary_item.item_id
    )
    route = _liquidity_route(
        observation,
        Direction.SHORT,
        101.0,
        context_draw=primary,
        primary_target=primary,
        conservative_contacts=True,
        authority_barrier_id="h4-cap",
        authority_barrier_price=98.0,
        tick_size=0.25,
    )
    assert route is not None
    assert route.intermediate_liquidity_ids == (waypoint.item_id,)
    evaluation = replace(_evaluation(), liquidity_route=route)

    waypoint_consumed = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        evaluation,
        replace(_context(), invalidated_source_ids=(waypoint.item_id,)),
    )
    assert waypoint_consumed.invalidated is False

    primary_consumed = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        evaluation,
        replace(_context(), invalidated_source_ids=(primary_item.item_id,)),
    )
    assert primary_consumed.invalidated is True
    assert primary_consumed.terminal_reason == "global_frozen_source_invalidated"


def test_common_plan_feasibility_preserves_context_draw_consumed_reason() -> None:
    evaluation = replace(
        _evaluation(),
        invalidation=StructuralLevel(
            price=102.0,
            side="above",
            source_level_id="sweep-extreme",
            observed_at=BASE,
            rationale="frozen sweep extreme",
        ),
        target_selection_failure_reason="context_draw_consumed",
    )
    feasibility = _plan_feasibility(None, evaluation, BrainConfig())
    assert feasibility.valid is False
    assert feasibility.failure_reason == "context_draw_consumed"


def test_lsr_contact_target_passes_risk_and_spoofed_bound_fails() -> None:
    observation = _lsr_observation()
    original_target = observation.liquidity_inventory[0]
    contact_target = replace(
        original_target,
        kind="previous_day_low",
        price=97.75,
        lower_bound=97.75,
        upper_bound=98.0,
        source_ids=("equal-low-a", "equal-low-b"),
    )
    barrier = replace(
        original_target,
        item_id="h4-protected-barrier",
        timeframe=Timeframe.H4,
        kind="previous_day_low",
        price=96.0,
        lower_bound=96.0,
        upper_bound=96.0,
        source_ids=("h4-authority",),
        structural_rank="external",
        is_protected_swing=True,
    )
    observation = replace(
        observation,
        liquidity_inventory=(
            contact_target,
            observation.liquidity_inventory[1],
            barrier,
        ),
    )
    brain = _brain()
    evaluation = _typed_lsr(
        observation,
        Direction.SHORT,
        brain.registry.for_playbook(
            Playbook.LIQUIDITY_SWEEP_REVERSAL
        ),
        None,
        BrainConfig(),
        global_context=_with_obstructions(
            _context(direction=Direction.LONG),
            Direction.SHORT,
            hard=(
                _obstruction(
                    barrier.item_id,
                    direction=Direction.SHORT,
                    price=barrier.price,
                    hard=True,
                    timeframe=Timeframe.H4,
                    source_ids=barrier.source_ids,
                ),
            ),
        ),
        scene_graph=None,
    )
    plan = evaluation.plan
    assert plan is not None
    assert plan.liquidity_route is not None
    assert (
        plan.liquidity_route.primary_target_price_basis
        == "conservative_contact"
    )
    assert plan.liquidity_route.authority_barrier_id == barrier.item_id
    assert plan.liquidity_route.authority_barrier_price == pytest.approx(
        barrier.price
    )
    assert plan.targets[0].price == pytest.approx(
        contact_target.upper_bound
    )
    assert plan.draw_selection is not None
    assert plan.draw_selection.price == plan.targets[0].price
    assert _valid_targets(plan, observation)

    decision = Decision(
        asof=observation.asof,
        selected_action=Action.ENTER,
        utilities=(
            ActionUtility(
                action=Action.ENTER,
                utility=1.0,
                components={},
                hypothesis_key="liquidity_sweep_reversal:short",
                reason="typed contact target test",
            ),
        ),
        best_hypothesis_key="liquidity_sweep_reversal:short",
        advantage=1.0,
        reasons=("typed contact target test",),
        plan=plan,
    )
    approved = StructuralRiskEngine().review(decision, observation)
    assert approved.passed
    assert approved.frozen_thesis is not None
    assert approved.frozen_thesis.original_targets == plan.targets

    forged_price = contact_target.lower_bound
    forged_target = replace(plan.targets[0], price=forged_price)
    forged_plan = replace(
        plan,
        targets=(forged_target,),
        primary_target_R=(
            abs(forged_price - plan.planned_entry) / plan.risk_points
        ),
        remaining_path_R=(
            max(
                0.0,
                plan.direction.sign
                * (forged_price - observation.price),
            )
            / plan.risk_points
        ),
        draw_selection=replace(
            plan.draw_selection,
            price=forged_price,
        ),
    )
    assert not _valid_targets(forged_plan, observation)

    crossed_route = replace(
        plan.liquidity_route,
        authority_barrier_price=(
            plan.planned_entry + plan.targets[0].price
        )
        / 2.0,
    )
    assert not _valid_targets(
        replace(plan, liquidity_route=crossed_route),
        observation,
    )
    insufficient_buffer = replace(
        plan.liquidity_route,
        authority_barrier_price=(
            plan.targets[0].price - 0.01
            if plan.direction is Direction.SHORT
            else plan.targets[0].price + 0.01
        ),
    )
    assert not _valid_targets(
        replace(plan, liquidity_route=insufficient_buffer),
        observation,
    )


def test_normal_lsr_nonzero_width_target_uses_inventory_price() -> None:
    observation = _lsr_observation()
    original_target = observation.liquidity_inventory[0]
    normal_target = replace(
        original_target,
        kind="previous_day_low",
        price=97.875,
        lower_bound=97.75,
        upper_bound=98.0,
        source_ids=("previous-day-low",),
    )
    observation = replace(
        observation,
        liquidity_inventory=(
            normal_target,
            *observation.liquidity_inventory[1:],
        ),
    )
    brain = _brain()
    evaluation = _typed_lsr(
        observation,
        Direction.SHORT,
        brain.registry.for_playbook(
            Playbook.LIQUIDITY_SWEEP_REVERSAL
        ),
        None,
        BrainConfig(),
        global_context=None,
        scene_graph=None,
    )
    plan = evaluation.plan
    assert plan is not None
    assert plan.liquidity_route is not None
    assert plan.liquidity_route.primary_target_price_basis == "inventory_price"
    assert plan.targets[0].price == pytest.approx(normal_target.price)
    assert _valid_targets(plan, observation)


def test_risk_rejects_lsr_target_crossing_frozen_hard_barrier() -> None:
    observation = _lsr_observation()
    brain = _brain()
    evaluation = _typed_lsr(
        observation,
        Direction.SHORT,
        brain.registry.for_playbook(
            Playbook.LIQUIDITY_SWEEP_REVERSAL
        ),
        None,
        BrainConfig(),
        global_context=None,
        scene_graph=None,
    )
    assert evaluation.plan is not None
    plan = evaluation.plan
    barrier_price = (plan.planned_entry + plan.targets[0].price) / 2.0
    context = _with_obstructions(
        _context(direction=Direction.SHORT),
        Direction.SHORT,
        hard=(
            _obstruction(
                "accepted-h1-opposition",
                direction=Direction.SHORT,
                price=barrier_price,
                hard=True,
            ),
        ),
    )
    routed = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        evaluation,
        context,
    )

    assert routed.plan is not None
    assert routed.plan.liquidity_route is not None
    assert routed.plan.liquidity_route.path_blocker_ids == (
        "accepted-h1-opposition",
    )
    assert routed.delivery_quality == 0.0
    assert not _valid_targets(routed.plan, observation)


def test_risk_approved_lsr_target_stays_frozen_after_new_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _lsr_observation()
    brain = _mapped_brain()
    graph = TemporalMarketSceneGraph()
    initial_delta = graph.update(observation)
    belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=initial_delta,
    )
    original = belief.hypotheses["liquidity_sweep_reversal:short"]
    assert original.plan is not None
    decision = _ready_decision_layer(brain).decide(observation, belief)
    approved = StructuralRiskEngine().review(decision, observation)
    assert approved.passed and approved.frozen_thesis is not None
    frozen_target = approved.frozen_thesis.original_targets[0]
    plan = original.plan
    position = PositionSnapshot(
        thesis_hash=approved.frozen_thesis.thesis_hash,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        playbook=original.playbook,
        direction=original.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=frozen_target,
        opened_at=observation.asof,
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.25,
        elapsed_minutes=1,
        setup_id=plan.setup_id,
        entry_location_id=plan.entry_location_id,
        entry_path_id=plan.entry_path_id,
    )

    later_asof = observation.asof + pd.Timedelta(minutes=1)
    blocker = replace(
        observation.liquidity_inventory[0],
        item_id="new-h4-protected-blocker",
        timeframe=Timeframe.H4,
        kind="previous_day_low",
        price=99.0,
        lower_bound=99.0,
        upper_bound=99.0,
        source_ids=("h4-low",),
        structural_rank="external",
        is_protected_swing=True,
    )
    new_near_target = replace(
        observation.liquidity_inventory[0],
        item_id="new-near-target",
        kind="previous_day_low",
        price=99.5,
        lower_bound=99.5,
        upper_bound=99.5,
        source_ids=("new-near-target-source",),
        structural_rank="internal",
        is_protected_swing=False,
    )
    later = _advance_observation(observation, later_asof)
    frames = dict(later.frames)
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        structures=(_h4_structure(observation.asof),),
    )
    later = replace(
        later,
        frames=frames,
        liquidity_inventory=(
            *observation.liquidity_inventory,
            new_near_target,
            blocker,
        ),
    )
    delta = graph.update(later)
    current_context = update_global_market_context(
        belief.global_context,
        later,
        delta,
        graph,
    )
    current_evaluation = _typed_lsr(
        later,
        Direction.SHORT,
        brain.registry.for_playbook(
            Playbook.LIQUIDITY_SWEEP_REVERSAL
        ),
        original,
        BrainConfig(),
        global_context=current_context,
        scene_graph=graph,
    )
    assert current_evaluation.plan is not None
    assert current_evaluation.plan.targets[0].level_id == new_near_target.item_id
    assert current_evaluation.delivery_quality > 0.0
    # This test isolates Brain routing/position freezing.  The public graph
    # update still supplies authority and blocker identities; the separate
    # sequential graph suite owns full historical LSR-chain connectivity.
    monkeypatch.setattr(
        playbooks_module,
        "_require_connected_graph_sequence",
        lambda _playbook, _protocol, evaluation, _graph, _prior: evaluation,
    )
    updated = brain.update(
        later,
        position=position,
        scene_graph=graph,
        scene_delta=delta,
    ).hypotheses["liquidity_sweep_reversal:short"]

    assert blocker.item_id in brain.current.global_context.path_blocker_ids
    assert updated.delivery_quality == 0.0
    assert updated.phase is PlaybookPhase.WEAKENING
    assert updated.plan == plan
    assert updated.plan.targets[0] == frozen_target
    assert position.primary_target == frozen_target
    assert approved.frozen_thesis.original_targets[0] == frozen_target
