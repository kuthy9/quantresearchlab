from __future__ import annotations

import pandas as pd

from smc_trader.brain_entry_sequence import brain_observation_view
from smc_trader.decision import UtilityDecisionLayer
from smc_trader.group5 import CausalGroup5Reducer
from smc_trader.model import (
    Candle,
    DealingRangeLifecycle,
    DisplacementObservation,
    Direction,
    EngineSnapshot,
    EntryLocationLifecycle,
    EventKind,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    MarketEvent,
    PathSequenceLifecycle,
    Playbook,
    PlaybookPhase,
    Timeframe,
)
from smc_trader.playbook_registry import load_playbook_registry
from smc_trader.playbooks import (
    PlaybookBrain,
    _stable_context_thesis_id,
    _typed_favr,
)
from smc_trader.risk import StructuralRiskEngine
from smc_trader.scene_graph import (
    SceneEdgeKind,
    TemporalMarketSceneGraph,
)
from smc_trader.visualization import DecisionVisualizer

from .helpers import (
    graph_free_action_belief,
    market_observation,
)
from .test_range_auction_primitives import (
    _m1 as _group4_m1,
    _mature_range,
    _protocol as _range_auction_protocol,
)
from .test_v4_typed_vertical import (
    ROOT,
    _execution,
    _fvg,
    _group5_protocol,
    _brain,
    _m1,
    _lsr_observation,
    _opposed_m5_bos,
    _interaction_update_from_brain_fields,
    replace,
    replace_market_observation,
)


def _unparked_favr_brain() -> PlaybookBrain:
    registry = load_playbook_registry(
        ROOT / "configs/playbooks.json"
    )
    protocols = tuple(
        replace(
            protocol,
            status="typed_causal_implemented_test_only",
        )
        if protocol.playbook
        is Playbook.FAILED_AUCTION_VALUE_RETURN
        else protocol
        for protocol in registry.protocols
    )
    return PlaybookBrain(registry=replace(registry, protocols=protocols))


def _range_failed_auction_fixture():
    tracker, _, _ = _mature_range(_range_auction_protocol())
    inventory = tracker.snapshot().range_boundary_inventory
    for index in range(15):
        output = tracker.on_completed_update(
            _group4_m1(index),
            prior_inventory=inventory,
            liquidity_pools=(),
        )
        inventory = output.range_boundary_inventory

    sweep_candle = _group4_m1(
        15,
        close=98.90,
        high=100.25,
        low=98.75,
    )
    swept = tracker.on_completed_update(
        sweep_candle,
        prior_inventory=inventory,
        liquidity_pools=(),
    )
    assert len(swept.manipulations) == 1
    assert (
        swept.manipulations[0].lifecycle
        is ManipulationLifecycle.SWEPT
    )

    candidate_candle = _group4_m1(
        16,
        close=99.50,
        high=100.00,
        low=99.25,
    )
    candidate = tracker.on_completed_update(
        candidate_candle,
        prior_inventory=swept.range_boundary_inventory,
        liquidity_pools=(),
    )
    reaccepted_candle = _group4_m1(
        17,
        close=99.60,
        high=99.90,
        low=99.30,
    )
    reaccepted = tracker.on_completed_update(
        reaccepted_candle,
        prior_inventory=candidate.range_boundary_inventory,
        liquidity_pools=(),
    )
    mature = next(
        state
        for state in reaccepted.dealing_ranges
        if state.lifecycle is DealingRangeLifecycle.MATURE
    )
    manipulation = reaccepted.manipulations[0]
    assert manipulation.lifecycle is ManipulationLifecycle.REACCEPTED
    assert manipulation.source_kind == "mature_range_boundary"
    return (
        mature,
        manipulation,
        reaccepted.range_boundary_inventory,
        (
            (sweep_candle, swept),
            (candidate_candle, candidate),
            (reaccepted_candle, reaccepted),
        ),
    )


def _range_failed_auction():
    mature, manipulation, inventory, _ = (
        _range_failed_auction_fixture()
    )
    return mature, manipulation, inventory


def _favr_fixture():
    mature, manipulation, inventory, range_auction_updates = (
        _range_failed_auction_fixture()
    )
    reducer = CausalGroup5Reducer(_group5_protocol())
    updates = []

    formation = _m1(
        30,
        open_=100.00,
        high=100.50,
        low=100.00,
        close=100.25,
    )
    fvg = _fvg(
        formation.end,
        direction=Direction.LONG,
        identity="favr-return-fvg",
        lower=99.25,
        upper=99.75,
    )
    output = reducer.on_completed_1m(
        formation,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=1.0,
    )
    updates.append((formation, output))
    assert len(output.entry_locations) == 1

    first_pullback = _m1(
        31,
        open_=100.25,
        high=100.25,
        low=99.50,
        close=99.50,
    )
    output = reducer.on_completed_1m(
        first_pullback,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=1.0,
    )
    updates.append((first_pullback, output))
    reclaim = _m1(
        32,
        open_=99.50,
        high=100.25,
        low=99.50,
        close=100.00,
    )
    output = reducer.on_completed_1m(
        reclaim,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=0.25,
    )
    updates.append((reclaim, output))
    hold = _m1(
        33,
        open_=100.00,
        high=100.25,
        low=99.75,
        close=100.00,
    )
    output = reducer.on_completed_1m(
        hold,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=0.25,
    )
    updates.append((hold, output))

    base = market_observation(asof=hold.end, price=hold.close)
    frames = dict(base.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        cutoff=hold.end,
        dealing_ranges=(mature,),
    )
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        cutoff=hold.end,
        fair_value_gaps=(fvg,),
        structure_breaks=(
            _opposed_m5_bos(
                formation.end,
                displacement_id=fvg.source_displacement_id,
                break_bar_id=fvg.source_candle_ids[-1],
                direction=Direction.LONG,
            ),
        ),
    )
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        cutoff=hold.end,
    )
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        cutoff=hold.end,
    )
    observation = brain_observation_view(replace(
        base,
        frames=frames,
        execution=_execution(hold.end, cost=0.025),
        liquidity_inventory=inventory,
        manipulations=(manipulation,),
        interaction_update=_interaction_update_from_brain_fields(
            entry_locations=output.entry_locations,
            qualified_reacceptances=output.qualified_reacceptances,
            micro_bos_references=output.micro_bos_references,
            path_sequences=output.path_sequences,
        ),
    ))
    return (
        observation,
        mature,
        manipulation,
        fvg,
        tuple(updates),
        inventory,
        range_auction_updates,
    )


def _favr_observation():
    observation, mature, manipulation, _, _, _, _ = _favr_fixture()
    return observation, mature, manipulation


def test_lsr_optional_range_context_enriches_route_without_becoming_gate() -> None:
    base = _lsr_observation()
    tracker, _, mature = _mature_range(_range_auction_protocol())
    range_inventory = tracker.snapshot().range_boundary_inventory
    swept_boundary = next(
        item for item in range_inventory if item.side == "above"
    )
    opposing_boundary = next(
        item for item in range_inventory if item.side == "below"
    )
    manipulation = base.manipulations[0]
    contextual_manipulation = replace(
        manipulation,
        coincident_source_ids=(mature.range_id,),
        crossed_source_ids=(
            manipulation.source_id,
            mature.range_id,
        ),
    )
    consumed_swept_boundary = replace(
        swept_boundary,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=manipulation.swept_at,
        lifecycle_reason="range_boundary_consumed",
    )
    frames = dict(base.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        dealing_ranges=(mature,),
    )
    contextual = replace(
        base,
        frames=frames,
        liquidity_inventory=(
            *base.liquidity_inventory,
            consumed_swept_boundary,
            opposing_boundary,
        ),
        manipulations=(contextual_manipulation,),
    )

    baseline = _brain().update(base).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    hypothesis = _brain().update(contextual).hypotheses[
        "liquidity_sweep_reversal:short"
    ]

    assert baseline.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.phase is baseline.phase
    assert hypothesis.hard_gate_results == baseline.hard_gate_results
    assert all(hypothesis.hard_gate_results.values())
    context_evidence = {
        item.primitive: item.value for item in hypothesis.supporting
    }
    assert context_evidence["mature_range_context_visible"] > 0.0
    assert context_evidence["opposing_range_boundary_visible"] == 1.0
    assert hypothesis.liquidity_route is not None
    route = hypothesis.liquidity_route
    assert route.range_context_id == mature.range_id
    assert route.range_midpoint == mature.midpoint
    assert route.swept_range_boundary_id == swept_boundary.item_id
    assert route.opposing_range_boundary_id == opposing_boundary.item_id
    assert route.context_draw_id == opposing_boundary.item_id
    assert mature.range_id in route.source_path_ids
    assert opposing_boundary.item_id in route.source_path_ids
    assert baseline.liquidity_route is not None
    assert baseline.liquidity_route.range_context_id is None


def test_lsr_optional_range_context_loss_does_not_invalidate_core_episode() -> None:
    base = _lsr_observation()
    tracker, _, mature = _mature_range(_range_auction_protocol())
    range_inventory = tracker.snapshot().range_boundary_inventory
    swept_boundary = next(
        item for item in range_inventory if item.side == "above"
    )
    opposing_boundary = next(
        item for item in range_inventory if item.side == "below"
    )
    manipulation = base.manipulations[0]
    contextual_manipulation = replace(
        manipulation,
        coincident_source_ids=(mature.range_id,),
        crossed_source_ids=(
            manipulation.source_id,
            mature.range_id,
        ),
    )
    consumed_swept_boundary = replace(
        swept_boundary,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=manipulation.swept_at,
        lifecycle_reason="range_boundary_consumed",
    )
    frames = dict(base.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        dealing_ranges=(mature,),
    )
    location = replace(
        base.entry_locations[0],
        lifecycle=EntryLocationLifecycle.APPROACHING,
        state_started_at=base.entry_locations[0].formed_at,
        last_updated_at=base.entry_locations[0].formed_at,
        age_real_1m_bars=0,
        state_duration_real_1m_bars=0,
        current_price=100.50,
        distance_to_zone_points=0.50,
        distance_to_failure_points=1.50,
        first_entered_at=None,
        entry_mode=None,
        contact_reference_price=None,
        first_penetration_fraction=0.0,
        transition_reason="departure_confirmed",
    )
    pool_path = next(
        path
        for path in base.path_sequences
        if path.context_kind == "pool_reversal"
    )
    zone_path = next(
        path
        for path in base.path_sequences
        if path.context_kind == "zone_return"
    )
    waiting_zone_path = replace(
        zone_path,
        lifecycle=PathSequenceLifecycle.ACTIVE,
        state_started_at=zone_path.formed_at,
        last_updated_at=zone_path.formed_at,
        age_real_1m_bars=0,
        state_duration_real_1m_bars=0,
        steps=zone_path.steps[:2],
        ended_at=None,
        transition_reason="context_registered",
    )
    contextual = replace(
        base,
        frames=frames,
        liquidity_inventory=(
            *base.liquidity_inventory,
            consumed_swept_boundary,
            opposing_boundary,
        ),
        manipulations=(contextual_manipulation,),
        entry_locations=(location,),
        qualified_reacceptances=(),
        micro_bos_references=tuple(
            reference
            for reference in base.micro_bos_references
            if reference.context_kind == "pool_reversal"
        ),
        path_sequences=(pool_path, waiting_zone_path),
    )
    brain = _brain()
    first = brain.update(contextual).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    assert first.phase is PlaybookPhase.WAITING_LOCATION
    assert first.liquidity_route is not None
    assert first.liquidity_route.range_context_id == mature.range_id

    later = contextual.asof + pd.Timedelta(minutes=24)
    broken_range = replace(
        mature,
        lifecycle=DealingRangeLifecycle.BROKEN,
        broken_at=later,
        state_started_at=later,
        last_updated_at=later,
        transition_reason="close_beyond_frozen_range",
    )
    later_frames = {
        timeframe: replace(
            frame,
            cutoff=later,
            **(
                {"dealing_ranges": (broken_range,)}
                if timeframe is Timeframe.H1
                else {}
            ),
        )
        for timeframe, frame in contextual.frames.items()
    }
    without_optional_context = replace_market_observation(
        contextual,
        asof=later,
        frames=later_frames,
        execution=_execution(later, cost=0.025),
    )
    updated = brain.update(without_optional_context).hypotheses[
        "liquidity_sweep_reversal:short"
    ]

    assert updated.phase is PlaybookPhase.WAITING_LOCATION
    assert updated.hard_gate_results == first.hard_gate_results
    assert updated.invalidation == first.invalidation
    assert updated.setup_context_id == first.setup_context_id
    assert updated.liquidity_route is not None
    assert updated.liquidity_route.range_context_id is None


def _favr_causal_histories(observation):
    durations = {
        Timeframe.H4: 240,
        Timeframe.H1: 60,
        Timeframe.M5: 5,
        Timeframe.M1: 1,
    }
    histories = {}
    for timeframe, minutes in durations.items():
        end = observation.frame(timeframe).cutoff
        histories[timeframe] = (
            Candle(
                timeframe=timeframe,
                start=end - pd.Timedelta(minutes=minutes),
                end=end,
                open=observation.price,
                high=101.25,
                low=98.50,
                close=observation.price,
                volume=100.0,
                symbol=observation.symbol,
                instrument_id=observation.instrument_id,
                observed_minutes=minutes,
                expected_minutes=minutes,
                complete=True,
                real_minutes=minutes,
            ),
        )
    return histories


def _group5_step_events(update) -> tuple[MarketEvent, ...]:
    events = []
    for sequence_no, (sequence_id, step) in enumerate(
        update.step_transitions
    ):
        source_ids = (
            sequence_id,
            step.step_id,
            step.source_entity_id,
            *(
                ()
                if step.source_event_id is None
                else (step.source_event_id,)
            ),
            *step.predecessor_step_ids,
        )
        events.append(
            MarketEvent(
                event_id=step.step_id,
                kind=EventKind.ENTRY_PATH_STEP,
                observed_at=step.observed_at,
                timeframe=Timeframe.M1,
                side=(
                    "above"
                    if step.direction is Direction.LONG
                    else "below"
                ),
                price=None,
                strength=step.strength,
                source_ids=source_ids,
                details={
                    "sequence_id": sequence_id,
                    "step_id": step.step_id,
                    "kind": step.kind,
                    "source_event_id": step.source_event_id,
                    "source_entity_id": step.source_entity_id,
                    "predecessor_step_ids": step.predecessor_step_ids,
                    "same_clock_relation": step.same_clock_relation,
                    "reason": step.reason,
                },
                direction=step.direction,
                transition_reason=step.reason,
                sequence_no=sequence_no,
            )
        )
    return tuple(events)


def _favr_context_graph_observation(
    asof,
    *,
    price,
    mature,
    inventory,
    manipulations=(),
    additional_ranges=(),
):
    observation = market_observation(asof=asof, price=price)
    frames = {
        timeframe: replace(frame, cutoff=asof)
        for timeframe, frame in observation.frames.items()
    }
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        dealing_ranges=(mature, *additional_ranges),
    )
    return replace(
        observation,
        frames=frames,
        recent_events=(),
        execution=_execution(asof, cost=0.025),
        liquidity_inventory=inventory,
        manipulations=tuple(manipulations),
    )


def _favr_public_graph_observation(
    candle,
    update,
    *,
    mature,
    manipulation,
    inventory,
    fvg,
    additional_ranges=(),
    displacement_id=None,
    displacement_direction=None,
):
    observation = market_observation(
        asof=candle.end,
        price=candle.close,
    )
    frames = dict(observation.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        cutoff=candle.end,
        dealing_ranges=(mature, *additional_ranges),
    )
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        cutoff=candle.end,
        fair_value_gaps=(fvg,),
        structure_breaks=(
            _opposed_m5_bos(
                fvg.confirmed_at,
                displacement_id=fvg.source_displacement_id,
                break_bar_id=fvg.source_candle_ids[-1],
                direction=Direction.LONG,
            ),
        ),
    )
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        cutoff=candle.end,
    )
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        cutoff=candle.end,
    )
    displacement = DisplacementObservation(
        asof=candle.end,
        lifecycle="active",
        current_entity_id=(
            fvg.source_displacement_id
            if displacement_id is None
            else displacement_id
        ),
        current_direction=(
            fvg.direction
            if displacement_direction is None
            else displacement_direction
        ),
        current_state_observed_at=fvg.source_displacement_active_at,
        current_started_at=fvg.source_displacement_started_at,
        current_active_at=fvg.source_displacement_active_at,
        current_last_admitted_at=fvg.confirmed_at,
        current_metrics=(
            ("mean_overlap_ratio", 0.1),
            ("max_overlap_ratio", 0.2),
            ("mean_directional_clv", 0.8),
        ),
        latest_transition=None,
    )
    return replace(
        observation,
        frames=frames,
        recent_events=_group5_step_events(update),
        execution=_execution(candle.end, cost=0.025),
        liquidity_inventory=inventory,
        manipulations=(manipulation,),
        displacement=displacement,
        interaction_update=_interaction_update_from_brain_fields(
            entry_locations=update.entry_locations,
            qualified_reacceptances=update.qualified_reacceptances,
            micro_bos_references=update.micro_bos_references,
            path_sequences=update.path_sequences,
        ),
    )


def _favr_public_scene_graph(
    *,
    omit_precedes: bool = False,
    cross_identities: bool = False,
) -> tuple:
    (
        original_observation,
        mature,
        manipulation,
        fvg,
        updates,
        inventory,
        range_auction_updates,
    ) = _favr_fixture()
    graph = TemporalMarketSceneGraph()
    boundary_id = manipulation.source_inventory_item_id
    graph_inventory = inventory
    additional_ranges = ()
    graph_displacement_id = fvg.source_displacement_id
    if cross_identities:
        decoy_broken_at = mature.mature_at + pd.Timedelta(minutes=1)
        decoy_range = replace(
            mature,
            range_id=f"decoy:{mature.range_id}",
            lifecycle=DealingRangeLifecycle.BROKEN,
            broken_at=decoy_broken_at,
            state_started_at=decoy_broken_at,
            last_updated_at=decoy_broken_at,
            transition_reason="decoy_range_closed",
        )
        additional_ranges = (decoy_range,)
        graph_inventory = tuple(
            replace(
                item,
                source_ids=tuple(
                    decoy_range.range_id
                    if source_id == mature.range_id
                    else source_id
                    for source_id in item.source_ids
                ),
            )
            if item.item_id == boundary_id
            else item
            for item in inventory
        )
        graph_displacement_id = (
            f"decoy:{fvg.source_displacement_id}"
        )
    manipulation_direction = (
        Direction.LONG
        if manipulation.side == "above"
        else Direction.SHORT
    )
    displacement_direction = (
        manipulation_direction if omit_precedes else fvg.direction
    )
    if not cross_identities:
        visible_inventory = tuple(
            replace(
                item,
                lifecycle=LiquidityInventoryLifecycle.VISIBLE,
                consumed_at=None,
                lifecycle_reason=None,
            )
            if item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            else item
            for item in graph_inventory
        )
        graph.update(
            _favr_context_graph_observation(
                mature.mature_at,
                price=mature.midpoint,
                mature=mature,
                inventory=visible_inventory,
            )
        )
        for candle, range_auction_update in range_auction_updates:
            graph.update(
                _favr_context_graph_observation(
                    candle.end,
                    price=candle.close,
                    mature=mature,
                    inventory=(
                        range_auction_update.range_boundary_inventory
                    ),
                    manipulations=range_auction_update.manipulations,
                )
            )
    else:
        graph.update(
            _favr_context_graph_observation(
                manipulation.reaccepted_at,
                price=manipulation.reentry_price,
                mature=mature,
                inventory=graph_inventory,
                manipulations=(manipulation,),
                additional_ranges=additional_ranges,
            )
        )
    observations = []
    for candle, update in updates:
        observation = _favr_public_graph_observation(
            candle,
            update,
            mature=mature,
            manipulation=manipulation,
            inventory=graph_inventory,
            fvg=fvg,
            additional_ranges=additional_ranges,
            displacement_id=graph_displacement_id,
            displacement_direction=displacement_direction,
        )
        graph.update(observation)
        observations.append(observation)
    return (
        graph,
        brain_observation_view(observations[-1]),
        original_observation,
        mature,
        manipulation,
    )


def test_favr_full_causal_chain_is_implemented_but_runtime_parked() -> None:
    observation, mature, manipulation = _favr_observation()
    belief = PlaybookBrain(
        registry=load_playbook_registry(
            ROOT / "configs/playbooks.json"
        )
    ).update(observation)
    hypothesis = belief.hypotheses[
        "failed_auction_value_return:long"
    ]

    assert hypothesis.phase is PlaybookPhase.INACTIVE
    assert hypothesis.plan is None
    assert hypothesis.sequence is not None
    assert hypothesis.sequence.complete
    assert all(hypothesis.hard_gate_results.values())
    assert hypothesis.context_id == mature.range_id
    assert hypothesis.episode_id == manipulation.manipulation_id
    assert hypothesis.draw_selection is not None
    assert (
        hypothesis.draw_selection.lifecycle
        is LiquidityInventoryLifecycle.TARGETED
    )


def test_favr_stable_context_identity_keeps_its_generic_contract() -> None:
    observation, _, _ = _favr_observation()
    brain = _unparked_favr_brain()
    evaluation = _typed_favr(
        observation,
        Direction.LONG,
        brain.registry.for_playbook(
            Playbook.FAILED_AUCTION_VALUE_RETURN
        ),
        None,
        brain.config,
    )

    assert evaluation.context_identity is not None
    assert evaluation.sequence_signals
    assert _stable_context_thesis_id(
        "epoch:favr-generic-context",
        Playbook.FAILED_AUCTION_VALUE_RETURN,
        Direction.LONG,
        evaluation,
        None,
    ) is not None


def test_favr_scene_graph_keeps_precedes_diagnostic_without_action_authority() -> None:
    graph, observation, _, mature, manipulation = (
        _favr_public_scene_graph()
    )
    assert graph.last_asof == observation.asof
    manipulation_node = next(
        node
        for node in graph.nodes
        if node.kind == "manipulation"
        and node.entity_id == manipulation.manipulation_id
    )
    manipulation_history = graph.node_history(
        manipulation_node.node_id
    )
    assert manipulation_history[0].lifecycle == "swept"
    assert manipulation_history[0].observed_at == manipulation.swept_at
    assert manipulation_history[-1].lifecycle == "reaccepted"
    assert (
        manipulation_history[-1].observed_at
        == manipulation.reaccepted_at
    )
    location = observation.entry_locations[0]
    path = next(
        item
        for item in observation.path_sequences
        if item.context_id == location.location_id
    )
    first_pullback = next(
        step for step in path.steps if step.kind == "first_pullback"
    )
    trigger = next(
        step
        for step in path.steps
        if step.kind in {"reacceptance_held", "micro_bos_confirmed"}
    )
    boundary_id = manipulation.source_inventory_item_id

    assert graph.has_direct_relation(
        (boundary_id,),
        SceneEdgeKind.SOURCED_FROM,
        (mature.range_id,),
        source_kind="liquidity",
        target_kind="range",
    )
    assert graph.has_direct_relation(
        (manipulation.manipulation_id,),
        SceneEdgeKind.SWEEPS,
        (boundary_id,),
        source_kind="manipulation",
        target_kind="liquidity",
    )
    assert graph.has_direct_relation(
        (manipulation.manipulation_id,),
        SceneEdgeKind.PRECEDES,
        (location.source_displacement_id,),
        source_kind="manipulation",
        target_kind="displacement",
    )
    assert graph.has_direct_relation(
        (location.source_displacement_id,),
        SceneEdgeKind.CREATES,
        (location.source_zone_id,),
        source_kind="displacement",
        target_kind=location.source_zone_kind,
    )
    assert graph.has_direct_relation(
        (location.location_id,),
        SceneEdgeKind.RETURNS_TO,
        (location.source_zone_id,),
        source_kind="entry_location",
        target_kind=location.source_zone_kind,
    )
    for step in (first_pullback, trigger):
        assert graph.has_direct_relation(
            (step.step_id,),
            SceneEdgeKind.SOURCED_FROM,
            (path.sequence_id,),
            source_kind="path_step",
            target_kind="path_sequence",
        )
        step_node = next(
            node
            for node in graph.nodes
            if node.kind == "path_step"
            and node.entity_id == step.step_id
        )
        assert dict(step_node.semantic_attributes)[
            "path_step_kind"
        ] == step.kind
        assert step.source_entity_id in step_node.source_ids
        assert set(step.predecessor_step_ids).issubset(step_node.source_ids)

    hypothesis = _unparked_favr_brain().update(
        observation,
        scene_graph=graph,
    ).hypotheses["failed_auction_value_return:long"]
    assert hypothesis.phase is not PlaybookPhase.EXECUTABLE
    assert hypothesis.sequence is not None
    assert not hypothesis.sequence.complete
    assert not hypothesis.hard_gate_results[
        "opposite_displacement_zone_inside_range"
    ]
    assert not hypothesis.hard_gate_results[
        "first_pullback_to_frozen_zone"
    ]
    assert not hypothesis.hard_gate_results["aligned_entry_trigger"]
    assert hypothesis.plan is None


def test_favr_scene_graph_missing_middle_relation_fails_closed() -> None:
    graph, observation, _, _, _ = _favr_public_scene_graph(
        omit_precedes=True,
    )

    hypothesis = _unparked_favr_brain().update(
        observation,
        scene_graph=graph,
    ).hypotheses["failed_auction_value_return:long"]

    assert hypothesis.phase is not PlaybookPhase.EXECUTABLE
    assert hypothesis.plan is None
    assert not hypothesis.hard_gate_results[
        "opposite_displacement_zone_inside_range"
    ]
    assert not hypothesis.hard_gate_results[
        "first_pullback_to_frozen_zone"
    ]
    assert not hypothesis.hard_gate_results["aligned_entry_trigger"]


def test_favr_scene_graph_rejects_crossed_range_and_zone_identities() -> None:
    graph, _, observation, mature, manipulation = (
        _favr_public_scene_graph(
            cross_identities=True,
        )
    )

    # A relation-agnostic BFS can still traverse the crossed decoys.  The
    # FAVR connector must require the exact selected identities instead.
    assert graph.find_path(
        (mature.range_id,),
        (
            manipulation.manipulation_id,
            manipulation.source_inventory_item_id,
        ),
    )
    hypothesis = _unparked_favr_brain().update(
        observation,
        scene_graph=graph,
    ).hypotheses["failed_auction_value_return:long"]

    assert hypothesis.phase is not PlaybookPhase.EXECUTABLE
    assert hypothesis.plan is None
    assert not hypothesis.hard_gate_results[
        "range_boundary_sweep_reaccepted"
    ]


def test_unparked_favr_freezes_range_sweep_entry_and_opposite_draw() -> None:
    observation, mature, manipulation = _favr_observation()
    belief = _unparked_favr_brain().update(observation)
    hypothesis = belief.hypotheses[
        "failed_auction_value_return:long"
    ]

    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.plan is not None
    plan = hypothesis.plan
    context = plan.range_auction
    assert context is not None
    assert (
        context.lower_bound,
        context.upper_bound,
        context.midpoint,
        context.value_price,
    ) == (
        mature.lower_bound,
        mature.upper_bound,
        mature.midpoint,
        mature.value_price,
    )
    assert context.manipulation_id == manipulation.manipulation_id
    assert context.manipulation_extreme == manipulation.sweep_extreme
    assert (
        context.reentry_candidate_at
        == manipulation.reentry_candidate_at
    )
    assert context.reentered_at == manipulation.reaccepted_at
    assert context.reentry_price == manipulation.reentry_price
    assert plan.invalidation.price == manipulation.sweep_extreme
    assert plan.targets[0].price == mature.upper_bound
    assert plan.planned_entry == 99.50
    assert plan.planned_entry != observation.price
    assert plan.draw_selection is not None
    assert plan.draw_selection.source_kind == "range_boundary"
    assert (
        plan.draw_selection.selection_reason
        == "failed_auction_value_return:"
        "frozen_opposite_range_boundary"
    )

    decision = UtilityDecisionLayer().decide(
        observation,
        graph_free_action_belief(belief),
    )
    assert decision.selected_action.value == "abstain"
    enter = next(
        item
        for item in decision.utilities
        if item.action.value == "enter"
        and item.hypothesis_key == hypothesis.key
    )
    assert "playbook_not_action_calibrated" in enter.reason
    assert "calibration_not_ready" in enter.reason
    risk = StructuralRiskEngine().review(decision, observation)
    assert risk.passed
    assert risk.final_action.value == "abstain"
    assert risk.frozen_thesis is None


def test_favr_terminalizes_late_mss_and_bound_location_failure() -> None:
    observation, _, _ = _favr_observation()
    key = "failed_auction_value_return:long"

    frames = dict(observation.frames)
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        structure_breaks=(
            replace(
                frames[Timeframe.M5].structure_breaks[0],
                resolved_at=observation.asof,
            ),
        ),
    )
    late = _unparked_favr_brain().update(
        replace(observation, frames=frames)
    ).hypotheses[key]
    assert late.phase is PlaybookPhase.INVALIDATED
    assert late.terminal_reason == "mss_confirmed_after_first_pullback"

    brain = _unparked_favr_brain()
    active = brain.update(observation).hypotheses[key]
    assert active.entry_location_id is not None
    location = observation.entry_locations[0]
    failed_location = replace(
        location,
        lifecycle=EntryLocationLifecycle.LEFT,
        state_started_at=observation.asof,
        last_updated_at=observation.asof,
        rejected_at=None,
        left_at=observation.asof,
        transition_reason="source_invalidated",
    )
    failed = brain.update(
        replace(observation, entry_locations=(failed_location,))
    ).hypotheses[key]
    assert failed.phase is PlaybookPhase.INVALIDATED
    assert failed.terminal_reason == "frozen_entry_zone_missing"
    assert failed.episode_id == active.episode_id


def test_favr_missing_frozen_sources_closes_episode_instead_of_resetting() -> None:
    observation, _, _ = _favr_observation()
    brain = _unparked_favr_brain()
    first = brain.update(observation).hypotheses[
        "failed_auction_value_return:long"
    ]
    assert first.phase is PlaybookPhase.EXECUTABLE

    next_clock = observation.asof + pd.Timedelta(minutes=1)
    frames = dict(observation.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        cutoff=next_clock,
        dealing_ranges=(),
    )
    for timeframe in (Timeframe.H4, Timeframe.M5, Timeframe.M1):
        frames[timeframe] = replace(
            frames[timeframe],
            cutoff=next_clock,
        )
    missing = replace_market_observation(
        observation,
        asof=next_clock,
        frames=frames,
        execution=_execution(next_clock, cost=0.025),
        liquidity_inventory=(),
        manipulations=(),
    )
    terminal = brain.update(missing).hypotheses[
        "failed_auction_value_return:long"
    ]

    assert terminal.phase is PlaybookPhase.INVALIDATED
    assert terminal.terminal_at == next_clock
    assert terminal.terminal_reason == "frozen_dealing_range_missing"
    assert first.context_id in terminal.terminal_source_ids
    assert first.episode_id in terminal.terminal_source_ids


def test_favr_decision_visual_keeps_frozen_geometry_visible(
    tmp_path,
) -> None:
    observation, mature, manipulation = _favr_observation()
    belief = _unparked_favr_brain().update(observation)
    key = "failed_auction_value_return:long"
    hypothesis = belief.hypotheses[key]
    assert hypothesis.plan is not None
    plan = hypothesis.plan
    decision = UtilityDecisionLayer().decide(observation, belief)
    risk = StructuralRiskEngine().review(decision, observation)
    assert risk.passed
    snapshot = EngineSnapshot(
        observation=observation,
        belief=belief,
        decision=decision,
        risk=risk,
    )

    artifact = DecisionVisualizer().render_decision(
        snapshot,
        _favr_causal_histories(observation),
        tmp_path / "favr-decision.png",
    )

    assert artifact.path.is_file()
    assert artifact.path.stat().st_size > 0
    assert artifact.maximum_market_time <= observation.asof
    assert artifact.hypothesis_key == key
    assert artifact.setup_id == plan.setup_id
    assert artifact.entry_location_id == plan.entry_location_id
    assert artifact.entry_path_id == plan.entry_path_id
    assert plan.range_auction is not None
    assert plan.range_auction.range_id == mature.range_id
    assert plan.range_auction.manipulation_id == manipulation.manipulation_id
    assert plan.invalidation.source_level_id == manipulation.manipulation_id
    assert plan.draw_selection is not None
    assert plan.draw_selection.draw_id == plan.selected_draw_id
