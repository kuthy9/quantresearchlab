from __future__ import annotations

from dataclasses import replace

import pandas as pd

from smc_trader.dol_ranking import (
    _feature_values,
    load_dol_ranking_protocol,
)
from smc_trader.foundation_adapter import CanonicalFoundationAdapter
from smc_trader.market_state import (
    DOLCandidateView,
    MarketSnapshot,
    MarketSnapshotPublisher,
    RelationResolver,
    SessionState,
    foundation_dol_timeframe_states,
)
from smc_trader.model import (
    Direction,
    EventKind,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    OpenMarketThesis,
    SMC_SEMANTIC_VERSION,
    ThesisEvidenceState,
    Timeframe,
)
from smc_trader.playbooks import _shadow_dol_facts, _visible_levels
from smc_trader.scene_graph import (
    StructuralScale,
    TemporalMarketSceneGraph,
    current_dol_inventory,
    dol_level_terminal_sources,
    update_global_market_context,
)
from smc_trader.semantic_foundation import FoundationProjectionReducer
from smc_trader.semantic_lifecycle import LiquidityLevelLifecycle

from .helpers import market_observation
from .test_foundation_adapter import (
    TICK,
    _bar,
    _clock,
    _cross_level,
    _seed_level,
)


def _source_item(
    base,
    *,
    source_identity: str = "above-level",
    kind: str = "swing",
    structural_rank: str = "external",
) -> LiquidityInventoryItem:
    template = next(
        item
        for item in base.liquidity_inventory
        if item.item_id == "above-level"
    )
    return replace(
        template,
        item_id=source_identity,
        kind=kind,
        price=100.0,
        lower_bound=100.0,
        upper_bound=100.0,
        source_ids=(f"source:{source_identity}",),
        structural_rank=structural_rank,
    )


def _consumed(
    item: LiquidityInventoryItem,
    *,
    minute: int,
    reason: str = "swing_swept",
) -> LiquidityInventoryItem:
    return replace(
        item,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=_clock(minute),
        lifecycle_reason=reason,
    )


def _observation(
    base,
    *,
    minute: int,
    price: float,
    projection,
    source_identity: str,
    source_item: LiquidityInventoryItem | None,
    pool_state: LiquidityPoolState | None = None,
    real_bar_ordinals=None,
):
    asof = _clock(minute)
    frames = {
        timeframe: replace(frame, cutoff=asof)
        for timeframe, frame in base.frames.items()
    }
    if pool_state is not None:
        frames[pool_state.timeframe] = replace(
            frames[pool_state.timeframe],
            liquidity_pools=(pool_state,),
        )
    inventory = tuple(
        item
        for item in base.liquidity_inventory
        if item.item_id not in {"above-level", source_identity}
    )
    if source_item is not None:
        inventory = (*inventory, source_item)

    publisher = MarketSnapshotPublisher(
        semantic_registry_identity="foundation-dol-test"
    )
    states = {
        timeframe: publisher._timeframe_state(
            frame,
            asof=asof,
            price=price,
            inventory=inventory,
            displacement=None,
            events=(),
            anomalies=(),
        )
        for timeframe, frame in frames.items()
    }
    templates = (
        {}
        if source_item is None
        else {
            source_identity: DOLCandidateView(
                candidate_id=source_identity,
                timeframe=source_item.timeframe,
                side=source_item.side,
                price=source_item.price,
                source_kind=source_item.kind,
                rank=source_item.structural_rank,
                strength=source_item.strength,
                lifecycle=LiquidityInventoryLifecycle.VISIBLE.value,
                age_bars=source_item.age_bars,
            )
        }
    )
    states = foundation_dol_timeframe_states(
        projection,
        states=states,
        price=price,
        candidate_templates=templates,
        real_bar_ordinals=real_bar_ordinals,
    )
    relations = RelationResolver(
        edges=MarketSnapshotPublisher._RELATION_EDGES
    ).resolve(states, price=price, asof=asof)
    session = SessionState(
        session_id="foundation-dol-session",
        name="regular",
        phase="open",
        session_open=price,
        session_high=price + 1.0,
        session_low=price - 1.0,
        opening_range_high=None,
        opening_range_low=None,
        prior_day_high=None,
        prior_day_low=None,
        overnight_high=None,
        overnight_low=None,
        elapsed_minutes=minute + 1,
        realized_volatility=0.0,
        relative_volume=None,
        known_at=asof,
        data_complete=True,
    )
    snapshot = MarketSnapshot(
        asof=asof,
        symbol=base.symbol,
        instrument_id=base.instrument_id,
        price=price,
        semantic_version=SMC_SEMANTIC_VERSION,
        semantic_registry_identity="foundation-dol-test",
        timeframe_states=states,
        relations=relations,
        session=session,
        events_this_update=(),
        labels=(),
        foundation=projection,
    )
    return replace(
        base,
        asof=asof,
        price=price,
        frames=frames,
        recent_events=(),
        semantic_events_this_update=(),
        liquidity_inventory=inventory,
        liquidity_pool_states=(
            () if pool_state is None else (pool_state,)
        ),
        market_snapshot=snapshot,
    )


def _thesis(context, *, asof, draw_id: str) -> OpenMarketThesis:
    root_id = "foundation-dol-root"
    evidence = ThesisEvidenceState(
        lifecycle="active",
        revision_id="foundation-dol-thesis-revision",
        changed_at=asof,
        supporting_event_ids=(root_id,),
        new_supporting_event_ids=(root_id,),
    )
    return OpenMarketThesis(
        thesis_id="foundation-dol-thesis",
        root_id=root_id,
        market_epoch_id=context.market_epoch_id,
        formed_at=asof,
        updated_at=asof,
        direction=Direction.LONG,
        source_timeframe=Timeframe.M1,
        structural_scale="internal",
        mechanism="directional_displacement",
        authority_relation="aligned",
        mechanism_event_ids=(root_id,),
        draw_candidate_ids=(draw_id,),
        evidence_state=evidence,
    )


def _canonical_nodes(graph, *, level_id: str):
    return tuple(
        node
        for node in graph.nodes
        if node.kind == "liquidity" and node.entity_id == level_id
    )


def _real_bar_ordinals(adapter: CanonicalFoundationAdapter):
    return {
        clock.timeframe: clock.count
        for clock in adapter.lifecycle.real_bar_clocks
    }


def test_sweep_departure_generation_two_reenters_graph_context_and_brain() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level_event = _seed_level(
        adapter,
        label="above-level",
        price=100.0,
    )
    level_id = adapter.lifecycle.levels[-1].level_id
    generation_one_id = adapter.lifecycle.levels[-1].active_generation_id
    assert generation_one_id is not None

    base = market_observation(asof=_clock(0), price=99.0)
    source = _source_item(base)
    observation0 = _observation(
        base,
        minute=0,
        price=99.0,
        projection=adapter.projection,
        source_identity="above-level",
        source_item=source,
    )
    graph = TemporalMarketSceneGraph()
    delta0 = graph.update(observation0)
    context0 = update_global_market_context(
        None,
        observation0,
        delta0,
        graph,
    )
    assert level_id in context0.external_draw_candidates["above"]

    _cross_level(
        adapter,
        level_event,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    consumed_source = _consumed(source, minute=1)
    observation1 = _observation(
        base,
        minute=1,
        price=99.75,
        projection=adapter.projection,
        source_identity="above-level",
        source_item=consumed_source,
    )
    delta1 = graph.update(observation1)
    context1 = update_global_market_context(
        context0,
        observation1,
        delta1,
        graph,
    )
    assert level_id not in context1.external_draw_candidates["above"]
    assert dol_level_terminal_sources(observation1, level_id)

    adapter.consume(_bar(2, high=100.0, low=99.0, close=99.75))
    level = adapter.lifecycle.levels[-1]
    generation_two_id = level.active_generation_id
    assert level.lifecycle is LiquidityLevelLifecycle.REARMED
    assert generation_two_id is not None
    assert generation_two_id != generation_one_id

    observation2 = _observation(
        base,
        minute=2,
        price=99.75,
        projection=adapter.projection,
        source_identity="above-level",
        source_item=consumed_source,
    )
    snapshot_ids = {
        candidate.candidate_id
        for candidate in observation2.market_snapshot.timeframe_states[
            Timeframe.H1
        ].liquidity.candidates
    }
    assert level_id in snapshot_ids
    delta2 = graph.update(observation2)
    context2 = update_global_market_context(
        context1,
        observation2,
        delta2,
        graph,
    )
    assert level_id in context2.external_draw_candidates["above"]
    assert level_id in {item.level_id for item in _visible_levels(observation2)}
    assert not dol_level_terminal_sources(observation2, level_id)

    nodes = {
        dict(node.semantic_attributes)[
            "foundation_interaction_generation_id"
        ]: node
        for node in _canonical_nodes(graph, level_id=level_id)
    }
    assert nodes[generation_one_id].lifecycle == "consumed"
    assert nodes[generation_two_id].lifecycle == "visible"
    assert nodes[generation_two_id].structural_scale is StructuralScale.EXTERNAL

    thesis = _thesis(context2, asof=observation2.asof, draw_id=level_id)
    context2 = replace(context2, open_market_theses=(thesis,))
    facts, _, exclusions = _shadow_dol_facts(observation2, context2)
    fact = next(
        item
        for item in facts[Direction.LONG.value]
        if item.candidate_id == level_id
    )
    assert level_event.event_id in fact.source_ids
    assert generation_two_id in fact.source_ids
    assert "bar:1m:2" in fact.source_ids
    assert all(
        not source_id.startswith("foundation-record:")
        and "foundation_state_changed" not in source_id
        for source_id in fact.source_ids
    )
    assert (level_id, "exact_inventory_join_missing") not in exclusions[
        Direction.LONG.value
    ]

    replayed = FoundationProjectionReducer.replay(adapter.projection.records)
    replay_observation = _observation(
        base,
        minute=2,
        price=99.75,
        projection=replayed,
        source_identity="above-level",
        source_item=consumed_source,
    )
    replay_graph = TemporalMarketSceneGraph()
    replay_delta = replay_graph.update(replay_observation)
    replay_context = update_global_market_context(
        None,
        replay_observation,
        replay_delta,
        replay_graph,
    )
    replay_context = replace(
        replay_context,
        open_market_theses=(
            _thesis(
                replay_context,
                asof=replay_observation.asof,
                draw_id=level_id,
            ),
        ),
    )
    replay_facts, _, replay_exclusions = _shadow_dol_facts(
        replay_observation,
        replay_context,
    )
    replay_fact = next(
        item
        for item in replay_facts[Direction.LONG.value]
        if item.candidate_id == level_id
    )
    streamed_view = next(
        item
        for item in current_dol_inventory(observation2)
        if item.item_id == level_id
    )
    replay_view = next(
        item
        for item in current_dol_inventory(replay_observation)
        if item.item_id == level_id
    )
    assert replayed == adapter.projection
    assert replay_view == streamed_view
    assert replay_context.external_draw_candidates == (
        context2.external_draw_candidates
    )
    assert replay_fact == fact
    assert replay_exclusions == exclusions


def test_acceptance_retirement_cannot_reenter_any_dol_consumer() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level_event = _seed_level(
        adapter,
        label="above-level",
        price=100.0,
    )
    level_id = adapter.lifecycle.levels[-1].level_id
    base = market_observation(asof=_clock(0), price=99.0)
    # Keep the compatibility source deliberately visible after acceptance.
    # The canonical terminal lifecycle, not a convenient legacy disappearance,
    # must suppress it.
    visible_source = _source_item(base)
    observation0 = _observation(
        base,
        minute=0,
        price=99.0,
        projection=adapter.projection,
        source_identity="above-level",
        source_item=visible_source,
    )
    graph = TemporalMarketSceneGraph()
    delta0 = graph.update(observation0)
    context0 = update_global_market_context(
        None,
        observation0,
        delta0,
        graph,
    )
    assert level_id in context0.external_draw_candidates["above"]

    _cross_level(
        adapter,
        level_event,
        crossed_minute=1,
        resolved_minute=2,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
    )
    assert (
        adapter.lifecycle.levels[-1].lifecycle
        is LiquidityLevelLifecycle.RETIRED
    )
    observation2 = _observation(
        base,
        minute=2,
        price=100.5,
        projection=adapter.projection,
        source_identity="above-level",
        source_item=visible_source,
    )
    delta2 = graph.update(observation2)
    context2 = update_global_market_context(
        context0,
        observation2,
        delta2,
        graph,
    )
    assert level_id not in context2.external_draw_candidates["above"]
    assert level_id not in {
        item.item_id for item in current_dol_inventory(observation2)
    }
    assert level_id not in {item.level_id for item in _visible_levels(observation2)}
    assert dol_level_terminal_sources(observation2, level_id)
    assert dol_level_terminal_sources(observation2, "above-level")

    adapter.consume(_bar(3, high=101.0, low=99.0, close=100.5))
    observation3 = _observation(
        base,
        minute=3,
        price=100.5,
        projection=adapter.projection,
        source_identity="above-level",
        source_item=visible_source,
    )
    delta3 = graph.update(observation3)
    context3 = update_global_market_context(
        context2,
        observation3,
        delta3,
        graph,
    )
    assert level_id not in context3.external_draw_candidates["above"]
    assert adapter.projection.active_dol_candidate_ids == ()


def test_formed_pool_never_materializes_without_its_visible_source() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _seed_level(
        adapter,
        label="pool:formed-source",
        source_kind="formed_liquidity_pool",
        price=100.0,
    )
    level_id = adapter.lifecycle.levels[-1].level_id
    base = market_observation(asof=_clock(0), price=99.0)
    members = ("source:below-level", "source:above-level")
    formed_at = _clock(0) - pd.Timedelta(hours=3)
    confirmed_at = _clock(0) - pd.Timedelta(hours=2)
    pool = LiquidityPoolState(
        pool_id="formed-source",
        timeframe=Timeframe.H1,
        side="above",
        lower_bound=99.75,
        upper_bound=100.0,
        midpoint=99.875,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        lifecycle=LiquidityPoolLifecycle.FORMED,
        member_swing_ids=members,
        touch_times=(formed_at, confirmed_at),
        age_bars=1,
        strength=0.75,
        total_touch_count=2,
    )
    visible_source = replace(
        next(
            item
            for item in base.liquidity_inventory
            if item.item_id == "above-level"
        ),
        item_id="pool:formed-source",
        kind="equal_highs",
        price=100.0,
        lower_bound=99.75,
        upper_bound=100.0,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        source_ids=members,
        structural_rank="external",
    )
    observation0 = _observation(
        base,
        minute=0,
        price=99.0,
        projection=adapter.projection,
        source_identity="pool:formed-source",
        source_item=visible_source,
        pool_state=pool,
    )
    graph = TemporalMarketSceneGraph()
    delta0 = graph.update(observation0)
    context0 = update_global_market_context(
        None,
        observation0,
        delta0,
        graph,
    )
    assert level_id in context0.external_draw_candidates["above"]

    # Foundation remains ACTIVE, but a formed-pool projection cannot become a
    # standalone candidate after its exact compatibility source is consumed.
    consumed_source = _consumed(
        visible_source,
        minute=1,
        reason="pool_swept",
    )
    observation1 = _observation(
        base,
        minute=1,
        price=99.75,
        projection=adapter.projection,
        source_identity="pool:formed-source",
        source_item=consumed_source,
        pool_state=pool,
    )
    delta1 = graph.update(observation1)
    context1 = update_global_market_context(
        context0,
        observation1,
        delta1,
        graph,
    )
    assert adapter.projection.active_dol_candidate_ids == (level_id,)
    assert level_id not in {
        item.item_id for item in current_dol_inventory(observation1)
    }
    assert level_id not in context1.external_draw_candidates["above"]
    assert level_id not in {item.level_id for item in _visible_levels(observation1)}


def test_rearmed_internal_level_is_not_promoted_by_h1_timeframe() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level_event = _seed_level(
        adapter,
        label="above-level",
        price=100.0,
    )
    _cross_level(
        adapter,
        level_event,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    adapter.consume(_bar(2, high=100.0, low=99.0, close=99.75))
    level = adapter.lifecycle.levels[-1]
    assert level.active_generation_id is not None

    base = market_observation(asof=_clock(0), price=99.0)
    internal_source = _source_item(base, structural_rank="internal")
    observation = _observation(
        base,
        minute=2,
        price=99.75,
        projection=adapter.projection,
        source_identity="above-level",
        source_item=_consumed(internal_source, minute=1),
    )
    graph = TemporalMarketSceneGraph()
    delta = graph.update(observation)
    context = update_global_market_context(None, observation, delta, graph)
    node = next(
        item
        for item in _canonical_nodes(graph, level_id=level.level_id)
        if dict(item.semantic_attributes).get(
            "foundation_interaction_generation_id"
        )
        == level.active_generation_id
    )
    assert node.structural_scale is StructuralScale.INTERNAL
    assert level.level_id in {
        item.item_id for item in current_dol_inventory(observation)
    }
    assert level.level_id not in context.external_draw_candidates["above"]


def test_rearmed_generation_age_advances_on_exact_real_bars_and_replays() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    level_events, level_event = _seed_level(
        adapter,
        label="above-level",
        price=100.0,
    )
    crossing_events, _ = _cross_level(
        adapter,
        level_event,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    departure = _bar(2, high=100.0, low=99.0, close=99.75)
    bar3 = _bar(3, high=99.75, low=99.0, close=99.5)
    bar4 = _bar(4, high=99.75, low=99.0, close=99.5)
    adapter.consume(departure)
    adapter.consume(bar3)
    adapter.consume(bar4)
    generation = adapter.lifecycle.interaction(
        adapter.lifecycle.levels[-1].active_generation_id
    )
    current_ordinal = _real_bar_ordinals(adapter)[Timeframe.M1]
    assert current_ordinal - generation.armed_real_bar_ordinal == 2

    base = market_observation(asof=_clock(0), price=99.0)
    source = _consumed(_source_item(base), minute=1)
    observation = _observation(
        base,
        minute=4,
        price=99.5,
        projection=adapter.projection,
        source_identity="above-level",
        source_item=source,
        real_bar_ordinals=_real_bar_ordinals(adapter),
    )
    level_id = adapter.lifecycle.levels[-1].level_id
    view = next(
        item
        for item in current_dol_inventory(observation)
        if item.item_id == level_id
    )
    assert view.age_bars == 2

    graph = TemporalMarketSceneGraph()
    delta = graph.update(observation)
    context = update_global_market_context(None, observation, delta, graph)
    context = replace(
        context,
        open_market_theses=(
            _thesis(context, asof=observation.asof, draw_id=level_id),
        ),
    )
    facts, _, _ = _shadow_dol_facts(observation, context)
    fact = next(
        item
        for item in facts[Direction.LONG.value]
        if item.candidate_id == level_id
    )
    assert fact.age_real_completed_bars == 2
    protocol = load_dol_ranking_protocol("configs/path_hypotheses.json")
    aged_features = dict(
        _feature_values(
            protocol,
            candidate=fact,
            distance_points=fact.target_price - observation.price,
            hard_count=0,
            soft_count=0,
        )
    )
    fresh_features = dict(
        _feature_values(
            protocol,
            candidate=replace(fact, age_real_completed_bars=0),
            distance_points=fact.target_price - observation.price,
            hard_count=0,
            soft_count=0,
        )
    )
    assert aged_features["age_freshness"] < fresh_features["age_freshness"]

    replayed = CanonicalFoundationAdapter.replay(
        (
            *level_events,
            *crossing_events,
            departure,
            bar3,
            bar4,
        ),
        tick_size=TICK,
    )
    replay_observation = _observation(
        base,
        minute=4,
        price=99.5,
        projection=replayed.projection,
        source_identity="above-level",
        source_item=source,
        real_bar_ordinals=_real_bar_ordinals(replayed),
    )
    replay_view = next(
        item
        for item in current_dol_inventory(replay_observation)
        if item.item_id == level_id
    )
    assert replayed.projection == adapter.projection
    assert _real_bar_ordinals(replayed) == _real_bar_ordinals(adapter)
    assert replay_view == view
