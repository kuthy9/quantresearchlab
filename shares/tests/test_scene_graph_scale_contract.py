from __future__ import annotations

from dataclasses import replace
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from eyes.core.causal import CausalMarketReader
from shares.core.engine import ContinuousSMCEngine
from contract.market import (
    Bar,
    Direction,
    Playbook,
    PlaybookPhase,
    Timeframe,
)
from contract.eye import (
    CandleStructureState,
    EventKind,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    MarketEvent,
)
from contract.brain import (
    GlobalConflictRole,
    SequenceStepState,
)
from eyes.core.observation import CausalObserver, ObserverConfig
from shares.core.scene_graph import (
    _open_thesis_closure,
    EvidenceStatus,
    FocusState,
    FocusedObservation,
    ScaleRole,
    ScaleSpec,
    SceneEdge,
    SceneEdgeKind,
    SceneGraphDelta,
    SceneNode,
    StructuralScale,
    TemporalMarketSceneGraph,
    build_hypothesis_states,
    parse_scale_specs,
    scale_registry_id,
    select_focus,
    update_global_market_context,
)

from shares.tests.helpers import (
    CORE_TEST_SCALE_SPECS,
    MODEL_SCALE_SPECS,
    market_observation,
    replace_market_observation,
)


TZ = "America/New_York"


def _clock(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=TZ)


def _node(
    identity: str,
    kind: str,
    timeframe: Timeframe,
    observed_at: pd.Timestamp,
    *,
    formed_at: pd.Timestamp | None = None,
    direction: Direction | None = Direction.LONG,
    lifecycle: str = "confirmed",
    status: EvidenceStatus = EvidenceStatus.CONFIRMED,
    source_ids: tuple[str, ...] = (),
    entity_id: str | None = None,
    bounds: tuple[float, float] | None = (99.0, 101.0),
    semantic_attributes: tuple[tuple[str, str], ...] = (),
) -> SceneNode:
    formed = formed_at or observed_at
    return SceneNode(
        node_id=identity,
        kind=kind,
        timeframe=timeframe.value,
        structural_scale=(
            StructuralScale.EXTERNAL
            if timeframe in {Timeframe.H4, Timeframe.H1}
            else StructuralScale.INTERMEDIATE
            if timeframe is Timeframe.M15
            else StructuralScale.INTERNAL
        ),
        direction=direction,
        formed_at=formed,
        confirmed_at=(
            None
            if lifecycle in {"forming", "started", "pending", "ambiguous"}
            else formed
        ),
        observed_at=observed_at,
        lifecycle=lifecycle,
        price_bounds=bounds,
        invalidation_rule="frozen_test_rule",
        ambiguity_state=status,
        source_ids=source_ids,
        entity_id=entity_id,
        semantic_attributes=semantic_attributes,
    )


def test_current_kind_cache_is_revision_bounded_and_pickle_safe() -> None:
    graph = TemporalMarketSceneGraph()
    asof = _clock("2025-01-06 10:00")
    first_node = _node(
        "cache-structure-1",
        "structure",
        Timeframe.H1,
        asof,
        entity_id="cache-structure-1",
    )
    graph.add_node(first_node)

    first = graph._current_kind_nodes("structure")
    first_context = graph._current_nonterminal_kind_nodes("structure")
    assert graph._current_kind_nodes("structure") is first
    assert graph._current_nonterminal_kind_nodes("structure") is first_context

    graph.add_node(
        _node(
            "cache-structure-2",
            "structure",
            Timeframe.H1,
            asof + pd.Timedelta(minutes=1),
            entity_id="cache-structure-2",
        )
    )
    revised = graph._current_kind_nodes("structure")
    assert revised is not first
    assert {node.entity_id for node in revised} == {
        "cache-structure-1",
        "cache-structure-2",
    }

    resumed = pickle.loads(pickle.dumps(graph))
    resumed_first = resumed._current_nonterminal_kind_nodes("structure")
    assert resumed._current_nonterminal_kind_nodes("structure") is resumed_first
    assert resumed_first == graph._current_nonterminal_kind_nodes(
        "structure"
    )


def test_current_neighbor_cache_tracks_permanent_and_dynamic_generations() -> None:
    graph = TemporalMarketSceneGraph()
    asof = _clock("2025-01-06 10:00")
    source = graph.add_node(
        _node("neighbor-source", "structure", Timeframe.H1, asof)
    )
    first_target = graph.add_node(
        _node("neighbor-first", "displacement", Timeframe.M5, asof)
    )
    graph.add_edge(
        SceneEdge(
            edge_id="edge:neighbor-first",
            source_node_id=source.node_id,
            relation=SceneEdgeKind.ALIGNS_WITH,
            target_node_id=first_target.node_id,
            observed_at=asof,
            source_ids=(source.node_id, first_target.node_id),
        )
    )
    first = tuple(graph._neighbors(source.node_id))
    assert graph._current_neighbors_cache[source.node_id] == first

    second_target = graph.add_node(
        _node("neighbor-second", "displacement", Timeframe.M5, asof)
    )
    graph.add_edge(
        SceneEdge(
            edge_id="edge:neighbor-second",
            source_node_id=source.node_id,
            relation=SceneEdgeKind.CONFIRMS,
            target_node_id=second_target.node_id,
            observed_at=asof,
            source_ids=(source.node_id, second_target.node_id),
        )
    )
    assert not graph._current_neighbors_cache
    assert {value[0] for value in graph._neighbors(source.node_id)} == {
        first_target.node_id,
        second_target.node_id,
    }

    liquidity = []
    for index, price in enumerate((101.0, 102.0, 103.0), start=1):
        liquidity.append(
            graph.add_node(
                _node(
                    f"neighbor-liquidity-{index}",
                    "liquidity",
                    Timeframe.M5,
                    asof,
                    lifecycle=LiquidityInventoryLifecycle.VISIBLE.value,
                    bounds=(price, price),
                )
            )
        )
    assert tuple(graph._neighbors(liquidity[0].node_id)) == ()
    graph._refresh_current_path_block_edges(
        SimpleNamespace(price=100.0, asof=asof)
    )
    dynamic = tuple(graph._neighbors(liquidity[0].node_id))
    assert len(dynamic) == 1
    assert dynamic[0][1].relation is SceneEdgeKind.BLOCKS_PATH_TO
    graph._refresh_current_path_block_edges(
        SimpleNamespace(price=102.5, asof=asof + pd.Timedelta(minutes=1))
    )
    assert tuple(graph._neighbors(liquidity[0].node_id)) == ()

    resumed = pickle.loads(pickle.dumps(graph))
    resumed.__dict__.pop("_current_neighbors_cache")
    assert tuple(resumed._neighbors(source.node_id)) == tuple(
        graph._neighbors(source.node_id)
    )


def test_scale_registry_hashes_the_full_contract_and_rejects_coercion() -> None:
    base = tuple(
        replace(item, history_limit=64) for item in MODEL_SCALE_SPECS
    )
    changed = tuple(
        replace(item, session_anchor="exchange_session")
        for item in base[:-1]
    ) + (replace(base[-1], history_limit=65),)
    assert scale_registry_id(base) != scale_registry_id(changed)
    with pytest.raises(ValueError):
        ScaleSpec(
            Timeframe.M15,
            ScaleRole.CONTEXT_BRIDGE,
            session_anchor="midnight",
        )
    with pytest.raises(ValueError):
        parse_scale_specs(
            [
                {
                    "timeframe": item.key,
                    "role": item.role.value,
                    "completed_bar_only": "false"
                    if item.native_timeframe is Timeframe.M1
                    else True,
                }
                for item in MODEL_SCALE_SPECS
            ]
        )


def test_m15_emits_only_on_completed_bar_across_dst_and_contract_reset() -> None:
    specs = tuple(
        replace(item, history_limit=64) for item in MODEL_SCALE_SPECS
    )
    reader = CausalMarketReader(scale_specs=specs)
    friday = pd.date_range(
        "2025-03-07 16:50",
        periods=10,
        freq="min",
        tz=TZ,
    )
    for timestamp in friday:
        update = reader.on_bar(
            Bar(timestamp, 100, 101, 99, 100, 10, "NQH5", 1)
        )
        assert not update.newly_completed[Timeframe.M15]
    sunday = pd.date_range(
        "2025-03-09 18:00",
        periods=15,
        freq="min",
        tz=TZ,
    )
    for index, timestamp in enumerate(sunday):
        update = reader.on_bar(
            Bar(timestamp, 101, 102, 100, 101, 10, "NQH5", 1)
        )
        assert bool(update.newly_completed[Timeframe.M15]) == (index == 14)
    candle = update.newly_completed[Timeframe.M15][0]
    assert candle.end == update.asof
    assert candle.symbol == "NQH5"

    reset_reader = CausalMarketReader(scale_specs=specs)
    timestamps = pd.date_range(
        "2025-03-10 09:00",
        periods=30,
        freq="min",
        tz=TZ,
    )
    for timestamp in timestamps[:5]:
        reset_reader.on_bar(
            Bar(timestamp, 100, 101, 99, 100, 10, "NQH5", 1)
        )
    for index, timestamp in enumerate(timestamps[5:]):
        update = reset_reader.on_bar(
            Bar(timestamp, 102, 103, 101, 102, 10, "NQM5", 2)
        )
        if index == 0:
            assert "contract_change_history_reset" in update.anomalies
    assert len(update.histories[Timeframe.M15]) == 1
    assert update.histories[Timeframe.M15][0].instrument_id == 2


def test_m15_blank_frame_is_immutable_until_first_completed_bar() -> None:
    specs = tuple(
        replace(item, history_limit=64) for item in MODEL_SCALE_SPECS
    )
    reader = CausalMarketReader(scale_specs=specs)
    observer = CausalObserver(ObserverConfig(scale_specs=specs))
    start = _clock("2025-03-10 09:00")
    first_frame = None
    first_cutoff = None
    for index in range(14):
        update = reader.on_bar(
            Bar(
                start + pd.Timedelta(minutes=index),
                100,
                100.5,
                99.5,
                100.25,
                10,
                "NQH5",
                1,
            )
        )
        observation = observer.observe(update)
        frame = observation.frame(Timeframe.M15)
        if first_frame is None:
            first_frame = frame
            first_cutoff = frame.cutoff
        else:
            assert frame is first_frame
            assert frame.cutoff == first_cutoff
    update = reader.on_bar(
        Bar(
            start + pd.Timedelta(minutes=14),
            100,
            100.5,
            99.5,
            100.25,
            10,
            "NQH5",
            1,
        )
    )
    completed = observer.observe(update).frame(Timeframe.M15)
    assert completed is not first_frame
    assert completed.cutoff == update.asof


def test_edge_close_and_ambiguity_resolution_are_append_only_asof() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    t1 = t0 + pd.Timedelta(minutes=1)
    source = graph.add_node(_node("source", "structure", Timeframe.H1, t0))
    target = graph.add_node(_node("target", "displacement", Timeframe.M5, t0))
    edge = graph.add_edge(
        SceneEdge(
            edge_id="edge:test",
            source_node_id=source.node_id,
            relation=SceneEdgeKind.ALIGNS_WITH,
            target_node_id=target.node_id,
            observed_at=t0,
            source_ids=(source.node_id, target.node_id),
        )
    )
    graph.add_node(
        replace(
            target,
            observed_at=t1,
            lifecycle="exhausted",
            ambiguity_state=EvidenceStatus.INVALIDATED,
            resolution_reason="qualified_opposite_displacement",
            revision_id="",
        )
    )
    assert graph.edges_asof(t0)[0].lifecycle == "active"
    closed = graph.edges_asof(t1)[0]
    assert closed.lifecycle == "closed"
    assert closed.first_observed_at == t0
    assert closed.observed_at == t1
    assert len(graph.edge_history(edge.edge_id)) == 2
    graph._add_relation(
        source.node_id,
        SceneEdgeKind.ALIGNS_WITH,
        target.node_id,
        t1 + pd.Timedelta(minutes=1),
        (source.node_id, target.node_id),
    )
    assert len(graph.edge_history(edge.edge_id)) == 2
    assert graph._edges[edge.edge_id].lifecycle == "closed"

    ambiguous = graph.add_node(
        _node(
            "ambiguous",
            "micro_bos",
            Timeframe.M1,
            t0,
            lifecycle="ambiguous",
            status=EvidenceStatus.AMBIGUOUS,
        )
    )
    graph.add_node(
        replace(
            ambiguous,
            confirmed_at=t1,
            observed_at=t1,
            lifecycle="confirmed",
            ambiguity_state=EvidenceStatus.CONFIRMED,
            resolution_reason="same_clock_order_resolved",
            revision_id="",
        )
    )
    assert not any(node.kind == "resolution" for node in graph.nodes_asof(t0))
    resolution_nodes = [
        node for node in graph.nodes_asof(t1) if node.kind == "resolution"
    ]
    assert len(resolution_nodes) == 1
    assert any(
        edge.relation is SceneEdgeKind.RESOLVES
        and edge.target_node_id == ambiguous.node_id
        for edge in graph.edges_asof(t1)
    )
    assert graph.node_history("ambiguous")[0].ambiguity_state is EvidenceStatus.AMBIGUOUS


def test_typed_zone_enrichment_preserves_the_prior_revision() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    displacement = graph.add_node(
        _node(
            "epoch:0:displacement:5m:disp-1",
            "displacement",
            Timeframe.M5,
            t0,
            lifecycle="active",
            source_ids=("disp-1",),
            entity_id="disp-1",
        )
    )
    event = MarketEvent(
        event_id="event:fvg",
        kind=EventKind.FVG_STATE,
        observed_at=t0,
        timeframe=Timeframe.M5,
        side=None,
        price=100.0,
        strength=0.5,
        source_ids=("disp-1",),
        entity_id="fvg-1",
        lifecycle="open",
        formed_at=t0,
        confirmed_at=t0,
        direction=Direction.LONG,
    )
    graph._adapt_market_event(event)
    state = SimpleNamespace(
        fvg_id="fvg-1",
        timeframe=Timeframe.M5,
        lifecycle="open",
        direction=Direction.LONG,
        formed_at=t0,
        confirmed_at=t0,
        state_started_at=t0,
        last_updated_at=t0,
        lower_bound=99.5,
        upper_bound=100.5,
        source_displacement_id="disp-1",
        transition_reason=None,
    )
    graph.add_node(graph._adapt_state(state, asof=t0))
    history = graph.node_history("epoch:0:fvg:5m:fvg-1")
    assert len(history) == 2
    assert history[0].price_bounds is None
    assert history[1].price_bounds == (99.5, 100.5)
    assert history[0].revision_id != history[1].revision_id
    assert graph._node_id_for_source("fvg-1") == history[1].node_id
    graph._derive_source_relations(
        market_observation(asof=t0, price=100.0)
    )
    assert any(
        edge.relation is SceneEdgeKind.CREATES
        and edge.source_node_id == displacement.node_id
        and edge.target_node_id == history[1].node_id
        for edge in graph.edges_asof(t0)
    )


def test_base_liquidity_node_has_no_global_route_role() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    item = SimpleNamespace(
        item_id="h4-liquidity-1",
        timeframe=Timeframe.H4,
        lifecycle="visible",
        side="above",
        formed_at=t0,
        confirmed_at=t0,
        price=101.0,
        lower_bound=101.0,
        upper_bound=101.0,
        source_ids=("swing-1",),
    )
    node = graph._adapt_state(item, asof=t0)
    assert node is not None
    assert node.liquidity_role is None


def test_terminal_pool_event_is_not_reopened_by_stale_frame_source() -> None:
    graph = TemporalMarketSceneGraph()
    asof = _clock("2025-01-06 10:00")
    formed_at = asof - pd.Timedelta(minutes=4)
    confirmed_at = asof - pd.Timedelta(minutes=2)
    pool = LiquidityPoolState(
        pool_id="same-clock-terminal-pool",
        timeframe=Timeframe.M1,
        side="above",
        lower_bound=100.75,
        upper_bound=101.25,
        midpoint=101.0,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        lifecycle=LiquidityPoolLifecycle.FORMED,
        member_swing_ids=("pool-a", "pool-b"),
        touch_times=(formed_at, confirmed_at),
        age_bars=2,
        strength=0.7,
        total_touch_count=2,
    )
    node_id = (
        "epoch:0:liquidity_pool:1m:same-clock-terminal-pool"
    )
    graph.add_node(
        SceneNode(
            node_id=node_id,
            kind="liquidity_pool",
            timeframe=Timeframe.M1.value,
            structural_scale=StructuralScale.INTERNAL,
            direction=Direction.LONG,
            formed_at=formed_at,
            confirmed_at=confirmed_at,
            observed_at=asof,
            lifecycle=LiquidityPoolLifecycle.ACCEPTED.value,
            price_bounds=(pool.lower_bound, pool.upper_bound),
            invalidation_rule="pool_consumed_or_accepted_outside",
            ambiguity_state=EvidenceStatus.CONFIRMED,
            source_ids=(pool.pool_id, *pool.member_swing_ids),
            entity_id=pool.pool_id,
        )
    )
    assert node_id not in graph._current_epoch_node_ids
    assert node_id in graph._entity_index[pool.pool_id]
    stale = graph._adapt_state(pool, asof=asof)
    assert stale is not None
    graph._add_frame_state_node(
        replace(
            stale,
            observed_at=asof + pd.Timedelta(minutes=1),
        )
    )
    assert graph._nodes[node_id].lifecycle == "accepted"
    assert len(graph.node_history(node_id)) == 1


def test_update_uses_lightweight_token_before_full_snapshot_adaptation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    item = LiquidityInventoryItem(
        item_id="heartbeat-liquidity",
        timeframe=Timeframe.M1,
        side="above",
        kind="previous_day_high",
        price=101.0,
        lower_bound=101.0,
        upper_bound=101.0,
        formed_at=t0 - pd.Timedelta(minutes=10),
        confirmed_at=t0 - pd.Timedelta(minutes=5),
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("previous-day-source",),
        age_bars=0,
        strength=0.7,
        structural_rank="external",
        visibility_strength=0.7,
    )
    first_observation = replace(
        market_observation(asof=t0, price=100.0),
        liquidity_inventory=(item,),
    )
    graph.update(first_observation)
    adapted: list[str] = []
    original = graph._adapt_snapshot_state

    def record_adaptation(state, *, asof, frame_state):
        adapted.append(type(state).__name__)
        return original(state, asof=asof, frame_state=frame_state)

    monkeypatch.setattr(graph, "_adapt_snapshot_state", record_adaptation)
    graph.update(
        replace_market_observation(
            first_observation,
            asof=t0 + pd.Timedelta(minutes=1),
            liquidity_inventory=(replace(item, age_bars=1),),
        )
    )

    assert adapted == []


def test_ordinary_update_does_not_rescan_all_retained_timelines() -> None:
    class CountingTimelines(dict):
        values_calls = 0

        def values(self):
            self.values_calls += 1
            return super().values()

    graph = TemporalMarketSceneGraph()
    timelines = CountingTimelines()
    observation = replace(
        market_observation(asof=_clock("2025-01-06 10:00")),
        retained_entity_timelines=timelines,
    )

    graph.update(observation)

    # One tail scan belongs to unseen-ledger collection.  The terminal
    # snapshot guard must reuse that result on an ordinary market minute.
    assert timelines.values_calls == 1


def test_pool_ledger_namespace_reuses_the_typed_state_node() -> None:
    graph = TemporalMarketSceneGraph()
    confirmed_at = _clock("2025-01-06 10:00")
    pool = LiquidityPoolState(
        pool_id="one-pool-identity",
        timeframe=Timeframe.M5,
        side="above",
        lower_bound=100.75,
        upper_bound=101.25,
        midpoint=101.0,
        formed_at=confirmed_at - pd.Timedelta(minutes=10),
        confirmed_at=confirmed_at,
        lifecycle=LiquidityPoolLifecycle.FORMED,
        member_swing_ids=("pool-a", "pool-b"),
        touch_times=(
            confirmed_at - pd.Timedelta(minutes=10),
            confirmed_at,
        ),
        age_bars=0,
        strength=0.7,
        total_touch_count=2,
    )
    event = MarketEvent(
        event_id="pool-formed-event",
        kind=EventKind.LIQUIDITY_POOL_STATE,
        observed_at=confirmed_at,
        timeframe=Timeframe.M5,
        side="above",
        price=pool.midpoint,
        strength=pool.strength,
        source_ids=pool.member_swing_ids,
        entity_id=f"pool:{pool.pool_id}",
        lifecycle=LiquidityPoolLifecycle.FORMED.value,
        formed_at=pool.formed_at,
        confirmed_at=pool.confirmed_at,
    )

    graph._adapt_market_event(event, enrichment=pool)
    node = graph._adapt_state(pool, asof=confirmed_at)
    assert node is not None
    graph._add_frame_state_node(node)

    pool_nodes = tuple(
        value for value in graph.nodes if value.kind == "liquidity_pool"
    )
    assert len(pool_nodes) == 1
    assert pool_nodes[0].entity_id == pool.pool_id
    assert pool_nodes[0].node_id.endswith(f":{pool.pool_id}")


def test_semantic_heartbeat_does_not_create_scene_revisions() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    state = SimpleNamespace(
        location_id="location-1",
        timeframe=Timeframe.M1,
        lifecycle="approaching",
        direction=Direction.LONG,
        formed_at=t0,
        state_started_at=t0,
        last_updated_at=t0,
        lower_bound=99.5,
        upper_bound=100.5,
        source_zone_id="zone-1",
        source_displacement_id="disp-1",
        transition_reason="source_registered",
        age_real_1m_bars=0,
    )
    graph.add_node(graph._adapt_state(state, asof=t0))
    heartbeat = SimpleNamespace(
        **{
            **vars(state),
            "last_updated_at": t0 + pd.Timedelta(minutes=1),
            "age_real_1m_bars": 1,
        }
    )
    graph.add_node(
        graph._adapt_state(
            heartbeat,
            asof=t0 + pd.Timedelta(minutes=1),
        )
    )
    assert len(
        graph.node_history("epoch:0:entry_location:1m:location-1")
    ) == 1


def test_support_resistance_freshness_is_not_a_persistent_revision() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    state = SimpleNamespace(
        zone_id="support-1",
        timeframe=Timeframe.M1,
        lifecycle="active",
        side="support",
        formed_at=t0 - pd.Timedelta(minutes=10),
        confirmed_at=t0,
        metadata_observed_at=t0,
        lower_bound=99.5,
        upper_bound=100.0,
        member_swing_ids=("swing-1",),
        source_kind="structural_swing",
        structural_rank="internal",
        is_protected_swing=False,
        zone_role="both",
        visibility_strength=0.8,
        reaction_quality=0.6,
        freshness=1.0,
        depletion_risk=0.1,
        age_bars=0,
    )
    node = graph._adapt_state(state, asof=t0)
    assert node is not None
    graph.add_node(node)
    node_id = node.node_id
    revision = graph.revision_id

    heartbeat = SimpleNamespace(
        **{
            **vars(state),
            "metadata_observed_at": t0 + pd.Timedelta(minutes=1),
            "freshness": 0.9,
            "age_bars": 1,
        }
    )
    heartbeat_node = graph._adapt_state(
        heartbeat,
        asof=t0 + pd.Timedelta(minutes=1),
    )
    assert heartbeat_node is not None
    graph.add_node(heartbeat_node)

    assert graph.revision_id == revision
    assert len(graph.node_history(node_id)) == 1
    assert "freshness" not in dict(graph.nodes[0].descriptive_metrics)

    changed = SimpleNamespace(
        **{
            **vars(heartbeat),
            "metadata_observed_at": t0 + pd.Timedelta(minutes=2),
            "reaction_quality": 0.7,
        }
    )
    changed_node = graph._adapt_state(
        changed,
        asof=t0 + pd.Timedelta(minutes=2),
    )
    assert changed_node is not None
    graph.add_node(changed_node)
    assert len(graph.node_history(node_id)) == 2

    promoted = SimpleNamespace(
        **{
            **vars(changed),
            "metadata_observed_at": t0 + pd.Timedelta(minutes=3),
            "structural_rank": "external",
            "is_protected_swing": True,
            "visibility_strength": 1.0,
        }
    )
    promoted_node = graph._adapt_state(
        promoted,
        asof=t0 + pd.Timedelta(minutes=3),
    )
    assert promoted_node is not None
    graph.add_node(promoted_node)
    assert graph.nodes[0].structural_scale is StructuralScale.EXTERNAL
    assert len(graph.node_history(node_id)) == 3

    demoted = SimpleNamespace(
        **{
            **vars(promoted),
            "metadata_observed_at": t0 + pd.Timedelta(minutes=4),
            "structural_rank": "internal",
            "is_protected_swing": False,
            "visibility_strength": 0.2,
        }
    )
    demoted_node = graph._adapt_state(
        demoted,
        asof=t0 + pd.Timedelta(minutes=4),
    )
    assert demoted_node is not None
    graph.add_node(demoted_node)
    assert graph.nodes[0].structural_scale is StructuralScale.INTERNAL
    assert len(graph.node_history(node_id)) == 4


def test_snapshot_semantic_signature_skips_heartbeat_adaptation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    state = SimpleNamespace(
        zone_id="signature-zone",
        timeframe=Timeframe.M1,
        lifecycle="active",
        side="support",
        formed_at=t0 - pd.Timedelta(minutes=10),
        confirmed_at=t0,
        metadata_observed_at=t0,
        lower_bound=99.5,
        upper_bound=100.0,
        member_swing_ids=("signature-swing",),
        source_kind="structural_swing",
        structural_rank="internal",
        is_protected_swing=False,
        zone_role="both",
        visibility_strength=0.8,
        reaction_quality=0.6,
        freshness=1.0,
        depletion_risk=0.1,
        age_bars=0,
    )
    original = graph._adapt_state
    adapted: list[str] = []

    def counted(item: object, *, asof: pd.Timestamp) -> SceneNode | None:
        adapted.append(str(getattr(item, "zone_id", "unknown")))
        return original(item, asof=asof)

    monkeypatch.setattr(graph, "_adapt_state", counted)
    graph._adapt_snapshot_state(state, asof=t0, frame_state=True)
    heartbeat = SimpleNamespace(
        **{
            **vars(state),
            "metadata_observed_at": t0 + pd.Timedelta(minutes=1),
            "freshness": 0.5,
            "age_bars": 1,
        }
    )
    graph._adapt_snapshot_state(
        heartbeat,
        asof=t0 + pd.Timedelta(minutes=1),
        frame_state=True,
    )

    node_id = "epoch:0:support_resistance:1m:signature-zone"
    assert adapted == ["signature-zone"]
    assert len(graph.node_history(node_id)) == 1

    metadata_only = SimpleNamespace(
        **{
            **vars(heartbeat),
            "metadata_observed_at": t0 + pd.Timedelta(minutes=2),
            "reaction_quality": 0.7,
            "structural_rank": "external",
            "is_protected_swing": True,
        }
    )
    graph._adapt_snapshot_state(
        metadata_only,
        asof=t0 + pd.Timedelta(minutes=2),
        frame_state=True,
    )
    assert adapted == ["signature-zone"]
    assert len(graph.node_history(node_id)) == 1

    changed = SimpleNamespace(
        **{
            **vars(metadata_only),
            "lifecycle": "tested",
            "tested_at": t0 + pd.Timedelta(minutes=3),
        }
    )
    graph._adapt_snapshot_state(
        changed,
        asof=t0 + pd.Timedelta(minutes=3),
        frame_state=True,
    )
    assert adapted == ["signature-zone", "signature-zone"]
    assert len(graph.node_history(node_id)) == 2


def test_terminal_sr_metadata_promotion_does_not_rewrite_frozen_graph_fact() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    reaccepted_at = t0 + pd.Timedelta(minutes=2)
    state = SimpleNamespace(
        zone_id="terminal-rank-zone",
        timeframe=Timeframe.M1,
        lifecycle="reaccepted",
        side="support",
        formed_at=t0 - pd.Timedelta(minutes=10),
        confirmed_at=t0,
        broken_at=t0 + pd.Timedelta(minutes=1),
        reaccepted_at=reaccepted_at,
        lower_bound=99.5,
        upper_bound=100.0,
        member_swing_ids=("terminal-rank-swing",),
        source_kind="structural_swing",
        structural_rank="internal",
        is_protected_swing=False,
        zone_role="both",
        visibility_strength=0.2,
        reaction_quality=0.5,
        depletion_risk=0.5,
    )
    stored = graph._adapt_snapshot_state(
        state,
        asof=reaccepted_at,
        frame_state=True,
    )
    assert stored is not None

    promoted_snapshot = SimpleNamespace(
        **{
            **vars(state),
            "structural_rank": "external",
            "is_protected_swing": True,
            "visibility_strength": 1.0,
            "metadata_observed_at": reaccepted_at + pd.Timedelta(minutes=1),
        }
    )
    retained = graph._adapt_snapshot_state(
        promoted_snapshot,
        asof=reaccepted_at + pd.Timedelta(minutes=1),
        frame_state=True,
    )

    assert retained is stored
    assert retained.structural_scale is StructuralScale.INTERNAL
    assert len(graph.node_history(stored.node_id)) == 1


def test_reaccepted_support_resistance_is_confirmed_and_terminal() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    reaccepted_at = t0 + pd.Timedelta(minutes=2)
    state = SimpleNamespace(
        zone_id="reaccepted-zone",
        timeframe=Timeframe.M1,
        lifecycle="reaccepted",
        side="support",
        formed_at=t0 - pd.Timedelta(minutes=10),
        confirmed_at=t0,
        broken_at=t0 + pd.Timedelta(minutes=1),
        reaccepted_at=reaccepted_at,
        lower_bound=99.5,
        upper_bound=100.0,
        member_swing_ids=("reaccepted-swing",),
        source_kind="structural_swing",
        structural_rank="internal",
        is_protected_swing=False,
        zone_role="both",
    )
    node = graph._adapt_state(state, asof=reaccepted_at)
    assert node is not None
    assert node.ambiguity_state is EvidenceStatus.CONFIRMED
    stored = graph.add_node(node)

    with pytest.raises(ValueError, match="rewrite frozen fact"):
        graph.add_node(
            replace(
                stored,
                observed_at=reaccepted_at + pd.Timedelta(minutes=1),
                lifecycle="tested",
                revision_id="",
            )
        )


def test_same_bar_ledger_transitions_remain_ordered_before_snapshot_fast_path() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    opened = MarketEvent(
        event_id="z-opened",
        kind=EventKind.FVG_STATE,
        observed_at=t0,
        timeframe=Timeframe.M5,
        side=None,
        price=None,
        strength=0.5,
        source_ids=("same-bar-fvg",),
        entity_id="same-bar-fvg",
        lifecycle="open",
        formed_at=t0,
        confirmed_at=t0,
        direction=Direction.LONG,
        sequence_no=0,
    )
    partial = replace(
        opened,
        event_id="a-partial",
        lifecycle="partial",
        transition_reason="first_partial_fill",
        sequence_no=1,
    )
    observation = replace(
        market_observation(asof=t0),
        # Reverse the container order and event-id lexical order: sequence_no
        # remains the sole same-clock causal ordering authority.
        recent_events=(partial, opened),
    )

    graph.update(observation)

    history = graph.node_history(
        "epoch:0:fvg:5m:same-bar-fvg"
    )
    assert tuple(node.lifecycle for node in history) == ("open", "partial")
    assert history[0].revision_id != history[1].revision_id


def test_path_block_edges_are_current_only_and_recomputed_by_adjacency() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")

    def liquidity(identity: str, price: float) -> LiquidityInventoryItem:
        return LiquidityInventoryItem(
            item_id=identity,
            timeframe=Timeframe.M1,
            side="above",
            kind="previous_day_high",
            price=price,
            lower_bound=price,
            upper_bound=price,
            formed_at=t0 - pd.Timedelta(minutes=10),
            confirmed_at=t0 - pd.Timedelta(minutes=5),
            lifecycle=LiquidityInventoryLifecycle.VISIBLE,
            source_ids=(f"source:{identity}",),
            age_bars=0,
            strength=0.7,
            structural_rank="external",
            visibility_strength=0.7,
        )

    near = liquidity("liquidity-near", 101.0)
    middle = liquidity("liquidity-middle", 102.0)
    far = liquidity("liquidity-far", 103.0)
    first = replace(
        market_observation(asof=t0, price=100.0),
        liquidity_inventory=(near, far),
    )
    first_delta = graph.update(first)
    first_edges = tuple(
        edge
        for edge in graph.edges
        if edge.relation is SceneEdgeKind.BLOCKS_PATH_TO
    )
    assert len(first_edges) == 1
    stale_edge_id = first_edges[0].edge_id
    assert graph.edge_history(stale_edge_id) == ()
    assert stale_edge_id not in first_delta.added_edge_ids

    second = replace(
        market_observation(
            asof=t0 + pd.Timedelta(minutes=1),
            price=100.0,
        ),
        liquidity_inventory=(near, middle, far),
    )
    second_delta = graph.update(second)
    node_by_id = {node.node_id: node for node in graph.nodes}
    current_block_edges = tuple(
        edge
        for edge in graph.edges
        if edge.relation is SceneEdgeKind.BLOCKS_PATH_TO
    )
    current_pairs = {
        (
            node_by_id[edge.source_node_id].entity_id,
            node_by_id[edge.target_node_id].entity_id,
        )
        for edge in current_block_edges
    }
    assert current_pairs == {
        ("liquidity-near", "liquidity-middle"),
        ("liquidity-middle", "liquidity-far"),
    }
    current_path = graph.find_path(
        (near.item_id,),
        (far.item_id,),
        asof=second.asof,
    )
    assert current_path.count(SceneEdgeKind.BLOCKS_PATH_TO.value) == 2
    assert stale_edge_id not in {
        edge.edge_id
        for edge in graph.edges
        if edge.relation is SceneEdgeKind.BLOCKS_PATH_TO
    }
    assert not any(
        edge.relation is SceneEdgeKind.BLOCKS_PATH_TO
        for edge in graph.edges_asof(t0)
    )
    assert all(
        graph.edge_history(edge.edge_id) == ()
        for edge in current_block_edges
    )
    assert not any(
        edge.relation is SceneEdgeKind.BLOCKS_PATH_TO
        for edge in graph._edges.values()
    )
    assert not set(second_delta.added_edge_ids).intersection(
        edge.edge_id for edge in current_block_edges
    )

    consumed_middle = replace(
        middle,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=t0 + pd.Timedelta(minutes=2),
        lifecycle_reason="reference_level_swept",
    )
    third = replace(
        market_observation(
            asof=t0 + pd.Timedelta(minutes=2),
            price=100.0,
        ),
        liquidity_inventory=(near, consumed_middle, far),
    )
    graph.update(third)
    third_nodes = {node.node_id: node for node in graph.nodes}
    third_pairs = {
        (
            third_nodes[edge.source_node_id].entity_id,
            third_nodes[edge.target_node_id].entity_id,
        )
        for edge in graph.edges
        if edge.relation is SceneEdgeKind.BLOCKS_PATH_TO
    }
    assert third_pairs == {("liquidity-near", "liquidity-far")}
    consumed_id = (
        "epoch:0:liquidity:1m:liquidity-middle"
    )
    assert consumed_id not in (
        graph._visible_liquidity_node_ids_by_side["above"]
    )


def test_located_at_uses_shared_source_indexes_for_all_authoritative_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    source_cases = (
        ("pool-source", "liquidity_pool", None),
        ("previous-day-source", "liquidity", "previous_day_high"),
        ("range-source", "liquidity", "range_boundary"),
    )
    expected: set[tuple[str, str]] = set()
    for index, (source_id, kind, inventory_kind) in enumerate(source_cases):
        zone = graph.add_node(
            SceneNode(
                node_id=f"epoch:0:support_resistance:1m:zone-{index}",
                kind="support_resistance",
                timeframe=Timeframe.M1.value,
                structural_scale=StructuralScale.EXTERNAL,
                direction=None,
                formed_at=t0,
                confirmed_at=t0,
                observed_at=t0,
                lifecycle="active",
                price_bounds=(100.0 + index, 100.5 + index),
                invalidation_rule="close_beyond_frozen_zone",
                ambiguity_state=EvidenceStatus.CONFIRMED,
                source_ids=(f"zone-{index}", source_id),
                entity_id=f"zone-{index}",
            )
        )
        liquidity = graph.add_node(
            SceneNode(
                node_id=f"epoch:0:{kind}:1m:liquidity-{index}",
                kind=kind,
                timeframe=Timeframe.M1.value,
                structural_scale=StructuralScale.EXTERNAL,
                direction=Direction.LONG,
                formed_at=t0,
                confirmed_at=t0,
                observed_at=t0,
                lifecycle=(
                    "formed" if kind == "liquidity_pool" else "visible"
                ),
                price_bounds=(100.25 + index, 100.25 + index),
                invalidation_rule="liquidity_consumed",
                ambiguity_state=EvidenceStatus.CONFIRMED,
                source_ids=(f"liquidity-{index}", source_id),
                entity_id=f"liquidity-{index}",
                semantic_attributes=(
                    ()
                    if inventory_kind is None
                    else (("inventory_kind", inventory_kind),)
                ),
            )
        )
        expected.add((liquidity.node_id, zone.node_id))

    def no_history_scan(_: pd.Timestamp) -> tuple[SceneNode, ...]:
        raise AssertionError("LOCATED_AT must not scan the full node ledger")

    monkeypatch.setattr(graph, "nodes_asof", no_history_scan)
    graph._derive_source_relations(market_observation(asof=t0))

    located = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
        if edge.relation is SceneEdgeKind.LOCATED_AT
    }
    assert located == expected
    assert not any(
        edge.relation is SceneEdgeKind.SOURCED_FROM
        and graph._nodes[edge.source_node_id].kind == "support_resistance"
        and graph._nodes[edge.target_node_id].kind == "liquidity"
        for edge in graph.edges
    )


def test_candle_structure_is_an_explicit_transient_scene_view() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    candle = CandleStructureState(
        timeframe=Timeframe.M1,
        start=t0 - pd.Timedelta(minutes=1),
        observed_at=t0,
        range_points=4.0,
        body_points=2.0,
        upper_wick_points=1.0,
        lower_wick_points=1.0,
        body_ratio=0.5,
        upper_wick_ratio=0.25,
        lower_wick_ratio=0.25,
        close_location=0.75,
        direction=1,
        real_completed=True,
        zero_range=False,
        body_class="normal",
        range_class="expanded",
        dominant_wick="balanced",
        close_class="near_high",
    )
    observation = market_observation(asof=t0)
    frames = dict(observation.frames)
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        candle_structure=candle,
    )
    observation = replace(observation, frames=frames)
    graph.update(observation)
    revision = graph.revision_id
    persistent_ids = {node.node_id for node in graph.nodes}

    nodes = graph.current_candle_structure_nodes(
        observation,
        timeframes=(Timeframe.M1,),
    )

    assert len(nodes) == 1
    node = nodes[0]
    assert node.kind == "candle_structure"
    assert node.direction is Direction.LONG
    assert node.ambiguity_state is EvidenceStatus.CONFIRMED
    assert dict(node.semantic_attributes) == {
        "body_class": "normal",
        "range_class": "expanded",
        "dominant_wick": "balanced",
        "close_class": "near_high",
        "real_completed": "true",
        "zero_range": "false",
    }
    assert dict(node.descriptive_metrics)["body_ratio"] == 0.5
    assert node.node_id not in persistent_ids
    assert node.node_id not in {value.node_id for value in graph.nodes}
    assert graph.node_history(node.node_id) == ()
    assert graph.revision_id == revision

    focus = FocusState(
        asof=t0,
        primary_timeframes=(Timeframe.M1.value,),
        reason_codes=("test_default_query",),
        question="verify default semantic query",
        resolution_status=EvidenceStatus.CONFIRMED,
        switched=False,
    )
    assert not any(
        value.kind == "candle_structure"
        for value in graph.query(focus).nodes
    )


def test_clock_only_projection_keeps_terminal_reason_without_revision() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    node = _node(
        "epoch:0:bos:1m:bos-clock-only",
        "bos",
        Timeframe.M1,
        t0,
        lifecycle="failed",
        status=EvidenceStatus.INVALIDATED,
        source_ids=("bos-clock-only", "swing-1"),
        entity_id="bos-clock-only",
    )
    graph.add_node(replace(node, resolution_reason="superseded"))
    graph.add_node(
        replace(
            node,
            observed_at=t0 + pd.Timedelta(minutes=1),
            resolution_reason="superseded",
        )
    )
    history = graph.node_history(node.node_id)
    assert len(history) == 1
    assert history[0].observed_at == t0
    assert history[0].resolution_reason == "superseded"


def test_entry_location_rejected_can_resolve_to_left() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 10:00")
    rejected = _node(
        "epoch:0:entry_location:1m:location-rejected-left",
        "entry_location",
        Timeframe.M1,
        t0,
        lifecycle="rejected",
        source_ids=("location-rejected-left", "zone-1"),
        entity_id="location-rejected-left",
    )
    graph.add_node(rejected)
    graph.add_node(
        replace(
            rejected,
            observed_at=t0 + pd.Timedelta(minutes=1),
            lifecycle="left",
            ambiguity_state=EvidenceStatus.INVALIDATED,
        )
    )
    history = graph.node_history(rejected.node_id)
    assert [item.lifecycle for item in history] == ["rejected", "left"]


def test_generic_bos_notification_does_not_toggle_confirmed_lifecycle() -> None:
    graph = TemporalMarketSceneGraph()
    resolved_at = _clock("2025-01-06 10:00")
    pending_at = resolved_at - pd.Timedelta(minutes=1)
    event = MarketEvent(
        event_id="event:bos-confirmed",
        kind=EventKind.STRUCTURE_BREAK,
        observed_at=resolved_at,
        timeframe=Timeframe.M1,
        side="above",
        price=101.0,
        strength=0.8,
        source_ids=("target-swing",),
        entity_id="bos-1",
        lifecycle="observed",
        formed_at=pending_at,
        confirmed_at=resolved_at,
        direction=Direction.LONG,
    )
    state = SimpleNamespace(
        bos_id="bos-1",
        timeframe=Timeframe.M1,
        lifecycle="confirmed",
        direction=Direction.LONG,
        pending_at=pending_at,
        resolved_at=resolved_at,
        target_swing_id="target-swing",
        transition_reason=None,
    )
    node = graph._adapt_state(state, asof=resolved_at)
    assert node is not None
    graph._add_frame_state_node(node)
    graph._adapt_market_event(event)
    graph._adapt_market_event(
        replace(event, event_id="event:bos-notification-repeat")
    )
    history = graph.node_history("epoch:0:bos:1m:bos-1")
    assert len(history) == 2
    assert {item.lifecycle for item in history} == {"confirmed"}
    assert history[-1].price_bounds == (101.0, 101.0)


def test_micro_bos_reference_keeps_its_own_entity_identity() -> None:
    graph = TemporalMarketSceneGraph()
    resolved_at = _clock("2025-01-06 10:00")
    reference = SimpleNamespace(
        reference_id="micro-reference-1",
        bos_id="upstream-bos-1",
        expected_direction=Direction.LONG,
        pending_at=resolved_at - pd.Timedelta(minutes=1),
        resolved_at=resolved_at,
        outcome="aligned",
    )
    node = graph._adapt_state(reference, asof=resolved_at)
    assert node is not None
    assert node.kind == "micro_bos"
    assert node.entity_id == "micro-reference-1"
    assert node.direction is Direction.LONG


def test_reference_retirement_resolves_the_prior_liquidity_node() -> None:
    graph = TemporalMarketSceneGraph()
    confirmed_at = _clock("2025-01-06 18:00")
    retired_at = _clock("2025-01-07 18:01")
    item = LiquidityInventoryItem(
        item_id="reference:session:2025-01-06:high:NQH5:1",
        timeframe=Timeframe.M1,
        side="above",
        kind="previous_session_high",
        price=101.0,
        lower_bound=101.0,
        upper_bound=101.0,
        formed_at=_clock("2025-01-06 09:30"),
        confirmed_at=confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("session:2025-01-06",),
        age_bars=0,
        strength=0.75,
        structural_rank="external",
        visibility_strength=0.75,
    )
    node = graph._adapt_state(item, asof=confirmed_at)
    assert node is not None
    graph.add_node(node)
    retirement = MarketEvent(
        event_id="retire:previous-session-high",
        kind=EventKind.LIQUIDITY_RETIRED,
        observed_at=retired_at,
        timeframe=Timeframe.M1,
        side="above",
        price=101.0,
        strength=0.0,
        source_ids=(item.item_id,),
        details={
            "source_kind": item.kind,
            "replacement_period": "2025-01-07",
        },
        transition_reason="reference_period_replaced",
    )
    graph._adapt_market_event(retirement)
    graph._derive_source_relations(
        market_observation(asof=retired_at)
    )

    history = graph.node_history(node.node_id)
    assert history[-1].lifecycle == "retired"
    assert history[-1].resolution_reason == "reference_period_replaced"
    assert any(
        edge.relation is SceneEdgeKind.RESOLVES
        and edge.target_node_id == node.node_id
        for edge in graph.edges
    )


def test_external_protected_liquidity_keeps_authority_in_scene_graph() -> None:
    graph = TemporalMarketSceneGraph()
    confirmed_at = _clock("2025-01-06 10:00")
    item = LiquidityInventoryItem(
        item_id="swing:external-draw",
        timeframe=Timeframe.M5,
        side="above",
        kind="swing",
        price=101.0,
        lower_bound=101.0,
        upper_bound=101.0,
        formed_at=confirmed_at - pd.Timedelta(minutes=15),
        confirmed_at=confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("external-draw",),
        age_bars=0,
        strength=0.8,
        structural_rank="external",
        is_protected_swing=True,
        visibility_strength=0.9,
    )

    node = graph._adapt_state(item, asof=confirmed_at)

    assert node is not None
    assert node.structural_scale is StructuralScale.EXTERNAL
    assert dict(node.semantic_attributes)["structural_rank"] == "external"
    assert dict(node.semantic_attributes)["is_protected_swing"] == "true"
    assert dict(node.descriptive_metrics)["visibility_strength"] == 0.9


def test_order_block_has_displacement_and_bos_creation_edges() -> None:
    graph = TemporalMarketSceneGraph()
    clock = _clock("2025-01-06 10:00")
    displacement = graph.add_node(
        _node(
            "epoch:0:displacement:5m:disp-ob",
            "displacement",
            Timeframe.M5,
            clock,
            source_ids=("disp-ob",),
            entity_id="disp-ob",
            lifecycle="active",
        )
    )
    bos = graph.add_node(
        _node(
            "epoch:0:bos:5m:bos-ob",
            "bos",
            Timeframe.M5,
            clock,
            source_ids=("bos-ob",),
            entity_id="bos-ob",
        )
    )
    order_block = graph.add_node(
        _node(
            "epoch:0:order_block:5m:ob",
            "order_block",
            Timeframe.M5,
            clock + pd.Timedelta(minutes=1),
            source_ids=("ob", "disp-ob", "bos-ob"),
            entity_id="ob",
            lifecycle="created",
        )
    )

    graph._derive_source_relations(
        market_observation(
            asof=clock + pd.Timedelta(minutes=1),
            price=100.0,
        )
    )

    creators = {
        edge.source_node_id
        for edge in graph.edges
        if edge.relation is SceneEdgeKind.CREATES
        and edge.target_node_id == order_block.node_id
    }
    assert creators == {displacement.node_id, bos.node_id}


def test_precedes_stays_diagnostic_but_cannot_form_a_causal_path_or_thesis() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 09:00")
    manipulation = graph.add_node(
        _node(
            "diagnostic-manipulation",
            "manipulation",
            Timeframe.M1,
            t0,
            direction=Direction.SHORT,
            lifecycle="reaccepted",
        )
    )
    displacement = graph.add_node(
        _node(
            "diagnostic-displacement",
            "displacement",
            Timeframe.M5,
            t0 + pd.Timedelta(minutes=1),
            direction=Direction.LONG,
            lifecycle="active",
        )
    )
    fvg = graph.add_node(
        _node(
            "diagnostic-fvg",
            "fvg",
            Timeframe.M5,
            t0 + pd.Timedelta(minutes=2),
            direction=Direction.LONG,
            lifecycle="open",
        )
    )
    graph._add_relation(
        manipulation.node_id,
        SceneEdgeKind.PRECEDES,
        displacement.node_id,
        t0 + pd.Timedelta(minutes=2),
        (manipulation.node_id, displacement.node_id),
    )
    graph._add_relation(
        displacement.node_id,
        SceneEdgeKind.CREATES,
        fvg.node_id,
        t0 + pd.Timedelta(minutes=2),
        (displacement.node_id, fvg.node_id),
    )

    diagnostic_path = graph.find_path(
        (manipulation.node_id,),
        (fvg.node_id,),
    )
    assert SceneEdgeKind.PRECEDES.value in diagnostic_path
    assert graph.has_direct_relation(
        (manipulation.node_id,),
        SceneEdgeKind.PRECEDES,
        (displacement.node_id,),
    )
    assert not graph.find_causal_path(
        (manipulation.node_id,),
        (fvg.node_id,),
    )
    assert not graph.find_action_path(
        (manipulation.node_id,),
        (fvg.node_id,),
    )
    assert SceneEdgeKind.CREATES.value in graph.find_causal_path(
        (displacement.node_id,),
        (fvg.node_id,),
    )
    assert {node.node_id for node in _open_thesis_closure(graph, manipulation)} == {
        manipulation.node_id
    }


def test_dfp_and_lsr_source_chains_are_reachable_but_not_invented() -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 09:00")
    h4 = graph.add_node(
        _node(
            "epoch:0:structure:4H:h4-raw",
            "structure",
            Timeframe.H4,
            t0,
            source_ids=("h4-raw",),
            entity_id="h4-raw",
        )
    )
    h1 = graph.add_node(
        _node(
            "epoch:0:structure:1H:h1-raw",
            "structure",
            Timeframe.H1,
            t0 + pd.Timedelta(minutes=5),
            source_ids=("h1-raw",),
            entity_id="h1-raw",
        )
    )
    bos = graph.add_node(
        _node(
            "epoch:0:bos:1H:h1-bos-raw",
            "bos",
            Timeframe.H1,
            t0 + pd.Timedelta(minutes=10),
            source_ids=("h1-bos-raw", "h1-raw"),
            entity_id="h1-bos-raw",
        )
    )
    displacement = graph.add_node(
        _node(
            "epoch:0:displacement:5m:m5-disp-raw",
            "displacement",
            Timeframe.M5,
            t0 + pd.Timedelta(minutes=15),
            formed_at=t0 + pd.Timedelta(minutes=15),
            lifecycle="active",
            source_ids=("m5-disp-raw",),
            entity_id="m5-disp-raw",
        )
    )
    observation = market_observation(
        asof=t0 + pd.Timedelta(minutes=15),
        price=100.0,
    )
    graph._derive_source_relations(observation)
    assert graph._node_id_for_source("h1-raw") == h1.node_id
    assert graph._node_id_for_source("h1-bos-raw") == bos.node_id
    assert graph._node_id_for_source("m5-disp-raw") == displacement.node_id
    assert graph.find_path(
        (bos.node_id,),
        (displacement.node_id,),
        asof=observation.asof,
    )
    assert graph.find_path(
        (h4.node_id,),
        (displacement.node_id,),
        asof=observation.asof,
    )

    pool = graph.add_node(
        _node(
            "epoch:0:liquidity_pool:1H:pool-raw",
            "liquidity_pool",
            Timeframe.H1,
            t0 + pd.Timedelta(minutes=20),
            direction=None,
            source_ids=("pool-raw",),
            entity_id="pool-raw",
        )
    )
    manipulation = graph.add_node(
        _node(
            "epoch:0:manipulation:1m:manip-raw",
            "manipulation",
            Timeframe.M1,
            t0 + pd.Timedelta(minutes=25),
            direction=Direction.SHORT,
            source_ids=("manip-raw", "pool-raw"),
            entity_id="manip-raw",
            lifecycle="reaccepted",
        )
    )
    reverse_displacement = graph.add_node(
        _node(
            "epoch:0:displacement:5m:reverse-disp-raw",
            "displacement",
            Timeframe.M5,
            t0 + pd.Timedelta(minutes=30),
            formed_at=t0 + pd.Timedelta(minutes=30),
            direction=Direction.LONG,
            lifecycle="active",
            source_ids=("reverse-disp-raw",),
            entity_id="reverse-disp-raw",
        )
    )
    graph.add_node(
        _node(
            "epoch:0:fvg:5m:reverse-fvg-raw",
            "fvg",
            Timeframe.M5,
            t0 + pd.Timedelta(minutes=31),
            source_ids=("reverse-fvg-raw", "reverse-disp-raw"),
            entity_id="reverse-fvg-raw",
        )
    )
    trigger = graph.add_node(
        _node(
            "epoch:0:micro_bos:1m:trigger-raw",
            "micro_bos",
            Timeframe.M1,
            t0 + pd.Timedelta(minutes=32),
            source_ids=("trigger-raw", "reverse-fvg-raw"),
            entity_id="trigger-raw",
        )
    )
    graph._derive_source_relations(
        market_observation(
            asof=t0 + pd.Timedelta(minutes=32),
            price=100.0,
        )
    )
    assert graph.find_path(
        (pool.node_id,),
        (trigger.node_id,),
        asof=t0 + pd.Timedelta(minutes=32),
    )
    assert any(
        edge.relation is SceneEdgeKind.PRECEDES
        and edge.source_node_id == manipulation.node_id
        and edge.target_node_id == reverse_displacement.node_id
        for edge in graph.edges_asof(t0 + pd.Timedelta(minutes=32))
    )

    late = graph.add_node(
        _node(
            "late-displacement",
            "displacement",
            Timeframe.M5,
            t0 + pd.Timedelta(minutes=100),
            formed_at=t0 + pd.Timedelta(minutes=100),
            direction=Direction.LONG,
            lifecycle="active",
        )
    )
    graph._derive_source_relations(
        market_observation(
            asof=t0 + pd.Timedelta(minutes=100),
            price=100.0,
        )
    )
    assert not any(
        edge.relation is SceneEdgeKind.RESPONDS_TO
        and edge.source_node_id == late.node_id
        for edge in graph.edges_asof(t0 + pd.Timedelta(minutes=100))
    )

    no_backfill = TemporalMarketSceneGraph()
    no_backfill.add_node(
        _node("existing-h4", "structure", Timeframe.H4, t0)
    )
    old_m5 = no_backfill.add_node(
        _node(
            "old-m5",
            "displacement",
            Timeframe.M5,
            t0 + pd.Timedelta(minutes=1),
            lifecycle="active",
        )
    )
    no_backfill._derive_source_relations(
        market_observation(
            asof=t0 + pd.Timedelta(minutes=1),
            price=100.0,
        )
    )
    later_h1 = no_backfill.add_node(
        _node(
            "later-h1",
            "structure",
            Timeframe.H1,
            t0 + pd.Timedelta(minutes=2),
        )
    )
    no_backfill._derive_source_relations(
        market_observation(
            asof=t0 + pd.Timedelta(minutes=2),
            price=100.0,
        )
    )
    assert not any(
        edge.source_node_id == old_m5.node_id
        and edge.target_node_id == later_h1.node_id
        for edge in no_backfill.edges
    )
    assert later_h1.node_id not in {
        node.node_id
        for node in no_backfill.nodes_asof(t0 + pd.Timedelta(minutes=1))
    }


def test_focus_and_hypothesis_ambiguity_stay_in_active_context() -> None:
    observation = market_observation()
    focus = select_focus(None, observation, None, None)
    assert set(focus.primary_timeframes).issubset(
        {timeframe.value for timeframe in observation.active_timeframes}
    )
    assert Timeframe.M15.value not in focus.primary_timeframes

    graph = TemporalMarketSceneGraph()
    asof = observation.asof
    context = graph.add_node(
        _node("context", "structure", Timeframe.H4, asof)
    )
    graph.add_node(
        _node(
            "unrelated-ambiguous",
            "displacement",
            Timeframe.M5,
            asof,
            lifecycle="ambiguous",
            status=EvidenceStatus.AMBIGUOUS,
        )
    )
    focused = FocusedObservation(
        asof=asof,
        scene_revision_id=graph.revision_id,
        focus_state=focus,
        nodes=graph.nodes_asof(asof),
        edges=(),
        graph_paths=(),
        unresolved_ambiguities=("unrelated-ambiguous",),
        cross_scale_conflicts=(),
        missing_evidence={"15m.displacement": EvidenceStatus.UNKNOWN},
    )
    step = SequenceStepState(
        step_id="m5_displacement",
        satisfied=False,
        value=0.0,
        observed_at=None,
    )
    belief = SimpleNamespace(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        phase=PlaybookPhase.FORMING,
        sequence=SimpleNamespace(steps=(step,)),
        context_id=context.node_id,
        setup_context_id=None,
        initiating_event_id=None,
        plan=None,
        liquidity_route=None,
        draw_selection=None,
        invalidation=None,
        evidence_revision_id="evidence:1",
    )
    contexts = build_hypothesis_states(
        {"dfp:long": belief},
        graph,
        focused,
    )
    assert len(contexts) == 1
    primary = next(iter(contexts.values()))
    assert primary.ambiguous_evidence == {}
    assert primary.missing_evidence["m5_displacement"] is EvidenceStatus.NOT_OBSERVED


def test_select_focus_uses_current_epoch_conflict_and_ambiguity_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = market_observation()
    asof = observation.asof
    graph = TemporalMarketSceneGraph()
    parent = graph.add_node(
        _node(
            "epoch:0:structure:4H:index-parent",
            "structure",
            Timeframe.H4,
            asof,
            direction=Direction.LONG,
            source_ids=("index-parent",),
            entity_id="index-parent",
        )
    )
    displacement = graph.add_node(
        _node(
            "epoch:0:bos:1H:index-opposition",
            "bos",
            Timeframe.H1,
            asof,
            direction=Direction.SHORT,
            source_ids=("index-opposition",),
            entity_id="index-opposition",
            semantic_attributes=(
                ("scope", "opposed"),
                ("post_break_state", "accepted"),
            ),
        )
    )
    ambiguous = graph.add_node(
        _node(
            "epoch:0:manipulation_candidate:1m:index-ambiguity",
            "manipulation_candidate",
            Timeframe.M1,
            asof,
            lifecycle="ambiguous",
            status=EvidenceStatus.AMBIGUOUS,
        )
    )
    graph._last_asof = asof
    original_nodes_asof = graph.nodes_asof
    original_edges_asof = graph.edges_asof

    def no_history_scan(_: pd.Timestamp):
        raise AssertionError("current focus selection must use live indexes")

    monkeypatch.setattr(graph, "nodes_asof", no_history_scan)
    monkeypatch.setattr(graph, "edges_asof", no_history_scan)
    ambiguous_focus = select_focus(None, observation, None, graph)
    assert ambiguous.node_id in graph._current_epoch_ambiguous_node_ids
    assert ambiguous_focus.resolution_status is EvidenceStatus.AMBIGUOUS

    opposition = graph.add_edge(
        SceneEdge(
            edge_id="edge:index-opposition",
            source_node_id=displacement.node_id,
            relation=SceneEdgeKind.OPPOSES,
            target_node_id=parent.node_id,
            observed_at=asof,
            source_ids=(displacement.node_id, parent.node_id),
        )
    )
    prior_focus = replace(
        ambiguous_focus,
        primary_timeframes=(Timeframe.H1.value, Timeframe.M1.value),
        resolution_status=EvidenceStatus.UNKNOWN,
        hypothesis_id="hyp:index-parent",
    )
    hypothesis = SimpleNamespace(
        phase=PlaybookPhase.FORMING,
        plan=None,
        key="displacement_first_pullback:short",
        context_id="index-parent",
        setup_context_id=None,
        initiating_event_id=None,
    )
    previous = SimpleNamespace(
        focus_state=prior_focus,
        dominant_hypothesis_id="hyp:index-parent",
        context_hypotheses={
            "hyp:index-parent": SimpleNamespace(
                context_root_ids=("index-parent",)
            )
        },
        ranked=lambda: [hypothesis],
    )
    global_context = update_global_market_context(
        None,
        observation,
        SceneGraphDelta(
            asof=asof,
            revision_id=graph.revision_id,
            added_node_ids=(parent.node_id, displacement.node_id, ambiguous.node_id),
            added_edge_ids=(opposition.edge_id,),
        ),
        graph,
    )
    conflicting_focus = select_focus(
        previous,
        observation,
        None,
        graph,
        global_context,
    )
    assert opposition.edge_id in (
        graph._current_epoch_active_opposes_edge_ids
    )
    assert "root_related_cross_scale_conflict" in conflicting_focus.reason_codes
    assert conflicting_focus.resolution_status is EvidenceStatus.CONFLICTING

    resolved_at = asof + pd.Timedelta(minutes=1)
    graph.add_edge(
        replace(
            opposition,
            observed_at=resolved_at,
            lifecycle="closed",
            resolution_reason="test_resolution",
        )
    )
    graph.add_node(
        replace(
            ambiguous,
            observed_at=resolved_at,
            lifecycle="confirmed",
            ambiguity_state=EvidenceStatus.CONFIRMED,
        )
    )
    graph._last_asof = resolved_at
    resolved_observation = market_observation(asof=resolved_at)
    resolved_context = update_global_market_context(
        global_context,
        resolved_observation,
        SceneGraphDelta(
            asof=resolved_at,
            revision_id=graph.revision_id,
            revised_node_ids=(ambiguous.node_id,),
            revised_edge_ids=(opposition.edge_id,),
        ),
        graph,
    )
    resolved_focus = select_focus(
        None,
        resolved_observation,
        None,
        graph,
        resolved_context,
    )
    assert opposition.edge_id not in (
        graph._current_epoch_active_opposes_edge_ids
    )
    assert ambiguous.node_id not in graph._current_epoch_ambiguous_node_ids
    assert resolved_focus.resolution_status is EvidenceStatus.FORMING

    # Historical focus reconstruction continues to use the revision ledger.
    monkeypatch.setattr(graph, "nodes_asof", original_nodes_asof)
    monkeypatch.setattr(graph, "edges_asof", original_edges_asof)
    historical_focus = select_focus(None, observation, None, graph)
    assert historical_focus.resolution_status is EvidenceStatus.AMBIGUOUS


def test_focus_retains_root_candidate_for_unrelated_conflict() -> None:
    observation = market_observation()
    graph = TemporalMarketSceneGraph()
    asof = observation.asof
    graph.add_node(
        _node(
            "epoch:0:structure:4H:focus-root",
            "structure",
            Timeframe.H4,
            asof,
            source_ids=("focus-root",),
        )
    )
    left = graph.add_node(_node("unrelated-left", "bos", Timeframe.M5, asof))
    right = graph.add_node(
        _node(
            "unrelated-right",
            "structure",
            Timeframe.H1,
            asof,
            direction=Direction.SHORT,
        )
    )
    graph.add_edge(
        SceneEdge(
            edge_id="edge:unrelated-opposes",
            source_node_id=left.node_id,
            relation=SceneEdgeKind.OPPOSES,
            target_node_id=right.node_id,
            observed_at=asof,
            source_ids=(left.node_id, right.node_id),
        )
    )
    prior_focus = FocusState(
        asof=asof - pd.Timedelta(minutes=1),
        primary_timeframes=(Timeframe.M5.value, Timeframe.M1.value),
        reason_codes=("new_key_event",),
        question="is the prior question resolved",
        resolution_status=EvidenceStatus.CONFIRMED,
        switched=False,
        hypothesis_id="hyp:focus-root",
    )
    hypothesis = SimpleNamespace(
        candidate_id="hyp:focus-root",
        required_root_id="focus-root",
        phase=PlaybookPhase.FORMING,
        plan=None,
        key="displacement_first_pullback:long",
        context_id="focus-root",
        setup_context_id=None,
        initiating_event_id=None,
    )
    previous = SimpleNamespace(
        focus_state=prior_focus,
        dominant_hypothesis_id="hyp:focus-root",
        context_hypotheses={
            "hyp:focus-root": SimpleNamespace(
                context_root_ids=("focus-root",)
            )
        },
        ranked=lambda: [hypothesis],
    )
    unrelated_delta = SceneGraphDelta(
        asof=asof,
        revision_id=graph.revision_id,
        added_node_ids=(left.node_id, right.node_id),
        added_edge_ids=("edge:unrelated-opposes",),
    )
    updated = select_focus(
        previous,
        observation,
        unrelated_delta,
        graph,
    )
    assert updated.primary_timeframes == prior_focus.primary_timeframes
    assert updated.reason_codes == ("new_key_event",)
    assert not updated.switched
    unresolved_previous = SimpleNamespace(
        **{
            **vars(previous),
            "focus_state": replace(
                prior_focus,
                resolution_status=EvidenceStatus.UNKNOWN,
            ),
        }
    )
    retained = select_focus(unresolved_previous, observation, None, graph)
    assert retained.primary_timeframes == prior_focus.primary_timeframes
    assert retained.reason_codes == ("new_key_event",)
    assert not retained.switched


def test_focus_reselects_when_current_root_evidence_revision_changes() -> None:
    observation = market_observation()
    graph = TemporalMarketSceneGraph()
    asof = observation.asof
    root = graph.add_node(
        _node(
            "epoch:0:displacement:5m:focus-root",
            "displacement",
            Timeframe.M5,
            asof,
            source_ids=("focus-root",),
            entity_id="focus-root",
        )
    )
    graph._last_asof = asof
    prior_focus = FocusState(
        asof=asof - pd.Timedelta(minutes=1),
        primary_timeframes=(Timeframe.M5.value, Timeframe.M1.value),
        reason_codes=("root_market_thesis",),
        question="what exact causal event is required next",
        resolution_status=EvidenceStatus.FORMING,
        hypothesis_id="hyp:focus-root",
        switched=False,
    )
    hypothesis = SimpleNamespace(
        candidate_id="hyp:focus-root",
        required_root_id="focus-root",
        market_thesis_root_id="focus-root",
        phase=PlaybookPhase.FORMING,
        plan=None,
        key="displacement_first_pullback:long",
        context_id="focus-root",
        setup_context_id=None,
        episode_id=None,
        initiating_event_id=None,
        entry_location_id=None,
        invalidation=None,
        sequence=None,
        liquidity_route=None,
    )
    prior_thesis = SimpleNamespace(
        root_id="focus-root",
        evidence_revision_id="thesis-evidence:prior",
        ambiguous_evidence=(),
        source_timeframe=Timeframe.M5,
    )
    current_thesis = SimpleNamespace(
        root_id="focus-root",
        evidence_revision_id="thesis-evidence:current",
        ambiguous_evidence=(),
        source_timeframe=Timeframe.M5,
    )
    prior_context = SimpleNamespace(
        open_market_theses=(prior_thesis,),
        material_conflicts=(),
    )
    current_context = SimpleNamespace(
        updated_at=asof,
        scene_revision_id=graph.revision_id,
        open_market_theses=(current_thesis,),
        material_conflicts=(),
        authority_timeframe=None,
        invalidated_source_ids=(),
        path_blocker_ids=(),
    )
    previous = SimpleNamespace(
        focus_state=prior_focus,
        global_context=prior_context,
        ranked=lambda: [hypothesis],
        resolve_hypothesis=lambda _: hypothesis,
    )

    updated = select_focus(
        previous,
        observation,
        SceneGraphDelta(
            asof=asof,
            revision_id=graph.revision_id,
            revised_node_ids=(root.node_id,),
        ),
        graph,
        current_context,
        preferred_candidate=hypothesis,
    )

    assert updated.hypothesis_id == hypothesis.candidate_id
    assert "root_market_thesis_revised" in updated.reason_codes
    assert updated.switched


def _lsr_focus_candidate(
    *,
    candidate_id: str = "lsr-candidate:zone-a",
    episode_id: str = "lsr-episode:zone-a",
    location_id: str = "entry-location:zone-a",
    path_id: str = "entry-path:zone-a",
    trigger_path_id: str | None = None,
    evidence_revision_id: str = "lsr-context-evidence:1",
) -> SimpleNamespace:
    trigger_path_id = path_id if trigger_path_id is None else trigger_path_id
    return SimpleNamespace(
        candidate_id=candidate_id,
        required_root_id="manipulation:shared-root",
        market_thesis_root_id="manipulation:shared-root",
        context_id="pool-path:shared-root",
        context_thesis_id="lsr-context:shared-root",
        parent_context_thesis_id="lsr-context:shared-root",
        setup_context_id=episode_id,
        episode_id=episode_id,
        initiating_event_id="manipulation:shared-root",
        entry_location_id=location_id,
        entry_path_id=path_id,
        context_metadata={
            "lsr_manipulation_id": "manipulation:shared-root",
            "lsr_pool_path_id": "pool-path:shared-root",
            "lsr_displacement_id": "displacement:shared-root",
            "lsr_entry_zone_id": f"zone:{location_id}",
        },
        selected_trigger=SimpleNamespace(
            trigger_id=f"trigger:{location_id}",
            setup_id=episode_id,
            entry_location_id=location_id,
            entry_path_id=trigger_path_id,
            source_entity_id=f"trigger-entity:{location_id}",
            source_event_id=f"trigger-event:{location_id}",
        ),
        sequence=SimpleNamespace(setup_id=episode_id, steps=()),
        liquidity_route=None,
        invalidation=None,
        plan=None,
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        direction=Direction.SHORT,
        phase=PlaybookPhase.WAITING_TRIGGER,
        key=candidate_id,
        evidence_revision_id=evidence_revision_id,
    )


def _focus_context(
    graph: TemporalMarketSceneGraph,
    asof: pd.Timestamp,
    *,
    open_market_theses: tuple[SimpleNamespace, ...] = (),
    material_conflicts: tuple[SimpleNamespace, ...] = (),
    invalidated_source_ids: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        updated_at=asof,
        scene_revision_id=graph.revision_id,
        open_market_theses=open_market_theses,
        material_conflicts=material_conflicts,
        invalidated_source_ids=invalidated_source_ids,
        path_blocker_ids=(),
        authority_timeframe=None,
    )


def _prior_focus_belief(
    candidate: SimpleNamespace,
    context: SimpleNamespace,
    asof: pd.Timestamp,
) -> SimpleNamespace:
    return SimpleNamespace(
        focus_state=FocusState(
            asof=asof - pd.Timedelta(minutes=1),
            primary_timeframes=(Timeframe.M5.value, Timeframe.M1.value),
            reason_codes=("exact_episode_focus",),
            question="has the exact child path produced its trigger",
            resolution_status=EvidenceStatus.FORMING,
            hypothesis_id=candidate.candidate_id,
            switched=False,
        ),
        global_context=context,
        ranked=lambda: [candidate],
        resolve_hypothesis=lambda _: candidate,
    )


def test_lsr_focus_uses_stable_context_revision_after_scene_root_projection_closes() -> None:
    observation = market_observation()
    asof = observation.asof
    graph = TemporalMarketSceneGraph()
    candidate = _lsr_focus_candidate()
    prior_thesis = SimpleNamespace(
        root_id=candidate.required_root_id,
        evidence_revision_id="scene-thesis:transient",
        ambiguous_evidence=(),
        source_timeframe=Timeframe.H1,
    )
    prior_context = _focus_context(
        graph,
        asof - pd.Timedelta(minutes=1),
        open_market_theses=(prior_thesis,),
    )
    previous = _prior_focus_belief(candidate, prior_context, asof)
    current_context = _focus_context(graph, asof)

    retained = select_focus(
        previous,
        observation,
        None,
        graph,
        current_context,
        preferred_candidate=candidate,
    )

    assert retained.reason_codes == ("exact_episode_focus",)
    assert retained.hypothesis_id == candidate.candidate_id
    assert not retained.switched

    revised_candidate = SimpleNamespace(
        **{
            **vars(candidate),
            "evidence_revision_id": "lsr-context-evidence:2",
        }
    )
    revised = select_focus(
        previous,
        observation,
        None,
        graph,
        current_context,
        preferred_candidate=revised_candidate,
    )
    assert "root_market_thesis_revised" in revised.reason_codes
    assert revised.switched


def test_lsr_focus_reselects_for_exact_path_not_sibling_path_invalidation() -> None:
    observation = market_observation()
    asof = observation.asof
    graph = TemporalMarketSceneGraph()
    candidate = _lsr_focus_candidate()
    prior_context = _focus_context(
        graph,
        asof - pd.Timedelta(minutes=1),
    )
    previous = _prior_focus_belief(candidate, prior_context, asof)

    sibling_only = select_focus(
        previous,
        observation,
        None,
        graph,
        _focus_context(
            graph,
            asof,
            invalidated_source_ids=("entry-path:zone-b",),
        ),
        preferred_candidate=candidate,
    )
    assert sibling_only.reason_codes == ("exact_episode_focus",)
    assert not sibling_only.switched

    exact = select_focus(
        previous,
        observation,
        None,
        graph,
        _focus_context(
            graph,
            asof,
            invalidated_source_ids=(candidate.entry_path_id,),
        ),
        preferred_candidate=candidate,
    )
    assert "active_thesis_source_invalidated" in exact.reason_codes
    assert exact.question == "which frozen source closed the active thesis"
    assert exact.switched


def test_lsr_focus_rejects_mixed_episode_path_and_trigger_binding() -> None:
    observation = market_observation()
    asof = observation.asof
    graph = TemporalMarketSceneGraph()
    malformed = _lsr_focus_candidate(trigger_path_id="entry-path:zone-b")
    prior_context = _focus_context(
        graph,
        asof - pd.Timedelta(minutes=1),
    )
    previous = _prior_focus_belief(malformed, prior_context, asof)

    retained = select_focus(
        previous,
        observation,
        None,
        graph,
        _focus_context(
            graph,
            asof,
            invalidated_source_ids=(
                malformed.entry_path_id,
                malformed.selected_trigger.entry_path_id,
            ),
        ),
        preferred_candidate=malformed,
    )

    assert retained.reason_codes == ("exact_episode_focus",)
    assert not retained.switched

    contexts = build_hypothesis_states(
        {malformed.candidate_id: malformed},
        None,
        None,
    )
    roots = contexts[malformed.candidate_id].context_root_ids
    assert malformed.entry_path_id not in roots
    assert malformed.selected_trigger.entry_path_id not in roots


def test_lsr_hypothesis_state_roots_include_only_the_exact_child_path() -> None:
    candidate = _lsr_focus_candidate()
    contexts = build_hypothesis_states(
        {candidate.candidate_id: candidate},
        None,
        None,
    )

    roots = contexts[candidate.candidate_id].context_root_ids
    assert candidate.episode_id in roots
    assert candidate.entry_location_id in roots
    assert candidate.entry_path_id in roots
    assert "entry-path:zone-b" not in roots


def test_lsr_focus_conflict_binds_summary_namespace_to_exact_child_path() -> None:
    observation = market_observation()
    asof = observation.asof
    graph = TemporalMarketSceneGraph()
    candidate = _lsr_focus_candidate()
    prior_context = _focus_context(
        graph,
        asof - pd.Timedelta(minutes=1),
    )
    previous = _prior_focus_belief(candidate, prior_context, asof)

    def conflict(path_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            conflict_id=f"conflict:{path_id}",
            event_id=path_id,
            source_node_id=f"epoch:0:path_sequence:1m:{path_id}",
            target_node_id="epoch:0:structure:4H:unrelated",
            role=GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE,
            affected_hypothesis_ids=(
                "liquidity_sweep_reversal:short",
            ),
        )

    sibling_only = select_focus(
        previous,
        observation,
        None,
        graph,
        _focus_context(
            graph,
            asof,
            material_conflicts=(conflict("entry-path:zone-b"),),
        ),
        preferred_candidate=candidate,
    )
    assert sibling_only.reason_codes == ("exact_episode_focus",)
    assert not sibling_only.switched

    exact = select_focus(
        previous,
        observation,
        None,
        graph,
        _focus_context(
            graph,
            asof,
            material_conflicts=(conflict(candidate.entry_path_id),),
        ),
        preferred_candidate=candidate,
    )
    assert "root_related_cross_scale_conflict" in exact.reason_codes
    assert exact.resolution_status is EvidenceStatus.CONFLICTING
    assert exact.switched


def test_focus_conflict_excludes_internal_pullback_but_keeps_h1_transition() -> None:
    graph = TemporalMarketSceneGraph()
    asof = _clock("2025-01-06 10:00")
    parent = graph.add_node(
        _node(
            "epoch:0:structure:4H:parent",
            "structure",
            Timeframe.H4,
            asof,
            direction=Direction.LONG,
            source_ids=("parent",),
            entity_id="parent",
        )
    )
    pullback = graph.add_node(
        _node(
            "epoch:0:swing:1m:pullback",
            "swing",
            Timeframe.M1,
            asof,
            direction=Direction.SHORT,
            source_ids=("pullback",),
            entity_id="pullback",
        )
    )
    displacement = graph.add_node(
        _node(
            "epoch:0:bos:1H:opposite",
            "bos",
            Timeframe.H1,
            asof,
            direction=Direction.SHORT,
            lifecycle="confirmed",
            source_ids=("opposite",),
            entity_id="opposite",
            semantic_attributes=(
                ("scope", "opposed"),
                ("post_break_state", "accepted"),
            ),
        )
    )
    for source in (pullback, displacement):
        graph.add_edge(
            SceneEdge(
                edge_id=f"edge:{source.entity_id}",
                source_node_id=source.node_id,
                relation=SceneEdgeKind.OPPOSES,
                target_node_id=parent.node_id,
                observed_at=asof,
                source_ids=(source.node_id, parent.node_id),
            )
        )
    graph._last_asof = asof
    observation = market_observation(asof=asof)
    update_global_market_context(
        None,
        observation,
        SceneGraphDelta(
            asof=asof,
            revision_id=graph.revision_id,
            added_node_ids=(parent.node_id, pullback.node_id, displacement.node_id),
            added_edge_ids=("edge:pullback", "edge:opposite"),
        ),
        graph,
    )
    focus = FocusState(
        asof=asof,
        primary_timeframes=(Timeframe.H1.value, Timeframe.M5.value, Timeframe.M1.value),
        reason_codes=("initial_context",),
        question="is the higher structure materially opposed",
        resolution_status=EvidenceStatus.FORMING,
        switched=False,
    )
    focused = graph.query(focus, context_ids=("parent",))
    assert focused.cross_scale_conflicts == ("edge:opposite",)
    assert any(edge.edge_id == "edge:pullback" for edge in focused.edges)

    unready = graph.query(
        focus,
        context_ids=("parent",),
        ready_timeframes=(Timeframe.M5.value, Timeframe.M1.value),
    )
    assert unready.cross_scale_conflicts == ()
    assert (
        unready.missing_evidence["1H.semantic_history"]
        is EvidenceStatus.UNKNOWN
    )


def test_current_bounded_query_uses_hot_indexes_and_keeps_old_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = TemporalMarketSceneGraph()
    t0 = _clock("2025-01-06 09:00")
    root = graph.add_node(
        _node(
            "epoch:0:structure:4H:old-root",
            "structure",
            Timeframe.H4,
            t0,
            source_ids=("old-root",),
        )
    )
    for index in range(6):
        graph.add_node(
            _node(
                f"recent-{index}",
                "micro_bos",
                Timeframe.M1,
                t0 + pd.Timedelta(minutes=index + 1),
            )
        )
    asof = t0 + pd.Timedelta(minutes=6)
    graph._last_asof = asof
    focus = FocusState(
        asof=asof,
        primary_timeframes=(Timeframe.M1.value,),
        reason_codes=("bounded_query_test",),
        question="retain the frozen root",
        resolution_status=EvidenceStatus.FORMING,
        switched=False,
    )

    def no_history_scan(_: pd.Timestamp):
        raise AssertionError("current scene query must not scan history")

    monkeypatch.setattr(graph, "nodes_asof", no_history_scan)
    monkeypatch.setattr(graph, "edges_asof", no_history_scan)
    result = graph.query(
        focus,
        context_ids=("old-root",),
        maximum_nodes=2,
    )
    assert len(result.nodes) == 2
    assert root.node_id in {node.node_id for node in result.nodes}


def test_m15_query_marks_unsupported_group3_evidence_unknown_not_false() -> None:
    observation = market_observation()
    m15 = replace(
        observation.frame(Timeframe.H1),
        timeframe=Timeframe.M15,
        swings=(),
        structures=(),
        structure_breaks=(),
        candle_structure=None,
        support_resistance=(),
        liquidity_pools=(),
        fair_value_gaps=(),
        order_blocks=(),
        dealing_ranges=(),
    )
    observation = replace(
        observation,
        frames={**observation.frames, Timeframe.M15: m15},
        active_timeframes=(
            Timeframe.H4,
            Timeframe.H1,
            Timeframe.M15,
            Timeframe.M5,
            Timeframe.M1,
        ),
        scale_registry_id="scale:test-five",
    )
    graph = TemporalMarketSceneGraph()
    graph.update(observation)
    focus = FocusState(
        asof=observation.asof,
        primary_timeframes=(Timeframe.M15.value,),
        reason_codes=("bridge_query",),
        question="what bridge evidence is causally observable",
        resolution_status=EvidenceStatus.FORMING,
        switched=False,
    )
    focused = graph.query(focus)
    assert focused.missing_evidence["15m.displacement"] is EvidenceStatus.UNKNOWN
    assert focused.missing_evidence["15m.fvg"] is EvidenceStatus.UNKNOWN
    assert focused.missing_evidence["15m.order_block"] is EvidenceStatus.UNKNOWN
    assert not any(
        node.timeframe == Timeframe.M15.value
        and node.kind in {"displacement", "fvg", "order_block"}
        for node in focused.nodes
    )


def test_m15_typed_liquidity_reaches_scene_graph() -> None:
    prior = market_observation()
    source = next(
        item
        for item in prior.liquidity_inventory
        if item.timeframe is Timeframe.H1 and item.side == "above"
    )
    bridge_item = replace(
        source,
        item_id="m15-bridge-above",
        timeframe=Timeframe.M15,
        kind="previous_session_high",
        price=100.5,
        lower_bound=100.5,
        upper_bound=100.5,
        source_ids=("previous-session:m15",),
    )
    m15 = replace(
        prior.frame(Timeframe.H1),
        timeframe=Timeframe.M15,
        swings=(),
        structures=(),
        structure_breaks=(),
        candle_structure=None,
        support_resistance=(),
        liquidity_pools=(),
        fair_value_gaps=(),
        order_blocks=(),
        dealing_ranges=(),
    )
    prior = replace(
        prior,
        frames={**prior.frames, Timeframe.M15: m15},
        liquidity_inventory=(*prior.liquidity_inventory, bridge_item),
        active_timeframes=(
            Timeframe.H4,
            Timeframe.H1,
            Timeframe.M15,
            Timeframe.M5,
            Timeframe.M1,
        ),
        scale_registry_id=scale_registry_id(MODEL_SCALE_SPECS),
    )
    graph = TemporalMarketSceneGraph()
    graph.update(prior)
    assert any(
        node.kind == "liquidity"
        and node.timeframe == Timeframe.M15.value
        and node.entity_id == bridge_item.item_id
        for node in graph.nodes
    )


def test_observer_transfers_revised_edge_ids_into_observation() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    update = reader.on_bar(
        Bar(
            _clock("2025-03-10 09:00"),
            100,
            100.5,
            99.5,
            100.25,
            10,
            "NQH5",
            1,
        )
    )
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=True,
        )
    )
    delta = SceneGraphDelta(
        asof=update.asof,
        revision_id="scene:test-revision",
        revised_edge_ids=("edge:closed-test",),
    )
    # The Eye publishes an unstamped observation; the Engine owns the graph and
    # stamps its delta onto that exact object.
    observation = observer.observe(update)
    assert observation.scene_revision_id is None

    engine = object.__new__(ContinuousSMCEngine)
    engine.observer = observer
    engine.scene_graph = SimpleNamespace(update=lambda observation: delta)
    engine.last_scene_delta = None
    projected = engine._project_scene_graph(observation)

    assert engine.last_scene_delta is delta
    assert projected.scene_revised_edge_ids == ("edge:closed-test",)


def test_scene_delta_canonicalizes_unordered_revised_edge_identities() -> None:
    delta = SceneGraphDelta(
        asof=_clock("2025-03-10 09:01"),
        revision_id="scene:canonical-revised-edges",
        revised_edge_ids=("edge:z", "edge:a"),
    )

    assert delta.revised_edge_ids == ("edge:a", "edge:z")


def test_runtime_compaction_bounds_cold_history_and_fails_closed() -> None:
    graph = TemporalMarketSceneGraph()
    start = _clock("2025-01-06 10:00")
    current = market_observation(asof=start, price=100.0)
    graph.update(current)
    initial_revision = graph.revision_id
    old_node = graph.add_node(
        _node(
            "cold-terminal",
            "displacement",
            Timeframe.M5,
            start,
            lifecycle="exhausted",
            status=EvidenceStatus.INVALIDATED,
            source_ids=("cold-terminal",),
            entity_id="cold-terminal",
        )
    )
    counts = []
    for cycle in range(2):
        for minute in range(1, 81):
            asof = start + pd.Timedelta(
                minutes=cycle * 80 + minute
            )
            current = replace_market_observation(current, asof=asof)
            graph.update(current)
            graph.add_node(
                _node(
                    f"cold-terminal:{cycle}:{minute}",
                    "displacement",
                    Timeframe.M5,
                    asof,
                    lifecycle="exhausted",
                    status=EvidenceStatus.INVALIDATED,
                    source_ids=(f"cold-terminal:{cycle}:{minute}",),
                    entity_id=f"cold-terminal:{cycle}:{minute}",
                )
            )
        result = graph.compact_runtime_history(current)
        counts.append(result["after"]["nodes"])
    assert graph.revision_id != initial_revision
    assert graph.revision_id.startswith("scene:r")
    assert graph.last_asof == current.asof
    assert graph.history_retention_floor == (
        current.asof - pd.Timedelta(minutes=60)
    )
    assert max(counts) <= counts[0] + 4
    assert old_node.node_id not in {node.node_id for node in graph.nodes}
    with pytest.raises(ValueError, match="predates the retained"):
        graph.nodes_asof(start)
    with pytest.raises(ValueError, match="predates the retained"):
        graph.edges_asof(start)
    with pytest.raises(ValueError, match="complete node history"):
        graph.node_history(next(iter(graph._nodes)))
    with pytest.raises(ValueError, match="complete edge history"):
        graph.edge_history("edge:any")
    with pytest.raises(ValueError, match="complete scene graph export"):
        graph.write_parquet("unused-after-compaction")


def test_compacted_stale_revision_discloses_new_source_at_current_clock() -> None:
    graph = TemporalMarketSceneGraph()
    disclosure_clock = _clock("2025-01-06 12:00")
    physical_clock = disclosure_clock - pd.Timedelta(hours=3)
    observation = market_observation(asof=disclosure_clock)
    graph.update(observation)

    # Isolate the explicit nodes below from the helper observation's delta.
    graph._current_added_nodes = []
    graph._current_revised_nodes = []
    graph._current_added_edges = []
    graph._current_revised_edges = []
    first_source = graph.add_node(
        _node(
            "epoch:0:swing:1H:first-source",
            "swing",
            Timeframe.H1,
            physical_clock,
            source_ids=("first-source",),
            entity_id="first-source",
        )
    )
    second_source = graph.add_node(
        _node(
            "epoch:0:swing:1H:second-source",
            "swing",
            Timeframe.H1,
            physical_clock,
            source_ids=("second-source",),
            entity_id="second-source",
        )
    )
    target = graph.add_node(
        _node(
            "epoch:0:structure:1H:stale-target",
            "structure",
            Timeframe.H1,
            physical_clock,
            lifecycle="forming",
            status=EvidenceStatus.FORMING,
            source_ids=("stale-target", "first-source"),
            entity_id="stale-target",
        )
    )
    graph._derive_source_relations(observation)

    first_edge = next(
        edge
        for edge in graph.edges
        if edge.source_node_id == target.node_id
        and edge.target_node_id == first_source.node_id
        and edge.relation is SceneEdgeKind.SOURCED_FROM
    )
    assert first_edge.observed_at == physical_clock

    result = graph.compact_runtime_history(observation)
    assert physical_clock < result["history_retention_floor"]
    graph._current_added_nodes = []
    graph._current_revised_nodes = []
    graph._current_added_edges = []
    graph._current_revised_edges = []
    graph.add_node(
        replace(
            target,
            source_ids=(
                *target.source_ids,
                "second-source",
            ),
            revision_id="",
        )
    )
    next_observation = replace_market_observation(
        observation,
        asof=disclosure_clock + pd.Timedelta(minutes=1),
    )
    graph._derive_source_relations(next_observation)

    second_edge = next(
        edge
        for edge in graph.edges
        if edge.source_node_id == target.node_id
        and edge.target_node_id == second_source.node_id
        and edge.relation is SceneEdgeKind.SOURCED_FROM
    )
    assert second_edge.observed_at == next_observation.asof
    assert first_edge == graph._edges[first_edge.edge_id]


def test_runtime_compaction_retains_current_and_materialized_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = TemporalMarketSceneGraph()
    asof = _clock("2025-01-06 12:00")
    observation = market_observation(asof=asof, price=100.0)
    graph.update(observation)

    hot = graph.add_node(
        _node(
            "long-lived-hot",
            "liquidity",
            Timeframe.H1,
            asof - pd.Timedelta(hours=4),
            lifecycle="visible",
            source_ids=("long-lived-hot",),
            entity_id="long-lived-hot",
        )
    )
    terminal_state = SimpleNamespace(
        zone_id="cooled-but-materialized",
        timeframe=Timeframe.M15,
        lifecycle="reaccepted",
        side="support",
        formed_at=asof - pd.Timedelta(hours=5),
        confirmed_at=asof - pd.Timedelta(hours=4),
        broken_at=asof - pd.Timedelta(hours=3),
        reaccepted_at=asof - pd.Timedelta(hours=2),
        lower_bound=99.0,
        upper_bound=99.5,
        member_swing_ids=("terminal-source-swing",),
        source_kind="structural_swing",
        structural_rank="intermediate",
        is_protected_swing=False,
        zone_role="both",
    )
    terminal = graph._adapt_snapshot_state(
        terminal_state,
        asof=asof,
        frame_state=True,
    )
    assert terminal is not None
    monkeypatch.setattr(
        graph,
        "_state_records",
        lambda _: ((terminal_state, True),),
    )

    result = graph.compact_runtime_history(observation)
    retained_ids = {node.node_id for node in graph.nodes}
    assert hot.node_id in retained_ids
    assert terminal.node_id in retained_ids

    next_observation = replace_market_observation(
        observation,
        asof=asof + pd.Timedelta(minutes=1),
    )
    delta = graph.update(next_observation)
    assert terminal.node_id not in delta.added_node_ids
    assert hot.node_id not in delta.added_node_ids
    assert result["after"]["nodes"] <= result["before"]["nodes"]


def test_neighbor_order_is_stable_across_processes_and_resumes() -> None:
    """Adjacency is stored in sets, so its order must be imposed, not inherited.

    ``_neighbors`` feeds a breadth-first path search that returns the first
    route it reaches.  While the yield order followed set iteration, two equally
    short routes were separated by nothing but the interpreter's hash seed, so
    the same event log could read out a different path in a different process
    -- and a resumed graph could disagree with the one it was pickled from.
    """

    graph = TemporalMarketSceneGraph()
    asof = _clock("2025-01-06 10:00")
    source = graph.add_node(
        _node("order-source", "structure", Timeframe.H1, asof)
    )
    # Names chosen so insertion order, alphabetical order and hash order are
    # all different; only an imposed order survives a pickle round trip.
    for name in ("zeta", "alpha", "mu", "beta", "omega"):
        target = graph.add_node(
            _node(f"order-{name}", "displacement", Timeframe.M5, asof)
        )
        graph.add_edge(
            SceneEdge(
                edge_id=f"edge:order-{name}",
                source_node_id=source.node_id,
                relation=SceneEdgeKind.ALIGNS_WITH,
                target_node_id=target.node_id,
                observed_at=asof,
                source_ids=(source.node_id, target.node_id),
            )
        )

    observed = tuple(graph._neighbors(source.node_id))
    assert [edge.edge_id for _, edge in observed] == sorted(
        edge.edge_id for _, edge in observed
    )

    resumed = pickle.loads(pickle.dumps(graph))
    resumed.__dict__.pop("_current_neighbors_cache")
    assert tuple(resumed._neighbors(source.node_id)) == observed
