from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.model import (
    Bar,
    Direction,
    EventKind,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    MarketEvent,
    Playbook,
    PlaybookPhase,
    SequenceStepState,
    Timeframe,
)
from smc_trader.observation import CausalObserver, ObserverConfig
from smc_trader.scene_graph import (
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
    development_scale_specs,
    legacy_scale_specs,
    parse_scale_specs,
    scale_registry_id,
    select_focus,
    supplement_focus_once,
)

from .helpers import market_observation


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
    )


def test_scale_registry_hashes_the_full_contract_and_rejects_coercion() -> None:
    base = legacy_scale_specs(history_limit=64)
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
                for item in legacy_scale_specs()
            ]
        )


def test_m15_emits_only_on_completed_bar_across_dst_and_contract_reset() -> None:
    specs = development_scale_specs(history_limit=64)
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
    specs = development_scale_specs(history_limit=64)
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
        edge.relation is SceneEdgeKind.RESPONDS_TO
        and edge.source_node_id == reverse_displacement.node_id
        and edge.target_node_id == manipulation.node_id
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
    stage_focus = replace(
        focus,
        primary_timeframes=(Timeframe.M5.value, Timeframe.M1.value),
        prior_timeframes=(Timeframe.H1.value, Timeframe.M5.value),
        phase_at_selection=PlaybookPhase.WAITING_TRIGGER.value,
        hypothesis_id="hyp:context-a",
    )
    prior_stage = SimpleNamespace(
        phase=PlaybookPhase.WAITING_TRIGGER,
        key="displacement_first_pullback:long",
    )
    current_stage = SimpleNamespace(
        phase=PlaybookPhase.DELIVERING,
        key="displacement_first_pullback:long",
    )
    five_scale = supplement_focus_once(
        stage_focus,
        prior_stage,
        current_stage,
        tuple(
            spec.native_timeframe
            for spec in development_scale_specs()
            if spec.enabled and spec.native_timeframe is not None
        ),
    )
    assert Timeframe.M15.value in five_scale.supplemental_timeframes
    legacy = supplement_focus_once(
        stage_focus,
        prior_stage,
        current_stage,
        tuple(spec.native_timeframe for spec in legacy_scale_specs()),
    )
    assert Timeframe.M15.value not in legacy.supplemental_timeframes
    assert five_scale.hypothesis_id == "hyp:context-a"

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
    assert len(contexts) <= 2
    primary = next(iter(contexts.values()))
    assert primary.ambiguous_evidence == {}
    assert primary.missing_evidence["m5_displacement"] is EvidenceStatus.NOT_OBSERVED


def test_focus_resolution_unsticks_but_unrelated_conflict_does_not_switch() -> None:
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
        supplemental_timeframes=(),
        reason_codes=("new_key_event",),
        trigger_event_ids=(),
        question="is the prior question resolved",
        resolution_status=EvidenceStatus.CONFIRMED,
        switched=False,
        switched_at=None,
        hypothesis_id="hyp:focus-root",
        phase_at_selection=PlaybookPhase.FORMING.value,
    )
    hypothesis = SimpleNamespace(
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
    updated = select_focus(previous, observation, None, graph)
    assert "focus_question_resolved_or_invalidated" in updated.reason_codes
    assert "cross_scale_conflict" not in updated.reason_codes
    assert updated.switched
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
    assert retained.reason_codes == ("focus_retained_no_switch_event",)


def test_focus_conflict_excludes_internal_pullback_but_keeps_displacement() -> None:
    graph = TemporalMarketSceneGraph()
    asof = _clock("2025-01-06 10:00")
    parent = graph.add_node(
        _node(
            "epoch:0:structure:1H:parent",
            "structure",
            Timeframe.H1,
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
            "epoch:0:displacement:5m:opposite",
            "displacement",
            Timeframe.M5,
            asof,
            direction=Direction.SHORT,
            lifecycle="active",
            source_ids=("opposite",),
            entity_id="opposite",
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
    focus = FocusState(
        asof=asof,
        primary_timeframes=(Timeframe.H1.value, Timeframe.M5.value, Timeframe.M1.value),
        supplemental_timeframes=(),
        reason_codes=("initial_context",),
        trigger_event_ids=(),
        question="is the higher structure materially opposed",
        resolution_status=EvidenceStatus.FORMING,
        switched=False,
        switched_at=None,
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


def test_bounded_query_never_drops_an_old_context_root() -> None:
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
        supplemental_timeframes=(),
        reason_codes=("bounded_query_test",),
        trigger_event_ids=(),
        question="retain the frozen root",
        resolution_status=EvidenceStatus.FORMING,
        switched=False,
        switched_at=None,
    )
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
        liquidity=(),
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
        supplemental_timeframes=(),
        reason_codes=("bridge_query",),
        trigger_event_ids=(),
        question="what bridge evidence is causally observable",
        resolution_status=EvidenceStatus.FORMING,
        switched=False,
        switched_at=None,
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


def test_m15_bridge_liquidity_reaches_minute_sweep_memory() -> None:
    prior = market_observation()
    source = prior.frame(Timeframe.H1).liquidity[1]
    bridge_level = replace(
        source,
        level_id="m15-bridge-above",
        timeframe=Timeframe.M15,
        price=100.5,
    )
    m15 = replace(
        prior.frame(Timeframe.H1),
        timeframe=Timeframe.M15,
        liquidity=(bridge_level,),
    )
    prior = replace(
        prior,
        frames={**prior.frames, Timeframe.M15: m15},
        active_timeframes=(
            Timeframe.H4,
            Timeframe.H1,
            Timeframe.M15,
            Timeframe.M5,
            Timeframe.M1,
        ),
        scale_registry_id="scale:test-five",
    )
    specs = development_scale_specs(history_limit=64)
    reader = CausalMarketReader(scale_specs=specs)
    update = reader.on_bar(
        Bar(
            prior.asof,
            100.0,
            101.0,
            99.75,
            100.25,
            10,
            "NQH5",
            1,
        )
    )
    observer = CausalObserver(ObserverConfig(scale_specs=specs))
    observer._prior = prior
    observer._record_minute_events(update, prior.frames)
    assert any(
        event.kind is EventKind.LIQUIDITY_SWEEP
        and bridge_level.level_id in event.source_ids
        for event in observer.memory.recent()
    )


def test_observer_transfers_revised_edge_ids_into_observation() -> None:
    reader = CausalMarketReader()
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
    observer = CausalObserver()
    delta = SceneGraphDelta(
        asof=update.asof,
        revision_id="scene:test-revision",
        revised_edge_ids=("edge:closed-test",),
    )
    observer.scene_graph = SimpleNamespace(update=lambda observation: delta)
    observation = observer.observe(update)
    assert observation.scene_revised_edge_ids == ("edge:closed-test",)
