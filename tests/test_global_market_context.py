from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_trader.model import (
    Direction,
    HypothesisSequenceState,
    LiquidityRoute,
    MarketBelief,
    MarketMode,
    PlaybookPhase,
    ScaleRelation,
    SequenceStepState,
    Timeframe,
)
from smc_trader.playbooks import (
    PlaybookBrain,
    _route_terminal_delta_to_global_context,
)
from smc_trader.scene_graph import (
    EvidenceStatus,
    SceneEdge,
    SceneEdgeKind,
    SceneGraphDelta,
    SceneNode,
    StructuralScale,
    TemporalMarketSceneGraph,
    select_focus,
    update_global_market_context,
)

from .helpers import executable_belief, market_observation


TZ = "America/New_York"


def _clock(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=TZ)


def _node(
    identity: str,
    kind: str,
    timeframe: Timeframe,
    observed_at: pd.Timestamp,
    *,
    direction: Direction | None,
    lifecycle: str = "confirmed",
    structural_scale: StructuralScale | None = None,
    bounds: tuple[float, float] | None = (99.0, 101.0),
    source_ids: tuple[str, ...] = (),
    semantic_attributes: tuple[tuple[str, str], ...] = (),
    market_epoch_id: str = "epoch:0",
) -> SceneNode:
    return SceneNode(
        node_id=f"{market_epoch_id}:{kind}:{timeframe.value}:{identity}",
        kind=kind,
        timeframe=timeframe.value,
        structural_scale=(
            structural_scale
            or (
                StructuralScale.EXTERNAL
                if timeframe in {Timeframe.H4, Timeframe.H1}
                else StructuralScale.INTERMEDIATE
                if timeframe is Timeframe.M15
                else StructuralScale.INTERNAL
            )
        ),
        direction=direction,
        formed_at=observed_at,
        confirmed_at=(
            None if lifecycle in {"forming", "started", "pending"} else observed_at
        ),
        observed_at=observed_at,
        lifecycle=lifecycle,
        price_bounds=bounds,
        invalidation_rule="frozen_test_rule",
        ambiguity_state=(
            EvidenceStatus.FORMING
            if lifecycle in {"forming", "started", "pending"}
            else EvidenceStatus.CONFIRMED
        ),
        source_ids=(identity, *source_ids),
        entity_id=identity,
        semantic_attributes=semantic_attributes,
        market_epoch_id=market_epoch_id,
    )


def _delta(
    graph: TemporalMarketSceneGraph,
    asof: pd.Timestamp,
    *,
    nodes: tuple[str, ...] = (),
    edges: tuple[str, ...] = (),
) -> SceneGraphDelta:
    return SceneGraphDelta(
        asof=asof,
        revision_id=graph.revision_id,
        added_node_ids=nodes,
        added_edge_ids=edges,
    )


def _add_authority_and_draws(
    graph: TemporalMarketSceneGraph,
    asof: pd.Timestamp,
) -> tuple[SceneNode, ...]:
    h4 = graph.add_node(
        _node(
            "h4-up",
            "structure",
            Timeframe.H4,
            asof,
            direction=Direction.LONG,
            source_ids=("h4-protected-low",),
        )
    )
    h1 = graph.add_node(
        _node(
            "h1-up",
            "structure",
            Timeframe.H1,
            asof,
            direction=Direction.LONG,
        )
    )
    above_near = graph.add_node(
        _node(
            "pd-high",
            "liquidity",
            Timeframe.H1,
            asof,
            direction=Direction.LONG,
            lifecycle="visible",
            bounds=(103.0, 103.0),
            structural_scale=StructuralScale.EXTERNAL,
            semantic_attributes=(("inventory_kind", "previous_day_high"),),
        )
    )
    above_far = graph.add_node(
        _node(
            "pw-high",
            "liquidity",
            Timeframe.H4,
            asof,
            direction=Direction.LONG,
            lifecycle="visible",
            bounds=(108.0, 108.0),
            structural_scale=StructuralScale.EXTERNAL,
            semantic_attributes=(("inventory_kind", "previous_week_high"),),
        )
    )
    below = graph.add_node(
        _node(
            "pd-low",
            "liquidity",
            Timeframe.H1,
            asof,
            direction=Direction.SHORT,
            lifecycle="visible",
            bounds=(96.0, 96.0),
            structural_scale=StructuralScale.EXTERNAL,
            semantic_attributes=(("inventory_kind", "previous_day_low"),),
        )
    )
    return h4, h1, above_near, above_far, below


def test_global_context_bootstraps_from_current_indexes_without_history_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof, price=100.0)
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, asof)
    graph._refresh_current_path_block_edges(observation)
    graph._last_asof = asof

    def no_history_scan(_: pd.Timestamp):
        raise AssertionError("global context must not scan graph history")

    monkeypatch.setattr(graph, "nodes_asof", no_history_scan)
    monkeypatch.setattr(graph, "edges_asof", no_history_scan)
    context = update_global_market_context(
        None,
        observation,
        _delta(graph, asof, nodes=tuple(node.node_id for node in nodes)),
        graph,
    )

    assert context.authority_timeframe is Timeframe.H4
    assert context.authority_direction is Direction.LONG
    assert "h4-up" in context.authority_source_ids
    assert context.market_mode is MarketMode.DIRECTIONAL
    assert context.scale_relations[Timeframe.H1.value] is ScaleRelation.ALIGNED
    assert context.scale_relations[Timeframe.M15.value] is ScaleRelation.UNKNOWN
    assert context.external_draw_candidates == {
        "above": ("pd-high", "pw-high"),
        "below": ("pd-low",),
    }
    assert context.path_blocker_ids == ("pd-high",)
    assert context.material_conflicts == ()
    assert not context.dislocated
    MarketBelief(
        asof=asof,
        hypotheses={},
        scene_revision_id=graph.revision_id,
        global_context=context,
    )


def test_brain_public_update_routes_delta_through_global_context_and_focus() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof, price=100.0)
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, asof)
    graph._refresh_current_path_block_edges(observation)
    graph._last_asof = asof
    delta = _delta(
        graph,
        asof,
        nodes=tuple(node.node_id for node in nodes),
    )

    belief = PlaybookBrain().update(
        observation,
        scene_graph=graph,
        scene_delta=delta,
    )

    assert belief.global_context is not None
    assert belief.global_context.authority_timeframe is Timeframe.H4
    assert belief.global_context.authority_direction is Direction.LONG
    assert belief.global_context.market_mode is MarketMode.DIRECTIONAL
    assert belief.scene_revision_id == graph.revision_id
    assert belief.focus_state is not None
    assert "global_authority_context" in belief.focus_state.reason_codes


def test_global_context_no_delta_keeps_identity_and_uses_live_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, t0)
    observation0 = market_observation(asof=t0)
    graph._refresh_current_path_block_edges(observation0)
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        observation0,
        _delta(graph, t0, nodes=tuple(node.node_id for node in nodes)),
        graph,
    )

    def no_history_scan(_: pd.Timestamp):
        raise AssertionError("no-delta update must use current indexes")

    monkeypatch.setattr(graph, "nodes_asof", no_history_scan)
    monkeypatch.setattr(graph, "edges_asof", no_history_scan)
    t1 = t0 + pd.Timedelta(minutes=1)
    observation1 = market_observation(asof=t1, price=100.25)
    heartbeat = graph.add_node(
        _node(
            "m1-candle",
            "candle_structure",
            Timeframe.M1,
            t1,
            direction=None,
            lifecycle="observed",
        )
    )
    graph._refresh_current_path_block_edges(observation1)
    graph._last_asof = t1

    def no_current_kind_scan(_: str):
        raise AssertionError(
            "irrelevant delta must reuse the prior static context"
        )

    monkeypatch.setattr(graph, "_current_kind_nodes", no_current_kind_scan)
    context1 = update_global_market_context(
        context0,
        observation1,
        _delta(graph, t1, nodes=(heartbeat.node_id,)),
        graph,
    )

    assert context1.updated_at == t1
    assert context1.scene_revision_id != context0.scene_revision_id
    assert context1.scene_revision_id == graph.revision_id
    assert context1.authority_source_ids == context0.authority_source_ids
    assert context1.scale_relations == context0.scale_relations
    assert context1.invalidated_source_ids == ()


def test_global_context_epoch_boundary_discards_prior_authority() -> None:
    t0 = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, t0)
    observation0 = market_observation(asof=t0)
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        observation0,
        _delta(graph, t0, nodes=tuple(node.node_id for node in nodes)),
        graph,
    )

    t1 = t0 + pd.Timedelta(minutes=1)
    graph._market_epoch_no = 1
    graph._market_epoch_id = "epoch:1"
    graph._clear_current_epoch_indexes()
    h1_down = graph.add_node(
        _node(
            "h1-down",
            "structure",
            Timeframe.H1,
            t1,
            direction=Direction.SHORT,
            market_epoch_id="epoch:1",
        )
    )
    graph._last_asof = t1
    context1 = update_global_market_context(
        context0,
        market_observation(asof=t1),
        _delta(graph, t1, nodes=(h1_down.node_id,)),
        graph,
    )

    assert context1.market_epoch_id == "epoch:1"
    assert context1.authority_timeframe is Timeframe.H1
    assert context1.authority_direction is Direction.SHORT
    assert context1.invalidated_source_ids == context0.authority_source_ids
    assert context1.market_mode is MarketMode.DIRECTIONAL


def test_mature_h1_balance_is_balanced_even_without_directional_authority() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)
    graph = TemporalMarketSceneGraph()
    mature = graph.add_node(
        _node(
            "h1-mature-balance",
            "range",
            Timeframe.H1,
            asof,
            direction=None,
            lifecycle="mature",
        )
    )
    graph._last_asof = asof

    context = update_global_market_context(
        None,
        observation,
        _delta(graph, asof, nodes=(mature.node_id,)),
        graph,
    )

    assert context.market_mode is MarketMode.BALANCED
    assert context.authority_timeframe is Timeframe.H1
    assert context.authority_direction is None
    assert "h1-mature-balance" in context.authority_source_ids


def test_h4_direction_remains_global_authority_over_local_h1_balance() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node(
            "h4-direction",
            "structure",
            Timeframe.H4,
            asof,
            direction=Direction.LONG,
        )
    )
    mature = graph.add_node(
        _node(
            "h1-local-balance",
            "range",
            Timeframe.H1,
            asof,
            direction=None,
            lifecycle="mature",
        )
    )
    graph._last_asof = asof

    context = update_global_market_context(
        None,
        observation,
        _delta(graph, asof, nodes=(h4.node_id, mature.node_id)),
        graph,
    )

    assert context.market_mode is MarketMode.DIRECTIONAL
    assert context.authority_timeframe is Timeframe.H4
    assert context.authority_direction is Direction.LONG


def test_terminal_delta_is_bound_to_prior_frozen_hypothesis_source() -> None:
    t0 = _clock("2025-01-06 10:00")
    prior_observation = market_observation(asof=t0)
    previous = executable_belief(prior_observation)
    hypothesis = next(iter(previous.hypotheses.values()))
    assert hypothesis.invalidation is not None
    source_id = hypothesis.invalidation.source_level_id

    t1 = t0 + pd.Timedelta(minutes=1)
    observation = market_observation(asof=t1)
    graph = TemporalMarketSceneGraph()
    terminal = replace(
        _node(
            source_id,
            "swing",
            Timeframe.H4,
            t1,
            direction=Direction.SHORT,
            lifecycle="broken",
        ),
        ambiguity_state=EvidenceStatus.INVALIDATED,
    )
    terminal = graph.add_node(terminal)
    graph._last_asof = t1
    delta = _delta(graph, t1, nodes=(terminal.node_id,))
    context = update_global_market_context(
        None,
        observation,
        delta,
        graph,
    )
    routed = _route_terminal_delta_to_global_context(
        context,
        previous,
        delta,
        graph,
    )

    assert source_id in routed.invalidated_source_ids


def test_normal_displacement_exhaustion_does_not_invalidate_latched_episode() -> None:
    t0 = _clock("2025-01-06 10:00")
    prior_observation = market_observation(asof=t0)
    base = executable_belief(prior_observation)
    hypothesis = next(iter(base.hypotheses.values()))
    displacement_id = "m5-displacement:episode-1"
    setup_id = "dfp-episode:1"
    sequence = HypothesisSequenceState(
        protocol_version="test",
        protocol_hash="test-protocol-hash",
        setup_id=setup_id,
        steps=(
            SequenceStepState(
                step_id="m5_displacement_zone",
                satisfied=True,
                value=1.0,
                observed_at=t0,
                source_ids=(displacement_id,),
            ),
        ),
        started_at=t0,
    )
    hypothesis = replace(
        hypothesis,
        phase=PlaybookPhase.ARMED,
        phase_started_at=t0,
        plan=None,
        sequence=sequence,
        setup_context_id=setup_id,
        context_id="dfp-context:1",
        initiating_event_id=displacement_id,
    )
    previous = MarketBelief(t0, {hypothesis.key: hypothesis})

    t1 = t0 + pd.Timedelta(minutes=1)
    observation = market_observation(asof=t1)
    graph = TemporalMarketSceneGraph()
    exhausted = replace(
        _node(
            displacement_id,
            "displacement",
            Timeframe.M5,
            t1,
            direction=Direction.LONG,
            lifecycle="exhausted",
        ),
        ambiguity_state=EvidenceStatus.INVALIDATED,
    )
    exhausted = graph.add_node(exhausted)
    graph._last_asof = t1
    delta = _delta(graph, t1, nodes=(exhausted.node_id,))
    context = update_global_market_context(
        None,
        observation,
        delta,
        graph,
    )

    routed = _route_terminal_delta_to_global_context(
        context,
        previous,
        delta,
        graph,
    )

    assert displacement_id not in routed.invalidated_source_ids


@pytest.mark.parametrize(
    ("liquidity_id", "expected_invalidated"),
    (("path-blocker:1", False), ("above-level", True)),
)
def test_consumed_liquidity_only_invalidates_a_frozen_draw_role(
    liquidity_id: str,
    expected_invalidated: bool,
) -> None:
    t0 = _clock("2025-01-06 10:00")
    observation0 = market_observation(asof=t0)
    base = executable_belief(observation0)
    hypothesis = next(iter(base.hypotheses.values()))
    route = LiquidityRoute(
        route_id="route:1",
        selected_at=t0,
        context_draw_id="above-level",
        intermediate_liquidity_ids=(),
        primary_deliverable_target_id="above-level",
        terminal_draw_id="above-level",
        path_blocker_ids=("path-blocker:1",),
    )
    hypothesis = replace(hypothesis, liquidity_route=route)
    previous = MarketBelief(t0, {hypothesis.key: hypothesis})

    t1 = t0 + pd.Timedelta(minutes=1)
    observation1 = market_observation(asof=t1)
    graph = TemporalMarketSceneGraph()
    consumed = replace(
        _node(
            liquidity_id,
            "liquidity",
            Timeframe.M5,
            t1,
            direction=Direction.LONG,
            lifecycle="consumed",
        ),
        ambiguity_state=EvidenceStatus.INVALIDATED,
    )
    consumed = graph.add_node(consumed)
    graph._last_asof = t1
    delta = _delta(graph, t1, nodes=(consumed.node_id,))
    context = update_global_market_context(
        None,
        observation1,
        delta,
        graph,
    )
    routed = _route_terminal_delta_to_global_context(
        context,
        previous,
        delta,
        graph,
    )

    assert (
        liquidity_id in routed.invalidated_source_ids
    ) is expected_invalidated


def test_same_direction_authority_refresh_is_not_transition_and_transition_decays() -> None:
    t0 = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    original = graph.add_node(
        _node(
            "h1-up-old",
            "structure",
            Timeframe.H1,
            t0,
            direction=Direction.LONG,
        )
    )
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        market_observation(asof=t0),
        _delta(graph, t0, nodes=(original.node_id,)),
        graph,
    )

    t1 = t0 + pd.Timedelta(minutes=1)
    refreshed = graph.add_node(
        _node(
            "h1-up-new",
            "structure",
            Timeframe.H1,
            t1,
            direction=Direction.LONG,
        )
    )
    graph._last_asof = t1
    context1 = update_global_market_context(
        context0,
        market_observation(asof=t1),
        _delta(graph, t1, nodes=(refreshed.node_id,)),
        graph,
    )
    assert context1.authority_source_ids != context0.authority_source_ids
    assert context1.market_mode is MarketMode.DIRECTIONAL

    # A one-clock authority transition without a still-active conflict must
    # not remain latched merely because the following bar has no context delta.
    transition = replace(context1, market_mode=MarketMode.TRANSITION)
    t2 = t1 + pd.Timedelta(minutes=1)
    heartbeat = graph.add_node(
        _node(
            "m1-candle-after-transition",
            "candle_structure",
            Timeframe.M1,
            t2,
            direction=None,
            lifecycle="observed",
        )
    )
    graph._last_asof = t2
    context2 = update_global_market_context(
        transition,
        market_observation(asof=t2),
        _delta(graph, t2, nodes=(heartbeat.node_id,)),
        graph,
    )
    assert context2.market_mode is MarketMode.DIRECTIONAL


def test_scale_relation_distinguishes_pullback_from_material_opposition_and_binds_conflict() -> None:
    t0 = _clock("2025-01-06 10:00")
    observation0 = market_observation(asof=t0)
    graph = TemporalMarketSceneGraph()
    h1 = graph.add_node(
        _node(
            "h1-up",
            "structure",
            Timeframe.H1,
            t0,
            direction=Direction.LONG,
        )
    )
    m5_pullback = graph.add_node(
        _node(
            "m5-down",
            "structure",
            Timeframe.M5,
            t0,
            direction=Direction.SHORT,
        )
    )
    pullback_edge = graph.add_edge(
        SceneEdge(
            edge_id="edge:normal-pullback",
            source_node_id=m5_pullback.node_id,
            relation=SceneEdgeKind.OPPOSES,
            target_node_id=h1.node_id,
            observed_at=t0,
            source_ids=(m5_pullback.node_id, h1.node_id),
        )
    )
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        observation0,
        _delta(
            graph,
            t0,
            nodes=(h1.node_id, m5_pullback.node_id),
            edges=(pullback_edge.edge_id,),
        ),
        graph,
    )
    assert context0.scale_relations[Timeframe.M5.value] is ScaleRelation.NORMAL_PULLBACK
    assert context0.material_conflicts == ()

    t1 = t0 + pd.Timedelta(minutes=1)
    displacement = graph.add_node(
        _node(
            "m5-opposite-displacement",
            "displacement",
            Timeframe.M5,
            t1,
            direction=Direction.SHORT,
            lifecycle="active",
        )
    )
    material_edge = graph.add_edge(
        SceneEdge(
            edge_id="edge:material-opposition",
            source_node_id=displacement.node_id,
            relation=SceneEdgeKind.OPPOSES,
            target_node_id=h1.node_id,
            observed_at=t1,
            source_ids=(displacement.node_id, h1.node_id),
        )
    )
    observation1 = market_observation(asof=t1)
    graph._last_asof = t1
    context1 = update_global_market_context(
        context0,
        observation1,
        _delta(
            graph,
            t1,
            nodes=(displacement.node_id,),
            edges=(material_edge.edge_id,),
        ),
        graph,
    )

    assert context1.scale_relations[Timeframe.M5.value] is ScaleRelation.MATERIAL_OPPOSITION
    assert context1.market_mode is MarketMode.TRANSITION
    assert context1.dislocated
    assert len(context1.material_conflicts) == 1
    conflict = context1.material_conflicts[0]
    assert conflict.conflict_id == material_edge.edge_id
    assert conflict.event_id == "m5-opposite-displacement"
    assert conflict.observed_at == t1
    assert conflict.source_timeframe is Timeframe.M5
    assert conflict.target_timeframe is Timeframe.H1
    assert conflict.source_direction is Direction.SHORT
    assert conflict.target_direction is Direction.LONG
    assert conflict.structural_scale == "internal"
    assert set(conflict.affected_hypothesis_ids) == {
        "displacement_first_pullback:short",
        "liquidity_sweep_reversal:short",
        "failed_auction_value_return:short",
        "displacement_first_pullback:long",
        "liquidity_sweep_reversal:long",
        "failed_auction_value_return:long",
    }
    assert context1.scale_relations[Timeframe.H1.value] is ScaleRelation.ALIGNED
    focus = select_focus(
        None,
        observation1,
        _delta(graph, t1),
        graph,
        context1,
    )
    assert focus.resolution_status is EvidenceStatus.CONFLICTING
    assert material_edge.edge_id in focus.trigger_event_ids

    unrelated = select_focus(
        executable_belief(observation1),
        observation1,
        _delta(graph, t1),
        graph,
        context1,
    )
    assert "cross_scale_conflict" not in unrelated.reason_codes
    assert material_edge.edge_id not in unrelated.trigger_event_ids


def test_focus_prioritizes_frozen_draw_and_delivery_blocker_changes() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, asof)
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        observation,
        _delta(graph, asof, nodes=tuple(node.node_id for node in nodes)),
        graph,
    )
    base = executable_belief(observation)
    hypothesis = next(iter(base.hypotheses.values()))
    route = LiquidityRoute(
        route_id="route:focus",
        selected_at=asof,
        context_draw_id="above-level",
        intermediate_liquidity_ids=(),
        primary_deliverable_target_id="above-level",
        terminal_draw_id="above-level",
        path_blocker_ids=("path-blocker:1",),
    )
    hypothesis = replace(hypothesis, liquidity_route=route)
    belief = MarketBelief(asof, {hypothesis.key: hypothesis})

    blocker_focus = select_focus(
        belief,
        observation,
        _delta(graph, asof),
        graph,
        replace(context, path_blocker_ids=("path-blocker:1",)),
    )
    assert "delivery_path_blocked" in blocker_focus.reason_codes
    assert "path-blocker:1" in blocker_focus.trigger_event_ids

    draw_focus = select_focus(
        belief,
        observation,
        _delta(graph, asof),
        graph,
        replace(context, invalidated_source_ids=("above-level",)),
    )
    assert "frozen_draw_consumed" in draw_focus.reason_codes
    assert "above-level" in draw_focus.trigger_event_ids


def test_global_context_rejects_stale_focus_clock() -> None:
    t0 = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, t0)
    observation = market_observation(asof=t0)
    graph._last_asof = t0
    context = update_global_market_context(
        None,
        observation,
        _delta(graph, t0, nodes=tuple(node.node_id for node in nodes)),
        graph,
    )
    with pytest.raises(ValueError, match="stale global market context"):
        select_focus(
            None,
            market_observation(asof=t0 + pd.Timedelta(minutes=1)),
            None,
            graph,
            context,
        )
