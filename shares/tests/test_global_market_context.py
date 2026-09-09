from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from contract.market import (
    Direction,
    MarketMode,
    PlaybookPhase,
    ScaleRelation,
    StructuralLevel,
    Timeframe,
)
from contract.eye import (
    BOSLifecycle,
    BOSPostBreakState,
    BOSScope,
    BreakOfStructureState,
    DisplacementObservation,
    DisplacementTransitionObservation,
    FrameObservation,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    ManipulationState,
    MarketObservation,
    StructureLifecycle,
    StructureSequenceState,
)
from contract.brain import (
    ContextThesisState,
    DeliveryObstruction,
    DirectionalObstructionView,
    GlobalConflictRole,
    HypothesisSequenceState,
    LiquidityRoute,
    MarketBelief,
    ScaleRelationState,
    SequenceStepState,
)
from shares.core.scene_graph import (
    build_open_market_theses,
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

from shares.tests.helpers import (
    executable_belief,
    market_observation,
    replace_market_observation,
)


TZ = "America/New_York"


def _clock(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=TZ)


def test_effective_probability_does_not_collapse_uncalibrated_dimensions() -> None:
    belief = executable_belief()
    hypothesis = next(iter(belief.hypotheses.values()))

    assert hypothesis.effective_probability == 0.0
    calibrated = replace(
        hypothesis,
        calibration_version="typed-calibration:test",
        delivery_quality=0.65,
    )
    assert calibrated.effective_probability == pytest.approx(0.65)
    waiting = replace(calibrated, phase=PlaybookPhase.WAITING_TRIGGER)
    assert waiting.effective_probability == 0.0


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
    descriptive_metrics: tuple[tuple[str, float], ...] = (),
    market_epoch_id: str = "epoch:0",
    resolution_reason: str | None = None,
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
        resolution_reason=resolution_reason,
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
        descriptive_metrics=descriptive_metrics,
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


def _structure_state(
    identity: str,
    timeframe: Timeframe,
    direction: Direction,
    clock: pd.Timestamp,
) -> StructureSequenceState:
    return StructureSequenceState(
        structure_id=identity,
        timeframe=timeframe,
        direction=direction,
        lifecycle=StructureLifecycle.CONFIRMED,
        formed_at=clock - pd.Timedelta(hours=3),
        confirmed_at=clock - pd.Timedelta(hours=1),
        broken_at=None,
        high_run=2,
        low_run=2,
        sequence_count=2,
        latest_high_id=f"{identity}:high",
        latest_low_id=f"{identity}:low",
        protected_swing_id=f"{identity}:protected",
        protected_price=97.0 if direction is Direction.LONG else 103.0,
        cumulative_magnitude_atr=1.5,
        age_bars=1,
    )


def _with_structures(
    observation: MarketObservation,
    h4: StructureSequenceState,
    h1: StructureSequenceState,
) -> MarketObservation:
    return replace(
        observation,
        frames={
            **observation.frames,
            Timeframe.H4: replace(
                observation.frame(Timeframe.H4),
                structures=(h4,),
            ),
            Timeframe.H1: replace(
                observation.frame(Timeframe.H1),
                structures=(h1,),
            ),
        },
    )


def _accepted_bos(
    identity: str,
    timeframe: Timeframe,
    direction: Direction,
    clock: pd.Timestamp,
    *,
    source_structure_id: str,
) -> BreakOfStructureState:
    return BreakOfStructureState(
        bos_id=identity,
        timeframe=timeframe,
        direction=direction,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.OPPOSED,
        target_swing_id=f"{identity}:target",
        source_structure_id=source_structure_id,
        target_price=99.0,
        target_ticks=396,
        pending_at=clock - pd.Timedelta(minutes=3),
        resolved_at=clock - pd.Timedelta(minutes=2),
        age_bars=1,
        strength=0.8,
        break_bar_id=f"{identity}:bar",
        break_distance_atr=0.5,
        source_displacement_id=f"{identity}:displacement",
        mss_qualified=True,
        post_break_state=BOSPostBreakState.ACCEPTED,
        accepted_at=clock - pd.Timedelta(minutes=1),
    )


def _pool_manipulation(
    swept_at: pd.Timestamp,
    *,
    lifecycle: ManipulationLifecycle,
    resolved_at: pd.Timestamp | None = None,
) -> ManipulationState:
    reaccepted = lifecycle is ManipulationLifecycle.REACCEPTED
    return ManipulationState(
        manipulation_id="manipulation:pool:public",
        protocol_hash="group4:test",
        source_group12_protocol_hash="group12:test",
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M1,
        lifecycle=lifecycle,
        side="above",
        source_kind="formed_liquidity_pool",
        source_id="pool:public",
        source_protocol_hash="group12:test",
        source_timeframe=Timeframe.H1,
        source_inventory_item_id="pool:pool:public",
        source_inventory_lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        coincident_source_ids=(),
        source_formed_at=swept_at - pd.Timedelta(minutes=5),
        source_eligible_at=swept_at - pd.Timedelta(minutes=1),
        source_lower_bound=100.75,
        source_upper_bound=101.0,
        formed_at=swept_at,
        confirmed_at=swept_at,
        swept_at=swept_at,
        reaccepted_at=resolved_at if reaccepted else None,
        accepted_outside_at=None,
        resolved_at=resolved_at if reaccepted else None,
        state_started_at=resolved_at if reaccepted else swept_at,
        last_updated_at=resolved_at if reaccepted else swept_at,
        sweep_extreme=101.25,
        close_outside_on_sweep=False,
        reentry_price=100.75 if reaccepted else None,
        resolved_side=None,
        outside_completed_bars=0,
        penetration_atr=0.8,
        strength=0.8,
        age_1m_bars=(0 if resolved_at is None else int(
            (resolved_at - swept_at) / pd.Timedelta(minutes=1)
        )),
        transition_reason=(
            "reentry_held_inside_swept_boundary"
            if reaccepted
            else "source_swept"
        ),
        censored_at=None,
        reentry_candidate_at=(
            resolved_at - pd.Timedelta(minutes=1)
            if reaccepted and resolved_at is not None
            else None
        ),
        reentry_candidate_price=100.75 if reaccepted else None,
        inside_hold_bars=1 if reaccepted else 0,
        reentry_failed_at=None,
        outside_run=0,
        outside_run_side=None,
        deadline_at=None,
        deadline_elapsed=False,
        crossed_source_ids=("pool:public",),
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


def _with_hard_blockers(
    context,
    *identities: str,
):
    hard = tuple(
        DeliveryObstruction(
            obstruction_id=identity,
            timeframe=Timeframe.H1,
            direction=Direction.SHORT,
            side="above",
            lower_bound=101.0,
            upper_bound=101.25,
            hard=True,
            source_kind="accepted_bos",
            source_ids=(identity,),
            structural_scope="external",
            acceptance_state="accepted",
        )
        for identity in identities
    )
    return replace(
        context,
        obstruction_views={
            Direction.LONG.value: DirectionalObstructionView(
                direction=Direction.LONG,
                nearest_draw_id=None,
                nearest_draw_price=None,
                hard_barriers=hard,
                soft_frictions=(),
            ),
            Direction.SHORT.value: DirectionalObstructionView(
                direction=Direction.SHORT,
                nearest_draw_id=None,
                nearest_draw_price=None,
                hard_barriers=(),
                soft_frictions=(),
            ),
        },
    )


def _with_m15(observation: MarketObservation) -> MarketObservation:
    frame = FrameObservation(
        timeframe=Timeframe.M15,
        cutoff=observation.asof,
        bars=20,
        metrics={},
        ready=True,
    )
    return replace(
        observation,
        frames={**observation.frames, Timeframe.M15: frame},
        active_timeframes=(*observation.active_timeframes, Timeframe.M15),
        scale_registry_id="test-five-scale-registry",
    )


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
    assert context.scale_relations[Timeframe.H1.value] is ScaleRelation.UNKNOWN
    assert not context.scale_relation_details[Timeframe.H1.value].graph_connected
    assert context.scale_relations[Timeframe.M15.value] is ScaleRelation.UNKNOWN
    assert context.external_draw_candidates == {
        "above": ("pd-high", "pw-high"),
        "below": ("pd-low",),
    }
    # Reference liquidity remains a draw.  The dynamic pd-high -> pw-high
    # price-order edge does not grant structural hard-barrier authority.
    assert context.path_blocker_ids == ()
    assert context.material_conflicts == ()
    assert not context.dislocated
    MarketBelief(
        asof=asof,
        hypotheses={},
        scene_revision_id=graph.revision_id,
        global_context=context,
    )


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


def test_no_semantic_delta_refreshes_price_ranked_external_draw_head() -> None:
    t0 = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, t0)
    observation0 = market_observation(asof=t0, price=100.0)
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        observation0,
        _delta(graph, t0, nodes=tuple(node.node_id for node in nodes)),
        graph,
    )
    assert context0.external_draw_candidates["above"] == (
        "pd-high",
        "pw-high",
    )
    assert (
        context0.obstruction_views[Direction.LONG.value].nearest_draw_id
        == "pd-high"
    )

    # This deliberately isolates the query-time ranking contract: no
    # liquidity lifecycle transition is supplied on this clock.
    t1 = t0 + pd.Timedelta(minutes=1)
    observation1 = market_observation(asof=t1, price=106.0)
    heartbeat = graph.add_node(
        _node(
            "m1-draw-rank-heartbeat",
            "candle_structure",
            Timeframe.M1,
            t1,
            direction=None,
            lifecycle="observed",
        )
    )
    graph._last_asof = t1
    context1 = update_global_market_context(
        context0,
        observation1,
        _delta(graph, t1, nodes=(heartbeat.node_id,)),
        graph,
    )

    assert context1.external_draw_candidates["above"] == (
        "pw-high",
        "pd-high",
    )
    assert (
        context1.obstruction_views[Direction.LONG.value].nearest_draw_id
        == "pw-high"
    )


def test_reaccepted_manipulation_is_a_one_clock_explanation_candidate() -> None:
    t0 = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    swept = graph.add_node(
        _node(
            "manipulation:pool:1",
            "manipulation",
            Timeframe.M1,
            t0,
            direction=Direction.SHORT,
            lifecycle="swept",
        )
    )
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        market_observation(asof=t0),
        _delta(graph, t0, nodes=(swept.node_id,)),
        graph,
    )
    assert context0.candidate_structured_episode_ids == (
        "manipulation:pool:1",
    )

    t1 = t0 + pd.Timedelta(minutes=1)
    reaccepted = graph.add_node(
        replace(
            swept,
            observed_at=t1,
            lifecycle="reaccepted",
            ambiguity_state=EvidenceStatus.CONFIRMED,
            resolution_reason="returned_inside_and_held",
            revision_id="",
        )
    )
    graph._last_asof = t1
    context1 = update_global_market_context(
        context0,
        market_observation(asof=t1),
        SceneGraphDelta(
            asof=t1,
            revision_id=graph.revision_id,
            revised_node_ids=(reaccepted.node_id,),
        ),
        graph,
    )
    # Brain's explanation classifier consumes the REACCEPTED state on this
    # exact clock; removing the candidate before classification would lose it.
    assert context1.candidate_structured_episode_ids == (
        "manipulation:pool:1",
    )

    t2 = t1 + pd.Timedelta(minutes=1)
    heartbeat = graph.add_node(
        _node(
            "m1-after-reacceptance",
            "candle_structure",
            Timeframe.M1,
            t2,
            direction=None,
            lifecycle="observed",
        )
    )
    graph._last_asof = t2
    context2 = update_global_market_context(
        context1,
        market_observation(asof=t2),
        _delta(graph, t2, nodes=(heartbeat.node_id,)),
        graph,
    )
    assert context2.candidate_structured_episode_ids == ()


@pytest.mark.parametrize("timeframe", (Timeframe.M15, Timeframe.M5, Timeframe.M1))
def test_lower_scale_continuation_bos_is_not_a_dfp_preforming_root(
    timeframe: Timeframe,
) -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)
    if timeframe is Timeframe.M15:
        observation = _with_m15(observation)
    graph = TemporalMarketSceneGraph()
    root = graph.add_node(
        _node(
            f"{timeframe.value}-continuation", "bos", timeframe, asof,
            direction=Direction.LONG,
            structural_scale=(
                StructuralScale.INTERMEDIATE
                if timeframe is Timeframe.M15
                else StructuralScale.INTERNAL
            ),
            semantic_attributes=(
                ("scope", "continuation"),
                ("post_break_state", "accepted"),
            ),
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        observation,
        _delta(graph, asof, nodes=(root.node_id,)),
        graph,
    )
    assert context.candidate_structured_episode_ids == ()


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
    observation = _with_m15(market_observation(asof=asof))
    graph = TemporalMarketSceneGraph()
    mature = graph.add_node(
        _node(
            "h1-mature-balance",
            "range",
            Timeframe.H1,
            asof,
            direction=None,
            lifecycle="active",
            resolution_reason="balance_claim_confirmed",
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
    assert context.authority_timeframe is None
    assert context.authority_direction is None
    assert context.authority_source_ids == ()
    assert context.balance_context is not None
    assert context.balance_context.context_id == "h1-mature-balance"
    assert context.balance_context.status == "authoritative"


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
            lifecycle="active",
            resolution_reason="balance_claim_confirmed",
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


def test_scale_relation_keeps_local_displacement_but_promotes_h1_bos() -> None:
    t0 = _clock("2025-01-06 10:00")
    observation0 = market_observation(asof=t0)
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node(
            "h4-up",
            "structure",
            Timeframe.H4,
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
            target_node_id=h4.node_id,
            observed_at=t0,
            source_ids=(m5_pullback.node_id, h4.node_id),
        )
    )
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        observation0,
        _delta(
            graph,
            t0,
            nodes=(h4.node_id, m5_pullback.node_id),
            edges=(pullback_edge.edge_id,),
        ),
        graph,
    )
    assert context0.scale_relations[Timeframe.M5.value] is ScaleRelation.NORMAL_PULLBACK
    assert len(context0.material_conflicts) == 1
    assert (
        context0.material_conflicts[0].role
        is GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
    )

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
            target_node_id=h4.node_id,
            observed_at=t1,
            source_ids=(displacement.node_id, h4.node_id),
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

    assert context1.scale_relations[Timeframe.M5.value] is ScaleRelation.NORMAL_PULLBACK
    assert context1.market_mode is MarketMode.DIRECTIONAL
    assert context1.dislocated
    assert len(context1.material_conflicts) == 2
    conflict = next(
        item
        for item in context1.material_conflicts
        if item.conflict_id == material_edge.edge_id
    )
    assert conflict.conflict_id == material_edge.edge_id
    assert conflict.event_id == "m5-opposite-displacement"
    assert conflict.observed_at == t1
    assert conflict.source_timeframe is Timeframe.M5
    assert conflict.target_timeframe is Timeframe.H4
    assert conflict.source_direction is Direction.SHORT
    assert conflict.target_direction is Direction.LONG
    assert conflict.structural_scale == "internal"
    assert conflict.role is GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
    assert set(conflict.affected_hypothesis_ids) == {
        "displacement_first_pullback:short",
        "liquidity_sweep_reversal:short",
        "failed_auction_value_return:short",
        "displacement_first_pullback:long",
        "liquidity_sweep_reversal:long",
        "failed_auction_value_return:long",
    }
    assert context1.scale_relations[Timeframe.H4.value] is ScaleRelation.ALIGNED
    focus = select_focus(
        None,
        observation1,
        _delta(graph, t1),
        graph,
        context1,
    )
    assert focus.resolution_status is not EvidenceStatus.CONFLICTING
    assert "root_related_cross_scale_conflict" not in focus.reason_codes

    unrelated = select_focus(
        executable_belief(observation1),
        observation1,
        _delta(graph, t1),
        graph,
        context1,
    )
    assert "root_related_cross_scale_conflict" not in unrelated.reason_codes

    t2 = t1 + pd.Timedelta(minutes=1)
    h1_bos = graph.add_node(
        _node(
            "h1-bearish-bos",
            "bos",
            Timeframe.H1,
            t2,
            direction=Direction.SHORT,
            structural_scale=StructuralScale.EXTERNAL,
            semantic_attributes=(
                ("scope", "opposed"),
                ("post_break_state", "accepted"),
            ),
        )
    )
    transition_edge = graph.add_edge(
        SceneEdge(
            edge_id="edge:authority-transition",
            source_node_id=h1_bos.node_id,
            relation=SceneEdgeKind.OPPOSES,
            target_node_id=h4.node_id,
            observed_at=t2,
            source_ids=(h1_bos.node_id, h4.node_id),
        )
    )
    observation2 = market_observation(asof=t2)
    graph._last_asof = t2
    context2 = update_global_market_context(
        context1,
        observation2,
        _delta(
            graph,
            t2,
            nodes=(h1_bos.node_id,),
            edges=(transition_edge.edge_id,),
        ),
        graph,
    )
    assert context2.market_mode is MarketMode.TRANSITION
    assert context2.scale_relations[Timeframe.H1.value] is ScaleRelation.MATERIAL_OPPOSITION
    transition = next(
        item
        for item in context2.material_conflicts
        if item.conflict_id == transition_edge.edge_id
    )
    assert transition.role is GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE
    transition_focus = select_focus(
        None,
        observation2,
        _delta(graph, t2),
        graph,
        context2,
    )
    # Without a root-specific action candidate Focus does not absorb a global
    # transition as if it were related to the current thesis.
    assert transition_focus.resolution_status is EvidenceStatus.FORMING
    assert "root_related_cross_scale_conflict" not in transition_focus.reason_codes
    focused = graph.query(
        transition_focus,
        context_ids=("h4-up", "h1-bearish-bos"),
        ready_timeframes=tuple(
            timeframe.value
            for timeframe in observation2.active_timeframes
            if observation2.frame(timeframe).ready
        ),
    )
    assert focused.cross_scale_conflicts == (transition_edge.edge_id,)


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
        _with_hard_blockers(context, "path-blocker:1"),
    )
    assert "delivery_path_blocked" in blocker_focus.reason_codes

    draw_focus = select_focus(
        belief,
        observation,
        _delta(graph, asof),
        graph,
        replace(context, invalidated_source_ids=("above-level",)),
    )
    assert "frozen_draw_consumed" in draw_focus.reason_codes


def test_rejected_h1_bos_is_local_context_not_authority_transition() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node(
            "h4-up",
            "structure",
            Timeframe.H4,
            asof,
            direction=Direction.LONG,
        )
    )
    rejected = graph.add_node(
        _node(
            "h1-rejected-bos",
            "bos",
            Timeframe.H1,
            asof,
            direction=Direction.SHORT,
            structural_scale=StructuralScale.EXTERNAL,
            semantic_attributes=(
                ("scope", "opposed"),
                ("post_break_state", "rejected"),
            ),
        )
    )
    edge = graph.add_edge(
        SceneEdge(
            edge_id="edge:rejected-h1-bos",
            source_node_id=rejected.node_id,
            relation=SceneEdgeKind.OPPOSES,
            target_node_id=h4.node_id,
            observed_at=asof,
            source_ids=(rejected.node_id, h4.node_id),
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        observation,
        _delta(
            graph,
            asof,
            nodes=(h4.node_id, rejected.node_id),
            edges=(edge.edge_id,),
        ),
        graph,
    )
    assert context.market_mode is MarketMode.DIRECTIONAL
    assert context.scale_relations[Timeframe.H1.value] is ScaleRelation.NORMAL_PULLBACK
    assert context.material_conflicts[0].role is GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
    focus = select_focus(None, observation, _delta(graph, asof), graph, context)
    assert focus.resolution_status is not EvidenceStatus.CONFLICTING
    assert "root_related_cross_scale_conflict" not in focus.reason_codes


def test_same_bar_double_terminal_opposition_does_not_claim_authority_invalidation() -> None:
    t0 = _clock("2025-01-06 10:00")
    h4 = _structure_state(
        "h4-long-authority", Timeframe.H4, Direction.LONG, t0
    )
    h1 = _structure_state(
        "h1-short-opposition", Timeframe.H1, Direction.SHORT, t0
    )
    observation0 = _with_structures(market_observation(asof=t0), h4, h1)
    graph = TemporalMarketSceneGraph()
    delta0 = graph.update(observation0)
    context0 = update_global_market_context(
        None,
        observation0,
        delta0,
        graph,
    )
    conflict_id = next(
        conflict.conflict_id
        for conflict in context0.material_conflicts
        if conflict.source_timeframe is Timeframe.H1
        and conflict.target_timeframe is Timeframe.H4
    )

    t1 = t0 + pd.Timedelta(minutes=1)
    broken_h4 = replace(
        h4,
        lifecycle=StructureLifecycle.BROKEN,
        broken_at=t1,
        failure_reason="protected_level_close_break",
    )
    broken_h1 = replace(
        h1,
        lifecycle=StructureLifecycle.BROKEN,
        broken_at=t1,
        failure_reason="protected_level_close_break",
    )
    observation1 = _with_structures(
        market_observation(asof=t1), broken_h4, broken_h1
    )
    delta1 = graph.update(observation1)
    context1 = update_global_market_context(
        context0,
        observation1,
        delta1,
        graph,
    )

    assert conflict_id in delta1.revised_edge_ids
    assert graph._edges[conflict_id].lifecycle == "closed"
    assert not any(
        conflict.conflict_id == conflict_id
        and conflict.role is GlobalConflictRole.AUTHORITY_INVALIDATION
        for conflict in context1.material_conflicts
    )


def test_global_path_blockers_include_single_authority_barrier_not_h1_internal() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof, price=100.0)
    graph = TemporalMarketSceneGraph()
    authority = graph.add_node(
        _node(
            "h4-up",
            "structure",
            Timeframe.H4,
            asof,
            direction=Direction.LONG,
            source_ids=("h4-protected-low",),
        )
    )
    protected = graph.add_node(
        _node(
            "protected-barrier",
            "liquidity",
            Timeframe.H4,
            asof,
            direction=Direction.SHORT,
            lifecycle="visible",
            bounds=(95.0, 95.0),
            source_ids=("h4-protected-low",),
            semantic_attributes=(
                ("structural_rank", "external"),
                ("is_protected_swing", "true"),
            ),
        )
    )
    internal = graph.add_node(
        _node(
            "h1-internal",
            "liquidity",
            Timeframe.H1,
            asof,
            direction=Direction.LONG,
            lifecycle="visible",
            bounds=(104.0, 104.0),
            semantic_attributes=(("structural_rank", "internal"),),
        )
    )
    farther_internal = graph.add_node(
        _node(
            "m1-internal-farther",
            "liquidity",
            Timeframe.M1,
            asof,
            direction=Direction.LONG,
            lifecycle="visible",
            bounds=(106.0, 106.0),
            semantic_attributes=(("structural_rank", "internal"),),
        )
    )
    low_scale_protected = graph.add_node(
        _node(
            "m1-protected-label",
            "liquidity",
            Timeframe.M1,
            asof,
            direction=Direction.LONG,
            lifecycle="visible",
            bounds=(107.0, 107.0),
            source_ids=("h4-protected-low",),
            semantic_attributes=(
                ("structural_rank", "external"),
                ("is_protected_swing", "true"),
            ),
        )
    )
    graph._refresh_current_path_block_edges(observation)
    # Dynamic adjacency describes price order only.  It must not promote the
    # nearer M1/H1 liquidity item into a structural hard barrier.
    assert graph._current_path_block_edges
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        observation,
        _delta(
            graph,
            asof,
            nodes=(
                authority.node_id,
                protected.node_id,
                internal.node_id,
                farther_internal.node_id,
                low_scale_protected.node_id,
            ),
        ),
        graph,
    )
    assert "protected-barrier" in context.path_blocker_ids
    assert "h1-internal" not in context.path_blocker_ids
    assert "m1-internal-farther" not in context.path_blocker_ids
    assert "m1-protected-label" not in context.path_blocker_ids


@pytest.mark.parametrize(
    "timeframe",
    (Timeframe.H4, Timeframe.H1, Timeframe.M15),
)
def test_intact_authority_directly_materializes_protected_swing_boundary(
    timeframe: Timeframe,
) -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof, price=100.0)
    if timeframe is Timeframe.M15:
        observation = _with_m15(observation)
    graph = TemporalMarketSceneGraph()
    protected_id = f"{timeframe.value}-protected-low"
    protected = graph.add_node(
        _node(
            protected_id,
            "swing",
            timeframe,
            asof,
            direction=None,
            structural_scale=StructuralScale.INTERNAL,
            bounds=(95.0, 95.0),
        )
    )
    authority = graph.add_node(
        _node(
            f"{timeframe.value}-long-authority",
            "structure",
            timeframe,
            asof,
            direction=Direction.LONG,
            source_ids=(protected_id,),
            semantic_attributes=(("protected_swing_id", protected_id),),
            bounds=(95.0, 95.0),
        )
    )
    graph._last_asof = asof

    context = update_global_market_context(
        None,
        observation,
        _delta(
            graph,
            asof,
            nodes=(protected.node_id, authority.node_id),
        ),
        graph,
    )

    barrier = next(
        item
        for item in context.obstruction_views[
            Direction.SHORT.value
        ].hard_barriers
        if item.source_kind == "authority_protected_boundary"
    )
    assert barrier.timeframe is timeframe
    assert barrier.lower_bound == pytest.approx(95.0)
    assert barrier.upper_bound == pytest.approx(95.0)
    assert protected_id in barrier.source_ids
    assert authority.entity_id in barrier.source_ids


def test_consumed_protected_liquidity_does_not_remove_intact_authority_boundary() -> None:
    t0 = _clock("2025-01-06 10:00")
    observation0 = market_observation(asof=t0, price=100.0)
    graph = TemporalMarketSceneGraph()
    protected_id = "h4-protected-low"
    protected = graph.add_node(
        _node(
            protected_id,
            "swing",
            Timeframe.H4,
            t0,
            direction=None,
            structural_scale=StructuralScale.INTERNAL,
            bounds=(95.0, 95.0),
        )
    )
    authority = graph.add_node(
        _node(
            "h4-long-authority",
            "structure",
            Timeframe.H4,
            t0,
            direction=Direction.LONG,
            source_ids=(protected_id,),
            semantic_attributes=(("protected_swing_id", protected_id),),
            bounds=(95.0, 95.0),
        )
    )
    liquidity = graph.add_node(
        _node(
            f"swing:{protected_id}",
            "liquidity",
            Timeframe.H4,
            t0,
            direction=Direction.SHORT,
            lifecycle="visible",
            bounds=(95.0, 95.0),
            source_ids=(protected_id,),
            semantic_attributes=(
                ("inventory_kind", "swing"),
                ("structural_rank", "external"),
                ("is_protected_swing", "true"),
            ),
        )
    )
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        observation0,
        _delta(
            graph,
            t0,
            nodes=(protected.node_id, authority.node_id, liquidity.node_id),
        ),
        graph,
    )
    initial = context0.obstruction_views[Direction.SHORT.value].hard_barriers
    assert len(initial) == 1
    assert protected_id in initial[0].source_ids

    t1 = t0 + pd.Timedelta(minutes=1)
    consumed = graph.add_node(
        replace(
            liquidity,
            observed_at=t1,
            lifecycle="consumed",
            ambiguity_state=EvidenceStatus.INVALIDATED,
            resolution_reason="swing_swept",
            revision_id="",
        )
    )
    graph._last_asof = t1
    context1 = update_global_market_context(
        context0,
        market_observation(asof=t1, price=99.0),
        SceneGraphDelta(
            asof=t1,
            revision_id=graph.revision_id,
            revised_node_ids=(consumed.node_id,),
        ),
        graph,
    )
    retained = context1.obstruction_views[
        Direction.SHORT.value
    ].hard_barriers
    assert len(retained) == 1
    assert retained[0].source_kind == "authority_protected_boundary"
    assert retained[0].lower_bound == pytest.approx(95.0)

    t2 = t1 + pd.Timedelta(minutes=1)
    broken_swing = graph.add_node(
        replace(
            protected,
            observed_at=t2,
            lifecycle="broken",
            ambiguity_state=EvidenceStatus.INVALIDATED,
            resolution_reason="close_beyond_swing",
            revision_id="",
        )
    )
    broken_structure = graph.add_node(
        replace(
            authority,
            observed_at=t2,
            lifecycle="broken",
            ambiguity_state=EvidenceStatus.INVALIDATED,
            resolution_reason="protected_level_close_break",
            revision_id="",
        )
    )
    graph._last_asof = t2
    context2 = update_global_market_context(
        context1,
        market_observation(asof=t2, price=94.5),
        SceneGraphDelta(
            asof=t2,
            revision_id=graph.revision_id,
            revised_node_ids=(broken_swing.node_id, broken_structure.node_id),
        ),
        graph,
    )
    assert all(
        item.source_kind != "authority_protected_boundary"
        for item in context2.obstruction_views[
            Direction.SHORT.value
        ].hard_barriers
    )


@pytest.mark.parametrize(
    ("timeframe", "expected_hard"),
    (
        (Timeframe.H1, True),
        (Timeframe.M15, True),
        (Timeframe.M5, False),
    ),
)
def test_only_authority_scale_accepted_bos_can_become_hard_obstruction(
    timeframe: Timeframe,
    expected_hard: bool,
) -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof, price=100.0)
    if timeframe is Timeframe.M15:
        observation = _with_m15(observation)
    graph = TemporalMarketSceneGraph()
    authority = graph.add_node(
        _node(
            "h4-up",
            "structure",
            Timeframe.H4,
            asof,
            direction=Direction.LONG,
            source_ids=("h4-authority-source",),
        )
    )
    bos = graph.add_node(
        _node(
            f"{timeframe.value}-accepted-short",
            "bos",
            timeframe,
            asof,
            direction=Direction.SHORT,
            structural_scale=StructuralScale.INTERMEDIATE,
            bounds=(102.0, 102.0),
            source_ids=("h4-authority-source",),
            semantic_attributes=(
                ("scope", "continuation"),
                ("post_break_state", "accepted"),
            ),
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        observation,
        _delta(graph, asof, nodes=(authority.node_id, bos.node_id)),
        graph,
    )
    hard_source_ids = {
        source_id
        for item in context.obstruction_views[
            Direction.LONG.value
        ].hard_barriers
        for source_id in item.source_ids
    }
    assert (bos.entity_id in hard_source_ids) is expected_hard


def test_qualified_fvg_is_hard_obstruction_only_with_structural_consequence() -> None:
    t0 = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    authority = graph.add_node(
        _node(
            "h4-up", "structure", Timeframe.H4, t0,
            direction=Direction.LONG,
            source_ids=("h4-protected-low",),
        )
    )
    fvg = graph.add_node(
        _node(
            "fvg:short:1", "fvg", Timeframe.M5, t0,
            direction=Direction.SHORT, lifecycle="open",
            bounds=(102.0, 103.0), source_ids=("disp:short:1",),
            semantic_attributes=(("qualification", "displacement_linked"),),
        )
    )
    graph._last_asof = t0
    context0 = update_global_market_context(
        None,
        market_observation(asof=t0, price=100.0),
        _delta(graph, t0, nodes=(authority.node_id, fvg.node_id)),
        graph,
    )
    assert "fvg:short:1" not in context0.path_blocker_ids
    assert any(
        item.obstruction_id == "fvg:short:1"
        for item in context0.obstruction_views[
            Direction.LONG.value
        ].soft_frictions
    )

    t1 = t0 + pd.Timedelta(minutes=1)
    displacement = graph.add_node(
        _node(
            "disp:short:1", "displacement", Timeframe.M5, t1,
            direction=Direction.SHORT, lifecycle="active",
        )
    )
    broken_swing = graph.add_node(
        _node(
            "swing:broken-by-disp", "swing", Timeframe.M5, t1,
            direction=Direction.LONG,
        )
    )
    break_edge = graph.add_edge(
        SceneEdge(
            "edge:disp-breaks-swing", displacement.node_id,
            SceneEdgeKind.BREAKS, broken_swing.node_id, t1,
            (displacement.node_id, broken_swing.node_id),
        )
    )
    graph._last_asof = t1
    context1 = update_global_market_context(
        context0,
        market_observation(asof=t1, price=100.0),
        _delta(
            graph,
            t1,
            nodes=(displacement.node_id, broken_swing.node_id),
            edges=(break_edge.edge_id,),
        ),
        graph,
    )
    # A displacement merely BREAKS an arbitrary swing; that is not an
    # accepted structural consequence and cannot harden the FVG.
    assert any(
        item.obstruction_id == "fvg:short:1"
        for item in context1.obstruction_views[
            Direction.LONG.value
        ].soft_frictions
    )
    assert "fvg:short:1" not in context1.path_blocker_ids

    t2 = t1 + pd.Timedelta(minutes=1)
    accepted_bos = graph.add_node(
        _node(
            "h1-bearish-bos", "bos", Timeframe.H1, t2,
            direction=Direction.SHORT,
            structural_scale=StructuralScale.EXTERNAL,
            bounds=(102.25, 102.25),
            source_ids=(
                "h4-protected-low",
                "disp:short:1",
            ),
            semantic_attributes=(
                ("scope", "opposed"),
                ("post_break_state", "accepted"),
                ("mss_qualified", "true"),
            ),
        )
    )
    graph._last_asof = t2
    context2 = update_global_market_context(
        context1,
        market_observation(asof=t2, price=100.0),
        _delta(graph, t2, nodes=(accepted_bos.node_id,)),
        graph,
    )
    long_barriers = context2.obstruction_views[
        Direction.LONG.value
    ].hard_barriers
    assert len(long_barriers) == 1
    assert "fvg:short:1" in long_barriers[0].source_ids
    assert "h1-bearish-bos" in long_barriers[0].source_ids
    assert long_barriers[0].lower_bound == pytest.approx(102.0)
    assert long_barriers[0].upper_bound == pytest.approx(103.0)

    t3 = t2 + pd.Timedelta(minutes=1)
    exhausted = graph.add_node(
        replace(
            displacement,
            observed_at=t3,
            lifecycle="exhausted",
            ambiguity_state=EvidenceStatus.INVALIDATED,
            resolution_reason="lost_directional_progress",
            revision_id="",
        )
    )
    # BREAKS is a persistent causal relation, so source episode exhaustion
    # cannot erase the FVG's already-observed structural consequence.
    assert graph._edges[break_edge.edge_id].lifecycle == "active"
    graph._last_asof = t3
    context3 = update_global_market_context(
        context2,
        market_observation(asof=t3, price=100.0),
        SceneGraphDelta(
            asof=t3,
            revision_id=graph.revision_id,
            revised_node_ids=(exhausted.node_id,),
        ),
        graph,
    )
    assert any(
        "fvg:short:1" in item.source_ids
        for item in context3.obstruction_views[
            Direction.LONG.value
        ].hard_barriers
    )


def test_unrelated_accepted_bos_is_not_a_global_hard_barrier() -> None:
    asof = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    authority = graph.add_node(
        _node(
            "h4-up", "structure", Timeframe.H4, asof,
            direction=Direction.LONG,
            source_ids=("h4-protected-low",),
        )
    )
    unrelated = graph.add_node(
        _node(
            "unrelated-h1-bos", "bos", Timeframe.H1, asof,
            direction=Direction.SHORT,
            structural_scale=StructuralScale.EXTERNAL,
            bounds=(102.0, 102.0),
            source_ids=("other-structure",),
            semantic_attributes=(
                ("scope", "continuation"),
                ("post_break_state", "accepted"),
            ),
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        market_observation(asof=asof, price=100.0),
        _delta(graph, asof, nodes=(authority.node_id, unrelated.node_id)),
        graph,
    )

    assert "unrelated-h1-bos" not in context.path_blocker_ids
    assert all(
        "unrelated-h1-bos" not in item.source_ids
        for item in context.obstruction_views[
            Direction.LONG.value
        ].hard_barriers
    )


@pytest.mark.parametrize(
    "attributes",
    (
        (
            ("scope", "local"),
            ("post_break_state", "accepted"),
        ),
        (
            ("scope", "opposed"),
            ("post_break_state", "accepted"),
            ("mss_qualified", "false"),
        ),
    ),
)
def test_local_or_unqualified_opposed_bos_cannot_become_hard(
    attributes: tuple[tuple[str, str], ...],
) -> None:
    asof = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    authority = graph.add_node(
        _node(
            "h4-up", "structure", Timeframe.H4, asof,
            direction=Direction.LONG,
            source_ids=("h4-protected-low",),
        )
    )
    bos = graph.add_node(
        _node(
            "connected-but-unqualified-bos", "bos", Timeframe.H1,
            asof, direction=Direction.SHORT,
            structural_scale=StructuralScale.EXTERNAL,
            bounds=(102.0, 102.0),
            source_ids=("h4-protected-low",),
            semantic_attributes=attributes,
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        market_observation(asof=asof, price=100.0),
        _delta(graph, asof, nodes=(authority.node_id, bos.node_id)),
        graph,
    )

    assert "connected-but-unqualified-bos" not in context.path_blocker_ids


def test_obstruction_clusters_merge_transitive_nested_price_zones() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof, price=100.0)
    graph = TemporalMarketSceneGraph()
    zones = tuple(
        graph.add_node(
            _node(
                identity,
                "support_resistance",
                Timeframe.M5,
                asof,
                direction=Direction.SHORT,
                lifecycle="tested",
                bounds=bounds,
            )
        )
        for identity, bounds in (
            ("sr:a", (102.0, 103.0)),
            ("sr:b", (102.75, 104.0)),
            ("sr:c", (103.75, 105.0)),
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        observation,
        _delta(
            graph,
            asof,
            nodes=tuple(node.node_id for node in zones),
        ),
        graph,
    )

    frictions = context.obstruction_views[
        Direction.LONG.value
    ].soft_frictions
    assert len(frictions) == 1
    assert frictions[0].lower_bound == pytest.approx(102.0)
    assert frictions[0].upper_bound == pytest.approx(105.0)
    assert {"sr:a", "sr:b", "sr:c"}.issubset(frictions[0].source_ids)


def test_authority_stack_keeps_h4_h1_challenger_and_m15_confirmation() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = _with_m15(market_observation(asof=asof))
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node("h4-incumbent", "structure", Timeframe.H4, asof,
              direction=Direction.LONG, source_ids=("h4-protected",))
    )
    h1 = graph.add_node(
        _node(
            "h1-accepted-challenger", "bos", Timeframe.H1, asof,
            direction=Direction.SHORT,
            structural_scale=StructuralScale.EXTERNAL,
            semantic_attributes=(("scope", "opposed"),
                                 ("post_break_state", "accepted")),
        )
    )
    m15 = graph.add_node(
        _node("m15-challenger-confirmation", "structure", Timeframe.M15,
              asof, direction=Direction.SHORT,
              structural_scale=StructuralScale.INTERMEDIATE)
    )
    m5 = graph.add_node(
        _node("m5-setup-only", "structure", Timeframe.M5, asof,
              direction=Direction.SHORT)
    )
    h1_edge = graph.add_edge(
        SceneEdge("edge:h1-challenges-h4", h1.node_id,
                  SceneEdgeKind.OPPOSES, h4.node_id, asof,
                  (h1.node_id, h4.node_id))
    )
    m15_edge = graph.add_edge(
        SceneEdge("edge:m15-confirms-h1", m15.node_id,
                  SceneEdgeKind.ALIGNS_WITH, h1.node_id, asof,
                  (m15.node_id, h1.node_id))
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None, observation,
        _delta(graph, asof,
               nodes=(h4.node_id, h1.node_id, m15.node_id, m5.node_id),
               edges=(h1_edge.edge_id, m15_edge.edge_id)),
        graph,
    )

    layers = {layer.timeframe: layer for layer in context.authority_stack}
    assert set(layers) == {Timeframe.H4, Timeframe.H1, Timeframe.M15}
    assert layers[Timeframe.H4].status == "intact"
    assert layers[Timeframe.H1].status == "challenging"
    assert layers[Timeframe.H1].acceptance_state == "accepted"
    assert layers[Timeframe.M15].status == "challenging"
    assert context.authority_timeframe is Timeframe.H4
    assert context.authority_direction is Direction.LONG
    assert context.market_mode is MarketMode.TRANSITION
    assert context.scale_relations[Timeframe.H1.value] is ScaleRelation.MATERIAL_OPPOSITION
    assert context.scale_relations[Timeframe.M15.value] is ScaleRelation.MATERIAL_OPPOSITION


def test_m15_accepted_bridge_supersedes_stale_m15_structure_owner() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = _with_m15(market_observation(asof=asof))
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node("h4-incumbent", "structure", Timeframe.H4, asof,
              direction=Direction.LONG)
    )
    h1 = graph.add_node(
        _node(
            "h1-short-challenger", "bos", Timeframe.H1, asof,
            direction=Direction.SHORT,
            structural_scale=StructuralScale.EXTERNAL,
            semantic_attributes=(("post_break_state", "accepted"),),
        )
    )
    stale_m15 = graph.add_node(
        _node(
            "m15-old-long", "structure", Timeframe.M15,
            asof - pd.Timedelta(hours=1), direction=Direction.LONG,
            structural_scale=StructuralScale.INTERMEDIATE,
        )
    )
    m15_bridge = graph.add_node(
        replace(
            _node(
                "m15-short-bridge", "bos", Timeframe.M15,
                asof - pd.Timedelta(minutes=15),
                direction=Direction.SHORT,
                structural_scale=StructuralScale.INTERMEDIATE,
                semantic_attributes=(("post_break_state", "accepted"),),
            ),
            observed_at=asof,
        )
    )
    h1_edge = graph.add_edge(
        SceneEdge(
            "edge:h1-short-challenges-h4", h1.node_id,
            SceneEdgeKind.OPPOSES, h4.node_id, asof,
            (h1.node_id, h4.node_id),
        )
    )
    m15_edge = graph.add_edge(
        SceneEdge(
            "edge:m15-short-confirms-h1", m15_bridge.node_id,
            SceneEdgeKind.ALIGNS_WITH, h1.node_id, asof,
            (m15_bridge.node_id, h1.node_id),
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        observation,
        _delta(
            graph,
            asof,
            nodes=(h4.node_id, h1.node_id, stale_m15.node_id,
                   m15_bridge.node_id),
            edges=(h1_edge.edge_id, m15_edge.edge_id),
        ),
        graph,
    )

    layer = next(
        item for item in context.authority_stack
        if item.timeframe is Timeframe.M15
    )
    assert layer.direction is Direction.SHORT
    assert layer.structure_id == "m15-short-bridge"
    assert layer.status == "challenging"
    assert context.scale_relation_details[Timeframe.M15.value].evidence_ids == (
        "m15-short-bridge",
    )


@pytest.mark.parametrize("timeframe", (Timeframe.M15, Timeframe.M5, Timeframe.M1))
def test_sub_h1_accepted_bos_cannot_independently_challenge_h4(
    timeframe: Timeframe,
) -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)
    if timeframe is Timeframe.M15:
        observation = _with_m15(observation)
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node("h4-incumbent", "structure", Timeframe.H4, asof,
              direction=Direction.LONG)
    )
    bos = graph.add_node(
        _node(
            f"{timeframe.value}-accepted-short", "bos", timeframe, asof,
            direction=Direction.SHORT,
            structural_scale=(
                StructuralScale.INTERMEDIATE
                if timeframe is Timeframe.M15
                else StructuralScale.INTERNAL
            ),
            semantic_attributes=(("post_break_state", "accepted"),),
        )
    )
    edge = graph.add_edge(
        SceneEdge(
            f"edge:{timeframe.value}-opposes-h4", bos.node_id,
            SceneEdgeKind.OPPOSES, h4.node_id, asof,
            (bos.node_id, h4.node_id),
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        observation,
        _delta(
            graph,
            asof,
            nodes=(h4.node_id, bos.node_id),
            edges=(edge.edge_id,),
        ),
        graph,
    )

    conflict = next(
        item for item in context.material_conflicts
        if item.conflict_id == edge.edge_id
    )
    assert conflict.role is GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
    assert context.market_mode is MarketMode.DIRECTIONAL
    assert context.authority_timeframe is Timeframe.H4
    assert context.scale_relations[timeframe.value] is ScaleRelation.NORMAL_PULLBACK


def test_accepted_bos_relation_clock_starts_at_acceptance_observation() -> None:
    asof = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node("h4-incumbent", "structure", Timeframe.H4, asof,
              direction=Direction.LONG)
    )
    confirmed_earlier = asof - pd.Timedelta(hours=1)
    h1 = graph.add_node(
        replace(
            _node(
                "h1-accepted-now", "bos", Timeframe.H1,
                confirmed_earlier, direction=Direction.SHORT,
                structural_scale=StructuralScale.EXTERNAL,
                semantic_attributes=(("post_break_state", "accepted"),),
            ),
            observed_at=asof,
        )
    )
    edge = graph.add_edge(
        SceneEdge(
            "edge:h1-accepted-now", h1.node_id,
            SceneEdgeKind.OPPOSES, h4.node_id, asof,
            (h1.node_id, h4.node_id),
        )
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        market_observation(asof=asof),
        _delta(
            graph,
            asof,
            nodes=(h4.node_id, h1.node_id),
            edges=(edge.edge_id,),
        ),
        graph,
    )

    relation = context.scale_relation_details[Timeframe.H1.value]
    layer = next(
        item for item in context.authority_stack
        if item.timeframe is Timeframe.H1
    )
    assert relation.since == asof
    assert relation.age_bars == 0
    assert layer.confirmed_at == asof


def test_m1_dislocation_is_retained_without_changing_global_authority() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node("h4-incumbent", "structure", Timeframe.H4, asof,
              direction=Direction.LONG)
    )
    m1 = graph.add_node(
        _node("m1-local-displacement", "displacement", Timeframe.M1,
              asof, direction=Direction.SHORT, lifecycle="active",
              structural_scale=StructuralScale.INTERNAL)
    )
    edge = graph.add_edge(
        SceneEdge("edge:m1-local-opposition", m1.node_id,
                  SceneEdgeKind.OPPOSES, h4.node_id, asof,
                  (m1.node_id, h4.node_id))
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None, observation,
        _delta(graph, asof, nodes=(h4.node_id, m1.node_id),
               edges=(edge.edge_id,)), graph,
    )

    assert context.authority_timeframe is Timeframe.H4
    assert context.market_mode is MarketMode.DIRECTIONAL
    assert context.dislocations_by_scale[Timeframe.M1.value] == (
        "m1-local-displacement",
    )
    assert context.dislocated
    assert context.dominant_dislocation_scale is Timeframe.M1
    assert all(layer.timeframe is not Timeframe.M1 for layer in context.authority_stack)
    assert context.scale_relations[Timeframe.M1.value] is (
        context.scale_relation_details[Timeframe.M1.value].relation
    )


def test_same_clock_h1_dual_direction_is_explicitly_ambiguous() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)
    graph = TemporalMarketSceneGraph()
    h4 = graph.add_node(
        _node("h4-incumbent", "structure", Timeframe.H4, asof,
              direction=Direction.LONG)
    )
    h1_long = graph.add_node(
        _node("h1-long-bos", "bos", Timeframe.H1, asof,
              direction=Direction.LONG,
              semantic_attributes=(("post_break_state", "accepted"),))
    )
    h1_short = graph.add_node(
        _node("h1-short-bos", "bos", Timeframe.H1, asof,
              direction=Direction.SHORT,
              semantic_attributes=(("post_break_state", "accepted"),))
    )
    edge = graph.add_edge(
        SceneEdge("edge:h1-short-opposes-h4", h1_short.node_id,
                  SceneEdgeKind.OPPOSES, h4.node_id, asof,
                  (h1_short.node_id, h4.node_id))
    )
    graph._last_asof = asof
    context = update_global_market_context(
        None, observation,
        _delta(graph, asof,
               nodes=(h4.node_id, h1_long.node_id, h1_short.node_id),
               edges=(edge.edge_id,)), graph,
    )

    relation = context.scale_relation_details[Timeframe.H1.value]
    assert relation.relation is ScaleRelation.UNKNOWN
    assert relation.ambiguous
    assert set(relation.evidence_ids) == {"h1-long-bos", "h1-short-bos"}
    assert context.market_mode is MarketMode.UNCERTAIN


def test_descriptive_balance_sets_mode_only_without_directional_authority() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = market_observation(asof=asof)

    def build(*, with_authority: bool):
        graph = TemporalMarketSceneGraph()
        forming = graph.add_node(
            _node(
                "h1-descriptive-balance", "range", Timeframe.H1, asof,
                direction=None, lifecycle="active",
                semantic_attributes=(("lower_source_zone_id", "sr:lower"),
                                     ("upper_source_zone_id", "sr:upper")),
                descriptive_metrics=(("lower_touch_count", 2.0),
                                     ("upper_touch_count", 2.0),
                                     ("midpoint_crossings", 2.0),
                                     ("inside_close_fraction", 0.8)),
            )
        )
        nodes = [forming]
        if with_authority:
            nodes.append(graph.add_node(
                _node("h4-directional", "structure", Timeframe.H4, asof,
                      direction=Direction.LONG)
            ))
        graph._last_asof = asof
        return update_global_market_context(
            None, observation,
            _delta(graph, asof, nodes=tuple(node.node_id for node in nodes)),
            graph,
        )

    balanced = build(with_authority=False)
    assert balanced.balance_context is not None
    assert balanced.balance_context.status == "descriptive"
    assert not balanced.balance_context.value_authoritative
    assert balanced.market_mode is MarketMode.BALANCED

    directional = build(with_authority=True)
    assert directional.balance_context is not None
    assert directional.balance_context.status == "descriptive"
    assert directional.market_mode is MarketMode.DIRECTIONAL


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


def test_global_context_rejects_future_authority_and_relation_clocks() -> None:
    asof = _clock("2025-01-06 10:00")
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, asof)
    graph._last_asof = asof
    context = update_global_market_context(
        None,
        market_observation(asof=asof),
        _delta(graph, asof, nodes=tuple(node.node_id for node in nodes)),
        graph,
    )
    future = asof + pd.Timedelta(minutes=1)

    with pytest.raises(ValueError, match="global market context is invalid"):
        replace(
            context,
            authority_stack=(
                replace(context.authority_stack[0], confirmed_at=future),
                *context.authority_stack[1:],
            ),
        )

    h4_relation = context.scale_relation_details[Timeframe.H4.value]
    with pytest.raises(ValueError, match="global market context is invalid"):
        replace(
            context,
            scale_relation_details={
                **context.scale_relation_details,
                Timeframe.H4.value: replace(h4_relation, since=future),
            },
        )
