from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.ai_review import (
    AIReviewAdapter,
    ReviewIssue,
    ai_review_identity,
    compute_primitive,
)
from smc_trader.decision_trace import (
    TRACE_SCHEMA_VERSION,
    build_decision_trace,
    build_frozen_decision_packet,
    decision_packet_bytes,
    decision_packet_payload_sha256,
    sealed_path_audit_context,
    validate_causal_histories,
)
from smc_trader.model import (
    Direction,
    EntryLocationLifecycle,
    EntryLocationState,
    HypothesisSequenceState,
    Bar,
    DrawSelection,
    LiquidityRoute,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
    Playbook,
    PlaybookPhase,
    SequenceStepState,
    Timeframe,
    content_hash,
    to_primitive,
)
from smc_trader.path_evidence import (
    PrimitivePathEvidenceQuery,
    PrimitivePathEvidenceRecorder,
    read_verified_primitive_path_evidence,
)
from smc_trader.scene_graph import (
    EvidenceStatus,
    FocusState,
    HypothesisState,
)
from smc_trader.validation import (
    PathTestResult,
    PrimitivePathCaseResult,
    finalize_primitive_evaluation,
)
from smc_trader.visualization import (
    DecisionVisualizer,
    SealedVisualAudit,
    _audit_hypothesis,
    _belief_geometry_overlay,
    _closed_trade_levels,
    _collision_safe_annotate,
    _event_timeline,
    _liquidity_route_text,
    _partial_geometry_text,
    _plan_display_context,
    _plan_overlay,
    _sequence_text,
    _short_identity,
    _temporal_market_reading_text,
)

from .helpers import candle, engine_snapshot


def _decision_packet(snapshot):
    source_candle = replace(
        candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
        close=snapshot.observation.price,
    )
    histories = {
        Timeframe.H4: (
            candle(Timeframe.H4, "2025-01-06 06:00", 99.0),
        ),
        Timeframe.H1: (
            candle(Timeframe.H1, "2025-01-06 09:00", 99.5),
        ),
        Timeframe.M5: (
            candle(Timeframe.M5, "2025-01-06 09:55", 100.0),
        ),
        Timeframe.M1: (
            source_candle,
        ),
    }
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    hypothesis = snapshot.belief.hypotheses[key]
    assert hypothesis.sequence is not None
    assert hypothesis.sequence.setup_id is not None
    source_bar = Bar(
        start=source_candle.start,
        open=source_candle.open,
        high=source_candle.high,
        low=source_candle.low,
        close=source_candle.close,
        volume=source_candle.volume,
        symbol=source_candle.symbol,
        instrument_id=source_candle.instrument_id,
    )
    return build_frozen_decision_packet(
        snapshot,
        histories,
        hypothesis_key=key,
        audit_context=sealed_path_audit_context(
            hypothesis.sequence.setup_id
        ),
        source_bar=source_bar,
    )


def _scene_reading_snapshot():
    snapshot = engine_snapshot()
    asof = snapshot.observation.asof
    m15_frame = replace(
        snapshot.observation.frame(Timeframe.H1),
        timeframe=Timeframe.M15,
        liquidity=(),
    )
    active = (
        Timeframe.H4,
        Timeframe.H1,
        Timeframe.M15,
        Timeframe.M5,
        Timeframe.M1,
    )
    observation = replace(
        snapshot.observation,
        frames={
            Timeframe.H4: snapshot.observation.frame(Timeframe.H4),
            Timeframe.H1: snapshot.observation.frame(Timeframe.H1),
            Timeframe.M15: m15_frame,
            Timeframe.M5: snapshot.observation.frame(Timeframe.M5),
            Timeframe.M1: snapshot.observation.frame(Timeframe.M1),
        },
        active_timeframes=active,
        scale_registry_id="scale:test-five",
        scene_revision_id="scene:revision-2",
        scene_added_node_ids=("node:m15-bos",),
        scene_revised_node_ids=("node:h1-structure",),
        scene_added_edge_ids=("edge:align",),
        scene_resolution_event_ids=("resolution:ambiguity",),
    )
    focus = FocusState(
        asof=asof,
        primary_timeframes=(Timeframe.M15.value, Timeframe.M5.value),
        supplemental_timeframes=(Timeframe.H1.value,),
        reason_codes=("new_key_event", "cross_scale_conflict"),
        trigger_event_ids=("event:m15-bos",),
        question="did the bridge BOS align with the setup displacement?",
        resolution_status=EvidenceStatus.AMBIGUOUS,
        switched=True,
        switched_at=asof,
        prior_timeframes=(Timeframe.H4.value, Timeframe.H1.value),
        hypothesis_id="hyp:dfp-long",
        phase_at_selection=PlaybookPhase.ARMED.value,
        supplemental_query_used=True,
    )
    dominant = HypothesisState(
        hypothesis_id="hyp:dfp-long",
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        context_root_ids=("node:h4-structure",),
        context_timeframe=Timeframe.H4.value,
        setup_timeframe=Timeframe.M5.value,
        trigger_timeframe=Timeframe.M1.value,
        sequence_stage=PlaybookPhase.ARMED.value,
        next_expected_event="first_pullback_to_frozen_zone",
        supporting_graph_paths=(
            (
                "node:h4-structure",
                "ALIGNS_WITH",
                "node:m15-bos",
                "CREATES",
                "node:m5-fvg",
            ),
        ),
        contradicting_graph_paths=(),
        missing_evidence={"micro_bos": EvidenceStatus.UNKNOWN},
        ambiguous_evidence={
            "reacceptance": EvidenceStatus.AMBIGUOUS,
        },
        context_draw_id="node:h4-draw",
        primary_target_id="node:m15-liquidity",
        invalidation_id="node:m5-protected-low",
        evidence_revision_id="evidence:2",
    )
    competitor = HypothesisState(
        hypothesis_id="hyp:lsr-short",
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        direction=Direction.SHORT,
        context_root_ids=("node:m15-pool",),
        context_timeframe=Timeframe.M15.value,
        setup_timeframe=Timeframe.M5.value,
        trigger_timeframe=Timeframe.M1.value,
        sequence_stage=PlaybookPhase.FORMING.value,
        next_expected_event="opposite_displacement",
        supporting_graph_paths=(),
        contradicting_graph_paths=(
            (
                "node:m15-pool",
                "OPPOSES",
                "node:h1-structure",
            ),
        ),
        missing_evidence={
            "opposite_displacement": EvidenceStatus.NOT_OBSERVED,
        },
        ambiguous_evidence={},
        context_draw_id="node:m15-pool",
        primary_target_id="node:m5-liquidity",
        invalidation_id="node:sweep-extreme",
        evidence_revision_id="evidence:1",
    )
    belief = replace(
        snapshot.belief,
        asof=asof,
        context_hypotheses={
            dominant.hypothesis_id: dominant,
            competitor.hypothesis_id: competitor,
        },
        dominant_hypothesis_id=dominant.hypothesis_id,
        competing_hypothesis_ids=(competitor.hypothesis_id,),
        focus_state=focus,
        cross_scale_conflicts=("conflict:m15-vs-h1",),
        unresolved_ambiguities=("ambiguity:reacceptance",),
        scene_revision_id="scene:revision-2",
    )
    return replace(snapshot, observation=observation, belief=belief)


def _scene_reading_histories():
    m15 = replace(
        candle(Timeframe.M5, "2025-01-06 09:45", 99.8),
        timeframe=Timeframe.M15,
        end=pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
        observed_minutes=15,
        expected_minutes=15,
        real_minutes=15,
    )
    return {
        Timeframe.H4: (
            candle(Timeframe.H4, "2025-01-06 06:00", 99.0),
        ),
        Timeframe.H1: (
            candle(Timeframe.H1, "2025-01-06 09:00", 99.5),
        ),
        Timeframe.M15: (m15,),
        Timeframe.M5: (
            candle(Timeframe.M5, "2025-01-06 09:55", 100.0),
        ),
        Timeframe.M1: (
            candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
        ),
    }


def _write_packet(packet, path):
    path.write_bytes(
        decision_packet_bytes(to_primitive(packet))
    )
    return path


def _registered_path_query(
    packet,
    hypothesis_key: str,
    issue: ReviewIssue,
    query_id: str,
    *,
    relevant_entity_ids=(),
):
    value = compute_primitive(
        issue,
        packet,
        hypothesis_key=hypothesis_key,
    )
    return PrimitivePathEvidenceQuery(
        query_id=query_id,
        issue=issue.value,
        primitive_name=value.primitive_name,
        formula_version=value.formula_version,
        definition_hash=value.definition_hash,
        relevant_entity_ids=tuple(relevant_entity_ids),
    )


def _snapshot_with_sequence():
    snapshot = engine_snapshot()
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    hypothesis = snapshot.belief.hypotheses[key]
    setup_id = "visual-setup"
    location_id = "visual-location"
    formed_at = snapshot.observation.asof - pd.Timedelta(minutes=3)
    departed_at = snapshot.observation.asof - pd.Timedelta(minutes=2)
    entered_at = snapshot.observation.asof - pd.Timedelta(minutes=1)
    path_steps = (
        PathSequenceStep(
            step_id="visual-zone-visible",
            kind="zone_visible",
            observed_at=formed_at,
            source_event_id=None,
            source_entity_id="visual-fvg",
            predecessor_step_ids=(),
            same_clock_relation="origin",
            direction=Direction.LONG,
            strength=0.8,
            reason="frozen test zone became visible",
        ),
        PathSequenceStep(
            step_id="visual-departure",
            kind="departure_confirmed",
            observed_at=departed_at,
            source_event_id=None,
            source_entity_id=location_id,
            predecessor_step_ids=("visual-zone-visible",),
            same_clock_relation="strictly_after",
            direction=Direction.LONG,
            strength=0.7,
            reason="price departed before the first return",
        ),
        PathSequenceStep(
            step_id="visual-first-pullback",
            kind="first_pullback",
            observed_at=entered_at,
            source_event_id=None,
            source_entity_id=location_id,
            predecessor_step_ids=("visual-departure",),
            same_clock_relation="strictly_after",
            direction=Direction.LONG,
            strength=0.6,
            reason="first completed-bar return to the frozen zone",
        ),
    )
    path = PathSequenceState(
        sequence_id=setup_id,
        protocol_hash="e" * 64,
        symbol=snapshot.observation.symbol,
        instrument_id=snapshot.observation.instrument_id,
        context_kind="zone_return",
        context_id=location_id,
        direction=Direction.LONG,
        lifecycle=PathSequenceLifecycle.ACTIVE,
        formed_at=formed_at,
        state_started_at=formed_at,
        last_updated_at=snapshot.observation.asof,
        age_real_1m_bars=3,
        state_duration_real_1m_bars=3,
        steps=path_steps,
    )
    location = EntryLocationState(
        location_id=location_id,
        protocol_hash="e" * 64,
        source_group3_protocol_hash="f" * 64,
        symbol=snapshot.observation.symbol,
        instrument_id=snapshot.observation.instrument_id,
        direction=Direction.LONG,
        source_zone_kind="fvg",
        source_zone_id="visual-fvg",
        source_zone_protocol_hash="f" * 64,
        source_displacement_id="visual-displacement",
        source_bos_id=None,
        lower_bound=99.0,
        upper_bound=101.0,
        midpoint=100.0,
        near_edge=101.0,
        far_edge=99.0,
        failure_boundary=99.0,
        formed_at=formed_at,
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        state_started_at=entered_at,
        last_updated_at=snapshot.observation.asof,
        age_real_1m_bars=3,
        state_duration_real_1m_bars=1,
        current_price=snapshot.observation.price,
        distance_to_zone_points=0.0,
        distance_to_failure_points=1.0,
        departure_confirmed_at=departed_at,
        first_entered_at=entered_at,
        entry_mode="crossed_near_edge",
        contact_reference_price=101.0,
        first_penetration_fraction=0.5,
    )
    sequence = HypothesisSequenceState(
        protocol_version="visual-test",
        protocol_hash="d" * 64,
        setup_id=setup_id,
        steps=(
            SequenceStepState(
                "complete",
                True,
                1.0,
                entered_at,
            ),
        ),
        started_at=formed_at,
    )
    assert hypothesis.plan is not None
    plan = replace(
        hypothesis.plan,
        setup_id=setup_id,
        entry_location_id=location_id,
        entry_path_id=setup_id,
        entry_zone_lower=99.0,
        entry_zone_upper=101.0,
        selected_draw_id=hypothesis.plan.targets[0].level_id,
    )
    hypothesis = replace(
        hypothesis,
        sequence=sequence,
        plan=plan,
        thesis_strength=0.8,
        sequence_progress=1.0,
        location_quality=0.8,
        entry_readiness=0.8,
        delivery_quality=0.8,
        evidence_group_scores={
            "structure": 0.8,
            "displacement": 0.8,
            "location": 0.8,
            "liquidity": 0.8,
            "trigger": 0.8,
            "execution": 0.8,
        },
        hard_gate_results={"typed_fixture_complete": True},
        setup_context_id=setup_id,
        entry_location_id=location_id,
    )
    return replace(
        snapshot,
        observation=replace(
            snapshot.observation,
            group5_authoritative=True,
            entry_locations=(location,),
            path_sequences=(path,),
        ),
        belief=replace(
            snapshot.belief,
            hypotheses={key: hypothesis},
        ),
        decision=replace(snapshot.decision, plan=plan),
    )


def _path_result(
    snapshot,
    *,
    outcome: str,
    resolved_at: pd.Timestamp,
):
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    belief = snapshot.belief.hypotheses[key]
    assert belief.sequence is not None
    plan = belief.plan
    assert plan is not None
    completed_at = max(
        step.observed_at
        for step in belief.sequence.steps
        if step.satisfied and step.observed_at is not None
    )
    return PathTestResult(
        setup_id=belief.sequence.setup_id,
        hypothesis_key=belief.key,
        entry_location_id=plan.entry_location_id,
        entry_path_id=plan.entry_path_id,
        playbook=belief.playbook.value,
        direction=belief.direction.value,
        setup_started_at=belief.sequence.started_at,
        sequence_completed_at=completed_at,
        decision_time=snapshot.observation.asof,
        resolved_at=resolved_at,
        outcome=outcome,
        success=outcome == "target",
        decision_hash=snapshot.snapshot_hash,
        entry=plan.planned_entry,
        invalidation=plan.invalidation.price,
        invalidation_source_id=plan.invalidation.source_level_id,
        target=plan.targets[0].price,
        target_source_id=plan.targets[0].level_id,
        deadline=plan.deadline,
        probability=belief.probability,
        raw_probability=(
            belief.probability
            if belief.raw_probability is None
            else belief.raw_probability
        ),
        uncertainty=belief.uncertainty,
        phase=belief.phase.value,
        calibration_version=belief.calibration_version,
        calibration_hash=belief.calibration_hash,
        mfe_R=0.0,
        mae_R=0.0,
        formation_minutes=1,
        entry_touched=False,
        entry_touched_at=None,
        time_to_entry_minutes=None,
        elapsed_minutes=max(
            0,
            int(
                (
                    resolved_at - snapshot.observation.asof
                ).total_seconds()
                // 60
            ),
        ),
        ambiguous_same_bar=False,
        protocol_version=belief.sequence.protocol_version,
        protocol_hash=belief.sequence.protocol_hash,
        config_hash="b" * 64,
        code_hash="c" * 64,
    )


def test_decision_and_future_reveal_are_separate_artifacts(tmp_path) -> None:
    snapshot = engine_snapshot()
    asof = snapshot.observation.asof
    histories = {
        Timeframe.H4: (candle(Timeframe.H4, "2025-01-06 06:00", 99.0),),
        Timeframe.H1: (candle(Timeframe.H1, "2025-01-06 09:00", 99.5),),
        Timeframe.M5: (candle(Timeframe.M5, "2025-01-06 09:55", 100.0),),
        Timeframe.M1: (candle(Timeframe.M1, "2025-01-06 09:59", 100.0),),
    }
    visualizer = DecisionVisualizer()
    decision = visualizer.render_decision(
        snapshot, histories, tmp_path / "decision.png"
    )
    assert decision.kind == "decision"
    assert decision.maximum_market_time <= asof
    future = (
        candle(Timeframe.M1, "2025-01-06 10:00", 100.5),
        candle(Timeframe.M1, "2025-01-06 10:01", 101.0),
    )
    reveal = visualizer.render_reveal(
        snapshot,
        visualizer.seal_reveal(snapshot),
        future,
        tmp_path / "future.png",
        revealed_at=pd.Timestamp("2025-01-06 10:02", tz="America/New_York"),
    )
    assert reveal.kind == "future_reveal"
    assert reveal.path != decision.path
    assert reveal.maximum_market_time > decision.maximum_market_time


def test_five_scale_trace_and_visual_freeze_temporal_market_reading(
    tmp_path,
) -> None:
    snapshot = _scene_reading_snapshot()
    histories = _scene_reading_histories()

    validated = validate_causal_histories(snapshot, histories)
    assert tuple(validated) == (
        Timeframe.H4,
        Timeframe.H1,
        Timeframe.M15,
        Timeframe.M5,
        Timeframe.M1,
    )
    trace = build_decision_trace(snapshot)
    assert trace["schema_version"] == TRACE_SCHEMA_VERSION == 1
    reading = trace["market_reading_t"]
    assert reading["active_timeframes"] == ["4H", "1H", "15m", "5m", "1m"]
    assert reading["focus_state"]["resolution_status"] == "ambiguous"
    assert reading["dominant_hypothesis_id"] == "hyp:dfp-long"
    assert reading["competing_hypothesis_ids"] == ["hyp:lsr-short"]
    assert (
        reading["context_hypotheses"]["hyp:dfp-long"]
        ["missing_evidence"]["micro_bos"]
        == "unknown"
    )
    assert reading["scene_delta"]["added_node_ids"] == ["node:m15-bos"]
    assert trace["future_path_included"] is False

    packet = build_frozen_decision_packet(snapshot, histories)
    assert list(packet["histories"]) == ["4H", "1H", "15m", "5m", "1m"]
    assert packet["market_reading_t"] == reading
    assert packet["future_path"]["included"] is False

    text = _temporal_market_reading_text(snapshot)
    assert "primary 15m, 5m" in text
    assert "ACTIVE COMPETING HYPOTHESES" in text
    assert "ALIGNS_WITH" in text
    assert "micro_bos=unknown" in text
    assert "UNKNOWN is unresolved evidence, never FALSE" in text

    artifact = DecisionVisualizer().render_decision(
        snapshot,
        histories,
        tmp_path / "five-scale-decision.png",
    )
    assert artifact.maximum_market_time == snapshot.observation.asof
    assert artifact.path.is_file()


def test_five_scale_history_validation_fails_closed_when_m15_is_missing() -> None:
    snapshot = _scene_reading_snapshot()
    histories = _scene_reading_histories()
    histories.pop(Timeframe.M15)

    with pytest.raises(ValueError, match="enabled scale registry"):
        validate_causal_histories(snapshot, histories)


def test_trace_and_visual_freeze_liquidity_route_roles() -> None:
    snapshot = engine_snapshot()
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    hypothesis = snapshot.belief.hypotheses[key]
    target = hypothesis.deliverable_targets[0]
    selection = DrawSelection(
        draw_id=target.level_id,
        selected_at=snapshot.observation.asof,
        selection_reason="nearest deliverable before deadline",
        source_timeframe=target.timeframe,
        source_kind="swing",
        side=target.side,
        price=target.price,
        source_confirmed_at=target.confirmed_at,
        strength=0.8,
    )
    route = LiquidityRoute(
        route_id="route:dfp-long",
        selected_at=snapshot.observation.asof,
        context_draw_id="draw:h4-external",
        intermediate_liquidity_ids=("liq:m15-intermediate",),
        primary_deliverable_target_id=target.level_id,
        terminal_draw_id="draw:h4-terminal",
        path_blocker_ids=("blocker:h1-opposing-pool",),
        source_path_ids=("path:h4-to-h1-to-m15",),
    )
    routed_hypothesis = replace(
        hypothesis,
        plan=None,
        draw_selection=selection,
        liquidity_route=route,
    )
    routed_belief = replace(
        snapshot.belief,
        hypotheses={key: routed_hypothesis},
    )
    routed_snapshot = replace(snapshot, belief=routed_belief)

    trace = build_decision_trace(routed_snapshot)
    frozen = trace["belief_t"][key]["liquidity_route"]
    assert frozen["context_draw_id"] == "draw:h4-external"
    assert frozen["primary_deliverable_target_id"] == target.level_id
    assert frozen["terminal_draw_id"] == "draw:h4-terminal"
    assert frozen["path_blocker_ids"] == ["blocker:h1-opposing-pool"]

    text = _partial_geometry_text(routed_snapshot, routed_hypothesis)
    assert "context draw draw:h4-external" in text
    assert f"primary deliverable {target.level_id}" in text
    assert "terminal draw draw:h4-terminal" in text
    assert "path blockers blocker:h1-opposing-pool" in text
    assert _liquidity_route_text(route) in text


def test_decision_view_rejects_stale_or_wrong_causal_history(tmp_path) -> None:
    snapshot = engine_snapshot()
    valid = {
        Timeframe.H4: (
            candle(Timeframe.H4, "2025-01-06 06:00", 99.0),
        ),
        Timeframe.H1: (
            candle(Timeframe.H1, "2025-01-06 09:00", 99.5),
        ),
        Timeframe.M5: (
            candle(Timeframe.M5, "2025-01-06 09:55", 100.0),
        ),
        Timeframe.M1: (
            candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
        ),
    }
    stale = dict(valid)
    stale[Timeframe.H4] = (
        candle(Timeframe.H4, "2025-01-06 05:00", 99.0),
    )
    wrong_contract = dict(valid)
    wrong_contract[Timeframe.M1] = (
        replace(valid[Timeframe.M1][0], instrument_id=2),
    )
    incomplete = dict(valid)
    incomplete[Timeframe.M5] = (
        replace(valid[Timeframe.M5][0], complete=False),
    )

    visualizer = DecisionVisualizer()
    for number, histories in enumerate(
        (stale, wrong_contract, incomplete),
        start=1,
    ):
        with pytest.raises(ValueError, match="causal history"):
            visualizer.render_decision(
                snapshot,
                histories,
                tmp_path / f"invalid-{number}.png",
            )


def test_terminal_plan_is_non_actionable_except_in_explicit_path_audit() -> None:
    snapshot = engine_snapshot()
    belief = next(iter(snapshot.belief.hypotheses.values()))
    terminal = replace(belief, phase=PlaybookPhase.INVALIDATED)
    terminal_market = replace(
        snapshot.belief,
        hypotheses={terminal.key: terminal},
    )
    terminal_snapshot = replace(snapshot, belief=terminal_market)

    heading, show = _plan_display_context(
        terminal_snapshot,
        terminal,
        explicit_audit_key=None,
    )
    assert heading == "HISTORICAL PLAN — NON-ACTIONABLE (invalidated)"
    assert not show

    audit_heading, audit_show = _plan_display_context(
        terminal_snapshot,
        terminal,
        explicit_audit_key=terminal.key,
    )
    assert "DIAGNOSTIC ONLY" in audit_heading
    assert audit_show


def test_offscreen_plan_levels_do_not_compress_candle_scale() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snapshot = engine_snapshot()
    plan = next(iter(snapshot.belief.hypotheses.values())).plan
    assert plan is not None
    target = replace(
        plan.targets[0],
        level_id="swing:ffc8bc6e0332d04ac477c1be",
    )
    plan = replace(
        plan,
        invalidation=replace(
            plan.invalidation,
            source_level_id="pool:803f604473d83f9e",
        ),
        targets=(target, *plan.targets[1:]),
    )
    values = (
        candle(Timeframe.M1, "2025-01-06 09:58", 100.0),
        candle(Timeframe.M1, "2025-01-06 09:59", 100.2),
    )
    figure, axis = plt.subplots()
    _plan_overlay(axis, plan, values)
    low, high = axis.get_ylim()
    x_low, x_high = axis.get_xlim()
    labels = tuple(item.get_text() for item in axis.texts)
    collision_boxes = tuple(axis._smc_annotation_boxes)
    anchors = tuple(item.xy for item in axis.texts)
    plt.close(figure)

    candle_low = min(item.low for item in values)
    candle_high = max(item.high for item in values)
    assert low < candle_low
    assert high > candle_high
    assert high - low < 2 * (candle_high - candle_low)
    assert any(
        _short_identity(plan.invalidation.source_level_id) in label
        for label in labels
    )
    assert any(
        _short_identity(plan.targets[0].level_id) in label
        for label in labels
    )
    assert len(collision_boxes) >= 2
    assert all(x_low <= x <= x_high for x, _ in anchors)


def test_partial_hypothesis_geometry_shows_draw_and_invalidation_without_scale_compression() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snapshot = engine_snapshot()
    hypothesis = next(iter(snapshot.belief.hypotheses.values()))
    partial = replace(hypothesis, plan=None)
    values = (
        candle(Timeframe.M1, "2025-01-06 09:58", 100.0),
        candle(Timeframe.M1, "2025-01-06 09:59", 100.2),
    )
    figure, axis = plt.subplots()
    _belief_geometry_overlay(
        axis,
        partial,
        values,
        direct_labels=True,
    )
    low, high = axis.get_ylim()
    plt.close(figure)

    text = _partial_geometry_text(snapshot, partial)
    assert partial.invalidation.source_level_id in text
    assert partial.deliverable_targets[0].level_id in text
    candle_low = min(item.low for item in values)
    candle_high = max(item.high for item in values)
    assert low < candle_low
    assert high > candle_high
    assert high - low < 2 * (candle_high - candle_low)


def test_annotation_lanes_are_collision_safe_and_event_memory_exposes_identity() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.set_xlim(0, 10)
    axis.set_ylim(99, 101)
    for number in range(6):
        _collision_safe_annotate(
            axis,
            f"dense label {number}",
            9.5,
            100.0 + number * 0.01,
            color="#2563eb",
        )
    boxes = list(axis._smc_annotation_boxes)
    plt.close(figure)
    assert len(boxes) == 6
    assert all(
        not (
            left[0] < right[2]
            and left[2] > right[0]
            and left[1] < right[3]
            and left[3] > right[1]
        )
        for index, left in enumerate(boxes)
        for right in boxes[index + 1 :]
    )
    timeline = _event_timeline(engine_snapshot())
    assert "id=sweep" in timeline
    assert "src=old-belo" in timeline


def test_prefixed_visual_identity_keeps_prefix_and_eight_hash_characters() -> None:
    assert (
        _short_identity("swing:ffc8bc6e0332d04ac477c1be")
        == "swing:ffc8bc6e"
    )
    assert (
        _short_identity("inventory:swing:0123456789abcdef")
        == "inventory:swing:01234567"
    )
    assert _short_identity("803f604473d83f9e") == "803f6044"


def test_dense_annotation_grid_and_overflow_never_accept_overlap() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.set_xlim(0, 10)
    axis.set_ylim(99, 101)
    for number in range(48):
        _collision_safe_annotate(
            axis,
            f"dense event identity {number:02d}",
            9.5,
            100.0,
            color="#2563eb",
        )
    boxes = list(axis._smc_annotation_boxes)
    assert len(boxes) == 48
    assert all(
        not (
            left[0] < right[2]
            and left[2] > right[0]
            and left[1] < right[3]
            and left[3] > right[1]
        )
        for index, left in enumerate(boxes)
        for right in boxes[index + 1 :]
    )

    occupied = plt.subplots()[1]
    occupied.set_xlim(0, 10)
    occupied.set_ylim(99, 101)
    occupied._smc_annotation_boxes = [(0.0, 0.0, 1.0, 1.0)]
    _collision_safe_annotate(
        occupied,
        "verified overflow identity",
        9.5,
        100.0,
        color="#2563eb",
    )
    overflow_box = occupied._smc_annotation_boxes[-1]
    annotation = occupied.texts[-1]
    assert overflow_box[2] < 0.0
    assert not annotation.get_clip_on()
    assert annotation.get_annotation_clip() is False
    plt.close("all")


def test_missing_audit_hypothesis_copy_does_not_claim_a_position(
    tmp_path,
    monkeypatch,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    snapshot = _snapshot_with_sequence()
    histories = {
        Timeframe.H4: (
            candle(Timeframe.H4, "2025-01-06 06:00", 99.0),
        ),
        Timeframe.H1: (
            candle(Timeframe.H1, "2025-01-06 09:00", 99.5),
        ),
        Timeframe.M5: (
            candle(Timeframe.M5, "2025-01-06 09:55", 100.0),
        ),
        Timeframe.M1: (
            candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
        ),
    }
    captured: list[str] = []
    original_savefig = Figure.savefig

    def capture_text(figure, *args, **kwargs):
        captured.extend(
            text.get_text()
            for axis in figure.axes
            for text in axis.texts
        )
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_text)
    DecisionVisualizer().render_decision(
        snapshot,
        histories,
        tmp_path / "unbound.png",
        suppress_audit_hypothesis=True,
        audit_context={
            "scenario": "sampled_decision_batch_anchor",
            "future_path_included": False,
        },
    )
    rendered_text = " ".join("\n".join(captured).split())
    expected = (
        "not shown — no exact audit hypothesis identity "
        "is available at this decision clock"
    )
    assert rendered_text.count(expected) == 2
    assert "managed position has no exact" not in rendered_text
    assert "thesis=0.800" in rendered_text
    assert "effective=0.800" in rendered_text
    assert "readiness=0.800" in rendered_text
    assert "probability: " not in rendered_text


def test_episode_deadline_is_visible_in_setup_and_partial_geometry(
    monkeypatch,
) -> None:
    import smc_trader.visualization as visualization

    snapshot = _snapshot_with_sequence()
    belief = next(iter(snapshot.belief.hypotheses.values()))
    deadline = snapshot.observation.asof + pd.Timedelta(minutes=37)
    proxy_values = dict(vars(belief))
    proxy_values["episode_deadline"] = deadline
    proxy = SimpleNamespace(**proxy_values)
    expected = f"episode deadline {deadline:%Y-%m-%d %H:%M %Z}"
    assert expected in _partial_geometry_text(snapshot, proxy)
    monkeypatch.setattr(
        visualization,
        "_audit_hypothesis",
        lambda *_args, **_kwargs: proxy,
    )
    assert expected in _sequence_text(snapshot, belief.key)


def test_closed_trade_audit_uses_frozen_geometry_not_current_plan(
    tmp_path,
    monkeypatch,
) -> None:
    import smc_trader.visualization as visualization

    snapshot = engine_snapshot()
    current_plan = next(iter(snapshot.belief.hypotheses.values())).plan
    assert current_plan is not None
    trade = {
        "thesis_hash": "closed-thesis",
        "playbook": "liquidity_sweep_reversal",
        "direction": "short",
        "decision_time": snapshot.observation.asof - pd.Timedelta(minutes=10),
        "opened_at": snapshot.observation.asof - pd.Timedelta(minutes=8),
        "closed_at": snapshot.observation.asof,
        "entry_price": 110.0,
        "original_invalidation": 112.0,
        "final_stop": 109.5,
        "target": 105.0,
        "exit_price": 105.0,
        "exit_reason": "target",
        "gross_R": 2.5,
        "cost_R": 0.1,
        "net_R": 2.4,
        "ambiguous_same_bar": False,
    }
    grouped = _closed_trade_levels(trade)
    assert [item[0] for item in grouped] == [
        "entry",
        "original invalidation",
        "final stop",
        "target / exit",
    ]
    assert {item[1] for item in grouped} == {110.0, 112.0, 109.5, 105.0}

    current_plan_calls = []
    closed_trade_calls = []
    monkeypatch.setattr(
        visualization,
        "_plan_overlay",
        lambda *args, **kwargs: current_plan_calls.append((args, kwargs)),
    )
    original_closed_overlay = visualization._closed_trade_overlay

    def capture_closed_overlay(axis, frozen, candles, *, direct_labels):
        closed_trade_calls.append((frozen, direct_labels))
        return original_closed_overlay(
            axis,
            frozen,
            candles,
            direct_labels=direct_labels,
        )

    monkeypatch.setattr(
        visualization,
        "_closed_trade_overlay",
        capture_closed_overlay,
    )
    histories = {
        Timeframe.H4: (
            candle(Timeframe.H4, "2025-01-06 06:00", 99.0),
        ),
        Timeframe.H1: (
            candle(Timeframe.H1, "2025-01-06 09:00", 99.5),
        ),
        Timeframe.M5: (
            candle(Timeframe.M5, "2025-01-06 09:55", 100.0),
        ),
        Timeframe.M1: (
            candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
        ),
    }
    artifact = DecisionVisualizer().render_decision(
        snapshot,
        histories,
        tmp_path / "closed-trade.png",
        audit_context={
            "scenario": "target",
            "closed_trade": trade,
            "future_present": False,
        },
    )
    assert artifact.path.exists()
    assert not current_plan_calls
    assert len(closed_trade_calls) == 4
    assert all(
        item[0]["thesis_hash"] == "closed-thesis"
        for item in closed_trade_calls
    )
    assert [item[1] for item in closed_trade_calls] == [
        False,
        False,
        True,
        True,
    ]


def test_protect_audit_overlays_management_stop_without_replacing_plan(
    tmp_path,
    monkeypatch,
) -> None:
    import smc_trader.visualization as visualization

    snapshot = engine_snapshot()
    plan_calls = []
    protected_calls = []
    original_plan_overlay = visualization._plan_overlay
    original_protected_overlay = visualization._protected_stop_overlay

    def capture_plan(axis, plan, candles, **kwargs):
        plan_calls.append((plan, kwargs))
        return original_plan_overlay(axis, plan, candles, **kwargs)

    def capture_protected(axis, stop, candles, *, direct_labels):
        protected_calls.append((stop, direct_labels))
        return original_protected_overlay(
            axis,
            stop,
            candles,
            direct_labels=direct_labels,
        )

    monkeypatch.setattr(visualization, "_plan_overlay", capture_plan)
    monkeypatch.setattr(
        visualization,
        "_protected_stop_overlay",
        capture_protected,
    )
    histories = {
        Timeframe.H4: (
            candle(Timeframe.H4, "2025-01-06 06:00", 99.0),
        ),
        Timeframe.H1: (
            candle(Timeframe.H1, "2025-01-06 09:00", 99.5),
        ),
        Timeframe.M5: (
            candle(Timeframe.M5, "2025-01-06 09:55", 100.0),
        ),
        Timeframe.M1: (
            candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
        ),
    }
    artifact = DecisionVisualizer().render_decision(
        snapshot,
        histories,
        tmp_path / "protect.png",
        audit_context={
            "scenario": "protect",
            "protected_stop": 100.25,
            "future_present": False,
        },
    )
    assert artifact.path.exists()
    assert len(plan_calls) == 4
    assert not protected_calls
    assert [
        (
            call[1]["protected_stop"],
            call[1]["direct_labels"],
        )
        for call in plan_calls
    ] == [
        (100.25, False),
        (100.25, False),
        (100.25, True),
        (100.25, True),
    ]


def test_terminal_boundary_can_reveal_an_explicit_empty_future(tmp_path) -> None:
    snapshot = _snapshot_with_sequence()
    result = _path_result(
        snapshot,
        outcome="contract_change",
        resolved_at=snapshot.observation.asof,
    )
    visualizer = DecisionVisualizer()
    reveal = visualizer.render_reveal(
        snapshot,
        visualizer.seal_reveal(snapshot),
        (),
        tmp_path / "empty_future.png",
        revealed_at=snapshot.observation.asof,
        path_result=result,
    )
    assert reveal.kind == "future_reveal"
    assert reveal.maximum_market_time == snapshot.observation.asof


def test_empty_future_cannot_hide_a_later_resolution(tmp_path) -> None:
    snapshot = _snapshot_with_sequence()
    later = snapshot.observation.asof + pd.Timedelta(minutes=1)
    result = _path_result(
        snapshot,
        outcome="deadline",
        resolved_at=later,
    )
    visualizer = DecisionVisualizer()
    with pytest.raises(ValueError, match="decision-boundary"):
        visualizer.render_reveal(
            snapshot,
            visualizer.seal_reveal(snapshot),
            (),
            tmp_path / "missing_future.png",
            revealed_at=later,
            path_result=result,
        )


def test_future_reveal_rejects_mismatched_frozen_identity(tmp_path) -> None:
    snapshot = _snapshot_with_sequence()
    resolved_at = snapshot.observation.asof + pd.Timedelta(minutes=1)
    result = _path_result(
        snapshot,
        outcome="target",
        resolved_at=resolved_at,
    )
    mismatched = replace(
        result,
        target_source_id="another-draw",
    )
    future = (
        candle(Timeframe.M1, "2025-01-06 10:00", 101.0),
    )
    visualizer = DecisionVisualizer()
    with pytest.raises(ValueError, match="frozen plan"):
        visualizer.render_reveal(
            snapshot,
            visualizer.seal_reveal(snapshot),
            future,
            tmp_path / "wrong-identity.png",
            revealed_at=resolved_at,
            path_result=mismatched,
        )


def test_future_reveal_cannot_omit_the_resolving_bar(tmp_path) -> None:
    snapshot = _snapshot_with_sequence()
    resolved_at = snapshot.observation.asof + pd.Timedelta(minutes=2)
    result = _path_result(
        snapshot,
        outcome="target",
        resolved_at=resolved_at,
    )
    future = (
        candle(Timeframe.M1, "2025-01-06 10:00", 101.0),
    )
    visualizer = DecisionVisualizer()
    with pytest.raises(ValueError, match="does not reach"):
        visualizer.render_reveal(
            snapshot,
            visualizer.seal_reveal(snapshot),
            future,
            tmp_path / "truncated-future.png",
            revealed_at=resolved_at,
            path_result=result,
        )


def test_visual_hypothesis_selection_is_fail_closed() -> None:
    snapshot = engine_snapshot()
    selected = snapshot.decision.best_hypothesis_key
    assert selected is not None
    assert _audit_hypothesis(snapshot).key == selected

    no_selection = replace(
        snapshot,
        decision=replace(
            snapshot.decision,
            best_hypothesis_key=None,
        ),
    )
    assert _audit_hypothesis(no_selection) is None
    with pytest.raises(ValueError, match="identity is absent"):
        _audit_hypothesis(snapshot, "missing:hypothesis")


def test_sealed_audit_binds_packet_and_future_reveal(tmp_path) -> None:
    snapshot = _snapshot_with_sequence()
    histories = {
        Timeframe.H4: (
            candle(Timeframe.H4, "2025-01-06 06:00", 99.0),
        ),
        Timeframe.H1: (
            candle(Timeframe.H1, "2025-01-06 09:00", 99.5),
        ),
        Timeframe.M5: (
            candle(Timeframe.M5, "2025-01-06 09:55", 100.0),
        ),
        Timeframe.M1: (
            candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
        ),
    }
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    audit = SealedVisualAudit.seal(
        DecisionVisualizer(),
        snapshot,
        histories,
        tmp_path / "sealed",
        hypothesis_key=key,
    )
    result = _path_result(
        snapshot,
        outcome="target",
        resolved_at=snapshot.observation.asof + pd.Timedelta(minutes=1),
    )
    future_bar = Bar(
        start=snapshot.observation.asof,
        open=100.0,
        high=104.0,
        low=99.0,
        close=103.0,
        volume=100.0,
        symbol=snapshot.observation.symbol,
        instrument_id=snapshot.observation.instrument_id,
    )
    audit.on_bar(
        replace(
            future_bar,
            symbol="NQM5",
            instrument_id=future_bar.instrument_id + 1,
        )
    )
    assert audit.future_1m == []
    audit.on_bar(future_bar)
    reveal, record = audit.reveal(result)
    assert reveal.kind == "future_reveal"
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["future_was_separate_from_decision"]
    packet = json.loads(
        audit.decision_packet_path.read_text(encoding="utf-8")
    )
    hypothesis = packet["belief_t"]["hypotheses"][key]
    plan = hypothesis["plan"]
    assert packet["audit_hypothesis_key"] == key
    assert hypothesis["sequence"]["setup_id"] == audit.permit.setup_id
    assert plan["setup_id"] == audit.permit.setup_id
    assert plan["entry_location_id"] == audit.permit.entry_location_id
    assert plan["entry_path_id"] == audit.permit.entry_path_id
    assert packet["audit_context"] == sealed_path_audit_context(
        audit.permit.setup_id
    )
    assert packet["decision_trace"]["selected_hypothesis_key"] == key
    assert audit.decision_packet_hash == packet["packet_hash"]
    assert (
        payload["pre_reveal_decision_packet"]["path"]
        == str(audit.decision_packet_path)
    )
    assert (
        payload["pre_reveal_decision_packet"]["packet_hash"]
        == audit.decision_packet_hash
    )
    assert (
        payload["pre_reveal_decision_packet"]["sha256"]
        == audit.decision_packet_sha256
    )


def test_ai_review_is_converted_to_unvalidated_primitive() -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    packet_sha256 = decision_packet_payload_sha256(packet)
    proposals = AIReviewAdapter().convert(
        {
            **ai_review_identity(snapshot),
            "decision_packet_hash": packet["packet_hash"],
            "decision_packet_sha256": packet_sha256,
            "reviewer_id": "reviewer",
            "issues": [
                {
                    "code": ReviewIssue.LATE_DISPLACEMENT_CHASE.value,
                    "confidence": 0.8,
                    "note": "extension dominates remaining visible path",
                }
            ],
        },
        snapshot,
        decision_packet_hash=packet["packet_hash"],
        decision_packet_sha256=packet_sha256,
        decision_packet=packet,
    )
    assert len(proposals) == 1
    assert proposals[0].status == "unvalidated"
    assert "remaining" in proposals[0].formula
    assert (
        proposals[0].origin_value.decision_packet_hash
        == packet["packet_hash"]
    )


def test_empty_ai_review_is_valid_and_identity_bound() -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    packet_sha256 = decision_packet_payload_sha256(packet)
    review = {
        **ai_review_identity(snapshot),
        "decision_packet_hash": packet["packet_hash"],
        "decision_packet_sha256": packet_sha256,
        "reviewer_id": "reviewer",
        "issues": [],
    }
    assert (
        AIReviewAdapter().convert(
            review,
            snapshot,
            decision_packet_hash=packet["packet_hash"],
            decision_packet_sha256=packet_sha256,
            decision_packet=packet,
        )
        == ()
    )
    review["hypothesis_key"] = "another-hypothesis"
    with pytest.raises(ValueError, match="sealed decision and setup"):
        AIReviewAdapter().convert(
            review,
            snapshot,
            decision_packet_hash=packet["packet_hash"],
            decision_packet_sha256=packet_sha256,
            decision_packet=packet,
        )


def test_empty_ai_review_rejects_packet_hash_mismatch() -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    packet_sha256 = decision_packet_payload_sha256(packet)
    review = {
        **ai_review_identity(snapshot),
        "decision_packet_hash": "0" * 64,
        "decision_packet_sha256": packet_sha256,
        "reviewer_id": "reviewer",
        "issues": [],
    }
    with pytest.raises(ValueError, match="reviewed decision packet"):
        AIReviewAdapter().convert(
            review,
            snapshot,
            decision_packet_hash=packet["packet_hash"],
            decision_packet_sha256=packet_sha256,
            decision_packet=packet,
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "setup_id",
        "entry_location_id",
        "entry_path_id",
        "audit_hypothesis_key",
        "audit_context",
        "decision_trace",
    ),
)
def test_empty_ai_review_rejects_rehashed_packet_identity_drift(
    corruption: str,
) -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    packet.pop("packet_hash")
    if corruption in {"setup_id", "entry_location_id", "entry_path_id"}:
        packet["belief_t"]["hypotheses"][key]["plan"][corruption] = (
            f"another-{corruption}"
        )
    elif corruption == "audit_hypothesis_key":
        packet["audit_hypothesis_key"] = "another-hypothesis"
    elif corruption == "audit_context":
        packet["audit_context"]["setup_id"] = "another-setup"
    else:
        packet["decision_trace"]["selected_hypothesis_key"] = (
            "another-hypothesis"
        )
    packet["packet_hash"] = content_hash(packet)
    packet_sha256 = decision_packet_payload_sha256(packet)
    review = {
        **ai_review_identity(snapshot),
        "decision_packet_hash": packet["packet_hash"],
        "decision_packet_sha256": packet_sha256,
        "reviewer_id": "reviewer",
        "issues": [],
    }
    with pytest.raises(ValueError):
        AIReviewAdapter().convert(
            review,
            snapshot,
            decision_packet_hash=packet["packet_hash"],
            decision_packet_sha256=packet_sha256,
            decision_packet=packet,
        )


def test_ai_action_or_outcome_label_is_rejected() -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    packet_sha256 = decision_packet_payload_sha256(packet)
    with pytest.raises(ValueError):
        AIReviewAdapter().convert(
            {
                **ai_review_identity(snapshot),
                "decision_packet_hash": packet["packet_hash"],
                "decision_packet_sha256": packet_sha256,
                "issues": [],
                "recommended_action": "buy",
            },
            snapshot,
            decision_packet_hash=packet["packet_hash"],
            decision_packet_sha256=packet_sha256,
            decision_packet=packet,
        )


def test_typed_future_evidence_is_separate_and_packet_bound(
    tmp_path,
) -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    packet_path = _write_packet(
        packet,
        tmp_path / "decision_packet.json",
    )
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    hypothesis = snapshot.belief.hypotheses[key]
    assert hypothesis.sequence is not None
    assert hypothesis.plan is not None
    query = _registered_path_query(
        packet,
        key,
        ReviewIssue.PULLBACK_AS_INVALIDATION,
        "future-query",
        relevant_entity_ids=(
            hypothesis.sequence.setup_id,
            hypothesis.plan.entry_location_id,
        ),
    )
    recorder = PrimitivePathEvidenceRecorder.from_decision_packet(
        packet_path,
        hypothesis_key=key,
        query=query,
    )
    bar = Bar(
        start=snapshot.observation.asof,
        open=100.0,
        high=100.5,
        low=99.75,
        close=100.25,
        volume=10.0,
        symbol=snapshot.observation.symbol,
        instrument_id=snapshot.observation.instrument_id,
    )
    assert recorder.observe(bar) is None
    recorder.close_right_boundary(bar.end)
    evidence_path = recorder.write(
        tmp_path / "future_evidence.json"
    )
    verified = read_verified_primitive_path_evidence(
        evidence_path,
        decision_packet_path=packet_path,
    )
    assert verified.identity.decision_hash == snapshot.snapshot_hash
    assert verified.finalization_reason == "right_boundary"
    assert len(verified.points) == 1
    assert "selected_action" not in evidence_path.read_text(
        encoding="utf-8"
    )


def test_typed_future_evidence_contract_boundary_excludes_new_ohlc(
    tmp_path,
) -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    packet_path = _write_packet(
        packet,
        tmp_path / "decision_packet.json",
    )
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    hypothesis = snapshot.belief.hypotheses[key]
    assert hypothesis.sequence is not None
    query = _registered_path_query(
        packet,
        key,
        ReviewIssue.MISSED_STRUCTURE_SEQUENCE,
        "contract-boundary-query",
        relevant_entity_ids=(hypothesis.sequence.setup_id,),
    )
    recorder = PrimitivePathEvidenceRecorder.from_decision_packet(
        packet_path,
        hypothesis_key=key,
        query=query,
    )
    new_contract_bar = Bar(
        start=snapshot.observation.asof,
        open=200.0,
        high=201.0,
        low=199.0,
        close=200.5,
        volume=20.0,
        symbol="NQM5",
        instrument_id=snapshot.observation.instrument_id + 1,
    )
    malformed_transition = {
        "family": "entry_path_boundary",
        "entity_id": hypothesis.sequence.setup_id,
        "revision_id": "old-path-censored",
        "lifecycle": "censored",
        "observed_at": new_contract_bar.end,
        "reason": "contract_change_reset",
    }
    with pytest.raises(ValueError, match="top-level schema"):
        recorder.observe(
            new_contract_bar,
            typed_state_transitions=(malformed_transition,),
        )
    frozen_path = snapshot.observation.path_sequences[0]
    terminal_path = replace(
        frozen_path,
        lifecycle=PathSequenceLifecycle.CENSORED,
        state_started_at=new_contract_bar.end,
        last_updated_at=new_contract_bar.end,
        state_duration_real_1m_bars=0,
        ended_at=new_contract_bar.end,
        transition_reason="contract_change_reset",
    )
    transition = {
        "family": "entry_path_boundary",
        "entity_id": hypothesis.sequence.setup_id,
        "revision_id": (
            "entry_path_boundary:"
            f"{hypothesis.sequence.setup_id}:censored:"
            f"{new_contract_bar.end}"
        ),
        "lifecycle": "censored",
        "observed_at": new_contract_bar.end,
        "reason": "contract_change_reset",
        "state": to_primitive(terminal_path),
    }
    with pytest.raises(ValueError, match="production schema"):
        recorder.observe(
            new_contract_bar,
            typed_state_transitions=(
                {
                    **transition,
                    "state": {
                        "sequence_id": hypothesis.sequence.setup_id,
                        "lifecycle": "censored",
                        "last_updated_at": new_contract_bar.end,
                        "transition_reason": "contract_change_reset",
                    },
                },
            ),
        )
    evidence = recorder.observe(
        new_contract_bar,
        typed_state_transitions=(transition,),
    )
    assert evidence is not None
    assert evidence.finalization_reason == "contract_change_reset"
    assert evidence.points == ()
    assert evidence.boundary_delta is not None
    assert evidence.boundary_delta.typed_state_transitions == (
        to_primitive(transition),
    )

    empty_recorder = PrimitivePathEvidenceRecorder.from_decision_packet(
        packet_path,
        hypothesis_key=key,
        query=_registered_path_query(
            packet,
            key,
            ReviewIssue.MISSED_STRUCTURE_SEQUENCE,
            "empty-contract-boundary-query",
            relevant_entity_ids=(hypothesis.sequence.setup_id,),
        ),
    )
    empty_evidence = empty_recorder.observe(new_contract_bar)
    assert empty_evidence is not None
    assert empty_evidence.finalization_reason == "contract_change_reset"
    assert empty_evidence.boundary_delta is not None
    assert empty_evidence.boundary_delta.typed_state_transitions == ()


def test_typed_future_evidence_rejects_labels_and_includes_deadline_bar(
    tmp_path,
) -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    packet_path = _write_packet(
        packet,
        tmp_path / "decision_packet.json",
    )
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    hypothesis = snapshot.belief.hypotheses[key]
    assert hypothesis.plan is not None
    recorder = PrimitivePathEvidenceRecorder.from_decision_packet(
        packet_path,
        hypothesis_key=key,
        query=_registered_path_query(
            packet,
            key,
            ReviewIssue.WRONG_LIQUIDITY_DRAW,
            "deadline-query",
        ),
    )

    def future_bar(start: pd.Timestamp) -> Bar:
        return Bar(
            start=start,
            open=100.0,
            high=100.25,
            low=99.75,
            close=100.0,
            volume=1.0,
            symbol=snapshot.observation.symbol,
            instrument_id=snapshot.observation.instrument_id,
        )

    first = future_bar(snapshot.observation.asof)
    with pytest.raises(ValueError, match="forbidden field"):
        recorder.observe(
            first,
            events_added=(
                {
                    "event_id": "forbidden-label",
                    "observed_at": first.end,
                    "action": "enter",
                },
            ),
        )
    cursor = snapshot.observation.asof
    result = None
    while cursor < hypothesis.plan.deadline:
        result = recorder.observe(future_bar(cursor))
        cursor += pd.Timedelta(minutes=1)
    assert result is not None
    assert result.finalization_reason == "deadline"
    assert result.points[-1].observed_at == hypothesis.plan.deadline


def test_implementation_checks_cannot_mark_ai_primitive_as_passed() -> None:
    snapshot = _snapshot_with_sequence()
    packet = _decision_packet(snapshot)
    packet_sha256 = decision_packet_payload_sha256(packet)
    proposal = AIReviewAdapter().convert(
        {
            **ai_review_identity(snapshot),
            "decision_packet_hash": packet["packet_hash"],
            "decision_packet_sha256": packet_sha256,
            "reviewer_id": "reviewer",
            "issues": [
                {
                    "code": ReviewIssue.LATE_DISPLACEMENT_CHASE.value,
                    "confidence": 0.8,
                    "note": "extension dominates remaining visible path",
                }
            ],
        },
        snapshot,
        decision_packet_hash=packet["packet_hash"],
        decision_packet_sha256=packet_sha256,
        decision_packet=packet,
    )[0]

    def implementation_case(index: int) -> PrimitivePathCaseResult:
        return PrimitivePathCaseResult(
            proposal_id=proposal.proposal_id,
            definition_hash=proposal.definition_hash,
            origin_packet_hash=proposal.decision_packet_hash,
            origin_packet_sha256=proposal.decision_packet_sha256,
            evaluation_packet_hash=str(index) * 64,
            evaluation_packet_sha256=str(index + 2) * 64,
            origin_decision_hash=proposal.decision_hash,
            evaluation_decision_hash=str(index + 4) * 64,
            origin_hypothesis_key=proposal.hypothesis_key,
            evaluation_hypothesis_key=proposal.hypothesis_key or "missing",
            origin_setup_id=proposal.setup_id,
            evaluation_setup_id=f"independent-setup-{index}",
            evaluation_entry_location_id=f"independent-location-{index}",
            evaluation_entry_path_id=f"independent-path-{index}",
            origin_window_role="development",
            evaluation_window_role=f"blind-window-{index}",
            distinct_case=True,
            non_overlapping_window=True,
            causal_clock_valid=True,
            prefix_invariant=True,
            bounds_valid=True,
            implementation_consistency_passed=True,
            path_evidence_hash=None,
            path_evidence_verified=False,
            path_property_passed=None,
            evaluable=True,
            status="implementation_checked",
            reason="implementation consistency only",
            no_pnl_fields_used=True,
            origin_values={},
            evaluation_values={},
        )

    result = finalize_primitive_evaluation(
        proposal,
        (implementation_case(1), implementation_case(2)),
    )
    assert result.implementation_checked_cases == 2
    assert result.path_passed_cases == 0
    assert result.status == "awaiting_path_evidence"
    assert "cannot approve" in result.reason

    eligible = replace(
        implementation_case(1),
        path_evidence_hash="e" * 64,
        path_evidence_verified=True,
        path_property_passed=True,
    )
    ineligible = replace(
        implementation_case(2),
        non_overlapping_window=False,
        path_evidence_hash="f" * 64,
        path_evidence_verified=True,
        path_property_passed=True,
    )
    guarded = finalize_primitive_evaluation(
        proposal,
        (eligible, eligible, ineligible),
    )
    assert guarded.path_passed_cases == 0
    assert guarded.path_failed_cases == 0
    assert guarded.status == "awaiting_path_evidence"
