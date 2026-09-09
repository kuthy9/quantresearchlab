"""Core causal decision-view and price-overlay regression tests."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from contract.market import (
    Direction,
    Playbook,
    PlaybookPhase,
    Timeframe,
)
from contract.eye import (
    EntryLocationLifecycle,
    EntryLocationState,
    InteractionUpdate,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
)
from contract.brain import (
    DrawSelection,
    GlobalConflictRole,
    HypothesisSequenceState,
    LiquidityRoute,
    SequenceStepState,
)
from shares.core.scene_graph import (
    EvidenceStatus,
    FocusState,
    HypothesisState,
)
from shares.core.visualization import (
    DecisionVisualizer,
    _belief_geometry_overlay,
    _collision_safe_annotate,
    _event_timeline,
    _group4_text,
    _liquidity_route_text,
    _metric_text,
    _partial_geometry_text,
    _plan_display_context,
    _plan_overlay,
    _sequence_text,
    _selected_group5_entities,
    _selected_hypothesis,
    _short_identity,
    _support_resistance_style,
    _temporal_market_reading_text,
    _typed_structure_overlay,
    validate_causal_histories,
)

from shares.tests.helpers import candle, engine_snapshot


def test_h1_metric_text_labels_rolling_envelope_as_descriptive_context() -> None:
    metrics = {
        "swing_progression": 0.25,
        "acceptance_direction": 0.1,
        "rejection_direction": -0.2,
        "rolling_range_position": 0.3,
        "up_path_obstruction_atr": 1.5,
        "down_path_obstruction_atr": 1.0,
    }
    observation = SimpleNamespace(
        frame=lambda timeframe: SimpleNamespace(metrics=metrics)
    )
    snapshot = SimpleNamespace(observation=observation)

    text = _metric_text(snapshot, Timeframe.H1, None)

    assert "rolling envelope position" in text
    assert "range position" not in text
    assert "descriptive proxy: acceptance" in text
    assert "descriptive proxy: rejection" in text


def test_h4_and_m5_rolling_metrics_are_labeled_as_descriptive_proxies() -> None:
    h4_metrics = {
        "directional_displacement": 0.4,
        "path_efficiency": 0.6,
        "structure_direction": 1.0,
        "structure_age_bars": 3.0,
        "range_position": 0.7,
        "external_above_distance_atr": 2.0,
        "external_below_distance_atr": 1.0,
    }
    m5_metrics = {"compression": 0.25}
    observation = SimpleNamespace(
        displacement=None,
        frame=lambda timeframe: SimpleNamespace(
            metrics=(
                h4_metrics
                if timeframe is Timeframe.H4
                else m5_metrics
            )
        ),
    )
    snapshot = SimpleNamespace(observation=observation)

    h4_text = _metric_text(snapshot, Timeframe.H4, None)
    m5_text = _metric_text(snapshot, Timeframe.M5, None)

    assert "descriptive proxy: directional move" in h4_text
    assert "not typed displacement" in h4_text
    assert "descriptive proxy: compression" in m5_text
    assert "not mature range" in m5_text


def test_eye_only_group5_selection_prioritizes_sampled_case_identity() -> None:
    paths = (
        SimpleNamespace(sequence_id="path:old", context_id="zone:old"),
        SimpleNamespace(sequence_id="path:focus", context_id="zone:focus"),
        SimpleNamespace(sequence_id="path:new", context_id="pool:new"),
        SimpleNamespace(sequence_id="path:peer", context_id="zone:focus"),
    )
    observation = SimpleNamespace(
        interaction_update=SimpleNamespace(
            zone_interactions=(
                SimpleNamespace(location_id="zone:old"),
                SimpleNamespace(location_id="zone:focus"),
            ),
            interaction_paths=paths,
            reacceptance_interactions=(
                SimpleNamespace(context_id="zone:old"),
                SimpleNamespace(context_id="zone:focus"),
            ),
            micro_break_facts=(
                SimpleNamespace(context_id="zone:old"),
                SimpleNamespace(context_id="zone:focus"),
            ),
        ),
    )
    selected = _selected_group5_entities(
        SimpleNamespace(observation=observation),
        None,
        "path:focus",
    )

    assert selected[0][-1].location_id == "zone:focus"
    assert selected[1][-1].sequence_id == "path:focus"
    assert selected[2][-1].context_id == "zone:focus"
    assert selected[3][-1].context_id == "zone:focus"


def test_eye_range_diagnostic_lists_failed_gates_and_margins() -> None:
    asof = pd.Timestamp("2023-07-19T21:00:00-04:00")
    diagnostic = SimpleNamespace(
        observed_at=asof,
        maturity_range_id="range:focus",
        unmet_maturity_gates=("duration", "compression"),
        maturity_gates=(
            ("duration", 2.0, 19.0, -17.0),
            ("compression", -0.2, 0.4, -0.6),
        ),
    )
    observation = SimpleNamespace(
        asof=asof,
        frame=lambda _timeframe: SimpleNamespace(dealing_ranges=()),
        manipulations=(),
        group4_boundary_range_transitions=(),
        group4_boundary_manipulation_transitions=(),
        group4_ambiguous_sweep_item_ids=(),
        group4_atr_unready_sweep_item_ids=(),
        group4_range_funnel=(diagnostic,),
    )

    text = _group4_text(
        SimpleNamespace(observation=observation),
        "range:focus",
    )

    assert "unmet=duration,compression" in text
    assert "duration: actual=2.000 threshold=19.000 margin=-17.000" in text
    assert "compression: actual=-0.200 threshold=0.400 margin=-0.600" in text


@pytest.mark.parametrize(
    ("source_kind", "expected_color"),
    (
        ("structural_swing", "#0f766e"),
        ("previous_session", "#2563eb"),
        ("previous_day", "#d97706"),
        ("previous_week", "#7c3aed"),
        ("range_boundary", "#be123c"),
    ),
)
def test_support_resistance_overlay_displays_source_kind_and_style(
    source_kind: str,
    expected_color: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    values = (
        candle(Timeframe.M1, "2025-01-06 09:58", 100.0),
        candle(Timeframe.M1, "2025-01-06 09:59", 100.1),
    )
    zone = SimpleNamespace(
        zone_id=f"zone:{source_kind}",
        side="resistance",
        lower_bound=99.5,
        upper_bound=100.5,
        anchor_price=100.0,
        confirmed_at=values[0].end,
        lifecycle=SimpleNamespace(value="active"),
        retired_at=None,
        reaccepted_at=None,
        broken_at=None,
        source_kind=source_kind,
        structural_rank="external",
        is_protected_swing=False,
        visibility_strength=0.8,
        reaction_quality=0.5,
        freshness=0.9,
        depletion_risk=0.1,
    )
    frame = SimpleNamespace(
        swings=(),
        structure_breaks=(),
        support_resistance=(zone,),
        liquidity_pools=(),
    )
    snapshot = SimpleNamespace(
        observation=SimpleNamespace(
            frame=lambda _timeframe: frame,
            liquidity_pool_states=(),
        )
    )
    figure, axis = plt.subplots()
    axis.set_xlim(-0.5, len(values) - 0.5)
    axis.set_ylim(98.0, 102.0)

    _typed_structure_overlay(
        axis,
        snapshot,
        Timeframe.M1,
        values,
    )

    labels = tuple(text.get_text() for text in axis.texts)
    rendered_colors = {
        to_hex(color, keep_alpha=False)
        for collection in axis.collections
        for getter_name in (
            "get_colors",
            "get_facecolors",
            "get_edgecolors",
        )
        for getter in (getattr(collection, getter_name, None),)
        if getter is not None
        for color in getter()
    }
    plt.close(figure)

    assert f"S/R src={source_kind}" in "\n".join(labels)
    assert expected_color in rendered_colors
    assert _support_resistance_style(source_kind)[0] == expected_color




def _scene_reading_snapshot():
    snapshot = engine_snapshot()
    asof = snapshot.observation.asof
    m15_frame = replace(
        snapshot.observation.frame(Timeframe.H1),
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
        primary_timeframes=(
            Timeframe.H1.value,
            Timeframe.M15.value,
            Timeframe.M5.value,
        ),
        reason_codes=("new_key_event", "cross_scale_conflict"),
        question="did the bridge BOS align with the setup displacement?",
        resolution_status=EvidenceStatus.AMBIGUOUS,
        switched=True,
        hypothesis_id="hyp:dfp-long",
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
        material_conflict_ids=(),
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
        material_conflict_ids=("conflict:m15-vs-h1",),
        missing_evidence={
            "opposite_displacement": EvidenceStatus.NOT_OBSERVED,
        },
        ambiguous_evidence={},
        context_draw_id="node:m15-pool",
        primary_target_id="node:m5-liquidity",
        invalidation_id="node:sweep-extreme",
        evidence_revision_id="evidence:1",
    )
    base_belief = next(iter(snapshot.belief.hypotheses.values()))

    def root_candidate(
        candidate_id: str,
        root_id: str,
        playbook: Playbook,
        direction: Direction,
    ):
        thesis_id = f"market-thesis:{root_id}"
        return replace(
            base_belief,
            playbook=playbook,
            direction=direction,
            plan=None,
            plan_feasibility=None,
            candidate_id=candidate_id,
            required_root_id=root_id,
            record_kind="root_candidate",
            market_thesis_ids=(thesis_id,),
            market_thesis_id=thesis_id,
            bound_market_thesis_id=thesis_id,
            market_thesis_root_id=root_id,
            market_thesis_mechanism="visual-test",
            market_thesis_authority_relation="visual-test",
            playbook_match_strength=1.0,
            market_thesis_binding_required=True,
            market_thesis_action_bound=True,
            market_thesis_match_status="exact_root_bound",
        )

    thesis_candidates = {
        dominant.hypothesis_id: root_candidate(
            dominant.hypothesis_id,
            "node:h4-structure",
            dominant.playbook,
            dominant.direction,
        ),
        competitor.hypothesis_id: root_candidate(
            competitor.hypothesis_id,
            "node:m15-pool",
            competitor.playbook,
            competitor.direction,
        ),
    }
    belief = replace(
        snapshot.belief,
        asof=asof,
        thesis_candidates=thesis_candidates,
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
        global_context=SimpleNamespace(
            updated_at=asof,
            scene_revision_id="scene:revision-2",
            open_market_theses=(
                SimpleNamespace(root_id="node:h4-structure"),
                SimpleNamespace(root_id="node:m15-pool"),
            ),
            material_conflicts=(
                SimpleNamespace(
                    conflict_id="conflict:m15-vs-h1",
                    role=GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE,
                    source_timeframe=Timeframe.M15,
                    target_timeframe=Timeframe.H1,
                    source_direction=Direction.SHORT,
                    target_direction=Direction.LONG,
                    event_id="event:m15-bos",
                ),
            ),
        ),
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


def _hard_boundary_visual_fixture():
    snapshot = engine_snapshot()
    asof = snapshot.observation.asof
    frames = dict(snapshot.observation.frames)
    for timeframe in (Timeframe.H4, Timeframe.H1, Timeframe.M5):
        frames[timeframe] = replace(
            frames[timeframe],
            cutoff=asof,
            bars=0,
            ready=False,
        )
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        cutoff=asof,
        bars=1,
        ready=False,
    )
    observation = replace(
        snapshot.observation,
        frames=frames,
        anomalies=("data_gap_history_reset",),
    )
    return (
        replace(snapshot, observation=observation),
        {
            Timeframe.H4: (),
            Timeframe.H1: (),
            Timeframe.M5: (),
            Timeframe.M1: (
                candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
            ),
        },
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
            source_event_id="visual-fvg",
            source_entity_id="visual-fvg",
            predecessor_step_ids=(),
            same_clock_relation="origin",
            direction=Direction.LONG,
            strength=0.8,
            reason="typed_entry_zone_registered",
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
            reason="later_close_on_delivery_side",
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
            reason="crossed_near_edge",
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
        source_zone_detector_protocol_hash="f" * 64,
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
            interaction_update=InteractionUpdate(
                zone_interactions=(location,),
                reacceptance_interactions=(),
                micro_break_facts=(),
                interaction_paths=(path,),
            ),
        ),
        belief=replace(
            snapshot.belief,
            hypotheses={key: hypothesis},
        ),
        decision=replace(snapshot.decision, plan=plan),
    )






def test_five_scale_visual_exposes_temporal_market_reading(
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
    assert snapshot.observation.active_timeframes == (
        Timeframe.H4,
        Timeframe.H1,
        Timeframe.M15,
        Timeframe.M5,
        Timeframe.M1,
    )
    assert snapshot.belief.focus_state is not None
    assert (
        snapshot.belief.focus_state.resolution_status
        is EvidenceStatus.AMBIGUOUS
    )
    assert snapshot.belief.dominant_hypothesis_id == "hyp:dfp-long"
    assert snapshot.belief.competing_hypothesis_ids == ("hyp:lsr-short",)

    text = _temporal_market_reading_text(snapshot)
    assert "primary 1H, 15m, 5m" in text
    assert "ACTIVE COMPETING HYPOTHESES" in text
    assert "ALIGNS_WITH" in text
    assert "authority_transition_candidate" in text
    assert "micro_bos=unknown" in text
    assert "UNKNOWN is unresolved evidence, never FALSE" in text

    artifact = DecisionVisualizer().render_decision(
        snapshot,
        histories,
        tmp_path / "five-scale-decision.png",
    )
    assert artifact.maximum_market_time == snapshot.observation.asof
    assert artifact.path.is_file()


def test_eye_only_visual_renders_typed_observation_without_brain(
    tmp_path,
) -> None:
    snapshot = _scene_reading_snapshot()
    histories = _scene_reading_histories()

    artifact = DecisionVisualizer().render_observation(
        snapshot.observation,
        histories,
        tmp_path / "eye-only-observation.png",
        case_id="eye-case-001",
    )

    assert artifact.kind == "eye_observation"
    assert artifact.decision_id == "eye-case-001"
    assert artifact.maximum_market_time == snapshot.observation.asof
    assert artifact.hypothesis_key is None
    assert artifact.path.is_file()


def test_eye_only_visual_artifact_binds_sampled_group5_path(
    tmp_path,
) -> None:
    snapshot = _snapshot_with_sequence()
    asof = snapshot.observation.asof
    histories = {
        Timeframe.H4: (
            candle(
                Timeframe.H4,
                f"{asof - pd.Timedelta(hours=4):%Y-%m-%d %H:%M}",
                99.0,
            ),
        ),
        Timeframe.H1: (
            candle(
                Timeframe.H1,
                f"{asof - pd.Timedelta(hours=1):%Y-%m-%d %H:%M}",
                99.5,
            ),
        ),
        Timeframe.M5: (
            candle(
                Timeframe.M5,
                f"{asof - pd.Timedelta(minutes=5):%Y-%m-%d %H:%M}",
                99.8,
            ),
        ),
        Timeframe.M1: (
            candle(
                Timeframe.M1,
                f"{asof - pd.Timedelta(minutes=1):%Y-%m-%d %H:%M}",
                100.0,
            ),
        ),
    }

    artifact = DecisionVisualizer().render_observation(
        snapshot.observation,
        histories,
        tmp_path / "eye-focused-path.png",
        case_id="eye-group5-case",
        case_entity_id="visual-setup",
    )

    assert artifact.entry_path_id == "visual-setup"
    assert artifact.path.is_file()


def test_hard_boundary_blank_higher_timeframe_histories_are_valid() -> None:
    snapshot, histories = _hard_boundary_visual_fixture()

    validated = validate_causal_histories(snapshot, histories)

    assert validated[Timeframe.H4] == ()
    assert validated[Timeframe.H1] == ()
    assert validated[Timeframe.M5] == ()
    assert validated[Timeframe.M1][-1].end == snapshot.observation.asof


@pytest.mark.parametrize(
    "invalid_frame_state",
    ("bars", "ready", "cutoff"),
)
def test_empty_history_requires_exact_blank_frame_state(
    invalid_frame_state: str,
) -> None:
    snapshot, histories = _hard_boundary_visual_fixture()
    asof = snapshot.observation.asof
    frame = snapshot.observation.frame(Timeframe.H4)
    updates = {
        "bars": {"bars": 1},
        "ready": {"ready": True},
        "cutoff": {"cutoff": asof - pd.Timedelta(minutes=1)},
    }
    frames = dict(snapshot.observation.frames)
    frames[Timeframe.H4] = replace(
        frame,
        **updates[invalid_frame_state],
    )
    snapshot = replace(
        snapshot,
        observation=replace(snapshot.observation, frames=frames),
    )

    with pytest.raises(
        ValueError,
        match="empty causal history disagrees with observation frame",
    ):
        validate_causal_histories(snapshot, histories)


def test_blank_m1_history_still_fails_the_decision_clock() -> None:
    snapshot, histories = _hard_boundary_visual_fixture()
    frames = dict(snapshot.observation.frames)
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        bars=0,
        ready=False,
    )
    snapshot = replace(
        snapshot,
        observation=replace(snapshot.observation, frames=frames),
    )
    histories = {**histories, Timeframe.M1: ()}

    with pytest.raises(
        ValueError,
        match="causal M1 history does not reach the decision clock",
    ):
        validate_causal_histories(snapshot, histories)


def test_nonempty_history_still_requires_exact_frame_cutoff() -> None:
    snapshot = engine_snapshot()
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
    frames = dict(snapshot.observation.frames)
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        cutoff=snapshot.observation.asof - pd.Timedelta(hours=1),
    )
    snapshot = replace(
        snapshot,
        observation=replace(snapshot.observation, frames=frames),
    )

    with pytest.raises(
        ValueError,
        match="causal history cutoff disagrees with observation",
    ):
        validate_causal_histories(snapshot, histories)


def test_hard_boundary_visuals_label_blank_panels_as_warmup(
    tmp_path,
    monkeypatch,
) -> None:
    import matplotlib
    import shares.core.visualization as visualization

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    snapshot, histories = _hard_boundary_visual_fixture()
    titles: list[str] = []
    plan_overlay_histories = []
    original_set_title = Axes.set_title
    original_plan_overlay = visualization._plan_overlay

    def recording_set_title(self, label, *args, **kwargs):
        titles.append(str(label))
        return original_set_title(self, label, *args, **kwargs)

    def recording_plan_overlay(axis, plan, candles, **kwargs):
        plan_overlay_histories.append(tuple(candles))
        return original_plan_overlay(axis, plan, candles, **kwargs)

    monkeypatch.setattr(Axes, "set_title", recording_set_title)
    monkeypatch.setattr(
        visualization,
        "_plan_overlay",
        recording_plan_overlay,
    )
    visualizer = DecisionVisualizer()

    eye_artifact = visualizer.render_observation(
        snapshot.observation,
        histories,
        tmp_path / "hard-boundary-eye.png",
    )
    eye_titles = tuple(titles)
    titles.clear()
    decision_artifact = visualizer.render_decision(
        snapshot,
        histories,
        tmp_path / "hard-boundary-decision.png",
    )
    decision_titles = tuple(titles)

    for rendered_titles in (eye_titles, decision_titles):
        for timeframe in (Timeframe.H4, Timeframe.H1, Timeframe.M5):
            assert any(
                title.startswith(timeframe.value)
                and "blank/warmup" in title
                and "causal cutoff" in title
                for title in rendered_titles
            )
    assert eye_artifact.maximum_market_time == snapshot.observation.asof
    assert decision_artifact.maximum_market_time == snapshot.observation.asof
    assert plan_overlay_histories
    assert all(plan_overlay_histories)
    assert eye_artifact.path.is_file()
    assert decision_artifact.path.is_file()


def test_five_scale_history_validation_fails_closed_when_m15_is_missing() -> None:
    snapshot = _scene_reading_snapshot()
    histories = _scene_reading_histories()
    histories.pop(Timeframe.M15)

    with pytest.raises(ValueError, match="enabled scale registry"):
        validate_causal_histories(snapshot, histories)


def test_visual_exposes_liquidity_route_roles() -> None:
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
        range_context_id="range:h1-balance",
        range_midpoint=100.0,
        swept_range_boundary_id="range-boundary:upper",
        opposing_range_boundary_id="range-boundary:lower",
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

    text = _partial_geometry_text(routed_snapshot, routed_hypothesis)
    assert "context draw draw:h4-external" in text
    assert f"primary deliverable {target.level_id}" in text
    assert "terminal draw draw:h4-terminal" in text
    assert "path blockers blocker:h1-opposing-pool" in text
    assert "optional range context range:h1-balance" in text
    assert "range midpoint/value 100.00" in text
    assert "swept range boundary range-boundary:upper" in text
    assert "opposing range liquidity range-boundary:lower" in text
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


def test_terminal_plan_is_non_actionable() -> None:
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
    )
    assert heading == "HISTORICAL PLAN — NON-ACTIONABLE (invalidated)"
    assert not show


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


def test_episode_deadline_is_visible_in_setup_and_partial_geometry(
    monkeypatch,
) -> None:
    import shares.core.visualization as visualization

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
        "_selected_hypothesis",
        lambda *_args, **_kwargs: proxy,
    )
    assert expected in _sequence_text(snapshot)


def test_visual_hypothesis_selection_uses_decision_then_ranked_fallback() -> None:
    snapshot = engine_snapshot()
    selected = snapshot.decision.best_hypothesis_key
    assert selected is not None
    assert _selected_hypothesis(snapshot).key == selected

    no_selection = replace(
        snapshot,
        decision=replace(
            snapshot.decision,
            best_hypothesis_key=None,
        ),
    )
    assert _selected_hypothesis(no_selection) is not None


def test_visual_hypothesis_selection_resolves_root_action_candidate() -> None:
    snapshot = engine_snapshot()
    candidate = next(iter(snapshot.belief.hypotheses.values()))
    candidate_id = "open-root-1|displacement_first_pullback|long"
    belief = SimpleNamespace(
        hypotheses=snapshot.belief.hypotheses,
        resolve_hypothesis=lambda identity: (
            candidate if identity == candidate_id else None
        ),
        ranked=snapshot.belief.ranked,
    )
    selected = SimpleNamespace(
        decision=SimpleNamespace(best_hypothesis_key=candidate_id),
        belief=belief,
    )

    assert _selected_hypothesis(selected) is candidate
