from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.artifact_stream import (
    new_stream_state,
    write_stream_manifest,
    write_stream_shard,
)
from smc_trader.causal_cases import (
    CAUSAL_CASE_INPUT_FIELD_TYPES,
    CAUSAL_CASE_OUTCOME_FIELD_TYPES,
    CAUSAL_CASE_PROTOCOL,
    CAUSAL_CASE_PROTOCOL_VERSION,
    CAUSAL_CASE_RECORDER_SCHEMA_VERSION,
    CausalCaseRecorder,
    case_input_to_representation_mapping,
    expected_causal_case_run_identity,
    validate_case_input_row,
    validate_case_library_rows,
    validate_episode_disjoint_splits,
    write_causal_case_library_manifest,
)
from smc_trader.interaction import (
    INTERACTION_ARTIFACT_COLLECTION_NAMES,
    INTERACTION_CURRENT_ARTIFACT_COLLECTION_NAMES,
    INTERACTION_DELTA_ARTIFACT_COLLECTION_NAMES,
)
from smc_trader.model import (
    Bar,
    ContextThesisState,
    Direction,
    DrawSelection,
    EntryEpisodeState,
    EntryLocationLifecycle,
    EntryLocationState,
    FrameObservation,
    FrozenLSRContext,
    FrozenTriggerState,
    InteractionUpdate,
    LiquidityLevel,
    LiquidityRoute,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
    Playbook,
    PlaybookPhase,
    StructuralLevel,
    Timeframe,
    TradePlan,
)
from smc_trader.market_representation import (
    representation_case_from_case_input_row,
)
from smc_trader.scene_graph import SceneEdgeKind, StructuralScale


def _clock(minutes: int = 0) -> pd.Timestamp:
    return pd.Timestamp("2025-01-06 10:00", tz="America/New_York") + pd.Timedelta(
        minutes=minutes
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _bar(asof: pd.Timestamp, *, synthetic_no_trade: bool = False) -> Bar:
    return Bar(
        start=asof - pd.Timedelta(minutes=1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
        symbol="NQH5",
        instrument_id=1,
        synthetic_no_trade=synthetic_no_trade,
    )


def _context(
    asof: pd.Timestamp,
    *,
    formed_at: pd.Timestamp | None = None,
    authority: tuple[str, ...] = ("root:1", "displacement:1"),
    supporting: tuple[str, ...] = ("authority:1",),
) -> ContextThesisState:
    formed = asof if formed_at is None else formed_at
    return ContextThesisState(
        context_thesis_id="context:1",
        market_epoch_id="epoch:1",
        direction=Direction.LONG,
        authority_ids=authority,
        context_draw=None,
        structural_invalidation=StructuralLevel(
            price=95.0,
            side="below",
            source_level_id=authority[0],
            observed_at=formed,
            rationale="test",
        ),
        supporting_event_ids=supporting,
        opposing_event_ids=(),
        lifecycle="active",
        formed_at=formed,
        updated_at=asof,
        thesis_deadline=asof + pd.Timedelta(hours=4),
    )


def _episode(
    asof: pd.Timestamp,
    *,
    playbook: Playbook = Playbook.LIQUIDITY_SWEEP_REVERSAL,
    location_id: str | None = None,
    path_id: str | None = None,
    first_pullback_at: pd.Timestamp | None = None,
    trigger: FrozenTriggerState | None = None,
    plan: TradePlan | None = None,
    phase: PlaybookPhase = PlaybookPhase.FORMING,
    terminal_at: pd.Timestamp | None = None,
    terminal_reason: str | None = None,
) -> EntryEpisodeState:
    return EntryEpisodeState(
        episode_id="episode:1",
        parent_context_thesis_id="context:1",
        candidate_id="candidate:1",
        playbook=playbook,
        direction=Direction.LONG,
        initiating_event_id="root:1",
        entry_location_id=location_id,
        entry_path_id=path_id,
        first_pullback_at=first_pullback_at,
        selected_trigger=trigger,
        plan=plan,
        invalidation=None if plan is None else plan.invalidation,
        deadline=_clock(240),
        phase=phase,
        # Deliberately not used as recorder admission time.
        formed_at=asof - pd.Timedelta(minutes=20),
        updated_at=asof,
        terminal_at=terminal_at,
        terminal_reason=terminal_reason,
    )


def _location(asof: pd.Timestamp, *, formed_at: pd.Timestamp | None = None) -> EntryLocationState:
    formed = asof if formed_at is None else formed_at
    return EntryLocationState(
        location_id="location:1",
        protocol_hash="p5",
        source_zone_detector_protocol_hash="p3",
        symbol="NQH5",
        instrument_id=1,
        direction=Direction.LONG,
        source_zone_kind="fvg",
        source_zone_id="zone:1",
        source_zone_protocol_hash="p3",
        source_displacement_id="displacement:1",
        source_bos_id=None,
        lower_bound=99.0,
        upper_bound=100.0,
        midpoint=99.5,
        near_edge=100.0,
        far_edge=99.0,
        failure_boundary=99.0,
        formed_at=formed,
        lifecycle=EntryLocationLifecycle.APPROACHING,
        state_started_at=formed,
        last_updated_at=asof,
        age_real_1m_bars=0,
        state_duration_real_1m_bars=0,
        current_price=100.5,
        distance_to_zone_points=0.5,
        distance_to_failure_points=1.5,
    )


def _path(asof: pd.Timestamp, *, formed_at: pd.Timestamp | None = None) -> PathSequenceState:
    formed = asof if formed_at is None else formed_at
    return PathSequenceState(
        sequence_id="path:1",
        protocol_hash="p5",
        symbol="NQH5",
        instrument_id=1,
        context_kind="zone_return",
        context_id="location:1",
        direction=Direction.LONG,
        lifecycle=PathSequenceLifecycle.ACTIVE,
        formed_at=formed,
        state_started_at=formed,
        last_updated_at=asof,
        age_real_1m_bars=0,
        state_duration_real_1m_bars=0,
        steps=(
            PathSequenceStep(
                step_id="step:zone-visible",
                kind="zone_visible",
                observed_at=formed,
                source_event_id="zone:1",
                source_entity_id="zone:1",
                predecessor_step_ids=(),
                same_clock_relation="origin",
                direction=Direction.LONG,
                strength=1.0,
                reason="typed_entry_zone_registered",
            ),
        ),
    )


def _plan(asof: pd.Timestamp, *, remaining_path_R: float = 1.2) -> TradePlan:
    invalidation = StructuralLevel(
        price=95.0,
        side="below",
        source_level_id="stop:1",
        observed_at=asof,
        rationale="test",
    )
    target = LiquidityLevel(
        level_id="draw:1",
        timeframe=Timeframe.H1,
        side="above",
        price=105.0,
        formed_at=asof,
        confirmed_at=asof,
        touches=1,
    )
    return TradePlan(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        planned_entry=99.5,
        invalidation=invalidation,
        targets=(target,),
        risk_points=4.5,
        primary_target_R=1.2,
        remaining_path_R=remaining_path_R,
        deadline=_clock(240),
        setup_id="episode:1",
        entry_location_id="location:1",
        entry_path_id="path:1",
        entry_zone_lower=99.0,
        entry_zone_upper=100.0,
        selected_draw_id="draw:1",
    )


def _lsr_plan(
    selected_at: pd.Timestamp,
    *,
    custody_at: pd.Timestamp,
    target_id: str = "draw:1",
    target_price: float = 105.0,
    route_id: str = "route:1",
    remaining_path_R: float = 1.2,
) -> TradePlan:
    swept_at = custody_at - pd.Timedelta(minutes=30)
    confirmed_at = custody_at - pd.Timedelta(minutes=29)
    target = LiquidityLevel(
        level_id=target_id,
        timeframe=Timeframe.M1,
        side="above",
        price=target_price,
        formed_at=swept_at,
        confirmed_at=confirmed_at,
        touches=0,
    )
    invalidation = StructuralLevel(
        price=95.0,
        side="below",
        source_level_id="root:1",
        observed_at=swept_at,
        rationale="exact original formed-pool sweep extreme",
    )
    return TradePlan(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        direction=Direction.LONG,
        planned_entry=99.5,
        invalidation=invalidation,
        targets=(target,),
        risk_points=4.5,
        primary_target_R=(target_price - 99.5) / 4.5,
        remaining_path_R=remaining_path_R,
        deadline=_clock(240),
        setup_id="episode:1",
        entry_location_id="location:1",
        entry_path_id="path:1",
        entry_zone_lower=99.0,
        entry_zone_upper=100.0,
        selected_draw_id=target_id,
        draw_selection=DrawSelection(
            draw_id=target_id,
            selected_at=selected_at,
            selection_reason="liquidity_sweep_reversal:provisional",
            source_timeframe=Timeframe.M1,
            source_kind="swing",
            side="above",
            price=target_price,
            source_confirmed_at=confirmed_at,
            strength=0.5,
        ),
        lsr_context=FrozenLSRContext(
            manipulation_id="root:1",
            manipulation_protocol_hash="group4:v1",
            source_pool_id="pool:1",
            pool_path_id="pool-path:1",
            pool_path_protocol_hash="group5:v1",
            displacement_id="displacement:1",
            direction=Direction.LONG,
            swept_at=swept_at,
            reaccepted_at=custody_at - pd.Timedelta(minutes=20),
            displacement_active_at=custody_at - pd.Timedelta(minutes=10),
            displacement_observed_at=custody_at - pd.Timedelta(minutes=9),
            sweep_extreme=95.0,
        ),
        liquidity_route=LiquidityRoute(
            route_id=route_id,
            selected_at=selected_at,
            context_draw_id="context-draw:1",
            intermediate_liquidity_ids=(),
            primary_deliverable_target_id=target_id,
            terminal_draw_id="context-draw:1",
            source_path_ids=("context-draw:1", target_id),
        ),
    )
def _snapshot(
    asof: pd.Timestamp,
    context: ContextThesisState,
    episode: EntryEpisodeState,
    *,
    locations: tuple[EntryLocationState, ...] = (),
    paths: tuple[PathSequenceState, ...] = (),
    scene_added: tuple[str, ...] = (),
    scene_revised: tuple[str, ...] = (),
    scene_added_edges: tuple[str, ...] = (),
    scene_revised_edges: tuple[str, ...] = (),
    scene_resolutions: tuple[str, ...] = (),
) -> SimpleNamespace:
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=asof,
            bars=index + 1,
            metrics={},
        )
        for index, timeframe in enumerate(
            (Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5, Timeframe.M1)
        )
    }
    observation = SimpleNamespace(
        asof=asof,
        symbol="NQH5",
        instrument_id=1,
        price=100.5,
        frames=frames,
        anomalies=(),
        typed_transition_delta_available=True,
        interaction_update=InteractionUpdate(
            zone_interactions=locations,
            reacceptance_interactions=(),
            micro_break_facts=(),
            interaction_paths=paths,
        ),
        scene_revision_id=f"scene:{asof.isoformat()}",
        scene_added_node_ids=scene_added,
        scene_revised_node_ids=scene_revised,
        scene_added_edge_ids=scene_added_edges,
        scene_revised_edge_ids=scene_revised_edges,
        scene_resolution_event_ids=scene_resolutions,
    )
    belief = SimpleNamespace(
        global_context=None,
        context_theses={context.context_thesis_id: context},
        entry_episodes={episode.candidate_id: episode},
        thesis_candidates={},
        retained_episode_candidates={},
        position_management_candidates={},
        hypotheses={},
        dominant_hypothesis_id=None,
        competing_hypothesis_ids=(),
        unresolved_ambiguities=(),
    )
    decision = SimpleNamespace(
        selected_action=SimpleNamespace(value="abstain"),
        best_hypothesis_key=None,
        advantage=0.0,
        reasons=("test",),
        plan=None,
    )
    risk = SimpleNamespace(
        final_action=SimpleNamespace(value="abstain"),
        passed=False,
        vetoes=(),
        reasons=("test",),
    )
    return SimpleNamespace(
        observation=observation,
        belief=belief,
        decision=decision,
        risk=risk,
    )


def _recorder(
    tmp_path: Path,
    *,
    source_sha256: str = "a" * 64,
) -> CausalCaseRecorder:
    return CausalCaseRecorder(
        source_path=tmp_path / "canonical.parquet",
        source_sha256=source_sha256,
        source_role="causal_previous_session_front",
        split_role="development",
        capture_start=_clock(),
        model_versions={"brain": 11, "case": 1},
    )


def _scene_graph(
    asof: pd.Timestamp,
    relations: dict[str, str],
) -> SimpleNamespace:
    source_node = SimpleNamespace(
        node_id="node:source",
        kind="displacement",
        liquidity_role=None,
        semantic_attributes=(("role", "impulse"),),
        timeframe="5m",
        structural_scale=StructuralScale.INTERNAL,
        lifecycle="active",
    )
    target_node = SimpleNamespace(
        node_id="node:target",
        kind="entry_location",
        liquidity_role=None,
        semantic_attributes=(),
        timeframe="1m",
        structural_scale=StructuralScale.INTERNAL,
        lifecycle="active",
    )
    edges = {
        edge_id: SimpleNamespace(
            edge_id=edge_id,
            source_node_id=source_node.node_id,
            target_node_id=target_node.node_id,
            relation=SceneEdgeKind(relation),
            lifecycle="active",
            observed_at=asof,
        )
        for edge_id, relation in relations.items()
    }
    return SimpleNamespace(
        last_asof=asof,
        revision_id=f"scene:{asof.isoformat()}",
        _edges=edges,
        _current_path_block_edges={},
        _nodes={
            source_node.node_id: source_node,
            target_node.node_id: target_node,
        },
    )


def _scene_case_row(
    tmp_path: Path,
    *,
    relations: dict[str, str],
    added_nodes: tuple[str, ...],
    revised_nodes: tuple[str, ...],
    added_edges: tuple[str, ...],
    revised_edges: tuple[str, ...],
    resolutions: tuple[str, ...],
) -> dict[str, object]:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            _episode(t0),
            scene_added=added_nodes,
            scene_revised=revised_nodes,
            scene_added_edges=added_edges,
            scene_revised_edges=revised_edges,
            scene_resolutions=resolutions,
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
        replay_update_ordinal=0,
        scene_graph=_scene_graph(t0, relations),
    )
    return recorder.drain_input_rows()[0].to_dict()


def _admitted_dfp_case(
    tmp_path: Path,
) -> tuple[CausalCaseRecorder, list[dict[str, object]]]:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            _episode(
                t0,
                playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
                location_id="location:1",
                path_id="path:1",
            ),
            locations=(
                _location(t0, formed_at=t0 - pd.Timedelta(minutes=5)),
            ),
            paths=(
                _path(t0, formed_at=t0 - pd.Timedelta(minutes=5)),
            ),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    return recorder, [item.to_dict() for item in recorder.drain_input_rows()]


def _shadow_payload(
    *,
    candidate_id: str = "shadow:1",
    event_kind: str = "open_market_thesis_revision",
    observed_at: pd.Timestamp | None = None,
    resolved_at: pd.Timestamp | None = None,
    resolution: str = "target_first",
    filled: bool = True,
    censored: bool = False,
    target_first: bool | None = True,
    invalidation_first: bool | None = False,
    same_bar_collision: bool = False,
) -> dict[str, object]:
    observed = _clock() if observed_at is None else observed_at
    resolved = _clock(1) if resolved_at is None else resolved_at
    return {
        "candidate_id": candidate_id,
        "event_kind": event_kind,
        "source_episode_id": "episode:1",
        "source_context_thesis_id": "context:1",
        "entry_location_id": "location:1",
        "entry_path_id": "path:1",
        "entry_episode_binding_status": "exact_entry_location",
        "observed_at": observed,
        "resolved_at": resolved,
        "target_before_invalidation": target_first,
        "invalidation_before_target": invalidation_first,
        "same_bar_collision": same_bar_collision,
        "draw_id": "draw:1",
        "time_to_draw_real_bars": 1 if target_first is True else None,
        "filled": filled,
        "resolution": resolution,
        "censored": censored,
    }


def test_causal_outcome_selection_contract_versions_are_explicit() -> None:
    assert CAUSAL_CASE_RECORDER_SCHEMA_VERSION == 8
    assert CAUSAL_CASE_PROTOCOL_VERSION == "entry-episode-causal-case-1.7.0"
    assert CAUSAL_CASE_PROTOCOL["shadow_outcome_selection"].endswith(
        "never_resolved_at_resolution_or_outcome"
    )
    assert CAUSAL_CASE_PROTOCOL["interaction_update_schema_version"] == 1
    assert CAUSAL_CASE_PROTOCOL["interaction_authority"] == (
        "raw_eye_physical_facts_only_no_brain_interpretation"
    )
    assert CAUSAL_CASE_PROTOCOL["interaction_collections"] == list(
        INTERACTION_ARTIFACT_COLLECTION_NAMES
    )


def test_sparse_episode_revisions_do_not_emit_heartbeats_and_keep_prefix_bounds(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    context = _context(t0)
    episode = _episode(t0)
    recorder.observe(_snapshot(t0, context, episode), source_bar=_bar(t0), source_row_ordinal=10)
    first = recorder.drain_input_rows()
    assert [item.revision_stage for item in first] == [
        "context_formed",
        "episode_created",
    ]
    assert {item.observable_regime for item in first} == {"sweep_failure"}
    first_representation = representation_case_from_case_input_row(
        first[-1].to_dict()
    )
    assert first_representation.regime == "sweep_failure"

    # A later snapshot with no evidence revision emits nothing.
    t1 = _clock(1)
    recorder.observe(
        _snapshot(t1, context, episode, scene_added=("event:between-revisions",)),
        source_bar=_bar(t1),
        source_row_ordinal=11,
    )
    assert recorder.drain_input_rows() == ()

    changed = replace(
        context,
        supporting_event_ids=("authority:1", "support:2"),
        updated_at=t1 + pd.Timedelta(minutes=1),
    )
    t2 = _clock(2)
    recorder.observe(
        _snapshot(t2, changed, episode),
        source_bar=_bar(t2),
        source_row_ordinal=12,
    )
    changed_row = recorder.drain_input_rows()[0]
    assert changed_row.revision_stage == "context_changed"
    transition = json.loads(changed_row.observation_transition_json)
    assert transition["coverage"] == {
        "all_typed_deltas_available": True,
        "complete": True,
        "coverage_end_at": t2.isoformat(),
        "coverage_end_replay_update_ordinal": 2,
        "coverage_start_at": t0.isoformat(),
        "coverage_start_exclusive": True,
        "coverage_start_replay_update_ordinal": 0,
        "eventful_update_count": 1,
        "gap_free": True,
        "last_observed_update_at": t2.isoformat(),
        "last_observed_replay_update_ordinal": 2,
        "observed_update_count": 2,
    }
    assert [item["asof"] for item in transition["updates"]] == [t1.isoformat()]
    assert "event:between-revisions" in json.loads(
        changed_row.added_event_ids_json
    )
    scene_delta = json.loads(changed_row.scene_graph_delta_json)
    assert "event:between-revisions" in scene_delta["added_node_ids"]
    prefix = json.loads(changed_row.prefix_refs_json)[-1]
    assert prefix["replay_view_1m_row_start"] == 10
    assert prefix["replay_view_1m_row_end_exclusive"] == 13
    assert prefix["end_at"] == t2.isoformat()
    assert "never_use_frame_row_as_global_index" in prefix["reload_rule"]
    changed_representation = representation_case_from_case_input_row(
        changed_row.to_dict()
    )
    assert changed_representation.asof == t2
    assert (
        changed_representation.prefixes["1m"].end_at
        > first_representation.prefixes["1m"].end_at
    )
    assert changed_representation.prefixes["1m"].resolve_external_rows_by_time

    t3 = _clock(3)
    terminal = replace(
        episode,
        phase=PlaybookPhase.INVALIDATED,
        updated_at=t3,
        terminal_at=t3,
        terminal_reason="zone_failed",
    )
    recorder.observe(
        _snapshot(t3, changed, terminal),
        source_bar=_bar(t3),
        source_row_ordinal=13,
    )
    terminal_row = recorder.drain_input_rows()[0]
    assert terminal_row.revision_stage == "terminal"
    recorder.close_unresolved(t3)
    outcome = recorder.drain_outcome_rows()[0]
    assert outcome.case_id == terminal_row.case_id
    assert outcome.first_event == "episode_terminal"
    assert outcome.terminal_reason == "zone_failed"
    assert not set(CAUSAL_CASE_OUTCOME_FIELD_TYPES).issubset(
        CAUSAL_CASE_INPUT_FIELD_TYPES
    )


def test_current_interaction_views_stay_per_update_not_aggregate_history(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    context = _context(t0)
    episode = _episode(
        t0,
        location_id="location:1",
        path_id="path:1",
    )
    recorder.observe(
        _snapshot(
            t0,
            context,
            episode,
            locations=(_location(t0),),
            paths=(_path(t0),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    recorder.drain_input_rows()

    for minute in (1, 2):
        clock = _clock(minute)
        recorder.observe(
            _snapshot(
                clock,
                context,
                episode,
                locations=(_location(clock, formed_at=t0),),
                paths=(_path(clock, formed_at=t0),),
                scene_added=(f"scene-event:{minute}",),
            ),
            source_bar=_bar(clock),
            source_row_ordinal=minute,
        )
        assert recorder.drain_input_rows() == ()

    t3 = _clock(3)
    changed = replace(
        context,
        supporting_event_ids=("authority:1", "support:current-view"),
        updated_at=t3,
    )
    recorder.observe(
        _snapshot(
            t3,
            changed,
            episode,
            locations=(_location(t3, formed_at=t0),),
            paths=(_path(t3, formed_at=t0),),
        ),
        source_bar=_bar(t3),
        source_row_ordinal=3,
    )
    row = recorder.drain_input_rows()[0]
    transition = json.loads(row.observation_transition_json)
    base_names = {
        "liquidity_inventory_transitions_this_update",
        "liquidity_pool_transitions_this_update",
        "group3_fvg_transitions_this_update",
        "group3_order_block_transitions_this_update",
        "group4_range_transitions_this_update",
        "group4_manipulation_transitions_this_update",
    }
    assert set(transition["collections"]) == base_names | set(
        INTERACTION_DELTA_ARTIFACT_COLLECTION_NAMES
    )
    assert set(transition["collections"]).isdisjoint(
        INTERACTION_CURRENT_ARTIFACT_COLLECTION_NAMES
    )
    assert len(transition["updates"]) == 2
    for update in transition["updates"]:
        collections = update["collections"]
        assert set(collections) == base_names | set(
            INTERACTION_ARTIFACT_COLLECTION_NAMES
        )
        assert len(collections["interaction_zone_interactions"]) == 1
        assert len(collections["interaction_paths"]) == 1
    validate_case_input_row(row.to_dict())

    legacy = dict(row.to_dict())
    legacy_transition = json.loads(legacy["observation_transition_json"])
    legacy_transition["updates"][0]["collections"][
        "group5_micro_bos_transitions_this_update"
    ] = []
    legacy["observation_transition_json"] = _canonical_json(legacy_transition)
    legacy["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="collection schema"):
        validate_case_input_row(legacy)

    brain_nested = dict(row.to_dict())
    brain_transition = json.loads(
        brain_nested["observation_transition_json"]
    )
    brain_transition["updates"][0]["collections"][
        "interaction_zone_interactions"
    ][0]["brain_response"] = {"qualified": True}
    brain_nested["observation_transition_json"] = _canonical_json(
        brain_transition
    )
    brain_nested["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="interaction artifact"):
        validate_case_input_row(brain_nested)


def test_old_context_is_carried_at_episode_admission_without_forged_formation(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    context = _context(t0, formed_at=t0 - pd.Timedelta(hours=1))
    episode = _episode(t0)
    recorder.observe(_snapshot(t0, context, episode), source_bar=_bar(t0), source_row_ordinal=0)
    rows = recorder.drain_input_rows()
    assert [row.revision_stage for row in rows] == ["episode_created"]
    payload = json.loads(rows[0].context_thesis_json)
    assert payload["formed_at"] == (t0 - pd.Timedelta(hours=1)).isoformat()
    assert rows[0].stage_observed_at == t0


def test_long_eventless_interval_advances_complete_coverage_without_payload_growth(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    context = _context(t0)
    episode = _episode(t0)
    recorder.observe(
        _snapshot(t0, context, episode),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    recorder.drain_input_rows()
    for minute in range(1, 51):
        asof = _clock(minute)
        recorder.observe(
            _snapshot(asof, context, episode),
            source_bar=_bar(asof),
            source_row_ordinal=minute,
        )
        assert recorder.drain_input_rows() == ()
    case = next(iter(recorder._cases.values()))
    assert case.coverage_observed_update_count == 50
    assert case.coverage_last_observed_at == _clock(50)
    assert case.transition_updates == []

    t51 = _clock(51)
    changed = replace(
        context,
        supporting_event_ids=("authority:1", "event:material"),
        updated_at=t51,
    )
    recorder.observe(
        _snapshot(t51, changed, episode),
        source_bar=_bar(t51),
        source_row_ordinal=51,
    )
    row = recorder.drain_input_rows()[0]
    transition = json.loads(row.observation_transition_json)
    assert transition["coverage"]["complete"] is True
    assert transition["coverage"]["observed_update_count"] == 51
    assert transition["coverage"]["eventful_update_count"] == 0
    assert transition["coverage"]["last_observed_update_at"] == t51.isoformat()
    assert transition["updates"] == []

    with pytest.raises(ValueError, match="contiguous replay update"):
        recorder.observe(
            _snapshot(_clock(53), changed, episode),
            source_bar=_bar(_clock(53)),
            source_row_ordinal=53,
            replay_update_ordinal=53,
        )


def test_replay_update_coverage_is_separate_from_synthetic_source_rows(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    context = _context(t0)
    episode = _episode(t0)
    recorder.observe(
        _snapshot(t0, context, episode),
        source_bar=_bar(t0),
        source_row_ordinal=0,
        replay_update_ordinal=0,
    )
    inputs = [item.to_dict() for item in recorder.drain_input_rows()]

    t1 = _clock(1)
    recorder.observe(
        _snapshot(t1, context, episode),
        source_bar=_bar(t1, synthetic_no_trade=True),
        source_row_ordinal=1,
        replay_update_ordinal=1,
    )
    assert recorder.drain_input_rows() == ()
    recorder = pickle.loads(pickle.dumps(recorder))

    t2 = _clock(2)
    changed = replace(
        context,
        supporting_event_ids=("authority:1", "event:after-synthetic"),
        updated_at=t2,
    )
    recorder.observe(
        _snapshot(t2, changed, episode),
        source_bar=_bar(t2),
        # The synthetic update did not consume a canonical source row.
        source_row_ordinal=1,
        replay_update_ordinal=2,
    )
    changed_row = recorder.drain_input_rows()[0].to_dict()
    inputs.append(changed_row)
    coverage = json.loads(changed_row["observation_transition_json"])["coverage"]
    assert coverage["coverage_start_replay_update_ordinal"] == 0
    assert coverage["coverage_end_replay_update_ordinal"] == 2
    assert coverage["observed_update_count"] == 2
    assert coverage["complete"] is True
    prefix = json.loads(changed_row["prefix_refs_json"])[-1]
    assert prefix["replay_view_1m_row_start"] == 0
    assert prefix["replay_view_1m_row_end_exclusive"] == 2

    recorder.close_unresolved(t2)
    outcomes = [item.to_dict() for item in recorder.drain_outcome_rows()]
    validate_case_library_rows(inputs, outcomes)
    damaged = dict(changed_row)
    damaged_transition = json.loads(damaged["observation_transition_json"])
    damaged_transition["coverage"]["observed_update_count"] = 1
    damaged["observation_transition_json"] = _canonical_json(damaged_transition)
    damaged_scene = json.loads(damaged["scene_graph_delta_json"])
    damaged_scene["coverage"]["observed_update_count"] = 1
    damaged["scene_graph_delta_json"] = _canonical_json(damaged_scene)
    damaged["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="replay coverage is incomplete"):
        validate_case_library_rows([*inputs[:-1], damaged], outcomes)


def test_scene_delta_carries_typed_relation_topology_without_prices(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    source_node = SimpleNamespace(
        node_id="node:source",
        kind="displacement",
        liquidity_role=None,
        semantic_attributes=(("role", "impulse"),),
        timeframe="5m",
        structural_scale=StructuralScale.INTERNAL,
        lifecycle="active",
    )
    target_node = SimpleNamespace(
        node_id="node:target",
        kind="entry_location",
        liquidity_role=None,
        semantic_attributes=(),
        timeframe="1m",
        structural_scale=StructuralScale.INTERNAL,
        lifecycle="active",
    )
    edge = SimpleNamespace(
        edge_id="edge:creates",
        source_node_id=source_node.node_id,
        target_node_id=target_node.node_id,
        relation=SceneEdgeKind.CREATES,
        lifecycle="active",
        observed_at=t0,
    )
    graph = SimpleNamespace(
        last_asof=t0,
        revision_id=f"scene:{t0.isoformat()}",
        _edges={edge.edge_id: edge},
        _current_path_block_edges={},
        _nodes={source_node.node_id: source_node, target_node.node_id: target_node},
    )
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            _episode(t0),
            scene_added_edges=(edge.edge_id,),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
        replay_update_ordinal=0,
        scene_graph=graph,
    )
    row = recorder.drain_input_rows()[0].to_dict()
    scene = json.loads(row["scene_graph_delta_json"])
    assert scene["relation_descriptors_complete"] is True
    descriptor = scene["relation_descriptors"][0]
    assert descriptor["relation"] == SceneEdgeKind.CREATES.value
    assert descriptor["source"]["kind"] == "displacement"
    assert descriptor["source"]["role"] == "displacement"
    assert descriptor["target"]["role"] == "entry_location"
    assert "price" not in json.dumps(descriptor)
    representation = case_input_to_representation_mapping(row)
    assert representation["graph"]["relation_descriptors"] == [descriptor]


def test_scene_case_projection_is_byte_stable_under_set_permutations(
    tmp_path: Path,
) -> None:
    relations = {
        "edge:z-added": SceneEdgeKind.CREATES.value,
        "edge:m-added": SceneEdgeKind.SOURCED_FROM.value,
        "edge:b-revised": SceneEdgeKind.BLOCKS_PATH_TO.value,
        "edge:a-revised": SceneEdgeKind.LOCATED_AT.value,
    }
    left = _scene_case_row(
        tmp_path,
        relations=relations,
        added_nodes=("node:z", "node:a", "node:z"),
        revised_nodes=("node:y", "node:b"),
        added_edges=("edge:z-added", "edge:m-added", "edge:z-added"),
        revised_edges=("edge:b-revised", "edge:a-revised"),
        resolutions=("resolution:z", "resolution:a", "resolution:z"),
    )
    right = _scene_case_row(
        tmp_path,
        relations=dict(reversed(tuple(relations.items()))),
        added_nodes=("node:a", "node:z"),
        revised_nodes=("node:b", "node:y", "node:b"),
        added_edges=("edge:m-added", "edge:z-added"),
        revised_edges=("edge:a-revised", "edge:b-revised", "edge:a-revised"),
        resolutions=("resolution:a", "resolution:z"),
    )

    for field_name in (
        "scene_graph_delta_json",
        "added_event_ids_json",
        "invalidated_event_ids_json",
        "input_fingerprint",
        "revision_id",
    ):
        assert left[field_name] == right[field_name]
    scene = json.loads(str(left["scene_graph_delta_json"]))
    update = scene["updates"][0]
    assert [item["edge_id"] for item in update["relation_descriptors"]] == [
        "edge:m-added",
        "edge:z-added",
        "edge:a-revised",
        "edge:b-revised",
    ]
    for field_name in (
        "added_node_ids",
        "revised_node_ids",
        "added_edge_ids",
        "revised_edge_ids",
        "resolution_event_ids",
    ):
        assert update[field_name] == sorted(set(update[field_name]))
        assert scene[field_name] == sorted(set(scene[field_name]))
    assert json.loads(str(left["added_event_ids_json"])) == sorted(
        set(json.loads(str(left["added_event_ids_json"])))
    )
    assert json.loads(str(left["invalidated_event_ids_json"])) == sorted(
        set(json.loads(str(left["invalidated_event_ids_json"])))
    )


def test_scene_case_projection_changes_for_real_descriptor_content(
    tmp_path: Path,
) -> None:
    common = {
        "added_nodes": ("node:a",),
        "revised_nodes": (),
        "added_edges": ("edge:a",),
        "revised_edges": (),
        "resolutions": (),
    }
    sourced = _scene_case_row(
        tmp_path,
        relations={"edge:a": SceneEdgeKind.SOURCED_FROM.value},
        **common,
    )
    blocks = _scene_case_row(
        tmp_path,
        relations={"edge:a": SceneEdgeKind.BLOCKS_PATH_TO.value},
        **common,
    )

    assert sourced["scene_graph_delta_json"] != blocks["scene_graph_delta_json"]
    assert sourced["input_fingerprint"] != blocks["input_fingerprint"]
    assert sourced["revision_id"] != blocks["revision_id"]


def test_scene_case_rejects_same_edge_as_added_and_revised(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    with pytest.raises(ValueError, match="both added and revised"):
        recorder.observe(
            _snapshot(
                t0,
                _context(t0),
                _episode(t0),
                scene_added_edges=("edge:both",),
                scene_revised_edges=("edge:both",),
            ),
            source_bar=_bar(t0),
            source_row_ordinal=0,
            scene_graph=_scene_graph(
                t0,
                {"edge:both": SceneEdgeKind.SOURCED_FROM.value},
            ),
        )


def test_scene_case_canonicalization_survives_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    context = _context(t0)
    episode = _episode(t0)
    recorder.observe(
        _snapshot(t0, context, episode),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    recorder.drain_input_rows()

    t1 = _clock(1)
    recorder.observe(
        _snapshot(
            t1,
            context,
            episode,
            scene_added=("node:z", "node:a", "node:z"),
            scene_added_edges=("edge:z", "edge:a", "edge:z"),
        ),
        source_bar=_bar(t1),
        source_row_ordinal=1,
        scene_graph=_scene_graph(
            t1,
            {
                "edge:z": SceneEdgeKind.SOURCED_FROM.value,
                "edge:a": SceneEdgeKind.CREATES.value,
            },
        ),
    )
    assert recorder.drain_input_rows() == ()
    resumed = pickle.loads(pickle.dumps(recorder))

    t2 = _clock(2)
    for candidate in (recorder, resumed):
        candidate.observe(
            _snapshot(
                t2,
                context,
                episode,
                scene_revised_edges=("edge:a",),
            ),
            source_bar=_bar(t2),
            source_row_ordinal=2,
            scene_graph=_scene_graph(
                t2,
                {"edge:a": SceneEdgeKind.BLOCKS_PATH_TO.value},
            ),
        )
        assert candidate.drain_input_rows() == ()

    t3 = _clock(3)
    changed = replace(
        context,
        supporting_event_ids=("authority:1", "support:checkpoint"),
        updated_at=t3,
    )
    rows = []
    for candidate in (recorder, resumed):
        candidate.observe(
            _snapshot(t3, changed, episode),
            source_bar=_bar(t3),
            source_row_ordinal=3,
        )
        rows.append(candidate.drain_input_rows()[0].to_dict())

    assert rows[0] == rows[1]
    scene = json.loads(str(rows[0]["scene_graph_delta_json"]))
    assert [update["asof"] for update in scene["updates"]] == [
        t1.isoformat(),
        t2.isoformat(),
    ]
    assert scene["added_edge_ids"] == ["edge:a", "edge:z"]
    assert scene["revised_edge_ids"] == ["edge:a"]
    assert [item["relation"] for item in scene["relation_descriptors"]] == [
        SceneEdgeKind.CREATES.value,
        SceneEdgeKind.SOURCED_FROM.value,
        SceneEdgeKind.BLOCKS_PATH_TO.value,
    ]


def test_scene_case_validator_rejects_noncanonical_and_ambiguous_rows(
    tmp_path: Path,
) -> None:
    row = _scene_case_row(
        tmp_path,
        relations={
            "edge:z": SceneEdgeKind.SOURCED_FROM.value,
            "edge:a": SceneEdgeKind.CREATES.value,
        },
        added_nodes=("node:a", "node:z"),
        revised_nodes=(),
        added_edges=("edge:a", "edge:z"),
        revised_edges=(),
        resolutions=(),
    )

    noncanonical = dict(row)
    noncanonical["scene_graph_delta_json"] = json.dumps(
        json.loads(str(noncanonical["scene_graph_delta_json"]))
    )
    noncanonical["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="not canonical JSON"):
        validate_case_input_row(noncanonical)

    nonfinite = dict(row)
    nonfinite["scene_graph_delta_json"] = "NaN"
    nonfinite["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="invalid JSON"):
        validate_case_input_row(nonfinite)

    unordered = dict(row)
    scene = json.loads(str(unordered["scene_graph_delta_json"]))
    scene["added_node_ids"].reverse()
    scene["updates"][0]["added_node_ids"].reverse()
    unordered["scene_graph_delta_json"] = _canonical_json(scene)
    unordered["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="not sorted unique"):
        validate_case_input_row(unordered)

    duplicated_descriptor = dict(row)
    scene = json.loads(str(duplicated_descriptor["scene_graph_delta_json"]))
    duplicate = dict(scene["updates"][0]["relation_descriptors"][0])
    scene["updates"][0]["relation_descriptors"].append(duplicate)
    scene["relation_descriptors"].append(duplicate)
    duplicated_descriptor["scene_graph_delta_json"] = _canonical_json(scene)
    duplicated_descriptor["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="descriptor is duplicated"):
        validate_case_input_row(duplicated_descriptor)

    overlapping_edge = dict(row)
    scene = json.loads(str(overlapping_edge["scene_graph_delta_json"]))
    scene["updates"][0]["revised_edge_ids"] = ["edge:a"]
    scene["revised_edge_ids"] = ["edge:a"]
    overlapping_edge["scene_graph_delta_json"] = _canonical_json(scene)
    overlapping_edge["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="both added and revised"):
        validate_case_input_row(overlapping_edge)


@pytest.mark.parametrize(
    ("pullback_offset", "trigger_offset", "reason"),
    [
        (-2, None, "late_admission_with_prior_first_pullback"),
        (-2, -1, "late_admission_with_prior_first_pullback"),
    ],
)
def test_late_episode_admission_is_skipped_by_recorder_owned_clock(
    tmp_path: Path,
    pullback_offset: int,
    trigger_offset: int | None,
    reason: str,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    pullback = t0 + pd.Timedelta(minutes=pullback_offset)
    trigger = None
    if trigger_offset is not None:
        trigger = FrozenTriggerState(
            trigger_id="trigger:1",
            trigger_kind="wick_rejection",
            observed_at=t0 + pd.Timedelta(minutes=trigger_offset),
            setup_id="episode:1",
            entry_path_id="path:1",
            entry_location_id="location:1",
            direction=Direction.LONG,
            source_entity_id="location:1",
            available_trigger_kinds=("wick_rejection",),
        )
    episode = _episode(
        t0,
        location_id="location:1",
        path_id="path:1",
        first_pullback_at=pullback,
        trigger=trigger,
    )
    recorder.observe(
        _snapshot(t0, _context(t0), episode),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    assert recorder.drain_input_rows() == ()
    assert recorder.summary["skipped_quality"] == {reason: 1}


def test_old_physical_zone_cannot_be_claimed_by_new_episode(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    episode = _episode(t0, location_id="location:1", path_id="path:1")
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            episode,
            locations=(_location(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
            paths=(_path(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    assert recorder.drain_input_rows() == ()
    assert recorder.summary["skipped_quality"] == {
        "late_admission_with_prior_zone_formation": 1
    }


@pytest.mark.parametrize(
    ("context", "location", "reason"),
    [
        (
            _context(_clock(), authority=("other-root", "displacement:1")),
            _location(_clock()),
            "admission_lsr_root_not_owned_by_context",
        ),
        (
            _context(_clock(), authority=("root:1", "other-displacement")),
            _location(_clock()),
            "admission_lsr_displacement_not_owned_by_context",
        ),
    ],
)
def test_lsr_admission_requires_exact_context_root_and_displacement_ownership(
    tmp_path: Path,
    context: ContextThesisState,
    location: EntryLocationState,
    reason: str,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    episode = _episode(t0, location_id="location:1", path_id="path:1")
    recorder.observe(
        _snapshot(t0, context, episode, locations=(location,), paths=(_path(t0),)),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    assert recorder.drain_input_rows() == ()
    assert recorder.summary["skipped_quality"] == {reason: 1}


def test_dfp_may_causally_admit_a_preexisting_zone(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    episode = _episode(
        t0,
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        location_id="location:1",
        path_id="path:1",
    )
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            episode,
            locations=(_location(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
            paths=(_path(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    rows = recorder.drain_input_rows()
    stages = [item.revision_stage for item in rows]
    assert stages == ["context_formed", "episode_created", "zone_registered"]
    assert {item.observable_regime for item in rows} == {"continuation"}
    assert recorder.summary["skipped_quality"] == {}


def test_plan_heartbeat_and_set_order_do_not_create_revisions(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    context = _context(
        t0,
        supporting=("authority:1", "support:a", "support:b"),
    )
    plan = _plan(t0)
    episode = _episode(
        t0,
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        location_id="location:1",
        path_id="path:1",
        plan=plan,
    )
    recorder.observe(
        _snapshot(
            t0,
            context,
            episode,
            locations=(_location(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
            paths=(_path(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    initial = recorder.drain_input_rows()
    assert [item.revision_stage for item in initial].count("plan_formed") == 1

    t1 = _clock(1)
    reordered_context = replace(
        context,
        supporting_event_ids=("support:b", "authority:1", "support:a"),
        updated_at=t1,
    )
    replacement_target = replace(
        plan.targets[0],
        level_id="draw:2",
        price=106.0,
    )
    dynamic_plan_view = replace(
        plan,
        targets=(replacement_target,),
        selected_draw_id="draw:2",
        primary_target_R=1.4,
        remaining_path_R=0.8,
    )
    dynamic_episode_view = replace(
        episode,
        plan=dynamic_plan_view,
        updated_at=t1,
    )
    recorder.observe(
        _snapshot(
            t1,
            reordered_context,
            dynamic_episode_view,
            locations=(_location(t1, formed_at=t0 - pd.Timedelta(minutes=5)),),
            paths=(_path(t1, formed_at=t0 - pd.Timedelta(minutes=5)),),
        ),
        source_bar=_bar(t1),
        source_row_ordinal=1,
    )
    assert recorder.drain_input_rows() == ()


def test_lsr_pre_owner_target_draw_route_re_evaluation_is_not_a_new_revision(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    plan = _lsr_plan(t0, custody_at=t0)
    episode = _episode(
        t0,
        location_id="location:1",
        path_id="path:1",
        plan=plan,
        phase=PlaybookPhase.WAITING_LOCATION,
    )
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            episode,
            locations=(_location(t0),),
            paths=(_path(t0),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    initial = recorder.drain_input_rows()
    assert [row.revision_stage for row in initial].count("plan_formed") == 1

    # Exercise the checkpointed object graph before the legal re-evaluation.
    recorder = pickle.loads(pickle.dumps(recorder))
    t1 = _clock(1)
    revised_plan = _lsr_plan(
        t1,
        custody_at=t0,
        target_id="draw:2",
        target_price=106.0,
        route_id="route:2",
        remaining_path_R=0.6,
    )
    recorder.observe(
        _snapshot(
            t1,
            _context(t0),
            replace(episode, plan=revised_plan, updated_at=t1),
            locations=(_location(t1, formed_at=t0),),
            paths=(_path(t1, formed_at=t0),),
        ),
        source_bar=_bar(t1),
        source_row_ordinal=1,
    )
    assert recorder.drain_input_rows() == ()
    case = next(iter(recorder._cases.values()))
    assert case.frozen_execution_plan_identity is None
    assert case.frozen_execution_lock_at is None


@pytest.mark.parametrize(
    "mutation",
    (
        "entry",
        "stop",
        "deadline",
        "setup",
        "zone",
        "path",
        "lsr_context",
    ),
)
def test_plan_formed_stable_custody_mutation_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    plan = _lsr_plan(t0, custody_at=t0)
    episode = _episode(
        t0,
        location_id="location:1",
        path_id="path:1",
        plan=plan,
        phase=PlaybookPhase.WAITING_LOCATION,
    )
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            episode,
            locations=(_location(t0),),
            paths=(_path(t0),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    recorder.drain_input_rows()

    t1 = _clock(1)
    if mutation == "entry":
        changed_plan = replace(plan, planned_entry=99.75, risk_points=4.75)
        changed_episode = replace(episode, plan=changed_plan, updated_at=t1)
        error = "immutable episode stage changed identity: plan_formed"
    elif mutation == "stop":
        changed_stop = replace(plan.invalidation, price=94.5)
        changed_plan = replace(plan)
        object.__setattr__(changed_plan, "invalidation", changed_stop)
        object.__setattr__(changed_plan, "risk_points", 5.0)
        changed_episode = replace(episode, updated_at=t1)
        object.__setattr__(changed_episode, "plan", changed_plan)
        object.__setattr__(changed_episode, "invalidation", changed_stop)
        error = "immutable episode stage changed identity: plan_formed"
    elif mutation == "deadline":
        changed_plan = replace(plan, deadline=_clock(239))
        changed_episode = replace(episode, plan=changed_plan, updated_at=t1)
        error = "immutable episode stage changed identity: plan_formed"
    elif mutation == "lsr_context":
        changed_context = replace(plan.lsr_context, source_pool_id="pool:2")
        changed_plan = replace(plan, lsr_context=changed_context)
        changed_episode = replace(episode, plan=changed_plan, updated_at=t1)
        error = "immutable episode stage changed identity: plan_formed"
    else:
        field = {
            "setup": "setup_id",
            "zone": "entry_location_id",
            "path": "entry_path_id",
        }[mutation]
        changed_plan = replace(plan, **{field: f"changed:{mutation}"})
        changed_episode = replace(episode, updated_at=t1)
        # Corrupt the otherwise typed view to prove the recorder itself remains
        # fail-closed even if an upstream constructor invariant is bypassed.
        object.__setattr__(changed_episode, "plan", changed_plan)
        error = (
            "frozen zone"
            if mutation == "zone"
            else "frozen path"
            if mutation == "path"
            else "immutable episode stage changed identity: plan_formed"
        )
        if mutation in {"zone", "path"}:
            object.__setattr__(
                changed_episode,
                "entry_location_id" if mutation == "zone" else "entry_path_id",
                f"changed:{mutation}",
            )
    with pytest.raises(ValueError, match=error):
        recorder.observe(
            _snapshot(
                t1,
                _context(t0),
                changed_episode,
                locations=(_location(t1, formed_at=t0),),
                paths=(_path(t1, formed_at=t0),),
            ),
            source_bar=_bar(t1),
            source_row_ordinal=1,
        )


@pytest.mark.parametrize(
    "owner_mutation",
    (
        "plan",
        "trigger",
        "trigger_core",
        "terminal_absent_plan",
        "terminal_absent_trigger",
        "terminal_trigger",
    ),
)
def test_lsr_execution_owner_freezes_full_plan_and_trigger_across_checkpoint(
    tmp_path: Path,
    owner_mutation: str,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    pre_owner_plan = _lsr_plan(t0, custody_at=t0)
    episode = _episode(
        t0,
        location_id="location:1",
        path_id="path:1",
        plan=pre_owner_plan,
        phase=PlaybookPhase.WAITING_LOCATION,
    )
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            episode,
            locations=(_location(t0),),
            paths=(_path(t0),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    recorder.drain_input_rows()

    t1 = _clock(1)
    pullback_episode = replace(
        episode,
        first_pullback_at=t1,
        phase=PlaybookPhase.WAITING_TRIGGER,
        updated_at=t1,
    )
    recorder.observe(
        _snapshot(
            t1,
            _context(t0),
            pullback_episode,
            locations=(_location(t1, formed_at=t0),),
            paths=(_path(t1, formed_at=t0),),
        ),
        source_bar=_bar(t1),
        source_row_ordinal=1,
    )
    recorder.drain_input_rows()

    t2 = _clock(2)
    trigger = FrozenTriggerState(
        trigger_id="trigger:1",
        trigger_kind="wick_rejection",
        observed_at=t2,
        setup_id="episode:1",
        entry_path_id="path:1",
        entry_location_id="location:1",
        direction=Direction.LONG,
        source_entity_id="location:1",
        strength=0.7,
        available_trigger_kinds=("wick_rejection",),
    )
    owner_plan = _lsr_plan(
        t2,
        custody_at=t0,
        target_id="draw:2",
        target_price=106.0,
        route_id="route:2",
        remaining_path_R=0.8,
    )
    owner_episode = replace(
        pullback_episode,
        selected_trigger=trigger,
        plan=owner_plan,
        invalidation=owner_plan.invalidation,
        phase=PlaybookPhase.EXECUTABLE,
        updated_at=t2,
    )
    recorder.observe(
        _snapshot(
            t2,
            _context(t0),
            owner_episode,
            locations=(_location(t2, formed_at=t0),),
            paths=(_path(t2, formed_at=t0),),
        ),
        source_bar=_bar(t2),
        source_row_ordinal=2,
    )
    assert [row.revision_stage for row in recorder.drain_input_rows()] == ["trigger"]
    case = next(iter(recorder._cases.values()))
    assert case.frozen_execution_plan_identity is not None
    assert case.frozen_execution_trigger_identity is not None
    assert case.frozen_execution_lock_at == t2

    recorder = pickle.loads(pickle.dumps(recorder))
    case = next(iter(recorder._cases.values()))
    frozen_identity = case.frozen_execution_plan_identity
    frozen_trigger_identity = case.frozen_execution_trigger_identity
    assert frozen_identity is not None
    assert frozen_trigger_identity is not None
    assert case.frozen_execution_lock_at == t2

    t3 = _clock(3)
    live_r_only = replace(owner_plan, remaining_path_R=0.2)
    diagnostic_trigger = replace(
        trigger,
        available_trigger_kinds=(
            "wick_rejection",
            "micro_bos_confirmed",
        ),
    )
    weakened = replace(
        owner_episode,
        plan=live_r_only,
        selected_trigger=diagnostic_trigger,
        phase=PlaybookPhase.WEAKENING,
        updated_at=t3,
    )
    recorder.observe(
        _snapshot(
            t3,
            _context(t0),
            weakened,
            locations=(_location(t3, formed_at=t0),),
            paths=(_path(t3, formed_at=t0),),
        ),
        source_bar=_bar(t3),
        source_row_ordinal=3,
    )
    assert recorder.drain_input_rows() == ()
    weakened_case = next(iter(recorder._cases.values()))
    assert weakened_case.frozen_execution_plan_identity == frozen_identity
    assert weakened_case.observed_execution_trigger_kinds == (
        "wick_rejection",
        "micro_bos_confirmed",
    )

    t4 = _clock(4)
    if owner_mutation in {
        "terminal_absent_plan",
        "terminal_absent_trigger",
        "terminal_trigger",
    }:
        terminal_trigger = (
            diagnostic_trigger
            if owner_mutation == "terminal_absent_plan"
            else replace(diagnostic_trigger, strength=0.9)
        )
        terminal_episode = replace(
            weakened,
            plan=(
                live_r_only
                if owner_mutation == "terminal_absent_trigger"
                else None
            ),
            selected_trigger=(
                None
                if owner_mutation == "terminal_absent_trigger"
                else terminal_trigger
            ),
            phase=PlaybookPhase.INVALIDATED,
            terminal_at=t4,
            terminal_reason="entry_zone_failed",
            updated_at=t4,
        )
        if owner_mutation == "terminal_trigger":
            with pytest.raises(
                ValueError,
                match="frozen LSR execution trigger mutated",
            ):
                recorder.observe(
                    _snapshot(
                        t4,
                        _context(t0),
                        terminal_episode,
                        locations=(_location(t4, formed_at=t0),),
                        paths=(_path(t4, formed_at=t0),),
                    ),
                    source_bar=_bar(t4),
                    source_row_ordinal=4,
                )
            return
        recorder.observe(
            _snapshot(
                t4,
                _context(t0),
                terminal_episode,
                locations=(_location(t4, formed_at=t0),),
                paths=(_path(t4, formed_at=t0),),
            ),
            source_bar=_bar(t4),
            source_row_ordinal=4,
        )
        terminal_rows = recorder.drain_input_rows()
        assert [row.revision_stage for row in terminal_rows] == ["terminal"]
        terminal_case = next(iter(recorder._cases.values()))
        assert terminal_case.frozen_execution_plan_identity == frozen_identity
        assert (
            terminal_case.frozen_execution_trigger_identity
            == frozen_trigger_identity
        )
        assert terminal_case.frozen_execution_lock_at == t2
        assert terminal_case.observed_execution_trigger_kinds == (
            "wick_rejection",
            "micro_bos_confirmed",
        )
        return

    if owner_mutation == "plan":
        mutated_plan = _lsr_plan(
            t4,
            custody_at=t0,
            target_id="draw:3",
            target_price=107.0,
            route_id="route:3",
            remaining_path_R=0.2,
        )
        mutated_trigger = diagnostic_trigger
    elif owner_mutation == "trigger_core":
        mutated_plan = live_r_only
        mutated_trigger = replace(
            diagnostic_trigger,
            source_entity_id="changed:source-entity",
        )
    else:
        mutated_plan = live_r_only
        mutated_trigger = replace(diagnostic_trigger, strength=0.9)
    error = (
        "frozen LSR execution plan mutated"
        if owner_mutation == "plan"
        else "frozen LSR execution trigger mutated"
    )
    with pytest.raises(ValueError, match=error):
        recorder.observe(
            _snapshot(
                t4,
                _context(t0),
                replace(
                    weakened,
                    plan=mutated_plan,
                    selected_trigger=mutated_trigger,
                    updated_at=t4,
                ),
                locations=(_location(t4, formed_at=t0),),
                paths=(_path(t4, formed_at=t0),),
            ),
            source_bar=_bar(t4),
            source_row_ordinal=4,
        )


def test_shadow_future_is_joined_only_after_input_revision(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    episode = _episode(
        t0,
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        location_id="location:1",
        path_id="path:1",
    )
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            episode,
            locations=(_location(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
            paths=(_path(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    input_rows = [item.to_dict() for item in recorder.drain_input_rows()]
    recorder.consume_shadow_records(
        (
            {
                "candidate_id": "shadow:1",
                "event_kind": "playbook_executable",
                "source_episode_id": "episode:1",
                "source_context_thesis_id": "context:1",
                "entry_location_id": "location:1",
                "entry_path_id": "path:1",
                "entry_episode_binding_status": "exact_action_candidate",
                "observed_at": t0,
                "resolved_at": _clock(5),
                "target_before_invalidation": True,
                "invalidation_before_target": False,
                "same_bar_collision": False,
                "mfe_points": 4.0,
                "mae_points": 1.0,
                "mfe_R": 2.0,
                "mae_R": 0.5,
                "hit_0_5R": True,
                "hit_1R": True,
                "hit_2R": True,
                "draw_id": "draw:1",
                "time_to_draw_real_bars": 5,
                "filled": True,
                "resolution": "target_first",
                "censored": False,
            },
        )
    )
    recorder.close_unresolved(_clock(5))
    outcome_rows = [item.to_dict() for item in recorder.drain_outcome_rows()]
    validate_case_library_rows(input_rows, outcome_rows)
    assert outcome_rows[0]["first_event"] == "target"
    assert outcome_rows[0]["draw_delivered"] is True
    assert "first_event" not in input_rows[0]
    representation = case_input_to_representation_mapping(input_rows[0])
    assert representation["decision_at"] == t0
    assert representation["prefixes"]
    assert "first_event" not in representation


@pytest.mark.parametrize(
    (
        "resolution",
        "filled",
        "censored",
        "target_first",
        "invalidation_first",
        "same_bar_collision",
        "expected_first_event",
        "expected_expired",
    ),
    (
        (
            "draw_consumed_before_entry",
            False,
            True,
            None,
            None,
            False,
            "right_censored",
            False,
        ),
        (
            "invalidation_before_entry",
            False,
            True,
            None,
            None,
            False,
            "right_censored",
            False,
        ),
        (
            "entry_target_same_bar_order_unknown",
            True,
            True,
            None,
            None,
            False,
            "right_censored",
            False,
        ),
        (
            "deadline_inside_completed_bar_censored",
            False,
            True,
            None,
            None,
            False,
            "right_censored",
            True,
        ),
        (
            "activation_geometry_invalid",
            True,
            True,
            None,
            None,
            False,
            "right_censored",
            False,
        ),
        (
            "contract_boundary",
            False,
            True,
            None,
            None,
            False,
            "right_censored",
            False,
        ),
        (
            "window_right_censored",
            True,
            True,
            None,
            None,
            False,
            "right_censored",
            False,
        ),
        (
            "target_first",
            True,
            False,
            True,
            False,
            False,
            "target",
            False,
        ),
        (
            "invalidation_first",
            True,
            False,
            False,
            True,
            False,
            "invalidation",
            False,
        ),
        (
            "same_bar_invalidation_priority",
            True,
            False,
            False,
            True,
            True,
            "invalidation",
            False,
        ),
        (
            "deadline_no_delivery",
            True,
            False,
            None,
            None,
            False,
            "deadline",
            True,
        ),
        (
            "entry_unfilled_deadline",
            False,
            False,
            None,
            None,
            False,
            "deadline",
            True,
        ),
        (
            "source_identity_invalidated",
            False,
            False,
            None,
            None,
            False,
            "invalidation",
            False,
        ),
        (
            "geometry_incomplete",
            False,
            False,
            None,
            None,
            False,
            "unresolved",
            False,
        ),
    ),
)
def test_shadow_resolution_projects_one_strict_causal_outcome(
    tmp_path: Path,
    resolution: str,
    filled: bool,
    censored: bool,
    target_first: bool | None,
    invalidation_first: bool | None,
    same_bar_collision: bool,
    expected_first_event: str,
    expected_expired: bool,
) -> None:
    recorder, inputs = _admitted_dfp_case(tmp_path)
    recorder.consume_shadow_records(
        (
            _shadow_payload(
                resolution=resolution,
                filled=filled,
                censored=censored,
                target_first=target_first,
                invalidation_first=invalidation_first,
                same_bar_collision=same_bar_collision,
            ),
        )
    )
    recorder.close_unresolved(_clock(1))
    outcomes = [item.to_dict() for item in recorder.drain_outcome_rows()]

    validate_case_library_rows(inputs, outcomes)
    assert len(outcomes) == 1
    assert outcomes[0]["first_event"] == expected_first_event
    assert outcomes[0]["deadline_first"] is (
        expected_first_event == "deadline"
    )
    assert outcomes[0]["expired"] is expected_expired
    assert outcomes[0]["resolution"] == resolution


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {
                "filled": True,
                "target_before_invalidation": True,
                "invalidation_before_target": False,
            },
            "censored Shadow has resolved first-event flags",
        ),
        (
            {
                "filled": True,
                "target_before_invalidation": False,
                "invalidation_before_target": True,
                "same_bar_collision": True,
            },
            "censored Shadow has resolved first-event flags",
        ),
        (
            {
                "target_before_invalidation": "yes",
            },
            "target_before_invalidation is not nullable boolean",
        ),
        (
            {
                "censored": False,
                "filled": True,
                "target_before_invalidation": False,
                "invalidation_before_target": False,
            },
            "pairwise first-event flags are inconsistent",
        ),
        (
            {
                "censored": False,
                "filled": True,
                "resolution": "target_first",
                "target_before_invalidation": True,
                "invalidation_before_target": False,
                "same_bar_collision": True,
            },
            "resolution family disagrees with collision flag",
        ),
        (
            {
                "censored": False,
                "filled": True,
                "resolution": "same_bar_invalidation_priority",
                "target_before_invalidation": False,
                "invalidation_before_target": True,
                "same_bar_collision": False,
            },
            "resolution family disagrees with collision flag",
        ),
        (
            {
                "censored": False,
                "filled": True,
                "resolution": "deadline_no_delivery",
                "target_before_invalidation": True,
                "invalidation_before_target": False,
            },
            "resolution family disagrees with first-event flags",
        ),
        (
            {
                "censored": False,
                "filled": True,
                "resolution": "target_first",
                "target_before_invalidation": False,
                "invalidation_before_target": True,
            },
            "resolution family disagrees with first-event flags",
        ),
        (
            {
                "resolution": "target_first",
            },
            "censored Shadow uses a non-censored resolution family",
        ),
        (
            {
                "resolution": "future_unknown_censored_reason",
            },
            "censored resolution family is unsupported",
        ),
    ),
)
def test_shadow_censor_and_collision_flags_fail_closed(
    changes: dict[str, object],
    message: str,
) -> None:
    payload = _shadow_payload(
        resolution="draw_consumed_before_entry",
        filled=False,
        censored=True,
        target_first=None,
        invalidation_first=None,
    )
    payload.update(changes)

    with pytest.raises(ValueError, match=message):
        CausalCaseRecorder._first_event(payload)


@pytest.mark.parametrize(
    "resolution",
    ("deadline_no_delivery", "invalidation_first"),
)
def test_materialized_shadow_resolution_cannot_contradict_target_flags(
    tmp_path: Path,
    resolution: str,
) -> None:
    recorder, inputs = _admitted_dfp_case(tmp_path)
    recorder.consume_shadow_records((_shadow_payload(),))
    recorder.close_unresolved(_clock(1))
    outcome = recorder.drain_outcome_rows()[0].to_dict()
    validate_case_library_rows(inputs, [outcome])

    damaged = {
        **outcome,
        "resolution": resolution,
        "expired": resolution == "deadline_no_delivery",
    }
    with pytest.raises(ValueError, match="resolution family"):
        validate_case_library_rows(inputs, [damaged])


def _selected_shadow_outcome(
    tmp_path: Path,
    payloads: tuple[dict[str, object], ...],
    *,
    checkpoint_after: int | None = None,
) -> dict[str, object]:
    recorder, inputs = _admitted_dfp_case(tmp_path)
    for index, payload in enumerate(payloads, start=1):
        recorder.consume_shadow_records((payload,))
        if checkpoint_after == index:
            recorder = pickle.loads(pickle.dumps(recorder))
    recorder.close_unresolved(
        max(
            _clock(10),
            *(payload["resolved_at"] for payload in payloads),
        )
    )
    outcomes = [item.to_dict() for item in recorder.drain_outcome_rows()]
    validate_case_library_rows(inputs, outcomes)
    assert len(outcomes) == 1
    return outcomes[0]


def test_shadow_tie_break_is_observation_only_order_and_checkpoint_stable(
    tmp_path: Path,
) -> None:
    earlier_observed_later_resolved = _shadow_payload(
        candidate_id="shadow:earlier-observed",
        observed_at=_clock(),
        resolved_at=_clock(10),
        resolution="target_first",
        target_first=True,
        invalidation_first=False,
    )
    later_observed_earlier_resolved = _shadow_payload(
        candidate_id="shadow:later-observed",
        observed_at=_clock(1),
        resolved_at=_clock(2),
        resolution="invalidation_first",
        target_first=False,
        invalidation_first=True,
    )
    natural_resolution_order = (
        later_observed_earlier_resolved,
        earlier_observed_later_resolved,
    )
    reversed_order = tuple(reversed(natural_resolution_order))

    natural = _selected_shadow_outcome(
        tmp_path / "natural",
        natural_resolution_order,
    )
    reversed_outcome = _selected_shadow_outcome(
        tmp_path / "reversed",
        reversed_order,
    )
    resumed = _selected_shadow_outcome(
        tmp_path / "resumed",
        natural_resolution_order,
        checkpoint_after=1,
    )

    assert natural == reversed_outcome == resumed
    assert natural["source_shadow_candidate_id"] == "shadow:earlier-observed"
    assert natural["first_event"] == "target"


def test_shadow_tie_break_uses_candidate_id_at_one_observation_clock(
    tmp_path: Path,
) -> None:
    candidate_a = _shadow_payload(
        candidate_id="shadow:a",
        observed_at=_clock(),
        resolved_at=_clock(10),
        resolution="target_first",
        target_first=True,
        invalidation_first=False,
    )
    candidate_b = _shadow_payload(
        candidate_id="shadow:b",
        observed_at=_clock(),
        resolved_at=_clock(2),
        resolution="invalidation_first",
        target_first=False,
        invalidation_first=True,
    )

    forward = _selected_shadow_outcome(
        tmp_path / "forward",
        (candidate_a, candidate_b),
    )
    reverse = _selected_shadow_outcome(
        tmp_path / "reverse",
        (candidate_b, candidate_a),
    )

    assert forward == reverse
    assert forward["source_shadow_candidate_id"] == "shadow:a"
    assert forward["first_event"] == "target"


def test_recorder_checkpoint_resume_matches_uninterrupted_case_state(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    context = _context(t0)
    episode = _episode(t0)
    recorder.observe(
        _snapshot(t0, context, episode),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    initial = [item.to_dict() for item in recorder.drain_input_rows()]
    resumed = pickle.loads(pickle.dumps(recorder))

    t1 = _clock(1)
    changed = replace(
        context,
        supporting_event_ids=("authority:1", "support:later"),
        updated_at=t1,
    )
    terminal = replace(
        episode,
        phase=PlaybookPhase.INVALIDATED,
        updated_at=t1,
        terminal_at=t1,
        terminal_reason="context_invalidated",
    )
    for candidate in (recorder, resumed):
        candidate.observe(
            _snapshot(t1, changed, terminal),
            source_bar=_bar(t1),
            source_row_ordinal=1,
        )
        candidate.close_unresolved(t1)

    assert [item.to_dict() for item in resumed.drain_input_rows()] == [
        item.to_dict() for item in recorder.drain_input_rows()
    ]
    assert [item.to_dict() for item in resumed.drain_outcome_rows()] == [
        item.to_dict() for item in recorder.drain_outcome_rows()
    ]
    assert resumed.summary == recorder.summary
    assert len(initial) == 2


@pytest.mark.parametrize("late_stage", ["first_pullback", "trigger", "terminal"])
def test_first_seen_episode_milestone_cannot_be_backfilled_from_an_older_clock(
    tmp_path: Path,
    late_stage: str,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    t1 = _clock(1)
    t2 = _clock(2)
    initial = _episode(
        t0,
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        location_id="location:1" if late_stage == "trigger" else None,
        path_id="path:1" if late_stage == "trigger" else None,
        first_pullback_at=t0 if late_stage == "trigger" else None,
    )
    recorder.observe(
        _snapshot(t0, _context(t0), initial),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    recorder.drain_input_rows()
    recorder.observe(
        _snapshot(t1, _context(t0), initial),
        source_bar=_bar(t1),
        source_row_ordinal=1,
    )
    assert recorder.drain_input_rows() == ()

    if late_stage == "first_pullback":
        backfilled = replace(initial, first_pullback_at=t1, updated_at=t2)
    elif late_stage == "trigger":
        backfilled = replace(
            initial,
            selected_trigger=FrozenTriggerState(
                trigger_id="trigger:late",
                trigger_kind="wick_rejection",
                observed_at=t1,
                setup_id="episode:1",
                entry_path_id="path:1",
                entry_location_id="location:1",
                direction=Direction.LONG,
                source_entity_id="location:1",
                available_trigger_kinds=("wick_rejection",),
            ),
            updated_at=t2,
        )
    else:
        backfilled = replace(
            initial,
            phase=PlaybookPhase.INVALIDATED,
            terminal_at=t1,
            terminal_reason="late_terminal",
            updated_at=t2,
        )
    with pytest.raises(ValueError, match=f"late backfill: {late_stage}"):
        recorder.observe(
            _snapshot(t2, _context(t0), backfilled),
            source_bar=_bar(t2),
            source_row_ordinal=2,
        )


def test_epoch_reset_emits_terminal_revision_from_old_bounded_prefix(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    recorder.observe(
        _snapshot(t0, _context(t0), _episode(t0)),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    admitted = recorder.drain_input_rows()
    old_epoch_id = admitted[0].market_epoch_id

    t1 = _clock(1)
    next_context = replace(
        _context(t1),
        context_thesis_id="context:2",
        market_epoch_id="runtime-epoch:2",
    )
    next_episode = replace(
        _episode(t1),
        episode_id="episode:2",
        candidate_id="candidate:2",
        parent_context_thesis_id="context:2",
    )
    boundary = _snapshot(t1, next_context, next_episode)
    boundary.belief.entry_episodes = {}
    boundary.observation.anomalies = ("contract_change_history_reset",)
    recorder.observe(
        boundary,
        source_bar=_bar(t1),
        source_row_ordinal=1,
    )
    rows = recorder.drain_input_rows()
    assert len(rows) == 1
    terminal = rows[0]
    assert terminal.revision_stage == "terminal"
    assert terminal.market_epoch_id == old_epoch_id
    assert terminal.asof == t1
    assert terminal.observed_terminal_reason == "contract_change_history_reset"
    prefixes = json.loads(terminal.prefix_refs_json)
    assert {item["asof"] for item in prefixes} == {t1.isoformat()}
    assert {item["end_at"] for item in prefixes} == {t0.isoformat()}
    assert {
        (item["replay_view_1m_row_start"], item["replay_view_1m_row_end_exclusive"])
        for item in prefixes
    } == {(0, 1)}
    transition = json.loads(terminal.observation_transition_json)
    assert transition["case_boundary_terminal"]["observed_at"] == t1.isoformat()
    assert json.loads(terminal.context_thesis_json)["market_epoch_id"] == "epoch:1"


def test_external_epoch_namespace_is_stable_and_source_bound(tmp_path: Path) -> None:
    t0 = _clock()
    snapshot = _snapshot(t0, _context(t0), _episode(t0))
    outputs = []
    for name, source_hash in (
        ("same-a", "a" * 64),
        ("same-b", "a" * 64),
        ("other-source", "b" * 64),
    ):
        recorder = _recorder(tmp_path / name, source_sha256=source_hash)
        recorder.observe(
            snapshot,
            source_bar=_bar(t0),
            source_row_ordinal=0,
        )
        outputs.append(recorder.drain_input_rows()[0])
    assert outputs[0].market_epoch_id == outputs[1].market_epoch_id
    assert outputs[0].case_id == outputs[1].case_id
    assert outputs[0].market_epoch_id != outputs[2].market_epoch_id
    assert outputs[0].case_id != outputs[2].case_id
    assert outputs[0].market_epoch_id.startswith("market-epoch:")
    assert json.loads(outputs[0].context_thesis_json)["market_epoch_id"] == "epoch:1"


@pytest.mark.parametrize(
    ("field", "bad_value", "counter"),
    [
        ("entry_location_id", "location:other", "shadow_entry_custody_missing_or_mismatch"),
        ("entry_path_id", None, "shadow_entry_custody_missing_or_mismatch"),
        (
            "lsr_displacement_id",
            "displacement:other",
            "shadow_lsr_zone_custody_missing_or_mismatch",
        ),
        (
            "lsr_entry_zone_id",
            "zone:other",
            "shadow_lsr_zone_custody_missing_or_mismatch",
        ),
    ],
)
def test_shadow_outcome_cannot_borrow_another_zone_or_path(
    tmp_path: Path,
    field: str,
    bad_value: str | None,
    counter: str,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    episode = _episode(t0, location_id="location:1", path_id="path:1")
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            episode,
            locations=(_location(t0),),
            paths=(_path(t0),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    recorder.drain_input_rows()
    payload = {
        "candidate_id": "shadow:wrong-custody",
        "event_kind": "playbook_executable",
        "source_episode_id": "episode:1",
        "source_context_thesis_id": "context:1",
        "entry_location_id": "location:1",
        "entry_path_id": "path:1",
        "lsr_displacement_id": "displacement:1",
        "lsr_entry_zone_id": "zone:1",
        "entry_episode_binding_status": "exact_action_candidate",
        "observed_at": t0,
        "resolved_at": _clock(5),
        "target_before_invalidation": True,
        "invalidation_before_target": False,
        "resolution": "target_first",
        "filled": True,
        "censored": False,
    }
    payload[field] = bad_value
    recorder.consume_shadow_records((payload,))
    assert recorder.summary["skipped_quality"] == {counter: 1}
    recorder.close_unresolved(_clock(5))
    outcome = recorder.drain_outcome_rows()[0]
    assert outcome.first_event == "right_censored"
    assert outcome.source_shadow_candidate_id is None


def test_rejected_episode_shadow_rows_never_accumulate_pending_state(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    rejected = _episode(
        t0,
        first_pullback_at=t0 - pd.Timedelta(minutes=1),
    )
    recorder.observe(
        _snapshot(t0, _context(t0), rejected),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    assert recorder.drain_input_rows() == ()
    recorder.consume_shadow_records(
        {"source_episode_id": "episode:1"} for _ in range(50)
    )
    assert recorder._pending_shadow_by_episode == {}
    assert recorder.summary["skipped_quality"][
        "shadow_for_ignored_or_rejected_episode"
    ] == 50

    recorder.consume_shadow_records(
        {"source_episode_id": "never-admitted"} for _ in range(10)
    )
    assert recorder._pending_shadow_by_episode == {}
    assert recorder.summary["skipped_quality"][
        "shadow_for_unadmitted_episode"
    ] == 10
    recorder.close_unresolved(t0)
    assert recorder._pending_shadow_by_episode == {}
    resumed = pickle.loads(pickle.dumps(recorder))
    assert resumed._pending_shadow_by_episode == {}


def test_leakage_guards_reject_future_input_and_shared_episode(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    recorder.observe(
        _snapshot(t0, _context(t0), _episode(t0)),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    row = recorder.drain_input_rows()[0].to_dict()
    leaked = dict(row)
    transition = json.loads(leaked["observation_transition_json"])
    transition["future_event"] = {"observed_at": _clock(1).isoformat()}
    leaked["observation_transition_json"] = _canonical_json(transition)
    leaked["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="clock exceeds asof"):
        validate_case_input_row(leaked)

    derived_deadline_leak = dict(row)
    transition = json.loads(derived_deadline_leak["observation_transition_json"])
    transition["deadline_result_at"] = _clock(1).isoformat()
    derived_deadline_leak["observation_transition_json"] = _canonical_json(transition)
    derived_deadline_leak["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="clock exceeds asof"):
        validate_case_input_row(derived_deadline_leak)

    naive_observed_clock = dict(row)
    transition = json.loads(naive_observed_clock["observation_transition_json"])
    transition["external_event"] = {"observed_at": "2025-01-06T10:01:00"}
    naive_observed_clock["observation_transition_json"] = _canonical_json(transition)
    naive_observed_clock["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="timezone-naive"):
        validate_case_input_row(naive_observed_clock)

    naive_deadline_result = dict(row)
    transition = json.loads(naive_deadline_result["observation_transition_json"])
    transition["deadline_result_at"] = "2025-01-06T10:01:00"
    naive_deadline_result["observation_transition_json"] = _canonical_json(transition)
    naive_deadline_result["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="timezone-naive"):
        validate_case_input_row(naive_deadline_result)

    naive_planned_deadline = dict(row)
    naive_planned_deadline["planned_deadline_at"] = "2025-01-06T14:00:00"
    naive_planned_deadline["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="timezone aware"):
        validate_case_input_row(naive_planned_deadline)

    for coverage_field, bad_value, message in (
        ("coverage_start_exclusive", "false", "boundary type"),
        ("observed_update_count", True, "update count"),
    ):
        malformed_coverage = dict(row)
        transition = json.loads(malformed_coverage["observation_transition_json"])
        scene = json.loads(malformed_coverage["scene_graph_delta_json"])
        transition["coverage"][coverage_field] = bad_value
        scene["coverage"][coverage_field] = bad_value
        malformed_coverage["observation_transition_json"] = _canonical_json(transition)
        malformed_coverage["scene_graph_delta_json"] = _canonical_json(scene)
        malformed_coverage["input_fingerprint"] = ""
        with pytest.raises(ValueError, match=message):
            validate_case_input_row(malformed_coverage)

    malformed_prefix = dict(row)
    prefixes = json.loads(malformed_prefix["prefix_refs_json"])
    prefixes[0]["frame_row_start"] = "0"
    malformed_prefix["prefix_refs_json"] = _canonical_json(prefixes)
    malformed_prefix["input_fingerprint"] = ""
    with pytest.raises(ValueError, match="not integers"):
        validate_case_input_row(malformed_prefix)

    with pytest.raises(ValueError, match="shared by data splits"):
        validate_episode_disjoint_splits({"train": [row], "validation": [row]})

    outcome_leak = dict(row)
    outcome_leak["first_event"] = "target"
    with pytest.raises(ValueError, match="schema differs"):
        validate_case_input_row(outcome_leak)

    with pytest.raises(ValueError, match="cardinality is incomplete"):
        validate_case_library_rows([row], [])


def test_materialized_outcome_semantics_fail_closed(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    recorder.observe(
        _snapshot(t0, _context(t0), _episode(t0)),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    inputs = [item.to_dict() for item in recorder.drain_input_rows()]
    recorder.close_unresolved(t0)
    outcome = recorder.drain_outcome_rows()[0].to_dict()
    validate_case_library_rows(inputs, [outcome])

    corruptions = (
        ({"resolved_at": _clock(-1)}, "precedes its last input"),
        ({"censored": 1}, "not boolean"),
        ({"mfe_R": float("inf")}, "not finite"),
        ({"first_event": "deadline", "deadline_first": False}, "inconsistent"),
        ({"first_event": "target", "target_first": False}, "inconsistent"),
        ({"expired": True}, "expiry semantics"),
        (
            {
                "first_event": "target",
                "target_first": True,
                "invalidation_first": False,
                "filled": True,
                "same_bar_collision": True,
                "censored": False,
                "resolution": "target_first",
            },
            "collision semantics",
        ),
        (
            {
                "first_event": "invalidation",
                "target_first": False,
                "invalidation_first": True,
                "filled": True,
                "censored": False,
                "resolution": "same_bar_invalidation_priority",
            },
            "collision flag is missing",
        ),
        ({"hit_2R": True, "hit_1R": False}, "R-hit ladder"),
    )
    for changes, message in corruptions:
        damaged = {**outcome, **changes}
        with pytest.raises(ValueError, match=message):
            validate_case_library_rows(inputs, [damaged])


def test_materialized_library_rejects_a_second_plan_formed_stage(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    episode = _episode(
        t0,
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        location_id="location:1",
        path_id="path:1",
        plan=_plan(t0),
    )
    recorder.observe(
        _snapshot(
            t0,
            _context(t0),
            episode,
            locations=(_location(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
            paths=(_path(t0, formed_at=t0 - pd.Timedelta(minutes=5)),),
        ),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    inputs = [item.to_dict() for item in recorder.drain_input_rows()]
    plan_row = next(row for row in inputs if row["revision_stage"] == "plan_formed")
    recorder.close_unresolved(t0)
    outcome = recorder.drain_outcome_rows()[0].to_dict()
    with pytest.raises(
        ValueError,
        match="immutable revision stage was materialized more than once",
    ):
        validate_case_library_rows([*inputs, dict(plan_row)], [outcome])


def test_separate_stream_manifests_are_hash_bound(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    t0 = _clock()
    recorder.observe(
        _snapshot(t0, _context(t0), _episode(t0)),
        source_bar=_bar(t0),
        source_row_ordinal=0,
    )
    inputs = [item.to_dict() for item in recorder.drain_input_rows()]
    recorder.close_unresolved(t0)
    outcomes = [item.to_dict() for item in recorder.drain_outcome_rows()]
    input_count = len(inputs)
    outcome_count = len(outcomes)
    input_state = new_stream_state(CAUSAL_CASE_INPUT_FIELD_TYPES)
    outcome_state = new_stream_state(CAUSAL_CASE_OUTCOME_FIELD_TYPES)
    write_stream_shard(
        tmp_path,
        "causal_case_input_shards",
        inputs,
        input_state,
        key_column="revision_id",
        field_types=CAUSAL_CASE_INPUT_FIELD_TYPES,
    )
    write_stream_shard(
        tmp_path,
        "causal_case_outcome_shards",
        outcomes,
        outcome_state,
        key_column="outcome_id",
        field_types=CAUSAL_CASE_OUTCOME_FIELD_TYPES,
    )
    input_manifest = write_stream_manifest(
        tmp_path,
        "causal_case_input_shards",
        input_state,
        artifact="case_inputs",
        bindings={"run_manifest": "run_manifest.json"},
    )
    outcome_manifest = write_stream_manifest(
        tmp_path,
        "causal_case_outcome_shards",
        outcome_state,
        artifact="case_outcomes",
        bindings={"run_manifest": "run_manifest.json"},
    )
    run_manifest = tmp_path / "run_manifest.json"
    run_payload = {
        "schema_version": 1,
        "runner": "continuous_replay",
        "causal_case_identity": expected_causal_case_run_identity(),
        "output": {
            "causal_case_library": True,
            "stream_families": [
                "causal_case_input_shards",
                "causal_case_outcome_shards",
            ],
        },
    }

    def write_run(payload: object) -> None:
        run_manifest.write_text(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    write_run(run_payload)
    pair = write_causal_case_library_manifest(
        tmp_path,
        input_stream_manifest=input_manifest.name,
        outcome_stream_manifest=outcome_manifest.name,
        run_manifest="run_manifest.json",
    )
    payload = json.loads(pair.read_text(encoding="utf-8"))
    assert payload["recorder_schema_version"] == CAUSAL_CASE_RECORDER_SCHEMA_VERSION
    assert payload["protocol"] == dict(CAUSAL_CASE_PROTOCOL)
    assert payload["bindings"]["run_manifest_sha256"] == (
        hashlib.sha256(run_manifest.read_bytes()).hexdigest()
    )
    assert payload["input_stream"]["rows"] == input_count
    assert payload["future_outcome_stream"]["rows"] == outcome_count
    assert payload["leakage_contract"]["outcome_fields_in_input_schema"] is False

    pair_bytes = pair.read_bytes()
    missing_identity = {
        key: value
        for key, value in run_payload.items()
        if key != "causal_case_identity"
    }
    schema_six = json.loads(json.dumps(run_payload))
    schema_six["causal_case_identity"]["recorder_schema_version"] = 6
    schema_six["causal_case_identity"]["protocol"]["protocol_version"] = (
        "entry-episode-causal-case-1.5.0"
    )
    tampered_future_blind = json.loads(json.dumps(run_payload))
    tampered_future_blind["causal_case_identity"][
        "future_visible_to_input"
    ] = True
    disabled_output = json.loads(json.dumps(run_payload))
    disabled_output["output"]["causal_case_library"] = False
    missing_outcome_family = json.loads(json.dumps(run_payload))
    missing_outcome_family["output"]["stream_families"] = [
        "causal_case_input_shards"
    ]
    duplicate_input_family = json.loads(json.dumps(run_payload))
    duplicate_input_family["output"]["stream_families"].append(
        "causal_case_input_shards"
    )
    for damaged_run in (
        missing_identity,
        schema_six,
        tampered_future_blind,
        disabled_output,
        missing_outcome_family,
        duplicate_input_family,
    ):
        write_run(damaged_run)
        with pytest.raises(ValueError, match="run manifest identity"):
            write_causal_case_library_manifest(
                tmp_path,
                input_stream_manifest=input_manifest.name,
                outcome_stream_manifest=outcome_manifest.name,
                run_manifest=run_manifest.name,
            )
        assert pair.read_bytes() == pair_bytes
    write_run(run_payload)
