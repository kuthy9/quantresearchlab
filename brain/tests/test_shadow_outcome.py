from __future__ import annotations

from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from shares.core.model import (
    Action,
    Bar,
    Direction,
    EntryLocationState,
    EntryLocationLifecycle,
    InteractionUpdate,
    LiquidityInventoryLifecycle,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
    Playbook,
    PlaybookPhase,
    StructureLifecycle,
    Timeframe,
)
from brain.core.shadow_outcome import (
    SHADOW_MOTIF_ROOT_SAMPLE_LIMIT,
    SHADOW_OUTCOME_PROTOCOL,
    ShadowCandidateOutcomeRecorder,
    _target_r_bucket,
    aggregate_shadow_mechanism_motifs,
    derive_shadow_episode_outcomes,
    derive_shadow_mechanism_challenges,
    derive_shadow_root_episode_records,
    derive_shadow_root_sequence_records,
)


TZ = "America/New_York"


def _bar(
    start: pd.Timestamp,
    *,
    open_: float = 100.0,
    high: float = 100.5,
    low: float = 99.5,
    close: float = 100.0,
) -> Bar:
    return Bar(
        start=start,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
        symbol="NQH4",
        instrument_id=1,
    )


def _hypothesis(
    playbook: Playbook,
    direction: Direction,
    *,
    executable: bool,
) -> SimpleNamespace:
    phase = (
        PlaybookPhase.EXECUTABLE
        if executable
        else PlaybookPhase.INACTIVE
    )
    root_id = f"displacement:{direction.value}"
    candidate_id = (
        f"candidate:{playbook.value}:{direction.value}:{root_id}"
    )
    return SimpleNamespace(
        playbook=playbook,
        direction=direction,
        phase=phase,
        key=candidate_id,
        candidate_id=candidate_id,
        required_root_id=root_id,
        plan=object() if executable else None,
        plan_feasibility=SimpleNamespace(
            valid=False,
            failure_reason="playbook_plan_unavailable",
        ),
        sequence=None,
        hard_gate_results={"causal_gate": executable},
        delivery_quality=1.0 if executable else 0.0,
        thesis_strength=0.8 if executable else 0.1,
        sequence_progress=1.0 if executable else 0.0,
        location_quality=1.0 if executable else 0.0,
        entry_readiness=1.0 if executable else 0.0,
        uncertainty=0.1,
        market_thesis_binding_required=True,
        market_thesis_action_bound=True,
        market_thesis_ids=(f"thesis:{direction.value}",),
        market_thesis_id=f"thesis:{direction.value}",
        bound_market_thesis_id=f"thesis:{direction.value}",
        market_thesis_root_id=root_id,
        playbook_match_strength=0.8 if executable else 0.0,
    )


def _snapshot(
    source_bar: Bar,
    direction: Direction,
    *,
    event: bool = True,
    stop: float | None = None,
    target: float | None = None,
    typed_delta: bool = True,
    executable_plan: bool = False,
    invalidated_source_ids: tuple[str, ...] = (),
) -> SimpleNamespace:
    asof = source_bar.end
    stop = stop if stop is not None else (
        98.0 if direction is Direction.LONG else 102.0
    )
    target = target if target is not None else (
        103.0 if direction is Direction.LONG else 97.0
    )
    structure = SimpleNamespace(
        lifecycle=StructureLifecycle.CONFIRMED,
        direction=direction,
        protected_price=stop,
        structure_id=f"structure:{direction.value}",
        protected_swing_id=f"swing:{direction.value}",
        latest_high_id=None,
        latest_low_id=None,
    )
    target_item = SimpleNamespace(
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        side=direction.opposing_liquidity_side,
        lower_bound=target,
        upper_bound=target,
        item_id=f"draw:{direction.value}",
        timeframe=Timeframe.H1,
        source_ids=(f"draw-source:{direction.value}",),
        is_protected_swing=False,
        structural_rank="external",
    )
    transition = SimpleNamespace(
        lifecycle="active",
        observed_at=asof,
        entity_id=f"displacement:{direction.value}",
        transition_id=f"transition:{asof.isoformat()}:{direction.value}",
        direction=direction,
    )
    observation = SimpleNamespace(
        asof=asof,
        symbol=source_bar.symbol,
        instrument_id=source_bar.instrument_id,
        price=source_bar.close,
        anomalies=(),
        typed_transition_delta_available=typed_delta,
        frames={
            Timeframe.M5: SimpleNamespace(
                timeframe=Timeframe.M5,
                structures=(structure,),
                structure_breaks=(),
            )
        },
        displacement=SimpleNamespace(
            transitions_this_update=(transition,) if event else ()
        ),
        liquidity_inventory=(target_item,),
        group4_manipulation_transitions_this_update=(),
        interaction_update=InteractionUpdate(
            zone_interactions=(),
            reacceptance_interactions=(),
            micro_break_facts=(),
            interaction_paths=(),
        ),
    )
    hypotheses = tuple(
        _hypothesis(
            playbook,
            direction,
            executable=(
                playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
            ),
        )
        for playbook in Playbook
    )
    if executable_plan:
        dfp = next(
            item
            for item in hypotheses
            if item.playbook
            is Playbook.DISPLACEMENT_FIRST_PULLBACK
        )
        dfp.episode_id = "episode:dfp:long"
        dfp.plan = SimpleNamespace(
            planned_entry=100.0,
            invalidation=SimpleNamespace(
                price=stop,
                source_level_id=f"structure:{direction.value}",
            ),
            targets=(
                SimpleNamespace(
                    price=target,
                    level_id=f"draw:{direction.value}",
                ),
            ),
            deadline=asof + pd.Timedelta(minutes=60),
            setup_id="setup:dfp",
            entry_location_id="location:dfp",
            entry_path_id="path:dfp",
            entry_zone_lower=99.5,
            entry_zone_upper=100.5,
            selected_draw_id=f"draw:{direction.value}",
        )
        dfp.plan_feasibility = SimpleNamespace(
            valid=True,
            failure_reason=None,
        )
    open_thesis = SimpleNamespace(
        thesis_id=f"thesis:{direction.value}",
        root_id=f"displacement:{direction.value}",
        direction=direction,
        mechanism_event_ids=(f"displacement:{direction.value}",),
        entry_location_ids=(),
        trigger_event_ids=(),
        authority_source_ids=(f"structure:{direction.value}",),
        source_timeframe=Timeframe.H1,
        mechanism="directional_displacement_continuation",
        authority_relation="aligned",
    )
    belief = SimpleNamespace(
        candidates=lambda: hypotheses,
        action_candidate_items=lambda: tuple(
            (item.candidate_id, item) for item in hypotheses
        ),
        lifecycle_candidate_items=lambda: tuple(
            (item.candidate_id, item) for item in hypotheses
        ),
        resolve_hypothesis=lambda identity: next(
            (
                item
                for item in hypotheses
                if identity in {item.candidate_id, item.key}
            ),
            None,
        ),
        global_context=SimpleNamespace(
            open_market_theses=(open_thesis,),
            invalidated_source_ids=invalidated_source_ids,
        ),
    )
    return SimpleNamespace(
        observation=observation,
        belief=belief,
        decision=SimpleNamespace(selected_action=Action.WAIT),
        risk=SimpleNamespace(final_action=Action.ABSTAIN),
    )


def _lsr_location(
    asof: pd.Timestamp,
    suffix: str,
    *,
    lifecycle: EntryLocationLifecycle,
    formed_at: pd.Timestamp | None = None,
) -> EntryLocationState:
    formed = (
        asof - pd.Timedelta(minutes=1)
        if formed_at is None and lifecycle is EntryLocationLifecycle.IN_ZONE
        else asof if formed_at is None else formed_at
    )
    entered_at = asof if lifecycle is EntryLocationLifecycle.IN_ZONE else None
    return EntryLocationState(
        location_id=f"location:lsr:{suffix}",
        protocol_hash="1" * 64,
        source_zone_detector_protocol_hash="2" * 64,
        symbol="NQH4",
        instrument_id=1,
        direction=Direction.LONG,
        source_zone_kind="fvg",
        source_zone_id=f"zone:lsr:{suffix}",
        source_zone_protocol_hash="2" * 64,
        source_displacement_id="displacement:lsr:shared",
        source_bos_id=None,
        lower_bound=98.0,
        upper_bound=100.0,
        midpoint=99.0,
        near_edge=100.0,
        far_edge=98.0,
        failure_boundary=98.0,
        formed_at=formed,
        lifecycle=lifecycle,
        state_started_at=entered_at or formed,
        last_updated_at=asof,
        age_real_1m_bars=max(
            0,
            int((asof - formed) / pd.Timedelta(minutes=1)),
        ),
        state_duration_real_1m_bars=0,
        current_price=100.0,
        distance_to_zone_points=0.0,
        distance_to_failure_points=2.0,
        departure_confirmed_at=formed,
        first_entered_at=entered_at,
        entry_mode=(
            "crossed_near_edge" if entered_at is not None else None
        ),
        contact_reference_price=(100.0 if entered_at is not None else None),
        first_penetration_fraction=(0.5 if entered_at is not None else 0.0),
        transition_reason=(
            "first_completed_bar_entered_zone"
            if entered_at is not None
            else "departure_confirmed"
        ),
    )


def _lsr_path(
    location: EntryLocationState,
    *,
    sequence_id: str | None = None,
) -> PathSequenceState:
    steps = [
        PathSequenceStep(
            step_id=f"step:visible:{location.location_id}",
            kind="zone_visible",
            observed_at=location.formed_at,
            source_event_id=location.source_zone_id,
            source_entity_id=location.source_zone_id,
            predecessor_step_ids=(),
            same_clock_relation="origin",
            direction=location.direction,
            strength=0.5,
            reason="typed_entry_zone_registered",
        ),
        PathSequenceStep(
            step_id=f"step:departed:{location.location_id}",
            kind="departure_confirmed",
            observed_at=location.departure_confirmed_at,
            source_event_id=None,
            source_entity_id=location.location_id,
            predecessor_step_ids=(f"step:visible:{location.location_id}",),
            same_clock_relation="same_clock_known",
            direction=location.direction,
            strength=0.5,
            reason="formation_close_on_delivery_side",
        ),
    ]
    if location.first_entered_at is not None:
        steps.append(
            PathSequenceStep(
                step_id=f"step:pullback:{location.location_id}",
                kind="first_pullback",
                observed_at=location.first_entered_at,
                source_event_id=None,
                source_entity_id=location.location_id,
                predecessor_step_ids=(steps[-1].step_id,),
                same_clock_relation="strictly_after",
                direction=location.direction,
                strength=0.5,
                reason="crossed_near_edge",
            )
        )
    return PathSequenceState(
        sequence_id=(
            f"path:lsr:{location.location_id}"
            if sequence_id is None
            else sequence_id
        ),
        protocol_hash=location.protocol_hash,
        symbol=location.symbol,
        instrument_id=location.instrument_id,
        context_kind="zone_return",
        context_id=location.location_id,
        direction=location.direction,
        lifecycle=PathSequenceLifecycle.ACTIVE,
        formed_at=location.formed_at,
        state_started_at=location.formed_at,
        last_updated_at=location.last_updated_at,
        age_real_1m_bars=location.age_real_1m_bars,
        state_duration_real_1m_bars=location.age_real_1m_bars,
        steps=tuple(steps),
        transition_reason="context_registered",
    )


def _set_lsr_interaction(
    snapshot: SimpleNamespace,
    locations: tuple[EntryLocationState, ...],
    *,
    path_ids: tuple[str, ...] | None = None,
) -> None:
    unique_locations = tuple(
        {item.location_id: item for item in locations}.values()
    )
    if path_ids is None:
        paths = tuple(_lsr_path(item) for item in unique_locations)
    else:
        if len(path_ids) != len(unique_locations):
            raise ValueError("test interaction path identities differ")
        paths = tuple(
            _lsr_path(item, sequence_id=path_id)
            for item, path_id in zip(unique_locations, path_ids)
        )
    snapshot.observation.interaction_update = InteractionUpdate(
        zone_interactions=unique_locations,
        reacceptance_interactions=(),
        micro_break_facts=(),
        interaction_paths=paths,
    )


def _with_lsr_zone_siblings(
    snapshot: SimpleNamespace,
    locations: tuple[SimpleNamespace, ...],
    *,
    transition_locations: tuple[SimpleNamespace, ...],
    duplicate_first_location: bool = False,
) -> tuple[SimpleNamespace, ...]:
    root_id = "manipulation:lsr:shared"
    thesis_id = "thesis:lsr:shared"
    context_id = "context:lsr:shared"
    base = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
    )
    siblings: list[SimpleNamespace] = []
    for index, location in enumerate(locations):
        bound_location = locations[0] if duplicate_first_location else location
        sibling = SimpleNamespace(**vars(base))
        sibling.candidate_id = f"candidate:lsr:{index}"
        sibling.key = sibling.candidate_id
        sibling.required_root_id = root_id
        sibling.market_thesis_root_id = root_id
        sibling.market_thesis_ids = (thesis_id,)
        sibling.market_thesis_id = thesis_id
        sibling.bound_market_thesis_id = thesis_id
        sibling.episode_id = f"episode:lsr:{index}"
        sibling.setup_context_id = sibling.episode_id
        sibling.context_thesis_id = context_id
        sibling.parent_context_thesis_id = context_id
        sibling.entry_location_id = bound_location.location_id
        sibling.entry_path_id = f"path:lsr:{bound_location.location_id}"
        sibling.context_metadata = {
            "lsr_displacement_id": (
                bound_location.source_displacement_id
            ),
            "lsr_entry_zone_id": bound_location.source_zone_id,
        }
        sibling.phase = (
            PlaybookPhase.WAITING_LOCATION
            if bound_location.lifecycle
            is EntryLocationLifecycle.APPROACHING
            else PlaybookPhase.WAITING_TRIGGER
        )
        sibling.plan = None
        sibling.plan_feasibility = SimpleNamespace(
            valid=False,
            failure_reason="playbook_plan_unavailable",
        )
        sibling.sequence = None
        sibling.market_thesis_action_bound = True
        sibling.market_thesis_match_status = "exact_root_bound"
        sibling.selected_trigger = None
        siblings.append(sibling)
    other = tuple(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is not Playbook.LIQUIDITY_SWEEP_REVERSAL
    )
    action_items = tuple(
        (item.candidate_id, item) for item in (*other, *siblings)
    )
    action_map = dict(action_items)
    entry_episodes = {
        item.candidate_id: SimpleNamespace(
            candidate_id=item.candidate_id,
            episode_id=item.episode_id,
            parent_context_thesis_id=context_id,
            playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
            direction=Direction.LONG,
            entry_location_id=item.entry_location_id,
            entry_path_id=item.entry_path_id,
            first_pullback_at=(
                next(
                    (
                        location.first_entered_at
                        for location in locations
                        if location.location_id == item.entry_location_id
                    ),
                    None,
                )
            ),
            selected_trigger=item.selected_trigger,
            plan=item.plan,
            phase=item.phase,
            terminal_at=None,
            terminal_reason=None,
        )
        for item in siblings
    }
    context_theses = {
        context_id: SimpleNamespace(
            context_thesis_id=context_id,
            direction=Direction.LONG,
            authority_ids=(root_id, "displacement:lsr:shared"),
            lifecycle="active",
            terminal_at=None,
            terminal_reason=None,
            child_episode_ids=tuple(item.episode_id for item in siblings),
        )
    }
    snapshot.belief.candidates = lambda: (*other, *siblings)
    snapshot.belief.action_candidate_items = lambda: action_items
    snapshot.belief.lifecycle_candidate_items = lambda: action_items
    snapshot.belief.resolve_hypothesis = action_map.get
    snapshot.belief.entry_episodes = entry_episodes
    snapshot.belief.context_theses = context_theses
    snapshot.belief.global_context.open_market_theses = (
        SimpleNamespace(
            thesis_id=thesis_id,
            root_id=root_id,
            direction=Direction.LONG,
            mechanism_event_ids=(root_id, "displacement:lsr:shared"),
            entry_location_ids=tuple(
                item.location_id for item in locations
            ),
            trigger_event_ids=(),
            authority_source_ids=("structure:long",),
            source_timeframe=Timeframe.H1,
            mechanism="liquidity_sweep_reversal",
            authority_relation="opposed",
        ),
    )
    del transition_locations
    _set_lsr_interaction(snapshot, locations)
    return tuple(siblings)


def _add_open_thesis_revision(
    snapshot: SimpleNamespace,
    revision: str,
    *,
    updated_at: pd.Timestamp | None = None,
    include_draw: bool = True,
) -> SimpleNamespace:
    thesis = snapshot.belief.global_context.open_market_theses[0]
    thesis.updated_at = (
        snapshot.observation.asof if updated_at is None else updated_at
    )
    thesis.evidence_revision_id = revision
    # The fixture's protected structure is materialized on its M5 frame.
    thesis.source_timeframe = Timeframe.M5
    thesis.draw_candidate_ids = (
        (f"draw:{thesis.direction.value}",) if include_draw else ()
    )
    thesis.obstruction_ids = ()
    thesis.conflict_ids = ()
    thesis.supporting_event_ids = thesis.mechanism_event_ids
    thesis.opposing_event_ids = ()
    return snapshot


def _start() -> pd.Timestamp:
    return pd.Timestamp("2024-01-08 09:30", tz=TZ)


def _register(
    recorder: ShadowCandidateOutcomeRecorder,
    direction: Direction,
) -> tuple[Bar, SimpleNamespace]:
    source = _bar(_start())
    snapshot = _snapshot(source, direction)
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    return source, snapshot


def _complete_engine_step(
    recorder: ShadowCandidateOutcomeRecorder,
    bar: Bar,
    direction: Direction,
    *,
    executable_plan: bool = False,
    invalidated_source_ids: tuple[str, ...] = (),
) -> None:
    recorder.on_bar(bar)
    recorder.observe(
        _snapshot(
            bar,
            direction,
            event=False,
            executable_plan=executable_plan,
            invalidated_source_ids=invalidated_source_ids,
        ),
        source_bar=bar,
    )


def _freeze_test_sequences(snapshot: SimpleNamespace) -> None:
    asof = snapshot.observation.asof
    for hypothesis in snapshot.belief.candidates():
        hypothesis.sequence = SimpleNamespace(
            steps=(
                SimpleNamespace(
                    step_id="market_root_formed",
                    satisfied=True,
                    observed_at=asof - pd.Timedelta(minutes=2),
                ),
                SimpleNamespace(
                    step_id="directional_reaction",
                    satisfied=True,
                    observed_at=asof - pd.Timedelta(minutes=1),
                ),
                SimpleNamespace(
                    step_id="future_trigger",
                    satisfied=False,
                    observed_at=None,
                ),
            )
        )


def _valid_neutral_challenges() -> tuple:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(source, Direction.LONG)
    _freeze_test_sequences(snapshot)
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=99.5),
        Direction.LONG,
    )
    return derive_shadow_mechanism_challenges(list(recorder.drain_rows()))


def test_candidate_cannot_see_its_formation_bar() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(
        _start(),
        high=104.0,
        low=97.0,
    )
    recorder.on_bar(source)
    recorder.observe(_snapshot(source, Direction.LONG), source_bar=source)

    assert recorder.rows == ()
    assert len(recorder.open_candidates) == 1

    later = _bar(source.end, high=103.5, low=99.5)
    _complete_engine_step(recorder, later, Direction.LONG)
    row = recorder.drain_rows()[0]
    assert row.entry_at == later.start
    assert row.resolution == "target_first"
    assert row.candidate_origin == "shadow_eye_candidate"


@pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
def test_long_short_target_resolution_is_symmetric(direction: Direction) -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, direction)
    later = (
        _bar(source.end, high=104.0, low=99.5)
        if direction is Direction.LONG
        else _bar(source.end, high=100.5, low=96.0)
    )
    _complete_engine_step(recorder, later, direction)

    row = recorder.drain_rows()[0]
    assert row.target_before_invalidation is True
    assert row.invalidation_before_target is False
    assert row.mfe_R == pytest.approx(1.5)
    assert row.hit_0_5R and row.hit_1R
    assert row.hit_2R is False


@pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
def test_same_bar_collision_is_conservatively_invalidated(
    direction: Direction,
) -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, direction)
    collision = _bar(source.end, high=104.0, low=97.0)
    _complete_engine_step(recorder, collision, direction)

    row = recorder.drain_rows()[0]
    assert row.resolution == "same_bar_invalidation_priority"
    assert row.same_bar_collision is True
    assert row.target_before_invalidation is False
    assert row.invalidation_before_target is True
    assert row.mfe_points == 0.0
    assert row.hit_0_5R is False


def test_geometry_is_frozen_and_open_state_is_pickle_safe() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    frozen = recorder.open_candidates[0]
    assert (frozen.invalidation_price, frozen.target_price) == (98.0, 103.0)

    restored = pickle.loads(pickle.dumps(recorder))
    quiet = _bar(source.end, high=101.0, low=99.0)
    restored.on_bar(quiet)
    changed = _snapshot(
        quiet,
        Direction.LONG,
        event=False,
        stop=97.0,
        target=106.0,
    )
    restored.observe(changed, source_bar=quiet)
    still_frozen = restored.open_candidates[0]
    assert (still_frozen.invalidation_price, still_frozen.target_price) == (
        98.0,
        103.0,
    )


def test_right_boundary_censors_without_assigning_a_quadrant() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    recorder.close_unresolved(source.end + pd.Timedelta(hours=1))

    row = recorder.drain_rows()[0]
    outcomes = json.loads(row.playbook_outcomes)
    assert row.censored is True
    assert row.resolution == "window_right_censored"
    assert all(item["quadrant"] is None for item in outcomes)
    assert recorder.summary["four_quadrants"] == {}


def test_neutral_candidate_does_not_pollute_episode_quadrants() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=99.5),
        Direction.LONG,
    )
    row = recorder.drain_rows()[0]
    outcomes = json.loads(row.playbook_outcomes)

    assert len(outcomes) == 3
    assert len({item["playbook"] for item in outcomes}) == 3
    assert recorder.summary["four_quadrants"] == {}
    assert SHADOW_OUTCOME_PROTOCOL["protocol_version"] == (
        "shadow-candidate-outcome-1.7.0"
    )


def test_lsr_eligible_zone_and_first_pullback_bind_the_exact_child() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed_a = _lsr_location(
        source.end,
        "a",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formed_b = _lsr_location(
        source.end,
        "b",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formation = _snapshot(source, Direction.LONG, event=False)
    siblings = _with_lsr_zone_siblings(
        formation,
        (formed_a, formed_b),
        transition_locations=(formed_a, formed_b),
    )
    recorder.on_bar(source)
    recorder.observe(formation, source_bar=source)

    eligible = tuple(
        item
        for item in recorder.open_candidates
        if item.event_kind == "eligible_entry_fvg"
    )
    assert len(eligible) == 2
    assert {
        (item.entry_location_id, item.source_episode_id)
        for item in eligible
    } == {
        (formed_a.location_id, siblings[0].episode_id),
        (formed_b.location_id, siblings[1].episode_id),
    }
    assert all(
        item.entry_episode_binding_status == "exact_entry_location"
        for item in eligible
    )

    later = _bar(source.end, high=101.0, low=99.0)
    waiting_a = _lsr_location(
        later.end,
        "a",
        lifecycle=EntryLocationLifecycle.APPROACHING,
        formed_at=source.end,
    )
    entered_b = _lsr_location(
        later.end,
        "b",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        formed_at=source.end,
    )
    first_pullback = _snapshot(later, Direction.LONG, event=False)
    later_siblings = _with_lsr_zone_siblings(
        first_pullback,
        (waiting_a, entered_b),
        transition_locations=(entered_b,),
    )
    recorder.on_bar(later)
    recorder.observe(first_pullback, source_bar=later)

    first_entries = tuple(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert len(first_entries) == 1
    first = first_entries[0]
    assert first.entry_location_id == entered_b.location_id
    assert first.entry_path_id == later_siblings[1].entry_path_id
    assert first.source_episode_id == later_siblings[1].episode_id
    assert first.source_episode_id != later_siblings[0].episode_id
    lsr = next(
        item
        for item in first.playbook_diagnostics
        if item["playbook"]
        == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["hypothesis_key"] == later_siblings[1].candidate_id
    assert lsr["episode_id"] == later_siblings[1].episode_id
    assert lsr["entry_location_id"] == entered_b.location_id
    assert lsr["entry_episode_binding_status"] == "exact_entry_location"
    assert recorder.summary["entry_episode_bindings"] == {
        "exact_entry_location": 3
    }


def test_lsr_exact_formation_custody_survives_terminal_before_pullback() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "terminal-after-formation",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formation = _snapshot(source, Direction.LONG, event=False)
    owner = _with_lsr_zone_siblings(
        formation,
        (formed,),
        transition_locations=(formed,),
    )[0]
    recorder.on_bar(source)
    recorder.observe(formation, source_bar=source)

    later = _bar(source.end, high=101.0, low=99.0)
    entered = _lsr_location(
        later.end,
        "terminal-after-formation",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        formed_at=source.end,
    )
    terminal = _snapshot(later, Direction.LONG, event=False)
    terminal_owner = _with_lsr_zone_siblings(
        terminal,
        (entered,),
        transition_locations=(entered,),
    )[0]
    terminal_owner.phase = PlaybookPhase.INVALIDATED
    terminal_episode = terminal.belief.entry_episodes[
        terminal_owner.candidate_id
    ]
    terminal_episode.phase = PlaybookPhase.INVALIDATED
    terminal_episode.terminal_at = later.end
    terminal_episode.terminal_reason = "entry_deadline_elapsed"
    terminal_context = terminal.belief.context_theses[
        terminal_owner.context_thesis_id
    ]
    terminal_context.lifecycle = "invalidated"
    terminal_context.terminal_at = later.end
    terminal_context.terminal_reason = "context_thesis_deadline_elapsed"

    recorder.on_bar(later)
    recorder.observe(terminal, source_bar=later)

    first = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert first.source_episode_id == owner.episode_id
    assert first.source_setup_id == owner.setup_context_id
    assert first.source_context_thesis_id == owner.context_thesis_id
    assert first.entry_location_id == formed.location_id
    assert first.entry_path_id == owner.entry_path_id
    assert first.entry_episode_binding_status == (
        "episode_terminal_before_first_entry"
    )
    assert first.entry_episode_terminal_at == later.end
    assert first.entry_episode_terminal_reason == "entry_deadline_elapsed"
    lsr = next(
        item
        for item in first.playbook_diagnostics
        if item["playbook"] == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["accepted"] is False
    assert lsr["first_failed_gate"] == (
        "episode_terminal_before_first_entry"
    )
    assert lsr["episode_id"] == owner.episode_id
    assert lsr["context_thesis_id"] == owner.context_thesis_id
    assert lsr["entry_episode_terminal_at"] == later.end.isoformat()
    assert lsr["entry_episode_terminal_reason"] == (
        "entry_deadline_elapsed"
    )


def test_lsr_terminal_tombstone_survives_projection_compaction_and_checkpoint(
) -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "terminal-tombstone",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formation = _snapshot(source, Direction.LONG, event=False)
    owner = _with_lsr_zone_siblings(
        formation,
        (formed,),
        transition_locations=(formed,),
    )[0]
    recorder.on_bar(source)
    recorder.observe(formation, source_bar=source)

    terminal_bar = _bar(source.end, high=101.0, low=99.8)
    still_waiting = _lsr_location(
        terminal_bar.end,
        "terminal-tombstone",
        lifecycle=EntryLocationLifecycle.APPROACHING,
        formed_at=source.end,
    )
    terminal_snapshot = _snapshot(
        terminal_bar,
        Direction.LONG,
        event=False,
    )
    terminal_owner = _with_lsr_zone_siblings(
        terminal_snapshot,
        (still_waiting,),
        transition_locations=(),
    )[0]
    terminal_owner.phase = PlaybookPhase.INVALIDATED
    terminal_episode = terminal_snapshot.belief.entry_episodes[
        terminal_owner.candidate_id
    ]
    terminal_episode.phase = PlaybookPhase.INVALIDATED
    terminal_episode.terminal_at = terminal_bar.end
    terminal_episode.terminal_reason = "entry_deadline_elapsed"
    recorder.on_bar(terminal_bar)
    recorder.observe(terminal_snapshot, source_bar=terminal_bar)

    custody = next(iter(recorder._lsr_zone_custody.values()))
    assert custody.source_episode_id == owner.episode_id
    assert custody.owner_terminal_at == terminal_bar.end
    assert custody.owner_terminal_reason == "entry_deadline_elapsed"
    restored = pickle.loads(pickle.dumps(recorder))

    previous = terminal_bar
    for _ in range(2):
        quiet = _bar(previous.end, high=101.0, low=99.8)
        restored.on_bar(quiet)
        restored.observe(
            _snapshot(quiet, Direction.LONG, event=False),
            source_bar=quiet,
        )
        previous = quiet

    first_bar = _bar(previous.end, high=101.0, low=99.0)
    entered = _lsr_location(
        first_bar.end,
        "terminal-tombstone",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        formed_at=source.end,
    )
    projection_free = _snapshot(
        first_bar,
        Direction.LONG,
        event=False,
    )
    _set_lsr_interaction(
        projection_free,
        (entered,),
        path_ids=(owner.entry_path_id,),
    )
    restored.on_bar(first_bar)
    restored.observe(projection_free, source_bar=first_bar)

    first = next(
        item
        for item in restored.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert first.source_episode_id == owner.episode_id
    assert first.source_context_thesis_id == owner.context_thesis_id
    assert first.entry_episode_binding_status == (
        "episode_terminal_before_first_entry"
    )
    assert first.entry_episode_terminal_at == terminal_bar.end
    assert first.entry_episode_terminal_reason == "entry_deadline_elapsed"
    restored.close_unresolved(first_bar.end + pd.Timedelta(hours=1))
    row = next(
        item
        for item in restored.drain_rows()
        if item.candidate_id == first.candidate_id
    )
    assert row.entry_episode_terminal_at == terminal_bar.end
    assert row.entry_episode_terminal_reason == "entry_deadline_elapsed"
    challenge = next(
        item
        for item in derive_shadow_mechanism_challenges((row,))
        if item.playbook == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert challenge.entry_episode_terminal_at == terminal_bar.end
    assert challenge.entry_episode_terminal_reason == (
        "entry_deadline_elapsed"
    )


def test_lsr_ambiguous_formation_cannot_be_rebound_by_future_unique_owner() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "ambiguous-formation",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formation = _snapshot(source, Direction.LONG, event=False)
    _with_lsr_zone_siblings(
        formation,
        (formed, formed),
        transition_locations=(formed,),
        duplicate_first_location=True,
    )
    recorder.on_bar(source)
    recorder.observe(formation, source_bar=source)
    eligible = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "eligible_entry_fvg"
    )
    assert eligible.entry_episode_binding_status == (
        "entry_episode_binding_ambiguous"
    )
    recorder = pickle.loads(pickle.dumps(recorder))

    later = _bar(source.end, high=101.0, low=99.0)
    entered = _lsr_location(
        later.end,
        "ambiguous-formation",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        formed_at=source.end,
    )
    unique = _snapshot(later, Direction.LONG, event=False)
    current_owner = _with_lsr_zone_siblings(
        unique,
        (entered,),
        transition_locations=(entered,),
    )[0]
    recorder.on_bar(later)
    recorder.observe(unique, source_bar=later)

    first = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert first.source_episode_id is None
    assert first.source_episode_id != current_owner.episode_id
    assert first.entry_episode_binding_status == (
        "entry_episode_binding_ambiguous"
    )


def test_lsr_terminal_grace_sibling_does_not_ambiguate_live_owner() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "terminal-grace",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    snapshot = _snapshot(source, Direction.LONG, event=False)
    terminal_owner, live_owner = _with_lsr_zone_siblings(
        snapshot,
        (formed, formed),
        transition_locations=(formed,),
        duplicate_first_location=True,
    )
    terminal_context_id = "context:lsr:terminal-grace"
    terminal_owner.context_thesis_id = terminal_context_id
    terminal_owner.parent_context_thesis_id = terminal_context_id
    terminal_owner.phase = PlaybookPhase.INVALIDATED
    terminal_episode = snapshot.belief.entry_episodes[
        terminal_owner.candidate_id
    ]
    terminal_episode.parent_context_thesis_id = terminal_context_id
    terminal_episode.phase = PlaybookPhase.INVALIDATED
    terminal_episode.terminal_at = source.end
    live_context = snapshot.belief.context_theses[live_owner.context_thesis_id]
    live_context.child_episode_ids = (live_owner.episode_id,)
    snapshot.belief.context_theses[terminal_context_id] = SimpleNamespace(
        context_thesis_id=terminal_context_id,
        direction=Direction.LONG,
        authority_ids=(
            terminal_owner.required_root_id,
            formed.source_displacement_id,
        ),
        lifecycle="invalidated",
        terminal_at=source.end,
        child_episode_ids=(terminal_owner.episode_id,),
    )

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    eligible = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "eligible_entry_fvg"
    )
    assert eligible.source_episode_id == live_owner.episode_id
    assert eligible.source_context_thesis_id == live_owner.context_thesis_id
    assert eligible.entry_episode_binding_status == "exact_entry_location"


def test_lsr_multiple_active_contexts_for_one_zone_fail_closed() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "multi-context",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    snapshot = _snapshot(source, Direction.LONG, event=False)
    first_owner, second_owner = _with_lsr_zone_siblings(
        snapshot,
        (formed, formed),
        transition_locations=(formed,),
        duplicate_first_location=True,
    )
    second_context_id = "context:lsr:second-active"
    second_owner.context_thesis_id = second_context_id
    second_owner.parent_context_thesis_id = second_context_id
    second_episode = snapshot.belief.entry_episodes[
        second_owner.candidate_id
    ]
    second_episode.parent_context_thesis_id = second_context_id
    first_context = snapshot.belief.context_theses[
        first_owner.context_thesis_id
    ]
    first_context.child_episode_ids = (first_owner.episode_id,)
    snapshot.belief.context_theses[second_context_id] = SimpleNamespace(
        context_thesis_id=second_context_id,
        direction=Direction.LONG,
        authority_ids=(
            second_owner.required_root_id,
            formed.source_displacement_id,
        ),
        lifecycle="active",
        terminal_at=None,
        child_episode_ids=(second_owner.episode_id,),
    )

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    eligible = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "eligible_entry_fvg"
    )
    assert eligible.source_episode_id is None
    assert eligible.source_context_thesis_id is None
    assert eligible.entry_episode_binding_status == (
        "lsr_context_binding_ambiguous"
    )


def test_lsr_first_entry_fails_closed_when_physical_path_drifted() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "path-drift",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formation = _snapshot(source, Direction.LONG, event=False)
    owner = _with_lsr_zone_siblings(
        formation,
        (formed,),
        transition_locations=(formed,),
    )[0]
    recorder.on_bar(source)
    recorder.observe(formation, source_bar=source)

    later = _bar(source.end, high=101.0, low=99.0)
    entered = _lsr_location(
        later.end,
        "path-drift",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        formed_at=source.end,
    )
    drifted = _snapshot(later, Direction.LONG, event=False)
    _with_lsr_zone_siblings(
        drifted,
        (entered,),
        transition_locations=(entered,),
    )
    drifted.belief.lifecycle_candidate_items = (
        drifted.belief.action_candidate_items
    )
    _set_lsr_interaction(
        drifted,
        (entered,),
        path_ids=("path:lsr:drifted-physical-path",),
    )
    recorder.on_bar(later)
    recorder.observe(drifted, source_bar=later)

    first = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert first.source_episode_id == owner.episode_id
    assert first.entry_path_id == owner.entry_path_id
    assert first.entry_episode_binding_status == (
        "entry_episode_binding_invalid"
    )
    lsr = next(
        item
        for item in first.playbook_diagnostics
        if item["playbook"] == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["accepted"] is False
    assert lsr["first_failed_gate"] == "entry_episode_binding_invalid"


def test_lsr_formation_custody_roundtrips_across_compacted_root_view() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "checkpoint-compaction",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formation = _snapshot(source, Direction.LONG, event=False)
    owner = _with_lsr_zone_siblings(
        formation,
        (formed,),
        transition_locations=(formed,),
    )[0]
    recorder.on_bar(source)
    recorder.observe(formation, source_bar=source)
    restored = pickle.loads(pickle.dumps(recorder))

    later = _bar(source.end, high=101.0, low=99.0)
    entered = _lsr_location(
        later.end,
        "checkpoint-compaction",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        formed_at=source.end,
    )
    compacted = _snapshot(later, Direction.LONG, event=False)
    retained_owner = _with_lsr_zone_siblings(
        compacted,
        (entered,),
        transition_locations=(entered,),
    )[0]
    retained_owner.record_kind = "retained_episode"
    action_hypotheses = tuple(
        hypothesis
        for _, hypothesis in compacted.belief.action_candidate_items()
        if hypothesis.playbook is not Playbook.LIQUIDITY_SWEEP_REVERSAL
    )
    action_items = tuple(
        (hypothesis.candidate_id, hypothesis)
        for hypothesis in action_hypotheses
    )
    lifecycle_items = (
        *action_items,
        (retained_owner.candidate_id, retained_owner),
    )
    compacted.belief.action_candidate_items = lambda: action_items
    compacted.belief.lifecycle_candidate_items = lambda: lifecycle_items
    compacted.belief.resolve_hypothesis = dict(lifecycle_items).get
    compacted.belief.global_context.open_market_theses = ()
    restored.on_bar(later)
    restored.observe(compacted, source_bar=later)

    first = next(
        item
        for item in restored.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert first.source_episode_id == owner.episode_id
    assert first.source_context_thesis_id == owner.context_thesis_id
    assert first.entry_path_id == owner.entry_path_id
    assert first.entry_episode_binding_status == (
        "exact_retained_entry_episode_custody"
    )
    lsr = next(
        item
        for item in first.playbook_diagnostics
        if item["playbook"] == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["accepted"] is False
    assert lsr["exact_root_bound"] is False
    assert lsr["first_failed_gate"] == (
        "exact_retained_entry_episode_custody"
    )
    assert lsr["episode_id"] == owner.episode_id
    assert lsr["context_thesis_id"] == owner.context_thesis_id


def test_lsr_preformation_trigger_freezes_invalid_custody() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "stale-trigger",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formation = _snapshot(source, Direction.LONG, event=False)
    owner = _with_lsr_zone_siblings(
        formation,
        (formed,),
        transition_locations=(formed,),
    )[0]
    stale_trigger = SimpleNamespace(
        trigger_id="trigger:stale-before-formation",
        trigger_kind="qualified_rejection",
        observed_at=source.end - pd.Timedelta(minutes=1),
        setup_id=owner.episode_id,
        entry_location_id=formed.location_id,
        entry_path_id=owner.entry_path_id,
        available_trigger_kinds=("qualified_rejection",),
    )
    owner.selected_trigger = stale_trigger
    formation.belief.entry_episodes[
        owner.candidate_id
    ].selected_trigger = stale_trigger
    recorder.on_bar(source)
    recorder.observe(formation, source_bar=source)

    eligible = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "eligible_entry_fvg"
    )
    assert eligible.source_episode_id == owner.episode_id
    assert eligible.entry_episode_binding_status == (
        "entry_episode_binding_invalid"
    )

    later = _bar(source.end, high=101.0, low=99.0)
    entered = _lsr_location(
        later.end,
        "stale-trigger",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        formed_at=source.end,
    )
    clean_future = _snapshot(later, Direction.LONG, event=False)
    _with_lsr_zone_siblings(
        clean_future,
        (entered,),
        transition_locations=(entered,),
    )
    recorder.on_bar(later)
    recorder.observe(clean_future, source_bar=later)
    first = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert first.source_episode_id == owner.episode_id
    assert first.entry_episode_binding_status == (
        "entry_episode_binding_invalid"
    )


def test_lsr_first_entry_rejects_trigger_not_after_its_pullback() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "first-entry-stale-trigger",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    formation = _snapshot(source, Direction.LONG, event=False)
    owner = _with_lsr_zone_siblings(
        formation,
        (formed,),
        transition_locations=(formed,),
    )[0]
    recorder.on_bar(source)
    recorder.observe(formation, source_bar=source)

    later = _bar(source.end, high=101.0, low=99.0)
    entered = _lsr_location(
        later.end,
        "first-entry-stale-trigger",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        formed_at=source.end,
    )
    first_entry = _snapshot(later, Direction.LONG, event=False)
    current_owner = _with_lsr_zone_siblings(
        first_entry,
        (entered,),
        transition_locations=(entered,),
    )[0]
    stale_trigger = SimpleNamespace(
        trigger_id="trigger:not-after-first-pullback",
        trigger_kind="qualified_reacceptance",
        observed_at=entered.first_entered_at,
        setup_id=current_owner.episode_id,
        entry_location_id=entered.location_id,
        entry_path_id=current_owner.entry_path_id,
        available_trigger_kinds=("qualified_reacceptance",),
    )
    current_owner.selected_trigger = stale_trigger
    first_entry.belief.entry_episodes[
        current_owner.candidate_id
    ].selected_trigger = stale_trigger
    recorder.on_bar(later)
    recorder.observe(first_entry, source_bar=later)

    first = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert first.source_episode_id == owner.episode_id
    assert first.entry_episode_binding_status == (
        "entry_episode_binding_invalid"
    )


def test_lsr_ambiguous_sibling_zone_binding_fails_closed() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    entered = _lsr_location(
        source.end,
        "ambiguous",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
    )
    snapshot = _snapshot(source, Direction.LONG, event=False)
    _with_lsr_zone_siblings(
        snapshot,
        (entered, entered),
        transition_locations=(entered,),
        duplicate_first_location=True,
    )
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    candidate = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert candidate.source_episode_id is None
    assert candidate.source_playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
    assert candidate.entry_episode_binding_status == (
        "entry_episode_binding_ambiguous"
    )
    lsr = next(
        item
        for item in candidate.playbook_diagnostics
        if item["playbook"]
        == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["hypothesis_key"] is None
    assert lsr["episode_id"] is None
    assert lsr["accepted"] is False
    assert lsr["first_failed_gate"] == (
        "entry_episode_binding_ambiguous"
    )
    assert lsr["graph_connected"] is True
    assert recorder.summary["entry_episode_bindings"] == {
        "entry_episode_binding_ambiguous": 1
    }


def test_lsr_formation_keeps_missing_child_in_context_denominator() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "missing-child",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    snapshot = _snapshot(source, Direction.LONG, event=False)
    context_only = _with_lsr_zone_siblings(
        snapshot,
        (formed,),
        transition_locations=(formed,),
    )[0]
    context_only.entry_location_id = None
    context_only.entry_path_id = None
    context_only.context_metadata["lsr_entry_zone_id"] = "none"
    context_only_episode = snapshot.belief.entry_episodes[
        context_only.candidate_id
    ]
    context_only_episode.entry_location_id = None
    context_only_episode.entry_path_id = None
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    eligible = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "eligible_entry_fvg"
    )
    assert eligible.source_episode_id is None
    assert eligible.source_context_thesis_id == (
        context_only.context_thesis_id
    )
    assert eligible.entry_location_id == formed.location_id
    assert eligible.entry_episode_binding_status == (
        "entry_episode_binding_missing"
    )
    lsr = next(
        item
        for item in eligible.playbook_diagnostics
        if item["playbook"]
        == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["hypothesis_key"] is None
    assert lsr["accepted"] is False
    assert lsr["first_failed_gate"] == "entry_episode_binding_missing"
    assert recorder.summary["entry_episode_bindings"] == {
        "entry_episode_binding_missing": 1
    }


def test_lsr_formation_keeps_live_zero_child_context_in_denominator() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    formed = _lsr_location(
        source.end,
        "zero-child-context",
        lifecycle=EntryLocationLifecycle.APPROACHING,
    )
    snapshot = _snapshot(source, Direction.LONG, event=False)
    context_candidate = _with_lsr_zone_siblings(
        snapshot,
        (formed,),
        transition_locations=(formed,),
    )[0]
    context_id = context_candidate.context_thesis_id
    other = tuple(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is not Playbook.LIQUIDITY_SWEEP_REVERSAL
    )
    action_items = tuple((item.candidate_id, item) for item in other)
    snapshot.belief.candidates = lambda: other
    snapshot.belief.action_candidate_items = lambda: action_items
    snapshot.belief.lifecycle_candidate_items = lambda: action_items
    snapshot.belief.resolve_hypothesis = dict(action_items).get
    snapshot.belief.entry_episodes = {}
    snapshot.belief.context_theses[context_id].child_episode_ids = ()

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    eligible = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "eligible_entry_fvg"
    )
    assert eligible.source_episode_id is None
    assert eligible.source_context_thesis_id == context_id
    assert eligible.entry_episode_binding_status == (
        "entry_episode_binding_missing"
    )
    lsr = next(
        item
        for item in eligible.playbook_diagnostics
        if item["playbook"] == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["accepted"] is False
    assert lsr["first_failed_gate"] == "entry_episode_binding_missing"


def test_lsr_zone_binding_can_recover_one_exact_entry_path() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    entered = _lsr_location(
        source.end,
        "path-only",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
    )
    snapshot = _snapshot(source, Direction.LONG, event=False)
    sibling = _with_lsr_zone_siblings(
        snapshot,
        (entered,),
        transition_locations=(entered,),
    )[0]
    sibling.entry_location_id = None
    snapshot.belief.entry_episodes[
        sibling.candidate_id
    ].entry_location_id = None
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    candidate = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert candidate.source_episode_id == sibling.episode_id
    assert candidate.entry_path_id == sibling.entry_path_id
    assert candidate.entry_episode_binding_status == "exact_entry_path"
    lsr = next(
        item
        for item in candidate.playbook_diagnostics
        if item["playbook"]
        == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["hypothesis_key"] == sibling.candidate_id
    assert lsr["entry_episode_binding_status"] == "exact_entry_path"


def test_lsr_zone_without_context_remains_root_level_and_unbound() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    entered = _lsr_location(
        source.end,
        "no-context",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
    )
    snapshot = _snapshot(source, Direction.LONG, event=False)
    _set_lsr_interaction(snapshot, (entered,))
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    candidate = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "first_entry_fvg"
    )
    assert candidate.source_episode_id is None
    assert candidate.source_context_thesis_id is None
    assert candidate.entry_episode_binding_status == (
        "lsr_context_binding_missing"
    )
    lsr = next(
        item
        for item in candidate.playbook_diagnostics
        if item["playbook"]
        == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert lsr["hypothesis_key"] is None
    assert lsr["accepted"] is False
    assert lsr["first_failed_gate"] == "lsr_context_binding_missing"


def test_lsr_zone_binding_is_frozen_before_divergent_future_outcomes() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    entered = _lsr_location(
        source.end,
        "future-blind",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
    )
    snapshot = _snapshot(source, Direction.LONG, event=False)
    sibling = _with_lsr_zone_siblings(
        snapshot,
        (entered,),
        transition_locations=(entered,),
    )[0]
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    target_recorder = pickle.loads(pickle.dumps(recorder))
    stop_recorder = pickle.loads(pickle.dumps(recorder))

    fill = _bar(source.end, high=100.0, low=98.9, close=99.75)
    _complete_engine_step(target_recorder, fill, Direction.LONG)
    _complete_engine_step(stop_recorder, fill, Direction.LONG)
    _complete_engine_step(
        target_recorder,
        _bar(fill.end, high=104.0, low=99.0, close=103.5),
        Direction.LONG,
    )
    _complete_engine_step(
        stop_recorder,
        _bar(fill.end, high=100.0, low=97.5, close=98.0),
        Direction.LONG,
    )
    target_row = target_recorder.drain_rows()[0]
    stop_row = stop_recorder.drain_rows()[0]

    for field in (
        "event_id",
        "source_episode_id",
        "source_context_thesis_id",
        "entry_location_id",
        "entry_path_id",
        "lsr_displacement_id",
        "lsr_entry_zone_id",
        "entry_episode_binding_status",
        "observed_at",
    ):
        assert getattr(target_row, field) == getattr(stop_row, field)
    assert target_row.source_episode_id == sibling.episode_id
    assert target_row.resolution == "target_first"
    assert stop_row.resolution == "invalidation_first"
    challenge = next(
        item
        for item in derive_shadow_mechanism_challenges([target_row])
        if item.playbook == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert challenge.context_thesis_id == sibling.context_thesis_id
    assert challenge.entry_location_id == entered.location_id
    assert challenge.entry_path_id == sibling.entry_path_id
    assert challenge.lsr_displacement_id == (
        entered.source_displacement_id
    )
    assert challenge.lsr_entry_zone_id == entered.source_zone_id
    assert challenge.entry_episode_binding_status == (
        "exact_entry_location"
    )


def test_neutral_candidate_emits_one_challenge_per_active_playbook() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    _complete_engine_step(
        recorder,
        _bar(source.end, high=103.5, low=99.5),
        Direction.LONG,
    )

    rows = recorder.drain_rows()
    challenges = derive_shadow_mechanism_challenges(list(rows))
    assert len(challenges) == 2
    assert {item.playbook for item in challenges} == {
        Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
    }
    assert len({item.challenge_id for item in challenges}) == 2
    assert all(item.outcome_evaluable for item in challenges)
    assert all(item.outcome_class == "path_valid" for item in challenges)
    assert all(item.target_R == pytest.approx(1.5) for item in challenges)
    assert all(item.target_R_bucket == "1_to_lt_2R" for item in challenges)
    assert all(item.risk_qualified_target_R for item in challenges)
    assert all(
        item.planned_target_before_invalidation_deadline is True
        for item in challenges
    )
    assert all(item.hit_1_5R is True for item in challenges)
    assert len(challenges) == 2


def test_no_connected_neutral_candidate_does_not_borrow_slot_root_metadata(
) -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(source, Direction.LONG)
    active = {
        item.playbook: item
        for item in snapshot.belief.candidates()
        if item.playbook
        in {
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
        }
    }
    dfp = active[Playbook.DISPLACEMENT_FIRST_PULLBACK]
    lsr = active[Playbook.LIQUIDITY_SWEEP_REVERSAL]
    dfp.required_root_id = "unrelated-dfp-root"
    dfp.market_thesis_root_id = "unrelated-dfp-root"
    dfp.market_thesis_ids = ("unrelated-dfp-thesis",)
    dfp.market_thesis_id = "unrelated-dfp-thesis"
    dfp.bound_market_thesis_id = "unrelated-dfp-thesis"
    lsr.required_root_id = "unrelated-lsr-root"
    lsr.market_thesis_root_id = "unrelated-lsr-root"
    lsr.market_thesis_ids = ("unrelated-lsr-thesis",)
    lsr.market_thesis_id = "unrelated-lsr-thesis"
    lsr.bound_market_thesis_id = "unrelated-lsr-thesis"
    snapshot.belief.global_context.open_market_theses = ()

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=99.5),
        Direction.LONG,
    )

    row = recorder.drain_rows()[0]
    diagnostics = tuple(
        item
        for item in json.loads(row.playbook_outcomes)
        if item["runtime_enabled"]
    )
    assert len(diagnostics) == 2
    assert all(
        item["market_thesis_match_status"]
        == "no_connected_open_thesis"
        for item in diagnostics
    )
    assert all(not item["graph_connected"] for item in diagnostics)
    assert all(not item["exact_root_bound"] for item in diagnostics)
    for item in diagnostics:
        assert item["market_thesis_id"] is None
        assert item["market_thesis_root_id"] is None
        assert item["market_thesis_mechanism"] is None
        assert item["market_thesis_authority_relation"] is None
    challenges = derive_shadow_mechanism_challenges([row])
    assert len(challenges) == 2
    assert all(item.market_thesis_root_id is None for item in challenges)


def test_shadow_freezes_compact_liquidity_route_for_independent_root() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(source, Direction.LONG)
    dfp = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    dfp.liquidity_route = SimpleNamespace(
        route_id="route:dfp:long",
        context_draw_id="draw:h4:terminal",
        intermediate_liquidity_ids=("draw:m5:waypoint", "draw:h1:waypoint"),
        primary_deliverable_target_id="draw:h1:primary",
        terminal_draw_id="draw:h4:terminal",
        authority_barrier_id="obstruction:h4:protected",
        authority_barrier_price=104.25,
    )
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=99.5),
        Direction.LONG,
    )

    row = recorder.drain_rows()[0]
    diagnostics = {
        item["playbook"]: item
        for item in json.loads(row.playbook_outcomes)
    }
    frozen = diagnostics[Playbook.DISPLACEMENT_FIRST_PULLBACK.value]
    assert frozen["liquidity_route_id"] == "route:dfp:long"
    assert frozen["context_draw_id"] == "draw:h4:terminal"
    assert frozen["intermediate_liquidity_ids"] == [
        "draw:m5:waypoint",
        "draw:h1:waypoint",
    ]
    assert frozen["primary_deliverable_target_id"] == "draw:h1:primary"
    assert frozen["terminal_draw_id"] == "draw:h4:terminal"
    assert frozen["authority_barrier_id"] == "obstruction:h4:protected"
    assert frozen["authority_barrier_price"] == pytest.approx(104.25)

    challenges = derive_shadow_mechanism_challenges([row])
    dfp_challenge = next(
        item
        for item in challenges
        if item.playbook == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    assert json.loads(dfp_challenge.intermediate_liquidity_ids) == [
        "draw:m5:waypoint",
        "draw:h1:waypoint",
    ]
    assert dfp_challenge.authority_barrier_id == "obstruction:h4:protected"
    assert dfp_challenge.authority_barrier_price == pytest.approx(104.25)

    root = derive_shadow_root_episode_records(challenges)[0]
    routes = json.loads(root.playbook_liquidity_routes)
    assert routes[Playbook.DISPLACEMENT_FIRST_PULLBACK.value] == {
        "authority_barrier_id": "obstruction:h4:protected",
        "authority_barrier_price": 104.25,
        "context_draw_id": "draw:h4:terminal",
        "intermediate_liquidity_ids": [
            "draw:m5:waypoint",
            "draw:h1:waypoint",
        ],
        "liquidity_route_id": "route:dfp:long",
        "primary_deliverable_target_id": "draw:h1:primary",
        "terminal_draw_id": "draw:h4:terminal",
    }


def test_open_market_thesis_registers_once_per_root_evidence_revision() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    first = _add_open_thesis_revision(
        _snapshot(source, Direction.LONG, event=False),
        "evidence:r1",
    )
    recorder.on_bar(source)
    recorder.observe(first, source_bar=source)

    candidates = tuple(
        item
        for item in recorder.open_candidates
        if item.candidate_origin == "open_market_thesis_revision"
    )
    assert len(candidates) == 1
    assert candidates[0].event_kind == "open_market_thesis_revision"
    assert candidates[0].event_id == "displacement:long|evidence:r1"

    later = _bar(source.end, high=101.0, low=99.0)
    repeated = _add_open_thesis_revision(
        _snapshot(later, Direction.LONG, event=False),
        "evidence:r1",
        updated_at=source.end,
    )
    recorder.on_bar(later)
    recorder.observe(repeated, source_bar=later)
    assert recorder.summary["candidate_events"][
        "open_market_thesis_revision"
    ] == 1

    revision_bar = _bar(later.end, high=101.0, low=99.0)
    revised = _add_open_thesis_revision(
        _snapshot(revision_bar, Direction.LONG, event=False),
        "evidence:r2",
    )
    recorder.on_bar(revision_bar)
    recorder.observe(revised, source_bar=revision_bar)
    assert recorder.summary["candidate_events"][
        "open_market_thesis_revision"
    ] == 2
    assert {
        item.event_id
        for item in recorder.open_candidates
        if item.candidate_origin == "open_market_thesis_revision"
    } == {
        "displacement:long|evidence:r1",
        "displacement:long|evidence:r2",
    }


def test_invalidated_open_thesis_revision_does_not_open_shadow_candidate() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _add_open_thesis_revision(
        _snapshot(source, Direction.LONG, event=False),
        "evidence:terminal",
    )
    snapshot.belief.global_context.open_market_theses[0].lifecycle = (
        "invalidated"
    )

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    assert all(
        item.candidate_origin != "open_market_thesis_revision"
        for item in recorder.open_candidates
    )


def test_open_thesis_revision_freezes_exact_root_candidate_plan() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _add_open_thesis_revision(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        "evidence:planned",
    )
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    candidate = next(
        item
        for item in recorder.open_candidates
        if item.candidate_origin == "open_market_thesis_revision"
    )
    assert candidate.source_playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    assert candidate.entry_reference_price == 100.0
    assert candidate.invalidation_price == 98.0
    assert candidate.invalidation_source_id == "structure:long"
    assert candidate.target_price == 103.0
    assert candidate.draw_id == "draw:long"
    assert candidate.deadline_at == source.end + pd.Timedelta(minutes=60)
    assert candidate.geometry_incomplete_reason is None


def test_window_visible_open_thesis_with_warmup_deadline_expires_at_observation() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    warmup = _bar(_start() - pd.Timedelta(days=4))
    warmup_snapshot = _add_open_thesis_revision(
        _snapshot(warmup, Direction.LONG, event=False),
        "evidence:warmup",
    )
    recorder.on_bar(warmup)
    recorder.prime(warmup_snapshot, source_bar=warmup)

    source = _bar(_start())
    snapshot = _add_open_thesis_revision(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        "evidence:first-window-revision",
    )
    hypothesis = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    hypothesis.phase = PlaybookPhase.FORMING
    hypothesis.plan.deadline = warmup.end + pd.Timedelta(minutes=60)

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    row = next(
        item
        for item in recorder.drain_rows()
        if item.event_kind == "open_market_thesis_revision"
    )
    assert row.deadline_at < row.observed_at
    assert row.resolved_at == row.observed_at
    assert row.resolution == "entry_unfilled_deadline"
    assert row.filled is False
    assert row.censored is False
    assert recorder.open_candidates == ()


def test_warmup_deadline_boundary_is_pickle_resume_identical() -> None:
    uninterrupted = ShadowCandidateOutcomeRecorder()
    warmup = _bar(_start() - pd.Timedelta(days=4))
    warmup_snapshot = _add_open_thesis_revision(
        _snapshot(warmup, Direction.SHORT, event=False),
        "evidence:warmup",
    )
    uninterrupted.on_bar(warmup)
    uninterrupted.prime(warmup_snapshot, source_bar=warmup)
    restored = pickle.loads(pickle.dumps(uninterrupted))

    source = _bar(_start())
    snapshot = _add_open_thesis_revision(
        _snapshot(
            source,
            Direction.SHORT,
            event=False,
            executable_plan=True,
        ),
        "evidence:first-window-revision",
    )
    hypothesis = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    hypothesis.phase = PlaybookPhase.FORMING
    hypothesis.plan.deadline = warmup.end + pd.Timedelta(minutes=60)

    for recorder in (uninterrupted, restored):
        recorder.on_bar(source)
        recorder.observe(snapshot, source_bar=source)

    uninterrupted_rows = uninterrupted.drain_rows()
    restored_rows = restored.drain_rows()
    assert uninterrupted_rows == restored_rows
    assert all(
        row.observed_at <= row.resolved_at
        for row in uninterrupted_rows
    )
    assert uninterrupted.summary == restored.summary


def test_pre_fix_stale_open_candidate_resumes_at_observation_clock() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    stale = recorder.open_candidates[0]
    stale.deadline_at = source.start - pd.Timedelta(minutes=1)
    restored = pickle.loads(pickle.dumps(recorder))

    later = _bar(source.end, high=101.0, low=99.0)
    _complete_engine_step(restored, later, Direction.LONG)

    rows = restored.drain_rows()
    row = next(
        item
        for item in rows
        if item.candidate_id == stale.candidate_id
    )
    assert row.deadline_at < row.observed_at
    assert row.resolved_at == row.observed_at
    assert row.resolution == "entry_unfilled_deadline"
    assert row.filled is False
    assert all(
        item.observed_at <= item.resolved_at
        for item in rows
    )


def test_shadow_resolution_rejects_a_clock_before_candidate_observation() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    candidate = recorder.open_candidates[0]

    with pytest.raises(
        ValueError,
        match="resolution cannot precede observation",
    ):
        recorder._resolve(
            candidate,
            source.start,
            "entry_unfilled_deadline",
            censored=False,
        )

    assert recorder.open_candidates == (candidate,)


def test_open_thesis_shadow_binds_only_the_frozen_selected_draw() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _add_open_thesis_revision(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        "evidence:draw-selection",
    )
    thesis = snapshot.belief.global_context.open_market_theses[0]
    thesis.draw_candidate_ids = ("draw:long", "draw:unselected")
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    candidate = next(
        item
        for item in recorder.open_candidates
        if item.candidate_origin == "open_market_thesis_revision"
    )
    assert "draw:long" in candidate.source_ids
    assert "draw:unselected" not in candidate.source_ids

    harmless = _bar(source.end, high=101.0, low=99.0)
    _complete_engine_step(
        recorder,
        harmless,
        Direction.LONG,
        executable_plan=True,
        invalidated_source_ids=("draw:unselected",),
    )
    assert candidate in recorder.open_candidates

    selected_invalidated = _bar(harmless.end, high=101.0, low=99.0)
    _complete_engine_step(
        recorder,
        selected_invalidated,
        Direction.LONG,
        executable_plan=True,
        invalidated_source_ids=("draw:long",),
    )
    row = next(
        item
        for item in recorder.drain_rows()
        if item.candidate_id == candidate.candidate_id
    )
    assert row.resolution == "source_identity_invalidated"


def test_open_thesis_missing_draw_fails_without_unlinked_fallback() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _add_open_thesis_revision(
        _snapshot(source, Direction.LONG, event=False),
        "evidence:no-draw",
        include_draw=False,
    )
    # A visible market draw exists in Observation, but it is deliberately not
    # linked to this thesis revision and therefore cannot complete geometry.
    assert snapshot.observation.liquidity_inventory
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    row = next(
        item
        for item in recorder.rows
        if item.candidate_origin == "open_market_thesis_revision"
    )
    assert row.geometry_complete is False
    assert row.geometry_incomplete_reason == "missing_draw"
    assert row.resolution == "geometry_incomplete"
    assert row.invalidation_price == 98.0
    assert row.draw_id is None
    assert row.target_price is None
    challenges = derive_shadow_mechanism_challenges([row])
    assert challenges
    assert all(
        item.geometry_incomplete_reason == "missing_draw"
        for item in challenges
    )


@pytest.mark.parametrize(
    ("target_r", "expected"),
    (
        (None, "unavailable"),
        (0.49, "lt_0_5R"),
        (0.5, "0_5_to_lt_1R"),
        (1.0, "1_to_lt_2R"),
        (2.0, "2_to_lt_3R"),
        (3.0, "ge_3R"),
    ),
)
def test_target_r_buckets_are_frozen_and_disjoint(
    target_r: float | None,
    expected: str,
) -> None:
    assert _target_r_bucket(target_r) == expected


def test_same_bar_conservative_outcome_propagates_to_challenges() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=97.0),
        Direction.LONG,
    )

    challenges = derive_shadow_mechanism_challenges(
        list(recorder.drain_rows())
    )
    assert len(challenges) == 2
    assert all(item.outcome_evaluable for item in challenges)
    assert all(item.outcome_class == "path_failed" for item in challenges)
    assert all(item.same_bar_collision for item in challenges)
    assert all(
        item.planned_target_before_invalidation_deadline is False
        for item in challenges
    )
    assert all(item.mfe_R == 0.0 for item in challenges)
    assert all(item.hit_0_5R is False for item in challenges)
    assert all(item.hit_1_5R is False for item in challenges)


def test_missing_transition_delta_fails_closed() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(source, Direction.LONG, typed_delta=False),
        source_bar=source,
    )

    assert recorder.open_candidates == ()
    assert recorder.summary["typed_delta_missing_observations"] == 1


def test_secondary_connected_thesis_is_not_reported_as_exact_bound() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(source, Direction.LONG)
    dfp = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    dfp.market_thesis_ids = ("thesis:primary", "thesis:secondary")
    dfp.market_thesis_id = "thesis:primary"
    dfp.bound_market_thesis_id = "thesis:primary"
    snapshot.belief.global_context.open_market_theses = (
        SimpleNamespace(
            thesis_id="thesis:primary",
            root_id="some-other-root",
            direction=Direction.LONG,
            mechanism_event_ids=("some-other-root",),
            entry_location_ids=(),
            trigger_event_ids=(),
            mechanism="directional_displacement_continuation",
            authority_relation="aligned",
            authority_source_ids=("structure:long",),
            source_timeframe=Timeframe.H1,
        ),
        SimpleNamespace(
            thesis_id="thesis:secondary",
            root_id="displacement:long",
            direction=Direction.LONG,
            mechanism_event_ids=("displacement:long",),
            entry_location_ids=(),
            trigger_event_ids=(),
            mechanism="directional_displacement_continuation",
            authority_relation="aligned",
            authority_source_ids=("structure:long",),
            source_timeframe=Timeframe.H1,
        ),
    )
    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=99.5),
        Direction.LONG,
    )

    row = recorder.drain_rows()[0]
    outcomes = json.loads(row.playbook_outcomes)
    diagnostic = next(
        item
        for item in outcomes
        if item["playbook"]
        == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    assert diagnostic["market_thesis_id"] == "thesis:secondary"
    assert diagnostic["market_thesis_match_status"] == "matched_root_unbound"
    assert diagnostic["selected_hypothesis_match_strength"] is None
    assert diagnostic["accepted"] is False
    assert diagnostic["first_failed_gate"] == (
        "market_thesis_exact_root_binding"
    )
    challenges = derive_shadow_mechanism_challenges(
        [
            replace(row, playbook_outcomes=json.dumps(outcomes))
        ]
    )
    dfp_challenge = next(
        item
        for item in challenges
        if item.playbook == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    assert dfp_challenge.first_failed_gate == (
        "market_thesis_exact_root_binding"
    )


def test_shared_evidence_cannot_override_candidate_canonical_root() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(source, Direction.LONG)
    dfp = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    dfp.market_thesis_ids = ("thesis:primary", "thesis:secondary")
    dfp.market_thesis_id = "thesis:primary"
    dfp.bound_market_thesis_id = "thesis:primary"
    snapshot.belief.global_context.open_market_theses = (
        SimpleNamespace(
            thesis_id="thesis:primary",
            root_id="primary-root",
            direction=Direction.LONG,
            # Both theses contain this displacement, but it is not the
            # canonical root of the primary episode.
            mechanism_event_ids=("primary-root", "displacement:long"),
            entry_location_ids=(),
            trigger_event_ids=(),
            mechanism="directional_displacement_continuation",
            authority_relation="aligned",
            authority_source_ids=("structure:long",),
            source_timeframe=Timeframe.H1,
        ),
        SimpleNamespace(
            thesis_id="thesis:secondary",
            root_id="displacement:long",
            direction=Direction.LONG,
            mechanism_event_ids=("displacement:long",),
            entry_location_ids=(),
            trigger_event_ids=(),
            mechanism="directional_displacement_continuation",
            authority_relation="aligned",
            authority_source_ids=("structure:long",),
            source_timeframe=Timeframe.H1,
        ),
    )

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=99.5),
        Direction.LONG,
    )

    row = recorder.drain_rows()[0]
    diagnostic = next(
        item
        for item in json.loads(row.playbook_outcomes)
        if item["playbook"]
        == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    assert diagnostic["market_thesis_id"] == "thesis:secondary"
    assert diagnostic["market_thesis_root_id"] == "displacement:long"
    assert diagnostic["market_thesis_match_status"] == "matched_root_unbound"
    assert diagnostic["exact_root_bound"] is False
    assert diagnostic["accepted"] is False
    assert diagnostic["first_failed_gate"] == (
        "market_thesis_exact_root_binding"
    )


def test_first_executable_episode_registers_one_normal_playbook_candidate() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    first = _snapshot(
        source,
        Direction.LONG,
        event=False,
        executable_plan=True,
    )
    first_dfp = next(
        item
        for item in first.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    first_dfp.selected_trigger = SimpleNamespace(
        trigger_kind="micro_bos",
        trigger_id="trigger:dfp",
        observed_at=source.end,
        available_trigger_kinds=(
            "qualified_reacceptance",
            "micro_bos",
        ),
    )
    recorder.on_bar(source)
    recorder.observe(first, source_bar=source)
    assert len(recorder.open_candidates) == 1
    assert recorder.open_candidates[0].event_kind == "playbook_executable"

    later = _bar(source.end, high=101.0, low=99.0)
    recorder.on_bar(later)
    repeated = _snapshot(
        later,
        Direction.LONG,
        event=False,
        executable_plan=True,
    )
    recorder.observe(repeated, source_bar=later)
    assert len(recorder.open_candidates) == 1

    _complete_engine_step(
        recorder,
        _bar(later.end, high=104.0, low=99.5),
        Direction.LONG,
        executable_plan=True,
    )
    row = recorder.drain_rows()[0]
    assert row.event_kind == "playbook_executable"
    assert row.entry_rule == SHADOW_OUTCOME_PROTOCOL["playbook_entry_rule"]
    assert row.entry_reference_price == 100.0
    assert row.deadline_at == source.end + pd.Timedelta(minutes=60)
    assert row.source_timeframe == Timeframe.H1.value
    diagnostic = next(
        item
        for item in json.loads(row.playbook_outcomes)
        if item["playbook"]
        == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    assert diagnostic["graph_connected"] is True
    assert diagnostic["exact_root_bound"] is True
    assert diagnostic["market_thesis_mechanism"] == (
        "directional_displacement_continuation"
    )
    assert diagnostic["market_thesis_authority_relation"] == "aligned"
    episode = derive_shadow_episode_outcomes([row])
    assert len(episode) == 1
    episode_row = episode[0]
    assert episode_row.episode_key == (
        "displacement_first_pullback|long|episode:dfp:long"
    )
    assert episode_row.first_executable_at == source.end
    assert episode_row.selected_trigger_kind == "micro_bos"
    assert episode_row.selected_trigger_id == "trigger:dfp"
    assert episode_row.selected_trigger_at == source.end
    assert json.loads(episode_row.available_trigger_kinds) == [
        "qualified_reacceptance",
        "micro_bos",
    ]
    assert episode_row.target_R == pytest.approx(1.5)
    assert episode_row.frozen_target_R == pytest.approx(1.5)
    assert episode_row.target_R_bucket == "1_to_lt_2R"
    assert episode_row.risk_qualified_target_R is True
    assert episode_row.outcome_evaluable is True
    assert episode_row.outcome_class == "path_valid"
    assert episode_row.target_first is True
    assert episode_row.invalidation_first is False
    assert episode_row.expired is False
    assert episode_row.same_bar_stop_first is False
    assert episode_row.hit_1_5R is True
    assert len(episode) == 1

    # The same stable episode may remain executable, but it cannot emit a
    # second episode-level outcome.
    repeated_after_terminal = _bar(later.end + pd.Timedelta(minutes=1))
    _complete_engine_step(
        recorder,
        repeated_after_terminal,
        Direction.LONG,
        executable_plan=True,
    )
    assert recorder.drain_rows() == ()
    with pytest.raises(ValueError, match="duplicate first-executable"):
        derive_shadow_episode_outcomes([row, row])

    # Rearm is a new causal episode, so the same playbook/direction may own
    # one additional row without weakening the episode-level uniqueness key.
    rearmed_diagnostics = json.loads(row.playbook_outcomes)
    for item in rearmed_diagnostics:
        if item["playbook"] == Playbook.DISPLACEMENT_FIRST_PULLBACK.value:
            item["episode_id"] = "episode:dfp:rearmed"
            item["setup_id"] = "episode:dfp:rearmed"
    rearmed_clock = row.observed_at + pd.Timedelta(minutes=5)
    rearmed_row = replace(
        row,
        candidate_id="shadow:playbook-rearmed",
        event_id=(
            "displacement_first_pullback:long:episode:dfp:rearmed"
        ),
        source_episode_id="episode:dfp:rearmed",
        source_setup_id="episode:dfp:rearmed",
        selected_trigger_id="trigger:dfp:rearmed",
        selected_trigger_at=rearmed_clock,
        observed_at=rearmed_clock,
        resolved_at=rearmed_clock + pd.Timedelta(minutes=2),
        playbook_outcomes=json.dumps(rearmed_diagnostics),
    )
    rearmed_episodes = derive_shadow_episode_outcomes([row, rearmed_row])
    assert {item.episode_id for item in rearmed_episodes} == {
        "episode:dfp:long",
        "episode:dfp:rearmed",
    }


def _freeze_root_absent_executable_lifecycle(
    snapshot: SimpleNamespace,
) -> SimpleNamespace:
    candidate = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    context_id = "context:dfp:long"
    candidate.plan.setup_id = candidate.episode_id
    candidate.setup_context_id = candidate.episode_id
    candidate.entry_location_id = candidate.plan.entry_location_id
    candidate.entry_path_id = candidate.plan.entry_path_id
    candidate.context_thesis_id = context_id
    candidate.parent_context_thesis_id = context_id
    candidate.record_kind = "root_candidate"
    candidate.market_thesis_match_status = "exact_root_bound"
    candidate.market_thesis_mechanism = (
        "directional_displacement_continuation"
    )
    candidate.market_thesis_authority_relation = "aligned"
    snapshot.belief.entry_episodes = {
        candidate.candidate_id: SimpleNamespace(
            candidate_id=candidate.candidate_id,
            episode_id=candidate.episode_id,
            parent_context_thesis_id=context_id,
            playbook=candidate.playbook,
            direction=candidate.direction,
            entry_location_id=candidate.entry_location_id,
            entry_path_id=candidate.entry_path_id,
            phase=PlaybookPhase.EXECUTABLE,
            plan=candidate.plan,
        )
    }
    snapshot.belief.context_theses = {
        context_id: SimpleNamespace(
            context_thesis_id=context_id,
            lifecycle="active",
            terminal_at=None,
            child_episode_ids=(candidate.episode_id,),
        )
    }
    snapshot.belief.global_context.open_market_theses = ()
    return candidate


def test_root_absent_executable_uses_only_exact_frozen_lifecycle_binding() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(
        source,
        Direction.LONG,
        event=False,
        executable_plan=True,
    )
    candidate = _freeze_root_absent_executable_lifecycle(snapshot)

    spec = recorder._playbook_candidate_specs(snapshot)[0]
    assert candidate.entry_path_id in spec.source_ids
    diagnostics = recorder._playbook_diagnostics(snapshot, spec)
    diagnostic = next(
        item
        for item in diagnostics
        if item["playbook"] == candidate.playbook.value
    )

    assert diagnostic["accepted"] is True
    assert diagnostic["graph_connected"] is True
    assert diagnostic["exact_root_bound"] is True
    assert diagnostic["market_thesis_id"] == candidate.market_thesis_id
    assert diagnostic["market_thesis_root_id"] == candidate.required_root_id


def test_root_absent_executable_identity_mismatch_fails_closed() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(
        source,
        Direction.LONG,
        event=False,
        executable_plan=True,
    )
    candidate = _freeze_root_absent_executable_lifecycle(snapshot)
    episode = snapshot.belief.entry_episodes[candidate.candidate_id]
    episode.entry_path_id = "path:other"
    spec = recorder._playbook_candidate_specs(snapshot)[0]

    with pytest.raises(
        ValueError,
        match="exact live frozen Context/EntryEpisode binding",
    ):
        recorder._playbook_diagnostics(snapshot, spec)


def test_executable_finalizer_rejects_mismatched_frozen_diagnostic() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        source_bar=source,
    )
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=99.5),
        Direction.LONG,
        executable_plan=True,
    )
    row = recorder.drain_rows()[0]
    diagnostics = json.loads(row.playbook_outcomes)
    source_diagnostic = next(
        item
        for item in diagnostics
        if item["playbook"] == row.source_playbook
    )
    source_diagnostic["episode_id"] = "wrong-episode"
    corrupted = replace(
        row,
        playbook_outcomes=json.dumps(diagnostics),
    )

    with pytest.raises(ValueError, match="exactly one matching episode"):
        derive_shadow_episode_outcomes([corrupted])


def test_two_executable_roots_register_two_independent_action_candidates() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(
        source,
        Direction.LONG,
        event=False,
        executable_plan=True,
    )
    primary = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    secondary_root = "displacement:long:secondary"
    secondary_thesis_id = "thesis:long:secondary"
    secondary = SimpleNamespace(**vars(primary))
    secondary.candidate_id = "candidate:dfp:long:secondary"
    secondary.key = secondary.candidate_id
    secondary.required_root_id = secondary_root
    secondary.market_thesis_root_id = secondary_root
    secondary.market_thesis_ids = (secondary_thesis_id,)
    secondary.market_thesis_id = secondary_thesis_id
    secondary.bound_market_thesis_id = secondary_thesis_id
    secondary.episode_id = "episode:dfp:secondary"
    secondary.setup_context_id = "setup:dfp:secondary"
    secondary.plan = SimpleNamespace(**vars(primary.plan))
    secondary.plan.setup_id = "setup:dfp:secondary"
    action_items = tuple(snapshot.belief.action_candidate_items()) + (
        (secondary.candidate_id, secondary),
    )
    action_map = dict(action_items)
    snapshot.belief.action_candidate_items = lambda: action_items
    snapshot.belief.resolve_hypothesis = action_map.get
    primary_thesis = snapshot.belief.global_context.open_market_theses[0]
    snapshot.belief.global_context.open_market_theses = (
        primary_thesis,
        SimpleNamespace(
            thesis_id=secondary_thesis_id,
            root_id=secondary_root,
            direction=Direction.LONG,
            mechanism_event_ids=(secondary_root,),
            entry_location_ids=(),
            trigger_event_ids=(),
            mechanism="directional_displacement_continuation",
            authority_relation="aligned",
            authority_source_ids=("structure:long",),
            source_timeframe=Timeframe.H1,
        ),
    )

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    executable = tuple(
        item
        for item in recorder.open_candidates
        if item.event_kind == "playbook_executable"
    )
    assert len(executable) == 2
    assert {item.event_id for item in executable} == {
        primary.candidate_id,
        secondary.candidate_id,
    }
    assert {item.source_episode_id for item in executable} == {
        primary.episode_id,
        secondary.episode_id,
    }
    for item in executable:
        diagnostics = tuple(
            diagnostic
            for diagnostic in item.playbook_diagnostics
            if diagnostic["playbook"]
            == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
            and diagnostic.get("episode_id") == item.source_episode_id
        )
        assert len(diagnostics) == 1
        assert diagnostics[0]["accepted"] is True
        assert diagnostics[0]["exact_root_bound"] is True


def test_executable_root_binding_does_not_fall_back_to_ambiguous_source_ids() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(
        source,
        Direction.LONG,
        event=False,
        executable_plan=True,
    )
    primary = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    secondary_root = "displacement:long:competing"
    secondary_thesis_id = "thesis:long:competing"
    secondary = SimpleNamespace(**vars(primary))
    secondary.candidate_id = "candidate:dfp:long:competing"
    secondary.key = secondary.candidate_id
    secondary.required_root_id = secondary_root
    secondary.market_thesis_root_id = secondary_root
    secondary.market_thesis_ids = (secondary_thesis_id,)
    secondary.market_thesis_id = secondary_thesis_id
    secondary.bound_market_thesis_id = secondary_thesis_id
    secondary.episode_id = "episode:dfp:competing"
    secondary.setup_context_id = "setup:dfp:competing"
    secondary.plan = SimpleNamespace(**vars(primary.plan))
    secondary.plan.setup_id = "setup:dfp:competing"
    action_items = tuple(snapshot.belief.action_candidate_items()) + (
        (secondary.candidate_id, secondary),
    )
    action_map = dict(action_items)
    snapshot.belief.action_candidate_items = lambda: action_items
    snapshot.belief.resolve_hypothesis = action_map.get
    primary_thesis = snapshot.belief.global_context.open_market_theses[0]
    competing_thesis = SimpleNamespace(
        thesis_id=secondary_thesis_id,
        root_id=secondary_root,
        direction=Direction.LONG,
        mechanism_event_ids=(secondary_root,),
        entry_location_ids=(),
        trigger_event_ids=(),
        mechanism="directional_displacement_continuation",
        authority_relation="aligned",
        authority_source_ids=("structure:long",),
        source_timeframe=Timeframe.H1,
    )
    snapshot.belief.global_context.open_market_theses = (
        primary_thesis,
        competing_thesis,
    )

    specs = recorder._playbook_candidate_specs(snapshot)
    primary_spec = next(
        item for item in specs if item.event_id == primary.candidate_id
    )
    # Reproduce the production ambiguity: a frozen executable can carry its
    # own root plus identities causally connected to a competing root.  Its
    # action-candidate ID, not this broad set, owns the diagnostic binding.
    primary_spec = replace(
        primary_spec,
        source_ids=tuple(
            dict.fromkeys(
                (*primary_spec.source_ids, secondary_root, secondary_thesis_id)
            )
        ),
    )
    diagnostics = recorder._playbook_diagnostics(snapshot, primary_spec)
    matching = tuple(
        item
        for item in diagnostics
        if item["playbook"] == primary.playbook.value
        and item.get("episode_id") == primary.episode_id
    )

    assert len(matching) == 1
    assert matching[0]["hypothesis_key"] == primary.candidate_id
    assert matching[0]["market_thesis_root_id"] == primary.required_root_id
    assert matching[0]["accepted"] is True
    assert matching[0]["exact_root_bound"] is True


def test_invalid_plan_feasibility_is_shadowed_but_not_executable() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _add_open_thesis_revision(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        "evidence:invalid-plan",
    )
    dfp = next(
        item
        for item in snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    dfp.plan_feasibility = SimpleNamespace(
        valid=False,
        failure_reason="hard_obstruction_before_target",
    )

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    assert all(
        item.event_kind != "playbook_executable"
        for item in recorder.open_candidates
    )
    thesis_candidate = next(
        item
        for item in recorder.open_candidates
        if item.event_kind == "open_market_thesis_revision"
    )
    diagnostic = next(
        item
        for item in thesis_candidate.playbook_diagnostics
        if item["playbook"]
        == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    assert diagnostic["plan_feasibility_valid"] is False
    assert diagnostic["plan_feasibility_failure_reason"] == (
        "hard_obstruction_before_target"
    )
    assert diagnostic["plan_delivery_valid"] is False
    assert diagnostic["accepted"] is False
    assert diagnostic["first_failed_gate"] == (
        "hard_obstruction_before_target"
    )


def test_shadow_diagnostic_freezes_event_order_from_candidate_clock() -> None:
    challenges = _valid_neutral_challenges()
    assert challenges
    assert {
        tuple(json.loads(item.event_order_signature))
        for item in challenges
    } == {("market_root_formed", "directional_reaction")}


def test_root_episode_collapses_revisions_before_motif_aggregation() -> None:
    challenges = _valid_neutral_challenges()
    first = challenges[0]
    revised_at = first.observed_at + pd.Timedelta(minutes=5)
    revisions = tuple(challenges) + tuple(
        replace(
            item,
            challenge_id=item.challenge_id + ":revision",
            candidate_id=item.candidate_id + ":revision",
            observed_at=revised_at,
            resolved_at=revised_at + pd.Timedelta(minutes=1),
        )
        for item in challenges
    )

    episodes = derive_shadow_root_episode_records(revisions)

    assert len(episodes) == 1
    assert episodes[0].candidate_id == first.candidate_id
    assert episodes[0].eligible_episode_evidence is True
    assert episodes[0].dfp_rejected and episodes[0].lsr_rejected
    assert episodes[0].action_authority is False


def test_root_sequence_keeps_outcome_representative_separate_and_child_scoped() -> None:
    base = next(
        item
        for item in _valid_neutral_challenges()
        if item.playbook == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    first_at = base.observed_at
    complete = (
        "h4_structure_and_draw",
        "m5_displacement_zone",
        "first_pullback_to_frozen_zone",
        "typed_entry_trigger",
    )
    first = replace(
        base,
        challenge_id="challenge:episode-a:first",
        candidate_id="candidate:episode-a:first",
        event_kind="confirmed_bos",
        episode_id="episode-a",
        setup_id="setup-a",
        accepted_at_candidate_clock=False,
        first_failed_gate="m5_displacement_zone",
        failed_gates=json.dumps(["m5_displacement_zone"]),
        event_order_signature=json.dumps([complete[0]]),
    )
    deepest = replace(
        first,
        challenge_id="challenge:episode-a:deepest",
        candidate_id="candidate:episode-a:deepest",
        event_kind="qualified_micro_bos",
        observed_at=first_at + pd.Timedelta(minutes=5),
        resolved_at=first_at + pd.Timedelta(minutes=8),
        accepted_at_candidate_clock=True,
        first_failed_gate=None,
        failed_gates="[]",
        event_order_signature=json.dumps(complete),
    )
    sibling = replace(
        first,
        challenge_id="challenge:episode-b",
        candidate_id="candidate:episode-b",
        episode_id="episode-b",
        setup_id="setup-b",
        observed_at=first_at + pd.Timedelta(minutes=3),
        resolved_at=first_at + pd.Timedelta(minutes=4),
        event_order_signature=json.dumps(complete[:2]),
    )

    rows = derive_shadow_root_sequence_records((deepest, sibling, first))

    assert len(rows) == 2
    episode_a = next(item for item in rows if item.episode_id == "episode-a")
    assert episode_a.scope_kind == "entry_episode"
    assert episode_a.revision_count == 2
    assert json.loads(episode_a.event_order_signature) == list(complete)
    assert json.loads(episode_a.observed_sequence_signatures) == [
        [complete[0]],
        list(complete),
    ]
    assert json.loads(episode_a.event_bigrams) == [
        list(complete[index : index + 2])
        for index in range(len(complete) - 1)
    ]
    assert json.loads(episode_a.event_trigrams) == [
        list(complete[index : index + 3])
        for index in range(len(complete) - 2)
    ]
    assert json.loads(episode_a.primitive_event_counts) == {
        "confirmed_bos": 1,
        "qualified_micro_bos": 1,
    }
    assert set(json.loads(episode_a.primitive_first_observed_at)) == {
        "confirmed_bos",
        "qualified_micro_bos",
    }
    assert set(json.loads(episode_a.step_first_observed_at)) == set(complete)
    assert episode_a.complete_sequence_observed is True
    assert episode_a.ever_accepted is True
    assert episode_a.first_accepted_at == deepest.observed_at
    assert episode_a.first_failed_gate == "m5_displacement_zone"
    assert episode_a.first_failed_gate_at == first.observed_at
    assert episode_a.lifecycle_terminal_status == "unknown_not_recorded"
    assert episode_a.action_authority is False
    assert {item.episode_id for item in rows} == {"episode-a", "episode-b"}
    assert derive_shadow_root_sequence_records(
        tuple(reversed((deepest, sibling, first)))
    ) == rows

    # Changing every later path-result field cannot alter a candidate-clock
    # lifecycle sequence.  The outcome representative remains a separate row.
    outcome_changed = tuple(
        replace(
            item,
            resolved_at=item.resolved_at + pd.Timedelta(days=2),
            target_price=999.0,
            invalidation_price=1.0,
            target_R=998.0,
            target_R_bucket="gte_3R",
            risk_qualified_target_R=True,
            planned_target_before_invalidation_deadline=False,
            filled=False,
            outcome_evaluable=False,
            outcome_class="censored",
            resolution="data_boundary",
            censored=True,
            mfe_R=99.0,
            mae_R=99.0,
        )
        for item in (deepest, sibling, first)
    )
    assert derive_shadow_root_sequence_records(outcome_changed) == rows


def test_root_sequence_separates_same_root_across_market_thesis_epochs() -> None:
    base = next(
        item
        for item in _valid_neutral_challenges()
        if item.playbook == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    first = replace(
        base,
        challenge_id="challenge:epoch-a",
        candidate_id="candidate:epoch-a",
        episode_id=None,
        setup_id=None,
        market_thesis_id="market-thesis:epoch-a",
        event_order_signature=json.dumps(["h4_structure_and_draw"]),
    )
    second = replace(
        first,
        challenge_id="challenge:epoch-b",
        candidate_id="candidate:epoch-b",
        market_thesis_id="market-thesis:epoch-b",
        observed_at=first.observed_at + pd.Timedelta(minutes=1),
        resolved_at=first.resolved_at + pd.Timedelta(minutes=1),
    )

    rows = derive_shadow_root_sequence_records((first, second))

    assert len(rows) == 2
    assert {item.root_id for item in rows} == {first.market_thesis_root_id}
    assert {item.scope_kind for item in rows} == {"context_thesis"}
    assert {item.scope_id for item in rows} == {
        "market-thesis:epoch-a",
        "market-thesis:epoch-b",
    }


def test_root_episode_does_not_promote_unbound_eye_candidate() -> None:
    challenges = tuple(
        replace(
            item,
            market_thesis_root_id=None,
            market_thesis_id=None,
            exact_root_bound=False,
            market_thesis_match_status="no_open_thesis",
        )
        for item in _valid_neutral_challenges()
    )

    assert derive_shadow_root_episode_records(challenges) == ()


def test_motif_requires_two_distinct_dates_and_joint_playbook_rejection() -> None:
    first = derive_shadow_root_episode_records(
        _valid_neutral_challenges()
    )[0]
    same_day = replace(
        first,
        root_episode_key=first.root_episode_key + ":same-day",
        root_id=str(first.root_id) + ":same-day",
        candidate_id=first.candidate_id + ":same-day",
        observed_at=first.observed_at + pd.Timedelta(hours=1),
    )
    next_day = replace(
        first,
        root_episode_key=first.root_episode_key + ":next-day",
        root_id=str(first.root_id) + ":next-day",
        candidate_id=first.candidate_id + ":next-day",
        observed_at=first.observed_at + pd.Timedelta(days=1),
        session_date_ny=(
            first.observed_at + pd.Timedelta(days=1)
        ).tz_convert(TZ).date().isoformat(),
    )

    one_date = aggregate_shadow_mechanism_motifs((first, same_day))
    assert len(one_date) == 1
    assert one_date[0].eligible_for_preregistration_review is False

    repeated = aggregate_shadow_mechanism_motifs((first, next_day))
    assert len(repeated) == 1
    assert repeated[0].eligible_episode_count == 2
    assert repeated[0].eligible_distinct_dates == 2
    assert repeated[0].eligible_for_preregistration_review is True
    assert repeated[0].action_authority is False
    assert repeated[0].sample_root_episode_count == 2
    assert repeated[0].root_episode_keys_truncated is False

    lsr_accepted = replace(
        next_day,
        lsr_rejected=False,
        eligible_episode_evidence=False,
    )
    rejected_by_one = aggregate_shadow_mechanism_motifs(
        (first, lsr_accepted)
    )
    assert rejected_by_one[0].eligible_for_preregistration_review is False


def test_motif_root_membership_is_a_bounded_deterministic_sample() -> None:
    first = derive_shadow_root_episode_records(
        _valid_neutral_challenges()
    )[0]
    episodes = tuple(
        replace(
            first,
            root_episode_key=f"{first.root_episode_key}:{index:03d}",
            root_id=f"{first.root_id}:{index:03d}",
            candidate_id=f"{first.candidate_id}:{index:03d}",
        )
        for index in range(SHADOW_MOTIF_ROOT_SAMPLE_LIMIT + 5)
    )

    forward = aggregate_shadow_mechanism_motifs(episodes)[0]
    reverse = aggregate_shadow_mechanism_motifs(tuple(reversed(episodes)))[0]
    expected = sorted(item.root_episode_key for item in episodes)[
        :SHADOW_MOTIF_ROOT_SAMPLE_LIMIT
    ]

    assert forward.episode_count == SHADOW_MOTIF_ROOT_SAMPLE_LIMIT + 5
    assert forward.sample_root_episode_count == SHADOW_MOTIF_ROOT_SAMPLE_LIMIT
    assert forward.root_episode_keys_truncated is True
    assert json.loads(forward.sample_root_episode_keys) == expected
    assert reverse.sample_root_episode_keys == forward.sample_root_episode_keys


@pytest.mark.parametrize(
    "replacement",
    (
        {"censored": True, "eligible_episode_evidence": False},
        {"filled": False, "eligible_episode_evidence": False},
        {
            "risk_qualified_target_R": False,
            "eligible_episode_evidence": False,
        },
        {"path_valid": False, "eligible_episode_evidence": False},
    ),
)
def test_motif_rejects_non_evaluable_episode_evidence(replacement) -> None:
    first = derive_shadow_root_episode_records(
        _valid_neutral_challenges()
    )[0]
    next_day = replace(
        first,
        root_episode_key=first.root_episode_key + ":next-day",
        root_id=str(first.root_id) + ":next-day",
        candidate_id=first.candidate_id + ":next-day",
        observed_at=first.observed_at + pd.Timedelta(days=1),
        session_date_ny=(
            first.observed_at + pd.Timedelta(days=1)
        ).tz_convert(TZ).date().isoformat(),
        **replacement,
    )
    motif = aggregate_shadow_mechanism_motifs((first, next_day))[0]
    assert motif.eligible_for_preregistration_review is False


def test_executable_summary_slot_is_not_an_action_candidate() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    snapshot = _snapshot(
        source,
        Direction.LONG,
        event=False,
        executable_plan=True,
    )
    assert any(
        item.phase is PlaybookPhase.EXECUTABLE
        for item in snapshot.belief.candidates()
    )
    snapshot.belief.action_candidate_items = lambda: ()
    snapshot.belief.resolve_hypothesis = lambda _identity: None

    recorder.on_bar(source)
    recorder.observe(snapshot, source_bar=source)

    assert recorder.open_candidates == ()


def test_limit_fill_bar_never_credits_unknown_target_or_mfe() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        source_bar=source,
    )
    first_touch = _bar(source.end, high=101.0, low=99.0)
    _complete_engine_step(
        recorder,
        first_touch,
        Direction.LONG,
        executable_plan=True,
    )

    candidate = recorder.open_candidates[0]
    assert candidate.entry_price == 100.0
    assert candidate.mfe_points == 0.0
    assert recorder.rows == ()

    _complete_engine_step(
        recorder,
        _bar(first_touch.end, high=103.1, low=99.5),
        Direction.LONG,
        executable_plan=True,
    )
    assert recorder.drain_rows()[0].resolution == "target_first"


def test_limit_entry_and_target_same_bar_is_censored() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        source_bar=source,
    )
    _complete_engine_step(
        recorder,
        _bar(source.end, high=104.0, low=99.0),
        Direction.LONG,
        executable_plan=True,
    )

    row = recorder.drain_rows()[0]
    assert row.resolution == "entry_target_same_bar_order_unknown"
    assert row.censored is True
    assert row.mfe_points == 0.0
    assert row.hit_0_5R is False


def test_executable_episode_same_bar_stop_and_target_is_stop_first() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        source_bar=source,
    )
    fill_bar = _bar(source.end, high=101.0, low=99.0)
    _complete_engine_step(
        recorder,
        fill_bar,
        Direction.LONG,
        executable_plan=True,
    )
    collision = _bar(fill_bar.end, high=104.0, low=97.0)
    _complete_engine_step(
        recorder,
        collision,
        Direction.LONG,
        executable_plan=True,
    )

    episode = derive_shadow_episode_outcomes(list(recorder.drain_rows()))[0]
    assert episode.outcome_evaluable is True
    assert episode.outcome_class == "path_failed"
    assert episode.target_first is False
    assert episode.invalidation_first is True
    assert episode.same_bar_collision is True
    assert episode.same_bar_stop_first is True


def test_filled_deadline_expiry_is_excluded_from_episode_quadrants() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        source_bar=source,
    )
    quiet_fill = _bar(source.end, high=101.0, low=99.0)
    recorder.open_candidates[0].deadline_at = quiet_fill.end
    _complete_engine_step(
        recorder,
        quiet_fill,
        Direction.LONG,
        executable_plan=True,
    )

    row = recorder.drain_rows()[0]
    assert row.resolution == "deadline_no_delivery"
    assert row.filled is True
    assert all(
        item["quadrant"] is None
        for item in json.loads(row.playbook_outcomes)
    )
    episode = derive_shadow_episode_outcomes([row])[0]
    assert episode.expired is True
    assert episode.outcome_evaluable is False
    assert episode.outcome_class == "expired"
    assert episode.outcome_evaluable is False


def test_unfilled_deadline_is_marked_expired_and_not_evaluable() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        source_bar=source,
    )
    no_touch = _bar(
        source.end,
        open_=101.5,
        high=102.0,
        low=101.0,
        close=101.5,
    )
    recorder.open_candidates[0].deadline_at = no_touch.end
    _complete_engine_step(
        recorder,
        no_touch,
        Direction.LONG,
        executable_plan=True,
    )

    row = recorder.drain_rows()[0]
    assert row.resolution == "entry_unfilled_deadline"
    assert row.filled is False
    episode = derive_shadow_episode_outcomes([row])[0]
    assert episode.expired is True
    assert episode.outcome_evaluable is False
    assert episode.outcome_class == "expired"


def test_unknown_neutral_event_kind_fails_closed() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    later = _bar(source.end, high=104.0, low=99.5)
    _complete_engine_step(recorder, later, Direction.LONG)
    row = recorder.drain_rows()[0]

    with pytest.raises(ValueError, match="unknown neutral shadow candidate"):
        derive_shadow_mechanism_challenges(
            [replace(row, event_kind="unexpected_kind")]
        )


@pytest.mark.parametrize(
    ("high", "low", "resolution"),
    (
        (99.0, 97.0, "invalidation_before_entry"),
        (104.0, 103.5, "draw_consumed_before_entry"),
    ),
)
def test_limit_is_cancelled_when_frozen_boundary_moves_before_fill(
    high: float,
    low: float,
    resolution: str,
) -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        source_bar=source,
    )
    later = _bar(
        source.end,
        open_=(high + low) / 2.0,
        high=high,
        low=low,
        close=(high + low) / 2.0,
    )
    _complete_engine_step(
        recorder,
        later,
        Direction.LONG,
        executable_plan=True,
    )

    row = recorder.drain_rows()[0]
    assert row.resolution == resolution
    assert row.filled is False
    assert row.censored is True
    assert all(
        item["quadrant"] is None
        for item in json.loads(row.playbook_outcomes)
    )
    episode = derive_shadow_episode_outcomes([row])[0]
    assert episode.outcome_evaluable is False
    assert episode.outcome_class == "censored"
    assert episode.outcome_evaluable is False


def test_deadline_inside_bar_censors_before_reading_ohlc() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    recorder.on_bar(source)
    recorder.observe(
        _snapshot(
            source,
            Direction.LONG,
            event=False,
            executable_plan=True,
        ),
        source_bar=source,
    )
    recorder.open_candidates[0].deadline_at = (
        source.end + pd.Timedelta(seconds=30)
    )
    later = _bar(source.end, high=104.0, low=97.0)
    _complete_engine_step(
        recorder,
        later,
        Direction.LONG,
        executable_plan=True,
    )

    row = recorder.drain_rows()[0]
    assert row.resolution == "deadline_inside_completed_bar_censored"
    assert row.filled is False
    assert row.mfe_points is None
    episode = derive_shadow_episode_outcomes([row])[0]
    assert episode.expired is True
    assert episode.outcome_evaluable is False
    assert episode.outcome_class == "censored"


def test_same_bar_source_invalidation_preempts_price_target() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source, _ = _register(recorder, Direction.LONG)
    later = _bar(source.end, high=104.0, low=99.5)
    _complete_engine_step(
        recorder,
        later,
        Direction.LONG,
        invalidated_source_ids=("displacement:long",),
    )

    row = recorder.drain_rows()[0]
    assert row.resolution == "source_identity_invalidated"
    assert row.target_before_invalidation is False
    assert row.mfe_points == 0.0


def test_rearmed_hypothesis_does_not_change_frozen_episode_evidence() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    first = _snapshot(source, Direction.LONG)
    frozen = next(
        item
        for item in first.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    frozen.episode_id = "episode:old"
    frozen.setup_context_id = "episode:old"
    recorder.on_bar(source)
    recorder.observe(first, source_bar=source)

    later = _bar(source.end, high=101.0, low=99.0)
    rearmed = _snapshot(later, Direction.LONG, event=False)
    current = next(
        item
        for item in rearmed.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    current.episode_id = "episode:new"
    current.setup_context_id = "episode:new"
    current.hard_gate_results = {"causal_gate": False}
    recorder.on_bar(later)
    recorder.observe(rearmed, source_bar=later)

    assert recorder.open_candidates[0].first_changed_evidence_id is None


def test_dormant_episode_gate_clear_is_not_shadow_evidence_change() -> None:
    recorder = ShadowCandidateOutcomeRecorder()
    source = _bar(_start())
    first = _snapshot(
        source,
        Direction.LONG,
        executable_plan=True,
    )
    frozen = next(
        item
        for item in first.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    frozen.setup_context_id = frozen.plan.setup_id
    frozen.record_kind = "root_candidate"
    recorder.on_bar(source)
    recorder.observe(first, source_bar=source)

    later = _bar(source.end, high=101.0, low=99.0)
    dormant_snapshot = _snapshot(
        later,
        Direction.LONG,
        event=False,
        executable_plan=True,
    )
    dormant = next(
        item
        for item in dormant_snapshot.belief.candidates()
        if item.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    dormant.episode_id = frozen.episode_id
    dormant.setup_context_id = frozen.setup_context_id
    dormant.bound_market_thesis_id = frozen.bound_market_thesis_id
    dormant.record_kind = "retained_episode"
    dormant.hard_gate_results = {
        name: False for name in frozen.hard_gate_results
    }
    recorder.on_bar(later)
    recorder.observe(dormant_snapshot, source_bar=later)

    executable = next(
        item
        for item in recorder.open_candidates
        if item.source_episode_id == frozen.episode_id
    )
    assert executable.first_changed_evidence_id is None
