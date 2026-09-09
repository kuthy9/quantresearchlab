from __future__ import annotations

from dataclasses import replace
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from brain.core.brain_entry_sequence import brain_observation_view
from contract.market import (
    Direction,
    MarketMode,
    ScaleRelation,
    Timeframe,
)
from contract.eye import (
    BOSScope,
    EntryLocationLifecycle,
    EntryLocationState,
    InteractionUpdate,
    MicroBreakFact,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
)
from contract.brain import (
    DirectionalObstructionView,
    GlobalMarketContext,
    NEUTRAL_MARKET_STATE_SCHEMA_VERSION,
    OpenMarketThesis,
    ScaleRelationState,
)
from shares.core.scene_graph import (
    build_neutral_market_state,
    market_episode_id,
)


BASE = pd.Timestamp("2025-01-07 09:30", tz="America/New_York")
SYMBOL = "NQH5"
INSTRUMENT_ID = 1


def _location(
    *,
    location_id: str = "location:one",
    zone_id: str = "zone:one",
    displacement_id: str = "displacement:one",
    direction: Direction = Direction.LONG,
    asof: pd.Timestamp = BASE,
    lifecycle: EntryLocationLifecycle = EntryLocationLifecycle.APPROACHING,
) -> EntryLocationState:
    first_pullback_at = (
        BASE + pd.Timedelta(minutes=2)
        if lifecycle is not EntryLocationLifecycle.APPROACHING
        else None
    )
    rejected_at = (
        first_pullback_at if lifecycle is EntryLocationLifecycle.REJECTED else None
    )
    left_at = asof if lifecycle is EntryLocationLifecycle.LEFT else None
    state_started_at = (
        rejected_at
        if rejected_at is not None
        else left_at if left_at is not None else BASE
    )
    lower, upper = (100.0, 101.0)
    near, far = (upper, lower) if direction is Direction.LONG else (lower, upper)
    return EntryLocationState(
        location_id=location_id,
        protocol_hash="group5-protocol",
        source_zone_detector_protocol_hash="group3-protocol",
        symbol=SYMBOL,
        instrument_id=INSTRUMENT_ID,
        direction=direction,
        source_zone_kind="fvg",
        source_zone_id=zone_id,
        source_zone_protocol_hash="zone-protocol",
        source_displacement_id=displacement_id,
        source_bos_id=None,
        lower_bound=lower,
        upper_bound=upper,
        midpoint=100.5,
        near_edge=near,
        far_edge=far,
        failure_boundary=far,
        formed_at=BASE,
        lifecycle=lifecycle,
        state_started_at=state_started_at,
        last_updated_at=asof,
        age_real_1m_bars=max(0, int((asof - BASE).total_seconds() // 60)),
        state_duration_real_1m_bars=0,
        current_price=101.25,
        distance_to_zone_points=0.25,
        distance_to_failure_points=1.25,
        departure_confirmed_at=(
            None if first_pullback_at is None else BASE + pd.Timedelta(minutes=1)
        ),
        first_entered_at=first_pullback_at,
        entry_mode=(None if first_pullback_at is None else "crossed_near_edge"),
        contact_reference_price=(None if first_pullback_at is None else near),
        first_penetration_fraction=(0.0 if first_pullback_at is None else 0.25),
        rejected_at=rejected_at,
        left_at=left_at,
        reaction_atr=0.25 if rejected_at is not None else 0.0,
        transition_reason=(
            "source_registered"
            if lifecycle is EntryLocationLifecycle.APPROACHING
            else (
                "same_bar_wick_rejection"
                if lifecycle is EntryLocationLifecycle.REJECTED
                else "close_beyond_far_edge"
            )
        ),
    )


def _path_step(
    step_id: str,
    kind: str,
    observed_at: pd.Timestamp,
    *,
    direction: Direction,
    source_entity_id: str,
    source_event_id: str | None,
    predecessor: str | None,
    same_clock_relation: str,
) -> PathSequenceStep:
    physical_reason = {
        "zone_visible": "typed_entry_zone_registered",
        "departure_confirmed": "later_close_on_delivery_side",
        "first_pullback": "crossed_near_edge",
        "wick_rejection": "same_bar_wick_rejection",
        "reacceptance_held": "later_real_completed_hold",
        "location_left": "close_beyond_far_edge",
    }.get(kind)
    if kind == "micro_bos_confirmed":
        physical_reason = (
            "confirmed_m1_break_at_anchor_clock"
            if same_clock_relation == "same_clock_unknown"
            else "first_strictly_later_confirmed_m1_break"
        )
    if physical_reason is None:
        raise ValueError(f"test path step kind is not physical: {kind}")
    return PathSequenceStep(
        step_id=step_id,
        kind=kind,
        observed_at=observed_at,
        source_event_id=source_event_id,
        source_entity_id=source_entity_id,
        predecessor_step_ids=() if predecessor is None else (predecessor,),
        same_clock_relation=same_clock_relation,
        direction=direction,
        strength=0.5,
        reason=physical_reason,
    )


def _path(
    location: EntryLocationState,
    *,
    path_id: str = "path:one",
    asof: pd.Timestamp = BASE,
    terminal_reason: str | None = None,
    include_pullback: bool = False,
    include_wick: bool = False,
    include_trigger: bool = False,
    censored: bool = False,
) -> PathSequenceState:
    origin = _path_step(
        f"{path_id}:origin",
        "zone_visible",
        BASE,
        direction=location.direction,
        source_entity_id=location.source_zone_id,
        source_event_id=location.source_zone_id,
        predecessor=None,
        same_clock_relation="origin",
    )
    steps = [origin]
    if include_pullback:
        departure = _path_step(
            f"{path_id}:departure",
            "departure_confirmed",
            BASE + pd.Timedelta(minutes=1),
            direction=location.direction,
            source_entity_id=location.location_id,
            source_event_id=None,
            predecessor=steps[-1].step_id,
            same_clock_relation="strictly_after",
        )
        steps.append(departure)
        pullback = _path_step(
            f"{path_id}:pullback",
            "first_pullback",
            BASE + pd.Timedelta(minutes=2),
            direction=location.direction,
            source_entity_id=location.location_id,
            source_event_id=None,
            predecessor=steps[-1].step_id,
            same_clock_relation="strictly_after",
        )
        steps.append(pullback)
    if include_wick:
        steps.append(
            _path_step(
                f"{path_id}:wick",
                "wick_rejection",
                BASE + pd.Timedelta(minutes=2),
                direction=location.direction,
                source_entity_id=location.location_id,
                source_event_id=None,
                predecessor=steps[-1].step_id,
                same_clock_relation="same_clock_known",
            )
        )
    if include_trigger:
        steps.append(
            _path_step(
                f"{path_id}:trigger",
                "micro_bos_confirmed",
                asof,
                direction=location.direction,
                source_entity_id="swing:trigger",
                source_event_id="bos:trigger",
                predecessor=steps[-1].step_id,
                same_clock_relation="strictly_after",
            )
        )
    if terminal_reason == "location_left":
        steps.append(
            _path_step(
                f"{path_id}:left",
                "location_left",
                asof,
                direction=location.direction,
                source_entity_id=location.location_id,
                source_event_id=None,
                predecessor=steps[-1].step_id,
                same_clock_relation="same_clock_known",
            )
        )
    lifecycle = (
        PathSequenceLifecycle.CENSORED
        if censored
        else (
            PathSequenceLifecycle.CLOSED
            if terminal_reason is not None
            else PathSequenceLifecycle.ACTIVE
        )
    )
    return PathSequenceState(
        sequence_id=path_id,
        protocol_hash="group5-protocol",
        symbol=location.symbol,
        instrument_id=location.instrument_id,
        context_kind="zone_return",
        context_id=location.location_id,
        direction=location.direction,
        lifecycle=lifecycle,
        formed_at=BASE,
        state_started_at=(
            asof if lifecycle is not PathSequenceLifecycle.ACTIVE else BASE
        ),
        last_updated_at=asof,
        age_real_1m_bars=max(0, int((asof - BASE).total_seconds() // 60)),
        state_duration_real_1m_bars=0,
        steps=tuple(steps),
        ended_at=asof if lifecycle is not PathSequenceLifecycle.ACTIVE else None,
        transition_reason=terminal_reason or "context_registered",
    )


def _thesis(
    index: int,
    location_id: str,
    *,
    direction: Direction | None,
    epoch: str,
    asof: pd.Timestamp,
) -> OpenMarketThesis:
    root_id = f"root:{index}"
    return OpenMarketThesis(
        thesis_id=f"thesis:{index}",
        root_id=root_id,
        market_epoch_id=epoch,
        formed_at=BASE,
        updated_at=asof,
        direction=direction,
        source_timeframe=Timeframe.M5,
        structural_scale="intermediate",
        mechanism="zone_return",
        authority_relation="diagnostic",
        mechanism_event_ids=(root_id,),
        entry_location_ids=(location_id,),
    )


def _context(
    asof: pd.Timestamp,
    *,
    epoch: str = "epoch:one",
    theses: tuple[OpenMarketThesis, ...] = (),
) -> GlobalMarketContext:
    revision = f"scene:{int((asof - BASE).total_seconds() // 60):04d}:{epoch}"
    return GlobalMarketContext(
        updated_at=asof,
        scene_revision_id=revision,
        market_epoch_id=epoch,
        authority_stack=(),
        market_mode=MarketMode.UNCERTAIN,
        scale_relation_details={
            timeframe.value: ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.UNKNOWN,
                direction=None,
                authority_layer_id=None,
                evidence_ids=(),
                evidence_kind=None,
                structural_scope=None,
                acceptance_state=None,
                since=None,
                age_bars=0,
                graph_connected=False,
                ambiguous=False,
            )
            for timeframe in Timeframe
        },
        external_draw_candidates={"above": (), "below": ()},
        obstruction_views={
            direction.value: DirectionalObstructionView(
                direction=direction,
                nearest_draw_id=None,
                nearest_draw_price=None,
                hard_barriers=(),
                soft_frictions=(),
            )
            for direction in Direction
        },
        material_conflicts=(),
        unknown_evidence=(),
        ambiguous_evidence=(),
        dislocations_by_scale={timeframe.value: () for timeframe in Timeframe},
        open_market_theses=theses,
    )


def _observation(
    context: GlobalMarketContext,
    *,
    locations: tuple[EntryLocationState, ...],
    paths: tuple[PathSequenceState, ...],
    boundary_paths: tuple[PathSequenceState, ...] = (),
    anomalies: tuple[str, ...] = (),
) -> object:
    physical_paths: list[PathSequenceState] = []
    facts: list[MicroBreakFact] = []
    locations_by_id = {location.location_id: location for location in locations}
    for path in paths:
        physical_steps: list[PathSequenceStep] = []
        for step in path.steps:
            if step.kind.startswith("micro_bos_"):
                location = locations_by_id[path.context_id]
                anchor_at = location.first_entered_at
                if anchor_at is None:
                    raise ValueError("test micro break requires its pullback")
                relation = (
                    "same_clock_unknown"
                    if step.observed_at == anchor_at
                    else "strictly_after"
                )
                physical_step = replace(
                    step,
                    step_id=f"{step.step_id}:physical",
                    kind="micro_break_observed",
                    predecessor_step_ids=(physical_steps[-1].step_id,),
                    reason=(
                        "confirmed_m1_break_at_anchor_clock"
                        if relation == "same_clock_unknown"
                        else "first_strictly_later_confirmed_m1_break"
                    ),
                )
                facts.append(
                    MicroBreakFact(
                        reference_id=f"reference:{step.step_id}",
                        protocol_hash=path.protocol_hash,
                        context_kind=path.context_kind,
                        context_id=path.context_id,
                        context_direction=path.direction,
                        anchor_at=anchor_at,
                        bos_id=step.source_event_id or "",
                        bos_direction=(
                            (
                                Direction.SHORT
                                if path.direction is Direction.LONG
                                else Direction.LONG
                            )
                            if path.transition_reason == "micro_bos_opposed"
                            else path.direction
                        ),
                        target_swing_id=step.source_entity_id,
                        scope=BOSScope.LOCAL,
                        pending_at=step.observed_at - pd.Timedelta(minutes=1),
                        resolved_at=step.observed_at,
                        relation=relation,
                        strength=step.strength,
                    )
                )
            else:
                physical_step = replace(
                    step,
                    predecessor_step_ids=(
                        ()
                        if not physical_steps
                        else (physical_steps[-1].step_id,)
                    ),
                )
            physical_steps.append(physical_step)
        physical_paths.append(
            replace(
                path,
                steps=tuple(physical_steps),
                transition_reason=(
                    "first_strict_micro_break_observed"
                    if path.transition_reason
                    in {
                        "micro_bos_aligned",
                        "micro_bos_opposed",
                        "micro_bos_ambiguous_same_clock",
                    }
                    else path.transition_reason
                ),
            )
        )
    raw = SimpleNamespace(
        asof=context.updated_at,
        scene_revision_id=context.scene_revision_id,
        symbol=SYMBOL,
        instrument_id=INSTRUMENT_ID,
        interaction_update=InteractionUpdate(
            zone_interactions=(() if boundary_paths else locations),
            reacceptance_interactions=(),
            micro_break_facts=(() if boundary_paths else tuple(facts)),
            interaction_paths=(
                () if boundary_paths else tuple(physical_paths)
            ),
            interaction_path_transitions=boundary_paths,
            boundary_reason=(
                None
                if not boundary_paths
                else boundary_paths[0].transition_reason
            ),
        ),
        anomalies=anomalies,
    )
    return brain_observation_view(raw)


def test_zero_one_many_claims_are_relations_on_one_physical_episode() -> None:
    location = _location()
    path = _path(location)
    context0 = _context(BASE)
    state0 = build_neutral_market_state(
        None,
        _observation(context0, locations=(location,), paths=(path,)),
        context0,
    )
    episode0 = state0.market_episodes[0]
    assert episode0.claims == ()
    assert episode0.active_claim_ids == ()
    assert episode0.claim_status == "unbound"

    at1 = BASE + pd.Timedelta(minutes=1)
    thesis1 = _thesis(
        1,
        location.location_id,
        direction=Direction.LONG,
        epoch="epoch:one",
        asof=at1,
    )
    context1 = _context(at1, theses=(thesis1,))
    state1 = build_neutral_market_state(
        state0,
        _observation(context1, locations=(location,), paths=(path,)),
        context1,
    )
    episode1 = state1.market_episodes[0]
    assert episode1.episode_id == episode0.episode_id
    assert episode1.active_claim_ids == ("thesis:1",)
    assert episode1.claim_status == "unique"

    at2 = BASE + pd.Timedelta(minutes=2)
    theses2 = (
        _thesis(
            1,
            location.location_id,
            direction=Direction.LONG,
            epoch="epoch:one",
            asof=at2,
        ),
        _thesis(
            2,
            location.location_id,
            direction=Direction.SHORT,
            epoch="epoch:one",
            asof=at2,
        ),
    )
    context2 = _context(at2, theses=theses2)
    state2 = build_neutral_market_state(
        state1,
        _observation(context2, locations=(location,), paths=(path,)),
        context2,
    )
    episode2 = state2.market_episodes[0]
    assert len(state2.market_episodes) == 1
    assert episode2.episode_id == episode0.episode_id
    assert episode2.active_claim_ids == ("thesis:1", "thesis:2")
    assert episode2.claim_status == "ambiguous"
    assert tuple(claim.relation for claim in episode2.claims) == (
        "aligned",
        "opposed",
    )

    at3 = BASE + pd.Timedelta(minutes=3)
    thesis1_at3 = _thesis(
        1, location.location_id, direction=Direction.LONG, epoch="epoch:one", asof=at3
    )
    context3 = _context(at3, theses=(thesis1_at3,))
    state3 = build_neutral_market_state(
        state2,
        _observation(context3, locations=(location,), paths=(path,)),
        context3,
    )
    episode3 = state3.market_episodes[0]
    assert episode3.episode_id == episode0.episode_id
    assert tuple(claim.thesis_id for claim in episode3.claims) == (
        "thesis:1",
        "thesis:2",
    )
    assert episode3.active_claim_ids == ("thesis:1",)
    assert episode3.claim_status == "unique"


def test_unchanged_episode_does_not_emit_a_heartbeat_transition() -> None:
    location = _location()
    path = _path(location)
    context0 = _context(BASE)
    state0 = build_neutral_market_state(
        None,
        _observation(context0, locations=(location,), paths=(path,)),
        context0,
    )
    at1 = BASE + pd.Timedelta(minutes=1)
    context1 = _context(at1)
    state1 = build_neutral_market_state(
        state0,
        _observation(context1, locations=(location,), paths=(path,)),
        context1,
    )
    assert state1.market_episodes == state0.market_episodes
    assert state1.episode_transitions_this_update == ()


def test_existing_episode_can_observe_first_pullback_at_current_clock() -> None:
    thesis0 = _thesis(
        1,
        "location:one",
        direction=Direction.LONG,
        epoch="epoch:one",
        asof=BASE,
    )
    location0 = _location()
    path0 = _path(location0)
    context0 = _context(BASE, theses=(thesis0,))
    state0 = build_neutral_market_state(
        None,
        _observation(context0, locations=(location0,), paths=(path0,)),
        context0,
    )
    previous = state0.market_episodes[0]
    assert previous.lifecycle == "registered"
    assert previous.updated_at == BASE

    at2 = BASE + pd.Timedelta(minutes=2)
    location2 = _location(
        asof=at2,
        lifecycle=EntryLocationLifecycle.REJECTED,
    )
    path2 = _path(
        location2,
        asof=at2,
        include_pullback=True,
        include_wick=True,
    )
    thesis2 = _thesis(
        1,
        location2.location_id,
        direction=Direction.LONG,
        epoch="epoch:one",
        asof=at2,
    )
    context2 = _context(at2, theses=(thesis2,))
    state2 = build_neutral_market_state(
        state0,
        _observation(context2, locations=(location2,), paths=(path2,)),
        context2,
    )

    current = state2.market_episodes[0]
    assert current.episode_id == previous.episode_id
    assert current.updated_at == at2
    assert current.first_pullback_at == at2
    assert current.lifecycle == "pullback"
    assert state2.episode_transitions_this_update == (current,)


def test_current_episode_transition_transport_is_complete_and_exact() -> None:
    location = _location()
    path = _path(location)
    context = _context(BASE)
    state = build_neutral_market_state(
        None,
        _observation(context, locations=(location,), paths=(path,)),
        context,
    )
    assert len(state.market_episodes) == 1
    assert state.schema_version == NEUTRAL_MARKET_STATE_SCHEMA_VERSION == 2
    assert state.episode_transitions_this_update == state.market_episodes

    with pytest.raises(ValueError, match="neutral market state is invalid"):
        replace(state, schema_version=1)

    with pytest.raises(ValueError, match="neutral market state is invalid"):
        replace(state, episode_transitions_this_update=())

    mismatched = replace(state.market_episodes[0], binding_status="unbound")
    with pytest.raises(ValueError, match="neutral market state is invalid"):
        replace(state, episode_transitions_this_update=(mismatched,))


def test_unbound_unique_ambiguous_unique_preserves_physical_identity() -> None:
    location = _location()
    path = _path(location)
    context0 = _context(BASE)
    with pytest.raises(ValueError, match="zones and paths differ"):
        _observation(context0, locations=(location,), paths=())

    at1 = BASE + pd.Timedelta(minutes=1)
    context1 = _context(at1)
    state1 = build_neutral_market_state(
        None,
        _observation(context1, locations=(location,), paths=(path,)),
        context1,
    )
    identity = state1.market_episodes[0].episode_id
    assert identity == market_episode_id(
        "epoch:one",
        location.location_id,
        path.sequence_id,
        location.direction,
    )

    at2 = BASE + pd.Timedelta(minutes=2)
    context2 = _context(at2)
    sibling_path = _path(location, path_id="path:two")
    with pytest.raises(ValueError, match="path contexts repeat"):
        _observation(
            context2,
            locations=(location,),
            paths=(path, sibling_path),
        )

    at3 = BASE + pd.Timedelta(minutes=3)
    context3 = _context(at3)
    state3 = build_neutral_market_state(
        state1,
        _observation(context3, locations=(location,), paths=(path,)),
        context3,
    )
    assert state3.market_episodes[0].episode_id == identity
    assert state3.market_episodes[0].binding_status == "unique"


def test_same_location_multi_path_never_admits_but_distinct_locations_do() -> None:
    first_location = _location()
    first_path = _path(first_location)
    second_path = _path(first_location, path_id="path:two")
    context = _context(BASE)
    with pytest.raises(ValueError, match="path contexts repeat"):
        _observation(
            context,
            locations=(first_location,),
            paths=(first_path, second_path),
        )

    second_location = _location(
        location_id="location:two",
        zone_id="zone:two",
        displacement_id="displacement:two",
    )
    distinct = build_neutral_market_state(
        None,
        _observation(
            context,
            locations=(first_location, second_location),
            paths=(first_path, _path(second_location, path_id="path:three")),
        ),
        context,
    )
    assert len(distinct.market_episodes) == 2
    assert len({episode.episode_id for episode in distinct.market_episodes}) == 2


def test_same_bar_pullback_survives_location_left_and_wick_rejection() -> None:
    at = BASE + pd.Timedelta(minutes=2)
    left_location = _location(
        asof=at,
        lifecycle=EntryLocationLifecycle.LEFT,
    )
    left_path = _path(
        left_location,
        asof=at,
        terminal_reason="location_left",
        include_pullback=True,
    )
    context = _context(at)
    terminal = build_neutral_market_state(
        None,
        _observation(context, locations=(left_location,), paths=(left_path,)),
        context,
    ).market_episodes[0]
    assert terminal.first_pullback_at == at
    assert terminal.lifecycle == "terminal"
    assert terminal.terminal_reason == "location_left"

    rejected_location = _location(
        asof=at,
        lifecycle=EntryLocationLifecycle.REJECTED,
    )
    rejected_path = _path(
        rejected_location,
        asof=at,
        include_pullback=True,
        include_wick=True,
    )
    reacted = build_neutral_market_state(
        None,
        _observation(
            context,
            locations=(rejected_location,),
            paths=(rejected_path,),
        ),
        context,
    ).market_episodes[0]
    assert reacted.first_pullback_at == at
    assert reacted.lifecycle == "pullback"
    assert reacted.terminal_at is None


def test_terminal_episode_freezes_claims_and_never_retransitions() -> None:
    at2 = BASE + pd.Timedelta(minutes=3)
    location = _location(
        asof=at2,
        lifecycle=EntryLocationLifecycle.LEFT,
    )
    path = _path(
        location,
        asof=at2,
        terminal_reason="micro_bos_opposed",
        include_pullback=True,
        include_trigger=True,
    )
    thesis = _thesis(
        1,
        location.location_id,
        direction=Direction.LONG,
        epoch="epoch:one",
        asof=at2,
    )
    context2 = _context(at2, theses=(thesis,))
    state2 = build_neutral_market_state(
        None,
        _observation(context2, locations=(location,), paths=(path,)),
        context2,
    )
    terminal = state2.market_episodes[0]
    assert terminal.lifecycle == "terminal"
    assert terminal.terminal_at == at2
    assert terminal.terminal_reason == "micro_bos_opposed"
    assert terminal.active_claim_ids == ("thesis:1",)
    state2 = pickle.loads(pickle.dumps(state2, protocol=pickle.HIGHEST_PROTOCOL))
    assert state2.market_episodes == (terminal,)

    at3 = BASE + pd.Timedelta(minutes=4)
    context3 = _context(at3)
    unchanged = build_neutral_market_state(
        state2,
        _observation(context3, locations=(location,), paths=(path,)),
        context3,
    )
    assert unchanged.market_episodes == (terminal,)
    assert unchanged.episode_transitions_this_update == ()

    with pytest.raises(ValueError, match="zones and paths differ"):
        _observation(context3, locations=(location,), paths=())

    changed_terminal = _path(
        location,
        asof=at3,
        terminal_reason="micro_bos_opposed",
        include_pullback=True,
        include_trigger=True,
    )
    with pytest.raises(ValueError, match="terminal custody changed"):
        build_neutral_market_state(
            state2,
            _observation(
                context3,
                locations=(location,),
                paths=(changed_terminal,),
            ),
            context3,
        )

    changed_reason = _path(
        location,
        asof=at2,
        terminal_reason="reacceptance_failed",
        include_pullback=True,
    )
    with pytest.raises(ValueError, match="terminal custody changed"):
        build_neutral_market_state(
            state2,
            _observation(
                context3,
                locations=(location,),
                paths=(changed_reason,),
            ),
            context3,
        )


def test_successful_closed_path_is_a_strict_post_pullback_pulse() -> None:
    at = BASE + pd.Timedelta(minutes=3)
    location = _location(
        asof=at,
        lifecycle=EntryLocationLifecycle.REJECTED,
    )
    path = _path(
        location,
        asof=at,
        terminal_reason="micro_bos_aligned",
        include_pullback=True,
        include_wick=True,
        include_trigger=True,
    )
    context = _context(at)
    episode = build_neutral_market_state(
        None,
        _observation(context, locations=(location,), paths=(path,)),
        context,
    ).market_episodes[0]
    assert episode.lifecycle == "triggered"
    assert episode.trigger_at == at
    assert episode.trigger_at > episode.first_pullback_at
    assert episode.successful_pulse_at == episode.trigger_at
    assert episode.successful_pulse_reason == "micro_bos_aligned"
    assert episode.terminal_at is None


def test_trigger_relation_is_to_pullback_not_same_clock_predecessor() -> None:
    pullback_at = BASE + pd.Timedelta(minutes=2)
    initial_location = _location(
        asof=pullback_at,
        lifecycle=EntryLocationLifecycle.REJECTED,
    )
    initial_path = _path(
        initial_location,
        asof=pullback_at,
        include_pullback=True,
    )
    initial_context = _context(pullback_at)
    initial = build_neutral_market_state(
        None,
        _observation(
            initial_context,
            locations=(initial_location,),
            paths=(initial_path,),
        ),
        initial_context,
    )
    assert initial.market_episodes[0].lifecycle == "pullback"

    trigger_at = BASE + pd.Timedelta(minutes=4)
    current_location = _location(
        asof=trigger_at,
        lifecycle=EntryLocationLifecycle.REJECTED,
    )
    current_path = _path(
        current_location,
        asof=trigger_at,
        terminal_reason="micro_bos_aligned",
        include_pullback=True,
    )
    held = _path_step(
        "path:one:held",
        "reacceptance_held",
        trigger_at,
        direction=current_location.direction,
        source_entity_id="reference:one",
        source_event_id=None,
        predecessor=current_path.steps[-1].step_id,
        same_clock_relation="strictly_after",
    )
    trigger = _path_step(
        "path:one:trigger",
        "micro_bos_confirmed",
        trigger_at,
        direction=current_location.direction,
        source_entity_id="swing:trigger",
        source_event_id="bos:trigger",
        predecessor=held.step_id,
        # This relation is to the immediately preceding held step, not the
        # earlier first-pullback anchor.
        same_clock_relation="same_clock_unknown",
    )
    current_path = replace(
        current_path,
        steps=(*current_path.steps, held, trigger),
    )
    current_context = _context(trigger_at)
    current_observation = _observation(
        current_context,
        locations=(current_location,),
        paths=(current_path,),
    )
    interpreted_trigger = next(
        step
        for step in current_observation.interaction.path_sequences[0].steps
        if step.source_event_id == "bos:trigger"
    )
    current = build_neutral_market_state(
        initial,
        current_observation,
        current_context,
    )
    episode = current.market_episodes[0]

    assert episode.first_pullback_at == pullback_at
    assert episode.trigger_step_id == interpreted_trigger.step_id
    assert episode.trigger_event_id == "bos:trigger"
    assert episode.trigger_at == trigger_at
    assert episode.successful_pulse_at == trigger_at
    assert episode.lifecycle == "triggered"
    assert pickle.loads(pickle.dumps(current)) == current


def test_micro_bos_at_pullback_clock_is_not_a_neutral_trigger() -> None:
    pullback_at = BASE + pd.Timedelta(minutes=2)
    location = _location(
        asof=pullback_at,
        lifecycle=EntryLocationLifecycle.REJECTED,
    )
    path = _path(
        location,
        asof=pullback_at,
        include_pullback=True,
    )
    simultaneous = _path_step(
        "path:one:simultaneous",
        "micro_bos_confirmed",
        pullback_at,
        direction=location.direction,
        source_entity_id="swing:simultaneous",
        source_event_id="bos:simultaneous",
        predecessor=path.steps[-1].step_id,
        same_clock_relation="same_clock_unknown",
    )
    path = replace(path, steps=(*path.steps, simultaneous))
    context = _context(pullback_at)
    episode = build_neutral_market_state(
        None,
        _observation(context, locations=(location,), paths=(path,)),
        context,
    ).market_episodes[0]

    assert episode.first_pullback_at == pullback_at
    assert episode.trigger_at is None
    assert episode.successful_pulse_at is None
    assert episode.lifecycle == "pullback"


def test_epoch_reset_emits_terminal_transition_and_state_pickles() -> None:
    location = _location()
    path = _path(location)
    context0 = _context(BASE)
    state0 = build_neutral_market_state(
        None,
        _observation(context0, locations=(location,), paths=(path,)),
        context0,
    )
    assert pickle.loads(pickle.dumps(state0)) == state0

    at1 = BASE + pd.Timedelta(minutes=1)
    context1 = _context(at1, epoch="epoch:two")
    boundary_path = _path(
        location,
        asof=at1,
        terminal_reason="contract_change_reset",
        censored=True,
    )
    state1 = build_neutral_market_state(
        state0,
        _observation(
            context1,
            locations=(),
            paths=(),
            boundary_paths=(boundary_path,),
            anomalies=("contract_change_history_reset",),
        ),
        context1,
    )
    assert state1.market_episodes == ()
    assert len(state1.episode_transitions_this_update) == 1
    terminal = state1.episode_transitions_this_update[0]
    assert terminal.episode_id == state0.market_episodes[0].episode_id
    assert terminal.lifecycle == "terminal"
    assert terminal.terminal_at == at1
    assert terminal.terminal_reason == "contract_change_reset"


def test_success_compaction_is_explicit_retirement_not_terminal() -> None:
    at3 = BASE + pd.Timedelta(minutes=3)
    location = _location(
        asof=at3,
        lifecycle=EntryLocationLifecycle.REJECTED,
    )
    path = _path(
        location,
        asof=at3,
        terminal_reason="micro_bos_aligned",
        include_pullback=True,
        include_wick=True,
        include_trigger=True,
    )
    context3 = _context(at3)
    state3 = build_neutral_market_state(
        None,
        _observation(context3, locations=(location,), paths=(path,)),
        context3,
    )
    episode = state3.market_episodes[0]
    assert episode.terminal_at is None

    at4 = BASE + pd.Timedelta(minutes=4)
    context4 = _context(at4)
    retired = build_neutral_market_state(
        state3,
        _observation(context4, locations=(), paths=()),
        context4,
    )
    assert retired.market_episodes == ()
    assert retired.retired_episode_ids_this_update == (episode.episode_id,)
    assert retired.retirement_reasons_this_update == (
        (episode.episode_id, "upstream_compacted_after_success"),
    )
    assert retired.episode_transitions_this_update == ()


def test_unknown_closed_path_reason_fails_at_interaction_boundary() -> None:
    at = BASE + pd.Timedelta(minutes=2)
    location = _location(
        asof=at,
        lifecycle=EntryLocationLifecycle.LEFT,
    )
    path = _path(
        location,
        asof=at,
        terminal_reason="ambiguous",
        include_pullback=True,
    )
    context = _context(at)
    with pytest.raises(
        ValueError,
        match="interaction path physical vocabulary or direction changed",
    ):
        _observation(
            context,
            locations=(location,),
            paths=(path,),
        )


def test_global_context_requires_typed_unique_canonical_thesis_roots() -> None:
    first = _thesis(
        1,
        "location:one",
        direction=Direction.LONG,
        epoch="epoch:one",
        asof=BASE,
    )
    second = _thesis(
        2,
        "location:two",
        direction=Direction.SHORT,
        epoch="epoch:one",
        asof=BASE,
    )
    canonical = _context(BASE, theses=(first, second))
    assert canonical.open_market_theses == (first, second)

    with pytest.raises(ValueError, match="global market context is invalid"):
        _context(BASE, theses=(second, first))
    with pytest.raises(ValueError, match="global market context is invalid"):
        _context(
            BASE,
            theses=(first, replace(first, thesis_id="thesis:duplicate")),
        )
    with pytest.raises(ValueError, match="global market context is invalid"):
        _context(
            BASE,
            theses=(SimpleNamespace(thesis_id="thesis:forged"),),  # type: ignore[arg-type]
        )
