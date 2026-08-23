from __future__ import annotations

from dataclasses import replace
import hashlib

import pandas as pd
import pytest

from smc_trader.model import (
    BalanceContext,
    Direction,
    EventKind,
    EventOrigin,
    MarketEvent,
    ScaleRelation,
    ScaleRelationState,
    SMC_SEMANTIC_VERSION,
    Timeframe,
)
from smc_trader.path_belief import PathKind, PathStatus
from smc_trader.playbooks import PlaybookBrain

from .test_brain_path_belief_integration import (
    _canonical_semantic_event,
    _clock,
    _final_context,
    _observation,
)


def _production_bos_resolution(
    *,
    crossed_at: pd.Timestamp,
    resolved_at: pd.Timestamp,
    timeframe: Timeframe,
    scope: str,
    direction: Direction,
    source_structure_id: str,
    accepted: bool,
    suffix: str,
) -> tuple[MarketEvent, MarketEvent]:
    """Mirror Observation's BOS raw-break and crossing-resolution envelopes."""

    bos_id = f"bos:{suffix}"
    target_swing_id = f"target:{suffix}"
    break_bar_id = f"candle:{suffix}"
    level_id = f"swing:{target_swing_id}"
    side = "above" if direction is Direction.LONG else "below"
    price = 101.0 if direction is Direction.LONG else 99.0
    raw = _canonical_semantic_event(
        MarketEvent(
            event_id="pending-canonical-id",
            kind=EventKind.RAW_BOUNDARY_BREAK,
            observed_at=crossed_at,
            timeframe=timeframe,
            side=side,
            price=price,
            strength=0.8,
            direction=direction,
            event_time=crossed_at,
            known_at=crossed_at,
            evidence={
                "bos_id": bos_id,
                "target_swing_id": target_swing_id,
                "scope": scope,
                "break_bar_id": break_bar_id,
                "break_distance_atr": 0.75,
                "break_close": price,
                "break_buffer_ticks": 0,
                "comparison": "strict_close_beyond",
                "break_standard": "close_beyond_confirmed_boundary",
                "source_displacement_id": None,
            },
            source_event_ids=(
                f"event:swing-confirmed:{suffix}",
                f"event:bar-completed:{suffix}",
            ),
            source_data_ids=(break_bar_id,),
            source_entity_ids=(
                bos_id,
                target_swing_id,
                source_structure_id,
            ),
            context_event_ids=(f"event:bos-state:{suffix}",),
            origin=EventOrigin.SEMANTIC_ATOMIC,
        )
    )
    generation_id = hashlib.sha256(
        (
            f"{SMC_SEMANTIC_VERSION}|crossing-v1|{timeframe.value}|"
            f"{level_id}|{crossed_at.isoformat()}"
        ).encode("utf-8")
    ).hexdigest()[:24]
    resolution_kind = (
        EventKind.ACCEPTANCE_CONFIRMED
        if accepted
        else EventKind.SWEEP_CONFIRMED
    )
    resolution_direction = (
        direction
        if accepted
        else Direction.SHORT
        if direction is Direction.LONG
        else Direction.LONG
    )
    resolution = _canonical_semantic_event(
        MarketEvent(
            event_id="pending-canonical-id",
            kind=resolution_kind,
            observed_at=resolved_at,
            timeframe=timeframe,
            side=side,
            price=price,
            strength=0.8,
            direction=resolution_direction,
            event_time=crossed_at,
            known_at=resolved_at,
            evidence={
                "bos_id": bos_id,
                "level_id": level_id,
                "target_swing_id": target_swing_id,
                "resolution_bars": 1,
                "resolution": (
                    "held_outside" if accepted else "returned_inside"
                ),
                "crossing_generation_id": generation_id,
                "crossed_at": crossed_at.isoformat(),
                "resolved_at": resolved_at.isoformat(),
            },
            source_event_ids=(
                f"event:penetration:{suffix}",
                f"event:resolution-bar:{suffix}",
            ),
            source_entity_ids=(level_id,),
            context_event_ids=(
                raw.event_id,
                f"event:bos-post-break:{suffix}",
            ),
            origin=EventOrigin.SEMANTIC_ATOMIC,
        )
    )
    return raw, resolution


def _runtime_observation(
    asof: pd.Timestamp,
    *,
    retained: tuple[MarketEvent, ...] = (),
    current: tuple[MarketEvent, ...] = (),
):
    base = _observation(asof, real_completed=True)
    return replace(
        base,
        recent_events=retained,
        semantic_events_this_update=current,
    )


def _production_structural_confirmation(
    raw: MarketEvent,
    *,
    kind: EventKind,
) -> MarketEvent:
    bos_id = str(raw.evidence["bos_id"])
    source_structure_id = raw.source_entity_ids[2]
    return _canonical_semantic_event(
        MarketEvent(
            event_id="pending-canonical-id",
            kind=kind,
            observed_at=raw.known_at,
            timeframe=raw.timeframe,
            side=raw.side,
            price=raw.price,
            strength=raw.strength,
            direction=raw.direction,
            event_time=raw.event_time,
            known_at=raw.known_at,
            evidence={
                "bos_id": bos_id,
                "scope": raw.evidence["scope"],
                "qualification": (
                    "aligned_with_confirmed_structure"
                    if kind is EventKind.QUALIFIED_BOS
                    else "first_opposed_confirmed_boundary_break"
                ),
            },
            source_event_ids=(
                raw.event_id,
                f"event:structure-direction:{bos_id}",
            ),
            source_entity_ids=(bos_id, source_structure_id),
            origin=EventOrigin.SEMANTIC_ATOMIC,
        )
    )


def _initialize(
    brain: PlaybookBrain,
    at: pd.Timestamp,
    *,
    observation=None,
):
    context = _final_context(
        asof=at,
        scene_revision_id=f"revision:{at.isoformat()}",
    )
    state, _, _, _ = brain._update_shadow_path_diagnostics(
        observation or _runtime_observation(at),
        context,
        {},
        {},
        None,
    )
    assert state is not None
    return state, context


@pytest.mark.parametrize(
    ("scope", "direction", "accepted", "winner"),
    (
        ("continuation", Direction.LONG, True, PathKind.CONTINUATION),
        ("opposed", Direction.SHORT, True, PathKind.REVERSAL),
        ("continuation", Direction.LONG, False, PathKind.FAILED_BREAKOUT),
    ),
)
def test_production_bos_resolution_realizes_exact_global_winner(
    scope: str,
    direction: Direction,
    accepted: bool,
    winner: PathKind,
) -> None:
    t0 = _clock("2025-01-06 09:00")
    crossed = _clock("2025-01-06 10:00")
    resolved = _clock("2025-01-06 14:00")
    brain = PlaybookBrain()
    first, context = _initialize(brain, t0)
    raw, terminal = _production_bos_resolution(
        crossed_at=crossed,
        resolved_at=resolved,
        timeframe=Timeframe.H4,
        scope=scope,
        direction=direction,
        source_structure_id=first.authority_structure_id,
        accepted=accepted,
        suffix=winner.value,
    )

    state, records, rankings, exclusions = (
        brain._update_shadow_path_diagnostics(
            _runtime_observation(
                resolved,
                retained=(raw, terminal),
                current=(terminal,),
            ),
            replace(context, updated_at=resolved),
            {},
            {},
            None,
        )
    )

    assert state is not None and state.status is PathStatus.REALIZED
    assert state.winner_path is winner
    assert state.outcome_source_event_ids == tuple(
        sorted((raw.event_id, terminal.event_id))
    )
    assert len(records) == 1 and records[0].applied_outcome_event is not None
    assert rankings == {} and exclusions == {}


@pytest.mark.parametrize(
    ("kind", "scope", "direction", "invalidated"),
    (
        (
            EventKind.MSS_CORE_CONFIRMED,
            "opposed",
            Direction.SHORT,
            (PathKind.CONTINUATION, PathKind.BALANCE),
        ),
        (
            EventKind.QUALIFIED_BOS,
            "continuation",
            Direction.LONG,
            (
                PathKind.DEEPER_RETRACEMENT,
                PathKind.REVERSAL,
                PathKind.BALANCE,
            ),
        ),
    ),
)
def test_global_falsification_requires_exact_current_raw_parent(
    kind: EventKind,
    scope: str,
    direction: Direction,
    invalidated: tuple[PathKind, ...],
) -> None:
    t0 = _clock("2025-01-06 09:00")
    event_clock = _clock("2025-01-06 10:00")
    brain = PlaybookBrain()
    first, context = _initialize(brain, t0)
    raw, _ = _production_bos_resolution(
        crossed_at=event_clock,
        resolved_at=_clock("2025-01-06 14:00"),
        timeframe=Timeframe.H4,
        scope=scope,
        direction=direction,
        source_structure_id=first.authority_structure_id,
        accepted=True,
        suffix=f"falsification-{kind.value}",
    )
    confirmation = _production_structural_confirmation(raw, kind=kind)
    state, records, _, _ = brain._update_shadow_path_diagnostics(
        _runtime_observation(
            event_clock,
            retained=(raw, confirmation),
            current=(raw, confirmation),
        ),
        replace(context, updated_at=event_clock),
        {},
        {},
        None,
    )

    assert state is not None and state.status is PathStatus.ACTIVE
    assert all(
        state.member(path).status is PathStatus.INVALIDATED
        for path in invalidated
    )
    assert {event.path for event in records[0].applied_terminal_events} == set(
        invalidated
    )

    second_brain = PlaybookBrain()
    _, second_context = _initialize(second_brain, t0)
    forged_parents = (
        "event:missing-raw-parent",
        confirmation.source_event_ids[1],
    )
    forged = _canonical_semantic_event(
        replace(
            confirmation,
            event_id="pending-canonical-id",
            source_ids=forged_parents,
            source_event_ids=forged_parents,
        )
    )
    unchanged, unchanged_records, _, _ = (
        second_brain._update_shadow_path_diagnostics(
            _runtime_observation(
                event_clock,
                retained=(forged,),
                current=(forged,),
            ),
            replace(second_context, updated_at=event_clock),
            {},
            {},
            None,
        )
    )
    assert unchanged is not None
    assert all(member.status is PathStatus.ACTIVE for member in unchanged.members)
    assert unchanged_records[0].applied_terminal_events == ()


def test_same_clock_terminal_precedes_conflicting_continuation_outcome() -> None:
    t0 = _clock("2025-01-06 09:00")
    continuation_crossed = _clock("2025-01-06 10:00")
    conflict_clock = _clock("2025-01-06 14:00")
    brain = PlaybookBrain()
    first, context = _initialize(brain, t0)
    continuation_raw, continuation_acceptance = _production_bos_resolution(
        crossed_at=continuation_crossed,
        resolved_at=conflict_clock,
        timeframe=Timeframe.H4,
        scope="continuation",
        direction=Direction.LONG,
        source_structure_id=first.authority_structure_id,
        accepted=True,
        suffix="terminal-precedence-continuation",
    )
    opposed_raw, _ = _production_bos_resolution(
        crossed_at=conflict_clock,
        resolved_at=_clock("2025-01-06 16:00"),
        timeframe=Timeframe.H4,
        scope="opposed",
        direction=Direction.SHORT,
        source_structure_id=first.authority_structure_id,
        accepted=True,
        suffix="terminal-precedence-opposed",
    )
    opposed_mss = _production_structural_confirmation(
        opposed_raw,
        kind=EventKind.MSS_CORE_CONFIRMED,
    )

    state, records, _, _ = brain._update_shadow_path_diagnostics(
        _runtime_observation(
            conflict_clock,
            retained=(
                continuation_raw,
                continuation_acceptance,
                opposed_raw,
                opposed_mss,
            ),
            current=(
                continuation_acceptance,
                opposed_raw,
                opposed_mss,
            ),
        ),
        replace(context, updated_at=conflict_clock),
        {},
        {},
        None,
    )

    assert state is not None and state.status is PathStatus.ACTIVE
    assert state.winner_path is None
    assert state.member(PathKind.CONTINUATION).status is PathStatus.INVALIDATED
    assert state.member(PathKind.BALANCE).status is PathStatus.INVALIDATED
    assert records[0].applied_outcome_event is None
    assert {event.path for event in records[0].applied_terminal_events} == {
        PathKind.CONTINUATION,
        PathKind.BALANCE,
    }


def test_new_scope_does_not_retroactively_consume_formation_clock_resolution() -> None:
    crossed = _clock("2025-01-06 09:00")
    formed = _clock("2025-01-06 13:00")
    next_clock = formed + pd.Timedelta(minutes=1)
    raw, terminal = _production_bos_resolution(
        crossed_at=crossed,
        resolved_at=formed,
        timeframe=Timeframe.H4,
        scope="continuation",
        direction=Direction.LONG,
        source_structure_id="authority-structure",
        accepted=True,
        suffix="formation-clock",
    )
    brain = PlaybookBrain()
    initial, context = _initialize(
        brain,
        formed,
        observation=_runtime_observation(
            formed,
            retained=(raw, terminal),
            current=(terminal,),
        ),
    )
    assert initial.status is PathStatus.ACTIVE

    carried, records, _, _ = brain._update_shadow_path_diagnostics(
        _runtime_observation(next_clock, retained=(raw, terminal)),
        replace(context, updated_at=next_clock),
        {},
        {},
        None,
    )

    assert carried is not None and carried.status is PathStatus.ACTIVE
    assert carried.winner_path is None
    assert records[0].applied_outcome_event is None


def test_realized_set_is_carried_without_a_second_reducer_advance() -> None:
    t0 = _clock("2025-01-06 09:00")
    crossed = _clock("2025-01-06 10:00")
    resolved = _clock("2025-01-06 14:00")
    brain = PlaybookBrain()
    first, context = _initialize(brain, t0)
    raw, terminal = _production_bos_resolution(
        crossed_at=crossed,
        resolved_at=resolved,
        timeframe=Timeframe.H4,
        scope="continuation",
        direction=Direction.LONG,
        source_structure_id=first.authority_structure_id,
        accepted=True,
        suffix="carry",
    )
    realized, _, _, _ = brain._update_shadow_path_diagnostics(
        _runtime_observation(
            resolved,
            retained=(raw, terminal),
            current=(terminal,),
        ),
        replace(context, updated_at=resolved),
        {},
        {},
        None,
    )
    next_clock = resolved + pd.Timedelta(minutes=1)
    carried, records, rankings, exclusions = (
        brain._update_shadow_path_diagnostics(
            _runtime_observation(next_clock, retained=(raw, terminal)),
            replace(context, updated_at=next_clock),
            {},
            {},
            None,
        )
    )

    assert realized is not None and carried is not None
    assert carried.asof == next_clock
    assert carried.realized_at == realized.realized_at == resolved
    assert carried.outcome_event_id == realized.outcome_event_id
    assert records == () and rankings == {} and exclusions == {}


def test_same_clock_distinct_winners_fail_closed() -> None:
    t0 = _clock("2025-01-06 09:00")
    crossed = _clock("2025-01-06 10:00")
    resolved = _clock("2025-01-06 14:00")
    brain = PlaybookBrain()
    first, context = _initialize(brain, t0)
    raw, terminal = _production_bos_resolution(
        crossed_at=crossed,
        resolved_at=resolved,
        timeframe=Timeframe.H4,
        scope="continuation",
        direction=Direction.LONG,
        source_structure_id=first.authority_structure_id,
        accepted=True,
        suffix="ambiguous",
    )
    range_id = "range:ambiguous"
    range_event = _canonical_semantic_event(
        MarketEvent(
            event_id="pending-canonical-id",
            kind=EventKind.DEALING_RANGE_ACTIVATED,
            observed_at=resolved,
            timeframe=Timeframe.H1,
            side=None,
            price=100.0,
            strength=0.8,
            event_time=resolved,
            known_at=resolved,
            evidence={"range_id": range_id},
            source_event_ids=("event:range-created", "event:h1-bar"),
            source_entity_ids=(range_id,),
            origin=EventOrigin.SEMANTIC_ATOMIC,
        )
    )
    balance = BalanceContext(
        context_id=range_id,
        timeframe=Timeframe.H1,
        status="authoritative",
        source_ids=(range_id,),
        bilateral_boundaries=True,
        internal_crossing=True,
        accepted_external_break=False,
        value_authoritative=True,
    )

    with pytest.raises(ValueError, match="ambiguous path realized-winner"):
        brain._update_shadow_path_diagnostics(
            _runtime_observation(
                resolved,
                retained=(raw, terminal, range_event),
                current=(terminal, range_event),
            ),
            replace(context, updated_at=resolved, balance_context=balance),
            {},
            {},
            None,
        )
    assert brain._path_competition_state == first
    assert brain.hypothesis_manager.state == first


@pytest.mark.parametrize("defect", ("forged_parent_id", "wrong_authority"))
def test_resolution_rejects_noncanonical_or_wrongly_bound_raw_parent(
    defect: str,
) -> None:
    t0 = _clock("2025-01-06 09:00")
    crossed = _clock("2025-01-06 10:00")
    resolved = _clock("2025-01-06 14:00")
    brain = PlaybookBrain()
    first, context = _initialize(brain, t0)
    raw, terminal = _production_bos_resolution(
        crossed_at=crossed,
        resolved_at=resolved,
        timeframe=Timeframe.H4,
        scope="continuation",
        direction=Direction.LONG,
        source_structure_id=(
            first.authority_structure_id
            if defect == "forged_parent_id"
            else "unrelated-structure"
        ),
        accepted=True,
        suffix=defect,
    )
    if defect == "forged_parent_id":
        forged_id = "forged-raw-break-id"
        raw = replace(raw, event_id=forged_id)
        terminal = _canonical_semantic_event(
            replace(
                terminal,
                event_id="pending-canonical-id",
                context_event_ids=(
                    forged_id,
                    terminal.context_event_ids[1],
                ),
            )
        )

    state, records, _, _ = brain._update_shadow_path_diagnostics(
        _runtime_observation(
            resolved,
            retained=(raw, terminal),
            current=(terminal,),
        ),
        replace(context, updated_at=resolved),
        {},
        {},
        None,
    )

    assert state is not None and state.status is PathStatus.ACTIVE
    assert records[0].applied_outcome_event is None


@pytest.mark.parametrize(
    ("connected", "evidence_matches", "realized"),
    (
        (True, True, True),
        (False, True, False),
        (True, False, False),
    ),
)
def test_deeper_retracement_requires_exact_graph_connected_lower_scale(
    connected: bool,
    evidence_matches: bool,
    realized: bool,
) -> None:
    t0 = _clock("2025-01-06 09:55")
    crossed = _clock("2025-01-06 10:00")
    resolved = _clock("2025-01-06 10:05")
    brain = PlaybookBrain()
    first, context = _initialize(brain, t0)
    raw, terminal = _production_bos_resolution(
        crossed_at=crossed,
        resolved_at=resolved,
        timeframe=Timeframe.M5,
        scope="opposed",
        direction=Direction.SHORT,
        source_structure_id="m5-local-structure",
        accepted=True,
        suffix=f"deeper-{connected}-{evidence_matches}",
    )
    relation = ScaleRelationState(
        timeframe=Timeframe.M5,
        relation=(
            ScaleRelation.NORMAL_PULLBACK
            if connected
            else ScaleRelation.UNKNOWN
        ),
        direction=Direction.SHORT,
        authority_layer_id=first.authority_structure_id,
        evidence_ids=(
            str(raw.evidence["bos_id"])
            if evidence_matches
            else "bos:unrelated-connected-relation",
        ),
        evidence_kind="bos",
        structural_scope="internal",
        acceptance_state="accepted",
        since=crossed,
        age_bars=1,
        graph_connected=connected,
        ambiguous=False,
    )
    local_terminal = object()
    state, records, _, _ = brain._update_shadow_path_diagnostics(
        _runtime_observation(
            resolved,
            retained=(raw, terminal),
            current=(terminal,),
        ),
        replace(
            context,
            updated_at=resolved,
            scale_relation_details={
                **context.scale_relation_details,
                Timeframe.M5.value: relation,
            },
        ),
        {},
        {"entry-episode:terminal": local_terminal},
        None,
    )

    assert state is not None
    if realized:
        assert state.status is PathStatus.REALIZED
        assert state.winner_path is PathKind.DEEPER_RETRACEMENT
    else:
        assert state.status is PathStatus.ACTIVE
        assert records[0].applied_outcome_event is None


def test_common_horizon_expiry_precedes_same_clock_realization() -> None:
    formed = _clock("2025-01-06 12:59")
    crossed = _clock("2025-01-06 13:00")
    expires = _clock("2025-01-06 17:00")
    brain = PlaybookBrain()
    first, context = _initialize(brain, formed)
    raw, terminal = _production_bos_resolution(
        crossed_at=crossed,
        resolved_at=expires,
        timeframe=Timeframe.H4,
        scope="continuation",
        direction=Direction.LONG,
        source_structure_id=first.authority_structure_id,
        accepted=True,
        suffix="horizon",
    )

    state, records, rankings, exclusions = (
        brain._update_shadow_path_diagnostics(
            _runtime_observation(
                expires,
                retained=(raw, terminal),
                current=(terminal,),
            ),
            replace(context, updated_at=expires),
            {},
            {},
            None,
        )
    )

    assert state is not None and state.status is PathStatus.EXPIRED
    assert state.winner_path is None
    assert records[0].common_horizon_expired
    assert rankings == {} and exclusions == {}
