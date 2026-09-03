from __future__ import annotations

from dataclasses import replace
import hashlib

import pandas as pd
import pytest

from smc_trader.market_state import (
    LiquidityRangeRole,
    TimeframeRangeState,
    build_structural_legs,
    reduce_timeframe_state,
)
from smc_trader.model import (
    BOSLifecycle,
    BOSPostBreakState,
    BOSScope,
    BreakOfStructureState,
    Candle,
    Direction,
    EventKind,
    EventOrigin,
    FrameObservation,
    MarketEvent,
    SMC_SEMANTIC_VERSION,
    StructuralLegState,
    SwingLifecycle,
    SwingPoint,
    SwingRank,
    SwingRelation,
    SwingSide,
    Timeframe,
)
from smc_trader.observation import CausalObserver, ObserverConfig
from smc_trader.semantic_event_emitter import SemanticEventEmitter
from smc_trader.semantics import SemanticRegistry

from .helpers import CORE_TEST_SCALE_SPECS


TZ = "America/New_York"
V1_1_REGISTRY_SHA256 = (
    "344fcf9cf285687fa23df15a922a7e5073296883d515d43888a0b54b7c84e7da"
)
V1_1_PARAMETERS_SHA256 = (
    "91827ca81864099cf09acd6acc626ff25b6bcd39a4437f15bcfe5c7f91068cba"
)


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2026-08-19 10:00", tz=TZ) + pd.Timedelta(minutes=minutes)


def _event(
    kind: EventKind,
    minutes: int,
    *,
    evidence: dict[str, object],
    event_id: str | None = None,
    direction: Direction | None = None,
    price: float | None = 100.0,
    side: str | None = None,
    zone: tuple[float, float] | None = None,
    source_event_ids: tuple[str, ...] = (),
    source_entity_ids: tuple[str, ...] = (),
    origin: EventOrigin = EventOrigin.SEMANTIC_ATOMIC,
) -> MarketEvent:
    known_at = _clock(minutes)
    return MarketEvent(
        event_id=event_id or f"{minutes:02d}:{kind.value}",
        kind=kind,
        observed_at=known_at,
        timeframe=Timeframe.M5,
        side=side,
        price=price,
        strength=0.75,
        details=evidence,
        direction=direction,
        sequence_no=minutes,
        event_time=known_at,
        known_at=known_at,
        semantic_version=SMC_SEMANTIC_VERSION,
        evidence=evidence,
        zone=zone,
        source_event_ids=source_event_ids,
        source_entity_ids=source_entity_ids,
        origin=origin,
    )


def _replay(events: tuple[MarketEvent, ...]):
    state = None
    for event in events:
        state = reduce_timeframe_state(
            state,
            event,
            semantic_registry_identity="v1.2-eye-test",
        )
    assert state is not None
    return state


def _protected_crossing_observer() -> CausalObserver:
    return CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=False,
        )
    )


def _assign_protected_swing(
    observer: CausalObserver,
    *,
    swing_id: str,
    event_minutes: int,
    direction: Direction,
    timeframe: Timeframe = Timeframe.M5,
) -> MarketEvent:
    assignment = observer._emitter._append_semantic_atomic(
        EventKind.PROTECTED_SWING_ASSIGNED,
        _clock(event_minutes),
        timeframe,
        "below" if direction is Direction.LONG else "above",
        95.0 if direction is Direction.LONG else 105.0,
        0.75,
        evidence={
            "protected_swing_id": swing_id,
            "bos_id": f"bos:{event_minutes}",
            "structure_id": f"structure:{event_minutes}",
            "origin_leg_id": f"leg:{event_minutes}",
            "break_standard": "later_acceptance_beyond",
        },
        direction=direction,
        source_entity_ids=(
            f"bos:{event_minutes}",
            f"structure:{event_minutes}",
            f"leg:{event_minutes}",
            swing_id,
        ),
    )
    observer._emitter._protected_swing_event_ids[swing_id] = assignment.event_id
    return assignment


def _append_protected_crossing_terminal(
    observer: CausalObserver,
    *,
    swing_id: str,
    crossed_minutes: int,
    resolved_minutes: int,
    direction: Direction,
    kind: EventKind = EventKind.ACCEPTANCE_CONFIRMED,
    timeframe: Timeframe = Timeframe.M5,
    source_timeframe: Timeframe | None = None,
) -> MarketEvent:
    side = "above" if direction is Direction.LONG else "below"
    return observer._emitter._append_crossing_resolution(
        kind,
        _clock(resolved_minutes),
        timeframe,
        side,
        105.0 if side == "above" else 95.0,
        0.75,
        (
            f"penetration:{crossed_minutes}",
            f"resolution-bar:{resolved_minutes}",
        ),
        {
            "level_id": f"swing:{swing_id}",
            **(
                {"source_timeframe": source_timeframe.value}
                if source_timeframe is not None
                else {}
            ),
        },
        direction=direction,
        crossed_at=_clock(crossed_minutes),
        known_at=_clock(resolved_minutes),
    )


def _record_continuation_break(
    observer: CausalObserver,
    *,
    identity: str,
    direction: Direction,
    origin_price: float,
    target_price: float,
    timeframe: Timeframe = Timeframe.M5,
    base_minutes: int = 10,
) -> tuple[MarketEvent, ...]:
    """Exercise the real canonical continuation producer with frozen inputs."""

    origin_id = f"{identity}:origin"
    target_id = f"{identity}:target"
    leg_id = f"{identity}:leg"
    structure_id = f"{identity}:structure"
    bos_id = f"{identity}:bos"
    break_bar_id = f"{identity}:break-bar"
    origin_side = (
        SwingSide.LOW if direction is Direction.LONG else SwingSide.HIGH
    )
    target_side = (
        SwingSide.HIGH if direction is Direction.LONG else SwingSide.LOW
    )
    origin = SwingPoint(
        swing_id=origin_id,
        timeframe=timeframe,
        symbol="NQ",
        instrument_id=1,
        side=origin_side,
        price=origin_price,
        price_ticks=int(origin_price / 0.25),
        pivot_start=_clock(base_minutes),
        pivot_end=_clock(base_minutes + 1),
        observed_at=_clock(base_minutes + 2),
        confirmed_at=_clock(base_minutes + 2),
        lifecycle=SwingLifecycle.CONFIRMED,
        relation=(
            SwingRelation.HL
            if direction is Direction.LONG
            else SwingRelation.LH
        ),
        prominence_atr=1.0,
    )
    target = SwingPoint(
        swing_id=target_id,
        timeframe=timeframe,
        symbol="NQ",
        instrument_id=1,
        side=target_side,
        price=target_price,
        price_ticks=int(target_price / 0.25),
        pivot_start=_clock(base_minutes + 3),
        pivot_end=_clock(base_minutes + 4),
        observed_at=_clock(base_minutes + 5),
        confirmed_at=_clock(base_minutes + 5),
        lifecycle=SwingLifecycle.CONFIRMED,
        relation=(
            SwingRelation.HH
            if direction is Direction.LONG
            else SwingRelation.LL
        ),
        prominence_atr=1.0,
    )
    leg = StructuralLegState(
        leg_id=leg_id,
        timeframe=timeframe,
        direction=direction,
        start_swing_id=origin_id,
        end_swing_id=target_id,
        start_event_time=origin.pivot_start,
        end_event_time=target.pivot_start,
        known_at=target.confirmed_at,
        start_price=origin_price,
        end_price=target_price,
        start_close=origin_price,
        end_close=target_price,
        amplitude_points=abs(target_price - origin_price),
        amplitude_atr=abs(target_price - origin_price) / 2.0,
        duration_bars=4,
        duration_minutes=3,
        efficiency=0.8,
        max_retracement_points=1.0,
        max_retracement_atr=0.5,
        source_swing_ids=(origin_id, target_id),
    )
    resolved_at = _clock(base_minutes + 8)
    break_state = BreakOfStructureState(
        bos_id=bos_id,
        timeframe=timeframe,
        direction=direction,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
        target_swing_id=target_id,
        source_structure_id=structure_id,
        target_price=target_price,
        target_ticks=int(target_price / 0.25),
        pending_at=_clock(base_minutes + 6),
        resolved_at=resolved_at,
        age_bars=1,
        strength=0.75,
        break_bar_id=break_bar_id,
        break_distance_atr=0.5,
        post_break_state=BOSPostBreakState.PENDING,
    )

    observer._emitter._confirmed_swing_event_ids.update(
        {
            origin_id: f"{origin_id}:event",
            target_id: f"{target_id}:event",
        }
    )
    observer._emitter._structural_leg_event_ids[leg_id] = f"{leg_id}:event"
    observer._emitter._structure_direction_event_ids[structure_id] = (
        f"{structure_id}:event"
    )
    observer._emitter._bar_event_ids_by_candle_id[break_bar_id] = (
        f"{break_bar_id}:event"
    )
    observer._emitter._bar_close_by_candle_id[break_bar_id] = (
        target_price + 0.25
        if direction is Direction.LONG
        else target_price - 0.25
    )
    observer._emitter._real_bar_event_ids_by_timeframe[timeframe].append(
        (resolved_at, f"{break_bar_id}:event")
    )
    observer._emitter._known_level_ids.update(
        {
            (origin_id, SwingLifecycle.CONFIRMED),
            (target_id, SwingLifecycle.CONFIRMED),
        }
    )
    observer._emitter._known_structural_leg_ids.add(leg_id)
    before = len(observer.memory._audit_pending)
    observer._emitter._record_frame_events(
        FrameObservation(
            timeframe=timeframe,
            cutoff=resolved_at,
            bars=9,
            metrics={},
            swings=(origin, target),
            structure_breaks=(break_state,),
            structural_legs=(leg,),
        ),
        newly_completed=True,
            prior=observer._prior,
    )
    return tuple(observer.memory._audit_pending[before:])


def _hierarchy_events() -> tuple[MarketEvent, ...]:
    first = _event(
        EventKind.SWING_CONFIRMED,
        0,
        event_id="swing-event-a",
        price=95.0,
        side="below",
        evidence={"source_entity_id": "swing-a", "semantic_rank": "micro"},
        source_entity_ids=("swing-a",),
    )
    second = _event(
        EventKind.SWING_CONFIRMED,
        1,
        event_id="swing-event-b",
        price=105.0,
        side="above",
        evidence={"source_entity_id": "swing-b", "semantic_rank": "micro"},
        source_entity_ids=("swing-b",),
    )
    leg = _event(
        EventKind.STRUCTURAL_LEG_CREATED,
        2,
        event_id="leg-event",
        price=105.0,
        side="above",
        direction=Direction.LONG,
        source_event_ids=(first.event_id, second.event_id),
        evidence={
            "path_class": "internal",
            "leg_id": "leg-a-b",
            "start_swing_id": "swing-a",
            "end_swing_id": "swing-b",
            "start_event_time": _clock(0).isoformat(),
            "end_event_time": _clock(1).isoformat(),
            "start_price": 95.0,
            "end_price": 105.0,
            "start_close": 96.0,
            "end_close": 104.0,
            "amplitude_points": 10.0,
            "amplitude_atr": 5.0,
            "duration_bars": 3,
            "duration_minutes": 5,
            "efficiency": 0.8,
            "max_retracement_points": 1.0,
            "max_retracement_atr": 0.5,
            "rank": "internal",
        },
        source_entity_ids=("leg-a-b", "swing-a", "swing-b"),
    )
    structure = _event(
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        3,
        event_id="structure-event",
        direction=Direction.LONG,
        source_event_ids=(first.event_id, second.event_id),
        evidence={
            "structure_id": "structure-a-b",
            "source_high_id": "swing-b",
            "source_low_id": "swing-a",
        },
        source_entity_ids=("structure-a-b", "swing-a", "swing-b"),
    )
    protected = _event(
        EventKind.PROTECTED_SWING_ASSIGNED,
        4,
        event_id="protected-event",
        direction=Direction.LONG,
        price=95.0,
        side="below",
        source_event_ids=(leg.event_id, first.event_id),
        evidence={"protected_swing_id": "swing-a"},
        source_entity_ids=("swing-a",),
    )
    return first, second, leg, structure, protected


def test_swing_hierarchy_assignments_are_append_only_causal_and_replay_stable() -> None:
    events = _hierarchy_events()
    snapshots = []
    state = None
    for event in events:
        state = reduce_timeframe_state(
            state,
            event,
            semantic_registry_identity="v1.2-eye-test",
        )
        assert state is not None
        snapshots.append(state)

    assert snapshots[0].swing_hierarchy[0].semantic_rank is SwingRank.MICRO
    assert all(
        view.semantic_rank is SwingRank.INTERNAL
        for view in snapshots[2].swing_hierarchy
    )
    assert all(
        view.semantic_rank is SwingRank.STRUCTURAL
        for view in snapshots[3].swing_hierarchy
    )
    final = {view.swing_id: view for view in snapshots[-1].swing_hierarchy}
    assert final["swing-a"].semantic_rank is SwingRank.EXTERNAL
    assert final["swing-a"].nesting_depth == 3
    assert [item.source_kind for item in final["swing-a"].assignments] == [
        EventKind.SWING_CONFIRMED.value,
        EventKind.STRUCTURAL_LEG_CREATED.value,
        EventKind.STRUCTURE_DIRECTION_CONFIRMED.value,
        EventKind.PROTECTED_SWING_ASSIGNED.value,
    ]
    assert [item.assigned_at for item in final["swing-a"].assignments] == [
        event.known_at for event in events if event is not events[1]
    ]
    assert final["swing-b"].semantic_rank is SwingRank.STRUCTURAL
    assert _replay(events) == snapshots[-1]


@pytest.mark.parametrize(
    ("assignment_direction", "acceptance_direction"),
    (
        (Direction.LONG, Direction.SHORT),
        (Direction.SHORT, Direction.LONG),
    ),
)
def test_exact_protected_acceptance_releases_only_its_assignment_generation(
    assignment_direction: Direction,
    acceptance_direction: Direction,
) -> None:
    observer = _protected_crossing_observer()
    swing_id = f"protected-{assignment_direction.value}"
    first_assignment = _assign_protected_swing(
        observer,
        swing_id=swing_id,
        event_minutes=0,
        direction=assignment_direction,
    )

    first = _append_protected_crossing_terminal(
        observer,
        swing_id=swing_id,
        crossed_minutes=1,
        resolved_minutes=2,
        direction=acceptance_direction,
    )

    assert first.evidence["protected_swing_id"] == swing_id
    assert first.evidence["protected_swing_event_id"] == (
        first_assignment.event_id
    )
    assert first.context_event_ids == (first_assignment.event_id,)
    assert swing_id not in observer._emitter._protected_swing_event_ids

    second = _append_protected_crossing_terminal(
        observer,
        swing_id=swing_id,
        crossed_minutes=3,
        resolved_minutes=4,
        direction=acceptance_direction,
    )

    assert "protected_swing_id" not in second.evidence
    assert "protected_swing_event_id" not in second.evidence
    assert second.context_event_ids == ()

    next_assignment = _assign_protected_swing(
        observer,
        swing_id=swing_id,
        event_minutes=5,
        direction=assignment_direction,
    )
    third = _append_protected_crossing_terminal(
        observer,
        swing_id=swing_id,
        crossed_minutes=6,
        resolved_minutes=7,
        direction=acceptance_direction,
    )

    assert next_assignment.event_id != first_assignment.event_id
    assert third.evidence["protected_swing_event_id"] == (
        next_assignment.event_id
    )
    assert third.context_event_ids == (next_assignment.event_id,)
    assert swing_id not in observer._emitter._protected_swing_event_ids


def test_non_terminal_protected_crossings_do_not_release_the_assignment() -> None:
    observer = _protected_crossing_observer()
    swing_id = "protected-low"
    assignment = _assign_protected_swing(
        observer,
        swing_id=swing_id,
        event_minutes=0,
        direction=Direction.LONG,
    )

    sweep = _append_protected_crossing_terminal(
        observer,
        swing_id=swing_id,
        crossed_minutes=1,
        resolved_minutes=2,
        direction=Direction.LONG,
        kind=EventKind.SWEEP_CONFIRMED,
    )
    same_direction_acceptance = _append_protected_crossing_terminal(
        observer,
        swing_id=swing_id,
        crossed_minutes=3,
        resolved_minutes=4,
        direction=Direction.LONG,
    )

    assert sweep.evidence["protected_swing_event_id"] == assignment.event_id
    assert same_direction_acceptance.context_event_ids == (
        assignment.event_id,
    )
    assert observer._emitter._protected_swing_event_ids[swing_id] == assignment.event_id


def test_wrong_timeframe_protected_context_does_not_release_the_assignment() -> None:
    observer = _protected_crossing_observer()
    swing_id = "protected-low"
    assignment = _assign_protected_swing(
        observer,
        swing_id=swing_id,
        event_minutes=0,
        direction=Direction.LONG,
        timeframe=Timeframe.H1,
    )

    acceptance = _append_protected_crossing_terminal(
        observer,
        swing_id=swing_id,
        crossed_minutes=1,
        resolved_minutes=2,
        direction=Direction.SHORT,
    )

    assert acceptance.context_event_ids == (assignment.event_id,)
    assert observer._emitter._protected_swing_event_ids[swing_id] == assignment.event_id


def test_m1_acceptance_releases_its_exact_h1_candidate_owner_assignment() -> None:
    observer = _protected_crossing_observer()
    swing_id = "protected-h1-low"
    assignment = _assign_protected_swing(
        observer,
        swing_id=swing_id,
        event_minutes=0,
        direction=Direction.LONG,
        timeframe=Timeframe.H1,
    )

    acceptance = _append_protected_crossing_terminal(
        observer,
        swing_id=swing_id,
        crossed_minutes=1,
        resolved_minutes=2,
        direction=Direction.SHORT,
        timeframe=Timeframe.M1,
        source_timeframe=Timeframe.H1,
    )

    assert acceptance.evidence["source_timeframe"] == Timeframe.H1.value
    assert acceptance.evidence["protected_swing_event_id"] == assignment.event_id
    assert swing_id not in observer._emitter._protected_swing_event_ids


def test_stale_protected_acceptance_cannot_clear_a_new_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _protected_crossing_observer()
    swing_id = "protected-low"
    stale_assignment = _assign_protected_swing(
        observer,
        swing_id=swing_id,
        event_minutes=0,
        direction=Direction.LONG,
    )
    next_assignment = observer._emitter._append_semantic_atomic(
        EventKind.PROTECTED_SWING_ASSIGNED,
        _clock(1),
        Timeframe.M5,
        "below",
        95.0,
        0.75,
        evidence={
            "protected_swing_id": swing_id,
            "bos_id": "bos:next",
            "structure_id": "structure:next",
            "origin_leg_id": "leg:next",
            "break_standard": "later_acceptance_beyond",
        },
        direction=Direction.LONG,
        source_entity_ids=(
            "bos:next",
            "structure:next",
            "leg:next",
            swing_id,
        ),
    )
    append = observer._emitter._append_semantic_atomic

    def append_then_reassign(*args: object, **kwargs: object) -> MarketEvent:
        event = append(*args, **kwargs)
        observer._emitter._protected_swing_event_ids[swing_id] = next_assignment.event_id
        return event

    monkeypatch.setattr(observer._emitter, "_append_semantic_atomic", append_then_reassign)

    acceptance = _append_protected_crossing_terminal(
        observer,
        swing_id=swing_id,
        crossed_minutes=2,
        resolved_minutes=3,
        direction=Direction.SHORT,
    )

    assert acceptance.evidence["protected_swing_event_id"] == (
        stale_assignment.event_id
    )
    assert observer._emitter._protected_swing_event_ids[swing_id] == (
        next_assignment.event_id
    )


def test_failed_protected_acceptance_append_keeps_the_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _protected_crossing_observer()
    swing_id = "protected-low"
    assignment = _assign_protected_swing(
        observer,
        swing_id=swing_id,
        event_minutes=0,
        direction=Direction.LONG,
    )

    def fail_append(*args: object, **kwargs: object) -> MarketEvent:
        raise RuntimeError("injected append failure")

    monkeypatch.setattr(observer._emitter, "_append_semantic_atomic", fail_append)

    with pytest.raises(RuntimeError, match="injected append failure"):
        _append_protected_crossing_terminal(
            observer,
            swing_id=swing_id,
            crossed_minutes=1,
            resolved_minutes=2,
            direction=Direction.SHORT,
        )

    assert observer._emitter._protected_swing_event_ids[swing_id] == assignment.event_id
    assert observer._emitter._terminal_crossing_events == {}


def test_future_protected_assignment_cannot_backfill_crossing_context() -> None:
    observer = _protected_crossing_observer()
    swing_id = "future-protected-low"
    assignment = _assign_protected_swing(
        observer,
        swing_id=swing_id,
        event_minutes=3,
        direction=Direction.LONG,
    )

    with pytest.raises(
        ValueError,
        match="protected assignment cannot be known after crossing resolution",
    ):
        _append_protected_crossing_terminal(
            observer,
            swing_id=swing_id,
            crossed_minutes=1,
            resolved_minutes=2,
            direction=Direction.SHORT,
        )

    assert observer._emitter._protected_swing_event_ids[swing_id] == assignment.event_id
    assert observer._emitter._terminal_crossing_events == {}


def test_opposite_continuation_keeps_raw_fact_without_rewriting_live_regime(
) -> None:
    observer = _protected_crossing_observer()
    assignment = _assign_protected_swing(
        observer,
        swing_id="live-low",
        event_minutes=0,
        direction=Direction.LONG,
    )
    assert observer.audit_store.get(assignment.event_id) is None

    emitted = _record_continuation_break(
        observer,
        identity="opposite-short",
        direction=Direction.SHORT,
        origin_price=105.0,
        target_price=90.0,
    )
    kinds = {event.kind for event in emitted}

    assert EventKind.RAW_BOUNDARY_BREAK in kinds
    assert EventKind.QUALIFIED_BOS not in kinds
    assert EventKind.PROTECTED_SWING_ASSIGNED not in kinds
    snapshot = _replay((*_hierarchy_events(), *emitted))
    assert snapshot.structure.external_direction is Direction.LONG
    assert snapshot.structure.protected_low_id == "swing-a"


def test_exact_acceptance_releases_then_allows_reverse_continuation() -> None:
    observer = _protected_crossing_observer()
    _assign_protected_swing(
        observer,
        swing_id="live-low",
        event_minutes=0,
        direction=Direction.LONG,
    )
    _append_protected_crossing_terminal(
        observer,
        swing_id="live-low",
        crossed_minutes=1,
        resolved_minutes=2,
        direction=Direction.SHORT,
    )
    assert "live-low" not in observer._emitter._protected_swing_event_ids

    emitted = _record_continuation_break(
        observer,
        identity="released-short",
        direction=Direction.SHORT,
        origin_price=104.0,
        target_price=90.0,
    )
    by_kind = {event.kind: event for event in emitted}

    assert by_kind[EventKind.QUALIFIED_BOS].direction is Direction.SHORT
    replacement = by_kind[EventKind.PROTECTED_SWING_ASSIGNED]
    assert replacement.direction is Direction.SHORT
    assert observer._emitter._protected_swing_event_ids == {
        "released-short:origin": replacement.event_id
    }


@pytest.mark.parametrize(
    ("direction", "prior_id", "origin_price", "target_price"),
    (
        (Direction.LONG, "prior-low", 97.0, 110.0),
        (Direction.SHORT, "prior-high", 103.0, 90.0),
    ),
)
def test_same_direction_continuation_replaces_protection_monotonically(
    direction: Direction,
    prior_id: str,
    origin_price: float,
    target_price: float,
) -> None:
    observer = _protected_crossing_observer()
    prior = _assign_protected_swing(
        observer,
        swing_id=prior_id,
        event_minutes=0,
        direction=direction,
    )

    emitted = _record_continuation_break(
        observer,
        identity=f"reinforce-{direction.value}",
        direction=direction,
        origin_price=origin_price,
        target_price=target_price,
    )
    assignments = tuple(
        event
        for event in emitted
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
    )

    assert len(assignments) == 1
    replacement = assignments[0]
    assert replacement.event_id != prior.event_id
    assert replacement.price == origin_price
    assert observer._emitter._protected_swing_event_ids == {
        f"reinforce-{direction.value}:origin": replacement.event_id
    }


def test_same_direction_continuation_cannot_loosen_live_protection() -> None:
    observer = _protected_crossing_observer()
    prior = _assign_protected_swing(
        observer,
        swing_id="prior-low",
        event_minutes=0,
        direction=Direction.LONG,
    )

    emitted = _record_continuation_break(
        observer,
        identity="weaker-long",
        direction=Direction.LONG,
        origin_price=94.0,
        target_price=110.0,
    )
    kinds = {event.kind for event in emitted}

    assert EventKind.QUALIFIED_BOS in kinds
    assert EventKind.PROTECTED_SWING_ASSIGNED not in kinds
    assert observer._emitter._protected_swing_event_ids == {
        "prior-low": prior.event_id
    }


def test_protected_assignment_custody_is_independent_across_timeframes() -> None:
    observer = _protected_crossing_observer()
    h1_assignment = _assign_protected_swing(
        observer,
        swing_id="h1-live-low",
        event_minutes=0,
        direction=Direction.LONG,
        timeframe=Timeframe.H1,
    )

    emitted = _record_continuation_break(
        observer,
        identity="m5-short",
        direction=Direction.SHORT,
        origin_price=104.0,
        target_price=90.0,
        timeframe=Timeframe.M5,
    )
    m5_assignment = next(
        event
        for event in emitted
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
    )

    assert observer._emitter._protected_swing_event_ids == {
        "h1-live-low": h1_assignment.event_id,
        "m5-short:origin": m5_assignment.event_id,
    }


def test_failed_reinforcement_append_preserves_exact_live_custody(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _protected_crossing_observer()
    prior = _assign_protected_swing(
        observer,
        swing_id="prior-low",
        event_minutes=0,
        direction=Direction.LONG,
    )
    original_append = observer._emitter._append_semantic_atomic

    def fail_assignment(*args: object, **kwargs: object) -> MarketEvent:
        if args and args[0] is EventKind.PROTECTED_SWING_ASSIGNED:
            raise RuntimeError("injected protected-assignment append failure")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(observer._emitter, "_append_semantic_atomic", fail_assignment)

    with pytest.raises(RuntimeError, match="injected protected-assignment"):
        _record_continuation_break(
            observer,
            identity="failed-reinforcement",
            direction=Direction.LONG,
            origin_price=97.0,
            target_price=110.0,
        )

    assert observer._emitter._protected_swing_event_ids == {
        "prior-low": prior.event_id
    }


def test_structural_leg_rank_does_not_backfill_future_authority() -> None:
    starts = tuple(_clock(index * 5) for index in range(4))
    candles = tuple(
        Candle(
            timeframe=Timeframe.M5,
            start=start,
            end=start + pd.Timedelta(minutes=5),
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.5 + index,
            volume=100.0,
            symbol="NQ",
            instrument_id=1,
            observed_minutes=5,
            expected_minutes=5,
            complete=True,
        )
        for index, start in enumerate(starts)
    )
    low = SwingPoint(
        swing_id="low",
        timeframe=Timeframe.M5,
        symbol="NQ",
        instrument_id=1,
        side=SwingSide.LOW,
        price=99.0,
        price_ticks=396,
        pivot_start=starts[0],
        pivot_end=starts[0] + pd.Timedelta(minutes=5),
        observed_at=_clock(10),
        confirmed_at=_clock(10),
        lifecycle=SwingLifecycle.CONFIRMED,
        semantic_rank=SwingRank.MICRO,
    )
    high = replace(
        low,
        swing_id="high",
        side=SwingSide.HIGH,
        price=104.0,
        price_ticks=416,
        pivot_start=starts[3],
        pivot_end=starts[3] + pd.Timedelta(minutes=5),
        observed_at=_clock(25),
        confirmed_at=_clock(25),
    )

    baseline = build_structural_legs(Timeframe.M5, (low, high), candles, atr=2.0)
    future_authority = build_structural_legs(
        Timeframe.M5,
        (low, high),
        candles,
        atr=2.0,
        protected_swing_ids=("low",),
        structural_swing_ids=("high",),
    )

    assert baseline == future_authority
    assert baseline[0].path_class is SwingRank.INTERNAL


def test_candidate_irl_erl_requires_an_active_same_timeframe_range() -> None:
    candidate_events = tuple(
        _event(
            EventKind.LIQUIDITY_LEVEL_CREATED,
            minute,
            event_id=f"candidate-{name}",
            price=price,
            side="above" if price >= 100.0 else "below",
            evidence={"level_id": name, "source_kind": "previous_day_high"},
            source_entity_ids=(name,),
            origin=EventOrigin.LEGACY_TRANSPORT,
        )
        for minute, name, price in (
            (0, "inside", 100.0),
            (1, "low-boundary", 90.0),
            (2, "high-boundary", 110.0),
            (3, "outside", 115.0),
        )
    )
    created = _event(
        EventKind.DEALING_RANGE_CREATED,
        4,
        event_id="range-created",
        price=100.0,
        evidence={"range_id": "range-1"},
        zone=(90.0, 110.0),
        origin=EventOrigin.LEGACY_TRANSPORT,
    )
    # v1.3 activates the range through the balance-maturity fact; the
    # location lifecycle no longer carries an activation of its own.
    activated = _event(
        EventKind.BALANCE_RANGE_MATURED,
        5,
        event_id="range-active",
        price=100.0,
        evidence={"range_id": "range-1"},
        zone=(90.0, 110.0),
        origin=EventOrigin.LEGACY_TRANSPORT,
    )
    invalidated = _event(
        EventKind.DEALING_RANGE_INVALIDATED,
        6,
        event_id="range-invalidated",
        price=111.0,
        evidence={"range_id": "range-1"},
        origin=EventOrigin.LEGACY_TRANSPORT,
    )

    no_range = _replay(candidate_events)
    assert all(
        item.range_role is LiquidityRangeRole.UNRESOLVED
        and item.normalized_location_in_range is None
        for item in no_range.liquidity.candidates
    )
    forming = reduce_timeframe_state(
        no_range,
        created,
        semantic_registry_identity="v1.2-eye-test",
    )
    assert forming is not None
    # v1.3: the structural interval locates price from creation.  Waiting for
    # the balance claim made a two-sided-test statistic a precondition for
    # arithmetic the interval could already do.
    forming_by_id = {
        item.candidate_id: item for item in forming.liquidity.candidates
    }
    assert forming_by_id["inside"].range_role is LiquidityRangeRole.IRL
    assert forming_by_id["inside"].normalized_location_in_range == 0.5
    active = reduce_timeframe_state(
        forming,
        activated,
        semantic_registry_identity="v1.2-eye-test",
    )
    assert active is not None
    by_id = {item.candidate_id: item for item in active.liquidity.candidates}
    assert by_id["inside"].range_role is LiquidityRangeRole.IRL
    assert by_id["inside"].normalized_location_in_range == 0.5
    for identity in ("low-boundary", "high-boundary", "outside"):
        assert by_id[identity].range_role is LiquidityRangeRole.ERL
    assert by_id["low-boundary"].normalized_location_in_range == 0.0
    assert by_id["high-boundary"].normalized_location_in_range == 1.0
    assert by_id["outside"].normalized_location_in_range == 1.25

    terminal = reduce_timeframe_state(
        active,
        invalidated,
        semantic_registry_identity="v1.2-eye-test",
    )
    assert terminal is not None
    assert all(
        item.range_role is LiquidityRangeRole.UNRESOLVED
        and item.normalized_location_in_range is None
        for item in terminal.liquidity.candidates
    )


def test_v1_1_artifacts_remain_byte_stable_and_explicitly_loadable() -> None:
    old = SemanticRegistry.from_file(
        "semantics/registry.yaml",
        required_version="smc_semantics_v1.1",
    )

    assert old.source_sha256 == V1_1_REGISTRY_SHA256
    assert old.parameters.source_sha256 == V1_1_PARAMETERS_SHA256
    assert hashlib.sha256(old.source_path.read_bytes()).hexdigest() == (
        V1_1_REGISTRY_SHA256
    )
    assert hashlib.sha256(old.parameters.source_path.read_bytes()).hexdigest() == (
        V1_1_PARAMETERS_SHA256
    )
    current = SemanticRegistry.from_file()
    assert current.semantic_version == SMC_SEMANTIC_VERSION == "smc_semantics_v1.3"
    assert ObserverConfig().semantic_registry == "semantics/registry_v1_3.yaml"


def test_reserved_and_compatibility_events_fail_closed_at_atomic_emitter() -> None:
    emitter = object.__new__(SemanticEventEmitter)
    emitter.semantic_registry = SemanticRegistry.from_file()
    by_kind = emitter.semantic_registry.event_binding_by_kind
    expected = {
        EventKind.FVG_TOUCHED: "compatibility_alias_not_emitted",
        EventKind.ORIGIN_ZONE_TOUCHED: "compatibility_alias_not_emitted",
        EventKind.DELIVERY_PHASE_CHANGED: "compatibility_alias_not_emitted",
        EventKind.FVG_EXPIRED: "reserved_not_emitted",
        EventKind.DEALING_RANGE_EXTENDED: "reserved_not_emitted",
    }

    for kind, status in expected.items():
        assert by_kind[kind].status == status
        with pytest.raises(
            ValueError,
            match="canonical semantic emitter rejects non-emitted event kind",
        ):
            emitter._append_semantic_atomic(
                kind,
                _clock(0),
                Timeframe.M5,
                None,
                100.0,
                0.0,
            )
