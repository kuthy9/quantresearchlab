from __future__ import annotations

from dataclasses import replace
import pickle

import pytest

from smc_trader.event_store import EventStore
from smc_trader.current_facts import (
    CurrentMarketFacts,
    _CURRENT_FACT_FAMILIES,
)
from smc_trader.market_state import (
    MARKET_SNAPSHOT_SCHEMA_VERSION,
    MarketSnapshot,
    MarketSnapshotPublisher,
    TimeframeEventReducer,
    _projection_event,
    foundation_record_projection_event,
)
from smc_trader.model import (
    Direction,
    EventKind,
    EventOrigin,
    FrameObservation,
    MarketEvent,
    PathSequenceLifecycle,
    Timeframe,
    to_primitive,
)
from smc_trader.observation import CausalObserver, EventMemory
from smc_trader.semantics import SemanticRegistry

from .test_event_provenance_contract import (
    _authoritative_phase23_chain,
    _crossing_chain,
    _event as _canonical_event,
    _external_range_chain,
    _normalized_bar as _canonical_bar,
)
from .test_interaction_eye_brain_boundary import _held_boundary_interaction
from .test_phase234_atomic_reducer import (
    _bar_event,
    _clock,
    _event,
    _foundation_record_history,
    _m1_candle,
)
from .test_v3_group5_primitives import _m1


def _reducer(store: EventStore) -> TimeframeEventReducer:
    return TimeframeEventReducer(
        event_store=store,
        semantic_registry_identity="definition-test",
    )


def _frames(asof) -> dict[Timeframe, FrameObservation]:
    return {
        Timeframe.H1: FrameObservation(
            timeframe=Timeframe.H1,
            cutoff=asof,
            bars=1,
            metrics={"atr": 2.0},
            ready=True,
        ),
        Timeframe.M1: FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=asof,
            bars=1,
            metrics={"atr": 1.0},
            ready=True,
        ),
    }


def _phase23_protected_prefix() -> tuple[
    tuple[MarketEvent, ...], MarketEvent, MarketEvent, MarketEvent, MarketEvent
]:
    chain = _authoritative_phase23_chain()
    protected_index = next(
        index
        for index, event in enumerate(chain)
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
    )
    prefix = chain[: protected_index + 1]
    raw = next(event for event in prefix if event.kind is EventKind.RAW_BOUNDARY_BREAK)
    qualified = next(event for event in prefix if event.kind is EventKind.QUALIFIED_BOS)
    direction = next(
        event
        for event in prefix
        if event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED
    )
    return prefix, raw, qualified, direction, prefix[-1]


def _bos_terminal_chain(
    *,
    prefix: str,
    raw: MarketEvent,
    kind: EventKind,
    candidate_minutes: int,
    strength: float = 0.75,
) -> tuple[MarketEvent, ...]:
    crossing = list(
        _crossing_chain(
            prefix=prefix,
            kind=kind,
            timeframe=raw.timeframe,
            side=str(raw.side),
            level_id=f"level:{raw.evidence['bos_id']}",
            candidate_minutes=candidate_minutes,
            crossed_minutes=candidate_minutes + 1,
            resolved_minutes=candidate_minutes + 2,
            terminal_details={"bos_id": raw.evidence["bos_id"]},
        )
    )
    crossing[-1] = replace(
        crossing[-1],
        context_event_ids=(raw.event_id,),
        strength=strength,
    )
    return tuple(crossing)


def _next_continuation_bos(
    *,
    minutes: int,
    bos_id: str,
    direction_parent: MarketEvent,
) -> tuple[MarketEvent, MarketEvent, MarketEvent]:
    bar = _canonical_bar(
        f"bar:{bos_id}",
        minutes,
        timeframe=direction_parent.timeframe,
        close=106.0,
        high=107.0,
        low=102.0,
        sequence_no=0,
    )
    raw = _canonical_event(
        f"raw:{bos_id}",
        minutes,
        canonical=True,
        kind=EventKind.RAW_BOUNDARY_BREAK,
        timeframe=direction_parent.timeframe,
        direction=Direction.LONG,
        side="above",
        price=105.0,
        sequence_no=1,
        source_event_ids=("swing-right", bar.event_id),
        source_data_ids=(f"detector:{bar.event_id}",),
        source_entity_ids=(bos_id, "swing-b", "structure-a"),
        details={
            "bos_id": bos_id,
            "target_swing_id": "swing-b",
            "scope": "continuation",
            "break_bar_id": f"detector:{bar.event_id}",
            "break_close": 106.0,
            "break_buffer_ticks": 0,
            "comparison": "strict_close_beyond",
            "break_standard": "close_beyond_confirmed_boundary",
        },
    )
    qualified = _canonical_event(
        f"qualified:{bos_id}",
        minutes,
        canonical=True,
        kind=EventKind.QUALIFIED_BOS,
        timeframe=direction_parent.timeframe,
        direction=Direction.LONG,
        side="above",
        price=105.0,
        sequence_no=2,
        source_event_ids=(raw.event_id, direction_parent.event_id),
        source_entity_ids=(bos_id, "structure-a"),
        details={
            "bos_id": bos_id,
            "scope": "continuation",
            "qualification": "aligned_with_confirmed_structure",
        },
    )
    return bar, raw, qualified


def test_store_driven_reducer_keeps_10k_interaction_deltas_out_of_hot_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EventStore()
    reducer = _reducer(store)
    events = tuple(
        _event(
            EventKind.ENTRY_PATH_STEP,
            0,
            Timeframe.M1,
            event_id=f"interaction-step:{index:05d}",
            sequence_no=index,
            evidence={"step_id": f"step:{index:05d}"},
        )
        for index in range(10_000)
    )
    store.append_batch(events)

    suffix_calls = 0
    original_events_since = store.events_since

    def events_since(index: int):
        nonlocal suffix_calls
        suffix_calls += 1
        return original_events_since(index)

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("hot reducer scanned EventStore history")

    monkeypatch.setattr(store, "events_since", events_since)
    monkeypatch.setattr(store, "events", forbidden_scan)
    monkeypatch.setattr(store, "prefix_fingerprint", forbidden_scan)

    assert reducer.consume_available() == events
    assert suffix_calls == 1
    assert reducer.cursor == 10_000
    assert reducer.states == {}
    assert reducer.current_facts().structure == ()
    assert not any(
        hasattr(reducer, name)
        for name in (
            "_event_ids",
            "_events_by_id",
            "_normalized_bar_event_ids",
            "_terminal_crossing_event_ids",
            "_latest_protected_assignment_event_ids",
            "_unresolved_forward_reference_ids",
        )
    )
    hot_bytes = pickle.dumps(
        (
            reducer.states,
            reducer.cursor,
            reducer._epoch_cursor,
            reducer._last_order_key,
            reducer._current_fact_event_ids,
            reducer._latest_real_m1_event_id,
            reducer.expected_timeframes,
            reducer._store_prefix_fingerprint,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    assert len(hot_bytes) < 2_048


def test_current_fact_surface_is_exact_registry_emitted_atomic_contract() -> None:
    registry = SemanticRegistry.from_file()
    assert set(_CURRENT_FACT_FAMILIES) == set(
        registry.canonical_emitted_event_kinds
    )
    assert EventKind.FVG_TOUCHED not in _CURRENT_FACT_FAMILIES
    assert EventKind.FVG_EXPIRED not in _CURRENT_FACT_FAMILIES
    assert EventKind.ORIGIN_ZONE_TOUCHED not in _CURRENT_FACT_FAMILIES
    assert EventKind.DEALING_RANGE_EXTENDED not in _CURRENT_FACT_FAMILIES
    assert EventKind.DEALING_RANGE_REPLACED in _CURRENT_FACT_FAMILIES

    canonical = _canonical_event(
        "direct-current-range",
        0,
        canonical=True,
        kind=EventKind.DEALING_RANGE_CREATED,
        zone=(90.0, 110.0),
        source_entity_ids=("range-direct",),
        details={"range_id": "range-direct"},
    )
    legacy = replace(canonical, origin=EventOrigin.LEGACY_TRANSPORT)
    with pytest.raises(ValueError, match="canonical atomic origin"):
        CurrentMarketFacts(ranges=(legacy,))

    class DerivedMarketEvent(MarketEvent):
        pass

    derived = DerivedMarketEvent(**canonical.__dict__)
    with pytest.raises(TypeError, match="MarketEvent values"):
        CurrentMarketFacts(ranges=(derived,))
    facts = CurrentMarketFacts(ranges=(canonical,))
    with pytest.raises(AttributeError):
        object.__setattr__(facts, "legacy_history", (canonical,))


def test_reducer_origin_gate_skips_transport_and_rejects_forged_authority() -> None:
    root = _canonical_bar(
        "origin-root",
        0,
        timeframe=Timeframe.M1,
    )
    store = EventStore.from_events((root,))
    reducer = _reducer(store)
    reducer.consume_available()
    before_states = dict(reducer.states)
    before_facts = reducer.current_facts()

    projection = _projection_event(
        EventKind.TIMEFRAME_STATE_CHANGED,
        asof=_clock(1),
        timeframe=Timeframe.M1,
        state_id="forged-legacy-state",
        state={"external_direction": "short"},
        source_event_ids=(root.event_id,),
    )
    legacy_path = _event(
        EventKind.ENTRY_PATH_STATE,
        2,
        Timeframe.M1,
        event_id="legacy-path-transport",
    )
    store.append_batch((projection, legacy_path))
    reducer.consume_available()
    assert reducer.states == before_states
    assert reducer.current_facts() == before_facts

    before_checkpoint_failure = pickle.dumps(
        (
            reducer.states,
            reducer.cursor,
            reducer._epoch_cursor,
            reducer._last_order_key,
            reducer._current_fact_event_ids,
            reducer._store_prefix_fingerprint,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    invalid_state = dict(reducer.__getstate__())
    invalid_state["_store_prefix_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="not store-bound"):
        reducer.__setstate__(invalid_state)
    assert pickle.dumps(
        (
            reducer.states,
            reducer.cursor,
            reducer._epoch_cursor,
            reducer._last_order_key,
            reducer._current_fact_event_ids,
            reducer._store_prefix_fingerprint,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    ) == before_checkpoint_failure
    assert reducer.cursor == len(store)
    assert reducer._store_prefix_fingerprint == store.fingerprint()

    forged = _event(
        EventKind.DEALING_RANGE_CREATED,
        3,
        Timeframe.M1,
        event_id="forged-legacy-current-range",
        evidence={"range_id": "forged-range"},
        zone=(90.0, 110.0),
    )
    store.append(forged)
    before = (
        reducer.cursor,
        dict(reducer.states),
        dict(reducer._current_fact_event_ids),
        reducer._last_order_key,
    )
    with pytest.raises(ValueError, match="noncanonical origin"):
        reducer.consume_available()
    assert (
        reducer.cursor,
        reducer.states,
        reducer._current_fact_event_ids,
        reducer._last_order_key,
    ) == before

    foundation_store = EventStore()
    foundation_reducer = _reducer(foundation_store)
    foundation = foundation_record_projection_event(
        _foundation_record_history()[0],
        timeframe=Timeframe.M5,
    )
    foundation_store.append(foundation)
    foundation_reducer.consume_available()
    assert foundation_reducer.states == {}
    assert foundation_reducer.current_facts() == CurrentMarketFacts()


def test_ten_thousand_current_revisions_keep_one_bounded_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = tuple(
        _canonical_event(
            f"range-revision-{index:05d}",
            index,
            canonical=True,
            kind=EventKind.DEALING_RANGE_CREATED,
            zone=(90.0, 110.0),
            source_entity_ids=("bounded-range",),
            details={"range_id": "bounded-range"},
        )
        for index in range(10_000)
    )
    store = EventStore.from_events(events)
    reducer = _reducer(store)
    original_events_since = store.events_since
    suffix_calls = 0

    def counted_suffix(index: int):
        nonlocal suffix_calls
        suffix_calls += 1
        return original_events_since(index)

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("hot current-fact publication scanned history")

    monkeypatch.setattr(store, "events_since", counted_suffix)
    monkeypatch.setattr(store, "events", forbidden_scan)
    monkeypatch.setattr(store, "prefix_fingerprint", forbidden_scan)
    reducer.consume_available()
    assert suffix_calls == 1
    assert len(reducer._current_fact_event_ids) == 1
    assert reducer.current_facts().ranges == (events[-1],)
    assert len(reducer.states) == 1
    hot_state = pickle.dumps(
        (
            reducer.states,
            reducer._current_fact_event_ids,
            reducer.cursor,
            reducer._store_prefix_fingerprint,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    assert len(hot_state) < 4_096

    monkeypatch.undo()
    resumed = pickle.loads(pickle.dumps(reducer))
    assert resumed.states == reducer.states
    assert resumed.current_facts() == reducer.current_facts()


def test_bos_resolution_is_bounded_structure_fact_and_cold_replay_exact() -> None:
    prefix, first_raw, first_qualified, direction, _ = (
        _phase23_protected_prefix()
    )
    acceptance_chain = _bos_terminal_chain(
        prefix="first-bos-resolution",
        raw=first_raw,
        kind=EventKind.ACCEPTANCE_CONFIRMED,
        candidate_minutes=8,
        strength=0.61,
    )
    accepted = acceptance_chain[-1]
    store = EventStore.from_events((*prefix, *acceptance_chain))
    reducer = _reducer(store)
    reducer.consume_available()

    facts = reducer.current_facts()
    assert {
        first_raw.event_id,
        first_qualified.event_id,
        accepted.event_id,
    }.issubset({event.event_id for event in facts.structure})
    assert facts.liquidity == ()
    assert accepted.direction is Direction.LONG
    assert accepted.strength == 0.61
    assert reducer._current_fact_event_ids[
        ("structure", "bos_resolution", "5m", "continuation:long")
    ] == accepted.event_id

    heartbeat = _event(
        EventKind.ENTRY_PATH_STEP,
        11,
        Timeframe.M1,
        event_id="unrelated-next-clock",
    )
    store.append(heartbeat)
    reducer.consume_available()
    assert accepted in reducer.current_facts().structure

    cold_store = EventStore.from_events(store.events_since(0))
    cold = _reducer(cold_store)
    cold.consume_available()
    assert cold.states == reducer.states
    assert cold.current_facts() == reducer.current_facts()
    resumed = pickle.loads(pickle.dumps(reducer))
    assert resumed.current_facts() == reducer.current_facts()

    next_bar, next_raw, next_qualified = _next_continuation_bos(
        minutes=12,
        bos_id="bos-next",
        direction_parent=direction,
    )
    store.append_batch((next_bar, next_raw))
    reducer.consume_available()
    ids = {event.event_id for event in reducer.current_facts().structure}
    # A raw attempt may never qualify, so it cannot revoke the last accepted
    # continuation incumbent.  The current raw-attempt slot is independent.
    assert accepted.event_id in ids
    assert first_raw.event_id in ids
    assert first_qualified.event_id in ids
    assert next_raw.event_id in ids

    store.append(next_qualified)
    reducer.consume_available()
    ids = {event.event_id for event in reducer.current_facts().structure}
    assert accepted.event_id not in ids
    assert first_raw.event_id not in ids
    assert first_qualified.event_id not in ids
    assert next_raw.event_id in ids
    assert next_qualified.event_id in ids
    sweep_chain = _bos_terminal_chain(
        prefix="next-bos-resolution",
        raw=next_raw,
        kind=EventKind.SWEEP_CONFIRMED,
        candidate_minutes=13,
    )
    swept = sweep_chain[-1]
    store.append_batch(sweep_chain)
    reducer.consume_available()
    assert swept.direction is Direction.SHORT
    assert reducer._current_fact_event_ids[
        ("structure", "bos_resolution", "5m", "continuation:long")
    ] == swept.event_id

    reset = _canonical_event(
        "epoch-reset",
        16,
        canonical=True,
        kind=EventKind.MARKET_EPOCH_RESET,
        timeframe=Timeframe.M1,
        side=None,
        price=None,
        details={"reason": "semantic_reset"},
    )
    store.append(reset)
    reducer.consume_available()
    assert reducer.current_facts().structure == ()


def test_external_owner_and_mss_challenger_context_never_become_dual_authority(
) -> None:
    chain = _authoritative_phase23_chain()
    mss_index = next(
        index
        for index, event in enumerate(chain)
        if event.kind is EventKind.MSS_CORE_CONFIRMED
    )
    prefix = chain[: mss_index + 1]
    store = EventStore.from_events(prefix)
    reducer = _reducer(store)
    reducer.consume_available()
    facts = reducer.current_facts()
    external = tuple(
        event
        for event in facts.structure
        if event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED
    )
    assert tuple(event.event_id for event in external) == (
        "structure-direction",
    )
    assert tuple(event.event_id for event in facts.structure_context) == (
        "structure-direction-opposed",
    )
    assert any(event.event_id == "mss-core" for event in facts.structure)
    assert reducer.states[Timeframe.M5].structure.external_direction is (
        Direction.LONG
    )
    assert reducer.states[Timeframe.M5].structure.internal_direction is (
        Direction.LONG
    )

    replacement_context = _canonical_event(
        "structure-direction-opposed-next",
        10,
        canonical=True,
        kind=EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        timeframe=Timeframe.M5,
        direction=Direction.SHORT,
        side="below",
        price=105.0,
        source_event_ids=("swing-right", "swing-left"),
        source_entity_ids=("structure-c", "swing-b", "swing-a"),
        details={
            "structure_id": "structure-c",
            "direction": "short",
            "source_high_id": "swing-b",
            "source_low_id": "swing-a",
            "sequence_count": 3,
            "candidate_protected_swing_id": "swing-b",
        },
    )
    store.append(replacement_context)
    reducer.consume_available()
    after = reducer.current_facts()
    assert tuple(event.event_id for event in after.structure_context) == (
        replacement_context.event_id,
    )
    assert all(event.kind is not EventKind.MSS_CORE_CONFIRMED for event in after.structure)
    assert tuple(
        event.event_id
        for event in after.structure
        if event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED
    ) == ("structure-direction",)

    cold_store = EventStore.from_events(store.events_since(0))
    cold = _reducer(cold_store)
    cold.consume_available()
    assert cold.states == reducer.states
    assert cold.current_facts() == reducer.current_facts()


def test_exact_protected_acceptance_clears_external_and_context_closure() -> None:
    prefix, _, _, _, protected = _phase23_protected_prefix()
    release = _crossing_chain(
        prefix="exact-protected-release",
        side="below",
        level_id="swing:swing-a",
        candidate_minutes=8,
        crossed_minutes=9,
        resolved_minutes=10,
    )
    terminal = replace(
        release[-1],
        details={
            **dict(release[-1].evidence),
            "protected_swing_id": "swing-a",
            "protected_swing_event_id": protected.event_id,
        },
        evidence={
            **dict(release[-1].evidence),
            "protected_swing_id": "swing-a",
            "protected_swing_event_id": protected.event_id,
        },
        context_event_ids=(protected.event_id,),
    )
    store = EventStore.from_events((*prefix, *release[:-1], terminal))
    reducer = _reducer(store)
    reducer.consume_available()
    state = reducer.states[Timeframe.M5].structure
    assert state.external_direction is None
    assert state.protected_swing_intact is False
    facts = reducer.current_facts()
    assert facts.structure_context == ()
    assert all(
        event.kind
        not in {
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            EventKind.QUALIFIED_BOS,
            EventKind.PROTECTED_SWING_ASSIGNED,
            EventKind.MSS_CORE_CONFIRMED,
        }
        for event in facts.structure
    )


def test_current_fact_categories_and_terminal_eviction_are_exact() -> None:
    structure_prefix, *_ = _phase23_protected_prefix()
    structure_store = EventStore.from_events(structure_prefix)
    structure_reducer = _reducer(structure_store)
    structure_reducer.consume_available()
    assert any(
        event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED
        for event in structure_reducer.current_facts().structure
    )

    crossing = _crossing_chain(
        prefix="current-level",
        candidate_minutes=0,
        crossed_minutes=2,
        resolved_minutes=3,
    )
    liquidity_store = EventStore.from_events(crossing[:-1])
    liquidity_reducer = _reducer(liquidity_store)
    liquidity_reducer.consume_available()
    assert tuple(
        event.kind for event in liquidity_reducer.current_facts().liquidity
    ) == (EventKind.LEVEL_PENETRATED,)
    liquidity_store.append(crossing[-1])
    liquidity_reducer.consume_available()
    assert liquidity_reducer.current_facts().liquidity == ()

    phase_chain = _authoritative_phase23_chain()
    displacement = next(
        event
        for event in phase_chain
        if event.kind is EventKind.DISPLACEMENT_OBSERVED
    )
    displacement_parents = tuple(
        next(event for event in phase_chain if event.event_id == event_id)
        for event_id in displacement.source_event_ids
    )
    displacement_store = EventStore.from_events(
        (*displacement_parents, displacement)
    )
    displacement_reducer = _reducer(displacement_store)
    displacement_reducer.consume_available()
    assert displacement_reducer.current_facts().displacement == (
        displacement,
    )

    fvg_bars = (
        _canonical_bar(
            "current-fvg-left",
            0,
            open_=99.0,
            high=100.0,
            low=98.0,
            close=99.0,
        ),
        _canonical_bar(
            "current-fvg-middle",
            1,
            open_=100.0,
            high=103.0,
            low=99.0,
            close=102.0,
        ),
        _canonical_bar(
            "current-fvg-right",
            2,
            open_=102.0,
            high=104.0,
            low=102.0,
            close=103.0,
        ),
    )
    fvg = _canonical_event(
        "current-fvg",
        2,
        canonical=True,
        kind=EventKind.FVG_CREATED,
        direction=Direction.LONG,
        side="below",
        price=101.0,
        zone=(100.0, 102.0),
        sequence_no=1,
        source_event_ids=tuple(event.event_id for event in fvg_bars),
        source_entity_ids=("current-fvg",),
        details={"fvg_id": "current-fvg"},
    )
    zone_store = EventStore.from_events((*fvg_bars, fvg))
    zone_reducer = _reducer(zone_store)
    zone_reducer.consume_available()
    assert zone_reducer.current_facts().zones == (fvg,)

    range_chain = _external_range_chain()
    range_store = EventStore.from_events(range_chain[:2])
    range_reducer = _reducer(range_store)
    range_reducer.consume_available()
    assert range_reducer.current_facts().ranges[-1].kind is (
        EventKind.DEALING_RANGE_ACTIVATED
    )
    range_store.append_batch(range_chain[2:])
    range_reducer.consume_available()
    assert range_reducer.current_facts().ranges == ()

    for store, reducer in (
        (structure_store, structure_reducer),
        (liquidity_store, liquidity_reducer),
        (displacement_store, displacement_reducer),
        (zone_store, zone_reducer),
        (range_store, range_reducer),
    ):
        cold_store = EventStore.from_events(store.events_since(0))
        cold = _reducer(cold_store)
        cold.consume_available()
        assert cold.states == reducer.states
        assert cold.current_facts() == reducer.current_facts()


def test_bos_resolution_missing_raw_context_is_failure_atomic() -> None:
    prefix, raw, *_ = _phase23_protected_prefix()
    crossing = list(
        _bos_terminal_chain(
            prefix="malformed-bos-resolution",
            raw=raw,
            kind=EventKind.ACCEPTANCE_CONFIRMED,
            candidate_minutes=8,
        )
    )
    crossing[-1] = replace(crossing[-1], context_event_ids=())
    store = EventStore.from_events((*prefix, *crossing[:-1]))
    reducer = _reducer(store)
    reducer.consume_available()
    before = pickle.dumps(
        (
            reducer.states,
            reducer.cursor,
            reducer._last_order_key,
            reducer._current_fact_event_ids,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    malformed = crossing[-1]
    store.append(malformed)

    with pytest.raises(ValueError, match="exact raw-break context"):
        reducer.consume_available()
    after = pickle.dumps(
        (
            reducer.states,
            reducer.cursor,
            reducer._last_order_key,
            reducer._current_fact_event_ids,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    assert after == before
    assert len(store) == len(prefix) + len(crossing)
    with pytest.raises(ValueError, match="exact raw-break context"):
        reducer.consume_available()


def test_ordinary_liquidity_terminal_does_not_create_structure_fact() -> None:
    crossing = _crossing_chain(
        prefix="ordinary-sweep",
        kind=EventKind.SWEEP_CONFIRMED,
        candidate_minutes=0,
        crossed_minutes=2,
        resolved_minutes=3,
    )
    store = EventStore.from_events(crossing)
    reducer = _reducer(store)
    reducer.consume_available()
    assert reducer.current_facts().structure == ()
    assert reducer.current_facts().liquidity == ()


def test_snapshot_carries_bounded_bos_support_across_later_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, raw, *_ = _phase23_protected_prefix()
    acceptance_chain = _bos_terminal_chain(
        prefix="dfp-bos-resolution",
        raw=raw,
        kind=EventKind.ACCEPTANCE_CONFIRMED,
        candidate_minutes=8,
        strength=0.67,
    )
    accepted = acceptance_chain[-1]
    store = EventStore.from_events((*prefix, *acceptance_chain))
    publisher = MarketSnapshotPublisher(
        event_store=store,
        semantic_registry_identity="definition-test",
    )
    first = publisher.publish(
        asof=_clock(10),
        symbol="NQH5",
        instrument_id=1,
        price=105.0,
        completed_1m=_m1_candle(10),
        frames=_frames(_clock(10)),
        inventory=(),
        displacement=None,
        anomalies=(),
        emit_projection_events=False,
    )[0]
    assert accepted in first.current_facts.structure

    store.append(
        _event(
            EventKind.ENTRY_PATH_STEP,
            11,
            Timeframe.M1,
            event_id="dfp-next-clock",
        )
    )

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("hot snapshot publication scanned store history")

    monkeypatch.setattr(store, "events", forbidden_scan)
    monkeypatch.setattr(store, "prefix_fingerprint", forbidden_scan)
    later = publisher.publish(
        asof=_clock(11),
        symbol="NQH5",
        instrument_id=1,
        price=105.0,
        completed_1m=_m1_candle(11),
        frames=_frames(_clock(11)),
        inventory=(),
        displacement=None,
        anomalies=(),
        emit_projection_events=False,
    )[0]
    support = tuple(
        event
        for event in later.current_facts.structure
        if event.kind is EventKind.ACCEPTANCE_CONFIRMED
        and event.evidence.get("bos_id") == "bos-a"
    )
    assert len(support) == 1
    assert support[0].strength == 0.67
    assert accepted not in later.events_this_update
    assert later.event_count == len(store)
    assert later.event_prefix_fingerprint == store.fingerprint()

    roundtrip = pickle.loads(pickle.dumps(later))
    assert roundtrip == later
    assert roundtrip.fingerprint == later.fingerprint
    assert roundtrip.replay_payload() == later.replay_payload()
    primitive = to_primitive(later)
    assert set(primitive["current_facts"]) == {
        "structure",
        "structure_context",
        "liquidity",
        "displacement",
        "zones",
        "ranges",
    }
    assert MARKET_SNAPSHOT_SCHEMA_VERSION == 3
    mismatched_support = replace(
        support[0],
        semantic_version="smc_semantics_v0.invalid",
    )
    mismatched_facts = replace(
        later.current_facts,
        structure=tuple(
            mismatched_support if event is support[0] else event
            for event in later.current_facts.structure
        ),
    )
    with pytest.raises(ValueError, match="identity or causal clock"):
        replace(later, current_facts=mismatched_facts)
    old = dict(later.__getstate__())
    old["schema_version"] = 2
    with pytest.raises(ValueError, match="pickle schema changed"):
        MarketSnapshot.__setstate__(object.__new__(MarketSnapshot), old)


def test_projection_tail_advances_but_physical_tail_fails_atomically() -> None:
    store = EventStore()
    publisher = MarketSnapshotPublisher(
        event_store=store,
        semantic_registry_identity="definition-test",
    )
    projection = _projection_event(
        EventKind.TIMEFRAME_STATE_CHANGED,
        asof=_clock(0),
        timeframe=Timeframe.M1,
        state_id="projection-tail",
        state={"value": 1},
        source_event_ids=(),
    )
    store.append(projection)
    restored = pickle.loads(
        pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert restored._event_reducer.cursor == 0
    assert restored._consume_committed_projection_tail() == (projection,)
    assert publisher._consume_committed_projection_tail() == (projection,)
    assert publisher._event_reducer.cursor == 1

    physical = _event(
        EventKind.ENTRY_PATH_STEP,
        1,
        Timeframe.M1,
        event_id="physical-tail",
    )
    store.append(physical)
    before = (
        publisher._event_reducer.cursor,
        dict(publisher._event_reducer.states),
        dict(publisher._event_reducer._current_fact_event_ids),
    )
    with pytest.raises(ValueError, match="contains a physical fact"):
        publisher._consume_committed_projection_tail()
    with pytest.raises(ValueError, match="checkpoint has a physical suffix"):
        pickle.loads(pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL))
    assert (
        publisher._event_reducer.cursor,
        publisher._event_reducer.states,
        publisher._event_reducer._current_fact_event_ids,
    ) == before

    foundation_store = EventStore()
    foundation_publisher = MarketSnapshotPublisher(
        event_store=foundation_store,
        semantic_registry_identity="definition-test",
    )
    foundation = foundation_record_projection_event(
        _foundation_record_history()[0],
        timeframe=Timeframe.M5,
    )
    foundation_store.append(foundation)
    with pytest.raises(ValueError, match="contains a physical fact"):
        foundation_publisher._consume_committed_projection_tail()
    with pytest.raises(ValueError, match="checkpoint has a physical suffix"):
        pickle.loads(
            pickle.dumps(
                foundation_publisher,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )


def test_boundary_interaction_terminal_is_ledgered_and_published_as_delta() -> None:
    boundary = _held_boundary_interaction()
    boundary_store = EventStore()
    observer = object.__new__(CausalObserver)
    observer.memory = EventMemory(32, audit_store=boundary_store)
    observer._record_interaction_events(boundary)
    observer.memory.flush_audit()
    terminal = tuple(
        event
        for event in boundary_store.events_since(0)
        if event.kind is EventKind.ENTRY_PATH_STATE
    )
    assert terminal
    assert all(
        event.lifecycle == PathSequenceLifecycle.CENSORED.value
        and event.ended_at is not None
        for event in terminal
    )

    boundary_asof = _m1(4).end
    boundary_publisher = MarketSnapshotPublisher(
        event_store=boundary_store,
        semantic_registry_identity="definition-test",
    )
    boundary_snapshot = boundary_publisher.publish(
        asof=boundary_asof,
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1(4),
        frames=_frames(boundary_asof),
        inventory=(),
        displacement=None,
        anomalies=("data_gap_reset",),
        emit_projection_events=False,
    )[0]
    assert all(event in boundary_snapshot.events_this_update for event in terminal)
