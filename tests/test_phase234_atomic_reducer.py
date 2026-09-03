from __future__ import annotations

from dataclasses import replace
import copy
import math
import pickle

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader, ReaderUpdate
from smc_trader.event_store import EventStore, event_order_key
from smc_trader.market_state import (
    MARKET_SNAPSHOT_SCHEMA_VERSION,
    TIMEFRAME_EVENT_REDUCER_SCHEMA_VERSION,
    DeliveryPhase,
    MarketSnapshot,
    MarketSnapshotPublisher,
    MarketSnapshotAuthority,
    RelationRole,
    RelationResolver,
    SessionStateReducer,
    TimeframeEventReducer,
    replay_atomic_market_snapshot,
    reduce_timeframe_state,
)
from smc_trader.model import (
    Candle,
    Bar,
    Direction,
    EventKind,
    EventOrigin,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    MarketEvent,
    SMC_SEMANTIC_VERSION,
    Timeframe,
    to_primitive,
)
from smc_trader.observation import CausalObserver, ObserverConfig

from .helpers import (
    CORE_TEST_SCALE_REGISTRY_ID,
    CORE_TEST_SCALE_SPECS,
)
from .test_event_provenance_contract import (
    _authoritative_phase23_chain,
    _normalized_bar as _canonical_normalized_bar,
)
from .test_semantic_foundation_geometry import _balance_range


TZ = "America/New_York"
PROJECTION_KINDS = {
    EventKind.TIMEFRAME_STATE_CHANGED,
    EventKind.RELATION_STATE_CHANGED,
    EventKind.SESSION_STATE_CHANGED,
    EventKind.FOUNDATION_STATE_CHANGED,
}


def _snapshot_publisher(**kwargs: object) -> MarketSnapshotPublisher:
    return MarketSnapshotPublisher(
        event_store=EventStore(),
        **kwargs,
    )


class _StoreDrivenReducer(TimeframeEventReducer):
    """Test adapter spelling out append -> consume for legacy assertions."""

    def apply(self, event: MarketEvent):
        self.event_store.append(event)
        self.consume_available()
        if (
            event.origin is EventOrigin.STATE_PROJECTION
            or event.kind is EventKind.FOUNDATION_STATE_CHANGED
        ):
            return None
        return self.states.get(event.timeframe)

    def replay(self, events: tuple[MarketEvent, ...]):
        self.event_store.append_batch(events)
        self.consume_available()
        return self.states


def _timeframe_reducer(**kwargs: object) -> _StoreDrivenReducer:
    return _StoreDrivenReducer(event_store=EventStore(), **kwargs)


def _publish(
    publisher: MarketSnapshotPublisher,
    *,
    semantic_events: tuple[MarketEvent, ...] = (),
    **kwargs: object,
) -> tuple[MarketSnapshot, tuple[MarketEvent, ...]]:
    publisher.event_store.append_batch(semantic_events)
    snapshot, projection_events, _ = publisher.publish(**kwargs)
    return snapshot, projection_events


def _reducer_hot_state(reducer: TimeframeEventReducer) -> bytes:
    return pickle.dumps(
        (
            reducer.states,
            reducer.cursor,
            reducer._epoch_cursor,
            reducer._last_order_key,
            reducer._latest_real_m1_event_id,
            reducer.expected_timeframes,
            reducer._store_prefix_fingerprint,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def test_timeframe_reducer_owns_only_store_cursor_and_compact_market_state() -> None:
    chain = _authoritative_phase23_chain()
    raw_index = next(
        index
        for index, event in enumerate(chain)
        if event.kind is EventKind.RAW_BOUNDARY_BREAK
    )
    prefix = chain[: raw_index + 1]
    raw_break = prefix[-1]
    store = EventStore.from_events(prefix)
    reducer = TimeframeEventReducer(
        event_store=store,
        semantic_registry_identity="definition-test",
    )

    assert reducer.consume_available() == prefix
    assert reducer.event_store is store
    assert reducer.cursor == len(prefix)
    assert raw_break.kind is EventKind.RAW_BOUNDARY_BREAK
    assert "_current_fact_event_ids" not in reducer.__dict__

    checkpoint = reducer.__getstate__()
    assert TIMEFRAME_EVENT_REDUCER_SCHEMA_VERSION == 3
    assert "_current_fact_event_ids" not in checkpoint
    old_checkpoint = dict(checkpoint)
    old_checkpoint["_reducer_schema_version"] = 2
    old_checkpoint["_current_fact_event_ids"] = {}
    before = dict(reducer.__dict__)
    with pytest.raises(ValueError, match="checkpoint schema changed"):
        reducer.__setstate__(old_checkpoint)
    assert reducer.__dict__ == before

    resumed = pickle.loads(pickle.dumps(reducer))
    cold_store = EventStore.from_events(prefix)
    cold = TimeframeEventReducer(
        event_store=cold_store,
        semantic_registry_identity="definition-test",
    )
    cold.consume_available()
    assert resumed.cursor == cold.cursor == reducer.cursor
    assert resumed.states == cold.states == reducer.states

    reset = _event(
        EventKind.MARKET_EPOCH_RESET,
        100,
        Timeframe.M1,
        event_id="market-state-reset",
        price=None,
        evidence={"reason": "test_boundary"},
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    store.append(reset)
    assert reducer.consume_available() == (reset,)
    assert reducer.states == {}

    forged = _event(
        EventKind.DEALING_RANGE_CREATED,
        101,
        Timeframe.M1,
        event_id="forged-legacy-range",
        evidence={"range_id": "forged"},
        zone=(90.0, 110.0),
    )
    store.append(forged)
    with pytest.raises(ValueError, match="noncanonical origin"):
        reducer.consume_available()
    assert reducer.cursor == len(prefix) + 1
    assert reducer.states == {}


def test_timeframe_reducer_10k_suffix_keeps_hot_state_history_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EventStore()
    reducer = TimeframeEventReducer(
        event_store=store,
        semantic_registry_identity="definition-test",
    )
    events = tuple(
        _event(
            EventKind.ENTRY_PATH_STEP,
            0,
            Timeframe.M1,
            event_id=f"legacy-delta:{index:05d}",
            sequence_no=index,
            evidence={"step_id": f"step:{index:05d}"},
        )
        for index in range(10_000)
    )
    store.append_batch(events)
    original_events_since = store.events_since
    suffix_calls = 0

    def counted_suffix(index: int):
        nonlocal suffix_calls
        suffix_calls += 1
        return original_events_since(index)

    def forbidden_history_scan(*args, **kwargs):
        raise AssertionError("hot reducer scanned EventStore history")

    monkeypatch.setattr(store, "events_since", counted_suffix)
    monkeypatch.setattr(store, "events", forbidden_history_scan)
    monkeypatch.setattr(store, "prefix_fingerprint", forbidden_history_scan)

    assert reducer.consume_available() == events
    assert suffix_calls == 1
    assert reducer.cursor == 10_000
    assert reducer.states == {}
    hot_state = pickle.dumps(
        (
            reducer.states,
            reducer.cursor,
            reducer._epoch_cursor,
            reducer._last_order_key,
            reducer._latest_real_m1_event_id,
            reducer.expected_timeframes,
            reducer._store_prefix_fingerprint,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    assert len(hot_state) < 1_024


def _publisher_hot_state(publisher: MarketSnapshotPublisher) -> bytes:
    return pickle.dumps(
        (
            _reducer_hot_state(publisher._event_reducer),
            publisher._session,
            publisher._last_projection_payloads,
            publisher._last_projection_event_ids,
            publisher._formal_structures,
            publisher._last_real_m1_price,
            publisher._last_real_m1_event_id,
            publisher._boundary_reset_pending,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2025-01-06 10:00", tz=TZ) + pd.Timedelta(
        minutes * 60,
        unit="s",
    )


def _event(
    kind: EventKind,
    minutes: int,
    timeframe: Timeframe,
    *,
    event_id: str | None = None,
    sequence_no: int = 0,
    side: str | None = None,
    price: float | None = 100.0,
    direction: Direction | None = None,
    evidence: dict[str, object] | None = None,
    source_ids: tuple[str, ...] = (),
    context_event_ids: tuple[str, ...] = (),
    zone: tuple[float, float] | None = None,
    semantic_version: str = SMC_SEMANTIC_VERSION,
    origin: EventOrigin = EventOrigin.LEGACY_TRANSPORT,
) -> MarketEvent:
    known_at = _clock(minutes)
    identity = event_id or (
        f"{minutes:04d}:{sequence_no:04d}:{timeframe.value}:{kind.value}"
    )
    return MarketEvent(
        event_id=identity,
        kind=kind,
        observed_at=known_at,
        timeframe=timeframe,
        side=side,
        price=price,
        strength=0.75,
        source_ids=source_ids,
        details={} if evidence is None else evidence,
        direction=direction,
        sequence_no=sequence_no,
        event_time=known_at,
        known_at=known_at,
        semantic_version=semantic_version,
        evidence={} if evidence is None else evidence,
        zone=zone,
        context_event_ids=context_event_ids,
        origin=origin,
    )


def _bar_event(
    minutes: int,
    timeframe: Timeframe,
    *,
    close: float,
    atr: float = 2.0,
    sequence_no: int = 0,
    event_id: str | None = None,
    semantic_version: str = SMC_SEMANTIC_VERSION,
    real_completed: bool = True,
) -> MarketEvent:
    return _event(
        EventKind.BAR_COMPLETED,
        minutes,
        timeframe,
        event_id=event_id,
        sequence_no=sequence_no,
        price=close,
        evidence={
            "close": close,
            "atr": atr,
            "data_complete": True,
            "real_completed": real_completed,
            "clock_only": not real_completed,
            "event_category": "normalized_data",
            "source_data_ids": (f"bar:{timeframe.value}:{minutes}",),
        },
        semantic_version=semantic_version,
    )


def _canonical_bar_event(
    minutes: int,
    timeframe: Timeframe,
    *,
    close: float,
    atr: float = 2.0,
    sequence_no: int = 0,
    event_id: str | None = None,
    semantic_version: str = SMC_SEMANTIC_VERSION,
    real_completed: bool = True,
) -> MarketEvent:
    """Build a registry-admissible normalized BAR for store-driven tests."""

    event = _canonical_normalized_bar(
        event_id or f"canonical:{timeframe.value}:{minutes}:{sequence_no}",
        minutes,
        timeframe=timeframe,
        close=close,
        atr=atr,
        real_completed=real_completed,
        instrument_id=1,
        sequence_no=sequence_no,
    )
    evidence = {**dict(event.evidence), "scale_registry_id": "test-scale-registry"}
    return replace(
        event,
        semantic_version=semantic_version,
        details=evidence,
        evidence=evidence,
    )


def _atomic_reset_event(
    minutes: int,
    *,
    event_id: str | None = None,
    sequence_no: int = 0,
    reason: str = "contract_change_reset",
) -> MarketEvent:
    return _event(
        EventKind.MARKET_EPOCH_RESET,
        minutes,
        Timeframe.M1,
        event_id=event_id,
        sequence_no=sequence_no,
        price=None,
        evidence={"reason": reason},
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )


def _reduce(events: tuple[MarketEvent, ...]):
    state = None
    for event in events:
        state = reduce_timeframe_state(
            state,
            event,
            semantic_registry_identity="definition-test",
        )
    assert state is not None
    return state


def test_native_htf_clock_only_bar_is_provenance_only_for_timeframe_state() -> None:
    initial = _reduce(
        (_bar_event(0, Timeframe.H1, close=100.0, atr=2.0),)
    )
    clock_only = _bar_event(
        1,
        Timeframe.H1,
        close=125.0,
        atr=25.0,
        real_completed=False,
    )

    advanced = reduce_timeframe_state(
        initial,
        clock_only,
        semantic_registry_identity="definition-test",
    )

    assert advanced == initial


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("foreign_timeframe", "foreign event"),
        ("wrong_semantic_version", "mix semantic versions"),
        ("out_of_order", "out-of-order knowledge"),
    ),
)
def test_native_htf_clock_only_noop_still_enforces_direct_reducer_contract(
    case: str,
    message: str,
) -> None:
    initial = _reduce(
        (_bar_event(0, Timeframe.H1, close=100.0, atr=2.0),)
    )
    if case == "foreign_timeframe":
        invalid = _bar_event(
            1,
            Timeframe.M5,
            close=100.0,
            atr=2.0,
            real_completed=False,
        )
    elif case == "wrong_semantic_version":
        invalid = _bar_event(
            1,
            Timeframe.H1,
            close=100.0,
            atr=2.0,
            real_completed=False,
            semantic_version="forged-semantic-version",
        )
    elif case == "out_of_order":
        invalid = _bar_event(
            -1,
            Timeframe.H1,
            close=100.0,
            atr=2.0,
            real_completed=False,
        )
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)

    with pytest.raises(ValueError, match=message):
        reduce_timeframe_state(
            initial,
            invalid,
            semantic_registry_identity="definition-test",
        )


def test_m1_clock_only_bar_advances_only_quality_heartbeat() -> None:
    initial = _reduce(
        (_bar_event(0, Timeframe.M1, close=100.0, atr=2.0),)
    )
    clock_only = _bar_event(
        1,
        Timeframe.M1,
        close=125.0,
        atr=25.0,
        real_completed=False,
    )

    advanced = reduce_timeframe_state(
        initial,
        clock_only,
        semantic_registry_identity="definition-test",
    )

    assert advanced is not None
    assert replace(advanced, quality=initial.quality) == initial
    assert advanced.quality.known_at == clock_only.known_at
    assert advanced.quality.source_cutoff == initial.quality.source_cutoff
    assert advanced.quality.atr == initial.quality.atr
    assert advanced.quality.data_complete == initial.quality.data_complete
    assert advanced.quality.last_event_id == clock_only.event_id
    assert advanced.quality.last_sequence_no == clock_only.sequence_no
    assert advanced.quality.events_applied == initial.quality.events_applied + 1


@pytest.mark.parametrize(
    ("marker", "value", "real_completed"),
    (
        ("owner_price_update_only", True, True),
        ("owner_clock_heartbeat_only", True, False),
        ("price_source_timeframe", Timeframe.M1.value, False),
    ),
)
def test_direct_reducer_rejects_forged_owner_fanout_markers(
    marker: str,
    value: object,
    real_completed: bool,
) -> None:
    initial = _reduce(
        (_bar_event(0, Timeframe.H1, close=100.0, atr=2.0),)
    )
    event = _bar_event(
        1,
        Timeframe.H1,
        close=125.0,
        atr=25.0,
        real_completed=real_completed,
    )
    evidence = {**dict(event.evidence), marker: value}
    forged = replace(event, details=evidence, evidence=evidence)

    with pytest.raises(ValueError, match="private owner fanout markers"):
        reduce_timeframe_state(
            initial,
            forged,
            semantic_registry_identity="definition-test",
        )


def test_private_m1_owner_fanout_preserves_registered_owner_contract() -> None:
    initial = _reduce(
        (_bar_event(0, Timeframe.H1, close=100.0, atr=2.0),)
    )
    clock = _bar_event(
        1,
        Timeframe.H1,
        close=125.0,
        atr=25.0,
        real_completed=False,
    )
    heartbeat = reduce_timeframe_state(
        initial,
        clock,
        semantic_registry_identity="definition-test",
        _m1_owner_fanout=True,
    )
    assert heartbeat is not None
    assert replace(heartbeat, quality=initial.quality) == initial
    assert heartbeat.quality.known_at == clock.known_at

    real = _bar_event(2, Timeframe.H1, close=130.0, atr=50.0)
    updated = reduce_timeframe_state(
        heartbeat,
        real,
        semantic_registry_identity="definition-test",
        _m1_owner_fanout=True,
    )
    assert updated is not None
    assert updated.quality.source_cutoff == initial.quality.source_cutoff
    assert updated.quality.atr == initial.quality.atr


@pytest.mark.parametrize(
    "child_kind",
    (
        EventKind.FVG_CREATED,
        EventKind.ORIGIN_ZONE_CREATED,
        EventKind.DEALING_RANGE_INVALIDATED,
    ),
)
def test_authoritative_atomic_parent_rejects_clock_only_bar(
    child_kind: EventKind,
) -> None:
    clock_only = _replayable_m1_event(
        1,
        100.0,
        real_completed=False,
    )
    child = _event(
        child_kind,
        1,
        Timeframe.M1,
        event_id=f"clock-only-parent:{child_kind.value}",
        source_ids=(clock_only.event_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )

    with pytest.raises(ValueError, match="exact real normalized BAR"):
        EventStore._require_authoritative_parent_origins(
            child,
            (clock_only,),
        )


def test_event_store_rejects_clock_only_root_without_coverage_proof() -> None:
    clock_only = _replayable_m1_event(
        1,
        100.0,
        real_completed=False,
    )
    evidence = dict(clock_only.evidence)
    for name in (
        "complete",
        "start",
        "observed_minutes",
        "expected_minutes",
        "real_minutes",
        "synthetic_minutes",
    ):
        evidence.pop(name)
    forged = replace(clock_only, details=evidence, evidence=evidence)

    with pytest.raises(ValueError, match="requires coverage"):
        EventStore._normalized_bar_identity(
            forged,
            bar_event_ids={},
        )


def test_event_store_rejects_clock_only_root_with_forged_registered_bounds() -> None:
    clock_only = _replayable_m1_event(
        1,
        100.0,
        real_completed=False,
    )
    evidence = {
        **dict(clock_only.evidence),
        "start": clock_only.evidence["start"]
        - pd.Timedelta(1, unit="min"),
    }
    forged = replace(clock_only, details=evidence, evidence=evidence)

    with pytest.raises(ValueError, match="registered bounds"):
        EventStore._normalized_bar_identity(
            forged,
            bar_event_ids={},
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("clock_only_event_time_drift", "event_time and known_at"),
        ("partial_real_coverage", "all-or-none"),
        ("real_with_synthetic_coverage", "coverage is inconsistent"),
    ),
)
def test_event_store_bar_root_contract_rejects_partial_or_conflicting_proof(
    case: str,
    message: str,
) -> None:
    if case == "clock_only_event_time_drift":
        invalid = replace(
            _replayable_m1_event(1, 100.0, real_completed=False),
            event_time=_clock(0),
        )
    else:
        event = _replayable_m1_event(1, 100.0)
        evidence = {
            **dict(event.evidence),
            "complete": True,
            "start": _clock(0),
        }
        if case == "real_with_synthetic_coverage":
            evidence.update(
                observed_minutes=1,
                expected_minutes=1,
                real_minutes=0,
                synthetic_minutes=1,
            )
        elif case != "partial_real_coverage":  # pragma: no cover
            raise AssertionError(case)
        invalid = replace(event, details=evidence, evidence=evidence)

    with pytest.raises(ValueError, match=message):
        EventStore._normalized_bar_identity(
            invalid,
            bar_event_ids={},
        )


@pytest.mark.parametrize(
    ("marker", "value"),
    (
        ("owner_price_update_only", True),
        ("owner_clock_heartbeat_only", True),
        ("price_source_timeframe", Timeframe.M1.value),
    ),
)
def test_event_store_rejects_private_owner_fanout_markers(
    marker: str,
    value: object,
) -> None:
    event = _replayable_m1_event(1, 100.0)
    evidence = {**dict(event.evidence), marker: value}
    forged = replace(event, details=evidence, evidence=evidence)

    with pytest.raises(ValueError, match="private owner fanout markers"):
        EventStore._normalized_bar_identity(
            forged,
            bar_event_ids={},
        )


def _relation_events() -> tuple[MarketEvent, ...]:
    """Minimal canonical roots for projection/relation parity tests.

    Structural semantics have their own complete-parent-chain tests. These
    tests exercise store/reducer/publisher plumbing and therefore use only
    normalized roots instead of the retired legacy shorthand facts.
    """

    return (
        _canonical_bar_event(0, Timeframe.H1, close=100.0),
        _canonical_bar_event(5, Timeframe.M5, close=105.0, atr=1.0),
        _replayable_m1_event(
            10,
            105.0,
            active_timeframes=(
                Timeframe.H1,
                Timeframe.M5,
                Timeframe.M1,
            ),
        ),
    )


def test_relation_resolver_preserves_parent_retracement_state_contract() -> None:
    """The fixture migration must not weaken physical relation semantics."""

    parent = _reduce(
        (
            _bar_event(0, Timeframe.H1, close=100.0),
            _event(
                EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                1,
                Timeframe.H1,
                direction=Direction.LONG,
                evidence={"structure_id": "h1-structure"},
            ),
            _event(
                EventKind.QUALIFIED_BOS,
                2,
                Timeframe.H1,
                price=108.0,
                side="above",
                direction=Direction.LONG,
                evidence={"bos_id": "h1-bos"},
            ),
            _event(
                EventKind.PROTECTED_SWING_ASSIGNED,
                3,
                Timeframe.H1,
                price=95.0,
                side="below",
                direction=Direction.LONG,
                evidence={"protected_swing_id": "h1-protected-low"},
            ),
            _event(
                EventKind.LIQUIDITY_LEVEL_CREATED,
                4,
                Timeframe.H1,
                price=110.0,
                side="above",
                evidence={
                    "level_id": "h1-bsl",
                    "source_kind": "confirmed_swing",
                },
            ),
        )
    )
    child = _reduce(
        (
            _bar_event(5, Timeframe.M5, close=105.0, atr=1.0),
            _event(
                EventKind.MSS_CORE_CONFIRMED,
                6,
                Timeframe.M5,
                price=103.0,
                side="below",
                direction=Direction.SHORT,
                evidence={"bos_id": "m5-opposed-break"},
            ),
        )
    )

    relation = RelationResolver(
        edges=((Timeframe.H1, Timeframe.M5),)
    ).resolve(
        {Timeframe.H1: parent, Timeframe.M5: child},
        price=105.0,
        asof=_clock(10),
    )["1H__5m"]

    assert relation.role is RelationRole.PARENT_RETRACEMENT
    assert relation.parent_direction is Direction.LONG
    assert relation.child_direction is Direction.SHORT
    assert relation.parent_state_invalidated is False
    assert relation.child_mss_against_parent is True


def _m1_candle(
    end_minutes: int,
    price: float = 105.0,
    *,
    symbol: str = "NQH5",
    instrument_id: int = 1,
    real_completed: bool = True,
) -> Candle:
    end = _clock(end_minutes)
    return Candle(
        timeframe=Timeframe.M1,
        start=end - pd.Timedelta(60, unit="s"),
        end=end,
        open=price - 0.25,
        high=price + 0.25,
        low=price - 0.5,
        close=price,
        volume=100.0,
        symbol=symbol,
        instrument_id=instrument_id,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=1 if real_completed else 0,
        synthetic_minutes=0 if real_completed else 1,
    )


def _replayable_m1_event(
    minutes: int,
    price: float,
    *,
    active_timeframes: tuple[Timeframe, ...] | None = None,
    scale_registry_id: str = "test-scale-registry",
    symbol: str = "NQH5",
    instrument_id: int = 1,
    real_completed: bool = True,
) -> MarketEvent:
    candle = _m1_candle(
        minutes,
        price,
        symbol=symbol,
        instrument_id=instrument_id,
        real_completed=real_completed,
    )
    return _event(
        EventKind.BAR_COMPLETED,
        minutes,
        Timeframe.M1,
        price=price,
        evidence={
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "atr": 1.0,
            "data_complete": True,
            "real_completed": real_completed,
            "clock_only": not real_completed,
            "symbol": candle.symbol,
            "instrument_id": candle.instrument_id,
            "scale_registry_id": scale_registry_id,
            "source_data_ids": (f"m1:{minutes}",),
            **(
                {}
                if real_completed
                else {
                    "complete": True,
                    "start": candle.start,
                    "observed_minutes": candle.observed_minutes,
                    "expected_minutes": candle.expected_minutes,
                    "real_minutes": candle.real_minutes,
                    "synthetic_minutes": candle.synthetic_minutes,
                }
            ),
            **(
                {}
                if active_timeframes is None
                else {
                    "active_timeframes": tuple(
                        timeframe.value
                        for timeframe in active_timeframes
                    )
                }
            ),
        },
        origin=EventOrigin.NORMALIZED_DATA,
    )


def test_session_clock_only_candle_advances_only_clock_fields() -> None:
    reducer = SessionStateReducer()
    first = reducer.update(_m1_candle(0, 100.0))
    synthetic_candle = replace(
        _m1_candle(1, 50.0, real_completed=False),
        open=40.0,
        high=60.0,
        low=30.0,
        volume=10_000.0,
    )
    synthetic = reducer.update(synthetic_candle)

    assert replace(
        synthetic,
        name=first.name,
        phase=first.phase,
        elapsed_minutes=first.elapsed_minutes,
        known_at=first.known_at,
    ) == first

    resumed = reducer.update(_m1_candle(2, 110.0))
    assert resumed.relative_volume == pytest.approx(1.0)
    assert resumed.realized_volatility == pytest.approx(
        abs(math.log(110.0 / 100.0))
    )


def test_session_rejects_unanchored_or_new_session_clock_only_atomically() -> None:
    empty = SessionStateReducer()
    before = copy.deepcopy(empty.__dict__)
    with pytest.raises(ValueError, match="requires a prior real"):
        empty.update(_m1_candle(0, 90.0, real_completed=False))
    assert empty.__dict__ == before

    anchored = SessionStateReducer()
    anchored.update(_m1_candle(0, 100.0))
    before = copy.deepcopy(anchored.__dict__)
    with pytest.raises(ValueError, match="cannot establish a new session"):
        anchored.update(_m1_candle(481, 90.0, real_completed=False))
    assert anchored.__dict__ == before


def test_pure_atomic_reducer_updates_structure_candidate_and_acceptance() -> None:
    protected_assignment_id = "h1-protected-assignment"
    events = (
        _bar_event(0, Timeframe.H1, close=100.0),
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            1,
            Timeframe.H1,
            direction=Direction.LONG,
            evidence={"structure_id": "h1-structure"},
        ),
        _event(
            EventKind.QUALIFIED_BOS,
            2,
            Timeframe.H1,
            price=105.0,
            direction=Direction.LONG,
            evidence={"bos_id": "h1-bos"},
        ),
        _event(
            EventKind.PROTECTED_SWING_ASSIGNED,
            3,
            Timeframe.H1,
            event_id=protected_assignment_id,
            price=95.0,
            side="below",
            direction=Direction.LONG,
            evidence={"protected_swing_id": "protected-low"},
        ),
        _event(
            EventKind.LIQUIDITY_LEVEL_CREATED,
            4,
            Timeframe.H1,
            price=95.0,
            side="below",
            evidence={
                "level_id": "protected-low",
                "source_kind": "confirmed_swing",
            },
        ),
        _event(
            EventKind.ACCEPTANCE_CONFIRMED,
            5,
            Timeframe.H1,
            price=95.0,
            side="below",
            direction=Direction.SHORT,
            evidence={
                "level_id": "protected-low",
                "protected_swing_id": "protected-low",
                "protected_swing_event_id": protected_assignment_id,
            },
            context_event_ids=(protected_assignment_id,),
            origin=EventOrigin.SEMANTIC_ATOMIC,
        ),
    )

    state = _reduce(events)

    assert state.structure.external_direction is None
    assert state.structure.internal_direction is Direction.LONG
    assert state.structure.protected_low == 95.0
    assert state.structure.protected_swing_intact is False
    assert state.liquidity.candidates == ()
    assert state.quality.source_cutoff == _clock(0)
    assert state.quality.known_at == _clock(5)
    assert state.quality.last_transition_known_at == _clock(5)
    assert state.quality.events_applied == len(events)


def test_candidate_replacement_and_dol_rank_strength_match_both_state_paths() -> None:
    bar = _bar_event(0, Timeframe.M1, close=100.0, atr=2.0)
    old_candidate = _event(
        EventKind.LIQUIDITY_LEVEL_CREATED,
        1,
        Timeframe.M1,
        price=110.0,
        side="above",
        evidence={
            "level_id": "previous-day-high:old",
            "source_kind": "previous_day_high",
            "rank": "external",
            "strength": 0.75,
        },
    )
    replacement = _event(
        EventKind.LIQUIDITY_LEVEL_CREATED,
        2,
        Timeframe.M1,
        price=112.0,
        side="above",
        evidence={
            "level_id": "previous-day-high:new",
            "source_kind": "previous_day_high",
            "replaces_level_id": "previous-day-high:old",
            "replaces_level_event_id": old_candidate.event_id,
            "rank": "external",
            "strength": 0.75,
        },
        context_event_ids=(old_candidate.event_id,),
    )
    atomic = _reduce((bar, old_candidate, replacement))
    atomic_candidate = atomic.liquidity.candidates[0]

    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test"
    )
    rich = publisher._timeframe_state(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=_clock(2),
            bars=3,
            metrics={"atr": 2.0},
            ready=True,
        ),
        asof=_clock(2),
        price=100.0,
        inventory=(
            LiquidityInventoryItem(
                item_id="previous-day-high:new",
                timeframe=Timeframe.M1,
                side="above",
                kind="previous_day_high",
                price=112.0,
                lower_bound=112.0,
                upper_bound=112.0,
                formed_at=_clock(0),
                confirmed_at=_clock(2),
                lifecycle=LiquidityInventoryLifecycle.VISIBLE,
                source_ids=("reference-source",),
                age_bars=0,
                strength=0.75,
                structural_rank="external",
            ),
        ),
        displacement=None,
        events=(),
        anomalies=(),
    )
    rich_candidate = rich.liquidity.candidates[0]

    assert atomic.liquidity.candidate_bsl_ids == (
        "previous-day-high:new",
    )
    assert atomic_candidate.rank == rich_candidate.rank == "external"
    assert atomic_candidate.strength == rich_candidate.strength == pytest.approx(
        0.75
    )
    assert atomic_candidate.path_obstacle_ids == ()
    assert rich_candidate.path_obstacle_ids == ()


def test_candidate_replacement_rejects_forged_identity_or_source_kind() -> None:
    bar = _bar_event(0, Timeframe.M1, close=100.0, atr=2.0)
    old_candidate = _event(
        EventKind.LIQUIDITY_LEVEL_CREATED,
        1,
        Timeframe.M1,
        price=110.0,
        side="above",
        evidence={
            "level_id": "previous-day-high:old",
            "source_kind": "previous_day_high",
        },
    )
    state = _reduce((bar, old_candidate))

    for replacement_event_id, source_kind, context_event_ids in (
        ("forged-old-event", "previous_day_high", ("forged-old-event",)),
        (
            old_candidate.event_id,
            "previous_week_high",
            (old_candidate.event_id,),
        ),
        (old_candidate.event_id, "previous_day_high", ()),
    ):
        forged = _event(
            EventKind.LIQUIDITY_LEVEL_CREATED,
            2,
            Timeframe.M1,
            price=112.0,
            side="above",
            evidence={
                "level_id": "forged-new",
                "source_kind": source_kind,
                "replaces_level_id": "previous-day-high:old",
                "replaces_level_event_id": replacement_event_id,
            },
            context_event_ids=context_event_ids,
        )
        with pytest.raises(ValueError, match="candidate replacement"):
            reduce_timeframe_state(
                state,
                forged,
                semantic_registry_identity="definition-test",
            )


def test_range_invalidation_removes_both_owned_boundaries_only() -> None:
    events = (
        _bar_event(0, Timeframe.H1, close=100.0),
        *(
            _event(
                EventKind.LIQUIDITY_LEVEL_CREATED,
                minute,
                Timeframe.H1,
                price=price,
                side=side,
                evidence={
                    "level_id": f"range-1:{side}",
                    "range_id": "range-1",
                    "source_kind": "mature_range_boundary",
                    "source_formed_at": _clock(0).isoformat(),
                    "source_confirmed_at": _clock(1).isoformat(),
                },
                zone=(price, price),
            )
            for minute, (side, price) in enumerate(
                (("below", 95.0), ("above", 105.0)),
                start=1,
            )
        ),
        _event(
            EventKind.LIQUIDITY_LEVEL_CREATED,
            3,
            Timeframe.H1,
            price=110.0,
            side="above",
            evidence={
                "level_id": "unrelated-swing",
                "source_kind": "confirmed_swing",
            },
        ),
    )
    state = _reduce(events)
    owned_candidates = tuple(
        candidate
        for candidate in state.liquidity.candidates
        if candidate.owner_id == "range-1"
    )
    assert len(owned_candidates) == 2
    assert {
        candidate.owner_id for candidate in state.liquidity.candidates
    } == {"range-1", None}
    assert all(
        candidate.lower_bound <= candidate.price <= candidate.upper_bound
        and candidate.formed_at <= candidate.confirmed_at
        for candidate in state.liquidity.candidates
    )
    below = next(
        candidate for candidate in owned_candidates if candidate.side == "below"
    )
    assert (
        below.lower_bound,
        below.upper_bound,
        below.formed_at,
        below.confirmed_at,
        below.owner_id,
    ) == (95.0, 95.0, _clock(0), _clock(1), "range-1")
    primitive = to_primitive(below)
    assert {
        field: primitive[field]
        for field in (
            "lower_bound",
            "upper_bound",
            "formed_at",
            "confirmed_at",
            "owner_id",
        )
    } == {
        "lower_bound": 95.0,
        "upper_bound": 95.0,
        "formed_at": _clock(0).isoformat(),
        "confirmed_at": _clock(1).isoformat(),
        "owner_id": "range-1",
    }
    assert pickle.loads(pickle.dumps(below)) == below
    replayed = _reduce(pickle.loads(pickle.dumps(events)))
    assert replayed.liquidity.candidates == state.liquidity.candidates

    invalidated = reduce_timeframe_state(
        state,
        _event(
            EventKind.DEALING_RANGE_INVALIDATED,
            4,
            Timeframe.H1,
            price=100.0,
            evidence={"range_id": "range-1"},
            zone=(95.0, 105.0),
        ),
        semantic_registry_identity="definition-test",
    )

    assert invalidated is not None
    assert tuple(
        candidate.candidate_id
        for candidate in invalidated.liquidity.candidates
    ) == ("unrelated-swing",)


@pytest.mark.parametrize(
    "field",
    (
        "lower_bound",
        "upper_bound",
        "formed_at",
        "confirmed_at",
        "owner_id",
    ),
)
def test_dol_candidate_current_fields_fail_closed(field: str) -> None:
    state = _reduce(
        (
            _bar_event(0, Timeframe.H1, close=100.0),
            _event(
                EventKind.LIQUIDITY_LEVEL_CREATED,
                1,
                Timeframe.H1,
                price=105.0,
                side="above",
                evidence={
                    "level_id": "range-1:above",
                    "range_id": "range-1",
                    "source_kind": "mature_range_boundary",
                    "source_formed_at": _clock(0).isoformat(),
                    "source_confirmed_at": _clock(1).isoformat(),
                },
                zone=(104.0, 106.0),
            ),
        )
    )
    candidate = state.liquidity.candidates[0]
    invalid_values = {
        "lower_bound": candidate.price + 1.0,
        "upper_bound": candidate.price - 1.0,
        "formed_at": candidate.confirmed_at + pd.Timedelta(minutes=1),
        "confirmed_at": candidate.formed_at - pd.Timedelta(minutes=1),
        "owner_id": "",
    }
    with pytest.raises(ValueError, match="DOL candidate view is invalid"):
        replace(candidate, **{field: invalid_values[field]})


def test_protected_swing_rejects_price_only_and_stale_assignment_acceptance() -> None:
    assignment_id = "live-protected-assignment"
    base_events = (
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            0,
            Timeframe.H1,
            direction=Direction.LONG,
        ),
        _event(
            EventKind.PROTECTED_SWING_ASSIGNED,
            1,
            Timeframe.H1,
            event_id=assignment_id,
            price=95.0,
            side="below",
            direction=Direction.LONG,
            evidence={"protected_swing_id": "protected-low"},
        ),
    )
    state = _reduce(base_events)

    price_only = _event(
        EventKind.ACCEPTANCE_CONFIRMED,
        2,
        Timeframe.H1,
        price=95.0,
        side="below",
        direction=Direction.SHORT,
        evidence={"level_id": "unrelated-at-same-price"},
    )
    state = reduce_timeframe_state(
        state,
        price_only,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is Direction.LONG
    assert state.structure.protected_swing_intact is True

    stale_assignment = _event(
        EventKind.ACCEPTANCE_CONFIRMED,
        3,
        Timeframe.H1,
        price=95.0,
        side="below",
        direction=Direction.SHORT,
        evidence={
            "level_id": "protected-low",
            "protected_swing_id": "protected-low",
            "protected_swing_event_id": "old-protected-assignment",
        },
        context_event_ids=("old-protected-assignment",),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    state = reduce_timeframe_state(
        state,
        stale_assignment,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is Direction.LONG
    assert state.structure.protected_swing_intact is True

    exact_assignment = _event(
        EventKind.ACCEPTANCE_CONFIRMED,
        4,
        Timeframe.H1,
        price=95.0,
        side="below",
        direction=Direction.SHORT,
        evidence={
            "level_id": "protected-low",
            "protected_swing_id": "protected-low",
            "protected_swing_event_id": assignment_id,
        },
        context_event_ids=(assignment_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    state = reduce_timeframe_state(
        state,
        exact_assignment,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is None
    assert state.structure.protected_swing_intact is False


def _live_long_protection_events() -> tuple[MarketEvent, ...]:
    return (
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            0,
            Timeframe.H1,
            direction=Direction.LONG,
            evidence={"structure_id": "long-structure"},
        ),
        _event(
            EventKind.PROTECTED_SWING_ASSIGNED,
            1,
            Timeframe.H1,
            event_id="long-protected-assignment",
            price=95.0,
            side="below",
            direction=Direction.LONG,
            evidence={"protected_swing_id": "long-protected-low"},
        ),
    )


def test_live_protected_regime_ignores_opposite_direction_until_mss() -> None:
    base = _reduce(_live_long_protection_events())
    opposite_generation = _event(
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        2,
        Timeframe.H1,
        direction=Direction.SHORT,
        evidence={"structure_id": "short-candidate-structure"},
    )

    after_generation = reduce_timeframe_state(
        base,
        opposite_generation,
        semantic_registry_identity="definition-test",
    )

    assert after_generation is not None
    assert after_generation.structure == base.structure
    assert (
        after_generation.quality.events_applied
        == base.quality.events_applied + 1
    )

    same_generation = _event(
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        3,
        Timeframe.H1,
        direction=Direction.LONG,
        evidence={"structure_id": "long-reinforcement"},
    )
    after_reinforcement = reduce_timeframe_state(
        after_generation,
        same_generation,
        semantic_registry_identity="definition-test",
    )
    assert after_reinforcement is not None
    assert after_reinforcement.structure.protected_swing_intact is True
    assert (
        after_reinforcement.structure.protected_low_event_id
        == "long-protected-assignment"
    )

    mss = _event(
        EventKind.MSS_CORE_CONFIRMED,
        4,
        Timeframe.H1,
        direction=Direction.SHORT,
        price=94.0,
        side="below",
        evidence={"bos_id": "opposed-close-break"},
    )
    after_mss = reduce_timeframe_state(
        after_reinforcement,
        mss,
        semantic_registry_identity="definition-test",
    )

    assert after_mss is not None
    assert after_mss.structure.external_direction is Direction.LONG
    assert after_mss.structure.internal_direction is Direction.SHORT
    assert after_mss.structure.protected_swing_intact is True
    assert after_mss.delivery.phase is DeliveryPhase.REVERSAL_ATTEMPT


def test_live_protected_regime_rejects_opposite_bos_and_assignment_atomically(
) -> None:
    before = _reduce(_live_long_protection_events())

    opposite_bos = _event(
        EventKind.QUALIFIED_BOS,
        2,
        Timeframe.H1,
        event_id="opposite-qualified-bos",
        direction=Direction.SHORT,
        price=94.0,
        side="below",
        evidence={"bos_id": "short-bos", "scope": "continuation"},
    )
    with pytest.raises(ValueError, match="opposes the live protected"):
        reduce_timeframe_state(
            before,
            opposite_bos,
            semantic_registry_identity="definition-test",
        )

    opposite_assignment = _event(
        EventKind.PROTECTED_SWING_ASSIGNED,
        3,
        Timeframe.H1,
        event_id="opposite-protected-assignment",
        direction=Direction.SHORT,
        price=105.0,
        side="above",
        evidence={"protected_swing_id": "short-protected-high"},
    )
    with pytest.raises(ValueError, match="opposes the live protected"):
        reduce_timeframe_state(
            before,
            opposite_assignment,
            semantic_registry_identity="definition-test",
        )

    accepted_mss = _event(
        EventKind.MSS_CORE_CONFIRMED,
        4,
        Timeframe.H1,
        event_id="accepted-after-rejections",
        direction=Direction.SHORT,
        price=94.0,
        side="below",
        evidence={"bos_id": "short-mss"},
    )
    after_mss = reduce_timeframe_state(
        before,
        accepted_mss,
        semantic_registry_identity="definition-test",
    )
    assert after_mss is not None
    assert after_mss.structure.external_direction is Direction.LONG
    assert after_mss.structure.internal_direction is Direction.SHORT


def test_exact_acceptance_is_the_only_terminal_before_opposite_regime() -> None:
    state = _reduce(_live_long_protection_events())
    missing_context = _event(
        EventKind.ACCEPTANCE_CONFIRMED,
        2,
        Timeframe.H1,
        price=95.0,
        side="below",
        direction=Direction.SHORT,
        evidence={
            "level_id": "long-protected-low",
            "protected_swing_id": "long-protected-low",
            "protected_swing_event_id": "long-protected-assignment",
        },
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    state = reduce_timeframe_state(
        state,
        missing_context,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is Direction.LONG
    assert state.structure.protected_swing_intact is True

    exact_acceptance = _event(
        EventKind.ACCEPTANCE_CONFIRMED,
        3,
        Timeframe.H1,
        price=95.0,
        side="below",
        direction=Direction.SHORT,
        evidence={
            "level_id": "long-protected-low",
            "protected_swing_id": "long-protected-low",
            "protected_swing_event_id": "long-protected-assignment",
        },
        context_event_ids=("long-protected-assignment",),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    state = reduce_timeframe_state(
        state,
        exact_acceptance,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is None
    assert state.structure.internal_direction is Direction.LONG
    assert state.structure.protected_swing_intact is False

    bos_only = _event(
        EventKind.QUALIFIED_BOS,
        4,
        Timeframe.H1,
        event_id="short-bos-after-acceptance",
        direction=Direction.SHORT,
        price=90.0,
        side="below",
        evidence={"bos_id": "short-bos-only", "scope": "continuation"},
    )
    bos_only_state = reduce_timeframe_state(
        state,
        bos_only,
        semantic_registry_identity="definition-test",
    )
    assert bos_only_state is not None
    assert bos_only_state.structure.external_direction is Direction.SHORT
    assert bos_only_state.structure.internal_direction is Direction.SHORT

    opposite_generation = _event(
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        4,
        Timeframe.H1,
        direction=Direction.SHORT,
        evidence={"structure_id": "short-structure"},
    )
    state = reduce_timeframe_state(
        state,
        opposite_generation,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is Direction.SHORT
    assert state.structure.internal_direction is Direction.SHORT

    opposite_bos = _event(
        EventKind.QUALIFIED_BOS,
        5,
        Timeframe.H1,
        direction=Direction.SHORT,
        price=90.0,
        side="below",
        evidence={"bos_id": "short-bos", "scope": "continuation"},
    )
    state = reduce_timeframe_state(
        state,
        opposite_bos,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is Direction.SHORT
    assert state.structure.internal_direction is Direction.SHORT
    assert state.structure.last_bos_direction is Direction.SHORT

    opposite_assignment = _event(
        EventKind.PROTECTED_SWING_ASSIGNED,
        6,
        Timeframe.H1,
        event_id="short-protected-assignment",
        direction=Direction.SHORT,
        price=105.0,
        side="above",
        evidence={"protected_swing_id": "short-protected-high"},
    )
    state = reduce_timeframe_state(
        state,
        opposite_assignment,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is Direction.SHORT
    assert state.structure.protected_high_id == "short-protected-high"
    assert state.structure.protected_swing_intact is True


def test_terminal_assignment_replay_cannot_clear_reestablished_external() -> None:
    state = _reduce(_live_long_protection_events())
    accepted = _event(
        EventKind.ACCEPTANCE_CONFIRMED,
        2,
        Timeframe.H1,
        event_id="terminal-long-protection",
        price=95.0,
        side="below",
        direction=Direction.SHORT,
        evidence={
            "level_id": "long-protected-low",
            "protected_swing_id": "long-protected-low",
            "protected_swing_event_id": "long-protected-assignment",
        },
        context_event_ids=("long-protected-assignment",),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    state = reduce_timeframe_state(
        state,
        accepted,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is None
    assert state.structure.protected_swing_intact is False

    state = reduce_timeframe_state(
        state,
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            3,
            Timeframe.H1,
            event_id="new-short-external",
            direction=Direction.SHORT,
            evidence={"structure_id": "short-structure"},
        ),
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is Direction.SHORT
    assert state.structure.protected_swing_intact is False

    replayed_old_acceptance = replace(
        accepted,
        event_id="replayed-terminal-long-protection",
        observed_at=_clock(4),
        event_time=_clock(4),
        known_at=_clock(4),
        direction=Direction.LONG,
    )
    state = reduce_timeframe_state(
        state,
        replayed_old_acceptance,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.structure.external_direction is Direction.SHORT
    assert state.structure.internal_direction is Direction.SHORT
    assert state.structure.protected_swing_intact is False


@pytest.mark.parametrize(
    "structure_overrides",
    (
        {"protected_low": None},
        {"protected_low_id": None},
        {"protected_low_event_id": None},
        {
            "protected_high": 105.0,
            "protected_high_id": "unexpected-protected-high",
            "protected_high_event_id": "unexpected-high-assignment",
        },
        {"external_direction": Direction.SHORT},
        {"external_direction": None},
    ),
    ids=(
        "missing-price",
        "missing-swing-identity",
        "missing-assignment-identity",
        "both-sides-live",
        "opposite-external",
        "missing-external",
    ),
)
@pytest.mark.parametrize(
    "event",
    (
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            2,
            Timeframe.H1,
            event_id="next-direction",
            direction=Direction.LONG,
            evidence={"structure_id": "next-long-structure"},
        ),
        _event(
            EventKind.QUALIFIED_BOS,
            2,
            Timeframe.H1,
            event_id="next-qualified-bos",
            price=106.0,
            side="above",
            direction=Direction.LONG,
            evidence={"bos_id": "next-long-bos", "scope": "continuation"},
        ),
        _event(
            EventKind.PROTECTED_SWING_ASSIGNED,
            2,
            Timeframe.H1,
            event_id="next-protected-assignment",
            price=96.0,
            side="below",
            direction=Direction.LONG,
            evidence={"protected_swing_id": "next-protected-low"},
        ),
        _event(
            EventKind.MSS_CORE_CONFIRMED,
            2,
            Timeframe.H1,
            event_id="next-mss",
            price=94.0,
            side="below",
            direction=Direction.SHORT,
            evidence={"bos_id": "next-short-mss"},
        ),
        _event(
            EventKind.ACCEPTANCE_CONFIRMED,
            2,
            Timeframe.H1,
            event_id="next-acceptance",
            price=95.0,
            side="below",
            direction=Direction.SHORT,
            evidence={"level_id": "unrelated-level"},
        ),
    ),
    ids=("direction", "qualified-bos", "protected", "mss", "acceptance"),
)
def test_corrupt_live_protection_fails_closed_without_owner_mutation(
    structure_overrides: dict[str, object],
    event: MarketEvent,
) -> None:
    valid = _reduce(_live_long_protection_events())
    corrupt = replace(
        valid,
        structure=replace(valid.structure, **structure_overrides),
    )

    with pytest.raises(ValueError, match="intact protected swing"):
        reduce_timeframe_state(
            corrupt,
            event,
            semantic_registry_identity="definition-test",
        )
    assert corrupt.structure == replace(valid.structure, **structure_overrides)


@pytest.mark.parametrize(
    ("direction", "side", "initial_price", "tighter_price", "looser_price"),
    (
        (Direction.LONG, "below", 95.0, 96.0, 90.0),
        (Direction.SHORT, "above", 105.0, 104.0, 110.0),
    ),
    ids=("long", "short"),
)
def test_live_protected_reassignment_only_tightens_atomically(
    direction: Direction,
    side: str,
    initial_price: float,
    tighter_price: float,
    looser_price: float,
) -> None:
    initial = _reduce(
        (
            _event(
                EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                0,
                Timeframe.H1,
                event_id=f"{direction.value}-structure",
                direction=direction,
                evidence={"structure_id": f"{direction.value}-structure"},
            ),
            _event(
                EventKind.PROTECTED_SWING_ASSIGNED,
                1,
                Timeframe.H1,
                event_id=f"{direction.value}-initial-assignment",
                price=initial_price,
                side=side,
                direction=direction,
                evidence={
                    "protected_swing_id": f"{direction.value}-initial-swing"
                },
            ),
        )
    )
    tighter_assignment = _event(
        EventKind.PROTECTED_SWING_ASSIGNED,
        2,
        Timeframe.H1,
        event_id=f"{direction.value}-tighter-assignment",
        price=tighter_price,
        side=side,
        direction=direction,
        evidence={
            "protected_swing_id": f"{direction.value}-tighter-swing"
        },
    )
    tightened = reduce_timeframe_state(
        initial,
        tighter_assignment,
        semantic_registry_identity="definition-test",
    )
    assert tightened is not None
    if direction is Direction.LONG:
        assert tightened.structure.protected_low == tighter_price
        assert (
            tightened.structure.protected_low_event_id
            == "long-tighter-assignment"
        )
    else:
        assert tightened.structure.protected_high == tighter_price
        assert (
            tightened.structure.protected_high_event_id
            == "short-tighter-assignment"
        )

    with pytest.raises(ValueError, match="cannot loosen live protection"):
        reduce_timeframe_state(
            tightened,
            _event(
                EventKind.PROTECTED_SWING_ASSIGNED,
                3,
                Timeframe.H1,
                event_id=f"{direction.value}-looser-assignment",
                price=looser_price,
                side=side,
                direction=direction,
                evidence={
                    "protected_swing_id": f"{direction.value}-looser-swing"
                },
            ),
            semantic_registry_identity="definition-test",
        )
    assert tightened is not None


def test_atomic_range_and_fvg_lifecycle_keep_unclamped_location_and_clocks() -> None:
    events = (
        _bar_event(0, Timeframe.M5, close=100.0),
        _event(
            EventKind.DEALING_RANGE_CREATED,
            1,
            Timeframe.M5,
            price=100.0,
            source_ids=("range-low", "range-high"),
            evidence={"range_id": "m5-range"},
            zone=(90.0, 110.0),
        ),
        _event(
            EventKind.FVG_CREATED,
            2,
            Timeframe.M5,
            price=102.0,
            direction=Direction.LONG,
            evidence={"fvg_id": "m5-fvg"},
            zone=(101.0, 103.0),
        ),
        _event(
            EventKind.FVG_MIDPOINT_TOUCHED,
            3,
            Timeframe.M5,
            price=102.0,
            direction=Direction.LONG,
            evidence={"fvg_id": "m5-fvg"},
            zone=(101.0, 103.0),
        ),
        # Expiry is tested only as an already-registered input event. The
        # detector intentionally has no unregistered age threshold.
        _event(
            EventKind.FVG_EXPIRED,
            4,
            Timeframe.M5,
            price=102.0,
            direction=Direction.LONG,
            evidence={"fvg_id": "m5-fvg"},
            zone=(101.0, 103.0),
        ),
        _bar_event(5, Timeframe.M5, close=112.0),
    )

    state = _reduce(events)

    assert state.zones.active_fvg == ()
    assert state.range.range_id == "m5-range"
    assert state.range.normalized_location == 1.1
    assert state.range.location_label == "premium"
    assert state.quality.source_cutoff == _clock(5)
    assert state.quality.known_at == _clock(5)
    assert state.quality.last_transition_known_at == _clock(4)


def test_projection_events_can_be_removed_without_changing_atomic_replay() -> None:
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    frames = {
        Timeframe.H1: FrameObservation(
            timeframe=Timeframe.H1,
            cutoff=_clock(0),
            bars=1,
            metrics={"atr": 2.0},
            ready=True,
        ),
        Timeframe.M5: FrameObservation(
            timeframe=Timeframe.M5,
            cutoff=_clock(5),
            bars=1,
            metrics={"atr": 1.0},
            ready=True,
        ),
        Timeframe.M1: FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=_clock(10),
            bars=1,
            metrics={"atr": 1.0},
            ready=True,
        ),
    }
    atomic_events = _relation_events()

    snapshot, projection_events = _publish(publisher,
        asof=_clock(10),
        symbol="NQH5",
        instrument_id=1,
        price=105.0,
        completed_1m=_m1_candle(10),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=atomic_events,
        anomalies=(),
    )
    full_journal = tuple(
        sorted((*atomic_events, *projection_events), key=event_order_key)
    )
    projection_free = tuple(
        event for event in full_journal if event.kind not in PROJECTION_KINDS
    )

    full_state = _timeframe_reducer(
        semantic_registry_identity="definition-test"
    ).replay(full_journal)
    atomic_state = _timeframe_reducer(
        semantic_registry_identity="definition-test"
    ).replay(projection_free)

    assert projection_events
    assert full_state == atomic_state == snapshot.timeframe_states
    assert set(snapshot.timeframe_states) == {
        Timeframe.H1,
        Timeframe.M5,
        Timeframe.M1,
    }

    relation = snapshot.relations["1H__5m"]
    assert relation.role is RelationRole.PARENT_TRANSITION
    assert relation.parent_direction is None
    assert relation.child_direction is None
    assert relation.parent_state_invalidated is False
    assert relation.known_at == _clock(10)
    assert relation.parent_source_cutoff == _clock(0)
    assert relation.child_source_cutoff == _clock(5)
    assert relation.parent_source_cutoff < relation.known_at
    assert relation.child_source_cutoff < relation.known_at


def test_projection_event_construction_can_be_disabled_without_state_drift() -> None:
    frames = {
        Timeframe.H1: FrameObservation(
            timeframe=Timeframe.H1,
            cutoff=_clock(0),
            bars=1,
            metrics={"atr": 2.0},
            ready=True,
        ),
        Timeframe.M5: FrameObservation(
            timeframe=Timeframe.M5,
            cutoff=_clock(5),
            bars=1,
            metrics={"atr": 1.0},
            ready=True,
        ),
        Timeframe.M1: FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=_clock(10),
            bars=1,
            metrics={"atr": 1.0},
            ready=True,
        ),
    }
    atomic_events = _relation_events()

    emitted_publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    emitted, projection_events = _publish(
        emitted_publisher,
        asof=_clock(10),
        symbol="NQH5",
        instrument_id=1,
        price=105.0,
        completed_1m=_m1_candle(10),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=atomic_events,
        anomalies=(),
    )
    suppressed_publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    suppressed, suppressed_projection_events = _publish(
        suppressed_publisher,
        asof=_clock(10),
        symbol="NQH5",
        instrument_id=1,
        price=105.0,
        completed_1m=_m1_candle(10),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=atomic_events,
        anomalies=(),
        emit_projection_events=False,
    )

    assert projection_events
    assert suppressed_projection_events == ()
    assert suppressed.timeframe_states == emitted.timeframe_states
    assert suppressed.relations == emitted.relations
    assert suppressed.session == emitted.session
    assert suppressed.events_this_update == tuple(
        sorted(atomic_events, key=event_order_key)
    )


def test_atomic_snapshot_replay_rebuilds_session_relations_and_epoch_reset() -> None:
    prior_epoch = (
        _canonical_bar_event(0, Timeframe.H1, close=100.0),
        _replayable_m1_event(3, 101.0),
    )
    reset = replace(
        _atomic_reset_event(4),
        evidence={
            "reason": "contract_change_reset",
            "new_symbol": "NQH5",
            "new_instrument_id": 1,
        },
        details={
            "reason": "contract_change_reset",
            "new_symbol": "NQH5",
            "new_instrument_id": 1,
        },
    )
    current_epoch = (
        _canonical_bar_event(5, Timeframe.H1, close=105.0),
        _canonical_bar_event(7, Timeframe.M5, close=104.0, atr=1.0),
        _replayable_m1_event(9, 104.0),
    )

    snapshot = replay_atomic_market_snapshot(
        (*prior_epoch, reset, *current_epoch),
        semantic_registry_identity="definition-test",
    )

    assert snapshot.asof == _clock(9)
    assert snapshot.session.elapsed_minutes == 1
    assert snapshot.timeframe_states[
        Timeframe.H1
    ].structure.external_direction is None
    assert snapshot.timeframe_states[
        Timeframe.H1
    ].liquidity.candidates == ()
    assert snapshot.relations["1H__5m"].role is RelationRole.PARENT_TRANSITION


def test_reducer_rejects_out_of_order_time_and_same_clock_sequence() -> None:
    chronological_later = _bar_event(1, Timeframe.H1, close=101.0)
    chronological_earlier = _bar_event(0, Timeframe.M5, close=100.0)
    with pytest.raises(ValueError, match="known_at order"):
        _timeframe_reducer(
            semantic_registry_identity="definition-test"
        ).replay((chronological_later, chronological_earlier))

    sequence_later = _bar_event(
        0,
        Timeframe.H1,
        close=100.0,
        event_id="same-clock-sequence-2",
        sequence_no=2,
    )
    sequence_earlier = _bar_event(
        0,
        Timeframe.M5,
        close=100.0,
        event_id="same-clock-sequence-1",
        sequence_no=1,
    )
    with pytest.raises(ValueError, match="known_at order"):
        _timeframe_reducer(
            semantic_registry_identity="definition-test"
        ).replay((sequence_later, sequence_earlier))


def test_reducer_rejects_foreign_timeframe_and_cross_tf_version_mix() -> None:
    h1_state = reduce_timeframe_state(
        None,
        _bar_event(0, Timeframe.H1, close=100.0),
        semantic_registry_identity="definition-test",
    )
    assert h1_state is not None
    with pytest.raises(ValueError, match="foreign event"):
        reduce_timeframe_state(
            h1_state,
            _bar_event(1, Timeframe.M5, close=100.0),
            semantic_registry_identity="definition-test",
        )

    reducer = _timeframe_reducer(
        semantic_registry_identity="definition-test"
    )
    reducer.apply(_canonical_bar_event(0, Timeframe.H1, close=100.0))
    with pytest.raises(ValueError, match="semantic versions"):
        reducer.apply(
            _canonical_bar_event(
                1,
                Timeframe.M5,
                close=100.0,
                semantic_version="smc_semantics_v2.0",
            )
        )


def test_observer_wires_definition_bound_store_and_atomic_authority() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )

    assert observer.audit_store.definition_identity == (
        observer.semantic_registry.definition_identity
    )
    assert observer.audit_store.semantic_definition_identity == (
        observer.semantic_registry.identity
    )
    assert observer.market_snapshot_publisher.atomic_authority is True


def test_atomic_authority_rejects_missing_current_root_without_projection_fallback() -> None:
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=0,
            metrics={},
            ready=False,
        )
        for timeframe in (Timeframe.H1, Timeframe.M5, Timeframe.M1)
    }

    with pytest.raises(
        RuntimeError,
        match="requires exactly one current normalized M1 BAR",
    ):
        _publish(publisher,
            asof=_clock(1),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(1, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(
                _canonical_bar_event(0, Timeframe.H1, close=100.0),
            ),
            anomalies=(),
        )


def test_atomic_authority_uses_only_complete_event_reducer_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(5 if timeframe is Timeframe.M5 else 0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in (Timeframe.H1, Timeframe.M5, Timeframe.M1)
    }

    def projection_is_forbidden(*_args, **_kwargs):
        raise AssertionError("atomic authority consulted frame projection")

    monkeypatch.setattr(
        publisher,
        "_timeframe_state",
        projection_is_forbidden,
    )
    snapshot, _ = _publish(publisher,
        asof=_clock(10),
        symbol="NQH5",
        instrument_id=1,
        price=105.0,
        completed_1m=_m1_candle(10),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=_relation_events(),
        anomalies=(),
    )

    assert snapshot.authority is MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER
    assert snapshot.timeframe_states == publisher._event_reducer.states


def test_atomic_authority_rejects_event_state_outside_frame_registry() -> None:
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(2 if timeframe is Timeframe.M1 else 0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in (Timeframe.H1, Timeframe.M1)
    }

    with pytest.raises(ValueError, match="outside its bound registry: 5m"):
        _publish(publisher,
            asof=_clock(2),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(2, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(
                _canonical_bar_event(0, Timeframe.H1, close=100.0),
                _canonical_bar_event(1, Timeframe.M5, close=100.0),
                _replayable_m1_event(2, 100.0),
            ),
            anomalies=(),
        )


def test_projection_compatibility_mode_is_explicit() -> None:
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=False,
    )
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(10),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in (Timeframe.H1, Timeframe.M5, Timeframe.M1)
    }

    snapshot, _ = _publish(publisher,
        asof=_clock(10),
        symbol="NQH5",
        instrument_id=1,
        price=105.0,
        completed_1m=_m1_candle(10, 105.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=_relation_events(),
        anomalies=(),
    )

    assert snapshot.authority is MarketSnapshotAuthority.FRAME_PROJECTION
    assert set(snapshot.timeframe_states) == set(frames)
    assert set(publisher._event_reducer.states) == set(frames)


def test_group4_projection_scanner_remains_explicit_compatibility_mode() -> None:
    candle = _m1_candle(0, 100.0)
    active_timeframes = tuple(
        spec.native_timeframe
        for spec in CORE_TEST_SCALE_SPECS
        if spec.enabled and spec.native_timeframe is not None
    )
    histories = {
        timeframe: (candle,) if timeframe is Timeframe.M1 else ()
        for timeframe in active_timeframes
    }
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=(
                "configs/primitives_structure_liquidity.json"
            ),
            liquidity_protocol=(
                "configs/primitives_structure_liquidity.json"
            ),
            range_auction_protocol="configs/primitives_range.json",
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=False,
            materialize_event_view=False,
            range_auction_projection_only=True,
        )
    )
    update = ReaderUpdate(
        asof=candle.end,
        completed_1m=candle,
        newly_completed=histories,
        histories=histories,
        anomalies=(),
        active_timeframes=active_timeframes,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
    )

    observation = observer.observe(update)

    assert observer.market_snapshot_publisher.atomic_authority is False
    assert observation.market_snapshot is not None
    assert (
        observation.market_snapshot.authority
        is MarketSnapshotAuthority.FRAME_PROJECTION
    )


def test_observer_atomic_warmup_state_is_complete_and_replayable() -> None:
    candle = _m1_candle(0, 100.0)
    active_timeframes = tuple(
        spec.native_timeframe
        for spec in CORE_TEST_SCALE_SPECS
        if spec.enabled and spec.native_timeframe is not None
    )
    histories = {
        timeframe: (candle,) if timeframe is Timeframe.M1 else ()
        for timeframe in active_timeframes
    }
    update = ReaderUpdate(
        asof=candle.end,
        completed_1m=candle,
        newly_completed=histories,
        histories=histories,
        anomalies=(),
        active_timeframes=active_timeframes,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
    )
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )

    observation = observer.observe(update)
    snapshot = observation.market_snapshot
    assert snapshot is not None
    assert snapshot.authority is MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER
    assert set(snapshot.timeframe_states) == set(active_timeframes)
    assert snapshot.timeframe_states[Timeframe.M1].quality.source_cutoff == candle.end
    assert all(
        snapshot.timeframe_states[timeframe].quality.source_cutoff is None
        for timeframe in active_timeframes
        if timeframe is not Timeframe.M1
    )
    normalized_m1 = next(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe is Timeframe.M1
    )
    assert tuple(normalized_m1.evidence["active_timeframes"]) == tuple(
        timeframe.value for timeframe in active_timeframes
    )
    assert normalized_m1.evidence["scale_registry_id"] == (
        CORE_TEST_SCALE_REGISTRY_ID
    )
    assert normalized_m1.source_entity_ids == (
        f"scale_registry:{CORE_TEST_SCALE_REGISTRY_ID}",
    )
    timeframe_projections = tuple(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.TIMEFRAME_STATE_CHANGED
    )
    assert {
        event.timeframe for event in timeframe_projections
    } == set(active_timeframes)
    assert all(
        normalized_m1.event_id in event.source_event_ids
        for event in timeframe_projections
    )

    atomic_events = tuple(
        event
        for event in observer.audit_store.events()
        if event.origin is not EventOrigin.STATE_PROJECTION
    )
    replayed = replay_atomic_market_snapshot(
        atomic_events,
        semantic_registry_identity=observer.semantic_registry.identity,
    )
    assert replayed.authority is MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER
    assert replayed.timeframe_states == snapshot.timeframe_states
    assert replayed.relations == snapshot.relations
    assert replayed.session == snapshot.session


@pytest.mark.parametrize(
    ("m5_bars", "expected_complete"),
    ((14, False), (24, True)),
)
def test_bootstrap_bar_completeness_uses_timeframe_readiness_not_atr_window(
    m5_bars: int,
    expected_complete: bool,
) -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    active_timeframes = tuple(
        spec.native_timeframe
        for spec in CORE_TEST_SCALE_SPECS
        if spec.enabled and spec.native_timeframe is not None
    )
    candles = tuple(
        Candle(
            timeframe=Timeframe.M5,
                start=_clock(index * 5) - pd.Timedelta(5, unit="min"),
            end=_clock(index * 5),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=100.0,
            symbol="NQH5",
            instrument_id=1,
            observed_minutes=5,
            expected_minutes=5,
            complete=True,
        )
        for index in range(1, m5_bars + 1)
    )
    current_m1 = _m1_candle(m5_bars * 5 + 1, 100.0)
    histories = {
        timeframe: (
            candles
            if timeframe is Timeframe.M5
            else (current_m1,)
            if timeframe is Timeframe.M1
            else ()
        )
        for timeframe in active_timeframes
    }
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=(
                candles[-1].end
                if timeframe is Timeframe.M5
                else current_m1.end
            ),
            bars=len(histories[timeframe]),
            metrics={},
            ready=(
                len(histories[timeframe])
                >= observer.config.minimum_bars[timeframe]
            ),
        )
        for timeframe in active_timeframes
    }
    update = ReaderUpdate(
        asof=current_m1.end,
        completed_1m=current_m1,
        newly_completed=histories,
        histories=histories,
        anomalies=(),
        active_timeframes=active_timeframes,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
    )

    observer._emitter._append_available_bar_events(update, histories, frames)
    observer.memory.flush_audit()
    latest_m5 = tuple(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe is Timeframe.M5
    )[-1]

    assert latest_m5.evidence["data_complete"] is expected_complete


def test_atomic_reset_event_restarts_session_and_owner_states() -> None:
    # Exact external/BOS/MSS current-fact owner closure on reset is covered by
    # test_current_fact_snapshot; this test retains Publisher/session replay.
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in (Timeframe.H1, Timeframe.M1)
    }
    first_events = (
        _canonical_bar_event(0, Timeframe.H1, close=100.0),
        _replayable_m1_event(
            2,
            100.0,
            active_timeframes=(Timeframe.H1, Timeframe.M1),
        ),
    )
    first, _ = _publish(publisher,
        asof=_clock(2),
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1_candle(2, 100.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=first_events,
        anomalies=(),
    )
    assert Timeframe.H1 in first.timeframe_states

    reset = _atomic_reset_event(3)
    second_events = (
        reset,
        _replayable_m1_event(
            4,
            101.0,
            active_timeframes=(Timeframe.H1, Timeframe.M1),
        ),
    )
    second, _ = _publish(publisher,
        asof=_clock(4),
        symbol="NQH5",
        instrument_id=1,
        price=101.0,
        completed_1m=_m1_candle(4, 101.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=second_events,
        anomalies=("contract_change_history_reset",),
    )

    assert second.session.elapsed_minutes == 1
    assert second.timeframe_states[
        Timeframe.H1
    ].structure.external_direction is None
    assert second.timeframe_states[
        Timeframe.H1
    ].quality.last_event_id == second_events[-1].event_id
    assert second.timeframe_states[
        Timeframe.H1
    ].quality.events_applied == 1
    replayed = replay_atomic_market_snapshot(
        (*first_events, *second_events),
        semantic_registry_identity="definition-test",
    )
    assert replayed.timeframe_states == second.timeframe_states
    assert replayed.relations == second.relations
    assert replayed.session == second.session


def test_atomic_retained_m1_prefix_uses_same_session_reducer_as_replay() -> None:
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(2),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in (Timeframe.H1, Timeframe.M1)
    }
    active = (Timeframe.H1, Timeframe.M1)
    events = tuple(
        _replayable_m1_event(
            minute,
            price,
            active_timeframes=active,
        )
        for minute, price in ((0, 100.0), (1, 101.0), (2, 99.5))
    )

    live, _ = _publish(publisher,
        asof=_clock(2),
        symbol="NQH5",
        instrument_id=1,
        price=99.5,
        completed_1m=_m1_candle(2, 99.5),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=events,
        anomalies=(),
    )
    replayed = replay_atomic_market_snapshot(
        events,
        semantic_registry_identity="definition-test",
    )

    assert live.session.elapsed_minutes == 3
    assert live.session == replayed.session
    assert live.timeframe_states == replayed.timeframe_states
    assert live.relations == replayed.relations


def test_timeframe_registry_permutation_has_one_canonical_order() -> None:
    left = _timeframe_reducer(
        semantic_registry_identity="definition-test",
        expected_timeframes=(Timeframe.H1, Timeframe.M5, Timeframe.M1),
    )
    right = _timeframe_reducer(
        semantic_registry_identity="definition-test",
        expected_timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.H1),
    )
    event = _replayable_m1_event(0, 100.0)

    left.apply(event)
    right.apply(event)

    assert left.expected_timeframes == right.expected_timeframes == (
        Timeframe.H1,
        Timeframe.M5,
        Timeframe.M1,
    )
    assert tuple(left.states) == tuple(right.states)
    assert left.states == right.states

    active = (Timeframe.H1, Timeframe.M5, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    reversed_frames = {
        timeframe: frames[timeframe]
        for timeframe in reversed(active)
    }
    atomic_event = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    snapshots = []
    for frame_registry in (frames, reversed_frames):
        publisher = _snapshot_publisher(
            semantic_registry_identity="definition-test",
            atomic_authority=True,
        )
        snapshot, _ = _publish(publisher,
            asof=_clock(0),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(0, 100.0),
            frames=frame_registry,
            inventory=(),
            displacement=None,
            semantic_events=(atomic_event,),
            anomalies=(),
        )
        snapshots.append(snapshot)

    assert tuple(snapshots[0].timeframe_states) == tuple(
        snapshots[1].timeframe_states
    )
    assert snapshots[0].fingerprint == snapshots[1].fingerprint


def test_atomic_snapshot_requires_event_store_canonical_input_order() -> None:
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(10),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in (Timeframe.H1, Timeframe.M5, Timeframe.M1)
    }
    events = _relation_events()
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    with pytest.raises(ValueError, match="known_at order"):
        _publish(
            publisher,
            asof=_clock(10),
            symbol="NQH5",
            instrument_id=1,
            price=105.0,
            completed_1m=_m1_candle(10),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=tuple(reversed(events)),
            anomalies=(),
        )
    assert len(publisher.event_store) == 0


def test_clock_only_snapshot_uses_last_real_price_without_relation_drift() -> None:
    active = (Timeframe.H1, Timeframe.M5, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(10),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    first_events = _relation_events()
    first, _ = _publish(publisher,
        asof=_clock(10),
        symbol="NQH5",
        instrument_id=1,
        price=105.0,
        completed_1m=_m1_candle(10, 105.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=first_events,
        anomalies=(),
    )
    clock_only = _replayable_m1_event(
        11,
        90.0,
        active_timeframes=active,
        real_completed=False,
    )
    second, _ = _publish(publisher,
        asof=_clock(11),
        symbol="NQH5",
        instrument_id=1,
        price=90.0,
        completed_1m=_m1_candle(11, 90.0, real_completed=False),
        frames={
            timeframe: replace(frame, cutoff=_clock(11))
            for timeframe, frame in frames.items()
        },
        inventory=(),
        displacement=None,
        semantic_events=(clock_only,),
        anomalies=(),
    )

    assert second.price == first.price == 105.0
    for relation_id, first_relation in first.relations.items():
        assert replace(
            second.relations[relation_id],
            known_at=first_relation.known_at,
        ) == first_relation
    assert replace(
        second.session,
        name=first.session.name,
        phase=first.session.phase,
        elapsed_minutes=first.session.elapsed_minutes,
        known_at=first.session.known_at,
    ) == first.session

    replayed = replay_atomic_market_snapshot(
        (*first_events, clock_only),
        semantic_registry_identity="definition-test",
    )
    assert replayed.price == second.price
    assert replayed.timeframe_states == second.timeframe_states
    assert replayed.relations == second.relations
    assert replayed.session == second.session


def test_unanchored_atomic_clock_only_snapshot_fails_before_mutation() -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    before_session = copy.deepcopy(publisher._session.__dict__)
    before_reducer = _reducer_hot_state(publisher._event_reducer)
    clock_only = _replayable_m1_event(
        0,
        90.0,
        active_timeframes=active,
        real_completed=False,
    )
    with pytest.raises(RuntimeError, match="requires a prior real"):
        _publish(publisher,
            asof=_clock(0),
            symbol="NQH5",
            instrument_id=1,
            price=90.0,
            completed_1m=_m1_candle(0, 90.0, real_completed=False),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(clock_only,),
            anomalies=(),
        )
    assert publisher._session.__dict__ == before_session
    assert _reducer_hot_state(publisher._event_reducer) == before_reducer
    assert publisher._last_projection_payloads == {}
    assert publisher._last_projection_event_ids == {}

    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    real = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    recovered, _ = _publish(publisher,
        asof=_clock(0),
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1_candle(0, 100.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=(real,),
        anomalies=(),
    )
    assert recovered.price == 100.0


def test_malformed_clock_root_fails_before_publisher_mutation() -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(1),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    real = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    clock_only = _replayable_m1_event(
        1,
        90.0,
        active_timeframes=active,
        real_completed=False,
    )
    incomplete_coverage = dict(clock_only.evidence)
    incomplete_coverage.pop("synthetic_minutes")
    malformed = replace(
        clock_only,
        details=incomplete_coverage,
        evidence=incomplete_coverage,
    )
    before = _publisher_hot_state(publisher)

    with pytest.raises(ValueError, match="all-or-none"):
        _publish(publisher,
            asof=_clock(1),
            symbol="NQH5",
            instrument_id=1,
            price=90.0,
            completed_1m=_m1_candle(1, 90.0, real_completed=False),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(real, malformed),
            anomalies=(),
        )

    assert _publisher_hot_state(publisher) == before


@pytest.mark.parametrize(
    ("marker", "value"),
    (
        ("owner_price_update_only", True),
        ("owner_clock_heartbeat_only", True),
        ("price_source_timeframe", Timeframe.M1.value),
    ),
)
@pytest.mark.parametrize(
    "origin",
    (EventOrigin.LEGACY_TRANSPORT, EventOrigin.STATE_PROJECTION),
)
@pytest.mark.parametrize("marker_after_current", (False, True))
def test_private_bar_marker_fails_before_publisher_mutation(
    marker: str,
    value: object,
    origin: EventOrigin,
    marker_after_current: bool,
) -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    current = replace(
        _replayable_m1_event(
            0,
            100.0,
            active_timeframes=active,
        ),
        event_id=f"current:{origin.value}:{marker}:{marker_after_current}",
        sequence_no=0 if marker_after_current else 1,
    )
    bar = _bar_event(
        0,
        Timeframe.H1,
        close=100.0,
        sequence_no=1 if marker_after_current else 0,
        event_id=f"marker:{origin.value}:{marker}:{marker_after_current}",
    )
    evidence = {**dict(bar.evidence), marker: value}
    forged = replace(
        bar,
        details=evidence,
        evidence=evidence,
        origin=origin,
    )
    before = _publisher_hot_state(publisher)

    expected_error = (
        "projection hash"
        if origin is EventOrigin.STATE_PROJECTION
        else "noncanonical origin"
    )
    batch = (
        (current, forged)
        if marker_after_current
        else (forged, current)
    )
    with pytest.raises(ValueError, match=expected_error):
        _publish(publisher,
            asof=_clock(0),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(0, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=batch,
            anomalies=(),
        )

    assert _publisher_hot_state(publisher) == before


@pytest.mark.parametrize(
    "case",
    ("missing_flag", "conflicting_flags", "non_boolean_flags"),
)
def test_legacy_bar_header_is_rejected_before_publisher_mutation(
    case: str,
) -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    current = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    bar = _bar_event(
        0,
        Timeframe.H1,
        close=100.0,
        sequence_no=1,
        event_id=f"malformed-legacy-header:{case}",
    )
    evidence = dict(bar.evidence)
    if case == "missing_flag":
        evidence.pop("clock_only")
    elif case == "conflicting_flags":
        evidence["clock_only"] = True
    else:
        evidence["real_completed"] = 1
    malformed = replace(bar, details=evidence, evidence=evidence)
    before = _publisher_hot_state(publisher)
    with pytest.raises(ValueError, match="noncanonical origin"):
        _publish(publisher,
            asof=_clock(0),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(0, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(current, malformed),
            anomalies=(),
        )

    assert _publisher_hot_state(publisher) == before


def test_legacy_bar_without_private_header_fields_is_rejected() -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    current = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    legacy = _bar_event(
        0,
        Timeframe.H1,
        close=100.0,
        sequence_no=1,
        event_id="valid-legacy-bar-after-current",
    )

    before = _publisher_hot_state(publisher)
    with pytest.raises(ValueError, match="noncanonical origin"):
        _publish(publisher,
            asof=_clock(0),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(0, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(current, legacy),
            anomalies=(),
        )
    assert _publisher_hot_state(publisher) == before


def test_epoch_reset_clock_only_snapshot_requires_new_epoch_real_anchor() -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    first = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    _publish(publisher,
        asof=_clock(0),
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1_candle(0, 100.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=(first,),
        anomalies=(),
    )
    publisher.on_boundary()
    reset = _atomic_reset_event(1)
    clock_only = _replayable_m1_event(
        2,
        90.0,
        active_timeframes=active,
        real_completed=False,
    )
    before_session = copy.deepcopy(publisher._session.__dict__)
    before_reducer = _reducer_hot_state(publisher._event_reducer)
    with pytest.raises(RuntimeError, match="requires a prior real"):
        _publish(publisher,
            asof=_clock(2),
            symbol="NQH5",
            instrument_id=1,
            price=90.0,
            completed_1m=_m1_candle(2, 90.0, real_completed=False),
            frames={
                timeframe: replace(frame, cutoff=_clock(2))
                for timeframe, frame in frames.items()
            },
            inventory=(),
            displacement=None,
            semantic_events=(reset, clock_only),
            anomalies=("contract_change_history_reset",),
        )
    assert publisher._session.__dict__ == before_session
    assert _reducer_hot_state(publisher._event_reducer) == before_reducer
    assert publisher._boundary_reset_pending is True


def test_epoch_reset_after_current_m1_fails_before_publisher_mutation() -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    first = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    _publish(publisher,
        asof=_clock(0),
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1_candle(0, 100.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=(first,),
        anomalies=(),
    )

    current = _replayable_m1_event(
        1,
        101.0,
        active_timeframes=active,
    )
    trailing_reset = _atomic_reset_event(
        1,
        event_id="malordered-contract-change-reset",
        sequence_no=1,
        reason="malordered_contract_change_reset",
    )
    current_frames = {
        timeframe: replace(frame, cutoff=_clock(1))
        for timeframe, frame in frames.items()
    }
    before = _publisher_hot_state(publisher)
    checkpoint = pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(RuntimeError, match="must strictly precede"):
        _publish(publisher,
            asof=_clock(1),
            symbol="NQH5",
            instrument_id=1,
            price=101.0,
            completed_1m=_m1_candle(1, 101.0),
            frames=current_frames,
            inventory=(),
            displacement=None,
            semantic_events=(current, trailing_reset),
            anomalies=("malordered_contract_change_reset",),
        )

    assert _publisher_hot_state(publisher) == before
    publisher = pickle.loads(checkpoint)
    recovered, _ = _publish(publisher,
        asof=_clock(1),
        symbol="NQH5",
        instrument_id=1,
        price=101.0,
        completed_1m=_m1_candle(1, 101.0),
        frames=current_frames,
        inventory=(),
        displacement=None,
        semantic_events=(current,),
        anomalies=(),
    )
    assert recovered.price == 101.0


def test_future_canonical_event_fails_before_publisher_mutation() -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    current = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    future = _canonical_bar_event(
        1,
        Timeframe.H1,
        close=101.0,
        event_id="future-h1-bar",
    )
    before = _publisher_hot_state(publisher)

    with pytest.raises(RuntimeError, match="event after snapshot asof"):
        _publish(publisher,
            asof=_clock(0),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(0, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(current, future),
            anomalies=(),
        )

    assert _publisher_hot_state(publisher) == before


def test_same_asof_canonical_event_after_current_m1_remains_valid() -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(0),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    current = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    same_asof = _canonical_bar_event(
        0,
        Timeframe.H1,
        close=100.0,
        sequence_no=1,
        event_id="same-asof-h1-bar",
    )

    snapshot, _ = _publish(publisher,
        asof=_clock(0),
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1_candle(0, 100.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=(current, same_asof),
        anomalies=(),
    )

    assert snapshot.price == 100.0
    assert snapshot.timeframe_states[
        Timeframe.H1
    ].quality.source_cutoff == _clock(0)


def test_snapshot_publisher_checkpoint_binds_price_and_session_to_replay() -> None:
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(1),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    events = (
        _replayable_m1_event(0, 100.0, active_timeframes=active),
        _replayable_m1_event(
            1,
            50.0,
            active_timeframes=active,
            real_completed=False,
        ),
    )
    _publish(publisher,
        asof=_clock(1),
        symbol="NQH5",
        instrument_id=1,
        price=50.0,
        completed_1m=_m1_candle(1, 50.0, real_completed=False),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=events,
        anomalies=(),
    )
    restored = pickle.loads(
        pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert restored._last_real_m1_price == 100.0
    assert restored._session.__dict__ == publisher._session.__dict__

    price_tamper = copy.deepcopy(restored)
    price_tamper._last_real_m1_price = 50.0
    with pytest.raises(ValueError, match="differs from M1 replay"):
        pickle.loads(
            pickle.dumps(price_tamper, protocol=pickle.HIGHEST_PROTOCOL)
        )

    session_tamper = copy.deepcopy(restored)
    session_tamper._session.__dict__.pop("_last_relative_volume")
    with pytest.raises(ValueError, match="differs from M1 replay"):
        pickle.loads(
            pickle.dumps(session_tamper, protocol=pickle.HIGHEST_PROTOCOL)
        )

    dual_anchor_tamper = copy.deepcopy(restored)
    dual_anchor_tamper._last_real_m1_price = 50.0
    dual_anchor_tamper._session._last_close = 50.0
    with pytest.raises(RuntimeError, match="real M1 anchors diverged"):
        dual_anchor_tamper._require_real_m1_anchor_consistency()

    missing_schema = restored.__getstate__()
    missing_schema.pop("_publisher_state_schema_version")
    with pytest.raises(ValueError, match="checkpoint schema changed"):
        object.__new__(MarketSnapshotPublisher).__setstate__(missing_schema)


def test_synthetic_no_trade_minute_is_replayable_clock_only_root() -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=True,
        )
    )
    starts = (_clock(0), _clock(1), _clock(2))
    updates = tuple(
        reader.on_bar(
            Bar(
                start=start,
                open=90.0 if index == 1 else 100.0,
                high=90.0 if index == 1 else 100.25,
                low=90.0 if index == 1 else 99.75,
                close=90.0 if index == 1 else 100.0,
                volume=0.0 if index == 1 else 10.0,
                symbol="NQH5",
                instrument_id=1,
                synthetic_no_trade=index == 1,
            )
        )
        for index, start in enumerate(starts)
    )

    first_observation = observer.observe(updates[0])
    first = first_observation.market_snapshot
    synthetic_observation = observer.observe(updates[1])
    synthetic = synthetic_observation.market_snapshot
    assert first is not None and synthetic is not None
    assert updates[1].completed_1m.close == 90.0
    assert synthetic.price == synthetic_observation.price == first.price == 100.0
    synthetic_root = next(
        event
        for event in synthetic_observation.semantic_events_this_update
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe is Timeframe.M1
    )
    assert synthetic_root.origin is EventOrigin.NORMALIZED_DATA
    assert synthetic_root.evidence["real_completed"] is False
    assert synthetic_root.evidence["clock_only"] is True
    assert synthetic.session.elapsed_minutes == 2
    assert synthetic.session.known_at == updates[1].asof
    assert replace(
        synthetic.session,
        name=first.session.name,
        phase=first.session.phase,
        elapsed_minutes=first.session.elapsed_minutes,
        known_at=first.session.known_at,
    ) == first.session
    assert synthetic.timeframe_states[
        Timeframe.M1
    ].quality.source_cutoff == updates[0].asof
    for timeframe, first_state in first.timeframe_states.items():
        synthetic_state = synthetic.timeframe_states[timeframe]
        assert replace(synthetic_state, quality=first_state.quality) == first_state
        changed_quality_fields = {
            name
            for name, value in vars(synthetic_state.quality).items()
            if value != getattr(first_state.quality, name)
        }
        assert changed_quality_fields <= {
            "known_at",
            "last_event_id",
            "last_sequence_no",
            "events_applied",
        }
        assert {
            "known_at",
            "last_event_id",
            "events_applied",
        } <= changed_quality_fields
        assert synthetic_state.quality.last_sequence_no == synthetic_root.sequence_no
        assert synthetic_state.quality.known_at == updates[1].asof
        assert (
            synthetic_state.quality.source_cutoff
            == first_state.quality.source_cutoff
        )
        assert synthetic_state.quality.atr == first_state.quality.atr
        assert (
            synthetic_state.quality.data_complete
            == first_state.quality.data_complete
        )
        assert (
            synthetic_state.quality.events_applied
            == first_state.quality.events_applied + 1
        )
    for relation_id, first_relation in first.relations.items():
        assert replace(
            synthetic.relations[relation_id],
            known_at=first_relation.known_at,
        ) == first_relation
    assert not any(
        event.origin is EventOrigin.SEMANTIC_ATOMIC
        for event in synthetic_observation.semantic_events_this_update
    )

    atomic_events = tuple(
        event
        for event in observer.audit_store.events()
        if event.origin is not EventOrigin.STATE_PROJECTION
    )
    replayed = replay_atomic_market_snapshot(
        atomic_events,
        semantic_registry_identity=observer.semantic_registry.identity,
    )
    assert replayed.timeframe_states == synthetic.timeframe_states
    assert replayed.relations == synthetic.relations
    assert replayed.session == synthetic.session
    assert replayed.price == synthetic.price

    resumed = observer.observe(updates[2]).market_snapshot
    assert resumed is not None
    assert resumed.session.elapsed_minutes == 3
    assert resumed.timeframe_states[
        Timeframe.M1
    ].quality.source_cutoff == updates[2].asof


def test_complete_nonreal_higher_timeframe_bar_is_published_as_clock_root() -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    final_observation = None
    prior_snapshot = None
    for index in range(5):
        update = reader.on_bar(
            Bar(
                start=_clock(index),
                open=100.0,
                high=100.25,
                low=99.75,
                close=100.0,
                volume=0.0 if index == 1 else 10.0,
                symbol="NQH5",
                instrument_id=1,
                synthetic_no_trade=index == 1,
            )
        )
        final_observation = observer.observe(update)
        if index == 3:
            prior_snapshot = final_observation.market_snapshot

    assert final_observation is not None and prior_snapshot is not None
    roots = tuple(
        event
        for event in final_observation.semantic_events_this_update
        if event.kind is EventKind.BAR_COMPLETED
        and event.origin is EventOrigin.NORMALIZED_DATA
        and event.known_at == _clock(5)
    )
    by_timeframe = {event.timeframe: event for event in roots}
    assert set(by_timeframe) == {Timeframe.M1, Timeframe.M5}
    m5 = by_timeframe[Timeframe.M5]
    assert m5.evidence["real_completed"] is False
    assert m5.evidence["clock_only"] is True
    assert m5.evidence["complete"] is True
    assert m5.evidence["start"] == _clock(0)
    assert m5.evidence["observed_minutes"] == 5
    assert m5.evidence["expected_minutes"] == 5
    assert m5.evidence["real_minutes"] == 4
    assert m5.evidence["synthetic_minutes"] == 1
    final_snapshot = final_observation.market_snapshot
    assert final_snapshot is not None
    prior_m5 = prior_snapshot.timeframe_states[Timeframe.M5]
    final_m5 = final_snapshot.timeframe_states[Timeframe.M5]
    assert replace(final_m5, quality=prior_m5.quality) == prior_m5
    assert final_m5.quality.events_applied == prior_m5.quality.events_applied + 1


@pytest.mark.parametrize(
    ("synthetic_index", "pivot_index", "source_end_minutes"),
    (
        (1, 2, (1, 3, 4)),
        (2, 1, (1, 2, 4)),
    ),
)
def test_confirmed_swing_sources_ignore_synthetic_clock_bar(
    synthetic_index: int,
    pivot_index: int,
    source_end_minutes: tuple[int, int, int],
) -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=(
                "configs/primitives_structure_liquidity.json"
            ),
            liquidity_protocol=(
                "configs/primitives_structure_liquidity.json"
            ),
            scale_specs=CORE_TEST_SCALE_SPECS,
        )
    )
    real_geometry = {
        0: (100.0, 100.5, 99.5, 100.0),
        pivot_index: (100.0, 102.0, 99.75, 101.0),
        3: (101.0, 101.5, 100.0, 100.5),
    }
    prefix = tuple(
        Bar(
            start=_clock(index),
            open=(
                101.0
                if index == synthetic_index
                else real_geometry[index][0]
            ),
            high=(
                101.0
                if index == synthetic_index
                else real_geometry[index][1]
            ),
            low=(
                101.0
                if index == synthetic_index
                else real_geometry[index][2]
            ),
            close=(
                101.0
                if index == synthetic_index
                else real_geometry[index][3]
            ),
            volume=0.0 if index == synthetic_index else 100.0,
            symbol="NQH5",
            instrument_id=1,
            synthetic_no_trade=index == synthetic_index,
        )
        for index in range(4)
    )
    bars = (
        *prefix,
        Bar(
            start=_clock(4),
            open=100.5,
            high=101.0,
            low=100.0,
            close=100.5,
            volume=100.0,
            symbol="NQH5",
            instrument_id=1,
        ),
    )

    observations = tuple(
        observer.observe(reader.on_bar(bar)) for bar in bars
    )
    swing = next(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.SWING_CONFIRMED
        and event.timeframe is Timeframe.M1
        and event.side == "above"
    )
    source_bars = tuple(
        observer.audit_store.get(event_id)
        for event_id in swing.source_event_ids
    )

    assert all(event is not None for event in source_bars)
    assert tuple(event.known_at for event in source_bars) == tuple(
        _clock(minutes) for minutes in source_end_minutes
    )
    assert all(event.evidence["real_completed"] for event in source_bars)
    assert swing.event_time == _clock(pivot_index)
    assert swing.known_at == _clock(4)
    synthetic_root = next(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe is Timeframe.M1
        and event.known_at == _clock(synthetic_index + 1)
    )
    assert synthetic_root.evidence["real_completed"] is False
    assert synthetic_root.event_id not in swing.source_event_ids
    assert observations[-1].asof == _clock(5)
    assert sum(
        event.kind is EventKind.SWING_CONFIRMED
        and event.entity_id == swing.entity_id
        for event in observer.audit_store.events()
    ) == 1

    replayed = replay_atomic_market_snapshot(
        tuple(
            event
            for event in observer.audit_store.events()
            if event.origin is not EventOrigin.STATE_PROJECTION
        ),
        semantic_registry_identity=observer.semantic_registry.identity,
    )
    assert observations[-1].market_snapshot is not None
    assert replayed.timeframe_states == (
        observations[-1].market_snapshot.timeframe_states
    )
    assert replayed.session == observations[-1].market_snapshot.session

@pytest.mark.parametrize("drift", ("scale_registry", "contract"))
def test_atomic_replay_rejects_stream_identity_drift_without_reset(
    drift: str,
) -> None:
    active = (Timeframe.H1, Timeframe.M1)
    first = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
        scale_registry_id="registry-a",
    )
    second = _replayable_m1_event(
        1,
        100.0,
        active_timeframes=active,
        scale_registry_id=(
            "registry-b" if drift == "scale_registry" else "registry-a"
        ),
        symbol="NQM5" if drift == "contract" else "NQH5",
        instrument_id=2 if drift == "contract" else 1,
    )

    with pytest.raises(ValueError, match="changed without MARKET_EPOCH_RESET"):
        replay_atomic_market_snapshot(
            (first, second),
            semantic_registry_identity="definition-test",
        )

    reset = _atomic_reset_event(
        1,
        event_id=f"reset-{drift}",
        sequence_no=1,
    )
    second = replace(second, sequence_no=2)
    replayed = replay_atomic_market_snapshot(
        (first, reset, second),
        semantic_registry_identity="definition-test",
    )
    assert replayed.asof == second.known_at


def test_atomic_on_boundary_requires_matching_reset_event() -> None:
    publisher = _snapshot_publisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    active = (Timeframe.H1, Timeframe.M1)
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=_clock(1),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in active
    }
    current = _replayable_m1_event(
        1,
        100.0,
        active_timeframes=active,
    )
    publisher.on_boundary()
    checkpoint = pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(RuntimeError, match="requires a MARKET_EPOCH_RESET"):
        _publish(publisher,
            asof=_clock(1),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(1, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(current,),
            anomalies=(),
        )

    publisher = pickle.loads(checkpoint)
    reset = _atomic_reset_event(0)
    snapshot, _ = _publish(publisher,
        asof=_clock(1),
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1_candle(1, 100.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=(reset, current),
        anomalies=(),
    )
    assert snapshot.authority is MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER


def test_quality_rejects_same_timeframe_duplicate_or_reverse_event() -> None:
    first = _bar_event(
        0,
        Timeframe.H1,
        close=100.0,
        sequence_no=2,
        event_id="h1-sequence-2",
    )
    state = reduce_timeframe_state(
        None,
        first,
        semantic_registry_identity="definition-test",
    )
    assert state is not None
    assert state.quality.last_sequence_no == 2

    with pytest.raises(ValueError, match="out-of-order knowledge"):
        reduce_timeframe_state(
            state,
            replace(first, event_id="h1-sequence-1", sequence_no=1),
            semantic_registry_identity="definition-test",
        )
    with pytest.raises(ValueError, match="out-of-order knowledge"):
        reduce_timeframe_state(
            state,
            first,
            semantic_registry_identity="definition-test",
        )
