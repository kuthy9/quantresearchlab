from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import math
import pickle

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader, ReaderUpdate
from smc_trader.event_store import ImmutableEventStore, event_order_key
from smc_trader.foundation_registry import (
    FOUNDATION_CANONICAL_IDENTITY,
    FOUNDATION_VERSION,
)
from smc_trader.market_state import (
    DeliveryPhase,
    MarketSnapshot,
    MarketSnapshotPublisher,
    MarketSnapshotAuthority,
    RelationRole,
    SessionStateReducer,
    TimeframeEventReducer,
    foundation_record_from_projection_event,
    foundation_record_projection_event,
    replay_atomic_market_snapshot,
    reduce_timeframe_state,
    _validate_foundation_authoritative_sources,
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
from smc_trader.semantic_lifecycle import (
    GenerationLifecycle,
    RelationGeneration,
)
from smc_trader.semantic_foundation import (
    FoundationProjection,
    FoundationProjectionReducer,
    FoundationRecord,
    FoundationRecordStatus,
)
from smc_trader.semantic_zones import (
    BaseOriginCore,
    CompatibleStructureKind,
    FVGStructuralLifecycle,
    FVGTerminationCause,
    ZoneEntrySide,
    ZoneFirstRetest,
    ZoneObjectKind,
    qualify_order_block,
    reduce_fvg_termination,
)

from .helpers import (
    CORE_TEST_SCALE_REGISTRY_ID,
    CORE_TEST_SCALE_SPECS,
)
from .test_semantic_foundation_geometry import _balance_range


TZ = "America/New_York"
PROJECTION_KINDS = {
    EventKind.TIMEFRAME_STATE_CHANGED,
    EventKind.RELATION_STATE_CHANGED,
    EventKind.SESSION_STATE_CHANGED,
    EventKind.FOUNDATION_STATE_CHANGED,
}


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
        ImmutableEventStore._require_authoritative_parent_origins(
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
        ImmutableEventStore._normalized_bar_identity(
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
        ImmutableEventStore._normalized_bar_identity(
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
        ImmutableEventStore._normalized_bar_identity(
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
        ImmutableEventStore._normalized_bar_identity(
            forged,
            bar_event_ids={},
        )


def _relation_events() -> tuple[MarketEvent, ...]:
    return (
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


def _foundation_record_history() -> tuple[FoundationRecord, FoundationRecord]:
    active = FVGStructuralLifecycle(
        fvg_id="foundation-fvg",
        source_creation_event_id="foundation-source-created",
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        created_at=_clock(0),
        known_at=_clock(0),
    )
    terminal = reduce_fvg_termination(
        active,
        cause=FVGTerminationCause.CONTRACT_ROLLOVER,
        known_at=_clock(1),
        cause_event_ids=("foundation-source-terminal",),
    )
    terminal = replace(
        terminal,
        terminal_event_id="foundation-source-terminal",
        terminal_source_event_ids=(
            "foundation-source-created",
            "foundation-source-terminal",
        ),
    )
    return FoundationRecord.from_dto(active), FoundationRecord.from_dto(terminal)


def _foundation_fvg_source_events() -> tuple[MarketEvent, ...]:
    """Build the exact canonical ancestry used by foundation replay fixtures."""

    bars: list[MarketEvent] = []
    for index, minutes in enumerate((-15, -10, -5), start=1):
        base = _replayable_m1_event(minutes, 100.0 + index * 0.25)
        evidence = {
            **dict(base.evidence),
            "source_data_ids": (f"foundation-m5:{index}",),
        }
        bars.append(
            replace(
                base,
                event_id=f"foundation-fvg-bar-{index}",
                timeframe=Timeframe.M5,
                details=evidence,
                evidence=evidence,
            )
        )
    creation = _event(
        EventKind.FVG_CREATED,
        0,
        Timeframe.M5,
        event_id="foundation-source-created",
        side="above",
        price=100.75,
        direction=Direction.LONG,
        evidence={"fvg_id": "foundation-fvg"},
        source_ids=tuple(bar.event_id for bar in bars),
        zone=(100.25, 100.75),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    return (*bars, creation)


def _foundation_m5_bar(
    minutes: int,
    *,
    event_id: str,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> MarketEvent:
    base = _replayable_m1_event(minutes, close)
    evidence = {
        **dict(base.evidence),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "source_data_ids": (f"source:{event_id}",),
    }
    return replace(
        base,
        event_id=event_id,
        timeframe=Timeframe.M5,
        price=close,
        details=evidence,
        evidence=evidence,
    )


def _tamper_foundation_projection_state(
    event: MarketEvent,
    *,
    field: str,
    value: object,
) -> MarketEvent:
    primitive = dict(event.evidence["projection_state"])
    primitive[field] = value
    encoded = json.dumps(
        to_primitive(primitive),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    evidence = {
        **dict(event.evidence),
        "projection_state": primitive,
        "projection_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return replace(event, details=evidence, evidence=evidence)


def test_foundation_record_projection_transport_roundtrips_exactly() -> None:
    active_record, _ = _foundation_record_history()

    event = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        sequence_no=7,
    )

    assert event.kind is EventKind.FOUNDATION_STATE_CHANGED
    assert event.origin is EventOrigin.STATE_PROJECTION
    assert event.semantic_version == SMC_SEMANTIC_VERSION
    assert event.evidence["canonical_semantic"] is False
    assert event.evidence["technical_projection"] == "foundation_record"
    assert event.evidence["state_id"] == active_record.record_id
    assert event.evidence["projection_state"]["record_id"] == (
        active_record.record_id
    )
    assert event.evidence["projection_state"]["foundation_version"] == (
        FOUNDATION_VERSION
    )
    assert event.evidence["projection_state"]["registry_identity"] == (
        FOUNDATION_CANONICAL_IDENTITY
    )
    assert foundation_record_from_projection_event(event) == active_record
    assert foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        sequence_no=7,
    ) == event


def test_foundation_projection_can_publish_old_warmup_record_at_current_clock() -> None:
    active_record, _ = _foundation_record_history()
    published_at = _clock(5)
    event = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        published_at=published_at,
        sequence_no=1,
    )

    assert event.event_time == active_record.known_at
    assert event.known_at == published_at
    assert foundation_record_from_projection_event(event) == active_record

    created_sources = _foundation_fvg_source_events()
    current_bar = _replayable_m1_event(
        5,
        100.5,
        active_timeframes=(Timeframe.M1, Timeframe.M5),
    )
    replayed = replay_atomic_market_snapshot(
        (*created_sources, current_bar, event),
        semantic_registry_identity="definition-test",
    )

    assert replayed.asof == published_at
    assert replayed.foundation is not None
    assert replayed.foundation.current_records == (active_record,)
    assert replayed.foundation.record_count == 1
    assert replayed.foundation.asof == active_record.known_at


def test_atomic_foundation_replay_rejects_fvg_creation_kind_substitution() -> None:
    active_record, _ = _foundation_record_history()
    created_sources = _foundation_fvg_source_events()
    wrong_source_id = created_sources[0].event_id
    payload = dict(active_record.payload)
    payload["source_creation_event_id"] = wrong_source_id
    forged = FoundationRecord(
        object_type=active_record.object_type,
        object_id=active_record.object_id,
        status=active_record.status,
        known_at=active_record.known_at,
        payload=payload,
        source_event_ids=(wrong_source_id,),
    )
    current_bar = _replayable_m1_event(
        5,
        100.5,
        active_timeframes=(Timeframe.M1, Timeframe.M5),
    )
    transport = foundation_record_projection_event(
        forged,
        timeframe=Timeframe.M5,
        published_at=current_bar.known_at,
        sequence_no=1,
    )

    with pytest.raises(ValueError, match="FVG creation.*kind"):
        replay_atomic_market_snapshot(
            (*created_sources, current_bar, transport),
            semantic_registry_identity="definition-test",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("symbol", "ESH5"),
        ("instrument_id", 2),
        ("created_at", _clock(-5).isoformat()),
    ),
)
def test_atomic_foundation_replay_binds_fvg_creation_clock_and_contract(
    field: str,
    value: object,
) -> None:
    active_record, _ = _foundation_record_history()
    created_sources = _foundation_fvg_source_events()
    payload = dict(active_record.payload)
    payload[field] = value
    forged = FoundationRecord(
        object_type=active_record.object_type,
        object_id=active_record.object_id,
        status=active_record.status,
        known_at=active_record.known_at,
        payload=payload,
        source_event_ids=active_record.source_event_ids,
    )
    transport = foundation_record_projection_event(
        forged,
        timeframe=Timeframe.M5,
        sequence_no=1,
    )

    with pytest.raises(ValueError, match="FVG creation"):
        replay_atomic_market_snapshot(
            (*created_sources, transport),
            semantic_registry_identity="definition-test",
        )


def test_foundation_transport_rejects_ob_retest_creation_from_sibling_displacement() -> None:
    core = BaseOriginCore(
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        source_displacement_id="qob-displacement",
        source_displacement_event_id="qob-displacement-event",
        anchor_bar_event_ids=("qob-anchor-1", "qob-anchor-2"),
        anchor_candle_ids=("qob-candle-1", "qob-candle-2"),
        anchor_completed_at=(_clock(-10), _clock(-5)),
        lower_bound=99.0,
        upper_bound=101.0,
        body_lower_bound=99.25,
        body_upper_bound=100.75,
        tick_size=0.25,
        formed_at=_clock(-5),
        known_at=_clock(0),
    )
    qualified = qualify_order_block(
        core,
        source_displacement_id=core.source_displacement_id,
        source_displacement_event_id=core.source_displacement_event_id,
        compatible_structure_event_id="qob-structure-event",
        compatible_structure_kind=CompatibleStructureKind.QUALIFIED_BOS,
        qualified_at=_clock(0),
        known_at=_clock(0),
    )
    projection = FoundationProjectionReducer.reduce(
        FoundationProjectionReducer.initial_projection(),
        FoundationRecord.from_dto(core),
    )
    projection = FoundationProjectionReducer.reduce(
        projection,
        FoundationRecord.from_dto(qualified),
    )
    displacement = _event(
        EventKind.DISPLACEMENT_OBSERVED,
        0,
        Timeframe.M5,
        event_id=core.source_displacement_event_id,
        direction=Direction.LONG,
        evidence={"displacement_id": core.source_displacement_id},
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    sibling_creation = _event(
        EventKind.ORIGIN_ZONE_CREATED,
        0,
        Timeframe.M5,
        event_id="qob-sibling-origin-zone",
        direction=Direction.LONG,
        evidence={
            "origin_zone_id": "legacy-sibling-zone",
            "source_displacement_id": "sibling-displacement",
        },
        source_ids=(displacement.event_id,),
        zone=(99.0, 101.0),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    interaction = _foundation_m5_bar(
        5,
        event_id="qob-first-retest-bar",
        open_=102.0,
        high=102.25,
        low=100.0,
        close=101.5,
    )
    retest = ZoneFirstRetest(
        object_kind=ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
        object_id=qualified.qualified_ob_id,
        creation_event_id=sibling_creation.event_id,
        departure_source_event_id=displacement.event_id,
        source_bar_event_id=interaction.event_id,
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        known_at=_clock(5),
        entry_side=ZoneEntrySide.FROM_ABOVE,
        fill_fraction=0.5,
        age_bars=1,
        age_seconds=300,
        session="new_york",
        context_event_ids=(),
    )
    exact_creation = replace(
        sibling_creation,
        event_id="qob-exact-origin-zone",
        details={
            **dict(sibling_creation.evidence),
            "source_displacement_id": core.source_displacement_id,
        },
        evidence={
            **dict(sibling_creation.evidence),
            "source_displacement_id": core.source_displacement_id,
        },
    )
    exact_retest = replace(retest, creation_event_id=exact_creation.event_id)
    _validate_foundation_authoritative_sources(
        FoundationRecord.from_dto(exact_retest),
        {
            displacement.event_id: displacement,
            exact_creation.event_id: exact_creation,
            interaction.event_id: interaction,
        },
        projection,
    )

    with pytest.raises(ValueError, match="Qualified OB creation is incompatible"):
        _validate_foundation_authoritative_sources(
            FoundationRecord.from_dto(retest),
            {
                displacement.event_id: displacement,
                sibling_creation.event_id: sibling_creation,
                interaction.event_id: interaction,
            },
            projection,
        )


def test_foundation_transport_binds_base_origin_anchor_geometry_and_clocks() -> None:
    def anchor(
        minutes: int,
        *,
        event_id: str,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> MarketEvent:
        bar = _foundation_m5_bar(
            minutes,
            event_id=event_id,
            open_=open_,
            high=high,
            low=low,
            close=close,
        )
        evidence = {
            **dict(bar.evidence),
            "detector_candle_id": f"candle:{event_id}",
        }
        return replace(bar, details=evidence, evidence=evidence)

    first = anchor(
        -10,
        event_id="base-origin-anchor-1",
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
    )
    second = anchor(
        -5,
        event_id="base-origin-anchor-2",
        open_=100.5,
        high=101.5,
        low=99.25,
        close=100.75,
    )
    displacement = _event(
        EventKind.DISPLACEMENT_OBSERVED,
        0,
        Timeframe.M5,
        event_id="base-origin-displacement-event",
        direction=Direction.LONG,
        evidence={"displacement_id": "base-origin-displacement"},
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    core = BaseOriginCore(
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        source_displacement_id="base-origin-displacement",
        source_displacement_event_id=displacement.event_id,
        anchor_bar_event_ids=(first.event_id, second.event_id),
        anchor_candle_ids=(
            first.evidence["detector_candle_id"],
            second.evidence["detector_candle_id"],
        ),
        anchor_completed_at=(first.known_at, second.known_at),
        lower_bound=99.0,
        upper_bound=101.5,
        body_lower_bound=100.0,
        body_upper_bound=100.75,
        tick_size=0.25,
        formed_at=second.known_at,
        known_at=displacement.known_at,
    )
    authority = {
        first.event_id: first,
        second.event_id: second,
        displacement.event_id: displacement,
    }
    _validate_foundation_authoritative_sources(
        FoundationRecord.from_dto(core),
        authority,
    )

    forged = replace(core, lower_bound=98.75)
    with pytest.raises(ValueError, match="Base Origin ancestry, clocks, or BAR geometry"):
        _validate_foundation_authoritative_sources(
            FoundationRecord.from_dto(forged),
            authority,
        )


def test_foundation_transport_binds_balance_range_lifecycle_fact() -> None:
    state = _balance_range()
    source_member_ids = (
        *state.lower_source_member_swing_ids,
        *state.upper_source_member_swing_ids,
    )
    evidence = {
        "range_id": state.range_id,
        "lifecycle": state.lifecycle.value,
        "lower_bound": state.lower_bound,
        "upper_bound": state.upper_bound,
        "lower_source_zone_id": state.lower_source_zone_id,
        "upper_source_zone_id": state.upper_source_zone_id,
        "source_member_swing_ids": source_member_ids,
    }
    lifecycle = replace(
        _event(
            EventKind.DEALING_RANGE_CREATED,
            0,
            Timeframe.H1,
            event_id="balance-range-created",
            evidence=evidence,
            zone=(state.lower_bound, state.upper_bound),
            origin=EventOrigin.SEMANTIC_ATOMIC,
        ),
        observed_at=state.state_started_at,
        event_time=state.formed_at,
        known_at=state.state_started_at,
        source_entity_ids=(
            state.range_id,
            state.lower_source_zone_id,
            state.upper_source_zone_id,
            *source_member_ids,
        ),
    )
    record = FoundationRecord.from_dto(
        state,
        source_event_ids=(lifecycle.event_id,),
    )
    _validate_foundation_authoritative_sources(
        record,
        {lifecycle.event_id: lifecycle},
    )

    wrong_evidence = {**evidence, "lower_bound": state.lower_bound - 0.25}
    forged_lifecycle = replace(
        lifecycle,
        details=wrong_evidence,
        evidence=wrong_evidence,
    )
    with pytest.raises(ValueError, match="BalanceRange lifecycle fact"):
        _validate_foundation_authoritative_sources(
            record,
            {forged_lifecycle.event_id: forged_lifecycle},
        )


@pytest.mark.parametrize(
    "mutation",
    (
        {"fill_fraction": 0.75},
        {"age_bars": 1},
        {"entry_side": ZoneEntrySide.FROM_BELOW},
    ),
)
def test_atomic_foundation_replay_recomputes_first_retest_metrics(
    mutation: dict[str, object],
) -> None:
    active_record, _ = _foundation_record_history()
    created_sources = _foundation_fvg_source_events()
    active_transport = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        sequence_no=1,
    )
    departure = _foundation_m5_bar(
        5,
        event_id="foundation-retest-departure",
        open_=100.75,
        high=101.25,
        low=100.5,
        close=101.0,
    )
    interaction = _foundation_m5_bar(
        10,
        event_id="foundation-retest-interaction",
        open_=101.0,
        high=101.25,
        low=100.5,
        close=101.0,
    )
    valid = ZoneFirstRetest(
        object_kind="fvg",
        object_id=active_record.object_id,
        creation_event_id="foundation-source-created",
        departure_source_event_id=departure.event_id,
        source_bar_event_id=interaction.event_id,
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        known_at=_clock(10),
        entry_side=ZoneEntrySide.FROM_ABOVE,
        fill_fraction=0.5,
        age_bars=2,
        age_seconds=600,
        session="new_york",
        context_event_ids=(),
    )
    forged = replace(valid, **mutation)
    forged_record = FoundationRecord.from_dto(forged)
    final_bar = replace(
        _replayable_m1_event(
            10,
            101.0,
            active_timeframes=(Timeframe.M1, Timeframe.M5),
        ),
        event_id="foundation-retest-current-m1",
        sequence_no=1,
    )
    forged_transport = foundation_record_projection_event(
        forged_record,
        timeframe=Timeframe.M5,
        published_at=_clock(10),
        sequence_no=2,
    )

    with pytest.raises(ValueError, match="first-retest causal metrics"):
        replay_atomic_market_snapshot(
            (
                *created_sources,
                active_transport,
                departure,
                interaction,
                final_bar,
                forged_transport,
            ),
            semantic_registry_identity="definition-test",
        )


def _first_retest_registered_clock_fixture(
    *,
    target_known_at: pd.Timestamp,
    interaction_known_at: pd.Timestamp,
) -> tuple[FoundationRecord, dict[str, MarketEvent], FoundationProjection]:
    active = FVGStructuralLifecycle(
        fvg_id="registered-clock-fvg",
        source_creation_event_id="registered-clock-fvg-created",
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        created_at=target_known_at,
        known_at=target_known_at,
    )
    projection = FoundationProjectionReducer.reduce(
        FoundationProjectionReducer.initial_projection(),
        FoundationRecord.from_dto(active),
    )
    creation = replace(
        _event(
            EventKind.FVG_CREATED,
            0,
            Timeframe.M5,
            event_id=active.source_creation_event_id,
            side="above",
            price=100.75,
            direction=Direction.LONG,
            evidence={"fvg_id": active.fvg_id},
            zone=(100.25, 100.75),
            origin=EventOrigin.SEMANTIC_ATOMIC,
        ),
        observed_at=target_known_at,
        event_time=target_known_at,
        known_at=target_known_at,
    )
    departure = replace(
        _foundation_m5_bar(
            0,
            event_id="registered-clock-departure",
            open_=100.75,
            high=101.25,
            low=100.5,
            close=101.0,
        ),
        observed_at=target_known_at,
        event_time=target_known_at,
        known_at=target_known_at,
    )
    interaction = replace(
        _foundation_m5_bar(
            5,
            event_id="registered-clock-interaction",
            open_=101.0,
            high=101.25,
            low=100.5,
            close=101.0,
        ),
        observed_at=interaction_known_at,
        event_time=interaction_known_at,
        known_at=interaction_known_at,
    )
    retest = ZoneFirstRetest(
        object_kind=ZoneObjectKind.FVG,
        object_id=active.fvg_id,
        creation_event_id=creation.event_id,
        departure_source_event_id=departure.event_id,
        source_bar_event_id=interaction.event_id,
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        known_at=interaction_known_at,
        entry_side=ZoneEntrySide.FROM_ABOVE,
        fill_fraction=0.5,
        age_bars=1,
        age_seconds=int(
            (interaction_known_at - target_known_at).total_seconds()
        ),
        session="new_york",
        context_event_ids=(),
    )
    authority = {
        creation.event_id: creation,
        departure.event_id: departure,
        interaction.event_id: interaction,
    }
    return FoundationRecord.from_dto(retest), authority, projection


def test_foundation_first_retest_replay_accepts_registered_memorial_reopen() -> None:
    record, authority, projection = _first_retest_registered_clock_fixture(
        target_known_at=pd.Timestamp(
            "2024-05-27 13:00",
            tz="America/New_York",
        ),
        interaction_known_at=pd.Timestamp(
            "2024-05-27 18:05",
            tz="America/New_York",
        ),
    )

    _validate_foundation_authoritative_sources(
        record,
        authority,
        projection,
    )


def test_foundation_relation_source_rejects_clock_only_bar() -> None:
    clock_only = _replayable_m1_event(
        0,
        100.0,
        real_completed=False,
    )
    relation = RelationGeneration(
        relation_generation_id="clock-only-relation",
        source_relation_id="H1:M5",
        parent_tf=Timeframe.H1,
        child_tf=Timeframe.M5,
        parent_structure_generation_id="parent-structure",
        child_structure_generation_id="child-structure",
        role="parent_retracement",
        lifecycle=GenerationLifecycle.ACTIVE,
        entered_at=clock_only.known_at,
        known_at=clock_only.known_at,
        last_updated_at=clock_only.known_at,
        observation_count=1,
        latest_relation_digest="clock-only-relation-digest",
        source_event_ids=(clock_only.event_id,),
    )

    with pytest.raises(ValueError, match="Relation Generation source.*exact real BAR"):
        _validate_foundation_authoritative_sources(
            FoundationRecord.from_dto(relation),
            {clock_only.event_id: clock_only},
        )


def test_foundation_first_retest_rejects_clock_only_interaction_bar() -> None:
    record, authority, projection = _first_retest_registered_clock_fixture(
        target_known_at=pd.Timestamp(
            "2024-06-03 10:00",
            tz="America/New_York",
        ),
        interaction_known_at=pd.Timestamp(
            "2024-06-03 10:05",
            tz="America/New_York",
        ),
    )
    interaction_id = record.payload["source_bar_event_id"]
    interaction = authority[interaction_id]
    forged_evidence = {
        **dict(interaction.evidence),
        "real_completed": False,
        "clock_only": True,
    }
    authority[interaction_id] = replace(
        interaction,
        details=forged_evidence,
        evidence=forged_evidence,
    )

    with pytest.raises(
        ValueError,
        match="first-retest interaction BAR.*exact real BAR",
    ):
        _validate_foundation_authoritative_sources(
            record,
            authority,
            projection,
        )


@pytest.mark.parametrize(
    ("target_known_at", "interaction_known_at"),
    (
        (
            pd.Timestamp("2024-05-27 13:00", tz="America/New_York"),
            pd.Timestamp("2024-05-27 18:10", tz="America/New_York"),
        ),
        (
            pd.Timestamp("2024-06-03 10:00", tz="America/New_York"),
            pd.Timestamp("2024-06-03 10:10", tz="America/New_York"),
        ),
    ),
)
def test_foundation_first_retest_replay_rejects_skipped_registered_bucket(
    target_known_at: pd.Timestamp,
    interaction_known_at: pd.Timestamp,
) -> None:
    record, authority, projection = _first_retest_registered_clock_fixture(
        target_known_at=target_known_at,
        interaction_known_at=interaction_known_at,
    )

    with pytest.raises(ValueError, match="first-retest causal metrics"):
        _validate_foundation_authoritative_sources(
            record,
            authority,
            projection,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("record_id", "foundation-record:" + "0" * 64),
        ("foundation_version", "unregistered-foundation-version"),
        ("registry_identity", "0" * 64),
    ),
)
def test_foundation_projection_transport_rejects_rehashed_payload_tamper(
    field: str,
    value: object,
) -> None:
    active_record, _ = _foundation_record_history()
    event = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
    )
    tampered = _tamper_foundation_projection_state(
        event,
        field=field,
        value=value,
    )

    with pytest.raises(ValueError, match="foundation projection"):
        foundation_record_from_projection_event(tampered)


def test_timeframe_reducer_does_not_consume_foundation_projection() -> None:
    active_record, _ = _foundation_record_history()
    reducer = TimeframeEventReducer(
        semantic_registry_identity="definition-test",
        expected_timeframes=(Timeframe.M1,),
    )
    first = replace(
        _replayable_m1_event(
            0,
            100.0,
            active_timeframes=(Timeframe.M1,),
        ),
        event_id="foundation-source-created",
    )
    reducer.apply(first)
    before = reducer.states

    projection = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        sequence_no=1,
    )

    assert reducer.apply(projection) is None
    assert reducer.states == before
    assert projection.event_id not in reducer._event_ids


def test_atomic_replay_rejects_projection_only_foundation_source() -> None:
    active_record, _ = _foundation_record_history()
    base_bar = replace(
        _replayable_m1_event(
            0,
            100.0,
            active_timeframes=(Timeframe.M1,),
        ),
        event_id="unrelated-normalized-bar",
    )
    projection_payload = {"legacy_projection": True}
    encoded = json.dumps(
        projection_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    false_source = _event(
        EventKind.TIMEFRAME_STATE_CHANGED,
        0,
        Timeframe.M1,
        event_id="foundation-source-created",
        sequence_no=1,
        price=None,
        evidence={
            "state_id": Timeframe.M1.value,
            "projection_state": projection_payload,
            "projection_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        origin=EventOrigin.STATE_PROJECTION,
    )
    foundation_event = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        sequence_no=2,
    )

    with pytest.raises(ValueError, match="unavailable earlier source"):
        replay_atomic_market_snapshot(
            (base_bar, false_source, foundation_event),
            semantic_registry_identity="definition-test",
        )


def test_atomic_replay_rejects_source_after_embedded_record_knowledge() -> None:
    active_record, _ = _foundation_record_history()
    late_source = replace(
        _replayable_m1_event(
            1,
            100.0,
            active_timeframes=(Timeframe.M1,),
        ),
        event_id="foundation-source-created",
    )
    foundation_event = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        published_at=_clock(2),
    )

    with pytest.raises(ValueError, match="after embedded record knowledge"):
        replay_atomic_market_snapshot(
            (late_source, foundation_event),
            semantic_registry_identity="definition-test",
        )


def test_cold_foundation_replay_rejects_noncritical_future_source() -> None:
    active_record, _ = _foundation_record_history()
    future_bar = replace(
        _replayable_m1_event(
            1,
            100.0,
            active_timeframes=(Timeframe.M1, Timeframe.M5),
        ),
        event_id="foundation-noncritical-future-source",
    )
    cold_record = FoundationRecord(
        object_type=active_record.object_type,
        object_id=active_record.object_id,
        status=active_record.status,
        known_at=active_record.known_at,
        payload=active_record.payload,
        source_event_ids=(
            *active_record.source_event_ids,
            future_bar.event_id,
        ),
    )

    assert future_bar.event_id not in cold_record.payload.get(
        "context_source_event_ids", ()
    )
    events = (*_foundation_fvg_source_events(), future_bar)
    authoritative = {
        event.event_id: event
        for event in events
        if event.origin
        in {EventOrigin.NORMALIZED_DATA, EventOrigin.SEMANTIC_ATOMIC}
    }
    _validate_foundation_authoritative_sources(
        cold_record,
        authoritative,
    )
    with pytest.raises(
        ValueError,
        match="cold foundation record sources occur after embedded",
    ):
        replay_atomic_market_snapshot(
            events,
            semantic_registry_identity="definition-test",
            foundation_records=(cold_record,),
        )


def test_atomic_replay_mixes_legacy_and_foundation_projection_with_history() -> None:
    active_record, terminal_record = _foundation_record_history()
    active_timeframes = (Timeframe.M1, Timeframe.M5)
    created_sources = _foundation_fvg_source_events()
    terminal_source = _event(
        EventKind.MARKET_EPOCH_RESET,
        1,
        Timeframe.M1,
        event_id="foundation-source-terminal",
        price=None,
        evidence={"reason": "contract_rollover"},
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    final_bar = _replayable_m1_event(
        2,
        100.5,
        active_timeframes=active_timeframes,
    )
    active_projection = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        sequence_no=1,
    )
    terminal_projection = foundation_record_projection_event(
        terminal_record,
        timeframe=Timeframe.M5,
        sequence_no=1,
    )
    repeated_superseded_projection = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        published_at=_clock(2),
        sequence_no=1,
    )
    legacy_state = {"legacy_projection": True}
    legacy_encoded = json.dumps(
        legacy_state,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_projection = _event(
        EventKind.TIMEFRAME_STATE_CHANGED,
        0,
        Timeframe.M1,
        event_id="legacy-state-projection",
        sequence_no=2,
        price=None,
        evidence={
            "state_id": Timeframe.M1.value,
            "projection_state": legacy_state,
            "projection_sha256": hashlib.sha256(legacy_encoded).hexdigest(),
        },
        origin=EventOrigin.STATE_PROJECTION,
    )
    mixed_events = (
        *created_sources,
        active_projection,
        legacy_projection,
        terminal_source,
        terminal_projection,
        final_bar,
        repeated_superseded_projection,
    )

    replayed = replay_atomic_market_snapshot(
        mixed_events,
        semantic_registry_identity="definition-test",
    )
    legacy_only = replay_atomic_market_snapshot(
        (*created_sources, legacy_projection, terminal_source, final_bar),
        semantic_registry_identity="definition-test",
    )

    assert replayed.foundation is not None
    assert replayed.foundation.current_records == (terminal_record,)
    assert replayed.foundation.record_count == 2
    assert "current_records" not in replayed.foundation.identity_payload()
    assert tuple(
        item["record_id"]
        for item in replayed.replay_payload()["foundation"]["current_records"]
    ) == (terminal_record.record_id,)
    missing_schema = replayed.__getstate__()
    missing_schema.pop("schema_version")
    with pytest.raises(ValueError, match="market snapshot pickle schema changed"):
        object.__new__(MarketSnapshot).__setstate__(missing_schema)
    unknown_field = replayed.__getstate__()
    unknown_field["unknown_field"] = "must-fail-closed"
    with pytest.raises(ValueError, match="market snapshot pickle schema changed"):
        object.__new__(MarketSnapshot).__setstate__(unknown_field)
    assert tuple(
        record
        for record in (active_record, terminal_record)
        if record.status is FoundationRecordStatus.ACTIVE
    ) == (active_record,)
    assert tuple(
        record
        for record in (active_record, terminal_record)
        if record.status is FoundationRecordStatus.TERMINAL
    ) == (terminal_record,)
    assert replayed.foundation.active_records == ()
    assert replayed.foundation.terminal_records == (terminal_record,)
    assert replayed.foundation.asof == _clock(1)
    assert replayed.timeframe_states == legacy_only.timeframe_states
    assert replayed.relations == legacy_only.relations
    assert replayed.session == legacy_only.session
    assert legacy_only.foundation is None
    assert "foundation" in replayed.replay_payload()
    assert "foundation" not in legacy_only.replay_payload()
    legacy_primitive = dict(to_primitive(legacy_only))
    legacy_primitive.pop("foundation")
    legacy_primitive.pop("foundation_range_locations")
    expected_legacy_fingerprint = hashlib.sha256(
        json.dumps(
            legacy_primitive,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert legacy_only.fingerprint == expected_legacy_fingerprint
    assert replayed.fingerprint != legacy_only.fingerprint


def test_epoch_reset_retains_foundation_history_until_explicit_terminal() -> None:
    active_record, _ = _foundation_record_history()
    active_dto = FVGStructuralLifecycle(
        fvg_id="foundation-fvg",
        source_creation_event_id="foundation-source-created",
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        created_at=_clock(0),
        known_at=_clock(0),
    )
    terminal_dto = reduce_fvg_termination(
        active_dto,
        cause=FVGTerminationCause.SEMANTIC_RESET,
        known_at=_clock(1),
        cause_event_ids=("foundation-reset",),
    )
    terminal_dto = replace(
        terminal_dto,
        terminal_event_id="foundation-reset",
        terminal_source_event_ids=(
            "foundation-source-created",
            "foundation-reset",
        ),
    )
    terminal_record = FoundationRecord.from_dto(terminal_dto)
    created_sources = _foundation_fvg_source_events()
    active_projection = foundation_record_projection_event(
        active_record,
        timeframe=Timeframe.M5,
        sequence_no=1,
    )
    reset = _event(
        EventKind.MARKET_EPOCH_RESET,
        1,
        Timeframe.M1,
        event_id="foundation-reset",
        price=None,
        evidence={"reason": "semantic_reset"},
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    terminal_projection = foundation_record_projection_event(
        terminal_record,
        timeframe=Timeframe.M5,
        published_at=reset.known_at,
        sequence_no=1,
    )
    new_contract_bar = _replayable_m1_event(
        2,
        101.0,
        active_timeframes=(Timeframe.M1,),
        symbol="NQM5",
        instrument_id=2,
    )

    replayed = replay_atomic_market_snapshot(
        (
                *created_sources,
            active_projection,
            reset,
            terminal_projection,
            new_contract_bar,
        ),
        semantic_registry_identity="definition-test",
    )

    assert replayed.symbol == "NQM5"
    assert replayed.foundation is not None
    assert replayed.foundation.current_records == (terminal_record,)
    assert replayed.foundation.record_count == 2
    assert active_record.status is FoundationRecordStatus.ACTIVE
    assert terminal_record.status is FoundationRecordStatus.TERMINAL
    assert replayed.foundation.latest_records == (terminal_record,)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("foundation_version", "unregistered-foundation-version"),
        ("registry_identity", "0" * 64),
    ),
)
def test_market_snapshot_rejects_foundation_registry_or_version_drift(
    field: str,
    value: str,
) -> None:
    active_record, _ = _foundation_record_history()
    projection = FoundationProjectionReducer.replay((active_record,))
    snapshot = replay_atomic_market_snapshot(
        (
            replace(
                _replayable_m1_event(
                    0,
                    100.0,
                    active_timeframes=(Timeframe.M1,),
                ),
                event_id="foundation-source-created",
            ),
        ),
        semantic_registry_identity="definition-test",
    )
    object.__setattr__(projection, field, value)

    with pytest.raises(ValueError, match="market snapshot identity"):
        replace(snapshot, foundation=projection)


def test_snapshot_identity_stays_constant_size_and_replay_rejects_record_tamper(
) -> None:
    active_record, _ = _foundation_record_history()
    projection = FoundationProjectionReducer.replay((active_record,))
    snapshot = replay_atomic_market_snapshot(
        (
            replace(
                _replayable_m1_event(
                    0,
                    100.0,
                    active_timeframes=(Timeframe.M1,),
                ),
                event_id="foundation-source-created",
            ),
        ),
        semantic_registry_identity="definition-test",
    )
    snapshot = replace(snapshot, foundation=projection)
    fingerprint = snapshot.fingerprint
    record_id = active_record.record_id
    object.__setattr__(
        active_record,
        "source_event_ids",
        (*active_record.source_event_ids, "forged-snapshot-source"),
    )

    assert active_record.record_id == record_id
    assert snapshot.fingerprint == fingerprint
    with pytest.raises(ValueError, match="current record identity"):
        snapshot.replay_payload()
    with pytest.raises(ValueError, match="current record identity"):
        FoundationProjectionReducer.checkpoint(projection)


def test_market_snapshot_rejects_future_foundation_clock() -> None:
    active_record, terminal_record = _foundation_record_history()
    future_projection = FoundationProjectionReducer.replay(
        (active_record, terminal_record)
    )
    snapshot = replay_atomic_market_snapshot(
        (
            replace(
                _replayable_m1_event(
                    0,
                    100.0,
                    active_timeframes=(Timeframe.M1,),
                ),
                event_id="foundation-source-created",
            ),
        ),
        semantic_registry_identity="definition-test",
    )

    with pytest.raises(ValueError, match="market snapshot identity"):
        replace(snapshot, foundation=future_projection)


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

    publisher = MarketSnapshotPublisher(
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
    reducer = TimeframeEventReducer(semantic_registry_identity="definition-test")
    for event in _live_long_protection_events():
        reducer.apply(event)
    before = reducer.states[Timeframe.H1]

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
        reducer.apply(opposite_bos)
    assert reducer.states[Timeframe.H1] == before

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
        reducer.apply(opposite_assignment)
    assert reducer.states[Timeframe.H1] == before

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
    after_mss = reducer.apply(accepted_mss)
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
    reducer = TimeframeEventReducer(semantic_registry_identity="definition-test")
    for base_event in _live_long_protection_events():
        reducer.apply(base_event)
    valid = reducer.states[Timeframe.H1]
    corrupt = replace(
        valid,
        structure=replace(valid.structure, **structure_overrides),
    )
    reducer.states[Timeframe.H1] = corrupt
    owner_before = (
        dict(reducer.states),
        reducer._last_order_key,
        set(reducer._event_ids),
        dict(reducer._events_by_id),
    )

    with pytest.raises(ValueError, match="intact protected swing"):
        reducer.apply(event)

    assert dict(reducer.states) == owner_before[0]
    assert reducer._last_order_key == owner_before[1]
    assert reducer._event_ids == owner_before[2]
    assert reducer._events_by_id == owner_before[3]


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
    reducer = TimeframeEventReducer(semantic_registry_identity="definition-test")
    reducer.apply(
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            0,
            Timeframe.H1,
            event_id=f"{direction.value}-structure",
            direction=direction,
            evidence={"structure_id": f"{direction.value}-structure"},
        )
    )
    reducer.apply(
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
        )
    )
    tightened = reducer.apply(
        _event(
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

    owner_before = (
        dict(reducer.states),
        reducer._last_order_key,
        set(reducer._event_ids),
        dict(reducer._events_by_id),
    )
    with pytest.raises(ValueError, match="cannot loosen live protection"):
        reducer.apply(
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
            )
        )
    assert dict(reducer.states) == owner_before[0]
    assert reducer._last_order_key == owner_before[1]
    assert reducer._event_ids == owner_before[2]
    assert reducer._events_by_id == owner_before[3]


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
    publisher = MarketSnapshotPublisher(
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

    snapshot, projection_events = publisher.publish(
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

    full_state = TimeframeEventReducer(
        semantic_registry_identity="definition-test"
    ).replay(full_journal)
    atomic_state = TimeframeEventReducer(
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
    assert relation.role is RelationRole.PARENT_RETRACEMENT
    assert relation.parent_direction is Direction.LONG
    assert relation.child_direction is Direction.SHORT
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

    emitted, projection_events = MarketSnapshotPublisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    ).publish(
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
    suppressed, suppressed_projection_events = MarketSnapshotPublisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    ).publish(
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
        _bar_event(0, Timeframe.H1, close=100.0),
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            1,
            Timeframe.H1,
            direction=Direction.LONG,
            evidence={"structure_id": "old-structure"},
        ),
        _event(
            EventKind.LIQUIDITY_LEVEL_CREATED,
            2,
            Timeframe.H1,
            side="above",
            price=120.0,
            evidence={
                "level_id": "old-candidate",
                "source_kind": "confirmed_swing",
            },
        ),
        _replayable_m1_event(3, 101.0),
    )
    reset = _event(
        EventKind.MARKET_EPOCH_RESET,
        4,
        Timeframe.M1,
        price=None,
        evidence={
            "reason": "contract_change_reset",
            "new_symbol": "NQH5",
            "new_instrument_id": 1,
        },
    )
    current_epoch = (
        _bar_event(5, Timeframe.H1, close=105.0),
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            6,
            Timeframe.H1,
            direction=Direction.SHORT,
            evidence={"structure_id": "new-structure"},
        ),
        _bar_event(7, Timeframe.M5, close=104.0, atr=1.0),
        _event(
            EventKind.MSS_CORE_CONFIRMED,
            8,
            Timeframe.M5,
            direction=Direction.LONG,
            price=106.0,
            evidence={"bos_id": "new-m5-mss"},
        ),
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
    ].structure.external_direction is Direction.SHORT
    assert snapshot.timeframe_states[
        Timeframe.H1
    ].liquidity.candidates == ()
    assert snapshot.relations["1H__5m"].role is RelationRole.PARENT_RETRACEMENT


def test_reducer_rejects_out_of_order_time_and_same_clock_sequence() -> None:
    chronological_later = _bar_event(1, Timeframe.H1, close=101.0)
    chronological_earlier = _bar_event(0, Timeframe.M5, close=100.0)
    with pytest.raises(ValueError, match="canonical order"):
        TimeframeEventReducer(
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
    with pytest.raises(ValueError, match="canonical order"):
        TimeframeEventReducer(
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

    reducer = TimeframeEventReducer(
        semantic_registry_identity="definition-test"
    )
    reducer.apply(_bar_event(0, Timeframe.H1, close=100.0))
    with pytest.raises(ValueError, match="semantic versions"):
        reducer.apply(
            _bar_event(
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
    publisher = MarketSnapshotPublisher(
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
        publisher.publish(
            asof=_clock(1),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(1, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(
                _bar_event(0, Timeframe.H1, close=100.0),
            ),
            anomalies=(),
        )


def test_atomic_authority_uses_only_complete_event_reducer_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = MarketSnapshotPublisher(
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
    snapshot, _ = publisher.publish(
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
    publisher = MarketSnapshotPublisher(
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
        publisher.publish(
            asof=_clock(2),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(2, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(
                _bar_event(0, Timeframe.H1, close=100.0),
                _bar_event(1, Timeframe.M5, close=100.0),
                _replayable_m1_event(2, 100.0),
            ),
            anomalies=(),
        )


def test_projection_compatibility_mode_is_explicit() -> None:
    publisher = MarketSnapshotPublisher(
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

    snapshot, _ = publisher.publish(
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
            group4_protocol="configs/primitives_range.json",
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=False,
            materialize_event_view=False,
            group4_projection_only=True,
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

    observer._append_available_bar_events(update, histories, frames)
    observer.memory.flush_audit()
    latest_m5 = tuple(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe is Timeframe.M5
    )[-1]

    assert latest_m5.evidence["data_complete"] is expected_complete


def test_atomic_reset_event_restarts_session_and_owner_states() -> None:
    publisher = MarketSnapshotPublisher(
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
        _bar_event(0, Timeframe.H1, close=100.0),
        _event(
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            1,
            Timeframe.H1,
            direction=Direction.LONG,
            evidence={"structure_id": "old-epoch-structure"},
        ),
        _replayable_m1_event(
            2,
            100.0,
            active_timeframes=(Timeframe.H1, Timeframe.M1),
        ),
    )
    first, _ = publisher.publish(
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
    assert first.timeframe_states[
        Timeframe.H1
    ].structure.external_direction is Direction.LONG

    reset = _event(
        EventKind.MARKET_EPOCH_RESET,
        3,
        Timeframe.M1,
        price=None,
        evidence={"reason": "contract_change_reset"},
    )
    second_events = (
        reset,
        _replayable_m1_event(
            4,
            101.0,
            active_timeframes=(Timeframe.H1, Timeframe.M1),
        ),
    )
    second, _ = publisher.publish(
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
    replayed = replay_atomic_market_snapshot(
        (*first_events, *second_events),
        semantic_registry_identity="definition-test",
    )
    assert replayed.timeframe_states == second.timeframe_states
    assert replayed.relations == second.relations
    assert replayed.session == second.session


def test_atomic_retained_m1_prefix_uses_same_session_reducer_as_replay() -> None:
    publisher = MarketSnapshotPublisher(
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

    live, _ = publisher.publish(
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
    left = TimeframeEventReducer(
        semantic_registry_identity="definition-test",
        expected_timeframes=(Timeframe.H1, Timeframe.M5, Timeframe.M1),
    )
    right = TimeframeEventReducer(
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
        publisher = MarketSnapshotPublisher(
            semantic_registry_identity="definition-test",
            atomic_authority=True,
        )
        snapshot, _ = publisher.publish(
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


def test_atomic_snapshot_canonicalizes_input_event_order() -> None:
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
    snapshots = []
    for stream in (events, tuple(reversed(events))):
        publisher = MarketSnapshotPublisher(
            semantic_registry_identity="definition-test",
            atomic_authority=True,
        )
        snapshot, _ = publisher.publish(
            asof=_clock(10),
            symbol="NQH5",
            instrument_id=1,
            price=105.0,
            completed_1m=_m1_candle(10),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=stream,
            anomalies=(),
        )
        snapshots.append(snapshot)

    assert snapshots[0].fingerprint == snapshots[1].fingerprint
    assert snapshots[0].events_this_update == snapshots[1].events_this_update


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
    publisher = MarketSnapshotPublisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    first_events = _relation_events()
    first, _ = publisher.publish(
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
    second, _ = publisher.publish(
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
    publisher = MarketSnapshotPublisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    before_session = copy.deepcopy(publisher._session.__dict__)
    before_reducer = copy.deepcopy(publisher._event_reducer.__dict__)
    clock_only = _replayable_m1_event(
        0,
        90.0,
        active_timeframes=active,
        real_completed=False,
    )
    with pytest.raises(RuntimeError, match="requires a prior real"):
        publisher.publish(
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
    assert publisher._event_reducer.__dict__ == before_reducer
    assert publisher._last_projection_payloads == {}
    assert publisher._last_projection_event_ids == {}

    real = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    recovered, _ = publisher.publish(
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
    publisher = MarketSnapshotPublisher(
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
    before = pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="all-or-none"):
        publisher.publish(
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

    assert pickle.dumps(
        publisher,
        protocol=pickle.HIGHEST_PROTOCOL,
    ) == before


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
    publisher = MarketSnapshotPublisher(
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
    before = pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="private owner fanout markers"):
        publisher.publish(
            asof=_clock(0),
            symbol="NQH5",
            instrument_id=1,
            price=100.0,
            completed_1m=_m1_candle(0, 100.0),
            frames=frames,
            inventory=(),
            displacement=None,
            semantic_events=(forged, current),
            anomalies=(),
        )

    assert pickle.dumps(
        publisher,
        protocol=pickle.HIGHEST_PROTOCOL,
    ) == before


@pytest.mark.parametrize(
    "case",
    ("missing_flag", "conflicting_flags", "non_boolean_flags"),
)
def test_legacy_bar_header_fails_before_publisher_mutation(
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
    publisher = MarketSnapshotPublisher(
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
    before = pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="exact complementary"):
        publisher.publish(
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

    assert pickle.dumps(
        publisher,
        protocol=pickle.HIGHEST_PROTOCOL,
    ) == before


def test_legacy_bar_without_private_header_fields_remains_valid() -> None:
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
    publisher = MarketSnapshotPublisher(
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

    snapshot, _ = publisher.publish(
        asof=_clock(0),
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1_candle(0, 100.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=(legacy, current),
        anomalies=(),
    )

    assert snapshot.price == 100.0
    assert snapshot.timeframe_states[
        Timeframe.H1
    ].quality.source_cutoff == _clock(0)


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
    publisher = MarketSnapshotPublisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    first = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    publisher.publish(
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
    reset = _event(
        EventKind.MARKET_EPOCH_RESET,
        1,
        Timeframe.M1,
        price=None,
        evidence={"reason": "contract_change_reset"},
    )
    clock_only = _replayable_m1_event(
        2,
        90.0,
        active_timeframes=active,
        real_completed=False,
    )
    before_session = copy.deepcopy(publisher._session.__dict__)
    before_reducer = copy.deepcopy(publisher._event_reducer.__dict__)
    with pytest.raises(RuntimeError, match="requires a prior real"):
        publisher.publish(
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
    assert publisher._event_reducer.__dict__ == before_reducer
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
    publisher = MarketSnapshotPublisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    first = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    publisher.publish(
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
    trailing_reset = _event(
        EventKind.MARKET_EPOCH_RESET,
        1,
        Timeframe.M1,
        sequence_no=1,
        price=None,
        evidence={"reason": "malordered_contract_change_reset"},
    )
    current_frames = {
        timeframe: replace(frame, cutoff=_clock(1))
        for timeframe, frame in frames.items()
    }
    before = pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(RuntimeError, match="must strictly precede"):
        publisher.publish(
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

    assert pickle.dumps(
        publisher,
        protocol=pickle.HIGHEST_PROTOCOL,
    ) == before
    recovered, _ = publisher.publish(
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


def test_future_semantic_event_fails_before_publisher_mutation() -> None:
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
    publisher = MarketSnapshotPublisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    current = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    future = _event(
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        1,
        Timeframe.H1,
        direction=Direction.LONG,
        evidence={"structure_id": "future-structure"},
    )
    before = pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(RuntimeError, match="event after snapshot asof"):
        publisher.publish(
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

    assert pickle.dumps(
        publisher,
        protocol=pickle.HIGHEST_PROTOCOL,
    ) == before


def test_same_asof_semantic_event_after_current_m1_remains_valid() -> None:
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
    publisher = MarketSnapshotPublisher(
        semantic_registry_identity="definition-test",
        atomic_authority=True,
    )
    current = _replayable_m1_event(
        0,
        100.0,
        active_timeframes=active,
    )
    same_asof = _event(
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        0,
        Timeframe.H1,
        sequence_no=1,
        direction=Direction.LONG,
        evidence={"structure_id": "same-asof-structure"},
    )

    snapshot, _ = publisher.publish(
        asof=_clock(0),
        symbol="NQH5",
        instrument_id=1,
        price=100.0,
        completed_1m=_m1_candle(0, 100.0),
        frames=frames,
        inventory=(),
        displacement=None,
        semantic_events=(same_asof, current),
        anomalies=(),
    )

    assert snapshot.price == 100.0
    assert snapshot.timeframe_states[
        Timeframe.H1
    ].structure.external_direction is Direction.LONG


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
    publisher = MarketSnapshotPublisher(
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
    publisher.publish(
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
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
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
    assert observer.scene_graph._last_price == first.price
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

    reset = _event(
        EventKind.MARKET_EPOCH_RESET,
        1,
        Timeframe.M1,
        event_id=f"reset-{drift}",
        sequence_no=1,
        price=None,
        evidence={"reason": "contract_change_reset"},
    )
    second = replace(second, sequence_no=2)
    replayed = replay_atomic_market_snapshot(
        (first, reset, second),
        semantic_registry_identity="definition-test",
    )
    assert replayed.asof == second.known_at


def test_atomic_on_boundary_requires_matching_reset_event() -> None:
    publisher = MarketSnapshotPublisher(
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
    with pytest.raises(RuntimeError, match="requires a MARKET_EPOCH_RESET"):
        publisher.publish(
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

    reset = _event(
        EventKind.MARKET_EPOCH_RESET,
        0,
        Timeframe.M1,
        price=None,
        evidence={"reason": "contract_change_reset"},
    )
    snapshot, _ = publisher.publish(
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
