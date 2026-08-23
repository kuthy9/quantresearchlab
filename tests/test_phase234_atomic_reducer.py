from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader, ReaderUpdate
from smc_trader.event_store import event_order_key
from smc_trader.market_state import (
    DeliveryPhase,
    MarketSnapshotPublisher,
    MarketSnapshotAuthority,
    RelationRole,
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
)
from smc_trader.observation import CausalObserver, ObserverConfig

from .helpers import (
    CORE_TEST_SCALE_REGISTRY_ID,
    CORE_TEST_SCALE_SPECS,
)


TZ = "America/New_York"
PROJECTION_KINDS = {
    EventKind.TIMEFRAME_STATE_CHANGED,
    EventKind.RELATION_STATE_CHANGED,
    EventKind.SESSION_STATE_CHANGED,
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
            start=_clock(index * 5) - pd.Timedelta(minutes=5),
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
        for index, start in enumerate(starts)
    )

    first = observer.observe(updates[0]).market_snapshot
    synthetic_observation = observer.observe(updates[1])
    synthetic = synthetic_observation.market_snapshot
    assert first is not None and synthetic is not None
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
    assert synthetic.timeframe_states[
        Timeframe.M1
    ].quality.source_cutoff == updates[0].asof
    assert synthetic.timeframe_states[
        Timeframe.M1
    ].quality.known_at == updates[1].asof
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

    resumed = observer.observe(updates[2]).market_snapshot
    assert resumed is not None
    assert resumed.session.elapsed_minutes == 3
    assert resumed.timeframe_states[
        Timeframe.M1
    ].quality.source_cutoff == updates[2].asof


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
