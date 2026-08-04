from __future__ import annotations

from dataclasses import replace
import hashlib
import pickle
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.causal import ReaderUpdate
from smc_trader.liquidity import CausalLiquidityTracker, LiquidityConfig
from smc_trader.model import (
    BOS_CONFIRMATION_REASON,
    BOSLifecycle,
    Candle,
    Direction,
    EventKind,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    MarketEvent,
    Playbook,
    SUPPORT_RESISTANCE_RETIREMENT_REASON,
    STRUCTURE_BREAK_FAILURE_REASON,
    STRUCTURE_FORMATION_FAILURE_REASON,
    StructureLifecycle,
    SupportResistanceLifecycle,
    SupportResistanceState,
    SwingLifecycle,
    SwingPoint,
    SwingRelation,
    SwingSide,
    Timeframe,
    to_primitive,
)
from smc_trader.observation import (
    CausalObserver,
    EventMemory,
    ObserverConfig,
    _atr,
    _candle_structure,
    _event,
)
from smc_trader.playbooks import BrainConfig, _plan, _visible_levels
from smc_trader.risk import _visible_level_ids
from smc_trader.structure import StructureConfig, StructureTracker

from .helpers import market_observation


TZ = "America/New_York"
BASE = pd.Timestamp("2025-01-06 10:00", tz=TZ)
STRUCTURE_PROTOCOL = (
    "configs/smc_primitives_v3_group12.json"
)


def _candle(
    index: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    synthetic: bool = False,
) -> Candle:
    start = BASE + pd.Timedelta(minutes=index)
    return Candle(
        timeframe=Timeframe.M1,
        start=start,
        end=start + pd.Timedelta(minutes=1),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=0.0 if synthetic else 100.0,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=0 if synthetic else 1,
        synthetic_minutes=1 if synthetic else 0,
    )


def _swing(
    identity: str,
    *,
    side: SwingSide,
    price: float,
    confirmed_index: int,
) -> SwingPoint:
    confirmed_at = BASE + pd.Timedelta(minutes=confirmed_index + 1)
    pivot_start = confirmed_at - pd.Timedelta(minutes=3)
    return SwingPoint(
        swing_id=identity,
        timeframe=Timeframe.M1,
        symbol="NQH5",
        instrument_id=1,
        side=side,
        price=price,
        price_ticks=int(round(price / 0.25)),
        pivot_start=pivot_start,
        pivot_end=pivot_start + pd.Timedelta(minutes=1),
        observed_at=confirmed_at,
        confirmed_at=confirmed_at,
        lifecycle=SwingLifecycle.CONFIRMED,
        relation=(
            SwingRelation.HH
            if side is SwingSide.HIGH
            else SwingRelation.HL
        ),
        delta_ticks=1,
        delta_points=0.25,
        magnitude_atr=0.25,
    )


def _with_authoritative_liquidity(
    observation,
    *,
    inventory: tuple[LiquidityInventoryItem, ...] = (),
    pools: tuple[LiquidityPoolState, ...] = (),
    swings: tuple[SwingPoint, ...] | None = None,
):
    frames = dict(observation.frames)
    for timeframe in observation.active_timeframes:
        sources = tuple(
            replace(
                pool,
                lifecycle=LiquidityPoolLifecycle.FORMED,
                swept_at=None,
                sweep_extreme=None,
                close_outside_on_sweep=None,
                resolved_at=None,
                resolution_reason=None,
            )
            for pool in pools
            if pool.timeframe is timeframe
        )
        frames[timeframe] = replace(
            frames[timeframe],
            liquidity_pools=sources,
            swings=(
                frames[timeframe].swings
                if swings is None
                else tuple(
                    item
                    for item in swings
                    if item.timeframe is timeframe
                )
            ),
        )
    return replace(
        observation,
        frames=frames,
        liquidity_inventory=inventory,
        liquidity_inventory_authoritative=True,
        liquidity_pool_states=pools,
    )


def _inventory_swing_source(
    item: LiquidityInventoryItem,
) -> SwingPoint:
    if item.kind != "swing" or len(item.source_ids) != 1:
        raise ValueError("test swing source requires one swing inventory item")
    return SwingPoint(
        swing_id=item.source_ids[0],
        timeframe=item.timeframe,
        symbol="NQH5",
        instrument_id=1,
        side=(
            SwingSide.HIGH
            if item.side == "above"
            else SwingSide.LOW
        ),
        price=item.price,
        price_ticks=int(round(item.price / 0.25)),
        pivot_start=item.formed_at - pd.Timedelta(minutes=1),
        pivot_end=item.formed_at,
        observed_at=item.confirmed_at,
        confirmed_at=item.confirmed_at,
        lifecycle=(
            SwingLifecycle.BROKEN
            if item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            else SwingLifecycle.CONFIRMED
        ),
        magnitude_atr=item.strength,
        age_bars=item.age_bars,
        broken_at=item.consumed_at,
        failure_reason=(
            "close_beyond_swing"
            if item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
            else None
        ),
    )


def _bull_structure_prefix() -> list[Candle]:
    values = [
        (9.0, 10.0, 8.0, 9.0),
        (10.0, 11.0, 9.0, 10.0),
        (11.0, 13.0, 10.0, 11.5),
        (10.5, 12.0, 9.0, 10.5),
        (9.5, 11.0, 8.0, 9.5),
        (10.0, 12.0, 9.0, 10.5),
        (12.0, 14.0, 10.0, 12.0),
        (11.5, 13.0, 10.0, 11.5),
        (10.5, 12.0, 9.0, 10.5),
        (11.5, 13.0, 10.0, 11.5),
        (13.0, 15.0, 11.0, 13.0),
    ]
    return [
        _candle(
            index,
            open_=open_,
            high=high,
            low=low,
            close=close,
        )
        for index, (open_, high, low, close) in enumerate(values)
    ]


def _forming_identity_regression_prefix() -> list[Candle]:
    """HH, LL, later HH, then HL: one long forming structure."""

    values = [
        (90.0, 85.0),
        (95.0, 84.0),
        (100.0, 83.0),
        (95.0, 82.0),
        (92.0, 81.0),
        (91.0, 80.0),
        (93.0, 81.0),
        (97.0, 82.0),
        (110.0, 83.0),
        (97.0, 82.0),
        (93.0, 75.0),
        (92.0, 70.0),
        (95.0, 75.0),
        (105.0, 80.0),
        (120.0, 85.0),
        (105.0, 82.0),
        (100.0, 80.0),
        (98.0, 75.0),
        (100.0, 80.0),
        (102.0, 82.0),
    ]
    return [
        _candle(
            index,
            open_=(high + low) / 2.0,
            high=high,
            low=low,
            close=(high + low) / 2.0,
        )
        for index, (high, low) in enumerate(values)
    ]


def test_candle_structure_handles_zero_range_synthetic_and_invalid_input() -> None:
    zero = _candle(
        0,
        open_=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        synthetic=True,
    )
    state = _candle_structure(zero)
    assert state.zero_range
    assert state.close_location == 0.5
    assert state.body_ratio == 0.0
    assert state.upper_wick_ratio == 0.0
    assert state.lower_wick_ratio == 0.0
    assert state.direction == 0
    assert not state.real_completed
    assert state.anomalies == ("synthetic_or_partial_completed_candle",)

    normal = _candle(
        1,
        open_=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
    )
    normal_state = _candle_structure(normal)
    assert normal_state.range_points == 3.0
    assert normal_state.body_points == 1.0
    assert normal_state.upper_wick_points == 1.0
    assert normal_state.lower_wick_points == 1.0
    assert normal_state.body_ratio == pytest.approx(1.0 / 3.0)
    assert normal_state.upper_wick_ratio == pytest.approx(1.0 / 3.0)
    assert normal_state.lower_wick_ratio == pytest.approx(1.0 / 3.0)
    assert normal_state.close_location == pytest.approx(2.0 / 3.0)
    assert normal_state.direction == 1
    assert normal_state.real_completed
    assert normal_state.anomalies == ()

    with pytest.raises(ValueError, match="ratios do not match"):
        replace(
            normal_state,
            body_ratio=0.50,
            upper_wick_ratio=0.25,
            lower_wick_ratio=0.25,
        )
    with pytest.raises(ValueError, match="provenance"):
        replace(normal_state, real_completed=False)
    with pytest.raises(ValueError, match="provenance"):
        replace(state, real_completed=True)
    with pytest.raises(ValueError, match="direction"):
        replace(normal_state, direction=0)

    with pytest.raises(ValueError, match="non-finite"):
        replace(zero, close=float("nan"))
    with pytest.raises(ValueError, match="OHLC"):
        replace(zero, high=99.0)
    with pytest.raises(ValueError, match="contract"):
        replace(zero, symbol="")


def test_synthetic_minutes_advance_cutoff_without_changing_semantics() -> None:
    first = _candle(
        0,
        open_=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
    )
    synthetic = _candle(
        1,
        open_=100.5,
        high=100.5,
        low=100.5,
        close=100.5,
        synthetic=True,
    )
    second = _candle(
        2,
        open_=100.5,
        high=102.0,
        low=100.0,
        close=101.5,
    )
    config = ObserverConfig()
    baseline = CausalObserver._observe_frame(
        Timeframe.M1,
        (first, second),
        second.end,
        config,
    )
    with_synthetic = CausalObserver._observe_frame(
        Timeframe.M1,
        (first, synthetic, second),
        second.end,
        config,
    )
    assert with_synthetic.metrics == baseline.metrics
    assert with_synthetic.liquidity == baseline.liquidity
    assert with_synthetic.bars == baseline.bars == 2
    assert _atr((first, synthetic, second), 14) == _atr(
        (first, second),
        14,
    )

    tail = _candle(
        3,
        open_=101.5,
        high=101.5,
        low=101.5,
        close=101.5,
        synthetic=True,
    )
    tail_frame = CausalObserver._observe_frame(
        Timeframe.M1,
        (first, synthetic, second, tail),
        tail.end,
        config,
    )
    assert tail_frame.metrics == baseline.metrics
    assert tail_frame.bars == baseline.bars
    assert tail_frame.cutoff == tail.end
    assert tail_frame.candle_structure is not None
    assert not tail_frame.candle_structure.real_completed

    structure_tracker = StructureTracker(
        Timeframe.M1,
        StructureConfig.from_file(STRUCTURE_PROTOCOL),
    )
    liquidity_tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(),
    )
    structure_tracker.on_candle(first)
    current_swings = structure_tracker.snapshot()[0]
    liquidity_tracker.on_candle(first, current_swings)
    before_structure = to_primitive(structure_tracker.snapshot())
    before_liquidity = to_primitive(liquidity_tracker.snapshot())
    before_ranges = tuple(liquidity_tracker._true_ranges)
    structure_tracker.on_candle(synthetic)
    current_swings = structure_tracker.snapshot()[0]
    liquidity_tracker.on_candle(synthetic, current_swings)
    assert to_primitive(structure_tracker.snapshot()) == before_structure
    assert to_primitive(liquidity_tracker.snapshot()) == before_liquidity
    assert tuple(liquidity_tracker._true_ranges) == before_ranges
    assert structure_tracker.last_end == synthetic.end
    assert liquidity_tracker.last_end == synthetic.end

    memory = EventMemory(16)
    impulse = _event(
        EventKind.IMPULSE,
        BASE,
        Timeframe.M1,
        "above",
        100.0,
        0.5,
        formed_at=BASE,
    )
    memory.append(impulse)
    memory.observe_minute(
        _candle(
            0,
            open_=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            synthetic=True,
        )
    )
    memory.observe_minute(
        _candle(
            1,
            open_=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            synthetic=True,
        )
    )
    memory.observe_minute(
        _candle(
            1,
            open_=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            synthetic=True,
        )
    )
    assert memory._synthetic_runs[-1][2] == 2
    with pytest.raises(
        ValueError,
        match="minute provenance changed on retry",
    ):
        memory.observe_minute(
            _candle(
                1,
                open_=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
            )
        )
    assert memory.durations(BASE + pd.Timedelta(minutes=2))[
        impulse.event_id
    ] == 0
    assert memory.ages(BASE + pd.Timedelta(minutes=2))[
        impulse.event_id
    ] == 0
    real_after_gap = _candle(
        2,
        open_=100.0,
        high=100.25,
        low=99.75,
        close=100.0,
    )
    memory.observe_minute(real_after_gap)
    assert memory.durations(real_after_gap.end)[impulse.event_id] == 1
    assert memory.ages(real_after_gap.end)[impulse.event_id] == 1


def test_mixed_frame_tail_is_clock_only_and_reuses_semantic_snapshot() -> None:
    real_m1 = tuple(
        (
            *_bull_structure_prefix(),
            _candle(
                11,
                open_=13.0,
                high=13.5,
                low=12.5,
                close=13.0,
            ),
        )
    )
    real_h4 = tuple(
        replace(candle, timeframe=Timeframe.H4)
        for candle in real_m1[:-1]
    )
    mixed_tail = replace(
        real_m1[-1],
        timeframe=Timeframe.H4,
        open=999.0,
        high=1001.0,
        low=998.0,
        close=1000.0,
        volume=0.0,
        real_minutes=0,
        synthetic_minutes=1,
    )

    def update(
        h4_history: tuple[Candle, ...],
    ) -> ReaderUpdate:
        histories = {
            Timeframe.H4: h4_history,
            Timeframe.H1: (),
            Timeframe.M5: (),
            Timeframe.M1: real_m1,
        }
        newly_completed = {
            Timeframe.H4: (h4_history[-1],),
            Timeframe.H1: (),
            Timeframe.M5: (),
            Timeframe.M1: (real_m1[-1],),
        }
        return ReaderUpdate(
            asof=real_m1[-1].end,
            completed_1m=real_m1[-1],
            newly_completed=newly_completed,
            histories=histories,
            anomalies=(),
        )

    config = ObserverConfig(structure_protocol=STRUCTURE_PROTOCOL)
    real_observation = CausalObserver(config).observe(update(real_h4))
    mixed_observer = CausalObserver(config)
    mixed_update = update((*real_h4, mixed_tail))
    mixed_observation = mixed_observer.observe(mixed_update)
    real_frame = real_observation.frame(Timeframe.H4)
    mixed_frame = mixed_observation.frame(Timeframe.H4)
    assert mixed_frame.cutoff == mixed_tail.end
    assert mixed_frame.bars == real_frame.bars == len(real_h4)
    assert mixed_frame.metrics == real_frame.metrics
    assert mixed_frame.liquidity == real_frame.liquidity
    assert mixed_frame.swings == real_frame.swings
    assert all(
        event.observed_at <= real_h4[-1].end
        for event in mixed_observation.recent_events
        if event.timeframe is Timeframe.H4
    )

    unchanged = replace(
        mixed_update,
        newly_completed={
            timeframe: () for timeframe in Timeframe
        },
    )
    assert (
        mixed_observer._prior_frame_if_unchanged(
            unchanged,
            Timeframe.H4,
        )
        is mixed_frame
    )
    mixed_observer._structure_trackers[
        Timeframe.H4
    ]._last_end = real_h4[-1].end
    with pytest.raises(RuntimeError, match="clock drift"):
        mixed_observer._prior_frame_if_unchanged(
            unchanged,
            Timeframe.H4,
        )
    assert mixed_observer._terminal_failure is not None


def test_frame_event_clock_uses_latest_real_completion() -> None:
    observer = CausalObserver()
    real_cutoff = BASE + pd.Timedelta(minutes=5)
    mixed_cutoff = real_cutoff + pd.Timedelta(minutes=5)
    metrics = {
        "impulse_direction": 1.0,
        "impulse_strength": 0.8,
        "impulse_age_bars": 0.0,
        "impulse_extension_atr": 1.2,
        "pullback_depth": 0.2,
        "pullback_completeness": 0.4,
        "reacceptance_direction": 0.5,
        "compression": 0.7,
        "atr": 1.0,
    }
    observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M5,
            cutoff=mixed_cutoff,
            bars=4,
            metrics=metrics,
        ),
        newly_completed=True,
        event_clock=real_cutoff,
    )
    m5_events = tuple(
        event
        for event in observer.memory.recent()
        if event.timeframe is Timeframe.M5
    )
    assert m5_events
    assert {
        event.observed_at for event in m5_events
    } == {real_cutoff}


def test_cold_event_memory_fails_when_age_origin_predates_m1_prefix() -> None:
    observer = CausalObserver()
    retained = _candle(
        0,
        open_=100.0,
        high=100.25,
        low=99.75,
        close=100.0,
    )
    observer.memory.append(
        _event(
            EventKind.STRUCTURE_STATE,
            retained.end,
            Timeframe.H1,
            "above",
            100.0,
            0.5,
            entity_id="old-structure",
            lifecycle=StructureLifecycle.CONFIRMED.value,
            formed_at=retained.start - pd.Timedelta(minutes=10),
            confirmed_at=retained.end,
        )
    )
    update = ReaderUpdate(
        asof=retained.end,
        completed_1m=retained,
        newly_completed={
            Timeframe.H4: (),
            Timeframe.H1: (),
            Timeframe.M5: (),
            Timeframe.M1: (retained,),
        },
        histories={
            Timeframe.H4: (),
            Timeframe.H1: (),
            Timeframe.M5: (),
            Timeframe.M1: (retained,),
        },
        anomalies=(),
    )
    with pytest.raises(
        ValueError,
        match="event origin predates retained 1m clock coverage",
    ):
        observer.observe(update)
    assert observer._terminal_failure is not None

    bounded = EventMemory(1)
    bounded.set_clock_coverage_start(retained.start)
    bounded.append(
        _event(
            EventKind.TRIGGER_HELD,
            retained.end,
            Timeframe.M1,
            "above",
            100.0,
            0.5,
        )
    )
    bounded.append(
        _event(
            EventKind.TRIGGER_HELD,
            retained.end + pd.Timedelta(minutes=1),
            Timeframe.M1,
            "above",
            100.25,
            0.5,
        )
    )
    with pytest.raises(
        ValueError,
        match="event origin predates retained 1m clock coverage",
    ):
        bounded.append(
            _event(
                EventKind.STRUCTURE_STATE,
                retained.end + pd.Timedelta(minutes=2),
                Timeframe.H4,
                "above",
                100.0,
                0.5,
                entity_id="evicted-old-structure",
                lifecycle=StructureLifecycle.BROKEN.value,
                formed_at=(
                    retained.start - pd.Timedelta(minutes=10)
                ),
            )
        )


def test_structure_identity_is_stable_from_forming_through_break() -> None:
    with pytest.raises(ValueError, match="swing magnitude"):
        replace(
            _swing(
                "negative-magnitude",
                side=SwingSide.HIGH,
                price=100.0,
                confirmed_index=0,
            ),
            magnitude_atr=-0.1,
        )
    tracker = StructureTracker(
        Timeframe.M1,
        StructureConfig.from_file(STRUCTURE_PROTOCOL),
    )
    _, initial_states, _ = tracker.snapshot()
    initial_long = next(
        item for item in initial_states if item.direction is Direction.LONG
    )
    with pytest.raises(ValueError, match="inactive structure"):
        replace(initial_long, failure_reason="arbitrary")

    forming_ids: list[str] = []
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
        _, states, _ = tracker.snapshot()
        long_state = next(
            item for item in states if item.direction is Direction.LONG
        )
        if long_state.lifecycle is StructureLifecycle.FORMING:
            forming_ids.append(long_state.structure_id)
    _, states, _ = tracker.snapshot()
    confirmed = next(
        item for item in states if item.direction is Direction.LONG
    )
    assert confirmed.lifecycle is StructureLifecycle.CONFIRMED
    assert forming_ids
    assert set(forming_ids) == {confirmed.structure_id}
    with pytest.raises(ValueError, match="non-negative"):
        replace(confirmed, cumulative_magnitude_atr=-0.1)

    break_candle = _candle(
        11,
        open_=9.5,
        high=10.0,
        low=8.0,
        close=8.75,
    )
    tracker.on_candle(break_candle)
    swings, states, _ = tracker.snapshot()
    broken_structure = next(
        item for item in states if item.direction is Direction.LONG
    )
    assert broken_structure.lifecycle is StructureLifecycle.BROKEN
    assert broken_structure.structure_id == confirmed.structure_id
    assert broken_structure.broken_at == break_candle.end
    assert broken_structure.failure_reason == STRUCTURE_BREAK_FAILURE_REASON
    with pytest.raises(ValueError, match="registered failure reason"):
        replace(broken_structure, failure_reason="arbitrary")
    broken_swings = [
        item
        for item in swings
        if item.lifecycle is SwingLifecycle.BROKEN
    ]
    assert broken_swings
    assert all(
        item.failure_reason == "close_beyond_swing"
        for item in broken_swings
    )
    protected = next(
        item
        for item in broken_swings
        if item.swing_id == confirmed.protected_swing_id
    )
    assert protected.broken_at == break_candle.end


def test_forming_structure_identity_and_failure_clock_are_frozen() -> None:
    tracker = StructureTracker(
        Timeframe.M1,
        StructureConfig.from_file(STRUCTURE_PROTOCOL),
    )
    long_forming_ids: list[str] = []
    long_formed_at: list[pd.Timestamp] = []
    short_forming_id: str | None = None
    short_formed_at: pd.Timestamp | None = None
    candles = _forming_identity_regression_prefix()
    for candle in candles:
        tracker.on_candle(candle)
        _, states, _ = tracker.snapshot()
        long_state = next(
            item for item in states if item.direction is Direction.LONG
        )
        short_state = next(
            item for item in states if item.direction is Direction.SHORT
        )
        if long_state.lifecycle is StructureLifecycle.FORMING:
            long_forming_ids.append(long_state.structure_id)
            long_formed_at.append(long_state.formed_at)
        if short_state.lifecycle is StructureLifecycle.FORMING:
            short_forming_id = short_state.structure_id
            short_formed_at = short_state.formed_at

    _, states, _ = tracker.snapshot()
    long_state = next(
        item for item in states if item.direction is Direction.LONG
    )
    short_state = next(
        item for item in states if item.direction is Direction.SHORT
    )
    assert long_state.lifecycle is StructureLifecycle.CONFIRMED
    assert len(long_forming_ids) >= 2
    assert set(long_forming_ids) == {long_state.structure_id}
    assert set(long_formed_at) == {long_state.formed_at}
    assert short_state.lifecycle is StructureLifecycle.FORMATION_FAILED
    assert short_state.structure_id == short_forming_id
    assert short_state.formed_at == short_formed_at
    assert short_state.formation_failed_at == candles[-1].end
    assert (
        short_state.failure_reason
        == STRUCTURE_FORMATION_FAILURE_REASON
    )
    with pytest.raises(ValueError, match="registered clock and reason"):
        replace(short_state, failure_reason="arbitrary")

    observer = CausalObserver()
    observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=short_state.formation_failed_at,
            bars=len(candles),
            metrics={},
            structures=(short_state,),
        ),
        newly_completed=True,
    )
    failed_event = next(
        item
        for item in observer.memory.recent()
        if item.entity_id == short_state.structure_id
    )
    assert failed_event.observed_at == short_state.formation_failed_at
    assert failed_event.ended_at == short_state.formation_failed_at
    assert (
        failed_event.transition_reason
        == STRUCTURE_FORMATION_FAILURE_REASON
    )


def test_swing_candidate_remains_visible_until_registered_resolution_clock() -> None:
    tracker = StructureTracker(
        Timeframe.M1,
        StructureConfig.from_file(STRUCTURE_PROTOCOL),
    )
    candles = [
        _candle(0, open_=99.0, high=100.0, low=98.0, close=99.0),
        _candle(1, open_=100.0, high=101.0, low=99.0, close=100.0),
        _candle(2, open_=102.0, high=104.0, low=101.0, close=102.0),
        _candle(3, open_=104.0, high=105.0, low=102.0, close=104.0),
        _candle(4, open_=102.0, high=103.0, low=101.0, close=102.0),
        _candle(5, open_=101.0, high=102.0, low=100.0, close=101.0),
    ]
    for candle in candles[:4]:
        tracker.on_candle(candle)
    swings, _, _ = tracker.snapshot()
    forming = next(
        item
        for item in swings
        if item.pivot_start == candles[2].start
        and item.side is SwingSide.HIGH
    )
    assert forming.lifecycle is SwingLifecycle.FORMING
    assert forming.age_bars == 1

    tracker.on_candle(candles[4])
    swings, _, _ = tracker.snapshot()
    failed = next(
        item
        for item in swings
        if item.swing_id == forming.swing_id
    )
    assert failed.lifecycle is SwingLifecycle.FORMATION_FAILED
    assert failed.failure_reason == "right_side_invalidated"
    assert failed.observed_at == candles[4].end
    assert failed.age_bars == 0
    with pytest.raises(ValueError, match="must resolve after"):
        replace(failed, observed_at=failed.pivot_end)

    tracker.on_candle(candles[5])
    swings, _, _ = tracker.snapshot()
    aged = next(item for item in swings if item.swing_id == forming.swing_id)
    assert aged.age_bars == 1

    confirmed = _swing(
        "clock-test",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=10,
    )
    with pytest.raises(ValueError, match="must follow"):
        replace(
            confirmed,
            observed_at=confirmed.pivot_end,
            confirmed_at=confirmed.pivot_end,
        )


def test_pending_bos_cannot_hide_raw_break_strength() -> None:
    tracker = StructureTracker(
        Timeframe.M1,
        StructureConfig.from_file(STRUCTURE_PROTOCOL),
    )
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
    _, _, states = tracker.snapshot()
    pending = next(
        item for item in states if item.lifecycle is BOSLifecycle.PENDING
    )
    with pytest.raises(ValueError, match="only failed BOS"):
        replace(pending, failure_reason="")
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(pending, strength=-0.25)
    resolved_at = (
        pending.last_attempt_at or pending.pending_at
    ) + pd.Timedelta(minutes=1)
    confirmed = replace(
        pending,
        lifecycle=BOSLifecycle.CONFIRMED,
        resolved_at=resolved_at,
        strength=0.25,
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(confirmed, strength=-0.25)

    observer = CausalObserver()
    observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=resolved_at,
            bars=1,
            metrics={},
            structure_breaks=(pending,),
        ),
        newly_completed=True,
    )
    observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=resolved_at,
            bars=2,
            metrics={},
            structure_breaks=(confirmed,),
        ),
        newly_completed=True,
    )
    events = {
        item.lifecycle: item
        for item in observer.memory.recent()
        if item.entity_id == confirmed.bos_id
    }
    assert events[BOSLifecycle.CONFIRMED.value].ended_at == resolved_at
    assert (
        events[BOSLifecycle.CONFIRMED.value].transition_reason
        == BOS_CONFIRMATION_REASON
    )
    assert confirmed.bos_id not in observer.memory._latest_by_entity


def test_equal_pool_and_zone_lifecycles_are_incremental_and_frozen() -> None:
    tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(
            tick_size=0.25,
            cluster_tolerance_atr=0.10,
        ),
    )
    first = _swing(
        "high-1",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=0,
    )
    second = _swing(
        "high-2",
        side=SwingSide.HIGH,
        price=100.25,
        confirmed_index=1,
    )
    candle0 = _candle(
        0,
        open_=99.5,
        high=101.0,
        low=99.0,
        close=99.75,
    )
    candle1 = _candle(
        1,
        open_=99.75,
        high=101.0,
        low=99.0,
        close=99.75,
    )
    tracker.on_candle(candle0, (first,))
    zones, pools, _ = tracker.snapshot()
    assert zones[0].lifecycle is SupportResistanceLifecycle.ACTIVE
    assert not pools

    tracker.on_candle(candle1, (first, second))
    zones, pools, inventory = tracker.snapshot()
    zone = zones[0]
    pool = pools[0]
    frozen_bounds = (zone.lower_bound, zone.upper_bound)
    assert zone.lifecycle is SupportResistanceLifecycle.TESTED
    assert zone.touch_count == 2
    assert pool.lifecycle is LiquidityPoolLifecycle.FORMED
    assert pool.touch_count == 2
    assert any(item.kind == "equal_highs" for item in inventory)

    checkpoint = pickle.loads(pickle.dumps(tracker))
    sweep = _candle(
        2,
        open_=100.0,
        high=100.75,
        low=99.5,
        close=100.0,
    )
    tracker.on_candle(sweep, (first, second))
    checkpoint.on_candle(sweep, (first, second))
    assert to_primitive(tracker.snapshot()) == to_primitive(
        checkpoint.snapshot()
    )
    zones, pools, inventory = tracker.snapshot()
    assert (zones[0].lower_bound, zones[0].upper_bound) == frozen_bounds
    assert pools[0].lifecycle is LiquidityPoolLifecycle.SWEPT
    assert pools[0].close_outside_on_sweep is False
    assert next(
        item for item in inventory if item.kind == "equal_highs"
    ).lifecycle is LiquidityInventoryLifecycle.CONSUMED

    reclaim = _candle(
        3,
        open_=100.0,
        high=100.25,
        low=99.5,
        close=100.0,
    )
    tracker.on_candle(reclaim, (first, second))
    zones, pools, _ = tracker.snapshot()
    assert pools[0].lifecycle is LiquidityPoolLifecycle.REJECTED
    assert pools[0].resolution_reason == "close_returned_inside"
    assert (zones[0].lower_bound, zones[0].upper_bound) == frozen_bounds


def test_zone_break_reaccept_and_pool_acceptance_are_distinguishable() -> None:
    tracker = CausalLiquidityTracker(Timeframe.M1)
    first = _swing(
        "high-a",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=0,
    )
    second = _swing(
        "high-b",
        side=SwingSide.HIGH,
        price=100.25,
        confirmed_index=1,
    )
    tracker.on_candle(
        _candle(
            0,
            open_=99.5,
            high=101.0,
            low=99.0,
            close=99.75,
        ),
        (first,),
    )
    tracker.on_candle(
        _candle(
            1,
            open_=99.75,
            high=101.0,
            low=99.0,
            close=99.75,
        ),
        (first, second),
    )
    tracker.on_candle(
        _candle(
            2,
            open_=100.0,
            high=100.75,
            low=99.75,
            close=100.5,
        ),
        (first, second),
    )
    zones, pools, _ = tracker.snapshot()
    assert zones[0].lifecycle is SupportResistanceLifecycle.BROKEN
    assert pools[0].lifecycle is LiquidityPoolLifecycle.SWEPT
    assert pools[0].close_outside_on_sweep is True

    tracker.on_candle(
        _candle(
            3,
            open_=100.5,
            high=100.75,
            low=100.25,
            close=100.5,
        ),
        (first, second),
    )
    zones, pools, _ = tracker.snapshot()
    assert zones[0].lifecycle is SupportResistanceLifecycle.BROKEN
    assert pools[0].lifecycle is LiquidityPoolLifecycle.ACCEPTED
    assert pools[0].resolution_reason == "close_held_outside"

    tracker.on_candle(
        _candle(
            4,
            open_=100.5,
            high=100.5,
            low=99.75,
            close=100.0,
        ),
        (first, second),
    )
    zones, _, _ = tracker.snapshot()
    assert zones[0].lifecycle is SupportResistanceLifecycle.REACCEPTED
    assert zones[0].transition_reason == "close_reentered_frozen_zone"


def test_brain_and_risk_share_the_same_top_level_inventory_universe() -> None:
    observation = market_observation()
    visible = LiquidityInventoryItem(
        item_id="swing:h4-swing",
        timeframe=Timeframe.H4,
        side="above",
        kind="swing",
        price=110.0,
        lower_bound=110.0,
        upper_bound=110.0,
        formed_at=observation.asof - pd.Timedelta(hours=8),
        confirmed_at=observation.asof - pd.Timedelta(hours=4),
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("h4-swing",),
        age_bars=1,
        strength=0.8,
    )
    consumed = LiquidityInventoryItem(
        item_id="swing:m1-swing",
        timeframe=Timeframe.M1,
        side="below",
        kind="swing",
        price=95.0,
        lower_bound=95.0,
        upper_bound=95.0,
        formed_at=observation.asof - pd.Timedelta(minutes=20),
        confirmed_at=observation.asof - pd.Timedelta(minutes=10),
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        source_ids=("m1-swing",),
        age_bars=10,
        strength=0.5,
        consumed_at=observation.asof - pd.Timedelta(minutes=1),
        lifecycle_reason="close_beyond_swing",
    )
    observation = _with_authoritative_liquidity(
        observation,
        inventory=(visible, consumed),
        swings=(
            _inventory_swing_source(visible),
            _inventory_swing_source(consumed),
        ),
    )
    brain_ids = {item.level_id for item in _visible_levels(observation)}
    risk_ids = _visible_level_ids(observation)
    assert brain_ids == risk_ids == {"swing:h4-swing"}


def test_authoritative_empty_inventory_never_falls_back_to_legacy_levels() -> None:
    legacy = market_observation()
    assert _visible_levels(legacy)
    assert _visible_level_ids(legacy)

    authoritative = _with_authoritative_liquidity(legacy)
    assert _visible_levels(authoritative) == []
    assert _visible_level_ids(authoritative) == set()

    observer = CausalObserver()
    observer._prior = authoritative
    crossing_bar = _candle(
        0,
        open_=102.5,
        high=104.0,
        low=102.0,
        close=103.5,
    )
    observer._record_minute_events(
        ReaderUpdate(
            asof=crossing_bar.end,
            completed_1m=crossing_bar,
            newly_completed={},
            histories={},
            anomalies=(),
        ),
        authoritative.frames,
    )
    assert not any(
        event.kind
        in {
            EventKind.LIQUIDITY_SWEEP,
            EventKind.LIQUIDITY_CONSUMED,
        }
        for event in observer.memory.recent()
    )


def test_observation_rejects_duplicate_or_post_cutoff_inventory() -> None:
    observation = market_observation()
    item = LiquidityInventoryItem(
        item_id="swing:m1-source",
        timeframe=Timeframe.M1,
        side="above",
        kind="swing",
        price=101.0,
        lower_bound=101.0,
        upper_bound=101.0,
        formed_at=observation.asof - pd.Timedelta(minutes=2),
        confirmed_at=observation.asof - pd.Timedelta(minutes=1),
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("m1-source",),
        age_bars=1,
        strength=0.5,
    )
    with pytest.raises(ValueError, match="duplicate liquidity"):
        replace(
            observation,
            liquidity_inventory=(item, item),
            liquidity_inventory_authoritative=True,
        )

    source_cutoff = observation.asof - pd.Timedelta(hours=1)
    h1_frame = replace(
        observation.frame(Timeframe.H1),
        cutoff=source_cutoff,
    )
    late_item = replace(
        item,
        item_id="h1-post-cutoff",
        timeframe=Timeframe.H1,
        formed_at=source_cutoff - pd.Timedelta(minutes=1),
        confirmed_at=source_cutoff + pd.Timedelta(minutes=1),
        source_ids=("h1-source",),
    )
    with pytest.raises(ValueError, match="postdates its source frame cutoff"):
        replace(
            observation,
            frames={
                **observation.frames,
                Timeframe.H1: h1_frame,
            },
            liquidity_inventory=(late_item,),
            liquidity_inventory_authoritative=True,
        )

    source = _inventory_swing_source(item)
    valid = _with_authoritative_liquidity(
        observation,
        inventory=(item,),
        swings=(source,),
    )
    with pytest.raises(
        ValueError,
        match="swing state and liquidity inventory disagree",
    ):
        replace(
            valid,
            liquidity_inventory=(replace(item, strength=0.7),),
        )
    with pytest.raises(
        ValueError,
        match="lacks one frame source",
    ):
        _with_authoritative_liquidity(
            observation,
            inventory=(item,),
            swings=(),
        )

    projected_sweep = replace(
        item,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=observation.asof,
        lifecycle_reason="swing_swept",
    )
    projected = _with_authoritative_liquidity(
        observation,
        inventory=(projected_sweep,),
        swings=(source,),
    )
    assert (
        projected.liquidity_inventory[0].lifecycle_reason
        == "swing_swept"
    )
    lagging_frames = dict(projected.frames)
    lagging_frames[Timeframe.M1] = replace(
        lagging_frames[Timeframe.M1],
        cutoff=observation.asof - pd.Timedelta(minutes=1),
    )
    with pytest.raises(
        ValueError,
        match="completed 1m cutoff",
    ):
        replace(projected, frames=lagging_frames)


def test_frame_rejects_legacy_liquidity_from_another_timeframe() -> None:
    observation = market_observation()
    h1_level = observation.frame(Timeframe.H1).liquidity[0]
    with pytest.raises(
        ValueError,
        match="legacy liquidity from another timeframe",
    ):
        replace(
            observation.frame(Timeframe.H4),
            liquidity=(h1_level,),
        )


def test_generic_plan_uses_all_inventory_for_draws_but_only_swings_for_stop() -> None:
    observation = market_observation()
    formed_at = observation.asof - pd.Timedelta(hours=2)
    confirmed_at = observation.asof - pd.Timedelta(hours=1)

    def inventory(
        item_id: str,
        *,
        side: str,
        kind: str,
        price: float,
    ) -> LiquidityInventoryItem:
        return LiquidityInventoryItem(
            item_id=item_id,
            timeframe=Timeframe.H1,
            side=side,
            kind=kind,
            price=price,
            lower_bound=price,
            upper_bound=price,
            formed_at=formed_at,
            confirmed_at=confirmed_at,
            lifecycle=LiquidityInventoryLifecycle.VISIBLE,
            source_ids=(
                (f"{item_id}-a", f"{item_id}-b")
                if kind in {"equal_highs", "equal_lows"}
                else (item_id.removeprefix("swing:"),)
            ),
            age_bars=1,
            strength=0.8,
        )

    equal_highs = inventory(
        "pool:equal-highs-draw",
        side="above",
        kind="equal_highs",
        price=103.0,
    )
    equal_lows = inventory(
        "pool:near-equal-lows",
        side="below",
        kind="equal_lows",
        price=99.0,
    )
    swing_low = inventory(
        "swing:confirmed-swing-low",
        side="below",
        kind="swing",
        price=98.0,
    )

    def pool(item: LiquidityInventoryItem) -> LiquidityPoolState:
        pool_id = item.item_id.removeprefix("pool:")
        return LiquidityPoolState(
            pool_id=pool_id,
            timeframe=item.timeframe,
            side=item.side,
            lower_bound=item.lower_bound,
            upper_bound=item.upper_bound,
            midpoint=item.price,
            formed_at=item.formed_at,
            confirmed_at=item.confirmed_at,
            lifecycle=LiquidityPoolLifecycle.FORMED,
            member_swing_ids=item.source_ids,
            touch_times=(
                item.confirmed_at - pd.Timedelta(minutes=1),
                item.confirmed_at,
            ),
            age_bars=item.age_bars,
            strength=item.strength,
            total_touch_count=2,
        )

    observation = _with_authoritative_liquidity(
        observation,
        inventory=(equal_highs, equal_lows, swing_low),
        pools=(pool(equal_highs), pool(equal_lows)),
        swings=(_inventory_swing_source(swing_low),),
    )
    plan = _plan(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        observation,
        BrainConfig(),
    )
    assert plan is not None
    assert plan.targets[0].level_id == "pool:equal-highs-draw"
    assert (
        plan.invalidation.source_level_id
        == "swing:confirmed-swing-low"
    )


def test_observation_rejects_pool_inventory_source_drift() -> None:
    observation = market_observation()
    formed_at = observation.asof - pd.Timedelta(minutes=4)
    confirmed_at = observation.asof - pd.Timedelta(minutes=2)
    pool = LiquidityPoolState(
        pool_id="drift-check",
        timeframe=Timeframe.M1,
        side="above",
        lower_bound=100.75,
        upper_bound=101.25,
        midpoint=101.0,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        lifecycle=LiquidityPoolLifecycle.FORMED,
        member_swing_ids=("drift-a", "drift-b"),
        touch_times=(formed_at, confirmed_at),
        age_bars=2,
        strength=0.7,
        total_touch_count=2,
    )
    item = LiquidityInventoryItem(
        item_id="pool:drift-check",
        timeframe=Timeframe.M1,
        side="above",
        kind="equal_highs",
        price=pool.upper_bound,
        lower_bound=pool.lower_bound,
        upper_bound=pool.upper_bound,
        formed_at=pool.formed_at,
        confirmed_at=pool.confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=pool.member_swing_ids,
        age_bars=pool.age_bars,
        strength=pool.strength,
    )
    valid = _with_authoritative_liquidity(
        observation,
        inventory=(item,),
        pools=(pool,),
    )
    with pytest.raises(ValueError, match="drifted from its frozen frame"):
        replace(
            valid,
            liquidity_pool_states=(
                replace(
                    pool,
                    lower_bound=100.5,
                    midpoint=100.875,
                ),
            ),
        )
    swept_at = observation.asof - pd.Timedelta(minutes=1)
    swept_pool = replace(
        pool,
        lifecycle=LiquidityPoolLifecycle.SWEPT,
        swept_at=swept_at,
        sweep_extreme=101.5,
        close_outside_on_sweep=False,
    )
    terminal_frames = dict(valid.frames)
    terminal_frames[Timeframe.M1] = replace(
        terminal_frames[Timeframe.M1],
        liquidity_pools=(swept_pool,),
    )
    with pytest.raises(ValueError, match="frozen formation view"):
        replace(valid, frames=terminal_frames)

    wrong_reason = replace(
        item,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=swept_at,
        lifecycle_reason="swing_swept",
    )
    with pytest.raises(
        ValueError,
        match="pool state and liquidity inventory disagree",
    ):
        _with_authoritative_liquidity(
            observation,
            inventory=(wrong_reason,),
            pools=(swept_pool,),
        )


def test_liquidity_protocol_file_is_the_executable_identity() -> None:
    config = LiquidityConfig.from_file(
        STRUCTURE_PROTOCOL,
        tick_size=0.25,
        atr_period=14,
    )
    source = Path(STRUCTURE_PROTOCOL)
    assert config.protocol_hash == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert config.cluster_tolerance_atr == 0.1
    assert config.cluster_tolerance_ticks == 1


def test_late_attach_rebuilds_coarse_pool_clocks_from_completed_m1() -> None:
    tracker = CausalLiquidityTracker(
        Timeframe.H1,
        LiquidityConfig(retained_touches=8),
    )

    def h1_candle(
        start: pd.Timestamp,
        *,
        high: float,
        low: float,
        close: float,
    ) -> Candle:
        return Candle(
            timeframe=Timeframe.H1,
            start=start,
            end=start + pd.Timedelta(hours=1),
            open=close,
            high=high,
            low=low,
            close=close,
            volume=100.0,
            symbol="NQH5",
            instrument_id=1,
            observed_minutes=60,
            expected_minutes=60,
            complete=True,
            real_minutes=60,
            synthetic_minutes=0,
        )

    def h1_swing(
        identity: str,
        confirmed_at: pd.Timestamp,
    ) -> SwingPoint:
        return SwingPoint(
            swing_id=identity,
            timeframe=Timeframe.H1,
            symbol="NQH5",
            instrument_id=1,
            side=SwingSide.HIGH,
            price=100.0,
            price_ticks=400,
            pivot_start=confirmed_at - pd.Timedelta(hours=2),
            pivot_end=confirmed_at - pd.Timedelta(hours=1),
            observed_at=confirmed_at,
            confirmed_at=confirmed_at,
            lifecycle=SwingLifecycle.CONFIRMED,
            relation=SwingRelation.HH,
            delta_ticks=1,
            delta_points=0.25,
            magnitude_atr=0.25,
        )

    first_bar = h1_candle(
        BASE - pd.Timedelta(hours=2),
        high=100.2,
        low=99.5,
        close=99.75,
    )
    second_bar = h1_candle(
        BASE - pd.Timedelta(hours=1),
        high=100.2,
        low=99.5,
        close=99.75,
    )
    first = h1_swing("coarse-a", first_bar.end)
    second = h1_swing("coarse-b", second_bar.end)
    tracker.on_candle(first_bar, (first,))
    tracker.on_candle(second_bar, (first, second))
    before_native_transition = pickle.loads(pickle.dumps(tracker))
    native_sweep_bar = h1_candle(
        BASE,
        high=101.0,
        low=99.5,
        close=100.0,
    )
    native_resolution_bar = h1_candle(
        BASE + pd.Timedelta(hours=1),
        high=100.2,
        low=99.5,
        close=100.0,
    )
    tracker.on_candle(native_sweep_bar, (first, second))
    tracker.on_candle(native_resolution_bar, (first, second))
    _, native_pools, native_inventory = tracker.snapshot()
    native_pool_inventory = tuple(
        item
        for item in native_inventory
        if item.kind in {"equal_highs", "equal_lows"}
    )
    native_pool = native_pools[0]
    assert native_pool.resolved_at == BASE + pd.Timedelta(hours=2)
    native_for_short_prefix = pickle.loads(pickle.dumps(tracker))

    pre_sweep = _candle(
        0,
        open_=99.75,
        high=100.1,
        low=99.5,
        close=99.75,
    )
    exact_sweep = _candle(
        1,
        open_=100.0,
        high=100.75,
        low=99.75,
        close=100.0,
    )
    exact_resolution = _candle(
        2,
        open_=100.0,
        high=100.2,
        low=99.5,
        close=100.0,
    )
    tail = Candle(
        timeframe=Timeframe.M1,
        start=BASE + pd.Timedelta(hours=2) - pd.Timedelta(minutes=1),
        end=BASE + pd.Timedelta(hours=2),
        open=100.0,
        high=100.1,
        low=99.9,
        close=100.0,
        volume=100.0,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=1,
        synthetic_minutes=0,
    )
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
        )
    )
    observer._liquidity_trackers[Timeframe.H1] = tracker
    atr_warmup = tuple(
        _candle(
            index,
            open_=99.75,
            high=100.1,
            low=99.5,
            close=99.75,
        )
        for index in range(-14, 0)
    )
    update = ReaderUpdate(
        asof=tail.end,
        completed_1m=tail,
        newly_completed={},
        histories={
            Timeframe.M1: (
                *atr_warmup,
                pre_sweep,
                exact_sweep,
                exact_resolution,
                tail,
            )
        },
        anomalies=(),
    )
    projected = observer._project_inventory(
        update,
        market_observation(asof=tail.end).frames,
        native_pool_inventory,
    )
    projected_pool_item = next(
        item for item in projected if item.kind == "equal_highs"
    )
    assert projected_pool_item.consumed_at == exact_sweep.end
    rebuilt_pool = next(
        item
        for item in tracker.snapshot()[1]
        if item.pool_id == native_pool.pool_id
    )
    assert rebuilt_pool.lifecycle is LiquidityPoolLifecycle.REJECTED
    assert rebuilt_pool.swept_at == exact_sweep.end
    assert rebuilt_pool.resolved_at == exact_resolution.end
    pool_events = [
        item
        for item in observer.memory.recent()
        if item.entity_id == projected_pool_item.item_id
    ]
    assert [item.lifecycle for item in pool_events] == [
        LiquidityPoolLifecycle.SWEPT.value,
        LiquidityPoolLifecycle.REJECTED.value,
    ]
    missing_predecessor = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
        )
    )
    missing_predecessor._liquidity_trackers[Timeframe.H1] = (
        native_for_short_prefix
    )
    with pytest.raises(
        ValueError,
        match="lacks the registered ATR warmup",
    ):
        missing_predecessor._project_inventory(
            replace(
                update,
                histories={
                    Timeframe.M1: (
                        *atr_warmup[-12:],
                        pre_sweep,
                        exact_sweep,
                        exact_resolution,
                        tail,
                    )
                },
            ),
            market_observation(asof=tail.end).frames,
            native_pool_inventory,
        )
    synthetic_cannot_warm = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
        )
    )
    synthetic_cannot_warm._liquidity_trackers[Timeframe.H1] = (
        pickle.loads(pickle.dumps(native_for_short_prefix))
    )
    with pytest.raises(
        ValueError,
        match="lacks the registered ATR warmup",
    ):
        synthetic_cannot_warm._project_inventory(
            replace(
                update,
                histories={
                    Timeframe.M1: (
                        *(
                            _candle(
                                index,
                                open_=99.75,
                                high=99.75,
                                low=99.75,
                                close=99.75,
                                synthetic=True,
                            )
                            for index in range(-13, 0)
                        ),
                        pre_sweep,
                        exact_sweep,
                        exact_resolution,
                        tail,
                    )
                },
            ),
            market_observation(asof=tail.end).frames,
            native_pool_inventory,
        )
    too_late = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
        )
    )
    too_late._liquidity_trackers[Timeframe.H1] = tracker
    with pytest.raises(
        ValueError,
        match="prefix starts after.*inventory draw was confirmed",
    ):
        too_late._project_inventory(
            replace(
                update,
                histories={
                    Timeframe.M1: (
                        exact_sweep,
                        exact_resolution,
                        tail,
                    )
                },
            ),
            market_observation(asof=tail.end).frames,
            native_pool_inventory,
        )

    swallowed_touches = tuple(
        h1_swing(
            f"coarse-post-resolution-touch-{index}",
            native_sweep_bar.end,
        )
        for index in range(16)
    )
    before_native_transition.on_candle(
        native_sweep_bar,
        (first, second, *swallowed_touches),
    )
    before_native_transition.on_candle(
        native_resolution_bar,
        (first, second, *swallowed_touches),
    )
    assert swallowed_touches[0].swing_id not in (
        before_native_transition.snapshot()[0][0].member_swing_ids
    )
    divergent_inventory = tuple(
        item
        for item in before_native_transition.snapshot()[2]
        if item.kind in {"equal_highs", "equal_lows"}
    )
    divergent = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
        )
    )
    divergent._liquidity_trackers[Timeframe.H1] = (
        before_native_transition
    )
    with pytest.raises(
        ValueError,
        match="swallowed a post-resolution.*touch",
    ):
        divergent._project_inventory(
            update,
            market_observation(asof=tail.end).frames,
            divergent_inventory,
        )


def test_cold_attach_rebuilds_first_swing_crossing_or_fails_closed() -> None:
    item = LiquidityInventoryItem(
        item_id="swing:cold-visible",
        timeframe=Timeframe.H1,
        side="above",
        kind="swing",
        price=100.0,
        lower_bound=100.0,
        upper_bound=100.0,
        formed_at=BASE - pd.Timedelta(minutes=1),
        confirmed_at=BASE,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("cold-visible",),
        age_bars=1,
        strength=0.5,
    )
    warmup = tuple(
        _candle(
            index,
            open_=99.5,
            high=99.75,
            low=99.25,
            close=99.5,
        )
        for index in range(-15, 1)
    )
    crossing = _candle(
        1,
        open_=99.75,
        high=100.75,
        low=99.5,
        close=100.0,
    )
    tail = _candle(
        2,
        open_=99.75,
        high=99.9,
        low=99.5,
        close=99.75,
    )
    update = ReaderUpdate(
        asof=tail.end,
        completed_1m=tail,
        newly_completed={},
        histories={
            Timeframe.M1: (*warmup, crossing, tail),
        },
        anomalies=(),
    )
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
        )
    )
    projected = observer._project_inventory(
        update,
        market_observation(asof=tail.end).frames,
        (item,),
    )
    rebuilt = projected[0]
    assert (
        rebuilt.lifecycle,
        rebuilt.consumed_at,
        rebuilt.lifecycle_reason,
    ) == (
        LiquidityInventoryLifecycle.CONSUMED,
        crossing.end,
        "swing_swept",
    )
    crossing_event = next(
        event
        for event in observer.memory.recent()
        if item.item_id in event.source_ids
    )
    assert crossing_event.kind is EventKind.LIQUIDITY_SWEEP
    assert crossing_event.observed_at == crossing.end

    too_late = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
        )
    )
    with pytest.raises(
        ValueError,
        match="prefix starts after.*inventory draw was confirmed",
    ):
        too_late._project_inventory(
            replace(
                update,
                histories={Timeframe.M1: (tail,)},
            ),
            market_observation(asof=tail.end).frames,
            (item,),
        )


def test_unregistered_range_boundary_projection_fails_closed() -> None:
    bar = _candle(
        0,
        open_=99.5,
        high=100.5,
        low=99.25,
        close=100.0,
    )
    update = ReaderUpdate(
        asof=bar.end,
        completed_1m=bar,
        newly_completed={},
        histories={Timeframe.M1: (bar,)},
        anomalies=(),
    )
    item = LiquidityInventoryItem(
        item_id="range:disabled",
        timeframe=Timeframe.H1,
        side="above",
        kind="range_boundary",
        price=100.0,
        lower_bound=100.0,
        upper_bound=100.0,
        formed_at=bar.start - pd.Timedelta(minutes=1),
        confirmed_at=bar.start,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("range:disabled",),
        age_bars=0,
        strength=0.5,
    )
    for cold_attach in (True, False):
        observer = CausalObserver()
        if not cold_attach:
            observer._prior = market_observation(asof=bar.start)
        with pytest.raises(
            ValueError,
            match="kind is not enabled for exact 1m lifecycle projection",
        ):
            observer._project_inventory(
                update,
                market_observation(asof=bar.end).frames,
                (item,),
            )


def test_retention_fails_closed_without_evicting_nonterminal_state() -> None:
    preflight = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(),
    )
    preflight_swing = _swing(
        "preflight-clock",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=0,
    )
    with pytest.raises(
        ValueError,
        match="admitted at its confirmation clock",
    ):
        preflight.on_candle(
            _candle(
                1,
                open_=100.0,
                high=100.2,
                low=99.8,
                close=100.0,
            ),
            (preflight_swing,),
        )
    assert preflight.last_end is None
    assert preflight._bar_index == -1
    assert preflight.snapshot() == ((), (), ())
    preflight.on_candle(
        _candle(
            0,
            open_=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
        ),
        (preflight_swing,),
    )

    tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(
            retained_zones=9,
            retained_pools=8,
            retained_touches=8,
        ),
    )
    all_swings: list[SwingPoint] = []
    for pool_index in range(8):
        price = 200.0 - 10.0 * pool_index
        for member_index in range(2):
            candle_index = 2 * pool_index + member_index
            swing = _swing(
                f"capacity-{pool_index}-{member_index}",
                side=SwingSide.HIGH,
                price=price,
                confirmed_index=candle_index,
            )
            all_swings.append(swing)
            tracker.on_candle(
                _candle(
                    candle_index,
                    open_=price,
                    high=price + 0.2,
                    low=price - 0.2,
                    close=price,
                ),
                tuple(all_swings),
            )
    zones, pools, _ = tracker.snapshot()
    assert len(zones) == 8
    assert len(pools) == 8
    assert all(
        item.lifecycle is LiquidityPoolLifecycle.FORMED
        for item in pools
    )

    protected = pools[0]
    ninth_first = _swing(
        "capacity-8-0",
        side=SwingSide.HIGH,
        price=120.0,
        confirmed_index=16,
    )
    all_swings.append(ninth_first)
    tracker.on_candle(
        _candle(
            16,
            open_=120.0,
            high=120.2,
            low=119.8,
            close=120.0,
        ),
        tuple(all_swings),
    )
    ninth_second = _swing(
        "capacity-8-1",
        side=SwingSide.HIGH,
        price=120.0,
        confirmed_index=17,
    )
    all_swings.append(ninth_second)
    before_pool_failure = to_primitive(tracker.snapshot())
    before_pool_clock = tracker.last_end
    before_pool_index = tracker._bar_index
    with pytest.raises(
        ValueError,
        match="pool retention is exhausted",
    ):
        tracker.on_candle(
            _candle(
                17,
                open_=120.0,
                high=max(item.upper_bound for item in pools) + 0.5,
                low=119.8,
                close=120.0,
            ),
            tuple(all_swings),
        )
    assert to_primitive(tracker.snapshot()) == before_pool_failure
    assert tracker.last_end == before_pool_clock
    assert tracker._bar_index == before_pool_index
    retained = {
        item.pool_id: item for item in tracker.snapshot()[1]
    }
    assert len(retained) == 8
    assert protected.pool_id in retained
    assert (
        retained[protected.pool_id].lifecycle
        is LiquidityPoolLifecycle.FORMED
    )

    zone_tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(
            retained_zones=8,
            retained_pools=8,
            retained_touches=8,
        ),
    )
    zone_swings: list[SwingPoint] = []
    for index in range(8):
        price = 200.0 - 10.0 * index
        swing = _swing(
            f"zone-capacity-{index}",
            side=SwingSide.HIGH,
            price=price,
            confirmed_index=index,
        )
        zone_swings.append(swing)
        zone_tracker.on_candle(
            _candle(
                index,
                open_=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
            ),
            tuple(zone_swings),
        )
    overflow = _swing(
        "zone-capacity-overflow",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=8,
    )
    zone_swings.append(overflow)
    before_zone_failure = to_primitive(zone_tracker.snapshot())
    before_zone_clock = zone_tracker.last_end
    before_zone_index = zone_tracker._bar_index
    with pytest.raises(
        ValueError,
        match="zone retention is exhausted",
    ):
        zone_tracker.on_candle(
            _candle(
                8,
                open_=100.0,
                high=100.2,
                low=99.8,
                close=100.0,
            ),
            tuple(zone_swings),
        )
    assert to_primitive(zone_tracker.snapshot()) == before_zone_failure
    assert zone_tracker.last_end == before_zone_clock
    assert zone_tracker._bar_index == before_zone_index
    assert len(zone_tracker.snapshot()[0]) == 8


def test_pool_terminal_exposure_avoids_same_bar_capacity_deadlock() -> None:
    tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(
            retained_zones=16,
            retained_pools=8,
            retained_touches=8,
        ),
    )
    swings: list[SwingPoint] = []
    for pool_index in range(8):
        price = 200.0 - 10.0 * pool_index
        for member_index in range(2):
            candle_index = 2 * pool_index + member_index
            swing = _swing(
                f"pool-exposure-{pool_index}-{member_index}",
                side=SwingSide.HIGH,
                price=price,
                confirmed_index=candle_index,
            )
            swings.append(swing)
            tracker.on_candle(
                _candle(
                    candle_index,
                    open_=price,
                    high=price + 0.2,
                    low=price - 0.2,
                    close=price,
                ),
                tuple(swings),
            )
    ninth_first = _swing(
        "pool-exposure-8-0",
        side=SwingSide.HIGH,
        price=120.0,
        confirmed_index=16,
    )
    tenth_first = _swing(
        "pool-exposure-9-0",
        side=SwingSide.LOW,
        price=110.0,
        confirmed_index=16,
    )
    swings.extend((ninth_first, tenth_first))
    tracker.on_candle(
        _candle(
            16,
            open_=115.0,
            high=120.2,
            low=109.8,
            close=115.0,
        ),
        tuple(swings),
    )

    ninth_second = _swing(
        "pool-exposure-8-1",
        side=SwingSide.HIGH,
        price=120.0,
        confirmed_index=17,
    )
    tenth_second = _swing(
        "pool-exposure-9-1",
        side=SwingSide.LOW,
        price=110.0,
        confirmed_index=17,
    )
    second_touches = (ninth_second, tenth_second)
    protected = tracker.snapshot()[1][:2]

    bounded_tracker = pickle.loads(pickle.dumps(tracker))
    bounded_tracker.project_pool_sweep(
        protected[0].pool_id,
        observed_at=BASE + pd.Timedelta(minutes=17),
        sweep_extreme=protected[0].upper_bound + 0.5,
        close_outside_on_sweep=False,
    )
    bounded_before = to_primitive(bounded_tracker.snapshot())
    bounded_clock = bounded_tracker.last_end
    bounded_index = bounded_tracker._bar_index
    bounded_known = set(bounded_tracker._known_swing_ids)
    with pytest.raises(
        ValueError,
        match="pool retention is exhausted",
    ):
        bounded_tracker.on_candle(
            _candle(
                17,
                open_=115.0,
                high=120.2,
                low=109.8,
                close=115.0,
            ),
            (*swings, *second_touches),
        )
    assert to_primitive(bounded_tracker.snapshot()) == bounded_before
    assert bounded_tracker.last_end == bounded_clock
    assert bounded_tracker._bar_index == bounded_index
    assert bounded_tracker._known_swing_ids == bounded_known

    for item in protected:
        tracker.project_pool_sweep(
            item.pool_id,
            observed_at=BASE + pd.Timedelta(minutes=17),
            sweep_extreme=item.upper_bound + 0.5,
            close_outside_on_sweep=False,
        )
    swings.extend(second_touches)
    tracker.on_candle(
        _candle(
            17,
            open_=115.0,
            high=120.2,
            low=109.8,
            close=115.0,
        ),
        tuple(swings),
    )
    pools = tracker.snapshot()[1]
    assert len(pools) == 10
    assert sum(
        item.lifecycle is LiquidityPoolLifecycle.REJECTED
        and item.resolved_at == BASE + pd.Timedelta(minutes=18)
        for item in pools
    ) == 2

    tracker.on_candle(
        _candle(
            18,
            open_=115.0,
            high=120.2,
            low=109.8,
            close=115.0,
            synthetic=True,
        ),
        tuple(swings),
    )
    assert len(tracker.snapshot()[1]) == 10

    tracker.on_candle(
        _candle(
            19,
            open_=115.0,
            high=120.2,
            low=109.8,
            close=115.0,
        ),
        tuple(swings),
    )
    assert len(tracker.snapshot()[1]) == 8


def test_observer_pool_event_projection_and_timeline_wiring() -> None:
    observer = CausalObserver(ObserverConfig(memory_events=2))
    formed_at = BASE + pd.Timedelta(minutes=1)
    confirmed_at = BASE + pd.Timedelta(minutes=2)
    swept_at = BASE + pd.Timedelta(minutes=3)
    resolved_at = BASE + pd.Timedelta(minutes=4)
    formed = LiquidityPoolState(
        pool_id="observer-terminal-pool",
        timeframe=Timeframe.M1,
        side="above",
        lower_bound=100.75,
        upper_bound=101.25,
        midpoint=101.0,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        lifecycle=LiquidityPoolLifecycle.FORMED,
        member_swing_ids=("pool-a", "pool-b"),
        touch_times=(formed_at, confirmed_at),
        age_bars=1,
        strength=0.7,
        total_touch_count=2,
    )
    inventory = LiquidityInventoryItem(
        item_id=f"pool:{formed.pool_id}",
        timeframe=Timeframe.M1,
        side=formed.side,
        kind="equal_highs",
        price=formed.upper_bound,
        lower_bound=formed.lower_bound,
        upper_bound=formed.upper_bound,
        formed_at=formed.formed_at,
        confirmed_at=formed.confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=formed.member_swing_ids,
        age_bars=formed.age_bars,
        strength=formed.strength,
    )
    formation_frame = FrameObservation(
        timeframe=Timeframe.M1,
        cutoff=confirmed_at,
        bars=5,
        metrics={},
        liquidity_pools=(formed,),
    )
    observer._record_frame_events(
        formation_frame,
        newly_completed=True,
    )
    sweep_candle = _candle(
        2,
        open_=101.0,
        high=102.0,
        low=100.5,
        close=101.0,
    )
    observer._append_projected_pool_sweep_events(
        inventory,
        sweep_candle,
        atr=1.0,
    )
    resolution_candle = _candle(
        3,
        open_=101.0,
        high=101.1,
        low=100.5,
        close=101.0,
    )
    observer._append_projected_pool_resolution_event(
        inventory,
        resolution_candle,
    )
    rejected = replace(
        formed,
        lifecycle=LiquidityPoolLifecycle.REJECTED,
        swept_at=swept_at,
        sweep_extreme=102.0,
        close_outside_on_sweep=False,
        resolved_at=resolved_at,
        resolution_reason="close_returned_inside",
    )
    terminal_frame = replace(
        formation_frame,
        cutoff=resolved_at,
    )
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=resolved_at,
            bars=1,
            metrics={},
            liquidity_pools=(
                (formed,)
                if timeframe is Timeframe.M1
                else ()
            ),
        )
        for timeframe in Timeframe
    }
    observer.memory.sync_retained_entity_timelines(
        observer._retained_timeline_keys(frames, (rejected,)),
        asof=resolved_at,
    )
    terminal_key = f"pool:{formed.pool_id}"
    base = _with_authoritative_liquidity(
        market_observation(asof=resolved_at, price=101.0),
        inventory=(
            replace(
                inventory,
                lifecycle=LiquidityInventoryLifecycle.CONSUMED,
                consumed_at=swept_at,
                lifecycle_reason="pool_swept",
            ),
        ),
        pools=(rejected,),
    )
    terminal_observation = replace(
        base,
        recent_events=observer.memory.recent(),
        event_durations_minutes=observer.memory.durations(
            resolved_at
        ),
        event_ages_minutes=observer.memory.ages(resolved_at),
        retained_entity_timelines=(
            observer.memory.entity_timelines()
        ),
    )
    assert tuple(
        event.lifecycle
        for event in terminal_observation.retained_entity_timelines[
            terminal_key
        ]
    ) == ("formed", "swept", "rejected")

    synthetic_at = resolved_at + pd.Timedelta(minutes=1)
    observer.memory.sync_retained_entity_timelines(
        observer._retained_timeline_keys(frames, (rejected,)),
        asof=synthetic_at,
    )
    assert terminal_key in observer.memory.entity_timelines()

    compacted_frames = dict(frames)
    compacted_frames[Timeframe.M1] = replace(
        terminal_frame,
        cutoff=synthetic_at + pd.Timedelta(minutes=1),
        liquidity_pools=(),
    )
    observer.memory.sync_retained_entity_timelines(
        observer._retained_timeline_keys(compacted_frames, ()),
        asof=synthetic_at + pd.Timedelta(minutes=1),
    )
    assert terminal_key not in observer.memory.entity_timelines()


def test_zone_visibility_retires_stale_evidence_and_pins_unresolved_pool() -> None:
    tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(
            retained_zones=8,
            retained_pools=16,
            retained_touches=8,
        ),
    )
    swings: list[SwingPoint] = []
    first_retired_zone_id: str | None = None
    overflow_observer = CausalObserver()
    first_retired_event_id: str | None = None
    for index in range(10):
        price = 300.0 - 10.0 * index
        swing = _swing(
            f"rolling-zone-{index}",
            side=SwingSide.HIGH,
            price=price,
            confirmed_index=index,
        )
        swings.append(swing)
        tracker.on_candle(
            _candle(
                index,
                open_=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
            ),
            tuple(swings[-8:]),
        )
        zones = tracker.snapshot()[0]
        if index == 8:
            retired = [
                item
                for item in zones
                if item.lifecycle
                is SupportResistanceLifecycle.RETIRED
            ]
            assert len(zones) == 9
            assert len(retired) == 1
            assert retired[0].member_swing_ids == (
                "rolling-zone-0",
            )
            assert retired[0].retired_at == (
                BASE + pd.Timedelta(minutes=9)
            )
            assert (
                retired[0].transition_reason
                == SUPPORT_RESISTANCE_RETIREMENT_REASON
            )
            with pytest.raises(
                ValueError,
                match="retirement must follow",
            ):
                replace(
                    retired[0],
                    retired_at=retired[0].confirmed_at,
                )
            with pytest.raises(
                ValueError,
                match="retired zone lifecycle",
            ):
                replace(
                    retired[0],
                    transition_reason="capacity_eviction",
            )
            first_retired_zone_id = retired[0].zone_id
            overflow_observer._record_frame_events(
                FrameObservation(
                    timeframe=Timeframe.M1,
                    cutoff=BASE + pd.Timedelta(minutes=9),
                    bars=9,
                    metrics={},
                    support_resistance=zones,
                ),
                newly_completed=True,
            )
            first_retired_event_id = next(
                item.event_id
                for item in overflow_observer.memory.recent()
                if (
                    item.entity_id == retired[0].zone_id
                    and item.lifecycle
                    == SupportResistanceLifecycle.RETIRED.value
                )
            )
        elif index == 9:
            assert len(zones) == 9
            assert first_retired_zone_id is not None
            assert all(
                item.zone_id != first_retired_zone_id
                for item in zones
            )
            retired = [
                item
                for item in zones
                if item.lifecycle
                is SupportResistanceLifecycle.RETIRED
            ]
            assert len(retired) == 1
            assert retired[0].member_swing_ids == (
                "rolling-zone-1",
            )
    tracker.on_candle(
        _candle(
            10,
            open_=210.0,
            high=210.2,
            low=209.8,
            close=210.0,
        ),
        tuple(swings[-8:]),
    )
    assert len(tracker.snapshot()[0]) == 8
    assert all(
        item.lifecycle is not SupportResistanceLifecycle.RETIRED
        for item in tracker.snapshot()[0]
    )
    assert first_retired_event_id is not None
    assert any(
        item.event_id == first_retired_event_id
        for item in overflow_observer.memory.recent()
    )

    pool_tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(
            retained_zones=8,
            retained_pools=8,
            retained_touches=8,
        ),
    )
    first = _swing(
        "pinned-zone-1",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=0,
    )
    second = _swing(
        "pinned-zone-2",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=1,
    )
    pool_tracker.on_candle(
        _candle(
            0,
            open_=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
        ),
        (first,),
    )
    pool_tracker.on_candle(
        _candle(
            1,
            open_=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
        ),
        (second,),
    )
    pool_tracker.on_candle(
        _candle(
            2,
            open_=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
        ),
        (),
    )
    zones, pools, _ = pool_tracker.snapshot()
    assert zones[0].lifecycle is SupportResistanceLifecycle.TESTED
    assert pools[0].lifecycle is LiquidityPoolLifecycle.FORMED

    pool_tracker.on_candle(
        _candle(
            3,
            open_=100.0,
            high=100.75,
            low=99.8,
            close=100.0,
        ),
        (),
    )
    zones, pools, _ = pool_tracker.snapshot()
    assert zones[0].lifecycle is SupportResistanceLifecycle.TESTED
    assert pools[0].lifecycle is LiquidityPoolLifecycle.SWEPT

    pool_tracker.on_candle(
        _candle(
            4,
            open_=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
        ),
        (),
    )
    zones, pools, _ = pool_tracker.snapshot()
    assert pools[0].lifecycle is LiquidityPoolLifecycle.REJECTED
    assert zones[0].lifecycle is SupportResistanceLifecycle.RETIRED
    assert zones[0].retired_at == BASE + pd.Timedelta(minutes=5)
    assert (
        zones[0].transition_reason
        == SUPPORT_RESISTANCE_RETIREMENT_REASON
    )

    break_tracker = CausalLiquidityTracker(Timeframe.M1)
    break_source = _swing(
        "retire-after-break",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=0,
    )
    break_tracker.on_candle(
        _candle(
            0,
            open_=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
        ),
        (break_source,),
    )
    break_tracker.on_candle(
        _candle(
            1,
            open_=100.5,
            high=101.0,
            low=100.5,
            close=101.0,
        ),
        (),
    )
    broken = break_tracker.snapshot()[0][0]
    assert broken.lifecycle is SupportResistanceLifecycle.BROKEN
    assert broken.broken_at == BASE + pd.Timedelta(minutes=2)
    assert broken.retired_at is None

    break_tracker.on_candle(
        _candle(
            2,
            open_=101.0,
            high=101.0,
            low=100.5,
            close=101.0,
        ),
        (),
    )
    retired_after_break = break_tracker.snapshot()[0][0]
    assert (
        retired_after_break.lifecycle
        is SupportResistanceLifecycle.RETIRED
    )
    assert (
        retired_after_break.retired_at
        == BASE + pd.Timedelta(minutes=3)
    )

    double_tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(
            retained_zones=8,
            retained_pools=16,
            retained_touches=8,
        ),
    )
    double_sources: list[SwingPoint] = []
    for index in range(8):
        price = 400.0 - 10.0 * index
        source = _swing(
            f"double-capacity-{index}",
            side=SwingSide.HIGH,
            price=price,
            confirmed_index=index,
        )
        double_sources.append(source)
        double_tracker.on_candle(
            _candle(
                index,
                open_=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
            ),
            tuple(double_sources),
        )
    new_high = _swing(
        "double-capacity-new-high",
        side=SwingSide.HIGH,
        price=300.0,
        confirmed_index=8,
    )
    new_low = _swing(
        "double-capacity-new-low",
        side=SwingSide.LOW,
        price=200.0,
        confirmed_index=8,
    )
    bounded_tracker = pickle.loads(pickle.dumps(double_tracker))
    before_bounded_failure = to_primitive(
        bounded_tracker.snapshot()
    )
    before_bounded_clock = bounded_tracker.last_end
    before_bounded_index = bounded_tracker._bar_index
    before_bounded_known = set(bounded_tracker._known_swing_ids)
    with pytest.raises(
        ValueError,
        match="zone retention is exhausted",
    ):
        bounded_tracker.on_candle(
            _candle(
                8,
                open_=250.0,
                high=300.2,
                low=199.8,
                close=250.0,
            ),
            (
                *double_sources[1:],
                new_high,
                new_low,
            ),
        )
    assert (
        to_primitive(bounded_tracker.snapshot())
        == before_bounded_failure
    )
    assert bounded_tracker.last_end == before_bounded_clock
    assert bounded_tracker._bar_index == before_bounded_index
    assert bounded_tracker._known_swing_ids == before_bounded_known

    current_sources = (
        *double_sources[2:],
        new_high,
        new_low,
    )
    double_tracker.on_candle(
        _candle(
            8,
            open_=250.0,
            high=300.2,
            low=199.8,
            close=250.0,
        ),
        current_sources,
    )
    double_zones = double_tracker.snapshot()[0]
    assert len(double_zones) == 10
    assert sum(
        item.lifecycle is SupportResistanceLifecycle.RETIRED
        for item in double_zones
    ) == 2

    double_tracker.on_candle(
        _candle(
            9,
            open_=250.0,
            high=300.2,
            low=199.8,
            close=250.0,
            synthetic=True,
        ),
        current_sources,
    )
    assert len(double_tracker.snapshot()[0]) == 10

    double_tracker.on_candle(
        _candle(
            10,
            open_=250.0,
            high=300.2,
            low=199.8,
            close=250.0,
        ),
        current_sources,
    )
    assert len(double_tracker.snapshot()[0]) == 8


def test_one_minute_pool_projection_is_authoritative_and_starts_new_generation() -> None:
    tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(retained_touches=8),
    )
    first = _swing(
        "generation-0-a",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=0,
    )
    second = _swing(
        "generation-0-b",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=1,
    )
    candle0 = _candle(
        0, open_=99.75, high=100.2, low=99.5, close=99.75
    )
    candle1 = _candle(
        1, open_=99.75, high=100.2, low=99.5, close=99.75
    )
    tracker.on_candle(candle0, (first,))
    tracker.on_candle(candle1, (first, second))
    _, pools, inventory = tracker.snapshot()
    original = pools[0]
    visible = next(item for item in inventory if item.kind == "equal_highs")

    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
        )
    )
    observer._liquidity_trackers[Timeframe.M1] = tracker
    observer._prior = _with_authoritative_liquidity(
        market_observation(asof=candle1.end),
        inventory=(visible,),
        pools=(original,),
    )
    sweep_bar = _candle(
        2, open_=100.0, high=100.75, low=99.75, close=100.0
    )
    tracker.on_candle(sweep_bar, (first, second))
    sweep_update = ReaderUpdate(
        asof=sweep_bar.end,
        completed_1m=sweep_bar,
        newly_completed={},
        histories={},
        anomalies=(),
    )
    swept_inventory = observer._project_inventory(
        sweep_update,
        market_observation(asof=sweep_bar.end).frames,
        tracker.snapshot()[2],
    )
    swept_pool_item = next(
        item for item in swept_inventory if item.kind == "equal_highs"
    )
    assert (
        swept_pool_item.lifecycle
        is LiquidityInventoryLifecycle.CONSUMED
    )
    _, pools, _ = tracker.snapshot()
    assert pools[0].lifecycle is LiquidityPoolLifecycle.SWEPT
    assert pools[0].swept_at == sweep_bar.end

    source_view = observer._pool_formation_source(pools[0])
    FrameObservation(
        timeframe=Timeframe.M1,
        cutoff=original.confirmed_at,
        bars=2,
        metrics={},
        liquidity_pools=(source_view,),
    )
    assert source_view.lifecycle is LiquidityPoolLifecycle.FORMED
    assert source_view.swept_at is None

    swept_pool = tracker.snapshot()[1][0]
    observer._prior = _with_authoritative_liquidity(
        market_observation(asof=sweep_bar.end),
        inventory=swept_inventory,
        pools=(swept_pool,),
        swings=(first, second),
    )
    resolution_bar = _candle(
        3, open_=100.0, high=100.2, low=99.5, close=100.0
    )
    resolution_update = ReaderUpdate(
        asof=resolution_bar.end,
        completed_1m=resolution_bar,
        newly_completed={},
        histories={},
        anomalies=(),
    )
    same_clock_touch = _swing(
        "generation-1-at-resolution",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=3,
    )
    observer._resolve_pending_pool_sweeps(resolution_update)
    tracker.on_candle(
        resolution_bar,
        (first, second, same_clock_touch),
    )
    observer._project_inventory(
        resolution_update,
        market_observation(asof=resolution_bar.end).frames,
        tracker.snapshot()[2],
    )
    _, pools, _ = tracker.snapshot()
    resolved = pools[0]
    assert resolved.lifecycle is LiquidityPoolLifecycle.REJECTED
    assert resolved.resolved_at == resolution_bar.end

    third = _swing(
        "generation-1-a",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=4,
    )
    fourth = _swing(
        "generation-1-b",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=5,
    )
    third_bar = _candle(
        4, open_=99.75, high=100.2, low=99.5, close=99.75
    )
    fourth_bar = _candle(
        5, open_=99.75, high=100.2, low=99.5, close=99.75
    )
    tracker.on_candle(
        third_bar,
        (first, second, same_clock_touch, third),
    )
    tracker.on_candle(
        fourth_bar,
        (first, second, same_clock_touch, third, fourth),
    )
    all_swings = [first, second, same_clock_touch, third, fourth]
    last_touch = fourth
    for index in range(6, 18):
        last_touch = _swing(
            f"generation-1-{index}",
            side=SwingSide.HIGH,
            price=100.0,
            confirmed_index=index,
        )
        all_swings.append(last_touch)
        tracker.on_candle(
            _candle(
                index,
                open_=99.75,
                high=100.2,
                low=99.5,
                close=99.75,
            ),
            tuple(all_swings),
        )
    zones, pools, _ = tracker.snapshot()
    formed = next(
        item
        for item in pools
        if item.lifecycle is LiquidityPoolLifecycle.FORMED
    )
    assert formed.pool_id != resolved.pool_id
    assert formed.member_swing_ids[:2] == (
        same_clock_touch.swing_id,
        third.swing_id,
    )
    assert next(
        item for item in pools if item.pool_id == resolved.pool_id
    ).lifecycle is LiquidityPoolLifecycle.REJECTED
    assert len(zones[0].member_swing_ids) <= 8
    assert len(formed.member_swing_ids) <= 8
    assert zones[0].touch_count == len(all_swings)
    assert formed.touch_count == len(all_swings) - 2
    assert last_touch.swing_id in zones[0].member_swing_ids
    assert last_touch.swing_id in formed.member_swing_ids


def test_terminal_event_duration_closes_and_broken_zone_remains_open() -> None:
    memory = EventMemory(16)
    formed = _event(
        EventKind.LIQUIDITY_POOL_STATE,
        BASE,
        Timeframe.M1,
        "above",
        100.0,
        0.5,
        ("a", "b"),
        entity_id="pool:duration",
        lifecycle=LiquidityPoolLifecycle.FORMED.value,
        formed_at=BASE - pd.Timedelta(minutes=2),
        confirmed_at=BASE,
    )
    swept_at = BASE + pd.Timedelta(minutes=2)
    swept = _event(
        EventKind.LIQUIDITY_POOL_STATE,
        swept_at,
        Timeframe.M1,
        "above",
        100.5,
        0.5,
        ("a", "b"),
        entity_id="pool:duration",
        lifecycle=LiquidityPoolLifecycle.SWEPT.value,
        formed_at=BASE - pd.Timedelta(minutes=2),
        confirmed_at=BASE,
    )
    terminal_at = BASE + pd.Timedelta(minutes=3)
    terminal = _event(
        EventKind.LIQUIDITY_POOL_STATE,
        terminal_at,
        Timeframe.M1,
        "above",
        100.0,
        0.5,
        ("a", "b"),
        entity_id="pool:duration",
        lifecycle=LiquidityPoolLifecycle.REJECTED.value,
        formed_at=BASE - pd.Timedelta(minutes=2),
        confirmed_at=BASE,
        ended_at=terminal_at,
        transition_reason="close_returned_inside",
    )
    memory.append(formed)
    memory.append(swept)
    memory.append(terminal)
    durations = memory.durations(terminal_at + pd.Timedelta(minutes=20))
    assert durations[formed.event_id] == 2
    assert durations[swept.event_id] == 1
    assert durations[terminal.event_id] == 0
    assert "pool:duration" not in memory._latest_by_entity

    tested_at = BASE + pd.Timedelta(minutes=1)
    tested = SupportResistanceState(
        zone_id="zone-tested-duration",
        timeframe=Timeframe.M1,
        side="resistance",
        lower_bound=99.75,
        upper_bound=100.25,
        anchor_price=100.0,
        formed_at=BASE - pd.Timedelta(minutes=2),
        confirmed_at=BASE,
        lifecycle=SupportResistanceLifecycle.TESTED,
        member_swing_ids=("swing-a", "swing-b"),
        touch_times=(BASE, tested_at),
        reaction_magnitudes_atr=(0.5, 0.4),
        age_bars=1,
        strength=0.5,
        tested_at=tested_at,
        total_touch_count=2,
    )
    tested_observer = CausalObserver()
    tested_observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=tested_at,
            bars=2,
            metrics={},
            support_resistance=(tested,),
        ),
        newly_completed=True,
    )
    event_count = len(tested_observer.memory._events)
    third_touch_at = BASE + pd.Timedelta(minutes=2)
    updated_tested = replace(
        tested,
        member_swing_ids=("swing-a", "swing-b", "swing-c"),
        touch_times=(BASE, tested_at, third_touch_at),
        reaction_magnitudes_atr=(0.5, 0.4, 0.3),
        age_bars=2,
        total_touch_count=3,
    )
    tested_observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=third_touch_at,
            bars=3,
            metrics={},
            support_resistance=(updated_tested,),
        ),
        newly_completed=True,
    )
    assert updated_tested.touch_count == 3
    assert len(tested_observer.memory._events) == event_count
    tested_event = next(iter(tested_observer.memory._events))
    assert tested_observer.memory.durations(
        BASE + pd.Timedelta(minutes=5)
    )[tested_event.event_id] == 4

    broken_at = BASE + pd.Timedelta(minutes=5)
    broken = SupportResistanceState(
        zone_id="zone-duration",
        timeframe=Timeframe.M1,
        side="resistance",
        lower_bound=99.75,
        upper_bound=100.25,
        anchor_price=100.0,
        formed_at=BASE - pd.Timedelta(minutes=2),
        confirmed_at=BASE,
        lifecycle=SupportResistanceLifecycle.BROKEN,
        member_swing_ids=("swing-a",),
        touch_times=(BASE,),
        reaction_magnitudes_atr=(0.5,),
        age_bars=5,
        strength=0.5,
        broken_at=broken_at,
        transition_reason="close_beyond_frozen_zone",
    )
    frame = FrameObservation(
        timeframe=Timeframe.M1,
        cutoff=broken_at,
        bars=5,
        metrics={},
        support_resistance=(broken,),
    )
    observer = CausalObserver()
    observer._record_frame_events(frame, newly_completed=True)
    broken_event = next(
        item
        for item in observer.memory.recent()
        if item.entity_id == broken.zone_id
    )
    assert broken_event.ended_at is None
    assert observer.memory.durations(
        broken_at + pd.Timedelta(minutes=4)
    )[broken_event.event_id] == 4

    count = len(observer.memory._events)
    observer._record_frame_events(
        replace(frame, cutoff=broken_at + pd.Timedelta(minutes=1)),
        newly_completed=False,
    )
    assert len(observer.memory._events) == count

    retired_at = broken_at + pd.Timedelta(minutes=2)
    retired = replace(
        broken,
        lifecycle=SupportResistanceLifecycle.RETIRED,
        retired_at=retired_at,
        transition_reason=SUPPORT_RESISTANCE_RETIREMENT_REASON,
    )
    observer._record_frame_events(
        replace(
            frame,
            cutoff=retired_at,
            bars=7,
            support_resistance=(retired,),
        ),
        newly_completed=True,
    )
    retired_event = next(
        item
        for item in observer.memory.recent()
        if (
            item.entity_id == retired.zone_id
            and item.lifecycle
            == SupportResistanceLifecycle.RETIRED.value
        )
    )
    assert retired_event.observed_at == retired_at
    assert retired_event.ended_at == retired_at
    assert (
        retired_event.transition_reason
        == SUPPORT_RESISTANCE_RETIREMENT_REASON
    )
    assert retired.zone_id not in observer.memory._latest_by_entity
    with pytest.raises(
        ValueError,
        match="future support/resistance",
    ):
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=retired_at - pd.Timedelta(minutes=1),
            bars=7,
            metrics={},
            support_resistance=(retired,),
        )


def _timeline_zone_events(
    identity: str,
) -> tuple[MarketEvent, MarketEvent, MarketEvent]:
    active_at = BASE + pd.Timedelta(minutes=1)
    tested_at = BASE + pd.Timedelta(minutes=4)
    retired_at = BASE + pd.Timedelta(minutes=8)
    return (
        _event(
            EventKind.SUPPORT_RESISTANCE_STATE,
            active_at,
            Timeframe.M1,
            "above",
            100.0,
            0.4,
            entity_id=identity,
            lifecycle=SupportResistanceLifecycle.ACTIVE.value,
            formed_at=BASE,
            confirmed_at=active_at,
        ),
        _event(
            EventKind.SUPPORT_RESISTANCE_STATE,
            tested_at,
            Timeframe.M1,
            "above",
            100.0,
            0.6,
            entity_id=identity,
            lifecycle=SupportResistanceLifecycle.TESTED.value,
            formed_at=BASE,
            confirmed_at=active_at,
        ),
        _event(
            EventKind.SUPPORT_RESISTANCE_STATE,
            retired_at,
            Timeframe.M1,
            "above",
            100.0,
            0.6,
            entity_id=identity,
            lifecycle=SupportResistanceLifecycle.RETIRED.value,
            formed_at=BASE,
            confirmed_at=active_at,
            ended_at=retired_at,
            transition_reason=SUPPORT_RESISTANCE_RETIREMENT_REASON,
        ),
    )


def test_retained_entity_timeline_survives_fifo_eviction() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        EventMemory(0)
    with pytest.raises(ValueError, match="positive integer"):
        EventMemory(True)

    memory = EventMemory(2)
    active, tested, retired = _timeline_zone_events("timeline-zone")
    for event in (active, tested, retired):
        memory.append(event)
        memory.sync_retained_entity_timelines(
            {"zone:timeline-zone"},
            asof=event.observed_at,
        )
    for offset in range(9, 13):
        memory.append(
            _event(
                EventKind.REJECTION,
                BASE + pd.Timedelta(minutes=offset),
                Timeframe.M1,
                "above",
                101.0,
                0.25,
            )
        )
    memory.sync_retained_entity_timelines(
        {"zone:timeline-zone"},
        asof=BASE + pd.Timedelta(minutes=12),
    )

    assert all(
        event.kind is EventKind.REJECTION
        for event in memory.recent()
    )
    assert tuple(
        event.lifecycle
        for event in memory.timeline("zone:timeline-zone")
    ) == ("active", "tested", "retired")
    durations = memory.durations(
        BASE + pd.Timedelta(minutes=20)
    )
    assert durations[active.event_id] == 3
    assert durations[tested.event_id] == 4
    assert durations[retired.event_id] == 0
    ages = memory.ages(BASE + pd.Timedelta(minutes=20))
    assert {
        ages[active.event_id],
        ages[tested.event_id],
        ages[retired.event_id],
    } == {20}

    before = memory.entity_timelines()
    memory.append(active)
    assert memory.entity_timelines() == before
    with pytest.raises(ValueError, match="event id conflicts"):
        memory.append(replace(active, strength=0.9))
    assert memory.entity_timelines() == before


def test_projection_dedupe_expiry_does_not_rewrite_retained_lifecycle() -> None:
    observer = CausalObserver()
    swing = _swing(
        "long-retained-swing",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=0,
    )
    frame = FrameObservation(
        timeframe=Timeframe.M1,
        cutoff=swing.confirmed_at,
        bars=3,
        metrics={},
        swings=(swing,),
    )
    observer._record_frame_events(frame, newly_completed=True)
    original = observer.memory._entity_timelines[
        "swing:long-retained-swing"
    ][0]

    observer._known_level_ids.clear()
    observer._known_level_order.clear()
    observer._record_frame_events(
        replace(
            frame,
            cutoff=swing.confirmed_at + pd.Timedelta(minutes=1),
            bars=4,
            swings=(replace(swing, age_bars=99),),
        ),
        newly_completed=True,
    )
    assert observer.memory._entity_timelines[
        "swing:long-retained-swing"
    ] == [original]
    assert original.details["age_bars"] == 0


def test_noninitial_cold_timeline_is_explicitly_incomplete() -> None:
    memory = EventMemory(4)
    confirmed_swing = _swing(
        "cold-confirmed-swing",
        side=SwingSide.HIGH,
        price=100.0,
        confirmed_index=0,
    )
    event = _event(
        EventKind.SWING_STATE,
        confirmed_swing.confirmed_at,
        Timeframe.M1,
        "above",
        confirmed_swing.price,
        0.4,
        entity_id=confirmed_swing.swing_id,
        lifecycle=SwingLifecycle.CONFIRMED.value,
        formed_at=confirmed_swing.pivot_end,
        confirmed_at=confirmed_swing.confirmed_at,
    )
    memory.append(event)
    memory.sync_retained_entity_timelines(
        {"swing:cold-confirmed-swing"},
        asof=event.observed_at,
    )
    assert memory.incomplete_entity_keys() == (
        "swing:cold-confirmed-swing",
    )

    memory.sync_retained_entity_timelines(
        set(),
        asof=event.observed_at,
    )
    assert memory.incomplete_entity_keys() == ()


def test_retained_entity_timeline_duration_excludes_synthetic_minutes() -> None:
    memory = EventMemory(4)
    for index in range(5):
        memory.observe_minute(
            _candle(
                index,
                open_=100.0,
                high=100.25,
                low=99.75,
                close=100.0,
                synthetic=index in {1, 2},
            )
        )
    active, _, _ = _timeline_zone_events("synthetic-zone")
    tested = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        BASE + pd.Timedelta(minutes=5),
        Timeframe.M1,
        "above",
        100.0,
        0.6,
        entity_id="synthetic-zone",
        lifecycle=SupportResistanceLifecycle.TESTED.value,
        formed_at=BASE,
        confirmed_at=active.confirmed_at,
    )
    memory.append(active)
    memory.append(tested)
    memory.sync_retained_entity_timelines(
        {"zone:synthetic-zone"},
        asof=tested.observed_at,
    )
    assert memory.durations(tested.observed_at)[active.event_id] == 2
    assert memory.ages(tested.observed_at)[tested.event_id] == 3


def test_retained_entity_timeline_prunes_with_typed_snapshot() -> None:
    memory = EventMemory(1)
    zone, _, _ = _timeline_zone_events("pruned-zone")
    pool = _event(
        EventKind.LIQUIDITY_POOL_STATE,
        BASE + pd.Timedelta(minutes=2),
        Timeframe.M5,
        "above",
        101.0,
        0.5,
        ("a", "b"),
        entity_id="pool:retained-pool",
        lifecycle=LiquidityPoolLifecycle.FORMED.value,
        formed_at=BASE,
        confirmed_at=BASE + pd.Timedelta(minutes=2),
    )
    memory.append(zone)
    memory.append(pool)
    memory.sync_retained_entity_timelines(
        {"zone:pruned-zone", "pool:retained-pool"},
        asof=pool.observed_at,
    )
    memory.append(
        _event(
            EventKind.REJECTION,
            BASE + pd.Timedelta(minutes=3),
            Timeframe.M1,
            "above",
            101.0,
            0.2,
        )
    )
    memory.sync_retained_entity_timelines(
        {"pool:retained-pool"},
        asof=BASE + pd.Timedelta(minutes=3),
    )
    assert set(memory.entity_timelines()) == {
        "pool:retained-pool"
    }
    assert zone.event_id not in memory.durations(
        BASE + pd.Timedelta(minutes=3)
    )
    assert zone.event_id not in memory.ages(
        BASE + pd.Timedelta(minutes=3)
    )
    with pytest.raises(
        ValueError,
        match="lacks a lifecycle timeline",
    ):
        memory.sync_retained_entity_timelines(
            {"zone:pruned-zone", "pool:retained-pool"},
            asof=BASE + pd.Timedelta(minutes=3),
        )


def test_hidden_typed_timeline_prunes_duration_bookkeeping() -> None:
    memory = EventMemory(1)
    active, tested, retired = _timeline_zone_events(
        "hidden-pruned-zone"
    )
    for event in (active, tested, retired):
        memory.append(event, include_in_recent=False)
    memory.sync_retained_entity_timelines(
        {"zone:hidden-pruned-zone"},
        asof=retired.observed_at,
    )
    assert memory._closed_durations

    memory.sync_retained_entity_timelines(
        set(),
        asof=retired.observed_at,
    )
    assert memory._closed_durations == {}
    assert memory._latest_by_entity == {}


def test_retained_entity_timeline_namespaces_and_order_fail_closed() -> None:
    memory = EventMemory(8)
    clock = BASE + pd.Timedelta(minutes=1)
    events = (
        _event(
            EventKind.SWING_STATE,
            clock,
            Timeframe.M1,
            "above",
            100.0,
            0.4,
            entity_id="same",
            lifecycle=SwingLifecycle.CONFIRMED.value,
            formed_at=BASE,
            confirmed_at=clock,
        ),
        _event(
            EventKind.STRUCTURE_STATE,
            clock,
            Timeframe.M1,
            "above",
            99.0,
            0.4,
            entity_id="same",
            lifecycle=StructureLifecycle.CONFIRMED.value,
            formed_at=BASE,
            confirmed_at=clock,
        ),
        _event(
            EventKind.BOS_STATE,
            clock,
            Timeframe.M1,
            "above",
            101.0,
            0.4,
            entity_id="same",
            lifecycle=BOSLifecycle.PENDING.value,
            formed_at=clock,
        ),
        _event(
            EventKind.SUPPORT_RESISTANCE_STATE,
            clock,
            Timeframe.M1,
            "above",
            100.0,
            0.4,
            entity_id="same",
            lifecycle=SupportResistanceLifecycle.ACTIVE.value,
            formed_at=BASE,
            confirmed_at=clock,
        ),
    )
    for event in events:
        memory.append(event)
    keys = {
        "swing:same",
        "structure:same",
        "bos:same",
        "zone:same",
    }
    memory.sync_retained_entity_timelines(keys, asof=clock)
    assert set(memory.entity_timelines()) == keys

    prior = memory.entity_timelines()
    with pytest.raises(ValueError, match="out of order"):
        memory.append(
            _event(
                EventKind.SUPPORT_RESISTANCE_STATE,
                BASE,
                Timeframe.M1,
                "above",
                100.0,
                0.4,
                entity_id="same",
                lifecycle=SupportResistanceLifecycle.TESTED.value,
                formed_at=BASE - pd.Timedelta(minutes=1),
                confirmed_at=BASE,
            )
        )
    assert memory.entity_timelines() == prior


def test_retained_entity_timeline_extends_clock_coverage_guard() -> None:
    memory = EventMemory(1)
    event, _, _ = _timeline_zone_events("coverage-zone")
    memory.append(event)
    memory.sync_retained_entity_timelines(
        {"zone:coverage-zone"},
        asof=event.observed_at,
    )
    memory.append(
        _event(
            EventKind.REJECTION,
            BASE + pd.Timedelta(minutes=2),
            Timeframe.M1,
            "above",
            101.0,
            0.2,
        )
    )
    assert all(
        item.event_id != event.event_id
        for item in memory.recent()
    )
    with pytest.raises(
        ValueError,
        match="origin predates retained 1m clock coverage",
    ):
        memory.set_clock_coverage_start(
            BASE + pd.Timedelta(minutes=1)
        )


def test_market_observation_validates_retained_entity_timeline() -> None:
    observation = market_observation()
    active_at = observation.asof - pd.Timedelta(minutes=3)
    tested_at = observation.asof - pd.Timedelta(minutes=1)
    active = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        active_at,
        Timeframe.M1,
        "above",
        100.0,
        0.4,
        entity_id="observation-zone",
        lifecycle=SupportResistanceLifecycle.ACTIVE.value,
        formed_at=active_at,
        confirmed_at=active_at,
    )
    tested = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        tested_at,
        Timeframe.M1,
        "above",
        100.0,
        0.5,
        entity_id="observation-zone",
        lifecycle=SupportResistanceLifecycle.TESTED.value,
        formed_at=active_at,
        confirmed_at=active_at,
    )
    durations = {
        **observation.event_durations_minutes,
        active.event_id: 2,
        tested.event_id: 1,
    }
    ages = {
        active.event_id: 3,
        tested.event_id: 3,
    }
    valid = replace(
        observation,
        event_durations_minutes=durations,
        event_ages_minutes=ages,
        retained_entity_timelines={
            "zone:observation-zone": (active, tested)
        },
    )
    assert valid.retained_entity_timelines[
        "zone:observation-zone"
    ] == (active, tested)

    with pytest.raises(ValueError, match="key disagrees"):
        replace(
            valid,
            retained_entity_timelines={
                "swing:observation-zone": (active, tested)
            },
        )
    with pytest.raises(ValueError, match="not causally ordered"):
        replace(
            valid,
            retained_entity_timelines={
                "zone:observation-zone": (tested, active)
            },
        )
    with pytest.raises(ValueError, match="lacks duration or age"):
        replace(valid, event_ages_minutes={})
    future = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        observation.asof + pd.Timedelta(minutes=1),
        Timeframe.M1,
        "above",
        100.0,
        0.4,
        entity_id="future-zone",
        lifecycle=SupportResistanceLifecycle.ACTIVE.value,
        formed_at=observation.asof,
        confirmed_at=observation.asof,
    )
    with pytest.raises(ValueError, match="future event"):
        replace(
            observation,
            event_durations_minutes={future.event_id: 0},
            event_ages_minutes={future.event_id: 0},
            retained_entity_timelines={
                "zone:future-zone": (future,)
            },
        )
