from __future__ import annotations

from dataclasses import replace
import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.causal import ReaderUpdate
from smc_trader.liquidity import (
    CausalLiquidityTracker,
    LiquidityConfig,
    LiquidityProtocolError,
)
from smc_trader.market_state import replay_atomic_market_snapshot
from smc_trader.model import (
    BOS_CONFIRMATION_REASON,
    BOSLifecycle,
    BOSPostBreakState,
    Candle,
    Direction,
    EventKind,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    MarketEvent,
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
    _strict_prior_atr,
)
from smc_trader.playbooks import _visible_levels
from smc_trader.risk import _visible_level_ids
from smc_trader.structure import StructureConfig, StructureTracker

from .helpers import (
    CORE_TEST_SCALE_REGISTRY_ID,
    CORE_TEST_SCALE_SPECS,
    market_observation,
)


TZ = "America/New_York"
BASE = pd.Timestamp("2025-01-06 10:00", tz=TZ)
STRUCTURE_PROTOCOL = (
    "configs/primitives_structure_liquidity.json"
)
CORE_TEST_TIMEFRAMES = tuple(
    spec.native_timeframe
    for spec in CORE_TEST_SCALE_SPECS
    if spec.enabled and spec.native_timeframe is not None
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


def _append_authoritative_high_swing_root(
    observer: CausalObserver,
    *,
    swing_id: str,
    pivot_index: int,
    price: float,
) -> MarketEvent:
    window = (
        _candle(
            pivot_index - 1,
            open_=price - 0.75,
            high=price - 0.5,
            low=price - 1.0,
            close=price - 0.75,
        ),
        _candle(
            pivot_index,
            open_=price - 0.5,
            high=price,
            low=price - 0.75,
            close=price - 0.25,
        ),
        _candle(
            pivot_index + 1,
            open_=price - 0.25,
            high=price - 0.25,
            low=price - 1.0,
            close=price - 0.75,
        ),
    )
    bar_event_ids = tuple(
        observer._append_completed_bar_event(
            candle,
            atr=1.0,
            data_complete=True,
        ).event_id
        for candle in window
    )
    pivot = window[1]
    confirmed_at = window[-1].end
    event = observer._append_semantic_atomic(
        EventKind.SWING_CONFIRMED,
        confirmed_at,
        Timeframe.M1,
        "above",
        price,
        0.25,
        bar_event_ids,
        {
            "source_entity_id": swing_id,
            "side": "high",
            "relation": "hh",
            "pivot_start": pivot.start.isoformat(),
            "pivot_end": pivot.end.isoformat(),
            "prominence_atr": 0.5,
            "legacy_same_side_magnitude_atr": 0.25,
            "confirmation_delay_bars": 1,
            "nesting_depth": 0,
            "semantic_rank": "micro",
            "delta_ticks": 1,
            "confirmation_delay_minutes": 2,
        },
        event_time=pivot.start,
        source_entity_ids=(swing_id,),
    )
    observer._confirmed_swing_event_ids[swing_id] = event.event_id
    return event


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
        liquidity_pool_states=pools,
    )


def test_confirmed_swing_crossing_key_is_timeframe_scoped_and_resolves() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    crossed_at = BASE + pd.Timedelta(minutes=5)
    swing = replace(
        _swing(
            "swing-crossing",
            side=SwingSide.HIGH,
            price=100.0,
            confirmed_index=0,
        ),
        lifecycle=SwingLifecycle.BROKEN,
        pivot_start=BASE - pd.Timedelta(minutes=1),
        pivot_end=BASE,
        confirmation_delay_bars=1,
        broken_at=crossed_at,
        failure_reason="close_beyond_swing",
    )
    level_id = f"swing:{swing.swing_id}"
    generation_id = observer._crossing_generation_id(
        level_id=level_id,
        timeframe=Timeframe.M1,
        crossed_at=crossed_at,
    )
    m1_key = observer._penetration_key(
        level_id=level_id,
        timeframe=Timeframe.M1,
        crossed_at=crossed_at,
    )
    m5_key = observer._penetration_key(
        level_id=level_id,
        timeframe=Timeframe.M5,
        crossed_at=crossed_at,
    )
    assert m1_key != m5_key

    swing_window = (
        _candle(
            -2,
            open_=98.5,
            high=99.0,
            low=98.0,
            close=98.75,
        ),
        _candle(
            -1,
            open_=99.0,
            high=100.0,
            low=98.75,
            close=99.5,
        ),
        _candle(
            0,
            open_=99.25,
            high=99.5,
            low=98.5,
            close=99.0,
        ),
    )
    swing_bar_event_ids = tuple(
        observer._append_completed_bar_event(
            candle,
            atr=1.0,
            data_complete=True,
        ).event_id
        for candle in swing_window
    )
    swing_root = observer._append_semantic_atomic(
        EventKind.SWING_CONFIRMED,
        swing.confirmed_at,
        Timeframe.M1,
        "above",
        swing.price,
        swing.magnitude_atr,
        swing_bar_event_ids,
        {
            "source_entity_id": swing.swing_id,
            "side": "high",
            "relation": swing.relation.value,
            "pivot_start": swing.pivot_start.isoformat(),
            "pivot_end": swing.pivot_end.isoformat(),
            "prominence_atr": swing.prominence_atr,
            "legacy_same_side_magnitude_atr": swing.magnitude_atr,
            "confirmation_delay_bars": 1,
            "nesting_depth": swing.nesting_depth,
            "semantic_rank": swing.semantic_rank.value,
            "delta_ticks": swing.delta_ticks,
            "confirmation_delay_minutes": 2,
        },
        event_time=swing.pivot_start,
        source_entity_ids=(swing.swing_id,),
    )

    break_bar = _candle(
        4,
        open_=99.75,
        high=100.5,
        low=99.5,
        close=100.25,
    )
    break_bar_event = observer._append_completed_bar_event(
        break_bar,
        atr=1.0,
        data_complete=True,
    )
    candidate = observer._append_semantic_atomic(
        EventKind.LIQUIDITY_LEVEL_CREATED,
        swing.confirmed_at,
        Timeframe.M1,
        "above",
        swing.price,
        swing.magnitude_atr,
        (swing_root.event_id,),
        {
            "level_id": level_id,
            "source_kind": "confirmed_swing",
            "candidate_only": True,
            "source_swing_id": swing.swing_id,
            "semantic_rank": swing.semantic_rank.value,
        },
        event_time=swing.pivot_start,
        zone=(swing.price, swing.price),
        source_entity_ids=(level_id, swing.swing_id),
    )
    touch = observer._append_semantic_atomic(
        EventKind.LEVEL_TOUCHED,
        crossed_at,
        Timeframe.M1,
        "above",
        swing.price,
        swing.magnitude_atr,
        (candidate.event_id, break_bar_event.event_id),
        {
            "level_id": level_id,
            "source_kind": "confirmed_swing",
        },
        event_time=crossed_at,
        zone=(swing.price, swing.price),
    )
    penetration = observer._append_semantic_atomic(
        EventKind.LEVEL_PENETRATED,
        crossed_at,
        Timeframe.M1,
        "above",
        swing.price,
        swing.magnitude_atr,
        (candidate.event_id, touch.event_id, break_bar_event.event_id),
        {
            "level_id": level_id,
            "source_kind": "confirmed_swing",
            "crossing_generation_id": generation_id,
            "crossed_at": crossed_at.isoformat(),
        },
        direction=Direction.LONG,
        event_time=crossed_at,
        zone=(swing.price, swing.price),
    )
    observer._penetration_event_ids[m1_key] = penetration.event_id
    observer._confirmed_swing_event_ids[swing.swing_id] = swing_root.event_id
    observer._candidate_level_event_ids[level_id] = candidate.event_id
    crossing_frame = FrameObservation(
        timeframe=Timeframe.M1,
        cutoff=crossed_at,
        bars=1,
        metrics={"atr": 1.0},
        ready=True,
        swings=(swing,),
    )
    observer._record_frame_events(
        crossing_frame,
        True,
        event_clock=crossed_at,
    )
    penetration_event_id = observer._penetration_event_ids.get(m1_key)
    assert penetration_event_id is not None
    observer._penetration_event_ids[m5_key] = "other-timeframe-penetration"
    assert generation_id not in observer._terminal_crossing_events

    synthetic_resolution = _candle(
        5,
        open_=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        synthetic=True,
    )
    synthetic_root = observer._append_completed_bar_event(
        synthetic_resolution,
        atr=1.0,
        data_complete=True,
    )
    observer._record_frame_events(
        replace(crossing_frame, cutoff=synthetic_resolution.end),
        True,
        event_clock=synthetic_resolution.end,
    )
    assert generation_id not in observer._terminal_crossing_events
    assert synthetic_root.event_id in {
        event_id
        for _, event_id in observer._bar_event_ids_by_timeframe[
            Timeframe.M1
        ]
    }
    with pytest.raises(
        ValueError,
        match="no exact completed-bar source",
    ):
        observer._bar_event_id_at(
            Timeframe.M1,
            synthetic_resolution.end,
        )

    resolution_bar = _candle(
        6,
        open_=100.0,
        high=100.5,
        low=99.75,
        close=100.25,
    )
    observer._append_completed_bar_event(
        resolution_bar,
        atr=1.0,
        data_complete=True,
    )
    observer._record_frame_events(
        replace(crossing_frame, cutoff=resolution_bar.end),
        True,
        event_clock=resolution_bar.end,
    )
    terminal = observer._terminal_crossing_events.get(generation_id)

    assert terminal is not None
    assert terminal.kind is EventKind.ACCEPTANCE_CONFIRMED
    assert terminal.source_event_ids[0] == penetration_event_id
    assert terminal.evidence["crossing_generation_id"] == generation_id
    assert terminal.evidence["resolved_at"] == resolution_bar.end.isoformat()
    observer.memory.flush_audit()
    assert observer.audit_store.get(terminal.event_id) is not None
    assert {
        observer.audit_store.get(event_id).kind
        for event_id in terminal.source_event_ids
    } == {
        EventKind.LEVEL_PENETRATED,
        EventKind.BAR_COMPLETED,
    }
    with pytest.raises(
        ValueError,
        match="conflicting terminal resolutions",
    ):
        observer._append_crossing_resolution(
            EventKind.ACCEPTANCE_CONFIRMED,
            resolution_bar.end + pd.Timedelta(minutes=1),
            Timeframe.M1,
            "above",
            float(resolution_bar.close),
            swing.magnitude_atr,
            terminal.source_event_ids,
            {"level_id": level_id},
            direction=Direction.LONG,
            crossed_at=crossed_at,
            zone=(swing.price, swing.price),
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
    assert state.body_class == "doji"
    assert state.range_class == "compressed"
    assert state.dominant_wick == "none"
    assert state.close_class == "middle"
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
    assert normal_state.body_class == "normal"
    assert normal_state.range_class == "normal"
    assert normal_state.dominant_wick == "balanced"
    assert normal_state.close_class == "middle"
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


def test_candle_derived_classes_use_only_strictly_prior_atr() -> None:
    prior = _candle(
        0,
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
    )
    large = _candle(
        1,
        open_=99.5,
        high=104.0,
        low=99.0,
        close=103.5,
    )
    prior_atr = _strict_prior_atr((prior, large), large, 14)
    assert prior_atr == pytest.approx(2.0)

    state = _candle_structure(large, prior_atr=prior_atr)
    assert state.body_class == "large"
    assert state.range_class == "expanded"
    assert state.dominant_wick == "none"
    assert state.close_class == "near_high"
    frame = CausalObserver._observe_frame(
        Timeframe.M1,
        (prior, large),
        large.end,
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS),
    )
    assert frame.candle_structure is not None
    assert frame.candle_structure.body_class == "large"
    assert frame.candle_structure.range_class == "expanded"

    small = _candle(
        2,
        open_=100.2,
        high=100.6,
        low=99.6,
        close=100.4,
    )
    small_state = _candle_structure(small, prior_atr=2.0)
    assert small_state.body_class == "small"
    assert small_state.range_class == "compressed"
    assert small_state.dominant_wick == "lower"
    assert small_state.close_class == "near_high"

    # With no causal ATR baseline, the same large geometry is not promoted.
    no_prior_state = _candle_structure(large)
    assert no_prior_state.body_class == "normal"
    assert no_prior_state.range_class == "normal"


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
    config = ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
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
    active_zone = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        BASE,
        Timeframe.M1,
        "below",
        100.0,
        0.5,
        entity_id="synthetic-gap-zone",
        lifecycle=SupportResistanceLifecycle.ACTIVE.value,
        formed_at=BASE,
        confirmed_at=BASE,
    )
    memory.append(active_zone)
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
    durations, ages = memory.temporal_metrics(
        BASE + pd.Timedelta(minutes=2)
    )
    assert durations[active_zone.event_id] == 0
    assert ages[active_zone.event_id] == 0
    real_after_gap = _candle(
        2,
        open_=100.0,
        high=100.25,
        low=99.75,
        close=100.0,
    )
    memory.observe_minute(real_after_gap)
    durations, ages = memory.temporal_metrics(real_after_gap.end)
    assert durations[active_zone.event_id] == 1
    assert ages[active_zone.event_id] == 1


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
            active_timeframes=CORE_TEST_TIMEFRAMES,
            scale_specs=CORE_TEST_SCALE_SPECS,
            scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
        )

    config = ObserverConfig(
        structure_protocol=STRUCTURE_PROTOCOL,
        scale_specs=CORE_TEST_SCALE_SPECS,
    )
    real_observation = CausalObserver(config).observe(update(real_h4))
    mixed_observer = CausalObserver(config)
    mixed_update = update((*real_h4, mixed_tail))
    mixed_observation = mixed_observer.observe(mixed_update)
    real_frame = real_observation.frame(Timeframe.H4)
    mixed_frame = mixed_observation.frame(Timeframe.H4)
    assert mixed_frame.cutoff == mixed_tail.end
    assert mixed_frame.bars == real_frame.bars == len(real_h4)
    assert mixed_frame.metrics == real_frame.metrics
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


def test_descriptive_compression_does_not_create_a_legacy_event() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    real_cutoff = BASE + pd.Timedelta(minutes=5)
    mixed_cutoff = real_cutoff + pd.Timedelta(minutes=5)
    metrics = {
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
    assert observer.memory.recent() == ()
    with pytest.raises(
        ValueError,
        match="frame event clock cannot exceed the observation cutoff",
    ):
        observer._record_frame_events(
            FrameObservation(
                timeframe=Timeframe.M5,
                cutoff=mixed_cutoff,
                bars=4,
                metrics=metrics,
            ),
            newly_completed=True,
            event_clock=mixed_cutoff + pd.Timedelta(minutes=1),
        )


def test_cold_event_memory_fails_when_age_origin_predates_m1_prefix() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
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
        active_timeframes=CORE_TEST_TIMEFRAMES,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
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
            EventKind.LIQUIDITY_CONSUMED,
            retained.end,
            Timeframe.M1,
            "above",
            100.0,
            0.5,
        )
    )
    bounded.append(
        _event(
            EventKind.LIQUIDITY_CONSUMED,
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
    # M1 uses a one-left/one-right internal swing, so the conflicting
    # relation becomes knowable one completed minute earlier than the old
    # five-bar structural-swing fixture expected.
    assert short_state.formation_failed_at == candles[-2].end
    assert (
        short_state.failure_reason
        == STRUCTURE_FORMATION_FAILURE_REASON
    )
    with pytest.raises(ValueError, match="registered clock and reason"):
        replace(short_state, failure_reason="arbitrary")

    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
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
    for candle in candles[:3]:
        tracker.on_candle(candle)
    swings, _, _ = tracker.snapshot()
    forming = next(
        item
        for item in swings
        if item.pivot_start == candles[2].start
        and item.side is SwingSide.HIGH
    )
    assert forming.lifecycle is SwingLifecycle.FORMING
    assert forming.age_bars == 0

    tracker.on_candle(candles[3])
    swings, _, _ = tracker.snapshot()
    failed = next(
        item
        for item in swings
        if item.swing_id == forming.swing_id
    )
    assert failed.lifecycle is SwingLifecycle.FORMATION_FAILED
    assert failed.failure_reason == "right_side_invalidated"
    assert failed.observed_at == candles[3].end
    assert failed.age_bars == 0
    with pytest.raises(ValueError, match="must resolve after"):
        replace(failed, observed_at=failed.pivot_end)

    tracker.on_candle(candles[4])
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


def test_confirmed_swing_uses_two_sided_local_prominence_not_same_side_delta() -> None:
    candles = (
        _candle(0, open_=99.5, high=100.0, low=99.0, close=99.5),
        _candle(1, open_=100.0, high=102.0, low=100.0, close=101.0),
        _candle(2, open_=100.0, high=100.5, low=99.5, close=100.0),
    )
    accepted = StructureTracker(
        Timeframe.M1,
        replace(StructureConfig(), minimum_prominence_atr=1.0),
    )
    rejected = StructureTracker(
        Timeframe.M1,
        replace(StructureConfig(), minimum_prominence_atr=1.5),
    )

    for candle in candles:
        accepted.on_candle(candle)
        rejected.on_candle(candle)

    accepted_swing = next(
        item
        for item in accepted.snapshot()[0]
        if item.pivot_start == candles[1].start
        and item.side is SwingSide.HIGH
    )
    rejected_swing = next(
        item
        for item in rejected.snapshot()[0]
        if item.pivot_start == candles[1].start
        and item.side is SwingSide.HIGH
    )

    assert accepted_swing.lifecycle is SwingLifecycle.CONFIRMED
    assert accepted_swing.prominence_atr == pytest.approx(1.5)
    assert accepted_swing.magnitude_atr == 0.0
    assert accepted_swing.confirmation_delay_bars == 1
    assert rejected_swing.lifecycle is SwingLifecycle.CONFIRMED

    stricter = StructureTracker(
        Timeframe.M1,
        replace(StructureConfig(), minimum_prominence_atr=1.75),
    )
    for candle in candles:
        stricter.on_candle(candle)
    failed = next(
        item
        for item in stricter.snapshot()[0]
        if item.pivot_start == candles[1].start
        and item.side is SwingSide.HIGH
    )
    assert failed.lifecycle is SwingLifecycle.FORMATION_FAILED
    assert failed.failure_reason == "insufficient_prominence"


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
        break_bar_id="manual-break-bar",
        break_distance_atr=0.25,
        post_break_state=BOSPostBreakState.PENDING,
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(confirmed, strength=-0.25)

    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
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
    assert events[BOSLifecycle.CONFIRMED.value].ended_at is None
    assert (
        events[BOSLifecycle.CONFIRMED.value].transition_reason
        == BOS_CONFIRMATION_REASON
    )
    assert confirmed.bos_id in observer.memory._latest_by_entity


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
    assert (pool.lower_bound, pool.upper_bound) == (100.0, 100.25)
    assert any(item.kind == "equal_highs" for item in inventory)
    compact = tracker.snapshot(
        group4_sources_only=True,
        include_support_resistance=False,
    )
    assert compact[0] == ()
    assert compact[1] == pools
    assert compact[2] == tuple(
        item
        for item in inventory
        if item.kind in {"equal_highs", "equal_lows"}
    )
    assert tracker.snapshot(
        group4_sources_only=True,
        include_support_resistance=False,
    ) is compact

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
    compact = tracker.snapshot(
        group4_sources_only=True,
        include_support_resistance=False,
    )
    assert compact[1][0].lifecycle is LiquidityPoolLifecycle.SWEPT
    assert compact[2][0].lifecycle is LiquidityInventoryLifecycle.CONSUMED

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


@pytest.mark.parametrize(
    ("side", "third_price", "confirmation_bar"),
    (
        (
            SwingSide.HIGH,
            100.25,
            _candle(2, open_=99.5, high=99.75, low=99.0, close=99.5),
        ),
        (
            SwingSide.LOW,
            99.75,
            _candle(2, open_=100.5, high=101.0, low=100.25, close=100.5),
        ),
    ),
)
def test_equal_pool_does_not_count_a_third_swing_beyond_member_boundary(
    side: SwingSide,
    third_price: float,
    confirmation_bar: Candle,
) -> None:
    tracker = CausalLiquidityTracker(Timeframe.M1)
    first = _swing("equal-first", side=side, price=100.0, confirmed_index=0)
    second = _swing("equal-second", side=side, price=100.0, confirmed_index=1)
    third = _swing("crossed-third", side=side, price=third_price, confirmed_index=2)
    neutral_close = 99.75 if side is SwingSide.HIGH else 100.25
    tracker.on_candle(
        _candle(0, open_=neutral_close, high=101.0, low=99.0, close=neutral_close),
        (first,),
    )
    tracker.on_candle(
        _candle(1, open_=neutral_close, high=101.0, low=99.0, close=neutral_close),
        (first, second),
    )
    tracker.on_candle(confirmation_bar, (first, second, third))

    pool = tracker.snapshot()[1][0]
    assert pool.member_swing_ids == (first.swing_id, second.swing_id)
    assert pool.touch_count == 2


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


def test_empty_typed_inventory_fails_closed() -> None:
    populated = market_observation()
    assert _visible_levels(populated)
    assert _visible_level_ids(populated)

    empty = _with_authoritative_liquidity(populated)
    assert _visible_levels(empty) == []
    assert _visible_level_ids(empty) == set()


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
        )

    source = _inventory_swing_source(item)
    with pytest.raises(
        ValueError,
        match="retained swing identity",
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
        high=100.0,
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
            scale_specs=CORE_TEST_SCALE_SPECS,
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
        active_timeframes=CORE_TEST_TIMEFRAMES,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
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
            scale_specs=CORE_TEST_SCALE_SPECS,
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
            scale_specs=CORE_TEST_SCALE_SPECS,
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
            scale_specs=CORE_TEST_SCALE_SPECS,
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
            scale_specs=CORE_TEST_SCALE_SPECS,
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


@pytest.mark.parametrize(
    ("side", "kind", "equal_bar", "crossing_bar"),
    (
        (
            "above",
            "swing",
            _candle(0, open_=99.75, high=100.0, low=99.5, close=99.75),
            _candle(1, open_=99.75, high=100.25, low=99.5, close=100.0),
        ),
        (
            "below",
            "swing",
            _candle(0, open_=100.25, high=100.5, low=100.0, close=100.25),
            _candle(1, open_=100.25, high=100.5, low=99.75, close=100.0),
        ),
    ),
)
def test_swing_liquidity_consumes_on_strict_wick_crossing(
    side: str,
    kind: str,
    equal_bar: Candle,
    crossing_bar: Candle,
) -> None:
    item = LiquidityInventoryItem(
        item_id=f"swing:strict-{side}",
        timeframe=Timeframe.H1,
        side=side,
        kind=kind,
        price=100.0,
        lower_bound=100.0,
        upper_bound=100.0,
        formed_at=BASE - pd.Timedelta(minutes=1),
        confirmed_at=BASE,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=(f"strict-{side}",),
        age_bars=0,
        strength=0.5,
    )
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
            scale_specs=CORE_TEST_SCALE_SPECS,
        )
    )
    warmup = tuple(
        _candle(
            index,
            open_=(99.50 if side == "above" else 100.50),
            high=(99.75 if side == "above" else 100.75),
            low=(99.25 if side == "above" else 100.25),
            close=(99.50 if side == "above" else 100.50),
        )
        for index in range(-14, 0)
    )
    update = ReaderUpdate(
        asof=crossing_bar.end,
        completed_1m=crossing_bar,
        newly_completed={},
        histories={Timeframe.M1: (*warmup, equal_bar, crossing_bar)},
        anomalies=(),
        active_timeframes=CORE_TEST_TIMEFRAMES,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
    )

    projected = observer._project_inventory(
        update,
        market_observation(asof=crossing_bar.end).frames,
        (item,),
    )

    assert projected[0].lifecycle is LiquidityInventoryLifecycle.CONSUMED
    assert projected[0].consumed_at == crossing_bar.end


def test_previous_session_reference_retires_into_completed_replacement() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observer._reference_coverage_start = pd.Timestamp(
        "2025-01-05 18:00",
        tz=TZ,
    )

    def reference_candle(start: str, high: float, low: float) -> Candle:
        clock = pd.Timestamp(start, tz=TZ)
        return replace(
            _candle(0, open_=100.0, high=high, low=low, close=100.0),
            start=clock,
            end=clock + pd.Timedelta(minutes=1),
        )

    observer._advance_reference_periods(
        reference_candle("2025-01-06 17:59", 101.0, 99.0),
        append_retirement_events=True,
    )
    observer._advance_reference_periods(
        reference_candle("2025-01-06 18:00", 102.0, 98.0),
        append_retirement_events=True,
    )
    old_items = tuple(
        item
        for item in observer._reference_inventory.values()
        if item.kind.startswith("previous_session_")
    )
    assert {item.source_ids[0] for item in old_items} == {
        "reference_source:session:2025-01-06:high:NQH5:1",
        "reference_source:session:2025-01-06:low:NQH5:1",
    }

    observer._advance_reference_periods(
        reference_candle("2025-01-07 18:00", 103.0, 97.0),
        append_retirement_events=True,
    )
    new_items = tuple(
        item
        for item in observer._reference_inventory.values()
        if item.kind.startswith("previous_session_")
    )
    assert {item.source_ids[0] for item in new_items} == {
        "reference_source:session:2025-01-07:high:NQH5:1",
        "reference_source:session:2025-01-07:low:NQH5:1",
    }
    retirements = tuple(
        event
        for event in observer.memory.recent()
        if event.kind is EventKind.LIQUIDITY_RETIRED
        and event.details.get("source_kind", "").startswith(
            "previous_session_"
        )
    )
    assert len(retirements) == 2
    assert {
        event.details["replacement_period"] for event in retirements
    } == {"2025-01-07"}


def test_reference_candidate_publishes_at_admission_with_exact_extreme_bars() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observer._reference_coverage_start = pd.Timestamp(
        "2025-01-05 18:00",
        tz=TZ,
    )

    def reference_candle(start: str, high: float, low: float) -> Candle:
        clock = pd.Timestamp(start, tz=TZ)
        return replace(
            _candle(0, open_=100.0, high=high, low=low, close=100.0),
            start=clock,
            end=clock + pd.Timedelta(minutes=1),
        )

    first = reference_candle("2025-01-06 17:57", 101.0, 99.0)
    extrema = reference_candle("2025-01-06 17:58", 103.0, 97.0)
    equal_extrema = reference_candle("2025-01-06 17:59", 103.0, 97.0)
    admission = reference_candle("2025-01-06 18:00", 102.0, 98.0)
    for candle in (first, extrema, equal_extrema, admission):
        observer._append_completed_bar_event(
            candle,
            atr=1.0,
            data_complete=True,
        )
        observer._advance_reference_periods(
            candle,
            append_retirement_events=True,
        )

    observer._publish_reference_candidate_events()
    observer.memory.flush_audit()
    candidates = {
        event.evidence["source_kind"]: event
        for event in observer.audit_store.events()
        if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED
        and str(event.evidence.get("source_kind", "")).startswith(
            "previous_session_"
        )
    }

    assert set(candidates) == {
        "previous_session_high",
        "previous_session_low",
    }
    for event in candidates.values():
        assert event.known_at == admission.end
        assert event.event_time == extrema.end
        assert event.evidence["reference_extreme_tie_rule"] == (
            "first_completed_m1_at_extreme"
        )
        assert event.evidence["rank"] == "external"
        assert event.evidence["strength"] == pytest.approx(0.75)
        parents = tuple(
            observer.audit_store.get(event_id)
            for event_id in event.source_event_ids
        )
        assert tuple(parent.kind for parent in parents) == (
            EventKind.BAR_COMPLETED,
            EventKind.BAR_COMPLETED,
        )
        assert {parent.known_at for parent in parents} == {
            extrema.end,
            admission.end,
        }


def test_reference_candidate_replacement_is_explicit_and_not_duplicated() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observer._reference_coverage_start = pd.Timestamp(
        "2025-01-05 18:00",
        tz=TZ,
    )

    def reference_candle(start: str, high: float, low: float) -> Candle:
        clock = pd.Timestamp(start, tz=TZ)
        return replace(
            _candle(0, open_=100.0, high=high, low=low, close=100.0),
            start=clock,
            end=clock + pd.Timedelta(minutes=1),
        )

    first_session_tail = reference_candle(
        "2025-01-06 17:59", 101.0, 99.0
    )
    first_admission = reference_candle(
        "2025-01-06 18:00", 102.0, 98.0
    )
    for candle in (first_session_tail, first_admission):
        observer._append_completed_bar_event(
            candle,
            atr=1.0,
            data_complete=True,
        )
        observer._advance_reference_periods(
            candle,
            append_retirement_events=True,
        )
    observer._publish_reference_candidate_events()
    observer.memory.flush_audit()
    first_candidates = {
        event.evidence["source_kind"]: event
        for event in observer.audit_store.events()
        if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED
        and str(event.evidence.get("source_kind", "")).startswith(
            "previous_session_"
        )
    }

    second_admission = reference_candle(
        "2025-01-07 18:00", 104.0, 96.0
    )
    observer._append_completed_bar_event(
        second_admission,
        atr=1.0,
        data_complete=True,
    )
    observer._advance_reference_periods(
        second_admission,
        append_retirement_events=True,
    )
    observer._publish_reference_candidate_events()
    observer._publish_reference_candidate_events()
    observer.memory.flush_audit()
    candidates = tuple(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED
        and str(event.evidence.get("source_kind", "")).startswith(
            "previous_session_"
        )
    )
    replacements = candidates[-2:]

    assert len(candidates) == 4
    assert {
        event.evidence["replaces_level_id"] for event in replacements
    } == {
        event.evidence["level_id"]
        for event in first_candidates.values()
    }
    for event in replacements:
        replaced = first_candidates[event.evidence["source_kind"]]
        assert event.evidence["replaces_level_event_id"] == replaced.event_id
        assert event.context_event_ids == (replaced.event_id,)
        assert tuple(
            observer.audit_store.get(event_id).kind
            for event_id in event.source_event_ids
        ) == (
            EventKind.BAR_COMPLETED,
            EventKind.BAR_COMPLETED,
        )
    assert len(
        {
            event.evidence["level_id"]
            for event in replacements
        }
    ) == 2
    replayed = replay_atomic_market_snapshot(
        observer.audit_store.events(),
        semantic_registry_identity=observer.semantic_registry.identity,
    )
    replayed_session = tuple(
        item
        for item in replayed.timeframe_states[
            Timeframe.M1
        ].liquidity.candidates
        if item.source_kind.startswith("previous_session_")
    )
    assert {
        item.candidate_id for item in replayed_session if item.side == "above"
    } == {
        event.evidence["level_id"]
        for event in replacements
        if event.side == "above"
    }
    assert {
        item.candidate_id for item in replayed_session if item.side == "below"
    } == {
        event.evidence["level_id"]
        for event in replacements
        if event.side == "below"
    }


def test_reference_candidate_cold_bootstrap_keeps_latest_and_reset_clears() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observer._reference_coverage_start = pd.Timestamp(
        "2025-01-05 18:00",
        tz=TZ,
    )

    def reference_candle(start: str, high: float, low: float) -> Candle:
        clock = pd.Timestamp(start, tz=TZ)
        return replace(
            _candle(0, open_=100.0, high=high, low=low, close=100.0),
            start=clock,
            end=clock + pd.Timedelta(minutes=1),
        )

    candles = (
        reference_candle("2025-01-06 17:59", 101.0, 99.0),
        reference_candle("2025-01-06 18:00", 102.0, 98.0),
        reference_candle("2025-01-07 18:00", 103.0, 97.0),
    )
    for candle in candles:
        observer._append_completed_bar_event(
            candle,
            atr=1.0,
            data_complete=True,
        )
        observer._advance_reference_periods(
            candle,
            append_retirement_events=False,
        )

    observer._publish_reference_candidate_events()
    observer.memory.flush_audit()
    session_candidates = tuple(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED
        and str(event.evidence.get("source_kind", "")).startswith(
            "previous_session_"
        )
    )

    assert len(session_candidates) == 2
    assert all(
        ":session:2025-01-07:" in event.evidence["level_id"]
        for event in session_candidates
    )
    assert all(
        "replaces_level_id" not in event.evidence
        for event in session_candidates
    )

    observer._reset_contract_state(
        reason="contract_change_history_reset",
        observed_at=candles[-1].end + pd.Timedelta(minutes=1),
        reset_anomalies=("contract_change_history_reset",),
        boundary_symbol="NQM5",
        boundary_instrument_id=2,
    )

    assert observer._reference_periods == {}
    assert observer._reference_inventory == {}
    assert observer._reference_candidate_sources == {}
    assert observer._candidate_level_event_ids == {}
    assert all(
        not events
        for events in observer._bar_event_ids_by_timeframe.values()
    )
    assert all(
        not events
        for events in observer._real_bar_event_ids_by_timeframe.values()
    )


def test_reference_sr_has_real_source_identity_and_independent_lifecycle() -> None:
    tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(retained_zones=8),
    )
    for index in range(13):
        tracker.on_candle(
            _candle(
                index,
                open_=98.0,
                high=98.5,
                low=97.5,
                close=98.0,
            ),
            (),
        )
    confirmation_bar = _candle(
        13,
        open_=98.0,
        high=150.0,
        low=50.0,
        close=98.0,
    )
    tracker.on_candle(confirmation_bar, ())
    source_id = "reference_source:day:2025-01-06:high:NQH5:1"
    reference = LiquidityInventoryItem(
        item_id="reference:day:2025-01-06:high:NQH5:1",
        timeframe=Timeframe.M1,
        side="above",
        kind="previous_day_high",
        price=100.0,
        lower_bound=100.0,
        upper_bound=100.0,
        formed_at=BASE,
        confirmed_at=confirmation_bar.end,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=(source_id,),
        age_bars=0,
        strength=0.9,
        structural_rank="external",
        visibility_strength=0.9,
    )
    wick_sweep = _candle(
        14,
        open_=99.75,
        high=100.10,
        low=99.50,
        close=99.90,
    )
    tracker.on_candle(
        wick_sweep,
        (),
        reference_sources=(reference,),
    )
    zone = next(
        item
        for item in tracker.snapshot()[0]
        if item.source_kind == "previous_day"
    )

    assert zone.member_swing_ids == ()
    assert zone.source_ids == (source_id,)
    assert zone.lifecycle is SupportResistanceLifecycle.TESTED
    assert zone.touch_count == 2
    assert zone.lower_bound == pytest.approx(99.75)
    assert zone.upper_bound == pytest.approx(100.25)
    assert CausalObserver._inventory_crossed(reference, wick_sweep)

    # Remaining inside one contact episode is not another independent test.
    tracker.on_candle(
        _candle(
            15,
            open_=99.9,
            high=100.05,
            low=99.8,
            close=99.95,
        ),
        (),
        reference_sources=(reference,),
    )
    assert next(
        item
        for item in tracker.snapshot()[0]
        if item.zone_id == zone.zone_id
    ).touch_count == 2

    # A strict close beyond the frozen band, not the earlier wick sweep,
    # breaks S/R; a later close back inside records descriptive reacceptance.
    tracker.on_candle(
        _candle(
            16,
            open_=99.5,
            high=99.6,
            low=99.2,
            close=99.4,
        ),
        (),
        reference_sources=(reference,),
    )
    tracker.on_candle(
        _candle(
            17,
            open_=99.7,
            high=100.0,
            low=99.6,
            close=99.9,
        ),
        (),
        reference_sources=(reference,),
    )
    assert next(
        item
        for item in tracker.snapshot()[0]
        if item.zone_id == zone.zone_id
    ).touch_count == 3
    tracker.on_candle(
        _candle(
            18,
            open_=100.0,
            high=100.6,
            low=99.9,
            close=100.5,
        ),
        (),
        reference_sources=(reference,),
    )
    broken = next(
        item
        for item in tracker.snapshot()[0]
        if item.zone_id == zone.zone_id
    )
    assert broken.lifecycle is SupportResistanceLifecycle.BROKEN
    tracker.on_candle(
        _candle(
            19,
            open_=100.4,
            high=100.5,
            low=100.0,
            close=100.1,
        ),
        (),
        reference_sources=(reference,),
    )
    reaccepted = next(
        item
        for item in tracker.snapshot()[0]
        if item.zone_id == zone.zone_id
    )
    assert reaccepted.lifecycle is SupportResistanceLifecycle.REACCEPTED
    assert tracker.snapshot()[1] == ()

    # A current completed-period source remains queryable after reaching a
    # terminal descriptive lifecycle.  Retention may evict older terminal
    # history, but must not evict the still-live reference and then attempt to
    # recreate it non-causally on a later bar.
    live_record = tracker._zones[zone.zone_id]
    for index in range(7):
        stale_id = f"stale-reference-zone-{index}"
        tracker._zones[stale_id] = replace(
            live_record,
            state=replace(
                reaccepted,
                zone_id=stale_id,
                source_ids=(f"stale-reference-source-{index}",),
            ),
        )
    tracker.on_candle(
        _candle(
            20,
            open_=98.0,
            high=98.5,
            low=97.5,
            close=98.0,
        ),
        (),
        reference_sources=(reference,),
    )
    tracker._prune_zones()
    assert zone.zone_id in tracker._zones
    assert len(tracker._zones) == 7


def test_reference_sr_replacement_retires_old_source_without_fake_swing() -> None:
    tracker = CausalLiquidityTracker(Timeframe.M1)
    first_bar = _candle(
        0,
        open_=100.0,
        high=100.5,
        low=99.5,
        close=100.0,
    )
    tracker.on_candle(first_bar, ())

    def source(identity: str, price: float, confirmed_at: pd.Timestamp):
        return LiquidityInventoryItem(
            item_id=f"reference:{identity}",
            timeframe=Timeframe.M1,
            side="above",
            kind="previous_session_high",
            price=price,
            lower_bound=price,
            upper_bound=price,
            formed_at=BASE,
            confirmed_at=confirmed_at,
            lifecycle=LiquidityInventoryLifecycle.VISIBLE,
            source_ids=(f"reference_source:{identity}",),
            age_bars=0,
            strength=0.75,
            structural_rank="external",
            visibility_strength=0.75,
        )

    old = source("session-old-high", 200.0, first_bar.end)
    second_bar = _candle(
        1,
        open_=100.0,
        high=100.5,
        low=99.5,
        close=100.0,
    )
    tracker.on_candle(second_bar, (), reference_sources=(old,))
    old_zone = next(
        item for item in tracker.snapshot()[0] if old.source_ids[0] in item.source_ids
    )
    assert old_zone.lifecycle is SupportResistanceLifecycle.ACTIVE

    new = source("session-new-high", 210.0, second_bar.end)
    tracker.on_candle(
        _candle(
            2,
            open_=199.75,
            high=201.0,
            low=199.5,
            close=200.75,
        ),
        (),
        reference_sources=(new,),
    )
    zones = tracker.snapshot()[0]
    retired = next(item for item in zones if item.zone_id == old_zone.zone_id)
    replacement = next(
        item for item in zones if new.source_ids[0] in item.source_ids
    )
    assert retired.lifecycle is SupportResistanceLifecycle.RETIRED
    assert retired.broken_at is None
    assert retired.reaccepted_at is None
    assert retired.transition_reason == SUPPORT_RESISTANCE_RETIREMENT_REASON
    assert replacement.member_swing_ids == ()
    assert replacement.lifecycle is SupportResistanceLifecycle.ACTIVE
    assert tracker.snapshot()[1] == ()


def test_reference_sr_bounded_contacts_preserve_frozen_test_evidence() -> None:
    tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(retained_touches=8),
    )
    confirmation = _candle(
        0,
        open_=98.0,
        high=98.5,
        low=97.5,
        close=98.0,
    )
    tracker.on_candle(confirmation, ())
    reference = LiquidityInventoryItem(
        item_id="reference:bounded-high",
        timeframe=Timeframe.M1,
        side="above",
        kind="previous_day_high",
        price=100.0,
        lower_bound=100.0,
        upper_bound=100.0,
        formed_at=BASE,
        confirmed_at=confirmation.end,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("reference_source:bounded-high",),
        age_bars=0,
        strength=0.9,
        structural_rank="external",
        visibility_strength=0.9,
    )
    tracker.on_candle(
        _candle(1, open_=98.0, high=98.5, low=97.5, close=98.0),
        (),
        reference_sources=(reference,),
    )
    first_test_at = None
    for episode in range(12):
        contact_index = 2 + episode * 2
        tracker.on_candle(
            _candle(
                contact_index,
                open_=99.8,
                high=100.1,
                low=99.6,
                close=99.9,
            ),
            (),
            reference_sources=(reference,),
        )
        zone = next(
            item
            for item in tracker.snapshot()[0]
            if item.source_ids == reference.source_ids
        )
        first_test_at = first_test_at or zone.tested_at
        tracker.on_candle(
            _candle(
                contact_index + 1,
                open_=99.0,
                high=99.4,
                low=98.8,
                close=99.0,
            ),
            (),
            reference_sources=(reference,),
        )

    zone = next(
        item
        for item in tracker.snapshot()[0]
        if item.source_ids == reference.source_ids
    )
    assert zone.total_touch_count == 13
    assert len(zone.touch_times) == 8
    assert zone.tested_at == first_test_at == zone.touch_times[1]


def test_reference_sr_rejects_late_or_rewritten_source() -> None:
    tracker = CausalLiquidityTracker(Timeframe.M1)
    confirmation = _candle(
        0,
        open_=98.0,
        high=98.5,
        low=97.5,
        close=98.0,
    )
    tracker.on_candle(confirmation, ())
    reference = LiquidityInventoryItem(
        item_id="reference:causal-high",
        timeframe=Timeframe.M1,
        side="above",
        kind="previous_day_high",
        price=100.0,
        lower_bound=100.0,
        upper_bound=100.0,
        formed_at=BASE,
        confirmed_at=confirmation.end,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("reference_source:causal-high",),
        age_bars=0,
        strength=0.9,
        structural_rank="external",
        visibility_strength=0.9,
    )
    tracker.on_candle(
        _candle(1, open_=98.0, high=98.5, low=97.5, close=98.0),
        (),
    )
    with pytest.raises(
        LiquidityProtocolError,
        match="first real candle",
    ):
        tracker.on_candle(
            _candle(2, open_=98.0, high=98.5, low=97.5, close=98.0),
            (),
            reference_sources=(reference,),
        )

    fresh = CausalLiquidityTracker(Timeframe.M1)
    fresh.on_candle(confirmation, ())
    fresh.on_candle(
        _candle(1, open_=98.0, high=98.5, low=97.5, close=98.0),
        (),
        reference_sources=(reference,),
    )
    rewritten = replace(
        reference,
        price=101.0,
        lower_bound=101.0,
        upper_bound=101.0,
    )
    with pytest.raises(
        LiquidityProtocolError,
        match="rewrite frozen",
    ):
        fresh.on_candle(
            _candle(2, open_=98.0, high=98.5, low=97.5, close=98.0),
            (),
            reference_sources=(rewritten,),
        )

    missing_atr = CausalLiquidityTracker(Timeframe.M1)
    missing_atr.on_candle(confirmation, ())
    missing_atr._strict_prior_atr_by_end.clear()
    with pytest.raises(
        LiquidityProtocolError,
        match="first real candle",
    ):
        missing_atr.on_candle(
            _candle(1, open_=98.0, high=98.5, low=97.5, close=98.0),
            (),
            reference_sources=(reference,),
        )


def _reference_sr_binding_fixture(
    *,
    family: str = "previous_week",
    zone_side: str = "resistance",
    publish: bool = True,
) -> tuple[
    CausalObserver,
    Candle,
    LiquidityInventoryItem,
    SupportResistanceState,
    MarketEvent | None,
]:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    extreme = _candle(
        0,
        open_=99.0,
        high=99.5,
        low=98.5,
        close=99.0,
    )
    admission = _candle(
        1,
        open_=99.0,
        high=99.25,
        low=98.75,
        close=99.0,
    )
    observer.memory.observe_minute(extreme)
    extreme_bar = observer._append_completed_bar_event(
        extreme,
        atr=1.0,
        data_complete=True,
    )
    observer.memory.observe_minute(admission)
    admission_bar = observer._append_completed_bar_event(
        admission,
        atr=1.0,
        data_complete=True,
    )
    suffix = "low" if zone_side == "support" else "high"
    item_side = "below" if zone_side == "support" else "above"
    price = extreme.low if zone_side == "support" else extreme.high
    identity_family = family.removeprefix("previous_")
    source_id = (
        f"reference_source:{identity_family}:fixture:{suffix}:NQH5:1"
    )
    item = LiquidityInventoryItem(
        item_id=f"reference:{identity_family}:fixture:{suffix}:NQH5:1",
        timeframe=Timeframe.M1,
        side=item_side,
        kind=f"{family}_{suffix}",
        price=price,
        lower_bound=price,
        upper_bound=price,
        formed_at=extreme.start,
        confirmed_at=extreme.end,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=(source_id,),
        age_bars=0,
        strength=0.75,
        structural_rank="external",
        visibility_strength=0.75,
    )
    observer._reference_inventory[item.item_id] = item
    candidate = None
    if publish:
        candidate = observer._append_semantic_atomic(
            EventKind.LIQUIDITY_LEVEL_CREATED,
            admission.end,
            Timeframe.M1,
            item.side,
            item.price,
            item.strength,
            (extreme_bar.event_id, admission_bar.event_id),
            {
                "level_id": item.item_id,
                "candidate_only": True,
                "source_kind": item.kind,
                "source_ids": item.source_ids,
                "source_confirmed_at": extreme.end.isoformat(),
                "reference_period_started_at": extreme.start.isoformat(),
                "reference_period_last_completed_at": extreme.end.isoformat(),
                "reference_extreme_at": extreme.end.isoformat(),
                "reference_admitted_at": admission.end.isoformat(),
                "reference_extreme_tie_rule": (
                    "first_completed_m1_at_extreme"
                ),
            },
            event_time=extreme.end,
            zone=(item.price, item.price),
            source_entity_ids=(item.item_id, *item.source_ids),
        )
        observer._candidate_level_event_ids[item.item_id] = candidate.event_id
    zone = SupportResistanceState(
        zone_id=f"reference-zone:{family}:{suffix}",
        timeframe=Timeframe.M1,
        side=zone_side,
        lower_bound=price - 0.25,
        upper_bound=price + 0.25,
        anchor_price=price,
        formed_at=extreme.start,
        confirmed_at=admission.end,
        lifecycle=SupportResistanceLifecycle.ACTIVE,
        member_swing_ids=(),
        touch_times=(admission.end,),
        reaction_magnitudes_atr=(0.0,),
        age_bars=0,
        strength=0.2,
        total_touch_count=1,
        source_kind=family,
        structural_rank="external",
        visibility_strength=1.0,
        source_ids=(source_id,),
    )
    return observer, admission, item, zone, candidate


@pytest.mark.parametrize(
    ("family", "zone_side"),
    tuple(
        (family, side)
        for family in (
            "previous_session",
            "previous_day",
            "previous_week",
        )
        for side in ("resistance", "support")
    ),
)
def test_reference_sr_level_binds_exact_published_reference_candidate(
    family: str,
    zone_side: str,
) -> None:
    observer, candle, _, zone, candidate = _reference_sr_binding_fixture(
        family=family,
        zone_side=zone_side,
    )
    assert candidate is not None
    observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=candle.end,
            bars=1,
            metrics={"atr": 1.0},
            support_resistance=(zone,),
        ),
        newly_completed=True,
        event_clock=candle.end,
    )

    state_event = next(
        item
        for item in observer.memory.recent()
        if item.kind is EventKind.SUPPORT_RESISTANCE_STATE
    )
    created = observer.memory.audit_event_including_pending(
        observer._candidate_level_event_ids[zone.zone_id]
    )
    assert created is not None
    assert state_event.entity_id == zone.zone_id
    assert state_event.source_ids == zone.source_ids
    assert state_event.details["source_kind"] == family
    assert created.source_event_ids == (candidate.event_id,)
    observer.memory.flush_audit()
    assert observer.audit_store.get(created.event_id) == created


@pytest.mark.parametrize(
    "mismatch",
    ("missing", "source_ids", "family", "side", "anchor"),
)
def test_reference_sr_level_rejects_zero_exact_inventory_sources(
    mismatch: str,
) -> None:
    observer, candle, item, zone, _ = _reference_sr_binding_fixture()
    if mismatch == "missing":
        observer._reference_inventory.clear()
    elif mismatch == "source_ids":
        zone = replace(zone, source_ids=("reference_source:different",))
    elif mismatch == "family":
        zone = replace(zone, source_kind="previous_day")
    elif mismatch == "side":
        zone = replace(
            zone,
            side="support",
            lower_bound=zone.anchor_price - 0.25,
            upper_bound=zone.anchor_price + 0.25,
        )
    else:
        zone = replace(
            zone,
            lower_bound=zone.lower_bound + 0.25,
            anchor_price=zone.anchor_price + 0.25,
            upper_bound=zone.upper_bound + 0.25,
        )
    assert item.item_id in observer._reference_inventory or mismatch == "missing"
    with pytest.raises(ValueError, match="exactly one exact inventory source"):
        observer._reference_zone_source_event_ids(
            zone,
            observed_at=candle.end,
        )


def test_reference_sr_level_rejects_ambiguous_inventory_sources() -> None:
    observer, candle, item, zone, _ = _reference_sr_binding_fixture()
    duplicate = replace(item, item_id=f"{item.item_id}:duplicate")
    observer._reference_inventory[duplicate.item_id] = duplicate
    duplicate_candidate = observer._append_semantic_atomic(
        EventKind.LIQUIDITY_LEVEL_CREATED,
        candle.end,
        Timeframe.M1,
        duplicate.side,
        duplicate.price,
        duplicate.strength,
        (),
        {
            "level_id": duplicate.item_id,
            "candidate_only": True,
            "source_kind": duplicate.kind,
            "source_ids": duplicate.source_ids,
        },
        event_time=candle.end,
        zone=(duplicate.price, duplicate.price),
        source_entity_ids=(duplicate.item_id, *duplicate.source_ids),
    )
    observer._candidate_level_event_ids[duplicate.item_id] = (
        duplicate_candidate.event_id
    )

    with pytest.raises(ValueError, match="exactly one exact inventory source"):
        observer._reference_zone_source_event_ids(
            zone,
            observed_at=candle.end,
        )


def test_reference_sr_level_rejects_unpublished_exact_source() -> None:
    observer, candle, _, zone, candidate = _reference_sr_binding_fixture(
        publish=False
    )
    assert candidate is None

    with pytest.raises(ValueError, match="source level is not published"):
        observer._reference_zone_source_event_ids(
            zone,
            observed_at=candle.end,
        )


def test_partial_cold_start_period_never_becomes_previous_session() -> None:
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )

    def reference_candle(start: str) -> Candle:
        clock = pd.Timestamp(start, tz=TZ)
        return replace(
            _candle(0, open_=100.0, high=101.0, low=99.0, close=100.0),
            start=clock,
            end=clock + pd.Timedelta(minutes=1),
        )

    observer._advance_reference_periods(
        reference_candle("2025-01-06 17:59"),
        append_retirement_events=False,
    )
    observer._advance_reference_periods(
        reference_candle("2025-01-06 18:00"),
        append_retirement_events=False,
    )

    assert not any(
        item.kind.startswith("previous_session_")
        for item in observer._reference_inventory.values()
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
        active_timeframes=CORE_TEST_TIMEFRAMES,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
    )
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
            scale_specs=CORE_TEST_SCALE_SPECS,
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

    observer._prior = SimpleNamespace(liquidity_inventory=projected)
    next_bar = _candle(
        3,
        open_=99.75,
        high=99.9,
        low=99.5,
        close=99.75,
    )
    next_update = replace(
        update,
        asof=next_bar.end,
        completed_1m=next_bar,
        histories={Timeframe.M1: (*update.histories[Timeframe.M1], next_bar)},
    )
    later_metadata = replace(
        item,
        age_bars=2,
        structural_rank="external",
        is_protected_swing=True,
        visibility_strength=1.0,
    )
    still_frozen = observer._project_inventory(
        next_update,
        market_observation(asof=next_bar.end).frames,
        (later_metadata,),
    )
    assert still_frozen == projected

    too_late = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
            scale_specs=CORE_TEST_SCALE_SPECS,
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
        active_timeframes=CORE_TEST_TIMEFRAMES,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
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
        observer = CausalObserver(
            ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
        )
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
    observer = CausalObserver(
        ObserverConfig(
            memory_events=2,
            scale_specs=CORE_TEST_SCALE_SPECS,
        )
    )
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
    _append_authoritative_high_swing_root(
        observer,
        swing_id="pool-a",
        pivot_index=-5,
        price=101.0,
    )
    _append_authoritative_high_swing_root(
        observer,
        swing_id="pool-b",
        pivot_index=-2,
        price=101.0,
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
    observer._append_completed_bar_event(
        sweep_candle,
        atr=1.0,
        data_complete=True,
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
    observer._append_completed_bar_event(
        resolution_candle,
        atr=1.0,
        data_complete=True,
    )
    observer._append_projected_pool_resolution_event(
        inventory,
        resolution_candle,
        crossed_at=sweep_candle.end,
    )
    observer.memory.flush_audit()
    terminal = next(
        event
        for event in observer.audit_store.events()
        if (
            event.kind is EventKind.SWEEP_CONFIRMED
            and event.evidence.get("level_id") == inventory.item_id
        )
    )
    terminal_sources = tuple(
        observer.audit_store.get(event_id)
        for event_id in terminal.source_event_ids
    )
    assert tuple(event.kind for event in terminal_sources) == (
        EventKind.LEVEL_PENETRATED,
        EventKind.BAR_COMPLETED,
    )
    assert terminal.direction is Direction.SHORT
    assert terminal_sources[0].direction is Direction.LONG
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
    durations, ages = observer.memory.temporal_metrics(resolved_at)
    terminal_observation = replace(
        base,
        recent_events=observer.memory.recent(),
        event_durations_minutes=durations,
        event_ages_minutes=ages,
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
    overflow_observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
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
            high=100.0,
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
    retired_zone_id = zones[0].zone_id
    retained_pool_id = pools[0].pool_id
    for index in range(5, 13):
        price = 200.0 + 10.0 * index
        source = _swing(
            f"post-pool-zone-{index}",
            side=SwingSide.HIGH,
            price=price,
            confirmed_index=index,
        )
        pool_tracker.on_candle(
            _candle(
                index,
                open_=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
            ),
            (source,),
        )
    zones, pools, inventory = pool_tracker.snapshot()
    assert retired_zone_id not in {item.zone_id for item in zones}
    assert retained_pool_id in {item.pool_id for item in pools}
    retained_pool_item = next(
        item
        for item in inventory
        if item.item_id == f"pool:{retained_pool_id}"
    )
    assert retained_pool_item.lifecycle is LiquidityInventoryLifecycle.CONSUMED

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
            scale_specs=CORE_TEST_SCALE_SPECS,
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
        active_timeframes=CORE_TEST_TIMEFRAMES,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
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
        active_timeframes=CORE_TEST_TIMEFRAMES,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
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
        4, open_=99.75, high=100.0, low=99.5, close=99.75
    )
    fourth_bar = _candle(
        5, open_=99.75, high=100.0, low=99.5, close=99.75
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
                high=100.0,
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
    durations, _ = memory.temporal_metrics(
        terminal_at + pd.Timedelta(minutes=20)
    )
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
    tested_observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
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
    assert len(tested_observer.memory._events) == event_count + 1
    revision_event = tested_observer.memory.recent()[-1]
    assert revision_event.entity_id is None
    assert revision_event.details["state_revision"] is True
    assert revision_event.details["touch_count"] == 3
    assert revision_event.transition_reason == (
        "support_resistance_evidence_revised"
    )
    tested_event = next(iter(tested_observer.memory._events))
    durations, _ = tested_observer.memory.temporal_metrics(
        BASE + pd.Timedelta(minutes=5)
    )
    assert durations[tested_event.event_id] == 4

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
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observer._record_frame_events(frame, newly_completed=True)
    broken_event = next(
        item
        for item in observer.memory.recent()
        if item.entity_id == broken.zone_id
    )
    assert broken_event.ended_at is None
    durations, _ = observer.memory.temporal_metrics(
        broken_at + pd.Timedelta(minutes=4)
    )
    assert durations[broken_event.event_id] == 4

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
                EventKind.LIQUIDITY_CONSUMED,
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
        event.kind is EventKind.LIQUIDITY_CONSUMED
        for event in memory.recent()
    )
    assert tuple(
        event.lifecycle
        for event in memory.timeline("zone:timeline-zone")
    ) == ("active", "tested", "retired")
    durations, ages = memory.temporal_metrics(
        BASE + pd.Timedelta(minutes=20)
    )
    assert durations[active.event_id] == 3
    assert durations[tested.event_id] == 4
    assert durations[retired.event_id] == 0
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
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
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
    durations, ages = memory.temporal_metrics(tested.observed_at)
    assert durations[active.event_id] == 2
    assert ages[tested.event_id] == 3


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
            EventKind.LIQUIDITY_CONSUMED,
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
    durations, ages = memory.temporal_metrics(
        BASE + pd.Timedelta(minutes=3)
    )
    assert zone.event_id not in durations
    assert zone.event_id not in ages
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
            EventKind.LIQUIDITY_CONSUMED,
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
