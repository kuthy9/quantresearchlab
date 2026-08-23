from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.model import (
    Bar,
    BOSLifecycle,
    BOSPostBreakState,
    BOSScope,
    Candle,
    Direction,
    EventKind,
    StructureLifecycle,
    SwingLifecycle,
    SwingRelation,
    SwingSide,
    Timeframe,
    to_primitive,
)
from smc_trader.observation import CausalObserver, ObserverConfig
from smc_trader.structure import StructureConfig, StructureTracker

from .helpers import MODEL_SCALE_SPECS


BASE = pd.Timestamp("2020-01-06 09:30:00", tz="America/New_York")
STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"


def _candle(
    index: int,
    high: float,
    low: float,
    *,
    open_: float | None = None,
    close: float | None = None,
    timeframe: Timeframe = Timeframe.M1,
    synthetic: bool = False,
) -> Candle:
    middle = (high + low) / 2.0
    start = BASE + pd.Timedelta(minutes=index)
    return Candle(
        timeframe=timeframe,
        start=start,
        end=start + pd.Timedelta(minutes=1),
        open=middle if open_ is None else open_,
        high=high,
        low=low,
        close=middle if close is None else close,
        volume=100.0,
        symbol="NQH0",
        instrument_id=1,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=0 if synthetic else 1,
        synthetic_minutes=1 if synthetic else 0,
    )


def _bull_structure_prefix() -> list[Candle]:
    # Confirmed high relations: first high, then HH at index 6.
    # Confirmed low relations: first low, then HL at index 8.
    return [
        _candle(0, 10.0, 8.0),
        _candle(1, 11.0, 9.0),
        _candle(2, 13.0, 10.0),
        _candle(3, 12.0, 9.0),
        _candle(4, 11.0, 8.0),
        _candle(5, 12.0, 9.0),
        _candle(6, 14.0, 10.0),
        _candle(7, 13.0, 10.0),
        _candle(8, 12.0, 9.0),
        _candle(9, 13.0, 10.0),
        _candle(10, 15.0, 11.0),
    ]


def _tracker() -> StructureTracker:
    return StructureTracker(
        Timeframe.M1,
        StructureConfig.from_file(STRUCTURE_PROTOCOL),
    )


def _state(
    tracker: StructureTracker,
    direction: Direction,
):
    _, structures, _ = tracker.snapshot()
    return next(item for item in structures if item.direction is direction)


def test_tick_relation_table_is_exact_and_atr_independent() -> None:
    relation = StructureTracker._relation
    assert relation(SwingSide.HIGH, 101, 100) is SwingRelation.HH
    assert relation(SwingSide.HIGH, 99, 100) is SwingRelation.LH
    assert relation(SwingSide.HIGH, 100, 100) is SwingRelation.EH
    assert relation(SwingSide.LOW, 101, 100) is SwingRelation.HL
    assert relation(SwingSide.LOW, 99, 100) is SwingRelation.LL
    assert relation(SwingSide.LOW, 100, 100) is SwingRelation.EL


def test_structure_rejects_off_grid_detector_candle_before_mutation() -> None:
    tracker = _tracker()
    before = tracker.snapshot()
    malformed = _candle(
        0,
        100.375,
        99.875,
        open_=100.125,
        close=100.125,
    )

    with pytest.raises(ValueError, match="off-grid"):
        tracker.on_candle(malformed)

    assert tracker.last_end is None
    assert tracker.snapshot() == before


def test_swing_span_is_one_for_m1_and_two_for_other_enabled_frames() -> None:
    config = StructureConfig.from_file(STRUCTURE_PROTOCOL)
    assert config.span_for(Timeframe.M1) == 1
    assert all(
        config.span_for(timeframe) == 2
        for timeframe in (
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.H1,
            Timeframe.H4,
        )
    )


def test_swing_span_values_are_registry_driven_not_code_hardcoded() -> None:
    config = StructureConfig.from_file(STRUCTURE_PROTOCOL)
    configured = replace(
        config,
        swing_spans=tuple(
            (timeframe, 1 if timeframe is Timeframe.M5 else span)
            for timeframe, span in config.swing_spans
        ),
    )

    assert configured.span_for(Timeframe.M5) == 1
    assert StructureTracker(Timeframe.M5, configured).swing_span == 1


def test_pending_bos_rearms_when_same_target_gains_structure_scope() -> None:
    tracker = _tracker()
    prefix = _bull_structure_prefix()
    for candle in prefix[:8]:
        tracker.on_candle(candle)

    _, _, before = tracker.snapshot()
    local = next(
        item
        for item in before
        if item.direction is Direction.LONG
        and item.lifecycle is BOSLifecycle.PENDING
        and item.target_price == 14.0
    )
    assert local.scope is BOSScope.LOCAL
    assert local.source_structure_id is None

    for candle in prefix[8:10]:
        tracker.on_candle(candle)

    _, structures, after = tracker.snapshot()
    bull = next(
        item
        for item in structures
        if item.direction is Direction.LONG
    )
    continuation = next(
        item
        for item in after
        if item.direction is Direction.LONG
        and item.lifecycle is BOSLifecycle.PENDING
        and item.target_swing_id == local.target_swing_id
    )
    superseded = next(item for item in after if item.bos_id == local.bos_id)
    assert bull.lifecycle is StructureLifecycle.CONFIRMED
    assert continuation.scope is BOSScope.CONTINUATION
    assert continuation.source_structure_id == bull.structure_id
    assert continuation.bos_id != local.bos_id
    assert continuation.pending_at == prefix[9].end
    assert superseded.lifecycle is BOSLifecycle.FAILED
    assert superseded.resolved_at == prefix[9].end
    assert superseded.failure_reason == "superseded"


@pytest.mark.parametrize("span", [1, 2])
def test_outside_bar_never_confirms_both_swing_sides(span: int) -> None:
    config = StructureConfig.from_file(STRUCTURE_PROTOCOL)
    config = replace(
        config,
        swing_spans=tuple(
            (timeframe, span if timeframe is Timeframe.M1 else value)
            for timeframe, value in config.swing_spans
        ),
    )
    tracker = StructureTracker(Timeframe.M1, config)
    pivot_index = span
    for index in range(span * 2 + 1):
        tracker.on_candle(
            _candle(
                index,
                12.0 if index == pivot_index else 10.0,
                7.0 if index == pivot_index else 9.0,
            )
        )

    swings, _, _ = tracker.snapshot()
    assert not any(
        item.pivot_start == _candle(pivot_index, 12.0, 7.0).start
        and item.lifecycle is SwingLifecycle.CONFIRMED
        for item in swings
    )


def test_structure_and_bos_are_prefix_invariant_and_not_backfilled() -> None:
    prefix = _bull_structure_prefix()
    tracker = _tracker()
    observed_prefixes = []
    for candle in prefix:
        tracker.on_candle(candle)
        observed_prefixes.append(to_primitive(tracker.snapshot()))

    replay = _tracker()
    for index, candle in enumerate(prefix):
        replay.on_candle(candle)
        assert to_primitive(replay.snapshot()) == observed_prefixes[index]

    swings, _, _ = tracker.snapshot()
    confirmed = [
        item for item in swings if item.lifecycle is SwingLifecycle.CONFIRMED
    ]
    assert confirmed
    assert all(item.confirmed_at == item.observed_at for item in confirmed)
    assert all(item.confirmed_at > item.pivot_end for item in confirmed)


def test_bull_structure_locks_and_protected_level_can_only_tighten() -> None:
    tracker = _tracker()
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
    bull = _state(tracker, Direction.LONG)
    assert bull.lifecycle is StructureLifecycle.CONFIRMED
    assert bull.protected_price == 9.0

    # A wick-only LL below protection does not break the locked structure.
    # The next low is locally HL versus that LL but remains below the frozen
    # protected price, so it cannot loosen protection from 9.00 to 8.75.
    later = [
        _candle(11, 12.0, 8.5, open_=9.5, close=9.5),
        _candle(12, 11.5, 9.25, open_=10.25, close=10.25),
        _candle(13, 11.0, 9.0),
        _candle(14, 11.5, 8.75, open_=9.25, close=9.25),
        _candle(15, 12.0, 9.0),
        _candle(16, 12.5, 9.25, open_=10.75, close=10.75),
    ]
    for candle in later:
        tracker.on_candle(candle)
    bull = _state(tracker, Direction.LONG)
    assert bull.lifecycle is StructureLifecycle.CONFIRMED
    assert bull.protected_price == 9.0


def test_wick_attempt_stays_pending_then_later_close_confirms_continuation() -> None:
    tracker = _tracker()
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
    settle = _candle(11, 14.0, 12.5, open_=13.0, close=13.75)
    tracker.on_candle(settle)
    _, _, before = tracker.snapshot()
    pending = next(
        item
        for item in before
        if item.direction is Direction.LONG
        and item.lifecycle is BOSLifecycle.PENDING
        and item.target_price == 15.0
    )

    wick = _candle(12, 15.5, 13.5, open_=14.0, close=14.75)
    tracker.on_candle(wick)
    _, _, after_wick = tracker.snapshot()
    attempted = next(item for item in after_wick if item.bos_id == pending.bos_id)
    assert attempted.lifecycle is BOSLifecycle.PENDING
    assert attempted.attempt_count == pending.attempt_count + 1
    assert attempted.last_attempt_at == wick.end
    assert attempted.attempt_clocks == (
        *pending.attempt_clocks,
        wick.end,
    )

    second_wick = _candle(
        13,
        16.25,
        13.5,
        open_=14.75,
        close=14.50,
    )
    tracker.on_candle(second_wick)
    _, _, after_second_wick = tracker.snapshot()
    twice_attempted = next(
        item
        for item in after_second_wick
        if item.bos_id == pending.bos_id
    )
    assert twice_attempted.lifecycle is BOSLifecycle.PENDING
    assert twice_attempted.attempt_clocks == (
        *attempted.attempt_clocks,
        second_wick.end,
    )

    close_break = _candle(
        14,
        15.75,
        14.0,
        open_=14.75,
        close=15.25,
    )
    tracker.on_candle(close_break)
    _, _, resolved = tracker.snapshot()
    confirmed = next(item for item in resolved if item.bos_id == pending.bos_id)
    assert confirmed.lifecycle is BOSLifecycle.CONFIRMED
    assert confirmed.scope is BOSScope.CONTINUATION
    assert confirmed.resolved_at == close_break.end
    assert confirmed.resolved_at > confirmed.pending_at
    assert confirmed.attempt_clocks == twice_attempted.attempt_clocks
    assert confirmed.break_bar_id == tracker._candle_id(close_break)
    assert confirmed.break_distance_atr is not None
    assert confirmed.post_break_state is BOSPostBreakState.PENDING
    assert confirmed.source_displacement_id is None
    assert confirmed.mss_qualified is False

    accepted_bar = _candle(
        15,
        16.0,
        14.75,
        open_=15.25,
        close=15.50,
    )
    tracker.on_candle(accepted_bar)
    accepted = next(
        item for item in tracker.snapshot()[2] if item.bos_id == pending.bos_id
    )
    assert accepted.post_break_state is BOSPostBreakState.ACCEPTED
    assert accepted.accepted_at == accepted_bar.end
    assert accepted.rejected_at is None


def test_same_clock_attempt_is_only_valid_for_failed_resolution() -> None:
    tracker = _tracker()
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
    tracker.on_candle(_candle(11, 14.0, 12.5, open_=13.0, close=13.75))
    wick = _candle(12, 15.5, 13.5, open_=14.0, close=14.75)
    tracker.on_candle(wick)
    _, _, states = tracker.snapshot()
    attempted = next(
        item
        for item in states
        if item.direction is Direction.LONG
        and item.lifecycle is BOSLifecycle.PENDING
        and item.last_attempt_at == wick.end
    )
    with pytest.raises(ValueError, match="wick-attempt"):
        replace(
            attempted,
            lifecycle=BOSLifecycle.CONFIRMED,
            resolved_at=wick.end,
        )
    failed = replace(
        attempted,
        lifecycle=BOSLifecycle.FAILED,
        resolved_at=wick.end,
        failure_reason="superseded",
    )
    assert failed.attempt_clocks[-1] == failed.resolved_at
    with pytest.raises(ValueError, match="same-clock"):
        replace(
            attempted,
            lifecycle=BOSLifecycle.FAILED,
            resolved_at=wick.end,
            failure_reason="data_gap_reset",
        )
    with pytest.raises(ValueError, match="registered failure"):
        replace(
            attempted,
            lifecycle=BOSLifecycle.FAILED,
            resolved_at=wick.end + pd.Timedelta(minutes=1),
            failure_reason="arbitrary_failure",
        )


def test_opposed_close_breaks_locked_structure_without_cross_timeframe_input() -> None:
    tracker = _tracker()
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
    break_bar = _candle(11, 10.0, 8.0, open_=9.5, close=8.75)
    tracker.on_candle(break_bar)
    bull = _state(tracker, Direction.LONG)
    assert bull.lifecycle is StructureLifecycle.BROKEN
    assert bull.broken_at == break_bar.end
    _, _, breaks = tracker.snapshot()
    opposed = [
        item
        for item in breaks
        if item.direction is Direction.SHORT
        and item.lifecycle is BOSLifecycle.CONFIRMED
        and item.scope is BOSScope.OPPOSED
    ]
    assert opposed
    assert all(item.timeframe is Timeframe.M1 for item in opposed)


def test_failed_bos_rearm_of_same_target_starts_new_generation() -> None:
    tracker = StructureTracker(
        Timeframe.M1,
        StructureConfig.from_file(
            "configs/primitives_structure_liquidity.json"
        ),
    )
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
    _, _, initial = tracker.snapshot()
    original = next(
        item
        for item in initial
        if item.direction is Direction.LONG
        and item.lifecycle is BOSLifecycle.PENDING
        and item.target_price == 14.0
    )

    break_bar = _candle(
        11,
        16.0,
        7.0,
        open_=10.0,
        close=8.5,
    )
    tracker.on_candle(break_bar)
    _, _, after_break = tracker.snapshot()
    failed = next(
        item for item in after_break
        if item.bos_id == original.bos_id
    )
    assert failed.lifecycle is BOSLifecycle.FAILED
    assert failed.resolved_at == break_bar.end
    assert failed.failure_reason == "opposite_structure_break"

    refresh_bars = (
        _candle(12, 16.5, 8.0, open_=9.0, close=9.0),
        _candle(13, 17.0, 8.5, open_=9.0, close=9.5),
    )
    for candle in refresh_bars:
        tracker.on_candle(candle)
    _, _, rearmed_snapshot = tracker.snapshot()
    assert len(rearmed_snapshot) == len(
        {item.bos_id for item in rearmed_snapshot}
    )
    rearmed = next(
        item
        for item in rearmed_snapshot
        if item.direction is Direction.LONG
        and item.lifecycle is BOSLifecycle.PENDING
        and item.target_swing_id == failed.target_swing_id
    )
    assert rearmed.bos_id != failed.bos_id
    assert rearmed.pending_at == refresh_bars[0].end
    assert rearmed.pending_at > failed.resolved_at

    close_break = _candle(
        14,
        15.0,
        9.0,
        open_=13.5,
        close=14.25,
    )
    tracker.on_candle(close_break)
    _, _, resolved = tracker.snapshot()
    assert len(resolved) == len({item.bos_id for item in resolved})
    old_terminal = next(
        item for item in resolved if item.bos_id == failed.bos_id
    )
    confirmed = next(
        item for item in resolved if item.bos_id == rearmed.bos_id
    )
    assert old_terminal.lifecycle is BOSLifecycle.FAILED
    assert confirmed.lifecycle is BOSLifecycle.CONFIRMED
    assert confirmed.resolved_at == close_break.end


def test_long_short_structure_is_symmetric_under_price_mirroring() -> None:
    bull_candles = _bull_structure_prefix()
    bear_candles = [
        replace(
            candle,
            open=200.0 - candle.open,
            high=200.0 - candle.low,
            low=200.0 - candle.high,
            close=200.0 - candle.close,
        )
        for candle in bull_candles
    ]
    bull_tracker = _tracker()
    bear_tracker = _tracker()
    for bull, bear in zip(bull_candles, bear_candles):
        bull_tracker.on_candle(bull)
        bear_tracker.on_candle(bear)
    bull = _state(bull_tracker, Direction.LONG)
    bear = _state(bear_tracker, Direction.SHORT)
    assert bull.lifecycle is StructureLifecycle.CONFIRMED
    assert bear.lifecycle is StructureLifecycle.CONFIRMED
    assert bull.sequence_count == bear.sequence_count
    assert bull.high_run == bear.low_run
    assert bull.low_run == bear.high_run


def test_explicit_reset_clears_structure_and_pending_bos() -> None:
    tracker = _tracker()
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
    assert _state(tracker, Direction.LONG).lifecycle is StructureLifecycle.CONFIRMED
    tracker.reset()
    swings, structures, breaks = tracker.snapshot()
    assert not swings
    assert not breaks
    assert all(
        item.lifecycle is StructureLifecycle.INACTIVE for item in structures
    )


def test_synthetic_market_time_cannot_change_structure_or_semantic_age() -> None:
    prefix = _bull_structure_prefix()
    tracker = _tracker()
    reference = _tracker()
    for candle in prefix:
        tracker.on_candle(candle)
        reference.on_candle(candle)
    before = to_primitive(tracker.snapshot())
    synthetic = _candle(
        11,
        99.0,
        1.0,
        open_=50.0,
        close=75.0,
        synthetic=True,
    )
    tracker.on_candle(synthetic)
    assert tracker.last_end == synthetic.end
    assert to_primitive(tracker.snapshot()) == before

    next_real = _candle(12, 13.5, 10.5, open_=11.5, close=12.0)
    tracker.on_candle(next_real)
    reference.on_candle(next_real)
    assert to_primitive(tracker.snapshot()) == to_primitive(
        reference.snapshot()
    )


def test_boundary_reset_terminalizes_pending_bos_before_clearing_state() -> None:
    tracker = _tracker()
    for candle in _bull_structure_prefix():
        tracker.on_candle(candle)
    _, _, before = tracker.snapshot()
    pending_ids = {
        item.bos_id
        for item in before
        if item.lifecycle is BOSLifecycle.PENDING
    }
    assert pending_ids
    reset_at = BASE + pd.Timedelta(minutes=20)
    failed = tracker.reset_for_boundary(
        reason="data_gap_reset",
        observed_at=reset_at,
    )
    assert {item.bos_id for item in failed} == pending_ids
    assert all(
        item.lifecycle is BOSLifecycle.FAILED
        and item.resolved_at == reset_at
        and item.failure_reason == "data_gap_reset"
        for item in failed
    )
    swings, structures, breaks = tracker.snapshot()
    assert not swings
    assert not breaks
    assert all(
        item.lifecycle is StructureLifecycle.INACTIVE
        for item in structures
    )


def test_observer_records_boundary_terminal_events_before_new_epoch() -> None:
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=MODEL_SCALE_SPECS,
            minimum_bars={
                timeframe: 1 for timeframe in Timeframe
            },
            structure_protocol=STRUCTURE_PROTOCOL,
        )
    )
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observation = None
    for candle in _bull_structure_prefix():
        observation = observer.observe(
            reader.on_bar(
                Bar(
                    start=candle.start,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    symbol=candle.symbol,
                    instrument_id=candle.instrument_id,
                )
            )
        )
    assert observation is not None
    assert any(
        item.lifecycle is BOSLifecycle.PENDING
        for item in observation.frame(Timeframe.M1).structure_breaks
    )
    reset_bar = Bar(
        start=BASE + pd.Timedelta(minutes=20),
        open=12.0,
        high=12.5,
        low=11.5,
        close=12.0,
        volume=100.0,
        symbol="NQH0",
        instrument_id=1,
        data_gap_before_minutes=9,
    )
    reset_observation = observer.observe(reader.on_bar(reset_bar))
    terminal = [
        event
        for event in reset_observation.recent_events
        if event.kind is EventKind.STRUCTURE_BREAK_FAILED
    ]
    assert terminal
    assert all(
        event.details["failure_reason"] == "data_gap_reset"
        and event.observed_at == reset_bar.end
        for event in terminal
    )
    visible_terminal = reset_observation.frame(
        Timeframe.M1
    ).structure_breaks
    assert visible_terminal
    assert all(
        item.lifecycle is BOSLifecycle.FAILED
        and item.resolved_at == reset_bar.end
        and item.failure_reason == "data_gap_reset"
        for item in visible_terminal
    )


def test_simultaneous_gap_and_contract_change_terminalizes_once() -> None:
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=MODEL_SCALE_SPECS,
            minimum_bars={
                timeframe: 1 for timeframe in Timeframe
            },
            structure_protocol=STRUCTURE_PROTOCOL,
        )
    )
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observation = None
    for candle in _bull_structure_prefix():
        observation = observer.observe(
            reader.on_bar(
                Bar(
                    start=candle.start,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    symbol=candle.symbol,
                    instrument_id=candle.instrument_id,
                )
            )
        )
    assert observation is not None
    pending_ids = {
        item.bos_id
        for item in observation.frame(Timeframe.M1).structure_breaks
        if item.lifecycle is BOSLifecycle.PENDING
    }
    assert pending_ids
    reset_bar = Bar(
        start=BASE + pd.Timedelta(minutes=20),
        open=12.0,
        high=12.5,
        low=11.5,
        close=12.0,
        volume=100.0,
        symbol="NQM0",
        instrument_id=2,
        data_gap_before_minutes=9,
    )
    reset = observer.observe(reader.on_bar(reset_bar))
    assert "contract_change_history_reset" in reset.anomalies
    assert "data_gap_history_reset" in reset.anomalies
    terminal = tuple(
        item
        for item in reset.frame(Timeframe.M1).structure_breaks
        if item.lifecycle is BOSLifecycle.FAILED
    )
    assert {item.bos_id for item in terminal} == pending_ids
    assert all(
        item.failure_reason == "contract_change_reset"
        and item.resolved_at == reset_bar.end
        for item in terminal
    )
    terminal_events = tuple(
        event
        for event in reset.recent_events
        if event.kind is EventKind.STRUCTURE_BREAK_FAILED
    )
    assert {
        event.details["bos_id"] for event in terminal_events
    } == pending_ids
    assert all(
        tuple(event.details["reset_anomalies"])
        == (
            "data_gap_history_reset",
            "contract_change_history_reset",
        )
        for event in terminal_events
    )


def test_boundary_terminal_state_survives_observation_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=MODEL_SCALE_SPECS,
            minimum_bars={
                timeframe: 1 for timeframe in Timeframe
            },
            structure_protocol=STRUCTURE_PROTOCOL,
        )
    )
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observation = None
    for candle in _bull_structure_prefix():
        observation = observer.observe(
            reader.on_bar(
                Bar(
                    start=candle.start,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    symbol=candle.symbol,
                    instrument_id=candle.instrument_id,
                )
            )
        )
    assert observation is not None
    pending_ids = {
        item.bos_id
        for item in observation.frame(Timeframe.M1).structure_breaks
        if item.lifecycle is BOSLifecycle.PENDING
    }
    reset_bar = Bar(
        start=BASE + pd.Timedelta(minutes=20),
        open=12.0,
        high=12.5,
        low=11.5,
        close=12.0,
        volume=100.0,
        symbol="NQH0",
        instrument_id=1,
        data_gap_before_minutes=9,
    )
    update = reader.on_bar(reset_bar)
    original = observer._observe_frame

    def fail_once(*args, **kwargs):
        raise RuntimeError("synthetic frame failure")

    monkeypatch.setattr(observer, "_observe_frame", fail_once)
    with pytest.raises(RuntimeError, match="synthetic frame failure"):
        observer.observe(update)
    monkeypatch.setattr(observer, "_observe_frame", original)
    retried = observer.observe(update)
    assert {
        item.bos_id
        for item in retried.frame(Timeframe.M1).structure_breaks
        if item.lifecycle is BOSLifecycle.FAILED
    } == pending_ids
    with pytest.raises(
        ValueError,
        match="successfully committed update",
    ):
        observer.observe(update)
