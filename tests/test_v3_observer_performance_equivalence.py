from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import pickle

import numpy as np
import pandas as pd
import pytest

import smc_trader.causal as causal_module
from smc_trader.causal import (
    CausalClockError,
    CausalMarketReader,
    _TimeframeAggregator,
)
from smc_trader.market_clock import (
    EQUITY_INDEX_CLOSE_OVERRIDES,
    MARKET_TIMEZONE,
    _special_session_close_for_date,
    special_session_close,
)
from smc_trader.liquidity import CausalLiquidityTracker
from smc_trader.model import (
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    Direction,
    FrameObservation,
    StructureLifecycle,
    StructureSequenceState,
    SwingLifecycle,
    SwingPoint,
    SwingRelation,
    SwingSide,
    Timeframe,
    aware_timestamp,
    content_hash,
    to_primitive,
)
from smc_trader.observation import (
    CausalObserver,
    _atr,
    _level_id,
    _level_id_digest,
)
from smc_trader.semantic_discovery_runner import observer_from_model_config
from smc_trader.structure import (
    StructureConfig,
    StructureTracker,
    _copy_with_age,
)
from tests.helpers import session_bars


MODEL_CONFIG = "configs/model_v3_0_exp001_structure_bos_identity.json"


class _FullRebuildObserver(CausalObserver):
    """Test oracle for the pre-optimization valid-stream code path."""

    def _prior_frame_if_unchanged(self, update, timeframe):
        # A scale with no completed candle has one immutable UNKNOWN frame;
        # rebuilding it every 1m heartbeat would fabricate semantic changes.
        if not update.histories[timeframe]:
            return super()._prior_frame_if_unchanged(update, timeframe)
        return None

    def _sync_structure_tracker(
        self,
        update,
        timeframe: Timeframe,
        tracker: StructureTracker,
        liquidity_tracker: CausalLiquidityTracker | None,
    ) -> None:
        history = update.histories[timeframe]
        incoming = (
            history
            if tracker.last_end is None
            else tuple(
                candle
                for candle in history
                if candle.end > tracker.last_end
            )
        )
        for candle in incoming:
            tracker.on_candle(candle)
            if liquidity_tracker is not None:
                swings, _, _ = tracker.snapshot()
                liquidity_tracker.on_candle(candle, swings)
        expected_end = history[-1].end if history else None
        if tracker.last_end != expected_end:
            raise RuntimeError("full-rebuild test oracle clock drift")
        if (
            liquidity_tracker is not None
            and liquidity_tracker.last_end != expected_end
        ):
            raise RuntimeError(
                "full-rebuild liquidity oracle clock drift"
            )


def _full_rebuild_observer() -> _FullRebuildObserver:
    reference = observer_from_model_config(MODEL_CONFIG)
    return _FullRebuildObserver(reference.config)


def test_aware_timestamp_fast_path_preserves_validation() -> None:
    aware = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")
    assert aware_timestamp(aware, name="aware") is aware
    assert aware_timestamp(aware.isoformat(), name="text") == aware
    with pytest.raises(ValueError, match="timezone aware"):
        aware_timestamp(pd.Timestamp("2025-01-06 10:00"), name="naive")


def test_level_id_digest_cache_preserves_timestamp_representation() -> None:
    _level_id_digest.cache_clear()
    local = pd.Timestamp(
        "2025-01-06 10:00",
        tz="America/New_York",
    )
    utc = local.tz_convert("UTC")
    local_first = _level_id(
        Timeframe.M1,
        "above",
        local,
        20_001.25,
    )
    local_second = _level_id(
        Timeframe.M1,
        "above",
        local,
        20_001.25,
    )
    utc_id = _level_id(
        Timeframe.M1,
        "above",
        utc,
        20_001.25,
    )
    assert local_first == local_second
    assert local_first != utc_id
    info = _level_id_digest.cache_info()
    assert info.misses == 2
    assert info.hits == 1
    _level_id_digest.cache_clear()


def test_explicit_frame_validation_remains_fail_closed() -> None:
    swing, structure, bos = _age_states()
    assert swing.confirmed_at is not None
    cutoff = swing.confirmed_at
    future = cutoff + pd.Timedelta(minutes=1)
    cases = (
        (
            {
                "swings": (
                    replace(swing, timeframe=Timeframe.M5),
                ),
            },
            "frame contains a swing from another timeframe",
        ),
        (
            {
                "structures": (
                    replace(structure, timeframe=Timeframe.M5),
                ),
            },
            "frame contains a structure from another timeframe",
        ),
        (
            {
                "structure_breaks": (
                    replace(bos, timeframe=Timeframe.M5),
                ),
            },
            "frame contains a BOS from another timeframe",
        ),
        (
            {
                "swings": (
                    replace(
                        swing,
                        observed_at=future,
                        confirmed_at=future,
                    ),
                ),
            },
            "frame contains a future-known swing state",
        ),
        (
            {
                "structures": (
                    replace(
                        structure,
                        formed_at=future,
                        confirmed_at=future,
                    ),
                ),
            },
            "frame contains a future-known structure state",
        ),
        (
            {
                "structure_breaks": (
                    replace(bos, pending_at=future),
                ),
            },
            "frame contains a future-known BOS state",
        ),
    )
    for additions, message in cases:
        with pytest.raises(ValueError, match=message):
            FrameObservation(
                timeframe=Timeframe.M1,
                cutoff=cutoff,
                bars=1,
                metrics={"finite": 0.0},
                **additions,
            )


def _age_states() -> tuple[
    SwingPoint,
    StructureSequenceState,
    BreakOfStructureState,
]:
    pivot = pd.Timestamp("2025-01-06 09:30", tz="America/New_York")
    confirmed = pivot + pd.Timedelta(minutes=3)
    swing = SwingPoint(
        swing_id="swing",
        timeframe=Timeframe.M1,
        symbol="NQ-SYNTH",
        instrument_id=1,
        side=SwingSide.HIGH,
        price=20_001.0,
        price_ticks=80_004,
        pivot_start=pivot,
        pivot_end=pivot + pd.Timedelta(minutes=1),
        observed_at=confirmed,
        confirmed_at=confirmed,
        lifecycle=SwingLifecycle.CONFIRMED,
        relation=SwingRelation.HH,
        age_bars=0,
    )
    structure = StructureSequenceState(
        structure_id="structure",
        timeframe=Timeframe.M1,
        direction=Direction.LONG,
        lifecycle=StructureLifecycle.CONFIRMED,
        formed_at=confirmed,
        confirmed_at=confirmed,
        broken_at=None,
        high_run=2,
        low_run=2,
        sequence_count=2,
        latest_high_id="swing",
        latest_low_id="low",
        protected_swing_id="low",
        protected_price=19_999.0,
        cumulative_magnitude_atr=2.0,
        age_bars=0,
    )
    bos = BreakOfStructureState(
        bos_id="bos",
        timeframe=Timeframe.M1,
        direction=Direction.LONG,
        lifecycle=BOSLifecycle.PENDING,
        scope=BOSScope.CONTINUATION,
        target_swing_id="swing",
        source_structure_id="structure",
        target_price=20_001.0,
        target_ticks=80_004,
        pending_at=confirmed,
        resolved_at=None,
        age_bars=0,
    )
    return swing, structure, bos


@pytest.mark.parametrize("age", [0, 1, 10_000])
def test_age_copy_matches_validated_replace_exactly(age: int) -> None:
    for original in _age_states():
        candidate = _copy_with_age(original, age)
        expected = replace(original, age_bars=age)
        assert type(candidate) is type(original)
        assert candidate is not original
        assert candidate.__dict__ is not original.__dict__
        assert candidate.__dict__ == expected.__dict__
        assert candidate == expected
        assert to_primitive(candidate) == to_primitive(expected)
        assert content_hash(candidate) == content_hash(expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.dumps(candidate, protocol=protocol) == pickle.dumps(
                expected,
                protocol=protocol,
            )
        with pytest.raises(FrozenInstanceError):
            candidate.age_bars = age + 1


def test_age_copy_rejects_untrusted_type_and_invalid_age() -> None:
    swing, _, _ = _age_states()
    for age in (-1, True, 1.0):
        with pytest.raises(ValueError, match="non-negative integer"):
            _copy_with_age(swing, age)
    with pytest.raises(TypeError, match="untrusted state type"):
        _copy_with_age(object(), 1)
    inactive = StructureSequenceState(
        structure_id=None,
        timeframe=Timeframe.M1,
        direction=Direction.LONG,
        lifecycle=StructureLifecycle.INACTIVE,
        formed_at=None,
        confirmed_at=None,
        broken_at=None,
        high_run=0,
        low_run=0,
        sequence_count=0,
        latest_high_id=None,
        latest_low_id=None,
        protected_swing_id=None,
        protected_price=None,
        cumulative_magnitude_atr=0.0,
        age_bars=0,
    )
    with pytest.raises(ValueError, match="inactive structure"):
        _copy_with_age(inactive, 1)


def _legacy_atr(candles: tuple[Candle, ...], period: int) -> float:
    if not candles:
        return 1.0
    true_ranges: list[float] = []
    prior_close: float | None = None
    for candle in candles:
        if prior_close is None:
            value = candle.high - candle.low
        else:
            value = max(
                candle.high - candle.low,
                abs(candle.high - prior_close),
                abs(candle.low - prior_close),
            )
        if math.isfinite(value):
            true_ranges.append(max(float(value), 0.0))
        prior_close = candle.close
    window = true_ranges[-max(1, int(period)) :]
    positive = [value for value in window if value > 0]
    return float(np.mean(positive)) if positive else 1.0


def _atr_candles(length: int) -> tuple[Candle, ...]:
    start = pd.Timestamp("2025-01-05 18:00", tz="America/New_York")
    output = []
    prior = 20_000.0
    for index in range(length):
        center = 20_000.0 + 0.017 * index + 5.0 * math.sin(index / 7.0)
        if index % 17 == 0:
            open_price = close = high = low = prior
        else:
            open_price = prior
            close = center + 0.2 * math.sin(index / 3.0)
            high = max(open_price, close) + 0.25 + (index % 5) * 0.05
            low = min(open_price, close) - 0.25 - (index % 3) * 0.05
        candle_start = start + pd.Timedelta(minutes=index)
        output.append(
            Candle(
                timeframe=Timeframe.M1,
                start=candle_start,
                end=candle_start + pd.Timedelta(minutes=1),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=100.0 + index,
                symbol="NQ-SYNTH",
                instrument_id=1,
                observed_minutes=1,
                expected_minutes=1,
                complete=True,
            )
        )
        prior = close
    return tuple(output)


@pytest.mark.parametrize("length", [0, 1, 2, 14, 15, 80, 1_024])
@pytest.mark.parametrize("period", [0, 1, 14, 30])
def test_tail_atr_is_float_exact(length: int, period: int) -> None:
    candles = _atr_candles(length)
    assert _atr(candles, period) == _legacy_atr(candles, period)


def test_reader_reuses_only_unchanged_history_tuple_views() -> None:
    reader = CausalMarketReader(maximum_history=1024)
    bars = session_bars(sessions=1)
    first = reader.on_bar(bars[0])
    second = reader.on_bar(bars[1])
    assert second.histories[Timeframe.M1] is not first.histories[Timeframe.M1]
    for timeframe in (Timeframe.M5, Timeframe.H1, Timeframe.H4):
        assert second.histories[timeframe] is first.histories[timeframe]

    update = second
    prior = update
    for bar in bars[2:5]:
        prior = update
        update = reader.on_bar(bar)
    assert update.newly_completed[Timeframe.M5]
    assert update.histories[Timeframe.M5] is not prior.histories[Timeframe.M5]
    assert update.histories[Timeframe.H1] is prior.histories[Timeframe.H1]
    assert update.histories[Timeframe.H4] is prior.histories[Timeframe.H4]


def test_special_session_close_cache_is_clock_exact() -> None:
    special_session_close.cache_clear()
    _special_session_close_for_date.cache_clear()

    first_minute = pd.Timestamp(
        "2025-01-09 08:00",
        tz=MARKET_TIMEZONE,
    )
    second_minute = first_minute + pd.Timedelta(minutes=1)
    expected = pd.Timestamp(
        "2025-01-09 09:30",
        tz=MARKET_TIMEZONE,
    )
    assert special_session_close(first_minute) == expected
    assert special_session_close(second_minute) == expected
    outer = special_session_close.cache_info()
    daily = _special_session_close_for_date.cache_info()
    assert outer.misses == 2
    assert outer.hits == 0
    assert daily.misses == 1
    assert daily.hits == 1

    special_session_close.cache_clear()
    _special_session_close_for_date.cache_clear()


def _legacy_bounds(
    aggregator: _TimeframeAggregator,
    timestamp: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    local = timestamp.tz_convert(MARKET_TIMEZONE)
    naive = local.tz_localize(None).floor("min")
    minute_of_day = naive.hour * 60 + naive.minute
    remainder = (
        minute_of_day - aggregator.anchor_minute
    ) % aggregator.minutes
    start_naive = naive - pd.Timedelta(minutes=remainder)
    end_naive = start_naive + pd.Timedelta(
        minutes=aggregator.minutes
    )
    if (
        aggregator.timeframe is Timeframe.H4
        and start_naive.hour == 14
        and start_naive.minute == 0
    ):
        end_naive = start_naive.replace(hour=17)
    close = special_session_close(local)
    if close is not None:
        close_naive = close.tz_localize(None)
        if start_naive < close_naive < end_naive:
            end_naive = close_naive
    return (
        start_naive.tz_localize(
            MARKET_TIMEZONE,
            ambiguous=True,
            nonexistent="shift_forward",
        ),
        end_naive.tz_localize(
            MARKET_TIMEZONE,
            ambiguous=True,
            nonexistent="shift_forward",
        ),
    )


def test_shared_clock_projection_matches_independent_formula() -> None:
    timestamps = [
        pd.Timestamp(value, tz=MARKET_TIMEZONE)
        for value in (
            "2025-01-06 00:00",
            "2025-01-06 09:59",
            "2025-01-06 14:00",
            "2025-01-06 16:59",
            "2025-01-09 09:29",
            "2025-03-09 01:59",
            "2025-03-09 03:00",
        )
    ]
    timestamps.extend(
        pd.Timestamp("2025-11-02 01:30").tz_localize(
            MARKET_TIMEZONE,
            ambiguous=value,
        )
        for value in (True, False)
    )
    aggregators = (
        _TimeframeAggregator(Timeframe.M5, 5, 0),
        _TimeframeAggregator(Timeframe.H1, 60, 0),
        _TimeframeAggregator(Timeframe.H4, 240, 18 * 60),
    )
    for timestamp in timestamps:
        local = timestamp.tz_convert(MARKET_TIMEZONE)
        naive = local.tz_localize(None).floor("min")
        minute_of_day = naive.hour * 60 + naive.minute
        close = special_session_close(local)
        for aggregator in aggregators:
            assert aggregator._bounds(
                naive=naive,
                minute_of_day=minute_of_day,
                special_close=close,
            ) == _legacy_bounds(aggregator, timestamp)


def test_reader_computes_special_close_once_per_accepted_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original = causal_module.special_session_close

    def counted(timestamp):
        calls.append(timestamp)
        return original(timestamp)

    monkeypatch.setattr(
        causal_module,
        "special_session_close",
        counted,
    )
    bars = session_bars(sessions=1)
    reader = CausalMarketReader(maximum_history=1024)
    reader.on_bar(bars[0])
    assert len(calls) == 1
    with pytest.raises(CausalClockError, match="duplicate"):
        reader.on_bar(bars[0])
    assert len(calls) == 1
    reader.on_bar(
        replace(
            bars[10],
            data_gap_before_minutes=9,
        )
    )
    assert len(calls) == 2
    ordinary = [
        pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
        pd.Timestamp("2025-01-11 10:00", tz="America/New_York"),
        pd.Timestamp("2025-03-09 01:30", tz="America/New_York"),
    ]
    folds = [
        pd.Timestamp("2025-11-02 01:30").tz_localize(
            "America/New_York",
            ambiguous=value,
        )
        for value in (True, False)
    ]
    overrides = [
        pd.Timestamp(
            year=value.year,
            month=value.month,
            day=value.day,
            hour=12,
            tz="America/New_York",
        )
        for value in EQUITY_INDEX_CLOSE_OVERRIDES
    ]
    for timestamp in ordinary + folds + overrides:
        assert special_session_close(timestamp) == (
            special_session_close.__wrapped__(timestamp)
        )

    instant = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")
    special_session_close.cache_clear()
    first = special_session_close(instant)
    second = special_session_close(instant.tz_convert("UTC"))
    info = special_session_close.cache_info()
    assert first == second
    assert info.misses == 1
    assert info.hits == 1
    with pytest.raises(TypeError):
        special_session_close(pd.Timestamp("2025-01-06 10:00"))
    special_session_close.cache_clear()


def test_cached_and_full_rebuild_observers_are_minute_exact() -> None:
    reader = CausalMarketReader(maximum_history=1024)
    cached = observer_from_model_config(MODEL_CONFIG)
    rebuilt = _full_rebuild_observer()
    for bar in session_bars(sessions=1):
        update = reader.on_bar(bar)
        cached_observation = cached.observe(update)
        rebuilt_observation = rebuilt.observe(update)
        assert cached_observation == rebuilt_observation
        assert tuple(cached.memory._events) == tuple(rebuilt.memory._events)
        assert cached.memory._closed_durations == rebuilt.memory._closed_durations
        assert cached.memory._latest_by_state == rebuilt.memory._latest_by_state
        assert (
            cached.memory.entity_timelines()
            == rebuilt.memory.entity_timelines()
        )
        assert set(cached_observation.retained_entity_timelines) == (
            cached._retained_timeline_keys(
                cached_observation.frames,
                cached_observation.liquidity_pool_states,
            )
        )


def test_optimized_reader_observer_pickle_resume_is_exact() -> None:
    bars = session_bars(sessions=1)
    reader = CausalMarketReader(maximum_history=1024)
    observer = observer_from_model_config(MODEL_CONFIG)
    baseline = []
    checkpoint = 317
    state = None
    for ordinal, bar in enumerate(bars):
        update = reader.on_bar(bar)
        baseline.append(observer.observe(update))
        if ordinal == checkpoint:
            state = pickle.dumps(
                (reader, observer),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    assert state is not None

    special_session_close.cache_clear()
    _special_session_close_for_date.cache_clear()
    _level_id_digest.cache_clear()
    resumed_reader, resumed_observer = pickle.loads(state)
    resumed = []
    for bar in bars[checkpoint + 1 :]:
        update = resumed_reader.on_bar(bar)
        resumed.append(resumed_observer.observe(update))
    assert resumed == baseline[checkpoint + 1 :]
    assert tuple(resumed_observer.memory._events) == tuple(observer.memory._events)
    assert (
        resumed_observer.memory._closed_durations
        == observer.memory._closed_durations
    )
    assert (
        resumed_observer.memory.entity_timelines()
        == observer.memory.entity_timelines()
    )


def test_late_tracker_bootstrap_reaches_history_tail() -> None:
    bars = session_bars(sessions=1)[:360]
    reader = CausalMarketReader(maximum_history=1024)
    continuous = observer_from_model_config(MODEL_CONFIG)
    update = None
    continuous_observation = None
    for bar in bars:
        update = reader.on_bar(bar)
        continuous_observation = continuous.observe(update)
    assert update is not None
    assert continuous_observation is not None
    assert (
        continuous_observation.incomplete_entity_timeline_keys
        == ()
    )

    late = observer_from_model_config(MODEL_CONFIG)
    late_observation = late.observe(update)
    assert late_observation.frames == continuous_observation.frames
    assert late_observation.incomplete_entity_timeline_keys
    assert (
        "clock_incomplete_entity_timeline"
        in late_observation.anomalies
    )
    for timeframe, tracker in late._structure_trackers.items():
        history = update.histories[timeframe]
        assert tracker.last_end == (
            history[-1].end if history else None
        )


def test_snapshot_swings_remain_in_frozen_sort_order() -> None:
    reader = CausalMarketReader(maximum_history=1024)
    tracker = StructureTracker(
        Timeframe.M1,
        StructureConfig.from_file(
            "configs/smc_primitives_v3_0_structure_bos.json",
        ),
    )
    for bar in session_bars(sessions=1):
        update = reader.on_bar(bar)
        tracker.sync(update.newly_completed[Timeframe.M1])
        swings, _, _ = tracker.snapshot()
        keys = tuple(
            (item.observed_at, item.pivot_start, item.side.value)
            for item in swings
        )
        assert keys == tuple(
            sorted(keys)
        )
