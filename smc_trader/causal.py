"""Causal one-minute clock and closed-bar multitimeframe aggregation."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from .market_clock import (
    MARKET_TIMEZONE,
    expected_trading_minutes,
    scheduled_gap_kind,
    special_session_close,
)
from .model import Bar, Candle, Timeframe
from .scene_graph import ScaleSpec, legacy_scale_specs, scale_registry_id


class CausalClockError(RuntimeError):
    """Raised when a continuous reader would otherwise hide a clock defect."""


@dataclass(frozen=True)
class ReaderUpdate:
    asof: pd.Timestamp
    completed_1m: Candle
    newly_completed: Mapping[Timeframe, tuple[Candle, ...]]
    histories: Mapping[Timeframe, tuple[Candle, ...]]
    anomalies: tuple[str, ...]
    active_timeframes: tuple[Timeframe, ...] = ()
    scale_specs: tuple[ScaleSpec, ...] = ()
    scale_registry_id: str = "legacy-four-scale"


class _TimeframeAggregator:
    def __init__(self, timeframe: Timeframe, minutes: int, anchor_minute: int) -> None:
        self.timeframe = timeframe
        self.minutes = int(minutes)
        self.anchor_minute = int(anchor_minute)
        self._bucket_start: pd.Timestamp | None = None
        self._bucket_end: pd.Timestamp | None = None
        self._bars: list[Bar] = []

    def reset(self) -> None:
        self._bucket_start = None
        self._bucket_end = None
        self._bars.clear()

    def _bounds(
        self,
        *,
        naive: pd.Timestamp,
        minute_of_day: int,
        special_close: pd.Timestamp | None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        remainder = (minute_of_day - self.anchor_minute) % self.minutes
        start_naive = naive - pd.Timedelta(minutes=remainder)
        end_naive = start_naive + pd.Timedelta(minutes=self.minutes)
        # CME equity-index futures have a scheduled 17:00-18:00 ET maintenance
        # closure. The final nominal 4H bucket is therefore a complete
        # session-aware 14:00-17:00 candle, not a defective 180/240-minute bar.
        if (
            self.timeframe is Timeframe.H4
            and start_naive.hour == 14
            and start_naive.minute == 0
        ):
            end_naive = start_naive.replace(hour=17)
        if special_close is not None:
            special_close_naive = special_close.tz_localize(None)
            if start_naive < special_close_naive < end_naive:
                end_naive = special_close_naive
        start = start_naive.tz_localize(
            MARKET_TIMEZONE, ambiguous=True, nonexistent="shift_forward"
        )
        end = end_naive.tz_localize(
            MARKET_TIMEZONE, ambiguous=True, nonexistent="shift_forward"
        )
        return start, end

    def _build(self) -> Candle:
        if self._bucket_start is None or self._bucket_end is None or not self._bars:
            raise CausalClockError("cannot build an empty timeframe bucket")
        bars = self._bars
        observed = len(bars)
        expected = expected_trading_minutes(self._bucket_start, self._bucket_end)
        exact_clock = (
            bars[0].start == self._bucket_start
            and bars[-1].end == self._bucket_end
            and all(
                left.end == right.start
                or scheduled_gap_kind(left.end, right.start) is not None
                for left, right in zip(bars[:-1], bars[1:])
            )
        )
        return Candle(
            timeframe=self.timeframe,
            start=self._bucket_start,
            end=self._bucket_end,
            open=float(bars[0].open),
            high=float(max(bar.high for bar in bars)),
            low=float(min(bar.low for bar in bars)),
            close=float(bars[-1].close),
            volume=float(sum(bar.volume for bar in bars)),
            symbol=bars[0].symbol,
            instrument_id=bars[0].instrument_id,
            observed_minutes=observed,
            expected_minutes=expected,
            complete=bool(exact_clock and observed == expected),
            real_minutes=sum(
                not bar.synthetic_no_trade for bar in bars
            ),
            synthetic_minutes=sum(
                bar.synthetic_no_trade for bar in bars
            ),
        )

    def append(
        self,
        bar: Bar,
        *,
        naive: pd.Timestamp,
        minute_of_day: int,
        special_close: pd.Timestamp | None,
    ) -> tuple[Candle, ...]:
        start, end = self._bounds(
            naive=naive,
            minute_of_day=minute_of_day,
            special_close=special_close,
        )
        output: list[Candle] = []
        if self._bucket_start is None:
            self._bucket_start, self._bucket_end = start, end
        elif start != self._bucket_start:
            prior = self._build()
            if prior.complete:
                output.append(prior)
            self._bucket_start, self._bucket_end = start, end
            self._bars = []

        if not (self._bucket_start <= bar.start and bar.end <= self._bucket_end):
            raise CausalClockError("1m bar lies outside its computed timeframe bucket")
        if (
            self._bars
            and self._bars[-1].end != bar.start
            and scheduled_gap_kind(self._bars[-1].end, bar.start) is None
        ):
            raise CausalClockError("timeframe bucket received a non-contiguous minute")
        self._bars.append(bar)

        if bar.end == self._bucket_end:
            completed = self._build()
            if completed.complete:
                output.append(completed)
            self._bucket_start = None
            self._bucket_end = None
            self._bars = []
        return tuple(output)


class CausalMarketReader:
    """Consumes exactly one newly completed, contiguous 1m bar per update."""

    def __init__(
        self,
        *,
        maximum_history: int = 1024,
        scale_specs: Sequence[ScaleSpec] | None = None,
    ) -> None:
        self.maximum_history = int(maximum_history)
        self.scale_specs = tuple(
            scale_specs
            if scale_specs is not None
            else legacy_scale_specs(history_limit=self.maximum_history)
        )
        if not self.scale_specs:
            raise ValueError("reader requires at least one scale specification")
        active_specs = tuple(item for item in self.scale_specs if item.enabled)
        active_timeframes = tuple(
            item.native_timeframe for item in active_specs
        )
        if (
            any(timeframe is None for timeframe in active_timeframes)
            or len(active_timeframes) != len(set(active_timeframes))
            or Timeframe.M1 not in active_timeframes
        ):
            raise ValueError("reader enabled scale registry is invalid")
        self.active_timeframes = tuple(
            timeframe
            for timeframe in active_timeframes
            if timeframe is not None
        )
        self._scale_by_timeframe = {
            item.native_timeframe: item
            for item in active_specs
            if item.native_timeframe is not None
        }
        self.scale_registry_id = scale_registry_id(self.scale_specs)
        self._history: dict[Timeframe, deque[Candle]] = {
            timeframe: deque(
                maxlen=self._scale_by_timeframe[timeframe].history_limit
            )
            for timeframe in self.active_timeframes
        }
        self._history_views: dict[Timeframe, tuple[Candle, ...]] = {
            timeframe: ()
            for timeframe in self.active_timeframes
        }
        self._aggregators = {
            timeframe: _TimeframeAggregator(
                timeframe,
                int(self._scale_by_timeframe[timeframe].minutes),
                18 * 60 if timeframe is Timeframe.H4 else 0,
            )
            for timeframe in self.active_timeframes
            if timeframe is not Timeframe.M1
        }
        self._last_bar: Bar | None = None
        self._contract: tuple[str, int] | None = None

    @property
    def last_asof(self) -> pd.Timestamp | None:
        return None if self._last_bar is None else self._last_bar.end

    def reset_contract(self) -> None:
        for aggregator in self._aggregators.values():
            aggregator.reset()
        for history in self._history.values():
            history.clear()
        for timeframe in self.active_timeframes:
            self._history_views[timeframe] = ()

    @staticmethod
    def _one_minute_candle(bar: Bar) -> Candle:
        return Candle(
            timeframe=Timeframe.M1,
            start=bar.start,
            end=bar.end,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            symbol=bar.symbol,
            instrument_id=bar.instrument_id,
            observed_minutes=1,
            expected_minutes=1,
            complete=True,
            real_minutes=0 if bar.synthetic_no_trade else 1,
            synthetic_minutes=1 if bar.synthetic_no_trade else 0,
        )

    def on_bar(self, bar: Bar) -> ReaderUpdate:
        anomalies: list[str] = []
        prior_contract = self._contract
        if self._last_bar is not None:
            if bar.start <= self._last_bar.start:
                raise CausalClockError("duplicate or out-of-order completed 1m bar")
            if bar.data_gap_before_minutes:
                if bar.start == self._last_bar.end:
                    raise CausalClockError(
                        "data-gap marker is inconsistent with a contiguous bar"
                    )
                self.reset_contract()
                anomalies.append("data_gap_history_reset")
            else:
                gap_kind = scheduled_gap_kind(self._last_bar.end, bar.start)
                if bar.start != self._last_bar.end and gap_kind is None:
                    raise CausalClockError(
                        f"non-contiguous completed 1m clock: {self._last_bar.end} -> {bar.start}; "
                        "densify known no-trade minutes or mark the feed invalid"
                    )
                if gap_kind is not None:
                    anomalies.append(gap_kind)

        contract = (bar.symbol, int(bar.instrument_id))
        if prior_contract is not None and contract != prior_contract:
            self.reset_contract()
            anomalies.append("contract_change_history_reset")
        self._contract = contract

        minute = self._one_minute_candle(bar)
        self._history[Timeframe.M1].append(minute)
        local = bar.start.tz_convert(MARKET_TIMEZONE)
        naive = local.tz_localize(None).floor("min")
        minute_of_day = naive.hour * 60 + naive.minute
        special_close = special_session_close(local)
        emitted: dict[Timeframe, tuple[Candle, ...]] = {
            timeframe: () for timeframe in self.active_timeframes
        }
        emitted[Timeframe.M1] = (minute,)
        for timeframe, aggregator in self._aggregators.items():
            candles = aggregator.append(
                bar,
                naive=naive,
                minute_of_day=minute_of_day,
                special_close=special_close,
            )
            emitted[timeframe] = candles
            for candle in candles:
                if candle.end > bar.end:
                    raise AssertionError("aggregator emitted a future candle")
                self._history[timeframe].append(candle)
        for timeframe, candles in emitted.items():
            if candles:
                self._history_views[timeframe] = tuple(
                    self._history[timeframe]
                )

        self._last_bar = bar
        return ReaderUpdate(
            asof=bar.end,
            completed_1m=minute,
            newly_completed=emitted,
            histories=dict(self._history_views),
            anomalies=tuple(anomalies),
            active_timeframes=self.active_timeframes,
            scale_specs=self.scale_specs,
            scale_registry_id=self.scale_registry_id,
        )

    def window(self, timeframe: Timeframe, bars: int) -> tuple[Candle, ...]:
        if bars <= 0 or timeframe not in self._history:
            return ()
        return tuple(self._history[timeframe])[-bars:]
