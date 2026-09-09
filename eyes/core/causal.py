"""Causal one-minute clock and closed-bar multitimeframe aggregation."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from shares.core.market_clock import (
    expected_trading_minutes,
    registered_native_bar_bounds,
    scheduled_gap_kind,
)
from contract.market import (
    Bar,
    Candle,
    Timeframe,
    ticks_to_price,
)
from shares.core.scale_registry import ScaleSpec, scale_registry_id


class CausalClockError(RuntimeError):
    """Raised when a continuous reader would otherwise hide a clock defect."""


@dataclass(frozen=True)
class ReaderUpdate:
    asof: pd.Timestamp
    completed_1m: Candle
    newly_completed: Mapping[Timeframe, tuple[Candle, ...]]
    histories: Mapping[Timeframe, tuple[Candle, ...]]
    anomalies: tuple[str, ...]
    active_timeframes: tuple[Timeframe, ...]
    scale_specs: tuple[ScaleSpec, ...]
    scale_registry_id: str


class _TimeframeAggregator:
    def __init__(
        self,
        timeframe: Timeframe,
        minutes: int,
        anchor_minute: int,
        tick_size: float,
    ) -> None:
        self.timeframe = timeframe
        self.minutes = int(minutes)
        self.anchor_minute = int(anchor_minute)
        self.tick_size = float(tick_size)
        self._bucket_start: pd.Timestamp | None = None
        self._bucket_end: pd.Timestamp | None = None
        self._bars: list[Bar] = []

    def reset(self) -> None:
        self._bucket_start = None
        self._bucket_end = None
        self._bars.clear()

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
        tick_rows = tuple(
            bar.normalized_ohlc_ticks
            for bar in bars
        )
        if any(row is None for row in tick_rows):
            raise AssertionError("aggregator received a non-normalized bar")
        integer_rows = tuple(
            row for row in tick_rows if row is not None
        )
        open_ticks = integer_rows[0][0]
        high_ticks = max(row[1] for row in integer_rows)
        low_ticks = min(row[2] for row in integer_rows)
        close_ticks = integer_rows[-1][3]
        return Candle(
            timeframe=self.timeframe,
            start=self._bucket_start,
            end=self._bucket_end,
            open=ticks_to_price(open_ticks, self.tick_size),
            high=ticks_to_price(high_ticks, self.tick_size),
            low=ticks_to_price(low_ticks, self.tick_size),
            close=ticks_to_price(close_ticks, self.tick_size),
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
            price_tick_size=self.tick_size,
            normalized_ohlc_ticks=(
                open_ticks,
                high_ticks,
                low_ticks,
                close_ticks,
            ),
        )

    def append(
        self,
        bar: Bar,
    ) -> tuple[Candle, ...]:
        start, end = registered_native_bar_bounds(
            bar.start,
            timeframe_minutes=self.minutes,
            anchor_minute=self.anchor_minute,
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
        scale_specs: Sequence[ScaleSpec],
        tick_size: float = 0.25,
    ) -> None:
        self.scale_specs = tuple(scale_specs)
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
        # Validate the grid before allocating mutable reader state.
        ticks_to_price(0, tick_size)
        self.tick_size = float(tick_size)
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
                self.tick_size,
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

    def _one_minute_candle(self, bar: Bar) -> Candle:
        if bar.normalized_ohlc_ticks is None:
            raise AssertionError("reader minute candle requires normalized OHLC")
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
            price_tick_size=self.tick_size,
            normalized_ohlc_ticks=bar.normalized_ohlc_ticks,
        )

    def on_bar(self, bar: Bar) -> ReaderUpdate:
        # Exact grid admission is the first operation. A rejected vendor bar
        # cannot reset histories, advance a clock, or change contract state.
        bar = bar.on_price_grid(self.tick_size)
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
        emitted: dict[Timeframe, tuple[Candle, ...]] = {
            timeframe: () for timeframe in self.active_timeframes
        }
        emitted[Timeframe.M1] = (minute,)
        for timeframe, aggregator in self._aggregators.items():
            candles = aggregator.append(bar)
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
