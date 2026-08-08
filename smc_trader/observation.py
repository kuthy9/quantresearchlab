"""Descriptive multitimeframe observation and bounded causal event memory."""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .causal import ReaderUpdate
from .displacement import DisplacementLifecycle, DisplacementProtocol
from .displacement_observer import (
    CONTRACT_BOUNDARY,
    DATA_GAP_BOUNDARY,
    REGISTERED_CLOSURE_ANOMALIES,
    CausalDisplacementEye,
)
from .group3 import (
    CausalGroup3Tracker,
    FVG_BOUNDARY_REASONS,
    Group3BOSSource,
    Group3Protocol,
    Group3Update,
)
from .group4 import (
    CausalGroup4Tracker,
    Group4Protocol,
    Group4Update,
)
from .group5 import (
    CausalGroup5Reducer,
    Group5Protocol,
    Group5Update,
)
from .liquidity import (
    CausalLiquidityTracker,
    LiquidityConfig,
    LiquidityProtocolError,
)
from .model import (
    BOS_CONFIRMATION_REASON,
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    CandleStructureState,
    CORE_TIMEFRAMES,
    DealingRangeLifecycle,
    DealingRangeState,
    Direction,
    EventKind,
    ExecutionObservation,
    FairValueGapLifecycle,
    FrameObservation,
    GROUP4_HARD_BOUNDARY_REASONS,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    ManipulationLifecycle,
    ManipulationState,
    MarketEvent,
    MarketObservation,
    OrderBlockLifecycle,
    PathSequenceLifecycle,
    PathSequenceState,
    StructureLifecycle,
    SupportResistanceLifecycle,
    SwingLifecycle,
    SwingRelation,
    Timeframe,
    clamp,
    typed_event_entity_key,
)
from .scene_graph import (
    ScaleSpec,
    SceneGraphDelta,
    TemporalMarketSceneGraph,
    scale_registry_id,
)
from .structure import StructureConfig, StructureTracker


@dataclass(frozen=True)
class ObserverConfig:
    atr_period: int = 14
    memory_events: int = 512
    minimum_bars: Mapping[Timeframe, int] = field(
        default_factory=lambda: {
            Timeframe.H4: 16,
            Timeframe.H1: 24,
            Timeframe.M15: 24,
            Timeframe.M5: 24,
            Timeframe.M1: 30,
        }
    )
    tick_size: float = 0.25
    point_value: float = 20.0
    structure_protocol: str | None = None
    liquidity_protocol: str | None = None
    displacement_protocol: str | None = None
    group3_protocol: str | None = None
    group4_protocol: str | None = None
    group5_protocol: str | None = None
    scale_specs: tuple[ScaleSpec, ...] = ()
    project_scene_graph: bool = True
    materialize_event_view: bool = True


@dataclass(frozen=True)
class ExecutionRealityInput:
    spread_points: float | None = None
    expected_slippage_points: float = 0.25
    commission_per_contract_per_side: float = 2.25
    quantity: int = 1
    deadline: pd.Timestamp | None = None
    data_age_seconds: float = 0.0
    size_available: float | None = None
    source: str = "missing"
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    depth_imbalance: float | None = None
    anomalies: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ReferencePeriod:
    """Incremental completed-period extrema; never uses an unfinished bar."""

    key: str
    started_at: pd.Timestamp
    last_end: pd.Timestamp
    high: float
    low: float
    symbol: str
    instrument_id: int
    coverage_complete: bool


_REFERENCE_KINDS = {
    "previous_session_high",
    "previous_session_low",
    "previous_day_high",
    "previous_day_low",
    "previous_week_high",
    "previous_week_low",
}

_INVENTORY_KINDS = {
    "swing",
    "equal_highs",
    "equal_lows",
    *_REFERENCE_KINDS,
}


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not math.isfinite(float(denominator)) or abs(float(denominator)) < 1e-12:
        return float(default)
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else float(default)


def _signed_tanh(value: float) -> float:
    return math.tanh(float(value)) if math.isfinite(float(value)) else 0.0


def _atr(candles: Sequence[Candle], period: int) -> float:
    real_candles = tuple(
        candle for candle in candles if candle.real_completed
    )
    if not real_candles:
        return 1.0
    limit = max(1, int(period))
    reverse_window: list[float] = []
    for index in range(len(real_candles) - 1, -1, -1):
        candle = real_candles[index]
        if index == 0:
            value = candle.high - candle.low
        else:
            prior_close = real_candles[index - 1].close
            value = max(
                candle.high - candle.low,
                abs(candle.high - prior_close),
                abs(candle.low - prior_close),
            )
        if math.isfinite(value):
            reverse_window.append(max(float(value), 0.0))
            if len(reverse_window) == limit:
                break
    window = list(reversed(reverse_window))
    positive = [value for value in window if value > 0]
    return float(np.mean(positive)) if positive else 1.0


def _strict_prior_atr(
    candles: Sequence[Candle],
    current: Candle,
    period: int,
) -> float | None:
    """ATR from real bars completed before ``current``; never includes it."""

    prior = tuple(
        candle
        for candle in candles
        if candle.real_completed and candle.end <= current.start
    )
    if not prior:
        return None
    limit = max(1, int(period))
    reverse_window: list[float] = []
    for index in range(len(prior) - 1, -1, -1):
        candle = prior[index]
        if index == 0:
            value = candle.high - candle.low
        else:
            prior_close = prior[index - 1].close
            value = max(
                candle.high - candle.low,
                abs(candle.high - prior_close),
                abs(candle.low - prior_close),
            )
        if math.isfinite(value) and value > 0.0:
            reverse_window.append(float(value))
            if len(reverse_window) == limit:
                break
    return (
        float(np.mean(tuple(reversed(reverse_window))))
        if reverse_window
        else None
    )


def _efficiency(closes: Sequence[float]) -> float:
    if len(closes) < 2:
        return 0.0
    travel = float(np.abs(np.diff(np.asarray(closes, dtype=float))).sum())
    return clamp(abs(float(closes[-1]) - float(closes[0])) / max(travel, 1e-12))


def _range_position(price: float, low: float, high: float) -> float:
    return clamp(_safe_div(price - low, high - low, 0.5))


def _candle_structure(
    candle: Candle,
    *,
    prior_atr: float | None = None,
) -> CandleStructureState:
    candle_range = max(0.0, float(candle.high - candle.low))
    body = abs(float(candle.close - candle.open))
    upper = max(
        0.0,
        float(candle.high - max(candle.open, candle.close)),
    )
    lower = max(
        0.0,
        float(min(candle.open, candle.close) - candle.low),
    )
    zero_range = candle_range <= 1e-12
    if zero_range:
        body_ratio = 0.0
        upper_ratio = 0.0
        lower_ratio = 0.0
        close_location = 0.5
        direction = 0
    else:
        body_ratio = body / candle_range
        upper_ratio = upper / candle_range
        lower_ratio = lower / candle_range
        close_location = (candle.close - candle.low) / candle_range
        direction = (
            1 if candle.close > candle.open
            else -1 if candle.close < candle.open
            else 0
        )

    # Fixed descriptive buckets only.  Small/large require both relative
    # geometry and a strictly prior ATR comparison; no prior ATR means normal.
    valid_prior_atr = (
        prior_atr is not None
        and math.isfinite(float(prior_atr))
        and float(prior_atr) > 0.0
    )
    if zero_range or body_ratio <= 0.10:
        body_class = "doji"
    elif (
        valid_prior_atr
        and body_ratio <= 0.35
        and body <= 0.35 * float(prior_atr)
    ):
        body_class = "small"
    elif (
        valid_prior_atr
        and body_ratio >= 0.65
        and body >= 0.75 * float(prior_atr)
    ):
        body_class = "large"
    else:
        body_class = "normal"

    if zero_range:
        range_class = "compressed"
    elif not valid_prior_atr:
        range_class = "normal"
    elif candle_range <= 0.75 * float(prior_atr):
        range_class = "compressed"
    elif candle_range >= 1.50 * float(prior_atr):
        range_class = "expanded"
    else:
        range_class = "normal"

    # A wick must occupy at least 20% of the range and be 1.5x its peer to
    # dominate; two material but similar wicks are balanced.
    if zero_range or max(upper_ratio, lower_ratio) < 0.20:
        dominant_wick = "none"
    elif upper >= 1.50 * lower:
        dominant_wick = "upper"
    elif lower >= 1.50 * upper:
        dominant_wick = "lower"
    else:
        dominant_wick = "balanced"

    if close_location >= 0.75:
        close_class = "near_high"
    elif close_location <= 0.25:
        close_class = "near_low"
    else:
        close_class = "middle"
    anomalies = ()
    if not candle.real_completed:
        anomalies = ("synthetic_or_partial_completed_candle",)
    return CandleStructureState(
        timeframe=candle.timeframe,
        start=candle.start,
        observed_at=candle.end,
        range_points=candle_range,
        body_points=body,
        upper_wick_points=upper,
        lower_wick_points=lower,
        body_ratio=body_ratio,
        upper_wick_ratio=upper_ratio,
        lower_wick_ratio=lower_ratio,
        close_location=close_location,
        direction=direction,
        real_completed=candle.real_completed,
        zero_range=zero_range,
        body_class=body_class,
        range_class=range_class,
        dominant_wick=dominant_wick,
        close_class=close_class,
        anomalies=anomalies,
    )


def _external_distances(
    inventory: Sequence[LiquidityInventoryItem], price: float, atr: float
) -> tuple[float, float, int, int]:
    above = [
        item
        for item in inventory
        if item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
        and item.side == "above"
        and item.price > price
    ]
    below = [
        item
        for item in inventory
        if item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
        and item.side == "below"
        and item.price < price
    ]
    above_distance = (
        min((item.price - price) / atr for item in above) if above else 99.0
    )
    below_distance = (
        min((price - item.price) / atr for item in below) if below else 99.0
    )
    return (
        float(min(99.0, max(0.0, above_distance))),
        float(min(99.0, max(0.0, below_distance))),
        len(above),
        len(below),
    )


def _blank_metrics(timeframe: Timeframe, price: float) -> dict[str, float]:
    if timeframe is Timeframe.H4:
        return {
            "directional_displacement": 0.0,
            "path_efficiency": 0.0,
            "structure_direction": 0.0,
            "structure_age_bars": 0.0,
            "range_position": 0.5,
            "rolling_range_low": price,
            "rolling_range_high": price,
            "external_above_distance_atr": 99.0,
            "external_below_distance_atr": 99.0,
            "external_above_count": 0.0,
            "external_below_count": 0.0,
            "atr": 1.0,
        }
    if timeframe in {Timeframe.H1, Timeframe.M15}:
        return {
            "swing_high_progression": 0.0,
            "swing_low_progression": 0.0,
            "swing_progression": 0.0,
            "acceptance_direction": 0.0,
            "rejection_direction": 0.0,
            "rejection_high": price,
            "rejection_low": price,
            "rolling_range_low": price,
            "rolling_range_high": price,
            "rolling_range_position": 0.5,
            "up_path_obstruction_atr": 99.0,
            "down_path_obstruction_atr": 99.0,
            "atr": 1.0,
        }
    if timeframe is Timeframe.M5:
        return {
            "compression": 0.0,
            "atr": 1.0,
        }
    return {
        "bar_progression_direction": 0.0,
        "bar_progression_run_bars": 0.0,
        "bar_progression_net_atr": 0.0,
        "acceleration": 0.0,
        "counter_pressure": 0.0,
        "atr": 1.0,
    }


def _observe_4h(
    candles: Sequence[Candle], asof: pd.Timestamp, config: ObserverConfig
) -> FrameObservation:
    price = candles[-1].close if candles else 0.0
    if not candles:
        return FrameObservation(
            Timeframe.H4, asof, 0, _blank_metrics(Timeframe.H4, price), ready=False
        )
    atr = _atr(candles, config.atr_period)
    closes = [candle.close for candle in candles[-7:]]
    displacement = _signed_tanh((closes[-1] - closes[0]) / atr) if len(closes) >= 2 else 0.0
    recent = list(candles)[-20:]
    low = min(candle.low for candle in recent)
    high = max(candle.high for candle in recent)
    metrics = {
        "directional_displacement": displacement,
        "path_efficiency": _efficiency(closes),
        "structure_direction": 0.0,
        "structure_age_bars": 0.0,
        "range_position": _range_position(price, low, high),
        "rolling_range_low": float(low),
        "rolling_range_high": float(high),
        "external_above_distance_atr": 99.0,
        "external_below_distance_atr": 99.0,
        "external_above_count": 0.0,
        "external_below_count": 0.0,
        "atr": atr,
    }
    return FrameObservation(
        timeframe=Timeframe.H4,
        cutoff=candles[-1].end,
        bars=len(candles),
        metrics=metrics,
        ready=len(candles) >= config.minimum_bars[Timeframe.H4],
    )


def _observe_1h(
    candles: Sequence[Candle],
    asof: pd.Timestamp,
    config: ObserverConfig,
    *,
    timeframe: Timeframe = Timeframe.H1,
) -> FrameObservation:
    price = candles[-1].close if candles else 0.0
    if not candles:
        return FrameObservation(
            timeframe, asof, 0, _blank_metrics(timeframe, price), ready=False
        )
    atr = _atr(candles, config.atr_period)
    reference = list(candles)[-13:-1]
    last = candles[-1]
    prior_high = max((candle.high for candle in reference), default=last.high)
    prior_low = min((candle.low for candle in reference), default=last.low)
    accepted_above = max(0.0, (last.close - prior_high) / atr)
    accepted_below = max(0.0, (prior_low - last.close) / atr)
    acceptance = _signed_tanh(accepted_above - accepted_below)
    rejection = 0.0
    if last.high > prior_high and last.close <= prior_high:
        rejection -= clamp((last.high - prior_high) / atr)
    if last.low < prior_low and last.close >= prior_low:
        rejection += clamp((prior_low - last.low) / atr)
    range_low = min(candle.low for candle in candles[-24:])
    range_high = max(candle.high for candle in candles[-24:])
    metrics = {
        "swing_high_progression": 0.0,
        "swing_low_progression": 0.0,
        "swing_progression": 0.0,
        "acceptance_direction": acceptance,
        "rejection_direction": float(np.clip(rejection, -1.0, 1.0)),
        "rejection_high": float(last.high),
        "rejection_low": float(last.low),
        "rolling_range_low": float(range_low),
        "rolling_range_high": float(range_high),
        "rolling_range_position": _range_position(price, range_low, range_high),
        "up_path_obstruction_atr": 99.0,
        "down_path_obstruction_atr": 99.0,
        "atr": atr,
    }
    return FrameObservation(
        timeframe=timeframe,
        cutoff=candles[-1].end,
        bars=len(candles),
        metrics=metrics,
        ready=len(candles) >= config.minimum_bars.get(timeframe, 24),
    )


def _observe_5m(
    candles: Sequence[Candle], asof: pd.Timestamp, config: ObserverConfig
) -> FrameObservation:
    price = candles[-1].close if candles else 0.0
    if not candles:
        return FrameObservation(
            Timeframe.M5, asof, 0, _blank_metrics(Timeframe.M5, price), ready=False
        )
    atr = _atr(candles, config.atr_period)
    ranges = np.asarray([candle.high - candle.low for candle in candles[-9:]], dtype=float)
    recent_range = float(np.mean(ranges[-3:])) if len(ranges) >= 3 else atr
    prior_range = float(np.mean(ranges[:-3])) if len(ranges) > 3 else atr
    compression = clamp(1.0 - _safe_div(recent_range, prior_range, 1.0))
    metrics = {
        "compression": compression,
        "atr": atr,
    }
    return FrameObservation(
        timeframe=Timeframe.M5,
        cutoff=candles[-1].end,
        bars=len(candles),
        metrics=metrics,
        ready=len(candles) >= config.minimum_bars[Timeframe.M5],
    )


def _observe_1m(
    candles: Sequence[Candle], asof: pd.Timestamp, config: ObserverConfig
) -> FrameObservation:
    price = candles[-1].close if candles else 0.0
    if not candles:
        return FrameObservation(
            Timeframe.M1, asof, 0, _blank_metrics(Timeframe.M1, price), ready=False
        )
    atr = _atr(candles, config.atr_period)
    closes = np.asarray([candle.close for candle in candles[-7:]], dtype=float)
    changes = np.diff(closes)
    recent = float(np.mean(changes[-3:])) if len(changes) >= 3 else 0.0
    prior = float(np.mean(changes[:-3])) if len(changes) > 3 else 0.0
    acceleration = _signed_tanh((recent - prior) / atr)
    progression_direction = 0
    progression_run = 0
    progression_net_atr = 0.0
    if len(changes):
        latest_change = float(changes[-1])
        progression_direction = (
            1 if latest_change > 0.0 else -1 if latest_change < 0.0 else 0
        )
        if progression_direction:
            for change in reversed(changes):
                if (
                    (change > 0.0 and progression_direction > 0)
                    or (change < 0.0 and progression_direction < 0)
                ):
                    progression_run += 1
                else:
                    break
            progression_net_atr = float(
                closes[-1] - closes[-progression_run - 1]
            ) / atr
    pressure: list[float] = []
    for candle in candles[-6:]:
        body_high = max(candle.open, candle.close)
        body_low = min(candle.open, candle.close)
        upper = candle.high - body_high
        lower = body_low - candle.low
        pressure.append(_safe_div(lower - upper, candle.high - candle.low, 0.0))
    metrics = {
        "bar_progression_direction": float(progression_direction),
        "bar_progression_run_bars": float(progression_run),
        "bar_progression_net_atr": progression_net_atr,
        "acceleration": acceleration,
        "counter_pressure": float(np.clip(np.mean(pressure), -1.0, 1.0)),
        "atr": atr,
    }
    return FrameObservation(
        timeframe=Timeframe.M1,
        cutoff=candles[-1].end,
        bars=len(candles),
        metrics=metrics,
        ready=len(candles) >= config.minimum_bars[Timeframe.M1],
    )


class EventMemory:
    _GROUP4_CREATION_SEQUENCE_FLOOR = 1_000_000
    _TIMELINE_TRANSITIONS: Mapping[
        str,
        Mapping[str, frozenset[str]],
    ] = {
        "swing": {
            SwingLifecycle.FORMING.value: frozenset(
                {
                    SwingLifecycle.CONFIRMED.value,
                    SwingLifecycle.FORMATION_FAILED.value,
                }
            ),
            SwingLifecycle.CONFIRMED.value: frozenset(
                {SwingLifecycle.BROKEN.value}
            ),
            SwingLifecycle.BROKEN.value: frozenset(),
            SwingLifecycle.FORMATION_FAILED.value: frozenset(),
        },
        "structure": {
            StructureLifecycle.FORMING.value: frozenset(
                {
                    StructureLifecycle.CONFIRMED.value,
                    StructureLifecycle.FORMATION_FAILED.value,
                }
            ),
            StructureLifecycle.CONFIRMED.value: frozenset(
                {StructureLifecycle.BROKEN.value}
            ),
            StructureLifecycle.BROKEN.value: frozenset(),
            StructureLifecycle.FORMATION_FAILED.value: frozenset(),
        },
        "bos": {
            BOSLifecycle.PENDING.value: frozenset(
                {
                    BOSLifecycle.CONFIRMED.value,
                    BOSLifecycle.FAILED.value,
                }
            ),
            BOSLifecycle.CONFIRMED.value: frozenset(),
            BOSLifecycle.FAILED.value: frozenset(),
        },
        "zone": {
            SupportResistanceLifecycle.ACTIVE.value: frozenset(
                {
                    SupportResistanceLifecycle.TESTED.value,
                    SupportResistanceLifecycle.BROKEN.value,
                    SupportResistanceLifecycle.RETIRED.value,
                }
            ),
            SupportResistanceLifecycle.TESTED.value: frozenset(
                {
                    SupportResistanceLifecycle.BROKEN.value,
                    SupportResistanceLifecycle.RETIRED.value,
                }
            ),
            SupportResistanceLifecycle.BROKEN.value: frozenset(
                {
                    SupportResistanceLifecycle.REACCEPTED.value,
                    SupportResistanceLifecycle.RETIRED.value,
                }
            ),
            SupportResistanceLifecycle.REACCEPTED.value: frozenset(),
            SupportResistanceLifecycle.RETIRED.value: frozenset(),
        },
        "pool": {
            LiquidityPoolLifecycle.FORMED.value: frozenset(
                {LiquidityPoolLifecycle.SWEPT.value}
            ),
            LiquidityPoolLifecycle.SWEPT.value: frozenset(
                {
                    LiquidityPoolLifecycle.ACCEPTED.value,
                    LiquidityPoolLifecycle.REJECTED.value,
                }
            ),
            LiquidityPoolLifecycle.ACCEPTED.value: frozenset(),
            LiquidityPoolLifecycle.REJECTED.value: frozenset(),
        },
        "fvg": {
            FairValueGapLifecycle.OPEN.value: frozenset(
                {
                    FairValueGapLifecycle.PARTIAL.value,
                    FairValueGapLifecycle.MITIGATED.value,
                    FairValueGapLifecycle.INVALIDATED.value,
                }
            ),
            FairValueGapLifecycle.PARTIAL.value: frozenset(
                {
                    FairValueGapLifecycle.MITIGATED.value,
                    FairValueGapLifecycle.INVALIDATED.value,
                }
            ),
            FairValueGapLifecycle.MITIGATED.value: frozenset(),
            FairValueGapLifecycle.INVALIDATED.value: frozenset(),
        },
        "order_block": {
            OrderBlockLifecycle.CREATED.value: frozenset(
                {
                    OrderBlockLifecycle.UNTESTED.value,
                    OrderBlockLifecycle.MITIGATED.value,
                    OrderBlockLifecycle.FAILED.value,
                }
            ),
            OrderBlockLifecycle.UNTESTED.value: frozenset(
                {
                    OrderBlockLifecycle.MITIGATED.value,
                    OrderBlockLifecycle.FAILED.value,
                }
            ),
            OrderBlockLifecycle.MITIGATED.value: frozenset(),
            OrderBlockLifecycle.FAILED.value: frozenset(),
        },
        "range": {
            DealingRangeLifecycle.FORMING.value: frozenset(
                {
                    DealingRangeLifecycle.MATURE.value,
                    DealingRangeLifecycle.BROKEN.value,
                }
            ),
            DealingRangeLifecycle.MATURE.value: frozenset(
                {DealingRangeLifecycle.BROKEN.value}
            ),
            DealingRangeLifecycle.BROKEN.value: frozenset(),
        },
        "manipulation": {
            ManipulationLifecycle.SWEPT.value: frozenset(
                {
                    ManipulationLifecycle.REACCEPTED.value,
                    ManipulationLifecycle.ACCEPTED_OUTSIDE.value,
                    "censored",
                }
            ),
            ManipulationLifecycle.REACCEPTED.value: frozenset(),
            ManipulationLifecycle.ACCEPTED_OUTSIDE.value: frozenset(),
            "censored": frozenset(),
        },
        "entry_path": {
            PathSequenceLifecycle.ACTIVE.value: frozenset(
                {
                    PathSequenceLifecycle.CLOSED.value,
                    PathSequenceLifecycle.CENSORED.value,
                }
            ),
            PathSequenceLifecycle.CLOSED.value: frozenset(),
            PathSequenceLifecycle.CENSORED.value: frozenset(),
        },
    }
    _TIMELINE_LIMITS: Mapping[str, int] = {
        "swing": 3,
        "structure": 3,
        "bos": 2,
        "zone": 5,
        "pool": 3,
        "fvg": 4,
        "order_block": 4,
        "range": 3,
        "manipulation": 2,
        "entry_path": 2,
    }
    _COMPLETE_INITIAL_LIFECYCLES: Mapping[str, frozenset[str]] = {
        "swing": frozenset({SwingLifecycle.FORMING.value}),
        "structure": frozenset({StructureLifecycle.FORMING.value}),
        "bos": frozenset({BOSLifecycle.PENDING.value}),
        "zone": frozenset(
            {SupportResistanceLifecycle.ACTIVE.value}
        ),
        "pool": frozenset({LiquidityPoolLifecycle.FORMED.value}),
        "fvg": frozenset({FairValueGapLifecycle.OPEN.value}),
        "order_block": frozenset(
            {OrderBlockLifecycle.CREATED.value}
        ),
        "range": frozenset({DealingRangeLifecycle.FORMING.value}),
        "manipulation": frozenset(
            {ManipulationLifecycle.SWEPT.value}
        ),
        "entry_path": frozenset(
            {PathSequenceLifecycle.ACTIVE.value}
        ),
    }

    def __init__(self, maximum_events: int) -> None:
        if type(maximum_events) is not int or maximum_events < 1:
            raise ValueError(
                "event memory maximum_events must be a positive integer"
            )
        self._events: deque[MarketEvent] = deque(maxlen=maximum_events)
        self._ids: set[str] = set()
        self._latest_by_entity: dict[str, MarketEvent] = {}
        self._closed_durations: dict[str, int] = {}
        self._sequence_counts: dict[pd.Timestamp, int] = {}
        self._synthetic_run_starts_ns: list[int] = []
        self._synthetic_runs: list[tuple[int, int, int]] = []
        self._last_minute_end: pd.Timestamp | None = None
        self._last_minute_real_completed: bool | None = None
        self._clock_coverage_start: pd.Timestamp | None = None
        self._entity_timelines: dict[str, list[MarketEvent]] = {}
        self._retained_entity_keys: set[str] = set()
        self._incomplete_entity_keys: set[str] = set()

    @property
    def last_minute_end(self) -> pd.Timestamp | None:
        return self._last_minute_end

    @property
    def clock_coverage_start(self) -> pd.Timestamp | None:
        return self._clock_coverage_start

    @staticmethod
    def _required_event_clock(event: MarketEvent) -> pd.Timestamp:
        age_origin = (
            event.formed_at
            or event.confirmed_at
            or event.observed_at
        )
        return min(age_origin, event.observed_at)

    def set_clock_coverage_start(
        self,
        value: pd.Timestamp,
    ) -> None:
        start = pd.Timestamp(value)
        if start.tzinfo is None:
            raise ValueError(
                "event memory clock coverage must be timezone aware"
            )
        if (
            self._clock_coverage_start is not None
            and start != self._clock_coverage_start
        ):
            raise ValueError(
                "event memory clock coverage cannot be rewritten"
            )
        retained_events = {
            event.event_id: event
            for timeline in self._entity_timelines.values()
            for event in timeline
        }
        retained_events.update(
            (event.event_id, event)
            for event in self._events
        )
        if any(
            self._required_event_clock(event) < start
            for event in retained_events.values()
        ):
            raise ValueError(
                "event origin predates retained 1m clock coverage"
            )
        self._clock_coverage_start = start

    def observe_minute(self, candle: Candle) -> None:
        """Advance market time while excluding synthetic minutes from age."""

        if (
            candle.timeframe is not Timeframe.M1
            or not candle.complete
        ):
            raise ValueError(
                "event memory clock requires a completed 1m candle"
            )
        if (
            self._last_minute_end is not None
            and candle.end < self._last_minute_end
        ):
            raise ValueError(
                "event memory received an out-of-order minute"
            )
        if candle.end == self._last_minute_end:
            if (
                self._last_minute_real_completed
                is not bool(candle.real_completed)
            ):
                raise ValueError(
                    "event memory minute provenance changed on retry"
                )
            return
        self._last_minute_end = candle.end
        self._last_minute_real_completed = bool(candle.real_completed)
        if candle.real_completed:
            return
        minute_ns = int(pd.Timedelta(minutes=1).value)
        end_ns = int(candle.end.value)
        if (
            self._synthetic_runs
            and end_ns == self._synthetic_runs[-1][1] + minute_ns
        ):
            start_ns, _, cumulative = self._synthetic_runs[-1]
            self._synthetic_runs[-1] = (
                start_ns,
                end_ns,
                cumulative + 1,
            )
            return
        prior_total = (
            0
            if not self._synthetic_runs
            else self._synthetic_runs[-1][2]
        )
        self._synthetic_run_starts_ns.append(end_ns)
        self._synthetic_runs.append(
            (end_ns, end_ns, prior_total + 1)
        )

    def _synthetic_count_through(self, timestamp: pd.Timestamp) -> int:
        if not self._synthetic_runs:
            return 0
        value = int(timestamp.value)
        index = bisect_right(
            self._synthetic_run_starts_ns,
            value,
        ) - 1
        if index < 0:
            return 0
        start_ns, end_ns, cumulative = self._synthetic_runs[index]
        prior_total = (
            0
            if index == 0
            else self._synthetic_runs[index - 1][2]
        )
        if value >= end_ns:
            return cumulative
        minute_ns = int(pd.Timedelta(minutes=1).value)
        return prior_total + max(
            0,
            int((value - start_ns) // minute_ns) + 1,
        )

    def _elapsed_minutes(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> int:
        wall_minutes = max(
            0,
            int((end - start).total_seconds() // 60),
        )
        synthetic_minutes = (
            self._synthetic_count_through(end)
            - self._synthetic_count_through(start)
        )
        return max(0, wall_minutes - synthetic_minutes)

    @staticmethod
    def _normalized_event(event: MarketEvent) -> MarketEvent:
        return replace(event, sequence_no=0)

    def _existing_event(
        self,
        event_id: str,
        entity_key: str | None,
    ) -> MarketEvent | None:
        if entity_key is not None:
            existing = next(
                (
                    event
                    for event in self._entity_timelines.get(
                        entity_key,
                        (),
                    )
                    if event.event_id == event_id
                ),
                None,
            )
            if existing is not None:
                return existing
        if event_id not in self._ids:
            return None
        return next(
            (
                event
                for event in self._events
                if event.event_id == event_id
            ),
            None,
        )

    @classmethod
    def _timeline_namespace(cls, entity_key: str) -> str:
        namespace, separator, identity = entity_key.partition(":")
        if (
            separator != ":"
            or not identity
            or namespace not in cls._TIMELINE_TRANSITIONS
        ):
            raise ValueError(
                "retained entity key has no registered namespace"
            )
        return namespace

    def _validate_timeline_append(
        self,
        entity_key: str,
        event: MarketEvent,
    ) -> None:
        namespace = self._timeline_namespace(entity_key)
        transitions = self._TIMELINE_TRANSITIONS[namespace]
        if event.lifecycle not in transitions:
            raise ValueError(
                "typed market event has an unregistered lifecycle"
            )
        timeline = self._entity_timelines.get(entity_key, ())
        if any(
            previous.lifecycle == event.lifecycle
            for previous in timeline
        ):
            raise ValueError(
                "typed entity lifecycle cannot be recorded twice"
            )
        if not timeline:
            return
        previous = timeline[-1]
        if event.observed_at <= previous.observed_at:
            raise ValueError(
                "typed entity lifecycle event is out of order"
            )
        if event.lifecycle not in transitions[previous.lifecycle]:
            raise ValueError(
                "typed entity lifecycle transition is not registered"
            )
        if len(timeline) >= self._TIMELINE_LIMITS[namespace]:
            raise ValueError(
                "typed entity lifecycle exceeds its bounded timeline"
            )

    @classmethod
    def _timeline_starts_incomplete(
        cls,
        entity_key: str,
        event: MarketEvent,
    ) -> bool:
        namespace = cls._timeline_namespace(entity_key)
        if event.lifecycle in cls._COMPLETE_INITIAL_LIFECYCLES[
            namespace
        ]:
            return False
        return not (
            namespace == "structure"
            and event.lifecycle
            == StructureLifecycle.CONFIRMED.value
            and event.formed_at == event.observed_at
        )

    def append(
        self,
        event: MarketEvent,
        *,
        include_in_recent: bool = True,
        sequence_floor: int | None = None,
    ) -> None:
        if type(include_in_recent) is not bool:
            raise ValueError(
                "event-memory recent inclusion flag must be boolean"
            )
        if (
            sequence_floor is not None
            and (
                type(sequence_floor) is not int
                or sequence_floor < 0
            )
        ):
            raise ValueError(
                "event-memory sequence floor must be a non-negative integer"
            )
        if (
            self._clock_coverage_start is not None
            and self._required_event_clock(event)
            < self._clock_coverage_start
        ):
            raise ValueError(
                "event origin predates retained 1m clock coverage"
            )
        entity_key = typed_event_entity_key(event)
        existing = self._existing_event(
            event.event_id,
            entity_key,
        )
        if existing is not None:
            if (
                self._normalized_event(existing)
                != self._normalized_event(event)
            ):
                raise ValueError(
                    "market event id conflicts with retained payload"
                )
            return
        if entity_key is not None:
            self._validate_timeline_append(entity_key, event)
        starts_incomplete = bool(
            entity_key is not None
            and entity_key not in self._entity_timelines
            and self._timeline_starts_incomplete(
                entity_key,
                event,
            )
        )
        sequence_no = self._sequence_counts.get(event.observed_at, 0)
        self._sequence_counts[event.observed_at] = sequence_no + 1
        if sequence_floor is not None:
            sequence_no += sequence_floor
        event = replace(event, sequence_no=sequence_no)
        if include_in_recent:
            if len(self._events) == self._events.maxlen and self._events:
                removed = self._events[0]
                self._ids.discard(removed.event_id)
                if removed.entity_id is None:
                    self._closed_durations.pop(
                        removed.event_id,
                        None,
                    )
                if removed.entity_id is not None:
                    latest_entity = self._latest_by_entity.get(
                        removed.entity_id
                    )
                    if (
                        latest_entity is not None
                        and latest_entity.event_id == removed.event_id
                    ):
                        self._latest_by_entity.pop(
                            removed.entity_id,
                            None,
                        )
            self._events.append(event)
            self._ids.add(event.event_id)
        if entity_key is not None:
            self._entity_timelines.setdefault(entity_key, []).append(
                event
            )
            if starts_incomplete:
                self._incomplete_entity_keys.add(entity_key)
        if event.entity_id is not None:
            previous = self._latest_by_entity.get(event.entity_id)
            if previous is not None:
                self._closed_durations[previous.event_id] = max(
                    0,
                    self._elapsed_minutes(
                        previous.observed_at,
                        event.observed_at,
                    ),
                )
            if event.ended_at is None:
                self._latest_by_entity[event.entity_id] = event
            else:
                self._latest_by_entity.pop(event.entity_id, None)
                self._closed_durations[event.event_id] = 0
        if len(self._sequence_counts) > self._events.maxlen * 2:
            clocks = {
                item.observed_at
                for item in self._events
            } | {
                item.observed_at
                for timeline in self._entity_timelines.values()
                for item in timeline
            }
            self._sequence_counts = {
                clock: count
                for clock, count in self._sequence_counts.items()
                if clock in clocks
            }

    def sync_retained_entity_timelines(
        self,
        entity_keys: Iterable[str],
        *,
        asof: pd.Timestamp,
    ) -> None:
        """Retain complete histories only for current typed snapshot entities."""

        clock = pd.Timestamp(asof)
        if clock.tzinfo is None:
            raise ValueError(
                "retained entity timeline cutoff must be timezone aware"
            )
        retained = set(entity_keys)
        for key in retained:
            self._timeline_namespace(key)
        missing = retained - set(self._entity_timelines)
        if missing:
            raise ValueError(
                "retained typed entity lacks a lifecycle timeline"
            )
        for timeline in self._entity_timelines.values():
            if any(event.observed_at > clock for event in timeline):
                raise ValueError(
                    "retained entity timeline contains the future"
                )
        next_timelines = {
            key: list(self._entity_timelines[key])
            for key in sorted(retained)
        }
        self._entity_timelines = next_timelines
        self._retained_entity_keys = retained
        self._incomplete_entity_keys.intersection_update(retained)
        retained_event_ids = {
            event.event_id
            for event in self._events
        } | {
            event.event_id
            for timeline in next_timelines.values()
            for event in timeline
        }
        self._closed_durations = {
            event_id: duration
            for event_id, duration in self._closed_durations.items()
            if event_id in retained_event_ids
        }
        self._latest_by_entity = {
            entity_id: event
            for entity_id, event in self._latest_by_entity.items()
            if event.event_id in retained_event_ids
        }

    def entity_timelines(
        self,
    ) -> Mapping[str, tuple[MarketEvent, ...]]:
        return {
            key: tuple(self._entity_timelines[key])
            for key in sorted(self._retained_entity_keys)
        }

    def timeline(self, entity_key: str) -> tuple[MarketEvent, ...]:
        if entity_key not in self._retained_entity_keys:
            return ()
        return tuple(self._entity_timelines[entity_key])

    def incomplete_entity_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._incomplete_entity_keys
                & self._retained_entity_keys
            )
        )

    def has_entity_lifecycle(
        self,
        entity_key: str,
        lifecycle: str,
    ) -> bool:
        return any(
            event.lifecycle == lifecycle
            for event in self._entity_timelines.get(entity_key, ())
        )

    def recent(self, limit: int = 64) -> tuple[MarketEvent, ...]:
        return tuple(self._events)[-int(limit) :]

    def temporal_metrics(
        self,
        asof: pd.Timestamp,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Materialize event duration and age in one causal traversal."""

        durations: dict[str, int] = {}
        ages: dict[str, int] = {}
        active = {
            event.event_id
            for event in self._latest_by_entity.values()
        }
        events = {
            event.event_id: event
            for event in self._events
        }
        for event in events.values():
            if event.event_id in self._closed_durations:
                durations[event.event_id] = self._closed_durations[
                    event.event_id
                ]
            elif event.event_id in active:
                durations[event.event_id] = max(
                    0,
                    self._elapsed_minutes(
                        event.formed_at or event.observed_at,
                        asof,
                    ),
                )
            else:
                durations[event.event_id] = 0
            origin = (
                event.formed_at
                or event.confirmed_at
                or event.observed_at
            )
            ages[event.event_id] = max(
                0,
                self._elapsed_minutes(origin, asof),
            )
        for timeline in self._entity_timelines.values():
            for index, event in enumerate(timeline):
                if index + 1 < len(timeline):
                    end = timeline[index + 1].observed_at
                    durations[event.event_id] = max(
                        0,
                        self._elapsed_minutes(event.observed_at, end),
                    )
                elif event.ended_at is not None:
                    durations[event.event_id] = 0
                else:
                    durations[event.event_id] = max(
                        0,
                        self._elapsed_minutes(event.observed_at, asof),
                    )
                origin = (
                    event.formed_at
                    or event.confirmed_at
                    or event.observed_at
                )
                ages[event.event_id] = max(
                    0,
                    self._elapsed_minutes(origin, asof),
                )
        return durations, ages


def _event(
    kind: EventKind,
    observed_at: pd.Timestamp,
    timeframe: Timeframe,
    side: str | None,
    price: float | None,
    strength: float,
    source_ids: Iterable[str] = (),
    details: Mapping[str, object] | None = None,
    *,
    entity_id: str | None = None,
    lifecycle: str | None = None,
    formed_at: pd.Timestamp | None = None,
    confirmed_at: pd.Timestamp | None = None,
    ended_at: pd.Timestamp | None = None,
    direction: Direction | None = None,
    transition_reason: str | None = None,
) -> MarketEvent:
    source_ids = tuple(
        str(value) for value in source_ids if value is not None
    )
    raw = (
        f"{kind.value}|{observed_at.isoformat()}|{timeframe.value}|{side}|"
        f"{price}|{'|'.join(source_ids)}|{entity_id}|{lifecycle}"
    )
    return MarketEvent(
        event_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        kind=kind,
        observed_at=observed_at,
        timeframe=timeframe,
        side=side,
        price=price,
        strength=clamp(strength),
        source_ids=tuple(source_ids),
        details={} if details is None else dict(details),
        entity_id=entity_id,
        lifecycle=lifecycle,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        ended_at=ended_at,
        direction=direction,
        transition_reason=transition_reason,
    )


def _structure_metrics_from_parts(
    structures: Sequence,
    structure_breaks: Sequence,
) -> dict[str, float]:
    by_direction = {item.direction: item for item in structures}
    long_state = by_direction.get(Direction.LONG)
    short_state = by_direction.get(Direction.SHORT)
    long_confirmed = float(
        long_state is not None
        and long_state.lifecycle is StructureLifecycle.CONFIRMED
    )
    short_confirmed = float(
        short_state is not None
        and short_state.lifecycle is StructureLifecycle.CONFIRMED
    )
    if long_confirmed and not short_confirmed:
        direction = 1.0
    elif short_confirmed and not long_confirmed:
        direction = -1.0
    else:
        direction = 0.0
    recent_breaks = list(structure_breaks)
    up_confirmed = any(
        item.direction is Direction.LONG
        and item.lifecycle is BOSLifecycle.CONFIRMED
        for item in recent_breaks
    )
    down_confirmed = any(
        item.direction is Direction.SHORT
        and item.lifecycle is BOSLifecycle.CONFIRMED
        for item in recent_breaks
    )
    return {
        "confirmed_structure_direction": direction,
        "bull_structure_confirmed": long_confirmed,
        "bear_structure_confirmed": short_confirmed,
        "bull_structure_sequence_count": float(
            0 if long_state is None else long_state.sequence_count
        ),
        "bear_structure_sequence_count": float(
            0 if short_state is None else short_state.sequence_count
        ),
        "up_bos_confirmed": float(up_confirmed),
        "down_bos_confirmed": float(down_confirmed),
    }


def _typed_progression(
    swings: Sequence,
    atr: float,
) -> tuple[float, float, float]:
    resolved = [
        item
        for item in swings
        if (
            item.confirmed_at is not None
            and item.lifecycle
            in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
        )
    ]
    highs = [item for item in resolved if item.side.value == "high"]
    lows = [item for item in resolved if item.side.value == "low"]
    high_step = (
        _signed_tanh(highs[-1].delta_points / atr)
        if highs and highs[-1].relation is not SwingRelation.NONE
        else 0.0
    )
    low_step = (
        _signed_tanh(lows[-1].delta_points / atr)
        if lows and lows[-1].relation is not SwingRelation.NONE
        else 0.0
    )
    return high_step, low_step, (high_step + low_step) / 2.0


class CausalObserver:
    """Describes current market state without producing or accepting actions."""

    def __init__(self, config: ObserverConfig) -> None:
        self.config = config
        if type(self.config.project_scene_graph) is not bool:
            raise ValueError("scene-graph projection flag must be boolean")
        if type(self.config.materialize_event_view) is not bool:
            raise ValueError("event-view materialization flag must be boolean")
        if not self.config.materialize_event_view and (
            self.config.project_scene_graph
            or self.config.group4_protocol is None
            or self.config.displacement_protocol is not None
            or self.config.group3_protocol is not None
            or self.config.group5_protocol is not None
        ):
            raise ValueError(
                "a lightweight event view is limited to the Group 1-2 + "
                "Group 4 authority scanner with Scene Graph disabled"
            )
        self.scale_specs = tuple(self.config.scale_specs)
        if not self.scale_specs:
            raise ValueError("observer requires an explicit scale registry")
        self._active_timeframes = tuple(
            spec.native_timeframe
            for spec in self.scale_specs
            if spec.enabled and spec.native_timeframe is not None
        )
        if (
            len(self._active_timeframes)
            != len(set(self._active_timeframes))
            or not set(CORE_TIMEFRAMES).issubset(self._active_timeframes)
        ):
            raise ValueError("observer scale registry is invalid")
        self._scale_registry_id = scale_registry_id(self.scale_specs)
        self.scene_graph = TemporalMarketSceneGraph()
        self.last_scene_delta: SceneGraphDelta | None = None
        displacement_protocol = (
            DisplacementProtocol.from_file(
                self.config.displacement_protocol
            )
            if self.config.displacement_protocol is not None
            else None
        )
        self._displacement_eye = (
            CausalDisplacementEye(displacement_protocol)
            if displacement_protocol is not None
            else None
        )
        self._displacement_downstream_authoritative = bool(
            displacement_protocol is not None
            and displacement_protocol.downstream_authoritative
        )
        structure_config = (
            StructureConfig.from_file(
                self.config.structure_protocol,
                atr_period=self.config.atr_period,
                tick_size=self.config.tick_size,
            )
            if self.config.structure_protocol is not None
            else None
        )
        if (
            self.config.liquidity_protocol is not None
            and structure_config is None
        ):
            raise ValueError(
                "liquidity protocol requires the confirmed-swing tracker"
            )
        if (
            self.config.group3_protocol is not None
            and (
                displacement_protocol is None
                or structure_config is None
            )
        ):
            raise ValueError(
                "Group 3 requires typed displacement and structure/BOS"
            )
        group3_protocol = (
            Group3Protocol.from_file(self.config.group3_protocol)
            if self.config.group3_protocol is not None
            else None
        )
        if group3_protocol is not None and (
            not math.isclose(
                group3_protocol.tick_size,
                self.config.tick_size,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or displacement_protocol is None
            or not math.isclose(
                group3_protocol.tick_size,
                displacement_protocol.tick_size,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(
                "Group 3, displacement and observer tick sizes disagree"
            )
        self._group3_tracker = (
            CausalGroup3Tracker(
                group3_protocol,
                displacement_protocol_hash=(
                    displacement_protocol.protocol_hash
                    if displacement_protocol is not None
                    else None
                ),
                structure_protocol_hash=(
                    structure_config.protocol_hash
                    if structure_config is not None
                    else None
                ),
            )
            if (
                group3_protocol is not None
                and self._displacement_downstream_authoritative
            )
            else None
        )
        self._group3_hidden_entity_ids: set[str] = set()
        self._structure_trackers = (
            {
                timeframe: StructureTracker(timeframe, structure_config)
                for timeframe in self._active_timeframes
            }
            if structure_config is not None
            else {}
        )
        liquidity_config = (
            LiquidityConfig.from_file(
                self.config.liquidity_protocol,
                tick_size=self.config.tick_size,
                atr_period=self.config.atr_period,
            )
            if self.config.liquidity_protocol is not None
            else None
        )
        self._liquidity_trackers = (
            {
                timeframe: CausalLiquidityTracker(
                    timeframe,
                    liquidity_config,
                )
                for timeframe in self._structure_trackers
            }
            if liquidity_config is not None
            else {}
        )
        # Native liquidity snapshots are immutable views at the tracker's
        # completed-timeframe cutoff.  Reuse unchanged higher-timeframe views;
        # exact 1m projection mutations explicitly invalidate their source.
        self._liquidity_snapshot_cache: dict[
            Timeframe,
            tuple[pd.Timestamp | None, tuple],
        ] = {}
        if (
            self.config.group4_protocol is not None
            and (
                structure_config is None
                or liquidity_config is None
                or Timeframe.H1 not in self._structure_trackers
                or Timeframe.H1 not in self._liquidity_trackers
            )
        ):
            raise ValueError(
                "Group 4 requires typed H1 structure, "
                "support/resistance and pools"
            )
        group4_protocol = (
            Group4Protocol.from_file(self.config.group4_protocol)
            if self.config.group4_protocol is not None
            else None
        )
        if group4_protocol is not None and (
            not math.isclose(
                group4_protocol.tick_size,
                self.config.tick_size,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or liquidity_config is None
            or structure_config is None
            or group4_protocol.source_group12_protocol_hash
            != liquidity_config.protocol_hash
        ):
            raise ValueError(
                "Group 4 and Group 1-2 protocol bindings disagree"
            )
        self._group4_tracker = (
            CausalGroup4Tracker(group4_protocol)
            if group4_protocol is not None
            else None
        )
        self._group4_boundary_update: Group4Update | None = None
        self._group4_bootstrap_range_transitions: list[
            DealingRangeState
        ] = []
        self._group4_cold_pairs_marked = False
        if self.config.group5_protocol is not None and (
            group3_protocol is None
            or group4_protocol is None
            or structure_config is None
            or liquidity_config is None
        ):
            raise ValueError(
                "Group 5 requires Groups 1-4 typed sources"
            )
        group5_protocol = (
            Group5Protocol.from_file(self.config.group5_protocol)
            if self.config.group5_protocol is not None
            else None
        )
        if group5_protocol is not None and (
            not math.isclose(
                group5_protocol.tick_size,
                self.config.tick_size,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or group3_protocol is None
            or group4_protocol is None
            or structure_config is None
            or liquidity_config is None
            or group5_protocol.source_group12_protocol_hash
            != structure_config.protocol_hash
            or group5_protocol.source_group12_protocol_hash
            != liquidity_config.protocol_hash
            or group5_protocol.source_group3_protocol_hash
            != group3_protocol.protocol_hash
            or group5_protocol.source_group4_protocol_hash
            != group4_protocol.protocol_hash
        ):
            raise ValueError(
                "Group 5 and its Groups 1-4 protocol bindings disagree"
            )
        self._group5_reducer = (
            CausalGroup5Reducer(group5_protocol)
            if group5_protocol is not None
            else None
        )
        self._group5_boundary_update: Group5Update | None = None
        self.memory = EventMemory(self.config.memory_events)
        self._terminal_failure: str | None = None
        self._last_displacement_input: tuple[object, ...] | None = None
        self._last_displacement_observation = None
        self._prior: MarketObservation | None = None
        self._known_level_ids: set[tuple[str, SwingLifecycle]] = set()
        self._known_level_order: deque[
            tuple[str, SwingLifecycle]
        ] = deque(
            maxlen=max(2048, self.config.memory_events * 4)
        )
        self._last_frame_cutoff: dict[Timeframe, pd.Timestamp] = {}
        self._known_structure_events: set[
            tuple[str, BOSLifecycle]
        ] = set()
        self._known_structure_event_order: deque[
            tuple[str, BOSLifecycle]
        ] = deque(maxlen=max(512, self.config.memory_events * 2))
        self._known_sequence_events: set[
            tuple[str, StructureLifecycle]
        ] = set()
        self._known_sequence_event_order: deque[
            tuple[str, StructureLifecycle]
        ] = deque(maxlen=max(512, self.config.memory_events * 2))
        self._liquidity_entity_revisions: dict[
            str,
            tuple[object, ...],
        ] = {}
        self._boundary_terminal_breaks: dict[
            Timeframe,
            tuple[BreakOfStructureState, ...],
        ] = {}
        self._boundary_reset_identity: tuple[
            pd.Timestamp,
            tuple[str, ...],
        ] | None = None
        self._mss_displacement_by_bos: dict[str, str] = {}
        self._inventory_consumption: dict[
            str,
            tuple[pd.Timestamp, str],
        ] = {}
        self._pending_pool_sweeps: dict[
            str,
            tuple[pd.Timestamp, LiquidityInventoryItem],
        ] = {}
        self._reference_periods: dict[str, _ReferencePeriod] = {}
        self._reference_inventory: dict[str, LiquidityInventoryItem] = {}
        self._reference_last_end: pd.Timestamp | None = None
        self._reference_coverage_start: pd.Timestamp | None = None

    @property
    def group5_protocol(self) -> Group5Protocol | None:
        return (
            None
            if self._group5_reducer is None
            else self._group5_reducer.protocol
        )

    @staticmethod
    def _remember_bounded(
        value,
        *,
        known: set,
        order: deque,
    ) -> bool:
        if value in known:
            return False
        if len(order) == order.maxlen and order:
            known.discard(order[0])
        order.append(value)
        known.add(value)
        return True

    def _reset_contract_state(
        self,
        *,
        reason: str,
        observed_at: pd.Timestamp,
        reset_anomalies: Sequence[str],
        boundary_symbol: str,
        boundary_instrument_id: int,
    ) -> None:
        self._group5_boundary_update = (
            self._group5_reducer.on_boundary(
                reason,
                observed_at,
                symbol=boundary_symbol,
                instrument_id=boundary_instrument_id,
            )
            if self._group5_reducer is not None
            else None
        )
        self._group4_boundary_update = (
            self._group4_tracker.on_boundary(reason, observed_at)
            if self._group4_tracker is not None
            else None
        )
        self._group4_bootstrap_range_transitions.clear()
        self._group4_cold_pairs_marked = False
        failed_bos = tuple(
            item
            for tracker in self._structure_trackers.values()
            for item in tracker.reset_for_boundary(
                reason=reason,
                observed_at=observed_at,
            )
        )
        self._boundary_terminal_breaks = {
            timeframe: tuple(
                item
                for item in failed_bos
                if item.timeframe is timeframe
            )
            for timeframe in self._active_timeframes
        }
        prior_memory = self.memory
        self.memory = EventMemory(self.config.memory_events)
        self.memory._synthetic_run_starts_ns = list(
            prior_memory._synthetic_run_starts_ns
        )
        self.memory._synthetic_runs = list(
            prior_memory._synthetic_runs
        )
        self.memory._last_minute_end = prior_memory.last_minute_end
        self.memory._last_minute_real_completed = (
            prior_memory._last_minute_real_completed
        )
        if prior_memory.clock_coverage_start is not None:
            self.memory.set_clock_coverage_start(
                prior_memory.clock_coverage_start
            )
        self._prior = None
        self._known_level_ids.clear()
        self._known_level_order.clear()
        self._last_frame_cutoff.clear()
        self._known_structure_events.clear()
        self._known_structure_event_order.clear()
        self._known_sequence_events.clear()
        self._known_sequence_event_order.clear()
        self._liquidity_entity_revisions.clear()
        self._mss_displacement_by_bos.clear()
        for tracker in self._liquidity_trackers.values():
            tracker.reset()
        self._liquidity_snapshot_cache.clear()
        self._inventory_consumption.clear()
        self._pending_pool_sweeps.clear()
        self._reference_periods.clear()
        self._reference_inventory.clear()
        self._reference_last_end = None
        self._reference_coverage_start = None
        for item in failed_bos:
            key = (item.bos_id, item.lifecycle)
            self._remember_bounded(
                key,
                known=self._known_structure_events,
                order=self._known_structure_event_order,
            )
            self.memory.append(
                _event(
                    EventKind.STRUCTURE_BREAK_FAILED,
                    item.resolved_at or observed_at,
                    item.timeframe,
                    (
                        "above"
                        if item.direction is Direction.LONG
                        else "below"
                    ),
                    item.target_price,
                    0.0,
                    tuple(
                        value
                        for value in (
                            item.target_swing_id,
                            item.source_structure_id,
                        )
                        if value is not None
                    ),
                    {
                        "bos_id": item.bos_id,
                        "scope": item.scope.value,
                        "pending_at": item.pending_at.isoformat(),
                        "resolved_at": (
                            None
                            if item.resolved_at is None
                            else item.resolved_at.isoformat()
                        ),
                        "failure_reason": item.failure_reason,
                        "direction": item.direction.value,
                        "lifecycle": item.lifecycle.value,
                        "target_swing_id": item.target_swing_id,
                        "source_structure_id": item.source_structure_id,
                        "target_ticks": item.target_ticks,
                        "age_bars": item.age_bars,
                        "attempt_count": item.attempt_count,
                        "attempt_clocks": tuple(
                            value.isoformat()
                            for value in item.attempt_clocks
                        ),
                        "last_attempt_at": (
                            None
                            if item.last_attempt_at is None
                            else item.last_attempt_at.isoformat()
                        ),
                        "reset_anomalies": tuple(reset_anomalies),
                    },
                    entity_id=item.bos_id,
                    lifecycle=item.lifecycle.value,
                    formed_at=item.pending_at,
                    ended_at=item.resolved_at or observed_at,
                    direction=item.direction,
                    transition_reason=item.failure_reason,
                )
            )

    def _execution(
        self,
        update: ReaderUpdate,
        reality: ExecutionRealityInput,
    ) -> ExecutionObservation:
        anomalies = list(reality.anomalies)
        spread = reality.spread_points
        if spread is None:
            spread = self.config.tick_size
            anomalies.append("spread_missing_used_one_tick")
        if reality.source.startswith("constant"):
            anomalies.append("execution_constant_assumption")
        if spread < 0:
            raise ValueError("spread cannot be negative")
        if reality.quantity <= 0:
            raise ValueError("execution quantity must be positive")
        commission_points = (
            2.0 * reality.commission_per_contract_per_side / self.config.point_value
        )
        cost = float(spread + 2.0 * reality.expected_slippage_points + commission_points)
        if reality.deadline is None:
            minutes = 24 * 60
            anomalies.append("deadline_missing")
        else:
            deadline = pd.Timestamp(reality.deadline)
            if deadline.tzinfo is None:
                raise ValueError("execution deadline must be timezone aware")
            minutes = int((deadline - update.asof).total_seconds() // 60)
        spread_ticks = spread / self.config.tick_size
        spread_score = math.exp(-0.18 * max(0.0, spread_ticks - 1.0))
        freshness = math.exp(-max(0.0, reality.data_age_seconds) / 30.0)
        deadline_score = clamp(max(0.0, minutes) / 20.0)
        size_score = (
            1.0
            if reality.size_available is None
            else clamp(reality.size_available / max(1, reality.quantity))
        )
        fillability = clamp(spread_score * freshness * max(0.25, deadline_score) * size_score)
        if reality.data_age_seconds > 60:
            anomalies.append("stale_market_data")
        if minutes <= 0:
            anomalies.append("deadline_elapsed")
        return ExecutionObservation(
            spread_points=float(spread),
            expected_slippage_points=float(reality.expected_slippage_points),
            expected_round_trip_cost_points=cost,
            minutes_to_deadline=minutes,
            fillability=fillability,
            data_age_seconds=float(reality.data_age_seconds),
            size_available=reality.size_available,
            anomalies=tuple(anomalies),
            source=reality.source,
            bid=reality.bid,
            ask=reality.ask,
            bid_size=reality.bid_size,
            ask_size=reality.ask_size,
            depth_imbalance=reality.depth_imbalance,
        )

    def _prior_frame_if_unchanged(
        self,
        update: ReaderUpdate,
        timeframe: Timeframe,
    ) -> FrameObservation | None:
        """Return an exact immutable HTF frame reuse, or require rebuild."""

        if self._prior is None:
            return None
        history = update.histories[timeframe]
        if update.newly_completed.get(timeframe, ()):
            return None
        prior = self._prior.frame(timeframe)
        if not history:
            hard_boundary = bool(
                {
                    "contract_change_history_reset",
                    "data_gap_history_reset",
                }.intersection(update.anomalies)
            )
            return (
                prior
                if prior.bars == 0 and not hard_boundary
                else None
            )
        if prior.cutoff != history[-1].end:
            return None
        tracker = self._structure_trackers.get(timeframe)
        if tracker is not None and tracker.last_end != history[-1].end:
            self._terminal_failure = (
                f"{timeframe.value} structure clock diverged; discard "
                "this observer and resume from the last checkpoint"
            )
            raise RuntimeError(
                f"{timeframe.value} observer/tracker clock drift"
            )
        liquidity_tracker = self._liquidity_trackers.get(timeframe)
        if (
            liquidity_tracker is not None
            and liquidity_tracker.last_end != history[-1].end
        ):
            self._terminal_failure = (
                f"{timeframe.value} liquidity clock diverged; discard "
                "this observer and resume from the last checkpoint"
            )
            raise RuntimeError(
                f"{timeframe.value} observer/liquidity clock drift"
            )
        return prior

    def _enrich_mss_breaks(
        self,
        states: Sequence[BreakOfStructureState],
    ) -> tuple[BreakOfStructureState, ...]:
        """Bind an opposed BOS to exact same-bar active M5 displacement."""

        batch = (
            ()
            if self._displacement_eye is None
            else self._displacement_eye.last_batch
        )
        candidates: dict[
            tuple[pd.Timestamp, Direction, str],
            str,
        ] = {}
        for candle, update in batch:
            displacement_state = update.state
            if (
                displacement_state is None
                or displacement_state.lifecycle
                is not DisplacementLifecycle.ACTIVE
                or displacement_state.prefix_last_admitted_at != candle.end
                or displacement_state.last_valid_candle_id
                not in displacement_state.admitted_candle_ids
            ):
                continue
            key = (
                candle.end,
                displacement_state.direction,
                displacement_state.last_valid_candle_id,
            )
            if key in candidates:
                raise RuntimeError(
                    "same completed bar exposed multiple active displacement "
                    "identities"
                )
            candidates[key] = displacement_state.entity_id

        output: list[BreakOfStructureState] = []
        for state in states:
            source_id = self._mss_displacement_by_bos.get(state.bos_id)
            if (
                source_id is None
                and state.lifecycle is BOSLifecycle.CONFIRMED
                and state.scope is BOSScope.OPPOSED
                and state.resolved_at is not None
                and state.break_bar_id is not None
            ):
                source_id = candidates.get(
                    (
                        state.resolved_at,
                        state.direction,
                        state.break_bar_id,
                    )
                )
                if source_id is not None:
                    self._mss_displacement_by_bos[state.bos_id] = source_id
            output.append(
                state
                if source_id is None
                else replace(
                    state,
                    source_displacement_id=source_id,
                    mss_qualified=True,
                )
            )
        return tuple(output)

    @staticmethod
    def _observe_frame(
        timeframe: Timeframe,
        history: Sequence[Candle],
        asof: pd.Timestamp,
        config: ObserverConfig,
    ) -> FrameObservation:
        semantic_history = tuple(
            candle for candle in history if candle.real_completed
        )
        if timeframe is Timeframe.H4:
            frame = _observe_4h(semantic_history, asof, config)
        elif timeframe is Timeframe.H1:
            frame = _observe_1h(semantic_history, asof, config)
        elif timeframe is Timeframe.M15:
            # Bridge-scale structure uses the same parameterized swing/BOS
            # and liquidity reducers.  M15 displacement/FVG/OB remain
            # explicitly unsupported rather than borrowing M5 semantics.
            frame = _observe_1h(
                semantic_history,
                asof,
                config,
                timeframe=Timeframe.M15,
            )
        elif timeframe is Timeframe.M5:
            frame = _observe_5m(semantic_history, asof, config)
        elif timeframe is Timeframe.M1:
            frame = _observe_1m(semantic_history, asof, config)
        else:
            raise ValueError("unsupported observation timeframe")
        if history:
            current = history[-1]
            frame = replace(
                frame,
                cutoff=current.end,
                candle_structure=_candle_structure(
                    current,
                    prior_atr=_strict_prior_atr(
                        history,
                        current,
                        config.atr_period,
                    ),
                ),
            )
        return frame

    def _sync_structure_tracker(
        self,
        update: ReaderUpdate,
        timeframe: Timeframe,
        tracker: StructureTracker,
        liquidity_tracker: CausalLiquidityTracker | None,
    ) -> None:
        history = update.histories[timeframe]
        reference_bootstrap = bool(
            timeframe is Timeframe.M1
            and self._reference_last_end is None
        )
        if reference_bootstrap and history:
            real_history = tuple(
                candle for candle in history if candle.real_completed
            )
            if real_history:
                self._reference_coverage_start = min(
                    candle.start for candle in real_history
                )
        incoming = (
            history
            if tracker.last_end is None
            else update.newly_completed.get(timeframe, ())
        )
        for candle in incoming:
            try:
                if (
                    timeframe is Timeframe.M1
                    and candle.real_completed
                    and (
                        self._reference_last_end is None
                        or candle.end > self._reference_last_end
                    )
                ):
                    self._advance_reference_periods(
                        candle,
                        append_retirement_events=(
                            not reference_bootstrap
                        ),
                    )
                    self._reference_last_end = candle.end
                tracker.on_candle(candle)
                if liquidity_tracker is not None:
                    swings, structures, _ = tracker.snapshot()
                    liquidity_tracker.on_candle(
                        candle,
                        swings,
                        structures,
                        reference_sources=(
                            tuple(self._reference_inventory.values())
                            if timeframe is Timeframe.M1
                            else ()
                        ),
                    )
                    self._invalidate_liquidity_snapshot(timeframe)
                    if (
                        timeframe is Timeframe.H1
                        and self._group4_tracker is not None
                        and self._prior is None
                    ):
                        support_resistance, _, _ = (
                            liquidity_tracker.snapshot()
                        )
                        coverage_start = (
                            self.memory.clock_coverage_start
                        )
                        within_group4_coverage = (
                            coverage_start is not None
                            and candle.end >= coverage_start
                        )
                        if (
                            within_group4_coverage
                            and not self._group4_cold_pairs_marked
                        ):
                            self._group4_tracker.mark_existing_source_pairs_ineligible(
                                tuple(
                                    zone
                                    for zone in support_resistance
                                    if zone.confirmed_at
                                    < coverage_start
                                )
                            )
                            self._group4_cold_pairs_marked = True
                        group4_update = (
                            self._group4_tracker.on_completed_h1(
                                candle,
                                (
                                    support_resistance
                                    if within_group4_coverage
                                    else ()
                                ),
                            )
                        )
                        self._group4_bootstrap_range_transitions.extend(
                            group4_update.range_transitions
                        )
            except Exception:
                self._terminal_failure = (
                    f"{timeframe.value} structure/liquidity update failed "
                    f"at {candle.end.isoformat()}; discard this observer "
                    "and resume from the last checkpoint"
                )
                raise
        expected_end = history[-1].end if history else None
        if tracker.last_end != expected_end:
            self._terminal_failure = (
                f"{timeframe.value} structure clock diverged; discard "
                "this observer and resume from the last checkpoint"
            )
            raise RuntimeError(
                f"{timeframe.value} structure tracker did not reach "
                "the causal history tail"
            )
        if (
            liquidity_tracker is not None
            and liquidity_tracker.last_end != expected_end
        ):
            self._terminal_failure = (
                f"{timeframe.value} liquidity clock diverged; discard "
                "this observer and resume from the last checkpoint"
            )
            raise RuntimeError(
                f"{timeframe.value} liquidity tracker did not reach "
                "the causal history tail"
            )

    def _record_frame_events(
        self,
        frame: FrameObservation,
        newly_completed: bool,
        *,
        event_clock: pd.Timestamp | None = None,
    ) -> None:
        if not newly_completed:
            return
        event_clock = frame.cutoff if event_clock is None else event_clock
        if event_clock > frame.cutoff:
            raise ValueError(
                "frame event clock cannot exceed the observation cutoff"
            )
        for swing in frame.swings:
            key = (swing.swing_id, swing.lifecycle)
            if self._remember_bounded(
                key,
                known=self._known_level_ids,
                order=self._known_level_order,
            ):
                if self.memory.has_entity_lifecycle(
                    f"swing:{swing.swing_id}",
                    swing.lifecycle.value,
                ):
                    continue
                observed_at = (
                    swing.broken_at
                    if swing.lifecycle is SwingLifecycle.BROKEN
                    else swing.observed_at
                )
                direction = (
                    Direction.LONG
                    if swing.relation
                    in {SwingRelation.HH, SwingRelation.HL}
                    else Direction.SHORT
                    if swing.relation
                    in {SwingRelation.LH, SwingRelation.LL}
                    else None
                )
                self.memory.append(
                    _event(
                        EventKind.SWING_STATE,
                        observed_at,
                        frame.timeframe,
                        "above" if swing.side.value == "high" else "below",
                        swing.price,
                        clamp(swing.magnitude_atr),
                        tuple(
                            value
                            for value in (
                                swing.swing_id,
                                swing.prior_same_side_id,
                            )
                            if value is not None
                        ),
                        {
                            "relation": swing.relation.value,
                            "pivot_start": swing.pivot_start.isoformat(),
                            "pivot_end": swing.pivot_end.isoformat(),
                            "confirmed_at": (
                                None
                                if swing.confirmed_at is None
                                else swing.confirmed_at.isoformat()
                            ),
                            "age_bars": swing.age_bars,
                            "delta_ticks": swing.delta_ticks,
                            "delta_points": swing.delta_points,
                            "magnitude_atr": swing.magnitude_atr,
                        },
                        entity_id=swing.swing_id,
                        lifecycle=swing.lifecycle.value,
                        formed_at=swing.pivot_end,
                        confirmed_at=swing.confirmed_at,
                        ended_at=(
                            observed_at
                            if swing.lifecycle
                            in {
                                SwingLifecycle.BROKEN,
                                SwingLifecycle.FORMATION_FAILED,
                            }
                            else None
                        ),
                        direction=direction,
                        transition_reason=swing.failure_reason,
                    )
                )
        for state in frame.structures:
            if (
                state.structure_id is None
                or state.lifecycle is StructureLifecycle.INACTIVE
            ):
                continue
            key = (state.structure_id, state.lifecycle)
            if not self._remember_bounded(
                key,
                known=self._known_sequence_events,
                order=self._known_sequence_event_order,
            ):
                continue
            if self.memory.has_entity_lifecycle(
                f"structure:{state.structure_id}",
                state.lifecycle.value,
            ):
                continue
            observed_at = (
                state.broken_at
                if state.lifecycle is StructureLifecycle.BROKEN
                else state.formation_failed_at
                if state.lifecycle
                is StructureLifecycle.FORMATION_FAILED
                else state.confirmed_at
                if state.lifecycle is StructureLifecycle.CONFIRMED
                else state.formed_at
            )
            self.memory.append(
                _event(
                    EventKind.STRUCTURE_STATE,
                    observed_at,
                    frame.timeframe,
                    (
                        "above"
                        if state.direction is Direction.LONG
                        else "below"
                    ),
                    state.protected_price,
                    clamp(state.cumulative_magnitude_atr),
                    tuple(
                        value
                        for value in (
                            state.latest_high_id,
                            state.latest_low_id,
                            state.protected_swing_id,
                        )
                        if value is not None
                    ),
                    {
                        "high_run": state.high_run,
                        "low_run": state.low_run,
                        "sequence_count": state.sequence_count,
                        "age_bars": state.age_bars,
                    },
                    entity_id=state.structure_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=(
                        state.broken_at
                        if state.lifecycle is StructureLifecycle.BROKEN
                        else state.formation_failed_at
                        if state.lifecycle
                        is StructureLifecycle.FORMATION_FAILED
                        else None
                    ),
                    direction=state.direction,
                    transition_reason=state.failure_reason,
                )
            )
        for zone in frame.support_resistance:
            revision = (
                zone.lifecycle.value,
                zone.total_touch_count,
                zone.member_swing_ids,
                zone.source_ids,
                zone.range_id,
                zone.source_zone_id,
                zone.source_kind,
                zone.structural_rank,
                zone.is_protected_swing,
                zone.zone_role,
                zone.reaction_quality,
                zone.depletion_risk,
            )
            prior_revision = self._liquidity_entity_revisions.get(
                zone.zone_id
            )
            if prior_revision == revision:
                continue
            self._liquidity_entity_revisions[zone.zone_id] = revision
            lifecycle_revision = bool(
                prior_revision is None
                or prior_revision[0] != zone.lifecycle.value
            )
            observed_at = (
                zone.retired_at
                or zone.reaccepted_at
                or zone.broken_at
                or (
                    zone.touch_times[-1]
                    if zone.lifecycle
                    is SupportResistanceLifecycle.TESTED
                    else None
                )
                or zone.confirmed_at
            )
            if not lifecycle_revision:
                observed_at = max(
                    observed_at,
                    zone.metadata_observed_at,
                )
            self.memory.append(
                _event(
                    EventKind.SUPPORT_RESISTANCE_STATE,
                    observed_at,
                    frame.timeframe,
                    (
                        "below"
                        if zone.side == "support"
                        else "above"
                    ),
                    zone.anchor_price,
                    zone.strength,
                    (
                        *(() if lifecycle_revision else (zone.zone_id,)),
                        *zone.causal_source_ids,
                    ),
                    {
                        "state_revision": not lifecycle_revision,
                        "lower_bound": zone.lower_bound,
                        "upper_bound": zone.upper_bound,
                        "touch_times": tuple(
                            value.isoformat()
                            for value in zone.touch_times
                        ),
                        "reaction_magnitudes_atr": (
                            zone.reaction_magnitudes_atr
                        ),
                        "touch_count": zone.touch_count,
                        "age_bars": zone.age_bars,
                        "source_kind": zone.source_kind,
                        "source_ids": zone.source_ids,
                        "range_id": zone.range_id,
                        "source_zone_id": zone.source_zone_id,
                        "zone_role": zone.zone_role,
                        "structural_rank": zone.structural_rank,
                        "is_protected_swing": zone.is_protected_swing,
                        "visibility_strength": zone.visibility_strength,
                        "reaction_quality": zone.reaction_quality,
                        "freshness": zone.freshness,
                        "depletion_risk": zone.depletion_risk,
                    },
                    entity_id=(
                        zone.zone_id if lifecycle_revision else None
                    ),
                    lifecycle=(
                        zone.lifecycle.value if lifecycle_revision else None
                    ),
                    formed_at=zone.formed_at,
                    confirmed_at=zone.confirmed_at,
                    ended_at=(
                        observed_at
                        if lifecycle_revision and zone.lifecycle
                        in {
                            SupportResistanceLifecycle.REACCEPTED,
                            SupportResistanceLifecycle.RETIRED,
                        }
                        else None
                    ),
                    transition_reason=(
                        zone.transition_reason
                        if lifecycle_revision
                        else "support_resistance_evidence_revised"
                    ),
                )
            )
        for pool in frame.liquidity_pools:
            if pool.lifecycle is not LiquidityPoolLifecycle.FORMED:
                continue
            entity_id = f"pool:{pool.pool_id}"
            revision = (
                pool.lifecycle.value,
                pool.touch_count,
                pool.member_swing_ids,
            )
            prior_revision = self._liquidity_entity_revisions.get(
                entity_id
            )
            if prior_revision == revision:
                continue
            self._liquidity_entity_revisions[entity_id] = revision
            lifecycle_revision = bool(
                prior_revision is None
                or prior_revision[0] != pool.lifecycle.value
            )
            self.memory.append(
                _event(
                    EventKind.LIQUIDITY_POOL_STATE,
                    (
                        pool.confirmed_at
                        if lifecycle_revision
                        else pool.touch_times[-1]
                    ),
                    frame.timeframe,
                    pool.side,
                    pool.midpoint,
                    pool.strength,
                    (
                        *(() if lifecycle_revision else (entity_id,)),
                        *pool.member_swing_ids,
                    ),
                    {
                        "state_revision": not lifecycle_revision,
                        "lower_bound": pool.lower_bound,
                        "upper_bound": pool.upper_bound,
                        "touch_times": tuple(
                            value.isoformat()
                            for value in pool.touch_times
                        ),
                        "touch_count": pool.touch_count,
                        "age_bars": pool.age_bars,
                    },
                    entity_id=entity_id if lifecycle_revision else None,
                    lifecycle=(
                        pool.lifecycle.value if lifecycle_revision else None
                    ),
                    formed_at=pool.formed_at,
                    confirmed_at=pool.confirmed_at,
                    transition_reason=(
                        None
                        if lifecycle_revision
                        else "liquidity_pool_membership_revised"
                    ),
                )
            )
        for item in frame.structure_breaks:
            key = (item.bos_id, item.lifecycle)
            is_new_lifecycle = self._remember_bounded(
                key,
                known=self._known_structure_events,
                order=self._known_structure_event_order,
            )
            if is_new_lifecycle and not self.memory.has_entity_lifecycle(
                f"bos:{item.bos_id}", item.lifecycle.value
            ):
                self.memory.append(_event(
                    (
                        EventKind.BOS_STATE
                        if item.lifecycle is BOSLifecycle.PENDING
                        else
                        EventKind.STRUCTURE_BREAK
                        if item.lifecycle is BOSLifecycle.CONFIRMED
                        else EventKind.STRUCTURE_BREAK_FAILED
                    ),
                    item.resolved_at or item.pending_at,
                    frame.timeframe,
                    "above" if item.direction is Direction.LONG else "below",
                    item.target_price,
                    item.strength,
                    tuple(
                        value
                        for value in (
                            item.target_swing_id,
                            item.source_structure_id,
                            item.source_displacement_id,
                            item.break_bar_id,
                        )
                        if value is not None
                    ),
                    {
                        "bos_id": item.bos_id,
                        "scope": item.scope.value,
                        "source_structure_id": item.source_structure_id,
                        "pending_at": item.pending_at.isoformat(),
                        "resolved_at": (
                            None
                            if item.resolved_at is None
                            else item.resolved_at.isoformat()
                        ),
                        "failure_reason": item.failure_reason,
                        "strength": item.strength,
                        "break_bar_id": item.break_bar_id,
                        "break_distance_atr": item.break_distance_atr,
                        "source_displacement_id": (
                            item.source_displacement_id
                        ),
                        "mss_qualified": item.mss_qualified,
                        "post_break_state": (
                            "pending"
                            if item.lifecycle is BOSLifecycle.CONFIRMED
                            else None
                        ),
                        "accepted_at": None,
                        "rejected_at": None,
                    },
                    entity_id=item.bos_id,
                    lifecycle=item.lifecycle.value,
                    formed_at=item.pending_at,
                    confirmed_at=(
                        item.resolved_at
                        if item.lifecycle is BOSLifecycle.CONFIRMED
                        else None
                    ),
                    ended_at=(
                        item.resolved_at
                        if item.lifecycle is BOSLifecycle.FAILED
                        else None
                    ),
                    direction=item.direction,
                    transition_reason=(
                        BOS_CONFIRMATION_REASON
                        if item.lifecycle is BOSLifecycle.CONFIRMED
                        else item.failure_reason
                    ),
                ))
            post_break_at = item.accepted_at or item.rejected_at
            if (
                item.lifecycle is BOSLifecycle.CONFIRMED
                and post_break_at is not None
                and item.post_break_state is not None
            ):
                self.memory.append(
                    _event(
                        EventKind.BOS_POST_BREAK_STATE,
                        post_break_at,
                        frame.timeframe,
                        (
                            "above"
                            if item.direction is Direction.LONG
                            else "below"
                        ),
                        item.target_price,
                        item.strength,
                        tuple(
                            value
                            for value in (
                                item.bos_id,
                                item.target_swing_id,
                                item.break_bar_id,
                                item.source_displacement_id,
                            )
                            if value is not None
                        ),
                        {
                            "bos_id": item.bos_id,
                            "scope": item.scope.value,
                            "post_break_state": (
                                item.post_break_state.value
                            ),
                            "accepted_at": (
                                None
                                if item.accepted_at is None
                                else item.accepted_at.isoformat()
                            ),
                            "rejected_at": (
                                None
                                if item.rejected_at is None
                                else item.rejected_at.isoformat()
                            ),
                        },
                        direction=item.direction,
                        transition_reason=(
                            f"post_break_{item.post_break_state.value}"
                        ),
                    )
                )

    def _record_group3_events(
        self,
        update: Group3Update,
    ) -> None:
        for state in update.fvg_transitions:
            observed_at = state.state_started_at
            terminal = state.lifecycle in {
                FairValueGapLifecycle.MITIGATED,
                FairValueGapLifecycle.INVALIDATED,
            }
            self.memory.append(
                _event(
                    EventKind.FVG_STATE,
                    observed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    (
                        state.source_displacement_id,
                        state.source_active_transition_id,
                        *state.source_candle_ids,
                    ),
                    {
                        "protocol_hash": state.protocol_hash,
                        "qualification": state.qualification.value,
                        "lower_bound": state.lower_bound,
                        "upper_bound": state.upper_bound,
                        "midpoint": state.midpoint,
                        "invalidation_price": (
                            state.invalidation_price
                        ),
                        "width_ticks": state.width_ticks,
                        "width_atr": state.width_atr,
                        "formation_atr": state.formation_atr,
                        "age_bars": state.age_bars,
                        "max_fill_fraction": (
                            state.max_fill_fraction
                        ),
                        "source_displacement_protocol_hash": (
                            state.source_displacement_protocol_hash
                        ),
                        "source_displacement_started_at": (
                            None
                            if state.source_displacement_started_at is None
                            else state.source_displacement_started_at.isoformat()
                        ),
                        "source_displacement_active_at": (
                            None
                            if state.source_displacement_active_at is None
                            else state.source_displacement_active_at.isoformat()
                        ),
                        "source_displacement_prefix_commitment": (
                            state.source_displacement_prefix_commitment
                        ),
                        "source_candle_starts": tuple(
                            value.isoformat()
                            for value in state.source_candle_starts
                        ),
                    },
                    entity_id=state.fvg_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=observed_at if terminal else None,
                    direction=state.direction,
                    transition_reason=state.transition_reason,
                ),
                include_in_recent=False,
            )
        for state in update.order_block_transitions:
            observed_at = state.state_started_at
            terminal = state.lifecycle in {
                OrderBlockLifecycle.MITIGATED,
                OrderBlockLifecycle.FAILED,
            }
            self.memory.append(
                _event(
                    EventKind.ORDER_BLOCK_STATE,
                    observed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    (
                        state.source_displacement_id,
                        state.source_active_transition_id,
                        state.source_bos_id,
                        state.anchor_candle_id,
                    ),
                    {
                        "protocol_hash": state.protocol_hash,
                        "lower_bound": state.lower_bound,
                        "upper_bound": state.upper_bound,
                        "midpoint": state.midpoint,
                        "invalidation_price": (
                            state.invalidation_price
                        ),
                        "anchor_open": state.anchor_open,
                        "anchor_close": state.anchor_close,
                        "anchor_candle_ids": state.anchor_candle_ids,
                        "body_lower_bound": state.body_lower_bound,
                        "body_upper_bound": state.body_upper_bound,
                        "width_ticks": state.width_ticks,
                        "width_atr": state.width_atr,
                        "age_bars": state.age_bars,
                        "source_displacement_protocol_hash": (
                            state.source_displacement_protocol_hash
                        ),
                        "source_displacement_seed_candle_id": (
                            state.source_displacement_seed_candle_id
                        ),
                        "source_displacement_started_at": (
                            state.source_displacement_started_at.isoformat()
                        ),
                        "source_displacement_active_at": (
                            state.source_displacement_active_at.isoformat()
                        ),
                        "source_displacement_prefix_commitment": (
                            state.source_displacement_prefix_commitment
                        ),
                        "source_bos_protocol_hash": (
                            state.source_bos_protocol_hash
                        ),
                        "source_bos_target_swing_id": (
                            state.source_bos_target_swing_id
                        ),
                        "source_bos_structure_id": (
                            state.source_bos_structure_id
                        ),
                        "source_bos_scope": (
                            state.source_bos_scope.value
                        ),
                        "source_bos_resolved_at": (
                            state.source_bos_resolved_at.isoformat()
                        ),
                        "source_bos_pending_at": (
                            state.source_bos_pending_at.isoformat()
                        ),
                        "source_bos_break_bar_id": (
                            state.source_bos_break_bar_id
                        ),
                        "source_bos_mss_qualified": (
                            state.source_bos_mss_qualified
                        ),
                        "anchor_start": state.anchor_start.isoformat(),
                        "anchor_end": state.anchor_end.isoformat(),
                    },
                    entity_id=state.order_block_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=observed_at if terminal else None,
                    direction=state.direction,
                    transition_reason=state.transition_reason,
                ),
                include_in_recent=False,
            )

    def _record_group4_events(
        self,
        update: Group4Update,
        *,
        include_ranges: bool = True,
        include_resolutions: bool = True,
        include_creations: bool = True,
    ) -> None:
        if any(
            type(value) is not bool
            for value in (
                include_ranges,
                include_resolutions,
                include_creations,
            )
        ):
            raise TypeError("Group 4 event phase flags must be boolean")
        if update.boundary_reason is not None:
            return
        if include_creations and self._prior is not None:
            prior_manipulations = {
                state.manipulation_id: state
                for state in self._prior.manipulations
            }
            revision_fields = (
                "reentry_candidate_at",
                "reentry_candidate_price",
                "inside_hold_bars",
                "reentry_failed_at",
                "outside_run",
                "outside_run_side",
            )
            for state in update.manipulations:
                prior_state = prior_manipulations.get(
                    state.manipulation_id
                )
                if (
                    prior_state is None
                    or state.lifecycle
                    is not ManipulationLifecycle.SWEPT
                    or state.deadline_elapsed
                    or all(
                        getattr(state, name)
                        == getattr(prior_state, name)
                        for name in revision_fields
                    )
                ):
                    continue
                if (
                    state.reentry_candidate_at is not None
                    and state.reentry_candidate_at
                    != prior_state.reentry_candidate_at
                ):
                    revision_reason = "reentry_candidate_started"
                elif (
                    state.reentry_failed_at is not None
                    and state.reentry_failed_at
                    != prior_state.reentry_failed_at
                ):
                    revision_reason = "reentry_candidate_failed"
                else:
                    revision_reason = "outside_acceptance_progressed"
                self.memory.append(
                    _event(
                        EventKind.MANIPULATION_STATE,
                        state.last_updated_at,
                        Timeframe.M1,
                        state.side,
                        (
                            state.reentry_candidate_price
                            if state.reentry_candidate_price is not None
                            else state.sweep_extreme
                        ),
                        state.strength,
                        (
                            state.manipulation_id,
                            state.source_inventory_item_id,
                            *state.crossed_source_ids,
                        ),
                        {
                            "state_revision": True,
                            "revision_reason": revision_reason,
                            "protocol_hash": state.protocol_hash,
                            "source_kind": state.source_kind,
                            "reentry_candidate_at": (
                                None
                                if state.reentry_candidate_at is None
                                else state.reentry_candidate_at.isoformat()
                            ),
                            "reentry_candidate_price": (
                                state.reentry_candidate_price
                            ),
                            "inside_hold_bars": state.inside_hold_bars,
                            "reentry_failed_at": (
                                None
                                if state.reentry_failed_at is None
                                else state.reentry_failed_at.isoformat()
                            ),
                            "outside_run": state.outside_run,
                            "outside_run_side": state.outside_run_side,
                            "age_1m_bars": state.age_1m_bars,
                        },
                        transition_reason=revision_reason,
                    )
                )
        for state in (
            update.range_transitions
            if include_ranges
            else ()
        ):
            terminal = (
                state.lifecycle is DealingRangeLifecycle.BROKEN
            )
            self.memory.append(
                _event(
                    EventKind.DEALING_RANGE_STATE,
                    state.state_started_at,
                    Timeframe.H1,
                    None,
                    state.midpoint,
                    state.strength,
                    (
                        state.lower_source_zone_id,
                        state.upper_source_zone_id,
                        *state.lower_source_member_swing_ids,
                        *state.upper_source_member_swing_ids,
                    ),
                    {
                        "protocol_hash": state.protocol_hash,
                        "lower_bound": state.lower_bound,
                        "upper_bound": state.upper_bound,
                        "value_price": state.value_price,
                        "candidate_real_h1_bars": (
                            state.candidate_real_h1_bars
                        ),
                        "lower_touch_count": state.lower_touch_count,
                        "upper_touch_count": state.upper_touch_count,
                        "midpoint_crossings": (
                            state.midpoint_crossings
                        ),
                        "inside_close_fraction": (
                            state.inside_close_fraction
                        ),
                        "compression_ratio": (
                            state.compression_ratio
                        ),
                        "age_h1_bars": state.age_h1_bars,
                    },
                    entity_id=state.range_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.mature_at,
                    ended_at=state.broken_at if terminal else None,
                    transition_reason=state.transition_reason,
                ),
                include_in_recent=False,
            )

        for state in update.manipulation_transitions:
            terminal = state.lifecycle in {
                ManipulationLifecycle.REACCEPTED,
                ManipulationLifecycle.ACCEPTED_OUTSIDE,
            } or state.deadline_elapsed
            if (
                (terminal and not include_resolutions)
                or (not terminal and not include_creations)
            ):
                continue
            self.memory.append(
                _event(
                    EventKind.MANIPULATION_STATE,
                    (
                        state.censored_at
                        if state.deadline_elapsed
                        else state.state_started_at
                    ),
                    Timeframe.M1,
                    state.side,
                    (
                        state.reentry_price
                        if state.reentry_price is not None
                        else state.sweep_extreme
                    ),
                    state.strength,
                    (
                        state.source_inventory_item_id,
                        *state.crossed_source_ids,
                    ),
                    {
                        "protocol_hash": state.protocol_hash,
                        "source_kind": state.source_kind,
                        "source_timeframe": (
                            state.source_timeframe.value
                        ),
                        "source_lower_bound": (
                            state.source_lower_bound
                        ),
                        "source_upper_bound": (
                            state.source_upper_bound
                        ),
                        "sweep_extreme": state.sweep_extreme,
                        "close_outside_on_sweep": (
                            state.close_outside_on_sweep
                        ),
                        "outside_completed_bars": (
                            state.outside_completed_bars
                        ),
                        "outside_run": state.outside_run,
                        "outside_run_side": state.outside_run_side,
                        "reentry_candidate_at": (
                            None
                            if state.reentry_candidate_at is None
                            else state.reentry_candidate_at.isoformat()
                        ),
                        "reentry_candidate_price": (
                            state.reentry_candidate_price
                        ),
                        "inside_hold_bars": state.inside_hold_bars,
                        "reentry_failed_at": (
                            None
                            if state.reentry_failed_at is None
                            else state.reentry_failed_at.isoformat()
                        ),
                        "deadline_at": (
                            None
                            if state.deadline_at is None
                            else state.deadline_at.isoformat()
                        ),
                        "deadline_elapsed": state.deadline_elapsed,
                        "penetration_atr": state.penetration_atr,
                        "age_1m_bars": state.age_1m_bars,
                        "resolved_side": state.resolved_side,
                    },
                    entity_id=state.manipulation_id,
                    lifecycle=(
                        "censored"
                        if state.deadline_elapsed
                        else state.lifecycle.value
                    ),
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=(
                        state.censored_at
                        if state.deadline_elapsed
                        else state.resolved_at if terminal else None
                    ),
                    transition_reason=state.transition_reason,
                ),
                include_in_recent=False,
                sequence_floor=(
                    None
                    if terminal
                    else EventMemory._GROUP4_CREATION_SEQUENCE_FLOOR
                ),
            )

    def _record_group5_events(self, update: Group5Update) -> None:
        if update.boundary_reason is not None:
            return

        active = tuple(
            state
            for state in update.path_transitions
            if state.lifecycle is PathSequenceLifecycle.ACTIVE
        )
        terminal = tuple(
            state
            for state in update.path_transitions
            if state.lifecycle is not PathSequenceLifecycle.ACTIVE
        )

        def append_path(state: PathSequenceState) -> None:
            self.memory.append(
                _event(
                    EventKind.ENTRY_PATH_STATE,
                    state.state_started_at,
                    Timeframe.M1,
                    (
                        "above"
                        if state.direction is Direction.LONG
                        else "below"
                    ),
                    None,
                    0.0,
                    tuple(step.step_id for step in state.steps),
                    {
                        "protocol_hash": state.protocol_hash,
                        "context_kind": state.context_kind,
                        "context_id": state.context_id,
                        "age_real_1m_bars": state.age_real_1m_bars,
                        "state_duration_real_1m_bars": (
                            state.state_duration_real_1m_bars
                        ),
                        "step_count": len(state.steps),
                    },
                    entity_id=state.sequence_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.formed_at,
                    ended_at=state.ended_at,
                    direction=state.direction,
                    transition_reason=state.transition_reason,
                ),
                include_in_recent=False,
            )

        for state in sorted(
            active,
            key=lambda item: (item.formed_at, item.sequence_id),
        ):
            append_path(state)
        for sequence_id, step in update.step_transitions:
            source_ids = (
                sequence_id,
                step.step_id,
                step.source_entity_id,
                *(
                    ()
                    if step.source_event_id is None
                    else (step.source_event_id,)
                ),
                *step.predecessor_step_ids,
            )
            self.memory.append(
                _event(
                    EventKind.ENTRY_PATH_STEP,
                    step.observed_at,
                    Timeframe.M1,
                    (
                        "above"
                        if step.direction is Direction.LONG
                        else "below"
                    ),
                    None,
                    step.strength,
                    source_ids,
                    {
                        "sequence_id": sequence_id,
                        "step_id": step.step_id,
                        "kind": step.kind,
                        "source_event_id": step.source_event_id,
                        "source_entity_id": step.source_entity_id,
                        "predecessor_step_ids": (
                            step.predecessor_step_ids
                        ),
                        "same_clock_relation": (
                            step.same_clock_relation
                        ),
                        "reason": step.reason,
                    },
                    direction=step.direction,
                    transition_reason=step.reason,
                )
            )
        for state in sorted(
            terminal,
            key=lambda item: (
                item.ended_at,
                item.sequence_id,
            ),
        ):
            append_path(state)

    @staticmethod
    def _group3_boundary_reason(
        anomalies: Sequence[str],
    ) -> str | None:
        if CONTRACT_BOUNDARY in anomalies:
            return "contract_change_reset"
        if DATA_GAP_BOUNDARY in anomalies:
            return "data_gap_reset"
        if any(
            value in REGISTERED_CLOSURE_ANOMALIES
            for value in anomalies
        ):
            return "registered_session_reset"
        return None

    def _observe_group3(
        self,
        update: ReaderUpdate,
        frames: Mapping[Timeframe, FrameObservation],
    ) -> Group3Update | None:
        if self._group3_tracker is None:
            return None
        if self._displacement_eye is None:
            raise RuntimeError(
                "Group 3 lost its configured displacement source"
            )
        boundary = self._group3_boundary_reason(update.anomalies)
        if boundary is not None:
            return self._visible_group3_update(
                self._group3_tracker.on_boundary(
                    boundary,
                    update.asof,
                )
            )
        batch = self._displacement_eye.last_batch
        expected = tuple(
            update.newly_completed.get(Timeframe.M5, ())
        )
        if tuple(candle for candle, _ in batch) != expected:
            raise RuntimeError(
                "Group 3 and displacement completed-M5 batches diverged"
            )
        result = Group3Update(
            *self._group3_tracker.snapshot(),
        )
        bos_states = frames[Timeframe.M5].structure_breaks
        for candle, displacement_update in batch:
            result = self._group3_tracker.on_completed_5m(
                candle,
                displacement_update,
                tuple(
                    Group3BOSSource(
                        state=bos,
                        symbol=candle.symbol,
                        instrument_id=candle.instrument_id,
                        protocol_hash=(
                            self._structure_trackers[
                                Timeframe.M5
                            ].config.protocol_hash
                        ),
                        tick_size=self.config.tick_size,
                    )
                    for bos in bos_states
                    if (
                        bos.lifecycle is BOSLifecycle.CONFIRMED
                        and bos.resolved_at == candle.end
                    )
                ),
            )
        return self._visible_group3_update(result)

    def _visible_group3_update(
        self,
        update: Group3Update,
    ) -> Group3Update:
        retained_ids = {
            state.fvg_id
            for state in update.fair_value_gaps
        } | {
            state.order_block_id
            for state in update.order_blocks
        }
        self._group3_hidden_entity_ids.intersection_update(
            retained_ids
        )
        if update.boundary_reason in FVG_BOUNDARY_REASONS:
            self._group3_hidden_entity_ids.update(retained_ids)
        return replace(
            update,
            fair_value_gaps=tuple(
                state
                for state in update.fair_value_gaps
                if state.fvg_id
                not in self._group3_hidden_entity_ids
            ),
            order_blocks=tuple(
                state
                for state in update.order_blocks
                if state.order_block_id
                not in self._group3_hidden_entity_ids
            ),
        )

    @staticmethod
    def _retained_timeline_keys(
        frames: Mapping[Timeframe, FrameObservation],
        liquidity_pool_states: Sequence[LiquidityPoolState],
        manipulations: Sequence[ManipulationState] = (),
        path_sequences: Sequence[PathSequenceState] = (),
    ) -> set[str]:
        keys: set[str] = set()
        for frame in frames.values():
            keys.update(
                f"swing:{item.swing_id}"
                for item in frame.swings
            )
            keys.update(
                f"structure:{item.structure_id}"
                for item in frame.structures
                if (
                    item.structure_id is not None
                    and item.lifecycle
                    is not StructureLifecycle.INACTIVE
                )
            )
            keys.update(
                f"bos:{item.bos_id}"
                for item in frame.structure_breaks
            )
            keys.update(
                f"zone:{item.zone_id}"
                for item in frame.support_resistance
            )
            keys.update(
                f"fvg:{item.fvg_id}"
                for item in frame.fair_value_gaps
            )
            keys.update(
                f"order_block:{item.order_block_id}"
                for item in frame.order_blocks
            )
            keys.update(
                f"range:{item.range_id}"
                for item in frame.dealing_ranges
            )
        keys.update(
            f"pool:{item.pool_id}"
            for item in liquidity_pool_states
        )
        keys.update(
            f"manipulation:{item.manipulation_id}"
            for item in manipulations
        )
        keys.update(
            f"entry_path:{item.sequence_id}"
            for item in path_sequences
        )
        return keys

    @staticmethod
    def _pool_formation_source(
        pool,
    ):
        """Return the frozen native-cutoff formation view of one pool."""

        return replace(
            pool,
            lifecycle=LiquidityPoolLifecycle.FORMED,
            swept_at=None,
            sweep_extreme=None,
            close_outside_on_sweep=None,
            resolved_at=None,
            resolution_reason=None,
        )

    @staticmethod
    def _formation_inventory_item(
        item: LiquidityInventoryItem,
    ) -> LiquidityInventoryItem:
        if item.kind not in {"equal_highs", "equal_lows"}:
            return item
        return replace(
            item,
            lifecycle=LiquidityInventoryLifecycle.VISIBLE,
            targeted_at=None,
            consumed_at=None,
            lifecycle_reason=None,
        )

    def _projected_pool_tracker(
        self,
        item: LiquidityInventoryItem,
    ) -> tuple[CausalLiquidityTracker, str]:
        prefix = "pool:"
        if (
            item.kind not in {"equal_highs", "equal_lows"}
            or not item.item_id.startswith(prefix)
        ):
            raise RuntimeError("projected pool inventory identity is invalid")
        tracker = self._liquidity_trackers.get(item.timeframe)
        if tracker is None:
            raise RuntimeError("projected pool has no formation tracker")
        return tracker, item.item_id[len(prefix) :]

    def _liquidity_snapshot(
        self,
        timeframe: Timeframe,
        tracker: CausalLiquidityTracker,
    ) -> tuple:
        """Return one native snapshot per changed timeframe cutoff.

        The 1m tracker remains intentionally live on every observation.  A
        higher-timeframe cache entry is removed whenever completed-1m pool
        projection changes that tracker, so reuse cannot hide a sweep or its
        subsequent resolution.
        """

        cached = self._liquidity_snapshot_cache.get(timeframe)
        if (
            timeframe is not Timeframe.M1
            and cached is not None
            and cached[0] == tracker.last_end
        ):
            return cached[1]
        snapshot = tracker.snapshot()
        self._liquidity_snapshot_cache[timeframe] = (
            tracker.last_end,
            snapshot,
        )
        return snapshot

    def _invalidate_liquidity_snapshot(
        self,
        timeframe: Timeframe,
    ) -> None:
        self._liquidity_snapshot_cache.pop(timeframe, None)

    @staticmethod
    def _inventory_crossed(
        item: LiquidityInventoryItem,
        candle: Candle,
    ) -> bool:
        return (
            candle.high > item.upper_bound
            if item.side == "above"
            else candle.low < item.lower_bound
        )

    @staticmethod
    def _reference_period_keys(
        candle: Candle,
    ) -> dict[str, str]:
        local = candle.start.tz_convert("America/New_York")
        civil_date = local.date()
        session_date = (
            local.normalize() + pd.Timedelta(days=1)
            if local.hour >= 18
            else local.normalize()
        ).date()
        iso_year, iso_week, _ = session_date.isocalendar()
        return {
            "session": session_date.isoformat(),
            "day": civil_date.isoformat(),
            "week": f"{iso_year:04d}-W{iso_week:02d}",
        }

    @staticmethod
    def _reference_period_start(
        family: str,
        candle: Candle,
    ) -> pd.Timestamp:
        local = candle.start.tz_convert("America/New_York")
        session_date = (
            local.normalize() + pd.Timedelta(days=1)
            if local.hour >= 18
            else local.normalize()
        )
        if family == "session":
            return session_date - pd.Timedelta(hours=6)
        if family == "day":
            return local.normalize()
        if family == "week":
            monday = session_date.normalize() - pd.Timedelta(
                days=session_date.weekday()
            )
            return monday - pd.Timedelta(hours=6)
        raise ValueError("unknown reference period family")

    def _materialize_reference_period(
        self,
        family: str,
        period: _ReferencePeriod,
    ) -> None:
        # These extrema are aggregated and projected by the completed-1m
        # clock; period identity lives in ``kind`` rather than pretending
        # they were confirmed by an H1/H4 candle.
        timeframe = Timeframe.M1
        visibility = {
            "session": 0.75,
            "day": 0.90,
            "week": 1.00,
        }[family]
        for side, suffix, price in (
            ("above", "high", period.high),
            ("below", "low", period.low),
        ):
            kind = f"previous_{family}_{suffix}"
            item_id = (
                f"reference:{family}:{period.key}:{suffix}:"
                f"{period.symbol}:{period.instrument_id}"
            )
            source_id = (
                f"reference_source:{family}:{period.key}:{suffix}:"
                f"{period.symbol}:{period.instrument_id}"
            )
            self._reference_inventory[item_id] = LiquidityInventoryItem(
                item_id=item_id,
                timeframe=timeframe,
                side=side,
                kind=kind,
                price=price,
                lower_bound=price,
                upper_bound=price,
                formed_at=period.started_at,
                confirmed_at=period.last_end,
                lifecycle=LiquidityInventoryLifecycle.VISIBLE,
                source_ids=(source_id,),
                age_bars=0,
                strength=visibility,
                structural_rank="external",
                is_protected_swing=False,
                visibility_strength=visibility,
            )

    def _advance_reference_periods(
        self,
        candle: Candle,
        *,
        append_retirement_events: bool,
    ) -> None:
        if not candle.real_completed:
            return
        self._reference_inventory = {
            item_id: replace(item, age_bars=item.age_bars + 1)
            for item_id, item in self._reference_inventory.items()
        }
        keys = self._reference_period_keys(candle)
        for family, key in keys.items():
            current = self._reference_periods.get(family)
            if current is None:
                period_start = self._reference_period_start(
                    family,
                    candle,
                )
                self._reference_periods[family] = _ReferencePeriod(
                    key=key,
                    started_at=period_start,
                    last_end=candle.end,
                    high=float(candle.high),
                    low=float(candle.low),
                    symbol=candle.symbol,
                    instrument_id=int(candle.instrument_id),
                    coverage_complete=bool(
                        self._reference_coverage_start is not None
                        and self._reference_coverage_start
                        <= period_start
                    ),
                )
                continue
            if current.key == key:
                self._reference_periods[family] = replace(
                    current,
                    last_end=candle.end,
                    high=max(current.high, float(candle.high)),
                    low=min(current.low, float(candle.low)),
                )
                continue
            retired = tuple(
                item
                for item in self._reference_inventory.values()
                if item.kind
                in {
                    f"previous_{family}_high",
                    f"previous_{family}_low",
                }
            )
            self._reference_inventory = {
                item_id: item
                for item_id, item in self._reference_inventory.items()
                if item.kind
                not in {
                    f"previous_{family}_high",
                    f"previous_{family}_low",
                }
            }
            if append_retirement_events:
                for item in retired:
                    self.memory.append(
                        _event(
                            EventKind.LIQUIDITY_RETIRED,
                            candle.end,
                            Timeframe.M1,
                            item.side,
                            item.price,
                            0.0,
                            (item.item_id,),
                            {
                                "source_kind": item.kind,
                                "replacement_period": current.key,
                            },
                            transition_reason=(
                                "reference_period_replaced"
                            ),
                        )
                    )
            if current.coverage_complete:
                self._materialize_reference_period(family, current)
            period_start = self._reference_period_start(
                family,
                candle,
            )
            self._reference_periods[family] = _ReferencePeriod(
                key=key,
                started_at=period_start,
                last_end=candle.end,
                high=float(candle.high),
                low=float(candle.low),
                symbol=candle.symbol,
                instrument_id=int(candle.instrument_id),
                coverage_complete=True,
            )

    @staticmethod
    def _pool_close_outside(
        item: LiquidityInventoryItem,
        candle: Candle,
    ) -> bool:
        return (
            candle.close > item.upper_bound
            if item.side == "above"
            else candle.close < item.lower_bound
        )

    def _append_inventory_crossing_event(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
        *,
        atr: float,
    ) -> None:
        outside = self._pool_close_outside(item, candle)
        extreme = candle.high if item.side == "above" else candle.low
        distance = (
            extreme - item.upper_bound
            if item.side == "above"
            else item.lower_bound - extreme
        )
        self.memory.append(
            _event(
                (
                    EventKind.LIQUIDITY_CONSUMED
                    if outside
                    else EventKind.LIQUIDITY_SWEEP
                ),
                candle.end,
                Timeframe.M1,
                item.side,
                extreme,
                clamp(distance / max(atr, self.config.tick_size)),
                (item.item_id,),
                {
                    "source_timeframe": item.timeframe.value,
                    "source_kind": item.kind,
                    "close_accepted_outside": outside,
                    "frozen_lower_bound": item.lower_bound,
                    "frozen_upper_bound": item.upper_bound,
                },
            )
        )

    def _append_projected_pool_sweep_events(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
        *,
        atr: float,
    ) -> None:
        self._append_inventory_crossing_event(
            item,
            candle,
            atr=atr,
        )
        outside = self._pool_close_outside(item, candle)
        extreme = candle.high if item.side == "above" else candle.low
        self.memory.append(
            _event(
                EventKind.LIQUIDITY_POOL_STATE,
                candle.end,
                item.timeframe,
                item.side,
                extreme,
                item.strength,
                item.source_ids,
                {
                    "source_timeframe": item.timeframe.value,
                    "pool_item_id": item.item_id,
                    "close_outside_on_sweep": outside,
                },
                entity_id=item.item_id,
                lifecycle=LiquidityPoolLifecycle.SWEPT.value,
                formed_at=item.formed_at,
                confirmed_at=item.confirmed_at,
            )
        )

    def _append_projected_pool_resolution_event(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
    ) -> None:
        outside = self._pool_close_outside(item, candle)
        self.memory.append(
            _event(
                EventKind.LIQUIDITY_POOL_STATE,
                candle.end,
                item.timeframe,
                item.side,
                candle.close,
                item.strength,
                item.source_ids,
                {
                    "source_timeframe": item.timeframe.value,
                    "pool_item_id": item.item_id,
                },
                entity_id=item.item_id,
                lifecycle=(
                    LiquidityPoolLifecycle.ACCEPTED.value
                    if outside
                    else LiquidityPoolLifecycle.REJECTED.value
                ),
                formed_at=item.formed_at,
                confirmed_at=item.confirmed_at,
                ended_at=candle.end,
                transition_reason=(
                    "close_held_outside"
                    if outside
                    else "close_returned_inside"
                ),
            )
        )

    def _bootstrap_inventory_lifecycles(
        self,
        update: ReaderUpdate,
        base_inventory: Sequence[LiquidityInventoryItem],
        *,
        projected_pool_states: dict[str, LiquidityPoolState] | None = None,
    ) -> None:
        """Rebuild exact draw transitions when an observer first attaches.

        Native higher-timeframe trackers know formation causally but can only
        timestamp a crossing at their coarse bar close. The retained completed
        1m prefix is the sole authority for every draw's first crossing and a
        pool's next-real-bar acceptance/rejection decision.
        """

        m1_history = tuple(update.histories.get(Timeframe.M1, ()))
        real_history = tuple(
            candle
            for candle in m1_history
            if candle.real_completed and candle.end <= update.asof
        )
        if not real_history:
            if base_inventory:
                raise LiquidityProtocolError(
                    "cannot recover liquidity inventory without a "
                    "completed 1m prefix"
                )
            return
        for item in base_inventory:
            if item.kind not in _INVENTORY_KINDS:
                raise LiquidityProtocolError(
                    "inventory draw kind is not enabled for exact 1m "
                    "lifecycle projection"
                )
            if m1_history[0].start > item.confirmed_at:
                raise LiquidityProtocolError(
                    "retained completed 1m prefix starts after an "
                    "inventory draw was confirmed"
                )
            eligible = [
                (index, candle)
                for index, candle in enumerate(real_history)
                if (
                    candle.start >= item.confirmed_at
                )
            ]
            sweep_match = next(
                (
                    (index, candle)
                    for index, candle in eligible
                    if self._inventory_crossed(item, candle)
                ),
                None,
            )
            if sweep_match is None:
                if (
                    item.lifecycle
                    is LiquidityInventoryLifecycle.CONSUMED
                ):
                    raise LiquidityProtocolError(
                        "cannot recover an already-consumed inventory draw "
                        "from the retained completed 1m prefix"
                    )
                continue
            sweep_index, sweep_candle = sweep_match
            if sweep_index + 1 < self.config.atr_period + 1:
                raise LiquidityProtocolError(
                    "retained completed 1m prefix lacks the registered "
                    "ATR warmup before the reconstructed draw crossing"
                )
            if (
                item.consumed_at is not None
                and sweep_candle.end > item.consumed_at
            ):
                raise LiquidityProtocolError(
                    "reconstructed draw crossing postdates its native "
                    "completed-bar transition"
                )
            if item.kind not in {"equal_highs", "equal_lows"}:
                self._inventory_consumption[item.item_id] = (
                    sweep_candle.end,
                    (
                        "swing_swept"
                        if item.kind == "swing"
                        else "reference_level_swept"
                    ),
                )
                self._append_inventory_crossing_event(
                    item,
                    sweep_candle,
                    atr=_atr(
                        real_history[: sweep_index + 1],
                        self.config.atr_period,
                    ),
                )
                continue
            resolution_candle = next(
                (
                    candle
                    for index, candle in eligible
                    if index > sweep_index
                ),
                None,
            )
            outside_on_sweep = self._pool_close_outside(
                item,
                sweep_candle,
            )
            tracker, pool_id = self._projected_pool_tracker(item)
            projected = tracker.bootstrap_pool_projection(
                pool_id,
                swept_at=sweep_candle.end,
                sweep_extreme=(
                    sweep_candle.high
                    if item.side == "above"
                    else sweep_candle.low
                ),
                close_outside_on_sweep=outside_on_sweep,
                resolved_at=(
                    None
                    if resolution_candle is None
                    else resolution_candle.end
                ),
                accepted_outside=(
                    None
                    if resolution_candle is None
                    else self._pool_close_outside(
                        item,
                        resolution_candle,
                    )
                ),
            )
            self._invalidate_liquidity_snapshot(item.timeframe)
            if projected_pool_states is not None:
                projected_pool_states[projected.pool_id] = projected
            self._inventory_consumption[item.item_id] = (
                sweep_candle.end,
                "pool_swept",
            )
            self._append_projected_pool_sweep_events(
                item,
                sweep_candle,
                atr=_atr(
                    real_history[: sweep_index + 1],
                    self.config.atr_period,
                ),
            )
            if resolution_candle is None:
                self._pending_pool_sweeps[item.item_id] = (
                    sweep_candle.end,
                    self._formation_inventory_item(item),
                )
            else:
                self._append_projected_pool_resolution_event(
                    item,
                    resolution_candle,
                )

    def _resolve_pending_pool_sweeps(
        self,
        update: ReaderUpdate,
        *,
        append_events: bool = True,
        projected_pool_states: dict[str, LiquidityPoolState] | None = None,
    ) -> tuple[tuple[LiquidityInventoryItem, Candle], ...]:
        """Resolve pending sweeps before any same-clock HTF state update."""

        if type(append_events) is not bool:
            raise TypeError("pool-resolution event flag must be boolean")
        bar = update.completed_1m
        if not bar.real_completed:
            return ()
        resolved: list[tuple[LiquidityInventoryItem, Candle]] = []
        for item_id, (swept_at, item) in tuple(
            self._pending_pool_sweeps.items()
        ):
            if update.asof <= swept_at:
                continue
            outside = self._pool_close_outside(item, bar)
            tracker, pool_id = self._projected_pool_tracker(item)
            projected = tracker.project_pool_resolution(
                pool_id,
                observed_at=update.asof,
                accepted_outside=outside,
            )
            self._invalidate_liquidity_snapshot(item.timeframe)
            if projected_pool_states is not None:
                projected_pool_states[projected.pool_id] = projected
            if append_events:
                self._append_projected_pool_resolution_event(item, bar)
            else:
                resolved.append((item, bar))
            self._pending_pool_sweeps.pop(item_id, None)
        return tuple(resolved)

    def _project_inventory(
        self,
        update: ReaderUpdate,
        frames: Mapping[Timeframe, FrameObservation],
        base_inventory: Sequence[LiquidityInventoryItem],
        *,
        projected_pool_states: dict[str, LiquidityPoolState] | None = None,
    ) -> tuple[LiquidityInventoryItem, ...]:
        """Apply the completed 1m path to native pool formation sources.

        A higher-timeframe frame keeps its own completed-bar cutoff and exposes
        only the frozen pool formation/source.  The current-minute lifecycle is
        authoritative in this top-level inventory and its typed event sequence;
        it must never be projected into an unfinished HTF frame.
        """

        bar = update.completed_1m
        if any(
            item.kind not in _INVENTORY_KINDS
            for item in base_inventory
        ):
            raise LiquidityProtocolError(
                "inventory draw kind is not enabled for exact 1m "
                "lifecycle projection"
            )
        base_ids = {item.item_id for item in base_inventory}
        if self._prior is None and self._liquidity_trackers:
            self._bootstrap_inventory_lifecycles(
                update,
                base_inventory,
                projected_pool_states=projected_pool_states,
            )
        may_transition = bar.real_completed
        self._resolve_pending_pool_sweeps(
            update,
            projected_pool_states=projected_pool_states,
        )

        successful_prior_by_id = {
            item.item_id: item
            for item in (
                ()
                if self._prior is None
                else self._prior.liquidity_inventory
            )
            if item.item_id in base_ids
        }
        prior_by_id = dict(successful_prior_by_id)
        for item in base_inventory:
            if item.confirmed_at <= bar.start:
                prior_by_id.setdefault(item.item_id, item)
        prior_inventory = tuple(prior_by_id.values())
        atr = frames[Timeframe.M1].metrics["atr"]
        for item in prior_inventory:
            if (
                not may_transition
                or
                item.lifecycle
                is not LiquidityInventoryLifecycle.VISIBLE
                or item.confirmed_at > bar.start
                or item.item_id in self._inventory_consumption
            ):
                continue
            if not self._inventory_crossed(item, bar):
                continue
            reason = (
                "pool_swept"
                if item.kind in {"equal_highs", "equal_lows"}
                else (
                    "swing_swept"
                    if item.kind == "swing"
                    else "reference_level_swept"
                )
            )
            self._inventory_consumption[item.item_id] = (
                update.asof,
                reason,
            )
            extreme = bar.high if item.side == "above" else bar.low
            if item.kind in {"equal_highs", "equal_lows"}:
                accepted_outside = self._pool_close_outside(
                    item,
                    bar,
                )
                tracker, pool_id = self._projected_pool_tracker(item)
                projected = tracker.project_pool_sweep(
                    pool_id,
                    observed_at=update.asof,
                    sweep_extreme=extreme,
                    close_outside_on_sweep=accepted_outside,
                )
                self._invalidate_liquidity_snapshot(item.timeframe)
                if projected_pool_states is not None:
                    projected_pool_states[projected.pool_id] = projected
                self._append_projected_pool_sweep_events(
                    item,
                    bar,
                    atr=atr,
                )
                self._pending_pool_sweeps[item.item_id] = (
                    update.asof,
                    item,
                )
            else:
                self._append_inventory_crossing_event(
                    item,
                    bar,
                    atr=atr,
                )

        output: list[LiquidityInventoryItem] = []
        for item in base_inventory:
            native_consumed = (
                None
                if item.lifecycle
                is not LiquidityInventoryLifecycle.CONSUMED
                else (
                    item.consumed_at,
                    item.lifecycle_reason,
                )
            )
            projected = self._inventory_consumption.get(item.item_id)
            if (
                item.kind in {"equal_highs", "equal_lows"}
                and projected is not None
            ):
                # Exact completed-1m projection is authoritative over a
                # native higher-timeframe lifecycle timestamp.
                candidates = [projected]
            else:
                candidates = [
                    value
                    for value in (native_consumed, projected)
                    if value is not None and value[0] is not None
                ]
            prior_item = successful_prior_by_id.get(item.item_id)
            if (
                prior_item is not None
                and prior_item.lifecycle
                is LiquidityInventoryLifecycle.CONSUMED
            ):
                if any(
                    consumed_at < prior_item.consumed_at
                    for consumed_at, _ in candidates
                ):
                    raise LiquidityProtocolError(
                        "later inventory state attempted to backdate a "
                        "frozen consumption"
                    )
                # Consumption is terminal.  Freeze the complete item at its
                # first completed-1m crossing so later structural-rank, age
                # or visibility metadata cannot rewrite that market fact.
                output.append(prior_item)
                continue
            if not candidates:
                output.append(item)
                continue
            consumed_at, reason = min(
                candidates,
                key=lambda value: value[0],
            )
            output.append(
                replace(
                    item,
                    lifecycle=LiquidityInventoryLifecycle.CONSUMED,
                    targeted_at=None,
                    consumed_at=consumed_at,
                    lifecycle_reason=reason,
                )
            )
        self._inventory_consumption = {
            item_id: value
            for item_id, value in self._inventory_consumption.items()
            if item_id in base_ids
        }
        return tuple(
            sorted(
                output,
                key=lambda item: (
                    item.confirmed_at,
                    item.timeframe.value,
                    item.item_id,
                ),
            )
        )

    def observe(
        self,
        update: ReaderUpdate,
        reality: ExecutionRealityInput | None = None,
    ) -> MarketObservation:
        if self._terminal_failure is not None:
            raise RuntimeError(self._terminal_failure)
        if self._prior is not None and update.asof <= self._prior.asof:
            raise ValueError(
                "observer received a duplicate or out-of-order "
                "successfully committed update"
            )
        update_timeframes = tuple(update.active_timeframes)
        if set(update_timeframes) != set(self._active_timeframes):
            raise ValueError(
                "reader and observer enabled scale registries disagree"
            )
        if (
            tuple(update.scale_specs) != self.scale_specs
            or update.scale_registry_id != self._scale_registry_id
        ):
            raise ValueError(
                "reader and observer scale registry contracts disagree"
            )
        reality = reality or ExecutionRealityInput()
        execution = self._execution(update, reality)
        if self.memory.clock_coverage_start is None:
            m1_history = tuple(
                update.histories.get(Timeframe.M1, ())
            )
            coverage_start = (
                m1_history[0].start
                if m1_history
                else update.completed_1m.start
            )
            try:
                self.memory.set_clock_coverage_start(coverage_start)
            except Exception:
                self._terminal_failure = (
                    "event memory cannot prove retained 1m clock "
                    "coverage; discard this observer and replay from "
                    "an earlier checkpoint"
                )
                raise
        if self._displacement_eye is None:
            displacement = None
        else:
            displacement_input = (
                update.asof,
                update.completed_1m,
                tuple(update.anomalies),
                tuple(
                    update.newly_completed.get(Timeframe.M5, ())
                ),
            )
            if (
                displacement_input == self._last_displacement_input
                and self._last_displacement_observation is not None
            ):
                displacement = self._last_displacement_observation
            else:
                displacement = self._displacement_eye.on_update(update)
                self._last_displacement_input = displacement_input
                self._last_displacement_observation = displacement
        reset_anomalies = tuple(
            value
            for value in update.anomalies
            if value
            in {
                "contract_change_history_reset",
                "data_gap_history_reset",
            }
        )
        if reset_anomalies:
            boundary_identity = (update.asof, reset_anomalies)
            if self._boundary_reset_identity != boundary_identity:
                reason = (
                    "contract_change_reset"
                    if "contract_change_history_reset"
                    in reset_anomalies
                    else "data_gap_reset"
                )
                try:
                    self._reset_contract_state(
                        reason=reason,
                        observed_at=update.asof,
                        reset_anomalies=reset_anomalies,
                        boundary_symbol=update.completed_1m.symbol,
                        boundary_instrument_id=(
                            update.completed_1m.instrument_id
                        ),
                    )
                except Exception:
                    self._terminal_failure = (
                        "boundary reset failed after state may have "
                        "changed; discard this observer and resume from "
                        "the last checkpoint"
                    )
                    raise
                self._boundary_reset_identity = boundary_identity
        try:
            if self.memory.last_minute_end is None:
                clock_history = tuple(
                    candle
                    for candle in update.histories.get(Timeframe.M1, ())
                    if candle.end <= update.asof
                )
                for candle in (
                    clock_history
                    if clock_history
                    else (update.completed_1m,)
                ):
                    self.memory.observe_minute(candle)
            else:
                self.memory.observe_minute(update.completed_1m)
        except Exception:
            self._terminal_failure = (
                "event-memory clock update failed after state may have "
                "changed; discard this observer and resume from the "
                "last checkpoint"
            )
            raise
        # A pending sweep is resolved by this completed real 1m bar before a
        # same-clock 5m/1H/4H candle can admit a new structural touch.
        try:
            deferred_pool_resolution_events = (
                self._resolve_pending_pool_sweeps(
                    update,
                    append_events=False,
                )
            )
        except Exception:
            self._terminal_failure = (
                "liquidity projection failed after state may have "
                "changed; discard this observer and resume from the "
                "last checkpoint"
            )
            raise
        histories = update.histories
        frames: dict[Timeframe, FrameObservation] = {}
        base_inventory: list[LiquidityInventoryItem] = []
        liquidity_snapshots: dict[Timeframe, tuple] = {}
        for timeframe in self._active_timeframes:
            prior = self._prior_frame_if_unchanged(update, timeframe)
            if prior is not None:
                frames[timeframe] = prior
                continue
            frame = self._observe_frame(
                timeframe,
                histories[timeframe],
                update.asof,
                self.config,
            )
            tracker = self._structure_trackers.get(timeframe)
            if tracker is None:
                frames[timeframe] = frame
                continue
            liquidity_tracker = self._liquidity_trackers.get(timeframe)
            self._sync_structure_tracker(
                update,
                timeframe,
                tracker,
                liquidity_tracker,
            )
            swings, structures, structure_breaks = tracker.snapshot()
            if liquidity_tracker is None:
                support_resistance = ()
                liquidity_pools = ()
                native_inventory = ()
            else:
                (
                    support_resistance,
                    liquidity_pools,
                    native_inventory,
                ) = self._liquidity_snapshot(
                    timeframe,
                    liquidity_tracker,
                )
                liquidity_snapshots[timeframe] = (
                    support_resistance,
                    liquidity_pools,
                    native_inventory,
                )
            terminal_breaks = self._boundary_terminal_breaks.get(
                timeframe,
                (),
            )
            visible_breaks = (
                *structure_breaks,
                *terminal_breaks,
            )
            metrics = {
                **frame.metrics,
                **_structure_metrics_from_parts(
                    structures,
                    visible_breaks,
                ),
            }
            atr = metrics["atr"]
            high_step, low_step, progression = _typed_progression(
                swings,
                atr,
            )
            semantic_tail = next(
                (
                    candle
                    for candle in reversed(histories[timeframe])
                    if candle.real_completed
                ),
                None,
            )
            if timeframe is Timeframe.H4:
                metrics["structure_direction"] = metrics[
                    "confirmed_structure_direction"
                ]
                confirmed_structures = tuple(
                    item
                    for item in structures
                    if item.lifecycle is StructureLifecycle.CONFIRMED
                )
                metrics["structure_age_bars"] = (
                    float(confirmed_structures[0].age_bars)
                    if len(confirmed_structures) == 1
                    else 0.0
                )
                if semantic_tail is not None:
                    (
                        metrics["external_above_distance_atr"],
                        metrics["external_below_distance_atr"],
                        above_count,
                        below_count,
                    ) = _external_distances(
                        native_inventory,
                        semantic_tail.close,
                        atr,
                    )
                    metrics["external_above_count"] = float(
                        above_count
                    )
                    metrics["external_below_count"] = float(
                        below_count
                    )
            elif timeframe in {Timeframe.H1, Timeframe.M15}:
                metrics["swing_high_progression"] = high_step
                metrics["swing_low_progression"] = low_step
                metrics["swing_progression"] = progression
                if semantic_tail is not None:
                    (
                        metrics["up_path_obstruction_atr"],
                        metrics["down_path_obstruction_atr"],
                        _,
                        _,
                    ) = _external_distances(
                        native_inventory,
                        semantic_tail.close,
                        atr,
                    )
            frames[timeframe] = replace(
                frame,
                metrics=metrics,
                swings=swings,
                structures=structures,
                structure_breaks=visible_breaks,
                support_resistance=support_resistance,
                # This frame is fixed at its own native completed-bar cutoff.
                # Current 1m sweep/resolution lives only in the top-level
                # inventory and typed event sequence.
                liquidity_pools=tuple(
                    self._pool_formation_source(item)
                    for item in liquidity_pools
                ),
            )
        if Timeframe.M5 in frames:
            frames[Timeframe.M5] = replace(
                frames[Timeframe.M5],
                structure_breaks=self._enrich_mss_breaks(
                    frames[Timeframe.M5].structure_breaks
                ),
            )
        try:
            group3_update = self._observe_group3(update, frames)
            if group3_update is not None:
                frames[Timeframe.M5] = replace(
                    frames[Timeframe.M5],
                    fair_value_gaps=group3_update.fair_value_gaps,
                    order_blocks=group3_update.order_blocks,
                )
        except Exception:
            self._terminal_failure = (
                "Group 3 update failed after a paired reducer may have "
                "advanced; discard this observer and resume from the "
                "last checkpoint"
            )
            raise
        for timeframe, tracker in self._liquidity_trackers.items():
            if timeframe not in liquidity_snapshots:
                liquidity_snapshots[timeframe] = (
                    self._liquidity_snapshot(timeframe, tracker)
                )
            _, _, native_inventory = liquidity_snapshots[timeframe]
            base_inventory.extend(native_inventory)
        base_inventory.extend(self._reference_inventory.values())
        retained_liquidity_entities = {
            item.zone_id
            for frame in frames.values()
            for item in frame.support_resistance
        } | {
            f"pool:{item.pool_id}"
            for frame in frames.values()
            for item in frame.liquidity_pools
        }
        self._liquidity_entity_revisions = {
            entity_id: revision
            for entity_id, revision
            in self._liquidity_entity_revisions.items()
            if entity_id in retained_liquidity_entities
        }
        pre_projection_pool_states = tuple(
            sorted(
                (
                    self._pool_formation_source(pool)
                    for _, pools, _ in liquidity_snapshots.values()
                    for pool in pools
                ),
                key=lambda item: (
                    item.confirmed_at,
                    item.timeframe.value,
                    item.pool_id,
                ),
            )
        )
        group4_update: Group4Update | None = None
        if self._group4_tracker is not None:
            try:
                if self._group4_boundary_update is not None:
                    group4_update = self._group4_boundary_update
                elif self._prior is None:
                    group4_update = (
                        self._group4_tracker
                        .bootstrap_completed_1m_prefix(
                            update.histories.get(Timeframe.M1, ()),
                            pool_inventory=base_inventory,
                            liquidity_pools=pre_projection_pool_states,
                        )
                    )
                else:
                    completed_h1 = tuple(
                        update.newly_completed.get(Timeframe.H1, ())
                    )
                    if len(completed_h1) > 1:
                        raise RuntimeError(
                            "one 1m update emitted multiple H1 candles"
                        )
                    group4_update = (
                        self._group4_tracker.on_completed_update(
                            update.completed_1m,
                            prior_inventory=(
                                self._prior.liquidity_inventory
                            ),
                            liquidity_pools=pre_projection_pool_states,
                            completed_h1=(
                                completed_h1[0]
                                if completed_h1
                                else None
                            ),
                            h1_support_resistance=(
                                frames[
                                    Timeframe.H1
                                ].support_resistance
                            ),
                        )
                    )
                frames[Timeframe.H1] = replace(
                    frames[Timeframe.H1],
                    dealing_ranges=group4_update.dealing_ranges,
                )
                # Append each manipulation timeline in lifecycle order now.
                # New SWEPT events carry a high same-clock sequence floor,
                # so inventory, HTF sources and ranges still sort before
                # creation; terminal resolutions keep the earliest sequence.
                self._record_group4_events(
                    group4_update,
                    include_ranges=False,
                    include_resolutions=True,
                    include_creations=True,
                )
            except Exception:
                self._terminal_failure = (
                    "Group 4 update failed after a paired reducer may "
                    "have advanced; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        try:
            for item, candle in deferred_pool_resolution_events:
                self._append_projected_pool_resolution_event(
                    item,
                    candle,
                )
            projected_pool_states: dict[str, LiquidityPoolState] = {}
            liquidity_inventory = self._project_inventory(
                update,
                frames,
                base_inventory,
                projected_pool_states=projected_pool_states,
            )
        except Exception:
            self._terminal_failure = (
                "liquidity projection failed after state may have "
                "changed; discard this observer and resume from the "
                "last checkpoint"
            )
            raise
        for timeframe, frame in frames.items():
            first_semantic_snapshot = (
                timeframe not in self._last_frame_cutoff
                and any(
                    candle.real_completed
                    for candle in histories[timeframe]
                )
            )
            newly_completed_real = any(
                candle.real_completed
                for candle in update.newly_completed.get(timeframe, ())
            )
            event_clock = next(
                (
                    candle.end
                    for candle in reversed(histories[timeframe])
                    if candle.real_completed
                ),
                frame.cutoff,
            )
            try:
                self._record_frame_events(
                    frame,
                    first_semantic_snapshot or newly_completed_real,
                    event_clock=event_clock,
                )
            except Exception:
                self._terminal_failure = (
                    "event projection failed after state may have "
                    "changed; discard this observer and resume from "
                    "the last checkpoint"
                )
                raise
            self._last_frame_cutoff[timeframe] = frame.cutoff
        if (
            group3_update is not None
            and group3_update.boundary_reason
            not in FVG_BOUNDARY_REASONS
        ):
            try:
                self._record_group3_events(group3_update)
            except Exception:
                self._terminal_failure = (
                    "Group 3 event projection failed after state may "
                    "have changed; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        if self._group4_bootstrap_range_transitions:
            try:
                self._record_group4_events(
                    Group4Update(
                        dealing_ranges=(),
                        manipulations=(),
                        range_boundary_inventory=(),
                        range_transitions=tuple(
                            self._group4_bootstrap_range_transitions
                        ),
                    ),
                    include_resolutions=False,
                    include_creations=False,
                )
                self._group4_bootstrap_range_transitions.clear()
            except Exception:
                self._terminal_failure = (
                    "Group 4 bootstrap event projection failed after "
                    "state may have changed; discard this observer and "
                    "resume from the last checkpoint"
                )
                raise
        if group4_update is not None:
            try:
                self._record_group4_events(
                    group4_update,
                    include_resolutions=False,
                    include_creations=False,
                )
                inventory_by_id = {
                    item.item_id: item
                    for item in (
                        *liquidity_inventory,
                        *group4_update.range_boundary_inventory,
                    )
                }
                liquidity_inventory = tuple(
                    sorted(
                        inventory_by_id.values(),
                        key=lambda item: (
                            item.confirmed_at,
                            item.timeframe.value,
                            item.item_id,
                        ),
                    )
                )
            except Exception:
                self._terminal_failure = (
                    "Group 4 event projection failed after state may "
                    "have changed; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        pool_by_id = {
            pool.pool_id: pool
            for _, pools, _ in liquidity_snapshots.values()
            for pool in pools
        }
        pool_by_id.update(projected_pool_states)
        liquidity_pool_states = tuple(
            sorted(
                pool_by_id.values(),
                key=lambda item: (
                    item.confirmed_at,
                    item.timeframe.value,
                    item.pool_id,
                ),
            )
        )
        group5_update: Group5Update | None = None
        if self._group5_reducer is not None:
            try:
                if self._group5_boundary_update is not None:
                    group5_update = self._group5_boundary_update
                else:
                    group5_update = self._group5_reducer.on_completed_1m(
                        update.completed_1m,
                        fair_value_gaps=(
                            frames[Timeframe.M5].fair_value_gaps
                        ),
                        order_blocks=(
                            frames[Timeframe.M5].order_blocks
                        ),
                        manipulations=(
                            ()
                            if group4_update is None
                            else tuple(
                                state
                                for state
                                in group4_update.manipulations
                                if state.source_kind
                                == "formed_liquidity_pool"
                            )
                        ),
                        m1_bos=(
                            frames[Timeframe.M1].structure_breaks
                        ),
                        liquidity_inventory=liquidity_inventory,
                        m1_atr=(
                            frames[Timeframe.M1].metrics["atr"]
                        ),
                    )
            except Exception:
                self._terminal_failure = (
                    "Group 5 update failed after upstream reducers may "
                    "have advanced; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        if group5_update is not None:
            try:
                self._record_group5_events(group5_update)
            except Exception:
                self._terminal_failure = (
                    "Group 5 event projection failed after state may "
                    "have changed; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        try:
            self.memory.sync_retained_entity_timelines(
                self._retained_timeline_keys(
                    frames,
                    liquidity_pool_states,
                    (
                        ()
                        if group4_update is None
                        else group4_update.manipulations
                    ),
                    (
                        ()
                        if group5_update is None
                        else group5_update.path_sequences
                    ),
                ),
                asof=update.asof,
            )
        except Exception:
            self._terminal_failure = (
                "entity timeline projection failed after state may have "
                "changed; discard this observer and resume from the "
                "last checkpoint"
            )
            raise

        anomalies = list(update.anomalies)
        if (
            group3_update is not None
            and group3_update.boundary_reason
            in FVG_BOUNDARY_REASONS
        ):
            anomalies.append(
                {
                    "data_gap_reset": "data_gap_history_reset",
                    "contract_change_reset": (
                        "contract_change_history_reset"
                    ),
                    "data_anomaly": "data_anomaly",
                    "tick_size_mismatch": "tick_size_mismatch",
                }[group3_update.boundary_reason]
            )
        if (
            group4_update is not None
            and group4_update.ambiguous_sweep_item_ids
        ):
            anomalies.append("group4_ambiguous_dual_side_sweep")
        if (
            group4_update is not None
            and group4_update.atr_unready_sweep_item_ids
        ):
            anomalies.append("group4_atr_unready_sweep")
        for timeframe, frame in frames.items():
            if not frame.ready:
                anomalies.append(f"warmup_{timeframe.value}")
        anomalies.extend(execution.anomalies)
        if self.config.materialize_event_view:
            incomplete_timeline_keys = (
                self.memory.incomplete_entity_keys()
            )
            if incomplete_timeline_keys:
                anomalies.append("clock_incomplete_entity_timeline")
            event_durations_minutes, event_ages_minutes = (
                self.memory.temporal_metrics(update.asof)
            )
            recent_events = self.memory.recent()
            retained_entity_timelines = self.memory.entity_timelines()
        else:
            # Coverage scans retain and update the authoritative EventMemory
            # clock/lifecycles, but do not need to copy its complete query
            # view into every immutable observation.
            incomplete_timeline_keys = ()
            event_durations_minutes = {}
            event_ages_minutes = {}
            recent_events = ()
            retained_entity_timelines = {}
        try:
            observation = MarketObservation(
                asof=update.asof,
                symbol=update.completed_1m.symbol,
                instrument_id=update.completed_1m.instrument_id,
                price=float(update.completed_1m.close),
                frames=frames,
                recent_events=recent_events,
                event_durations_minutes=event_durations_minutes,
                execution=execution,
                anomalies=tuple(dict.fromkeys(anomalies)),
                displacement=displacement,
                liquidity_inventory=liquidity_inventory,
                liquidity_pool_states=liquidity_pool_states,
                event_ages_minutes=event_ages_minutes,
                retained_entity_timelines=retained_entity_timelines,
                incomplete_entity_timeline_keys=(
                    incomplete_timeline_keys
                ),
                group3_boundary_fvg_transitions=(
                    group3_update.fvg_transitions
                    if (
                        group3_update is not None
                        and group3_update.boundary_reason
                        in FVG_BOUNDARY_REASONS
                    )
                    else ()
                ),
                group3_boundary_order_block_transitions=(
                    group3_update.order_block_transitions
                    if (
                        group3_update is not None
                        and group3_update.boundary_reason
                        in FVG_BOUNDARY_REASONS
                    )
                    else ()
                ),
                manipulations=(
                    ()
                    if group4_update is None
                    else group4_update.manipulations
                ),
                group4_boundary_range_transitions=(
                    group4_update.range_transitions
                    if (
                        group4_update is not None
                        and group4_update.boundary_reason
                        in GROUP4_HARD_BOUNDARY_REASONS
                    )
                    else ()
                ),
                group4_boundary_manipulation_transitions=(
                    group4_update.manipulation_transitions
                    if (
                        group4_update is not None
                        and group4_update.boundary_reason
                        in GROUP4_HARD_BOUNDARY_REASONS
                    )
                    else ()
                ),
                group4_ambiguous_sweep_item_ids=(
                    ()
                    if group4_update is None
                    else group4_update.ambiguous_sweep_item_ids
                ),
                group4_atr_unready_sweep_item_ids=(
                    ()
                    if group4_update is None
                    else group4_update.atr_unready_sweep_item_ids
                ),
                group5_typed_available=(
                    self._group5_reducer is not None
                ),
                entry_locations=(
                    ()
                    if group5_update is None
                    else group5_update.entry_locations
                ),
                qualified_reacceptances=(
                    ()
                    if group5_update is None
                    else group5_update.qualified_reacceptances
                ),
                micro_bos_references=(
                    ()
                    if group5_update is None
                    else group5_update.micro_bos_references
                ),
                path_sequences=(
                    ()
                    if group5_update is None
                    else group5_update.path_sequences
                ),
                group5_boundary_path_transitions=(
                    ()
                    if (
                        group5_update is None
                        or group5_update.boundary_reason is None
                    )
                    else group5_update.path_transitions
                ),
                group5_boundary_reacceptance_transitions=(
                    ()
                    if (
                        group5_update is None
                        or group5_update.boundary_reason is None
                    )
                    else group5_update.reacceptance_transitions
                ),
                active_timeframes=self._active_timeframes,
                scale_registry_id=update.scale_registry_id,
            )
        except Exception:
            self._terminal_failure = (
                "market observation assembly failed after reducers "
                "advanced; discard this observer and resume from the "
                "last checkpoint"
            )
            raise
        if self.config.project_scene_graph:
            try:
                self.last_scene_delta = self.scene_graph.update(observation)
                observation = replace(
                    observation,
                    scene_revision_id=self.last_scene_delta.revision_id,
                    scene_added_node_ids=(
                        self.last_scene_delta.added_node_ids
                    ),
                    scene_revised_node_ids=(
                        self.last_scene_delta.revised_node_ids
                    ),
                    scene_added_edge_ids=(
                        self.last_scene_delta.added_edge_ids
                    ),
                    scene_revised_edge_ids=(
                        self.last_scene_delta.revised_edge_ids
                    ),
                    scene_resolution_event_ids=(
                        self.last_scene_delta.resolution_event_ids
                    ),
                )
            except Exception:
                self._terminal_failure = (
                    "scene-graph projection failed after semantic reducers "
                    "advanced; discard this observer and resume from the "
                    "last checkpoint"
                )
                raise
        else:
            self.last_scene_delta = None
        self._boundary_terminal_breaks.clear()
        self._boundary_reset_identity = None
        self._group4_boundary_update = None
        self._group5_boundary_update = None
        self._prior = observation
        return observation


__all__ = [
    "CausalObserver",
    "EventMemory",
    "ExecutionRealityInput",
    "ObserverConfig",
]
