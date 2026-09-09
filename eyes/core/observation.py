"""Descriptive multitimeframe observation over the registered semantics."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
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
from .event_memory import EventMemory
from .event_store import EventStore
from .zone import (
    CausalZoneTracker,
    FVG_BOUNDARY_REASONS,
    ZoneBOSSource,
    ZoneProtocol,
    ZoneUpdate,
)
from .range_auction import (
    CausalRangeAuctionTracker,
    RangeAuctionProtocol,
    RangeAuctionUpdate,
)
from .interaction import (
    InteractionProtocol,
    InteractionSemantics,
    InteractionUpdate,
)
from .liquidity import (
    CausalLiquidityTracker,
    LiquidityConfig,
    LiquidityProtocolError,
)
from contract.market import (
    CORE_TIMEFRAMES,
    Candle,
    Direction,
    Timeframe,
    clamp,
)
from contract.execution import (
    ExecutionObservation,
    execution_not_evaluated,
)
from contract.eye import (
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    CandleStructureState,
    DealingRangeState,
    EventKind,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    ManipulationLifecycle,
    ManipulationState,
    MarketObservation,
    PathSequenceState,
    RANGE_AUCTION_HARD_BOUNDARY_REASONS,
    StructureLifecycle,
    SwingLifecycle,
    SwingRelation,
)
from .market_state import (
    MarketSnapshot,
    MarketSnapshotPublisher,
    build_structural_legs,
)
from shares.core.scale_registry import ScaleSpec, scale_registry_id
from .semantic_event_emitter import SemanticEventEmitter
from .semantics import SemanticRegistry
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
    zone_protocol: str | None = None
    range_auction_protocol: str | None = None
    interaction_protocol: str | None = None
    semantic_registry: str = "semantics/registry_v1_3.yaml"
    scale_specs: tuple[ScaleSpec, ...] = ()
    # Scene Graph is an optional downstream research view, not part of the
    # causal Eye publication path. Full Engine composition opts in explicitly.
    # The Eye does not build a Scene Graph.  This flag declares that a
    # downstream consumer (the Engine) will project one from this
    # observation, which is what requires a materialized event view.
    project_scene_graph: bool = False
    materialize_event_view: bool = True
    range_auction_projection_only: bool = False
    eye_authority_mode: bool = False
    typed_transition_delta_transport: bool = False
    persist_state_projections: bool = True


# These fields advance mechanically while an entity remains in the same
# semantic state. They belong in the authoritative snapshot but must not
# create a transition-delta heartbeat every minute/bar.
_TYPED_DELTA_HEARTBEAT_FIELDS = frozenset(
    {
        "age_bars",
        "age_h1_bars",
        "age_1m_bars",
        "age_real_1m_bars",
        "state_duration_real_1m_bars",
        "last_updated_at",
        "freshness",
    }
)
_ENTRY_LOCATION_VIEW_FIELDS = frozenset(
    {
        "current_price",
        "distance_to_zone_points",
        "distance_to_failure_points",
        "nearest_visible_draw_distance_points",
    }
)


def _typed_semantic_signature(
    state: object,
    *,
    excluded_fields: frozenset[str] = frozenset(),
) -> tuple[tuple[str, object], ...]:
    """Stable state signature without descriptive clock heartbeats."""

    if not is_dataclass(state):
        raise TypeError("typed transition state must be a dataclass")
    ignored = _TYPED_DELTA_HEARTBEAT_FIELDS | excluded_fields
    return tuple(
        (item.name, getattr(state, item.name))
        for item in fields(state)
        if item.name not in ignored
    )


def _typed_state_delta_from_cache(
    *,
    candidates: Sequence[object],
    signatures: dict[str, tuple[tuple[str, object], ...]],
    identity_field: str,
    excluded_fields: frozenset[str] = frozenset(),
    retained_identities: frozenset[str] | None = None,
) -> tuple[object, ...]:
    """Compare candidate states with a compact observer-local signature map.

    This never rescans the prior immutable Observation. Candidates may contain
    multiple same-clock revisions of one entity; they are emitted in reducer
    order and update the cache in order.
    """

    output: list[object] = []
    for state in candidates:
        identity = str(getattr(state, identity_field))
        signature = _typed_semantic_signature(
            state,
            excluded_fields=excluded_fields,
        )
        if signatures.get(identity) == signature:
            continue
        output.append(state)
        signatures[identity] = signature
    if retained_identities is not None:
        for identity in tuple(signatures):
            if identity not in retained_identities:
                signatures.pop(identity, None)
    return tuple(output)


def _typed_native_transitions_or_baseline(
    *,
    current: Sequence[object],
    transitions: Sequence[object],
    first_observation: bool,
    boundary_reason: str | None,
) -> tuple[object, ...]:
    """Use one cold-start snapshot, but never hide boundary terminals."""

    if first_observation and boundary_reason is None:
        return tuple(current)
    return tuple(transitions)


@dataclass(frozen=True)
class _ReferencePeriod:
    """Incremental completed-period extrema; never uses an unfinished bar."""

    key: str
    started_at: pd.Timestamp
    last_end: pd.Timestamp
    high: float
    high_at: pd.Timestamp
    low: float
    low_at: pd.Timestamp
    symbol: str
    instrument_id: int
    coverage_complete: bool


@dataclass(frozen=True)
class _ReferenceCandidateSource:
    """Frozen clocks needed to publish one completed-period candidate."""

    extreme_at: pd.Timestamp
    admitted_at: pd.Timestamp
    period_started_at: pd.Timestamp
    period_last_end: pd.Timestamp
    replaces_level_id: str | None = None


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

    def __init__(
        self,
        config: ObserverConfig,
        *,
        semantic_registry: SemanticRegistry | None = None,
    ) -> None:
        self.config = config
        if semantic_registry is None:
            semantic_registry = SemanticRegistry.from_file(
                self.config.semantic_registry
            )
        elif not isinstance(semantic_registry, SemanticRegistry):
            raise TypeError("observer semantic_registry must be loaded")
        else:
            configured_path = Path(self.config.semantic_registry)
            if not configured_path.is_absolute() and not configured_path.exists():
                configured_path = (
                    Path(__file__).resolve().parents[2] / configured_path
                )
            if (
                configured_path.resolve()
                != semantic_registry.source_path.resolve()
            ):
                raise ValueError(
                    "observer semantic registry path differs from selection"
                )
        self.semantic_registry = semantic_registry
        self.audit_store = EventStore(
            semantic_version=self.semantic_registry.semantic_version,
            definition_identity=self.semantic_registry.definition_identity,
        )
        self.market_snapshot_publisher = MarketSnapshotPublisher(
            event_store=self.audit_store,
            semantic_registry_identity=self.semantic_registry.identity,
            # The normal Trading Eye is event-authoritative.  The explicitly
            # constrained Group-4 authority scanner does not publish atomic
            # BAR/semantic facts and therefore remains a labelled projection
            # compatibility mode instead of pretending to be atomic.
            atomic_authority=not self.config.range_auction_projection_only,
        )
        self.last_market_snapshot: MarketSnapshot | None = None
        if type(self.config.project_scene_graph) is not bool:
            raise ValueError("scene-graph projection flag must be boolean")
        if type(self.config.materialize_event_view) is not bool:
            raise ValueError("event-view materialization flag must be boolean")
        if type(self.config.range_auction_projection_only) is not bool:
            raise ValueError("Group 4 projection-only flag must be boolean")
        if type(self.config.eye_authority_mode) is not bool:
            raise ValueError("eye-authority mode flag must be boolean")
        if type(self.config.typed_transition_delta_transport) is not bool:
            raise ValueError(
                "typed transition delta transport flag must be boolean"
            )
        if type(self.config.persist_state_projections) is not bool:
            raise ValueError(
                "state-projection persistence flag must be boolean"
            )
        if self.config.eye_authority_mode:
            if (
                self.config.project_scene_graph
                and not self.config.materialize_event_view
            ):
                raise ValueError(
                    "Scene Graph projection requires a materialized "
                    "EventMemory view"
                )
            typed_protocols = (
                self.config.structure_protocol,
                self.config.liquidity_protocol,
                self.config.displacement_protocol,
                self.config.zone_protocol,
                self.config.range_auction_protocol,
                self.config.interaction_protocol,
            )
            if (
                self.config.range_auction_projection_only
                or any(protocol is None for protocol in typed_protocols)
            ):
                raise ValueError(
                    "eye-authority mode requires all typed protocols, "
                    "and Group 4 projection-only mode disabled"
                )
        if (
            not self.config.eye_authority_mode
            and not self.config.materialize_event_view
            and (
                self.config.project_scene_graph
                or self.config.range_auction_protocol is None
            )
        ):
            raise ValueError(
                "a lightweight event view requires a typed pipeline with "
                "Scene Graph disabled"
            )
        if self.config.range_auction_projection_only and (
            self.config.materialize_event_view
            or self.config.project_scene_graph
            or self.config.range_auction_protocol is None
            or self.config.displacement_protocol is not None
            or self.config.zone_protocol is not None
            or self.config.interaction_protocol is not None
        ):
            raise ValueError(
                "Group 4 projection-only mode is limited to the authority "
                "scanner"
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
        # Keep the exact registered detector contract available to the
        # semantic publisher.  In particular, a confirmed swing must source
        # the complete left/pivot/right bar window used by this config.
        self._structure_config = structure_config
        if (
            self.config.liquidity_protocol is not None
            and structure_config is None
        ):
            raise ValueError(
                "liquidity protocol requires the confirmed-swing tracker"
            )
        if (
            self.config.zone_protocol is not None
            and (
                displacement_protocol is None
                or structure_config is None
            )
        ):
            raise ValueError(
                "Group 3 requires typed displacement and structure/BOS"
            )
        zone_protocol = (
            ZoneProtocol.from_file(self.config.zone_protocol)
            if self.config.zone_protocol is not None
            else None
        )
        if zone_protocol is not None and (
            not math.isclose(
                zone_protocol.tick_size,
                self.config.tick_size,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or displacement_protocol is None
            or not math.isclose(
                zone_protocol.tick_size,
                displacement_protocol.tick_size,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(
                "Group 3, displacement and observer tick sizes disagree"
            )
        self._zone_tracker = (
            CausalZoneTracker(
                zone_protocol,
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
                zone_protocol is not None
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
            self.config.range_auction_protocol is not None
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
        range_auction_protocol = (
            RangeAuctionProtocol.from_file(self.config.range_auction_protocol)
            if self.config.range_auction_protocol is not None
            else None
        )
        if range_auction_protocol is not None and (
            not math.isclose(
                range_auction_protocol.tick_size,
                self.config.tick_size,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or liquidity_config is None
            or structure_config is None
            or range_auction_protocol.source_group12_protocol_hash
            != liquidity_config.protocol_hash
        ):
            raise ValueError(
                "Group 4 and Group 1-2 protocol bindings disagree"
            )
        self._range_auction_tracker = (
            CausalRangeAuctionTracker(range_auction_protocol)
            if range_auction_protocol is not None
            else None
        )
        self._group4_boundary_update: RangeAuctionUpdate | None = None
        self._group4_bootstrap_range_transitions: list[
            DealingRangeState
        ] = []
        self._group4_cold_pairs_marked = False
        if self.config.interaction_protocol is not None and (
            zone_protocol is None
            or range_auction_protocol is None
            or structure_config is None
            or liquidity_config is None
        ):
            raise ValueError(
                "interaction semantics require Groups 1-4 typed sources"
            )
        interaction_protocol = (
            InteractionProtocol.from_file(self.config.interaction_protocol)
            if self.config.interaction_protocol is not None
            else None
        )
        if interaction_protocol is not None and (
            not math.isclose(
                interaction_protocol.tick_size,
                self.config.tick_size,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or zone_protocol is None
            or range_auction_protocol is None
            or structure_config is None
            or liquidity_config is None
            or interaction_protocol.source_group12_protocol_hash
            != structure_config.protocol_hash
            or interaction_protocol.source_group12_protocol_hash
            != liquidity_config.protocol_hash
            or interaction_protocol.source_zone_protocol_hash
            != zone_protocol.protocol_hash
            or interaction_protocol.source_range_auction_protocol_hash
            != range_auction_protocol.protocol_hash
        ):
            raise ValueError(
                "interaction and Groups 1-4 protocol bindings disagree"
            )
        self._interaction_semantics = (
            InteractionSemantics(interaction_protocol)
            if interaction_protocol is not None
            else None
        )
        self._interaction_boundary_update: InteractionUpdate | None = None
        self.memory = EventMemory(
            self.config.memory_events,
            audit_store=self.audit_store,
        )
        self._terminal_failure: str | None = None
        self._last_displacement_input: tuple[object, ...] | None = None
        self._last_displacement_observation = None
        self._prior: MarketObservation | None = None
        # One owner for canonical emission and the cross-detector event
        # ancestry index every emitted fact has to cite.
        self._emitter = SemanticEventEmitter(
            config=self.config,
            semantic_registry=self.semantic_registry,
            audit_store=self.audit_store,
            memory=self.memory,
            active_timeframes=self._active_timeframes,
            scale_registry_id=self._scale_registry_id,
            structure_config=self._structure_config,
        )
        self._structural_leg_cache: dict[
            Timeframe,
            tuple[tuple[str, ...], tuple],
        ] = {}
        self._last_frame_cutoff: dict[Timeframe, pd.Timestamp] = {}
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
        self._pending_level_crossings: dict[
            str,
            tuple[pd.Timestamp, LiquidityInventoryItem],
        ] = {}
        self._reference_periods: dict[str, _ReferencePeriod] = {}
        self._reference_inventory: dict[str, LiquidityInventoryItem] = {}
        self._reference_candidate_sources: dict[
            str,
            _ReferenceCandidateSource,
        ] = {}
        self._reference_last_end: pd.Timestamp | None = None
        self._reference_coverage_start: pd.Timestamp | None = None
        # Optional transport cache for authority scans and bounded diagnostics.
        # Production Engine observers do not populate it unless explicitly
        # requested, so ordinary replay pays no typed-delta comparison cost.
        self._typed_delta_signatures: dict[
            str,
            dict[str, tuple[tuple[str, object], ...]],
        ] = {}

    @property
    def interaction_protocol(self) -> InteractionProtocol | None:
        return (
            None
            if self._interaction_semantics is None
            else self._interaction_semantics.protocol
        )


    def mark_terminal_failure(self, reason: str) -> None:
        """Refuse further observation after a post-reduction consumer failed.

        The reducers have already advanced when a downstream projection raises,
        so the observer cannot be reused.  Callers that advance state after
        ``observe()`` returns report the failure here instead of leaving a
        half-advanced Eye alive.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("terminal failure requires an explicit reason")
        if self._terminal_failure is None:
            self._terminal_failure = reason

    def _reset_contract_state(
        self,
        *,
        reason: str,
        observed_at: pd.Timestamp,
        reset_anomalies: Sequence[str],
        boundary_symbol: str,
        boundary_instrument_id: int,
    ) -> None:
        self.market_snapshot_publisher.on_boundary()
        self.last_market_snapshot = None
        # Delta transport is an authority-scan projection, not causal state.
        # A hard epoch boundary must not retain prior-contract signatures.
        self._typed_delta_signatures.clear()
        self._interaction_boundary_update = (
            self._interaction_semantics.on_boundary(
                reason,
                observed_at,
                symbol=boundary_symbol,
                instrument_id=boundary_instrument_id,
            )
            if self._interaction_semantics is not None
            else None
        )
        self._group4_boundary_update = (
            self._range_auction_tracker.on_boundary(reason, observed_at)
            if self._range_auction_tracker is not None
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
        self.memory = EventMemory(
            self.config.memory_events,
            audit_store=self.audit_store,
        )
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
        self.memory.retain_live_prefixes_from(
            prior_memory,
            asof=observed_at,
        )
        self.memory.transfer_pending_from(prior_memory)
        self._emitter.rebind_memory(self.memory)
        self._emitter._append_semantic_atomic(
            EventKind.MARKET_EPOCH_RESET,
            observed_at,
            Timeframe.M1,
            None,
            None,
            0.0,
            (),
            {
                "reason": reason,
                "reset_anomalies": tuple(reset_anomalies),
                "new_symbol": boundary_symbol,
                "new_instrument_id": boundary_instrument_id,
            },
            event_time=observed_at,
        )
        self._prior = None
        self._structural_leg_cache.clear()
        self._last_frame_cutoff.clear()
        self._emitter.reset_contract_state()
        self._mss_displacement_by_bos.clear()
        for tracker in self._liquidity_trackers.values():
            tracker.reset()
        self._liquidity_snapshot_cache.clear()
        self._inventory_consumption.clear()
        self._pending_pool_sweeps.clear()
        self._pending_level_crossings.clear()
        self._reference_periods.clear()
        self._reference_inventory.clear()
        self._reference_candidate_sources.clear()
        self._reference_last_end = None
        self._reference_coverage_start = None
        self._emitter.emit_boundary_structure_break_failed(
            failed_bos,
            observed_at,
            reset_anomalies,
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
                            and not self.config.range_auction_projection_only
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
                        and self._range_auction_tracker is not None
                        and self._prior is None
                    ):
                        support_resistance, _, _ = (
                            liquidity_tracker.snapshot(
                                range_auction_sources_only=True,
                                include_support_resistance=True,
                            )
                            if self.config.range_auction_projection_only
                            else liquidity_tracker.snapshot()
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
                            self._range_auction_tracker.mark_existing_source_pairs_ineligible(
                                tuple(
                                    zone
                                    for zone in support_resistance
                                    if zone.confirmed_at
                                    < coverage_start
                                )
                            )
                            self._group4_cold_pairs_marked = True
                        range_auction_update = (
                            self._range_auction_tracker.on_completed_h1(
                                candle,
                                (
                                    support_resistance
                                    if within_group4_coverage
                                    else ()
                                ),
                            )
                        )
                        self._group4_bootstrap_range_transitions.extend(
                            range_auction_update.range_transitions
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
    ) -> ZoneUpdate | None:
        if self._zone_tracker is None:
            return None
        if self._displacement_eye is None:
            raise RuntimeError(
                "Group 3 lost its configured displacement source"
            )
        boundary = self._group3_boundary_reason(update.anomalies)
        if boundary is not None:
            projected = self._visible_zone_update(
                self._zone_tracker.on_boundary(
                    boundary,
                    update.asof,
                )
            )
            return projected
        batch = self._displacement_eye.last_batch
        expected = tuple(
            update.newly_completed.get(Timeframe.M5, ())
        )
        if tuple(candle for candle, _ in batch) != expected:
            raise RuntimeError(
                "Group 3 and displacement completed-M5 batches diverged"
            )
        result = self._zone_tracker.current_update()
        order_block_funnel = []
        fvg_transitions = []
        order_block_transitions = []
        bos_states = frames[Timeframe.M5].structure_breaks
        for candle, displacement_update in batch:
            result = self._zone_tracker.on_completed_5m(
                candle,
                displacement_update,
                tuple(
                    ZoneBOSSource(
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
            fvg_transitions.extend(result.fvg_transitions)
            order_block_transitions.extend(
                result.order_block_transitions
            )
            order_block_funnel.extend(result.order_block_funnel)
        if (
            fvg_transitions
            or order_block_transitions
            or order_block_funnel
        ):
            result = replace(
                result,
                fvg_transitions=tuple(fvg_transitions),
                order_block_transitions=tuple(
                    order_block_transitions
                ),
                order_block_funnel=tuple(order_block_funnel),
            )
        return self._visible_zone_update(result)

    def _visible_zone_update(
        self,
        update: ZoneUpdate,
    ) -> ZoneUpdate:
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

    def _retained_timeline_keys(
        self,
        frames: Mapping[Timeframe, FrameObservation],
        liquidity_pool_states: Sequence[LiquidityPoolState],
        manipulations: Sequence[ManipulationState] = (),
        path_sequences: Sequence[PathSequenceState] = (),
        *,
        terminal_entity_keys: Iterable[str] = (),
    ) -> set[str]:
        terminal = set(terminal_entity_keys)
        for key in terminal:
            self.memory._timeline_namespace(key)
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
                # A pending BOS has no lifecycle history yet: its timeline
                # begins at the terminal that resolves it.
                f"bos:{item.bos_id}"
                for item in frame.structure_breaks
                if item.lifecycle is not BOSLifecycle.PENDING
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
        # A public reducer snapshot is intentionally bounded and can omit an
        # entity before a later lifecycle transition is emitted.  Keep only
        # still-transitionable prefixes hot; terminal histories cool as soon
        # as the current typed snapshot no longer exposes them.
        keys.update(set(self.memory.live_entity_keys()) - terminal)
        # A reducer boundary may expose its terminal transition through the
        # dedicated boundary channel without appending that transition to the
        # hot EventMemory timeline.  Such an entity must cool immediately;
        # retaining its pre-boundary OPEN/CREATED prefix would make a
        # terminal FVG/OB look live after a data anomaly.
        keys.difference_update(terminal)
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
        snapshot = (
            tracker.snapshot(
                range_auction_sources_only=True,
                include_support_resistance=(timeframe is Timeframe.H1),
            )
            if self.config.range_auction_projection_only
            else tracker.snapshot()
        )
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
        *,
        admitted_at: pd.Timestamp,
        replaces_by_kind: Mapping[str, str],
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
        for side, suffix, price, extreme_at in (
            ("above", "high", period.high, period.high_at),
            ("below", "low", period.low, period.low_at),
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
            self._reference_candidate_sources[item_id] = (
                _ReferenceCandidateSource(
                    extreme_at=extreme_at,
                    admitted_at=admitted_at,
                    period_started_at=period.started_at,
                    period_last_end=period.last_end,
                    replaces_level_id=replaces_by_kind.get(kind),
                )
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
                    high_at=candle.end,
                    low=float(candle.low),
                    low_at=candle.end,
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
                new_high = float(candle.high) > current.high
                new_low = float(candle.low) < current.low
                self._reference_periods[family] = replace(
                    current,
                    last_end=candle.end,
                    high=max(current.high, float(candle.high)),
                    high_at=(candle.end if new_high else current.high_at),
                    low=min(current.low, float(candle.low)),
                    low_at=(candle.end if new_low else current.low_at),
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
            retired_ids = {item.item_id for item in retired}
            self._reference_candidate_sources = {
                item_id: source
                for item_id, source in self._reference_candidate_sources.items()
                if item_id not in retired_ids
            }
            if append_retirement_events:
                self._emitter.emit_reference_period_retirements(
                    retired,
                    observed_at=candle.end,
                    replacement_period=current.key,
                )
            if current.coverage_complete:
                self._materialize_reference_period(
                    family,
                    current,
                    admitted_at=candle.end,
                    replaces_by_kind={item.kind: item.item_id for item in retired},
                )
            period_start = self._reference_period_start(
                family,
                candle,
            )
            self._reference_periods[family] = _ReferencePeriod(
                key=key,
                started_at=period_start,
                last_end=candle.end,
                high=float(candle.high),
                high_at=candle.end,
                low=float(candle.low),
                low_at=candle.end,
                symbol=candle.symbol,
                instrument_id=int(candle.instrument_id),
                coverage_complete=True,
            )

    def _publish_reference_candidate_events(self) -> None:
        """Publish visible completed-period extrema at their admission clock.

        Reference inventory is built before normalized BAR facts are appended.
        This second pass runs immediately after those roots exist, so a level is
        authoritative before any touch while retaining the exact extreme bar
        and the first completed M1 bar that made the prior period knowable.
        """

        if self.config.range_auction_projection_only:
            return
        visible_ids = set(self._reference_inventory)
        self._reference_candidate_sources = {
            item_id: source
            for item_id, source in self._reference_candidate_sources.items()
            if item_id in visible_ids
        }
        missing_sources = visible_ids - set(self._reference_candidate_sources)
        if missing_sources:
            raise ValueError(
                "reference candidates lack frozen source clocks: "
                + ", ".join(sorted(missing_sources))
            )
        ordered = sorted(
            self._reference_inventory.values(),
            key=lambda item: (
                self._reference_candidate_sources[item.item_id].admitted_at,
                item.kind,
                item.item_id,
            ),
        )
        for item in ordered:
            if item.item_id in self._emitter._candidate_level_event_ids:
                continue
            source = self._reference_candidate_sources.get(item.item_id)
            if source is None:
                raise ValueError(
                    "reference candidate lacks its frozen source clocks"
                )
            extreme_bar_event_id = self._emitter._bar_event_id_at(
                Timeframe.M1,
                source.extreme_at,
            )
            admission_bar_event_id = self._emitter._bar_event_id_at(
                Timeframe.M1,
                source.admitted_at,
            )
            evidence: dict[str, object] = {
                "level_id": item.item_id,
                "candidate_only": True,
                "source_kind": item.kind,
                "source_inventory_kind": item.kind,
                "source_ids": item.source_ids,
                "source_formed_at": item.formed_at.isoformat(),
                "source_confirmed_at": source.period_last_end.isoformat(),
                "reference_period_started_at": (
                    source.period_started_at.isoformat()
                ),
                "reference_period_last_completed_at": (
                    source.period_last_end.isoformat()
                ),
                "reference_extreme_at": source.extreme_at.isoformat(),
                "reference_admitted_at": source.admitted_at.isoformat(),
                "reference_extreme_tie_rule": (
                    "first_completed_m1_at_extreme"
                ),
                "rank": item.structural_rank,
                "strength": float(item.strength),
            }
            replacement_event_id = (
                None
                if source.replaces_level_id is None
                else self._emitter._candidate_level_event_ids.get(
                    source.replaces_level_id
                )
            )
            if replacement_event_id is not None:
                evidence["replaces_level_id"] = source.replaces_level_id
                evidence["replaces_level_event_id"] = replacement_event_id
            candidate = self._emitter._append_semantic_atomic(
                EventKind.LIQUIDITY_LEVEL_CREATED,
                source.admitted_at,
                Timeframe.M1,
                item.side,
                item.price,
                item.strength,
                (extreme_bar_event_id, admission_bar_event_id),
                evidence,
                event_time=source.extreme_at,
                zone=(item.lower_bound, item.upper_bound),
                source_entity_ids=(item.item_id, *item.source_ids),
                context_event_ids=(
                    ()
                    if replacement_event_id is None
                    else (replacement_event_id,)
                ),
            )
            self._emitter._candidate_level_event_ids[item.item_id] = candidate.event_id





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
                self._emitter._append_inventory_crossing_event(
                    item,
                    sweep_candle,
                    atr=_atr(
                        real_history[: sweep_index + 1],
                        self.config.atr_period,
                    ),
                )
                if self._emitter._pool_close_outside(item, sweep_candle):
                    resolution_candle = next(
                        (
                            candle
                            for index, candle in eligible
                            if index > sweep_index
                        ),
                        None,
                    )
                    if resolution_candle is None:
                        self._pending_level_crossings[item.item_id] = (
                            sweep_candle.end,
                            item,
                        )
                    else:
                        self._emitter._append_level_resolution_event(
                            item,
                            resolution_candle,
                            crossed_at=sweep_candle.end,
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
            outside_on_sweep = self._emitter._pool_close_outside(
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
                    else self._emitter._pool_close_outside(
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
            self._emitter._append_projected_pool_sweep_events(
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
                self._emitter._append_projected_pool_resolution_event(
                    item,
                    resolution_candle,
                    crossed_at=sweep_candle.end,
                    range_auction_tracker=self._range_auction_tracker,
                )

    def _resolve_pending_pool_sweeps(
        self,
        update: ReaderUpdate,
        *,
        append_events: bool = True,
        projected_pool_states: dict[str, LiquidityPoolState] | None = None,
    ) -> tuple[tuple[LiquidityInventoryItem, Candle, pd.Timestamp], ...]:
        """Resolve pending sweeps before any same-clock HTF state update."""

        if type(append_events) is not bool:
            raise TypeError("pool-resolution event flag must be boolean")
        bar = update.completed_1m
        if not bar.real_completed:
            return ()
        resolved: list[
            tuple[LiquidityInventoryItem, Candle, pd.Timestamp]
        ] = []
        for item_id, (swept_at, item) in tuple(
            self._pending_pool_sweeps.items()
        ):
            if update.asof <= swept_at:
                continue
            outside = self._emitter._pool_close_outside(item, bar)
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
                self._emitter._append_projected_pool_resolution_event(
                    item,
                    bar,
                    crossed_at=swept_at,
                    range_auction_tracker=self._range_auction_tracker,
                )
            else:
                resolved.append((item, bar, swept_at))
            self._pending_pool_sweeps.pop(item_id, None)
        return tuple(resolved)


    def _resolve_pending_level_crossings(
        self,
        update: ReaderUpdate,
        *,
        append_events: bool = True,
    ) -> tuple[tuple[LiquidityInventoryItem, Candle, pd.Timestamp], ...]:
        """Resolve generic candidate penetrations without a pool lifecycle."""

        if type(append_events) is not bool:
            raise TypeError("level-resolution event flag must be boolean")
        bar = update.completed_1m
        if not bar.real_completed:
            return ()
        resolved: list[
            tuple[LiquidityInventoryItem, Candle, pd.Timestamp]
        ] = []
        for item_id, (crossed_at, item) in tuple(
            self._pending_level_crossings.items()
        ):
            if update.asof <= crossed_at:
                continue
            if append_events:
                self._emitter._append_level_resolution_event(
                    item,
                    bar,
                    crossed_at=crossed_at,
                )
            else:
                resolved.append((item, bar, crossed_at))
            self._pending_level_crossings.pop(item_id, None)
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
                accepted_outside = self._emitter._pool_close_outside(
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
                self._emitter._append_projected_pool_sweep_events(
                    item,
                    bar,
                    atr=atr,
                )
                self._pending_pool_sweeps[item.item_id] = (
                    update.asof,
                    item,
                )
            else:
                self._emitter._append_inventory_crossing_event(
                    item,
                    bar,
                    atr=atr,
                )
                if self._emitter._pool_close_outside(item, bar):
                    self._pending_level_crossings[item.item_id] = (
                        update.asof,
                        item,
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
        execution: ExecutionObservation | None = None,
    ) -> MarketObservation:
        """Reduce one reader update into deterministic market facts.

        ``execution`` is transported, never derived: scoring broker/feed
        reality is the execution layer's job, and the Eye only carries the
        result so a downstream consumer reads one observation.
        """
        if self._terminal_failure is not None:
            raise RuntimeError(self._terminal_failure)
        audit_start = len(self.audit_store)
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
        if self.config.eye_authority_mode and execution is not None:
            raise ValueError(
                "eye-authority mode does not evaluate execution reality"
            )
        if execution is None:
            execution = execution_not_evaluated()
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
                    if (
                        displacement is not None
                        and not self.config.range_auction_projection_only
                    ):
                        # Boundary transitions close the prior displacement
                        # epoch and therefore still reference its normalized
                        # bars.  Publish those immutable terminal facts before
                        # clearing the prior-epoch BAR lookup tables.
                        self._emitter._record_displacement_events(displacement)
                        displacement = None
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
            deferred_level_resolution_events = (
                self._resolve_pending_level_crossings(
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
            protected_swing_ids = tuple(
                item.protected_swing_id
                for item in structures
                if item.protected_swing_id is not None
                and item.lifecycle is StructureLifecycle.CONFIRMED
            )
            structural_swing_ids = tuple(
                item.target_swing_id
                for item in visible_breaks
                if item.lifecycle is BOSLifecycle.CONFIRMED
            )
            resolved_swing_ids = tuple(
                item.swing_id
                for item in swings
                if (
                    item.lifecycle
                    in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
                    and item.confirmed_at is not None
                )
            )
            cached_legs = self._structural_leg_cache.get(timeframe)
            if (
                cached_legs is not None
                and cached_legs[0] == resolved_swing_ids
            ):
                structural_legs = cached_legs[1]
            else:
                projected_legs = build_structural_legs(
                    timeframe,
                    swings,
                    histories[timeframe],
                    atr=atr,
                    protected_swing_ids=protected_swing_ids,
                    structural_swing_ids=structural_swing_ids,
                )
                frozen_by_id = {
                    item.leg_id: item
                    for item in (
                        () if cached_legs is None else cached_legs[1]
                    )
                }
                structural_legs = tuple(
                    frozen_by_id.get(item.leg_id, item)
                    for item in projected_legs
                )
                self._structural_leg_cache[timeframe] = (
                    resolved_swing_ids,
                    structural_legs,
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
                structural_legs=structural_legs,
            )
        if Timeframe.M5 in frames:
            frames[Timeframe.M5] = replace(
                frames[Timeframe.M5],
                structure_breaks=self._enrich_mss_breaks(
                    frames[Timeframe.M5].structure_breaks
                ),
            )
        try:
            zone_update = self._observe_group3(update, frames)
            if zone_update is not None:
                frames[Timeframe.M5] = replace(
                    frames[Timeframe.M5],
                    fair_value_gaps=zone_update.fair_value_gaps,
                    order_blocks=zone_update.order_blocks,
                )
        except Exception:
            self._terminal_failure = (
                "Group 3 update failed after a paired reducer may have "
                "advanced; discard this observer and resume from the "
                "last checkpoint"
            )
            raise
        if not self.config.range_auction_projection_only:
            try:
                # Normalized data facts are the roots of the semantic DAG.
                # Publish them before Displacement, Group 3, structure, or
                # liquidity events attempt to reference the consumed bars.
                self._emitter._append_available_bar_events(
                    update,
                    histories,
                    frames,
                )
                self._publish_reference_candidate_events()
                self._emitter._record_displacement_events(displacement)
            except Exception:
                self._terminal_failure = (
                    "normalized bar/reference/displacement semantic projection "
                    "failed "
                    "after state may have changed; discard this observer "
                    "and resume from the last checkpoint"
                )
                raise
        # Publish every visible Group 1-2 candidate before any completed-M1
        # crossing can consume it. LIQUIDITY_LEVEL_CREATED is the sole
        # admission authority; a crossing must never manufacture its parent.
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
            if not self.config.range_auction_projection_only:
                try:
                    self._emitter._record_frame_events(
                        frame,
                        first_semantic_snapshot or newly_completed_real,
                        event_clock=event_clock,
                        prior=self._prior,
                    )
                except Exception:
                    self._terminal_failure = (
                        "event projection failed after state may have "
                        "changed; discard this observer and resume from "
                        "the last checkpoint"
                    )
                    raise
            self._last_frame_cutoff[timeframe] = frame.cutoff
        for timeframe, tracker in self._liquidity_trackers.items():
            if timeframe not in liquidity_snapshots:
                liquidity_snapshots[timeframe] = (
                    self._liquidity_snapshot(timeframe, tracker)
                )
            _, _, native_inventory = liquidity_snapshots[timeframe]
            base_inventory.extend(native_inventory)
        if not self.config.range_auction_projection_only:
            base_inventory.extend(self._reference_inventory.values())
        if not self.config.range_auction_projection_only:
            retained_liquidity_entities = {
                item.zone_id
                for frame in frames.values()
                for item in frame.support_resistance
            } | {
                f"pool:{item.pool_id}"
                for frame in frames.values()
                for item in frame.liquidity_pools
            }
            self._emitter._liquidity_entity_revisions = {
                entity_id: revision
                for entity_id, revision
                in self._emitter._liquidity_entity_revisions.items()
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
        range_auction_update: RangeAuctionUpdate | None = None
        if self._range_auction_tracker is not None:
            try:
                if self._group4_boundary_update is not None:
                    range_auction_update = self._group4_boundary_update
                elif self._prior is None:
                    range_auction_update = (
                        self._range_auction_tracker
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
                    range_auction_update = (
                        self._range_auction_tracker.on_completed_update(
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
                    dealing_ranges=range_auction_update.dealing_ranges,
                )
                # Append each manipulation timeline in lifecycle order now.
                # New SWEPT events carry a high same-clock sequence floor,
                # so inventory, HTF sources and ranges still sort before
                # creation; terminal resolutions keep the earliest sequence.
                if not self.config.range_auction_projection_only:
                    self._emitter._record_group4_events(
                        range_auction_update,
                        include_ranges=False,
                        include_resolutions=True,
                        include_creations=True,
                        prior=self._prior,
                    )
            except Exception:
                self._terminal_failure = (
                    "Group 4 update failed after a paired reducer may "
                    "have advanced; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        try:
            if not self.config.range_auction_projection_only:
                for item, candle, crossed_at in deferred_pool_resolution_events:
                    self._emitter._append_projected_pool_resolution_event(
                        item,
                        candle,
                        crossed_at=crossed_at,
                        range_auction_tracker=self._range_auction_tracker,
                    )
                for (
                    item,
                    candle,
                    crossed_at,
                ) in deferred_level_resolution_events:
                    self._emitter._append_level_resolution_event(
                        item,
                        candle,
                        crossed_at=crossed_at,
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
        if (
            zone_update is not None
            and zone_update.boundary_reason
            not in FVG_BOUNDARY_REASONS
        ):
            try:
                self._emitter._record_group3_events(zone_update)
            except Exception:
                self._terminal_failure = (
                    "Group 3 event projection failed after state may "
                    "have changed; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        if (
            self._group4_bootstrap_range_transitions
            and not self.config.range_auction_projection_only
        ):
            try:
                self._emitter._record_group4_events(
                    RangeAuctionUpdate(
                        dealing_ranges=(),
                        manipulations=(),
                        range_boundary_inventory=(),
                        range_transitions=tuple(
                            self._group4_bootstrap_range_transitions
                        ),
                    ),
                    include_resolutions=False,
                    include_creations=False,
                    prior=self._prior,
                )
                self._group4_bootstrap_range_transitions.clear()
            except Exception:
                self._terminal_failure = (
                    "Group 4 bootstrap event projection failed after "
                    "state may have changed; discard this observer and "
                    "resume from the last checkpoint"
                )
                raise
        elif self.config.range_auction_projection_only:
            self._group4_bootstrap_range_transitions.clear()
        if range_auction_update is not None:
            try:
                if not self.config.range_auction_projection_only:
                    self._emitter._record_group4_events(
                        range_auction_update,
                        include_resolutions=False,
                        include_creations=False,
                        prior=self._prior,
                    )
                inventory_by_id = {
                    item.item_id: item
                    for item in (
                        *liquidity_inventory,
                        *range_auction_update.range_boundary_inventory,
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
        interaction_update: InteractionUpdate | None = None
        if self._interaction_semantics is not None:
            try:
                if self._interaction_boundary_update is not None:
                    interaction_update = self._interaction_boundary_update
                else:
                    interaction_update = (
                        self._interaction_semantics.on_completed_1m(
                            update.completed_1m,
                            fair_value_gaps=(
                                frames[Timeframe.M5].fair_value_gaps
                            ),
                            order_blocks=(
                                frames[Timeframe.M5].order_blocks
                            ),
                            manipulations=(
                                ()
                                if range_auction_update is None
                                else tuple(
                                    state
                                    for state
                                    in range_auction_update.manipulations
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
                    )
            except Exception:
                self._terminal_failure = (
                    "interaction update failed after upstream reducers may "
                    "have advanced; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        if interaction_update is not None:
            try:
                self._emitter._record_interaction_events(interaction_update)
            except Exception:
                self._terminal_failure = (
                    "interaction event projection failed after state may "
                    "have changed; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        if not self.config.range_auction_projection_only:
            try:
                group4_boundary_range_keys = {
                    f"range:{state.range_id}"
                    for state in (
                        ()
                        if (
                            range_auction_update is None
                            or range_auction_update.boundary_reason
                            not in RANGE_AUCTION_HARD_BOUNDARY_REASONS
                        )
                        else range_auction_update.range_transitions
                    )
                }
                self.memory.sync_retained_entity_timelines(
                    self._retained_timeline_keys(
                        frames,
                        liquidity_pool_states,
                        (
                            ()
                            if range_auction_update is None
                            else range_auction_update.manipulations
                        ),
                        (
                            ()
                            if interaction_update is None
                            else interaction_update.interaction_paths
                        ),
                        terminal_entity_keys=(
                            ()
                            if (
                                zone_update is None
                                or zone_update.boundary_reason
                                not in FVG_BOUNDARY_REASONS
                            )
                            else (
                                *(
                                    f"fvg:{state.fvg_id}"
                                    for state in zone_update.fvg_transitions
                                ),
                                *(
                                    "order_block:"
                                    f"{state.order_block_id}"
                                    for state in (
                                        zone_update.order_block_transitions
                                    )
                                ),
                            )
                        ),
                    )
                    | group4_boundary_range_keys,
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
            zone_update is not None
            and zone_update.boundary_reason
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
                    "semantic_reset": "semantic_reset",
                }[zone_update.boundary_reason]
            )
        if (
            range_auction_update is not None
            and range_auction_update.ambiguous_sweep_item_ids
        ):
            anomalies.append("group4_ambiguous_dual_side_sweep")
        if (
            range_auction_update is not None
            and range_auction_update.atr_unready_sweep_item_ids
        ):
            anomalies.append("group4_atr_unready_sweep")
        for timeframe, frame in frames.items():
            if not frame.ready:
                anomalies.append(f"warmup_{timeframe.value}")
        market_anomalies = tuple(dict.fromkeys(anomalies))
        anomalies.extend(execution.anomalies)
        try:
            self.memory.flush_audit()
            (
                market_snapshot,
                projection_events,
                delivery_transitions,
            ) = (
                self.market_snapshot_publisher.publish(
                    asof=update.asof,
                    symbol=update.completed_1m.symbol,
                    instrument_id=update.completed_1m.instrument_id,
                    price=float(update.completed_1m.close),
                    completed_1m=update.completed_1m,
                    frames=frames,
                    inventory=liquidity_inventory,
                    displacement=displacement,
                    anomalies=market_anomalies,
                    emit_projection_events=(
                        self.config.persist_state_projections
                    ),
                )
            )
            if self.config.persist_state_projections:
                for event in projection_events:
                    self.memory.append(
                        event,
                        include_in_recent=False,
                        sequence_floor=(
                            EventMemory._GROUP4_CREATION_SEQUENCE_FLOOR
                            + 1_000_000
                        ),
                    )
            if self.config.persist_state_projections:
                self.memory.flush_audit()
            self.market_snapshot_publisher._consume_committed_projection_tail()
            # The delivery-phase lifecycle is an atomic fact rather than
            # technical projection transport, so it is published whether or
            # not state projections are persisted, and it is appended after
            # the projection tail so the reducer never sees a physical fact in
            # that tail.  The next completed bar reduces it in stream order.
            self._emitter._record_delivery_phase_events(delivery_transitions)
            self.memory.flush_audit()
            if delivery_transitions:
                self.market_snapshot_publisher._consume_committed_delivery_phase_tail()
            semantic_events = self.audit_store.events_since(audit_start)
            market_snapshot = replace(
                market_snapshot,
                events_this_update=semantic_events,
                event_count=self.market_snapshot_publisher._event_reducer.cursor,
                event_prefix_fingerprint=self.audit_store.fingerprint(),
            )
            self.last_market_snapshot = market_snapshot
        except Exception:
            self._terminal_failure = (
                "hierarchical state publication or audit commit failed after "
                "reducers advanced; discard this observer and resume from "
                "the last checkpoint"
            )
            raise
        incomplete_timeline_keys = (
            self.memory.incomplete_entity_keys()
            if (
                self.config.materialize_event_view
                or self.config.eye_authority_mode
            )
            else ()
        )
        if incomplete_timeline_keys:
            anomalies.append("clock_incomplete_entity_timeline")
        if self.config.materialize_event_view:
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

        prior_observation = self._prior
        current_fvgs = frames[Timeframe.M5].fair_value_gaps
        current_order_blocks = frames[Timeframe.M5].order_blocks
        current_ranges = frames[Timeframe.H1].dealing_ranges
        current_manipulations = (
            ()
            if range_auction_update is None
            else range_auction_update.manipulations
        )
        typed_delta_available = bool(
            self.config.eye_authority_mode
            or self.config.typed_transition_delta_transport
        )
        liquidity_inventory_delta: tuple[object, ...] = ()
        liquidity_pool_delta: tuple[object, ...] = ()
        group3_fvg_delta: tuple[object, ...] = ()
        group3_order_block_delta: tuple[object, ...] = ()
        group4_range_delta: tuple[object, ...] = ()
        group4_manipulation_delta: tuple[object, ...] = ()
        if typed_delta_available:
            baseline = prior_observation is None
            group4_boundary = bool(
                range_auction_update is not None
                and range_auction_update.boundary_reason is not None
            )

            def cached(
                name: str,
                candidates: Sequence[object],
                identity_field: str,
                *,
                excluded_fields: frozenset[str] = frozenset(),
                retained_states: Sequence[object] | None = None,
            ) -> tuple[object, ...]:
                return _typed_state_delta_from_cache(
                    candidates=candidates,
                    signatures=self._typed_delta_signatures.setdefault(
                        name,
                        {},
                    ),
                    identity_field=identity_field,
                    excluded_fields=excluded_fields,
                    retained_identities=(
                        None
                        if retained_states is None
                        else frozenset(
                            str(getattr(state, identity_field))
                            for state in retained_states
                        )
                    ),
                )

            # Inventory and pool reducers do not yet expose a complete
            # ordinary transition batch.  Compare only the current view with
            # an observer-local signature cache; never rescan the prior
            # immutable Observation.
            liquidity_inventory_delta = cached(
                "liquidity_inventory",
                liquidity_inventory,
                "item_id",
                retained_states=liquidity_inventory,
            )
            liquidity_pool_delta = cached(
                "liquidity_pool",
                liquidity_pool_states,
                "pool_id",
                retained_states=liquidity_pool_states,
            )

            # Groups 3 and 4 expose authoritative reducer transitions.  The
            # full state is read once as a warmup baseline; subsequent bars
            # consume only the native transition batch.  A live swept
            # manipulation is the sole snapshot fallback because its
            # reclaim/outside counters are meaningful ordinary revisions.
            group3_fvg_delta = _typed_native_transitions_or_baseline(
                current=current_fvgs,
                transitions=(
                    ()
                    if zone_update is None
                    else zone_update.fvg_transitions
                ),
                first_observation=baseline,
                boundary_reason=(
                    None
                    if zone_update is None
                    else zone_update.boundary_reason
                ),
            )
            group3_order_block_delta = _typed_native_transitions_or_baseline(
                current=current_order_blocks,
                transitions=(
                    ()
                    if zone_update is None
                    else zone_update.order_block_transitions
                ),
                first_observation=baseline,
                boundary_reason=(
                    None
                    if zone_update is None
                    else zone_update.boundary_reason
                ),
            )
            group4_range_delta = _typed_native_transitions_or_baseline(
                current=current_ranges,
                transitions=(
                    ()
                    if range_auction_update is None
                    else range_auction_update.range_transitions
                ),
                first_observation=baseline,
                boundary_reason=(
                    None
                    if range_auction_update is None
                    else range_auction_update.boundary_reason
                ),
            )
            live_manipulation = next(
                (
                    state
                    for state in reversed(current_manipulations)
                    if (
                        state.lifecycle is ManipulationLifecycle.SWEPT
                        and state.censored_at is None
                    )
                ),
                None,
            )
            live_manipulations = (
                ()
                if live_manipulation is None
                else (live_manipulation,)
            )
            group4_manipulation_delta = cached(
                "group4_manipulation",
                (
                    current_manipulations
                    if baseline and not group4_boundary
                    else (
                        *(
                            ()
                            if range_auction_update is None
                            else range_auction_update.manipulation_transitions
                        ),
                        *live_manipulations,
                    )
                ),
                "manipulation_id",
                retained_states=current_manipulations,
            )

        try:
            observation = MarketObservation(
                market_snapshot=market_snapshot,
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
                typed_transition_delta_available=typed_delta_available,
                liquidity_inventory_transitions_this_update=(
                    liquidity_inventory_delta
                ),
                liquidity_pool_transitions_this_update=(
                    liquidity_pool_delta
                ),
                group3_fvg_transitions_this_update=(
                    group3_fvg_delta
                ),
                group3_order_block_transitions_this_update=(
                    group3_order_block_delta
                ),
                group4_range_transitions_this_update=(
                    group4_range_delta
                ),
                group4_manipulation_transitions_this_update=(
                    group4_manipulation_delta
                ),
                group3_boundary_fvg_transitions=(
                    zone_update.fvg_transitions
                    if (
                        zone_update is not None
                        and zone_update.boundary_reason
                        in FVG_BOUNDARY_REASONS
                    )
                    else ()
                ),
                group3_boundary_order_block_transitions=(
                    zone_update.order_block_transitions
                    if (
                        zone_update is not None
                        and zone_update.boundary_reason
                        in FVG_BOUNDARY_REASONS
                    )
                    else ()
                ),
                group3_order_block_funnel=(
                    ()
                    if zone_update is None
                    else zone_update.order_block_funnel
                ),
                manipulations=(
                    ()
                    if range_auction_update is None
                    else range_auction_update.manipulations
                ),
                group4_boundary_range_transitions=(
                    range_auction_update.range_transitions
                    if (
                        range_auction_update is not None
                        and range_auction_update.boundary_reason
                        in RANGE_AUCTION_HARD_BOUNDARY_REASONS
                    )
                    else ()
                ),
                group4_boundary_manipulation_transitions=(
                    range_auction_update.manipulation_transitions
                    if (
                        range_auction_update is not None
                        and range_auction_update.boundary_reason
                        in RANGE_AUCTION_HARD_BOUNDARY_REASONS
                    )
                    else ()
                ),
                group4_ambiguous_sweep_item_ids=(
                    ()
                    if range_auction_update is None
                    else range_auction_update.ambiguous_sweep_item_ids
                ),
                group4_atr_unready_sweep_item_ids=(
                    ()
                    if range_auction_update is None
                    else range_auction_update.atr_unready_sweep_item_ids
                ),
                group4_source_dispositions=(
                    ()
                    if range_auction_update is None
                    else range_auction_update.source_dispositions
                ),
                group4_range_funnel=(
                    ()
                    if range_auction_update is None
                    else range_auction_update.range_funnel
                ),
                interaction_update=interaction_update,
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
        self._boundary_terminal_breaks.clear()
        self._boundary_reset_identity = None
        self._group4_boundary_update = None
        self._interaction_boundary_update = None
        self._prior = observation
        return observation


__all__ = [
    "CausalObserver",
    "EventMemory",
    "ObserverConfig",
]
