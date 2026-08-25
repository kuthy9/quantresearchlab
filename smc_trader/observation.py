"""Descriptive multitimeframe observation and bounded causal event memory."""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass, replace
import hashlib
import json
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
from .event_store import ImmutableEventStore, event_order_key
from .foundation_adapter import CanonicalFoundationAdapter
from .group3 import (
    CausalGroup3Tracker,
    FVG_BOUNDARY_REASONS,
    Group3BOSSource,
    Group3Protocol,
    Group3RawOnlyStructureDisposition,
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
    EventOrigin,
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
    PathSequenceStep,
    PathSequenceLifecycle,
    PathSequenceState,
    StructureLifecycle,
    SupportResistanceLifecycle,
    SupportResistanceState,
    SwingLifecycle,
    SwingRelation,
    Timeframe,
    SMC_SEMANTIC_VERSION,
    candle_identity,
    clamp,
    price_to_ticks,
    to_primitive,
    typed_event_entity_key,
)
from .market_state import (
    DOLCandidateView,
    LiquidityClusterState,
    MarketSnapshot,
    MarketSnapshotPublisher,
    RelationResolver,
    StructuralRangeState,
    SwingGeometryAssignment,
    SwingGeometryNode,
    build_structural_range,
    build_structural_legs,
    build_swing_geometry_nodes,
    foundation_dual_range_locations,
    foundation_dol_candidate_template,
    foundation_dol_protected_candidate_template,
    foundation_dol_timeframe_states,
    foundation_record_projection_event,
    session_name_phase,
    terminate_structural_range,
    update_liquidity_clusters,
    update_swing_geometry_assignments,
)
from .scene_graph import (
    ScaleSpec,
    SceneGraphDelta,
    TemporalMarketSceneGraph,
    scale_registry_id,
)
from .semantics import SemanticRegistry
from .semantic_zones import (
    CompatibleStructureKind,
    FVGTerminationCause,
    ZoneObjectKind,
)
from .semantic_foundation import (
    FoundationObjectType,
    FoundationProjectionReducer,
    FoundationRecord,
)
from .semantic_lifecycle import (
    GenerationLifecycle,
    LiquidityLevelLifecycle,
    NormalizedLifecycleTransition,
    NormalizedTransitionKind,
    StructureGenerationLifecycle,
    StructureScope,
    canonical_semantic_id,
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
    semantic_registry: str = "semantics/registry_v1_2.yaml"
    scale_specs: tuple[ScaleSpec, ...] = ()
    project_scene_graph: bool = True
    materialize_event_view: bool = True
    group4_projection_only: bool = False
    eye_authority_mode: bool = False
    canonical_foundation_enabled: bool = False
    typed_transition_delta_transport: bool = False
    persist_state_projections: bool = True


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
                    FairValueGapLifecycle.EXPIRED.value,
                }
            ),
            FairValueGapLifecycle.PARTIAL.value: frozenset(
                {
                    FairValueGapLifecycle.MITIGATED.value,
                    FairValueGapLifecycle.INVALIDATED.value,
                    FairValueGapLifecycle.EXPIRED.value,
                }
            ),
            FairValueGapLifecycle.MITIGATED.value: frozenset(),
            FairValueGapLifecycle.INVALIDATED.value: frozenset(),
            FairValueGapLifecycle.EXPIRED.value: frozenset(),
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

    def __init__(
        self,
        maximum_events: int,
        *,
        audit_store: ImmutableEventStore | None = None,
    ) -> None:
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
        self._boundary_cooling_entity_keys: set[str] = set()
        # Sequence-clock retention is synchronized once after the completed
        # observation has projected every same-clock event.  Keeping only a
        # pending bit here avoids rescanning every retained lifecycle timeline
        # after each individual append when the clock table is above its
        # bounded cleanup threshold.
        self._sequence_counts_prune_pending = False
        self._semantic_version: str | None = None
        self._audit_store = audit_store
        self._audit_pending: list[MarketEvent] = []

    @property
    def last_minute_end(self) -> pd.Timestamp | None:
        return self._last_minute_end

    @property
    def clock_coverage_start(self) -> pd.Timestamp | None:
        return self._clock_coverage_start

    @property
    def semantic_version(self) -> str | None:
        return self._semantic_version

    @staticmethod
    def _required_event_clock(event: MarketEvent) -> pd.Timestamp:
        age_origin = (
            event.formed_at
            or event.confirmed_at
            or event.event_time
        )
        return min(age_origin, event.known_at)

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

    def _retain_recent_event(self, event: MarketEvent) -> None:
        """Retain one event in the bounded hot view without duplicating it."""

        if event.event_id in self._ids:
            return
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

    def append(
        self,
        event: MarketEvent,
        *,
        include_in_recent: bool = True,
        sequence_floor: int | None = None,
        audit: bool = True,
    ) -> MarketEvent:
        if type(include_in_recent) is not bool:
            raise ValueError(
                "event-memory recent inclusion flag must be boolean"
            )
        if (
            self._semantic_version is not None
            and event.semantic_version != self._semantic_version
        ):
            raise ValueError(
                "event memory cannot mix semantic versions"
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
            return existing
        if entity_key is not None:
            self._validate_timeline_append(entity_key, event)
        if self._semantic_version is None:
            self._semantic_version = event.semantic_version
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
        if audit and self._audit_store is not None:
            self._audit_pending.append(event)
        if include_in_recent:
            self._retain_recent_event(event)
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
            self._sequence_counts_prune_pending = True
        return event

    def flush_audit(self) -> int:
        """Atomically append this update's events in canonical availability order."""

        if self._audit_store is None or not self._audit_pending:
            self._audit_pending.clear()
            return 0
        ordered = tuple(sorted(self._audit_pending, key=event_order_key))
        appended = self._audit_store.append_batch(ordered)
        self._audit_pending.clear()
        return appended

    def audit_event_including_pending(self, event_id: str) -> MarketEvent | None:
        """Resolve an exact audit event without flushing the current update."""

        pending = tuple(
            event for event in self._audit_pending if event.event_id == event_id
        )
        if len(pending) > 1:
            raise ValueError("pending audit event identity is duplicated")
        committed = (
            None if self._audit_store is None else self._audit_store.get(event_id)
        )
        if pending and committed is not None and pending[0] != committed:
            raise ValueError("pending audit event conflicts with committed identity")
        return pending[0] if pending else committed

    def transfer_pending_from(self, prior: "EventMemory") -> int:
        """Move uncommitted events while preserving audit and causal timelines.

        Typed live prefixes are already retained solely for a terminal join;
        they deliberately do not re-enter the new epoch's bounded recent view.
        """

        if not isinstance(prior, EventMemory) or prior is self:
            raise TypeError("pending event-memory source is invalid")
        if self._audit_store is not prior._audit_store:
            raise ValueError("pending events cannot change audit store")
        if self._audit_pending:
            raise ValueError("pending events require an empty destination")
        pending = tuple(prior._audit_pending)
        for event in pending:
            entity_key = typed_event_entity_key(event)
            existing = self._existing_event(event.event_id, entity_key)
            if existing is not None:
                if (
                    self._normalized_event(existing)
                    != self._normalized_event(event)
                ):
                    raise ValueError(
                        "pending event conflicts with retained boundary prefix"
                    )
                transferred = existing
            else:
                transferred = self.append(
                    event,
                    include_in_recent=event.event_id in prior._ids,
                    audit=False,
                )
            self._audit_pending.append(transferred)
        prior._audit_pending.clear()
        return len(pending)

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
            # ``append`` enforces a strictly increasing lifecycle clock, so
            # the tail is the maximum observation time for this entity.
            if timeline and timeline[-1].observed_at > clock:
                raise ValueError(
                    "retained entity timeline contains the future"
                )
        if getattr(self, "_sequence_counts_prune_pending", False):
            # Match the former post-append retention boundary exactly: the
            # final event of this completed update could still see every
            # pre-synchronization timeline.  Timeline membership is narrowed
            # only after the clock table has been pruned against that same
            # view, preserving checkpoint state as well as same-clock counts.
            retained_clocks = {
                event.observed_at
                for event in self._events
            } | {
                event.observed_at
                for timeline in self._entity_timelines.values()
                for event in timeline
            }
            self._sequence_counts = {
                event_clock: count
                for event_clock, count in self._sequence_counts.items()
                if event_clock in retained_clocks
            }
            self._sequence_counts_prune_pending = False
        next_timelines = {
            # Retention changes dictionary membership, never lifecycle list
            # ownership.  Keeping the internal list avoids copying every
            # retained history on each completed minute; public readers still
            # receive immutable tuples from ``entity_timelines``/``timeline``.
            key: self._entity_timelines[key]
            for key in sorted(retained)
        }
        self._entity_timelines = next_timelines
        self._retained_entity_keys = retained
        self._incomplete_entity_keys.intersection_update(retained)
        self._boundary_cooling_entity_keys.intersection_update(retained)
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

    def live_entity_keys(self) -> tuple[str, ...]:
        """Return retained lifecycle prefixes which can still transition.

        Reducer snapshots are bounded projections and may transiently omit an
        entity before its later terminal transition is emitted.  The prefix is
        therefore part of the hot causal state until its latest lifecycle has
        no registered successor.  Terminal timelines remain eligible for the
        normal snapshot-driven cooling performed by
        :meth:`sync_retained_entity_timelines`.
        """

        output: list[str] = []
        for entity_key, timeline in self._entity_timelines.items():
            if (
                not timeline
                or entity_key in self._boundary_cooling_entity_keys
            ):
                continue
            namespace = self._timeline_namespace(entity_key)
            if self._TIMELINE_TRANSITIONS[namespace][timeline[-1].lifecycle]:
                output.append(entity_key)
        return tuple(sorted(output))

    def retain_live_prefixes_from(
        self,
        prior: "EventMemory",
        *,
        asof: pd.Timestamp,
    ) -> tuple[str, ...]:
        """Carry only transitionable prefixes across one hard boundary.

        Boundary reducers emit their terminal transitions after the Observer
        has reset contract-local state.  Keeping the old recent deque or every
        terminal timeline would turn EventMemory into an unbounded audit
        archive; copying only live prefixes gives those same-clock terminal
        events their causal history.  The imported keys are excluded from the
        ordinary live-retention union, so a key not exposed by the new typed
        snapshot cools on the next synchronization.
        """

        if not isinstance(prior, EventMemory) or prior is self:
            raise TypeError("boundary EventMemory source is invalid")
        clock = pd.Timestamp(asof)
        if clock.tzinfo is None:
            raise ValueError("boundary prefix cutoff must be timezone aware")
        if self._entity_timelines or self._retained_entity_keys:
            raise ValueError("boundary prefixes require an empty EventMemory")
        keys = prior.live_entity_keys()
        timelines = {
            key: list(prior._entity_timelines[key])
            for key in keys
        }
        versions = {
            event.semantic_version
            for timeline in timelines.values()
            for event in timeline
        }
        if len(versions) > 1 or (
            versions
            and prior.semantic_version not in versions
        ):
            raise ValueError("boundary prefix mixes semantic versions")
        self._semantic_version = prior.semantic_version
        if any(
            event.observed_at > clock
            for timeline in timelines.values()
            for event in timeline
        ):
            raise ValueError("boundary prefix contains a future event")
        if (
            self._clock_coverage_start is not None
            and any(
                self._required_event_clock(event)
                < self._clock_coverage_start
                for timeline in timelines.values()
                for event in timeline
            )
        ):
            raise ValueError(
                "boundary prefix predates retained 1m clock coverage"
            )
        self._entity_timelines = timelines
        self._retained_entity_keys = set(keys)
        self._incomplete_entity_keys = (
            prior._incomplete_entity_keys & set(keys)
        )
        self._boundary_cooling_entity_keys = set(keys)
        retained_event_ids = {
            event.event_id
            for timeline in timelines.values()
            for event in timeline
        }
        self._closed_durations = {
            event_id: duration
            for event_id, duration in prior._closed_durations.items()
            if event_id in retained_event_ids
        }
        self._latest_by_entity = {
            entity_id: event
            for entity_id, event in prior._latest_by_entity.items()
            if event.event_id in retained_event_ids
        }
        retained_clocks = {
            event.observed_at
            for timeline in timelines.values()
            for event in timeline
        }
        self._sequence_counts = {
            event_clock: count
            for event_clock, count in prior._sequence_counts.items()
            if event_clock in retained_clocks
        }
        self._sequence_counts_prune_pending = False
        return keys

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

        # A retained lifecycle commonly contributes the same formation clock
        # to several events, while the current ``asof`` is shared by every
        # active duration and age.  Resolve each timestamp onto the existing
        # real-minute clock once per materialization instead of repeatedly
        # constructing Timedelta objects and bisecting synthetic runs.
        #
        # Keep the sub-minute remainder alongside the minute coordinate.  The
        # remainder correction makes this exactly equivalent to
        # ``floor((end - start) / one_minute)`` even for non-aligned aware
        # timestamps; synthetic minutes retain the original (start, end]
        # inclusion convention from ``_synthetic_count_through``.
        minute_ns = 60_000_000_000
        clock_coordinates: dict[int, tuple[int, int, int]] = {}

        def coordinate(
            timestamp: pd.Timestamp,
        ) -> tuple[int, int, int]:
            timestamp_ns = int(timestamp.value)
            cached = clock_coordinates.get(timestamp_ns)
            if cached is not None:
                return cached
            minute, remainder = divmod(timestamp_ns, minute_ns)
            value = (
                minute,
                remainder,
                self._synthetic_count_through(timestamp),
            )
            clock_coordinates[timestamp_ns] = value
            return value

        def elapsed_minutes(
            start: pd.Timestamp,
            end: pd.Timestamp,
        ) -> int:
            start_minute, start_remainder, start_synthetic = coordinate(
                start
            )
            end_minute, end_remainder, end_synthetic = coordinate(end)
            wall_minutes = max(
                0,
                end_minute
                - start_minute
                - int(end_remainder < start_remainder),
            )
            return max(
                0,
                wall_minutes - (end_synthetic - start_synthetic),
            )

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
                    elapsed_minutes(
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
                elapsed_minutes(origin, asof),
            )
        for timeline in self._entity_timelines.values():
            for index, event in enumerate(timeline):
                if index + 1 < len(timeline):
                    end = timeline[index + 1].observed_at
                    durations[event.event_id] = max(
                        0,
                        elapsed_minutes(event.observed_at, end),
                    )
                elif event.ended_at is not None:
                    durations[event.event_id] = 0
                else:
                    durations[event.event_id] = max(
                        0,
                        elapsed_minutes(event.observed_at, asof),
                    )
                origin = (
                    event.formed_at
                    or event.confirmed_at
                    or event.event_time
                )
                ages[event.event_id] = max(
                    0,
                    elapsed_minutes(origin, asof),
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
    event_time: pd.Timestamp | None = None,
    known_at: pd.Timestamp | None = None,
    semantic_version: str = SMC_SEMANTIC_VERSION,
    evidence: Mapping[str, object] | None = None,
    zone: tuple[float, float] | None = None,
    source_event_ids: Iterable[str] = (),
    source_data_ids: Iterable[str] = (),
    source_entity_ids: Iterable[str] = (),
    context_event_ids: Iterable[str] = (),
    origin: EventOrigin = EventOrigin.LEGACY_TRANSPORT,
) -> MarketEvent:
    source_ids = tuple(
        str(value) for value in source_ids if value is not None
    )
    explicit_event_ids = tuple(
        str(value) for value in source_event_ids if value is not None
    )
    data_ids = tuple(
        str(value) for value in source_data_ids if value is not None
    )
    entity_ids = tuple(
        str(value) for value in source_entity_ids if value is not None
    )
    context_ids = tuple(
        str(value) for value in context_event_ids if value is not None
    )
    raw = (
        f"{semantic_version}|{kind.value}|{observed_at.isoformat()}|"
        f"{timeframe.value}|{side}|"
        f"{price}|{'|'.join(source_ids)}|{'|'.join(explicit_event_ids)}|"
        f"{'|'.join(data_ids)}|{'|'.join(entity_ids)}|"
        f"{'|'.join(context_ids)}|{EventOrigin(origin).value}|"
        f"{entity_id}|{lifecycle}"
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
        event_time=event_time or observed_at,
        known_at=known_at or observed_at,
        semantic_version=semantic_version,
        evidence={} if evidence is None else dict(evidence),
        zone=zone,
        source_event_ids=explicit_event_ids,
        source_data_ids=data_ids,
        source_entity_ids=entity_ids,
        context_event_ids=context_ids,
        origin=origin,
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
                    Path(__file__).resolve().parents[1] / configured_path
                )
            if (
                configured_path.resolve()
                != semantic_registry.source_path.resolve()
            ):
                raise ValueError(
                    "observer semantic registry path differs from selection"
                )
        self.semantic_registry = semantic_registry
        self.audit_store = ImmutableEventStore(
            semantic_version=self.semantic_registry.semantic_version,
            definition_identity=self.semantic_registry.definition_identity,
        )
        self.market_snapshot_publisher = MarketSnapshotPublisher(
            semantic_registry_identity=self.semantic_registry.identity,
            # The normal Trading Eye is event-authoritative.  The explicitly
            # constrained Group-4 authority scanner does not publish atomic
            # BAR/semantic facts and therefore remains a labelled projection
            # compatibility mode instead of pretending to be atomic.
            atomic_authority=not self.config.group4_projection_only,
        )
        self.last_market_snapshot: MarketSnapshot | None = None
        if type(self.config.project_scene_graph) is not bool:
            raise ValueError("scene-graph projection flag must be boolean")
        if type(self.config.materialize_event_view) is not bool:
            raise ValueError("event-view materialization flag must be boolean")
        if type(self.config.group4_projection_only) is not bool:
            raise ValueError("Group 4 projection-only flag must be boolean")
        if type(self.config.eye_authority_mode) is not bool:
            raise ValueError("eye-authority mode flag must be boolean")
        if type(self.config.canonical_foundation_enabled) is not bool:
            raise ValueError(
                "canonical-foundation enabled flag must be boolean"
            )
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
                self.config.group3_protocol,
                self.config.group4_protocol,
                self.config.group5_protocol,
            )
            if (
                self.config.group4_projection_only
                or any(protocol is None for protocol in typed_protocols)
            ):
                raise ValueError(
                    "eye-authority mode requires all typed protocols, "
                    "and Group 4 projection-only mode disabled"
                )
        if self.config.canonical_foundation_enabled:
            typed_protocols = (
                self.config.structure_protocol,
                self.config.liquidity_protocol,
                self.config.displacement_protocol,
                self.config.group3_protocol,
                self.config.group4_protocol,
                self.config.group5_protocol,
            )
            if (
                self.config.group4_projection_only
                or any(protocol is None for protocol in typed_protocols)
            ):
                raise ValueError(
                    "canonical-foundation projection requires all typed "
                    "protocols and atomic Group 4 authority"
                )
        if (
            not self.config.eye_authority_mode
            and not self.config.canonical_foundation_enabled
            and not self.config.materialize_event_view
            and (
                self.config.project_scene_graph
                or self.config.group4_protocol is None
                or self.config.displacement_protocol is not None
                or self.config.group3_protocol is not None
                or self.config.group5_protocol is not None
            )
        ):
            raise ValueError(
                "a lightweight event view is limited to the Group 1-2 + "
                "Group 4 authority scanner with Scene Graph disabled"
            )
        if self.config.group4_projection_only and (
            self.config.materialize_event_view
            or self.config.project_scene_graph
            or self.config.group4_protocol is None
            or self.config.displacement_protocol is not None
            or self.config.group3_protocol is not None
            or self.config.group5_protocol is not None
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
        self._last_group3_foundation_projection: Group3Update | None = None
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
        self.memory = EventMemory(
            self.config.memory_events,
            audit_store=self.audit_store,
        )
        # Foundation v2 is an additive, non-action-authoritative projection
        # over the exact atomic facts emitted by the production Eye.  The
        # explicit production flag is independent of the scan-only Eye mode,
        # which intentionally does not evaluate execution reality.
        # Compatibility observers keep their historical snapshot/fingerprint
        # contract unless either path opts in.
        self._foundation_adapter = (
            CanonicalFoundationAdapter(tick_size=self.config.tick_size)
            if (
                self.config.eye_authority_mode
                or self.config.canonical_foundation_enabled
            )
            else None
        )
        self._foundation_geometry_nodes: dict[str, SwingGeometryNode] = {}
        self._foundation_geometry_assignments: tuple[
            SwingGeometryAssignment, ...
        ] = ()
        self._foundation_active_clusters: tuple[
            LiquidityClusterState, ...
        ] = ()
        self._foundation_structural_ranges: dict[
            Timeframe, StructuralRangeState
        ] = {}
        self._foundation_fvg_contexts: dict[
            str, tuple[str | None, str | None]
        ] = {}
        # Compact typed revision cache for immutable/transition DTO plans.
        # It is a local optimization only: the adapter projection remains the
        # authority.  A staged copy is committed with the projection so a
        # failed publication cannot suppress a later valid revision.
        self._foundation_plan_revisions: dict[
            tuple[object, ...], object
        ] = {}
        self._foundation_dol_templates: dict[str, DOLCandidateView] = {}
        self._foundation_published_record_ids: set[str] = set()
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
        self._known_structural_leg_ids: set[str] = set()
        self._known_structural_leg_order: deque[str] = deque(
            maxlen=max(2048, self.config.memory_events * 4)
        )
        self._confirmed_swing_event_ids: dict[str, str] = {}
        self._structural_leg_event_ids: dict[str, str] = {}
        self._structure_direction_event_ids: dict[str, str] = {}
        self._latest_structure_direction_event_ids: dict[
            Timeframe,
            str,
        ] = {}
        self._bar_event_ids_by_candle_id: dict[str, str] = {}
        self._bar_close_by_candle_id: dict[str, float] = {}
        self._bar_close_by_event_id: dict[str, float] = {}
        self._bar_event_ids_by_timeframe: dict[
            Timeframe,
            list[tuple[pd.Timestamp, str]],
        ] = {
            timeframe: [] for timeframe in self._active_timeframes
        }
        # Keep the inclusive normalized M1 clock/root index above for session
        # replay and cold-prefix initialization.  Semantic detectors consume
        # only real-completed bars, matching StructureTracker and the other
        # primitive reducers that treat synthetic no-trade minutes as clock
        # advancement only.
        self._real_bar_event_ids_by_timeframe: dict[
            Timeframe,
            list[tuple[pd.Timestamp, str]],
        ] = {
            timeframe: [] for timeframe in self._active_timeframes
        }
        self._level_touch_event_ids: dict[
            tuple[str, pd.Timestamp],
            str,
        ] = {}
        self._candidate_level_event_ids: dict[str, str] = {}
        self._known_level_touch_ids: set[str] = set()
        # Source reducers expose complete touch histories for live zones.
        # Evicting these occurrence keys causes old touches to be rediscovered
        # on every later frame, so retain the compact IDs for the contract
        # epoch and clear them only at a hard boundary.
        self._known_level_touch_order: deque[str] = deque()
        self._penetration_event_ids: dict[
            tuple[str, Timeframe, pd.Timestamp],
            str,
        ] = {}
        self._raw_break_event_ids: dict[str, str] = {}
        self._raw_only_structure_dispositions: dict[
            str,
            Group3RawOnlyStructureDisposition,
        ] = {}
        self._qualified_structure_event_ids: dict[str, str] = {}
        self._displacement_event_ids: dict[str, str] = {}
        self._protected_swing_event_ids: dict[str, str] = {}
        self._terminal_crossing_events: dict[str, MarketEvent] = {}
        self._fvg_created_event_ids: dict[str, str] = {}
        self._fvg_terminal_event_ids: dict[str, str] = {}
        self._origin_zone_created_event_ids: dict[str, str] = {}
        self._last_market_epoch_reset_event_id: str | None = None
        self._range_created_event_ids: dict[str, str] = {}
        self._range_active_event_ids: dict[str, str] = {}
        self._range_terminal_event_ids: dict[str, str] = {}
        self._range_boundary_level_ids: dict[tuple[str, str], str] = {}
        self._last_invalidated_range_event_id: str | None = None
        self._known_displacement_transition_ids: set[str] = set()
        self._known_displacement_transition_order: deque[str] = deque(
            maxlen=max(512, self.config.memory_events * 2)
        )
        self._structural_leg_cache: dict[
            Timeframe,
            tuple[tuple[str, ...], tuple],
        ] = {}
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
    def group5_protocol(self) -> Group5Protocol | None:
        return (
            None
            if self._group5_reducer is None
            else self._group5_reducer.protocol
        )

    @property
    def last_group3_foundation_projection(self) -> Group3Update | None:
        """Latest committed Group-3 foundation view, outside v1.2 DTOs."""

        return self._last_group3_foundation_projection

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
        self.market_snapshot_publisher.on_boundary()
        self.last_market_snapshot = None
        # Delta transport is an authority-scan projection, not causal state.
        # A hard epoch boundary must not retain prior-contract signatures.
        self._typed_delta_signatures.clear()
        self._last_group3_foundation_projection = None
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
        reset_event = self._append_semantic_atomic(
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
        self._last_market_epoch_reset_event_id = reset_event.event_id
        self._prior = None
        self._known_level_ids.clear()
        self._known_level_order.clear()
        self._known_structural_leg_ids.clear()
        self._known_structural_leg_order.clear()
        self._confirmed_swing_event_ids.clear()
        self._structural_leg_event_ids.clear()
        self._structure_direction_event_ids.clear()
        self._latest_structure_direction_event_ids.clear()
        self._bar_event_ids_by_candle_id.clear()
        self._bar_close_by_candle_id.clear()
        self._bar_close_by_event_id.clear()
        for bar_events in self._bar_event_ids_by_timeframe.values():
            bar_events.clear()
        for bar_events in self._real_bar_event_ids_by_timeframe.values():
            bar_events.clear()
        self._level_touch_event_ids.clear()
        self._candidate_level_event_ids.clear()
        self._known_level_touch_ids.clear()
        self._known_level_touch_order.clear()
        self._penetration_event_ids.clear()
        self._raw_break_event_ids.clear()
        self._raw_only_structure_dispositions.clear()
        self._qualified_structure_event_ids.clear()
        self._displacement_event_ids.clear()
        self._protected_swing_event_ids.clear()
        self._terminal_crossing_events.clear()
        self._fvg_created_event_ids.clear()
        self._fvg_terminal_event_ids.clear()
        self._origin_zone_created_event_ids.clear()
        self._range_created_event_ids.clear()
        self._range_active_event_ids.clear()
        self._range_terminal_event_ids.clear()
        self._range_boundary_level_ids.clear()
        self._last_invalidated_range_event_id = None
        self._known_displacement_transition_ids.clear()
        self._known_displacement_transition_order.clear()
        self._structural_leg_cache.clear()
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
        self._pending_level_crossings.clear()
        self._reference_periods.clear()
        self._reference_inventory.clear()
        self._reference_candidate_sources.clear()
        self._reference_last_end = None
        self._reference_coverage_start = None
        for item in (
            () if self.config.group4_projection_only else failed_bos
        ):
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

    @staticmethod
    def _execution_not_evaluated() -> ExecutionObservation:
        """Return an inert contract value for Eye-only semantic replay."""

        return ExecutionObservation(
            spread_points=0.0,
            expected_slippage_points=0.0,
            expected_round_trip_cost_points=0.0,
            minutes_to_deadline=0,
            fillability=0.0,
            data_age_seconds=0.0,
            size_available=None,
            anomalies=(),
            source="not_evaluated",
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
                            and not self.config.group4_projection_only
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
                            liquidity_tracker.snapshot(
                                group4_sources_only=True,
                                include_support_resistance=True,
                            )
                            if self.config.group4_projection_only
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

    def _append_semantic_atomic(
        self,
        kind: EventKind,
        known_at: pd.Timestamp,
        timeframe: Timeframe,
        side: str | None,
        price: float | None,
        strength: float,
        source_event_ids: Iterable[str] = (),
        evidence: Mapping[str, object] | None = None,
        *,
        direction: Direction | None = None,
        event_time: pd.Timestamp | None = None,
        zone: tuple[float, float] | None = None,
        source_data_ids: Iterable[str] = (),
        source_entity_ids: Iterable[str] = (),
        context_event_ids: Iterable[str] = (),
    ) -> MarketEvent:
        registry = getattr(self, "semantic_registry", None)
        if registry is None:
            raise ValueError(
                "canonical semantic emitter requires a loaded registry"
            )
        # Epoch reset is journal/control infrastructure rather than an SMC
        # semantic concept.  It intentionally remains outside the semantic
        # registry while sharing the immutable append path.
        if (
            kind is not EventKind.MARKET_EPOCH_RESET
            and kind not in registry.canonical_emitted_event_kinds
        ):
            binding = registry.event_binding_by_kind.get(kind)
            status = "unregistered" if binding is None else binding.status
            raise ValueError(
                "canonical semantic emitter rejects non-emitted event kind: "
                f"{kind.value} ({status})"
            )
        source_event_ids = tuple(
            dict.fromkeys(
                str(value) for value in source_event_ids if value is not None
            )
        )
        source_data_ids = tuple(
            dict.fromkeys(
                str(value) for value in source_data_ids if value is not None
            )
        )
        source_entity_ids = tuple(
            dict.fromkeys(
                str(value) for value in source_entity_ids if value is not None
            )
        )
        context_event_ids = tuple(
            dict.fromkeys(
                str(value) for value in context_event_ids if value is not None
            )
        )
        session_name, session_phase = session_name_phase(known_at)
        payload = {
            "canonical_semantic": True,
            "projection_only": False,
            "session_name": session_name,
            "session_phase": session_phase,
            **({} if evidence is None else dict(evidence)),
        }
        event = _event(
            kind,
            known_at,
            timeframe,
            side,
            price,
            strength,
            source_event_ids,
            payload,
            direction=direction,
            event_time=event_time or known_at,
            known_at=known_at,
            evidence=payload,
            zone=zone,
            source_event_ids=source_event_ids,
            source_data_ids=source_data_ids,
            source_entity_ids=source_entity_ids,
            context_event_ids=context_event_ids,
            origin=EventOrigin.SEMANTIC_ATOMIC,
        )
        identity_payload = {
            "semantic_version": event.semantic_version,
            "semantic_type": event.kind.value,
            "event_time": event.event_time,
            "known_at": event.known_at,
            "timeframe": event.timeframe.value,
            "side": event.side,
            "price": event.price,
            "direction": (
                None if event.direction is None else event.direction.value
            ),
            "source_event_ids": event.source_event_ids,
            "source_data_ids": event.source_data_ids,
            "source_entity_ids": event.source_entity_ids,
            "context_event_ids": event.context_event_ids,
            "origin": event.origin.value,
            "evidence": event.evidence,
            "zone": event.zone,
        }
        event = replace(
            event,
            event_id=hashlib.sha256(
                json.dumps(
                    to_primitive(identity_payload),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24],
        )
        existing = self.audit_store.get(event.event_id)
        if existing is not None:
            # Event identity intentionally excludes the convenience
            # ``strength`` score: a long-lived zone can revise that score
            # after the touch occurred. If a bounded hot-memory key is later
            # evicted and the same occurrence is rediscovered, retain the
            # first-known frozen score rather than rewriting history or
            # creating a second semantic occurrence.
            retry = replace(
                event,
                sequence_no=existing.sequence_no,
                strength=existing.strength,
            )
            if retry != existing:
                raise ValueError(
                    "canonical semantic event id conflicts with audit history: "
                    f"{event.event_id} ({event.kind.value})"
                )
            return existing
        self.memory.append(event, include_in_recent=False)
        return event

    def _normalized_crossing_level_id(self, level_id: str) -> str:
        value = str(level_id)
        if value.startswith("swing:"):
            return value
        if (
            value in self._confirmed_swing_event_ids
            and f"swing:{value}" in self._candidate_level_event_ids
        ):
            return f"swing:{value}"
        return value

    def _crossing_generation_id(
        self,
        *,
        level_id: str,
        timeframe: Timeframe,
        crossed_at: pd.Timestamp,
    ) -> str:
        normalized_level_id = self._normalized_crossing_level_id(level_id)
        payload = (
            f"{self.semantic_registry.semantic_version}|crossing-v1|"
            f"{timeframe.value}|{normalized_level_id}|"
            f"{pd.Timestamp(crossed_at).isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _penetration_key(
        self,
        *,
        level_id: str,
        timeframe: Timeframe,
        crossed_at: pd.Timestamp,
    ) -> tuple[str, Timeframe, pd.Timestamp]:
        """Return the unique lookup key for one crossing generation."""

        return (
            self._normalized_crossing_level_id(level_id),
            timeframe,
            pd.Timestamp(crossed_at),
        )

    def _live_protected_assignments(
        self,
        timeframe: Timeframe,
        *,
        asof: pd.Timestamp,
    ) -> tuple[tuple[str, str, MarketEvent], ...]:
        """Resolve the exact pending-or-committed protection for one TF.

        ``_protected_swing_event_ids`` is the producer-side custody index for
        the assignment that the persistent market-state reducer considers
        live.  Reading through ``audit_event_including_pending`` is important:
        an assignment and a later break may be published at the same observer
        clock before the atomic audit batch is flushed.
        """

        clock = pd.Timestamp(asof)
        if clock.tzinfo is None:
            raise ValueError("protected-swing custody clock must be aware")
        assignments: list[tuple[str, str, MarketEvent]] = []
        for swing_id, event_id in tuple(
            self._protected_swing_event_ids.items()
        ):
            event = self.memory.audit_event_including_pending(event_id)
            if (
                event is None
                or event.kind is not EventKind.PROTECTED_SWING_ASSIGNED
                or not event.is_canonical_semantic
                or event.evidence.get("protected_swing_id") != swing_id
                or swing_id not in event.source_entity_ids
                or event.direction not in {Direction.LONG, Direction.SHORT}
                or event.price is None
                or event.known_at > clock
            ):
                raise ValueError(
                    "live protected-swing custody references an invalid "
                    f"assignment: {swing_id} -> {event_id}"
                )
            if event.timeframe is timeframe:
                assignments.append((swing_id, event_id, event))
        if len(assignments) > 1:
            raise ValueError(
                "one timeframe cannot retain multiple live protected-swing "
                "assignments"
            )
        return tuple(assignments)

    def _historical_live_protected_assignment(
        self,
        timeframe: Timeframe,
        *,
        at_event: MarketEvent,
    ) -> MarketEvent | None:
        """Replay protected custody only through one immutable event order."""

        if (
            at_event.kind is not EventKind.RAW_BOUNDARY_BREAK
            or not at_event.is_canonical_semantic
            or at_event.timeframe is not timeframe
        ):
            raise ValueError(
                "raw-only QOB disposition historical custody requires its "
                "exact raw boundary-break event"
            )
        events_by_id = {
            event.event_id: event for event in self.audit_store.events()
        }
        for event in self.memory._audit_pending:
            prior = events_by_id.get(event.event_id)
            if prior is not None:
                committed = self.audit_store.get(event.event_id)
                if (
                    committed is None
                    or self.audit_store.event_digest(event.event_id)
                    != self.audit_store.recompute_event_digest(event)
                ):
                    raise ValueError(
                        "pending protected-swing history conflicts with audit"
                    )
                continue
            events_by_id[event.event_id] = event
        cutoff = event_order_key(at_event)
        live: MarketEvent | None = None
        for event in sorted(events_by_id.values(), key=event_order_key):
            if event_order_key(event) > cutoff:
                break
            if (
                event.kind is EventKind.PROTECTED_SWING_ASSIGNED
                and event.timeframe is timeframe
            ):
                protected_swing_id = event.evidence.get(
                    "protected_swing_id"
                )
                if (
                    not event.is_canonical_semantic
                    or not isinstance(protected_swing_id, str)
                    or not protected_swing_id
                    or protected_swing_id not in event.source_entity_ids
                    or event.direction
                    not in {Direction.LONG, Direction.SHORT}
                    or event.price is None
                ):
                    raise ValueError(
                        "historical protected-swing assignment is invalid"
                    )
                live = event
                continue
            if (
                live is None
                or event.kind is not EventKind.ACCEPTANCE_CONFIRMED
                or event.evidence.get("protected_swing_event_id")
                != live.event_id
            ):
                continue
            protected_swing_id = live.evidence.get("protected_swing_id")
            source_timeframe = event.evidence.get(
                "source_timeframe", event.timeframe.value
            )
            if (
                not event.is_canonical_semantic
                or event.evidence.get("protected_swing_id")
                != protected_swing_id
                or live.event_id not in event.context_event_ids
                or source_timeframe != timeframe.value
                or event.direction
                is not (
                    Direction.SHORT
                    if live.direction is Direction.LONG
                    else Direction.LONG
                )
            ):
                raise ValueError(
                    "historical protected-swing terminal is invalid"
                )
            live = None
        return live

    def _replace_live_protected_assignment(
        self,
        event: MarketEvent,
        prior_assignments: tuple[tuple[str, str, MarketEvent], ...],
    ) -> None:
        """Publish a successfully appended assignment into producer custody."""

        protected_swing_id = event.evidence.get("protected_swing_id")
        if (
            event.kind is not EventKind.PROTECTED_SWING_ASSIGNED
            or not event.is_canonical_semantic
            or not isinstance(protected_swing_id, str)
            or not protected_swing_id
            or protected_swing_id not in event.source_entity_ids
        ):
            raise ValueError(
                "protected-swing custody requires a valid assignment"
            )
        for swing_id, event_id, prior in prior_assignments:
            if prior.timeframe is not event.timeframe:
                raise ValueError(
                    "protected-swing replacement crossed timeframe custody"
                )
            if self._protected_swing_event_ids.get(swing_id) == event_id:
                self._protected_swing_event_ids.pop(swing_id)
        self._protected_swing_event_ids[protected_swing_id] = event.event_id

    def _append_crossing_resolution(
        self,
        kind: EventKind,
        resolved_at: pd.Timestamp,
        timeframe: Timeframe,
        side: str,
        price: float,
        strength: float,
        source_event_ids: Iterable[str],
        evidence: Mapping[str, object],
        *,
        direction: Direction,
        crossed_at: pd.Timestamp,
        zone: tuple[float, float] | None = None,
        context_event_ids: Iterable[str] = (),
    ) -> MarketEvent:
        """Append exactly one terminal result for one crossing generation."""

        if kind not in {
            EventKind.SWEEP_CONFIRMED,
            EventKind.ACCEPTANCE_CONFIRMED,
        }:
            raise ValueError("crossing terminal kind must be sweep or acceptance")
        level_id = str(evidence.get("level_id", ""))
        if not level_id:
            raise ValueError("crossing terminal requires a level_id")
        generation_id = self._crossing_generation_id(
            level_id=level_id,
            timeframe=timeframe,
            crossed_at=crossed_at,
        )
        normalized_level_id = self._normalized_crossing_level_id(level_id)
        protected_swing_id = (
            normalized_level_id.removeprefix("swing:")
            if normalized_level_id.startswith("swing:")
            else normalized_level_id
        )
        protected_event_id = self._protected_swing_event_ids.get(
            protected_swing_id
        )
        protected_assignment = (
            self.memory.audit_event_including_pending(protected_event_id)
            if protected_event_id is not None
            else None
        )
        if (
            protected_assignment is not None
            and protected_assignment.known_at > pd.Timestamp(resolved_at)
        ):
            raise ValueError(
                "protected assignment cannot be known after crossing "
                "resolution"
            )
        source_ids = tuple(source_event_ids)
        context_ids = tuple(context_event_ids)
        if (
            protected_event_id is not None
            and protected_event_id not in context_ids
        ):
            context_ids = (*context_ids, protected_event_id)
        prior = self._terminal_crossing_events.get(generation_id)
        if prior is not None:
            if (
                prior.kind is not kind
                or prior.direction is not direction
                or prior.side != side
                or prior.timeframe is not timeframe
                or prior.known_at != pd.Timestamp(resolved_at)
                or prior.event_time != pd.Timestamp(crossed_at)
                or prior.evidence.get("crossing_generation_id")
                != generation_id
                or prior.evidence.get("crossed_at")
                != pd.Timestamp(crossed_at).isoformat()
            ):
                raise ValueError(
                    "one crossing generation produced conflicting terminal "
                    f"resolutions: {generation_id}"
                )
            return prior
        payload = {
            **dict(evidence),
            "crossing_generation_id": generation_id,
            "crossed_at": pd.Timestamp(crossed_at).isoformat(),
            "resolved_at": pd.Timestamp(resolved_at).isoformat(),
            **(
                {
                    "protected_swing_id": protected_swing_id,
                    "protected_swing_event_id": protected_event_id,
                }
                if protected_event_id is not None
                else {}
            ),
        }
        event = self._append_semantic_atomic(
            kind,
            resolved_at,
            timeframe,
            side,
            price,
            strength,
            source_ids,
            payload,
            direction=direction,
            event_time=crossed_at,
            zone=zone,
            source_entity_ids=(normalized_level_id,),
            context_event_ids=context_ids,
        )
        self._terminal_crossing_events[generation_id] = event
        opposite_assignment_direction = (
            Direction.SHORT
            if (
                protected_assignment is not None
                and protected_assignment.direction is Direction.LONG
            )
            else Direction.LONG
            if (
                protected_assignment is not None
                and protected_assignment.direction is Direction.SHORT
            )
            else None
        )
        if (
            kind is EventKind.ACCEPTANCE_CONFIRMED
            and protected_assignment is not None
            and protected_assignment.kind
            is EventKind.PROTECTED_SWING_ASSIGNED
            and protected_assignment.is_canonical_semantic
            and (
                protected_assignment.timeframe is timeframe
                or (
                    timeframe is Timeframe.M1
                    and event.evidence.get("source_timeframe")
                    == protected_assignment.timeframe.value
                )
            )
            and protected_assignment.evidence.get("protected_swing_id")
            == protected_swing_id
            and protected_swing_id
            in protected_assignment.source_entity_ids
            and event.kind is EventKind.ACCEPTANCE_CONFIRMED
            and event.direction is opposite_assignment_direction
            and event.evidence.get("protected_swing_id")
            == protected_swing_id
            and event.evidence.get("protected_swing_event_id")
            == protected_event_id
            and protected_event_id in event.context_event_ids
            and self._protected_swing_event_ids.get(protected_swing_id)
            == protected_event_id
        ):
            # Acceptance terminalizes only the exact live assignment that it
            # names.  Comparing the captured event ID before popping prevents
            # a stale crossing from deleting a newer assignment of the same
            # swing; failed appends never reach this mutation.
            self._protected_swing_event_ids.pop(protected_swing_id)
        return event

    def _resolve_zone_crossing_if_due(
        self,
        zone: SupportResistanceState,
        *,
        asof: pd.Timestamp,
    ) -> MarketEvent | None:
        """Resolve a frozen S/R-zone penetration on its first later TF bar."""

        if zone.broken_at is None:
            return None
        penetration_event_id = self._penetration_event_ids.get(
            self._penetration_key(
                level_id=zone.zone_id,
                timeframe=zone.timeframe,
                crossed_at=zone.broken_at,
            )
        )
        if penetration_event_id is None:
            return None
        generation_id = self._crossing_generation_id(
            level_id=zone.zone_id,
            timeframe=zone.timeframe,
            crossed_at=zone.broken_at,
        )
        prior = self._terminal_crossing_events.get(generation_id)
        if prior is not None:
            return prior
        next_bar = next(
            (
                (clock, event_id)
                for clock, event_id in self._real_bar_event_ids_by_timeframe[
                    zone.timeframe
                ]
                if zone.broken_at < clock <= asof
            ),
            None,
        )
        if next_bar is None:
            return None
        resolved_at, resolution_bar_event_id = next_bar
        resolved_close = self._bar_close_by_event_id.get(
            resolution_bar_event_id
        )
        if resolved_close is None:
            raise ValueError(
                "zone crossing resolution lacks its frozen completed close"
            )
        accepted_outside = (
            resolved_close < zone.lower_bound
            if zone.side == "support"
            else resolved_close > zone.upper_bound
        )
        crossing_direction = (
            Direction.SHORT
            if zone.side == "support"
            else Direction.LONG
        )
        reaction_direction = (
            crossing_direction
            if accepted_outside
            else (
                Direction.LONG
                if crossing_direction is Direction.SHORT
                else Direction.SHORT
            )
        )
        return self._append_crossing_resolution(
            (
                EventKind.ACCEPTANCE_CONFIRMED
                if accepted_outside
                else EventKind.SWEEP_CONFIRMED
            ),
            resolved_at,
            zone.timeframe,
            "below" if zone.side == "support" else "above",
            zone.anchor_price,
            zone.strength,
            (penetration_event_id, resolution_bar_event_id),
            {
                "level_id": zone.zone_id,
                "source_kind": zone.source_kind,
                "resolution_bars": 1,
                "resolved_close": float(resolved_close),
                "resolution": (
                    "later_close_held_outside_frozen_zone"
                    if accepted_outside
                    else "later_close_returned_inside_frozen_zone"
                ),
            },
            direction=reaction_direction,
            crossed_at=zone.broken_at,
            zone=(zone.lower_bound, zone.upper_bound),
        )

    def _resolve_swing_crossing_if_due(
        self,
        swing,
        *,
        timeframe: Timeframe,
        asof: pd.Timestamp,
    ) -> MarketEvent | None:
        """Resolve a confirmed-swing price crossing on the next TF close."""

        if swing.broken_at is None:
            return None
        level_id = f"swing:{swing.swing_id}"
        penetration_event_id = self._penetration_event_ids.get(
            self._penetration_key(
                level_id=level_id,
                timeframe=timeframe,
                crossed_at=swing.broken_at,
            )
        )
        if penetration_event_id is None:
            return None
        generation_id = self._crossing_generation_id(
            level_id=level_id,
            timeframe=timeframe,
            crossed_at=swing.broken_at,
        )
        prior = self._terminal_crossing_events.get(generation_id)
        if prior is not None:
            return prior
        next_bar = next(
            (
                (clock, event_id)
                for clock, event_id in self._real_bar_event_ids_by_timeframe[
                    timeframe
                ]
                if swing.broken_at < clock <= asof
            ),
            None,
        )
        if next_bar is None:
            return None
        resolved_at, resolution_bar_event_id = next_bar
        resolved_close = self._bar_close_by_event_id.get(
            resolution_bar_event_id
        )
        if resolved_close is None:
            raise ValueError(
                "swing crossing resolution lacks its completed close"
            )
        crossed_above = swing.side.value == "high"
        accepted_outside = (
            resolved_close > swing.price
            if crossed_above
            else resolved_close < swing.price
        )
        crossing_direction = (
            Direction.LONG if crossed_above else Direction.SHORT
        )
        reaction_direction = (
            crossing_direction
            if accepted_outside
            else (
                Direction.SHORT
                if crossing_direction is Direction.LONG
                else Direction.LONG
            )
        )
        return self._append_crossing_resolution(
            (
                EventKind.ACCEPTANCE_CONFIRMED
                if accepted_outside
                else EventKind.SWEEP_CONFIRMED
            ),
            resolved_at,
            timeframe,
            "above" if crossed_above else "below",
            swing.price,
            clamp(swing.magnitude_atr),
            (penetration_event_id, resolution_bar_event_id),
            {
                "level_id": level_id,
                "source_kind": "confirmed_swing",
                "target_swing_id": swing.swing_id,
                "resolution_bars": 1,
                "resolved_close": float(resolved_close),
                "resolution": (
                    "later_close_held_outside_confirmed_swing_price"
                    if accepted_outside
                    else "later_close_returned_inside_confirmed_swing_price"
                ),
            },
            direction=reaction_direction,
            crossed_at=swing.broken_at,
            zone=(swing.price, swing.price),
        )

    def _append_completed_bar_event(
        self,
        candle: Candle,
        *,
        atr: float,
        data_complete: bool,
    ) -> MarketEvent:
        """Append one normalized data fact consumed by event reducers.

        This is deliberately not an SMC interpretation.  Its external data
        identity is carried in evidence so derived semantic events can later
        distinguish event lineage from raw candle/entity provenance.
        """

        data_id = hashlib.sha256(
            json.dumps(
                to_primitive(
                    {
                        "data_type": "completed_ohlcv_bar",
                        "timeframe": candle.timeframe,
                        "start": candle.start,
                        "end": candle.end,
                        "open": float(candle.open),
                        "high": float(candle.high),
                        "low": float(candle.low),
                        "close": float(candle.close),
                        "volume": float(candle.volume),
                        "symbol": candle.symbol,
                        "instrument_id": int(candle.instrument_id),
                        "complete": bool(candle.complete),
                        "observed_minutes": candle.observed_minutes,
                        "expected_minutes": candle.expected_minutes,
                        "real_minutes": candle.real_minutes,
                        "synthetic_minutes": candle.synthetic_minutes,
                    }
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        try:
            detector_candle_id = candle_identity(
                candle,
                tick_size=self.config.tick_size,
            )
        except ValueError:
            # Normalization must remain able to audit off-grid vendor/test
            # bars even though registered semantic detectors reject them.
            # Such a bar can be replayed by its exact data digest, but it can
            # never masquerade as a tick-grid detector candle identity.
            detector_candle_id = data_id
        session_name, session_phase = session_name_phase(candle.end)
        bar_evidence = {
            "event_category": "normalized_data",
            "source_data_ids": (data_id,),
            "detector_candle_id": detector_candle_id,
            # The enabled owner registry is part of the normalized M1 replay
            # contract.  One completed M1 fact can therefore initialize empty
            # higher-timeframe state without inventing a completed HTF bar or
            # consulting the rich FrameObservation projection.
            "active_timeframes": (
                tuple(
                    timeframe.value
                    for timeframe in self._active_timeframes
                )
                if candle.timeframe is Timeframe.M1
                else ()
            ),
            "scale_registry_id": self._scale_registry_id,
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(candle.volume),
            "atr": float(atr),
            "data_complete": bool(data_complete),
            "real_completed": bool(candle.real_completed),
            "clock_only": not candle.real_completed,
            "symbol": candle.symbol,
            "instrument_id": int(candle.instrument_id),
            "session_name": session_name,
            "session_phase": session_phase,
        }
        if not candle.real_completed:
            # Clock-only roots must expose the exact coverage defect they
            # transport.  Preserve historical real-root evidence/identities.
            bar_evidence.update(
                {
                    "complete": bool(candle.complete),
                    "start": candle.start,
                    "observed_minutes": int(candle.observed_minutes),
                    "expected_minutes": int(candle.expected_minutes),
                    "real_minutes": int(candle.real_minutes),
                    "synthetic_minutes": int(candle.synthetic_minutes),
                }
            )
        event = _event(
            EventKind.BAR_COMPLETED,
            candle.end,
            candle.timeframe,
            None,
            float(candle.close),
            0.0,
            (),
            bar_evidence,
            event_time=candle.end,
            known_at=candle.end,
            evidence=bar_evidence,
            source_data_ids=(data_id,),
            source_entity_ids=(
                f"scale_registry:{self._scale_registry_id}",
            ),
            origin=EventOrigin.NORMALIZED_DATA,
        )
        existing = self.audit_store.get(event.event_id)
        if existing is not None:
            self._bar_event_ids_by_candle_id[detector_candle_id] = (
                existing.event_id
            )
            self._bar_close_by_candle_id[detector_candle_id] = float(
                candle.close
            )
            self._bar_close_by_event_id[existing.event_id] = float(
                candle.close
            )
            if not any(
                event_id == existing.event_id
                for _, event_id in self._bar_event_ids_by_timeframe[
                    candle.timeframe
                ]
            ):
                self._bar_event_ids_by_timeframe[candle.timeframe].append(
                    (candle.end, existing.event_id)
                )
            if candle.real_completed and not any(
                event_id == existing.event_id
                for _, event_id in self._real_bar_event_ids_by_timeframe[
                    candle.timeframe
                ]
            ):
                self._real_bar_event_ids_by_timeframe[
                    candle.timeframe
                ].append((candle.end, existing.event_id))
            return existing
        self.memory.append(event, include_in_recent=False)
        self._bar_event_ids_by_candle_id[detector_candle_id] = event.event_id
        self._bar_close_by_candle_id[detector_candle_id] = float(candle.close)
        self._bar_close_by_event_id[event.event_id] = float(candle.close)
        self._bar_event_ids_by_timeframe[candle.timeframe].append(
            (candle.end, event.event_id)
        )
        if candle.real_completed:
            self._real_bar_event_ids_by_timeframe[
                candle.timeframe
            ].append((candle.end, event.event_id))
        return event

    def _append_available_bar_events(
        self,
        update: ReaderUpdate,
        histories: Mapping[Timeframe, Sequence[Candle]],
        frames: Mapping[Timeframe, FrameObservation],
    ) -> None:
        """Publish normalized bars before semantics that consume them.

        On the first attached snapshot the observer may receive a retained
        causal prefix rather than one bar.  We publish that prefix once and
        calculate each bar's ATR only from bars available through that bar;
        using the final frame ATR for old bars would itself leak the future
        into the normalized event stream.  Later updates publish only the
        newly completed bars and may reuse the frame's current causal ATR.
        """

        candidates: list[tuple[Candle, float, bool]] = []
        coverage_start = self.memory.clock_coverage_start
        for timeframe in self._active_timeframes:
            frame = frames[timeframe]
            known_bars = self._bar_event_ids_by_timeframe[timeframe]
            if known_bars:
                for candle in update.newly_completed.get(timeframe, ()):
                    if (
                        candle.complete
                        and (
                            coverage_start is None
                            or candle.end >= coverage_start
                        )
                    ):
                        candidates.append(
                            (
                                candle,
                                float(frame.metrics.get("atr", 0.0)),
                                bool(frame.ready),
                            )
                        )
                continue

            eligible_history = tuple(
                candle
                for candle in histories[timeframe]
                if (
                    candle.complete
                    and (
                        coverage_start is None
                        or candle.end >= coverage_start
                    )
                )
            )
            true_ranges: deque[float] = deque(
                maxlen=max(1, self.config.atr_period)
            )
            prior_close: float | None = None
            real_bars_seen = 0
            for candle in eligible_history:
                if candle.real_completed:
                    real_bars_seen += 1
                    true_range = (
                        float(candle.high - candle.low)
                        if prior_close is None
                        else max(
                            float(candle.high - candle.low),
                            abs(float(candle.high) - prior_close),
                            abs(float(candle.low) - prior_close),
                        )
                    )
                    true_ranges.append(max(0.0, true_range))
                    prior_close = float(candle.close)
                positive = tuple(value for value in true_ranges if value > 0.0)
                causal_atr = (
                    float(np.mean(positive)) if positive else 0.0
                )
                candidates.append(
                    (
                        candle,
                        causal_atr,
                        real_bars_seen
                        >= self.config.minimum_bars[timeframe],
                    )
                )

        for candle, atr, data_complete in sorted(
            candidates,
            key=lambda value: (
                value[0].end,
                value[0].timeframe.value,
                value[0].start,
            ),
        ):
            self._append_completed_bar_event(
                candle,
                atr=atr,
                data_complete=data_complete,
            )

    def _bar_event_id_for_candle_id(self, candle_id: str) -> str:
        try:
            return self._bar_event_ids_by_candle_id[candle_id]
        except KeyError as error:
            raise ValueError(
                "semantic source candle has no BAR_COMPLETED event: "
                f"{candle_id}"
            ) from error

    def _bar_event_id_at(
        self,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
    ) -> str:
        clock = pd.Timestamp(known_at)
        for event_clock, event_id in reversed(
            self._real_bar_event_ids_by_timeframe[timeframe]
        ):
            if event_clock == clock:
                return event_id
            if event_clock < clock:
                break
        raise ValueError(
            "semantic occurrence has no exact completed-bar source: "
            f"{timeframe.value}@{clock.isoformat()}"
        )

    def _clock_root_event_id_at(
        self,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
    ) -> str:
        """Return one exact normalized BAR root, including clock-only M1."""

        clock = pd.Timestamp(known_at)
        matches = tuple(
            event_id
            for event_clock, event_id in self._bar_event_ids_by_timeframe[
                timeframe
            ]
            if event_clock == clock
        )
        if len(matches) != 1:
            raise ValueError(
                "semantic context requires one exact inclusive clock root: "
                f"{timeframe.value}@{clock.isoformat()}"
            )
        event = self.memory.audit_event_including_pending(matches[0])
        if (
            event is None
            or event.origin is not EventOrigin.NORMALIZED_DATA
            or event.kind is not EventKind.BAR_COMPLETED
            or event.timeframe is not timeframe
            or event.event_time != clock
            or event.known_at != clock
        ):
            raise ValueError("inclusive clock root is not an exact normalized BAR")
        return event.event_id

    def _synthetic_m1_context_event_ids_for_m5_terminal(
        self,
        known_at: pd.Timestamp,
    ) -> tuple[str, ...]:
        """Return every clock-only M1 constituent of an incomplete M5 bar."""

        clock = pd.Timestamp(known_at)
        interval_start = clock - pd.Timedelta(minutes=5)
        current_root_id = self._clock_root_event_id_at(Timeframe.M1, clock)
        current_root = self.memory.audit_event_including_pending(
            current_root_id
        )
        if current_root is None:
            raise ValueError(
                "synthetic displacement terminal lacks its current M1 root"
            )
        market_identity = (
            current_root.evidence.get("symbol"),
            current_root.evidence.get("instrument_id"),
        )
        interval_roots: list[tuple[pd.Timestamp, str]] = []
        roots: list[tuple[pd.Timestamp, str]] = []
        for event_clock, event_id in self._bar_event_ids_by_timeframe[
            Timeframe.M1
        ]:
            if not interval_start < event_clock <= clock:
                continue
            event = self.memory.audit_event_including_pending(event_id)
            if (
                event is None
                or event.origin is not EventOrigin.NORMALIZED_DATA
                or event.kind is not EventKind.BAR_COMPLETED
                or event.timeframe is not Timeframe.M1
                or event.event_time != event_clock
                or event.known_at != event_clock
            ):
                raise ValueError(
                    "synthetic displacement M5 interval has an invalid M1 root"
                )
            real_completed = event.evidence.get("real_completed")
            clock_only = event.evidence.get("clock_only")
            if (
                not isinstance(real_completed, bool)
                or not isinstance(clock_only, bool)
                or clock_only is not (not real_completed)
                or (
                    event.evidence.get("symbol"),
                    event.evidence.get("instrument_id"),
                )
                != market_identity
            ):
                raise ValueError(
                    "synthetic displacement M5 interval has inconsistent M1 "
                    "root evidence"
                )
            interval_roots.append((event_clock, event.event_id))
            if clock_only:
                roots.append((event_clock, event.event_id))
        ordered_interval = tuple(sorted(interval_roots))
        expected_clocks = tuple(
            interval_start + pd.Timedelta(minutes=offset)
            for offset in range(1, 6)
        )
        if (
            len(ordered_interval) != 5
            or tuple(item[0] for item in ordered_interval) != expected_clocks
            or len({item[1] for item in ordered_interval}) != 5
        ):
            raise ValueError(
                "synthetic displacement M5 interval lacks five contiguous "
                "unique M1 roots"
            )
        ordered = tuple(sorted(roots))
        event_ids = tuple(item[1] for item in ordered)
        if (
            not ordered
            or len(ordered) != len(set(ordered))
            or len(event_ids) != len(set(event_ids))
        ):
            raise ValueError(
                "synthetic displacement terminal lacks exact clock-only M1 "
                "constituent roots"
            )
        clocks = tuple(item[0] for item in ordered)
        if len(clocks) != len(set(clocks)):
            raise ValueError(
                "synthetic displacement M5 interval repeats an M1 root clock"
            )
        return tuple(item[1] for item in ordered)

    def _swing_window_event_ids(self, swing) -> tuple[str, ...]:
        if self._structure_config is None:
            raise ValueError("confirmed swing lacks a structure protocol")
        span = self._structure_config.span_for(swing.timeframe)
        bar_events = self._real_bar_event_ids_by_timeframe[swing.timeframe]
        pivot_index = next(
            (
                index
                for index, (clock, _) in enumerate(bar_events)
                if clock == swing.pivot_end
            ),
            None,
        )
        if (
            pivot_index is None
            or pivot_index < span
            or pivot_index + span >= len(bar_events)
        ):
            raise ValueError(
                "confirmed swing lacks its full registered bar window"
            )
        window = tuple(
            event_id
            for _, event_id in bar_events[
                pivot_index - span : pivot_index + span + 1
            ]
        )
        if bar_events[pivot_index + span][0] != swing.confirmed_at:
            raise ValueError(
                "confirmed swing bar window and known_at disagree"
            )
        return window

    def _append_reference_zone_admission_prefixes(
        self,
        zone: SupportResistanceState,
        *,
        final_observed_at: pd.Timestamp,
    ) -> None:
        """Preserve same-admission reference-zone lifecycle order.

        A completed previous-session/day/week source can be admitted and then
        tested or broken by the first tradable bar before a public snapshot is
        built.  This is not a general cold-start backfill: it applies only to a
        reference S/R entity's first projection when its terminal/current
        transition occurred at this exact projection clock.  Prefix events use
        only frozen geometry and source identity, stay out of the recent deque,
        and therefore cannot carry later reaction or extreme information into
        an earlier clock.
        """

        if (
            zone.source_kind == "structural_swing"
            or zone.lifecycle is SupportResistanceLifecycle.ACTIVE
        ):
            return
        final_clocks = {
            SupportResistanceLifecycle.TESTED: zone.tested_at,
            SupportResistanceLifecycle.BROKEN: zone.broken_at,
            SupportResistanceLifecycle.REACCEPTED: zone.reaccepted_at,
            SupportResistanceLifecycle.RETIRED: zone.retired_at,
        }
        final_clock = final_clocks.get(zone.lifecycle)
        if final_clock is None or final_clock != final_observed_at:
            return
        entity_key = f"zone:{zone.zone_id}"
        if self.memory.timeline(entity_key):
            return
        frozen_details = {
            "causal_prefix_recovered": True,
            "lower_bound": zone.lower_bound,
            "upper_bound": zone.upper_bound,
            "source_kind": zone.source_kind,
            "source_ids": zone.source_ids,
            "range_id": zone.range_id,
            "source_zone_id": zone.source_zone_id,
            "zone_role": zone.zone_role,
            "structural_rank": zone.structural_rank,
            "is_protected_swing": zone.is_protected_swing,
        }

        def append_prefix(
            lifecycle: SupportResistanceLifecycle,
            observed_at: pd.Timestamp,
            transition_reason: str | None = None,
        ) -> None:
            self.memory.append(
                _event(
                    EventKind.SUPPORT_RESISTANCE_STATE,
                    observed_at,
                    zone.timeframe,
                    "below" if zone.side == "support" else "above",
                    zone.anchor_price,
                    0.0,
                    zone.causal_source_ids,
                    frozen_details,
                    entity_id=zone.zone_id,
                    lifecycle=lifecycle.value,
                    formed_at=zone.formed_at,
                    confirmed_at=zone.confirmed_at,
                    transition_reason=transition_reason,
                ),
                include_in_recent=False,
                audit=False,
            )

        append_prefix(
            SupportResistanceLifecycle.ACTIVE,
            zone.confirmed_at,
        )
        prior_clock = zone.confirmed_at
        broken_prefix_clock = (
            zone.broken_at
            if (
                zone.lifecycle
                in {
                    SupportResistanceLifecycle.REACCEPTED,
                    SupportResistanceLifecycle.RETIRED,
                }
                and zone.broken_at is not None
                and zone.broken_at < final_observed_at
            )
            else None
        )
        tested_upper_bound = broken_prefix_clock or final_observed_at
        if (
            zone.tested_at is not None
            and prior_clock < zone.tested_at < tested_upper_bound
        ):
            append_prefix(
                SupportResistanceLifecycle.TESTED,
                zone.tested_at,
            )
            prior_clock = zone.tested_at
        if broken_prefix_clock is not None and prior_clock < broken_prefix_clock:
            append_prefix(
                SupportResistanceLifecycle.BROKEN,
                broken_prefix_clock,
                "close_beyond_frozen_zone",
            )

    def _reference_zone_source_event_ids(
        self,
        zone: SupportResistanceState,
        *,
        observed_at: pd.Timestamp,
    ) -> tuple[str, ...]:
        """Bind one reference reaction band to its exact published level."""

        if zone.source_kind not in {
            "previous_session",
            "previous_day",
            "previous_week",
        }:
            raise ValueError(
                "reference-zone binding requires a registered reference family"
            )
        expected_side = "below" if zone.side == "support" else "above"
        expected_kind = (
            f"{zone.source_kind}_"
            f"{'low' if zone.side == 'support' else 'high'}"
        )
        matches = tuple(
            item
            for item in self._reference_inventory.values()
            if (
                item.source_ids == zone.source_ids
                and item.kind == expected_kind
                and item.side == expected_side
                and item.timeframe is zone.timeframe
                and float(item.price) == float(zone.anchor_price)
            )
        )
        if len(matches) != 1:
            raise ValueError(
                "reference support/resistance requires exactly one exact "
                "inventory source"
            )
        item = matches[0]
        event_id = self._candidate_level_event_ids.get(item.item_id)
        if event_id is None:
            raise ValueError(
                "reference support/resistance source level is not published"
            )
        event = self.memory.audit_event_including_pending(event_id)
        if (
            event is None
            or event.kind is not EventKind.LIQUIDITY_LEVEL_CREATED
            or event.origin is not EventOrigin.SEMANTIC_ATOMIC
            or event.timeframe is not item.timeframe
            or event.side != item.side
            or event.price is None
            or float(event.price) != float(item.price)
            or event.evidence.get("level_id") != item.item_id
            or event.evidence.get("candidate_only") is not True
            or event.evidence.get("source_kind") != item.kind
            or event.evidence.get("source_ids") != item.source_ids
            or event.source_entity_ids
            != (item.item_id, *item.source_ids)
            or event.known_at > pd.Timestamp(observed_at)
        ):
            raise ValueError(
                "reference support/resistance source level is not its exact "
                "published candidate"
            )
        return (event.event_id,)

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
        atomic_bar_roots_available = bool(
            self._real_bar_event_ids_by_timeframe[frame.timeframe]
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
                swing_state_event = _event(
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
                            "prominence_atr": swing.prominence_atr,
                            "confirmation_delay_bars": (
                                swing.confirmation_delay_bars
                            ),
                            "nesting_depth": swing.nesting_depth,
                            "semantic_rank": swing.semantic_rank.value,
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
                        event_time=(
                            swing.pivot_start
                            if swing.lifecycle
                            in {
                                SwingLifecycle.FORMING,
                                SwingLifecycle.CONFIRMED,
                            }
                            else observed_at
                        ),
                    )
                self.memory.append(swing_state_event)
                if (
                    swing.lifecycle
                    in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
                    and swing.confirmed_at is not None
                    and swing.swing_id
                    not in self._confirmed_swing_event_ids
                    and atomic_bar_roots_available
                ):
                    source_bar_events = self._swing_window_event_ids(
                        swing
                    )
                    canonical = self._append_semantic_atomic(
                        EventKind.SWING_CONFIRMED,
                        swing.confirmed_at,
                        frame.timeframe,
                        "above" if swing.side.value == "high" else "below",
                        swing.price,
                        clamp(swing.magnitude_atr),
                        source_bar_events,
                        {
                            "source_entity_id": swing.swing_id,
                            "side": swing.side.value,
                            "relation": swing.relation.value,
                            "pivot_start": swing.pivot_start.isoformat(),
                            "pivot_end": swing.pivot_end.isoformat(),
                            "prominence_atr": swing.prominence_atr,
                            "legacy_same_side_magnitude_atr": (
                                swing.magnitude_atr
                            ),
                            "confirmation_delay_bars": (
                                swing.confirmation_delay_bars
                            ),
                            "nesting_depth": swing.nesting_depth,
                            "semantic_rank": swing.semantic_rank.value,
                            "delta_ticks": swing.delta_ticks,
                            "confirmation_delay_minutes": int(
                                (
                                    swing.confirmed_at - swing.pivot_start
                                ).total_seconds()
                                // 60
                            ),
                        },
                        direction=direction,
                        event_time=swing.pivot_start,
                        source_entity_ids=(swing.swing_id,),
                        context_event_ids=(swing_state_event.event_id,),
                    )
                    self._confirmed_swing_event_ids[swing.swing_id] = (
                        canonical.event_id
                    )
                    swing_level_id = f"swing:{swing.swing_id}"
                    candidate = self._append_semantic_atomic(
                        EventKind.LIQUIDITY_LEVEL_CREATED,
                        swing.confirmed_at,
                        frame.timeframe,
                        "above" if swing.side.value == "high" else "below",
                        swing.price,
                        clamp(swing.magnitude_atr),
                        (canonical.event_id,),
                        {
                            "level_id": swing_level_id,
                            "candidate_only": True,
                            "source_kind": "confirmed_swing",
                            "source_swing_id": swing.swing_id,
                            "semantic_rank": swing.semantic_rank.value,
                        },
                        event_time=swing.pivot_start,
                        zone=(swing.price, swing.price),
                        source_entity_ids=(swing_level_id, swing.swing_id),
                    )
                    self._candidate_level_event_ids[swing_level_id] = (
                        candidate.event_id
                    )
                if (
                    swing.lifecycle is SwingLifecycle.BROKEN
                    and swing.broken_at is not None
                    and atomic_bar_roots_available
                ):
                    swing_level_id = f"swing:{swing.swing_id}"
                    candidate_event_id = self._candidate_level_event_ids.get(
                        swing_level_id
                    )
                    if candidate_event_id is None:
                        raise ValueError(
                            "broken swing lacks its candidate liquidity level"
                        )
                    penetration_key = self._penetration_key(
                        level_id=swing_level_id,
                        timeframe=frame.timeframe,
                        crossed_at=swing.broken_at,
                    )
                    if penetration_key not in self._penetration_event_ids:
                        break_bar_event_id = self._bar_event_id_at(
                            frame.timeframe,
                            swing.broken_at,
                        )
                        touch_event = self._append_semantic_atomic(
                            EventKind.LEVEL_TOUCHED,
                            swing.broken_at,
                            frame.timeframe,
                            (
                                "above"
                                if swing.side.value == "high"
                                else "below"
                            ),
                            swing.price,
                            clamp(swing.magnitude_atr),
                            (candidate_event_id, break_bar_event_id),
                            {
                                "level_id": swing_level_id,
                                "source_kind": "confirmed_swing",
                                "target_swing_id": swing.swing_id,
                                "touch_reason": "raw_swing_price_crossing",
                            },
                            event_time=swing.broken_at,
                            zone=(swing.price, swing.price),
                            source_entity_ids=(
                                swing_level_id,
                                swing.swing_id,
                            ),
                        )
                        crossing_generation_id = (
                            self._crossing_generation_id(
                                level_id=swing_level_id,
                                timeframe=frame.timeframe,
                                crossed_at=swing.broken_at,
                            )
                        )
                        break_bar_event = (
                            self.memory.audit_event_including_pending(
                                break_bar_event_id
                            )
                        )
                        if break_bar_event is None:
                            raise ValueError(
                                "swing penetration lacks its exact BAR"
                            )
                        penetration_price = float(
                            break_bar_event.evidence[
                                "high"
                                if swing.side.value == "high"
                                else "low"
                            ]
                        )
                        penetrated = self._append_semantic_atomic(
                            EventKind.LEVEL_PENETRATED,
                            swing.broken_at,
                            frame.timeframe,
                            (
                                "above"
                                if swing.side.value == "high"
                                else "below"
                            ),
                            penetration_price,
                            clamp(swing.magnitude_atr),
                            (
                                candidate_event_id,
                                touch_event.event_id,
                                break_bar_event_id,
                            ),
                            {
                                "level_id": swing_level_id,
                                "source_kind": "confirmed_swing",
                                "target_swing_id": swing.swing_id,
                                "penetration_standard": (
                                    "strict_close_beyond_confirmed_swing_"
                                    "price"
                                ),
                                "crossing_generation_id": (
                                    crossing_generation_id
                                ),
                                "crossed_at": (
                                    swing.broken_at.isoformat()
                                ),
                            },
                            direction=(
                                Direction.LONG
                                if swing.side.value == "high"
                                else Direction.SHORT
                            ),
                            event_time=swing.broken_at,
                            zone=(swing.price, swing.price),
                            source_entity_ids=(
                                swing_level_id,
                                swing.swing_id,
                            ),
                        )
                        self._penetration_event_ids[penetration_key] = (
                            penetrated.event_id
                        )
            if (
                swing.lifecycle is SwingLifecycle.BROKEN
                and swing.broken_at is not None
                and atomic_bar_roots_available
            ):
                # Lifecycle projection is deduplicated above, but a crossing
                # remains pending until the first *later* native-timeframe
                # close.  Revisit it on subsequent frame updates.
                self._resolve_swing_crossing_if_due(
                    swing,
                    timeframe=frame.timeframe,
                    asof=event_clock,
                )
        for leg in frame.structural_legs:
            if not self._remember_bounded(
                leg.leg_id,
                known=self._known_structural_leg_ids,
                order=self._known_structural_leg_order,
            ):
                continue
            source_event_ids = tuple(
                event_id
                for swing_id in leg.source_swing_ids
                if (
                    event_id := self._confirmed_swing_event_ids.get(
                        swing_id
                    )
                )
            )
            if len(source_event_ids) != 2:
                if not atomic_bar_roots_available:
                    continue
                raise ValueError(
                    "structural leg requires exactly two confirmed-swing "
                    "source events"
                )
            foundation_evidence: dict[str, object] = {}
            foundation_context_event_ids: tuple[str, ...] = ()
            foundation_source_data_ids: tuple[str, ...] = ()
            if leg.foundation_version is not None:
                foundation_source_data_ids = (
                    *leg.atr_source_candle_ids,
                    *leg.path_candle_ids,
                )
                foundation_context_event_ids = tuple(
                    self._bar_event_id_for_candle_id(candle_id)
                    for candle_id in foundation_source_data_ids
                )
                bound_bars = tuple(
                    self.memory.audit_event_including_pending(event_id)
                    for event_id in foundation_context_event_ids
                )
                if any(event is None for event in bound_bars):
                    raise ValueError(
                        "foundation structural leg lacks canonical BAR ancestry"
                    )
                first_bar = bound_bars[0]
                foundation_evidence = {
                    "foundation_version": leg.foundation_version,
                    "amplitude_ticks": leg.amplitude_ticks,
                    "atr_at_leg_start": leg.atr_at_leg_start,
                    "atr_source_candle_ids": leg.atr_source_candle_ids,
                    "duration_seconds": leg.duration_seconds,
                    "close_efficiency": leg.close_efficiency,
                    "extreme_path_efficiency": (
                        leg.extreme_path_efficiency
                    ),
                    "close_mae_points": leg.close_mae_points,
                    "close_mae_atr": leg.close_mae_atr,
                    "wick_mae_points": leg.wick_mae_points,
                    "wick_mae_atr": leg.wick_mae_atr,
                    "path_candle_ids": leg.path_candle_ids,
                    "tick_size": self.config.tick_size,
                    "symbol": first_bar.evidence.get("symbol"),
                    "instrument_id": first_bar.evidence.get(
                        "instrument_id"
                    ),
                }
            canonical_leg = self._append_semantic_atomic(
                EventKind.STRUCTURAL_LEG_CREATED,
                leg.known_at,
                frame.timeframe,
                "above" if leg.direction is Direction.LONG else "below",
                leg.end_price,
                clamp(leg.efficiency),
                source_event_ids,
                {
                    "leg_id": leg.leg_id,
                    "start_swing_id": leg.start_swing_id,
                    "end_swing_id": leg.end_swing_id,
                    "start_event_time": leg.start_event_time.isoformat(),
                    "end_event_time": leg.end_event_time.isoformat(),
                    "start_price": leg.start_price,
                    "end_price": leg.end_price,
                    "start_close": leg.start_close,
                    "end_close": leg.end_close,
                    "amplitude_points": leg.amplitude_points,
                    "amplitude_atr": leg.amplitude_atr,
                    "duration_bars": leg.duration_bars,
                    "duration_minutes": leg.duration_minutes,
                    "efficiency": leg.efficiency,
                    "max_retracement_points": (
                        leg.max_retracement_points
                    ),
                    "max_retracement_atr": leg.max_retracement_atr,
                    "rank": leg.rank.value,
                    **foundation_evidence,
                },
                direction=leg.direction,
                event_time=leg.end_event_time,
                source_entity_ids=(leg.leg_id, *leg.source_swing_ids),
                source_data_ids=foundation_source_data_ids,
                context_event_ids=foundation_context_event_ids,
            )
            self._structural_leg_event_ids[leg.leg_id] = (
                canonical_leg.event_id
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
            structure_state_event = _event(
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
            self.memory.append(structure_state_event)
            if (
                state.lifecycle is StructureLifecycle.CONFIRMED
                and state.confirmed_at is not None
                and atomic_bar_roots_available
            ):
                # ``latest_*`` belongs to the current snapshot and may point
                # to a swing confirmed after this structure generation.  A
                # retrospective first attach must reconstruct parents from
                # facts that were actually knowable at ``confirmed_at``.
                causal_swings = tuple(
                    max(
                        (
                            swing
                            for swing in frame.swings
                            if (
                                swing.side.value == side
                                and swing.confirmed_at is not None
                                and swing.confirmed_at <= state.confirmed_at
                                and swing.swing_id
                                in self._confirmed_swing_event_ids
                            )
                        ),
                        key=lambda swing: (
                            swing.confirmed_at,
                            swing.pivot_start,
                            swing.swing_id,
                        ),
                    )
                    for side in ("high", "low")
                )
                source_swing_events = tuple(
                    self._confirmed_swing_event_ids[swing.swing_id]
                    for swing in causal_swings
                )
                if len(source_swing_events) != 2:
                    raise ValueError(
                        "confirmed structure requires its exact high and low "
                        "swing events"
                    )
                structure_direction_event = self._append_semantic_atomic(
                    EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                    state.confirmed_at,
                    frame.timeframe,
                    (
                        "above"
                        if state.direction is Direction.LONG
                        else "below"
                    ),
                    state.protected_price,
                    clamp(state.cumulative_magnitude_atr),
                    source_swing_events,
                    {
                        "structure_id": state.structure_id,
                        "direction": state.direction.value,
                        "source_high_id": causal_swings[0].swing_id,
                        "source_low_id": causal_swings[1].swing_id,
                        "sequence_count": state.sequence_count,
                        "candidate_protected_swing_id": (
                            state.protected_swing_id
                        ),
                    },
                    direction=state.direction,
                    event_time=state.formed_at,
                    source_entity_ids=(
                        state.structure_id,
                        causal_swings[0].swing_id,
                        causal_swings[1].swing_id,
                    ),
                    context_event_ids=(structure_state_event.event_id,),
                )
                self._structure_direction_event_ids[state.structure_id] = (
                    structure_direction_event.event_id
                )
                self._latest_structure_direction_event_ids[
                    frame.timeframe
                ] = structure_direction_event.event_id
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
                self._resolve_zone_crossing_if_due(
                    zone,
                    asof=event_clock,
                )
                continue
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
            if prior_revision is None and self._prior is not None:
                # Completed reference-period levels are admitted only when
                # the next period is observed. Their source geometry was
                # complete earlier, but the semantic level first becomes
                # available at this admission clock.
                observed_at = max(observed_at, event_clock)
            availability_floor = (
                observed_at
                if prior_revision is None
                else zone.confirmed_at
            )
            if not lifecycle_revision:
                observed_at = max(
                    observed_at,
                    zone.metadata_observed_at,
                )
            if prior_revision is None:
                self._append_reference_zone_admission_prefixes(
                    zone,
                    final_observed_at=observed_at,
                )
            self._liquidity_entity_revisions[zone.zone_id] = revision
            zone_state_event = _event(
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
            self.memory.append(zone_state_event)
            if zone.zone_id not in self._candidate_level_event_ids:
                if zone.source_kind in {
                    "previous_session",
                    "previous_day",
                    "previous_week",
                }:
                    source_level_events = (
                        self._reference_zone_source_event_ids(
                            zone,
                            observed_at=observed_at,
                        )
                    )
                else:
                    source_level_events = tuple(
                        event_id
                        for source_id in zone.causal_source_ids
                        if (
                            event_id
                            := self._confirmed_swing_event_ids.get(source_id)
                        )
                    )
                created = self._append_semantic_atomic(
                    EventKind.LIQUIDITY_LEVEL_CREATED,
                    observed_at,
                    frame.timeframe,
                    "below" if zone.side == "support" else "above",
                    zone.anchor_price,
                    zone.strength,
                    source_level_events,
                    {
                        "level_id": zone.zone_id,
                        "candidate_only": True,
                        "source_kind": zone.source_kind,
                        "source_ids": zone.source_ids,
                        "structural_rank": zone.structural_rank,
                        "is_protected_swing": zone.is_protected_swing,
                    },
                    event_time=zone.formed_at,
                    zone=(zone.lower_bound, zone.upper_bound),
                    source_entity_ids=(zone.zone_id, *zone.causal_source_ids),
                    context_event_ids=(zone_state_event.event_id,),
                )
                self._candidate_level_event_ids[zone.zone_id] = (
                    created.event_id
                )
            candidate_event_id = self._candidate_level_event_ids[
                zone.zone_id
            ]
            for touch_ordinal, touch_at in enumerate(
                zone.touch_times,
                start=1,
            ):
                if (
                    touch_at <= zone.confirmed_at
                    or touch_at < availability_floor
                ):
                    continue
                touch_identity = (
                    f"{zone.zone_id}|{pd.Timestamp(touch_at).isoformat()}"
                )
                if not self._remember_bounded(
                    touch_identity,
                    known=self._known_level_touch_ids,
                    order=self._known_level_touch_order,
                ):
                    continue
                if not atomic_bar_roots_available:
                    continue
                bar_event_id = self._bar_event_id_at(
                    frame.timeframe,
                    touch_at,
                )
                touch_event = self._append_semantic_atomic(
                    EventKind.LEVEL_TOUCHED,
                    touch_at,
                    frame.timeframe,
                    "below" if zone.side == "support" else "above",
                    zone.anchor_price,
                    zone.strength,
                    (candidate_event_id, bar_event_id),
                    {
                        "level_id": zone.zone_id,
                        "touch_ordinal": touch_ordinal,
                        "source_kind": zone.source_kind,
                    },
                    event_time=touch_at,
                    zone=(zone.lower_bound, zone.upper_bound),
                )
                self._level_touch_event_ids[
                    (zone.zone_id, pd.Timestamp(touch_at))
                ] = touch_event.event_id
            if (
                zone.broken_at is not None
                and self._penetration_key(
                    level_id=zone.zone_id,
                    timeframe=frame.timeframe,
                    crossed_at=zone.broken_at,
                ) not in self._penetration_event_ids
            ):
                if not atomic_bar_roots_available:
                    continue
                penetration_known_at = max(
                    zone.broken_at,
                    availability_floor,
                )
                bar_event_id = self._bar_event_id_at(
                    frame.timeframe,
                    zone.broken_at,
                )
                touch_key = (zone.zone_id, pd.Timestamp(zone.broken_at))
                touch_event_id = self._level_touch_event_ids.get(touch_key)
                if touch_event_id is None:
                    touch_event = self._append_semantic_atomic(
                        EventKind.LEVEL_TOUCHED,
                        penetration_known_at,
                        frame.timeframe,
                        "below" if zone.side == "support" else "above",
                        zone.anchor_price,
                        zone.strength,
                        (candidate_event_id, bar_event_id),
                        {
                            "level_id": zone.zone_id,
                            "touch_ordinal": zone.total_touch_count + 1,
                            "source_kind": zone.source_kind,
                            "touch_reason": "boundary_crossing",
                        },
                        event_time=zone.broken_at,
                        zone=(zone.lower_bound, zone.upper_bound),
                    )
                    touch_event_id = touch_event.event_id
                    self._level_touch_event_ids[touch_key] = touch_event_id
                crossing_generation_id = self._crossing_generation_id(
                    level_id=zone.zone_id,
                    timeframe=frame.timeframe,
                    crossed_at=zone.broken_at,
                )
                penetration_bar = self.memory.audit_event_including_pending(
                    bar_event_id
                )
                if penetration_bar is None:
                    raise ValueError(
                        "zone penetration lacks its exact canonical BAR"
                    )
                penetration_price = float(
                    penetration_bar.evidence[
                        "low" if zone.side == "support" else "high"
                    ]
                )
                penetrated = self._append_semantic_atomic(
                    EventKind.LEVEL_PENETRATED,
                    penetration_known_at,
                    frame.timeframe,
                    "below" if zone.side == "support" else "above",
                    penetration_price,
                    zone.strength,
                    (candidate_event_id, touch_event_id, bar_event_id),
                    {
                        "level_id": zone.zone_id,
                        "source_kind": zone.source_kind,
                        "penetration_standard": "close_beyond_frozen_zone",
                        "crossing_generation_id": crossing_generation_id,
                        "crossed_at": zone.broken_at.isoformat(),
                    },
                    direction=(
                        Direction.SHORT
                        if zone.side == "support"
                        else Direction.LONG
                    ),
                    event_time=zone.broken_at,
                    zone=(zone.lower_bound, zone.upper_bound),
                )
                self._penetration_event_ids[
                    self._penetration_key(
                        level_id=zone.zone_id,
                        timeframe=frame.timeframe,
                        crossed_at=zone.broken_at,
                    )
                ] = penetrated.event_id
            self._resolve_zone_crossing_if_due(
                zone,
                asof=event_clock,
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
            pool_observed_at = (
                max(pool.confirmed_at, event_clock)
                if prior_revision is None and self._prior is not None
                else (
                    pool.confirmed_at
                    if lifecycle_revision
                    else pool.touch_times[-1]
                )
            )
            pool_state_event = _event(
                    EventKind.LIQUIDITY_POOL_STATE,
                    pool_observed_at,
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
            self.memory.append(pool_state_event)
            if entity_id not in self._candidate_level_event_ids:
                source_swing_events = tuple(
                    event_id
                    for source_id in pool.member_swing_ids
                    if (
                        event_id
                        := self._confirmed_swing_event_ids.get(source_id)
                    )
                )
                created = self._append_semantic_atomic(
                    EventKind.LIQUIDITY_LEVEL_CREATED,
                    pool_observed_at,
                    frame.timeframe,
                    pool.side,
                    pool.midpoint,
                    pool.strength,
                    source_swing_events,
                    {
                        "level_id": entity_id,
                        "candidate_only": True,
                        "source_kind": "formed_liquidity_pool",
                        "member_swing_ids": pool.member_swing_ids,
                    },
                    event_time=pool.formed_at,
                    zone=(pool.lower_bound, pool.upper_bound),
                    source_entity_ids=(entity_id, *pool.member_swing_ids),
                    context_event_ids=(pool_state_event.event_id,),
                )
                self._candidate_level_event_ids[entity_id] = (
                    created.event_id
                )
            candidate_event_id = self._candidate_level_event_ids[
                entity_id
            ]
            for touch_ordinal, touch_at in enumerate(
                pool.touch_times,
                start=1,
            ):
                if touch_at <= pool.confirmed_at:
                    continue
                touch_identity = (
                    f"{entity_id}|{pd.Timestamp(touch_at).isoformat()}"
                )
                if not self._remember_bounded(
                    touch_identity,
                    known=self._known_level_touch_ids,
                    order=self._known_level_touch_order,
                ):
                    continue
                if not atomic_bar_roots_available:
                    continue
                bar_event_id = self._bar_event_id_at(
                    frame.timeframe,
                    touch_at,
                )
                touch_event = self._append_semantic_atomic(
                    EventKind.LEVEL_TOUCHED,
                    touch_at,
                    frame.timeframe,
                    pool.side,
                    pool.midpoint,
                    pool.strength,
                    (candidate_event_id, bar_event_id),
                    {
                        "level_id": entity_id,
                        "touch_ordinal": touch_ordinal,
                        "source_kind": "formed_liquidity_pool",
                    },
                    event_time=touch_at,
                    zone=(pool.lower_bound, pool.upper_bound),
                )
                self._level_touch_event_ids[
                    (entity_id, pd.Timestamp(touch_at))
                ] = touch_event.event_id
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
                bos_state_event = _event(
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
                )
                self.memory.append(bos_state_event)
                if (
                    item.lifecycle is BOSLifecycle.CONFIRMED
                    and item.resolved_at is not None
                    and atomic_bar_roots_available
                ):
                    target_event_id = (
                        self._confirmed_swing_event_ids.get(
                            item.target_swing_id
                        )
                    )
                    if target_event_id is None or item.break_bar_id is None:
                        raise ValueError(
                            "raw boundary break lacks its target swing or "
                            "break-bar identity"
                        )
                    break_bar_event_id = self._bar_event_id_for_candle_id(
                        item.break_bar_id
                    )
                    break_close = self._bar_close_by_candle_id[
                        item.break_bar_id
                    ]
                    raw_break = self._append_semantic_atomic(
                        EventKind.RAW_BOUNDARY_BREAK,
                        item.resolved_at,
                        frame.timeframe,
                        (
                            "above"
                            if item.direction is Direction.LONG
                            else "below"
                        ),
                        item.target_price,
                        item.strength,
                        (target_event_id, break_bar_event_id),
                        {
                            "bos_id": item.bos_id,
                            "target_swing_id": item.target_swing_id,
                            "scope": item.scope.value,
                            "break_bar_id": item.break_bar_id,
                            "break_distance_atr": item.break_distance_atr,
                            "break_close": break_close,
                            "break_buffer_ticks": 0,
                            "comparison": "strict_close_beyond",
                            "break_standard": (
                                "close_beyond_confirmed_boundary"
                            ),
                            "source_displacement_id": (
                                item.source_displacement_id
                            ),
                        },
                        direction=item.direction,
                        event_time=item.resolved_at,
                        source_data_ids=(item.break_bar_id,),
                        source_entity_ids=(
                            item.bos_id,
                            item.target_swing_id,
                            *((
                                item.source_structure_id,
                            ) if item.source_structure_id else ()),
                            *((
                                item.source_displacement_id,
                            ) if item.source_displacement_id else ()),
                        ),
                        context_event_ids=(
                            bos_state_event.event_id,
                            *((
                                self._displacement_event_ids[
                                    item.source_displacement_id
                                ],
                            ) if (
                                item.source_displacement_id
                                in self._displacement_event_ids
                            ) else ()),
                        ),
                    )
                    self._raw_break_event_ids[item.bos_id] = (
                        raw_break.event_id
                    )
                    live_protected_assignments = ()
                    continuation_opposes_live_protection = False
                    if item.scope is BOSScope.CONTINUATION:
                        live_protected_assignments = (
                            self._live_protected_assignments(
                                frame.timeframe,
                                asof=item.resolved_at,
                            )
                        )
                        live_protected_event = (
                            live_protected_assignments[0][2]
                            if live_protected_assignments
                            else None
                        )
                        continuation_opposes_live_protection = bool(
                            live_protected_event is not None
                            and live_protected_event.direction
                            is not item.direction
                        )
                    if (
                        continuation_opposes_live_protection
                        and frame.timeframe is Timeframe.M5
                        and item.source_structure_id is not None
                        and item.break_bar_id is not None
                    ):
                        if len(live_protected_assignments) != 1:
                            raise ValueError(
                                "raw-only continuation suppression lacks one "
                                "exact protected-assignment witness"
                            )
                        disposition = Group3RawOnlyStructureDisposition(
                            bos_id=item.bos_id,
                            raw_break_event_id=raw_break.event_id,
                            protected_assignment_event_id=(
                                live_protected_assignments[0][1]
                            ),
                            timeframe=frame.timeframe,
                            direction=item.direction,
                            resolved_at=item.resolved_at,
                            source_structure_id=item.source_structure_id,
                            target_swing_id=item.target_swing_id,
                            break_bar_id=item.break_bar_id,
                            bos_source_displacement_id=(
                                item.source_displacement_id
                            ),
                        )
                        prior_disposition = (
                            self._raw_only_structure_dispositions.get(
                                item.bos_id
                            )
                        )
                        if (
                            prior_disposition is not None
                            and prior_disposition != disposition
                        ):
                            raise ValueError(
                                "raw-only continuation disposition drifted"
                            )
                        self._raw_only_structure_dispositions[
                            item.bos_id
                        ] = disposition
                    if (
                        item.scope is BOSScope.CONTINUATION
                        and not continuation_opposes_live_protection
                    ):
                        structure_direction_event_id = (
                            self._structure_direction_event_ids.get(
                                item.source_structure_id
                            )
                            if item.source_structure_id is not None
                            else None
                        )
                        if structure_direction_event_id is None:
                            raise ValueError(
                                "qualified BOS lacks the exact prior "
                                "structure-direction event"
                            )
                        qualified_bos = self._append_semantic_atomic(
                            EventKind.QUALIFIED_BOS,
                            item.resolved_at,
                            frame.timeframe,
                            (
                                "above"
                                if item.direction is Direction.LONG
                                else "below"
                            ),
                            item.target_price,
                            item.strength,
                            (
                                raw_break.event_id,
                                structure_direction_event_id,
                            ),
                            {
                                "bos_id": item.bos_id,
                                "scope": item.scope.value,
                                "qualification": (
                                    "aligned_with_confirmed_structure"
                                ),
                                "displacement_context_present": bool(
                                    item.source_displacement_id
                                ),
                            },
                            direction=item.direction,
                            event_time=item.resolved_at,
                            source_entity_ids=(
                                item.bos_id,
                                item.source_structure_id,
                            ),
                        )
                        self._qualified_structure_event_ids[
                            item.bos_id
                        ] = qualified_bos.event_id
                        origin_leg = next(
                            (
                                leg
                                for leg in reversed(frame.structural_legs)
                                if (
                                    leg.end_swing_id
                                    == item.target_swing_id
                                    and leg.direction is item.direction
                                )
                            ),
                            None,
                        )
                        if origin_leg is not None:
                            origin_swing = next(
                                (
                                    swing
                                    for swing in frame.swings
                                    if swing.swing_id
                                    == origin_leg.start_swing_id
                                ),
                                None,
                            )
                            if origin_swing is not None:
                                protected_source = (
                                    self._confirmed_swing_event_ids.get(
                                        origin_swing.swing_id
                                    )
                                )
                                origin_leg_event_id = (
                                    self._structural_leg_event_ids.get(
                                        origin_leg.leg_id
                                    )
                                )
                                if (
                                    protected_source is None
                                    or origin_leg_event_id is None
                                ):
                                    raise ValueError(
                                        "protected swing lacks its exact "
                                        "swing or structural-leg event"
                                    )
                                live_protected_event = (
                                    live_protected_assignments[0][2]
                                    if live_protected_assignments
                                    else None
                                )
                                protection_is_monotonic = bool(
                                    live_protected_event is None
                                    or (
                                        item.direction is Direction.LONG
                                        and origin_swing.price
                                        >= float(live_protected_event.price)
                                    )
                                    or (
                                        item.direction is Direction.SHORT
                                        and origin_swing.price
                                        <= float(live_protected_event.price)
                                    )
                                )
                                if protection_is_monotonic:
                                    protected_event = (
                                        self._append_semantic_atomic(
                                            EventKind.PROTECTED_SWING_ASSIGNED,
                                            item.resolved_at,
                                            frame.timeframe,
                                            (
                                                "below"
                                                if item.direction
                                                is Direction.LONG
                                                else "above"
                                            ),
                                            origin_swing.price,
                                            clamp(origin_leg.efficiency),
                                            (
                                                qualified_bos.event_id,
                                                origin_leg_event_id,
                                                protected_source,
                                            ),
                                            {
                                                "bos_id": item.bos_id,
                                                "structure_id": (
                                                    item.source_structure_id
                                                ),
                                                "origin_leg_id": (
                                                    origin_leg.leg_id
                                                ),
                                                "protected_swing_id": (
                                                    origin_swing.swing_id
                                                ),
                                                "break_standard": (
                                                    "later_acceptance_beyond"
                                                ),
                                            },
                                            direction=item.direction,
                                            event_time=(
                                                origin_swing.pivot_start
                                            ),
                                            source_entity_ids=(
                                                item.bos_id,
                                                item.source_structure_id,
                                                origin_leg.leg_id,
                                                origin_swing.swing_id,
                                            ),
                                        )
                                    )
                                    self._replace_live_protected_assignment(
                                        protected_event,
                                        live_protected_assignments,
                                    )
                    elif item.scope is BOSScope.OPPOSED:
                        structure_direction_event_id = (
                            self._structure_direction_event_ids.get(
                                item.source_structure_id
                            )
                            if item.source_structure_id is not None
                            else None
                        )
                        if structure_direction_event_id is None:
                            raise ValueError(
                                "MSS Core lacks the exact prior "
                                "structure-direction event"
                            )
                        mss_core = self._append_semantic_atomic(
                            EventKind.MSS_CORE_CONFIRMED,
                            item.resolved_at,
                            frame.timeframe,
                            (
                                "above"
                                if item.direction is Direction.LONG
                                else "below"
                            ),
                            item.target_price,
                            item.strength,
                            (
                                raw_break.event_id,
                                structure_direction_event_id,
                            ),
                            {
                                "bos_id": item.bos_id,
                                "scope": item.scope.value,
                                "core_definition": (
                                    "first_opposed_confirmed_boundary_break"
                                ),
                                "prior_sweep": None,
                                "displacement_context_present": bool(
                                    item.source_displacement_id
                                ),
                                "legacy_mss_qualified_context": (
                                    item.mss_qualified
                                ),
                            },
                            direction=item.direction,
                            event_time=item.resolved_at,
                            source_entity_ids=(
                                item.bos_id,
                                item.source_structure_id,
                            ),
                        )
                        self._qualified_structure_event_ids[
                            item.bos_id
                        ] = mss_core.event_id
            post_break_at = item.accepted_at or item.rejected_at
            if (
                item.lifecycle is BOSLifecycle.CONFIRMED
                and post_break_at is not None
                and item.post_break_state is not None
            ):
                self.memory.append(
                    post_break_event := _event(
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
                raw_event_id = self._raw_break_event_ids.get(item.bos_id)
                if raw_event_id is not None:
                    crossing_level_id = f"swing:{item.target_swing_id}"
                    if (
                        crossing_level_id
                        not in self._candidate_level_event_ids
                    ):
                        raise ValueError(
                            "BOS post-break resolution lacks its raw-swing "
                            "candidate-level identity"
                        )
                    penetration_event_id = self._penetration_event_ids.get(
                        self._penetration_key(
                            level_id=crossing_level_id,
                            timeframe=frame.timeframe,
                            crossed_at=item.resolved_at,
                        )
                    )
                    if penetration_event_id is None:
                        raise ValueError(
                            "BOS post-break resolution lacks its canonical "
                            "penetration event"
                        )
                    accepted = item.accepted_at is not None
                    resolution_bar_event_id = self._bar_event_id_at(
                        frame.timeframe,
                        post_break_at,
                    )
                    reaction_direction = (
                        item.direction
                        if accepted
                        else (
                            Direction.SHORT
                            if item.direction is Direction.LONG
                            else Direction.LONG
                        )
                    )
                    self._append_crossing_resolution(
                        (
                            EventKind.ACCEPTANCE_CONFIRMED
                            if accepted
                            else EventKind.SWEEP_CONFIRMED
                        ),
                        post_break_at,
                        frame.timeframe,
                        (
                            "above"
                            if item.direction is Direction.LONG
                            else "below"
                        ),
                        item.target_price,
                        item.strength,
                        (
                            penetration_event_id,
                            resolution_bar_event_id,
                        ),
                        {
                            "bos_id": item.bos_id,
                            "level_id": crossing_level_id,
                            "target_swing_id": item.target_swing_id,
                            "resolution_bars": 1,
                            "resolution": (
                                "held_outside"
                                if accepted
                                else "returned_inside"
                            ),
                        },
                        direction=reaction_direction,
                        crossed_at=item.resolved_at,
                        context_event_ids=(
                            raw_event_id,
                            post_break_event.event_id,
                        ),
                    )

    def _group3_foundation_source_event(
        self,
        event_id: str,
        *,
        expected_kinds: frozenset[EventKind],
    ) -> MarketEvent:
        event = self.memory.audit_event_including_pending(event_id)
        if event is None or event.kind not in expected_kinds:
            raise ValueError(
                "Group 3 foundation source is absent or has the wrong kind"
            )
        if event.kind is EventKind.BAR_COMPLETED:
            if (
                event.origin is not EventOrigin.NORMALIZED_DATA
                or event.evidence.get("real_completed") is not True
                or event.evidence.get("clock_only") is not False
            ):
                raise ValueError(
                    "Group 3 foundation BAR source is not a real normalized BAR"
                )
        elif not event.is_canonical_semantic:
            raise ValueError(
                "Group 3 foundation source is not canonical semantic data"
            )
        return event

    def _validate_group3_foundation_projection(
        self,
        update: Group3Update,
    ) -> None:
        cores_by_id = {
            core.core_id: core for core in update.base_origin_cores
        }
        qualified_by_id = {
            qualified.qualified_ob_id: qualified
            for qualified in update.qualified_order_blocks
        }
        for core in update.base_origin_cores:
            displacement = self._group3_foundation_source_event(
                core.source_displacement_event_id,
                expected_kinds=frozenset(
                    {EventKind.DISPLACEMENT_OBSERVED}
                ),
            )
            if (
                displacement.evidence.get("displacement_id")
                != core.source_displacement_id
            ):
                raise ValueError(
                    "Base Origin Core displacement event is not exact"
                )
            for candle_id, bar_event_id in zip(
                core.anchor_candle_ids,
                core.anchor_bar_event_ids,
            ):
                bar = self._group3_foundation_source_event(
                    bar_event_id,
                    expected_kinds=frozenset(
                        {EventKind.BAR_COMPLETED}
                    ),
                )
                if bar.evidence.get("detector_candle_id") != candle_id:
                    raise ValueError(
                        "Base Origin Core BAR event is not exact"
                    )
        for qualified in update.qualified_order_blocks:
            core = cores_by_id.get(qualified.base_origin_core_id)
            if core is None:
                raise ValueError(
                    "Qualified OB lost its exact Base Origin Core"
                )
            displacement = self._group3_foundation_source_event(
                qualified.source_displacement_event_id,
                expected_kinds=frozenset(
                    {EventKind.DISPLACEMENT_OBSERVED}
                ),
            )
            if (
                displacement.evidence.get("displacement_id")
                != qualified.source_displacement_id
                or qualified.source_displacement_event_id
                != core.source_displacement_event_id
            ):
                raise ValueError(
                    "Qualified OB displacement event is not exact"
                )
            expected_kind = (
                EventKind.QUALIFIED_BOS
                if qualified.compatible_structure_kind
                is CompatibleStructureKind.QUALIFIED_BOS
                else EventKind.MSS_CORE_CONFIRMED
            )
            self._group3_foundation_source_event(
                qualified.compatible_structure_event_id,
                expected_kinds=frozenset({expected_kind}),
            )
        for lifecycle in update.fvg_structural_lifecycles:
            creation = self._group3_foundation_source_event(
                lifecycle.source_creation_event_id,
                expected_kinds=frozenset({EventKind.FVG_CREATED}),
            )
            if creation.evidence.get("fvg_id") != lifecycle.fvg_id:
                raise ValueError("FVG foundation creation event is not exact")
            for context_event_id in lifecycle.context_source_event_ids:
                self._group3_foundation_source_event(
                    context_event_id,
                    expected_kinds=frozenset(
                        {
                            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                            EventKind.SWING_CONFIRMED,
                        }
                    ),
                )
            if lifecycle.terminal_reason is not None:
                if (
                    lifecycle.terminal_event_id is None
                    or lifecycle.terminal_event_id
                    not in lifecycle.terminal_source_event_ids
                    or tuple(
                        lifecycle.terminal_source_event_ids[
                            : 1
                            + len(lifecycle.context_source_event_ids)
                        ]
                    )
                    != (
                        lifecycle.source_creation_event_id,
                        *lifecycle.context_source_event_ids,
                    )
                ):
                    raise ValueError(
                        "FVG terminal fact is absent from exact ancestry"
                    )
                if (
                    lifecycle.terminal_reason
                    is FVGTerminationCause.CLOSE_THROUGH_FAR_EDGE
                ):
                    source_kind = frozenset({EventKind.BAR_COMPLETED})
                elif (
                    lifecycle.terminal_reason
                    is FVGTerminationCause.DATA_GAP
                ):
                    source_kind = frozenset(
                        {
                            EventKind.MARKET_EPOCH_RESET,
                            EventKind.DISPLACEMENT_OBSERVED,
                        }
                    )
                elif lifecycle.terminal_reason in {
                    FVGTerminationCause.PARENT_STRUCTURE_TERMINATED,
                    FVGTerminationCause.STRUCTURAL_RANGE_REPLACED,
                }:
                    source_kind = frozenset(
                        {
                            EventKind.ACCEPTANCE_CONFIRMED,
                            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                            EventKind.MARKET_EPOCH_RESET,
                        }
                    )
                else:
                    source_kind = frozenset(
                        {EventKind.MARKET_EPOCH_RESET}
                    )
                cause_event_ids = tuple(
                    event_id
                    for event_id in lifecycle.terminal_source_event_ids[
                        1
                        + len(lifecycle.context_source_event_ids)
                        :
                    ]
                    if event_id != lifecycle.terminal_event_id
                )
                for event_id in cause_event_ids:
                    source = self._group3_foundation_source_event(
                        event_id,
                        expected_kinds=source_kind,
                    )
                    if (
                        source.kind is EventKind.DISPLACEMENT_OBSERVED
                        and (
                            source.evidence.get("lifecycle") != "censored"
                            or source.evidence.get("terminal_reason")
                            not in {
                                "data_gap_history_reset",
                                "data_anomaly",
                                "synthetic_interruption",
                            }
                        )
                    ):
                        raise ValueError(
                            "FVG censorship event is not an exact boundary"
                        )
                if (
                    lifecycle.terminal_reason
                    is FVGTerminationCause.CLOSE_THROUGH_FAR_EDGE
                ):
                    terminal_kind = frozenset(
                        {EventKind.FVG_INVALIDATED}
                    )
                else:
                    terminal_kind = source_kind
                terminal = self._group3_foundation_source_event(
                    lifecycle.terminal_event_id,
                    expected_kinds=terminal_kind,
                )
                if (
                    lifecycle.terminal_reason
                    is FVGTerminationCause.DATA_GAP
                    and terminal.kind is EventKind.DISPLACEMENT_OBSERVED
                    and (
                        terminal.evidence.get("lifecycle") != "censored"
                        or terminal.evidence.get("terminal_reason")
                        not in {
                            "data_gap_history_reset",
                            "data_anomaly",
                            "synthetic_interruption",
                        }
                    )
                ):
                    raise ValueError(
                        "FVG censorship terminal is not an exact boundary"
                    )
        for retest in update.first_retests:
            creation_kind = (
                EventKind.FVG_CREATED
                if retest.object_kind is ZoneObjectKind.FVG
                else EventKind.ORIGIN_ZONE_CREATED
            )
            creation = self._group3_foundation_source_event(
                retest.creation_event_id,
                expected_kinds=frozenset({creation_kind}),
            )
            qualified = qualified_by_id.get(retest.object_id)
            if retest.object_kind is ZoneObjectKind.FVG:
                if creation.evidence.get("fvg_id") != retest.object_id:
                    raise ValueError(
                        "first FVG retest creation event is not exact"
                    )
                expected_displacement_id = creation.evidence.get(
                    "source_displacement_id"
                )
            else:
                if (
                    qualified is None
                    or creation.evidence.get("source_displacement_id")
                    != qualified.source_displacement_id
                    or creation.zone
                    != (qualified.lower_bound, qualified.upper_bound)
                ):
                    raise ValueError(
                        "first OB retest creation event is not exact"
                    )
                expected_displacement_id = (
                    qualified.source_displacement_id
                )
            departure = self._group3_foundation_source_event(
                retest.departure_source_event_id,
                expected_kinds=frozenset(
                    {
                        EventKind.DISPLACEMENT_OBSERVED,
                        EventKind.BAR_COMPLETED,
                    }
                ),
            )
            if (
                departure.kind is EventKind.DISPLACEMENT_OBSERVED
                and departure.evidence.get("displacement_id")
                != expected_displacement_id
            ):
                raise ValueError(
                    "first retest departure event is not exact"
                )
            source_bar = self._group3_foundation_source_event(
                retest.source_bar_event_id,
                expected_kinds=frozenset({EventKind.BAR_COMPLETED}),
            )
            if (
                source_bar.timeframe is not retest.timeframe
                or source_bar.known_at != retest.known_at
                or source_bar.evidence.get("real_completed") is not True
                or source_bar.evidence.get("clock_only") is not False
            ):
                raise ValueError("first retest BAR event is not exact")
            for event_id in retest.context_event_ids:
                if self.memory.audit_event_including_pending(event_id) is None:
                    raise ValueError("first retest context event is absent")

    def _group3_raw_only_structure_dispositions(
        self,
    ) -> tuple[Group3RawOnlyStructureDisposition, ...]:
        """Prove provisional QOB sources that intentionally stopped at RAW."""

        if self._group3_tracker is None:
            return ()
        raw_only: list[Group3RawOnlyStructureDisposition] = []
        for completed in self._group3_tracker._pending_foundation_completed:
            for seed in completed.new_qualified_order_blocks:
                state = seed.legacy_state
                bos_id = seed.compatible_structure_entity_id
                disposition = self._raw_only_structure_dispositions.get(
                    bos_id
                )
                if bos_id in self._qualified_structure_event_ids:
                    if disposition is not None:
                        raise ValueError(
                            "raw-only QOB disposition conflicts with a "
                            "qualified BOS/MSS binding"
                        )
                    continue
                if disposition is None:
                    # A generic missing binding is not evidence that the BOS
                    # was intentionally left raw; finalization remains
                    # fail-closed for that case.
                    continue
                raw_event_id = self._raw_break_event_ids.get(bos_id)
                if raw_event_id != disposition.raw_break_event_id:
                    raise ValueError(
                        "raw-only QOB disposition lost its exact raw-break "
                        "custody"
                    )
                raw = self._group3_foundation_source_event(
                    disposition.raw_break_event_id,
                    expected_kinds=frozenset(
                        {EventKind.RAW_BOUNDARY_BREAK}
                    ),
                )
                relation_events = tuple(
                    event
                    for event in (
                        *self.audit_store.events(),
                        *self.memory._audit_pending,
                    )
                    if event.kind
                    in {
                        EventKind.QUALIFIED_BOS,
                        EventKind.MSS_CORE_CONFIRMED,
                    }
                    and event.evidence.get("bos_id") == bos_id
                )
                if relation_events:
                    raise ValueError(
                        "raw-only QOB disposition has a qualified BOS/MSS "
                        "audit fact"
                    )
                target_event_id = self._confirmed_swing_event_ids.get(
                    state.source_bos_target_swing_id
                )
                break_bar_event_id = self._bar_event_ids_by_candle_id.get(
                    state.source_bos_break_bar_id
                )
                structure_event_id = self._structure_direction_event_ids.get(
                    state.source_bos_structure_id
                )
                active_displacement_event_id = (
                    self._displacement_event_ids.get(
                        state.source_active_transition_id
                    )
                )
                bos_source_displacement_event_id = (
                    None
                    if disposition.bos_source_displacement_id is None
                    else (
                        raw.context_event_ids[1]
                        if len(raw.context_event_ids) == 2
                        else None
                    )
                )
                if (
                    disposition.bos_source_displacement_id is not None
                    and bos_source_displacement_event_id is None
                ):
                    raise ValueError(
                        "raw-only QOB disposition lacks its exact optional "
                        "BOS displacement source"
                    )
                if any(
                    event_id is None
                    for event_id in (
                        target_event_id,
                        break_bar_event_id,
                        structure_event_id,
                        active_displacement_event_id,
                    )
                ):
                    raise ValueError(
                        "raw-only QOB disposition lacks its exact canonical "
                        "source chain"
                    )
                expected_raw_context = (
                    raw.context_event_ids[:1]
                    + (
                        ()
                        if bos_source_displacement_event_id is None
                        else (bos_source_displacement_event_id,)
                    )
                )
                if (
                    not raw.context_event_ids
                    or raw.context_event_ids != expected_raw_context
                ):
                    raise ValueError(
                        "raw-only QOB disposition has an invalid context chain"
                    )
                protected_assignment = (
                    self._historical_live_protected_assignment(
                        state.timeframe,
                        at_event=raw,
                    )
                )
                if (
                    protected_assignment is None
                    or protected_assignment.event_id
                    != disposition.protected_assignment_event_id
                ):
                    raise ValueError(
                        "raw-only QOB disposition lost its exact historical "
                        "protected-assignment witness"
                    )
                bos_state = self.memory.audit_event_including_pending(
                    raw.context_event_ids[0]
                )
                active_displacement = (
                    self.memory.audit_event_including_pending(
                        active_displacement_event_id
                    )
                )
                bos_source_displacement = (
                    None
                    if bos_source_displacement_event_id is None
                    else self.memory.audit_event_including_pending(
                        bos_source_displacement_event_id
                    )
                )
                structure = self.memory.audit_event_including_pending(
                    structure_event_id
                )
                if (
                    bos_state is None
                    or active_displacement is None
                    or structure is None
                    or (
                        bos_source_displacement_event_id is not None
                        and bos_source_displacement is None
                    )
                ):
                    raise ValueError(
                        "raw-only QOB disposition lacks its exact audit "
                        "source chain"
                    )
                expected_source_entities = (
                    bos_id,
                    state.source_bos_target_swing_id,
                    state.source_bos_structure_id,
                    *(
                        ()
                        if disposition.bos_source_displacement_id is None
                        else (disposition.bos_source_displacement_id,)
                    ),
                )
                expected_bos_sources = (
                    state.source_bos_target_swing_id,
                    state.source_bos_structure_id,
                    *(
                        ()
                        if disposition.bos_source_displacement_id is None
                        else (disposition.bos_source_displacement_id,)
                    ),
                    state.source_bos_break_bar_id,
                )
                invalid_bos_source_displacement = bool(
                    bos_source_displacement is not None
                    and (
                        bos_source_displacement.kind
                        is not EventKind.DISPLACEMENT_OBSERVED
                        or not bos_source_displacement.is_canonical_semantic
                        or bos_source_displacement.timeframe
                        is not state.timeframe
                        or bos_source_displacement.direction
                        is not state.direction
                        or bos_source_displacement.evidence.get(
                            "displacement_id"
                        )
                        != disposition.bos_source_displacement_id
                    )
                )
                if (
                    seed.compatible_structure_kind
                    is not CompatibleStructureKind.QUALIFIED_BOS
                    or state.source_bos_id != bos_id
                    or state.source_bos_scope is not BOSScope.CONTINUATION
                    or state.source_bos_mss_qualified
                    or state.timeframe is not Timeframe.M5
                    or state.source_bos_resolved_at != state.confirmed_at
                    or state.confirmed_at != completed.candle.end
                    or state.source_bos_break_bar_id
                    != completed.candle_id
                    or disposition.bos_id != bos_id
                    or disposition.timeframe is not state.timeframe
                    or disposition.direction is not state.direction
                    or disposition.resolved_at
                    != state.source_bos_resolved_at
                    or disposition.source_structure_id
                    != state.source_bos_structure_id
                    or disposition.target_swing_id
                    != state.source_bos_target_swing_id
                    or disposition.break_bar_id
                    != state.source_bos_break_bar_id
                    or protected_assignment.event_id
                    != disposition.protected_assignment_event_id
                    or protected_assignment.timeframe is not state.timeframe
                    or protected_assignment.direction is state.direction
                    or protected_assignment.known_at
                    > state.source_bos_resolved_at
                    or raw.timeframe is not state.timeframe
                    or raw.direction is not state.direction
                    or raw.event_time != state.source_bos_resolved_at
                    or raw.known_at != state.source_bos_resolved_at
                    or raw.evidence.get("bos_id") != bos_id
                    or raw.evidence.get("scope")
                    != BOSScope.CONTINUATION.value
                    or raw.evidence.get("target_swing_id")
                    != state.source_bos_target_swing_id
                    or raw.evidence.get("break_bar_id")
                    != state.source_bos_break_bar_id
                    or raw.evidence.get("source_displacement_id")
                    != disposition.bos_source_displacement_id
                    or raw.source_event_ids
                    != (target_event_id, break_bar_event_id)
                    or raw.source_data_ids
                    != (state.source_bos_break_bar_id,)
                    or raw.source_entity_ids != expected_source_entities
                    or raw.context_event_ids != expected_raw_context
                    or bos_state.kind is not EventKind.STRUCTURE_BREAK
                    or bos_state.entity_id != bos_id
                    or bos_state.lifecycle != BOSLifecycle.CONFIRMED.value
                    or bos_state.timeframe is not state.timeframe
                    or bos_state.direction is not state.direction
                    or bos_state.formed_at != state.source_bos_pending_at
                    or bos_state.confirmed_at
                    != state.source_bos_resolved_at
                    or bos_state.known_at != state.source_bos_resolved_at
                    or bos_state.source_ids != expected_bos_sources
                    or bos_state.evidence.get("bos_id") != bos_id
                    or bos_state.evidence.get("scope")
                    != BOSScope.CONTINUATION.value
                    or bos_state.evidence.get("source_structure_id")
                    != state.source_bos_structure_id
                    or bos_state.evidence.get("break_bar_id")
                    != state.source_bos_break_bar_id
                    or bos_state.evidence.get("source_displacement_id")
                    != disposition.bos_source_displacement_id
                    or structure.kind
                    is not EventKind.STRUCTURE_DIRECTION_CONFIRMED
                    or not structure.is_canonical_semantic
                    or structure.timeframe is not state.timeframe
                    or structure.direction is not state.direction
                    or structure.evidence.get("structure_id")
                    != state.source_bos_structure_id
                    or not structure.source_entity_ids
                    or structure.source_entity_ids[0]
                    != state.source_bos_structure_id
                    or active_displacement.kind
                    is not EventKind.DISPLACEMENT_OBSERVED
                    or not active_displacement.is_canonical_semantic
                    or active_displacement.timeframe is not state.timeframe
                    or active_displacement.direction is not state.direction
                    or active_displacement.known_at
                    != state.source_displacement_active_at
                    or active_displacement.event_time
                    != state.source_displacement_started_at
                    or active_displacement.evidence.get("transition_id")
                    != state.source_active_transition_id
                    or active_displacement.evidence.get("displacement_id")
                    != state.source_displacement_id
                    or active_displacement.evidence.get("lifecycle")
                    != DisplacementLifecycle.ACTIVE.value
                    or active_displacement.source_entity_ids
                    != (state.source_displacement_id,)
                    or invalid_bos_source_displacement
                ):
                    raise ValueError(
                        "raw-only QOB disposition does not match its exact "
                        "BOS, structure, displacement, or BAR provenance"
                    )
                raw_only.append(disposition)
        if len(raw_only) != len({item.bos_id for item in raw_only}):
            raise ValueError("raw-only QOB disposition is ambiguous")
        return tuple(raw_only)

    def _finalize_group3_foundation(
        self,
        update: Group3Update,
        reader_update: ReaderUpdate,
    ) -> Group3Update:
        if self._group3_tracker is None:
            return update
        sessions = {
            candle.end: session_name_phase(candle.end)[0]
            for candle in reader_update.newly_completed.get(
                Timeframe.M5,
                (),
            )
        }
        raw_only_dispositions = (
            self._group3_raw_only_structure_dispositions()
        )
        finalized = self._group3_tracker.finalize_foundation(
            update,
            bar_event_ids_by_candle_id=(
                self._bar_event_ids_by_candle_id
            ),
            displacement_event_ids_by_identity=(
                self._displacement_event_ids
            ),
            displacement_event_known_at_by_identity={
                identity: event.known_at
                for identity, event_id in self._displacement_event_ids.items()
                if (
                    event := self.memory.audit_event_including_pending(
                        event_id
                    )
                ) is not None
                and event.kind is EventKind.DISPLACEMENT_OBSERVED
            },
            structure_event_ids_by_entity=(
                self._qualified_structure_event_ids
            ),
            raw_only_structure_dispositions=(
                raw_only_dispositions
            ),
            fvg_creation_event_ids_by_entity=(
                self._fvg_created_event_ids
            ),
            fvg_terminal_event_ids_by_entity=(
                self._fvg_terminal_event_ids
            ),
            order_block_creation_event_ids_by_entity=(
                self._origin_zone_created_event_ids
            ),
            sessions_by_clock=sessions,
        )
        self._validate_group3_foundation_projection(finalized)
        if any(
            self._raw_only_structure_dispositions.get(disposition.bos_id)
            != disposition
            for disposition in raw_only_dispositions
        ):
            raise RuntimeError(
                "raw-only QOB disposition changed during finalization"
            )
        for disposition in raw_only_dispositions:
            del self._raw_only_structure_dispositions[disposition.bos_id]
        return finalized

    def _finalize_group3_foundation_boundary(
        self,
        update: Group3Update,
    ) -> Group3Update:
        if self._group3_tracker is None:
            return update
        pending = self._group3_tracker._pending_foundation_boundary
        if pending is None:
            return update
        reason, clock = pending
        boundary_event_id = self._last_market_epoch_reset_event_id
        if boundary_event_id is None:
            expected_terminal_reason = {
                "data_gap_reset": "data_gap_history_reset",
                "data_anomaly": "data_anomaly",
                "synthetic_interruption": "synthetic_interruption",
            }.get(reason)
            candidates = tuple(
                event
                for event_id in dict.fromkeys(
                    self._displacement_event_ids.values()
                )
                if (
                    (event := self.memory.audit_event_including_pending(
                        event_id
                    ))
                    is not None
                    and event.kind is EventKind.DISPLACEMENT_OBSERVED
                    and event.known_at == clock
                    and event.evidence.get("lifecycle") == "censored"
                    and event.evidence.get("terminal_reason")
                    == expected_terminal_reason
                )
            )
            if len(candidates) != 1:
                raise ValueError(
                    "foundation FVG censorship lacks its exact canonical "
                    "boundary event"
                )
            boundary_event_id = candidates[0].event_id
        finalized = self._group3_tracker.finalize_foundation_boundary(
            update,
            boundary_event_id=boundary_event_id,
        )
        self._validate_group3_foundation_projection(finalized)
        return finalized

    def _foundation_exact_event(self, event_id: str) -> MarketEvent:
        event = self.audit_store.get(event_id)
        if (
            event is None
            or event.kind is EventKind.FOUNDATION_STATE_CHANGED
            or event.origin
            not in {
                EventOrigin.NORMALIZED_DATA,
                EventOrigin.SEMANTIC_ATOMIC,
            }
        ):
            raise ValueError(
                "foundation projection requires an earlier authoritative fact"
            )
        return event

    def _foundation_real_bar_event_id(
        self,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
    ) -> str:
        matches = tuple(
            event_id
            for clock, event_id in self._real_bar_event_ids_by_timeframe[
                timeframe
            ]
            if clock == known_at
        )
        if len(matches) != 1:
            raise ValueError(
                "foundation projection lacks one exact real native BAR"
            )
        event = self._foundation_exact_event(matches[0])
        if (
            event.kind is not EventKind.BAR_COMPLETED
            or event.timeframe is not timeframe
            or event.known_at != known_at
            or event.evidence.get("real_completed") is not True
            or event.evidence.get("clock_only") is not False
        ):
            raise ValueError("foundation native BAR binding is not exact")
        return event.event_id

    @staticmethod
    def _foundation_unique_ids(values: Iterable[str]) -> tuple[str, ...]:
        result = tuple(dict.fromkeys(values))
        if not result or any(not value for value in result):
            raise ValueError("foundation ancestry requires non-empty identities")
        return result

    @staticmethod
    def _foundation_plan_is_new(
        revisions: dict[tuple[object, ...], object],
        *,
        key: tuple[object, ...],
        value: object,
    ) -> bool:
        """Register one typed immutable revision without serializing it.

        Keys identify a semantic revision, not merely an object.  Repeating
        the same frozen DTO is a no-op; different content under the same
        revision identity fails closed instead of being hidden by this
        performance cache.
        """

        incumbent = revisions.get(key)
        if incumbent is None:
            revisions[key] = value
            return True
        if type(incumbent) is not type(value) or incumbent != value:
            raise ValueError(
                "foundation typed revision changed after first publication"
            )
        return False

    @staticmethod
    def _foundation_plan_sort_identity(value: object) -> tuple[str, str]:
        """Return a cheap deterministic identity for an already typed plan."""

        for attribute in (
            "fact_id",
            "assignment_id",
            "swing_id",
            "leg_id",
            "core_id",
            "qualified_ob_id",
            "first_retest_event_id",
            "fvg_id",
            "range_id",
        ):
            identity = getattr(value, attribute, None)
            if isinstance(identity, str) and identity:
                return type(value).__name__, identity
        raise TypeError("foundation plan lacks a deterministic typed identity")

    def _foundation_group3_plans(
        self,
        update: Group3Update | None,
        revisions: dict[tuple[object, ...], object],
    ) -> tuple[tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None], ...]:
        if update is None:
            return ()
        plans: list[
            tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None]
        ] = []
        for core in update.base_origin_cores:
            if self._foundation_plan_is_new(
                revisions,
                key=("base_origin_core", core.core_id, core.known_at),
                value=core,
            ):
                plans.append((core.known_at, 40, "dto", core, None))
        for qualified in update.qualified_order_blocks:
            if self._foundation_plan_is_new(
                revisions,
                key=(
                    "qualified_order_block",
                    qualified.qualified_ob_id,
                    qualified.known_at,
                ),
                value=qualified,
            ):
                plans.append(
                    (qualified.known_at, 50, "dto", qualified, None)
                )
        # Transitions retain an ACTIVE revision that may no longer be present
        # in the bounded current view.  Preserve that revision before its
        # terminal state instead of publishing only hindsight state.
        fvg_values = (
            *update.fvg_structural_transitions,
            *update.fvg_structural_lifecycles,
        )
        pending_fvgs: dict[tuple[object, ...], object] = {}
        for lifecycle in fvg_values:
            if lifecycle.terminal_reason is None and lifecycle.age_bars != 0:
                continue
            key = (
                "fvg_structural_lifecycle",
                lifecycle.fvg_id,
                lifecycle.last_updated_at,
                lifecycle.availability.value,
            )
            incumbent = revisions.get(key)
            if incumbent is not None:
                if type(incumbent) is not type(lifecycle) or incumbent != lifecycle:
                    raise ValueError(
                        "foundation typed revision changed after first publication"
                    )
                continue
            pending = pending_fvgs.get(key)
            if pending is not None:
                if pending != lifecycle:
                    raise ValueError(
                        "Group 3 emitted conflicting same-clock FVG revisions"
                    )
                continue
            # Creation-time context can be bound while this plan is staged.
            # Register the final bound DTO in the clock loop, not this
            # pre-binding value.
            pending_fvgs[key] = lifecycle
            plans.append(
                (
                    lifecycle.last_updated_at,
                    60,
                    "fvg",
                    lifecycle,
                    None,
                )
            )
        retests = (*update.first_retest_transitions, *update.first_retests)
        for retest in retests:
            if self._foundation_plan_is_new(
                revisions,
                key=(
                    "zone_first_retest",
                    retest.first_retest_event_id,
                    retest.known_at,
                ),
                value=retest,
            ):
                # Group 3 freezes geometric first reinteraction before it
                # applies the same completed BAR's terminal FVG transition.
                # Preserve that causal order in the append-only projection.
                plans.append((retest.known_at, 55, "dto", retest, None))
        return tuple(plans)

    def _foundation_geometry_plans(
        self,
        *,
        frames: Mapping[Timeframe, FrameObservation],
        histories: Mapping[Timeframe, Sequence[Candle]],
        known_at: pd.Timestamp,
        revisions: dict[tuple[object, ...], object],
    ) -> tuple[
        dict[str, SwingGeometryNode],
        tuple[SwingGeometryAssignment, ...],
        tuple[
            tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None], ...
        ],
    ]:
        nodes = dict(self._foundation_geometry_nodes)
        new_nodes: list[SwingGeometryNode] = []
        for timeframe in self._active_timeframes:
            unseen_swings = tuple(
                swing
                for swing in frames[timeframe].swings
                if swing.swing_id not in nodes
            )
            if not unseen_swings:
                continue
            for node in build_swing_geometry_nodes(
                unseen_swings,
                histories[timeframe],
                tick_size=self.config.tick_size,
            ):
                incumbent = nodes.get(node.swing_id)
                if incumbent is not None:
                    if incumbent != node:
                        raise ValueError(
                            "swing geometry changed after first publication"
                        )
                    continue
                nodes[node.swing_id] = node
                new_nodes.append(node)

        assignments = tuple(self._foundation_geometry_assignments)
        appended_assignments: list[SwingGeometryAssignment] = []
        for clock in sorted({node.known_at for node in new_nodes}):
            updated = update_swing_geometry_assignments(
                tuple(nodes.values()),
                assignments,
                known_at=clock,
            )
            appended_assignments.extend(updated[len(assignments) :])
            assignments = updated

        plans: list[
            tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None]
        ] = []
        for node in new_nodes:
            if not self._foundation_plan_is_new(
                revisions,
                key=("swing_geometry_node", node.swing_id, node.known_at),
                value=node,
            ):
                continue
            swing_event_id = self._confirmed_swing_event_ids.get(node.swing_id)
            if swing_event_id is None:
                raise ValueError("swing geometry lacks its confirmed Swing fact")
            bar_event_ids = tuple(
                self._bar_event_ids_by_candle_id.get(candle_id, "")
                for candle_id in node.source_candle_ids
            )
            plans.append(
                (
                    node.known_at,
                    10,
                    "dto",
                    node,
                    self._foundation_unique_ids(
                        (swing_event_id, *bar_event_ids)
                    ),
                )
            )
        for assignment in appended_assignments:
            if not self._foundation_plan_is_new(
                revisions,
                key=(
                    "swing_geometry_assignment",
                    assignment.assignment_id,
                    assignment.assigned_at,
                ),
                value=assignment,
            ):
                continue
            source_ids = [
                self._confirmed_swing_event_ids.get(
                    assignment.child_swing_id,
                    "",
                )
            ]
            if assignment.parent_swing_id is not None:
                source_ids.append(
                    self._confirmed_swing_event_ids.get(
                        assignment.parent_swing_id,
                        "",
                    )
                )
            plans.append(
                (
                    assignment.assigned_at,
                    20,
                    "dto",
                    assignment,
                    self._foundation_unique_ids(source_ids),
                )
            )
        return nodes, assignments, tuple(plans)

    @staticmethod
    def _foundation_geometry_invalidated(
        authoritative_events: Sequence[MarketEvent],
    ) -> bool:
        """Return whether this clock can introduce swing geometry."""

        return any(
            event.kind is EventKind.SWING_CONFIRMED
            for event in authoritative_events
        )

    @staticmethod
    def _foundation_cluster_membership_invalidated(
        new_records: Sequence[FoundationRecord],
        authoritative_events: Sequence[MarketEvent],
    ) -> bool:
        """Return whether canonical cluster membership can have changed."""

        return any(
            event.kind is EventKind.MARKET_EPOCH_RESET
            for event in authoritative_events
        ) or any(
            record.object_type is FoundationObjectType.LIQUIDITY_LEVEL
            for record in new_records
        )

    def _foundation_leg_plans(
        self,
        frames: Mapping[Timeframe, FrameObservation],
        revisions: dict[tuple[object, ...], object],
    ) -> tuple[
        tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None], ...
    ]:
        plans: list[
            tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None]
        ] = []
        for frame in frames.values():
            for leg in frame.structural_legs:
                if leg.foundation_version is None:
                    continue
                if not self._foundation_plan_is_new(
                    revisions,
                    key=("structural_leg", leg.leg_id, leg.known_at),
                    value=leg,
                ):
                    continue
                leg_event_id = self._structural_leg_event_ids.get(leg.leg_id)
                swing_event_ids = tuple(
                    self._confirmed_swing_event_ids.get(swing_id, "")
                    for swing_id in leg.source_swing_ids
                )
                bar_event_ids = tuple(
                    self._bar_event_ids_by_candle_id.get(candle_id, "")
                    for candle_id in (
                        *leg.atr_source_candle_ids,
                        *leg.path_candle_ids,
                    )
                )
                plans.append(
                    (
                        leg.known_at,
                        30,
                        "dto",
                        leg,
                        self._foundation_unique_ids(
                            (
                                leg_event_id or "",
                                *swing_event_ids,
                                *bar_event_ids,
                            )
                        ),
                    )
                )
        return tuple(plans)

    def _foundation_balance_plans(
        self,
        group4_update: Group4Update | None,
        revisions: dict[tuple[object, ...], object],
    ) -> tuple[
        tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None], ...
    ]:
        if group4_update is None:
            return ()
        # Persist lifecycle transitions, not the continuously revised current
        # view.  Balance metrics remain live in Group 4; foundation records
        # freeze entry/terminal clocks without one technical heartbeat per M1
        # bar.
        values = group4_update.range_transitions
        plans: list[
            tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None]
        ] = []
        for state in values:
            if not self._foundation_plan_is_new(
                revisions,
                key=(
                    "balance_range",
                    state.range_id,
                    state.last_updated_at,
                    state.lifecycle.value,
                ),
                value=state,
            ):
                continue
            event_id = {
                DealingRangeLifecycle.FORMING: self._range_created_event_ids,
                DealingRangeLifecycle.MATURE: self._range_active_event_ids,
                DealingRangeLifecycle.BROKEN: self._range_terminal_event_ids,
            }[state.lifecycle].get(state.range_id)
            if event_id is None:
                raise ValueError("BalanceRange lacks its canonical lifecycle fact")
            sources = [event_id]
            # Current range metrics are revised by this exact completed M1
            # observation, not by the older lifecycle transition alone.
            try:
                sources.append(
                    self._foundation_real_bar_event_id(
                        Timeframe.M1,
                        state.last_updated_at,
                    )
                )
            except ValueError:
                event = self._foundation_exact_event(event_id)
                if event.known_at != state.last_updated_at:
                    raise
            plans.append(
                (
                    state.last_updated_at,
                    80,
                    "dto",
                    state,
                    self._foundation_unique_ids(sources),
                )
            )
        return tuple(plans)

    def _foundation_boundary_plans(
        self,
        frames: Mapping[Timeframe, FrameObservation],
        revisions: dict[tuple[object, ...], object],
    ) -> tuple[
        tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None], ...
    ]:
        plans: list[
            tuple[pd.Timestamp, int, str, object, tuple[str, ...] | None]
        ] = []
        for frame in frames.values():
            for state in frame.structure_breaks:
                target_event_id = self._confirmed_swing_event_ids.get(
                    state.target_swing_id
                )
                if target_event_id is None:
                    continue
                for clock in state.attempt_clocks:
                    bar_event_id = self._foundation_real_bar_event_id(
                        state.timeframe,
                        clock,
                    )
                    bar = self._foundation_exact_event(bar_event_id)
                    fact = NormalizedLifecycleTransition(
                        fact_id=canonical_semantic_id(
                            "observer-boundary-attack",
                            state.bos_id,
                            bar_event_id,
                        ),
                        kind=(
                            NormalizedTransitionKind.BOUNDARY_ATTACK_OBSERVED
                        ),
                        known_at=clock,
                        timeframe=state.timeframe,
                        source_event_ids=(target_event_id, bar_event_id),
                        payload={
                            "bos_generation_id": state.bos_id,
                            "direction": state.direction.value,
                            "target_swing_event_id": target_event_id,
                            "bar_event_id": bar_event_id,
                            "boundary_ticks": state.target_ticks,
                            "high_ticks": price_to_ticks(
                                float(bar.evidence["high"]),
                                self.config.tick_size,
                                name="boundary attack high",
                            ),
                            "low_ticks": price_to_ticks(
                                float(bar.evidence["low"]),
                                self.config.tick_size,
                                name="boundary attack low",
                            ),
                            "close_ticks": price_to_ticks(
                                float(bar.evidence["close"]),
                                self.config.tick_size,
                                name="boundary attack close",
                            ),
                        },
                    )
                    if not self._foundation_plan_is_new(
                        revisions,
                        key=(
                            "boundary_attack",
                            fact.fact_id,
                            fact.known_at,
                        ),
                        value=fact,
                    ):
                        continue
                    plans.append((clock, 35, "boundary", fact, None))
        return tuple(plans)

    @staticmethod
    def _foundation_structure_at(
        adapter: CanonicalFoundationAdapter,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
    ):
        candidates = tuple(
            generation
            for generation in adapter.lifecycle.structure_generations
            if generation.timeframe is timeframe
            and generation.scope is StructureScope.EXTERNAL
            and generation.confirmed_at is not None
            and generation.confirmed_at <= known_at
            and (
                generation.terminated_at is None
                or generation.terminated_at > known_at
            )
        )
        return (
            None
            if not candidates
            else max(
                candidates,
                key=lambda generation: (
                    generation.confirmed_at,
                    generation.generation_id,
                ),
            )
        )

    def _foundation_zero_width_structural_origin(
        self,
        origin: MarketEvent,
    ) -> bool:
        """Return whether exact opposite source Swings occupy one tick."""

        if origin.kind is not EventKind.STRUCTURE_DIRECTION_CONFIRMED:
            raise ValueError("structural range owner lacks its exact origin fact")
        source_ids = {
            "low": origin.evidence.get("source_low_id"),
            "high": origin.evidence.get("source_high_id"),
        }
        if any(
            not isinstance(swing_id, str) or not swing_id
            for swing_id in source_ids.values()
        ):
            raise ValueError("structure generation lacks causal range Swings")
        source_ticks: dict[str, int] = {}
        for side, swing_id in source_ids.items():
            event_id = self._confirmed_swing_event_ids.get(swing_id)
            if event_id is None or event_id not in origin.source_event_ids:
                raise ValueError(
                    "structural range origin lacks its exact confirmed Swing fact"
                )
            source = self._foundation_exact_event(event_id)
            if (
                source.kind is not EventKind.SWING_CONFIRMED
                or source.timeframe is not origin.timeframe
                or source.evidence.get("source_entity_id") != swing_id
                or source.evidence.get("side") != side
                or source.price is None
            ):
                raise ValueError(
                    "structural range origin Swing provenance is incompatible"
                )
            source_ticks[side] = price_to_ticks(
                source.price,
                self.config.tick_size,
                name=f"structural range {side} Swing",
            )
        return source_ticks["low"] == source_ticks["high"]

    def _foundation_update_structural_ranges(
        self,
        *,
        adapter: CanonicalFoundationAdapter,
        ranges: dict[Timeframe, StructuralRangeState],
        frames: Mapping[Timeframe, FrameObservation],
        known_at: pd.Timestamp,
        clock_events: Sequence[MarketEvent],
        revisions: dict[tuple[object, ...], object],
    ) -> tuple[
        dict[Timeframe, StructuralRangeState],
        tuple[tuple[str, str], ...],
    ]:
        updated = dict(ranges)
        terminated: list[tuple[str, str]] = []
        reset = next(
            (
                event
                for event in clock_events
                if event.kind is EventKind.MARKET_EPOCH_RESET
            ),
            None,
        )
        if reset is not None:
            reset_reason = {
                "contract_change_reset": "contract_reset",
                "contract_reset": "contract_reset",
                "data_gap_reset": "data_reset",
                "data_reset": "data_reset",
                "semantic_reset": "semantic_reset",
            }.get(str(reset.evidence.get("reason")))
            if reset_reason is None:
                raise ValueError("structural range reset reason is unregistered")
            for timeframe, state in tuple(updated.items()):
                if state.terminated_at is not None:
                    continue
                terminal = terminate_structural_range(
                    state,
                    terminated_at=known_at,
                    reason=reset_reason,
                )
                if self._foundation_plan_is_new(
                    revisions,
                    key=(
                        "structural_range",
                        terminal.range_id,
                        terminal.updated_at,
                        terminal.termination_reason,
                    ),
                    value=terminal,
                ):
                    adapter.append_dto(
                        terminal,
                        source_event_ids=(reset.event_id,),
                    )
                updated[timeframe] = terminal
                terminated.append((terminal.range_id, reset.event_id))

        for timeframe, state in tuple(updated.items()):
            if state.terminated_at is not None:
                continue
            owner = next(
                (
                    generation
                    for generation in adapter.lifecycle.structure_generations
                    if generation.generation_id
                    == state.structure_generation_id
                ),
                None,
            )
            if (
                owner is None
                or owner.terminated_at is None
                or owner.terminated_at > known_at
            ):
                continue
            cause_event_id = owner.protected_acceptance_event_id
            if cause_event_id is None:
                causes = tuple(
                    event_id
                    for event_id in reversed(owner.source_event_ids)
                    if self._foundation_exact_event(event_id).known_at
                    == owner.terminated_at
                )
                if not causes:
                    raise ValueError(
                        "structural range owner termination lacks a cause fact"
                    )
                cause_event_id = causes[0]
            terminal = terminate_structural_range(
                state,
                terminated_at=owner.terminated_at,
                reason="structure_generation_terminated",
            )
            if self._foundation_plan_is_new(
                revisions,
                key=(
                    "structural_range",
                    terminal.range_id,
                    terminal.updated_at,
                    terminal.termination_reason,
                ),
                value=terminal,
            ):
                adapter.append_dto(
                    terminal,
                    source_event_ids=(cause_event_id,),
                )
            updated[timeframe] = terminal
            terminated.append((terminal.range_id, cause_event_id))

        swings_by_timeframe = {
            timeframe: {
                swing.swing_id: swing for swing in frame.swings
            }
            for timeframe, frame in frames.items()
        }
        for timeframe in self._active_timeframes:
            generation = self._foundation_structure_at(
                adapter,
                timeframe,
                known_at,
            )
            if generation is None:
                continue
            incumbent = updated.get(timeframe)
            if (
                incumbent is not None
                and incumbent.terminated_at is None
                and incumbent.structure_generation_id
                == generation.generation_id
            ):
                continue
            origin = self._foundation_exact_event(generation.origin_event_id)
            zero_width = self._foundation_zero_width_structural_origin(origin)
            if generation.confirmed_at != known_at:
                if incumbent is None and not zero_width:
                    raise ValueError(
                        "structural range missed its generation confirmation clock"
                    )
                continue
            low_id = origin.evidence.get("source_low_id")
            high_id = origin.evidence.get("source_high_id")
            if not isinstance(low_id, str) or not isinstance(high_id, str):
                raise ValueError("structure generation lacks causal range Swings")
            try:
                low = swings_by_timeframe[timeframe][low_id]
                high = swings_by_timeframe[timeframe][high_id]
            except KeyError as error:
                raise ValueError(
                    "structural range source Swing is absent from the frame"
                ) from error
            if (low.price_ticks == high.price_ticks) is not zero_width:
                raise ValueError(
                    "structural range frame and atomic Swing prices conflict"
                )
            if zero_width:
                # Opposite equal-price pivots are balance geometry, not a
                # positive-width Structural Range.  Preserve the generation
                # and both SwingGeometry facts, but publish no range or
                # Premium/Discount coordinate for this immutable origin.
                continue
            cause_event_id = generation.confirmation_event_id
            if cause_event_id is None:
                raise ValueError("confirmed generation lacks its exact fact")
            if incumbent is not None and incumbent.terminated_at is None:
                terminal = terminate_structural_range(
                    incumbent,
                    terminated_at=known_at,
                    reason="structure_generation_terminated",
                )
                if self._foundation_plan_is_new(
                    revisions,
                    key=(
                        "structural_range",
                        terminal.range_id,
                        terminal.updated_at,
                        terminal.termination_reason,
                    ),
                    value=terminal,
                ):
                    adapter.append_dto(
                        terminal,
                        source_event_ids=(cause_event_id,),
                    )
                updated[timeframe] = terminal
                terminated.append((terminal.range_id, cause_event_id))
            state = build_structural_range(
                generation.generation_id,
                generation.direction,
                low,
                high,
                known_at=known_at,
                supersedes_range_id=(
                    None if incumbent is None else incumbent.range_id
                ),
            )
            source_ids = self._foundation_unique_ids(
                (
                    cause_event_id,
                    self._confirmed_swing_event_ids.get(low_id, ""),
                    self._confirmed_swing_event_ids.get(high_id, ""),
                )
            )
            if self._foundation_plan_is_new(
                revisions,
                key=(
                    "structural_range",
                    state.range_id,
                    state.updated_at,
                    state.termination_reason,
                ),
                value=state,
            ):
                adapter.append_dto(state, source_event_ids=source_ids)
            updated[timeframe] = state
        return updated, tuple(terminated)

    def _foundation_bind_fvg_context(
        self,
        *,
        adapter: CanonicalFoundationAdapter,
        ranges: Mapping[Timeframe, StructuralRangeState],
        lifecycle,
        contexts: dict[str, tuple[str | None, str | None]],
    ):
        prior = contexts.get(lifecycle.fvg_id)
        if prior is None:
            generation = self._foundation_structure_at(
                adapter,
                lifecycle.timeframe,
                lifecycle.known_at,
            )
            structural_range = ranges.get(lifecycle.timeframe)
            if (
                structural_range is not None
                and (
                    structural_range.known_at > lifecycle.known_at
                    or (
                        structural_range.terminated_at is not None
                        and structural_range.terminated_at
                        <= lifecycle.known_at
                    )
                    or generation is None
                    or structural_range.structure_generation_id
                    != generation.generation_id
                )
            ):
                structural_range = None
            prior = (
                None if generation is None else generation.generation_id,
                None if structural_range is None else structural_range.range_id,
            )
            contexts[lifecycle.fvg_id] = prior
        parent_id, range_id = prior
        if parent_id is None:
            if (
                lifecycle.parent_structure_generation_id is not None
                or lifecycle.structural_range_id is not None
            ):
                raise ValueError("FVG context conflicts with creation-time state")
            return lifecycle
        if lifecycle.parent_structure_generation_id is not None:
            if (
                lifecycle.parent_structure_generation_id != parent_id
                or lifecycle.structural_range_id != range_id
                or not lifecycle.context_source_event_ids
            ):
                raise ValueError("FVG structural context is not immutable")
            return lifecycle
        generation = next(
            item
            for item in adapter.lifecycle.structure_generations
            if item.generation_id == parent_id
        )
        context_sources = [generation.confirmation_event_id or ""]
        if range_id is not None:
            structural_range = next(
                state
                for state in ranges.values()
                if state.range_id == range_id
            )
            context_sources.extend(
                (
                    self._confirmed_swing_event_ids.get(
                        structural_range.lower_swing_id,
                        "",
                    ),
                    self._confirmed_swing_event_ids.get(
                        structural_range.upper_swing_id,
                        "",
                    ),
                )
            )
        sources = self._foundation_unique_ids(context_sources)
        if lifecycle.parent_structure_generation_id is None:
            if lifecycle.age_bars != 0:
                raise ValueError(
                    "FVG structural context was not bound at creation"
                )
            if self._group3_tracker is None:
                raise ValueError("FVG context requires the canonical Group3 owner")
            refreshed = self._group3_tracker.bind_fvg_foundation_context(
                fvg_id=lifecycle.fvg_id,
                parent_structure_generation_id=parent_id,
                structural_range_id=range_id,
                source_event_ids=sources,
            )
            matches = tuple(
                item
                for item in refreshed.fvg_structural_lifecycles
                if item.fvg_id == lifecycle.fvg_id
            )
            if len(matches) != 1:
                raise ValueError("Group3 lost its bound FVG lifecycle")
            lifecycle = matches[0]
        if (
            lifecycle.parent_structure_generation_id != parent_id
            or lifecycle.structural_range_id != range_id
            or tuple(lifecycle.context_source_event_ids) != sources
        ):
            raise ValueError("FVG structural context is not immutable")
        return lifecycle

    def _foundation_update_clusters(
        self,
        *,
        adapter: CanonicalFoundationAdapter,
        known_at: pd.Timestamp,
        authoritative_events: Sequence[MarketEvent],
        revisions: dict[tuple[object, ...], object],
    ) -> tuple[LiquidityClusterState, ...]:
        prior = tuple(self._foundation_active_clusters)
        reset = next(
            (
                event
                for event in authoritative_events
                if event.kind is EventKind.MARKET_EPOCH_RESET
            ),
            None,
        )
        if reset is not None and prior:
            reason = {
                "contract_change_reset": "contract_reset",
                "contract_reset": "contract_reset",
                "data_gap_reset": "data_reset",
                "data_reset": "data_reset",
                "semantic_reset": "semantic_reset",
            }.get(str(reset.evidence.get("reason")))
            if reason is None:
                raise ValueError("liquidity cluster reset reason is unregistered")
            for state in prior:
                terminal = replace(
                    state,
                    updated_at=known_at,
                    terminated_at=known_at,
                    termination_reason=reason,
                )
                if self._foundation_plan_is_new(
                    revisions,
                    key=(
                        "liquidity_cluster",
                        terminal.cluster_id,
                        terminal.updated_at,
                        terminal.termination_reason,
                    ),
                    value=terminal,
                ):
                    adapter.append_dto(
                        terminal,
                        source_event_ids=(reset.event_id,),
                    )
            prior = ()
        # Cluster the canonical level map itself.  Legacy inventory can remain
        # VISIBLE after a foundation terminal or can freeze CONSUMED before a
        # same-level rearm; using it here would therefore pollute membership in
        # both directions.  LiquidityInventoryItem is only the existing pure
        # geometry function's transport shape; its unused ``kind`` metadata
        # is normalized without changing the canonical source identity.
        cluster_inputs: list[LiquidityInventoryItem] = []
        levels_by_id = {level.level_id: level for level in adapter.lifecycle.levels}
        for level in sorted(
            adapter.lifecycle.levels,
            key=lambda item: (
                item.source_timeframe.value,
                item.side,
                item.price_ticks,
                item.level_id,
            ),
        ):
            if (
                level.lifecycle
                not in {
                    LiquidityLevelLifecycle.ACTIVE,
                    LiquidityLevelLifecycle.REARMED,
                }
                or level.lower_bound_ticks != level.price_ticks
                or level.upper_bound_ticks != level.price_ticks
            ):
                continue
            price = level.price_ticks * level.tick_size
            if level.active_generation_id is None:
                raise ValueError("active liquidity level lacks its generation")
            interaction = adapter.lifecycle.interaction(
                level.active_generation_id
            )
            cluster_inputs.append(
                LiquidityInventoryItem(
                    item_id=level.level_id,
                    timeframe=level.source_timeframe,
                    side=level.side,
                    # ``update_liquidity_clusters`` never reads this legacy
                    # transport field.  Use one fixed valid value rather than
                    # guessing a taxonomy from canonical source_kind text.
                    kind="swing",
                    price=price,
                    lower_bound=price,
                    upper_bound=price,
                    formed_at=level.created_at,
                    # Cluster membership starts when the *current*
                    # interaction generation arms.  A rearmed level must not
                    # backdate its new cluster age to Generation 1 creation.
                    confirmed_at=interaction.armed_at,
                    lifecycle=LiquidityInventoryLifecycle.VISIBLE,
                    source_ids=(level.source_identity,),
                    age_bars=0,
                    strength=0.0,
                )
            )
        update = update_liquidity_clusters(
            tuple(cluster_inputs),
            prior,
            tick_size=self.config.tick_size,
            known_at=known_at,
        )

        def level_sources(member_ids: Sequence[str]) -> tuple[str, ...]:
            return self._foundation_unique_ids(
                event_id
                for member_id in member_ids
                for event_id in levels_by_id[member_id].source_event_ids
                if adapter.is_known_input_event_id(event_id)
            )

        for terminal in update.terminated:
            if self._foundation_plan_is_new(
                revisions,
                key=(
                    "liquidity_cluster",
                    terminal.cluster_id,
                    terminal.updated_at,
                    terminal.termination_reason,
                ),
                value=terminal,
            ):
                adapter.append_dto(
                    terminal,
                    source_event_ids=level_sources(
                        terminal.member_level_ids
                    ),
                )
        for started in update.started:
            if self._foundation_plan_is_new(
                revisions,
                key=(
                    "liquidity_cluster",
                    started.cluster_id,
                    started.updated_at,
                    started.termination_reason,
                ),
                value=started,
            ):
                adapter.append_dto(
                    started,
                    source_event_ids=level_sources(started.member_level_ids),
                )
        active_by_id = {state.cluster_id: state for state in update.active}
        prior_by_id = {state.cluster_id: state for state in prior}
        for supersession in update.supersessions:
            member_ids: list[str] = []
            old = prior_by_id.get(supersession.superseded_cluster_id)
            if old is not None:
                member_ids.extend(old.member_level_ids)
            for replacement_id in supersession.replacement_cluster_ids:
                replacement = active_by_id.get(replacement_id)
                if replacement is not None:
                    member_ids.extend(replacement.member_level_ids)
            if self._foundation_plan_is_new(
                revisions,
                key=(
                    "liquidity_cluster_supersession",
                    supersession.superseded_cluster_id,
                    supersession.replacement_cluster_ids,
                    supersession.known_at,
                ),
                value=supersession,
            ):
                adapter.append_dto(
                    supersession,
                    source_event_ids=level_sources(member_ids),
                )
        return update.active

    def _foundation_relation_delivery(
        self,
        *,
        adapter: CanonicalFoundationAdapter,
        snapshot: MarketSnapshot,
        current_bar_event_id: str | None,
    ) -> None:
        if current_bar_event_id is None:
            return
        current_source = self._foundation_exact_event(current_bar_event_id)
        if (
            current_source.kind is not EventKind.BAR_COMPLETED
            or current_source.timeframe is not Timeframe.M1
            or current_source.known_at != snapshot.asof
        ):
            raise ValueError("foundation current observation root is not exact")
        active_structures = {
            timeframe: self._foundation_structure_at(
                adapter,
                timeframe,
                snapshot.asof,
            )
            for timeframe in self._active_timeframes
        }
        for relation in sorted(
            snapshot.relations.values(),
            key=lambda item: (item.parent_tf.value, item.child_tf.value),
        ):
            parent = active_structures.get(relation.parent_tf)
            child = active_structures.get(relation.child_tf)
            active = next(
                (
                    item
                    for item in adapter.lifecycle.relation_generations
                    if item.parent_tf is relation.parent_tf
                    and item.child_tf is relation.child_tf
                    and item.lifecycle is GenerationLifecycle.ACTIVE
                ),
                None,
            )
            if parent is None or child is None:
                if active is not None:
                    adapter.terminate_relation(
                        relation_generation_id=active.generation_id,
                        known_at=snapshot.asof,
                        reason=(
                            "parent_invalidated"
                            if parent is None
                            else "child_realigned"
                        ),
                        source_event_ids=(current_bar_event_id,),
                    )
                continue
            if relation.parent_direction is not parent.direction:
                # RelationState is a v1.2 compatibility snapshot whose parent
                # tracker may flip before the protected canonical EXTERNAL
                # owner terminates.  Foundation v2 freezes both endpoints as
                # confirmed EXTERNAL generations; child internal/MSS remains
                # relation evidence only.  A contradictory legacy parent
                # snapshot therefore cannot revise or replace the still-live
                # canonical relation generation.  Explicit owner termination
                # already closes dependent relations in the lifecycle reducer.
                continue
            cutoff_sources: list[str] = []
            cutoff_advanced = active is None
            for source_timeframe, cutoff in (
                (relation.parent_tf, relation.parent_source_cutoff),
                (relation.child_tf, relation.child_source_cutoff),
            ):
                if cutoff is None:
                    continue
                if active is None or cutoff > active.last_updated_at:
                    cutoff_advanced = True
                    cutoff_sources.append(
                        self._foundation_real_bar_event_id(
                            source_timeframe,
                            cutoff,
                        )
                    )
            same_generation = bool(
                active is not None
                and active.parent_structure_generation_id
                == parent.generation_id
                and active.child_structure_generation_id
                == child.generation_id
                and active.role == relation.role.value
            )
            # RelationState.known_at advances with the public M1 snapshot, but
            # that alias alone is not a semantic observation.  Preserve one
            # generation and revise it only when a bound parent/child native
            # cutoff advances.  A changed owner/role remains a real
            # reclassification and cites the current M1 root below.
            if same_generation and not cutoff_advanced:
                continue
            source_ids = self._foundation_unique_ids(
                (
                    parent.confirmation_event_id or "",
                    child.confirmation_event_id or "",
                    *cutoff_sources,
                    current_bar_event_id,
                )
            )
            adapter.observe_relation(
                relation,
                parent_structure_generation_id=parent.generation_id,
                child_structure_generation_id=child.generation_id,
                source_event_ids=source_ids,
            )

        for timeframe, state in sorted(
            snapshot.timeframe_states.items(),
            key=lambda item: item[0].value,
        ):
            try:
                native_bar_event_id = self._foundation_real_bar_event_id(
                    timeframe,
                    snapshot.asof,
                )
            except ValueError:
                # Delivery age is defined in real native completed bars, not
                # the M1 publication heartbeat or a synthetic scale clock.
                continue
            native_bar = self._foundation_exact_event(native_bar_event_id)
            if (
                native_bar.kind is not EventKind.BAR_COMPLETED
                or native_bar.timeframe is not timeframe
                or native_bar.known_at != snapshot.asof
                or native_bar.evidence.get("real_completed") is not True
                or native_bar.evidence.get("clock_only") is not False
            ):
                raise ValueError("delivery update lacks its exact native BAR")
            parent = active_structures.get(timeframe)
            active = next(
                (
                    item
                    for item in adapter.lifecycle.delivery_generations
                    if item.timeframe is timeframe
                    and item.lifecycle is GenerationLifecycle.ACTIVE
                ),
                None,
            )
            if parent is None:
                if active is not None:
                    adapter.terminate_delivery(
                        delivery_generation_id=active.generation_id,
                        known_at=snapshot.asof,
                        reason="parent_structure_terminated",
                        source_event_ids=(native_bar_event_id,),
                    )
                continue
            if (
                active is not None
                and active.parent_structure_generation_id
                != parent.generation_id
            ):
                adapter.terminate_delivery(
                    delivery_generation_id=active.generation_id,
                    known_at=snapshot.asof,
                    reason="parent_structure_terminated",
                    source_event_ids=(native_bar_event_id,),
                )
                active = None
            current_price_ticks = price_to_ticks(
                native_bar.evidence["close"],
                self.config.tick_size,
                name="delivery current price",
            )
            origin_event_id = (
                active.origin_event_id
                if active is not None
                and active.parent_structure_generation_id
                == parent.generation_id
                and active.phase == state.delivery.phase.value
                else native_bar_event_id
            )
            adapter.observe_delivery_phase(
                state.delivery.phase,
                timeframe=timeframe,
                known_at=snapshot.asof,
                parent_structure_generation_id=parent.generation_id,
                origin_event_id=origin_event_id,
                source_event_ids=self._foundation_unique_ids(
                    (
                        parent.confirmation_event_id or "",
                        origin_event_id,
                        native_bar_event_id,
                    )
                ),
                current_price_ticks=current_price_ticks,
            )

    def _foundation_reference_retirement(
        self,
        *,
        adapter: CanonicalFoundationAdapter,
        known_at: pd.Timestamp,
        clock_events: Sequence[MarketEvent],
        retirement_events: Sequence[MarketEvent],
    ) -> None:
        """Bind registered level retirement to exact canonical causes.

        Rearm is deliberately *not* projected here.  The foundation adapter
        owns the preregistered same-level departure rule.  A generic inventory
        disappearance is also deliberately insufficient: retained-swing cache
        capacity must never change canonical retirement.  A reference rollover
        cites its exact current normalized M1 BAR; a mature range boundary cites
        the exact DEALING_RANGE_INVALIDATED semantic fact.
        """

        real_bars = {
            event.timeframe: event
            for event in clock_events
            if event.kind is EventKind.BAR_COMPLETED
            and event.origin is EventOrigin.NORMALIZED_DATA
            and event.evidence.get("real_completed") is True
            and event.evidence.get("clock_only") is False
        }
        current_m1 = real_bars.get(Timeframe.M1)
        if current_m1 is not None:
            for retirement in retirement_events:
                if (
                    retirement.kind is not EventKind.LIQUIDITY_RETIRED
                    or retirement.transition_reason
                    != "reference_period_replaced"
                ):
                    continue
                if len(retirement.source_ids) != 1:
                    raise ValueError(
                        "reference retirement identity is ambiguous"
                    )
                source_identity = retirement.source_ids[0]
                matches = tuple(
                    level
                    for level in adapter.lifecycle.levels
                    if level.source_identity == source_identity
                )
                if not matches:
                    # A compatibility-only reference that never entered the
                    # canonical map has no foundation object to retire.
                    continue
                if len(matches) != 1:
                    raise ValueError(
                        "reference retirement identity is ambiguous"
                    )
                level = matches[0]
                if level.lifecycle in {
                    LiquidityLevelLifecycle.RETIRED,
                    LiquidityLevelLifecycle.ARCHIVED,
                }:
                    continue
                adapter.retire_level(
                    source_level_id=level.source_identity,
                    reason="reference_rollover",
                    source_event_ids=(current_m1.event_id,),
                    known_at=known_at,
                    timeframe=level.source_timeframe,
                )

        for invalidation in clock_events:
            if (
                invalidation.kind is not EventKind.DEALING_RANGE_INVALIDATED
                or invalidation.origin is not EventOrigin.SEMANTIC_ATOMIC
            ):
                continue
            range_id = invalidation.evidence.get("range_id")
            if not isinstance(range_id, str) or not range_id:
                raise ValueError(
                    "dealing-range invalidation lacks its range identity"
                )
            boundary_level_ids = tuple(
                level_id
                for (candidate_range_id, _), level_id
                in self._range_boundary_level_ids.items()
                if candidate_range_id == range_id
            )
            if not boundary_level_ids:
                # A forming range can fail before boundary inventory exists.
                continue
            self._foundation_retire_structural_levels(
                adapter=adapter,
                source_identities=boundary_level_ids,
                reason="source_range_terminated",
                cause_event_id=invalidation.event_id,
                known_at=known_at,
            )

    def _foundation_retire_structural_levels(
        self,
        *,
        adapter: CanonicalFoundationAdapter,
        source_identities: Sequence[str],
        reason: str,
        cause_event_id: str,
        known_at: pd.Timestamp,
    ) -> None:
        identities = frozenset(value for value in source_identities if value)
        for level in tuple(adapter.lifecycle.levels):
            normalized = level.source_identity.removeprefix("swing:")
            if (
                level.source_identity not in identities
                and normalized not in identities
            ):
                continue
            if level.lifecycle in {
                LiquidityLevelLifecycle.RETIRED,
                LiquidityLevelLifecycle.ARCHIVED,
            }:
                continue
            adapter.retire_level(
                source_level_id=level.source_identity,
                reason=reason,
                source_event_ids=(cause_event_id,),
                known_at=known_at,
                timeframe=level.source_timeframe,
            )

    def _foundation_expire_group3(
        self,
        *,
        adapter: CanonicalFoundationAdapter,
        known_at: pd.Timestamp,
        clock_events: Sequence[MarketEvent],
        processed_structure_ids: set[str],
        range_terminations: Sequence[tuple[str, str]],
    ) -> Group3Update | None:
        if self._group3_tracker is None:
            return None
        latest: Group3Update | None = None
        reset = next(
            (
                event
                for event in clock_events
                if event.kind is EventKind.MARKET_EPOCH_RESET
            ),
            None,
        )
        for generation in adapter.lifecycle.structure_generations:
            if (
                generation.scope is not StructureScope.EXTERNAL
                or generation.lifecycle
                is not StructureGenerationLifecycle.TERMINATED
                or generation.terminated_at != known_at
                or generation.generation_id in processed_structure_ids
            ):
                continue
            cause_event_id = generation.protected_acceptance_event_id
            if cause_event_id is None and reset is not None:
                cause_event_id = reset.event_id
            if cause_event_id is None:
                candidates = tuple(
                    event_id
                    for event_id in reversed(generation.source_event_ids)
                    if self._foundation_exact_event(event_id).known_at
                    == known_at
                )
                if not candidates:
                    raise ValueError(
                        "structure termination lacks an exact canonical cause"
                    )
                cause_event_id = candidates[0]
            self._foundation_exact_event(cause_event_id)
            latest = self._group3_tracker.expire_fvg_foundation_context(
                cause=FVGTerminationCause.PARENT_STRUCTURE_TERMINATED,
                related_entity_id=generation.generation_id,
                known_at=known_at,
                cause_event_id=cause_event_id,
            )
            for lifecycle in latest.fvg_structural_transitions:
                adapter.append_dto(lifecycle)
            processed_structure_ids.add(generation.generation_id)
        for range_id, cause_event_id in range_terminations:
            self._foundation_exact_event(cause_event_id)
            boundary_level_ids = tuple(
                level_id
                for (candidate_range_id, _), level_id
                in self._range_boundary_level_ids.items()
                if candidate_range_id == range_id
            )
            self._foundation_retire_structural_levels(
                adapter=adapter,
                source_identities=boundary_level_ids,
                reason="structure_generation_terminated",
                cause_event_id=cause_event_id,
                known_at=known_at,
            )
            latest = self._group3_tracker.expire_fvg_foundation_context(
                cause=FVGTerminationCause.STRUCTURAL_RANGE_REPLACED,
                related_entity_id=range_id,
                known_at=known_at,
                cause_event_id=cause_event_id,
            )
            for lifecycle in latest.fvg_structural_transitions:
                adapter.append_dto(lifecycle)
        if latest is not None:
            self._validate_group3_foundation_projection(latest)
        return latest

    def _stage_foundation_projection(
        self,
        *,
        asof: pd.Timestamp,
        frames: Mapping[Timeframe, FrameObservation],
        histories: Mapping[Timeframe, Sequence[Candle]],
        group3_update: Group3Update | None,
        group4_update: Group4Update | None,
        snapshot: MarketSnapshot,
        semantic_events: Sequence[MarketEvent],
    ):
        if self._foundation_adapter is None:
            raise RuntimeError("foundation staging requires Eye authority mode")
        candidate: CanonicalFoundationAdapter | None = None
        authoritative = tuple(
            sorted(
                (
                    event
                    for event in semantic_events
                    if event.origin
                    in {
                        EventOrigin.NORMALIZED_DATA,
                        EventOrigin.SEMANTIC_ATOMIC,
                    }
                    and event.kind is not EventKind.FOUNDATION_STATE_CHANGED
                ),
                key=event_order_key,
            )
        )
        plan_revisions = dict(
            getattr(self, "_foundation_plan_revisions", {})
        )
        dol_templates = (
            {}
            if any(
                event.kind is EventKind.MARKET_EPOCH_RESET
                for event in authoritative
            )
            else dict(getattr(self, "_foundation_dol_templates", {}))
        )
        for state in snapshot.timeframe_states.values():
            for template in state.liquidity.candidates:
                dol_templates.setdefault(
                    template.candidate_id,
                    foundation_dol_candidate_template(template),
                )
        for event in authoritative:
            if event.kind is not EventKind.PROTECTED_SWING_ASSIGNED:
                continue
            protected_swing_id = event.evidence.get("protected_swing_id")
            if (
                not isinstance(protected_swing_id, str)
                or not protected_swing_id
            ):
                raise ValueError(
                    "protected assignment lacks its swing identity"
                )
            source_identity = f"swing:{protected_swing_id}"
            template = dol_templates.get(source_identity)
            if template is not None:
                dol_templates[source_identity] = (
                    foundation_dol_protected_candidate_template(
                        template,
                        protected_swing_id=protected_swing_id,
                    )
                )
        if self._foundation_geometry_invalidated(authoritative):
            nodes, assignments, geometry_plans = (
                self._foundation_geometry_plans(
                    frames=frames,
                    histories=histories,
                    known_at=asof,
                    revisions=plan_revisions,
                )
            )
        else:
            nodes = self._foundation_geometry_nodes
            assignments = self._foundation_geometry_assignments
            geometry_plans = ()
        plans = [
            *geometry_plans,
            *self._foundation_leg_plans(frames, plan_revisions),
            *self._foundation_boundary_plans(frames, plan_revisions),
            *self._foundation_group3_plans(
                group3_update,
                plan_revisions,
            ),
            *self._foundation_balance_plans(
                group4_update,
                plan_revisions,
            ),
        ]
        plans_by_clock: dict[
            pd.Timestamp,
            list[tuple[int, str, object, tuple[str, ...] | None]],
        ] = {}
        for clock, priority, kind, value, source_ids in plans:
            plans_by_clock.setdefault(clock, []).append(
                (priority, kind, value, source_ids)
            )
        events_by_clock: dict[pd.Timestamp, list[MarketEvent]] = {}
        for event in authoritative:
            events_by_clock.setdefault(event.known_at, []).append(event)
        retirements_by_clock: dict[pd.Timestamp, list[MarketEvent]] = {}
        for event in semantic_events:
            if (
                event.kind is EventKind.LIQUIDITY_RETIRED
                and event.origin is EventOrigin.LEGACY_TRANSPORT
            ):
                retirements_by_clock.setdefault(event.known_at, []).append(
                    event
                )
        clocks = tuple(
            sorted(
                {
                    *plans_by_clock,
                    *events_by_clock,
                    *retirements_by_clock,
                }
            )
        )
        ranges = dict(self._foundation_structural_ranges)
        contexts = dict(self._foundation_fvg_contexts)
        processed_structure_ids = {
            generation.generation_id
            for generation in self._foundation_adapter.lifecycle.structure_generations
            if generation.lifecycle
            is StructureGenerationLifecycle.TERMINATED
        }
        prior_record_count = len(
            self._foundation_adapter.projection.records
        )
        latest_group3 = group3_update
        contextual_fvg_transitions: list[object] = []
        for clock in clocks:
            clock_events = tuple(
                sorted(events_by_clock.get(clock, ()), key=event_order_key)
            )
            candidate, _ = (
                self._foundation_adapter
                if candidate is None
                else candidate
            ).stage_batch(clock_events)
            self._foundation_reference_retirement(
                adapter=candidate,
                known_at=clock,
                clock_events=clock_events,
                retirement_events=retirements_by_clock.get(clock, ()),
            )
            ordered_clock_plans = tuple(
                sorted(
                    plans_by_clock.get(clock, ()),
                    key=lambda item: (
                        item[0],
                        item[1],
                        self._foundation_plan_sort_identity(item[2]),
                    ),
                )
            )
            # StructuralRange references exact SwingGeometryNode identities.
            # Nodes/assignments are the registered priorities 10/20, so append
            # them before same-clock range construction instead of relying on
            # a later plan in the batch to repair an orphan reference.
            for priority, kind, value, source_ids in ordered_clock_plans:
                if priority > 20:
                    continue
                if kind != "dto":
                    raise ValueError(
                        "foundation geometry priority must contain a DTO"
                    )
                record = FoundationProjectionReducer.record_from_dto(
                    value,
                    source_event_ids=source_ids,
                )
                if not candidate.contains_projection_record_id(
                    record.record_id
                ):
                    candidate.append_dto(
                        value,
                        source_event_ids=source_ids,
                    )
            ranges, range_terminations = (
                self._foundation_update_structural_ranges(
                    adapter=candidate,
                    ranges=ranges,
                    frames=frames,
                    known_at=clock,
                    clock_events=clock_events,
                    revisions=plan_revisions,
                )
            )
            for priority, kind, value, source_ids in ordered_clock_plans:
                if priority <= 20:
                    continue
                if kind == "boundary":
                    candidate.observe_boundary_attack(value)
                elif kind == "fvg":
                    lifecycle = self._foundation_bind_fvg_context(
                        adapter=candidate,
                        ranges=ranges,
                        lifecycle=value,
                        contexts=contexts,
                    )
                    self._foundation_plan_is_new(
                        plan_revisions,
                        key=(
                            "fvg_structural_lifecycle",
                            lifecycle.fvg_id,
                            lifecycle.last_updated_at,
                            lifecycle.availability.value,
                        ),
                        value=lifecycle,
                    )
                    record = FoundationProjectionReducer.record_from_dto(
                        lifecycle
                    )
                    if not candidate.contains_projection_record_id(
                        record.record_id
                    ):
                        candidate.append_dto(lifecycle)
                else:
                    record = FoundationProjectionReducer.record_from_dto(
                        value,
                        source_event_ids=source_ids,
                    )
                    if not candidate.contains_projection_record_id(
                        record.record_id
                    ):
                        candidate.append_dto(
                            value,
                            source_event_ids=source_ids,
                        )
            expired = self._foundation_expire_group3(
                adapter=candidate,
                known_at=clock,
                clock_events=clock_events,
                processed_structure_ids=processed_structure_ids,
                range_terminations=range_terminations,
            )
            if expired is not None:
                latest_group3 = expired
                contextual_fvg_transitions.extend(
                    expired.fvg_structural_transitions
                )

        if candidate is None:
            # A real observation normally contributes at least its normalized
            # M1 BAR.  Keep the helper total for compatibility fixtures whose
            # complete plan is empty without mutating the committed adapter.
            candidate, _ = self._foundation_adapter.stage_batch(())

        reset_in_update = any(
            event.kind is EventKind.MARKET_EPOCH_RESET
            for event in authoritative
        )
        current_bar_event_id: str | None
        try:
            current_bar_event_id = self._foundation_real_bar_event_id(
                Timeframe.M1,
                asof,
            )
        except ValueError:
            current_bar_event_id = None
        new_precluster_records = candidate.projection.records[
            prior_record_count:
        ]
        clusters_invalidated = (
            self._foundation_cluster_membership_invalidated(
                new_precluster_records,
                authoritative,
            )
        )
        clusters = self._foundation_update_clusters(
            adapter=candidate,
            known_at=asof,
            authoritative_events=authoritative,
            revisions=plan_revisions,
        ) if (
            (current_bar_event_id is not None or reset_in_update)
            and clusters_invalidated
        ) else tuple(self._foundation_active_clusters)
        foundation_states = foundation_dol_timeframe_states(
            candidate.projection,
            states=snapshot.timeframe_states,
            price=snapshot.price,
            candidate_templates=dol_templates,
            real_bar_ordinals={
                item.timeframe: item.count
                for item in candidate.lifecycle.real_bar_clocks
            },
        )
        snapshot = replace(
            snapshot,
            timeframe_states=foundation_states,
            relations=RelationResolver(
                edges=MarketSnapshotPublisher._RELATION_EDGES
            ).resolve(
                foundation_states,
                price=snapshot.price,
                asof=snapshot.asof,
            ),
        )
        self._foundation_relation_delivery(
            adapter=candidate,
            snapshot=snapshot,
            current_bar_event_id=current_bar_event_id,
        )
        if self._group3_tracker is not None and group3_update is not None:
            current_group3 = self._group3_tracker.current_update()
            transitions_by_identity = {
                (
                    item.fvg_id,
                    item.last_updated_at,
                    item.availability.value,
                ): item
                for item in (
                    *group3_update.fvg_structural_transitions,
                    *contextual_fvg_transitions,
                )
            }
            latest_group3 = replace(
                group3_update,
                base_origin_cores=current_group3.base_origin_cores,
                qualified_order_blocks=(
                    current_group3.qualified_order_blocks
                ),
                first_retests=current_group3.first_retests,
                fvg_structural_lifecycles=(
                    current_group3.fvg_structural_lifecycles
                ),
                fvg_structural_transitions=tuple(
                    transitions_by_identity.values()
                ),
            )
            self._validate_group3_foundation_projection(latest_group3)
        new_records = candidate.projection.records[prior_record_count:]
        if any(
            record.record_id in self._foundation_published_record_ids
            for record in new_records
        ):
            raise ValueError("foundation transport would republish a record")
        candidate.seal_staged_candidate()
        return (
            candidate,
            nodes,
            assignments,
            clusters,
            ranges,
            contexts,
            new_records,
            latest_group3,
            plan_revisions,
            dol_templates,
        )

    @staticmethod
    def _foundation_transport_timeframe(record: FoundationRecord) -> Timeframe:
        value = (
            record.payload.get("timeframe")
            or record.payload.get("source_timeframe")
            or record.payload.get("child_tf")
        )
        try:
            return Timeframe(value)
        except (TypeError, ValueError):
            return Timeframe.M1

    def _record_group3_events(
        self,
        update: Group3Update,
    ) -> None:
        for state in update.fvg_transitions:
            midpoint_revision = bool(
                state.lifecycle is FairValueGapLifecycle.PARTIAL
                and state.transition_reason == "midpoint_touched"
            )
            observed_at = (
                state.midpoint_touched_at
                if midpoint_revision
                else state.state_started_at
            )
            if observed_at is None:
                raise ValueError("FVG transition lacks its causal clock")
            terminal = state.lifecycle in {
                FairValueGapLifecycle.MITIGATED,
                FairValueGapLifecycle.INVALIDATED,
                FairValueGapLifecycle.EXPIRED,
            }
            fvg_state_event = _event(
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
                        "state_revision": midpoint_revision,
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
                        "midpoint_touched_at": (
                            None
                            if state.midpoint_touched_at is None
                            else state.midpoint_touched_at.isoformat()
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
                    entity_id=(None if midpoint_revision else state.fvg_id),
                    lifecycle=(
                        None
                        if midpoint_revision
                        else state.lifecycle.value
                    ),
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=observed_at if terminal else None,
                    direction=state.direction,
                    transition_reason=state.transition_reason,
                )
            self.memory.append(
                fvg_state_event,
                include_in_recent=False,
            )
            if state.lifecycle is FairValueGapLifecycle.OPEN:
                source_bar_events = tuple(
                    self._bar_event_id_for_candle_id(candle_id)
                    for candle_id in state.source_candle_ids
                )
                if len(source_bar_events) != 3:
                    raise ValueError(
                        "FVG creation requires exactly three completed-bar "
                        "source events"
                    )
                displacement_context_id = self._displacement_event_ids.get(
                    state.source_active_transition_id
                ) or self._displacement_event_ids.get(
                    state.source_displacement_id
                )
                created = self._append_semantic_atomic(
                    EventKind.FVG_CREATED,
                    state.confirmed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    source_bar_events,
                    {
                        "fvg_id": state.fvg_id,
                        "qualification": state.qualification.value,
                        "width_ticks": state.width_ticks,
                        "width_atr": state.width_atr,
                        "source_displacement_id": (
                            state.source_displacement_id
                        ),
                        "source_candle_ids": state.source_candle_ids,
                    },
                    direction=state.direction,
                    event_time=state.formed_at,
                    zone=(state.lower_bound, state.upper_bound),
                    source_data_ids=state.source_candle_ids,
                    source_entity_ids=(
                        state.fvg_id,
                        state.source_displacement_id,
                    ),
                    context_event_ids=(
                        fvg_state_event.event_id,
                        *((
                            displacement_context_id,
                        ) if displacement_context_id else ()),
                    ),
                )
                self._fvg_created_event_ids[state.fvg_id] = (
                    created.event_id
                )
            elif state.lifecycle in {
                FairValueGapLifecycle.PARTIAL,
                FairValueGapLifecycle.MITIGATED,
                FairValueGapLifecycle.INVALIDATED,
                FairValueGapLifecycle.EXPIRED,
            }:
                created_event_id = self._fvg_created_event_ids.get(
                    state.fvg_id
                )
                if created_event_id is None:
                    raise ValueError(
                        "FVG lifecycle transition lacks its creation event"
                    )
                lifecycle_kind = {
                    FairValueGapLifecycle.PARTIAL: (
                        EventKind.FVG_MIDPOINT_TOUCHED
                        if midpoint_revision
                        else EventKind.FVG_PARTIALLY_FILLED
                    ),
                    FairValueGapLifecycle.MITIGATED: (
                        EventKind.FVG_FULLY_FILLED
                    ),
                    FairValueGapLifecycle.INVALIDATED: (
                        EventKind.FVG_INVALIDATED
                    ),
                    FairValueGapLifecycle.EXPIRED: EventKind.FVG_EXPIRED,
                }[state.lifecycle]
                try:
                    transition_bar_event_id = self._bar_event_id_at(
                        Timeframe.M5,
                        observed_at,
                    )
                except ValueError:
                    transition_bar_event_id = None
                source_event_ids = (
                    created_event_id,
                    *((
                        transition_bar_event_id,
                    ) if transition_bar_event_id else ()),
                )
                evidence = {
                    "fvg_id": state.fvg_id,
                    "lifecycle": state.lifecycle.value,
                    "max_fill_fraction": state.max_fill_fraction,
                    "midpoint_touched": bool(
                        state.midpoint_touched_at is not None
                    ),
                    "midpoint_touched_at": (
                        None
                        if state.midpoint_touched_at is None
                        else state.midpoint_touched_at.isoformat()
                    ),
                    "fully_filled": bool(
                        state.lifecycle
                        is FairValueGapLifecycle.MITIGATED
                    ),
                    "transition_reason": state.transition_reason,
                }
                lifecycle_event = self._append_semantic_atomic(
                    lifecycle_kind,
                    observed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    source_event_ids,
                    evidence,
                    direction=state.direction,
                    event_time=observed_at,
                    zone=(state.lower_bound, state.upper_bound),
                    source_entity_ids=(state.fvg_id,),
                    context_event_ids=(fvg_state_event.event_id,),
                )
                if state.lifecycle in {
                    FairValueGapLifecycle.INVALIDATED,
                    FairValueGapLifecycle.EXPIRED,
                }:
                    self._fvg_terminal_event_ids[state.fvg_id] = (
                        lifecycle_event.event_id
                    )
        for state in update.order_block_transitions:
            observed_at = state.state_started_at
            terminal = state.lifecycle in {
                OrderBlockLifecycle.MITIGATED,
                OrderBlockLifecycle.FAILED,
            }
            order_block_state_event = _event(
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
                )
            self.memory.append(
                order_block_state_event,
                include_in_recent=False,
            )
            created_event_id = self._origin_zone_created_event_ids.get(
                state.order_block_id
            )
            if state.lifecycle in {
                OrderBlockLifecycle.CREATED,
                OrderBlockLifecycle.UNTESTED,
            } and created_event_id is None:
                displacement_event_id = self._displacement_event_ids.get(
                    state.source_active_transition_id
                ) or self._displacement_event_ids.get(
                    state.source_displacement_id
                )
                raw_break_event_id = self._raw_break_event_ids.get(
                    state.source_bos_id
                )
                if (
                    displacement_event_id is None
                    or raw_break_event_id is None
                ):
                    raise ValueError(
                        "origin zone lacks its exact displacement or raw "
                        "boundary-break event"
                    )
                anchor_bar_events = tuple(
                    self._bar_event_id_for_candle_id(candle_id)
                    for candle_id in state.anchor_candle_ids
                )
                created = self._append_semantic_atomic(
                    EventKind.ORIGIN_ZONE_CREATED,
                    state.confirmed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    (
                        displacement_event_id,
                        raw_break_event_id,
                        *anchor_bar_events,
                    ),
                    {
                        "origin_zone_id": state.order_block_id,
                        "geometry": "frozen_group3_order_block_range",
                        "source_displacement_id": (
                            state.source_displacement_id
                        ),
                        "source_bos_id": state.source_bos_id,
                        "anchor_candle_ids": state.anchor_candle_ids,
                        "lifecycle": state.lifecycle.value,
                    },
                    direction=state.direction,
                    event_time=state.formed_at,
                    zone=(state.lower_bound, state.upper_bound),
                    source_data_ids=state.anchor_candle_ids,
                    source_entity_ids=(
                        state.order_block_id,
                        state.source_displacement_id,
                        state.source_bos_id,
                    ),
                    context_event_ids=(order_block_state_event.event_id,),
                )
                self._origin_zone_created_event_ids[
                    state.order_block_id
                ] = created.event_id
            elif state.lifecycle in {
                OrderBlockLifecycle.MITIGATED,
                OrderBlockLifecycle.FAILED,
            }:
                if created_event_id is None:
                    raise ValueError(
                        "origin-zone terminal transition lacks creation event"
                    )
                try:
                    transition_bar_event_id = self._bar_event_id_at(
                        Timeframe.M5,
                        observed_at,
                    )
                except ValueError:
                    transition_bar_event_id = None
                source_events = (
                    created_event_id,
                    *((
                        transition_bar_event_id,
                    ) if transition_bar_event_id else ()),
                )
                self._append_semantic_atomic(
                    (
                        EventKind.ORIGIN_ZONE_MITIGATED
                        if state.lifecycle
                        is OrderBlockLifecycle.MITIGATED
                        else EventKind.ORIGIN_ZONE_INVALIDATED
                    ),
                    observed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    source_events,
                    {
                        "origin_zone_id": state.order_block_id,
                        "lifecycle": state.lifecycle.value,
                        "transition_reason": state.transition_reason,
                    },
                    direction=state.direction,
                    event_time=observed_at,
                    zone=(state.lower_bound, state.upper_bound),
                    source_entity_ids=(state.order_block_id,),
                    context_event_ids=(order_block_state_event.event_id,),
                )

    def _record_displacement_events(
        self,
        displacement,
    ) -> None:
        if displacement is None:
            return
        for transition in displacement.transitions_this_update:
            if not self._remember_bounded(
                transition.transition_id,
                known=self._known_displacement_transition_ids,
                order=self._known_displacement_transition_order,
            ):
                continue
            metrics = {
                str(name): float(value)
                for name, value in transition.state_metrics
            }
            source_bar_events = tuple(
                self._bar_event_id_for_candle_id(candle_id)
                for candle_id in transition.admitted_candle_ids
            )
            source_bar_facts = tuple(
                self.memory.audit_event_including_pending(event_id)
                for event_id in source_bar_events
            )
            if not source_bar_facts or any(
                event is None
                or event.origin is not EventOrigin.NORMALIZED_DATA
                or event.kind is not EventKind.BAR_COMPLETED
                or event.timeframe is not Timeframe.M5
                or event.evidence.get("real_completed") is not True
                or event.evidence.get("clock_only") is not False
                or not isinstance(
                    event.evidence.get("detector_candle_id"), str
                )
                or not event.evidence.get("detector_candle_id")
                for event in source_bar_facts
            ):
                raise ValueError(
                    "displacement semantic source BAR detector lineage is invalid"
                )
            source_bar_detector_ids = tuple(
                str(event.evidence["detector_candle_id"])
                for event in source_bar_facts
                if event is not None
            )
            if (
                len(source_bar_detector_ids)
                != len(set(source_bar_detector_ids))
                or len(transition.admitted_candle_ids)
                != len(set(transition.admitted_candle_ids))
                or set(source_bar_detector_ids)
                != set(transition.admitted_candle_ids)
            ):
                raise ValueError(
                    "displacement admitted detector candle lineage is invalid"
                )
            synthetic_context_event_ids: tuple[str, ...] = ()
            if (
                transition.lifecycle == "censored"
                and transition.reason == "synthetic_interruption"
            ):
                synthetic_context_event_ids = (
                    self._synthetic_m1_context_event_ids_for_m5_terminal(
                        transition.observed_at,
                    )
                )
            displacement_event = self._append_semantic_atomic(
                EventKind.DISPLACEMENT_OBSERVED,
                transition.observed_at,
                Timeframe.M5,
                (
                    "above"
                    if transition.direction is Direction.LONG
                    else "below"
                ),
                None,
                clamp(metrics.get("efficiency", 0.0)),
                source_bar_events,
                {
                    "transition_id": transition.transition_id,
                    "displacement_id": transition.entity_id,
                    "lifecycle": transition.lifecycle,
                    "terminal_reason": transition.reason,
                    "state_metrics": metrics,
                    "admitted_candle_ids": (
                        transition.admitted_candle_ids
                    ),
                    "prefix_last_admitted_at": (
                        None
                        if transition.prefix_last_admitted_at is None
                        else transition.prefix_last_admitted_at.isoformat()
                    ),
                },
                direction=transition.direction,
                event_time=(
                    transition.started_at or transition.observed_at
                ),
                source_data_ids=transition.admitted_candle_ids,
                source_entity_ids=(transition.entity_id,),
                context_event_ids=synthetic_context_event_ids,
            )
            self._displacement_event_ids[transition.transition_id] = (
                displacement_event.event_id
            )
            self._displacement_event_ids[transition.entity_id] = (
                displacement_event.event_id
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
        boundary_reason = update.boundary_reason
        if boundary_reason is not None and any(
            state.lifecycle is not DealingRangeLifecycle.BROKEN
            or state.broken_at != state.state_started_at
            or state.transition_reason != boundary_reason
            for state in update.range_transitions
        ):
            raise ValueError(
                "Group 4 boundary range transition is not an exact terminal"
            )
        if (
            boundary_reason is None
            and include_creations
            and self._prior is not None
        ):
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
            range_state_event = _event(
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
                )
            self.memory.append(
                range_state_event,
                include_in_recent=False,
            )
            if boundary_reason is not None:
                # Contract/data-gap resets censor the previous market epoch;
                # they are not a market-observed H1 acceptance.  Preserve the
                # lifecycle timeline transport, while the MARKET_EPOCH_RESET
                # event owns the authoritative causal transition.
                continue
            range_kind = {
                DealingRangeLifecycle.FORMING: (
                    EventKind.DEALING_RANGE_CREATED
                ),
                DealingRangeLifecycle.MATURE: (
                    EventKind.DEALING_RANGE_ACTIVATED
                ),
                DealingRangeLifecycle.BROKEN: (
                    EventKind.DEALING_RANGE_INVALIDATED
                ),
            }[state.lifecycle]
            created_event_id = self._range_created_event_ids.get(
                state.range_id
            )
            active_event_id = self._range_active_event_ids.get(
                state.range_id
            )
            anchor_event_ids = tuple(
                dict.fromkeys(
                    event_id
                    for event_id in (
                        self._candidate_level_event_ids.get(
                            state.lower_source_zone_id
                        ),
                        self._candidate_level_event_ids.get(
                            state.upper_source_zone_id
                        ),
                        *(
                            self._confirmed_swing_event_ids.get(swing_id)
                            for swing_id in (
                                *state.lower_source_member_swing_ids,
                                *state.upper_source_member_swing_ids,
                            )
                        ),
                    )
                    if event_id is not None
                )
            )
            try:
                transition_bar_event_id = self._bar_event_id_at(
                    Timeframe.H1,
                    state.state_started_at,
                )
            except ValueError:
                transition_bar_event_id = None
            if (
                state.lifecycle is not DealingRangeLifecycle.FORMING
                and created_event_id is None
                and transition_bar_event_id is None
            ):
                # Private compatibility callers can project an isolated range
                # lifecycle state without its normalized history.  Keep only
                # the legacy lifecycle transport in that case; an authoritative
                # semantic transition may never invent missing ancestry.
                continue
            external_acceptance_event_id: str | None = None
            if (
                state.lifecycle is DealingRangeLifecycle.BROKEN
                and state.transition_reason == "close_beyond_frozen_range"
                and active_event_id is not None
            ):
                if transition_bar_event_id is None:
                    raise ValueError(
                        "active dealing-range acceptance lacks its exact "
                        "completed H1 BAR event"
                    )
                accepted_close = self._bar_close_by_event_id.get(
                    transition_bar_event_id
                )
                if accepted_close is None:
                    raise ValueError(
                        "active dealing-range acceptance lacks its frozen "
                        "completed H1 close"
                    )
                if accepted_close < state.lower_bound:
                    accepted_side = "below"
                    accepted_price = float(state.lower_bound)
                    accepted_direction = Direction.SHORT
                elif accepted_close > state.upper_bound:
                    accepted_side = "above"
                    accepted_price = float(state.upper_bound)
                    accepted_direction = Direction.LONG
                else:
                    raise ValueError(
                        "close-beyond range transition does not close "
                        "strictly outside its frozen bounds"
                    )
                level_id = self._range_boundary_level_ids.get(
                    (state.range_id, accepted_side)
                )
                if level_id is None:
                    raise ValueError(
                        "active dealing-range acceptance lacks its frozen "
                        "candidate boundary identity"
                    )
                candidate_event_id = self._candidate_level_event_ids.get(
                    level_id
                )
                if candidate_event_id is None:
                    raise ValueError(
                        "active dealing-range boundary lacks its candidate "
                        "creation event"
                    )
                crossing_generation_id = self._crossing_generation_id(
                    level_id=level_id,
                    timeframe=Timeframe.H1,
                    crossed_at=state.state_started_at,
                )
                touch_event = self._append_semantic_atomic(
                    EventKind.LEVEL_TOUCHED,
                    state.state_started_at,
                    Timeframe.H1,
                    accepted_side,
                    accepted_price,
                    state.strength,
                    (candidate_event_id, transition_bar_event_id),
                    {
                        "level_id": level_id,
                        "range_id": state.range_id,
                        "source_kind": "mature_range_boundary",
                        "touch_reason": "external_h1_close_crossing",
                    },
                    event_time=state.state_started_at,
                    zone=(accepted_price, accepted_price),
                    source_entity_ids=(level_id, state.range_id),
                )
                penetrated_event = self._append_semantic_atomic(
                    EventKind.LEVEL_PENETRATED,
                    state.state_started_at,
                    Timeframe.H1,
                    accepted_side,
                    accepted_price,
                    state.strength,
                    (
                        candidate_event_id,
                        touch_event.event_id,
                        transition_bar_event_id,
                    ),
                    {
                        "level_id": level_id,
                        "range_id": state.range_id,
                        "source_kind": "mature_range_boundary",
                        "penetration_standard": (
                            "first_completed_h1_close_strictly_outside_"
                            "frozen_range"
                        ),
                        "crossing_generation_id": crossing_generation_id,
                        "crossed_at": state.state_started_at.isoformat(),
                        "accepted_close": float(accepted_close),
                    },
                    direction=accepted_direction,
                    event_time=state.state_started_at,
                    zone=(accepted_price, accepted_price),
                    source_entity_ids=(level_id, state.range_id),
                )
                accepted_event = self._append_crossing_resolution(
                    EventKind.ACCEPTANCE_CONFIRMED,
                    state.state_started_at,
                    Timeframe.H1,
                    accepted_side,
                    accepted_price,
                    state.strength,
                    (
                        penetrated_event.event_id,
                        transition_bar_event_id,
                    ),
                    {
                        "level_id": level_id,
                        "range_id": state.range_id,
                        "source_kind": "mature_range_boundary",
                        "acceptance_bars": 1,
                        "accepted_close": float(accepted_close),
                        "resolution_standard": (
                            "first_completed_h1_close_strictly_outside_"
                            "frozen_range"
                        ),
                    },
                    direction=accepted_direction,
                    crossed_at=state.state_started_at,
                    zone=(accepted_price, accepted_price),
                )
                external_acceptance_event_id = accepted_event.event_id
            if state.lifecycle is DealingRangeLifecycle.FORMING:
                range_sources = anchor_event_ids
            else:
                if created_event_id is None:
                    raise ValueError(
                        "dealing-range transition lacks its creation event"
                    )
                range_sources = (
                    created_event_id,
                    *((active_event_id,) if active_event_id else ()),
                    *((
                        transition_bar_event_id,
                    ) if transition_bar_event_id else ()),
                    *((
                        external_acceptance_event_id,
                    ) if external_acceptance_event_id else ()),
                )
            semantic_transition_reason = state.transition_reason
            if (
                state.lifecycle is DealingRangeLifecycle.BROKEN
                and state.transition_reason == "close_beyond_frozen_range"
                and active_event_id is None
            ):
                # A forming candidate can fail before it ever becomes the
                # active range.  It has no activated boundary inventory and
                # therefore cannot manufacture the active-range Acceptance
                # ancestry used by a mature range invalidation.
                semantic_transition_reason = (
                    "close_beyond_frozen_range_before_activation"
                )
            range_event = self._append_semantic_atomic(
                range_kind,
                state.state_started_at,
                Timeframe.H1,
                None,
                state.midpoint,
                state.strength,
                range_sources,
                {
                    "range_id": state.range_id,
                    "lifecycle": state.lifecycle.value,
                    "lower_bound": state.lower_bound,
                    "upper_bound": state.upper_bound,
                    "normalized_location_unclamped": None,
                    "lower_source_zone_id": (
                        state.lower_source_zone_id
                    ),
                    "upper_source_zone_id": (
                        state.upper_source_zone_id
                    ),
                    "source_member_swing_ids": (
                        *state.lower_source_member_swing_ids,
                        *state.upper_source_member_swing_ids,
                    ),
                    "transition_reason": semantic_transition_reason,
                },
                event_time=(
                    state.formed_at
                    if state.lifecycle is DealingRangeLifecycle.FORMING
                    else state.state_started_at
                ),
                zone=(state.lower_bound, state.upper_bound),
                source_entity_ids=(
                    state.range_id,
                    state.lower_source_zone_id,
                    state.upper_source_zone_id,
                    *state.lower_source_member_swing_ids,
                    *state.upper_source_member_swing_ids,
                ),
                context_event_ids=(range_state_event.event_id,),
            )
            if state.lifecycle is DealingRangeLifecycle.FORMING:
                self._range_created_event_ids[state.range_id] = (
                    range_event.event_id
                )
                if self._last_invalidated_range_event_id is not None:
                    self._append_semantic_atomic(
                        EventKind.DEALING_RANGE_REPLACED,
                        state.state_started_at,
                        Timeframe.H1,
                        None,
                        state.midpoint,
                        state.strength,
                        (
                            self._last_invalidated_range_event_id,
                            range_event.event_id,
                        ),
                        {
                            "replacement_range_id": state.range_id,
                            "replacement_standard": (
                                "new_registered_range_after_invalidation"
                            ),
                        },
                        event_time=state.formed_at,
                        zone=(state.lower_bound, state.upper_bound),
                    )
                    self._last_invalidated_range_event_id = None
            elif state.lifecycle is DealingRangeLifecycle.MATURE:
                self._range_active_event_ids[state.range_id] = (
                    range_event.event_id
                )
                boundary_items = tuple(
                    item
                    for item in update.range_boundary_inventory
                    if item.kind == "range_boundary"
                    and state.range_id in item.source_ids
                )
                if {item.side for item in boundary_items} != {
                    "above",
                    "below",
                }:
                    if transition_bar_event_id is not None:
                        raise ValueError(
                            "active dealing range lacks its two frozen "
                            "boundary inventory identities"
                        )
                    # Private legacy projector fixtures may exercise the
                    # lifecycle transport without first registering normalized
                    # BAR roots.  They are not an authoritative semantic DAG,
                    # so do not manufacture boundary identities for them.
                    boundary_items = ()
                for item in boundary_items:
                    self._range_boundary_level_ids[
                        (state.range_id, item.side)
                    ] = item.item_id
                    if item.item_id in self._candidate_level_event_ids:
                        continue
                    candidate = self._append_semantic_atomic(
                        EventKind.LIQUIDITY_LEVEL_CREATED,
                        state.state_started_at,
                        Timeframe.H1,
                        item.side,
                        item.price,
                        item.strength,
                        (range_event.event_id,),
                        {
                            "level_id": item.item_id,
                            "range_id": state.range_id,
                            "candidate_only": True,
                            "source_kind": "mature_range_boundary",
                            "source_ids": item.source_ids,
                            "source_confirmed_at": (
                                item.confirmed_at.isoformat()
                            ),
                        },
                        event_time=item.confirmed_at,
                        zone=(item.lower_bound, item.upper_bound),
                        source_entity_ids=(
                            item.item_id,
                            state.range_id,
                            *item.source_ids,
                        ),
                    )
                    self._candidate_level_event_ids[item.item_id] = (
                        candidate.event_id
                    )
            elif state.lifecycle is DealingRangeLifecycle.BROKEN:
                self._range_terminal_event_ids[state.range_id] = (
                    range_event.event_id
                )
                self._last_invalidated_range_event_id = (
                    range_event.event_id
                )

        # A hard reset has already replaced EventMemory and imported only
        # transitionable old-epoch prefixes.  Join a range's BROKEN transition
        # to that exact identity at the boundary clock, but do not reinterpret
        # a boundary-censored manipulation (whose reducer lifecycle is still
        # SWEPT) as a new creation in the fresh epoch.
        if boundary_reason is not None:
            return

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
            manipulation_state_event = _event(
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
                )
            self.memory.append(
                manipulation_state_event,
                include_in_recent=False,
                sequence_floor=(
                    None
                    if terminal
                    else EventMemory._GROUP4_CREATION_SEQUENCE_FLOOR
                ),
            )
            if terminal and not state.deadline_elapsed:
                accepted = (
                    state.lifecycle
                    is ManipulationLifecycle.ACCEPTED_OUTSIDE
                )
                penetration_event_id = self._penetration_event_ids.get(
                    self._penetration_key(
                        level_id=state.source_inventory_item_id,
                        timeframe=Timeframe.M1,
                        crossed_at=state.formed_at,
                    )
                )
                if penetration_event_id is None:
                    if not self._real_bar_event_ids_by_timeframe[Timeframe.M1]:
                        # Legacy private lifecycle fixtures do not register
                        # normalized roots and therefore cannot publish an
                        # authoritative terminal semantic.
                        continue
                    raise ValueError(
                        "Group 4 terminal resolution lacks its canonical "
                        "penetration event"
                    )
                resolution_bar_event_id = self._bar_event_id_at(
                    Timeframe.M1,
                    state.resolved_at,
                )
                self._append_crossing_resolution(
                    (
                        EventKind.ACCEPTANCE_CONFIRMED
                        if accepted
                        else EventKind.SWEEP_CONFIRMED
                    ),
                    state.resolved_at,
                    Timeframe.M1,
                    state.side,
                    (
                        state.reentry_price
                        if state.reentry_price is not None
                        else state.sweep_extreme
                    ),
                    state.strength,
                    (penetration_event_id, resolution_bar_event_id),
                    {
                        "level_id": state.source_inventory_item_id,
                        "manipulation_id": state.manipulation_id,
                        "source_kind": state.source_kind,
                        "source_timeframe": state.source_timeframe.value,
                        "penetration_atr": state.penetration_atr,
                        "outside_completed_bars": (
                            state.outside_completed_bars
                        ),
                        "outside_run": state.outside_run,
                        "inside_hold_bars": state.inside_hold_bars,
                        "resolution": (
                            "registered_outside_acceptance"
                            if accepted
                            else "registered_reacceptance"
                        ),
                    },
                    direction=(
                        (
                            Direction.LONG
                            if state.side == "above"
                            else Direction.SHORT
                        )
                        if accepted
                        else (
                            Direction.SHORT
                            if state.side == "above"
                            else Direction.LONG
                        )
                    ),
                    crossed_at=state.formed_at,
                    zone=(
                        state.source_lower_bound,
                        state.source_upper_bound,
                    ),
                    context_event_ids=(manipulation_state_event.event_id,),
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
            projected = self._visible_group3_update(
                self._group3_tracker.on_boundary(
                    boundary,
                    update.asof,
                    foundation_boundary_event_id=(
                        self._last_market_epoch_reset_event_id
                    ),
                )
            )
            self._validate_group3_foundation_projection(projected)
            return projected
        batch = self._displacement_eye.last_batch
        expected = tuple(
            update.newly_completed.get(Timeframe.M5, ())
        )
        if tuple(candle for candle, _ in batch) != expected:
            raise RuntimeError(
                "Group 3 and displacement completed-M5 batches diverged"
            )
        result = self._group3_tracker.current_update()
        order_block_funnel = []
        fvg_transitions = []
        order_block_transitions = []
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
                group4_sources_only=True,
                include_support_resistance=(timeframe is Timeframe.H1),
            )
            if self.config.group4_projection_only
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

        if self.config.group4_projection_only:
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
            if item.item_id in self._candidate_level_event_ids:
                continue
            source = self._reference_candidate_sources.get(item.item_id)
            if source is None:
                raise ValueError(
                    "reference candidate lacks its frozen source clocks"
                )
            extreme_bar_event_id = self._bar_event_id_at(
                Timeframe.M1,
                source.extreme_at,
            )
            admission_bar_event_id = self._bar_event_id_at(
                Timeframe.M1,
                source.admitted_at,
            )
            evidence: dict[str, object] = {
                "level_id": item.item_id,
                "candidate_only": True,
                "source_kind": item.kind,
                "source_inventory_kind": item.kind,
                "source_ids": item.source_ids,
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
                else self._candidate_level_event_ids.get(
                    source.replaces_level_id
                )
            )
            if replacement_event_id is not None:
                evidence["replaces_level_id"] = source.replaces_level_id
                evidence["replaces_level_event_id"] = replacement_event_id
            candidate = self._append_semantic_atomic(
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
            self._candidate_level_event_ids[item.item_id] = candidate.event_id

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
        defer_resolution: bool = False,
    ) -> None:
        if self.config.group4_projection_only:
            return
        semantic_source_kind = (
            "confirmed_swing" if item.kind == "swing" else item.kind
        )
        outside = self._pool_close_outside(item, candle)
        extreme = candle.high if item.side == "above" else candle.low
        distance = (
            extreme - item.upper_bound
            if item.side == "above"
            else item.lower_bound - extreme
        )
        try:
            bar_event_id = self._bar_event_id_at(Timeframe.M1, candle.end)
        except ValueError:
            # Direct compatibility projection callers may intentionally omit
            # the normalized event stream.  Preserve their legacy lifecycle
            # transport, but never manufacture a canonical semantic fact
            # without a BAR_COMPLETED root.
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
            return
        candidate_event_id = self._candidate_level_event_ids.get(
            item.item_id
        )
        if candidate_event_id is None:
            source_events = tuple(
                event_id
                for source_id in item.source_ids
                if (
                    event_id
                    := self._confirmed_swing_event_ids.get(source_id)
                )
            )
            if not source_events:
                try:
                    source_events = (
                        self._bar_event_id_at(
                            item.timeframe,
                            item.confirmed_at,
                        ),
                    )
                except ValueError:
                    source_events = ()
            candidate = self._append_semantic_atomic(
                EventKind.LIQUIDITY_LEVEL_CREATED,
                candle.end,
                item.timeframe,
                item.side,
                item.price,
                item.strength,
                source_events,
                {
                    "level_id": item.item_id,
                    "candidate_only": True,
                    "source_kind": semantic_source_kind,
                    "source_inventory_kind": item.kind,
                    "source_ids": item.source_ids,
                    "admitted_from_inventory": True,
                    "source_confirmed_at": item.confirmed_at.isoformat(),
                },
                event_time=item.formed_at,
                zone=(item.lower_bound, item.upper_bound),
                source_entity_ids=(item.item_id, *item.source_ids),
            )
            candidate_event_id = candidate.event_id
            self._candidate_level_event_ids[item.item_id] = (
                candidate_event_id
            )
        touch_identity = f"{item.item_id}|{candle.end.isoformat()}"
        touch_event_id = self._level_touch_event_ids.get(
            (item.item_id, pd.Timestamp(candle.end))
        )
        if self._remember_bounded(
            touch_identity,
            known=self._known_level_touch_ids,
            order=self._known_level_touch_order,
        ):
            touch_event = self._append_semantic_atomic(
                EventKind.LEVEL_TOUCHED,
                candle.end,
                Timeframe.M1,
                item.side,
                item.price,
                item.strength,
                (candidate_event_id, bar_event_id),
                {
                    "level_id": item.item_id,
                    "source_timeframe": item.timeframe.value,
                    "source_kind": semantic_source_kind,
                    "source_inventory_kind": item.kind,
                },
                event_time=candle.end,
                zone=(item.lower_bound, item.upper_bound),
            )
            touch_event_id = touch_event.event_id
            self._level_touch_event_ids[
                (item.item_id, pd.Timestamp(candle.end))
            ] = touch_event_id
        if touch_event_id is None:
            raise ValueError(
                "level penetration lacks its exact touch event"
            )
        crossing_generation_id = self._crossing_generation_id(
            level_id=item.item_id,
            timeframe=Timeframe.M1,
            crossed_at=candle.end,
        )
        penetration = self._append_semantic_atomic(
            EventKind.LEVEL_PENETRATED,
            candle.end,
            Timeframe.M1,
            item.side,
            extreme,
            clamp(distance / max(atr, self.config.tick_size)),
            (candidate_event_id, touch_event_id, bar_event_id),
            {
                "level_id": item.item_id,
                "source_timeframe": item.timeframe.value,
                "source_kind": semantic_source_kind,
                "source_inventory_kind": item.kind,
                "penetration_points": max(0.0, distance),
                "close_accepted_outside": outside,
                "frozen_lower_bound": item.lower_bound,
                "frozen_upper_bound": item.upper_bound,
                "penetration_standard": (
                    "intrabar_trade_beyond_frozen_candidate_level"
                ),
                "strict_close_beyond_confirmed_swing_price": bool(
                    item.kind == "swing" and outside
                ),
                "crossing_generation_id": crossing_generation_id,
                "crossed_at": candle.end.isoformat(),
            },
            direction=(
                Direction.LONG
                if item.side == "above"
                else Direction.SHORT
            ),
            event_time=candle.end,
            zone=(item.lower_bound, item.upper_bound),
        )
        self._penetration_event_ids[
            self._penetration_key(
                level_id=item.item_id,
                timeframe=Timeframe.M1,
                crossed_at=candle.end,
            )
        ] = penetration.event_id
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
        if not outside and not defer_resolution:
            self._append_crossing_resolution(
                EventKind.SWEEP_CONFIRMED,
                candle.end,
                Timeframe.M1,
                item.side,
                extreme,
                clamp(distance / max(atr, self.config.tick_size)),
                (penetration.event_id, bar_event_id),
                {
                    "level_id": item.item_id,
                    "resolution_bars": 0,
                    "resolution": "same_bar_close_returned_inside",
                    "penetration_points": max(0.0, distance),
                },
                direction=(
                    Direction.SHORT
                    if item.side == "above"
                    else Direction.LONG
                ),
                crossed_at=candle.end,
                zone=(item.lower_bound, item.upper_bound),
            )

    def _append_projected_pool_sweep_events(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
        *,
        atr: float,
    ) -> None:
        if self.config.group4_projection_only:
            return
        self._append_inventory_crossing_event(
            item,
            candle,
            atr=atr,
            defer_resolution=True,
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
        *,
        crossed_at: pd.Timestamp | None = None,
    ) -> None:
        if self.config.group4_projection_only:
            return
        outside = self._pool_close_outside(item, candle)
        resolution_state_event = _event(
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
        self.memory.append(resolution_state_event)
        try:
            resolution_bar_event_id = self._bar_event_id_at(
                Timeframe.M1,
                candle.end,
            )
        except ValueError:
            return
        penetration_event_id = (
            None
            if crossed_at is None
            else self._penetration_event_ids.get(
                self._penetration_key(
                    level_id=item.item_id,
                    timeframe=Timeframe.M1,
                    crossed_at=crossed_at,
                )
            )
        )
        if self._group4_tracker is not None:
            # A source admitted by registered Group 4 is resolved solely by
            # that manipulation protocol.  Rejected/unadmitted pool sources
            # still need the generic crossing terminal; otherwise a published
            # penetration would remain permanently unresolved.
            group4_claims_source = any(
                state.source_inventory_item_id == item.item_id
                for state in self._group4_tracker.snapshot().manipulations
            )
            if group4_claims_source:
                return
        if penetration_event_id is None:
            raise ValueError(
                "projected pool resolution lacks its canonical penetration"
            )
        if crossed_at is None:
            raise ValueError(
                "canonical projected pool resolution requires its original "
                "crossing clock"
            )
        self._append_crossing_resolution(
            (
                EventKind.ACCEPTANCE_CONFIRMED
                if outside
                else EventKind.SWEEP_CONFIRMED
            ),
            candle.end,
            Timeframe.M1,
            item.side,
            candle.close,
            item.strength,
            (penetration_event_id, resolution_bar_event_id),
            {
                "level_id": item.item_id,
                "source_timeframe": item.timeframe.value,
                "resolution_bars": 1,
                "resolution": (
                    "later_close_held_outside"
                    if outside
                    else "later_close_returned_inside"
                ),
            },
            direction=(
                (
                    Direction.LONG
                    if item.side == "above"
                    else Direction.SHORT
                )
                if outside
                else (
                    Direction.SHORT
                    if item.side == "above"
                    else Direction.LONG
                )
            ),
            crossed_at=crossed_at,
            zone=(item.lower_bound, item.upper_bound),
            context_event_ids=(resolution_state_event.event_id,),
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
                if self._pool_close_outside(item, sweep_candle):
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
                        self._append_level_resolution_event(
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
                    crossed_at=sweep_candle.end,
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
                self._append_projected_pool_resolution_event(
                    item,
                    bar,
                    crossed_at=swept_at,
                )
            else:
                resolved.append((item, bar, swept_at))
            self._pending_pool_sweeps.pop(item_id, None)
        return tuple(resolved)

    def _append_level_resolution_event(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
        *,
        crossed_at: pd.Timestamp,
    ) -> None:
        """Resolve one non-pool penetration on the next completed real bar."""

        outside = self._pool_close_outside(item, candle)
        try:
            resolution_bar_event_id = self._bar_event_id_at(
                Timeframe.M1,
                candle.end,
            )
        except ValueError:
            return
        penetration_event_id = self._penetration_event_ids.get(
            self._penetration_key(
                level_id=item.item_id,
                timeframe=Timeframe.M1,
                crossed_at=crossed_at,
            )
        )
        if penetration_event_id is None:
            raise ValueError(
                "level resolution lacks its exact crossing penetration"
            )
        self._append_crossing_resolution(
            (
                EventKind.ACCEPTANCE_CONFIRMED
                if outside
                else EventKind.SWEEP_CONFIRMED
            ),
            candle.end,
            Timeframe.M1,
            item.side,
            float(candle.close),
            item.strength,
            (penetration_event_id, resolution_bar_event_id),
            {
                "level_id": item.item_id,
                "source_timeframe": item.timeframe.value,
                "source_kind": item.kind,
                "resolution_bars": 1,
                "resolution": (
                    "later_close_held_outside"
                    if outside
                    else "later_close_returned_inside"
                ),
                "crossed_at": crossed_at.isoformat(),
            },
            direction=(
                (
                    Direction.LONG
                    if item.side == "above"
                    else Direction.SHORT
                )
                if outside
                else (
                    Direction.SHORT
                    if item.side == "above"
                    else Direction.LONG
                )
            ),
            crossed_at=crossed_at,
            zone=(item.lower_bound, item.upper_bound),
        )

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
                self._append_level_resolution_event(
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
                if self._pool_close_outside(item, bar):
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
        reality: ExecutionRealityInput | None = None,
    ) -> MarketObservation:
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
        if self.config.eye_authority_mode:
            if reality is not None:
                raise ValueError(
                    "eye-authority mode does not evaluate execution reality"
                )
            execution = self._execution_not_evaluated()
        else:
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
                    if (
                        displacement is not None
                        and not self.config.group4_projection_only
                    ):
                        # Boundary transitions close the prior displacement
                        # epoch and therefore still reference its normalized
                        # bars.  Publish those immutable terminal facts before
                        # clearing the prior-epoch BAR lookup tables.
                        self._record_displacement_events(displacement)
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
        if not self.config.group4_projection_only:
            try:
                # Normalized data facts are the roots of the semantic DAG.
                # Publish them before Displacement, Group 3, structure, or
                # liquidity events attempt to reference the consumed bars.
                self._append_available_bar_events(
                    update,
                    histories,
                    frames,
                )
                self._publish_reference_candidate_events()
                self._record_displacement_events(displacement)
            except Exception:
                self._terminal_failure = (
                    "normalized bar/reference/displacement semantic projection "
                    "failed "
                    "after state may have changed; discard this observer "
                    "and resume from the last checkpoint"
                )
                raise
        for timeframe, tracker in self._liquidity_trackers.items():
            if timeframe not in liquidity_snapshots:
                liquidity_snapshots[timeframe] = (
                    self._liquidity_snapshot(timeframe, tracker)
                )
            _, _, native_inventory = liquidity_snapshots[timeframe]
            base_inventory.extend(native_inventory)
        if not self.config.group4_projection_only:
            base_inventory.extend(self._reference_inventory.values())
        if not self.config.group4_projection_only:
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
                if not self.config.group4_projection_only:
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
            if not self.config.group4_projection_only:
                for item, candle, crossed_at in deferred_pool_resolution_events:
                    self._append_projected_pool_resolution_event(
                        item,
                        candle,
                        crossed_at=crossed_at,
                    )
                for (
                    item,
                    candle,
                    crossed_at,
                ) in deferred_level_resolution_events:
                    self._append_level_resolution_event(
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
            if not self.config.group4_projection_only:
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
        if (
            group3_update is not None
            and group3_update.boundary_reason is None
        ):
            try:
                group3_update = self._finalize_group3_foundation(
                    group3_update,
                    update,
                )
            except Exception:
                self._terminal_failure = (
                    "Group 3 foundation projection failed exact source "
                    "binding; discard this observer and resume from the "
                    "last checkpoint"
                )
                raise
        elif group3_update is not None:
            try:
                group3_update = (
                    self._finalize_group3_foundation_boundary(
                        group3_update
                    )
                )
            except Exception:
                self._terminal_failure = (
                    "Group 3 foundation boundary projection failed exact "
                    "source binding; discard this observer and resume "
                    "from the last checkpoint"
                )
                raise
        if (
            self._group4_bootstrap_range_transitions
            and not self.config.group4_projection_only
        ):
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
        elif self.config.group4_projection_only:
            self._group4_bootstrap_range_transitions.clear()
        if group4_update is not None:
            try:
                if not self.config.group4_projection_only:
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
        if not self.config.group4_projection_only:
            try:
                group4_boundary_range_keys = {
                    f"range:{state.range_id}"
                    for state in (
                        ()
                        if (
                            group4_update is None
                            or group4_update.boundary_reason
                            not in GROUP4_HARD_BOUNDARY_REASONS
                        )
                        else group4_update.range_transitions
                    )
                }
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
                        terminal_entity_keys=(
                            ()
                            if (
                                group3_update is None
                                or group3_update.boundary_reason
                                not in FVG_BOUNDARY_REASONS
                            )
                            else (
                                *(
                                    f"fvg:{state.fvg_id}"
                                    for state in group3_update.fvg_transitions
                                ),
                                *(
                                    "order_block:"
                                    f"{state.order_block_id}"
                                    for state in (
                                        group3_update.order_block_transitions
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
                    "semantic_reset": "semantic_reset",
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
        market_anomalies = tuple(dict.fromkeys(anomalies))
        anomalies.extend(execution.anomalies)
        try:
            self.memory.flush_audit()
            semantic_events = self.audit_store.events_since(audit_start)
            market_snapshot, projection_events = (
                self.market_snapshot_publisher.publish(
                    asof=update.asof,
                    symbol=update.completed_1m.symbol,
                    instrument_id=update.completed_1m.instrument_id,
                    price=float(update.completed_1m.close),
                    completed_1m=update.completed_1m,
                    frames=frames,
                    inventory=liquidity_inventory,
                    displacement=displacement,
                    semantic_events=semantic_events,
                    anomalies=market_anomalies,
                    emit_projection_events=(
                        self.config.persist_state_projections
                    ),
                )
            )
            foundation_stage = None
            foundation_events: tuple[MarketEvent, ...] = ()
            if self._foundation_adapter is not None:
                foundation_stage = self._stage_foundation_projection(
                    asof=update.asof,
                    frames=frames,
                    histories=histories,
                    group3_update=group3_update,
                    group4_update=group4_update,
                    snapshot=market_snapshot,
                    semantic_events=semantic_events,
                )
                foundation_events = tuple(
                    foundation_record_projection_event(
                        record,
                        timeframe=self._foundation_transport_timeframe(
                            record
                        ),
                        published_at=update.asof,
                    )
                    for record in foundation_stage[6]
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
            for event in foundation_events:
                self.memory.append(
                    event,
                    include_in_recent=False,
                    sequence_floor=(
                        EventMemory._GROUP4_CREATION_SEQUENCE_FLOOR
                        + 2_000_000
                    ),
                )
            if self.config.persist_state_projections or foundation_events:
                self.memory.flush_audit()
            semantic_events = self.audit_store.events_since(audit_start)
            foundation_projection = (
                None
                if foundation_stage is None
                or not foundation_stage[0].projection.records
                else foundation_stage[0].projection
            )
            foundation_states = foundation_dol_timeframe_states(
                foundation_projection,
                states=market_snapshot.timeframe_states,
                price=market_snapshot.price,
                candidate_templates=(
                    None if foundation_stage is None else foundation_stage[9]
                ),
                real_bar_ordinals=(
                    None
                    if foundation_stage is None
                    else {
                        item.timeframe: item.count
                        for item in foundation_stage[0].lifecycle.real_bar_clocks
                    }
                ),
            )
            market_snapshot = replace(
                market_snapshot,
                timeframe_states=foundation_states,
                relations=RelationResolver(
                    edges=MarketSnapshotPublisher._RELATION_EDGES
                ).resolve(
                    foundation_states,
                    price=market_snapshot.price,
                    asof=market_snapshot.asof,
                ),
                events_this_update=semantic_events,
                foundation=foundation_projection,
                foundation_range_locations=(
                    foundation_dual_range_locations(
                        foundation_projection,
                        price=market_snapshot.price,
                        timeframes=market_snapshot.timeframe_states,
                    )
                ),
            )
            if foundation_stage is not None:
                foundation_stage[0].commit_staged_candidate()
                (
                    self._foundation_adapter,
                    self._foundation_geometry_nodes,
                    self._foundation_geometry_assignments,
                    self._foundation_active_clusters,
                    self._foundation_structural_ranges,
                    self._foundation_fvg_contexts,
                    new_foundation_records,
                    group3_update,
                    self._foundation_plan_revisions,
                    self._foundation_dol_templates,
                ) = foundation_stage
                self._foundation_published_record_ids.update(
                    record.record_id for record in new_foundation_records
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
            if group4_update is None
            else group4_update.manipulations
        )
        current_entry_locations = (
            ()
            if group5_update is None
            else group5_update.entry_locations
        )
        current_reacceptances = (
            ()
            if group5_update is None
            else group5_update.qualified_reacceptances
        )
        current_micro_bos = (
            ()
            if group5_update is None
            else group5_update.micro_bos_references
        )
        current_paths = (
            ()
            if group5_update is None
            else group5_update.path_sequences
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
        group5_entry_location_delta: tuple[object, ...] = ()
        group5_reacceptance_delta: tuple[object, ...] = ()
        group5_micro_bos_delta: tuple[object, ...] = ()
        group5_path_delta: tuple[object, ...] = ()
        group5_step_delta: tuple[tuple[str, PathSequenceStep], ...] = ()
        if typed_delta_available:
            baseline = prior_observation is None
            group4_boundary = bool(
                group4_update is not None
                and group4_update.boundary_reason is not None
            )
            group5_boundary = bool(
                group5_update is not None
                and group5_update.boundary_reason is not None
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
                    if group3_update is None
                    else group3_update.fvg_transitions
                ),
                first_observation=baseline,
                boundary_reason=(
                    None
                    if group3_update is None
                    else group3_update.boundary_reason
                ),
            )
            group3_order_block_delta = _typed_native_transitions_or_baseline(
                current=current_order_blocks,
                transitions=(
                    ()
                    if group3_update is None
                    else group3_update.order_block_transitions
                ),
                first_observation=baseline,
                boundary_reason=(
                    None
                    if group3_update is None
                    else group3_update.boundary_reason
                ),
            )
            group4_range_delta = _typed_native_transitions_or_baseline(
                current=current_ranges,
                transitions=(
                    ()
                    if group4_update is None
                    else group4_update.range_transitions
                ),
                first_observation=baseline,
                boundary_reason=(
                    None
                    if group4_update is None
                    else group4_update.boundary_reason
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
                            if group4_update is None
                            else group4_update.manipulation_transitions
                        ),
                        *live_manipulations,
                    )
                ),
                "manipulation_id",
                retained_states=current_manipulations,
            )

            # Group 5 has complete path/step transitions.  Entry location,
            # qualified reacceptance and micro-BOS remain bounded semantic
            # snapshot fallbacks until their reducer exposes ordinary deltas.
            group5_entry_location_delta = cached(
                "group5_entry_location",
                current_entry_locations,
                "location_id",
                excluded_fields=_ENTRY_LOCATION_VIEW_FIELDS,
                retained_states=current_entry_locations,
            )
            group5_reacceptance_delta = cached(
                "group5_reacceptance",
                (
                    *current_reacceptances,
                    *(
                        ()
                        if group5_update is None
                        else group5_update.reacceptance_transitions
                    ),
                ),
                "reacceptance_id",
                retained_states=current_reacceptances,
            )
            group5_micro_bos_delta = cached(
                "group5_micro_bos",
                current_micro_bos,
                "reference_id",
                retained_states=current_micro_bos,
            )
            if baseline and not group5_boundary:
                path_candidates = current_paths
                group5_step_delta = tuple(
                    (path.sequence_id, step)
                    for path in current_paths
                    for step in path.steps
                )
            else:
                group5_step_delta = (
                    ()
                    if group5_update is None
                    else group5_update.step_transitions
                )
                step_path_ids = {
                    sequence_id
                    for sequence_id, _ in group5_step_delta
                }
                path_candidates = (
                    *(
                        ()
                        if group5_update is None
                        else group5_update.path_transitions
                    ),
                    *(
                        state
                        for state in current_paths
                        if state.sequence_id in step_path_ids
                    ),
                )
            group5_path_delta = _typed_state_delta_from_cache(
                candidates=path_candidates,
                signatures={},
                identity_field="sequence_id",
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
                group5_entry_location_transitions_this_update=(
                    group5_entry_location_delta
                ),
                group5_reacceptance_transitions_this_update=(
                    group5_reacceptance_delta
                ),
                group5_micro_bos_transitions_this_update=(
                    group5_micro_bos_delta
                ),
                group5_path_transitions_this_update=(
                    group5_path_delta
                ),
                group5_step_transitions_this_update=group5_step_delta,
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
                group3_order_block_funnel=(
                    ()
                    if group3_update is None
                    else group3_update.order_block_funnel
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
                group4_source_dispositions=(
                    ()
                    if group4_update is None
                    else group4_update.source_dispositions
                ),
                group4_range_funnel=(
                    ()
                    if group4_update is None
                    else group4_update.range_funnel
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
                observation = observation._with_scene_delta(
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
        self._last_group3_foundation_projection = group3_update
        self._last_market_epoch_reset_event_id = None
        self._prior = observation
        return observation


__all__ = [
    "CausalObserver",
    "EventMemory",
    "ExecutionRealityInput",
    "ObserverConfig",
]
