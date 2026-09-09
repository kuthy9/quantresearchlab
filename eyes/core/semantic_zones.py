"""Immutable zone-semantics foundations for the next canonical contract.

This module deliberately does not select candles, detect displacement, or
emit :class:`~contract.eye.observation.MarketEvent` objects.  It receives facts that
the existing Eye already knows and supplies small, pure lifecycle reducers.
Keeping it independent lets the current v1.2 replay remain historical while a
later semantic version binds these contracts explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from .foundation_registry import FOUNDATION_VERSION
from shares.core.market_clock import next_registered_native_completion
from contract.market import (
    Direction,
    Timeframe,
    aware_timestamp,
    price_to_ticks,
    to_primitive,
)


_NATIVE_TIMEFRAME_INTERVAL = {
    Timeframe.M1: pd.Timedelta(1, unit="min"),
    Timeframe.M5: pd.Timedelta(5, unit="min"),
    Timeframe.M15: pd.Timedelta(15, unit="min"),
    Timeframe.H1: pd.Timedelta(1, unit="h"),
    Timeframe.H4: pd.Timedelta(4, unit="h"),
}

_NATIVE_TIMEFRAME_MINUTES = {
    timeframe: int(interval / pd.Timedelta(1, unit="min"))
    for timeframe, interval in _NATIVE_TIMEFRAME_INTERVAL.items()
}

_NATIVE_TIMEFRAME_ANCHOR_MINUTES = {
    Timeframe.M1: 0,
    Timeframe.M5: 0,
    Timeframe.M15: 0,
    Timeframe.H1: 0,
    Timeframe.H4: 18 * 60,
}


def _clock(value: Any, *, name: str) -> pd.Timestamp:
    result = aware_timestamp(value, name=name)
    if pd.isna(result):
        raise ValueError(f"{name} cannot be NaT")
    return result


def _identities(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identities")
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(not isinstance(value, str) or not value.strip() for value in result)
    ):
        raise ValueError(f"{name} must contain unique non-empty identities")
    return result


def _optional_identities(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    result = tuple(values)
    if (
        len(result) != len(set(result))
        or any(not isinstance(value, str) or not value.strip() for value in result)
    ):
        raise ValueError(f"{name} must contain unique non-empty identities")
    return result


def _identity(prefix: str, value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _foundation_version(value: str) -> str:
    if value != FOUNDATION_VERSION:
        raise ValueError("zone semantic foundation version is not frozen v2")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class CompatibleStructureKind(str, Enum):
    QUALIFIED_BOS = "qualified_bos"
    MSS_CORE_CONFIRMED = "mss_core_confirmed"


@dataclass(frozen=True)
class BaseOriginCore:
    """Frozen full-candle-cluster origin known at displacement start.

    Selection is intentionally out of scope.  The existing Group-3 selector
    supplies the already ordered cluster and its full-range/body geometry.
    """

    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    source_displacement_id: str
    source_displacement_event_id: str
    anchor_bar_event_ids: tuple[str, ...]
    anchor_candle_ids: tuple[str, ...]
    anchor_completed_at: tuple[pd.Timestamp, ...]
    lower_bound: float
    upper_bound: float
    body_lower_bound: float
    body_upper_bound: float
    tick_size: float
    formed_at: pd.Timestamp
    known_at: pd.Timestamp
    semantic_version: str = FOUNDATION_VERSION
    core_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self, "semantic_version", _foundation_version(self.semantic_version)
        )
        if (
            not isinstance(self.symbol, str)
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or not self.source_displacement_id
            or not self.source_displacement_event_id
        ):
            raise ValueError("base origin core identity is incomplete")
        bar_ids = _identities(
            self.anchor_bar_event_ids,
            name="base origin anchor BAR event ids",
        )
        candle_ids = _identities(
            self.anchor_candle_ids,
            name="base origin anchor candle ids",
        )
        clocks = tuple(
            _clock(value, name="base origin anchor completion")
            for value in self.anchor_completed_at
        )
        if (
            len(bar_ids) != len(candle_ids)
            or len(clocks) != len(bar_ids)
            or clocks != tuple(sorted(clocks))
            or len(clocks) != len(set(clocks))
            or any(
                right - left != _NATIVE_TIMEFRAME_INTERVAL[self.timeframe]
                for left, right in zip(clocks, clocks[1:])
            )
        ):
            raise ValueError(
                "base origin cluster ancestry is not contiguous ordered one-to-one"
            )
        object.__setattr__(self, "anchor_bar_event_ids", bar_ids)
        object.__setattr__(self, "anchor_candle_ids", candle_ids)
        object.__setattr__(self, "anchor_completed_at", clocks)
        tick_size = _finite_positive(self.tick_size, name="base origin tick_size")
        object.__setattr__(self, "tick_size", tick_size)
        lower = float(self.lower_bound)
        upper = float(self.upper_bound)
        body_lower = float(self.body_lower_bound)
        body_upper = float(self.body_upper_bound)
        if (
            not all(
                math.isfinite(value)
                for value in (lower, upper, body_lower, body_upper)
            )
            or not lower < upper
            or not lower <= body_lower < body_upper <= upper
        ):
            raise ValueError("base origin full-cluster geometry is invalid")
        lower_ticks = price_to_ticks(lower, tick_size, name="base origin lower")
        upper_ticks = price_to_ticks(upper, tick_size, name="base origin upper")
        price_to_ticks(body_lower, tick_size, name="base origin body lower")
        price_to_ticks(body_upper, tick_size, name="base origin body upper")
        if upper_ticks <= lower_ticks:
            raise ValueError("base origin width must be positive on the tick grid")
        formed_at = _clock(self.formed_at, name="base origin formed_at")
        known_at = _clock(self.known_at, name="base origin known_at")
        if clocks[-1] != formed_at or formed_at > known_at:
            raise ValueError("base origin formation clocks are inconsistent")
        object.__setattr__(self, "formed_at", formed_at)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "body_lower_bound", body_lower)
        object.__setattr__(self, "body_upper_bound", body_upper)
        payload = {
            "semantic_version": self.semantic_version,
            "symbol": self.symbol,
            "instrument_id": self.instrument_id,
            "timeframe": self.timeframe.value,
            "direction": self.direction.value,
            "source_displacement_id": self.source_displacement_id,
            "source_displacement_event_id": self.source_displacement_event_id,
            "anchor_bar_event_ids": bar_ids,
            "anchor_candle_ids": candle_ids,
            "anchor_completed_at": clocks,
            "lower_bound_ticks": lower_ticks,
            "upper_bound_ticks": upper_ticks,
            "body_lower_bound": body_lower,
            "body_upper_bound": body_upper,
            "formed_at": formed_at,
            "known_at": known_at,
        }
        object.__setattr__(self, "core_id", _identity("base-origin-core", payload))

    @property
    def midpoint(self) -> float:
        return (self.lower_bound + self.upper_bound) / 2.0

    @property
    def width_ticks(self) -> int:
        return price_to_ticks(
            self.upper_bound,
            self.tick_size,
            name="base origin upper",
        ) - price_to_ticks(
            self.lower_bound,
            self.tick_size,
            name="base origin lower",
        )

    @property
    def source_event_ids(self) -> tuple[str, ...]:
        return (self.source_displacement_event_id, *self.anchor_bar_event_ids)


@dataclass(frozen=True)
class QualifiedOrderBlock:
    """A Base Origin Core qualified by exact displacement and structure facts."""

    base_origin_core_id: str
    source_displacement_id: str
    source_displacement_event_id: str
    compatible_structure_event_id: str
    compatible_structure_kind: CompatibleStructureKind
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    lower_bound: float
    upper_bound: float
    body_lower_bound: float
    body_upper_bound: float
    tick_size: float
    qualified_at: pd.Timestamp
    known_at: pd.Timestamp
    semantic_version: str = FOUNDATION_VERSION
    qualified_ob_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self,
            "compatible_structure_kind",
            CompatibleStructureKind(self.compatible_structure_kind),
        )
        object.__setattr__(
            self, "semantic_version", _foundation_version(self.semantic_version)
        )
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.base_origin_core_id,
                self.source_displacement_id,
                self.source_displacement_event_id,
                self.compatible_structure_event_id,
                self.symbol,
            )
        ) or type(self.instrument_id) is not int or self.instrument_id < 0:
            raise ValueError("qualified OB identity is incomplete")
        tick = _finite_positive(self.tick_size, name="qualified OB tick_size")
        object.__setattr__(self, "tick_size", tick)
        values = (
            float(self.lower_bound),
            float(self.upper_bound),
            float(self.body_lower_bound),
            float(self.body_upper_bound),
        )
        if not all(math.isfinite(value) for value in values) or not (
            values[0] < values[1]
            and values[0] <= values[2] < values[3] <= values[1]
        ):
            raise ValueError("qualified OB geometry is invalid")
        for name, value in zip(
            ("lower", "upper", "body lower", "body upper"), values
        ):
            price_to_ticks(value, tick, name=f"qualified OB {name}")
        object.__setattr__(self, "lower_bound", values[0])
        object.__setattr__(self, "upper_bound", values[1])
        object.__setattr__(self, "body_lower_bound", values[2])
        object.__setattr__(self, "body_upper_bound", values[3])
        qualified_at = _clock(self.qualified_at, name="qualified OB qualified_at")
        known_at = _clock(self.known_at, name="qualified OB known_at")
        if qualified_at > known_at:
            raise ValueError("qualified OB cannot be known before qualification")
        object.__setattr__(self, "qualified_at", qualified_at)
        object.__setattr__(self, "known_at", known_at)
        payload = {
            name: value
            for name, value in self.__dict__.items()
            if name != "qualified_ob_id"
        }
        object.__setattr__(
            self,
            "qualified_ob_id",
            _identity("qualified-order-block", payload),
        )

    @property
    def source_event_ids(self) -> tuple[str, ...]:
        return (
            self.source_displacement_event_id,
            self.compatible_structure_event_id,
        )


def qualify_order_block(
    core: BaseOriginCore,
    *,
    source_displacement_id: str,
    source_displacement_event_id: str,
    compatible_structure_event_id: str,
    compatible_structure_kind: CompatibleStructureKind,
    qualified_at: pd.Timestamp,
    known_at: pd.Timestamp,
) -> QualifiedOrderBlock:
    """Purely qualify one exact core; sibling displacement borrowing fails."""

    if not isinstance(core, BaseOriginCore):
        raise TypeError("qualified OB requires a BaseOriginCore")
    if (
        source_displacement_id != core.source_displacement_id
        or source_displacement_event_id != core.source_displacement_event_id
    ):
        raise ValueError("qualified OB displacement does not bind its exact core")
    known = _clock(known_at, name="qualified OB known_at")
    qualified = _clock(qualified_at, name="qualified OB qualified_at")
    if qualified < core.known_at or known < qualified:
        raise ValueError("qualified OB predates its Base Origin Core")
    return QualifiedOrderBlock(
        base_origin_core_id=core.core_id,
        source_displacement_id=source_displacement_id,
        source_displacement_event_id=source_displacement_event_id,
        compatible_structure_event_id=compatible_structure_event_id,
        compatible_structure_kind=compatible_structure_kind,
        symbol=core.symbol,
        instrument_id=core.instrument_id,
        timeframe=core.timeframe,
        direction=core.direction,
        lower_bound=core.lower_bound,
        upper_bound=core.upper_bound,
        body_lower_bound=core.body_lower_bound,
        body_upper_bound=core.body_upper_bound,
        tick_size=core.tick_size,
        qualified_at=qualified,
        known_at=known,
        semantic_version=core.semantic_version,
    )


class ZoneObjectKind(str, Enum):
    FVG = "fvg"
    BASE_ORIGIN_CORE = "base_origin_core"
    QUALIFIED_ORDER_BLOCK = "qualified_order_block"
    STRUCTURAL_RANGE = "structural_range"
    LIQUIDITY_ZONE = "liquidity_zone"


class ZoneEntrySide(str, Enum):
    FROM_ABOVE = "from_above"
    FROM_BELOW = "from_below"
    GAP_OPENED_INSIDE = "gap_opened_inside"


@dataclass(frozen=True)
class CompletedZoneBar:
    bar_event_id: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    known_at: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    session: str
    context_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        if (
            not self.bar_event_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or not self.session
        ):
            raise ValueError("completed zone BAR identity is incomplete")
        object.__setattr__(
            self, "known_at", _clock(self.known_at, name="completed zone BAR known_at")
        )
        values = tuple(
            float(value)
            for value in (self.open, self.high, self.low, self.close)
        )
        if not all(math.isfinite(value) for value in values) or not (
            values[2] <= values[0] <= values[1]
            and values[2] <= values[3] <= values[1]
        ):
            raise ValueError("completed zone BAR OHLC is invalid")
        for name, value in zip(("open", "high", "low", "close"), values):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "context_event_ids",
            _optional_identities(
                self.context_event_ids,
                name="completed zone BAR context event ids",
            ),
        )


@dataclass(frozen=True)
class ZoneFirstRetestSpec:
    object_kind: ZoneObjectKind
    object_id: str
    creation_event_id: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    lower_bound: float
    upper_bound: float
    tick_size: float
    object_created_at: pd.Timestamp
    object_known_at: pd.Timestamp
    departure_confirmed_at: pd.Timestamp | None
    departure_source_event_id: str | None
    creation_declared_departed: bool
    semantic_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_kind", ZoneObjectKind(self.object_kind))
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self, "semantic_version", _foundation_version(self.semantic_version)
        )
        if (
            not self.object_id
            or not self.creation_event_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or type(self.creation_declared_departed) is not bool
        ):
            raise ValueError(
                "first-retest registration identity/departure is incomplete"
            )
        tick = _finite_positive(self.tick_size, name="first-retest tick_size")
        object.__setattr__(self, "tick_size", tick)
        lower = float(self.lower_bound)
        upper = float(self.upper_bound)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("first-retest frozen zone is invalid")
        price_to_ticks(lower, tick, name="first-retest lower")
        price_to_ticks(upper, tick, name="first-retest upper")
        created = _clock(self.object_created_at, name="first-retest object_created_at")
        known = _clock(self.object_known_at, name="first-retest object_known_at")
        departed = (
            None
            if self.departure_confirmed_at is None
            else _clock(
                self.departure_confirmed_at,
                name="first-retest departure_confirmed_at",
            )
        )
        has_departure = departed is not None and bool(
            self.departure_source_event_id
        )
        if (
            created > known
            or self.creation_declared_departed != has_departure
            or (
                departed is not None
                and not created <= departed <= known
            )
            or (
                departed is None
                and self.departure_source_event_id is not None
            )
        ):
            raise ValueError(
                "first-retest departure declaration is inconsistent"
            )
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "object_created_at", created)
        object.__setattr__(self, "object_known_at", known)
        object.__setattr__(self, "departure_confirmed_at", departed)


@dataclass(frozen=True)
class ZoneFirstRetest:
    object_kind: ZoneObjectKind
    object_id: str
    creation_event_id: str
    departure_source_event_id: str
    source_bar_event_id: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    known_at: pd.Timestamp
    entry_side: ZoneEntrySide
    fill_fraction: float
    age_bars: int
    age_seconds: int
    session: str
    context_event_ids: tuple[str, ...]
    semantic_version: str = FOUNDATION_VERSION
    first_retest_event_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_kind", ZoneObjectKind(self.object_kind))
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(self, "entry_side", ZoneEntrySide(self.entry_side))
        object.__setattr__(
            self, "semantic_version", _foundation_version(self.semantic_version)
        )
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.object_id,
                self.creation_event_id,
                self.departure_source_event_id,
                self.source_bar_event_id,
                self.symbol,
                self.session,
            )
        ) or type(self.instrument_id) is not int or self.instrument_id < 0:
            raise ValueError("first-retest event identity is incomplete")
        known = _clock(self.known_at, name="first-retest known_at")
        object.__setattr__(self, "known_at", known)
        fill = float(self.fill_fraction)
        if (
            not math.isfinite(fill)
            or not 0.0 <= fill <= 1.0
            or type(self.age_bars) is not int
            or self.age_bars < 1
            or type(self.age_seconds) is not int
            or self.age_seconds <= 0
        ):
            raise ValueError("first-retest causal metrics are invalid")
        object.__setattr__(self, "fill_fraction", fill)
        object.__setattr__(
            self,
            "context_event_ids",
            _optional_identities(
                self.context_event_ids,
                name="first-retest context event ids",
            ),
        )
        payload = {
            name: value
            for name, value in self.__dict__.items()
            if name != "first_retest_event_id"
        }
        object.__setattr__(
            self,
            "first_retest_event_id",
            _identity("zone-first-retest", payload),
        )

    @property
    def source_event_ids(self) -> tuple[str, str, str]:
        return (
            self.creation_event_id,
            self.departure_source_event_id,
            self.source_bar_event_id,
        )


@dataclass(frozen=True)
class ZoneFirstReinteractionTracker:
    """Immutable tracker that emits at most one native-timeframe retest."""

    spec: ZoneFirstRetestSpec
    observed_native_bars: int = 0
    last_observed_at: pd.Timestamp | None = None
    departure_confirmed_at: pd.Timestamp | None = None
    departure_source_event_id: str | None = None
    first_retest: ZoneFirstRetest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ZoneFirstRetestSpec):
            raise TypeError("first-reinteraction tracker requires a frozen spec")
        if type(self.observed_native_bars) is not int or self.observed_native_bars < 0:
            raise ValueError("first-reinteraction observed BAR count is invalid")
        departure_clock = self.departure_confirmed_at
        departure_event_id = self.departure_source_event_id
        if self.spec.creation_declared_departed:
            if departure_clock is None:
                departure_clock = self.spec.departure_confirmed_at
            if departure_event_id is None:
                departure_event_id = self.spec.departure_source_event_id
        if (departure_clock is None) != (departure_event_id is None):
            raise ValueError("first-reinteraction departure provenance is incomplete")
        if departure_clock is not None:
            departure_clock = _clock(
                departure_clock,
                name="first-reinteraction departure_confirmed_at",
            )
            if (
                self.spec.creation_declared_departed
                and departure_clock > self.spec.object_known_at
            ) or (
                not self.spec.creation_declared_departed
                and departure_clock <= self.spec.object_known_at
            ):
                raise ValueError("first-reinteraction departure clock is invalid")
        object.__setattr__(self, "departure_confirmed_at", departure_clock)
        object.__setattr__(self, "departure_source_event_id", departure_event_id)
        if self.last_observed_at is not None:
            last = _clock(
                self.last_observed_at,
                name="first-reinteraction last_observed_at",
            )
            if last <= self.spec.object_known_at:
                raise ValueError("first-reinteraction BAR must be strictly later")
            object.__setattr__(self, "last_observed_at", last)
            if departure_clock is not None and departure_clock > last:
                raise ValueError("first-reinteraction departure exceeds observed history")
        if self.first_retest is not None:
            if (
                not isinstance(self.first_retest, ZoneFirstRetest)
                or self.first_retest.object_id != self.spec.object_id
                or self.first_retest.known_at != self.last_observed_at
            ):
                raise ValueError("first-reinteraction terminal does not bind its spec")

    def on_completed_bar(
        self,
        bar: CompletedZoneBar,
    ) -> "ZoneFirstReinteractionTracker":
        if self.first_retest is not None:
            # A terminal tracker is immutable and deliberately does not inspect
            # future path values.
            return self
        if not isinstance(bar, CompletedZoneBar):
            raise TypeError("first-reinteraction requires CompletedZoneBar")
        spec = self.spec
        if (
            bar.symbol != spec.symbol
            or bar.instrument_id != spec.instrument_id
            or bar.timeframe is not spec.timeframe
        ):
            raise ValueError("first-reinteraction BAR scope differs from object")
        if bar.known_at <= spec.object_known_at:
            raise ValueError("first-reinteraction BAR must be strictly later")
        if self.last_observed_at is not None and bar.known_at <= self.last_observed_at:
            raise ValueError("first-reinteraction BAR is duplicated or out of order")
        prior_clock = (
            self.spec.object_known_at
            if self.last_observed_at is None
            else self.last_observed_at
        )
        try:
            expected_completion = next_registered_native_completion(
                prior_clock,
                timeframe_minutes=_NATIVE_TIMEFRAME_MINUTES[spec.timeframe],
                anchor_minute=_NATIVE_TIMEFRAME_ANCHOR_MINUTES[spec.timeframe],
            )
        except ValueError as error:
            raise ValueError(
                "first-reinteraction BAR path is not contiguous at the native timeframe"
            ) from error
        if bar.known_at != expected_completion:
            raise ValueError(
                "first-reinteraction BAR path is not contiguous at the native timeframe"
            )
        age_bars = self.observed_native_bars + 1
        if self.departure_confirmed_at is None:
            departed = (
                bar.close > spec.upper_bound
                if spec.direction is Direction.LONG
                else bar.close < spec.lower_bound
            )
            return replace(
                self,
                observed_native_bars=age_bars,
                last_observed_at=bar.known_at,
                departure_confirmed_at=(bar.known_at if departed else None),
                departure_source_event_id=(bar.bar_event_id if departed else None),
            )
        if bar.known_at <= self.departure_confirmed_at:
            raise ValueError("first reinteraction must follow confirmed departure")
        intersects = bar.high >= spec.lower_bound and bar.low <= spec.upper_bound
        if not intersects:
            return replace(
                self,
                observed_native_bars=age_bars,
                last_observed_at=bar.known_at,
            )
        if spec.lower_bound <= bar.open <= spec.upper_bound:
            side = ZoneEntrySide.GAP_OPENED_INSIDE
        elif bar.open > spec.upper_bound:
            side = ZoneEntrySide.FROM_ABOVE
        else:
            side = ZoneEntrySide.FROM_BELOW
        width = spec.upper_bound - spec.lower_bound
        fill = (
            (spec.upper_bound - max(bar.low, spec.lower_bound)) / width
            if spec.direction is Direction.LONG
            else (min(bar.high, spec.upper_bound) - spec.lower_bound) / width
        )
        event = ZoneFirstRetest(
            object_kind=spec.object_kind,
            object_id=spec.object_id,
            creation_event_id=spec.creation_event_id,
            departure_source_event_id=self.departure_source_event_id,
            source_bar_event_id=bar.bar_event_id,
            symbol=spec.symbol,
            instrument_id=spec.instrument_id,
            timeframe=spec.timeframe,
            direction=spec.direction,
            known_at=bar.known_at,
            entry_side=side,
            fill_fraction=min(1.0, max(0.0, float(fill))),
            age_bars=age_bars,
            age_seconds=int((bar.known_at - spec.object_known_at).total_seconds()),
            session=bar.session,
            context_event_ids=bar.context_event_ids,
            semantic_version=spec.semantic_version,
        )
        return replace(
            self,
            observed_native_bars=age_bars,
            last_observed_at=bar.known_at,
            first_retest=event,
        )


class FVGAvailability(str, Enum):
    ACTIVE = "active"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    CENSORED = "censored"


class FVGTerminationCause(str, Enum):
    CLOSE_THROUGH_FAR_EDGE = "close_through_far_edge"
    PARENT_STRUCTURE_TERMINATED = "parent_structure_terminated"
    STRUCTURAL_RANGE_REPLACED = "structural_range_replaced"
    CONTRACT_ROLLOVER = "contract_rollover"
    SEMANTIC_RESET = "semantic_reset"
    DATA_GAP = "data_gap"


_FVG_CAUSE_DISPOSITION = {
    FVGTerminationCause.CLOSE_THROUGH_FAR_EDGE: FVGAvailability.INVALIDATED,
    FVGTerminationCause.PARENT_STRUCTURE_TERMINATED: FVGAvailability.EXPIRED,
    FVGTerminationCause.STRUCTURAL_RANGE_REPLACED: FVGAvailability.EXPIRED,
    FVGTerminationCause.CONTRACT_ROLLOVER: FVGAvailability.EXPIRED,
    FVGTerminationCause.SEMANTIC_RESET: FVGAvailability.EXPIRED,
    FVGTerminationCause.DATA_GAP: FVGAvailability.CENSORED,
}


@dataclass(frozen=True)
class FVGStructuralLifecycle:
    """Termination authority orthogonal to fill/mitigation observations."""

    fvg_id: str
    source_creation_event_id: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    created_at: pd.Timestamp
    known_at: pd.Timestamp
    parent_structure_generation_id: str | None = None
    structural_range_id: str | None = None
    context_source_event_ids: tuple[str, ...] = ()
    age_bars: int = 0
    age_seconds: int = 0
    availability: FVGAvailability = FVGAvailability.ACTIVE
    last_updated_at: pd.Timestamp | None = None
    terminal_reason: FVGTerminationCause | None = None
    terminal_event_id: str | None = None
    terminal_source_event_ids: tuple[str, ...] = ()
    semantic_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "availability", FVGAvailability(self.availability))
        object.__setattr__(
            self, "semantic_version", _foundation_version(self.semantic_version)
        )
        if (
            not self.fvg_id
            or not self.source_creation_event_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.parent_structure_generation_id == ""
            or self.structural_range_id == ""
        ):
            raise ValueError("FVG structural lifecycle identity is incomplete")
        created = _clock(self.created_at, name="FVG lifecycle created_at")
        known = _clock(self.known_at, name="FVG lifecycle known_at")
        updated = _clock(
            self.last_updated_at if self.last_updated_at is not None else known,
            name="FVG lifecycle last_updated_at",
        )
        if created > known or updated < known:
            raise ValueError("FVG structural lifecycle clocks are invalid")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "known_at", known)
        object.__setattr__(self, "last_updated_at", updated)
        sources = _optional_identities(
            self.terminal_source_event_ids,
            name="FVG terminal source event ids",
        )
        object.__setattr__(self, "terminal_source_event_ids", sources)
        context_sources = _optional_identities(
            self.context_source_event_ids,
            name="FVG structural-context source event ids",
        )
        object.__setattr__(
            self,
            "context_source_event_ids",
            context_sources,
        )
        if (
            type(self.age_bars) is not int
            or self.age_bars < 0
            or type(self.age_seconds) is not int
            or self.age_seconds < 0
            or self.age_seconds
            != int((updated - known).total_seconds())
        ):
            raise ValueError("FVG continuous age is invalid")
        if (
            self.structural_range_id is not None
            and self.parent_structure_generation_id is None
        ):
            raise ValueError("FVG structural range lacks its parent generation")
        if (
            (self.parent_structure_generation_id is None)
            != (not context_sources)
        ):
            raise ValueError("FVG structural context ancestry is incomplete")
        active = self.availability is FVGAvailability.ACTIVE
        if active:
            if (
                self.terminal_reason is not None
                or self.terminal_event_id is not None
                or sources
            ):
                raise ValueError("active FVG cannot carry a terminal transition")
        else:
            reason = FVGTerminationCause(self.terminal_reason)
            object.__setattr__(self, "terminal_reason", reason)
            if (
                _FVG_CAUSE_DISPOSITION[reason] is not self.availability
                or not self.terminal_event_id
                or not sources
                or sources[0] != self.source_creation_event_id
                or updated <= known
            ):
                raise ValueError("FVG terminal disposition is inconsistent")


def bind_fvg_structural_context(
    state: FVGStructuralLifecycle,
    *,
    parent_structure_generation_id: str,
    structural_range_id: str | None,
    source_event_ids: Sequence[str],
) -> FVGStructuralLifecycle:
    """Freeze creation-time structural ownership without hindsight rebinding."""

    if not isinstance(state, FVGStructuralLifecycle):
        raise TypeError("FVG context binding requires FVGStructuralLifecycle")
    if (
        not isinstance(parent_structure_generation_id, str)
        or not parent_structure_generation_id
        or structural_range_id == ""
    ):
        raise ValueError("FVG structural context identity is invalid")
    sources = _identities(
        source_event_ids,
        name="FVG structural context source event ids",
    )
    if state.availability is not FVGAvailability.ACTIVE or state.age_bars != 0:
        raise ValueError("FVG structural context must bind at creation time")
    if state.parent_structure_generation_id is not None:
        expected = (
            state.parent_structure_generation_id,
            state.structural_range_id,
            state.context_source_event_ids,
        )
        actual = (
            parent_structure_generation_id,
            structural_range_id,
            sources,
        )
        if actual == expected:
            return state
        raise ValueError("FVG structural context is immutable")
    return replace(
        state,
        parent_structure_generation_id=parent_structure_generation_id,
        structural_range_id=structural_range_id,
        context_source_event_ids=sources,
    )


def reduce_fvg_termination(
    state: FVGStructuralLifecycle,
    *,
    cause: FVGTerminationCause,
    known_at: pd.Timestamp,
    cause_event_ids: Sequence[str],
    related_entity_id: str | None = None,
) -> FVGStructuralLifecycle:
    """Apply one explicit terminal cause; no age/TTL transition exists."""

    if not isinstance(state, FVGStructuralLifecycle):
        raise TypeError("FVG termination requires FVGStructuralLifecycle")
    cause = FVGTerminationCause(cause)
    clock = _clock(known_at, name="FVG termination known_at")
    cause_ids = _identities(
        cause_event_ids,
        name="FVG termination cause event ids",
    )
    sources = tuple(
        dict.fromkeys(
            (
                state.source_creation_event_id,
                *state.context_source_event_ids,
                *cause_ids,
            )
        )
    )
    if sources[0] != state.source_creation_event_id:
        raise ValueError("FVG terminal ancestry lost its creation event")
    if cause is FVGTerminationCause.PARENT_STRUCTURE_TERMINATED:
        if (
            state.parent_structure_generation_id is None
            or related_entity_id != state.parent_structure_generation_id
        ):
            raise ValueError("FVG expiry does not bind its exact parent structure")
    elif cause is FVGTerminationCause.STRUCTURAL_RANGE_REPLACED:
        if (
            state.structural_range_id is None
            or related_entity_id != state.structural_range_id
        ):
            raise ValueError("FVG expiry does not bind its exact structural range")
    elif related_entity_id is not None:
        raise ValueError("FVG terminal cause cannot borrow an unrelated entity")
    disposition = _FVG_CAUSE_DISPOSITION[cause]
    terminal_id = _identity(
        "fvg-terminal",
        {
            "semantic_version": state.semantic_version,
            "fvg_id": state.fvg_id,
            "availability": disposition.value,
            "reason": cause.value,
            "known_at": clock,
            "source_event_ids": sources,
            "related_entity_id": related_entity_id,
        },
    )
    if state.availability is not FVGAvailability.ACTIVE:
        if (
            state.availability is disposition
            and state.terminal_reason is cause
            and state.last_updated_at == clock
            and state.terminal_event_id == terminal_id
            and state.terminal_source_event_ids == sources
        ):
            return state
        raise ValueError("FVG terminal lifecycle is immutable")
    if clock < state.last_updated_at or clock <= state.known_at:
        raise ValueError("FVG terminal clock must be after creation and age state")
    return replace(
        state,
        availability=disposition,
        age_seconds=int((clock - state.known_at).total_seconds()),
        last_updated_at=clock,
        terminal_reason=cause,
        terminal_event_id=terminal_id,
        terminal_source_event_ids=sources,
    )


__all__ = [
    "BaseOriginCore",
    "CompatibleStructureKind",
    "CompletedZoneBar",
    "FVGAvailability",
    "FVGStructuralLifecycle",
    "FVGTerminationCause",
    "bind_fvg_structural_context",
    "QualifiedOrderBlock",
    "ZoneEntrySide",
    "ZoneFirstReinteractionTracker",
    "ZoneFirstRetest",
    "ZoneFirstRetestSpec",
    "ZoneObjectKind",
    "qualify_order_block",
    "reduce_fvg_termination",
]
