"""Market primitives: the vocabulary and price arithmetic every layer names.

This is the base of the contract layering. It depends on nothing else in
``contract/`` and everything else depends on it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import pandas as pd


SMC_SEMANTIC_VERSION = "smc_semantics_v1.3"


def _exact_dataclass_pickle_state(
    value: Any,
    *,
    schema_version: int,
    label: str,
) -> Mapping[str, Any]:
    names = tuple(item.name for item in fields(value))
    if set(value.__dict__) != set(names):
        raise ValueError(f"{label} pickle state is not exact")
    return {
        "schema_version": schema_version,
        "fields": tuple((name, getattr(value, name)) for name in names),
    }


def _restore_exact_dataclass_pickle_state(
    value: Any,
    state: Mapping[str, Any],
    *,
    schema_version: int,
    label: str,
) -> None:
    names = tuple(item.name for item in fields(value))
    serialized = state.get("fields") if isinstance(state, Mapping) else None
    if (
        not isinstance(state, Mapping)
        or set(state) != {"schema_version", "fields"}
        or state.get("schema_version") != schema_version
        or not isinstance(serialized, tuple)
        or len(serialized) != len(names)
        or any(
            not isinstance(item, tuple) or len(item) != 2
            for item in serialized
        )
        or tuple(item[0] for item in serialized) != names
    ):
        raise ValueError(f"{label} pickle schema changed")
    candidate = object.__new__(type(value))
    for name, item in serialized:
        object.__setattr__(candidate, name, item)
    candidate.__post_init__()
    value.__dict__.clear()
    value.__dict__.update(candidate.__dict__)


def _deep_freeze(value: Any) -> Any:
    """Return an audit-safe immutable copy of a semantic payload."""

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted(
                (_deep_freeze(item) for item in value),
                key=repr,
            )
        )
    return value


def _strict_payload_equal(
    left: Any,
    right: Any,
) -> bool:
    """Compare evidence without Python's cross-type equality coercions.

    ``dict.__eq__`` considers values such as ``True`` and ``1`` equal.  That
    is not a safe basis for aliasing two canonical payloads because replacing
    one with the other changes its primitive type and therefore its evidence
    bytes.  Only recursively type-identical, supported primitive structures
    may share one frozen mapping; unfamiliar objects conservatively remain
    separate.
    """

    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        if all(type(key) is str for key in left) and all(
            type(key) is str for key in right
        ):
            if set(left) != set(right):
                return False
            return all(
                _strict_payload_equal(left[key], right[key])
                for key in left
            )
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            for index, (right_key, right_value) in enumerate(unmatched):
                if not _strict_payload_equal(left_key, right_key):
                    continue
                if not _strict_payload_equal(left_value, right_value):
                    return False
                unmatched.pop(index)
                break
            else:
                return False
        return not unmatched
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _strict_payload_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, Enum):
        return _strict_payload_equal(left.value, right.value)
    if isinstance(left, pd.Timestamp):
        return left.isoformat() == right.isoformat()
    if isinstance(left, Path):
        return str(left) == str(right)
    if isinstance(left, float):
        return left.hex() == right.hex()
    if isinstance(left, (str, bytes, int, bool, type(None))):
        return left == right
    return False


class FrozenDict(dict):
    """A pickle/JSON-friendly mapping that rejects post-construction edits.

    ``dataclass(frozen=True)`` protects only the event attributes themselves;
    a normal ``dict`` stored in ``details`` could still be mutated and would
    silently rewrite history.  This small dict subclass retains compatibility
    with existing serializers and readers while closing that hole.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        dict.__init__(
            self,
            {
                key: _deep_freeze(item)
                for key, item in values.items()
            },
        )

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("semantic event payload is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "FrozenDict":
        return self

    def __reduce__(self) -> tuple[type["FrozenDict"], tuple[dict[Any, Any]]]:
        return FrozenDict, (dict(self),)


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> float:
        return 1.0 if self is Direction.LONG else -1.0

    @property
    def opposing_liquidity_side(self) -> str:
        return "above" if self is Direction.LONG else "below"

    @property
    def invalidation_side(self) -> str:
        return "below" if self is Direction.LONG else "above"


class Timeframe(str, Enum):
    H4 = "4H"
    H1 = "1H"
    M5 = "5m"
    M1 = "1m"
    # Optional context bridge enabled through the explicit ScaleSpec registry.
    M15 = "15m"

    @property
    def minutes(self) -> int:
        """The completed-bar length of this scale."""

        return {"1m": 1, "5m": 5, "15m": 15, "1H": 60, "4H": 240}[self.value]


CORE_TIMEFRAMES = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M5,
    Timeframe.M1,
)


class Playbook(str, Enum):
    DISPLACEMENT_FIRST_PULLBACK = "displacement_first_pullback"
    LIQUIDITY_SWEEP_REVERSAL = "liquidity_sweep_reversal"
    FAILED_AUCTION_VALUE_RETURN = "failed_auction_value_return"


class PlaybookPhase(str, Enum):
    INACTIVE = "inactive"
    FORMING = "forming"
    ARMED = "armed"
    WAITING_LOCATION = "waiting_location"
    WAITING_TRIGGER = "waiting_trigger"
    EXECUTABLE = "executable"
    ENTERED = "entered"
    WEAKENING = "weakening"
    DELIVERING = "delivering"
    COMPLETED = "completed"
    INVALIDATED = "invalidated"


class MarketMode(str, Enum):
    """Small, descriptive state space for the market-wide context."""

    DIRECTIONAL = "directional"
    BALANCED = "balanced"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"


class ScaleRelation(str, Enum):
    """How one completed scale relates to the current structural authority."""

    ALIGNED = "aligned"
    NORMAL_PULLBACK = "normal_pullback"
    MATERIAL_OPPOSITION = "material_opposition"
    UNKNOWN = "unknown"


def aware_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = (
        value
        if isinstance(value, pd.Timestamp)
        else pd.Timestamp(value)
    )
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return timestamp


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("non-finite continuous value")
    return float(min(high, max(low, value)))


def _finite_decimal(value: Any, *, name: str) -> Decimal:
    """Parse one numeric contract value without binary-float rounding."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _tick_size_decimal(tick_size: Any) -> Decimal:
    parsed = _finite_decimal(tick_size, name="tick_size")
    if parsed <= 0:
        raise ValueError("tick_size must be positive")
    return parsed


def price_to_ticks(
    price: Any,
    tick_size: Any,
    *,
    name: str = "price",
) -> int:
    """Return the exact integer grid coordinate for one vendor price.

    Decimal text parsing plus integer-ratio arithmetic is intentional:
    admission never depends on binary floats, banker's rounding, or the
    process-wide Decimal precision context.
    """

    parsed_price = _finite_decimal(price, name=name)
    parsed_tick = _tick_size_decimal(tick_size)
    price_numerator, price_denominator = parsed_price.as_integer_ratio()
    tick_numerator, tick_denominator = parsed_tick.as_integer_ratio()
    coordinate_numerator = price_numerator * tick_denominator
    coordinate_denominator = price_denominator * tick_numerator
    integral, remainder = divmod(
        coordinate_numerator,
        coordinate_denominator,
    )
    if remainder:
        raise ValueError(
            f"{name} is off-grid for tick_size {parsed_tick}"
        )
    return integral


def ticks_to_price(
    ticks: int,
    tick_size: Any,
    *,
    name: str = "ticks",
) -> float:
    """Return the canonical float projection of an integer tick coordinate."""

    if type(ticks) is not int:
        raise ValueError(f"{name} must be an integer")
    tick_numerator, tick_denominator = (
        _tick_size_decimal(tick_size).as_integer_ratio()
    )
    return (ticks * tick_numerator) / tick_denominator


def ohlc_to_ticks(
    open_price: Any,
    high_price: Any,
    low_price: Any,
    close_price: Any,
    tick_size: Any,
) -> tuple[int, int, int, int]:
    """Validate vendor/detector OHLC and return its integer representation."""

    output = (
        price_to_ticks(open_price, tick_size, name="open"),
        price_to_ticks(high_price, tick_size, name="high"),
        price_to_ticks(low_price, tick_size, name="low"),
        price_to_ticks(close_price, tick_size, name="close"),
    )
    open_ticks, high_ticks, low_ticks, close_ticks = output
    if (
        high_ticks < max(open_ticks, close_ticks)
        or low_ticks > min(open_ticks, close_ticks)
        or high_ticks < low_ticks
    ):
        raise ValueError("integer OHLC geometry is invalid")
    return output


def _validate_normalized_ohlc(
    *,
    values: tuple[Any, Any, Any, Any],
    price_tick_size: float | None,
    normalized_ohlc_ticks: tuple[int, int, int, int] | None,
) -> tuple[float | None, tuple[int, int, int, int] | None]:
    if price_tick_size is None:
        if normalized_ohlc_ticks is not None:
            raise ValueError(
                "normalized OHLC ticks require their price tick size"
            )
        return None, None
    canonical_tick = float(_tick_size_decimal(price_tick_size))
    expected = ohlc_to_ticks(*values, canonical_tick)
    if normalized_ohlc_ticks is None:
        return canonical_tick, expected
    supplied = tuple(normalized_ohlc_ticks)
    if (
        len(supplied) != 4
        or any(type(value) is not int for value in supplied)
        or supplied != expected
    ):
        raise ValueError("stored normalized OHLC ticks disagree with prices")
    return canonical_tick, supplied


@dataclass(frozen=True)
class Bar:
    """One completed 1m bar; ``start`` is the minute open timestamp."""

    start: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    instrument_id: int
    synthetic_no_trade: bool = False
    data_gap_before_minutes: int = 0
    price_tick_size: float | None = field(default=None, compare=False)
    normalized_ohlc_ticks: tuple[int, int, int, int] | None = field(
        default=None,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", aware_timestamp(self.start, name="bar.start"))
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("bar contains non-finite OHLCV")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("bar high/low does not contain open and close")
        if self.high < self.low or self.volume < 0:
            raise ValueError("bar range or volume is invalid")
        if not self.symbol or int(self.instrument_id) < 0:
            raise ValueError("bar contract identity is invalid")
        if int(self.data_gap_before_minutes) < 0:
            raise ValueError("bar data-gap duration cannot be negative")
        if self.synthetic_no_trade and self.data_gap_before_minutes:
            raise ValueError("synthetic no-trade bar cannot also begin a data gap")
        price_tick_size, normalized_ticks = _validate_normalized_ohlc(
            values=(self.open, self.high, self.low, self.close),
            price_tick_size=self.price_tick_size,
            normalized_ohlc_ticks=self.normalized_ohlc_ticks,
        )
        object.__setattr__(self, "price_tick_size", price_tick_size)
        object.__setattr__(self, "normalized_ohlc_ticks", normalized_ticks)

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(1, unit="min")

    @property
    def open_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[0]

    @property
    def high_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[1]

    @property
    def low_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[2]

    @property
    def close_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[3]

    def on_price_grid(self, tick_size: float) -> "Bar":
        """Return this immutable bar with exact normalized tick coordinates."""

        requested = _tick_size_decimal(tick_size)
        if self.price_tick_size is not None:
            if _tick_size_decimal(self.price_tick_size) != requested:
                raise ValueError(
                    "bar price grid disagrees with reader tick size"
                )
            return self
        return replace(
            self,
            price_tick_size=float(requested),
            normalized_ohlc_ticks=None,
        )


class BarCoverage(str, Enum):
    """How completely one native bar's bucket was actually observed.

    A no-trade minute carries no vendor record, so the reader synthesizes it to
    keep the registered clock whole.  That bar still covers its whole bucket and
    is not the same thing as one that never observed its minutes.  Both the
    producer that builds a definitional path and the contract that re-validates
    it decide admission from this one classification, so the two cannot drift
    apart.
    """

    REAL = "real"
    DENSIFIED = "densified"
    INCOMPLETE = "incomplete"

    @property
    def admits_definitional_path(self) -> bool:
        """A v2 definitional path needs full bucket coverage, real or not."""

        return self is not BarCoverage.INCOMPLETE

    @property
    def admits_atr_window(self) -> bool:
        """ATR needs real price discovery: a no-trade bar has no true range."""

        return self is BarCoverage.REAL


def classify_bar_coverage(
    *,
    complete: object,
    observed_minutes: object,
    expected_minutes: object,
    real_minutes: object,
    synthetic_minutes: object,
) -> BarCoverage:
    """Classify one bar's coverage from its minute accounting alone."""

    if complete is not True:
        return BarCoverage.INCOMPLETE
    values = (observed_minutes, expected_minutes, real_minutes, synthetic_minutes)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return BarCoverage.INCOMPLETE
    observed, expected, real, synthetic = values
    if observed != expected or real + synthetic != observed or synthetic < 0:
        return BarCoverage.INCOMPLETE
    return BarCoverage.REAL if synthetic == 0 else BarCoverage.DENSIFIED


def candle_coverage(candle: "Candle") -> BarCoverage:
    """Classify a producer-side candle."""

    return classify_bar_coverage(
        complete=candle.complete,
        observed_minutes=candle.observed_minutes,
        expected_minutes=candle.expected_minutes,
        real_minutes=candle.real_minutes,
        synthetic_minutes=candle.synthetic_minutes,
    )


def bar_evidence_coverage(evidence: Mapping[str, object]) -> BarCoverage:
    """Classify a BAR event from the evidence it transports.

    A real root records neither its minute accounting nor ``complete``; the
    emitter adds those only when the bar is not real, so an absent accounting
    with ``real_completed`` true is a fully real bar.
    """

    if (
        evidence.get("real_completed") is True
        and evidence.get("clock_only") is False
    ):
        return BarCoverage.REAL
    return classify_bar_coverage(
        complete=evidence.get("complete"),
        observed_minutes=evidence.get("observed_minutes"),
        expected_minutes=evidence.get("expected_minutes"),
        real_minutes=evidence.get("real_minutes"),
        synthetic_minutes=evidence.get("synthetic_minutes"),
    )


@dataclass(frozen=True)
class Candle:
    timeframe: Timeframe
    start: pd.Timestamp
    end: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    instrument_id: int
    observed_minutes: int
    expected_minutes: int
    complete: bool
    real_minutes: int | None = None
    synthetic_minutes: int = 0
    price_tick_size: float | None = field(default=None, compare=False)
    normalized_ohlc_ticks: tuple[int, int, int, int] | None = field(
        default=None,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", aware_timestamp(self.start, name="candle.start"))
        object.__setattr__(self, "end", aware_timestamp(self.end, name="candle.end"))
        if self.end <= self.start:
            raise ValueError("candle end must follow start")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("candle contains non-finite OHLCV")
        if (
            self.high < max(self.open, self.close)
            or self.low > min(self.open, self.close)
            or self.high < self.low
            or self.volume < 0
        ):
            raise ValueError("candle OHLC is invalid")
        if not self.symbol or int(self.instrument_id) < 0:
            raise ValueError("candle contract identity is invalid")
        if (
            type(self.observed_minutes) is not int
            or type(self.expected_minutes) is not int
            or type(self.synthetic_minutes) is not int
            or self.observed_minutes > self.expected_minutes
            or self.observed_minutes <= 0
            or self.synthetic_minutes < 0
            or self.synthetic_minutes > self.observed_minutes
            or (
                self.complete
                and self.observed_minutes != self.expected_minutes
            )
        ):
            raise ValueError("candle minute coverage is invalid")
        real_minutes = (
            self.observed_minutes - self.synthetic_minutes
            if self.real_minutes is None
            else self.real_minutes
        )
        if (
            type(real_minutes) is not int
            or real_minutes < 0
            or real_minutes + self.synthetic_minutes
            != self.observed_minutes
        ):
            raise ValueError(
                "candle real/synthetic provenance is inconsistent"
            )
        object.__setattr__(self, "real_minutes", real_minutes)
        price_tick_size, normalized_ticks = _validate_normalized_ohlc(
            values=(self.open, self.high, self.low, self.close),
            price_tick_size=self.price_tick_size,
            normalized_ohlc_ticks=self.normalized_ohlc_ticks,
        )
        object.__setattr__(self, "price_tick_size", price_tick_size)
        object.__setattr__(self, "normalized_ohlc_ticks", normalized_ticks)

    @property
    def real_completed(self) -> bool:
        return bool(self.complete and self.synthetic_minutes == 0)

    @property
    def open_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[0]

    @property
    def high_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[1]

    @property
    def low_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[2]

    @property
    def close_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[3]

    def ohlc_ticks_for(self, tick_size: float) -> tuple[int, int, int, int]:
        """Read stored ticks on the same grid or validate a direct test candle."""

        requested = _tick_size_decimal(tick_size)
        if self.price_tick_size is not None:
            if _tick_size_decimal(self.price_tick_size) != requested:
                raise ValueError("candle price grid disagrees with detector tick size")
            if self.normalized_ohlc_ticks is None:
                raise AssertionError("normalized candle lost its integer OHLC")
            return self.normalized_ohlc_ticks
        return ohlc_to_ticks(
            self.open,
            self.high,
            self.low,
            self.close,
            float(requested),
        )


def candle_identity(candle: Candle, *, tick_size: float) -> str:
    """Return one shared candle identity for every semantic reducer."""

    ticks = candle.ohlc_ticks_for(tick_size)

    parts = (
        "candle-v1",
        candle.timeframe.value,
        candle.start.isoformat(),
        candle.end.isoformat(),
        *(str(value) for value in ticks),
        format(float(candle.volume), ".17g"),
        candle.symbol,
        str(candle.instrument_id),
        str(candle.observed_minutes),
        str(candle.expected_minutes),
        str(candle.real_minutes),
        str(candle.synthetic_minutes),
        str(candle.complete),
    )
    raw = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LiquidityLevel:
    level_id: str
    timeframe: Timeframe
    side: str
    price: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    touches: int
    swept: bool = False

    def __post_init__(self) -> None:
        if self.side not in {"above", "below"}:
            raise ValueError("liquidity side must be above or below")
        if not math.isfinite(float(self.price)) or self.price <= 0 or self.touches < 0:
            raise ValueError("liquidity price or touch count is invalid")
        object.__setattr__(
            self, "formed_at", aware_timestamp(self.formed_at, name="level.formed_at")
        )
        object.__setattr__(
            self, "confirmed_at", aware_timestamp(self.confirmed_at, name="level.confirmed_at")
        )
        if self.confirmed_at < self.formed_at:
            raise ValueError("liquidity confirmation cannot predate formation")


@dataclass(frozen=True)
class StructuralLevel:
    price: float
    side: str
    source_level_id: str
    observed_at: pd.Timestamp
    rationale: str

    def __post_init__(self) -> None:
        if self.side not in {"above", "below"}:
            raise ValueError("structural side must be above or below")
        if (
            not math.isfinite(float(self.price))
            or self.price <= 0
            or not self.source_level_id
        ):
            raise ValueError("structural level price and source are required")
        object.__setattr__(
            self, "observed_at", aware_timestamp(self.observed_at, name="structure.observed_at")
        )


def to_primitive(value: Any) -> Any:
    # The component parity walk reaches this function millions of times per
    # clock prefix.  Most leaves and containers are exact built-in types, so
    # dispatch those without repeatedly invoking ABC/subclass machinery.  The
    # fallback below intentionally preserves the original ``isinstance``
    # ordering for subclasses such as IntEnum and custom Mapping objects.
    value_type = type(value)
    if (
        value_type is str
        or value_type is bytes
        or value_type is int
        or value_type is bool
        or value is None
    ):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("cannot serialize a non-finite value")
        return value
    if value_type is pd.Timestamp:
        return value.isoformat()
    if value_type is dict or value_type is FrozenDict:
        return {
            str(key.value if isinstance(key, Enum) else key): to_primitive(item)
            for key, item in value.items()
        }
    if value_type is tuple or value_type is list:
        return [to_primitive(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize a non-finite value")
        return value
    if isinstance(value, (str, bytes, int, bool, type(None))):
        return value
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): to_primitive(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if is_dataclass(value):
        # ``asdict`` recursively deep-copies the entire object graph before
        # this function recursively normalizes it a second time. Market
        # snapshots contain immutable nested histories, so field-wise reading
        # is both semantically exact and materially cheaper during replay.
        return {
            item.name: to_primitive(getattr(value, item.name))
            for item in fields(value)
            if item.metadata.get("primitive", True)
        }
    return value


def content_hash(value: Any) -> str:
    payload = json.dumps(to_primitive(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "Bar",
    "BarCoverage",
    "CORE_TIMEFRAMES",
    "Candle",
    "Direction",
    "FrozenDict",
    "LiquidityLevel",
    "MarketMode",
    "Playbook",
    "PlaybookPhase",
    "SMC_SEMANTIC_VERSION",
    "ScaleRelation",
    "StructuralLevel",
    "Timeframe",
    "aware_timestamp",
    "bar_evidence_coverage",
    "candle_coverage",
    "candle_identity",
    "clamp",
    "classify_bar_coverage",
    "content_hash",
    "ohlc_to_ticks",
    "price_to_ticks",
    "ticks_to_price",
    "to_primitive",
]
