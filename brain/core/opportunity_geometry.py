"""Object aliases in, prices out.

The LLM names an entry object, an invalidation object and a target object.
This module — and only this module — turns those into an entry, a stop and a
target price from the objects' geometry, and computes the reward-to-risk ratio.
Every rule is named in ``rule_ids`` so a journal reader can recompute it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from contract.brain.state import InvalidationMode, Opportunity, OpportunityState, TradeDirection
from contract.decision import GeometryError, OpportunityGeometry

ZONE_KINDS = frozenset({"fvg", "ob"})
POOL_KINDS = frozenset({"bsl", "ssl"})
SWING_KINDS = frozenset({"swing_high", "swing_low"})
RANGE_KINDS = frozenset({"range"})
# Minutes per scale: the close-beyond buffer scales one 1m ATR by the
# square root of the invalidation object's bar length — the standard
# estimate of one bar's range on that scale, the room a wick needs before
# the bar closes.
TIMEFRAME_MINUTES: Mapping[str, int] = {"4H": 240, "1H": 60, "15m": 15, "5m": 5, "1m": 1}
# How many scaled 1m ATRs beyond the object's far edge the hard stop of a
# CLOSE_BEYOND invalidation sits; a keyword argument of ``resolve_geometry``.
CLOSE_BEYOND_BUFFER_ATR = 1.0
# The stop floor (2026-09-21): the hard stop is never nearer the entry than
# this many bars of the thesis's governing scale — one bar's range being
# ``atr_1m × √minutes`` as above.  Forty-four benchmark fills had a median
# stop of 1.5 one-minute ATRs on 15m theses whose one bar spans 3.9; 70 %
# went 1R against within the hour, 17 of them before a 2R run.
STOP_FLOOR_GOVERNING_BARS = 1.0


@dataclass(frozen=True)
class ObjectGeometry:
    """The geometry of one aliased Eye object.

    ``anchor`` is the object's own reference price: a range's value price, a
    pool's midpoint, a swing's price, a zone's midpoint."""

    alias: str
    kind: str
    timeframe: str
    lower: float
    upper: float
    anchor: float

    def __post_init__(self) -> None:
        if not self.lower <= self.upper:
            raise ValueError(f"object {self.alias} has lower above upper")


def _family(kind: str) -> str:
    if kind in ZONE_KINDS:
        return "zone"
    if kind in POOL_KINDS:
        return "pool"
    if kind in SWING_KINDS:
        return "swing"
    if kind in RANGE_KINDS:
        return "range"
    raise GeometryError(f"object kind {kind!r} has no geometry rule")


def _entry(obj: ObjectGeometry, direction: TradeDirection, close: float) -> tuple[float, str]:
    """The limit a trade rests at on the object.  A zone below price (LONG)
    is entered at its near edge; a zone that *contains* price at its
    midpoint when that is on the resting side of the close, else at its far
    edge (2026-09-20) — the near edge of a containing zone is above the
    market for a LONG, and a limit there is a buy at the market.  A range
    has no entry level: its value price lies wherever the profile puts it."""
    family = _family(obj.kind)
    long = direction is TradeDirection.LONG
    if family == "zone":
        if obj.lower <= close <= obj.upper:
            midpoint = (obj.lower + obj.upper) / 2.0
            if (midpoint <= close) if long else (midpoint >= close):
                return midpoint, "entry.zone.inside_midpoint"
            return (obj.lower if long else obj.upper), "entry.zone.inside_far_edge"
        return (obj.upper if long else obj.lower), "entry.zone.near_edge"
    if family == "range":
        raise GeometryError(f"a range is not an entry object ({obj.alias}); name the zone, pool or swing inside it")
    if family == "pool":
        return obj.anchor, "entry.pool.midpoint"
    return obj.anchor, "entry.swing.price"


def entry_side_error(direction: TradeDirection, entry: float, close: float) -> str | None:
    """``None`` when a limit at ``entry`` rests — at or below the close for
    a LONG, at or above it for a SHORT — else why it would fill at the
    market (2026-09-20).  Judged when an opportunity is proposed and when an
    order is submitted, never on the bars in between: a working limit the
    tape crosses must fill, not be cancelled."""
    if direction is TradeDirection.LONG and entry > close:
        return f"LONG entry {entry} lies above price {close} — a trade is expressed on a pullback, not at the market"
    if direction is TradeDirection.SHORT and entry < close:
        return f"SHORT entry {entry} lies below price {close} — a trade is expressed on a pullback, not at the market"
    return None


def _round_away(price: float, tick: float, direction: TradeDirection) -> float:
    """Round a stop to the tick on the losing side: down for a LONG, up for a SHORT."""
    steps = price / tick
    rounded = math.floor(steps + 1e-9) if direction is TradeDirection.LONG else math.ceil(steps - 1e-9)
    return round(rounded * tick, 10)


def _stop(
    obj: ObjectGeometry, direction: TradeDirection, tick: float, *, mode: InvalidationMode, atr_1m: float | None, buffer_atr: float
) -> tuple[float, str]:
    family = _family(obj.kind)
    edge = obj.anchor if family == "swing" else (obj.lower if direction is TradeDirection.LONG else obj.upper)
    name = "price" if family == "swing" else "far_edge"
    if mode is InvalidationMode.TOUCH:
        price = edge - tick if direction is TradeDirection.LONG else edge + tick
        return price, f"stop.{family}.{name}"
    if atr_1m is None or atr_1m <= 0.0:
        raise GeometryError("a CLOSE_BEYOND invalidation needs a positive 1m atr")
    minutes = TIMEFRAME_MINUTES.get(obj.timeframe)
    if minutes is None:
        raise GeometryError(f"object {obj.alias} has no known scale for a CLOSE_BEYOND buffer")
    buffer = buffer_atr * atr_1m * math.sqrt(minutes)
    price = edge - buffer if direction is TradeDirection.LONG else edge + buffer
    return _round_away(price, tick, direction), f"stop.{family}.close_beyond"


def _floor_stop(
    entry: float, stop: float, rule: str, direction: TradeDirection, *, governing: str | None, atr_1m: float | None, tick: float, floor_bars: float
) -> tuple[float, str]:
    """The stop, no nearer the entry than ``floor_bars`` bars of the
    governing scale; the object's own rule when it already lies beyond."""
    if governing is None or floor_bars <= 0.0:
        return stop, rule
    if atr_1m is None or atr_1m <= 0.0:
        raise GeometryError("a stop floor needs a positive 1m atr")
    minutes = TIMEFRAME_MINUTES.get(governing)
    if minutes is None:
        raise GeometryError(f"governing scale {governing!r} has no known bar length for the stop floor")
    floor = floor_bars * atr_1m * math.sqrt(minutes)
    if abs(entry - stop) >= floor:
        return stop, rule
    price = entry - floor if direction is TradeDirection.LONG else entry + floor
    return _round_away(price, tick, direction), "stop.floor.governing_bar"


def _target(obj: ObjectGeometry, direction: TradeDirection) -> tuple[float, str]:
    family = _family(obj.kind)
    if family in ("zone", "range"):
        return (obj.lower if direction is TradeDirection.LONG else obj.upper), f"target.{family}.near_edge"
    if family == "pool":
        return obj.anchor, "target.pool.midpoint"
    return obj.anchor, "target.swing.price"


def resolve_geometry(
    opportunity: Opportunity,
    objects: Mapping[str, ObjectGeometry],
    *,
    close: float,
    tick: float,
    atr_1m: float | None = None,
    buffer_atr: float = CLOSE_BEYOND_BUFFER_ATR,
    floor_bars: float = STOP_FLOOR_GOVERNING_BARS,
) -> OpportunityGeometry:
    """Resolve an opportunity to prices; ``GeometryError`` when it cannot be.
    A ``CLOSE_BEYOND`` invalidation needs ``atr_1m`` for its buffer, and so
    does the stop floor of any opportunity with a ``governing_timeframe``
    (2026-09-21): the stop sits at the object's rule or ``floor_bars``
    governing bars from the entry, whichever is farther."""

    if opportunity.state is OpportunityState.NONE:
        raise GeometryError("opportunity state is NONE")
    direction = opportunity.direction
    if direction is None:
        raise GeometryError("opportunity has no direction")
    if tick <= 0.0:
        raise GeometryError("tick must be positive")
    resolved: dict[str, ObjectGeometry] = {}
    for role, alias in (
        ("entry", opportunity.entry_object_id),
        ("invalidation", opportunity.invalidation_object_id),
        ("target", opportunity.target_object_id),
    ):
        obj = objects.get(alias or "")
        if obj is None:
            raise GeometryError(f"{role} object {alias!r} is not a visible object")
        resolved[role] = obj
    entry, entry_rule = _entry(resolved["entry"], direction, close)
    stop, stop_rule = _stop(
        resolved["invalidation"], direction, tick, mode=opportunity.invalidation_mode, atr_1m=atr_1m, buffer_atr=buffer_atr
    )
    stop, stop_rule = _floor_stop(
        entry, stop, stop_rule, direction, governing=opportunity.governing_timeframe, atr_1m=atr_1m, tick=tick, floor_bars=floor_bars
    )
    target, target_rule = _target(resolved["target"], direction)
    if direction is TradeDirection.LONG and not stop < entry < target:
        raise GeometryError(
            f"LONG requires stop < entry < target, got stop={stop} entry={entry} target={target}"
        )
    if direction is TradeDirection.SHORT and not target < entry < stop:
        raise GeometryError(
            f"SHORT requires target < entry < stop, got stop={stop} entry={entry} target={target}"
        )
    risk = abs(entry - stop)
    if risk <= 0.0:
        raise GeometryError("risk distance is not positive")
    return OpportunityGeometry(
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        reward_risk=abs(target - entry) / risk,
        rule_ids=(entry_rule, stop_rule, target_rule),
    )


def coherence_error(
    opportunity: Opportunity,
    objects: Mapping[str, ObjectGeometry],
    *,
    close: float,
    tick: float,
    atr_1m: float | None = None,
) -> str | None:
    """``None`` when the opportunity resolves (or is NONE) and its limit
    rests on the right side of ``close``, else the reason."""

    if opportunity.state is OpportunityState.NONE:
        return None
    try:
        geometry = resolve_geometry(opportunity, objects, close=close, tick=tick, atr_1m=atr_1m)
    except GeometryError as error:
        return str(error)
    assert opportunity.direction is not None
    return entry_side_error(opportunity.direction, geometry.entry_price, close)


__all__ = [
    "CLOSE_BEYOND_BUFFER_ATR",
    "STOP_FLOOR_GOVERNING_BARS",
    "TIMEFRAME_MINUTES",
    "ObjectGeometry",
    "POOL_KINDS",
    "RANGE_KINDS",
    "SWING_KINDS",
    "ZONE_KINDS",
    "coherence_error",
    "entry_side_error",
    "resolve_geometry",
]
