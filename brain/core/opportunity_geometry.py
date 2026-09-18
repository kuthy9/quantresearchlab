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


def _entry(obj: ObjectGeometry, direction: TradeDirection) -> tuple[float, str]:
    family = _family(obj.kind)
    if family == "zone":
        return (obj.upper if direction is TradeDirection.LONG else obj.lower), "entry.zone.near_edge"
    if family == "range":
        return obj.anchor, "entry.range.value"
    if family == "pool":
        return obj.anchor, "entry.pool.midpoint"
    return obj.anchor, "entry.swing.price"


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
) -> OpportunityGeometry:
    """Resolve an opportunity to prices; ``GeometryError`` when it cannot be.
    A ``CLOSE_BEYOND`` invalidation needs ``atr_1m`` for its buffer."""

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
    entry, entry_rule = _entry(resolved["entry"], direction)
    stop, stop_rule = _stop(
        resolved["invalidation"], direction, tick, mode=opportunity.invalidation_mode, atr_1m=atr_1m, buffer_atr=buffer_atr
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
    """``None`` when the opportunity resolves (or is NONE), else the reason."""

    if opportunity.state is OpportunityState.NONE:
        return None
    try:
        resolve_geometry(opportunity, objects, close=close, tick=tick, atr_1m=atr_1m)
    except GeometryError as error:
        return str(error)
    return None


__all__ = [
    "CLOSE_BEYOND_BUFFER_ATR",
    "TIMEFRAME_MINUTES",
    "ObjectGeometry",
    "POOL_KINDS",
    "RANGE_KINDS",
    "SWING_KINDS",
    "ZONE_KINDS",
    "coherence_error",
    "resolve_geometry",
]
