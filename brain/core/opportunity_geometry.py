"""Object aliases in, prices out.

The LLM names an entry object, an invalidation object and a target object.
This module — and only this module — turns those into an entry, a stop and a
target price from the objects' geometry, and computes the reward-to-risk ratio.
Every rule is named in ``rule_ids`` so a journal reader can recompute it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from contract.brain.state import Opportunity, OpportunityState, TradeDirection
from contract.decision import GeometryError, OpportunityGeometry

ZONE_KINDS = frozenset({"fvg", "ob"})
POOL_KINDS = frozenset({"bsl", "ssl"})
SWING_KINDS = frozenset({"swing_high", "swing_low"})
RANGE_KINDS = frozenset({"range"})


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


def _stop(obj: ObjectGeometry, direction: TradeDirection, tick: float) -> tuple[float, str]:
    family = _family(obj.kind)
    if family == "swing":
        price = obj.anchor - tick if direction is TradeDirection.LONG else obj.anchor + tick
        return price, "stop.swing.price"
    price = obj.lower - tick if direction is TradeDirection.LONG else obj.upper + tick
    return price, f"stop.{family}.far_edge"


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
) -> OpportunityGeometry:
    """Resolve an opportunity to prices; ``GeometryError`` when it cannot be."""

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
    stop, stop_rule = _stop(resolved["invalidation"], direction, tick)
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
) -> str | None:
    """``None`` when the opportunity resolves (or is NONE), else the reason."""

    if opportunity.state is OpportunityState.NONE:
        return None
    try:
        resolve_geometry(opportunity, objects, close=close, tick=tick)
    except GeometryError as error:
        return str(error)
    return None


__all__ = [
    "ObjectGeometry",
    "POOL_KINDS",
    "RANGE_KINDS",
    "SWING_KINDS",
    "ZONE_KINDS",
    "coherence_error",
    "resolve_geometry",
]
