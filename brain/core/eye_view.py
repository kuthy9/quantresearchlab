"""What the Brain reads from one Eye observation.

The only Brain module that touches Eye types.  ``build_eye_context`` turns a
``MarketObservation`` into the aliased, JSON-ready ``EyeContext`` the LLM sees
and the controller / reducer act on; nothing in it postdates ``known_at``, and
``assert_causal`` is the check every serialized payload passes before it is
sent or journaled."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any

import pandas as pd

from brain.core.brain_entry_sequence import brain_interaction_view
from brain.core.object_registry import ObjectRegistry
from brain.core.opportunity_geometry import ObjectGeometry
from contract.brain.state import EvidenceItem, isoformat_utc
from contract.eye import (
    LiquidityInventoryLifecycle,
    MarketEvent,
    MarketObservation,
    PathSequenceLifecycle,
    PathSequenceState,
)
from contract.market import Timeframe
from contract.market.primitives import FrozenDict

# The scales whose objects the LLM may name; 1m is microstructure and publishes
# only its structure and delivery reads.
ALIASED_TIMEFRAMES: tuple[Timeframe, ...] = (Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5)
CONTEXT_TIMEFRAMES: tuple[Timeframe, ...] = ALIASED_TIMEFRAMES + (Timeframe.M1,)
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class CausalityError(ValueError):
    """A payload carries a timestamp after the bar it is published for."""


def visible_liquidity_ids(observation: MarketObservation) -> set[str]:
    """Inventory items a downstream consumer may act on: VISIBLE, confirmed by
    ``asof``, and not published twice with different identity or geometry."""
    seen: dict[str, tuple] = {}
    ambiguous: set[str] = set()
    for item in observation.liquidity_inventory:
        if (
            item.lifecycle is not LiquidityInventoryLifecycle.VISIBLE
            or item.confirmed_at > observation.asof
        ):
            continue
        if item.item_id in ambiguous:
            continue
        identity = (
            item.timeframe,
            item.side,
            round(float(item.price), 9),
            item.formed_at,
            item.confirmed_at,
        )
        prior = seen.get(item.item_id)
        if prior is None:
            seen[item.item_id] = identity
        elif prior != identity:
            del seen[item.item_id]
            ambiguous.add(item.item_id)
    return set(seen)


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


# What the LLM reads: where the OBJECT lies relative to price.  ``offset_atr``
# is signed the same way — positive above price, negative below, zero when the
# object contains price.  (``price_relation`` below keeps the price's own point
# of view for the code that computes it.)
POSITIONS: Mapping[str, str] = FrozenDict({"above": "below_price", "below": "above_price", "inside": "contains_price"})


def object_position(close: float, lower: float, upper: float, atr: float | None) -> tuple[str, float | None]:
    """``(position, offset_atr)`` of the object ``[lower, upper]`` seen from
    ``close``: ``above_price`` with a positive offset, ``below_price`` with a
    negative one, ``contains_price`` with zero; ``None`` while the ATR is not warm."""
    relation, distance = price_relation(close, lower, upper, atr)
    return POSITIONS[relation], None if distance is None else _round(-distance)


def price_relation(
    close: float, lower: float, upper: float, atr: float | None
) -> tuple[str, float | None]:
    """Where ``close`` sits against ``[lower, upper]`` and how far, in ATR.

    Positive distance is above the object's upper bound, negative below its
    lower bound, zero inside; ``None`` while the ATR is not warm."""
    if close > upper:
        relation, distance = "above", close - upper
    elif close < lower:
        relation, distance = "below", close - lower
    else:
        relation, distance = "inside", 0.0
    if atr is None or atr <= 0.0:
        return relation, None
    return relation, _round(distance / atr)


def assert_causal(payload: Any, known_at: pd.Timestamp) -> None:
    """Every ISO-8601 UTC string inside ``payload`` must be ``<= known_at``."""
    limit = pd.Timestamp(known_at).tz_convert("UTC")

    def walk(value: Any, path: str) -> None:
        if isinstance(value, str):
            if _ISO_UTC.match(value) and pd.Timestamp(value) > limit:
                raise CausalityError(f"{path} = {value} postdates known_at {isoformat_utc(limit)}")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload, "$")


def evidence_id(event: MarketEvent) -> str:
    return f"ev_{event.event_id}"


@dataclass(frozen=True)
class EvidenceRule:
    """Which events are the reducer's evidence: transition kinds at the
    configured timeframes (state re-publications and heartbeats excluded)."""

    timeframes: frozenset[str]
    heartbeat_kinds: frozenset[str]
    state_suffix: str

    def is_transition(self, kind: str) -> bool:
        return not kind.endswith(self.state_suffix) and kind not in self.heartbeat_kinds

    def is_evidence(self, event: MarketEvent) -> bool:
        return self.is_transition(event.kind.value) and event.timeframe.value in self.timeframes


@dataclass(frozen=True)
class ObjectView:
    alias: str
    kind: str
    timeframe: str
    lower: float
    upper: float
    anchor: float
    lifecycle: str
    direction: str | None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", FrozenDict(dict(self.attributes)))

    def geometry(self) -> ObjectGeometry:
        return ObjectGeometry(self.alias, self.kind, self.timeframe, self.lower, self.upper, self.anchor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.alias,
            "kind": self.kind,
            "timeframe": self.timeframe,
            "lower": _round(self.lower),
            "upper": _round(self.upper),
            "lifecycle": self.lifecycle,
            "direction": self.direction,
            **{key: value for key, value in self.attributes.items()},
        }


@dataclass(frozen=True)
class EyeContext:
    known_at: pd.Timestamp
    close: float
    atr_1m: float | None
    bar: Mapping[str, float | None]
    session: Mapping[str, Any]
    scales: Mapping[str, Any]
    interaction: tuple[Mapping[str, Any], ...]
    objects: tuple[ObjectView, ...]
    events: tuple[EvidenceItem, ...]
    price_relations: tuple[Mapping[str, Any], ...]
    open_interaction: bool

    def visible_aliases(self) -> frozenset[str]:
        return frozenset(view.alias for view in self.objects)

    def object_map(self) -> Mapping[str, ObjectView]:
        return {view.alias: view for view in self.objects}

    def geometries(self) -> Mapping[str, ObjectGeometry]:
        return {view.alias: view.geometry() for view in self.objects}

    def relation_of(self, alias: str) -> str | None:
        """The object's ``position`` relative to price; None when not visible."""
        for relation in self.price_relations:
            if relation["object_id"] == alias:
                return str(relation["position"])
        return None


def _direction_value(direction: Any) -> str | None:
    return None if direction is None else getattr(direction, "value", str(direction))


def _range_value_price(observation: MarketObservation, timeframe: Timeframe, range_id: str) -> float | None:
    frame = observation.frames.get(timeframe)
    if frame is None:
        return None
    for state in frame.dealing_ranges:
        if state.range_id == range_id:
            return float(state.value_price)
    return None


def _scale_objects(
    observation: MarketObservation, timeframe: Timeframe, registry: ObjectRegistry
) -> tuple[list[ObjectView], dict[str, Any]]:
    """The aliased objects of one scale plus its published summary."""
    state = observation.market_snapshot.timeframe_states[timeframe]
    tf = timeframe.value
    views: list[ObjectView] = []
    structure = state.structure
    protected: dict[str, str | None] = {"protected_high": None, "protected_low": None}
    for role, kind, entity_id, price in (
        ("protected_high", "swing_high", structure.protected_high_id, structure.protected_high),
        ("protected_low", "swing_low", structure.protected_low_id, structure.protected_low),
    ):
        if entity_id is None or price is None:
            continue
        alias = registry.alias_for(entity_id, kind=kind, timeframe=tf)
        protected[role] = alias
        views.append(
            ObjectView(
                alias, kind, tf, float(price), float(price), float(price), "protected", None,
                {"intact": structure.protected_swing_intact},
            )
        )
    range_state = state.range
    range_alias: str | None = None
    if range_state.range_id is not None:
        range_alias = registry.alias_for(range_state.range_id, kind="range", timeframe=tf)
        value = _range_value_price(observation, timeframe, range_state.range_id)
        low, high = float(range_state.low), float(range_state.high)
        views.append(
            ObjectView(
                range_alias, "range", tf, low, high, value if value is not None else (low + high) / 2.0,
                str(range_state.lifecycle), None,
                {
                    "location_label": range_state.location_label,
                    "normalized_location": _round(range_state.normalized_location),
                    "range_kind": range_state.range_kind,
                },
            )
        )
    zone_aliases: dict[str, list[str]] = {"fvg": [], "ob": []}
    for kind, zones in (("fvg", state.zones.active_fvg), ("ob", state.zones.active_ob)):
        for zone in zones:
            alias = registry.alias_for(zone.zone_id, kind=kind, timeframe=tf)
            zone_aliases[kind].append(alias)
            lower, upper = float(zone.lower), float(zone.upper)
            views.append(
                ObjectView(alias, kind, tf, lower, upper, (lower + upper) / 2.0, str(zone.lifecycle), _direction_value(zone.direction))
            )
    liquidity = state.liquidity
    swept = set(liquidity.recently_swept_ids)
    pools: dict[str, list[str]] = {"bsl": [], "ssl": [], "recently_swept": []}
    for candidate in liquidity.candidates:
        kind = "bsl" if candidate.side == "above" else "ssl"
        alias = registry.alias_for(candidate.candidate_id, kind=kind, timeframe=tf)
        pools[kind].append(alias)
        if candidate.candidate_id in swept:
            pools["recently_swept"].append(alias)
        views.append(
            ObjectView(
                alias, kind, tf, float(candidate.lower_bound), float(candidate.upper_bound), float(candidate.price),
                str(candidate.lifecycle), None,
                {
                    "rank": candidate.rank,
                    "source_kind": candidate.source_kind,
                    "strength": _round(candidate.strength),
                    "age_bars": int(candidate.age_bars),
                    "recently_swept": candidate.candidate_id in swept,
                },
            )
        )
    summary = {
        "structure": {
            "external_direction": _direction_value(structure.external_direction),
            "internal_direction": _direction_value(structure.internal_direction),
            "protected_high": protected["protected_high"],
            "protected_low": protected["protected_low"],
            "protected_swing_intact": structure.protected_swing_intact,
            "last_bos_direction": _direction_value(structure.last_bos_direction),
            "last_mss_direction": _direction_value(structure.last_mss_direction),
        },
        "delivery": {
            "phase": state.delivery.phase.value,
            "active_leg_direction": _direction_value(state.delivery.active_leg_direction),
            "displacement_score": _round(state.delivery.displacement_score),
        },
        "range": {
            "object_id": range_alias,
            "location_label": range_state.location_label,
            "normalized_location": _round(range_state.normalized_location),
        },
        "liquidity": pools,
        "zones": zone_aliases,
    }
    return views, summary


def _session_payload(observation: MarketObservation) -> dict[str, Any]:
    session = observation.market_snapshot.session
    return {
        "session_id": session.session_id,
        "name": session.name,
        "phase": session.phase,
        "elapsed_minutes": int(session.elapsed_minutes),
        "session_open": _round(session.session_open),
        "session_high": _round(session.session_high),
        "session_low": _round(session.session_low),
        "opening_range_high": _round(session.opening_range_high),
        "opening_range_low": _round(session.opening_range_low),
        "prior_day_high": _round(session.prior_day_high),
        "prior_day_low": _round(session.prior_day_low),
        "overnight_high": _round(session.overnight_high),
        "overnight_low": _round(session.overnight_low),
        "realized_volatility": _round(session.realized_volatility),
        "relative_volume": _round(session.relative_volume),
        "known_at": isoformat_utc(session.known_at),
        "data_complete": bool(session.data_complete),
    }


def interaction_rows(
    paths: tuple[PathSequenceState, ...],
    *,
    manipulation_sources: Mapping[str, str],
    registry: ObjectRegistry,
    known_at: pd.Timestamp,
    since: pd.Timestamp | None,
) -> tuple[dict[str, Any], ...]:
    """The ACTIVE interaction paths as the Brain reads them.

    A path's ``context_id`` is the interaction tracker's own key, never an
    Eye object: the object is the first step's source — the zone itself for a
    ``zone_return``, and for a ``pool_reversal`` the manipulation's source
    inventory item, looked up through ``manipulation_sources``
    (manipulation id → inventory item id).  ``stepped_since_last_call`` is
    what makes an interaction *open* for the sleep gate: a step observed
    after ``since`` (the previous LLM call), or on this very bar when there
    was no previous call.  The Eye keeps a path ACTIVE for as long as its
    context lives — hours, on a held pool reversal — so "any ACTIVE path"
    would never let the Brain sleep."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.lifecycle is not PathSequenceLifecycle.ACTIVE or not path.steps:
            continue
        source = path.steps[0].source_entity_id
        source = manipulation_sources.get(source, source)
        last = path.steps[-1]
        stepped = last.observed_at >= known_at if since is None else last.observed_at > since
        rows.append(
            {
                "context_kind": path.context_kind,
                "object_id": registry.alias_of(source),
                "direction": _direction_value(path.direction),
                "last_step": last.kind,
                "last_step_at": isoformat_utc(last.observed_at),
                "steps": len(path.steps),
                "stepped_since_last_call": bool(stepped),
            }
        )
    return tuple(rows)


def _interaction_payload(
    observation: MarketObservation, registry: ObjectRegistry, *, known_at: pd.Timestamp, since: pd.Timestamp | None
) -> tuple[dict[str, Any], ...]:
    view = brain_interaction_view(observation)
    sources = {item.manipulation_id: item.source_inventory_item_id for item in observation.manipulations}
    return interaction_rows(
        tuple(view.path_sequences), manipulation_sources=sources, registry=registry, known_at=known_at, since=since
    )


def build_eye_context(
    observation: MarketObservation,
    registry: ObjectRegistry,
    *,
    rule: EvidenceRule,
    interaction_since: pd.Timestamp | None = None,
) -> EyeContext:
    """``interaction_since`` is the previous LLM call's ``known_at`` (``None``
    on a wake); an interaction is open when a path stepped after it."""
    snapshot = observation.market_snapshot
    if snapshot is None:
        raise ValueError("observation has no market snapshot")
    known_at = pd.Timestamp(snapshot.asof).tz_convert("UTC")
    close = float(snapshot.price)
    m1 = snapshot.timeframe_states.get(Timeframe.M1)
    atr_raw = None if m1 is None else m1.quality.atr
    atr_1m = None if atr_raw is None or float(atr_raw) <= 0.0 else float(atr_raw)

    objects: list[ObjectView] = []
    scales: dict[str, Any] = {}
    for timeframe in CONTEXT_TIMEFRAMES:
        if timeframe not in snapshot.timeframe_states:
            continue
        if timeframe in ALIASED_TIMEFRAMES:
            views, summary = _scale_objects(observation, timeframe, registry)
            objects.extend(views)
            scales[timeframe.value] = summary
        else:
            state = snapshot.timeframe_states[timeframe]
            scales[timeframe.value] = {
                "structure": {
                    "external_direction": _direction_value(state.structure.external_direction),
                    "internal_direction": _direction_value(state.structure.internal_direction),
                    "last_bos_direction": _direction_value(state.structure.last_bos_direction),
                    "last_mss_direction": _direction_value(state.structure.last_mss_direction),
                },
                "delivery": {
                    "phase": state.delivery.phase.value,
                    "active_leg_direction": _direction_value(state.delivery.active_leg_direction),
                    "displacement_score": _round(state.delivery.displacement_score),
                },
            }

    events: list[EvidenceItem] = []
    for event in observation.events_this_update:
        if not rule.is_evidence(event):
            continue
        events.append(
            EvidenceItem(
                evidence_id=evidence_id(event),
                known_at=event.known_at,
                kind=event.kind.value,
                timeframe=event.timeframe.value,
                object_id=registry.alias_of(event.entity_id),
                direction=_direction_value(event.direction),
                side=event.side,
            )
        )

    relations = tuple(
        {
            "object_id": view.alias,
            "position": position,
            "offset_atr": offset,
        }
        for view in objects
        for position, offset in (object_position(close, view.lower, view.upper, atr_1m),)
    )
    interaction = _interaction_payload(observation, registry, known_at=known_at, since=interaction_since)
    context = EyeContext(
        known_at=known_at,
        close=close,
        atr_1m=atr_1m,
        bar=FrozenDict({"close": _round(close), "atr_1m": _round(atr_1m)}),
        session=FrozenDict(_session_payload(observation)),
        scales=FrozenDict(scales),
        interaction=tuple(FrozenDict(row) for row in interaction),
        objects=tuple(objects),
        events=tuple(events),
        price_relations=tuple(FrozenDict(row) for row in relations),
        open_interaction=any(row["stepped_since_last_call"] for row in interaction),
    )
    assert_causal(
        {
            "session": context.session,
            "scales": context.scales,
            "events": [item.to_dict() for item in context.events],
            "interaction": list(context.interaction),
        },
        known_at,
    )
    return context


__all__ = [
    "ALIASED_TIMEFRAMES",
    "CONTEXT_TIMEFRAMES",
    "CausalityError",
    "EvidenceRule",
    "EyeContext",
    "ObjectView",
    "assert_causal",
    "build_eye_context",
    "evidence_id",
    "interaction_rows",
    "object_position",
    "price_relation",
    "visible_liquidity_ids",
]
