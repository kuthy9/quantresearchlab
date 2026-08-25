"""Private bounded current-fact projection over the authoritative EventStore.

This module owns no mutable state, checkpoint, schema version, or public
facade.  ``TimeframeEventReducer`` remains the sole owner of the current-fact
ID index and delegates deterministic index transitions and materialization
here so ``market_state`` does not also become an event-protocol module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .event_store import EventStore, _require_exact_market_event
from .model import (
    BOSScope,
    Direction,
    EventKind,
    EventOrigin,
    MarketEvent,
    Timeframe,
)


CurrentFactKey = tuple[str, str, str, str]
CurrentFactIndex = MutableMapping[CurrentFactKey, str]


_CURRENT_FACT_FAMILIES: Mapping[EventKind, tuple[str, str]] = {
    EventKind.SWING_CONFIRMED: ("structure", "swing"),
    EventKind.STRUCTURAL_LEG_CREATED: ("structure", "leg"),
    EventKind.RAW_BOUNDARY_BREAK: ("structure", "raw_break"),
    EventKind.STRUCTURE_DIRECTION_CONFIRMED: ("structure", "direction"),
    EventKind.QUALIFIED_BOS: ("structure", "bos"),
    EventKind.PROTECTED_SWING_ASSIGNED: ("structure", "protected_swing"),
    EventKind.MSS_CORE_CONFIRMED: ("structure", "mss"),
    EventKind.LIQUIDITY_LEVEL_CREATED: ("liquidity", "level"),
    EventKind.LEVEL_TOUCHED: ("liquidity", "level"),
    EventKind.LEVEL_PENETRATED: ("liquidity", "level"),
    EventKind.SWEEP_CONFIRMED: ("liquidity", "level"),
    EventKind.ACCEPTANCE_CONFIRMED: ("liquidity", "level"),
    EventKind.DISPLACEMENT_OBSERVED: ("displacement", "episode"),
    EventKind.FVG_CREATED: ("zones", "fvg"),
    EventKind.FVG_PARTIALLY_FILLED: ("zones", "fvg"),
    EventKind.FVG_MIDPOINT_TOUCHED: ("zones", "fvg"),
    EventKind.FVG_FULLY_FILLED: ("zones", "fvg"),
    EventKind.FVG_INVALIDATED: ("zones", "fvg"),
    EventKind.ORIGIN_ZONE_CREATED: ("zones", "origin_zone"),
    EventKind.ORIGIN_ZONE_MITIGATED: ("zones", "origin_zone"),
    EventKind.ORIGIN_ZONE_INVALIDATED: ("zones", "origin_zone"),
    EventKind.DEALING_RANGE_CREATED: ("ranges", "dealing_range"),
    EventKind.DEALING_RANGE_ACTIVATED: ("ranges", "dealing_range"),
    EventKind.DEALING_RANGE_INVALIDATED: ("ranges", "dealing_range"),
    EventKind.DEALING_RANGE_REPLACED: ("ranges", "dealing_range"),
}

_CURRENT_FACT_TERMINAL_KINDS = frozenset(
    {
        EventKind.FVG_FULLY_FILLED,
        EventKind.FVG_INVALIDATED,
        EventKind.ORIGIN_ZONE_MITIGATED,
        EventKind.ORIGIN_ZONE_INVALIDATED,
        EventKind.DEALING_RANGE_INVALIDATED,
        EventKind.SWEEP_CONFIRMED,
        EventKind.ACCEPTANCE_CONFIRMED,
    }
)


def _current_fact_category(event: MarketEvent) -> str | None:
    value = _CURRENT_FACT_FAMILIES.get(event.kind)
    return None if value is None else value[0]


def _structure_id(event: MarketEvent) -> str | None:
    if event.kind in {
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        EventKind.PROTECTED_SWING_ASSIGNED,
    }:
        value = event.evidence.get("structure_id")
    elif event.kind in {EventKind.QUALIFIED_BOS, EventKind.MSS_CORE_CONFIRMED}:
        value = event.source_entity_ids[1] if len(event.source_entity_ids) > 1 else None
    elif event.kind is EventKind.RAW_BOUNDARY_BREAK:
        value = event.source_entity_ids[2] if len(event.source_entity_ids) > 2 else None
    else:
        return None
    return value if isinstance(value, str) and value else None


def _references_structure(event: MarketEvent, owner: MarketEvent) -> bool:
    owner_id = _structure_id(owner)
    if not owner_id:
        return False
    if event.kind is EventKind.RAW_BOUNDARY_BREAK:
        return owner_id in event.source_entity_ids[2:]
    return _structure_id(event) == owner_id


def _range_id(event: MarketEvent) -> str | None:
    field = (
        "replacement_range_id"
        if event.kind is EventKind.DEALING_RANGE_REPLACED
        else "range_id"
    )
    value = event.evidence.get(field)
    return value if isinstance(value, str) and value else None


def _is_bos_resolution_fact(event: MarketEvent) -> bool:
    bos_id = event.evidence.get("bos_id")
    return bool(
        event.kind in {EventKind.ACCEPTANCE_CONFIRMED, EventKind.SWEEP_CONFIRMED}
        and isinstance(bos_id, str)
        and bos_id
    )


def _bos_slot_from_raw_break(event: MarketEvent) -> str:
    if (
        event.kind is not EventKind.RAW_BOUNDARY_BREAK
        or not isinstance(event.evidence.get("bos_id"), str)
        or not event.evidence.get("bos_id")
        or event.direction is None
    ):
        raise ValueError("BOS resolution lacks its exact raw-break identity")
    try:
        scope = BOSScope(str(event.evidence["scope"]))
    except (KeyError, ValueError) as error:
        raise ValueError("BOS resolution raw break has an invalid scope") from error
    return f"{scope.value}:{event.direction.value}"


def _bos_slot_from_qualification(event: MarketEvent) -> str:
    if (
        event.kind not in {EventKind.QUALIFIED_BOS, EventKind.MSS_CORE_CONFIRMED}
        or not isinstance(event.evidence.get("bos_id"), str)
        or not event.evidence.get("bos_id")
        or event.direction is None
    ):
        raise ValueError("BOS resolution lacks its exact qualification")
    try:
        scope = BOSScope(str(event.evidence["scope"]))
    except (KeyError, ValueError) as error:
        raise ValueError("BOS qualification has an invalid scope") from error
    if (
        scope is BOSScope.CONTINUATION
        and event.kind is not EventKind.QUALIFIED_BOS
    ) or (
        scope is BOSScope.OPPOSED
        and event.kind is not EventKind.MSS_CORE_CONFIRMED
    ) or scope is BOSScope.LOCAL:
        raise ValueError("BOS qualification kind and scope differ")
    return f"{scope.value}:{event.direction.value}"


def _bos_resolution_pair(
    store: EventStore,
    event: MarketEvent,
) -> tuple[CurrentFactKey, CurrentFactKey, MarketEvent]:
    if not _is_bos_resolution_fact(event):
        raise ValueError("event is not a BOS resolution fact")
    bos_id = str(event.evidence["bos_id"])
    raw_breaks = tuple(
        parent
        for event_id in event.context_event_ids
        if (parent := store.get(event_id)) is not None
        and parent.kind is EventKind.RAW_BOUNDARY_BREAK
        and parent.evidence.get("bos_id") == bos_id
    )
    if len(raw_breaks) != 1:
        raise ValueError("BOS resolution lacks one exact raw-break context")
    raw_break = raw_breaks[0]
    if raw_break.timeframe is not event.timeframe or raw_break.known_at > event.known_at:
        raise ValueError("BOS resolution raw-break clock or owner changed")
    slot = _bos_slot_from_raw_break(raw_break)
    return (
        ("structure", "bos_resolution", event.timeframe.value, slot),
        ("structure", "bos_resolution_context", event.timeframe.value, slot),
        raw_break,
    )


def _current_fact_key(event: MarketEvent) -> CurrentFactKey | None:
    category_family = _CURRENT_FACT_FAMILIES.get(event.kind)
    if category_family is None:
        return None
    category, family = category_family
    evidence_fields = {
        "swing": ("source_entity_id", "swing_id"),
        "leg": ("leg_id",),
        "raw_break": ("break_id", "structure_id"),
        "direction": ("structure_id",),
        "bos": ("bos_id",),
        "protected_swing": ("protected_swing_id",),
        "mss": ("mss_id", "bos_id"),
        "level": ("level_id",),
        "episode": ("displacement_id",),
        "fvg": ("fvg_id",),
        "origin_zone": ("origin_zone_id", "zone_id"),
        "dealing_range": ("range_id", "replacement_range_id"),
    }[family]
    identity = event.entity_id or next(
        (
            str(event.evidence[name])
            for name in evidence_fields
            if isinstance(event.evidence.get(name), str) and event.evidence[name]
        ),
        None,
    )
    if not identity and event.source_entity_ids:
        identity = event.source_entity_ids[0]
    if not identity:
        return None
    key_timeframe = event.timeframe
    if category == "liquidity" and event.evidence.get("source_timeframe"):
        key_timeframe = Timeframe(str(event.evidence["source_timeframe"]))
    if category == "structure":
        if family == "swing":
            entity_slot = event.side or "current"
        elif family == "raw_break":
            entity_slot = _bos_slot_from_raw_break(event)
        else:
            entity_slot = "current"
    elif category == "displacement":
        entity_slot = "current" if event.direction is None else event.direction.value
    elif category == "ranges":
        entity_slot = "current"
    else:
        entity_slot = identity
    return (category, family, key_timeframe.value, entity_slot)


def _current_fact_is_terminal(event: MarketEvent) -> bool:
    return bool(
        event.kind in _CURRENT_FACT_TERMINAL_KINDS
        or event.ended_at is not None
        or (
            event.kind is EventKind.DISPLACEMENT_OBSERVED
            and event.evidence.get("lifecycle") in {"exhausted", "censored"}
        )
    )


@dataclass(frozen=True, slots=True)
class CurrentMarketFacts:
    """Bounded immutable current facts, never an event timeline."""

    structure: tuple[MarketEvent, ...] = ()
    structure_context: tuple[MarketEvent, ...] = ()
    liquidity: tuple[MarketEvent, ...] = ()
    displacement: tuple[MarketEvent, ...] = ()
    zones: tuple[MarketEvent, ...] = ()
    ranges: tuple[MarketEvent, ...] = ()

    def __post_init__(self) -> None:
        groups = {
            name: tuple(getattr(self, name))
            for name in (
                "structure",
                "structure_context",
                "liquidity",
                "displacement",
                "zones",
                "ranges",
            )
        }
        for name, values in groups.items():
            object.__setattr__(self, name, values)
            for event in values:
                _require_exact_market_event(event)
            if any(event.origin is not EventOrigin.SEMANTIC_ATOMIC for event in values):
                raise ValueError("current market facts require canonical atomic origin")
            keys = tuple((event.known_at, event.sequence_no, event.event_id) for event in values)
            if keys != tuple(sorted(keys)) or len(values) != len(
                {event.event_id for event in values}
            ):
                raise ValueError("current market facts are not canonical and unique")
            for event in values:
                if name == "structure_context":
                    valid = event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED
                else:
                    valid = _current_fact_category(event) == name or (
                        name == "structure" and _is_bos_resolution_fact(event)
                    )
                if not valid:
                    raise ValueError("current market fact is in the wrong category")


def _current_event(
    store: EventStore,
    index: Mapping[CurrentFactKey, str],
    key: CurrentFactKey,
) -> MarketEvent | None:
    event_id = index.get(key)
    return None if event_id is None else store.get(event_id)


def _exact_source_parent(
    store: EventStore,
    event: MarketEvent,
    kind: EventKind,
) -> MarketEvent:
    parents = tuple(
        parent
        for event_id in event.source_event_ids
        if (parent := store.get(event_id)) is not None and parent.kind is kind
    )
    if len(parents) != 1:
        raise ValueError(f"current {event.kind.value} lacks one exact {kind.value} parent")
    return parents[0]


def _clear_structure_facts(
    index: CurrentFactIndex,
    timeframe: Timeframe,
    *,
    include_external: bool,
    include_protected: bool,
    include_context: bool,
) -> None:
    families = {
        "raw_break",
        "bos",
        "mss",
        "bos_resolution",
        "bos_resolution_context",
        "bos_resolution_qualified",
    }
    if include_external:
        families.add("direction")
    if include_protected:
        families.add("protected_swing")
    for key in tuple(index):
        if key[2] != timeframe.value:
            continue
        if key[0] == "structure" and key[1] in families:
            index.pop(key)
        elif include_context and key[0] == "structure_context":
            index.pop(key)


def _clear_bos_resolution_scope(
    index: CurrentFactIndex,
    timeframe: Timeframe,
    scope: BOSScope,
) -> None:
    for key in tuple(index):
        if (
            key[0] == "structure"
            and key[1]
            in {"bos_resolution", "bos_resolution_context", "bos_resolution_qualified"}
            and key[2] == timeframe.value
            and key[3].startswith(f"{scope.value}:")
        ):
            index.pop(key)


def materialize_current_facts(
    *,
    event_store: EventStore,
    index: Mapping[CurrentFactKey, str],
) -> CurrentMarketFacts:
    grouped: dict[str, dict[str, MarketEvent]] = {
        name: {}
        for name in (
            "structure",
            "structure_context",
            "liquidity",
            "displacement",
            "zones",
            "ranges",
        )
    }
    for key, event_id in index.items():
        event = event_store.get(event_id)
        expected_key = _current_fact_key(event) if event is not None else None
        if event is not None and key[1] == "bos_resolution":
            expected_key = _bos_resolution_pair(event_store, event)[0]
        elif event is not None and key[1] == "bos_resolution_context":
            expected_key = (
                "structure",
                "bos_resolution_context",
                event.timeframe.value,
                _bos_slot_from_raw_break(event),
            )
        elif event is not None and key[1] == "bos_resolution_qualified":
            expected_key = (
                "structure",
                "bos_resolution_qualified",
                event.timeframe.value,
                _bos_slot_from_qualification(event),
            )
        elif event is not None and key[0] == "structure_context":
            expected_key = (
                "structure_context",
                "challenger",
                event.timeframe.value,
                "current" if event.direction is None else event.direction.value,
            )
        if (
            event is None
            or expected_key != key
            or event.origin is not EventOrigin.SEMANTIC_ATOMIC
            or (_current_fact_is_terminal(event) and key[1] != "bos_resolution")
            or event_store.recompute_event_digest(event)
            != event_store.event_digest(event_id)
        ):
            raise ValueError("current Eye fact index differs from EventStore")
        grouped[key[0]][event.event_id] = event

    for key, event_id in index.items():
        if key[1] != "bos_resolution":
            continue
        context_key = (key[0], "bos_resolution_context", key[2], key[3])
        context_id = index.get(context_key)
        event = event_store.get(event_id)
        if context_id is None or event is None or context_id not in event.context_event_ids:
            raise ValueError("current BOS resolution lost its raw context")
        raw = event_store.get(context_id)
        if raw is None:
            raise ValueError("current BOS resolution raw context disappeared")
        scope = BOSScope(str(raw.evidence["scope"]))
        qualified_key = (key[0], "bos_resolution_qualified", key[2], key[3])
        qualified_id = index.get(qualified_key)
        if scope in {BOSScope.CONTINUATION, BOSScope.OPPOSED}:
            qualified = None if qualified_id is None else event_store.get(qualified_id)
            if (
                qualified is None
                or qualified.evidence.get("bos_id") != event.evidence.get("bos_id")
                or _bos_slot_from_qualification(qualified) != key[3]
            ):
                raise ValueError("current BOS resolution lost its exact qualification")
        elif qualified_id is not None:
            raise ValueError("local BOS resolution cannot own qualification")

    for timeframe in Timeframe:
        tf = timeframe.value
        external = _current_event(event_store, index, ("structure", "direction", tf, "current"))
        context_ids = {
            event_id
            for key, event_id in index.items()
            if key[0] == "structure_context" and key[2] == tf
        }
        if external is not None and external.event_id in context_ids:
            raise ValueError("external structure is duplicated as context")
        for family in ("bos", "mss"):
            fact = _current_event(event_store, index, ("structure", family, tf, "current"))
            if fact is None:
                continue
            parent = _exact_source_parent(event_store, fact, EventKind.STRUCTURE_DIRECTION_CONFIRMED)
            valid = (
                external is not None and parent.event_id == external.event_id
                if family == "bos"
                else (
                    (external is not None and parent.event_id == external.event_id)
                    or parent.event_id in context_ids
                )
            )
            if not valid:
                raise ValueError(f"current {family} lost its exact structure context")
        protected = _current_event(
            event_store,
            index,
            ("structure", "protected_swing", tf, "current"),
        )
        if protected is not None and (
            external is None or _structure_id(protected) != _structure_id(external)
        ):
            raise ValueError("current protected assignment differs from external owner")

    canonical = {
        name: tuple(
            sorted(
                values.values(),
                key=lambda event: (event.known_at, event.sequence_no, event.event_id),
            )
        )
        for name, values in grouped.items()
    }
    return CurrentMarketFacts(**canonical)


def advance_current_facts(
    *,
    event_store: EventStore,
    index: CurrentFactIndex,
    states: Mapping[Timeframe, Any],
    event: MarketEvent,
    source_timeframe: Timeframe | None,
) -> None:
    """Mutate only the caller-owned bounded ID index for one atomic fact."""

    if event.origin is not EventOrigin.SEMANTIC_ATOMIC:
        return
    fact_key = _current_fact_key(event)
    if fact_key is None:
        return
    timeframe = event.timeframe
    tf = timeframe.value
    external_key = ("structure", "direction", tf, "current")
    protected_key = ("structure", "protected_swing", tf, "current")
    bos_key = ("structure", "bos", tf, "current")
    mss_key = ("structure", "mss", tf, "current")
    current = lambda key: _current_event(event_store, index, key)

    if event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED:
        state = states.get(timeframe)
        if state is None:
            raise ValueError("current structure direction lacks owner state")
        if state.structure.external_direction is not event.direction:
            if event.direction is None:
                raise ValueError("structure challenger lacks direction")
            key = ("structure_context", "challenger", tf, event.direction.value)
            prior = current(key)
            index[key] = event.event_id
            current_mss = current(mss_key)
            if prior is not None and current_mss is not None and prior.event_id in current_mss.source_event_ids:
                index.pop(mss_key, None)
                _clear_bos_resolution_scope(index, timeframe, BOSScope.OPPOSED)
            return
        prior_external = current(external_key)
        live_protected = current(protected_key)
        if (
            prior_external is not None
            and live_protected is not None
            and _structure_id(prior_external) != _structure_id(event)
        ):
            if event.direction is None:
                raise ValueError("structure reinforcement lacks direction")
            index[("structure_context", "challenger", tf, event.direction.value)] = event.event_id
            return
        if prior_external is not None and _structure_id(prior_external) != _structure_id(event):
            _clear_structure_facts(
                index,
                timeframe,
                include_external=True,
                include_protected=True,
                include_context=True,
            )
        index[external_key] = event.event_id
        if event.direction is not None:
            index.pop(("structure_context", "challenger", tf, event.direction.value), None)
        return

    if event.kind is EventKind.RAW_BOUNDARY_BREAK:
        external = current(external_key)
        bound = external is not None and _references_structure(event, external)
        if not bound and event.direction is not None:
            opposite = Direction.SHORT if event.direction is Direction.LONG else Direction.LONG
            challenger = current(("structure_context", "challenger", tf, opposite.value))
            bound = challenger is not None and _references_structure(event, challenger)
        if bound:
            index[fact_key] = event.event_id
        return

    if event.kind in {EventKind.QUALIFIED_BOS, EventKind.MSS_CORE_CONFIRMED}:
        parent = _exact_source_parent(event_store, event, EventKind.STRUCTURE_DIRECTION_CONFIRMED)
        if event.kind is EventKind.QUALIFIED_BOS:
            external = current(external_key)
            if external is None or external.event_id != parent.event_id:
                return
            index.pop(bos_key, None)
            _clear_bos_resolution_scope(index, timeframe, BOSScope.CONTINUATION)
            index[bos_key] = event.event_id
        else:
            external = current(external_key)
            challenger_key = (
                "structure_context",
                "challenger",
                tf,
                "current" if parent.direction is None else parent.direction.value,
            )
            challenger = current(challenger_key)
            if not (
                (external is not None and external.event_id == parent.event_id)
                or (challenger is not None and challenger.event_id == parent.event_id)
            ):
                return
            index.pop(mss_key, None)
            _clear_bos_resolution_scope(index, timeframe, BOSScope.OPPOSED)
            if external is None or external.event_id != parent.event_id:
                index[challenger_key] = parent.event_id
            index[mss_key] = event.event_id
        return

    if event.kind is EventKind.PROTECTED_SWING_ASSIGNED:
        state = states.get(timeframe)
        assignment_id = None if state is None else (
            state.structure.protected_low_event_id
            if event.direction is Direction.LONG
            else state.structure.protected_high_event_id
        )
        if state is None or state.structure.protected_swing_intact is not True or assignment_id != event.event_id:
            raise ValueError("protected assignment is not the live state owner")
        qualified = _exact_source_parent(event_store, event, EventKind.QUALIFIED_BOS)
        parent = _exact_source_parent(event_store, qualified, EventKind.STRUCTURE_DIRECTION_CONFIRMED)
        external = current(external_key)
        if external is None or external.event_id != parent.event_id:
            _clear_structure_facts(
                index,
                timeframe,
                include_external=True,
                include_protected=True,
                include_context=True,
            )
            index[external_key] = parent.event_id
        index[bos_key] = qualified.event_id
        raw = _exact_source_parent(event_store, qualified, EventKind.RAW_BOUNDARY_BREAK)
        raw_key = _current_fact_key(raw)
        if raw_key is None:
            raise ValueError("qualified BOS raw current-fact identity is missing")
        index[raw_key] = raw.event_id
        index[protected_key] = event.event_id
        return

    if fact_key[0] == "structure":
        index[fact_key] = event.event_id
        return
    if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED:
        replaced = event.evidence.get("replaces_level_id")
        if isinstance(replaced, str) and replaced:
            index.pop(("liquidity", "level", fact_key[2], replaced), None)
    if event.kind is EventKind.DEALING_RANGE_REPLACED:
        invalidated = _exact_source_parent(event_store, event, EventKind.DEALING_RANGE_INVALIDATED)
        created = _exact_source_parent(event_store, event, EventKind.DEALING_RANGE_CREATED)
        if (
            _range_id(event) != _range_id(created)
            or _range_id(invalidated) == _range_id(created)
            or event.timeframe is not created.timeframe
            or event.zone != created.zone
            or event.known_at != created.known_at
        ):
            raise ValueError("range replacement does not bind old and new ranges")
    if _current_fact_is_terminal(event):
        if event.kind is EventKind.DEALING_RANGE_INVALIDATED:
            visible = current(fact_key)
            if visible is not None and _range_id(visible) == _range_id(event):
                index.pop(fact_key)
        else:
            index.pop(fact_key, None)
    else:
        index[fact_key] = event.event_id

    if event.kind is not EventKind.ACCEPTANCE_CONFIRMED:
        return
    protected_event_id = event.evidence.get("protected_swing_event_id")
    owner_timeframe = source_timeframe or timeframe
    owner_key = ("structure", "protected_swing", owner_timeframe.value, "current")
    protected = current(owner_key)
    owner_state = states.get(owner_timeframe)
    if (
        isinstance(protected_event_id, str)
        and protected is not None
        and protected.event_id == protected_event_id
        and protected.event_id in event.context_event_ids
        and owner_state is not None
        and owner_state.structure.protected_swing_intact is False
        and owner_state.structure.external_direction is None
    ):
        _clear_structure_facts(
            index,
            owner_timeframe,
            include_external=True,
            include_protected=True,
            include_context=True,
        )


def advance_bos_resolution(
    *,
    event_store: EventStore,
    index: CurrentFactIndex,
    event: MarketEvent,
) -> None:
    if (
        event.origin is not EventOrigin.SEMANTIC_ATOMIC
        or not _is_bos_resolution_fact(event)
        or isinstance(event.evidence.get("protected_swing_event_id"), str)
    ):
        return
    resolution_key, context_key, raw = _bos_resolution_pair(event_store, event)
    scope = BOSScope(str(raw.evidence["scope"]))
    if scope in {BOSScope.CONTINUATION, BOSScope.OPPOSED}:
        family = "bos" if scope is BOSScope.CONTINUATION else "mss"
        qualification = _current_event(
            event_store,
            index,
            ("structure", family, raw.timeframe.value, "current"),
        )
        if (
            qualification is None
            or qualification.evidence.get("bos_id") != event.evidence.get("bos_id")
            or _bos_slot_from_qualification(qualification) != resolution_key[3]
        ):
            raise ValueError("BOS resolution lacks its current exact qualification")
        index[("structure", "bos_resolution_qualified", event.timeframe.value, resolution_key[3])] = qualification.event_id
    index[resolution_key] = event.event_id
    index[context_key] = raw.event_id
