"""An origin zone is two facts, not one.

The geometry of the last opposite candle before an impulse is true whether or
not that impulse went on to displace and break structure.  v1.2 published only
the qualified object, so the Brain could never see the geometry without also
inheriting the order-block reading attached to it.  v1.3 separates
``base_origin_core`` (geometry and the impulse that located it) from
``qualified_origin_zone`` (that core plus the active displacement and the
qualified BOS), and retires the single ``ORIGIN_ZONE_CREATED`` fact.
"""
from __future__ import annotations

import pytest

from eyes.core.event_store import EventStore
from shares.core.model import EventKind
from eyes.core.semantics import SemanticRegistry

from eyes.tests.test_event_provenance_contract import (
    _authoritative_phase23_chain,
    _event,
)


def _chain_without_origin_zone() -> tuple:
    return tuple(
        event
        for event in _authoritative_phase23_chain()
        if event.kind is not EventKind.ORIGIN_ZONE_CREATED
    )


def _by_id(events, event_id: str):
    return next(event for event in events if event.event_id == event_id)


def test_a_base_origin_core_cites_only_the_candles_it_froze() -> None:
    chain = _chain_without_origin_zone()
    core = _event(
        "base-origin-core",
        13,
        canonical=True,
        kind=EventKind.BASE_ORIGIN_CORE_CREATED,
        source_event_ids=("bar-left",),
    )

    store = EventStore.from_events((*chain, core))

    assert store.get(core.event_id) is not None


def test_a_qualified_zone_binds_its_core_displacement_and_break() -> None:
    chain = _chain_without_origin_zone()
    core = _event(
        "base-origin-core",
        13,
        canonical=True,
        kind=EventKind.BASE_ORIGIN_CORE_CREATED,
        source_event_ids=("bar-left",),
    )
    qualified = _event(
        "qualified-origin-zone",
        13,
        canonical=True,
        kind=EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
        source_event_ids=(
            core.event_id,
            "displacement",
            "raw-break",
        ),
    )

    store = EventStore.from_events((*chain, core, qualified))

    assert store.get(qualified.event_id) is not None


def test_a_qualified_zone_without_its_core_is_refused() -> None:
    chain = _chain_without_origin_zone()
    orphan = _event(
        "qualified-origin-zone",
        13,
        canonical=True,
        kind=EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
        source_event_ids=("displacement", "raw-break", "bar-left"),
    )

    with pytest.raises(ValueError, match="authoritative parent contract"):
        EventStore.from_events((*chain, orphan))


def test_the_single_origin_zone_creation_fact_is_retired() -> None:
    registry = SemanticRegistry.from_file()

    assert (
        EventKind.ORIGIN_ZONE_CREATED
        not in registry.canonical_emitted_event_kinds
    )
    assert {
        EventKind.BASE_ORIGIN_CORE_CREATED,
        EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
    } <= registry.canonical_emitted_event_kinds
