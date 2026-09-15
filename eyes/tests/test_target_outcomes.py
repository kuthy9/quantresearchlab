"""A candidate target ends as reached or invalidated, on its own timeframe.

The Eye enumerates candidate targets -- the liquidity inventory, each item with
its ``formed_at`` / ``confirmed_at`` and lifecycle -- and it published what
happened to them only as 1m crossing events (``LEVEL_TOUCHED`` on
``timeframe=1m`` with the level's own scale buried in
``details.source_timeframe``) and a legacy ``LIQUIDITY_RETIRED`` transport.  A
question such as "was the 5m target reached before the 15m one" therefore had
to be re-derived from the tape.  Every inventory item now ends in exactly one
of two atomic outcomes published on the level's own timeframe:
``LEVEL_REACHED`` when price first touches it, ``LEVEL_INVALIDATED`` when it
leaves the candidate set untouched.
"""
from __future__ import annotations

from collections import Counter

import pytest

from contract.eye import EventKind, LiquidityInventoryLifecycle
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_one_crossing_one_touch import _noisy_session
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


def _replay(bars):
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            displacement_protocol="configs/primitives_displacement.json",
            zone_protocol="configs/primitives_zones.json",
            range_auction_protocol="configs/primitives_range.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    observation = None
    consumed_seen: set[str] = set()
    for bar in bars:
        observation = observer.observe(reader.on_bar(bar))
        consumed_seen.update(
            item.item_id
            for item in observation.liquidity_inventory
            if item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
        )
    return observer.audit_store.events(), observation, consumed_seen


@pytest.fixture(scope="module")
def noisy():
    return _replay(_noisy_session())


@pytest.fixture(scope="module")
def three_sessions():
    return _replay(session_bars(3))


def _level(event):
    return event.details["level_id"]


def test_every_consumed_item_is_reached_exactly_once_on_its_own_timeframe(noisy) -> None:
    events, observation, consumed = noisy
    reached = [event for event in events if event.kind is EventKind.LEVEL_REACHED]
    assert consumed, "the replay consumed no inventory item"
    reached_counts = Counter(_level(event) for event in reached)
    assert set(reached_counts.values()) == {1}
    # Every item ever exposed as consumed was reached, and every reached
    # level was exposed as consumed on its bar, before the retention rules
    # (``consumed_item_retention_native_bars``,
    # ``terminal_state_retention_native_bars``) dropped it.
    assert consumed == set(reached_counts)
    offered_now = {
        item.item_id
        for item in observation.liquidity_inventory
        if item.lifecycle is not LiquidityInventoryLifecycle.CONSUMED
    }
    assert not set(reached_counts) & offered_now
    for event in reached:
        assert event.timeframe.value == event.details["source_timeframe"]
        assert _level(event) in event.source_entity_ids


def test_reached_is_known_when_the_first_touch_is_known(noisy) -> None:
    events, _, _ = noisy
    by_id = {event.event_id: event for event in events}
    for event in events:
        if event.kind is EventKind.LEVEL_REACHED:
            touches = [
                by_id[parent]
                for parent in event.source_event_ids
                if by_id[parent].kind is EventKind.LEVEL_TOUCHED
            ]
            assert len(touches) == 1
            assert touches[0].known_at == event.known_at
            assert _level(touches[0]) == _level(event)


def test_a_reference_level_replaced_untouched_is_invalidated_on_its_own_timeframe(
    three_sessions,
) -> None:
    events, _, _ = three_sessions
    retired = [
        event
        for event in events
        if event.kind is EventKind.LIQUIDITY_RETIRED
        and event.details["reason"] == "reference_period_replaced"
    ]
    assert retired, "three sessions replaced no reference level"
    reached = {_level(event) for event in events if event.kind is EventKind.LEVEL_REACHED}
    invalidated = {
        _level(event): event
        for event in events
        if event.kind is EventKind.LEVEL_INVALIDATED
        and event.details["reason"] == "reference_period_replaced"
    }
    assert all(event.details["replacement_period"] for event in retired)
    untouched = {_level(event) for event in retired} - reached
    assert untouched, "every replaced reference level had already been reached"
    assert set(invalidated) == untouched
    for level_id, event in invalidated.items():
        assert event.timeframe.value == event.details["source_timeframe"]
        assert level_id in event.source_entity_ids
