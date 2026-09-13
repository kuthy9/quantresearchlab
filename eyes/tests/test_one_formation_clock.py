"""An atomic event names the clock the fact formed on the same field a transport does.

Every ``MarketEvent`` carries ``formed_at`` / ``known_at``.  Legacy lifecycle
transports filled ``formed_at``; the canonical atomic emitter carried the same
clock only as ``event_time`` and left ``formed_at`` empty, so a consumer had to
know an event's origin to find when its fact formed.  The formation clock is
now the same field whatever produced the event, and a confirmation-lagged fact
-- a swing confirmed two bars after its pivot, a level created when that swing
confirmed -- is visibly earlier than the bar it became known on.
"""
from __future__ import annotations

import pytest

from contract.eye import EventKind, EventOrigin
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_one_crossing_one_touch import _noisy_session
from shares.tests.helpers import MODEL_SCALE_SPECS


@pytest.fixture(scope="module")
def events():
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
    for bar in _noisy_session():
        observer.observe(reader.on_bar(bar))
    return observer.audit_store.events()


def _atomic(events):
    found = [event for event in events if event.origin is EventOrigin.SEMANTIC_ATOMIC]
    assert found
    return found


def test_every_atomic_event_names_its_formation_clock(events) -> None:
    for event in _atomic(events):
        assert event.formed_at == event.event_time, event.kind.value
        assert event.formed_at <= event.known_at


def test_a_confirmed_swing_formed_before_it_was_known(events) -> None:
    lagged = [
        event
        for event in _atomic(events)
        if event.kind is EventKind.SWING_CONFIRMED and event.formed_at < event.known_at
    ]
    assert lagged, "no swing confirmation carried a formation clock earlier than its bar"


@pytest.fixture(scope="module")
def observations():
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
    return [observer.observe(reader.on_bar(bar)) for bar in _noisy_session()]


def test_a_candidate_is_published_before_it_is_confirmed(observations) -> None:
    """The candidate state is its own channel, not a lagged confirmation.

    A swing whose pivot has completed but whose confirmation bars have not yet
    closed is published in ``frames[tf].swings`` as ``forming`` with no
    ``confirmed_at``; the same identity is later published as ``confirmed``.
    """
    from contract.eye import SwingLifecycle

    first_seen_forming: dict[str, object] = {}
    confirmed_after_forming = 0
    for observation in observations:
        for frame in observation.frames.values():
            for swing in frame.swings:
                if swing.lifecycle is SwingLifecycle.FORMING:
                    assert swing.confirmed_at is None
                    first_seen_forming.setdefault(swing.swing_id, observation.asof)
                elif (
                    swing.lifecycle is SwingLifecycle.CONFIRMED
                    and swing.swing_id in first_seen_forming
                ):
                    assert swing.confirmed_at is not None
                    assert swing.confirmed_at > first_seen_forming[swing.swing_id]
                    confirmed_after_forming += 1
    assert first_seen_forming, "no forming swing was ever published"
    assert confirmed_after_forming, "no published candidate was later confirmed"
