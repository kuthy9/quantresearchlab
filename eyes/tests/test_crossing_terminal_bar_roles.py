"""A terminal names the bars that made it, not just the events above it.

``SWEEP_CONFIRMED`` and ``ACCEPTANCE_CONFIRMED`` cited their parent events, so
the bars that actually decided them -- which bar pierced the level, which bar
closed back inside, which bar held outside -- were only reachable by walking
the ancestry and re-deriving the roles.  Any study that groups by "how the
level was taken" had to reconstruct the answer the Eye already knew.  Every
crossing terminal now carries those roles itself.
"""
from __future__ import annotations

import pytest

from eyes.core.causal import CausalMarketReader
from contract.eye import EventKind
from eyes.core.observation import CausalObserver, ObserverConfig

from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


ROLE_KEYS = (
    "constituent_bar_ids",
    "penetration_bar_id",
    "reentry_bar_id",
    "hold_bar_id",
    "outside_close_ids",
)
TERMINALS = (EventKind.SWEEP_CONFIRMED, EventKind.ACCEPTANCE_CONFIRMED)


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
    for bar in session_bars(2):
        observer.observe(reader.on_bar(bar))
    return observer.audit_store.events()


def _terminals(events):
    found = [event for event in events if event.kind in TERMINALS]
    assert found, "the replay produced no crossing terminal"
    return found


def test_every_crossing_terminal_carries_all_five_roles(events) -> None:
    for terminal in _terminals(events):
        for key in ROLE_KEYS:
            assert key in terminal.evidence, (
                f"{terminal.kind.value} does not name {key}"
            )


def test_the_named_bars_are_real_completed_bars_the_terminal_could_see(
    events,
) -> None:
    by_id = {event.event_id: event for event in events}

    for terminal in _terminals(events):
        constituents = terminal.evidence["constituent_bar_ids"]
        assert constituents, "a terminal was decided by no bar at all"
        for bar_id in constituents:
            bar = by_id[bar_id]
            assert bar.kind is EventKind.BAR_COMPLETED
            assert bar.evidence["real_completed"] is True
            assert bar.known_at <= terminal.known_at


def test_the_resolution_role_matches_the_verdict(events) -> None:
    """A sweep came back inside; an acceptance held outside.  Never both."""

    for terminal in _terminals(events):
        evidence = terminal.evidence
        held = terminal.kind is EventKind.ACCEPTANCE_CONFIRMED
        assert (evidence["hold_bar_id"] is not None) is held
        assert (evidence["reentry_bar_id"] is not None) is not held
        assert bool(evidence["outside_close_ids"]) is held


def test_every_named_role_is_one_of_the_constituents(events) -> None:
    for terminal in _terminals(events):
        evidence = terminal.evidence
        constituents = set(evidence["constituent_bar_ids"])
        named = [
            evidence["penetration_bar_id"],
            evidence["reentry_bar_id"],
            evidence["hold_bar_id"],
            *evidence["outside_close_ids"],
        ]
        for bar_id in named:
            assert bar_id is None or bar_id in constituents


def test_a_penetration_bar_never_follows_the_bar_that_resolved_it(
    events,
) -> None:
    """A crossing can be resolved by the bar that made it, but never earlier."""

    by_id = {event.event_id: event for event in events}

    for terminal in _terminals(events):
        evidence = terminal.evidence
        penetration_id = evidence["penetration_bar_id"]
        if penetration_id is None:
            continue
        resolution_id = evidence["hold_bar_id"] or evidence["reentry_bar_id"]
        assert resolution_id is not None
        assert by_id[penetration_id].known_at <= by_id[resolution_id].known_at
