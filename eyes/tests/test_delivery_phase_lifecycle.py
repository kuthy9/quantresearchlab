"""A delivery phase is something price is *in*, not a label recomputed per bar.

v1.2 published the phase only as a snapshot field, so "how long has this been
expansion" and "what did it come from" could not be asked without re-deriving
them from the whole event log.  v1.3 gives each occupancy a generation with an
entry, an age, an origin and a successor, and keeps it strictly independent of
the structure regime: the same regime spans many phases and the same phase
occurs under opposite regimes.
"""
from __future__ import annotations

import collections

import pytest

from eyes.core.causal import CausalMarketReader
from shares.core.model import EventKind
from eyes.core.observation import CausalObserver, ObserverConfig

from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"
DISPLACEMENT_PROTOCOL = "configs/primitives_displacement.json"
GROUP3_PROTOCOL = "configs/primitives_zones.json"
GROUP4_PROTOCOL = "configs/primitives_range.json"

LIFECYCLE_KINDS = (
    EventKind.DELIVERY_PHASE_ENTERED,
    EventKind.DELIVERY_PHASE_UPDATED,
    EventKind.DELIVERY_PHASE_EXITED,
)


@pytest.fixture(scope="module")
def replay():
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
            displacement_protocol=DISPLACEMENT_PROTOCOL,
            zone_protocol=GROUP3_PROTOCOL,
            range_auction_protocol=GROUP4_PROTOCOL,
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    bars = session_bars(2)
    for bar in bars:
        observer.observe(reader.on_bar(bar))
    return observer.audit_store.events(), len(bars)


def _lifecycle(events, timeframe=None):
    return [
        event
        for event in events
        if event.kind in LIFECYCLE_KINDS
        and (timeframe is None or event.timeframe is timeframe)
    ]


def test_every_phase_occupancy_is_entered_before_it_is_updated_or_exited(
    replay,
) -> None:
    events, _ = replay
    lifecycle = _lifecycle(events)
    assert lifecycle, "no delivery-phase lifecycle was published"

    open_phase: dict[object, str] = {}
    last_exited: dict[object, str] = {}
    for event in lifecycle:
        timeframe = event.timeframe
        phase = event.details["phase"]
        if event.kind is EventKind.DELIVERY_PHASE_ENTERED:
            assert timeframe not in open_phase, "entered twice without exiting"
            assert event.details["previous_phase"] == last_exited.get(timeframe)
            open_phase[timeframe] = phase
        elif event.kind is EventKind.DELIVERY_PHASE_UPDATED:
            assert open_phase.get(timeframe) == phase
            assert event.details["age_bars"] >= 0
        else:
            assert open_phase.pop(timeframe) == phase
            last_exited[timeframe] = phase
            assert event.details["next_phase"] is not None
            assert event.details["next_phase"] != phase


def test_an_exit_names_the_phase_the_next_entry_declares(replay) -> None:
    events, _ = replay
    lifecycle = _lifecycle(events)
    pairs = 0
    pending: dict[object, str] = {}
    for event in lifecycle:
        if event.kind is EventKind.DELIVERY_PHASE_EXITED:
            pending[event.timeframe] = event.details["next_phase"]
        elif (
            event.kind is EventKind.DELIVERY_PHASE_ENTERED
            and event.timeframe in pending
        ):
            assert pending.pop(event.timeframe) == event.details["phase"]
            assert event.details["previous_phase"] is not None
            pairs += 1
    assert pairs >= 1


def test_entry_freezes_its_origin_age_and_parent_structure_generation(
    replay,
) -> None:
    events, _ = replay
    entries = [
        event
        for event in _lifecycle(events)
        if event.kind is EventKind.DELIVERY_PHASE_ENTERED
    ]
    by_id = {event.event_id: event for event in events}
    for entry in entries:
        assert entry.details["entered_at"] == entry.known_at.isoformat()
        assert entry.details["age_bars"] == 0
        origin = entry.details["origin_event"]
        assert origin is None or origin in by_id
        # v1.3 named the most recent break here, so a phase spanning several
        # breaks reported several parents.  It now names one structural claim.
        parent = entry.details["parent_structure_generation"]
        assert parent is None or parent not in by_id
        assert parent is None or "_generation_" in parent


def test_regime_and_phase_stay_two_independent_dimensions(replay) -> None:
    """Neither dimension may be recoverable from the other.

    These two registered sessions only ever establish one regime direction, so
    they can show one half of the independence: a single regime spanning
    several phases.  The other half -- one phase occurring under opposite
    regimes -- needs a real replay long enough to change direction.
    """

    events, _ = replay
    entries = [
        event
        for event in _lifecycle(events)
        if event.kind is EventKind.DELIVERY_PHASE_ENTERED
    ]
    phases_by_regime = collections.defaultdict(set)
    for entry in entries:
        assert "structure_regime" in entry.details
        assert "phase" in entry.details
        phases_by_regime[entry.details["structure_regime"]].add(
            entry.details["phase"]
        )
    assert any(
        len(phases) > 1 for phases in phases_by_regime.values()
    ), dict(phases_by_regime)


def test_updates_are_input_driven_rather_than_one_per_bar(replay) -> None:
    events, bars = replay
    updates = [
        event
        for event in _lifecycle(events)
        if event.kind is EventKind.DELIVERY_PHASE_UPDATED
    ]
    # A per-bar update would produce roughly bars x timeframes events.
    assert len(updates) < bars
