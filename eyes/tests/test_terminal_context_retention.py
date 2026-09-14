"""A closed Group 5 path is exposed once, then leaves the current update.

The interaction update carried every closed context path until
``maximum_context_states`` (256) evicted the oldest, and re-validated its whole
graph on every bar: at bar 3,000 on the real tape 169 of its 187 paths were
closed, some for 1,640 minutes, and the update's cost grew with them.  The
closing transition is delivered in ``interaction_path_transitions`` on the bar
it happens and the fact stays in the event log; the terminal path itself now
appears in ``interaction_paths`` for exactly
``terminal_context_retention_real_1m_bars`` completed outputs (one: the closing
bar), then is compacted.  A live path is never compacted, and a zone whose
context was compacted is not reported as a cold source.
"""
from __future__ import annotations

from collections import Counter

import pytest

from contract.eye import PathSequenceLifecycle
from eyes.core.causal import CausalMarketReader
from eyes.core.interaction import InteractionProtocol
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

TERMINAL = {PathSequenceLifecycle.CLOSED, PathSequenceLifecycle.CENSORED}


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
            interaction_protocol="configs/primitives_interaction.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    return [observer.observe(reader.on_bar(bar)) for bar in _noisy(session_bars(1)[:900])]


def test_the_protocol_names_the_terminal_retention() -> None:
    protocol = InteractionProtocol.from_file("configs/primitives_interaction.json")
    assert protocol.terminal_context_retention_bars == 1


def test_a_terminal_path_is_exposed_then_compacted(observations) -> None:
    exposures: Counter[str] = Counter()
    for observation in observations:
        update = observation.interaction_update
        if update is None:
            continue
        for path in update.interaction_paths:
            if path.lifecycle in TERMINAL:
                exposures[path.sequence_id] += 1
    assert exposures, "the replay closed no Group 5 path"
    assert set(exposures.values()) == {1}, exposures.most_common(3)


def test_a_live_path_is_never_compacted_by_retention(observations) -> None:
    updates = [
        observation.interaction_update
        for observation in observations
        if observation.interaction_update is not None
    ]
    previous_live: set[str] = set()
    for update in updates:
        present = {
            path.sequence_id
            for path in (
                *update.interaction_paths,
                *update.interaction_path_transitions,
            )
        }
        # A path live in the previous output is in this one, live or freshly
        # terminal — never silently gone.
        assert previous_live <= present, sorted(previous_live - present)
        previous_live = {
            path.sequence_id
            for path in update.interaction_paths
            if path.lifecycle is PathSequenceLifecycle.ACTIVE
        }


def test_a_compacted_zone_is_not_a_cold_source(observations) -> None:
    compacted_zone_ids: set[str] = set()
    previous_terminal: dict[str, str] = {}
    for observation in observations:
        update = observation.interaction_update
        if update is None:
            continue
        present = {path.sequence_id for path in update.interaction_paths}
        for sequence_id, zone_id in previous_terminal.items():
            if sequence_id not in present:
                compacted_zone_ids.add(zone_id)
        assert not (compacted_zone_ids & set(update.cold_source_ids)), (
            compacted_zone_ids & set(update.cold_source_ids)
        )
        locations = {
            location.location_id: location.source_zone_id
            for location in update.zone_interactions
        }
        previous_terminal = {
            path.sequence_id: locations[path.context_id]
            for path in update.interaction_paths
            if path.lifecycle in TERMINAL
            and path.context_kind == "zone_return"
            and path.context_id in locations
        }
    assert compacted_zone_ids, "the replay compacted no zone context"
