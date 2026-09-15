"""The 1m tape is microstructure; the main event clock runs on the higher scales.

On 2022-02-01→05, 1m events were 83.8 % of everything the Eye published and
86 % of its transition events; the design of the information-gain gate had to
define its own clocks (≥5m, ≥15m) to see past them.  The observation now
names the timeframes that form its main event clock
(``published_timeframes``): ``semantic_events_this_update`` carries only
those, and every 1m event of the update moves to
``microstructure_events_this_update``, the channel a trigger reads.  Nothing
is dropped -- the two channels partition the snapshot's events, and the event
store keeps every scale for provenance.
"""
from __future__ import annotations

import pytest

from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_one_crossing_one_touch import _noisy_session
from shares.tests.helpers import MODEL_SCALE_SPECS


def _observe(**config):
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
            **config,
        )
    )
    return [observer.observe(reader.on_bar(bar)) for bar in _noisy_session(200)]


@pytest.fixture(scope="module")
def observations():
    return _observe()


def test_the_main_event_clock_excludes_the_1m_scale_by_default(observations) -> None:
    for observation in observations:
        assert set(observation.published_timeframes) == (
            set(observation.active_timeframes) - {Timeframe.M1}
        )
    for observation in observations:
        assert all(
            event.timeframe is not Timeframe.M1
            for event in observation.semantic_events_this_update
        )


def test_the_two_channels_partition_the_update(observations) -> None:
    saw_microstructure = False
    for observation in observations:
        main = observation.semantic_events_this_update
        micro = observation.microstructure_events_this_update
        assert all(event.timeframe is Timeframe.M1 for event in micro)
        assert {event.event_id for event in main}.isdisjoint(
            event.event_id for event in micro
        )
        assert {event.event_id for event in main} | {
            event.event_id for event in micro
        } == {event.event_id for event in observation.market_snapshot.events_this_update}
        saw_microstructure = saw_microstructure or bool(micro)
    assert saw_microstructure, "no 1m event was ever published"


def test_publishing_every_timeframe_keeps_one_channel() -> None:
    every_scale = tuple(
        spec.native_timeframe for spec in MODEL_SCALE_SPECS if spec.enabled
    )
    observations = _observe(published_timeframes=every_scale)
    for observation in observations:
        assert set(observation.published_timeframes) == set(observation.active_timeframes)
        assert observation.microstructure_events_this_update == ()
        assert observation.semantic_events_this_update == (
            observation.market_snapshot.events_this_update
        )
