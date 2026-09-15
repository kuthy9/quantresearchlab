"""The dealing range is one definition applied on every registered scale.

Group 4 formed ranges on 1H only: the tracker validated 1H candles and zones,
stamped every range, boundary item and manipulation source with 1H, and the
emitter and store contracts hard-coded the same.  The range protocol now names
its scales and the tracker runs the same formation, balance and break rules
once per scale, counting that scale's own completed bars, while one
manipulation funnel spans every scale's mature boundaries and every pool.
"""
from __future__ import annotations

from collections import Counter

import pytest

from contract.eye import DealingRangeLifecycle, EventKind
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig
from eyes.core.range_auction import RangeAuctionProtocol

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

RANGE_PROTOCOL = "configs/primitives_range.json"


@pytest.fixture(scope="module")
def replay():
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            displacement_protocol="configs/primitives_displacement.json",
            zone_protocol="configs/primitives_zones.json",
            range_auction_protocol=RANGE_PROTOCOL,
            interaction_protocol="configs/primitives_interaction.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    observations = [
        observer.observe(reader.on_bar(bar))
        for bar in _noisy(session_bars(2), seed=11)
    ]
    return observer.audit_store.events(), observations


def test_the_protocol_names_its_scales() -> None:
    protocol = RangeAuctionProtocol.from_file(RANGE_PROTOCOL)
    assert protocol.timeframes == (Timeframe.M15, Timeframe.H1, Timeframe.H4)
    assert Timeframe.M1 not in protocol.timeframes


def test_a_range_forms_on_the_15m_scale_from_15m_sources(replay) -> None:
    events, observations = replay
    seen = {
        (range_state.timeframe, range_state.range_id)
        for observation in observations
        for timeframe, frame in observation.frames.items()
        for range_state in frame.dealing_ranges
    }
    by_scale = Counter(timeframe for timeframe, _ in seen)
    assert by_scale[Timeframe.M15] >= 1, by_scale
    for observation in observations:
        for timeframe, frame in observation.frames.items():
            zone_ids = {zone.zone_id for zone in frame.support_resistance}
            for range_state in frame.dealing_ranges:
                assert range_state.timeframe is timeframe
                # Its sources are that scale's own zones.
                assert range_state.lower_source_zone_id in zone_ids or (
                    range_state.lifecycle is DealingRangeLifecycle.BROKEN
                )
            live = [
                r for r in frame.dealing_ranges
                if r.lifecycle is DealingRangeLifecycle.ACTIVE
            ]
            assert len(live) <= 1, (timeframe, live)


def test_range_facts_carry_their_scale(replay) -> None:
    events, observations = replay
    range_events = [
        event for event in events if event.kind is EventKind.DEALING_RANGE_STATE
    ]
    assert {event.timeframe for event in range_events} >= {Timeframe.M15}
    for event in range_events:
        assert event.timeframe is not Timeframe.M1
    boundaries = [
        item
        for observation in observations
        for item in observation.liquidity_inventory
        if item.kind == "range_boundary"
    ]
    if boundaries:
        ranges_by_id = {
            r.range_id: r
            for observation in observations
            for frame in observation.frames.values()
            for r in frame.dealing_ranges
        }
        for item in boundaries:
            (range_id,) = [s for s in item.source_ids if s in ranges_by_id]
            assert item.timeframe is ranges_by_id[range_id].timeframe


def test_every_funnel_snapshot_names_its_scale(replay) -> None:
    _, observations = replay
    for observation in observations:
        keys = [
            (item.timeframe, item.observed_at)
            for item in observation.group4_range_funnel
        ]
        assert len(keys) == len(set(keys))
        assert all(timeframe is not Timeframe.M1 for timeframe, _ in keys)
