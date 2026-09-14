"""A consumed swing item is exposed for its retention, then leaves the inventory.

The inventory kept every consumed swing for as long as the structure tracker
retained the swing (256 per scale): at bar 2,500 on the real tape 194 of the
225 inventory items were consumed swings, re-projected into every observation
and scanned by every consumer.  Consumption is delivered as LEVEL_TOUCHED /
LEVEL_REACHED on the bar it happens and the fact stays in the event log; the
consumed item now stays in ``liquidity_inventory`` for
``consumed_item_retention_native_bars`` of its own scale's bars, counting the
bar of consumption, and is then dropped from the observation.  The reducer's
candidate is untouched: a swept level still disarms, can re-arm, and retires
by the age and distance rules.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from contract.eye import LiquidityInventoryLifecycle
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.liquidity import LiquidityConfig
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

PROTOCOL = "configs/primitives_structure_liquidity.json"


@pytest.fixture(scope="module")
def observations():
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=PROTOCOL,
            liquidity_protocol=PROTOCOL,
            displacement_protocol="configs/primitives_displacement.json",
            zone_protocol="configs/primitives_zones.json",
            range_auction_protocol="configs/primitives_range.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    return [observer.observe(reader.on_bar(bar)) for bar in _noisy(session_bars(1)[:900])]


def test_the_protocol_names_the_consumed_retention() -> None:
    config = LiquidityConfig.from_file(PROTOCOL, tick_size=0.25, atr_period=14)
    assert config.consumed_item_retention_native_bars == 1


def test_a_consumed_swing_is_exposed_for_one_native_bar_then_dropped(observations) -> None:
    exposures: dict[str, list] = defaultdict(list)
    timeframes: dict[str, Timeframe] = {}
    for observation in observations:
        for item in observation.liquidity_inventory:
            if item.kind == "swing" and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED:
                exposures[item.item_id].append(observation.asof)
                timeframes[item.item_id] = item.timeframe
    assert exposures, "the replay consumed no swing"
    assert {tf for tf in timeframes.values()} >= {Timeframe.M1, Timeframe.M5}
    for item_id, clocks in exposures.items():
        # One native bar of the item's own scale, counting the bar of
        # consumption: one observation on 1m, five on 5m.
        assert 1 <= len(clocks) <= timeframes[item_id].minutes, (item_id, clocks)


def test_the_reducer_candidate_outlives_the_consumed_item(observations) -> None:
    last = observations[-1]
    offered = {item.item_id for item in last.liquidity_inventory}
    disarmed = [
        candidate
        for candidate in last.market_snapshot.timeframe_states[Timeframe.M1].liquidity.candidates
        if candidate.source_kind == "confirmed_swing"
        and candidate.lifecycle in {"disarmed", "rearmable", "rearmed"}
        and candidate.candidate_id not in offered
    ]
    assert disarmed, "no swept 1m swing survived in the reducer after leaving the inventory"
