"""A pool candidate leaves the reducer when its pool leaves the authoritative set.

Candidate retirement by age and distance leaves pools and range boundaries to
their own lifecycles -- but the pool lifecycle ended in the liquidity tracker
alone: a resolved pool was compacted from the snapshot and nothing told the
reducer, whose ``formed_liquidity_pool`` candidate stayed for the life of the
process.  Over 2022-02 that was 177 of the 261 1m candidates at bar 6,000 and
897 candidates by month end.  A pool candidate whose pool is no longer in the
authoritative pool set now ends as ``LIQUIDITY_RETIRED`` (``pool_resolved``)
on the bar it leaves, exactly as an aged-out swing does; a range-boundary
candidate ends the same way when its boundary item leaves the inventory.
"""
from __future__ import annotations

from contract.eye import EventKind
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

OWNED = {"formed_liquidity_pool", "range_boundary", "mature_range_boundary"}


def test_pool_candidates_follow_the_authoritative_pool_set() -> None:
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
    previous_offered: set[str] = set()
    pool_retirements = 0
    ever_left = 0
    for bar in _noisy(session_bars(1)[:900]):
        observation = observer.observe(reader.on_bar(bar))
        offered = {
            f"pool:{pool.pool_id}"
            for frame in observation.frames.values()
            for pool in frame.liquidity_pools
        } | {
            item.item_id
            for item in observation.liquidity_inventory
            if item.kind == "range_boundary"
        }
        for timeframe, state in observation.market_snapshot.timeframe_states.items():
            owned = {
                candidate.candidate_id
                for candidate in state.liquidity.candidates
                if candidate.source_kind in OWNED
            }
            # The snapshot is published before this bar's retirements run,
            # so a pool that left the set on this bar may still be in it;
            # nothing older may.
            stale = owned - offered - previous_offered
            assert not stale, (timeframe, sorted(stale)[:3])
        for event in observation.events_this_update:
            if (
                event.kind is EventKind.LIQUIDITY_RETIRED
                and event.details.get("reason") == "pool_resolved"
            ):
                pool_retirements += 1
        ever_left += len(previous_offered - offered)
        previous_offered = offered
    assert ever_left, "the replay resolved no pool"
    assert pool_retirements >= 1
