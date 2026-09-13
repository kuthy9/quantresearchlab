"""A primitive is a definition, not a timeframe role.

Displacement, FVG / order-block zones and dealing ranges each ran on one
scale fixed in their protocol file (5m, 5m, 1H), so on the 2022 gate dataset
twelve of the Brain's 150 features were NaN on every row:
``{1m,15m,1H,4H}_displacement_score`` and
``{1m,5m,15m,4H}_range_{width_atr,location}``.  Structure and liquidity
already ran per scale through one tracker per timeframe; the three
single-scale trackers now do the same, and the protocol file names the
scales it publishes on.  The 5m displacement facts and the 1H dealing-range
facts are the same as before -- the definition did not move, it was applied
more widely.
"""
from __future__ import annotations

from collections import Counter

import pytest

from contract.eye import EventKind
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.displacement import DisplacementProtocol
from eyes.core.observation import CausalObserver, ObserverConfig

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
    observations = [observer.observe(reader.on_bar(bar)) for bar in bars]
    return observer.audit_store.events(), observations


@pytest.fixture(scope="module")
def replay():
    return _replay(session_bars(3))


def test_the_displacement_protocol_names_the_scales_it_publishes_on() -> None:
    protocol = DisplacementProtocol.from_file("configs/primitives_displacement.json")
    assert Timeframe.M5 in protocol.timeframes
    assert Timeframe.M15 in protocol.timeframes
    assert Timeframe.H1 in protocol.timeframes


def test_displacement_is_observed_on_more_than_the_5m_scale(replay) -> None:
    events, _ = replay
    by_timeframe = Counter(
        event.timeframe
        for event in events
        if event.kind is EventKind.DISPLACEMENT_OBSERVED
    )
    assert by_timeframe[Timeframe.M5], "the 5m scale lost its displacement facts"
    assert by_timeframe[Timeframe.M15], "no displacement was observed on 15m"


def test_a_higher_scale_delivery_state_carries_a_displacement_score(replay) -> None:
    _, observations = replay
    scored = {
        timeframe
        for observation in observations
        for timeframe, state in observation.market_snapshot.timeframe_states.items()
        if state.delivery.displacement_score is not None
    }
    assert Timeframe.M5 in scored
    assert Timeframe.M15 in scored
