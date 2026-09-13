"""One frozen level crossed at one clock is one touch and one penetration.

The registry binds ``liquidity_sweep`` as "First emit LEVEL_TOUCHED and
LEVEL_PENETRATED for one frozen level and crossing_generation_id".  Two emitters
used to publish the same crossing -- the structure frame's swing-break path and
the 1m inventory-crossing path -- each with its own dedup registry, so a 1m
swing broken on the bar that crossed it produced two touches and two
penetrations with distinct ``event_id`` and identical facts.  Two levels that
share a price are a different matter: they are two entities, and every crossing
event now cites its level in ``source_entity_ids`` -- the provenance namespace
for entities (``entity_id`` is reserved for typed lifecycle transports) -- so a
consumer can tell them apart without parsing ``details``.
"""
from __future__ import annotations

from collections import Counter
import random

import pandas as pd
import pytest

from contract.eye import EventKind
from contract.market import Bar
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from shares.tests.helpers import MODEL_SCALE_SPECS


CROSSINGS = (EventKind.LEVEL_TOUCHED, EventKind.LEVEL_PENETRATED)


def _noisy_session(bars: int = 300, *, seed: int = 7) -> list[Bar]:
    """A 1m random walk inside one session.

    The smooth ``session_bars`` grid never confirms a 1m swing, so it never
    reaches the swing-break path; a tick-grid random walk confirms dozens and
    breaks many of them on the bar that crosses the level.
    """
    rng = random.Random(seed)
    start = pd.Timestamp("2025-01-06 09:30", tz="America/New_York")
    price = 100.0
    output: list[Bar] = []
    for index in range(bars):
        close = round(price + rng.choice((-0.75, -0.5, -0.25, 0.25, 0.5, 0.75)), 2)
        high = max(price, close) + 0.25 * rng.choice((0, 1, 2))
        low = min(price, close) - 0.25 * rng.choice((0, 1, 2))
        output.append(
            Bar(start + pd.Timedelta(minutes=index), price, high, low, close, 100.0, "NQH5", 1)
        )
        price = close
    return output


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


def _crossings(events, kind):
    found = [event for event in events if event.kind is kind]
    assert found, f"the replay produced no {kind.value}"
    return found


@pytest.mark.parametrize("kind", CROSSINGS, ids=lambda kind: kind.value)
def test_one_level_crossed_at_one_clock_is_published_once(events, kind) -> None:
    seen = Counter(
        (event.details["level_id"], event.event_time, event.timeframe)
        for event in _crossings(events, kind)
    )
    duplicated = {key: count for key, count in seen.items() if count > 1}
    assert not duplicated, (
        f"{len(duplicated)} {kind.value} crossings were published more than "
        f"once: {sorted(duplicated)[:3]}"
    )


@pytest.mark.parametrize("kind", CROSSINGS, ids=lambda kind: kind.value)
def test_every_crossing_cites_its_level_as_an_entity(events, kind) -> None:
    for event in _crossings(events, kind):
        assert event.details["level_id"] in event.source_entity_ids
