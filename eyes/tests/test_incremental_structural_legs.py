"""Structural legs are projected from the new swings only.

``build_structural_legs`` is a left fold over the confirmed swings sorted by
pivot: every iteration ends with the current swing as the next anchor, so the
legs of a prefix never change when later swings resolve.  The observer used to
rebuild every leg from every retained swing (up to 2,048) each time one swing
resolved on a scale; it now folds the last anchor over the new swings and keeps
the legs it already projected.  The full rebuild is still what an out-of-order
confirmation or a swing eviction falls back to.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from contract.market import Timeframe
from eyes.core import observation as observation_module
from eyes.core.causal import CausalMarketReader
from eyes.core.market_state import build_structural_legs
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

WARM_UP_BARS = 300


@pytest.fixture(scope="module")
def replay(monkeypatch_module):
    calls: dict[Timeframe, list[tuple[int, int]]] = defaultdict(list)
    bar_index = 0
    original = observation_module.build_structural_legs

    def recording(timeframe, swings, candles, **kwargs):
        calls[timeframe].append((bar_index, len(swings)))
        return original(timeframe, swings, candles, **kwargs)

    monkeypatch_module.setattr(observation_module, "build_structural_legs", recording)
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
    observations = []
    update = None
    for bar_index, bar in enumerate(_noisy(session_bars(1)[:900])):
        update = reader.on_bar(bar)
        observations.append(observer.observe(update))
    return calls, observations, update


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


def test_a_resolved_swing_projects_only_the_new_legs(replay) -> None:
    calls, observations, _ = replay
    frame = observations[-1].frames[Timeframe.M1]
    resolved = [
        swing for swing in frame.swings if swing.confirmed_at is not None
    ]
    assert len(resolved) > 40, len(resolved)
    late = [size for index, size in calls[Timeframe.M1] if index >= WARM_UP_BARS]
    assert len(late) > 20, "too few 1m swings resolved after the warm-up"
    # The last anchor plus the swings that resolved on this bar.
    assert max(late) <= 4, sorted(late)[-5:]


def test_incremental_legs_agree_with_a_full_rebuild(replay) -> None:
    _, observations, update = replay
    for timeframe, frame in observations[-1].frames.items():
        if not frame.structural_legs:
            continue
        rebuilt = build_structural_legs(
            timeframe,
            frame.swings,
            update.histories[timeframe],
            atr=float(frame.metrics["atr"]),
        )
        # The frame holds exactly the legs a rebuild projects, in order, and
        # the newest leg is identical field for field (older legs keep the
        # start ATR frozen when they were first projected; a rebuild whose
        # candle window has moved on would fall back to the current ATR).
        assert [leg.leg_id for leg in rebuilt] == [
            leg.leg_id for leg in frame.structural_legs
        ], timeframe
        assert rebuilt[-1] == frame.structural_legs[-1], timeframe
