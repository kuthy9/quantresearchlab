"""A swing's lifecycle prefix cools once the structure tracker lets it go.

EventMemory keeps a still-transitionable prefix hot even when the current
typed snapshot omits its entity, so a later terminal transition finds its
history.  A swing, however, can only transition through the structure
tracker, and the tracker drops swings without a fact: the oldest confirmed
swing leaves the bounded ``retained_swings`` deque, and a forming candidate
an outside bar dominates on both sides is declined outright.  Every such
prefix stayed live for the life of the process -- ``swing:confirmed`` grew
33 -> 933 and ``swing:forming`` 33 -> 770 over 20,000 bars of 2022-02, and
every completed minute walked all of them.  A swing prefix now stays hot
only while the tracker still exposes the swing.
"""
from __future__ import annotations

from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


def test_live_swing_prefixes_are_exactly_the_swings_the_tracker_exposes() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    for bar in _noisy(session_bars(1)[:400], seed=3):
        observation = observer.observe(reader.on_bar(bar))
        exposed = {
            f"swing:{swing.swing_id}"
            for frame in observation.frames.values()
            for swing in frame.swings
        }
        retained = {
            key
            for key in observer.memory._entity_timelines
            if key.startswith("swing:")
        }
        live = {
            key
            for key in observer.memory.live_entity_keys()
            if key.startswith("swing:")
        }
        # Every retained swing prefix, live or terminal, belongs to a swing
        # the tracker still holds; nothing outlives its owner.
        assert retained <= exposed, sorted(retained - exposed)[:5]
        assert live <= exposed
    assert exposed, "the replay exposed no swing"
