"""A validated swing hierarchy is trusted until it changes.

``TimeframeState`` is rebuilt by ``replace`` on nearly every reduced event and
every publication, and each rebuild re-validated the whole hot Swing working
set -- 2,048 views once full, 17 M generator steps per 500 bars at bar 20,000
of 2022-02 -- although the tuple it validated was the very object it had
validated a moment before.  Rank projection rebuilt the id-to-rank map from
the same tuple on every candidate projection, and the geometry settle
re-viewed every Swing of every timeframe whenever any one Swing moved.

A hierarchy is validated once, against its timeframe and the clock it was
validated at; a state that carries the same object at that clock or later
trusts it, anything else is validated afresh.  The validated tuple carries
its derived maps, and the settle re-views only the Swings that moved.
"""
from __future__ import annotations

import pandas as pd
import pytest
from dataclasses import replace

import eyes.core.market_state as market_state
from eyes.core.market_state import (
    ValidatedSwingHierarchy,
    _project_candidate_views,
    reduce_timeframe_state,
)
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from eyes.tests.test_swing_hierarchy_retention import _confirmed
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


class _Counting(ValidatedSwingHierarchy):
    walks = 0

    def __iter__(self):
        type(self).walks += 1
        return super().__iter__()


def _state(count: int = 12):
    state = None
    for event in (_confirmed(index) for index in range(count)):
        state = reduce_timeframe_state(
            state, event, semantic_registry_identity="cache-test"
        )
    return state


def test_a_state_keeps_the_validated_hierarchy_object_across_replace() -> None:
    state = _state()
    hierarchy = state.swing_hierarchy
    assert isinstance(hierarchy, ValidatedSwingHierarchy)
    assert hierarchy.timeframe is Timeframe.M5
    later = replace(
        state,
        quality=replace(state.quality, known_at=state.quality.known_at + pd.Timedelta(minutes=1)),
    )
    assert later.swing_hierarchy is hierarchy
    assert tuple(later.swing_hierarchy) == tuple(hierarchy)


def test_a_trusted_hierarchy_is_not_walked_again() -> None:
    state = _state()
    trusted = _Counting(
        state.swing_hierarchy,
        timeframe=state.timeframe,
        known_at=state.quality.known_at,
    )
    _Counting.walks = 0
    rebuilt = replace(state, swing_hierarchy=trusted)
    assert rebuilt.swing_hierarchy is trusted
    assert _Counting.walks == 0


def test_a_plain_tuple_is_validated_and_a_stale_bound_revalidates() -> None:
    state = _state()
    foreign = tuple(
        replace(view, timeframe=Timeframe.M1) for view in state.swing_hierarchy
    )
    with pytest.raises(ValueError, match="invalid swing hierarchy"):
        replace(state, swing_hierarchy=foreign)
    # Validated at a later clock than the state carries: the future check
    # runs again, and the later assignments are in that state's future.
    earlier_state = _state(4)
    assert state.swing_hierarchy.known_at > earlier_state.quality.known_at
    with pytest.raises(ValueError, match="invalid swing hierarchy"):
        replace(earlier_state, swing_hierarchy=state.swing_hierarchy)


def test_rank_projection_reads_the_hierarchy_once_per_object() -> None:
    state = _state()
    counting = _Counting(
        state.swing_hierarchy,
        timeframe=state.timeframe,
        known_at=state.quality.known_at,
    )
    _Counting.walks = 0
    first = _project_candidate_views(state.liquidity, counting, state.range)
    second = _project_candidate_views(state.liquidity, counting, state.range)
    assert first == second
    assert _Counting.walks <= 1
    assert counting.ranks == {
        view.swing_id: view.semantic_rank.value for view in state.swing_hierarchy
    }


def test_geometry_settle_reviews_only_the_swings_that_moved(monkeypatch) -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    bars = _noisy(session_bars(1)[:400], seed=7)
    for bar in bars[:300]:
        observer.observe(reader.on_bar(bar))
    tree_type = type(observer.market_snapshot_publisher._swing_geometry)
    moved_total = 0
    viewed = 0
    original_admit, original_view = tree_type.admit, tree_type.view_of

    def admit(tree, swing):
        nonlocal moved_total
        moved = original_admit(tree, swing)
        moved_total += len(moved)
        return moved

    def view_of(tree, swing):
        nonlocal viewed
        viewed += 1
        return original_view(tree, swing)

    monkeypatch.setattr(tree_type, "admit", admit)
    monkeypatch.setattr(tree_type, "view_of", view_of)
    for bar in bars[300:]:
        observer.observe(reader.on_bar(bar))
    hierarchy_size = sum(
        len(state.swing_hierarchy)
        for state in observer.last_market_snapshot.timeframe_states.values()
    )
    assert moved_total > 0
    assert viewed <= moved_total, (viewed, moved_total, hierarchy_size)
