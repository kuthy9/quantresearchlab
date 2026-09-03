"""Geometric nesting is not a semantic role.

``SwingHierarchyView`` only ever carried ``semantic_rank`` and the role depth
derived from it, so "how deeply is this swing nested inside other swings" and
"what job is this swing currently doing" were the same number.  They are not
the same question: a MICRO swing can sit three levels deep inside larger ones,
and an EXTERNAL swing can be a geometric root.  v1.3 gives every confirmed
swing its own geometric tree -- parent, depth and children -- decided purely by
containment of the definitional window in time and in price.
"""
from __future__ import annotations

import collections

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.market_state import SwingHierarchyView
from smc_trader.model import EventKind, Timeframe
from smc_trader.observation import CausalObserver, ObserverConfig

from .helpers import MODEL_SCALE_SPECS, session_bars


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"
DISPLACEMENT_PROTOCOL = "configs/primitives_displacement.json"
GROUP3_PROTOCOL = "configs/primitives_zones.json"
GROUP4_PROTOCOL = "configs/primitives_range.json"


@pytest.fixture(scope="module")
def replay():
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
            displacement_protocol=DISPLACEMENT_PROTOCOL,
            zone_protocol=GROUP3_PROTOCOL,
            range_auction_protocol=GROUP4_PROTOCOL,
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    snapshots = []
    for bar in session_bars(2):
        observer.observe(reader.on_bar(bar))
        snapshots.append(observer.last_market_snapshot)
    return observer, snapshots


def _hierarchies(snapshots):
    """Every swing visible in the richest published snapshot, by identity.

    The tree spans timeframes: an H1 swing's window encloses whole M5 swings,
    while two swings on one timeframe share a window length and can never
    enclose each other.  So it is settled over the whole visible population,
    not per timeframe.
    """

    best: dict[str, object] = {}
    for snapshot in snapshots:
        if snapshot is None:
            continue
        merged = {
            swing.swing_id: swing
            for state in snapshot.timeframe_states.values()
            for swing in state.swing_hierarchy
        }
        if len(merged) > len(best):
            best = merged
    assert best, "no timeframe ever published a swing hierarchy"
    return best


def test_a_confirmed_swing_carries_its_definitional_window(replay) -> None:
    _, snapshots = replay

    for swing in _hierarchies(snapshots).values():
        assert swing.window_start is not None
        assert swing.window_end is not None
        assert swing.window_start < swing.window_end
        assert 0.0 < swing.window_low <= swing.window_high


def test_the_geometric_parent_contains_the_child_in_time_and_price(
    replay,
) -> None:
    _, snapshots = replay

    by_id = _hierarchies(snapshots)
    for swing in by_id.values():
        if swing.geometric_parent_id is None:
            assert swing.geometric_depth == 0
            continue
        parent = by_id[swing.geometric_parent_id]
        assert parent.window_start <= swing.window_start
        assert parent.window_end >= swing.window_end
        assert parent.window_low <= swing.window_low
        assert parent.window_high >= swing.window_high
        assert parent.duration_seconds > swing.duration_seconds
        assert swing.geometric_depth == parent.geometric_depth + 1


def test_children_and_parents_name_each_other(replay) -> None:
    _, snapshots = replay

    by_id = _hierarchies(snapshots)
    expected: dict[str, set[str]] = collections.defaultdict(set)
    for swing in by_id.values():
        if swing.geometric_parent_id is not None:
            expected[swing.geometric_parent_id].add(swing.swing_id)
    assert any(expected.values()), "the tree never nested anything"
    for swing in by_id.values():
        assert set(swing.child_ids) == expected[swing.swing_id]
        assert len(set(swing.child_ids)) == len(swing.child_ids)
        for child_id in swing.child_ids:
            assert by_id[child_id].geometric_parent_id == swing.swing_id


def test_the_tree_is_acyclic_and_reaches_a_root(replay) -> None:
    _, snapshots = replay

    by_id = _hierarchies(snapshots)
    for swing in by_id.values():
        seen = {swing.swing_id}
        cursor = swing
        while cursor.geometric_parent_id is not None:
            assert cursor.geometric_parent_id not in seen
            seen.add(cursor.geometric_parent_id)
            cursor = by_id[cursor.geometric_parent_id]
        assert cursor.geometric_depth == 0


def test_geometric_depth_is_not_a_restatement_of_the_semantic_role(
    replay,
) -> None:
    """The decoupling itself: neither number determines the other."""

    _, snapshots = replay
    pairs = {
        (swing.semantic_rank, swing.geometric_depth)
        for swing in _hierarchies(snapshots).values()
    }

    depths_per_rank = collections.defaultdict(set)
    ranks_per_depth = collections.defaultdict(set)
    for rank, depth in pairs:
        depths_per_rank[rank].add(depth)
        ranks_per_depth[depth].add(rank)

    assert any(len(depths) > 1 for depths in depths_per_rank.values()), (
        "every semantic rank sits at exactly one geometric depth; the two are "
        "still the same number"
    )
    # The role depth remains available and untouched by the geometry.
    for swing in _hierarchies(snapshots).values():
        assert swing.role_depth == swing.nesting_depth


def test_the_window_a_swing_publishes_is_the_window_it_was_confirmed_from(
    replay,
) -> None:
    """Geometry must be the frozen bars, not a later re-derivation."""

    observer, snapshots = replay
    events = observer.audit_store.events()
    by_id = {event.event_id: event for event in events}
    confirmations = [
        event for event in events if event.kind is EventKind.SWING_CONFIRMED
    ]
    assert confirmations

    for event in confirmations:
        parents = [by_id[source] for source in event.source_event_ids]
        assert parents
        assert event.evidence["window_low"] == pytest.approx(
            min(float(parent.evidence["low"]) for parent in parents)
        )
        assert event.evidence["window_high"] == pytest.approx(
            max(float(parent.evidence["high"]) for parent in parents)
        )
        assert event.evidence["window_end"] == max(
            parent.known_at for parent in parents
        ).isoformat()


def test_a_bar_that_confirms_nothing_re_derives_nothing() -> None:
    """The property the whole tree depends on to stay affordable.

    Settling the population from scratch on every bar cost 70% of a real
    replay and grows with the square of a population that only ever grows.
    Only a newly confirmed Swing may cause work.
    """

    from smc_trader.market_state import SwingGeometryTree

    tree = SwingGeometryTree()
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
            displacement_protocol=DISPLACEMENT_PROTOCOL,
            zone_protocol=GROUP3_PROTOCOL,
            range_auction_protocol=GROUP4_PROTOCOL,
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    quiet = 0
    for bar in session_bars(2):
        observer.observe(reader.on_bar(bar))
        snapshot = observer.last_market_snapshot
        if snapshot is None:
            continue
        for state in snapshot.timeframe_states.values():
            for swing in state.swing_hierarchy:
                if tree.admit(swing) == set():
                    quiet += 1

    assert quiet, "every swing was new; the cache was never exercised"


def test_a_later_window_adopts_the_swings_it_encloses() -> None:
    """Adoption must be retroactive, or almost everything stays a root.

    A Swing's enclosing higher-timeframe window is confirmed at or after the
    Swing it contains -- an H1 window cannot be known before the M5 Swings
    inside it are.  An append-only assignment would therefore find no parent
    for nearly every Swing.
    """

    from smc_trader.market_state import SwingGeometryTree
    from smc_trader.model import SwingRank
    from smc_trader.market_state import SwingRankAssignment

    def _view(swing_id, timeframe, start_minutes, span_minutes, low, high):
        clock = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")
        start = clock + pd.Timedelta(minutes=start_minutes)
        return SwingHierarchyView(
            swing_id=swing_id,
            timeframe=timeframe,
            semantic_rank=SwingRank.MICRO,
            nesting_depth=0,
            assignments=(
                SwingRankAssignment(
                    rank=SwingRank.MICRO,
                    assigned_at=start,
                    assignment_event_id=f"assign-{swing_id}",
                    source_kind="swing_confirmed",
                ),
            ),
            window_start=start,
            window_end=start + pd.Timedelta(minutes=span_minutes),
            window_low=low,
            window_high=high,
        )

    tree = SwingGeometryTree()
    child = _view("child", Timeframe.M5, 10, 25, 100.0, 110.0)
    tree.admit(child)
    assert tree.view_of(child).geometric_parent_id is None

    parent = _view("parent", Timeframe.H1, 0, 300, 90.0, 120.0)
    moved = tree.admit(parent)

    assert "child" in moved
    settled_child = tree.view_of(child)
    settled_parent = tree.view_of(parent)
    assert settled_child.geometric_parent_id == "parent"
    assert settled_child.geometric_depth == 1
    assert settled_parent.geometric_depth == 0
    assert settled_parent.child_ids == ("child",)
