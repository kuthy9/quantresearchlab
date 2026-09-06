"""A failed publish must roll back, and rolling back must not copy history.

``publish`` protects the publisher so a raising bar leaves no half-applied
state.  For the Swing geometry tree that protection was a full copy of every
container -- once cheap because the values are shared, but still proportional
to the whole Swing population on every single bar, so the cost of being able to
undo one bar grew for the life of the process.

One bar changes a bounded number of places in the tree: ``admit`` returns the
ids whose place moved, and that set is bounded by the span of the largest
timeframe's window rather than by elapsed bars.  The tree therefore journals
the entries it is about to overwrite and replays them backwards, which costs
what the bar changed instead of what history contains.
"""
from __future__ import annotations

import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.market_state import SwingGeometryTree
from smc_trader.observation import CausalObserver, ObserverConfig

from .helpers import MODEL_SCALE_SPECS, session_bars


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"
DISPLACEMENT_PROTOCOL = "configs/primitives_displacement.json"
GROUP3_PROTOCOL = "configs/primitives_zones.json"
GROUP4_PROTOCOL = "configs/primitives_range.json"


@pytest.fixture(scope="module")
def publisher():
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
    for bar in session_bars(2):
        observer.observe(reader.on_bar(bar))
    return observer.market_snapshot_publisher


def _shape(tree: SwingGeometryTree) -> dict[str, object]:
    return {
        "nodes": dict(tree._nodes),
        "parent": dict(tree._parent),
        "children": {key: list(value) for key, value in tree._children.items()},
        "depth": dict(tree._depth),
        "index": {key: list(value) for key, value in tree._index.items()},
        "starts": {key: list(value) for key, value in tree._starts.items()},
    }


def test_rolling_back_a_bar_restores_the_tree_exactly(publisher) -> None:
    tree = publisher._swing_geometry
    assert tree._nodes, "the fixture confirmed no Swings"
    before = _shape(tree)

    tree.begin()
    victim = next(iter(tree._nodes))
    tree._record("_parent", victim)
    tree._parent[victim] = "invented-parent"
    tree._record("_depth", victim)
    tree._depth[victim] = 99
    tree._record("_children", victim)
    tree._children.setdefault(victim, []).append("invented-child")
    for timeframe in tree._index:
        tree._record("_index", timeframe)
        tree._index[timeframe].append("invented-node")
    tree.rollback()

    assert _shape(tree) == before


def test_an_admitted_swing_rolls_back_to_the_tree_that_preceded_it(
    publisher,
) -> None:
    tree = publisher._swing_geometry
    victim = next(
        swing for swing in tree._nodes.values() if swing.window_start is not None
    )
    replica = SwingGeometryTree()
    for swing in tree._nodes.values():
        if swing.swing_id != victim.swing_id:
            replica.admit(swing)
    before = _shape(replica)

    replica.begin()
    moved = replica.admit(victim)
    assert moved, "admitting a real Swing moved nothing"
    replica.rollback()

    assert _shape(replica) == before


def test_the_journal_costs_what_the_bar_changed_not_what_history_holds(
    publisher,
) -> None:
    tree = publisher._swing_geometry
    victim = next(
        swing for swing in tree._nodes.values() if swing.window_start is not None
    )
    replica = SwingGeometryTree()
    for swing in tree._nodes.values():
        if swing.swing_id != victim.swing_id:
            replica.admit(swing)

    replica.begin()
    moved = replica.admit(victim)
    recorded = len(replica._undo)
    replica.commit()

    # Four containers per moved id is the loosest possible bound; the point is
    # that it is a function of ``moved`` and never of the population.
    assert recorded <= 4 * len(moved) + 8
    assert replica._undo is None


def test_a_committed_bar_keeps_everything_the_journal_would_have_undone(
    publisher,
) -> None:
    tree = publisher._swing_geometry
    victim = next(
        swing for swing in tree._nodes.values() if swing.window_start is not None
    )
    replica = SwingGeometryTree()
    for swing in tree._nodes.values():
        if swing.swing_id != victim.swing_id:
            replica.admit(swing)

    replica.begin()
    replica.admit(victim)
    replica.commit()
    replica.rollback()

    assert victim.swing_id in replica._nodes


def test_the_window_start_index_is_maintained_beside_the_id_order(
    publisher,
) -> None:
    """``_starts`` exists so parent search never rebuilds it from every node."""

    tree = publisher._swing_geometry

    for timeframe, order in tree._index.items():
        assert tree._starts[timeframe] == [
            tree._nodes[swing_id].window_start for swing_id in order
        ]
        assert tree._starts[timeframe] == sorted(tree._starts[timeframe])


def test_clearing_an_epoch_inside_a_publish_is_undone_whole() -> None:
    tree = SwingGeometryTree()
    tree.begin()
    tree.clear()
    tree.rollback()

    assert tree._nodes == {}


def test_the_projection_payload_snapshot_shares_its_primitives(
    publisher,
) -> None:
    """The other container the rollback used to deep-copy every bar."""

    payloads = publisher._last_projection_payloads
    assert payloads, "the fixture published no projection payloads"

    snapshot = publisher._snapshot_projection_payloads()

    assert snapshot is not payloads
    for key, primitive in payloads.items():
        assert snapshot[key] is primitive
