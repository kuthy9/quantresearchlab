"""A failed publish must roll back, and rolling back must not copy history.

``publish`` snapshots the publisher before it runs so a raising bar leaves no
half-applied state.  That snapshot was taken with ``copy.deepcopy`` over two
containers that grow with every confirmed Swing, so the cost of being able to
roll back one bar grew without bound: at 4,000 bars it was 56% of replay time.
Both containers hold values that are never mutated in place -- frozen views and
freshly built primitives -- so a shallow snapshot restores exactly as much
state, at a cost that does not grow with history.
"""
from __future__ import annotations

import pandas as pd
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


def test_the_geometry_snapshot_shares_its_views_instead_of_copying_them(
    publisher,
) -> None:
    tree = publisher._swing_geometry
    assert tree._nodes, "the fixture confirmed no Swings"

    state = tree.snapshot()

    for swing_id, view in tree._nodes.items():
        assert state["_nodes"][swing_id] is view


def test_the_geometry_snapshot_survives_mutation_of_the_live_tree(
    publisher,
) -> None:
    """Containers must be copied even though their values need not be."""

    tree = publisher._swing_geometry
    state = tree.snapshot()
    parents_before = dict(tree._parent)
    children_before = {k: list(v) for k, v in tree._children.items()}
    depth_before = dict(tree._depth)
    index_before = {k: list(v) for k, v in tree._index.items()}

    victim = next(iter(tree._nodes))
    tree._parent[victim] = "invented-parent"
    tree._depth[victim] = 99
    tree._children.setdefault(victim, []).append("invented-child")
    for order in tree._index.values():
        order.append("invented-node")

    tree.restore(state)

    assert tree._parent == parents_before
    assert tree._depth == depth_before
    assert {k: list(v) for k, v in tree._children.items()} == children_before
    assert {k: list(v) for k, v in tree._index.items()} == index_before


def test_a_fresh_tree_snapshots_and_restores_without_nodes() -> None:
    tree = SwingGeometryTree()

    state = tree.snapshot()
    tree._parent["ghost"] = None
    tree.restore(state)

    assert tree._parent == {}
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
