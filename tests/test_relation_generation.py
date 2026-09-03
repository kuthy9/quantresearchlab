"""A cross-timeframe relation is one episode, not one sample per bar.

``RelationState`` is recomputed every completed minute, so a parent retracement
that holds for fifty 5m bars produces fifty identical rows.  Any HTF/LTF study
that treats those as observations counts one market fact fifty times and
inflates its own significance.  v1.3 gives each continuous relation occupancy a
generation: it opens when a role is established, counts how many observations it
spanned, and closes when the role changes or its parent structure ends.
"""
from __future__ import annotations

import collections

import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.market_state import RelationRole
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


def _generations(snapshots):
    seen: dict[str, object] = {}
    for snapshot in snapshots:
        for generation in snapshot.relation_generations.values():
            seen[generation.generation_id] = generation
    return seen


def test_a_relation_occupancy_is_published_as_one_generation(replay) -> None:
    _, snapshots = replay
    generations = _generations(snapshots)
    assert generations, "no relation generation was published"

    for generation in generations.values():
        assert generation.generation_id.startswith(
            f"{generation.relation_id}_generation_"
        )
        assert isinstance(generation.role, RelationRole)
        assert generation.entered_at is not None
        assert generation.observation_count >= 1
        assert generation.updated_at >= generation.entered_at


def test_a_held_role_accumulates_observations_instead_of_repeating(
    replay,
) -> None:
    """The whole point: fifty bars of one role are one generation, not fifty."""

    _, snapshots = replay
    generations = _generations(snapshots)

    spans = [g.observation_count for g in generations.values()]
    assert max(spans) > 1, "no relation ever held a role across bars"
    # Every published snapshot re-resolves every relation, so the number of
    # generations must be far below the number of (snapshot, relation) pairs.
    resolved_rows = sum(
        len(snapshot.relations) for snapshot in snapshots if snapshot is not None
    )
    assert len(generations) < resolved_rows / 2


def test_one_relation_never_holds_two_open_generations(replay) -> None:
    _, snapshots = replay

    for snapshot in snapshots:
        if snapshot is None:
            continue
        open_by_relation = collections.Counter(
            generation.relation_id
            for generation in snapshot.relation_generations.values()
            if generation.terminated_at is None
        )
        assert not [k for k, v in open_by_relation.items() if v > 1]


def test_a_role_change_closes_the_generation_and_names_the_successor(
    replay,
) -> None:
    _, snapshots = replay
    generations = _generations(snapshots)

    closed = [g for g in generations.values() if g.terminated_at is not None]
    assert closed, "no relation generation ever closed"
    for generation in closed:
        assert generation.termination_reason in {
            "role_changed",
            "parent_structure_terminated",
            "child_structure_terminated",
        }
        assert generation.terminated_at >= generation.entered_at
        if generation.termination_reason == "role_changed":
            assert generation.next_role is not None
            assert generation.next_role is not generation.role


def test_a_generation_names_the_structure_claims_it_spanned(replay) -> None:
    _, snapshots = replay
    known = set()
    for snapshot in snapshots:
        if snapshot is not None:
            known.update(snapshot.structure_generations)

    for generation in _generations(snapshots).values():
        for name in (
            generation.parent_structure_generation_id,
            generation.child_structure_generation_id,
        ):
            assert name is None or name in known
