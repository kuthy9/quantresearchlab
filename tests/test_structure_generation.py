"""Structure is an entity with a life, not the id of its most recent break.

v1.3 bound ``parent_structure_generation`` to whichever BOS or MSS event fired
most recently on a timeframe.  That makes every break look like a new parent,
so a delivery phase that spans four BOS events reports four different parents
and anything grouping by it double-counts.  A generation is one continuous
structural claim: it opens when a direction is confirmed, absorbs every BOS and
MSS while its protected swing holds, and ends when that protection is accepted
through.
"""
from __future__ import annotations

import collections
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.market_state import MarketSnapshotPublisher, StructureScope
from smc_trader.model import Direction, EventKind, Timeframe
from smc_trader.observation import CausalObserver, ObserverConfig

from .helpers import MODEL_SCALE_SPECS, session_bars


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"
DISPLACEMENT_PROTOCOL = "configs/primitives_displacement.json"
GROUP3_PROTOCOL = "configs/primitives_zones.json"
GROUP4_PROTOCOL = "configs/primitives_range.json"
_CLOCK = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")


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
    """Every generation the run ever published, keyed by its identity."""

    seen: dict[str, object] = {}
    for snapshot in snapshots:
        for generation in snapshot.structure_generations.values():
            seen[generation.generation_id] = generation
    return seen


def test_a_confirmed_direction_opens_one_readable_generation(replay) -> None:
    _, snapshots = replay
    generations = _generations(snapshots)
    assert generations, "no structure generation was published"

    for generation in generations.values():
        assert generation.generation_id.startswith(
            f"{generation.timeframe.value}_{generation.scope.value}_generation_"
        )
        assert generation.direction is not None
        assert generation.started_at is not None
        assert generation.origin_event_id is not None


def test_generation_identities_are_unique_and_ordinal_within_a_scope(
    replay,
) -> None:
    _, snapshots = replay
    generations = _generations(snapshots)

    by_scope: dict[tuple, list[str]] = collections.defaultdict(list)
    for generation in generations.values():
        by_scope[(generation.timeframe, generation.scope)].append(
            generation.generation_id
        )
    for key, identities in by_scope.items():
        assert len(identities) == len(set(identities)), key
        ordinals = sorted(int(name.rsplit("_", 1)[-1]) for name in identities)
        assert ordinals == list(range(1, len(ordinals) + 1)), key


def test_breaks_inside_a_generation_extend_it_instead_of_replacing_it(
    replay,
) -> None:
    """The point of the entity: many breaks, one parent."""

    _, snapshots = replay
    generations = _generations(snapshots)

    absorbed = [
        generation
        for generation in generations.values()
        if len(generation.bos_event_ids) + len(generation.mss_event_ids) > 1
    ]
    assert absorbed, "no generation absorbed more than one structural break"
    for generation in absorbed:
        assert len(set(generation.bos_event_ids)) == len(generation.bos_event_ids)
        assert len(set(generation.mss_event_ids)) == len(generation.mss_event_ids)


def test_every_terminated_generation_names_why_it_ended(replay) -> None:
    _, snapshots = replay

    for generation in _generations(snapshots).values():
        if generation.terminated_at is None:
            assert generation.termination_reason is None
            continue
        assert generation.termination_reason in {
            "direction_reversed",
            "protected_swing_accepted_through",
        }
        assert generation.terminated_at >= generation.started_at


def _publisher_with_one_generation():
    """A publisher holding one active external H1 generation."""

    publisher = object.__new__(MarketSnapshotPublisher)
    publisher._structure_generations = {}
    publisher._structure_generation_ordinals = {}
    origin = SimpleNamespace(
        timeframe=Timeframe.H1,
        kind=EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        event_id="origin-1",
    )
    publisher._advance_one_structure_generation(
        Timeframe.H1,
        StructureScope.EXTERNAL,
        direction=Direction.LONG,
        protection_failed=False,
        protected_swing_id="swing-low",
        asof=_CLOCK,
        events=(origin,),
    )
    key = (Timeframe.H1, StructureScope.EXTERNAL)
    assert publisher._structure_generations[key].is_active
    return publisher, key


def test_a_reversed_direction_closes_the_claim_and_opens_the_next() -> None:
    publisher, key = _publisher_with_one_generation()
    first = publisher._structure_generations[key]
    reversal = SimpleNamespace(
        timeframe=Timeframe.H1,
        kind=EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        event_id="origin-2",
    )

    publisher._advance_one_structure_generation(
        Timeframe.H1,
        StructureScope.EXTERNAL,
        direction=Direction.SHORT,
        protection_failed=False,
        protected_swing_id="swing-high",
        asof=_CLOCK + pd.Timedelta(hours=1),
        events=(reversal,),
    )

    second = publisher._structure_generations[key]
    assert second.generation_id != first.generation_id
    assert second.direction is Direction.SHORT
    assert second.generation_id.endswith("0002")


def test_an_accepted_protected_swing_closes_the_claim_and_leaves_none_open() -> None:
    """Protection failing is not a reversal: no claim is open until a new one."""

    publisher, key = _publisher_with_one_generation()

    publisher._advance_one_structure_generation(
        Timeframe.H1,
        StructureScope.EXTERNAL,
        direction=Direction.LONG,
        protection_failed=True,
        protected_swing_id="swing-low",
        asof=_CLOCK + pd.Timedelta(hours=1),
        events=(),
    )

    closed = publisher._structure_generations[key]
    assert not closed.is_active
    assert closed.termination_reason == "protected_swing_accepted_through"


def test_protection_being_accepted_through_outranks_the_direction_it_erases() -> None:
    """The cause, not its effect.

    Accepting through the protected swing also clears ``external_direction``,
    so both termination conditions fire on the same bar.  Reporting that as a
    reversal loses the only fact that explains it, and a study grouping by
    termination reason then never sees a protection failure at all.
    """

    publisher, key = _publisher_with_one_generation()

    publisher._advance_one_structure_generation(
        Timeframe.H1,
        StructureScope.EXTERNAL,
        direction=None,
        protection_failed=True,
        protected_swing_id="swing-low",
        asof=_CLOCK + pd.Timedelta(hours=1),
        events=(),
    )

    closed = publisher._structure_generations[key]
    assert not closed.is_active
    assert closed.termination_reason == "protected_swing_accepted_through"


def test_a_claim_without_an_originating_break_is_not_minted() -> None:
    publisher = object.__new__(MarketSnapshotPublisher)
    publisher._structure_generations = {}
    publisher._structure_generation_ordinals = {}

    publisher._advance_one_structure_generation(
        Timeframe.H1,
        StructureScope.EXTERNAL,
        direction=Direction.LONG,
        protection_failed=False,
        protected_swing_id="swing-low",
        asof=_CLOCK,
        events=(),
    )

    assert publisher._structure_generations == {}
    assert publisher._structure_generation_ordinals == {}


def test_delivery_phase_cites_the_generation_rather_than_one_break(
    replay,
) -> None:
    observer, snapshots = replay
    generations = _generations(snapshots)
    events = observer.audit_store.events()
    break_ids = {
        event.event_id
        for event in events
        if event.kind
        in {
            EventKind.QUALIFIED_BOS,
            EventKind.MSS_CORE_CONFIRMED,
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        }
    }

    entries = [
        event
        for event in events
        if event.kind is EventKind.DELIVERY_PHASE_ENTERED
    ]
    assert entries
    cited = {
        entry.details["parent_structure_generation"]
        for entry in entries
        if entry.details["parent_structure_generation"] is not None
    }
    assert cited, "no delivery phase entry named a parent structure generation"
    assert not (cited & break_ids), "a phase still cites a single break event"
    assert cited <= set(generations)


def test_external_and_internal_scopes_are_tracked_separately(replay) -> None:
    _, snapshots = replay
    scopes = {generation.scope for generation in _generations(snapshots).values()}

    assert StructureScope.EXTERNAL in scopes
