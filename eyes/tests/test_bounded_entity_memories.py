"""The bar-keyed memories are bounded, and bounding changes no event.

The emitter remembers, per bar root, its close and price range and the
roots of each scale in clock order; the store keeps the eligible-bar index
of the structural-leg contract and its same-clock reservations.  Every one
of them grew for the life of the run (25,752 bar roots per scale a month)
although what a later event can still cite is bounded in clock: a Swing
freezes its window at confirmation, a leg reaches its start pivot, a
reference level cites the root of the previous period's extreme.  Each of
these memories now keeps its newest entries, and the journal owns the rest.

The per-entity memories (the event that last spoke about a swing, level,
BOS, zone or range) are *not* count-bounded: the emitter re-walks every
retained entity on every frame, so a memory is read for as long as the
slowest scale retains the entity -- 256 4H swings is months -- and the
month replay showed lookups reaching back over the whole run
(``eyes/docs/evidence/eye_memory_2022-02_2026-09-14.md``).  A count bound
there evicts entries that are still read.

The proof that a bar-family bound is above every contract's reach: a
two-session replay under a bound a small fraction of the default emits
exactly the same events, bar for bar, as the unbounded replay did, while
every bounded memory stays at or under the bound.
"""
from __future__ import annotations

import pytest

import eyes.core.event_store as store_module
import eyes.core.semantic_event_emitter as emitter_module
from eyes.core.bounded import BoundedDict
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

SMALL = 1024


def _hash_stream(bars) -> tuple[tuple[str, int], ...]:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            displacement_protocol="configs/primitives_displacement.json",
            zone_protocol="configs/primitives_zones.json",
            range_auction_protocol="configs/primitives_range.json",
            interaction_protocol="configs/primitives_interaction.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
            # The registered Eye's own settings: projection *events* would
            # multiply the journal by ten without adding a semantic fact.
            materialize_event_view=False,
            persist_state_projections=False,
        )
    )
    rows = []
    for bar in bars:
        observer.observe(reader.on_bar(bar))
        rows.append((observer.audit_store.fingerprint(), len(observer.audit_store)))
    return tuple(rows), observer


def _memory_sizes(observer) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for owner_name, owner in (("emitter", observer._emitter), ("store", observer.audit_store)):
        for name, value in vars(owner).items():
            if isinstance(value, BoundedDict):
                sizes[f"{owner_name}.{name}"] = len(value)
            elif isinstance(value, dict) and value and all(
                isinstance(item, list) for item in value.values()
            ):
                for key, item in value.items():
                    sizes[f"{owner_name}.{name}[{getattr(key, 'value', key)}]"] = len(item)
    return sizes


@pytest.fixture(scope="module")
def bars():
    return _noisy(session_bars(2), seed=21)


@pytest.fixture(scope="module")
def unbounded(bars):
    return _hash_stream(bars)


def test_bounding_changes_no_event_and_every_memory_stays_bounded(
    bars, unbounded, monkeypatch
) -> None:
    reference, _ = unbounded
    # A previous-session reference level cites the 1m root of the prior
    # session's extreme, so the per-scale root window must span a session;
    # the bar-keyed memories are shared by every scale and must span the
    # widest swing window in clock time (a 4H window is twenty hours).
    monkeypatch.setattr(emitter_module, "BAR_ROOT_RETENTION_PER_SCALE", 2 * SMALL)
    monkeypatch.setattr(emitter_module, "BAR_MEMORY_RETENTION", 3 * SMALL)
    monkeypatch.setattr(store_module, "ELIGIBLE_BAR_INDEX_RETENTION_PER_SCALE", 2 * SMALL)
    monkeypatch.setattr(store_module, "RESERVATION_MEMORY_RETENTION", 3 * SMALL)
    bounded, observer = _hash_stream(bars)
    assert bounded == reference
    sizes = _memory_sizes(observer)
    assert sizes, "no bounded memory was found"
    # The bar-keyed memories are shared by every scale and bounded by a
    # clock window five times the per-scale bound; everything else by SMALL.
    def bound(name: str) -> int:
        if "by_timeframe" in name or "_eligible_bar_index" in name:
            return 2 * SMALL
        return 3 * SMALL

    over = {name: size for name, size in sizes.items() if size > bound(name)}
    assert not over, over
    # The bounds were exercised: the unbounded populations exceed them.
    _, reference_observer = unbounded
    exceeded = {
        name
        for name, size in _memory_sizes(reference_observer).items()
        if size > bound(name)
    }
    assert {
        "emitter._bar_event_ids_by_timeframe[1m]",
        "emitter._real_bar_event_ids_by_timeframe[1m]",
        "emitter._bar_range_by_event_id",
        "emitter._bar_close_by_event_id",
        "store._normalized_bar_event_ids",
    } <= exceeded, exceeded
    assert any("_eligible_bar_index" in name for name in exceeded), exceeded


def test_default_bounds_hold_the_two_session_populations(unbounded) -> None:
    _, observer = unbounded
    sizes = _memory_sizes(observer)
    assert max(sizes.values()) <= emitter_module.BAR_MEMORY_RETENTION
    assert observer.audit_store.cold_count == 0


# Read for as long as the slowest scale retains the entity, which the
# month replay measured at the whole run; never count-bounded.
ENTITY_MEMORIES = (
    "_confirmed_swing_event_ids",
    "_structural_leg_event_ids",
    "_structure_direction_event_ids",
    "_bar_event_ids_by_candle_id",
    "_level_touch_event_ids",
    "_candidate_level_event_ids",
    "_reached_level_ids",
    "_retired_level_ids",
    "_penetration_event_ids",
    "_raw_break_event_ids",
    "_displacement_event_ids",
    "_protected_swing_event_ids",
    "_terminal_crossing_events",
    "_fvg_created_event_ids",
    "_fvg_first_retest_event_ids",
    "_fvg_terminal_event_ids",
    "_base_origin_core_event_ids",
    "_origin_zone_created_event_ids",
    "_range_created_event_ids",
    "_range_active_event_ids",
    "_balance_range_observed_event_ids",
    "_range_terminal_event_ids",
    "_range_boundary_level_ids",
    "_liquidity_entity_revisions",
)


def test_entity_memories_follow_tracker_retention_not_a_count(unbounded) -> None:
    _, observer = unbounded
    emitter = observer._emitter
    bounded = {
        name
        for name in ENTITY_MEMORIES
        if isinstance(getattr(emitter, name), BoundedDict)
    }
    assert not bounded, bounded
    assert emitter._known_level_touch_order.maxlen is None
    assert not hasattr(emitter_module, "ENTITY_MEMORY_RETENTION")
