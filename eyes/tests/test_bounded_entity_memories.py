"""Per-entity and per-bar memories are bounded, and bounding changes no event.

The emitter remembers, per swing, level, leg, displacement, zone, range and
bar root, the event that last spoke about it so a later event can cite it;
the store keeps the same kind of memory for its reservations and its
eligible-bar index.  Every one of them grew for the life of the run (about
150,000 entries a month, 25,752 bar roots per scale) although the entities a
later event can still cite are bounded by their trackers' retention.  Each
memory now keeps its newest entries -- a shared bound far above any live
population -- and the journal owns the rest.

The proof that a bound is above every contract's reach: a two-session
replay under a bound a small fraction of the default emits exactly the same
events, bar for bar, as the unbounded replay did, while every memory stays
at or under the bound.  The bar-root and bar-keyed memories -- nine tenths
of the entries -- are exercised here; the per-entity memories grow by a few
hundred entries a session, so their default holds for several weeks and is
exercised by the month replays instead.
"""
from __future__ import annotations

import pytest

import eyes.core.event_store as store_module
import eyes.core.semantic_event_emitter as emitter_module
from eyes.core.bounded import BoundedDict, BoundedSet
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
            if isinstance(value, (BoundedDict, BoundedSet)):
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
        if name.startswith("emitter._bar_") or "_normalized_bar_event_ids" in name:
            return 3 * SMALL
        return emitter_module.ENTITY_MEMORY_RETENTION

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
