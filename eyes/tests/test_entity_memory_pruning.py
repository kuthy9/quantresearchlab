"""Entity memories follow tracker retention: pruning changes no event.

The emitter remembers, per swing, level, BOS, displacement, zone or range,
the event that last spoke about it.  Such an entry is read for as long as
some tracker or reducer retains the entity or another entity that cites it,
and no longer: once nothing the Eye publishes names the entity, no later
event can cite it.  The observer therefore harvests every identifier from
the states it publishes and drops the entries nothing reaches, whenever the
memories have doubled since the last pass.

The proof: a two-session replay pruning at every doubling above a tiny
floor emits exactly the same events, bar for bar, as the replay that never
prunes, while its entity memories end smaller.
"""
from __future__ import annotations

import pytest

import eyes.core.observation as observation_module
import eyes.core.semantic_event_emitter as emitter_module
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

ENTITY_MEMORIES = (
    "_confirmed_swing_event_ids",
    "_structural_leg_event_ids",
    "_structure_direction_event_ids",
    "_bar_event_ids_by_candle_id",
    "_level_touch_event_ids",
    "_candidate_level_event_ids",
    "_reached_level_ids",
    "_retired_level_ids",
    "_known_level_touch_ids",
    "_known_level_touch_order",
    "_penetration_event_ids",
    "_raw_break_event_ids",
    "_displacement_event_ids",
    "_protected_swing_event_ids",
    "_terminal_crossing_events",
    "_terminal_crossing_levels",
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


def _replay(bars, *, every_bar: bool = False):
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
            materialize_event_view=False,
            persist_state_projections=False,
        )
    )
    rows = []
    for bar in bars:
        if every_bar:
            # Pull the doubling mark back so the pass runs on every bar.
            observer._entity_memory_prune_mark = 0
        observer.observe(reader.on_bar(bar))
        rows.append((observer.audit_store.fingerprint(), len(observer.audit_store)))
    return tuple(rows), observer


def _sizes(observer) -> dict[str, int]:
    emitter = observer._emitter
    return {name: len(getattr(emitter, name)) for name in ENTITY_MEMORIES}


@pytest.fixture(scope="module")
def bars():
    return _noisy(session_bars(2), seed=21)


@pytest.fixture(scope="module")
def unpruned(bars, monkeypatch_module):
    monkeypatch_module.setattr(observation_module, "ENTITY_MEMORY_PRUNE_FLOOR", None)
    return _replay(bars)


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patcher:
        yield patcher


def test_pruning_at_every_doubling_changes_no_event(bars, unpruned, monkeypatch) -> None:
    reference, reference_observer = unpruned
    monkeypatch.setattr(observation_module, "ENTITY_MEMORY_PRUNE_FLOOR", 256)
    # The candle-id memory keeps a recent window besides the live entities;
    # shrink it so the window is exercised on two sessions.
    monkeypatch.setattr(emitter_module, "BAR_MEMORY_RETENTION", 3072)
    pruned, observer = _replay(bars)
    assert pruned == reference
    assert observer.entity_memory_prunes > 3, observer.entity_memory_prunes
    before = _sizes(reference_observer)
    after = _sizes(observer)
    assert sum(after.values()) < sum(before.values()), (before, after)
    shrank = {name for name in ENTITY_MEMORIES if after[name] < before[name]}
    # Two sessions retire candidates and their crossings; swings stay in
    # the hierarchy's hot retention (2,048 per scale) for far longer.
    assert {
        "_candidate_level_event_ids",
        "_level_touch_event_ids",
        "_penetration_event_ids",
        "_terminal_crossing_events",
        "_terminal_crossing_levels",
    } <= shrank, shrank
    assert after["_terminal_crossing_levels"] == after["_terminal_crossing_events"]
    assert not any(after[name] > before[name] for name in ENTITY_MEMORIES)


def test_pruning_on_every_bar_changes_no_event(bars, unpruned, monkeypatch) -> None:
    # However often the pass runs, it must keep every entity a tracker can
    # still cite.  A base-origin core is the case the published states miss:
    # its impulse publishes it once, and until a break qualifies the order
    # block only the zone tracker's pending candidate holds it.
    reference, _ = unpruned
    monkeypatch.setattr(observation_module, "ENTITY_MEMORY_PRUNE_FLOOR", 0)
    pruned, observer = _replay(bars, every_bar=True)
    assert pruned == reference
    assert observer.entity_memory_prunes > len(bars) // 2
    cores = observer._emitter._base_origin_core_event_ids
    for tracker in observer._zone_trackers.values():
        assert tracker.pending_base_origin_core_ids() <= cores.keys()


def test_a_namespaced_key_survives_on_its_bare_identity(unpruned) -> None:
    # The harvest holds the bare id of every namespaced citation; a memory
    # keyed ``swing:…`` must survive while the bare swing id is cited.
    _, observer = unpruned
    emitter = observer._emitter
    key = next(
        key for key in emitter._candidate_level_event_ids if key.startswith("swing:")
    )
    bare = key.removeprefix("swing:")
    dead = emitter.dead_entity_memory_keys(frozenset({bare}))
    assert key not in dead.get("_candidate_level_event_ids", ())
    assert key in emitter.dead_entity_memory_keys(frozenset())["_candidate_level_event_ids"]
    touch = next(
        value for value in emitter._known_level_touch_ids if value.startswith(key + "|")
    )
    assert touch not in dead.get("_known_level_touch_ids", ())
