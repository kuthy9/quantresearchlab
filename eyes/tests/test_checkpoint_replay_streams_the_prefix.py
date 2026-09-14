"""Checkpoint restore verifies the committed prefix in bounded chunks.

Restoring a reducer or publisher checkpoint cold-replays the committed prefix
to prove the compact state is exactly reproducible.  It did so by taking
``events_since(0)`` as one tuple and building a second full store from it, so
a restore held the whole journal twice and could only ever be done by a
process that already had every event in memory.  The replay now streams the
prefix through the store in bounded chunks -- the verifier store is fed one
chunk at a time and the verifier reducer consumes each -- so the check
depends on the store's journal API alone, never on an in-memory list.
"""
from __future__ import annotations

import copy
import pickle

import pytest

import eyes.core.market_state as market_state
from eyes.core.causal import CausalMarketReader
from eyes.core.event_store import EventStore
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

CHUNK = 512


def _replayed_publisher():
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    for bar in _noisy(session_bars(1)[:300], seed=13):
        observer.observe(reader.on_bar(bar))
    return observer


def test_restore_streams_the_prefix_in_chunks_and_never_materializes_it(
    monkeypatch,
) -> None:
    observer = _replayed_publisher()
    publisher = observer.market_snapshot_publisher
    store = observer.audit_store
    cursor = publisher._event_reducer.cursor
    assert cursor > 3 * CHUNK
    monkeypatch.setattr(market_state, "CHECKPOINT_REPLAY_CHUNK_EVENTS", CHUNK)

    original_since = EventStore.events_since
    original_from_events = EventStore.from_events.__func__
    chunks: list[int] = []
    original_iter_chunks = EventStore.iter_chunks

    def guarded_since(self, index):
        if index == 0 and len(self) > CHUNK:
            raise AssertionError("checkpoint restore materialized the whole prefix")
        return original_since(self, index)

    def guarded_from_events(cls, events, **kwargs):
        events = tuple(events)
        if len(events) > CHUNK:
            raise AssertionError("checkpoint restore rebuilt a full store at once")
        return original_from_events(cls, events, **kwargs)

    def spying_iter_chunks(self, start, stop, size):
        for chunk in original_iter_chunks(self, start, stop, size):
            chunks.append(len(chunk))
            yield chunk

    monkeypatch.setattr(EventStore, "events_since", guarded_since)
    monkeypatch.setattr(EventStore, "from_events", classmethod(guarded_from_events))
    monkeypatch.setattr(EventStore, "iter_chunks", spying_iter_chunks)

    restored = pickle.loads(pickle.dumps(publisher, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored._event_reducer.states == publisher._event_reducer.states
    assert restored._event_reducer.cursor == cursor
    assert chunks, "the restore did not stream the prefix"
    assert max(chunks) <= CHUNK
    # The reducer verifies itself on unpickle and the publisher verifies it
    # again; each pass streams exactly the consumed prefix.
    assert sum(chunks) % cursor == 0


def test_a_tampered_compact_state_still_fails_replay(monkeypatch) -> None:
    observer = _replayed_publisher()
    publisher = observer.market_snapshot_publisher
    monkeypatch.setattr(market_state, "CHECKPOINT_REPLAY_CHUNK_EVENTS", CHUNK)
    tampered = copy.deepcopy(publisher)
    timeframe, state = next(iter(tampered._event_reducer.states.items()))
    tampered._event_reducer.states[timeframe] = market_state.replace(
        state, structural_legs=()
    ) if state.structural_legs else market_state.replace(
        state, swing_hierarchy=state.swing_hierarchy[:-1]
    )
    with pytest.raises(ValueError, match="failed replay"):
        pickle.loads(pickle.dumps(tampered, protocol=pickle.HIGHEST_PROTOCOL))


def test_an_observer_checkpoint_taken_after_settled_projections_restores() -> None:
    """The publisher persists settled geometry and candidate projections.

    Those values depend on when the publisher projected them, so the cold
    fold could never reproduce them and every restore taken after the first
    settled Swing failed with "compact checkpoint failed replay" -- on the
    tree as far back as a52241f.  Verification now compares the fold-owned
    part of each state; the projections are carried by the checkpoint.
    """
    observer = _replayed_publisher()
    restored = pickle.loads(pickle.dumps(observer, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored.market_snapshot_publisher._event_reducer.states == (
        observer.market_snapshot_publisher._event_reducer.states
    )
    assert restored.audit_store.fingerprint() == observer.audit_store.fingerprint()
