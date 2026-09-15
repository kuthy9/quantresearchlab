"""Committed events older than the hot window live on disk, not in memory.

The audit store held every event object of the run in three maps -- about
2 KB per event in the heap, 26 MB per 1,000 bars, a gigabyte per month of
1m replay.  The hot path needs only the digest of every event (idempotent
retries, conflicts, the prefix fingerprint), the recent events by id and by
index, and a way to reach an older parent when a lifecycle event cites one.
An observer configured with an audit journal directory therefore spills
events below the reducer cursor and older than ``audit_hot_window_minutes``
to an append-only journal file; the store still answers ``get``, ``iter_events``
and ``events_since`` for them, its fingerprint is unchanged, and a checkpoint
restores from the journal -- or refuses to, if the journal was altered.
"""
from __future__ import annotations

import pickle

import pytest

from eyes.core.causal import CausalMarketReader
from eyes.core.event_store import EventStore
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars

HOT_MINUTES = 90


def _observer(journal_dir=None) -> tuple[CausalMarketReader, CausalObserver]:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
            audit_journal_dir=None if journal_dir is None else str(journal_dir),
            audit_hot_window_minutes=HOT_MINUTES,
        )
    )
    return reader, observer


def _replay(observer, reader, bars) -> None:
    for bar in bars:
        observer.observe(reader.on_bar(bar))


def test_spilled_events_leave_the_heap_but_stay_addressable(tmp_path) -> None:
    bars = _noisy(session_bars(1)[:400], seed=17)
    reader, journaled = _observer(tmp_path)
    _replay(journaled, reader, bars)
    reader, resident = _observer()
    _replay(resident, reader, bars)

    cold, hot = journaled.audit_store, resident.audit_store
    assert len(cold) == len(hot) > 3000
    assert cold.fingerprint() == hot.fingerprint()
    # The hot list holds the window plus the reducer's unconsumed suffix,
    # never the whole journal.
    assert cold.cold_count > len(cold) // 2, (cold.cold_count, len(cold))
    assert len(cold._events) == len(cold) - cold.cold_count
    assert len(cold._by_id) == len(cold._events)
    journal_files = list(tmp_path.glob("*.evlog"))
    assert len(journal_files) == 1 and journal_files[0].stat().st_size > 0

    oldest = hot.events_since(0)[0]
    assert oldest.event_id not in cold._by_id
    assert cold.get(oldest.event_id) == oldest
    assert cold.event_digest(oldest.event_id) == hot.event_digest(oldest.event_id)
    assert tuple(cold.iter_events(0, len(cold))) == hot.events_since(0)
    assert cold.events_since(0) == hot.events_since(0)
    assert cold.prefix_fingerprint(cold.cold_count) == hot.prefix_fingerprint(cold.cold_count)
    assert journaled.last_market_snapshot.timeframe_states == (
        resident.last_market_snapshot.timeframe_states
    )


def test_a_checkpoint_restores_from_the_journal_and_refuses_a_tampered_one(tmp_path) -> None:
    bars = _noisy(session_bars(1)[:400], seed=17)
    reader, observer = _observer(tmp_path)
    _replay(observer, reader, bars[:300])
    frozen = pickle.dumps(observer, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(frozen)
    assert restored.audit_store.fingerprint() == observer.audit_store.fingerprint()
    assert restored.audit_store.cold_count == observer.audit_store.cold_count
    assert len(restored.audit_store._events) == len(observer.audit_store._events)
    # Both continue identically from the checkpoint.
    reader_a = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    reader_b = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    for bar in bars[:300]:
        reader_a.on_bar(bar)
        reader_b.on_bar(bar)
    for bar in bars[300:]:
        observer.observe(reader_a.on_bar(bar))
        restored.observe(reader_b.on_bar(bar))
    assert restored.audit_store.fingerprint() == observer.audit_store.fingerprint()

    # Flip one byte inside a record the checkpoint committed to (the file
    # has grown past the checkpoint since; those records are not its).
    checkpointed = observer.audit_store
    journal = next(tmp_path.glob("*.evlog"))
    payload = bytearray(journal.read_bytes())
    target = checkpointed._cold_offsets[restored.audit_store.cold_count // 2] + 8
    payload[target] ^= 0xFF
    journal.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="journal"):
        pickle.loads(frozen)


def test_a_store_without_a_journal_cannot_spill(tmp_path) -> None:
    store = EventStore()
    with pytest.raises(ValueError, match="journal"):
        store.spill(before_index=0)
