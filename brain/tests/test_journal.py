from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from brain.core.journal import BrainJournal, JournalError, JournalReader
from brain.core.position_ledger import InMemoryPositionLedger, PositionRecord
from contract.brain.state import TradeDirection

T = pd.Timestamp("2022-01-04T15:00:00Z")


def test_chain_verifies_and_reader_reproduces_records(tmp_path: Path) -> None:
    j = BrainJournal(tmp_path, run_id="abc")
    j.write_run({"run_id": "abc", "model": "fake"})
    j.open_episode("EP_1", T)
    j.write("wake", episode_id="EP_1", known_at=T, payload={"reasons": ["ev_1"]})
    j.write("llm_call", episode_id="EP_1", known_at=T, payload={"input_sha": "s1", "reply": {"content": "{}", "reasoning_content": None, "usage": {}, "latency_ms": 1, "model": "fake"}})
    j.write("state", episode_id="EP_1", known_at=T, payload={"revision": 0, "state": {}})
    j.write("state", episode_id="EP_1", known_at=T + pd.Timedelta(minutes=1), payload={"revision": 1, "state": {}})
    j.close_episode("EP_1", T + pd.Timedelta(minutes=2), payload={"revisions": 2})
    r = JournalReader(tmp_path)
    assert r.run()["model"] == "fake" and r.run_id() == "abc" and r.episode_ids() == ("EP_1",)
    recs = r.records("EP_1")
    assert [x.record for x in recs] == ["episode_opened", "wake", "llm_call", "state", "state", "sleep"]
    assert recs[0].prev_hash == "abc" and recs[1].prev_hash == recs[0].hash
    assert [x.seq for x in recs] == list(range(6))
    r.verify_chain("EP_1")
    r.verify_chain("EP_1", run_id="abc")
    with pytest.raises(JournalError):
        r.verify_chain("EP_1", run_id="other")
    assert "s1" in r.recorded_replies() and r.recorded_replies()["s1"].content == "{}"
    index = r.index()
    assert index[0]["episode_id"] == "EP_1" and index[0]["revisions"] == 2 and index[0]["records"] == 6
    raw = json.loads((tmp_path / "episodes" / "EP_1.jsonl").read_text().splitlines()[3])
    assert raw["payload"]["revision"] == 0 and raw["known_at"] == "2022-01-04T15:00:00Z"


def test_tampering_breaks_verification(tmp_path: Path) -> None:
    j = BrainJournal(tmp_path, run_id="abc")
    j.open_episode("EP_1", T)
    j.write("tick", episode_id="EP_1", known_at=T, payload={"revision": 0, "n": 1})
    path = tmp_path / "episodes" / "EP_1.jsonl"
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace('"n":1', '"n":2')
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(JournalError, match="hash"):
        JournalReader(tmp_path).verify_chain("EP_1")


def test_non_monotone_known_at_and_revision_gap_refused(tmp_path: Path) -> None:
    j = BrainJournal(tmp_path, run_id="abc")
    j.open_episode("EP_1", T)
    with pytest.raises(JournalError, match="precedes"):
        j.write("tick", episode_id="EP_1", known_at=T - pd.Timedelta(minutes=1), payload={"revision": 0})
    j.write("state", episode_id="EP_1", known_at=T, payload={"revision": 0})
    with pytest.raises(JournalError, match="revision"):
        j.write("state", episode_id="EP_1", known_at=T, payload={"revision": 2})
    j.write("tick", episode_id="EP_1", known_at=T, payload={"revision": 1})
    with pytest.raises(JournalError, match="revision"):
        j.write("tick", episode_id="EP_1", known_at=T, payload={"revision": 1})
    with pytest.raises(JournalError):
        j.write("bogus", episode_id="EP_1", known_at=T, payload={})
    with pytest.raises(JournalError):
        j.write("sleep", episode_id="EP_1", known_at=T, payload={})
    with pytest.raises(JournalError, match="not open"):
        j.write("tick", episode_id="EP_2", known_at=T, payload={"revision": 0})
    with pytest.raises(JournalError, match="already exists"):
        j.open_episode("EP_1", T)
    j.close_episode("EP_1", T, payload={})
    with pytest.raises(JournalError, match="closed"):
        j.write("tick", episode_id="EP_1", known_at=T, payload={"revision": 2})


def test_ledger() -> None:
    ledger = InMemoryPositionLedger()
    assert not ledger.has_open_position()
    ledger.open(PositionRecord("p1", TradeDirection.LONG, T, "FVG_5m_3"))
    assert ledger.has_open_position() and ledger.open_positions()[0].position_id == "p1"
    with pytest.raises(ValueError):
        ledger.open(PositionRecord("p1", TradeDirection.LONG, T, "FVG_5m_3"))
    ledger.close("p1")
    assert not ledger.has_open_position()
    with pytest.raises(ValueError):
        ledger.close("p1")
