from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from brain.core.journal import BrainJournal, JournalReader
from brain.core.llm_client import EchoClient, LLMReply
from brain.core.main_brain import MainBrain, MainBrainConfig
from brain.core.position_ledger import InMemoryPositionLedger, PositionRecord
from brain.core.runtime import BrainRuntime, RuntimeStatus
from brain.core.sleep_controller import ControllerConfig, Decision
from contract.brain.state import TradeDirection
from shares.core.eye_factory import build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ControllerConfig.from_json(ROOT / "brain" / "configs" / "sleep_controller.json")
CONFIG = MainBrainConfig.from_json(ROOT / "brain" / "configs" / "main_brain.json")


@pytest.fixture(scope="module")
def observations():
    reader, observer = build_eye(ROOT / "configs" / "model.json", root=ROOT, audit_journal_dir=None)
    return [o for o in (observer.observe(reader.on_bar(b)) for b in session_bars(2)) if o.market_snapshot is not None]


def run(observations, client, journal=None, ledger=None):
    ledger = ledger or InMemoryPositionLedger()
    runtime = BrainRuntime(
        controller=CONTROLLER,
        brain=MainBrain(client=client, config=CONFIG, ledger=ledger, sleep=lambda s: None),
        journal=journal, ledger=ledger, tick=0.25,
    )
    return [runtime.step(obs) for obs in observations], runtime


def test_sleep_wake_update_tick_sleep_cycle(observations, tmp_path: Path) -> None:
    journal = BrainJournal(tmp_path, run_id="test")
    results, runtime = run(observations, EchoClient(sleep_after=3), journal=journal)
    decisions = [r.decision for r in results]
    assert Decision.WAKE in decisions and Decision.UPDATE in decisions and Decision.TICK in decisions
    first_wake = decisions.index(Decision.WAKE)
    assert all(r.decision is Decision.STAY_ASLEEP for r in results[:first_wake])
    assert results[first_wake].revision == 0 and results[first_wake].llm_called
    slept = [r for r in results if r.slept]
    assert slept and slept[0].status_after is RuntimeStatus.SLEEP
    reader = JournalReader(tmp_path)
    assert reader.episode_ids()
    for ep in reader.episode_ids():
        reader.verify_chain(ep, run_id="test")
        kinds = [r.record for r in reader.records(ep)]
        assert kinds[:4] == ["episode_opened", "wake", "llm_call", "state"]
        if kinds[-1] == "sleep":
            # either a full cycle, or a wake whose first reasoning already met the exit conditions
            assert ("tick" in kinds or "llm_call" in kinds[4:]) or kinds == ["episode_opened", "wake", "llm_call", "state", "sleep"]
    cycles = [
        [r.record for r in reader.records(ep)] for ep in reader.episode_ids()
    ]
    assert any("tick" in kinds and "sleep" in kinds for kinds in cycles), "no episode ran WAKE → UPDATE/TICK → sleep"
    assert reader.episode_ids()[0].startswith("EP_2025")
    assert reader.index() and reader.index()[0]["revisions"] >= 1
    # revisions inside an episode are contiguous across state and tick records
    for ep in reader.episode_ids():
        revisions = [r.payload["revision"] for r in reader.records(ep) if r.record in ("state", "tick")]
        assert revisions == list(range(len(revisions)))


class NeutralSleeper(EchoClient):
    """Asks to sleep on every call while leaving every verdict NEUTRAL."""

    def complete(self, *, system: str, user: str):
        reply = super().complete(system=system, user=user)
        payload = json.loads(reply.content)
        payload["continue_active"] = False
        return LLMReply(json.dumps(payload), None, {}, 1, "neutral-sleeper")


def test_no_llm_call_without_evidence_and_unresolved_blocks_sleep(observations) -> None:
    client = NeutralSleeper()
    results, _ = run(observations, client)
    ticks = [r for r in results if r.decision is Decision.TICK]
    assert ticks and all(not r.llm_called for r in ticks)
    updates = [r for r in results if r.decision is Decision.UPDATE]
    assert updates and all(r.llm_called for r in updates)
    refused = [r for r in results if "sleep_refused:unresolved_evidence" in r.rejections]
    assert refused


def test_incident_keeps_active_and_is_journaled_with_its_message(observations, tmp_path: Path) -> None:
    journal = BrainJournal(tmp_path, run_id="incident")
    results, _ = run(observations, EchoClient(fail_calls=range(2, 7)), journal=journal)
    incidents = [r for r in results if r.incident]
    assert incidents and incidents[0].incident == "LLMTimeout"
    assert incidents[0].status_after is RuntimeStatus.ACTIVE and incidents[0].llm_called
    reader = JournalReader(tmp_path)
    recorded = [r for ep in reader.episode_ids() for r in reader.records(ep) if r.record == "incident"]
    assert recorded and recorded[0].payload["kind"] == "LLMTimeout"
    assert recorded[0].payload["message"] == "echo client scripted timeout"


def test_open_position_forces_active(observations) -> None:
    ledger = InMemoryPositionLedger()
    ledger.open(PositionRecord("p1", TradeDirection.LONG, pd.Timestamp("2025-01-05T23:00:00Z"), "FVG_5m_1"))
    results, runtime = run(observations, EchoClient(sleep_after=1), ledger=ledger)
    assert not any(r.slept for r in results) and runtime.status is RuntimeStatus.ACTIVE
    assert any("sleep_refused:open_position" in r.rejections for r in results)


def test_known_at_must_increase(observations) -> None:
    _, runtime = run(observations[:200], EchoClient())
    with pytest.raises(ValueError, match="known_at"):
        runtime.step(observations[0])


def test_episode_ids_number_per_day_and_hook_sees_every_step(observations) -> None:
    seen = []
    ledger = InMemoryPositionLedger()
    runtime = BrainRuntime(
        controller=CONTROLLER,
        brain=MainBrain(client=EchoClient(sleep_after=2), config=CONFIG, ledger=ledger, sleep=lambda s: None),
        journal=None, ledger=ledger, tick=0.25, on_step=lambda r, s, i: seen.append((r, s, i)),
    )
    results = [runtime.step(obs) for obs in observations]
    assert len(seen) == len(results)
    episodes = [r.episode_id for r in results if r.decision is Decision.WAKE]
    assert len(episodes) == len(set(episodes)) and len(episodes) >= 2
    day_numbers = {}
    for ep in episodes:
        _, day, number = ep.split("_")
        day_numbers.setdefault(day, []).append(int(number))
    for numbers in day_numbers.values():
        assert numbers == list(range(1, len(numbers) + 1))
    wake_inputs = [i for r, s, i in seen if r.decision is Decision.WAKE]
    assert wake_inputs and wake_inputs[0].to_dict()["prior_state"] is None
