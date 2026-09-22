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
    """Leaves every verdict NEUTRAL and asks to sleep from the third call on."""

    def complete(self, *, system: str, user: str):
        reply = super().complete(system=system, user=user)
        payload = json.loads(reply.content)
        for verdict in payload["evidence_verdicts"]:
            verdict["verdict"] = "NEUTRAL"
        payload["continue_active"] = self.calls < 3
        return LLMReply(json.dumps(payload), None, {}, 1, "neutral-sleeper")


def test_no_llm_call_without_evidence_and_a_neutral_sleeper_sleeps(observations) -> None:
    client = NeutralSleeper()
    results, _ = run(observations, client)
    ticks = [r for r in results if r.decision is Decision.TICK]
    assert ticks and all(not r.llm_called for r in ticks)
    updates = [r for r in results if r.decision is Decision.UPDATE]
    assert updates and all(r.llm_called for r in updates)
    # NEUTRAL is a verdict, not a pending question: it never refuses sleep.
    assert not any("sleep_refused:unresolved_evidence" in r.rejections for r in results)
    assert any(r.slept for r in results)


def test_evidence_parked_by_an_incident_is_verdicted_later_and_sleep_follows(observations) -> None:
    results, _ = run(observations, EchoClient(fail_calls=range(2, 6), sleep_after=6))
    incidents = [i for i, r in enumerate(results) if r.incident]
    assert incidents, "the scripted timeouts produced no incident"
    parked = results[incidents[0]]
    assert parked.status_after is RuntimeStatus.ACTIVE
    slept = [i for i, r in enumerate(results) if r.slept and r.episode_id == parked.episode_id]
    assert slept and slept[0] > incidents[0], "the episode that parked evidence never slept"
    assert not any("sleep_refused:unresolved_evidence" in r.rejections for r in results[slept[0]:])


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


def test_idle_episode_is_archived_after_the_configured_updates(observations, tmp_path: Path) -> None:
    journal = BrainJournal(tmp_path, run_id="idle")
    results, _ = run(observations, EchoClient(), journal=journal)  # never asks to sleep, never proposes
    slept = [r for r in results if r.slept]
    assert slept, "an idle episode never archived"
    first = slept[0]
    updates_before = [r for r in results[: results.index(first) + 1] if r.episode_id == first.episode_id and r.decision is Decision.UPDATE]
    assert len(updates_before) >= CONTROLLER.idle_archive_after_updates
    reader = JournalReader(tmp_path)
    sleeps = [r for ep in reader.episode_ids() for r in reader.records(ep) if r.record == "sleep"]
    assert sleeps and sleeps[0].payload["reason"] == "idle"


def test_bookkeeping_evidence_is_deferred_to_the_next_llm_call(observations) -> None:
    inputs = []
    ledger = InMemoryPositionLedger()
    runtime = BrainRuntime(
        controller=CONTROLLER,
        brain=MainBrain(client=EchoClient(), config=CONFIG, ledger=ledger, sleep=lambda s: None),
        journal=None, ledger=ledger, tick=0.25, on_step=lambda r, s, i: inputs.append(i) if i is not None else None,
    )
    for obs in observations:
        runtime.step(obs)
    late = [
        (i, e) for i in inputs for e in i.to_dict()["new_evidence"]
        if e["pending_since"] is None and e["known_at"] < i.to_dict()["known_at"]
    ]
    assert late, "no call carried evidence from an earlier TICK bar"
    assert all(e["kind"] in CONTROLLER.bookkeeping_kinds for _, e in late)


def test_step_result_reports_the_llm_latency_and_timings_cover_each_phase(observations, tmp_path: Path) -> None:
    from shares.core.timing import Timings

    timings = Timings()
    ledger = InMemoryPositionLedger()
    runtime = BrainRuntime(
        controller=CONTROLLER,
        brain=MainBrain(client=EchoClient(), config=CONFIG, ledger=ledger, sleep=lambda s: None, timings=timings),
        journal=BrainJournal(tmp_path, run_id="t"), ledger=ledger, tick=0.25, timings=timings,
    )
    results = [runtime.step(obs) for obs in observations]
    called = [r for r in results if r.llm_called]
    assert called and all(r.llm_latency_ms == 1 for r in called), "EchoClient replies with latency 1 ms"
    assert all(r.llm_latency_ms is None for r in results if not r.llm_called)
    summary = timings.summary()
    assert {"controller", "input", "llm", "reduce", "journal"} <= set(summary)
    assert summary["controller"]["count"] == len(results) and summary["llm"]["count"] == len(called)


def test_a_watched_5m_objects_flip_is_not_an_update_but_a_15m_ones_is() -> None:
    from types import SimpleNamespace

    from brain.tests.test_brain_state import make_state
    from contract.brain.state import RegisteredObject, WatchItem

    runtime = BrainRuntime(
        controller=CONTROLLER, brain=MainBrain(client=EchoClient(), config=CONFIG, ledger=InMemoryPositionLedger(), sleep=lambda s: None),
        journal=None, ledger=InMemoryPositionLedger(), tick=0.25,
    )
    registry = {"FVG_5m_3": RegisteredObject("a" * 24, "fvg", "5m"), "BSL_15m_1": RegisteredObject("b" * 24, "bsl", "15m")}
    state = make_state(watch_next=(WatchItem("FVG_5m_3", "?"), WatchItem("BSL_15m_1", "?")), destination_candidates=(), object_registry=registry)
    assert runtime._watched(state) == ("BSL_15m_1",)
    t0 = pd.Timestamp("2022-01-04T15:00:00Z")
    before = SimpleNamespace(relation_of=lambda alias: "above_price", known_at=t0)
    after = SimpleNamespace(relation_of=lambda alias: "below_price", known_at=t0 + pd.Timedelta(minutes=1))
    runtime._remember_relations(state, before)
    assert runtime._relation_changes(after) == ("BSL_15m_1",), "the 5m object's flip is not remembered, so it cannot trigger"


def test_a_relation_flip_of_the_same_alias_triggers_once_per_debounce_window() -> None:
    from types import SimpleNamespace

    from brain.tests.test_brain_state import make_state
    from contract.brain.state import RegisteredObject, WatchItem

    runtime = BrainRuntime(
        controller=CONTROLLER, brain=MainBrain(client=EchoClient(), config=CONFIG, ledger=InMemoryPositionLedger(), sleep=lambda s: None),
        journal=None, ledger=InMemoryPositionLedger(), tick=0.25,
    )
    registry = {"BSL_15m_1": RegisteredObject("b" * 24, "bsl", "15m")}
    state = make_state(watch_next=(WatchItem("BSL_15m_1", "?"),), destination_candidates=(), object_registry=registry)
    t0 = pd.Timestamp("2022-01-04T15:00:00Z")
    above = lambda at: SimpleNamespace(relation_of=lambda alias: "above_price", known_at=at)
    below = lambda at: SimpleNamespace(relation_of=lambda alias: "below_price", known_at=at)
    runtime._remember_relations(state, above(t0))
    first = t0 + pd.Timedelta(minutes=1)
    assert runtime._relation_changes(below(first)) == ("BSL_15m_1",)
    runtime._relation_triggered_at(("BSL_15m_1",), first)
    runtime._remember_relations(state, below(first))
    assert runtime._relation_changes(above(t0 + pd.Timedelta(minutes=5))) == (), "a second flip within the window does not trigger"
    assert runtime._relation_changes(above(t0 + pd.Timedelta(minutes=16))) == ("BSL_15m_1",)


def test_an_event_window_archives_the_episode_flat_and_wakes_it_fresh_when_the_window_ends(observations, tmp_path: Path) -> None:
    # 2026-09-21: a scheduled release puts an active episode to sleep without a call; the first bar after the window wakes a new one
    import dataclasses

    from brain.core.event_calendar import CalendarEvent, EventFilter, EventRule

    plain, _ = run(observations, EchoClient())
    active = [i for i, r in enumerate(plain) if r.status_after is RuntimeStatus.ACTIVE and i + 200 < len(plain)]
    assert active, "the echo run never stays active with 200 bars to spare"
    k = active[0] + 1  # an active episode on the bar before the window opens
    start = pd.Timestamp(observations[k].asof).tz_convert("UTC")
    release = start + pd.Timedelta(minutes=60)
    events = EventFilter.from_events(
        [CalendarEvent("cpi", "Consumer Price Index", release, ("BLS",))], (EventRule("CPI", "Consumer Price Index", 60, 30),), sha256="test",
    )
    window = events.events[0]
    controller = dataclasses.replace(CONTROLLER, events=events)
    journal = BrainJournal(tmp_path, run_id="event")
    ledger = InMemoryPositionLedger()
    runtime = BrainRuntime(
        controller=controller, brain=MainBrain(client=EchoClient(), config=CONFIG, ledger=ledger, sleep=lambda s: None),
        journal=journal, ledger=ledger, tick=0.25,
    )
    results = [runtime.step(obs) for obs in observations]
    slept = results[k]
    assert slept.decision is Decision.EVENT_SLEEP and slept.slept and slept.status_after is RuntimeStatus.SLEEP and not slept.llm_called
    assert slept.event == f"event:CPI:{release.strftime('%Y-%m-%dT%H:%M:%SZ')}" and slept.episode_id == plain[k - 1].episode_id
    inside = [r for r in results if window.start <= r.known_at < window.end]
    assert inside and all(r.decision is Decision.STAY_ASLEEP for r in inside[1:]) and all(r.status_after is RuntimeStatus.SLEEP for r in inside)
    woken = next(r for r in results if r.known_at >= window.end)
    assert woken.decision is Decision.WAKE and woken.llm_called and woken.episode_id != slept.episode_id and woken.revision == 0
    reader = JournalReader(tmp_path)
    sleeps = {ep: r.payload for ep in reader.episode_ids() for r in reader.records(ep) if r.record == "sleep"}
    assert sleeps[slept.episode_id]["reason"] == slept.event
    wake = next(r.payload for r in reader.records(woken.episode_id) if r.record == "wake")
    assert list(wake["reasons"]) == [f"event_ended:CPI:{release.strftime('%Y-%m-%dT%H:%M:%SZ')}"]
    for ep in reader.episode_ids():
        reader.verify_chain(ep, run_id="event")
    assert CONTROLLER.events.active(start) is None, "the repository calendar has no release on the synthetic tape's dates"
