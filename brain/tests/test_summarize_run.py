from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from brain.core.journal import BrainJournal, JournalReader
from brain.core.main_brain import MainBrain, MainBrainConfig
from brain.core.runtime import BrainRuntime
from brain.core.sleep_controller import ControllerConfig
from brain.scripts.summarize_run import cost_usd, direction_accuracy, load_pricing, main, missed_trends, render, sharp_move_coverage, summarize
from contract.market.primitives import Bar
from execution.core.order_fsm import OrderMachine
from execution.core.stack import TradingStack
from execution.tests.test_stack_e2e import RISK, LongAtTheNearestZone, executor
from risk.core.gate import RiskGate
from shares.core.eye_factory import build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ControllerConfig.from_json(ROOT / "brain" / "configs" / "sleep_controller.json")
CONFIG = MainBrainConfig.from_json(ROOT / "brain" / "configs" / "main_brain.json")
PRICING = load_pricing(ROOT / "brain" / "configs" / "llm_pricing.json")


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """A stack run on the synthetic tape: a journal with trades, and its bars."""
    run_dir = tmp_path_factory.mktemp("run")
    reader, observer = build_eye(ROOT / "configs" / "model.json", root=ROOT, audit_journal_dir=None)
    pairs = []
    for bar in session_bars(2):
        obs = observer.observe(reader.on_bar(bar))
        if obs.market_snapshot is not None:
            pairs.append((obs, bar))
    journal = BrainJournal(run_dir, run_id="summary")
    broker = executor()
    machine = OrderMachine(broker, RiskGate(RISK), journal=journal)
    runtime = BrainRuntime(
        controller=CONTROLLER,
        brain=MainBrain(client=LongAtTheNearestZone(), config=CONFIG, ledger=machine.ledger, sleep=lambda s: None),
        journal=journal, ledger=machine.ledger, tick=0.25,
    )
    stack = TradingStack(runtime, machine, tick=0.25)
    decisions = {}
    for obs, bar in pairs:
        result = stack.step(obs, bar)
        decisions[result.decision.value] = decisions.get(result.decision.value, 0) + 1
    journal.write_run({
        "run_id": "summary", "model": "scripted", "broker": "sim", "decisions": decisions, "llm_calls": runtime._brain._client.calls,
        "machine_stats": dict(machine.stats), "simulated_account": broker.account.summary(), "sim_equity": 1_000_000.0,
        "timings": {"llm": {"count": 1, "total_s": 0.1, "mean_ms": 100.0, "p50_ms": 100.0, "p95_ms": 100.0, "max_ms": 100.0}},
    })
    return run_dir, [bar for _, bar in pairs], broker, machine


def test_summary_counts_calls_orders_and_the_account(synthetic_run) -> None:
    run_dir, bars, broker, machine = synthetic_run
    summary = summarize(run_dir, pricing=PRICING, bars=bars)
    reader = JournalReader(run_dir)
    records = [r for ep in reader.episode_ids() for r in reader.records(ep)]
    assert summary["llm"]["calls"] == sum(1 for r in records if r.record == "llm_call")
    assert summary["controller"]["episodes"] == len(reader.episode_ids())
    assert summary["orders"]["submitted"] == machine.stats["submitted"] >= 1
    assert summary["orders"]["filled"] == machine.stats["filled"]
    assert summary["risk"]["vetoes"] == machine.stats["veto"]
    assert summary["invariants"] == {"double_entry": 0, "positions_over_limit": 0, "position_without_fill": 0, "stale_snapshots": 0}
    assert summary["account"]["cash"] == broker.account.cash and summary["account"]["realized_pnl"] == broker.account.cash - 1_000_000.0
    assert summary["coverage"]["sharp_moves"] >= 0 and summary["coverage"]["covered"] <= summary["coverage"]["sharp_moves"]
    assert summary["timings"]["llm"]["count"] == 1
    assert summary["llm"]["calls_by_trigger"]["WAKE"] == summary["controller"]["episodes"]


def test_cost_follows_the_published_rates() -> None:
    usage = {"prompt_cache_hit_tokens": 1_000_000, "prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 1_000_000}
    rates = PRICING["per_million_tokens"]["deepseek-flash"]
    assert cost_usd(usage, rates["peak"]) == pytest.approx(0.006 + 0.3 + 1.2)
    assert cost_usd(usage, rates["off_peak"]) == pytest.approx(0.753)


def test_veto_metrics_count_reproposals_and_reanalyses(tmp_path: Path) -> None:
    journal = BrainJournal(tmp_path, run_id="veto")
    journal.write_run({"run_id": "veto", "model": "deepseek:deepseek-flash@low", "broker": "sim"})
    t0 = pd.Timestamp("2022-01-03T14:31:00Z")
    ep = "EP_20220103_001"
    journal.open_episode(ep, t0)

    def veto(minute: int, signature: str, proposals: int) -> None:
        journal.write("trade", episode_id=ep, known_at=t0 + pd.Timedelta(minutes=minute), payload={
            "kind": "veto", "machine": "IDLE", "signature": signature, "plan": {"direction": "SHORT", "entry": {"alias": "FVG_5m_1"}},
            "verdict": {"passed": False, "vetoes": ["reward_risk"], "reasons": ["reward-to-risk 1.1 < 1.5"]},
            "account_asof": (t0 + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ"), "proposals_vetoed": proposals, "bars_vetoed": minute + 1,
        })

    def call(minute: int, *, feedback: bool, opportunity: dict, prior_opportunity: dict) -> None:
        known_at = t0 + pd.Timedelta(minutes=minute)
        execution = {"status": "IDLE", "order": None, "position": None, "last_outcome": None, "last_veto": None}
        if feedback:
            execution["last_veto"] = {"direction": "SHORT", "entry_object_id": "FVG_5m_1", "invalidation_object_id": "BSL_5m_1", "target_object_id": "SSL_1H_1", "vetoes": ["reward_risk"]}
        journal.write("llm_call", episode_id=ep, known_at=known_at, payload={
            "input_sha": f"sha{minute}", "input": {"trigger": {"kind": "UPDATE", "reasons": ["ev_x"]}, "known_at": known_at.strftime("%Y-%m-%dT%H:%M:%SZ"), "prior_state": {"opportunity": prior_opportunity, "watch_next": [], "execution": execution}},
            "reply": {"content": json.dumps({"understanding_holds": True, "opportunity": opportunity, "watch_next": [], "reasoning_confidence": "LOW"}), "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 10}, "latency_ms": 100, "model": "x", "reasoning_content": None, "finish_reason": "stop"},
            "attempts": 1, "repaired": False, "repair_reason": None, "rejected_reply": None,
        })

    same = {"state": "ACTIONABLE", "direction": "SHORT", "entry_object_id": "FVG_5m_1", "invalidation_object_id": "BSL_5m_1", "target_object_id": "SSL_1H_1"}
    other = {**same, "target_object_id": "SSL_4H_1"}
    none = {"state": "NONE", "direction": None, "entry_object_id": None, "invalidation_object_id": None, "target_object_id": None}
    call(0, feedback=False, opportunity=same, prior_opportunity=none)
    veto(0, "sig-a", 1)
    call(3, feedback=True, opportunity=same, prior_opportunity=same)  # kept it: re-proposed
    veto(3, "sig-a", 2)
    call(6, feedback=True, opportunity=other, prior_opportunity=same)  # re-analysed: a new plan
    veto(6, "sig-b", 1)
    call(9, feedback=True, opportunity=none, prior_opportunity=other)  # re-analysed: dropped
    for minute, reason in ((10, "thesis_closed"), (11, "stop_cooldown"), (12, "stop_cooldown")):
        journal.write("trade", episode_id=ep, known_at=t0 + pd.Timedelta(minutes=minute), payload={
            "kind": "thesis_refused", "machine": "IDLE", "signature": "sig-c", "thesis_id": "T1", "reason": reason, "plan": {}, "cooldown_bars_left": 3,
        })
    summary = summarize(tmp_path, pricing=PRICING, bars=None)
    risk = summary["risk"]
    assert risk["vetoes"] == 3 and risk["veto_signatures"] == 2 and risk["reproposals_after_veto"] == 1
    assert risk["veto_repeat_rate"] == pytest.approx(1 / 3, abs=1e-4) and risk["vetoes_by_code"] == {"reward_risk": 3}
    assert risk["thesis_refused"] == 3 and risk["refusals_by_reason"] == {"stop_cooldown": 2, "thesis_closed": 1}
    assert risk["daily_stop_vetoes"] == 0 and risk["halted"] is None and summary["orders"]["flattened"] == 0
    assert risk["calls_with_veto_feedback"] == 3 and risk["kept_after_veto"] == 1 and risk["reanalysed_after_veto"] == 2
    assert risk["stale_snapshots"] == 0 and summary["coverage"] is None
    assert summary["llm"]["cost_usd_peak"] == pytest.approx(4 * (10 * 0.3 + 5 * 1.2) / 1_000_000)


def test_sharp_move_coverage_counts_calls_near_each_move() -> None:
    from contract.market.primitives import Bar

    start = pd.Timestamp("2022-01-03T14:30:00Z")
    bars = []
    for i in range(80):
        base = 16000.0 + (60.0 if 40 <= i < 55 else 0.0) * ((i - 40) / 15)
        bars.append(Bar(start=start + pd.Timedelta(minutes=i), open=base, high=base + 1.0, low=base - 1.0, close=base, volume=1.0, symbol="NQ", instrument_id=1))
    quiet = sharp_move_coverage(bars, call_times=[])
    assert quiet["sharp_moves"] > 0 and quiet["covered"] == 0
    covered = sharp_move_coverage(bars, call_times=[start + pd.Timedelta(minutes=30)])
    assert covered["covered"] >= 1


def test_render_and_main_write_a_summary(synthetic_run, capsys) -> None:
    run_dir, bars, broker, machine = synthetic_run
    assert main(["--run-dir", str(run_dir), "--write", "--no-coverage"]) == 0
    written = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert written["orders"]["submitted"] == machine.stats["submitted"] and written["coverage"] is None
    text = render([written, written])
    assert "submitted" in text and "cost_usd_peak" in text and text.count(str(machine.stats["submitted"])) >= 2


def test_until_bounds_the_summary_to_a_common_window(synthetic_run) -> None:
    run_dir, bars, broker, machine = synthetic_run
    reader = JournalReader(run_dir)
    calls = sorted(r.known_at for ep in reader.episode_ids() for r in reader.records(ep) if r.record == "llm_call")
    cut = calls[len(calls) // 2]
    bounded = summarize(run_dir, pricing=PRICING, bars=None, until=cut)
    assert 0 < bounded["llm"]["calls"] == sum(1 for t in calls if t <= cut) < len(calls)
    assert bounded["account"] is None and bounded["timings"] is None and bounded["run"]["until"] == cut.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert main(["--run-dir", str(run_dir), "--no-coverage", "--until", cut.strftime("%Y-%m-%dT%H:%M:%SZ")]) == 0


def test_direction_accuracy_reads_the_close_an_hour_later() -> None:
    start = pd.Timestamp("2022-01-03T14:00:00Z")
    bars = [Bar(start=start + pd.Timedelta(minutes=i), open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.0 + i, volume=1.0, symbol="NQ", instrument_id=1) for i in range(130)]
    readings = [
        (pd.Timestamp("2022-01-03T14:10:00Z"), "LONG"),   # rising tape: right
        (pd.Timestamp("2022-01-03T14:20:00Z"), "SHORT"),  # wrong
        (pd.Timestamp("2022-01-03T15:50:00Z"), "LONG"),   # no bar an hour later: not counted
    ]
    assert direction_accuracy(readings, bars) == {"readings": 2, "agreed": 1, "accuracy": 0.5}
    assert direction_accuracy([], bars) == {"readings": 0, "agreed": 0, "accuracy": None}


def test_summary_carries_the_bias_section(synthetic_run) -> None:
    run_dir, bars, _, _ = synthetic_run
    summary = summarize(run_dir, pricing=PRICING, bars=bars)
    bias = summary["bias"]
    assert set(bias) >= {"changes", "neutral_revisions", "opportunities_against_bias", "direction_accuracy_60m"}
    assert bias["opportunities_against_bias"] == 0


def test_missed_trends_count_expiries_the_tape_ran_away_from() -> None:
    start = pd.Timestamp("2022-01-03T14:00:00Z")
    bars = [Bar(start=start + pd.Timedelta(minutes=i), open=16450.0 + i, high=16451.0 + i, low=16449.5 + i, close=16450.5 + i, volume=1.0, symbol="NQ", instrument_id=1) for i in range(40)]
    expiries = [
        {"direction": "LONG", "limit_price": 16449.25, "stop_price": 16439.5, "submitted_at": start + pd.Timedelta(minutes=1), "expired_at": start + pd.Timedelta(minutes=16)},   # never touched; from the 16451 open the tape ran +16 pts >= 1 R (9.75)
        {"direction": "SHORT", "limit_price": 16470.0, "stop_price": 16480.0, "submitted_at": start + pd.Timedelta(minutes=1), "expired_at": start + pd.Timedelta(minutes=16)},  # never touched either; the tape rose towards the limit, not away from it
    ]
    assert missed_trends(expiries, bars) == {"expired": 2, "missed": 1}
