from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from brain.core.journal import BrainJournal, JournalReader
from contract.brain.state import TradeDirection
from contract.decision import OpportunityGeometry
from contract.execution import BrokerEvent, OrderStatus
from contract.market.primitives import Bar
from contract.risk import ObjectRef, TradePlan
from brain.core.position_ledger import IDLE_VIEW
from execution.core.order_fsm import MachineState, OrderMachine
from execution.core.simulated_executor import SimulatedExecutor, SimulatorConfig
from risk.core.gate import RiskConfig, RiskGate

ROOT = Path(__file__).resolve().parents[2]
# The lifecycle tests size at the v1 budget (0.5 %) with no leverage cap so
# one contract is one contract; sizing is the gate's tests' business.
CONFIG = replace(RiskConfig.from_json(ROOT / "risk" / "configs" / "risk.json"), risk_fraction={"BASE": 0.005, "A_PLUS": 0.005}, max_leverage=1_000.0)
SIM = SimulatorConfig.from_json(ROOT / "execution" / "configs" / "simulated_executor.json")
T0 = pd.Timestamp("2022-01-03T14:12:00Z")
EP = "EP_20220103_001"


def at(minute: int) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minute)


def bar(minute: int, low: float, high: float) -> Bar:
    return Bar(start=at(minute) - pd.Timedelta(minutes=1), open=(low + high) / 2, high=high, low=low, close=(low + high) / 2, volume=1.0, symbol="NQ", instrument_id=1)


def short_plan(entry=16387.5, stop=16411.5, target=16330.0, target_id="swing:c", thesis_id="", direction=TradeDirection.SHORT, **fields) -> TradePlan:
    """A SHORT with a 24-point stop and a 57.5-point target (2.4 R); ``fields`` reach the plan (mode, level, ...)."""
    return TradePlan(
        EP, 12, T0, direction,
        ObjectRef("FVG_5m_10", "fvg:a", "fvg", "5m"), ObjectRef("BSL_5m_3", "swing:b", "bsl", "5m"), ObjectRef("SSL_4H_2", target_id, "ssl", "4H"),
        OpportunityGeometry(entry, stop, target, abs(target - entry) / abs(entry - stop), ("e", "s", "t")), close=16390.0,
        thesis_id=thesis_id, governing_timeframe="15m", **fields,
    )


def last_trade(journal_dir: Path, kind: str | None = None) -> dict:
    records = [r.payload for r in JournalReader(journal_dir).records(EP) if r.record == "trade" and (kind is None or r.payload["kind"] == kind)]
    return records[-1]


def machine(tmp_path: Path, equity: float = 100_000.0):
    journal = BrainJournal(tmp_path, run_id="fsm")
    journal.open_episode(EP, T0)
    broker = executor(equity=equity)
    return OrderMachine(broker, RiskGate(CONFIG), journal=journal), broker, journal


def executor(equity: float = 100_000.0, **overrides) -> SimulatedExecutor:
    config = SIM if not overrides else SimulatorConfig(**{**SIM.__dict__, **overrides})
    return SimulatedExecutor(config, tick_size=0.25, point_value=20.0, equity=equity)


def quiet_bars(machine_, plan, start: int, count: int, low=16370.0, high=16380.0, visible=lambda alias: True):
    kinds = []
    for minute in range(start, start + count):
        kinds.extend(machine_.on_bar(at(minute), bar(minute, low, high), plan, episode_id=EP, visible=visible))
    return tuple(kinds)


def test_submits_once_and_a_re_signal_is_a_no_op(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    kinds = m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True)
    assert kinds == ("submitted",) and m.state is MachineState.WORKING and m.ledger.has_working_order()
    kinds = quiet_bars(m, short_plan(), 2, 3)
    assert "submitted" not in kinds and len(broker.snapshot(at(4)).open_entry_orders()) == 1
    records = [r for r in JournalReader(tmp_path).records(EP) if r.record == "trade"]
    assert [r.payload["kind"] for r in records] == ["submitted", "working"]
    submitted = records[0].payload
    assert submitted["signature"] == short_plan().signature and submitted["verdict"]["quantity"] == 1
    assert submitted["plan"]["entry"]["entity_id"] == "fvg:a"


def test_fill_opens_a_position_and_the_stop_closes_it(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True)
    kinds = m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(), episode_id=EP, visible=lambda a: True)
    assert "filled" in kinds and "position_opened" in kinds and m.state is MachineState.IN_POSITION
    assert m.ledger.has_open_position() and not m.ledger.has_working_order()
    assert m.ledger.open_positions()[0].direction is TradeDirection.SHORT and m.ledger.open_positions()[0].entry_object_id == "FVG_5m_10"
    # the Brain drops the opportunity: a position is not an order, nothing is cancelled
    kinds = m.on_bar(at(3), bar(3, 16380.0, 16390.0), None, episode_id=EP, visible=lambda a: True)
    assert kinds == () and m.state is MachineState.IN_POSITION
    kinds = m.on_bar(at(4), bar(4, 16400.0, 16420.0), None, episode_id=EP, visible=lambda a: True)
    assert "position_closed" in kinds and m.state is MachineState.IDLE and not m.ledger.has_open_position()
    closed = [r for r in JournalReader(tmp_path).records(EP) if r.record == "trade" and r.payload["kind"] == "position_closed"][0]
    assert closed.payload["exit_role"] == "stop" and closed.payload["exit_price"] == 16411.5
    JournalReader(tmp_path).verify_chain(EP, run_id="fsm")


def test_ttl_cancels_and_blocks_the_same_signature_until_it_changes(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    kinds = quiet_bars(m, short_plan(), 1, CONFIG.order_ttl_bars + 2)
    assert "cancel_requested" in kinds and "expired" in kinds and m.state is MachineState.IDLE
    kinds = quiet_bars(m, short_plan(), CONFIG.order_ttl_bars + 3, 3)
    assert "submitted" not in kinds, "an expired signature is not resubmitted while the plan is unchanged"
    kinds = quiet_bars(m, short_plan(target_id="swing:d"), CONFIG.order_ttl_bars + 6, 1)
    assert kinds == ("submitted",)


def test_a_changed_signature_cancels_then_submits_the_new_intent(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    quiet_bars(m, short_plan(), 1, 1)
    kinds = quiet_bars(m, short_plan(target_id="swing:d"), 2, 1)
    assert kinds == ("working", "cancel_requested") and m.state is MachineState.WORKING
    kinds = quiet_bars(m, short_plan(target_id="swing:d"), 3, 1)
    assert kinds == ("cancelled", "submitted")
    assert len(broker.snapshot(at(3)).open_entry_orders()) == 1 and broker.snapshot(at(3)).cancelled


def test_an_invisible_entry_object_cancels(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    quiet_bars(m, short_plan(), 1, 1)
    kinds = quiet_bars(m, short_plan(), 2, 2, visible=lambda alias: alias != "FVG_5m_10")
    assert "cancel_requested" in kinds and "cancelled" in kinds and m.state is MachineState.IDLE


def test_a_veto_is_recorded_once_per_signature(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path, equity=5_000.0)
    kinds = quiet_bars(m, short_plan(), 1, 4)
    assert kinds == ("veto",) and m.state is MachineState.IDLE
    veto = [r for r in JournalReader(tmp_path).records(EP) if r.record == "trade"][0].payload
    assert list(veto["verdict"]["vetoes"]) == ["position_size"]
    kinds = quiet_bars(m, short_plan(target_id="swing:d"), 5, 1)
    assert kinds == ("veto",)


def test_partial_fill_is_both_a_position_and_a_working_order(tmp_path: Path) -> None:
    journal = BrainJournal(tmp_path, run_id="fsm")
    journal.open_episode(EP, T0)
    broker = executor(equity=250_000.0, max_fill_per_bar=1)  # 250k buys 2 contracts at 0.5 % risk
    m = OrderMachine(broker, RiskGate(CONFIG), journal=journal)
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True)
    kinds = m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(), episode_id=EP, visible=lambda a: True)
    assert "partial" in kinds and m.state is MachineState.PARTIAL
    assert m.ledger.has_open_position() and m.ledger.has_working_order()
    kinds = m.on_bar(at(3), bar(3, 16380.0, 16390.0), short_plan(), episode_id=EP, visible=lambda a: True)
    assert "filled" in kinds and "position_opened" in kinds and m.state is MachineState.IN_POSITION


def test_a_rejected_entry_blocks_the_signature_until_the_plan_changes(tmp_path: Path) -> None:
    journal = BrainJournal(tmp_path, run_id="fsm")
    journal.open_episode(EP, T0)
    broker = executor(margin_per_contract=150_000.0)  # the margin refusal arrives on the poll after submission, as at TWS
    m = OrderMachine(broker, RiskGate(CONFIG), journal=journal)
    kinds = quiet_bars(m, short_plan(), 1, 2)
    assert kinds == ("submitted", "rejected") and m.state is MachineState.IDLE
    assert quiet_bars(m, short_plan(), 3, 3) == (), "a rejected signature is not resubmitted while the plan is unchanged"
    assert quiet_bars(m, short_plan(target_id="swing:d"), 6, 1) == ("submitted",)


def test_an_expired_signature_is_tradeable_again_in_a_new_episode(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    kinds = quiet_bars(m, short_plan(), 1, CONFIG.order_ttl_bars + 2)
    assert "expired" in kinds and m.state is MachineState.IDLE
    assert quiet_bars(m, short_plan(), CONFIG.order_ttl_bars + 3, 1) == ()
    episode_2 = "EP_20220103_002"
    journal.open_episode(episode_2, at(30))
    plan_2 = replace(short_plan(), episode_id=episode_2)
    kinds = m.on_bar(at(30), bar(30, 16370.0, 16380.0), plan_2, episode_id=episode_2, visible=lambda a: True)
    assert kinds == ("submitted",), "a new episode's identical plan is a new intent"


class LosesStopBroker(SimulatedExecutor):
    """Reports the stop leg cancelled right after the entry fills — what a
    rejected or cancelled child order at the broker looks like."""

    def poll(self, asof, bar):
        events = list(super().poll(asof, bar))
        for bracket in self._brackets.values():
            if bracket.entry.status is OrderStatus.FILLED and bracket.stop.status is OrderStatus.WORKING:
                lost = replace(bracket.stop, status=OrderStatus.CANCELLED, updated_at=asof)
                bracket.stop = lost
                events.append(BrokerEvent("cancelled", lost, None, asof))
        return tuple(events)


def test_a_lost_exit_leg_is_journaled_once(tmp_path: Path) -> None:
    journal = BrainJournal(tmp_path, run_id="fsm")
    journal.open_episode(EP, T0)
    broker = LosesStopBroker(SIM, tick_size=0.25, point_value=20.0)
    m = OrderMachine(broker, RiskGate(CONFIG), journal=journal)
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True)
    kinds = m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(), episode_id=EP, visible=lambda a: True)
    assert "filled" in kinds and "exit_leg_lost" in kinds and m.state is MachineState.IN_POSITION
    assert quiet_bars(m, None, 3, 2, low=16380.0, high=16390.0) == ()
    lost = [r for r in JournalReader(tmp_path).records(EP) if r.record == "trade" and r.payload["kind"] == "exit_leg_lost"]
    assert len(lost) == 1 and lost[0].payload["exit_role"] == "stop" and lost[0].payload["order"]["status"] == "cancelled"


def test_execution_view_follows_the_intent(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    assert m.ledger.execution_view() == IDLE_VIEW
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True)
    view = m.ledger.execution_view()
    assert view["status"] == "WORKING" and view["positions"] == [] and view["last_outcome"] is None and view["last_veto"] is None
    assert view["order"] == {
        "direction": "SHORT", "entry_object_id": "FVG_5m_10", "invalidation_object_id": "BSL_5m_3", "target_object_id": "SSL_4H_2", "thesis_id": "",
        "submitted_at": "2022-01-03T14:13:00Z", "bars_working": 0, "quantity": 1, "filled_quantity": 0, "ttl_bars": CONFIG.order_ttl_bars,
    }
    assert view["theses"] == [] and view["cooldown_bars_left"] == 0 and view["daily_stop"] is False and view["halted"] is False
    m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(), episode_id=EP, visible=lambda a: True)
    view = m.ledger.execution_view()
    assert view["status"] == "IN_POSITION" and view["order"] is None
    assert view["positions"] == [{"direction": "SHORT", "entry_object_id": "FVG_5m_10", "opened_at": "2022-01-03T14:14:00Z", "quantity": 1, "thesis_id": "", "invalidation_mode": "TOUCH"}]
    m.on_bar(at(3), bar(3, 16400.0, 16420.0), None, episode_id=EP, visible=lambda a: True)
    view = m.ledger.execution_view()
    assert view["status"] == "IDLE" and view["positions"] == [] and view["cooldown_bars_left"] == CONFIG.thesis.stop_cooldown_bars
    assert view["last_outcome"] == {"kind": "position_closed", "at": "2022-01-03T14:15:00Z", "reason": None, "exit_role": "stop", "thesis_id": ""}


def test_bars_working_counts_toward_the_ttl_and_an_expiry_is_the_last_outcome(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    quiet_bars(m, short_plan(), 1, 4)
    assert m.ledger.execution_view()["order"]["bars_working"] == 3
    quiet_bars(m, short_plan(), 5, CONFIG.order_ttl_bars)
    view = m.ledger.execution_view()
    assert view["status"] == "IDLE" and view["last_outcome"]["kind"] == "expired" and view["last_outcome"]["reason"] == "ttl"


def test_a_veto_is_counted_per_bar_and_recorded_per_llm_proposal(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path, equity=5_000.0)
    kinds = m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("veto",)
    assert quiet_bars(m, short_plan(), 2, 3) == (), "the TICK bars between two LLM calls add no record"
    veto = m.ledger.execution_view()["last_veto"]
    assert veto["vetoes"] == ["position_size"] and veto["bars_vetoed"] == 4 and veto["proposals_vetoed"] == 1
    assert veto["direction"] == "SHORT" and veto["entry_object_id"] == "FVG_5m_10" and veto["target_object_id"] == "SSL_4H_2"
    assert veto["first_vetoed_at"] == "2022-01-03T14:13:00Z" and veto["last_vetoed_at"] == "2022-01-03T14:16:00Z"
    assert veto["reward_risk"] == round(short_plan().geometry.reward_risk, 2) and veto["reasons"][0].startswith("risk budget")
    kinds = m.on_bar(at(5), bar(5, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("veto",), "the LLM saw the veto and proposed the same plan again: that is recorded"
    assert m.ledger.execution_view()["last_veto"]["proposals_vetoed"] == 2
    assert m.stats["veto_bars"] == 5 and m.stats["veto"] == 2
    records = [r.payload for r in JournalReader(tmp_path).records(EP) if r.record == "trade"]
    assert [r["kind"] for r in records] == ["veto", "veto"] and records[1]["proposals_vetoed"] == 2


def test_veto_memory_is_per_episode_and_cleared_by_a_submission(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path, equity=5_000.0)
    assert quiet_bars(m, short_plan(), 1, 2) == ("veto",)
    episode_2 = "EP_20220103_002"
    journal.open_episode(episode_2, at(3))
    plan_2 = replace(short_plan(), episode_id=episode_2)
    kinds = m.on_bar(at(3), bar(3, 16370.0, 16380.0), plan_2, episode_id=episode_2, visible=lambda a: True)
    assert kinds == ("veto",), "a new episode's identical veto is recorded again"
    veto = m.ledger.execution_view()["last_veto"]
    assert veto["bars_vetoed"] == 1 and veto["proposals_vetoed"] == 1
    m2, broker2, journal2 = machine(tmp_path / "b")
    assert quiet_bars(m2, short_plan(target_id="swing:x"), 1, 1) == ("submitted",)
    assert m2.ledger.execution_view()["last_veto"] is None
    m3, broker3, journal3 = machine(tmp_path / "c", equity=5_000.0)
    assert quiet_bars(m3, short_plan(), 1, 1) == ("veto",)
    broker3.account.cash = 100_000.0  # the account grows: the same plan now passes
    assert quiet_bars(m3, short_plan(), 2, 1) == ("submitted",)
    assert m3.ledger.execution_view()["last_veto"] is None and m3.stats["submitted"] == 1


def test_a_closed_position_is_not_reentered_until_the_plan_changes(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True)
    m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(), episode_id=EP, visible=lambda a: True)
    kinds = m.on_bar(at(3), bar(3, 16400.0, 16420.0), short_plan(), episode_id=EP, visible=lambda a: True)
    assert kinds == ("position_closed",) and m.state is MachineState.IDLE, "the plan the Brain still holds is not re-entered on the exit bar"
    assert quiet_bars(m, short_plan(), 4, 3) == ()
    assert m.ledger.execution_view()["last_outcome"]["kind"] == "position_closed"
    changed = short_plan(target_id="swing:d")
    assert quiet_bars(m, changed, 7, 1) == (), "even a changed plan waits for the Brain's next call after a close"
    kinds = m.on_bar(at(8), bar(8, 16370.0, 16380.0), changed, episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("thesis_refused",) and last_trade(tmp_path)["reason"] == "stop_cooldown", "the stop-out's cooldown holds every new entry"
    assert quiet_bars(m, changed, 9, 24) == (), "the same refusal is not journaled again on quiet bars"
    assert quiet_bars(m, changed, 33, 1) == ("submitted",), "the cooldown ends 30 bars after the stop"


def test_after_a_close_nothing_is_submitted_until_the_brain_has_been_called(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True, llm_called=True)
    m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(), episode_id=EP, visible=lambda a: True)
    other = short_plan(target_id="swing:d")  # proposed while in the position, on an earlier call
    kinds = m.on_bar(at(3), bar(3, 16400.0, 16420.0), other, episode_id=EP, visible=lambda a: True)
    assert kinds == ("position_closed",), "a plan chosen before the outcome was known is not entered on the exit bar"
    assert quiet_bars(m, other, 4, 2) == ()
    kinds = m.on_bar(at(6), bar(6, 16370.0, 16380.0), other, episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("thesis_refused",), "the first plan the Brain holds after seeing the close still waits out the cooldown"
    assert quiet_bars(m, other, 7, 27) == ("submitted",), "the first plan the Brain holds after the cooldown is entered"


# ----------------------------------------------------------------- v2: several positions, the thesis book, the exits at market


def test_three_same_direction_theses_open_three_positions_and_a_fourth_is_vetoed(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    for i in range(3):
        p = short_plan(target_id=f"swing:{i}", thesis_id=f"T{i}")
        assert "submitted" in m.on_bar(at(2 * i + 1), bar(2 * i + 1, 16370.0, 16380.0), p, episode_id=EP, visible=lambda a: True)
        assert "position_opened" in m.on_bar(at(2 * i + 2), bar(2 * i + 2, 16380.0, 16390.0), p, episode_id=EP, visible=lambda a: True)
    assert len(m.positions()) == 3 and m.state is MachineState.IN_POSITION and m.ledger.has_open_position()
    assert broker.snapshot(at(6)).net_position("NQ") == -3 and len(broker.snapshot(at(6)).open_orders) == 6
    kinds = m.on_bar(at(7), bar(7, 16370.0, 16380.0), short_plan(target_id="swing:x", thesis_id="T9"), episode_id=EP, visible=lambda a: True)
    assert kinds == ("veto",) and list(last_trade(tmp_path)["verdict"]["vetoes"]) == ["exposure"]
    view = m.ledger.execution_view()
    assert [p["thesis_id"] for p in view["positions"]] == ["T0", "T1", "T2"] and [t["status"] for t in view["theses"]] == ["OPEN"] * 3 + ["OPEN"]
    # one target closes one position; the others live on
    kinds = m.on_bar(at(8), bar(8, 16320.0, 16340.0), None, episode_id=EP, visible=lambda a: True)
    assert kinds.count("position_closed") == 3, "one bar reaching the target closes all three (same prices)"


def test_an_opposite_direction_plan_is_vetoed_while_a_position_is_open(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(thesis_id="T1"), episode_id=EP, visible=lambda a: True)
    m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(thesis_id="T1"), episode_id=EP, visible=lambda a: True)
    long_ = short_plan(direction=TradeDirection.LONG, entry=16380.0, stop=16360.0, target=16430.0, thesis_id="T2", target_id="swing:d")
    kinds = m.on_bar(at(3), bar(3, 16380.0, 16390.0), long_, episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("veto",) and "opposite" in last_trade(tmp_path)["verdict"]["reasons"][0]


def test_a_closed_thesis_is_refused_once_per_proposal_and_a_new_one_waits_out_the_cooldown(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(thesis_id="T1"), episode_id=EP, visible=lambda a: True)
    m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(thesis_id="T1"), episode_id=EP, visible=lambda a: True)
    assert m.on_bar(at(3), bar(3, 16400.0, 16420.0), short_plan(thesis_id="T1"), episode_id=EP, visible=lambda a: True) == ("position_closed",)
    again = short_plan(thesis_id="T1", target_id="swing:d")  # the same thesis through another object
    kinds = m.on_bar(at(4), bar(4, 16370.0, 16380.0), again, episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("thesis_refused",) and last_trade(tmp_path)["reason"] == "thesis_closed"
    assert quiet_bars(m, again, 5, 3) == (), "quiet bars do not repeat the refusal"
    kinds = m.on_bar(at(8), bar(8, 16370.0, 16380.0), again, episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("thesis_refused",) and m.stats["thesis_refused"] == 2, "the LLM proposed it again: recorded"
    fresh = short_plan(thesis_id="T2", target_id="swing:e")
    kinds = m.on_bar(at(9), bar(9, 16370.0, 16380.0), fresh, episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("thesis_refused",) and last_trade(tmp_path)["reason"] == "stop_cooldown" and last_trade(tmp_path)["cooldown_bars_left"] == 24
    assert quiet_bars(m, fresh, 10, 23) == () and quiet_bars(m, fresh, 33, 1) == ("submitted",)
    view = m.ledger.execution_view()
    assert view["theses"][0] == {"thesis_id": "T1", "direction": "SHORT", "governing_timeframe": "15m", "status": "CLOSED", "closed_reason": "stopped", "expressions": 1, "last_outcome": "position_closed:stop"}
    assert view["theses"][1]["status"] == "OPEN" and view["theses"][1]["expressions"] == 1
    kinds = quiet_bars(m, short_plan(thesis_id="T1", target_id="swing:f"), 34, 1)
    assert "submitted" not in kinds, "the closed thesis stays closed (and the changed plan cancels T2's working entry)"


def test_a_thesis_is_capped_at_two_expressions_in_an_episode(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    for i in range(2):
        p = short_plan(thesis_id="T1", target_id=f"swing:{i}")
        start = 1 + i * (CONFIG.order_ttl_bars + 2)
        kinds = quiet_bars(m, p, start, CONFIG.order_ttl_bars + 2)
        assert "submitted" in kinds and "expired" in kinds
    third = short_plan(thesis_id="T1", target_id="swing:z")
    kinds = m.on_bar(at(40), bar(40, 16370.0, 16380.0), third, episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("thesis_refused",) and last_trade(tmp_path)["reason"] == "expressions_exhausted"
    assert m.ledger.execution_view()["theses"][0]["closed_reason"] == "expressions_exhausted"


def test_close_beyond_exits_at_market_when_the_scale_closes_beyond_the_object(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path, equity=250_000.0)  # 1 250 of budget holds the 42.5-point hard stop
    # the invalidation object (a 5m pool) sits at 16411.5; the hard stop carries the buffer at 16430
    p = short_plan(stop=16430.0, target=16290.0, thesis_id="T1", invalidation_mode="CLOSE_BEYOND", invalidation_level=16411.5)  # 97.5 / 42.5 = 2.3 R
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), p, episode_id=EP, visible=lambda a: True)
    assert "position_opened" in m.on_bar(at(2), bar(2, 16380.0, 16390.0), p, episode_id=EP, visible=lambda a: True)
    wick = Bar(start=at(3) - pd.Timedelta(minutes=1), open=16390.0, high=16418.0, low=16388.0, close=16405.0, volume=1.0, symbol="NQ", instrument_id=1)
    assert m.on_bar(at(3), wick, p, episode_id=EP, visible=lambda a: True, closed_timeframes=frozenset({"5m"})) == (), "a wick through the level with a 5m close inside is not an invalidation"
    beyond = Bar(start=at(4) - pd.Timedelta(minutes=1), open=16405.0, high=16416.0, low=16404.0, close=16413.0, volume=1.0, symbol="NQ", instrument_id=1)
    assert m.on_bar(at(4), beyond, p, episode_id=EP, visible=lambda a: True, closed_timeframes=frozenset({"15m"})) == (), "another scale's close does not count"
    kinds = m.on_bar(at(5), beyond, p, episode_id=EP, visible=lambda a: True, closed_timeframes=frozenset({"5m"}))
    assert kinds == ("invalidation_close", "cancel_requested", "cancel_requested") and m.state is MachineState.IN_POSITION
    closed = last_trade(tmp_path, "invalidation_close")
    assert closed["close"] == 16413.0 and closed["invalidation_level"] == 16411.5 and closed["timeframe"] == "5m"
    nxt = Bar(start=at(6) - pd.Timedelta(minutes=1), open=16414.0, high=16420.0, low=16410.0, close=16415.0, volume=1.0, symbol="NQ", instrument_id=1)
    kinds = m.on_bar(at(6), nxt, p, episode_id=EP, visible=lambda a: True)
    assert "position_closed" in kinds and "flattened" in kinds and "exit_leg_lost" not in kinds and m.state is MachineState.IDLE
    closed = last_trade(tmp_path, "position_closed")
    assert closed["exit_role"] == "invalidation" and closed["exit_price"] == 16414.0 and closed["reason"] == "invalidation"
    assert broker.account.positions == {} and broker.snapshot(at(6)).open_orders == ()
    assert m.ledger.execution_view()["theses"][0]["closed_reason"] == "stopped" and m.ledger.execution_view()["cooldown_bars_left"] == CONFIG.thesis.stop_cooldown_bars


def test_the_hard_stop_halts_cancels_and_flattens_everything(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path, equity=250_000.0)
    p = short_plan(stop=16430.0, target=16290.0, thesis_id="T1")
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), p, episode_id=EP, visible=lambda a: True)
    m.on_bar(at(2), bar(2, 16380.0, 16390.0), p, episode_id=EP, visible=lambda a: True)
    second = short_plan(stop=16430.0, target=16290.0, thesis_id="T2", target_id="swing:d")
    assert m.on_bar(at(3), bar(3, 16380.0, 16390.0), second, episode_id=EP, visible=lambda a: True) == ("submitted",)
    assert m.state is MachineState.WORKING and len(m.positions()) == 1
    broker.account.cash = 232_500.0  # the account marks 7 % below its 250 000 peak
    kinds = m.on_bar(at(4), bar(4, 16370.0, 16385.0), second, episode_id=EP, visible=lambda a: True)  # the second entry is not touched
    assert kinds.count("cancel_requested") == 3 and kinds[-1] == "halted" and m.halted
    halted = last_trade(tmp_path, "halted")
    assert halted["peak"] >= 250_000.0 and halted["drawdown"] >= 0.065 and halted["positions_flattened"] == 1  # the peak was marked with the open position
    kinds = m.on_bar(at(5), bar(5, 16385.0, 16395.0), second, episode_id=EP, visible=lambda a: True)
    assert "cancelled" in kinds and "position_closed" in kinds and "flattened" in kinds
    assert last_trade(tmp_path, "position_closed")["exit_role"] == "flatten" and m.state is MachineState.IDLE
    assert broker.account.positions == {} and broker.snapshot(at(5)).open_orders == ()
    assert quiet_bars(m, short_plan(thesis_id="T3", target_id="swing:e"), 6, 3) == (), "nothing is ever submitted again"
    assert m.ledger.execution_view()["halted"] is True and m.halt_record["positions_flattened"] == 1
