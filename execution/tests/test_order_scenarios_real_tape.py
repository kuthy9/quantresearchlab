"""The order lifecycle on the real 2022-01-03 tape.

The Brain is replaced by scripted plans whose limit sits on a price the
tape reaches on a known later bar, so every fill, cancel, partial and race
below is deterministic, and the virtual account is checked after each step:
cash, positions, the orders' states and the margin held.  Nothing here
reads a bar the machine has not been handed yet."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from brain.scripts._run_identity import DEFAULT_SOURCE
from contract.brain.state import TradeDirection
from contract.decision import OpportunityGeometry
from contract.execution import OrderRole, OrderStatus, Position
from contract.market.primitives import Bar
from contract.risk import ObjectRef, TradePlan
from execution.core.order_fsm import MachineState, OrderMachine
from execution.core.simulated_executor import SimulatedExecutor, SimulatorConfig
from risk.core.gate import RiskConfig, RiskGate
from shares.core.io import iter_completed_bars, load_ohlcv

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / DEFAULT_SOURCE
# The scenarios size at the v1 budget (0.5 %) with no leverage cap so the
# contract counts below stay one and two; sizing is the gate's tests' business.
RISK = replace(RiskConfig.from_json(ROOT / "risk" / "configs" / "risk.json"), risk_fraction={"BASE": 0.005, "A_PLUS": 0.005}, max_leverage=1_000.0)
SIM = SimulatorConfig.from_json(ROOT / "execution" / "configs" / "simulated_executor.json")
EP = "EP_20220103_001"
POINT = RISK.contract.point_value
MARGIN = SIM.margin_per_contract
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="materialized NQ tape not linked")


@pytest.fixture(scope="module")
def rth() -> list[Bar]:
    loaded = load_ohlcv(SOURCE, start="2022-01-03 09:30", end="2022-01-03 16:00")
    bars = list(iter_completed_bars(loaded.frame))
    assert len(bars) > 300
    return bars


def asof(bar: Bar) -> pd.Timestamp:
    return pd.Timestamp(bar.start) + pd.Timedelta(minutes=1)


def executor(equity: float = SIM.initial_equity, **overrides) -> SimulatedExecutor:
    config = SIM if not overrides else SimulatorConfig(**{**SIM.__dict__, **overrides})
    return SimulatedExecutor(config, tick_size=RISK.contract.tick_size, point_value=POINT, equity=equity)


def machine(broker: SimulatedExecutor) -> OrderMachine:
    return OrderMachine(broker, RiskGate(RISK), journal=None)


def plan(direction: TradeDirection, entry: float, *, stop_points: float = 20.0, target_points: float = 40.0, target_id: str = "swing:c") -> TradePlan:
    sign = 1.0 if direction is TradeDirection.SHORT else -1.0
    stop = entry + sign * stop_points
    target = entry - sign * target_points
    return TradePlan(
        EP, 3, pd.Timestamp("2022-01-03T14:31:00Z"), direction,
        ObjectRef("FVG_5m_10", "fvg:a", "fvg", "5m"), ObjectRef("BSL_5m_3", "swing:b", "bsl", "5m"), ObjectRef("SSL_4H_2", target_id, "ssl", "4H"),
        OpportunityGeometry(entry, stop, target, target_points / stop_points, ("e", "s", "t")), close=entry,
    )


def first_sell_touch(bars: list[Bar], *, quiet: int, start: int = 5, after_quiet_bars: int = 1, exits: tuple[float, float] | None = None) -> tuple[int, float]:
    """The first submission index ``i`` and price ``p`` such that bars
    ``i+1 … i+quiet`` stay below ``p`` and bars ``i+quiet+1 … i+quiet+after_quiet_bars``
    all reach it (their highs); with ``exits`` (stop above, target below) the
    touching bars after the first also stay inside them."""
    for i in range(start, len(bars) - quiet - after_quiet_bars - 20):
        touching = bars[i + quiet + 1: i + quiet + 1 + after_quiet_bars]
        p = min(float(b.high) for b in touching)
        if all(float(b.high) < p for b in bars[i + 1: i + quiet + 1]) and p == float(touching[0].high):
            if exits is not None:
                stop, target = p + exits[0], p - exits[1]
                if any(float(b.high) >= stop or float(b.low) <= target for b in touching[1:]):
                    continue
            return i, p
    raise AssertionError("no such bar sequence in the window")


def step(m: OrderMachine, bars: list[Bar], index: int, the_plan: TradePlan | None, **kw) -> tuple[str, ...]:
    return m.on_bar(asof(bars[index]), bars[index], the_plan, episode_id=EP, visible=lambda alias: True, **kw)


def run_until(m: OrderMachine, bars: list[Bar], start: int, the_plan: TradePlan | None, kind: str, *, limit: int = 300) -> tuple[int, tuple[str, ...]]:
    for index in range(start, min(len(bars), start + limit)):
        kinds = step(m, bars, index, the_plan)
        if kind in kinds:
            return index, kinds
    raise AssertionError(f"{kind} never happened within {limit} bars")


# ----------------------------------------------------------------- scenarios


def test_normal_fill_moves_the_account_only_at_the_exit(rth) -> None:
    i, p = first_sell_touch(rth, quiet=3)
    broker = executor()
    m = machine(broker)
    assert step(m, rth, i, plan(TradeDirection.SHORT, p)) == ("submitted",)
    assert step(m, rth, i + 1, plan(TradeDirection.SHORT, p)) == ("working",)
    for k in range(i + 2, i + 4):
        assert step(m, rth, k, plan(TradeDirection.SHORT, p)) == ()
    kinds = step(m, rth, i + 4, plan(TradeDirection.SHORT, p))
    assert kinds == ("filled", "position_opened") and m.state is MachineState.IN_POSITION
    snapshot = broker.snapshot(asof(rth[i + 4]))
    assert snapshot.positions == (Position("NQ", -1, p),) and broker.account.cash == SIM.initial_equity
    assert len(snapshot.open_orders) == 2 and {o.role for o in snapshot.open_orders} == {OrderRole.STOP, OrderRole.TARGET}
    assert broker.account.margin_held(MARGIN) == MARGIN
    index, kinds = run_until(m, rth, i + 5, plan(TradeDirection.SHORT, p), "position_closed")
    assert kinds == ("position_closed",) and m.state is MachineState.IDLE and broker.account.positions == {}
    assert step(m, rth, index + 1, plan(TradeDirection.SHORT, p)) == (), "the unchanged plan is not re-entered after the exit"
    exit_price = [f for f in broker.account.fills][-1].price
    assert exit_price in (p + 20.0, p - 40.0)
    assert broker.account.cash == pytest.approx(SIM.initial_equity + (p - exit_price) * POINT)
    summary = broker.account.summary()
    assert summary["orders"] == {"pending": 0, "filled": 2, "cancelled": 1, "rejected": 0} and summary["fills"] == 2
    assert broker.account.margin_held(MARGIN) == 0.0


def test_a_long_fills_on_the_low_and_mirrors_the_pnl(rth) -> None:
    # mirror: the first bar whose low reaches p after three quiet bars above it
    for i in range(5, len(rth) - 30):
        p = float(rth[i + 4].low)
        if all(float(b.low) > p for b in rth[i + 1: i + 4]):
            break
    broker = executor()
    m = machine(broker)
    long = plan(TradeDirection.LONG, p)
    step(m, rth, i, long)
    for k in range(i + 1, i + 4):
        step(m, rth, k, long)
    assert step(m, rth, i + 4, long) == ("filled", "position_opened")
    assert broker.snapshot(asof(rth[i + 4])).positions == (Position("NQ", 1, p),)
    run_until(m, rth, i + 5, long, "position_closed")
    exit_price = broker.account.fills[-1].price
    assert exit_price in (p - 20.0, p + 40.0)
    assert broker.account.cash == pytest.approx(SIM.initial_equity + (exit_price - p) * POINT)


def test_waiting_then_the_plan_is_dropped_cancels_and_releases_margin(rth) -> None:
    i, p = first_sell_touch(rth, quiet=6)
    broker = executor()
    m = machine(broker)
    step(m, rth, i, plan(TradeDirection.SHORT, p))
    step(m, rth, i + 1, plan(TradeDirection.SHORT, p))
    assert broker.account.margin_held(MARGIN) == MARGIN
    assert step(m, rth, i + 2, None) == ("cancel_requested",)
    assert step(m, rth, i + 3, None) == ("cancelled",) and m.state is MachineState.IDLE
    snapshot = broker.snapshot(asof(rth[i + 3]))
    assert snapshot.positions == () and snapshot.open_orders == () and broker.account.margin_held(MARGIN) == 0.0
    assert broker.account.summary()["orders"] == {"pending": 0, "filled": 0, "cancelled": 3, "rejected": 0}
    # the touch comes and goes: nothing is working any more
    assert step(m, rth, i + 7, None) == () and broker.account.cash == SIM.initial_equity


def test_waiting_then_a_new_plan_cancels_first_and_submits_next_bar(rth) -> None:
    i, p = first_sell_touch(rth, quiet=6)
    broker = executor()
    m = machine(broker)
    first = plan(TradeDirection.SHORT, p)
    second = plan(TradeDirection.SHORT, p - 1.0, target_id="swing:d")
    assert first.signature != second.signature
    step(m, rth, i, first)
    first_id = broker.snapshot(asof(rth[i])).open_entry_orders()[0].order_id
    assert step(m, rth, i + 1, second) == ("working", "cancel_requested")
    assert step(m, rth, i + 2, second) == ("cancelled", "submitted")
    entries = broker.snapshot(asof(rth[i + 2])).open_entry_orders()
    assert len(entries) == 1 and entries[0].order_id != first_id and entries[0].limit_price == p - 1.0
    for k in range(i, i + 3):
        pass
    assert broker.account.summary()["orders"]["pending"] == 3 and broker.account.margin_held(MARGIN) == MARGIN
    assert m.working_signature == second.signature


def test_waiting_then_reanalysis_keeps_or_cancels_the_order(rth) -> None:
    i, p = first_sell_touch(rth, quiet=6)
    broker = executor()
    m = machine(broker)
    same = plan(TradeDirection.SHORT, p)
    step(m, rth, i, same)
    order_id = broker.snapshot(asof(rth[i])).open_entry_orders()[0].order_id
    # the Brain re-analyses and keeps the plan: the same order keeps working
    for k in range(i + 1, i + 4):
        kinds = step(m, rth, k, same)
        assert "submitted" not in kinds
        assert [o.order_id for o in broker.snapshot(asof(rth[k])).open_entry_orders()] == [order_id]
    # the Brain re-analyses and finds nothing: the order is cancelled and no new one appears
    assert step(m, rth, i + 4, None) == ("cancel_requested",)
    assert step(m, rth, i + 5, None) == ("cancelled",)
    for k in range(i + 6, i + 12):
        assert step(m, rth, k, None) == ()
    assert broker.snapshot(asof(rth[i + 11])).open_orders == () and broker.account.positions == {}


def test_partial_fill_keeps_the_account_equal_to_the_filled_quantity(rth) -> None:
    i, p = first_sell_touch(rth, quiet=3, after_quiet_bars=2, exits=(25.0, 50.0))
    broker = executor(equity=200_000.0, max_fill_per_bar=1)  # 0.5 % of 200k buys two contracts at 25 points
    m = machine(broker)
    two = plan(TradeDirection.SHORT, p, stop_points=25.0, target_points=50.0)
    assert step(m, rth, i, two) == ("submitted",)
    entry = broker.snapshot(asof(rth[i])).open_entry_orders()[0]
    assert entry.quantity == 2
    for k in range(i + 1, i + 4):
        step(m, rth, k, two)
    assert step(m, rth, i + 4, two) == ("partial",) and m.state is MachineState.PARTIAL
    snapshot = broker.snapshot(asof(rth[i + 4]))
    assert snapshot.positions == (Position("NQ", -1, p),)
    assert snapshot.open_entry_orders()[0].filled_quantity == 1 and snapshot.open_entry_orders()[0].status is OrderStatus.PARTIAL
    assert [o.quantity for o in snapshot.open_orders if o.role is not OrderRole.ENTRY] == [1, 1]
    assert broker.account.margin_held(MARGIN) == 2 * MARGIN  # one open, one still working
    assert step(m, rth, i + 5, two) == ("filled", "position_opened") and m.state is MachineState.IN_POSITION
    snapshot = broker.snapshot(asof(rth[i + 5]))
    assert snapshot.positions == (Position("NQ", -2, p),) and snapshot.open_entry_orders() == ()
    assert [o.quantity for o in snapshot.open_orders] == [2, 2] and broker.account.cash == 200_000.0
    assert m.ledger.execution_view()["positions"][0]["quantity"] == 2


def test_cancel_after_a_partial_keeps_the_filled_part_and_its_exits(rth) -> None:
    i, p = first_sell_touch(rth, quiet=3, after_quiet_bars=2, exits=(25.0, 50.0))
    broker = executor(equity=200_000.0, max_fill_per_bar=1)
    m = machine(broker)
    two = plan(TradeDirection.SHORT, p, stop_points=25.0, target_points=50.0)
    step(m, rth, i, two)
    for k in range(i + 1, i + 4):
        step(m, rth, k, two)
    assert step(m, rth, i + 4, None) == ("partial", "cancel_requested")
    kinds = step(m, rth, i + 5, None)
    assert "cancelled" in kinds and "filled" not in kinds and m.state is MachineState.IN_POSITION
    snapshot = broker.snapshot(asof(rth[i + 5]))
    assert snapshot.positions == (Position("NQ", -1, p),) and snapshot.open_entry_orders() == ()
    assert [o.quantity for o in snapshot.open_orders] == [1, 1] and broker.account.margin_held(MARGIN) == MARGIN
    entry = next(o for o in broker.account.orders.values() if o.role is OrderRole.ENTRY)
    assert entry.status is OrderStatus.CANCELLED and entry.filled_quantity == 1
    # a new same-direction plan while the partial position lives is a second position (Risk v2 allows three)
    kinds = step(m, rth, i + 6, plan(TradeDirection.SHORT, p - 2.0, target_id="swing:d"))
    assert kinds == ("submitted",) or "position_closed" in kinds
    assert len(broker.snapshot(asof(rth[i + 6])).open_entry_orders()) == (1 if kinds == ("submitted",) else 0)


def test_a_risk_veto_leaves_the_broker_untouched(rth) -> None:
    i, p = first_sell_touch(rth, quiet=3)
    broker = executor()
    m = machine(broker)
    thin = plan(TradeDirection.SHORT, p, stop_points=20.0, target_points=20.0)  # reward-to-risk 1.0 < 2
    assert step(m, rth, i, thin, llm_called=True) == ("veto",)
    for k in range(i + 1, i + 8):
        assert step(m, rth, k, thin) == ()
    assert broker.account.orders == {} and broker.account.positions == {} and broker.account.fills == []
    assert m.ledger.execution_view()["last_veto"]["vetoes"] == ["reward_risk"]


def test_funds_and_position_limits_block_the_order(rth) -> None:
    i, p = first_sell_touch(rth, quiet=3)
    # margin above the available funds: accepted, then rejected by the next poll, never a position
    broker = executor(margin_per_contract=150_000.0)
    m = machine(broker)
    assert step(m, rth, i, plan(TradeDirection.SHORT, p)) == ("submitted",)
    assert step(m, rth, i + 1, plan(TradeDirection.SHORT, p)) == ("rejected",) and m.state is MachineState.IDLE
    for k in range(i + 2, i + 6):
        assert step(m, rth, k, plan(TradeDirection.SHORT, p)) == ()
    assert broker.account.positions == {} and broker.account.summary()["orders"] == {"pending": 0, "filled": 0, "cancelled": 2, "rejected": 1}
    # a position already in the account: the gate vetoes on exposure before any order
    broker = executor()
    broker.account.positions["NQ"] = Position("NQ", 1, p)
    m = machine(broker)
    assert step(m, rth, i, plan(TradeDirection.SHORT, p), llm_called=True) == ("veto",)
    assert m.ledger.execution_view()["last_veto"]["vetoes"] == ["exposure"] and broker.account.orders == {}
    # a large account: the quantity is capped at max_quantity
    broker = executor(equity=10_000_000.0)
    m = machine(broker)
    assert step(m, rth, i, plan(TradeDirection.SHORT, p)) == ("submitted",)
    assert broker.snapshot(asof(rth[i])).open_entry_orders()[0].quantity == RISK.max_quantity == 5


def test_the_same_signal_while_an_order_works_never_duplicates(rth) -> None:
    i, p = first_sell_touch(rth, quiet=8)
    broker = executor()
    m = machine(broker)
    same = plan(TradeDirection.SHORT, p)
    submitted = 0
    for k in range(i, i + 10):
        kinds = step(m, rth, k, same, llm_called=(k % 2 == 0))
        submitted += kinds.count("submitted")
        assert len(broker.snapshot(asof(rth[k])).open_entry_orders()) <= 1
    assert submitted == 1 and m.state is MachineState.IN_POSITION
    assert broker.account.summary()["orders"]["filled"] == 1 and broker.snapshot(asof(rth[i + 9])).positions == (Position("NQ", -1, p),)


class FillsDespiteCancel(SimulatedExecutor):
    """Applies the bar's touch before a pending cancel — the race a live
    broker can lose: the cancel request arrives after the fill."""

    def poll(self, asof, bar):
        requested = {entry_id for entry_id, bracket in self._brackets.items() if bracket.cancel_requested}
        for entry_id in requested:
            self._brackets[entry_id].cancel_requested = False
        events = super().poll(asof, bar)
        for entry_id in requested:
            bracket = self._brackets.get(entry_id)
            if bracket is not None and bracket.entry.is_open:
                bracket.cancel_requested = True
        return events


def test_a_fill_that_beats_the_cancel_is_a_position_not_an_error(rth) -> None:
    i, p = first_sell_touch(rth, quiet=3)
    broker = FillsDespiteCancel(SIM, tick_size=RISK.contract.tick_size, point_value=POINT)
    m = machine(broker)
    same = plan(TradeDirection.SHORT, p)
    step(m, rth, i, same)
    step(m, rth, i + 1, same)
    step(m, rth, i + 2, same)
    assert step(m, rth, i + 3, None) == ("cancel_requested",)
    kinds = step(m, rth, i + 4, None)  # the touch bar: the fill wins
    assert kinds == ("filled", "position_opened") and m.state is MachineState.IN_POSITION
    snapshot = broker.snapshot(asof(rth[i + 4]))
    assert snapshot.positions == (Position("NQ", -1, p),) and snapshot.open_entry_orders() == ()
    assert [o.role for o in snapshot.open_orders] == [OrderRole.STOP, OrderRole.TARGET] and broker.account.cash == SIM.initial_equity
    assert m.ledger.execution_view()["positions"][0]["quantity"] == 1 and m.ledger.has_open_position()
    assert step(m, rth, i + 5, None) in ((), ("position_closed",))  # the late cancel has nothing left to cancel
    run_until(m, rth, i + 6, None, "position_closed")
    assert broker.account.positions == {} and broker.account.summary()["orders"]["pending"] == 0
