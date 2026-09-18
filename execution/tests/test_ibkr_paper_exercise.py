"""The paper exercise driven through the ``Broker`` protocol against the fake TWS."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from contract.execution import OrderRole, OrderStatus
from execution.core.ibkr_broker import IBKRBroker, IBKRRefused
from execution.scripts.ibkr_paper_exercise import exercise
from execution.tests.test_ibkr_broker import CONFIG, FakeIB, T


class ScriptedIB(FakeIB):
    """Every ``waitOnUpdate`` moves the session on: a fresh order becomes
    Submitted; a SELL limit below the reference (or a market order) fills."""

    def __init__(self, reference: float) -> None:
        super().__init__()
        self.reference = reference
        self.updates = 0

    def waitOnUpdate(self, timeout=0.0):
        self.updates += 1
        for trade in self._trades:
            status, order = trade.orderStatus.status, trade.order
            if status == "PreSubmitted":
                trade.orderStatus.status = "Submitted"
            elif status == "Submitted" and not order.parentId:
                marketable = order.orderType == "MKT" or (order.orderType == "LMT" and order.action == "SELL" and order.lmtPrice <= self.reference)
                if marketable:
                    price = self.reference if order.orderType == "MKT" else order.lmtPrice
                    self.set_status(order.orderId, "Filled", filled=float(order.totalQuantity), avg=price)
                    if order.action == "SELL" and order.orderType == "LMT":
                        self._positions = [SimpleNamespace(account=self._accounts[0], contract=SimpleNamespace(symbol="NQ", multiplier="20"), position=-float(order.totalQuantity), avgCost=price * 20)]
                    elif order.orderType == "MKT":
                        self._positions = []
        return True


def clock():
    ticks = [T + pd.Timedelta(seconds=i) for i in range(10_000)]
    return lambda: ticks.pop(0)


def test_working_cancel_and_replace_leave_the_account_flat() -> None:
    ib = ScriptedIB(reference=16400.0)
    broker = IBKRBroker(ib, CONFIG, live_execution_allowed=False)
    receipt = exercise(broker, reference_price=16400.0, tick_size=0.25, marketable=False, clock=clock(), sleep=lambda s: None, log=lambda m: None)
    assert [step["name"] for step in receipt["steps"]] == ["unmarketable_bracket", "cancel", "replace_bracket", "cancel_replacement"]
    assert all(step["ok"] for step in receipt["steps"]), receipt["steps"]
    kinds = [event["kind"] for step in receipt["steps"] for event in step["events"]]
    assert kinds.count("working") >= 2 and kinds.count("cancelled") >= 2 and "filled" not in kinds
    assert receipt["final_snapshot"]["positions"] == [] and receipt["flat_at_end"] is True
    assert ib.cancelled and receipt["steps"][0]["events"][0]["order"]["limit_price"] == 16564.0  # 1 % above, on the tick


def test_marketable_fill_is_flattened_with_a_market_order() -> None:
    ib = ScriptedIB(reference=16400.0)
    broker = IBKRBroker(ib, CONFIG, live_execution_allowed=False)
    receipt = exercise(broker, reference_price=16400.0, tick_size=0.25, marketable=True, clock=clock(), sleep=lambda s: None, log=lambda m: None)
    names = [step["name"] for step in receipt["steps"]]
    assert names[-3:] == ["marketable_bracket", "cancel_exits", "flatten"] and all(step["ok"] for step in receipt["steps"]), receipt["steps"]
    fill = next(e for step in receipt["steps"] for e in step["events"] if e["kind"] == "filled")
    assert fill["fill"]["price"] == 16318.0 and fill["order"]["side"] == "SELL"
    flatten = [order for _, order in ib.placed if order.orderType == "MKT"]
    assert len(flatten) == 1 and flatten[0].action == "BUY" and flatten[0].totalQuantity == 1
    assert receipt["flat_at_end"] is True and receipt["final_snapshot"]["positions"] == []


def test_a_non_flat_account_is_refused_before_any_order() -> None:
    ib = ScriptedIB(reference=16400.0)
    ib._positions = [SimpleNamespace(account="DU1234567", contract=SimpleNamespace(symbol="NQ", multiplier="20"), position=1.0, avgCost=16400.0 * 20)]
    broker = IBKRBroker(ib, CONFIG, live_execution_allowed=False)
    with pytest.raises(IBKRRefused):
        exercise(broker, reference_price=16400.0, tick_size=0.25, marketable=False, clock=clock(), sleep=lambda s: None, log=lambda m: None)
    assert ib.placed == []


def test_flatten_places_a_market_order_on_the_other_side() -> None:
    ib = FakeIB()
    broker = IBKRBroker(ib, CONFIG, live_execution_allowed=False)
    state = broker.flatten("NQ", 2, "BUY", T, "paper-exercise:flatten")
    assert state.side == "BUY" and state.quantity == 2 and state.limit_price is None and state.status is OrderStatus.SUBMITTED
    assert ib.placed[-1][1].orderType == "MKT" and ib.placed[-1][1].orderRef == "paper-exercise:flatten" and state.role is OrderRole.FLATTEN
