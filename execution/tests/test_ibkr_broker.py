from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from contract.execution import BracketIntent, OrderRole, OrderStatus
from contract.execution import AccountSnapshot, OrderState, Position
from execution.core.ibkr_broker import IBKRBroker, IBKRConfig, IBKRDisconnected, IBKRRefused, require_flat

ROOT = Path(__file__).resolve().parents[2]
CONFIG = IBKRConfig.from_json(ROOT / "execution" / "configs" / "ibkr.json")
T = pd.Timestamp("2022-01-03T14:12:00Z")


@dataclass
class FakeOrder:
    orderId: int
    action: str
    totalQuantity: float
    orderType: str
    lmtPrice: float = 0.0
    auxPrice: float = 0.0
    parentId: int = 0
    orderRef: str = ""
    tif: str = "DAY"
    transmit: bool = True


@dataclass
class FakeTrade:
    contract: object
    order: FakeOrder
    orderStatus: SimpleNamespace
    fills: list = field(default_factory=list)


class FakeIB:
    def __init__(self, accounts=("DU1234567",)) -> None:
        self._accounts = list(accounts)
        self._trades: list[FakeTrade] = []
        self.cancelled: list[int] = []
        self.placed: list[tuple[object, FakeOrder]] = []
        self._next = 100
        self._fills: list = []
        self._positions: list = []
        self.connected = True

    def isConnected(self):
        return self.connected

    # --- what the adapter reads
    def managedAccounts(self):
        return self._accounts

    def accountSummary(self, account=""):
        return [
            SimpleNamespace(account=self._accounts[0], tag="NetLiquidation", value="250000.5", currency="USD"),
            SimpleNamespace(account=self._accounts[0], tag="AvailableFunds", value="200000", currency="USD"),
            SimpleNamespace(account=self._accounts[0], tag="BuyingPower", value="1000000", currency="USD"),
        ]

    def positions(self, account=""):
        return self._positions

    def openTrades(self):
        return [t for t in self._trades if t.orderStatus.status in ("PendingSubmit", "PreSubmitted", "Submitted")]

    def trades(self):
        return self._trades

    def fills(self):
        return self._fills

    # --- what the adapter drives
    def bracketOrder(self, action, quantity, limitPrice, takeProfitPrice, stopLossPrice, **kwargs):
        parent = FakeOrder(self._next, action, quantity, "LMT", lmtPrice=limitPrice, transmit=False, **kwargs)
        reverse = "SELL" if action == "BUY" else "BUY"
        take = FakeOrder(self._next + 1, reverse, quantity, "LMT", lmtPrice=takeProfitPrice, parentId=parent.orderId, transmit=False, **kwargs)
        stop = FakeOrder(self._next + 2, reverse, quantity, "STP", auxPrice=stopLossPrice, parentId=parent.orderId, transmit=True, **kwargs)
        self._next += 3
        return SimpleNamespace(parent=parent, takeProfit=take, stopLoss=stop)

    def placeOrder(self, contract, order):
        if not order.orderId:  # ib_async assigns the id at placement
            order.orderId = self._next
            self._next += 1
        trade = FakeTrade(contract, order, SimpleNamespace(status="PreSubmitted", filled=0.0, remaining=order.totalQuantity, avgFillPrice=0.0))
        self._trades.append(trade)
        self.placed.append((contract, order))
        return trade

    def cancelOrder(self, order):
        self.cancelled.append(order.orderId)
        for trade in self._trades:
            # TWS cancels the children attached to a cancelled parent
            if trade.order.orderId == order.orderId or trade.order.parentId == order.orderId:
                trade.orderStatus.status = "Cancelled"

    def waitOnUpdate(self, timeout=0.0):
        return True

    # --- test helpers
    def set_status(self, order_id, status, filled=0.0, avg=0.0):
        for trade in self._trades:
            if trade.order.orderId == order_id:
                trade.orderStatus = SimpleNamespace(status=status, filled=filled, remaining=trade.order.totalQuantity - filled, avgFillPrice=avg)
                if filled:
                    self._fills.append(SimpleNamespace(execution=SimpleNamespace(orderId=order_id, shares=filled, price=avg, time=T.to_pydatetime()), contract=trade.contract))


def broker(ib=None) -> IBKRBroker:
    return IBKRBroker(ib or FakeIB(), CONFIG, live_execution_allowed=False)


def test_config_holds_no_secret_and_names_the_contract() -> None:
    assert CONFIG.host == "127.0.0.1" and CONFIG.port in (7497, 4002) and CONFIG.client_id >= 1
    assert CONFIG.contract.symbol == "NQ" and CONFIG.contract.exchange == "CME" and len(CONFIG.contract.last_trade_month) == 6
    assert CONFIG.paper_only is True


def test_refuses_a_live_account_and_a_live_flag() -> None:
    with pytest.raises(IBKRRefused, match="paper"):
        IBKRBroker(FakeIB(accounts=("U1234567",)), CONFIG, live_execution_allowed=False)
    with pytest.raises(IBKRRefused, match="live_execution_allowed"):
        IBKRBroker(FakeIB(), CONFIG, live_execution_allowed=True)


def test_snapshot_maps_account_positions_and_open_orders() -> None:
    ib = FakeIB()
    ib._positions = [SimpleNamespace(account="DU1234567", contract=SimpleNamespace(symbol="NQ", localSymbol="NQH2"), position=-2.0, avgCost=16387.5 * 20)]
    b = broker(ib)
    b.submit_bracket(BracketIntent("EP:sig", "NQ", "SELL", 2, 16387.5, 16411.5, 16350.75, "sig"), T)
    snap = b.snapshot(T)
    assert snap.account_id == "DU1234567" and snap.equity == 250000.5 and snap.available_funds == 200000.0 and snap.buying_power == 1000000.0
    assert snap.net_position("NQ") == -2 and snap.positions[0].average_price == 16387.5
    assert [o.role for o in snap.open_orders] == [OrderRole.ENTRY, OrderRole.TARGET, OrderRole.STOP]
    assert snap.open_entry_orders()[0].client_ref == "EP:sig" and snap.source == "ibkr"


def test_bracket_is_three_orders_with_the_right_shape() -> None:
    ib = FakeIB()
    b = broker(ib)
    entry = b.submit_bracket(BracketIntent("EP:sig", "NQ", "BUY", 1, 16350.0, 16330.0, 16390.0, "sig"), T)
    assert entry.role is OrderRole.ENTRY and entry.status is OrderStatus.SUBMITTED and entry.side == "BUY" and entry.limit_price == 16350.0
    orders = [order for _, order in ib.placed]
    assert [o.orderType for o in orders] == ["LMT", "LMT", "STP"] and [o.action for o in orders] == ["BUY", "SELL", "SELL"]
    assert orders[1].lmtPrice == 16390.0 and orders[2].auxPrice == 16330.0
    assert all(o.orderRef == "EP:sig" and o.tif == "GTC" for o in orders) and orders[-1].transmit is True
    contract = ib.placed[0][0]
    assert contract.symbol == "NQ" and contract.exchange == "CME" and contract.lastTradeDateOrContractMonth == CONFIG.contract.last_trade_month


def test_poll_diffs_statuses_into_events_and_cancel_cancels_the_entry() -> None:
    ib = FakeIB()
    b = broker(ib)
    entry = b.submit_bracket(BracketIntent("EP:sig", "NQ", "SELL", 2, 16387.5, 16411.5, 16350.75, "sig"), T)
    ib.set_status(int(entry.order_id), "Submitted")
    events = b.poll(T, None)
    assert [e.kind for e in events] == ["working"] and events[0].order.status is OrderStatus.WORKING
    assert b.poll(T, None) == (), "nothing changed, nothing reported"
    ib.set_status(int(entry.order_id), "Filled", filled=2.0, avg=16387.5)
    events = b.poll(T + pd.Timedelta(minutes=1), None)
    assert [e.kind for e in events] == ["filled"] and events[0].fill is not None and events[0].fill.price == 16387.5 and events[0].fill.quantity == 2
    b.cancel(entry.order_id, T)
    assert ib.cancelled == [int(entry.order_id)]


def test_poll_and_snapshot_raise_when_the_socket_is_gone() -> None:
    ib = FakeIB()
    b = broker(ib)
    b.submit_bracket(BracketIntent("EP:sig", "NQ", "SELL", 2, 16387.5, 16411.5, 16350.75, "sig"), T)
    ib.connected = False
    with pytest.raises(IBKRDisconnected):
        b.poll(T, None)
    with pytest.raises(IBKRDisconnected):
        b.snapshot(T)


def flat_snapshot(**overrides) -> AccountSnapshot:
    base = dict(account_id="DU1", asof=T, equity=1.0, available_funds=1.0, buying_power=1.0, source="ibkr")
    return AccountSnapshot(**{**base, **overrides})


def test_require_flat_names_what_is_in_the_way() -> None:
    assert require_flat(flat_snapshot(), "NQ") is None
    with pytest.raises(IBKRRefused, match="position"):
        require_flat(flat_snapshot(positions=(Position("NQ", -2, 16387.5),)), "NQ")
    entry = OrderState("7", "x", OrderRole.ENTRY, "BUY", 1, 16350.0, None, 0, None, OrderStatus.WORKING, T, T)
    with pytest.raises(IBKRRefused, match="entry order"):
        require_flat(flat_snapshot(open_orders=(entry,)), "NQ")
    # another contract's position is not this run's business
    assert require_flat(flat_snapshot(positions=(Position("ES", 1, 4700.0),)), "NQ") is None
