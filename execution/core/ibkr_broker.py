"""The IBKR paper adapter over ``ib_async`` (TWS / IB Gateway socket API).

TWS is already logged in; this adapter only opens the local socket, so no
credential exists anywhere in the repository.  It refuses to construct unless
the session's managed accounts are paper accounts (ids starting with ``DU``)
and ``live_execution_allowed`` is false — what "live" would mean is a later
phase.  A dropped socket is an error (``IBKRDisconnected``) on the next
``snapshot`` or ``poll``, never a silently stale cache; the GTC bracket
stays at TWS and there is no automatic reconnect or recovery, which is why
``require_flat`` refuses to start a run over an existing position or entry
order.  ``ib`` is injected so the tests drive a fake; ``IBKRBroker.connect``
is the one place ``ib_async`` is imported."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from contract.execution import (
    AccountSnapshot,
    BracketIntent,
    BrokerEvent,
    Fill,
    OrderRole,
    OrderState,
    OrderStatus,
    Position,
)
from contract.market.primitives import Bar

IBKR_SCHEMA_VERSION = 1
PAPER_PREFIX = "DU"
# ib_async / TWS order statuses → ours.
_STATUS = {
    "PendingSubmit": OrderStatus.SUBMITTED,
    "ApiPending": OrderStatus.SUBMITTED,
    "PreSubmitted": OrderStatus.SUBMITTED,
    "Submitted": OrderStatus.WORKING,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "ApiCancelled": OrderStatus.CANCELLED,
    "Inactive": OrderStatus.REJECTED,
}


class IBKRRefused(RuntimeError):
    """The adapter will not run against this session."""


class IBKRDisconnected(RuntimeError):
    """The socket to TWS is gone; nothing read from ib_async is current."""


def require_flat(snapshot: AccountSnapshot, symbol: str) -> None:
    """Refuse to start over what this run cannot manage: a position or a
    working entry order in the contract that predates the machine."""
    positions = [item for item in snapshot.positions if item.symbol == symbol]
    entries = snapshot.open_entry_orders()
    if positions:
        raise IBKRRefused(f"account {snapshot.account_id} holds a position in {symbol}: {[p.to_dict() for p in positions]}; recovery is not implemented")
    if entries:
        raise IBKRRefused(f"account {snapshot.account_id} has an entry order working: {[o.order_id for o in entries]}; recovery is not implemented")


@dataclass(frozen=True)
class IBKRContract:
    symbol: str
    exchange: str
    currency: str
    last_trade_month: str  # YYYYMM of the front month the run trades


@dataclass(frozen=True)
class IBKRConfig:
    host: str
    port: int
    client_id: int
    paper_only: bool
    poll_timeout_s: float
    contract: IBKRContract

    @classmethod
    def from_json(cls, path: Path) -> "IBKRConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != IBKR_SCHEMA_VERSION:
            raise ValueError("unsupported ibkr schema_version")
        contract = payload["contract"]
        return cls(
            host=str(payload["host"]), port=int(payload["port"]), client_id=int(payload["client_id"]),
            paper_only=bool(payload["paper_only"]), poll_timeout_s=float(payload["poll_timeout_s"]),
            contract=IBKRContract(
                symbol=str(contract["symbol"]), exchange=str(contract["exchange"]), currency=str(contract["currency"]),
                last_trade_month=str(contract["last_trade_month"]),
            ),
        )


def _ts(value: Any, fallback: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return fallback
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class IBKRBroker:
    paper = True

    def __init__(self, ib: Any, config: IBKRConfig, *, live_execution_allowed: bool) -> None:
        if live_execution_allowed:
            raise IBKRRefused("live_execution_allowed is true; this adapter only runs while it is false")
        accounts = list(ib.managedAccounts())
        if not accounts or any(not str(account).startswith(PAPER_PREFIX) for account in accounts):
            raise IBKRRefused(f"only paper accounts ({PAPER_PREFIX}…) are accepted; session has {accounts}")
        if not config.paper_only:
            raise IBKRRefused("execution/configs/ibkr.json must keep paper_only true")
        self._ib = ib
        self._config = config
        self.account_id = str(accounts[0])
        self._contract = self._make_contract()
        # order id → (OrderState as last reported, client_ref)
        self._orders: dict[str, OrderState] = {}
        self._children: dict[str, tuple[str, str]] = {}  # entry id → (target id, stop id)
        self._fills_seen: set[tuple[str, str, float, float]] = set()

    @classmethod
    def connect(cls, config: IBKRConfig, *, live_execution_allowed: bool) -> "IBKRBroker":
        from ib_async import IB  # the only import of the dependency

        ib = IB()
        ib.connect(config.host, config.port, clientId=config.client_id, readonly=False)
        return cls(ib, config, live_execution_allowed=live_execution_allowed)

    def _make_contract(self) -> Any:
        spec = self._config.contract
        try:
            from ib_async import Future
        except ImportError:  # the tests run without ib_async; a plain object carries the same fields
            from types import SimpleNamespace as Future  # type: ignore[assignment]
        return Future(symbol=spec.symbol, lastTradeDateOrContractMonth=spec.last_trade_month, exchange=spec.exchange, currency=spec.currency)

    # ------------------------------------------------------------ mapping

    def _order_state(self, trade: Any, asof: pd.Timestamp, known: OrderState | None) -> OrderState:
        order, status = trade.order, trade.orderStatus
        order_id = str(order.orderId)
        if order.parentId:
            role = OrderRole.STOP if order.orderType == "STP" else OrderRole.TARGET
        else:
            role = OrderRole.FLATTEN if order.orderType == "MKT" else OrderRole.ENTRY
        filled = int(round(float(status.filled or 0.0)))
        avg = float(status.avgFillPrice or 0.0)
        mapped = _STATUS.get(str(status.status), OrderStatus.WORKING)
        if mapped in (OrderStatus.WORKING, OrderStatus.SUBMITTED) and 0 < filled < int(order.totalQuantity):
            mapped = OrderStatus.PARTIAL
        return OrderState(
            order_id=order_id, client_ref=str(order.orderRef or (known.client_ref if known else order_id)), role=role,
            side=str(order.action), quantity=int(order.totalQuantity),
            limit_price=float(order.lmtPrice) if order.orderType == "LMT" and order.lmtPrice else None,
            stop_price=float(order.auxPrice) if order.orderType == "STP" and order.auxPrice else None,
            filled_quantity=filled, average_fill_price=avg if avg > 0 else None, status=mapped,
            submitted_at=known.submitted_at if known else asof, updated_at=asof,
            parent_id=str(order.parentId) if order.parentId else None,
        )

    def _require_connected(self) -> None:
        if not self._ib.isConnected():
            raise IBKRDisconnected(f"lost the connection to TWS at {self._config.host}:{self._config.port}")

    # ------------------------------------------------------------ protocol

    def snapshot(self, asof: pd.Timestamp) -> AccountSnapshot:
        self._require_connected()
        asof = pd.Timestamp(asof).tz_convert("UTC")
        summary = {str(item.tag): item for item in self._ib.accountSummary(self.account_id)}

        def money(tag: str) -> float:
            item = summary.get(tag)
            return float(item.value) if item is not None else 0.0

        symbol = self._config.contract.symbol
        positions = []
        for item in self._ib.positions(self.account_id):
            if str(item.contract.symbol) != symbol or not item.position:
                continue
            quantity = int(round(float(item.position)))
            # ib_async reports avgCost per contract in currency (price × multiplier)
            multiplier = float(getattr(item.contract, "multiplier", 0) or 0) or 20.0
            positions.append(Position(symbol, quantity, float(item.avgCost) / multiplier))
        open_orders, cancelled = [], []
        for trade in self._ib.trades():
            order_id = str(trade.order.orderId)
            state = self._order_state(trade, asof, self._orders.get(order_id))
            if state.is_open:
                open_orders.append(state)
            elif state.status in (OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
                cancelled.append(state)
        fills = tuple(
            Fill(str(item.execution.orderId), int(round(float(item.execution.shares))), float(item.execution.price), _ts(item.execution.time, asof))
            for item in self._ib.fills()
            if str(item.execution.orderId) in self._orders
        )
        return AccountSnapshot(
            account_id=self.account_id, asof=asof, equity=money("NetLiquidation"), available_funds=money("AvailableFunds"),
            buying_power=money("BuyingPower"), positions=tuple(positions), open_orders=tuple(open_orders), fills=fills,
            cancelled=tuple(cancelled), source="ibkr",
        )

    def submit_bracket(self, intent: BracketIntent, asof: pd.Timestamp) -> OrderState:
        asof = pd.Timestamp(asof).tz_convert("UTC")
        bracket = self._ib.bracketOrder(
            intent.side, intent.quantity, intent.limit_price, intent.target_price, intent.stop_price,
            orderRef=intent.client_ref, tif="GTC",
        )
        trades = [self._ib.placeOrder(self._contract, order) for order in (bracket.parent, bracket.takeProfit, bracket.stopLoss)]
        states = [self._order_state(trade, asof, None) for trade in trades]
        for state in states:
            self._orders[state.order_id] = state
        self._children[states[0].order_id] = (states[1].order_id, states[2].order_id)
        return states[0]

    def flatten(self, symbol: str, quantity: int, side: str, asof: pd.Timestamp, client_ref: str) -> OrderState:
        """A market order in the contract: the close-beyond exit and the halt
        (the order machine), and the paper exercise's way out of a position
        it opened on purpose."""
        asof = pd.Timestamp(asof).tz_convert("UTC")
        if symbol != self._config.contract.symbol:
            raise ValueError(f"flatten {symbol!r}: this broker trades {self._config.contract.symbol}")
        try:
            from ib_async import MarketOrder
            order = MarketOrder(side, quantity, orderRef=client_ref, tif="DAY")
        except ImportError:  # the tests' fake assigns the id at placement
            from types import SimpleNamespace
            order = SimpleNamespace(orderId=0, action=side, totalQuantity=quantity, orderType="MKT", lmtPrice=0.0, auxPrice=0.0, parentId=0, orderRef=client_ref, tif="DAY", transmit=True)
        trade = self._ib.placeOrder(self._contract, order)
        state = self._order_state(trade, asof, None)
        self._orders[state.order_id] = state
        return state

    def cancel(self, order_id: str, asof: pd.Timestamp) -> None:
        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                return
        raise ValueError(f"order {order_id!r} is not known to this session")

    def poll(self, asof: pd.Timestamp, bar: Bar | None) -> tuple[BrokerEvent, ...]:
        self._require_connected()
        asof = pd.Timestamp(asof).tz_convert("UTC")
        self._ib.waitOnUpdate(self._config.poll_timeout_s)
        fills_by_order: dict[str, list[Any]] = {}
        for item in self._ib.fills():
            fills_by_order.setdefault(str(item.execution.orderId), []).append(item)
        events: list[BrokerEvent] = []
        for trade in self._ib.trades():
            order_id = str(trade.order.orderId)
            known = self._orders.get(order_id)
            if known is None:
                continue  # not placed by this session
            state = self._order_state(trade, asof, known)
            if state.status is known.status and state.filled_quantity == known.filled_quantity:
                continue
            fill = None
            for item in fills_by_order.get(order_id, ()):
                key = (order_id, str(item.execution.time), float(item.execution.shares), float(item.execution.price))
                if key in self._fills_seen:
                    continue
                self._fills_seen.add(key)
                fill = Fill(order_id, int(round(float(item.execution.shares))), float(item.execution.price), _ts(item.execution.time, asof))
            self._orders[order_id] = state
            events.append(BrokerEvent(state.status.value, state, fill, asof))
        return tuple(events)


__all__ = [
    "IBKR_SCHEMA_VERSION",
    "PAPER_PREFIX",
    "IBKRBroker",
    "IBKRConfig",
    "IBKRContract",
    "IBKRDisconnected",
    "IBKRRefused",
    "require_flat",
]
