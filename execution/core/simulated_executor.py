"""The local simulated executor: the IBKR account and matching replaced by a
virtual account and fixed OHLCV rules, behind the same ``Broker`` interface
the order machine drives against TWS.

Matching is an approximation at bar level — no bid / ask, no queue, no
market microstructure:

- an entry limit works from the bar after its submission;
- ``BUY LIMIT`` fills when a later bar's ``low <= limit``, ``SELL LIMIT``
  when ``high >= limit``, at the limit price, ``max_fill_per_bar`` contracts
  per touching bar (``None``: the whole remaining quantity — ``partial``
  events until the last bar, then ``filled``);
- the stop and the target work from the bar after the first fill, for the
  filled quantity; a bar that touches both fills the stop (conservative);
  the other leg is cancelled, and so is any unfilled remainder of the entry;
- ``submit_bracket`` accepts every intent; the next ``poll`` reports
  ``rejected`` when ``quantity × margin_per_contract`` exceeds the
  available funds — asynchronously, as TWS does;
- ``cancel`` takes effect on the next ``poll``; a partial keeps its filled
  part as a position with the stop and target still working;
- realized PnL moves the account's cash; ``equity`` is cash plus the open
  positions marked at the last polled close of their symbol (2026-09-17);
  available funds are cash less the margin held for open and working
  contracts;
- ``cancel`` also accepts a stop or a target of a filled bracket (the leg
  is cancelled on the next poll; a position whose both exits are cancelled
  stays open with none, as at TWS);
- ``flatten`` submits a market order (``OrderRole.FLATTEN``) that fills on
  the next poll at that bar's open, before any limit or stop of the bar is
  matched.

The same bars give the same events, so a ``sim`` journal replays.  There is
no in-place order modification here or at IBKR: the order machine replaces
an order by cancelling it and submitting a new one."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
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

SIMULATOR_SCHEMA_VERSION = 1
_CANCELLED = (OrderStatus.CANCELLED, OrderStatus.EXPIRED)


@dataclass(frozen=True)
class SimulatorConfig:
    initial_equity: float
    margin_per_contract: float  # USD held per open or working contract
    max_fill_per_bar: int | None  # None: the whole remaining quantity fills on a touch
    sha256: str

    @classmethod
    def from_json(cls, path: Path) -> "SimulatorConfig":
        raw = Path(path).read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("schema_version") != SIMULATOR_SCHEMA_VERSION:
            raise ValueError("unsupported simulated_executor schema_version")
        per_bar = payload["max_fill_per_bar"]
        config = cls(
            initial_equity=float(payload["initial_equity"]),
            margin_per_contract=float(payload["margin_per_contract"]),
            max_fill_per_bar=None if per_bar is None else int(per_bar),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        if config.initial_equity <= 0.0 or config.margin_per_contract < 0.0 or (config.max_fill_per_bar is not None and config.max_fill_per_bar < 1):
            raise ValueError("simulated_executor config values out of range")
        return config


class VirtualAccount:
    """Cash, positions and every order the executor ever accepted, in
    submission order and at its latest state."""

    def __init__(self, cash: float) -> None:
        self.cash = float(cash)
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, OrderState] = {}
        self.fills: list[Fill] = []
        self.last_price: dict[str, float] = {}  # the last polled close per symbol

    def equity(self, point_value: float) -> float:
        """Cash plus the open positions marked at the last polled close (cash
        alone for a symbol never polled)."""
        marked = self.cash
        for symbol, position in self.positions.items():
            last = self.last_price.get(symbol)
            if last is not None:
                marked += (last - position.average_price) * position.quantity * point_value
        return marked

    def record(self, order: OrderState) -> OrderState:
        self.orders[order.order_id] = order
        return order

    def _with_status(self, *statuses: OrderStatus) -> tuple[OrderState, ...]:
        return tuple(order for order in self.orders.values() if order.status in statuses)

    def pending(self) -> tuple[OrderState, ...]:
        return tuple(order for order in self.orders.values() if order.is_open)

    def filled(self) -> tuple[OrderState, ...]:
        return self._with_status(OrderStatus.FILLED)

    def cancelled(self) -> tuple[OrderState, ...]:
        return self._with_status(*_CANCELLED)

    def rejected(self) -> tuple[OrderState, ...]:
        return self._with_status(OrderStatus.REJECTED)

    def margin_held(self, margin_per_contract: float) -> float:
        contracts = sum(abs(position.quantity) for position in self.positions.values())
        contracts += sum(order.remaining for order in self.pending() if order.role is OrderRole.ENTRY)
        return contracts * margin_per_contract

    def available_funds(self, margin_per_contract: float) -> float:
        return self.cash - self.margin_held(margin_per_contract)

    def apply_fill(self, symbol: str, side: str, quantity: int, price: float, point_value: float) -> None:
        """Move the position; realize PnL on the part that closes."""
        signed = quantity if side == "BUY" else -quantity
        current = self.positions.get(symbol)
        if current is None:
            self.positions[symbol] = Position(symbol, signed, price)
            return
        new_quantity = current.quantity + signed
        if (current.quantity > 0) != (signed > 0):
            closed = min(abs(signed), abs(current.quantity))
            direction = 1.0 if current.quantity > 0 else -1.0
            self.cash += direction * (price - current.average_price) * closed * point_value
        if new_quantity == 0:
            del self.positions[symbol]
        elif (current.quantity > 0) == (signed > 0):
            # adding to the position: the average price is the contract-weighted one
            average = (current.average_price * abs(current.quantity) + price * abs(signed)) / abs(new_quantity)
            self.positions[symbol] = Position(symbol, new_quantity, average)
        elif (new_quantity > 0) == (current.quantity > 0):
            self.positions[symbol] = Position(symbol, new_quantity, current.average_price)  # reduced, not flipped
        else:
            self.positions[symbol] = Position(symbol, new_quantity, price)  # flipped: the remainder opened at this price

    def summary(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": [self.positions[key].to_dict() for key in sorted(self.positions)],
            "orders": {
                "pending": len(self.pending()), "filled": len(self.filled()),
                "cancelled": len(self.cancelled()), "rejected": len(self.rejected()),
            },
            "fills": len(self.fills),
        }


@dataclass
class _Bracket:
    entry: OrderState
    stop: OrderState
    target: OrderState
    symbol: str
    entry_live_from: pd.Timestamp  # the bar-end of the submission; works after it
    reject_reason: str | None = None
    first_fill_at: pd.Timestamp | None = None  # the exits work after it
    cancel_requested: bool = False
    cancel_exits: set[str] = field(default_factory=set)  # "stop" / "target" legs whose cancel was requested


@dataclass
class _Flatten:
    order: OrderState
    symbol: str


class SimulatedExecutor:
    paper = True

    def __init__(
        self, config: SimulatorConfig, *, tick_size: float, point_value: float, equity: float | None = None, account_id: str = "SIM"
    ) -> None:
        self.config = config
        self.tick_size = float(tick_size)
        self.point_value = float(point_value)
        self.account_id = account_id
        self.account = VirtualAccount(config.initial_equity if equity is None else equity)
        self._brackets: dict[str, _Bracket] = {}
        self._flattens: list[_Flatten] = []
        self._sequence = 0

    # ------------------------------------------------------------ helpers

    def _next_id(self) -> str:
        self._sequence += 1
        return f"sim-{self._sequence}"

    @staticmethod
    def _touched(bar: Bar, price: float, side: str) -> bool:
        """A BUY limit / SELL stop fills when the bar trades at or below the
        price; a SELL limit / BUY stop when at or above."""
        return float(bar.low) <= price if side == "BUY" else float(bar.high) >= price

    def _set(self, bracket: _Bracket, leg: str, order: OrderState) -> OrderState:
        setattr(bracket, leg, order)
        return self.account.record(order)

    def _fill(self, order: OrderState, quantity: int, price: float, asof: pd.Timestamp) -> tuple[OrderState, Fill]:
        filled_quantity = order.filled_quantity + quantity
        status = OrderStatus.FILLED if filled_quantity == order.quantity else OrderStatus.PARTIAL
        updated = replace(order, filled_quantity=filled_quantity, average_fill_price=price, status=status, updated_at=asof)
        fill = Fill(order.order_id, quantity, price, asof)
        self.account.fills.append(fill)
        return updated, fill

    # ------------------------------------------------------------ protocol

    def snapshot(self, asof: pd.Timestamp) -> AccountSnapshot:
        account = self.account
        margin = self.config.margin_per_contract
        available = account.available_funds(margin)
        return AccountSnapshot(
            account_id=self.account_id, asof=asof, equity=account.equity(self.point_value), available_funds=available, buying_power=available,
            positions=tuple(account.positions[key] for key in sorted(account.positions)),
            open_orders=account.pending(), fills=tuple(account.fills),
            cancelled=tuple(order for order in account.orders.values() if order.status in _CANCELLED or order.status is OrderStatus.REJECTED),
            source="simulated",
        )

    def submit_bracket(self, intent: BracketIntent, asof: pd.Timestamp) -> OrderState:
        asof = pd.Timestamp(asof).tz_convert("UTC")
        margin = self.config.margin_per_contract
        required = intent.quantity * margin
        available = self.account.available_funds(margin)
        reject = None if required <= available else f"margin {required:.2f} exceeds available funds {available:.2f}"
        entry_id = self._next_id()
        exit_side = "SELL" if intent.side == "BUY" else "BUY"
        entry = OrderState(entry_id, intent.client_ref, OrderRole.ENTRY, intent.side, intent.quantity, intent.limit_price, None, 0, None, OrderStatus.SUBMITTED, asof, asof)
        stop = OrderState(self._next_id(), intent.client_ref, OrderRole.STOP, exit_side, intent.quantity, None, intent.stop_price, 0, None, OrderStatus.SUBMITTED, asof, asof, entry_id)
        target = OrderState(self._next_id(), intent.client_ref, OrderRole.TARGET, exit_side, intent.quantity, intent.target_price, None, 0, None, OrderStatus.SUBMITTED, asof, asof, entry_id)
        for order in (entry, stop, target):
            self.account.record(order)
        self._brackets[entry_id] = _Bracket(entry, stop, target, intent.symbol, entry_live_from=asof, reject_reason=reject)
        return entry

    def cancel(self, order_id: str, asof: pd.Timestamp) -> None:
        bracket = self._brackets.get(order_id)
        if bracket is not None and bracket.entry.is_open:
            bracket.cancel_requested = True
            return
        for bracket in self._brackets.values():
            for leg in ("stop", "target"):
                order = getattr(bracket, leg)
                if order.order_id == order_id and order.is_open:
                    bracket.cancel_exits.add(leg)
                    return
        raise ValueError(f"order {order_id!r} is not an open entry, stop or target")

    def flatten(self, symbol: str, quantity: int, side: str, asof: pd.Timestamp, client_ref: str) -> OrderState:
        """A market order that closes ``quantity`` contracts; it fills on the
        next poll at that bar's open."""
        asof = pd.Timestamp(asof).tz_convert("UTC")
        if type(quantity) is not int or quantity < 1 or side not in ("BUY", "SELL"):
            raise ValueError("flatten needs a positive quantity and a BUY / SELL side")
        order = OrderState(self._next_id(), client_ref, OrderRole.FLATTEN, side, quantity, None, None, 0, None, OrderStatus.SUBMITTED, asof, asof)
        self.account.record(order)
        self._flattens.append(_Flatten(order, symbol))
        return order

    def _close_bracket(self, entry_id: str, bracket: _Bracket, asof: pd.Timestamp, *, children_status: OrderStatus) -> None:
        for leg in ("stop", "target"):
            order = getattr(bracket, leg)
            if order.is_open:
                self._set(bracket, leg, replace(order, status=children_status, updated_at=asof))
        del self._brackets[entry_id]

    def poll(self, asof: pd.Timestamp, bar: Bar | None) -> tuple[BrokerEvent, ...]:
        asof = pd.Timestamp(asof).tz_convert("UTC")
        events: list[BrokerEvent] = []
        if bar is not None:
            self.account.last_price[bar.symbol] = float(bar.close)
            # market orders first: they fill at the open, before the bar's range is matched
            for item in self._flattens:
                filled, fill = self._fill(item.order, item.order.quantity, float(bar.open), asof)
                self.account.record(filled)
                self.account.apply_fill(item.symbol, filled.side, filled.quantity, fill.price, self.point_value)
                events.append(BrokerEvent("filled", filled, fill, asof))
            self._flattens = []
        for entry_id in list(self._brackets):
            bracket = self._brackets[entry_id]
            entry = bracket.entry
            for leg in sorted(bracket.cancel_exits):
                order = getattr(bracket, leg)
                if order.is_open:
                    cancelled = self._set(bracket, leg, replace(order, status=OrderStatus.CANCELLED, updated_at=asof))
                    events.append(BrokerEvent("cancelled", cancelled, None, asof))
            bracket.cancel_exits = set()
            if not entry.is_open and not bracket.stop.is_open and not bracket.target.is_open:
                del self._brackets[entry_id]  # the position lives on with no exits
                continue
            if entry.status is OrderStatus.SUBMITTED:
                if bracket.reject_reason is not None:
                    rejected = self._set(bracket, "entry", replace(entry, status=OrderStatus.REJECTED, updated_at=asof))
                    events.append(BrokerEvent("rejected", rejected, None, asof))
                    self._close_bracket(entry_id, bracket, asof, children_status=OrderStatus.CANCELLED)
                    continue
                entry = self._set(bracket, "entry", replace(entry, status=OrderStatus.WORKING, updated_at=asof))
                events.append(BrokerEvent("working", entry, None, asof))
                if asof <= bracket.entry_live_from:
                    continue  # the submission bar never fills: the order was not yet working while it traded
            if bracket.cancel_requested and entry.is_open:
                entry = self._set(bracket, "entry", replace(entry, status=OrderStatus.CANCELLED, updated_at=asof))
                events.append(BrokerEvent("cancelled", entry, None, asof))
                if entry.filled_quantity == 0:
                    self._close_bracket(entry_id, bracket, asof, children_status=OrderStatus.CANCELLED)
                    continue
            if bar is None:
                continue
            # the exits, for the filled part, from the bar after the first fill
            if bracket.first_fill_at is not None and asof > bracket.first_fill_at and entry.filled_quantity > 0:
                stop, target = bracket.stop, bracket.target
                stop_hit = stop.is_open and self._touched(bar, float(stop.stop_price), "BUY" if stop.side == "SELL" else "SELL")
                target_hit = target.is_open and self._touched(bar, float(target.limit_price), target.side)
                if stop_hit or target_hit:
                    winner, loser, price = ("stop", "target", float(stop.stop_price)) if stop_hit else ("target", "stop", float(target.limit_price))
                    winning = getattr(bracket, winner)
                    filled, fill = self._fill(winning, winning.quantity, price, asof)
                    self._set(bracket, winner, filled)
                    self.account.apply_fill(bracket.symbol, filled.side, filled.quantity, price, self.point_value)
                    events.append(BrokerEvent("filled", filled, fill, asof))
                    losing = getattr(bracket, loser)
                    if losing.is_open:
                        cancelled = self._set(bracket, loser, replace(losing, status=OrderStatus.CANCELLED, updated_at=asof))
                        events.append(BrokerEvent("cancelled", cancelled, None, asof))
                    if entry.is_open:  # the bracket is done: its unfilled remainder goes with it
                        entry = self._set(bracket, "entry", replace(entry, status=OrderStatus.CANCELLED, updated_at=asof))
                        events.append(BrokerEvent("cancelled", entry, None, asof))
                    del self._brackets[entry_id]
                    continue
            # the entry, from the bar after its submission
            if entry.is_open and asof > bracket.entry_live_from and self._touched(bar, float(entry.limit_price), entry.side):
                per_bar = self.config.max_fill_per_bar
                quantity = entry.remaining if per_bar is None else min(entry.remaining, per_bar)
                limit = float(entry.limit_price)
                # a limit through the market (BUY at or above the open, SELL at or below) fills at the open
                price = min(limit, float(bar.open)) if entry.side == "BUY" else max(limit, float(bar.open))
                filled, fill = self._fill(entry, quantity, price, asof)
                entry = self._set(bracket, "entry", filled)
                if bracket.first_fill_at is None:
                    bracket.first_fill_at = asof
                self.account.apply_fill(bracket.symbol, filled.side, quantity, fill.price, self.point_value)
                events.append(BrokerEvent(filled.status.value, filled, fill, asof))
                for leg in ("stop", "target"):
                    order = getattr(bracket, leg)
                    sized = self._set(bracket, leg, replace(order, quantity=filled.filled_quantity, status=OrderStatus.WORKING, updated_at=asof))
                    events.append(BrokerEvent("working", sized, None, asof))
        return tuple(events)


__all__ = ["SIMULATOR_SCHEMA_VERSION", "SimulatedExecutor", "SimulatorConfig", "VirtualAccount"]
