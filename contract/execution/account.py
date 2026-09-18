"""The account and order facts a broker reports, as Risk and Execution read them.

``AccountSnapshot`` is what the executor gathers before the Risk gate runs:
equity and funds, the positions, the orders still open, the fills and the
cancellations it knows of.  ``OrderState`` is one order as the broker last
reported it; ``BracketIntent`` is the one thing the executor ever submits
(an entry limit with an attached stop and target); ``BrokerEvent`` is one
change the broker reported since the last poll.  Nothing here names a Brain
type: the boundary is prices, sides and quantities."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import pandas as pd

from contract.market.primitives import aware_timestamp

SIDES: frozenset[str] = frozenset({"BUY", "SELL"})


class OrderStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


OPEN_STATUSES: frozenset[OrderStatus] = frozenset({OrderStatus.SUBMITTED, OrderStatus.WORKING, OrderStatus.PARTIAL})


class OrderRole(str, Enum):
    ENTRY = "entry"
    STOP = "stop"
    TARGET = "target"
    # A market order that closes a position (the close-beyond exit, the halt).
    FLATTEN = "flatten"


def _iso(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _price(value: Any, *, name: str) -> float:
    price = float(value)
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError(f"{name} must be a positive finite price")
    return price


def _opt_price(value: Any, *, name: str) -> float | None:
    return None if value is None else _price(value, name=name)


def _quantity(value: Any, *, name: str, allow_zero: bool = False) -> int:
    if type(value) is not int or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} integer")
    return value


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: int  # signed: long > 0, short < 0
    average_price: float

    def __post_init__(self) -> None:
        if not self.symbol or type(self.quantity) is not int or self.quantity == 0:
            raise ValueError("position needs a symbol and a non-zero signed quantity")
        object.__setattr__(self, "average_price", _price(self.average_price, name="position.average_price"))

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "quantity": self.quantity, "average_price": self.average_price}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Position":
        return cls(str(payload["symbol"]), int(payload["quantity"]), float(payload["average_price"]))


@dataclass(frozen=True)
class OrderState:
    order_id: str
    client_ref: str
    role: OrderRole
    side: str
    quantity: int
    limit_price: float | None
    stop_price: float | None
    filled_quantity: int
    average_fill_price: float | None
    status: OrderStatus
    submitted_at: pd.Timestamp
    updated_at: pd.Timestamp
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.order_id or not self.client_ref:
            raise ValueError("order needs an order_id and a client_ref")
        object.__setattr__(self, "role", OrderRole(self.role))
        object.__setattr__(self, "status", OrderStatus(self.status))
        if self.side not in SIDES:
            raise ValueError("order side must be BUY or SELL")
        _quantity(self.quantity, name="order.quantity")
        _quantity(self.filled_quantity, name="order.filled_quantity", allow_zero=True)
        if self.filled_quantity > self.quantity:
            raise ValueError("order filled_quantity exceeds quantity")
        object.__setattr__(self, "limit_price", _opt_price(self.limit_price, name="order.limit_price"))
        object.__setattr__(self, "stop_price", _opt_price(self.stop_price, name="order.stop_price"))
        object.__setattr__(self, "average_fill_price", _opt_price(self.average_fill_price, name="order.average_fill_price"))
        object.__setattr__(self, "submitted_at", aware_timestamp(self.submitted_at, name="order.submitted_at"))
        object.__setattr__(self, "updated_at", aware_timestamp(self.updated_at, name="order.updated_at"))
        if self.updated_at < self.submitted_at:
            raise ValueError("order updated_at precedes submitted_at")

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "client_ref": self.client_ref,
            "role": self.role.value,
            "side": self.side,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "status": self.status.value,
            "submitted_at": _iso(self.submitted_at),
            "updated_at": _iso(self.updated_at),
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrderState":
        return cls(
            order_id=str(payload["order_id"]), client_ref=str(payload["client_ref"]), role=payload["role"],
            side=str(payload["side"]), quantity=int(payload["quantity"]), limit_price=payload.get("limit_price"),
            stop_price=payload.get("stop_price"), filled_quantity=int(payload["filled_quantity"]),
            average_fill_price=payload.get("average_fill_price"), status=payload["status"],
            submitted_at=pd.Timestamp(payload["submitted_at"]), updated_at=pd.Timestamp(payload["updated_at"]),
            parent_id=payload.get("parent_id"),
        )


@dataclass(frozen=True)
class Fill:
    order_id: str
    quantity: int
    price: float
    at: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("fill needs an order_id")
        _quantity(self.quantity, name="fill.quantity")
        object.__setattr__(self, "price", _price(self.price, name="fill.price"))
        object.__setattr__(self, "at", aware_timestamp(self.at, name="fill.at"))

    def to_dict(self) -> dict[str, Any]:
        return {"order_id": self.order_id, "quantity": self.quantity, "price": self.price, "at": _iso(self.at)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Fill":
        return cls(str(payload["order_id"]), int(payload["quantity"]), float(payload["price"]), pd.Timestamp(payload["at"]))


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    asof: pd.Timestamp
    equity: float
    available_funds: float
    buying_power: float
    positions: tuple[Position, ...] = ()
    open_orders: tuple[OrderState, ...] = ()
    fills: tuple[Fill, ...] = ()
    cancelled: tuple[OrderState, ...] = ()
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not self.account_id:
            raise ValueError("account snapshot needs an account_id")
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="account.asof"))
        for name in ("equity", "available_funds", "buying_power"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"account {name} must be finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "open_orders", tuple(self.open_orders))
        object.__setattr__(self, "fills", tuple(self.fills))
        object.__setattr__(self, "cancelled", tuple(self.cancelled))

    def open_entry_orders(self) -> tuple[OrderState, ...]:
        return tuple(order for order in self.open_orders if order.role is OrderRole.ENTRY and order.is_open)

    def net_position(self, symbol: str) -> int:
        return sum(position.quantity for position in self.positions if position.symbol == symbol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "asof": _iso(self.asof),
            "equity": self.equity,
            "available_funds": self.available_funds,
            "buying_power": self.buying_power,
            "positions": [item.to_dict() for item in self.positions],
            "open_orders": [item.to_dict() for item in self.open_orders],
            "fills": [item.to_dict() for item in self.fills],
            "cancelled": [item.to_dict() for item in self.cancelled],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AccountSnapshot":
        return cls(
            account_id=str(payload["account_id"]), asof=pd.Timestamp(payload["asof"]), equity=float(payload["equity"]),
            available_funds=float(payload["available_funds"]), buying_power=float(payload["buying_power"]),
            positions=tuple(Position.from_dict(item) for item in payload.get("positions", ())),
            open_orders=tuple(OrderState.from_dict(item) for item in payload.get("open_orders", ())),
            fills=tuple(Fill.from_dict(item) for item in payload.get("fills", ())),
            cancelled=tuple(OrderState.from_dict(item) for item in payload.get("cancelled", ())),
            source=str(payload.get("source", "unknown")),
        )


@dataclass(frozen=True)
class BracketIntent:
    """An entry limit with its attached stop and target: the only thing the
    executor submits.  ``signature`` is the plan's identity across bars."""

    client_ref: str
    symbol: str
    side: str
    quantity: int
    limit_price: float
    stop_price: float
    target_price: float
    signature: str

    def __post_init__(self) -> None:
        if not self.client_ref or not self.symbol or not self.signature:
            raise ValueError("bracket intent needs a client_ref, a symbol and a signature")
        if self.side not in SIDES:
            raise ValueError("bracket side must be BUY or SELL")
        _quantity(self.quantity, name="bracket.quantity")
        for name in ("limit_price", "stop_price", "target_price"):
            object.__setattr__(self, name, _price(getattr(self, name), name=f"bracket.{name}"))
        long = self.side == "BUY"
        if (long and not self.stop_price < self.limit_price) or (not long and not self.stop_price > self.limit_price):
            raise ValueError("bracket stop must lie on the losing side of the limit")
        if (long and not self.target_price > self.limit_price) or (not long and not self.target_price < self.limit_price):
            raise ValueError("bracket target must lie on the winning side of the limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_ref": self.client_ref,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BracketIntent":
        return cls(
            str(payload["client_ref"]), str(payload["symbol"]), str(payload["side"]), int(payload["quantity"]),
            float(payload["limit_price"]), float(payload["stop_price"]), float(payload["target_price"]), str(payload["signature"]),
        )


@dataclass(frozen=True)
class BrokerEvent:
    """One change the broker reported: ``kind`` is the order's new status
    (``working`` / ``partial`` / ``filled`` / ``cancelled`` / ``expired`` /
    ``rejected``), ``fill`` the execution behind it when there was one."""

    kind: str
    order: OrderState
    fill: Fill | None
    at: pd.Timestamp

    def __post_init__(self) -> None:
        if self.kind not in {status.value for status in OrderStatus}:
            raise ValueError(f"broker event kind {self.kind!r} is not an order status")
        object.__setattr__(self, "at", aware_timestamp(self.at, name="event.at"))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "order": self.order.to_dict(), "fill": None if self.fill is None else self.fill.to_dict(), "at": _iso(self.at)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BrokerEvent":
        fill = payload.get("fill")
        return cls(str(payload["kind"]), OrderState.from_dict(payload["order"]), None if fill is None else Fill.from_dict(fill), pd.Timestamp(payload["at"]))


__all__ = [
    "OPEN_STATUSES",
    "SIDES",
    "AccountSnapshot",
    "BracketIntent",
    "BrokerEvent",
    "Fill",
    "OrderRole",
    "OrderState",
    "OrderStatus",
    "Position",
]
