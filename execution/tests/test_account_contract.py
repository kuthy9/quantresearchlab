from __future__ import annotations

import json

import pandas as pd
import pytest

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

T = pd.Timestamp("2022-01-03T14:12:00Z")


def order(order_id="o1", role=OrderRole.ENTRY, status=OrderStatus.WORKING, filled=0, quantity=2, parent=None) -> OrderState:
    return OrderState(
        order_id=order_id, client_ref="EP_1:abcd", role=role, side="SELL", quantity=quantity,
        limit_price=16387.5, stop_price=None if role is not OrderRole.STOP else 16411.5,
        filled_quantity=filled, average_fill_price=None if not filled else 16387.5, status=status,
        submitted_at=T, updated_at=T, parent_id=parent,
    )


def snapshot(**over) -> AccountSnapshot:
    base = dict(
        account_id="SIM", asof=T, equity=100_000.0, available_funds=90_000.0, buying_power=400_000.0,
        positions=(Position("NQ", -2, 16387.5),), open_orders=(order(), order("o2", OrderRole.STOP, parent="o1")),
        fills=(Fill("o1", 1, 16387.5, T),), cancelled=(order("o0", status=OrderStatus.CANCELLED),), source="test",
    )
    base.update(over)
    return AccountSnapshot(**base)


def test_every_type_round_trips_through_json() -> None:
    snap = snapshot()
    again = AccountSnapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
    assert again == snap
    intent = BracketIntent("EP_1:abcd", "NQ", "BUY", 1, 16380.0, 16360.0, 16420.0, "abcd")
    assert BracketIntent.from_dict(intent.to_dict()) == intent
    event = BrokerEvent("filled", order(filled=2, status=OrderStatus.FILLED), Fill("o1", 2, 16387.5, T), T)
    assert BrokerEvent.from_dict(json.loads(json.dumps(event.to_dict()))) == event


def test_order_invariants() -> None:
    with pytest.raises(ValueError, match="filled"):
        order(filled=3)
    with pytest.raises(ValueError, match="side"):
        OrderState("o", "r", OrderRole.ENTRY, "HOLD", 1, 1.0, None, 0, None, OrderStatus.WORKING, T, T, None)
    assert order(filled=1).remaining == 1
    assert order(filled=2, status=OrderStatus.FILLED).is_open is False and order().is_open is True


def test_open_entry_orders_are_only_live_entries() -> None:
    snap = snapshot(open_orders=(
        order("a"), order("b", status=OrderStatus.PARTIAL, filled=1), order("c", OrderRole.STOP, parent="a"),
        order("d", status=OrderStatus.FILLED, filled=2),
    ))
    assert [o.order_id for o in snap.open_entry_orders()] == ["a", "b"]
    assert snap.net_position("NQ") == -2 and snap.net_position("ES") == 0


def test_bracket_intent_prices_must_agree_with_side() -> None:
    with pytest.raises(ValueError, match="stop"):
        BracketIntent("r", "NQ", "BUY", 1, 16380.0, 16390.0, 16420.0, "s")  # stop above a BUY limit
    with pytest.raises(ValueError, match="target"):
        BracketIntent("r", "NQ", "SELL", 1, 16380.0, 16400.0, 16390.0, "s")  # target above a SELL limit
