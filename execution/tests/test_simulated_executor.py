from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from contract.execution import BracketIntent, OrderRole, OrderStatus
from contract.market.primitives import Bar
from execution.core.simulated_executor import SimulatedExecutor, SimulatorConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG = SimulatorConfig.from_json(ROOT / "execution" / "configs" / "simulated_executor.json")
T0 = pd.Timestamp("2022-01-03T14:12:00Z")


def bar(minute: int, low: float, high: float, close: float | None = None) -> Bar:
    close = (low + high) / 2 if close is None else close
    return Bar(start=T0 + pd.Timedelta(minutes=minute), open=close, high=high, low=low, close=close, volume=100.0, symbol="NQ", instrument_id=1)


def at(minute: int) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minute + 1)


def short(quantity: int = 2) -> BracketIntent:
    return BracketIntent("EP_1:sig", "NQ", "SELL", quantity, 16387.5, 16411.5, 16350.75, "sig")


def long() -> BracketIntent:
    return BracketIntent("EP_1:sig", "NQ", "BUY", 1, 16350.0, 16330.0, 16390.0, "sig")


def executor(**overrides) -> SimulatedExecutor:
    config = CONFIG if not overrides else SimulatorConfig(**{**CONFIG.__dict__, **overrides})
    return SimulatedExecutor(config, tick_size=0.25, point_value=20.0)


def run(broker: SimulatedExecutor, intent: BracketIntent, bars) -> list:
    events = []
    entry = broker.submit_bracket(intent, at(0))
    assert entry.status is OrderStatus.SUBMITTED and entry.role is OrderRole.ENTRY
    for minute, b in enumerate(bars, start=1):
        events.extend(broker.poll(at(minute), b))
    return events


def test_config_starts_the_account_at_one_hundred_thousand() -> None:
    assert CONFIG.initial_equity == 100_000.0 and CONFIG.margin_per_contract > 0 and CONFIG.max_fill_per_bar is None
    ex = executor()
    snap = ex.snapshot(at(0))
    assert snap.equity == 100_000.0 and snap.available_funds == 100_000.0 and snap.source == "simulated" and ex.paper is True
    assert SimulatedExecutor(CONFIG, tick_size=0.25, point_value=20.0, equity=50_000.0).account.cash == 50_000.0


def test_entry_works_from_the_next_bar_and_fills_at_the_limit_when_touched() -> None:
    ex = executor()
    events = run(ex, short(), [bar(1, 16370.0, 16380.0), bar(2, 16375.0, 16390.0)])
    kinds = [(e.kind, e.order.role.value) for e in events]
    assert kinds[:2] == [("working", "entry"), ("filled", "entry")]
    fill = next(e for e in events if e.kind == "filled")
    assert fill.fill is not None and fill.fill.price == 16387.5 and fill.fill.quantity == 2
    snap = ex.snapshot(at(2))
    assert snap.net_position("NQ") == -2 and [o.role for o in snap.open_orders] == [OrderRole.STOP, OrderRole.TARGET]


def test_no_fill_on_the_submission_bar_itself() -> None:
    ex = executor()
    ex.submit_bracket(short(), at(0))
    events = ex.poll(at(0), bar(0, 16380.0, 16395.0))
    assert [e.kind for e in events] == ["working"]


def test_a_buy_limit_needs_the_low_at_or_under_it() -> None:
    ex = executor()
    events = run(ex, long(), [bar(1, 16350.25, 16360.0), bar(2, 16350.0, 16360.0)])
    fills = [e for e in events if e.kind == "filled"]
    assert len(fills) == 1 and fills[0].at == at(2) and fills[0].fill.price == 16350.0


def test_long_mirrors_and_the_target_closes_the_position_with_pnl() -> None:
    ex = executor()
    events = run(ex, long(), [bar(1, 16345.0, 16360.0), bar(2, 16355.0, 16370.0), bar(3, 16380.0, 16395.0)])
    kinds = [(e.kind, e.order.role.value) for e in events]
    assert ("filled", "entry") in kinds and ("filled", "target") in kinds and ("cancelled", "stop") in kinds
    snap = ex.snapshot(at(3))
    assert snap.positions == () and snap.open_orders == ()
    assert snap.equity == pytest.approx(100_000.0 + (16390.0 - 16350.0) * 1 * 20.0)
    assert len(snap.fills) == 2 and len(snap.cancelled) == 1


def test_stop_wins_when_a_bar_touches_both() -> None:
    ex = executor()
    events = run(ex, short(), [bar(1, 16385.0, 16390.0), bar(2, 16340.0, 16420.0)])
    kinds = [(e.kind, e.order.role.value) for e in events]
    assert ("filled", "stop") in kinds and ("cancelled", "target") in kinds and ("filled", "target") not in kinds
    assert ex.snapshot(at(2)).equity == pytest.approx(100_000.0 - (16411.5 - 16387.5) * 2 * 20.0)


def test_cancel_of_a_working_entry_drops_its_children() -> None:
    ex = executor()
    entry = ex.submit_bracket(short(), at(0))
    ex.poll(at(1), bar(1, 16370.0, 16380.0))
    ex.cancel(entry.order_id, at(2))
    events = ex.poll(at(2), bar(2, 16370.0, 16380.0))
    assert [e.kind for e in events] == ["cancelled"] and events[0].order.role is OrderRole.ENTRY
    snap = ex.snapshot(at(2))
    assert snap.open_orders == () and snap.cancelled[0].order_id == entry.order_id
    assert ex.poll(at(3), bar(3, 16380.0, 16395.0)) == ()


def test_same_bars_give_the_same_events() -> None:
    bars = [bar(1, 16385.0, 16390.0), bar(2, 16360.0, 16400.0), bar(3, 16340.0, 16360.0)]
    a = [e.to_dict() for e in run(executor(), short(), bars)]
    b = [e.to_dict() for e in run(executor(), short(), bars)]
    assert a == b and len(a) >= 3


def test_partial_fills_across_touching_bars_then_filled() -> None:
    ex = executor(max_fill_per_bar=1)
    events = run(ex, short(3), [bar(1, 16385.0, 16390.0), bar(2, 16370.0, 16380.0), bar(3, 16385.0, 16390.0), bar(4, 16385.0, 16390.0)])
    entry_kinds = [(e.kind, e.order.filled_quantity) for e in events if e.order.role is OrderRole.ENTRY]
    assert entry_kinds == [("working", 0), ("partial", 1), ("partial", 2), ("filled", 3)]
    partial = next(e for e in events if e.kind == "partial")
    assert partial.fill is not None and partial.fill.quantity == 1 and partial.order.status is OrderStatus.PARTIAL
    snap = ex.snapshot(at(4))
    assert snap.net_position("NQ") == -3 and len(snap.fills) == 3
    # the exits work for the filled quantity, resized with every partial
    assert [e.order.quantity for e in events if e.kind == "working" and e.order.role is OrderRole.STOP] == [1, 2, 3]
    assert [o.quantity for o in snap.open_orders] == [3, 3]


def test_cancel_after_a_partial_keeps_the_position_and_its_exits() -> None:
    ex = executor(max_fill_per_bar=1)
    entry = ex.submit_bracket(short(2), at(0))
    ex.poll(at(1), bar(1, 16385.0, 16390.0))  # working, partial 1
    ex.cancel(entry.order_id, at(2))
    events = ex.poll(at(2), bar(2, 16385.0, 16390.0))
    assert [e.kind for e in events] == ["cancelled"] and events[0].order.filled_quantity == 1
    snap = ex.snapshot(at(2))
    assert snap.net_position("NQ") == -1 and [(o.role, o.quantity) for o in snap.open_orders] == [(OrderRole.STOP, 1), (OrderRole.TARGET, 1)]
    events = ex.poll(at(3), bar(3, 16400.0, 16420.0))
    assert [(e.kind, e.order.role.value) for e in events] == [("filled", "stop"), ("cancelled", "target")]
    assert ex.snapshot(at(3)).positions == () and ex.account.cash == pytest.approx(100_000.0 - (16411.5 - 16387.5) * 20.0)


def test_insufficient_margin_rejects_on_the_next_poll() -> None:
    ex = executor(margin_per_contract=60_000.0)
    entry = ex.submit_bracket(short(2), at(0))
    assert entry.status is OrderStatus.SUBMITTED
    events = ex.poll(at(1), bar(1, 16385.0, 16390.0))
    assert [e.kind for e in events] == ["rejected"] and events[0].order.status is OrderStatus.REJECTED
    snap = ex.snapshot(at(1))
    assert snap.open_orders == () and snap.positions == () and ex.account.rejected()[0].order_id == entry.order_id
    assert ex.poll(at(2), bar(2, 16385.0, 16390.0)) == ()


def test_available_funds_hold_margin_for_working_and_open_contracts() -> None:
    ex = executor(margin_per_contract=10_000.0)
    ex.submit_bracket(short(2), at(0))
    assert ex.snapshot(at(0)).available_funds == 80_000.0
    ex.poll(at(1), bar(1, 16385.0, 16390.0))  # filled
    assert ex.snapshot(at(1)).available_funds == 80_000.0 and ex.snapshot(at(1)).equity == 100_000.0


def test_account_summary_lists_every_order_bucket() -> None:
    ex = executor(max_fill_per_bar=1)
    first = ex.submit_bracket(short(2), at(0))
    ex.poll(at(1), bar(1, 16385.0, 16390.0))  # partial 1
    ex.cancel(first.order_id, at(2))
    ex.poll(at(2), bar(2, 16370.0, 16380.0))  # cancelled with 1 filled
    summary = ex.account.summary()
    assert summary["cash"] == 100_000.0 and summary["positions"] == [{"symbol": "NQ", "quantity": -1, "average_price": 16387.5}]
    assert summary["orders"] == {"pending": 2, "filled": 0, "cancelled": 1, "rejected": 0} and summary["fills"] == 1
    assert [o.status for o in ex.account.pending()] == [OrderStatus.WORKING, OrderStatus.WORKING]


def test_equity_marks_open_positions_at_the_last_polled_close() -> None:
    ex = executor()
    run(ex, short(quantity=2), [bar(1, 16370.0, 16380.0), bar(2, 16375.0, 16390.0, close=16385.0), bar(3, 16380.0, 16400.0, close=16395.0)])
    snap = ex.snapshot(at(3))
    assert snap.positions[0].quantity == -2 and ex.account.cash == 100_000.0
    assert snap.equity == 100_000.0 + (16387.5 - 16395.0) * 2 * 20.0 and snap.available_funds == 100_000.0 - 2 * CONFIG.margin_per_contract


def test_a_filled_brackets_exits_can_be_cancelled_and_a_flatten_fills_at_the_next_open() -> None:
    ex = executor()
    run(ex, short(quantity=2), [bar(1, 16370.0, 16380.0), bar(2, 16375.0, 16390.0, close=16385.0)])
    entry = next(o for o in ex.account.orders.values() if o.role is OrderRole.ENTRY)
    exits = [o for o in ex.snapshot(at(2)).open_orders if o.parent_id == entry.order_id]
    assert {o.role for o in exits} == {OrderRole.STOP, OrderRole.TARGET}
    for o in exits:
        ex.cancel(o.order_id, at(2))
    flat = ex.flatten("NQ", 2, "BUY", at(2), "EP_1:sig:invalidation")
    assert flat.role is OrderRole.FLATTEN and flat.client_ref == "EP_1:sig:invalidation" and flat.status is OrderStatus.SUBMITTED
    b3 = Bar(start=T0 + pd.Timedelta(minutes=3), open=16392.0, high=16420.0, low=16390.0, close=16410.0, volume=1.0, symbol="NQ", instrument_id=1)
    events = ex.poll(at(3), b3)
    kinds = [(e.kind, e.order.role.value) for e in events]
    assert ("cancelled", "stop") in kinds and ("cancelled", "target") in kinds and ("filled", "flatten") in kinds
    assert ("filled", "stop") not in kinds, "the flatten fills at the open before the bar's range is matched; the stop was cancelled"
    fill = next(e for e in events if e.order.role is OrderRole.FLATTEN).fill
    assert fill.price == 16392.0 and not ex.snapshot(at(3)).positions
    assert ex.account.cash == 100_000.0 + (16387.5 - 16392.0) * 2 * 20.0 and not ex.snapshot(at(3)).open_orders


def test_cancel_still_refuses_an_unknown_or_finished_order() -> None:
    ex = executor()
    with pytest.raises(ValueError):
        ex.cancel("sim-99", at(0))


def test_adding_to_a_position_averages_the_entry_price_so_each_exit_realizes_its_own_pnl() -> None:
    # Two same-direction brackets (as Risk v2 allows): the second fill must not inherit the first's average price.
    ex = executor()
    first = BracketIntent("EP_1:a", "NQ", "SELL", 2, 16400.0, 16420.0, 16350.0, "a")
    second = BracketIntent("EP_1:b", "NQ", "SELL", 2, 16410.0, 16420.0, 16350.0, "b")
    ex.submit_bracket(first, at(0))
    ex.poll(at(1), bar(1, 16380.0, 16405.0, close=16400.0))   # first fills at 16400
    ex.submit_bracket(second, at(1))
    ex.poll(at(2), bar(2, 16395.0, 16412.0, close=16405.0))   # second fills at 16410
    position = ex.snapshot(at(2)).positions[0]
    assert position.quantity == -4 and position.average_price == 16405.0
    ex.poll(at(3), bar(3, 16410.0, 16425.0, close=16420.0))   # both stops at 16420
    assert ex.snapshot(at(3)).positions == ()
    assert ex.account.cash == 100_000.0 + ((16400.0 - 16420.0) * 2 + (16410.0 - 16420.0) * 2) * 20.0  # −1 200, not −1 600
