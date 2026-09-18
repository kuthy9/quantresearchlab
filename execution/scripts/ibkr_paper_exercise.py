"""Exercise the IBKR paper session through the executor's own broker adapter.

    .venv/bin/python -m execution.scripts.ibkr_paper_exercise --reference-price 16400 --i-place-paper-orders [--marketable]

With TWS / IB Gateway logged into the paper account and the API enabled on
the port in ``execution/configs/ibkr.json``, this places the orders the
order machine would place and records what the API reports back — the
receipt the backtest cannot give: order statuses, cancels, a replacement
(cancel + resubmit, as the machine does it) and, with ``--marketable``, a
fill followed by a flatten.  Every ``BrokerEvent`` and every snapshot goes
to ``outputs/ibkr_paper/<timestamp>.json``.

Steps: refuse a non-flat account; submit a SELL bracket 1 % above the
reference price (unmarketable) and wait for ``working``; cancel it and wait
for ``cancelled``; submit again 1.5 % above and cancel again (the
replacement); with ``--marketable``, submit a SELL bracket 0.5 % below the
reference, wait for ``filled``, cancel its stop and target, flatten with a
market BUY and wait for the account to be flat.  A partial fill cannot be
forced on the paper session; the receipt records whatever TWS reports.

The script needs ``--i-place-paper-orders`` to do anything: it is the
only path in the repository outside ``run_llm_brain.py --broker ibkr`` that
sends an order, and it is run by hand, never by the tests."""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

from contract.execution import BracketIntent, OrderRole
from execution.core.broker import Broker
from execution.core.ibkr_broker import IBKRBroker, IBKRConfig, IBKRRefused, require_flat
from risk.core.gate import round_to_tick

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = "outputs/ibkr_paper"
POLLS_PER_STEP = 30
STOP_POINTS = 20.0
TARGET_POINTS = 40.0


def _wait_for(broker: Broker, order_id: str, kind: str, *, clock: Callable[[], pd.Timestamp], sleep: Callable[[float], None], events: list[dict[str, Any]]) -> bool:
    return _wait_for_all(broker, {(order_id, kind)}, clock=clock, sleep=sleep, events=events)


def _wait_for_all(broker: Broker, targets: set[tuple[str, str]], *, clock: Callable[[], pd.Timestamp], sleep: Callable[[float], None], events: list[dict[str, Any]]) -> bool:
    """Poll until every (order id, kind) has been reported; a whole poll batch
    is kept, since siblings' events arrive together."""
    pending = set(targets)
    for _ in range(POLLS_PER_STEP):
        for event in broker.poll(clock(), None):
            events.append(event.to_dict())
            pending.discard((event.order.order_id, event.kind))
        if not pending:
            return True
        sleep(1.0)
    return False


def _bracket(client_ref: str, side: str, limit: float, tick: float) -> BracketIntent:
    limit = round_to_tick(limit, tick)
    sign = 1.0 if side == "SELL" else -1.0
    return BracketIntent(client_ref, "NQ", side, 1, limit, round_to_tick(limit + sign * STOP_POINTS, tick), round_to_tick(limit - sign * TARGET_POINTS, tick), client_ref)


def exercise(
    broker: Broker, *, reference_price: float, tick_size: float, marketable: bool,
    clock: Callable[[], pd.Timestamp], sleep: Callable[[float], None], log: Callable[[str], None],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {"reference_price": reference_price, "steps": [], "initial_snapshot": None, "final_snapshot": None, "flat_at_end": None}
    snapshot = broker.snapshot(clock())
    receipt["initial_snapshot"] = snapshot.to_dict()
    require_flat(snapshot, "NQ")

    def step(name: str, action: Callable[[list[dict[str, Any]]], bool]) -> None:
        events: list[dict[str, Any]] = []
        ok = action(events)
        receipt["steps"].append({"name": name, "ok": ok, "events": events, "snapshot": broker.snapshot(clock()).to_dict()})
        log(f"{name}: {'ok' if ok else 'NOT as expected'} ({len(events)} events)")

    entry_ids: dict[str, str] = {}

    def submit(name: str, intent: BracketIntent, expect: str) -> None:
        def action(events: list[dict[str, Any]]) -> bool:
            entry = broker.submit_bracket(intent, clock())
            entry_ids[name] = entry.order_id
            events.append({"kind": "submitted", "order": entry.to_dict(), "fill": None, "at": entry.submitted_at.strftime("%Y-%m-%dT%H:%M:%SZ")})
            return _wait_for(broker, entry.order_id, expect, clock=clock, sleep=sleep, events=events)
        step(name, action)

    def cancel(name: str, of: str) -> None:
        def action(events: list[dict[str, Any]]) -> bool:
            broker.cancel(entry_ids[of], clock())
            return _wait_for(broker, entry_ids[of], "cancelled", clock=clock, sleep=sleep, events=events)
        step(name, action)

    submit("unmarketable_bracket", _bracket("paper-exercise:1", "SELL", reference_price * 1.01, tick_size), "working")
    cancel("cancel", "unmarketable_bracket")
    submit("replace_bracket", _bracket("paper-exercise:2", "SELL", reference_price * 1.015, tick_size), "working")
    cancel("cancel_replacement", "replace_bracket")
    if marketable:
        submit("marketable_bracket", _bracket("paper-exercise:3", "SELL", reference_price * 0.995, tick_size), "filled")

        def cancel_exits(events: list[dict[str, Any]]) -> bool:
            snapshot = broker.snapshot(clock())
            exits = [order for order in snapshot.open_orders if order.role in (OrderRole.STOP, OrderRole.TARGET) and order.parent_id == entry_ids["marketable_bracket"]]
            for order in exits:
                broker.cancel(order.order_id, clock())
            return _wait_for_all(broker, {(order.order_id, "cancelled") for order in exits}, clock=clock, sleep=sleep, events=events)
        step("cancel_exits", cancel_exits)

        def flatten(events: list[dict[str, Any]]) -> bool:
            snapshot = broker.snapshot(clock())
            quantity = snapshot.net_position("NQ")
            if quantity == 0:
                return True
            order = broker.flatten("NQ", abs(quantity), "BUY" if quantity < 0 else "SELL", clock(), "paper-exercise:flatten")
            events.append({"kind": "submitted", "order": order.to_dict(), "fill": None, "at": order.submitted_at.strftime("%Y-%m-%dT%H:%M:%SZ")})
            return _wait_for(broker, order.order_id, "filled", clock=clock, sleep=sleep, events=events)
        step("flatten", flatten)
    final = broker.snapshot(clock())
    receipt["final_snapshot"] = final.to_dict()
    receipt["flat_at_end"] = final.net_position("NQ") == 0 and not final.open_orders
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="execution/configs/ibkr.json")
    parser.add_argument("--model-path", default="configs/model.json")
    parser.add_argument("--reference-price", type=float, required=True, help="the last NQ price shown in TWS; the orders are priced from it")
    parser.add_argument("--marketable", action="store_true", help="also take a fill and flatten it")
    parser.add_argument("--i-place-paper-orders", action="store_true", help="required: this script places orders on the paper account")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.i_place_paper_orders:
        print("refused: pass --i-place-paper-orders to place orders on the paper session", file=sys.stderr)
        return 2
    config = IBKRConfig.from_json(ROOT / args.config)
    model = json.loads((ROOT / args.model_path).read_text(encoding="utf-8"))
    try:
        broker = IBKRBroker.connect(config, live_execution_allowed=bool(model["release_readiness"]["live_execution_allowed"]))
        receipt = exercise(
            broker, reference_price=args.reference_price, tick_size=float(model["tick_size"]), marketable=args.marketable,
            clock=lambda: pd.Timestamp.now(tz="UTC"), sleep=time.sleep, log=print,
        )
    except IBKRRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"receipt → {path}; flat at end: {receipt['flat_at_end']}")
    return 0 if receipt["flat_at_end"] and all(step["ok"] for step in receipt["steps"]) else 1


if __name__ == "__main__":
    sys.exit(main())
