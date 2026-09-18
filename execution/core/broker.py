"""The broker boundary the order machine talks to.

``Broker`` is one seam with two implementations: ``SimulatedExecutor``
(``execution/core/simulated_executor.py``) for historical replay and tests,
``IBKRBroker`` (``execution/core/ibkr_broker.py``) for the TWS paper session.
A snapshot of the account, one bracket submission, one cancel (an entry, or a
stop or target of a filled bracket), one market flatten, and a poll that
returns what changed since the last one.  Neither implementation modifies
an order in place: the machine replaces by cancel + submit."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from contract.execution import AccountSnapshot, BracketIntent, BrokerEvent, OrderState
from contract.market.primitives import Bar


class Broker(Protocol):
    paper: bool

    def snapshot(self, asof: pd.Timestamp) -> AccountSnapshot: ...

    def submit_bracket(self, intent: BracketIntent, asof: pd.Timestamp) -> OrderState: ...

    def cancel(self, order_id: str, asof: pd.Timestamp) -> None: ...

    def poll(self, asof: pd.Timestamp, bar: Bar | None) -> tuple[BrokerEvent, ...]: ...

    def flatten(self, symbol: str, quantity: int, side: str, asof: pd.Timestamp, client_ref: str) -> OrderState:
        """A market order closing ``quantity`` contracts; ``client_ref`` names
        the intent it belongs to."""
        ...


__all__ = ["Broker"]
