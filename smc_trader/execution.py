"""Causal top-of-book adapter for execution-reality observations."""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from .model import Direction, aware_timestamp
from .observation import ExecutionRealityInput


@dataclass(frozen=True)
class TopOfBook:
    observed_at: pd.Timestamp
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at", aware_timestamp(self.observed_at, name="book.observed_at")
        )
        values = (self.bid, self.ask, self.bid_size, self.ask_size)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("top of book contains non-finite values")
        if self.bid <= 0 or self.ask <= self.bid or self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("top of book is crossed or invalid")


class TopOfBookExecutionProvider:
    """Turns a causal book snapshot into descriptive cost/fillability inputs."""

    def __init__(
        self,
        *,
        tick_size: float = 0.25,
        commission_per_side: float = 2.25,
    ) -> None:
        self.tick_size = float(tick_size)
        self.commission_per_side = float(commission_per_side)

    def observe(
        self,
        book: TopOfBook,
        *,
        decision_clock: pd.Timestamp,
        deadline: pd.Timestamp,
        direction: Direction | None = None,
        quantity: int = 1,
    ) -> ExecutionRealityInput:
        decision_clock = aware_timestamp(decision_clock, name="decision_clock")
        deadline = aware_timestamp(deadline, name="deadline")
        if book.observed_at > decision_clock:
            raise ValueError("book snapshot is later than the decision clock")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        spread = float(book.ask - book.bid)
        near_size = (
            min(book.bid_size, book.ask_size)
            if direction is None
            else book.ask_size
            if direction is Direction.LONG
            else book.bid_size
        )
        # For a one-level BBO observation, price impact beyond the best quote is
        # unknowable. Do not invent a constant slippage charge: record zero
        # additional impact only when displayed best-side size covers the order,
        # and fail closed through a risk anomaly otherwise.
        sufficient_bbo_depth = near_size >= quantity
        slippage = 0.0
        age = max(0.0, (decision_clock - book.observed_at).total_seconds())
        return ExecutionRealityInput(
            spread_points=spread,
            expected_slippage_points=slippage,
            commission_per_contract_per_side=self.commission_per_side,
            quantity=quantity,
            deadline=deadline,
            data_age_seconds=age,
            size_available=near_size,
            source="observed_bbo",
            bid=book.bid,
            ask=book.ask,
            bid_size=book.bid_size,
            ask_size=book.ask_size,
            depth_imbalance=(
                (book.bid_size - book.ask_size)
                / max(1.0, book.bid_size + book.ask_size)
            ),
            anomalies=(
                ()
                if sufficient_bbo_depth
                else ("insufficient_top_of_book_depth",)
            ),
        )


__all__ = ["TopOfBook", "TopOfBookExecutionProvider"]
