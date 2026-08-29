"""Execution-reality inputs, scoring, and the causal top-of-book adapter.

The Trading Eye publishes deterministic market facts only. Everything in this
module is execution-layer interpretation of broker/feed reality: the input
contract a caller supplies, the deterministic cost/fillability score derived
from it, and the inert value used by runs that do not evaluate execution at
all. The Eye transports the result; it never derives it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from .model import Direction, ExecutionObservation, aware_timestamp, clamp


@dataclass(frozen=True)
class ExecutionRealityInput:
    spread_points: float | None = None
    expected_slippage_points: float = 0.25
    commission_per_contract_per_side: float = 2.25
    quantity: int = 1
    deadline: pd.Timestamp | None = None
    data_age_seconds: float = 0.0
    size_available: float | None = None
    source: str = "missing"
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    depth_imbalance: float | None = None
    anomalies: tuple[str, ...] = ()


def observe_execution_reality(
    reality: ExecutionRealityInput,
    *,
    asof: pd.Timestamp,
    tick_size: float,
    point_value: float,
) -> ExecutionObservation:
    """Score one causal execution-reality input at one decision clock."""

    anomalies = list(reality.anomalies)
    spread = reality.spread_points
    if spread is None:
        spread = tick_size
        anomalies.append("spread_missing_used_one_tick")
    if reality.source.startswith("constant"):
        anomalies.append("execution_constant_assumption")
    if spread < 0:
        raise ValueError("spread cannot be negative")
    if reality.quantity <= 0:
        raise ValueError("execution quantity must be positive")
    commission_points = (
        2.0 * reality.commission_per_contract_per_side / point_value
    )
    cost = float(spread + 2.0 * reality.expected_slippage_points + commission_points)
    if reality.deadline is None:
        minutes = 24 * 60
        anomalies.append("deadline_missing")
    else:
        deadline = pd.Timestamp(reality.deadline)
        if deadline.tzinfo is None:
            raise ValueError("execution deadline must be timezone aware")
        minutes = int((deadline - asof).total_seconds() // 60)
    spread_ticks = spread / tick_size
    spread_score = math.exp(-0.18 * max(0.0, spread_ticks - 1.0))
    freshness = math.exp(-max(0.0, reality.data_age_seconds) / 30.0)
    deadline_score = clamp(max(0.0, minutes) / 20.0)
    size_score = (
        1.0
        if reality.size_available is None
        else clamp(reality.size_available / max(1, reality.quantity))
    )
    fillability = clamp(spread_score * freshness * max(0.25, deadline_score) * size_score)
    if reality.data_age_seconds > 60:
        anomalies.append("stale_market_data")
    if minutes <= 0:
        anomalies.append("deadline_elapsed")
    return ExecutionObservation(
        spread_points=float(spread),
        expected_slippage_points=float(reality.expected_slippage_points),
        expected_round_trip_cost_points=cost,
        minutes_to_deadline=minutes,
        fillability=fillability,
        data_age_seconds=float(reality.data_age_seconds),
        size_available=reality.size_available,
        anomalies=tuple(anomalies),
        source=reality.source,
        bid=reality.bid,
        ask=reality.ask,
        bid_size=reality.bid_size,
        ask_size=reality.ask_size,
        depth_imbalance=reality.depth_imbalance,
    )


def execution_not_evaluated() -> ExecutionObservation:
    """Return an inert contract value for Eye-only semantic replay."""

    return ExecutionObservation(
        spread_points=0.0,
        expected_slippage_points=0.0,
        expected_round_trip_cost_points=0.0,
        minutes_to_deadline=0,
        fillability=0.0,
        data_age_seconds=0.0,
        size_available=None,
        anomalies=(),
        source="not_evaluated",
    )


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


__all__ = [
    "ExecutionRealityInput",
    "TopOfBook",
    "TopOfBookExecutionProvider",
    "execution_not_evaluated",
    "observe_execution_reality",
]
