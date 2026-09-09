"""Execution reality: what a venue reported and what the account holds.

Scored by the engine and only transported by the Eye. It names market
primitives and nothing else, so an observation may carry it."""
from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd

from contract.market.primitives import Direction, LiquidityLevel, Playbook, StructuralLevel, aware_timestamp, clamp


@dataclass(frozen=True)
class ExecutionObservation:
    spread_points: float
    expected_slippage_points: float
    expected_round_trip_cost_points: float
    minutes_to_deadline: int
    fillability: float
    data_age_seconds: float
    size_available: float | None
    anomalies: tuple[str, ...] = ()
    source: str = "unknown"
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    depth_imbalance: float | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.spread_points,
            self.expected_slippage_points,
            self.expected_round_trip_cost_points,
            self.data_age_seconds,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in numeric):
            raise ValueError("execution observation contains invalid values")
        optional = (
            self.bid,
            self.ask,
            self.bid_size,
            self.ask_size,
            self.depth_imbalance,
        )
        if any(
            value is not None and not math.isfinite(float(value))
            for value in optional
        ):
            raise ValueError("execution observation contains invalid book values")
        if self.bid is not None and self.ask is not None and self.bid >= self.ask:
            raise ValueError("execution observation contains a crossed book")
        if (
            (self.bid_size is not None and self.bid_size < 0)
            or (self.ask_size is not None and self.ask_size < 0)
        ):
            raise ValueError("execution observation contains negative book size")
        if self.depth_imbalance is not None and not -1.0 <= self.depth_imbalance <= 1.0:
            raise ValueError("execution depth imbalance must be in [-1, 1]")
        if not self.source:
            raise ValueError("execution observation source is required")
        object.__setattr__(self, "fillability", clamp(self.fillability))


def execution_not_evaluated() -> "ExecutionObservation":
    """The inert execution value an Eye-only replay transports.

    The Eye never derives execution reality; when a caller supplies none this
    is what the observation carries, and ``source`` says so plainly rather than
    presenting a default-derived score as if it had been observed.
    """

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
class PositionSnapshot:
    thesis_hash: str
    symbol: str
    instrument_id: int
    playbook: Playbook
    direction: Direction
    entry_price: float
    original_invalidation: StructuralLevel
    current_stop: float
    primary_target: LiquidityLevel
    opened_at: pd.Timestamp
    deadline: pd.Timestamp
    quantity: int
    unrealized_R: float
    elapsed_minutes: int
    mfe_R: float = 0.0
    mae_R: float = 0.0
    protection_candidate: StructuralLevel | None = None
    status: str = "open"
    setup_id: str | None = None
    entry_location_id: str | None = None
    entry_path_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "opened_at", aware_timestamp(self.opened_at, name="position.opened_at"))
        object.__setattr__(self, "deadline", aware_timestamp(self.deadline, name="position.deadline"))
        if not self.symbol or int(self.instrument_id) < 0:
            raise ValueError("position contract identity is invalid")
        if self.deadline <= self.opened_at:
            raise ValueError("position deadline must follow its opening")
        if self.quantity <= 0:
            raise ValueError("position quantity must be positive")
        if self.elapsed_minutes < 0:
            raise ValueError("position elapsed time cannot be negative")
        if not all(
            math.isfinite(float(value))
            for value in (self.unrealized_R, self.mfe_R, self.mae_R)
        ):
            raise ValueError("position R state must be finite")
        if self.mfe_R < -1e-12 or self.mae_R > 1e-12:
            raise ValueError("position MFE/MAE signs are invalid")
        typed_identity = (
            self.setup_id,
            self.entry_location_id,
            self.entry_path_id,
        )
        if any(value is not None for value in typed_identity) and (
            any(
                not isinstance(value, str) or not value
                for value in typed_identity
            )
        ):
            raise ValueError(
                "position typed identities must be complete non-empty text"
            )


@dataclass(frozen=True)
class AccountState:
    equity: float
    open_risk_fraction: float = 0.0
    requested_risk_fraction: float = 0.005
    quantity: int = 1
    point_value: float = 20.0
    position: PositionSnapshot | None = None

    def __post_init__(self) -> None:
        if self.equity <= 0 or self.quantity <= 0 or self.point_value <= 0:
            raise ValueError("account equity, quantity, and point value must be positive")
        if self.open_risk_fraction < 0 or self.requested_risk_fraction < 0:
            raise ValueError("account risk fractions cannot be negative")


__all__ = [
    "AccountState",
    "ExecutionObservation",
    "PositionSnapshot",
    "execution_not_evaluated",
]
