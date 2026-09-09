"""One factual, versioned structural-outcome engine.

The engine describes what OHLCV can establish.  In particular, a bar that
touches target and invalidation is factually ambiguous.  Conservative
execution treatment is an explicit downstream projection and never rewrites
the factual research outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from .foundation_registry import FOUNDATION_VERSION
from contract.market import (
    Direction,
    Timeframe,
    aware_timestamp,
    price_to_ticks,
    to_primitive,
)


STRUCTURAL_OUTCOME_VERSION = "structural_outcome_v1"

_TIMEFRAME_SECONDS = {
    Timeframe.M1: 60,
    Timeframe.M5: 5 * 60,
    Timeframe.M15: 15 * 60,
    Timeframe.H1: 60 * 60,
    Timeframe.H4: 4 * 60 * 60,
}


def _clock(value: Any, *, name: str) -> pd.Timestamp:
    result = aware_timestamp(value, name=name)
    if pd.isna(result):
        raise ValueError(f"{name} cannot be NaT")
    return result


def _identity(prefix: str, value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class OutcomeTerminal(str, Enum):
    TARGET_FIRST = "target_first"
    INVALIDATION_FIRST = "invalidation_first"
    AMBIGUOUS_SAME_BAR = "ambiguous_same_bar"
    CENSORED = "censored"


@dataclass(frozen=True)
class OutcomeBar:
    """One completed native-timeframe price fact used by the engine."""

    bar_event_id: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    known_at: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        if (
            not self.bar_event_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
        ):
            raise ValueError("outcome BAR identity is incomplete")
        object.__setattr__(
            self, "known_at", _clock(self.known_at, name="outcome BAR known_at")
        )
        values = tuple(
            float(value)
            for value in (self.open, self.high, self.low, self.close)
        )
        if not all(math.isfinite(value) for value in values) or not (
            values[2] <= values[0] <= values[1]
            and values[2] <= values[3] <= values[1]
        ):
            raise ValueError("outcome BAR OHLC is invalid")
        for name, value in zip(("open", "high", "low", "close"), values):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class StructuralOutcomeSpec:
    """Frozen estimand and censoring contract for one semantic source."""

    source_event_id: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    observation_start_known_at: pd.Timestamp
    observation_window_end_exclusive: pd.Timestamp
    reference_price: float
    target_price: float
    invalidation_price: float
    target_definition: str
    invalidation_definition: str
    atr_at_start: float
    tick_size: float
    horizon_bars: int
    horizon_seconds: int
    contract_boundary_policy: str = "censor"
    window_boundary_policy: str = "censor"
    semantic_version: str = FOUNDATION_VERSION
    outcome_version: str = STRUCTURAL_OUTCOME_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        if (
            not self.source_event_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or not self.target_definition
            or not self.invalidation_definition
            or self.semantic_version != FOUNDATION_VERSION
            or self.outcome_version != STRUCTURAL_OUTCOME_VERSION
            or self.contract_boundary_policy != "censor"
            or self.window_boundary_policy != "censor"
            or type(self.horizon_bars) is not int
            or self.horizon_bars < 1
            or type(self.horizon_seconds) is not int
            or self.horizon_seconds < 1
        ):
            raise ValueError("structural outcome spec is incomplete")
        start = _clock(
            self.observation_start_known_at,
            name="outcome observation_start_known_at",
        )
        end = _clock(
            self.observation_window_end_exclusive,
            name="outcome observation_window_end_exclusive",
        )
        if end <= start:
            raise ValueError("structural outcome observation window is empty")
        object.__setattr__(self, "observation_start_known_at", start)
        object.__setattr__(self, "observation_window_end_exclusive", end)
        reference = _finite_positive(self.reference_price, name="outcome reference")
        target = _finite_positive(self.target_price, name="outcome target")
        invalidation = _finite_positive(
            self.invalidation_price,
            name="outcome invalidation",
        )
        atr = _finite_positive(self.atr_at_start, name="outcome ATR_at_start")
        tick = _finite_positive(self.tick_size, name="outcome tick_size")
        object.__setattr__(self, "reference_price", reference)
        object.__setattr__(self, "target_price", target)
        object.__setattr__(self, "invalidation_price", invalidation)
        object.__setattr__(self, "atr_at_start", atr)
        object.__setattr__(self, "tick_size", tick)
        for name, value in (
            ("reference", reference),
            ("target", target),
            ("invalidation", invalidation),
        ):
            price_to_ticks(value, tick, name=f"outcome {name}")
        geometry_valid = (
            invalidation < reference < target
            if self.direction is Direction.LONG
            else target < reference < invalidation
        )
        if not geometry_valid:
            raise ValueError("structural outcome prices oppose the direction")
        payload = {
            name: value for name, value in self.__dict__.items() if name != "spec_id"
        }
        object.__setattr__(self, "spec_id", _identity("outcome-spec", payload))


@dataclass(frozen=True)
class StructuralOutcome:
    source_event_id: str
    spec_id: str
    semantic_version: str
    outcome_version: str
    symbol: str
    instrument_id: int
    direction: Direction
    observation_start_known_at: pd.Timestamp
    observation_window_end_exclusive: pd.Timestamp
    horizon_bars: int
    horizon_timeframe: Timeframe
    horizon_seconds: int
    reference_price: float
    target_price: float
    invalidation_price: float
    target_definition: str
    invalidation_definition: str
    atr_at_start: float
    tick_size: float
    contract_boundary_policy: str
    window_boundary_policy: str
    terminal: OutcomeTerminal
    resolved_at: pd.Timestamp
    target_hit_at: pd.Timestamp | None
    invalidation_hit_at: pd.Timestamp | None
    mfe_atr: float | None
    mae_atr: float | None
    observed_bars: int
    full_horizon_observed: bool
    path_censor_reason: str | None
    source_bar_event_ids: tuple[str, ...]
    outcome_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terminal", OutcomeTerminal(self.terminal))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(self, "horizon_timeframe", Timeframe(self.horizon_timeframe))
        start = _clock(
            self.observation_start_known_at,
            name="structural outcome observation start",
        )
        resolved = _clock(self.resolved_at, name="structural outcome resolved_at")
        window_end = _clock(
            self.observation_window_end_exclusive,
            name="structural outcome observation window end",
        )
        object.__setattr__(self, "observation_start_known_at", start)
        object.__setattr__(self, "observation_window_end_exclusive", window_end)
        object.__setattr__(self, "resolved_at", resolved)
        for name in ("target_hit_at", "invalidation_hit_at"):
            value = getattr(self, name)
            if value is not None:
                value = _clock(value, name=f"structural outcome {name}")
                if value <= start or value >= window_end:
                    raise ValueError("structural outcome hit clock is invalid")
                object.__setattr__(self, name, value)
        bar_ids = tuple(self.source_bar_event_ids)
        reference = _finite_positive(
            self.reference_price,
            name="structural outcome reference",
        )
        target = _finite_positive(
            self.target_price,
            name="structural outcome target",
        )
        invalidation = _finite_positive(
            self.invalidation_price,
            name="structural outcome invalidation",
        )
        atr = _finite_positive(self.atr_at_start, name="structural outcome ATR")
        tick = _finite_positive(self.tick_size, name="structural outcome tick_size")
        object.__setattr__(self, "reference_price", reference)
        object.__setattr__(self, "target_price", target)
        object.__setattr__(self, "invalidation_price", invalidation)
        object.__setattr__(self, "atr_at_start", atr)
        object.__setattr__(self, "tick_size", tick)
        for name, value in (
            ("reference", reference),
            ("target", target),
            ("invalidation", invalidation),
        ):
            price_to_ticks(value, tick, name=f"structural outcome {name}")
        geometry_valid = (
            invalidation < reference < target
            if self.direction is Direction.LONG
            else target < reference < invalidation
        )
        if (
            not self.source_event_id
            or not self.spec_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.semantic_version != FOUNDATION_VERSION
            or self.outcome_version != STRUCTURAL_OUTCOME_VERSION
            or window_end <= start
            or resolved > window_end
            or resolved <= start
            or not geometry_valid
            or not self.target_definition
            or not self.invalidation_definition
            or self.contract_boundary_policy != "censor"
            or self.window_boundary_policy != "censor"
            or type(self.horizon_bars) is not int
            or self.horizon_bars < 1
            or type(self.horizon_seconds) is not int
            or self.horizon_seconds < 1
            or type(self.observed_bars) is not int
            or self.observed_bars < 0
            or self.observed_bars != len(bar_ids)
            or len(bar_ids) != len(set(bar_ids))
            or any(not isinstance(value, str) or not value for value in bar_ids)
            or self.full_horizon_observed != (self.observed_bars == self.horizon_bars)
            or (self.mfe_atr is None) != (not self.full_horizon_observed)
            or (self.mae_atr is None) != (not self.full_horizon_observed)
            or any(
                value is not None and (not math.isfinite(float(value)) or value < 0.0)
                for value in (self.mfe_atr, self.mae_atr)
            )
        ):
            raise ValueError("structural outcome invariant failed")
        if self.terminal is OutcomeTerminal.TARGET_FIRST:
            valid_terminal = (
                self.target_hit_at is not None
                and (
                    self.invalidation_hit_at is None
                    or self.target_hit_at < self.invalidation_hit_at
                )
            )
        elif self.terminal is OutcomeTerminal.INVALIDATION_FIRST:
            valid_terminal = (
                self.invalidation_hit_at is not None
                and (
                    self.target_hit_at is None
                    or self.invalidation_hit_at < self.target_hit_at
                )
            )
        elif self.terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR:
            valid_terminal = (
                self.target_hit_at is not None
                and self.target_hit_at == self.invalidation_hit_at
            )
        else:
            valid_terminal = (
                self.target_hit_at is None and self.invalidation_hit_at is None
            )
        if not valid_terminal:
            raise ValueError("structural outcome terminal conflicts with hit clocks")
        if (self.path_censor_reason is None) != self.full_horizon_observed:
            raise ValueError("structural outcome path censor status is inconsistent")
        payload = {
            name: value for name, value in self.__dict__.items() if name != "outcome_id"
        }
        object.__setattr__(self, "outcome_id", _identity("structural-outcome", payload))


class StructuralOutcomeEngine:
    """Pure OHLCV evaluator shared by semantic research consumers."""

    @staticmethod
    def evaluate(
        spec: StructuralOutcomeSpec,
        bars: Sequence[OutcomeBar],
    ) -> StructuralOutcome:
        if not isinstance(spec, StructuralOutcomeSpec):
            raise TypeError("StructuralOutcomeEngine requires a frozen spec")
        ordered = tuple(bars)
        if any(not isinstance(bar, OutcomeBar) for bar in ordered):
            raise TypeError("StructuralOutcomeEngine requires OutcomeBar inputs")
        order_keys = tuple((bar.known_at, bar.bar_event_id) for bar in ordered)
        clocks = tuple(bar.known_at for bar in ordered)
        if (
            order_keys != tuple(sorted(order_keys))
            or len(order_keys) != len(set(order_keys))
            or len(clocks) != len(set(clocks))
        ):
            raise ValueError("outcome BAR inputs are duplicated or out of order")

        deadline = spec.observation_start_known_at + pd.Timedelta(
            spec.horizon_seconds,
            unit="s",
        )
        path: list[OutcomeBar] = []
        path_censor_reason: str | None = None
        censor_clock: pd.Timestamp | None = None
        expected_clock = spec.observation_start_known_at + pd.Timedelta(
            _TIMEFRAME_SECONDS[spec.timeframe],
            unit="s",
        )
        for bar in ordered:
            # The source/decision BAR is never part of its own future path.
            if bar.known_at <= spec.observation_start_known_at:
                continue
            if bar.timeframe is not spec.timeframe:
                raise ValueError("outcome BAR timeframe differs from frozen horizon")
            if bar.known_at >= spec.observation_window_end_exclusive:
                path_censor_reason = "observation_window_end"
                censor_clock = spec.observation_window_end_exclusive
                break
            if bar.known_at > deadline:
                path_censor_reason = "horizon_seconds_elapsed"
                censor_clock = deadline
                break
            if bar.known_at != expected_clock:
                # A horizon in completed bars is meaningful only when the
                # native completed-bar census is contiguous.  A later bar
                # cannot silently stand in for a missing formation/response
                # observation; scheduled closure and data loss are both
                # conservatively censored unless a future version registers
                # an explicit session calendar.
                path_censor_reason = "missing_native_completed_bar"
                censor_clock = expected_clock
                break
            if bar.symbol != spec.symbol or bar.instrument_id != spec.instrument_id:
                path_censor_reason = "contract_change"
                censor_clock = bar.known_at
                break
            for name, value in (
                ("open", bar.open),
                ("high", bar.high),
                ("low", bar.low),
                ("close", bar.close),
            ):
                price_to_ticks(value, spec.tick_size, name=f"outcome BAR {name}")
            path.append(bar)
            expected_clock = expected_clock + pd.Timedelta(
                _TIMEFRAME_SECONDS[spec.timeframe],
                unit="s",
            )
            if len(path) == spec.horizon_bars:
                break

        full_horizon = len(path) == spec.horizon_bars
        if not full_horizon and path_censor_reason is None:
            if path and path[-1].known_at >= deadline:
                path_censor_reason = "horizon_seconds_elapsed"
                censor_clock = deadline
            elif spec.observation_window_end_exclusive <= deadline:
                path_censor_reason = "observation_window_end"
                censor_clock = spec.observation_window_end_exclusive
            else:
                path_censor_reason = "incomplete_completed_bar_census"
                censor_clock = path[-1].known_at if path else deadline

        target_hit_at: pd.Timestamp | None = None
        invalidation_hit_at: pd.Timestamp | None = None
        favorable: list[float] = []
        adverse: list[float] = []
        for bar in path:
            if spec.direction is Direction.LONG:
                target_hit = bar.high >= spec.target_price
                invalidation_hit = bar.low <= spec.invalidation_price
                favorable.append(max(0.0, bar.high - spec.reference_price))
                adverse.append(max(0.0, spec.reference_price - bar.low))
            else:
                target_hit = bar.low <= spec.target_price
                invalidation_hit = bar.high >= spec.invalidation_price
                favorable.append(max(0.0, spec.reference_price - bar.low))
                adverse.append(max(0.0, bar.high - spec.reference_price))
            if target_hit and target_hit_at is None:
                target_hit_at = bar.known_at
            if invalidation_hit and invalidation_hit_at is None:
                invalidation_hit_at = bar.known_at

        if target_hit_at is not None and target_hit_at == invalidation_hit_at:
            terminal = OutcomeTerminal.AMBIGUOUS_SAME_BAR
            resolved_at = target_hit_at
        elif target_hit_at is not None and (
            invalidation_hit_at is None or target_hit_at < invalidation_hit_at
        ):
            terminal = OutcomeTerminal.TARGET_FIRST
            resolved_at = target_hit_at
        elif invalidation_hit_at is not None:
            terminal = OutcomeTerminal.INVALIDATION_FIRST
            resolved_at = invalidation_hit_at
        else:
            terminal = OutcomeTerminal.CENSORED
            resolved_at = censor_clock or deadline
        return StructuralOutcome(
            source_event_id=spec.source_event_id,
            spec_id=spec.spec_id,
            semantic_version=spec.semantic_version,
            outcome_version=spec.outcome_version,
            symbol=spec.symbol,
            instrument_id=spec.instrument_id,
            direction=spec.direction,
            observation_start_known_at=spec.observation_start_known_at,
            observation_window_end_exclusive=(
                spec.observation_window_end_exclusive
            ),
            horizon_bars=spec.horizon_bars,
            horizon_timeframe=spec.timeframe,
            horizon_seconds=spec.horizon_seconds,
            reference_price=spec.reference_price,
            target_price=spec.target_price,
            invalidation_price=spec.invalidation_price,
            target_definition=spec.target_definition,
            invalidation_definition=spec.invalidation_definition,
            atr_at_start=spec.atr_at_start,
            tick_size=spec.tick_size,
            contract_boundary_policy=spec.contract_boundary_policy,
            window_boundary_policy=spec.window_boundary_policy,
            terminal=terminal,
            resolved_at=resolved_at,
            target_hit_at=target_hit_at,
            invalidation_hit_at=invalidation_hit_at,
            mfe_atr=(
                max(favorable, default=0.0) / spec.atr_at_start
                if full_horizon
                else None
            ),
            mae_atr=(
                max(adverse, default=0.0) / spec.atr_at_start
                if full_horizon
                else None
            ),
            observed_bars=len(path),
            full_horizon_observed=full_horizon,
            path_censor_reason=path_censor_reason,
            source_bar_event_ids=tuple(bar.bar_event_id for bar in path),
        )


@dataclass(frozen=True)
class ConservativeExecutionProjection:
    factual_outcome_id: str
    factual_terminal: OutcomeTerminal
    execution_terminal: OutcomeTerminal
    conservative_assumption_applied: bool
    reason: str
    projection_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "factual_terminal",
            OutcomeTerminal(self.factual_terminal),
        )
        object.__setattr__(
            self, "execution_terminal", OutcomeTerminal(self.execution_terminal)
        )
        if (
            not self.factual_outcome_id
            or not self.reason
            or (
                self.conservative_assumption_applied
                != (self.factual_terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR)
            )
            or (
                self.factual_terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
                and self.execution_terminal
                is not OutcomeTerminal.INVALIDATION_FIRST
            )
            or (
                self.factual_terminal is not OutcomeTerminal.AMBIGUOUS_SAME_BAR
                and self.execution_terminal is not self.factual_terminal
            )
        ):
            raise ValueError("conservative execution projection is inconsistent")
        payload = {
            name: value
            for name, value in self.__dict__.items()
            if name != "projection_id"
        }
        object.__setattr__(
            self,
            "projection_id",
            _identity("conservative-execution-projection", payload),
        )


def project_conservative_execution(
    outcome: StructuralOutcome,
) -> ConservativeExecutionProjection:
    if not isinstance(outcome, StructuralOutcome):
        raise TypeError("execution projection requires StructuralOutcome")
    ambiguous = outcome.terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
    return ConservativeExecutionProjection(
        factual_outcome_id=outcome.outcome_id,
        factual_terminal=outcome.terminal,
        execution_terminal=(
            OutcomeTerminal.INVALIDATION_FIRST if ambiguous else outcome.terminal
        ),
        conservative_assumption_applied=ambiguous,
        reason=(
            "same_bar_ambiguity_projected_invalidation_first"
            if ambiguous
            else "factual_terminal_preserved"
        ),
    )


__all__ = [
    "ConservativeExecutionProjection",
    "OutcomeBar",
    "OutcomeTerminal",
    "STRUCTURAL_OUTCOME_VERSION",
    "StructuralOutcome",
    "StructuralOutcomeEngine",
    "StructuralOutcomeSpec",
    "project_conservative_execution",
]
