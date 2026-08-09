"""Shared immutable contracts for the continuous SMC engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> float:
        return 1.0 if self is Direction.LONG else -1.0

    @property
    def opposing_liquidity_side(self) -> str:
        return "above" if self is Direction.LONG else "below"

    @property
    def invalidation_side(self) -> str:
        return "below" if self is Direction.LONG else "above"


class Timeframe(str, Enum):
    H4 = "4H"
    H1 = "1H"
    M5 = "5m"
    M1 = "1m"
    # Optional context bridge enabled through the explicit ScaleSpec registry.
    M15 = "15m"


CORE_TIMEFRAMES = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M5,
    Timeframe.M1,
)


class Playbook(str, Enum):
    DISPLACEMENT_FIRST_PULLBACK = "displacement_first_pullback"
    LIQUIDITY_SWEEP_REVERSAL = "liquidity_sweep_reversal"
    FAILED_AUCTION_VALUE_RETURN = "failed_auction_value_return"


class PlaybookPhase(str, Enum):
    INACTIVE = "inactive"
    FORMING = "forming"
    ARMED = "armed"
    WAITING_LOCATION = "waiting_location"
    WAITING_TRIGGER = "waiting_trigger"
    EXECUTABLE = "executable"
    ENTERED = "entered"
    WEAKENING = "weakening"
    DELIVERING = "delivering"
    COMPLETED = "completed"
    INVALIDATED = "invalidated"


class Action(str, Enum):
    ENTER = "enter"
    WAIT = "wait"
    HOLD = "hold"
    PROTECT = "protect"
    EXIT = "exit"
    ABSTAIN = "abstain"


class EventKind(str, Enum):
    SWING_FORMED = "swing_formed"
    SWING_STATE = "swing_state"
    STRUCTURE_STATE = "structure_state"
    BOS_STATE = "bos_state"
    BOS_POST_BREAK_STATE = "bos_post_break_state"
    SUPPORT_RESISTANCE_STATE = "support_resistance_state"
    LIQUIDITY_POOL_STATE = "liquidity_pool_state"
    FVG_STATE = "fvg_state"
    ORDER_BLOCK_STATE = "order_block_state"
    DEALING_RANGE_STATE = "dealing_range_state"
    MANIPULATION_STATE = "manipulation_state"
    ENTRY_PATH_STATE = "entry_path_state"
    ENTRY_PATH_STEP = "entry_path_step"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    LIQUIDITY_CONSUMED = "liquidity_consumed"
    LIQUIDITY_RETIRED = "liquidity_retired"
    STRUCTURE_BREAK = "structure_break"
    STRUCTURE_BREAK_FAILED = "structure_break_failed"


class SwingSide(str, Enum):
    HIGH = "high"
    LOW = "low"


class SwingRelation(str, Enum):
    NONE = "none"
    HH = "HH"
    LH = "LH"
    EH = "EH"
    HL = "HL"
    LL = "LL"
    EL = "EL"


class SwingLifecycle(str, Enum):
    FORMING = "forming"
    CONFIRMED = "confirmed"
    BROKEN = "broken"
    FORMATION_FAILED = "formation_failed"


class StructureLifecycle(str, Enum):
    INACTIVE = "inactive"
    FORMING = "forming"
    FORMATION_FAILED = "formation_failed"
    CONFIRMED = "confirmed"
    BROKEN = "broken"


class BOSLifecycle(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class BOSScope(str, Enum):
    CONTINUATION = "continuation"
    OPPOSED = "opposed"
    LOCAL = "local"


class BOSPostBreakState(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SupportResistanceLifecycle(str, Enum):
    ACTIVE = "active"
    TESTED = "tested"
    BROKEN = "broken"
    REACCEPTED = "reaccepted"
    RETIRED = "retired"


class LiquidityPoolLifecycle(str, Enum):
    FORMED = "formed"
    SWEPT = "swept"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class LiquidityInventoryLifecycle(str, Enum):
    VISIBLE = "visible"
    TARGETED = "targeted"
    CONSUMED = "consumed"


class FairValueGapLifecycle(str, Enum):
    OPEN = "open"
    PARTIAL = "partial"
    MITIGATED = "mitigated"
    INVALIDATED = "invalidated"


class FVGQualification(str, Enum):
    RAW = "raw"
    DISPLACEMENT_LINKED = "displacement_linked"


class OrderBlockLifecycle(str, Enum):
    CREATED = "created"
    UNTESTED = "untested"
    MITIGATED = "mitigated"
    FAILED = "failed"


class OrderBlockAttemptOutcome(str, Enum):
    """Mutually exclusive result of one completed-5m OB eligibility pass."""

    NO_ACTIVE_DISPLACEMENT = "no_active_displacement"
    NO_COMPATIBLE_BOS = "no_compatible_bos"
    BREAK_BAR_NOT_IN_DISPLACEMENT = (
        "break_bar_not_in_displacement"
    )
    DUPLICATE_ELIGIBLE_BOS = "duplicate_eligible_bos"
    REVERSE_ANCHOR_CLUSTER_MISSING = (
        "reverse_anchor_cluster_missing"
    )
    ACTIVE_TRANSITION_MISSING = "active_transition_missing"
    INVALID_ANCHOR_WIDTH = "invalid_anchor_width"
    DUPLICATE_ORDER_BLOCK = "duplicate_order_block"
    CREATED = "created"


class DealingRangeLifecycle(str, Enum):
    FORMING = "forming"
    MATURE = "mature"
    BROKEN = "broken"


class ManipulationLifecycle(str, Enum):
    SWEPT = "swept"
    REACCEPTED = "reaccepted"
    ACCEPTED_OUTSIDE = "accepted_outside"


class ManipulationSourceDispositionKind(str, Enum):
    """One mutually-exclusive result for a raw crossed Group 4 source."""

    REJECTED_PRIOR_CLOSE = "rejected_prior_close"
    REJECTED_SOURCE_MISSING_OR_STALE = (
        "rejected_source_missing_or_stale"
    )
    REJECTED_RANGE_INVALIDATED_SAME_CLOCK = (
        "rejected_range_invalidated_same_clock"
    )
    AMBIGUOUS_DUAL_SIDE = "ambiguous_dual_side"
    ATR_UNREADY = "atr_unready"
    BLOCKED_EXISTING_LIVE = "blocked_existing_live"
    BLOCKED_LIVE_RESOLVED_SAME_BAR = (
        "blocked_live_resolved_same_bar"
    )
    SELECTED_PRIMARY = "selected_primary"
    ATTACHED_COINCIDENT_SECONDARY = (
        "attached_coincident_secondary"
    )
    ATTACHED_SAME_SIDE_SECONDARY = (
        "attached_same_side_secondary"
    )


class EntryLocationLifecycle(str, Enum):
    APPROACHING = "approaching"
    IN_ZONE = "in_zone"
    REJECTED = "rejected"
    LEFT = "left"


class QualifiedReacceptanceLifecycle(str, Enum):
    LEFT = "left"
    RECLAIMED = "reclaimed"
    HELD = "held"
    FAILED = "failed"
    CENSORED = "censored"


class PathSequenceLifecycle(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    CENSORED = "censored"


GROUP4_HARD_BOUNDARY_REASONS = frozenset(
    {
        "data_gap_reset",
        "contract_change_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
GROUP5_HARD_BOUNDARY_REASONS = GROUP4_HARD_BOUNDARY_REASONS
GROUP5_CONTEXT_KINDS = frozenset(
    {"zone_return", "pool_reversal"}
)
GROUP5_PATH_STEP_KINDS = frozenset(
    {
        "zone_visible",
        "departure_confirmed",
        "first_pullback",
        "wick_rejection",
        "reference_left",
        "reference_reclaimed",
        "reacceptance_held",
        "reacceptance_failed",
        "micro_bos_simultaneous",
        "micro_bos_confirmed",
        "micro_bos_opposed",
        "micro_bos_ambiguous",
        "location_left",
        "pool_swept",
        "opposite_displacement",
        "opposite_displacement_ambiguous",
        "accepted_outside",
    }
)
GROUP5_SAME_CLOCK_RELATIONS = frozenset(
    {
        "origin",
        "strictly_after",
        "same_clock_known",
        "same_clock_unknown",
    }
)
GROUP5_SAME_CLOCK_REACCEPTANCE_FAILURE_REASONS = frozenset(
    {
        "source_invalidated",
        "accepted_outside",
        "context_closed_before_hold",
    }
)


BOS_FAILURE_REASONS = frozenset(
    {
        "data_gap_reset",
        "contract_change_reset",
        "opposite_structure_break",
        "superseded",
    }
)
BOS_SAME_CLOCK_FAILURE_REASONS = frozenset(
    {
        "opposite_structure_break",
        "superseded",
    }
)
BOS_CONFIRMATION_REASON = "close_beyond_confirmed_swing"
STRUCTURE_FORMATION_FAILURE_REASON = "alignment_lost_before_confirmation"
STRUCTURE_BREAK_FAILURE_REASON = "protected_level_close_break"
SUPPORT_RESISTANCE_RETIREMENT_REASON = "source_evidence_retired"


class VetoCode(str, Enum):
    NONE = "none"
    DATA_ANOMALY = "data_anomaly"
    STALE_DATA = "stale_data"
    SPREAD = "spread"
    COST = "cost"
    DEADLINE = "deadline"
    FILLABILITY = "fillability"
    ACCOUNT_RISK = "account_risk"
    INVALID_STOP = "invalid_stop"
    INVALID_TARGET = "invalid_target"
    REWARD_RISK = "reward_risk"
    NO_PLAN = "no_plan"
    PROTECTION_NOT_TIGHTER = "protection_not_tighter"


def aware_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = (
        value
        if isinstance(value, pd.Timestamp)
        else pd.Timestamp(value)
    )
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return timestamp


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("non-finite continuous value")
    return float(min(high, max(low, value)))


@dataclass(frozen=True)
class Bar:
    """One completed 1m bar; ``start`` is the minute open timestamp."""

    start: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    instrument_id: int
    synthetic_no_trade: bool = False
    data_gap_before_minutes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", aware_timestamp(self.start, name="bar.start"))
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("bar contains non-finite OHLCV")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("bar high/low does not contain open and close")
        if self.high < self.low or self.volume < 0:
            raise ValueError("bar range or volume is invalid")
        if not self.symbol or int(self.instrument_id) < 0:
            raise ValueError("bar contract identity is invalid")
        if int(self.data_gap_before_minutes) < 0:
            raise ValueError("bar data-gap duration cannot be negative")
        if self.synthetic_no_trade and self.data_gap_before_minutes:
            raise ValueError("synthetic no-trade bar cannot also begin a data gap")

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class Candle:
    timeframe: Timeframe
    start: pd.Timestamp
    end: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    instrument_id: int
    observed_minutes: int
    expected_minutes: int
    complete: bool
    real_minutes: int | None = None
    synthetic_minutes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", aware_timestamp(self.start, name="candle.start"))
        object.__setattr__(self, "end", aware_timestamp(self.end, name="candle.end"))
        if self.end <= self.start:
            raise ValueError("candle end must follow start")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("candle contains non-finite OHLCV")
        if (
            self.high < max(self.open, self.close)
            or self.low > min(self.open, self.close)
            or self.high < self.low
            or self.volume < 0
        ):
            raise ValueError("candle OHLC is invalid")
        if not self.symbol or int(self.instrument_id) < 0:
            raise ValueError("candle contract identity is invalid")
        if (
            type(self.observed_minutes) is not int
            or type(self.expected_minutes) is not int
            or type(self.synthetic_minutes) is not int
            or self.observed_minutes > self.expected_minutes
            or self.observed_minutes <= 0
            or self.synthetic_minutes < 0
            or self.synthetic_minutes > self.observed_minutes
            or (
                self.complete
                and self.observed_minutes != self.expected_minutes
            )
        ):
            raise ValueError("candle minute coverage is invalid")
        real_minutes = (
            self.observed_minutes - self.synthetic_minutes
            if self.real_minutes is None
            else self.real_minutes
        )
        if (
            type(real_minutes) is not int
            or real_minutes < 0
            or real_minutes + self.synthetic_minutes
            != self.observed_minutes
        ):
            raise ValueError(
                "candle real/synthetic provenance is inconsistent"
            )
        object.__setattr__(self, "real_minutes", real_minutes)

    @property
    def real_completed(self) -> bool:
        return bool(self.complete and self.synthetic_minutes == 0)


def candle_identity(candle: Candle, *, tick_size: float) -> str:
    """Return one shared candle identity for every semantic reducer."""

    tick = float(tick_size)
    if not math.isfinite(tick) or tick <= 0.0:
        raise ValueError("candle identity requires a positive tick size")

    def ticks(value: float) -> int:
        scaled = float(value) / tick
        rounded = round(scaled)
        if not math.isfinite(scaled) or abs(scaled - rounded) > 1e-6:
            raise ValueError("candle identity received an off-grid price")
        return int(rounded)

    parts = (
        "candle-v1",
        candle.timeframe.value,
        candle.start.isoformat(),
        candle.end.isoformat(),
        str(ticks(candle.open)),
        str(ticks(candle.high)),
        str(ticks(candle.low)),
        str(ticks(candle.close)),
        format(float(candle.volume), ".17g"),
        candle.symbol,
        str(candle.instrument_id),
        str(candle.observed_minutes),
        str(candle.expected_minutes),
        str(candle.real_minutes),
        str(candle.synthetic_minutes),
        str(candle.complete),
    )
    raw = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LiquidityLevel:
    level_id: str
    timeframe: Timeframe
    side: str
    price: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    touches: int
    swept: bool = False

    def __post_init__(self) -> None:
        if self.side not in {"above", "below"}:
            raise ValueError("liquidity side must be above or below")
        if not math.isfinite(float(self.price)) or self.price <= 0 or self.touches < 0:
            raise ValueError("liquidity price or touch count is invalid")
        object.__setattr__(
            self, "formed_at", aware_timestamp(self.formed_at, name="level.formed_at")
        )
        object.__setattr__(
            self, "confirmed_at", aware_timestamp(self.confirmed_at, name="level.confirmed_at")
        )
        if self.confirmed_at < self.formed_at:
            raise ValueError("liquidity confirmation cannot predate formation")


@dataclass(frozen=True)
class SwingPoint:
    swing_id: str
    timeframe: Timeframe
    symbol: str
    instrument_id: int
    side: SwingSide
    price: float
    price_ticks: int
    pivot_start: pd.Timestamp
    pivot_end: pd.Timestamp
    observed_at: pd.Timestamp
    confirmed_at: pd.Timestamp | None
    lifecycle: SwingLifecycle
    relation: SwingRelation = SwingRelation.NONE
    prior_same_side_id: str | None = None
    delta_ticks: int = 0
    delta_points: float = 0.0
    magnitude_atr: float = 0.0
    age_bars: int = 0
    broken_at: pd.Timestamp | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.swing_id or not self.symbol or int(self.instrument_id) < 0:
            raise ValueError("swing identity is invalid")
        if not math.isfinite(float(self.price)) or self.price <= 0:
            raise ValueError("swing price is invalid")
        for name in ("pivot_start", "pivot_end", "observed_at"):
            object.__setattr__(
                self,
                name,
                aware_timestamp(getattr(self, name), name=f"swing.{name}"),
            )
        if self.pivot_end <= self.pivot_start:
            raise ValueError("swing pivot end must follow its start")
        if self.observed_at < self.pivot_end:
            raise ValueError("swing cannot be observed before its pivot completes")
        if self.confirmed_at is not None:
            object.__setattr__(
                self,
                "confirmed_at",
                aware_timestamp(self.confirmed_at, name="swing.confirmed_at"),
            )
            if self.confirmed_at <= self.pivot_end:
                raise ValueError("swing confirmation must follow its pivot end")
            if self.observed_at != self.confirmed_at:
                raise ValueError(
                    "resolved swing must be observed at its confirmation clock"
                )
        if self.broken_at is not None:
            object.__setattr__(
                self,
                "broken_at",
                aware_timestamp(self.broken_at, name="swing.broken_at"),
            )
            if self.confirmed_at is None or self.broken_at <= self.confirmed_at:
                raise ValueError("swing break must follow confirmation")
        if self.lifecycle in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}:
            if self.confirmed_at is None:
                raise ValueError("confirmed or broken swing requires confirmed_at")
        elif self.confirmed_at is not None:
            raise ValueError("forming or failed swing cannot have confirmed_at")
        if (
            self.lifecycle is SwingLifecycle.FORMING
            and self.observed_at != self.pivot_end
        ):
            raise ValueError(
                "forming swing must be observed when its pivot completes"
            )
        if (
            self.lifecycle is SwingLifecycle.FORMATION_FAILED
            and self.observed_at <= self.pivot_end
        ):
            raise ValueError(
                "failed swing formation must resolve after its pivot completes"
            )
        if self.side is SwingSide.HIGH and self.relation not in {
            SwingRelation.NONE,
            SwingRelation.HH,
            SwingRelation.LH,
            SwingRelation.EH,
        }:
            raise ValueError("high swing has a low-side relation")
        if self.side is SwingSide.LOW and self.relation not in {
            SwingRelation.NONE,
            SwingRelation.HL,
            SwingRelation.LL,
            SwingRelation.EL,
        }:
            raise ValueError("low swing has a high-side relation")
        if (
            self.lifecycle
            in {SwingLifecycle.FORMING, SwingLifecycle.FORMATION_FAILED}
            and self.relation is not SwingRelation.NONE
        ):
            raise ValueError("unconfirmed swing cannot have a relation")
        if self.lifecycle is SwingLifecycle.BROKEN and self.broken_at is None:
            raise ValueError("broken swing requires broken_at")
        if self.lifecycle is not SwingLifecycle.BROKEN and self.broken_at is not None:
            raise ValueError("only broken swing may have broken_at")
        expected_reason = {
            SwingLifecycle.FORMATION_FAILED: "right_side_invalidated",
            SwingLifecycle.BROKEN: "close_beyond_swing",
        }.get(self.lifecycle)
        if expected_reason is None and self.failure_reason is not None:
            raise ValueError(
                "open or confirmed swing cannot carry a failure reason"
            )
        if expected_reason is not None and self.failure_reason != expected_reason:
            raise ValueError(
                "terminal swing requires its registered failure reason"
            )
        if self.age_bars < 0:
            raise ValueError("swing age cannot be negative")
        if not all(
            math.isfinite(float(value))
            for value in (self.delta_points, self.magnitude_atr)
        ) or self.magnitude_atr < 0:
            raise ValueError("swing magnitude is invalid")


@dataclass(frozen=True)
class StructureSequenceState:
    structure_id: str | None
    timeframe: Timeframe
    direction: Direction
    lifecycle: StructureLifecycle
    formed_at: pd.Timestamp | None
    confirmed_at: pd.Timestamp | None
    broken_at: pd.Timestamp | None
    high_run: int
    low_run: int
    sequence_count: int
    latest_high_id: str | None
    latest_low_id: str | None
    protected_swing_id: str | None
    protected_price: float | None
    cumulative_magnitude_atr: float
    age_bars: int
    failure_reason: str | None = None
    formation_failed_at: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        for name in (
            "formed_at",
            "confirmed_at",
            "broken_at",
            "formation_failed_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"structure.{name}"),
                )
        if min(self.high_run, self.low_run, self.sequence_count, self.age_bars) < 0:
            raise ValueError("structure counts and age cannot be negative")
        if self.sequence_count != min(self.high_run, self.low_run):
            raise ValueError("structure sequence_count must be the weakest side run")
        if (
            not math.isfinite(float(self.cumulative_magnitude_atr))
            or self.cumulative_magnitude_atr < 0
        ):
            raise ValueError(
                "structure magnitude must be finite and non-negative"
            )
        if self.lifecycle is StructureLifecycle.INACTIVE:
            if any(
                value is not None
                for value in (
                    self.structure_id,
                    self.formed_at,
                    self.confirmed_at,
                    self.broken_at,
                    self.latest_high_id,
                    self.latest_low_id,
                    self.protected_swing_id,
                    self.protected_price,
                    self.formation_failed_at,
                    self.failure_reason,
                )
            ):
                raise ValueError("inactive structure cannot retain active state")
            if (
                self.high_run
                or self.low_run
                or self.sequence_count
                or self.age_bars
                or self.cumulative_magnitude_atr
            ):
                raise ValueError("inactive structure counters must be zero")
            return
        if not self.structure_id or self.formed_at is None:
            raise ValueError("active structure requires identity and formed_at")
        if (
            self.confirmed_at is not None
            and self.confirmed_at < self.formed_at
        ):
            raise ValueError("structure confirmation cannot predate formation")
        if self.broken_at is not None and (
            self.confirmed_at is None
            or self.broken_at <= self.confirmed_at
        ):
            raise ValueError("structure break must follow confirmation")
        if (
            self.formation_failed_at is not None
            and self.formation_failed_at <= self.formed_at
        ):
            raise ValueError(
                "structure formation failure must follow formation"
            )
        if self.lifecycle in {
            StructureLifecycle.CONFIRMED,
            StructureLifecycle.BROKEN,
        }:
            if (
                self.confirmed_at is None
                or self.protected_swing_id is None
                or self.protected_price is None
            ):
                raise ValueError(
                    "confirmed or broken structure requires confirmation and protection"
                )
        elif any(
            value is not None
            for value in (
                self.confirmed_at,
                self.broken_at,
                self.protected_swing_id,
                self.protected_price,
            )
        ):
            raise ValueError("forming structure cannot be confirmed or protected")
        if self.lifecycle is StructureLifecycle.BROKEN and self.broken_at is None:
            raise ValueError("broken structure requires broken_at")
        if (
            self.lifecycle is not StructureLifecycle.BROKEN
            and self.broken_at is not None
        ):
            raise ValueError("only broken structure may have broken_at")
        if self.lifecycle is StructureLifecycle.FORMATION_FAILED:
            if (
                self.formation_failed_at is None
                or self.failure_reason
                != STRUCTURE_FORMATION_FAILURE_REASON
            ):
                raise ValueError(
                    "failed structure formation requires its registered "
                    "clock and reason"
                )
        elif self.formation_failed_at is not None:
            raise ValueError(
                "only failed structure formation may have formation_failed_at"
            )
        if self.lifecycle is StructureLifecycle.BROKEN:
            if self.failure_reason != STRUCTURE_BREAK_FAILURE_REASON:
                raise ValueError(
                    "broken structure requires its registered failure reason"
                )
        elif (
            self.lifecycle is not StructureLifecycle.FORMATION_FAILED
            and self.failure_reason is not None
        ):
            raise ValueError(
                "only terminal structure may carry a failure reason"
            )
        if self.protected_price is not None and (
            not math.isfinite(float(self.protected_price))
            or self.protected_price <= 0
        ):
            raise ValueError("structure protected price is invalid")


@dataclass(frozen=True)
class BreakOfStructureState:
    bos_id: str
    timeframe: Timeframe
    direction: Direction
    lifecycle: BOSLifecycle
    scope: BOSScope
    target_swing_id: str
    source_structure_id: str | None
    target_price: float
    target_ticks: int
    pending_at: pd.Timestamp
    resolved_at: pd.Timestamp | None
    age_bars: int
    attempt_count: int = 0
    last_attempt_at: pd.Timestamp | None = None
    attempt_clocks: tuple[pd.Timestamp, ...] = ()
    failure_reason: str | None = None
    strength: float = 0.0
    break_bar_id: str | None = None
    break_distance_atr: float | None = None
    source_displacement_id: str | None = None
    mss_qualified: bool = False
    post_break_state: BOSPostBreakState | None = None
    accepted_at: pd.Timestamp | None = None
    rejected_at: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not self.bos_id or not self.target_swing_id:
            raise ValueError("BOS identity and target are required")
        if not math.isfinite(float(self.target_price)) or self.target_price <= 0:
            raise ValueError("BOS target price is invalid")
        object.__setattr__(
            self,
            "pending_at",
            aware_timestamp(self.pending_at, name="bos.pending_at"),
        )
        for name in (
            "resolved_at",
            "last_attempt_at",
            "accepted_at",
            "rejected_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"bos.{name}"),
                )
        normalized_attempts = tuple(
            aware_timestamp(value, name="bos.attempt_clocks")
            for value in self.attempt_clocks
        )
        object.__setattr__(self, "attempt_clocks", normalized_attempts)
        if (
            type(self.age_bars) is not int
            or type(self.attempt_count) is not int
            or min(self.age_bars, self.attempt_count) < 0
        ):
            raise ValueError(
                "BOS age and attempt count must be non-negative integers"
            )
        if self.lifecycle is BOSLifecycle.PENDING and self.resolved_at is not None:
            raise ValueError("pending BOS cannot have resolved_at")
        if self.lifecycle is not BOSLifecycle.PENDING and self.resolved_at is None:
            raise ValueError("resolved BOS requires resolved_at")
        if self.resolved_at is not None and self.resolved_at <= self.pending_at:
            raise ValueError("BOS resolution must follow its pending clock")
        if self.last_attempt_at is not None and (
            self.last_attempt_at <= self.pending_at
            or (
                self.resolved_at is not None
                and self.last_attempt_at > self.resolved_at
            )
        ):
            raise ValueError("BOS wick-attempt clock is outside its lifecycle")
        if (
            len(normalized_attempts) != self.attempt_count
            or len(normalized_attempts) != len(set(normalized_attempts))
            or normalized_attempts != tuple(sorted(normalized_attempts))
            or any(value <= self.pending_at for value in normalized_attempts)
            or (
                self.resolved_at is not None
                and any(
                    (
                        value >= self.resolved_at
                        if self.lifecycle is BOSLifecycle.CONFIRMED
                        else value > self.resolved_at
                    )
                    for value in normalized_attempts
                )
            )
            or (
                (not normalized_attempts and self.last_attempt_at is not None)
                or (
                    normalized_attempts
                    and self.last_attempt_at != normalized_attempts[-1]
                )
            )
        ):
            raise ValueError("BOS wick-attempt history is inconsistent")
        if self.scope is BOSScope.LOCAL and self.source_structure_id is not None:
            raise ValueError("local BOS cannot claim a source structure")
        if (
            self.scope in {BOSScope.CONTINUATION, BOSScope.OPPOSED}
            and self.source_structure_id is None
        ):
            raise ValueError("structural BOS requires its source structure")
        if (
            self.lifecycle is BOSLifecycle.FAILED
            and self.failure_reason not in BOS_FAILURE_REASONS
        ):
            raise ValueError("failed BOS requires a registered failure reason")
        if (
            self.lifecycle is not BOSLifecycle.FAILED
            and self.failure_reason is not None
        ):
            raise ValueError("only failed BOS may carry a failure reason")
        raw_strength = float(self.strength)
        if not math.isfinite(raw_strength) or raw_strength < 0.0:
            raise ValueError("BOS strength must be finite and non-negative")
        if (
            self.lifecycle is not BOSLifecycle.CONFIRMED
            and raw_strength != 0.0
        ):
            raise ValueError(
                "only confirmed BOS may carry break strength"
            )
        object.__setattr__(self, "strength", clamp(raw_strength))
        if (
            self.lifecycle is BOSLifecycle.FAILED
            and self.last_attempt_at is not None
            and self.last_attempt_at == self.resolved_at
            and self.failure_reason not in BOS_SAME_CLOCK_FAILURE_REASONS
        ):
            raise ValueError(
                "same-clock failed BOS requires supersession or "
                "structural invalidation"
            )
        confirmed = self.lifecycle is BOSLifecycle.CONFIRMED
        break_distance = self.break_distance_atr
        if confirmed:
            if (
                not self.break_bar_id
                or break_distance is None
                or not math.isfinite(float(break_distance))
                or float(break_distance) <= 0.0
                or not isinstance(self.post_break_state, BOSPostBreakState)
            ):
                raise ValueError(
                    "confirmed BOS requires its break bar, distance and "
                    "post-break state"
                )
        elif any(
            value is not None
            for value in (
                self.break_bar_id,
                break_distance,
                self.source_displacement_id,
                self.post_break_state,
                self.accepted_at,
                self.rejected_at,
            )
        ) or self.mss_qualified:
            raise ValueError(
                "unconfirmed BOS cannot carry break or MSS evidence"
            )
        if type(self.mss_qualified) is not bool:
            raise ValueError("BOS MSS qualification must be boolean")
        if bool(self.source_displacement_id) != self.mss_qualified:
            raise ValueError(
                "BOS MSS qualification and displacement identity disagree"
            )
        if self.source_displacement_id == "":
            raise ValueError("BOS displacement identity cannot be empty")
        if self.mss_qualified and self.scope is not BOSScope.OPPOSED:
            raise ValueError("only an opposed BOS can qualify as MSS")
        if confirmed:
            if self.post_break_state is BOSPostBreakState.PENDING:
                if self.accepted_at is not None or self.rejected_at is not None:
                    raise ValueError(
                        "pending post-break state cannot be resolved"
                    )
            elif self.post_break_state is BOSPostBreakState.ACCEPTED:
                if (
                    self.accepted_at is None
                    or self.accepted_at <= self.resolved_at
                    or self.rejected_at is not None
                ):
                    raise ValueError(
                        "accepted BOS requires one later acceptance clock"
                    )
            elif (
                self.rejected_at is None
                or self.rejected_at <= self.resolved_at
                or self.accepted_at is not None
            ):
                raise ValueError(
                    "rejected BOS requires one later rejection clock"
                )


@dataclass(frozen=True)
class CandleStructureState:
    """Single completed-bar geometry with explicit zero-range provenance."""

    timeframe: Timeframe
    start: pd.Timestamp
    observed_at: pd.Timestamp
    range_points: float
    body_points: float
    upper_wick_points: float
    lower_wick_points: float
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    close_location: float
    direction: int
    real_completed: bool
    zero_range: bool
    body_class: str
    range_class: str
    dominant_wick: str
    close_class: str
    anomalies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start",
            aware_timestamp(self.start, name="candle_structure.start"),
        )
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="candle_structure.observed_at",
            ),
        )
        if self.observed_at <= self.start:
            raise ValueError("candle structure clock must follow its start")
        values = (
            self.range_points,
            self.body_points,
            self.upper_wick_points,
            self.lower_wick_points,
            self.body_ratio,
            self.upper_wick_ratio,
            self.lower_wick_ratio,
            self.close_location,
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in values
        ):
            raise ValueError("candle structure contains invalid geometry")
        if self.direction not in {-1, 0, 1}:
            raise ValueError("candle structure direction must be -1, 0 or 1")
        if self.body_class not in {"doji", "small", "normal", "large"}:
            raise ValueError("candle structure body class is invalid")
        if self.range_class not in {"compressed", "normal", "expanded"}:
            raise ValueError("candle structure range class is invalid")
        if self.dominant_wick not in {"upper", "lower", "balanced", "none"}:
            raise ValueError("candle structure dominant wick is invalid")
        if self.close_class not in {"near_high", "middle", "near_low"}:
            raise ValueError("candle structure close class is invalid")
        if type(self.real_completed) is not bool or type(self.zero_range) is not bool:
            raise ValueError("candle structure provenance flags must be boolean")
        if (
            len(self.anomalies) != len(set(self.anomalies))
            or any(not isinstance(value, str) or not value for value in self.anomalies)
        ):
            raise ValueError("candle structure anomalies are invalid")
        provenance_anomaly = "synthetic_or_partial_completed_candle"
        if self.real_completed == (provenance_anomaly in self.anomalies):
            raise ValueError("candle structure provenance is inconsistent")
        if not 0.0 <= self.close_location <= 1.0:
            raise ValueError("candle close location must be in [0, 1]")
        tolerance = max(1e-9, self.range_points * 1e-9)
        if abs(
            self.body_points
            + self.upper_wick_points
            + self.lower_wick_points
            - self.range_points
        ) > tolerance:
            raise ValueError("candle components do not reconstruct its range")
        if self.zero_range:
            if (
                self.range_points
                or self.body_points
                or self.upper_wick_points
                or self.lower_wick_points
                or self.body_ratio
                or self.upper_wick_ratio
                or self.lower_wick_ratio
                or self.close_location != 0.5
                or self.direction
                or self.body_class != "doji"
                or self.range_class != "compressed"
                or self.dominant_wick != "none"
                or self.close_class != "middle"
            ):
                raise ValueError("zero-range candle structure is inconsistent")
        elif (
            self.range_points <= 0.0
            or abs(
                self.body_ratio
                + self.upper_wick_ratio
                + self.lower_wick_ratio
                - 1.0
            )
            > 1e-9
        ):
            raise ValueError("nonzero candle ratios must partition the range")
        elif any(
            not math.isclose(
                ratio,
                points / self.range_points,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for points, ratio in (
                (self.body_points, self.body_ratio),
                (self.upper_wick_points, self.upper_wick_ratio),
                (self.lower_wick_points, self.lower_wick_ratio),
            )
        ):
            raise ValueError(
                "candle ratios do not match their point components"
            )
        if (self.body_points == 0.0) != (self.direction == 0):
            raise ValueError("candle body and direction are inconsistent")


@dataclass(frozen=True)
class SupportResistanceState:
    """A frozen reaction zone with an explicit causal source identity."""

    zone_id: str
    timeframe: Timeframe
    side: str
    lower_bound: float
    upper_bound: float
    anchor_price: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    lifecycle: SupportResistanceLifecycle
    member_swing_ids: tuple[str, ...]
    touch_times: tuple[pd.Timestamp, ...]
    reaction_magnitudes_atr: tuple[float, ...]
    age_bars: int
    strength: float
    tested_at: pd.Timestamp | None = None
    broken_at: pd.Timestamp | None = None
    reaccepted_at: pd.Timestamp | None = None
    retired_at: pd.Timestamp | None = None
    transition_reason: str | None = None
    total_touch_count: int | None = None
    source_kind: str = "structural_swing"
    structural_rank: str = "internal"
    is_protected_swing: bool = False
    zone_role: str = "both"
    visibility_strength: float = 0.0
    reaction_quality: float = 0.0
    freshness: float = 1.0
    depletion_risk: float = 0.0
    metadata_observed_at: pd.Timestamp | None = None
    source_ids: tuple[str, ...] = ()
    range_id: str | None = None
    source_zone_id: str | None = None

    def __post_init__(self) -> None:
        member_swing_ids = tuple(self.member_swing_ids)
        source_ids = tuple(self.source_ids)
        object.__setattr__(self, "member_swing_ids", member_swing_ids)
        object.__setattr__(self, "source_ids", source_ids)
        if not self.zone_id or self.side not in {"support", "resistance"}:
            raise ValueError("support/resistance identity or side is invalid")
        if (
            self.source_kind
            not in {
                "structural_swing",
                "previous_session",
                "previous_day",
                "previous_week",
                "range_boundary",
            }
            or self.structural_rank not in {"internal", "external"}
            or type(self.is_protected_swing) is not bool
            or self.zone_role
            not in {"reaction_zone", "liquidity_target", "both"}
        ):
            raise ValueError("support/resistance source metadata is invalid")
        if not (
            math.isfinite(float(self.lower_bound))
            and math.isfinite(float(self.upper_bound))
            and math.isfinite(float(self.anchor_price))
            and 0 < self.lower_bound <= self.anchor_price <= self.upper_bound
        ):
            raise ValueError("support/resistance bounds are invalid")
        for name in (
            "formed_at",
            "confirmed_at",
            "tested_at",
            "broken_at",
            "reaccepted_at",
            "retired_at",
            "metadata_observed_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(
                        value,
                        name=f"support_resistance.{name}",
                    ),
                )
        if self.metadata_observed_at is None:
            object.__setattr__(
                self, "metadata_observed_at", self.confirmed_at
            )
        elif self.metadata_observed_at < self.confirmed_at:
            raise ValueError(
                "support/resistance metadata cannot predate confirmation"
            )
        if self.confirmed_at < self.formed_at:
            raise ValueError("zone confirmation cannot predate formation")
        if (
            len(member_swing_ids) != len(set(member_swing_ids))
            or len(source_ids) != len(set(source_ids))
            or any(
                not isinstance(value, str) or not value
                for value in (*member_swing_ids, *source_ids)
            )
        ):
            raise ValueError("support/resistance source identity is invalid")
        if self.source_kind == "structural_swing":
            if (
                not member_swing_ids
                or source_ids
                or self.range_id is not None
                or self.source_zone_id is not None
            ):
                raise ValueError(
                    "structural support/resistance requires real swing members"
                )
        elif self.source_kind in {
            "previous_session",
            "previous_day",
            "previous_week",
        }:
            if (
                member_swing_ids
                or len(source_ids) != 1
                or self.range_id is not None
                or self.source_zone_id is not None
            ):
                raise ValueError(
                    "reference support/resistance requires one non-swing source"
                )
        elif (
            member_swing_ids
            or not self.range_id
            or not self.source_zone_id
            or self.range_id == self.source_zone_id
            or not {self.range_id, self.source_zone_id}.issubset(source_ids)
        ):
            raise ValueError(
                "range-boundary support/resistance source identity is invalid"
            )
        touches = tuple(
            aware_timestamp(value, name="support_resistance.touch_times")
            for value in self.touch_times
        )
        object.__setattr__(self, "touch_times", touches)
        total_touch_count = (
            len(touches)
            if self.total_touch_count is None
            else self.total_touch_count
        )
        if (
            type(total_touch_count) is not int
            or total_touch_count < len(touches)
        ):
            raise ValueError(
                "support/resistance total touch count is invalid"
            )
        object.__setattr__(
            self,
            "total_touch_count",
            total_touch_count,
        )
        if (
            not touches
            or (
                self.source_kind == "structural_swing"
                and len(touches) != len(member_swing_ids)
            )
            or len(self.reaction_magnitudes_atr) != len(touches)
            or touches != tuple(sorted(touches))
            or touches[0] != self.confirmed_at
            or any(value < 0 or not math.isfinite(float(value))
                   for value in self.reaction_magnitudes_atr)
            or type(self.age_bars) is not int
            or self.age_bars < 0
        ):
            raise ValueError("support/resistance touch history is invalid")
        object.__setattr__(self, "strength", clamp(self.strength))
        for name in (
            "visibility_strength",
            "reaction_quality",
            "freshness",
            "depletion_risk",
        ):
            raw_value = float(getattr(self, name))
            if not math.isfinite(raw_value):
                raise ValueError(
                    "support/resistance quality fields must be finite"
                )
            object.__setattr__(self, name, clamp(raw_value))
        if self.is_protected_swing and self.structural_rank != "external":
            raise ValueError("protected swing zone must be externally ranked")
        if self.tested_at is not None and (
            len(touches) < 2 or self.tested_at != touches[1]
        ):
            raise ValueError("tested zone requires its second confirmed touch")
        if self.broken_at is not None and self.broken_at <= self.confirmed_at:
            raise ValueError("zone break must follow confirmation")
        if self.reaccepted_at is not None and (
            self.broken_at is None or self.reaccepted_at <= self.broken_at
        ):
            raise ValueError("zone reacceptance must follow its break")
        if self.retired_at is not None and self.retired_at <= max(
            value
            for value in (
                self.confirmed_at,
                self.tested_at,
                self.broken_at,
                self.reaccepted_at,
            )
            if value is not None
        ):
            raise ValueError(
                "zone retirement must follow its latest evidence clock"
            )
        if self.lifecycle is SupportResistanceLifecycle.ACTIVE:
            if total_touch_count != 1 or len(touches) != 1 or any(
                value is not None
                for value in (
                    self.tested_at,
                    self.broken_at,
                    self.reaccepted_at,
                    self.retired_at,
                    self.transition_reason,
                )
            ):
                raise ValueError("active zone lifecycle is inconsistent")
        elif self.lifecycle is SupportResistanceLifecycle.TESTED:
            if (
                total_touch_count < 2
                or len(touches) < 2
                or self.tested_at is None
                or self.broken_at is not None
                or self.reaccepted_at is not None
                or self.retired_at is not None
                or self.transition_reason is not None
            ):
                raise ValueError("tested zone lifecycle is inconsistent")
        elif self.lifecycle is SupportResistanceLifecycle.BROKEN:
            if (
                self.broken_at is None
                or self.reaccepted_at is not None
                or self.retired_at is not None
                or self.transition_reason != "close_beyond_frozen_zone"
            ):
                raise ValueError("broken zone lifecycle is inconsistent")
        elif self.lifecycle is SupportResistanceLifecycle.REACCEPTED:
            if (
                self.broken_at is None
                or self.reaccepted_at is None
                or self.retired_at is not None
                or self.transition_reason
                != "close_reentered_frozen_zone"
            ):
                raise ValueError(
                    "reaccepted zone lifecycle is inconsistent"
                )
        elif (
            self.retired_at is None
            or self.reaccepted_at is not None
            or self.transition_reason
            != SUPPORT_RESISTANCE_RETIREMENT_REASON
        ):
            raise ValueError("retired zone lifecycle is inconsistent")

    @property
    def touch_count(self) -> int:
        return int(self.total_touch_count)

    @property
    def causal_source_ids(self) -> tuple[str, ...]:
        """Return real upstream identities without manufacturing swing IDs."""

        return tuple(
            dict.fromkeys(
                (
                    *self.member_swing_ids,
                    *self.source_ids,
                    *((self.range_id,) if self.range_id is not None else ()),
                    *(
                        (self.source_zone_id,)
                        if self.source_zone_id is not None
                        else ()
                    ),
                )
            )
        )


@dataclass(frozen=True)
class LiquidityPoolState:
    """Equal-high/low pool separated from its later sweep resolution."""

    pool_id: str
    timeframe: Timeframe
    side: str
    lower_bound: float
    upper_bound: float
    midpoint: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    lifecycle: LiquidityPoolLifecycle
    member_swing_ids: tuple[str, ...]
    touch_times: tuple[pd.Timestamp, ...]
    age_bars: int
    strength: float
    swept_at: pd.Timestamp | None = None
    sweep_extreme: float | None = None
    close_outside_on_sweep: bool | None = None
    resolved_at: pd.Timestamp | None = None
    resolution_reason: str | None = None
    total_touch_count: int | None = None

    def __post_init__(self) -> None:
        if not self.pool_id or self.side not in {"above", "below"}:
            raise ValueError("liquidity pool identity or side is invalid")
        if not (
            math.isfinite(float(self.lower_bound))
            and math.isfinite(float(self.upper_bound))
            and math.isfinite(float(self.midpoint))
            and 0 < self.lower_bound <= self.midpoint <= self.upper_bound
        ):
            raise ValueError("liquidity pool bounds are invalid")
        for name in ("formed_at", "confirmed_at", "swept_at", "resolved_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"liquidity_pool.{name}"),
                )
        touches = tuple(
            aware_timestamp(value, name="liquidity_pool.touch_times")
            for value in self.touch_times
        )
        object.__setattr__(self, "touch_times", touches)
        total_touch_count = (
            len(touches)
            if self.total_touch_count is None
            else self.total_touch_count
        )
        if (
            type(total_touch_count) is not int
            or total_touch_count < len(touches)
            or total_touch_count < 2
        ):
            raise ValueError("liquidity pool total touch count is invalid")
        object.__setattr__(
            self,
            "total_touch_count",
            total_touch_count,
        )
        if (
            self.confirmed_at < self.formed_at
            or len(self.member_swing_ids) < 2
            or len(self.member_swing_ids) != len(set(self.member_swing_ids))
            or len(touches) != len(self.member_swing_ids)
            or touches != tuple(sorted(touches))
            or self.confirmed_at != touches[1]
            or type(self.age_bars) is not int
            or self.age_bars < 0
        ):
            raise ValueError("liquidity pool formation history is invalid")
        object.__setattr__(self, "strength", clamp(self.strength))
        if self.sweep_extreme is not None and (
            not math.isfinite(float(self.sweep_extreme))
            or self.sweep_extreme <= 0
        ):
            raise ValueError("liquidity pool sweep extreme is invalid")
        if (
            self.close_outside_on_sweep is not None
            and type(self.close_outside_on_sweep) is not bool
        ):
            raise ValueError(
                "liquidity pool sweep-close state must be boolean"
            )
        if self.swept_at is not None and self.swept_at <= self.confirmed_at:
            raise ValueError("pool sweep must follow confirmation")
        if self.resolved_at is not None and (
            self.swept_at is None or self.resolved_at <= self.swept_at
        ):
            raise ValueError("pool resolution must follow its sweep")
        if self.lifecycle is LiquidityPoolLifecycle.FORMED:
            if any(
                value is not None
                for value in (
                    self.swept_at,
                    self.sweep_extreme,
                    self.close_outside_on_sweep,
                    self.resolved_at,
                    self.resolution_reason,
                )
            ):
                raise ValueError("formed liquidity pool lifecycle is inconsistent")
        elif self.lifecycle is LiquidityPoolLifecycle.SWEPT:
            if (
                self.swept_at is None
                or self.sweep_extreme is None
                or self.close_outside_on_sweep is None
                or self.resolved_at is not None
                or self.resolution_reason is not None
            ):
                raise ValueError("swept liquidity pool lifecycle is inconsistent")
        elif self.lifecycle is LiquidityPoolLifecycle.ACCEPTED:
            if (
                self.swept_at is None
                or self.sweep_extreme is None
                or self.close_outside_on_sweep is None
                or self.resolved_at is None
                or self.resolution_reason != "close_held_outside"
            ):
                raise ValueError("accepted liquidity pool lifecycle is inconsistent")
        elif (
            self.swept_at is None
            or self.sweep_extreme is None
            or self.close_outside_on_sweep is None
            or self.resolved_at is None
            or self.resolution_reason != "close_returned_inside"
        ):
            raise ValueError("rejected liquidity pool lifecycle is inconsistent")

    @property
    def touch_count(self) -> int:
        return int(self.total_touch_count)


RANGE_PAIR_FUNNEL_COUNTS = (
    "live_structural_pairs",
    "invalid_geometry_pairs",
    "geometry_valid_pairs",
    "close_outside_pair_pairs",
    "already_admitted_pairs",
    "cold_start_blocked_pairs",
    "same_bar_terminal_blocked_pairs",
    "live_range_blocked_pairs",
    "atr_unready_pairs",
    "eligible_pairs",
    "forming_selected",
)

RANGE_MATURITY_GATE_NAMES = (
    "duration",
    "bilateral_touches",
    "midpoint_crossing",
    "inside_close_fraction",
    "width",
    "compression",
)


@dataclass(frozen=True)
class RangeFormationFunnelSnapshot:
    """One completed-H1 range-selection and maturity-gate diagnostic."""

    observed_at: pd.Timestamp
    pair_counts: tuple[tuple[str, int], ...]
    selected_source_pair_ids: tuple[str, str] | None = None
    selected_range_id: str | None = None
    maturity_range_id: str | None = None
    maturity_gates: tuple[
        tuple[str, float, float, float],
        ...,
    ] = ()
    unmet_maturity_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="range_formation_funnel.observed_at",
            ),
        )
        counts = tuple(self.pair_counts)
        gates = tuple(self.maturity_gates)
        unmet = tuple(self.unmet_maturity_gates)
        object.__setattr__(self, "pair_counts", counts)
        object.__setattr__(self, "maturity_gates", gates)
        object.__setattr__(self, "unmet_maturity_gates", unmet)
        if (
            tuple(name for name, _ in counts)
            != RANGE_PAIR_FUNNEL_COUNTS
            or any(type(value) is not int or value < 0 for _, value in counts)
        ):
            raise ValueError("range pair funnel counts are invalid")
        values = dict(counts)
        if (
            values["live_structural_pairs"]
            != values["invalid_geometry_pairs"]
            + values["geometry_valid_pairs"]
            or values["geometry_valid_pairs"]
            != sum(
                values[name]
                for name in (
                    "close_outside_pair_pairs",
                    "already_admitted_pairs",
                    "cold_start_blocked_pairs",
                    "same_bar_terminal_blocked_pairs",
                    "live_range_blocked_pairs",
                    "atr_unready_pairs",
                    "eligible_pairs",
                )
            )
            or values["forming_selected"] not in {0, 1}
            or values["forming_selected"] > values["eligible_pairs"]
        ):
            raise ValueError("range pair funnel is not conserved")
        selected = self.selected_source_pair_ids
        if selected is not None:
            selected = tuple(selected)
            object.__setattr__(self, "selected_source_pair_ids", selected)
        if (
            (selected is None) != (self.selected_range_id is None)
            or (selected is not None)
            != (values["forming_selected"] == 1)
            or (
                selected is not None
                and (
                    len(selected) != 2
                    or len(set(selected)) != 2
                    or any(
                        not isinstance(value, str) or not value
                        for value in selected
                    )
                    or not self.selected_range_id
                )
            )
        ):
            raise ValueError("selected forming range identity is invalid")
        if self.maturity_range_id is None:
            if gates or unmet:
                raise ValueError(
                    "unevaluated maturity gates cannot carry results"
                )
            return
        if (
            not self.maturity_range_id
            or tuple(row[0] for row in gates)
            != RANGE_MATURITY_GATE_NAMES
            or any(
                len(row) != 4
                or any(
                    not math.isfinite(float(value))
                    for value in row[1:]
                )
                for row in gates
            )
            or any(
                threshold <= 0.0
                or not math.isclose(
                    margin,
                    (
                        threshold - actual
                        if name in {"width", "compression"}
                        else actual - threshold
                    )
                    / threshold,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for name, actual, threshold, margin in gates
            )
            or unmet
            != tuple(row[0] for row in gates if float(row[3]) < 0.0)
        ):
            raise ValueError("range maturity gate diagnostic is invalid")


@dataclass(frozen=True)
class DealingRangeState:
    """One H1 accumulation candidate and its frozen mature range."""

    range_id: str
    protocol_hash: str
    source_group12_protocol_hash: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    lifecycle: DealingRangeLifecycle
    lower_source_zone_id: str
    upper_source_zone_id: str
    lower_source_confirmed_at: pd.Timestamp
    upper_source_confirmed_at: pd.Timestamp
    lower_source_tested_at: pd.Timestamp | None
    upper_source_tested_at: pd.Timestamp | None
    lower_source_member_swing_ids: tuple[str, ...]
    upper_source_member_swing_ids: tuple[str, ...]
    lower_source_lower_bound: float
    lower_source_upper_bound: float
    upper_source_lower_bound: float
    upper_source_upper_bound: float
    formed_at: pd.Timestamp
    mature_at: pd.Timestamp | None
    broken_at: pd.Timestamp | None
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    lower_bound: float
    upper_bound: float
    midpoint: float
    value_price: float
    formation_atr: float
    width_points: float
    width_atr_at_formation: float
    candidate_real_h1_bars: int
    lower_touch_count: int
    upper_touch_count: int
    midpoint_crossings: int
    inside_close_fraction: float
    compression_ratio: float
    narrowness_strength: float
    compression_strength: float
    boundary_test_strength: float
    crossing_strength: float
    strength: float
    age_h1_bars: int
    transition_reason: str | None

    def __post_init__(self) -> None:
        lower_members = tuple(self.lower_source_member_swing_ids)
        upper_members = tuple(self.upper_source_member_swing_ids)
        object.__setattr__(
            self,
            "lower_source_member_swing_ids",
            lower_members,
        )
        object.__setattr__(
            self,
            "upper_source_member_swing_ids",
            upper_members,
        )
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.range_id,
                    self.symbol,
                    self.lower_source_zone_id,
                    self.upper_source_zone_id,
                )
            )
            or self.lower_source_zone_id == self.upper_source_zone_id
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.timeframe is not Timeframe.H1
            or not isinstance(self.lifecycle, DealingRangeLifecycle)
            or not self.protocol_hash
            or not self.source_group12_protocol_hash
        ):
            raise ValueError("dealing-range identity or protocol is invalid")
        if (
            not lower_members
            or not upper_members
            or len(lower_members) != len(set(lower_members))
            or len(upper_members) != len(set(upper_members))
            or any(
                not isinstance(value, str) or not value
                for value in (*lower_members, *upper_members)
            )
        ):
            raise ValueError("dealing-range source swing identities are invalid")
        for name in (
            "lower_source_confirmed_at",
            "upper_source_confirmed_at",
            "formed_at",
            "state_started_at",
            "last_updated_at",
        ):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"dealing_range.{name}",
                ),
            )
        for name in (
            "lower_source_tested_at",
            "upper_source_tested_at",
            "mature_at",
            "broken_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(
                        value,
                        name=f"dealing_range.{name}",
                    ),
                )
        source_geometry = (
            self.lower_source_lower_bound,
            self.lower_source_upper_bound,
            self.upper_source_lower_bound,
            self.upper_source_upper_bound,
        )
        geometry = (
            *source_geometry,
            self.lower_bound,
            self.upper_bound,
            self.midpoint,
            self.value_price,
            self.formation_atr,
            self.width_points,
            self.width_atr_at_formation,
        )
        if (
            any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in geometry
            )
            or self.lower_source_lower_bound
            > self.lower_source_upper_bound
            or self.upper_source_lower_bound
            > self.upper_source_upper_bound
            or self.lower_source_upper_bound
            >= self.upper_source_lower_bound
            or not math.isclose(
                self.lower_bound,
                self.lower_source_lower_bound,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.upper_bound,
                self.upper_source_upper_bound,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.midpoint,
                (self.lower_bound + self.upper_bound) / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.value_price,
                self.midpoint,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.width_points,
                self.upper_bound - self.lower_bound,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.width_atr_at_formation,
                self.width_points / self.formation_atr,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("dealing-range frozen geometry is invalid")
        integer_fields = (
            self.candidate_real_h1_bars,
            self.lower_touch_count,
            self.upper_touch_count,
            self.midpoint_crossings,
            self.age_h1_bars,
        )
        if (
            any(type(value) is not int for value in integer_fields)
            or not 1 <= self.candidate_real_h1_bars <= 24
            or self.lower_touch_count < len(lower_members)
            or self.upper_touch_count < len(upper_members)
            or self.midpoint_crossings < 0
            or self.age_h1_bars < self.candidate_real_h1_bars - 1
        ):
            raise ValueError("dealing-range incremental counts are invalid")
        continuous_fields = (
            self.inside_close_fraction,
            self.compression_ratio,
            self.narrowness_strength,
            self.compression_strength,
            self.boundary_test_strength,
            self.crossing_strength,
            self.strength,
        )
        if (
            any(not math.isfinite(float(value)) for value in continuous_fields)
            or not 0.0 <= self.inside_close_fraction <= 1.0
            or self.compression_ratio < 0.0
            or any(
                not 0.0 <= float(value) <= 1.0
                for value in (
                    self.narrowness_strength,
                    self.compression_strength,
                    self.boundary_test_strength,
                    self.crossing_strength,
                    self.strength,
                )
            )
        ):
            raise ValueError("dealing-range statistics are invalid")
        expected_components = (
            clamp(1.0 - self.width_atr_at_formation / 4.0),
            clamp(1.0 - self.compression_ratio),
            clamp(min(self.lower_touch_count, self.upper_touch_count) / 3.0),
            clamp(self.midpoint_crossings / 4.0),
        )
        actual_components = (
            self.narrowness_strength,
            self.compression_strength,
            self.boundary_test_strength,
            self.crossing_strength,
        )
        expected_strength = (
            sum(expected_components) + self.inside_close_fraction
        ) / 5.0
        if (
            any(
                not math.isclose(
                    actual,
                    expected,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for actual, expected in zip(
                    actual_components,
                    expected_components,
                )
            )
            or not math.isclose(
                self.strength,
                expected_strength,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("dealing-range strength components disagree")
        if (
            self.lower_source_confirmed_at > self.formed_at
            or self.upper_source_confirmed_at > self.formed_at
            or self.state_started_at < self.formed_at
            or self.last_updated_at < self.state_started_at
        ):
            raise ValueError("dealing-range knowledge clock is invalid")
        for confirmed_at, tested_at, touch_count in (
            (
                self.lower_source_confirmed_at,
                self.lower_source_tested_at,
                self.lower_touch_count,
            ),
            (
                self.upper_source_confirmed_at,
                self.upper_source_tested_at,
                self.upper_touch_count,
            ),
        ):
            if tested_at is not None and (
                tested_at <= confirmed_at
                or tested_at > self.last_updated_at
                or touch_count < 2
            ):
                raise ValueError("dealing-range source test clock is invalid")
            if tested_at is None and touch_count > 1:
                raise ValueError(
                    "repeated range-boundary touches require a tested clock"
                )
        if self.mature_at is not None and (
            self.mature_at < self.formed_at
            or self.mature_at > self.last_updated_at
            or self.candidate_real_h1_bars < 8
            or self.lower_touch_count < 2
            or self.upper_touch_count < 2
            or self.midpoint_crossings < 2
            or self.inside_close_fraction < 0.8
            or self.width_atr_at_formation > 4.0
            or self.compression_ratio > 0.8
            or self.lower_source_tested_at is None
            or self.upper_source_tested_at is None
        ):
            raise ValueError("mature dealing-range evidence is invalid")
        if self.broken_at is not None and (
            self.broken_at <= self.formed_at
            or self.broken_at > self.last_updated_at
            or (
                self.mature_at is not None
                and self.broken_at <= self.mature_at
            )
        ):
            raise ValueError("dealing-range break clock is invalid")
        if self.transition_reason == "":
            raise ValueError("dealing-range transition reason cannot be empty")
        if self.lifecycle is DealingRangeLifecycle.FORMING:
            if (
                self.mature_at is not None
                or self.broken_at is not None
                or self.state_started_at != self.formed_at
                or self.candidate_real_h1_bars >= 24
                or self.transition_reason
                not in {None, "source_pair_selected"}
            ):
                raise ValueError("forming dealing-range lifecycle is inconsistent")
        elif self.lifecycle is DealingRangeLifecycle.MATURE:
            if (
                self.mature_at is None
                or self.broken_at is not None
                or self.state_started_at != self.mature_at
                or not self.transition_reason
            ):
                raise ValueError("mature dealing-range lifecycle is inconsistent")
        elif (
            self.broken_at is None
            or self.state_started_at != self.broken_at
            or self.last_updated_at != self.broken_at
            or not self.transition_reason
        ):
            raise ValueError("broken dealing-range lifecycle is inconsistent")


@dataclass(frozen=True)
class ManipulationSourceDisposition:
    """Per-crossing accounting emitted only for the current Group 4 update."""

    source_inventory_item_id: str
    observed_at: pd.Timestamp
    disposition: ManipulationSourceDispositionKind

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_inventory_item_id, str)
            or not self.source_inventory_item_id
            or not isinstance(
                self.disposition,
                ManipulationSourceDispositionKind,
            )
        ):
            raise ValueError(
                "manipulation source disposition is invalid"
            )
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="manipulation_source_disposition.observed_at",
            ),
        )


@dataclass(frozen=True)
class ManipulationState:
    """A typed completed-1m sweep and its multi-bar resolution."""

    manipulation_id: str
    protocol_hash: str
    source_group12_protocol_hash: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    lifecycle: ManipulationLifecycle
    side: str
    source_kind: str
    source_id: str
    source_protocol_hash: str
    source_timeframe: Timeframe
    source_inventory_item_id: str
    source_inventory_lifecycle: LiquidityInventoryLifecycle
    coincident_source_ids: tuple[str, ...]
    source_formed_at: pd.Timestamp
    source_eligible_at: pd.Timestamp
    source_lower_bound: float
    source_upper_bound: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    swept_at: pd.Timestamp
    reaccepted_at: pd.Timestamp | None
    accepted_outside_at: pd.Timestamp | None
    resolved_at: pd.Timestamp | None
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    sweep_extreme: float
    close_outside_on_sweep: bool
    reentry_price: float | None
    resolved_side: str | None
    outside_completed_bars: int
    penetration_atr: float
    strength: float
    age_1m_bars: int
    transition_reason: str | None
    censored_at: pd.Timestamp | None
    reentry_candidate_at: pd.Timestamp | None
    reentry_candidate_price: float | None
    inside_hold_bars: int
    reentry_failed_at: pd.Timestamp | None
    outside_run: int
    outside_run_side: str | None
    deadline_at: pd.Timestamp | None
    deadline_elapsed: bool
    crossed_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        coincident_ids = tuple(self.coincident_source_ids)
        object.__setattr__(
            self,
            "coincident_source_ids",
            coincident_ids,
        )
        crossed_ids = tuple(self.crossed_source_ids)
        object.__setattr__(self, "crossed_source_ids", crossed_ids)
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.manipulation_id,
                    self.symbol,
                    self.source_id,
                    self.source_inventory_item_id,
                )
            )
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.timeframe is not Timeframe.M1
            or not isinstance(self.lifecycle, ManipulationLifecycle)
            or self.side not in {"above", "below"}
            or self.source_kind
            not in {"mature_range_boundary", "formed_liquidity_pool"}
            or not isinstance(self.source_timeframe, Timeframe)
            or not isinstance(
                self.source_inventory_lifecycle,
                LiquidityInventoryLifecycle,
            )
            or self.source_inventory_lifecycle
            is not LiquidityInventoryLifecycle.VISIBLE
            or not self.protocol_hash
            or not self.source_protocol_hash
            or not self.source_group12_protocol_hash
        ):
            raise ValueError(
                "manipulation identity, source or protocol is invalid"
            )
        if (
            len(coincident_ids) != len(set(coincident_ids))
            or any(
                not isinstance(value, str) or not value
                for value in coincident_ids
            )
        ):
            raise ValueError("manipulation coincident source ids are invalid")
        if (
            len(crossed_ids) != len(set(crossed_ids))
            or not crossed_ids
            or crossed_ids[0] != self.source_id
            or not {self.source_id, *coincident_ids}.issubset(
                crossed_ids
            )
        ):
            raise ValueError("manipulation crossed source ids are invalid")
        if (
            self.source_kind == "mature_range_boundary"
            and self.source_timeframe is not Timeframe.H1
        ):
            raise ValueError("range manipulation source must be H1")
        if (
            self.source_kind == "formed_liquidity_pool"
            and self.source_protocol_hash
            != self.source_group12_protocol_hash
        ):
            raise ValueError(
                "pool manipulation source must bind the Group 1-2 protocol"
            )
        for name in (
            "source_formed_at",
            "source_eligible_at",
            "formed_at",
            "confirmed_at",
            "swept_at",
            "state_started_at",
            "last_updated_at",
        ):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"manipulation.{name}",
                ),
            )
        for name in (
            "reaccepted_at",
            "accepted_outside_at",
            "resolved_at",
            "censored_at",
            "reentry_candidate_at",
            "reentry_failed_at",
            "deadline_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(
                        value,
                        name=f"manipulation.{name}",
                    ),
                )
        if (
            not math.isfinite(float(self.source_lower_bound))
            or not math.isfinite(float(self.source_upper_bound))
            or not math.isfinite(float(self.sweep_extreme))
            or self.source_lower_bound <= 0.0
            or self.source_upper_bound < self.source_lower_bound
            or self.sweep_extreme <= 0.0
            or (
                self.side == "above"
                and self.sweep_extreme <= self.source_upper_bound
            )
            or (
                self.side == "below"
                and self.sweep_extreme >= self.source_lower_bound
            )
        ):
            raise ValueError("manipulation frozen geometry is invalid")
        if (
            type(self.close_outside_on_sweep) is not bool
            or type(self.outside_completed_bars) is not int
            or self.outside_completed_bars < 0
            or type(self.age_1m_bars) is not int
            or self.age_1m_bars < 0
            or self.outside_completed_bars > self.age_1m_bars + 1
            or type(self.inside_hold_bars) is not int
            or self.inside_hold_bars < 0
            or type(self.outside_run) is not int
            or self.outside_run < 0
            or self.outside_run > self.outside_completed_bars
            or self.outside_run_side not in {None, "above", "below"}
            or (self.outside_run == 0) != (self.outside_run_side is None)
            or (
                self.outside_run_side is not None
                and self.outside_run_side != self.side
            )
            or (
                self.reentry_candidate_at is None
                and self.inside_hold_bars != 0
            )
            or (
                self.reentry_candidate_at is not None
                and self.outside_run != 0
            )
            or type(self.deadline_elapsed) is not bool
            or not math.isfinite(float(self.penetration_atr))
            or self.penetration_atr <= 0.0
        ):
            raise ValueError("manipulation statistics are invalid")
        raw_strength = float(self.strength)
        if (
            not math.isfinite(raw_strength)
            or not math.isclose(
                raw_strength,
                clamp(self.penetration_atr),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("manipulation strength is inconsistent")
        object.__setattr__(self, "strength", clamp(raw_strength))
        if self.reentry_price is not None and (
            not math.isfinite(float(self.reentry_price))
            or self.reentry_price <= 0.0
        ):
            raise ValueError("manipulation reentry price is invalid")
        if self.reentry_candidate_price is not None and (
            not math.isfinite(float(self.reentry_candidate_price))
            or self.reentry_candidate_price <= 0.0
        ):
            raise ValueError("manipulation reentry candidate is invalid")
        if (self.reentry_candidate_at is None) != (
            self.reentry_candidate_price is None
        ):
            raise ValueError(
                "manipulation reentry candidate clock and price disagree"
            )
        if self.resolved_side not in {None, "above", "below"}:
            raise ValueError("manipulation resolved side is invalid")
        if (
            self.source_formed_at > self.source_eligible_at
            or self.source_eligible_at
            > self.swept_at - pd.Timedelta(minutes=1)
            or self.formed_at != self.swept_at
            or self.confirmed_at != self.swept_at
            or self.state_started_at < self.swept_at
            or self.last_updated_at < self.state_started_at
        ):
            raise ValueError("manipulation knowledge clock is invalid")
        resolution_clocks = tuple(
            value
            for value in (
                self.reaccepted_at,
                self.accepted_outside_at,
                self.resolved_at,
            )
            if value is not None
        )
        if any(
            value <= self.swept_at or value > self.last_updated_at
            for value in resolution_clocks
        ):
            raise ValueError(
                "manipulation resolution clock is outside its lifecycle"
            )
        if self.censored_at is not None and (
            self.censored_at <= self.swept_at
            or self.censored_at > self.last_updated_at
        ):
            raise ValueError("manipulation censorship clock is invalid")
        if any(
            value is not None
            and (value <= self.swept_at or value > self.last_updated_at)
            for value in (
                self.reentry_candidate_at,
                self.reentry_failed_at,
            )
        ):
            raise ValueError("manipulation reentry clock is invalid")
        if self.deadline_elapsed != (
            self.censored_at is not None
            and self.transition_reason == "deadline_elapsed"
        ):
            raise ValueError("manipulation deadline state is inconsistent")
        if (
            (self.deadline_at is None) != (not self.deadline_elapsed)
            or (
                self.deadline_at is not None
                and self.deadline_at != self.censored_at
            )
        ):
            raise ValueError("manipulation deadline clock is inconsistent")
        if self.deadline_elapsed and self.age_1m_bars != 5:
            raise ValueError(
                "deadline censorship requires five real completed bars"
            )
        if self.transition_reason == "":
            raise ValueError("manipulation transition reason cannot be empty")
        if self.lifecycle is ManipulationLifecycle.SWEPT:
            if (
                any(
                    value is not None
                    for value in (
                        self.reaccepted_at,
                        self.accepted_outside_at,
                        self.resolved_at,
                        self.reentry_price,
                        self.resolved_side,
                    )
                )
                or self.state_started_at != self.swept_at
                or (
                    self.censored_at is None
                    and self.transition_reason
                    not in {None, "source_swept"}
                )
                or (
                    self.censored_at is not None
                    and (
                        self.last_updated_at != self.censored_at
                        or self.transition_reason
                        not in {
                            *GROUP4_HARD_BOUNDARY_REASONS,
                            "deadline_elapsed",
                        }
                    )
                )
            ):
                raise ValueError("swept manipulation lifecycle is inconsistent")
        elif self.lifecycle is ManipulationLifecycle.REACCEPTED:
            if (
                self.reaccepted_at is None
                or self.accepted_outside_at is not None
                or self.resolved_at != self.reaccepted_at
                or self.state_started_at != self.resolved_at
                or self.last_updated_at != self.resolved_at
                or self.reentry_price is None
                or self.resolved_side is not None
                or self.censored_at is not None
                or not self.transition_reason
                or self.reentry_candidate_at is None
                or self.reentry_candidate_at >= self.reaccepted_at
                or self.reentry_candidate_price != self.reentry_price
                or self.inside_hold_bars != 1
                or self.outside_run != 0
                or self.outside_run_side is not None
            ):
                raise ValueError(
                    "reaccepted manipulation lifecycle is inconsistent"
                )
        elif (
            self.accepted_outside_at is None
            or self.reaccepted_at is not None
            or self.resolved_at != self.accepted_outside_at
            or self.state_started_at != self.resolved_at
            or self.last_updated_at != self.resolved_at
            or self.reentry_price is not None
            or self.resolved_side is None
            or self.resolved_side != self.side
            or self.outside_run != 2
            or self.reentry_candidate_at is not None
            or self.reentry_candidate_price is not None
            or self.inside_hold_bars != 0
            or self.censored_at is not None
            or not self.transition_reason
        ):
            raise ValueError(
                "accepted-outside manipulation lifecycle is inconsistent"
            )


ORDER_BLOCK_FUNNEL_STAGES = (
    "active_displacement",
    "compatible_bos",
    "break_bar_belongs_to_displacement",
    "reverse_anchor_cluster_found",
    "unique_eligible_bos",
    "ob_created",
)


@dataclass(frozen=True)
class OrderBlockFunnelSnapshot:
    """Lightweight producer-side diagnostics for one real completed 5m bar."""

    observed_at: pd.Timestamp
    stages: tuple[tuple[str, int], ...]
    outcome: OrderBlockAttemptOutcome

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="order_block_funnel.observed_at",
            ),
        )
        stages = tuple(self.stages)
        object.__setattr__(self, "stages", stages)
        if (
            tuple(name for name, _ in stages)
            != ORDER_BLOCK_FUNNEL_STAGES
            or any(
                not isinstance(name, str)
                or type(count) is not int
                or count < 0
                for name, count in stages
            )
            or not isinstance(self.outcome, OrderBlockAttemptOutcome)
        ):
            raise ValueError("order-block funnel snapshot is invalid")
        counts = dict(stages)
        if (
            counts["active_displacement"] not in {0, 1}
            or counts["break_bar_belongs_to_displacement"]
            > counts["compatible_bos"]
            or counts["reverse_anchor_cluster_found"]
            > int(
                counts["break_bar_belongs_to_displacement"] > 0
            )
            or counts["unique_eligible_bos"]
            != int(
                counts["break_bar_belongs_to_displacement"] == 1
                and counts["reverse_anchor_cluster_found"] == 1
            )
            or counts["ob_created"]
            > counts["unique_eligible_bos"]
            or (
                self.outcome is OrderBlockAttemptOutcome.CREATED
            )
            != (counts["ob_created"] == 1)
        ):
            raise ValueError(
                "order-block funnel stages or outcome are inconsistent"
            )


@dataclass(frozen=True)
class FairValueGapState:
    """A strict three-completed-bar gap with frozen formation qualification."""

    fvg_id: str
    protocol_hash: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    lifecycle: FairValueGapLifecycle
    qualification: FVGQualification
    source_displacement_id: str | None
    source_active_transition_id: str | None
    source_displacement_protocol_hash: str | None
    source_displacement_started_at: pd.Timestamp | None
    source_displacement_active_at: pd.Timestamp | None
    source_displacement_prefix_commitment: str | None
    source_candle_ids: tuple[str, str, str]
    source_candle_starts: tuple[
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
    lower_bound: float
    upper_bound: float
    midpoint: float
    invalidation_price: float
    width_points: float
    width_ticks: int
    formation_atr: float
    width_atr: float
    strength: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    age_bars: int
    max_fill_fraction: float
    partial_at: pd.Timestamp | None = None
    mitigated_at: pd.Timestamp | None = None
    invalidated_at: pd.Timestamp | None = None
    transition_reason: str | None = None

    def __post_init__(self) -> None:
        source_ids = tuple(self.source_candle_ids)
        object.__setattr__(self, "source_candle_ids", source_ids)
        if (
            any(
                not isinstance(value, str) or not value
                for value in (self.fvg_id, self.symbol)
            )
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.timeframe is not Timeframe.M5
            or not isinstance(self.direction, Direction)
            or not isinstance(self.lifecycle, FairValueGapLifecycle)
            or not isinstance(self.qualification, FVGQualification)
            or len(source_ids) != 3
            or len(set(source_ids)) != 3
            or any(
                not isinstance(source_id, str) or not source_id
                for source_id in source_ids
            )
        ):
            raise ValueError("FVG identity or frozen source is invalid")
        if not self.protocol_hash:
            raise ValueError("FVG protocol identity is required")
        displacement_fields = (
            self.source_displacement_id,
            self.source_active_transition_id,
            self.source_displacement_protocol_hash,
            self.source_displacement_started_at,
            self.source_displacement_active_at,
            self.source_displacement_prefix_commitment,
        )
        if self.qualification is FVGQualification.RAW:
            if any(value is not None for value in displacement_fields):
                raise ValueError(
                    "raw FVG cannot carry displacement qualification"
                )
        elif any(value is None for value in displacement_fields) or any(
            not isinstance(value, str) or not value
            for value in (
                self.source_displacement_id,
                self.source_active_transition_id,
                self.source_displacement_protocol_hash,
                self.source_displacement_prefix_commitment,
            )
        ):
            raise ValueError(
                "linked FVG requires complete displacement qualification"
            )
        if not (
            math.isfinite(float(self.lower_bound))
            and math.isfinite(float(self.upper_bound))
            and math.isfinite(float(self.midpoint))
            and math.isfinite(float(self.invalidation_price))
            and math.isfinite(float(self.width_points))
            and math.isfinite(float(self.formation_atr))
            and math.isfinite(float(self.width_atr))
            and 0 < self.lower_bound < self.upper_bound
            and self.invalidation_price > 0
            and self.width_points > 0
            and self.formation_atr > 0
            and type(self.width_ticks) is int
            and self.width_ticks > 0
            and self.width_atr > 0
            and math.isclose(
                self.width_points,
                self.width_ticks * 0.25,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                self.midpoint,
                (self.lower_bound + self.upper_bound) / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                self.width_points,
                self.upper_bound - self.lower_bound,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                self.width_atr,
                self.width_points / self.formation_atr,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("FVG frozen geometry is invalid")
        expected_invalidation = (
            self.lower_bound
            if self.direction is Direction.LONG
            else self.upper_bound
        )
        if not math.isclose(
            self.invalidation_price,
            expected_invalidation,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("FVG invalidation must equal its frozen far edge")
        starts = tuple(
            aware_timestamp(value, name="fvg.source_candle_starts")
            for value in self.source_candle_starts
        )
        object.__setattr__(self, "source_candle_starts", starts)
        if (
            len(starts) != 3
            or len(set(starts)) != 3
            or starts != tuple(sorted(starts))
            or starts[1] - starts[0] != pd.Timedelta(minutes=5)
            or starts[2] - starts[1] != pd.Timedelta(minutes=5)
        ):
            raise ValueError("FVG source candle clocks are invalid")
        for name in (
            "formed_at",
            "confirmed_at",
            "state_started_at",
            "last_updated_at",
        ):
            object.__setattr__(
                self,
                name,
                aware_timestamp(getattr(self, name), name=f"fvg.{name}"),
            )
        for name in (
            "source_displacement_started_at",
            "source_displacement_active_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"fvg.{name}"),
                )
        for name in (
            "partial_at",
            "mitigated_at",
            "invalidated_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"fvg.{name}"),
                )
        if (
            starts[-1] > self.formed_at
            or self.formed_at
            != starts[-1] + pd.Timedelta(minutes=5)
            or self.confirmed_at != self.formed_at
            or self.state_started_at < self.confirmed_at
            or self.last_updated_at < self.state_started_at
            or type(self.age_bars) is not int
            or self.age_bars < 0
        ):
            raise ValueError("FVG formation, update clock or age is invalid")
        if (
            self.qualification is FVGQualification.DISPLACEMENT_LINKED
            and (
                self.source_displacement_started_at
                > self.source_displacement_active_at
                or self.source_displacement_active_at > self.formed_at
            )
        ):
            raise ValueError(
                "FVG displacement qualification clocks are invalid"
            )
        transition_clocks = tuple(
            value
            for value in (
                self.partial_at,
                self.mitigated_at,
                self.invalidated_at,
            )
            if value is not None
        )
        if any(
            value <= self.confirmed_at or value > self.last_updated_at
            for value in transition_clocks
        ):
            raise ValueError("FVG transition clock is outside its lifecycle")
        if (
            self.partial_at is not None
            and self.mitigated_at is not None
            and self.partial_at > self.mitigated_at
        ):
            raise ValueError("FVG mitigation cannot predate its partial fill")
        if (
            self.partial_at is not None
            and self.invalidated_at is not None
            and self.partial_at > self.invalidated_at
        ):
            raise ValueError("FVG invalidation cannot predate its partial fill")
        if (
            self.mitigated_at is not None
            and self.invalidated_at is not None
            and self.mitigated_at > self.invalidated_at
        ):
            raise ValueError("FVG invalidation cannot predate mitigation")
        raw_strength = float(self.strength)
        if not math.isfinite(raw_strength) or raw_strength < 0.0:
            raise ValueError("FVG strength must be finite and non-negative")
        object.__setattr__(self, "strength", clamp(raw_strength))
        raw_fill = float(self.max_fill_fraction)
        if not math.isfinite(raw_fill) or not 0.0 <= raw_fill <= 1.0:
            raise ValueError("FVG maximum fill fraction must be in [0, 1]")
        object.__setattr__(self, "max_fill_fraction", raw_fill)
        if self.transition_reason == "":
            raise ValueError("FVG transition reason cannot be empty")
        if self.lifecycle is FairValueGapLifecycle.OPEN:
            if (
                self.state_started_at != self.confirmed_at
                or raw_fill != 0.0
                or any(
                    value is not None
                    for value in (
                        self.partial_at,
                        self.mitigated_at,
                        self.invalidated_at,
                        self.transition_reason,
                    )
                )
            ):
                raise ValueError("open FVG lifecycle is inconsistent")
        elif self.lifecycle is FairValueGapLifecycle.PARTIAL:
            if (
                self.partial_at is None
                or self.state_started_at != self.partial_at
                or not 0.0 < raw_fill < 1.0
                or self.mitigated_at is not None
                or self.invalidated_at is not None
                or not self.transition_reason
            ):
                raise ValueError("partial FVG lifecycle is inconsistent")
        elif self.lifecycle is FairValueGapLifecycle.MITIGATED:
            if (
                self.mitigated_at is None
                or self.state_started_at != self.mitigated_at
                or self.last_updated_at != self.mitigated_at
                or not math.isclose(raw_fill, 1.0, abs_tol=1e-12)
                or self.invalidated_at is not None
                or not self.transition_reason
            ):
                raise ValueError("mitigated FVG lifecycle is inconsistent")
        elif (
            self.invalidated_at is None
            or self.state_started_at != self.invalidated_at
            or self.last_updated_at != self.invalidated_at
            or self.mitigated_at is not None
            or not self.transition_reason
        ):
            raise ValueError(
                "invalidated FVG requires its clock and reason"
            )


@dataclass(frozen=True)
class OrderBlockState:
    """A frozen pre-BOS candle linked to qualified displacement and BOS."""

    order_block_id: str
    protocol_hash: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    lifecycle: OrderBlockLifecycle
    source_displacement_id: str
    source_active_transition_id: str
    source_displacement_protocol_hash: str
    source_displacement_seed_candle_id: str
    source_displacement_started_at: pd.Timestamp
    source_displacement_active_at: pd.Timestamp
    source_displacement_prefix_commitment: str
    source_bos_id: str
    source_bos_protocol_hash: str
    source_bos_target_swing_id: str
    source_bos_structure_id: str | None
    source_bos_scope: BOSScope
    source_bos_pending_at: pd.Timestamp
    source_bos_resolved_at: pd.Timestamp
    source_bos_break_bar_id: str
    source_bos_mss_qualified: bool
    anchor_candle_id: str
    anchor_candle_ids: tuple[str, ...]
    anchor_start: pd.Timestamp
    anchor_end: pd.Timestamp
    anchor_open: float
    anchor_close: float
    lower_bound: float
    upper_bound: float
    body_lower_bound: float
    body_upper_bound: float
    midpoint: float
    invalidation_price: float
    width_points: float
    width_ticks: int
    width_atr: float
    strength: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    age_bars: int
    first_test_at: pd.Timestamp | None = None
    mitigated_at: pd.Timestamp | None = None
    failed_at: pd.Timestamp | None = None
    transition_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.order_block_id,
                    self.symbol,
                    self.source_displacement_id,
                    self.source_active_transition_id,
                    self.source_displacement_seed_candle_id,
                    self.source_displacement_prefix_commitment,
                    self.source_bos_id,
                    self.source_bos_protocol_hash,
                    self.source_bos_target_swing_id,
                    self.source_bos_break_bar_id,
                    self.anchor_candle_id,
                )
            )
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.timeframe is not Timeframe.M5
            or not isinstance(self.direction, Direction)
            or not isinstance(self.lifecycle, OrderBlockLifecycle)
            or (
                self.source_bos_structure_id is not None
                and (
                    not isinstance(self.source_bos_structure_id, str)
                    or not self.source_bos_structure_id
                )
            )
            or not isinstance(self.source_bos_scope, BOSScope)
        ):
            raise ValueError(
                "order-block identity or frozen source is invalid"
            )
        anchor_ids = tuple(self.anchor_candle_ids)
        object.__setattr__(self, "anchor_candle_ids", anchor_ids)
        if (
            not anchor_ids
            or len(anchor_ids) != len(set(anchor_ids))
            or any(not isinstance(value, str) or not value for value in anchor_ids)
            or anchor_ids[-1] != self.anchor_candle_id
        ):
            raise ValueError("order-block anchor cluster identity is invalid")
        if not self.protocol_hash or not self.source_displacement_protocol_hash:
            raise ValueError("order-block protocol identity is required")
        if (
            self.source_bos_scope is BOSScope.LOCAL
            or self.source_bos_structure_id is None
            or (
                self.source_bos_scope is BOSScope.OPPOSED
                and not self.source_bos_mss_qualified
            )
            or (
                self.source_bos_scope is BOSScope.CONTINUATION
                and self.source_bos_mss_qualified
            )
            or type(self.source_bos_mss_qualified) is not bool
        ):
            raise ValueError(
                "order-block requires continuation BOS or opposed MSS"
            )
        if not (
            math.isfinite(float(self.lower_bound))
            and math.isfinite(float(self.upper_bound))
            and math.isfinite(float(self.midpoint))
            and math.isfinite(float(self.invalidation_price))
            and math.isfinite(float(self.anchor_open))
            and math.isfinite(float(self.anchor_close))
            and math.isfinite(float(self.body_lower_bound))
            and math.isfinite(float(self.body_upper_bound))
            and math.isfinite(float(self.width_points))
            and math.isfinite(float(self.width_atr))
            and 0 < self.lower_bound < self.upper_bound
            and self.lower_bound
            <= self.anchor_open
            <= self.upper_bound
            and self.lower_bound
            <= self.anchor_close
            <= self.upper_bound
            and self.lower_bound
            <= self.body_lower_bound
            < self.body_upper_bound
            <= self.upper_bound
            and self.body_lower_bound
            <= min(self.anchor_open, self.anchor_close)
            <= max(self.anchor_open, self.anchor_close)
            <= self.body_upper_bound
            and self.invalidation_price > 0
            and self.width_points > 0
            and type(self.width_ticks) is int
            and self.width_ticks > 0
            and self.width_atr > 0
            and math.isclose(
                self.width_points,
                self.width_ticks * 0.25,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                self.midpoint,
                (self.lower_bound + self.upper_bound) / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                self.width_points,
                self.upper_bound - self.lower_bound,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("order-block frozen geometry is invalid")
        expected_invalidation = (
            self.lower_bound
            if self.direction is Direction.LONG
            else self.upper_bound
        )
        if not math.isclose(
            self.invalidation_price,
            expected_invalidation,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "order-block invalidation must equal its frozen distal edge"
            )
        if (
            self.direction is Direction.LONG
            and self.anchor_close >= self.anchor_open
        ) or (
            self.direction is Direction.SHORT
            and self.anchor_close <= self.anchor_open
        ):
            raise ValueError(
                "order-block anchor body must oppose displacement"
            )
        for name in (
            "source_displacement_started_at",
            "source_displacement_active_at",
            "source_bos_pending_at",
            "source_bos_resolved_at",
            "anchor_start",
            "anchor_end",
            "formed_at",
            "confirmed_at",
            "state_started_at",
            "last_updated_at",
        ):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"order_block.{name}",
                ),
            )
        for name in (
            "first_test_at",
            "mitigated_at",
            "failed_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"order_block.{name}"),
                )
        if (
            self.anchor_end <= self.anchor_start
            or self.anchor_end > self.formed_at
            or self.anchor_end
            != self.source_displacement_started_at - pd.Timedelta(minutes=5)
            or self.source_displacement_started_at
            > self.source_displacement_active_at
            or self.source_bos_pending_at > self.source_displacement_started_at
            or self.source_displacement_active_at > self.formed_at
            or self.source_bos_resolved_at != self.formed_at
            or self.confirmed_at != self.formed_at
            or self.state_started_at < self.confirmed_at
            or self.last_updated_at < self.state_started_at
            or type(self.age_bars) is not int
            or self.age_bars < 0
        ):
            raise ValueError(
                "order-block formation, update clock or age is invalid"
            )
        transition_clocks = tuple(
            value
            for value in (
                self.first_test_at,
                self.mitigated_at,
                self.failed_at,
            )
            if value is not None
        )
        if any(
            value <= self.confirmed_at or value > self.last_updated_at
            for value in transition_clocks
        ):
            raise ValueError(
                "order-block transition clock is outside its lifecycle"
            )
        if (
            self.first_test_at is not None
            and self.mitigated_at is not None
            and self.first_test_at > self.mitigated_at
        ):
            raise ValueError(
                "order-block mitigation cannot predate its first test"
            )
        if (
            self.first_test_at is not None
            and self.failed_at is not None
            and self.first_test_at > self.failed_at
        ):
            raise ValueError(
                "order-block failure cannot predate its first test"
            )
        if (
            self.mitigated_at is not None
            and self.failed_at is not None
            and self.mitigated_at > self.failed_at
        ):
            raise ValueError(
                "order-block failure cannot predate mitigation"
            )
        raw_strength = float(self.strength)
        if not math.isfinite(raw_strength) or raw_strength < 0.0:
            raise ValueError(
                "order-block strength must be finite and non-negative"
            )
        object.__setattr__(self, "strength", clamp(raw_strength))
        if self.transition_reason == "":
            raise ValueError("order-block transition reason cannot be empty")
        if self.lifecycle is OrderBlockLifecycle.CREATED:
            if (
                self.state_started_at != self.confirmed_at
                or any(
                    value is not None
                    for value in (
                        self.first_test_at,
                        self.mitigated_at,
                        self.failed_at,
                        self.transition_reason,
                    )
                )
            ):
                raise ValueError(
                    "created order-block lifecycle is inconsistent"
                )
        elif self.lifecycle is OrderBlockLifecycle.UNTESTED:
            if (
                self.state_started_at <= self.confirmed_at
                or self.first_test_at is not None
                or self.mitigated_at is not None
                or self.failed_at is not None
                or not self.transition_reason
            ):
                raise ValueError(
                    "untested order-block lifecycle is inconsistent"
                )
        elif self.lifecycle is OrderBlockLifecycle.MITIGATED:
            if (
                self.first_test_at is None
                or self.mitigated_at is None
                or self.state_started_at != self.mitigated_at
                or self.last_updated_at != self.mitigated_at
                or self.failed_at is not None
                or not self.transition_reason
            ):
                raise ValueError(
                    "mitigated order-block lifecycle is inconsistent"
                )
        elif (
            self.failed_at is None
            or self.state_started_at != self.failed_at
            or self.last_updated_at != self.failed_at
            or self.mitigated_at is not None
            or not self.transition_reason
        ):
            raise ValueError(
                "failed order-block requires its clock and reason"
            )


@dataclass(frozen=True)
class EntryLocationState:
    """The first completed-1m visit to one exact frozen 5m entry zone."""

    location_id: str
    protocol_hash: str
    source_group3_protocol_hash: str
    symbol: str
    instrument_id: int
    direction: Direction
    source_zone_kind: str
    source_zone_id: str
    source_zone_protocol_hash: str
    source_displacement_id: str
    source_bos_id: str | None
    lower_bound: float
    upper_bound: float
    midpoint: float
    near_edge: float
    far_edge: float
    failure_boundary: float
    formed_at: pd.Timestamp
    lifecycle: EntryLocationLifecycle
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    age_real_1m_bars: int
    state_duration_real_1m_bars: int
    current_price: float
    distance_to_zone_points: float
    distance_to_failure_points: float
    nearest_visible_draw_id: str | None = None
    nearest_visible_draw_distance_points: float | None = None
    departure_confirmed_at: pd.Timestamp | None = None
    first_entered_at: pd.Timestamp | None = None
    entry_mode: str | None = None
    contact_reference_price: float | None = None
    first_penetration_fraction: float = 0.0
    rejected_at: pd.Timestamp | None = None
    left_at: pd.Timestamp | None = None
    reaction_atr: float = 0.0
    transition_reason: str = "source_registered"

    def __post_init__(self) -> None:
        if (
            not self.location_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.source_zone_kind not in {"fvg", "order_block"}
            or not self.source_zone_id
            or not self.source_displacement_id
            or (
                self.source_zone_kind == "fvg"
                and self.source_bos_id is not None
            )
            or (
                self.source_zone_kind == "order_block"
                and not self.source_bos_id
            )
            or not isinstance(self.direction, Direction)
            or not isinstance(self.lifecycle, EntryLocationLifecycle)
        ):
            raise ValueError("entry-location identity or source is invalid")
        if not all(
            (
                self.protocol_hash,
                self.source_group3_protocol_hash,
                self.source_zone_protocol_hash,
            )
        ):
            raise ValueError("entry-location protocol identity is required")
        prices = (
            self.lower_bound,
            self.upper_bound,
            self.midpoint,
            self.near_edge,
            self.far_edge,
            self.failure_boundary,
            self.current_price,
            self.distance_to_zone_points,
            self.distance_to_failure_points,
            self.first_penetration_fraction,
            self.reaction_atr,
        )
        if (
            not all(math.isfinite(float(value)) for value in prices)
            or not 0 < self.lower_bound < self.upper_bound
            or not math.isclose(
                self.midpoint,
                (self.lower_bound + self.upper_bound) / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or self.distance_to_zone_points < 0
            or not 0.0 <= self.first_penetration_fraction <= 1.0
            or self.reaction_atr < 0
            or type(self.age_real_1m_bars) is not int
            or self.age_real_1m_bars < 0
            or type(self.state_duration_real_1m_bars) is not int
            or self.state_duration_real_1m_bars < 0
        ):
            raise ValueError("entry-location geometry or metrics are invalid")
        expected_near = (
            self.upper_bound
            if self.direction is Direction.LONG
            else self.lower_bound
        )
        expected_far = (
            self.lower_bound
            if self.direction is Direction.LONG
            else self.upper_bound
        )
        if (
            not math.isclose(
                self.near_edge,
                expected_near,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.far_edge,
                expected_far,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.failure_boundary,
                expected_far,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("entry-location directional edges are invalid")
        for name in (
            "formed_at",
            "state_started_at",
            "last_updated_at",
            "departure_confirmed_at",
            "first_entered_at",
            "rejected_at",
            "left_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"entry_location.{name}"),
                )
        if (
            self.state_started_at < self.formed_at
            or self.last_updated_at < self.state_started_at
            or any(
                value is not None
                and (
                    value < self.formed_at
                    or value > self.last_updated_at
                )
                for value in (
                    self.departure_confirmed_at,
                    self.first_entered_at,
                    self.rejected_at,
                    self.left_at,
                )
            )
            or (
                self.first_entered_at is not None
                and (
                    self.departure_confirmed_at is None
                    or self.first_entered_at
                    <= self.departure_confirmed_at
                )
            )
        ):
            raise ValueError("entry-location knowledge clocks are invalid")
        if (
            (self.nearest_visible_draw_id is None)
            != (self.nearest_visible_draw_distance_points is None)
            or self.nearest_visible_draw_id == ""
            or (
                self.nearest_visible_draw_distance_points is not None
                and (
                    not math.isfinite(
                        float(self.nearest_visible_draw_distance_points)
                    )
                    or self.nearest_visible_draw_distance_points <= 0
                )
            )
        ):
            raise ValueError("entry-location nearest draw view is invalid")
        if self.first_entered_at is None:
            if (
                self.entry_mode is not None
                or self.contact_reference_price is not None
                or self.first_penetration_fraction != 0.0
                or self.rejected_at is not None
            ):
                raise ValueError("entry-location first visit is inconsistent")
        elif (
            self.entry_mode
            not in {"crossed_near_edge", "gap_opened_inside"}
            or self.contact_reference_price is None
            or not math.isfinite(float(self.contact_reference_price))
            or not (
                self.lower_bound
                <= self.contact_reference_price
                <= self.upper_bound
            )
        ):
            raise ValueError("entry-location first-pullback record is invalid")
        if self.lifecycle is EntryLocationLifecycle.APPROACHING:
            if any(
                value is not None
                for value in (
                    self.first_entered_at,
                    self.rejected_at,
                    self.left_at,
                )
            ):
                raise ValueError("approaching location already has a visit")
        elif self.lifecycle is EntryLocationLifecycle.IN_ZONE:
            if (
                self.first_entered_at is None
                or self.rejected_at is not None
                or self.left_at is not None
            ):
                raise ValueError("in-zone location lifecycle is inconsistent")
        elif self.lifecycle is EntryLocationLifecycle.REJECTED:
            if (
                self.first_entered_at is None
                or self.rejected_at is None
                or self.left_at is not None
                or self.state_started_at != self.rejected_at
            ):
                raise ValueError("rejected location lifecycle is inconsistent")
        elif (
            self.left_at is None
            or self.rejected_at is not None
            or self.state_started_at != self.left_at
        ):
            raise ValueError("left location lifecycle is inconsistent")
        if not self.transition_reason:
            raise ValueError("entry-location transition reason is required")


@dataclass(frozen=True)
class QualifiedReacceptanceState:
    """A frozen reference leave, later reclaim and later completed hold."""

    reacceptance_id: str
    protocol_hash: str
    symbol: str
    instrument_id: int
    context_kind: str
    context_id: str
    source_entity_id: str
    direction: Direction
    reference_price: float
    failure_boundary: float
    lifecycle: QualifiedReacceptanceLifecycle
    formed_at: pd.Timestamp
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    left_at: pd.Timestamp
    age_real_1m_bars: int
    state_duration_real_1m_bars: int
    required_later_hold_bars: int
    hold_real_1m_bars: int = 0
    reclaimed_at: pd.Timestamp | None = None
    held_at: pd.Timestamp | None = None
    failed_at: pd.Timestamp | None = None
    censored_at: pd.Timestamp | None = None
    reclaim_margin_atr: float = 0.0
    hold_margin_atr: float = 0.0
    strength: float = 0.0
    transition_reason: str = "reference_left"

    def __post_init__(self) -> None:
        if (
            not self.reacceptance_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.context_kind != "entry_zone"
            or not self.context_id
            or not self.source_entity_id
            or not isinstance(self.direction, Direction)
            or not isinstance(
                self.lifecycle,
                QualifiedReacceptanceLifecycle,
            )
            or not self.protocol_hash
            or not math.isfinite(float(self.reference_price))
            or not math.isfinite(float(self.failure_boundary))
            or self.reference_price <= 0
            or self.failure_boundary <= 0
            or type(self.age_real_1m_bars) is not int
            or self.age_real_1m_bars < 0
            or type(self.state_duration_real_1m_bars) is not int
            or self.state_duration_real_1m_bars < 0
            or type(self.required_later_hold_bars) is not int
            or self.required_later_hold_bars < 1
            or type(self.hold_real_1m_bars) is not int
            or not 0 <= self.hold_real_1m_bars
            <= self.required_later_hold_bars
        ):
            raise ValueError("qualified reacceptance identity is invalid")
        for name in (
            "formed_at",
            "state_started_at",
            "last_updated_at",
            "left_at",
            "reclaimed_at",
            "held_at",
            "failed_at",
            "censored_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"reacceptance.{name}"),
                )
        same_clock_failure = (
            self.lifecycle is QualifiedReacceptanceLifecycle.FAILED
            and self.failed_at == self.left_at
            and self.transition_reason
            in GROUP5_SAME_CLOCK_REACCEPTANCE_FAILURE_REASONS
        )
        if (
            self.formed_at != self.left_at
            or self.state_started_at < self.formed_at
            or self.last_updated_at < self.state_started_at
            or any(
                value is not None
                and (
                    (
                        value <= self.left_at
                        and not (
                            name == "failed_at"
                            and same_clock_failure
                        )
                    )
                    or value > self.last_updated_at
                )
                for name, value in (
                    ("reclaimed_at", self.reclaimed_at),
                    ("held_at", self.held_at),
                    ("failed_at", self.failed_at),
                    ("censored_at", self.censored_at),
                )
            )
            or (
                self.held_at is not None
                and (
                    self.reclaimed_at is None
                    or self.held_at <= self.reclaimed_at
                )
            )
        ):
            raise ValueError("qualified reacceptance clocks are invalid")
        for value in (
            self.reclaim_margin_atr,
            self.hold_margin_atr,
            self.strength,
        ):
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError("qualified reacceptance strength is invalid")
        if self.lifecycle is QualifiedReacceptanceLifecycle.LEFT:
            if any(
                value is not None
                for value in (
                    self.reclaimed_at,
                    self.held_at,
                    self.failed_at,
                    self.censored_at,
                )
            ):
                raise ValueError("left reacceptance lifecycle is inconsistent")
        elif self.lifecycle is QualifiedReacceptanceLifecycle.RECLAIMED:
            if (
                self.reclaimed_at is None
                or self.held_at is not None
                or self.failed_at is not None
                or self.censored_at is not None
                or self.state_started_at != self.reclaimed_at
            ):
                raise ValueError(
                    "reclaimed reacceptance lifecycle is inconsistent"
                )
        elif self.lifecycle is QualifiedReacceptanceLifecycle.HELD:
            if (
                self.reclaimed_at is None
                or self.held_at is None
                or self.failed_at is not None
                or self.censored_at is not None
                or self.state_started_at != self.held_at
                or self.hold_real_1m_bars
                != self.required_later_hold_bars
            ):
                raise ValueError("held reacceptance lifecycle is inconsistent")
        elif self.lifecycle is QualifiedReacceptanceLifecycle.FAILED:
            if (
                self.failed_at is None
                or self.held_at is not None
                or self.censored_at is not None
                or self.state_started_at != self.failed_at
            ):
                raise ValueError(
                    "failed reacceptance lifecycle is inconsistent"
                )
        elif (
            self.censored_at is None
            or self.held_at is not None
            or self.failed_at is not None
            or self.state_started_at != self.censored_at
            or self.transition_reason != "hard_boundary_censored"
        ):
            raise ValueError(
                "censored reacceptance lifecycle is inconsistent"
            )
        if not self.transition_reason:
            raise ValueError(
                "qualified reacceptance transition reason is required"
            )


@dataclass(frozen=True)
class MicroBOSReference:
    """A non-recomputed reference to an exact upstream confirmed M1 BOS."""

    reference_id: str
    protocol_hash: str
    context_kind: str
    context_id: str
    expected_direction: Direction
    anchor_at: pd.Timestamp
    bos_id: str
    bos_direction: Direction
    target_swing_id: str
    scope: BOSScope
    pending_at: pd.Timestamp
    resolved_at: pd.Timestamp
    relation: str
    outcome: str
    qualified: bool
    strength: float

    def __post_init__(self) -> None:
        if (
            not self.reference_id
            or not self.protocol_hash
            or self.context_kind not in GROUP5_CONTEXT_KINDS
            or not self.context_id
            or not isinstance(self.expected_direction, Direction)
            or not self.bos_id
            or not isinstance(self.bos_direction, Direction)
            or not self.target_swing_id
            or not isinstance(self.scope, BOSScope)
            or self.relation not in {"strictly_after", "same_clock_unknown"}
            or self.outcome
            not in {
                "aligned",
                "opposed",
                "simultaneous_unknown",
                "ambiguous_same_clock",
            }
            or type(self.qualified) is not bool
            or not math.isfinite(float(self.strength))
            or not 0.0 <= self.strength <= 1.0
        ):
            raise ValueError("micro-BOS reference is invalid")
        for name in ("anchor_at", "pending_at", "resolved_at"):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"micro_bos.{name}",
                ),
            )
        if (
            self.pending_at >= self.resolved_at
            or self.resolved_at < self.anchor_at
            or (
                self.relation == "strictly_after"
                and self.resolved_at <= self.anchor_at
            )
            or (
                self.relation == "same_clock_unknown"
                and self.resolved_at != self.anchor_at
            )
            or self.qualified
            != (
                self.relation == "strictly_after"
                and self.outcome == "aligned"
            )
            or (
                self.outcome == "aligned"
                and self.bos_direction is not self.expected_direction
            )
            or (
                self.outcome == "opposed"
                and self.bos_direction is self.expected_direction
            )
            or (
                self.outcome == "simultaneous_unknown"
                and self.relation != "same_clock_unknown"
            )
            or (
                self.outcome != "simultaneous_unknown"
                and self.relation == "same_clock_unknown"
            )
            or (
                self.outcome == "ambiguous_same_clock"
                and self.relation != "strictly_after"
            )
        ):
            raise ValueError("micro-BOS reference clocks or outcome are invalid")


@dataclass(frozen=True)
class PathSequenceStep:
    """One immutable, source-bound milestone in a Group 5 context."""

    step_id: str
    kind: str
    observed_at: pd.Timestamp
    source_event_id: str | None
    source_entity_id: str
    predecessor_step_ids: tuple[str, ...]
    same_clock_relation: str
    direction: Direction
    strength: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="path_step.observed_at",
            ),
        )
        object.__setattr__(
            self,
            "predecessor_step_ids",
            tuple(self.predecessor_step_ids),
        )
        if (
            not self.step_id
            or self.kind not in GROUP5_PATH_STEP_KINDS
            or self.source_event_id == ""
            or not self.source_entity_id
            or len(self.predecessor_step_ids)
            != len(set(self.predecessor_step_ids))
            or any(not value for value in self.predecessor_step_ids)
            or self.same_clock_relation
            not in GROUP5_SAME_CLOCK_RELATIONS
            or not isinstance(self.direction, Direction)
            or not math.isfinite(float(self.strength))
            or not 0.0 <= self.strength <= 1.0
            or not self.reason
        ):
            raise ValueError("path-sequence step is invalid")


@dataclass(frozen=True)
class PathSequenceState:
    """A bounded ordered path; it is not a scalar score or action label."""

    sequence_id: str
    protocol_hash: str
    symbol: str
    instrument_id: int
    context_kind: str
    context_id: str
    direction: Direction
    lifecycle: PathSequenceLifecycle
    formed_at: pd.Timestamp
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    age_real_1m_bars: int
    state_duration_real_1m_bars: int
    steps: tuple[PathSequenceStep, ...]
    ended_at: pd.Timestamp | None = None
    transition_reason: str = "context_registered"

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if (
            not self.sequence_id
            or not self.protocol_hash
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.context_kind not in GROUP5_CONTEXT_KINDS
            or not self.context_id
            or not isinstance(self.direction, Direction)
            or not isinstance(self.lifecycle, PathSequenceLifecycle)
            or type(self.age_real_1m_bars) is not int
            or self.age_real_1m_bars < 0
            or type(self.state_duration_real_1m_bars) is not int
            or self.state_duration_real_1m_bars < 0
            or not self.steps
            or not self.transition_reason
        ):
            raise ValueError("path-sequence state identity is invalid")
        for name in (
            "formed_at",
            "state_started_at",
            "last_updated_at",
            "ended_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"path_sequence.{name}"),
                )
        step_ids = tuple(step.step_id for step in self.steps)
        if (
            self.state_started_at < self.formed_at
            or self.last_updated_at < self.state_started_at
            or len(step_ids) != len(set(step_ids))
            or self.steps[0].observed_at != self.formed_at
            or any(
                left.observed_at > right.observed_at
                for left, right in zip(self.steps, self.steps[1:])
            )
            or any(
                step.observed_at > self.last_updated_at
                or (
                    index == 0
                    and (
                        step.predecessor_step_ids
                        or step.same_clock_relation != "origin"
                    )
                )
                or (
                    index > 0
                    and step.predecessor_step_ids
                    != (step_ids[index - 1],)
                )
                or (
                    index > 0
                    and step.observed_at
                    > self.steps[index - 1].observed_at
                    and step.same_clock_relation != "strictly_after"
                )
                or (
                    index > 0
                    and step.observed_at
                    == self.steps[index - 1].observed_at
                    and step.same_clock_relation
                    not in {"same_clock_known", "same_clock_unknown"}
                )
                for index, step in enumerate(self.steps)
            )
        ):
            raise ValueError("path-sequence history is invalid")
        if self.lifecycle is PathSequenceLifecycle.ACTIVE:
            if self.ended_at is not None:
                raise ValueError("active path sequence cannot be ended")
        elif (
            self.ended_at is None
            or self.ended_at != self.state_started_at
            or self.last_updated_at != self.ended_at
        ):
            raise ValueError("terminal path sequence clocks are invalid")
        if (
            self.lifecycle is PathSequenceLifecycle.CENSORED
            and self.transition_reason not in GROUP5_HARD_BOUNDARY_REASONS
        ):
            raise ValueError("censored path sequence reason is invalid")


@dataclass(frozen=True)
class LiquidityInventoryItem:
    """One causal draw candidate; targeting is a downstream decision overlay."""

    item_id: str
    timeframe: Timeframe
    side: str
    kind: str
    price: float
    lower_bound: float
    upper_bound: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    lifecycle: LiquidityInventoryLifecycle
    source_ids: tuple[str, ...]
    age_bars: int
    strength: float
    targeted_at: pd.Timestamp | None = None
    consumed_at: pd.Timestamp | None = None
    lifecycle_reason: str | None = None
    structural_rank: str = "internal"
    is_protected_swing: bool = False
    visibility_strength: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.item_id
            or self.side not in {"above", "below"}
            or self.kind
            not in {
                "swing",
                "equal_highs",
                "equal_lows",
                "previous_session_high",
                "previous_session_low",
                "previous_day_high",
                "previous_day_low",
                "previous_week_high",
                "previous_week_low",
                "range_boundary",
            }
            or not self.source_ids
            or self.structural_rank not in {"internal", "external"}
            or type(self.is_protected_swing) is not bool
        ):
            raise ValueError("liquidity inventory identity is invalid")
        if not (
            math.isfinite(float(self.price))
            and math.isfinite(float(self.lower_bound))
            and math.isfinite(float(self.upper_bound))
            and 0 < self.lower_bound <= self.price <= self.upper_bound
        ):
            raise ValueError("liquidity inventory price bounds are invalid")
        for name in (
            "formed_at",
            "confirmed_at",
            "targeted_at",
            "consumed_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"liquidity_inventory.{name}"),
                )
        if (
            self.confirmed_at < self.formed_at
            or type(self.age_bars) is not int
            or self.age_bars < 0
            or len(self.source_ids) != len(set(self.source_ids))
        ):
            raise ValueError("liquidity inventory history is invalid")
        object.__setattr__(self, "strength", clamp(self.strength))
        visibility = float(self.visibility_strength)
        if not math.isfinite(visibility):
            raise ValueError("liquidity visibility must be finite")
        object.__setattr__(
            self,
            "visibility_strength",
            clamp(visibility),
        )
        if self.is_protected_swing and self.structural_rank != "external":
            raise ValueError(
                "protected swing liquidity must be externally ranked"
            )
        if self.lifecycle is LiquidityInventoryLifecycle.VISIBLE:
            if any(
                value is not None
                for value in (
                    self.targeted_at,
                    self.consumed_at,
                    self.lifecycle_reason,
                )
            ):
                raise ValueError("visible liquidity inventory state is inconsistent")
        elif self.lifecycle is LiquidityInventoryLifecycle.TARGETED:
            if (
                self.targeted_at is None
                or self.targeted_at < self.confirmed_at
                or self.consumed_at is not None
                or self.lifecycle_reason != "selected_draw"
            ):
                raise ValueError("targeted liquidity inventory state is inconsistent")
        elif (
            self.consumed_at is None
            or self.consumed_at <= self.confirmed_at
            or self.lifecycle_reason not in {
                "close_beyond_swing",
                "swing_swept",
                "pool_swept",
                "reference_level_swept",
                "range_boundary_consumed",
            }
        ):
            raise ValueError("consumed liquidity inventory state is inconsistent")


@dataclass(frozen=True)
class DrawSelection:
    """A downstream, episode-frozen TARGETED overlay on visible liquidity."""

    draw_id: str
    selected_at: pd.Timestamp
    selection_reason: str
    source_timeframe: Timeframe
    source_kind: str
    side: str
    price: float
    source_confirmed_at: pd.Timestamp
    strength: float
    lifecycle: LiquidityInventoryLifecycle = (
        LiquidityInventoryLifecycle.TARGETED
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_at",
            aware_timestamp(self.selected_at, name="draw_selection.selected_at"),
        )
        object.__setattr__(
            self,
            "source_confirmed_at",
            aware_timestamp(
                self.source_confirmed_at,
                name="draw_selection.source_confirmed_at",
            ),
        )
        if (
            not self.draw_id
            or not self.selection_reason
            or not isinstance(self.source_timeframe, Timeframe)
            or self.source_kind
            not in {
                "swing",
                "equal_highs",
                "equal_lows",
                "previous_session_high",
                "previous_session_low",
                "previous_day_high",
                "previous_day_low",
                "previous_week_high",
                "previous_week_low",
                "range_boundary",
            }
            or self.side not in {"above", "below"}
            or self.lifecycle is not LiquidityInventoryLifecycle.TARGETED
            or not math.isfinite(float(self.price))
            or self.price <= 0.0
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
            or self.selected_at < self.source_confirmed_at
        ):
            raise ValueError("draw-selection identity, source or clock is invalid")


@dataclass(frozen=True)
class LiquidityRoute:
    """Frozen separation of directional draw and executable delivery target."""

    route_id: str
    selected_at: pd.Timestamp
    context_draw_id: str | None
    intermediate_liquidity_ids: tuple[str, ...]
    primary_deliverable_target_id: str | None
    terminal_draw_id: str | None
    path_blocker_ids: tuple[str, ...] = ()
    source_path_ids: tuple[str, ...] = ()
    range_context_id: str | None = None
    range_midpoint: float | None = None
    swept_range_boundary_id: str | None = None
    opposing_range_boundary_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_at",
            aware_timestamp(self.selected_at, name="liquidity_route.selected_at"),
        )
        for name in (
            "intermediate_liquidity_ids",
            "path_blocker_ids",
            "source_path_ids",
        ):
            values = tuple(dict.fromkeys(getattr(self, name)))
            object.__setattr__(self, name, values)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError("liquidity-route identities are invalid")
        if not self.route_id or any(
            value is not None and (not isinstance(value, str) or not value)
            for value in (
                self.context_draw_id,
                self.primary_deliverable_target_id,
                self.terminal_draw_id,
            )
        ):
            raise ValueError("liquidity-route identity is invalid")
        range_identity = (
            self.range_context_id,
            self.swept_range_boundary_id,
        )
        if any(value is not None for value in range_identity) != all(
            value is not None for value in range_identity
        ):
            raise ValueError(
                "liquidity-route range context is only partially identified"
            )
        if self.range_context_id is None:
            if (
                self.range_midpoint is not None
                or self.opposing_range_boundary_id is not None
            ):
                raise ValueError(
                    "liquidity-route range metadata lacks a range context"
                )
        elif (
            any(
                not isinstance(value, str) or not value
                for value in range_identity
            )
            or self.range_midpoint is None
            or not math.isfinite(float(self.range_midpoint))
            or self.range_midpoint <= 0.0
            or (
                self.opposing_range_boundary_id is not None
                and (
                    not isinstance(self.opposing_range_boundary_id, str)
                    or not self.opposing_range_boundary_id
                )
            )
        ):
            raise ValueError("liquidity-route range context is invalid")


@dataclass(frozen=True)
class StructuralLevel:
    price: float
    side: str
    source_level_id: str
    observed_at: pd.Timestamp
    rationale: str

    def __post_init__(self) -> None:
        if self.side not in {"above", "below"}:
            raise ValueError("structural side must be above or below")
        if (
            not math.isfinite(float(self.price))
            or self.price <= 0
            or not self.source_level_id
        ):
            raise ValueError("structural level price and source are required")
        object.__setattr__(
            self, "observed_at", aware_timestamp(self.observed_at, name="structure.observed_at")
        )


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    kind: EventKind
    observed_at: pd.Timestamp
    timeframe: Timeframe
    side: str | None
    price: float | None
    strength: float
    source_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    entity_id: str | None = None
    lifecycle: str | None = None
    formed_at: pd.Timestamp | None = None
    confirmed_at: pd.Timestamp | None = None
    ended_at: pd.Timestamp | None = None
    direction: Direction | None = None
    transition_reason: str | None = None
    sequence_no: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at", aware_timestamp(self.observed_at, name="event.observed_at")
        )
        object.__setattr__(self, "strength", clamp(self.strength))
        for name in ("formed_at", "confirmed_at", "ended_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"event.{name}"),
                )
        if (self.entity_id is None) != (self.lifecycle is None):
            raise ValueError(
                "typed market event requires both entity_id and lifecycle"
            )
        if self.entity_id == "" or self.lifecycle == "":
            raise ValueError("typed market event identity cannot be empty")
        if type(self.sequence_no) is not int or self.sequence_no < 0:
            raise ValueError("market event sequence must be non-negative")
        if self.formed_at is not None and self.formed_at > self.observed_at:
            raise ValueError("event formation cannot be in the future")
        if self.confirmed_at is not None and (
            self.formed_at is None
            or self.confirmed_at < self.formed_at
            or self.confirmed_at > self.observed_at
        ):
            raise ValueError("event confirmation clock is invalid")
        if self.ended_at is not None and self.ended_at != self.observed_at:
            raise ValueError(
                "terminal event transition must end at its observation clock"
            )
        if self.ended_at is not None and not self.transition_reason:
            raise ValueError("terminal event requires a transition reason")
        if self.transition_reason == "":
            raise ValueError("event transition reason cannot be empty")


_TYPED_EVENT_ENTITY_PREFIXES: Mapping[EventKind, str] = {
    EventKind.SWING_STATE: "swing",
    EventKind.STRUCTURE_STATE: "structure",
    EventKind.BOS_STATE: "bos",
    EventKind.STRUCTURE_BREAK: "bos",
    EventKind.STRUCTURE_BREAK_FAILED: "bos",
    EventKind.SUPPORT_RESISTANCE_STATE: "zone",
    EventKind.LIQUIDITY_POOL_STATE: "pool",
    EventKind.FVG_STATE: "fvg",
    EventKind.ORDER_BLOCK_STATE: "order_block",
    EventKind.DEALING_RANGE_STATE: "range",
    EventKind.MANIPULATION_STATE: "manipulation",
    EventKind.ENTRY_PATH_STATE: "entry_path",
}


def typed_event_entity_key(event: MarketEvent) -> str | None:
    """Return the collision-safe key for a registered typed lifecycle event."""

    if event.entity_id is None:
        return None
    prefix = _TYPED_EVENT_ENTITY_PREFIXES.get(event.kind)
    if prefix is None:
        raise ValueError(
            "typed market event kind has no registered entity timeline"
        )
    if prefix == "pool":
        expected_prefix = "pool:"
        if not event.entity_id.startswith(expected_prefix):
            raise ValueError(
                "liquidity-pool event entity id must use the pool namespace"
            )
        suffix = event.entity_id[len(expected_prefix) :]
    else:
        suffix = event.entity_id
    if not suffix:
        raise ValueError("typed market event timeline identity is empty")
    return f"{prefix}:{suffix}"


@dataclass(frozen=True)
class FrameObservation:
    timeframe: Timeframe
    cutoff: pd.Timestamp
    bars: int
    metrics: Mapping[str, float]
    ready: bool = False
    swings: tuple[SwingPoint, ...] = ()
    structures: tuple[StructureSequenceState, ...] = ()
    structure_breaks: tuple[BreakOfStructureState, ...] = ()
    candle_structure: CandleStructureState | None = None
    support_resistance: tuple[SupportResistanceState, ...] = ()
    liquidity_pools: tuple[LiquidityPoolState, ...] = ()
    fair_value_gaps: tuple[FairValueGapState, ...] = ()
    order_blocks: tuple[OrderBlockState, ...] = ()
    dealing_ranges: tuple[DealingRangeState, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff", aware_timestamp(self.cutoff, name="frame.cutoff"))
        for name, value in self.metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{self.timeframe.value}.{name} is non-finite")
        for item in self.swings:
            if item.timeframe is not self.timeframe:
                raise ValueError("frame contains a swing from another timeframe")
        for item in self.structures:
            if item.timeframe is not self.timeframe:
                raise ValueError("frame contains a structure from another timeframe")
        for item in self.structure_breaks:
            if item.timeframe is not self.timeframe:
                raise ValueError("frame contains a BOS from another timeframe")
        if (
            self.candle_structure is not None
            and self.candle_structure.timeframe is not self.timeframe
        ):
            raise ValueError(
                "frame contains candle structure from another timeframe"
            )
        for item in self.support_resistance:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains support/resistance from another timeframe"
                )
        for item in self.liquidity_pools:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains a liquidity pool from another timeframe"
                )
        for item in self.fair_value_gaps:
            if item.timeframe is not self.timeframe:
                raise ValueError("frame contains an FVG from another timeframe")
        for item in self.order_blocks:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains an order block from another timeframe"
                )
        for item in self.dealing_ranges:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains a dealing range from another timeframe"
                )
        for swing in self.swings:
            if (
                (
                    swing.observed_at is not None
                    and swing.observed_at > self.cutoff
                )
                or (
                    swing.confirmed_at is not None
                    and swing.confirmed_at > self.cutoff
                )
                or (
                    swing.broken_at is not None
                    and swing.broken_at > self.cutoff
                )
            ):
                raise ValueError(
                    "frame contains a future-known swing state"
                )
        for structure in self.structures:
            if (
                (
                    structure.formed_at is not None
                    and structure.formed_at > self.cutoff
                )
                or (
                    structure.confirmed_at is not None
                    and structure.confirmed_at > self.cutoff
                )
                or (
                    structure.broken_at is not None
                    and structure.broken_at > self.cutoff
                )
                or (
                    structure.formation_failed_at is not None
                    and structure.formation_failed_at > self.cutoff
                )
            ):
                raise ValueError(
                    "frame contains a future-known structure state"
                )
        for item in self.structure_breaks:
            if (
                (
                    item.pending_at is not None
                    and item.pending_at > self.cutoff
                )
                or (
                    item.resolved_at is not None
                    and item.resolved_at > self.cutoff
                )
                or (
                    item.last_attempt_at is not None
                    and item.last_attempt_at > self.cutoff
                )
            ):
                raise ValueError("frame contains a future-known BOS state")
        if (
            self.candle_structure is not None
            and self.candle_structure.observed_at > self.cutoff
        ):
            raise ValueError("frame contains future candle structure")
        for item in self.support_resistance:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    item.confirmed_at,
                    item.tested_at,
                    item.broken_at,
                    item.reaccepted_at,
                    item.retired_at,
                    *item.touch_times,
                )
            ):
                raise ValueError(
                    "frame contains future support/resistance state"
                )
        for item in self.liquidity_pools:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    item.confirmed_at,
                    item.swept_at,
                    item.resolved_at,
                    *item.touch_times,
                )
            ):
                raise ValueError("frame contains future liquidity pool state")
        for item in self.fair_value_gaps:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    *item.source_candle_starts,
                    item.source_displacement_started_at,
                    item.source_displacement_active_at,
                    item.formed_at,
                    item.confirmed_at,
                    item.state_started_at,
                    item.last_updated_at,
                    item.partial_at,
                    item.mitigated_at,
                    item.invalidated_at,
                )
            ):
                raise ValueError("frame contains future FVG state")
        for item in self.order_blocks:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    item.source_displacement_started_at,
                    item.source_displacement_active_at,
                    item.source_bos_resolved_at,
                    item.anchor_start,
                    item.anchor_end,
                    item.formed_at,
                    item.confirmed_at,
                    item.state_started_at,
                    item.last_updated_at,
                    item.first_test_at,
                    item.mitigated_at,
                    item.failed_at,
                )
            ):
                raise ValueError("frame contains future order-block state")
        for item in self.dealing_ranges:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    item.lower_source_confirmed_at,
                    item.upper_source_confirmed_at,
                    item.lower_source_tested_at,
                    item.upper_source_tested_at,
                    item.formed_at,
                    item.mature_at,
                    item.broken_at,
                    item.state_started_at,
                    item.last_updated_at,
                )
            ):
                raise ValueError("frame contains future dealing-range state")
        range_ids = tuple(item.range_id for item in self.dealing_ranges)
        if (
            len(range_ids) != len(set(range_ids))
            or sum(
                item.lifecycle
                in {
                    DealingRangeLifecycle.FORMING,
                    DealingRangeLifecycle.MATURE,
                }
                for item in self.dealing_ranges
            )
            > 1
        ):
            raise ValueError(
                "frame dealing-range identities or live capacity are invalid"
            )


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


@dataclass(frozen=True)
class DisplacementTransitionObservation:
    """A descriptive lifecycle transition kept outside legacy event memory."""

    transition_id: str
    entity_id: str
    lifecycle: str
    reason: str | None
    observed_at: pd.Timestamp
    direction: Direction
    ordinal: int = 0
    started_at: pd.Timestamp | None = None
    active_at: pd.Timestamp | None = None
    state_started_at: pd.Timestamp | None = None
    terminal_at: pd.Timestamp | None = None
    prefix_last_admitted_at: pd.Timestamp | None = None
    terminal_evidence_candle_id: str | None = None
    admitted_candle_ids: tuple[str, ...] = ()
    state_metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at",
            aware_timestamp(self.observed_at, name="displacement transition"),
        )
        object.__setattr__(self, "state_metrics", tuple(self.state_metrics))
        object.__setattr__(
            self,
            "admitted_candle_ids",
            tuple(self.admitted_candle_ids),
        )
        terminal = self.lifecycle in {"exhausted", "censored"}
        if (
            not self.transition_id
            or not self.entity_id
            or self.lifecycle not in {"started", "active", "exhausted", "censored"}
            or not isinstance(self.direction, Direction)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or terminal != bool(self.reason)
            or not self.admitted_candle_ids
            or any(not value for value in self.admitted_candle_ids)
        ):
            raise ValueError("invalid displacement transition observation")
        clocks = (
            self.started_at,
            self.active_at,
            self.state_started_at,
            self.terminal_at,
            self.prefix_last_admitted_at,
        )
        if any(
            aware_timestamp(value, name="displacement transition clock")
            > self.observed_at
            for value in clocks
            if value is not None
        ):
            raise ValueError("displacement transition contains the future")
        if terminal != bool(self.terminal_at):
            raise ValueError("displacement terminal clock is inconsistent")
        if any(
            not name or not math.isfinite(float(value))
            for name, value in self.state_metrics
        ):
            raise ValueError("displacement transition metrics are invalid")


@dataclass(frozen=True)
class DisplacementObservation:
    """The current displacement state plus an isolated transition history."""

    asof: pd.Timestamp
    lifecycle: str
    current_entity_id: str | None
    current_direction: Direction | None
    current_state_observed_at: pd.Timestamp | None
    current_started_at: pd.Timestamp | None
    current_active_at: pd.Timestamp | None
    current_last_admitted_at: pd.Timestamp | None
    current_metrics: tuple[tuple[str, float], ...] | None
    latest_transition: DisplacementTransitionObservation | None
    recent_transitions: tuple[DisplacementTransitionObservation, ...] = ()
    transitions_this_update: tuple[
        DisplacementTransitionObservation, ...
    ] = ()
    reader_anomalies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "asof",
            aware_timestamp(self.asof, name="displacement observation"),
        )
        object.__setattr__(
            self, "recent_transitions", tuple(self.recent_transitions)
        )
        object.__setattr__(
            self,
            "transitions_this_update",
            tuple(self.transitions_this_update),
        )
        object.__setattr__(self, "reader_anomalies", tuple(self.reader_anomalies))
        if self.current_metrics is not None:
            object.__setattr__(
                self,
                "current_metrics",
                tuple(tuple(item) for item in self.current_metrics),
            )
        if (
            self.lifecycle not in {"idle", "started", "active"}
            or len(self.recent_transitions) > 64
        ):
            raise ValueError("invalid current displacement lifecycle or history")
        transitions_are_suffix = (
            not self.transitions_this_update
            or (
                len(self.transitions_this_update)
                <= len(self.recent_transitions)
                and self.recent_transitions[
                    -len(self.transitions_this_update) :
                ]
                == self.transitions_this_update
            )
        )
        if not transitions_are_suffix:
            raise ValueError("current displacement transitions are not a history suffix")
        if tuple(
            item.ordinal for item in self.transitions_this_update
        ) != tuple(range(len(self.transitions_this_update))):
            raise ValueError("current displacement transition order is invalid")
        if any(
            right.observed_at < left.observed_at
            for left, right in zip(
                self.recent_transitions[:-1],
                self.recent_transitions[1:],
            )
        ):
            raise ValueError("displacement transition history is out of order")
        expected_latest = (
            self.recent_transitions[-1] if self.recent_transitions else None
        )
        if self.latest_transition != expected_latest:
            raise ValueError("latest displacement transition does not match history")
        current_fields = (
            self.current_entity_id,
            self.current_direction,
            self.current_state_observed_at,
            self.current_started_at,
            self.current_active_at,
            self.current_last_admitted_at,
            self.current_metrics,
        )
        if self.lifecycle == "idle":
            if any(value is not None for value in current_fields):
                raise ValueError("idle displacement cannot expose current state")
        elif (
            not self.current_entity_id
            or not isinstance(self.current_direction, Direction)
            or self.current_state_observed_at is None
            or self.current_started_at is None
            or self.current_last_admitted_at is None
            or self.current_metrics is None
        ):
            raise ValueError("open displacement state is incomplete")
        if (
            (self.lifecycle == "started" and self.current_active_at is not None)
            or (self.lifecycle == "active" and self.current_active_at is None)
        ):
            raise ValueError("current displacement active clock is inconsistent")
        clocks = tuple(
            item.observed_at for item in self.recent_transitions
        ) + (
            self.current_state_observed_at,
            self.current_started_at,
            self.current_active_at,
            self.current_last_admitted_at,
        )
        if any(
            aware_timestamp(value, name="displacement clock") > self.asof
            for value in clocks
            if value is not None
        ):
            raise ValueError("displacement observation contains the future")
        if self.current_metrics is not None and any(
            not name or not math.isfinite(float(value))
            for name, value in self.current_metrics
        ):
            raise ValueError("current displacement metrics are invalid")


@dataclass(frozen=True)
class MarketObservation:
    asof: pd.Timestamp
    symbol: str
    instrument_id: int
    price: float
    frames: Mapping[Timeframe, FrameObservation]
    recent_events: tuple[MarketEvent, ...]
    event_durations_minutes: Mapping[str, int]
    execution: ExecutionObservation
    anomalies: tuple[str, ...] = ()
    displacement: DisplacementObservation | None = None
    liquidity_inventory: tuple[LiquidityInventoryItem, ...] = ()
    liquidity_pool_states: tuple[LiquidityPoolState, ...] = ()
    event_ages_minutes: Mapping[str, int] = field(default_factory=dict)
    retained_entity_timelines: Mapping[
        str,
        tuple[MarketEvent, ...],
    ] = field(default_factory=dict)
    incomplete_entity_timeline_keys: tuple[str, ...] = ()
    # One transport-level semantic delta for lightweight causal consumers.
    # Reducer snapshots remain the authoritative current state; these tuples
    # contain only baseline entities or entities whose lifecycle/evidence
    # changed on this completed update.  Age/freshness heartbeats are omitted.
    typed_transition_delta_available: bool = False
    liquidity_inventory_transitions_this_update: tuple[
        LiquidityInventoryItem,
        ...,
    ] = ()
    liquidity_pool_transitions_this_update: tuple[
        LiquidityPoolState,
        ...,
    ] = ()
    group3_fvg_transitions_this_update: tuple[
        FairValueGapState,
        ...,
    ] = ()
    group3_order_block_transitions_this_update: tuple[
        OrderBlockState,
        ...,
    ] = ()
    group4_range_transitions_this_update: tuple[
        DealingRangeState,
        ...,
    ] = ()
    group4_manipulation_transitions_this_update: tuple[
        ManipulationState,
        ...,
    ] = ()
    group5_entry_location_transitions_this_update: tuple[
        EntryLocationState,
        ...,
    ] = ()
    group5_reacceptance_transitions_this_update: tuple[
        QualifiedReacceptanceState,
        ...,
    ] = ()
    group5_micro_bos_transitions_this_update: tuple[
        MicroBOSReference,
        ...,
    ] = ()
    group5_path_transitions_this_update: tuple[
        PathSequenceState,
        ...,
    ] = ()
    group5_step_transitions_this_update: tuple[
        tuple[str, PathSequenceStep],
        ...,
    ] = ()
    group3_boundary_fvg_transitions: tuple[
        FairValueGapState,
        ...,
    ] = ()
    group3_boundary_order_block_transitions: tuple[
        OrderBlockState,
        ...,
    ] = ()
    group3_order_block_funnel: tuple[
        OrderBlockFunnelSnapshot,
        ...,
    ] = ()
    manipulations: tuple[ManipulationState, ...] = ()
    group4_boundary_range_transitions: tuple[
        DealingRangeState,
        ...,
    ] = ()
    group4_boundary_manipulation_transitions: tuple[
        ManipulationState,
        ...,
    ] = ()
    group4_ambiguous_sweep_item_ids: tuple[str, ...] = ()
    group4_atr_unready_sweep_item_ids: tuple[str, ...] = ()
    group4_source_dispositions: tuple[
        ManipulationSourceDisposition,
        ...,
    ] = ()
    group4_range_funnel: tuple[
        RangeFormationFunnelSnapshot,
        ...,
    ] = ()
    group5_typed_available: bool = False
    entry_locations: tuple[EntryLocationState, ...] = ()
    qualified_reacceptances: tuple[
        QualifiedReacceptanceState,
        ...,
    ] = ()
    micro_bos_references: tuple[MicroBOSReference, ...] = ()
    path_sequences: tuple[PathSequenceState, ...] = ()
    group5_boundary_path_transitions: tuple[
        PathSequenceState,
        ...,
    ] = ()
    group5_boundary_reacceptance_transitions: tuple[
        QualifiedReacceptanceState,
        ...,
    ] = ()
    active_timeframes: tuple[Timeframe, ...] = ()
    scale_registry_id: str = ""
    scene_revision_id: str | None = None
    scene_added_node_ids: tuple[str, ...] = ()
    scene_revised_node_ids: tuple[str, ...] = ()
    scene_added_edge_ids: tuple[str, ...] = ()
    scene_revised_edge_ids: tuple[str, ...] = ()
    scene_resolution_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="observation.asof"))
        active_timeframes = tuple(self.active_timeframes)
        if not active_timeframes:
            raise ValueError("observation requires an explicit scale registry")
        object.__setattr__(self, "active_timeframes", active_timeframes)
        for name in (
            "scene_added_node_ids",
            "scene_revised_node_ids",
            "scene_added_edge_ids",
            "scene_revised_edge_ids",
            "scene_resolution_event_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (
            "liquidity_inventory_transitions_this_update",
            "liquidity_pool_transitions_this_update",
            "group3_fvg_transitions_this_update",
            "group3_order_block_transitions_this_update",
            "group4_range_transitions_this_update",
            "group4_manipulation_transitions_this_update",
            "group5_entry_location_transitions_this_update",
            "group5_reacceptance_transitions_this_update",
            "group5_micro_bos_transitions_this_update",
            "group5_path_transitions_this_update",
            "group5_step_transitions_this_update",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "group3_boundary_fvg_transitions",
            tuple(self.group3_boundary_fvg_transitions),
        )
        object.__setattr__(
            self,
            "group3_boundary_order_block_transitions",
            tuple(self.group3_boundary_order_block_transitions),
        )
        object.__setattr__(
            self,
            "group3_order_block_funnel",
            tuple(self.group3_order_block_funnel),
        )
        object.__setattr__(
            self,
            "manipulations",
            tuple(self.manipulations),
        )
        object.__setattr__(
            self,
            "group4_boundary_range_transitions",
            tuple(self.group4_boundary_range_transitions),
        )
        object.__setattr__(
            self,
            "group4_boundary_manipulation_transitions",
            tuple(self.group4_boundary_manipulation_transitions),
        )
        object.__setattr__(
            self,
            "group4_ambiguous_sweep_item_ids",
            tuple(self.group4_ambiguous_sweep_item_ids),
        )
        object.__setattr__(
            self,
            "group4_atr_unready_sweep_item_ids",
            tuple(self.group4_atr_unready_sweep_item_ids),
        )
        object.__setattr__(
            self,
            "group4_source_dispositions",
            tuple(self.group4_source_dispositions),
        )
        object.__setattr__(
            self,
            "group4_range_funnel",
            tuple(self.group4_range_funnel),
        )
        object.__setattr__(
            self,
            "entry_locations",
            tuple(self.entry_locations),
        )
        object.__setattr__(
            self,
            "qualified_reacceptances",
            tuple(self.qualified_reacceptances),
        )
        object.__setattr__(
            self,
            "micro_bos_references",
            tuple(self.micro_bos_references),
        )
        object.__setattr__(
            self,
            "path_sequences",
            tuple(self.path_sequences),
        )
        object.__setattr__(
            self,
            "group5_boundary_path_transitions",
            tuple(self.group5_boundary_path_transitions),
        )
        object.__setattr__(
            self,
            "group5_boundary_reacceptance_transitions",
            tuple(self.group5_boundary_reacceptance_transitions),
        )
        object.__setattr__(self, "anomalies", tuple(self.anomalies))
        if type(self.typed_transition_delta_available) is not bool:
            raise ValueError(
                "typed transition delta availability must be boolean"
            )
        typed_transition_collections = (
            self.liquidity_inventory_transitions_this_update,
            self.liquidity_pool_transitions_this_update,
            self.group3_fvg_transitions_this_update,
            self.group3_order_block_transitions_this_update,
            self.group4_range_transitions_this_update,
            self.group4_manipulation_transitions_this_update,
            self.group5_entry_location_transitions_this_update,
            self.group5_reacceptance_transitions_this_update,
            self.group5_micro_bos_transitions_this_update,
            self.group5_path_transitions_this_update,
            self.group5_step_transitions_this_update,
        )
        if (
            not self.typed_transition_delta_available
            and any(typed_transition_collections)
        ):
            raise ValueError(
                "typed transition delta payload requires availability"
            )
        typed_transition_contracts = (
            (
                self.liquidity_inventory_transitions_this_update,
                LiquidityInventoryItem,
            ),
            (
                self.liquidity_pool_transitions_this_update,
                LiquidityPoolState,
            ),
            (
                self.group3_fvg_transitions_this_update,
                FairValueGapState,
            ),
            (
                self.group3_order_block_transitions_this_update,
                OrderBlockState,
            ),
            (
                self.group4_range_transitions_this_update,
                DealingRangeState,
            ),
            (
                self.group4_manipulation_transitions_this_update,
                ManipulationState,
            ),
            (
                self.group5_entry_location_transitions_this_update,
                EntryLocationState,
            ),
            (
                self.group5_reacceptance_transitions_this_update,
                QualifiedReacceptanceState,
            ),
            (
                self.group5_micro_bos_transitions_this_update,
                MicroBOSReference,
            ),
            (
                self.group5_path_transitions_this_update,
                PathSequenceState,
            ),
        )
        if any(
            not isinstance(item, expected_type)
            for collection, expected_type in typed_transition_contracts
            for item in collection
        ):
            raise TypeError("typed transition delta contains an invalid state")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], PathSequenceStep)
            for item in self.group5_step_transitions_this_update
        ):
            raise TypeError("Group 5 step delta contains an invalid transition")
        if not self.symbol or int(self.instrument_id) < 0:
            raise ValueError("observation contract identity is invalid")
        if not math.isfinite(float(self.price)):
            raise ValueError("observation price is invalid")
        if (
            any(
                not isinstance(value, str) or not value
                for value in self.anomalies
            )
            or len(self.anomalies) != len(set(self.anomalies))
        ):
            raise ValueError("observation anomalies are invalid")
        if (
            set(self.frames) != set(active_timeframes)
            or not set(CORE_TIMEFRAMES).issubset(active_timeframes)
            or len(active_timeframes) != len(set(active_timeframes))
            or not self.scale_registry_id
        ):
            raise ValueError(
                "observation frames must match its enabled scale registry"
            )
        if self.scene_revision_id is not None and not self.scene_revision_id:
            raise ValueError("scene revision identity cannot be empty")
        if self.scene_revision_id is None and any(
            (
                self.scene_added_node_ids,
                self.scene_revised_node_ids,
                self.scene_added_edge_ids,
                self.scene_revised_edge_ids,
                self.scene_resolution_event_ids,
            )
        ):
            raise ValueError("scene delta identities require a scene revision")
        if any(
            len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
            for values in (
                self.scene_added_node_ids,
                self.scene_revised_node_ids,
                self.scene_added_edge_ids,
                self.scene_revised_edge_ids,
                self.scene_resolution_event_ids,
            )
        ):
            raise ValueError("scene delta identities are invalid")
        for timeframe, frame in self.frames.items():
            if frame.timeframe is not timeframe:
                raise ValueError("frame mapping key does not match frame timeframe")
            if frame.cutoff > self.asof:
                raise ValueError("observation includes an uncompleted future frame")
            if any(
                (
                    state.symbol,
                    state.instrument_id,
                )
                != (self.symbol, self.instrument_id)
                for state in (
                    *frame.fair_value_gaps,
                    *frame.order_blocks,
                    *frame.dealing_ranges,
                )
            ):
                raise ValueError(
                    "observation contains a cross-contract typed entity"
                )
        if any(event.observed_at > self.asof for event in self.recent_events):
            raise ValueError("observation contains a future market event")
        if (
            any(
                not isinstance(item, OrderBlockFunnelSnapshot)
                or item.observed_at > self.asof
                for item in self.group3_order_block_funnel
            )
            or len(
                {
                    item.observed_at
                    for item in self.group3_order_block_funnel
                }
            )
            != len(self.group3_order_block_funnel)
        ):
            raise ValueError(
                "observation contains an invalid Group 3 OB funnel"
            )
        boundary_anomaly_by_reason = {
            "data_gap_reset": "data_gap_history_reset",
            "contract_change_reset": "contract_change_history_reset",
            "data_anomaly": "data_anomaly",
            "tick_size_mismatch": "tick_size_mismatch",
        }
        fvg_boundary_ids = tuple(
            state.fvg_id
            for state in self.group3_boundary_fvg_transitions
        )
        order_block_boundary_ids = tuple(
            state.order_block_id
            for state in self.group3_boundary_order_block_transitions
        )
        range_boundary_ids = tuple(
            state.range_id
            for state in self.group4_boundary_range_transitions
        )
        manipulation_boundary_ids = tuple(
            state.manipulation_id
            for state in self.group4_boundary_manipulation_transitions
        )
        if (
            len(fvg_boundary_ids) != len(set(fvg_boundary_ids))
            or len(order_block_boundary_ids)
            != len(set(order_block_boundary_ids))
            or len(range_boundary_ids) != len(set(range_boundary_ids))
            or len(manipulation_boundary_ids)
            != len(set(manipulation_boundary_ids))
        ):
            raise ValueError(
                "observation contains duplicate typed boundary transitions"
            )
        if any(
            state.lifecycle is not FairValueGapLifecycle.INVALIDATED
            or state.invalidated_at != self.asof
            or state.transition_reason
            not in boundary_anomaly_by_reason
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            for state in self.group3_boundary_fvg_transitions
        ):
            raise ValueError(
                "observation contains an invalid boundary FVG transition"
            )
        if any(
            state.lifecycle is not OrderBlockLifecycle.FAILED
            or state.failed_at != self.asof
            or state.transition_reason
            not in boundary_anomaly_by_reason
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            for state in self.group3_boundary_order_block_transitions
        ):
            raise ValueError(
                "observation contains an invalid boundary OB transition"
            )
        if any(
            state.lifecycle is not DealingRangeLifecycle.BROKEN
            or state.broken_at != self.asof
            or state.transition_reason
            not in boundary_anomaly_by_reason
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            for state in self.group4_boundary_range_transitions
        ):
            raise ValueError(
                "observation contains an invalid boundary range transition"
            )
        if any(
            state.lifecycle is not ManipulationLifecycle.SWEPT
            or state.censored_at != self.asof
            or state.last_updated_at != self.asof
            or state.deadline_elapsed
            or state.transition_reason
            not in boundary_anomaly_by_reason
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            for state in self.group4_boundary_manipulation_transitions
        ):
            raise ValueError(
                "observation contains an invalid boundary manipulation transition"
            )
        boundary_reasons = {
            state.transition_reason
            for state in (
                *self.group3_boundary_fvg_transitions,
                *self.group3_boundary_order_block_transitions,
                *self.group4_boundary_range_transitions,
                *self.group4_boundary_manipulation_transitions,
                *self.group5_boundary_path_transitions,
            )
        }
        if len(boundary_reasons) > 1:
            raise ValueError(
                "observation mixes typed boundary transition reasons"
            )
        if any(value < 0 for value in self.event_durations_minutes.values()):
            raise ValueError("event durations cannot be negative")
        if any(value < 0 for value in self.event_ages_minutes.values()):
            raise ValueError("event ages cannot be negative")
        timeline_event_ids: set[str] = set()
        for entity_key, timeline in self.retained_entity_timelines.items():
            if (
                not isinstance(entity_key, str)
                or not entity_key
                or not isinstance(timeline, tuple)
                or not timeline
            ):
                raise ValueError(
                    "retained entity timeline identity or history is invalid"
                )
            prior_order: tuple[pd.Timestamp, int, str] | None = None
            lifecycles: set[str] = set()
            for event in timeline:
                if typed_event_entity_key(event) != entity_key:
                    raise ValueError(
                        "retained entity timeline key disagrees with its event"
                    )
                order = (
                    event.observed_at,
                    event.sequence_no,
                    event.event_id,
                )
                if prior_order is not None and order <= prior_order:
                    raise ValueError(
                        "retained entity timeline is not causally ordered"
                    )
                if event.observed_at > self.asof:
                    raise ValueError(
                        "retained entity timeline contains a future event"
                    )
                if event.event_id in timeline_event_ids:
                    raise ValueError(
                        "retained entity timelines contain a duplicate event"
                    )
                if event.lifecycle in lifecycles:
                    raise ValueError(
                        "retained entity timeline repeats a lifecycle"
                    )
                if (
                    event.event_id not in self.event_durations_minutes
                    or event.event_id not in self.event_ages_minutes
                ):
                    raise ValueError(
                        "retained entity timeline lacks duration or age"
                    )
                timeline_event_ids.add(event.event_id)
                lifecycles.add(event.lifecycle)
                prior_order = order
        if (
            tuple(sorted(set(self.incomplete_entity_timeline_keys)))
            != self.incomplete_entity_timeline_keys
            or not set(self.incomplete_entity_timeline_keys).issubset(
                self.retained_entity_timelines
            )
        ):
            raise ValueError(
                "incomplete entity timeline keys are invalid"
            )
        if (
            self.displacement is not None
            and self.displacement.asof != self.asof
        ):
            raise ValueError("displacement and market observation clocks differ")
        manipulation_ids = tuple(
            item.manipulation_id for item in self.manipulations
        )
        if (
            len(manipulation_ids) != len(set(manipulation_ids))
            or sum(
                item.lifecycle is ManipulationLifecycle.SWEPT
                and item.censored_at is None
                for item in self.manipulations
            )
            > 1
        ):
            raise ValueError(
                "observation manipulation identities or live capacity are invalid"
            )
        if any(
            (item.symbol, item.instrument_id)
            != (self.symbol, self.instrument_id)
            or item.last_updated_at > self.frames[Timeframe.M1].cutoff
            or (
                item.censored_at is not None
                and not item.deadline_elapsed
            )
            for item in self.manipulations
        ):
            raise ValueError(
                "ordinary manipulation contract, clock or boundary state is invalid"
            )
        inventory_ids = [
            item.item_id for item in self.liquidity_inventory
        ]
        if len(inventory_ids) != len(set(inventory_ids)):
            raise ValueError(
                "observation contains duplicate liquidity inventory ids"
            )
        if any(
            not isinstance(item, ManipulationSourceDisposition)
            for item in self.group4_source_dispositions
        ):
            raise TypeError(
                "observation Group 4 source disposition is not typed"
            )
        if (
            any(
                not isinstance(item, RangeFormationFunnelSnapshot)
                or item.observed_at > self.asof
                for item in self.group4_range_funnel
            )
            or len(
                {item.observed_at for item in self.group4_range_funnel}
            )
            != len(self.group4_range_funnel)
        ):
            raise ValueError(
                "observation Group 4 range funnel is invalid"
            )
        disposition_ids = tuple(
            item.source_inventory_item_id
            for item in self.group4_source_dispositions
        )
        if (
            any(
                item.observed_at > self.asof
                for item in self.group4_source_dispositions
            )
            or len(disposition_ids) != len(set(disposition_ids))
        ):
            raise ValueError(
                "observation Group 4 source dispositions are invalid"
            )
        disposition_ambiguous_ids = {
            item.source_inventory_item_id
            for item in self.group4_source_dispositions
            if item.disposition
            is ManipulationSourceDispositionKind.AMBIGUOUS_DUAL_SIDE
        }
        disposition_atr_unready_ids = {
            item.source_inventory_item_id
            for item in self.group4_source_dispositions
            if item.disposition
            is ManipulationSourceDispositionKind.ATR_UNREADY
        }
        if (
            len(self.group4_ambiguous_sweep_item_ids)
            != len(set(self.group4_ambiguous_sweep_item_ids))
            or len(self.group4_atr_unready_sweep_item_ids)
            != len(set(self.group4_atr_unready_sweep_item_ids))
            or set(self.group4_ambiguous_sweep_item_ids)
            & set(self.group4_atr_unready_sweep_item_ids)
            or any(
                not value or value not in set(inventory_ids)
                for value in (
                    *self.group4_ambiguous_sweep_item_ids,
                    *self.group4_atr_unready_sweep_item_ids,
                )
            )
            or (
                self.group4_source_dispositions
                and (
                    disposition_ambiguous_ids
                    != set(self.group4_ambiguous_sweep_item_ids)
                    or disposition_atr_unready_ids
                    != set(self.group4_atr_unready_sweep_item_ids)
                )
            )
        ):
            raise ValueError(
                "observation inventory or Group 4 ambiguity is invalid"
            )
        for item in self.liquidity_inventory:
            if item.lifecycle is LiquidityInventoryLifecycle.TARGETED:
                raise ValueError(
                    "the descriptive eye cannot target liquidity"
                )
            source_cutoff = self.frames[item.timeframe].cutoff
            if (
                item.formed_at > source_cutoff
                or item.confirmed_at > source_cutoff
            ):
                raise ValueError(
                    "liquidity inventory postdates its source frame cutoff"
                )
            if any(
                value is not None and value > self.asof
                for value in (
                    item.confirmed_at,
                    item.targeted_at,
                    item.consumed_at,
                )
            ):
                raise ValueError(
                    "observation contains future liquidity inventory state"
                )
            if (
                item.consumed_at is not None
                and item.consumed_at > self.frames[Timeframe.M1].cutoff
            ):
                raise ValueError(
                    "liquidity consumption postdates the completed 1m cutoff"
                )
        pool_ids = [item.pool_id for item in self.liquidity_pool_states]
        if len(pool_ids) != len(set(pool_ids)):
            raise ValueError(
                "observation contains duplicate liquidity pool ids"
            )
        inventory_by_id = {
            item.item_id: item for item in self.liquidity_inventory
        }
        swing_ids_by_timeframe = {
            timeframe: {swing.swing_id for swing in frame.swings}
            for timeframe, frame in self.frames.items()
        }
        range_ids = {
            item.range_id
            for item in self.frames[Timeframe.H1].dealing_ranges
        }
        for item in self.liquidity_inventory:
            if item.kind == "range_boundary" and (
                item.timeframe is not Timeframe.H1
                or len(set(item.source_ids) & range_ids) != 1
            ):
                raise ValueError(
                    "range-boundary inventory lacks one retained range identity"
                )
            if item.kind == "swing" and (
                len(item.source_ids) != 1
                or item.source_ids[0]
                not in swing_ids_by_timeframe[item.timeframe]
            ):
                raise ValueError(
                    "swing inventory lacks one retained swing identity"
                )
        frozen_source_pool_id_list = [
            pool.pool_id
            for frame in self.frames.values()
            for pool in frame.liquidity_pools
        ]
        if (
            len(frozen_source_pool_id_list)
            != len(set(frozen_source_pool_id_list))
            or set(frozen_source_pool_id_list) != set(pool_ids)
        ):
            raise ValueError(
                "frozen pool sources and authoritative pool set disagree"
            )
        inventory_pool_ids = {
            item.item_id
            for item in self.liquidity_inventory
            if item.kind in {"equal_highs", "equal_lows"}
        }
        if inventory_pool_ids != {
            f"pool:{pool_id}" for pool_id in pool_ids
        }:
            raise ValueError(
                "pool inventory and authoritative pool set disagree"
            )
        for item in self.manipulations:
            if item.source_inventory_item_id in inventory_by_id:
                continue
            if (
                (
                    item.lifecycle is ManipulationLifecycle.SWEPT
                    and item.censored_at is None
                )
                or item.last_updated_at == self.asof
            ):
                continue
            else:
                raise ValueError(
                    "manipulation lacks its retained inventory identity"
                )
        if type(self.group5_typed_available) is not bool:
            raise ValueError("Group 5 availability flag must be boolean")
        if (
            not self.group5_typed_available
            and (
                self.entry_locations
                or self.qualified_reacceptances
                or self.micro_bos_references
                or self.path_sequences
                or self.group5_boundary_path_transitions
                or self.group5_boundary_reacceptance_transitions
                or self.group5_entry_location_transitions_this_update
                or self.group5_reacceptance_transitions_this_update
                or self.group5_micro_bos_transitions_this_update
                or self.group5_path_transitions_this_update
                or self.group5_step_transitions_this_update
            )
        ):
            raise ValueError(
                "typed Group 5 state requires an available data contract"
            )
        location_ids = tuple(
            item.location_id for item in self.entry_locations
        )
        reacceptance_ids = tuple(
            item.reacceptance_id
            for item in self.qualified_reacceptances
        )
        reference_ids = tuple(
            item.reference_id for item in self.micro_bos_references
        )
        sequence_ids = tuple(
            item.sequence_id for item in self.path_sequences
        )
        boundary_sequence_ids = tuple(
            item.sequence_id
            for item in self.group5_boundary_path_transitions
        )
        boundary_reacceptance_ids = tuple(
            item.reacceptance_id
            for item
            in self.group5_boundary_reacceptance_transitions
        )
        if any(
            len(values) != len(set(values))
            for values in (
                location_ids,
                reacceptance_ids,
                reference_ids,
                sequence_ids,
                boundary_sequence_ids,
                boundary_reacceptance_ids,
            )
        ):
            raise ValueError("Group 5 observation identities repeat")
        delta_paths_by_id: dict[str, list[PathSequenceState]] = {}
        for state in self.group5_path_transitions_this_update:
            delta_paths_by_id.setdefault(state.sequence_id, []).append(state)
        if any(
            sequence_id not in delta_paths_by_id
            or not any(
                step in state.steps
                for state in delta_paths_by_id[sequence_id]
            )
            for sequence_id, step
            in self.group5_step_transitions_this_update
        ):
            raise ValueError(
                "Group 5 step delta lacks its path-state transition"
            )
        if set(sequence_ids) & set(boundary_sequence_ids):
            raise ValueError(
                "ordinary and boundary Group 5 paths overlap"
            )
        if set(reacceptance_ids) & set(boundary_reacceptance_ids):
            raise ValueError(
                "ordinary and boundary Group 5 reacceptances overlap"
            )
        group5_clocked = (
            *self.entry_locations,
            *self.qualified_reacceptances,
            *self.path_sequences,
        )
        if any(
            item.symbol != self.symbol
            or item.instrument_id != self.instrument_id
            or item.last_updated_at > self.frames[Timeframe.M1].cutoff
            or item.last_updated_at > self.asof
            for item in group5_clocked
        ):
            raise ValueError(
                "Group 5 state contract or completed clock is invalid"
            )
        if any(
            reference.resolved_at > self.frames[Timeframe.M1].cutoff
            or reference.resolved_at > self.asof
            for reference in self.micro_bos_references
        ):
            raise ValueError("Group 5 micro BOS is in the future")
        if any(
            state.lifecycle is not PathSequenceLifecycle.CENSORED
            or state.ended_at != self.asof
            or state.last_updated_at != self.asof
            or state.transition_reason
            not in GROUP5_HARD_BOUNDARY_REASONS
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            or (
                state.transition_reason == "contract_change_reset"
                and (
                    state.symbol,
                    state.instrument_id,
                )
                == (self.symbol, self.instrument_id)
            )
            or (
                state.transition_reason != "contract_change_reset"
                and (
                    state.symbol,
                    state.instrument_id,
                )
                != (self.symbol, self.instrument_id)
            )
            for state in self.group5_boundary_path_transitions
        ):
            raise ValueError(
                "Group 5 boundary path transition is invalid"
            )
        if any(
            state.lifecycle
            is not QualifiedReacceptanceLifecycle.CENSORED
            or state.censored_at != self.asof
            or state.last_updated_at != self.asof
            or state.transition_reason != "hard_boundary_censored"
            for state
            in self.group5_boundary_reacceptance_transitions
        ):
            raise ValueError(
                "Group 5 boundary reacceptance transition is invalid"
            )
        boundary_path_contexts = {
            (state.context_kind, state.context_id)
            for state in self.group5_boundary_path_transitions
        }
        if any(
            (
                (
                    "zone_return"
                    if state.context_kind == "entry_zone"
                    else "pool_reversal"
                ),
                state.context_id,
            )
            not in boundary_path_contexts
            for state
            in self.group5_boundary_reacceptance_transitions
        ):
            raise ValueError(
                "boundary reacceptance lacks its censored path context"
            )
        path_contexts = {
            (state.context_kind, state.context_id)
            for state in self.path_sequences
        }
        if len(path_contexts) != len(self.path_sequences):
            raise ValueError(
                "Group 5 retained path contexts must be unique"
            )
        if any(
            ("zone_return", state.location_id) not in path_contexts
            for state in self.entry_locations
        ):
            raise ValueError(
                "entry location lacks its retained path sequence"
            )
        if any(
            (
                (
                    "zone_return"
                    if state.context_kind == "entry_zone"
                    else "pool_reversal"
                ),
                state.context_id,
            )
            not in path_contexts
            for state in self.qualified_reacceptances
        ):
            raise ValueError(
                "qualified reacceptance lacks its path context"
            )
        if any(
            (reference.context_kind, reference.context_id)
            not in path_contexts
            for reference in self.micro_bos_references
        ):
            raise ValueError(
                "micro BOS reference lacks its path context"
            )

    def frame(self, timeframe: Timeframe) -> FrameObservation:
        return self.frames[timeframe]


@dataclass(frozen=True)
class Evidence:
    primitive: str
    value: float
    weight: float
    supports: bool
    observed_at: pd.Timestamp
    explanation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", clamp(self.value))
        if not math.isfinite(float(self.weight)) or self.weight < 0:
            raise ValueError("evidence weight is invalid")
        object.__setattr__(
            self, "observed_at", aware_timestamp(self.observed_at, name="evidence.observed_at")
        )


@dataclass(frozen=True)
class SequenceStepState:
    step_id: str
    satisfied: bool
    value: float
    observed_at: pd.Timestamp | None
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("sequence step id is required")
        object.__setattr__(self, "value", clamp(self.value))
        if self.observed_at is not None:
            object.__setattr__(
                self,
                "observed_at",
                aware_timestamp(self.observed_at, name="sequence_step.observed_at"),
            )
        if self.satisfied and self.observed_at is None:
            raise ValueError("a satisfied sequence step requires an observation clock")


@dataclass(frozen=True)
class HypothesisSequenceState:
    protocol_version: str
    protocol_hash: str
    setup_id: str | None
    steps: tuple[SequenceStepState, ...]
    started_at: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not self.protocol_version or not self.protocol_hash:
            raise ValueError("sequence state requires a frozen protocol identity")
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise ValueError("sequence state contains duplicate step ids")
        if self.started_at is not None:
            object.__setattr__(
                self,
                "started_at",
                aware_timestamp(self.started_at, name="sequence.started_at"),
            )
        if (self.setup_id is None) != (self.started_at is None):
            raise ValueError("sequence setup identity and start clock must coexist")

    @property
    def completed_steps(self) -> int:
        return sum(step.satisfied for step in self.steps)

    @property
    def complete(self) -> bool:
        return bool(self.steps) and self.completed_steps == len(self.steps)


@dataclass(frozen=True)
class FrozenRangeAuctionContext:
    """Decision-time snapshot of the exact mature-range failed auction."""

    range_id: str
    manipulation_id: str
    lower_bound: float
    upper_bound: float
    midpoint: float
    value_price: float
    mature_at: pd.Timestamp
    manipulation_side: str
    swept_at: pd.Timestamp
    manipulation_extreme: float
    reentry_candidate_at: pd.Timestamp
    reentered_at: pd.Timestamp
    reentry_price: float
    opposite_liquidity_id: str

    def __post_init__(self) -> None:
        for name in (
            "mature_at",
            "swept_at",
            "reentry_candidate_at",
            "reentered_at",
        ):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"range_auction.{name}",
                ),
            )
        prices = (
            self.lower_bound,
            self.upper_bound,
            self.midpoint,
            self.value_price,
            self.manipulation_extreme,
            self.reentry_price,
        )
        if (
            not self.range_id
            or not self.manipulation_id
            or not self.opposite_liquidity_id
            or self.manipulation_side not in {"above", "below"}
            or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in prices
            )
            or not self.lower_bound < self.upper_bound
            or not math.isclose(
                self.midpoint,
                (self.lower_bound + self.upper_bound) / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.value_price,
                self.midpoint,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not self.lower_bound
            <= self.reentry_price
            <= self.upper_bound
            or (
                self.manipulation_side == "above"
                and self.manipulation_extreme <= self.upper_bound
            )
            or (
                self.manipulation_side == "below"
                and self.manipulation_extreme >= self.lower_bound
            )
            or not self.mature_at
            < self.swept_at
            < self.reentry_candidate_at
            < self.reentered_at
        ):
            raise ValueError("frozen range-auction context is invalid")


@dataclass(frozen=True)
class TradePlan:
    playbook: Playbook
    direction: Direction
    planned_entry: float
    invalidation: StructuralLevel
    targets: tuple[LiquidityLevel, ...]
    risk_points: float
    primary_target_R: float
    remaining_path_R: float
    deadline: pd.Timestamp
    setup_id: str | None = None
    entry_location_id: str | None = None
    entry_path_id: str | None = None
    entry_zone_lower: float | None = None
    entry_zone_upper: float | None = None
    selected_draw_id: str | None = None
    draw_selection: DrawSelection | None = None
    range_auction: FrozenRangeAuctionContext | None = None
    liquidity_route: LiquidityRoute | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "deadline", aware_timestamp(self.deadline, name="plan.deadline"))
        numeric = (
            self.planned_entry,
            self.risk_points,
            self.primary_target_R,
            self.remaining_path_R,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("plan contains non-finite values")
        if self.planned_entry <= 0 or self.risk_points <= 0:
            raise ValueError("plan risk must be positive")
        if not self.targets:
            raise ValueError("plan requires visible liquidity targets")
        expected_side = self.direction.invalidation_side
        if self.invalidation.side != expected_side:
            raise ValueError("plan invalidation is on the wrong thesis side")
        typed_identity = (
            self.setup_id,
            self.entry_location_id,
            self.entry_path_id,
            self.entry_zone_lower,
            self.entry_zone_upper,
            self.selected_draw_id,
        )
        if any(value is not None for value in typed_identity):
            if (
                any(
                    value is None
                    for value in typed_identity
                )
                or not all(
                    isinstance(value, str) and value
                    for value in (
                        self.setup_id,
                        self.entry_location_id,
                        self.entry_path_id,
                        self.selected_draw_id,
                    )
                )
                or not math.isfinite(float(self.entry_zone_lower))
                or not math.isfinite(float(self.entry_zone_upper))
                or not (
                    0
                    < float(self.entry_zone_lower)
                    < float(self.entry_zone_upper)
                )
                or not (
                    float(self.entry_zone_lower)
                    <= self.planned_entry
                    <= float(self.entry_zone_upper)
                )
                or self.targets[0].level_id != self.selected_draw_id
            ):
                raise ValueError(
                    "typed trade plan entry-zone or draw identity is invalid"
                )
        if self.draw_selection is not None and (
            self.selected_draw_id != self.draw_selection.draw_id
            or self.targets[0].level_id != self.draw_selection.draw_id
            or self.targets[0].timeframe
            is not self.draw_selection.source_timeframe
            or self.targets[0].side != self.draw_selection.side
            or not math.isclose(
                self.targets[0].price,
                self.draw_selection.price,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or self.draw_selection.selected_at > self.deadline
        ):
            raise ValueError("trade plan and targeted draw overlay disagree")
        if self.liquidity_route is not None and (
            self.liquidity_route.primary_deliverable_target_id
            != self.selected_draw_id
            or self.liquidity_route.selected_at > self.deadline
        ):
            raise ValueError(
                "trade plan and frozen liquidity route disagree"
            )
        if (
            self.playbook is Playbook.FAILED_AUCTION_VALUE_RETURN
            and self.setup_id is not None
        ):
            if (
                self.range_auction is None
                or self.draw_selection is None
                or self.range_auction.opposite_liquidity_id
                != self.selected_draw_id
                or not self.range_auction.lower_bound
                <= self.planned_entry
                <= self.range_auction.upper_bound
                or self.range_auction.manipulation_side
                != (
                    "below"
                    if self.direction is Direction.LONG
                    else "above"
                )
                or self.invalidation.source_level_id
                != self.range_auction.manipulation_id
                or self.invalidation.observed_at
                != self.range_auction.swept_at
                or not math.isclose(
                    self.invalidation.price,
                    self.range_auction.manipulation_extreme,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or self.targets[0].side
                != self.direction.opposing_liquidity_side
                or not math.isclose(
                    self.targets[0].price,
                    (
                        self.range_auction.upper_bound
                        if self.direction is Direction.LONG
                        else self.range_auction.lower_bound
                    ),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "FAVR plan requires its frozen range-auction context"
                )
        elif self.range_auction is not None:
            raise ValueError("only FAVR may carry a range-auction context")


@dataclass(frozen=True)
class HypothesisBelief:
    playbook: Playbook
    direction: Direction
    probability: float
    phase: PlaybookPhase
    phase_started_at: pd.Timestamp
    supporting: tuple[Evidence, ...]
    contradicting: tuple[Evidence, ...]
    invalidation: StructuralLevel | None
    deliverable_targets: tuple[LiquidityLevel, ...]
    remaining_path_R: float | None
    uncertainty: float
    plan: TradePlan | None
    sequence: HypothesisSequenceState | None = None
    raw_probability: float | None = None
    calibration_version: str = "identity-unvalidated"
    thesis_strength: float | None = None
    sequence_progress: float | None = None
    location_quality: float | None = None
    entry_readiness: float | None = None
    delivery_quality: float | None = None
    evidence_group_scores: Mapping[str, float] = field(
        default_factory=dict
    )
    hard_gate_results: Mapping[str, bool] = field(
        default_factory=dict
    )
    setup_context_id: str | None = None
    entry_location_id: str | None = None
    context_id: str | None = None
    episode_id: str | None = None
    episode_deadline: pd.Timestamp | None = None
    initiating_event_id: str | None = None
    evidence_revision_id: str | None = None
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None
    terminal_source_ids: tuple[str, ...] = ()
    draw_selection: DrawSelection | None = None
    raw_quality_dimensions: Mapping[str, float] = field(
        default_factory=dict
    )
    liquidity_route: LiquidityRoute | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_started_at",
            aware_timestamp(
                self.phase_started_at,
                name="belief.phase_started_at",
            ),
        )
        object.__setattr__(self, "probability", clamp(self.probability))
        if self.raw_probability is not None:
            object.__setattr__(
                self,
                "raw_probability",
                clamp(self.raw_probability),
            )
        object.__setattr__(self, "uncertainty", clamp(self.uncertainty))
        for name in (
            "thesis_strength",
            "sequence_progress",
            "location_quality",
            "entry_readiness",
            "delivery_quality",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, clamp(value))
        group_scores = dict(self.evidence_group_scores)
        hard_gates = dict(self.hard_gate_results)
        raw_dimensions = dict(self.raw_quality_dimensions)
        object.__setattr__(
            self,
            "raw_quality_dimensions",
            raw_dimensions,
        )
        expected_groups = {
            "structure",
            "displacement",
            "location",
            "liquidity",
            "trigger",
            "execution",
        }
        quality_values = (
            self.thesis_strength,
            self.sequence_progress,
            self.location_quality,
            self.entry_readiness,
            self.delivery_quality,
        )
        if (
            any(value is None for value in quality_values)
            or set(group_scores) != expected_groups
            or not hard_gates
        ):
            raise ValueError(
                "typed belief requires all five quality dimensions, "
                "six evidence groups and hard gates"
            )
        if (
            set(group_scores) != expected_groups
            or any(
                not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in group_scores.values()
            )
        ):
            raise ValueError("belief evidence-group scores are invalid")
        if any(type(value) is not bool for value in hard_gates.values()):
            raise ValueError("belief hard-gate results must be boolean")
        expected_raw_dimensions = {
            "thesis_strength",
            "sequence_progress",
            "location_quality",
            "entry_readiness",
            "delivery_quality",
            "uncertainty",
        }
        if (
            set(raw_dimensions) != expected_raw_dimensions
            or any(
                not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in raw_dimensions.values()
            )
        ):
            raise ValueError(
                "belief raw quality dimensions are invalid"
            )
        if self.setup_context_id == "" or self.entry_location_id == "":
            raise ValueError("belief typed context identity cannot be empty")
        for name in (
            "context_id",
            "episode_id",
            "initiating_event_id",
            "evidence_revision_id",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value
            ):
                raise ValueError(
                    f"belief {name} must be non-empty text when present"
                )
        if self.episode_deadline is not None:
            object.__setattr__(
                self,
                "episode_deadline",
                aware_timestamp(
                    self.episode_deadline,
                    name="belief.episode_deadline",
                ),
            )
        object.__setattr__(
            self,
            "terminal_source_ids",
            tuple(self.terminal_source_ids),
        )
        if (
            len(self.terminal_source_ids)
            != len(set(self.terminal_source_ids))
            or any(
                not isinstance(value, str) or not value
                for value in self.terminal_source_ids
            )
        ):
            raise ValueError(
                "belief terminal source identities are invalid"
            )
        if (
            self.entry_location_id is not None
            and self.setup_context_id is None
        ):
            raise ValueError(
                "belief entry location requires its setup context"
            )
        if (
            self.sequence is not None
            and self.sequence.setup_id != self.setup_context_id
        ):
            raise ValueError(
                "typed belief and hypothesis sequence identities disagree"
            )
        if (
            self.episode_id is not None
            and self.setup_context_id != self.episode_id
        ):
            raise ValueError(
                "belief episode identity must own the active setup"
            )
        if (self.episode_id is None) != (self.episode_deadline is None):
            raise ValueError(
                "belief episode identity and frozen deadline must coexist"
            )
        if (
            self.episode_deadline is not None
            and self.sequence is not None
            and self.sequence.started_at is not None
            and self.episode_deadline < self.sequence.started_at
        ):
            raise ValueError(
                "belief episode deadline must follow episode formation"
            )
        if (
            self.plan is not None
            and self.episode_deadline is not None
            and self.plan.deadline > self.episode_deadline
        ):
            raise ValueError(
                "trade plan cannot extend its frozen episode deadline"
            )
        terminal = self.phase in {
            PlaybookPhase.COMPLETED,
            PlaybookPhase.INVALIDATED,
        }
        terminal_at = self.terminal_at
        terminal_reason = self.terminal_reason
        terminal_source_ids = self.terminal_source_ids
        if terminal:
            if terminal_at is None:
                terminal_at = self.phase_started_at
            else:
                terminal_at = aware_timestamp(
                    terminal_at,
                    name="belief.terminal_at",
                )
            if terminal_reason is None:
                terminal_reason = (
                    "completed"
                    if self.phase is PlaybookPhase.COMPLETED
                    else "invalidated"
                )
            if not terminal_source_ids:
                fallback_sources = (
                    None
                    if self.invalidation is None
                    else self.invalidation.source_level_id,
                    self.episode_id,
                    self.setup_context_id,
                    None
                    if self.sequence is None
                    else self.sequence.setup_id,
                    self.key,
                )
                terminal_source_ids = tuple(
                    dict.fromkeys(
                        value
                        for value in fallback_sources
                        if value is not None
                    )
                )
            if (
                not isinstance(terminal_reason, str)
                or not terminal_reason
                or terminal_at != self.phase_started_at
            ):
                raise ValueError(
                    "terminal belief requires one frozen clock and reason"
                )
        elif (
            terminal_at is not None
            or terminal_reason is not None
            or terminal_source_ids
        ):
            raise ValueError(
                "nonterminal belief cannot carry terminal closure"
            )
        object.__setattr__(self, "terminal_at", terminal_at)
        object.__setattr__(self, "terminal_reason", terminal_reason)
        object.__setattr__(
            self,
            "terminal_source_ids",
            terminal_source_ids,
        )
        object.__setattr__(
            self,
            "evidence_group_scores",
            group_scores,
        )
        object.__setattr__(
            self,
            "hard_gate_results",
            hard_gates,
        )
        if not self.calibration_version:
            raise ValueError("belief calibration version is required")

    @property
    def key(self) -> str:
        return f"{self.playbook.value}:{self.direction.value}"

    @property
    def eligible(self) -> bool:
        """Whether this belief may own a current model instruction."""

        return self.phase not in {
            PlaybookPhase.INACTIVE,
            PlaybookPhase.COMPLETED,
            PlaybookPhase.INVALIDATED,
        }

    @property
    def effective_probability(self) -> float:
        """Action-facing score; terminal/inactive theses have no authority."""

        if not self.eligible:
            return 0.0
        typed = (
            self.thesis_strength,
            self.sequence_progress,
            self.location_quality,
            self.entry_readiness,
            self.delivery_quality,
        )
        return min(float(value) for value in typed if value is not None)


@dataclass(frozen=True)
class MarketBelief:
    asof: pd.Timestamp
    hypotheses: Mapping[str, HypothesisBelief]
    context_hypotheses: Mapping[str, Any] = field(default_factory=dict)
    dominant_hypothesis_id: str | None = None
    competing_hypothesis_ids: tuple[str, ...] = ()
    focus_state: Any | None = None
    cross_scale_conflicts: tuple[str, ...] = ()
    unresolved_ambiguities: tuple[str, ...] = ()
    scene_revision_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="belief.asof"))
        object.__setattr__(
            self,
            "competing_hypothesis_ids",
            tuple(self.competing_hypothesis_ids),
        )
        object.__setattr__(
            self,
            "cross_scale_conflicts",
            tuple(self.cross_scale_conflicts),
        )
        object.__setattr__(
            self,
            "unresolved_ambiguities",
            tuple(self.unresolved_ambiguities),
        )
        if any(
            key != hypothesis.key
            or hypothesis.phase_started_at > self.asof
            or any(
                evidence.observed_at > self.asof
                for evidence in (
                    *hypothesis.supporting,
                    *hypothesis.contradicting,
                )
            )
            or (
                hypothesis.invalidation is not None
                and hypothesis.invalidation.observed_at > self.asof
            )
            or any(
                target.confirmed_at > self.asof
                for target in hypothesis.deliverable_targets
            )
            or (
                hypothesis.sequence is not None
                and (
                    (
                        hypothesis.sequence.started_at is not None
                        and hypothesis.sequence.started_at > self.asof
                    )
                    or any(
                        step.observed_at is not None
                        and step.observed_at > self.asof
                        for step in hypothesis.sequence.steps
                    )
                )
            )
            or (
                hypothesis.terminal_at is not None
                and hypothesis.terminal_at > self.asof
            )
            or (
                hypothesis.draw_selection is not None
                and hypothesis.draw_selection.selected_at > self.asof
            )
            or (
                hypothesis.liquidity_route is not None
                and hypothesis.liquidity_route.selected_at > self.asof
            )
            for key, hypothesis in self.hypotheses.items()
        ):
            raise ValueError(
                "belief mapping identity or causal clock is invalid"
            )
        context_values = tuple(self.context_hypotheses.values())
        if any(
            key != getattr(value, "hypothesis_id", None)
            for key, value in self.context_hypotheses.items()
        ):
            raise ValueError("context hypothesis mapping identity is invalid")
        slot_counts: dict[tuple[Playbook, Direction], int] = {}
        for value in context_values:
            slot = (value.playbook, value.direction)
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
        if any(count > 2 for count in slot_counts.values()):
            raise ValueError(
                "a playbook-direction slot may retain at most two contexts"
            )
        context_ids = set(self.context_hypotheses)
        if (
            self.dominant_hypothesis_id is not None
            and self.dominant_hypothesis_id not in context_ids
        ):
            raise ValueError("dominant context hypothesis is not retained")
        if (
            len(self.competing_hypothesis_ids)
            != len(set(self.competing_hypothesis_ids))
            or not set(self.competing_hypothesis_ids).issubset(context_ids)
            or self.dominant_hypothesis_id in self.competing_hypothesis_ids
        ):
            raise ValueError("competing context hypothesis identities are invalid")
        if self.focus_state is not None and self.focus_state.asof != self.asof:
            raise ValueError("belief focus and belief clocks disagree")
        if self.scene_revision_id is not None and not self.scene_revision_id:
            raise ValueError("belief scene revision cannot be empty")

    def candidates(self) -> tuple[HypothesisBelief, ...]:
        """Action-facing dominant candidates; context projections stay diagnostic."""

        return tuple(self.hypotheses.values())

    def resolve_hypothesis(self, identity: str | None) -> HypothesisBelief | None:
        if identity is None:
            return None
        direct = self.hypotheses.get(identity)
        if direct is not None:
            return direct
        context = self.context_hypotheses.get(identity)
        if context is None:
            return None
        return self.hypotheses.get(
            f"{context.playbook.value}:{context.direction.value}"
        )

    def for_slot(
        self,
        playbook: Playbook,
        direction: Direction,
    ) -> tuple[HypothesisBelief, ...]:
        value = self.hypotheses.get(f"{playbook.value}:{direction.value}")
        return () if value is None else (value,)

    def ranked(self) -> list[HypothesisBelief]:
        def rank_key(
            belief: HypothesisBelief,
        ) -> tuple[float, float, float, float]:
            readiness = belief.effective_probability
            return (
                float(belief.eligible),
                readiness,
                float(belief.thesis_strength),
                -belief.uncertainty,
            )

        return sorted(
            self.hypotheses.values(),
            key=rank_key,
            reverse=True,
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


@dataclass(frozen=True)
class ActionUtility:
    action: Action
    utility: float
    components: Mapping[str, float]
    hypothesis_key: str | None
    reason: str

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.utility)):
            raise ValueError("action utility must be finite")
        if any(not math.isfinite(float(value)) for value in self.components.values()):
            raise ValueError("action utility components must be finite")


@dataclass(frozen=True)
class Decision:
    asof: pd.Timestamp
    selected_action: Action
    utilities: tuple[ActionUtility, ...]
    best_hypothesis_key: str | None
    advantage: float
    reasons: tuple[str, ...]
    plan: TradePlan | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="decision.asof"))
        if not self.utilities or not math.isfinite(float(self.advantage)):
            raise ValueError("decision requires finite compared utilities")


@dataclass(frozen=True)
class RiskAssessment:
    requested_action: Action
    final_action: Action
    passed: bool
    vetoes: tuple[VetoCode, ...]
    reasons: tuple[str, ...]
    frozen_thesis: "FrozenThesis | None" = None
    protected_stop: float | None = None


@dataclass(frozen=True)
class FrozenThesis:
    thesis_hash: str
    created_at: pd.Timestamp
    playbook: Playbook
    direction: Direction
    entry: float
    original_invalidation: StructuralLevel
    original_targets: tuple[LiquidityLevel, ...]
    deadline: pd.Timestamp
    setup_id: str | None = None
    entry_location_id: str | None = None
    entry_path_id: str | None = None
    draw_selection: DrawSelection | None = None
    range_auction: FrozenRangeAuctionContext | None = None
    liquidity_route: LiquidityRoute | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", aware_timestamp(self.created_at, name="thesis.created_at"))
        object.__setattr__(self, "deadline", aware_timestamp(self.deadline, name="thesis.deadline"))
        typed_identity = (
            self.setup_id,
            self.entry_location_id,
            self.entry_path_id,
        )
        if any(value is not None for value in typed_identity) and any(
            not isinstance(value, str) or not value
            for value in typed_identity
        ):
            raise ValueError(
                "frozen thesis typed identities must be complete "
                "non-empty text"
            )
        if self.deadline <= self.created_at or not self.original_targets:
            raise ValueError("frozen thesis requires a future deadline and targets")
        if (
            self.playbook is Playbook.FAILED_AUCTION_VALUE_RETURN
            and self.setup_id is not None
        ):
            if (
                self.range_auction is None
                or self.draw_selection is None
            ):
                raise ValueError(
                    "frozen FAVR thesis lacks its range-auction context"
                )
        elif self.range_auction is not None:
            raise ValueError("non-FAVR thesis cannot own a range auction")


@dataclass(frozen=True)
class EngineSnapshot:
    observation: MarketObservation
    belief: MarketBelief
    decision: Decision
    risk: RiskAssessment


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key.value if isinstance(key, Enum) else key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize a non-finite value")
        return value
    return value


def content_hash(value: Any) -> str:
    payload = json.dumps(to_primitive(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
