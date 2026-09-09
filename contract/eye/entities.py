"""Eye entities: the immutable per-detector state the Eye publishes.

One dataclass per market object. Each validates its own causal clocks,
identities and lifecycle transitions at construction."""
from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd

from contract.market.primitives import Direction, Timeframe, aware_timestamp, clamp
from contract.eye.vocabulary import BALANCE_CLAIM_ABANDONED, BALANCE_CLAIM_CONFIRMED, BALANCE_PRICE_TEST_KINDS, BOSLifecycle, BOSPostBreakState, BOSScope, BOS_FAILURE_REASONS, BOS_SAME_CLOCK_FAILURE_REASONS, DealingRangeLifecycle, EntryLocationLifecycle, FVGQualification, FairValueGapLifecycle, GROUP5_SAME_CLOCK_REACCEPTANCE_FAILURE_REASONS, LiquidityInventoryLifecycle, LiquidityPoolLifecycle, ManipulationLifecycle, ManipulationSourceDispositionKind, ORDER_BLOCK_FUNNEL_STAGES, OrderBlockAttemptOutcome, OrderBlockLifecycle, RANGE_AUCTION_HARD_BOUNDARY_REASONS, RANGE_MATURITY_GATE_NAMES, RANGE_PAIR_FUNNEL_COUNTS, ReacceptanceLifecycle, STRUCTURE_BREAK_FAILURE_REASON, STRUCTURE_FORMATION_FAILURE_REASON, SUPPORT_RESISTANCE_RETIREMENT_REASON, StructureLifecycle, SupportResistanceLifecycle, SwingLifecycle, SwingRank, SwingRelation, SwingSide
from eyes.core.foundation_registry import FOUNDATION_VERSION


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
    # Legacy ``magnitude_atr`` is the distance from the prior same-side
    # swing.  Local pivot prominence is a distinct, preregistered feature.
    prominence_atr: float = 0.0
    confirmation_delay_bars: int = 0
    nesting_depth: int = 0
    semantic_rank: SwingRank = SwingRank.UNRESOLVED
    age_bars: int = 0
    broken_at: pd.Timestamp | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_rank", SwingRank(self.semantic_rank))
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
            SwingLifecycle.FORMATION_FAILED: {
                "right_side_invalidated",
                "insufficient_prominence",
            },
            SwingLifecycle.BROKEN: "close_beyond_swing",
        }.get(self.lifecycle)
        if expected_reason is None and self.failure_reason is not None:
            raise ValueError(
                "open or confirmed swing cannot carry a failure reason"
            )
        if expected_reason is not None and (
            self.failure_reason not in expected_reason
            if isinstance(expected_reason, set)
            else self.failure_reason != expected_reason
        ):
            raise ValueError(
                "terminal swing requires its registered failure reason"
            )
        if (
            self.age_bars < 0
            or type(self.confirmation_delay_bars) is not int
            or self.confirmation_delay_bars < 0
            or type(self.nesting_depth) is not int
            or self.nesting_depth < 0
        ):
            raise ValueError("swing age cannot be negative")
        if not all(
            math.isfinite(float(value))
            for value in (
                self.delta_points,
                self.magnitude_atr,
                self.prominence_atr,
            )
        ) or self.magnitude_atr < 0 or self.prominence_atr < 0:
            raise ValueError("swing magnitude is invalid")


@dataclass(frozen=True)
class StructuralLegState:
    """One replay-stable movement between opposite confirmed swings."""

    leg_id: str
    timeframe: Timeframe
    direction: Direction
    start_swing_id: str
    end_swing_id: str
    start_event_time: pd.Timestamp
    end_event_time: pd.Timestamp
    known_at: pd.Timestamp
    start_price: float
    end_price: float
    start_close: float
    end_close: float
    amplitude_points: float
    amplitude_atr: float
    duration_bars: int
    duration_minutes: int
    efficiency: float
    max_retracement_points: float
    max_retracement_atr: float
    # Named for what it measures: how long the leg's own bar path is, which
    # is independent of any role a swing later takes on.
    path_class: SwingRank = SwingRank.INTERNAL
    source_swing_ids: tuple[str, str] = ("", "")
    # Foundation-v2 path metrics are additive so frozen v1.2 event payloads
    # remain readable.  A leg produced by the foundation builder populates
    # every optional field; ``None``/empty values identify historical v1.2
    # compatibility objects rather than silently reconstructed evidence.
    amplitude_ticks: int | None = None
    atr_at_leg_start: float | None = None
    duration_seconds: int | None = None
    close_efficiency: float | None = None
    extreme_path_efficiency: float | None = None
    close_mae_points: float | None = None
    close_mae_atr: float | None = None
    wick_mae_points: float | None = None
    wick_mae_atr: float | None = None
    path_candle_ids: tuple[str, ...] = ()
    atr_source_candle_ids: tuple[str, ...] = ()
    # Densified no-trade minutes inside the path.  Zero means every admitted
    # bar carried real price discovery; a consumer that needs the historical
    # real-only guarantee filters on this being zero.
    synthetic_path_minutes: int = 0
    foundation_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("start_event_time", "end_event_time", "known_at"):
            object.__setattr__(
                self,
                name,
                aware_timestamp(getattr(self, name), name=f"leg.{name}"),
            )
        object.__setattr__(self, "path_class", SwingRank(self.path_class))
        source_ids = tuple(self.source_swing_ids)
        object.__setattr__(self, "source_swing_ids", source_ids)
        path_candle_ids = tuple(self.path_candle_ids)
        object.__setattr__(self, "path_candle_ids", path_candle_ids)
        atr_source_candle_ids = tuple(self.atr_source_candle_ids)
        object.__setattr__(
            self,
            "atr_source_candle_ids",
            atr_source_candle_ids,
        )
        duration_seconds = self.duration_seconds
        if duration_seconds is None:
            duration_seconds = int(
                (self.end_event_time - self.start_event_time).total_seconds()
            )
            object.__setattr__(self, "duration_seconds", duration_seconds)
        close_efficiency = self.close_efficiency
        if close_efficiency is None:
            close_efficiency = float(self.efficiency)
            object.__setattr__(self, "close_efficiency", close_efficiency)
        continuous = (
            self.start_price,
            self.end_price,
            self.start_close,
            self.end_close,
            self.amplitude_points,
            self.amplitude_atr,
            self.efficiency,
            self.max_retracement_points,
            self.max_retracement_atr,
            close_efficiency,
        )
        expected_direction = (
            Direction.LONG
            if self.end_price > self.start_price
            else Direction.SHORT
        )
        if (
            not self.leg_id
            or not self.start_swing_id
            or not self.end_swing_id
            or self.start_swing_id == self.end_swing_id
            or source_ids != (self.start_swing_id, self.end_swing_id)
            or self.start_event_time >= self.end_event_time
            or self.known_at < self.end_event_time
            or self.direction is not expected_direction
            or any(not math.isfinite(float(value)) for value in continuous)
            or min(
                self.start_price,
                self.end_price,
                self.start_close,
                self.end_close,
            )
            <= 0.0
            or self.amplitude_points <= 0.0
            or self.amplitude_atr < 0.0
            or not 0.0 <= self.efficiency <= 1.0
            or self.max_retracement_points < 0.0
            or self.max_retracement_atr < 0.0
            or type(self.duration_bars) is not int
            or self.duration_bars < 2
            or type(self.duration_minutes) is not int
            or self.duration_minutes <= 0
            or type(duration_seconds) is not int
            or duration_seconds <= 0
            or duration_seconds
            != int(
                (self.end_event_time - self.start_event_time).total_seconds()
            )
            or not 0.0 <= float(close_efficiency) <= 1.0
            or not math.isclose(
                float(close_efficiency),
                float(self.efficiency),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("structural leg geometry or provenance is invalid")
        optional_non_negative = (
            self.extreme_path_efficiency,
            self.close_mae_points,
            self.close_mae_atr,
            self.wick_mae_points,
            self.wick_mae_atr,
        )
        if (
            self.amplitude_ticks is not None
            and (
                type(self.amplitude_ticks) is not int
                or self.amplitude_ticks <= 0
            )
        ) or (
            self.atr_at_leg_start is not None
            and (
                not math.isfinite(float(self.atr_at_leg_start))
                or float(self.atr_at_leg_start) <= 0.0
            )
        ) or any(
            value is not None
            and (
                not math.isfinite(float(value))
                or float(value) < 0.0
            )
            for value in optional_non_negative
        ) or (
            self.extreme_path_efficiency is not None
            and float(self.extreme_path_efficiency) > 1.0
        ):
            raise ValueError("structural leg foundation metrics are invalid")
        if (
            path_candle_ids
            and (
                len(path_candle_ids) != self.duration_bars
                or len(path_candle_ids) != len(set(path_candle_ids))
                or any(
                    not isinstance(value, str) or not value
                    for value in path_candle_ids
                )
            )
        ):
            raise ValueError("structural leg path ancestry is invalid")
        if atr_source_candle_ids and (
            len(atr_source_candle_ids) != 14
            or len(atr_source_candle_ids)
            != len(set(atr_source_candle_ids))
            or any(
                not isinstance(value, str) or not value
                for value in atr_source_candle_ids
            )
        ):
            raise ValueError("structural leg ATR ancestry is invalid")
        foundation_fields = (
            self.amplitude_ticks,
            self.atr_at_leg_start,
            self.extreme_path_efficiency,
            self.close_mae_points,
            self.close_mae_atr,
            self.wick_mae_points,
            self.wick_mae_atr,
        )
        has_foundation_metrics = any(
            value is not None for value in foundation_fields
        ) or bool(path_candle_ids) or bool(
            atr_source_candle_ids
        ) or self.foundation_version is not None
        if has_foundation_metrics and (
            any(value is None for value in foundation_fields)
            or not path_candle_ids
            or len(atr_source_candle_ids) != 14
            or self.foundation_version != FOUNDATION_VERSION
        ):
            raise ValueError(
                "structural leg foundation metrics must be complete and versioned"
            )
        if self.atr_at_leg_start is not None:
            atr0 = float(self.atr_at_leg_start)
            normalized_pairs = (
                (self.amplitude_points, self.amplitude_atr),
                (self.max_retracement_points, self.max_retracement_atr),
                (self.close_mae_points, self.close_mae_atr),
                (self.wick_mae_points, self.wick_mae_atr),
            )
            if any(
                points is None
                or normalized is None
                or not math.isclose(
                    float(normalized),
                    float(points) / atr0,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for points, normalized in normalized_pairs
            ):
                raise ValueError(
                    "structural leg ATR metrics do not use ATR at leg start"
                )


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
    balance_confirmed_at: pd.Timestamp | None
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
    # Balance evidence, kept strictly separate from the structural touch counts
    # above.  One generation is one continuous visit to a boundary's tolerance
    # band: price has to leave the band and come back for the next to open, so
    # a bar-by-bar hug of the level counts once, not once per bar.
    balance_lower_test_generations: int = 0
    balance_upper_test_generations: int = 0
    balance_lower_test_kinds: tuple[str, ...] = ()
    balance_upper_test_kinds: tuple[str, ...] = ()

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
            "balance_confirmed_at",
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
        for side in ("lower", "upper"):
            kinds = tuple(getattr(self, f"balance_{side}_test_kinds"))
            object.__setattr__(self, f"balance_{side}_test_kinds", kinds)
            generations = getattr(self, f"balance_{side}_test_generations")
            if (
                type(generations) is not int
                or generations < 0
                or len(kinds) != generations
                or any(kind not in BALANCE_PRICE_TEST_KINDS for kind in kinds)
            ):
                raise ValueError("dealing-range balance price tests are invalid")
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
        if self.balance_confirmed_at is not None and (
            self.balance_confirmed_at < self.formed_at
            or self.balance_confirmed_at > self.last_updated_at
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
        ):
            raise ValueError("dealing-range break clock is invalid")
        if self.transition_reason == "":
            raise ValueError("dealing-range transition reason cannot be empty")
        if self.lifecycle is DealingRangeLifecycle.ACTIVE:
            # A candidate that ran out of room to prove balance keeps its
            # structural interval: the balance claim ended, the interval did
            # not.  Such a range legitimately restarts its state clock and
            # carries the deadline's bar count.  The same is true once the
            # claim is confirmed -- neither verdict is a state of the range.
            claim_settled = self.transition_reason in {
                BALANCE_CLAIM_ABANDONED,
                BALANCE_CLAIM_CONFIRMED,
            }
            if (
                self.broken_at is not None
                or (
                    self.state_started_at != self.formed_at
                    if not claim_settled
                    else self.state_started_at < self.formed_at
                )
                or (not claim_settled and self.candidate_real_h1_bars >= 24)
                or self.transition_reason
                not in {
                    None,
                    "source_pair_selected",
                    BALANCE_CLAIM_ABANDONED,
                    BALANCE_CLAIM_CONFIRMED,
                }
            ):
                raise ValueError("active dealing-range lifecycle is inconsistent")
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
                            *RANGE_AUCTION_HARD_BOUNDARY_REASONS,
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
    midpoint_touched_at: pd.Timestamp | None = None
    mitigated_at: pd.Timestamp | None = None
    invalidated_at: pd.Timestamp | None = None
    expired_at: pd.Timestamp | None = None
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
            "midpoint_touched_at",
            "mitigated_at",
            "invalidated_at",
            "expired_at",
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
                self.midpoint_touched_at,
                self.mitigated_at,
                self.invalidated_at,
                self.expired_at,
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
            and self.midpoint_touched_at is not None
            and self.partial_at > self.midpoint_touched_at
        ):
            raise ValueError("FVG midpoint touch cannot predate partial fill")
        if (
            self.midpoint_touched_at is not None
            and self.mitigated_at is not None
            and self.midpoint_touched_at > self.mitigated_at
        ):
            raise ValueError("FVG mitigation cannot predate midpoint touch")
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
                        self.midpoint_touched_at,
                        self.mitigated_at,
                        self.invalidated_at,
                        self.expired_at,
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
                or self.expired_at is not None
                or (
                    (raw_fill >= 0.5)
                    != (self.midpoint_touched_at is not None)
                )
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
                or self.expired_at is not None
                or self.midpoint_touched_at is None
                or not self.transition_reason
            ):
                raise ValueError("mitigated FVG lifecycle is inconsistent")
        elif self.lifecycle is FairValueGapLifecycle.INVALIDATED and (
            self.invalidated_at is None
            or self.state_started_at != self.invalidated_at
            or self.last_updated_at != self.invalidated_at
            or self.mitigated_at is not None
            or self.expired_at is not None
            or not self.transition_reason
        ):
            raise ValueError(
                "invalidated FVG requires its clock and reason"
            )
        elif self.lifecycle is FairValueGapLifecycle.EXPIRED and (
            self.expired_at is None
            or self.state_started_at != self.expired_at
            or self.last_updated_at != self.expired_at
            or self.mitigated_at is not None
            or self.invalidated_at is not None
            or not self.transition_reason
        ):
            raise ValueError("expired FVG lifecycle is inconsistent")


@dataclass(frozen=True)
class BaseOriginCoreState:
    """The frozen opposite candles an impulse left behind, and nothing else.

    Published the moment the impulse locks them.  A core carries no order-block
    reading and no break: it is true whether or not the impulse went on to
    displace and break structure, which is exactly why it can be counted
    without hindsight.
    """

    base_origin_core_id: str
    protocol_hash: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    source_displacement_id: str
    source_displacement_transition_id: str
    anchor_candle_id: str
    anchor_candle_ids: tuple[str, ...]
    anchor_start: pd.Timestamp
    anchor_end: pd.Timestamp
    lower_bound: float
    upper_bound: float
    body_lower_bound: float
    body_upper_bound: float
    midpoint: float
    observed_at: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self, "anchor_candle_ids", tuple(self.anchor_candle_ids)
        )
        if (
            not self.base_origin_core_id
            or not self.protocol_hash
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or not self.source_displacement_id
            or not self.source_displacement_transition_id
            or not self.anchor_candle_id
            or self.anchor_candle_id not in self.anchor_candle_ids
            or len(set(self.anchor_candle_ids)) != len(self.anchor_candle_ids)
            or self.anchor_start >= self.anchor_end
            # The core may never cite a bar the clock has not reached.
            or self.anchor_end > self.observed_at
            or not 0.0 < self.lower_bound <= self.upper_bound
            or not self.lower_bound <= self.body_lower_bound
            or not self.body_upper_bound <= self.upper_bound
            or self.body_lower_bound > self.body_upper_bound
            or self.midpoint != (self.lower_bound + self.upper_bound) / 2.0
        ):
            raise ValueError("base origin core state is invalid")


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
    # The geometry this zone qualified, published earlier by the impulse.
    base_origin_core_id: str | None = None

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
    source_zone_detector_protocol_hash: str
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
                self.source_zone_detector_protocol_hash,
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
class ReacceptanceState:
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
    lifecycle: ReacceptanceLifecycle
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
                ReacceptanceLifecycle,
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
            raise ValueError("reacceptance identity is invalid")
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
            self.lifecycle is ReacceptanceLifecycle.FAILED
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
            raise ValueError("reacceptance clocks are invalid")
        for value in (
            self.reclaim_margin_atr,
            self.hold_margin_atr,
            self.strength,
        ):
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError("reacceptance strength is invalid")
        if self.lifecycle is ReacceptanceLifecycle.LEFT:
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
        elif self.lifecycle is ReacceptanceLifecycle.RECLAIMED:
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
        elif self.lifecycle is ReacceptanceLifecycle.HELD:
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
        elif self.lifecycle is ReacceptanceLifecycle.FAILED:
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
                "reacceptance transition reason is required"
            )


# Cold import/pickle compatibility for pre-rename artifacts.  New pickles use
# the physical class name because this is an alias, not a wrapper type.
QualifiedReacceptanceState = ReacceptanceState


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


__all__ = [
    "BaseOriginCoreState",
    "BreakOfStructureState",
    "CandleStructureState",
    "DealingRangeState",
    "EntryLocationState",
    "FairValueGapState",
    "LiquidityInventoryItem",
    "LiquidityPoolState",
    "ManipulationSourceDisposition",
    "ManipulationState",
    "OrderBlockFunnelSnapshot",
    "OrderBlockState",
    "QualifiedReacceptanceState",
    "RangeFormationFunnelSnapshot",
    "ReacceptanceState",
    "StructuralLegState",
    "StructureSequenceState",
    "SupportResistanceState",
    "SwingPoint",
]
