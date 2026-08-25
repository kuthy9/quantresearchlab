"""Shared immutable contracts for the continuous SMC engine."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from .dol_probability import DOLProbabilityResult
from .dol_ranking import DOLRankingResult
from .foundation_registry import FOUNDATION_VERSION
from .path_belief import PathBeliefUpdateRecord, PathCompetitionSetState

if TYPE_CHECKING:
    from .market_state import MarketSnapshot


SMC_SEMANTIC_VERSION = "smc_semantics_v1.2"
MARKET_OBSERVATION_SCHEMA_VERSION = 3
ENGINE_SNAPSHOT_SCHEMA_VERSION = 2
NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION = 2
INTERACTION_UPDATE_SCHEMA_VERSION = 1


def _exact_dataclass_pickle_state(
    value: Any,
    *,
    schema_version: int,
    label: str,
) -> Mapping[str, Any]:
    names = tuple(item.name for item in fields(value))
    if set(value.__dict__) != set(names):
        raise ValueError(f"{label} pickle state is not exact")
    return {
        "schema_version": schema_version,
        "fields": tuple((name, getattr(value, name)) for name in names),
    }


def _restore_exact_dataclass_pickle_state(
    value: Any,
    state: Mapping[str, Any],
    *,
    schema_version: int,
    label: str,
) -> None:
    names = tuple(item.name for item in fields(value))
    serialized = state.get("fields") if isinstance(state, Mapping) else None
    if (
        not isinstance(state, Mapping)
        or set(state) != {"schema_version", "fields"}
        or state.get("schema_version") != schema_version
        or not isinstance(serialized, tuple)
        or len(serialized) != len(names)
        or any(
            not isinstance(item, tuple) or len(item) != 2
            for item in serialized
        )
        or tuple(item[0] for item in serialized) != names
    ):
        raise ValueError(f"{label} pickle schema changed")
    for name, item in serialized:
        object.__setattr__(value, name, item)


def _deep_freeze(value: Any) -> Any:
    """Return an audit-safe immutable copy of a semantic payload."""

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted(
                (_deep_freeze(item) for item in value),
                key=repr,
            )
        )
    return value


def _strict_payload_equal(
    left: Any,
    right: Any,
) -> bool:
    """Compare evidence without Python's cross-type equality coercions.

    ``dict.__eq__`` considers values such as ``True`` and ``1`` equal.  That
    is not a safe basis for aliasing two canonical payloads because replacing
    one with the other changes its primitive type and therefore its evidence
    bytes.  Only recursively type-identical, supported primitive structures
    may share one frozen mapping; unfamiliar objects conservatively remain
    separate.
    """

    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        if all(type(key) is str for key in left) and all(
            type(key) is str for key in right
        ):
            if set(left) != set(right):
                return False
            return all(
                _strict_payload_equal(left[key], right[key])
                for key in left
            )
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            for index, (right_key, right_value) in enumerate(unmatched):
                if not _strict_payload_equal(left_key, right_key):
                    continue
                if not _strict_payload_equal(left_value, right_value):
                    return False
                unmatched.pop(index)
                break
            else:
                return False
        return not unmatched
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _strict_payload_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, Enum):
        return _strict_payload_equal(left.value, right.value)
    if isinstance(left, pd.Timestamp):
        return left.isoformat() == right.isoformat()
    if isinstance(left, Path):
        return str(left) == str(right)
    if isinstance(left, float):
        return left.hex() == right.hex()
    if isinstance(left, (str, bytes, int, bool, type(None))):
        return left == right
    return False


class FrozenDict(dict):
    """A pickle/JSON-friendly mapping that rejects post-construction edits.

    ``dataclass(frozen=True)`` protects only the event attributes themselves;
    a normal ``dict`` stored in ``details`` could still be mutated and would
    silently rewrite history.  This small dict subclass retains compatibility
    with existing serializers and readers while closing that hole.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        dict.__init__(
            self,
            {
                key: _deep_freeze(item)
                for key, item in values.items()
            },
        )

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("semantic event payload is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "FrozenDict":
        return self

    def __reduce__(self) -> tuple[type["FrozenDict"], tuple[dict[Any, Any]]]:
        return FrozenDict, (dict(self),)


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


class MarketMode(str, Enum):
    """Small, descriptive state space for the market-wide context."""

    DIRECTIONAL = "directional"
    BALANCED = "balanced"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"


class ScaleRelation(str, Enum):
    """How one completed scale relates to the current structural authority."""

    ALIGNED = "aligned"
    NORMAL_PULLBACK = "normal_pullback"
    MATERIAL_OPPOSITION = "material_opposition"
    UNKNOWN = "unknown"


class GlobalConflictRole(str, Enum):
    """Deterministic market-level meaning of one cross-scale opposition."""

    LOCAL_COUNTERTREND_DELIVERY = "local_countertrend_delivery"
    AUTHORITY_TRANSITION_CANDIDATE = "authority_transition_candidate"
    AUTHORITY_INVALIDATION = "authority_invalidation"


class HypothesisConflictRelation(str, Enum):
    """Direction-aware meaning of global evidence for one hypothesis."""

    CHALLENGES_INCUMBENT = "challenges_incumbent"
    SUPPORTS_CHALLENGER = "supports_challenger"
    LOCAL_COUNTERTREND = "local_countertrend"
    UNRELATED = "unrelated"


class Action(str, Enum):
    ENTER = "enter"
    WAIT = "wait"
    HOLD = "hold"
    PROTECT = "protect"
    EXIT = "exit"
    ABSTAIN = "abstain"


class EventKind(str, Enum):
    # Canonical preregistered semantic events.  The older ``*_STATE`` kinds
    # below remain compatibility lifecycle transport for existing reducers.
    MARKET_EPOCH_RESET = "market_epoch_reset"
    BAR_COMPLETED = "bar_completed"
    SWING_CONFIRMED = "swing_confirmed"
    STRUCTURAL_LEG_CREATED = "structural_leg_created"
    LIQUIDITY_LEVEL_CREATED = "liquidity_level_created"
    LEVEL_TOUCHED = "level_touched"
    LEVEL_PENETRATED = "level_penetrated"
    SWEEP_CONFIRMED = "sweep_confirmed"
    ACCEPTANCE_CONFIRMED = "acceptance_confirmed"
    DISPLACEMENT_OBSERVED = "displacement_observed"
    FVG_CREATED = "fvg_created"
    FVG_TOUCHED = "fvg_touched"
    FVG_PARTIALLY_FILLED = "fvg_partially_filled"
    FVG_MIDPOINT_TOUCHED = "fvg_midpoint_touched"
    FVG_FULLY_FILLED = "fvg_fully_filled"
    FVG_INVALIDATED = "fvg_invalidated"
    FVG_EXPIRED = "fvg_expired"
    RAW_BOUNDARY_BREAK = "raw_boundary_break"
    STRUCTURE_DIRECTION_CONFIRMED = "structure_direction_confirmed"
    QUALIFIED_BOS = "qualified_bos"
    PROTECTED_SWING_ASSIGNED = "protected_swing_assigned"
    MSS_CORE_CONFIRMED = "mss_core_confirmed"
    DEALING_RANGE_CREATED = "dealing_range_created"
    DEALING_RANGE_ACTIVATED = "dealing_range_activated"
    DEALING_RANGE_EXTENDED = "dealing_range_extended"
    DEALING_RANGE_INVALIDATED = "dealing_range_invalidated"
    DEALING_RANGE_REPLACED = "dealing_range_replaced"
    DELIVERY_PHASE_CHANGED = "delivery_phase_changed"
    ORIGIN_ZONE_CREATED = "origin_zone_created"
    ORIGIN_ZONE_TOUCHED = "origin_zone_touched"
    ORIGIN_ZONE_MITIGATED = "origin_zone_mitigated"
    ORIGIN_ZONE_INVALIDATED = "origin_zone_invalidated"
    TIMEFRAME_STATE_CHANGED = "timeframe_state_changed"
    RELATION_STATE_CHANGED = "relation_state_changed"
    SESSION_STATE_CHANGED = "session_state_changed"
    # Technical, rebuildable transport for the versioned foundation
    # projection.  This is not a new canonical SMC market-language term.
    FOUNDATION_STATE_CHANGED = "foundation_state_changed"
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


class EventOrigin(str, Enum):
    """Authority class for one immutable market-event record."""

    NORMALIZED_DATA = "normalized_data"
    SEMANTIC_ATOMIC = "semantic_atomic"
    STATE_PROJECTION = "state_projection"
    LEGACY_TRANSPORT = "legacy_transport"


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


class SwingRank(str, Enum):
    UNRESOLVED = "unresolved"
    MICRO = "micro"
    INTERNAL = "internal"
    STRUCTURAL = "structural"
    EXTERNAL = "external"


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
    EXPIRED = "expired"


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


RANGE_AUCTION_HARD_BOUNDARY_REASONS = frozenset(
    {
        "data_gap_reset",
        "contract_change_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
GROUP5_HARD_BOUNDARY_REASONS = RANGE_AUCTION_HARD_BOUNDARY_REASONS
GROUP5_CONTEXT_KINDS = frozenset(
    {"zone_return", "pool_reversal"}
)
INTERACTION_PHYSICAL_PATH_STEP_KINDS = frozenset(
    {
        "zone_visible",
        "departure_confirmed",
        "first_pullback",
        "wick_rejection",
        "reference_left",
        "reference_reclaimed",
        "reacceptance_held",
        "reacceptance_failed",
        "micro_break_observed",
        "location_left",
        "pool_swept",
        "opposite_displacement",
        "opposite_displacement_ambiguous",
        "accepted_outside",
    }
)
_INTERACTION_PHYSICAL_PATH_STEP_REASONS = {
    "zone_visible": frozenset({"typed_entry_zone_registered"}),
    "departure_confirmed": frozenset(
        {
            "formation_close_on_delivery_side",
            "later_close_on_delivery_side",
        }
    ),
    "first_pullback": frozenset(
        {"crossed_near_edge", "gap_opened_inside"}
    ),
    "wick_rejection": frozenset(
        {
            "gap_inside_recovery",
            "same_bar_wick_rejection",
            "later_zone_rejection",
        }
    ),
    "reference_left": frozenset({"completed_close_on_adverse_side"}),
    "reference_reclaimed": frozenset(
        {"strict_completed_close_reclaim"}
    ),
    "reacceptance_held": frozenset(
        {"later_real_completed_hold", "group4_reentry_held"}
    ),
    "reacceptance_failed": frozenset(
        {
            "close_beyond_failure_boundary",
            "reclaim_lost_before_hold",
            "source_invalidated",
            "accepted_outside",
            "context_closed_before_hold",
        }
    ),
    "micro_break_observed": frozenset(
        {
            "confirmed_m1_break_at_anchor_clock",
            "first_strictly_later_confirmed_m1_break",
        }
    ),
    "location_left": frozenset(
        {
            "fvg_invalidated",
            "order_block_failed",
            "close_beyond_far_edge",
            "gap_through_frozen_zone",
        }
    ),
    "pool_swept": frozenset({"typed_pool_manipulation_swept"}),
    "opposite_displacement": frozenset(
        {"opposite_displacement_after_reacceptance"}
    ),
    "opposite_displacement_ambiguous": frozenset(
        {"multiple_opposite_displacements_same_clock"}
    ),
    "accepted_outside": frozenset({"group4_accepted_outside"}),
}
if frozenset(_INTERACTION_PHYSICAL_PATH_STEP_REASONS) != (
    INTERACTION_PHYSICAL_PATH_STEP_KINDS
):
    raise RuntimeError("interaction physical step reason registry is incomplete")
_INTERACTION_PHYSICAL_PATH_REASONS = frozenset(
    {
        "context_registered",
        "qualified_reacceptance_held",
        "zone_rejection_observed",
        "location_left",
        "reacceptance_failed",
        "first_strict_micro_break_observed",
        "accepted_outside",
        "opposite_displacement_ambiguous_same_clock",
        "manipulation_resolution_deadline",
        *GROUP5_HARD_BOUNDARY_REASONS,
    }
)
_BRAIN_INTERPRETED_PATH_STEP_KINDS = frozenset(
    {
        "micro_bos_simultaneous",
        "micro_bos_confirmed",
        "micro_bos_opposed",
        "micro_bos_ambiguous",
    }
)
# Frozen legacy PathSequence readers still need the interpreted vocabulary;
# canonical InteractionUpdate admission below accepts only the physical set.
GROUP5_PATH_STEP_KINDS = (
    INTERACTION_PHYSICAL_PATH_STEP_KINDS
    | _BRAIN_INTERPRETED_PATH_STEP_KINDS
)
_BRAIN_INTERPRETED_PATH_REASONS = frozenset(
    {
        "micro_bos_aligned",
        "micro_bos_opposed",
        "micro_bos_ambiguous_same_clock",
        "pool_reversal_sequence_observed",
    }
)


NEUTRAL_MARKET_STATE_SCHEMA_VERSION = 2
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


def _finite_decimal(value: Any, *, name: str) -> Decimal:
    """Parse one numeric contract value without binary-float rounding."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _tick_size_decimal(tick_size: Any) -> Decimal:
    parsed = _finite_decimal(tick_size, name="tick_size")
    if parsed <= 0:
        raise ValueError("tick_size must be positive")
    return parsed


def price_to_ticks(
    price: Any,
    tick_size: Any,
    *,
    name: str = "price",
) -> int:
    """Return the exact integer grid coordinate for one vendor price.

    Decimal text parsing plus integer-ratio arithmetic is intentional:
    admission never depends on binary floats, banker's rounding, or the
    process-wide Decimal precision context.
    """

    parsed_price = _finite_decimal(price, name=name)
    parsed_tick = _tick_size_decimal(tick_size)
    price_numerator, price_denominator = parsed_price.as_integer_ratio()
    tick_numerator, tick_denominator = parsed_tick.as_integer_ratio()
    coordinate_numerator = price_numerator * tick_denominator
    coordinate_denominator = price_denominator * tick_numerator
    integral, remainder = divmod(
        coordinate_numerator,
        coordinate_denominator,
    )
    if remainder:
        raise ValueError(
            f"{name} is off-grid for tick_size {parsed_tick}"
        )
    return integral


def ticks_to_price(
    ticks: int,
    tick_size: Any,
    *,
    name: str = "ticks",
) -> float:
    """Return the canonical float projection of an integer tick coordinate."""

    if type(ticks) is not int:
        raise ValueError(f"{name} must be an integer")
    tick_numerator, tick_denominator = (
        _tick_size_decimal(tick_size).as_integer_ratio()
    )
    return (ticks * tick_numerator) / tick_denominator


def ohlc_to_ticks(
    open_price: Any,
    high_price: Any,
    low_price: Any,
    close_price: Any,
    tick_size: Any,
) -> tuple[int, int, int, int]:
    """Validate vendor/detector OHLC and return its integer representation."""

    output = (
        price_to_ticks(open_price, tick_size, name="open"),
        price_to_ticks(high_price, tick_size, name="high"),
        price_to_ticks(low_price, tick_size, name="low"),
        price_to_ticks(close_price, tick_size, name="close"),
    )
    open_ticks, high_ticks, low_ticks, close_ticks = output
    if (
        high_ticks < max(open_ticks, close_ticks)
        or low_ticks > min(open_ticks, close_ticks)
        or high_ticks < low_ticks
    ):
        raise ValueError("integer OHLC geometry is invalid")
    return output


def _validate_normalized_ohlc(
    *,
    values: tuple[Any, Any, Any, Any],
    price_tick_size: float | None,
    normalized_ohlc_ticks: tuple[int, int, int, int] | None,
) -> tuple[float | None, tuple[int, int, int, int] | None]:
    if price_tick_size is None:
        if normalized_ohlc_ticks is not None:
            raise ValueError(
                "normalized OHLC ticks require their price tick size"
            )
        return None, None
    canonical_tick = float(_tick_size_decimal(price_tick_size))
    expected = ohlc_to_ticks(*values, canonical_tick)
    if normalized_ohlc_ticks is None:
        return canonical_tick, expected
    supplied = tuple(normalized_ohlc_ticks)
    if (
        len(supplied) != 4
        or any(type(value) is not int for value in supplied)
        or supplied != expected
    ):
        raise ValueError("stored normalized OHLC ticks disagree with prices")
    return canonical_tick, supplied


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
    price_tick_size: float | None = field(default=None, compare=False)
    normalized_ohlc_ticks: tuple[int, int, int, int] | None = field(
        default=None,
        compare=False,
    )

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
        price_tick_size, normalized_ticks = _validate_normalized_ohlc(
            values=(self.open, self.high, self.low, self.close),
            price_tick_size=self.price_tick_size,
            normalized_ohlc_ticks=self.normalized_ohlc_ticks,
        )
        object.__setattr__(self, "price_tick_size", price_tick_size)
        object.__setattr__(self, "normalized_ohlc_ticks", normalized_ticks)

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(1, unit="min")

    @property
    def open_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[0]

    @property
    def high_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[1]

    @property
    def low_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[2]

    @property
    def close_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[3]

    def on_price_grid(self, tick_size: float) -> "Bar":
        """Return this immutable bar with exact normalized tick coordinates."""

        requested = _tick_size_decimal(tick_size)
        if self.price_tick_size is not None:
            if _tick_size_decimal(self.price_tick_size) != requested:
                raise ValueError(
                    "bar price grid disagrees with reader tick size"
                )
            return self
        return replace(
            self,
            price_tick_size=float(requested),
            normalized_ohlc_ticks=None,
        )


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
    price_tick_size: float | None = field(default=None, compare=False)
    normalized_ohlc_ticks: tuple[int, int, int, int] | None = field(
        default=None,
        compare=False,
    )

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
        price_tick_size, normalized_ticks = _validate_normalized_ohlc(
            values=(self.open, self.high, self.low, self.close),
            price_tick_size=self.price_tick_size,
            normalized_ohlc_ticks=self.normalized_ohlc_ticks,
        )
        object.__setattr__(self, "price_tick_size", price_tick_size)
        object.__setattr__(self, "normalized_ohlc_ticks", normalized_ticks)

    @property
    def real_completed(self) -> bool:
        return bool(self.complete and self.synthetic_minutes == 0)

    @property
    def open_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[0]

    @property
    def high_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[1]

    @property
    def low_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[2]

    @property
    def close_ticks(self) -> int | None:
        return None if self.normalized_ohlc_ticks is None else self.normalized_ohlc_ticks[3]

    def ohlc_ticks_for(self, tick_size: float) -> tuple[int, int, int, int]:
        """Read stored ticks on the same grid or validate a direct test candle."""

        requested = _tick_size_decimal(tick_size)
        if self.price_tick_size is not None:
            if _tick_size_decimal(self.price_tick_size) != requested:
                raise ValueError("candle price grid disagrees with detector tick size")
            if self.normalized_ohlc_ticks is None:
                raise AssertionError("normalized candle lost its integer OHLC")
            return self.normalized_ohlc_ticks
        return ohlc_to_ticks(
            self.open,
            self.high,
            self.low,
            self.close,
            float(requested),
        )


def candle_identity(candle: Candle, *, tick_size: float) -> str:
    """Return one shared candle identity for every semantic reducer."""

    ticks = candle.ohlc_ticks_for(tick_size)

    parts = (
        "candle-v1",
        candle.timeframe.value,
        candle.start.isoformat(),
        candle.end.isoformat(),
        *(str(value) for value in ticks),
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
    rank: SwingRank = SwingRank.INTERNAL
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
    foundation_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("start_event_time", "end_event_time", "known_at"):
            object.__setattr__(
                self,
                name,
                aware_timestamp(getattr(self, name), name=f"leg.{name}"),
            )
        object.__setattr__(self, "rank", SwingRank(self.rank))
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
class MicroBreakFact:
    """Exact confirmed M1 break bound to an interaction clock.

    This is an Eye fact.  It deliberately carries neither setup
    qualification nor an aligned/opposed outcome; those are Brain-owned
    interpretations of the raw direction and ordering fields below.
    ``reference_id`` retains the frozen Group 5 identity preimage.
    """

    reference_id: str
    protocol_hash: str
    context_kind: str
    context_id: str
    context_direction: Direction
    anchor_at: pd.Timestamp
    bos_id: str
    bos_direction: Direction
    target_swing_id: str
    scope: BOSScope
    pending_at: pd.Timestamp
    resolved_at: pd.Timestamp
    relation: str
    strength: float

    def __post_init__(self) -> None:
        if (
            not self.reference_id
            or not self.protocol_hash
            or self.context_kind not in GROUP5_CONTEXT_KINDS
            or not self.context_id
            or not isinstance(self.context_direction, Direction)
            or not self.bos_id
            or not isinstance(self.bos_direction, Direction)
            or not self.target_swing_id
            or not isinstance(self.scope, BOSScope)
            or self.relation not in {"strictly_after", "same_clock_unknown"}
            or not math.isfinite(float(self.strength))
            or not 0.0 <= self.strength <= 1.0
        ):
            raise ValueError("micro-break fact is invalid")
        for name in ("anchor_at", "pending_at", "resolved_at"):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"micro_break.{name}",
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
        ):
            raise ValueError("micro-break clocks are invalid")

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
    source_active_at: pd.Timestamp | None = None

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
        if self.source_active_at is not None:
            object.__setattr__(
                self,
                "source_active_at",
                aware_timestamp(
                    self.source_active_at,
                    name="path_step.source_active_at",
                ),
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
            or (
                self.kind == "opposite_displacement"
                and (
                    self.source_event_id is None
                    or self.source_active_at is None
                )
            )
            or (
                self.kind != "opposite_displacement"
                and self.source_active_at is not None
            )
            or (
                self.source_active_at is not None
                and self.source_active_at > self.observed_at
            )
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
class InteractionUpdate:
    """Canonical Eye output for physical zone/pool interaction facts.

    The reducer owns ordering and source binding only.  This canonical DTO has
    no Brain interpretation or legacy entry-qualification properties.
    """

    zone_interactions: tuple[EntryLocationState, ...]
    reacceptance_interactions: tuple[QualifiedReacceptanceState, ...]
    micro_break_facts: tuple[MicroBreakFact, ...]
    interaction_paths: tuple[PathSequenceState, ...]
    interaction_path_transitions: tuple[PathSequenceState, ...] = ()
    reacceptance_interaction_transitions: tuple[
        QualifiedReacceptanceState,
        ...,
    ] = ()
    milestone_transitions: tuple[tuple[str, PathSequenceStep], ...] = ()
    cold_source_ids: tuple[str, ...] = ()
    boundary_reason: str | None = None

    schema_version: ClassVar[int] = INTERACTION_UPDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "zone_interactions",
            "reacceptance_interactions",
            "micro_break_facts",
            "interaction_paths",
            "interaction_path_transitions",
            "reacceptance_interaction_transitions",
            "milestone_transitions",
            "cold_source_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self.validate_canonical_bindings()

    def validate_canonical_bindings(self) -> None:
        """Fail closed unless every physical fact has one exact owner.

        This is the single cross-record validator used both at DTO admission
        and immediately before Brain interpretation.  It deliberately checks
        only the bounded current interaction graph (or one boundary batch),
        never retained reducer history.
        """

        if set(self.__dict__) != {
            item.name for item in fields(InteractionUpdate)
        }:
            raise ValueError("interaction update shape changed")

        def exact_values(values: tuple[Any, ...], kind: type, label: str) -> None:
            """Re-admit every nested DTO through its sole canonical contract."""

            names = tuple(item.name for item in fields(kind))
            expected = set(names)
            for value in values:
                if (
                    type(value) is not kind
                    or set(getattr(value, "__dict__", ())) != expected
                ):
                    raise ValueError(f"interaction {label} shape changed")
                state = {name: getattr(value, name) for name in names}
                try:
                    canonical = kind(**state)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"interaction {label} canonical state changed"
                    ) from error
                if any(
                    type(state[name]) is not type(getattr(canonical, name))
                    or state[name] != getattr(canonical, name)
                    for name in names
                ):
                    raise ValueError(
                        f"interaction {label} canonical state changed"
                    )

        typed_collections = (
            (self.zone_interactions, EntryLocationState, "zone"),
            (
                self.reacceptance_interactions,
                QualifiedReacceptanceState,
                "reacceptance",
            ),
            (self.micro_break_facts, MicroBreakFact, "micro-break"),
            (self.interaction_paths, PathSequenceState, "path"),
            (
                self.interaction_path_transitions,
                PathSequenceState,
                "path transition",
            ),
            (
                self.reacceptance_interaction_transitions,
                QualifiedReacceptanceState,
                "reacceptance transition",
            ),
        )
        for values, kind, label in typed_collections:
            exact_values(values, kind, label)
        all_paths = (*self.interaction_paths, *self.interaction_path_transitions)
        exact_values(
            tuple(step for path in all_paths for step in path.steps),
            PathSequenceStep,
            "path step",
        )
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or type(pair[1]) is not PathSequenceStep
            for pair in self.milestone_transitions
        ):
            raise ValueError("interaction milestone transition shape changed")
        exact_values(
            tuple(pair[1] for pair in self.milestone_transitions),
            PathSequenceStep,
            "milestone",
        )
        if any(type(value) is not str or not value for value in self.cold_source_ids):
            raise ValueError("interaction cold source identity changed")
        if self.cold_source_ids != tuple(sorted(set(self.cold_source_ids))):
            raise ValueError(
                "interaction cold source identities must be unique and sorted"
            )
        if (
            self.boundary_reason is not None
            and self.boundary_reason not in GROUP5_HARD_BOUNDARY_REASONS
        ):
            raise ValueError("interaction update has an unregistered boundary")
        if self.boundary_reason is None and self.reacceptance_interaction_transitions:
            raise ValueError(
                "interaction reacceptance censor requires a hard boundary"
            )
        if any(
            state.lifecycle is not QualifiedReacceptanceLifecycle.CENSORED
            or state.censored_at is None
            or state.transition_reason != "hard_boundary_censored"
            for state in self.reacceptance_interaction_transitions
        ):
            raise ValueError(
                "interaction boundary reacceptance transition is invalid"
            )
        if self.boundary_reason is not None and any(
            (
                self.zone_interactions,
                self.reacceptance_interactions,
                self.micro_break_facts,
                self.interaction_paths,
                self.milestone_transitions,
                self.cold_source_ids,
            )
        ):
            raise ValueError("interaction boundary cannot publish current entities")

        def index_by(
            values: tuple[Any, ...], attribute: str, label: str
        ) -> dict[Any, Any]:
            output = {getattr(value, attribute): value for value in values}
            if len(output) != len(values):
                raise ValueError(f"interaction {label} identities repeat")
            return output

        def index_by_context(
            values: tuple[Any, ...], label: str
        ) -> dict[tuple[str, str], Any]:
            output = {
                (value.context_kind, value.context_id): value for value in values
            }
            if len(output) != len(values):
                raise ValueError(f"interaction {label} contexts repeat")
            return output

        locations = index_by(self.zone_interactions, "location_id", "zone")
        index_by(self.reacceptance_interactions, "reacceptance_id", "reacceptance")
        index_by(self.micro_break_facts, "reference_id", "micro-break")
        paths = index_by(self.interaction_paths, "sequence_id", "path")
        index_by(
            self.interaction_path_transitions, "sequence_id", "path transition"
        )
        index_by(
            self.reacceptance_interaction_transitions,
            "reacceptance_id",
            "reacceptance transition",
        )
        paths_by_context = index_by_context(self.interaction_paths, "path")
        transition_paths_by_context = index_by_context(
            self.interaction_path_transitions,
            "path transition",
        )
        index_by_context(self.reacceptance_interactions, "reacceptance")
        index_by_context(
            self.reacceptance_interaction_transitions,
            "reacceptance transition",
        )
        milestone_keys = tuple(
            (sequence_id, step.step_id)
            for sequence_id, step in self.milestone_transitions
        )
        if len(milestone_keys) != len(set(milestone_keys)):
            raise ValueError("interaction milestone transitions repeat")
        steps_by_path = {
            path.sequence_id: {step.step_id: step for step in path.steps}
            for path in self.interaction_paths
        }
        if any(
            steps_by_path.get(sequence_id, {}).get(step.step_id) != step
            for sequence_id, step in self.milestone_transitions
        ):
            raise ValueError("interaction milestone lacks its exact current path")
        step_ordinals_by_path = {
            path.sequence_id: {
                step.step_id: ordinal
                for ordinal, step in enumerate(path.steps)
            }
            for path in self.interaction_paths
        }
        last_milestone_ordinal: dict[str, int] = {}
        for sequence_id, step in self.milestone_transitions:
            ordinal = step_ordinals_by_path[sequence_id][step.step_id]
            if ordinal <= last_milestone_ordinal.get(sequence_id, -1):
                raise ValueError(
                    "interaction milestones are not in path predecessor order"
                )
            last_milestone_ordinal[sequence_id] = ordinal

        def validate_path(path: PathSequenceState, *, current: bool) -> None:
            if (
                path.transition_reason not in _INTERACTION_PHYSICAL_PATH_REASONS
                or any(
                    step.direction is not path.direction
                    or step.kind not in INTERACTION_PHYSICAL_PATH_STEP_KINDS
                    or step.reason
                    not in _INTERACTION_PHYSICAL_PATH_STEP_REASONS[step.kind]
                    for step in path.steps
                )
            ):
                raise ValueError(
                    "interaction path physical vocabulary or direction changed"
                )
            if path.context_kind == "zone_return":
                location = locations.get(path.context_id)
                if current and location is None:
                    raise ValueError("zone-return path lacks its exact zone")
                if (
                    path.steps[0].kind != "zone_visible"
                    or path.steps[0].source_event_id is None
                    or path.steps[0].source_event_id
                    != path.steps[0].source_entity_id
                    or (
                        current
                        and (
                            path.protocol_hash != location.protocol_hash
                            or path.symbol != location.symbol
                            or path.instrument_id != location.instrument_id
                            or path.direction is not location.direction
                            or path.formed_at != location.formed_at
                            or path.steps[0].source_event_id
                            != location.source_zone_id
                        )
                    )
                ):
                    raise ValueError("zone-return path custody changed")
            elif (
                path.steps[0].kind != "pool_swept"
                or path.steps[0].source_event_id != path.context_id
                or path.steps[0].source_entity_id != path.context_id
            ):
                raise ValueError("pool-reversal path custody changed")

        for path in self.interaction_paths:
            validate_path(path, current=True)
        if {
            path.context_id
            for path in self.interaction_paths
            if path.context_kind == "zone_return"
        } != set(locations):
            raise ValueError("interaction zones and paths differ")

        reference_steps: dict[tuple[str, str], PathSequenceStep] = {}
        micro_steps: dict[
            tuple[str, str, str, pd.Timestamp],
            PathSequenceStep,
        ] = {}
        pool_anchors: dict[tuple[str, str], pd.Timestamp] = {}
        for path in self.interaction_paths:
            context = (path.context_kind, path.context_id)
            opposite_steps = 0
            for step in path.steps:
                if step.kind == "reference_left":
                    key = (path.context_id, step.source_entity_id)
                    if key in reference_steps:
                        raise ValueError("interaction reference milestone repeats")
                    reference_steps[key] = step
                elif step.kind == "micro_break_observed":
                    if step.source_event_id is None:
                        raise ValueError("physical micro-break lacks its BOS source")
                    key = (*context, step.source_event_id, step.observed_at)
                    if key in micro_steps:
                        raise ValueError("physical micro-break source repeats")
                    micro_steps[key] = step
                elif step.kind == "opposite_displacement":
                    opposite_steps += 1
                    pool_anchors[context] = step.observed_at
            if path.context_kind == "pool_reversal" and opposite_steps != 1:
                pool_anchors.pop(context, None)

        reacceptance_keys: set[tuple[str, str]] = set()
        for state in self.reacceptance_interactions:
            path = paths_by_context.get(("zone_return", state.context_id))
            location = locations.get(state.context_id)
            reference = reference_steps.get(
                (state.context_id, state.reacceptance_id)
            )
            if (
                path is None
                or location is None
                or state.protocol_hash != path.protocol_hash
                or state.symbol != path.symbol
                or state.instrument_id != path.instrument_id
                or state.direction is not path.direction
                or state.source_entity_id != location.source_zone_id
                or state.reference_price != location.near_edge
                or state.failure_boundary != location.failure_boundary
                or reference is None
                or reference.observed_at != state.left_at
                or reference.source_event_id is not None
            ):
                raise ValueError("interaction reacceptance custody changed")
            reacceptance_keys.add((state.context_id, state.reacceptance_id))
        if set(reference_steps) != reacceptance_keys:
            raise ValueError("interaction reference milestones lack reacceptances")

        fact_keys: set[tuple[str, str, str, pd.Timestamp]] = set()
        strict_fact_clocks: dict[
            tuple[str, str], set[pd.Timestamp]
        ] = {}
        for fact in self.micro_break_facts:
            context = (fact.context_kind, fact.context_id)
            path = paths_by_context.get(context)
            key = (
                *context,
                fact.bos_id,
                fact.resolved_at,
            )
            step = micro_steps.get(key)
            if path is None:
                anchor = None
            elif path.context_kind == "zone_return":
                anchor = locations[path.context_id].first_entered_at
            else:
                anchor = pool_anchors.get(context)
            expected_reason = (
                "confirmed_m1_break_at_anchor_clock"
                if fact.relation == "same_clock_unknown"
                else "first_strictly_later_confirmed_m1_break"
            )
            if (
                key in fact_keys
                or path is None
                or fact.protocol_hash != path.protocol_hash
                or fact.context_direction is not path.direction
                or anchor is None
                or fact.anchor_at != anchor
                or step is None
                or step.direction is not fact.context_direction
                or step.source_entity_id != fact.target_swing_id
                or step.strength != fact.strength
                or step.reason != expected_reason
            ):
                raise ValueError("micro-break fact custody changed")
            fact_keys.add(key)
            if fact.relation == "strictly_after":
                strict_fact_clocks.setdefault(context, set()).add(
                    fact.resolved_at
                )
        if fact_keys != set(micro_steps):
            raise ValueError("micro-break facts and path milestones differ")
        if any(
            (
                path.transition_reason
                == "first_strict_micro_break_observed"
                and strict_fact_clocks.get(
                    (path.context_kind, path.context_id)
                )
                != {path.ended_at}
            )
            for path in self.interaction_paths
        ):
            raise ValueError(
                "strict micro-break path lacks its exact closing fact"
            )
        dominance_steps = {
            "location_left": "location_left",
            "reacceptance_failed": "reacceptance_failed",
            "accepted_outside": "accepted_outside",
            "opposite_displacement_ambiguous_same_clock": (
                "opposite_displacement_ambiguous"
            ),
        }
        for path in self.interaction_paths:
            clocks = strict_fact_clocks.get(
                (path.context_kind, path.context_id),
                set(),
            )
            if not clocks:
                continue
            dominance_step = dominance_steps.get(path.transition_reason)
            if (
                path.lifecycle is not PathSequenceLifecycle.CLOSED
                or clocks != {path.ended_at}
                or (
                    path.transition_reason
                    != "first_strict_micro_break_observed"
                    and (
                        dominance_step is None
                        or not any(
                            step.kind == dominance_step
                            and step.observed_at == path.ended_at
                            for step in path.steps
                        )
                    )
                )
            ):
                raise ValueError(
                    "strict micro-break path closing reason changed"
                )

        if self.boundary_reason is None:
            for transition in self.interaction_path_transitions:
                current = paths.get(transition.sequence_id)
                if (
                    current is None
                    or transition.protocol_hash != current.protocol_hash
                    or transition.symbol != current.symbol
                    or transition.instrument_id != current.instrument_id
                    or transition.context_kind != current.context_kind
                    or transition.context_id != current.context_id
                    or transition.direction is not current.direction
                    or transition.formed_at != current.formed_at
                    or transition.steps
                    != current.steps[: len(transition.steps)]
                ):
                    raise ValueError("interaction path transition changed custody")
        else:
            for path in self.interaction_path_transitions:
                validate_path(path, current=False)
            if any(
                path.lifecycle is not PathSequenceLifecycle.CENSORED
                or path.transition_reason != self.boundary_reason
                for path in self.interaction_path_transitions
            ):
                raise ValueError("interaction boundary path is invalid")
            for state in self.reacceptance_interaction_transitions:
                path = transition_paths_by_context.get(
                    ("zone_return", state.context_id)
                )
                if (
                    path is None
                    or state.protocol_hash != path.protocol_hash
                    or state.symbol != path.symbol
                    or state.instrument_id != path.instrument_id
                    or state.direction is not path.direction
                    or state.source_entity_id
                    != path.steps[0].source_entity_id
                    or not any(
                        step.kind == "reference_left"
                        and step.source_entity_id == state.reacceptance_id
                        and step.observed_at == state.left_at
                        for step in path.steps
                    )
                ):
                    raise ValueError(
                        "interaction boundary reacceptance changed custody"
                    )
            reference_keys: list[tuple[str, str]] = []
            terminal_reference_keys: set[tuple[str, str]] = set()
            for path in self.interaction_path_transitions:
                for step in path.steps:
                    key = (path.context_id, step.source_entity_id)
                    if step.kind == "reference_left":
                        reference_keys.append(key)
                    elif step.kind in {
                        "reacceptance_held",
                        "reacceptance_failed",
                    }:
                        terminal_reference_keys.add(key)
            if len(reference_keys) != len(set(reference_keys)):
                raise ValueError(
                    "interaction boundary reference milestones repeat"
                )
            live_reference_keys = set(reference_keys) - terminal_reference_keys
            transition_keys = {
                (state.context_id, state.reacceptance_id)
                for state in self.reacceptance_interaction_transitions
            }
            if live_reference_keys != transition_keys:
                raise ValueError(
                    "interaction boundary live reacceptance lineage differs"
                )

    def __getstate__(self) -> Mapping[str, Any]:
        self.validate_canonical_bindings()
        return _exact_dataclass_pickle_state(
            self,
            schema_version=INTERACTION_UPDATE_SCHEMA_VERSION,
            label="InteractionUpdate",
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _restore_exact_dataclass_pickle_state(
            self,
            state,
            schema_version=INTERACTION_UPDATE_SCHEMA_VERSION,
            label="InteractionUpdate",
        )
        self.__post_init__()

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
    authority_barrier_id: str | None = None
    authority_barrier_price: float | None = None
    primary_target_price_basis: str = "inventory_price"
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
                self.authority_barrier_id,
            )
        ):
            raise ValueError("liquidity-route identity is invalid")
        if (self.authority_barrier_id is None) != (
            self.authority_barrier_price is None
        ):
            raise ValueError(
                "liquidity-route authority barrier is only partially identified"
            )
        if self.authority_barrier_price is not None and (
            not math.isfinite(float(self.authority_barrier_price))
            or float(self.authority_barrier_price) <= 0.0
        ):
            raise ValueError("liquidity-route authority barrier price is invalid")
        if self.primary_target_price_basis not in {
            "inventory_price",
            "conservative_contact",
        }:
            raise ValueError(
                "liquidity-route primary target price basis is invalid"
            )
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
    event_time: pd.Timestamp | None = None
    known_at: pd.Timestamp | None = None
    semantic_version: str = SMC_SEMANTIC_VERSION
    evidence: Mapping[str, Any] = field(default_factory=dict)
    zone: tuple[float, float] | None = None
    # ``source_ids`` remains the legacy compatibility surface.  New semantic
    # producers must classify provenance explicitly so event ancestry cannot
    # be confused with raw-data, entity, or contextual evidence identities.
    # These fields intentionally live at the end of the dataclass to preserve
    # every existing positional constructor call.
    source_event_ids: tuple[str, ...] = ()
    source_data_ids: tuple[str, ...] = ()
    source_entity_ids: tuple[str, ...] = ()
    context_event_ids: tuple[str, ...] = ()
    origin: EventOrigin = EventOrigin.LEGACY_TRANSPORT

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at", aware_timestamp(self.observed_at, name="event.observed_at")
        )
        event_time = self.event_time or self.observed_at
        known_at = self.known_at or self.observed_at
        object.__setattr__(
            self,
            "event_time",
            aware_timestamp(event_time, name="event.event_time"),
        )
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(known_at, name="event.known_at"),
        )
        if self.known_at != self.observed_at:
            raise ValueError(
                "event observed_at is the legacy alias of known_at and must match"
            )
        if self.event_time > self.known_at:
            raise ValueError("event cannot be known before its market event time")
        if (
            not isinstance(self.semantic_version, str)
            or not self.semantic_version.strip()
        ):
            raise ValueError("event semantic_version is required")
        try:
            origin = EventOrigin(self.origin)
        except (TypeError, ValueError) as error:
            raise ValueError("event origin is invalid") from error
        object.__setattr__(self, "origin", origin)
        raw_details = self.details
        raw_evidence = self.evidence
        details = FrozenDict(raw_details)
        # ``details`` is the legacy name of the canonical evidence mapping.
        # Producers commonly supply both aliases with exactly the same
        # immutable payload.  Share that one frozen value instead of retaining
        # and serializing two complete copies on every historical event.
        evidence = (
            details
            if (
                not raw_evidence
                or raw_evidence is raw_details
                or _strict_payload_equal(raw_evidence, raw_details)
            )
            else FrozenDict(raw_evidence)
        )
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "evidence", evidence)
        source_namespaces: dict[str, tuple[str, ...]] = {}
        for name in (
            "source_ids",
            "source_event_ids",
            "source_data_ids",
            "source_entity_ids",
            "context_event_ids",
        ):
            raw_values = getattr(self, name)
            if isinstance(raw_values, str):
                raise ValueError(
                    f"event {name} identities must be a sequence"
                )
            try:
                values = tuple(raw_values)
            except TypeError as error:
                raise ValueError(
                    f"event {name} identities must be a sequence"
                ) from error
            if any(
                not isinstance(value, str) or not value.strip()
                for value in values
            ):
                raise ValueError(
                    f"event {name} identities must be non-empty text"
                )
            if len(values) != len(set(values)):
                if name == "source_ids":
                    # Older lifecycle transports legitimately repeated one
                    # entity through several structural roles.  Preserve
                    # that API while exposing a stable de-duplicated alias.
                    values = tuple(dict.fromkeys(values))
                else:
                    raise ValueError(
                        f"event {name} identities must be unique"
                    )
            source_namespaces[name] = values

        legacy_source_ids = source_namespaces["source_ids"]
        explicit_event_ids = source_namespaces["source_event_ids"]
        classified_event_ancestry = origin in {
            EventOrigin.SEMANTIC_ATOMIC,
            EventOrigin.STATE_PROJECTION,
        }
        if classified_event_ancestry:
            if (
                legacy_source_ids
                and explicit_event_ids
                and legacy_source_ids != explicit_event_ids
            ):
                raise ValueError(
                    "canonical event source_ids must equal "
                    "source_event_ids"
                )
            # This is the only compatibility normalization that is
            # unambiguous: a canonical producer supplied just one of the two
            # event-ancestry spellings.  Mixed raw/entity identities must use
            # their explicit namespaces instead.
            canonical_event_ids = explicit_event_ids or legacy_source_ids
            legacy_source_ids = canonical_event_ids
            explicit_event_ids = canonical_event_ids
        elif not explicit_event_ids:
            # Preserve the former ``source_event_ids`` alias for old,
            # non-canonical lifecycle fixtures and consumers.
            explicit_event_ids = legacy_source_ids

        source_namespaces["source_ids"] = legacy_source_ids
        source_namespaces["source_event_ids"] = explicit_event_ids
        for name, values in source_namespaces.items():
            object.__setattr__(self, name, values)

        explicit_namespaces = {
            name: source_namespaces[name]
            for name in (
                "source_event_ids",
                "source_data_ids",
                "source_entity_ids",
                "context_event_ids",
            )
        }
        owners: dict[str, str] = {}
        for name, values in explicit_namespaces.items():
            for value in values:
                previous = owners.setdefault(value, name)
                if previous != name:
                    raise ValueError(
                        "event provenance identities must be mutually "
                        f"exclusive across namespaces: {value!r} appears "
                        f"in {previous} and {name}"
                    )
        zone = self.zone
        if zone is None:
            lower = details.get("lower_bound")
            upper = details.get("upper_bound")
            if lower is not None and upper is not None:
                zone = (float(lower), float(upper))
        if zone is not None:
            if (
                not isinstance(zone, (tuple, list))
                or len(zone) != 2
                or not all(math.isfinite(float(value)) for value in zone)
                or float(zone[0]) > float(zone[1])
            ):
                raise ValueError("event zone is invalid")
            zone = (float(zone[0]), float(zone[1]))
        object.__setattr__(self, "zone", zone)
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

    @property
    def semantic_type(self) -> str:
        """Canonical semantic name used by the v1 event contract."""

        return self.kind.value

    @property
    def is_canonical_semantic(self) -> bool:
        """Whether this is an authoritative preregistered semantic fact."""

        return self.origin is EventOrigin.SEMANTIC_ATOMIC

    @property
    def is_projection(self) -> bool:
        """Whether this event is a rebuildable state projection."""

        return self.origin is EventOrigin.STATE_PROJECTION


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
    structural_legs: tuple[StructuralLegState, ...] = ()

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
        for item in self.structural_legs:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains a structural leg from another timeframe"
                )
            if item.known_at > self.cutoff:
                raise ValueError(
                    "frame contains a future-known structural leg"
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
    market_snapshot: "MarketSnapshot | None"
    frames: Mapping[Timeframe, FrameObservation]
    recent_events: tuple[MarketEvent, ...]
    event_durations_minutes: Mapping[str, int]
    execution: ExecutionObservation
    _snapshot_free_identity: tuple[
        pd.Timestamp,
        str,
        int,
        float,
        tuple[MarketEvent, ...],
    ] | None = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"primitive": False},
    )
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
    interaction_update: InteractionUpdate | None = None
    active_timeframes: tuple[Timeframe, ...] = ()
    scale_registry_id: str = ""
    scene_revision_id: str | None = None
    scene_added_node_ids: tuple[str, ...] = ()
    scene_revised_node_ids: tuple[str, ...] = ()
    scene_added_edge_ids: tuple[str, ...] = ()
    scene_revised_edge_ids: tuple[str, ...] = ()
    scene_resolution_event_ids: tuple[str, ...] = ()

    schema_version: ClassVar[int] = MARKET_OBSERVATION_SCHEMA_VERSION

    @property
    def asof(self) -> pd.Timestamp:
        if self.market_snapshot is not None:
            return self.market_snapshot.asof
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[0]

    @property
    def symbol(self) -> str:
        if self.market_snapshot is not None:
            return self.market_snapshot.symbol
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[1]

    @property
    def instrument_id(self) -> int:
        if self.market_snapshot is not None:
            return self.market_snapshot.instrument_id
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[2]

    @property
    def price(self) -> float:
        if self.market_snapshot is not None:
            return self.market_snapshot.price
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[3]

    @property
    def semantic_events_this_update(self) -> tuple[MarketEvent, ...]:
        if self.market_snapshot is not None:
            return self.market_snapshot.events_this_update
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[4]

    def __post_init__(self) -> None:
        identity = self._snapshot_free_identity
        if self.market_snapshot is None:
            if (
                not isinstance(identity, tuple)
                or len(identity) != 5
                or not isinstance(identity[1], str)
                or not identity[1]
                or type(identity[2]) is not int
                or identity[2] < 0
                or not math.isfinite(float(identity[3]))
                or not isinstance(identity[4], tuple)
            ):
                raise ValueError(
                    "snapshot-free observation identity is invalid"
                )
            object.__setattr__(
                self,
                "_snapshot_free_identity",
                (
                    aware_timestamp(identity[0], name="observation.asof"),
                    identity[1],
                    identity[2],
                    float(identity[3]),
                    tuple(identity[4]),
                ),
            )
        else:
            # Local import preserves the model/market_state module boundary:
            # ``market_state`` owns the concrete snapshot and imports these
            # shared contracts in turn.  Published observations must carry
            # that exact immutable authority object, never a duck-typed shim
            # whose aliases could disagree or evade snapshot validation.
            from .market_state import MarketSnapshot

            if type(self.market_snapshot) is not MarketSnapshot:
                raise TypeError(
                    "published observation requires an exact MarketSnapshot"
                )
            if identity is not None:
                raise ValueError(
                    "published observation cannot duplicate snapshot identity"
                )
        if any(
            not isinstance(event, MarketEvent)
            or event.known_at > self.asof
            for event in self.semantic_events_this_update
        ):
            raise ValueError(
                "observation semantic event delta contains future or invalid data"
            )
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
        )
        if any(
            not isinstance(item, expected_type)
            for collection, expected_type in typed_transition_contracts
            for item in collection
        ):
            raise TypeError("typed transition delta contains an invalid state")
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
            )
        }
        if (
            self.interaction_update is not None
            and self.interaction_update.boundary_reason is not None
        ):
            boundary_reasons.add(self.interaction_update.boundary_reason)
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
        if (
            self.interaction_update is not None
            and not isinstance(self.interaction_update, InteractionUpdate)
        ):
            raise TypeError("interaction update has an invalid contract")
        if self.interaction_update is not None:
            interaction = self.interaction_update
            clocked_interactions = (
                *interaction.zone_interactions,
                *interaction.reacceptance_interactions,
                *interaction.interaction_paths,
            )
            if any(
                item.symbol != self.symbol
                or item.instrument_id != self.instrument_id
                or item.last_updated_at > self.frames[Timeframe.M1].cutoff
                or item.last_updated_at > self.asof
                for item in clocked_interactions
            ):
                raise ValueError(
                    "interaction fact contract or completed clock is invalid"
                )
            if any(
                fact.resolved_at > self.frames[Timeframe.M1].cutoff
                or fact.resolved_at > self.asof
                for fact in interaction.micro_break_facts
            ):
                raise ValueError("interaction micro-break fact is in the future")
            location_ids = tuple(
                item.location_id for item in interaction.zone_interactions
            )
            reacceptance_ids = tuple(
                item.reacceptance_id
                for item in interaction.reacceptance_interactions
            )
            reference_ids = tuple(
                item.reference_id for item in interaction.micro_break_facts
            )
            sequence_ids = tuple(
                item.sequence_id for item in interaction.interaction_paths
            )
            if any(
                len(values) != len(set(values))
                for values in (
                    location_ids,
                    reacceptance_ids,
                    reference_ids,
                    sequence_ids,
                )
            ):
                raise ValueError("interaction observation identities repeat")
            path_contexts = {
                (state.context_kind, state.context_id)
                for state in interaction.interaction_paths
            }
            if len(path_contexts) != len(interaction.interaction_paths):
                raise ValueError(
                    "retained interaction path contexts must be unique"
                )
            if any(
                ("zone_return", state.location_id) not in path_contexts
                for state in interaction.zone_interactions
            ):
                raise ValueError(
                    "zone interaction lacks its retained physical path"
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
                for state in interaction.reacceptance_interactions
            ):
                raise ValueError(
                    "reacceptance interaction lacks its physical path"
                )
            if interaction.boundary_reason is not None:
                if (
                    boundary_anomaly_by_reason[interaction.boundary_reason]
                    not in self.anomalies
                ):
                    raise ValueError(
                        "interaction boundary lacks its observation anomaly"
                    )
                boundary_paths = interaction.interaction_path_transitions
                boundary_reacceptances = (
                    interaction.reacceptance_interaction_transitions
                )
                if any(
                    state.lifecycle is not PathSequenceLifecycle.CENSORED
                    or state.ended_at != self.asof
                    or state.last_updated_at != self.asof
                    or state.transition_reason
                    != interaction.boundary_reason
                    or boundary_anomaly_by_reason[state.transition_reason]
                    not in self.anomalies
                    or (
                        state.transition_reason == "contract_change_reset"
                        and (state.symbol, state.instrument_id)
                        == (self.symbol, self.instrument_id)
                    )
                    or (
                        state.transition_reason != "contract_change_reset"
                        and (state.symbol, state.instrument_id)
                        != (self.symbol, self.instrument_id)
                    )
                    for state in boundary_paths
                ):
                    raise ValueError(
                        "interaction boundary path transition is invalid"
                    )
                if any(
                    state.censored_at != self.asof
                    or state.last_updated_at != self.asof
                    for state in boundary_reacceptances
                ):
                    raise ValueError(
                        "interaction boundary reacceptance clock is invalid"
                    )
                boundary_path_contexts = {
                    (state.context_kind, state.context_id)
                    for state in boundary_paths
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
                    for state in boundary_reacceptances
                ):
                    raise ValueError(
                        "boundary reacceptance lacks its censored path context"
                    )

    def _with_scene_delta(
        self,
        *,
        scene_revision_id: str | None,
        scene_added_node_ids: tuple[str, ...] = (),
        scene_revised_node_ids: tuple[str, ...] = (),
        scene_added_edge_ids: tuple[str, ...] = (),
        scene_revised_edge_ids: tuple[str, ...] = (),
        scene_resolution_event_ids: tuple[str, ...] = (),
    ) -> "MarketObservation":
        """Attach the graph delta without revalidating the immutable Eye payload.

        ``MarketObservation`` has already completed its full causal and typed
        contract validation before the Scene Graph consumes it.  Graph
        projection changes only these six transport fields, so rebuilding the
        dataclass through :func:`dataclasses.replace` needlessly repeats the
        complete (and deliberately strict) validation of frames, inventories,
        timelines, and Group 3-5 state.

        This private clone path preserves every validated object reference and
        validates exactly the newly attached revision/delta identities.  It is
        intentionally not a general-purpose unchecked replacement API.
        """
        delta_names = (
            "scene_added_node_ids",
            "scene_revised_node_ids",
            "scene_added_edge_ids",
            "scene_revised_edge_ids",
            "scene_resolution_event_ids",
        )
        delta_values = tuple(
            tuple(value)
            for value in (
                scene_added_node_ids,
                scene_revised_node_ids,
                scene_added_edge_ids,
                scene_revised_edge_ids,
                scene_resolution_event_ids,
            )
        )
        if scene_revision_id is not None and not scene_revision_id:
            raise ValueError("scene revision identity cannot be empty")
        if scene_revision_id is None and any(delta_values):
            raise ValueError("scene delta identities require a scene revision")
        if any(
            len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
            for values in delta_values
        ):
            raise ValueError("scene delta identities are invalid")

        clone = object.__new__(type(self))
        object.__setattr__(clone, "__dict__", self.__dict__.copy())
        object.__setattr__(clone, "scene_revision_id", scene_revision_id)
        for name, values in zip(delta_names, delta_values, strict=True):
            object.__setattr__(clone, name, values)
        return clone

    def __getstate__(self) -> Mapping[str, Any]:
        if self.market_snapshot is not None:
            from .market_state import MarketSnapshot

            if type(self.market_snapshot) is not MarketSnapshot:
                raise TypeError(
                    "published observation requires an exact MarketSnapshot"
                )
        return _exact_dataclass_pickle_state(
            self,
            schema_version=MARKET_OBSERVATION_SCHEMA_VERSION,
            label="MarketObservation",
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _restore_exact_dataclass_pickle_state(
            self,
            state,
            schema_version=MARKET_OBSERVATION_SCHEMA_VERSION,
            label="MarketObservation",
        )
        self.__post_init__()

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
class FrozenTriggerState:
    """First qualified entry trigger owned by one frozen setup episode.

    Later qualified trigger kinds may strengthen the same episode, but they
    cannot replace the trigger identity or clock that first made the episode
    ready.  A different ``setup_id`` is therefore required to select a new
    trigger.
    """

    trigger_id: str
    trigger_kind: str
    observed_at: pd.Timestamp
    setup_id: str
    entry_path_id: str
    entry_location_id: str
    direction: Direction
    source_entity_id: str
    source_event_id: str | None = None
    strength: float = 0.0
    available_trigger_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="frozen_trigger.observed_at",
            ),
        )
        object.__setattr__(self, "direction", Direction(self.direction))
        kinds = tuple(dict.fromkeys(self.available_trigger_kinds))
        object.__setattr__(self, "available_trigger_kinds", kinds)
        allowed_kinds = {
            "wick_rejection",
            "reacceptance_held",
            "micro_bos_confirmed",
        }
        if (
            not self.trigger_id
            or self.trigger_kind not in allowed_kinds
            or not self.setup_id
            or not self.entry_path_id
            or not self.entry_location_id
            or not self.source_entity_id
            or self.source_event_id == ""
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
            or not kinds
            or self.trigger_kind not in kinds
            or any(kind not in allowed_kinds for kind in kinds)
        ):
            raise ValueError("frozen trigger identity or evidence is invalid")


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
class FrozenLSRContext:
    """Decision-time provenance for one long-lived LSR reversal Context.

    The parent manipulation/reacceptance/displacement mechanism is frozen
    independently of the child entry zone.  A Risk review can therefore
    validate a later FVG/OB Episode without treating that zone as the Context
    identity or depending on the bounded Group5 path still being materialized.
    """

    manipulation_id: str
    manipulation_protocol_hash: str
    source_pool_id: str
    pool_path_id: str
    pool_path_protocol_hash: str
    displacement_id: str
    direction: Direction
    swept_at: pd.Timestamp
    reaccepted_at: pd.Timestamp
    displacement_active_at: pd.Timestamp
    displacement_observed_at: pd.Timestamp
    sweep_extreme: float

    def __post_init__(self) -> None:
        for name in (
            "swept_at",
            "reaccepted_at",
            "displacement_active_at",
            "displacement_observed_at",
        ):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"lsr_context.{name}",
                ),
            )
        object.__setattr__(self, "direction", Direction(self.direction))
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.manipulation_id,
                    self.manipulation_protocol_hash,
                    self.source_pool_id,
                    self.pool_path_id,
                    self.pool_path_protocol_hash,
                    self.displacement_id,
                )
            )
            or not math.isfinite(float(self.sweep_extreme))
            or float(self.sweep_extreme) <= 0.0
            or not self.swept_at
            < self.reaccepted_at
            < self.displacement_active_at
            <= self.displacement_observed_at
        ):
            raise ValueError("frozen LSR Context provenance is invalid")

    def entry_episode_id(self, zone_id: str) -> str:
        """Return the registered child identity for one exact entry zone."""

        if not isinstance(zone_id, str) or not zone_id:
            raise ValueError("LSR entry Episode requires a zone identity")
        raw = (
            f"{self.manipulation_id}|{self.displacement_id}|{zone_id}|"
            f"{self.direction.value}"
        )
        return (
            "lsr-entry-episode:"
            f"{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
        )


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
    lsr_context: FrozenLSRContext | None = None
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
        if (
            self.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
            and self.setup_id is not None
        ):
            if (
                self.lsr_context is None
                or self.lsr_context.direction is not self.direction
                or self.invalidation.source_level_id
                != self.lsr_context.manipulation_id
                or self.invalidation.observed_at != self.lsr_context.swept_at
                or not math.isclose(
                    self.invalidation.price,
                    self.lsr_context.sweep_extreme,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "LSR plan requires its frozen parent Context provenance"
                )
        elif self.lsr_context is not None:
            raise ValueError("only LSR may carry frozen Context provenance")

    @property
    def causal_observation_clocks(self) -> tuple[pd.Timestamp, ...]:
        """All observations that causally support this frozen plan.

        Prospective deadlines are deliberately excluded: they bound plan
        validity but are expected to be later than the assessment clock.
        """

        clocks = [
            self.invalidation.observed_at,
            *(
                clock
                for target in self.targets
                for clock in (target.formed_at, target.confirmed_at)
            ),
        ]
        if self.draw_selection is not None:
            clocks.extend(
                (
                    self.draw_selection.source_confirmed_at,
                    self.draw_selection.selected_at,
                )
            )
        if self.liquidity_route is not None:
            clocks.append(self.liquidity_route.selected_at)
        if self.range_auction is not None:
            clocks.extend(
                (
                    self.range_auction.mature_at,
                    self.range_auction.swept_at,
                    self.range_auction.reentry_candidate_at,
                    self.range_auction.reentered_at,
                )
            )
        if self.lsr_context is not None:
            clocks.extend(
                (
                    self.lsr_context.swept_at,
                    self.lsr_context.reaccepted_at,
                    self.lsr_context.displacement_active_at,
                    self.lsr_context.displacement_observed_at,
                )
            )
        return tuple(clocks)


@dataclass(frozen=True)
class PlanFeasibility:
    """Common, descriptive validation of one playbook-proposed plan.

    A playbook still owns the allowed invalidation and draw sources.  This
    object only reports whether the resulting frozen geometry is currently
    usable; it neither invents a stop/target nor grants action authority.
    """

    valid: bool
    planned_entry: float | None
    invalidation: StructuralLevel | None
    target: LiquidityLevel | None
    remaining_path_R: float | None
    deadline: pd.Timestamp | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if self.deadline is not None:
            object.__setattr__(
                self,
                "deadline",
                aware_timestamp(
                    self.deadline,
                    name="plan_feasibility.deadline",
                ),
            )
        numeric = (self.planned_entry, self.remaining_path_R)
        if any(
            value is not None and not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("plan feasibility contains a non-finite value")
        complete = bool(
            self.planned_entry is not None
            and self.invalidation is not None
            and self.target is not None
            and self.remaining_path_R is not None
            and self.deadline is not None
        )
        if (
            type(self.valid) is not bool
            or self.failure_reason == ""
            or self.valid != (complete and self.failure_reason is None)
        ):
            raise ValueError("plan feasibility contract is inconsistent")


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
    entry_path_id: str | None = None
    context_id: str | None = None
    episode_id: str | None = None
    # Explicit lifecycle ownership.  ``context_id``/``episode_id`` remain the
    # evaluator-native identities; these IDs express the longer-lived market
    # thesis and its short-lived child opportunity without conflating them.
    context_thesis_id: str | None = None
    parent_context_thesis_id: str | None = None
    thesis_deadline: pd.Timestamp | None = None
    episode_deadline: pd.Timestamp | None = None
    initiating_event_id: str | None = None
    evidence_revision_id: str | None = None
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None
    terminal_source_ids: tuple[str, ...] = ()
    thesis_draw: LiquidityLevel | None = None
    draw_selection: DrawSelection | None = None
    raw_quality_dimensions: Mapping[str, float] = field(
        default_factory=dict
    )
    liquidity_route: LiquidityRoute | None = None
    context_metadata: Mapping[str, str] = field(default_factory=dict)
    competing_episode_ids: tuple[str, ...] = ()
    market_thesis_ids: tuple[str, ...] = ()
    market_thesis_id: str | None = None
    bound_market_thesis_id: str | None = None
    market_thesis_root_id: str | None = None
    market_thesis_mechanism: str | None = None
    market_thesis_authority_relation: str | None = None
    playbook_match_strength: float = 0.0
    market_thesis_binding_required: bool = False
    market_thesis_action_bound: bool = False
    market_thesis_match_status: str = "not_required"
    selected_trigger: FrozenTriggerState | None = None
    # Root-specific Brain candidates use a stable identity that is distinct
    # from the six playbook-direction summary slots.  ``required_root_id`` is
    # the canonical open-thesis root that the typed evaluator was constrained
    # to consume; the pair is absent on summary beliefs.
    candidate_id: str | None = None
    required_root_id: str | None = None
    record_kind: str = "summary"
    summary_source_candidate_id: str | None = None
    plan_feasibility: PlanFeasibility | None = None

    def __post_init__(self) -> None:
        # Backward-compatible graph-free/summary construction may provide
        # only the evaluator-native context identity.  Runtime root
        # candidates must supply the stronger epoch-scoped identity
        # explicitly; silently promoting their local context would let an
        # incomplete candidate manufacture a long-lived parent thesis.
        context_thesis_id = self.context_thesis_id
        if context_thesis_id is None and self.candidate_id is None:
            context_thesis_id = self.context_id
        parent_context_thesis_id = self.parent_context_thesis_id
        if self.episode_id is not None and parent_context_thesis_id is None:
            parent_context_thesis_id = context_thesis_id
        object.__setattr__(self, "context_thesis_id", context_thesis_id)
        object.__setattr__(
            self,
            "parent_context_thesis_id",
            parent_context_thesis_id,
        )
        entry_path_id = self.entry_path_id
        if entry_path_id is None:
            if self.plan is not None:
                entry_path_id = self.plan.entry_path_id
            elif self.selected_trigger is not None:
                entry_path_id = self.selected_trigger.entry_path_id
        object.__setattr__(self, "entry_path_id", entry_path_id)
        object.__setattr__(
            self,
            "phase_started_at",
            aware_timestamp(
                self.phase_started_at,
                name="belief.phase_started_at",
            ),
        )
        for name in ("thesis_deadline", "episode_deadline"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"belief.{name}"),
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
        context_metadata = {
            str(name): str(value)
            for name, value in self.context_metadata.items()
        }
        object.__setattr__(
            self,
            "raw_quality_dimensions",
            raw_dimensions,
        )
        object.__setattr__(self, "context_metadata", context_metadata)
        competing_episode_ids = tuple(
            dict.fromkeys(self.competing_episode_ids)
        )
        object.__setattr__(
            self,
            "competing_episode_ids",
            competing_episode_ids,
        )
        market_thesis_ids = tuple(dict.fromkeys(self.market_thesis_ids))
        object.__setattr__(self, "market_thesis_ids", market_thesis_ids)
        object.__setattr__(
            self,
            "playbook_match_strength",
            clamp(self.playbook_match_strength),
        )
        thesis_identity_fields = (
            self.market_thesis_id,
            self.bound_market_thesis_id,
            self.market_thesis_root_id,
            self.market_thesis_mechanism,
            self.market_thesis_authority_relation,
        )
        candidate_identity_fields = (
            self.candidate_id,
            self.required_root_id,
            self.summary_source_candidate_id,
        )
        valid_match_statuses = {
            "not_required",
            "no_open_thesis",
            "no_direction_match",
            "no_mechanism_match",
            "root_identity_unbound",
            "exact_root_bound",
        }
        if (
            type(self.market_thesis_binding_required) is not bool
            or type(self.market_thesis_action_bound) is not bool
            or self.market_thesis_match_status not in valid_match_statuses
            or any(
                value is not None
                and (not isinstance(value, str) or not value)
                for value in thesis_identity_fields
            )
            or (
                self.market_thesis_id is None
                and any(value is not None for value in thesis_identity_fields[1:])
            )
            or (
                self.market_thesis_id is not None
                and any(
                    value is None
                    for value in (
                        self.market_thesis_root_id,
                        self.market_thesis_mechanism,
                        self.market_thesis_authority_relation,
                    )
                )
            )
            or (
                self.market_thesis_id is not None
                and (
                    not market_thesis_ids
                    or self.market_thesis_id != market_thesis_ids[0]
                )
            )
            or bool(market_thesis_ids) != (self.market_thesis_id is not None)
            or (
                self.bound_market_thesis_id is not None
                and (
                    self.bound_market_thesis_id not in market_thesis_ids
                    or self.bound_market_thesis_id
                    != self.market_thesis_id
                )
            )
            or self.market_thesis_action_bound
            != (self.bound_market_thesis_id is not None)
            or self.market_thesis_binding_required
            != (self.market_thesis_match_status != "not_required")
            or self.market_thesis_action_bound
            != (self.market_thesis_match_status == "exact_root_bound")
            or (self.market_thesis_id is not None)
            != (
                self.market_thesis_match_status
                in {"root_identity_unbound", "exact_root_bound"}
            )
            or (
                self.market_thesis_id is None
                and self.playbook_match_strength != 0.0
            )
            or (self.candidate_id is None) != (
                self.required_root_id is None
            )
            or any(
                value is not None
                and (not isinstance(value, str) or not value)
                for value in candidate_identity_fields
            )
            or (
                self.required_root_id is not None
                and self.market_thesis_root_id
                != self.required_root_id
            )
            or self.record_kind
            not in {
                "summary",
                "root_candidate",
                "retained_episode",
                "position_management",
            }
            or (
                self.record_kind == "summary"
                and (
                    self.candidate_id is not None
                    or self.required_root_id is not None
                )
            )
            or (
                self.record_kind != "summary"
                and (
                    self.candidate_id is None
                    or self.required_root_id is None
                    or self.summary_source_candidate_id is not None
                )
            )
            or (
                self.record_kind == "summary"
                and self.summary_source_candidate_id is not None
                and not self.summary_source_candidate_id
            )
        ):
            raise ValueError("market thesis binding diagnostics are invalid")
        if self.plan_feasibility is not None:
            feasibility = self.plan_feasibility
            if self.plan is None:
                if feasibility.valid:
                    raise ValueError(
                        "belief without a plan cannot be plan-feasible"
                    )
            elif (
                feasibility.planned_entry != self.plan.planned_entry
                or feasibility.invalidation != self.plan.invalidation
                or feasibility.target != self.plan.targets[0]
                or feasibility.remaining_path_R
                != self.plan.remaining_path_R
                or feasibility.deadline != self.plan.deadline
            ):
                raise ValueError(
                    "belief plan and common feasibility view disagree"
                )
        if self.selected_trigger is not None and (
            self.setup_context_id != self.selected_trigger.setup_id
            or self.entry_location_id
            != self.selected_trigger.entry_location_id
            or self.entry_path_id != self.selected_trigger.entry_path_id
            or self.direction is not self.selected_trigger.direction
            or (
                self.sequence is not None
                and self.sequence.setup_id != self.selected_trigger.setup_id
            )
            or (
                self.plan is not None
                and self.plan.entry_path_id
                != self.selected_trigger.entry_path_id
            )
        ):
            raise ValueError("frozen trigger does not belong to the hypothesis episode")
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
        uncertainty_names = (
            "uncertainty_conflict",
            "uncertainty_required_evidence_missing",
            "uncertainty_authority_missing",
            "uncertainty_graph_ambiguity",
            "uncertainty_total",
        )
        if any(name in context_metadata for name in uncertainty_names):
            if not all(name in context_metadata for name in uncertainty_names):
                raise ValueError(
                    "belief uncertainty component metadata is incomplete"
                )
            try:
                uncertainty_values = tuple(
                    float(context_metadata[name])
                    for name in uncertainty_names
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "belief uncertainty component metadata is invalid"
                ) from error
            components = uncertainty_values[:-1]
            total = uncertainty_values[-1]
            recomputed = 1.0 - math.prod(
                1.0 - value for value in components
            )
            if (
                any(
                    not math.isfinite(value) or not 0.0 <= value <= 1.0
                    for value in uncertainty_values
                )
                or not math.isclose(
                    total,
                    recomputed,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    total,
                    float(raw_dimensions["uncertainty"]),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    total,
                    float(self.uncertainty),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "belief uncertainty total disagrees with its components"
                )
        if (
            any(not name or not value for name, value in context_metadata.items())
            or any(
                not isinstance(value, str) or not value
                for value in competing_episode_ids
            )
            or any(
                not isinstance(value, str) or not value
                for value in market_thesis_ids
            )
            or (
                self.market_thesis_action_bound
                and not market_thesis_ids
            )
        ):
            raise ValueError("belief context diagnostics are invalid")
        if self.setup_context_id == "" or self.entry_location_id == "":
            raise ValueError("belief typed context identity cannot be empty")
        if self.entry_path_id == "":
            raise ValueError("belief entry path identity cannot be empty")
        for name in (
            "context_id",
            "episode_id",
            "context_thesis_id",
            "parent_context_thesis_id",
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
        if (
            self.parent_context_thesis_id is not None
            and self.context_thesis_id
            != self.parent_context_thesis_id
        ):
            raise ValueError(
                "entry episode must reference its owning context thesis"
            )
        if (
            self.episode_id is not None
            and self.parent_context_thesis_id is None
        ):
            raise ValueError(
                "entry episode requires a parent context thesis"
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
        if self.entry_path_id is not None and (
            self.setup_context_id is None
            or self.entry_location_id is None
        ):
            raise ValueError(
                "belief entry path requires its setup and location"
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
        if (
            self.plan is not None
            and self.plan.entry_path_id is not None
            and self.entry_path_id != self.plan.entry_path_id
        ):
            raise ValueError(
                "belief and trade plan entry paths disagree"
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
        return self.candidate_id or (
            f"{self.playbook.value}:{self.direction.value}"
        )

    @property
    def causal_observation_clocks(self) -> tuple[pd.Timestamp, ...]:
        """All observations that causally support this belief snapshot."""

        clocks = [
            self.phase_started_at,
            *(evidence.observed_at for evidence in self.supporting),
            *(evidence.observed_at for evidence in self.contradicting),
            *(
                clock
                for target in self.deliverable_targets
                for clock in (target.formed_at, target.confirmed_at)
            ),
        ]
        if self.invalidation is not None:
            clocks.append(self.invalidation.observed_at)
        if self.plan is not None:
            clocks.extend(self.plan.causal_observation_clocks)
        if self.sequence is not None:
            if self.sequence.started_at is not None:
                clocks.append(self.sequence.started_at)
            clocks.extend(
                step.observed_at
                for step in self.sequence.steps
                if step.observed_at is not None
            )
        if self.terminal_at is not None:
            clocks.append(self.terminal_at)
        if self.thesis_draw is not None:
            clocks.extend(
                (self.thesis_draw.formed_at, self.thesis_draw.confirmed_at)
            )
        if self.draw_selection is not None:
            clocks.extend(
                (
                    self.draw_selection.source_confirmed_at,
                    self.draw_selection.selected_at,
                )
            )
        if self.liquidity_route is not None:
            clocks.append(self.liquidity_route.selected_at)
        if self.selected_trigger is not None:
            clocks.append(self.selected_trigger.observed_at)
        return tuple(clocks)

    @property
    def selected_trigger_kind(self) -> str | None:
        return (
            None
            if self.selected_trigger is None
            else self.selected_trigger.trigger_kind
        )

    @property
    def selected_trigger_id(self) -> str | None:
        return (
            None
            if self.selected_trigger is None
            else self.selected_trigger.trigger_id
        )

    @property
    def selected_trigger_at(self) -> pd.Timestamp | None:
        return (
            None
            if self.selected_trigger is None
            else self.selected_trigger.observed_at
        )

    @property
    def available_trigger_kinds(self) -> tuple[str, ...]:
        return (
            ()
            if self.selected_trigger is None
            else self.selected_trigger.available_trigger_kinds
        )

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
        """Compatibility view of calibrated executable delivery readiness.

        The five typed dimensions are distinct causal questions and therefore
        must not be collapsed into a pseudo-probability.  Decision consumes
        them directly; this compatibility scalar is exposed only when an
        executable hypothesis has a non-identity calibration.
        """

        if (
            self.phase is not PlaybookPhase.EXECUTABLE
            or self.calibration_version == "identity-unvalidated"
            or self.delivery_quality is None
        ):
            return 0.0
        return clamp(self.delivery_quality)


@dataclass(frozen=True)
class GlobalConflictEvidence:
    """One causally clocked and explicitly classified cross-scale opposition.

    The object records graph identities rather than a generic conflict flag so
    downstream hypothesis routing can distinguish relevant opposition from an
    unrelated graph fact.
    """

    conflict_id: str
    event_id: str
    observed_at: pd.Timestamp
    source_node_id: str
    target_node_id: str
    source_timeframe: Timeframe
    target_timeframe: Timeframe
    source_direction: Direction | None
    target_direction: Direction | None
    structural_scale: str
    role: GlobalConflictRole
    reason: str
    affected_hypothesis_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="global_conflict.observed_at",
            ),
        )
        object.__setattr__(
            self,
            "source_timeframe",
            Timeframe(self.source_timeframe),
        )
        object.__setattr__(
            self,
            "target_timeframe",
            Timeframe(self.target_timeframe),
        )
        if self.source_direction is not None:
            object.__setattr__(
                self,
                "source_direction",
                Direction(self.source_direction),
            )
        if self.target_direction is not None:
            object.__setattr__(
                self,
                "target_direction",
                Direction(self.target_direction),
            )
        object.__setattr__(self, "role", GlobalConflictRole(self.role))
        affected = tuple(dict.fromkeys(self.affected_hypothesis_ids))
        object.__setattr__(self, "affected_hypothesis_ids", affected)
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.conflict_id,
                    self.event_id,
                    self.source_node_id,
                    self.target_node_id,
                    self.structural_scale,
                    self.reason,
                )
            )
            or self.source_node_id == self.target_node_id
            or (
                self.source_direction is not None
                and self.target_direction is not None
                and self.source_direction is self.target_direction
            )
            or self.structural_scale
            not in {"internal", "intermediate", "external"}
            or not affected
            or any(
                not isinstance(value, str) or not value
                for value in affected
            )
        ):
            raise ValueError("global conflict evidence is invalid")


@dataclass(frozen=True)
class AuthorityLayer:
    """One current structural authority layer; never a playbook verdict."""

    timeframe: Timeframe
    direction: Direction
    structure_id: str
    confirmed_at: pd.Timestamp
    protected_level_id: str | None
    structural_scope: str
    acceptance_state: str
    status: str
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self,
            "confirmed_at",
            aware_timestamp(
                self.confirmed_at,
                name="authority_layer.confirmed_at",
            ),
        )
        sources = tuple(dict.fromkeys(self.source_ids))
        object.__setattr__(self, "source_ids", sources)
        if (
            self.timeframe not in {Timeframe.H4, Timeframe.H1, Timeframe.M15}
            or not self.structure_id
            or (
                self.protected_level_id is not None
                and not self.protected_level_id
            )
            or self.structural_scope
            not in {"internal", "intermediate", "external"}
            or self.acceptance_state
            not in {"pending", "rejected", "accepted", "confirmed", "unknown"}
            or self.status not in {"intact", "challenging", "invalidated"}
            or any(not isinstance(value, str) or not value for value in sources)
        ):
            raise ValueError("authority layer is invalid")


@dataclass(frozen=True)
class ScaleRelationState:
    """Causally sourced relation of one scale to dominant authority."""

    timeframe: Timeframe
    relation: ScaleRelation
    direction: Direction | None
    authority_layer_id: str | None
    evidence_ids: tuple[str, ...]
    evidence_kind: str | None
    structural_scope: str | None
    acceptance_state: str | None
    since: pd.Timestamp | None
    age_bars: int
    graph_connected: bool
    ambiguous: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "relation", ScaleRelation(self.relation))
        if self.direction is not None:
            object.__setattr__(self, "direction", Direction(self.direction))
        if self.since is not None:
            object.__setattr__(
                self,
                "since",
                aware_timestamp(self.since, name="scale_relation.since"),
            )
        evidence = tuple(dict.fromkeys(self.evidence_ids))
        object.__setattr__(self, "evidence_ids", evidence)
        if (
            (self.authority_layer_id is not None and not self.authority_layer_id)
            or any(not isinstance(value, str) or not value for value in evidence)
            or (self.evidence_kind is not None and not self.evidence_kind)
            or self.structural_scope
            not in {None, "internal", "intermediate", "external"}
            or self.acceptance_state
            not in {None, "pending", "rejected", "accepted", "confirmed", "unknown"}
            or type(self.age_bars) is not int
            or self.age_bars < 0
            or type(self.graph_connected) is not bool
            or type(self.ambiguous) is not bool
            or (self.ambiguous and self.relation is not ScaleRelation.UNKNOWN)
            or (
                self.relation is ScaleRelation.MATERIAL_OPPOSITION
                and (
                    not self.graph_connected
                    or self.direction is None
                    or not evidence
                )
            )
        ):
            raise ValueError("scale relation state is invalid")


@dataclass(frozen=True)
class BalanceContext:
    """Descriptive balance context, separate from strict FAVR authority."""

    context_id: str
    timeframe: Timeframe
    status: str
    source_ids: tuple[str, ...]
    bilateral_boundaries: bool
    internal_crossing: bool
    accepted_external_break: bool
    value_authoritative: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        sources = tuple(dict.fromkeys(self.source_ids))
        object.__setattr__(self, "source_ids", sources)
        if (
            not self.context_id
            or self.timeframe not in {Timeframe.H4, Timeframe.H1, Timeframe.M15}
            or self.status not in {"candidate", "descriptive", "authoritative"}
            or not sources
            or any(not isinstance(value, str) or not value for value in sources)
            or any(
                type(value) is not bool
                for value in (
                    self.bilateral_boundaries,
                    self.internal_crossing,
                    self.accepted_external_break,
                    self.value_authoritative,
                )
            )
            or (
                self.status == "authoritative"
                and (
                    not self.bilateral_boundaries
                    or not self.internal_crossing
                    or not self.value_authoritative
                    or self.accepted_external_break
                )
            )
            or (
                self.status != "authoritative"
                and self.value_authoritative
            )
        ):
            raise ValueError("balance context is invalid")


@dataclass(frozen=True)
class DeliveryObstruction:
    """One price-geometric obstruction candidate, independent of a plan."""

    obstruction_id: str
    timeframe: Timeframe
    direction: Direction | None
    side: str
    lower_bound: float
    upper_bound: float
    hard: bool
    source_kind: str
    source_ids: tuple[str, ...]
    structural_scope: str
    acceptance_state: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        if self.direction is not None:
            object.__setattr__(self, "direction", Direction(self.direction))
        sources = tuple(dict.fromkeys(self.source_ids))
        object.__setattr__(self, "source_ids", sources)
        if (
            not self.obstruction_id
            or self.side not in {"above", "below"}
            or not math.isfinite(float(self.lower_bound))
            or not math.isfinite(float(self.upper_bound))
            or not 0.0 < float(self.lower_bound) <= float(self.upper_bound)
            or type(self.hard) is not bool
            or not self.source_kind
            or not sources
            or any(not isinstance(value, str) or not value for value in sources)
            or self.structural_scope
            not in {"internal", "intermediate", "external"}
            or self.acceptance_state
            not in {None, "pending", "rejected", "accepted", "confirmed", "unknown"}
        ):
            raise ValueError("delivery obstruction is invalid")

    def contact_price(self, direction: Direction) -> float:
        """Conservative first contact in the proposed delivery direction."""

        direction = Direction(direction)
        return (
            float(self.lower_bound)
            if direction is Direction.LONG
            else float(self.upper_bound)
        )


@dataclass(frozen=True)
class DirectionalObstructionView:
    """Current draw/obstruction inventory for one direction, not a gate."""

    direction: Direction
    nearest_draw_id: str | None
    nearest_draw_price: float | None
    hard_barriers: tuple[DeliveryObstruction, ...]
    soft_frictions: tuple[DeliveryObstruction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction(self.direction))
        hard = tuple(self.hard_barriers)
        soft = tuple(self.soft_frictions)
        object.__setattr__(self, "hard_barriers", hard)
        object.__setattr__(self, "soft_frictions", soft)
        identities = tuple(
            item.obstruction_id for item in (*hard, *soft)
        )
        if (
            (self.nearest_draw_id is None) != (self.nearest_draw_price is None)
            or (
                self.nearest_draw_id is not None
                and (
                    not self.nearest_draw_id
                    or not math.isfinite(float(self.nearest_draw_price))
                    or float(self.nearest_draw_price) <= 0.0
                )
            )
            or any(
                not isinstance(item, DeliveryObstruction)
                for item in (*hard, *soft)
            )
            or any(not item.hard for item in hard)
            or any(item.hard for item in soft)
            or len(identities) != len(set(identities))
        ):
            raise ValueError("directional obstruction view is invalid")


@dataclass(frozen=True)
class ThesisEvidenceState:
    """Incremental factual state for one playbook-neutral market thesis."""

    lifecycle: str
    revision_id: str
    changed_at: pd.Timestamp
    supporting_event_ids: tuple[str, ...] = ()
    opposing_event_ids: tuple[str, ...] = ()
    new_supporting_event_ids: tuple[str, ...] = ()
    new_opposing_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "changed_at",
            aware_timestamp(
                self.changed_at,
                name="thesis_evidence.changed_at",
            ),
        )
        for name in (
            "supporting_event_ids",
            "opposing_event_ids",
            "new_supporting_event_ids",
            "new_opposing_event_ids",
        ):
            values = tuple(dict.fromkeys(getattr(self, name)))
            object.__setattr__(self, name, values)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"thesis evidence {name} is invalid")
        if (
            self.lifecycle
            not in {"forming", "active", "weakening", "invalidated"}
            or not self.revision_id
            or not set(self.new_supporting_event_ids).issubset(
                self.supporting_event_ids
            )
            or not set(self.new_opposing_event_ids).issubset(
                self.opposing_event_ids
            )
        ):
            raise ValueError("thesis evidence lifecycle or revision is invalid")


@dataclass(frozen=True)
class OpenMarketThesis:
    """One active, playbook-neutral interpretation of a connected graph root.

    The object carries only causal identities already observed by the Eye.  It
    neither selects a playbook nor authorizes an action; typed playbooks may
    subsequently match and constrain it.
    """

    thesis_id: str
    root_id: str
    market_epoch_id: str
    formed_at: pd.Timestamp
    updated_at: pd.Timestamp
    direction: Direction | None
    source_timeframe: Timeframe
    structural_scale: str
    mechanism: str
    authority_relation: str
    authority_source_ids: tuple[str, ...] = ()
    mechanism_event_ids: tuple[str, ...] = ()
    draw_candidate_ids: tuple[str, ...] = ()
    entry_location_ids: tuple[str, ...] = ()
    trigger_event_ids: tuple[str, ...] = ()
    obstruction_ids: tuple[str, ...] = ()
    conflict_ids: tuple[str, ...] = ()
    unknown_evidence: tuple[str, ...] = ()
    ambiguous_evidence: tuple[str, ...] = ()
    evidence_state: ThesisEvidenceState | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "formed_at",
            aware_timestamp(self.formed_at, name="market_thesis.formed_at"),
        )
        object.__setattr__(
            self,
            "updated_at",
            aware_timestamp(self.updated_at, name="market_thesis.updated_at"),
        )
        if self.direction is not None:
            object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self,
            "source_timeframe",
            Timeframe(self.source_timeframe),
        )
        identity_fields = (
            "authority_source_ids",
            "mechanism_event_ids",
            "draw_candidate_ids",
            "entry_location_ids",
            "trigger_event_ids",
            "obstruction_ids",
            "conflict_ids",
            "unknown_evidence",
            "ambiguous_evidence",
        )
        for name in identity_fields:
            values = tuple(dict.fromkeys(getattr(self, name)))
            object.__setattr__(self, name, values)
            if any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(
                    f"market thesis {name} contains an invalid identity"
                )
        if (
            not self.thesis_id
            or not self.root_id
            or not self.market_epoch_id
            or self.formed_at > self.updated_at
            or self.structural_scale
            not in {"internal", "intermediate", "external"}
            or not self.mechanism
            or not self.authority_relation
            or self.root_id not in self.mechanism_event_ids
            or (
                self.evidence_state is not None
                and (
                    self.evidence_state.changed_at > self.updated_at
                    or self.root_id
                    not in self.evidence_state.supporting_event_ids
                )
            )
        ):
            raise ValueError("open market thesis identity or clock is invalid")

    @property
    def lifecycle(self) -> str:
        return (
            "forming"
            if self.evidence_state is None
            else self.evidence_state.lifecycle
        )

    @property
    def evidence_revision_id(self) -> str:
        return (
            f"thesis-evidence:{self.thesis_id}"
            if self.evidence_state is None
            else self.evidence_state.revision_id
        )

    @property
    def supporting_event_ids(self) -> tuple[str, ...]:
        return (
            self.mechanism_event_ids
            if self.evidence_state is None
            else self.evidence_state.supporting_event_ids
        )

    @property
    def opposing_event_ids(self) -> tuple[str, ...]:
        return (
            self.conflict_ids
            if self.evidence_state is None
            else self.evidence_state.opposing_event_ids
        )


@dataclass(frozen=True)
class ContextThesisState:
    """Long-lived causal market view shared by zero or more entry episodes.

    This state is descriptive and never grants action authority.  The stable
    identity is frozen from market epoch, authority structure, direction and
    context draw by the Brain; children retain only that identity.
    """

    context_thesis_id: str
    market_epoch_id: str
    direction: Direction
    authority_ids: tuple[str, ...]
    context_draw: LiquidityLevel | None
    structural_invalidation: StructuralLevel | None
    supporting_event_ids: tuple[str, ...]
    opposing_event_ids: tuple[str, ...]
    lifecycle: str
    formed_at: pd.Timestamp
    updated_at: pd.Timestamp
    thesis_deadline: pd.Timestamp | None
    # Bounded current-child projection; historical counts are diagnostics.
    child_episode_ids: tuple[str, ...] = ()
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction(self.direction))
        for name in ("formed_at", "updated_at", "thesis_deadline", "terminal_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"context_thesis.{name}"),
                )
        for name in (
            "authority_ids",
            "supporting_event_ids",
            "opposing_event_ids",
            "child_episode_ids",
        ):
            values = tuple(dict.fromkeys(getattr(self, name)))
            object.__setattr__(self, name, values)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"context thesis {name} is invalid")
        terminal = self.lifecycle in {"completed", "invalidated", "censored"}
        invalid_reasons = tuple(
            reason
            for reason, invalid in (
                ("missing_context_thesis_id", not self.context_thesis_id),
                ("missing_market_epoch_id", not self.market_epoch_id),
                ("missing_authority_ids", not self.authority_ids),
                ("formed_after_update", self.formed_at > self.updated_at),
                (
                    "unknown_lifecycle",
                    self.lifecycle
                    not in {
                        "forming",
                        "active",
                        "weakening",
                        "completed",
                        "invalidated",
                        "censored",
                    },
                ),
                (
                    "terminal_clock_mismatch",
                    terminal != (self.terminal_at is not None),
                ),
                (
                    "terminal_reason_mismatch",
                    terminal != (self.terminal_reason is not None),
                ),
                (
                    "deadline_before_formation",
                    self.thesis_deadline is not None
                    and self.thesis_deadline < self.formed_at,
                ),
                (
                    "draw_confirmed_after_update",
                    self.context_draw is not None
                    and self.context_draw.confirmed_at > self.updated_at,
                ),
                (
                    "invalidation_observed_after_update",
                    self.structural_invalidation is not None
                    and self.structural_invalidation.observed_at
                    > self.updated_at,
                ),
            )
            if invalid
        )
        if invalid_reasons:
            raise ValueError(
                "context thesis identity or lifecycle is invalid: "
                + ",".join(invalid_reasons)
            )


@dataclass(frozen=True)
class EntryEpisodeState:
    """Short-lived entry opportunity owned by exactly one context thesis."""

    episode_id: str
    parent_context_thesis_id: str
    candidate_id: str
    playbook: Playbook
    direction: Direction
    initiating_event_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    first_pullback_at: pd.Timestamp | None
    selected_trigger: FrozenTriggerState | None
    plan: TradePlan | None
    invalidation: StructuralLevel | None
    deadline: pd.Timestamp
    phase: PlaybookPhase
    formed_at: pd.Timestamp
    updated_at: pd.Timestamp
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "playbook", Playbook(self.playbook))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(self, "phase", PlaybookPhase(self.phase))
        for name in (
            "first_pullback_at",
            "deadline",
            "formed_at",
            "updated_at",
            "terminal_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"entry_episode.{name}"),
                )
        terminal = self.phase in {
            PlaybookPhase.COMPLETED,
            PlaybookPhase.INVALIDATED,
        }
        identity_values = (
            self.episode_id,
            self.parent_context_thesis_id,
            self.candidate_id,
        )
        optional_identities = (
            self.initiating_event_id,
            self.entry_location_id,
            self.entry_path_id,
        )
        if (
            any(not isinstance(value, str) or not value for value in identity_values)
            or any(
                value is not None
                and (not isinstance(value, str) or not value)
                for value in optional_identities
            )
            or self.formed_at > self.updated_at
            or self.deadline < self.formed_at
            or terminal != (self.terminal_at is not None)
            or terminal != (self.terminal_reason is not None)
            or (
                self.selected_trigger is not None
                and (
                    self.selected_trigger.setup_id != self.episode_id
                    or self.selected_trigger.entry_location_id
                    != self.entry_location_id
                    or self.selected_trigger.entry_path_id
                    != self.entry_path_id
                )
            )
            or (
                self.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
                and self.selected_trigger is not None
                and (
                    self.first_pullback_at is None
                    or self.selected_trigger.observed_at
                    <= self.first_pullback_at
                )
            )
            or (
                self.plan is not None
                and (
                    self.plan.setup_id != self.episode_id
                    or self.plan.entry_location_id != self.entry_location_id
                    or self.plan.entry_path_id != self.entry_path_id
                    or self.plan.invalidation != self.invalidation
                    or self.plan.deadline > self.deadline
                )
            )
        ):
            raise ValueError("entry episode identity or lifecycle is invalid")

    @property
    def causal_observation_clocks(self) -> tuple[pd.Timestamp, ...]:
        """All observations that causally support this episode snapshot."""

        clocks = [self.formed_at, self.updated_at]
        for value in (self.first_pullback_at, self.terminal_at):
            if value is not None:
                clocks.append(value)
        if self.selected_trigger is not None:
            clocks.append(self.selected_trigger.observed_at)
        if self.invalidation is not None:
            clocks.append(self.invalidation.observed_at)
        if self.plan is not None:
            clocks.extend(self.plan.causal_observation_clocks)
        return tuple(clocks)


@dataclass(frozen=True)
class GlobalMarketContext:
    """Compact market-wide interpretation of the current scene graph.

    This is descriptive context.  It neither chooses a playbook draw nor owns
    an action, and execution observations never enter this contract.
    """

    updated_at: pd.Timestamp
    scene_revision_id: str
    market_epoch_id: str
    authority_stack: tuple[AuthorityLayer, ...]
    market_mode: MarketMode
    scale_relation_details: Mapping[str, ScaleRelationState]
    external_draw_candidates: Mapping[str, tuple[str, ...]]
    obstruction_views: Mapping[str, DirectionalObstructionView]
    material_conflicts: tuple[GlobalConflictEvidence, ...]
    unknown_evidence: tuple[str, ...]
    ambiguous_evidence: tuple[str, ...]
    dislocations_by_scale: Mapping[str, tuple[str, ...]]
    balance_context: BalanceContext | None = None
    invalidated_source_ids: tuple[str, ...] = ()
    candidate_structured_episode_ids: tuple[str, ...] = ()
    unexplained_structured_episode_ids: tuple[str, ...] = ()
    open_market_theses: tuple[OpenMarketThesis, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "updated_at",
            aware_timestamp(
                self.updated_at,
                name="global_context.updated_at",
            ),
        )
        authority_rank = {
            Timeframe.H4: 3,
            Timeframe.H1: 2,
            Timeframe.M15: 1,
        }
        authority_stack = tuple(
            sorted(
                self.authority_stack,
                key=lambda value: -authority_rank[value.timeframe],
            )
        )
        object.__setattr__(self, "authority_stack", authority_stack)
        object.__setattr__(self, "market_mode", MarketMode(self.market_mode))
        relation_details = {
            str(getattr(key, "value", key)): value
            for key, value in self.scale_relation_details.items()
        }
        draws = {
            str(getattr(side, "value", side)): tuple(dict.fromkeys(values))
            for side, values in self.external_draw_candidates.items()
        }
        obstruction_views = {
            str(getattr(direction, "value", direction)): value
            for direction, value in self.obstruction_views.items()
        }
        dislocations = {
            str(getattr(key, "value", key)): tuple(dict.fromkeys(values))
            for key, values in self.dislocations_by_scale.items()
        }
        object.__setattr__(self, "scale_relation_details", relation_details)
        object.__setattr__(self, "external_draw_candidates", draws)
        object.__setattr__(self, "obstruction_views", obstruction_views)
        object.__setattr__(self, "dislocations_by_scale", dislocations)
        tuple_fields = (
            "unknown_evidence",
            "ambiguous_evidence",
            "invalidated_source_ids",
            "candidate_structured_episode_ids",
            "unexplained_structured_episode_ids",
        )
        for name in tuple_fields:
            values = tuple(dict.fromkeys(getattr(self, name)))
            object.__setattr__(self, name, values)
            if any(
                not isinstance(value, str) or not value
                for value in values
            ):
                raise ValueError(
                    f"global context {name} contains an invalid identity"
                )
        conflicts = tuple(self.material_conflicts)
        object.__setattr__(self, "material_conflicts", conflicts)
        open_theses = tuple(self.open_market_theses)
        object.__setattr__(self, "open_market_theses", open_theses)
        open_theses_are_typed = all(
            type(thesis) is OpenMarketThesis for thesis in open_theses
        )
        open_theses_are_canonical = bool(
            open_theses_are_typed
            and open_theses
            == tuple(
                sorted(
                    open_theses,
                    key=lambda thesis: (
                        thesis.formed_at,
                        thesis.thesis_id,
                    ),
                )
            )
        )
        expected_scales = {timeframe.value for timeframe in Timeframe}
        if (
            not self.scene_revision_id
            or not self.market_epoch_id
            or set(relation_details) != expected_scales
            or any(
                not isinstance(value, ScaleRelationState)
                or value.timeframe.value != key
                for key, value in relation_details.items()
            )
            or set(draws) != {"above", "below"}
            or set(obstruction_views) != {
                Direction.LONG.value,
                Direction.SHORT.value,
            }
            or any(
                not isinstance(value, DirectionalObstructionView)
                or value.direction.value != key
                for key, value in obstruction_views.items()
            )
            or set(dislocations) != expected_scales
            or any(
                any(not isinstance(value, str) or not value for value in values)
                for values in dislocations.values()
            )
            or any(
                len(values) != len(set(values))
                or any(
                    not isinstance(value, str) or not value
                    for value in values
                )
                for values in draws.values()
            )
            or (
                len({layer.timeframe for layer in authority_stack})
                != len(authority_stack)
            )
            or any(not isinstance(layer, AuthorityLayer) for layer in authority_stack)
            or any(
                layer.confirmed_at > self.updated_at
                for layer in authority_stack
            )
            or any(
                state.since is not None and state.since > self.updated_at
                for state in relation_details.values()
            )
            or (
                self.market_mode is MarketMode.DIRECTIONAL
                and self.authority_direction is None
            )
            or (
                self.balance_context is not None
                and not isinstance(self.balance_context, BalanceContext)
            )
            or any(
                not isinstance(conflict, GlobalConflictEvidence)
                or conflict.observed_at > self.updated_at
                for conflict in conflicts
            )
            or len({conflict.conflict_id for conflict in conflicts})
            != len(conflicts)
            or not open_theses_are_typed
            or not open_theses_are_canonical
            or len({thesis.thesis_id for thesis in open_theses})
            != len(open_theses)
            or len({thesis.root_id for thesis in open_theses})
            != len(open_theses)
            or any(
                thesis.market_epoch_id != self.market_epoch_id
                or thesis.updated_at > self.updated_at
                for thesis in open_theses
            )
        ):
            raise ValueError("global market context is invalid")

    @property
    def dominant_authority_layer(self) -> AuthorityLayer | None:
        """Highest intact H4/H1/M15 layer; the stack is the sole source."""

        return next(
            (
                layer
                for layer in self.authority_stack
                if layer.status == "intact"
            ),
            None,
        )

    @property
    def authority_timeframe(self) -> Timeframe | None:
        layer = self.dominant_authority_layer
        return None if layer is None else layer.timeframe

    @property
    def authority_direction(self) -> Direction | None:
        layer = self.dominant_authority_layer
        return None if layer is None else layer.direction

    @property
    def authority_source_ids(self) -> tuple[str, ...]:
        layer = self.dominant_authority_layer
        if layer is None:
            return ()
        return tuple(
            dict.fromkeys(
                (
                    layer.structure_id,
                    *((layer.protected_level_id,) if layer.protected_level_id else ()),
                    *layer.source_ids,
                )
            )
        )

    @property
    def scale_relations(self) -> Mapping[str, ScaleRelation]:
        """Compatibility read view; detail states remain authoritative."""

        return {
            key: value.relation
            for key, value in self.scale_relation_details.items()
        }

    @property
    def path_blocker_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                obstruction.obstruction_id
                for view in self.obstruction_views.values()
                for obstruction in view.hard_barriers
            )
        )

    @property
    def dislocated(self) -> bool:
        return any(self.dislocations_by_scale.values())

    @property
    def dominant_dislocation_scale(self) -> Timeframe | None:
        for timeframe in (Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5, Timeframe.M1):
            if self.dislocations_by_scale[timeframe.value]:
                return timeframe
        return None


@dataclass(frozen=True)
class OpenMarketThesisClaimRelation:
    """One descriptive thesis-to-physical-episode relation.

    Claims are diagnostics, never admission or action authority.  Their
    canonical relation identity is the exact OpenMarketThesis identity; a
    directionally opposed claim remains attached to the same physical market
    episode instead of creating a second episode.
    """

    thesis_id: str
    root_id: str
    relation: str
    thesis_direction: Direction | None
    mechanism: str
    authority_relation: str
    evidence_revision_id: str
    lifecycle: str

    def __post_init__(self) -> None:
        if self.thesis_direction is not None:
            object.__setattr__(
                self,
                "thesis_direction",
                Direction(self.thesis_direction),
            )
        if (
            not self.thesis_id
            or not self.root_id
            or self.relation not in {"aligned", "opposed", "unknown"}
            or not self.mechanism
            or not self.authority_relation
            or not self.evidence_revision_id
            or self.lifecycle
            not in {"forming", "active", "weakening", "invalidated"}
        ):
            raise ValueError("open market thesis claim relation is invalid")


@dataclass(frozen=True)
class MarketEpisodeState:
    """Playbook-neutral lifecycle of one exact physical Group 5 pair."""

    episode_id: str
    market_epoch_id: str
    symbol: str
    instrument_id: int
    direction: Direction
    entry_location_id: str
    entry_path_id: str
    source_zone_id: str
    source_displacement_id: str
    entry_location_protocol_hash: str
    source_zone_detector_protocol_hash: str
    source_zone_kind: str
    source_zone_protocol_hash: str
    source_bos_id: str | None
    lower_bound: float
    upper_bound: float
    midpoint: float
    near_edge: float
    far_edge: float
    failure_boundary: float
    formed_at: pd.Timestamp
    updated_at: pd.Timestamp
    binding_status: str
    claims: tuple[OpenMarketThesisClaimRelation, ...]
    active_claim_ids: tuple[str, ...]
    claim_status: str
    first_pullback_step_id: str | None = None
    first_pullback_at: pd.Timestamp | None = None
    trigger_step_id: str | None = None
    trigger_event_id: str | None = None
    trigger_at: pd.Timestamp | None = None
    successful_pulse_at: pd.Timestamp | None = None
    successful_pulse_reason: str | None = None
    lifecycle: str = "registered"
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction(self.direction))
        for name in (
            "formed_at",
            "updated_at",
            "first_pullback_at",
            "trigger_at",
            "successful_pulse_at",
            "terminal_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"market_episode.{name}"),
                )
        claims = tuple(self.claims)
        object.__setattr__(self, "claims", claims)
        active_claim_ids = tuple(self.active_claim_ids)
        object.__setattr__(self, "active_claim_ids", active_claim_ids)
        claim_keys = tuple(
            (claim.thesis_id, claim.root_id, claim.relation)
            for claim in claims
        )
        identity_fields = (
            self.episode_id,
            self.market_epoch_id,
            self.symbol,
            self.entry_location_id,
            self.entry_path_id,
            self.source_zone_id,
            self.source_displacement_id,
            self.entry_location_protocol_hash,
            self.source_zone_detector_protocol_hash,
            self.source_zone_kind,
            self.source_zone_protocol_hash,
        )
        pullback_fields = (
            self.first_pullback_step_id,
            self.first_pullback_at,
        )
        trigger_fields = (
            self.trigger_step_id,
            self.trigger_event_id,
            self.trigger_at,
        )
        pulse_fields = (
            self.successful_pulse_at,
            self.successful_pulse_reason,
        )
        terminal_fields = (self.terminal_at, self.terminal_reason)
        expected_lifecycle = (
            "terminal"
            if self.terminal_at is not None
            else "triggered"
            if self.trigger_at is not None
            else "pullback"
            if self.first_pullback_at is not None
            else "registered"
        )
        if (
            any(not isinstance(value, str) or not value for value in identity_fields)
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.source_zone_kind not in {"fvg", "order_block"}
            or (
                self.source_zone_kind == "fvg"
                and self.source_bos_id is not None
            )
            or (
                self.source_zone_kind == "order_block"
                and (
                    not isinstance(self.source_bos_id, str)
                    or not self.source_bos_id
                )
            )
            or (
                self.source_bos_id is not None
                and (
                    not isinstance(self.source_bos_id, str)
                    or not self.source_bos_id
                )
            )
            or not all(
                math.isfinite(float(value))
                for value in (
                    self.lower_bound,
                    self.upper_bound,
                    self.midpoint,
                    self.near_edge,
                    self.far_edge,
                    self.failure_boundary,
                )
            )
            or not 0 < self.lower_bound < self.upper_bound
            or not math.isclose(
                self.midpoint,
                (self.lower_bound + self.upper_bound) / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.near_edge,
                self.upper_bound
                if self.direction is Direction.LONG
                else self.lower_bound,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.far_edge,
                self.lower_bound
                if self.direction is Direction.LONG
                else self.upper_bound,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.failure_boundary,
                self.far_edge,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or self.binding_status not in {"unique", "unbound", "ambiguous"}
            or self.formed_at > self.updated_at
            or any(not isinstance(claim, OpenMarketThesisClaimRelation) for claim in claims)
            or claim_keys != tuple(sorted(claim_keys))
            or len({claim.thesis_id for claim in claims}) != len(claims)
            or active_claim_ids != tuple(sorted(active_claim_ids))
            or len(active_claim_ids) != len(set(active_claim_ids))
            or not set(active_claim_ids).issubset(
                {claim.thesis_id for claim in claims}
            )
            or self.claim_status
            != (
                "unbound"
                if not active_claim_ids
                else "unique"
                if len(active_claim_ids) == 1
                else "ambiguous"
            )
            or any((value is None) != (pullback_fields[0] is None) for value in pullback_fields)
            or any((value is None) != (trigger_fields[0] is None) for value in trigger_fields)
            or any((value is None) != (pulse_fields[0] is None) for value in pulse_fields)
            or any((value is None) != (terminal_fields[0] is None) for value in terminal_fields)
            or any(
                value is not None
                and (not isinstance(value, str) or not value)
                for value in (
                    self.first_pullback_step_id,
                    self.trigger_step_id,
                    self.trigger_event_id,
                )
            )
            or (
                self.first_pullback_at is not None
                and not self.formed_at <= self.first_pullback_at <= self.updated_at
            )
            or (
                self.trigger_at is not None
                and (
                    self.first_pullback_at is None
                    or self.trigger_at <= self.first_pullback_at
                    or self.trigger_at > self.updated_at
                )
            )
            or (
                self.successful_pulse_at is not None
                and (
                    self.successful_pulse_at > self.updated_at
                    or self.trigger_at is None
                    or self.successful_pulse_at != self.trigger_at
                    or self.successful_pulse_reason
                    not in {
                        "micro_bos_aligned",
                        "pool_reversal_sequence_observed",
                    }
                )
            )
            or (
                self.terminal_at is not None
                and (
                    self.terminal_at > self.updated_at
                    or not self.terminal_reason
                )
            )
            or self.lifecycle != expected_lifecycle
            or any(
                claim.relation
                != (
                    "unknown"
                    if claim.thesis_direction is None
                    else "aligned"
                    if claim.thesis_direction is self.direction
                    else "opposed"
                )
                for claim in claims
            )
        ):
            raise ValueError("market episode identity or lifecycle is invalid")


@dataclass(frozen=True)
class NeutralMarketState:
    """Independent, bounded neutral projection for one completed clock."""

    schema_version: int
    asof: pd.Timestamp
    market_epoch_id: str
    scene_revision_id: str
    global_context: GlobalMarketContext
    market_episodes: tuple[MarketEpisodeState, ...]
    episode_transitions_this_update: tuple[MarketEpisodeState, ...] = ()
    unbound_entry_location_ids: tuple[str, ...] = ()
    ambiguous_entry_location_path_ids: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()
    rejected_entry_location_path_ids: tuple[tuple[str, str], ...] = ()
    retired_episode_ids_this_update: tuple[str, ...] = ()
    retirement_reasons_this_update: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "asof",
            aware_timestamp(self.asof, name="neutral_market_state.asof"),
        )
        for name in (
            "market_episodes",
            "episode_transitions_this_update",
            "unbound_entry_location_ids",
            "ambiguous_entry_location_path_ids",
            "rejected_entry_location_path_ids",
            "retired_episode_ids_this_update",
            "retirement_reasons_this_update",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        episode_keys = tuple(
            (episode.formed_at, episode.episode_id)
            for episode in self.market_episodes
        )
        transition_keys = tuple(
            (episode.updated_at, episode.episode_id)
            for episode in self.episode_transitions_this_update
        )
        current_episodes = {
            episode.episode_id: episode for episode in self.market_episodes
        }
        transition_episodes = {
            episode.episode_id: episode
            for episode in self.episode_transitions_this_update
        }
        unbound = self.unbound_entry_location_ids
        ambiguous = self.ambiguous_entry_location_path_ids
        rejected = self.rejected_entry_location_path_ids
        retired = self.retired_episode_ids_this_update
        retirement_reasons = self.retirement_reasons_this_update
        ambiguous_locations = tuple(item[0] for item in ambiguous)
        rejected_locations = tuple(item[0] for item in rejected)
        if (
            self.schema_version != NEUTRAL_MARKET_STATE_SCHEMA_VERSION
            or not self.market_epoch_id
            or not self.scene_revision_id
            or not isinstance(self.global_context, GlobalMarketContext)
            or self.global_context.updated_at != self.asof
            or self.global_context.market_epoch_id != self.market_epoch_id
            or self.global_context.scene_revision_id != self.scene_revision_id
            or any(not isinstance(episode, MarketEpisodeState) for episode in self.market_episodes)
            or any(
                not isinstance(episode, MarketEpisodeState)
                for episode in self.episode_transitions_this_update
            )
            or episode_keys != tuple(sorted(episode_keys))
            or len({episode.episode_id for episode in self.market_episodes})
            != len(self.market_episodes)
            or any(
                episode.market_epoch_id != self.market_epoch_id
                or episode.updated_at > self.asof
                for episode in self.market_episodes
            )
            or transition_keys != tuple(sorted(transition_keys))
            or len(
                {
                    episode.episode_id
                    for episode in self.episode_transitions_this_update
                }
            )
            != len(self.episode_transitions_this_update)
            or any(
                episode.updated_at != self.asof
                for episode in self.episode_transitions_this_update
            )
            or any(
                episode.updated_at == self.asof
                and transition_episodes.get(episode.episode_id) != episode
                for episode in self.market_episodes
            )
            or any(
                (
                    transition.market_epoch_id == self.market_epoch_id
                    and current_episodes.get(transition.episode_id)
                    != transition
                )
                or (
                    transition.market_epoch_id != self.market_epoch_id
                    and (
                        transition.episode_id in current_episodes
                        or transition.lifecycle != "terminal"
                        or transition.terminal_at != self.asof
                    )
                )
                for transition in self.episode_transitions_this_update
            )
            or unbound != tuple(sorted(unbound))
            or len(unbound) != len(set(unbound))
            or any(not value for value in unbound)
            or ambiguous != tuple(sorted(ambiguous))
            or len(ambiguous_locations) != len(set(ambiguous_locations))
            or any(
                not location_id
                or len(path_ids) < 2
                or tuple(path_ids) != tuple(sorted(path_ids))
                or len(path_ids) != len(set(path_ids))
                or any(not path_id for path_id in path_ids)
                for location_id, path_ids in ambiguous
            )
            or rejected != tuple(sorted(rejected))
            or len(rejected) != len(set(rejected))
            or any(not location_id or not path_id for location_id, path_id in rejected)
            or set(unbound) & set(ambiguous_locations)
            or set(unbound) & set(rejected_locations)
            or set(ambiguous_locations) & set(rejected_locations)
            or retired != tuple(sorted(retired))
            or len(retired) != len(set(retired))
            or any(not episode_id for episode_id in retired)
            or retirement_reasons != tuple(sorted(retirement_reasons))
            or tuple(episode_id for episode_id, _ in retirement_reasons)
            != retired
            or any(
                reason
                not in {
                    "upstream_compacted_after_success",
                    "upstream_compacted_after_terminal",
                }
                for _, reason in retirement_reasons
            )
        ):
            raise ValueError("neutral market state is invalid")

    @property
    def market_episode_by_id(self) -> Mapping[str, MarketEpisodeState]:
        return {
            episode.episode_id: episode
            for episode in self.market_episodes
        }

    @property
    def open_market_theses(self) -> tuple[OpenMarketThesis, ...]:
        return self.global_context.open_market_theses


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
    global_context: GlobalMarketContext | None = None
    thesis_candidates: Mapping[str, HypothesisBelief] = field(
        default_factory=dict
    )
    # A disappeared analytical root is retained here until a causal terminal
    # outcome is observed.  Dormant episodes are deliberately excluded from
    # the action interface; a stale executable snapshot can never re-enter.
    retained_episode_candidates: Mapping[str, HypothesisBelief] = field(
        default_factory=dict
    )
    position_management_candidates: Mapping[str, HypothesisBelief] = field(
        default_factory=dict
    )
    context_theses: Mapping[str, ContextThesisState] = field(
        default_factory=dict
    )
    entry_episodes: Mapping[str, EntryEpisodeState] = field(
        default_factory=dict
    )
    path_competition_state: PathCompetitionSetState | None = None
    path_update_records_this_clock: tuple[
        PathBeliefUpdateRecord,
        ...,
    ] = ()
    dol_rankings: Mapping[str, DOLRankingResult] = field(
        default_factory=dict
    )
    # Phase 7 path-marginal distributions.  The legacy rankings above remain
    # visible for diagnostics and compatibility, but Signal Policy consumes
    # these complete candidate-plus-no-target distributions.
    dol_probabilities: Mapping[str, DOLProbabilityResult] = field(
        default_factory=dict
    )
    dol_candidate_exclusions: Mapping[
        str,
        tuple[tuple[str, str], ...],
    ] = field(
        default_factory=dict
    )
    path_protocol_status: str = "development_unvalidated"
    path_authority: str = "shadow_only"
    dol_probability_protocol_fingerprint: str | None = None
    signal_policy_protocol_fingerprint: str | None = None
    # Kept structurally typed here to avoid a model -> signal_policy -> model
    # import cycle.  __post_init__ validates the immutable shadow authority
    # boundary and exact current-clock identities.
    signal_assessments: Mapping[str, Any] = field(default_factory=dict)
    trade_intents: Mapping[str, Any] = field(default_factory=dict)
    shadow_signal_rejections: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )

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
        path_records = tuple(self.path_update_records_this_clock)
        dol_rankings = dict(self.dol_rankings)
        dol_probabilities = FrozenDict(self.dol_probabilities)
        signal_assessments = FrozenDict(self.signal_assessments)
        trade_intents = FrozenDict(self.trade_intents)
        shadow_signal_rejections = FrozenDict(
            {
                str(key): tuple(values)
                for key, values in self.shadow_signal_rejections.items()
            }
        )
        dol_unresolved = {
            str(key): tuple(values)
            for key, values in self.dol_candidate_exclusions.items()
        }
        object.__setattr__(
            self,
            "path_update_records_this_clock",
            path_records,
        )
        object.__setattr__(self, "dol_rankings", dol_rankings)
        object.__setattr__(self, "dol_probabilities", dol_probabilities)
        object.__setattr__(self, "signal_assessments", signal_assessments)
        object.__setattr__(self, "trade_intents", trade_intents)
        object.__setattr__(
            self,
            "shadow_signal_rejections",
            shadow_signal_rejections,
        )
        object.__setattr__(
            self,
            "dol_candidate_exclusions",
            dol_unresolved,
        )
        path_state = self.path_competition_state
        # A caller that deliberately projects away the complete legacy path
        # scope (for example a Decision-neutral A/B view) also projects away
        # every subordinate Phase 7 object.  This is safer than retaining a
        # detached probability or intent and preserves the historical
        # ``dataclasses.replace(... path_competition_state=None ...)`` API.
        if (
            path_state is None
            and not path_records
            and not dol_rankings
            and not dol_unresolved
        ):
            dol_probabilities = FrozenDict()
            signal_assessments = FrozenDict()
            trade_intents = FrozenDict()
            object.__setattr__(self, "dol_probabilities", dol_probabilities)
            object.__setattr__(self, "signal_assessments", signal_assessments)
            object.__setattr__(self, "trade_intents", trade_intents)
        if (
            self.path_protocol_status != "development_unvalidated"
            or self.path_authority != "shadow_only"
            or any(
                not isinstance(record, PathBeliefUpdateRecord)
                or record.asof != self.asof
                for record in path_records
            )
            or any(
                not identity
                or any(
                    not isinstance(value, (tuple, list))
                    or len(value) != 2
                    or not value[0]
                    or not value[1]
                    for value in values
                )
                or len(values) != len({value[0] for value in values})
                for identity, values in dol_unresolved.items()
            )
            or any(
                not identity
                or not values
                or len(values) != len(set(values))
                or any(not isinstance(value, str) or not value for value in values)
                for identity, values in shadow_signal_rejections.items()
            )
        ):
            raise ValueError("belief path diagnostics are not shadow-only")
        if path_state is None:
            if (
                path_records
                or dol_rankings
                or dol_probabilities
                or dol_unresolved
                or signal_assessments
                or trade_intents
            ):
                raise ValueError(
                    "belief path diagnostics require a current competition set"
                )
        else:
            if not isinstance(path_state, PathCompetitionSetState):
                raise ValueError("belief path diagnostic scope is inconsistent")
            path_active = path_state.status.value == "active"
            expected_direction_keys = {
                Direction.LONG.value,
                Direction.SHORT.value,
            }
            if (
                path_state.asof != self.asof
                or path_state.protocol_status != self.path_protocol_status
                or path_state.authority != self.path_authority
                or any(
                    record.competition_set_id
                    != path_state.competition_set_id
                    for record in path_records
                )
                or (
                    path_active
                    and (
                        set(dol_rankings) != expected_direction_keys
                        or set(dol_probabilities)
                        not in (set(), expected_direction_keys)
                        or set(dol_unresolved) != expected_direction_keys
                    )
                )
                or (
                    not path_active
                    and (
                        dol_rankings
                        or dol_probabilities
                        or dol_unresolved
                        or signal_assessments
                        or trade_intents
                    )
                )
                or any(
                    not isinstance(result, DOLRankingResult)
                    or result.direction.value != direction
                    or result.competition_set_id
                    != path_state.competition_set_id
                    or result.path_asof != path_state.asof
                    or result.status != self.path_protocol_status
                    or result.authority != self.path_authority
                    for direction, result in dol_rankings.items()
                )
                or any(
                    not isinstance(result, DOLProbabilityResult)
                    or result.direction.value != direction
                    or result.competition_set_id
                    != path_state.competition_set_id
                    or result.path_asof != path_state.asof
                    or result.status != self.path_protocol_status
                    or result.authority != self.path_authority
                    or result.action_authority is not False
                    or result.path_protocol_fingerprint
                    != path_state.protocol_fingerprint
                    or result.path_model_version != path_state.model_version
                    or result.protocol_fingerprint
                    != self.dol_probability_protocol_fingerprint
                    for direction, result in dol_probabilities.items()
                )
            ):
                raise ValueError("belief path diagnostic scope is inconsistent")
        if (
            (dol_probabilities and (
                not isinstance(self.dol_probability_protocol_fingerprint, str)
                or len(self.dol_probability_protocol_fingerprint) != 64
            ))
            or ((signal_assessments or trade_intents) and (
                not isinstance(self.signal_policy_protocol_fingerprint, str)
                or len(self.signal_policy_protocol_fingerprint) != 64
            ))
            or any(
                key != getattr(value, "candidate_id", None)
                or getattr(value, "assessed_at", None) != self.asof
                or getattr(value, "authority", None) != "shadow_only"
                or getattr(value, "can_authorize_trade", None) is not False
                or key not in self.thesis_candidates
                or getattr(value, "competition_set_id", None)
                != (
                    None
                    if path_state is None
                    else path_state.competition_set_id
                )
                or getattr(value, "instrument_id", None)
                != (None if path_state is None else path_state.instrument_id)
                or getattr(value, "policy_protocol_fingerprint", None)
                != self.signal_policy_protocol_fingerprint
                or getattr(value, "direction", None)
                is not self.thesis_candidates[key].direction
                or getattr(value, "dol_ranking_id", None)
                != getattr(
                    dol_probabilities.get(
                        self.thesis_candidates[key].direction.value
                    ),
                    "probability_id",
                    None,
                )
                for key, value in signal_assessments.items()
            )
            or any(
                key != getattr(value, "candidate_id", None)
                or getattr(value, "created_at", None) != self.asof
                or getattr(value, "authority", None) != "shadow_only"
                or getattr(value, "submission_allowed", None) is not False
                or key not in signal_assessments
                or getattr(value, "signal_id", None)
                != getattr(signal_assessments[key], "signal_id", None)
                or getattr(value, "policy_protocol_fingerprint", None)
                != self.signal_policy_protocol_fingerprint
                or getattr(value, "competition_set_id", None)
                != (
                    None
                    if path_state is None
                    else path_state.competition_set_id
                )
                for key, value in trade_intents.items()
            )
        ):
            raise ValueError("belief shadow signal outputs are inconsistent")
        if any(
            key != hypothesis.key
            or any(
                clock > self.asof
                for clock in hypothesis.causal_observation_clocks
            )
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
                hypothesis.thesis_draw is not None
                and hypothesis.thesis_draw.confirmed_at > self.asof
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
                hypothesis.selected_trigger is not None
                and hypothesis.selected_trigger.observed_at > self.asof
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
        if any(
            hypothesis.record_kind != "summary"
            or (
                hypothesis.summary_source_candidate_id is not None
                and hypothesis.summary_source_candidate_id
                not in self.thesis_candidates
                and hypothesis.summary_source_candidate_id
                not in self.position_management_candidates
            )
            for hypothesis in self.hypotheses.values()
        ):
            raise ValueError("six-slot beliefs must be read-only summaries")
        candidates = tuple(self.thesis_candidates.values())
        if any(
            key != candidate.candidate_id
            or candidate.required_root_id is None
            or candidate.record_kind != "root_candidate"
            or any(
                clock > self.asof
                for clock in candidate.causal_observation_clocks
            )
            or candidate.phase_started_at > self.asof
            or (
                candidate.market_thesis_root_id
                != candidate.required_root_id
            )
            for key, candidate in self.thesis_candidates.items()
        ):
            raise ValueError("root-specific thesis candidate mapping is invalid")
        retained_candidates = tuple(
            self.retained_episode_candidates.values()
        )
        if (
            not set(self.thesis_candidates).isdisjoint(
                self.retained_episode_candidates
            )
            or any(
            key != candidate.candidate_id
            or candidate.required_root_id is None
            or candidate.record_kind != "retained_episode"
            or any(
                clock > self.asof
                for clock in candidate.causal_observation_clocks
            )
            or candidate.phase_started_at > self.asof
            or candidate.market_thesis_root_id
            != candidate.required_root_id
            or candidate.phase is PlaybookPhase.EXECUTABLE
            for key, candidate in self.retained_episode_candidates.items()
            )
        ):
            raise ValueError("retained entry episode mapping is invalid")
        candidate_slots = {
            (
                candidate.required_root_id,
                candidate.playbook,
                candidate.direction,
                candidate.episode_id,
            )
            for candidate in candidates
        }
        if len(candidate_slots) != len(candidates):
            raise ValueError(
                "root-specific EntryEpisode candidates contain duplicates"
            )
        management_candidates = tuple(
            self.position_management_candidates.values()
        )
        if (
            len(management_candidates) > 1
            or any(
                key != candidate.candidate_id
                or candidate.required_root_id is None
                or candidate.record_kind != "position_management"
                or any(
                    clock > self.asof
                    for clock in candidate.causal_observation_clocks
                )
                or candidate.phase_started_at > self.asof
                or candidate.market_thesis_root_id
                != candidate.required_root_id
                or not candidate.market_thesis_action_bound
                or candidate.market_thesis_match_status
                != "exact_root_bound"
                or candidate.phase
                not in {
                    PlaybookPhase.ENTERED,
                    PlaybookPhase.DELIVERING,
                    PlaybookPhase.WEAKENING,
                    PlaybookPhase.COMPLETED,
                    PlaybookPhase.INVALIDATED,
                }
                for key, candidate
                in self.position_management_candidates.items()
            )
            or not set(self.position_management_candidates).isdisjoint(
                self.thesis_candidates
            )
            or not set(self.position_management_candidates).isdisjoint(
                self.retained_episode_candidates
            )
        ):
            raise ValueError(
                "position-management thesis candidate mapping is invalid"
            )
        candidate_ids = {
            *self.thesis_candidates,
            *self.retained_episode_candidates,
            *self.position_management_candidates,
        }
        realtime_candidate_ids = {
            *self.thesis_candidates,
            *self.position_management_candidates,
        }
        context_ids = set(self.context_hypotheses)
        if not context_ids.issubset(realtime_candidate_ids):
            raise ValueError(
                "candidate context views must reference realtime candidates"
            )
        if (
            self.dominant_hypothesis_id is not None
            and self.dominant_hypothesis_id not in realtime_candidate_ids
        ):
            raise ValueError("dominant root candidate is not realtime")
        if (
            len(self.competing_hypothesis_ids)
            != len(set(self.competing_hypothesis_ids))
            or not set(self.competing_hypothesis_ids).issubset(
                realtime_candidate_ids
            )
            or self.dominant_hypothesis_id in self.competing_hypothesis_ids
        ):
            raise ValueError(
                "competing realtime candidate identities are invalid"
            )
        if self.focus_state is not None and self.focus_state.asof != self.asof:
            raise ValueError("belief focus and belief clocks disagree")
        if (
            self.focus_state is not None
            and self.focus_state.hypothesis_id is not None
            and self.focus_state.hypothesis_id not in realtime_candidate_ids
        ):
            raise ValueError("focus must reference a realtime root candidate")
        if self.scene_revision_id is not None and not self.scene_revision_id:
            raise ValueError("belief scene revision cannot be empty")
        if self.global_context is not None:
            if self.global_context.updated_at != self.asof:
                raise ValueError("belief global context and belief clocks disagree")
            if (
                self.scene_revision_id is not None
                and self.global_context.scene_revision_id
                != self.scene_revision_id
            ):
                raise ValueError(
                    "belief global context and scene revisions disagree"
                )
            authoritative_conflict_ids = {
                conflict.conflict_id
                for conflict in self.global_context.material_conflicts
                if GlobalConflictRole(conflict.role)
                is not GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
            }
            if not set(self.cross_scale_conflicts).issubset(
                authoritative_conflict_ids
            ):
                raise ValueError(
                    "belief conflicts must be filtered GlobalMarketContext IDs"
                )
            # Discovery-root visibility is not an EntryEpisode lifetime
            # signal.  A root-specific action candidate may therefore retain
            # its frozen exact binding while the open-thesis projection is
            # absent, provided PlaybookBrain can still resolve the identical
            # live setup/location/path.  PlaybookBrain's exact frozen-source
            # matching and mapping disjointness—not current root visibility—
            # form the action-authority boundary.
        elif candidates or retained_candidates or management_candidates:
            raise ValueError(
                "market thesis candidates require GlobalMarketContext"
            )
        context_theses = dict(self.context_theses)
        entry_episodes = dict(self.entry_episodes)
        if any(
            key != thesis.context_thesis_id
            or thesis.updated_at > self.asof
            for key, thesis in context_theses.items()
        ):
            raise ValueError("context thesis mapping identity or clock is invalid")
        if any(
            key != episode.candidate_id
            or episode.parent_context_thesis_id not in context_theses
            or any(
                clock > self.asof
                for clock in episode.causal_observation_clocks
            )
            or episode.updated_at > self.asof
            or key not in candidate_ids
            for key, episode in entry_episodes.items()
        ):
            raise ValueError("entry episode mapping identity or clock is invalid")
        if any(
            candidate.context_thesis_id is not None
            and (
                candidate.context_thesis_id not in context_theses
                or context_theses[candidate.context_thesis_id].direction
                is not candidate.direction
            )
            for candidate in (
                *candidates,
                *retained_candidates,
                *management_candidates,
            )
        ):
            raise ValueError("candidate references an unknown context thesis")
        if any(
            candidate.episode_id is not None
            and (
                candidate.candidate_id not in entry_episodes
                or entry_episodes[candidate.candidate_id].episode_id
                != candidate.episode_id
                or entry_episodes[candidate.candidate_id]
                .parent_context_thesis_id
                != candidate.parent_context_thesis_id
                or entry_episodes[candidate.candidate_id].playbook
                is not candidate.playbook
                or entry_episodes[candidate.candidate_id].direction
                is not candidate.direction
                or entry_episodes[candidate.candidate_id].phase
                is not candidate.phase
                or entry_episodes[candidate.candidate_id].entry_location_id
                != candidate.entry_location_id
                or entry_episodes[candidate.candidate_id].entry_path_id
                != candidate.entry_path_id
                or entry_episodes[candidate.candidate_id].selected_trigger
                != candidate.selected_trigger
                or entry_episodes[candidate.candidate_id].plan
                != candidate.plan
                or entry_episodes[candidate.candidate_id].invalidation
                != candidate.invalidation
                or entry_episodes[candidate.candidate_id].deadline
                != candidate.episode_deadline
            )
            for candidate in (
                *candidates,
                *retained_candidates,
                *management_candidates,
            )
        ):
            raise ValueError("candidate and entry episode lifecycle disagree")
        if any(
            not {
                episode.episode_id
                for episode in entry_episodes.values()
                if episode.parent_context_thesis_id == identity
            }.issubset(thesis.child_episode_ids)
            for identity, thesis in context_theses.items()
        ):
            raise ValueError("context thesis children disagree with entry episodes")
        # Exact entry-path ownership is an action/position invariant.  A
        # dormant analytical projection may overlap the position owner (or
        # another unresolved dormant root) while the graph is unable to prove
        # which root owned the historical path.  Those snapshots have no
        # action authority and must not crash the whole completed-bar update.
        realtime_episode_ids = {
            *self.thesis_candidates,
            *self.position_management_candidates,
        }
        active_episodes = tuple(
            episode
            for candidate_id, episode in entry_episodes.items()
            if candidate_id in realtime_episode_ids
            and episode.phase
            not in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
        )
        for field_name in ("entry_location_id", "entry_path_id"):
            identities = tuple(
                getattr(episode, field_name)
                for episode in active_episodes
                if getattr(episode, field_name) is not None
            )
            if len(identities) != len(set(identities)):
                raise ValueError(
                    f"active entry episodes cannot share {field_name}"
                )
        trigger_ids = tuple(
            episode.selected_trigger.trigger_id
            for episode in active_episodes
            if episode.selected_trigger is not None
        )
        if len(trigger_ids) != len(set(trigger_ids)):
            raise ValueError("active entry episodes cannot share a trigger")

    def candidates(self) -> tuple[HypothesisBelief, ...]:
        """Compatibility view of the six playbook-direction summaries."""

        return tuple(self.hypotheses.values())

    def owns_actionable_entry_episode(
        self,
        candidate_id: str,
        hypothesis: HypothesisBelief,
    ) -> bool:
        """Return whether one action root owns its exact live projections."""

        candidate = self.thesis_candidates.get(candidate_id)
        context_id = hypothesis.context_thesis_id
        episode_id = hypothesis.episode_id
        plan = hypothesis.plan
        if (
            candidate is not hypothesis
            or hypothesis.candidate_id != candidate_id
            or not isinstance(context_id, str)
            or not context_id
            or not isinstance(episode_id, str)
            or not episode_id
            or hypothesis.parent_context_thesis_id != context_id
            or plan is None
            or plan.setup_id != episode_id
            or hypothesis.setup_context_id != episode_id
        ):
            return False
        context = self.context_theses.get(context_id)
        episode = self.entry_episodes.get(candidate_id)
        return bool(
            context is not None
            and episode is not None
            and context.direction is hypothesis.direction
            and context.lifecycle
            not in {"completed", "invalidated", "censored"}
            and episode_id in context.child_episode_ids
            and episode.candidate_id == candidate_id
            and episode.episode_id == episode_id
            and episode.parent_context_thesis_id == context_id
            and episode.playbook is hypothesis.playbook
            and episode.direction is hypothesis.direction
            and episode.phase is hypothesis.phase
            and episode.entry_location_id == hypothesis.entry_location_id
            and episode.entry_path_id == hypothesis.entry_path_id
            and episode.selected_trigger == hypothesis.selected_trigger
            and episode.plan == plan
            and episode.invalidation == hypothesis.invalidation
            and episode.deadline == hypothesis.episode_deadline
        )

    def action_candidate_items(
        self,
    ) -> tuple[tuple[str, HypothesisBelief], ...]:
        """Stable action identities paired with their root-specific beliefs."""

        # The six playbook-direction slots are read-only summaries.  A
        # missing graph/context therefore yields no action identity instead
        # of silently promoting a summary into an executable hypothesis.
        # Graph-free construction needed by unit tests belongs in a test
        # helper, not in this production contract.
        return tuple(self.thesis_candidates.items())

    def lifecycle_candidate_items(
        self,
    ) -> tuple[tuple[str, HypothesisBelief], ...]:
        """All candidates whose causal lifecycle still needs observation.

        Unlike :meth:`action_candidate_items`, this includes dormant roots.
        Calibration/diagnostic recorders may observe them, but Decision must
        never use this interface to authorize an entry.
        """

        return (
            *tuple(self.thesis_candidates.items()),
            *tuple(self.retained_episode_candidates.items()),
            *tuple(self.position_management_candidates.items()),
        )

    def position_candidate_items(
        self,
    ) -> tuple[tuple[str, HypothesisBelief], ...]:
        """Candidates resolvable for an already-open frozen position.

        A closed analytical root may remain here only to manage the exact
        position that it created.  It is deliberately excluded from
        :meth:`action_candidate_items`, so it can never authorize a new
        ``ENTER``.
        """

        return (
            *self.action_candidate_items(),
            *tuple(self.position_management_candidates.items()),
        )

    @property
    def market_theses(self) -> tuple[OpenMarketThesis, ...]:
        """Current playbook-neutral theses, never direct action candidates."""

        return (
            ()
            if self.global_context is None
            else self.global_context.open_market_theses
        )

    def resolve_hypothesis(self, identity: str | None) -> HypothesisBelief | None:
        if identity is None:
            return None
        direct = self.thesis_candidates.get(identity)
        if direct is not None:
            return direct
        direct = self.retained_episode_candidates.get(identity)
        if direct is not None:
            return direct
        direct = self.position_management_candidates.get(identity)
        if direct is not None:
            return direct
        direct = self.hypotheses.get(identity)
        if direct is not None:
            return direct
        context = self.context_hypotheses.get(identity)
        if context is None:
            return None
        return self.thesis_candidates.get(identity) or (
            self.position_management_candidates.get(identity)
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
        ) -> tuple[float, float, float, float, float]:
            phase_rank = {
                PlaybookPhase.INVALIDATED: 0.0,
                PlaybookPhase.COMPLETED: 0.0,
                PlaybookPhase.INACTIVE: 1.0,
                PlaybookPhase.FORMING: 2.0,
                PlaybookPhase.ARMED: 3.0,
                PlaybookPhase.WAITING_LOCATION: 4.0,
                PlaybookPhase.WAITING_TRIGGER: 5.0,
                PlaybookPhase.EXECUTABLE: 6.0,
                PlaybookPhase.WEAKENING: 7.0,
                PlaybookPhase.DELIVERING: 8.0,
                PlaybookPhase.ENTERED: 9.0,
            }[belief.phase]
            return (
                float(belief.eligible),
                phase_rank,
                float(belief.entry_readiness or 0.0),
                float(belief.thesis_strength),
                -belief.uncertainty,
            )

        source = (
            {
                **self.thesis_candidates,
                **self.position_management_candidates,
            }.values()
            if self.global_context is not None
            else self.hypotheses.values()
        )
        return sorted(
            source,
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
    lsr_context: FrozenLSRContext | None = None
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
        if (
            self.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
            and self.setup_id is not None
        ):
            if self.lsr_context is None:
                raise ValueError(
                    "frozen LSR thesis lacks its parent Context provenance"
                )
        elif self.lsr_context is not None:
            raise ValueError("non-LSR thesis cannot own LSR Context provenance")


@dataclass(frozen=True)
class EngineSnapshot:
    observation: MarketObservation
    belief: MarketBelief
    decision: Decision
    risk: RiskAssessment
    neutral_market_state: NeutralMarketState | None = None

    schema_version: ClassVar[int] = ENGINE_SNAPSHOT_SCHEMA_VERSION

    @property
    def market_snapshot(self) -> "MarketSnapshot | None":
        return self.observation.market_snapshot

    def __post_init__(self) -> None:
        if (
            self.neutral_market_state is not None
            and (
                not isinstance(self.neutral_market_state, NeutralMarketState)
                or self.neutral_market_state.asof != self.observation.asof
                or self.neutral_market_state.scene_revision_id
                != self.observation.scene_revision_id
            )
        ):
            raise ValueError("Engine snapshot observation and neutral state differ")

    def __getstate__(self) -> Mapping[str, Any]:
        return _exact_dataclass_pickle_state(
            self,
            schema_version=ENGINE_SNAPSHOT_SCHEMA_VERSION,
            label="EngineSnapshot",
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _restore_exact_dataclass_pickle_state(
            self,
            state,
            schema_version=ENGINE_SNAPSHOT_SCHEMA_VERSION,
            label="EngineSnapshot",
        )
        self.__post_init__()


@dataclass(frozen=True)
class NeutralEngineSnapshot:
    """Action-free Engine projection for neutral market-case input."""

    observation: MarketObservation
    neutral_market_state: NeutralMarketState

    schema_version: ClassVar[int] = NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION

    @property
    def market_snapshot(self) -> "MarketSnapshot | None":
        return self.observation.market_snapshot

    def __post_init__(self) -> None:
        if (
            not isinstance(self.neutral_market_state, NeutralMarketState)
            or self.observation.asof != self.neutral_market_state.asof
            or self.observation.scene_revision_id
            != self.neutral_market_state.scene_revision_id
        ):
            raise ValueError(
                "neutral Engine snapshot observation and state differ"
            )

    def __getstate__(self) -> Mapping[str, Any]:
        return _exact_dataclass_pickle_state(
            self,
            schema_version=NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION,
            label="NeutralEngineSnapshot",
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _restore_exact_dataclass_pickle_state(
            self,
            state,
            schema_version=NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION,
            label="NeutralEngineSnapshot",
        )
        self.__post_init__()


def to_primitive(value: Any) -> Any:
    # The component parity walk reaches this function millions of times per
    # clock prefix.  Most leaves and containers are exact built-in types, so
    # dispatch those without repeatedly invoking ABC/subclass machinery.  The
    # fallback below intentionally preserves the original ``isinstance``
    # ordering for subclasses such as IntEnum and custom Mapping objects.
    value_type = type(value)
    if (
        value_type is str
        or value_type is bytes
        or value_type is int
        or value_type is bool
        or value is None
    ):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("cannot serialize a non-finite value")
        return value
    if value_type is pd.Timestamp:
        return value.isoformat()
    if value_type is dict or value_type is FrozenDict:
        return {
            str(key.value if isinstance(key, Enum) else key): to_primitive(item)
            for key, item in value.items()
        }
    if value_type is tuple or value_type is list:
        return [to_primitive(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize a non-finite value")
        return value
    if isinstance(value, (str, bytes, int, bool, type(None))):
        return value
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): to_primitive(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if is_dataclass(value):
        # ``asdict`` recursively deep-copies the entire object graph before
        # this function recursively normalizes it a second time. Market
        # snapshots contain immutable nested histories, so field-wise reading
        # is both semantically exact and materially cheaper during replay.
        return {
            item.name: to_primitive(getattr(value, item.name))
            for item in fields(value)
            if item.metadata.get("primitive", True)
        }
    return value


def content_hash(value: Any) -> str:
    payload = json.dumps(to_primitive(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
