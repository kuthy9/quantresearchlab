"""Eye vocabulary: the lifecycle enums and frozen reason strings.

Every value here is a name the detectors emit and downstream layers match
on. Adding a member is a semantic change, not a refactor."""
from __future__ import annotations

from enum import Enum


MARKET_OBSERVATION_SCHEMA_VERSION = 5


INTERACTION_UPDATE_SCHEMA_VERSION = 2


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
    FVG_FIRST_RETEST = "fvg_first_retest"
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
    BALANCE_RANGE_OBSERVED = "balance_range_observed"
    BALANCE_RANGE_MATURED = "balance_range_matured"
    DEALING_RANGE_ACTIVATED = "dealing_range_activated"
    DEALING_RANGE_EXTENDED = "dealing_range_extended"
    DEALING_RANGE_INVALIDATED = "dealing_range_invalidated"
    DEALING_RANGE_REPLACED = "dealing_range_replaced"
    DELIVERY_PHASE_CHANGED = "delivery_phase_changed"
    DELIVERY_PHASE_ENTERED = "delivery_phase_entered"
    DELIVERY_PHASE_UPDATED = "delivery_phase_updated"
    DELIVERY_PHASE_EXITED = "delivery_phase_exited"
    ORIGIN_ZONE_CREATED = "origin_zone_created"
    BASE_ORIGIN_CORE_CREATED = "base_origin_core_created"
    QUALIFIED_ORIGIN_ZONE_CREATED = "qualified_origin_zone_created"
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
    """A Structural Range is created, is a location, and is eventually left.

    ``mature`` was a grade the range earned by balancing, and ``forming`` meant
    it had not earned one yet.  Balance turned out to be rare in the market
    rather than absent from these intervals -- 0 of 84 ranges reached a
    two-sided test where an arbitrary H1 window reaches one 4.06% of the time,
    which the sample cannot separate -- so the grade recorded a failure that was
    never occurring.  A location has no grade.
    """

    ACTIVE = "active"
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


class ReacceptanceLifecycle(str, Enum):
    LEFT = "left"
    RECLAIMED = "reclaimed"
    HELD = "held"
    FAILED = "failed"
    CENSORED = "censored"


# Cold import/pickle compatibility for pre-rename artifacts.  Runtime owners
# use ``ReacceptanceLifecycle`` directly; this alias is not re-exported.
QualifiedReacceptanceLifecycle = ReacceptanceLifecycle


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
        "hold_completed",
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
        "qualified_reacceptance_held",
        "micro_bos_aligned",
        "micro_bos_opposed",
        "micro_bos_ambiguous_same_clock",
        "pool_reversal_sequence_observed",
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


# One visit to a boundary band, classified by how far price got.  The order is
# the escalation order: a generation keeps the deepest interaction it reached.
BALANCE_PRICE_TEST_KINDS = (
    "touch_only",
    "shallow_penetration",
    "deep_penetration",
    "close_outside",
)


RANGE_MATURITY_GATE_NAMES = (
    "duration",
    # Balance evidence is price interacting with the frozen boundary, not the
    # source zone's structural touch count -- a structural_swing zone only
    # counts a touch when another confirmed swing forms inside it, which is
    # not what "both sides were tested" means.  See balance_range_v1.2.
    "bilateral_price_tests",
    "midpoint_crossing",
    "inside_close_fraction",
    "width",
    "compression",
)


# A candidate that never balanced has lost its balance claim and nothing else.
# The structural interval keeps locating price until price closes outside it.
BALANCE_CLAIM_ABANDONED = "balance_claim_abandoned"


# The balance claim met the registered standard.  This settles the claim and
# promotes the boundaries to inventory; it is not a state of the range, which
# stays ACTIVE until price closes outside it.
BALANCE_CLAIM_CONFIRMED = "balance_claim_confirmed"


ORDER_BLOCK_FUNNEL_STAGES = (
    "active_displacement",
    "compatible_bos",
    "break_bar_belongs_to_displacement",
    "reverse_anchor_cluster_found",
    "unique_eligible_bos",
    "ob_created",
)


__all__ = [
    "BALANCE_CLAIM_ABANDONED",
    "BALANCE_CLAIM_CONFIRMED",
    "BALANCE_PRICE_TEST_KINDS",
    "BOSLifecycle",
    "BOSPostBreakState",
    "BOSScope",
    "BOS_CONFIRMATION_REASON",
    "BOS_FAILURE_REASONS",
    "BOS_SAME_CLOCK_FAILURE_REASONS",
    "DealingRangeLifecycle",
    "EntryLocationLifecycle",
    "EventKind",
    "EventOrigin",
    "FVGQualification",
    "FairValueGapLifecycle",
    "GROUP5_CONTEXT_KINDS",
    "GROUP5_HARD_BOUNDARY_REASONS",
    "GROUP5_PATH_STEP_KINDS",
    "GROUP5_SAME_CLOCK_REACCEPTANCE_FAILURE_REASONS",
    "GROUP5_SAME_CLOCK_RELATIONS",
    "INTERACTION_PHYSICAL_PATH_STEP_KINDS",
    "INTERACTION_UPDATE_SCHEMA_VERSION",
    "LiquidityInventoryLifecycle",
    "LiquidityPoolLifecycle",
    "MARKET_OBSERVATION_SCHEMA_VERSION",
    "ManipulationLifecycle",
    "ManipulationSourceDispositionKind",
    "ORDER_BLOCK_FUNNEL_STAGES",
    "OrderBlockAttemptOutcome",
    "OrderBlockLifecycle",
    "PathSequenceLifecycle",
    "QualifiedReacceptanceLifecycle",
    "RANGE_AUCTION_HARD_BOUNDARY_REASONS",
    "RANGE_MATURITY_GATE_NAMES",
    "RANGE_PAIR_FUNNEL_COUNTS",
    "ReacceptanceLifecycle",
    "STRUCTURE_BREAK_FAILURE_REASON",
    "STRUCTURE_FORMATION_FAILURE_REASON",
    "SUPPORT_RESISTANCE_RETIREMENT_REASON",
    "StructureLifecycle",
    "SupportResistanceLifecycle",
    "SwingLifecycle",
    "SwingRank",
    "SwingRelation",
    "SwingSide",
]
