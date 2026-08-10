"""Lightweight, outcome-blind statistics for the typed market eye.

The recorder consumes the immutable :class:`MarketObservation` emitted for
each completed 1m clock.  It deliberately does not interpret playbooks,
actions, PnL, MBO, or future paths.  State is reduced to bounded counters,
identity sets, numeric distributions, and a small deterministic case index.

The recorder does not pretend that a downstream snapshot can explain a
producer decision which was never exposed.  Displacement seed and Group 5
source-admission rejections remain explicitly ``not_exposed``; Order Block and
range-formation denominators use their lightweight producer funnels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, ClassVar, Iterable, Mapping, TYPE_CHECKING

import pandas as pd

from .model import MarketObservation, Timeframe

if TYPE_CHECKING:  # pragma: no cover - imported only for static analysis
    from .causal import ReaderUpdate


_CASE_LIMIT = 40
_CASE_MINIMUM = 20
_CASES_PER_STRATUM = 40
_BAD_ID_LIMIT = 8

_GROUP4_SOURCE_DISPOSITIONS = frozenset(
    {
        "rejected_prior_close",
        "rejected_source_missing_or_stale",
        "rejected_range_invalidated_same_clock",
        "ambiguous_dual_side",
        "atr_unready",
        "blocked_existing_live",
        "blocked_live_resolved_same_bar",
        "selected_primary",
        "attached_coincident_secondary",
        "attached_same_side_secondary",
    }
)

_FROZEN_CASE_STRATA = (
    "all_recognized_mature",
    "obvious_mature_looking_but_rejected",
    "near_mature_single_gate",
    "multiple_gate_rejected",
    "forming_reasonably_broken",
    "source_identity_or_reset_failure",
    "mature_range_manipulation",
    "group5_complete_path",
    "group5_interrupted_path",
)

_GROUP5_TRIGGER_STEPS = frozenset(
    {"wick_rejection", "reacceptance_held", "micro_bos_confirmed"}
)
_FAVR_TRIGGER_STEPS = frozenset(
    {"reacceptance_held", "micro_bos_confirmed"}
)
_GROUP5_COMPLETE_REASONS = frozenset(
    {"micro_bos_aligned", "pool_reversal_sequence_observed"}
)
_DISPLACEMENT_ACTIVATION_RATIO_METRICS = (
    "activation_episode_bar_count_ratio",
    "activation_relative_atr_ratio",
    "activation_efficiency_ratio",
    "activation_speed_ratio",
    "activation_mean_body_fraction_ratio",
    "activation_body_continuity_ratio",
)
_RANGE_REASONABLY_BROKEN_REASONS = frozenset(
    {"close_beyond_frozen_range", "maturity_deadline_elapsed"}
)
_RANGE_SOURCE_OR_RESET_REASONS = frozenset(
    {
        "forming_source_invalidated",
        "source_identity_changed",
        "data_gap_reset",
        "contract_change_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)


def _new_clock_state() -> dict[str, Any]:
    """Module-level defaultdict factory so recorder checkpoints pickle."""

    return {
        "observations": 0,
        "ready_observations": 0,
        "not_ready_observations": 0,
        "ready_transitions": 0,
        "real_completed": 0,
        "synthetic_completed": 0,
        "first_cutoff": None,
        "last_cutoff": None,
        "completion_denominator_status": "not_observed",
    }


def _value(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


def _clock(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("eye-statistics clocks must be timezone-aware")
    return result


def _clock_text(value: Any) -> str | None:
    result = _clock(value)
    return None if result is None else result.isoformat()


def _mapping_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _strata_key(values: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (str(name), normalized)
            for name, value in values.items()
            if (normalized := _value(value)) is not None
        )
    )


def _strata_payload(
    values: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    return dict(values)


def _source_ids(state: Any) -> tuple[str, ...]:
    output: list[str] = []
    for name in (
        "source_ids",
        "member_swing_ids",
        "source_displacement_id",
        "source_bos_id",
        "source_zone_id",
        "source_id",
        "source_inventory_item_id",
        "context_id",
        "target_swing_id",
        "lower_source_zone_id",
        "upper_source_zone_id",
    ):
        value = _mapping_value(state, name)
        if value is None:
            continue
        values = value if isinstance(value, (tuple, list, set)) else (value,)
        for item in values:
            text = str(item)
            if text and text not in output:
                output.append(text)
    return tuple(output[:12])


@dataclass
class _NumericSummary:
    """A bounded, deterministic numeric sketch suitable for checkpoints."""

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    running_mean: float = 0.0
    squared_deviation_total: float = 0.0
    signed_log2_histogram: Counter[str] = field(default_factory=Counter)

    _MIN_EXPONENT: ClassVar[int] = -12
    _MAX_EXPONENT: ClassVar[int] = 12

    @classmethod
    def _histogram_bucket(cls, number: float) -> str:
        if number == 0.0:
            return "zero"
        sign = "positive" if number > 0.0 else "negative"
        magnitude = abs(number)
        exponent = math.floor(math.log2(magnitude))
        if exponent < cls._MIN_EXPONENT:
            return f"{sign}:abs_lt_2^{cls._MIN_EXPONENT}"
        if exponent > cls._MAX_EXPONENT:
            return f"{sign}:abs_ge_2^{cls._MAX_EXPONENT + 1}"
        return f"{sign}:2^{exponent}_to_2^{exponent + 1}"

    def add(self, value: Any) -> None:
        number = _finite(value)
        if number is None:
            return
        self.count += 1
        self.total += number
        delta = number - self.running_mean
        self.running_mean += delta / self.count
        self.squared_deviation_total += delta * (number - self.running_mean)
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)
        self.signed_log2_histogram[self._histogram_bucket(number)] += 1

    def payload(self) -> dict[str, Any]:
        variance = (
            None
            if self.count == 0
            else max(0.0, self.squared_deviation_total / self.count)
        )
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": None if self.count == 0 else self.running_mean,
            "standard_deviation": (
                None if variance is None else math.sqrt(variance)
            ),
            "standard_deviation_kind": "population",
            "distribution": {
                "kind": "fixed_signed_log2_histogram",
                "minimum_exponent": self._MIN_EXPONENT,
                "maximum_exponent": self._MAX_EXPONENT,
                "counts": dict(sorted(self.signed_log2_histogram.items())),
            },
        }


_OPEN_LIFECYCLES: dict[str, frozenset[str]] = {
    "swing": frozenset({"forming", "confirmed"}),
    "structure": frozenset({"forming", "confirmed"}),
    "bos": frozenset({"pending"}),
    "support_resistance": frozenset({"active", "tested", "broken"}),
    "liquidity_pool": frozenset({"formed", "swept"}),
    "liquidity_inventory": frozenset({"visible", "targeted"}),
    "displacement": frozenset({"started", "active"}),
    "fvg": frozenset({"open", "partial"}),
    "order_block": frozenset({"created", "untested"}),
    "dealing_range": frozenset({"forming", "mature"}),
    "manipulation": frozenset({"swept"}),
    "entry_location": frozenset({"approaching", "in_zone", "rejected"}),
    "qualified_reacceptance": frozenset({"left", "reclaimed"}),
    "path_sequence": frozenset({"active"}),
}

# A terminal here means that the producer contract forbids another lifecycle
# for the same identity.  S/R ``reaccepted`` is intentionally absent because
# that entity may still retire later.
_TERMINAL_LIFECYCLES: dict[str, frozenset[str]] = {
    "swing": frozenset({"broken", "formation_failed"}),
    "structure": frozenset({"broken", "formation_failed"}),
    "bos": frozenset({"confirmed", "failed"}),
    "bos_post_break": frozenset({"accepted", "rejected"}),
    "support_resistance": frozenset({"retired"}),
    "liquidity_pool": frozenset({"accepted", "rejected"}),
    "liquidity_inventory": frozenset({"consumed"}),
    "displacement": frozenset({"exhausted", "censored"}),
    "fvg": frozenset({"mitigated", "invalidated"}),
    "order_block": frozenset({"mitigated", "failed"}),
    "dealing_range": frozenset({"broken"}),
    "manipulation": frozenset(
        {
            "reaccepted",
            "accepted_outside",
            "deadline_censored",
            "hard_boundary_censored",
        }
    ),
    "entry_location": frozenset({"left"}),
    "qualified_reacceptance": frozenset({"held", "failed", "censored"}),
    "path_sequence": frozenset({"closed", "censored"}),
}

_IMMUTABLE_EVENT_PRIMITIVES = frozenset({"micro_bos", "path_step"})


class EyeAuthorityStatistics:
    """Incrementally aggregate typed Eye authority evidence.

    ``observe`` is idempotent for the same symbol/instrument/as-of clock.
    Passing ``ReaderUpdate`` is preferred because it supplies exact real versus
    synthetic completed-bar denominators.  Without it, one changed frame
    cutoff can be counted, but a batched update cannot be reconstructed.
    """

    schema_version = 1

    def __init__(
        self,
        *,
        start: Any = None,
        end_exclusive: Any = None,
        coverage_start: Any = None,
        group4_protocol: Any = None,
    ) -> None:
        self.start = _clock(start)
        self.end_exclusive = _clock(end_exclusive)
        self.coverage_start = _clock(coverage_start)
        if (self.start is None) != (self.end_exclusive is None):
            raise ValueError("eye statistics require both start and end_exclusive")
        if (
            self.start is not None
            and self.end_exclusive is not None
            and self.end_exclusive <= self.start
        ):
            raise ValueError("eye-statistics window must be positive")
        if (
            self.coverage_start is not None
            and self.start is not None
            and self.coverage_start > self.start
        ):
            raise ValueError("coverage_start cannot follow the report start")
        self.group4_protocol = group4_protocol
        self._last_observation_key: tuple[str, int, str] | None = None
        self._observation_count = 0
        self._first_asof: pd.Timestamp | None = None
        self._last_asof: pd.Timestamp | None = None
        self._last_input_asof: pd.Timestamp | None = None
        self._clocks: dict[str, dict[str, Any]] = defaultdict(_new_clock_state)
        self._ready_seen: set[str] = set()
        self._ready_at_window_start: dict[str, bool] = {}
        self._window_start_readiness_captured = False
        self._last_classified_candle_cutoff: dict[str, str] = {}
        self._last_typed_frame_cutoff: dict[str, str] = {}
        self._reader_anomalies: Counter[str] = Counter()
        self._observation_anomalies: Counter[str] = Counter()
        self._reducer_updates: Counter[tuple[str, str, str]] = Counter()
        self._categorical: dict[
            tuple[str, str, str, tuple[tuple[str, str], ...]],
            Counter[str],
        ] = defaultdict(Counter)
        self._numeric: dict[
            tuple[str, str, str, str, tuple[tuple[str, str], ...]],
            _NumericSummary,
        ] = defaultdict(_NumericSummary)

        # Funnel keys are (group, primitive, exact strata).  Every transition
        # is also projected into the empty/global stratum.
        self._cohort_entities: dict[
            tuple[str, str, tuple[tuple[str, str], ...]], set[str]
        ] = defaultdict(set)
        self._lifecycle_entities: dict[
            tuple[str, str, tuple[tuple[str, str], ...], str], set[str]
        ] = defaultdict(set)
        self._lifecycle_transitions: Counter[
            tuple[str, str, tuple[tuple[str, str], ...], str]
        ] = Counter()
        self._reason_counts: Counter[
            tuple[str, str, tuple[tuple[str, str], ...], str]
        ] = Counter()
        # One fingerprint per currently transitionable entity replaces the
        # former all-history transition set.  Terminal identities retain only
        # their final lifecycle/clock tombstone so a repeated fallback
        # snapshot remains idempotent without keeping full state payloads.
        self._last_transition_by_entity: dict[
            tuple[str, str, str], tuple[str, str]
        ] = {}
        self._terminal_transition_by_entity: dict[
            tuple[str, str, str], tuple[str, str]
        ] = {}
        # Identities which are live in the final warmup snapshot belong to a
        # left-truncated cohort.  Keep this separate from the identities which
        # are actually encountered and suppressed inside the report window so
        # the summary reports observed censoring, not every warmup entity.
        self._warmup_entity_keys: set[tuple[str, str, str]] = set()
        self._left_boundary_entities: set[tuple[str, str, str]] = set()
        self._left_boundary_counts: Counter[tuple[str, str]] = Counter()
        self._open_latest: dict[
            tuple[str, str, str],
            tuple[int, str, tuple[tuple[str, str], ...], str | None],
        ] = {}
        self._terminal_latest_counts: Counter[
            tuple[str, str, tuple[tuple[str, str], ...], str]
        ] = Counter()
        self._cohort_strata_by_entity: dict[
            tuple[str, str, str], tuple[tuple[str, str], ...]
        ] = {}
        self._cohort_strata_revision_counts: Counter[tuple[str, str]] = Counter()
        self._sequence = 0
        self._pending_warmup_observation: MarketObservation | None = None
        self._inventory_state_by_id: dict[str, Any] = {}
        self._window_inventory_primed = False
        self._transport_counts: Counter[str] = Counter()
        self._final_reconciled_observation_key: tuple[str, int, str] | None = None

        self._mature_months: Counter[str] = Counter()
        self._seen_mature_ranges: set[str] = set()
        self._last_range_gate_clock: dict[str, str] = {}
        self._range_unmet_combinations: Counter[str] = Counter()
        self._range_unmet_sizes: Counter[str] = Counter()
        self._range_gate_observations = 0
        self._range_funnel_contract_seen = False
        self._last_range_funnel_clock: pd.Timestamp | None = None
        self._range_pair_counts: Counter[str] = Counter()
        self._range_funnel_snapshots = 0
        self._range_gate_first_by_id: dict[str, dict[str, Any]] = {}
        self._range_gate_latest_by_id: dict[str, dict[str, Any]] = {}
        self._left_boundary_range_gate_evaluations = 0
        self._disposition_ids: dict[
            tuple[str, str, str, str, str, str], set[tuple[str, str]]
        ] = defaultdict(set)
        self._source_disposition_by_clock_id: dict[tuple[str, str], str] = {}
        self._source_metadata: dict[str, dict[str, str]] = {}
        self._visible_eligible_sources: set[str] = set()
        self._visible_eligible_source_ids: dict[
            tuple[str, str, str, str, str], set[str]
        ] = defaultdict(set)
        self._disposition_contract_seen = False
        self._pool_rank_exposed: set[str] = set()
        self._pool_rank_missing: set[str] = set()
        self._manipulation_metadata: dict[str, dict[str, Any]] = {}
        self._manipulation_metrics_seen: set[str] = set()
        self._mature_range_details: dict[str, dict[str, Any]] = {}
        self._opposed_mss_by_displacement: dict[
            str, dict[str, dict[str, Any]]
        ] = defaultdict(dict)

        self._displacement_started: set[str] = set()
        self._displacement_active: set[str] = set()
        self._displacement_terminal_reason: dict[str, str] = {}
        self._displacement_latest_activation_ratios: dict[
            str, dict[str, float]
        ] = {}
        self._displacement_same_bar_restarts = 0
        self._displacement_durations = _NumericSummary()
        self._displacement_interruptions = _NumericSummary()
        self._ob_funnel_attempts = 0
        self._ob_funnel_stages: Counter[str] = Counter()
        self._ob_funnel_outcomes: Counter[str] = Counter()
        self._last_ob_funnel_clock: pd.Timestamp | None = None
        self._ob_funnel_contract_seen = False

        self._group5_contract_seen = False
        # Keep only the immutable fields needed by final Group 5/FAVR joins.
        # Full PathSequenceStep objects belong to the reducer, not the annual
        # aggregate statistics checkpoint.
        self._path_steps_by_id: dict[
            str, tuple[tuple[str, str | None, pd.Timestamp | None], ...]
        ] = {}
        self._path_has_trigger_ids: set[str] = set()
        self._path_terminal_class_by_id: dict[str, str] = {}
        self._path_terminal_reasons: Counter[str] = Counter()
        self._path_terminal_classes: Counter[str] = Counter()
        self._group5_path_order_errors = 0
        self._group5_manipulation_paths: dict[str, str] = {}
        self._manipulation_cohort_ids: set[str] = set()
        self._group5_manipulation_funnel_ids: dict[
            tuple[str, str, str], set[str]
        ] = defaultdict(set)
        self._group5_qualified_zone_metadata: dict[
            str, tuple[str, str]
        ] = {}
        self._group5_entry_location_metadata: dict[
            str, tuple[str, str]
        ] = {}
        self._group5_zone_path_metadata: dict[
            str, tuple[str, str]
        ] = {}
        self._group5_zone_funnel_ids: dict[
            tuple[str, str, str], set[str]
        ] = defaultdict(set)
        # Exact producer identities retained for bounded conservation checks.
        # These are not an inferred causal graph: every link comes directly
        # from a typed state/context identity emitted by Group 3/5.
        self._group5_qualified_zone_details: dict[str, dict[str, Any]] = {}
        self._group5_entry_location_details: dict[str, dict[str, Any]] = {}
        self._group5_entry_source_zone_by_id: dict[str, str] = {}
        self._group5_zone_path_location_by_id: dict[str, str] = {}
        self._group5_pool_path_manipulation_by_id: dict[str, str] = {}
        self._group5_entry_binding_conflicts: set[str] = set()
        self._group5_zone_path_binding_conflicts: set[str] = set()
        self._group5_pool_path_binding_conflicts: set[str] = set()
        self._group5_zone_trigger_paths: set[str] = set()
        self._group5_zone_terminal_paths: set[str] = set()
        self._group5_pool_trigger_paths: set[str] = set()
        self._group5_pool_terminal_paths: set[str] = set()
        self._group5_path_directions: dict[str, str] = {}
        self._favr_reaccepted_manipulation_ids: set[str] = set()

        self._range_formation_atr_by_id: dict[str, float] = {}

        self._case_candidate_count = 0
        self._case_pools: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._range_case_episode_strata_seen: set[tuple[str, str]] = set()

    @staticmethod
    def _bind_exact_identity(
        bindings: dict[str, str],
        *,
        child_id: str,
        parent_id: str,
        conflicts: set[str],
    ) -> None:
        """Latch one producer-emitted parent identity without rewriting it."""

        if not child_id:
            return
        previous = bindings.get(child_id)
        if previous is None:
            bindings[child_id] = parent_id
        elif previous != parent_id:
            conflicts.add(child_id)

    def _in_window(self, value: Any) -> bool:
        clock = _clock(value)
        if clock is None:
            return False
        if self.start is not None and clock < self.start:
            return False
        if self.end_exclusive is not None and clock >= self.end_exclusive:
            return False
        return True

    @staticmethod
    def _transition_clock(state: Any, lifecycle: str, fallback: Any) -> pd.Timestamp:
        names_by_lifecycle = {
            "forming": ("formed_at",),
            "formation_failed": ("formation_failed_at", "observed_at"),
            "confirmed": ("confirmed_at", "resolved_at"),
            "broken": ("broken_at",),
            "pending": ("pending_at",),
            "failed": ("failed_at", "resolved_at", "state_started_at"),
            "active": ("active_at", "state_started_at", "formed_at"),
            "tested": ("tested_at", "state_started_at"),
            "reaccepted": ("reaccepted_at", "resolved_at", "state_started_at"),
            "accepted": ("resolved_at", "accepted_at"),
            "rejected": ("rejected_at", "resolved_at", "state_started_at"),
            "retired": ("retired_at",),
            "visible": ("confirmed_at",),
            "targeted": ("targeted_at",),
            "consumed": ("consumed_at",),
            "started": ("started_at", "state_started_at"),
            "exhausted": ("terminal_at", "state_started_at"),
            "censored": ("censored_at", "terminal_at", "state_started_at"),
            "open": ("formed_at", "state_started_at"),
            "partial": ("partial_at", "state_started_at"),
            "mitigated": ("mitigated_at", "state_started_at"),
            "invalidated": ("invalidated_at", "state_started_at"),
            "created": ("formed_at", "state_started_at"),
            "untested": ("state_started_at",),
            "mature": ("mature_at", "state_started_at"),
            "swept": ("swept_at", "state_started_at"),
            "accepted_outside": ("accepted_outside_at", "resolved_at"),
            "approaching": ("state_started_at", "formed_at"),
            "in_zone": ("first_entered_at", "state_started_at"),
            "left": ("left_at", "state_started_at", "formed_at"),
            "reclaimed": ("reclaimed_at", "state_started_at"),
            "held": ("held_at", "state_started_at"),
            "closed": ("ended_at", "state_started_at"),
        }
        for name in names_by_lifecycle.get(lifecycle, ()):
            value = _mapping_value(state, name)
            if value is not None:
                return _clock(value)  # type: ignore[return-value]
        for name in ("state_started_at", "observed_at", "formed_at", "confirmed_at"):
            value = _mapping_value(state, name)
            if value is not None:
                return _clock(value)  # type: ignore[return-value]
        result = _clock(fallback)
        if result is None:
            raise ValueError("typed state has no causal transition clock")
        return result

    @staticmethod
    def _funnel_keys(
        group: str,
        primitive: str,
        strata: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]:
        global_key = (group, primitive, ())
        exact_key = (group, primitive, strata)
        return (global_key,) if exact_key == global_key else (global_key, exact_key)

    def _add_case(
        self,
        *,
        group: str,
        primitive: str,
        entity_id: str,
        lifecycle: str,
        event_clock: pd.Timestamp,
        strata: tuple[tuple[str, str], ...],
        state: Any,
    ) -> None:
        stratum = self._case_stratum(
            group,
            primitive,
            lifecycle,
            strata,
            state,
        )
        if stratum is None:
            return
        self._add_named_case(
            stratum=stratum,
            group=group,
            primitive=primitive,
            entity_id=entity_id,
            lifecycle=lifecycle,
            event_clock=event_clock,
            strata=strata,
            state=state,
        )

    def _add_named_case(
        self,
        *,
        stratum: str,
        group: str,
        primitive: str,
        entity_id: str,
        lifecycle: str,
        event_clock: pd.Timestamp,
        strata: tuple[tuple[str, str], ...],
        state: Any,
    ) -> None:
        if stratum not in _FROZEN_CASE_STRATA:
            raise ValueError(f"unregistered authority case stratum: {stratum}")
        if primitive == "range_maturity_evaluation":
            representative_key = (stratum, entity_id)
            if representative_key in self._range_case_episode_strata_seen:
                return
            # Pre-registered rule: retain the first causal evaluation for one
            # range episode in each blind-review category.  Later snapshots
            # cannot crowd the case pool or replace it using eventual outcome.
            self._range_case_episode_strata_seen.add(representative_key)
        identity = "|".join(
            (stratum, group, primitive, entity_id, lifecycle, event_clock.isoformat())
        )
        case_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        record = {
            "case_id": case_id,
            "stratum": stratum,
            "group": group,
            "primitive": primitive,
            "entity_id": entity_id,
            "lifecycle_or_outcome": lifecycle,
            "event_clock": event_clock.isoformat(),
            "strata": _strata_payload(strata),
            "source_ids": list(_source_ids(state)),
        }
        pool = self._case_pools[stratum]
        if case_id in pool:
            return
        self._case_candidate_count += 1
        pool[case_id] = record
        if len(pool) > _CASES_PER_STRATUM:
            pool.pop(max(pool))

    @staticmethod
    def _group5_terminal_classification(
        lifecycle: str,
        reason: str,
    ) -> str | None:
        """Classify only terminal Group 5 paths.

        A trigger step is evidence inside the path, not proof that the causal
        path completed successfully.  Only the two registered successful
        terminal reasons are complete; every other terminal is interrupted.
        """

        if lifecycle == "closed" and reason in _GROUP5_COMPLETE_REASONS:
            return "complete"
        if lifecycle in {"closed", "censored"}:
            return "interrupted"
        return None

    @staticmethod
    def _case_stratum(
        group: str,
        primitive: str,
        lifecycle: str,
        strata: tuple[tuple[str, str], ...],
        state: Any,
    ) -> str | None:
        """Map objective transitions into the nine profile-registered strata."""

        reason = _value(_mapping_value(state, "transition_reason")) or ""
        if group == "group4" and primitive == "dealing_range":
            if lifecycle == "mature":
                return "all_recognized_mature"
            if lifecycle == "broken":
                if reason in _RANGE_REASONABLY_BROKEN_REASONS:
                    return "forming_reasonably_broken"
                if reason in _RANGE_SOURCE_OR_RESET_REASONS:
                    return "source_identity_or_reset_failure"
        if (
            group == "group4"
            and primitive == "manipulation"
            and dict(strata).get("source_kind") == "mature_range_boundary"
        ):
            return "mature_range_manipulation"
        if group == "group5" and primitive == "path_sequence":
            terminal_class = EyeAuthorityStatistics._group5_terminal_classification(
                lifecycle,
                reason,
            )
            if terminal_class == "complete":
                return "group5_complete_path"
            if terminal_class == "interrupted":
                return "group5_interrupted_path"
        return None

    def _record_entity(
        self,
        *,
        group: str,
        primitive: str,
        entity_id: Any,
        lifecycle: Any,
        transition_clock: Any,
        strata: Mapping[str, Any],
        reason: Any = None,
        state: Any,
        case_candidate: bool = True,
    ) -> bool:
        identity = str(entity_id or "")
        lifecycle_text = _value(lifecycle)
        event_clock = _clock(transition_clock)
        if not identity or lifecycle_text is None or event_clock is None:
            return False
        if not self._in_window(event_clock):
            return False
        if self._is_left_boundary_entity(
            group=group,
            primitive=primitive,
            entity_id=identity,
            state=state,
        ):
            return False
        entity_key = (group, primitive, identity)
        fingerprint = (lifecycle_text, event_clock.isoformat())
        terminal = self._terminal_transition_by_entity.get(entity_key)
        if terminal is not None:
            if terminal != fingerprint:
                raise ValueError(
                    "eye statistics received a lifecycle after a terminal "
                    f"transition: {entity_key} ({terminal}, {fingerprint})"
                )
            return False
        if self._last_transition_by_entity.get(entity_key) == fingerprint:
            return False
        observed_strata = _strata_key(strata)
        exact_strata = self._cohort_strata_by_entity.setdefault(
            entity_key,
            observed_strata,
        )
        if observed_strata != exact_strata:
            self._cohort_strata_revision_counts[(group, primitive)] += 1
        reason_text = _value(reason)
        self._sequence += 1
        latest = (
            self._sequence,
            lifecycle_text,
            exact_strata,
            reason_text,
        )
        is_terminal = (
            primitive in _IMMUTABLE_EVENT_PRIMITIVES
            or lifecycle_text
            in _TERMINAL_LIFECYCLES.get(primitive, frozenset())
        )
        if is_terminal:
            self._terminal_transition_by_entity[entity_key] = fingerprint
            self._last_transition_by_entity.pop(entity_key, None)
            self._open_latest.pop(entity_key, None)
            self._cohort_strata_by_entity.pop(entity_key, None)
        else:
            self._last_transition_by_entity[entity_key] = fingerprint
            self._open_latest[entity_key] = latest
        for key in self._funnel_keys(group, primitive, exact_strata):
            self._cohort_entities[key].add(identity)
            self._lifecycle_entities[(*key, lifecycle_text)].add(identity)
            self._lifecycle_transitions[(*key, lifecycle_text)] += 1
            if is_terminal:
                self._terminal_latest_counts[
                    (key[0], key[1], key[2], lifecycle_text)
                ] += 1
            if reason_text:
                self._reason_counts[(*key, reason_text)] += 1
        if case_candidate:
            self._add_case(
                group=group,
                primitive=primitive,
                entity_id=identity,
                lifecycle=lifecycle_text,
                event_clock=event_clock,
                strata=exact_strata,
                state=state,
            )
        return True

    def _is_left_boundary_entity(
        self,
        *,
        group: str,
        primitive: str,
        entity_id: str,
        state: Any,
    ) -> bool:
        """Exclude cohorts whose causal origin predates the report window."""

        if self.start is None:
            return False
        key = (group, primitive, entity_id)
        origin = next(
            (
                clock
                for name in (
                    "formed_at",
                    "pending_at",
                    "started_at",
                    "confirmed_at",
                )
                if (clock := _clock(_mapping_value(state, name))) is not None
            ),
            None,
        )
        if key not in self._warmup_entity_keys and (
            origin is None or origin >= self.start
        ):
            return False
        self._report_left_boundary_key(key)
        return True

    def _report_left_boundary_key(
        self,
        key: tuple[str, str, str],
    ) -> None:
        """Record a left-censored identity once when it reaches the window."""

        if key not in self._left_boundary_entities:
            self._left_boundary_entities.add(key)
            self._left_boundary_counts[key[:2]] += 1

    def _categorical_add(
        self,
        group: str,
        primitive: str,
        dimension: str,
        value: Any,
        *,
        strata: Mapping[str, Any] | None = None,
    ) -> None:
        text = _value(value)
        if text is None:
            return
        key = (group, primitive, dimension, _strata_key(strata or {}))
        self._categorical[key][text] += 1

    def _numeric_add(
        self,
        group: str,
        primitive: str,
        lifecycle: str,
        metric: str,
        value: Any,
        *,
        strata: Mapping[str, Any] | None = None,
    ) -> None:
        key = (
            group,
            primitive,
            lifecycle,
            metric,
            _strata_key(strata or {}),
        )
        self._numeric[key].add(value)

    def _observe_clock(
        self,
        update: Any,
        observation: MarketObservation,
        previous: MarketObservation | None,
    ) -> None:
        newly = (
            _mapping_value(update, "newly_completed", {})
            if update is not None
            else {}
        )
        update_anomalies = (
            _mapping_value(update, "anomalies", ())
            if update is not None
            else ()
        )
        reader_anomalies = {str(value) for value in update_anomalies}
        self._reader_anomalies.update(reader_anomalies)
        execution_anomalies = {
            str(value)
            for value in _mapping_value(observation.execution, "anomalies", ())
        }
        self._observation_anomalies.update(
            str(value)
            for value in observation.anomalies
            if (
                str(value) not in reader_anomalies
                and not self._is_execution_anomaly(
                    str(value),
                    execution_anomalies,
                )
            )
        )

        prior_frames = {} if previous is None else previous.frames
        for timeframe, frame in observation.frames.items():
            tf = _value(timeframe) or str(timeframe)
            state = self._clocks[tf]
            state["observations"] += 1
            if bool(frame.ready):
                state["ready_observations"] += 1
                if tf not in self._ready_seen:
                    self._ready_seen.add(tf)
                    state["ready_transitions"] += 1
            else:
                state["not_ready_observations"] += 1
            cutoff = _clock(frame.cutoff)
            if cutoff is not None:
                state["first_cutoff"] = state["first_cutoff"] or cutoff
                state["last_cutoff"] = cutoff

            completed = (
                tuple(newly.get(timeframe, ()))
                if isinstance(newly, Mapping)
                else ()
            )
            if update is not None:
                state["completion_denominator_status"] = "reader_update"
                for item in completed:
                    item_clock = _mapping_value(item, "end", observation.asof)
                    if not self._in_window(item_clock):
                        continue
                    completion_kind = (
                        "real"
                        if bool(_mapping_value(item, "real_completed", True))
                        else "synthetic"
                    )
                    state[f"{completion_kind}_completed"] += 1
                    self._record_reducer_updates(tf, completion_kind)
            else:
                prior = (
                    prior_frames.get(timeframe)
                    if isinstance(prior_frames, Mapping)
                    else None
                )
                changed = prior is None or _clock(prior.cutoff) != cutoff
                if changed and frame.candle_structure is not None:
                    state["completion_denominator_status"] = "cutoff_fallback"
                    completion_kind = (
                        "real"
                        if bool(frame.candle_structure.real_completed)
                        else "synthetic"
                    )
                    state[f"{completion_kind}_completed"] += 1
                    self._record_reducer_updates(tf, completion_kind)

            candle = frame.candle_structure
            cutoff_text = None if cutoff is None else cutoff.isoformat()
            if (
                candle is not None
                and cutoff_text is not None
                and self._last_classified_candle_cutoff.get(tf) != cutoff_text
            ):
                self._last_classified_candle_cutoff[tf] = cutoff_text
                candle_strata = {"timeframe": tf}
                for dimension in (
                    "body_class",
                    "range_class",
                    "dominant_wick",
                    "close_class",
                ):
                    self._categorical_add(
                        "group12",
                        "candle_structure",
                        dimension,
                        _mapping_value(candle, dimension),
                        strata=candle_strata,
                    )

    @staticmethod
    def _is_execution_anomaly(
        value: str,
        execution_anomalies: set[str],
    ) -> bool:
        if value in execution_anomalies:
            return True
        return value.startswith(
            ("execution_", "spread_", "fillability_", "mbo_", "cost_")
        )

    def _record_reducer_updates(self, timeframe: str, kind: str) -> None:
        components = ["group12_structure_liquidity"]
        if timeframe == Timeframe.M5.value:
            components.extend(("displacement", "group3_zones"))
        elif timeframe == Timeframe.H1.value:
            components.append("group4_range")
        elif timeframe == Timeframe.M1.value:
            components.extend(("group4_manipulation", "group5_entry"))
        for component in components:
            self._reducer_updates[(component, timeframe, kind)] += 1

    def _remember_warmup_entity(
        self,
        *,
        group: str,
        primitive: str,
        entity_id: Any,
        lifecycle: Any,
    ) -> None:
        identity = str(entity_id or "")
        lifecycle_text = _value(lifecycle)
        if (
            identity
            and lifecycle_text in _OPEN_LIFECYCLES.get(primitive, frozenset())
        ):
            self._warmup_entity_keys.add((group, primitive, identity))

    def _cache_baseline(self, observation: MarketObservation) -> None:
        """Retain warmup readiness, source identity, and live cohort identities."""

        for timeframe, frame in observation.frames.items():
            tf = _value(timeframe) or str(timeframe)
            if bool(frame.ready):
                self._ready_seen.add(tf)
            cutoff = _clock(frame.cutoff)
            if cutoff is not None:
                self._last_classified_candle_cutoff[tf] = cutoff.isoformat()
                self._last_typed_frame_cutoff[tf] = cutoff.isoformat()
            for primitive, collection_name, identity_name in (
                ("swing", "swings", "swing_id"),
                ("structure", "structures", "structure_id"),
                ("bos", "structure_breaks", "bos_id"),
                (
                    "support_resistance",
                    "support_resistance",
                    "zone_id",
                ),
                ("fvg", "fair_value_gaps", "fvg_id"),
                ("order_block", "order_blocks", "order_block_id"),
                ("dealing_range", "dealing_ranges", "range_id"),
            ):
                group = "group12"
                if primitive in {"fvg", "order_block"}:
                    group = "group3"
                elif primitive == "dealing_range":
                    group = "group4"
                for state in tuple(_mapping_value(frame, collection_name, ())):
                    self._remember_warmup_entity(
                        group=group,
                        primitive=primitive,
                        entity_id=_mapping_value(state, identity_name),
                        lifecycle=_mapping_value(state, "lifecycle"),
                    )
                    if (
                        primitive == "dealing_range"
                        and _value(_mapping_value(state, "lifecycle"))
                        == "mature"
                        and (
                            range_id := str(
                                _mapping_value(state, "range_id") or ""
                            )
                        )
                        and (
                            lower_bound := _finite(
                                _mapping_value(state, "lower_bound")
                            )
                        )
                        is not None
                        and (
                            upper_bound := _finite(
                                _mapping_value(state, "upper_bound")
                            )
                        )
                        is not None
                    ):
                        midpoint = _finite(_mapping_value(state, "midpoint"))
                        self._mature_range_details.setdefault(
                            range_id,
                            {
                                "lower_bound": lower_bound,
                                "upper_bound": upper_bound,
                                "midpoint": (
                                    midpoint
                                    if midpoint is not None
                                    else (lower_bound + upper_bound) / 2.0
                                ),
                                "mature_at": _clock(
                                    _mapping_value(state, "mature_at")
                                ),
                                "broken_at": None,
                            },
                        )

        displacement = _mapping_value(observation, "displacement")
        if displacement is not None:
            self._remember_warmup_entity(
                group="displacement",
                primitive="displacement",
                entity_id=_mapping_value(displacement, "current_entity_id"),
                lifecycle=_mapping_value(displacement, "lifecycle"),
            )

        for group, primitive, collection_name, identity_name in (
            (
                "group12",
                "liquidity_pool",
                "liquidity_pool_states",
                "pool_id",
            ),
            ("group4", "manipulation", "manipulations", "manipulation_id"),
            ("group5", "entry_location", "entry_locations", "location_id"),
            (
                "group5",
                "qualified_reacceptance",
                "qualified_reacceptances",
                "reacceptance_id",
            ),
            ("group5", "path_sequence", "path_sequences", "sequence_id"),
        ):
            for state in tuple(_mapping_value(observation, collection_name, ())):
                if primitive == "liquidity_pool":
                    self._index_pool_metadata(state)
                elif primitive == "manipulation":
                    self._index_manipulation_metadata(state)
                self._remember_warmup_entity(
                    group=group,
                    primitive=primitive,
                    entity_id=_mapping_value(state, identity_name),
                    lifecycle=_mapping_value(state, "lifecycle"),
                )

    def _observe_group12_frame(self, frame: Any, asof: pd.Timestamp) -> None:
        timeframe = _value(frame.timeframe)
        for state in frame.swings:
            lifecycle = _value(state.lifecycle)
            if lifecycle is None:
                continue
            event_clock = self._transition_clock(state, lifecycle, asof)
            new = self._record_entity(
                group="group12",
                primitive="swing",
                entity_id=state.swing_id,
                lifecycle=lifecycle,
                transition_clock=event_clock,
                strata={"timeframe": timeframe, "side": state.side},
                reason=state.failure_reason,
                state=state,
            )
            if (
                new
                and lifecycle == "confirmed"
                and _value(state.relation) not in {None, "none"}
            ):
                self._categorical_add(
                    "group12",
                    "swing",
                    "relation",
                    state.relation,
                    strata={"timeframe": timeframe, "side": state.side},
                )
                self._numeric_add(
                    "group12",
                    "swing",
                    lifecycle,
                    "magnitude_atr",
                    state.magnitude_atr,
                    strata={"timeframe": timeframe, "side": state.side},
                )

        for state in frame.structures:
            if not _mapping_value(state, "structure_id"):
                continue
            lifecycle = _value(state.lifecycle)
            if lifecycle is None:
                continue
            event_clock = self._transition_clock(state, lifecycle, asof)
            self._record_entity(
                group="group12",
                primitive="structure",
                entity_id=state.structure_id,
                lifecycle=lifecycle,
                transition_clock=event_clock,
                strata={"timeframe": timeframe, "direction": state.direction},
                reason=state.failure_reason,
                state=state,
            )

        for state in frame.structure_breaks:
            lifecycle = _value(state.lifecycle)
            if lifecycle is None:
                continue
            event_clock = self._transition_clock(state, lifecycle, asof)
            new = self._record_entity(
                group="group12",
                primitive="bos",
                entity_id=state.bos_id,
                lifecycle=lifecycle,
                transition_clock=event_clock,
                strata={
                    "timeframe": timeframe,
                    "direction": state.direction,
                    "scope": state.scope,
                },
                reason=state.failure_reason,
                state=state,
            )
            if new and lifecycle == "confirmed":
                self._categorical_add(
                    "group12",
                    "bos",
                    "scope",
                    state.scope,
                    strata={"timeframe": timeframe, "direction": state.direction},
                )
                if (
                    timeframe == Timeframe.M5.value
                    and _value(_mapping_value(state, "scope")) == "opposed"
                    and bool(_mapping_value(state, "mss_qualified", False))
                    and (
                        displacement_id := str(
                            _mapping_value(state, "source_displacement_id") or ""
                        )
                    )
                    and (
                        resolved_at := _clock(
                            _mapping_value(state, "resolved_at")
                        )
                    )
                    is not None
                ):
                    self._opposed_mss_by_displacement[displacement_id][
                        str(_mapping_value(state, "bos_id"))
                    ] = {
                        "resolved_at": resolved_at,
                        "direction": _value(_mapping_value(state, "direction")),
                    }
            post_state = _value(_mapping_value(state, "post_break_state"))
            post_clock = _mapping_value(state, "accepted_at") or _mapping_value(
                state, "rejected_at"
            )
            if post_state not in {None, "pending"} and post_clock is not None:
                self._record_entity(
                    group="group12",
                    primitive="bos_post_break",
                    entity_id=f"{state.bos_id}:{post_state}",
                    lifecycle=post_state,
                    transition_clock=post_clock,
                    strata={
                        "timeframe": timeframe,
                        "direction": state.direction,
                        "scope": state.scope,
                    },
                    reason=f"post_break_{post_state}",
                    state=state,
                )

        for state in frame.support_resistance:
            lifecycle = _value(state.lifecycle)
            if lifecycle is None:
                continue
            self._record_entity(
                group="group12",
                primitive="support_resistance",
                entity_id=state.zone_id,
                lifecycle=lifecycle,
                transition_clock=self._transition_clock(state, lifecycle, asof),
                strata={
                    "timeframe": timeframe,
                    "side": state.side,
                    "source_kind": state.source_kind,
                    "structural_rank": state.structural_rank,
                },
                reason=state.transition_reason,
                state=state,
            )

        # Frame pool projections can lag the all-scale 1m crossing projection
        # carried by ``MarketObservation.liquidity_pool_states``.  Counting
        # both paths can replay a stale FORMED frame after the authoritative
        # aggregate has already reached ACCEPTED/REJECTED.  Pools therefore
        # have one statistics authority: the observation-level collection (or
        # its typed transition delta) consumed by ``observe()``.

    def _index_pool_metadata(self, state: Any) -> tuple[str, str]:
        """Retain descriptive source rank without counting a lifecycle."""

        pool_id = str(_mapping_value(state, "pool_id") or "")
        metadata = self._source_metadata.get(f"pool:{pool_id}")
        structural_rank = (
            "unknown"
            if metadata is None
            else metadata.get("structural_rank", "unknown")
        )
        internal_external = (
            structural_rank
            if structural_rank in {"internal", "external"}
            else "unknown"
        )
        if metadata is None:
            self._pool_rank_missing.add(pool_id)
        else:
            self._pool_rank_exposed.add(pool_id)
            self._pool_rank_missing.discard(pool_id)
        return structural_rank, internal_external

    def _observe_pool(self, state: Any, asof: pd.Timestamp) -> None:
        lifecycle = _value(state.lifecycle)
        if lifecycle is None:
            return
        structural_rank, internal_external = self._index_pool_metadata(state)
        self._record_entity(
            group="group12",
            primitive="liquidity_pool",
            entity_id=state.pool_id,
            lifecycle=lifecycle,
            transition_clock=self._transition_clock(state, lifecycle, asof),
            strata={
                "timeframe": state.timeframe,
                "side": state.side,
                "structural_rank": structural_rank,
                "internal_external": internal_external,
            },
            reason=state.resolution_reason,
            state=state,
        )

    def _observe_inventory(self, state: Any, asof: pd.Timestamp) -> None:
        lifecycle = _value(state.lifecycle)
        if lifecycle is None:
            return
        self._record_entity(
            group="group12",
            primitive="liquidity_inventory",
            entity_id=state.item_id,
            lifecycle=lifecycle,
            transition_clock=self._transition_clock(state, lifecycle, asof),
            strata={
                "timeframe": state.timeframe,
                "side": state.side,
                "source_kind": state.kind,
                "structural_rank": state.structural_rank,
            },
            reason=state.lifecycle_reason,
            state=state,
            case_candidate=lifecycle == "consumed",
        )

    def _observe_displacement(self, displacement: Any) -> None:
        if displacement is None:
            return
        transitions = tuple(
            transition
            for transition in displacement.transitions_this_update
            if self._in_window(transition.observed_at)
        )
        transitions = tuple(
            transition
            for transition in transitions
            if not self._is_left_boundary_entity(
                group="displacement",
                primitive="displacement",
                entity_id=str(transition.entity_id),
                state=transition,
            )
        )
        self._displacement_same_bar_restarts += sum(
            left.lifecycle in {"exhausted", "censored"}
            and right.lifecycle == "started"
            and left.observed_at == right.observed_at
            and left.entity_id != right.entity_id
            for left, right in zip(transitions[:-1], transitions[1:])
        )
        for transition in transitions:
            lifecycle = _value(transition.lifecycle)
            if lifecycle is None:
                continue
            new = self._record_entity(
                group="displacement",
                primitive="displacement",
                entity_id=transition.entity_id,
                lifecycle=lifecycle,
                transition_clock=transition.observed_at,
                strata={"timeframe": "5m", "direction": transition.direction},
                reason=transition.reason,
                state=transition,
            )
            if not new:
                continue
            transition_metrics = dict(
                tuple(_mapping_value(transition, "state_metrics", ()))
            )
            activation_ratios = {
                name: number
                for name in _DISPLACEMENT_ACTIVATION_RATIO_METRICS
                if (number := _finite(transition_metrics.get(name))) is not None
            }
            if activation_ratios:
                self._displacement_latest_activation_ratios[
                    str(transition.entity_id)
                ] = activation_ratios
            if lifecycle == "started":
                self._displacement_started.add(transition.entity_id)
            elif lifecycle == "active":
                self._displacement_active.add(transition.entity_id)
            elif lifecycle in {"exhausted", "censored"}:
                terminal_reason = _value(transition.reason) or lifecycle
                self._displacement_terminal_reason[
                    transition.entity_id
                ] = terminal_reason
                started_at = _clock(transition.started_at)
                terminal_at = _clock(transition.terminal_at)
                if started_at is not None and terminal_at is not None:
                    duration_minutes = (
                        terminal_at - started_at
                    ).total_seconds() / 60.0
                    self._displacement_durations.add(duration_minutes)
                    self._numeric_add(
                        "displacement",
                        "displacement",
                        "terminal",
                        "duration_minutes",
                        duration_minutes,
                        strata={"direction": transition.direction},
                    )
            for name, value in transition.state_metrics:
                if (
                    lifecycle in {"exhausted", "censored"}
                    and str(name) == "total_interruption_bars"
                ):
                    self._displacement_interruptions.add(value)
                self._numeric_add(
                    "displacement",
                    "displacement",
                    lifecycle,
                    str(name),
                    value,
                    strata={"direction": transition.direction},
                )
        current_entity_id = str(
            _mapping_value(displacement, "current_entity_id") or ""
        )
        if current_entity_id in self._displacement_started:
            current_metrics = dict(
                tuple(_mapping_value(displacement, "current_metrics", ()) or ())
            )
            current_ratios = {
                name: number
                for name in _DISPLACEMENT_ACTIVATION_RATIO_METRICS
                if (number := _finite(current_metrics.get(name))) is not None
            }
            if current_ratios:
                self._displacement_latest_activation_ratios[
                    current_entity_id
                ] = current_ratios

    def _observe_fvg(self, state: Any, asof: pd.Timestamp) -> None:
        lifecycle = _value(state.lifecycle)
        if lifecycle is None:
            return
        new = self._record_entity(
            group="group3",
            primitive="fvg",
            entity_id=state.fvg_id,
            lifecycle=lifecycle,
            transition_clock=self._transition_clock(state, lifecycle, asof),
            strata={
                "timeframe": state.timeframe,
                "direction": state.direction,
                "qualification": state.qualification,
            },
            reason=state.transition_reason,
            state=state,
        )
        fvg_id = str(_mapping_value(state, "fvg_id") or "")
        if (
            new
            and _value(_mapping_value(state, "qualification"))
            == "displacement_linked"
            and fvg_id not in self._group5_qualified_zone_metadata
        ):
            metadata = (
                "fvg",
                _value(_mapping_value(state, "timeframe")) or "unknown",
            )
            self._group5_qualified_zone_metadata[fvg_id] = metadata
            self._group5_zone_funnel_ids[(*metadata, "qualified_zone")].add(
                fvg_id
            )
            self._group5_qualified_zone_details[fvg_id] = {
                "kind": metadata[0],
                "timeframe": metadata[1],
                "direction": _value(_mapping_value(state, "direction")),
                "source_displacement_id": str(
                    _mapping_value(state, "source_displacement_id") or ""
                ),
                "confirmed_at": _clock(
                    _mapping_value(state, "confirmed_at")
                    or _mapping_value(state, "formed_at")
                ),
                "source_displacement_active_at": _clock(
                    _mapping_value(state, "source_displacement_active_at")
                ),
                "lower_bound": _finite(_mapping_value(state, "lower_bound")),
                "upper_bound": _finite(_mapping_value(state, "upper_bound")),
            }

    def _observe_order_block_funnel(self, values: Iterable[Any]) -> None:
        for snapshot in values:
            observed_at = _clock(_mapping_value(snapshot, "observed_at"))
            if observed_at is None or not self._in_window(observed_at):
                continue
            if (
                self._last_ob_funnel_clock is not None
                and observed_at < self._last_ob_funnel_clock
            ):
                raise ValueError("Order Block funnel clocks are out of order")
            if observed_at == self._last_ob_funnel_clock:
                continue
            self._last_ob_funnel_clock = observed_at
            stages = tuple(_mapping_value(snapshot, "stages", ()))
            outcome = _value(_mapping_value(snapshot, "outcome"))
            if not stages or outcome is None:
                raise ValueError("Order Block funnel snapshot is incomplete")
            self._ob_funnel_attempts += 1
            self._ob_funnel_outcomes[outcome] += 1
            for name, count in stages:
                if not isinstance(name, str) or type(count) is not int or count < 0:
                    raise ValueError("Order Block funnel stage is invalid")
                self._ob_funnel_stages[name] += count

    def _observe_order_block(self, state: Any, asof: pd.Timestamp) -> None:
        lifecycle = _value(state.lifecycle)
        if lifecycle is None:
            return
        new = self._record_entity(
            group="group3",
            primitive="order_block",
            entity_id=state.order_block_id,
            lifecycle=lifecycle,
            transition_clock=self._transition_clock(state, lifecycle, asof),
            strata={
                "timeframe": state.timeframe,
                "direction": state.direction,
                "bos_scope": state.source_bos_scope,
                "mss_qualified": state.source_bos_mss_qualified,
            },
            reason=state.transition_reason,
            state=state,
        )
        order_block_id = str(_mapping_value(state, "order_block_id") or "")
        if new and order_block_id not in self._group5_qualified_zone_metadata:
            metadata = (
                "order_block",
                _value(_mapping_value(state, "timeframe")) or "unknown",
            )
            self._group5_qualified_zone_metadata[order_block_id] = metadata
            self._group5_zone_funnel_ids[(*metadata, "qualified_zone")].add(
                order_block_id
            )
            self._group5_qualified_zone_details[order_block_id] = {
                "kind": metadata[0],
                "timeframe": metadata[1],
                "direction": _value(_mapping_value(state, "direction")),
                "source_displacement_id": str(
                    _mapping_value(state, "source_displacement_id") or ""
                ),
                "confirmed_at": _clock(
                    _mapping_value(state, "confirmed_at")
                    or _mapping_value(state, "formed_at")
                ),
                "source_displacement_active_at": _clock(
                    _mapping_value(state, "source_displacement_active_at")
                ),
                "lower_bound": _finite(_mapping_value(state, "lower_bound")),
                "upper_bound": _finite(_mapping_value(state, "upper_bound")),
            }

    def _observe_range(self, state: Any, asof: pd.Timestamp) -> None:
        lifecycle = _value(state.lifecycle)
        if lifecycle is None:
            return
        range_id = str(_mapping_value(state, "range_id") or "")
        if self._is_left_boundary_entity(
            group="group4",
            primitive="dealing_range",
            entity_id=range_id,
            state=state,
        ):
            # Neither the terminal transition nor its evolving maturity-gate
            # metrics belong to a cohort whose origin predates the window.
            return
        new = self._record_entity(
            group="group4",
            primitive="dealing_range",
            entity_id=range_id,
            lifecycle=lifecycle,
            transition_clock=self._transition_clock(state, lifecycle, asof),
            strata={"timeframe": state.timeframe},
            reason=state.transition_reason,
            state=state,
        )
        if (
            new
            and lifecycle == "mature"
            and (
                lower_bound := _finite(_mapping_value(state, "lower_bound"))
            )
            is not None
            and (
                upper_bound := _finite(_mapping_value(state, "upper_bound"))
            )
            is not None
        ):
            midpoint = _finite(_mapping_value(state, "midpoint"))
            self._mature_range_details.setdefault(
                range_id,
                {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "midpoint": (
                        midpoint
                        if midpoint is not None
                        else (lower_bound + upper_bound) / 2.0
                    ),
                    "mature_at": _clock(_mapping_value(state, "mature_at")),
                    "broken_at": None,
                },
            )
        if new and lifecycle == "broken" and range_id in self._mature_range_details:
            self._mature_range_details[range_id]["broken_at"] = _clock(
                _mapping_value(state, "broken_at")
            )
        update_clock = _clock(
            _mapping_value(state, "last_updated_at")
            or _mapping_value(state, "state_started_at")
            or asof
        )
        update_text = None if update_clock is None else update_clock.isoformat()
        gate_sample = (
            update_clock is not None
            and self._in_window(update_clock)
            and self._last_range_gate_clock.get(str(state.range_id)) != update_text
        )
        if gate_sample:
            self._last_range_gate_clock[str(state.range_id)] = str(update_text)
            for metric in (
                "candidate_real_h1_bars",
                "lower_touch_count",
                "upper_touch_count",
                "midpoint_crossings",
                "inside_close_fraction",
                "width_atr_at_formation",
                "compression_ratio",
            ):
                self._numeric_add(
                    "group4",
                    "dealing_range",
                    lifecycle,
                    metric,
                    _mapping_value(state, metric),
                    strata={"timeframe": state.timeframe},
                )
        formation_atr = _finite(_mapping_value(state, "formation_atr"))
        if (
            new
            and formation_atr is not None
            and formation_atr > 0.0
            and range_id not in self._range_formation_atr_by_id
        ):
            self._range_formation_atr_by_id[range_id] = formation_atr
        if (
            new
            and lifecycle == "mature"
            and state.range_id not in self._seen_mature_ranges
        ):
            self._seen_mature_ranges.add(state.range_id)
            mature_at = _clock(state.mature_at)
            if mature_at is not None:
                month = mature_at.tz_convert("America/New_York").strftime("%Y-%m")
                self._mature_months[month] += 1

    def _observe_range_funnel(self, values: Iterable[Any]) -> None:
        for snapshot in values:
            observed_at = _clock(_mapping_value(snapshot, "observed_at"))
            if observed_at is None or not self._in_window(observed_at):
                continue
            if (
                self._last_range_funnel_clock is not None
                and observed_at < self._last_range_funnel_clock
            ):
                raise ValueError("range funnel clocks are out of order")
            if observed_at == self._last_range_funnel_clock:
                continue
            self._last_range_funnel_clock = observed_at
            self._range_funnel_snapshots += 1
            pair_counts = tuple(_mapping_value(snapshot, "pair_counts", ()))
            if not pair_counts:
                raise ValueError("range funnel pair counts are absent")
            for name, count in pair_counts:
                if not isinstance(name, str) or type(count) is not int or count < 0:
                    raise ValueError("range funnel pair count is invalid")
                self._range_pair_counts[name] += count

            maturity_range_id = _mapping_value(snapshot, "maturity_range_id")
            gates = tuple(_mapping_value(snapshot, "maturity_gates", ()))
            unmet = tuple(_mapping_value(snapshot, "unmet_maturity_gates", ()))
            if maturity_range_id is None:
                if gates or unmet:
                    raise ValueError("unevaluated range funnel carries maturity gates")
                continue
            range_key = (
                "group4",
                "dealing_range",
                str(maturity_range_id),
            )
            if (
                range_key in self._warmup_entity_keys
                or range_key in self._left_boundary_entities
            ):
                self._report_left_boundary_key(range_key)
                self._left_boundary_range_gate_evaluations += 1
                continue
            self._range_gate_observations += 1
            for row in gates:
                if len(row) != 4:
                    raise ValueError("range maturity gate row is invalid")
                name, value, threshold, margin = row
                self._numeric_add(
                    "group4",
                    "dealing_range",
                    "maturity_evaluation",
                    f"gate_margin_{name}",
                    margin,
                    strata={"timeframe": Timeframe.H1},
                )
                self._numeric_add(
                    "group4",
                    "dealing_range",
                    "maturity_evaluation",
                    f"gate_value_{name}",
                    value,
                    strata={"timeframe": Timeframe.H1},
                )
                self._numeric_add(
                    "group4",
                    "dealing_range",
                    "maturity_evaluation",
                    f"gate_threshold_{name}",
                    threshold,
                    strata={"timeframe": Timeframe.H1},
                )
            combination = "+".join(unmet) if unmet else "none"
            size = (
                "none"
                if not unmet
                else "one"
                if len(unmet) == 1
                else "two"
                if len(unmet) == 2
                else "three_plus"
            )
            finite_margins = {
                str(name): value
                for name, _, _, margin in gates
                if (value := _finite(margin)) is not None
            }
            weakest_gate = (
                min(finite_margins, key=lambda name: (finite_margins[name], name))
                if finite_margins
                else None
            )
            episode_diagnostic = {
                "observed_at": observed_at.isoformat(),
                "unmet_gate_combination": combination,
                "unmet_gate_cardinality": size,
                "weakest_gate": weakest_gate,
                "weakest_margin": (
                    None if weakest_gate is None else finite_margins[weakest_gate]
                ),
            }
            range_identity = str(maturity_range_id)
            self._range_gate_first_by_id.setdefault(
                range_identity,
                episode_diagnostic,
            )
            self._range_gate_latest_by_id[range_identity] = episode_diagnostic
            self._range_unmet_combinations[combination] += 1
            self._range_unmet_sizes[size] += 1
            if len(unmet) == 1:
                case_stratum = "near_mature_single_gate"
            elif len(unmet) >= 2:
                case_stratum = "multiple_gate_rejected"
            else:
                # The typed MATURE transition supplies the canonical case;
                # do not duplicate it with the same-clock gate snapshot.
                case_stratum = None
            if case_stratum is None:
                continue
            self._add_named_case(
                stratum=case_stratum,
                group="group4",
                primitive="range_maturity_evaluation",
                entity_id=str(maturity_range_id),
                lifecycle=combination,
                event_clock=observed_at,
                strata=_strata_key({"timeframe": Timeframe.H1}),
                state=snapshot,
            )

    def _manipulation_bucket(self, state: Any) -> str:
        lifecycle = _value(state.lifecycle) or "swept"
        if bool(_mapping_value(state, "deadline_elapsed", False)):
            return "deadline_censored"
        if _mapping_value(state, "censored_at") is not None:
            reason = _value(_mapping_value(state, "transition_reason")) or ""
            if reason in {
                "data_gap_reset",
                "contract_change_reset",
                "data_anomaly",
                "tick_size_mismatch",
            }:
                return "hard_boundary_censored"
        return lifecycle

    def _index_manipulation_metadata(
        self,
        state: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Retain source identity context without counting a lifecycle."""

        source_inventory_id = str(
            _mapping_value(state, "source_inventory_item_id") or ""
        )
        source_metadata = self._source_metadata.get(source_inventory_id, {})
        structural_rank = source_metadata.get("structural_rank", "unknown")
        internal_external = (
            structural_rank
            if structural_rank in {"internal", "external"}
            else "unknown"
        )
        manipulation_id = str(_mapping_value(state, "manipulation_id") or "")
        metadata = {
            "source_timeframe": (
                _value(_mapping_value(state, "source_timeframe")) or "unknown"
            ),
            "source_kind": (
                _value(_mapping_value(state, "source_kind")) or "unknown"
            ),
            "source_id": str(_mapping_value(state, "source_id") or ""),
            "source_inventory_item_id": source_inventory_id,
            "side": _value(_mapping_value(state, "side")) or "unknown",
            "swept_at": _clock(_mapping_value(state, "swept_at")),
            "reaccepted_at": _clock(_mapping_value(state, "reaccepted_at")),
            "structural_rank": structural_rank,
            "internal_external": internal_external,
        }
        self._manipulation_metadata[manipulation_id] = metadata
        return manipulation_id, metadata

    def _observe_manipulation(self, state: Any, asof: pd.Timestamp) -> None:
        bucket = self._manipulation_bucket(state)
        manipulation_id, metadata = self._index_manipulation_metadata(state)
        structural_rank = metadata["structural_rank"]
        internal_external = metadata["internal_external"]
        clock_value = (
            _mapping_value(state, "censored_at")
            if bucket.endswith("censored")
            else self._transition_clock(state, bucket, asof)
        )
        new = self._record_entity(
            group="group4",
            primitive="manipulation",
            entity_id=state.manipulation_id,
            lifecycle=bucket,
            transition_clock=clock_value,
            strata={
                "timeframe": state.timeframe,
                "side": state.side,
                "source_kind": state.source_kind,
                "source_timeframe": state.source_timeframe,
                "structural_rank": structural_rank,
                "internal_external": internal_external,
            },
            reason=state.transition_reason,
            state=state,
        )
        if new and manipulation_id not in self._manipulation_cohort_ids:
            self._manipulation_cohort_ids.add(manipulation_id)
            self._group5_manipulation_funnel_ids[
                (
                    metadata["source_kind"],
                    metadata["source_timeframe"],
                    "manipulation",
                )
            ].add(manipulation_id)
        if (
            new
            and bucket == "reaccepted"
            and metadata["source_kind"]
            == "mature_range_boundary"
        ):
            self._favr_reaccepted_manipulation_ids.add(manipulation_id)
        if new and manipulation_id not in self._manipulation_metrics_seen:
            self._manipulation_metrics_seen.add(manipulation_id)
            strata = {
                "source_kind": state.source_kind,
                "source_timeframe": state.source_timeframe,
                "side": state.side,
                "structural_rank": structural_rank,
                "internal_external": internal_external,
            }
            self._numeric_add(
                "group4",
                "manipulation",
                "created_episode",
                "penetration_atr",
                _mapping_value(state, "penetration_atr"),
                strata=strata,
            )
            swept_clock = _clock(_mapping_value(state, "swept_at"))
            for clock_name, metric_name in (
                ("source_formed_at", "source_age_from_formed_minutes"),
                ("source_eligible_at", "source_age_from_eligible_minutes"),
            ):
                source_clock = _clock(_mapping_value(state, clock_name))
                if source_clock is not None and swept_clock is not None:
                    self._numeric_add(
                        "group4",
                        "manipulation",
                        "created_episode",
                        metric_name,
                        (swept_clock - source_clock).total_seconds() / 60.0,
                        strata=strata,
                    )
            self._numeric_add(
                "group4",
                "manipulation",
                "created_episode",
                "coincident_source_count",
                len(tuple(_mapping_value(state, "coincident_source_ids", ()))),
                strata=strata,
            )

    def _observe_entry_location(self, state: Any, asof: pd.Timestamp) -> None:
        lifecycle = _value(state.lifecycle)
        if lifecycle is None:
            return
        new = self._record_entity(
            group="group5",
            primitive="entry_location",
            entity_id=state.location_id,
            lifecycle=lifecycle,
            transition_clock=self._transition_clock(state, lifecycle, asof),
            strata={
                "direction": state.direction,
                "source_zone_kind": state.source_zone_kind,
            },
            reason=state.transition_reason,
            state=state,
        )
        location_id = str(_mapping_value(state, "location_id") or "")
        source_zone_id = str(_mapping_value(state, "source_zone_id") or "")
        if new:
            self._bind_exact_identity(
                self._group5_entry_source_zone_by_id,
                child_id=location_id,
                parent_id=source_zone_id,
                conflicts=self._group5_entry_binding_conflicts,
            )
            self._group5_entry_location_details.setdefault(
                location_id,
                {
                    "source_zone_id": source_zone_id,
                    "source_displacement_id": str(
                        _mapping_value(state, "source_displacement_id") or ""
                    ),
                    "direction": _value(_mapping_value(state, "direction")),
                    "formed_at": _clock(_mapping_value(state, "formed_at")),
                    "lower_bound": _finite(_mapping_value(state, "lower_bound")),
                    "upper_bound": _finite(_mapping_value(state, "upper_bound")),
                },
            )
        metadata = self._group5_qualified_zone_metadata.get(source_zone_id)
        if (
            new
            and metadata is not None
            and location_id not in self._group5_entry_location_metadata
        ):
            self._group5_entry_location_metadata[location_id] = metadata
            self._group5_zone_funnel_ids[
                (*metadata, "entry_location")
            ].add(location_id)

    def _observe_reacceptance(self, state: Any, asof: pd.Timestamp) -> None:
        lifecycle = _value(state.lifecycle)
        if lifecycle is None:
            return
        self._record_entity(
            group="group5",
            primitive="qualified_reacceptance",
            entity_id=state.reacceptance_id,
            lifecycle=lifecycle,
            transition_clock=self._transition_clock(state, lifecycle, asof),
            strata={
                "direction": state.direction,
                "context_kind": state.context_kind,
            },
            reason=state.transition_reason,
            state=state,
        )

    def _observe_micro_bos(self, state: Any) -> None:
        outcome = _value(state.outcome) or "unknown"
        self._record_entity(
            group="group5",
            primitive="micro_bos",
            entity_id=state.reference_id,
            lifecycle=outcome,
            transition_clock=state.resolved_at,
            strata={
                "direction": state.expected_direction,
                "context_kind": state.context_kind,
                "relation": state.relation,
                "qualified": state.qualified,
            },
            reason=state.outcome,
            state=state,
        )

    def _observe_path(self, state: Any, asof: pd.Timestamp) -> None:
        lifecycle = _value(state.lifecycle)
        if lifecycle is None:
            return
        sequence_id = str(_mapping_value(state, "sequence_id") or "")
        if self._is_left_boundary_entity(
            group="group5",
            primitive="path_sequence",
            entity_id=sequence_id,
            state=state,
        ):
            return
        steps = tuple(_mapping_value(state, "steps", ()))
        prior_summaries = self._path_steps_by_id.get(sequence_id, ())
        prior_step_ids = tuple(item[0] for item in prior_summaries)
        current_step_ids = tuple(
            str(_mapping_value(step, "step_id") or "") for step in steps
        )
        if (
            len(current_step_ids) < len(prior_step_ids)
            or current_step_ids[: len(prior_step_ids)] != prior_step_ids
        ):
            self._group5_path_order_errors += 1
            new_steps: tuple[Any, ...] = ()
        else:
            new_steps = steps[len(prior_step_ids) :]
        if new_steps:
            previous_clock = (
                None if not prior_summaries else prior_summaries[-1][2]
            )
            clocks = (
                previous_clock,
                *(
                    _clock(_mapping_value(step, "observed_at"))
                    for step in new_steps
                ),
            )
            ordered = all(
                left is not None and right is not None and right >= left
                for left, right in zip(clocks[:-1], clocks[1:])
            ) if prior_summaries else all(
                left is not None and right is not None and right >= left
                for left, right in zip(clocks[1:-1], clocks[2:])
            )
            predecessors_valid = all(
                (
                    index == 0
                    and not prior_step_ids
                )
                or (
                    (
                        prior_step_ids[-1]
                        if index == 0
                        else str(_mapping_value(new_steps[index - 1], "step_id"))
                    )
                    in tuple(_mapping_value(step, "predecessor_step_ids", ()))
                )
                for index, step in enumerate(new_steps)
            )
            if not ordered or not predecessors_valid:
                self._group5_path_order_errors += 1
            appended_summaries = tuple(
                (
                    str(_mapping_value(step, "step_id") or ""),
                    _value(_mapping_value(step, "kind")),
                    _clock(_mapping_value(step, "observed_at")),
                )
                for step in new_steps
            )
            self._path_steps_by_id[sequence_id] = (
                *prior_summaries,
                *appended_summaries,
            )
            if any(
                summary[1] in _GROUP5_TRIGGER_STEPS
                for summary in appended_summaries
            ):
                self._path_has_trigger_ids.add(sequence_id)
        elif sequence_id not in self._path_steps_by_id:
            self._path_steps_by_id[sequence_id] = ()
        new = self._record_entity(
            group="group5",
            primitive="path_sequence",
            entity_id=state.sequence_id,
            lifecycle=lifecycle,
            transition_clock=self._transition_clock(state, lifecycle, asof),
            strata={
                "direction": state.direction,
                "context_kind": state.context_kind,
            },
            reason=state.transition_reason,
            state=state,
        )
        if new and lifecycle in {"closed", "censored"}:
            terminal_reason = (
                _value(_mapping_value(state, "transition_reason")) or lifecycle
            )
            self._path_terminal_reasons[terminal_reason] += 1
            terminal_class = self._group5_terminal_classification(
                lifecycle,
                terminal_reason,
            )
            if terminal_class is not None:
                self._path_terminal_classes[terminal_class] += 1
                self._path_terminal_class_by_id[sequence_id] = terminal_class
        context_kind = _value(_mapping_value(state, "context_kind")) or ""
        context_id = str(_mapping_value(state, "context_id") or "")
        has_trigger = sequence_id in self._path_has_trigger_ids
        terminal_reason = (
            _value(_mapping_value(state, "transition_reason")) or lifecycle
        )
        terminal_class = self._path_terminal_class_by_id.get(sequence_id)
        if new:
            self._group5_path_directions[sequence_id] = (
                _value(_mapping_value(state, "direction")) or "unknown"
            )
            if context_kind == "zone_return":
                self._bind_exact_identity(
                    self._group5_zone_path_location_by_id,
                    child_id=sequence_id,
                    parent_id=context_id,
                    conflicts=self._group5_zone_path_binding_conflicts,
                )
            elif context_kind == "pool_reversal":
                self._bind_exact_identity(
                    self._group5_pool_path_manipulation_by_id,
                    child_id=sequence_id,
                    parent_id=context_id,
                    conflicts=self._group5_pool_path_binding_conflicts,
                )
        if context_kind == "zone_return":
            if has_trigger:
                self._group5_zone_trigger_paths.add(sequence_id)
            if terminal_class is not None:
                self._group5_zone_terminal_paths.add(sequence_id)
            metadata = self._group5_entry_location_metadata.get(context_id)
            if metadata is not None:
                if sequence_id not in self._group5_zone_path_metadata:
                    self._group5_zone_path_metadata[sequence_id] = metadata
                    self._group5_zone_funnel_ids[
                        (*metadata, "path")
                    ].add(sequence_id)
                if has_trigger:
                    self._group5_zone_funnel_ids[
                        (*metadata, "trigger")
                    ].add(sequence_id)
                if terminal_class is not None:
                    self._group5_zone_funnel_ids[
                        (*metadata, "path_terminal_all")
                    ].add(sequence_id)
                    self._group5_zone_funnel_ids[
                        (*metadata, f"terminal_{terminal_class}")
                    ].add(sequence_id)
                    if has_trigger:
                        self._group5_zone_funnel_ids[
                            (*metadata, "terminal_after_trigger")
                        ].add(sequence_id)
        elif context_kind == "pool_reversal":
            if has_trigger:
                self._group5_pool_trigger_paths.add(sequence_id)
            if terminal_class is not None:
                self._group5_pool_terminal_paths.add(sequence_id)
            source = self._manipulation_metadata.get(context_id)
            if source is not None:
                self._group5_manipulation_paths[sequence_id] = source[
                    "source_timeframe"
                ]
            if source is not None and context_id in self._manipulation_cohort_ids:
                metadata = (source["source_kind"], source["source_timeframe"])
                self._group5_manipulation_funnel_ids[
                    (*metadata, "path")
                ].add(sequence_id)
                if has_trigger:
                    self._group5_manipulation_funnel_ids[
                        (*metadata, "trigger")
                    ].add(sequence_id)
                if terminal_class is not None:
                    self._group5_manipulation_funnel_ids[
                        (*metadata, "path_terminal_all")
                    ].add(sequence_id)
                    self._group5_manipulation_funnel_ids[
                        (*metadata, f"terminal_{terminal_class}")
                    ].add(sequence_id)
                    if has_trigger:
                        self._group5_manipulation_funnel_ids[
                            (*metadata, "terminal_after_trigger")
                        ].add(sequence_id)
        for step in new_steps:
            self._record_entity(
                group="group5",
                primitive="path_step",
                entity_id=step.step_id,
                lifecycle=step.kind,
                transition_clock=step.observed_at,
                strata={
                    "direction": step.direction,
                    "context_kind": state.context_kind,
                    "same_clock_relation": step.same_clock_relation,
                },
                reason=step.reason,
                state=step,
            )

    def _observe_source_dispositions(
        self,
        values: Iterable[Any],
        *,
        inventory: Mapping[str, Any],
        fallback_asof: pd.Timestamp,
    ) -> None:
        for value in values:
            source_id = str(
                _mapping_value(value, "source_inventory_item_id")
                or _mapping_value(value, "source_id")
                or _mapping_value(value, "item_id")
                or ""
            )
            disposition = _value(_mapping_value(value, "disposition")) or ""
            if not source_id or not disposition:
                raise ValueError("Group4 source disposition lacks identity or outcome")
            if disposition not in _GROUP4_SOURCE_DISPOSITIONS:
                raise ValueError(f"unknown Group4 source disposition: {disposition}")
            observed_at = _clock(
                _mapping_value(value, "observed_at") or fallback_asof
            )
            if observed_at is None or not self._in_window(observed_at):
                continue
            source_clock_key = (observed_at.isoformat(), source_id)
            prior_disposition = self._source_disposition_by_clock_id.get(
                source_clock_key
            )
            if prior_disposition is not None:
                if prior_disposition != disposition:
                    raise ValueError(
                        "Group4 source-clock received conflicting dispositions: "
                        f"{source_clock_key} ({prior_disposition}, {disposition})"
                    )
                # An identical producer row is idempotent at its true key.
                continue
            self._source_disposition_by_clock_id[source_clock_key] = disposition
            source = inventory.get(source_id)
            metadata = self._source_metadata.get(source_id, {})
            source_kind = (
                _value(_mapping_value(value, "source_kind"))
                or metadata.get("source_kind")
                or self._group4_source_kind(_mapping_value(source, "kind"))
            )
            timeframe = (
                _value(_mapping_value(value, "source_timeframe"))
                or metadata.get("source_timeframe")
                or _value(_mapping_value(source, "timeframe"))
                or "unknown"
            )
            side = (
                _value(_mapping_value(value, "side"))
                or metadata.get("side")
                or _value(_mapping_value(source, "side"))
                or "unknown"
            )
            structural_rank = (
                metadata.get("structural_rank")
                or _value(_mapping_value(source, "structural_rank"))
                or "unknown"
            )
            internal_external = (
                structural_rank
                if structural_rank in {"internal", "external"}
                else "unknown"
            )
            self._disposition_ids[
                (
                    source_kind,
                    timeframe,
                    side,
                    structural_rank,
                    internal_external,
                    disposition,
                )
            ].add(source_clock_key)

    @staticmethod
    def _group4_source_kind(inventory_kind: Any) -> str:
        kind = _value(inventory_kind)
        if kind in {"equal_highs", "equal_lows"}:
            return "formed_liquidity_pool"
        if kind == "range_boundary":
            return "mature_range_boundary"
        return kind or "unknown"

    def _process_inventory_snapshot(
        self,
        values: Iterable[Any],
        *,
        asof: pd.Timestamp,
        in_window: bool,
        replace_current: bool = True,
    ) -> dict[str, Any]:
        """Index and aggregate one inventory snapshot in one traversal."""

        inventory_by_id: dict[str, Any] = {}
        for state in values:
            source_id = str(_mapping_value(state, "item_id") or "")
            if not source_id:
                continue
            inventory_by_id[source_id] = state
            kind = _value(_mapping_value(state, "kind")) or "unknown"
            structural_rank = (
                _value(_mapping_value(state, "structural_rank")) or "unknown"
            )
            self._source_metadata[source_id] = {
                "source_kind": self._group4_source_kind(kind),
                "inventory_kind": kind,
                "source_timeframe": (
                    _value(_mapping_value(state, "timeframe")) or "unknown"
                ),
                "side": _value(_mapping_value(state, "side")) or "unknown",
                "structural_rank": structural_rank,
            }
            if (
                in_window
                and kind in {"equal_highs", "equal_lows", "range_boundary"}
                and _value(_mapping_value(state, "lifecycle")) == "visible"
            ):
                self._visible_eligible_sources.add(source_id)
                internal_external = (
                    structural_rank
                    if structural_rank in {"internal", "external"}
                    else "unknown"
                )
                self._visible_eligible_source_ids[
                    (
                        self._group4_source_kind(kind),
                        _value(_mapping_value(state, "timeframe")) or "unknown",
                        _value(_mapping_value(state, "side")) or "unknown",
                        structural_rank,
                        internal_external,
                    )
                ].add(source_id)
            if in_window:
                self._observe_inventory(state, asof)
            else:
                self._remember_warmup_entity(
                    group="group12",
                    primitive="liquidity_inventory",
                    entity_id=source_id,
                    lifecycle=_mapping_value(state, "lifecycle"),
                )
        if replace_current:
            self._inventory_state_by_id = inventory_by_id
        else:
            self._inventory_state_by_id.update(inventory_by_id)
        return self._inventory_state_by_id

    def _prime_warmup_baseline(self) -> None:
        baseline = self._pending_warmup_observation
        if baseline is None:
            return
        baseline_asof = _clock(baseline.asof)
        if baseline_asof is None:
            raise ValueError("warmup baseline lacks an as-of clock")
        self._process_inventory_snapshot(
            baseline.liquidity_inventory,
            asof=baseline_asof,
            in_window=False,
            replace_current=True,
        )
        self._cache_baseline(baseline)
        self._pending_warmup_observation = None
        self._transport_counts["warmup_baselines"] += 1

    def observe(
        self,
        update: "ReaderUpdate | Any | None",
        observation: MarketObservation,
        previous_observation: MarketObservation | None = None,
        group4_source_dispositions: Iterable[Any] = (),
    ) -> bool:
        """Consume one completed observation; return ``False`` for a retry."""

        asof = _clock(observation.asof)
        if asof is None:
            raise ValueError("MarketObservation lacks an as-of clock")
        key = (str(observation.symbol), int(observation.instrument_id), asof.isoformat())
        if key == self._last_observation_key:
            return False
        if self._last_input_asof is not None and asof <= self._last_input_asof:
            raise ValueError("eye statistics require strictly increasing observation clocks")
        if self.coverage_start is not None and asof < self.coverage_start:
            raise ValueError("eye statistics received data before coverage_start")
        self._last_observation_key = key
        self._last_input_asof = asof
        in_window = self._in_window(asof)
        if not in_window:
            # A single retained immutable snapshot is sufficient to establish
            # the left-truncated live cohort.  Re-scanning every warmup minute
            # adds no authority evidence and dominated long replay cost.
            self._pending_warmup_observation = observation
            self._transport_counts["warmup_observations_retained"] += 1
            return True

        self._prime_warmup_baseline()
        delta_mode = bool(
            _mapping_value(
                observation,
                "typed_transition_delta_available",
                False,
            )
        )
        if delta_mode and self._window_inventory_primed:
            inventory_by_id = self._process_inventory_snapshot(
                _mapping_value(
                    observation,
                    "liquidity_inventory_transitions_this_update",
                    (),
                ),
                asof=asof,
                in_window=True,
                replace_current=False,
            )
            self._transport_counts["transition_delta_observations"] += 1
        else:
            inventory_by_id = self._process_inventory_snapshot(
                observation.liquidity_inventory,
                asof=asof,
                in_window=True,
                replace_current=True,
            )
            self._window_inventory_primed = True
            self._transport_counts[
                "initial_or_fallback_snapshot_observations"
            ] += 1

        self._observation_count += 1
        self._first_asof = self._first_asof or asof
        self._last_asof = asof
        if not self._window_start_readiness_captured:
            self._ready_at_window_start = {
                (_value(timeframe) or str(timeframe)): (
                    (_value(timeframe) or str(timeframe)) in self._ready_seen
                    or bool(frame.ready)
                )
                for timeframe, frame in observation.frames.items()
            }
            self._window_start_readiness_captured = True

        self._observe_clock(update, observation, previous_observation)
        for frame in observation.frames.values():
            timeframe = _value(frame.timeframe) or str(frame.timeframe)
            cutoff = _clock(frame.cutoff)
            cutoff_text = None if cutoff is None else cutoff.isoformat()
            if (
                cutoff_text is not None
                and self._last_typed_frame_cutoff.get(timeframe) == cutoff_text
            ):
                continue
            if cutoff_text is not None:
                self._last_typed_frame_cutoff[timeframe] = cutoff_text
            self._observe_group12_frame(frame, asof)
            if not delta_mode:
                for state in frame.fair_value_gaps:
                    self._observe_fvg(state, asof)
                for state in frame.order_blocks:
                    self._observe_order_block(state, asof)
                for state in frame.dealing_ranges:
                    self._observe_range(state, asof)

        self._observe_displacement(observation.displacement)
        pool_values = (
            _mapping_value(
                observation,
                "liquidity_pool_transitions_this_update",
                (),
            )
            if delta_mode
            else observation.liquidity_pool_states
        )
        for state in pool_values:
            self._observe_pool(state, asof)
        fvg_values = (
            _mapping_value(
                observation,
                "group3_fvg_transitions_this_update",
                (),
            )
            if delta_mode
            else observation.group3_boundary_fvg_transitions
        )
        for state in fvg_values:
            self._observe_fvg(state, asof)
        order_block_values = (
            _mapping_value(
                observation,
                "group3_order_block_transitions_this_update",
                (),
            )
            if delta_mode
            else observation.group3_boundary_order_block_transitions
        )
        for state in order_block_values:
            self._observe_order_block(state, asof)
        self._observe_order_block_funnel(
            _mapping_value(observation, "group3_order_block_funnel", ())
        )
        self._ob_funnel_contract_seen = self._ob_funnel_contract_seen or hasattr(
            observation, "group3_order_block_funnel"
        )
        manipulation_values = (
            _mapping_value(
                observation,
                "group4_manipulation_transitions_this_update",
                (),
            )
            if delta_mode
            else observation.manipulations
        )
        for state in manipulation_values:
            self._observe_manipulation(state, asof)
        if delta_mode:
            for state in _mapping_value(
                observation,
                "group4_range_transitions_this_update",
                (),
            ):
                self._observe_range(state, asof)
        else:
            for state in observation.group4_boundary_range_transitions:
                self._observe_range(state, asof)
        self._observe_range_funnel(
            _mapping_value(observation, "group4_range_funnel", ())
        )
        self._range_funnel_contract_seen = (
            self._range_funnel_contract_seen
            or hasattr(observation, "group4_range_funnel")
        )
        if not delta_mode:
            for state in observation.group4_boundary_manipulation_transitions:
                self._observe_manipulation(state, asof)
        entry_values = (
            _mapping_value(
                observation,
                "group5_entry_location_transitions_this_update",
                (),
            )
            if delta_mode
            else observation.entry_locations
        )
        for state in entry_values:
            self._observe_entry_location(state, asof)
        reacceptance_values = (
            _mapping_value(
                observation,
                "group5_reacceptance_transitions_this_update",
                (),
            )
            if delta_mode
            else observation.qualified_reacceptances
        )
        for state in reacceptance_values:
            self._observe_reacceptance(state, asof)
        micro_values = (
            _mapping_value(
                observation,
                "group5_micro_bos_transitions_this_update",
                (),
            )
            if delta_mode
            else observation.micro_bos_references
        )
        for state in micro_values:
            self._observe_micro_bos(state)
        path_values = (
            _mapping_value(
                observation,
                "group5_path_transitions_this_update",
                (),
            )
            if delta_mode
            else observation.path_sequences
        )
        for state in path_values:
            self._observe_path(state, asof)
        if not delta_mode:
            for state in observation.group5_boundary_path_transitions:
                self._observe_path(state, asof)
            for state in observation.group5_boundary_reacceptance_transitions:
                self._observe_reacceptance(state, asof)
        self._group5_contract_seen = (
            self._group5_contract_seen
            or bool(_mapping_value(observation, "group5_typed_available", False))
            or hasattr(observation, "path_sequences")
        )

        produced_dispositions = tuple(
            _mapping_value(observation, "group4_source_dispositions", ())
        )
        self._disposition_contract_seen = (
            self._disposition_contract_seen
            or hasattr(observation, "group4_source_dispositions")
        )
        supplied_dispositions = tuple(group4_source_dispositions)
        if produced_dispositions:
            dispositions: tuple[Any, ...] = produced_dispositions
        elif supplied_dispositions:
            dispositions = supplied_dispositions
        else:
            automatic_dispositions: list[dict[str, Any]] = []
            automatic_dispositions.extend(
                {
                    "source_inventory_item_id": item_id,
                    "observed_at": asof,
                    "disposition": "ambiguous_dual_side",
                }
                for item_id in observation.group4_ambiguous_sweep_item_ids
            )
            automatic_dispositions.extend(
                {
                    "source_inventory_item_id": item_id,
                    "observed_at": asof,
                    "disposition": "atr_unready",
                }
                for item_id in observation.group4_atr_unready_sweep_item_ids
            )
            dispositions = tuple(automatic_dispositions)
        self._observe_source_dispositions(
            dispositions,
            inventory=inventory_by_id,
            fallback_asof=asof,
        )
        return True

    def _selected_cases(self) -> tuple[list[dict[str, Any]], str]:
        raw_pools = {
            key: [values[item] for item in sorted(values)]
            for key in _FROZEN_CASE_STRATA
            if (values := self._case_pools.get(key))
        }
        selected: list[dict[str, Any]] = []
        selected_entities: set[tuple[str, str, str]] = set()

        def episode_identity(case: Mapping[str, Any]) -> tuple[str, str, str]:
            primitive = str(case["primitive"])
            if case["group"] == "group4" and primitive in {
                "dealing_range",
                "range_maturity_evaluation",
            }:
                primitive = "range_episode"
            return (
                str(case["group"]),
                primitive,
                str(case["entity_id"]),
            )

        def add_case(case: dict[str, Any]) -> bool:
            identity = episode_identity(case)
            if identity in selected_entities or len(selected) >= _CASE_LIMIT:
                return False
            selected.append(case)
            selected_entities.add(identity)
            return True

        # All recognized mature cases have priority because they are the
        # finite authority population under review.  The per-stratum bounded
        # pool has already retained the smallest hashes if more than 40 exist.
        for case in raw_pools.get("all_recognized_mature", ()):
            add_case(case)
        if len(selected) >= _CASE_LIMIT:
            return selected, "complete"

        remaining_strata = tuple(
            key for key in _FROZEN_CASE_STRATA if key != "all_recognized_mature"
        )
        positions = {key: 0 for key in remaining_strata}
        while len(selected) < _CASE_LIMIT:
            added = False
            for key in remaining_strata:
                values = raw_pools.get(key, ())
                while positions[key] < len(values):
                    case = values[positions[key]]
                    positions[key] += 1
                    if add_case(case):
                        added = True
                        break
                if len(selected) == _CASE_LIMIT:
                    break
            if not added:
                break
        status = "complete" if len(selected) >= _CASE_MINIMUM else "insufficient_candidates"
        return selected, status

    def _funnel_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(self._cohort_entities):
            group, primitive, strata = key
            entity_ids = self._cohort_entities[key]
            lifecycle_entities = {
                lifecycle: len(values)
                for (g, p, s, lifecycle), values in sorted(
                    self._lifecycle_entities.items()
                )
                if (g, p, s) == key
            }
            lifecycle_transitions = {
                lifecycle: int(count)
                for (g, p, s, lifecycle), count in sorted(
                    self._lifecycle_transitions.items()
                )
                if (g, p, s) == key
            }
            latest_counts: Counter[str] = Counter()
            for (
                g,
                p,
                entity_id,
            ), (_, lifecycle, entity_strata, _) in self._open_latest.items():
                if g != group or p != primitive or entity_id not in entity_ids:
                    continue
                if strata and entity_strata != strata:
                    continue
                latest_counts[lifecycle] += 1
            for (
                g,
                p,
                terminal_strata,
                lifecycle,
            ), count in self._terminal_latest_counts.items():
                if g == group and p == primitive and terminal_strata == strata:
                    latest_counts[lifecycle] += count
            latest_total = sum(latest_counts.values())
            reasons = {
                reason: int(count)
                for (g, p, s, reason), count in sorted(self._reason_counts.items())
                if (g, p, s) == key
            }
            open_states = _OPEN_LIFECYCLES.get(primitive, frozenset())
            right_censored = sum(
                count for lifecycle, count in latest_counts.items() if lifecycle in open_states
            )
            rows.append(
                {
                    "group": group,
                    "primitive": primitive,
                    "strata": _strata_payload(strata),
                    "strata_basis": "first_in_window_transition_cohort",
                    "entities": len(entity_ids),
                    "lifecycle_entities": dict(sorted(lifecycle_entities.items())),
                    "lifecycle_transitions": dict(sorted(lifecycle_transitions.items())),
                    "latest_lifecycle_counts": dict(sorted(latest_counts.items())),
                    "transition_reasons": reasons,
                    "right_censored": right_censored,
                    "conservation": {
                        "entities": len(entity_ids),
                        "classified_latest": latest_total,
                        "balanced": len(entity_ids) == latest_total,
                    },
                }
            )
        return rows

    def _manipulation_conservation(self) -> dict[str, Any]:
        entity_ids = self._cohort_entities.get(
            ("group4", "manipulation", ()), set()
        )
        outcomes = Counter(
            {
                "reaccepted": 0,
                "accepted_outside": 0,
                "deadline_censored": 0,
                "hard_boundary_censored": 0,
                "right_censored": 0,
            }
        )
        for entity_id in entity_ids:
            entity_key = ("group4", "manipulation", entity_id)
            latest = self._open_latest.get(entity_key)
            terminal = self._terminal_transition_by_entity.get(entity_key)
            lifecycle = (
                terminal[0]
                if terminal is not None
                else None if latest is None else latest[1]
            )
            if lifecycle in outcomes and lifecycle != "right_censored":
                outcomes[lifecycle] += 1
            else:
                outcomes["right_censored"] += 1
        classified = sum(outcomes.values())
        return {
            "swept_created": len(entity_ids),
            **dict(outcomes),
            "conservation": {
                "left": len(entity_ids),
                "right": classified,
                "balanced": len(entity_ids) == classified,
            },
        }

    @staticmethod
    def _identity_funnel_rows(
        values: Mapping[tuple[str, str, str], set[str]],
        *,
        denominator_stage: str,
    ) -> list[dict[str, Any]]:
        strata = sorted({(kind, timeframe) for kind, timeframe, _ in values})
        rows: list[dict[str, Any]] = []
        for source_kind, source_timeframe in strata:
            stages = {
                stage: len(ids)
                for (kind, timeframe, stage), ids in sorted(values.items())
                if (kind, timeframe) == (source_kind, source_timeframe)
            }
            denominator = int(stages.get(denominator_stage, 0))
            path_count = int(stages.get("path", 0))
            trigger_count = int(stages.get("trigger", 0))
            terminal_after_trigger = int(
                stages.get("terminal_after_trigger", 0)
            )
            path_terminal_all = int(stages.get("path_terminal_all", 0))
            complete_count = int(stages.get("terminal_complete", 0))
            interrupted_count = int(stages.get("terminal_interrupted", 0))
            if denominator_stage == "qualified_zone":
                entry_count = int(stages.get("entry_location", 0))
                monotone = (
                    denominator
                    >= entry_count
                    >= path_count
                    >= trigger_count
                    >= terminal_after_trigger
                    and path_count >= path_terminal_all
                )
            else:
                monotone = (
                    denominator
                    >= path_count
                    >= trigger_count
                    >= terminal_after_trigger
                    and path_count >= path_terminal_all
                )
            rows.append(
                {
                    "source_kind": source_kind,
                    "source_timeframe": source_timeframe,
                    "identity_stage_counts": dict(sorted(stages.items())),
                    "conservation": {
                        "path_terminal_all": path_terminal_all,
                        "terminal_after_trigger": terminal_after_trigger,
                        "complete_plus_interrupted": (
                            complete_count + interrupted_count
                        ),
                        "terminal_balanced": path_terminal_all
                        == complete_count + interrupted_count,
                        "funnel_monotone": monotone,
                    },
                }
            )
        return rows

    @staticmethod
    def _bounded_ids(values: Iterable[str]) -> list[str]:
        return sorted({str(value) for value in values if str(value)})[
            :_BAD_ID_LIMIT
        ]

    @classmethod
    def _one_to_one_link_summary(
        cls,
        *,
        roots: set[str],
        child_to_parent: Mapping[str, str],
        binding_conflicts: set[str],
    ) -> dict[str, Any]:
        children_by_parent: dict[str, set[str]] = defaultdict(set)
        orphan_children: set[str] = set()
        for child_id, parent_id in child_to_parent.items():
            if parent_id not in roots:
                orphan_children.add(child_id)
            else:
                children_by_parent[parent_id].add(child_id)
        duplicate_parent_ids = {
            parent_id
            for parent_id, child_ids in children_by_parent.items()
            if len(child_ids) > 1
        }
        linked_roots = set(children_by_parent)
        unlinked_roots = roots - linked_roots
        exact_violations = (
            len(orphan_children)
            + len(duplicate_parent_ids)
            + len(binding_conflicts)
        )
        return {
            "root_identities": len(roots),
            "child_identities": len(child_to_parent),
            "valid_parent_child_links": sum(
                len(child_ids) for child_ids in children_by_parent.values()
            ),
            "linked_roots": len(linked_roots),
            "unlinked_roots": len(unlinked_roots),
            "orphan_children": len(orphan_children),
            "parents_with_multiple_children": len(duplicate_parent_ids),
            "binding_conflict_children": len(binding_conflicts),
            "exact_identity_violation_count": exact_violations,
            "exact_identity_conserved": exact_violations == 0,
            "bad_ids": {
                "orphan_children": cls._bounded_ids(orphan_children),
                "parents_with_multiple_children": cls._bounded_ids(
                    duplicate_parent_ids
                ),
                "binding_conflict_children": cls._bounded_ids(
                    binding_conflicts
                ),
            },
        }

    @classmethod
    def _subset_summary(
        cls,
        *,
        members: set[str],
        parents: set[str],
    ) -> dict[str, Any]:
        outside = members - parents
        return {
            "members": len(members),
            "parent_paths": len(parents),
            "outside_parent_paths": len(outside),
            "is_subset": not outside,
            "bad_ids": cls._bounded_ids(outside),
        }

    def _group5_identity_conservation(self) -> dict[str, Any]:
        zone_roots = set(self._group5_qualified_zone_details)
        entry_roots = set(self._group5_entry_source_zone_by_id)
        zone_paths = set(self._group5_zone_path_location_by_id)
        manipulation_roots = set(self._manipulation_cohort_ids)
        pool_paths = set(self._group5_pool_path_manipulation_by_id)
        zone_summary = {
            "qualified_roots_by_kind": dict(
                sorted(
                    Counter(
                        details["kind"]
                        for details in self._group5_qualified_zone_details.values()
                    ).items()
                )
            ),
            "qualified_zone_to_entry_location": self._one_to_one_link_summary(
                roots=zone_roots,
                child_to_parent=self._group5_entry_source_zone_by_id,
                binding_conflicts=self._group5_entry_binding_conflicts,
            ),
            "entry_location_to_path_sequence": self._one_to_one_link_summary(
                roots=entry_roots,
                child_to_parent=self._group5_zone_path_location_by_id,
                binding_conflicts=self._group5_zone_path_binding_conflicts,
            ),
            "trigger_is_path_subset": self._subset_summary(
                members=self._group5_zone_trigger_paths,
                parents=zone_paths,
            ),
            "terminal_is_path_subset": self._subset_summary(
                members=self._group5_zone_terminal_paths,
                parents=zone_paths,
            ),
        }
        manipulation_summary = {
            "manipulation_to_pool_path": self._one_to_one_link_summary(
                roots=manipulation_roots,
                child_to_parent=self._group5_pool_path_manipulation_by_id,
                binding_conflicts=self._group5_pool_path_binding_conflicts,
            ),
            "trigger_is_path_subset": self._subset_summary(
                members=self._group5_pool_trigger_paths,
                parents=pool_paths,
            ),
            "terminal_is_path_subset": self._subset_summary(
                members=self._group5_pool_terminal_paths,
                parents=pool_paths,
            ),
        }
        violation_count = sum(
            (
                zone_summary["qualified_zone_to_entry_location"]
                ["exact_identity_violation_count"],
                zone_summary["entry_location_to_path_sequence"]
                ["exact_identity_violation_count"],
                zone_summary["trigger_is_path_subset"]["outside_parent_paths"],
                zone_summary["terminal_is_path_subset"]["outside_parent_paths"],
                manipulation_summary["manipulation_to_pool_path"]
                ["exact_identity_violation_count"],
                manipulation_summary["trigger_is_path_subset"]
                ["outside_parent_paths"],
                manipulation_summary["terminal_is_path_subset"]
                ["outside_parent_paths"],
            )
        )
        return {
            "qualified_zone_chain": zone_summary,
            "manipulation_pool_path_chain": manipulation_summary,
            "exact_identity_violation_count": violation_count,
            "exact_identity_conserved": violation_count == 0,
        }

    @staticmethod
    def _geometry_matches(
        zone: Mapping[str, Any],
        location: Mapping[str, Any],
    ) -> bool | None:
        values = (
            zone.get("lower_bound"),
            zone.get("upper_bound"),
            location.get("lower_bound"),
            location.get("upper_bound"),
        )
        if any(value is None for value in values):
            return None
        return math.isclose(
            float(values[0]),
            float(values[2]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ) and math.isclose(
            float(values[1]),
            float(values[3]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    def _favr_observation_chain(self) -> dict[str, Any]:
        """Reproduce the frozen FAVR eye chain without invoking the Brain.

        A mature-range manipulation is not a Group5 ``pool_reversal`` source;
        Group5 deliberately registers only its later qualified FVG/OB as a
        ``zone_return`` path.  The join below therefore follows the same exact
        source identity, clocks, frozen range geometry and first-candidate rule
        used by typed FAVR.  It never associates events by nearest time or by a
        future outcome.
        """

        roots = set(self._favr_reaccepted_manipulation_ids)
        stages = Counter({"reaccepted_mature_range_manipulation": len(roots)})
        attrition: Counter[str] = Counter()
        identity_violations: Counter[str] = Counter()
        contract_not_exposed: Counter[str] = Counter()
        bad_ids: dict[str, set[str]] = defaultdict(set)
        zone_paths_by_location: dict[str, list[str]] = defaultdict(list)
        for path_id, location_id in self._group5_zone_path_location_by_id.items():
            zone_paths_by_location[location_id].append(path_id)

        complete_roots: set[str] = set()
        for manipulation_id in sorted(roots):
            source = self._manipulation_metadata.get(manipulation_id, {})
            range_id = str(source.get("source_id") or "")
            reaccepted_at = _clock(source.get("reaccepted_at"))
            expected_direction = {
                "above": "short",
                "below": "long",
            }.get(str(source.get("side", "")))
            if not range_id or reaccepted_at is None or expected_direction is None:
                contract_not_exposed["manipulation_source_side_or_clock"] += 1
                bad_ids["manipulation_source_side_or_clock"].add(manipulation_id)
                continue
            dealing_range = self._mature_range_details.get(range_id)
            if dealing_range is None:
                attrition["mature_range_identity_missing"] += 1
                bad_ids["mature_range_identity_missing"].add(range_id)
                continue
            range_values = tuple(
                dealing_range.get(name)
                for name in ("lower_bound", "upper_bound", "midpoint")
            )
            if any(value is None for value in range_values):
                contract_not_exposed["mature_range_geometry"] += 1
                bad_ids["mature_range_geometry"].add(range_id)
                continue
            range_lower, range_upper, range_midpoint = map(float, range_values)
            range_broken_at = _clock(dealing_range.get("broken_at"))
            stages["mature_range_identity"] += 1
            if (
                range_broken_at is not None
                and range_broken_at <= reaccepted_at
            ):
                attrition["range_broken_before_or_at_reacceptance"] += 1
                continue

            # Match the first location which was causally eligible for this
            # manipulation.  This is frozen before inspecting pullback/trigger.
            candidates: list[tuple[pd.Timestamp, str, str, str]] = []
            for location_id, zone_id in self._group5_entry_source_zone_by_id.items():
                location = self._group5_entry_location_details.get(location_id)
                zone = self._group5_qualified_zone_details.get(zone_id)
                if location is None or zone is None:
                    continue
                formed_at = _clock(location.get("formed_at"))
                confirmed_at = _clock(zone.get("confirmed_at"))
                active_at = _clock(zone.get("source_displacement_active_at"))
                displacement_id = str(zone.get("source_displacement_id") or "")
                geometry_matches = self._geometry_matches(zone, location)
                zone_lower = zone.get("lower_bound")
                zone_upper = zone.get("upper_bound")
                if (
                    formed_at is None
                    or confirmed_at is None
                    or active_at is None
                    or not displacement_id
                    or geometry_matches is None
                    or zone_lower is None
                    or zone_upper is None
                ):
                    contract_not_exposed["zone_location_clock_or_geometry"] += 1
                    bad_ids["zone_location_clock_or_geometry"].add(location_id)
                    continue
                zone_midpoint = (float(zone_lower) + float(zone_upper)) / 2.0
                if (
                    not geometry_matches
                    or location.get("source_zone_id") != zone_id
                    or location.get("source_displacement_id") != displacement_id
                ):
                    identity_violations[
                        "zone_entry_identity_or_geometry_mismatch"
                    ] += 1
                    bad_ids["zone_entry_identity_or_geometry_mismatch"].add(
                        location_id
                    )
                    continue
                if not (
                    location.get("direction") == expected_direction
                    and zone.get("direction") == expected_direction
                    and formed_at == confirmed_at
                    and formed_at > reaccepted_at
                    and active_at > reaccepted_at
                    and (
                        range_broken_at is None
                        or formed_at < range_broken_at
                    )
                    and range_lower <= float(zone_lower)
                    and float(zone_upper) <= range_upper
                    and (
                        zone_midpoint < range_midpoint
                        if expected_direction == "long"
                        else zone_midpoint > range_midpoint
                    )
                ):
                    continue
                candidates.append(
                    (formed_at, location_id, zone_id, displacement_id)
                )
            if not candidates:
                attrition["opposite_displacement_zone_inside_range_missing"] += 1
                continue
            selected_formed_at, location_id, zone_id, displacement_id = min(
                candidates
            )
            stages["opposite_displacement"] += 1
            stages["opposite_displacement_linked_zone"] += 1
            stages["entry_location"] += 1

            mss_candidates = sorted(
                (
                    details["resolved_at"],
                    bos_id,
                )
                for bos_id, details in self._opposed_mss_by_displacement.get(
                    displacement_id, {}
                ).items()
                if (
                    details.get("direction") == expected_direction
                    and details.get("resolved_at") is not None
                    and details["resolved_at"] > reaccepted_at
                    and (
                        range_broken_at is None
                        or details["resolved_at"] < range_broken_at
                    )
                )
            )
            if not mss_candidates:
                attrition["opposed_mss_missing"] += 1
                continue
            return_mss_at, _ = mss_candidates[0]
            stages["opposed_mss"] += 1

            zone_paths = sorted(zone_paths_by_location.get(location_id, ()))
            if not zone_paths:
                attrition["zone_return_path_missing"] += 1
                continue
            if len(zone_paths) != 1:
                identity_violations["multiple_zone_return_paths"] += 1
                bad_ids["multiple_zone_return_paths"].add(location_id)
                continue
            zone_path_id = zone_paths[0]
            if self._group5_path_directions.get(zone_path_id) != expected_direction:
                identity_violations["zone_return_path_direction_mismatch"] += 1
                bad_ids["zone_return_path_direction_mismatch"].add(zone_path_id)
                continue
            stages["zone_return_path"] += 1
            zone_steps = self._path_steps_by_id.get(zone_path_id, ())
            kinds = tuple(step[1] for step in zone_steps)
            if "first_pullback" not in kinds:
                attrition["first_pullback_missing"] += 1
                continue
            pullback_index = kinds.index("first_pullback")
            pullback_clock = zone_steps[pullback_index][2]
            if pullback_clock is None:
                contract_not_exposed["first_pullback_clock"] += 1
                bad_ids["first_pullback_clock"].add(zone_path_id)
                continue
            if pullback_clock < selected_formed_at:
                identity_violations["first_pullback_before_zone"] += 1
                bad_ids["first_pullback_before_zone"].add(zone_path_id)
                continue
            if (
                range_broken_at is not None
                and range_broken_at <= pullback_clock
            ):
                attrition["range_broken_before_or_at_first_pullback"] += 1
                continue
            stages["first_pullback"] += 1
            if return_mss_at > pullback_clock:
                attrition["opposed_mss_after_first_pullback"] += 1
                continue
            trigger_steps = tuple(
                step
                for step in zone_steps[pullback_index + 1 :]
                if step[1] in _FAVR_TRIGGER_STEPS
            )
            if not trigger_steps:
                attrition["trigger_missing_after_first_pullback"] += 1
                continue
            trigger_clocks = tuple(
                step[2] for step in trigger_steps if step[2] is not None
            )
            if not trigger_clocks:
                contract_not_exposed["trigger_clock"] += 1
                bad_ids["trigger_clock"].add(zone_path_id)
                continue
            trigger_clock = min(trigger_clocks)
            if (
                range_broken_at is not None
                and range_broken_at <= trigger_clock
            ):
                attrition["range_broken_before_or_at_trigger"] += 1
                continue
            stages["trigger"] += 1
            complete_roots.add(manipulation_id)

        if contract_not_exposed:
            status = "authoritative_join_not_exposed"
        elif not roots:
            status = "authoritative_identity_join_exposed_no_root_cases"
        elif identity_violations:
            status = "authoritative_identity_join_exposed_with_violations"
        else:
            status = "authoritative_identity_join_exposed"
        return {
            "status": status,
            "join_contract": [
                "manipulation.source_id_to_mature_range.range_id",
                "zone.source_displacement_active_at_after_manipulation.reaccepted_at",
                "zone_frozen_geometry_inside_range_on_swept_side_half",
                "zone.source_displacement_id_to_confirmed_opposed_mss",
                "zone_id_to_entry_location.source_zone_id",
                "zone_geometry_to_entry_location.frozen_geometry",
                "entry_location_id_to_zone_return_path.context_id",
                "first_pullback_then_aligned_trigger",
            ],
            "stage_counts_by_root_manipulation": dict(sorted(stages.items())),
            "complete_chains": len(complete_roots),
            "attrition": dict(sorted(attrition.items())),
            "identity_violations": dict(sorted(identity_violations.items())),
            "authoritative_join_not_exposed": dict(
                sorted(contract_not_exposed.items())
            ),
            "bad_ids": {
                name: self._bounded_ids(ids)
                for name, ids in sorted(bad_ids.items())
            },
        }

    @staticmethod
    def _range_episode_diagnostic_summary(
        values: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        combinations = Counter(
            str(value["unmet_gate_combination"])
            for value in values.values()
        )
        cardinalities = Counter(
            str(value["unmet_gate_cardinality"])
            for value in values.values()
        )
        weakest = Counter(
            str(value["weakest_gate"] or "not_exposed")
            for value in values.values()
        )
        margins = _NumericSummary()
        for value in values.values():
            margin = value.get("weakest_margin")
            if margin is not None:
                margins.add(float(margin))
        return {
            "unique_range_episodes": len(values),
            "unmet_gate_combinations": dict(sorted(combinations.items())),
            "unmet_gate_cardinality": dict(sorted(cardinalities.items())),
            "weakest_gate": dict(sorted(weakest.items())),
            "weakest_margin": margins.payload(),
        }

    @staticmethod
    def _linear_quantile(values: tuple[float, ...], probability: float) -> float:
        if not values:
            raise ValueError("quantile requires at least one value")
        position = (len(values) - 1) * probability
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        if lower_index == upper_index:
            return values[lower_index]
        weight = position - lower_index
        return (
            values[lower_index] * (1.0 - weight)
            + values[upper_index] * weight
        )

    def _range_formation_atr_tertiles(self) -> dict[str, Any]:
        ordered = tuple(sorted(self._range_formation_atr_by_id.values()))
        if not ordered:
            return {
                "usage": "outcome_blind_aggregate_only_not_a_reducer_gate_or_case_selector",
                "method": "annual_empirical_linear_tertiles",
                "cut_population": "all_in_window_range_candidates_before_outcome",
                "cohort_count": 0,
                "cut_points": None,
                "by_tertile": [],
            }
        lower_cut = self._linear_quantile(ordered, 1.0 / 3.0)
        upper_cut = self._linear_quantile(ordered, 2.0 / 3.0)
        bucket_ids: dict[str, set[str]] = defaultdict(set)
        for range_id, formation_atr in self._range_formation_atr_by_id.items():
            if formation_atr <= lower_cut:
                bucket = "lower"
            elif formation_atr <= upper_cut:
                bucket = "middle"
            else:
                bucket = "upper"
            bucket_ids[bucket].add(range_id)
        rows = []
        for bucket in ("lower", "middle", "upper"):
            ids = bucket_ids.get(bucket, set())
            rows.append(
                {
                    "tertile": bucket,
                    "range_entities": len(ids),
                    "ever_mature": len(ids & self._seen_mature_ranges),
                }
            )
        return {
            "usage": "outcome_blind_aggregate_only_not_a_reducer_gate_or_case_selector",
            "method": "annual_empirical_linear_tertiles",
            "cut_population": "all_in_window_range_candidates_before_outcome",
            "cohort_count": len(self._range_formation_atr_by_id),
            "cut_points": {
                "p33_formation_atr": lower_cut,
                "p67_formation_atr": upper_cut,
            },
            "by_tertile": rows,
        }

    def _displacement_never_active_diagnostics(self) -> dict[str, Any]:
        by_status: dict[str, Any] = {}
        never_active = self._displacement_started - self._displacement_active
        for status in ("terminal", "right_censored"):
            ids = {
                entity_id
                for entity_id in never_active
                if (entity_id in self._displacement_terminal_reason)
                == (status == "terminal")
            }
            failed_gate_counts: Counter[str] = Counter()
            failed_combinations: Counter[str] = Counter()
            margins = {
                name: _NumericSummary()
                for name in _DISPLACEMENT_ACTIVATION_RATIO_METRICS
            }
            ratios_available = 0
            for entity_id in ids:
                ratios = self._displacement_latest_activation_ratios.get(
                    entity_id,
                    {},
                )
                if ratios:
                    ratios_available += 1
                failed = []
                for name in _DISPLACEMENT_ACTIVATION_RATIO_METRICS:
                    ratio = ratios.get(name)
                    if ratio is None:
                        continue
                    margins[name].add(ratio - 1.0)
                    if ratio < 1.0:
                        failed.append(name)
                        failed_gate_counts[name] += 1
                if not ratios:
                    combination = "ratios_unavailable"
                elif len(ratios) < len(_DISPLACEMENT_ACTIVATION_RATIO_METRICS):
                    combination = "ratios_incomplete"
                else:
                    combination = "+".join(failed) if failed else "none"
                failed_combinations[combination] += 1
            by_status[status] = {
                "episodes": len(ids),
                "episodes_with_final_ratios": ratios_available,
                "episodes_without_final_ratios": len(ids) - ratios_available,
                "failed_gate_counts": dict(sorted(failed_gate_counts.items())),
                "failed_gate_combinations": dict(
                    sorted(failed_combinations.items())
                ),
                "gate_margin_ratio_minus_one": {
                    name: margins[name].payload()
                    for name in _DISPLACEMENT_ACTIVATION_RATIO_METRICS
                },
            }
        return {
            "gate_ratio_fields": list(_DISPLACEMENT_ACTIVATION_RATIO_METRICS),
            "failure_rule": "activation_ratio_below_one",
            "by_status": by_status,
        }

    def _reconcile_final_snapshot(
        self,
        observation: MarketObservation,
    ) -> None:
        """Reconcile open/right-censored state once at the scan boundary."""

        asof = _clock(observation.asof)
        if asof is None:
            raise ValueError("final eye snapshot lacks an as-of clock")
        key = (
            str(observation.symbol),
            int(observation.instrument_id),
            asof.isoformat(),
        )
        if key == self._final_reconciled_observation_key:
            return
        for frame in observation.frames.values():
            self._observe_group12_frame(frame, asof)
            for state in frame.fair_value_gaps:
                self._observe_fvg(state, asof)
            for state in frame.order_blocks:
                self._observe_order_block(state, asof)
            for state in frame.dealing_ranges:
                self._observe_range(state, asof)
        self._process_inventory_snapshot(
            observation.liquidity_inventory,
            asof=asof,
            in_window=True,
            replace_current=True,
        )
        for state in observation.liquidity_pool_states:
            self._observe_pool(state, asof)
        for state in observation.manipulations:
            self._observe_manipulation(state, asof)
        for state in observation.entry_locations:
            self._observe_entry_location(state, asof)
        for state in observation.qualified_reacceptances:
            self._observe_reacceptance(state, asof)
        for state in observation.micro_bos_references:
            self._observe_micro_bos(state)
        for state in observation.path_sequences:
            self._observe_path(state, asof)
        self._final_reconciled_observation_key = key
        self._transport_counts["final_snapshot_reconciliations"] += 1

    def finalize(
        self,
        final_observation: MarketObservation | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable lightweight summary."""

        if final_observation is not None:
            self._reconcile_final_snapshot(final_observation)

        cases, case_status = self._selected_cases()
        clock_rows = [
            {
                **state,
                "timeframe": timeframe,
                "ready_at_window_start": self._ready_at_window_start.get(
                    timeframe, False
                ),
                "first_cutoff": _clock_text(state["first_cutoff"]),
                "last_cutoff": _clock_text(state["last_cutoff"]),
            }
            for timeframe, state in sorted(self._clocks.items())
        ]
        categorical_rows = [
            {
                "group": group,
                "primitive": primitive,
                "dimension": dimension,
                "strata": _strata_payload(strata),
                "counts": dict(sorted(counts.items())),
            }
            for (group, primitive, dimension, strata), counts in sorted(
                self._categorical.items()
            )
        ]
        numeric_rows = [
            {
                "group": group,
                "primitive": primitive,
                "lifecycle": lifecycle,
                "metric": metric,
                "strata": _strata_payload(strata),
                **summary.payload(),
            }
            for (group, primitive, lifecycle, metric, strata), summary in sorted(
                self._numeric.items()
            )
            if summary.count
        ]
        disposition_rows = [
            {
                "source_kind": source_kind,
                "source_timeframe": timeframe,
                "side": side,
                "structural_rank": structural_rank,
                "internal_external": internal_external,
                "disposition": disposition,
                "source_clock_decisions": len(ids),
                "unique_sources": len({source_id for _, source_id in ids}),
            }
            for (
                source_kind,
                timeframe,
                side,
                structural_rank,
                internal_external,
                disposition,
            ), ids in sorted(self._disposition_ids.items())
        ]
        disposition_total = len(self._source_disposition_by_clock_id)
        disposition_unique_sources = len(
            {
                source_id
                for _, source_id in self._source_disposition_by_clock_id
            }
        )
        disposition_sum = sum(
            len(ids) for ids in self._disposition_ids.values()
        )
        disposition_totals = {
            disposition: sum(
                len(ids)
                for key, ids in self._disposition_ids.items()
                if key[-1] == disposition
            )
            for disposition in sorted(_GROUP4_SOURCE_DISPOSITIONS)
        }
        selected_primary_identities = {
            source_clock
            for source_clock, disposition in (
                self._source_disposition_by_clock_id.items()
            )
            if disposition == "selected_primary"
        }
        created_episode_identities = {
            (swept_at.isoformat(), str(metadata["source_inventory_item_id"]))
            for manipulation_id in self._manipulation_cohort_ids
            if (
                (metadata := self._manipulation_metadata.get(manipulation_id))
                is not None
                and (
                    swept_at := _clock(metadata.get("swept_at"))
                )
                is not None
                and metadata.get("source_inventory_item_id")
            )
        }
        selected_without_episode = (
            selected_primary_identities - created_episode_identities
        )
        episode_without_selected = (
            created_episode_identities - selected_primary_identities
        )
        started_not_active = Counter(
            self._displacement_terminal_reason.get(entity_id, "right_censored")
            for entity_id in self._displacement_started - self._displacement_active
        )
        ob_outcomes_total = sum(self._ob_funnel_outcomes.values())
        ob_created = int(self._ob_funnel_stages.get("ob_created", 0))
        ob_created_outcomes = int(self._ob_funnel_outcomes.get("created", 0))
        range_live_pairs = int(self._range_pair_counts.get("live_structural_pairs", 0))
        range_invalid_pairs = int(self._range_pair_counts.get("invalid_geometry_pairs", 0))
        range_geometry_pairs = int(self._range_pair_counts.get("geometry_valid_pairs", 0))
        range_classified_geometry = sum(
            int(self._range_pair_counts.get(name, 0))
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
        range_eligible = int(self._range_pair_counts.get("eligible_pairs", 0))
        range_selected = int(self._range_pair_counts.get("forming_selected", 0))
        reducer_rows = [
            {
                "component": component,
                "timeframe": timeframe,
                "completion_kind": completion_kind,
                "updates": int(count),
                "exceptions": 0,
            }
            for (component, timeframe, completion_kind), count in sorted(
                self._reducer_updates.items()
            )
        ]
        range_reasons = {
            reason: int(count)
            for (group, primitive, strata, reason), count in sorted(
                self._reason_counts.items()
            )
            if group == "group4"
            and primitive == "dealing_range"
            and not strata
        }
        case_counts = {
            stratum: len(self._case_pools.get(stratum, {}))
            for stratum in _FROZEN_CASE_STRATA
        }
        selected_case_counts = {
            stratum: sum(
                str(case["stratum"]) == stratum for case in cases
            )
            for stratum in _FROZEN_CASE_STRATA
        }
        left_boundary_rows = [
            {
                "group": group,
                "primitive": primitive,
                "entities": int(count),
            }
            for (group, primitive), count in sorted(
                self._left_boundary_counts.items()
            )
        ]
        left_boundary_total = sum(self._left_boundary_counts.values())
        category_coverage = {
            stratum: {
                "retained_cases": case_counts[stratum],
                "covered": case_counts[stratum] > 0,
                "status": (
                    "requires_blind_image_classification_not_inferred"
                    if stratum == "obvious_mature_looking_but_rejected"
                    else "covered"
                    if case_counts[stratum] > 0
                    else "no_natural_case_observed"
                ),
            }
            for stratum in _FROZEN_CASE_STRATA
        }

        def reader_anomaly_total(*needles: str) -> int:
            return sum(
                int(count)
                for name, count in self._reader_anomalies.items()
                if any(needle in name.lower() for needle in needles)
            )

        paths_after_pullback = {"with_trigger": 0, "without_trigger": 0}
        for steps in self._path_steps_by_id.values():
            kinds = tuple(step[1] for step in steps)
            if "first_pullback" not in kinds:
                continue
            pullback_index = kinds.index("first_pullback")
            has_later_trigger = any(
                kind in _GROUP5_TRIGGER_STEPS
                for kind in kinds[pullback_index + 1 :]
            )
            paths_after_pullback[
                "with_trigger" if has_later_trigger else "without_trigger"
            ] += 1
        manipulations_by_source_timeframe = Counter(
            metadata["source_timeframe"]
            for metadata in self._manipulation_metadata.values()
        )
        cohort_manipulations_by_source_timeframe = Counter(
            metadata["source_timeframe"]
            for manipulation_id in self._manipulation_cohort_ids
            if (
                metadata := self._manipulation_metadata.get(manipulation_id)
            ) is not None
        )
        group5_manipulations_by_source_timeframe = Counter(
            self._group5_manipulation_paths.values()
        )
        pool_metadata_total = len(self._pool_rank_exposed | self._pool_rank_missing)
        if not pool_metadata_total or not self._pool_rank_exposed:
            pool_metadata_status = "not_exposed"
        elif self._pool_rank_missing:
            pool_metadata_status = "partially_exposed"
        else:
            pool_metadata_status = "fully_exposed"
        visible_eligible_rows = [
            {
                "source_kind": source_kind,
                "source_timeframe": timeframe,
                "side": side,
                "structural_rank": structural_rank,
                "internal_external": internal_external,
                "unique_sources": len(ids),
            }
            for (
                source_kind,
                timeframe,
                side,
                structural_rank,
                internal_external,
            ), ids in sorted(self._visible_eligible_source_ids.items())
        ]
        visible_eligible_unique = len(
            {
                source_id
                for ids in self._visible_eligible_source_ids.values()
                for source_id in ids
            }
        )
        visible_eligible_row_sum = sum(
            row["unique_sources"] for row in visible_eligible_rows
        )
        eligible_source_ids_by_timeframe: dict[str, set[str]] = defaultdict(
            set
        )
        for (
            _source_kind,
            source_timeframe,
            _side,
            _structural_rank,
            _internal_external,
        ), source_ids in self._visible_eligible_source_ids.items():
            eligible_source_ids_by_timeframe[source_timeframe].update(source_ids)
        created_manipulation_ids_by_timeframe: dict[str, set[str]] = defaultdict(
            set
        )
        created_source_ids_by_timeframe: dict[str, set[str]] = defaultdict(set)
        for manipulation_id in self._manipulation_cohort_ids:
            metadata = self._manipulation_metadata.get(manipulation_id)
            if metadata is None:
                continue
            source_timeframe = str(metadata["source_timeframe"])
            created_manipulation_ids_by_timeframe[source_timeframe].add(
                manipulation_id
            )
            source_inventory_item_id = str(
                metadata.get("source_inventory_item_id") or ""
            )
            if source_inventory_item_id:
                created_source_ids_by_timeframe[source_timeframe].add(
                    source_inventory_item_id
                )
        manipulation_rates_by_source_timeframe: list[dict[str, Any]] = []
        for source_timeframe in sorted(
            set(eligible_source_ids_by_timeframe)
            | set(created_manipulation_ids_by_timeframe)
        ):
            eligible_count = len(
                eligible_source_ids_by_timeframe[source_timeframe]
            )
            created_source_count = len(
                created_source_ids_by_timeframe[source_timeframe]
                & eligible_source_ids_by_timeframe[source_timeframe]
            )
            created_manipulation_count = len(
                created_manipulation_ids_by_timeframe[source_timeframe]
            )
            real_completed_bars = int(
                self._clocks.get(source_timeframe, {}).get(
                    "real_completed",
                    0,
                )
            )
            manipulation_rates_by_source_timeframe.append(
                {
                    "source_timeframe": source_timeframe,
                    "unique_eligible_sources": eligible_count,
                    "unique_eligible_sources_creating_manipulation": (
                        created_source_count
                    ),
                    "created_manipulations": created_manipulation_count,
                    "created_per_unique_eligible_source": (
                        None
                        if eligible_count == 0
                        else created_source_count / eligible_count
                    ),
                    "source_timeframe_real_completed_bars": (
                        real_completed_bars
                    ),
                    "created_manipulations_per_1000_real_completed_bars": (
                        None
                        if real_completed_bars == 0
                        else (
                            1000.0
                            * created_manipulation_count
                            / real_completed_bars
                        )
                    ),
                }
            )
        manipulation_path_funnel = self._identity_funnel_rows(
            self._group5_manipulation_funnel_ids,
            denominator_stage="manipulation",
        )
        qualified_zone_path_funnel = self._identity_funnel_rows(
            self._group5_zone_funnel_ids,
            denominator_stage="qualified_zone",
        )
        group5_identity_conservation = self._group5_identity_conservation()
        favr_observation_chain = self._favr_observation_chain()
        manipulation_conservation = self._manipulation_conservation()
        return {
            "schema_version": self.schema_version,
            "scope": "typed_eye_only_no_brain_action_pnl_mbo_or_future",
            "execution_reality_status": "not_evaluated",
            "window": {
                "configured_start": _clock_text(self.start),
                "configured_end_exclusive": _clock_text(self.end_exclusive),
                "coverage_start": _clock_text(self.coverage_start),
                "first_observation": _clock_text(self._first_asof),
                "last_observation": _clock_text(self._last_asof),
                "observations": self._observation_count,
                "left_boundary_censored_total": left_boundary_total,
                "left_boundary_censored_entities": left_boundary_rows,
            },
            "statistics_transport": {
                "mode": "authoritative_delta_with_cutoff_snapshot_fallback",
                "counts": dict(sorted(self._transport_counts.items())),
                "active_transition_fingerprints": len(
                    self._last_transition_by_entity
                ),
                "terminal_identity_tombstones": len(
                    self._terminal_transition_by_entity
                ),
                "open_latest_states": len(self._open_latest),
            },
            "data_clock": {
                "by_timeframe": clock_rows,
                "reducer_updates": reducer_rows,
                "reader_anomalies": dict(sorted(self._reader_anomalies.items())),
                "observation_anomalies": dict(
                    sorted(self._observation_anomalies.items())
                ),
                "integrity": {
                    "clock_authority": "reader_update",
                    "duplicate_anomalies": reader_anomaly_total("duplicate"),
                    "out_of_order_anomalies": reader_anomaly_total(
                        "out_of_order", "out-of-order"
                    ),
                    "contract_resets": reader_anomaly_total("contract_change"),
                    "data_gap_resets": reader_anomaly_total("data_gap"),
                    "reducer_exception_anomalies": 0,
                    "fatal_exception_count": 0,
                    "status": "successful_finalize_no_fatal_exception",
                },
            },
            "denominators": [
                {
                    "group": "displacement",
                    "primitive": "displacement",
                    "name": "seed_rejection_reason",
                    "denominator_status": "not_exposed",
                    "count": None,
                },
                {
                    "group": "group3",
                    "primitive": "order_block",
                    "name": "admission_attempt",
                    "denominator_status": (
                        "producer_exposed"
                        if self._ob_funnel_contract_seen
                        else "not_exposed"
                    ),
                    "count": (
                        self._ob_funnel_attempts
                        if self._ob_funnel_contract_seen
                        else None
                    ),
                },
                {
                    "group": "group5",
                    "primitive": "path_sequence",
                    "name": "source_admission_reason",
                    "denominator_status": "not_exposed",
                    "count": None,
                },
                {
                    "group": "group4",
                    "primitive": "dealing_range",
                    "name": "blocked_pair_candidate",
                    "denominator_status": (
                        "producer_exposed"
                        if self._range_funnel_contract_seen
                        else "not_exposed"
                    ),
                    "count": (
                        range_geometry_pairs
                        if self._range_funnel_contract_seen
                        else None
                    ),
                },
                {
                    "group": "group4",
                    "primitive": "manipulation_source",
                    "name": "raw_crossed_source",
                    "denominator_status": (
                        "producer_exposed"
                        if self._disposition_contract_seen
                        else "not_exposed"
                    ),
                    "count": (
                        disposition_total
                        if self._disposition_contract_seen
                        else None
                    ),
                },
            ],
            "funnels": self._funnel_rows(),
            "categorical_counts": categorical_rows,
            "cohort_strata_revisions": [
                {
                    "group": group,
                    "primitive": primitive,
                    "later_transition_strata_changes": count,
                }
                for (group, primitive), count in sorted(
                    self._cohort_strata_revision_counts.items()
                )
            ],
            "numeric_distributions": numeric_rows,
            "displacement": {
                "started": len(self._displacement_started),
                "ever_active": len(self._displacement_active),
                "started_not_active_by_terminal_reason": dict(
                    sorted(started_not_active.items())
                ),
                "terminal_reasons": dict(
                    sorted(Counter(self._displacement_terminal_reason.values()).items())
                ),
                "duration_minutes": self._displacement_durations.payload(),
                "total_interruption_bars": (
                    self._displacement_interruptions.payload()
                ),
                "same_bar_terminal_then_restart": (
                    self._displacement_same_bar_restarts
                ),
                "started_never_active_diagnostics": (
                    self._displacement_never_active_diagnostics()
                ),
            },
            "group3": {
                "order_block_admission": {
                    "denominator_status": (
                        "producer_exposed"
                        if self._ob_funnel_contract_seen
                        else "not_exposed"
                    ),
                    "attempts": (
                        self._ob_funnel_attempts
                        if self._ob_funnel_contract_seen
                        else None
                    ),
                    "stage_totals": dict(sorted(self._ob_funnel_stages.items())),
                    "mutually_exclusive_outcomes": dict(
                        sorted(self._ob_funnel_outcomes.items())
                    ),
                    "conservation": {
                        "attempts": self._ob_funnel_attempts,
                        "outcomes": ob_outcomes_total,
                        "created_stage": ob_created,
                        "created_outcome": ob_created_outcomes,
                        "balanced": (
                            self._ob_funnel_attempts == ob_outcomes_total
                            and ob_created == ob_created_outcomes
                        ),
                    },
                }
            },
            "group12": {
                "liquidity_pool_structural_metadata": {
                    "denominator_status": pool_metadata_status,
                    "pool_entities_seen": pool_metadata_total,
                    "structural_rank_exposed": len(self._pool_rank_exposed),
                    "structural_rank_missing": len(self._pool_rank_missing),
                }
            },
            "group4": {
                "mature_ranges_by_month": dict(sorted(self._mature_months.items())),
                "mature_range_formation_atr_tertiles": (
                    self._range_formation_atr_tertiles()
                ),
                "range_gate_status": (
                    "producer_exposed"
                    if self._range_funnel_contract_seen
                    else "not_exposed"
                ),
                "range_formation_funnel": {
                    "denominator_status": (
                        "producer_exposed"
                        if self._range_funnel_contract_seen
                        else "not_exposed"
                    ),
                    "counting_basis": "evaluation_weighted",
                    "completed_h1_snapshots": self._range_funnel_snapshots,
                    "pair_stage_totals": dict(sorted(self._range_pair_counts.items())),
                    "conservation": {
                        "live_structural_pairs": range_live_pairs,
                        "invalid_plus_geometry": (
                            range_invalid_pairs + range_geometry_pairs
                        ),
                        "geometry_valid_pairs": range_geometry_pairs,
                        "classified_geometry_pairs": range_classified_geometry,
                        "eligible_pairs": range_eligible,
                        "forming_selected": range_selected,
                        "balanced": (
                            range_live_pairs
                            == range_invalid_pairs + range_geometry_pairs
                            and range_geometry_pairs == range_classified_geometry
                            and range_selected <= range_eligible
                        ),
                    },
                },
                "range_gate_observations": self._range_gate_observations,
                "range_gate_counting_basis": "evaluation_weighted",
                "left_boundary_range_gate_evaluations_excluded": (
                    self._left_boundary_range_gate_evaluations
                ),
                "unmet_gate_combinations": dict(
                    sorted(self._range_unmet_combinations.items())
                ),
                "unmet_gate_cardinality": dict(
                    sorted(self._range_unmet_sizes.items())
                ),
                "unique_range_episode_diagnostics": {
                    "counting_basis": "unique_range_episode",
                    "first_evaluation": self._range_episode_diagnostic_summary(
                        self._range_gate_first_by_id
                    ),
                    "latest_evaluation": self._range_episode_diagnostic_summary(
                        self._range_gate_latest_by_id
                    ),
                },
                "range_transition_reasons": range_reasons,
                "visible_eligible_sources": visible_eligible_unique,
                "visible_eligible_sources_by_strata": visible_eligible_rows,
                "visible_eligible_sources_by_strata_counting_basis": (
                    "unique_within_each_stratum_non_additive_when_a_source_"
                    "identity_changes_reported_structural_metadata"
                ),
                "visible_eligible_sources_by_strata_row_sum": (
                    visible_eligible_row_sum
                ),
                "visible_eligible_sources_excess_assignments_above_union": (
                    visible_eligible_row_sum - visible_eligible_unique
                ),
                "manipulation_rates_by_source_timeframe": (
                    manipulation_rates_by_source_timeframe
                ),
                "manipulation_rate_counting_basis": {
                    "eligible_sources": (
                        "unique source inventory identities unioned across "
                        "all reported strata within each source timeframe"
                    ),
                    "conversion": (
                        "unique eligible primary source inventory identities "
                        "which created an in-window manipulation divided by "
                        "unique eligible source identities"
                    ),
                    "per_1000_bars": (
                        "in-window created manipulation identities per 1000 "
                        "real completed bars of the corresponding source "
                        "timeframe; synthetic bars are excluded"
                    ),
                },
                "source_dispositions": disposition_rows,
                "source_disposition_conservation": {
                    "raw_crossed_source_ids": disposition_total,
                    "raw_crossed_source_clock_ids": disposition_total,
                    "raw_crossed_unique_source_ids": disposition_unique_sources,
                    "sum_across_ten_dispositions": disposition_sum,
                    "expected_disposition_kinds": sorted(
                        _GROUP4_SOURCE_DISPOSITIONS
                    ),
                    "totals_by_disposition": disposition_totals,
                    "balanced": disposition_total == disposition_sum,
                },
                "manipulation_conservation": manipulation_conservation,
                "selected_primary_to_episode_join": {
                    "identity": "(swept_at_iso8601, source_inventory_item_id)",
                    "selected_primary_source_clocks": len(
                        selected_primary_identities
                    ),
                    "created_episode_source_clocks": len(
                        created_episode_identities
                    ),
                    "swept_created": manipulation_conservation["swept_created"],
                    "selected_without_episode": len(selected_without_episode),
                    "episode_without_selected": len(episode_without_selected),
                    "bad_ids": {
                        "selected_without_episode": self._bounded_ids(
                            f"{clock}|{source_id}"
                            for clock, source_id in selected_without_episode
                        ),
                        "episode_without_selected": self._bounded_ids(
                            f"{clock}|{source_id}"
                            for clock, source_id in episode_without_selected
                        ),
                    },
                    "balanced": (
                        selected_primary_identities
                        == created_episode_identities
                        and len(created_episode_identities)
                        == manipulation_conservation["swept_created"]
                    ),
                },
            },
            "group5": {
                "denominator_status": (
                    "producer_exposed" if self._group5_contract_seen else "not_exposed"
                ),
                "first_pullback_paths": sum(paths_after_pullback.values()),
                "after_first_pullback": paths_after_pullback,
                "trigger_definition": sorted(_GROUP5_TRIGGER_STEPS),
                "path_order_errors": self._group5_path_order_errors,
                "terminal_reasons": dict(sorted(self._path_terminal_reasons.items())),
                "terminal_classification": {
                    "complete": int(self._path_terminal_classes.get("complete", 0)),
                    "interrupted": int(
                        self._path_terminal_classes.get("interrupted", 0)
                    ),
                },
                "group4_manipulations_by_source_timeframe": dict(
                    sorted(manipulations_by_source_timeframe.items())
                ),
                "group4_manipulations_by_source_timeframe_basis": (
                    "all_observed_identities_including_left_warmup_context"
                ),
                "group4_in_window_manipulation_cohort_by_source_timeframe": dict(
                    sorted(cohort_manipulations_by_source_timeframe.items())
                ),
                "group4_in_window_manipulation_cohort_counting_basis": (
                    "unique_manipulation_identity_created_inside_registered_window"
                ),
                "manipulations_entering_group5_by_source_timeframe": dict(
                    sorted(group5_manipulations_by_source_timeframe.items())
                ),
                "manipulation_to_group5_join_status": (
                    "identity_join_exposed"
                    if self._group5_contract_seen
                    else "not_exposed"
                ),
                "manipulation_to_path_identity_funnel": (
                    manipulation_path_funnel
                ),
                "qualified_zone_to_terminal_identity_funnel": (
                    qualified_zone_path_funnel
                ),
                "exact_identity_conservation": group5_identity_conservation,
                "favr_observation_chain": favr_observation_chain,
            },
            "case_selection": {
                "method": (
                    "all_recognized_mature_first_smallest_sha256_then_"
                    "registered_strata_round_robin_with_global_episode_"
                    "uniqueness"
                ),
                "future_or_pnl_used": False,
                "frozen_strata": list(_FROZEN_CASE_STRATA),
                "retained_by_stratum": case_counts,
                "selected_by_stratum": selected_case_counts,
                "category_coverage": category_coverage,
                "category_status": {
                    "obvious_mature_looking_but_rejected": (
                        "requires_blind_image_classification_not_inferred"
                    )
                },
                "candidate_events_seen": self._case_candidate_count,
                "retained_candidates": sum(case_counts.values()),
                "selected": len(cases),
                "minimum_requested": _CASE_MINIMUM,
                "maximum": _CASE_LIMIT,
                "status": case_status,
                "status_scope": (
                    "selected_case_count_only; category coverage is reported "
                    "separately and visual maturity classification is not "
                    "inferred by the scanner"
                ),
            },
            "case_index": cases,
        }


__all__ = ["EyeAuthorityStatistics"]
