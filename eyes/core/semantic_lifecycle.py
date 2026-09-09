"""Replayable lifecycle projection for the canonical semantic foundation.

This module deliberately does not detect market semantics.  It accepts only
explicitly normalized lifecycle transitions (optionally adapted from an
already-authoritative :class:`MarketEvent`, ``RelationState`` or
``DeliveryPhase``) and projects immutable generation state.

The foundation version is intentionally independent from the current Eye
registry version.  Adopting it in the authoritative event pipeline therefore
requires an explicit registry migration instead of silently changing v1.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import pandas as pd

from .foundation_registry import FOUNDATION_VERSION
from shares.core.market_clock import next_registered_native_completion
from contract.market import (
    Direction,
    FrozenDict,
    Timeframe,
    aware_timestamp,
    content_hash,
)
from contract.eye import MarketEvent

if TYPE_CHECKING:
    from .market_state import DeliveryPhase, RelationState


def canonical_semantic_id(namespace: str, *identity: object) -> str:
    """Return one version-bound, deterministic semantic identity."""

    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("canonical semantic namespace is required")
    digest = content_hash(
        {
            "semantic_version": FOUNDATION_VERSION,
            "namespace": namespace,
            "identity": identity,
        }
    )
    return f"{namespace}:{digest[:32]}"


class NormalizedTransitionKind(str, Enum):
    REAL_BAR_COMPLETED = "real_bar_completed"
    RESET = "reset"
    LIQUIDITY_LEVEL_CREATED = "liquidity_level_created"
    LIQUIDITY_TOUCHED = "liquidity_touched"
    LIQUIDITY_PENETRATED = "liquidity_penetrated"
    LIQUIDITY_SWEEP_TERMINAL = "liquidity_sweep_terminal"
    LIQUIDITY_ACCEPTANCE_TERMINAL = "liquidity_acceptance_terminal"
    LIQUIDITY_UNRESOLVED_TERMINAL = "liquidity_unresolved_terminal"
    LIQUIDITY_LEVEL_REARMABLE = "liquidity_level_rearmable"
    LIQUIDITY_LEVEL_REARMED = "liquidity_level_rearmed"
    LIQUIDITY_LEVEL_RETIRED = "liquidity_level_retired"
    LIQUIDITY_LEVEL_ARCHIVED = "liquidity_level_archived"
    STRUCTURE_GENERATION_STARTED = "structure_generation_started"
    STRUCTURE_GENERATION_CONFIRMED = "structure_generation_confirmed"
    STRUCTURE_GENERATION_EVIDENCE = "structure_generation_evidence"
    STRUCTURE_GENERATION_TERMINATED = "structure_generation_terminated"
    MSS_TRANSITION_STARTED = "mss_transition_started"
    STRUCTURE_TRANSITION_EVIDENCE = "structure_transition_evidence"
    STRUCTURE_TRANSITION_CONFIRMED = "structure_transition_confirmed"
    STRUCTURE_DIRECTION_RESUMED = "structure_direction_resumed"
    RELATION_OBSERVED = "relation_observed"
    RELATION_TERMINATED = "relation_terminated"
    DELIVERY_PHASE_OBSERVED = "delivery_phase_observed"
    DELIVERY_PHASE_TERMINATED = "delivery_phase_terminated"
    BOUNDARY_ATTACK_OBSERVED = "boundary_attack_observed"


class LiquidityLevelLifecycle(str, Enum):
    ACTIVE = "active"
    DISARMED = "disarmed"
    REARMABLE = "rearmable"
    REARMED = "rearmed"
    RETIRED = "retired"
    ARCHIVED = "archived"


class LiquidityInteractionLifecycle(str, Enum):
    ARMED = "armed"
    TOUCHED = "touched"
    PENETRATED = "penetrated"
    TERMINAL = "terminal"


class LiquidityInteractionTerminal(str, Enum):
    SWEEP = "sweep"
    ACCEPTANCE = "acceptance"
    UNRESOLVED = "unresolved"
    CENSORED = "censored"
    EXPIRED = "expired"


class StructureScope(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class StructureGenerationLifecycle(str, Enum):
    FORMING = "forming"
    CONFIRMED = "confirmed"
    TERMINATED = "terminated"


class StructureTransitionLifecycle(str, Enum):
    STARTED = "started"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CENSORED = "censored"


class GenerationLifecycle(str, Enum):
    ACTIVE = "active"
    TERMINATED = "terminated"


_RESET_REASONS = frozenset({"data_reset", "semantic_reset", "contract_reset"})
_LEVEL_RETIREMENT_REASONS = frozenset(
    {
        "reference_rollover",
        "supersession",
        "contract_reset",
        "source_retired",
        "source_range_terminated",
        "structure_generation_terminated",
    }
)
_LEVEL_TERMINAL_REASONS = frozenset(
    {*_LEVEL_RETIREMENT_REASONS, *_RESET_REASONS, "acceptance"}
)
_INTERACTION_UNRESOLVED_REASONS = frozenset({"unresolved"})
_STRUCTURE_TERMINATION_REASONS = frozenset(
    {
        "protected_break_accepted",
        "scope_rollover",
        "contract_rollover",
        "data_reset",
        "semantic_reset",
        "superseded",
    }
)
_RELATION_ROLES = frozenset(
    {
        "aligned_expansion",
        "parent_retracement",
        "reversal_attempt",
        "parent_transition",
        "balance_inside_parent",
        "unresolved",
    }
)
_DELIVERY_PHASES = frozenset(
    {"balance", "expansion", "retracement", "reversal_attempt", "transition"}
)
_RELATION_TERMINATION_REASONS = frozenset(
    {
        "child_realigned",
        "parent_invalidated",
        "parent_rollover",
        "relation_reclassified",
        "contract_reset",
        "data_reset",
        "semantic_reset",
    }
)
_DELIVERY_TERMINATION_REASONS = frozenset(
    {
        "phase_changed",
        "parent_structure_terminated",
        "contract_reset",
        "data_reset",
        "semantic_reset",
    }
)
_STRUCTURE_TRANSITION_CENSOR_REASONS = frozenset(
    {
        *_RESET_REASONS,
        *(
            _STRUCTURE_TERMINATION_REASONS
            - {"protected_break_accepted"}
        ),
    }
)
_TIMEFRAME_INTERVAL = {
    Timeframe.M1: pd.Timedelta(1, unit="min"),
    Timeframe.M5: pd.Timedelta(5, unit="min"),
    Timeframe.M15: pd.Timedelta(15, unit="min"),
    Timeframe.H1: pd.Timedelta(1, unit="h"),
    Timeframe.H4: pd.Timedelta(4, unit="h"),
}
_TIMEFRAME_ANCHOR_MINUTE = {
    Timeframe.M1: 0,
    Timeframe.M5: 0,
    Timeframe.M15: 0,
    Timeframe.H1: 0,
    Timeframe.H4: 18 * 60,
}
LIFECYCLE_STATE_SCHEMA_VERSION = 3
LIFECYCLE_CHECKPOINT_SCHEMA_VERSION = 4

# Exact aliases frozen by foundation_v2_0.yaml.  This is deliberately not a
# prefix/family heuristic: adding an alias requires a versioned registry edit.
REARMABLE_LIQUIDITY_SOURCE_KINDS = frozenset(
    {
        "confirmed_swing",
        "structural_swing",
        "previous_session_high",
        "previous_session_low",
        "previous_day_high",
        "previous_day_low",
        "previous_week_high",
        "previous_week_low",
        "range_boundary",
        "mature_range_boundary",
    }
)
_STRUCTURE_OWNED_SWING_SOURCE_KINDS = frozenset(
    {"confirmed_swing", "structural_swing"}
)


def _unique_text(values: Iterable[object], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty text identities")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} identities must be unique")
    return result


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"normalized transition requires {key}")
    return value


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"normalized transition {key} is invalid")
    return value


def _required_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = payload.get(key)
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"normalized transition {key} is invalid")
    return value


def _optional_nonnegative_number(
    payload: Mapping[str, Any], key: str
) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"normalized transition {key} is invalid")
    return float(value)


def _payload_clock(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: pd.Timestamp | None = None,
) -> pd.Timestamp | None:
    value = payload.get(key)
    if value is None:
        return default
    return aware_timestamp(value, name=f"normalized_transition.{key}")


@dataclass(frozen=True)
class NormalizedLifecycleTransition:
    """One explicit, causally clocked input to the lifecycle projection."""

    fact_id: str
    kind: NormalizedTransitionKind
    known_at: pd.Timestamp
    timeframe: Timeframe | None
    source_event_ids: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)
    sequence_no: int = 0
    semantic_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.fact_id, str) or not self.fact_id.strip():
            raise ValueError("normalized transition fact_id is required")
        object.__setattr__(self, "kind", NormalizedTransitionKind(self.kind))
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(self.known_at, name="normalized_transition.known_at"),
        )
        if self.timeframe is not None:
            object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        if self.kind is not NormalizedTransitionKind.RESET and self.timeframe is None:
            raise ValueError("non-reset normalized transition requires timeframe")
        object.__setattr__(
            self,
            "source_event_ids",
            _unique_text(self.source_event_ids, name="source_event_ids"),
        )
        if not self.source_event_ids:
            raise ValueError("normalized transition requires explicit source events")
        object.__setattr__(self, "payload", FrozenDict(self.payload))
        if type(self.sequence_no) is not int or self.sequence_no < 0:
            raise ValueError("normalized transition sequence must be non-negative")
        if self.semantic_version != FOUNDATION_VERSION:
            raise ValueError("normalized transition semantic version mismatch")

    @classmethod
    def from_market_event(
        cls,
        event: MarketEvent,
        *,
        kind: NormalizedTransitionKind,
        payload: Mapping[str, Any] | None = None,
    ) -> "NormalizedLifecycleTransition":
        """Adapt one authoritative event without inferring a new detector.

        Callers must explicitly select the foundation transition kind.  The
        event evidence is copied and may be completed with versioned adapter
        fields in ``payload``; no SMC classification is guessed here.
        """

        if not isinstance(event, MarketEvent):
            raise TypeError("market-event adapter requires MarketEvent")
        merged = dict(event.evidence)
        if payload is not None:
            merged.update(payload)
        if kind is NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED:
            source_level_id = merged.get("level_id")
            if "source_identity" not in merged and isinstance(source_level_id, str):
                merged["source_identity"] = source_level_id
            if "side" not in merged and event.side is not None:
                merged["side"] = event.side
        source_ids = tuple(
            dict.fromkeys((event.event_id, *event.source_event_ids))
        )
        return cls(
            fact_id=canonical_semantic_id(
                "normalized-fact", event.event_id, NormalizedTransitionKind(kind).value
            ),
            kind=kind,
            known_at=event.known_at,
            timeframe=(None if kind is NormalizedTransitionKind.RESET else event.timeframe),
            source_event_ids=source_ids,
            payload=merged,
            sequence_no=event.sequence_no,
        )

    @classmethod
    def from_relation_state(
        cls,
        relation: "RelationState",
        *,
        parent_structure_generation_id: str,
        child_structure_generation_id: str,
        source_event_ids: Sequence[str],
    ) -> "NormalizedLifecycleTransition":
        known_at = getattr(relation, "known_at", None)
        if known_at is None:
            raise ValueError("relation observation requires known_at")
        parent_tf = Timeframe(getattr(relation, "parent_tf"))
        child_tf = Timeframe(getattr(relation, "child_tf"))
        role_value = getattr(getattr(relation, "role"), "value", getattr(relation, "role"))
        relation_id = getattr(relation, "relation_id")
        payload = {
            "source_relation_id": relation_id,
            "parent_tf": parent_tf.value,
            "child_tf": child_tf.value,
            "parent_structure_generation_id": parent_structure_generation_id,
            "child_structure_generation_id": child_structure_generation_id,
            "role": role_value,
            "parent_direction": (
                getattr(relation.parent_direction, "value", relation.parent_direction)
            ),
            "child_direction": (
                getattr(relation.child_direction, "value", relation.child_direction)
            ),
            "relation_digest": content_hash(relation),
        }
        return cls(
            fact_id=canonical_semantic_id(
                "relation-observation",
                relation_id,
                known_at,
                parent_structure_generation_id,
                child_structure_generation_id,
                role_value,
            ),
            kind=NormalizedTransitionKind.RELATION_OBSERVED,
            known_at=known_at,
            timeframe=child_tf,
            source_event_ids=tuple(source_event_ids),
            payload=payload,
        )

    @classmethod
    def from_delivery_phase(
        cls,
        phase: "DeliveryPhase",
        *,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
        parent_structure_generation_id: str,
        origin_event_id: str,
        source_event_ids: Sequence[str],
        current_price_ticks: int,
        extension_ticks: float | None = None,
        retracement_ticks: float | None = None,
    ) -> "NormalizedLifecycleTransition":
        phase_value = getattr(phase, "value", phase)
        payload = {
            "phase": phase_value,
            "parent_structure_generation_id": parent_structure_generation_id,
            "origin_event_id": origin_event_id,
            "current_price_ticks": current_price_ticks,
            "extension_ticks": extension_ticks,
            "retracement_ticks": retracement_ticks,
        }
        return cls(
            fact_id=canonical_semantic_id(
                "delivery-observation",
                Timeframe(timeframe).value,
                known_at,
                parent_structure_generation_id,
                phase_value,
                origin_event_id,
            ),
            kind=NormalizedTransitionKind.DELIVERY_PHASE_OBSERVED,
            known_at=known_at,
            timeframe=timeframe,
            source_event_ids=tuple(source_event_ids),
            payload=payload,
        )


@dataclass(frozen=True)
class RealBarClock:
    timeframe: Timeframe
    count: int
    last_completed_at: pd.Timestamp
    last_bar_event_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(
            self,
            "last_completed_at",
            aware_timestamp(self.last_completed_at, name="real_bar.last_completed_at"),
        )
        if self.count < 1 or not self.last_bar_event_id:
            raise ValueError("real-bar clock is invalid")


@dataclass(frozen=True)
class RegisteredBarClock:
    """Inclusive native-bar transport clock for one timeframe and epoch."""

    timeframe: Timeframe
    count: int
    last_completed_at: pd.Timestamp
    last_bar_event_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(
            self,
            "last_completed_at",
            aware_timestamp(
                self.last_completed_at,
                name="registered_bar.last_completed_at",
            ),
        )
        if self.count < 1 or not self.last_bar_event_id:
            raise ValueError("registered-bar clock is invalid")


@dataclass(frozen=True)
class LiquidityLevelState:
    level_id: str
    source_timeframe: Timeframe
    source_kind: str
    source_identity: str
    side: str
    price_ticks: int
    lower_bound_ticks: int
    upper_bound_ticks: int
    tick_size: float
    lifecycle: LiquidityLevelLifecycle
    created_at: pd.Timestamp
    updated_at: pd.Timestamp
    active_generation_id: str | None
    interaction_generation_ids: tuple[str, ...]
    last_terminal_generation_id: str | None = None
    rearmable_from_generation_id: str | None = None
    rearmable_fact_id: str | None = None
    rearm_departure_bar_event_id: str | None = None
    rearm_departure_ticks: int | None = None
    rearmed_from_generation_id: str | None = None
    rearmable_at: pd.Timestamp | None = None
    retired_at: pd.Timestamp | None = None
    retirement_reason: str | None = None
    superseded_by_level_id: str | None = None
    archived_at: pd.Timestamp | None = None
    archive_reason: str | None = None
    source_event_ids: tuple[str, ...] = ()
    price_anchor_rule: str = "exact_event_price"
    owner_structure_generation_id: str | None = None
    semantic_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_timeframe", Timeframe(self.source_timeframe)
        )
        object.__setattr__(self, "lifecycle", LiquidityLevelLifecycle(self.lifecycle))
        for name in (
            "created_at",
            "updated_at",
            "rearmable_at",
            "retired_at",
            "archived_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, aware_timestamp(value, name=f"liquidity_level.{name}")
                )
        object.__setattr__(
            self,
            "interaction_generation_ids",
            _unique_text(self.interaction_generation_ids, name="interaction_generation_ids"),
        )
        object.__setattr__(
            self, "source_event_ids", _unique_text(self.source_event_ids, name="source_event_ids")
        )
        if (
            not self.level_id
            or not self.source_kind
            or not self.source_identity
            or self.side not in {"above", "below"}
            or type(self.price_ticks) is not int
            or type(self.lower_bound_ticks) is not int
            or type(self.upper_bound_ticks) is not int
            or not 0 < self.lower_bound_ticks <= self.price_ticks <= self.upper_bound_ticks
            or self.price_anchor_rule not in {
                "exact_event_price",
                "near_side_tradable_zone_boundary_for_nontradable_midpoint",
            }
            or (
                self.owner_structure_generation_id is not None
                and (
                    not isinstance(self.owner_structure_generation_id, str)
                    or not self.owner_structure_generation_id
                )
            )
            or not math.isfinite(float(self.tick_size))
            or self.tick_size <= 0.0
            or self.updated_at < self.created_at
            or self.semantic_version != FOUNDATION_VERSION
        ):
            raise ValueError("liquidity-level state is invalid")
        if not self.source_event_ids:
            raise ValueError("liquidity level requires exact source ancestry")
        if not self.interaction_generation_ids:
            raise ValueError("liquidity level requires an interaction history")
        active = self.lifecycle in {
            LiquidityLevelLifecycle.ACTIVE,
            LiquidityLevelLifecycle.REARMED,
        }
        if active != (self.active_generation_id is not None):
            raise ValueError("liquidity level active generation is inconsistent")
        if (
            self.active_generation_id is not None
            and self.active_generation_id not in self.interaction_generation_ids
        ):
            raise ValueError("liquidity level active generation is not registered")
        if (
            self.last_terminal_generation_id is not None
            and self.last_terminal_generation_id
            not in self.interaction_generation_ids
        ):
            raise ValueError("liquidity level terminal generation is not registered")
        if (
            self.lifecycle
            in {
                LiquidityLevelLifecycle.DISARMED,
                LiquidityLevelLifecycle.REARMABLE,
                LiquidityLevelLifecycle.REARMED,
                LiquidityLevelLifecycle.RETIRED,
                LiquidityLevelLifecycle.ARCHIVED,
            }
            and self.last_terminal_generation_id is None
        ):
            raise ValueError("liquidity level lacks its last terminal generation")
        terminal = self.lifecycle in {
            LiquidityLevelLifecycle.RETIRED,
            LiquidityLevelLifecycle.ARCHIVED,
        }
        if terminal != (self.retired_at is not None and self.retirement_reason is not None):
            raise ValueError("liquidity-level retirement state is inconsistent")
        if self.lifecycle is LiquidityLevelLifecycle.ARCHIVED:
            if (
                self.archived_at is None
                or self.archived_at != self.updated_at
                or not self.archive_reason
                or self.archive_reason not in _RESET_REASONS
                or self.retired_at is None
                or self.retired_at > self.archived_at
            ):
                raise ValueError("archived liquidity level lacks archive provenance")
        elif self.archived_at is not None or self.archive_reason is not None:
            raise ValueError("non-archived liquidity level has archive metadata")
        if terminal:
            if (
                self.retired_at is None
                or not self.created_at <= self.retired_at <= self.updated_at
                or self.retirement_reason not in _LEVEL_TERMINAL_REASONS
                or (
                    self.lifecycle is LiquidityLevelLifecycle.RETIRED
                    and self.retired_at != self.updated_at
                )
            ):
                raise ValueError("liquidity-level retirement clock or reason is invalid")
        if (
            self.superseded_by_level_id is not None
            and self.retirement_reason != "supersession"
        ):
            raise ValueError("liquidity-level supersession reason is inconsistent")
        has_rearmable_provenance = all(
            value is not None
            for value in (
                self.rearmable_at,
                self.rearmable_from_generation_id,
                self.rearmable_fact_id,
                self.rearm_departure_bar_event_id,
                self.rearm_departure_ticks,
            )
        )
        if self.lifecycle in {
            LiquidityLevelLifecycle.REARMABLE,
            LiquidityLevelLifecycle.REARMED,
        }:
            if (
                not has_rearmable_provenance
                or self.rearm_departure_ticks < 1
                or self.rearmable_at < self.created_at
                or self.rearmable_at > self.updated_at
                or self.rearmable_from_generation_id
                != self.last_terminal_generation_id
                or self.rearm_departure_bar_event_id
                not in self.source_event_ids
            ):
                raise ValueError("rearmable liquidity level lacks exact provenance")
        elif any(
            value is not None
            for value in (
                self.rearmable_at,
                self.rearmable_from_generation_id,
                self.rearmable_fact_id,
                self.rearm_departure_bar_event_id,
                self.rearm_departure_ticks,
            )
        ):
            raise ValueError("non-rearmable level carries live rearm provenance")
        if (
            self.lifecycle is LiquidityLevelLifecycle.REARMED
        ) != (self.rearmed_from_generation_id is not None):
            raise ValueError("rearmed liquidity level prior generation is inconsistent")
        if (
            self.rearmed_from_generation_id is not None
            and self.rearmed_from_generation_id
            != self.last_terminal_generation_id
        ):
            raise ValueError("rearmed level does not bind its terminal generation")


@dataclass(frozen=True)
class InteractionConstituent:
    """One definitional BAR role, never a later temporal relationship."""

    bar_event_id: str
    role: str
    known_at: pd.Timestamp
    high_ticks: int
    low_ticks: int
    close_ticks: int

    def __post_init__(self) -> None:
        if not self.bar_event_id or self.role not in {
            "penetration",
            "reentry",
            "hold",
            "confirmation",
        }:
            raise ValueError("interaction constituent role or BAR is invalid")
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(self.known_at, name="interaction_constituent.known_at"),
        )
        if (
            type(self.high_ticks) is not int
            or type(self.low_ticks) is not int
            or type(self.close_ticks) is not int
            or not 0 < self.low_ticks <= self.close_ticks <= self.high_ticks
        ):
            raise ValueError("interaction constituent BAR geometry is invalid")


def _close_is_outside(
    *,
    side: str,
    lower_bound_ticks: int,
    upper_bound_ticks: int,
    close_ticks: int,
) -> bool:
    return (
        close_ticks > upper_bound_ticks
        if side == "above"
        else close_ticks < lower_bound_ticks
    )


def _bar_penetration_ticks(
    *,
    side: str,
    lower_bound_ticks: int,
    upper_bound_ticks: int,
    high_ticks: int,
    low_ticks: int,
) -> int:
    return (
        max(0, high_ticks - upper_bound_ticks)
        if side == "above"
        else max(0, lower_bound_ticks - low_ticks)
    )


def _distinct_formation_bars(
    constituents: Sequence[InteractionConstituent],
) -> tuple[InteractionConstituent, ...]:
    bars: list[InteractionConstituent] = []
    seen: set[str] = set()
    for item in constituents:
        if item.bar_event_id in seen:
            prior = bars[-1] if bars else None
            if (
                prior is None
                or prior.bar_event_id != item.bar_event_id
                or (
                    prior.known_at,
                    prior.high_ticks,
                    prior.low_ticks,
                    prior.close_ticks,
                )
                != (
                    item.known_at,
                    item.high_ticks,
                    item.low_ticks,
                    item.close_ticks,
                )
            ):
                raise ValueError(
                    "interaction BAR roles must be contiguous and geometrically identical"
                )
            continue
        seen.add(item.bar_event_id)
        bars.append(item)
    return tuple(bars)


def _expected_formation_role_ledger(
    bars: Sequence[InteractionConstituent],
    *,
    terminal_state: LiquidityInteractionTerminal,
    side: str,
    lower_bound_ticks: int,
    upper_bound_ticks: int,
) -> tuple[tuple[str, str], ...]:
    ordered = tuple(bars)
    if not ordered or _bar_penetration_ticks(
        side=side,
        lower_bound_ticks=lower_bound_ticks,
        upper_bound_ticks=upper_bound_ticks,
        high_ticks=ordered[0].high_ticks,
        low_ticks=ordered[0].low_ticks,
    ) < 1:
        raise ValueError("interaction formation does not begin with penetration")
    outside = tuple(
        _close_is_outside(
            side=side,
            lower_bound_ticks=lower_bound_ticks,
            upper_bound_ticks=upper_bound_ticks,
            close_ticks=item.close_ticks,
        )
        for item in ordered
    )
    ledger: list[tuple[str, str]] = [(ordered[0].bar_event_id, "penetration")]
    if terminal_state is LiquidityInteractionTerminal.ACCEPTANCE:
        if (
            len(ordered) < 2
            or not outside[-1]
        ):
            raise ValueError(
                "Acceptance requires a later completed close held or confirmed outside"
            )
        if outside[0]:
            for item in ordered[1:]:
                ledger.append((item.bar_event_id, "hold"))
        else:
            ledger.append((ordered[0].bar_event_id, "reentry"))
            for item in ordered[1:-1]:
                ledger.append((item.bar_event_id, "hold"))
        ledger.append((ordered[-1].bar_event_id, "confirmation"))
        return tuple(ledger)
    if terminal_state is not LiquidityInteractionTerminal.SWEEP:
        raise ValueError("formation role ledger requires Sweep or Acceptance")
    if outside[-1]:
        raise ValueError("Sweep requires a completed close returned inside")
    if not outside[0]:
        ledger.append((ordered[0].bar_event_id, "reentry"))
    for index, item in enumerate(ordered[1:-1], start=1):
        ledger.append(
            (
                item.bar_event_id,
                "reentry"
                if not outside[index] and outside[index - 1]
                else "hold",
            )
        )
    if len(ordered) == 1:
        ledger.append((ordered[0].bar_event_id, "hold"))
    elif outside[-2]:
        ledger.append((ordered[-1].bar_event_id, "reentry"))
    ledger.append((ordered[-1].bar_event_id, "confirmation"))
    return tuple(ledger)


def _validate_formation_constituents(
    constituents: Sequence[InteractionConstituent],
    *,
    terminal_state: LiquidityInteractionTerminal,
    side: str,
    lower_bound_ticks: int,
    upper_bound_ticks: int,
) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """Validate exact terminal-specific BAR roles and return frozen path facts."""

    items = tuple(constituents)
    if not items:
        raise ValueError("resolved interaction requires formation ancestry")
    bars = _distinct_formation_bars(items)
    actual = tuple((item.bar_event_id, item.role) for item in items)
    expected = _expected_formation_role_ledger(
        bars,
        terminal_state=terminal_state,
        side=side,
        lower_bound_ticks=lower_bound_ticks,
        upper_bound_ticks=upper_bound_ticks,
    )
    if actual != expected:
        raise ValueError(
            "interaction constituent roles disagree with frozen BAR geometry"
        )
    penetrations = tuple(
        _bar_penetration_ticks(
            side=side,
            lower_bound_ticks=lower_bound_ticks,
            upper_bound_ticks=upper_bound_ticks,
            high_ticks=item.high_ticks,
            low_ticks=item.low_ticks,
        )
        for item in bars
    )
    inside = next(
        (
            item.known_at
            for item in bars
            if not _close_is_outside(
                side=side,
                lower_bound_ticks=lower_bound_ticks,
                upper_bound_ticks=upper_bound_ticks,
                close_ticks=item.close_ticks,
            )
        ),
        None,
    )
    outside = next(
        (
            item.known_at
            for item in bars
            if _close_is_outside(
                side=side,
                lower_bound_ticks=lower_bound_ticks,
                upper_bound_ticks=upper_bound_ticks,
                close_ticks=item.close_ticks,
            )
        ),
        None,
    )
    return max(penetrations), inside, outside


@dataclass(frozen=True)
class LiquidityInteractionGeneration:
    generation_id: str
    level_id: str
    source_timeframe: Timeframe
    interaction_timeframe: Timeframe
    level_side: str
    lower_bound_ticks: int
    upper_bound_ticks: int
    generation_number: int
    lifecycle: LiquidityInteractionLifecycle
    armed_at: pd.Timestamp
    known_at: pd.Timestamp
    updated_at: pd.Timestamp
    armed_real_bar_ordinal: int
    previous_generation_id: str | None = None
    rearm_fact_id: str | None = None
    first_touch_at: pd.Timestamp | None = None
    first_penetration_at: pd.Timestamp | None = None
    max_penetration_ticks: int = 0
    first_inside_close_at: pd.Timestamp | None = None
    first_outside_close_at: pd.Timestamp | None = None
    terminal_event_id: str | None = None
    terminal_state: LiquidityInteractionTerminal | None = None
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None
    terminal_real_bar_ordinal: int | None = None
    touch_bar_ids: tuple[str, ...] = ()
    constituents: tuple[InteractionConstituent, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    semantic_version: str = FOUNDATION_VERSION

    @property
    def started_at(self) -> pd.Timestamp:
        return self.armed_at

    @property
    def terminated_at(self) -> pd.Timestamp | None:
        return self.terminal_at

    @property
    def termination_reason(self) -> str | None:
        return self.terminal_reason

    @property
    def constituent_bar_ids(self) -> tuple[str, ...]:
        return tuple(item.bar_event_id for item in self.constituents)

    @property
    def constituent_bar_roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.constituents)

    def require_response_clock(self, known_at: pd.Timestamp) -> pd.Timestamp:
        clock = aware_timestamp(known_at, name="interaction_response.known_at")
        if self.terminal_at is None or clock <= self.terminal_at:
            raise ValueError("interaction response window starts strictly after terminal")
        return clock

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_timeframe", Timeframe(self.source_timeframe)
        )
        object.__setattr__(
            self,
            "interaction_timeframe",
            Timeframe(self.interaction_timeframe),
        )
        object.__setattr__(
            self, "lifecycle", LiquidityInteractionLifecycle(self.lifecycle)
        )
        if self.terminal_state is not None:
            object.__setattr__(
                self, "terminal_state", LiquidityInteractionTerminal(self.terminal_state)
            )
        for name in (
            "armed_at",
            "known_at",
            "updated_at",
            "first_touch_at",
            "first_penetration_at",
            "first_inside_close_at",
            "first_outside_close_at",
            "terminal_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, aware_timestamp(value, name=f"interaction.{name}")
                )
        object.__setattr__(
            self,
            "touch_bar_ids",
            _unique_text(self.touch_bar_ids, name="touch_bar_ids"),
        )
        object.__setattr__(self, "constituents", tuple(self.constituents))
        if any(
            not isinstance(item, InteractionConstituent)
            for item in self.constituents
        ):
            raise ValueError("interaction constituents must be typed BAR roles")
        if tuple(item.known_at for item in self.constituents) != tuple(
            sorted(item.known_at for item in self.constituents)
        ):
            raise ValueError("interaction constituent clocks must be ordered")
        object.__setattr__(
            self, "source_event_ids", _unique_text(self.source_event_ids, name="source_event_ids")
        )
        if (
            not self.generation_id
            or not self.level_id
            or self.level_side not in {"above", "below"}
            or type(self.lower_bound_ticks) is not int
            or type(self.upper_bound_ticks) is not int
            or not 0 < self.lower_bound_ticks <= self.upper_bound_ticks
            or type(self.generation_number) is not int
            or self.generation_number < 1
            or type(self.armed_real_bar_ordinal) is not int
            or self.armed_real_bar_ordinal < 0
            or type(self.max_penetration_ticks) is not int
            or self.max_penetration_ticks < 0
            or not self.armed_at <= self.known_at <= self.updated_at
            or self.updated_at < self.armed_at
            or self.semantic_version != FOUNDATION_VERSION
        ):
            raise ValueError("liquidity interaction generation is invalid")
        if not self.source_event_ids:
            raise ValueError("liquidity interaction requires exact source ancestry")
        if (
            self.generation_number == 1
            and (
                self.previous_generation_id is not None
                or self.rearm_fact_id is not None
            )
        ) or (
            self.generation_number > 1
            and (
                not isinstance(self.previous_generation_id, str)
                or not self.previous_generation_id
                or not isinstance(self.rearm_fact_id, str)
                or not self.rearm_fact_id
            )
        ):
            raise ValueError("liquidity interaction generation ancestry is inconsistent")
        path_clocks = tuple(
            value
            for value in (
                self.first_touch_at,
                self.first_penetration_at,
                self.first_inside_close_at,
                self.first_outside_close_at,
            )
            if value is not None
        )
        if any(
            value < self.armed_at or value > self.updated_at
            for value in path_clocks
        ) or (
            self.first_penetration_at is not None
            and (
                self.first_touch_at is None
                or self.first_penetration_at < self.first_touch_at
            )
        ):
            raise ValueError("liquidity interaction path clocks are inconsistent")
        sources = frozenset(self.source_event_ids)
        if any(value not in sources for value in self.touch_bar_ids) or any(
            item.bar_event_id not in sources for item in self.constituents
        ):
            raise ValueError("liquidity interaction BAR ancestry is incomplete")
        is_terminal = self.lifecycle is LiquidityInteractionLifecycle.TERMINAL
        terminal_fields = (
            self.terminal_event_id,
            self.terminal_state,
            self.terminal_at,
            self.terminal_reason,
            self.terminal_real_bar_ordinal,
        )
        if is_terminal != all(value is not None for value in terminal_fields):
            raise ValueError("interaction terminal state is inconsistent")
        if not is_terminal and any(value is not None for value in terminal_fields):
            raise ValueError("live interaction has terminal metadata")
        if is_terminal and (
            self.terminal_at != self.updated_at
            or type(self.terminal_real_bar_ordinal) is not int
            or self.terminal_real_bar_ordinal < self.armed_real_bar_ordinal
        ):
            raise ValueError("interaction terminal clock or ordinal is invalid")
        if is_terminal:
            permitted_reasons = {
                LiquidityInteractionTerminal.SWEEP: frozenset({"sweep"}),
                LiquidityInteractionTerminal.ACCEPTANCE: frozenset(
                    {"acceptance"}
                ),
                LiquidityInteractionTerminal.UNRESOLVED: (
                    _INTERACTION_UNRESOLVED_REASONS
                ),
                LiquidityInteractionTerminal.CENSORED: _RESET_REASONS,
                LiquidityInteractionTerminal.EXPIRED: (
                    _LEVEL_RETIREMENT_REASONS
                ),
            }[self.terminal_state]
            if self.terminal_reason not in permitted_reasons:
                raise ValueError("interaction terminal reason is not preregistered")
            if (
                self.terminal_state
                in {
                    LiquidityInteractionTerminal.SWEEP,
                    LiquidityInteractionTerminal.ACCEPTANCE,
                }
                and self.terminal_event_id not in sources
            ):
                raise ValueError("interaction terminal event is not exact ancestry")
        if self.lifecycle is LiquidityInteractionLifecycle.ARMED:
            if (
                self.first_touch_at is not None
                or self.first_penetration_at is not None
                or self.max_penetration_ticks != 0
                or self.touch_bar_ids
                or self.constituents
                or self.first_inside_close_at is not None
                or self.first_outside_close_at is not None
            ):
                raise ValueError("armed interaction contains later-stage facts")
        elif self.lifecycle is LiquidityInteractionLifecycle.TOUCHED:
            if (
                self.first_touch_at is None
                or not self.touch_bar_ids
                or self.first_penetration_at is not None
                or self.max_penetration_ticks != 0
                or self.constituents
                or self.first_inside_close_at is not None
                or self.first_outside_close_at is not None
            ):
                raise ValueError("touched interaction stage is inconsistent")
        elif self.lifecycle is LiquidityInteractionLifecycle.PENETRATED and (
            self.first_touch_at is None
            or not self.touch_bar_ids
            or self.first_penetration_at is None
            or self.max_penetration_ticks < 1
        ):
            raise ValueError("penetrated interaction stage is inconsistent")
        if self.terminal_state in {
            LiquidityInteractionTerminal.SWEEP,
            LiquidityInteractionTerminal.ACCEPTANCE,
        }:
            maximum, inside, outside = _validate_formation_constituents(
                self.constituents,
                terminal_state=self.terminal_state,
                side=self.level_side,
                lower_bound_ticks=self.lower_bound_ticks,
                upper_bound_ticks=self.upper_bound_ticks,
            )
            if (
                self.max_penetration_ticks != maximum
                or self.first_inside_close_at != inside
                or self.first_outside_close_at != outside
            ):
                raise ValueError(
                    "interaction path facts disagree with complete formation BARs"
                )
        elif (
            self.lifecycle is LiquidityInteractionLifecycle.PENETRATED
            and (
                len(self.constituents) != 1
                or self.constituents[0].role != "penetration"
                or _bar_penetration_ticks(
                    side=self.level_side,
                    lower_bound_ticks=self.lower_bound_ticks,
                    upper_bound_ticks=self.upper_bound_ticks,
                    high_ticks=self.constituents[0].high_ticks,
                    low_ticks=self.constituents[0].low_ticks,
                )
                > self.max_penetration_ticks
            )
        ):
            raise ValueError("penetrated interaction lacks exact BAR geometry")
        elif (
            self.lifecycle is LiquidityInteractionLifecycle.TERMINAL
            and self.terminal_state
            not in {
                LiquidityInteractionTerminal.SWEEP,
                LiquidityInteractionTerminal.ACCEPTANCE,
            }
        ):
            has_penetration = (
                self.first_penetration_at is not None
                or self.max_penetration_ticks != 0
                or bool(self.constituents)
            )
            has_touch = self.first_touch_at is not None or bool(self.touch_bar_ids)
            if has_penetration:
                if (
                    self.first_touch_at is None
                    or not self.touch_bar_ids
                    or self.first_penetration_at is None
                    or self.max_penetration_ticks < 1
                    or len(self.constituents) != 1
                    or self.constituents[0].role != "penetration"
                    or _bar_penetration_ticks(
                        side=self.level_side,
                        lower_bound_ticks=self.lower_bound_ticks,
                        upper_bound_ticks=self.upper_bound_ticks,
                        high_ticks=self.constituents[0].high_ticks,
                        low_ticks=self.constituents[0].low_ticks,
                    )
                    > self.max_penetration_ticks
                    or self.first_inside_close_at is not None
                    or self.first_outside_close_at is not None
                ):
                    raise ValueError(
                        "terminal interaction penetration stage is inconsistent"
                    )
            elif has_touch:
                if (
                    self.first_touch_at is None
                    or not self.touch_bar_ids
                    or self.first_penetration_at is not None
                    or self.max_penetration_ticks != 0
                    or self.constituents
                    or self.first_inside_close_at is not None
                    or self.first_outside_close_at is not None
                ):
                    raise ValueError(
                        "terminal interaction touch stage is inconsistent"
                    )
            elif (
                self.first_touch_at is not None
                or self.touch_bar_ids
                or self.first_penetration_at is not None
                or self.max_penetration_ticks != 0
                or self.constituents
                or self.first_inside_close_at is not None
                or self.first_outside_close_at is not None
            ):
                raise ValueError("terminal interaction pre-terminal stage is inconsistent")


@dataclass(frozen=True)
class StructureGeneration:
    structure_generation_id: str
    timeframe: Timeframe
    scope: StructureScope
    direction: Direction
    lifecycle: StructureGenerationLifecycle
    started_at: pd.Timestamp
    known_at: pd.Timestamp
    updated_at: pd.Timestamp
    origin_event_id: str
    origin_swing_id: str
    confirmation_event_id: str | None = None
    confirmed_at: pd.Timestamp | None = None
    protected_swing_id: str | None = None
    protected_swing_assignment_event_id: str | None = None
    bos_event_ids: tuple[str, ...] = ()
    mss_event_ids: tuple[str, ...] = ()
    terminated_at: pd.Timestamp | None = None
    termination_reason: str | None = None
    protected_acceptance_event_id: str | None = None
    source_event_ids: tuple[str, ...] = ()
    semantic_version: str = FOUNDATION_VERSION

    @property
    def generation_id(self) -> str:
        return self.structure_generation_id

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "scope", StructureScope(self.scope))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self, "lifecycle", StructureGenerationLifecycle(self.lifecycle)
        )
        for name in (
            "started_at",
            "known_at",
            "updated_at",
            "confirmed_at",
            "terminated_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, aware_timestamp(value, name=f"structure_generation.{name}")
                )
        for name in ("bos_event_ids", "mss_event_ids", "source_event_ids"):
            object.__setattr__(
                self, name, _unique_text(getattr(self, name), name=name)
            )
        if (self.protected_swing_id is None) != (
            self.protected_swing_assignment_event_id is None
        ):
            raise ValueError(
                "protected Swing assignment provenance must be complete"
            )
        if (
            self.protected_swing_assignment_event_id is not None
            and self.protected_swing_assignment_event_id
            not in self.source_event_ids
        ):
            raise ValueError(
                "protected Swing assignment event must be exact ancestry"
            )
        if (
            not self.structure_generation_id
            or not self.origin_event_id
            or not self.origin_swing_id
            or not self.started_at <= self.known_at <= self.updated_at
            or self.updated_at < self.started_at
            or self.semantic_version != FOUNDATION_VERSION
        ):
            raise ValueError("structure generation is invalid")
        if not self.source_event_ids or any(
            event_id not in self.source_event_ids
            for event_id in (
                self.origin_event_id,
                *self.bos_event_ids,
                *self.mss_event_ids,
                *(
                    ()
                    if self.confirmation_event_id is None
                    else (self.confirmation_event_id,)
                ),
                *(
                    ()
                    if self.protected_swing_assignment_event_id is None
                    else (self.protected_swing_assignment_event_id,)
                ),
                *(
                    ()
                    if self.protected_acceptance_event_id is None
                    else (self.protected_acceptance_event_id,)
                ),
            )
        ):
            raise ValueError("structure generation event ancestry is incomplete")
        if self.confirmed_at is not None and (
            self.confirmed_at < self.started_at
            or self.confirmed_at > self.updated_at
            or (
                self.scope is StructureScope.INTERNAL
                and (
                    self.confirmed_at <= self.started_at
                    or self.confirmation_event_id == self.origin_event_id
                )
            )
        ):
            raise ValueError("structure confirmation clock is inconsistent")
        if self.lifecycle is StructureGenerationLifecycle.FORMING:
            if self.confirmed_at is not None or self.confirmation_event_id is not None:
                raise ValueError("forming structure cannot already be confirmed")
        elif self.lifecycle is StructureGenerationLifecycle.CONFIRMED:
            if self.confirmed_at is None or self.confirmation_event_id is None:
                raise ValueError("confirmed structure lacks confirmation provenance")
        elif (self.confirmed_at is None) != (self.confirmation_event_id is None):
            raise ValueError("terminated structure has partial confirmation provenance")
        if self.lifecycle is StructureGenerationLifecycle.TERMINATED:
            if (
                self.terminated_at is None
                or self.terminated_at != self.updated_at
                or self.termination_reason not in _STRUCTURE_TERMINATION_REASONS
            ):
                raise ValueError("terminated structure lacks terminal provenance")
        elif self.terminated_at is not None or self.termination_reason is not None:
            raise ValueError("live structure has terminal metadata")
        if (
            self.protected_acceptance_event_id is not None
        ) != (self.termination_reason == "protected_break_accepted"):
            raise ValueError("protected structure termination evidence is inconsistent")


@dataclass(frozen=True)
class StructureTransition:
    structure_transition_id: str
    timeframe: Timeframe
    scope: StructureScope
    incumbent_structure_generation_id: str
    incumbent_direction: Direction
    challenger_direction: Direction
    lifecycle: StructureTransitionLifecycle
    started_at: pd.Timestamp
    updated_at: pd.Timestamp
    mss_event_ids: tuple[str, ...]
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None
    protected_acceptance_event_id: str | None = None
    opposite_structure_generation_id: str | None = None
    opposite_confirmation_event_id: str | None = None
    resumption_event_id: str | None = None
    source_event_ids: tuple[str, ...] = ()
    semantic_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "scope", StructureScope(self.scope))
        object.__setattr__(self, "incumbent_direction", Direction(self.incumbent_direction))
        object.__setattr__(self, "challenger_direction", Direction(self.challenger_direction))
        object.__setattr__(
            self, "lifecycle", StructureTransitionLifecycle(self.lifecycle)
        )
        for name in ("started_at", "updated_at", "terminal_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, aware_timestamp(value, name=f"structure_transition.{name}")
                )
        for name in ("mss_event_ids", "source_event_ids"):
            object.__setattr__(
                self, name, _unique_text(getattr(self, name), name=name)
            )
        if (
            not self.structure_transition_id
            or not self.incumbent_structure_generation_id
            or self.incumbent_direction is self.challenger_direction
            or not self.mss_event_ids
            or self.updated_at < self.started_at
            or self.semantic_version != FOUNDATION_VERSION
        ):
            raise ValueError("structure transition is invalid")
        if any(value not in self.source_event_ids for value in self.mss_event_ids):
            raise ValueError("structure-transition MSS ancestry is incomplete")
        terminal = self.lifecycle is not StructureTransitionLifecycle.STARTED
        if terminal != (self.terminal_at is not None and self.terminal_reason is not None):
            raise ValueError("structure-transition terminal state is inconsistent")
        if terminal and self.terminal_at != self.updated_at:
            raise ValueError("structure-transition terminal clock is inconsistent")
        opposite_pair = (
            self.opposite_structure_generation_id,
            self.opposite_confirmation_event_id,
        )
        if (opposite_pair[0] is None) != (opposite_pair[1] is None):
            raise ValueError(
                "structure-transition opposite confirmation is incomplete"
            )
        cited_event_ids = tuple(
            value
            for value in (
                self.protected_acceptance_event_id,
                self.opposite_confirmation_event_id,
                self.resumption_event_id,
            )
            if value is not None
        )
        if any(value not in self.source_event_ids for value in cited_event_ids):
            raise ValueError(
                "structure-transition evidence must be exact ancestry"
            )
        if self.lifecycle is StructureTransitionLifecycle.STARTED:
            if any(
                value is not None
                for value in (*opposite_pair, self.resumption_event_id)
            ):
                raise ValueError("started transition contains terminal evidence")
        elif self.lifecycle is StructureTransitionLifecycle.CONFIRMED:
            if (
                self.protected_acceptance_event_id is None
                or any(value is None for value in opposite_pair)
                or self.resumption_event_id is not None
                or self.terminal_reason
                != "protected_acceptance_plus_opposite_confirmation"
            ):
                raise ValueError(
                    "confirmed transition lacks exact confirmation evidence"
                )
        elif self.lifecycle is StructureTransitionLifecycle.FAILED:
            if (
                self.resumption_event_id is None
                or self.protected_acceptance_event_id is not None
                or self.terminal_reason != "original_direction_resumed"
                or any(value is not None for value in opposite_pair)
            ):
                raise ValueError(
                    "failed transition lacks exclusive resumption evidence"
                )
        elif self.lifecycle is StructureTransitionLifecycle.CENSORED:
            if (
                any(value is not None for value in opposite_pair)
                or self.resumption_event_id is not None
                or self.terminal_reason
                not in _STRUCTURE_TRANSITION_CENSOR_REASONS
            ):
                raise ValueError(
                    "censored transition contains invented terminal evidence"
                )


@dataclass(frozen=True)
class RelationGeneration:
    relation_generation_id: str
    source_relation_id: str
    parent_tf: Timeframe
    child_tf: Timeframe
    parent_structure_generation_id: str
    child_structure_generation_id: str
    role: str
    lifecycle: GenerationLifecycle
    entered_at: pd.Timestamp
    known_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    observation_count: int
    latest_relation_digest: str
    terminated_at: pd.Timestamp | None = None
    termination_reason: str | None = None
    source_event_ids: tuple[str, ...] = ()
    semantic_version: str = FOUNDATION_VERSION

    @property
    def generation_id(self) -> str:
        return self.relation_generation_id

    @property
    def started_at(self) -> pd.Timestamp:
        return self.entered_at

    @property
    def updated_at(self) -> pd.Timestamp:
        return self.last_updated_at

    @property
    def signature(self) -> tuple[str, str, str]:
        return (
            self.parent_structure_generation_id,
            self.child_structure_generation_id,
            self.role,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_tf", Timeframe(self.parent_tf))
        object.__setattr__(self, "child_tf", Timeframe(self.child_tf))
        object.__setattr__(self, "lifecycle", GenerationLifecycle(self.lifecycle))
        for name in ("entered_at", "known_at", "last_updated_at", "terminated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, aware_timestamp(value, name=f"relation_generation.{name}")
                )
        object.__setattr__(
            self, "source_event_ids", _unique_text(self.source_event_ids, name="source_event_ids")
        )
        if (
            not self.relation_generation_id
            or not self.source_relation_id
            or _TIMEFRAME_INTERVAL[self.parent_tf]
            <= _TIMEFRAME_INTERVAL[self.child_tf]
            or not self.parent_structure_generation_id
            or not self.child_structure_generation_id
            or self.role not in _RELATION_ROLES
            or type(self.observation_count) is not int
            or self.observation_count < 1
            or not self.latest_relation_digest
            or not self.entered_at <= self.known_at <= self.last_updated_at
            or self.last_updated_at < self.entered_at
            or self.semantic_version != FOUNDATION_VERSION
        ):
            raise ValueError("relation generation is invalid")
        terminal = self.lifecycle is GenerationLifecycle.TERMINATED
        if terminal != (self.terminated_at is not None and self.termination_reason is not None):
            raise ValueError("relation terminal state is inconsistent")
        if terminal and (
            self.terminated_at != self.last_updated_at
            or self.termination_reason not in _RELATION_TERMINATION_REASONS
        ):
            raise ValueError("relation terminal clock or reason is invalid")


@dataclass(frozen=True)
class DeliveryPhaseGeneration:
    delivery_generation_id: str
    timeframe: Timeframe
    phase: str
    parent_structure_generation_id: str
    lifecycle: GenerationLifecycle
    entered_at: pd.Timestamp
    known_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    entered_real_bar_ordinal: int
    duration_bars: int
    duration_seconds: float
    origin_event_id: str
    observation_count: int
    origin_price_ticks: int
    current_price_ticks: int
    max_extension_ticks: float
    max_retracement_ticks: float
    terminated_at: pd.Timestamp | None = None
    termination_reason: str | None = None
    next_phase: str | None = None
    source_event_ids: tuple[str, ...] = ()
    semantic_version: str = FOUNDATION_VERSION

    @property
    def generation_id(self) -> str:
        return self.delivery_generation_id

    @property
    def started_at(self) -> pd.Timestamp:
        return self.entered_at

    @property
    def updated_at(self) -> pd.Timestamp:
        return self.last_updated_at

    @property
    def signature(self) -> tuple[str, str]:
        return self.parent_structure_generation_id, self.phase

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "lifecycle", GenerationLifecycle(self.lifecycle))
        for name in ("entered_at", "known_at", "last_updated_at", "terminated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, aware_timestamp(value, name=f"delivery_generation.{name}")
                )
        object.__setattr__(
            self, "source_event_ids", _unique_text(self.source_event_ids, name="source_event_ids")
        )
        if (
            not self.delivery_generation_id
            or self.phase not in _DELIVERY_PHASES
            or not self.parent_structure_generation_id
            or type(self.entered_real_bar_ordinal) is not int
            or self.entered_real_bar_ordinal < 0
            or type(self.duration_bars) is not int
            or self.duration_bars < 0
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds < 0.0
            or type(self.observation_count) is not int
            or self.observation_count < 1
            or type(self.origin_price_ticks) is not int
            or self.origin_price_ticks < 1
            or type(self.current_price_ticks) is not int
            or self.current_price_ticks < 1
            or not math.isfinite(float(self.max_extension_ticks))
            or self.max_extension_ticks < 0.0
            or not math.isfinite(float(self.max_retracement_ticks))
            or self.max_retracement_ticks < 0.0
            or not self.entered_at <= self.known_at <= self.last_updated_at
            or self.last_updated_at < self.entered_at
            or self.semantic_version != FOUNDATION_VERSION
        ):
            raise ValueError("delivery-phase generation is invalid")
        if (
            not self.origin_event_id
            or self.origin_event_id not in self.source_event_ids
            or self.duration_seconds
            != (self.last_updated_at - self.entered_at).total_seconds()
        ):
            raise ValueError("delivery origin ancestry or duration is inconsistent")
        terminal = self.lifecycle is GenerationLifecycle.TERMINATED
        if terminal != (self.terminated_at is not None and self.termination_reason is not None):
            raise ValueError("delivery terminal state is inconsistent")
        if terminal and (
            self.terminated_at != self.last_updated_at
            or self.termination_reason not in _DELIVERY_TERMINATION_REASONS
        ):
            raise ValueError("delivery terminal clock or reason is invalid")
        if self.lifecycle is GenerationLifecycle.ACTIVE and self.next_phase is not None:
            raise ValueError("active delivery cannot name a next phase")
        if self.termination_reason == "phase_changed":
            if self.next_phase not in _DELIVERY_PHASES or self.next_phase == self.phase:
                raise ValueError("delivery phase change lacks a distinct next phase")
        elif self.next_phase is not None:
            raise ValueError("non-phase delivery termination has a next phase")


@dataclass(frozen=True)
class BoundaryAttackFact:
    boundary_attack_id: str
    bos_generation_id: str
    timeframe: Timeframe
    direction: Direction
    target_swing_event_id: str
    bar_event_id: str
    boundary_ticks: int
    extreme_ticks: int
    close_ticks: int
    penetration_ticks: int
    attempt_ordinal: int
    known_at: pd.Timestamp
    source_event_ids: tuple[str, ...]
    semantic_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self, "known_at", aware_timestamp(self.known_at, name="boundary_attack.known_at")
        )
        object.__setattr__(
            self, "source_event_ids", _unique_text(self.source_event_ids, name="source_event_ids")
        )
        exact_ticks = (
            self.boundary_ticks,
            self.extreme_ticks,
            self.close_ticks,
            self.penetration_ticks,
            self.attempt_ordinal,
        )
        if any(type(value) is not int for value in exact_ticks):
            raise ValueError("boundary-attack tick geometry must use exact integers")
        directional_geometry = (
            self.extreme_ticks > self.boundary_ticks
            and self.penetration_ticks
            == self.extreme_ticks - self.boundary_ticks
            and self.close_ticks <= self.boundary_ticks
            if self.direction is Direction.LONG
            else self.extreme_ticks < self.boundary_ticks
            and self.penetration_ticks
            == self.boundary_ticks - self.extreme_ticks
            and self.close_ticks >= self.boundary_ticks
        )
        if (
            not self.boundary_attack_id
            or not self.bos_generation_id
            or not self.target_swing_event_id
            or not self.bar_event_id
            or min(
                self.boundary_ticks,
                self.extreme_ticks,
                self.close_ticks,
            )
            < 1
            or self.penetration_ticks < 1
            or self.attempt_ordinal < 1
            or not directional_geometry
            or self.target_swing_event_id not in self.source_event_ids
            or self.bar_event_id not in self.source_event_ids
            or self.semantic_version != FOUNDATION_VERSION
        ):
            raise ValueError("boundary-attack fact is invalid")


@dataclass(frozen=True)
class SemanticLifecycleState:
    levels: tuple[LiquidityLevelState, ...] = ()
    interactions: tuple[LiquidityInteractionGeneration, ...] = ()
    structure_generations: tuple[StructureGeneration, ...] = ()
    structure_transitions: tuple[StructureTransition, ...] = ()
    relation_generations: tuple[RelationGeneration, ...] = ()
    delivery_generations: tuple[DeliveryPhaseGeneration, ...] = ()
    boundary_attacks: tuple[BoundaryAttackFact, ...] = ()
    registered_bar_clocks: tuple[RegisteredBarClock, ...] = ()
    real_bar_clocks: tuple[RealBarClock, ...] = ()
    asof: pd.Timestamp | None = None
    epoch: int = 0
    semantic_version: str = FOUNDATION_VERSION
    schema_version: int = LIFECYCLE_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "levels",
            "interactions",
            "structure_generations",
            "structure_transitions",
            "relation_generations",
            "delivery_generations",
            "boundary_attacks",
            "registered_bar_clocks",
            "real_bar_clocks",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.asof is not None:
            object.__setattr__(
                self, "asof", aware_timestamp(self.asof, name="lifecycle_state.asof")
            )
        if (
            self.epoch < 0
            or self.semantic_version != FOUNDATION_VERSION
            or self.schema_version != LIFECYCLE_STATE_SCHEMA_VERSION
        ):
            raise ValueError("semantic lifecycle state version or epoch is invalid")
        identity_specs = (
            (self.levels, "level_id"),
            (self.interactions, "generation_id"),
            (self.structure_generations, "structure_generation_id"),
            (self.structure_transitions, "structure_transition_id"),
            (self.relation_generations, "relation_generation_id"),
            (self.delivery_generations, "delivery_generation_id"),
            (self.boundary_attacks, "boundary_attack_id"),
            (self.registered_bar_clocks, "timeframe"),
            (self.real_bar_clocks, "timeframe"),
        )
        for items, attribute in identity_specs:
            identities = tuple(getattr(item, attribute) for item in items)
            if len(identities) != len(set(identities)):
                raise ValueError(f"semantic lifecycle state repeats {attribute}")
        active_structure_keys = [
            (item.timeframe, item.scope)
            for item in self.structure_generations
            if item.lifecycle is not StructureGenerationLifecycle.TERMINATED
        ]
        if len(active_structure_keys) != len(set(active_structure_keys)):
            raise ValueError("multiple live structure generations share one scope")
        active_relation_edges = [
            (item.parent_tf, item.child_tf)
            for item in self.relation_generations
            if item.lifecycle is GenerationLifecycle.ACTIVE
        ]
        if len(active_relation_edges) != len(set(active_relation_edges)):
            raise ValueError("multiple live relation generations share one edge")
        active_delivery_tfs = [
            item.timeframe
            for item in self.delivery_generations
            if item.lifecycle is GenerationLifecycle.ACTIVE
        ]
        if len(active_delivery_tfs) != len(set(active_delivery_tfs)):
            raise ValueError("multiple live delivery generations share one timeframe")
        registered_by_timeframe = {
            item.timeframe: item for item in self.registered_bar_clocks
        }
        if any(
            real.timeframe not in registered_by_timeframe
            or real.count > registered_by_timeframe[real.timeframe].count
            or real.last_completed_at
            > registered_by_timeframe[real.timeframe].last_completed_at
            or (
                real.last_completed_at
                == registered_by_timeframe[real.timeframe].last_completed_at
                and real.last_bar_event_id
                != registered_by_timeframe[real.timeframe].last_bar_event_id
            )
            or (
                real.count == registered_by_timeframe[real.timeframe].count
                and (
                    real.last_completed_at
                    != registered_by_timeframe[real.timeframe].last_completed_at
                    or real.last_bar_event_id
                    != registered_by_timeframe[real.timeframe].last_bar_event_id
                )
            )
            for real in self.real_bar_clocks
        ):
            raise ValueError(
                "real BAR clock conflicts with its registered transport clock"
            )

    def __getstate__(self) -> Mapping[str, Any]:
        return {
            "levels": self.levels,
            "interactions": self.interactions,
            "structure_generations": self.structure_generations,
            "structure_transitions": self.structure_transitions,
            "relation_generations": self.relation_generations,
            "delivery_generations": self.delivery_generations,
            "boundary_attacks": self.boundary_attacks,
            "registered_bar_clocks": self.registered_bar_clocks,
            "real_bar_clocks": self.real_bar_clocks,
            "asof": self.asof,
            "epoch": self.epoch,
            "semantic_version": self.semantic_version,
            "schema_version": self.schema_version,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {
            "levels",
            "interactions",
            "structure_generations",
            "structure_transitions",
            "relation_generations",
            "delivery_generations",
            "boundary_attacks",
            "registered_bar_clocks",
            "real_bar_clocks",
            "asof",
            "epoch",
            "semantic_version",
            "schema_version",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("schema_version") != LIFECYCLE_STATE_SCHEMA_VERSION
        ):
            raise ValueError("semantic lifecycle state pickle schema changed")
        for name in expected:
            object.__setattr__(self, name, state[name])
        self.__post_init__()

    def level(self, level_id: str) -> LiquidityLevelState:
        return _find(self.levels, "level_id", level_id)

    def interaction(self, generation_id: str) -> LiquidityInteractionGeneration:
        return _find(self.interactions, "generation_id", generation_id)

    def structure(self, generation_id: str) -> StructureGeneration:
        return _find(
            self.structure_generations, "structure_generation_id", generation_id
        )

    def transition(self, transition_id: str) -> StructureTransition:
        return _find(
            self.structure_transitions, "structure_transition_id", transition_id
        )


@dataclass(frozen=True)
class LifecycleCheckpoint:
    state: SemanticLifecycleState
    state_digest: str
    schema_version: int = LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
    semantic_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        if (
            "schema_version" not in vars(self)
            or not isinstance(self.state, SemanticLifecycleState)
            or "registered_bar_clocks" not in vars(self.state)
            or "schema_version" not in vars(self.state)
            or "applied_transition_fingerprints" in vars(self.state)
            or "transition_count" in vars(self.state)
            or self.schema_version != LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
            or self.semantic_version != FOUNDATION_VERSION
            or self.state.semantic_version != self.semantic_version
            or self.state_digest != content_hash(self.state)
        ):
            raise ValueError("semantic lifecycle checkpoint is invalid")

    def __getstate__(self) -> Mapping[str, Any]:
        return {
            "state": self.state,
            "state_digest": self.state_digest,
            "schema_version": self.schema_version,
            "semantic_version": self.semantic_version,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {
            "state",
            "state_digest",
            "schema_version",
            "semantic_version",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("semantic lifecycle checkpoint schema changed")
        for name in expected:
            object.__setattr__(self, name, state[name])
        self.__post_init__()


def _find(items: Sequence[Any], attribute: str, identity: object) -> Any:
    matches = tuple(item for item in items if getattr(item, attribute) == identity)
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous {attribute}: {identity!r}")
    return matches[0]


def _replace_item(
    items: Sequence[Any], attribute: str, value: Any
) -> tuple[Any, ...]:
    identity = getattr(value, attribute)
    found = False
    output: list[Any] = []
    for item in items:
        if getattr(item, attribute) == identity:
            if found:
                raise ValueError(f"duplicate {attribute}: {identity!r}")
            output.append(value)
            found = True
        else:
            output.append(item)
    if not found:
        raise ValueError(f"unknown {attribute}: {identity!r}")
    return tuple(output)


def _append_item(
    items: Sequence[Any], attribute: str, value: Any
) -> tuple[Any, ...]:
    identity = getattr(value, attribute)
    if any(getattr(item, attribute) == identity for item in items):
        raise ValueError(f"duplicate {attribute}: {identity!r}")
    return (*tuple(items), value)


def _merged_ids(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _bar_count(state: SemanticLifecycleState, timeframe: Timeframe) -> int:
    return next(
        (
            item.count
            for item in state.real_bar_clocks
            if item.timeframe is timeframe
        ),
        0,
    )


def _active_interaction(
    state: SemanticLifecycleState, level: LiquidityLevelState
) -> LiquidityInteractionGeneration:
    if level.active_generation_id is None:
        raise ValueError("liquidity level has no armed interaction generation")
    interaction = state.interaction(level.active_generation_id)
    if interaction.lifecycle is LiquidityInteractionLifecycle.TERMINAL:
        raise ValueError("liquidity level points at a terminal interaction")
    return interaction


def _require_source(
    transition: NormalizedLifecycleTransition,
    identity: str,
    *,
    role: str,
) -> None:
    if identity not in transition.source_event_ids:
        raise ValueError(f"{role} must be exact normalized source ancestry")


class SemanticLifecycleReducer:
    """Stateless reducer for foundation lifecycle facts.

    Every method returns a new :class:`SemanticLifecycleState`.  Validation is
    completed before that value is returned, so a rejected fact cannot mutate
    the caller's prior state or a terminal generation.
    """

    @staticmethod
    def initial_state() -> SemanticLifecycleState:
        return SemanticLifecycleState()

    @classmethod
    def reduce(
        cls,
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        return cls.reduce_hot(state, transition)

    @classmethod
    def reduce_hot(
        cls,
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        """Project one cursor-ordered fact without embedding replay history."""

        if not isinstance(state, SemanticLifecycleState):
            raise TypeError("semantic lifecycle reducer requires its state DTO")
        if not isinstance(transition, NormalizedLifecycleTransition):
            raise TypeError("semantic lifecycle reducer requires normalized input")
        if state.semantic_version != FOUNDATION_VERSION:
            raise ValueError("semantic lifecycle state version mismatch")
        if state.asof is not None and transition.known_at < state.asof:
            raise ValueError("normalized lifecycle replay moved backwards in time")
        handler = {
            NormalizedTransitionKind.REAL_BAR_COMPLETED: cls._real_bar_completed,
            NormalizedTransitionKind.RESET: cls._reset,
            NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED: cls._level_created,
            NormalizedTransitionKind.LIQUIDITY_TOUCHED: cls._liquidity_touched,
            NormalizedTransitionKind.LIQUIDITY_PENETRATED: cls._liquidity_penetrated,
            NormalizedTransitionKind.LIQUIDITY_SWEEP_TERMINAL: cls._liquidity_terminal,
            NormalizedTransitionKind.LIQUIDITY_ACCEPTANCE_TERMINAL: cls._liquidity_terminal,
            NormalizedTransitionKind.LIQUIDITY_UNRESOLVED_TERMINAL: cls._liquidity_unresolved,
            NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMABLE: cls._level_rearmable,
            NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMED: cls._level_rearmed,
            NormalizedTransitionKind.LIQUIDITY_LEVEL_RETIRED: cls._level_retired,
            NormalizedTransitionKind.LIQUIDITY_LEVEL_ARCHIVED: cls._level_archived,
            NormalizedTransitionKind.STRUCTURE_GENERATION_STARTED: cls._structure_started,
            NormalizedTransitionKind.STRUCTURE_GENERATION_CONFIRMED: cls._structure_confirmed,
            NormalizedTransitionKind.STRUCTURE_GENERATION_EVIDENCE: cls._structure_evidence,
            NormalizedTransitionKind.STRUCTURE_GENERATION_TERMINATED: cls._structure_terminated,
            NormalizedTransitionKind.MSS_TRANSITION_STARTED: cls._transition_started,
            NormalizedTransitionKind.STRUCTURE_TRANSITION_EVIDENCE: cls._transition_evidence,
            NormalizedTransitionKind.STRUCTURE_TRANSITION_CONFIRMED: cls._transition_confirmed,
            NormalizedTransitionKind.STRUCTURE_DIRECTION_RESUMED: cls._transition_failed,
            NormalizedTransitionKind.RELATION_OBSERVED: cls._relation_observed,
            NormalizedTransitionKind.RELATION_TERMINATED: cls._relation_terminated,
            NormalizedTransitionKind.DELIVERY_PHASE_OBSERVED: cls._delivery_observed,
            NormalizedTransitionKind.DELIVERY_PHASE_TERMINATED: cls._delivery_terminated,
            NormalizedTransitionKind.BOUNDARY_ATTACK_OBSERVED: cls._boundary_attack,
        }[transition.kind]
        projected = handler(state, transition)
        return replace(projected, asof=transition.known_at)

    @classmethod
    def replay(
        cls,
        transitions: Iterable[NormalizedLifecycleTransition],
        *,
        initial_state: SemanticLifecycleState | None = None,
    ) -> SemanticLifecycleState:
        facts = tuple(transitions)
        cls._preflight_batch(facts)
        state = initial_state or cls.initial_state()
        for transition in facts:
            state = cls.reduce(state, transition)
        return state

    @staticmethod
    def _preflight_batch(
        transitions: Sequence[NormalizedLifecycleTransition],
    ) -> None:
        fact_ids: set[str] = set()
        delivery_signatures: dict[
            tuple[Timeframe, pd.Timestamp], tuple[str, str]
        ] = {}
        for transition in transitions:
            if not isinstance(transition, NormalizedLifecycleTransition):
                raise TypeError("lifecycle replay accepts only normalized facts")
            if transition.fact_id in fact_ids:
                raise ValueError("batch repeats normalized fact identity")
            fact_ids.add(transition.fact_id)
            if transition.kind is not NormalizedTransitionKind.DELIVERY_PHASE_OBSERVED:
                continue
            signature = (
                _required_text(
                    transition.payload, "parent_structure_generation_id"
                ),
                _required_text(transition.payload, "phase"),
            )
            key = (Timeframe(transition.timeframe), transition.known_at)
            prior_signature = delivery_signatures.setdefault(key, signature)
            if prior_signature != signature:
                raise ValueError(
                    "same-clock delivery observations must be coalesced before transition"
                )

    @classmethod
    def reduce_market_event(
        cls,
        state: SemanticLifecycleState,
        event: MarketEvent,
        *,
        kind: NormalizedTransitionKind,
        payload: Mapping[str, Any] | None = None,
    ) -> SemanticLifecycleState:
        return cls.reduce(
            state,
            NormalizedLifecycleTransition.from_market_event(
                event, kind=kind, payload=payload
            ),
        )

    @classmethod
    def observe_relation(
        cls,
        state: SemanticLifecycleState,
        relation: "RelationState",
        *,
        parent_structure_generation_id: str,
        child_structure_generation_id: str,
        source_event_ids: Sequence[str],
    ) -> SemanticLifecycleState:
        return cls.reduce(
            state,
            NormalizedLifecycleTransition.from_relation_state(
                relation,
                parent_structure_generation_id=parent_structure_generation_id,
                child_structure_generation_id=child_structure_generation_id,
                source_event_ids=source_event_ids,
            ),
        )

    @classmethod
    def observe_delivery_phase(
        cls,
        state: SemanticLifecycleState,
        phase: "DeliveryPhase",
        *,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
        parent_structure_generation_id: str,
        origin_event_id: str,
        source_event_ids: Sequence[str],
        current_price_ticks: int,
        extension_ticks: float | None = None,
        retracement_ticks: float | None = None,
    ) -> SemanticLifecycleState:
        return cls.reduce(
            state,
            NormalizedLifecycleTransition.from_delivery_phase(
                phase,
                timeframe=timeframe,
                known_at=known_at,
                parent_structure_generation_id=parent_structure_generation_id,
                origin_event_id=origin_event_id,
                source_event_ids=source_event_ids,
                current_price_ticks=current_price_ticks,
                extension_ticks=extension_ticks,
                retracement_ticks=retracement_ticks,
            ),
        )

    @staticmethod
    def checkpoint(state: SemanticLifecycleState) -> LifecycleCheckpoint:
        return LifecycleCheckpoint(state=state, state_digest=content_hash(state))

    @staticmethod
    def restore(checkpoint: LifecycleCheckpoint) -> SemanticLifecycleState:
        if not isinstance(checkpoint, LifecycleCheckpoint):
            raise TypeError("semantic lifecycle restore requires a checkpoint DTO")
        if (
            "schema_version" not in vars(checkpoint)
            or not isinstance(checkpoint.state, SemanticLifecycleState)
            or "registered_bar_clocks" not in vars(checkpoint.state)
            or "schema_version" not in vars(checkpoint.state)
            or "applied_transition_fingerprints" in vars(checkpoint.state)
            or "transition_count" in vars(checkpoint.state)
            or checkpoint.schema_version != LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
            or checkpoint.state_digest != content_hash(checkpoint.state)
        ):
            raise ValueError("semantic lifecycle checkpoint digest changed")
        return checkpoint.state

    @staticmethod
    def _real_bar_completed(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        real_completed = transition.payload.get("real_completed")
        clock_only = transition.payload.get("clock_only")
        if real_completed is True and "clock_only" not in transition.payload:
            # Preserve the frozen real-transition payload/fingerprint.  The
            # external BAR event still requires both exact provenance flags.
            clock_only = False
        elif (
            type(real_completed) is not bool
            or type(clock_only) is not bool
            or clock_only is not (not real_completed)
        ):
            raise ValueError(
                "completed-BAR transition requires exact complementary "
                "real_completed/clock_only flags"
            )
        timeframe = Timeframe(transition.timeframe)
        bar_event_id = (
            _optional_text(transition.payload, "bar_event_id")
            or transition.source_event_ids[0]
        )
        _require_source(transition, bar_event_id, role="registered BAR")
        timeframe_minutes = int(
            _TIMEFRAME_INTERVAL[timeframe] / pd.Timedelta(1, unit="min")
        )
        anchor_minute = _TIMEFRAME_ANCHOR_MINUTE[timeframe]
        prior_registered = next(
            (
                item
                for item in state.registered_bar_clocks
                if item.timeframe is timeframe
            ),
            None,
        )
        if prior_registered is not None:
            expected_completion = next_registered_native_completion(
                prior_registered.last_completed_at,
                timeframe_minutes=timeframe_minutes,
                anchor_minute=anchor_minute,
            )
            if transition.known_at != expected_completion:
                raise ValueError(
                    "registered completed BAR clocks must be exactly contiguous "
                    "on the registered native clock within an epoch"
                )
        updated_registered = RegisteredBarClock(
            timeframe=timeframe,
            count=(
                1 if prior_registered is None else prior_registered.count + 1
            ),
            last_completed_at=transition.known_at,
            last_bar_event_id=bar_event_id,
        )
        registered_clocks = (
            _append_item(
                state.registered_bar_clocks,
                "timeframe",
                updated_registered,
            )
            if prior_registered is None
            else _replace_item(
                state.registered_bar_clocks,
                "timeframe",
                updated_registered,
            )
        )
        if not real_completed:
            return replace(state, registered_bar_clocks=registered_clocks)

        prior_real = next(
            (item for item in state.real_bar_clocks if item.timeframe is timeframe),
            None,
        )
        updated_real = RealBarClock(
            timeframe=timeframe,
            count=1 if prior_real is None else prior_real.count + 1,
            last_completed_at=transition.known_at,
            last_bar_event_id=bar_event_id,
        )
        real_clocks = (
            _append_item(state.real_bar_clocks, "timeframe", updated_real)
            if prior_real is None
            else _replace_item(
                state.real_bar_clocks,
                "timeframe",
                updated_real,
            )
        )
        return replace(
            state,
            registered_bar_clocks=registered_clocks,
            real_bar_clocks=real_clocks,
        )

    @staticmethod
    def _reset(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        reason = _required_text(transition.payload, "reason")
        if reason not in _RESET_REASONS:
            raise ValueError("reset reason is not preregistered")
        ordinal_by_tf = {
            item.timeframe: item.count for item in state.real_bar_clocks
        }
        interactions = tuple(
            item
            if item.lifecycle is LiquidityInteractionLifecycle.TERMINAL
            else replace(
                item,
                lifecycle=LiquidityInteractionLifecycle.TERMINAL,
                updated_at=transition.known_at,
                terminal_event_id=transition.fact_id,
                terminal_state=LiquidityInteractionTerminal.CENSORED,
                terminal_at=transition.known_at,
                terminal_reason=reason,
                terminal_real_bar_ordinal=ordinal_by_tf.get(
                    item.interaction_timeframe, 0
                ),
                source_event_ids=_merged_ids(
                    item.source_event_ids, transition.source_event_ids
                ),
            )
            for item in state.interactions
        )
        levels = tuple(
            item
            if item.lifecycle is LiquidityLevelLifecycle.ARCHIVED
            else replace(
                item,
                lifecycle=LiquidityLevelLifecycle.ARCHIVED,
                updated_at=transition.known_at,
                active_generation_id=None,
                last_terminal_generation_id=(
                    item.active_generation_id
                    or item.last_terminal_generation_id
                ),
                rearmable_at=None,
                rearmable_from_generation_id=None,
                rearmable_fact_id=None,
                rearm_departure_bar_event_id=None,
                rearm_departure_ticks=None,
                rearmed_from_generation_id=None,
                retired_at=item.retired_at or transition.known_at,
                retirement_reason=item.retirement_reason or reason,
                archived_at=transition.known_at,
                archive_reason=reason,
                source_event_ids=_merged_ids(
                    item.source_event_ids, transition.source_event_ids
                ),
            )
            for item in state.levels
        )
        structure_reason = {
            "contract_reset": "contract_rollover",
            "data_reset": "data_reset",
            "semantic_reset": "semantic_reset",
        }[reason]
        structures = tuple(
            item
            if item.lifecycle is StructureGenerationLifecycle.TERMINATED
            else replace(
                item,
                lifecycle=StructureGenerationLifecycle.TERMINATED,
                updated_at=transition.known_at,
                terminated_at=transition.known_at,
                termination_reason=structure_reason,
                source_event_ids=_merged_ids(
                    item.source_event_ids, transition.source_event_ids
                ),
            )
            for item in state.structure_generations
        )
        structure_transitions = tuple(
            item
            if item.lifecycle is not StructureTransitionLifecycle.STARTED
            else replace(
                item,
                lifecycle=StructureTransitionLifecycle.CENSORED,
                updated_at=transition.known_at,
                terminal_at=transition.known_at,
                terminal_reason=reason,
                source_event_ids=_merged_ids(
                    item.source_event_ids, transition.source_event_ids
                ),
            )
            for item in state.structure_transitions
        )
        relations = tuple(
            item
            if item.lifecycle is GenerationLifecycle.TERMINATED
            else replace(
                item,
                lifecycle=GenerationLifecycle.TERMINATED,
                last_updated_at=transition.known_at,
                terminated_at=transition.known_at,
                termination_reason=reason,
                source_event_ids=_merged_ids(
                    item.source_event_ids, transition.source_event_ids
                ),
            )
            for item in state.relation_generations
        )
        deliveries = tuple(
            item
            if item.lifecycle is GenerationLifecycle.TERMINATED
            else replace(
                item,
                lifecycle=GenerationLifecycle.TERMINATED,
                last_updated_at=transition.known_at,
                duration_bars=max(
                    0,
                    ordinal_by_tf.get(item.timeframe, 0)
                    - item.entered_real_bar_ordinal,
                ),
                duration_seconds=(transition.known_at - item.entered_at).total_seconds(),
                terminated_at=transition.known_at,
                termination_reason=reason,
                source_event_ids=_merged_ids(
                    item.source_event_ids, transition.source_event_ids
                ),
            )
            for item in state.delivery_generations
        )
        return replace(
            state,
            levels=levels,
            interactions=interactions,
            structure_generations=structures,
            structure_transitions=structure_transitions,
            relation_generations=relations,
            delivery_generations=deliveries,
            registered_bar_clocks=(),
            real_bar_clocks=(),
            epoch=state.epoch + 1,
        )

    @staticmethod
    def _level_created(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        source_timeframe = Timeframe(transition.timeframe)
        interaction_timeframe = Timeframe(
            _required_text(transition.payload, "interaction_timeframe")
        )
        source_kind = _required_text(transition.payload, "source_kind")
        source_identity = _required_text(transition.payload, "source_identity")
        side = _required_text(transition.payload, "side")
        if side not in {"above", "below"}:
            raise ValueError("liquidity level side must be above or below")
        price_ticks = _required_int(transition.payload, "price_ticks", minimum=1)
        lower_bound_ticks = transition.payload.get(
            "lower_bound_ticks", price_ticks
        )
        upper_bound_ticks = transition.payload.get(
            "upper_bound_ticks", price_ticks
        )
        if (
            type(lower_bound_ticks) is not int
            or type(upper_bound_ticks) is not int
            or not 0 < lower_bound_ticks <= price_ticks <= upper_bound_ticks
        ):
            raise ValueError("liquidity level frozen bounds are invalid")
        tick_size_value = transition.payload.get("tick_size")
        if (
            isinstance(tick_size_value, bool)
            or not isinstance(tick_size_value, (int, float))
            or not math.isfinite(float(tick_size_value))
            or float(tick_size_value) <= 0.0
        ):
            raise ValueError("liquidity level requires a positive tick_size")
        tick_size = float(tick_size_value)
        price_anchor_rule = (
            _optional_text(transition.payload, "price_anchor_rule")
            or "exact_event_price"
        )
        if price_anchor_rule not in {
            "exact_event_price",
            "near_side_tradable_zone_boundary_for_nontradable_midpoint",
        }:
            raise ValueError("liquidity price anchor rule is not registered")
        level_id = canonical_semantic_id(
            "liquidity-level",
            source_timeframe.value,
            source_kind,
            source_identity,
            price_ticks,
            lower_bound_ticks,
            upper_bound_ticks,
            price_anchor_rule,
            transition.known_at,
        )
        asserted_level_id = _optional_text(
            transition.payload, "foundation_level_id"
        )
        if asserted_level_id is not None and asserted_level_id != level_id:
            raise ValueError("asserted liquidity level id is not canonical")
        same_source = tuple(
            item
            for item in state.levels
            if (
                item.source_timeframe is source_timeframe
                and item.source_kind == source_kind
                and item.source_identity == source_identity
            )
        )
        if any(item.retirement_reason == "acceptance" for item in same_source):
            raise ValueError(
                "accepted liquidity cannot be re-created from the same source identity"
            )
        supersedes_level_id = _optional_text(
            transition.payload, "supersedes_level_id"
        )
        if same_source and supersedes_level_id is None:
            raise ValueError("liquidity source identity already has a level")

        levels = state.levels
        interactions = state.interactions
        if supersedes_level_id is not None:
            prior = state.level(supersedes_level_id)
            if prior.lifecycle in {
                LiquidityLevelLifecycle.RETIRED,
                LiquidityLevelLifecycle.ARCHIVED,
            }:
                raise ValueError("supersession cannot rewrite a retired level")
            if transition.known_at <= prior.updated_at:
                raise ValueError("liquidity supersession must be strictly later")
            if prior.active_generation_id is not None:
                live = _active_interaction(state, prior)
                expired = replace(
                    live,
                    lifecycle=LiquidityInteractionLifecycle.TERMINAL,
                    updated_at=transition.known_at,
                    terminal_event_id=transition.fact_id,
                    terminal_state=LiquidityInteractionTerminal.EXPIRED,
                    terminal_at=transition.known_at,
                    terminal_reason="supersession",
                    terminal_real_bar_ordinal=_bar_count(
                        state, live.interaction_timeframe
                    ),
                    source_event_ids=_merged_ids(
                        live.source_event_ids, transition.source_event_ids
                    ),
                )
                interactions = _replace_item(
                    interactions, "generation_id", expired
                )
            retired = replace(
                prior,
                lifecycle=LiquidityLevelLifecycle.RETIRED,
                updated_at=transition.known_at,
                active_generation_id=None,
                last_terminal_generation_id=(
                    prior.active_generation_id
                    or prior.last_terminal_generation_id
                ),
                rearmable_at=None,
                rearmable_from_generation_id=None,
                rearmable_fact_id=None,
                rearm_departure_bar_event_id=None,
                rearm_departure_ticks=None,
                rearmed_from_generation_id=None,
                retired_at=transition.known_at,
                retirement_reason="supersession",
                superseded_by_level_id=level_id,
                source_event_ids=_merged_ids(
                    prior.source_event_ids, transition.source_event_ids
                ),
            )
            levels = _replace_item(levels, "level_id", retired)

        generation_id = canonical_semantic_id(
            "liquidity-interaction",
            level_id,
            1,
            transition.known_at,
            None,
        )
        interaction = LiquidityInteractionGeneration(
            generation_id=generation_id,
            level_id=level_id,
            source_timeframe=source_timeframe,
            interaction_timeframe=interaction_timeframe,
            level_side=side,
            lower_bound_ticks=lower_bound_ticks,
            upper_bound_ticks=upper_bound_ticks,
            generation_number=1,
            lifecycle=LiquidityInteractionLifecycle.ARMED,
            armed_at=transition.known_at,
            known_at=transition.known_at,
            updated_at=transition.known_at,
            armed_real_bar_ordinal=_bar_count(state, interaction_timeframe),
            source_event_ids=transition.source_event_ids,
        )
        owners = tuple(
            generation
            for generation in state.structure_generations
            if source_kind in _STRUCTURE_OWNED_SWING_SOURCE_KINDS
            and generation.timeframe is source_timeframe
            and generation.scope is StructureScope.EXTERNAL
            and generation.lifecycle is StructureGenerationLifecycle.CONFIRMED
        )
        if len(owners) > 1:
            raise ValueError("liquidity level has ambiguous structure ownership")
        level = LiquidityLevelState(
            level_id=level_id,
            source_timeframe=source_timeframe,
            source_kind=source_kind,
            source_identity=source_identity,
            side=side,
            price_ticks=price_ticks,
            lower_bound_ticks=lower_bound_ticks,
            upper_bound_ticks=upper_bound_ticks,
            tick_size=tick_size,
            lifecycle=LiquidityLevelLifecycle.ACTIVE,
            created_at=transition.known_at,
            updated_at=transition.known_at,
            active_generation_id=generation_id,
            interaction_generation_ids=(generation_id,),
            source_event_ids=transition.source_event_ids,
            price_anchor_rule=price_anchor_rule,
            owner_structure_generation_id=(
                None if not owners else owners[0].generation_id
            ),
        )
        return replace(
            state,
            levels=_append_item(levels, "level_id", level),
            interactions=_append_item(
                interactions, "generation_id", interaction
            ),
        )

    @staticmethod
    def _liquidity_touched(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        level_id = _required_text(transition.payload, "level_id")
        level = state.level(level_id)
        interaction = _active_interaction(state, level)
        if interaction.interaction_timeframe is not transition.timeframe or level.lifecycle not in {
            LiquidityLevelLifecycle.ACTIVE,
            LiquidityLevelLifecycle.REARMED,
        }:
            raise ValueError("touch references an inactive or foreign level")
        if interaction.lifecycle not in {
            LiquidityInteractionLifecycle.ARMED,
            LiquidityInteractionLifecycle.TOUCHED,
        }:
            raise ValueError("touch cannot follow penetration or terminal resolution")
        if transition.known_at < interaction.armed_at:
            raise ValueError("touch precedes interaction arming")
        bar_event_id = _required_text(transition.payload, "bar_event_id")
        _require_source(transition, bar_event_id, role="touch BAR")
        touched = replace(
            interaction,
            lifecycle=LiquidityInteractionLifecycle.TOUCHED,
            updated_at=transition.known_at,
            first_touch_at=interaction.first_touch_at or transition.known_at,
            touch_bar_ids=_merged_ids(
                interaction.touch_bar_ids, (bar_event_id,)
            ),
            source_event_ids=_merged_ids(
                interaction.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            interactions=_replace_item(
                state.interactions, "generation_id", touched
            ),
            levels=_replace_item(
                state.levels,
                "level_id",
                replace(
                    level,
                    updated_at=transition.known_at,
                    source_event_ids=_merged_ids(
                        level.source_event_ids, transition.source_event_ids
                    ),
                ),
            ),
        )

    @staticmethod
    def _liquidity_penetrated(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        level_id = _required_text(transition.payload, "level_id")
        level = state.level(level_id)
        interaction = _active_interaction(state, level)
        if interaction.interaction_timeframe is not transition.timeframe or level.lifecycle not in {
            LiquidityLevelLifecycle.ACTIVE,
            LiquidityLevelLifecycle.REARMED,
        }:
            raise ValueError("penetration references an inactive or foreign level")
        if interaction.lifecycle not in {
            LiquidityInteractionLifecycle.TOUCHED,
            LiquidityInteractionLifecycle.PENETRATED,
        }:
            raise ValueError("penetration requires an earlier touch")
        penetration_ticks = _required_int(
            transition.payload, "penetration_ticks", minimum=1
        )
        high_ticks = _required_int(
            transition.payload, "high_ticks", minimum=1
        )
        low_ticks = _required_int(
            transition.payload, "low_ticks", minimum=1
        )
        close_ticks = _required_int(
            transition.payload, "close_ticks", minimum=1
        )
        if (
            not low_ticks <= close_ticks <= high_ticks
            or penetration_ticks
            != _bar_penetration_ticks(
                side=level.side,
                lower_bound_ticks=level.lower_bound_ticks,
                upper_bound_ticks=level.upper_bound_ticks,
                high_ticks=high_ticks,
                low_ticks=low_ticks,
            )
        ):
            raise ValueError("penetration fact disagrees with frozen BAR/zone geometry")
        bar_event_id = _required_text(transition.payload, "bar_event_id")
        _require_source(transition, bar_event_id, role="penetration BAR")
        constituents = interaction.constituents
        if interaction.first_penetration_at is None:
            constituents = (
                InteractionConstituent(
                    bar_event_id=bar_event_id,
                    role="penetration",
                    known_at=transition.known_at,
                    high_ticks=high_ticks,
                    low_ticks=low_ticks,
                    close_ticks=close_ticks,
                ),
            )
        penetrated = replace(
            interaction,
            lifecycle=LiquidityInteractionLifecycle.PENETRATED,
            updated_at=transition.known_at,
            first_penetration_at=(
                interaction.first_penetration_at or transition.known_at
            ),
            max_penetration_ticks=max(
                interaction.max_penetration_ticks, penetration_ticks
            ),
            constituents=constituents,
            source_event_ids=_merged_ids(
                interaction.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            interactions=_replace_item(
                state.interactions, "generation_id", penetrated
            ),
            levels=_replace_item(
                state.levels,
                "level_id",
                replace(
                    level,
                    updated_at=transition.known_at,
                    source_event_ids=_merged_ids(
                        level.source_event_ids, transition.source_event_ids
                    ),
                ),
            ),
        )

    @staticmethod
    def _liquidity_terminal(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        level_id = _required_text(transition.payload, "level_id")
        level = state.level(level_id)
        interaction = _active_interaction(state, level)
        if interaction.interaction_timeframe is not transition.timeframe:
            raise ValueError("liquidity terminal crossed interaction timeframe")
        asserted_generation_id = _required_text(
            transition.payload, "interaction_generation_id"
        )
        if asserted_generation_id != interaction.generation_id:
            raise ValueError("liquidity terminal references a stale generation")
        if interaction.lifecycle is not LiquidityInteractionLifecycle.PENETRATED:
            raise ValueError("Sweep or Acceptance requires penetration evidence")
        raw_bar_ids = transition.payload.get("constituent_bar_ids")
        raw_roles = transition.payload.get("constituent_bar_roles")
        raw_clocks = transition.payload.get("constituent_bar_known_at")
        raw_highs = transition.payload.get("constituent_bar_high_ticks")
        raw_lows = transition.payload.get("constituent_bar_low_ticks")
        raw_closes = transition.payload.get("constituent_bar_close_ticks")
        if not all(
            isinstance(value, (tuple, list))
            for value in (
                raw_bar_ids,
                raw_roles,
                raw_clocks,
                raw_highs,
                raw_lows,
                raw_closes,
            )
        ):
            raise ValueError(
                "liquidity terminal requires complete BAR ids, roles, clocks and geometry"
            )
        bar_ids = tuple(raw_bar_ids)
        roles = tuple(raw_roles)
        clocks = tuple(
            aware_timestamp(value, name="terminal.constituent_bar_known_at")
            for value in raw_clocks
        )
        highs = tuple(raw_highs)
        lows = tuple(raw_lows)
        closes = tuple(raw_closes)
        if (
            not bar_ids
            or len(bar_ids) != len(roles)
            or len(clocks) != len(roles)
            or len(highs) != len(roles)
            or len(lows) != len(roles)
            or len(closes) != len(roles)
            or any(not isinstance(value, str) or not value for value in bar_ids)
            or clocks != tuple(sorted(clocks))
            or clocks[-1] != transition.known_at
            or not interaction.constituents
            or bar_ids[0] != interaction.constituents[0].bar_event_id
            or clocks[0] != interaction.first_penetration_at
        ):
            raise ValueError("liquidity terminal formation ledger is invalid")
        for bar_id in set(bar_ids):
            _require_source(transition, bar_id, role="terminal constituent BAR")
        constituents = tuple(
            InteractionConstituent(
                bar_event_id=bar_id,
                role=role,
                known_at=clock,
                high_ticks=high_ticks,
                low_ticks=low_ticks,
                close_ticks=close_ticks,
            )
            for bar_id, role, clock, high_ticks, low_ticks, close_ticks in zip(
                bar_ids,
                roles,
                clocks,
                highs,
                lows,
                closes,
                strict=True,
            )
        )
        first = interaction.constituents[0]
        if (
            constituents[0].bar_event_id,
            constituents[0].known_at,
            constituents[0].high_ticks,
            constituents[0].low_ticks,
            constituents[0].close_ticks,
        ) != (
            first.bar_event_id,
            first.known_at,
            first.high_ticks,
            first.low_ticks,
            first.close_ticks,
        ):
            raise ValueError("terminal rewrites initial penetration BAR geometry")
        terminal_event_id = _required_text(
            transition.payload, "terminal_event_id"
        )
        _require_source(
            transition, terminal_event_id, role="canonical terminal event"
        )
        is_sweep = (
            transition.kind
            is NormalizedTransitionKind.LIQUIDITY_SWEEP_TERMINAL
        )
        terminal_state = (
            LiquidityInteractionTerminal.SWEEP
            if is_sweep
            else LiquidityInteractionTerminal.ACCEPTANCE
        )
        computed_maximum, computed_inside, computed_outside = (
            _validate_formation_constituents(
                constituents,
                terminal_state=terminal_state,
                side=level.side,
                lower_bound_ticks=level.lower_bound_ticks,
                upper_bound_ticks=level.upper_bound_ticks,
            )
        )
        maximum = transition.payload.get("max_penetration_ticks")
        if (
            type(maximum) is not int
            or maximum != computed_maximum
            or maximum < interaction.max_penetration_ticks
        ):
            raise ValueError(
                "terminal max penetration disagrees with complete BAR path"
            )
        if (
            "first_inside_close_at" not in transition.payload
            or "first_outside_close_at" not in transition.payload
        ):
            raise ValueError("terminal requires explicit first close clocks")
        inside = _payload_clock(
            transition.payload,
            "first_inside_close_at",
        )
        outside = _payload_clock(
            transition.payload,
            "first_outside_close_at",
        )
        if inside != computed_inside or outside != computed_outside:
            raise ValueError("terminal first close clocks disagree with BAR path")
        for clock in (inside, outside):
            if clock is not None and (
                interaction.first_penetration_at is None
                or clock < interaction.first_penetration_at
                or clock > transition.known_at
            ):
                raise ValueError("liquidity close ancestry has an invalid clock")
        terminal = replace(
            interaction,
            lifecycle=LiquidityInteractionLifecycle.TERMINAL,
            updated_at=transition.known_at,
            max_penetration_ticks=maximum,
            first_inside_close_at=inside,
            first_outside_close_at=outside,
            terminal_event_id=terminal_event_id,
            terminal_state=terminal_state,
            terminal_at=transition.known_at,
            terminal_reason=terminal_state.value,
            terminal_real_bar_ordinal=_bar_count(
                state, interaction.interaction_timeframe
            ),
            constituents=constituents,
            source_event_ids=_merged_ids(
                interaction.source_event_ids, transition.source_event_ids
            ),
        )
        level_update = replace(
            level,
            lifecycle=(
                LiquidityLevelLifecycle.DISARMED
                if is_sweep
                else LiquidityLevelLifecycle.RETIRED
            ),
            updated_at=transition.known_at,
            active_generation_id=None,
            last_terminal_generation_id=interaction.generation_id,
            rearmable_at=None,
            rearmable_from_generation_id=None,
            rearmable_fact_id=None,
            rearm_departure_bar_event_id=None,
            rearm_departure_ticks=None,
            rearmed_from_generation_id=None,
            retired_at=None if is_sweep else transition.known_at,
            retirement_reason=None if is_sweep else "acceptance",
            source_event_ids=_merged_ids(
                level.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            interactions=_replace_item(
                state.interactions, "generation_id", terminal
            ),
            levels=_replace_item(state.levels, "level_id", level_update),
        )

    @staticmethod
    def _liquidity_unresolved(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        level_id = _required_text(transition.payload, "level_id")
        generation_id = _required_text(
            transition.payload, "interaction_generation_id"
        )
        reason = _required_text(transition.payload, "reason")
        if reason not in _INTERACTION_UNRESOLVED_REASONS:
            raise ValueError("unresolved interaction reason is not preregistered")
        level = state.level(level_id)
        interaction = _active_interaction(state, level)
        if (
            interaction.interaction_timeframe is not transition.timeframe
            or interaction.generation_id != generation_id
            or interaction.lifecycle is LiquidityInteractionLifecycle.TERMINAL
        ):
            raise ValueError("unresolved terminal references a stale interaction")
        unresolved = replace(
            interaction,
            lifecycle=LiquidityInteractionLifecycle.TERMINAL,
            updated_at=transition.known_at,
            terminal_event_id=transition.fact_id,
            terminal_state=LiquidityInteractionTerminal.UNRESOLVED,
            terminal_at=transition.known_at,
            terminal_reason=reason,
            terminal_real_bar_ordinal=_bar_count(
                state, interaction.interaction_timeframe
            ),
            source_event_ids=_merged_ids(
                interaction.source_event_ids, transition.source_event_ids
            ),
        )
        disarmed = replace(
            level,
            lifecycle=LiquidityLevelLifecycle.DISARMED,
            updated_at=transition.known_at,
            active_generation_id=None,
            last_terminal_generation_id=generation_id,
            rearmable_at=None,
            rearmable_from_generation_id=None,
            rearmable_fact_id=None,
            rearm_departure_bar_event_id=None,
            rearm_departure_ticks=None,
            rearmed_from_generation_id=None,
            source_event_ids=_merged_ids(
                level.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            levels=_replace_item(state.levels, "level_id", disarmed),
            interactions=_replace_item(
                state.interactions, "generation_id", unresolved
            ),
        )

    @staticmethod
    def _level_rearmable(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        level_id = _required_text(transition.payload, "level_id")
        prior_generation_id = _required_text(
            transition.payload, "prior_generation_id"
        )
        departure_price_ticks = _required_int(
            transition.payload, "departure_price_ticks", minimum=1
        )
        departure_bar_event_id = _required_text(
            transition.payload, "departure_bar_event_id"
        )
        _require_source(
            transition, departure_bar_event_id, role="rearm departure BAR"
        )
        level = state.level(level_id)
        prior = state.interaction(prior_generation_id)
        if (
            level.source_kind not in REARMABLE_LIQUIDITY_SOURCE_KINDS
            or prior.interaction_timeframe is not transition.timeframe
            or level.lifecycle is not LiquidityLevelLifecycle.DISARMED
            or level.last_terminal_generation_id != prior_generation_id
            or level.active_generation_id is not None
        ):
            raise ValueError(
                "liquidity rearm requires the exact registered disarmed level"
            )
        if (
            prior.level_id != level_id
            or prior.terminal_state is not LiquidityInteractionTerminal.SWEEP
            or prior.terminal_at is None
            or prior.terminal_real_bar_ordinal is None
        ):
            raise ValueError("only a swept interaction generation can rearm")
        current_real_bar = _bar_count(state, prior.interaction_timeframe)
        current_bar_clock = next(
            (
                item
                for item in state.real_bar_clocks
                if item.timeframe is prior.interaction_timeframe
            ),
            None,
        )
        if (
            current_bar_clock is None
            or current_bar_clock.last_bar_event_id != departure_bar_event_id
            or current_bar_clock.last_completed_at > transition.known_at
        ):
            raise ValueError("rearm departure must use the latest exact real BAR")
        departure_ticks = (
            level.lower_bound_ticks - departure_price_ticks
            if level.side == "above"
            else departure_price_ticks - level.upper_bound_ticks
        )
        asserted_departure = transition.payload.get("departure_ticks")
        if asserted_departure is not None and asserted_departure != departure_ticks:
            raise ValueError("asserted departure_ticks disagrees with frozen level side")
        if (
            transition.known_at <= prior.terminal_at
            or current_real_bar < prior.terminal_real_bar_ordinal + 1
            or departure_ticks < 1
        ):
            raise ValueError(
                "rearm requires a strictly later fact, one real BAR and one tick departure"
            )
        rearmable = replace(
            level,
            lifecycle=LiquidityLevelLifecycle.REARMABLE,
            updated_at=transition.known_at,
            rearmable_at=transition.known_at,
            rearmable_from_generation_id=prior_generation_id,
            rearmable_fact_id=transition.fact_id,
            rearm_departure_bar_event_id=departure_bar_event_id,
            rearm_departure_ticks=departure_ticks,
            source_event_ids=_merged_ids(
                level.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            levels=_replace_item(state.levels, "level_id", rearmable),
        )

    @staticmethod
    def _level_rearmed(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        level_id = _required_text(transition.payload, "level_id")
        prior_generation_id = _required_text(
            transition.payload, "prior_generation_id"
        )
        rearmable_fact_id = _required_text(
            transition.payload, "rearmable_fact_id"
        )
        departure_bar_event_id = _required_text(
            transition.payload, "departure_bar_event_id"
        )
        _require_source(
            transition, departure_bar_event_id, role="rearm departure BAR"
        )
        level = state.level(level_id)
        prior = state.interaction(prior_generation_id)
        if (
            level.lifecycle is not LiquidityLevelLifecycle.REARMABLE
            or level.active_generation_id is not None
            or level.last_terminal_generation_id != prior_generation_id
            or level.rearmable_from_generation_id != prior_generation_id
            or level.rearmable_fact_id != rearmable_fact_id
            or level.rearm_departure_bar_event_id != departure_bar_event_id
            or level.rearmable_at != transition.known_at
            or prior.level_id != level_id
            or prior.interaction_timeframe is not transition.timeframe
            or prior.terminal_state is not LiquidityInteractionTerminal.SWEEP
        ):
            raise ValueError(
                "liquidity rearm activation requires exact REARMABLE provenance"
            )
        current_real_bar = _bar_count(state, prior.interaction_timeframe)
        generation_number = prior.generation_number + 1
        generation_id = canonical_semantic_id(
            "liquidity-interaction",
            level_id,
            generation_number,
            transition.known_at,
            prior_generation_id,
        )
        interaction = LiquidityInteractionGeneration(
            generation_id=generation_id,
            level_id=level_id,
            source_timeframe=level.source_timeframe,
            interaction_timeframe=prior.interaction_timeframe,
            level_side=level.side,
            lower_bound_ticks=level.lower_bound_ticks,
            upper_bound_ticks=level.upper_bound_ticks,
            generation_number=generation_number,
            lifecycle=LiquidityInteractionLifecycle.ARMED,
            armed_at=transition.known_at,
            known_at=transition.known_at,
            updated_at=transition.known_at,
            armed_real_bar_ordinal=current_real_bar,
            previous_generation_id=prior_generation_id,
            rearm_fact_id=transition.fact_id,
            source_event_ids=transition.source_event_ids,
        )
        level_update = replace(
            level,
            lifecycle=LiquidityLevelLifecycle.REARMED,
            updated_at=transition.known_at,
            active_generation_id=generation_id,
            interaction_generation_ids=(
                *level.interaction_generation_ids,
                generation_id,
            ),
            rearmed_from_generation_id=prior_generation_id,
            source_event_ids=_merged_ids(
                level.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            levels=_replace_item(state.levels, "level_id", level_update),
            interactions=_append_item(
                state.interactions, "generation_id", interaction
            ),
        )

    @staticmethod
    def _level_retired(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        level_id = _required_text(transition.payload, "level_id")
        reason = _required_text(transition.payload, "reason")
        if reason not in _LEVEL_RETIREMENT_REASONS:
            raise ValueError("liquidity retirement reason is not preregistered")
        level = state.level(level_id)
        if (
            level.source_timeframe is not transition.timeframe
            or level.lifecycle in {
                LiquidityLevelLifecycle.RETIRED,
                LiquidityLevelLifecycle.ARCHIVED,
            }
        ):
            raise ValueError("liquidity retirement cannot rewrite terminal state")
        interactions = state.interactions
        if level.active_generation_id is not None:
            active = _active_interaction(state, level)
            expired = replace(
                active,
                lifecycle=LiquidityInteractionLifecycle.TERMINAL,
                updated_at=transition.known_at,
                terminal_event_id=transition.fact_id,
                terminal_state=LiquidityInteractionTerminal.EXPIRED,
                terminal_at=transition.known_at,
                terminal_reason=reason,
                terminal_real_bar_ordinal=_bar_count(
                    state, active.interaction_timeframe
                ),
                source_event_ids=_merged_ids(
                    active.source_event_ids, transition.source_event_ids
                ),
            )
            interactions = _replace_item(
                interactions, "generation_id", expired
            )
        retired = replace(
            level,
            lifecycle=LiquidityLevelLifecycle.RETIRED,
            updated_at=transition.known_at,
            active_generation_id=None,
            last_terminal_generation_id=(
                level.active_generation_id
                or level.last_terminal_generation_id
            ),
            rearmable_at=None,
            rearmable_from_generation_id=None,
            rearmable_fact_id=None,
            rearm_departure_bar_event_id=None,
            rearm_departure_ticks=None,
            rearmed_from_generation_id=None,
            retired_at=transition.known_at,
            retirement_reason=reason,
            source_event_ids=_merged_ids(
                level.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            levels=_replace_item(state.levels, "level_id", retired),
            interactions=interactions,
        )

    @staticmethod
    def _level_archived(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        level_id = _required_text(transition.payload, "level_id")
        reason = _required_text(transition.payload, "reason")
        level = state.level(level_id)
        if (
            level.source_timeframe is not transition.timeframe
            or level.lifecycle is not LiquidityLevelLifecycle.RETIRED
            or transition.known_at < level.retired_at
        ):
            raise ValueError("archive requires an explicitly retired level")
        archived = replace(
            level,
            lifecycle=LiquidityLevelLifecycle.ARCHIVED,
            updated_at=transition.known_at,
            archived_at=transition.known_at,
            archive_reason=reason,
            source_event_ids=_merged_ids(
                level.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            levels=_replace_item(state.levels, "level_id", archived),
        )

    @staticmethod
    def _structure_started(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        timeframe = Timeframe(transition.timeframe)
        scope = StructureScope(_required_text(transition.payload, "scope"))
        direction = Direction(_required_text(transition.payload, "direction"))
        origin_event_id = _required_text(transition.payload, "origin_event_id")
        origin_swing_id = _required_text(transition.payload, "origin_swing_id")
        _require_source(transition, origin_event_id, role="structure origin")
        if any(
            item.timeframe is timeframe
            and item.scope is scope
            and item.lifecycle is not StructureGenerationLifecycle.TERMINATED
            for item in state.structure_generations
        ):
            raise ValueError(
                "structure generation cannot silently supersede a live scope"
            )
        generation_id = canonical_semantic_id(
            "structure-generation",
            timeframe.value,
            scope.value,
            direction.value,
            transition.known_at,
            origin_event_id,
            origin_swing_id,
        )
        asserted = _optional_text(
            transition.payload, "structure_generation_id"
        )
        if asserted is not None and asserted != generation_id:
            raise ValueError("asserted structure generation id is not canonical")
        generation = StructureGeneration(
            structure_generation_id=generation_id,
            timeframe=timeframe,
            scope=scope,
            direction=direction,
            lifecycle=StructureGenerationLifecycle.FORMING,
            started_at=transition.known_at,
            known_at=transition.known_at,
            updated_at=transition.known_at,
            origin_event_id=origin_event_id,
            origin_swing_id=origin_swing_id,
            source_event_ids=transition.source_event_ids,
        )
        return replace(
            state,
            structure_generations=_append_item(
                state.structure_generations,
                "structure_generation_id",
                generation,
            ),
        )

    @staticmethod
    def _structure_confirmed(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        generation_id = _required_text(
            transition.payload, "structure_generation_id"
        )
        confirmation_event_id = _required_text(
            transition.payload, "confirmation_event_id"
        )
        _require_source(
            transition, confirmation_event_id, role="structure confirmation"
        )
        generation = state.structure(generation_id)
        if (
            generation.timeframe is not transition.timeframe
            or generation.lifecycle is not StructureGenerationLifecycle.FORMING
            or transition.known_at < generation.started_at
            or (
                generation.scope is StructureScope.INTERNAL
                and (
                    transition.known_at <= generation.started_at
                    or confirmation_event_id == generation.origin_event_id
                )
            )
        ):
            raise ValueError("structure confirmation references a non-forming generation")
        confirmed = replace(
            generation,
            lifecycle=StructureGenerationLifecycle.CONFIRMED,
            updated_at=transition.known_at,
            confirmation_event_id=confirmation_event_id,
            confirmed_at=transition.known_at,
            source_event_ids=_merged_ids(
                generation.source_event_ids, transition.source_event_ids
            ),
        )
        levels = state.levels
        if generation.scope is StructureScope.EXTERNAL:
            levels = tuple(
                replace(
                    level,
                    updated_at=transition.known_at,
                    owner_structure_generation_id=generation_id,
                    source_event_ids=_merged_ids(
                        level.source_event_ids,
                        transition.source_event_ids,
                    ),
                )
                if (
                    level.source_timeframe is generation.timeframe
                    and level.source_kind in _STRUCTURE_OWNED_SWING_SOURCE_KINDS
                    and level.owner_structure_generation_id is None
                    and level.lifecycle
                    not in {
                        LiquidityLevelLifecycle.RETIRED,
                        LiquidityLevelLifecycle.ARCHIVED,
                    }
                )
                else level
                for level in levels
            )
        return replace(
            state,
            levels=levels,
            structure_generations=_replace_item(
                state.structure_generations,
                "structure_generation_id",
                confirmed,
            ),
        )

    @staticmethod
    def _structure_evidence(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        generation_id = _required_text(
            transition.payload, "structure_generation_id"
        )
        evidence_kind = _required_text(transition.payload, "evidence_kind")
        evidence_event_id = _required_text(
            transition.payload, "evidence_event_id"
        )
        if evidence_kind not in {"bos", "mss", "protected_swing_assignment"}:
            raise ValueError("structure evidence kind is not preregistered")
        _require_source(transition, evidence_event_id, role="structure evidence")
        generation = state.structure(generation_id)
        if (
            generation.timeframe is not transition.timeframe
            or generation.lifecycle is StructureGenerationLifecycle.TERMINATED
            or (
                evidence_kind == "mss"
                and generation.scope is not StructureScope.INTERNAL
            )
            or (
                evidence_kind != "mss"
                and generation.lifecycle is not StructureGenerationLifecycle.CONFIRMED
            )
        ):
            raise ValueError(
                "structure evidence requires a compatible live generation"
            )
        values: dict[str, Any] = {}
        if evidence_kind == "bos":
            values["bos_event_ids"] = _merged_ids(
                generation.bos_event_ids, (evidence_event_id,)
            )
        elif evidence_kind == "mss":
            values["mss_event_ids"] = _merged_ids(
                generation.mss_event_ids, (evidence_event_id,)
            )
        else:
            protected_swing_id = _required_text(
                transition.payload, "protected_swing_id"
            )
            values["protected_swing_id"] = protected_swing_id
            values["protected_swing_assignment_event_id"] = evidence_event_id
        updated = replace(
            generation,
            updated_at=transition.known_at,
            source_event_ids=_merged_ids(
                generation.source_event_ids, transition.source_event_ids
            ),
            **values,
        )
        return replace(
            state,
            structure_generations=_replace_item(
                state.structure_generations,
                "structure_generation_id",
                updated,
            ),
        )

    @staticmethod
    def _structure_terminated(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        generation_id = _required_text(
            transition.payload, "structure_generation_id"
        )
        reason = _required_text(transition.payload, "reason")
        if reason not in _STRUCTURE_TERMINATION_REASONS:
            raise ValueError("structure termination reason is not preregistered")
        generation = state.structure(generation_id)
        if (
            generation.timeframe is not transition.timeframe
            or generation.lifecycle is StructureGenerationLifecycle.TERMINATED
        ):
            raise ValueError("structure terminal state is immutable")
        protected_acceptance_event_id = _optional_text(
            transition.payload, "protected_acceptance_event_id"
        )
        if reason == "protected_break_accepted":
            if (
                generation.lifecycle is not StructureGenerationLifecycle.CONFIRMED
                or protected_acceptance_event_id is None
            ):
                raise ValueError(
                    "protected-break termination requires confirmed generation and Acceptance"
                )
            _require_source(
                transition,
                protected_acceptance_event_id,
                role="protected Acceptance",
            )
        elif protected_acceptance_event_id is not None:
            raise ValueError("non-protected termination cannot cite Acceptance authority")
        terminated = replace(
            generation,
            lifecycle=StructureGenerationLifecycle.TERMINATED,
            updated_at=transition.known_at,
            terminated_at=transition.known_at,
            termination_reason=reason,
            protected_acceptance_event_id=protected_acceptance_event_id,
            source_event_ids=_merged_ids(
                generation.source_event_ids, transition.source_event_ids
            ),
        )

        levels = state.levels
        interactions = state.interactions
        if generation.scope is StructureScope.EXTERNAL:
            revised_levels: list[LiquidityLevelState] = []
            for level in levels:
                if (
                    level.owner_structure_generation_id != generation_id
                    or level.lifecycle
                    in {
                        LiquidityLevelLifecycle.RETIRED,
                        LiquidityLevelLifecycle.ARCHIVED,
                    }
                ):
                    revised_levels.append(level)
                    continue
                if level.active_generation_id is not None:
                    active = _active_interaction(state, level)
                    expired = replace(
                        active,
                        lifecycle=LiquidityInteractionLifecycle.TERMINAL,
                        updated_at=transition.known_at,
                        terminal_event_id=transition.fact_id,
                        terminal_state=LiquidityInteractionTerminal.EXPIRED,
                        terminal_at=transition.known_at,
                        terminal_reason="structure_generation_terminated",
                        terminal_real_bar_ordinal=_bar_count(
                            state,
                            active.interaction_timeframe,
                        ),
                        source_event_ids=_merged_ids(
                            active.source_event_ids,
                            transition.source_event_ids,
                        ),
                    )
                    interactions = _replace_item(
                        interactions,
                        "generation_id",
                        expired,
                    )
                revised_levels.append(
                    replace(
                        level,
                        lifecycle=LiquidityLevelLifecycle.RETIRED,
                        updated_at=transition.known_at,
                        active_generation_id=None,
                        last_terminal_generation_id=(
                            level.active_generation_id
                            or level.last_terminal_generation_id
                        ),
                        rearmable_at=None,
                        rearmable_from_generation_id=None,
                        rearmable_fact_id=None,
                        rearm_departure_bar_event_id=None,
                        rearm_departure_ticks=None,
                        rearmed_from_generation_id=None,
                        retired_at=transition.known_at,
                        retirement_reason="structure_generation_terminated",
                        source_event_ids=_merged_ids(
                            level.source_event_ids,
                            transition.source_event_ids,
                        ),
                    )
                )
            levels = tuple(revised_levels)

        relations: list[RelationGeneration] = []
        for relation in state.relation_generations:
            if relation.lifecycle is GenerationLifecycle.TERMINATED:
                relations.append(relation)
                continue
            if relation.parent_structure_generation_id == generation_id:
                relation_reason = (
                    "parent_invalidated"
                    if reason == "protected_break_accepted"
                    else "parent_rollover"
                )
            elif relation.child_structure_generation_id == generation_id:
                relation_reason = "child_realigned"
            else:
                relations.append(relation)
                continue
            relations.append(
                replace(
                    relation,
                    lifecycle=GenerationLifecycle.TERMINATED,
                    last_updated_at=transition.known_at,
                    terminated_at=transition.known_at,
                    termination_reason=relation_reason,
                    source_event_ids=_merged_ids(
                        relation.source_event_ids, transition.source_event_ids
                    ),
                )
            )

        deliveries: list[DeliveryPhaseGeneration] = []
        for delivery in state.delivery_generations:
            if (
                delivery.lifecycle is GenerationLifecycle.TERMINATED
                or delivery.parent_structure_generation_id != generation_id
            ):
                deliveries.append(delivery)
                continue
            deliveries.append(
                replace(
                    delivery,
                    lifecycle=GenerationLifecycle.TERMINATED,
                    last_updated_at=transition.known_at,
                    duration_bars=max(
                        0,
                        _bar_count(state, delivery.timeframe)
                        - delivery.entered_real_bar_ordinal,
                    ),
                    duration_seconds=(
                        transition.known_at - delivery.entered_at
                    ).total_seconds(),
                    terminated_at=transition.known_at,
                    termination_reason="parent_structure_terminated",
                    source_event_ids=_merged_ids(
                        delivery.source_event_ids, transition.source_event_ids
                    ),
                )
            )

        transitions: list[StructureTransition] = []
        for candidate in state.structure_transitions:
            incumbent = state.structure(
                candidate.incumbent_structure_generation_id
            )
            exact_challenger_lineage = (
                generation.scope is StructureScope.INTERNAL
                and generation.lifecycle
                in {
                    StructureGenerationLifecycle.FORMING,
                    StructureGenerationLifecycle.CONFIRMED,
                }
                and candidate.lifecycle
                is StructureTransitionLifecycle.STARTED
                and candidate.timeframe is generation.timeframe
                and candidate.challenger_direction is generation.direction
                and candidate.started_at == generation.started_at
                and candidate.mss_event_ids == generation.mss_event_ids
                and incumbent.scope is StructureScope.EXTERNAL
                and incumbent.timeframe is candidate.timeframe
                and incumbent.direction is candidate.incumbent_direction
            )
            exact_pre_acceptance_rollover = (
                exact_challenger_lineage
                and reason == "superseded"
                and candidate.protected_acceptance_event_id is None
                and incumbent.lifecycle
                is StructureGenerationLifecycle.CONFIRMED
            )
            exact_post_acceptance_rollover = (
                exact_challenger_lineage
                and reason == "scope_rollover"
                and candidate.protected_acceptance_event_id is not None
                and incumbent.lifecycle
                is StructureGenerationLifecycle.TERMINATED
                and incumbent.termination_reason
                == "protected_break_accepted"
                and incumbent.protected_acceptance_event_id
                == candidate.protected_acceptance_event_id
                and incumbent.terminated_at is not None
                and incumbent.terminated_at <= transition.known_at
            )
            should_censor = (
                candidate.lifecycle is StructureTransitionLifecycle.STARTED
                and (
                    candidate.incumbent_structure_generation_id == generation_id
                    or candidate.opposite_structure_generation_id == generation_id
                    or exact_pre_acceptance_rollover
                    or exact_post_acceptance_rollover
                )
                and reason != "protected_break_accepted"
            )
            transitions.append(
                replace(
                    candidate,
                    lifecycle=StructureTransitionLifecycle.CENSORED,
                    updated_at=transition.known_at,
                    terminal_at=transition.known_at,
                    terminal_reason=reason,
                    source_event_ids=_merged_ids(
                        candidate.source_event_ids, transition.source_event_ids
                    ),
                )
                if should_censor
                else candidate
            )
        return replace(
            state,
            levels=levels,
            interactions=interactions,
            structure_generations=_replace_item(
                state.structure_generations,
                "structure_generation_id",
                terminated,
            ),
            structure_transitions=tuple(transitions),
            relation_generations=tuple(relations),
            delivery_generations=tuple(deliveries),
        )

    @staticmethod
    def _transition_started(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        incumbent_id = _required_text(
            transition.payload, "incumbent_structure_generation_id"
        )
        challenger = Direction(
            _required_text(transition.payload, "challenger_direction")
        )
        mss_event_id = _required_text(transition.payload, "mss_event_id")
        _require_source(transition, mss_event_id, role="MSS transition evidence")
        incumbent = state.structure(incumbent_id)
        if (
            incumbent.timeframe is not transition.timeframe
            or incumbent.lifecycle is not StructureGenerationLifecycle.CONFIRMED
            or incumbent.direction is challenger
        ):
            raise ValueError("MSS transition evidence lacks an opposed active generation")
        active = tuple(
            item
            for item in state.structure_transitions
            if item.timeframe is incumbent.timeframe
            and item.scope is incumbent.scope
            and item.lifecycle is StructureTransitionLifecycle.STARTED
        )
        structures = state.structure_generations
        incumbent_update = replace(
            incumbent,
            updated_at=transition.known_at,
            mss_event_ids=_merged_ids(incumbent.mss_event_ids, (mss_event_id,)),
            source_event_ids=_merged_ids(
                incumbent.source_event_ids, transition.source_event_ids
            ),
        )
        structures = _replace_item(
            structures,
            "structure_generation_id",
            incumbent_update,
        )
        if active:
            if (
                len(active) != 1
                or active[0].incumbent_structure_generation_id != incumbent_id
                or active[0].challenger_direction is not challenger
            ):
                raise ValueError("one structure scope has conflicting live transitions")
            current = active[0]
            updated = replace(
                current,
                updated_at=transition.known_at,
                mss_event_ids=_merged_ids(current.mss_event_ids, (mss_event_id,)),
                source_event_ids=_merged_ids(
                    current.source_event_ids, transition.source_event_ids
                ),
            )
            return replace(
                state,
                structure_generations=structures,
                structure_transitions=_replace_item(
                    state.structure_transitions,
                    "structure_transition_id",
                    updated,
                ),
            )
        transition_id = canonical_semantic_id(
            "structure-transition",
            incumbent_id,
            challenger.value,
            transition.known_at,
            mss_event_id,
        )
        started = StructureTransition(
            structure_transition_id=transition_id,
            timeframe=incumbent.timeframe,
            scope=incumbent.scope,
            incumbent_structure_generation_id=incumbent_id,
            incumbent_direction=incumbent.direction,
            challenger_direction=challenger,
            lifecycle=StructureTransitionLifecycle.STARTED,
            started_at=transition.known_at,
            updated_at=transition.known_at,
            mss_event_ids=(mss_event_id,),
            source_event_ids=transition.source_event_ids,
        )
        return replace(
            state,
            structure_generations=structures,
            structure_transitions=_append_item(
                state.structure_transitions,
                "structure_transition_id",
                started,
            ),
        )

    @staticmethod
    def _transition_evidence(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        transition_id = _required_text(
            transition.payload, "structure_transition_id"
        )
        acceptance_event_id = _required_text(
            transition.payload, "protected_acceptance_event_id"
        )
        _require_source(
            transition, acceptance_event_id, role="protected Acceptance"
        )
        candidate = state.transition(transition_id)
        if (
            candidate.timeframe is not transition.timeframe
            or candidate.lifecycle is not StructureTransitionLifecycle.STARTED
        ):
            raise ValueError("transition evidence cannot rewrite terminal history")
        if (
            candidate.protected_acceptance_event_id is not None
            and candidate.protected_acceptance_event_id != acceptance_event_id
        ):
            raise ValueError("transition already records a different protected Acceptance")
        if candidate.protected_acceptance_event_id == acceptance_event_id:
            return state
        incumbent = state.structure(
            candidate.incumbent_structure_generation_id
        )
        if (
            incumbent.lifecycle is StructureGenerationLifecycle.TERMINATED
            and (
                incumbent.termination_reason != "protected_break_accepted"
                or incumbent.protected_acceptance_event_id != acceptance_event_id
            )
        ):
            raise ValueError("transition Acceptance conflicts with incumbent terminal")
        updated = replace(
            candidate,
            updated_at=transition.known_at,
            protected_acceptance_event_id=acceptance_event_id,
            source_event_ids=_merged_ids(
                candidate.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            structure_transitions=_replace_item(
                state.structure_transitions,
                "structure_transition_id",
                updated,
            ),
        )

    @staticmethod
    def _transition_confirmed(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        transition_id = _required_text(
            transition.payload, "structure_transition_id"
        )
        acceptance_event_id = _required_text(
            transition.payload, "protected_acceptance_event_id"
        )
        opposite_generation_id = _required_text(
            transition.payload, "opposite_structure_generation_id"
        )
        opposite_confirmation_event_id = _required_text(
            transition.payload, "opposite_confirmation_event_id"
        )
        for identity, role in (
            (acceptance_event_id, "protected Acceptance"),
            (opposite_confirmation_event_id, "opposite structure confirmation"),
        ):
            _require_source(transition, identity, role=role)
        candidate = state.transition(transition_id)
        if (
            candidate.timeframe is not transition.timeframe
            or candidate.lifecycle is not StructureTransitionLifecycle.STARTED
        ):
            raise ValueError("structure transition terminal state is immutable")
        incumbent = state.structure(
            candidate.incumbent_structure_generation_id
        )
        opposite = state.structure(opposite_generation_id)
        if (
            candidate.protected_acceptance_event_id != acceptance_event_id
            or
            incumbent.lifecycle is not StructureGenerationLifecycle.TERMINATED
            or incumbent.termination_reason != "protected_break_accepted"
            or incumbent.protected_acceptance_event_id != acceptance_event_id
            or opposite.lifecycle is not StructureGenerationLifecycle.CONFIRMED
            or opposite.timeframe is not candidate.timeframe
            or opposite.scope is not candidate.scope
            or opposite.direction is not candidate.challenger_direction
            or opposite.confirmation_event_id != opposite_confirmation_event_id
            or opposite.confirmed_at is None
            or opposite.confirmed_at <= candidate.started_at
            or transition.known_at < max(incumbent.terminated_at, opposite.confirmed_at)
        ):
            raise ValueError(
                "transition confirmation lacks exact Acceptance and opposite generation"
            )
        confirmed = replace(
            candidate,
            lifecycle=StructureTransitionLifecycle.CONFIRMED,
            updated_at=transition.known_at,
            terminal_at=transition.known_at,
            terminal_reason="protected_acceptance_plus_opposite_confirmation",
            protected_acceptance_event_id=acceptance_event_id,
            opposite_structure_generation_id=opposite_generation_id,
            opposite_confirmation_event_id=opposite_confirmation_event_id,
            source_event_ids=_merged_ids(
                candidate.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            structure_transitions=_replace_item(
                state.structure_transitions,
                "structure_transition_id",
                confirmed,
            ),
        )

    @staticmethod
    def _transition_failed(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        transition_id = _required_text(
            transition.payload, "structure_transition_id"
        )
        resumed_generation_id = _required_text(
            transition.payload, "resumed_structure_generation_id"
        )
        resumption_event_id = _required_text(
            transition.payload, "resumption_event_id"
        )
        _require_source(transition, resumption_event_id, role="direction resumption")
        candidate = state.transition(transition_id)
        resumed = state.structure(resumed_generation_id)
        if (
            candidate.timeframe is not transition.timeframe
            or candidate.lifecycle is not StructureTransitionLifecycle.STARTED
            or transition.known_at <= candidate.started_at
            or resumed.timeframe is not candidate.timeframe
            or resumed.scope is not candidate.scope
            or resumed.direction is not candidate.incumbent_direction
            or resumed.lifecycle is not StructureGenerationLifecycle.CONFIRMED
        ):
            raise ValueError("transition failure lacks later original-direction evidence")
        failed = replace(
            candidate,
            lifecycle=StructureTransitionLifecycle.FAILED,
            updated_at=transition.known_at,
            terminal_at=transition.known_at,
            terminal_reason="original_direction_resumed",
            resumption_event_id=resumption_event_id,
            source_event_ids=_merged_ids(
                candidate.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            structure_transitions=_replace_item(
                state.structure_transitions,
                "structure_transition_id",
                failed,
            ),
        )

    @staticmethod
    def _relation_observed(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        source_relation_id = _required_text(
            transition.payload, "source_relation_id"
        )
        parent_tf = Timeframe(_required_text(transition.payload, "parent_tf"))
        child_tf = Timeframe(_required_text(transition.payload, "child_tf"))
        parent_generation_id = _required_text(
            transition.payload, "parent_structure_generation_id"
        )
        child_generation_id = _required_text(
            transition.payload, "child_structure_generation_id"
        )
        role = _required_text(transition.payload, "role")
        relation_digest = _required_text(
            transition.payload, "relation_digest"
        )
        if (
            role not in _RELATION_ROLES
            or parent_tf is child_tf
            or child_tf is not transition.timeframe
        ):
            raise ValueError("relation observation identity or role is invalid")
        parent = state.structure(parent_generation_id)
        child = state.structure(child_generation_id)
        if (
            parent.timeframe is not parent_tf
            or child.timeframe is not child_tf
            or parent.lifecycle is not StructureGenerationLifecycle.CONFIRMED
            or child.lifecycle is not StructureGenerationLifecycle.CONFIRMED
        ):
            raise ValueError("relation generation requires two confirmed owners")
        parent_direction = _optional_text(
            transition.payload, "parent_direction"
        )
        child_direction = _optional_text(transition.payload, "child_direction")
        if (
            parent_direction is not None
            and Direction(parent_direction) is not parent.direction
        ):
            raise ValueError("relation parent direction conflicts with its owner")
        if child_direction is not None:
            # RelationState.child_direction is the child's *internal*
            # delivery/transition evidence.  The bound child generation above
            # remains its confirmed external owner, so an internal MSS against
            # that owner must not be rejected or rebound to a forming regime.
            Direction(child_direction)

        signature = (parent_generation_id, child_generation_id, role)
        active = next(
            (
                item
                for item in state.relation_generations
                if item.parent_tf is parent_tf
                and item.child_tf is child_tf
                and item.lifecycle is GenerationLifecycle.ACTIVE
            ),
            None,
        )
        relations = state.relation_generations
        if active is not None and active.signature == signature:
            if transition.known_at < active.last_updated_at:
                raise ValueError("relation update moved backwards")
            updated = replace(
                active,
                source_relation_id=source_relation_id,
                last_updated_at=transition.known_at,
                observation_count=(
                    active.observation_count
                    if transition.known_at == active.last_updated_at
                    else active.observation_count + 1
                ),
                latest_relation_digest=relation_digest,
                source_event_ids=_merged_ids(
                    active.source_event_ids, transition.source_event_ids
                ),
            )
            return replace(
                state,
                relation_generations=_replace_item(
                    relations, "relation_generation_id", updated
                ),
            )
        if active is not None:
            if transition.known_at <= active.last_updated_at:
                raise ValueError(
                    "same-clock relation reclassification is not an ordered generation"
                )
            if active.parent_structure_generation_id != parent_generation_id:
                old_parent = state.structure(
                    active.parent_structure_generation_id
                )
                reason = (
                    "parent_invalidated"
                    if old_parent.termination_reason == "protected_break_accepted"
                    else "parent_rollover"
                )
            elif active.child_structure_generation_id != child_generation_id:
                reason = "child_realigned"
            else:
                reason = "relation_reclassified"
            ended = replace(
                active,
                lifecycle=GenerationLifecycle.TERMINATED,
                last_updated_at=transition.known_at,
                terminated_at=transition.known_at,
                termination_reason=reason,
                source_event_ids=_merged_ids(
                    active.source_event_ids, transition.source_event_ids
                ),
            )
            relations = _replace_item(
                relations, "relation_generation_id", ended
            )
        generation_id = canonical_semantic_id(
            "relation-generation",
            parent_tf.value,
            child_tf.value,
            *signature,
            transition.known_at,
        )
        generation = RelationGeneration(
            relation_generation_id=generation_id,
            source_relation_id=source_relation_id,
            parent_tf=parent_tf,
            child_tf=child_tf,
            parent_structure_generation_id=parent_generation_id,
            child_structure_generation_id=child_generation_id,
            role=role,
            lifecycle=GenerationLifecycle.ACTIVE,
            entered_at=transition.known_at,
            known_at=transition.known_at,
            last_updated_at=transition.known_at,
            observation_count=1,
            latest_relation_digest=relation_digest,
            source_event_ids=transition.source_event_ids,
        )
        return replace(
            state,
            relation_generations=_append_item(
                relations, "relation_generation_id", generation
            ),
        )

    @staticmethod
    def _relation_terminated(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        generation_id = _required_text(
            transition.payload, "relation_generation_id"
        )
        reason = _required_text(transition.payload, "reason")
        if reason not in _RELATION_TERMINATION_REASONS:
            raise ValueError("relation termination reason is not preregistered")
        generation = _find(
            state.relation_generations, "relation_generation_id", generation_id
        )
        if (
            generation.child_tf is not transition.timeframe
            or generation.lifecycle is GenerationLifecycle.TERMINATED
        ):
            raise ValueError("relation terminal state is immutable")
        terminated = replace(
            generation,
            lifecycle=GenerationLifecycle.TERMINATED,
            last_updated_at=transition.known_at,
            terminated_at=transition.known_at,
            termination_reason=reason,
            source_event_ids=_merged_ids(
                generation.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            relation_generations=_replace_item(
                state.relation_generations,
                "relation_generation_id",
                terminated,
            ),
        )

    @staticmethod
    def _delivery_observed(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        timeframe = Timeframe(transition.timeframe)
        phase = _required_text(transition.payload, "phase")
        if phase not in _DELIVERY_PHASES:
            raise ValueError("delivery phase is not registered")
        parent_generation_id = _required_text(
            transition.payload, "parent_structure_generation_id"
        )
        origin_event_id = _required_text(
            transition.payload, "origin_event_id"
        )
        _require_source(transition, origin_event_id, role="delivery origin")
        parent = state.structure(parent_generation_id)
        if (
            parent.timeframe is not timeframe
            or parent.lifecycle is not StructureGenerationLifecycle.CONFIRMED
        ):
            raise ValueError("delivery generation requires a confirmed parent")
        current_price_ticks = _required_int(
            transition.payload, "current_price_ticks", minimum=1
        )
        asserted_extension = _optional_nonnegative_number(
            transition.payload, "extension_ticks"
        )
        asserted_retracement = _optional_nonnegative_number(
            transition.payload, "retracement_ticks"
        )
        active = next(
            (
                item
                for item in state.delivery_generations
                if item.timeframe is timeframe
                and item.lifecycle is GenerationLifecycle.ACTIVE
            ),
            None,
        )
        signature = (parent_generation_id, phase)
        deliveries = state.delivery_generations
        if active is not None and active.signature == signature:
            if transition.known_at < active.last_updated_at:
                raise ValueError("delivery update moved backwards")
            raw_delta = current_price_ticks - active.origin_price_ticks
            favorable = (
                max(raw_delta, 0)
                if parent.direction is Direction.LONG
                else max(-raw_delta, 0)
            )
            adverse = (
                max(-raw_delta, 0)
                if parent.direction is Direction.LONG
                else max(raw_delta, 0)
            )
            max_extension = max(active.max_extension_ticks, float(favorable))
            max_retracement = max(
                active.max_retracement_ticks, float(adverse)
            )
            if (
                asserted_extension is not None
                and asserted_extension != max_extension
            ) or (
                asserted_retracement is not None
                and asserted_retracement != max_retracement
            ):
                raise ValueError(
                    "caller delivery extrema disagree with frozen price path"
                )
            updated = replace(
                active,
                last_updated_at=transition.known_at,
                duration_bars=max(
                    0,
                    _bar_count(state, timeframe)
                    - active.entered_real_bar_ordinal,
                ),
                duration_seconds=(
                    transition.known_at - active.entered_at
                ).total_seconds(),
                observation_count=(
                    active.observation_count
                    if transition.known_at == active.last_updated_at
                    else active.observation_count + 1
                ),
                current_price_ticks=current_price_ticks,
                max_extension_ticks=max_extension,
                max_retracement_ticks=max_retracement,
                source_event_ids=_merged_ids(
                    active.source_event_ids, transition.source_event_ids
                ),
            )
            return replace(
                state,
                delivery_generations=_replace_item(
                    deliveries, "delivery_generation_id", updated
                ),
            )
        if active is not None:
            if transition.known_at <= active.last_updated_at:
                raise ValueError(
                    "same-clock delivery observations must be coalesced"
                )
            if active.parent_structure_generation_id != parent_generation_id:
                raise ValueError(
                    "delivery parent change requires explicit prior generation termination"
                )
            raw_delta = current_price_ticks - active.origin_price_ticks
            favorable = (
                max(raw_delta, 0)
                if parent.direction is Direction.LONG
                else max(-raw_delta, 0)
            )
            adverse = (
                max(-raw_delta, 0)
                if parent.direction is Direction.LONG
                else max(raw_delta, 0)
            )
            ended = replace(
                active,
                lifecycle=GenerationLifecycle.TERMINATED,
                last_updated_at=transition.known_at,
                duration_bars=max(
                    0,
                    _bar_count(state, timeframe)
                    - active.entered_real_bar_ordinal,
                ),
                duration_seconds=(
                    transition.known_at - active.entered_at
                ).total_seconds(),
                observation_count=active.observation_count + 1,
                current_price_ticks=current_price_ticks,
                max_extension_ticks=max(
                    active.max_extension_ticks, float(favorable)
                ),
                max_retracement_ticks=max(
                    active.max_retracement_ticks, float(adverse)
                ),
                terminated_at=transition.known_at,
                termination_reason="phase_changed",
                next_phase=phase,
                source_event_ids=_merged_ids(
                    active.source_event_ids, transition.source_event_ids
                ),
            )
            deliveries = _replace_item(
                deliveries, "delivery_generation_id", ended
            )
        if (
            asserted_extension not in {None, 0.0}
            or asserted_retracement not in {None, 0.0}
        ):
            raise ValueError(
                "new delivery generation extrema must begin at frozen origin"
            )
        generation_id = canonical_semantic_id(
            "delivery-generation",
            timeframe.value,
            parent_generation_id,
            phase,
            transition.known_at,
            origin_event_id,
        )
        generation = DeliveryPhaseGeneration(
            delivery_generation_id=generation_id,
            timeframe=timeframe,
            phase=phase,
            parent_structure_generation_id=parent_generation_id,
            lifecycle=GenerationLifecycle.ACTIVE,
            entered_at=transition.known_at,
            known_at=transition.known_at,
            last_updated_at=transition.known_at,
            entered_real_bar_ordinal=_bar_count(state, timeframe),
            duration_bars=0,
            duration_seconds=0.0,
            origin_event_id=origin_event_id,
            observation_count=1,
            origin_price_ticks=current_price_ticks,
            current_price_ticks=current_price_ticks,
            max_extension_ticks=0.0,
            max_retracement_ticks=0.0,
            source_event_ids=transition.source_event_ids,
        )
        return replace(
            state,
            delivery_generations=_append_item(
                deliveries, "delivery_generation_id", generation
            ),
        )

    @staticmethod
    def _delivery_terminated(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        generation_id = _required_text(
            transition.payload, "delivery_generation_id"
        )
        reason = _required_text(transition.payload, "reason")
        if reason not in _DELIVERY_TERMINATION_REASONS:
            raise ValueError("delivery termination reason is not preregistered")
        generation = _find(
            state.delivery_generations, "delivery_generation_id", generation_id
        )
        if (
            generation.timeframe is not transition.timeframe
            or generation.lifecycle is GenerationLifecycle.TERMINATED
        ):
            raise ValueError("delivery terminal state is immutable")
        terminated = replace(
            generation,
            lifecycle=GenerationLifecycle.TERMINATED,
            last_updated_at=transition.known_at,
            duration_bars=max(
                0,
                _bar_count(state, generation.timeframe)
                - generation.entered_real_bar_ordinal,
            ),
            duration_seconds=(
                transition.known_at - generation.entered_at
            ).total_seconds(),
            terminated_at=transition.known_at,
            termination_reason=reason,
            source_event_ids=_merged_ids(
                generation.source_event_ids, transition.source_event_ids
            ),
        )
        return replace(
            state,
            delivery_generations=_replace_item(
                state.delivery_generations,
                "delivery_generation_id",
                terminated,
            ),
        )

    @staticmethod
    def _boundary_attack(
        state: SemanticLifecycleState,
        transition: NormalizedLifecycleTransition,
    ) -> SemanticLifecycleState:
        timeframe = Timeframe(transition.timeframe)
        bos_generation_id = _required_text(
            transition.payload, "bos_generation_id"
        )
        direction = Direction(_required_text(transition.payload, "direction"))
        target_swing_event_id = _required_text(
            transition.payload, "target_swing_event_id"
        )
        bar_event_id = _required_text(transition.payload, "bar_event_id")
        for identity, role in (
            (target_swing_event_id, "boundary target Swing"),
            (bar_event_id, "boundary attack BAR"),
        ):
            _require_source(transition, identity, role=role)
        boundary_ticks = _required_int(
            transition.payload, "boundary_ticks", minimum=1
        )
        high_ticks = _required_int(
            transition.payload, "high_ticks", minimum=1
        )
        low_ticks = _required_int(
            transition.payload, "low_ticks", minimum=1
        )
        close_ticks = _required_int(
            transition.payload, "close_ticks", minimum=1
        )
        if not low_ticks <= close_ticks <= high_ticks:
            raise ValueError("boundary attack BAR has invalid tick geometry")
        if direction is Direction.LONG:
            if not high_ticks > boundary_ticks or close_ticks > boundary_ticks:
                raise ValueError(
                    "long boundary attack requires strict high cross without close break"
                )
            extreme_ticks = high_ticks
            penetration_ticks = high_ticks - boundary_ticks
        else:
            if not low_ticks < boundary_ticks or close_ticks < boundary_ticks:
                raise ValueError(
                    "short boundary attack requires strict low cross without close break"
                )
            extreme_ticks = low_ticks
            penetration_ticks = boundary_ticks - low_ticks
        boundary_attack_id = canonical_semantic_id(
            "boundary-attack",
            timeframe.value,
            bos_generation_id,
            bar_event_id,
            direction.value,
            boundary_ticks,
        )
        if any(
            item.boundary_attack_id == boundary_attack_id
            for item in state.boundary_attacks
        ):
            raise ValueError("one BOS generation cannot duplicate an attack BAR")
        attempt_ordinal = 1 + sum(
            item.bos_generation_id == bos_generation_id
            for item in state.boundary_attacks
        )
        fact = BoundaryAttackFact(
            boundary_attack_id=boundary_attack_id,
            bos_generation_id=bos_generation_id,
            timeframe=timeframe,
            direction=direction,
            target_swing_event_id=target_swing_event_id,
            bar_event_id=bar_event_id,
            boundary_ticks=boundary_ticks,
            extreme_ticks=extreme_ticks,
            close_ticks=close_ticks,
            penetration_ticks=penetration_ticks,
            attempt_ordinal=attempt_ordinal,
            known_at=transition.known_at,
            source_event_ids=transition.source_event_ids,
        )
        return replace(
            state,
            boundary_attacks=_append_item(
                state.boundary_attacks, "boundary_attack_id", fact
            ),
        )
