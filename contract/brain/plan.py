"""Brain plan contracts: frozen entry geometry, routes and feasibility.

A plan freezes entry, invalidation, targets and deadline. Later extrema may
not rewrite it; feasibility only validates geometry it was given."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import pandas as pd

from contract.market.primitives import Direction, LiquidityLevel, Playbook, StructuralLevel, Timeframe, aware_timestamp
from contract.eye.vocabulary import LiquidityInventoryLifecycle


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
    balance_confirmed_at: pd.Timestamp
    manipulation_side: str
    swept_at: pd.Timestamp
    manipulation_extreme: float
    reentry_candidate_at: pd.Timestamp
    reentered_at: pd.Timestamp
    reentry_price: float
    opposite_liquidity_id: str

    def __post_init__(self) -> None:
        for name in (
            "balance_confirmed_at",
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
            or not self.balance_confirmed_at
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
                    self.range_auction.balance_confirmed_at,
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


__all__ = [
    "DrawSelection",
    "FrozenLSRContext",
    "FrozenRangeAuctionContext",
    "FrozenTriggerState",
    "LiquidityRoute",
    "PlanFeasibility",
    "TradePlan",
]
