"""Brain context contracts: authority, obstruction, thesis and neutral state.

``GlobalMarketContext`` is the per-clock authority view; ``OpenMarketThesis``
and its Context/Episode lifecycles are playbook-neutral descriptions that
never carry action authority by themselves."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import pandas as pd

from contract.market.primitives import Direction, LiquidityLevel, MarketMode, Playbook, PlaybookPhase, ScaleRelation, StructuralLevel, Timeframe, aware_timestamp
from contract.brain.vocabulary import GlobalConflictRole, NEUTRAL_MARKET_STATE_SCHEMA_VERSION
from contract.brain.plan import FrozenTriggerState, TradePlan


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


__all__ = [
    "AuthorityLayer",
    "BalanceContext",
    "ContextThesisState",
    "DeliveryObstruction",
    "DirectionalObstructionView",
    "EntryEpisodeState",
    "GlobalConflictEvidence",
    "GlobalMarketContext",
    "MarketEpisodeState",
    "NeutralMarketState",
    "OpenMarketThesis",
    "OpenMarketThesisClaimRelation",
    "ScaleRelationState",
    "ThesisEvidenceState",
]
