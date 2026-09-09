"""Brain hypothesis contracts: typed evidence, sequence progress and belief.

``HypothesisBelief`` is one candidate's complete typed projection: strength,
sequence progress, location quality, entry readiness, delivery and
uncertainty, plus the frozen plan and terminal custody."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import pandas as pd

from contract.market.primitives import Direction, LiquidityLevel, Playbook, PlaybookPhase, StructuralLevel, aware_timestamp, clamp
from contract.brain.plan import DrawSelection, FrozenTriggerState, LiquidityRoute, PlanFeasibility, TradePlan


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


__all__ = [
    "Evidence",
    "HypothesisBelief",
    "HypothesisSequenceState",
    "SequenceStepState",
]
