"""Net-utility comparison across explicit trading actions."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .calibration import TYPED_ACTIVE_PLAYBOOKS
from shares.core.model import (
    AccountState,
    Action,
    ActionUtility,
    Decision,
    Direction,
    HypothesisBelief,
    MarketBelief,
    MarketObservation,
    PlaybookPhase,
    clamp,
)
from .risk import causal_protection_candidate


@dataclass(frozen=True)
class DecisionConfig:
    minimum_utility_advantage: float = 0.12
    uncertainty_penalty: float = 0.35
    deadline_penalty_minutes: int = 20
    maximum_reward_R: float = 3.0


def _cost_R(observation: MarketObservation, risk_points: float) -> float:
    if risk_points <= 0:
        return 99.0
    return observation.execution.expected_round_trip_cost_points / risk_points


def _deadline_penalty(observation: MarketObservation, config: DecisionConfig) -> float:
    remaining = observation.execution.minutes_to_deadline
    return clamp(
        (config.deadline_penalty_minutes - remaining)
        / max(1.0, float(config.deadline_penalty_minutes))
    )


def _evidence_reason(belief: HypothesisBelief) -> str:
    support = ", ".join(item.primitive for item in belief.supporting[:3]) or "no strong support"
    against = ", ".join(item.primitive for item in belief.contradicting[:2]) or "no strong contradiction"
    typed = ""
    if belief.thesis_strength is not None:
        failed_gates = [
            name
            for name, passed in belief.hard_gate_results.items()
            if not passed
        ]
        typed = (
            f"; thesis={belief.thesis_strength:.3f}, "
            f"sequence={belief.sequence_progress:.3f}, "
            f"location={belief.location_quality:.3f}, "
            f"readiness={belief.entry_readiness:.3f}, "
            f"delivery={belief.delivery_quality:.3f}; "
            f"failed gates: {', '.join(failed_gates) or 'none'}"
        )
    return (
        f"{belief.playbook.value}/{belief.direction.value} p={belief.probability:.3f}, "
        f"phase={belief.phase.value}{typed}; supports: {support}; against: {against}"
    )


def _is_typed(belief: HypothesisBelief) -> bool:
    return all(
        value is not None
        for value in (
            belief.thesis_strength,
            belief.sequence_progress,
            belief.location_quality,
            belief.entry_readiness,
            belief.delivery_quality,
        )
    )


def _graph_ambiguity_count(belief: HypothesisBelief) -> int:
    """Return the typed Brain's contemporaneous graph ambiguity count."""

    raw = belief.context_metadata.get("ambiguity_count", "0")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        # An unreadable typed diagnostic is not safe directional evidence.
        return 1
    return max(0, value)


def _frozen_primary_target_R(hypothesis: HypothesisBelief) -> float | None:
    """Derive reward/R from frozen prices rather than a cached score."""

    plan = hypothesis.plan
    if plan is None or not plan.targets:
        return None
    risk_points = abs(plan.planned_entry - plan.invalidation.price)
    if not math.isfinite(risk_points) or risk_points <= 0.0:
        return None
    reward_points = plan.direction.sign * (
        plan.targets[0].price - plan.planned_entry
    )
    reward_R = reward_points / risk_points
    if not math.isfinite(reward_R) or reward_R <= 0.0:
        return None
    return float(reward_R)


def _action_candidate_items(
    belief: MarketBelief,
) -> tuple[tuple[str, HypothesisBelief], ...]:
    """Return the strict action-facing candidate contract.

    ``MarketBelief.action_candidate_items`` is the authoritative interface for
    root-specific candidates.  A belief without that interface is not a valid
    production action source and therefore fails closed.  Graph-free fixtures
    may provide an explicit test-only implementation of the same interface.
    """

    provider = getattr(belief, "action_candidate_items", None)
    if callable(provider):
        items = tuple(provider())
    else:
        items = ()
    if any(
        not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(hypothesis, HypothesisBelief)
        for candidate_id, hypothesis in items
    ):
        raise ValueError(
            "action candidates must be (non-empty candidate_id, "
            "HypothesisBelief) pairs"
        )
    if len({candidate_id for candidate_id, _ in items}) != len(items):
        raise ValueError("action candidate IDs must be unique")
    return items


def _position_candidate_items(
    belief: MarketBelief,
) -> tuple[tuple[str, HypothesisBelief], ...]:
    """Return candidates allowed to manage an already-frozen position.

    Graph-backed beliefs may retain one closed-root candidate for
    HOLD/PROTECT/EXIT resolution.  That candidate is intentionally absent
    from ``_action_candidate_items`` and therefore cannot authorize ENTER.
    """

    provider = getattr(belief, "position_candidate_items", None)
    items = (
        tuple(provider())
        if callable(provider)
        else _action_candidate_items(belief)
    )
    if any(
        not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(hypothesis, HypothesisBelief)
        for candidate_id, hypothesis in items
    ):
        raise ValueError(
            "position candidates must be (non-empty candidate_id, "
            "HypothesisBelief) pairs"
        )
    if len({candidate_id for candidate_id, _ in items}) != len(items):
        raise ValueError("position candidate IDs must be unique")
    return items


def _concrete_action_identity(
    utility: ActionUtility,
    belief: MarketBelief,
) -> tuple[object, ...]:
    """Return the exact instruction represented by one raw utility row."""

    if utility.action is not Action.ENTER:
        return ("verb", utility.action.value)

    if utility.hypothesis_key is None:
        hypothesis = None
    else:
        resolver = getattr(belief, "resolve_hypothesis", None)
        hypothesis = (
            resolver(utility.hypothesis_key)
            if callable(resolver)
            else belief.hypotheses.get(utility.hypothesis_key)
        )
    plan = None if hypothesis is None else hypothesis.plan
    if plan is None:
        # An ENTER without a frozen plan is not executable. Keep it separate
        # by hypothesis (or by raw row when even that identity is absent) so
        # malformed candidates can never inflate the measured advantage.
        malformed_identity: object = (
            utility.hypothesis_key
            if utility.hypothesis_key is not None
            else id(utility)
        )
        return ("enter_without_plan", malformed_identity)

    invalidation = plan.invalidation
    targets = tuple(
        (
            target.level_id,
            target.timeframe.value,
            target.side,
            target.price,
            target.confirmed_at,
        )
        for target in plan.targets
    )
    return (
        "enter",
        plan.direction.value,
        plan.planned_entry,
        (
            invalidation.price,
            invalidation.side,
            invalidation.source_level_id,
            invalidation.observed_at,
        ),
        targets,
        plan.deadline,
        plan.setup_id,
        plan.entry_location_id,
        plan.entry_path_id,
        plan.entry_zone_lower,
        plan.entry_zone_upper,
        plan.selected_draw_id,
        getattr(plan, "draw_selection", None),
        getattr(plan, "range_auction", None),
        getattr(plan, "lsr_context", None),
    )


class UtilityDecisionLayer:
    def __init__(
        self,
        config: DecisionConfig | None = None,
        *,
        calibration_ready: bool = False,
        calibration_version: str = "identity-unvalidated",
    ) -> None:
        self.config = config or DecisionConfig()
        version = str(calibration_version).strip()
        if not version:
            raise ValueError("decision calibration version is required")
        if calibration_ready and version == "identity-unvalidated":
            raise ValueError(
                "a ready decision layer requires an explicit calibration version"
            )
        self.calibration_ready = bool(calibration_ready)
        self.calibration_version = version

    def _causal_entry_gate(
        self,
        belief: MarketBelief,
        candidate_id: str,
        observation: MarketObservation,
        hypothesis: HypothesisBelief,
    ) -> tuple[bool, tuple[str, ...]]:
        """Validate causal eligibility without interpreting quality scores.

        Thesis, location and readiness are descriptive/calibrated dimensions,
        not probabilities and not duplicate gates.  The phase, current hard
        gates, exact identities and frozen plan are the action authority.
        """

        reasons: list[str] = []
        context_thesis_id = hypothesis.context_thesis_id
        episode_id = hypothesis.episode_id
        if not isinstance(context_thesis_id, str) or not context_thesis_id:
            reasons.append("context_thesis_id_missing")
        if not isinstance(episode_id, str) or not episode_id:
            reasons.append("entry_episode_id_missing")
        if hypothesis.parent_context_thesis_id != context_thesis_id:
            reasons.append("entry_episode_parent_mismatch")
        if (
            hypothesis.plan is None
            or not isinstance(episode_id, str)
            or not episode_id
            or hypothesis.plan.setup_id != episode_id
        ):
            reasons.append("entry_episode_plan_setup_mismatch")
        ownership = getattr(belief, "owns_actionable_entry_episode", None)
        if not callable(ownership) or not ownership(candidate_id, hypothesis):
            reasons.append("entry_episode_projection_mismatch")
        if hypothesis.playbook not in TYPED_ACTIVE_PLAYBOOKS:
            reasons.append("playbook_not_action_calibrated")
        if hypothesis.phase is not PlaybookPhase.EXECUTABLE:
            reasons.append("phase_not_executable")
        failed_gates = tuple(
            name
            for name, passed in hypothesis.hard_gate_results.items()
            if not passed
        )
        if not hypothesis.hard_gate_results:
            reasons.append("current_hard_gates_missing")
        elif failed_gates:
            reasons.append("current_hard_gates_failed:" + ",".join(failed_gates))
        plan = hypothesis.plan
        if plan is None:
            reasons.append("frozen_plan_missing")
        else:
            trigger = hypothesis.selected_trigger
            if (
                plan.playbook is not hypothesis.playbook
                or plan.direction is not hypothesis.direction
                or plan.setup_id is None
                or plan.setup_id != hypothesis.setup_context_id
                or hypothesis.sequence is None
                or hypothesis.sequence.setup_id != plan.setup_id
                or plan.entry_location_id is None
                or plan.entry_location_id != hypothesis.entry_location_id
                or plan.entry_path_id is None
                or plan.entry_path_id != hypothesis.entry_path_id
                or trigger is None
                or trigger.setup_id != plan.setup_id
                or trigger.entry_location_id != plan.entry_location_id
                or trigger.entry_path_id != plan.entry_path_id
                or trigger.direction is not plan.direction
                or hypothesis.invalidation != plan.invalidation
                or hypothesis.deliverable_targets != plan.targets
            ):
                reasons.append("frozen_plan_hypothesis_identity_mismatch")
            frozen_target_R = _frozen_primary_target_R(hypothesis)
            if plan.deadline <= observation.asof:
                reasons.append("frozen_plan_deadline_elapsed")
            if (
                not math.isfinite(float(plan.risk_points))
                or plan.risk_points <= 0.0
                or not math.isclose(
                    float(plan.risk_points),
                    abs(plan.planned_entry - plan.invalidation.price),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isfinite(float(plan.primary_target_R))
                or plan.primary_target_R <= 0.0
                or frozen_target_R is None
                or not math.isclose(
                    float(plan.primary_target_R),
                    frozen_target_R,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isfinite(float(plan.remaining_path_R))
                or plan.remaining_path_R <= 0.0
            ):
                reasons.append("frozen_plan_delivery_invalid")
            if (
                not plan.targets
                or plan.selected_draw_id is None
                or plan.targets[0].level_id != plan.selected_draw_id
            ):
                reasons.append("frozen_draw_target_invalid")
            if (
                plan.liquidity_route is not None
                and plan.liquidity_route.path_blocker_ids
            ):
                reasons.append("hard_barrier_before_target")
        if _graph_ambiguity_count(hypothesis) > 0:
            reasons.append("graph_ambiguity")
        if hypothesis.market_thesis_binding_required and not (
            hypothesis.market_thesis_action_bound
            and hypothesis.market_thesis_match_status == "exact_root_bound"
            and hypothesis.bound_market_thesis_id
            == hypothesis.market_thesis_id
        ):
            reasons.append("market_thesis_exact_root_unbound")
        required_root_id = getattr(hypothesis, "required_root_id", None)
        if (
            required_root_id is not None
            and required_root_id != hypothesis.market_thesis_root_id
        ):
            reasons.append("market_thesis_required_root_mismatch")
        return not reasons, tuple(reasons)

    def _delivery_probability(
        self,
        hypothesis: HypothesisBelief,
    ) -> tuple[bool, float, tuple[str, ...]]:
        """Return calibrated delivery probability and its availability.

        The observed value remains visible for audit even when a causal gate
        fails.  Only readiness/version/range errors make the probability
        unavailable; they never rewrite the observed value to zero.
        """

        reasons: list[str] = []
        raw = hypothesis.delivery_quality
        try:
            probability = float(raw)
        except (TypeError, ValueError):
            probability = 0.0
            reasons.append("delivery_probability_missing_or_invalid")
        else:
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                probability = 0.0
                reasons.append("delivery_probability_missing_or_invalid")
        if not self.calibration_ready:
            reasons.append("calibration_not_ready")
        elif hypothesis.calibration_version != self.calibration_version:
            reasons.append("calibration_version_mismatch")
        return not reasons, probability, tuple(reasons)

    def _entry_qualification(
        self,
        belief: MarketBelief,
        candidate_id: str,
        observation: MarketObservation,
        hypothesis: HypothesisBelief,
    ) -> tuple[bool, tuple[str, ...]]:
        """Compatibility composition of causal and probability authority."""

        causal_eligible, causal_reasons = self._causal_entry_gate(
            belief,
            candidate_id,
            observation,
            hypothesis,
        )
        probability_available, _, probability_reasons = (
            self._delivery_probability(hypothesis)
        )
        return (
            causal_eligible and probability_available,
            causal_reasons + probability_reasons,
        )

    def _executable_wait_utility(
        self,
        belief: MarketBelief,
        observation: MarketObservation,
        hypothesis: HypothesisBelief,
        candidate_id: str,
        *,
        uncertainty_cost: float,
        deadline_cost: float,
    ) -> ActionUtility | None:
        """Offer WAIT only for a concrete, still-valid price improvement."""

        causal_eligible, _ = self._causal_entry_gate(
            belief,
            candidate_id,
            observation,
            hypothesis,
        )
        probability_available, _, _ = self._delivery_probability(hypothesis)
        if not (causal_eligible and probability_available):
            return None
        plan = hypothesis.plan
        assert plan is not None
        if (
            plan.entry_zone_lower is None
            or plan.entry_zone_upper is None
            or not plan.entry_zone_lower
            <= observation.price
            <= plan.entry_zone_upper
        ):
            return None
        improvement_points = hypothesis.direction.sign * (
            observation.price - plan.planned_entry
        )
        if improvement_points <= 0.0:
            return None
        improvement_R = improvement_points / plan.risk_points
        if not math.isfinite(improvement_R) or improvement_R <= 0.0:
            return None
        option_value = min(self.config.maximum_reward_R, improvement_R)
        wait = option_value - 0.55 * uncertainty_cost - 0.65 * deadline_cost
        return ActionUtility(
            action=Action.WAIT,
            utility=float(wait),
            components={
                "frozen_entry_improvement_R": option_value,
                "uncertainty": -0.55 * uncertainty_cost,
                "deadline": -0.65 * deadline_cost,
            },
            hypothesis_key=candidate_id,
            reason=(
                "wait only for the still-valid frozen planned entry, which "
                f"improves location by {option_value:.3f}R; "
                f"{_evidence_reason(hypothesis)}"
            ),
        )

    def _flat_utilities(
        self,
        observation: MarketObservation,
        belief: MarketBelief,
    ) -> list[ActionUtility]:
        output: list[ActionUtility] = [
            ActionUtility(
                action=Action.ABSTAIN,
                utility=0.0,
                components={"capital_preservation": 0.0},
                hypothesis_key=None,
                reason="preserve optionality when no action has a clear net advantage",
            )
        ]
        deadline = _deadline_penalty(observation, self.config)
        for candidate_id, hypothesis in _action_candidate_items(belief):
            plan = hypothesis.plan
            typed = _is_typed(hypothesis)
            thesis = float(hypothesis.thesis_strength or 0.0)
            sequence = float(hypothesis.sequence_progress or 0.0)
            location = float(hypothesis.location_quality or 0.0)
            readiness = float(hypothesis.entry_readiness or 0.0)
            uncertainty = (
                self.config.uncertainty_penalty
                * hypothesis.uncertainty
            )
            ownership = getattr(
                belief,
                "owns_actionable_entry_episode",
                None,
            )
            if (
                typed
                and hypothesis.phase
                in {
                    PlaybookPhase.FORMING,
                    PlaybookPhase.ARMED,
                    PlaybookPhase.WAITING_LOCATION,
                    PlaybookPhase.WAITING_TRIGGER,
                }
                and callable(ownership)
                and ownership(candidate_id, hypothesis)
            ):
                wait_option = (
                    thesis
                    * sequence
                    * ((1.0 - location) + (1.0 - readiness))
                    / 2.0
                    * self.config.maximum_reward_R
                    * 0.42
                )
                wait = wait_option - 0.55 * uncertainty - 0.65 * deadline
                output.append(
                    ActionUtility(
                        action=Action.WAIT,
                        utility=float(wait),
                        components={
                            "thesis_strength": thesis,
                            "sequence_progress": sequence,
                            "location_gap": -(1.0 - location),
                            "trigger_gap": -(1.0 - readiness),
                            "option_value_R": wait_option,
                            "uncertainty": -0.55 * uncertainty,
                            "deadline": -0.65 * deadline,
                        },
                        hypothesis_key=candidate_id,
                        reason=(
                            "retain the causal setup while location or "
                            f"trigger is incomplete; {_evidence_reason(hypothesis)}"
                        ),
                    )
                )
            elif hypothesis.phase is PlaybookPhase.EXECUTABLE:
                executable_wait = self._executable_wait_utility(
                    belief,
                    observation,
                    hypothesis,
                    candidate_id,
                    uncertainty_cost=uncertainty,
                    deadline_cost=deadline,
                )
                if executable_wait is not None:
                    output.append(executable_wait)
            frozen_target_R = _frozen_primary_target_R(hypothesis)
            reward = 0.0 if frozen_target_R is None else frozen_target_R
            cost = (
                0.0
                if plan is None
                else _cost_R(observation, plan.risk_points)
            )
            causal_eligible, causal_reasons = self._causal_entry_gate(
                belief,
                candidate_id,
                observation,
                hypothesis,
            )
            (
                probability_available,
                delivery_probability,
                probability_reasons,
            ) = self._delivery_probability(hypothesis)
            action_authorized = causal_eligible and probability_available
            gross = 0.0
            enter = -1.0
            if action_authorized:
                gross = (
                    delivery_probability * reward
                    - (1.0 - delivery_probability)
                )
                enter = gross - cost
            ineligible_reasons = causal_reasons + probability_reasons
            output.append(
                ActionUtility(
                    action=Action.ENTER,
                    utility=float(enter),
                    components={
                        "expected_gross_R": gross,
                        "cost_R": -cost,
                        "uncertainty_observed": hypothesis.uncertainty,
                        "causal_eligible": float(causal_eligible),
                        "probability_available": float(probability_available),
                        "p_delivery": delivery_probability,
                        "action_authorized": float(action_authorized),
                    },
                    hypothesis_key=candidate_id,
                    reason=(
                        "calibrated delivery expected utility; "
                        + _evidence_reason(hypothesis)
                        if action_authorized
                        else "entry ineligible: "
                        + "; ".join(ineligible_reasons)
                        + "; "
                        + _evidence_reason(hypothesis)
                    ),
                )
            )
        return output

    def _position_utilities(
        self,
        observation: MarketObservation,
        belief: MarketBelief,
        account: AccountState,
    ) -> list[ActionUtility]:
        """Choose management verbs from frozen structure, never a pseudo-p.

        Once a position exists, entry delivery calibration is no longer a
        probability of the management action.  Frozen stop/deadline/target,
        exact episode identity and newly confirmed protection structure decide
        EXIT/PROTECT/HOLD directly.  Missing or ambiguous identity fails closed.
        """

        position = account.position
        if position is None:
            raise ValueError("position utilities require an open position")
        original_risk = abs(
            position.entry_price - position.original_invalidation.price
        )
        if original_risk <= 0:
            raise ValueError("open position has invalid original risk")
        sign = position.direction.sign
        mark_R = (
            sign * (observation.price - position.entry_price) / original_risk
        )
        cost = (
            observation.execution.expected_round_trip_cost_points
            / original_risk
        )
        priority = max(
            0.25,
            self.config.minimum_utility_advantage + 0.01,
        )

        identity_complete = all(
            isinstance(value, str) and bool(value)
            for value in (
                position.setup_id,
                position.entry_location_id,
                position.entry_path_id,
            )
        )
        exact_matches: list[tuple[str, HypothesisBelief]] = []
        if identity_complete:
            for candidate_id, candidate in _position_candidate_items(belief):
                plan = candidate.plan
                if (
                    candidate.playbook is position.playbook
                    and candidate.direction is position.direction
                    and candidate.setup_context_id == position.setup_id
                    and candidate.entry_location_id
                    == position.entry_location_id
                    and plan is not None
                    and plan.setup_id == position.setup_id
                    and plan.entry_location_id == position.entry_location_id
                    and plan.entry_path_id == position.entry_path_id
                ):
                    exact_matches.append((candidate_id, candidate))
        candidate_id: str | None
        hypothesis: HypothesisBelief | None
        if len(exact_matches) == 1:
            candidate_id, hypothesis = exact_matches[0]
        else:
            candidate_id, hypothesis = None, None

        identity_uncertain = bool(
            hypothesis is None
            or _graph_ambiguity_count(hypothesis) > 0
            or (
                hypothesis.market_thesis_binding_required
                and not (
                    hypothesis.market_thesis_action_bound
                    and hypothesis.market_thesis_match_status
                    == "exact_root_bound"
                    and hypothesis.bound_market_thesis_id
                    == hypothesis.market_thesis_id
                )
            )
        )
        stop_touched = sign * (
            observation.price - position.current_stop
        ) <= 0.0
        target_touched = sign * (
            position.primary_target.price - observation.price
        ) <= 0.0
        deadline_elapsed = observation.asof >= position.deadline
        terminal_phase = bool(
            hypothesis is not None
            and hypothesis.phase
            in {PlaybookPhase.INVALIDATED, PlaybookPhase.COMPLETED}
        )
        position_not_open = position.status != "open"
        hard_exit_reasons = tuple(
            reason
            for reason, present in (
                ("frozen_stop_touched", stop_touched),
                ("frozen_deadline_elapsed", deadline_elapsed),
                ("frozen_target_reached", target_touched),
                ("hypothesis_terminal", terminal_phase),
                ("position_not_open", position_not_open),
            )
            if present
        )

        def deterministic_pair(
            action: Action,
            *,
            action_utility: float,
            components: dict[str, float],
            reason: str,
        ) -> list[ActionUtility]:
            return [
                ActionUtility(
                    action,
                    float(action_utility),
                    components,
                    candidate_id,
                    reason,
                ),
                ActionUtility(
                    Action.ABSTAIN,
                    float(action_utility - priority),
                    {
                        "no_new_instruction_mark_R": mark_R,
                        "structural_action_priority_R": -priority,
                    },
                    candidate_id,
                    "do not leave a managed position without a clear "
                    "structural instruction",
                ),
            ]

        if hard_exit_reasons:
            return deterministic_pair(
                Action.EXIT,
                action_utility=mark_R - 0.5 * cost,
                components={
                    "locked_mark_R": mark_R,
                    "exit_cost_R": -0.5 * cost,
                    "structural_exit": 1.0,
                },
                reason="exit on " + ",".join(hard_exit_reasons),
            )
        if identity_uncertain:
            return deterministic_pair(
                Action.EXIT,
                action_utility=mark_R - 0.5 * cost,
                components={
                    "locked_mark_R": mark_R,
                    "exit_cost_R": -0.5 * cost,
                    "identity_fail_closed": 1.0,
                },
                reason=(
                    "exit because the frozen setup/location/path identity is "
                    "missing, non-unique, or unresolved"
                ),
            )

        assert hypothesis is not None and candidate_id is not None
        candidate = causal_protection_candidate(position, observation)
        if candidate is not None:
            tighter = (
                candidate.price > position.current_stop
                if position.direction is Direction.LONG
                else candidate.price < position.current_stop
            )
            if tighter:
                candidate_R = (
                    sign
                    * (candidate.price - position.entry_price)
                    / original_risk
                )
                return deterministic_pair(
                    Action.PROTECT,
                    action_utility=mark_R - 0.25 * cost,
                    components={
                        "mark_R": mark_R,
                        "protected_level_R": candidate_R,
                        "amendment_cost_R": -0.25 * cost,
                        "qualified_structural_protection": 1.0,
                    },
                    reason=(
                        "protect at newly confirmed causal level "
                        + candidate.source_level_id
                    ),
                )
        return deterministic_pair(
            Action.HOLD,
            action_utility=mark_R,
            components={
                "mark_R": mark_R,
                "frozen_thesis_valid": 1.0,
                "hard_exit_absent": 1.0,
            },
            reason=(
                "hold because the exact frozen thesis remains valid and no "
                "hard exit or qualified protection event occurred"
            ),
        )

    def decide(
        self,
        observation: MarketObservation,
        belief: MarketBelief,
        account: AccountState | None = None,
    ) -> Decision:
        account = account or AccountState(equity=100_000.0)
        utilities = (
            self._position_utilities(observation, belief, account)
            if account.position is not None
            else self._flat_utilities(observation, belief)
        )
        ranked = sorted(utilities, key=lambda item: item.utility, reverse=True)
        best = ranked[0]
        best_hypothesis = (
            None
            if best.hypothesis_key is None
            else belief.resolve_hypothesis(best.hypothesis_key)
        )
        best_plan = (
            None
            if best_hypothesis is None
            else best_hypothesis.plan
        )
        best_identity = _concrete_action_identity(best, belief)
        distinct_runner_up = next(
            (
                candidate
                for candidate in ranked[1:]
                if _concrete_action_identity(candidate, belief)
                != best_identity
            ),
            None,
        )
        # If there is no genuinely different instruction, no comparative
        # advantage was measured. Keep the fail-safe audit value finite.
        advantage = (
            float(best.utility - distinct_runner_up.utility)
            if distinct_runner_up is not None
            else 0.0
        )
        selected = best.action
        reasons = [best.reason]
        global_context = belief.global_context
        unexplained = (
            ()
            if global_context is None
            else global_context.unexplained_structured_episode_ids
        )
        if unexplained:
            diagnostic_ids = unexplained[:3]
            reasons.append(
                "unexplained_structured_episodes="
                f"count:{len(unexplained)};"
                " ids:"
                + ",".join(diagnostic_ids)
                + "; no ad-hoc playbook was created"
            )
        if belief.focus_state is not None:
            focus = belief.focus_state
            reasons.append(
                "focus="
                + ",".join(focus.primary_timeframes)
                + "; why="
                + ",".join(focus.reason_codes)
                + "; question="
                + focus.question
                + "; resolution="
                + focus.resolution_status.value
                + "; competing="
                + str(len(belief.competing_hypothesis_ids))
            )
        if best_plan is not None and best_plan.liquidity_route is not None:
            route = best_plan.liquidity_route
            reasons.append(
                "liquidity_route="
                f"context:{route.context_draw_id};"
                f"primary:{route.primary_deliverable_target_id};"
                f"terminal:{route.terminal_draw_id};"
                f"authority_barrier:{route.authority_barrier_id}@"
                f"{route.authority_barrier_price};"
                f"blockers:{','.join(route.path_blocker_ids) or 'none'}"
            )
        if selected is not Action.ABSTAIN and advantage < self.config.minimum_utility_advantage:
            selected = Action.ABSTAIN
            reasons.insert(
                0,
                f"best action advantage {advantage:.3f}R is below "
                f"{self.config.minimum_utility_advantage:.3f}R",
            )
        if best.action is Action.ENTER and best_plan is None:
            selected = Action.ABSTAIN
            reasons.insert(
                0,
                "enter requires a resolvable frozen execution plan",
            )
        if (
            account.position is None
            and any(
                name.startswith("warmup_")
                for name in observation.anomalies
            )
        ):
            selected = Action.ABSTAIN
            reasons.insert(0, "multitimeframe observer is still warming up")
        if selected is Action.ABSTAIN and account.position is None:
            qualification_blocks = []
            for candidate_id, hypothesis in _action_candidate_items(belief):
                if hypothesis.phase is not PlaybookPhase.EXECUTABLE:
                    continue
                eligible, blocked_by = self._entry_qualification(
                    belief,
                    candidate_id,
                    observation,
                    hypothesis,
                )
                if not eligible:
                    qualification_blocks.append(
                        candidate_id + "=" + ",".join(blocked_by)
                    )
            if qualification_blocks:
                reasons.append(
                    "executable entry qualification failed: "
                    + " | ".join(qualification_blocks)
                )
        return Decision(
            asof=observation.asof,
            selected_action=selected,
            utilities=tuple(ranked),
            best_hypothesis_key=best.hypothesis_key,
            advantage=advantage,
            reasons=tuple(reasons),
            plan=best_plan,
        )


__all__ = ["DecisionConfig", "UtilityDecisionLayer"]
