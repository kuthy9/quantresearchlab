"""Net-utility comparison across explicit trading actions."""
from __future__ import annotations

from dataclasses import dataclass

from .model import (
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


def _phase_penalty(phase: PlaybookPhase) -> float:
    return {
        PlaybookPhase.EXECUTABLE: 0.0,
        PlaybookPhase.ARMED: 0.30,
        PlaybookPhase.WAITING_LOCATION: 0.35,
        PlaybookPhase.WAITING_TRIGGER: 0.20,
        PlaybookPhase.WAITING_PULLBACK: 0.45,
        PlaybookPhase.FORMING: 0.90,
        PlaybookPhase.INACTIVE: 1.50,
        PlaybookPhase.INVALIDATED: 2.00,
        PlaybookPhase.COMPLETED: 2.00,
        PlaybookPhase.ENTERED: 2.00,
        PlaybookPhase.WEAKENING: 2.00,
        PlaybookPhase.DELIVERING: 2.00,
    }[phase]


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
    )


class UtilityDecisionLayer:
    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()

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
        fill_penalty = 1.0 - observation.execution.fillability
        for hypothesis in belief.candidates():
            plan = hypothesis.plan
            typed = _is_typed(hypothesis)
            if typed:
                thesis = float(hypothesis.thesis_strength)
                sequence = float(hypothesis.sequence_progress)
                location = float(hypothesis.location_quality)
                readiness = float(hypothesis.entry_readiness)
                delivery = float(hypothesis.delivery_quality)
                wait_option = (
                    thesis
                    * sequence
                    * (
                        (1.0 - location)
                        + (1.0 - readiness)
                    )
                    / 2.0
                    * self.config.maximum_reward_R
                    * 0.42
                )
                uncertainty = (
                    self.config.uncertainty_penalty
                    * hypothesis.uncertainty
                )
                wait = (
                    wait_option
                    - 0.55 * uncertainty
                    - 0.65 * deadline
                )
                if hypothesis.phase not in {
                    PlaybookPhase.FORMING,
                    PlaybookPhase.ARMED,
                    PlaybookPhase.WAITING_LOCATION,
                    PlaybookPhase.WAITING_TRIGGER,
                    PlaybookPhase.EXECUTABLE,
                }:
                    wait = min(wait, -1.0)
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
                        hypothesis_key=hypothesis.key,
                        reason=(
                            "retain the causal setup while location or "
                            f"trigger is incomplete; {_evidence_reason(hypothesis)}"
                        ),
                    )
                )
            if plan is None:
                continue
            reward = min(self.config.maximum_reward_R, max(0.0, plan.primary_target_R))
            cost = _cost_R(observation, plan.risk_points)
            uncertainty = self.config.uncertainty_penalty * hypothesis.uncertainty
            effective_probability = (
                min(
                    float(hypothesis.thesis_strength),
                    float(hypothesis.sequence_progress),
                    float(hypothesis.location_quality),
                    float(hypothesis.entry_readiness),
                    float(hypothesis.delivery_quality),
                )
                if typed
                else hypothesis.probability
            )
            gross = (
                effective_probability * reward
                - (1.0 - effective_probability)
            )
            phase = _phase_penalty(hypothesis.phase)
            enter = gross - cost - uncertainty - deadline - 0.35 * fill_penalty - phase
            gates_pass = bool(
                not typed
                or (
                    hypothesis.hard_gate_results
                    and all(hypothesis.hard_gate_results.values())
                )
            )
            if (
                hypothesis.phase is not PlaybookPhase.EXECUTABLE
                or not gates_pass
            ):
                enter = min(enter, -1.0)
            output.append(
                ActionUtility(
                    action=Action.ENTER,
                    utility=float(enter),
                    components={
                        "expected_gross_R": gross,
                        "cost_R": -cost,
                        "uncertainty": -uncertainty,
                        "deadline": -deadline,
                        "fillability": -0.35 * fill_penalty,
                        "phase_readiness": -phase,
                        "effective_readiness": effective_probability,
                        "hard_gates_pass": float(gates_pass),
                    },
                    hypothesis_key=hypothesis.key,
                    reason=_evidence_reason(hypothesis),
                )
            )
            if not typed:
                wait_option = (
                    max(0.0, hypothesis.probability - 0.35)
                    * min(
                        self.config.maximum_reward_R,
                        max(0.0, plan.remaining_path_R),
                    )
                    * 0.42
                )
                wait = (
                    wait_option
                    - 0.55 * uncertainty
                    - 0.65 * deadline
                    - (
                        0.05
                        if hypothesis.phase
                        is PlaybookPhase.WAITING_PULLBACK
                        else 0.15
                    )
                )
                if hypothesis.phase not in {
                    PlaybookPhase.FORMING,
                    PlaybookPhase.ARMED,
                    PlaybookPhase.WAITING_PULLBACK,
                    PlaybookPhase.EXECUTABLE,
                }:
                    wait = min(wait, -1.0)
                output.append(
                    ActionUtility(
                        action=Action.WAIT,
                        utility=float(wait),
                        components={
                            "option_value_R": wait_option,
                            "uncertainty": -0.55 * uncertainty,
                            "deadline": -0.65 * deadline,
                        },
                        hypothesis_key=hypothesis.key,
                        reason=(
                            "retain the setup without paying entry cost; "
                            f"{_evidence_reason(hypothesis)}"
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
        position = account.position
        if position is None:
            raise ValueError("position utilities require an open position")
        key = f"{position.playbook.value}:{position.direction.value}"
        hypothesis = belief.resolve_hypothesis(key)
        setup_matches = bool(
            hypothesis is not None
            and (
                position.setup_id is None
                or (
                    hypothesis.setup_context_id == position.setup_id
                    and hypothesis.entry_location_id
                    == position.entry_location_id
                    and hypothesis.plan is not None
                    and hypothesis.plan.entry_path_id
                    == position.entry_path_id
                )
            )
        )
        probability = (
            0.5
            if hypothesis is None
            else (
                min(
                    float(hypothesis.thesis_strength),
                    float(hypothesis.delivery_quality),
                )
                if setup_matches and _is_typed(hypothesis)
                else hypothesis.probability
                if setup_matches
                else 0.0
            )
        )
        uncertainty = (
            hypothesis.uncertainty
            if setup_matches and hypothesis is not None
            else 1.0
        )
        original_risk = abs(position.entry_price - position.original_invalidation.price)
        if original_risk <= 0:
            raise ValueError("open position has invalid original risk")
        sign = position.direction.sign
        mark_R = sign * (observation.price - position.entry_price) / original_risk
        target_R = sign * (position.primary_target.price - position.entry_price) / original_risk
        remaining = max(0.0, target_R - mark_R)
        stop_R = sign * (position.current_stop - position.entry_price) / original_risk
        giveback = max(0.0, mark_R - stop_R)
        cost = observation.execution.expected_round_trip_cost_points / original_risk
        deadline = _deadline_penalty(observation, self.config)
        uncertainty_cost = self.config.uncertainty_penalty * uncertainty

        hold = (
            mark_R
            + probability * remaining
            - (1.0 - probability) * giveback
            - uncertainty_cost
            - deadline
        )
        exit_now = mark_R - 0.5 * cost
        utilities = [
            ActionUtility(
                Action.HOLD,
                float(hold),
                {
                    "mark_R": mark_R,
                    "delivery_option_R": probability * remaining,
                    "giveback_risk_R": -(1.0 - probability) * giveback,
                    "uncertainty": -uncertainty_cost,
                    "deadline": -deadline,
                },
                key,
                "keep the frozen thesis unchanged while expected delivery exceeds giveback",
            ),
            ActionUtility(
                Action.EXIT,
                float(exit_now),
                {"locked_mark_R": mark_R, "exit_cost_R": -0.5 * cost},
                key,
                "close at the current observable mark and stop thesis exposure",
            ),
            ActionUtility(
                Action.ABSTAIN,
                float(mark_R),
                {"no_new_instruction_mark_R": mark_R},
                key,
                "issue no management instruction while utilities are ambiguous",
            ),
        ]
        candidate = causal_protection_candidate(position, observation)
        if candidate is not None:
            candidate_R = sign * (candidate.price - position.entry_price) / original_risk
            tighter = (
                candidate.price > position.current_stop
                if position.direction is Direction.LONG
                else candidate.price < position.current_stop
            )
            reduced_giveback = max(0.0, mark_R - candidate_R)
            protect = (
                mark_R
                + probability * remaining
                - (1.0 - probability) * reduced_giveback
                - uncertainty_cost
                - 0.25 * cost
                - (0.0 if tighter else 2.0)
            )
            utilities.append(
                ActionUtility(
                    Action.PROTECT,
                    float(protect),
                    {
                        "mark_R": mark_R,
                        "delivery_option_R": probability * remaining,
                        "protected_giveback_R": -(1.0 - probability) * reduced_giveback,
                        "amendment_cost_R": -0.25 * cost,
                        "structurally_tighter": 1.0 if tighter else -2.0,
                    },
                    key,
                    "tighten only to a newly confirmed causal structural level",
                )
            )
        return utilities

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
        if any(name.startswith("warmup_") for name in observation.anomalies):
            selected = Action.ABSTAIN
            reasons.insert(0, "multitimeframe observer is still warming up")
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
