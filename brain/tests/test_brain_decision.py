from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from brain.core.decision import (
    DecisionConfig,
    UtilityDecisionLayer,
    _concrete_action_identity,
)
from shares.core.engine import RuntimeActionBeliefView
from shares.core.model import (
    AccountState,
    Action,
    ActionUtility,
    Direction,
    DirectionalObstructionView,
    FrozenTriggerState,
    GlobalMarketContext,
    HypothesisSequenceState,
    MarketBelief as ProductionMarketBelief,
    MarketMode,
    OpenMarketThesis,
    Playbook,
    PlaybookPhase,
    PositionSnapshot,
    ScaleRelation,
    ScaleRelationState,
    SequenceStepState,
    Timeframe,
)
from brain.core.playbooks import BrainConfig, PlaybookBrain
from brain.core.risk import StructuralRiskEngine

from shares.tests.helpers import (
    GraphFreeActionBelief as MarketBelief,
    executable_belief,
    flat_account,
    long_plan,
    market_observation,
    replace_market_observation,
)


class _ScriptedUtilityDecisionLayer(UtilityDecisionLayer):
    def __init__(
        self,
        utilities: tuple[ActionUtility, ...],
        config: DecisionConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._scripted_utilities = utilities

    def _flat_utilities(self, observation, belief):
        return list(self._scripted_utilities)

    def _position_utilities(self, observation, belief, account):
        return list(self._scripted_utilities)


def _utility(
    action: Action,
    value: float,
    key: str | None,
    reason: str | None = None,
) -> ActionUtility:
    return ActionUtility(
        action,
        value,
        {"scripted": value},
        key,
        reason or f"{action.value}:{key}",
    )


def _empty_belief(observation) -> MarketBelief:
    return MarketBelief(observation.asof, {})


def _ready_executable_belief(
    observation,
    *,
    probability: float = 0.85,
    delivery: float = 0.85,
    calibration_version: str = "typed-test-ready",
) -> MarketBelief:
    base = executable_belief(
        observation,
        probability=probability,
        uncertainty=0.1,
    )
    hypothesis = next(iter(base.hypotheses.values()))
    assert hypothesis.plan is not None
    plan = replace(
        hypothesis.plan,
        setup_id="setup:test-ready",
        entry_location_id="location:test-ready",
        entry_path_id="path:test-ready",
        entry_zone_lower=observation.price - 1.0,
        entry_zone_upper=observation.price + 1.0,
        selected_draw_id=hypothesis.plan.targets[0].level_id,
    )
    sequence = HypothesisSequenceState(
        protocol_version="typed-test",
        protocol_hash="typed-test-hash",
        setup_id=plan.setup_id,
        steps=(
            SequenceStepState(
                step_id="synthetic_ready",
                satisfied=True,
                value=1.0,
                observed_at=observation.asof,
                source_ids=(plan.entry_path_id,),
            ),
        ),
        started_at=observation.asof,
    )
    trigger = FrozenTriggerState(
        trigger_id="trigger:test-ready",
        trigger_kind="micro_bos_confirmed",
        observed_at=observation.asof,
        setup_id=plan.setup_id,
        entry_path_id=plan.entry_path_id,
        entry_location_id=plan.entry_location_id,
        direction=plan.direction,
        source_entity_id="micro-bos:test-ready",
        source_event_id="bos:test-ready",
        strength=1.0,
        available_trigger_kinds=("micro_bos_confirmed",),
    )
    updated = replace(
        hypothesis,
        plan=plan,
        sequence=sequence,
        invalidation=plan.invalidation,
        deliverable_targets=plan.targets,
        remaining_path_R=plan.remaining_path_R,
        delivery_quality=delivery,
        calibration_version=calibration_version,
        setup_context_id=plan.setup_id,
        entry_location_id=plan.entry_location_id,
        selected_trigger=trigger,
        context_id="context:test-ready",
        context_thesis_id="context-thesis:test-ready",
        parent_context_thesis_id="context-thesis:test-ready",
        episode_id=plan.setup_id,
        thesis_deadline=plan.deadline,
        episode_deadline=plan.deadline,
    )
    return MarketBelief(observation.asof, {updated.key: updated})


def _ready_layer(
    config: DecisionConfig | None = None,
    *,
    version: str = "typed-test-ready",
) -> UtilityDecisionLayer:
    return UtilityDecisionLayer(
        config,
        calibration_ready=True,
        calibration_version=version,
    )


def _unexplained_context(
    observation,
    *,
    episode_ids: tuple[str, ...] = ("episode:unexplained",),
) -> GlobalMarketContext:
    return GlobalMarketContext(
        updated_at=observation.asof,
        scene_revision_id="scene:r000000000001",
        market_epoch_id="epoch:0",
        authority_stack=(),
        market_mode=MarketMode.UNCERTAIN,
        scale_relation_details={
            timeframe.value: ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.UNKNOWN,
                direction=None,
                authority_layer_id=None,
                evidence_ids=(),
                evidence_kind=None,
                structural_scope=None,
                acceptance_state=None,
                since=None,
                age_bars=0,
                graph_connected=False,
                ambiguous=False,
            )
            for timeframe in Timeframe
        },
        external_draw_candidates={"above": (), "below": ()},
        obstruction_views={
            direction.value: DirectionalObstructionView(
                direction=direction,
                nearest_draw_id=None,
                nearest_draw_price=None,
                hard_barriers=(),
                soft_frictions=(),
            )
            for direction in Direction
        },
        material_conflicts=(),
        unknown_evidence=(),
        ambiguous_evidence=(),
        dislocations_by_scale={
            timeframe.value: () for timeframe in Timeframe
        },
        unexplained_structured_episode_ids=episode_ids,
    )


def _root_candidate_belief_with_unexplained(
    observation,
    *,
    phase: PlaybookPhase = PlaybookPhase.EXECUTABLE,
    episode_ids: tuple[str, ...] = ("episode:unexplained",),
) -> MarketBelief:
    base = _ready_executable_belief(
        observation,
        probability=0.95,
        delivery=0.95,
    )
    summary = next(iter(base.hypotheses.values()))
    assert summary.plan is not None
    root_id = summary.plan.setup_id
    assert root_id is not None
    thesis_id = "market-thesis:decision-authorized-root"
    thesis = OpenMarketThesis(
        thesis_id=thesis_id,
        root_id=root_id,
        market_epoch_id="epoch:0",
        formed_at=observation.asof,
        updated_at=observation.asof,
        direction=summary.direction,
        source_timeframe=Timeframe.M5,
        structural_scale="intermediate",
        mechanism="directional_displacement",
        authority_relation="aligned",
        mechanism_event_ids=(root_id,),
        draw_candidate_ids=(summary.plan.targets[0].level_id,),
        entry_location_ids=(summary.plan.entry_location_id,),
        trigger_event_ids=(summary.selected_trigger.trigger_id,),
    )
    context = replace(
        _unexplained_context(
            observation,
            episode_ids=episode_ids,
        ),
        open_market_theses=(thesis,),
    )
    candidate_id = (
        f"{thesis_id}|{summary.playbook.value}|{summary.direction.value}"
    )
    candidate = replace(
        summary,
        phase=phase,
        phase_started_at=observation.asof,
        market_thesis_ids=(thesis_id,),
        market_thesis_id=thesis_id,
        bound_market_thesis_id=thesis_id,
        market_thesis_root_id=root_id,
        market_thesis_mechanism=thesis.mechanism,
        market_thesis_authority_relation=thesis.authority_relation,
        playbook_match_strength=1.0,
        market_thesis_binding_required=True,
        market_thesis_action_bound=True,
        market_thesis_match_status="exact_root_bound",
        candidate_id=candidate_id,
        required_root_id=root_id,
        record_kind="root_candidate",
        context_id=None,
        context_thesis_id=None,
        parent_context_thesis_id=None,
        episode_id=None,
        thesis_deadline=None,
        episode_deadline=None,
    )
    return MarketBelief(
        observation.asof,
        base.hypotheses,
        global_context=context,
        thesis_candidates={candidate_id: candidate},
    )


def _two_plan_belief(observation, *, second_plan=None) -> MarketBelief:
    first = next(
        iter(
            executable_belief(
                observation,
                probability=0.95,
                uncertainty=0.02,
            ).hypotheses.values()
        )
    )
    plan = first.plan
    assert plan is not None
    second_plan = second_plan or replace(
        plan,
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
    )
    second = replace(
        first,
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        probability=0.93,
        plan=second_plan,
        invalidation=second_plan.invalidation,
        deliverable_targets=second_plan.targets,
        remaining_path_R=second_plan.remaining_path_R,
    )
    return MarketBelief(
        observation.asof,
        {first.key: first, second.key: second},
    )


def test_runtime_action_view_blocks_only_new_entries_for_disabled_playbook() -> None:
    dfp_id = "candidate:dfp"
    lsr_id = "candidate:lsr"
    dfp = SimpleNamespace(playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK)
    lsr = SimpleNamespace(playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL)
    raw_items = ((dfp_id, dfp), (lsr_id, lsr))
    lifecycle_items = (*raw_items, ("retained:lsr", lsr))
    position_items = ((lsr_id, lsr),)
    by_id = dict((*lifecycle_items,))
    belief = SimpleNamespace(
        action_candidate_items=lambda: raw_items,
        lifecycle_candidate_items=lambda: lifecycle_items,
        position_candidate_items=lambda: position_items,
        owns_actionable_entry_episode=lambda candidate_id, hypothesis: (
            by_id.get(candidate_id) is hypothesis
        ),
        resolve_hypothesis=lambda identity: by_id.get(identity),
    )
    view = RuntimeActionBeliefView(
        belief,
        (Playbook.LIQUIDITY_SWEEP_REVERSAL,),
    )

    assert view.action_candidate_items() == ((dfp_id, dfp),)
    assert belief.action_candidate_items() == raw_items
    assert view.owns_actionable_entry_episode(lsr_id, lsr) is False
    assert view.owns_actionable_entry_episode(dfp_id, dfp) == (
        belief.owns_actionable_entry_episode(dfp_id, dfp)
    )
    assert view.lifecycle_candidate_items() == lifecycle_items
    assert view.position_candidate_items() == position_items
    assert view.position_candidate_items()
    assert view.resolve_hypothesis(lsr_id) is lsr


def _open_account(observation) -> AccountState:
    plan = long_plan(observation)
    position = PositionSnapshot(
        thesis_hash="a" * 64,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        playbook=plan.playbook,
        direction=plan.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=observation.asof - pd.Timedelta(minutes=1),
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=1,
    )
    return replace(flat_account(), position=position)


def _typed_open_account(
    observation,
    belief: MarketBelief,
    *,
    deadline=None,
) -> AccountState:
    hypothesis = next(iter(belief.candidates()))
    plan = hypothesis.plan
    assert plan is not None
    position = PositionSnapshot(
        thesis_hash="b" * 64,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        playbook=plan.playbook,
        direction=plan.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=observation.asof - pd.Timedelta(minutes=1),
        deadline=plan.deadline if deadline is None else deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=1,
        setup_id=plan.setup_id,
        entry_location_id=plan.entry_location_id,
        entry_path_id=plan.entry_path_id,
    )
    return replace(flat_account(), position=position)


def test_brain_maintains_only_three_playbooks_in_both_directions() -> None:
    observation = market_observation()
    brain = PlaybookBrain(BrainConfig())
    belief = brain.update(observation)
    assert len(belief.hypotheses) == 6
    assert {item.playbook for item in belief.hypotheses.values()} == set(Playbook)
    assert all(
        isinstance(item.phase, PlaybookPhase)
        for item in belief.hypotheses.values()
    )
    assert all(0.0 <= item.probability <= 1.0 for item in belief.hypotheses.values())


def test_clear_executable_hypothesis_can_win_enter_utility() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(
        observation,
        probability=0.95,
        delivery=0.95,
    )
    decision = _ready_layer(
        DecisionConfig(minimum_utility_advantage=0.01),
    ).decide(observation, belief, flat_account())
    assert decision.selected_action is Action.ENTER
    assert decision.plan is not None


def test_identity_unvalidated_executable_hypothesis_cannot_enter() -> None:
    observation = market_observation()
    summary_belief = executable_belief(
        observation,
        probability=0.95,
        uncertainty=0.02,
    )
    belief = MarketBelief(
        observation.asof,
        summary_belief.hypotheses,
    )

    decision = UtilityDecisionLayer().decide(
        observation,
        belief,
        flat_account(),
    )

    assert decision.selected_action is Action.ABSTAIN
    enter = next(item for item in decision.utilities if item.action is Action.ENTER)
    assert enter.components["action_authorized"] == 0.0
    assert enter.components["p_delivery"] == pytest.approx(1.0)
    assert "calibration_not_ready" in enter.reason
    assert any("calibration_not_ready" in reason for reason in decision.reasons)


def test_production_graph_free_summary_belief_has_no_action_authority() -> None:
    observation = market_observation()
    summary_belief = executable_belief(
        observation,
        probability=0.99,
        uncertainty=0.0,
    )
    belief = ProductionMarketBelief(
        observation.asof,
        summary_belief.hypotheses,
    )

    decision = _ready_layer(
        DecisionConfig(minimum_utility_advantage=0.0),
    ).decide(observation, belief, flat_account())

    assert belief.action_candidate_items() == ()
    assert decision.selected_action is Action.ABSTAIN
    assert decision.best_hypothesis_key is None
    assert all(item.action is Action.ABSTAIN for item in decision.utilities)


def test_unready_or_version_mismatched_executable_cannot_wait_for_price() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)
    hypothesis = next(iter(belief.hypotheses.values()))
    assert hypothesis.plan is not None
    planned_entry = observation.price - 0.25
    risk = planned_entry - hypothesis.plan.invalidation.price
    plan = replace(
        hypothesis.plan,
        planned_entry=planned_entry,
        risk_points=risk,
        primary_target_R=(hypothesis.plan.targets[0].price - planned_entry) / risk,
        remaining_path_R=(hypothesis.plan.targets[0].price - observation.price) / risk,
    )
    cases = (
        (
            UtilityDecisionLayer(),
            "identity-unvalidated",
            "calibration_not_ready",
        ),
        (
            _ready_layer(version="other-version"),
            "typed-test-ready",
            "calibration_version_mismatch",
        ),
    )
    for layer, hypothesis_version, expected_reason in cases:
        candidate = replace(
            hypothesis,
            plan=plan,
            calibration_version=hypothesis_version,
        )
        belief = MarketBelief(
            observation.asof,
            {candidate.key: candidate},
        )

        decision = layer.decide(observation, belief, flat_account())

        assert decision.selected_action is Action.ABSTAIN
        assert all(
            item.action is not Action.WAIT for item in decision.utilities
        )
        assert any(
            expected_reason in reason for reason in decision.reasons
        )


def test_entry_uses_calibrated_delivery_probability_not_five_dimension_minimum() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation, delivery=0.60)
    hypothesis = next(iter(belief.hypotheses.values()))
    assert hypothesis.plan is not None
    farther_target = replace(
        hypothesis.plan.targets[0],
        price=hypothesis.plan.planned_entry + 5.0 * hypothesis.plan.risk_points,
    )
    plan = replace(
        hypothesis.plan,
        targets=(farther_target,),
        primary_target_R=5.0,
        remaining_path_R=5.0,
    )
    hypothesis = replace(
        hypothesis,
        plan=plan,
        deliverable_targets=plan.targets,
        remaining_path_R=plan.remaining_path_R,
        thesis_strength=0.31,
        sequence_progress=0.42,
        location_quality=0.53,
        entry_readiness=0.64,
    )
    belief = MarketBelief(observation.asof, {hypothesis.key: hypothesis})

    decision = _ready_layer().decide(observation, belief, flat_account())
    enter = next(item for item in decision.utilities if item.action is Action.ENTER)
    expected_cost_R = (
        observation.execution.expected_round_trip_cost_points
        / hypothesis.plan.risk_points
    )
    expected_gross_R = 0.60 * 5.0 - 0.40

    assert enter.components["p_delivery"] == pytest.approx(0.60)
    assert enter.components["causal_eligible"] == 1.0
    assert enter.components["probability_available"] == 1.0
    assert enter.components["action_authorized"] == 1.0
    assert enter.components["expected_gross_R"] == pytest.approx(expected_gross_R)
    assert enter.utility == pytest.approx(expected_gross_R - expected_cost_R)


def test_failed_causal_gate_keeps_delivery_probability_visible_but_never_enters() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation, delivery=0.99)
    hypothesis = next(iter(belief.candidates()))
    blocked = replace(
        hypothesis,
        hard_gate_results={"frozen_structure_intact": False},
    )
    blocked_belief = MarketBelief(observation.asof, {blocked.key: blocked})

    decision = _ready_layer(
        DecisionConfig(minimum_utility_advantage=0.0),
    ).decide(observation, blocked_belief, flat_account())
    enter = next(item for item in decision.utilities if item.action is Action.ENTER)

    assert decision.selected_action is not Action.ENTER
    assert enter.components["causal_eligible"] == 0.0
    assert enter.components["probability_available"] == 1.0
    assert enter.components["p_delivery"] == pytest.approx(0.99)
    assert enter.components["action_authorized"] == 0.0


def test_action_utilities_retain_root_specific_candidate_identity() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)
    hypothesis = next(iter(belief.candidates()))
    candidate_id = "thesis-candidate:root-specific-test"
    root_specific_view = SimpleNamespace(
        action_candidate_items=lambda: ((candidate_id, hypothesis),),
    )

    utilities = _ready_layer()._flat_utilities(
        observation,
        root_specific_view,
    )

    assert {
        item.hypothesis_key
        for item in utilities
        if item.action is Action.ENTER
    } == {candidate_id}


def test_entry_eu_ignores_non_delivery_dimensions_once_causal_gates_pass() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation, delivery=0.72)
    hypothesis = next(iter(belief.candidates()))

    base = next(
        item
        for item in _ready_layer()._flat_utilities(observation, belief)
        if item.action is Action.ENTER
    )
    changed = replace(
        hypothesis,
        thesis_strength=0.01,
        sequence_progress=0.02,
        location_quality=0.03,
        entry_readiness=0.04,
    )
    changed_belief = MarketBelief(observation.asof, {changed.key: changed})
    changed_enter = next(
        item
        for item in _ready_layer()._flat_utilities(
            observation,
            changed_belief,
        )
        if item.action is Action.ENTER
    )

    assert changed_enter.components["causal_eligible"] == 1.0
    assert changed_enter.components["p_delivery"] == pytest.approx(0.72)
    assert changed_enter.utility == pytest.approx(base.utility)


def test_calibration_version_mismatch_and_graph_ambiguity_fail_closed() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)

    mismatch = _ready_layer(version="other-version").decide(
        observation,
        belief,
        flat_account(),
    )
    mismatch_enter = next(
        item for item in mismatch.utilities if item.action is Action.ENTER
    )
    assert mismatch_enter.components["causal_eligible"] == 1.0
    assert mismatch_enter.components["probability_available"] == 0.0
    assert mismatch_enter.components["action_authorized"] == 0.0
    assert "calibration_version_mismatch" in mismatch_enter.reason
    assert any(
        "calibration_version_mismatch" in reason
        for reason in mismatch.reasons
    )

    hypothesis = next(iter(belief.hypotheses.values()))
    ambiguous = replace(
        hypothesis,
        context_metadata={"ambiguity_count": "1"},
    )
    ambiguous_belief = MarketBelief(
        observation.asof,
        {ambiguous.key: ambiguous},
    )
    decision = _ready_layer().decide(
        observation,
        ambiguous_belief,
        flat_account(),
    )
    assert decision.selected_action is Action.ABSTAIN
    enter = next(item for item in decision.utilities if item.action is Action.ENTER)
    assert "graph_ambiguity" in enter.reason
    assert any("graph_ambiguity" in reason for reason in decision.reasons)


def test_entry_qualification_keeps_current_gates_plan_draw_and_delivery_fail_closed() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)
    hypothesis = next(iter(belief.hypotheses.values()))
    assert hypothesis.plan is not None
    variants = (
        (
            replace(
                hypothesis,
                hard_gate_results={"synthetic_current_gate": False},
            ),
            "current_hard_gates_failed",
        ),
        (
            replace(
                hypothesis,
                plan=long_plan(observation),
                selected_trigger=None,
            ),
            "frozen_draw_target_invalid",
        ),
        (
            replace(
                hypothesis,
                market_thesis_ids=("thesis:test",),
                market_thesis_id="thesis:test",
                market_thesis_root_id="root:test",
                market_thesis_mechanism="continuation",
                market_thesis_authority_relation="aligned",
                playbook_match_strength=0.8,
                market_thesis_binding_required=True,
                market_thesis_action_bound=False,
                market_thesis_match_status="root_identity_unbound",
            ),
            "market_thesis_exact_root_unbound",
        ),
    )

    for candidate, expected_reason in variants:
        candidate_belief = MarketBelief(
            observation.asof,
            {candidate.key: candidate},
        )
        decision = _ready_layer().decide(
            observation,
            candidate_belief,
            flat_account(),
        )
        enter = next(
            item for item in decision.utilities if item.action is Action.ENTER
        )
        assert decision.selected_action is not Action.ENTER
        assert enter.components["action_authorized"] == 0.0
        assert expected_reason in enter.reason


@pytest.mark.parametrize(
    "candidate_factory",
    (
        # Keep the plan itself model-valid and make the owning hypothesis
        # disagree instead.  A typed LSR plan without FrozenLSRContext is now
        # rejected at construction time, before Decision can inspect it.
        lambda hypothesis: replace(
            hypothesis,
            playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        ),
        lambda hypothesis: replace(
            hypothesis,
            direction=Direction.SHORT,
            selected_trigger=None,
        ),
        lambda hypothesis: replace(
            hypothesis,
            plan=replace(hypothesis.plan, setup_id="setup:other"),
            selected_trigger=None,
        ),
        lambda hypothesis: replace(
            hypothesis,
            plan=replace(
                hypothesis.plan,
                entry_location_id="location:other",
            ),
            selected_trigger=None,
        ),
    ),
)
def test_entry_plan_must_match_hypothesis_and_sequence_identity(
    candidate_factory,
) -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)
    hypothesis = next(iter(belief.hypotheses.values()))
    candidate = candidate_factory(hypothesis)
    candidate_belief = MarketBelief(
        observation.asof,
        {candidate.key: candidate},
    )

    decision = _ready_layer().decide(
        observation,
        candidate_belief,
        flat_account(),
    )
    enter = next(
        item for item in decision.utilities if item.action is Action.ENTER
    )

    assert decision.selected_action is Action.ABSTAIN
    assert enter.components["causal_eligible"] == 0.0
    assert enter.components["probability_available"] == 1.0
    assert enter.components["action_authorized"] == 0.0
    assert "frozen_plan_hypothesis_identity_mismatch" in enter.reason


def test_entry_plan_path_must_match_explicit_episode_path() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)
    hypothesis = next(iter(belief.hypotheses.values()))

    with pytest.raises(
        ValueError,
        match="belief and trade plan entry paths disagree",
    ):
        replace(
            hypothesis,
            plan=replace(hypothesis.plan, entry_path_id="path:other"),
            selected_trigger=None,
        )


def test_entry_eu_does_not_mix_market_uncertainty_or_execution_veto_fields() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation, delivery=0.75)
    hypothesis = next(iter(belief.hypotheses.values()))
    high_uncertainty = replace(hypothesis, uncertainty=0.95)
    high_uncertainty_belief = MarketBelief(
        observation.asof,
        {high_uncertainty.key: high_uncertainty},
    )
    execution_limited = replace(
        observation,
        execution=replace(
            observation.execution,
            fillability=0.01,
            minutes_to_deadline=1,
        ),
    )

    base_enter = next(
        item
        for item in _ready_layer()._flat_utilities(observation, belief)
        if item.action is Action.ENTER
    )
    limited_enter = next(
        item
        for item in _ready_layer()._flat_utilities(
            execution_limited,
            high_uncertainty_belief,
        )
        if item.action is Action.ENTER
    )

    assert limited_enter.utility == pytest.approx(base_enter.utility)
    assert limited_enter.components["uncertainty_observed"] == pytest.approx(0.95)
    assert "deadline" not in limited_enter.components
    assert "fillability" not in limited_enter.components


def test_executable_wait_requires_price_inside_zone_and_better_frozen_entry() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)
    hypothesis = next(iter(belief.hypotheses.values()))
    assert hypothesis.plan is not None
    risk = observation.price - 0.25 - hypothesis.plan.invalidation.price
    plan = replace(
        hypothesis.plan,
        planned_entry=observation.price - 0.25,
        risk_points=risk,
        primary_target_R=(hypothesis.plan.targets[0].price - (observation.price - 0.25)) / risk,
        remaining_path_R=(hypothesis.plan.targets[0].price - (observation.price - 0.25)) / risk,
    )
    hypothesis = replace(hypothesis, plan=plan)
    belief = MarketBelief(observation.asof, {hypothesis.key: hypothesis})

    utilities = _ready_layer()._flat_utilities(observation, belief)
    assert any(item.action is Action.WAIT for item in utilities)

    outside = replace_market_observation(
        observation,
        price=plan.entry_zone_upper + 0.25,
    )
    outside_utilities = _ready_layer()._flat_utilities(outside, belief)
    assert all(item.action is not Action.WAIT for item in outside_utilities)


def test_ambiguous_action_advantage_forces_abstain() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(
        observation,
        probability=0.95,
        delivery=0.95,
    )
    decision = _ready_layer(
        DecisionConfig(minimum_utility_advantage=100.0),
    ).decide(observation, belief, flat_account())
    assert decision.selected_action is Action.ABSTAIN
    assert "below" in decision.reasons[0]


def test_wait_variants_share_one_concrete_action_but_raw_rows_remain() -> None:
    observation = market_observation()
    belief = _two_plan_belief(observation)
    first_key, second_key = belief.hypotheses
    utilities = (
        _utility(Action.WAIT, 0.40, first_key, "best wait"),
        _utility(Action.WAIT, 0.35, second_key, "other wait"),
        _utility(Action.ABSTAIN, 0.05, None),
    )
    decision = _ScriptedUtilityDecisionLayer(utilities).decide(
        observation,
        belief,
        flat_account(),
    )

    assert decision.selected_action is Action.WAIT
    assert decision.advantage == pytest.approx(0.35)
    assert decision.best_hypothesis_key == first_key
    assert decision.reasons == ("best wait",)
    assert decision.plan is belief.hypotheses[first_key].plan
    assert decision.utilities == utilities
    assert [item.action for item in decision.utilities].count(Action.WAIT) == 2


def test_wait_margin_below_frozen_threshold_still_abstains() -> None:
    observation = market_observation()
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.WAIT, 0.11, "dfp:long"),
            _utility(Action.WAIT, 0.10, "lsr:long"),
            _utility(Action.ABSTAIN, 0.0, None),
        )
    ).decide(observation, _empty_belief(observation), flat_account())

    assert decision.advantage == pytest.approx(0.11)
    assert decision.selected_action is Action.ABSTAIN
    assert "0.120R" in decision.reasons[0]


def test_wait_and_enter_remain_distinct_margin_competitors() -> None:
    observation = market_observation()
    belief = executable_belief(observation)
    key = next(iter(belief.hypotheses))
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.WAIT, 0.50, key),
            _utility(Action.ENTER, 0.45, key),
            _utility(Action.ABSTAIN, 0.0, None),
        )
    ).decide(observation, belief, flat_account())

    assert decision.advantage == pytest.approx(0.05)
    assert decision.selected_action is Action.ABSTAIN


def test_identical_enter_plans_do_not_compete_with_each_other() -> None:
    observation = market_observation()
    belief = _two_plan_belief(observation)
    first_key, second_key = belief.hypotheses
    utilities = (
        _utility(Action.ENTER, 0.50, first_key),
        _utility(Action.ENTER, 0.48, second_key),
        _utility(Action.ABSTAIN, 0.0, None),
    )
    decision = _ScriptedUtilityDecisionLayer(utilities).decide(
        observation,
        belief,
        flat_account(),
    )

    assert decision.selected_action is Action.ENTER
    assert decision.advantage == pytest.approx(0.50)
    assert decision.best_hypothesis_key == first_key
    assert decision.plan is belief.hypotheses[first_key].plan
    assert decision.utilities == utilities


def _plan_stub(plan, **changes):
    payload = {
        "direction": plan.direction,
        "planned_entry": plan.planned_entry,
        "invalidation": plan.invalidation,
        "targets": plan.targets,
        "deadline": plan.deadline,
        "setup_id": "setup-a",
        "entry_location_id": "location-a",
        "entry_path_id": "path-a",
        "entry_zone_lower": plan.planned_entry - 1.0,
        "entry_zone_upper": plan.planned_entry + 1.0,
        "selected_draw_id": plan.targets[0].level_id,
    }
    payload.update(changes)
    return SimpleNamespace(**payload)


def _enter_identity(plan, *, key: str = "candidate"):
    utility = _utility(Action.ENTER, 1.0, key)
    belief = SimpleNamespace(
        hypotheses={key: SimpleNamespace(plan=plan)},
    )
    return _concrete_action_identity(utility, belief)


@pytest.mark.parametrize(
    "changed_plan",
    (
        lambda plan: _plan_stub(plan, direction=Direction.SHORT),
        lambda plan: _plan_stub(
            plan,
            planned_entry=plan.planned_entry + 0.25,
        ),
        lambda plan: _plan_stub(
            plan,
            invalidation=replace(plan.invalidation, price=97.75),
        ),
        lambda plan: _plan_stub(
            plan,
            invalidation=replace(plan.invalidation, side="above"),
        ),
        lambda plan: _plan_stub(
            plan,
            invalidation=replace(
                plan.invalidation,
                source_level_id="other-stop-source",
            ),
        ),
        lambda plan: _plan_stub(
            plan,
            invalidation=replace(
                plan.invalidation,
                observed_at=plan.invalidation.observed_at
                + pd.Timedelta(minutes=1),
            ),
        ),
        lambda plan: _plan_stub(
            plan,
            targets=(replace(plan.targets[0], level_id="other-target"),),
        ),
        lambda plan: _plan_stub(
            plan,
            targets=(replace(plan.targets[0], timeframe=Timeframe.M5),),
        ),
        lambda plan: _plan_stub(
            plan,
            targets=(replace(plan.targets[0], side="below"),),
        ),
        lambda plan: _plan_stub(
            plan,
            targets=(replace(plan.targets[0], price=103.25),),
        ),
        lambda plan: _plan_stub(
            plan,
            targets=(
                replace(
                    plan.targets[0],
                    confirmed_at=plan.targets[0].confirmed_at
                    + pd.Timedelta(minutes=1),
                ),
            ),
        ),
        lambda plan: _plan_stub(
            plan,
            deadline=plan.deadline + pd.Timedelta(minutes=1),
        ),
        lambda plan: _plan_stub(plan, setup_id="setup-b"),
        lambda plan: _plan_stub(plan, entry_location_id="location-b"),
        lambda plan: _plan_stub(plan, entry_path_id="path-b"),
        lambda plan: _plan_stub(
            plan,
            entry_zone_lower=plan.planned_entry - 1.25,
        ),
        lambda plan: _plan_stub(
            plan,
            entry_zone_upper=plan.planned_entry + 1.25,
        ),
        lambda plan: _plan_stub(plan, selected_draw_id="other-draw"),
    ),
)
def test_enter_identity_preserves_every_frozen_plan_field(
    changed_plan,
) -> None:
    plan = long_plan()
    assert _enter_identity(_plan_stub(plan)) != _enter_identity(
        changed_plan(plan)
    )


def test_enter_identity_preserves_all_targets_and_their_order() -> None:
    plan = long_plan()
    secondary = replace(
        plan.targets[0],
        level_id="secondary-target",
        price=plan.targets[0].price + 1.0,
    )
    base = _plan_stub(plan, targets=(plan.targets[0], secondary))
    changed_secondary = _plan_stub(
        plan,
        targets=(
            plan.targets[0],
            replace(
                secondary,
                confirmed_at=secondary.confirmed_at
                + pd.Timedelta(minutes=1),
            ),
        ),
    )
    reversed_targets = _plan_stub(
        plan,
        targets=(secondary, plan.targets[0]),
    )

    assert _enter_identity(base) != _enter_identity(changed_secondary)
    assert _enter_identity(base) != _enter_identity(reversed_targets)


def test_distinct_enter_plans_below_margin_still_abstain() -> None:
    observation = market_observation()
    first = long_plan(observation)
    changed_invalidation = replace(
        first.invalidation,
        price=first.invalidation.price - 0.25,
    )
    second = replace(
        first,
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        invalidation=changed_invalidation,
        risk_points=first.risk_points + 0.25,
    )
    belief = _two_plan_belief(observation, second_plan=second)
    first_key, second_key = belief.hypotheses
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.ENTER, 0.50, first_key),
            _utility(Action.ENTER, 0.45, second_key),
            _utility(Action.ABSTAIN, 0.0, None),
        )
    ).decide(observation, belief, flat_account())

    assert decision.advantage == pytest.approx(0.05)
    assert decision.selected_action is Action.ABSTAIN


def test_position_verbs_remain_distinct_and_risk_receives_requested_action() -> None:
    observation = market_observation()
    utilities = (
        _utility(Action.HOLD, 0.70, "position"),
        _utility(Action.PROTECT, 0.50, "position"),
        _utility(Action.EXIT, 0.20, "position"),
        _utility(Action.ABSTAIN, 0.0, "position"),
    )
    layer = _ScriptedUtilityDecisionLayer(utilities)
    belief = _empty_belief(observation)
    account = _open_account(observation)
    decision = layer.decide(observation, belief, account)

    identities = {
        _concrete_action_identity(item, belief)
        for item in utilities
    }
    assert len(identities) == 4
    assert decision.selected_action is Action.HOLD
    assert decision.advantage == pytest.approx(0.20)
    risk = StructuralRiskEngine().review(decision, observation, account)
    assert risk.requested_action is Action.HOLD
    assert risk.final_action is Action.HOLD


def test_position_holds_valid_exact_frozen_thesis_without_delivery_probability() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation, delivery=0.0)
    hypothesis = next(iter(belief.candidates()))
    hypothesis = replace(
        hypothesis,
        phase=PlaybookPhase.ENTERED,
        phase_started_at=observation.asof,
        thesis_strength=0.01,
        delivery_quality=0.0,
    )
    belief = MarketBelief(observation.asof, {hypothesis.key: hypothesis})

    decision = _ready_layer().decide(
        observation,
        belief,
        _typed_open_account(observation, belief),
    )
    hold = next(item for item in decision.utilities if item.action is Action.HOLD)

    assert decision.selected_action is Action.HOLD
    assert "probability" not in hold.components
    assert hold.components["frozen_thesis_valid"] == 1.0


@pytest.mark.parametrize("exit_kind", ("stop", "deadline", "terminal"))
def test_position_hard_structural_conditions_exit(exit_kind: str) -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)
    hypothesis = next(iter(belief.candidates()))
    hypothesis = replace(
        hypothesis,
        phase=PlaybookPhase.ENTERED,
        phase_started_at=observation.asof,
    )
    belief = MarketBelief(observation.asof, {hypothesis.key: hypothesis})
    account = _typed_open_account(observation, belief)

    if exit_kind == "stop":
        observation = replace_market_observation(
            observation,
            price=account.position.current_stop,
        )
    elif exit_kind == "deadline":
        account = _typed_open_account(
            observation,
            belief,
            deadline=observation.asof,
        )
    else:
        hypothesis = replace(
            hypothesis,
            phase=PlaybookPhase.INVALIDATED,
            phase_started_at=observation.asof,
        )
        belief = MarketBelief(observation.asof, {hypothesis.key: hypothesis})

    decision = _ready_layer().decide(observation, belief, account)

    assert decision.selected_action is Action.EXIT
    assert any(
        item.action is Action.EXIT
        and item.components["structural_exit"] == 1.0
        for item in decision.utilities
    )


def test_position_missing_frozen_identity_exits_instead_of_optimistic_hold() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation, delivery=0.99)

    decision = _ready_layer().decide(
        observation,
        belief,
        _open_account(observation),
    )

    assert decision.selected_action is Action.EXIT
    assert all(item.action is not Action.HOLD for item in decision.utilities)
    assert any(
        item.components.get("identity_fail_closed") == 1.0
        for item in decision.utilities
    )


def test_position_protects_only_at_new_confirmed_causal_level() -> None:
    observation = market_observation()
    belief = _ready_executable_belief(observation)
    hypothesis = next(iter(belief.candidates()))
    hypothesis = replace(
        hypothesis,
        phase=PlaybookPhase.ENTERED,
        phase_started_at=observation.asof,
    )
    belief = MarketBelief(observation.asof, {hypothesis.key: hypothesis})
    account = _typed_open_account(observation, belief)
    assert account.position is not None
    source = next(
        item
        for item in observation.liquidity_inventory
        if (
            item.kind == "swing"
            and item.side == "below"
            and item.timeframe is Timeframe.H1
        )
    )
    source_swing = next(
        swing
        for swing in observation.frame(source.timeframe).swings
        if swing.swing_id == source.source_ids[0]
    )
    protection_source_id = "post-entry-swing-low"
    protection_swing = replace(
        source_swing,
        swing_id=protection_source_id,
        price=99.0,
        price_ticks=396,
        pivot_start=observation.asof - pd.Timedelta(hours=2),
        pivot_end=observation.asof - pd.Timedelta(hours=1),
        observed_at=observation.asof,
        confirmed_at=observation.asof,
    )
    protection = replace(
        source,
        item_id="post-entry-protected-low",
        price=99.0,
        lower_bound=99.0,
        upper_bound=99.0,
        formed_at=observation.asof - pd.Timedelta(minutes=1),
        confirmed_at=observation.asof,
        source_ids=(protection_source_id,),
    )
    frame = observation.frame(source.timeframe)
    observation = replace_market_observation(
        observation,
        frames={
            **observation.frames,
            source.timeframe: replace(
                frame,
                swings=(*frame.swings, protection_swing),
            ),
        },
        liquidity_inventory=(*observation.liquidity_inventory, protection),
    )

    decision = _ready_layer().decide(observation, belief, account)
    protect = next(
        item for item in decision.utilities if item.action is Action.PROTECT
    )

    assert decision.selected_action is Action.PROTECT
    assert protect.components["qualified_structural_protection"] == 1.0
    assert "probability" not in protect.components


def test_warmup_anomaly_still_hard_overrides_distinct_action_margin() -> None:
    observation = replace(
        market_observation(),
        anomalies=("warmup_h4",),
    )
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.WAIT, 0.50, "dfp:long"),
            _utility(Action.ABSTAIN, 0.0, None),
        )
    ).decide(observation, _empty_belief(observation), flat_account())

    assert decision.advantage == pytest.approx(0.50)
    assert decision.selected_action is Action.ABSTAIN
    assert decision.reasons[0] == "multitimeframe observer is still warming up"
    risk = StructuralRiskEngine().review(
        decision,
        observation,
        flat_account(),
    )
    assert risk.requested_action is Action.ABSTAIN


def test_unexplained_structured_episode_is_diagnostic_and_does_not_override_wait() -> None:
    observation = market_observation()
    belief = MarketBelief(
        observation.asof,
        {},
        global_context=_unexplained_context(
            observation,
            episode_ids=tuple(f"episode:{index}" for index in range(5)),
        ),
    )
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.WAIT, 0.50, "unknown"),
            _utility(Action.ABSTAIN, 0.0, None),
        )
    ).decide(observation, belief, flat_account())

    assert decision.selected_action is Action.WAIT
    assert decision.best_hypothesis_key == "unknown"
    diagnostic = next(
        reason
        for reason in decision.reasons
        if reason.startswith("unexplained_structured_episodes=")
    )
    assert "count:5" in diagnostic
    assert "ids:episode:0,episode:1,episode:2" in diagnostic
    assert "episode:3" not in diagnostic
    assert "episode:4" not in diagnostic
    assert "no ad-hoc playbook was created" in diagnostic


def test_root_without_context_episode_projection_cannot_enter() -> None:
    observation = market_observation()
    belief = _root_candidate_belief_with_unexplained(observation)

    decision = _ready_layer(
        DecisionConfig(minimum_utility_advantage=0.01),
    ).decide(observation, belief, flat_account())

    assert decision.selected_action is not Action.ENTER
    enter = next(
        item for item in decision.utilities if item.action is Action.ENTER
    )
    assert enter.components["action_authorized"] == 0.0
    assert "context_thesis_id_missing" in enter.reason
    assert "entry_episode_id_missing" in enter.reason
    assert "entry_episode_projection_mismatch" in enter.reason
    assert any(
        "unexplained_structured_episodes=count:1" in reason
        for reason in decision.reasons
    )


def test_root_without_context_episode_projection_cannot_wait() -> None:
    observation = market_observation()
    belief = _root_candidate_belief_with_unexplained(
        observation,
        phase=PlaybookPhase.WAITING_TRIGGER,
    )

    decision = _ready_layer(
        DecisionConfig(minimum_utility_advantage=0.0),
    ).decide(observation, belief, flat_account())

    assert decision.selected_action is Action.ABSTAIN
    assert all(item.action is not Action.WAIT for item in decision.utilities)


def test_unexplained_episode_does_not_override_frozen_position_hold() -> None:
    observation = market_observation()
    belief = _root_candidate_belief_with_unexplained(
        observation,
        phase=PlaybookPhase.ENTERED,
    )
    decision = _ready_layer().decide(
        observation,
        belief,
        _typed_open_account(observation, belief),
    )

    assert decision.selected_action is Action.HOLD
    assert any(
        "unexplained_structured_episodes=count:1" in reason
        for reason in decision.reasons
    )


def test_graph_context_without_action_candidate_naturally_abstains() -> None:
    observation = market_observation()
    summary_belief = _ready_executable_belief(observation)
    belief = replace(
        summary_belief,
        global_context=_unexplained_context(observation),
    )

    decision = _ready_layer().decide(observation, belief, flat_account())

    assert decision.selected_action is Action.ABSTAIN
    assert decision.best_hypothesis_key is None
    assert all(item.action is Action.ABSTAIN for item in decision.utilities)
    assert any(
        "unexplained_structured_episodes=count:1" in reason
        for reason in decision.reasons
    )


def test_single_enter_without_plan_is_explicitly_fail_closed() -> None:
    observation = market_observation()
    utilities = (
        _utility(Action.ENTER, 0.50, "broken-a"),
        _utility(Action.ABSTAIN, 0.0, None),
    )
    decision = _ScriptedUtilityDecisionLayer(utilities).decide(
        observation,
        _empty_belief(observation),
        flat_account(),
    )

    assert decision.advantage == pytest.approx(0.50)
    assert decision.selected_action is Action.ABSTAIN
    assert decision.best_hypothesis_key == "broken-a"
    assert decision.plan is None
    assert decision.reasons[0] == (
        "enter requires a resolvable frozen execution plan"
    )


def test_enter_without_plan_candidates_do_not_merge() -> None:
    observation = market_observation()
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.ENTER, 0.50, "broken-a"),
            _utility(Action.ENTER, 0.49, "broken-b"),
            _utility(Action.ABSTAIN, 0.0, None),
        )
    ).decide(observation, _empty_belief(observation), flat_account())

    assert decision.advantage == pytest.approx(0.01)
    assert decision.selected_action is Action.ABSTAIN
    assert decision.reasons[0] == (
        "enter requires a resolvable frozen execution plan"
    )


def test_missing_distinct_runner_up_has_zero_advantage() -> None:
    observation = market_observation()
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.WAIT, 0.50, "dfp:long"),
            _utility(Action.WAIT, 0.40, "lsr:long"),
        )
    ).decide(observation, _empty_belief(observation), flat_account())

    assert decision.advantage == 0.0
    assert decision.selected_action is Action.ABSTAIN
