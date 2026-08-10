from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.decision import (
    DecisionConfig,
    UtilityDecisionLayer,
    _concrete_action_identity,
)
from smc_trader.model import (
    AccountState,
    Action,
    ActionUtility,
    Direction,
    GlobalMarketContext,
    MarketBelief,
    MarketMode,
    Playbook,
    PlaybookPhase,
    PositionSnapshot,
    ScaleRelation,
    Timeframe,
)
from smc_trader.playbooks import BrainConfig, PlaybookBrain
from smc_trader.risk import StructuralRiskEngine

from .helpers import (
    executable_belief,
    flat_account,
    long_plan,
    market_observation,
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


def _unexplained_context(observation) -> GlobalMarketContext:
    return GlobalMarketContext(
        updated_at=observation.asof,
        scene_revision_id="scene:r000000000001",
        market_epoch_id="epoch:0",
        authority_timeframe=None,
        authority_direction=None,
        authority_source_ids=(),
        market_mode=MarketMode.UNCERTAIN,
        scale_relations={
            timeframe.value: ScaleRelation.UNKNOWN
            for timeframe in Timeframe
        },
        external_draw_candidates={"above": (), "below": ()},
        path_blocker_ids=(),
        material_conflicts=(),
        unknown_evidence=(),
        ambiguous_evidence=(),
        unexplained_structured_episode_ids=("episode:unexplained",),
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
    belief = executable_belief(observation, probability=0.95, uncertainty=0.02)
    decision = UtilityDecisionLayer(
        DecisionConfig(minimum_utility_advantage=0.01)
    ).decide(observation, belief, flat_account())
    assert decision.selected_action is Action.ENTER
    assert decision.plan is not None


def test_ambiguous_action_advantage_forces_abstain() -> None:
    observation = market_observation()
    belief = executable_belief(observation, probability=0.95, uncertainty=0.02)
    decision = UtilityDecisionLayer(
        DecisionConfig(minimum_utility_advantage=100.0)
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


def test_unexplained_structured_episode_abstains_without_inventing_playbook() -> None:
    observation = market_observation()
    belief = MarketBelief(
        observation.asof,
        {},
        global_context=_unexplained_context(observation),
    )
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.WAIT, 0.50, "unknown"),
            _utility(Action.ABSTAIN, 0.0, None),
        )
    ).decide(observation, belief, flat_account())

    assert decision.selected_action is Action.ABSTAIN
    assert decision.best_hypothesis_key == "unknown"
    assert decision.reasons[0] == (
        "no fixed playbook explains the current high-salience structured episode"
    )
    assert any(
        "no ad-hoc playbook was created" in reason
        for reason in decision.reasons
    )


def test_unexplained_competing_episode_does_not_override_active_fixed_playbook() -> None:
    observation = market_observation()
    base = executable_belief(
        observation,
        probability=0.95,
        uncertainty=0.02,
    )
    hypothesis = replace(
        next(iter(base.hypotheses.values())),
        setup_context_id="episode:active-fixed-playbook",
    )
    belief = replace(
        base,
        hypotheses={hypothesis.key: hypothesis},
        global_context=_unexplained_context(observation),
    )
    decision = _ScriptedUtilityDecisionLayer(
        (
            _utility(Action.ENTER, 0.50, hypothesis.key),
            _utility(Action.ABSTAIN, 0.0, None),
        )
    ).decide(observation, belief, flat_account())

    assert decision.selected_action is Action.ENTER
    assert decision.best_hypothesis_key == hypothesis.key
    assert any(
        "episode:unexplained" in reason
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
