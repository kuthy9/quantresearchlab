from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from smc_trader.action_equivalence import (
    ActionEquivalenceDecisionLayer,
    ActionEquivalenceProtocol,
    PlanRelation,
    equivalent_action_groups,
    plan_relation,
)
from smc_trader.decision import DecisionConfig, UtilityDecisionLayer
from smc_trader.managed_net_value import (
    FEATURE_NAMES,
    ManagedNetValueDecisionLayer,
    RidgeManagedGrossModel,
    fit_fixed_ridge,
    managed_action_features,
)
from smc_trader.model import (
    Action,
    MarketBelief,
    Playbook,
    Timeframe,
    VetoCode,
)
from smc_trader.visual_audit import (
    AuditScenario,
    ScenarioVisualAuditSampler,
    classify_audit_scenarios,
)
from smc_trader.visualization import DecisionVisualizer

from .helpers import (
    candle,
    engine_snapshot,
    executable_belief,
    flat_account,
    market_observation,
)


def _protocol() -> ActionEquivalenceProtocol:
    return ActionEquivalenceProtocol(
        version="2.2-test",
        fingerprint="a" * 64,
        tick_size=0.25,
        status="preregistered_before_v2_2_fit",
    )


def _equivalent_belief() -> MarketBelief:
    observation = market_observation()
    first = next(
        iter(
            executable_belief(
                observation,
                probability=0.95,
                uncertainty=0.02,
            ).hypotheses.values()
        )
    )
    second_plan = replace(
        first.plan,
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
    )
    second = replace(
        first,
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        probability=0.93,
        plan=second_plan,
    )
    return MarketBelief(
        asof=observation.asof,
        hypotheses={first.key: first, second.key: second},
    )


def test_equivalent_playbook_evidence_is_one_action_not_margin_competition() -> None:
    observation = market_observation()
    belief = _equivalent_belief()
    config = DecisionConfig(minimum_utility_advantage=0.12)
    legacy = UtilityDecisionLayer(config).decide(
        observation,
        belief,
        flat_account(),
    )
    assert legacy.selected_action is Action.ENTER
    assert legacy.advantage >= config.minimum_utility_advantage

    unique = ActionEquivalenceDecisionLayer(
        _protocol(),
        config,
    ).decide(
        observation,
        belief,
        flat_account(),
    )
    assert unique.selected_action is Action.ENTER
    enter = [
        utility
        for utility in unique.utilities
        if utility.action is Action.ENTER
    ]
    assert len(enter) == 1
    assert enter[0].components["equivalent_hypothesis_count"] == 2.0
    assert "liquidity_sweep_reversal:long" in enter[0].reason


def test_disabled_playbook_is_removed_before_action_evidence_grouping() -> None:
    observation = market_observation()
    belief = _equivalent_belief()
    unique = ActionEquivalenceDecisionLayer(
        _protocol(),
        DecisionConfig(minimum_utility_advantage=0.12),
        disabled_playbooks=(Playbook.LIQUIDITY_SWEEP_REVERSAL,),
    ).decide(
        observation,
        belief,
        flat_account(),
    )
    enter = [
        utility
        for utility in unique.utilities
        if utility.action is Action.ENTER
    ]
    assert len(enter) == 1
    assert enter[0].components["equivalent_hypothesis_count"] == 1.0
    assert "liquidity_sweep_reversal:long" not in enter[0].reason


def test_different_structural_stop_remains_a_real_competing_plan() -> None:
    belief = _equivalent_belief()
    first, second = belief.hypotheses.values()
    plan = second.plan
    assert plan is not None
    changed_stop = replace(
        plan.invalidation,
        price=97.75,
        source_level_id="different-confirmed-stop",
    )
    changed_plan = replace(
        plan,
        invalidation=changed_stop,
        risk_points=2.25,
        primary_target_R=3.0 / 2.25,
        remaining_path_R=3.0 / 2.25,
    )
    changed = replace(second, plan=changed_plan, invalidation=changed_stop)
    separated = MarketBelief(
        asof=belief.asof,
        hypotheses={first.key: first, changed.key: changed},
    )
    assert plan_relation(
        first.plan,
        changed.plan,
        tick_size=0.25,
    ) is PlanRelation.COMPETING_STRUCTURAL_STOP
    raw = UtilityDecisionLayer()._flat_utilities(
        market_observation(),
        separated,
    )
    groups = equivalent_action_groups(raw, separated, tick_size=0.25)
    assert len(
        [group for group in groups if group.identity.verb is Action.ENTER]
    ) == 2


def test_managed_action_features_retain_correlated_evidence_without_sum() -> None:
    observation = market_observation()
    belief = _equivalent_belief()
    raw = UtilityDecisionLayer()._flat_utilities(observation, belief)
    group = next(
        group
        for group in equivalent_action_groups(raw, belief, tick_size=0.25)
        if group.identity.verb is Action.ENTER
    )
    features = managed_action_features(group, belief, observation)
    values = features.as_mapping()
    assert features.names == FEATURE_NAMES
    assert values["equivalent_hypothesis_count"] == 2.0
    assert values["max_calibrated_belief"] == 0.95
    assert values["second_calibrated_belief"] == 0.93
    assert values["max_calibrated_belief"] < (
        values["max_calibrated_belief"]
        + values["second_calibrated_belief"]
    )
    assert values["evidence_dfp"] == 1.0
    assert values["evidence_lsr"] == 1.0


def test_fixed_ridge_uses_only_registered_width_and_recovers_signal() -> None:
    rows = 240
    base = np.linspace(-1.0, 1.0, rows)
    matrix = np.column_stack(
        [np.sin((index + 1) * base) for index in range(len(FEATURE_NAMES))]
    )
    labels = 0.4 * matrix[:, 0] - 0.2 * matrix[:, 6] + 0.1
    fitted = fit_fixed_ridge(matrix, labels, ridge_lambda=10.0)
    correlation = np.corrcoef(fitted["predictions"], labels)[0, 1]
    assert correlation > 0.95
    assert fitted["coefficients"].shape == (len(FEATURE_NAMES),)
    assert fitted["residual_rmse_R"] < labels.std()


def test_managed_net_value_is_deducted_at_unique_action_clock() -> None:
    observation = market_observation()
    belief = _equivalent_belief()
    width = len(FEATURE_NAMES)
    model = RidgeManagedGrossModel(
        version="test",
        fingerprint="b" * 64,
        feature_names=FEATURE_NAMES,
        means=(0.0,) * width,
        scales=(1.0,) * width,
        coefficients=(0.0,) * width,
        intercept=1.0,
        residual_rmse_R=0.1,
        support_min_z=(-10.0,) * width,
        support_max_z=(10.0,) * width,
        oof_metrics={"spearman": 0.1},
        status="ready",
    )
    decision = ManagedNetValueDecisionLayer(
        _protocol(),
        model,
        DecisionConfig(minimum_utility_advantage=0.01),
    ).decide(
        observation,
        belief,
        flat_account(),
    )
    enter = next(
        utility
        for utility in decision.utilities
        if utility.action is Action.ENTER
    )
    assert enter.components["managed_gross_mean_R"] == 1.0
    assert enter.components["managed_gross_lower_R"] == 0.9
    assert enter.components["cost_R"] < 0.0
    assert enter.components["equivalent_hypothesis_count"] == 2.0
    assert decision.selected_action is Action.ENTER


def test_scenario_classifier_distinguishes_required_audit_events() -> None:
    snapshot = engine_snapshot()
    enter = classify_audit_scenarios(snapshot)
    assert {item[0] for item in enter} == {AuditScenario.ENTER}

    wait_snapshot = replace(
        snapshot,
        decision=replace(snapshot.decision, selected_action=Action.WAIT),
        risk=replace(
            snapshot.risk,
            requested_action=Action.WAIT,
            final_action=Action.WAIT,
        ),
    )
    assert AuditScenario.WAIT in {
        item[0] for item in classify_audit_scenarios(wait_snapshot)
    }

    abstain_snapshot = replace(
        snapshot,
        decision=replace(
            snapshot.decision,
            selected_action=Action.ABSTAIN,
        ),
        risk=replace(
            snapshot.risk,
            requested_action=Action.ABSTAIN,
            final_action=Action.ABSTAIN,
            passed=True,
            vetoes=(),
        ),
    )
    assert AuditScenario.ABSTAIN in {
        item[0] for item in classify_audit_scenarios(abstain_snapshot)
    }

    veto_snapshot = replace(
        snapshot,
        risk=replace(
            snapshot.risk,
            final_action=Action.ABSTAIN,
            passed=False,
            vetoes=(VetoCode.COST,),
        ),
    )
    assert AuditScenario.VETO in {
        item[0] for item in classify_audit_scenarios(veto_snapshot)
    }

    protect_snapshot = replace(
        snapshot,
        decision=replace(snapshot.decision, selected_action=Action.PROTECT),
        risk=replace(
            snapshot.risk,
            requested_action=Action.PROTECT,
            final_action=Action.PROTECT,
            protected_stop=100.25,
        ),
    )
    assert AuditScenario.PROTECT in {
        item[0] for item in classify_audit_scenarios(protect_snapshot)
    }

    stop = SimpleNamespace(
        thesis_hash="stop-thesis",
        playbook="displacement_first_pullback",
        direction="short",
        decision_time=snapshot.observation.asof - pd.Timedelta(minutes=10),
        opened_at=snapshot.observation.asof - pd.Timedelta(minutes=8),
        closed_at=snapshot.observation.asof,
        entry_price=100.0,
        original_invalidation=101.0,
        final_stop=100.5,
        target=98.0,
        exit_price=100.5,
        exit_reason="structural_stop",
        gross_R=-1.0,
        cost_R=0.1,
        net_R=-1.1,
        ambiguous_same_bar=False,
    )
    target = SimpleNamespace(
        thesis_hash="target-thesis",
        playbook="liquidity_sweep_reversal",
        direction="long",
        decision_time=snapshot.observation.asof - pd.Timedelta(minutes=12),
        opened_at=snapshot.observation.asof - pd.Timedelta(minutes=9),
        closed_at=snapshot.observation.asof,
        entry_price=100.0,
        original_invalidation=99.0,
        final_stop=100.25,
        target=102.0,
        exit_price=102.0,
        exit_reason="primary_target",
        gross_R=1.5,
        cost_R=0.1,
        net_R=1.4,
        ambiguous_same_bar=False,
    )
    closure = classify_audit_scenarios(snapshot, (stop, target))
    assert {AuditScenario.STOP, AuditScenario.TARGET}.issubset(
        {item[0] for item in closure}
    )
    target_context = next(
        context
        for scenario, context in closure
        if scenario is AuditScenario.TARGET
    )
    assert target_context["closed_trade"]["entry_price"] == 100.0
    assert target_context["closed_trade"]["original_invalidation"] == 99.0
    assert target_context["closed_trade"]["final_stop"] == 100.25
    assert target_context["closed_trade"]["target"] == 102.0
    assert target_context["closed_trade"]["exit_price"] == 102.0


def test_scenario_manifest_reports_missing_real_coverage(tmp_path) -> None:
    snapshot = engine_snapshot()
    histories = {
        Timeframe.H4: (
            candle(Timeframe.H4, "2025-01-06 06:00", 99.0),
        ),
        Timeframe.H1: (
            candle(Timeframe.H1, "2025-01-06 09:00", 99.5),
        ),
        Timeframe.M5: (
            candle(Timeframe.M5, "2025-01-06 09:55", 100.0),
        ),
        Timeframe.M1: (
            candle(Timeframe.M1, "2025-01-06 09:59", 100.0),
        ),
    }
    sampler = ScenarioVisualAuditSampler(tmp_path, quota_per_scenario=1)
    captures = sampler.observe(
        snapshot,
        histories,
        DecisionVisualizer(),
    )
    assert len(captures) == 1
    assert captures[0].decision_packet.exists()
    manifest_path = sampler.write_manifest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["enter"] == 1
    assert not manifest["coverage_complete"]
    assert "stop" in manifest["missing_scenarios"]
