from __future__ import annotations

import pytest

from smc_trader.calibration import CalibrationError
from smc_trader.decision import DecisionConfig, UtilityDecisionLayer
from smc_trader.model import Action, Playbook
from smc_trader.policy_value import (
    ManagedPolicyValueCalibrator,
    ManagedUtilityDecisionLayer,
    PlaybookPolicyValueMap,
    PolicyValuePoint,
    monotone_policy_value_points,
)

from .helpers import executable_belief, flat_account, market_observation


def _calibrator(
    *,
    low_value_R: float,
    high_value_R: float,
) -> ManagedPolicyValueCalibrator:
    playbook = Playbook.DISPLACEMENT_FIRST_PULLBACK
    mapping = PlaybookPolicyValueMap(
        playbook=playbook,
        episodes=120,
        direction_episodes={"long": 60, "short": 60},
        points=(
            PolicyValuePoint(0.0, low_value_R, 60),
            PolicyValuePoint(2.0, high_value_R, 60),
        ),
    )
    return ManagedPolicyValueCalibrator(
        version="test",
        fingerprint="artifact",
        policy_protocol_hash="protocol",
        registry_hash="registry",
        managed_policy_code_hash="code",
        managed_policy_pipeline_hash="pipeline",
        policy_base_config_hash="config",
        maps={playbook: mapping},
        status="ready",
    )


def test_managed_value_fit_is_monotone_without_threshold_search() -> None:
    x = [0.1] * 15 + [0.4] * 15 + [0.8] * 15 + [1.2] * 15
    y = [-1.0] * 10 + [1.0] * 5
    y += [-1.0] * 7 + [1.0] * 8
    y += [-1.0] * 8 + [1.0] * 7
    y += [-1.0] * 3 + [1.0] * 12
    points = monotone_policy_value_points(
        x,
        y,
        bins=4,
        minimum_bin_episodes=15,
        prior_weight=0,
    )
    assert len(points) >= 2
    assert sum(point.episodes for point in points) == len(x)
    assert all(
        left.managed_gross_R <= right.managed_gross_R
        for left, right in zip(points[:-1], points[1:])
    )


def test_managed_value_map_forbids_extrapolation() -> None:
    mapping = next(iter(_calibrator(low_value_R=0.2, high_value_R=0.8).maps.values()))
    assert mapping.supports(1.0)
    assert not mapping.supports(2.1)
    with pytest.raises(CalibrationError):
        mapping.apply(2.1)


def test_managed_value_can_filter_behavior_policy_enter() -> None:
    observation = market_observation()
    belief = executable_belief(
        observation,
        probability=0.95,
        uncertainty=0.02,
    )
    config = DecisionConfig(minimum_utility_advantage=0.01)
    behavior = UtilityDecisionLayer(config).decide(
        observation,
        belief,
        flat_account(),
    )
    assert behavior.selected_action is Action.ENTER
    managed = ManagedUtilityDecisionLayer(
        _calibrator(low_value_R=-0.8, high_value_R=-0.4),
        config,
    ).decide(
        observation,
        belief,
        flat_account(),
    )
    assert managed.selected_action is not Action.ENTER
    assert "conservatively filtered" in managed.reasons[0]


def test_managed_value_cannot_promote_unselected_counterfactual_enter() -> None:
    observation = market_observation()
    belief = executable_belief(
        observation,
        probability=0.5,
        uncertainty=0.1,
    )
    config = DecisionConfig(minimum_utility_advantage=0.01)
    behavior = UtilityDecisionLayer(config).decide(
        observation,
        belief,
        flat_account(),
    )
    assert behavior.selected_action is not Action.ENTER
    managed = ManagedUtilityDecisionLayer(
        _calibrator(low_value_R=2.0, high_value_R=2.5),
        config,
    ).decide(
        observation,
        belief,
        flat_account(),
    )
    assert managed == behavior
