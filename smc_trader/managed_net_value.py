"""Action-clock managed gross value and causal execution-cost decomposition."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_equivalence import (
    ActionEquivalenceDecisionLayer,
    ActionEquivalenceProtocol,
    EquivalentActionGroup,
    action_utility_is_ready,
    collapse_equivalent_action_utilities,
    equivalent_action_groups,
)
from .decision import DecisionConfig, UtilityDecisionLayer
from .engine import ContinuousSMCEngine
from .model import (
    Action,
    ActionUtility,
    MarketBelief,
    MarketObservation,
    Playbook,
    PlaybookPhase,
)


FEATURE_NAMES = (
    "max_calibrated_belief",
    "second_calibrated_belief",
    "belief_dispersion",
    "equivalent_hypothesis_count",
    "representative_raw_probability",
    "calibration_gap",
    "primary_target_R_capped",
    "remaining_path_R_capped",
    "minutes_to_deadline_scaled",
    "evidence_dfp",
    "evidence_lsr",
    "evidence_favr",
)


@dataclass(frozen=True)
class ManagedActionFeatureVector:
    action_key: str
    representative_hypothesis_key: str
    evidence_hypothesis_keys: tuple[str, ...]
    names: tuple[str, ...]
    values: tuple[float, ...]

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))


@dataclass(frozen=True)
class RidgeManagedGrossModel:
    version: str
    fingerprint: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    residual_rmse_R: float
    support_min_z: tuple[float, ...]
    support_max_z: tuple[float, ...]
    oof_metrics: Mapping[str, float]
    status: str

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_protocol_hash: str,
        expected_action_equivalence_hash: str,
        expected_code_hash: str,
        expected_base_config_hash: str,
    ) -> "RidgeManagedGrossModel":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("managed-net-value artifact root must be an object")
        if payload.get("status") != "ready":
            raise ValueError("managed-net-value artifact is not ready")
        bindings = {
            "managed_net_value_protocol_hash": expected_protocol_hash,
            "action_equivalence_protocol_hash": expected_action_equivalence_hash,
            "managed_net_value_code_hash": expected_code_hash,
            "policy_base_config_hash": expected_base_config_hash,
        }
        for field, expected in bindings.items():
            if str(payload.get(field, "")) != str(expected):
                raise ValueError(f"managed-net-value artifact {field} is stale")
        names = tuple(str(item) for item in payload.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError("managed-net-value artifact feature order changed")
        model = payload.get("model")
        support = payload.get("support")
        if not isinstance(model, Mapping) or not isinstance(support, Mapping):
            raise ValueError("managed-net-value artifact omits model/support")
        values = {
            "means": tuple(float(item) for item in model.get("means", ())),
            "scales": tuple(float(item) for item in model.get("scales", ())),
            "coefficients": tuple(
                float(item) for item in model.get("coefficients", ())
            ),
            "support_min_z": tuple(
                float(item) for item in support.get("minimum_z", ())
            ),
            "support_max_z": tuple(
                float(item) for item in support.get("maximum_z", ())
            ),
        }
        if any(len(items) != len(FEATURE_NAMES) for items in values.values()):
            raise ValueError("managed-net-value artifact vector width is invalid")
        try:
            intercept = float(model["intercept"])
            residual_rmse = float(model["residual_rmse_R"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "managed-net-value artifact omits scalar model fields"
            ) from exc
        finite = [
            number for items in values.values() for number in items
        ] + [intercept, residual_rmse]
        if not all(math.isfinite(number) for number in finite):
            raise ValueError("managed-net-value artifact contains non-finite values")
        if (
            any(scale <= 0 for scale in values["scales"])
            or residual_rmse < 0
            or any(
                right < left
                for left, right in zip(
                    values["support_min_z"],
                    values["support_max_z"],
                )
            )
        ):
            raise ValueError("managed-net-value artifact scales/support are invalid")
        metrics = payload.get("oof_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("managed-net-value artifact omits OOF metrics")
        parsed_metrics = {str(key): float(value) for key, value in metrics.items()}
        if not all(math.isfinite(value) for value in parsed_metrics.values()):
            raise ValueError("managed-net-value OOF metrics are invalid")
        return cls(
            version=str(payload.get("calibration_version", "")),
            fingerprint=hashlib.sha256(raw).hexdigest(),
            feature_names=names,
            means=values["means"],
            scales=values["scales"],
            coefficients=values["coefficients"],
            intercept=intercept,
            residual_rmse_R=residual_rmse,
            support_min_z=values["support_min_z"],
            support_max_z=values["support_max_z"],
            oof_metrics=parsed_metrics,
            status="ready",
        )

    def standardized(
        self,
        features: ManagedActionFeatureVector,
    ) -> np.ndarray:
        if features.names != self.feature_names:
            raise ValueError("managed-net-value feature order mismatch")
        values = np.asarray(features.values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("managed-net-value features are non-finite")
        return (
            values - np.asarray(self.means, dtype=float)
        ) / np.asarray(self.scales, dtype=float)

    def supports(self, features: ManagedActionFeatureVector) -> bool:
        values = self.standardized(features)
        return bool(
            np.all(values >= np.asarray(self.support_min_z) - 0.25)
            and np.all(values <= np.asarray(self.support_max_z) + 0.25)
        )

    def predict_mean_R(self, features: ManagedActionFeatureVector) -> float:
        values = self.standardized(features)
        return float(
            self.intercept
            + values @ np.asarray(self.coefficients, dtype=float)
        )

    def predict_lower_R(self, features: ManagedActionFeatureVector) -> float:
        return self.predict_mean_R(features) - self.residual_rmse_R


def _raw_probability(value: Any) -> float:
    raw = value.raw_probability
    return float(value.probability if raw is None else raw)


def managed_action_features(
    group: EquivalentActionGroup,
    belief: MarketBelief,
    observation: MarketObservation,
) -> ManagedActionFeatureVector:
    representative_key = group.representative.hypothesis_key
    if representative_key is None:
        raise ValueError("managed action group lacks representative hypothesis")
    representative = belief.hypotheses.get(representative_key)
    if representative is None or representative.plan is None:
        raise ValueError("managed action representative has no causal plan")
    probabilities = tuple(
        sorted(
            (
                float(belief.hypotheses[key].probability)
                for key in group.hypothesis_keys
            ),
            reverse=True,
        )
    )
    maximum = probabilities[0]
    second = probabilities[1] if len(probabilities) > 1 else 0.0
    minimum = probabilities[-1]
    raw = _raw_probability(representative)
    plan = representative.plan
    remaining_minutes = max(
        0.0,
        (plan.deadline - observation.asof).total_seconds() / 60.0,
    )
    playbooks = {
        belief.hypotheses[key].playbook
        for key in group.hypothesis_keys
    }
    values = (
        maximum,
        second,
        maximum - minimum,
        float(len(group.hypothesis_keys)),
        raw,
        float(representative.probability) - raw,
        min(3.0, max(0.0, float(plan.primary_target_R))),
        min(3.0, max(0.0, float(plan.remaining_path_R))),
        min(1.0, remaining_minutes / 120.0),
        1.0 if Playbook.DISPLACEMENT_FIRST_PULLBACK in playbooks else 0.0,
        1.0 if Playbook.LIQUIDITY_SWEEP_REVERSAL in playbooks else 0.0,
        1.0 if Playbook.FAILED_AUCTION_VALUE_RETURN in playbooks else 0.0,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("managed action features contain non-finite values")
    return ManagedActionFeatureVector(
        action_key=group.identity.key,
        representative_hypothesis_key=representative_key,
        evidence_hypothesis_keys=group.hypothesis_keys,
        names=FEATURE_NAMES,
        values=values,
    )


def fit_fixed_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_lambda: float = 10.0,
) -> dict[str, Any]:
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != target.shape[0]
        or matrix.shape[1] != len(FEATURE_NAMES)
        or matrix.shape[0] < 2
        or not np.isfinite(matrix).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("fixed ridge inputs are invalid")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("fixed ridge penalty must be positive")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    standardized = (matrix - means) / scales
    centered = target - target.mean()
    gram = standardized.T @ standardized
    penalty = ridge_lambda * np.eye(matrix.shape[1])
    coefficients = np.linalg.solve(
        gram + penalty,
        standardized.T @ centered,
    )
    intercept = float(target.mean())
    predictions = intercept + standardized @ coefficients
    residual_rmse = float(np.sqrt(np.mean((target - predictions) ** 2)))
    return {
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "intercept": intercept,
        "residual_rmse_R": residual_rmse,
        "minimum_z": standardized.min(axis=0),
        "maximum_z": standardized.max(axis=0),
        "predictions": predictions,
    }


class ManagedNetValueDecisionLayer(ActionEquivalenceDecisionLayer):
    """Use managed action value only after unique-action evidence aggregation."""

    def __init__(
        self,
        action_protocol: ActionEquivalenceProtocol,
        model: RidgeManagedGrossModel,
        config: DecisionConfig | None = None,
        *,
        disabled_playbooks: Sequence[Playbook] = (),
    ) -> None:
        super().__init__(
            action_protocol,
            config,
            disabled_playbooks=disabled_playbooks,
        )
        self.model = model

    def _flat_utilities(
        self,
        observation: MarketObservation,
        belief: MarketBelief,
    ) -> list[ActionUtility]:
        enabled = {
            key: hypothesis
            for key, hypothesis in belief.hypotheses.items()
            if hypothesis.playbook not in self.disabled_playbooks
        }
        filtered = MarketBelief(asof=belief.asof, hypotheses=enabled)
        raw = [
            utility
            for utility in UtilityDecisionLayer._flat_utilities(
                self,
                observation,
                filtered,
            )
            if action_utility_is_ready(utility, filtered)
        ]
        groups = equivalent_action_groups(
            raw,
            filtered,
            tick_size=self.protocol.tick_size,
        )
        collapsed = collapse_equivalent_action_utilities(
            raw,
            filtered,
            tick_size=self.protocol.tick_size,
        )
        output = [
            utility
            for utility in collapsed
            if utility.action is not Action.ENTER
        ]
        for group in groups:
            if group.identity.verb is not Action.ENTER:
                continue
            representative = group.representative
            hypothesis = filtered.hypotheses[
                representative.hypothesis_key or ""
            ]
            executable = any(
                filtered.hypotheses[key].phase is PlaybookPhase.EXECUTABLE
                for key in group.hypothesis_keys
            )
            features = managed_action_features(
                group,
                filtered,
                observation,
            )
            supported = self.model.supports(features)
            mean_R = (
                self.model.predict_mean_R(features)
                if supported
                else -1.0
            )
            lower_R = (
                self.model.predict_lower_R(features)
                if supported
                else -1.0
            )
            plan = hypothesis.plan
            if plan is None or plan.risk_points <= 0:
                raise ValueError("managed unique action has invalid risk")
            cost_R = (
                observation.execution.expected_round_trip_cost_points
                / plan.risk_points
            )
            fillability_penalty = 0.35 * (
                1.0 - observation.execution.fillability
            )
            available = supported and executable
            net_R = (
                lower_R - cost_R - fillability_penalty
                if available
                else -1.0
            )
            components = {
                "managed_gross_mean_R": float(mean_R),
                "managed_gross_lower_R": float(lower_R),
                "managed_residual_rmse_R": float(
                    self.model.residual_rmse_R
                ),
                "cost_R": -float(cost_R),
                "fillability": -float(fillability_penalty),
                "managed_value_in_support": 1.0 if supported else -1.0,
                "phase_executable": 1.0 if executable else -1.0,
                **features.as_mapping(),
            }
            output.append(
                ActionUtility(
                    action=Action.ENTER,
                    utility=float(net_R),
                    components=components,
                    hypothesis_key=representative.hypothesis_key,
                    reason=(
                        f"managed-net-value {self.model.version}; unique action "
                        f"{features.action_key[:12]}; evidence "
                        f"[{', '.join(features.evidence_hypothesis_keys)}]; "
                        f"mean={mean_R:.3f}R lower={lower_R:.3f}R "
                        f"cost={cost_R:.3f}R fill penalty="
                        f"{fillability_penalty:.3f}R"
                    ),
                )
            )
        return output


def build_v2_2_policy_base_engine(
    config_path: str | Path = "configs/model_v2_2_policy_base.json",
    *,
    disabled_playbooks: Sequence[Playbook] = (),
) -> ContinuousSMCEngine:
    source = Path(config_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not str(payload.get("version", "")).startswith("2.2."):
        raise ValueError("v2.2 policy-base engine requires a v2.2 config")
    if payload.get("managed_net_value_artifact") is not None:
        raise ValueError("policy-base engine cannot load a fitted value artifact")
    protocol = ActionEquivalenceProtocol.from_file(
        payload.get(
            "action_equivalence_protocol",
            "configs/action_equivalence_v2_2.json",
        )
    )
    engine = ContinuousSMCEngine.from_config(source)
    engine.decision = ActionEquivalenceDecisionLayer(
        protocol,
        engine.decision.config,
        disabled_playbooks=disabled_playbooks,
    )
    return engine


def managed_net_value_code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("action_equivalence.py", "managed_net_value.py"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "FEATURE_NAMES",
    "ManagedActionFeatureVector",
    "ManagedNetValueDecisionLayer",
    "RidgeManagedGrossModel",
    "build_v2_2_policy_base_engine",
    "fit_fixed_ridge",
    "managed_action_features",
    "managed_net_value_code_fingerprint",
]
