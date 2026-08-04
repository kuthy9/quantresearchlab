"""Frozen regularized v2.3 fill, gross, downside and delta value models."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .action_clock import (
    FEATURE_NAMES,
    FLAT_ACTIONS,
    ActionClockProtocol,
    PlanLineageStore,
    action_clock_features,
)
from .action_equivalence import (
    ActionEquivalenceDecisionLayer,
    ActionEquivalenceProtocol,
    action_utility_is_ready,
    equivalent_action_groups,
)
from .calibration import model_code_fingerprint
from .decision import DecisionConfig, UtilityDecisionLayer
from .engine import ContinuousSMCEngine
from .model import (
    AccountState,
    Action,
    ActionUtility,
    Direction,
    MarketBelief,
    MarketObservation,
    Playbook,
    StructuralLevel,
    Timeframe,
)
from .shadow_replay import (
    POSITION_ACTIONS,
    POSITION_FEATURE_NAMES,
    POSITION_SAMPLING_STRIDE_MINUTES,
)
from .risk import causal_protection_candidate


BatchFactory = Callable[[], Iterable[tuple[np.ndarray, np.ndarray]]]


def _validate_batch(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
) -> None:
    if (
        matrix.ndim != 2
        or target.ndim != 1
        or matrix.shape[0] != target.shape[0]
        or matrix.shape[1] != len(feature_names)
        or not np.isfinite(matrix).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("action-value training batch is invalid")


def _standardization(
    batches: BatchFactory,
    feature_names: Sequence[str],
) -> tuple[int, np.ndarray, np.ndarray, float]:
    count = 0
    total = np.zeros(len(feature_names), dtype=float)
    total_sq = np.zeros(len(feature_names), dtype=float)
    target_total = 0.0
    for matrix, target in batches():
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        _validate_batch(matrix, target, feature_names)
        count += len(target)
        total += matrix.sum(axis=0)
        total_sq += np.square(matrix).sum(axis=0)
        target_total += float(target.sum())
    if count < 2:
        raise ValueError("action-value fit requires at least two samples")
    means = total / count
    variance = np.maximum(0.0, total_sq / count - np.square(means))
    scales = np.sqrt(variance)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    return count, means, scales, target_total / count


def fit_streaming_ridge(
    batches: BatchFactory,
    *,
    ridge_lambda: float,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict[str, Any]:
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge penalty must be positive")
    names = tuple(str(item) for item in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("ridge feature names are empty or duplicated")
    count, means, scales, target_mean = _standardization(batches, names)
    gram = np.zeros((len(names), len(names)), dtype=float)
    cross = np.zeros(len(names), dtype=float)
    for matrix, target in batches():
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        _validate_batch(matrix, target, names)
        standardized = (matrix - means) / scales
        gram += standardized.T @ standardized
        cross += standardized.T @ (target - target_mean)
    coefficients = np.linalg.solve(
        gram + ridge_lambda * np.eye(len(names)),
        cross,
    )
    squared_error = 0.0
    residuals: list[np.ndarray] = []
    for matrix, target in batches():
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        predicted = target_mean + ((matrix - means) / scales) @ coefficients
        residual = target - predicted
        squared_error += float(np.square(residual).sum())
        residuals.append(residual.astype(np.float32, copy=False))
    all_residuals = np.concatenate(residuals)
    return {
        "kind": "ridge",
        "samples": int(count),
        "feature_names": list(names),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": float(target_mean),
        "rmse": float(math.sqrt(squared_error / count)),
        "residual_q10": float(np.quantile(all_residuals, 0.10)),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_streaming_logistic(
    batches: BatchFactory,
    *,
    ridge_lambda: float,
    maximum_iterations: int = 15,
    tolerance: float = 1e-7,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict[str, Any]:
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("logistic penalty must be positive")
    if maximum_iterations < 1 or tolerance <= 0:
        raise ValueError("logistic convergence settings are invalid")
    names = tuple(str(item) for item in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("logistic feature names are empty or duplicated")
    count, means, scales, target_mean = _standardization(batches, names)
    if not 0.0 < target_mean < 1.0:
        raise ValueError("logistic target requires both classes")
    coefficients = np.zeros(len(names), dtype=float)
    intercept = float(math.log(target_mean / (1.0 - target_mean)))
    converged = False
    iterations = 0
    for iteration in range(maximum_iterations):
        gradient_intercept = 0.0
        gradient = -ridge_lambda * coefficients
        h00 = 0.0
        h0x = np.zeros(len(names), dtype=float)
        hxx = ridge_lambda * np.eye(len(names), dtype=float)
        for matrix, target in batches():
            matrix = np.asarray(matrix, dtype=float)
            target = np.asarray(target, dtype=float)
            _validate_batch(matrix, target, names)
            standardized = (matrix - means) / scales
            probability = _sigmoid(intercept + standardized @ coefficients)
            residual = target - probability
            weight = np.clip(probability * (1.0 - probability), 1e-8, None)
            gradient_intercept += float(residual.sum())
            gradient += standardized.T @ residual
            h00 += float(weight.sum())
            h0x += standardized.T @ weight
            hxx += standardized.T @ (standardized * weight[:, None])
        hessian = np.empty(
            (len(names) + 1, len(names) + 1),
            dtype=float,
        )
        hessian[0, 0] = h00
        hessian[0, 1:] = h0x
        hessian[1:, 0] = h0x
        hessian[1:, 1:] = hxx
        gradient_all = np.concatenate(([gradient_intercept], gradient))
        step = np.linalg.solve(hessian, gradient_all)
        # A deterministic trust bound prevents separation from producing a
        # single unstable Newton leap while preserving the frozen fit rule.
        step_norm = float(np.linalg.norm(step))
        if step_norm > 5.0:
            step *= 5.0 / step_norm
        intercept += float(step[0])
        coefficients += step[1:]
        iterations = iteration + 1
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break
    log_loss = 0.0
    brier = 0.0
    for matrix, target in batches():
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        probability = _sigmoid(
            intercept + ((matrix - means) / scales) @ coefficients
        )
        log_loss -= float(
            (
                target * np.log(np.clip(probability, 1e-12, 1.0))
                + (1.0 - target)
                * np.log(np.clip(1.0 - probability, 1e-12, 1.0))
            ).sum()
        )
        brier += float(np.square(target - probability).sum())
    return {
        "kind": "logistic",
        "samples": int(count),
        "feature_names": list(names),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "iterations": int(iterations),
        "converged": bool(converged),
        "log_loss": float(log_loss / count),
        "brier": float(brier / count),
    }


@dataclass(frozen=True)
class FrozenLinearModel:
    kind: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    samples: int
    residual_q10: float | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_feature_names: Sequence[str] | None = None,
    ) -> "FrozenLinearModel":
        kind = str(payload.get("kind", ""))
        if kind not in {"ridge", "logistic"}:
            raise ValueError("unknown frozen action-value model kind")
        names = tuple(str(item) for item in payload.get("feature_names", ()))
        if expected_feature_names is not None and names != tuple(
            expected_feature_names
        ):
            raise ValueError("frozen action-value feature order changed")
        if not names or len(set(names)) != len(names):
            raise ValueError("frozen action-value feature names are invalid")
        means = tuple(float(item) for item in payload.get("means", ()))
        scales = tuple(float(item) for item in payload.get("scales", ()))
        coefficients = tuple(
            float(item) for item in payload.get("coefficients", ())
        )
        if any(
            len(items) != len(names)
            for items in (means, scales, coefficients)
        ):
            raise ValueError("frozen action-value vector width is invalid")
        intercept = float(payload.get("intercept"))
        samples = int(payload.get("samples", 0))
        residual_q10 = (
            None
            if payload.get("residual_q10") is None
            else float(payload["residual_q10"])
        )
        finite = [*means, *scales, *coefficients, intercept]
        if residual_q10 is not None:
            finite.append(residual_q10)
        if (
            samples < 2
            or any(scale <= 0 for scale in scales)
            or not all(math.isfinite(value) for value in finite)
        ):
            raise ValueError("frozen action-value parameters are invalid")
        return cls(
            kind=kind,
            feature_names=names,
            means=means,
            scales=scales,
            coefficients=coefficients,
            intercept=intercept,
            samples=samples,
            residual_q10=residual_q10,
        )

    def predict(self, frame: pd.DataFrame | Mapping[str, float]) -> np.ndarray:
        if isinstance(frame, Mapping):
            matrix = np.asarray(
                [[float(frame[name]) for name in self.feature_names]],
                dtype=float,
            )
        else:
            matrix = frame[list(self.feature_names)].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("action-value inference features are non-finite")
        linear = self.intercept + (
            (matrix - np.asarray(self.means))
            / np.asarray(self.scales)
        ) @ np.asarray(self.coefficients)
        return _sigmoid(linear) if self.kind == "logistic" else linear


@dataclass(frozen=True)
class ActionValueEstimate:
    fill_probability: float
    expiry_probability: float
    expiry_utility_R: float
    conditional_gross_mean_R: float
    conditional_loss_probability: float
    conditional_downside_R: float
    expected_gross_R: float
    conditional_cost_R: float | None
    cost_source: str
    expected_net_R: float | None


@dataclass(frozen=True)
class FrozenActionValueArtifact:
    version: str
    fingerprint: str
    protocol_hash: str
    runtime_semantics_hash: str
    status: str
    flat_action_models: Mapping[
        str,
        Mapping[str, FrozenLinearModel],
    ]
    gross_delta_models: Mapping[str, FrozenLinearModel]
    net_delta_models: Mapping[str, FrozenLinearModel]
    position_models: Mapping[str, FrozenLinearModel]
    release_gates: Mapping[str, Any]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_protocol_hash: str,
    ) -> "FrozenActionValueArtifact":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("action-value artifact root must be an object")
        if payload.get("status") != "ready":
            raise ValueError("action-value artifact is fail-closed/unavailable")
        if payload.get("action_value_code_hash") != action_value_code_fingerprint():
            raise ValueError("action-value artifact code binding is stale")
        if payload.get("base_model_code_hash") != model_code_fingerprint():
            raise ValueError("action-value artifact base-model binding is stale")
        if (
            payload.get("training_pipeline_hash")
            != action_value_pipeline_fingerprint()
        ):
            raise ValueError(
                "action-value artifact training-pipeline binding is stale"
            )
        if payload.get("action_clock_protocol_hash") != expected_protocol_hash:
            raise ValueError("action-value artifact protocol binding is stale")
        models = payload.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("action-value artifact omits models")
        flat_raw = models.get("flat_actions")
        delta_raw = models.get("incremental_deltas")
        position_raw = models.get("position_actions")
        if (
            not isinstance(flat_raw, Mapping)
            or not isinstance(delta_raw, Mapping)
            or set(delta_raw) != {"gross", "net"}
            or not all(
                isinstance(delta_raw.get(family), Mapping)
                for family in ("gross", "net")
            )
            or not isinstance(
                position_raw,
                Mapping,
            )
        ):
            raise ValueError("action-value artifact omits action families")
        expected_flat = set(FLAT_ACTIONS) - {"abstain"}
        if set(flat_raw) != expected_flat:
            raise ValueError("action-value artifact flat model family changed")
        expected_deltas = set(FLAT_ACTIONS) - {"enter_now"}
        if any(
            set(delta_raw[family]) != expected_deltas
            for family in ("gross", "net")
        ):
            raise ValueError("action-value artifact delta family changed")
        if set(position_raw) != set(POSITION_ACTIONS):
            raise ValueError("action-value artifact position family changed")
        gates = payload.get("release_gates")
        if (
            not isinstance(gates, Mapping)
            or not gates
            or not all(value is True for value in gates.values())
        ):
            raise ValueError("action-value artifact release gates are not all passed")
        for action_models in flat_raw.values():
            if (
                not isinstance(action_models, Mapping)
                or set(action_models)
                != {
                    "fill",
                    "conditional_gross",
                    "conditional_cost",
                    "conditional_loss",
                }
            ):
                raise ValueError("flat action model components changed")
        runtime_semantics_hash = str(
            payload.get("runtime_semantics_hash", "")
        )
        if (
            len(runtime_semantics_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in runtime_semantics_hash
            )
        ):
            raise ValueError("action-value runtime-semantics binding is invalid")
        return cls(
            version=str(payload.get("calibration_version", "")),
            fingerprint=hashlib.sha256(raw).hexdigest(),
            protocol_hash=expected_protocol_hash,
            runtime_semantics_hash=runtime_semantics_hash,
            status="ready",
            flat_action_models={
                str(action): {
                    str(family): FrozenLinearModel.from_mapping(
                        value,
                        expected_feature_names=FEATURE_NAMES,
                    )
                    for family, value in action_models.items()
                }
                for action, action_models in flat_raw.items()
            },
            gross_delta_models={
                str(key): FrozenLinearModel.from_mapping(
                    value,
                    expected_feature_names=FEATURE_NAMES,
                )
                for key, value in delta_raw["gross"].items()
            },
            net_delta_models={
                str(key): FrozenLinearModel.from_mapping(
                    value,
                    expected_feature_names=FEATURE_NAMES,
                )
                for key, value in delta_raw["net"].items()
            },
            position_models={
                str(key): FrozenLinearModel.from_mapping(
                    value,
                    expected_feature_names=POSITION_FEATURE_NAMES,
                )
                for key, value in position_raw.items()
            },
            release_gates=dict(gates),
        )

    def estimate(
        self,
        features: Mapping[str, float],
        *,
        action_id: str,
    ) -> ActionValueEstimate:
        action_models = self.flat_action_models.get(action_id)
        if action_models is None or set(action_models) != {
            "fill",
            "conditional_gross",
            "conditional_cost",
            "conditional_loss",
        }:
            raise ValueError("flat action model family is unavailable")
        fill_model = action_models["fill"]
        gross_model = action_models["conditional_gross"]
        cost_model = action_models["conditional_cost"]
        loss_model = action_models["conditional_loss"]
        fill = float(fill_model.predict(features)[0])
        gross = float(gross_model.predict(features)[0])
        loss = float(loss_model.predict(features)[0])
        residual_q10 = gross_model.residual_q10
        if residual_q10 is None:
            raise ValueError("conditional gross model has no downside residual")
        downside = gross + residual_q10
        expected_gross = fill * gross
        mbo_available = float(features["mbo_available"]) >= 0.5
        if not mbo_available:
            conditional_cost = None
            cost_source = "missing_mbo"
        elif action_id in {"enter_now", "wait_better_price"}:
            conditional_cost = float(
                features["expected_round_trip_cost_R"]
            )
            cost_source = "current_causal_mbo"
        else:
            conditional_cost = max(
                0.0,
                float(cost_model.predict(features)[0]),
            )
            cost_source = "frozen_future_submission_cost_model"
        expected_net = (
            expected_gross - fill * conditional_cost
            if conditional_cost is not None
            else None
        )
        return ActionValueEstimate(
            fill_probability=fill,
            expiry_probability=float(1.0 - fill),
            expiry_utility_R=0.0,
            conditional_gross_mean_R=gross,
            conditional_loss_probability=loss,
            conditional_downside_R=float(downside),
            expected_gross_R=float(expected_gross),
            conditional_cost_R=conditional_cost,
            cost_source=cost_source,
            expected_net_R=(
                None if expected_net is None else float(expected_net)
            ),
        )


def _position_features(
    observation: MarketObservation,
    belief: MarketBelief,
    account: AccountState,
    *,
    action_id: str,
    protection_candidate: StructuralLevel | None = None,
) -> dict[str, float]:
    if action_id not in POSITION_ACTIONS:
        raise ValueError("unregistered position action")
    position = account.position
    if position is None:
        raise ValueError("position features require an open position")
    original_risk = abs(
        position.entry_price - position.original_invalidation.price
    )
    if original_risk <= 0:
        raise ValueError("position has invalid original risk")
    key = f"{position.playbook.value}:{position.direction.value}"
    hypothesis = belief.hypotheses.get(key)
    sign = position.direction.sign
    m5 = observation.frame(Timeframe.M5).metrics
    m1 = observation.frame(Timeframe.M1).metrics
    protection = protection_candidate
    applied_stop = (
        protection.price
        if action_id == "protect" and protection is not None
        else position.current_stop
    )
    values = {
        "position_belief_probability": (
            0.5 if hypothesis is None else float(hypothesis.probability)
        ),
        "position_uncertainty": (
            1.0 if hypothesis is None else float(hypothesis.uncertainty)
        ),
        "position_mark_R": float(position.unrealized_R),
        "position_parent_mfe_R": float(position.mfe_R),
        "position_parent_mae_R": float(position.mae_R),
        "position_elapsed_scaled": min(
            1.0,
            max(0.0, position.elapsed_minutes / 240.0),
        ),
        "position_remaining_target_R": max(
            0.0,
            sign
            * (position.primary_target.price - observation.price)
            / original_risk,
        ),
        "position_stop_distance_R": (
            abs(observation.price - applied_stop) / original_risk
        ),
        "position_minutes_to_deadline_scaled": min(
            1.0,
            max(0.0, observation.execution.minutes_to_deadline / 240.0),
        ),
        "position_m5_reacceptance_aligned": sign
        * float(m5["reacceptance_direction"]),
        "position_m1_path_aligned": sign * float(m1["path_sequence"]),
        "position_m1_counter_pressure_against": max(
            0.0,
            -sign * float(m1["counter_pressure"]),
        ),
        "position_m1_trigger_hold_aligned": sign
        * float(m1["trigger_hold_direction"]),
        "position_protection_available": (
            1.0 if protection is not None else 0.0
        ),
        "position_applied_stop_R": sign
        * (applied_stop - position.entry_price)
        / original_risk,
        "position_action_hold": 1.0 if action_id == "hold" else 0.0,
        "position_action_protect": (
            1.0 if action_id == "protect" else 0.0
        ),
        "position_action_exit": 1.0 if action_id == "exit" else 0.0,
    }
    if tuple(values) != POSITION_FEATURE_NAMES or not all(
        math.isfinite(float(value)) for value in values.values()
    ):
        raise ValueError("runtime position features violate protocol")
    return values


class ActionClockValueDecisionLayer(ActionEquivalenceDecisionLayer):
    """Compare frozen decomposed Q values without fitting online."""

    def __init__(
        self,
        action_protocol: ActionEquivalenceProtocol,
        model: FrozenActionValueArtifact,
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
        self.lineage = PlanLineageStore()

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
        if "contract_change_history_reset" in observation.anomalies:
            self.lineage.reset()
        self.lineage.observe(filtered)
        raw = [
            utility
            for utility in UtilityDecisionLayer._flat_utilities(
                self,
                observation,
                filtered,
            )
            if action_utility_is_ready(utility, filtered)
        ]
        groups = [
            group
            for group in equivalent_action_groups(
                raw,
                filtered,
                tick_size=self.protocol.tick_size,
            )
            if group.identity.verb is Action.ENTER
        ]
        output = [
            ActionUtility(
                action=Action.ABSTAIN,
                utility=0.0,
                components={"capital_preservation": 0.0},
                hypothesis_key=None,
                reason="v2.3 abstain preserves capital at exactly 0R",
            )
        ]
        for group in groups:
            representative_key = group.representative.hypothesis_key
            if representative_key is None:
                continue
            hypothesis = filtered.hypotheses[representative_key]
            initial = self.lineage.initial_for(hypothesis)
            estimates: dict[str, ActionValueEstimate] = {}
            features_by_action: dict[str, dict[str, float]] = {}
            for action_id in FLAT_ACTIONS:
                if action_id == "abstain":
                    continue
                features = action_clock_features(
                    group,
                    filtered,
                    observation,
                    initial,
                    action_id=action_id,
                    tick_size=self.protocol.tick_size,
                )
                features_by_action[action_id] = features
                estimates[action_id] = self.model.estimate(
                    features,
                    action_id=action_id,
                )
            enter_features = features_by_action["enter_now"]
            gross_delta_predictions = {
                alternative: float(
                    self.model.gross_delta_models[alternative].predict(
                        enter_features
                    )[0]
                )
                for alternative in (
                    "wait_one_bar",
                    "wait_better_price",
                    "wait_reacceptance",
                    "abstain",
                )
            }
            net_delta_predictions = {
                alternative: float(
                    self.model.net_delta_models[alternative].predict(
                        enter_features
                    )[0]
                )
                for alternative in (
                    "wait_one_bar",
                    "wait_better_price",
                    "wait_reacceptance",
                    "abstain",
                )
            }
            enter = estimates["enter_now"]
            decomposed_consistent = all(
                (
                    enter.expected_net_R is not None
                    and estimates[alternative].expected_net_R is not None
                    and enter.expected_net_R
                    > estimates[alternative].expected_net_R
                )
                for alternative in (
                    "wait_one_bar",
                    "wait_better_price",
                    "wait_reacceptance",
                )
            ) and enter.expected_net_R is not None and enter.expected_net_R > 0.0
            direct_gross_consistent = all(
                value > 0.0 for value in gross_delta_predictions.values()
            )
            direct_net_consistent = all(
                value > 0.0 for value in net_delta_predictions.values()
            )
            enter_available = bool(
                decomposed_consistent
                and direct_gross_consistent
                and direct_net_consistent
                and enter.conditional_downside_R > -1.0
            )
            enter_utility = (
                float(enter.expected_net_R)
                if enter_available and enter.expected_net_R is not None
                else -1.0
            )
            output.append(
                ActionUtility(
                    action=Action.ENTER,
                    utility=enter_utility,
                    components={
                        "fill_probability": enter.fill_probability,
                        "expiry_probability": enter.expiry_probability,
                        "expiry_utility_R": enter.expiry_utility_R,
                        "conditional_gross_mean_R": (
                            enter.conditional_gross_mean_R
                        ),
                        "conditional_loss_probability": (
                            enter.conditional_loss_probability
                        ),
                        "conditional_downside_R": (
                            enter.conditional_downside_R
                        ),
                        "conditional_cost_R": (
                            -1.0
                            if enter.conditional_cost_R is None
                            else enter.conditional_cost_R
                        ),
                        "expected_gross_R": enter.expected_gross_R,
                        "expected_net_R": (
                            -1.0
                            if enter.expected_net_R is None
                            else enter.expected_net_R
                        ),
                        "decomposed_delta_consistent": (
                            1.0 if decomposed_consistent else -1.0
                        ),
                        "direct_gross_delta_consistent": (
                            1.0 if direct_gross_consistent else -1.0
                        ),
                        "direct_net_delta_consistent": (
                            1.0 if direct_net_consistent else -1.0
                        ),
                        "mbo_available": enter_features["mbo_available"],
                    },
                    hypothesis_key=representative_key,
                    reason=(
                        f"frozen {self.model.version} enter_now Q; "
                        f"decomposed_consistent={decomposed_consistent}; "
                        "direct_gross_delta_consistent="
                        f"{direct_gross_consistent}; "
                        "direct_net_delta_consistent="
                        f"{direct_net_consistent}; "
                        f"downside={enter.conditional_downside_R:.3f}R"
                    ),
                )
            )
            wait_alternatives = (
                "wait_one_bar",
                "wait_better_price",
                "wait_reacceptance",
            )

            def _wait_utility(alternative: str) -> float:
                estimate = estimates[alternative]
                return float(
                    estimate.expected_net_R
                    if estimate.expected_net_R is not None
                    else estimate.expected_gross_R
                )

            # The offline action-clock dataset retains all three wait
            # counterfactuals.  At runtime they all map to the single external
            # WAIT instruction, so compare ENTER against the best available
            # wait policy once.  Otherwise two nearly equal wait variants can
            # occupy first and second place and spuriously suppress WAIT via
            # the minimum-advantage rule.
            best_wait = max(wait_alternatives, key=_wait_utility)
            estimate = estimates[best_wait]
            wait_policy_utilities = {
                alternative: _wait_utility(alternative)
                for alternative in wait_alternatives
            }
            output.append(
                ActionUtility(
                    action=Action.WAIT,
                    utility=wait_policy_utilities[best_wait],
                    components={
                        "fill_probability": estimate.fill_probability,
                        "expiry_probability": estimate.expiry_probability,
                        "expiry_utility_R": estimate.expiry_utility_R,
                        "conditional_gross_mean_R": (
                            estimate.conditional_gross_mean_R
                        ),
                        "conditional_loss_probability": (
                            estimate.conditional_loss_probability
                        ),
                        "conditional_downside_R": (
                            estimate.conditional_downside_R
                        ),
                        "conditional_cost_R": (
                            -1.0
                            if estimate.conditional_cost_R is None
                            else estimate.conditional_cost_R
                        ),
                        "expected_gross_R": estimate.expected_gross_R,
                        "expected_net_R": (
                            estimate.expected_gross_R
                            if estimate.expected_net_R is None
                            else estimate.expected_net_R
                        ),
                        "direct_gross_enter_minus_wait_R": (
                            gross_delta_predictions[best_wait]
                        ),
                        "direct_net_enter_minus_wait_R": (
                            net_delta_predictions[best_wait]
                        ),
                        **{
                            f"{alternative}_Q_R": value
                            for alternative, value in wait_policy_utilities.items()
                        },
                    },
                    hypothesis_key=representative_key,
                    reason=(
                        f"frozen {self.model.version} best WAIT policy "
                        f"{best_wait}; all wait counterfactuals remain in the "
                        "offline action-clock evidence"
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
            raise ValueError("position Q requires an open position")
        key = f"{position.playbook.value}:{position.direction.value}"
        if (
            position.elapsed_minutes < 1
            or (position.elapsed_minutes - 1)
            % POSITION_SAMPLING_STRIDE_MINUTES
            != 0
        ):
            return [
                ActionUtility(
                    action=Action.ABSTAIN,
                    utility=float(position.unrealized_R),
                    components={
                        "no_new_instruction_mark_R": float(
                            position.unrealized_R
                        ),
                        "position_action_clock_open": 0.0,
                    },
                    hypothesis_key=key,
                    reason=(
                        "no position-management instruction between the "
                        "preregistered five-minute action clocks"
                    ),
                )
            ]
        available_actions = ["hold", "exit"]
        protection = causal_protection_candidate(position, observation)
        if protection is not None:
            available_actions.append("protect")
        cost_R = (
            observation.execution.expected_round_trip_cost_points
            / (
                2.0
                * abs(
                    position.entry_price
                    - position.original_invalidation.price
                )
            )
            if observation.execution.source == "mbo_reconstructed"
            else 0.0
        )
        output = [
            ActionUtility(
                action=Action.ABSTAIN,
                utility=float(position.unrealized_R),
                components={
                    "no_new_instruction_mark_R": float(position.unrealized_R)
                },
                hypothesis_key=key,
                reason="issue no new position instruction while Q values are close",
            )
        ]
        mapping = {
            "hold": Action.HOLD,
            "protect": Action.PROTECT,
            "exit": Action.EXIT,
        }
        for action_id in available_actions:
            features = _position_features(
                observation,
                belief,
                account,
                action_id=action_id,
                protection_candidate=protection,
            )
            gross_Q = float(
                self.model.position_models[action_id].predict(features)[0]
            )
            net_Q = gross_Q - cost_R
            output.append(
                ActionUtility(
                    action=mapping[action_id],
                    utility=float(net_Q),
                    components={
                        "position_gross_Q_R": gross_Q,
                        "remaining_execution_cost_R": -cost_R,
                        "position_net_Q_R": net_Q,
                    },
                    hypothesis_key=key,
                    reason=(
                        f"frozen {self.model.version} position {action_id} Q"
                    ),
                )
            )
        return output


def build_v2_3_value_engine(
    config_path: str | Path,
    *,
    disabled_playbooks: Sequence[Playbook] = (),
) -> ContinuousSMCEngine:
    source = Path(config_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not str(payload.get("version", "")).startswith("2.3."):
        raise ValueError("v2.3 value engine requires a v2.3 config")
    artifact_path = payload.get("action_clock_value_artifact")
    if not artifact_path:
        raise ValueError("v2.3 runtime config has no frozen value artifact")
    protocol_path = payload.get(
        "action_clock_value_protocol",
        "configs/action_clock_value_protocol_v2_3.json",
    )
    protocol = ActionClockProtocol.from_file(protocol_path)
    action_protocol = ActionEquivalenceProtocol.from_file(
        payload.get(
            "action_equivalence_protocol",
            "configs/action_equivalence_v2_2.json",
        )
    )
    model = FrozenActionValueArtifact.from_file(
        artifact_path,
        expected_protocol_hash=protocol.fingerprint,
    )
    if model.runtime_semantics_hash != runtime_semantics_fingerprint(payload):
        raise ValueError(
            "v2.3 runtime config semantics differ from fitted calibration"
        )
    engine = ContinuousSMCEngine.from_config(source)
    engine.decision = ActionClockValueDecisionLayer(
        action_protocol,
        model,
        engine.decision.config,
        disabled_playbooks=disabled_playbooks,
    )
    return engine


def action_value_code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def runtime_semantics_fingerprint(
    payload: Mapping[str, Any] | str | Path,
) -> str:
    if isinstance(payload, (str, Path)):
        values = json.loads(Path(payload).read_text(encoding="utf-8"))
    else:
        values = json.loads(json.dumps(dict(payload)))
    if not isinstance(values, dict):
        raise ValueError("runtime config semantics require an object")
    values["action_clock_value_artifact"] = None
    values.pop("version", None)
    raw = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def action_value_pipeline_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "smc_trader/action_clock.py",
        "smc_trader/shadow_replay.py",
        "smc_trader/action_value.py",
        "scripts/build_action_clock_episodes.py",
        "scripts/audit_action_clock_data.py",
        "scripts/fit_action_clock_value.py",
        "configs/action_clock_value_protocol_v2_3.json",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "ActionValueEstimate",
    "ActionClockValueDecisionLayer",
    "BatchFactory",
    "FrozenActionValueArtifact",
    "FrozenLinearModel",
    "action_value_code_fingerprint",
    "action_value_pipeline_fingerprint",
    "build_v2_3_value_engine",
    "fit_streaming_logistic",
    "fit_streaming_ridge",
    "runtime_semantics_fingerprint",
]
