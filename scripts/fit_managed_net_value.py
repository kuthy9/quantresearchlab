#!/usr/bin/env python3
"""Fit and temporally validate the preregistered v2.2 managed-value model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_managed_net_calibration import (  # noqa: E402
    calibration_pipeline_hash_is_accepted,
)
from scripts.build_managed_net_episodes import (  # noqa: E402
    episode_builder_code_fingerprint,
)
from smc_trader.action_equivalence import (  # noqa: E402
    ActionEquivalenceProtocol,
)
from smc_trader.managed_net_value import (  # noqa: E402
    FEATURE_NAMES,
    fit_fixed_ridge,
    managed_net_value_code_fingerprint,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def managed_net_fit_code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--calibration-version",
        default="2.2.0-managed-net-2023.1",
    )
    parser.add_argument(
        "--config",
        default="configs/model_v2_2_policy_base.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2_2.json",
    )
    parser.add_argument(
        "--managed-net-value-protocol",
        default="configs/managed_net_value_protocol_v2_2.json",
    )
    parser.add_argument(
        "--action-equivalence-protocol",
        default="configs/action_equivalence_v2_2.json",
    )
    return parser.parse_args()


def _temporal_oof(
    frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    *,
    ridge_lambda: float,
    minimum_fold: int,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for index, fold in enumerate(folds):
        train_end = pd.Timestamp(fold["train_end_exclusive"]).tz_convert(
            "UTC"
        )
        validate_start = pd.Timestamp(fold["validate_start"]).tz_convert(
            "UTC"
        )
        validate_end = pd.Timestamp(
            fold["validate_end_exclusive"]
        ).tz_convert("UTC")
        train = frame.loc[frame["decision_time"] < train_end]
        validate = frame.loc[
            (frame["decision_time"] >= validate_start)
            & (frame["decision_time"] < validate_end)
        ]
        if len(train) < minimum_fold or len(validate) < minimum_fold:
            fold_rows.append(
                {
                    "fold": index + 1,
                    "train_episodes": int(len(train)),
                    "validation_episodes": int(len(validate)),
                    "status": "insufficient_train_or_validation_episodes",
                }
            )
            continue
        fitted = fit_fixed_ridge(
            train[list(FEATURE_NAMES)].to_numpy(float),
            train["managed_gross_R_clipped"].to_numpy(float),
            ridge_lambda=ridge_lambda,
        )
        validation_matrix = validate[list(FEATURE_NAMES)].to_numpy(float)
        standardized = (
            validation_matrix - fitted["means"]
        ) / fitted["scales"]
        predicted = (
            float(fitted["intercept"])
            + standardized @ fitted["coefficients"]
        )
        actual = validate["managed_gross_R_clipped"].to_numpy(float)
        baseline = np.full(len(validate), float(fitted["intercept"]))
        predictions.append(predicted)
        labels.append(actual)
        baselines.append(baseline)
        fold_rows.append(
            {
                "fold": index + 1,
                "train_episodes": int(len(train)),
                "validation_episodes": int(len(validate)),
                "prediction_mean_R": float(predicted.mean()),
                "realized_mean_R": float(actual.mean()),
                "mse": float(np.mean((actual - predicted) ** 2)),
                "baseline_mse": float(np.mean((actual - baseline) ** 2)),
                "status": "evaluated",
            }
        )
    if not predictions:
        return {
            "episodes": 0,
            "folds": fold_rows,
            "spearman": float("nan"),
            "mse": float("inf"),
            "baseline_mse": float("inf"),
            "positive_bucket_episodes": 0,
            "positive_bucket_realized_mean_R": float("nan"),
        }
    predicted = np.concatenate(predictions)
    actual = np.concatenate(labels)
    baseline = np.concatenate(baselines)
    spearman = pd.Series(predicted).corr(
        pd.Series(actual),
        method="spearman",
    )
    positive = predicted > 0.0
    return {
        "episodes": int(len(actual)),
        "folds": fold_rows,
        "spearman": float(spearman),
        "mse": float(np.mean((actual - predicted) ** 2)),
        "baseline_mse": float(np.mean((actual - baseline) ** 2)),
        "prediction_mean_R": float(predicted.mean()),
        "realized_mean_R": float(actual.mean()),
        "positive_bucket_episodes": int(positive.sum()),
        "positive_bucket_realized_mean_R": (
            float(actual[positive].mean()) if positive.any() else float("nan")
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite managed-net artifact")
    episode_path = Path(args.episodes)
    manifest_path = Path(args.episode_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = Path(args.managed_net_value_protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validation = load_validation_protocol(args.validation_protocol)
    action_protocol = ActionEquivalenceProtocol.from_file(
        args.action_equivalence_protocol
    )
    replay_pipeline_hash = str(manifest.get("calibration_pipeline_hash", ""))
    if not calibration_pipeline_hash_is_accepted(replay_pipeline_hash):
        raise ValueError("managed-net replay pipeline lineage is stale")
    bindings = {
        "episode_table_sha256": _sha256(episode_path),
        "validation_protocol_hash": validation.fingerprint,
        "action_equivalence_protocol_hash": action_protocol.fingerprint,
        "managed_net_value_protocol_hash": _sha256(protocol_path),
        "calibration_pipeline_hash": replay_pipeline_hash,
        "managed_net_value_code_hash": managed_net_value_code_fingerprint(),
        "policy_base_config_hash": _sha256(args.config),
        "episode_builder_code_hash": episode_builder_code_fingerprint(),
        "future_path_used": False,
        "holdout_used": False,
    }
    for field, expected in bindings.items():
        if manifest.get(field) != expected:
            raise ValueError(f"managed-net episode manifest {field} is stale")
    policy_variant = str(manifest.get("policy_variant", ""))
    if policy_variant not in {
        "all_three",
        "without_failed_auction_value_return",
        "without_liquidity_sweep_reversal",
    }:
        raise ValueError("managed-net episodes have an unregistered policy variant")
    episodes = pd.read_parquet(episode_path)
    required = {
        "fit_eligible",
        "decision_time",
        "managed_gross_R_clipped",
        "evidence_state_json",
        *FEATURE_NAMES,
    }
    missing = sorted(required - set(episodes))
    if missing:
        raise ValueError(f"managed-net episodes omit fields: {missing}")
    frame = episodes.loc[episodes["fit_eligible"].astype(bool)].copy()
    if len(frame) != int(manifest["fit_eligible_resolved_actions"]):
        raise ValueError("managed-net eligible episode count is not conserved")
    frame["decision_time"] = pd.to_datetime(
        frame["decision_time"],
        utc=True,
        errors="coerce",
    )
    if frame["decision_time"].isna().any():
        raise ValueError("managed-net episodes contain invalid clocks")
    matrix = frame[list(FEATURE_NAMES)].to_numpy(float)
    labels = frame["managed_gross_R_clipped"].to_numpy(float)
    if not np.isfinite(matrix).all() or not np.isfinite(labels).all():
        raise ValueError("managed-net fit inputs contain non-finite values")

    fit_protocol = protocol["fit"]
    gates = protocol["time_out_of_sample_gates"]
    folds = fit_protocol["temporal_folds"]
    ridge_lambda = float(fit_protocol["ridge_lambda"])
    minimum_fold = int(gates["minimum_each_fold_episodes"])
    primary = _temporal_oof(
        frame,
        folds,
        ridge_lambda=ridge_lambda,
        minimum_fold=minimum_fold,
    )
    primary["eligible_actions"] = int(len(frame))
    primary["unique_action_keys"] = int(frame["action_key"].nunique())
    fold_counts_ok = all(
        row.get("status") == "evaluated"
        for row in primary["folds"]
    )
    gate_results = {
        "minimum_oof_episodes": (
            int(primary["episodes"]) >= int(gates["minimum_oof_episodes"])
        ),
        "minimum_each_fold_episodes": fold_counts_ok,
        "spearman_prediction_to_label_strictly_positive": (
            math.isfinite(float(primary["spearman"]))
            and float(primary["spearman"]) > 0.0
        ),
        "mse_must_beat_training_mean_baseline": (
            float(primary["mse"]) < float(primary["baseline_mse"])
        ),
        "positive_value_bucket_must_have_positive_realized_mean": (
            int(primary["positive_bucket_episodes"]) > 0
            and math.isfinite(
                float(primary["positive_bucket_realized_mean_R"])
            )
            and float(primary["positive_bucket_realized_mean_R"]) > 0.0
        ),
    }
    ready = all(gate_results.values())
    artifact: dict[str, Any] = {
        "calibration_version": args.calibration_version,
        "status": "ready" if ready else "unavailable",
        **bindings,
        "managed_net_fit_code_hash": managed_net_fit_code_fingerprint(),
        "policy_variant": policy_variant,
        "disabled_playbooks": list(manifest.get("disabled_playbooks", ())),
        "feature_names": list(FEATURE_NAMES),
        "fit_method": {
            "model": fit_protocol["model"],
            "ridge_lambda": ridge_lambda,
            "hyperparameter_search": False,
            "threshold_or_pnl_search": False,
        },
        "oof_metrics": {
            key: value
            for key, value in primary.items()
            if key != "folds"
        },
        "temporal_folds": primary["folds"],
        "gate_results": gate_results,
        "leave_one_playbook_out": {
            "method": (
                "separate full decision-and-risk replay per preregistered "
                "policy variant; this artifact contains one variant only"
            ),
            "current_variant": policy_variant,
            "cross_variant_comparison_pending": True,
        },
        "revealed_june_july_used_for_fit_or_selection": False,
        "sealed_holdout_read": False,
    }
    if ready:
        fitted = fit_fixed_ridge(
            matrix,
            labels,
            ridge_lambda=ridge_lambda,
        )
        artifact["model"] = {
            "means": fitted["means"].tolist(),
            "scales": fitted["scales"].tolist(),
            "coefficients": fitted["coefficients"].tolist(),
            "intercept": float(fitted["intercept"]),
            "residual_rmse_R": float(fitted["residual_rmse_R"]),
        }
        artifact["support"] = {
            "minimum_z": fitted["minimum_z"].tolist(),
            "maximum_z": fitted["maximum_z"].tolist(),
        }
    else:
        artifact["reason"] = (
            "one or more preregistered temporal out-of-sample gates failed; "
            "managed enter value remains unavailable"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _json_safe(artifact),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(gate_results), sort_keys=True))


if __name__ == "__main__":
    main()
