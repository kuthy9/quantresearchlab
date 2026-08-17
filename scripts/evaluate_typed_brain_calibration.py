#!/usr/bin/env python3
"""Evaluate one frozen typed Brain artifact on registered validation rows.

This entry point never fits or changes a calibration map.  It applies the
already frozen per-playbook maps to resolved causal rows and reports compact
reliability/discrimination summaries, including the preregistered context
strata carried by those rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_typed_brain_calibration import (  # noqa: E402
    CALIBRATION_UNIT_COLUMNS,
    FITTED_DIMENSIONS,
    _aware_utc,
    _input_files,
    _load_run_bindings,
    _read_rows,
    _resolve,
    _strict_bool,
    _validate_identity_columns,
    _validate_run_window,
    resolve_model_bindings,
)
from smc_trader.artifact_stream import atomic_bytes, sha256_file  # noqa: E402
from smc_trader.calibration import (  # noqa: E402
    TYPED_ACTIVE_PLAYBOOKS,
    TypedBrainCalibrator,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


def _unit_keys(values: pd.DataFrame) -> pd.Series:
    return values.loc[:, CALIBRATION_UNIT_COLUMNS].astype(str).agg(
        lambda row: json.dumps(list(row), separators=(",", ":")),
        axis=1,
    )


def _unit_weights(values: pd.DataFrame) -> pd.Series:
    keys = _unit_keys(values)
    counts = keys.groupby(keys, sort=False).transform("size").astype(float)
    return pd.Series(1.0 / counts.to_numpy(), index=values.index)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_auc(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    positive = float(np.sum(weights[outcomes == 1.0]))
    negative = float(np.sum(weights[outcomes == 0.0]))
    if positive <= 0.0 or negative <= 0.0:
        return None
    order = np.argsort(predictions, kind="stable")
    score = 0.0
    negative_before = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while (
            end < len(order)
            and predictions[order[end]] == predictions[order[index]]
        ):
            end += 1
        members = order[index:end]
        positive_at_score = float(
            np.sum(weights[members][outcomes[members] == 1.0])
        )
        negative_at_score = float(
            np.sum(weights[members][outcomes[members] == 0.0])
        )
        score += positive_at_score * (
            negative_before + 0.5 * negative_at_score
        )
        negative_before += negative_at_score
        index = end
    return float(score / (positive * negative))


def _reliability_bins(
    values: pd.DataFrame,
    *,
    raw_support: Sequence[float],
) -> tuple[list[dict[str, Any]], float, float]:
    support = np.asarray(sorted(set(float(item) for item in raw_support)))
    boundaries = (
        np.asarray([], dtype=float)
        if len(support) < 2
        else (support[:-1] + support[1:]) / 2.0
    )
    assignments = np.searchsorted(
        boundaries,
        values["raw_value"].to_numpy(float),
        side="right",
    )
    bins: list[dict[str, Any]] = []
    total_weight = float(values["_unit_weight"].sum())
    weighted_gap = 0.0
    maximum_gap = 0.0
    for bin_index in sorted(set(int(item) for item in assignments)):
        members = values.iloc[np.flatnonzero(assignments == bin_index)]
        weights = members["_unit_weight"].to_numpy(float)
        prediction = _weighted_mean(
            members["calibrated_value"].to_numpy(float), weights
        )
        outcome = _weighted_mean(
            members["outcome_value"].to_numpy(float), weights
        )
        gap = abs(prediction - outcome)
        mass = float(np.sum(weights))
        weighted_gap += mass * gap
        maximum_gap = max(maximum_gap, gap)
        bins.append(
            {
                "frozen_support_index": bin_index,
                "rows": int(len(members)),
                "unique_causal_units": int(_unit_keys(members).nunique()),
                "effective_unit_mass": mass,
                "mean_raw_value": _weighted_mean(
                    members["raw_value"].to_numpy(float), weights
                ),
                "mean_calibrated_value": prediction,
                "observed_rate": outcome,
                "absolute_gap": gap,
            }
        )
    return bins, weighted_gap / total_weight, maximum_gap


def _metric_payload(
    values: pd.DataFrame,
    *,
    raw_support: Sequence[float],
) -> dict[str, Any]:
    if values.empty:
        return {"status": "no_resolved_rows", "rows": 0}
    weights = values["_unit_weight"].to_numpy(float)
    predictions = values["calibrated_value"].to_numpy(float)
    outcomes = values["outcome_value"].to_numpy(float)
    bins, expected_gap, maximum_gap = _reliability_bins(
        values,
        raw_support=raw_support,
    )
    positive = outcomes == 1.0
    negative = outcomes == 0.0
    positive_mean = (
        None
        if not positive.any()
        else _weighted_mean(predictions[positive], weights[positive])
    )
    negative_mean = (
        None
        if not negative.any()
        else _weighted_mean(predictions[negative], weights[negative])
    )
    clipped = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    return {
        "status": (
            "evaluated"
            if positive.any() and negative.any()
            else "evaluated_insufficient_outcome_variation"
        ),
        "rows": int(len(values)),
        "unique_causal_units": int(_unit_keys(values).nunique()),
        "effective_unit_mass": float(np.sum(weights)),
        "outcome_rate": _weighted_mean(outcomes, weights),
        "mean_calibrated_value": _weighted_mean(predictions, weights),
        "brier_score": _weighted_mean((predictions - outcomes) ** 2, weights),
        "log_loss": _weighted_mean(
            -(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped)),
            weights,
        ),
        "expected_calibration_error": expected_gap,
        "maximum_calibration_gap": maximum_gap,
        "weighted_auc": _weighted_auc(predictions, outcomes, weights),
        "mean_prediction_positive": positive_mean,
        "mean_prediction_negative": negative_mean,
        "prediction_separation": (
            None
            if positive_mean is None or negative_mean is None
            else positive_mean - negative_mean
        ),
        "calibrated_support_levels": int(
            values["calibrated_value"].nunique()
        ),
        "reliability_bins": bins,
    }


def _obstruction_bucket(row: pd.Series) -> str:
    if bool(row["hard_barrier_before_target"]):
        return "hard_barrier_before_target"
    distance = row["obstruction_distance_R"]
    if pd.isna(distance):
        return (
            "no_obstruction"
            if int(row["soft_obstruction_count"]) == 0
            else "soft_obstruction_distance_unknown"
        )
    value = float(distance)
    if value < 0.5:
        return "soft_obstruction_[0,0.5R)"
    if value < 1.0:
        return "soft_obstruction_[0.5R,1R)"
    if value < 2.0:
        return "soft_obstruction_[1R,2R)"
    return "soft_obstruction_[2R,+inf)"


def _strata(values: pd.DataFrame) -> Mapping[str, pd.Series]:
    return {
        "calendar_month": values["sampled_at"].dt.strftime("%Y-%m"),
        "direction": values["direction"].astype(str),
        "global_market_mode": values["global_market_mode"].astype(str),
        "authority_relation": values["authority_relation"].astype(str),
        "authority_rank_gap": values["authority_rank_gap"].astype(str),
        "conflict_role": values["conflict_role"].astype(str),
        "conflict_scope": values["conflict_scope"].astype(str),
        "obstruction": values.apply(_obstruction_bucket, axis=1),
    }


def _stratified_payload(
    values: pd.DataFrame,
    *,
    raw_support: Sequence[float],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for name, projected in _strata(values).items():
        groups: list[dict[str, Any]] = []
        for stratum, indexes in projected.groupby(projected, sort=True).groups.items():
            group = values.loc[indexes]
            groups.append(
                {
                    "value": str(stratum),
                    **_metric_payload(group, raw_support=raw_support),
                }
            )
        output[name] = groups
    return output


def _load_artifact(
    artifact: str | Path,
    *,
    bindings: Mapping[str, Any],
    validation_schema_version: int,
) -> tuple[Path, Mapping[str, Any], TypedBrainCalibrator]:
    path = Path(artifact).resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("frozen typed calibration artifact is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("status") != "ready"
        or payload.get("training_window_role") != "calibration"
        or payload.get("validation_schema_version") != validation_schema_version
        or payload.get("brain_target_protocol_versions")
        != bindings["playbook_schema_versions"]
        or payload.get("rolling_oof_used") is not False
        or payload.get("holdout_used") is not False
    ):
        raise ValueError(
            "artifact is not a current frozen 2022 typed calibration mapping"
        )
    training_manifest_path = Path(str(payload.get("source_run_manifest", "")))
    if not training_manifest_path.is_absolute():
        training_manifest_path = ROOT / training_manifest_path
    if not training_manifest_path.is_file() or training_manifest_path.is_symlink():
        raise ValueError("artifact source calibration run manifest is unavailable")
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    training_model = (
        training_manifest.get("model_config")
        if isinstance(training_manifest, Mapping)
        else None
    )
    training_window = (
        training_manifest.get("window")
        if isinstance(training_manifest, Mapping)
        else None
    )
    if (
        not isinstance(training_model, Mapping)
        or training_model.get("identity") != bindings["model_config_identity"]
        or not isinstance(training_window, Mapping)
        or training_window.get("role") != "calibration"
    ):
        raise ValueError("artifact source calibration run binding is stale")
    calibrator = TypedBrainCalibrator.from_file(
        path,
        expected_registry_hash=bindings["registry_hash"],
    )
    if not calibrator.is_ready or any(
        not set(FITTED_DIMENSIONS).issubset(calibrator.maps.get(playbook, {}))
        for playbook in TYPED_ACTIVE_PLAYBOOKS
    ):
        raise ValueError(
            "artifact does not provide ready four-dimension maps for every "
            "active playbook"
        )
    return path, payload, calibrator


def evaluate_frozen_typed_brain_calibration(
    *,
    row_paths: Sequence[str | Path],
    run_manifest: str | Path,
    artifact: str | Path,
    output: str | Path,
    model_config: str | Path = "configs/model.json",
    validation_protocol: str | Path = "configs/data_splits.json",
) -> dict[str, Any]:
    """Apply, never fit, one frozen artifact on a Brain validation run."""

    files = _input_files(row_paths)
    destination = Path(output).resolve()
    protected_inputs = {
        _resolve(run_manifest),
        _resolve(artifact),
        _resolve(model_config),
        _resolve(validation_protocol),
        *files,
        *(_resolve(path) for path in row_paths),
    }
    if destination in protected_inputs:
        raise ValueError(
            "OOS output must not overwrite an input, artifact, manifest, "
            "model config or validation protocol"
        )
    protocol = load_validation_protocol(validation_protocol)
    bindings = resolve_model_bindings(model_config)
    run_identity = _load_run_bindings(
        run_manifest,
        bindings,
        row_files=files,
    )
    if run_identity["window_role"] != "brain_validation":
        raise ValueError("frozen typed OOS evaluation requires brain_validation rows")
    bound_window = protocol.classify_ohlcv(
        run_identity["window_start"],
        run_identity["window_end_exclusive"],
    )
    if bound_window.role != "brain_validation":
        raise ValueError(
            "Brain validation run manifest does not match the registered split"
        )
    frozen_artifact_path = Path(artifact).resolve()
    if (
        not frozen_artifact_path.is_file()
        or frozen_artifact_path.is_symlink()
    ):
        raise FileNotFoundError("frozen typed calibration artifact is missing")
    artifact_hash = sha256_file(frozen_artifact_path)
    artifact_path, artifact_payload, calibrator = _load_artifact(
        frozen_artifact_path,
        bindings=bindings,
        validation_schema_version=protocol.schema_version,
    )

    frame = _read_rows(files)
    allowed_dimensions = set(FITTED_DIMENSIONS) | {
        "sequence_progress",
        "uncertainty",
    }
    unexpected_dimensions = sorted(
        set(frame["dimension"].astype(str)) - allowed_dimensions
    )
    if unexpected_dimensions:
        raise ValueError(
            "unsupported typed validation dimensions: "
            f"{unexpected_dimensions}"
        )
    _validate_identity_columns(frame)
    if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).eq("").any():
        raise ValueError("sample_id cannot be empty")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Brain validation rows contain duplicate sample_id values")
    frame["censored"] = _strict_bool(frame["censored"], "censored")
    frame["fit_eligible"] = _strict_bool(frame["fit_eligible"], "fit_eligible")
    frame["sampled_at"] = _aware_utc(
        frame["sampled_at"], "sampled_at", nullable=False
    )
    frame["resolved_at"] = _aware_utc(
        frame["resolved_at"], "resolved_at", nullable=False
    )
    frame["deadline"] = _aware_utc(frame["deadline"], "deadline", nullable=True)
    if bool((frame["resolved_at"] < frame["sampled_at"]).any()):
        raise ValueError("validation rows resolve before they were sampled")
    _validate_run_window(frame, run_identity)
    if set(frame["playbook"].astype(str)) - {
        item.value for item in TYPED_ACTIVE_PLAYBOOKS
    }:
        raise ValueError("Brain validation rows contain an unsupported playbook")
    if set(frame["direction"].astype(str)) - {"long", "short"}:
        raise ValueError("direction must be long or short")
    immediate = frame["dimension"].isin(
        {"sequence_progress", "uncertainty"}
    )
    if bool(frame.loc[immediate, "fit_eligible"].any()):
        raise ValueError(
            "sequence_progress and uncertainty cannot be OOS probability rows"
        )
    if bool((frame["fit_eligible"] & frame["censored"]).any()):
        raise ValueError("censored validation rows cannot be fit eligible")
    if not set(frame["target_deadline_kind"].astype(str)).issubset(
        {"thesis_deadline", "entry_deadline", "plan_deadline"}
    ):
        raise ValueError("target_deadline_kind is invalid")
    fitted = frame["dimension"].isin(FITTED_DIMENSIONS)
    expected_deadline_kind = frame["dimension"].map(
        {
            "thesis_strength": "thesis_deadline",
            "location_quality": "entry_deadline",
            "entry_readiness": "entry_deadline",
            "delivery_quality": "plan_deadline",
        }
    )
    if bool((fitted & frame["deadline"].isna()).any()) or bool(
        (fitted & frame["target_deadline_kind"].ne(expected_deadline_kind)).any()
    ):
        raise ValueError("validation targets lack their registered frozen deadline")
    if bool((fitted & (frame["deadline"] <= frame["sampled_at"])).any()) or bool(
        (fitted & (frame["resolved_at"] > frame["deadline"])).any()
    ):
        raise ValueError("validation target clocks violate their frozen deadline")
    if bool(
        (
            frame["fit_eligible"]
            & frame["dimension"].eq("delivery_quality")
            & frame["hard_barrier_before_target"]
        ).any()
    ):
        raise ValueError("hard-barrier delivery rows cannot enter OOS metrics")

    eligible = frame.loc[
        frame["fit_eligible"]
        & ~frame["censored"]
        & frame["dimension"].isin(FITTED_DIMENSIONS)
    ].copy()
    if eligible.empty:
        raise ValueError("Brain validation run has no resolved fitted-dimension rows")
    if eligible["outcome_value"].isna().any():
        raise ValueError("fit-eligible validation rows require resolved outcomes")
    numeric = eligible.loc[:, ["raw_value", "outcome_value"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(float)).all()
        or ((numeric < 0.0) | (numeric > 1.0)).any().any()
    ):
        raise ValueError("validation raw values and outcomes must lie in [0, 1]")
    eligible.loc[:, "raw_value"] = numeric["raw_value"]
    eligible.loc[:, "outcome_value"] = numeric["outcome_value"]
    eligible["_unit_weight"] = 0.0
    eligible["calibrated_value"] = 0.0

    playbooks: dict[str, Any] = {}
    for playbook in TYPED_ACTIVE_PLAYBOOKS:
        dimensions: dict[str, Any] = {}
        for dimension in FITTED_DIMENSIONS:
            mask = eligible["playbook"].eq(playbook.value) & eligible[
                "dimension"
            ].eq(dimension)
            values = eligible.loc[mask].copy()
            if values.empty:
                dimensions[dimension] = {"status": "no_resolved_rows", "rows": 0}
                continue
            weights = _unit_weights(values)
            eligible.loc[values.index, "_unit_weight"] = weights
            mapped = values["raw_value"].map(
                lambda raw: calibrator.apply(playbook, dimension, float(raw))
            )
            eligible.loc[values.index, "calibrated_value"] = mapped
            values["_unit_weight"] = weights
            values["calibrated_value"] = mapped
            reliability_map = calibrator.maps[playbook][dimension]
            support = [point.raw_value for point in reliability_map.points]
            dimensions[dimension] = {
                "overall": _metric_payload(values, raw_support=support),
                "strata": _stratified_payload(values, raw_support=support),
            }
        playbooks[playbook.value] = {"dimensions": dimensions}

    uncertainty = frame.loc[frame["dimension"].eq("uncertainty")].copy()
    uncertainty_payload: dict[str, Any] = {
        "status": "descriptive_not_fitted",
        "rows": int(len(uncertainty)),
        "components": {},
    }
    if not uncertainty.empty:
        for name in (
            "uncertainty_conflict",
            "uncertainty_required_evidence_missing",
            "uncertainty_authority_missing",
            "uncertainty_graph_ambiguity",
            "uncertainty_total",
        ):
            values = pd.to_numeric(uncertainty[name], errors="coerce")
            uncertainty_payload["components"][name] = {
                "mean": float(values.mean()),
                "minimum": float(values.min()),
                "median": float(values.median()),
                "maximum": float(values.max()),
            }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen_oos_evaluation_complete",
        "method": {
            "artifact_application": "frozen_map_interpolation_only",
            "mapping_fitted": False,
            "threshold_search": False,
            "context_strata_fitted": False,
            "revision_weighting": "equal_total_weight_per_causal_owner",
            "reliability_bins": "frozen_artifact_raw_support_midpoints",
            "sequence_progress": "deterministic_not_probability_evaluated",
            "uncertainty": "descriptive_components_not_probability_fitted",
        },
        "calibration_artifact": {
            "path": str(artifact_path),
            "sha256": artifact_hash,
            "calibration_version": calibrator.version,
            "training_start": artifact_payload.get("training_start"),
            "training_end": artifact_payload.get("training_end"),
        },
        "validation_run_manifest": str(Path(run_identity["path"]).resolve()),
        "validation_window_role": run_identity["window_role"],
        "validation_start": eligible["sampled_at"].min().isoformat(),
        "validation_end": eligible["resolved_at"].max().isoformat(),
        "resolved_rows": int(len(eligible)),
        "playbooks": playbooks,
        "uncertainty_distribution": uncertainty_payload,
        "fitted": False,
        "artifact_modified": False,
        "pnl_labels_used": False,
        "mbo_used": False,
    }
    if sha256_file(artifact_path) != artifact_hash:
        raise RuntimeError(
            "frozen calibration artifact changed before OOS output commit"
        )
    atomic_bytes(
        destination,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
    )
    if sha256_file(artifact_path) != artifact_hash:
        raise RuntimeError("frozen calibration artifact changed during OOS evaluation")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument(
        "--validation-protocol",
        default="configs/data_splits.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_frozen_typed_brain_calibration(
        row_paths=args.rows,
        run_manifest=args.run_manifest,
        artifact=args.artifact,
        output=args.output,
        model_config=args.model_config,
        validation_protocol=args.validation_protocol,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "resolved_rows": payload["resolved_rows"],
                "calibration_version": payload["calibration_artifact"][
                    "calibration_version"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
