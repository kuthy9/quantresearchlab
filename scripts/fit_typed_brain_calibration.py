#!/usr/bin/env python3
"""Fit the v4 typed Brain maps from resolved causal recorder rows.

This is intentionally a small, one-purpose fitter.  It does not inspect
actions, PnL, MFE/MAE, MBO, or holdout data, and it does not search thresholds.
Four future-resolved quality dimensions use fixed quantile bins plus weighted
PAVA.  Sequence progress remains a deterministic state-machine field and
uncertainty remains the registered contemporaneous conflict/missing-authority
formula, represented by an explicit ready identity map for artifact loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.calibration import (  # noqa: E402
    CalibrationError,
    TYPED_ACTIVE_PLAYBOOKS,
    TYPED_PARKED_PLAYBOOKS,
    model_code_fingerprint,
    monotone_reliability_points,
)
from smc_trader.engine import _configured_primitive_protocol_hashes  # noqa: E402
from smc_trader.model import Playbook  # noqa: E402
from smc_trader.playbook_registry import load_playbook_registry  # noqa: E402
from smc_trader.scene_graph import (  # noqa: E402
    brain_input_contract_hash,
    parse_scale_specs,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


FITTED_DIMENSIONS = (
    "thesis_strength",
    "location_quality",
    "entry_readiness",
    "delivery_quality",
)
SEQUENCE_DIMENSION = "sequence_progress"
UNCERTAINTY_DIMENSION = "uncertainty"
UNCERTAINTY_FORMULA_VERSION = (
    "4.0.0-contemporaneous-conflict-missing-authority.2"
)
UNCERTAINTY_FORMULA = (
    "market uncertainty is the bounded contemporaneous combination of "
    "conflicting causal evidence, missing required market evidence, and "
    "missing semantic authority; execution, spread, MBO availability, PnL, "
    "and future path outcomes are excluded"
)
FORBIDDEN_ECONOMIC_COLUMNS = frozenset(
    {
        "pnl",
        "profit",
        "gross_R",
        "net_R",
        "mfe_R",
        "mae_R",
        "realized_R",
        "action_utility",
    }
)
REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "hypothesis_key",
        "scene_hypothesis_id",
        "competing_scene_hypothesis_ids",
        "context_root_ids",
        "scene_revision_id",
        "playbook",
        "direction",
        "dimension",
        "sampled_at",
        "resolved_at",
        "raw_value",
        "outcome_value",
        "resolution",
        "censored",
        "fit_eligible",
        "protocol_version",
        "protocol_hash",
        "registry_hash",
        "model_code_hash",
        "config_hash",
        "primitive_protocol_hashes",
        "brain_input_contract_hash",
        "liquidity_route_id",
        "context_draw_id",
        "intermediate_liquidity_ids",
        "primary_deliverable_target_id",
        "terminal_draw_id",
        "path_blocker_ids",
        "source_path_ids",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: str | Path) -> Path:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = ROOT / source
    return source.resolve()


def _input_files(values: Sequence[str | Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for value in values:
        source = _resolve(value)
        if source.is_dir():
            files.extend(sorted(source.glob("*.parquet")))
            files.extend(sorted(source.glob("*.jsonl")))
        else:
            files.append(source)
    unique = tuple(dict.fromkeys(files))
    if not unique:
        raise ValueError("no Brain calibration recorder shards were found")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Brain calibration shards do not exist: {missing}")
    unsupported = [
        str(path) for path in unique if path.suffix.lower() not in {".parquet", ".jsonl"}
    ]
    if unsupported:
        raise ValueError(f"unsupported Brain calibration shard types: {unsupported}")
    return unique


def _read_rows(files: Sequence[Path]) -> pd.DataFrame:
    frames = [
        (
            pd.read_parquet(path)
            if path.suffix.lower() == ".parquet"
            else pd.read_json(path, lines=True)
        )
        for path in files
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise ValueError("Brain calibration recorder shards contain no rows")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Brain calibration rows omit fields: {missing}")
    forbidden = sorted(FORBIDDEN_ECONOMIC_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(
            "typed Brain calibration rows must not contain economic labels: "
            f"{forbidden}"
        )
    return frame


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    valid = series.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not bool(valid.all()):
        raise ValueError(f"{name} must contain only booleans")
    return series.astype(bool)


def _unique_text(frame: pd.DataFrame, field: str) -> str:
    values = set(frame[field].dropna().astype(str))
    if len(values) != 1 or not next(iter(values), "").strip():
        raise ValueError(f"Brain calibration rows mix or omit {field}")
    return next(iter(values))


def _validate_identity_columns(frame: pd.DataFrame) -> None:
    for field in (
        "hypothesis_key",
        "scene_hypothesis_id",
        "scene_revision_id",
    ):
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"{field} cannot be empty")
    for field in (
        "competing_scene_hypothesis_ids",
        "context_root_ids",
        "intermediate_liquidity_ids",
        "path_blocker_ids",
        "source_path_ids",
    ):
        for raw in frame[field]:
            try:
                values = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{field} must be canonical JSON") from error
            if (
                not isinstance(values, list)
                or any(not isinstance(item, str) or not item for item in values)
                or len(values) != len(set(values))
                or json.dumps(values, separators=(",", ":")) != raw
            ):
                raise ValueError(f"{field} identities are invalid")


def _aware_utc(series: pd.Series, field: str, *, nullable: bool) -> pd.Series:
    if not nullable and series.isna().any():
        raise ValueError(f"{field} cannot be null")
    for value in series.dropna():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} contains an invalid timestamp") from error
        if timestamp.tzinfo is None:
            raise ValueError(f"{field} timestamps must be timezone aware")
    return pd.to_datetime(series, errors="coerce", utc=True)


def resolve_model_bindings(model_config: str | Path) -> dict[str, Any]:
    """Resolve the exact registry, code, config, and primitive identities."""

    config_path = _resolve(model_config)
    raw = config_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("model config root must be an object")
    if payload.get("calibration_artifact") not in (None, ""):
        raise ValueError(
            "typed calibration must be fit from an identity/unvalidated model config"
        )
    registry_path = payload.get("playbook_registry")
    if not isinstance(registry_path, str) or not registry_path.strip():
        raise ValueError("model config omits playbook_registry")
    registry = load_playbook_registry(registry_path)
    for playbook in TYPED_PARKED_PLAYBOOKS:
        if "parked" not in registry.for_playbook(playbook).status:
            raise ValueError(f"{playbook.value} must remain parked during calibration")
    observer = payload.get("observer")
    if not isinstance(observer, Mapping):
        raise ValueError("model config omits observer protocol bindings")
    return {
        "config_path": config_path,
        "config_hash": hashlib.sha256(raw).hexdigest(),
        "registry_hash": registry.fingerprint,
        "registry_schema_version": registry.schema_version,
        "playbook_schema_versions": {
            playbook.value: registry.for_playbook(playbook).schema_version
            for playbook in TYPED_ACTIVE_PLAYBOOKS
        },
        "model_code_hash": model_code_fingerprint(),
        "primitive_protocol_hashes": _configured_primitive_protocol_hashes(observer),
        "brain_input_contract_hash": brain_input_contract_hash(
            parse_scale_specs(payload.get("scales"))
        ),
    }


def _validate_bindings(frame: pd.DataFrame, bindings: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "registry_hash": _unique_text(frame, "registry_hash"),
        "model_code_hash": _unique_text(frame, "model_code_hash"),
        "config_hash": _unique_text(frame, "config_hash"),
        "protocol_hash": _unique_text(frame, "protocol_hash"),
        "brain_input_contract_hash": _unique_text(
            frame,
            "brain_input_contract_hash",
        ),
    }
    for field in ("registry_hash", "model_code_hash", "config_hash"):
        if observed[field] != bindings[field]:
            raise ValueError(f"Brain calibration row {field} is stale")
    if observed["protocol_hash"] != bindings["registry_hash"]:
        raise ValueError("Brain calibration row protocol_hash is stale")
    if (
        observed["brain_input_contract_hash"]
        != bindings["brain_input_contract_hash"]
    ):
        raise ValueError("Brain calibration row input contract hash is stale")

    protocol_versions: dict[str, str] = {}
    for playbook, expected in bindings["playbook_schema_versions"].items():
        expected_text = str(expected)
        values = set(
            frame.loc[frame["playbook"].eq(playbook), "protocol_version"]
            .dropna()
            .astype(str)
        )
        if values != {expected_text}:
            raise ValueError(
                f"Brain calibration row protocol_version is stale for {playbook}"
            )
        protocol_versions[playbook] = expected_text

    raw_values = frame["primitive_protocol_hashes"].dropna()
    if len(raw_values) != len(frame):
        raise ValueError("primitive_protocol_hashes cannot be null")
    normalized: set[str] = set()
    for raw in raw_values:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, Mapping):
            raise ValueError("primitive_protocol_hashes must be a mapping or JSON")
        normalized.add(
            json.dumps(
                {str(key): str(item) for key, item in value.items()},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    expected = json.dumps(
        dict(bindings["primitive_protocol_hashes"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    if normalized != {expected}:
        raise ValueError("Brain calibration row primitive protocol hashes are stale")
    observed["protocol_versions"] = protocol_versions
    return observed


def _dimension_payload(
    values: pd.DataFrame,
    *,
    playbook: Playbook,
    dimension: str,
    bins: int,
    minimum_bin_samples: int,
    minimum_dimension_samples: int,
) -> dict[str, Any]:
    if len(values) < minimum_dimension_samples:
        raise CalibrationError(
            f"{playbook.value}.{dimension} has {len(values)} eligible samples; "
            f"at least {minimum_dimension_samples} are required"
        )
    raw = pd.to_numeric(values["raw_value"], errors="raise").to_numpy(float)
    outcome = pd.to_numeric(values["outcome_value"], errors="raise").to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(outcome).all():
        raise CalibrationError(f"{playbook.value}.{dimension} contains non-finite values")
    if ((raw < 0.0) | (raw > 1.0)).any() or (
        (outcome < 0.0) | (outcome > 1.0)
    ).any():
        raise CalibrationError(
            f"{playbook.value}.{dimension} values must lie in [0, 1]"
        )
    if len(np.unique(raw)) < 2:
        raise CalibrationError(
            f"{playbook.value}.{dimension} has fewer than two unique raw values"
        )
    if len(np.unique(outcome)) < 2:
        raise CalibrationError(
            f"{playbook.value}.{dimension} has no resolved outcome variation"
        )
    points = monotone_reliability_points(
        raw,
        outcome,
        bins=bins,
        minimum_bin_episodes=minimum_bin_samples,
    )
    return {
        "status": "fitted_causal_target",
        "episodes": int(len(values)),
        "outcome_mean": float(np.mean(outcome)),
        "raw_brier_descriptive_only": float(np.mean((raw - outcome) ** 2)),
        "points": [
            {
                "raw_value": point.raw_probability,
                "calibrated_value": point.calibrated_probability,
                "episodes": point.episodes,
            }
            for point in points
        ],
    }


def _uncertainty_payload(episodes: int) -> dict[str, Any]:
    count = max(0, int(episodes))
    return {
        "status": "authorized_formula_passthrough",
        "formula_version": UNCERTAINTY_FORMULA_VERSION,
        "formula_hash": hashlib.sha256(UNCERTAINTY_FORMULA.encode()).hexdigest(),
        "episodes": count,
        "points": [
            {"raw_value": 0.0, "calibrated_value": 0.0, "episodes": count},
            {"raw_value": 1.0, "calibrated_value": 1.0, "episodes": count},
        ],
    }


def fit_typed_brain_calibration(
    *,
    row_paths: Sequence[str | Path],
    output: str | Path,
    model_config: str | Path = "configs/model.json",
    validation_protocol: str | Path = "configs/data_splits.json",
    bins: int = 10,
    minimum_bin_samples: int = 20,
    minimum_dimension_samples: int = 200,
    calibration_version: str = "4.0.0-typed-isotonic.4-scene-contract",
) -> dict[str, Any]:
    if bins < 2 or minimum_bin_samples < 1 or minimum_dimension_samples < 2:
        raise ValueError("calibration sample and bin limits are invalid")
    if not calibration_version.strip() or calibration_version == "identity-unvalidated":
        raise ValueError("calibration_version must be an explicit authorized version")
    destination = Path(output)

    files = _input_files(row_paths)
    frame = _read_rows(files)
    if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).eq("").any():
        raise ValueError("sample_id cannot be empty")
    _validate_identity_columns(frame)
    duplicates = frame["sample_id"].astype(str).duplicated(keep=False)
    if bool(duplicates.any()):
        raise ValueError("Brain calibration rows contain duplicate sample_id values")
    frame["censored"] = _strict_bool(frame["censored"], "censored")
    frame["fit_eligible"] = _strict_bool(frame["fit_eligible"], "fit_eligible")
    frame["sampled_at"] = _aware_utc(frame["sampled_at"], "sampled_at", nullable=False)
    frame["resolved_at"] = _aware_utc(frame["resolved_at"], "resolved_at", nullable=True)
    if frame["sampled_at"].isna().any():
        raise ValueError("sampled_at contains invalid timestamps")

    allowed_playbooks = {playbook.value for playbook in TYPED_ACTIVE_PLAYBOOKS}
    unexpected_playbooks = sorted(set(frame["playbook"].astype(str)) - allowed_playbooks)
    if unexpected_playbooks:
        raise ValueError(
            "typed calibration recorder may contain only active DFP/LSR rows; "
            f"got {unexpected_playbooks}"
        )
    if not set(frame["direction"].astype(str)).issubset({"long", "short"}):
        raise ValueError("direction must be long or short")
    allowed_dimensions = set(FITTED_DIMENSIONS) | {
        SEQUENCE_DIMENSION,
        UNCERTAINTY_DIMENSION,
    }
    unexpected_dimensions = sorted(
        set(frame["dimension"].astype(str)) - allowed_dimensions
    )
    if unexpected_dimensions:
        raise ValueError(f"unsupported typed calibration dimensions: {unexpected_dimensions}")
    immediate = frame["dimension"].isin({SEQUENCE_DIMENSION, UNCERTAINTY_DIMENSION})
    if bool((frame.loc[immediate, "fit_eligible"]).any()):
        raise ValueError("sequence_progress and uncertainty cannot be fitted to future outcomes")
    if bool((frame["fit_eligible"] & frame["censored"]).any()):
        raise ValueError("censored rows cannot be fit eligible")

    bindings = resolve_model_bindings(model_config)
    row_identity = _validate_bindings(frame, bindings)
    protocol = load_validation_protocol(validation_protocol)
    latest_resolution = frame.loc[frame["fit_eligible"], "resolved_at"].max()
    if pd.isna(latest_resolution):
        latest_resolution = frame["sampled_at"].max()
    window = protocol.classify_ohlcv(
        frame["sampled_at"].min(),
        max(frame["sampled_at"].max(), latest_resolution) + pd.Timedelta(nanoseconds=1),
    )
    if window.role not in {"calibration", "belief_calibration"}:
        raise ValueError(
            "typed Brain calibration may use only a registered calibration window, "
            f"got {window.role}"
        )

    eligible = frame.loc[
        frame["fit_eligible"]
        & ~frame["censored"]
        & frame["dimension"].isin(FITTED_DIMENSIONS)
    ].copy()
    if eligible.empty:
        raise CalibrationError("no resolved typed Brain targets are eligible for fitting")
    if eligible["resolved_at"].isna().any() or eligible["outcome_value"].isna().any():
        raise ValueError("fit-eligible rows require resolved_at and outcome_value")
    if bool((eligible["resolved_at"] < eligible["sampled_at"]).any()):
        raise ValueError("fit-eligible rows resolve before they were sampled")

    playbooks: dict[str, Any] = {}
    for playbook in TYPED_ACTIVE_PLAYBOOKS:
        dimensions: dict[str, Any] = {}
        playbook_rows = eligible.loc[eligible["playbook"].eq(playbook.value)]
        for dimension in FITTED_DIMENSIONS:
            values = playbook_rows.loc[playbook_rows["dimension"].eq(dimension)]
            dimensions[dimension] = _dimension_payload(
                values,
                playbook=playbook,
                dimension=dimension,
                bins=bins,
                minimum_bin_samples=minimum_bin_samples,
                minimum_dimension_samples=minimum_dimension_samples,
            )
        dimensions[UNCERTAINTY_DIMENSION] = _uncertainty_payload(
            frame.loc[
                frame["playbook"].eq(playbook.value)
                & frame["dimension"].eq(UNCERTAINTY_DIMENSION),
                "sample_id",
            ].nunique()
        )
        playbooks[playbook.value] = {
            "status": "active",
            "dimensions": dimensions,
        }
    for playbook in TYPED_PARKED_PLAYBOOKS:
        playbooks[playbook.value] = {
            "status": "parked_missing_natural_authority",
            "dimensions": {},
        }

    payload: dict[str, Any] = {
        "calibration_version": calibration_version,
        "status": "ready",
        "method": {
            "name": "tie_preserving_nearest_quantile_bins_weighted_pava_flat_blocks",
            "fitted_dimensions": list(FITTED_DIMENSIONS),
            "sequence_progress": "deterministic_passthrough_not_fitted",
            "uncertainty": "authorized_contemporaneous_formula_passthrough",
            "uncertainty_formula_version": UNCERTAINTY_FORMULA_VERSION,
            "uncertainty_formula_hash": hashlib.sha256(
                UNCERTAINTY_FORMULA.encode()
            ).hexdigest(),
            "bins": bins,
            "minimum_bin_samples": minimum_bin_samples,
            "minimum_dimension_samples": minimum_dimension_samples,
            "directions_pooled": True,
            "threshold_search": False,
            "pnl_labels_used": False,
            "mbo_used": False,
        },
        "validation_schema_version": protocol.schema_version,
        "validation_protocol_hash": protocol.fingerprint,
        "training_window_role": window.role,
        "training_start": frame["sampled_at"].min().isoformat(),
        "training_end": latest_resolution.isoformat(),
        "brain_target_protocol_versions": row_identity["protocol_versions"],
        "brain_target_protocol_hash": row_identity["protocol_hash"],
        "playbook_registry_hash": bindings["registry_hash"],
        "model_code_hash": bindings["model_code_hash"],
        "training_config_hash": bindings["config_hash"],
        "primitive_protocol_hashes": dict(bindings["primitive_protocol_hashes"]),
        "brain_input_contract_hash": bindings["brain_input_contract_hash"],
        "source_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in files
        ],
        "playbooks": playbooks,
        "favr_parked": True,
        "holdout_used": False,
        "rolling_oof_used": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument(
        "--validation-protocol",
        default="configs/data_splits.json",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--minimum-bin-samples", type=int, default=20)
    parser.add_argument("--minimum-dimension-samples", type=int, default=200)
    parser.add_argument(
        "--calibration-version",
        default="4.0.0-typed-isotonic.4-scene-contract",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = fit_typed_brain_calibration(
        row_paths=args.rows,
        output=args.output,
        model_config=args.model_config,
        validation_protocol=args.validation_protocol,
        bins=args.bins,
        minimum_bin_samples=args.minimum_bin_samples,
        minimum_dimension_samples=args.minimum_dimension_samples,
        calibration_version=args.calibration_version,
    )
    print(
        json.dumps(
            {
                playbook: {
                    dimension: values["episodes"]
                    for dimension, values in item["dimensions"].items()
                }
                for playbook, item in payload["playbooks"].items()
                if item["status"] == "active"
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
