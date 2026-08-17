#!/usr/bin/env python3
"""Fit typed Brain maps from resolved causal recorder rows.

This is intentionally a small, one-purpose fitter.  It does not inspect
actions, PnL, MFE/MAE, MBO, or holdout data, and it does not search thresholds.
Thesis and delivery use a small L2-regularized monotone logistic map;
location and readiness use fixed quantile bins plus weighted PAVA.  Sequence
progress remains a deterministic state-machine field and uncertainty remains
the registered contemporaneous conflict/missing-authority formula.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
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
)
from smc_trader.brain_calibration import (  # noqa: E402
    BrainCalibrationRecord,
    RECORDER_SCHEMA_VERSION,
)
from smc_trader.artifact_stream import verify_stream_shards  # noqa: E402
from smc_trader.model import Playbook, PlaybookPhase  # noqa: E402
from smc_trader.playbook_registry import load_playbook_registry  # noqa: E402
from smc_trader.validation import (  # noqa: E402
    BrainCalibrationFitAdmission,
    load_validation_protocol,
)


FITTED_DIMENSIONS = (
    "thesis_strength",
    "location_quality",
    "entry_readiness",
    "delivery_quality",
)
SEQUENCE_DIMENSION = "sequence_progress"
UNCERTAINTY_DIMENSION = "uncertainty"
UNCERTAINTY_FORMULA_VERSION = (
    "5.0.0-stage-aware-decomposed-noisy-or.1"
)
UNCERTAINTY_FORMULA = (
    "uncertainty_total = 1 - product(1 - component) over conflict, "
    "stage-required evidence missing, semantic authority missing, and "
    "graph ambiguity; future causal steps, execution, spread, MBO, PnL, "
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
    field.name for field in fields(BrainCalibrationRecord)
)
CALIBRATION_UNIT_COLUMNS = (
    "playbook",
    "direction",
    "calibration_unit_kind",
    "calibration_unit_id",
)
REQUIRED_STREAM_FIELD_TYPES = {
    "calibration_unit_id": "large_string",
    "calibration_unit_kind": "large_string",
    "context_thesis_id": "large_string",
    "parent_context_thesis_id": "large_string",
    "free_path_R": "float64",
    "obstruction_distance_R": "float64",
    "soft_obstruction_count": "int64",
    "hard_barrier_before_target": "bool",
    "market_thesis_id": "large_string",
    "bound_market_thesis_id": "large_string",
    "market_thesis_root_id": "large_string",
    "market_thesis_mechanism": "large_string",
    "market_thesis_authority_relation": "large_string",
    "playbook_match_strength": "float64",
    "market_thesis_binding_required": "bool",
    "market_thesis_action_bound": "bool",
    "market_thesis_match_status": "large_string",
    "playbook_first_failed_hard_gate_id": "large_string",
    "playbook_plan_delivery_valid": "bool",
    "global_market_mode": "large_string",
    "path_blocker_ids": "large_string",
    "uncertainty_conflict": "float64",
    "uncertainty_required_evidence_missing": "float64",
    "uncertainty_authority_missing": "float64",
    "uncertainty_graph_ambiguity": "float64",
    "uncertainty_total": "float64",
    "target_deadline_kind": "large_string",
}


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


def _validate_identity_columns(frame: pd.DataFrame) -> None:
    for field in (
        "hypothesis_key",
        "setup_id",
        "calibration_unit_id",
        "calibration_unit_kind",
        "phase",
        "symbol",
        "authority_relation",
        "conflict_role",
        "conflict_scope",
        "acceptance_state",
        "target_deadline_kind",
        "global_market_mode",
        "context_thesis_id",
    ):
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"{field} cannot be empty")
    episode_owned = frame["episode_id"].notna()
    if (
        frame.loc[episode_owned, "parent_context_thesis_id"].isna().any()
        or frame.loc[
            episode_owned,
            "parent_context_thesis_id",
        ].astype(str).str.strip().eq("").any()
        or not frame.loc[
            episode_owned,
            "parent_context_thesis_id",
        ].astype(str).eq(
            frame.loc[episode_owned, "context_thesis_id"].astype(str)
        ).all()
    ):
        raise ValueError(
            "entry episode parent_context_thesis_id must match context_thesis_id"
        )
    expected_kinds = {
        (Playbook.DISPLACEMENT_FIRST_PULLBACK.value, "thesis_strength"):
            "dfp_context_thesis",
        (Playbook.LIQUIDITY_SWEEP_REVERSAL.value, "thesis_strength"):
            "lsr_context_thesis",
        (Playbook.DISPLACEMENT_FIRST_PULLBACK.value, "location_quality"):
            "entry_path_location",
        (Playbook.LIQUIDITY_SWEEP_REVERSAL.value, "location_quality"):
            "entry_path_location",
        (Playbook.DISPLACEMENT_FIRST_PULLBACK.value, "entry_readiness"):
            "trigger_entry_path",
        (Playbook.LIQUIDITY_SWEEP_REVERSAL.value, "entry_readiness"):
            "trigger_entry_path",
        (Playbook.DISPLACEMENT_FIRST_PULLBACK.value, "delivery_quality"):
            "trigger_entry_path",
        (Playbook.LIQUIDITY_SWEEP_REVERSAL.value, "delivery_quality"):
            "trigger_entry_path",
        (Playbook.DISPLACEMENT_FIRST_PULLBACK.value, "sequence_progress"):
            "hypothesis_sequence",
        (Playbook.LIQUIDITY_SWEEP_REVERSAL.value, "sequence_progress"):
            "hypothesis_sequence",
        (Playbook.DISPLACEMENT_FIRST_PULLBACK.value, "uncertainty"):
            "decision_hypothesis",
        (Playbook.LIQUIDITY_SWEEP_REVERSAL.value, "uncertainty"):
            "decision_hypothesis",
    }
    actual_kinds = frame.apply(
        lambda row: expected_kinds.get(
            (str(row["playbook"]), str(row["dimension"]))
        ),
        axis=1,
    )
    if actual_kinds.isna().any() or not frame[
        "calibration_unit_kind"
    ].astype(str).eq(actual_kinds.astype(str)).all():
        raise ValueError(
            "calibration_unit_kind disagrees with its causal dimension owner"
        )
    unit_ids = frame["calibration_unit_id"].astype(str)
    dimensions = frame["dimension"].astype(str)
    playbooks = frame["playbook"].astype(str)
    dfp_thesis = dimensions.eq("thesis_strength") & playbooks.eq(
        Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    lsr_thesis = dimensions.eq("thesis_strength") & playbooks.eq(
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    path_owned = dimensions.isin(
        {"location_quality", "entry_readiness", "delivery_quality"}
    )
    sequence_owned = dimensions.eq("sequence_progress")
    decision_owned = dimensions.eq("uncertainty")
    bad_owner = (
        (
            dfp_thesis
            & ~unit_ids.eq(frame["context_thesis_id"].astype(str))
        )
        | (
            lsr_thesis
            & ~unit_ids.eq(frame["context_thesis_id"].astype(str))
        )
        | (path_owned & ~unit_ids.eq(frame["entry_path_id"].astype(str)))
        | (sequence_owned & ~unit_ids.eq(frame["setup_id"].astype(str)))
        | (
            decision_owned
            & ~frame.apply(
                lambda row: str(row["calibration_unit_id"])
                in {
                    str(value)
                    for value in (
                        row["episode_id"],
                        row["context_thesis_id"],
                        row["setup_id"],
                    )
                    if pd.notna(value)
                },
                axis=1,
            )
        )
    )
    if bool(bad_owner.any()):
        raise ValueError(
            "calibration_unit_id disagrees with its causal dimension owner"
        )
    if bool(lsr_thesis.any()):
        lsr_rows = frame.loc[lsr_thesis]
        nullable_child_fields = (
            "draw_id",
            "draw_price",
            "liquidity_route_id",
            "context_draw_id",
            "primary_deliverable_target_id",
            "terminal_draw_id",
            "authority_barrier_id",
            "authority_barrier_price",
            "obstruction_distance_R",
            "free_path_R",
        )
        empty_route_lists = (
            lsr_rows["intermediate_liquidity_ids"].astype(str).eq("[]")
            & lsr_rows["path_blocker_ids"].astype(str).eq("[]")
            & lsr_rows["source_path_ids"].astype(str).eq("[]")
        )
        invalidation_prices = pd.to_numeric(
            lsr_rows["invalidation_price"],
            errors="coerce",
        )
        invalidation_sources = lsr_rows["invalidation_source_id"]
        if (
            any(
                bool(lsr_rows[field].notna().any())
                for field in nullable_child_fields
            )
            or not bool(empty_route_lists.all())
            or not bool(lsr_rows["soft_obstruction_count"].eq(0).all())
            or bool(lsr_rows["hard_barrier_before_target"].astype(bool).any())
            or bool(invalidation_prices.isna().any())
            or not bool(invalidation_prices.gt(0.0).all())
            or bool(invalidation_sources.isna().any())
            or bool(
                invalidation_sources.astype(str).str.strip().eq("").any()
            )
            or bool(lsr_rows["deadline"].isna().any())
            or not bool(
                lsr_rows["target_deadline_kind"].eq(
                    "thesis_deadline"
                ).all()
            )
        ):
            raise ValueError("LSR Context thesis target custody is invalid")
    for field in (
        "intermediate_liquidity_ids",
        "path_blocker_ids",
        "source_path_ids",
    ):
        for raw in frame[field]:
            try:
                values = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{field} must be a JSON identity list") from error
            if (
                not isinstance(values, list)
                or any(not isinstance(item, str) or not item for item in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"{field} identities are invalid")

    match_status = frame["market_thesis_match_status"].astype(str)
    valid_match_statuses = {
        "not_required",
        "no_open_thesis",
        "no_direction_match",
        "no_mechanism_match",
        "root_identity_unbound",
        "exact_root_bound",
    }
    if not match_status.isin(valid_match_statuses).all():
        raise ValueError("market_thesis_match_status is invalid")
    binding_required = _strict_bool(
        frame["market_thesis_binding_required"],
        "market_thesis_binding_required",
    )
    action_bound = _strict_bool(
        frame["market_thesis_action_bound"],
        "market_thesis_action_bound",
    )
    frame["playbook_plan_delivery_valid"] = _strict_bool(
        frame["playbook_plan_delivery_valid"],
        "playbook_plan_delivery_valid",
    )

    def present(field: str) -> pd.Series:
        return frame[field].notna() & frame[field].astype(str).str.strip().ne("")

    primary_present = present("market_thesis_id")
    bound_present = present("bound_market_thesis_id")
    matched_identity_present = (
        present("market_thesis_root_id")
        & present("market_thesis_mechanism")
        & present("market_thesis_authority_relation")
    )
    partial_matched_identity = (
        present("market_thesis_root_id")
        | present("market_thesis_mechanism")
        | present("market_thesis_authority_relation")
    )
    matched_status = match_status.isin(
        {"root_identity_unbound", "exact_root_bound"}
    )
    exact_status = match_status.eq("exact_root_bound")
    if bool(
        (
            binding_required.ne(match_status.ne("not_required"))
            | action_bound.ne(exact_status)
            | bound_present.ne(exact_status)
            | primary_present.ne(matched_status)
            | matched_identity_present.ne(matched_status)
            | (partial_matched_identity & ~matched_status)
            | (
                exact_status
                & frame["bound_market_thesis_id"].astype(str).ne(
                    frame["market_thesis_id"].astype(str)
                )
            )
        ).any()
    ):
        raise ValueError("market thesis binding diagnostics are inconsistent")
    match_strength = pd.to_numeric(
        frame["playbook_match_strength"],
        errors="coerce",
    )
    match_strength_bool = frame["playbook_match_strength"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if (
        match_strength.isna().any()
        or match_strength_bool.any()
        or not np.isfinite(match_strength.to_numpy(float)).all()
        or (match_strength < 0.0).any()
        or (match_strength > 1.0).any()
        or (match_strength[~matched_status] != 0.0).any()
    ):
        raise ValueError("playbook_match_strength is inconsistent")
    failed_gate = frame["playbook_first_failed_hard_gate_id"]
    if (
        failed_gate.notna()
        & failed_gate.astype(str).str.strip().eq("")
    ).any():
        raise ValueError("playbook_first_failed_hard_gate_id is invalid")

    for field in (
        "authority_rank_gap",
        "soft_obstruction_count",
        "ambiguity_count",
    ):
        numeric = pd.to_numeric(frame[field], errors="coerce")
        if (
            numeric.isna().any()
            or (numeric < 0).any()
            or (numeric % 1 != 0).any()
        ):
            raise ValueError(f"{field} must contain non-negative integers")
    obstruction = pd.to_numeric(
        frame["obstruction_distance_R"],
        errors="coerce",
    )
    present = frame["obstruction_distance_R"].notna()
    obstruction_bool = frame.loc[present, "obstruction_distance_R"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if obstruction_bool.any() or (obstruction.loc[present] < 0.0).any() or not np.isfinite(
        obstruction.loc[present]
    ).all():
        raise ValueError(
            "obstruction_distance_R must be null or a non-negative finite number"
        )
    free_path = pd.to_numeric(frame["free_path_R"], errors="coerce")
    free_present = frame["free_path_R"].notna()
    free_bool = frame.loc[free_present, "free_path_R"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if free_bool.any() or (free_path.loc[free_present] < 0.0).any() or not np.isfinite(
        free_path.loc[free_present]
    ).all():
        raise ValueError(
            "free_path_R must be null or a non-negative finite number"
        )
    frame["hard_barrier_before_target"] = _strict_bool(
        frame["hard_barrier_before_target"],
        "hard_barrier_before_target",
    )
    blocker_present = frame["path_blocker_ids"].map(
        lambda value: bool(json.loads(value))
    )
    if not frame["hard_barrier_before_target"].eq(blocker_present).all() or bool(
        (
            frame["hard_barrier_before_target"]
            & (
                frame["free_path_R"].isna()
                | frame["obstruction_distance_R"].isna()
            )
        ).any()
    ):
        raise ValueError(
            "hard barrier state disagrees with its frozen blocker geometry"
        )
    uncertainty_fields = (
        "uncertainty_conflict",
        "uncertainty_required_evidence_missing",
        "uncertainty_authority_missing",
        "uncertainty_graph_ambiguity",
        "uncertainty_total",
    )
    components = frame.loc[:, uncertainty_fields].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        components.isna().any().any()
        or not np.isfinite(components.to_numpy(float)).all()
        or ((components < 0.0) | (components > 1.0)).any().any()
    ):
        raise ValueError("uncertainty components must be finite within [0, 1]")
    recomputed = 1.0 - np.prod(
        1.0
        - components.loc[
            :,
            uncertainty_fields[:-1],
        ].to_numpy(float),
        axis=1,
    )
    if not np.allclose(
        recomputed,
        components["uncertainty_total"].to_numpy(float),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError("uncertainty total disagrees with its components")
    uncertainty_rows = frame["dimension"].eq(UNCERTAINTY_DIMENSION)
    if not np.allclose(
        pd.to_numeric(
            frame.loc[uncertainty_rows, "raw_value"],
            errors="coerce",
        ).to_numpy(float),
        components.loc[uncertainty_rows, "uncertainty_total"].to_numpy(float),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError("uncertainty raw values disagree with uncertainty_total")


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
    """Resolve the current typed playbook registry and schema versions."""

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
    return {
        "config_path": config_path,
        "model_config_identity": hashlib.sha256(raw).hexdigest(),
        "registry_hash": registry.fingerprint,
        "registry_schema_version": registry.schema_version,
        "playbook_schema_versions": {
            playbook.value: registry.for_playbook(playbook).schema_version
            for playbook in TYPED_ACTIVE_PLAYBOOKS
        },
    }


def _load_run_bindings(
    run_manifest: str | Path,
    bindings: Mapping[str, Any],
    *,
    row_files: Sequence[Path],
) -> dict[str, Any]:
    source = _resolve(run_manifest)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("runner") != "continuous_replay"
    ):
        raise ValueError("Brain calibration run manifest is invalid")
    model_identity = payload.get("model_config")
    if (
        not isinstance(model_identity, Mapping)
        or model_identity.get("identity")
        != bindings["model_config_identity"]
    ):
        raise ValueError(
            "Brain calibration run manifest model identity is stale"
        )
    output = payload.get("output")
    stream_families = (
        output.get("stream_families")
        if isinstance(output, Mapping)
        else None
    )
    if (
        not isinstance(output, Mapping)
        or output.get("brain_calibration") is not True
        or output.get("brain_calibration_schema_version")
        != RECORDER_SCHEMA_VERSION
        or not isinstance(stream_families, list)
        or "brain_calibration_shards"
        not in stream_families
    ):
        raise ValueError(
            "run manifest does not describe the current Brain calibration stream"
        )
    identity = payload.get("brain_calibration_identity")
    expected = {
        "recorder_schema_version": RECORDER_SCHEMA_VERSION,
        "registry_fingerprint": bindings["registry_hash"],
        "registry_schema_version": bindings["registry_schema_version"],
        "playbook_schema_versions": {
            key: value
            for key, value in sorted(
                bindings["playbook_schema_versions"].items()
            )
        },
    }
    if identity != expected:
        raise ValueError(
            "Brain calibration run manifest registry identity is stale"
        )
    try:
        fit_admission = BrainCalibrationFitAdmission.from_mapping(
            payload.get("brain_calibration_fit_admission")
        )
    except ValueError as error:
        raise ValueError(
            "Brain calibration run manifest fit admission is invalid"
        ) from error

    window = payload.get("window")
    if not isinstance(window, Mapping):
        raise ValueError("Brain calibration run manifest window is invalid")
    try:
        window_start = pd.Timestamp(window["start"])
        window_end = pd.Timestamp(window["end_exclusive"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Brain calibration run manifest window is invalid") from error
    if (
        window_start.tzinfo is None
        or window_end.tzinfo is None
        or window_start >= window_end
        or window.get("role") not in {
            "calibration",
            "brain_validation",
            "brain_calibration_trial",
        }
    ):
        raise ValueError("Brain calibration run manifest window is invalid")

    run_directory = source.parent.resolve()
    stream_manifest_path = run_directory / (
        "brain_calibration_shards.manifest.json"
    )
    if not stream_manifest_path.is_file() or stream_manifest_path.is_symlink():
        raise ValueError("Brain calibration shard manifest is missing or invalid")
    stream_manifest = json.loads(
        stream_manifest_path.read_text(encoding="utf-8")
    )
    if (
        not isinstance(stream_manifest, Mapping)
        or stream_manifest.get("format_version") != 1
        or stream_manifest.get("artifact")
        != "continuous_development_brain_calibration_shards"
        or stream_manifest.get("status") != "complete"
        or stream_manifest.get("stream") != "brain_calibration_shards"
        or not isinstance(stream_manifest.get("bindings"), Mapping)
        or stream_manifest["bindings"].get("run_manifest") != source.name
        or not isinstance(stream_manifest.get("shards"), list)
        or any(
            not isinstance(item, Mapping)
            for item in stream_manifest.get("shards", ())
        )
    ):
        raise ValueError("Brain calibration shard manifest is invalid")
    field_types = stream_manifest.get("field_types")
    if (
        not isinstance(field_types, Mapping)
        or set(field_types) != REQUIRED_COLUMNS
        or any(
            field_types.get(name) != expected
            for name, expected in REQUIRED_STREAM_FIELD_TYPES.items()
        )
        or not isinstance(stream_manifest.get("schema_fingerprint"), str)
    ):
        raise ValueError(
            "Brain calibration shard manifest omits the current typed schema"
        )

    expected_files: list[Path] = []
    for shard in stream_manifest["shards"]:
        try:
            candidate = (run_directory / str(shard["path"])).resolve()
            candidate.relative_to(run_directory)
        except (KeyError, ValueError) as error:
            raise ValueError(
                "Brain calibration shard manifest contains an invalid path"
            ) from error
        expected_files.append(candidate)
    requested_files = {path.resolve() for path in row_files}
    if requested_files != set(expected_files) or len(expected_files) != len(
        requested_files
    ):
        raise ValueError(
            "requested Brain calibration shards do not exactly match the bound run"
        )

    stream_state: dict[str, Any] = {
        "rows": stream_manifest.get("rows"),
        "next_shard_index": len(stream_manifest["shards"]),
        "committed_shards": list(stream_manifest["shards"]),
    }
    for name in ("schema_fingerprint", "field_types"):
        if name in stream_manifest:
            stream_state[name] = stream_manifest[name]
    verify_stream_shards(run_directory, stream_state)
    return {
        "path": source,
        "window_start": window_start.tz_convert("UTC"),
        "window_end_exclusive": window_end.tz_convert("UTC"),
        "window_role": str(window["role"]),
        "fit_admission": fit_admission,
        **expected,
    }


def _validate_run_window(
    frame: pd.DataFrame,
    run_identity: Mapping[str, Any],
) -> None:
    start = run_identity["window_start"]
    end = run_identity["window_end_exclusive"]
    sampled_outside = (frame["sampled_at"] < start) | (
        frame["sampled_at"] >= end
    )
    resolved_outside = (frame["resolved_at"] < start) | (
        frame["resolved_at"] >= end
    )
    if bool(sampled_outside.any()) or bool(resolved_outside.any()):
        raise ValueError(
            "Brain calibration row clocks fall outside the bound run window"
        )


def _unique_calibration_units(values: pd.DataFrame) -> int:
    return int(
        values.loc[:, CALIBRATION_UNIT_COLUMNS]
        .drop_duplicates()
        .shape[0]
    )


def _dimension_fit_diagnostics(
    values: pd.DataFrame,
    *,
    playbook: Playbook,
    dimension: str,
    minimum_dimension_units: int | None,
) -> tuple[pd.Series, pd.Series, int, list[str]]:
    """Return one shared validation result for admission and fitting."""

    raw = pd.to_numeric(values["raw_value"], errors="coerce")
    outcome = pd.to_numeric(values["outcome_value"], errors="coerce")
    unique_units = _unique_calibration_units(values)
    identity = f"{playbook.value}.{dimension}"
    failures: list[str] = []
    if (
        minimum_dimension_units is not None
        and unique_units < minimum_dimension_units
    ):
        failures.append(
            f"{identity} has {unique_units} unique calibration units values "
            f"across {len(values)} eligible rows; at least "
            f"{minimum_dimension_units} independent units are required"
        )
    raw_finite = bool(
        not raw.isna().any() and np.isfinite(raw.to_numpy(float)).all()
    )
    outcome_finite = bool(
        not outcome.isna().any()
        and np.isfinite(outcome.to_numpy(float)).all()
    )
    if not raw_finite or not outcome_finite:
        failures.append(f"{identity} contains non-finite values")
    elif not bool(((raw >= 0.0) & (raw <= 1.0)).all()) or not bool(
        ((outcome >= 0.0) & (outcome <= 1.0)).all()
    ):
        failures.append(f"{identity} values must lie in [0, 1]")
    else:
        unit_outcomes = (
            values.assign(_validated_outcome=outcome)
            .groupby(
                list(CALIBRATION_UNIT_COLUMNS),
                dropna=False,
                sort=False,
            )["_validated_outcome"]
            .nunique(dropna=False)
        )
        conflicting_units = int((unit_outcomes > 1).sum())
        if conflicting_units:
            failures.append(
                f"{identity} has conflicting outcomes for "
                f"{conflicting_units} independent calibration units"
            )
        if int(raw.nunique()) < 2:
            failures.append(
                f"{identity} has fewer than two unique raw values"
            )
        if int(outcome.nunique()) < 2:
            failures.append(
                f"{identity} has no resolved outcome variation"
            )
    return raw, outcome, unique_units, failures


def _validated_dimension_arrays(
    values: pd.DataFrame,
    *,
    playbook: Playbook,
    dimension: str,
    minimum_dimension_units: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_series, outcome_series, unique_units, failures = (
        _dimension_fit_diagnostics(
            values,
            playbook=playbook,
            dimension=dimension,
            minimum_dimension_units=minimum_dimension_units,
        )
    )
    if failures:
        raise CalibrationError("; ".join(failures))
    unit_keys = values.loc[:, CALIBRATION_UNIT_COLUMNS].astype(str).agg(
        lambda row: json.dumps(
            list(row),
            separators=(",", ":"),
        ),
        axis=1,
    )
    raw = raw_series.to_numpy(float)
    outcome = outcome_series.to_numpy(float)
    revision_counts = unit_keys.groupby(unit_keys, sort=False).transform(
        "size"
    ).to_numpy(float)
    unit_weights = 1.0 / revision_counts
    if not np.isclose(float(np.sum(unit_weights)), float(unique_units)):
        raise CalibrationError(
            f"{playbook.value}.{dimension} calibration-unit weights do not "
            "conserve one unit per independent causal owner"
        )
    return raw, outcome, unit_weights, unit_keys.to_numpy(str)


def _unique_runtime_units(values: pd.DataFrame, identity_field: str) -> int:
    """Count stable runtime roots/episodes without counting row revisions."""

    present = values[identity_field].notna() & values[
        identity_field
    ].astype(str).str.strip().ne("")
    if not bool(present.any()):
        return 0
    return int(
        values.loc[
            present,
            ("symbol", "instrument_id", identity_field),
        ]
        .astype(str)
        .drop_duplicates()
        .shape[0]
    )


def _fit_admission_funnel(
    frame: pd.DataFrame,
    *,
    minimum_dimension_units: int | None,
    minimum_plan_valid_roots: int | None,
    minimum_executable_episodes: int | None,
    window_role: str,
    enforce_thresholds: bool = True,
) -> dict[str, Any]:
    """Describe whether all registered dimension maps are ready to fit.

    Admission deliberately consumes each dimension's existing causal capture
    contract.  It never applies a shared phase filter: thesis revisions,
    first location observations, qualified triggers, and frozen delivery
    plans are different sampling clocks and need not all be executable rows.
    """

    threshold_values = (
        minimum_dimension_units,
        minimum_plan_valid_roots,
        minimum_executable_episodes,
    )
    if enforce_thresholds and any(value is None for value in threshold_values):
        raise ValueError(
            "formal fitting requires all three fit-admission thresholds"
        )
    dimension_minimum = (
        int(minimum_dimension_units) if enforce_thresholds else None
    )
    plan_minimum = (
        int(minimum_plan_valid_roots) if enforce_thresholds else None
    )
    executable_minimum = (
        int(minimum_executable_episodes) if enforce_thresholds else None
    )

    playbook_payloads: dict[str, Any] = {}
    all_failures: list[str] = []
    for playbook in TYPED_ACTIVE_PLAYBOOKS:
        dimensions: dict[str, Any] = {}
        playbook_rows = frame.loc[frame["playbook"].eq(playbook.value)]
        plan_delivery_valid_rows = playbook_rows.loc[
            playbook_rows["playbook_plan_delivery_valid"]
            & playbook_rows["market_thesis_action_bound"]
            & playbook_rows["market_thesis_match_status"].eq(
                "exact_root_bound"
            )
        ]
        executable_rows = plan_delivery_valid_rows.loc[
            plan_delivery_valid_rows["phase"].eq(
                PlaybookPhase.EXECUTABLE.value
            )
        ]
        unique_plan_delivery_valid_units = _unique_runtime_units(
            plan_delivery_valid_rows,
            "market_thesis_root_id",
        )
        unique_executable_units = _unique_runtime_units(
            executable_rows,
            "episode_id",
        )
        playbook_failures: list[str] = []
        if (
            plan_minimum is not None
            and unique_plan_delivery_valid_units < plan_minimum
        ):
            playbook_failures.append(
                f"{playbook.value} has "
                f"{unique_plan_delivery_valid_units} unique "
                "plan_delivery_valid roots; at least "
                f"{plan_minimum} are required"
            )
        if (
            executable_minimum is not None
            and unique_executable_units < executable_minimum
        ):
            playbook_failures.append(
                f"{playbook.value} has {unique_executable_units} unique "
                "executable episodes with a valid plan; at least "
                f"{executable_minimum} are required"
            )
        for dimension in FITTED_DIMENSIONS:
            values = playbook_rows.loc[
                playbook_rows["dimension"].eq(dimension)
            ]
            resolved = values.loc[values["outcome_value"].notna()]
            censored = values.loc[values["censored"]]
            eligible = values.loc[
                values["fit_eligible"] & ~values["censored"]
            ]
            hard_barrier = values.loc[
                values["hard_barrier_before_target"]
                & values["dimension"].eq("delivery_quality")
            ]

            raw, outcome, eligible_units, quality_issues = (
                _dimension_fit_diagnostics(
                    eligible,
                    playbook=playbook,
                    dimension=dimension,
                    minimum_dimension_units=dimension_minimum,
                )
            )
            raw_finite = bool(
                not raw.isna().any()
                and np.isfinite(raw.to_numpy(float)).all()
            )
            raw_support_levels = (
                int(raw.nunique()) if raw_finite else 0
            )
            positive = eligible.loc[outcome.eq(1.0)]
            negative = eligible.loc[outcome.eq(0.0)]

            phase_distribution = {
                str(phase): {
                    "rows": int(len(group)),
                    "unique_calibration_units": _unique_calibration_units(
                        group
                    ),
                }
                for phase, group in values.groupby(
                    "phase",
                    dropna=False,
                    sort=True,
                )
            }
            failures = quality_issues if enforce_thresholds else []

            all_failures.extend(failures)
            dimensions[dimension] = {
                "status": (
                    "observed"
                    if not enforce_thresholds
                    else "admitted" if not failures else "blocked"
                ),
                "rows": int(len(values)),
                "resolved_rows": int(len(resolved)),
                "censored_rows": int(len(censored)),
                "fit_eligible_rows": int(len(eligible)),
                "unique_causal_units": _unique_calibration_units(values),
                "resolved_unique_causal_units": (
                    _unique_calibration_units(resolved)
                ),
                "fit_eligible_unique_causal_units": eligible_units,
                "raw_support_levels": raw_support_levels,
                "positive_outcome_units": _unique_calibration_units(
                    positive
                ),
                "negative_outcome_units": _unique_calibration_units(
                    negative
                ),
                "delivery_hard_barrier_rows": int(len(hard_barrier)),
                "delivery_hard_barrier_units": (
                    _unique_calibration_units(hard_barrier)
                ),
                "phase_distribution": phase_distribution,
                "failure_reasons": failures,
                "quality_observations": quality_issues,
            }
        all_failures.extend(playbook_failures)
        dimension_admitted = all(
            item["status"] == "admitted" for item in dimensions.values()
        )
        playbook_payloads[playbook.value] = {
            "status": (
                "observed"
                if not enforce_thresholds
                else (
                    "admitted"
                    if dimension_admitted and not playbook_failures
                    else "blocked"
                )
            ),
            "plan_delivery_valid_rows": int(len(plan_delivery_valid_rows)),
            "unique_plan_delivery_valid_units": (
                unique_plan_delivery_valid_units
            ),
            "executable_plan_delivery_valid_rows": int(
                len(executable_rows)
            ),
            "unique_executable_units": unique_executable_units,
            "runtime_funnel_failure_reasons": playbook_failures,
            "dimensions": dimensions,
        }

    data_ready = not all_failures if enforce_thresholds else None
    window_fit_authorized = window_role == "calibration"
    return {
        "status": (
            "coverage_observed"
            if not enforce_thresholds
            else "admitted" if data_ready else "blocked"
        ),
        "data_ready": data_ready,
        "window_role": window_role,
        "window_fit_authorized": window_fit_authorized,
        "fit_allowed": bool(
            enforce_thresholds and data_ready and window_fit_authorized
        ),
        "thresholds_enforced": enforce_thresholds,
        "threshold_source": (
            "source_run_manifest.brain_calibration_fit_admission"
        ),
        "phase_filter": "none_dimension_specific_causal_capture",
        "runtime_gate_only": (
            "plan_delivery_valid roots and executable episodes authorize "
            "fitting but never filter dimension rows"
        ),
        "playbooks": playbook_payloads,
        "failure_reasons": all_failures,
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_isotonic_points(
    raw: np.ndarray,
    outcome: np.ndarray,
    setup_weights: np.ndarray,
    setup_ids: np.ndarray,
    *,
    bins: int,
    minimum_bin_setups: int,
) -> list[dict[str, Any]]:
    """Tie-preserving weighted PAVA with one total unit per setup.

    A setup may contribute revisions to more than one raw-value bin, but the
    sum of those revision weights remains exactly one.  ``episodes`` remains
    an integer count of distinct contributing setup identities for loader
    compatibility; ``setup_weight`` records the conserved effective mass.
    """

    order = np.argsort(raw, kind="stable")
    sorted_raw = raw[order]
    run_starts = np.r_[0, np.flatnonzero(np.diff(sorted_raw)) + 1]
    runs = tuple(np.split(order, run_starts[1:]))
    run_weights = np.asarray(
        [float(np.sum(setup_weights[index])) for index in runs],
        dtype=float,
    )
    total_weight = float(np.sum(run_weights))
    target_bins = min(
        int(bins),
        len(runs),
        int(np.floor(total_weight / float(minimum_bin_setups))),
    )
    if target_bins < 2:
        raise CalibrationError(
            "too few independently weighted setups for populated reliability bins"
        )

    cumulative = np.cumsum(run_weights)
    legal_boundaries = tuple(
        (run_index + 1, float(cumulative[run_index]))
        for run_index in range(len(runs) - 1)
    )
    boundaries = sorted(
        {
            min(
                legal_boundaries,
                key=lambda candidate: (
                    abs(candidate[1] - cut),
                    candidate[0],
                ),
            )[0]
            for cut in (
                index * total_weight / target_bins
                for index in range(1, target_bins)
            )
        }
    )
    run_groups = [
        list(group)
        for group in np.split(np.arange(len(runs)), boundaries)
        if len(group)
    ]
    index = 0
    while index < len(run_groups):
        group_weight = float(
            sum(run_weights[item] for item in run_groups[index])
        )
        if group_weight + 1e-12 >= float(minimum_bin_setups):
            index += 1
            continue
        if len(run_groups) == 1:
            break
        if index == 0:
            run_groups[1] = run_groups[0] + run_groups[1]
            del run_groups[0]
        else:
            run_groups[index - 1].extend(run_groups[index])
            del run_groups[index]
            index -= 1
    groups = [
        np.concatenate([runs[item] for item in group])
        for group in run_groups
        if float(sum(run_weights[item] for item in group)) + 1e-12
        >= float(minimum_bin_setups)
    ]
    if len(groups) < 2:
        raise CalibrationError(
            "too few independently weighted setups for populated reliability bins"
        )

    blocks: list[dict[str, Any]] = []
    for members in groups:
        member_weights = setup_weights[members]
        mass = float(np.sum(member_weights))
        raw_value = _weighted_mean(raw[members], member_weights)
        setup_count = len(set(str(item) for item in setup_ids[members]))
        blocks.append(
            {
                "x": raw_value,
                "y": float(
                    (np.sum(outcome[members] * member_weights) + 1.0)
                    / (mass + 2.0)
                ),
                "weight": mass,
                "setup_ids": set(str(item) for item in setup_ids[members]),
                "members": ((raw_value, setup_count, mass),),
            }
        )

    index = 0
    while index < len(blocks) - 1:
        if blocks[index]["y"] <= blocks[index + 1]["y"]:
            index += 1
            continue
        left = blocks[index]
        right = blocks[index + 1]
        mass = float(left["weight"] + right["weight"])
        blocks[index : index + 2] = [
            {
                "x": (
                    left["x"] * left["weight"]
                    + right["x"] * right["weight"]
                )
                / mass,
                "y": (
                    left["y"] * left["weight"]
                    + right["y"] * right["weight"]
                )
                / mass,
                "weight": mass,
                "setup_ids": left["setup_ids"] | right["setup_ids"],
                "members": left["members"] + right["members"],
            }
        ]
        index = max(0, index - 1)
    if len(blocks) < 2:
        raise CalibrationError(
            "calibration collapsed to one level; retain identity and gather "
            "more independent setups"
        )

    points = [
        {
            "raw_value": float(raw_value),
            "calibrated_value": float(block["y"]),
            "episodes": int(setup_count),
            "setup_weight": float(setup_mass),
        }
        for block in blocks
        for raw_value, setup_count, setup_mass in block["members"]
    ]
    if any(
        right["raw_value"] <= left["raw_value"]
        for left, right in zip(points, points[1:])
    ):
        raise CalibrationError(
            "raw values do not provide distinct calibration levels; retain "
            "identity and gather more varied setups"
        )
    return points


def _isotonic_dimension_payload(
    values: pd.DataFrame,
    *,
    playbook: Playbook,
    dimension: str,
    bins: int,
    minimum_bin_samples: int,
    minimum_dimension_units: int,
) -> dict[str, Any]:
    raw, outcome, setup_weights, setup_ids = _validated_dimension_arrays(
        values,
        playbook=playbook,
        dimension=dimension,
        minimum_dimension_units=minimum_dimension_units,
    )
    points = _weighted_isotonic_points(
        raw,
        outcome,
        setup_weights,
        setup_ids,
        bins=bins,
        minimum_bin_setups=minimum_bin_samples,
    )
    setup_count = _unique_calibration_units(values)
    return {
        "status": "fitted_causal_target",
        "episodes": setup_count,
        "calibration_units": setup_count,
        "rows": int(len(values)),
        "calibration_unit_weighting": (
            "equal_total_weight_per_causal_owner"
        ),
        "outcome_mean": _weighted_mean(outcome, setup_weights),
        "raw_brier_descriptive_only": _weighted_mean(
            (raw - outcome) ** 2,
            setup_weights,
        ),
        "points": points,
    }


def _regularized_monotone_logistic_payload(
    values: pd.DataFrame,
    *,
    playbook: Playbook,
    dimension: str,
    bins: int,
    minimum_dimension_units: int,
    l2_strength: float = 1.0,
) -> dict[str, Any]:
    """Fit one non-negative-slope logistic map without threshold search."""

    raw, outcome, setup_weights, setup_ids = _validated_dimension_arrays(
        values,
        playbook=playbook,
        dimension=dimension,
        minimum_dimension_units=minimum_dimension_units,
    )
    design = np.column_stack((np.ones(len(raw)), raw))
    mean = float(
        np.clip(_weighted_mean(outcome, setup_weights), 1e-6, 1.0 - 1e-6)
    )
    beta = np.asarray([np.log(mean / (1.0 - mean)), 0.0], dtype=float)
    penalty = np.diag([0.0, float(l2_strength)])
    for _ in range(100):
        linear = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        curvature = np.maximum(probability * (1.0 - probability), 1e-8)
        gradient = (
            design.T @ (setup_weights * (outcome - probability))
            - penalty @ beta
        )
        hessian = (
            design.T
            @ ((setup_weights * curvature)[:, None] * design)
            + penalty
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise CalibrationError(
                f"{playbook.value}.{dimension} logistic fit is singular"
            ) from error
        proposed = beta + step
        proposed[1] = max(0.0, float(proposed[1]))
        if float(np.max(np.abs(proposed - beta))) <= 1e-10:
            beta = proposed
            break
        beta = proposed

    unique, inverse = np.unique(raw, return_inverse=True)
    unique_weights = np.asarray(
        [float(np.sum(setup_weights[inverse == index])) for index in range(len(unique))]
    )
    support_count = min(max(2, int(bins)), len(unique))
    cumulative = np.cumsum(unique_weights)
    cuts = np.linspace(0.0, float(cumulative[-1]), support_count)
    indexes = np.searchsorted(cumulative, cuts, side="left")
    indexes[0] = 0
    indexes[-1] = len(unique) - 1
    support = unique[np.unique(indexes)]
    if len(support) < 2:
        raise CalibrationError(
            f"{playbook.value}.{dimension} has insufficient logistic support"
        )
    calibrated = 1.0 / (
        1.0 + np.exp(-np.clip(beta[0] + beta[1] * support, -30.0, 30.0))
    )
    boundaries = np.concatenate(
        (
            [-np.inf],
            (support[:-1] + support[1:]) / 2.0,
            [np.inf],
        )
    )
    assignments = np.digitize(raw, boundaries[1:-1], right=False)
    point_setups = [
        len(set(str(item) for item in setup_ids[assignments == index]))
        for index in range(len(support))
    ]
    point_weights = [
        float(np.sum(setup_weights[assignments == index]))
        for index in range(len(support))
    ]
    setup_count = _unique_calibration_units(values)
    return {
        "status": "fitted_regularized_monotone_logistic",
        "episodes": setup_count,
        "calibration_units": setup_count,
        "rows": int(len(values)),
        "calibration_unit_weighting": (
            "equal_total_weight_per_causal_owner"
        ),
        "outcome_mean": _weighted_mean(outcome, setup_weights),
        "raw_brier_descriptive_only": _weighted_mean(
            (raw - outcome) ** 2,
            setup_weights,
        ),
        "regularization": {"kind": "l2", "strength": float(l2_strength)},
        "coefficients": {
            "intercept": float(beta[0]),
            "raw_value": float(beta[1]),
        },
        "points": [
            {
                "raw_value": float(raw_value),
                "calibrated_value": float(calibrated_value),
                "episodes": setup_count_at_point,
                "setup_weight": setup_weight,
            }
            for raw_value, calibrated_value, setup_count_at_point, setup_weight in zip(
                support,
                calibrated,
                point_setups,
                point_weights,
                strict=True,
            )
        ],
    }


def _context_feature_coverage(values: pd.DataFrame) -> dict[str, Any]:
    """Describe covariate coverage without fitting unregistered strata."""

    def distance_bucket(value: Any) -> str:
        if pd.isna(value):
            return "missing"
        distance = float(value)
        if distance < 0.5:
            return "[0,0.5R)"
        if distance < 1.0:
            return "[0.5R,1R)"
        if distance < 2.0:
            return "[1R,2R)"
        return "[2R,+inf)"

    feature_values: dict[str, pd.Series] = {
        "global_market_mode": values["global_market_mode"].astype(str),
        "authority_relation": values["authority_relation"].astype(str),
        "authority_rank_gap": values["authority_rank_gap"].astype(str),
        "conflict_role": values["conflict_role"].astype(str),
        "conflict_scope": values["conflict_scope"].astype(str),
        "acceptance_state": values["acceptance_state"].astype(str),
        "obstruction_distance_R": values["obstruction_distance_R"].map(
            distance_bucket
        ),
        "free_path_R": values["free_path_R"].map(distance_bucket),
        "soft_obstruction_count": values["soft_obstruction_count"].astype(str),
        "hard_barrier_before_target": values[
            "hard_barrier_before_target"
        ].map(lambda value: str(bool(value)).lower()),
        "ambiguity_count": values["ambiguity_count"].astype(str),
    }
    strata: list[dict[str, Any]] = []
    for feature, projected in feature_values.items():
        augmented = values.assign(_feature_value=projected)
        for (
            playbook,
            direction,
            dimension,
            feature_value,
        ), group in augmented.groupby(
            ["playbook", "direction", "dimension", "_feature_value"],
            dropna=False,
            sort=True,
        ):
            eligible = group.loc[group["fit_eligible"]]
            positive = group.loc[group["outcome_value"].eq(1.0)]
            negative = group.loc[group["outcome_value"].eq(0.0)]
            censored = group.loc[group["censored"]]
            strata.append(
                {
                    "playbook": str(playbook),
                    "direction": str(direction),
                    "dimension": str(dimension),
                    "feature": feature,
                    "feature_value": str(feature_value),
                    "rows": int(len(group)),
                    "unique_calibration_units": (
                        _unique_calibration_units(group)
                    ),
                    "fit_eligible_calibration_units": (
                        _unique_calibration_units(eligible)
                    ),
                    "positive_outcome_units": (
                        _unique_calibration_units(positive)
                    ),
                    "negative_outcome_units": (
                        _unique_calibration_units(negative)
                    ),
                    "censored_units": _unique_calibration_units(censored),
                }
            )

    unit_count = _unique_calibration_units(values)
    return {
        "status": "calibration_pending_no_stratified_model",
        "reason": (
            "context covariates are recorded for OOF coverage review; hard "
            "rules remain deterministic and no sparse subgroup map is fit. "
            "One calibration unit may occupy multiple feature-value buckets, "
            "so bucket counts are not additive across values."
        ),
        "episodes": unit_count,
        "unique_calibration_units": unit_count,
        "fit_eligible_calibration_units": _unique_calibration_units(
            values.loc[values["fit_eligible"]]
        ),
        "rows": int(len(values)),
        "strata": strata,
    }


def _uncertainty_payload(episodes: int) -> dict[str, Any]:
    count = max(0, int(episodes))
    return {
        "status": "authorized_formula_passthrough",
        "formula_version": UNCERTAINTY_FORMULA_VERSION,
        "formula": UNCERTAINTY_FORMULA,
        "episodes": count,
        "points": [
            {"raw_value": 0.0, "calibrated_value": 0.0, "episodes": count},
            {"raw_value": 1.0, "calibrated_value": 1.0, "episodes": count},
        ],
    }


def fit_typed_brain_calibration(
    *,
    row_paths: Sequence[str | Path],
    run_manifest: str | Path,
    output: str | Path,
    model_config: str | Path = "configs/model.json",
    validation_protocol: str | Path = "configs/data_splits.json",
    bins: int = 10,
    minimum_bin_samples: int = 20,
    calibration_version: str = "typed-monotone-logistic-isotonic",
    coverage_only: bool = False,
) -> dict[str, Any]:
    if bins < 2 or minimum_bin_samples < 1:
        raise ValueError("calibration sample and bin limits are invalid")
    if not calibration_version.strip() or calibration_version == "identity-unvalidated":
        raise ValueError("calibration_version must be an explicit authorized version")
    destination = Path(output)

    files = _input_files(row_paths)
    protocol = load_validation_protocol(validation_protocol)
    bindings = resolve_model_bindings(model_config)
    run_identity = _load_run_bindings(
        run_manifest,
        bindings,
        row_files=files,
    )
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
    frame["resolved_at"] = _aware_utc(
        frame["resolved_at"],
        "resolved_at",
        nullable=False,
    )
    frame["deadline"] = _aware_utc(
        frame["deadline"],
        "deadline",
        nullable=True,
    )
    if frame["sampled_at"].isna().any():
        raise ValueError("sampled_at contains invalid timestamps")
    if frame["resolved_at"].isna().any():
        raise ValueError("resolved_at contains invalid timestamps")
    _validate_run_window(frame, run_identity)

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
    if bool(
        (
            frame["fit_eligible"]
            & frame["dimension"].eq("delivery_quality")
            & frame["hard_barrier_before_target"]
        ).any()
    ):
        raise ValueError(
            "hard-barrier delivery rows are descriptive and cannot be fitted"
        )
    if not set(frame["target_deadline_kind"].astype(str)).issubset(
        {"thesis_deadline", "entry_deadline", "plan_deadline"}
    ):
        raise ValueError("target_deadline_kind is invalid")
    fitted_dimension = frame["dimension"].isin(FITTED_DIMENSIONS)
    if bool((fitted_dimension & frame["deadline"].isna()).any()):
        raise ValueError("fitted calibration targets require a frozen deadline")
    expected_deadline_kind = frame["dimension"].map(
        {
            "thesis_strength": "thesis_deadline",
            "location_quality": "entry_deadline",
            "entry_readiness": "entry_deadline",
            "delivery_quality": "plan_deadline",
        }
    )
    if bool(
        (
            fitted_dimension
            & frame["target_deadline_kind"].ne(expected_deadline_kind)
        ).any()
    ):
        raise ValueError(
            "fitted calibration target uses the wrong causal deadline kind"
        )
    non_future_target = (
        fitted_dimension
        & (frame["deadline"] <= frame["sampled_at"])
    )
    if bool(non_future_target.any()):
        raise ValueError(
            "fitted calibration targets must be registered before their deadline"
        )
    if bool(
        (
            fitted_dimension
            & (frame["resolved_at"] > frame["deadline"])
        ).any()
    ):
        raise ValueError(
            "fitted calibration targets cannot resolve after their deadline"
        )

    bound_window = protocol.classify_ohlcv(
        run_identity["window_start"],
        run_identity["window_end_exclusive"],
    )
    if bound_window.role != run_identity["window_role"]:
        raise ValueError(
            "Brain calibration run manifest window role does not match the "
            "registered data split"
        )
    latest_resolution = frame.loc[frame["fit_eligible"], "resolved_at"].max()
    if pd.isna(latest_resolution):
        latest_resolution = frame["sampled_at"].max()
    window = protocol.classify_ohlcv(
        frame["sampled_at"].min(),
        max(frame["sampled_at"].max(), latest_resolution) + pd.Timedelta(nanoseconds=1),
    )
    allowed_roles = (
        {
            "calibration",
            "brain_validation",
            "brain_calibration_trial",
        }
        if coverage_only
        else {"calibration"}
    )
    if window.role not in allowed_roles:
        raise ValueError(
            "typed Brain fitting may use only 2022 calibration rows; "
            "validation/trial rows are descriptive only, "
            f"got {window.role}"
        )
    if window.role != bound_window.role:
        raise ValueError(
            "Brain calibration rows and bound run window use different data splits"
        )

    thresholds = run_identity["fit_admission"]
    fit_admission = _fit_admission_funnel(
        frame,
        minimum_dimension_units=(
            None if coverage_only else thresholds.minimum_dimension_units
        ),
        minimum_plan_valid_roots=(
            None if coverage_only else thresholds.minimum_plan_valid_roots
        ),
        minimum_executable_episodes=(
            None
            if coverage_only
            else thresholds.minimum_executable_episodes
        ),
        window_role=window.role,
        enforce_thresholds=not coverage_only,
    )

    if coverage_only:
        coverage_payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "descriptive_coverage_only",
            "recorder_schema_version": RECORDER_SCHEMA_VERSION,
            "source_run_manifest": str(run_identity["path"]),
            "training_window_role": window.role,
            "training_start": frame["sampled_at"].min().isoformat(),
            "training_end": frame["resolved_at"].max().isoformat(),
            "playbooks": {
                playbook.value: _context_feature_coverage(
                    frame.loc[frame["playbook"].eq(playbook.value)]
                )
                for playbook in TYPED_ACTIVE_PLAYBOOKS
            },
            "fitted": False,
            "fit_admission": fit_admission,
            "threshold_search": False,
            "pnl_labels_used": False,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(coverage_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return coverage_payload

    if not fit_admission["fit_allowed"]:
        details = "; ".join(fit_admission["failure_reasons"])
        if not details:
            details = (
                f"window role {window.role!r} is not authorized for fitting"
            )
        raise CalibrationError(f"fit admission blocked: {details}")

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
            if dimension in {"thesis_strength", "delivery_quality"}:
                dimensions[dimension] = (
                    _regularized_monotone_logistic_payload(
                        values,
                        playbook=playbook,
                        dimension=dimension,
                        bins=bins,
                        minimum_dimension_units=(
                            thresholds.minimum_dimension_units
                        ),
                    )
                )
            else:
                dimensions[dimension] = _isotonic_dimension_payload(
                    values,
                    playbook=playbook,
                    dimension=dimension,
                    bins=bins,
                    minimum_bin_samples=minimum_bin_samples,
                    minimum_dimension_units=(
                        thresholds.minimum_dimension_units
                    ),
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
            "context_conditioning": _context_feature_coverage(
                frame.loc[frame["playbook"].eq(playbook.value)]
            ),
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
            "name": "typed_dimension_specific_monotone_maps",
            "fitted_dimensions": list(FITTED_DIMENSIONS),
            "thesis_and_delivery": (
                "l2_regularized_non_negative_slope_logistic"
            ),
            "location_and_readiness": (
                "tie_preserving_quantile_bins_weighted_pava"
            ),
            "sequence_progress": "deterministic_passthrough_not_fitted",
            "uncertainty": "authorized_contemporaneous_formula_passthrough",
            "uncertainty_formula_version": UNCERTAINTY_FORMULA_VERSION,
            "uncertainty_formula": UNCERTAINTY_FORMULA,
            "bins": bins,
            "minimum_bin_samples": minimum_bin_samples,
            "fit_admission_threshold_source": (
                "source_run_manifest.brain_calibration_fit_admission"
            ),
            "revision_weighting": (
                "equal_total_weight_per_causal_owner"
            ),
            "directions_pooled": True,
            "threshold_search": False,
            "context_strata_fitted": False,
            "hard_rules_calibrated": False,
            "hard_rules": [
                "frozen_source_invalidation",
                "protected_invalidation_breach",
                "hard_barrier_before_target",
                "same_bar_directional_ambiguity",
                "execution_hard_veto",
            ],
            "pnl_labels_used": False,
            "mbo_used": False,
        },
        "validation_schema_version": protocol.schema_version,
        "training_window_role": window.role,
        "training_start": frame["sampled_at"].min().isoformat(),
        "training_end": latest_resolution.isoformat(),
        "brain_target_protocol_versions": run_identity[
            "playbook_schema_versions"
        ],
        "playbook_registry_hash": bindings["registry_hash"],
        "playbook_registry_schema_version": run_identity[
            "registry_schema_version"
        ],
        "source_run_manifest": str(run_identity["path"]),
        "source_files": [str(path) for path in files],
        "fit_admission": fit_admission,
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
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument(
        "--validation-protocol",
        default="configs/data_splits.json",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--minimum-bin-samples", type=int, default=20)
    parser.add_argument(
        "--calibration-version",
        default="typed-monotone-logistic-isotonic",
    )
    parser.add_argument("--coverage-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = fit_typed_brain_calibration(
        row_paths=args.rows,
        run_manifest=args.run_manifest,
        output=args.output,
        model_config=args.model_config,
        validation_protocol=args.validation_protocol,
        bins=args.bins,
        minimum_bin_samples=args.minimum_bin_samples,
        calibration_version=args.calibration_version,
        coverage_only=args.coverage_only,
    )
    if args.coverage_only:
        print(
            json.dumps(
                {
                    playbook: values["unique_calibration_units"]
                    for playbook, values in payload["playbooks"].items()
                },
                sort_keys=True,
            )
        )
        return
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
