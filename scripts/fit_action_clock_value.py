#!/usr/bin/env python3
"""Fit frozen v2.3 decomposed Q models with expanding temporal OOF."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.action_clock import (  # noqa: E402
    ActionClockProtocol,
    FEATURE_NAMES,
)
from smc_trader.action_value import (  # noqa: E402
    FrozenLinearModel,
    action_value_code_fingerprint,
    action_value_pipeline_fingerprint,
    fit_streaming_logistic,
    fit_streaming_ridge,
    runtime_semantics_fingerprint,
)
from smc_trader.artifact_stream import (  # noqa: E402
    atomic_bytes,
    canonical_json,
    sha256_file,
    verify_stream_shards,
)
from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.shadow_replay import (  # noqa: E402
    POSITION_ACTIONS,
    POSITION_FEATURE_NAMES,
)


ALTERNATIVES = (
    "wait_one_bar",
    "wait_better_price",
    "wait_reacceptance",
    "abstain",
)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman correlation without requiring SciPy."""

    paired = pd.concat(
        [
            pd.to_numeric(left, errors="coerce").rename("left"),
            pd.to_numeric(right, errors="coerce").rename("right"),
        ],
        axis=1,
    ).dropna()
    if (
        len(paired) < 2
        or paired["left"].nunique() < 2
        or paired["right"].nunique() < 2
    ):
        return float("nan")
    return float(
        paired["left"].rank(method="average").corr(
            paired["right"].rank(method="average")
        )
    )


def _load_manifest(root: Path, name: str) -> tuple[dict[str, Any], list[Path]]:
    path = root / f"{name}.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("stream") != name
    ):
        raise ValueError(f"invalid episode stream manifest: {path}")
    state = {
        "rows": int(payload["rows"]),
        "next_shard_index": len(payload["shards"]),
        "committed_shards": list(payload["shards"]),
    }
    verify_stream_shards(root, state)
    return payload, [root / str(item["path"]) for item in payload["shards"]]


def _episode_sources(
    roots: Sequence[Path],
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    output = {
        "flat_episode_shards": [],
        "delta_episode_shards": [],
        "position_episode_shards": [],
    }
    lineage: dict[str, Any] = {}
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    sample_semantics: dict[str, Any] | None = None
    semantic_binding_fields = (
        "source_sha256",
        "validation_protocol_hash",
        "belief_calibration_valid_from",
        "action_clock_protocol_hash",
        "action_equivalence_protocol_hash",
        "config_sha256",
        "base_model_code_hash",
        "action_equivalence_code_hash",
        "action_clock_code_hash",
        "shadow_replay_code_hash",
        "pipeline_hash",
        "policy_variant",
        "disabled_playbooks",
        "warmup_days",
        "hash_mode",
    )
    for root in roots:
        completion_path = root / "COMPLETED.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completion.get("status") != "complete"
            or completion.get("artifact") != "v2_3_action_clock_episodes"
        ):
            raise ValueError(f"episode root is incomplete: {root}")
        completion_bindings = completion.get("bindings")
        if not isinstance(completion_bindings, dict):
            raise ValueError(f"episode completion omits bindings: {root}")
        calibration_bindings = completion_bindings.get(
            "calibration_bindings"
        )
        if (
            not isinstance(calibration_bindings, dict)
            or calibration_bindings.get("policy_variant") != "all_three"
        ):
            raise ValueError(
                "primary v2.3 fit accepts only all-three episode roots"
            )
        current_semantics = {
            field: calibration_bindings.get(field)
            for field in semantic_binding_fields
        }
        if sample_semantics is None:
            sample_semantics = current_semantics
        elif current_semantics != sample_semantics:
            differing = [
                field
                for field in semantic_binding_fields
                if current_semantics[field] != sample_semantics[field]
            ]
            raise ValueError(
                "episode roots mix incompatible sample-generation semantics: "
                + ", ".join(differing)
            )
        root_lineage = {
            "completion_sha256": sha256_file(completion_path),
            "streams": {},
        }
        for name in output:
            manifest, paths = _load_manifest(root, name)
            expected_hashes = completion.get("stream_manifest_sha256")
            if (
                not isinstance(expected_hashes, dict)
                or expected_hashes.get(name)
                != sha256_file(root / f"{name}.manifest.json")
            ):
                raise ValueError(
                    f"episode completion does not bind {name} manifest"
                )
            output[name].extend(paths)
            root_lineage["streams"][name] = {
                "manifest_sha256": sha256_file(root / f"{name}.manifest.json"),
                "rows": int(manifest["rows"]),
            }
            if name == "flat_episode_shards" and paths:
                minimum: pd.Timestamp | None = None
                maximum: pd.Timestamp | None = None
                for shard in paths:
                    clocks = pd.to_datetime(
                        pd.read_parquet(
                            shard,
                            columns=["decision_time"],
                        )["decision_time"],
                        utc=True,
                        errors="coerce",
                    )
                    if clocks.isna().any():
                        raise ValueError(
                            f"episode shard has invalid clocks: {shard}"
                        )
                    if clocks.empty:
                        continue
                    low = pd.Timestamp(clocks.min())
                    high = pd.Timestamp(clocks.max())
                    minimum = low if minimum is None else min(minimum, low)
                    maximum = high if maximum is None else max(maximum, high)
                if minimum is not None and maximum is not None:
                    intervals.append((minimum, maximum, str(root)))
                    root_lineage["decision_time_min"] = minimum.isoformat()
                    root_lineage["decision_time_max"] = maximum.isoformat()
        lineage[str(root)] = root_lineage
    ordered = sorted(intervals, key=lambda item: item[0])
    for left, right in zip(ordered[:-1], ordered[1:]):
        if left[1] >= right[0]:
            raise ValueError(
                "episode roots have overlapping decision clocks: "
                f"{left[2]} and {right[2]}"
            )
    return output, lineage


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("fit interval clocks must be timezone aware")
    return timestamp.tz_convert("UTC")


def _factory(
    paths: Sequence[Path],
    *,
    feature_names: Sequence[str],
    target: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    eligible: str | None = None,
    action_id: str | None = None,
    require_structural_plan: bool = False,
    require_mbo: bool = False,
    label_time_column: str | None = None,
) -> Callable[[], Iterable[tuple[np.ndarray, np.ndarray]]]:
    names = tuple(feature_names)

    def batches() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        columns = {
            "decision_time",
            target,
            *names,
        }
        if eligible is not None:
            columns.add(eligible)
        if action_id is not None:
            columns.add("action_id")
        if require_structural_plan:
            columns.add("structural_plan_valid")
        if require_mbo:
            columns.add("mbo_available")
        if label_time_column is not None:
            columns.add(label_time_column)
        for path in paths:
            frame = pd.read_parquet(path, columns=sorted(columns))
            clocks = pd.to_datetime(
                frame["decision_time"],
                utc=True,
                errors="coerce",
            )
            mask = clocks.ge(start) & clocks.lt(end)
            if label_time_column is not None:
                label_clocks = pd.to_datetime(
                    frame[label_time_column],
                    utc=True,
                    errors="coerce",
                )
                mask &= label_clocks.notna() & label_clocks.lt(end)
            if eligible is not None:
                mask &= frame[eligible].fillna(False).astype(bool)
            if action_id is not None:
                mask &= frame["action_id"].eq(action_id)
            if require_structural_plan:
                mask &= frame["structural_plan_valid"].fillna(False).astype(bool)
            if require_mbo:
                mask &= pd.to_numeric(
                    frame["mbo_available"],
                    errors="coerce",
                ).ge(0.5)
            mask &= pd.to_numeric(frame[target], errors="coerce").notna()
            selected = frame.loc[mask]
            if selected.empty:
                continue
            matrix = selected[list(names)].to_numpy(dtype=float)
            values = pd.to_numeric(
                selected[target],
                errors="raise",
            ).to_numpy(dtype=float)
            yield matrix, values

    return batches


def _fit_logistic_checked(
    batches: Callable[[], Iterable[tuple[np.ndarray, np.ndarray]]],
    *,
    penalty: float,
) -> dict[str, Any]:
    model = fit_streaming_logistic(
        batches,
        ridge_lambda=penalty,
    )
    if not bool(model.get("converged")):
        raise ValueError("fixed logistic fit did not converge")
    return model


def _is_insufficient_sample_error(error: ValueError) -> bool:
    return "requires at least two samples" in str(error)


def _fit_flat_models(
    paths: Sequence[Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    penalty: float,
) -> dict[str, dict[str, Any]]:
    common = {
        "paths": paths,
        "feature_names": FEATURE_NAMES,
        "start": start,
        "end": end,
        "require_structural_plan": True,
        "label_time_column": "resolved_at",
    }
    output: dict[str, dict[str, Any]] = {}
    for action in (
        "enter_now",
        "wait_one_bar",
        "wait_better_price",
        "wait_reacceptance",
    ):
        action_models: dict[str, Any] = {
            "fill": _fit_logistic_checked(
                _factory(
                    **common,
                    target="fill_label",
                    eligible="fit_eligible_fill",
                    action_id=action,
                ),
                penalty=penalty,
            ),
            "conditional_gross": fit_streaming_ridge(
                _factory(
                    **common,
                    target="conditional_gross_R",
                    eligible="fit_eligible_conditional_gross",
                    action_id=action,
                ),
                ridge_lambda=penalty,
            ),
            "conditional_cost": None,
            "conditional_loss": _fit_logistic_checked(
                _factory(
                    **common,
                    target="loss_label",
                    eligible="fit_eligible_conditional_gross",
                    action_id=action,
                ),
                penalty=penalty,
            ),
        }
        try:
            action_models["conditional_cost"] = fit_streaming_ridge(
                _factory(
                    **common,
                    target="cost_R",
                    eligible="fit_eligible_cost",
                    action_id=action,
                ),
                ridge_lambda=penalty,
            )
        except ValueError as error:
            # Structural OOF folds before MBO coverage remain valid for gross
            # mapping, but their net Q is explicitly unavailable.
            if not _is_insufficient_sample_error(error):
                raise
            action_models["conditional_cost"] = None
        output[action] = action_models
    return output


def _fit_delta_models(
    paths: Sequence[Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    penalty: float,
    utility_family: str,
) -> dict[str, Any]:
    if utility_family not in {"gross", "net"}:
        raise ValueError("delta utility family must be gross or net")
    return {
        alternative: fit_streaming_ridge(
            _factory(
                paths,
                feature_names=FEATURE_NAMES,
                target=(
                    f"{utility_family}_delta_enter_vs_"
                    f"{alternative}_R"
                ),
                start=start,
                end=end,
                require_structural_plan=True,
                require_mbo=utility_family == "net",
                label_time_column="label_resolved_at",
            ),
            ridge_lambda=penalty,
        )
        for alternative in ALTERNATIVES
    }


def _fit_position_models(
    paths: Sequence[Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    penalty: float,
) -> dict[str, Any]:
    return {
        action: fit_streaming_ridge(
            _factory(
                paths,
                feature_names=POSITION_FEATURE_NAMES,
                target="gross_R",
                start=start,
                end=end,
                eligible="fit_eligible_gross",
                action_id=action,
                label_time_column="resolved_at",
            ),
            ridge_lambda=penalty,
            feature_names=POSITION_FEATURE_NAMES,
        )
        for action in POSITION_ACTIONS
    }


def _models_for_prediction(payload: dict[str, Any]) -> dict[
    str,
    dict[str, FrozenLinearModel],
]:
    return {
        action: {
            family: FrozenLinearModel.from_mapping(
                value,
                expected_feature_names=FEATURE_NAMES,
            )
            for family, value in action_models.items()
            if value is not None
        }
        for action, action_models in payload.items()
    }


def _predict_validation_flat(
    paths: Sequence[Path],
    models: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    fitted = _models_for_prediction(models)
    output: list[pd.DataFrame] = []
    columns = [
        "candidate_id",
        "action_id",
        "decision_time",
        "representative_playbook",
        "direction",
        "risk_passed",
        "structural_plan_valid",
        "status",
        "resolved_at",
        "right_censored",
        "filled",
        "gross_action_utility_R",
        "net_action_utility_R",
        "cost_R",
        *FEATURE_NAMES,
    ]
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        clocks = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        mask = (
            clocks.ge(start)
            & clocks.lt(end)
            & pd.to_datetime(
                frame["resolved_at"],
                utc=True,
                errors="coerce",
            ).lt(end)
            & frame["status"].eq("resolved")
            & ~frame["right_censored"].astype(bool)
            & frame["structural_plan_valid"].astype(bool)
        )
        frame = frame.loc[mask].copy()
        if frame.empty:
            continue
        abstain = frame["action_id"].eq("abstain").to_numpy()
        fill = np.zeros(len(frame), dtype=float)
        gross = np.zeros(len(frame), dtype=float)
        loss = np.zeros(len(frame), dtype=float)
        downside = np.zeros(len(frame), dtype=float)
        conditional_cost = np.full(len(frame), np.nan, dtype=float)
        mbo = frame["mbo_available"].to_numpy(dtype=float) >= 0.5
        for action, action_models in fitted.items():
            selected = frame["action_id"].eq(action).to_numpy()
            if not selected.any():
                continue
            matrix = frame.loc[selected, list(FEATURE_NAMES)]
            fill[selected] = action_models["fill"].predict(matrix)
            gross[selected] = action_models["conditional_gross"].predict(matrix)
            loss[selected] = action_models["conditional_loss"].predict(matrix)
            downside[selected] = (
                gross[selected]
                + float(
                    models[action]["conditional_gross"]["residual_q10"]
                )
            )
            selected_mbo = selected & mbo
            if action in {"enter_now", "wait_better_price"}:
                conditional_cost[selected_mbo] = frame.loc[
                    selected_mbo,
                    "expected_round_trip_cost_R",
                ].to_numpy(dtype=float)
            elif "conditional_cost" in action_models:
                conditional_cost[selected_mbo] = np.maximum(
                    0.0,
                    action_models["conditional_cost"].predict(
                        frame.loc[selected_mbo]
                    ),
                )
        frame["predicted_fill"] = fill
        frame["predicted_conditional_gross_R"] = gross
        frame["predicted_loss_probability"] = loss
        frame["predicted_gross_Q_R"] = fill * gross
        net = np.full(len(frame), np.nan, dtype=float)
        cost_available = mbo & np.isfinite(conditional_cost)
        net[cost_available] = (
            frame["predicted_gross_Q_R"].to_numpy(dtype=float)[
                cost_available
            ]
            - fill[cost_available] * conditional_cost[cost_available]
        )
        net[abstain] = 0.0
        frame["predicted_net_Q_R"] = net
        frame["predicted_downside_R"] = downside
        frame["predicted_conditional_cost_R"] = conditional_cost
        output.append(
            frame[
                [
                    "candidate_id",
                    "action_id",
                    "decision_time",
                    "representative_playbook",
                    "direction",
                    "risk_passed",
                    "filled",
                    "gross_action_utility_R",
                    "net_action_utility_R",
                    "cost_R",
                    "mbo_available",
                    "h4_directional_displacement_aligned",
                    "h4_path_efficiency",
                    "m5_compression",
                    "predicted_fill",
                    "predicted_conditional_gross_R",
                    "predicted_loss_probability",
                    "predicted_conditional_cost_R",
                    "predicted_gross_Q_R",
                    "predicted_net_Q_R",
                    "predicted_downside_R",
                ]
            ]
        )
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _predict_validation_deltas(
    paths: Sequence[Path],
    gross_models: dict[str, Any],
    net_models: dict[str, Any] | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    fitted_gross = {
        key: FrozenLinearModel.from_mapping(
            value,
            expected_feature_names=FEATURE_NAMES,
        )
        for key, value in gross_models.items()
    }
    fitted_net = {
        key: FrozenLinearModel.from_mapping(
            value,
            expected_feature_names=FEATURE_NAMES,
        )
        for key, value in (net_models or {}).items()
    }
    output: list[pd.DataFrame] = []
    columns = [
        "candidate_id",
        "decision_time",
        "label_resolved_at",
        "structural_plan_valid",
        *FEATURE_NAMES,
        *[
            f"gross_delta_enter_vs_{alternative}_R"
            for alternative in ALTERNATIVES
        ],
        *[
            f"net_delta_enter_vs_{alternative}_R"
            for alternative in ALTERNATIVES
        ],
    ]
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        clocks = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        mask = (
            clocks.ge(start)
            & clocks.lt(end)
            & pd.to_datetime(
                frame["label_resolved_at"],
                utc=True,
                errors="coerce",
            ).lt(end)
            & frame["structural_plan_valid"].astype(bool)
        )
        frame = frame.loc[mask].copy()
        if frame.empty:
            continue
        for alternative in ALTERNATIVES:
            frame[
                f"predicted_gross_delta_enter_vs_{alternative}_R"
            ] = fitted_gross[
                alternative
            ].predict(frame)
            frame[
                f"predicted_net_delta_enter_vs_{alternative}_R"
            ] = (
                fitted_net[alternative].predict(frame)
                if alternative in fitted_net
                else np.nan
            )
        output.append(
            frame[
                [
                    "candidate_id",
                    *[
                        field
                        for alternative in ALTERNATIVES
                        for field in (
                            f"gross_delta_enter_vs_{alternative}_R",
                            (
                                "predicted_gross_delta_enter_vs_"
                                f"{alternative}_R"
                            ),
                            (
                                "predicted_net_delta_enter_vs_"
                                f"{alternative}_R"
                            ),
                            f"net_delta_enter_vs_{alternative}_R",
                        )
                    ],
                ]
            ]
        )
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _predict_validation_position(
    paths: Sequence[Path],
    models: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold: int,
) -> pd.DataFrame:
    fitted = {
        action: FrozenLinearModel.from_mapping(
            value,
            expected_feature_names=POSITION_FEATURE_NAMES,
        )
        for action, value in models.items()
    }
    output: list[pd.DataFrame] = []
    columns = [
        "position_action_key",
        "position_state_id",
        "parent_action_key",
        "action_id",
        "decision_time",
        "resolved_at",
        "direction",
        "gross_R",
        "fit_eligible_gross",
        *POSITION_FEATURE_NAMES,
    ]
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        clocks = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        mask = (
            clocks.ge(start)
            & clocks.lt(end)
            & pd.to_datetime(
                frame["resolved_at"],
                utc=True,
                errors="coerce",
            ).lt(end)
            & frame["fit_eligible_gross"].fillna(False).astype(bool)
        )
        frame = frame.loc[mask].copy()
        if frame.empty:
            continue
        prediction = np.full(len(frame), np.nan, dtype=float)
        for action, model in fitted.items():
            selected = frame["action_id"].eq(action).to_numpy()
            if selected.any():
                prediction[selected] = model.predict(frame.loc[selected])
        frame["predicted_position_gross_Q_R"] = prediction
        frame["position_clock_key"] = (
            frame["parent_action_key"].astype(str)
            + "|"
            + frame["position_state_id"].astype(str)
            + "|"
            + pd.to_datetime(frame["decision_time"], utc=True).astype(str)
        )
        frame["position_expected_action_count"] = np.where(
            pd.to_numeric(
                frame["position_protection_available"],
                errors="coerce",
            ).ge(0.5),
            3,
            2,
        )
        frame["fold"] = int(fold)
        output.append(
            frame[
                [
                    "position_action_key",
                    "position_clock_key",
                    "action_id",
                    "decision_time",
                    "direction",
                    "gross_R",
                    "predicted_position_gross_Q_R",
                    "position_expected_action_count",
                    "fold",
                ]
            ]
        )
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _candidate_oof(
    action_predictions: pd.DataFrame,
    delta_predictions: pd.DataFrame,
    *,
    fold: int,
) -> pd.DataFrame:
    if action_predictions.empty or delta_predictions.empty:
        return pd.DataFrame()
    predicted = action_predictions.pivot(
        index="candidate_id",
        columns="action_id",
        values=["predicted_gross_Q_R", "predicted_net_Q_R"],
    )
    enter = action_predictions.loc[
        action_predictions["action_id"].eq("enter_now")
    ].copy()
    if enter["candidate_id"].duplicated().any():
        raise ValueError("OOF enter candidates are duplicated")
    enter = enter.set_index("candidate_id")
    for action in ("enter_now", *ALTERNATIVES):
        enter[f"predicted_gross_Q__{action}"] = predicted[
            ("predicted_gross_Q_R", action)
        ]
        enter[f"predicted_net_Q__{action}"] = predicted[
            ("predicted_net_Q_R", action)
        ]
    enter = enter.join(
        delta_predictions.set_index("candidate_id"),
        how="inner",
        validate="one_to_one",
    )
    decomposed_wins = np.logical_and.reduce(
        [
            enter["predicted_gross_Q__enter_now"].to_numpy()
            > enter[f"predicted_gross_Q__{alternative}"].to_numpy()
            for alternative in ALTERNATIVES
        ]
    )
    direct_wins = np.logical_and.reduce(
        [
            enter[
                f"predicted_gross_delta_enter_vs_{alternative}_R"
            ].to_numpy()
            > 0.0
            for alternative in ALTERNATIVES
        ]
    )
    direct_net_wins = np.logical_and.reduce(
        [
            enter[
                f"predicted_net_delta_enter_vs_{alternative}_R"
            ].to_numpy()
            > 0.0
            for alternative in ALTERNATIVES
        ]
    )
    enter["structural_positive_bucket"] = (
        (enter["predicted_gross_Q__enter_now"] > 0.0)
        & decomposed_wins
        & direct_wins
        & (enter["predicted_downside_R"] > -1.0)
    )
    enter["cost_stressed_utility_R"] = (
        enter["gross_action_utility_R"]
        - 0.20 * enter["filled"].astype(bool).astype(float)
    )
    net_wins = np.logical_and.reduce(
        [
            enter["predicted_net_Q__enter_now"].to_numpy()
            > enter[f"predicted_net_Q__{alternative}"].to_numpy()
            for alternative in ALTERNATIVES
        ]
    )
    enter["deployable_positive_bucket"] = (
        enter["structural_positive_bucket"]
        & enter["risk_passed"].astype(bool)
        & enter["predicted_net_Q__enter_now"].notna()
        & (enter["predicted_net_Q__enter_now"] > 0.0)
        & net_wins
        & direct_net_wins
    )
    displacement = enter["h4_directional_displacement_aligned"].abs()
    enter["regime"] = np.where(
        (displacement >= 0.35) & (enter["h4_path_efficiency"] >= 0.45),
        "directional_efficient",
        np.where(
            (displacement < 0.20) & (enter["m5_compression"] >= 0.35),
            "balanced",
            "transitional",
        ),
    )
    enter["fold"] = int(fold)
    return enter.reset_index()


def _group_metrics(
    frame: pd.DataFrame,
    column: str,
    *,
    value_column: str = "gross_action_utility_R",
) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for key, group in frame.groupby(column, dropna=False):
        values = pd.to_numeric(
            group[value_column],
            errors="coerce",
        ).dropna()
        output[str(key)] = {
            "actions": int(len(values)),
            "mean_R": None if values.empty else float(values.mean()),
        }
    return output


def _weekly_bootstrap_lower(
    frame: pd.DataFrame,
    *,
    value_column: str,
    seed: int = 2301,
    resamples: int = 2000,
) -> float | None:
    values = frame[["decision_time", value_column]].copy()
    values["decision_time"] = pd.to_datetime(values["decision_time"], utc=True)
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    values = values.dropna()
    if values.empty:
        return None
    values["week"] = values["decision_time"].dt.to_period("W-SUN").astype(str)
    weekly = values.groupby("week")[value_column].agg(["sum", "count"])
    if len(weekly) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    sums = weekly["sum"].to_numpy(dtype=float)
    counts = weekly["count"].to_numpy(dtype=float)
    for index in range(resamples):
        sample = rng.integers(0, len(weekly), size=len(weekly))
        means[index] = sums[sample].sum() / counts[sample].sum()
    return float(np.quantile(means, 0.025))


def _position_metrics(oof: pd.DataFrame) -> dict[str, Any]:
    if oof.empty:
        return {
            "position_oof_rows": 0,
            "position_oof_clocks": 0,
            "position_mapping_by_action": {},
            "position_nonhold_selected": 0,
            "position_incomplete_clocks_excluded": 0,
            "position_increment_vs_hold_mean_R": None,
            "position_increment_by_fold": {},
            "position_increment_by_direction": {},
            "position_weekly_bootstrap_lower_95_R": None,
        }
    mapping: dict[str, dict[str, float | int | None]] = {}
    for action, group in oof.groupby("action_id"):
        correlation = _spearman(
            group["predicted_position_gross_Q_R"],
            group["gross_R"],
        )
        mapping[str(action)] = {
            "actions": int(len(group)),
            "spearman": (
                None if pd.isna(correlation) else float(correlation)
            ),
        }
    selected_rows: list[dict[str, Any]] = []
    incomplete_clocks = 0
    for clock_key, group in oof.groupby("position_clock_key", sort=False):
        if group["action_id"].duplicated().any():
            raise ValueError(
                f"position OOF clock has duplicate actions: {clock_key}"
            )
        expected_counts = set(
            pd.to_numeric(
                group["position_expected_action_count"],
                errors="raise",
            ).astype(int)
        )
        if len(expected_counts) != 1:
            raise ValueError(
                f"position OOF clock disagrees on available actions: {clock_key}"
            )
        expected_count = next(iter(expected_counts))
        expected_actions = (
            {"hold", "exit", "protect"}
            if expected_count == 3
            else {"hold", "exit"}
        )
        actual_actions = set(group["action_id"].astype(str))
        # A decision near a fold boundary can have an immediately resolved EXIT
        # label while HOLD/PROTECT remains unresolved.  The action clock is not
        # eligible for same-clock policy comparison until every action that was
        # causally available at that clock has matured.
        if actual_actions != expected_actions:
            incomplete_clocks += 1
            continue
        indexed = group.set_index("action_id")
        best_action = str(
            indexed["predicted_position_gross_Q_R"].idxmax()
        )
        predicted_advantage = float(
            indexed.loc[best_action, "predicted_position_gross_Q_R"]
            - indexed.loc["hold", "predicted_position_gross_Q_R"]
        )
        selected_action = (
            best_action
            if best_action != "hold" and predicted_advantage >= 0.12
            else "hold"
        )
        selected_rows.append(
            {
                "decision_time": indexed.loc[
                    selected_action,
                    "decision_time",
                ],
                "direction": indexed.loc[selected_action, "direction"],
                "fold": int(indexed.loc[selected_action, "fold"]),
                "selected_action": selected_action,
                "increment_vs_hold_R": float(
                    indexed.loc[selected_action, "gross_R"]
                    - indexed.loc["hold", "gross_R"]
                ),
            }
        )
    selected = pd.DataFrame(selected_rows)
    nonhold = selected.loc[~selected["selected_action"].eq("hold")].copy()
    return {
        "position_oof_rows": int(len(oof)),
        "position_oof_clocks": int(len(selected)),
        "position_incomplete_clocks_excluded": int(incomplete_clocks),
        "position_mapping_by_action": mapping,
        "position_nonhold_selected": int(len(nonhold)),
        "position_increment_vs_hold_mean_R": (
            None
            if nonhold.empty
            else float(nonhold["increment_vs_hold_R"].mean())
        ),
        "position_increment_by_fold": _group_metrics(
            nonhold,
            "fold",
            value_column="increment_vs_hold_R",
        ),
        "position_increment_by_direction": _group_metrics(
            nonhold,
            "direction",
            value_column="increment_vs_hold_R",
        ),
        "position_weekly_bootstrap_lower_95_R": (
            _weekly_bootstrap_lower(
                nonhold,
                value_column="increment_vs_hold_R",
            )
        ),
    }


def _delta_mapping_metrics(oof: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if oof.empty:
        return output
    for alternative in ALTERNATIVES:
        families: dict[str, Any] = {}
        for family in ("gross", "net"):
            actual_column = (
                f"{family}_delta_enter_vs_{alternative}_R"
            )
            predicted_column = (
                f"predicted_{family}_delta_enter_vs_{alternative}_R"
            )
            paired = pd.DataFrame(
                {
                    "actual": pd.to_numeric(
                        oof[actual_column],
                        errors="coerce",
                    ),
                    "predicted": pd.to_numeric(
                        oof[predicted_column],
                        errors="coerce",
                    ),
                }
            )
            if family == "net":
                paired = paired.loc[
                    pd.to_numeric(
                        oof["mbo_available"],
                        errors="coerce",
                    ).ge(0.5)
                ]
            paired = paired.dropna()
            correlation = (
                None
                if len(paired) < 2
                else _spearman(
                    paired["predicted"],
                    paired["actual"],
                )
            )
            families[family] = {
                "paired_actions": int(len(paired)),
                "spearman": (
                    None
                    if correlation is None or pd.isna(correlation)
                    else float(correlation)
                ),
                "prediction_mean_R": (
                    None
                    if paired.empty
                    else float(paired["predicted"].mean())
                ),
                "realized_mean_R": (
                    None
                    if paired.empty
                    else float(paired["actual"].mean())
                ),
            }
        output[alternative] = families
    return output


def _metrics(
    oof: pd.DataFrame,
    position_oof: pd.DataFrame,
) -> dict[str, Any]:
    if oof.empty:
        output = {
            "oof_candidates": 0,
            "mapping_spearman": None,
            "incremental_delta_mapping": {},
            "structural_positive_candidates": 0,
            "deployable_positive_candidates": 0,
        }
        output.update(_position_metrics(position_oof))
        return output
    predicted = oof["predicted_gross_Q__enter_now"]
    actual = oof["gross_action_utility_R"]
    structural = oof.loc[oof["structural_positive_bucket"]].copy()
    deployable = oof.loc[oof["deployable_positive_bucket"]].copy()
    structural["calendar_year"] = pd.to_datetime(
        structural["decision_time"],
        utc=True,
    ).dt.year
    deployable["calendar_year"] = pd.to_datetime(
        deployable["decision_time"],
        utc=True,
    ).dt.year
    july = deployable.loc[
        pd.to_datetime(deployable["decision_time"], utc=True).between(
            pd.Timestamp("2024-07-01T00:00:00Z"),
            pd.Timestamp("2024-08-02T00:00:00Z"),
            inclusive="left",
        )
    ].copy()
    july["cost_stress_net_R"] = (
        july["gross_action_utility_R"]
        - 1.5 * pd.to_numeric(july["cost_R"], errors="coerce")
    )
    output = {
        "oof_candidates": int(len(oof)),
        "mapping_spearman": (
            None
            if len(oof) < 2
            else float(
                _spearman(pd.Series(predicted), pd.Series(actual))
            )
        ),
        "incremental_delta_mapping": _delta_mapping_metrics(oof),
        "structural_positive_candidates": int(len(structural)),
        "structural_positive_gross_mean_R": (
            None
            if structural.empty
            else float(structural["gross_action_utility_R"].mean())
        ),
        "structural_positive_by_fold": _group_metrics(structural, "fold"),
        "structural_positive_by_year": _group_metrics(
            structural,
            "calendar_year",
        ),
        "structural_positive_by_direction": _group_metrics(
            structural,
            "direction",
        ),
        "structural_positive_by_regime": _group_metrics(structural, "regime"),
        "cost_stressed_positive_by_fold": _group_metrics(
            structural,
            "fold",
            value_column="cost_stressed_utility_R",
        ),
        "cost_stressed_positive_by_year": _group_metrics(
            structural,
            "calendar_year",
            value_column="cost_stressed_utility_R",
        ),
        "cost_stressed_positive_by_direction": _group_metrics(
            structural,
            "direction",
            value_column="cost_stressed_utility_R",
        ),
        "cost_stressed_positive_by_regime": _group_metrics(
            structural,
            "regime",
            value_column="cost_stressed_utility_R",
        ),
        "cost_stressed_weekly_bootstrap_lower_95_R": (
            _weekly_bootstrap_lower(
                structural,
                value_column="cost_stressed_utility_R",
            )
        ),
        "deployable_positive_candidates": int(len(deployable)),
        "deployable_net_mean_R": (
            None
            if deployable.empty
            else float(
                pd.to_numeric(
                    deployable["net_action_utility_R"],
                    errors="coerce",
                ).mean()
            )
        ),
        "deployable_positive_by_fold": _group_metrics(
            deployable,
            "fold",
            value_column="net_action_utility_R",
        ),
        "deployable_positive_by_year": _group_metrics(
            deployable,
            "calendar_year",
            value_column="net_action_utility_R",
        ),
        "deployable_positive_by_direction": _group_metrics(
            deployable,
            "direction",
            value_column="net_action_utility_R",
        ),
        "deployable_positive_by_regime": _group_metrics(
            deployable,
            "regime",
            value_column="net_action_utility_R",
        ),
        "july_mbo_degradation_candidates": int(len(july)),
        "july_mbo_gross_mean_R": (
            None if july.empty else float(july["gross_action_utility_R"].mean())
        ),
        "july_mbo_net_mean_R": (
            None
            if july.empty
            else float(july["net_action_utility_R"].mean())
        ),
        "july_mbo_1_5x_cost_stress_mean_R": (
            None if july.empty else float(july["cost_stress_net_R"].mean())
        ),
    }
    output.update(_position_metrics(position_oof))
    return output


def _strict_group_gate(
    groups: dict[str, dict[str, Any]],
    *,
    minimum: int,
) -> bool:
    return bool(groups) and all(
        int(value["actions"]) >= minimum
        and value["mean_R"] is not None
        and float(value["mean_R"]) > 0.0
        for value in groups.values()
    )


def _release_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    structural_folds = metrics.get("structural_positive_by_fold", {})
    structural_years = metrics.get("structural_positive_by_year", {})
    structural_directions = metrics.get(
        "structural_positive_by_direction",
        {},
    )
    structural_regimes = metrics.get("structural_positive_by_regime", {})
    stressed_folds = metrics.get("cost_stressed_positive_by_fold", {})
    stressed_years = metrics.get("cost_stressed_positive_by_year", {})
    stressed_directions = metrics.get(
        "cost_stressed_positive_by_direction",
        {},
    )
    stressed_regimes = metrics.get(
        "cost_stressed_positive_by_regime",
        {},
    )
    position_mapping = metrics.get("position_mapping_by_action", {})
    position_folds = metrics.get("position_increment_by_fold", {})
    position_directions = metrics.get(
        "position_increment_by_direction",
        {},
    )
    delta_mapping = metrics.get("incremental_delta_mapping", {})

    def delta_family_ready(family: str, minimum: int) -> bool:
        return (
            set(delta_mapping) == set(ALTERNATIVES)
            and all(
                int(value[family]["paired_actions"]) >= minimum
                and value[family]["spearman"] is not None
                and float(value[family]["spearman"]) > 0.0
                for value in delta_mapping.values()
            )
        )

    return {
        "minimum_oof_candidate_clocks": int(
            metrics.get("oof_candidates", 0)
        )
        >= 5000,
        "mapping_spearman_positive": (
            metrics.get("mapping_spearman") is not None
            and float(metrics["mapping_spearman"]) > 0.0
        ),
        "gross_incremental_delta_mapping_positive": (
            delta_family_ready("gross", 500)
        ),
        "mbo_net_incremental_delta_mapping_positive": (
            delta_family_ready("net", 50)
        ),
        "structural_positive_bucket_each_fold": (
            set(structural_folds) == {"1", "2", "3", "4", "5"}
            and _strict_group_gate(structural_folds, minimum=50)
        ),
        "structural_positive_bucket_each_year": (
            set(structural_years) == {"2024", "2025", "2026"}
            and _strict_group_gate(structural_years, minimum=50)
        ),
        "structural_positive_bucket_both_directions": (
            set(structural_directions) == {"long", "short"}
            and _strict_group_gate(structural_directions, minimum=100)
        ),
        "structural_positive_bucket_registered_regimes": (
            set(structural_regimes)
            == {"directional_efficient", "balanced", "transitional"}
            and _strict_group_gate(structural_regimes, minimum=100)
        ),
        "cost_stressed_positive_bucket_each_fold": (
            set(stressed_folds) == {"1", "2", "3", "4", "5"}
            and _strict_group_gate(stressed_folds, minimum=50)
        ),
        "cost_stressed_positive_bucket_each_year": (
            set(stressed_years) == {"2024", "2025", "2026"}
            and _strict_group_gate(stressed_years, minimum=50)
        ),
        "cost_stressed_positive_bucket_both_directions": (
            set(stressed_directions) == {"long", "short"}
            and _strict_group_gate(stressed_directions, minimum=100)
        ),
        "cost_stressed_positive_bucket_registered_regimes": (
            set(stressed_regimes)
            == {"directional_efficient", "balanced", "transitional"}
            and _strict_group_gate(stressed_regimes, minimum=100)
        ),
        "weekly_cluster_bootstrap_lower_positive": (
            metrics.get(
                "cost_stressed_weekly_bootstrap_lower_95_R"
            )
            is not None
            and float(
                metrics[
                    "cost_stressed_weekly_bootstrap_lower_95_R"
                ]
            )
            > 0.0
        ),
        "revealed_july_mbo_net_positive": (
            int(metrics.get("july_mbo_degradation_candidates", 0)) >= 50
            and metrics.get("july_mbo_net_mean_R") is not None
            and float(metrics["july_mbo_net_mean_R"]) > 0.0
        ),
        "revealed_july_1_5x_cost_stress_positive": (
            int(metrics.get("july_mbo_degradation_candidates", 0)) >= 50
            and metrics.get("july_mbo_1_5x_cost_stress_mean_R") is not None
            and float(metrics["july_mbo_1_5x_cost_stress_mean_R"]) > 0.0
        ),
        "position_mapping_positive_each_action": (
            set(position_mapping) == set(POSITION_ACTIONS)
            and all(
                int(value["actions"]) >= 100
                and value["spearman"] is not None
                and float(value["spearman"]) > 0.0
                for value in position_mapping.values()
            )
        ),
        "position_nonhold_increment_positive_each_fold": (
            set(position_folds) == {"1", "2", "3", "4", "5"}
            and _strict_group_gate(position_folds, minimum=20)
        ),
        "position_nonhold_increment_positive_both_directions": (
            set(position_directions) == {"long", "short"}
            and _strict_group_gate(position_directions, minimum=50)
        ),
        "position_weekly_bootstrap_lower_positive": (
            metrics.get("position_weekly_bootstrap_lower_95_R") is not None
            and float(metrics["position_weekly_bootstrap_lower_95_R"]) > 0.0
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-root",
        action="append",
        required=True,
        help="Repeat for each completed chronological v2.3 episode root.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-quality-audit", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/action_clock_value_protocol_v2_3.json",
    )
    parser.add_argument(
        "--calibration-version",
        default="2.3.0-action-clock-walk-forward.1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite action-value artifact")
    protocol_path = Path(args.protocol)
    protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol = ActionClockProtocol.from_file(protocol_path)
    audit_path = Path(args.data_quality_audit)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("artifact") != "v2_3_action_clock_data_quality_audit"
        or audit.get("status") != "passed"
        or audit.get("profitability_evaluated") is not False
        or audit.get("action_clock_protocol_hash") != protocol.fingerprint
        or audit.get("audit_code_sha256")
        != sha256_file(ROOT / "scripts/audit_action_clock_data.py")
    ):
        raise ValueError("pre-fit action-clock data-quality audit did not pass")
    sources, lineage = _episode_sources(
        [Path(value) for value in args.episode_root]
    )
    audited_episode_hashes = {
        item.get("episode_completion_sha256")
        for item in audit.get("roots", ())
        if isinstance(item, dict)
    }
    fitted_episode_hashes = {
        item["completion_sha256"] for item in lineage.values()
    }
    if audited_episode_hashes != fitted_episode_hashes:
        raise ValueError("data-quality audit does not cover fitted episode roots")
    folds = protocol_payload["fit"]["walk_forward_folds"]
    penalty = protocol.regularization_lambda
    oof_rows: list[pd.DataFrame] = []
    position_oof_rows: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for index, fold in enumerate(folds, start=1):
        train_start = _utc(fold["train_start"])
        train_end = _utc(fold["train_end_exclusive"])
        validate_start = _utc(fold["validate_start"])
        validate_end = _utc(fold["validate_end_exclusive"])
        try:
            flat_models = _fit_flat_models(
                sources["flat_episode_shards"],
                start=train_start,
                end=train_end,
                penalty=penalty,
            )
            gross_delta_models = _fit_delta_models(
                sources["delta_episode_shards"],
                start=train_start,
                end=train_end,
                penalty=penalty,
                utility_family="gross",
            )
            try:
                net_delta_models = _fit_delta_models(
                    sources["delta_episode_shards"],
                    start=train_start,
                    end=train_end,
                    penalty=penalty,
                    utility_family="net",
                )
            except ValueError as error:
                if not _is_insufficient_sample_error(error):
                    raise
                net_delta_models = None
            action_predictions = _predict_validation_flat(
                sources["flat_episode_shards"],
                flat_models,
                start=validate_start,
                end=validate_end,
            )
            delta_predictions = _predict_validation_deltas(
                sources["delta_episode_shards"],
                gross_delta_models,
                net_delta_models,
                start=validate_start,
                end=validate_end,
            )
            candidates = _candidate_oof(
                action_predictions,
                delta_predictions,
                fold=index,
            )
            position_models = _fit_position_models(
                sources["position_episode_shards"],
                start=train_start,
                end=train_end,
                penalty=penalty,
            )
            position_predictions = _predict_validation_position(
                sources["position_episode_shards"],
                position_models,
                start=validate_start,
                end=validate_end,
                fold=index,
            )
            oof_rows.append(candidates)
            position_oof_rows.append(position_predictions)
            fold_reports.append(
                {
                    "fold": index,
                    "status": "evaluated",
                    "train_start": train_start.isoformat(),
                    "train_end_exclusive": train_end.isoformat(),
                    "validate_start": validate_start.isoformat(),
                    "validate_end_exclusive": validate_end.isoformat(),
                    "validation_candidates": int(len(candidates)),
                    "validation_position_rows": int(
                        len(position_predictions)
                    ),
                    "flat_action_model_samples": {
                        action: {
                            family: (
                                0
                                if model is None
                                else int(model["samples"])
                            )
                            for family, model in action_models.items()
                        }
                        for action, action_models in flat_models.items()
                    },
                }
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            fold_reports.append(
                {
                    "fold": index,
                    "status": "unavailable",
                    "reason": str(error),
                }
            )
    oof = (
        pd.concat(oof_rows, ignore_index=True)
        if oof_rows
        else pd.DataFrame()
    )
    position_oof = (
        pd.concat(position_oof_rows, ignore_index=True)
        if position_oof_rows
        else pd.DataFrame()
    )
    metrics = _metrics(oof, position_oof)
    metrics["folds"] = fold_reports
    gates = _release_gates(metrics)
    all_folds_evaluated = len(fold_reports) == len(folds) and all(
        item["status"] == "evaluated" for item in fold_reports
    )
    gates["all_folds_evaluated"] = all_folds_evaluated

    full_start = min(_utc(fold["train_start"]) for fold in folds)
    full_end = max(_utc(fold["validate_end_exclusive"]) for fold in folds)
    final_models: dict[str, Any] = {}
    final_error = None
    try:
        final_models["flat_actions"] = _fit_flat_models(
            sources["flat_episode_shards"],
            start=full_start,
            end=full_end,
            penalty=penalty,
        )
        missing_cost_actions = [
            action
            for action, models in final_models["flat_actions"].items()
            if models.get("conditional_cost") is None
        ]
        if missing_cost_actions:
            raise ValueError(
                "final flat action family lacks causal MBO cost samples: "
                + ", ".join(sorted(missing_cost_actions))
            )
        final_models["incremental_deltas"] = {
            family: _fit_delta_models(
                sources["delta_episode_shards"],
                start=full_start,
                end=full_end,
                penalty=penalty,
                utility_family=family,
            )
            for family in ("gross", "net")
        }
        final_models["position_actions"] = _fit_position_models(
            sources["position_episode_shards"],
            start=full_start,
            end=full_end,
            penalty=penalty,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        final_error = str(error)
        gates["final_models_fit"] = False
    else:
        gates["final_models_fit"] = True
    status = "ready" if all(gates.values()) else "unavailable"
    artifact = {
        "format_version": 1,
        "calibration_version": args.calibration_version,
        "status": status,
        "failure_behavior": (
            None
            if status == "ready"
            else "fail_closed_enter_and_protect_unavailable"
        ),
        "action_clock_protocol_hash": protocol.fingerprint,
        "runtime_semantics_hash": runtime_semantics_fingerprint(
            ROOT / "configs/model_v2_3_action_clock_base.json"
        ),
        "action_value_code_hash": action_value_code_fingerprint(),
        "base_model_code_hash": model_code_fingerprint(),
        "training_pipeline_hash": action_value_pipeline_fingerprint(),
        "fit_script_hash": sha256_file(Path(__file__)),
        "episode_lineage": lineage,
        "data_quality_audit_sha256": sha256_file(audit_path),
        "feature_names": list(FEATURE_NAMES),
        "position_feature_names": list(POSITION_FEATURE_NAMES),
        "models": final_models,
        "final_fit_error": final_error,
        "oof_metrics": metrics,
        "release_gates": gates,
        "holdout_used": False,
        "sealed_ohlcv_read": False,
        "sealed_mbo_read": False,
        "online_retraining": False,
        "hyperparameter_search": False,
        "neural_network": False,
    }
    atomic_bytes(output, canonical_json(_json_safe(artifact)))
    print(
        json.dumps(
            {
                "status": status,
                "release_gates": gates,
                "oof_candidates": metrics.get("oof_candidates", 0),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
