#!/usr/bin/env python3
"""Fit the preregistered 2023 monotone managed-policy gross-value maps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.calibration import (  # noqa: E402
    CalibrationError,
    model_code_fingerprint,
)
from smc_trader.model import Direction, Playbook  # noqa: E402
from smc_trader.playbook_registry import load_playbook_registry  # noqa: E402
from smc_trader.policy_value import (  # noqa: E402
    managed_policy_code_fingerprint,
    managed_policy_pipeline_fingerprint,
    monotone_policy_value_points,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--calibration-version",
        default="2.1.0-managed-policy-2023.1",
    )
    parser.add_argument(
        "--policy-value-protocol",
        default="configs/policy_value_protocol_v2_1.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2_1.json",
    )
    parser.add_argument(
        "--policy-base-config",
        default="configs/model_v2_1_policy_base.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite managed-policy calibration")
    episode_path = Path(args.episodes)
    manifest_path = Path(args.episode_manifest)
    protocol_path = Path(args.policy_value_protocol)
    base_config_path = Path(args.policy_base_config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    validation = load_validation_protocol(args.validation_protocol)
    registry = load_playbook_registry(
        base_config.get("playbook_registry", "configs/playbooks_v2.json")
    )
    bindings = {
        "status": "ready",
        "episode_table_sha256": _sha256(episode_path),
        "validation_protocol_hash": validation.fingerprint,
        "policy_value_protocol_hash": _sha256(protocol_path),
        "playbook_registry_hash": registry.fingerprint,
        "belief_path_code_hash": model_code_fingerprint(),
        "managed_policy_code_hash": managed_policy_code_fingerprint(),
        "managed_policy_pipeline_hash": managed_policy_pipeline_fingerprint(),
        "policy_base_config_hash": _sha256(base_config_path),
        "future_path_used": False,
        "holdout_used": False,
    }
    for field, expected in bindings.items():
        if manifest.get(field) != expected:
            raise ValueError(f"episode manifest {field} binding is stale")
    expected_window = validation.ohlcv_windows["managed_policy_calibration"]
    if manifest.get("window") != {
        "role": expected_window.role,
        "start": expected_window.start.isoformat(),
        "end_exclusive": expected_window.end_exclusive.isoformat(),
    }:
        raise ValueError("episode manifest window is not the complete 2023 window")

    episodes = pd.read_parquet(episode_path)
    required = {
        "episode_status",
        "fit_eligible",
        "playbook",
        "direction",
        "path_structural_score_R",
        "gross_R",
        "managed_action_gross_R",
        "managed_gross_R_clipped",
        "cost_R",
    }
    missing = sorted(required - set(episodes))
    if missing:
        raise ValueError(f"managed-policy episodes omit fields: {missing}")
    eligible = episodes.loc[
        episodes["episode_status"].isin(["closed", "unfilled"])
        & episodes["fit_eligible"].astype(bool)
    ].copy()
    if len(eligible) != int(
        manifest.get("fit_eligible_resolved_attempts", -1)
    ):
        raise ValueError("eligible managed-policy episode count is not conserved")
    if not eligible.empty and (
        pd.to_numeric(eligible["cost_R"], errors="raise").abs() > 1e-12
    ).any():
        raise ValueError("managed gross-value fitting cannot include execution costs")
    clip_low, clip_high = protocol["managed_policy_episode"]["label_clip_R"]
    action_gross = pd.to_numeric(
        eligible["managed_action_gross_R"],
        errors="raise",
    )
    if not (
        action_gross.loc[eligible["episode_status"].eq("unfilled")]
        .eq(0.0)
        .all()
    ):
        raise ValueError("unfilled enter actions must carry a 0R gross outcome")
    closed_rows = eligible["episode_status"].eq("closed")
    if not np.allclose(
        action_gross.loc[closed_rows],
        pd.to_numeric(
            eligible.loc[closed_rows, "gross_R"],
            errors="raise",
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("closed action labels differ from actual trade gross_R")
    recomputed_clip = action_gross.clip(float(clip_low), float(clip_high))
    if not np.allclose(
        recomputed_clip,
        pd.to_numeric(eligible["managed_gross_R_clipped"], errors="raise"),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("managed-policy clipped labels do not match the protocol")

    fit = protocol["fit"]
    minimum_playbook = int(fit["minimum_playbook_episodes"])
    minimum_direction = int(fit["minimum_direction_episodes"])
    playbooks: dict[str, dict] = {}
    for playbook in Playbook:
        values = eligible.loc[eligible["playbook"].eq(playbook.value)].copy()
        direction_counts = {
            direction.value: int(
                values["direction"].eq(direction.value).sum()
            )
            for direction in Direction
        }
        insufficiencies: list[str] = []
        if len(values) < minimum_playbook:
            insufficiencies.append(
                f"{len(values)} < {minimum_playbook} playbook episodes"
            )
        for direction, count in direction_counts.items():
            if count < minimum_direction:
                insufficiencies.append(
                    f"{direction} {count} < {minimum_direction} episodes"
                )
        if insufficiencies:
            playbooks[playbook.value] = {
                "status": "unavailable",
                "episodes": int(len(values)),
                "direction_episodes": direction_counts,
                "reason": "; ".join(insufficiencies),
                "points": [],
            }
            continue
        x = pd.to_numeric(
            values["path_structural_score_R"],
            errors="raise",
        ).to_numpy(float)
        y = pd.to_numeric(
            values["managed_gross_R_clipped"],
            errors="raise",
        ).to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(
                f"{playbook.value} managed-policy fit contains non-finite values"
            )
        try:
            points = monotone_policy_value_points(
                x,
                y,
                bins=int(fit["bins"]),
                minimum_bin_episodes=int(fit["minimum_bin_episodes"]),
                prior_mean_R=float(fit["smoothing_prior_mean_R"]),
                prior_weight=int(fit["smoothing_prior_weight"]),
            )
        except CalibrationError as exc:
            playbooks[playbook.value] = {
                "status": "unavailable",
                "episodes": int(len(values)),
                "direction_episodes": direction_counts,
                "reason": (
                    "monotone managed value has no usable discrimination: "
                    f"{exc}"
                ),
                "points": [],
            }
            continue
        fitted = np.interp(
            x,
            [point.structural_score_R for point in points],
            [point.managed_gross_R for point in points],
        )
        playbooks[playbook.value] = {
            "status": "ready",
            "episodes": int(len(values)),
            "direction_episodes": direction_counts,
            "score_support_R": [
                float(points[0].structural_score_R),
                float(points[-1].structural_score_R),
            ],
            "gross_R_mean": float(np.mean(y)),
            "gross_R_median": float(np.median(y)),
            "in_sample_value_mae_R_descriptive_only": float(
                np.mean(np.abs(fitted - y))
            ),
            "points": [
                {
                    "structural_score_R": point.structural_score_R,
                    "managed_gross_R": point.managed_gross_R,
                    "episodes": point.episodes,
                }
                for point in points
            ],
        }

    payload = {
        "calibration_version": str(args.calibration_version),
        "status": "ready",
        "method": {
            "name": fit["method"],
            "grouping": fit["grouping"],
            "bins": int(fit["bins"]),
            "minimum_bin_episodes": int(fit["minimum_bin_episodes"]),
            "minimum_playbook_episodes": minimum_playbook,
            "minimum_direction_episodes": minimum_direction,
            "smoothing_prior_mean_R": float(fit["smoothing_prior_mean_R"]),
            "smoothing_prior_weight": int(fit["smoothing_prior_weight"]),
            "threshold_or_pnl_search": False,
            "extrapolation": "forbidden",
        },
        "policy_value_protocol_hash": _sha256(protocol_path),
        "validation_protocol_hash": validation.fingerprint,
        "playbook_registry_hash": registry.fingerprint,
        "belief_path_code_hash": model_code_fingerprint(),
        "managed_policy_code_hash": managed_policy_code_fingerprint(),
        "managed_policy_pipeline_hash": managed_policy_pipeline_fingerprint(),
        "policy_base_config_hash": _sha256(base_config_path),
        "model_config_hash": _sha256(base_config_path),
        "source_sha256": manifest.get("source_sha256"),
        "episode_table": str(episode_path),
        "episode_table_hash": _sha256(episode_path),
        "episode_manifest": str(manifest_path),
        "episode_manifest_hash": _sha256(manifest_path),
        "training_window": manifest["window"],
        "playbooks": playbooks,
        "revealed_june_july_mbo_used_for_fit_or_selection": False,
        "future_path_used": False,
        "holdout_used": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                playbook: {
                    "status": values["status"],
                    "episodes": values["episodes"],
                }
                for playbook, values in playbooks.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
