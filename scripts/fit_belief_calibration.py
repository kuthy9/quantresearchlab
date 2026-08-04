#!/usr/bin/env python3
"""Fit preregistered monotone belief reliability maps from frozen path tests."""
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

from smc_trader.calibration import monotone_reliability_points  # noqa: E402
from smc_trader.model import Playbook  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-tests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2.json",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--minimum-bin-episodes", type=int, default=20)
    parser.add_argument("--minimum-playbook-episodes", type=int, default=200)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument(
        "--source-summary",
        help=(
            "manifest-like replay summary proving the path-test source covers "
            "the complete requested calibration interval"
        ),
    )
    parser.add_argument(
        "--calibration-version",
        default="2.0.0-monotone-reliability.1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {destination}")
    frames = [pd.read_parquet(path) for path in args.path_tests]
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "playbook",
        "decision_time",
        "outcome",
        "success",
        "probability",
        "raw_probability",
        "calibration_version",
        "protocol_hash",
        "config_hash",
        "code_hash",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"path-test inputs omit fields: {missing}")
    frame["decision_time"] = pd.to_datetime(
        frame["decision_time"],
        errors="coerce",
        utc=True,
    )
    if frame["decision_time"].isna().any():
        raise ValueError("path-test inputs contain invalid decision clocks")
    if bool(args.start) != bool(args.end):
        raise ValueError("--start and --end must be provided together")
    requested_start = None
    requested_end = None
    if args.start:
        requested_start = pd.Timestamp(args.start)
        requested_end = pd.Timestamp(args.end)
        if (
            requested_start.tzinfo is None
            or requested_end.tzinfo is None
            or requested_end <= requested_start
        ):
            raise ValueError("calibration filter requires a positive aware interval")
        frame = frame.loc[
            frame["decision_time"].ge(requested_start)
            & frame["decision_time"].lt(requested_end)
        ].copy()
        if frame.empty:
            raise ValueError("calibration filter contains no path tests")
    eligible = frame.loc[
        frame["outcome"].isin(["target", "invalidation", "deadline"])
    ].copy()
    if eligible.empty:
        raise ValueError("no resolved path tests are eligible for calibration")
    protocol = load_validation_protocol(args.validation_protocol)
    window = protocol.classify_ohlcv(
        eligible["decision_time"].min(),
        eligible["decision_time"].max() + pd.Timedelta(nanoseconds=1),
    )
    if window.role not in {"calibration", "belief_calibration"}:
        raise ValueError(
            "belief calibration may use only a registered belief-calibration "
            f"window, got {window.role}"
        )
    if requested_start is not None and not (
        window.start == requested_start
        and window.end_exclusive == requested_end
    ):
        raise ValueError(
            "filtered belief calibration must use the entire registered window"
        )
    registry_hashes = set(eligible["protocol_hash"].astype(str))
    config_hashes = set(eligible["config_hash"].astype(str))
    code_hashes = set(eligible["code_hash"].astype(str))
    if (
        len(registry_hashes) != 1
        or len(config_hashes) != 1
        or len(code_hashes) != 1
    ):
        raise ValueError(
            "calibration inputs mix protocol, model-config, or model-code identities"
        )
    calibration_versions = set(eligible["calibration_version"].astype(str))
    if calibration_versions != {"identity-unvalidated"}:
        raise ValueError(
            "calibration must be fit from identity/unvalidated raw beliefs, "
            f"got {sorted(calibration_versions)}"
        )
    source_summary = None
    source_summary_path = None
    if requested_start is not None:
        if not args.source_summary:
            raise ValueError(
                "filtered belief calibration requires --source-summary"
            )
        source_summary_path = Path(args.source_summary)
        source_summary = json.loads(
            source_summary_path.read_text(encoding="utf-8")
        )
        summary_start = pd.Timestamp(source_summary.get("start"))
        summary_end = pd.Timestamp(source_summary.get("end_exclusive"))
        if (
            summary_start.tzinfo is None
            or summary_end.tzinfo is None
            or summary_start > requested_start
            or summary_end < requested_end
        ):
            raise ValueError(
                "path-test source summary does not cover the requested window"
            )
        summary_bindings = {
            "source_sha256": protocol.causal_front_sha256,
            "model_code_hash": next(iter(code_hashes)),
            "config_hash": next(iter(config_hashes)),
            "contract_selection_causal": True,
            "execution_authority": False,
            "profitability_evaluated": False,
            "future_path_visible_to_model": False,
        }
        for field, expected in summary_bindings.items():
            if source_summary.get(field) != expected:
                raise ValueError(
                    f"path-test source summary {field} binding is stale"
                )

    playbooks: dict[str, dict] = {}
    for playbook in Playbook:
        values = eligible.loc[eligible["playbook"].eq(playbook.value)].copy()
        if len(values) < args.minimum_playbook_episodes:
            raise ValueError(
                f"{playbook.value} has {len(values)} episodes; "
                f"{args.minimum_playbook_episodes} are preregistered as required"
            )
        raw = pd.to_numeric(
            values["raw_probability"],
            errors="raise",
        ).to_numpy(float)
        outcomes = values["success"].astype(bool).to_numpy(float)
        points = monotone_reliability_points(
            raw,
            outcomes,
            bins=args.bins,
            minimum_bin_episodes=args.minimum_bin_episodes,
        )
        calibrated = np.interp(
            raw,
            [point.raw_probability for point in points],
            [point.calibrated_probability for point in points],
        )
        playbooks[playbook.value] = {
            "episodes": len(values),
            "successes": int(outcomes.sum()),
            "base_rate": float(outcomes.mean()),
            "raw_brier": float(np.mean((raw - outcomes) ** 2)),
            "in_sample_calibrated_brier_descriptive_only": float(
                np.mean((calibrated - outcomes) ** 2)
            ),
            "points": [
                {
                    "raw_probability": point.raw_probability,
                    "calibrated_probability": point.calibrated_probability,
                    "episodes": point.episodes,
                }
                for point in points
            ],
        }
    payload = {
        "calibration_version": str(args.calibration_version),
        "status": "ready",
        "method": {
            "name": "fixed_quantile_bins_weighted_pava",
            "bins": args.bins,
            "minimum_bin_episodes": args.minimum_bin_episodes,
            "minimum_playbook_episodes": args.minimum_playbook_episodes,
            "beta_smoothing": "Beta(1,1)",
            "directions_pooled": True,
            "threshold_search": False,
        },
        "validation_protocol_version": protocol.version,
        "validation_protocol_hash": protocol.fingerprint,
        "training_window_role": window.role,
        "training_start": eligible["decision_time"].min().isoformat(),
        "training_end": eligible["decision_time"].max().isoformat(),
        "playbook_registry_hash": next(iter(registry_hashes)),
        "training_config_hash": next(iter(config_hashes)),
        "model_code_hash": next(iter(code_hashes)),
        "source_files": [
            {
                "path": str(Path(path)),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path in args.path_tests
        ],
        "source_summary": (
            None
            if source_summary_path is None
            else {
                "path": str(source_summary_path),
                "sha256": hashlib.sha256(
                    source_summary_path.read_bytes()
                ).hexdigest(),
            }
        ),
        "playbooks": playbooks,
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
                playbook: values["episodes"]
                for playbook, values in playbooks.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
