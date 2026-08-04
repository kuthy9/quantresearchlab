#!/usr/bin/env python3
"""Stream the registered calibration window into frozen path-test artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.engine import ContinuousSMCEngine  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import content_hash, to_primitive  # noqa: E402
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.validation import (  # noqa: E402
    FrozenPathTestRecorder,
    PathTestResult,
    load_validation_protocol,
    records_frame,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _deadline(timestamp: pd.Timestamp) -> pd.Timestamp:
    local = timestamp.tz_convert("America/New_York")
    day = local.tz_localize(None).normalize()
    if local.hour >= 18:
        day += pd.Timedelta(days=1)
    return (day + pd.Timedelta(hours=17)).tz_localize(
        "America/New_York",
        ambiguous=True,
        nonexistent="shift_forward",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/model_v2.json")
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2.json",
    )
    parser.add_argument("--warmup-days", type=int, default=45)
    parser.add_argument("--progress-bars", type=int, default=100_000)
    parser.add_argument(
        "--window-role",
        default="calibration",
        help="exact preregistered OHLCV window used for path calibration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty calibration output: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    protocol = load_validation_protocol(args.validation_protocol)
    window = protocol.ohlcv_windows.get(args.window_role)
    if window is None:
        raise ValueError(
            f"validation protocol has no OHLCV window {args.window_role!r}"
        )
    source_hash = _sha256_file(args.source)
    if source_hash != protocol.causal_front_sha256:
        raise RuntimeError("calibration source is not the preregistered causal front")
    loaded = load_ohlcv(
        args.source,
        start=window.start - pd.Timedelta(days=args.warmup_days),
        end=window.end_exclusive,
    )
    if not loaded.contract_selection_causal:
        raise RuntimeError("calibration requires causal previous-session contracts")

    config_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    code_hash = model_code_fingerprint()
    engine = ContinuousSMCEngine.from_config(args.config)
    paths = FrozenPathTestRecorder(
        config_hash=config_hash,
        code_hash=code_hash,
    )
    frozen_setup_ids: set[str] = set()
    processed = 0
    observed = 0
    content_hashes = 0
    last_asof = window.start

    for bar in iter_completed_bars(loaded.frame):
        if bar.end >= window.end_exclusive:
            break
        paths.on_bar(bar)
        update = engine.reader.on_bar(bar)
        observation = engine.observer.observe(
            update,
            ExecutionRealityInput(
                spread_points=0.25,
                expected_slippage_points=0.0,
                deadline=_deadline(bar.end),
                source="constant_calibration_path_only",
            ),
        )
        if "contract_change_history_reset" in observation.anomalies:
            engine.brain.reset()
        belief = engine.brain.update(
            observation,
            position=None,
            scene_graph=engine.observer.scene_graph,
            scene_delta=engine.observer.last_scene_delta,
        )
        processed += 1
        if observation.asof < window.start:
            continue
        last_asof = observation.asof
        observed += 1
        eligible_hypotheses = {
            key: hypothesis
            for key, hypothesis in belief.hypotheses.items()
            if (
                hypothesis.sequence is not None
                and hypothesis.sequence.started_at is not None
                and hypothesis.sequence.started_at >= window.start
            )
        }
        new_complete_setups = {
            hypothesis.sequence.setup_id
            for hypothesis in eligible_hypotheses.values()
            if (
                hypothesis.sequence is not None
                and hypothesis.sequence.setup_id is not None
                and hypothesis.sequence.complete
                and hypothesis.plan is not None
                and hypothesis.sequence.setup_id not in frozen_setup_ids
            )
        }
        if new_complete_setups:
            snapshot_hash = content_hash(
                {
                    "observation": observation,
                    "belief": belief,
                    "config_hash": config_hash,
                    "code_hash": code_hash,
                }
            )
            content_hashes += 1
            snapshot = SimpleNamespace(
                observation=observation,
                belief=SimpleNamespace(hypotheses=eligible_hypotheses),
                snapshot_hash=snapshot_hash,
            )
            paths.observe(snapshot)
            frozen_setup_ids.update(new_complete_setups)
        if args.progress_bars > 0 and observed % args.progress_bars == 0:
            print(
                json.dumps(
                    {
                        "calibration_decision_bars": observed,
                        "resolved_paths": len(paths.results),
                        "open_paths": len(paths.open_tests),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    paths.close_unresolved(last_asof)
    path_frame = records_frame(paths.results, record_type=PathTestResult)
    path_frame.to_parquet(destination / "path_tests.parquet", index=False)
    eligible = (
        path_frame["outcome"].isin(["target", "invalidation", "deadline"])
        if not path_frame.empty
        else pd.Series(dtype=bool)
    )
    summary = {
        "version": "2.0.0-calibration-path-stream.1",
        "source": str(loaded.source),
        "source_sha256": source_hash,
        "validation_protocol_version": protocol.version,
        "validation_protocol_hash": protocol.fingerprint,
        "validation_window_role": window.role,
        "start": window.start.isoformat(),
        "end_exclusive": window.end_exclusive.isoformat(),
        "warmup_days": args.warmup_days,
        "processed_bars_including_warmup": processed,
        "calibration_decision_bars": observed,
        "path_tests": len(path_frame),
        "eligible_calibration_paths": int(eligible.sum()),
        "full_content_hashes_computed": content_hashes,
        "path_test_outcomes": (
            path_frame["outcome"].value_counts().sort_index().to_dict()
            if not path_frame.empty
            else {}
        ),
        "config_hash": config_hash,
        "model_code_hash": code_hash,
        "contract_selection_causal": loaded.contract_selection_causal,
        "execution_authority": False,
        "profitability_evaluated": False,
        "future_path_visible_to_model": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(to_primitive(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path_tests": len(path_frame),
                "eligible_calibration_paths": int(eligible.sum()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
