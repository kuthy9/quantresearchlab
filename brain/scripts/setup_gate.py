"""The Setup first-passage gate: does a Group-5 Setup change where price goes first?

Pre-registered in brain/docs/specs/2026-09-13-setup-first-passage-gate-design.md.
M₀ = geometry (``setup_features.GEOMETRY_COLUMNS``); M₁ = M₀ + the Setup
(``setup_matrix``); M₂ = M₁ + the Eye state, reported only. Cells are
{K0, K1, K2} × {zone_return, pool_reversal}; a cell is judged only with every
class ≥ 5 % on its primary training rows and ≥ 200 primary OOS rows.

Usage::

    python -m brain.scripts.setup_gate --run-id <id>

where ``<id>`` is the directory ``build_gate_blocks.py --record-paths`` wrote
under ``outputs/setup_gate/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.research.event_log import load_blocks  # noqa: E402
from brain.research.first_passage import CLASS_COUNT  # noqa: E402
from brain.research.gate_family import family_verdict, log_loss_rows, session_block_bootstrap  # noqa: E402
from brain.research.gate_models import Z_CLIP, fit_predict_proba, minutes_of, select_logistic_c  # noqa: E402
from brain.research.setup_features import (  # noqa: E402
    analytic_target_probability,
    eye_state_matrix,
    geometry_matrix,
    setup_matrix,
)
from brain.research.setup_labels import (  # noqa: E402
    CENSORED,
    FAILURE_FIRST,
    HORIZON_MINUTES,
    LABEL_NAMES,
    TARGET_FIRST,
    label_instances,
)
from brain.scripts.information_gain_gate import folds_for  # noqa: E402
from brain.scripts.predictability_gate import Fold, session_labels, standardize_pair  # noqa: E402

CLOCKS: dict[str, tuple[str, ...]] = {
    "K0": ("zone_visible", "pool_swept"),
    "K1": ("reacceptance_held",),
    "K2": ("micro_break_observed",),
}
CONTEXT_KINDS: tuple[str, ...] = ("zone_return", "pool_reversal")
TARGET_NAME = "target_first"
PRIMARY: tuple[int, int] = (90, 20)
ROLLING: tuple[int, int, int] = (60, 10, 10)
MODELS: tuple[str, ...] = ("logistic", "lightgbm")
MINIMUM_CLASS_SHARE = 0.05
MINIMUM_OOS_ROWS = 200
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


class SetupGateError(RuntimeError):
    pass


def clock_instances(instances: pd.DataFrame, step_kinds: tuple[str, ...]) -> pd.DataFrame:
    kept = instances[(instances["drop_reason"] == "") & instances["step_kind"].isin(step_kinds)]
    kept = kept.assign(known_at=pd.to_datetime(kept["known_at"], utc=True))
    kept = kept.sort_values(["known_at", "sequence_id"], kind="stable")
    return kept.drop_duplicates("sequence_id", keep="first").reset_index(drop=True)


def _analytic_loss(instances: pd.DataFrame, rows: np.ndarray) -> float:
    """Two-class log-loss of the driftless ratio on resolved rows."""

    labels = instances["label"].to_numpy()[rows]
    resolved = labels != CENSORED
    if not resolved.any():
        return float("nan")
    p = np.clip(analytic_target_probability(instances.iloc[rows])[resolved], 1e-6, 1 - 1e-6)
    y = labels[resolved] == TARGET_FIRST
    return float(-np.mean(np.where(y, np.log(p), np.log(1 - p))))


def _descriptive(cell: str, sub: pd.DataFrame, fold: Fold | None, dropped: pd.Series) -> dict:
    row: dict = {"clock": cell, "rows": int(len(sub))}
    labels = sub["label"].to_numpy()
    for value, name in zip((TARGET_FIRST, FAILURE_FIRST, CENSORED), LABEL_NAMES):
        row[f"share_{name}"] = float(np.mean(labels == value)) if len(sub) else float("nan")
    row["share_same_bar"] = float(sub["same_bar"].astype(bool).mean()) if len(sub) else float("nan")
    for reason, count in dropped.items():
        row[f"dropped_{reason}"] = int(count)
    row["analytic_loss"] = _analytic_loss(sub, fold.holdout) if fold is not None else float("nan")
    for value, name in zip((TARGET_FIRST, FAILURE_FIRST, CENSORED), LABEL_NAMES):
        part = sub[labels == value]
        for column in ("time_to_resolve", "mae_atr"):
            for q in QUANTILES:
                row[f"{column}_p{int(q * 100)}_{name}"] = float(part[column].quantile(q)) if len(part) else float("nan")
    return row


def run_setup_gate(
    instances: pd.DataFrame,
    *,
    out_dir: Path,
    clocks: dict[str, tuple[str, ...]] = CLOCKS,
    context_kinds: tuple[str, ...] = CONTEXT_KINDS,
    primary: tuple[int, int] = PRIMARY,
    rolling: tuple[int, int, int] = ROLLING,
    models: tuple[str, ...] = MODELS,
    minimum_oos_rows: int = MINIMUM_OOS_ROWS,
    log: Callable[[str], None] = lambda line: print(line, flush=True),
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    pooled: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    descriptives: list[dict] = []
    refused: list[dict] = []
    judged_cells: dict[str, dict] = {}

    for clock, step_kinds in clocks.items():
        for kind in context_kinds:
            cell = f"{clock}:{kind}"
            raw = instances[(instances["context_kind"] == kind) & instances["step_kind"].isin(step_kinds)]
            dropped = raw.loc[raw["drop_reason"] != "", "drop_reason"].value_counts()
            sub = clock_instances(raw, step_kinds)
            if len(sub) == 0:
                log(f"{cell}: no instances")
                descriptives.append(_descriptive(cell, sub, None, dropped))
                refused.append({"clock": cell, "refusal": "no_rows", "oos_rows": 0, "min_class_share": float("nan")})
                continue
            index = pd.DatetimeIndex(sub["known_at"])
            y = sub["label"].to_numpy(dtype=int)
            sessions = session_labels(index)
            try:
                folds = folds_for(index, np.ones(len(sub), dtype=bool), primary=primary, rolling=rolling,
                                  embargo_minutes=HORIZON_MINUTES)
            except Exception as error:  # too few sessions for one fold
                log(f"{cell}: {error}")
                descriptives.append(_descriptive(cell, sub, None, dropped))
                refused.append({"clock": cell, "refusal": "folds", "oos_rows": 0, "min_class_share": float("nan")})
                continue
            first, _ = folds[0]
            shares = np.bincount(y[first.train], minlength=CLASS_COUNT) / max(first.train.size, 1)
            oos_rows = int(first.holdout.size)
            descriptives.append(_descriptive(cell, sub, first, dropped))
            log(f"{cell}: {len(sub)} instances, primary train {first.train.size} / OOS {oos_rows}, "
                f"class shares (target/failure/censored) {np.round(shares, 3).tolist()}")
            refusal = None
            if shares.min() < MINIMUM_CLASS_SHARE:
                refusal = "class_share"
            elif oos_rows < minimum_oos_rows:
                refusal = "oos_rows"
            if refusal is not None:
                log(f"{cell}: refused ({refusal}); reported, not judged")
                refused.append({"clock": cell, "refusal": refusal, "oos_rows": oos_rows, "min_class_share": float(shares.min())})
                continue
            judged_cells[cell] = {"oos_rows": oos_rows, "min_class_share": float(shares.min())}

            base = geometry_matrix(sub)
            increment, _ = setup_matrix(sub)
            full = np.hstack([base, increment])
            wide = np.hstack([full, eye_state_matrix(sub)])
            times = minutes_of(index)
            penalties: dict[str, float] = {}

            def losses(model: str, fold: Fold, x: np.ndarray, *, key: str) -> tuple[np.ndarray, float]:
                train_x, test_x = standardize_pair(x[fold.train], x[fold.holdout])
                np.clip(train_x, -Z_CLIP, Z_CLIP, out=train_x)
                np.clip(test_x, -Z_CLIP, Z_CLIP, out=test_x)
                c = None
                if model == "logistic":
                    c = penalties.get(key)
                    if c is None:
                        c = select_logistic_c(train_x, y[fold.train], times=times[fold.train], embargo_minutes=HORIZON_MINUTES)
                        penalties[key] = c
                proba = fit_predict_proba(model, train_x, y[fold.train], test_x, c=c,
                                          times=times[fold.train], embargo_minutes=HORIZON_MINUTES)
                accuracy = float(np.mean(proba.argmax(axis=1) == y[fold.holdout]))
                return log_loss_rows(proba, y[fold.holdout]), accuracy

            for fold, is_primary in folds:
                for model in models:
                    tick = time.monotonic()
                    m0, m0_acc = losses(model, fold, base, key=f"{model}:m0")
                    m1, m1_acc = losses(model, fold, full, key=f"{model}:m1")
                    m2, _ = losses(model, fold, wide, key=f"{model}:m2")
                    diff = m1 - m0
                    hold_sessions = sessions[fold.holdout]
                    mean, low, high, p = session_block_bootstrap(diff, hold_sessions)
                    results.append({
                        "clock": cell, "target": TARGET_NAME, "model": model, "fold": fold.index,
                        "primary": is_primary, "holdout_from": fold.holdout_sessions[0],
                        "holdout_to": fold.holdout_sessions[1], "rows": int(diff.size),
                        "m0_loss": float(m0.mean()), "m1_loss": float(m1.mean()), "m2_loss": float(m2.mean()),
                        "m0_accuracy": m0_acc, "m1_accuracy": m1_acc,
                        "analytic_loss": _analytic_loss(sub, fold.holdout),
                        "delta_logloss": float(mean), "ci_low": low, "ci_high": high, "p_raw": p,
                    })
                    log(f"  {cell} fold {fold.index} {model:9s} M0 {m0.mean():.4f} M1 {m1.mean():.4f} "
                        f"M2 {m2.mean():.4f} delta {mean:+.4f} [{low:+.4f}, {high:+.4f}] ({time.monotonic() - tick:.0f}s)")
                    if is_primary:
                        pooled.setdefault((cell, TARGET_NAME, model), []).append((diff, hold_sessions))

    frame = pd.DataFrame(results)
    frame.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(descriptives).to_csv(out_dir / "descriptives.csv", index=False)

    verdicts = []
    for model in models:
        cells = frame[(frame["model"] == model) & frame["clock"].isin(judged_cells)] if len(frame) else frame
        if len(cells) == 0:
            continue
        family = {key: value for key, value in pooled.items() if key[2] == model}
        verdicts.append(family_verdict(cells, family, alpha=0.10))
    verdict = pd.concat(verdicts, ignore_index=True) if verdicts else pd.DataFrame(
        columns=["clock", "target", "model", "primary_delta", "primary_ci_low", "primary_ci_high", "p_raw",
                 "folds", "folds_beating", "consistent", "robust", "p_holm_reject", "PASS"]
    )
    verdict["judged"] = True
    verdict["refusal"] = ""
    verdict["oos_rows"] = [judged_cells[c]["oos_rows"] for c in verdict["clock"]]
    verdict["min_class_share"] = [judged_cells[c]["min_class_share"] for c in verdict["clock"]]
    for item in refused:
        for model in models:
            verdict.loc[len(verdict)] = {
                "clock": item["clock"], "target": TARGET_NAME, "model": model, "primary_delta": float("nan"),
                "primary_ci_low": float("nan"), "primary_ci_high": float("nan"), "p_raw": float("nan"), "folds": 0,
                "folds_beating": float("nan"), "consistent": False, "robust": False, "p_holm_reject": False,
                "PASS": False, "judged": False, "refusal": item["refusal"], "oos_rows": item["oos_rows"],
                "min_class_share": item["min_class_share"],
            }
    for column in ("consistent", "robust", "p_holm_reject", "PASS", "judged"):
        verdict[column] = verdict[column].astype(bool)
    verdict = verdict.sort_values(["model", "clock"], kind="stable").reset_index(drop=True)
    verdict.to_csv(out_dir / "verdict.csv", index=False)
    log(verdict.to_string(index=False))
    log(f"VERDICT: {'PASS' if bool(verdict['PASS'].any()) else 'FAIL'}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/setup_gate")
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()

    run_root = ROOT / args.output_root / args.run_id
    log_path = run_root / "gate.log"
    handle = log_path.open("a", encoding="utf-8")

    def log(line: str) -> None:
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    data = load_blocks(run_root / "blocks")
    meta = json.loads((run_root / "run.json").read_text())
    from shares.core.io import load_ohlcv

    tape = load_ohlcv(meta["source"], start=meta["blocks"][0]["warmup_start"], end=meta["blocks"][-1]["end"]).frame
    paths = data["paths"]
    log(f"{len(paths)} path steps over {len(data['blocks'])} blocks; "
        f"context found on {float(paths['context_found'].astype(bool).mean()):.1%}")
    instances = label_instances(paths, tape)
    instances.to_parquet(run_root / "instances.parquet", index=False)
    kept = instances["drop_reason"] == ""
    log(f"{int(kept.sum())} labelled instances of {len(instances)}; drops: "
        f"{instances.loc[~kept, 'drop_reason'].value_counts().to_dict()}")
    run_setup_gate(instances, out_dir=run_root, models=tuple(args.models.split(",")), log=log)
    handle.close()


if __name__ == "__main__":
    main()
