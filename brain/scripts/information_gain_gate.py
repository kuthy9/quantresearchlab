"""The information-gain gate: does Δₜ improve sixty-minute forecasts on event clocks?

Pre-registered in brain/docs/specs/2026-09-11-information-gain-gate-design.md.
M₀ = RAW + S (the existing raw and eye groups of ``predictability_gate``);
M₁ = M₀ + Δ (``brain/research/event_sequence``). Both are fitted and scored
on the same clock set. Verdict cells are the three first-passage targets on
C1, C5 and C15, Holm-corrected per model class; C60 and the every-minute
clock are reported on the primary fold only and never judged.

Usage::

    python -m brain.scripts.information_gain_gate --run-id <id>

where ``<id>`` is the directory ``brain/scripts/build_gate_blocks.py`` wrote
under ``outputs/information_gain_gate/``.
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

from brain.core.hypothesis_proposer import FEATURE_NAMES  # noqa: E402
from brain.research.event_log import TRANSITION_KINDS, load_blocks  # noqa: E402
from brain.research.event_sequence import clock_mask, sequence_features, without_kind  # noqa: E402
from brain.research.first_passage import (  # noqa: E402
    CLASS_COUNT,
    FIRST_PASSAGE_TARGETS,
    HORIZON_ATR_SCALE,
    first_passage_labels,
)
from brain.research.gate_family import (  # noqa: E402
    family_verdict,
    log_loss_rows,
    session_block_bootstrap,
)
from brain.research.gate_models import (  # noqa: E402
    Z_CLIP,
    fit_predict_proba,
    select_logistic_c,
)
from brain.scripts.predictability_gate import (  # noqa: E402
    HORIZON,
    TAPE_FEATURE_NAMES,
    Fold,
    build_folds,
    fit_predict,
    raw_features,
    session_labels,
    standardize_pair,
)

VERDICT_TARGETS: tuple[str, ...] = tuple(name for name, _, _ in FIRST_PASSAGE_TARGETS)
CONTINUOUS_TARGETS: tuple[str, ...] = ("asymmetry_60", "range_60")
CLOCKS: dict[str, int] = {"C1": 1, "C5": 5, "C15": 15}
# Reported, never judged: too thin (C60) or the dilution reference (ALL).
DESCRIPTIVE_CLOCKS: dict[str, int] = {"C60": 60, "ALL": 0}
PRIMARY: tuple[int, int] = (90, 20)        # train sessions, holdout sessions
ROLLING: tuple[int, int, int] = (60, 10, 10)  # train, holdout, step
MODELS: tuple[str, ...] = ("logistic", "lightgbm")
# A target whose primary training rows leave any class below this share is
# refused: it would be scored as a coin flip, not as a first passage.
MINIMUM_CLASS_SHARE = 0.05


class GateRunError(RuntimeError):
    pass


def continuous_targets(
    prices: np.ndarray, future_highs: np.ndarray, future_lows: np.ndarray
) -> dict[str, np.ndarray]:
    anchor = prices[:, 0].reshape(-1, 1)
    scale = prices[:, 3].reshape(-1, 1)
    up = np.maximum((future_highs[:, :HORIZON].max(axis=1).reshape(-1, 1) - anchor) / scale, 0.0)[:, 0]
    down = np.maximum((anchor - future_lows[:, :HORIZON].min(axis=1).reshape(-1, 1)) / scale, 0.0)[:, 0]
    span = up + down
    return {
        "asymmetry_60": np.divide(up - down, span, out=np.zeros_like(span), where=span > 1e-9),
        "range_60": span,
    }


def folds_for(
    index: pd.DatetimeIndex, mask: np.ndarray, *, primary: tuple[int, int], rolling: tuple[int, int, int],
    embargo_minutes: int = HORIZON,
) -> list[tuple[Fold, bool]]:
    """The primary fold, then the rolling folds, all on the clock's rows only.
    Row indices are mapped back to the full dataset."""

    rows = np.flatnonzero(mask)
    sub = index[rows]

    def remap(fold: Fold, number: int) -> Fold:
        return Fold(number, rows[fold.train], rows[fold.holdout], fold.train_sessions, fold.holdout_sessions)

    first = build_folds(
        sub, train_sessions=primary[0], holdout_sessions=primary[1],
        embargo_minutes=embargo_minutes, step_sessions=10_000, start_session=None,
    )[0]
    out: list[tuple[Fold, bool]] = [(remap(first, 0), True)]
    for number, fold in enumerate(
        build_folds(
            sub, train_sessions=rolling[0], holdout_sessions=rolling[1],
            embargo_minutes=embargo_minutes, step_sessions=rolling[2], start_session=None,
        ),
        start=1,
    ):
        out.append((remap(fold, number), False))
    return out


def run_gate(
    data: dict,
    *,
    out_dir: Path,
    clocks: dict[str, int] = CLOCKS,
    descriptive: dict[str, int] = DESCRIPTIVE_CLOCKS,
    primary: tuple[int, int] = PRIMARY,
    rolling: tuple[int, int, int] = ROLLING,
    models: tuple[str, ...] = MODELS,
    ablation_kinds: tuple[str, ...] = TRANSITION_KINDS,
    ablation_models: tuple[str, ...] = ("lightgbm",),
    log: Callable[[str], None] = lambda line: print(line, flush=True),
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DatetimeIndex(data["index"])
    index = index.tz_localize("UTC") if index.tz is None else index
    prices = np.asarray(data["prices"], dtype=float)
    volumes = np.asarray(data.get("volumes", np.ones(len(index))), dtype=float)
    events = data["events"]

    raw, raw_names = raw_features(
        closes=prices[:, 0], highs=prices[:, 1], lows=prices[:, 2],
        volumes=volumes, atrs=prices[:, 3], index=index,
    )
    keep = [i for i, name in enumerate(FEATURE_NAMES) if name not in TAPE_FEATURE_NAMES]
    base = np.hstack([raw, np.asarray(data["features"], dtype=float)[:, keep]]).astype(np.float32)
    started = time.monotonic()
    delta, delta_names = sequence_features(index, events)
    log(f"M0 {base.shape[1]} columns, delta {delta.shape[1]} columns ({time.monotonic() - started:.0f}s)")
    full = np.hstack([base, delta])

    labels = {
        name: first_passage_labels(
            prices=prices, future_highs=data["future_highs"], future_lows=data["future_lows"],
            up_atr=up, down_atr=down, unit=HORIZON_ATR_SCALE,
        )
        for name, up, down in FIRST_PASSAGE_TARGETS
    }
    continuous = continuous_targets(prices, data["future_highs"], data["future_lows"])
    sessions = session_labels(index)
    ordered_sessions = list(dict.fromkeys(sessions))
    primary_train = np.isin(sessions, ordered_sessions[: primary[0]])
    for name, y in labels.items():
        shares = np.bincount(y[primary_train], minlength=CLASS_COUNT) / max(int(primary_train.sum()), 1)
        log(f"{name}: class shares on the primary training rows (neither/upper/lower) {np.round(shares, 3).tolist()}")
        if shares.min() < MINIMUM_CLASS_SHARE:
            raise GateRunError(
                f"{name} is degenerate: a class holds {shares.min():.1%} of the primary "
                f"training rows (minimum {MINIMUM_CLASS_SHARE:.0%}); check the barrier unit"
            )

    results: list[dict] = []
    pooled: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    ablation_rows: list[dict] = []
    penalties: dict[tuple[str, str, str], float] = {}
    primary_cells: dict[tuple[str, str, str], tuple[Fold, np.ndarray]] = {}

    def row_losses(target: str, model: str, fold: Fold, x: np.ndarray, *, key: tuple[str, str, str]) -> np.ndarray:
        train_x, test_x = standardize_pair(x[fold.train], x[fold.holdout])
        # Δ has many near-constant columns (a kind seen a handful of times in
        # training). A z-score on such a column is arbitrarily large the first
        # time the kind fires out of sample, which is a numerical event, not
        # information; both matrices are clipped to +-10 standard deviations
        # for every model so the linear class cannot be undone by one column.
        np.clip(train_x, -Z_CLIP, Z_CLIP, out=train_x)
        np.clip(test_x, -Z_CLIP, Z_CLIP, out=test_x)
        if target in labels:
            y = labels[target]
            c = None
            if model == "logistic":
                # The penalty is chosen once per (clock, target, matrix) on
                # the primary fold's training rows and reused on the rolling
                # folds; nothing is selected on any holdout.
                c = penalties.get(key)
                if c is None:
                    c = select_logistic_c(train_x, y[fold.train])
                    penalties[key] = c
            proba = fit_predict_proba(model, train_x, y[fold.train], test_x, c=c)
            return log_loss_rows(proba, y[fold.holdout])
        y = continuous[target]
        regressor = "ridge" if model == "logistic" else "lightgbm"
        predicted = fit_predict(regressor, train_x, y[fold.train], test_x)
        return (y[fold.holdout] - predicted) ** 2

    for clock, minutes in {**clocks, **descriptive}.items():
        judged = clock in clocks
        mask = clock_mask(index, events, min_timeframe_minutes=minutes)
        log(f"{clock}: {int(mask.sum())} clocks of {mask.size} ({mask.mean():.1%})")
        for fold, is_primary in folds_for(index, mask, primary=primary, rolling=rolling):
            if not judged and not is_primary:
                continue
            for target in (*VERDICT_TARGETS, *CONTINUOUS_TARGETS):
                for model in models:
                    tick = time.monotonic()
                    m0 = row_losses(target, model, fold, base, key=(clock, target, f"{model}:m0"))
                    m1 = row_losses(target, model, fold, full, key=(clock, target, f"{model}:m1"))
                    diff = m1 - m0
                    hold_sessions = sessions[fold.holdout]
                    mean, low, high, p = session_block_bootstrap(diff, hold_sessions)
                    results.append(
                        {
                            "clock": clock, "target": target, "model": model, "fold": fold.index,
                            "primary": is_primary, "holdout_from": fold.holdout_sessions[0],
                            "holdout_to": fold.holdout_sessions[1], "rows": int(diff.size),
                            "m0_loss": float(m0.mean()), "m1_loss": float(m1.mean()),
                            "delta_logloss": float(mean), "ci_low": low, "ci_high": high, "p_raw": p,
                        }
                    )
                    log(
                        f"  {clock} fold {fold.index} {target:13s} {model:9s} "
                        f"M0 {m0.mean():.4f} M1 {m1.mean():.4f} delta {mean:+.4f} "
                        f"[{low:+.4f}, {high:+.4f}] ({time.monotonic() - tick:.0f}s)"
                    )
                    if is_primary and judged:
                        pooled.setdefault((clock, target, model), []).append((diff, hold_sessions))
                        primary_cells[(clock, target, model)] = (fold, m1)
    frame = pd.DataFrame(results)
    frame.to_csv(out_dir / "results.csv", index=False)
    verdicts = []
    for model in models:
        cells = frame[
            (frame["model"] == model) & frame["target"].isin(VERDICT_TARGETS) & frame["clock"].isin(clocks)
        ]
        family = {key: value for key, value in pooled.items() if key[2] == model}
        verdicts.append(family_verdict(cells, family, alpha=0.10))
    verdict = pd.concat(verdicts, ignore_index=True)
    verdict.to_csv(out_dir / "verdict.csv", index=False)
    log(verdict.to_string(index=False))

    # Ablation, after the verdict is on disk: M_Full - E_i on the primary
    # fold of every judged clock, one encoding per (clock, kind) and one
    # refit per verdict target and ablation model with the penalty the full
    # model chose. It informs, it does not judge.
    for clock in clocks:
        for kind in ablation_kinds:
            reduced, _ = sequence_features(index, without_kind(events, kind))
            reduced_full = np.hstack([base, reduced])
            for target in VERDICT_TARGETS:
                for model in ablation_models:
                    cell = primary_cells.get((clock, target, model))
                    if cell is None:
                        continue
                    fold, m1 = cell
                    ablated = row_losses(target, model, fold, reduced_full, key=(clock, target, f"{model}:m1"))
                    ablation_rows.append(
                        {
                            "clock": clock, "target": target, "model": model, "kind": kind,
                            "delta_logloss_vs_full": float((ablated - m1).mean()),
                        }
                    )
            if ablation_rows:
                pd.DataFrame(ablation_rows).sort_values(
                    ["clock", "target", "model", "delta_logloss_vs_full"],
                    ascending=[True, True, True, False],
                ).to_csv(out_dir / "ablation.csv", index=False)
            log(f"  ablation {clock} {kind} done")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/information_gain_gate")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument(
        "--ablation", default="all",
        help="'all' for every transition kind, 'none', or a comma-separated list of kinds",
    )
    parser.add_argument("--ablation-models", default="lightgbm")
    args = parser.parse_args()

    run_root = ROOT / args.output_root / args.run_id
    data = load_blocks(run_root / "blocks")
    meta = json.loads((run_root / "run.json").read_text())
    from shares.core.io import load_ohlcv

    first_block = meta["blocks"][0]["warmup_start"]
    last_block = meta["blocks"][-1]["end"]
    tape = load_ohlcv(meta["source"], start=first_block, end=last_block).frame
    # A snapshot's ``asof`` is the close of the bar that produced it, i.e. the
    # *next* bar's ``ts`` on the tape. The completed bar's volume sits one
    # minute earlier; reading at ``asof`` itself would read the future.
    completed = pd.DatetimeIndex(data["index"]) - pd.Timedelta(minutes=1)
    volumes = tape["volume"].reindex(completed.tz_convert(tape.index.tz)).to_numpy(dtype=float)
    covered = float(np.isfinite(volumes).mean())
    print(f"volume aligned on {covered:.1%} of clocks", flush=True)
    if covered < 0.99:
        raise GateRunError("volume alignment failed; check the tape window and the asof offset")
    data["volumes"] = np.where(np.isfinite(volumes), volumes, 1.0)

    if args.ablation == "all":
        kinds: tuple[str, ...] = TRANSITION_KINDS
    elif args.ablation == "none":
        kinds = ()
    else:
        kinds = tuple(k for k in args.ablation.split(",") if k)
    run_gate(
        data, out_dir=run_root, models=tuple(args.models.split(",")), ablation_kinds=kinds,
        ablation_models=tuple(args.ablation_models.split(",")),
    )


if __name__ == "__main__":
    main()
