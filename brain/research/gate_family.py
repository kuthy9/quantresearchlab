"""Scoring and the pre-registered family verdict of the information-gain gate.

Differences are per-row losses of M₁ minus M₀ on the same clocks, so a
negative mean is an improvement. Intervals resample whole sessions: rows
inside a session share a regime, and an i.i.d. interval on minute data is
far too narrow. The verdict is
brain/docs/specs/2026-09-11-information-gain-gate-design.md §5.9.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def log_loss_rows(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return -np.log(clipped[np.arange(labels.size), labels])


def _session_sums(differences: np.ndarray, sessions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(sessions, kind="stable")
    _, starts = np.unique(sessions[order], return_index=True)
    sums = np.add.reduceat(differences[order], starts)
    counts = np.diff(np.append(starts, sessions.size))
    return sums, counts


def session_block_bootstrap(
    differences: np.ndarray, sessions: np.ndarray, *, draws: int = 2000, seed: int = 29
) -> tuple[float, float, float, float]:
    """(mean, low95, high95, one-sided p): p is the share of resampled means
    at or above zero, i.e. the evidence against "M₁ is better"."""

    return pooled_session_bootstrap([(differences, sessions)], draws=draws, seed=seed)


def pooled_session_bootstrap(
    parts: Sequence[tuple[np.ndarray, np.ndarray]], *, draws: int = 2000, seed: int = 31
) -> tuple[float, float, float, float]:
    """One interval over every fold's rows, each fold resampled on its own
    sessions and then pooled, so no single fold carries the interval."""

    rng = np.random.default_rng(seed)
    totals = np.zeros(draws)
    weights = np.zeros(draws)
    for differences, sessions in parts:
        differences = np.asarray(differences, dtype=float)
        sessions = np.asarray(sessions)
        sums, counts = _session_sums(differences, sessions)
        picks = rng.integers(0, sums.size, size=(draws, sums.size))
        totals += sums[picks].sum(axis=1)
        weights += counts[picks].sum(axis=1)
    means = totals / weights
    low, high = np.percentile(means, [2.5, 97.5])
    everything = np.concatenate([np.asarray(d, dtype=float) for d, _ in parts])
    return float(everything.mean()), float(low), float(high), float((means >= 0.0).mean())


def holm(pvalues: Sequence[float], *, alpha: float) -> list[bool]:
    """Step-down Holm: reject in ascending order while p_(k) <= alpha / (m - k + 1)."""

    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    rejected = [False] * m
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            rejected[i] = True
        else:
            break
    return rejected


def family_verdict(
    results: pd.DataFrame,
    pooled: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]],
    *,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """One row per (clock, target, model) in ``results``; Holm runs across
    exactly the rows given, so the caller passes one family at a time."""

    rows = []
    for (clock, target, model), block in results.groupby(["clock", "target", "model"], sort=False):
        primary = block[block["primary"]]
        rolling = block[~block["primary"]]
        mean, low, high, p_raw = pooled_session_bootstrap(pooled[(clock, target, model)])
        primary_delta = float(primary["delta_logloss"].iloc[0]) if len(primary) else float("nan")
        beating = float((rolling["delta_logloss"] < 0.0).mean()) if len(rolling) else 0.0
        without_best = (
            float(rolling.sort_values("delta_logloss").iloc[1:]["delta_logloss"].mean())
            if len(rolling) > 1
            else float("nan")
        )
        rows.append(
            {
                "clock": clock,
                "target": target,
                "model": model,
                "primary_delta": primary_delta,
                "primary_ci_low": low,
                "primary_ci_high": high,
                "p_raw": p_raw,
                "folds": int(len(rolling)),
                "folds_beating": beating,
                "consistent": bool(beating >= 0.8),
                "robust": bool(np.isfinite(without_best) and without_best < 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm_reject"] = holm(frame["p_raw"].tolist(), alpha=alpha)
    frame["PASS"] = (
        (frame["primary_delta"] < 0.0)
        & (frame["primary_ci_high"] < 0.0)
        & frame["p_holm_reject"]
        & frame["consistent"]
        & frame["robust"]
    )
    return frame


__all__ = [
    "family_verdict",
    "holm",
    "log_loss_rows",
    "pooled_session_bootstrap",
    "session_block_bootstrap",
]
