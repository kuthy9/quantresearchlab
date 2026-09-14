"""Classifier fitting shared by the gates.

Penalty selection and early stopping are purged inside the training window
so the choice cannot see the holdout. The every-minute gates purge by a row
gap (sixty rows = sixty minutes); a gate whose rows are Setup instances
minutes or hours apart purges by clock instead: a row is kept for fitting
when its label window, ``embargo_minutes`` long, ends strictly before the
boundary's clock, so no bar is shared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.first_passage import CLASS_COUNT
from brain.research.gate_family import log_loss_rows

LOGISTIC_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0)
Z_CLIP = 10.0
# The sixty-minute horizon of the every-minute gates, as rows.
DEFAULT_GAP_ROWS = 60


class GateModelError(RuntimeError):
    pass


def minutes_of(index: pd.DatetimeIndex) -> np.ndarray:
    return (pd.DatetimeIndex(index).asi8 // 60_000_000_000).astype(np.int64)


def purged_before(
    rows_before: np.ndarray, boundary: int, *, times: np.ndarray | None,
    embargo_minutes: int, gap_rows: int,
) -> np.ndarray:
    """Rows (all positioned before ``boundary``) whose label window ends
    before the boundary's clock, or a plain row gap when no clock is given."""

    rows_before = np.asarray(rows_before, dtype=int)
    if times is None:
        return rows_before[rows_before < boundary - gap_rows]
    return rows_before[times[rows_before] + embargo_minutes < times[boundary]]


def purged_after(
    rows_after: np.ndarray, last_test: int, *, times: np.ndarray | None,
    embargo_minutes: int, gap_rows: int,
) -> np.ndarray:
    """Rows positioned after ``last_test`` far enough that the test rows'
    label windows cannot reach them."""

    rows_after = np.asarray(rows_after, dtype=int)
    if times is None:
        return rows_after[rows_after > last_test + gap_rows]
    return rows_after[times[rows_after] > times[last_test] + embargo_minutes]


def full_proba(estimator, x: np.ndarray, *, class_count: int = CLASS_COUNT) -> np.ndarray:
    """Probabilities over every class even when training saw fewer."""

    proba = np.full((x.shape[0], class_count), 1e-6)
    partial = estimator.predict_proba(x)
    for position, label in enumerate(estimator.classes_):
        proba[:, int(label)] = partial[:, position]
    return proba / proba.sum(axis=1, keepdims=True)


def select_logistic_c(
    train_x: np.ndarray, train_y: np.ndarray, *, times: np.ndarray | None = None,
    embargo_minutes: int = 240, class_count: int = CLASS_COUNT,
) -> float:
    """Blocked, purged selection of the L2 penalty, the same protocol
    ``predictability_gate.select_ridge_alpha`` uses for ridge."""

    from sklearn.linear_model import LogisticRegression

    rows = train_x.shape[0]
    blocks = 3
    edges = np.linspace(0, rows, blocks + 1, dtype=int)
    scores: dict[float, list[float]] = {c: [] for c in LOGISTIC_C_GRID}
    for b in range(blocks):
        lo, hi = edges[b], edges[b + 1]
        test = np.arange(lo, hi)
        if test.size == 0:
            continue
        train = np.concatenate(
            [
                purged_before(
                    np.arange(0, lo), lo, times=times, embargo_minutes=embargo_minutes,
                    gap_rows=DEFAULT_GAP_ROWS,
                ),
                purged_after(
                    np.arange(hi, rows), hi - 1, times=times, embargo_minutes=embargo_minutes,
                    gap_rows=DEFAULT_GAP_ROWS,
                ),
            ]
        )
        if train.size < 200 or test.size < 100 or len(np.unique(train_y[train])) < 2:
            continue
        for c in LOGISTIC_C_GRID:
            fitted = LogisticRegression(C=c, max_iter=500).fit(train_x[train], train_y[train])
            scores[c].append(
                -float(
                    log_loss_rows(
                        full_proba(fitted, train_x[test], class_count=class_count), train_y[test]
                    ).mean()
                )
            )
    means = {c: float(np.mean(v)) for c, v in scores.items() if v}
    if not means:
        return LOGISTIC_C_GRID[0]
    return max(means, key=means.get)


def fit_predict_proba(
    model: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *,
    c: float | None = None, times: np.ndarray | None = None, embargo_minutes: int = 240,
    class_count: int = CLASS_COUNT,
) -> np.ndarray:
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression

        penalty = (
            select_logistic_c(
                train_x, train_y, times=times, embargo_minutes=embargo_minutes, class_count=class_count,
            )
            if c is None
            else c
        )
        fitted = LogisticRegression(C=penalty, max_iter=500).fit(train_x, train_y)
        return full_proba(fitted, test_x, class_count=class_count)
    if model == "lightgbm":
        import lightgbm as lgb

        # Early-stop on a purged tail of the training window, as the
        # regression gate does: last fifth, a purge before it.
        cut = int(train_x.shape[0] * 0.8)
        fit = purged_before(
            np.arange(0, cut), cut, times=times, embargo_minutes=embargo_minutes, gap_rows=DEFAULT_GAP_ROWS,
        )
        fit_x, fit_y = train_x[fit], train_y[fit]
        tail_x, tail_y = train_x[cut:], train_y[cut:]
        objective = {"objective": "binary"} if class_count == 2 else {"objective": "multiclass", "num_class": class_count}
        estimator = lgb.LGBMClassifier(
            **objective, n_estimators=400,
            learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, random_state=0,
            verbose=-1,
        )
        estimator.fit(
            fit_x, fit_y, eval_set=[(tail_x, tail_y)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        return full_proba(estimator, test_x, class_count=class_count)
    raise GateModelError(f"unknown classifier {model}")


__all__ = [
    "DEFAULT_GAP_ROWS",
    "GateModelError",
    "LOGISTIC_C_GRID",
    "Z_CLIP",
    "fit_predict_proba",
    "full_proba",
    "minutes_of",
    "purged_after",
    "purged_before",
    "select_logistic_c",
]
