"""Scoring, the session-block bootstrap, Holm, and the family verdict."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.gate_family import (
    family_verdict,
    holm,
    log_loss_rows,
    pooled_session_bootstrap,
    session_block_bootstrap,
)


def test_log_loss_rows_is_negative_log_probability_of_the_true_class() -> None:
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    y = np.array([0, 2])
    assert np.allclose(log_loss_rows(p, y), -np.log([0.7, 0.8]))


def test_log_loss_rows_clips_a_zero_probability() -> None:
    p = np.array([[1.0, 0.0, 0.0]])
    assert np.isfinite(log_loss_rows(p, np.array([1]))).all()


def test_holm_rejects_in_step_down_order() -> None:
    assert holm([0.001, 0.02, 0.04, 0.5], alpha=0.10) == [True, True, True, False]
    assert holm([0.05, 0.05, 0.05], alpha=0.10) == [False, False, False]
    assert holm([0.5, 0.01], alpha=0.10) == [False, True]


def test_session_bootstrap_flags_a_clear_improvement() -> None:
    rng = np.random.default_rng(0)
    sessions = np.repeat(np.arange(20).astype(str), 50)
    differences = rng.normal(-0.05, 0.1, size=sessions.size)
    mean, low, high, p = session_block_bootstrap(differences, sessions)
    assert mean < 0 and high < 0 and p < 0.01


def test_session_bootstrap_does_not_flag_noise() -> None:
    rng = np.random.default_rng(1)
    sessions = np.repeat(np.arange(20).astype(str), 50)
    differences = rng.normal(0.0, 0.1, size=sessions.size)
    differences -= differences.mean()  # exactly zero mean by construction
    _, low, high, p = session_block_bootstrap(differences, sessions)
    assert low < 0 < high
    assert 0.3 < p < 0.7


def test_pooled_bootstrap_resamples_each_fold_on_its_own_sessions() -> None:
    rng = np.random.default_rng(2)
    parts = []
    for fold in range(3):
        sessions = np.repeat((np.arange(10) + 10 * fold).astype(str), 40)
        parts.append((rng.normal(-0.04, 0.1, sessions.size), sessions))
    mean, low, high, p = pooled_session_bootstrap(parts)
    assert mean < 0 and high < 0 and p < 0.01


def _results(delta_primary: float, rolling: list[float]) -> tuple[pd.DataFrame, dict]:
    rows = [{"clock": "C5", "target": "fp_1.0_1.0", "model": "lightgbm", "fold": 0, "primary": True,
             "delta_logloss": delta_primary, "holdout_from": "2022-05-10"}]
    rows += [{"clock": "C5", "target": "fp_1.0_1.0", "model": "lightgbm", "fold": i + 1, "primary": False,
              "delta_logloss": d, "holdout_from": f"2022-0{2 + i}-01"} for i, d in enumerate(rolling)]
    rng = np.random.default_rng(3)
    sessions = np.repeat(np.arange(20).astype(str), 40)
    pooled = {("C5", "fp_1.0_1.0", "lightgbm"): [(rng.normal(delta_primary, 0.05, sessions.size), sessions)]}
    return pd.DataFrame(rows), pooled


def test_family_verdict_passes_a_consistent_robust_significant_cell() -> None:
    results, pooled = _results(-0.03, [-0.02, -0.01, -0.03, -0.02, -0.01])
    verdict = family_verdict(results, pooled, alpha=0.10)
    assert bool(verdict.loc[0, "PASS"])


def test_family_verdict_fails_when_the_best_fold_carries_it() -> None:
    results, pooled = _results(-0.03, [-0.20, 0.01, 0.02, 0.01, 0.02])
    verdict = family_verdict(results, pooled, alpha=0.10)
    assert not bool(verdict.loc[0, "PASS"])
    assert not bool(verdict.loc[0, "consistent"])


def test_family_verdict_fails_when_the_primary_interval_covers_zero() -> None:
    results, pooled = _results(0.0, [-0.02, -0.01, -0.03, -0.02, -0.01])
    verdict = family_verdict(results, pooled, alpha=0.10)
    assert not bool(verdict.loc[0, "PASS"])
