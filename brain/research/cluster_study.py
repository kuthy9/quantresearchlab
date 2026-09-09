"""Choosing how finely to cut a continuum, and checking the cut survives.

Conditional future clouds are continuous rather than island-shaped, so there is
no natural number of clusters to discover — silhouette falls with ``k`` while
explained variance rises with it, and the two never agree. Any ``k`` is a
decision about resolution, not a fact recovered from the data.

So this module does not pick ``k`` by one score. It reports four things per
``k`` and leaves the trade-off visible:

* ``silhouette``          — how separated the cut is
* ``eta2_r60``            — how much of the realized sixty-minute return it explains
* ``reproduction``        — whether the same centroid shapes come back on a
                            window the basis was not fitted on
* ``stability``           — whether they come back under block resampling

``reproduction`` is the one that matters most. A resolution whose centroids do
not reappear out of sample is describing one window's noise, however tidy its
silhouette.

This is a research surface. It reads the future by construction and its output is
``shadow_only``.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from brain.research.forecast_index import PrincipalBasis


class ClusterStudyError(RuntimeError):
    """A study refuses to report a number it cannot compute honestly."""


def _fit(scores: np.ndarray, algorithm: str, k: int) -> np.ndarray:
    if algorithm == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(scores)
    if algorithm == "ward":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scores)
    if algorithm == "gmm":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(
            n_components=k, covariance_type="full", n_init=3, random_state=0
        ).fit_predict(scores)
    raise ClusterStudyError(f"unknown algorithm {algorithm!r}")


def medoids(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One representative row index per cluster, nearest to its centre.

    K-Medoids rather than a centroid: the average of two opposite futures is a
    third future that never happened, and a representative that never happened
    cannot be checked against anything.
    """

    representatives = []
    for label in sorted(set(int(v) for v in labels if v >= 0)):
        members = np.flatnonzero(labels == label)
        centre = scores[members].mean(axis=0)
        local = int(np.argmin(np.linalg.norm(scores[members] - centre, axis=1)))
        representatives.append(int(members[local]))
    return np.array(representatives, dtype=int)


def eta_squared(outcome: np.ndarray, labels: np.ndarray) -> float:
    """Share of the outcome's variance the partition explains."""

    mask = labels >= 0
    outcome, labels = outcome[mask], labels[mask]
    groups = sorted(set(labels.tolist()))
    if outcome.size < 10 or len(groups) < 2:
        return float("nan")
    mean = outcome.mean()
    between = sum(
        ((outcome[labels == g].mean() - mean) ** 2) * int((labels == g).sum())
        for g in groups
    )
    total = float(((outcome - mean) ** 2).sum())
    return float(between / total) if total > 0 else float("nan")


def centroid_reproduction(
    reference_curves: np.ndarray,
    candidate_curves: np.ndarray,
    *,
    gate: float,
) -> dict[str, float]:
    """How well one window's representative shapes reappear in another.

    Matches the two sets one-to-one by curve distance (Hungarian, so the answer
    does not depend on iteration order) and reports the mean matched distance
    and the share of shapes that found a partner inside ``gate``.
    """

    from scipy.optimize import linear_sum_assignment

    left = np.asarray(reference_curves, dtype=float)
    right = np.asarray(candidate_curves, dtype=float)
    if left.size == 0 or right.size == 0:
        return {"matched_fraction": float("nan"), "mean_distance": float("nan")}
    cost = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    # Per-point RMS distance, so the number does not grow with curve length and
    # stays readable in ATR units.
    cost = cost / np.sqrt(left.shape[1])
    rows, columns = linear_sum_assignment(cost)
    distances = np.array([cost[r, c] for r, c in zip(rows, columns)])
    return {
        "matched_fraction": float((distances <= gate).mean()),
        "mean_distance": float(distances.mean()),
    }


def block_stability(
    scores: np.ndarray,
    *,
    algorithm: str,
    k: int,
    block: int = 60,
    repeats: int = 8,
    keep_fraction: float = 0.7,
) -> float:
    """Mean adjusted Rand index between a full fit and contiguous-block refits.

    Blocks, not i.i.d. resampling: neighbouring observation points share almost
    their whole future window, so an i.i.d. bootstrap would report a stability
    the data does not have.
    """

    from sklearn.metrics import adjusted_rand_score

    reference = _fit(scores, algorithm, k)
    blocks = max(1, scores.shape[0] // block)
    results: list[float] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(1000 + repeat)
        mask = np.zeros(scores.shape[0], dtype=bool)
        for index in rng.choice(
            blocks, size=max(1, int(blocks * keep_fraction)), replace=False
        ):
            mask[index * block : (index + 1) * block] = True
        rows = np.flatnonzero(mask)
        if rows.size <= k:
            continue
        refit = _fit(scores[rows], algorithm, k)
        results.append(float(adjusted_rand_score(reference[rows], refit)))
    return float(np.mean(results)) if results else float("nan")


def sweep_cluster_count(
    *,
    fit_scores: np.ndarray,
    fit_curves: np.ndarray,
    fit_r60: np.ndarray,
    holdout_scores: np.ndarray | None = None,
    holdout_curves: np.ndarray | None = None,
    holdout_r60: np.ndarray | None = None,
    algorithms: Sequence[str] = ("kmeans", "ward", "gmm"),
    cluster_counts: Sequence[int] = tuple(range(2, 21)),
    reproduction_gate: float = 1.0,
    decimation: int = 60,
) -> pd.DataFrame:
    """Score every (algorithm, k) on separation, explanation and reproduction.

    ``decimation`` keeps one point per that many minutes for the metrics that
    assume independent samples. Consecutive observation points share fifty-nine
    of their sixty future minutes, so undecimated silhouette and eta-squared
    measure autocorrelation as much as structure.
    """

    from sklearn.metrics import silhouette_score

    fit_scores = np.asarray(fit_scores, dtype=float)
    keep = np.arange(0, fit_scores.shape[0], max(1, int(decimation)))
    rows: list[dict[str, Any]] = []

    for algorithm in algorithms:
        for k in cluster_counts:
            if k >= fit_scores.shape[0]:
                continue
            labels = _fit(fit_scores, algorithm, k)
            representatives = medoids(fit_scores, labels)
            record: dict[str, Any] = {
                "algorithm": algorithm,
                "k": int(k),
                "silhouette": float("nan"),
                "silhouette_disjoint": float("nan"),
                "eta2_r60": eta_squared(np.asarray(fit_r60, dtype=float), labels),
                "largest_cluster_share": float("nan"),
                "stability_ari": block_stability(fit_scores, algorithm=algorithm, k=k),
                "reproduction_matched": float("nan"),
                "reproduction_distance": float("nan"),
                "holdout_eta2_r60": float("nan"),
            }
            if len(set(labels.tolist())) >= 2:
                record["silhouette"] = float(silhouette_score(fit_scores, labels))
                if keep.size > k:
                    record["silhouette_disjoint"] = float(
                        silhouette_score(fit_scores[keep], labels[keep])
                    )
                sizes = np.array(
                    [int((labels == g).sum()) for g in sorted(set(labels.tolist()))],
                    dtype=float,
                )
                record["largest_cluster_share"] = float(sizes.max() / sizes.sum())

            if holdout_scores is not None and holdout_curves is not None:
                holdout_scores = np.asarray(holdout_scores, dtype=float)
                if k < holdout_scores.shape[0]:
                    holdout_labels = _fit(holdout_scores, algorithm, k)
                    holdout_reps = medoids(holdout_scores, holdout_labels)
                    record.update(
                        {
                            f"reproduction_{name}": value
                            for name, value in centroid_reproduction(
                                np.asarray(fit_curves, dtype=float)[representatives],
                                np.asarray(holdout_curves, dtype=float)[holdout_reps],
                                gate=reproduction_gate,
                            ).items()
                        }
                    )
                    record["reproduction_matched"] = record.pop(
                        "reproduction_matched_fraction", float("nan")
                    )
                    record["reproduction_distance"] = record.pop(
                        "reproduction_mean_distance", float("nan")
                    )
                    if holdout_r60 is not None:
                        record["holdout_eta2_r60"] = eta_squared(
                            np.asarray(holdout_r60, dtype=float), holdout_labels
                        )
            rows.append(record)
    return pd.DataFrame(rows)


def basis_report(basis: PrincipalBasis) -> pd.DataFrame:
    """Per-component explained variance, so the choice of five is inspectable."""

    ratios = np.asarray(basis.explained_variance_ratio, dtype=float)
    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(ratios.size)],
            "explained_variance_ratio": ratios,
            "cumulative": np.cumsum(ratios),
        }
    )


__all__ = [
    "ClusterStudyError",
    "basis_report",
    "block_stability",
    "centroid_reproduction",
    "eta_squared",
    "medoids",
    "sweep_cluster_count",
]
