"""Telling a real change of claim from clustering jitter.

The pool re-extracts representative futures every clock, so every SPAWN, SPLIT,
MERGE and RETIRE has two possible causes: the conditional cloud genuinely moved,
or the same cloud simply landed in a different local optimum. A threshold tuned
without separating those two is tuned against noise.

The separation is measurable. Re-cluster the *same* cloud under different
K-Means initializations: whatever changes under that is jitter, because nothing
about the market changed between the runs. Whatever survives it is the cloud
actually moving.

``cloud_drift`` measures the other half — how far the retrieved neighbourhood
itself moved from one clock to the next. A lifecycle event on a clock with high
jitter and low drift is an artefact; one with low jitter and high drift is
information.

This is a research surface. Nothing here is a runtime authority, and the pool
never consults it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


class ChurnDiagnosticsError(RuntimeError):
    """A diagnostic refuses to report a number it cannot compute honestly."""


@dataclass(frozen=True)
class JitterSample:
    """How stable one clock's extraction is under re-initialization alone."""

    asof: pd.Timestamp
    label_agreement: float
    centroid_shift: float
    cluster_count: int


def cluster_jitter(
    scores: np.ndarray,
    *,
    cluster_count: int,
    restarts: int = 5,
    seeds: Sequence[int] = (0, 1, 2, 3),
) -> tuple[float, float]:
    """Agreement between repeated clusterings of one unchanged cloud.

    Returns the mean adjusted Rand index across seed pairs and the mean centroid
    displacement between them. An ARI near one means the extraction is a
    property of the cloud; well below one means the reported nodes are partly an
    artefact of where the algorithm started.
    """

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    data = np.asarray(scores, dtype=float)
    k = min(int(cluster_count), data.shape[0])
    if k < 2:
        return float("nan"), float("nan")

    labellings: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    for seed in seeds:
        model = KMeans(n_clusters=k, n_init=restarts, random_state=int(seed))
        labellings.append(model.fit_predict(data))
        centres.append(np.sort(model.cluster_centers_, axis=0))

    agreements = [
        float(adjusted_rand_score(labellings[i], labellings[j]))
        for i in range(len(labellings))
        for j in range(i + 1, len(labellings))
    ]
    shifts = [
        float(np.linalg.norm(centres[i] - centres[j]) / np.sqrt(k))
        for i in range(len(centres))
        for j in range(i + 1, len(centres))
    ]
    return (
        float(np.mean(agreements)) if agreements else float("nan"),
        float(np.mean(shifts)) if shifts else float("nan"),
    )


def cloud_drift(previous_rows: np.ndarray, current_rows: np.ndarray) -> float:
    """How much the retrieved neighbourhood itself turned over, in [0, 1].

    One minus the Jaccard overlap of the two retrieved index sets. Zero means
    the Brain is looking at exactly the same history it was looking at a minute
    ago, so any change in its claims came from somewhere else.
    """

    left = set(int(v) for v in np.asarray(previous_rows).ravel())
    right = set(int(v) for v in np.asarray(current_rows).ravel())
    if not left and not right:
        return 0.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return float(1.0 - len(left & right) / union)


def summarize_churn(records: pd.DataFrame) -> pd.DataFrame:
    """Per-operation churn rates against the jitter and drift they occurred under.

    ``records`` needs one row per lifecycle event with columns ``operation``,
    ``jitter_ari``, ``cloud_drift`` and ``association_distance``.
    """

    required = {"operation", "jitter_ari", "cloud_drift", "association_distance"}
    missing = required - set(records.columns)
    if missing:
        raise ChurnDiagnosticsError(f"churn records are missing {sorted(missing)}")
    grouped = records.groupby("operation")
    summary = grouped.agg(
        events=("operation", "size"),
        mean_jitter_ari=("jitter_ari", "mean"),
        mean_cloud_drift=("cloud_drift", "mean"),
        mean_association_distance=("association_distance", "mean"),
    )
    # An event that happens when the cloud has barely moved but the clustering
    # is unstable is the signature of an artefact.
    summary["artefact_suspicion"] = (
        (1.0 - summary["mean_jitter_ari"].clip(0.0, 1.0))
        * (1.0 - summary["mean_cloud_drift"].clip(0.0, 1.0))
    )
    return summary.reset_index()


def association_distance_profile(distances: Sequence[float]) -> pd.DataFrame:
    """Where an association gate would actually sit, read off the data.

    The gate separating "same claim" from "different claim" should be chosen
    from the distribution of observed match distances, not guessed. This reports
    the percentiles a gate would land on.
    """

    values = np.asarray([float(v) for v in distances], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ChurnDiagnosticsError("no finite association distances to profile")
    percentiles = (5, 10, 25, 50, 75, 90, 95, 99)
    return pd.DataFrame(
        {
            "percentile": percentiles,
            "association_distance": [
                float(np.percentile(values, p)) for p in percentiles
            ],
        }
    )


__all__ = [
    "ChurnDiagnosticsError",
    "JitterSample",
    "association_distance_profile",
    "cloud_drift",
    "cluster_jitter",
    "summarize_churn",
]
