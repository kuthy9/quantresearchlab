"""The measurements the v3 design rests on, separated from the scripts that run them.

Four questions, in the order they matter:

1. Does splitting the trajectory into a Direction channel and a detrended Shape
   channel give the pool something raw-curve PCA could not? Raw PCA put 80.7% of
   the variance on one axis that was essentially "where did it end", so distance
   became a quantization of direction and every question about *path* was
   answered by accident.
2. When a hypothesis survives a clock, is it the same hypothesis? Geometry says
   yes whenever the centroid barely moved. The supporting historical samples can
   say no.
3. Does the local cut need to adapt? A constant k manufactures structure on
   clocks where the cloud is one thing and hides it on clocks where it is three.
4. Does retrieval have any skill at all? If the conditional cloud is no closer
   to what actually happened than a random slice of history, the entire
   lifecycle is decoration.

Question four is the one that can end the project, so it is measured with a
paired test on out-of-sample clocks and reported without softening.

This is a research surface. It reads the future by construction and its output
is ``shadow_only``; it grants no research, empirical or trading authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from brain.core.trajectory import attributes_from_future, direction_vector, shape_matrix
from brain.research.churn_diagnostics import cluster_jitter
from brain.research.cluster_study import centroid_reproduction, medoids
from brain.research.forecast_index import fit_principal_basis, standardize
from contract.brain.forecast import (
    DIRECTION_DIM,
    SHAPE_COMPONENT_COUNT,
    TRAJECTORY_CURVE_LENGTH,
)


class DesignStudyError(RuntimeError):
    """A study refuses to report a number it cannot compute honestly."""


@dataclass(frozen=True)
class Representation:
    """One way of placing trajectories in a space, and how to reapply it."""

    name: str
    scores: np.ndarray
    project: object  # callable: (curves, attributes) -> scores
    leading_share: float

    def apply(self, curves: np.ndarray, attributes: list) -> np.ndarray:
        return self.project(curves, attributes)  # type: ignore[operator]


def path_attribute_rows(
    *,
    anchor_prices: np.ndarray,
    anchor_atrs: np.ndarray,
    future_closes: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
) -> list:
    """Describe every row's realized future once, for reuse by both channels."""

    return [
        attributes_from_future(
            anchor_price=float(anchor_prices[row]),
            anchor_atr=float(anchor_atrs[row]),
            closes=future_closes[row],
            highs=future_highs[row],
            lows=future_lows[row],
        )
        for row in range(len(anchor_prices))
    ]


def raw_representation(curves: np.ndarray, attributes: list) -> Representation:
    """The v2 design: principal components of the raw cumulative-return curve."""

    from sklearn.decomposition import PCA

    data = np.asarray(curves, dtype=float)
    model = PCA(n_components=SHAPE_COMPONENT_COUNT, svd_solver="full", random_state=0)
    scores = model.fit_transform(data)

    def project(new_curves: np.ndarray, _attributes: list) -> np.ndarray:
        return model.transform(np.asarray(new_curves, dtype=float))

    return Representation(
        name="raw_curve_pca",
        scores=scores,
        project=project,
        leading_share=float(model.explained_variance_ratio_[0]),
    )


def two_channel_representation(
    curves: np.ndarray, attributes: list, *, direction_weight: float = 0.5
) -> Representation:
    """The v3 design: Direction, plus the detrended Shape's principal basis."""

    curves = np.asarray(curves, dtype=float)
    direction_raw = np.array([direction_vector(item) for item in attributes], dtype=float)
    shapes = shape_matrix(curves)
    basis = fit_principal_basis(shapes)
    shape_raw = (shapes - basis.mean) @ basis.components.T

    _, direction_centre, direction_spread = standardize(direction_raw)
    _, shape_centre, shape_spread = standardize(shape_raw)
    direction_gain = math.sqrt(direction_weight / DIRECTION_DIM)
    shape_gain = math.sqrt((1.0 - direction_weight) / SHAPE_COMPONENT_COUNT)

    def project(new_curves: np.ndarray, new_attributes: list) -> np.ndarray:
        raw = np.array([direction_vector(item) for item in new_attributes], dtype=float)
        left = (raw - direction_centre) / direction_spread
        right = (
            (shape_matrix(np.asarray(new_curves, dtype=float)) - basis.mean)
            @ basis.components.T - shape_centre
        ) / shape_spread
        return np.hstack(
            [
                np.nan_to_num(left, nan=0.0, posinf=0.0, neginf=0.0) * direction_gain,
                np.nan_to_num(right, nan=0.0, posinf=0.0, neginf=0.0) * shape_gain,
            ]
        )

    scores = project(curves, attributes)
    # The leading share of a two-channel space is read the same way, so the two
    # designs can be compared on it: eigenvalues of the assembled coordinates.
    spectrum = np.linalg.svd(scores - scores.mean(axis=0), compute_uv=False) ** 2
    return Representation(
        name="direction_shape",
        scores=scores,
        project=project,
        leading_share=float(spectrum[0] / spectrum.sum()),
    )


def prototype_geometry(
    scores: np.ndarray, curves: np.ndarray, *, cluster_count: int
) -> pd.DataFrame:
    """What the prototypes of one representation actually look like.

    ``direction_span`` is how far apart the prototypes' endpoints are, and
    ``shape_span`` how far apart their detrended shapes are. A representation
    that has degenerated into direction quantization scores a wide direction
    span and a near-zero shape span: its prototypes differ only in where they
    ended, which is the failure this whole redesign exists to escape.
    """

    from sklearn.cluster import KMeans

    scores = np.asarray(scores, dtype=float)
    curves = np.asarray(curves, dtype=float)
    labels = KMeans(n_clusters=cluster_count, n_init=5, random_state=0).fit_predict(scores)
    rows = medoids(scores, labels)
    prototypes = curves[rows]
    shapes = shape_matrix(prototypes)
    endpoints = prototypes[:, -1]

    def _span(matrix: np.ndarray) -> float:
        if matrix.shape[0] < 2:
            return 0.0
        gram = np.linalg.norm(matrix[:, None] - matrix[None, :], axis=-1)
        return float(gram[np.triu_indices(matrix.shape[0], k=1)].mean())

    return pd.DataFrame(
        [
            {
                "cluster": int(label),
                "share": float((labels == label).mean()),
                "terminal_return": float(endpoints[position]),
                "shape_energy": float(np.abs(shapes[position]).mean()),
            }
            for position, label in enumerate(sorted(set(int(v) for v in labels)))
        ]
    ).assign(
        direction_span=_span(endpoints.reshape(-1, 1)),
        shape_span=_span(shapes),
    )


def leading_hypothesis_flips(
    *,
    neighbourhoods: list[np.ndarray],
    scores: np.ndarray,
    curves: np.ndarray,
    cluster_count: int,
    flip_gate: float = 0.5,
) -> float:
    """How often the top representative changes materially between clocks.

    A flip is not "the medoid row index changed" — the cloud is re-clustered
    every minute and the medoid is a discrete choice, so that number is mostly
    noise. It is "the published curve moved further than ``flip_gate`` ATR per
    point", which is a change a reader of the belief would actually notice.
    """

    from sklearn.cluster import KMeans

    previous: np.ndarray | None = None
    flips = 0
    compared = 0
    for rows in neighbourhoods:
        if rows.size < cluster_count:
            continue
        local = scores[rows]
        labels = KMeans(n_clusters=cluster_count, n_init=5, random_state=0).fit_predict(local)
        counts = np.bincount(labels)
        leading = int(np.argmax(counts))
        members = np.flatnonzero(labels == leading)
        centre = local[members].mean(axis=0)
        medoid = rows[members[int(np.argmin(np.linalg.norm(local[members] - centre, axis=1)))]]
        current = curves[medoid]
        if previous is not None:
            compared += 1
            distance = float(
                np.linalg.norm(current - previous) / math.sqrt(TRAJECTORY_CURVE_LENGTH)
            )
            flips += int(distance > flip_gate)
        previous = current
    if compared == 0:
        raise DesignStudyError("no consecutive clocks were comparable")
    return flips / compared


def representation_report(
    *,
    representation: Representation,
    fit_curves: np.ndarray,
    holdout_scores: np.ndarray,
    holdout_curves: np.ndarray,
    neighbourhoods: list[np.ndarray],
    cluster_count: int,
) -> dict[str, float]:
    """One row of the Raw-vs-two-channel comparison.

    ``neighbourhoods`` index the *fit* window: each is one holdout clock's
    retrieved conditional cloud. The flip rate is therefore measured on what the
    Brain would actually have published as the context moved, not on a sliding
    window of adjacent rows.
    """

    from sklearn.cluster import KMeans

    fit_labels = KMeans(n_clusters=cluster_count, n_init=5, random_state=0).fit_predict(
        representation.scores
    )
    holdout_labels = KMeans(n_clusters=cluster_count, n_init=5, random_state=0).fit_predict(
        holdout_scores
    )
    reproduction = centroid_reproduction(
        fit_curves[medoids(representation.scores, fit_labels)],
        holdout_curves[medoids(holdout_scores, holdout_labels)],
        gate=0.5,
    )
    jitter, shift = cluster_jitter(holdout_scores, cluster_count=cluster_count)
    geometry = prototype_geometry(
        representation.scores, fit_curves, cluster_count=cluster_count
    )
    return {
        "representation": representation.name,
        "leading_variance_share": representation.leading_share,
        "direction_span": float(geometry["direction_span"].iloc[0]),
        "shape_span": float(geometry["shape_span"].iloc[0]),
        "oos_reproduction_distance": reproduction["mean_distance"],
        "oos_reproduction_matched": reproduction["matched_fraction"],
        "cluster_jitter_ari": jitter,
        "cluster_jitter_shift": shift,
        "leading_flip_rate": leading_hypothesis_flips(
            neighbourhoods=neighbourhoods,
            scores=representation.scores,
            curves=fit_curves,
            cluster_count=cluster_count,
        ),
    }


def retrieval_skill(
    *,
    neighbourhoods: list[np.ndarray],
    reference_curves: np.ndarray,
    realized_curves: np.ndarray,
    draws: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Is the conditional cloud closer to what happened than random history?

    For every clock the retrieved cloud's mean curve is scored against the path
    that actually followed, and the same is done for a random cloud of the same
    size drawn from the same history, and for the unconditional mean of all of
    it. The comparison is paired — same clock, same realized future — so the
    difference is the retrieval's contribution and nothing else.

    Random clouds are drawn ``draws`` times per clock and averaged, so a lucky
    draw cannot decide the verdict.
    """

    reference_curves = np.asarray(reference_curves, dtype=float)
    realized_curves = np.asarray(realized_curves, dtype=float)
    if len(neighbourhoods) != realized_curves.shape[0]:
        raise DesignStudyError("one neighbourhood is required per realized path")
    rng = np.random.default_rng(seed)
    climatology = reference_curves.mean(axis=0)
    root = math.sqrt(TRAJECTORY_CURVE_LENGTH)

    rows = []
    for position, neighbours in enumerate(neighbourhoods):
        if neighbours.size == 0:
            continue
        actual = realized_curves[position]
        conditional = reference_curves[neighbours].mean(axis=0)
        random_errors = []
        random_signs = []
        for _ in range(draws):
            sample = rng.choice(
                reference_curves.shape[0], size=neighbours.size, replace=False
            )
            drawn = reference_curves[sample].mean(axis=0)
            random_errors.append(float(np.linalg.norm(actual - drawn) / root))
            random_signs.append(float(np.sign(drawn[-1]) == np.sign(actual[-1])))
        rows.append(
            {
                "conditional_rmse": float(np.linalg.norm(actual - conditional) / root),
                "random_rmse": float(np.mean(random_errors)),
                "climatology_rmse": float(np.linalg.norm(actual - climatology) / root),
                "conditional_sign": float(
                    np.sign(conditional[-1]) == np.sign(actual[-1])
                ),
                "random_sign": float(np.mean(random_signs)),
                "neighbours": int(neighbours.size),
            }
        )
    if not rows:
        raise DesignStudyError("no clock produced a usable neighbourhood")
    return pd.DataFrame(rows)


def skill_profile(
    *,
    pool_features: np.ndarray,
    pool_curves: np.ndarray,
    scored_features: np.ndarray,
    scored_curves: np.ndarray,
    neighbours: int = 200,
    stride: int = 15,
) -> dict[str, float]:
    """Magnitude-free skill of one retrieval pool against one set of clocks.

    RMSE alone cannot separate "has no information" from "has information and
    states it too boldly": a prediction of zero scores well simply by not
    committing. So this also reports the correlation between the predicted and
    realized sixty-minute return, the sign agreement, and the least-squares
    optimal scaling ``alpha`` of the conditional mean. An ``alpha`` at or below
    zero means the best available use of the prediction is to ignore it.

    ``neighbour_distance`` against ``context_scale`` says whether the retrieval
    found genuine analogues at all: neighbours as far away as two contexts
    picked at random are not precedent, and a failure there is a different
    problem from a failure of the futures to carry information.
    """

    reference, centre, scale = standardize(np.asarray(pool_features, dtype=float))
    query = np.nan_to_num(
        (np.asarray(scored_features, dtype=float) - centre) / scale,
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    pool_curves = np.asarray(pool_curves, dtype=float)
    scored_curves = np.asarray(scored_curves, dtype=float)
    rows = np.arange(0, query.shape[0], stride)
    take = min(neighbours, reference.shape[0])
    if rows.size < 2 or take < 2:
        raise DesignStudyError("a skill profile needs at least two clocks")

    predictions, distances = [], []
    for row in rows:
        spread = np.linalg.norm(reference - query[row], axis=1)
        head = np.argpartition(spread, take - 1)[:take]
        predictions.append(pool_curves[head].mean(axis=0))
        distances.append(float(spread[head].mean()))
    predicted = np.array(predictions)
    actual = scored_curves[rows]
    root = math.sqrt(TRAJECTORY_CURVE_LENGTH)
    denominator = float((predicted * predicted).sum())
    return {
        "clocks": int(rows.size),
        "neighbour_distance": float(np.mean(distances)),
        "context_scale": _typical_row_distance(reference),
        "correlation": float(np.corrcoef(predicted[:, -1], actual[:, -1])[0, 1]),
        "sign_agreement": float(
            np.mean(np.sign(predicted[:, -1]) == np.sign(actual[:, -1]))
        ),
        "optimal_alpha": float((actual * predicted).sum() / denominator)
        if denominator > 0
        else 0.0,
        "conditional_rmse": float(np.linalg.norm(actual - predicted, axis=1).mean() / root),
        "climatology_rmse": float(
            np.linalg.norm(actual - pool_curves.mean(axis=0), axis=1).mean() / root
        ),
    }


def _typical_row_distance(matrix: np.ndarray, *, sample: int = 400) -> float:
    data = np.asarray(matrix, dtype=float)
    take = min(sample, data.shape[0])
    rows = np.random.default_rng(0).choice(data.shape[0], size=take, replace=False)
    subset = data[rows]
    gram = np.linalg.norm(subset[:, None, :] - subset[None, :, :], axis=2)
    return float(np.median(gram[np.triu_indices(take, k=1)]))


def paired_verdict(frame: pd.DataFrame, *, left: str, right: str) -> dict[str, float]:
    """A paired comparison of two per-clock error columns.

    Reports the mean improvement, the share of clocks it holds on, and a
    block-bootstrap interval. Blocks, because consecutive observation points
    share fifty-nine of their sixty future minutes and an i.i.d. interval would
    be far too narrow to mean anything.
    """

    difference = (frame[left] - frame[right]).to_numpy(dtype=float)
    if difference.size < 2:
        raise DesignStudyError("a paired verdict needs at least two clocks")
    block = min(60, max(1, difference.size // 10))
    blocks = difference.size // block
    rng = np.random.default_rng(11)
    means = []
    for _ in range(2000):
        picked = rng.integers(0, max(1, blocks), size=max(1, blocks))
        means.append(
            float(
                np.mean(
                    np.concatenate(
                        [difference[i * block : (i + 1) * block] for i in picked]
                    )
                )
            )
        )
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_difference": float(difference.mean()),
        "share_improved": float((difference < 0).mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "significant": bool(high < 0.0),
    }


def local_cut_profile(clouds: list) -> pd.DataFrame:
    """How the adaptive cut behaved, and what the pool did with it."""

    return pd.DataFrame(
        [
            {
                "k_t": int(cloud.cluster_count),
                "nodes": len(cloud.nodes),
                "covered_mass": float(cloud.covered_mass),
                "neighbours": int(cloud.neighbour_count),
                "mean_neighbour_distance": float(cloud.mean_neighbour_distance),
            }
            for cloud in clouds
        ]
    )


def fixed_versus_adaptive(
    *,
    neighbourhoods: list[np.ndarray],
    scores: np.ndarray,
    proposer,
    fixed_counts: tuple[int, ...] = (2, 3, 4, 5, 6),
) -> pd.DataFrame:
    """Compare a fixed cut against the one the cloud chooses for itself.

    ``separation`` is the mean silhouette the cut achieved. A fixed k that beats
    the adaptive rule on separation would mean the rule is choosing badly; a
    fixed k that loses means the constant was manufacturing structure on clocks
    that did not have it.
    """

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows = []
    for rows_index in neighbourhoods:
        if rows_index.size < max(fixed_counts) + 1:
            continue
        local = scores[rows_index]
        labels, chosen, separation = proposer.local_cut(local)
        rows.append({"policy": "adaptive", "k": chosen, "separation": separation})
        for count in fixed_counts:
            fixed = KMeans(n_clusters=count, n_init=5, random_state=0).fit_predict(local)
            rows.append(
                {
                    "policy": f"fixed_{count}",
                    "k": count,
                    "separation": float(silhouette_score(local, fixed)),
                }
            )
    if not rows:
        raise DesignStudyError("no neighbourhood was large enough to cut")
    return pd.DataFrame(rows)


__all__ = [
    "DesignStudyError",
    "Representation",
    "fixed_versus_adaptive",
    "leading_hypothesis_flips",
    "local_cut_profile",
    "paired_verdict",
    "path_attribute_rows",
    "prototype_geometry",
    "raw_representation",
    "representation_report",
    "retrieval_skill",
    "skill_profile",
    "two_channel_representation",
]
