"""Discovering the trajectory modes the market actually produced.

HDBSCAN finds the leaf modes: it is density-based, so it does not have to be
told how many modes exist, and it is allowed to call a trajectory *noise*
instead of forcing it into a cluster.  That refusal is the point — an
observation the market only produced once is not a mode, and pretending
otherwise is how a hypothesis engine learns to be confidently wrong.

K-Medoids then picks each mode's representative.  A centroid would average two
opposite futures into a third that never happened; a medoid is always a real
observed trajectory.

The leaves are agglomerated (Ward, on the standardized medoids) into a binary
hierarchy.  The internal nodes are what give the pool its SPLIT and MERGE
structure: a parent is a coarser claim about the next hour, its two children
the finer alternatives it can decompose into.

``compare_algorithms`` exists because the choice above has to be defensible.
It scores HDBSCAN against K-Means, a Gaussian mixture and Ward on the same
vectors, including on a temporally decimated sample — adjacent observation
points share fifty-nine of their sixty future minutes, so any metric computed
on overlapping points measures autocorrelation as much as structure.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from contract.brain.forecast import (
    FORECAST_SCHEMA_VERSION,
    TRAJECTORY_COMPONENTS,
    TRAJECTORY_DIM,
    ModeLibrary,
    TrajectoryMode,
)

# The dispersion floor keeps a tight mode from claiming impossible precision:
# a mode whose members agree to within a hundredth of an ATR would otherwise
# reject every real path as a falsification.
MINIMUM_DISPERSION = 0.05


class ModeDiscoveryError(RuntimeError):
    """Discovery refuses to publish a library it cannot stand behind."""


def _require_sklearn() -> Any:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise ModeDiscoveryError(
            "mode discovery needs scikit-learn; install the 'research' extra "
            "(uv sync --extra research)"
        ) from error
    return sklearn


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns, returning the transformed matrix, centre and scale.

    NaN is imputed to the column centre, which makes a missing component
    contribute nothing to a distance instead of poisoning it.
    """

    data = np.asarray(matrix, dtype=float)
    centre = np.nanmean(data, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    scale = np.nanstd(data, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    z = (data - centre) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), centre, scale


def kmedoids(distances: np.ndarray, k: int, *, max_iterations: int = 100) -> np.ndarray:
    """Deterministic alternating k-medoids over a precomputed distance matrix.

    Seeded by the most central point and then by farthest-point traversal, so
    the result depends only on the data — no random state to record or drift.
    """

    size = distances.shape[0]
    if size == 0:
        raise ModeDiscoveryError("k-medoids needs at least one point")
    k = max(1, min(int(k), size))
    medoids = [int(np.argmin(distances.sum(axis=1)))]
    while len(medoids) < k:
        nearest = distances[:, medoids].min(axis=1)
        candidate = int(np.argmax(nearest))
        if candidate in medoids:
            break
        medoids.append(candidate)
    current = np.array(sorted(medoids))
    for _ in range(max_iterations):
        labels = np.argmin(distances[:, current], axis=1)
        updated = []
        for slot in range(len(current)):
            members = np.flatnonzero(labels == slot)
            if members.size == 0:
                updated.append(int(current[slot]))
                continue
            within = distances[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[int(np.argmin(within))]))
        updated_array = np.array(sorted(updated))
        if np.array_equal(updated_array, current):
            break
        current = updated_array
    return current


# A medoid search is O(n^2) in memory, and the Ward root spans every
# observation.  Above this many members the search runs on an evenly spaced
# subsample: the result is still a real observed trajectory, which is the
# property that matters, and the cost stays bounded.
MEDOID_SEARCH_LIMIT = 1500


def _medoid_of(rows: np.ndarray) -> int:
    """Index (within ``rows``) of the point minimizing total distance to the rest."""

    count = rows.shape[0]
    if count == 1:
        return 0
    if count > MEDOID_SEARCH_LIMIT:
        stride = np.linspace(0, count - 1, MEDOID_SEARCH_LIMIT).astype(int)
        sample = rows[stride]
        distances = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
        return int(stride[int(kmedoids(distances, 1)[0])])
    distances = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=-1)
    return int(kmedoids(distances, 1)[0])


def _dispersion(members: np.ndarray, medoid: np.ndarray) -> np.ndarray:
    """Per-component spread around the medoid, floored so it stays usable."""

    if members.shape[0] < 2:
        return np.full(TRAJECTORY_DIM, max(MINIMUM_DISPERSION, 1.0))
    deviation = np.sqrt(np.mean((members - medoid) ** 2, axis=0))
    return np.maximum(deviation, MINIMUM_DISPERSION)


@dataclass(frozen=True)
class DiscoveryConfig:
    """How permissive discovery is about calling something a mode."""

    min_cluster_size: int = 25
    min_samples: int | None = None
    cluster_selection_method: str = "eom"
    build_hierarchy: bool = True

    def __post_init__(self) -> None:
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if self.cluster_selection_method not in ("eom", "leaf"):
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")


@dataclass(frozen=True)
class DiscoveryResult:
    """A fitted library plus everything the runtime proposer needs to retrieve."""

    library: ModeLibrary
    assignments: tuple[str | None, ...]
    feature_centre: np.ndarray
    feature_scale: np.ndarray
    trajectory_centre: np.ndarray
    trajectory_scale: np.ndarray


def _library_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def discover_modes(
    *,
    trajectories: np.ndarray,
    features: np.ndarray,
    fitted_at: pd.Timestamp,
    config: DiscoveryConfig | None = None,
    library_id: str = "natural_hypothesis_modes_v1",
) -> DiscoveryResult:
    """Fit the mode library from realized trajectories.

    ``features`` is not clustered; it is only standardized here so the runtime
    proposer retrieves in exactly the space discovery recorded.
    """

    _require_sklearn()
    from sklearn.cluster import HDBSCAN

    config = config or DiscoveryConfig()
    trajectories = np.asarray(trajectories, dtype=float)
    if trajectories.ndim != 2 or trajectories.shape[1] != TRAJECTORY_DIM:
        raise ModeDiscoveryError(
            f"trajectories must be (n, {TRAJECTORY_DIM}), got {trajectories.shape}"
        )
    if trajectories.shape[0] < config.min_cluster_size:
        raise ModeDiscoveryError(
            f"{trajectories.shape[0]} trajectories cannot support a minimum "
            f"cluster size of {config.min_cluster_size}"
        )

    z_traj, traj_centre, traj_scale = standardize(trajectories)
    _, feat_centre, feat_scale = standardize(np.asarray(features, dtype=float))

    labels = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method=config.cluster_selection_method,
        copy=True,
    ).fit_predict(z_traj)
    leaf_labels = sorted({int(value) for value in labels if value >= 0})
    if not leaf_labels:
        raise ModeDiscoveryError(
            "HDBSCAN found no density mode at this min_cluster_size; the "
            "trajectories are a single diffuse cloud"
        )

    modes: list[TrajectoryMode] = []
    assignments: list[str | None] = [None] * len(labels)
    leaf_medoid_z: list[np.ndarray] = []
    leaf_members: list[np.ndarray] = []
    for order, label in enumerate(leaf_labels):
        rows = np.flatnonzero(labels == label)
        member_z = z_traj[rows]
        local = _medoid_of(member_z)
        mode_id = f"mode_{order:02d}"
        medoid_raw = trajectories[rows[local]]
        modes.append(
            TrajectoryMode(
                mode_id=mode_id,
                medoid=tuple(float(v) for v in medoid_raw),
                dispersion=tuple(
                    float(v) for v in _dispersion(trajectories[rows], medoid_raw)
                ),
                support=int(rows.size),
            )
        )
        leaf_medoid_z.append(member_z[local])
        leaf_members.append(rows)
        for row in rows:
            assignments[int(row)] = mode_id

    if config.build_hierarchy and len(modes) >= 2:
        modes = _agglomerate(modes, leaf_medoid_z, leaf_members, trajectories, z_traj)

    payload = {
        "library_id": library_id,
        "schema_version": FORECAST_SCHEMA_VERSION,
        "components": list(TRAJECTORY_COMPONENTS),
        "modes": [
            {
                "mode_id": mode.mode_id,
                "medoid": [round(v, 10) for v in mode.medoid],
                "dispersion": [round(v, 10) for v in mode.dispersion],
                "support": mode.support,
                "parent_mode_id": mode.parent_mode_id,
                "child_mode_ids": list(mode.child_mode_ids),
            }
            for mode in modes
        ],
    }
    library = ModeLibrary(
        library_id=library_id,
        fingerprint=_library_fingerprint(payload),
        fitted_at=fitted_at,
        algorithm=(
            f"hdbscan(min_cluster_size={config.min_cluster_size},"
            f"selection={config.cluster_selection_method})+kmedoids+ward_hierarchy"
        ),
        modes=tuple(modes),
        feature_names=TRAJECTORY_COMPONENTS,
        observation_count=int(trajectories.shape[0]),
        noise_count=int((labels < 0).sum()),
    )
    return DiscoveryResult(
        library=library,
        assignments=tuple(assignments),
        feature_centre=feat_centre,
        feature_scale=feat_scale,
        trajectory_centre=traj_centre,
        trajectory_scale=traj_scale,
    )


def _agglomerate(
    leaves: Sequence[TrajectoryMode],
    leaf_medoid_z: Sequence[np.ndarray],
    leaf_members: Sequence[np.ndarray],
    trajectories: np.ndarray,
    z_traj: np.ndarray,
) -> list[TrajectoryMode]:
    """Grow a binary parent hierarchy over the leaf modes by Ward linkage.

    A parent's medoid is recomputed over the union of its descendants' members,
    so it is a real trajectory too, not an average of two medoids.
    """

    from scipy.cluster.hierarchy import linkage

    stacked = np.vstack(list(leaf_medoid_z))
    tree = linkage(stacked, method="ward")

    modes: dict[str, TrajectoryMode] = {mode.mode_id: mode for mode in leaves}
    order: list[str] = [mode.mode_id for mode in leaves]
    node_members: dict[int, np.ndarray] = {
        index: np.asarray(rows) for index, rows in enumerate(leaf_members)
    }
    node_id: dict[int, str] = {index: mode.mode_id for index, mode in enumerate(leaves)}
    leaf_count = len(leaves)

    for step, row in enumerate(tree):
        left, right = int(row[0]), int(row[1])
        members = np.concatenate([node_members[left], node_members[right]])
        member_z = z_traj[members]
        local = _medoid_of(member_z)
        medoid_raw = trajectories[members[local]]
        parent_key = f"group_{step:02d}"
        parent = TrajectoryMode(
            mode_id=parent_key,
            medoid=tuple(float(v) for v in medoid_raw),
            dispersion=tuple(
                float(v) for v in _dispersion(trajectories[members], medoid_raw)
            ),
            support=int(members.size),
            child_mode_ids=(node_id[left], node_id[right]),
        )
        modes[parent_key] = parent
        order.append(parent_key)
        for child_key in (node_id[left], node_id[right]):
            child = modes[child_key]
            modes[child_key] = TrajectoryMode(
                mode_id=child.mode_id,
                medoid=child.medoid,
                dispersion=child.dispersion,
                support=child.support,
                parent_mode_id=parent_key,
                child_mode_ids=child.child_mode_ids,
            )
        node_members[leaf_count + step] = members
        node_id[leaf_count + step] = parent_key

    return [modes[key] for key in order]


def library_payload(result: DiscoveryResult) -> dict[str, Any]:
    """Serialize a fitted library and its retrieval space to a JSON artifact."""

    library = result.library
    return {
        "library_id": library.library_id,
        "fingerprint": library.fingerprint,
        "fitted_at": library.fitted_at.isoformat(),
        "algorithm": library.algorithm,
        "schema_version": library.schema_version,
        "authority": library.authority,
        "protocol_status": library.protocol_status,
        "action_authority_ready": False,
        "observation_count": library.observation_count,
        "noise_count": library.noise_count,
        "components": list(TRAJECTORY_COMPONENTS),
        "modes": [
            {
                "mode_id": mode.mode_id,
                "medoid": list(mode.medoid),
                "dispersion": list(mode.dispersion),
                "support": mode.support,
                "parent_mode_id": mode.parent_mode_id,
                "child_mode_ids": list(mode.child_mode_ids),
            }
            for mode in library.modes
        ],
        "feature_centre": [float(v) for v in result.feature_centre],
        "feature_scale": [float(v) for v in result.feature_scale],
        "assignments": list(result.assignments),
    }


def load_library_payload(payload: Mapping[str, Any]) -> tuple[ModeLibrary, tuple[str | None, ...], np.ndarray, np.ndarray]:
    """Rebuild a library, its assignments and its retrieval space from JSON."""

    modes = tuple(
        TrajectoryMode(
            mode_id=item["mode_id"],
            medoid=tuple(float(v) for v in item["medoid"]),
            dispersion=tuple(float(v) for v in item["dispersion"]),
            support=int(item["support"]),
            parent_mode_id=item.get("parent_mode_id"),
            child_mode_ids=tuple(item.get("child_mode_ids", ())),
        )
        for item in payload["modes"]
    )
    library = ModeLibrary(
        library_id=payload["library_id"],
        fingerprint=payload["fingerprint"],
        fitted_at=pd.Timestamp(payload["fitted_at"]),
        algorithm=payload["algorithm"],
        modes=modes,
        feature_names=tuple(payload["components"]),
        observation_count=int(payload["observation_count"]),
        noise_count=int(payload["noise_count"]),
    )
    assignments = tuple(payload["assignments"])
    return (
        library,
        assignments,
        np.asarray(payload["feature_centre"], dtype=float),
        np.asarray(payload["feature_scale"], dtype=float),
    )


def compare_algorithms(
    trajectories: np.ndarray,
    *,
    decimation: int = 60,
    cluster_counts: Sequence[int] = (2, 3, 4, 5, 6, 8, 10),
    min_cluster_sizes: Sequence[int] = (15, 25, 50, 100, 200),
) -> pd.DataFrame:
    """Score HDBSCAN against K-Means, a Gaussian mixture and Ward.

    ``decimation`` keeps one point per that many minutes.  Consecutive
    observation points share almost their whole future window, so metrics on
    the full sample flatter every algorithm equally and prove nothing; the
    decimated sample has non-overlapping futures and is the honest comparison.

    ``eta2_r60`` is the share of variance in the realized sixty-minute return
    that the partition explains — the only column here that speaks to whether a
    clustering is *useful* rather than merely tidy.
    """

    _require_sklearn()
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.mixture import GaussianMixture

    trajectories = np.asarray(trajectories, dtype=float)
    z, _, _ = standardize(trajectories)
    r60 = trajectories[:, TRAJECTORY_COMPONENTS.index("r_60")]
    keep = np.arange(0, z.shape[0], max(1, int(decimation)))
    samples = {"overlapping": (z, r60), "disjoint": (z[keep], r60[keep])}

    def eta_squared(outcome: np.ndarray, labels: np.ndarray) -> float:
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

    rows: list[dict[str, Any]] = []
    for sample, (matrix, outcome) in samples.items():
        candidates: list[tuple[str, str, np.ndarray]] = []
        for k in cluster_counts:
            if k >= matrix.shape[0]:
                continue
            candidates.append(
                ("kmeans", f"k={k}", KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix))
            )
            candidates.append(
                (
                    "gmm",
                    f"k={k}",
                    GaussianMixture(
                        n_components=k, covariance_type="full", n_init=3, random_state=0
                    ).fit_predict(matrix),
                )
            )
            candidates.append(
                ("ward", f"k={k}", AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matrix))
            )
        for size in min_cluster_sizes:
            if size >= matrix.shape[0] // 2:
                continue
            candidates.append(
                (
                    "hdbscan",
                    f"min_cluster_size={size}",
                    HDBSCAN(min_cluster_size=size, cluster_selection_method="eom", copy=True).fit_predict(matrix),
                )
            )
        for algorithm, parameter, labels in candidates:
            mask = labels >= 0
            found = len(set(labels[mask].tolist()))
            row = {
                "sample": sample,
                "algorithm": algorithm,
                "parameter": parameter,
                "clusters": found,
                "noise_fraction": float(1.0 - mask.mean()),
                "silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"),
                "eta2_r60": eta_squared(outcome, labels),
                "largest_cluster_share": float("nan"),
            }
            if found >= 2 and int(mask.sum()) > 10:
                row["silhouette"] = float(silhouette_score(matrix[mask], labels[mask]))
                row["davies_bouldin"] = float(davies_bouldin_score(matrix[mask], labels[mask]))
                row["calinski_harabasz"] = float(
                    calinski_harabasz_score(matrix[mask], labels[mask])
                )
                sizes = np.array(
                    [int((labels[mask] == g).sum()) for g in sorted(set(labels[mask].tolist()))],
                    dtype=float,
                )
                row["largest_cluster_share"] = float(sizes.max() / sizes.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def block_stability(
    trajectories: np.ndarray,
    *,
    algorithm: str,
    block: int = 20,
    repeats: int = 8,
    keep_fraction: float = 0.7,
    **parameters: Any,
) -> float:
    """Mean adjusted Rand index between a full fit and contiguous-block refits.

    Blocks, not i.i.d. resampling: neighbouring rows are near-duplicates, so an
    i.i.d. bootstrap would report a stability the data does not have.
    """

    _require_sklearn()
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    def fit(matrix: np.ndarray) -> np.ndarray:
        if algorithm == "kmeans":
            return KMeans(n_init=10, random_state=0, **parameters).fit_predict(matrix)
        if algorithm == "gmm":
            return GaussianMixture(
                covariance_type="full", n_init=3, random_state=0, **parameters
            ).fit_predict(matrix)
        if algorithm == "ward":
            return AgglomerativeClustering(linkage="ward", **parameters).fit_predict(matrix)
        if algorithm == "hdbscan":
            return HDBSCAN(cluster_selection_method="eom", copy=True, **parameters).fit_predict(matrix)
        raise ModeDiscoveryError(f"unknown algorithm {algorithm!r}")

    z, _, _ = standardize(np.asarray(trajectories, dtype=float))
    reference = fit(z)
    blocks = max(1, z.shape[0] // block)
    scores: list[float] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(1000 + repeat)
        mask = np.zeros(z.shape[0], dtype=bool)
        chosen = rng.choice(blocks, size=max(1, int(blocks * keep_fraction)), replace=False)
        for index in chosen:
            mask[index * block : (index + 1) * block] = True
        rows = np.flatnonzero(mask)
        if rows.size < 20:
            continue
        refit = fit(z[rows])
        both = (reference[rows] >= 0) & (refit >= 0)
        if int(both.sum()) > 10:
            scores.append(float(adjusted_rand_score(reference[rows][both], refit[both])))
    return float(np.mean(scores)) if scores else float("nan")


__all__ = [
    "MEDOID_SEARCH_LIMIT",
    "MINIMUM_DISPERSION",
    "DiscoveryConfig",
    "DiscoveryResult",
    "ModeDiscoveryError",
    "block_stability",
    "compare_algorithms",
    "discover_modes",
    "kmedoids",
    "library_payload",
    "load_library_payload",
    "standardize",
]
