"""Fitting the one globally learned object the runtime still needs.

The Brain no longer carries a library of modes. What it carries is a retrieval
index: the standardized contexts it can search, the realized futures those
contexts led to, and the principal basis those futures are compared in.

The basis is fitted on the whole dataset because the runtime must be able to
compare a node extracted at 09:31 with one extracted at 09:32. A locally refitted
basis would rotate between clocks and make association meaningless.

The basis is fitted on **detrended shapes**, not raw curves. Fitting raw curves
put 80.7% of the variance on one component that was essentially "where did it
end", so distance collapsed onto direction and the shape of the path — the part
that distinguishes "fell, came back, rallied" from "rallied straight" — was
crushed to a rounding error. Direction is now its own explicit channel, and the
basis carries only what is left after the endpoint trend is removed.

This is a research surface. It reads the future by construction and its output is
``shadow_only``; it grants no research, empirical or trading authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from brain.core.hypothesis_proposer import FEATURE_DIM, ForecastIndex
from brain.core.trajectory import (
    attributes_from_future,
    curve_matrix,
    direction_vector,
    shape_matrix,
)
from contract.brain.forecast import (
    DIRECTION_DIM,
    REPRESENTATION_DIM,
    SHAPE_COMPONENT_COUNT,
    TRAJECTORY_CURVE_LENGTH,
    PathAttributes,
)


class ForecastIndexError(RuntimeError):
    """Index construction refuses to publish something it cannot stand behind."""


ATTRIBUTE_NAMES: tuple[str, ...] = tuple(
    PathAttributes(
        r_5=0.0, r_15=0.0, r_30=0.0, r_60=0.0,
        mfe_0_15=0.0, mfe_15_30=0.0, mfe_30_60=0.0,
        mae_0_15=0.0, mae_15_30=0.0, mae_30_60=0.0,
        time_to_mfe=0.0, time_to_mae=0.0, path_efficiency=0.0,
        rv_30=0.0, rv_60=0.0,
    ).as_mapping()
)


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns, returning the transformed matrix, centre and scale.

    NaN is imputed to the column centre, so a component the Eye never published
    over this window contributes nothing to a distance instead of poisoning it.
    A column that is entirely NaN is legitimate and must not warn.
    """

    data = np.asarray(matrix, dtype=float)
    valid = np.isfinite(data)
    counts = valid.sum(axis=0)
    filled = np.where(valid, data, 0.0)
    centre = np.divide(
        filled.sum(axis=0), counts, out=np.zeros(data.shape[1]), where=counts > 0
    )
    variance = np.divide(
        (np.where(valid, data - centre, 0.0) ** 2).sum(axis=0),
        counts,
        out=np.zeros(data.shape[1]),
        where=counts > 0,
    )
    scale = np.sqrt(variance)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    z = (data - centre) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), centre, scale


@dataclass(frozen=True)
class PrincipalBasis:
    """The globally fitted curve basis, plus how much shape it retains."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def retained(self) -> float:
        return float(self.explained_variance_ratio.sum())


def fit_principal_basis(
    shapes: np.ndarray, *, components: int = SHAPE_COMPONENT_COUNT
) -> PrincipalBasis:
    """Fit the principal basis of the detrended, unit-RMS shapes."""

    from sklearn.decomposition import PCA

    data = np.asarray(shapes, dtype=float)
    if data.ndim != 2 or data.shape[1] != TRAJECTORY_CURVE_LENGTH:
        raise ForecastIndexError(
            f"curves must be (n, {TRAJECTORY_CURVE_LENGTH}), got {data.shape}"
        )
    if data.shape[0] <= components:
        raise ForecastIndexError(
            f"{data.shape[0]} curves cannot support {components} components"
        )
    model = PCA(n_components=components, svd_solver="full", random_state=0)
    model.fit(data)
    return PrincipalBasis(
        mean=np.asarray(model.mean_, dtype=float),
        components=np.asarray(model.components_, dtype=float),
        explained_variance_ratio=np.asarray(
            model.explained_variance_ratio_, dtype=float
        ),
    )


def _typical_distance(matrix: np.ndarray, *, sample: int = 512) -> float:
    """The median distance between two unrelated rows.

    This is the yardstick retrieval confidence is read against: a neighbourhood
    whose members sit as far away as two contexts picked at random is not
    precedent. Sampled with a fixed seed so an index is reproducible.
    """

    data = np.asarray(matrix, dtype=float)
    if data.shape[0] < 2:
        raise ForecastIndexError("a context scale needs at least two rows")
    take = min(sample, data.shape[0])
    rows = np.random.default_rng(0).choice(data.shape[0], size=take, replace=False)
    subset = data[rows]
    distances = np.linalg.norm(subset[:, None, :] - subset[None, :, :], axis=2)
    upper = distances[np.triu_indices(take, k=1)]
    value = float(np.median(upper))
    if not np.isfinite(value) or value <= 0.0:
        raise ForecastIndexError("contexts in this window are indistinguishable")
    return value


def build_index(
    *,
    features: np.ndarray,
    anchor_prices: np.ndarray,
    anchor_atrs: np.ndarray,
    future_closes: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    direction_weight: float = 0.5,
    index_id: str = "local_conditional_index_v3",
) -> tuple[ForecastIndex, PrincipalBasis]:
    """Assemble everything the runtime proposer needs from one window.

    ``direction_weight`` splits the representation's total variance between the
    two channels. At the default of one half they contribute equally, so neither
    a thirteen-dimension Direction channel nor a five-dimension Shape channel
    dominates distance purely by having more axes.
    """

    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ForecastIndexError(
            f"features must be (n, {FEATURE_DIM}), got {features.shape}"
        )
    if not 0.0 < direction_weight < 1.0:
        raise ForecastIndexError("direction_weight must lie in (0, 1)")

    curves = curve_matrix(
        anchor_prices=anchor_prices,
        anchor_atrs=anchor_atrs,
        future_closes=future_closes,
    )
    attribute_rows = [
        attributes_from_future(
            anchor_price=float(anchor_prices[row]),
            anchor_atr=float(anchor_atrs[row]),
            closes=future_closes[row],
            highs=future_highs[row],
            lows=future_lows[row],
        )
        for row in range(curves.shape[0])
    ]
    attributes = np.array(
        [list(item.as_mapping().values()) for item in attribute_rows], dtype=float
    )
    direction_raw = np.array(
        [direction_vector(item) for item in attribute_rows], dtype=float
    )

    basis = fit_principal_basis(shape_matrix(curves))
    shape_scores = (shape_matrix(curves) - basis.mean) @ basis.components.T

    # Standardize each channel on its own, then scale so the two contribute the
    # intended share of total variance regardless of how many axes each has.
    direction_z, direction_centre, direction_spread = standardize(direction_raw)
    shape_z, shape_centre, shape_spread = standardize(shape_scores)
    direction_gain = math.sqrt(direction_weight / max(1, DIRECTION_DIM))
    shape_gain = math.sqrt((1.0 - direction_weight) / max(1, SHAPE_COMPONENT_COUNT))
    scores = np.hstack([direction_z * direction_gain, shape_z * shape_gain])
    if scores.shape[1] != REPRESENTATION_DIM:
        raise ForecastIndexError(
            f"representation must be (n, {REPRESENTATION_DIM}), got {scores.shape}"
        )

    context_z, centre, scale = standardize(features)
    context_scale = _typical_distance(context_z)

    payload = {
        "index_id": index_id,
        "rows": int(curves.shape[0]),
        "direction_weight": round(float(direction_weight), 10),
        "principal_mean": [round(float(v), 10) for v in basis.mean],
        "principal_components": [
            [round(float(v), 10) for v in row] for row in basis.components
        ],
        "direction_centre": [round(float(v), 10) for v in direction_centre],
        "direction_spread": [round(float(v), 10) for v in direction_spread],
        "shape_centre": [round(float(v), 10) for v in shape_centre],
        "shape_spread": [round(float(v), 10) for v in shape_spread],
        "feature_center": [round(float(v), 10) for v in centre],
        "feature_scale": [round(float(v), 10) for v in scale],
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    index = ForecastIndex(
        fingerprint=fingerprint,
        feature_center=centre,
        feature_scale=scale,
        reference_features=features,
        reference_curves=curves,
        reference_scores=scores,
        reference_attributes=attributes,
        attribute_names=ATTRIBUTE_NAMES,
        principal_mean=basis.mean,
        principal_components=basis.components,
        direction_centre=direction_centre,
        direction_spread=direction_spread,
        shape_centre=shape_centre,
        shape_spread=shape_spread,
        direction_weight=float(direction_weight),
        # A *distance* scale, not a per-axis spread: gates and ambiguity are
        # measured as fractions of how far apart two unrelated futures typically
        # are. Comparing a distance against a per-axis standard deviation
        # saturates every ratio in eighteen dimensions and says nothing.
        component_scale=_typical_distance(scores),
        context_scale=context_scale,
    )
    return index, basis


def save_index(index: ForecastIndex, basis: PrincipalBasis, path: Path) -> None:
    """Persist an index as one compressed archive plus a readable manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fingerprint=np.array([index.fingerprint]),
        feature_center=index.feature_center,
        feature_scale=index.feature_scale,
        reference_features=index.reference_features,
        reference_curves=index.reference_curves,
        reference_scores=index.reference_scores,
        reference_attributes=index.reference_attributes,
        attribute_names=np.array(index.attribute_names),
        principal_mean=index.principal_mean,
        principal_components=index.principal_components,
        direction_centre=index.direction_centre,
        direction_spread=index.direction_spread,
        shape_centre=index.shape_centre,
        shape_spread=index.shape_spread,
        direction_weight=np.array([index.direction_weight]),
        component_scale=np.array([index.component_scale]),
        context_scale=np.array([index.context_scale]),
        explained_variance_ratio=basis.explained_variance_ratio,
    )
    manifest = {
        "fingerprint": index.fingerprint,
        "rows": len(index),
        "curve_length_minutes": TRAJECTORY_CURVE_LENGTH,
        "shape_components": SHAPE_COMPONENT_COUNT,
        "direction_dim": DIRECTION_DIM,
        "representation_dim": REPRESENTATION_DIM,
        "direction_weight": index.direction_weight,
        "explained_variance_ratio": [
            round(float(v), 6) for v in basis.explained_variance_ratio
        ],
        "retained_variance": round(basis.retained, 6),
        "component_scale": round(index.component_scale, 6),
        "context_scale": round(index.context_scale, 6),
        "attribute_names": list(index.attribute_names),
        "authority": "shadow_only",
        "protocol_status": "development_unvalidated",
        "action_authority_ready": False,
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_index(path: Path) -> ForecastIndex:
    """Rebuild a persisted index."""

    stored = np.load(path, allow_pickle=False)
    return ForecastIndex(
        fingerprint=str(stored["fingerprint"][0]),
        feature_center=stored["feature_center"],
        feature_scale=stored["feature_scale"],
        reference_features=stored["reference_features"],
        reference_curves=stored["reference_curves"],
        reference_scores=stored["reference_scores"],
        reference_attributes=stored["reference_attributes"],
        attribute_names=tuple(str(v) for v in stored["attribute_names"]),
        principal_mean=stored["principal_mean"],
        principal_components=stored["principal_components"],
        direction_centre=stored["direction_centre"],
        direction_spread=stored["direction_spread"],
        shape_centre=stored["shape_centre"],
        shape_spread=stored["shape_spread"],
        direction_weight=float(stored["direction_weight"][0]),
        component_scale=float(stored["component_scale"][0]),
        context_scale=float(stored["context_scale"][0]),
    )


def index_manifest(path: Path) -> Mapping[str, Any]:
    return json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))


__all__ = [
    "ATTRIBUTE_NAMES",
    "ForecastIndexError",
    "PrincipalBasis",
    "build_index",
    "fit_principal_basis",
    "index_manifest",
    "load_index",
    "save_index",
    "standardize",
]
