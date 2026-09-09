"""Fitting the one globally learned object the runtime still needs.

The Brain no longer carries a library of modes. What it carries is a retrieval
index: the standardized contexts it can search, the realized futures those
contexts led to, and the principal basis those futures are compared in.

The basis is fitted on the whole dataset because the runtime must be able to
compare a node extracted at 09:31 with one extracted at 09:32. A locally refitted
basis would rotate between clocks and make association meaningless.

This is a research surface. It reads the future by construction and its output is
``shadow_only``; it grants no research, empirical or trading authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from brain.core.hypothesis_proposer import FEATURE_DIM, ForecastIndex
from brain.core.trajectory import attributes_from_future, curve_matrix
from contract.brain.forecast import (
    PRINCIPAL_COMPONENT_COUNT,
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
    curves: np.ndarray, *, components: int = PRINCIPAL_COMPONENT_COUNT
) -> PrincipalBasis:
    """Fit the principal basis of the standardized trajectory curves."""

    from sklearn.decomposition import PCA

    data = np.asarray(curves, dtype=float)
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


def build_index(
    *,
    features: np.ndarray,
    anchor_prices: np.ndarray,
    anchor_atrs: np.ndarray,
    future_closes: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    index_id: str = "local_conditional_index_v2",
) -> tuple[ForecastIndex, PrincipalBasis]:
    """Assemble everything the runtime proposer needs from one window."""

    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ForecastIndexError(
            f"features must be (n, {FEATURE_DIM}), got {features.shape}"
        )
    curves = curve_matrix(
        anchor_prices=anchor_prices,
        anchor_atrs=anchor_atrs,
        future_closes=future_closes,
    )
    basis = fit_principal_basis(curves)
    scores = (curves - basis.mean) @ basis.components.T
    _, centre, scale = standardize(features)

    attributes = np.array(
        [
            [
                value
                for value in attributes_from_future(
                    anchor_price=float(anchor_prices[row]),
                    anchor_atr=float(anchor_atrs[row]),
                    closes=future_closes[row],
                    highs=future_highs[row],
                    lows=future_lows[row],
                )
                .as_mapping()
                .values()
            ]
            for row in range(curves.shape[0])
        ],
        dtype=float,
    )

    payload = {
        "index_id": index_id,
        "rows": int(curves.shape[0]),
        "principal_mean": [round(float(v), 10) for v in basis.mean],
        "principal_components": [
            [round(float(v), 10) for v in row] for row in basis.components
        ],
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
        # The spread of the basis itself, used to keep distribution ambiguity
        # comparable across windows fitted on different volatility regimes.
        component_scale=float(np.mean(scores.std(axis=0))),
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
        component_scale=np.array([index.component_scale]),
        explained_variance_ratio=basis.explained_variance_ratio,
    )
    manifest = {
        "fingerprint": index.fingerprint,
        "rows": len(index),
        "curve_length_minutes": TRAJECTORY_CURVE_LENGTH,
        "principal_components": PRINCIPAL_COMPONENT_COUNT,
        "explained_variance_ratio": [
            round(float(v), 6) for v in basis.explained_variance_ratio
        ],
        "retained_variance": round(basis.retained, 6),
        "component_scale": round(index.component_scale, 6),
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
        component_scale=float(stored["component_scale"][0]),
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
