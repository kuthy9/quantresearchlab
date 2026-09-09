"""What the market has historically done next, from where it stands now.

The proposer answers one question each minute: *given a context like this one,
what did the next sixty minutes actually do?*  It reads the current Eye state
into a context vector ``X_t``, retrieves the nearest historical contexts, reads
their realized futures as a **conditional future cloud**, and clusters that
cloud locally to extract at most three representative trajectory nodes.

Nothing is fitted globally except the retrieval space and the principal basis.
There is no library of modes: the representatives are re-extracted every clock
from whichever futures the current context actually retrieves. Persistence
across clocks is the pool's job, by association, not this module's.

This module also owns the definition of ``X_t``. The offline builder in
``brain/research/`` imports it from here rather than the other way round, so the
research package never becomes a runtime dependency.

Nothing here reads the future *of the current clock*. ``X_t`` is built only from
facts the Eye has already published at ``t``; the futures in the cloud belong to
historical neighbours whose sixty minutes are long since complete.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from contract.brain.forecast import (
    MAX_CLOUD_NODES,
    PRINCIPAL_COMPONENT_COUNT,
    TRAJECTORY_CURVE_LENGTH,
    ConditionalCloud,
    PathAttributes,
    TrajectoryNode,
    node_identity,
)
from contract.market import Direction, Timeframe

# The scales the context vector reads, coarsest first.  This mirrors the
# registered scale registry; a timeframe missing from a snapshot is a fault,
# not a defaulted zero.
CONTEXT_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)

# The Eye's registered delivery phases, one-hot encoded.  These are the exact
# values of ``eyes.core.market_state.DeliveryPhase``; an unrecognized phase
# lands in "other" rather than silently colliding with a known one, and
# ``test_hypothesis_forecast`` pins this tuple against the enum so a new phase
# cannot appear without this vector noticing.
DELIVERY_PHASES: tuple[str, ...] = (
    "balance",
    "expansion",
    "retracement",
    "reversal_attempt",
    "transition",
    "other",
)

# The registered exchange-session phases, from
# ``eyes.core.market_state.session_name_phase``.  The session name is a coarser
# label derived from the same clock, so encoding the phase alone loses nothing.
SESSION_PHASES: tuple[str, ...] = (
    "overnight_delivery",
    "pre_open",
    "opening_expansion",
    "morning_delivery",
    "midday_balance",
    "afternoon_delivery",
    "closing_rotation",
    "other",
)

_PER_TIMEFRAME_FEATURES: tuple[str, ...] = (
    "ext_dir",
    "int_dir",
    "prot_intact",
    "dist_prot_high_atr",
    "dist_prot_low_atr",
    "last_bos_dir",
    "last_mss_dir",
    *(f"phase_{name}" for name in DELIVERY_PHASES),
    "leg_dir",
    "displacement_score",
    "range_location",
    "range_width_atr",
    "unswept_bsl",
    "unswept_ssl",
    "dist_bsl_atr",
    "dist_ssl_atr",
    "active_fvg",
    "active_ob",
    "structural_legs",
)

_RELATION_FEATURES: tuple[str, ...] = (
    "rel_count",
    "rel_aligned",
    "rel_reversal_warning",
    "rel_mss_against_parent",
    "rel_child_location_mean",
    "rel_parent_invalidation_min_atr",
)

_SESSION_FEATURES: tuple[str, ...] = (
    "session_elapsed_minutes",
    "session_realized_volatility",
    "session_relative_volume",
    "session_range_location",
    "session_dist_prior_day_high_atr",
    "session_dist_prior_day_low_atr",
    "session_dist_overnight_high_atr",
    "session_dist_overnight_low_atr",
    *(f"session_phase_{name}" for name in SESSION_PHASES),
)

# Raw one-minute price context.  The Eye reports structure; these say how the
# tape got here, which is what makes two structurally identical contexts
# distinguishable.
_PRICE_FEATURES: tuple[str, ...] = (
    "past_r_5_atr",
    "past_r_15_atr",
    "past_r_30_atr",
    "past_r_60_atr",
    "past_rv_30_atr",
    "past_rv_60_atr",
    "bar_range_atr",
    "atr_ratio_m1_h1",
)

FEATURE_NAMES: tuple[str, ...] = (
    *(
        f"{timeframe.value}_{name}"
        for timeframe in CONTEXT_TIMEFRAMES
        for name in _PER_TIMEFRAME_FEATURES
    ),
    *_RELATION_FEATURES,
    *_SESSION_FEATURES,
    *_PRICE_FEATURES,
)
FEATURE_DIM = len(FEATURE_NAMES)

class HypothesisProposerError(RuntimeError):
    """The proposer refuses to guess when its inputs are not what it needs."""


def _direction(value: object) -> float:
    """Signed direction: the Eye names these LONG and SHORT, not bullish/bearish."""

    if value is Direction.LONG:
        return 1.0
    if value is Direction.SHORT:
        return -1.0
    return 0.0


def _tristate(value: object) -> float:
    if value is None:
        return 0.0
    return 1.0 if value else -1.0


def _ratio(numerator: float | None, scale: float | None) -> float:
    """A distance in ATR units, or NaN when either side is unavailable.

    NaN is deliberate: an absent protected swing is not a distance of zero, and
    the retrieval step imputes missing components explicitly.
    """

    if numerator is None or scale is None:
        return math.nan
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        return math.nan
    value = float(numerator) / scale
    return value if math.isfinite(value) else math.nan


def _price_context(closes: Sequence[float], highs_lows: tuple[float, float] | None,
                   atr: float) -> dict[str, float]:
    """Backward-looking tape features over the completed one-minute history."""

    out: dict[str, float] = {}
    if not math.isfinite(atr) or atr <= 0.0 or not closes:
        for name in _PRICE_FEATURES:
            out[name] = math.nan
        return out
    last = float(closes[-1])
    for lag in (5, 15, 30, 60):
        out[f"past_r_{lag}_atr"] = (
            (last - float(closes[-1 - lag])) / atr if len(closes) > lag else math.nan
        )
    diffs = np.diff(np.asarray(closes, dtype=float))
    for window in (30, 60):
        if diffs.size >= window:
            segment = diffs[-window:]
            out[f"past_rv_{window}_atr"] = float(
                math.sqrt(float(segment @ segment)) / atr
            )
        else:
            out[f"past_rv_{window}_atr"] = math.nan
    if highs_lows is None:
        out["bar_range_atr"] = math.nan
    else:
        high, low = highs_lows
        out["bar_range_atr"] = (float(high) - float(low)) / atr
    out["atr_ratio_m1_h1"] = math.nan
    return out


def observation_features(
    snapshot: Any,
    *,
    closes: Sequence[float] = (),
    bar_high_low: tuple[float, float] | None = None,
) -> tuple[float, ...]:
    """Read one ``MarketSnapshot`` into the fixed-width context vector ``X_t``.

    ``closes`` is the completed one-minute close history ending at ``t``; it
    supplies the raw tape context the Eye does not publish.  Components the
    snapshot cannot support come back as NaN so the retrieval step can impute
    them deliberately rather than pretending a zero was observed.
    """

    states = getattr(snapshot, "timeframe_states", None)
    if not states:
        raise HypothesisProposerError("snapshot carries no timeframe states")
    price = float(snapshot.price)
    values: dict[str, float] = {}

    m1_atr = math.nan
    h1_atr = math.nan
    for timeframe in CONTEXT_TIMEFRAMES:
        state = states.get(timeframe)
        if state is None:
            raise HypothesisProposerError(
                f"snapshot is missing the registered {timeframe.value} scale"
            )
        key = timeframe.value
        structure = state.structure
        delivery = state.delivery
        band = state.range
        liquidity = state.liquidity
        zones = state.zones
        atr = state.quality.atr
        atr = float(atr) if atr is not None else math.nan
        if timeframe is Timeframe.M1:
            m1_atr = atr
        elif timeframe is Timeframe.H1:
            h1_atr = atr

        values[f"{key}_ext_dir"] = _direction(structure.external_direction)
        values[f"{key}_int_dir"] = _direction(structure.internal_direction)
        values[f"{key}_prot_intact"] = _tristate(structure.protected_swing_intact)
        values[f"{key}_dist_prot_high_atr"] = _ratio(
            None if structure.protected_high is None else structure.protected_high - price,
            atr,
        )
        values[f"{key}_dist_prot_low_atr"] = _ratio(
            None if structure.protected_low is None else price - structure.protected_low,
            atr,
        )
        values[f"{key}_last_bos_dir"] = _direction(structure.last_bos_direction)
        values[f"{key}_last_mss_dir"] = _direction(structure.last_mss_direction)

        phase = getattr(delivery.phase, "value", None)
        matched = phase if phase in DELIVERY_PHASES else "other"
        for name in DELIVERY_PHASES:
            values[f"{key}_phase_{name}"] = 1.0 if name == matched else 0.0
        values[f"{key}_leg_dir"] = _direction(delivery.active_leg_direction)
        values[f"{key}_displacement_score"] = (
            math.nan
            if delivery.displacement_score is None
            else float(delivery.displacement_score)
        )

        values[f"{key}_range_location"] = (
            math.nan if band.normalized_location is None else float(band.normalized_location)
        )
        values[f"{key}_range_width_atr"] = _ratio(
            None if (band.high is None or band.low is None) else band.high - band.low,
            atr,
        )

        above = [level for level in liquidity.unswept_bsl if level > price]
        below = [level for level in liquidity.unswept_ssl if level < price]
        values[f"{key}_unswept_bsl"] = float(len(liquidity.unswept_bsl))
        values[f"{key}_unswept_ssl"] = float(len(liquidity.unswept_ssl))
        values[f"{key}_dist_bsl_atr"] = _ratio(
            min(above) - price if above else None, atr
        )
        values[f"{key}_dist_ssl_atr"] = _ratio(
            price - max(below) if below else None, atr
        )
        values[f"{key}_active_fvg"] = float(len(zones.active_fvg))
        values[f"{key}_active_ob"] = float(len(zones.active_ob))
        values[f"{key}_structural_legs"] = float(len(state.structural_legs))

    relations = getattr(snapshot, "relations", {}) or {}
    aligned = warning = mss_against = 0
    locations: list[float] = []
    invalidations: list[float] = []
    for relation in relations.values():
        if (
            relation.parent_direction is not None
            and relation.parent_direction is relation.child_direction
        ):
            aligned += 1
        warning += int(bool(relation.reversal_warning))
        mss_against += int(bool(relation.child_mss_against_parent))
        if relation.child_location_in_parent_range is not None:
            locations.append(float(relation.child_location_in_parent_range))
        if relation.parent_invalidation_distance_atr is not None:
            invalidations.append(float(relation.parent_invalidation_distance_atr))
    values["rel_count"] = float(len(relations))
    values["rel_aligned"] = float(aligned)
    values["rel_reversal_warning"] = float(warning)
    values["rel_mss_against_parent"] = float(mss_against)
    values["rel_child_location_mean"] = (
        float(np.mean(locations)) if locations else math.nan
    )
    values["rel_parent_invalidation_min_atr"] = (
        float(np.min(invalidations)) if invalidations else math.nan
    )

    session = snapshot.session
    values["session_elapsed_minutes"] = float(session.elapsed_minutes or 0)
    values["session_realized_volatility"] = (
        math.nan
        if session.realized_volatility is None
        else float(session.realized_volatility)
    )
    values["session_relative_volume"] = (
        math.nan if session.relative_volume is None else float(session.relative_volume)
    )
    if session.session_high is not None and session.session_low is not None:
        width = float(session.session_high) - float(session.session_low)
        values["session_range_location"] = (
            (price - float(session.session_low)) / width if width > 0 else math.nan
        )
    else:
        values["session_range_location"] = math.nan
    values["session_dist_prior_day_high_atr"] = _ratio(
        None if session.prior_day_high is None else session.prior_day_high - price, m1_atr
    )
    values["session_dist_prior_day_low_atr"] = _ratio(
        None if session.prior_day_low is None else price - session.prior_day_low, m1_atr
    )
    values["session_dist_overnight_high_atr"] = _ratio(
        None if session.overnight_high is None else session.overnight_high - price, m1_atr
    )
    values["session_dist_overnight_low_atr"] = _ratio(
        None if session.overnight_low is None else price - session.overnight_low, m1_atr
    )
    phase_name = str(session.phase or "")
    matched_phase = phase_name if phase_name in SESSION_PHASES else "other"
    for name in SESSION_PHASES:
        values[f"session_phase_{name}"] = 1.0 if name == matched_phase else 0.0

    values.update(_price_context(closes, bar_high_low, m1_atr))
    values["atr_ratio_m1_h1"] = (
        m1_atr / h1_atr
        if math.isfinite(m1_atr) and math.isfinite(h1_atr) and h1_atr > 0
        else math.nan
    )

    missing = [name for name in FEATURE_NAMES if name not in values]
    if missing:
        raise HypothesisProposerError(f"context vector is incomplete: {missing[:5]}")
    return tuple(values[name] for name in FEATURE_NAMES)




# A node's per-point dispersion is floored so a tight cluster cannot claim
# impossible precision and reject every real path as a falsification.
MINIMUM_DISPERSION = 0.05


@dataclass(frozen=True)
class ProposerConfig:
    """How far the retrieval reaches, and how selective the extraction is."""

    neighbours: int = 200
    minimum_neighbours: int = 25
    cluster_count: int = 6
    max_nodes: int = 4
    minimum_mass: float = 0.12
    kmeans_restarts: int = 5

    def __post_init__(self) -> None:
        if self.neighbours < 1:
            raise ValueError("neighbours must be positive")
        if not 1 <= self.minimum_neighbours <= self.neighbours:
            raise ValueError("minimum_neighbours must lie in [1, neighbours]")
        if self.cluster_count < 2:
            raise ValueError("cluster_count must be at least 2")
        if not 1 <= self.max_nodes <= MAX_CLOUD_NODES:
            raise ValueError(f"max_nodes must lie in [1, {MAX_CLOUD_NODES}]")
        if not 0.0 < self.minimum_mass < 1.0:
            raise ValueError("minimum_mass must lie in (0, 1)")
        if self.kmeans_restarts < 1:
            raise ValueError("kmeans_restarts must be positive")

    @classmethod
    def from_protocol(cls, payload: Mapping[str, Any]) -> "ProposerConfig":
        section = payload.get("proposer", {})
        return cls(
            neighbours=int(section["neighbours"]),
            minimum_neighbours=int(section["minimum_neighbours"]),
            cluster_count=int(section["cluster_count"]),
            max_nodes=int(section["max_nodes"]),
            minimum_mass=float(section["minimum_mass"]),
            kmeans_restarts=int(section["kmeans_restarts"]),
        )


def load_hypothesis_protocol(path: str | Path) -> Mapping[str, Any]:
    """Read the hypothesis protocol and refuse anything claiming authority."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("authority") != "shadow_only":
        raise HypothesisProposerError("the hypothesis protocol must be shadow-only")
    if payload.get("action_authority_ready", False):
        raise HypothesisProposerError(
            "the hypothesis protocol may not declare action authority"
        )
    if payload.get("protocol_status") != "development_unvalidated":
        raise HypothesisProposerError(
            "the hypothesis protocol is development-unvalidated"
        )
    return payload


@dataclass(frozen=True)
class ForecastIndex:
    """Everything the runtime needs to answer "what followed contexts like this".

    This is the only globally fitted object left. It carries the retrieval space
    (standardized contexts), the realized futures those contexts led to, and the
    principal basis the futures are compared in. It holds no modes: what counts
    as a representative future is decided per clock, from the neighbourhood.
    """

    fingerprint: str
    feature_center: np.ndarray
    feature_scale: np.ndarray
    reference_features: np.ndarray
    reference_curves: np.ndarray
    reference_scores: np.ndarray
    reference_attributes: np.ndarray
    attribute_names: tuple[str, ...]
    principal_mean: np.ndarray
    principal_components: np.ndarray
    component_scale: float

    def __post_init__(self) -> None:
        rows = self.reference_features.shape[0]
        checks = {
            "feature_center": (self.feature_center.shape, (FEATURE_DIM,)),
            "feature_scale": (self.feature_scale.shape, (FEATURE_DIM,)),
            "reference_features": (self.reference_features.shape, (rows, FEATURE_DIM)),
            "reference_curves": (
                self.reference_curves.shape,
                (rows, TRAJECTORY_CURVE_LENGTH),
            ),
            "reference_scores": (
                self.reference_scores.shape,
                (rows, PRINCIPAL_COMPONENT_COUNT),
            ),
            "principal_mean": (self.principal_mean.shape, (TRAJECTORY_CURVE_LENGTH,)),
            "principal_components": (
                self.principal_components.shape,
                (PRINCIPAL_COMPONENT_COUNT, TRAJECTORY_CURVE_LENGTH),
            ),
        }
        for name, (actual, expected) in checks.items():
            if actual != expected:
                raise HypothesisProposerError(
                    f"{name} must have shape {expected}, got {actual}"
                )
        if self.reference_attributes.shape != (rows, len(self.attribute_names)):
            raise HypothesisProposerError("reference attributes are misaligned")
        if np.any(self.feature_scale <= 0.0):
            raise HypothesisProposerError("feature scale must be positive")
        if not math.isfinite(self.component_scale) or self.component_scale <= 0.0:
            raise HypothesisProposerError("component_scale must be finite and positive")
        if not self.fingerprint:
            raise HypothesisProposerError("a forecast index must carry a fingerprint")

    def __len__(self) -> int:
        return int(self.reference_features.shape[0])

    def project(self, curves: np.ndarray) -> np.ndarray:
        """Project raw curves onto the principal basis."""

        centred = np.asarray(curves, dtype=float) - self.principal_mean
        return centred @ self.principal_components.T


class HypothesisProposer:
    """Extracts this clock's representative futures from its conditional cloud.

    Retrieval is exact k-nearest-neighbour in the standardized context space —
    deterministic, and cheap enough per minute that no approximation is
    warranted at this scale. Clustering is K-Means in the principal basis:
    conditional future clouds are continuous rather than island-shaped, so a
    density method abstains on them and a partitional cut is the honest tool.
    """

    def __init__(
        self,
        *,
        index: ForecastIndex,
        config: ProposerConfig | None = None,
    ) -> None:
        self.index = index
        self.config = config or ProposerConfig()
        self._reference = np.nan_to_num(
            (index.reference_features - index.feature_center) / index.feature_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _standardize(self, context: Sequence[float]) -> np.ndarray:
        z = (
            np.asarray(context, dtype=float).reshape(1, -1) - self.index.feature_center
        ) / self.index.feature_scale
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)[0]

    def neighbourhood(self, context: Sequence[float]) -> np.ndarray:
        """Row indices of the nearest historical contexts, nearest first."""

        if self._reference.shape[0] == 0:
            return np.empty(0, dtype=int)
        query = self._standardize(context)
        distances = np.linalg.norm(self._reference - query, axis=1)
        take = min(self.config.neighbours, distances.size)
        head = np.argpartition(distances, take - 1)[:take]
        return head[np.argsort(distances[head], kind="stable")]

    def propose(self, context: Sequence[float], *, asof) -> ConditionalCloud:
        """Retrieve, cluster locally, and surface the top nodes by mass.

        A neighbourhood too small to say anything yields a cloud with no nodes,
        whose residual is one — the Brain says nothing rather than extrapolating
        from a handful of points.
        """

        rows = self.neighbourhood(context)
        if rows.size < self.config.minimum_neighbours:
            return ConditionalCloud(
                asof=asof,
                neighbour_count=int(rows.size),
                assigned_count=0,
                cluster_count=0,
                nodes=(),
                component_scale=self.index.component_scale,
            )

        scores = self.index.reference_scores[rows]
        curves = self.index.reference_curves[rows]
        attributes = self.index.reference_attributes[rows]
        clusters = min(self.config.cluster_count, scores.shape[0])

        from sklearn.cluster import KMeans

        labels = KMeans(
            n_clusters=clusters,
            n_init=self.config.kmeans_restarts,
            random_state=0,
        ).fit_predict(scores)

        total = float(rows.size)
        candidates: list[tuple[float, int, TrajectoryNode]] = []
        assigned = 0
        for label in sorted(set(int(v) for v in labels)):
            members = np.flatnonzero(labels == label)
            mass = members.size / total
            if mass < self.config.minimum_mass:
                continue
            centre = scores[members].mean(axis=0)
            local = int(np.argmin(np.linalg.norm(scores[members] - centre, axis=1)))
            # The medoid is what we publish; the centroid is what we match on.
            representative = int(members[local])
            member_curves = curves[members]
            dispersion = np.maximum(
                member_curves.std(axis=0) if members.size > 1
                else np.full(TRAJECTORY_CURVE_LENGTH, MINIMUM_DISPERSION),
                MINIMUM_DISPERSION,
            )
            curve = tuple(float(v) for v in curves[representative])
            node = TrajectoryNode(
                node_id=node_identity(curve),
                curve=curve,
                components=tuple(float(v) for v in centre),
                dispersion=tuple(float(v) for v in dispersion),
                mass=mass,
                member_count=int(members.size),
                attributes=PathAttributes(
                    **dict(zip(self.index.attribute_names, attributes[representative]))
                ),
            )
            candidates.append((mass, -members.size, node))
            assigned += int(members.size)

        # Highest mass first; ties broken by member count then identity, so two
        # runs over the same cloud always surface the same nodes in the same
        # order.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2].node_id))
        kept = [node for _, _, node in candidates[: self.config.max_nodes]]
        return ConditionalCloud(
            asof=asof,
            neighbour_count=int(rows.size),
            assigned_count=sum(node.member_count for node in kept),
            cluster_count=clusters,
            nodes=tuple(kept),
            component_scale=self.index.component_scale,
        )


__all__ = [
    "CONTEXT_TIMEFRAMES",
    "DELIVERY_PHASES",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "MINIMUM_DISPERSION",
    "SESSION_PHASES",
    "ForecastIndex",
    "HypothesisProposer",
    "HypothesisProposerError",
    "ProposerConfig",
    "load_hypothesis_protocol",
    "observation_features",
]
