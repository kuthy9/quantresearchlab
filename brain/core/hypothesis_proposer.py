"""What the market has historically done next, from where it stands now.

The proposer answers one question each minute: *given this context, which
trajectory modes actually followed in the past?*  It never invents a future.
It reads the current Eye state into a context vector ``X_t``, retrieves the
nearest historical contexts, and reports how often each mode followed them.

This module also owns the definition of ``X_t`` itself.  The offline dataset
builder in ``brain/research/`` imports it from here rather than the other way
round, so the research package never becomes a runtime dependency.

Nothing here reads the future.  ``X_t`` is built only from facts the Eye has
already published at ``t``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from contract.brain.forecast import HypothesisProposal, ModeLibrary
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


@dataclass(frozen=True)
class ProposerConfig:
    """How far the proposer looks back and how selective it is."""

    neighbours: int = 200
    minimum_neighbours: int = 25
    max_proposals: int = 6
    minimum_prior: float = 0.05

    def __post_init__(self) -> None:
        if self.neighbours < 1:
            raise ValueError("neighbours must be positive")
        if not 1 <= self.minimum_neighbours <= self.neighbours:
            raise ValueError("minimum_neighbours must lie in [1, neighbours]")
        if self.max_proposals < 1:
            raise ValueError("max_proposals must be positive")
        if not 0.0 <= self.minimum_prior < 1.0:
            raise ValueError("minimum_prior must lie in [0, 1)")

    @classmethod
    def from_protocol(cls, payload: Mapping[str, Any]) -> "ProposerConfig":
        section = payload.get("proposer", {})
        return cls(
            neighbours=int(section["neighbours"]),
            minimum_neighbours=int(section["minimum_neighbours"]),
            max_proposals=int(section["max_proposals"]),
            minimum_prior=float(section["minimum_prior"]),
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


class HypothesisProposer:
    """Retrieves the modes that historically followed contexts like this one.

    The reference set is the fitted dataset: one row per historical observation
    point, standardized in the same way the mode library was, each already
    assigned to a mode (or to noise).  Retrieval is exact k-nearest-neighbour in
    that standardized space — deterministic, and cheap enough per minute that
    no approximation is warranted at this scale.
    """

    def __init__(
        self,
        *,
        library: ModeLibrary,
        reference_features: np.ndarray,
        reference_modes: Sequence[str | None],
        center: np.ndarray,
        scale: np.ndarray,
        config: ProposerConfig | None = None,
    ) -> None:
        features = np.asarray(reference_features, dtype=float)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise HypothesisProposerError(
                f"reference features must be (n, {FEATURE_DIM}), got {features.shape}"
            )
        modes = tuple(reference_modes)
        if len(modes) != features.shape[0]:
            raise HypothesisProposerError(
                "reference features and mode assignments disagree in length"
            )
        known = set(library.mode_ids)
        unknown = {m for m in modes if m is not None and m not in known}
        if unknown:
            raise HypothesisProposerError(
                f"reference rows cite modes absent from the library: {sorted(unknown)[:5]}"
            )
        self.library = library
        self.config = config or ProposerConfig()
        self._center = np.asarray(center, dtype=float)
        self._scale = np.asarray(scale, dtype=float)
        if self._center.shape != (FEATURE_DIM,) or self._scale.shape != (FEATURE_DIM,):
            raise HypothesisProposerError("standardization vectors are the wrong width")
        if np.any(self._scale <= 0.0):
            raise HypothesisProposerError("standardization scale must be positive")
        # Impute once, at construction: a NaN component becomes the reference
        # centre, which is the same thing the query does, so a missing value
        # contributes nothing to the distance instead of poisoning it.
        self._reference = self._standardize(features)
        self._modes = modes

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        z = (np.asarray(features, dtype=float) - self._center) / self._scale
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def propose(self, context: Sequence[float]) -> tuple[HypothesisProposal, ...]:
        """Rank the modes that followed the nearest historical contexts.

        Returns an empty tuple when too few neighbours are available to say
        anything — the pool then holds a residual of one rather than spawning on
        a handful of points.
        """

        query = self._standardize(np.asarray(context, dtype=float).reshape(1, -1))[0]
        if self._reference.shape[0] == 0:
            return ()
        distances = np.linalg.norm(self._reference - query, axis=1)
        take = min(self.config.neighbours, distances.size)
        # argpartition then sort the head: O(n) instead of a full sort per minute.
        head = np.argpartition(distances, take - 1)[:take]
        head = head[np.argsort(distances[head], kind="stable")]

        counts: dict[str, int] = {}
        distance_sums: dict[str, float] = {}
        assigned = 0
        for index in head:
            mode_id = self._modes[int(index)]
            if mode_id is None:
                continue
            assigned += 1
            counts[mode_id] = counts.get(mode_id, 0) + 1
            distance_sums[mode_id] = distance_sums.get(mode_id, 0.0) + float(
                distances[int(index)]
            )
        if assigned < self.config.minimum_neighbours:
            return ()

        # Priors are shares of the retrieved neighbourhood, so unassigned
        # (noise) neighbours dilute every mode instead of being redistributed.
        # That dilution is the residual's first source of evidence.
        total = float(len(head))
        frontier = self._frontier(counts, distance_sums, total)
        proposals = [
            HypothesisProposal(
                mode_id=mode_id,
                prior=weight / total,
                neighbour_count=int(weight),
                neighbour_distance=distance,
            )
            for mode_id, weight, distance in frontier
        ]
        proposals.sort(key=lambda p: (-p.prior, p.neighbour_distance, p.mode_id))
        return tuple(proposals[: self.config.max_proposals])

    def _frontier(
        self,
        counts: Mapping[str, int],
        distance_sums: Mapping[str, float],
        total: float,
    ) -> list[tuple[str, float, float]]:
        """Choose how specific a claim the neighbourhood actually supports.

        A leaf mode that the neighbours clearly agree on is proposed as itself.
        When the neighbours split across several fine modes so that none of them
        clears ``minimum_prior`` alone, their shared ancestor is proposed
        instead: the honest reading of an ambiguous context is a coarser claim,
        not silence. That is also what later gives ``SPLIT`` something to split
        — a leaf has no children, so a pool that only ever holds leaves could
        never split at all.

        The returned nodes are mutually non-ancestral, so their neighbour sets
        are disjoint and their priors sum to at most one.
        """

        library = self.library
        parent = {mode.mode_id: mode.parent_mode_id for mode in library.modes}
        rolled: dict[str, float] = {}
        weighted_distance: dict[str, float] = {}
        for mode_id, count in counts.items():
            node: str | None = mode_id
            while node is not None:
                rolled[node] = rolled.get(node, 0.0) + count
                node = parent.get(node)

        selected: set[str] = set()
        for mode_id in counts:
            node = mode_id
            while node is not None and rolled[node] / total < self.config.minimum_prior:
                node = parent.get(node)
            if node is not None:
                selected.add(node)

        # Keep only the most specific selections: an ancestor of another
        # selected node would double-count the same neighbours.
        ancestors: set[str] = set()
        for node in selected:
            walker = parent.get(node)
            while walker is not None:
                ancestors.add(walker)
                walker = parent.get(walker)
        frontier = sorted(selected - ancestors)

        # A frontier node's distance is the neighbour-weighted mean over the
        # leaves it absorbed.
        for node in frontier:
            leaves = [
                mode_id
                for mode_id in counts
                if node == mode_id or self._is_ancestor(parent, node, mode_id)
            ]
            weight = sum(counts[leaf] for leaf in leaves)
            weighted_distance[node] = (
                sum(distance_sums[leaf] for leaf in leaves) / weight
                if weight
                else 0.0
            )
        return [(node, rolled[node], weighted_distance[node]) for node in frontier]

    @staticmethod
    def _is_ancestor(parent: Mapping[str, str | None], node: str, of: str) -> bool:
        walker = parent.get(of)
        while walker is not None:
            if walker == node:
                return True
            walker = parent.get(walker)
        return False


__all__ = [
    "CONTEXT_TIMEFRAMES",
    "DELIVERY_PHASES",
    "SESSION_PHASES",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "HypothesisProposer",
    "HypothesisProposerError",
    "ProposerConfig",
    "load_hypothesis_protocol",
    "observation_features",
]
