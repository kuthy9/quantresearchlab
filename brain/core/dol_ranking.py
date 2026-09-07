"""Deterministic shadow-only DOL candidate ranking.

The caller owns all market-fact resolution.  This module consumes normalized
views of the existing ``external_draw_candidates`` and directional
``obstruction_views`` together with one existing path competition set.  It
does not discover liquidity, update path beliefs, choose an action, or grant
trading authority.

The configured candidate weights are frozen development parameters.  The
softmax values and the path-weighted joint quality are diagnostic quantities,
not fitted likelihoods or calibrated Bayesian posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .path_belief import (
    PathCompetitionSetState,
    PathKind,
    PathStatus,
)


DOL_RANKING_SCHEMA_VERSION = 1
DOL_FEATURE_NAMES = (
    "intercept",
    "candidate_strength",
    "timeframe_value",
    "structural_rank_value",
    "distance_proximity",
    "age_freshness",
    "hard_obstacle_fraction",
    "soft_obstacle_fraction",
)
_TIMEFRAMES = ("1m", "5m", "15m", "1H", "4H")
_STRUCTURAL_RANKS = ("internal", "external")


class DOLRankingProtocolError(ValueError):
    """Raised when the frozen diagnostic ranking protocol is unsafe."""


class DOLDirection(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def candidate_side(self) -> str:
        return "above" if self is DOLDirection.LONG else "below"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _exact_source_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    raw = tuple(values)
    if (
        not raw
        or len(raw) != len(set(raw))
        or any(not isinstance(value, str) or not value for value in raw)
    ):
        raise ValueError(f"{name} must contain unique non-empty identities")
    return tuple(sorted(raw))


def _string_float_mapping(
    value: Any,
    *,
    name: str,
    required_keys: Sequence[str],
) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping) or set(value) != set(required_keys):
        raise DOLRankingProtocolError(
            f"{name} must define exactly {tuple(required_keys)}"
        )
    try:
        return tuple(
            (
                key,
                _finite_number(value[key], name=f"{name}.{key}"),
            )
            for key in required_keys
        )
    except ValueError as error:
        raise DOLRankingProtocolError(str(error)) from error


def _mapping_value(values: tuple[tuple[str, float], ...], key: str) -> float:
    for item_key, item_value in values:
        if item_key == key:
            return item_value
    raise ValueError(f"unregistered diagnostic ranking value: {key}")


@dataclass(frozen=True)
class DOLRankingProtocol:
    schema_version: int
    protocol_version: str
    model_version: str
    status: str
    authority: str
    probability_interpretation: str
    joint_quality_interpretation: str
    target_interval_rule: str
    target_self_exclusion_rule: str
    co_location_tolerance_points: float
    distance_scale_points: float
    age_scale_real_completed_bars: float
    obstacle_count_cap: int
    timeframe_values: tuple[tuple[str, float], ...]
    structural_rank_values: tuple[tuple[str, float], ...]
    feature_weights: tuple[tuple[str, float], ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != DOL_RANKING_SCHEMA_VERSION
            or not self.protocol_version
            or not self.model_version
            or self.status != "development_unvalidated"
            or self.authority != "shadow_only"
            or "not_calibrated_posterior"
            not in self.probability_interpretation
            or "not_calibrated_posterior"
            not in self.joint_quality_interpretation
            or self.target_interval_rule
            != "strict_open_interval_from_current_price_to_target_price"
            or self.target_self_exclusion_rule
            != "exclude_identity_source_intersection_or_colocated_target_obstructions"
            or not math.isfinite(float(self.co_location_tolerance_points))
            or self.co_location_tolerance_points < 0.0
            or not math.isfinite(float(self.distance_scale_points))
            or self.distance_scale_points <= 0.0
            or not math.isfinite(float(self.age_scale_real_completed_bars))
            or self.age_scale_real_completed_bars <= 0.0
            or type(self.obstacle_count_cap) is not int
            or self.obstacle_count_cap <= 0
            or tuple(key for key, _ in self.timeframe_values) != _TIMEFRAMES
            or tuple(key for key, _ in self.structural_rank_values)
            != _STRUCTURAL_RANKS
            or tuple(key for key, _ in self.feature_weights)
            != DOL_FEATURE_NAMES
            or any(
                not math.isfinite(float(value))
                for _, value in (
                    *self.timeframe_values,
                    *self.structural_rank_values,
                    *self.feature_weights,
                )
            )
            or len(self.fingerprint) != 64
        ):
            raise ValueError("DOL diagnostic ranking protocol is invalid")

    def feature_weight(self, name: str) -> float:
        return _mapping_value(self.feature_weights, name)

    def timeframe_value(self, timeframe: str) -> float:
        return _mapping_value(self.timeframe_values, timeframe)

    def structural_rank_value(self, rank: str) -> float:
        return _mapping_value(self.structural_rank_values, rank)


def load_dol_ranking_protocol(
    path: str | Path,
) -> DOLRankingProtocol:
    """Load and fail-close the DOL subsection of the frozen path protocol."""

    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[2] / source
    try:
        root = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DOLRankingProtocolError(
            f"unable to load DOL diagnostic ranking protocol: {source}"
        ) from error
    if not isinstance(root, Mapping):
        raise DOLRankingProtocolError("path hypothesis protocol root must be an object")
    payload = root.get("dol_diagnostic_ranking")
    if not isinstance(payload, Mapping):
        raise DOLRankingProtocolError(
            "dol_diagnostic_ranking protocol section is missing"
        )
    required = {
        "schema_version",
        "protocol_version",
        "model_version",
        "status",
        "authority",
        "probability_interpretation",
        "joint_quality_interpretation",
        "target_interval_rule",
        "target_self_exclusion_rule",
        "co_location_tolerance_points",
        "distance_scale_points",
        "age_scale_real_completed_bars",
        "obstacle_count_cap",
        "timeframe_values",
        "structural_rank_values",
        "feature_weights",
    }
    if set(payload) != required:
        raise DOLRankingProtocolError(
            "DOL diagnostic ranking protocol fields are not frozen exactly"
        )
    try:
        schema_version = int(payload["schema_version"])
        obstacle_count_cap = int(payload["obstacle_count_cap"])
        if isinstance(payload["schema_version"], bool) or isinstance(
            payload["obstacle_count_cap"], bool
        ):
            raise TypeError
        protocol = DOLRankingProtocol(
            schema_version=schema_version,
            protocol_version=str(payload["protocol_version"]).strip(),
            model_version=str(payload["model_version"]).strip(),
            status=str(payload["status"]).strip(),
            authority=str(payload["authority"]).strip(),
            probability_interpretation=str(
                payload["probability_interpretation"]
            ).strip(),
            joint_quality_interpretation=str(
                payload["joint_quality_interpretation"]
            ).strip(),
            target_interval_rule=str(payload["target_interval_rule"]).strip(),
            target_self_exclusion_rule=str(
                payload["target_self_exclusion_rule"]
            ).strip(),
            co_location_tolerance_points=_finite_number(
                payload["co_location_tolerance_points"],
                name="co_location_tolerance_points",
            ),
            distance_scale_points=_finite_number(
                payload["distance_scale_points"],
                name="distance_scale_points",
            ),
            age_scale_real_completed_bars=_finite_number(
                payload["age_scale_real_completed_bars"],
                name="age_scale_real_completed_bars",
            ),
            obstacle_count_cap=obstacle_count_cap,
            timeframe_values=_string_float_mapping(
                payload["timeframe_values"],
                name="timeframe_values",
                required_keys=_TIMEFRAMES,
            ),
            structural_rank_values=_string_float_mapping(
                payload["structural_rank_values"],
                name="structural_rank_values",
                required_keys=_STRUCTURAL_RANKS,
            ),
            feature_weights=_string_float_mapping(
                payload["feature_weights"],
                name="feature_weights",
                required_keys=DOL_FEATURE_NAMES,
            ),
            fingerprint=_canonical_hash(payload),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, DOLRankingProtocolError):
            raise
        raise DOLRankingProtocolError(str(error)) from error
    return protocol


@dataclass(frozen=True)
class DOLCandidateFact:
    """Caller-resolved external draw fact; this module does no discovery."""

    candidate_id: str
    timeframe: str
    side: str
    target_price: float
    source_kind: str
    source_ids: tuple[str, ...]
    structural_rank: str
    strength: float
    age_real_completed_bars: int
    path: PathKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", PathKind(self.path))
        object.__setattr__(
            self,
            "source_ids",
            _exact_source_ids(self.source_ids, name="DOL candidate source_ids"),
        )
        if (
            not self.candidate_id
            or not self.timeframe
            or self.side not in {"above", "below"}
            or not math.isfinite(float(self.target_price))
            or self.target_price <= 0.0
            or not self.source_kind
            or not self.structural_rank
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
            or type(self.age_real_completed_bars) is not int
            or self.age_real_completed_bars < 0
            or self.path is PathKind.RESIDUAL_UNKNOWN
        ):
            raise ValueError("DOL candidate fact is invalid")


@dataclass(frozen=True)
class DOLObstructionFact:
    """Caller-resolved price obstacle from one directional obstruction view."""

    obstruction_id: str
    lower_bound: float
    upper_bound: float
    hard: bool
    source_kind: str
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_ids",
            _exact_source_ids(self.source_ids, name="DOL obstruction source_ids"),
        )
        if (
            not self.obstruction_id
            or not math.isfinite(float(self.lower_bound))
            or not math.isfinite(float(self.upper_bound))
            or not 0.0 < float(self.lower_bound) <= float(self.upper_bound)
            or type(self.hard) is not bool
            or not self.source_kind
        ):
            raise ValueError("DOL obstruction fact is invalid")

    def contact_price(self, direction: DOLDirection) -> float:
        direction = DOLDirection(direction)
        return (
            float(self.lower_bound)
            if direction is DOLDirection.LONG
            else float(self.upper_bound)
        )


@dataclass(frozen=True)
class DOLObstructionViewFact:
    """One already-resolved directional view; it remains descriptive."""

    direction: DOLDirection
    hard_barriers: tuple[DOLObstructionFact, ...]
    soft_frictions: tuple[DOLObstructionFact, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", DOLDirection(self.direction))
        hard = tuple(self.hard_barriers)
        soft = tuple(self.soft_frictions)
        object.__setattr__(self, "hard_barriers", hard)
        object.__setattr__(self, "soft_frictions", soft)
        identities = tuple(item.obstruction_id for item in (*hard, *soft))
        if (
            any(not isinstance(item, DOLObstructionFact) for item in (*hard, *soft))
            or any(not item.hard for item in hard)
            or any(item.hard for item in soft)
            or len(identities) != len(set(identities))
        ):
            raise ValueError("DOL obstruction view fact is invalid")


@dataclass(frozen=True)
class RankedDOLCandidate:
    rank: int
    candidate_id: str
    direction: DOLDirection
    target_price: float
    distance_points: float
    candidate_source_kind: str
    candidate_source_ids: tuple[str, ...]
    path: PathKind
    path_hypothesis_id: str
    path_probability: float
    conditional_log_weight: float
    normalized_diagnostic_weight: float
    diagnostic_joint_quality: float
    feature_values: tuple[tuple[str, float], ...]
    hard_obstacle_ids: tuple[str, ...]
    soft_obstacle_ids: tuple[str, ...]
    excluded_obstacles: tuple[tuple[str, str], ...]
    status: str
    authority: str
    model_version: str
    protocol_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", DOLDirection(self.direction))
        object.__setattr__(self, "path", PathKind(self.path))
        identities = (*self.hard_obstacle_ids, *self.soft_obstacle_ids)
        excluded_ids = tuple(identity for identity, _ in self.excluded_obstacles)
        if (
            type(self.rank) is not int
            or self.rank <= 0
            or not self.candidate_id
            or not math.isfinite(float(self.target_price))
            or self.target_price <= 0.0
            or not math.isfinite(float(self.distance_points))
            or self.distance_points <= 0.0
            or not self.candidate_source_kind
            or not self.candidate_source_ids
            or not self.path_hypothesis_id
            or not math.isfinite(float(self.path_probability))
            or not 0.0 < self.path_probability <= 1.0
            or not math.isfinite(float(self.conditional_log_weight))
            or not math.isfinite(float(self.normalized_diagnostic_weight))
            or not 0.0 < self.normalized_diagnostic_weight <= 1.0
            or not math.isfinite(float(self.diagnostic_joint_quality))
            or self.diagnostic_joint_quality <= 0.0
            or tuple(name for name, _ in self.feature_values) != DOL_FEATURE_NAMES
            or any(not math.isfinite(float(value)) for _, value in self.feature_values)
            or len(identities) != len(set(identities))
            or len(excluded_ids) != len(set(excluded_ids))
            or not set(identities).isdisjoint(excluded_ids)
            or self.status != "development_unvalidated"
            or self.authority != "shadow_only"
            or not self.model_version
            or len(self.protocol_fingerprint) != 64
        ):
            raise ValueError("ranked DOL candidate is invalid")


@dataclass(frozen=True)
class DOLRankingResult:
    ranking_id: str
    competition_set_id: str
    path_asof: pd.Timestamp
    direction: DOLDirection
    current_price: float
    ranked_candidates: tuple[RankedDOLCandidate, ...]
    excluded_candidates: tuple[tuple[str, str], ...]
    status: str
    authority: str
    protocol_version: str
    model_version: str
    protocol_fingerprint: str
    path_model_version: str
    path_protocol_fingerprint: str
    probability_interpretation: str
    joint_quality_interpretation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_asof", pd.Timestamp(self.path_asof))
        object.__setattr__(self, "direction", DOLDirection(self.direction))
        candidate_ids = tuple(item.candidate_id for item in self.ranked_candidates)
        excluded_ids = tuple(identity for identity, _ in self.excluded_candidates)
        if (
            not self.ranking_id
            or not self.competition_set_id
            or pd.isna(self.path_asof)
            or self.path_asof.tzinfo is None
            or not math.isfinite(float(self.current_price))
            or self.current_price <= 0.0
            or len(candidate_ids) != len(set(candidate_ids))
            or len(excluded_ids) != len(set(excluded_ids))
            or not set(candidate_ids).isdisjoint(excluded_ids)
            or tuple(item.rank for item in self.ranked_candidates)
            != tuple(range(1, len(self.ranked_candidates) + 1))
            or (
                self.ranked_candidates
                and not math.isclose(
                    math.fsum(
                        item.normalized_diagnostic_weight
                        for item in self.ranked_candidates
                    ),
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            or self.status != "development_unvalidated"
            or self.authority != "shadow_only"
            or not self.protocol_version
            or not self.model_version
            or len(self.protocol_fingerprint) != 64
            or not self.path_model_version
            or len(self.path_protocol_fingerprint) != 64
            or "not_calibrated_posterior"
            not in self.probability_interpretation
            or "not_calibrated_posterior"
            not in self.joint_quality_interpretation
        ):
            raise ValueError("DOL ranking result is invalid")


@dataclass(frozen=True)
class _ScoredCandidate:
    candidate: DOLCandidateFact
    path_hypothesis_id: str
    path_probability: float
    distance_points: float
    conditional_log_weight: float
    feature_values: tuple[tuple[str, float], ...]
    hard_obstacle_ids: tuple[str, ...]
    soft_obstacle_ids: tuple[str, ...]
    excluded_obstacles: tuple[tuple[str, str], ...]
    normalized_diagnostic_weight: float = 0.0
    diagnostic_joint_quality: float = 0.0


def _target_self_reason(
    protocol: DOLRankingProtocol,
    candidate: DOLCandidateFact,
    obstruction: DOLObstructionFact,
) -> str | None:
    if obstruction.obstruction_id == candidate.candidate_id:
        return "target_self_identity"
    candidate_sources = set(candidate.source_ids)
    obstruction_sources = set(obstruction.source_ids)
    if (
        obstruction.obstruction_id in candidate_sources
        or candidate.candidate_id in obstruction_sources
        or not candidate_sources.isdisjoint(obstruction_sources)
    ):
        return "target_self_source"
    tolerance = protocol.co_location_tolerance_points
    if (
        obstruction.lower_bound - tolerance
        <= candidate.target_price
        <= obstruction.upper_bound + tolerance
    ):
        return "target_self_colocated"
    return None


def _obstacles_for_candidate(
    protocol: DOLRankingProtocol,
    *,
    direction: DOLDirection,
    current_price: float,
    candidate: DOLCandidateFact,
    view: DOLObstructionViewFact,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    obstacles = tuple(
        sorted(
            (*view.hard_barriers, *view.soft_frictions),
            key=lambda item: (
                abs(item.contact_price(direction) - current_price),
                item.contact_price(direction),
                item.obstruction_id,
            ),
        )
    )
    hard: list[str] = []
    soft: list[str] = []
    excluded: list[tuple[str, str]] = []
    for obstruction in obstacles:
        reason = _target_self_reason(protocol, candidate, obstruction)
        if reason is None:
            contact = obstruction.contact_price(direction)
            inside = (
                current_price < contact < candidate.target_price
                if direction is DOLDirection.LONG
                else candidate.target_price < contact < current_price
            )
            if not inside:
                reason = "outside_strict_path_interval"
        if reason is not None:
            excluded.append((obstruction.obstruction_id, reason))
        elif obstruction.hard:
            hard.append(obstruction.obstruction_id)
        else:
            soft.append(obstruction.obstruction_id)
    return tuple(hard), tuple(soft), tuple(excluded)


def _feature_values(
    protocol: DOLRankingProtocol,
    *,
    candidate: DOLCandidateFact,
    distance_points: float,
    hard_count: int,
    soft_count: int,
) -> tuple[tuple[str, float], ...]:
    cap = protocol.obstacle_count_cap
    values = {
        "intercept": 1.0,
        "candidate_strength": float(candidate.strength),
        "timeframe_value": protocol.timeframe_value(candidate.timeframe),
        "structural_rank_value": protocol.structural_rank_value(
            candidate.structural_rank
        ),
        "distance_proximity": 1.0
        / (1.0 + distance_points / protocol.distance_scale_points),
        "age_freshness": 1.0
        / (
            1.0
            + candidate.age_real_completed_bars
            / protocol.age_scale_real_completed_bars
        ),
        "hard_obstacle_fraction": min(hard_count, cap) / cap,
        "soft_obstacle_fraction": min(soft_count, cap) / cap,
    }
    return tuple((name, float(values[name])) for name in DOL_FEATURE_NAMES)


def _ranked_payload(
    *,
    competition_set_id: str,
    path_asof: pd.Timestamp,
    direction: DOLDirection,
    current_price: float,
    ranked: Sequence[RankedDOLCandidate],
    excluded_candidates: Sequence[tuple[str, str]],
    protocol: DOLRankingProtocol,
    path_state: PathCompetitionSetState,
) -> dict[str, Any]:
    return {
        "competition_set_id": competition_set_id,
        "path_asof": path_asof.isoformat(),
        "direction": direction.value,
        "current_price": current_price,
        "protocol_fingerprint": protocol.fingerprint,
        "path_protocol_fingerprint": path_state.protocol_fingerprint,
        "ranked_candidates": [
            {
                "rank": item.rank,
                "candidate_id": item.candidate_id,
                "target_price": item.target_price,
                "candidate_source_ids": list(item.candidate_source_ids),
                "path": item.path.value,
                "path_hypothesis_id": item.path_hypothesis_id,
                "path_probability": item.path_probability,
                "conditional_log_weight": item.conditional_log_weight,
                "normalized_diagnostic_weight": (
                    item.normalized_diagnostic_weight
                ),
                "diagnostic_joint_quality": item.diagnostic_joint_quality,
                "feature_values": list(item.feature_values),
                "hard_obstacle_ids": list(item.hard_obstacle_ids),
                "soft_obstacle_ids": list(item.soft_obstacle_ids),
                "excluded_obstacles": list(item.excluded_obstacles),
            }
            for item in ranked
        ],
        "excluded_candidates": list(excluded_candidates),
    }


def rank_dol_candidates(
    protocol: DOLRankingProtocol,
    *,
    direction: DOLDirection,
    current_price: float,
    external_draw_candidates: Sequence[DOLCandidateFact],
    obstruction_view: DOLObstructionViewFact,
    path_state: PathCompetitionSetState,
) -> DOLRankingResult:
    """Rank caller-provided draws without creating facts or action authority.

    Candidate feature weights are softmax-normalized across the eligible input
    set.  ``diagnostic_joint_quality`` multiplies that diagnostic weight by the
    referenced path probability solely for ordering and audit.  Neither value
    is a calibrated posterior.
    """

    if not isinstance(protocol, DOLRankingProtocol):
        raise TypeError("protocol must be DOLRankingProtocol")
    if not isinstance(path_state, PathCompetitionSetState):
        raise TypeError("path_state must be PathCompetitionSetState")
    direction = DOLDirection(direction)
    current_price = _finite_number(current_price, name="current_price")
    if current_price <= 0.0:
        raise ValueError("current_price must be positive")
    if not isinstance(obstruction_view, DOLObstructionViewFact):
        raise TypeError("obstruction_view must be DOLObstructionViewFact")
    if obstruction_view.direction is not direction:
        raise ValueError("obstruction view direction does not match ranking direction")
    if path_state.status is not PathStatus.ACTIVE:
        raise ValueError("DOL ranking requires an active path competition set")
    if (
        path_state.protocol_status != "development_unvalidated"
        or path_state.authority != "shadow_only"
    ):
        raise ValueError("DOL ranking requires a shadow-only path state")

    candidates = tuple(external_draw_candidates)
    if any(not isinstance(item, DOLCandidateFact) for item in candidates):
        raise TypeError("external_draw_candidates must contain DOLCandidateFact")
    candidate_ids = tuple(item.candidate_id for item in candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("external draw candidate identities must be unique")

    excluded_candidates: list[tuple[str, str]] = []
    scored: list[_ScoredCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if candidate.side != direction.candidate_side:
            excluded_candidates.append(
                (candidate.candidate_id, "side_opposes_requested_direction")
            )
            continue
        strictly_ahead = (
            candidate.target_price > current_price
            if direction is DOLDirection.LONG
            else candidate.target_price < current_price
        )
        if not strictly_ahead:
            excluded_candidates.append(
                (candidate.candidate_id, "target_not_strictly_ahead")
            )
            continue
        member = path_state.member(candidate.path)
        if member.status is not PathStatus.ACTIVE or member.probability <= 0.0:
            excluded_candidates.append(
                (candidate.candidate_id, "path_hypothesis_inactive")
            )
            continue
        hard_ids, soft_ids, excluded_obstacles = _obstacles_for_candidate(
            protocol,
            direction=direction,
            current_price=current_price,
            candidate=candidate,
            view=obstruction_view,
        )
        distance_points = abs(candidate.target_price - current_price)
        features = _feature_values(
            protocol,
            candidate=candidate,
            distance_points=distance_points,
            hard_count=len(hard_ids),
            soft_count=len(soft_ids),
        )
        conditional_log_weight = math.fsum(
            protocol.feature_weight(name) * value
            for name, value in features
        )
        if not math.isfinite(conditional_log_weight):
            raise ValueError("DOL conditional log weight is not finite")
        scored.append(
            _ScoredCandidate(
                candidate=candidate,
                path_hypothesis_id=member.hypothesis_id,
                path_probability=float(member.probability),
                distance_points=distance_points,
                conditional_log_weight=conditional_log_weight,
                feature_values=features,
                hard_obstacle_ids=hard_ids,
                soft_obstacle_ids=soft_ids,
                excluded_obstacles=excluded_obstacles,
            )
        )

    if scored:
        maximum = max(item.conditional_log_weight for item in scored)
        exponentials = tuple(
            math.exp(item.conditional_log_weight - maximum) for item in scored
        )
        denominator = math.fsum(exponentials)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("DOL softmax normalizer is invalid")
        scored = [
            replace(
                item,
                normalized_diagnostic_weight=weight / denominator,
                diagnostic_joint_quality=(
                    item.path_probability * (weight / denominator)
                ),
            )
            for item, weight in zip(scored, exponentials)
        ]

    ordered = tuple(
        sorted(
            scored,
            key=lambda item: (
                -item.diagnostic_joint_quality,
                -item.normalized_diagnostic_weight,
                item.distance_points,
                item.candidate.candidate_id,
            ),
        )
    )
    ranked = tuple(
        RankedDOLCandidate(
            rank=index,
            candidate_id=item.candidate.candidate_id,
            direction=direction,
            target_price=float(item.candidate.target_price),
            distance_points=item.distance_points,
            candidate_source_kind=item.candidate.source_kind,
            candidate_source_ids=item.candidate.source_ids,
            path=item.candidate.path,
            path_hypothesis_id=item.path_hypothesis_id,
            path_probability=item.path_probability,
            conditional_log_weight=item.conditional_log_weight,
            normalized_diagnostic_weight=(
                item.normalized_diagnostic_weight
            ),
            diagnostic_joint_quality=item.diagnostic_joint_quality,
            feature_values=item.feature_values,
            hard_obstacle_ids=item.hard_obstacle_ids,
            soft_obstacle_ids=item.soft_obstacle_ids,
            excluded_obstacles=item.excluded_obstacles,
            status=protocol.status,
            authority=protocol.authority,
            model_version=protocol.model_version,
            protocol_fingerprint=protocol.fingerprint,
        )
        for index, item in enumerate(ordered, start=1)
    )
    excluded = tuple(sorted(excluded_candidates))
    payload = _ranked_payload(
        competition_set_id=path_state.competition_set_id,
        path_asof=path_state.asof,
        direction=direction,
        current_price=current_price,
        ranked=ranked,
        excluded_candidates=excluded,
        protocol=protocol,
        path_state=path_state,
    )
    return DOLRankingResult(
        ranking_id=f"dol-ranking:{_canonical_hash(payload)[:32]}",
        competition_set_id=path_state.competition_set_id,
        path_asof=path_state.asof,
        direction=direction,
        current_price=current_price,
        ranked_candidates=ranked,
        excluded_candidates=excluded,
        status=protocol.status,
        authority=protocol.authority,
        protocol_version=protocol.protocol_version,
        model_version=protocol.model_version,
        protocol_fingerprint=protocol.fingerprint,
        path_model_version=path_state.model_version,
        path_protocol_fingerprint=path_state.protocol_fingerprint,
        probability_interpretation=protocol.probability_interpretation,
        joint_quality_interpretation=protocol.joint_quality_interpretation,
    )


__all__ = [
    "DOLCandidateFact",
    "DOLDirection",
    "DOLObstructionFact",
    "DOLObstructionViewFact",
    "DOLRankingProtocol",
    "DOLRankingProtocolError",
    "DOLRankingResult",
    "RankedDOLCandidate",
    "load_dol_ranking_protocol",
    "rank_dol_candidates",
]
