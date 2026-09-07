"""Path-marginalized, shadow-only DOL probability boundary.

This module does not discover liquidity or obstacles.  It reuses the exact
candidate eligibility and obstruction resolution implemented by
``brain.core.dol_ranking`` and marginalizes the resulting conditional model
over one existing path competition set::

    P(DOL=j) = sum_h P(h) * P(DOL=j | h, state)

``no_target_before_common_horizon`` is an explicit outcome in every path
conditional distribution.  The bundled parameters define an unfitted
protocol template only; they cannot produce a probability result.  An exact
fitted/admitted artifact bound to the active path model is required, and this
bounded component remains shadow-only and never grants action authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .dol_ranking import (
    DOLCandidateFact,
    DOLDirection,
    DOLObstructionViewFact,
    DOLRankingProtocol,
    RankedDOLCandidate,
    rank_dol_candidates,
)
from .path_belief import (
    PATH_KINDS,
    PathCompetitionSetState,
    PathKind,
    PathStatus,
)


DOL_PROBABILITY_SCHEMA_VERSION = 1
DOL_PROBABILITY_ARTIFACT_SCHEMA_VERSION = 2
NO_TARGET_BEFORE_HORIZON = "no_target_before_common_horizon"
_PROBABILITY_FORMULA = (
    "p_dol_j_equals_sum_over_active_paths_p_h_times_p_j_given_h_state"
)
_CANDIDATE_ELIGIBILITY_SOURCE = (
    "brain.core.dol_ranking.rank_dol_candidates"
)
_OBSTACLE_FACT_SOURCE = "brain.core.dol_ranking.DOLObstructionViewFact"
_FITTED_PROBABILITY_INTERPRETATION = (
    "normalized_path_marginal_distribution_fitted_admitted_artifact"
)
_HEX_CHARACTERS = frozenset("0123456789abcdef")


class DOLProbabilityProtocolError(ValueError):
    """Raised when the DOL probability protocol is incomplete or unsafe."""


class DOLProbabilityArtifactError(ValueError):
    """Raised when a fitted-model artifact is not exact and admitted."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX_CHARACTERS)
    )


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


def _path_value_mapping(
    value: Any,
    *,
    name: str,
) -> tuple[tuple[PathKind, float], ...]:
    expected = {path.value for path in PATH_KINDS}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} must define exactly every registered path")
    return tuple(
        (
            path,
            _finite_number(value[path.value], name=f"{name}.{path.value}"),
        )
        for path in PATH_KINDS
    )


def _path_value(
    values: tuple[tuple[PathKind, float], ...],
    path: PathKind,
) -> float:
    return values[PATH_KINDS.index(PathKind(path))][1]


@dataclass(frozen=True)
class DOLConditionalModelParameters:
    """Minimal parameters applied after the existing candidate fact scorer."""

    candidate_logit_scale: float
    candidate_path_log_weight_adjustment: tuple[tuple[PathKind, float], ...]
    no_target_log_weight: tuple[tuple[PathKind, float], ...]

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.candidate_logit_scale))
            or tuple(
                path for path, _ in self.candidate_path_log_weight_adjustment
            )
            != PATH_KINDS
            or tuple(path for path, _ in self.no_target_log_weight)
            != PATH_KINDS
            or any(
                not math.isfinite(float(value))
                for _, value in (
                    *self.candidate_path_log_weight_adjustment,
                    *self.no_target_log_weight,
                )
            )
        ):
            raise ValueError("DOL conditional model parameters are invalid")

    def candidate_adjustment(self, path: PathKind) -> float:
        return _path_value(self.candidate_path_log_weight_adjustment, path)

    def no_target_weight(self, path: PathKind) -> float:
        return _path_value(self.no_target_log_weight, path)

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_logit_scale": float(self.candidate_logit_scale),
            "candidate_path_log_weight_adjustment": {
                path.value: float(value)
                for path, value in self.candidate_path_log_weight_adjustment
            },
            "no_target_log_weight": {
                path.value: float(value)
                for path, value in self.no_target_log_weight
            },
        }


def _parameters(value: Any, *, name: str) -> DOLConditionalModelParameters:
    required = {
        "candidate_logit_scale",
        "candidate_path_log_weight_adjustment",
        "no_target_log_weight",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{name} fields are not frozen exactly")
    return DOLConditionalModelParameters(
        candidate_logit_scale=_finite_number(
            value["candidate_logit_scale"],
            name=f"{name}.candidate_logit_scale",
        ),
        candidate_path_log_weight_adjustment=_path_value_mapping(
            value["candidate_path_log_weight_adjustment"],
            name=f"{name}.candidate_path_log_weight_adjustment",
        ),
        no_target_log_weight=_path_value_mapping(
            value["no_target_log_weight"],
            name=f"{name}.no_target_log_weight",
        ),
    )


def _protocol_payload(
    *,
    schema_version: int,
    protocol_version: str,
    model_version: str,
    status: str,
    calibration_status: str,
    authority: str,
    action_authority: bool,
    probability_formula: str,
    no_target_outcome: str,
    candidate_eligibility_source: str,
    obstacle_fact_source: str,
    ranking_protocol_fingerprint: str,
    development_model: DOLConditionalModelParameters,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "protocol_version": protocol_version,
        "model_version": model_version,
        "status": status,
        "calibration_status": calibration_status,
        "authority": authority,
        "action_authority": action_authority,
        "probability_formula": probability_formula,
        "no_target_outcome": no_target_outcome,
        "candidate_eligibility_source": candidate_eligibility_source,
        "obstacle_fact_source": obstacle_fact_source,
        "ranking_protocol_fingerprint": ranking_protocol_fingerprint,
        "development_model": development_model.payload(),
    }


@dataclass(frozen=True)
class DOLProbabilityProtocol:
    schema_version: int
    protocol_version: str
    model_version: str
    status: str
    calibration_status: str
    authority: str
    action_authority: bool
    probability_formula: str
    no_target_outcome: str
    candidate_eligibility_source: str
    obstacle_fact_source: str
    ranking_protocol_fingerprint: str
    development_model: DOLConditionalModelParameters
    fingerprint: str
    development_model_fingerprint: str

    def __post_init__(self) -> None:
        expected_protocol = _canonical_hash(
            _protocol_payload(
                schema_version=self.schema_version,
                protocol_version=self.protocol_version,
                model_version=self.model_version,
                status=self.status,
                calibration_status=self.calibration_status,
                authority=self.authority,
                action_authority=self.action_authority,
                probability_formula=self.probability_formula,
                no_target_outcome=self.no_target_outcome,
                candidate_eligibility_source=(
                    self.candidate_eligibility_source
                ),
                obstacle_fact_source=self.obstacle_fact_source,
                ranking_protocol_fingerprint=(
                    self.ranking_protocol_fingerprint
                ),
                development_model=self.development_model,
            )
        )
        expected_model = _canonical_hash(
            {
                "fit_status": "not_fitted",
                "admission_status": "not_admitted",
                "model_version": self.model_version,
                "protocol_fingerprint": expected_protocol,
                "ranking_protocol_fingerprint": (
                    self.ranking_protocol_fingerprint
                ),
                "parameters": self.development_model.payload(),
            }
        )
        if (
            self.schema_version != DOL_PROBABILITY_SCHEMA_VERSION
            or not self.protocol_version
            or not self.model_version
            or self.status != "development_unvalidated"
            or self.calibration_status != "not_fitted_not_admitted"
            or self.authority != "shadow_only"
            or self.action_authority is not False
            or self.probability_formula != _PROBABILITY_FORMULA
            or self.no_target_outcome != NO_TARGET_BEFORE_HORIZON
            or self.candidate_eligibility_source
            != _CANDIDATE_ELIGIBILITY_SOURCE
            or self.obstacle_fact_source != _OBSTACLE_FACT_SOURCE
            or not _is_sha256(self.ranking_protocol_fingerprint)
            or self.fingerprint != expected_protocol
            or self.development_model_fingerprint != expected_model
        ):
            raise ValueError("DOL probability protocol boundary is invalid")


def load_dol_probability_protocol(
    path: str | Path = "brain/configs/dol_probability.json",
) -> DOLProbabilityProtocol:
    """Load the exact versioned DOL probability protocol."""

    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[2] / source
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DOLProbabilityProtocolError(
            f"unable to load DOL probability protocol: {source}"
        ) from error
    required = {
        "schema_version",
        "protocol_version",
        "model_version",
        "status",
        "calibration_status",
        "authority",
        "action_authority",
        "probability_formula",
        "no_target_outcome",
        "candidate_eligibility_source",
        "obstacle_fact_source",
        "ranking_protocol_fingerprint",
        "development_model",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise DOLProbabilityProtocolError(
            "DOL probability protocol fields are not frozen exactly"
        )
    try:
        if type(payload["schema_version"]) is not int:
            raise TypeError("schema_version must be an integer")
        schema_version = payload["schema_version"]
        if payload["action_authority"] is not False:
            raise TypeError("action_authority must be false")
        development_model = _parameters(
            payload["development_model"],
            name="development_model",
        )
        semantic = _protocol_payload(
            schema_version=schema_version,
            protocol_version=str(payload["protocol_version"]).strip(),
            model_version=str(payload["model_version"]).strip(),
            status=str(payload["status"]).strip(),
            calibration_status=str(payload["calibration_status"]).strip(),
            authority=str(payload["authority"]).strip(),
            action_authority=False,
            probability_formula=str(payload["probability_formula"]).strip(),
            no_target_outcome=str(payload["no_target_outcome"]).strip(),
            candidate_eligibility_source=str(
                payload["candidate_eligibility_source"]
            ).strip(),
            obstacle_fact_source=str(payload["obstacle_fact_source"]).strip(),
            ranking_protocol_fingerprint=str(
                payload["ranking_protocol_fingerprint"]
            ).strip(),
            development_model=development_model,
        )
        fingerprint = _canonical_hash(semantic)
        model_fingerprint = _canonical_hash(
            {
                "fit_status": "not_fitted",
                "admission_status": "not_admitted",
                "model_version": semantic["model_version"],
                "protocol_fingerprint": fingerprint,
                "ranking_protocol_fingerprint": semantic[
                    "ranking_protocol_fingerprint"
                ],
                "parameters": development_model.payload(),
            }
        )
        return DOLProbabilityProtocol(
            schema_version=schema_version,
            protocol_version=semantic["protocol_version"],
            model_version=semantic["model_version"],
            status=semantic["status"],
            calibration_status=semantic["calibration_status"],
            authority=semantic["authority"],
            action_authority=False,
            probability_formula=semantic["probability_formula"],
            no_target_outcome=semantic["no_target_outcome"],
            candidate_eligibility_source=semantic[
                "candidate_eligibility_source"
            ],
            obstacle_fact_source=semantic["obstacle_fact_source"],
            ranking_protocol_fingerprint=semantic[
                "ranking_protocol_fingerprint"
            ],
            development_model=development_model,
            fingerprint=fingerprint,
            development_model_fingerprint=model_fingerprint,
        )
    except (TypeError, ValueError) as error:
        raise DOLProbabilityProtocolError(str(error)) from error


def _artifact_semantic_payload(
    *,
    schema_version: int,
    artifact_id: str,
    model_version: str,
    protocol_fingerprint: str,
    ranking_protocol_fingerprint: str,
    source_path_protocol_fingerprint: str,
    source_path_model_version: str,
    fit_status: str,
    admission_status: str,
    calibration_status: str,
    authority: str,
    action_authority: bool,
    parameters: DOLConditionalModelParameters,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "artifact_id": artifact_id,
        "model_version": model_version,
        "protocol_fingerprint": protocol_fingerprint,
        "ranking_protocol_fingerprint": ranking_protocol_fingerprint,
        "source_path_protocol_fingerprint": source_path_protocol_fingerprint,
        "source_path_model_version": source_path_model_version,
        "fit_status": fit_status,
        "admission_status": admission_status,
        "calibration_status": calibration_status,
        "authority": authority,
        "action_authority": action_authority,
        "parameters": parameters.payload(),
    }


@dataclass(frozen=True)
class DOLProbabilityModelArtifact:
    """Exact fitted/admitted conditional model; never action authority."""

    schema_version: int
    artifact_id: str
    model_version: str
    protocol_fingerprint: str
    ranking_protocol_fingerprint: str
    source_path_protocol_fingerprint: str
    source_path_model_version: str
    fit_status: str
    admission_status: str
    calibration_status: str
    authority: str
    action_authority: bool
    parameters: DOLConditionalModelParameters
    fingerprint: str

    def __post_init__(self) -> None:
        expected = _canonical_hash(
            _artifact_semantic_payload(
                schema_version=self.schema_version,
                artifact_id=self.artifact_id,
                model_version=self.model_version,
                protocol_fingerprint=self.protocol_fingerprint,
                ranking_protocol_fingerprint=(
                    self.ranking_protocol_fingerprint
                ),
                source_path_protocol_fingerprint=(
                    self.source_path_protocol_fingerprint
                ),
                source_path_model_version=self.source_path_model_version,
                fit_status=self.fit_status,
                admission_status=self.admission_status,
                calibration_status=self.calibration_status,
                authority=self.authority,
                action_authority=self.action_authority,
                parameters=self.parameters,
            )
        )
        if (
            self.schema_version != DOL_PROBABILITY_ARTIFACT_SCHEMA_VERSION
            or not self.artifact_id
            or not self.model_version
            or not _is_sha256(self.protocol_fingerprint)
            or not _is_sha256(self.ranking_protocol_fingerprint)
            or not _is_sha256(self.source_path_protocol_fingerprint)
            or not self.source_path_model_version
            or self.fit_status != "fitted"
            or self.admission_status != "admitted"
            or self.calibration_status != "fitted_admitted"
            or self.authority != "shadow_only"
            or self.action_authority is not False
            or self.fingerprint != expected
        ):
            raise ValueError("DOL probability artifact boundary is invalid")


def load_dol_probability_model_artifact(
    path: str | Path,
    *,
    protocol: DOLProbabilityProtocol,
    expected_fingerprint: str,
) -> DOLProbabilityModelArtifact:
    """Load an exact fitted/admitted artifact and verify all bindings.

    ``expected_fingerprint`` must come from the caller's separate admission
    record.  An artifact cannot admit itself merely by recomputing its own
    canonical digest.
    """

    if not isinstance(protocol, DOLProbabilityProtocol):
        raise TypeError("protocol must be DOLProbabilityProtocol")
    if not _is_sha256(expected_fingerprint):
        raise DOLProbabilityArtifactError(
            "expected admitted artifact fingerprint is invalid"
        )
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[2] / source
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DOLProbabilityArtifactError(
            f"unable to load DOL probability artifact: {source}"
        ) from error
    required = {
        "schema_version",
        "artifact_id",
        "model_version",
        "protocol_fingerprint",
        "ranking_protocol_fingerprint",
        "source_path_protocol_fingerprint",
        "source_path_model_version",
        "fit_status",
        "admission_status",
        "calibration_status",
        "authority",
        "action_authority",
        "parameters",
        "artifact_fingerprint",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise DOLProbabilityArtifactError(
            "DOL probability artifact fields are not frozen exactly"
        )
    try:
        if type(payload["schema_version"]) is not int:
            raise TypeError("schema_version must be an integer")
        if payload["action_authority"] is not False:
            raise TypeError("action_authority must be false")
        artifact = DOLProbabilityModelArtifact(
            schema_version=payload["schema_version"],
            artifact_id=str(payload["artifact_id"]).strip(),
            model_version=str(payload["model_version"]).strip(),
            protocol_fingerprint=str(
                payload["protocol_fingerprint"]
            ).strip(),
            ranking_protocol_fingerprint=str(
                payload["ranking_protocol_fingerprint"]
            ).strip(),
            source_path_protocol_fingerprint=str(
                payload["source_path_protocol_fingerprint"]
            ).strip(),
            source_path_model_version=str(
                payload["source_path_model_version"]
            ).strip(),
            fit_status=str(payload["fit_status"]).strip(),
            admission_status=str(payload["admission_status"]).strip(),
            calibration_status=str(payload["calibration_status"]).strip(),
            authority=str(payload["authority"]).strip(),
            action_authority=False,
            parameters=_parameters(payload["parameters"], name="parameters"),
            fingerprint=str(payload["artifact_fingerprint"]).strip(),
        )
    except (TypeError, ValueError) as error:
        raise DOLProbabilityArtifactError(str(error)) from error
    if (
        artifact.fingerprint != expected_fingerprint
        or artifact.model_version != protocol.model_version
        or artifact.protocol_fingerprint != protocol.fingerprint
        or artifact.ranking_protocol_fingerprint
        != protocol.ranking_protocol_fingerprint
    ):
        raise DOLProbabilityArtifactError(
            "DOL probability artifact binding does not match the protocol"
        )
    return artifact


@dataclass(frozen=True)
class DOLCandidatePathSupport:
    """One existing DOL fact associated with one or more path hypotheses."""

    candidate: DOLCandidateFact
    supported_paths: tuple[PathKind, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DOLCandidateFact):
            raise TypeError("candidate must be DOLCandidateFact")
        raw = tuple(PathKind(path) for path in self.supported_paths)
        if (
            not raw
            or len(raw) != len(set(raw))
            or PathKind.RESIDUAL_UNKNOWN in raw
            or self.candidate.path not in raw
        ):
            raise ValueError("DOL candidate path support is invalid")
        object.__setattr__(
            self,
            "supported_paths",
            tuple(path for path in PATH_KINDS if path in raw),
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: DOLCandidateFact,
    ) -> "DOLCandidatePathSupport":
        return cls(candidate=candidate, supported_paths=(candidate.path,))


@dataclass(frozen=True)
class DOLPathMarginalContribution:
    outcome_id: str
    path: PathKind
    path_hypothesis_id: str
    path_probability: float
    conditional_probability: float
    marginal_probability: float
    outcome_log_weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", PathKind(self.path))
        if (
            not self.outcome_id
            or not self.path_hypothesis_id
            or not math.isfinite(float(self.path_probability))
            or not 0.0 <= float(self.path_probability) <= 1.0
            or not math.isfinite(float(self.conditional_probability))
            or not 0.0 <= float(self.conditional_probability) <= 1.0
            or not math.isfinite(float(self.marginal_probability))
            or not 0.0 <= float(self.marginal_probability) <= 1.0
            or not math.isfinite(float(self.outcome_log_weight))
            or not math.isclose(
                float(self.marginal_probability),
                float(self.path_probability)
                * float(self.conditional_probability),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("DOL path marginal contribution is invalid")


@dataclass(frozen=True)
class DOLCandidateProbability:
    rank: int
    candidate_id: str
    target_price: float
    distance_points: float
    candidate_source_kind: str
    candidate_source_ids: tuple[str, ...]
    supported_paths: tuple[PathKind, ...]
    probability: float
    path_contributions: tuple[DOLPathMarginalContribution, ...]
    base_conditional_log_weight: float
    feature_values: tuple[tuple[str, float], ...]
    hard_obstacle_ids: tuple[str, ...]
    soft_obstacle_ids: tuple[str, ...]
    excluded_obstacles: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_paths",
            tuple(PathKind(path) for path in self.supported_paths),
        )
        contribution_paths = tuple(
            contribution.path for contribution in self.path_contributions
        )
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
            or not self.supported_paths
            or tuple(
                path for path in PATH_KINDS if path in self.supported_paths
            )
            != self.supported_paths
            or len(self.supported_paths) != len(set(self.supported_paths))
            or not math.isfinite(float(self.probability))
            or not 0.0 <= float(self.probability) <= 1.0
            or not self.path_contributions
            or len(contribution_paths) != len(set(contribution_paths))
            or any(
                contribution.outcome_id != self.candidate_id
                or contribution.path not in self.supported_paths
                for contribution in self.path_contributions
            )
            or not math.isclose(
                math.fsum(
                    contribution.marginal_probability
                    for contribution in self.path_contributions
                ),
                float(self.probability),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isfinite(float(self.base_conditional_log_weight))
            or len((*self.hard_obstacle_ids, *self.soft_obstacle_ids))
            != len(set((*self.hard_obstacle_ids, *self.soft_obstacle_ids)))
        ):
            raise ValueError("DOL candidate probability is invalid")


@dataclass(frozen=True)
class DOLProbabilityResult:
    schema_version: int
    probability_id: str
    competition_set_id: str
    path_asof: pd.Timestamp
    common_expires_at: pd.Timestamp
    direction: DOLDirection
    current_price: float
    ranked_candidates: tuple[DOLCandidateProbability, ...]
    no_target_outcome: str
    no_target_probability: float
    no_target_path_contributions: tuple[DOLPathMarginalContribution, ...]
    excluded_candidates: tuple[tuple[str, str], ...]
    status: str
    calibration_status: str
    authority: str
    action_authority: bool
    protocol_version: str
    protocol_fingerprint: str
    model_version: str
    model_fingerprint: str
    model_source: str
    ranking_protocol_version: str
    ranking_protocol_fingerprint: str
    path_model_version: str
    path_protocol_fingerprint: str
    probability_interpretation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_asof", pd.Timestamp(self.path_asof))
        object.__setattr__(
            self,
            "common_expires_at",
            pd.Timestamp(self.common_expires_at),
        )
        object.__setattr__(self, "direction", DOLDirection(self.direction))
        candidate_ids = tuple(
            candidate.candidate_id for candidate in self.ranked_candidates
        )
        excluded_ids = tuple(identity for identity, _ in self.excluded_candidates)
        all_contributions = tuple(
            contribution
            for candidate in self.ranked_candidates
            for contribution in candidate.path_contributions
        ) + self.no_target_path_contributions
        path_groups: dict[PathKind, list[DOLPathMarginalContribution]] = {}
        for contribution in all_contributions:
            path_groups.setdefault(contribution.path, []).append(contribution)
        path_probabilities: list[float] = []
        conservation_invalid = False
        for contributions in path_groups.values():
            hypotheses = {item.path_hypothesis_id for item in contributions}
            probabilities = {item.path_probability for item in contributions}
            outcomes = {item.outcome_id for item in contributions}
            if (
                len(hypotheses) != 1
                or len(probabilities) != 1
                or len(outcomes) != len(contributions)
                or not math.isclose(
                    math.fsum(
                        item.conditional_probability for item in contributions
                    ),
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    math.fsum(
                        item.marginal_probability for item in contributions
                    ),
                    contributions[0].path_probability,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                conservation_invalid = True
            path_probabilities.append(contributions[0].path_probability)
        identity_payload = {
            "schema_version": self.schema_version,
            "competition_set_id": self.competition_set_id,
            "path_asof": self.path_asof.isoformat(),
            "common_expires_at": self.common_expires_at.isoformat(),
            "direction": self.direction.value,
            "current_price": self.current_price,
            "ranked_candidates": tuple(
                _candidate_payload(candidate)
                for candidate in self.ranked_candidates
            ),
            "no_target_outcome": self.no_target_outcome,
            "no_target_probability": self.no_target_probability,
            "no_target_path_contributions": tuple(
                (
                    contribution.path.value,
                    contribution.path_hypothesis_id,
                    contribution.path_probability,
                    contribution.conditional_probability,
                    contribution.marginal_probability,
                    contribution.outcome_log_weight,
                )
                for contribution in self.no_target_path_contributions
            ),
            "excluded_candidates": self.excluded_candidates,
            "status": self.status,
            "calibration_status": self.calibration_status,
            "authority": self.authority,
            "action_authority": self.action_authority,
            "protocol_version": self.protocol_version,
            "protocol_fingerprint": self.protocol_fingerprint,
            "model_version": self.model_version,
            "model_fingerprint": self.model_fingerprint,
            "model_source": self.model_source,
            "ranking_protocol_version": self.ranking_protocol_version,
            "ranking_protocol_fingerprint": (
                self.ranking_protocol_fingerprint
            ),
            "path_model_version": self.path_model_version,
            "path_protocol_fingerprint": self.path_protocol_fingerprint,
            "probability_interpretation": self.probability_interpretation,
        }
        expected_probability_id = (
            f"dol-probability:{_canonical_hash(identity_payload)[:32]}"
        )
        expected_source_semantics = {
            "fitted_admitted_artifact": (
                "fitted_admitted",
                _FITTED_PROBABILITY_INTERPRETATION,
            ),
        }
        if (
            self.schema_version != DOL_PROBABILITY_SCHEMA_VERSION
            or self.probability_id != expected_probability_id
            or not self.competition_set_id
            or pd.isna(self.path_asof)
            or self.path_asof.tzinfo is None
            or pd.isna(self.common_expires_at)
            or self.common_expires_at.tzinfo is None
            or self.path_asof >= self.common_expires_at
            or not math.isfinite(float(self.current_price))
            or self.current_price <= 0.0
            or len(candidate_ids) != len(set(candidate_ids))
            or tuple(candidate.rank for candidate in self.ranked_candidates)
            != tuple(range(1, len(self.ranked_candidates) + 1))
            or len(excluded_ids) != len(set(excluded_ids))
            or not set(candidate_ids).isdisjoint(excluded_ids)
            or self.no_target_outcome != NO_TARGET_BEFORE_HORIZON
            or not math.isfinite(float(self.no_target_probability))
            or not 0.0 <= float(self.no_target_probability) <= 1.0
            or any(
                contribution.outcome_id != NO_TARGET_BEFORE_HORIZON
                for contribution in self.no_target_path_contributions
            )
            or not math.isclose(
                math.fsum(
                    contribution.marginal_probability
                    for contribution in self.no_target_path_contributions
                ),
                float(self.no_target_probability),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                math.fsum(
                    [
                        *(candidate.probability for candidate in self.ranked_candidates),
                        float(self.no_target_probability),
                    ]
                ),
                1.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not path_groups
            or conservation_invalid
            or not math.isclose(
                math.fsum(path_probabilities),
                1.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or self.status != "development_unvalidated"
            or self.calibration_status != "fitted_admitted"
            or self.authority != "shadow_only"
            or self.action_authority is not False
            or not self.protocol_version
            or not _is_sha256(self.protocol_fingerprint)
            or not self.model_version
            or not _is_sha256(self.model_fingerprint)
            or self.model_source != "fitted_admitted_artifact"
            or expected_source_semantics.get(self.model_source)
            != (
                self.calibration_status,
                self.probability_interpretation,
            )
            or not self.ranking_protocol_version
            or not _is_sha256(self.ranking_protocol_fingerprint)
            or not self.path_model_version
            or not _is_sha256(self.path_protocol_fingerprint)
        ):
            raise ValueError("DOL probability result is invalid")


def _normalize_candidate_supports(
    values: Sequence[DOLCandidateFact | DOLCandidatePathSupport],
) -> tuple[DOLCandidatePathSupport, ...]:
    normalized: list[DOLCandidatePathSupport] = []
    for value in tuple(values):
        if isinstance(value, DOLCandidatePathSupport):
            normalized.append(value)
        elif isinstance(value, DOLCandidateFact):
            normalized.append(DOLCandidatePathSupport.from_candidate(value))
        else:
            raise TypeError(
                "external_draw_candidates must contain DOLCandidateFact or "
                "DOLCandidatePathSupport"
            )
    candidate_ids = tuple(item.candidate.candidate_id for item in normalized)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("external draw candidate identities must be unique")
    return tuple(sorted(normalized, key=lambda item: item.candidate.candidate_id))


def _softmax(log_weights: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in log_weights)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("DOL conditional log weights are invalid")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    denominator = math.fsum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("DOL conditional probability normalizer is invalid")
    return tuple(value / denominator for value in exponentials)


def _resolved_fact_signature(item: RankedDOLCandidate) -> tuple[Any, ...]:
    return (
        item.target_price,
        item.distance_points,
        item.candidate_source_kind,
        item.candidate_source_ids,
        item.conditional_log_weight,
        item.feature_values,
        item.hard_obstacle_ids,
        item.soft_obstacle_ids,
        item.excluded_obstacles,
    )


def _candidate_payload(candidate: DOLCandidateProbability) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "candidate_id": candidate.candidate_id,
        "target_price": candidate.target_price,
        "candidate_source_ids": candidate.candidate_source_ids,
        "supported_paths": tuple(path.value for path in candidate.supported_paths),
        "probability": candidate.probability,
        "path_contributions": tuple(
            (
                contribution.path.value,
                contribution.path_hypothesis_id,
                contribution.path_probability,
                contribution.conditional_probability,
                contribution.marginal_probability,
                contribution.outcome_log_weight,
            )
            for contribution in candidate.path_contributions
        ),
        "base_conditional_log_weight": candidate.base_conditional_log_weight,
        "feature_values": candidate.feature_values,
        "hard_obstacle_ids": candidate.hard_obstacle_ids,
        "soft_obstacle_ids": candidate.soft_obstacle_ids,
        "excluded_obstacles": candidate.excluded_obstacles,
    }


def marginalize_dol_probabilities(
    protocol: DOLProbabilityProtocol,
    *,
    ranking_protocol: DOLRankingProtocol,
    direction: DOLDirection,
    current_price: float,
    external_draw_candidates: Sequence[
        DOLCandidateFact | DOLCandidatePathSupport
    ],
    obstruction_view: DOLObstructionViewFact,
    path_state: PathCompetitionSetState,
    model_artifact: DOLProbabilityModelArtifact | None = None,
) -> DOLProbabilityResult:
    """Compute a deterministic path-marginal DOL distribution.

    Inputs are already-resolved market facts.  No detector or market-context
    mutation runs here.  A probability result exists only for an exact
    fitted/admitted artifact; uncalibrated candidate ordering belongs to the
    separate DOL ranking boundary.
    """

    if not isinstance(protocol, DOLProbabilityProtocol):
        raise TypeError("protocol must be DOLProbabilityProtocol")
    if not isinstance(ranking_protocol, DOLRankingProtocol):
        raise TypeError("ranking_protocol must be DOLRankingProtocol")
    if not isinstance(path_state, PathCompetitionSetState):
        raise TypeError("path_state must be PathCompetitionSetState")
    if not isinstance(obstruction_view, DOLObstructionViewFact):
        raise TypeError("obstruction_view must be DOLObstructionViewFact")
    direction = DOLDirection(direction)
    price = _finite_number(current_price, name="current_price")
    if price <= 0.0:
        raise ValueError("current_price must be positive")
    if obstruction_view.direction is not direction:
        raise ValueError("obstruction view direction does not match direction")
    if ranking_protocol.fingerprint != protocol.ranking_protocol_fingerprint:
        raise DOLProbabilityProtocolError(
            "DOL ranking protocol fingerprint does not match probability protocol"
        )
    if (
        path_state.status is not PathStatus.ACTIVE
        or path_state.protocol_status != "development_unvalidated"
        or path_state.authority != "shadow_only"
    ):
        raise ValueError(
            "DOL probability requires one active shadow path competition set"
        )

    if model_artifact is None:
        raise DOLProbabilityArtifactError(
            "DOL probability requires a fitted/admitted model artifact"
        )
    if not isinstance(model_artifact, DOLProbabilityModelArtifact):
        raise TypeError("model_artifact must be DOLProbabilityModelArtifact")
    if (
        model_artifact.model_version != protocol.model_version
        or model_artifact.protocol_fingerprint != protocol.fingerprint
        or model_artifact.ranking_protocol_fingerprint
        != ranking_protocol.fingerprint
        or model_artifact.source_path_protocol_fingerprint
        != path_state.protocol_fingerprint
        or model_artifact.source_path_model_version
        != path_state.model_version
        or model_artifact.fit_status != "fitted"
        or model_artifact.admission_status != "admitted"
        or model_artifact.authority != "shadow_only"
        or model_artifact.action_authority is not False
    ):
        raise DOLProbabilityArtifactError(
            "DOL probability artifact binding is invalid"
        )
    parameters = model_artifact.parameters
    model_fingerprint = model_artifact.fingerprint
    model_source = "fitted_admitted_artifact"
    calibration_status = model_artifact.calibration_status

    supports = _normalize_candidate_supports(external_draw_candidates)
    by_id = {item.candidate.candidate_id: item for item in supports}
    resolved: dict[str, RankedDOLCandidate] = {}
    exclusions_by_candidate: dict[str, dict[PathKind, str]] = {
        candidate_id: {} for candidate_id in by_id
    }
    candidate_contributions: dict[
        str, list[DOLPathMarginalContribution]
    ] = {candidate_id: [] for candidate_id in by_id}
    no_target_contributions: list[DOLPathMarginalContribution] = []

    active_members = tuple(
        member
        for member in path_state.members
        if member.status is PathStatus.ACTIVE
    )
    for member in active_members:
        path = member.path
        path_supports = tuple(
            item
            for item in supports
            if path in item.supported_paths
        )
        if member.probability <= 0.0 or path is PathKind.RESIDUAL_UNKNOWN:
            reason = (
                "path_hypothesis_zero_probability"
                if member.probability <= 0.0
                else "residual_unknown_reserved_for_no_target"
            )
            for item in path_supports:
                exclusions_by_candidate[item.candidate.candidate_id][path] = reason
            no_target_contributions.append(
                DOLPathMarginalContribution(
                    outcome_id=NO_TARGET_BEFORE_HORIZON,
                    path=path,
                    path_hypothesis_id=member.hypothesis_id,
                    path_probability=float(member.probability),
                    conditional_probability=1.0,
                    marginal_probability=float(member.probability),
                    outcome_log_weight=parameters.no_target_weight(path),
                )
            )
            continue

        ranking = rank_dol_candidates(
            ranking_protocol,
            direction=direction,
            current_price=price,
            external_draw_candidates=tuple(
                replace(item.candidate, path=path) for item in path_supports
            ),
            obstruction_view=obstruction_view,
            path_state=path_state,
        )
        eligible = {
            item.candidate_id: item for item in ranking.ranked_candidates
        }
        for candidate_id, reason in ranking.excluded_candidates:
            exclusions_by_candidate[candidate_id][path] = reason

        candidate_ids = tuple(sorted(eligible))
        candidate_log_weights: list[float] = []
        for candidate_id in candidate_ids:
            item = eligible[candidate_id]
            previous = resolved.get(candidate_id)
            if (
                previous is not None
                and _resolved_fact_signature(previous)
                != _resolved_fact_signature(item)
            ):
                raise ValueError(
                    "DOL fact or obstacle resolution changed across path support"
                )
            resolved[candidate_id] = item
            candidate_log_weight = (
                parameters.candidate_logit_scale
                * item.conditional_log_weight
                + parameters.candidate_adjustment(path)
            )
            if not math.isfinite(candidate_log_weight):
                raise ValueError("DOL candidate conditional log weight is not finite")
            candidate_log_weights.append(candidate_log_weight)

        no_target_log_weight = parameters.no_target_weight(path)
        probabilities = _softmax(
            (*candidate_log_weights, no_target_log_weight)
        )
        for candidate_id, log_weight, conditional_probability in zip(
            candidate_ids,
            candidate_log_weights,
            probabilities[:-1],
        ):
            marginal_probability = (
                float(member.probability) * conditional_probability
            )
            candidate_contributions[candidate_id].append(
                DOLPathMarginalContribution(
                    outcome_id=candidate_id,
                    path=path,
                    path_hypothesis_id=member.hypothesis_id,
                    path_probability=float(member.probability),
                    conditional_probability=conditional_probability,
                    marginal_probability=marginal_probability,
                    outcome_log_weight=log_weight,
                )
            )
        no_target_conditional = probabilities[-1]
        no_target_contributions.append(
            DOLPathMarginalContribution(
                outcome_id=NO_TARGET_BEFORE_HORIZON,
                path=path,
                path_hypothesis_id=member.hypothesis_id,
                path_probability=float(member.probability),
                conditional_probability=no_target_conditional,
                marginal_probability=(
                    float(member.probability) * no_target_conditional
                ),
                outcome_log_weight=no_target_log_weight,
            )
        )

    for item in supports:
        candidate_id = item.candidate.candidate_id
        for path in item.supported_paths:
            member = path_state.member(path)
            if member.status is not PathStatus.ACTIVE:
                exclusions_by_candidate[candidate_id][path] = (
                    "path_hypothesis_inactive"
                )

    unranked_candidates: list[DOLCandidateProbability] = []
    excluded_candidates: list[tuple[str, str]] = []
    for candidate_id, support in sorted(by_id.items()):
        contributions = tuple(
            sorted(
                candidate_contributions[candidate_id],
                key=lambda item: PATH_KINDS.index(item.path),
            )
        )
        if not contributions:
            path_reasons = exclusions_by_candidate[candidate_id]
            if not path_reasons:
                reason = "no_active_supported_path"
            else:
                unique_reasons = set(path_reasons.values())
                if len(unique_reasons) == 1:
                    reason = next(iter(unique_reasons))
                else:
                    reason = ";".join(
                        f"{path.value}:{path_reasons[path]}"
                        for path in PATH_KINDS
                        if path in path_reasons
                    )
            excluded_candidates.append((candidate_id, reason))
            continue
        fact = resolved[candidate_id]
        unranked_candidates.append(
            DOLCandidateProbability(
                rank=1,
                candidate_id=candidate_id,
                target_price=float(fact.target_price),
                distance_points=float(fact.distance_points),
                candidate_source_kind=fact.candidate_source_kind,
                candidate_source_ids=fact.candidate_source_ids,
                supported_paths=support.supported_paths,
                probability=math.fsum(
                    contribution.marginal_probability
                    for contribution in contributions
                ),
                path_contributions=contributions,
                base_conditional_log_weight=float(
                    fact.conditional_log_weight
                ),
                feature_values=fact.feature_values,
                hard_obstacle_ids=fact.hard_obstacle_ids,
                soft_obstacle_ids=fact.soft_obstacle_ids,
                excluded_obstacles=fact.excluded_obstacles,
            )
        )

    ordered = tuple(
        sorted(
            unranked_candidates,
            key=lambda item: (
                -item.probability,
                item.distance_points,
                item.candidate_id,
            ),
        )
    )
    ranked_candidates = tuple(
        replace(candidate, rank=index)
        for index, candidate in enumerate(ordered, start=1)
    )
    no_target_path_contributions = tuple(
        sorted(
            no_target_contributions,
            key=lambda item: PATH_KINDS.index(item.path),
        )
    )
    no_target_probability = math.fsum(
        contribution.marginal_probability
        for contribution in no_target_path_contributions
    )
    excluded = tuple(sorted(excluded_candidates))
    identity_payload = {
        "schema_version": DOL_PROBABILITY_SCHEMA_VERSION,
        "competition_set_id": path_state.competition_set_id,
        "path_asof": path_state.asof.isoformat(),
        "common_expires_at": path_state.common_expires_at.isoformat(),
        "direction": direction.value,
        "current_price": price,
        "ranked_candidates": tuple(
            _candidate_payload(candidate) for candidate in ranked_candidates
        ),
        "no_target_outcome": NO_TARGET_BEFORE_HORIZON,
        "no_target_probability": no_target_probability,
        "no_target_path_contributions": tuple(
            (
                contribution.path.value,
                contribution.path_hypothesis_id,
                contribution.path_probability,
                contribution.conditional_probability,
                contribution.marginal_probability,
                contribution.outcome_log_weight,
            )
            for contribution in no_target_path_contributions
        ),
        "excluded_candidates": excluded,
        "status": protocol.status,
        "calibration_status": calibration_status,
        "authority": protocol.authority,
        "action_authority": False,
        "protocol_version": protocol.protocol_version,
        "protocol_fingerprint": protocol.fingerprint,
        "model_version": protocol.model_version,
        "model_fingerprint": model_fingerprint,
        "model_source": model_source,
        "ranking_protocol_version": ranking_protocol.protocol_version,
        "ranking_protocol_fingerprint": ranking_protocol.fingerprint,
        "path_model_version": path_state.model_version,
        "path_protocol_fingerprint": path_state.protocol_fingerprint,
        "probability_interpretation": _FITTED_PROBABILITY_INTERPRETATION,
    }
    return DOLProbabilityResult(
        schema_version=DOL_PROBABILITY_SCHEMA_VERSION,
        probability_id=f"dol-probability:{_canonical_hash(identity_payload)[:32]}",
        competition_set_id=path_state.competition_set_id,
        path_asof=path_state.asof,
        common_expires_at=path_state.common_expires_at,
        direction=direction,
        current_price=price,
        ranked_candidates=ranked_candidates,
        no_target_outcome=NO_TARGET_BEFORE_HORIZON,
        no_target_probability=no_target_probability,
        no_target_path_contributions=no_target_path_contributions,
        excluded_candidates=excluded,
        status=protocol.status,
        calibration_status=calibration_status,
        authority=protocol.authority,
        action_authority=False,
        protocol_version=protocol.protocol_version,
        protocol_fingerprint=protocol.fingerprint,
        model_version=protocol.model_version,
        model_fingerprint=model_fingerprint,
        model_source=model_source,
        ranking_protocol_version=ranking_protocol.protocol_version,
        ranking_protocol_fingerprint=ranking_protocol.fingerprint,
        path_model_version=path_state.model_version,
        path_protocol_fingerprint=path_state.protocol_fingerprint,
        probability_interpretation=_FITTED_PROBABILITY_INTERPRETATION,
    )


__all__ = [
    "DOLCandidatePathSupport",
    "DOLCandidateProbability",
    "DOLConditionalModelParameters",
    "DOLPathMarginalContribution",
    "DOLProbabilityArtifactError",
    "DOLProbabilityModelArtifact",
    "DOLProbabilityProtocol",
    "DOLProbabilityProtocolError",
    "DOLProbabilityResult",
    "NO_TARGET_BEFORE_HORIZON",
    "load_dol_probability_model_artifact",
    "load_dol_probability_protocol",
    "marginalize_dol_probabilities",
]
