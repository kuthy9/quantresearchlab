"""Fail-closed, shadow-only signal policy for the Phase 7 Brain boundary.

The existing path competition and DOL ranking surfaces intentionally publish
development diagnostics.  This module does not reinterpret those diagnostics
as posteriors.  A signal can be assessed only when three separately admitted,
time-bounded artifacts are supplied:

* a path-likelihood calibration for the complete competition set;
* a DOL calibration for the candidate cohort; and
* a model for the distinct ``target_before_invalidation`` estimand.

Typed playbook candidates contribute their immutable identity, plan, lifecycle
and setup family only.  In particular, ``HypothesisBelief.probability`` and its
other playbook quality scalars are never consumed here.  Every result remains
``shadow_only`` and carries no Decision, Risk, or Execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .dol_probability import (
    DOLCandidateProbability,
    DOLProbabilityModelArtifact,
    DOLProbabilityResult,
    NO_TARGET_BEFORE_HORIZON,
)
from .dol_ranking import DOLDirection, DOLRankingResult, RankedDOLCandidate
from .model import (
    Direction,
    EntryEpisodeState,
    HypothesisBelief,
    Playbook,
    PlaybookPhase,
    TradePlan,
)
from .path_belief import (
    PATH_KINDS,
    PathBeliefProtocol,
    PathCompetitionSetState,
    PathHypothesisState,
    PathKind,
    PathStatus,
)


SIGNAL_POLICY_SCHEMA_VERSION = 2
PROBABILITY_ADMISSION_SCHEMA_VERSION = 1
SIGNAL_ASSESSMENT_SCHEMA_VERSION = 2
TARGET_BEFORE_INVALIDATION_ESTIMAND = "target_before_invalidation"
SHADOW_AUTHORITY = "shadow_only"
ADMITTED_SHADOW_STATUS = "admitted_shadow_only"


class SetupFamily(str, Enum):
    """The setup families admitted to the initial signal-policy surface."""

    DFP = "dfp"
    LSR = "lsr"


class SignalDisposition(str, Enum):
    ELIGIBLE_SHADOW = "eligible_shadow_only"
    REJECTED = "rejected"


class SignalRejection(str, Enum):
    SETUP_FAMILY_PARKED = "setup_family_parked"
    SETUP_NOT_EXECUTABLE = "setup_not_executable"
    SETUP_IDENTITY_MISMATCH = "setup_identity_mismatch"
    TRADE_PLAN_INCOMPLETE = "trade_plan_incomplete"
    PLAN_TARGET_MISMATCH = "plan_target_mismatch"
    PATH_SET_INACTIVE = "path_set_inactive"
    PATH_HYPOTHESIS_INACTIVE = "path_hypothesis_inactive"
    PATH_IDENTITY_MISMATCH = "path_identity_mismatch"
    DOL_CANDIDATE_UNAVAILABLE = "dol_candidate_unavailable"
    DOL_IDENTITY_MISMATCH = "dol_identity_mismatch"
    DOL_PROBABILITY_NOT_FITTED = "dol_probability_not_fitted"
    MISSING_DOL_PROBABILITY_MODEL_ARTIFACT = (
        "missing_dol_probability_model_artifact"
    )
    INPUT_FROM_FUTURE = "input_from_future"
    INPUT_STALE = "input_stale"
    SIGNAL_EXPIRED = "signal_expired"
    SIGNAL_HALF_LIFE_ELAPSED = "signal_half_life_elapsed"
    REAL_COMPLETED_BAR_AGE_UNAVAILABLE = "real_completed_bar_age_unavailable"
    MISSING_PATH_LIKELIHOOD_ARTIFACT = "missing_path_likelihood_artifact"
    MISSING_DOL_CALIBRATION_ARTIFACT = "missing_dol_calibration_artifact"
    MISSING_OUTCOME_MODEL_ARTIFACT = "missing_outcome_model_artifact"
    ARTIFACT_NOT_ADMITTED = "artifact_not_admitted"
    ARTIFACT_OUTSIDE_VALIDITY = "artifact_outside_validity"
    ARTIFACT_IDENTITY_MISMATCH = "artifact_identity_mismatch"
    COVERAGE_INCOMPLETE = "coverage_incomplete"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    COST_UNAVAILABLE = "cost_unavailable"
    COST_EXCEEDS_LIMIT = "cost_exceeds_limit"
    PROBABILITY_BELOW_MINIMUM = "probability_below_minimum"
    EDGE_BELOW_MINIMUM = "edge_below_minimum"


class SignalPolicyProtocolError(ValueError):
    """Raised when the frozen signal-policy protocol cannot be loaded."""


class SignalPolicyArtifactError(ValueError):
    """Raised when one externally pinned Phase 7 artifact is unusable."""


@dataclass(frozen=True)
class SignalArtifactPins:
    """External deployment pins for one exact three-artifact admission set.

    Artifact objects self-identify, but cannot admit themselves.  Eligibility
    additionally requires this separately supplied pin set and the exact
    admitted path protocol that produced the competition state.
    """

    signal_policy_fingerprint: str
    path_protocol_fingerprint: str
    path_likelihood_artifact_id: str
    dol_calibration_artifact_id: str
    outcome_model_artifact_id: str
    dol_probability_model_fingerprint: str | None = None
    admission_id: str = field(init=False)

    def __post_init__(self) -> None:
        fingerprints = (
            self.signal_policy_fingerprint,
            self.path_protocol_fingerprint,
        )
        identities = (
            self.path_likelihood_artifact_id,
            self.dol_calibration_artifact_id,
            self.outcome_model_artifact_id,
        )
        if (
            any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in fingerprints
            )
            or any(not isinstance(value, str) or not value for value in identities)
            or (
                self.dol_probability_model_fingerprint is not None
                and (
                    not isinstance(self.dol_probability_model_fingerprint, str)
                    or len(self.dol_probability_model_fingerprint) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in self.dol_probability_model_fingerprint
                    )
                )
            )
        ):
            raise ValueError("signal artifact pins are invalid")
        payload = {
            "signal_policy_fingerprint": self.signal_policy_fingerprint,
            "path_protocol_fingerprint": self.path_protocol_fingerprint,
            "path_likelihood_artifact_id": self.path_likelihood_artifact_id,
            "dol_calibration_artifact_id": self.dol_calibration_artifact_id,
            "outcome_model_artifact_id": self.outcome_model_artifact_id,
            "dol_probability_model_fingerprint": (
                self.dol_probability_model_fingerprint
            ),
        }
        object.__setattr__(
            self,
            "admission_id",
            f"signal-artifact-admission:{_canonical_hash(payload)[:32]}",
        )


class CancelConditionKind(str, Enum):
    SIGNAL_EXPIRY_REACHED = "signal_expiry_reached"
    SIGNAL_HALF_LIFE_ELAPSED = "signal_half_life_elapsed"
    PLAN_DEADLINE_REACHED = "plan_deadline_reached"
    ARTIFACT_EXPIRY_REACHED = "artifact_expiry_reached"
    PATH_SET_NO_LONGER_ACTIVE = "path_set_no_longer_active"
    PATH_HYPOTHESIS_NO_LONGER_ACTIVE = "path_hypothesis_no_longer_active"
    DOL_CANDIDATE_UNAVAILABLE = "dol_candidate_unavailable"
    EPISODE_NO_LONGER_EXECUTABLE = "episode_no_longer_executable"
    STRUCTURAL_INVALIDATION_REACHED = "structural_invalidation_reached"
    COVERAGE_BELOW_MINIMUM = "coverage_below_minimum"
    OUT_OF_DISTRIBUTION_DETECTED = "out_of_distribution_detected"
    COST_EXCEEDS_LIMIT = "cost_exceeds_limit"
    DELIVERY_PROBABILITY_BELOW_MINIMUM = "delivery_probability_below_minimum"
    NET_EDGE_BELOW_MINIMUM = "net_edge_below_minimum"


def _aware_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return result


def _positive_timedelta(value: Any, *, name: str) -> pd.Timedelta:
    result = pd.Timedelta(value)
    if pd.isna(result) or result <= pd.Timedelta(0):
        raise ValueError(f"{name} must be a positive duration")
    return result


def _finite_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a probability")
    return result


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _exact_ids(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    raw = tuple(values)
    if (
        (not allow_empty and not raw)
        or len(raw) != len(set(raw))
        or any(not isinstance(value, str) or not value for value in raw)
    ):
        raise ValueError(f"{name} must contain unique non-empty identities")
    return tuple(sorted(raw))


def _logit(probability: float) -> float:
    bounded = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return math.log(bounded / (1.0 - bounded))


def _logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _playbook_setup_family(playbook: Playbook) -> SetupFamily | None:
    playbook = Playbook(playbook)
    if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK:
        return SetupFamily.DFP
    if playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL:
        return SetupFamily.LSR
    # FAVR remains deliberately parked until it has its own admitted model.
    return None


def _artifact_clock_payload(
    *,
    trained_through: pd.Timestamp,
    valid_from: pd.Timestamp,
    expires_at: pd.Timestamp,
) -> Mapping[str, str]:
    return {
        "trained_through": trained_through.isoformat(),
        "valid_from": valid_from.isoformat(),
        "expires_at": expires_at.isoformat(),
    }


def _validate_admission_clocks(
    instance: Any,
    *,
    name: str,
) -> None:
    for field_name in ("trained_through", "valid_from", "expires_at"):
        object.__setattr__(
            instance,
            field_name,
            _aware_timestamp(
                getattr(instance, field_name),
                name=f"{name}.{field_name}",
            ),
        )
    if not (instance.trained_through < instance.valid_from < instance.expires_at):
        raise ValueError(f"{name} requires frozen train-before-validity clocks")


def _validate_shadow_admission(instance: Any, *, name: str) -> None:
    identities = (
        instance.protocol_id,
        instance.model_id,
        instance.model_version,
        instance.calibration_id,
        instance.source_dataset_id,
        instance.coverage_id,
    )
    if (
        any(not isinstance(value, str) or not value for value in identities)
        or instance.status != ADMITTED_SHADOW_STATUS
        or instance.authority != SHADOW_AUTHORITY
        or type(instance.action_authority_ready) is not bool
        or instance.action_authority_ready
    ):
        raise ValueError(f"{name} must be admitted shadow-only")


@dataclass(frozen=True)
class AdmittedPathLikelihoodArtifact:
    """Temperature/bias calibration of one complete path competition set."""

    protocol_id: str
    model_id: str
    model_version: str
    calibration_id: str
    source_dataset_id: str
    coverage_id: str
    source_path_protocol_fingerprint: str
    source_path_model_version: str
    trained_through: pd.Timestamp
    valid_from: pd.Timestamp
    expires_at: pd.Timestamp
    temperature: float = 1.0
    path_log_biases: tuple[tuple[PathKind, float], ...] = tuple(
        (path, 0.0) for path in PATH_KINDS
    )
    schema_version: int = PROBABILITY_ADMISSION_SCHEMA_VERSION
    status: str = ADMITTED_SHADOW_STATUS
    authority: str = SHADOW_AUTHORITY
    action_authority_ready: bool = False
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_admission_clocks(self, name="path likelihood artifact")
        _validate_shadow_admission(self, name="path likelihood artifact")
        raw_biases = tuple(
            (PathKind(path), float(value)) for path, value in self.path_log_biases
        )
        object.__setattr__(self, "path_log_biases", raw_biases)
        if (
            self.schema_version != PROBABILITY_ADMISSION_SCHEMA_VERSION
            or len(self.source_path_protocol_fingerprint) != 64
            or not self.source_path_model_version
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0.0
            or tuple(path for path, _ in raw_biases) != PATH_KINDS
            or any(not math.isfinite(value) for _, value in raw_biases)
        ):
            raise ValueError("path likelihood artifact is invalid")
        payload = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "calibration_id": self.calibration_id,
            "source_dataset_id": self.source_dataset_id,
            "coverage_id": self.coverage_id,
            "source_path_protocol_fingerprint": (self.source_path_protocol_fingerprint),
            "source_path_model_version": self.source_path_model_version,
            **_artifact_clock_payload(
                trained_through=self.trained_through,
                valid_from=self.valid_from,
                expires_at=self.expires_at,
            ),
            "temperature": self.temperature,
            "path_log_biases": [(path.value, value) for path, value in raw_biases],
            "status": self.status,
            "authority": self.authority,
            "action_authority_ready": self.action_authority_ready,
        }
        object.__setattr__(
            self,
            "artifact_id",
            f"path-likelihood-artifact:{_canonical_hash(payload)[:32]}",
        )

    def current_at(self, asof: pd.Timestamp) -> bool:
        clock = _aware_timestamp(asof, name="path artifact asof")
        return self.valid_from <= clock < self.expires_at

    def admits(self, state: PathCompetitionSetState) -> bool:
        return bool(
            isinstance(state, PathCompetitionSetState)
            and state.protocol_fingerprint == self.source_path_protocol_fingerprint
            and state.model_version == self.source_path_model_version
            and "no_likelihood_artifact" not in state.model_version
        )

    def probabilities(
        self,
        state: PathCompetitionSetState,
    ) -> Mapping[PathKind, float]:
        if not self.admits(state) or state.status is not PathStatus.ACTIVE:
            raise ValueError("path artifact does not admit this active state")
        biases = dict(self.path_log_biases)
        active = tuple(
            member
            for member in state.members
            if member.status is PathStatus.ACTIVE and member.probability > 0.0
        )
        if not active:
            raise ValueError("path calibration requires active probability mass")
        scores = tuple(
            math.log(float(member.probability)) / self.temperature + biases[member.path]
            for member in active
        )
        maximum = max(scores)
        weights = tuple(math.exp(score - maximum) for score in scores)
        denominator = math.fsum(weights)
        admitted = {
            member.path: weight / denominator
            for member, weight in zip(active, weights, strict=True)
        }
        return {path: admitted.get(path, 0.0) for path in PATH_KINDS}


@dataclass(frozen=True)
class AdmittedDOLCalibrationArtifact:
    """Cohort-preserving calibration of one exact DOL distribution.

    ``source_dol_model_fingerprint`` is mandatory for the Phase 7
    path-marginal probability surface.  It remains optional only for the
    legacy diagnostic-ranking compatibility path, whose historical fixtures
    predate a separately identified fitted DOL model.
    """

    protocol_id: str
    model_id: str
    model_version: str
    calibration_id: str
    source_dataset_id: str
    coverage_id: str
    source_dol_protocol_fingerprint: str
    source_dol_model_version: str
    source_path_protocol_fingerprint: str
    source_path_model_version: str
    trained_through: pd.Timestamp
    valid_from: pd.Timestamp
    expires_at: pd.Timestamp
    temperature: float = 1.0
    source_dol_model_fingerprint: str | None = None
    schema_version: int = PROBABILITY_ADMISSION_SCHEMA_VERSION
    status: str = ADMITTED_SHADOW_STATUS
    authority: str = SHADOW_AUTHORITY
    action_authority_ready: bool = False
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_admission_clocks(self, name="DOL calibration artifact")
        _validate_shadow_admission(self, name="DOL calibration artifact")
        fingerprints = (
            self.source_dol_protocol_fingerprint,
            self.source_path_protocol_fingerprint,
        )
        if (
            self.schema_version != PROBABILITY_ADMISSION_SCHEMA_VERSION
            or any(len(value) != 64 for value in fingerprints)
            or not self.source_dol_model_version
            or not self.source_path_model_version
            or (
                self.source_dol_model_fingerprint is not None
                and len(self.source_dol_model_fingerprint) != 64
            )
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0.0
        ):
            raise ValueError("DOL calibration artifact is invalid")
        payload = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "calibration_id": self.calibration_id,
            "source_dataset_id": self.source_dataset_id,
            "coverage_id": self.coverage_id,
            "source_dol_protocol_fingerprint": (self.source_dol_protocol_fingerprint),
            "source_dol_model_version": self.source_dol_model_version,
            "source_dol_model_fingerprint": (
                self.source_dol_model_fingerprint
            ),
            "source_path_protocol_fingerprint": (self.source_path_protocol_fingerprint),
            "source_path_model_version": self.source_path_model_version,
            **_artifact_clock_payload(
                trained_through=self.trained_through,
                valid_from=self.valid_from,
                expires_at=self.expires_at,
            ),
            "temperature": self.temperature,
            "status": self.status,
            "authority": self.authority,
            "action_authority_ready": self.action_authority_ready,
        }
        object.__setattr__(
            self,
            "artifact_id",
            f"dol-calibration-artifact:{_canonical_hash(payload)[:32]}",
        )

    def current_at(self, asof: pd.Timestamp) -> bool:
        clock = _aware_timestamp(asof, name="DOL artifact asof")
        return self.valid_from <= clock < self.expires_at

    def admits(
        self,
        ranking: DOLRankingResult | DOLProbabilityResult,
    ) -> bool:
        common = bool(
            isinstance(ranking, (DOLRankingResult, DOLProbabilityResult))
            and ranking.protocol_fingerprint
            == self.source_dol_protocol_fingerprint
            and ranking.model_version == self.source_dol_model_version
            and ranking.path_protocol_fingerprint
            == self.source_path_protocol_fingerprint
            and ranking.path_model_version == self.source_path_model_version
        )
        if not common:
            return False
        if isinstance(ranking, DOLProbabilityResult):
            return bool(
                ranking.model_source == "fitted_admitted_artifact"
                and ranking.calibration_status == "fitted_admitted"
                and self.source_dol_model_fingerprint is not None
                and ranking.model_fingerprint
                == self.source_dol_model_fingerprint
            )
        return self.source_dol_model_fingerprint is None

    def probabilities(
        self,
        ranking: DOLRankingResult | DOLProbabilityResult,
    ) -> Mapping[str, float]:
        if not self.admits(ranking) or not ranking.ranked_candidates:
            raise ValueError("DOL artifact does not admit this candidate cohort")
        if isinstance(ranking, DOLProbabilityResult):
            # Calibrate the complete mutually-exclusive outcome distribution.
            # The explicit no-target mass stays in the denominator; candidate
            # probabilities therefore cannot silently renormalize to one.
            outcomes = (
                *(float(item.probability) for item in ranking.ranked_candidates),
                float(ranking.no_target_probability),
            )
            scores = tuple(
                math.log(max(value, 1e-300)) / self.temperature
                for value in outcomes
            )
            maximum = max(scores)
            weights = tuple(math.exp(score - maximum) for score in scores)
            denominator = math.fsum(weights)
            return {
                item.candidate_id: weight / denominator
                for item, weight in zip(
                    ranking.ranked_candidates,
                    weights[:-1],
                    strict=True,
                )
            }
        scores = tuple(
            math.log(float(item.normalized_diagnostic_weight)) / self.temperature
            for item in ranking.ranked_candidates
        )
        maximum = max(scores)
        weights = tuple(math.exp(score - maximum) for score in scores)
        denominator = math.fsum(weights)
        return {
            item.candidate_id: weight / denominator
            for item, weight in zip(
                ranking.ranked_candidates,
                weights,
                strict=True,
            )
        }


@dataclass(frozen=True)
class SetupDeliveryModel:
    """One preregistered setup-family cell for the delivery estimand."""

    setup_family: SetupFamily
    intercept: float
    path_logit_coefficient: float
    dol_logit_coefficient: float
    half_life_real_completed_bars: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "setup_family",
            SetupFamily(self.setup_family),
        )
        coefficients = (
            self.intercept,
            self.path_logit_coefficient,
            self.dol_logit_coefficient,
        )
        if (
            any(not math.isfinite(float(value)) for value in coefficients)
            or type(self.half_life_real_completed_bars) is not int
            or self.half_life_real_completed_bars <= 0
        ):
            raise ValueError("setup delivery model is invalid")

    def estimate(self, *, p_path: float, p_dol: float) -> float:
        p_path = _finite_probability(p_path, name="p_path")
        p_dol = _finite_probability(p_dol, name="p_dol")
        linear = (
            float(self.intercept)
            + float(self.path_logit_coefficient) * _logit(p_path)
            + float(self.dol_logit_coefficient) * _logit(p_dol)
        )
        return _logistic(linear)


@dataclass(frozen=True)
class TargetBeforeInvalidationArtifact:
    """Admitted model for a signal outcome distinct from path/DOL beliefs."""

    protocol_id: str
    model_id: str
    model_version: str
    calibration_id: str
    source_dataset_id: str
    coverage_id: str
    path_likelihood_artifact_id: str
    dol_calibration_artifact_id: str
    signal_policy_fingerprint: str
    trained_through: pd.Timestamp
    valid_from: pd.Timestamp
    expires_at: pd.Timestamp
    setup_models: tuple[SetupDeliveryModel, ...]
    minimum_supported_coverage: float
    estimand: str = TARGET_BEFORE_INVALIDATION_ESTIMAND
    schema_version: int = PROBABILITY_ADMISSION_SCHEMA_VERSION
    status: str = ADMITTED_SHADOW_STATUS
    authority: str = SHADOW_AUTHORITY
    action_authority_ready: bool = False
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_admission_clocks(self, name="outcome model artifact")
        _validate_shadow_admission(self, name="outcome model artifact")
        models = tuple(self.setup_models)
        object.__setattr__(self, "setup_models", models)
        identities = (
            self.path_likelihood_artifact_id,
            self.dol_calibration_artifact_id,
        )
        minimum_coverage = _finite_probability(
            self.minimum_supported_coverage,
            name="minimum_supported_coverage",
        )
        object.__setattr__(
            self,
            "minimum_supported_coverage",
            minimum_coverage,
        )
        if (
            self.schema_version != PROBABILITY_ADMISSION_SCHEMA_VERSION
            or self.estimand != TARGET_BEFORE_INVALIDATION_ESTIMAND
            or any(not isinstance(value, str) or not value for value in identities)
            or len(self.signal_policy_fingerprint) != 64
            or any(not isinstance(item, SetupDeliveryModel) for item in models)
            or tuple(item.setup_family for item in models)
            != (SetupFamily.DFP, SetupFamily.LSR)
        ):
            raise ValueError("target-before-invalidation artifact is invalid")
        payload = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "calibration_id": self.calibration_id,
            "source_dataset_id": self.source_dataset_id,
            "coverage_id": self.coverage_id,
            "path_likelihood_artifact_id": (self.path_likelihood_artifact_id),
            "dol_calibration_artifact_id": self.dol_calibration_artifact_id,
            "signal_policy_fingerprint": self.signal_policy_fingerprint,
            **_artifact_clock_payload(
                trained_through=self.trained_through,
                valid_from=self.valid_from,
                expires_at=self.expires_at,
            ),
            "setup_models": [
                {
                    "setup_family": item.setup_family.value,
                    "intercept": item.intercept,
                    "path_logit_coefficient": item.path_logit_coefficient,
                    "dol_logit_coefficient": item.dol_logit_coefficient,
                    "half_life_real_completed_bars": (
                        item.half_life_real_completed_bars
                    ),
                }
                for item in models
            ],
            "minimum_supported_coverage": minimum_coverage,
            "estimand": self.estimand,
            "status": self.status,
            "authority": self.authority,
            "action_authority_ready": self.action_authority_ready,
        }
        object.__setattr__(
            self,
            "artifact_id",
            f"delivery-outcome-artifact:{_canonical_hash(payload)[:32]}",
        )

    def current_at(self, asof: pd.Timestamp) -> bool:
        clock = _aware_timestamp(asof, name="outcome artifact asof")
        return self.valid_from <= clock < self.expires_at

    def model_for(self, setup_family: SetupFamily) -> SetupDeliveryModel:
        family = SetupFamily(setup_family)
        return self.setup_models[(SetupFamily.DFP, SetupFamily.LSR).index(family)]


@dataclass(frozen=True)
class SignalPolicyProtocol:
    protocol_id: str
    protocol_version: str
    minimum_p_target_before_invalidation: float
    minimum_net_edge_R: float
    maximum_estimated_cost_R: float
    minimum_coverage: float
    maximum_input_age: pd.Timedelta
    maximum_half_life_real_completed_bars: int
    invalidation_loss_R: float = 1.0
    estimand: str = TARGET_BEFORE_INVALIDATION_ESTIMAND
    schema_version: int = SIGNAL_POLICY_SCHEMA_VERSION
    status: str = "development_unvalidated"
    authority: str = SHADOW_AUTHORITY
    action_authority_ready: bool = False
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        minimum_probability = _finite_probability(
            self.minimum_p_target_before_invalidation,
            name="minimum_p_target_before_invalidation",
        )
        minimum_coverage = _finite_probability(
            self.minimum_coverage,
            name="minimum_coverage",
        )
        object.__setattr__(
            self,
            "minimum_p_target_before_invalidation",
            minimum_probability,
        )
        object.__setattr__(self, "minimum_coverage", minimum_coverage)
        object.__setattr__(
            self,
            "maximum_input_age",
            _positive_timedelta(
                self.maximum_input_age,
                name="maximum_input_age",
            ),
        )
        numeric = (
            self.minimum_net_edge_R,
            self.maximum_estimated_cost_R,
            self.invalidation_loss_R,
        )
        if (
            self.schema_version != SIGNAL_POLICY_SCHEMA_VERSION
            or not self.protocol_id
            or not self.protocol_version
            or self.estimand != TARGET_BEFORE_INVALIDATION_ESTIMAND
            or self.status != "development_unvalidated"
            or self.authority != SHADOW_AUTHORITY
            or type(self.action_authority_ready) is not bool
            or self.action_authority_ready
            or any(not math.isfinite(float(value)) for value in numeric)
            or self.maximum_estimated_cost_R < 0.0
            or self.invalidation_loss_R <= 0.0
            or type(self.maximum_half_life_real_completed_bars) is not int
            or self.maximum_half_life_real_completed_bars <= 0
        ):
            raise ValueError("signal policy protocol is invalid")
        payload = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "minimum_p_target_before_invalidation": minimum_probability,
            "minimum_net_edge_R": self.minimum_net_edge_R,
            "maximum_estimated_cost_R": self.maximum_estimated_cost_R,
            "minimum_coverage": minimum_coverage,
            "maximum_input_age_ns": int(self.maximum_input_age.value),
            "maximum_half_life_real_completed_bars": (
                self.maximum_half_life_real_completed_bars
            ),
            "invalidation_loss_R": self.invalidation_loss_R,
            "estimand": self.estimand,
            "status": self.status,
            "authority": self.authority,
            "action_authority_ready": self.action_authority_ready,
            "setup_families": [
                SetupFamily.DFP.value,
                SetupFamily.LSR.value,
            ],
        }
        object.__setattr__(self, "fingerprint", _canonical_hash(payload))


def load_signal_policy_protocol(
    path: str | Path = "configs/signal_policy.json",
) -> SignalPolicyProtocol:
    """Load the exact fail-closed shadow Signal Policy protocol.

    Relative defaults resolve against the repository package root so replay
    and checkpoint workers do not depend on their process working directory.
    Extra or missing fields are rejected instead of being silently defaulted.
    """

    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[1] / source
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SignalPolicyProtocolError(
            f"unable to load signal policy protocol: {source}"
        ) from error
    required = {
        "schema_version",
        "protocol_id",
        "protocol_version",
        "minimum_p_target_before_invalidation",
        "minimum_net_edge_R",
        "maximum_estimated_cost_R",
        "minimum_coverage",
        "maximum_input_age",
        "maximum_half_life_real_completed_bars",
        "invalidation_loss_R",
        "estimand",
        "status",
        "authority",
        "action_authority_ready",
        "setup_families",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise SignalPolicyProtocolError(
            "signal policy protocol fields are not frozen exactly"
        )
    try:
        if type(payload["schema_version"]) is not int:
            raise TypeError("schema_version must be an integer")
        if payload["action_authority_ready"] is not False:
            raise TypeError("action_authority_ready must be false")
        numeric_names = (
            "minimum_p_target_before_invalidation",
            "minimum_net_edge_R",
            "maximum_estimated_cost_R",
            "minimum_coverage",
            "invalidation_loss_R",
        )
        if any(
            isinstance(payload[name], bool)
            or not isinstance(payload[name], (int, float))
            for name in numeric_names
        ):
            raise TypeError("signal policy numeric fields must be JSON numbers")
        if type(payload["maximum_half_life_real_completed_bars"]) is not int:
            raise TypeError("maximum half-life must be an integer")
        if any(
            not isinstance(payload[name], str) or not payload[name].strip()
            for name in (
                "protocol_id",
                "protocol_version",
                "maximum_input_age",
                "estimand",
                "status",
                "authority",
            )
        ):
            raise TypeError("signal policy text fields must be non-empty strings")
        if payload["setup_families"] != [
            SetupFamily.DFP.value,
            SetupFamily.LSR.value,
        ]:
            raise ValueError("signal policy setup families are not frozen")
        return SignalPolicyProtocol(
            protocol_id=str(payload["protocol_id"]).strip(),
            protocol_version=str(payload["protocol_version"]).strip(),
            minimum_p_target_before_invalidation=float(
                payload["minimum_p_target_before_invalidation"]
            ),
            minimum_net_edge_R=float(payload["minimum_net_edge_R"]),
            maximum_estimated_cost_R=float(
                payload["maximum_estimated_cost_R"]
            ),
            minimum_coverage=float(payload["minimum_coverage"]),
            maximum_input_age=pd.Timedelta(payload["maximum_input_age"]),
            maximum_half_life_real_completed_bars=payload[
                "maximum_half_life_real_completed_bars"
            ],
            invalidation_loss_R=float(payload["invalidation_loss_R"]),
            estimand=str(payload["estimand"]).strip(),
            schema_version=payload["schema_version"],
            status=str(payload["status"]).strip(),
            authority=str(payload["authority"]).strip(),
            action_authority_ready=False,
        )
    except (TypeError, ValueError) as error:
        raise SignalPolicyProtocolError(str(error)) from error


def _load_phase7_artifact_payload(
    path: str | Path,
    *,
    artifact_kind: str,
    required_fields: set[str],
) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[1] / source
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SignalPolicyArtifactError(
            f"unable to load {artifact_kind} artifact: {source}"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required_fields
        or payload.get("artifact_kind") != artifact_kind
    ):
        raise SignalPolicyArtifactError(
            f"{artifact_kind} artifact fields are not frozen exactly"
        )
    return payload


_ADMISSION_COMMON_FIELDS = {
    "artifact_kind",
    "schema_version",
    "protocol_id",
    "model_id",
    "model_version",
    "calibration_id",
    "source_dataset_id",
    "coverage_id",
    "trained_through",
    "valid_from",
    "expires_at",
    "status",
    "authority",
    "action_authority_ready",
    "artifact_id",
}


def _admission_common(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (
        type(payload["schema_version"]) is not int
        or payload["action_authority_ready"] is not False
        or any(
            not isinstance(payload[name], str) or not payload[name].strip()
            for name in (
                "protocol_id",
                "model_id",
                "model_version",
                "calibration_id",
                "source_dataset_id",
                "coverage_id",
                "trained_through",
                "valid_from",
                "expires_at",
                "status",
                "authority",
                "artifact_id",
            )
        )
    ):
        raise SignalPolicyArtifactError("artifact admission fields are invalid")
    return {
        "protocol_id": payload["protocol_id"].strip(),
        "model_id": payload["model_id"].strip(),
        "model_version": payload["model_version"].strip(),
        "calibration_id": payload["calibration_id"].strip(),
        "source_dataset_id": payload["source_dataset_id"].strip(),
        "coverage_id": payload["coverage_id"].strip(),
        "trained_through": pd.Timestamp(payload["trained_through"]),
        "valid_from": pd.Timestamp(payload["valid_from"]),
        "expires_at": pd.Timestamp(payload["expires_at"]),
        "schema_version": payload["schema_version"],
        "status": payload["status"].strip(),
        "authority": payload["authority"].strip(),
        "action_authority_ready": False,
    }


def load_signal_artifact_pins(
    path: str | Path,
    *,
    expected_admission_id: str,
) -> SignalArtifactPins:
    """Load the external admission record; it may not self-admit."""

    required = {
        "artifact_kind",
        "schema_version",
        "signal_policy_fingerprint",
        "path_protocol_fingerprint",
        "path_likelihood_artifact_id",
        "dol_calibration_artifact_id",
        "outcome_model_artifact_id",
        "dol_probability_model_fingerprint",
        "admission_id",
    }
    payload = _load_phase7_artifact_payload(
        path,
        artifact_kind="signal_artifact_pins",
        required_fields=required,
    )
    try:
        if payload["schema_version"] != PROBABILITY_ADMISSION_SCHEMA_VERSION:
            raise ValueError("signal artifact pin schema is invalid")
        pins = SignalArtifactPins(
            signal_policy_fingerprint=payload["signal_policy_fingerprint"],
            path_protocol_fingerprint=payload["path_protocol_fingerprint"],
            path_likelihood_artifact_id=payload[
                "path_likelihood_artifact_id"
            ],
            dol_calibration_artifact_id=payload[
                "dol_calibration_artifact_id"
            ],
            outcome_model_artifact_id=payload["outcome_model_artifact_id"],
            dol_probability_model_fingerprint=payload[
                "dol_probability_model_fingerprint"
            ],
        )
    except (TypeError, ValueError, KeyError) as error:
        raise SignalPolicyArtifactError(str(error)) from error
    if (
        not isinstance(expected_admission_id, str)
        or not expected_admission_id
        or payload["admission_id"] != expected_admission_id
        or pins.admission_id != expected_admission_id
    ):
        raise SignalPolicyArtifactError(
            "signal artifact pin admission identity is stale"
        )
    return pins


def load_path_likelihood_artifact(
    path: str | Path,
    *,
    expected_artifact_id: str,
) -> AdmittedPathLikelihoodArtifact:
    required = _ADMISSION_COMMON_FIELDS | {
        "source_path_protocol_fingerprint",
        "source_path_model_version",
        "temperature",
        "path_log_biases",
    }
    payload = _load_phase7_artifact_payload(
        path,
        artifact_kind="path_likelihood_calibration",
        required_fields=required,
    )
    try:
        raw_biases = payload["path_log_biases"]
        if (
            not isinstance(raw_biases, Mapping)
            or set(raw_biases) != {path.value for path in PATH_KINDS}
        ):
            raise ValueError("path likelihood biases are incomplete")
        artifact = AdmittedPathLikelihoodArtifact(
            **_admission_common(payload),
            source_path_protocol_fingerprint=payload[
                "source_path_protocol_fingerprint"
            ],
            source_path_model_version=payload["source_path_model_version"],
            temperature=payload["temperature"],
            path_log_biases=tuple(
                (path, raw_biases[path.value]) for path in PATH_KINDS
            ),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise SignalPolicyArtifactError(str(error)) from error
    if (
        payload["artifact_id"] != expected_artifact_id
        or artifact.artifact_id != expected_artifact_id
    ):
        raise SignalPolicyArtifactError("path likelihood artifact pin is stale")
    return artifact


def load_dol_calibration_artifact(
    path: str | Path,
    *,
    expected_artifact_id: str,
) -> AdmittedDOLCalibrationArtifact:
    required = _ADMISSION_COMMON_FIELDS | {
        "source_dol_protocol_fingerprint",
        "source_dol_model_version",
        "source_dol_model_fingerprint",
        "source_path_protocol_fingerprint",
        "source_path_model_version",
        "temperature",
    }
    payload = _load_phase7_artifact_payload(
        path,
        artifact_kind="dol_probability_calibration",
        required_fields=required,
    )
    try:
        artifact = AdmittedDOLCalibrationArtifact(
            **_admission_common(payload),
            source_dol_protocol_fingerprint=payload[
                "source_dol_protocol_fingerprint"
            ],
            source_dol_model_version=payload["source_dol_model_version"],
            source_dol_model_fingerprint=payload[
                "source_dol_model_fingerprint"
            ],
            source_path_protocol_fingerprint=payload[
                "source_path_protocol_fingerprint"
            ],
            source_path_model_version=payload["source_path_model_version"],
            temperature=payload["temperature"],
        )
    except (TypeError, ValueError, KeyError) as error:
        raise SignalPolicyArtifactError(str(error)) from error
    if (
        payload["artifact_id"] != expected_artifact_id
        or artifact.artifact_id != expected_artifact_id
    ):
        raise SignalPolicyArtifactError("DOL calibration artifact pin is stale")
    return artifact


def load_outcome_model_artifact(
    path: str | Path,
    *,
    expected_artifact_id: str,
) -> TargetBeforeInvalidationArtifact:
    required = _ADMISSION_COMMON_FIELDS | {
        "path_likelihood_artifact_id",
        "dol_calibration_artifact_id",
        "signal_policy_fingerprint",
        "setup_models",
        "minimum_supported_coverage",
        "estimand",
    }
    payload = _load_phase7_artifact_payload(
        path,
        artifact_kind="target_before_invalidation_model",
        required_fields=required,
    )
    try:
        raw_models = payload["setup_models"]
        model_fields = {
            "setup_family",
            "intercept",
            "path_logit_coefficient",
            "dol_logit_coefficient",
            "half_life_real_completed_bars",
        }
        if (
            not isinstance(raw_models, list)
            or len(raw_models) != 2
            or any(
                not isinstance(item, Mapping) or set(item) != model_fields
                for item in raw_models
            )
        ):
            raise ValueError("outcome setup models are not frozen exactly")
        artifact = TargetBeforeInvalidationArtifact(
            **_admission_common(payload),
            path_likelihood_artifact_id=payload[
                "path_likelihood_artifact_id"
            ],
            dol_calibration_artifact_id=payload[
                "dol_calibration_artifact_id"
            ],
            signal_policy_fingerprint=payload["signal_policy_fingerprint"],
            setup_models=tuple(
                SetupDeliveryModel(
                    setup_family=item["setup_family"],
                    intercept=item["intercept"],
                    path_logit_coefficient=item[
                        "path_logit_coefficient"
                    ],
                    dol_logit_coefficient=item["dol_logit_coefficient"],
                    half_life_real_completed_bars=item[
                        "half_life_real_completed_bars"
                    ],
                )
                for item in raw_models
            ),
            minimum_supported_coverage=payload[
                "minimum_supported_coverage"
            ],
            estimand=payload["estimand"],
        )
    except (TypeError, ValueError, KeyError) as error:
        raise SignalPolicyArtifactError(str(error)) from error
    if (
        payload["artifact_id"] != expected_artifact_id
        or artifact.artifact_id != expected_artifact_id
    ):
        raise SignalPolicyArtifactError("outcome model artifact pin is stale")
    return artifact


@dataclass(frozen=True)
class SignalEvaluationContext:
    coverage_id: str
    coverage_fraction: float
    ood_assessment_id: str
    out_of_distribution: bool
    cost_estimate_id: str | None
    estimated_cost_R: float | None
    source_event_ids: tuple[str, ...]
    real_completed_bars_since_setup: int | None = None

    def __post_init__(self) -> None:
        coverage = _finite_probability(
            self.coverage_fraction,
            name="coverage_fraction",
        )
        object.__setattr__(self, "coverage_fraction", coverage)
        sources = _exact_ids(
            self.source_event_ids,
            name="evaluation source_event_ids",
        )
        object.__setattr__(self, "source_event_ids", sources)
        if (
            not self.coverage_id
            or not self.ood_assessment_id
            or type(self.out_of_distribution) is not bool
            or (self.estimated_cost_R is None) != (self.cost_estimate_id is None)
            or (
                self.estimated_cost_R is not None
                and (
                    not math.isfinite(float(self.estimated_cost_R))
                    or self.estimated_cost_R < 0.0
                )
            )
            or (
                self.real_completed_bars_since_setup is not None
                and (
                    type(self.real_completed_bars_since_setup) is not int
                    or self.real_completed_bars_since_setup < 0
                )
            )
        ):
            raise ValueError("signal evaluation context is invalid")


@dataclass(frozen=True)
class CancelCondition:
    kind: CancelConditionKind
    reference_id: str
    operator: str
    source_ids: tuple[str, ...]
    trigger_at: pd.Timestamp | None = None
    threshold: float | None = None
    expected_state: str | None = None
    condition_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CancelConditionKind(self.kind))
        sources = _exact_ids(
            self.source_ids,
            name="cancel condition source_ids",
        )
        object.__setattr__(self, "source_ids", sources)
        if self.trigger_at is not None:
            object.__setattr__(
                self,
                "trigger_at",
                _aware_timestamp(
                    self.trigger_at,
                    name="cancel condition trigger_at",
                ),
            )
        if self.threshold is not None and not math.isfinite(float(self.threshold)):
            raise ValueError("cancel condition threshold must be finite")
        if (
            not self.reference_id
            or not self.operator
            or (
                self.trigger_at is None
                and self.threshold is None
                and not self.expected_state
            )
        ):
            raise ValueError("cancel condition is incomplete")
        payload = {
            "kind": self.kind.value,
            "reference_id": self.reference_id,
            "operator": self.operator,
            "source_ids": list(sources),
            "trigger_at": (
                None if self.trigger_at is None else self.trigger_at.isoformat()
            ),
            "threshold": self.threshold,
            "expected_state": self.expected_state,
        }
        object.__setattr__(
            self,
            "condition_id",
            f"cancel-condition:{_canonical_hash(payload)[:32]}",
        )


def trade_plan_identity(plan: TradePlan) -> str:
    """Return a deterministic identity without changing the legacy plan."""

    if not isinstance(plan, TradePlan):
        raise TypeError("trade plan identity requires TradePlan")
    payload = {
        "playbook": plan.playbook.value,
        "direction": plan.direction.value,
        "planned_entry": plan.planned_entry,
        "setup_id": plan.setup_id,
        "entry_location_id": plan.entry_location_id,
        "entry_path_id": plan.entry_path_id,
        "entry_zone_lower": plan.entry_zone_lower,
        "entry_zone_upper": plan.entry_zone_upper,
        "selected_draw_id": plan.selected_draw_id,
        "invalidation": {
            "price": plan.invalidation.price,
            "side": plan.invalidation.side,
            "source_level_id": plan.invalidation.source_level_id,
            "observed_at": plan.invalidation.observed_at.isoformat(),
            "rationale": plan.invalidation.rationale,
        },
        "targets": [
            {
                "level_id": target.level_id,
                "timeframe": target.timeframe.value,
                "side": target.side,
                "price": target.price,
                "formed_at": target.formed_at.isoformat(),
                "confirmed_at": target.confirmed_at.isoformat(),
            }
            for target in plan.targets
        ],
        "risk_points": plan.risk_points,
        "primary_target_R": plan.primary_target_R,
        "remaining_path_R": plan.remaining_path_R,
        "deadline": plan.deadline.isoformat(),
    }
    return f"trade-plan:{_canonical_hash(payload)[:32]}"


@dataclass(frozen=True)
class SignalAssessment:
    assessed_at: pd.Timestamp
    disposition: SignalDisposition
    symbol: str
    instrument_id: str
    candidate_id: str
    episode_id: str
    setup_id: str
    setup_family: SetupFamily | None
    playbook: Playbook
    direction: Direction
    trade_plan_id: str | None
    competition_set_id: str
    path_hypothesis_id: str | None
    path: PathKind | None
    dol_ranking_id: str
    dol_candidate_id: str
    policy_protocol_id: str
    policy_protocol_version: str
    policy_protocol_fingerprint: str
    path_likelihood_artifact_id: str | None
    path_model_id: str | None
    path_model_version: str | None
    path_calibration_id: str | None
    dol_calibration_artifact_id: str | None
    dol_model_id: str | None
    dol_model_version: str | None
    dol_calibration_id: str | None
    outcome_model_artifact_id: str | None
    outcome_model_id: str | None
    outcome_model_version: str | None
    outcome_calibration_id: str | None
    coverage_id: str
    ood_assessment_id: str
    cost_estimate_id: str | None
    source_event_ids: tuple[str, ...]
    source_identity_ids: tuple[str, ...]
    raw_path_probability: float | None
    p_path_hypothesis: float | None
    raw_dol_probability: float | None
    p_dol_candidate: float | None
    p_target_before_invalidation: float | None
    estimated_cost_R: float | None
    gross_edge_R: float | None
    net_edge_R: float | None
    age_real_completed_bars: int | None
    half_life_real_completed_bars: int | None
    # Retained as an explicit compatibility tombstone.  Schema v2 never maps
    # real-completed-bar lifetime to wall-clock time.
    half_life_expires_at: pd.Timestamp | None
    expires_at: pd.Timestamp
    cancel_conditions: tuple[CancelCondition, ...]
    rejection_reasons: tuple[SignalRejection, ...]
    estimand: str = TARGET_BEFORE_INVALIDATION_ESTIMAND
    schema_version: int = SIGNAL_ASSESSMENT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    can_authorize_trade: bool = False
    playbook_probability_consumed: bool = False
    signal_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assessed_at",
            _aware_timestamp(self.assessed_at, name="signal assessed_at"),
        )
        object.__setattr__(
            self,
            "expires_at",
            _aware_timestamp(
                self.expires_at,
                name="signal expires_at",
            ),
        )
        if self.half_life_expires_at is not None:
            object.__setattr__(
                self,
                "half_life_expires_at",
                _aware_timestamp(
                    self.half_life_expires_at,
                    name="signal half_life_expires_at",
                ),
            )
        object.__setattr__(
            self,
            "disposition",
            SignalDisposition(self.disposition),
        )
        object.__setattr__(self, "playbook", Playbook(self.playbook))
        object.__setattr__(self, "direction", Direction(self.direction))
        if self.setup_family is not None:
            object.__setattr__(
                self,
                "setup_family",
                SetupFamily(self.setup_family),
            )
        if self.path is not None:
            object.__setattr__(self, "path", PathKind(self.path))
        reasons = tuple(SignalRejection(value) for value in self.rejection_reasons)
        if len(reasons) != len(set(reasons)):
            raise ValueError("signal rejection reasons are duplicated")
        object.__setattr__(self, "rejection_reasons", reasons)
        sources = _exact_ids(self.source_event_ids, name="signal source_event_ids")
        identities = _exact_ids(
            self.source_identity_ids,
            name="signal source_identity_ids",
        )
        object.__setattr__(self, "source_event_ids", sources)
        object.__setattr__(self, "source_identity_ids", identities)
        probabilities = (
            self.raw_path_probability,
            self.p_path_hypothesis,
            self.raw_dol_probability,
            self.p_dol_candidate,
            self.p_target_before_invalidation,
        )
        for index, probability in enumerate(probabilities):
            if probability is not None:
                _finite_probability(probability, name=f"signal probability {index}")
        for value in (
            self.estimated_cost_R,
            self.gross_edge_R,
            self.net_edge_R,
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("signal metrics must be finite")
        if (
            self.age_real_completed_bars is not None
            and (
                type(self.age_real_completed_bars) is not int
                or self.age_real_completed_bars < 0
            )
        ):
            raise ValueError("signal real-completed-bar age is invalid")
        if (
            self.half_life_real_completed_bars is not None
            and (
                type(self.half_life_real_completed_bars) is not int
                or self.half_life_real_completed_bars <= 0
            )
        ):
            raise ValueError("signal real-completed-bar half-life is invalid")
        required_ids = (
            self.symbol,
            self.instrument_id,
            self.candidate_id,
            self.episode_id,
            self.setup_id,
            self.competition_set_id,
            self.dol_ranking_id,
            self.dol_candidate_id,
            self.policy_protocol_id,
            self.policy_protocol_version,
            self.policy_protocol_fingerprint,
            self.coverage_id,
            self.ood_assessment_id,
        )
        eligible = self.disposition is SignalDisposition.ELIGIBLE_SHADOW
        eligible_values = (
            self.setup_family,
            self.trade_plan_id,
            self.path_hypothesis_id,
            self.path,
            self.path_likelihood_artifact_id,
            self.path_model_id,
            self.path_model_version,
            self.path_calibration_id,
            self.dol_calibration_artifact_id,
            self.dol_model_id,
            self.dol_model_version,
            self.dol_calibration_id,
            self.outcome_model_artifact_id,
            self.outcome_model_id,
            self.outcome_model_version,
            self.outcome_calibration_id,
            self.p_path_hypothesis,
            self.p_dol_candidate,
            self.p_target_before_invalidation,
            self.estimated_cost_R,
            self.gross_edge_R,
            self.net_edge_R,
            self.age_real_completed_bars,
            self.half_life_real_completed_bars,
        )
        if (
            self.schema_version != SIGNAL_ASSESSMENT_SCHEMA_VERSION
            or any(not isinstance(value, str) or not value for value in required_ids)
            or len(self.policy_protocol_fingerprint) != 64
            or self.estimand != TARGET_BEFORE_INVALIDATION_ESTIMAND
            or self.authority != SHADOW_AUTHORITY
            or type(self.can_authorize_trade) is not bool
            or self.can_authorize_trade
            or type(self.playbook_probability_consumed) is not bool
            or self.playbook_probability_consumed
            or self.half_life_expires_at is not None
            or eligible != (not reasons)
            or (eligible and any(value is None for value in eligible_values))
            or (eligible and not self.cancel_conditions)
            or (eligible and self.expires_at <= self.assessed_at)
            or (
                eligible
                and self.age_real_completed_bars
                >= self.half_life_real_completed_bars
            )
            or any(
                not isinstance(condition, CancelCondition)
                for condition in self.cancel_conditions
            )
            or len({item.condition_id for item in self.cancel_conditions})
            != len(self.cancel_conditions)
        ):
            raise ValueError("signal assessment contract is invalid")
        payload = {
            name: (
                value.value
                if isinstance(value, Enum)
                else (
                    value.isoformat()
                    if isinstance(value, pd.Timestamp)
                    else (
                        [item.value for item in value]
                        if name == "rejection_reasons"
                        else (
                            [item.condition_id for item in value]
                            if name == "cancel_conditions"
                            else list(value) if isinstance(value, tuple) else value
                        )
                    )
                )
            )
            for name, value in self.__dict__.items()
            if name != "signal_id"
        }
        object.__setattr__(
            self,
            "signal_id",
            f"signal-assessment:{_canonical_hash(payload)[:32]}",
        )

    @property
    def eligible(self) -> bool:
        return self.disposition is SignalDisposition.ELIGIBLE_SHADOW

    @property
    def dol_probability_id(self) -> str | None:
        """Compatibility-safe access to the Phase 7 probability identity."""

        return (
            self.dol_ranking_id
            if self.dol_ranking_id.startswith("dol-probability:")
            else None
        )


def _append_reason(
    reasons: list[SignalRejection],
    reason: SignalRejection,
) -> None:
    if reason not in reasons:
        reasons.append(reason)


DOLSignalInput = DOLRankingResult | DOLProbabilityResult
DOLSignalCandidate = RankedDOLCandidate | DOLCandidateProbability


def _dol_result_id(value: DOLSignalInput) -> str:
    return (
        value.probability_id
        if isinstance(value, DOLProbabilityResult)
        else value.ranking_id
    )


def _setup_path(candidate: HypothesisBelief) -> PathKind | None:
    """Map an existing typed setup relation to one registered market path."""

    if candidate.market_thesis_mechanism == "range_failed_auction":
        return PathKind.FAILED_BREAKOUT
    return {
        "aligned": PathKind.CONTINUATION,
        "local_countertrend": PathKind.DEEPER_RETRACEMENT,
        "challenges_incumbent": PathKind.REVERSAL,
    }.get(candidate.market_thesis_authority_relation)


def _candidate_path_member(
    *,
    value: DOLSignalInput,
    candidate: DOLSignalCandidate,
    setup_candidate: HypothesisBelief,
    path_state: PathCompetitionSetState,
) -> PathHypothesisState | None:
    if isinstance(value, DOLRankingResult):
        path = _setup_path(setup_candidate)
        if path is None or candidate.path is not path:
            return None
        return path_state.member(path)
    path = _setup_path(setup_candidate)
    if path is None or path not in candidate.supported_paths:
        return None
    return path_state.member(path)


def _artifact_rejections(
    *,
    asof: pd.Timestamp,
    path_state: PathCompetitionSetState,
    ranking: DOLSignalInput,
    policy: SignalPolicyProtocol,
    path_protocol: PathBeliefProtocol | None,
    artifact_pins: SignalArtifactPins | None,
    dol_probability_model_artifact: DOLProbabilityModelArtifact | None,
    path_artifact: AdmittedPathLikelihoodArtifact | None,
    dol_artifact: AdmittedDOLCalibrationArtifact | None,
    outcome_artifact: TargetBeforeInvalidationArtifact | None,
) -> tuple[SignalRejection, ...]:
    reasons: list[SignalRejection] = []
    artifacts_supplied = any(
        artifact is not None
        for artifact in (path_artifact, dol_artifact, outcome_artifact)
    )
    if artifacts_supplied:
        if not isinstance(artifact_pins, SignalArtifactPins):
            reasons.append(SignalRejection.ARTIFACT_NOT_ADMITTED)
        elif (
            artifact_pins.signal_policy_fingerprint != policy.fingerprint
            or not isinstance(path_artifact, AdmittedPathLikelihoodArtifact)
            or artifact_pins.path_likelihood_artifact_id
            != path_artifact.artifact_id
            or not isinstance(dol_artifact, AdmittedDOLCalibrationArtifact)
            or artifact_pins.dol_calibration_artifact_id
            != dol_artifact.artifact_id
            or not isinstance(outcome_artifact, TargetBeforeInvalidationArtifact)
            or artifact_pins.outcome_model_artifact_id
            != outcome_artifact.artifact_id
            or (
                isinstance(ranking, DOLProbabilityResult)
                and artifact_pins.dol_probability_model_fingerprint
                != ranking.model_fingerprint
            )
            or (
                isinstance(ranking, DOLRankingResult)
                and artifact_pins.dol_probability_model_fingerprint is not None
            )
        ):
            reasons.append(SignalRejection.ARTIFACT_IDENTITY_MISMATCH)
        if not isinstance(path_protocol, PathBeliefProtocol):
            _append_reason(reasons, SignalRejection.ARTIFACT_NOT_ADMITTED)
        elif (
            not path_protocol.can_apply_bayesian_update
            or path_protocol.fingerprint != path_state.protocol_fingerprint
            or path_protocol.model_version != path_state.model_version
            or (
                artifact_pins is not None
                and artifact_pins.path_protocol_fingerprint
                != path_protocol.fingerprint
            )
        ):
            _append_reason(reasons, SignalRejection.ARTIFACT_IDENTITY_MISMATCH)
    if (
        not isinstance(ranking, DOLProbabilityResult)
        or (
            ranking.model_source != "fitted_admitted_artifact"
            or ranking.calibration_status != "fitted_admitted"
        )
    ):
        reasons.append(SignalRejection.DOL_PROBABILITY_NOT_FITTED)
    if isinstance(ranking, DOLProbabilityResult):
        if dol_probability_model_artifact is None:
            reasons.append(
                SignalRejection.MISSING_DOL_PROBABILITY_MODEL_ARTIFACT
            )
        elif not isinstance(
            dol_probability_model_artifact,
            DOLProbabilityModelArtifact,
        ):
            _append_reason(reasons, SignalRejection.ARTIFACT_NOT_ADMITTED)
        elif (
            dol_probability_model_artifact.fingerprint
            != ranking.model_fingerprint
            or dol_probability_model_artifact.model_version
            != ranking.model_version
            or dol_probability_model_artifact.protocol_fingerprint
            != ranking.protocol_fingerprint
            or dol_probability_model_artifact.ranking_protocol_fingerprint
            != ranking.ranking_protocol_fingerprint
            or dol_probability_model_artifact.fit_status != "fitted"
            or dol_probability_model_artifact.admission_status != "admitted"
            or dol_probability_model_artifact.calibration_status
            != "fitted_admitted"
            or dol_probability_model_artifact.authority != "shadow_only"
            or dol_probability_model_artifact.action_authority is not False
            or (
                artifact_pins is not None
                and artifact_pins.dol_probability_model_fingerprint
                != dol_probability_model_artifact.fingerprint
            )
        ):
            _append_reason(
                reasons,
                SignalRejection.ARTIFACT_IDENTITY_MISMATCH,
            )
    if path_artifact is None:
        reasons.append(SignalRejection.MISSING_PATH_LIKELIHOOD_ARTIFACT)
    elif not isinstance(path_artifact, AdmittedPathLikelihoodArtifact):
        reasons.append(SignalRejection.ARTIFACT_NOT_ADMITTED)
    elif not path_artifact.current_at(asof):
        reasons.append(SignalRejection.ARTIFACT_OUTSIDE_VALIDITY)
    elif not path_artifact.admits(path_state):
        reasons.append(SignalRejection.ARTIFACT_IDENTITY_MISMATCH)

    if dol_artifact is None:
        reasons.append(SignalRejection.MISSING_DOL_CALIBRATION_ARTIFACT)
    elif not isinstance(dol_artifact, AdmittedDOLCalibrationArtifact):
        _append_reason(reasons, SignalRejection.ARTIFACT_NOT_ADMITTED)
    elif not dol_artifact.current_at(asof):
        _append_reason(reasons, SignalRejection.ARTIFACT_OUTSIDE_VALIDITY)
    elif not dol_artifact.admits(ranking):
        _append_reason(reasons, SignalRejection.ARTIFACT_IDENTITY_MISMATCH)

    if outcome_artifact is None:
        reasons.append(SignalRejection.MISSING_OUTCOME_MODEL_ARTIFACT)
    elif not isinstance(outcome_artifact, TargetBeforeInvalidationArtifact):
        _append_reason(reasons, SignalRejection.ARTIFACT_NOT_ADMITTED)
    elif not outcome_artifact.current_at(asof):
        _append_reason(reasons, SignalRejection.ARTIFACT_OUTSIDE_VALIDITY)
    elif (
        not isinstance(
            path_artifact,
            AdmittedPathLikelihoodArtifact,
        )
        or not isinstance(
            dol_artifact,
            AdmittedDOLCalibrationArtifact,
        )
        or outcome_artifact.path_likelihood_artifact_id != path_artifact.artifact_id
        or outcome_artifact.dol_calibration_artifact_id != dol_artifact.artifact_id
        or outcome_artifact.signal_policy_fingerprint != policy.fingerprint
    ):
        _append_reason(reasons, SignalRejection.ARTIFACT_IDENTITY_MISMATCH)
    return tuple(reasons)


def _cancel_conditions(
    *,
    assessment_clock: pd.Timestamp,
    signal_expiry: pd.Timestamp,
    half_life_real_completed_bars: int,
    plan: TradePlan,
    episode: EntryEpisodeState,
    path_state: PathCompetitionSetState,
    path_member: PathHypothesisState,
    ranking: DOLSignalInput,
    dol_candidate: DOLSignalCandidate,
    policy: SignalPolicyProtocol,
    minimum_supported_coverage: float,
    context: SignalEvaluationContext,
    artifacts: Sequence[Any],
    source_ids: tuple[str, ...],
) -> tuple[CancelCondition, ...]:
    conditions = [
        CancelCondition(
            kind=CancelConditionKind.SIGNAL_EXPIRY_REACHED,
            reference_id="signal-expiry",
            operator=">=",
            trigger_at=signal_expiry,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.SIGNAL_HALF_LIFE_ELAPSED,
            reference_id=episode.episode_id,
            operator=">=",
            threshold=float(half_life_real_completed_bars),
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.PLAN_DEADLINE_REACHED,
            reference_id=trade_plan_identity(plan),
            operator=">=",
            trigger_at=plan.deadline,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.PATH_SET_NO_LONGER_ACTIVE,
            reference_id=path_state.competition_set_id,
            operator="state_not_equal",
            expected_state=PathStatus.ACTIVE.value,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.PATH_HYPOTHESIS_NO_LONGER_ACTIVE,
            reference_id=path_member.hypothesis_id,
            operator="state_not_equal",
            expected_state=PathStatus.ACTIVE.value,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.DOL_CANDIDATE_UNAVAILABLE,
            reference_id=dol_candidate.candidate_id,
            operator="state_not_equal",
            expected_state="available",
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.EPISODE_NO_LONGER_EXECUTABLE,
            reference_id=episode.episode_id,
            operator="state_not_equal",
            expected_state=PlaybookPhase.EXECUTABLE.value,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.STRUCTURAL_INVALIDATION_REACHED,
            reference_id=plan.invalidation.source_level_id,
            operator=("<=" if plan.direction is Direction.LONG else ">="),
            threshold=plan.invalidation.price,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.COVERAGE_BELOW_MINIMUM,
            reference_id=context.coverage_id,
            operator="<",
            threshold=minimum_supported_coverage,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.OUT_OF_DISTRIBUTION_DETECTED,
            reference_id=context.ood_assessment_id,
            operator="state_equal",
            expected_state="out_of_distribution",
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.COST_EXCEEDS_LIMIT,
            reference_id=context.cost_estimate_id or "cost-estimate:missing",
            operator=">",
            threshold=policy.maximum_estimated_cost_R,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=(CancelConditionKind.DELIVERY_PROBABILITY_BELOW_MINIMUM),
            reference_id=policy.fingerprint,
            operator="<",
            threshold=policy.minimum_p_target_before_invalidation,
            source_ids=source_ids,
        ),
        CancelCondition(
            kind=CancelConditionKind.NET_EDGE_BELOW_MINIMUM,
            reference_id=policy.fingerprint,
            operator="<",
            threshold=policy.minimum_net_edge_R,
            source_ids=source_ids,
        ),
    ]
    for artifact in artifacts:
        conditions.append(
            CancelCondition(
                kind=CancelConditionKind.ARTIFACT_EXPIRY_REACHED,
                reference_id=artifact.artifact_id,
                operator=">=",
                trigger_at=artifact.expires_at,
                source_ids=source_ids,
            )
        )
    # The clock is included in the signature through the prospective expiry;
    # accepting it here also makes a future accidental past-expiry call loud.
    if signal_expiry <= assessment_clock:
        raise ValueError("cannot create cancel conditions for an expired signal")
    return tuple(conditions)


def assess_signal(
    policy: SignalPolicyProtocol,
    *,
    asof: pd.Timestamp,
    symbol: str,
    instrument_id: str,
    path_state: PathCompetitionSetState,
    dol_ranking: DOLSignalInput,
    dol_candidate_id: str,
    setup_candidate: HypothesisBelief,
    entry_episode: EntryEpisodeState,
    evaluation: SignalEvaluationContext,
    path_likelihood_artifact: AdmittedPathLikelihoodArtifact | None,
    dol_calibration_artifact: AdmittedDOLCalibrationArtifact | None,
    outcome_model_artifact: TargetBeforeInvalidationArtifact | None,
    dol_probability_model_artifact: DOLProbabilityModelArtifact | None = None,
    path_protocol: PathBeliefProtocol | None = None,
    artifact_pins: SignalArtifactPins | None = None,
) -> SignalAssessment:
    """Assess one exact typed setup without granting trade authority."""

    if not isinstance(policy, SignalPolicyProtocol):
        raise TypeError("signal assessment requires SignalPolicyProtocol")
    if not isinstance(path_state, PathCompetitionSetState):
        raise TypeError("signal assessment requires PathCompetitionSetState")
    if not isinstance(dol_ranking, (DOLRankingResult, DOLProbabilityResult)):
        raise TypeError(
            "signal assessment requires DOLRankingResult or "
            "DOLProbabilityResult"
        )
    if not isinstance(setup_candidate, HypothesisBelief):
        raise TypeError("signal assessment requires typed HypothesisBelief")
    if not isinstance(entry_episode, EntryEpisodeState):
        raise TypeError("signal assessment requires EntryEpisodeState")
    if not isinstance(evaluation, SignalEvaluationContext):
        raise TypeError("signal assessment requires SignalEvaluationContext")
    if path_protocol is not None and not isinstance(path_protocol, PathBeliefProtocol):
        raise TypeError("signal assessment path protocol must be exact")
    if artifact_pins is not None and not isinstance(artifact_pins, SignalArtifactPins):
        raise TypeError("signal assessment artifact pins must be exact")
    if (
        dol_probability_model_artifact is not None
        and not isinstance(
            dol_probability_model_artifact,
            DOLProbabilityModelArtifact,
        )
    ):
        raise TypeError("signal assessment DOL model artifact must be exact")
    artifact_types = (
        (path_likelihood_artifact, AdmittedPathLikelihoodArtifact),
        (dol_calibration_artifact, AdmittedDOLCalibrationArtifact),
        (outcome_model_artifact, TargetBeforeInvalidationArtifact),
    )
    if any(
        artifact is not None and not isinstance(artifact, expected)
        for artifact, expected in artifact_types
    ):
        raise TypeError("signal assessment received an untyped admission artifact")
    clock = _aware_timestamp(asof, name="signal assessment asof")
    if not symbol or not instrument_id or not dol_candidate_id:
        raise ValueError("signal scope identities must be non-empty")

    reasons: list[SignalRejection] = []
    setup_family = _playbook_setup_family(setup_candidate.playbook)
    if setup_family is None:
        reasons.append(SignalRejection.SETUP_FAMILY_PARKED)
    if (
        setup_candidate.phase is not PlaybookPhase.EXECUTABLE
        or entry_episode.phase is not PlaybookPhase.EXECUTABLE
        or setup_candidate.terminal_at is not None
        or entry_episode.terminal_at is not None
    ):
        reasons.append(SignalRejection.SETUP_NOT_EXECUTABLE)

    plan = setup_candidate.plan
    identity_ok = bool(
        setup_candidate.record_kind != "summary"
        and setup_candidate.candidate_id == entry_episode.candidate_id
        and setup_candidate.episode_id == entry_episode.episode_id
        and setup_candidate.context_thesis_id == entry_episode.parent_context_thesis_id
        and setup_candidate.playbook is entry_episode.playbook
        and setup_candidate.direction is entry_episode.direction
        and plan is not None
        and plan == entry_episode.plan
        and plan.playbook is setup_candidate.playbook
        and plan.direction is setup_candidate.direction
        and plan.setup_id == entry_episode.episode_id
        and plan.entry_location_id == entry_episode.entry_location_id
        and plan.entry_path_id == entry_episode.entry_path_id
    )
    if not identity_ok:
        reasons.append(SignalRejection.SETUP_IDENTITY_MISMATCH)
    if (
        plan is None
        or plan.setup_id is None
        or plan.entry_location_id is None
        or plan.entry_path_id is None
        or plan.selected_draw_id is None
        or plan.primary_target_R <= 0.0
    ):
        reasons.append(SignalRejection.TRADE_PLAN_INCOMPLETE)

    ranked = next(
        (
            item
            for item in dol_ranking.ranked_candidates
            if item.candidate_id == dol_candidate_id
        ),
        None,
    )
    if ranked is None:
        reasons.append(SignalRejection.DOL_CANDIDATE_UNAVAILABLE)
    if plan is not None:
        target_matches = bool(
            plan.selected_draw_id == dol_candidate_id
            and plan.targets
            and plan.targets[0].level_id == dol_candidate_id
            and ranked is not None
            and math.isclose(
                plan.targets[0].price,
                ranked.target_price,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and plan.targets[0].side
            == setup_candidate.direction.opposing_liquidity_side
        )
        if not target_matches:
            reasons.append(SignalRejection.PLAN_TARGET_MISMATCH)

    if path_state.status is not PathStatus.ACTIVE:
        reasons.append(SignalRejection.PATH_SET_INACTIVE)
    if (
        path_state.instrument_id != instrument_id
        or dol_ranking.competition_set_id != path_state.competition_set_id
        or dol_ranking.path_asof != path_state.asof
        or dol_ranking.path_protocol_fingerprint != path_state.protocol_fingerprint
        or dol_ranking.path_model_version != path_state.model_version
    ):
        reasons.append(SignalRejection.PATH_IDENTITY_MISMATCH)

    path_member: PathHypothesisState | None = None
    if ranked is not None:
        path_member = _candidate_path_member(
            value=dol_ranking,
            candidate=ranked,
            setup_candidate=setup_candidate,
            path_state=path_state,
        )
        if path_member is None:
            reasons.append(SignalRejection.DOL_IDENTITY_MISMATCH)
        elif path_member.status is not PathStatus.ACTIVE:
            reasons.append(SignalRejection.PATH_HYPOTHESIS_INACTIVE)
        if path_member is not None:
            if isinstance(dol_ranking, DOLRankingResult):
                if (
                    ranked.path_hypothesis_id != path_member.hypothesis_id
                    or not math.isclose(
                        ranked.path_probability,
                        path_member.probability,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    reasons.append(SignalRejection.DOL_IDENTITY_MISMATCH)
            else:
                contribution = next(
                    (
                        item
                        for item in ranked.path_contributions
                        if item.path is path_member.path
                    ),
                    None,
                )
                if (
                    contribution is None
                    or contribution.path_hypothesis_id
                    != path_member.hypothesis_id
                    or not math.isclose(
                        contribution.path_probability,
                        path_member.probability,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    reasons.append(SignalRejection.DOL_IDENTITY_MISMATCH)
        expected_direction = (
            DOLDirection.LONG
            if setup_candidate.direction is Direction.LONG
            else DOLDirection.SHORT
        )
        if dol_ranking.direction is not expected_direction or (
            isinstance(ranked, RankedDOLCandidate)
            and ranked.direction is not expected_direction
        ):
            reasons.append(SignalRejection.DOL_IDENTITY_MISMATCH)

    source_event_ids = set(evaluation.source_event_ids)
    if entry_episode.initiating_event_id is not None:
        source_event_ids.add(entry_episode.initiating_event_id)
    source_events = tuple(sorted(source_event_ids))
    source_identity_ids = {
        setup_candidate.candidate_id or entry_episode.candidate_id,
        entry_episode.episode_id,
        path_state.competition_set_id,
        _dol_result_id(dol_ranking),
        dol_candidate_id,
        evaluation.coverage_id,
        evaluation.ood_assessment_id,
    }
    if evaluation.cost_estimate_id is not None:
        source_identity_ids.add(evaluation.cost_estimate_id)
    if artifact_pins is not None:
        source_identity_ids.add(artifact_pins.admission_id)
    if ranked is not None:
        source_identity_ids.update(ranked.candidate_source_ids)

    current_state_clocks = (
        path_state.asof,
        dol_ranking.path_asof,
        entry_episode.updated_at,
    )
    causal_observation_clocks = (
        *setup_candidate.causal_observation_clocks,
        *entry_episode.causal_observation_clocks,
    )
    if any(
        value > clock
        for value in (
            *current_state_clocks,
            *causal_observation_clocks,
        )
    ):
        reasons.append(SignalRejection.INPUT_FROM_FUTURE)
    if any(
        clock - value > policy.maximum_input_age
        for value in current_state_clocks
    ):
        reasons.append(SignalRejection.INPUT_STALE)
    if (
        clock >= path_state.common_expires_at
        or clock >= entry_episode.deadline
        or (plan is not None and clock >= plan.deadline)
    ):
        reasons.append(SignalRejection.SIGNAL_EXPIRED)

    for reason in _artifact_rejections(
        asof=clock,
        path_state=path_state,
        ranking=dol_ranking,
        policy=policy,
        path_protocol=path_protocol,
        artifact_pins=artifact_pins,
        dol_probability_model_artifact=dol_probability_model_artifact,
        path_artifact=path_likelihood_artifact,
        dol_artifact=dol_calibration_artifact,
        outcome_artifact=outcome_model_artifact,
    ):
        _append_reason(reasons, reason)

    if evaluation.out_of_distribution:
        reasons.append(SignalRejection.OUT_OF_DISTRIBUTION)
    supported_coverage = policy.minimum_coverage
    if outcome_model_artifact is not None:
        supported_coverage = max(
            supported_coverage,
            outcome_model_artifact.minimum_supported_coverage,
        )
    if evaluation.coverage_fraction < supported_coverage:
        reasons.append(SignalRejection.COVERAGE_INCOMPLETE)
    if evaluation.estimated_cost_R is None:
        reasons.append(SignalRejection.COST_UNAVAILABLE)
    elif evaluation.estimated_cost_R > policy.maximum_estimated_cost_R:
        reasons.append(SignalRejection.COST_EXCEEDS_LIMIT)

    raw_path_probability = (
        None if path_member is None else path_member.probability
    )
    raw_dol_probability = (
        None
        if ranked is None
        else ranked.probability
        if isinstance(ranked, DOLCandidateProbability)
        else ranked.normalized_diagnostic_weight
    )
    p_path: float | None = None
    p_dol: float | None = None
    p_delivery: float | None = None
    gross_edge_R: float | None = None
    net_edge_R: float | None = None
    age_real_completed_bars = evaluation.real_completed_bars_since_setup
    half_life_real_completed_bars: int | None = None
    expires_at = min(
        path_state.common_expires_at,
        entry_episode.deadline,
        plan.deadline if plan is not None else entry_episode.deadline,
    )

    artifact_ready = bool(
        isinstance(
            path_likelihood_artifact,
            AdmittedPathLikelihoodArtifact,
        )
        and isinstance(
            dol_calibration_artifact,
            AdmittedDOLCalibrationArtifact,
        )
        and isinstance(
            outcome_model_artifact,
            TargetBeforeInvalidationArtifact,
        )
        and isinstance(path_protocol, PathBeliefProtocol)
        and isinstance(artifact_pins, SignalArtifactPins)
        and isinstance(dol_ranking, DOLProbabilityResult)
        and path_state.status is PathStatus.ACTIVE
        and path_member is not None
        and path_member.status is PathStatus.ACTIVE
        and not any(
            reason
            in {
                SignalRejection.ARTIFACT_NOT_ADMITTED,
                SignalRejection.ARTIFACT_OUTSIDE_VALIDITY,
                SignalRejection.ARTIFACT_IDENTITY_MISMATCH,
            }
            for reason in reasons
        )
    )
    if (
        artifact_ready
        and ranked is not None
        and path_member is not None
        and setup_family is not None
    ):
        admitted_paths = path_likelihood_artifact.probabilities(path_state)
        admitted_dols = dol_calibration_artifact.probabilities(dol_ranking)
        p_path = admitted_paths[path_member.path]
        p_dol = admitted_dols[ranked.candidate_id]
        setup_model = outcome_model_artifact.model_for(setup_family)
        p_delivery = setup_model.estimate(p_path=p_path, p_dol=p_dol)
        half_life_real_completed_bars = min(
            setup_model.half_life_real_completed_bars,
            policy.maximum_half_life_real_completed_bars,
        )
        expires_at = min(
            expires_at,
            path_likelihood_artifact.expires_at,
            dol_calibration_artifact.expires_at,
            outcome_model_artifact.expires_at,
        )
        if age_real_completed_bars is None:
            _append_reason(
                reasons,
                SignalRejection.REAL_COMPLETED_BAR_AGE_UNAVAILABLE,
            )
        elif age_real_completed_bars >= half_life_real_completed_bars:
            _append_reason(reasons, SignalRejection.SIGNAL_HALF_LIFE_ELAPSED)
        if expires_at <= clock:
            _append_reason(reasons, SignalRejection.SIGNAL_EXPIRED)
        if p_delivery < policy.minimum_p_target_before_invalidation:
            reasons.append(SignalRejection.PROBABILITY_BELOW_MINIMUM)
        if plan is not None and evaluation.estimated_cost_R is not None:
            gross_edge_R = (
                p_delivery * plan.primary_target_R
                - (1.0 - p_delivery) * policy.invalidation_loss_R
            )
            net_edge_R = gross_edge_R - evaluation.estimated_cost_R
            if net_edge_R < policy.minimum_net_edge_R:
                reasons.append(SignalRejection.EDGE_BELOW_MINIMUM)

    disposition = (
        SignalDisposition.ELIGIBLE_SHADOW if not reasons else SignalDisposition.REJECTED
    )
    conditions: tuple[CancelCondition, ...] = ()
    if (
        disposition is SignalDisposition.ELIGIBLE_SHADOW
        and plan is not None
        and path_member is not None
        and ranked is not None
        and path_likelihood_artifact is not None
        and dol_calibration_artifact is not None
        and outcome_model_artifact is not None
        and half_life_real_completed_bars is not None
    ):
        conditions = _cancel_conditions(
            assessment_clock=clock,
            signal_expiry=expires_at,
            half_life_real_completed_bars=half_life_real_completed_bars,
            plan=plan,
            episode=entry_episode,
            path_state=path_state,
            path_member=path_member,
            ranking=dol_ranking,
            dol_candidate=ranked,
            policy=policy,
            minimum_supported_coverage=supported_coverage,
            context=evaluation,
            artifacts=(
                path_likelihood_artifact,
                dol_calibration_artifact,
                outcome_model_artifact,
            ),
            source_ids=source_events,
        )

    return SignalAssessment(
        assessed_at=clock,
        disposition=disposition,
        symbol=symbol,
        instrument_id=instrument_id,
        candidate_id=setup_candidate.candidate_id or entry_episode.candidate_id,
        episode_id=entry_episode.episode_id,
        setup_id=entry_episode.episode_id,
        setup_family=setup_family,
        playbook=setup_candidate.playbook,
        direction=setup_candidate.direction,
        trade_plan_id=(None if plan is None else trade_plan_identity(plan)),
        competition_set_id=path_state.competition_set_id,
        path_hypothesis_id=(None if path_member is None else path_member.hypothesis_id),
        path=None if path_member is None else path_member.path,
        dol_ranking_id=_dol_result_id(dol_ranking),
        dol_candidate_id=dol_candidate_id,
        policy_protocol_id=policy.protocol_id,
        policy_protocol_version=policy.protocol_version,
        policy_protocol_fingerprint=policy.fingerprint,
        path_likelihood_artifact_id=(
            None
            if path_likelihood_artifact is None
            else path_likelihood_artifact.artifact_id
        ),
        path_model_id=(
            None
            if path_likelihood_artifact is None
            else path_likelihood_artifact.model_id
        ),
        path_model_version=(
            None
            if path_likelihood_artifact is None
            else path_likelihood_artifact.model_version
        ),
        path_calibration_id=(
            None
            if path_likelihood_artifact is None
            else path_likelihood_artifact.calibration_id
        ),
        dol_calibration_artifact_id=(
            None
            if dol_calibration_artifact is None
            else dol_calibration_artifact.artifact_id
        ),
        dol_model_id=(
            None
            if dol_calibration_artifact is None
            else dol_calibration_artifact.model_id
        ),
        dol_model_version=(
            None
            if dol_calibration_artifact is None
            else dol_calibration_artifact.model_version
        ),
        dol_calibration_id=(
            None
            if dol_calibration_artifact is None
            else dol_calibration_artifact.calibration_id
        ),
        outcome_model_artifact_id=(
            None
            if outcome_model_artifact is None
            else outcome_model_artifact.artifact_id
        ),
        outcome_model_id=(
            None if outcome_model_artifact is None else outcome_model_artifact.model_id
        ),
        outcome_model_version=(
            None
            if outcome_model_artifact is None
            else outcome_model_artifact.model_version
        ),
        outcome_calibration_id=(
            None
            if outcome_model_artifact is None
            else outcome_model_artifact.calibration_id
        ),
        coverage_id=evaluation.coverage_id,
        ood_assessment_id=evaluation.ood_assessment_id,
        cost_estimate_id=evaluation.cost_estimate_id,
        source_event_ids=source_events,
        source_identity_ids=tuple(sorted(source_identity_ids)),
        raw_path_probability=raw_path_probability,
        p_path_hypothesis=p_path,
        raw_dol_probability=raw_dol_probability,
        p_dol_candidate=p_dol,
        p_target_before_invalidation=p_delivery,
        estimated_cost_R=evaluation.estimated_cost_R,
        gross_edge_R=gross_edge_R,
        net_edge_R=net_edge_R,
        age_real_completed_bars=age_real_completed_bars,
        half_life_real_completed_bars=half_life_real_completed_bars,
        half_life_expires_at=None,
        expires_at=expires_at,
        cancel_conditions=conditions,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "ADMITTED_SHADOW_STATUS",
    "AdmittedDOLCalibrationArtifact",
    "AdmittedPathLikelihoodArtifact",
    "CancelCondition",
    "CancelConditionKind",
    "PROBABILITY_ADMISSION_SCHEMA_VERSION",
    "SHADOW_AUTHORITY",
    "SIGNAL_ASSESSMENT_SCHEMA_VERSION",
    "SIGNAL_POLICY_SCHEMA_VERSION",
    "SetupDeliveryModel",
    "SetupFamily",
    "SignalArtifactPins",
    "SignalAssessment",
    "SignalDisposition",
    "SignalEvaluationContext",
    "SignalPolicyProtocol",
    "SignalPolicyArtifactError",
    "SignalPolicyProtocolError",
    "SignalRejection",
    "TARGET_BEFORE_INVALIDATION_ESTIMAND",
    "TargetBeforeInvalidationArtifact",
    "assess_signal",
    "load_dol_calibration_artifact",
    "load_outcome_model_artifact",
    "load_path_likelihood_artifact",
    "load_signal_artifact_pins",
    "load_signal_policy_protocol",
    "trade_plan_identity",
]
