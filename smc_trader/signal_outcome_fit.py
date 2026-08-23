"""Research-only fits for the two missing empirical signal artifacts.

This module is deliberately outside ``signal_policy`` and the production
Engine.  It fits only caller-supplied, causally timestamped observations and
never reads a dataset, writes a pin, or grants action authority.

Two independent artifacts are defined:

* a scalar temperature for a complete DOL outcome distribution (including
  the registered no-target outcome); and
* setup-specific DFP/LSR logistic delivery models for
  ``target_before_invalidation`` using only ``p_path`` and ``p_dol``.

Fit artifacts are always ``fitted_not_admitted``/``CLOSED``.  A separate
validation call can emit an admitted *research evidence* receipt only for a
disjoint rolling-OOF cohort.  The already-open 2024-06 W1/W2 windows are
permanently ineligible for rolling-OOF admission here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .signal_policy import SetupFamily, TARGET_BEFORE_INVALIDATION_ESTIMAND


SIGNAL_OUTCOME_FIT_SCHEMA_VERSION = 1
DOL_TEMPERATURE_FIT_SCHEMA = "signal_dol_temperature_fit_v1"
DELIVERY_MODEL_FIT_SCHEMA = "signal_delivery_logistic_fit_v1"
SIGNAL_OUTCOME_ADMISSION_SCHEMA = "signal_outcome_admission_receipt_v1"

_JUNE_DEVELOPMENT_WINDOWS = frozenset(
    {"W1", "W2", "2024-06-week-1", "2024-06-week-2"}
)
_ALLOWED_COHORT_ROLES = frozenset(
    {
        "development_fit",
        "development_cross_fit",
        "calibration",
        "historical_validation",
        "rolling_oof",
    }
)
_HEX = frozenset("0123456789abcdef")
_DOL_TEMPERATURE_BOUNDS = (0.05, 20.0)
_DOL_OPTIMIZER_ITERATIONS = 120
_LOGISTIC_L2 = 1e-4
_LOGISTIC_MAX_ITERATIONS = 100
_LOGISTIC_TOLERANCE = 1e-10
_PROBABILITY_EPSILON = 1e-12


class SignalOutcomeFitError(ValueError):
    """Raised for leakage, invalid lineage, or unidentified fits."""


class DOLSupportLabel(str, Enum):
    CANDIDATE_TARGET = "candidate_target"
    NO_TARGET = "no_target_before_common_horizon"


class SignalOutcomeArtifactKind(str, Enum):
    DOL_TEMPERATURE = "dol_probability_calibration"
    DELIVERY_MODEL = "target_before_invalidation_model"


def _identity(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise SignalOutcomeFitError(f"{name} must be canonical non-empty text")
    return value


def _sha256(value: Any, *, name: str) -> str:
    result = _identity(value, name=name)
    if len(result) != 64 or any(character not in _HEX for character in result):
        raise SignalOutcomeFitError(f"{name} must be a lowercase SHA-256")
    return result


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SignalOutcomeFitError(f"{name} must be a timestamp") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise SignalOutcomeFitError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC")


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise SignalOutcomeFitError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SignalOutcomeFitError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = " and positive" if positive else ""
        raise SignalOutcomeFitError(f"{name} must be finite{qualifier}")
    return result


def _probability(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise SignalOutcomeFitError(f"{name} must lie in [0, 1]")
    return result


def _ids(
    values: Iterable[str],
    *,
    name: str,
    allow_empty: bool = False,
    sort: bool = False,
) -> tuple[str, ...]:
    result = tuple(_identity(value, name=name) for value in values)
    if (
        (not allow_empty and not result)
        or len(result) != len(set(result))
    ):
        raise SignalOutcomeFitError(f"{name} must contain unique non-empty identities")
    return tuple(sorted(result)) if sort else result


def _normal(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if is_dataclass(value):
        return {
            item.name: _normal(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_") and hasattr(value, item.name)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normal(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise SignalOutcomeFitError("canonical payload cannot contain non-finite floats")
    if hasattr(value, "item"):
        return _normal(value.item())
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _normal(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SignalOutcomeFitError("value is not canonical-JSON serializable") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _logit(value: float) -> float:
    bounded = min(1.0 - _PROBABILITY_EPSILON, max(_PROBABILITY_EPSILON, value))
    return math.log(bounded / (1.0 - bounded))


def _logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class SignalOutcomeCohort:
    """Exact source/split role for a caller-materialized cohort."""

    cohort_name: str
    cohort_role: str
    window_ids: tuple[str, ...]
    fold_ids: tuple[str, ...]
    source_dataset_sha256: str
    manifest_sha256: str
    cohort_identity_sha256: str
    split_protocol_sha256: str
    prediction_coverage: float
    sealed_oos_opened: bool = False
    schema_version: int = SIGNAL_OUTCOME_FIT_SCHEMA_VERSION
    cohort_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _identity(self.cohort_name, name="cohort_name")
        role = _identity(self.cohort_role, name="cohort_role")
        windows = _ids(self.window_ids, name="window_ids", sort=True)
        folds = _ids(self.fold_ids, name="fold_ids", sort=True)
        hashes = (
            _sha256(self.source_dataset_sha256, name="source_dataset_sha256"),
            _sha256(self.manifest_sha256, name="manifest_sha256"),
            _sha256(self.cohort_identity_sha256, name="cohort_identity_sha256"),
            _sha256(self.split_protocol_sha256, name="split_protocol_sha256"),
        )
        coverage = _probability(self.prediction_coverage, name="prediction_coverage")
        if (
            self.schema_version != SIGNAL_OUTCOME_FIT_SCHEMA_VERSION
            or role not in _ALLOWED_COHORT_ROLES
            or type(self.sealed_oos_opened) is not bool
            or self.sealed_oos_opened
        ):
            raise SignalOutcomeFitError("signal outcome cohort is invalid")
        if role == "rolling_oof" and any(
            window in _JUNE_DEVELOPMENT_WINDOWS for window in windows
        ):
            raise SignalOutcomeFitError("2024-06 W1/W2 cannot be relabelled rolling_oof")
        object.__setattr__(self, "cohort_name", name)
        object.__setattr__(self, "cohort_role", role)
        object.__setattr__(self, "window_ids", windows)
        object.__setattr__(self, "fold_ids", folds)
        object.__setattr__(self, "prediction_coverage", coverage)
        payload = {
            "schema_version": self.schema_version,
            "cohort_name": name,
            "cohort_role": role,
            "window_ids": windows,
            "fold_ids": folds,
            "source_dataset_sha256": hashes[0],
            "manifest_sha256": hashes[1],
            "cohort_identity_sha256": hashes[2],
            "split_protocol_sha256": hashes[3],
            "prediction_coverage": coverage,
            "sealed_oos_opened": False,
        }
        object.__setattr__(self, "cohort_id", f"signal-outcome-cohort:{_digest(payload)[:32]}")

    @property
    def contains_june_development(self) -> bool:
        return any(window in _JUNE_DEVELOPMENT_WINDOWS for window in self.window_ids)


@dataclass(frozen=True)
class DOLModelLineage:
    source_dol_protocol_fingerprint: str
    source_dol_model_version: str
    source_dol_model_fingerprint: str
    source_path_protocol_fingerprint: str
    source_path_model_version: str
    lineage_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_dol_protocol_fingerprint",
            "source_dol_model_fingerprint",
            "source_path_protocol_fingerprint",
        ):
            _sha256(getattr(self, name), name=name)
        _identity(self.source_dol_model_version, name="source_dol_model_version")
        _identity(self.source_path_model_version, name="source_path_model_version")
        object.__setattr__(self, "lineage_id", f"dol-fit-lineage:{_digest(self)[:32]}")


@dataclass(frozen=True)
class DeliveryModelLineage:
    path_likelihood_artifact_id: str
    dol_calibration_fit_artifact_id: str
    signal_policy_fingerprint: str
    lineage_id: str = field(init=False)

    def __post_init__(self) -> None:
        _identity(self.path_likelihood_artifact_id, name="path_likelihood_artifact_id")
        _identity(self.dol_calibration_fit_artifact_id, name="dol_calibration_fit_artifact_id")
        _sha256(self.signal_policy_fingerprint, name="signal_policy_fingerprint")
        object.__setattr__(self, "lineage_id", f"delivery-fit-lineage:{_digest(self)[:32]}")


@dataclass(frozen=True)
class RealCompletedBarEvidence:
    """One real, gap-free completed bar counted toward delivery half-life."""

    bar_id: str
    real_completed_bar_ordinal: int
    known_at: pd.Timestamp
    synthetic_no_trade: bool = False
    data_gap_before_minutes: int = 0
    source_reset: bool = False

    def __post_init__(self) -> None:
        _identity(self.bar_id, name="real completed bar_id")
        known = _aware(self.known_at, name="real completed bar known_at")
        if (
            type(self.real_completed_bar_ordinal) is not int
            or self.real_completed_bar_ordinal < 1
            or type(self.synthetic_no_trade) is not bool
            or self.synthetic_no_trade
            or type(self.data_gap_before_minutes) is not int
            or self.data_gap_before_minutes != 0
            or type(self.source_reset) is not bool
            or self.source_reset
        ):
            raise SignalOutcomeFitError(
                "half-life evidence must be one real, gap-free completed bar"
            )
        object.__setattr__(self, "known_at", known)


@dataclass(frozen=True)
class DOLTemperatureObservation:
    case_id: str
    prediction_id: str
    outcome_event_id: str
    window_id: str
    fold_id: str
    cluster_id: str
    prediction_known_at: pd.Timestamp
    outcome_known_at: pd.Timestamp
    outcome_probabilities: tuple[tuple[str, float], ...]
    realized_outcome_id: str
    support_label: DOLSupportLabel
    source_dol_model_fingerprint: str
    sample_weight: float = 1.0
    observation_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "prediction_id",
            "outcome_event_id",
            "window_id",
            "fold_id",
            "cluster_id",
            "realized_outcome_id",
        ):
            _identity(getattr(self, name), name=f"DOL observation {name}")
        prediction = _aware(self.prediction_known_at, name="DOL prediction_known_at")
        outcome = _aware(self.outcome_known_at, name="DOL outcome_known_at")
        probabilities = tuple(
            sorted(
                (
                    _identity(outcome_id, name="DOL outcome id"),
                    _probability(probability, name=f"DOL probability {outcome_id}"),
                )
                for outcome_id, probability in self.outcome_probabilities
            )
        )
        if (
            prediction >= outcome
            or len(probabilities) < 2
            or len({key for key, _ in probabilities}) != len(probabilities)
            or self.realized_outcome_id not in dict(probabilities)
            or not math.isclose(
                math.fsum(value for _, value in probabilities),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise SignalOutcomeFitError(
                "DOL observation requires a complete causal outcome distribution"
            )
        weight = _finite(self.sample_weight, name="DOL sample_weight", positive=True)
        support = DOLSupportLabel(self.support_label)
        _sha256(self.source_dol_model_fingerprint, name="source_dol_model_fingerprint")
        if (
            support is DOLSupportLabel.NO_TARGET
            and self.realized_outcome_id != DOLSupportLabel.NO_TARGET.value
        ):
            raise SignalOutcomeFitError("DOL no-target support label conflicts with outcome")
        if (
            support is DOLSupportLabel.CANDIDATE_TARGET
            and self.realized_outcome_id == DOLSupportLabel.NO_TARGET.value
        ):
            raise SignalOutcomeFitError("DOL candidate support label conflicts with outcome")
        object.__setattr__(self, "prediction_known_at", prediction)
        object.__setattr__(self, "outcome_known_at", outcome)
        object.__setattr__(self, "outcome_probabilities", probabilities)
        object.__setattr__(self, "support_label", support)
        object.__setattr__(self, "sample_weight", weight)
        object.__setattr__(
            self,
            "observation_id",
            f"dol-temperature-observation:{_digest(self)[:32]}",
        )


@dataclass(frozen=True)
class DeliveryOutcomeObservation:
    case_id: str
    prediction_id: str
    outcome_event_id: str
    window_id: str
    fold_id: str
    cluster_id: str
    setup_family: SetupFamily
    prediction_known_at: pd.Timestamp
    outcome_known_at: pd.Timestamp
    p_path: float
    p_dol: float
    target_before_invalidation: bool
    prediction_real_completed_bar_ordinal: int
    real_completed_bars: tuple[RealCompletedBarEvidence, ...]
    path_likelihood_artifact_id: str
    dol_calibration_fit_artifact_id: str
    sample_weight: float = 1.0
    estimand: str = TARGET_BEFORE_INVALIDATION_ESTIMAND
    observation_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "prediction_id",
            "outcome_event_id",
            "window_id",
            "fold_id",
            "cluster_id",
            "path_likelihood_artifact_id",
            "dol_calibration_fit_artifact_id",
        ):
            _identity(getattr(self, name), name=f"delivery observation {name}")
        setup = SetupFamily(self.setup_family)
        if setup not in {SetupFamily.DFP, SetupFamily.LSR}:
            raise SignalOutcomeFitError("delivery fit supports only DFP and LSR")
        prediction = _aware(self.prediction_known_at, name="delivery prediction_known_at")
        outcome = _aware(self.outcome_known_at, name="delivery outcome_known_at")
        bars = tuple(self.real_completed_bars)
        ordinals = tuple(item.real_completed_bar_ordinal for item in bars)
        if (
            prediction >= outcome
            or type(self.target_before_invalidation) is not bool
            or self.estimand != TARGET_BEFORE_INVALIDATION_ESTIMAND
            or type(self.prediction_real_completed_bar_ordinal) is not int
            or self.prediction_real_completed_bar_ordinal < 0
            or not bars
            or any(not isinstance(item, RealCompletedBarEvidence) for item in bars)
            or len({item.bar_id for item in bars}) != len(bars)
            or any(not prediction < item.known_at <= outcome for item in bars)
            or tuple(sorted(bars, key=lambda item: item.real_completed_bar_ordinal)) != bars
            or ordinals
            != tuple(
                range(
                    self.prediction_real_completed_bar_ordinal + 1,
                    self.prediction_real_completed_bar_ordinal + 1 + len(ordinals),
                )
            )
            or any(
                prior.known_at >= current.known_at
                for prior, current in zip(bars, bars[1:])
            )
            or bars[-1].known_at != outcome
        ):
            raise SignalOutcomeFitError(
                "delivery observation has leakage or incomplete real-bar ancestry"
            )
        path_probability = _probability(self.p_path, name="p_path")
        dol_probability = _probability(self.p_dol, name="p_dol")
        weight = _finite(self.sample_weight, name="delivery sample_weight", positive=True)
        object.__setattr__(self, "setup_family", setup)
        object.__setattr__(self, "prediction_known_at", prediction)
        object.__setattr__(self, "outcome_known_at", outcome)
        object.__setattr__(self, "p_path", path_probability)
        object.__setattr__(self, "p_dol", dol_probability)
        object.__setattr__(self, "real_completed_bars", bars)
        object.__setattr__(self, "sample_weight", weight)
        object.__setattr__(
            self,
            "observation_id",
            f"delivery-outcome-observation:{_digest(self)[:32]}",
        )

    @property
    def real_completed_bars_to_outcome(self) -> int:
        return len(self.real_completed_bars)


def _validate_cohort_rows(
    rows: Sequence[DOLTemperatureObservation | DeliveryOutcomeObservation],
    cohort: SignalOutcomeCohort,
) -> None:
    if not isinstance(cohort, SignalOutcomeCohort):
        raise TypeError("cohort must be SignalOutcomeCohort")
    if (
        not rows
        or len({item.observation_id for item in rows}) != len(rows)
        or len({item.case_id for item in rows}) != len(rows)
        or {item.window_id for item in rows} != set(cohort.window_ids)
        or {item.fold_id for item in rows} != set(cohort.fold_ids)
        or any(
            item.window_id not in cohort.window_ids or item.fold_id not in cohort.fold_ids
            for item in rows
        )
    ):
        raise SignalOutcomeFitError("observations do not form the exact bound cohort")


def _temperature_probabilities(
    probabilities: Sequence[tuple[str, float]],
    temperature: float,
) -> dict[str, float]:
    temperature = _finite(temperature, name="temperature", positive=True)
    scores = tuple(
        math.log(max(float(value), 1e-300)) / temperature
        for _, value in probabilities
    )
    maximum = max(scores)
    weights = tuple(math.exp(score - maximum) for score in scores)
    denominator = math.fsum(weights)
    return {
        outcome_id: weight / denominator
        for (outcome_id, _), weight in zip(probabilities, weights, strict=True)
    }


def _weighted_log_loss_dol(
    rows: Sequence[DOLTemperatureObservation],
    temperature: float,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        probability = _temperature_probabilities(
            row.outcome_probabilities,
            temperature,
        )[row.realized_outcome_id]
        numerator -= row.sample_weight * math.log(max(probability, 1e-300))
        denominator += row.sample_weight
    return numerator / denominator


def _weighted_brier_dol(
    rows: Sequence[DOLTemperatureObservation],
    temperature: float,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        predictions = _temperature_probabilities(row.outcome_probabilities, temperature)
        score = math.fsum(
            (value - float(outcome_id == row.realized_outcome_id)) ** 2
            for outcome_id, value in predictions.items()
        )
        numerator += row.sample_weight * score
        denominator += row.sample_weight
    return numerator / denominator


def _fit_temperature(rows: Sequence[DOLTemperatureObservation]) -> float:
    lower = math.log(_DOL_TEMPERATURE_BOUNDS[0])
    upper = math.log(_DOL_TEMPERATURE_BOUNDS[1])
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_loss = _weighted_log_loss_dol(rows, math.exp(left))
    right_loss = _weighted_log_loss_dol(rows, math.exp(right))
    for _ in range(_DOL_OPTIMIZER_ITERATIONS):
        if left_loss <= right_loss:
            upper = right
            right = left
            right_loss = left_loss
            left = upper - ratio * (upper - lower)
            left_loss = _weighted_log_loss_dol(rows, math.exp(left))
        else:
            lower = left
            left = right
            left_loss = right_loss
            right = lower + ratio * (upper - lower)
            right_loss = _weighted_log_loss_dol(rows, math.exp(right))
    fitted = math.exp((lower + upper) / 2.0)
    baseline = _weighted_log_loss_dol(rows, 1.0)
    if _weighted_log_loss_dol(rows, fitted) >= baseline - 1e-12:
        return 1.0
    return fitted


@dataclass(frozen=True)
class DOLTemperatureFitArtifact:
    model_version: str
    lineage: DOLModelLineage
    source_cohort_id: str
    source_cohort_role: str
    source_window_ids: tuple[str, ...]
    source_fold_ids: tuple[str, ...]
    source_dataset_sha256: str
    manifest_sha256: str
    cohort_identity_sha256: str
    split_protocol_sha256: str
    source_observation_ids: tuple[str, ...]
    source_case_ids: tuple[str, ...]
    source_cluster_ids: tuple[str, ...]
    trained_through: pd.Timestamp
    temperature: float
    raw_log_loss: float
    calibrated_log_loss: float
    raw_brier: float
    calibrated_brier: float
    fit_sample_count: int
    support_units: tuple[tuple[str, int], ...]
    algorithm: str = "bounded_log_temperature_golden_section_v1"
    fit_status: str = "fitted_not_admitted"
    admission_status: str = "CLOSED"
    authority: str = "research_only"
    action_authority: bool = False
    pins_issued: bool = False
    schema_version: str = DOL_TEMPERATURE_FIT_SCHEMA
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, DOLModelLineage):
            raise TypeError("DOL temperature artifact requires DOLModelLineage")
        trained = _aware(self.trained_through, name="DOL fit trained_through")
        observations = _ids(self.source_observation_ids, name="DOL source_observation_ids")
        cases = _ids(self.source_case_ids, name="DOL source_case_ids")
        clusters = _ids(self.source_cluster_ids, name="DOL source_cluster_ids", sort=True)
        supports = tuple(self.support_units)
        windows = _ids(self.source_window_ids, name="DOL source_window_ids", sort=True)
        folds = _ids(self.source_fold_ids, name="DOL source_fold_ids", sort=True)
        metrics = (
            self.temperature,
            self.raw_log_loss,
            self.calibrated_log_loss,
            self.raw_brier,
            self.calibrated_brier,
        )
        expected_supports = tuple(
            sorted(
                (
                    DOLSupportLabel.CANDIDATE_TARGET.value,
                    DOLSupportLabel.NO_TARGET.value,
                )
            )
        )
        if (
            self.schema_version != DOL_TEMPERATURE_FIT_SCHEMA
            or self.fit_status != "fitted_not_admitted"
            or self.admission_status != "CLOSED"
            or self.authority != "research_only"
            or self.action_authority is not False
            or self.pins_issued is not False
            or self.algorithm != "bounded_log_temperature_golden_section_v1"
            or not isinstance(self.lineage, DOLModelLineage)
            or type(self.fit_sample_count) is not int
            or self.fit_sample_count != len(observations)
            or len(cases) != len(observations)
            or len(clusters) > len(observations)
            or any(not math.isfinite(float(value)) for value in metrics)
            or not _DOL_TEMPERATURE_BOUNDS[0]
            <= self.temperature
            <= _DOL_TEMPERATURE_BOUNDS[1]
            or any(float(value) < 0.0 for value in metrics[1:])
            or tuple(sorted(supports)) != supports
            or tuple(name for name, _ in supports) != expected_supports
            or any(type(count) is not int or count <= 0 for _, count in supports)
            or sum(count for _, count in supports) != self.fit_sample_count
            or self.source_cohort_role not in _ALLOWED_COHORT_ROLES
            or (
                self.source_cohort_role == "rolling_oof"
                and any(window in _JUNE_DEVELOPMENT_WINDOWS for window in windows)
            )
        ):
            raise SignalOutcomeFitError("DOL temperature fit artifact is invalid")
        for name in (
            "source_dataset_sha256",
            "manifest_sha256",
            "cohort_identity_sha256",
            "split_protocol_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        _identity(self.model_version, name="DOL fit model_version")
        _identity(self.source_cohort_id, name="DOL fit source_cohort_id")
        _identity(self.source_cohort_role, name="DOL fit source_cohort_role")
        object.__setattr__(self, "trained_through", trained)
        object.__setattr__(self, "source_observation_ids", observations)
        object.__setattr__(self, "source_case_ids", cases)
        object.__setattr__(self, "source_cluster_ids", clusters)
        object.__setattr__(self, "source_window_ids", windows)
        object.__setattr__(self, "source_fold_ids", folds)
        object.__setattr__(self, "support_units", supports)
        object.__setattr__(self, "artifact_id", f"dol-temperature-fit:{_digest(self)[:32]}")

    def to_payload(self) -> dict[str, Any]:
        return _normal(self)


def fit_dol_temperature(
    observations: Sequence[DOLTemperatureObservation],
    *,
    cohort: SignalOutcomeCohort,
    lineage: DOLModelLineage,
    model_version: str = "signal_dol_temperature_v1",
) -> DOLTemperatureFitArtifact:
    """Fit one scalar temperature; the result remains closed until receipt."""

    rows = tuple(observations)
    if any(not isinstance(item, DOLTemperatureObservation) for item in rows):
        raise TypeError("DOL temperature fit requires typed observations")
    rows = tuple(sorted(rows, key=lambda item: item.observation_id))
    _validate_cohort_rows(rows, cohort)
    if not isinstance(lineage, DOLModelLineage):
        raise TypeError("lineage must be DOLModelLineage")
    if any(
        item.source_dol_model_fingerprint != lineage.source_dol_model_fingerprint
        for item in rows
    ):
        raise SignalOutcomeFitError("DOL observations do not bind the fitted model lineage")
    if len(rows) < 2 or len({item.support_label for item in rows}) < 2:
        raise SignalOutcomeFitError("DOL temperature is unidentified without both support classes")
    temperature = _fit_temperature(rows)
    supports = Counter(item.support_label.value for item in rows)
    return DOLTemperatureFitArtifact(
        model_version=model_version,
        lineage=lineage,
        source_cohort_id=cohort.cohort_id,
        source_cohort_role=cohort.cohort_role,
        source_window_ids=cohort.window_ids,
        source_fold_ids=cohort.fold_ids,
        source_dataset_sha256=cohort.source_dataset_sha256,
        manifest_sha256=cohort.manifest_sha256,
        cohort_identity_sha256=cohort.cohort_identity_sha256,
        split_protocol_sha256=cohort.split_protocol_sha256,
        source_observation_ids=tuple(item.observation_id for item in rows),
        source_case_ids=tuple(item.case_id for item in rows),
        source_cluster_ids=tuple(sorted({item.cluster_id for item in rows})),
        trained_through=max(item.outcome_known_at for item in rows),
        temperature=temperature,
        raw_log_loss=_weighted_log_loss_dol(rows, 1.0),
        calibrated_log_loss=_weighted_log_loss_dol(rows, temperature),
        raw_brier=_weighted_brier_dol(rows, 1.0),
        calibrated_brier=_weighted_brier_dol(rows, temperature),
        fit_sample_count=len(rows),
        support_units=tuple(sorted(supports.items())),
    )


def _weighted_binary_metrics(
    outcomes: Sequence[bool],
    predictions: Sequence[float],
    weights: Sequence[float],
) -> tuple[float, float]:
    if not outcomes or not (len(outcomes) == len(predictions) == len(weights)):
        raise SignalOutcomeFitError("binary metric inputs are invalid")
    denominator = math.fsum(weights)
    log_loss = 0.0
    brier = 0.0
    for outcome, prediction, weight in zip(outcomes, predictions, weights, strict=True):
        bounded = min(
            1.0 - _PROBABILITY_EPSILON,
            max(_PROBABILITY_EPSILON, float(prediction)),
        )
        target = float(outcome)
        log_loss -= weight * (
            target * math.log(bounded) + (1.0 - target) * math.log(1.0 - bounded)
        )
        brier += weight * (bounded - target) ** 2
    return log_loss / denominator, brier / denominator


def _penalized_logistic_loss(
    design: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    linear = design @ coefficients
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(linear, -700.0, 700.0)))
    probabilities = np.clip(probabilities, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    loss = -np.sum(
        weights
        * (
            outcomes * np.log(probabilities)
            + (1.0 - outcomes) * np.log(1.0 - probabilities)
        )
    )
    penalty = 0.5 * _LOGISTIC_L2 * float(np.dot(coefficients[1:], coefficients[1:]))
    return float(loss + penalty)


def _fit_logistic_coefficients(
    rows: Sequence[DeliveryOutcomeObservation],
) -> tuple[float, float, float]:
    design = np.asarray(
        [
            (1.0, _logit(item.p_path), _logit(item.p_dol))
            for item in rows
        ],
        dtype=float,
    )
    outcomes = np.asarray(
        [float(item.target_before_invalidation) for item in rows],
        dtype=float,
    )
    weights = np.asarray([item.sample_weight for item in rows], dtype=float)
    prevalence = float(np.sum(weights * outcomes) / np.sum(weights))
    coefficients = np.asarray([_logit(prevalence), 0.0, 0.0], dtype=float)
    penalty = np.diag((0.0, _LOGISTIC_L2, _LOGISTIC_L2))
    for _ in range(_LOGISTIC_MAX_ITERATIONS):
        linear = design @ coefficients
        probabilities = np.asarray([_logistic(float(value)) for value in linear])
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = design.T @ (weights * (probabilities - outcomes)) + penalty @ coefficients
        hessian = design.T @ ((weights * variance)[:, None] * design) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        prior_loss = _penalized_logistic_loss(
            design,
            outcomes,
            weights,
            coefficients,
        )
        scale = 1.0
        candidate = coefficients - step
        while (
            _penalized_logistic_loss(design, outcomes, weights, candidate)
            > prior_loss + 1e-12
            and scale > 2.0**-20
        ):
            scale *= 0.5
            candidate = coefficients - scale * step
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        if change <= _LOGISTIC_TOLERANCE:
            break
    if np.any(~np.isfinite(coefficients)):
        raise SignalOutcomeFitError("delivery logistic fit did not converge to finite coefficients")
    return tuple(float(value) for value in coefficients)  # type: ignore[return-value]


def _weighted_median_real_bars(
    rows: Sequence[DeliveryOutcomeObservation],
) -> int:
    ordered = sorted(
        (
            item.real_completed_bars_to_outcome,
            item.observation_id,
            item.sample_weight,
        )
        for item in rows
    )
    threshold = math.fsum(weight for _, _, weight in ordered) / 2.0
    cumulative = 0.0
    for count, _, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return max(1, int(count))
    raise AssertionError("weighted median has no terminal element")


@dataclass(frozen=True)
class DeliverySetupModelFit:
    setup_family: SetupFamily
    intercept: float
    path_logit_coefficient: float
    dol_logit_coefficient: float
    half_life_real_completed_bars: int
    fit_prevalence: float
    fit_log_loss: float
    fit_brier: float
    fit_sample_count: int
    positive_count: int
    negative_count: int
    half_life_method: str = "weighted_median_resolved_outcome_real_completed_bars_v1"

    def __post_init__(self) -> None:
        setup = SetupFamily(self.setup_family)
        if setup not in {SetupFamily.DFP, SetupFamily.LSR}:
            raise SignalOutcomeFitError("delivery setup model supports only DFP/LSR")
        numeric = (
            self.intercept,
            self.path_logit_coefficient,
            self.dol_logit_coefficient,
            self.fit_prevalence,
            self.fit_log_loss,
            self.fit_brier,
        )
        if (
            any(not math.isfinite(float(value)) for value in numeric)
            or not 0.0 < float(self.fit_prevalence) < 1.0
            or type(self.half_life_real_completed_bars) is not int
            or self.half_life_real_completed_bars <= 0
            or type(self.fit_sample_count) is not int
            or type(self.positive_count) is not int
            or type(self.negative_count) is not int
            or self.positive_count <= 0
            or self.negative_count <= 0
            or self.fit_sample_count != self.positive_count + self.negative_count
            or self.half_life_method
            != "weighted_median_resolved_outcome_real_completed_bars_v1"
        ):
            raise SignalOutcomeFitError("delivery setup model fit is invalid")
        object.__setattr__(self, "setup_family", setup)

    def estimate(self, *, p_path: float, p_dol: float) -> float:
        path = _probability(p_path, name="p_path")
        dol = _probability(p_dol, name="p_dol")
        return _logistic(
            self.intercept
            + self.path_logit_coefficient * _logit(path)
            + self.dol_logit_coefficient * _logit(dol)
        )


@dataclass(frozen=True)
class DeliveryModelFitArtifact:
    model_version: str
    lineage: DeliveryModelLineage
    source_cohort_id: str
    source_cohort_role: str
    source_window_ids: tuple[str, ...]
    source_fold_ids: tuple[str, ...]
    source_dataset_sha256: str
    manifest_sha256: str
    cohort_identity_sha256: str
    split_protocol_sha256: str
    source_observation_ids: tuple[str, ...]
    source_case_ids: tuple[str, ...]
    source_cluster_ids: tuple[str, ...]
    trained_through: pd.Timestamp
    setup_models: tuple[DeliverySetupModelFit, ...]
    minimum_supported_coverage: float
    fit_sample_count: int
    algorithm: str = "setup_specific_weighted_logistic_irls_l2_v1"
    estimand: str = TARGET_BEFORE_INVALIDATION_ESTIMAND
    fit_status: str = "fitted_not_admitted"
    admission_status: str = "CLOSED"
    authority: str = "research_only"
    action_authority: bool = False
    pins_issued: bool = False
    schema_version: str = DELIVERY_MODEL_FIT_SCHEMA
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, DeliveryModelLineage):
            raise TypeError("delivery artifact requires DeliveryModelLineage")
        trained = _aware(self.trained_through, name="delivery fit trained_through")
        observations = _ids(self.source_observation_ids, name="delivery source_observation_ids")
        cases = _ids(self.source_case_ids, name="delivery source_case_ids")
        clusters = _ids(
            self.source_cluster_ids,
            name="delivery source_cluster_ids",
            sort=True,
        )
        models = tuple(self.setup_models)
        windows = _ids(
            self.source_window_ids,
            name="delivery source_window_ids",
            sort=True,
        )
        folds = _ids(self.source_fold_ids, name="delivery source_fold_ids", sort=True)
        coverage = _probability(
            self.minimum_supported_coverage,
            name="minimum_supported_coverage",
        )
        if (
            self.schema_version != DELIVERY_MODEL_FIT_SCHEMA
            or self.estimand != TARGET_BEFORE_INVALIDATION_ESTIMAND
            or self.fit_status != "fitted_not_admitted"
            or self.admission_status != "CLOSED"
            or self.authority != "research_only"
            or self.action_authority is not False
            or self.pins_issued is not False
            or self.algorithm != "setup_specific_weighted_logistic_irls_l2_v1"
            or tuple(item.setup_family for item in models)
            != (SetupFamily.DFP, SetupFamily.LSR)
            or any(not isinstance(item, DeliverySetupModelFit) for item in models)
            or type(self.fit_sample_count) is not int
            or self.fit_sample_count != len(observations)
            or len(cases) != len(observations)
            or len(clusters) > len(observations)
            or self.fit_sample_count != sum(item.fit_sample_count for item in models)
            or self.source_cohort_role not in _ALLOWED_COHORT_ROLES
            or (
                self.source_cohort_role == "rolling_oof"
                and any(window in _JUNE_DEVELOPMENT_WINDOWS for window in windows)
            )
        ):
            raise SignalOutcomeFitError("delivery model fit artifact is invalid")
        for name in (
            "source_dataset_sha256",
            "manifest_sha256",
            "cohort_identity_sha256",
            "split_protocol_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        _identity(self.model_version, name="delivery fit model_version")
        _identity(self.source_cohort_id, name="delivery fit source_cohort_id")
        _identity(self.source_cohort_role, name="delivery fit source_cohort_role")
        object.__setattr__(self, "trained_through", trained)
        object.__setattr__(self, "source_observation_ids", observations)
        object.__setattr__(self, "source_case_ids", cases)
        object.__setattr__(self, "source_cluster_ids", clusters)
        object.__setattr__(self, "source_window_ids", windows)
        object.__setattr__(self, "source_fold_ids", folds)
        object.__setattr__(self, "setup_models", models)
        object.__setattr__(self, "minimum_supported_coverage", coverage)
        object.__setattr__(self, "artifact_id", f"delivery-model-fit:{_digest(self)[:32]}")

    def model_for(self, setup_family: SetupFamily) -> DeliverySetupModelFit:
        typed = SetupFamily(setup_family)
        return self.setup_models[(SetupFamily.DFP, SetupFamily.LSR).index(typed)]

    def to_payload(self) -> dict[str, Any]:
        return _normal(self)


def fit_delivery_model(
    observations: Sequence[DeliveryOutcomeObservation],
    *,
    cohort: SignalOutcomeCohort,
    lineage: DeliveryModelLineage,
    minimum_supported_coverage: float = 0.95,
    model_version: str = "signal_target_before_invalidation_logistic_v1",
) -> DeliveryModelFitArtifact:
    """Fit independent DFP/LSR logits and real-completed-bar half-lives."""

    rows = tuple(observations)
    if any(not isinstance(item, DeliveryOutcomeObservation) for item in rows):
        raise TypeError("delivery fit requires typed observations")
    rows = tuple(sorted(rows, key=lambda item: item.observation_id))
    _validate_cohort_rows(rows, cohort)
    if not isinstance(lineage, DeliveryModelLineage):
        raise TypeError("lineage must be DeliveryModelLineage")
    if any(
        item.path_likelihood_artifact_id != lineage.path_likelihood_artifact_id
        or item.dol_calibration_fit_artifact_id
        != lineage.dol_calibration_fit_artifact_id
        for item in rows
    ):
        raise SignalOutcomeFitError("delivery observations do not bind prediction lineage")
    models: list[DeliverySetupModelFit] = []
    for setup in (SetupFamily.DFP, SetupFamily.LSR):
        cell = tuple(item for item in rows if item.setup_family is setup)
        outcomes = tuple(item.target_before_invalidation for item in cell)
        if len(cell) < 4 or len(set(outcomes)) != 2:
            raise SignalOutcomeFitError(f"delivery model is unidentified for {setup.value}")
        intercept, path_coefficient, dol_coefficient = _fit_logistic_coefficients(cell)
        predictions = tuple(
            _logistic(
                intercept
                + path_coefficient * _logit(item.p_path)
                + dol_coefficient * _logit(item.p_dol)
            )
            for item in cell
        )
        weights = tuple(item.sample_weight for item in cell)
        log_loss, brier = _weighted_binary_metrics(outcomes, predictions, weights)
        positive = sum(outcomes)
        models.append(
            DeliverySetupModelFit(
                setup_family=setup,
                intercept=intercept,
                path_logit_coefficient=path_coefficient,
                dol_logit_coefficient=dol_coefficient,
                half_life_real_completed_bars=_weighted_median_real_bars(cell),
                fit_prevalence=(
                    math.fsum(
                        item.sample_weight * float(item.target_before_invalidation)
                        for item in cell
                    )
                    / math.fsum(item.sample_weight for item in cell)
                ),
                fit_log_loss=log_loss,
                fit_brier=brier,
                fit_sample_count=len(cell),
                positive_count=positive,
                negative_count=len(cell) - positive,
            )
        )
    return DeliveryModelFitArtifact(
        model_version=model_version,
        lineage=lineage,
        source_cohort_id=cohort.cohort_id,
        source_cohort_role=cohort.cohort_role,
        source_window_ids=cohort.window_ids,
        source_fold_ids=cohort.fold_ids,
        source_dataset_sha256=cohort.source_dataset_sha256,
        manifest_sha256=cohort.manifest_sha256,
        cohort_identity_sha256=cohort.cohort_identity_sha256,
        split_protocol_sha256=cohort.split_protocol_sha256,
        source_observation_ids=tuple(item.observation_id for item in rows),
        source_case_ids=tuple(item.case_id for item in rows),
        source_cluster_ids=tuple(sorted({item.cluster_id for item in rows})),
        trained_through=max(item.outcome_known_at for item in rows),
        setup_models=tuple(models),
        minimum_supported_coverage=minimum_supported_coverage,
        fit_sample_count=len(rows),
    )


@dataclass(frozen=True)
class SignalOutcomeAdmissionThresholds:
    minimum_resolved_units: int = 200
    minimum_units_per_support: int = 30
    minimum_prediction_coverage: float = 0.95
    minimum_rolling_folds: int = 2
    maximum_ece: float = 0.05
    maximum_log_loss_degradation: float = 0.0
    maximum_brier_degradation: float = 0.0
    minimum_improving_fold_fraction: float = 0.7
    maximum_single_fold_log_loss_degradation: float = 0.05

    def __post_init__(self) -> None:
        if (
            type(self.minimum_resolved_units) is not int
            or self.minimum_resolved_units < 1
            or type(self.minimum_units_per_support) is not int
            or self.minimum_units_per_support < 1
            or type(self.minimum_rolling_folds) is not int
            or self.minimum_rolling_folds < 2
            or not 0.0 <= float(self.minimum_prediction_coverage) <= 1.0
            or not 0.0 <= float(self.maximum_ece) <= 1.0
            or not math.isfinite(float(self.maximum_log_loss_degradation))
            or not math.isfinite(float(self.maximum_brier_degradation))
            or not 0.0 <= float(self.minimum_improving_fold_fraction) <= 1.0
            or not math.isfinite(float(self.maximum_single_fold_log_loss_degradation))
        ):
            raise SignalOutcomeFitError("signal outcome admission thresholds are invalid")


@dataclass(frozen=True)
class SignalOutcomeAdmissionReceipt:
    artifact_kind: SignalOutcomeArtifactKind
    model_artifact_id: str
    fit_cohort_id: str
    fit_cohort_role: str
    fit_cohort_identity_sha256: str
    fit_window_ids: tuple[str, ...]
    fit_fold_ids: tuple[str, ...]
    validation_cohort_id: str
    validation_cohort_role: str
    validation_window_ids: tuple[str, ...]
    validation_fold_ids: tuple[str, ...]
    source_dataset_sha256: str
    manifest_sha256: str
    cohort_identity_sha256: str
    split_protocol_sha256: str
    validation_observation_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    fit_cluster_ids: tuple[str, ...]
    validation_cluster_ids: tuple[str, ...]
    thresholds: SignalOutcomeAdmissionThresholds
    metrics: tuple[tuple[str, float], ...]
    support_units: tuple[tuple[str, int], ...]
    blockers: tuple[str, ...]
    admitted: bool
    status: str
    authority: str = "shadow_evidence_only"
    action_authority: bool = False
    pins_issued: bool = False
    schema_version: str = SIGNAL_OUTCOME_ADMISSION_SCHEMA
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        kind = SignalOutcomeArtifactKind(self.artifact_kind)
        for name in (
            "model_artifact_id",
            "fit_cohort_id",
            "fit_cohort_role",
            "validation_cohort_id",
            "validation_cohort_role",
        ):
            _identity(getattr(self, name), name=f"admission {name}")
        for name in (
            "fit_cohort_identity_sha256",
            "source_dataset_sha256",
            "manifest_sha256",
            "cohort_identity_sha256",
            "split_protocol_sha256",
        ):
            _sha256(getattr(self, name), name=f"admission {name}")
        observations = _ids(
            self.validation_observation_ids,
            name="validation_observation_ids",
        )
        cases = _ids(self.validation_case_ids, name="validation_case_ids")
        fit_clusters = _ids(self.fit_cluster_ids, name="fit_cluster_ids", sort=True)
        validation_clusters = _ids(
            self.validation_cluster_ids,
            name="validation_cluster_ids",
            sort=True,
        )
        fit_windows = _ids(self.fit_window_ids, name="fit_window_ids", sort=True)
        fit_folds = _ids(self.fit_fold_ids, name="fit_fold_ids", sort=True)
        validation_windows = _ids(
            self.validation_window_ids,
            name="validation_window_ids",
            sort=True,
        )
        validation_folds = _ids(
            self.validation_fold_ids,
            name="validation_fold_ids",
            sort=True,
        )
        metrics = tuple(self.metrics)
        supports = tuple(self.support_units)
        blockers = tuple(self.blockers)
        if (
            self.schema_version != SIGNAL_OUTCOME_ADMISSION_SCHEMA
            or not isinstance(self.thresholds, SignalOutcomeAdmissionThresholds)
            or self.fit_cohort_role not in _ALLOWED_COHORT_ROLES
            or self.validation_cohort_role not in _ALLOWED_COHORT_ROLES
            or len(observations) != len(cases)
            or len(validation_clusters) > len(observations)
            or tuple(sorted(metrics)) != metrics
            or len({name for name, _ in metrics}) != len(metrics)
            or any(not math.isfinite(float(value)) for _, value in metrics)
            or tuple(sorted(supports)) != supports
            or any(type(count) is not int or count < 0 for _, count in supports)
            or blockers != tuple(sorted(set(blockers)))
            or self.admitted != (not blockers)
            or self.status != ("admitted_shadow" if self.admitted else "CLOSED")
            or (self.admitted and self.validation_cohort_role != "rolling_oof")
            or (
                self.admitted
                and bool(set(fit_clusters).intersection(validation_clusters))
            )
            or (
                self.admitted
                and any(
                    window in _JUNE_DEVELOPMENT_WINDOWS
                    for window in (*fit_windows, *validation_windows)
                )
            )
            or self.authority != "shadow_evidence_only"
            or self.action_authority is not False
            or self.pins_issued is not False
        ):
            raise SignalOutcomeFitError("signal outcome admission receipt is invalid")
        object.__setattr__(self, "artifact_kind", kind)
        object.__setattr__(self, "fit_window_ids", fit_windows)
        object.__setattr__(self, "fit_fold_ids", fit_folds)
        object.__setattr__(self, "validation_window_ids", validation_windows)
        object.__setattr__(self, "validation_fold_ids", validation_folds)
        object.__setattr__(self, "validation_observation_ids", observations)
        object.__setattr__(self, "validation_case_ids", cases)
        object.__setattr__(self, "fit_cluster_ids", fit_clusters)
        object.__setattr__(self, "validation_cluster_ids", validation_clusters)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "support_units", supports)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(
            self,
            "receipt_id",
            f"signal-outcome-admission:{_digest(self)[:32]}",
        )

    def to_payload(self) -> dict[str, Any]:
        return _normal(self)


def _weighted_ece_binary(
    outcomes: Sequence[bool],
    predictions: Sequence[float],
    weights: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    total = math.fsum(weights)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = tuple(
            row
            for row, prediction in enumerate(predictions)
            if lower <= prediction < upper or (index == bins - 1 and prediction == 1.0)
        )
        if not members:
            continue
        bin_weight = math.fsum(weights[row] for row in members)
        confidence = math.fsum(
            weights[row] * predictions[row] for row in members
        ) / bin_weight
        frequency = math.fsum(
            weights[row] * float(outcomes[row]) for row in members
        ) / bin_weight
        error += bin_weight / total * abs(confidence - frequency)
    return error


def _weighted_ece_dol(
    rows: Sequence[DOLTemperatureObservation],
    temperature: float,
    *,
    bins: int = 10,
) -> float:
    confidences: list[float] = []
    correct: list[bool] = []
    weights: list[float] = []
    for item in rows:
        distribution = _temperature_probabilities(item.outcome_probabilities, temperature)
        label, confidence = max(distribution.items(), key=lambda pair: (pair[1], pair[0]))
        confidences.append(confidence)
        correct.append(label == item.realized_outcome_id)
        weights.append(item.sample_weight)
    return _weighted_ece_binary(correct, confidences, weights, bins=bins)


def _independence_blockers(
    *,
    artifact_cohort_identity_sha256: str,
    artifact_case_ids: Sequence[str],
    artifact_observation_ids: Sequence[str],
    artifact_cluster_ids: Sequence[str],
    artifact_window_ids: Sequence[str],
    artifact_split_protocol_sha256: str,
    trained_through: pd.Timestamp,
    cohort: SignalOutcomeCohort,
    rows: Sequence[DOLTemperatureObservation | DeliveryOutcomeObservation],
    thresholds: SignalOutcomeAdmissionThresholds,
) -> list[str]:
    blockers: list[str] = []
    if cohort.cohort_role != "rolling_oof":
        blockers.append("COHORT_NOT_ROLLING_OOF")
    if cohort.contains_june_development:
        blockers.append("JUNE_2024_DEVELOPMENT_PERMANENTLY_CLOSED")
    if any(window in _JUNE_DEVELOPMENT_WINDOWS for window in artifact_window_ids):
        blockers.append("FIT_SOURCE_JUNE_2024_PERMANENTLY_CLOSED")
    if cohort.cohort_identity_sha256 == artifact_cohort_identity_sha256:
        blockers.append("FIT_VALIDATION_COHORT_NOT_INDEPENDENT")
    if cohort.split_protocol_sha256 != artifact_split_protocol_sha256:
        blockers.append("FIT_VALIDATION_SPLIT_PROTOCOL_MISMATCH")
    if set(artifact_case_ids).intersection(item.case_id for item in rows) or set(
        artifact_observation_ids
    ).intersection(item.observation_id for item in rows):
        blockers.append("FIT_VALIDATION_UNIT_OVERLAP")
    if set(artifact_cluster_ids).intersection(item.cluster_id for item in rows):
        blockers.append("FIT_VALIDATION_CLUSTER_OVERLAP")
    if trained_through >= min(item.prediction_known_at for item in rows):
        blockers.append("FIT_NOT_STRICTLY_BEFORE_VALIDATION_PREDICTIONS")
    if len({item.fold_id for item in rows}) < thresholds.minimum_rolling_folds:
        blockers.append("ROLLING_OOF_FOLD_SUPPORT_INSUFFICIENT")
    if len(rows) < thresholds.minimum_resolved_units:
        blockers.append("RESOLVED_UNIT_SUPPORT_INSUFFICIENT")
    if cohort.prediction_coverage < thresholds.minimum_prediction_coverage:
        blockers.append("PREDICTION_COVERAGE_INSUFFICIENT")
    return blockers


def _quality_blockers(
    *,
    model_log_loss: float,
    baseline_log_loss: float,
    model_brier: float,
    baseline_brier: float,
    ece: float,
    fold_deltas: Sequence[float],
    thresholds: SignalOutcomeAdmissionThresholds,
) -> list[str]:
    blockers: list[str] = []
    if model_log_loss - baseline_log_loss > thresholds.maximum_log_loss_degradation:
        blockers.append("LOG_LOSS_NOT_ACCEPTABLE_VS_BASELINE")
    if model_brier - baseline_brier > thresholds.maximum_brier_degradation:
        blockers.append("BRIER_NOT_ACCEPTABLE_VS_BASELINE")
    if ece > thresholds.maximum_ece:
        blockers.append("ECE_EXCEEDS_THRESHOLD")
    improving_fraction = math.fsum(value < 0.0 for value in fold_deltas) / len(fold_deltas)
    if improving_fraction < thresholds.minimum_improving_fold_fraction:
        blockers.append("ROLLING_FOLD_STABILITY_INSUFFICIENT")
    if max(fold_deltas) > thresholds.maximum_single_fold_log_loss_degradation:
        blockers.append("ROLLING_FOLD_DEGRADATION_EXCEEDED")
    return blockers


def evaluate_dol_temperature_admission(
    artifact: DOLTemperatureFitArtifact,
    validation_observations: Sequence[DOLTemperatureObservation],
    *,
    cohort: SignalOutcomeCohort,
    thresholds: SignalOutcomeAdmissionThresholds = SignalOutcomeAdmissionThresholds(),
) -> SignalOutcomeAdmissionReceipt:
    """Evaluate DOL temperature evidence on a disjoint rolling-OOF cohort."""

    if not isinstance(artifact, DOLTemperatureFitArtifact):
        raise TypeError("artifact must be DOLTemperatureFitArtifact")
    rows = tuple(validation_observations)
    if any(not isinstance(item, DOLTemperatureObservation) for item in rows):
        raise TypeError("DOL admission requires typed observations")
    rows = tuple(sorted(rows, key=lambda item: item.observation_id))
    _validate_cohort_rows(rows, cohort)
    if any(
        item.source_dol_model_fingerprint
        != artifact.lineage.source_dol_model_fingerprint
        for item in rows
    ):
        raise SignalOutcomeFitError("DOL validation observations changed model lineage")
    supports = Counter(item.support_label.value for item in rows)
    blockers = _independence_blockers(
        artifact_cohort_identity_sha256=artifact.cohort_identity_sha256,
        artifact_case_ids=artifact.source_case_ids,
        artifact_observation_ids=artifact.source_observation_ids,
        artifact_cluster_ids=artifact.source_cluster_ids,
        artifact_window_ids=artifact.source_window_ids,
        artifact_split_protocol_sha256=artifact.split_protocol_sha256,
        trained_through=artifact.trained_through,
        cohort=cohort,
        rows=rows,
        thresholds=thresholds,
    )
    for label in (DOLSupportLabel.CANDIDATE_TARGET, DOLSupportLabel.NO_TARGET):
        if supports[label.value] < thresholds.minimum_units_per_support:
            blockers.append(f"SUPPORT_INSUFFICIENT:{label.value}")
    model_log_loss = _weighted_log_loss_dol(rows, artifact.temperature)
    baseline_log_loss = _weighted_log_loss_dol(rows, 1.0)
    model_brier = _weighted_brier_dol(rows, artifact.temperature)
    baseline_brier = _weighted_brier_dol(rows, 1.0)
    ece = _weighted_ece_dol(rows, artifact.temperature)
    fold_deltas = tuple(
        _weighted_log_loss_dol(
            tuple(item for item in rows if item.fold_id == fold),
            artifact.temperature,
        )
        - _weighted_log_loss_dol(
            tuple(item for item in rows if item.fold_id == fold),
            1.0,
        )
        for fold in sorted({item.fold_id for item in rows})
    )
    blockers.extend(
        _quality_blockers(
            model_log_loss=model_log_loss,
            baseline_log_loss=baseline_log_loss,
            model_brier=model_brier,
            baseline_brier=baseline_brier,
            ece=ece,
            fold_deltas=fold_deltas,
            thresholds=thresholds,
        )
    )
    blocker_tuple = tuple(sorted(set(blockers)))
    metrics = tuple(
        sorted(
            {
                "baseline_brier": baseline_brier,
                "baseline_log_loss": baseline_log_loss,
                "improving_fold_fraction": (
                    math.fsum(value < 0.0 for value in fold_deltas) / len(fold_deltas)
                ),
                "maximum_fold_log_loss_delta": max(fold_deltas),
                "model_brier": model_brier,
                "model_log_loss": model_log_loss,
                "prediction_coverage": cohort.prediction_coverage,
                "top_label_ece": ece,
            }.items()
        )
    )
    return SignalOutcomeAdmissionReceipt(
        artifact_kind=SignalOutcomeArtifactKind.DOL_TEMPERATURE,
        model_artifact_id=artifact.artifact_id,
        fit_cohort_id=artifact.source_cohort_id,
        fit_cohort_role=artifact.source_cohort_role,
        fit_cohort_identity_sha256=artifact.cohort_identity_sha256,
        fit_window_ids=artifact.source_window_ids,
        fit_fold_ids=artifact.source_fold_ids,
        validation_cohort_id=cohort.cohort_id,
        validation_cohort_role=cohort.cohort_role,
        validation_window_ids=cohort.window_ids,
        validation_fold_ids=cohort.fold_ids,
        source_dataset_sha256=cohort.source_dataset_sha256,
        manifest_sha256=cohort.manifest_sha256,
        cohort_identity_sha256=cohort.cohort_identity_sha256,
        split_protocol_sha256=cohort.split_protocol_sha256,
        validation_observation_ids=tuple(item.observation_id for item in rows),
        validation_case_ids=tuple(item.case_id for item in rows),
        fit_cluster_ids=artifact.source_cluster_ids,
        validation_cluster_ids=tuple(sorted({item.cluster_id for item in rows})),
        thresholds=thresholds,
        metrics=metrics,
        support_units=tuple(sorted(supports.items())),
        blockers=blocker_tuple,
        admitted=not blocker_tuple,
        status="admitted_shadow" if not blocker_tuple else "CLOSED",
    )


def _delivery_predictions(
    artifact: DeliveryModelFitArtifact,
    rows: Sequence[DeliveryOutcomeObservation],
) -> tuple[float, ...]:
    return tuple(
        artifact.model_for(item.setup_family).estimate(
            p_path=item.p_path,
            p_dol=item.p_dol,
        )
        for item in rows
    )


def _delivery_baselines(
    artifact: DeliveryModelFitArtifact,
    rows: Sequence[DeliveryOutcomeObservation],
) -> tuple[float, ...]:
    return tuple(
        artifact.model_for(item.setup_family).fit_prevalence for item in rows
    )


def evaluate_delivery_model_admission(
    artifact: DeliveryModelFitArtifact,
    validation_observations: Sequence[DeliveryOutcomeObservation],
    *,
    cohort: SignalOutcomeCohort,
    thresholds: SignalOutcomeAdmissionThresholds = SignalOutcomeAdmissionThresholds(),
) -> SignalOutcomeAdmissionReceipt:
    """Evaluate delivery logits independently on disjoint rolling OOF."""

    if not isinstance(artifact, DeliveryModelFitArtifact):
        raise TypeError("artifact must be DeliveryModelFitArtifact")
    rows = tuple(validation_observations)
    if any(not isinstance(item, DeliveryOutcomeObservation) for item in rows):
        raise TypeError("delivery admission requires typed observations")
    rows = tuple(sorted(rows, key=lambda item: item.observation_id))
    _validate_cohort_rows(rows, cohort)
    if any(
        item.path_likelihood_artifact_id
        != artifact.lineage.path_likelihood_artifact_id
        or item.dol_calibration_fit_artifact_id
        != artifact.lineage.dol_calibration_fit_artifact_id
        for item in rows
    ):
        raise SignalOutcomeFitError("delivery validation observations changed lineage")
    supports = Counter(
        f"{item.setup_family.value}:{'positive' if item.target_before_invalidation else 'negative'}"
        for item in rows
    )
    blockers = _independence_blockers(
        artifact_cohort_identity_sha256=artifact.cohort_identity_sha256,
        artifact_case_ids=artifact.source_case_ids,
        artifact_observation_ids=artifact.source_observation_ids,
        artifact_cluster_ids=artifact.source_cluster_ids,
        artifact_window_ids=artifact.source_window_ids,
        artifact_split_protocol_sha256=artifact.split_protocol_sha256,
        trained_through=artifact.trained_through,
        cohort=cohort,
        rows=rows,
        thresholds=thresholds,
    )
    required_support = tuple(
        f"{setup.value}:{outcome}"
        for setup in (SetupFamily.DFP, SetupFamily.LSR)
        for outcome in ("negative", "positive")
    )
    for label in required_support:
        if supports[label] < thresholds.minimum_units_per_support:
            blockers.append(f"SUPPORT_INSUFFICIENT:{label}")
    predictions = _delivery_predictions(artifact, rows)
    baselines = _delivery_baselines(artifact, rows)
    outcomes = tuple(item.target_before_invalidation for item in rows)
    weights = tuple(item.sample_weight for item in rows)
    model_log_loss, model_brier = _weighted_binary_metrics(
        outcomes,
        predictions,
        weights,
    )
    baseline_log_loss, baseline_brier = _weighted_binary_metrics(
        outcomes,
        baselines,
        weights,
    )
    ece = _weighted_ece_binary(outcomes, predictions, weights)
    fold_deltas: list[float] = []
    for fold in sorted({item.fold_id for item in rows}):
        positions = tuple(index for index, item in enumerate(rows) if item.fold_id == fold)
        fold_outcomes = tuple(outcomes[index] for index in positions)
        fold_weights = tuple(weights[index] for index in positions)
        fold_model, _ = _weighted_binary_metrics(
            fold_outcomes,
            tuple(predictions[index] for index in positions),
            fold_weights,
        )
        fold_baseline, _ = _weighted_binary_metrics(
            fold_outcomes,
            tuple(baselines[index] for index in positions),
            fold_weights,
        )
        fold_deltas.append(fold_model - fold_baseline)
    blockers.extend(
        _quality_blockers(
            model_log_loss=model_log_loss,
            baseline_log_loss=baseline_log_loss,
            model_brier=model_brier,
            baseline_brier=baseline_brier,
            ece=ece,
            fold_deltas=fold_deltas,
            thresholds=thresholds,
        )
    )
    blocker_tuple = tuple(sorted(set(blockers)))
    metrics = tuple(
        sorted(
            {
                "baseline_brier": baseline_brier,
                "baseline_log_loss": baseline_log_loss,
                "improving_fold_fraction": (
                    math.fsum(value < 0.0 for value in fold_deltas)
                    / len(fold_deltas)
                ),
                "maximum_fold_log_loss_delta": max(fold_deltas),
                "model_brier": model_brier,
                "model_log_loss": model_log_loss,
                "prediction_coverage": cohort.prediction_coverage,
                "top_label_ece": ece,
            }.items()
        )
    )
    return SignalOutcomeAdmissionReceipt(
        artifact_kind=SignalOutcomeArtifactKind.DELIVERY_MODEL,
        model_artifact_id=artifact.artifact_id,
        fit_cohort_id=artifact.source_cohort_id,
        fit_cohort_role=artifact.source_cohort_role,
        fit_cohort_identity_sha256=artifact.cohort_identity_sha256,
        fit_window_ids=artifact.source_window_ids,
        fit_fold_ids=artifact.source_fold_ids,
        validation_cohort_id=cohort.cohort_id,
        validation_cohort_role=cohort.cohort_role,
        validation_window_ids=cohort.window_ids,
        validation_fold_ids=cohort.fold_ids,
        source_dataset_sha256=cohort.source_dataset_sha256,
        manifest_sha256=cohort.manifest_sha256,
        cohort_identity_sha256=cohort.cohort_identity_sha256,
        split_protocol_sha256=cohort.split_protocol_sha256,
        validation_observation_ids=tuple(item.observation_id for item in rows),
        validation_case_ids=tuple(item.case_id for item in rows),
        fit_cluster_ids=artifact.source_cluster_ids,
        validation_cluster_ids=tuple(sorted({item.cluster_id for item in rows})),
        thresholds=thresholds,
        metrics=metrics,
        support_units=tuple(sorted(supports.items())),
        blockers=blocker_tuple,
        admitted=not blocker_tuple,
        status="admitted_shadow" if not blocker_tuple else "CLOSED",
    )


__all__ = [
    "DELIVERY_MODEL_FIT_SCHEMA",
    "DOL_TEMPERATURE_FIT_SCHEMA",
    "SIGNAL_OUTCOME_ADMISSION_SCHEMA",
    "SIGNAL_OUTCOME_FIT_SCHEMA_VERSION",
    "DOLModelLineage",
    "DOLSupportLabel",
    "DOLTemperatureFitArtifact",
    "DOLTemperatureObservation",
    "DeliveryModelFitArtifact",
    "DeliveryModelLineage",
    "DeliveryOutcomeObservation",
    "DeliverySetupModelFit",
    "RealCompletedBarEvidence",
    "SignalOutcomeAdmissionReceipt",
    "SignalOutcomeAdmissionThresholds",
    "SignalOutcomeArtifactKind",
    "SignalOutcomeCohort",
    "SignalOutcomeFitError",
    "evaluate_delivery_model_admission",
    "evaluate_dol_temperature_admission",
    "fit_delivery_model",
    "fit_dol_temperature",
]
