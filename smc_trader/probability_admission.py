"""Validation, rolling-OOF and admission gates for Phase-7 probabilities.

The module evaluates already-produced predictions.  It cannot fit a model,
open a sealed source, or authorize action.  Admission receipts are always
shadow-only and bind the exact model, manifest and cohort identities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from smc_trader.probability_cohorts import (
    NO_TARGET_OUTCOME,
    aware_timestamp,
    canonical_identity,
    canonical_sha256,
)


PROBABILITY_ADMISSION_SCHEMA_VERSION = 1
FIRST_HIT_SUPPORT = "first_hit_candidate"


def _sha256(value: str, *, name: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


@dataclass(frozen=True)
class RollingSession:
    session_id: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp

    def __post_init__(self) -> None:
        start = aware_timestamp(self.start, name="rolling session start")
        end = aware_timestamp(self.end_exclusive, name="rolling session end")
        if not self.session_id or end <= start:
            raise ValueError("rolling session is invalid")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)


@dataclass(frozen=True)
class RollingAssignment:
    competition_set_id: str
    market_epoch_id: str
    session_id: str
    prediction_known_at: pd.Timestamp
    outcome_known_at: pd.Timestamp
    role: str
    fold_id: str

    def __post_init__(self) -> None:
        prediction = aware_timestamp(
            self.prediction_known_at,
            name="rolling prediction_known_at",
        )
        outcome = aware_timestamp(
            self.outcome_known_at,
            name="rolling outcome_known_at",
        )
        role = str(self.role).strip()
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.competition_set_id,
                    self.market_epoch_id,
                    self.session_id,
                    self.fold_id,
                )
            )
            or role not in {"fit", "validation"}
            or outcome < prediction
        ):
            raise ValueError("rolling assignment is invalid")
        object.__setattr__(self, "prediction_known_at", prediction)
        object.__setattr__(self, "outcome_known_at", outcome)
        object.__setattr__(self, "role", role)


def validate_rolling_group_session_split(
    assignments: Sequence[RollingAssignment],
    sessions: Sequence[RollingSession],
    *,
    purge_calendar_days: int = 14,
    embargo_trading_sessions: int = 5,
) -> None:
    """Validate expanding folds at whole competition and session grain."""

    typed = tuple(assignments)
    calendar = tuple(sessions)
    if (
        not typed
        or any(not isinstance(item, RollingAssignment) for item in typed)
        or not calendar
        or any(not isinstance(item, RollingSession) for item in calendar)
        or type(purge_calendar_days) is not int
        or purge_calendar_days < 0
        or type(embargo_trading_sessions) is not int
        or embargo_trading_sessions < 0
    ):
        raise ValueError("rolling split inputs are invalid")
    session_keys = tuple((item.start, item.session_id) for item in calendar)
    if (
        session_keys != tuple(sorted(session_keys))
        or len({item.session_id for item in calendar}) != len(calendar)
        or any(
            left.end_exclusive > right.start
            for left, right in zip(calendar, calendar[1:])
        )
    ):
        raise ValueError("rolling session calendar overlaps or is unordered")
    by_session = {item.session_id: item for item in calendar}
    session_index = {item.session_id: index for index, item in enumerate(calendar)}
    for item in typed:
        session = by_session.get(item.session_id)
        if session is None:
            raise ValueError("rolling assignment references an unknown session")
        if not session.start <= item.prediction_known_at < session.end_exclusive:
            raise ValueError("rolling prediction clock is outside its session")

    fold_order: list[tuple[pd.Timestamp, str]] = []
    for fold_id in sorted({item.fold_id for item in typed}):
        fold = tuple(item for item in typed if item.fold_id == fold_id)
        fit = tuple(item for item in fold if item.role == "fit")
        validation = tuple(item for item in fold if item.role == "validation")
        if not fit or not validation:
            raise ValueError("every rolling fold requires fit and validation units")
        owners: dict[tuple[str, str], str] = {}
        session_owners: dict[str, str] = {}
        for item in fold:
            group = (item.market_epoch_id, item.competition_set_id)
            prior = owners.setdefault(group, item.role)
            if prior != item.role:
                raise ValueError("competition generation crosses fold roles")
            prior_session = session_owners.setdefault(item.session_id, item.role)
            if prior_session != item.role:
                raise ValueError("market session crosses fold roles")
        validation_start = min(by_session[item.session_id].start for item in validation)
        first_validation_index = min(session_index[item.session_id] for item in validation)
        fold_order.append((validation_start, fold_id))
        if any(item.prediction_known_at >= validation_start for item in fit):
            raise ValueError("rolling fit contains a non-prior prediction")
        purge_boundary = validation_start - pd.DateOffset(days=purge_calendar_days)
        if any(item.outcome_known_at > purge_boundary for item in fit):
            raise ValueError("rolling fit violates the registered outcome purge")
        if first_validation_index < embargo_trading_sessions:
            raise ValueError("rolling calendar cannot establish the required embargo")
        embargo_ids = {
            calendar[index].session_id
            for index in range(
                first_validation_index - embargo_trading_sessions,
                first_validation_index,
            )
        }
        if any(item.session_id in embargo_ids for item in fit):
            raise ValueError("rolling fit uses an embargoed trading session")
    if fold_order != sorted(fold_order) or len(fold_order) != len(set(fold_order)):
        raise ValueError("rolling validation folds are not chronologically unique")


def _probability_mapping(
    values: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    name: str,
    tolerance: float,
) -> tuple[tuple[str, float], ...]:
    source = dict(values)
    normalized = tuple(sorted((str(label), float(value)) for label, value in source.items()))
    if (
        not normalized
        or any(not label or not math.isfinite(value) or value < 0.0 for label, value in normalized)
        or not math.isclose(
            math.fsum(value for _, value in normalized),
            1.0,
            abs_tol=tolerance,
        )
    ):
        raise ValueError(f"{name} is not a complete probability distribution")
    return normalized


@dataclass(frozen=True)
class ProbabilityPrediction:
    unit_id: str
    cluster_id: str
    fold_id: str
    realized_label: str
    support_label: str
    model_probabilities: tuple[tuple[str, float], ...]
    baseline_probabilities: tuple[tuple[str, float], ...]
    probability_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.unit_id,
                    self.cluster_id,
                    self.fold_id,
                    self.realized_label,
                    self.support_label,
                )
            )
            or not math.isfinite(self.probability_tolerance)
            or self.probability_tolerance <= 0.0
        ):
            raise ValueError("probability prediction identity is invalid")
        model = _probability_mapping(
            self.model_probabilities,
            name="model probabilities",
            tolerance=self.probability_tolerance,
        )
        baseline = _probability_mapping(
            self.baseline_probabilities,
            name="baseline probabilities",
            tolerance=self.probability_tolerance,
        )
        if self.realized_label not in dict(model) or self.realized_label not in dict(baseline):
            raise ValueError("realized label is absent from a prediction distribution")
        object.__setattr__(self, "model_probabilities", model)
        object.__setattr__(self, "baseline_probabilities", baseline)


def dol_support_label(realized_label: str) -> str:
    return NO_TARGET_OUTCOME if realized_label == NO_TARGET_OUTCOME else FIRST_HIT_SUPPORT


def _unit_loss_values(
    records: Sequence[ProbabilityPrediction],
    *,
    model: bool,
) -> dict[str, float]:
    by_unit: defaultdict[str, list[float]] = defaultdict(list)
    for record in records:
        probabilities = dict(
            record.model_probabilities if model else record.baseline_probabilities
        )
        by_unit[record.unit_id].append(
            -math.log(max(probabilities[record.realized_label], 1e-300))
        )
    return {
        unit: math.fsum(values) / len(values) for unit, values in by_unit.items()
    }


def multiclass_log_loss(
    records: Sequence[ProbabilityPrediction],
    *,
    model: bool = True,
) -> float:
    values = _unit_loss_values(records, model=model)
    if not values:
        raise ValueError("log loss requires predictions")
    return math.fsum(values.values()) / len(values)


def multiclass_brier(
    records: Sequence[ProbabilityPrediction],
    *,
    model: bool = True,
) -> float:
    by_unit: defaultdict[str, list[float]] = defaultdict(list)
    for record in records:
        probabilities = dict(
            record.model_probabilities if model else record.baseline_probabilities
        )
        by_unit[record.unit_id].append(
            math.fsum(
                (probability - (1.0 if label == record.realized_label else 0.0)) ** 2
                for label, probability in probabilities.items()
            )
        )
    if not by_unit:
        raise ValueError("Brier score requires predictions")
    unit_values = [math.fsum(values) / len(values) for values in by_unit.values()]
    return math.fsum(unit_values) / len(unit_values)


def top_label_ece(
    records: Sequence[ProbabilityPrediction],
    *,
    maximum_bins: int = 10,
    minimum_bin_units: int = 30,
) -> float:
    if maximum_bins < 1 or minimum_bin_units < 1:
        raise ValueError("ECE bin contract is invalid")
    by_unit: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        probabilities = dict(record.model_probabilities)
        label, confidence = max(
            probabilities.items(),
            key=lambda item: (item[1], item[0]),
        )
        by_unit[record.unit_id].append(
            (float(confidence), 1.0 if label == record.realized_label else 0.0)
        )
    if not by_unit:
        raise ValueError("ECE requires predictions")
    units = sorted(
        (
            math.fsum(value[0] for value in values) / len(values),
            math.fsum(value[1] for value in values) / len(values),
            unit,
        )
        for unit, values in by_unit.items()
    )
    bin_count = max(1, min(maximum_bins, len(units) // minimum_bin_units))
    bins = np.array_split(np.asarray(units, dtype=object), bin_count)
    total = len(units)
    return math.fsum(
        (len(bucket) / total)
        * abs(
            math.fsum(float(row[0]) for row in bucket) / len(bucket)
            - math.fsum(float(row[1]) for row in bucket) / len(bucket)
        )
        for bucket in bins
        if len(bucket)
    )


@dataclass(frozen=True)
class BootstrapDelta:
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    replicates: int
    seed: int

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(value)
                for value in (self.point_estimate, self.ci_low, self.ci_high)
            )
            or not 0.0 < self.confidence < 1.0
            or self.replicates < 1
            or self.ci_low > self.ci_high
        ):
            raise ValueError("bootstrap delta is invalid")


def cluster_bootstrap_log_loss_delta(
    records: Sequence[ProbabilityPrediction],
    *,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 731,
) -> BootstrapDelta:
    typed = tuple(records)
    if (
        not typed
        or any(not isinstance(item, ProbabilityPrediction) for item in typed)
        or type(replicates) is not int
        or replicates < 1
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("cluster bootstrap contract is invalid")
    model = _unit_loss_values(typed, model=True)
    baseline = _unit_loss_values(typed, model=False)
    clusters_by_unit: dict[str, str] = {}
    for record in typed:
        prior = clusters_by_unit.setdefault(record.unit_id, record.cluster_id)
        if prior != record.cluster_id:
            raise ValueError("one probability unit crosses bootstrap clusters")
    units_by_cluster: defaultdict[str, list[str]] = defaultdict(list)
    for unit, cluster in clusters_by_unit.items():
        units_by_cluster[cluster].append(unit)
    clusters = tuple(sorted(units_by_cluster))
    if len(clusters) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    unit_delta = {unit: model[unit] - baseline[unit] for unit in model}
    point = math.fsum(unit_delta.values()) / len(unit_delta)
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = generator.choice(clusters, size=len(clusters), replace=True)
        values = [
            unit_delta[unit]
            for cluster in sampled
            for unit in units_by_cluster[str(cluster)]
        ]
        draws[index] = math.fsum(values) / len(values)
    tail = (1.0 - confidence) / 2.0
    return BootstrapDelta(
        point_estimate=point,
        ci_low=float(np.quantile(draws, tail)),
        ci_high=float(np.quantile(draws, 1.0 - tail)),
        confidence=confidence,
        replicates=replicates,
        seed=seed,
    )


@dataclass(frozen=True)
class AdmissionThresholds:
    minimum_resolved_units: int = 200
    minimum_support_units: int = 30
    minimum_prediction_coverage: float = 0.95
    maximum_ece: float = 0.05
    maximum_fold_log_loss_degradation: float = 0.05
    minimum_improving_fold_fraction: float = 0.70
    bootstrap_confidence: float = 0.95
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 731
    probability_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if (
            self.minimum_resolved_units < 1
            or self.minimum_support_units < 1
            or not 0.0 < self.minimum_prediction_coverage <= 1.0
            or not 0.0 <= self.maximum_ece <= 1.0
            or self.maximum_fold_log_loss_degradation < 0.0
            or not 0.0 <= self.minimum_improving_fold_fraction <= 1.0
            or not 0.0 < self.bootstrap_confidence < 1.0
            or self.bootstrap_replicates < 1
            or self.probability_tolerance <= 0.0
        ):
            raise ValueError("admission thresholds are invalid")


@dataclass(frozen=True)
class ProbabilityAdmissionReceipt:
    artifact_kind: str
    model_artifact_id: str
    source_dataset_sha256: str
    manifest_sha256: str
    cohort_identity_sha256: str
    cohort_role: str
    thresholds: AdmissionThresholds
    metrics: tuple[tuple[str, float], ...]
    support_units: tuple[tuple[str, int], ...]
    blockers: tuple[str, ...]
    admitted: bool
    status: str
    authority: str = "shadow_only"
    action_authority: bool = False
    schema_version: int = PROBABILITY_ADMISSION_SCHEMA_VERSION
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_dataset_sha256, name="source_dataset_sha256")
        _sha256(self.manifest_sha256, name="manifest_sha256")
        _sha256(self.cohort_identity_sha256, name="cohort_identity_sha256")
        if (
            self.schema_version != PROBABILITY_ADMISSION_SCHEMA_VERSION
            or self.artifact_kind not in {"path_probability", "dol_probability"}
            or not self.model_artifact_id
            or self.cohort_role not in {
                "development_cross_fit",
                "historical_validation",
                "rolling_oof",
            }
            or self.authority != "shadow_only"
            or self.action_authority is not False
            or self.status
            != ("admitted_shadow" if self.admitted else "rejected_shadow")
            or self.admitted != (not self.blockers)
            or tuple(self.blockers) != tuple(sorted(set(self.blockers)))
        ):
            raise ValueError("probability admission receipt is invalid")
        payload = {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "model_artifact_id": self.model_artifact_id,
            "source_dataset_sha256": self.source_dataset_sha256,
            "manifest_sha256": self.manifest_sha256,
            "cohort_identity_sha256": self.cohort_identity_sha256,
            "cohort_role": self.cohort_role,
            "thresholds": self.thresholds,
            "metrics": self.metrics,
            "support_units": self.support_units,
            "blockers": self.blockers,
            "admitted": self.admitted,
            "status": self.status,
            "authority": self.authority,
            "action_authority": self.action_authority,
        }
        object.__setattr__(
            self,
            "receipt_id",
            canonical_identity("probability-admission", payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["receipt_id"] = self.receipt_id
        return payload


def _fold_deltas(records: Sequence[ProbabilityPrediction]) -> tuple[float, ...]:
    values = []
    for fold_id in sorted({item.fold_id for item in records}):
        fold = tuple(item for item in records if item.fold_id == fold_id)
        values.append(
            multiclass_log_loss(fold, model=True)
            - multiclass_log_loss(fold, model=False)
        )
    return tuple(values)


def evaluate_probability_admission(
    records: Sequence[ProbabilityPrediction],
    *,
    artifact_kind: str,
    model_artifact_id: str,
    source_dataset_sha256: str,
    manifest_sha256: str,
    cohort_identity_sha256: str,
    cohort_role: str,
    prediction_coverage: float,
    required_support_labels: Sequence[str],
    thresholds: AdmissionThresholds = AdmissionThresholds(),
) -> ProbabilityAdmissionReceipt:
    """Evaluate the preregistered shadow-admission gates."""

    typed = tuple(records)
    if not typed or any(not isinstance(item, ProbabilityPrediction) for item in typed):
        raise ValueError("probability admission requires typed predictions")
    if artifact_kind not in {"path_probability", "dol_probability"}:
        raise ValueError("unsupported probability artifact kind")
    if not math.isfinite(prediction_coverage) or not 0.0 <= prediction_coverage <= 1.0:
        raise ValueError("prediction coverage must lie in [0, 1]")
    unit_outcomes: dict[str, tuple[str, str]] = {}
    for item in typed:
        prior = unit_outcomes.setdefault(
            item.unit_id,
            (item.realized_label, item.support_label),
        )
        if prior != (item.realized_label, item.support_label):
            raise ValueError("one probability unit has conflicting outcomes")
    supports = Counter(value[1] for value in unit_outcomes.values())
    model_log_loss = multiclass_log_loss(typed, model=True)
    baseline_log_loss = multiclass_log_loss(typed, model=False)
    model_brier = multiclass_brier(typed, model=True)
    baseline_brier = multiclass_brier(typed, model=False)
    ece = top_label_ece(typed)
    bootstrap = cluster_bootstrap_log_loss_delta(
        typed,
        replicates=thresholds.bootstrap_replicates,
        confidence=thresholds.bootstrap_confidence,
        seed=thresholds.bootstrap_seed,
    )
    fold_deltas = _fold_deltas(typed)
    improving_fraction = math.fsum(value < 0.0 for value in fold_deltas) / len(
        fold_deltas
    )
    blockers: list[str] = []
    if cohort_role != "rolling_oof":
        blockers.append("COHORT_NOT_ROLLING_OOF")
    if len(unit_outcomes) < thresholds.minimum_resolved_units:
        blockers.append("RESOLVED_UNIT_SUPPORT_INSUFFICIENT")
    for label in sorted(set(required_support_labels)):
        if supports[label] < thresholds.minimum_support_units:
            blockers.append(f"SUPPORT_INSUFFICIENT:{label}")
    if prediction_coverage < thresholds.minimum_prediction_coverage:
        blockers.append("PREDICTION_COVERAGE_INSUFFICIENT")
    if bootstrap.ci_high >= 0.0:
        blockers.append("LOG_LOSS_BOOTSTRAP_NOT_SUPERIOR")
    if model_brier > baseline_brier:
        blockers.append("BRIER_WORSE_THAN_BASELINE")
    if ece > thresholds.maximum_ece:
        blockers.append("ECE_EXCEEDS_THRESHOLD")
    if improving_fraction < thresholds.minimum_improving_fold_fraction:
        blockers.append("ROLLING_FOLD_STABILITY_INSUFFICIENT")
    if max(fold_deltas) > thresholds.maximum_fold_log_loss_degradation:
        blockers.append("ROLLING_FOLD_DEGRADATION_EXCEEDED")
    blocker_tuple = tuple(sorted(set(blockers)))
    metrics = tuple(
        sorted(
            {
                "baseline_brier": baseline_brier,
                "baseline_log_loss": baseline_log_loss,
                "bootstrap_delta_ci_high": bootstrap.ci_high,
                "bootstrap_delta_ci_low": bootstrap.ci_low,
                "bootstrap_delta_point": bootstrap.point_estimate,
                "improving_fold_fraction": improving_fraction,
                "maximum_fold_log_loss_delta": max(fold_deltas),
                "model_brier": model_brier,
                "model_log_loss": model_log_loss,
                "prediction_coverage": float(prediction_coverage),
                "top_label_ece": ece,
            }.items()
        )
    )
    admitted = not blocker_tuple
    return ProbabilityAdmissionReceipt(
        artifact_kind=artifact_kind,
        model_artifact_id=model_artifact_id,
        source_dataset_sha256=_sha256(
            source_dataset_sha256,
            name="source_dataset_sha256",
        ),
        manifest_sha256=_sha256(manifest_sha256, name="manifest_sha256"),
        cohort_identity_sha256=_sha256(
            cohort_identity_sha256,
            name="cohort_identity_sha256",
        ),
        cohort_role=cohort_role,
        thresholds=thresholds,
        metrics=metrics,
        support_units=tuple(sorted(supports.items())),
        blockers=blocker_tuple,
        admitted=admitted,
        status="admitted_shadow" if admitted else "rejected_shadow",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON-serializable: {type(value).__name__}")


def write_json_no_overwrite(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Create one canonical JSON artifact atomically and never replace a file."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = _jsonable(payload)
    if not isinstance(normalized, Mapping):
        raise TypeError("JSON artifact payload must be a mapping")
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = canonical_sha256(normalized)
    temporary = destination.with_name(f".{destination.name}.{digest}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


__all__ = [
    "AdmissionThresholds",
    "BootstrapDelta",
    "FIRST_HIT_SUPPORT",
    "PROBABILITY_ADMISSION_SCHEMA_VERSION",
    "ProbabilityAdmissionReceipt",
    "ProbabilityPrediction",
    "RollingAssignment",
    "RollingSession",
    "cluster_bootstrap_log_loss_delta",
    "dol_support_label",
    "evaluate_probability_admission",
    "multiclass_brier",
    "multiclass_log_loss",
    "top_label_ece",
    "validate_rolling_group_session_split",
    "write_json_no_overwrite",
]
