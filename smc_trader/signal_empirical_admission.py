"""Strict fitted-to-signal admission boundary.

The production signal DTOs live in :mod:`smc_trader.signal_policy`, while the
current probability fitters deliberately emit research artifacts with no
deployment authority.  This module performs exact, externally bound mappings
for path calibration, DOL temperature calibration, and the DFP/LSR
target-before-invalidation model after independent rolling-OOF receipts admit
their evidence.

It never creates ``SignalArtifactPins``.  The converted DTOs remain
``shadow_only`` with action authority disabled, and a complete signal bundle
therefore stays closed until a separate external pinning step.  The previously
opened 2024-06 W1/W2 data remain permanently ineligible for admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

import pandas as pd

from .path_belief import PATH_KINDS, PathBeliefProtocol, PathKind
from .probability_admission import ProbabilityAdmissionReceipt
from .probability_fit import PathCalibrationArtifact
from .signal_outcome_fit import (
    DOLTemperatureFitArtifact,
    DeliveryModelFitArtifact,
    SignalOutcomeAdmissionReceipt,
    SignalOutcomeArtifactKind,
)
from .signal_policy import (
    AdmittedDOLCalibrationArtifact,
    AdmittedPathLikelihoodArtifact,
    SetupDeliveryModel,
    TargetBeforeInvalidationArtifact,
)


SIGNAL_EMPIRICAL_ADMISSION_SCHEMA_VERSION = 1
_JUNE_DEVELOPMENT_WINDOWS = frozenset({"W1", "W2", "2024-06-week-1", "2024-06-week-2"})
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


class SignalEmpiricalAdmissionError(ValueError):
    """Raised when an empirical conversion is not externally admissible."""


class SignalEmpiricalBlocker(str, Enum):
    SOURCE_NOT_ROLLING_OOF = "source_not_rolling_oof"
    PATH_FIT_MISSING = "path_fit_missing"
    PATH_RECEIPT_MISSING = "path_receipt_missing"
    PATH_FIT_PIN_MISMATCH = "path_fit_pin_mismatch"
    PATH_RECEIPT_PIN_MISMATCH = "path_receipt_pin_mismatch"
    PATH_SOURCE_HASH_MISMATCH = "path_source_hash_mismatch"
    PATH_PROTOCOL_MISMATCH = "path_protocol_mismatch"
    PATH_CALIBRATION_GAUGE_MISMATCH = "path_calibration_gauge_mismatch"
    PATH_RECEIPT_NOT_ADMITTED = "path_receipt_not_admitted"
    OUTCOME_BINDING_MISSING = "outcome_binding_missing"
    DOL_FIT_MISSING = "dol_fit_missing"
    DOL_RECEIPT_MISSING = "dol_receipt_missing"
    DOL_FIT_PIN_MISMATCH = "dol_fit_pin_mismatch"
    DOL_RECEIPT_PIN_MISMATCH = "dol_receipt_pin_mismatch"
    DOL_RECEIPT_NOT_ADMITTED = "dol_receipt_not_admitted"
    DOL_SOURCE_BINDING_MISMATCH = "dol_source_binding_mismatch"
    DOL_JUNE_SOURCE_CLOSED = "dol_june_source_closed"
    DELIVERY_FIT_MISSING = "delivery_fit_missing"
    DELIVERY_RECEIPT_MISSING = "delivery_receipt_missing"
    DELIVERY_FIT_PIN_MISMATCH = "delivery_fit_pin_mismatch"
    DELIVERY_RECEIPT_PIN_MISMATCH = "delivery_receipt_pin_mismatch"
    DELIVERY_RECEIPT_NOT_ADMITTED = "delivery_receipt_not_admitted"
    DELIVERY_SOURCE_BINDING_MISMATCH = "delivery_source_binding_mismatch"
    DELIVERY_JUNE_SOURCE_CLOSED = "delivery_june_source_closed"
    DELIVERY_PATH_ARTIFACT_MISMATCH = "delivery_path_artifact_mismatch"
    DELIVERY_DOL_ARTIFACT_MISMATCH = "delivery_dol_artifact_mismatch"
    OUTCOME_VALIDITY_CLOCK_MISMATCH = "outcome_validity_clock_mismatch"
    DOL_CALIBRATION_FIT_SCHEMA_MISSING = "dol_calibration_fit_schema_missing"
    TARGET_BEFORE_INVALIDATION_FIT_SCHEMA_MISSING = (
        "target_before_invalidation_fit_schema_missing"
    )
    EXTERNAL_SIGNAL_ARTIFACT_PINS_MISSING = "external_signal_artifact_pins_missing"


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty identity")
    return value


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a timestamp") from error
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class EmpiricalPathAdmissionBinding:
    """External exact pins and validity clocks for one path conversion."""

    window_id: str
    cohort_role: str
    source_dataset_sha256: str
    manifest_sha256: str
    cohort_identity_sha256: str
    expected_fit_artifact_id: str
    expected_admission_receipt_id: str
    expected_path_protocol_fingerprint: str
    expected_path_model_version: str
    trained_through: pd.Timestamp
    valid_from: pd.Timestamp
    expires_at: pd.Timestamp
    sealed_oos_opened: bool = False
    authority: str = "external_shadow_admission_binding"
    action_authority: bool = False
    schema_version: int = SIGNAL_EMPIRICAL_ADMISSION_SCHEMA_VERSION
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        window_id = _identity(self.window_id, name="window_id")
        role = _identity(self.cohort_role, name="cohort_role")
        identities = (
            _identity(
                self.expected_fit_artifact_id,
                name="expected_fit_artifact_id",
            ),
            _identity(
                self.expected_admission_receipt_id,
                name="expected_admission_receipt_id",
            ),
            _identity(
                self.expected_path_model_version,
                name="expected_path_model_version",
            ),
        )
        hashes = (
            _sha256(self.source_dataset_sha256, name="source_dataset_sha256"),
            _sha256(self.manifest_sha256, name="manifest_sha256"),
            _sha256(
                self.cohort_identity_sha256,
                name="cohort_identity_sha256",
            ),
            _sha256(
                self.expected_path_protocol_fingerprint,
                name="expected_path_protocol_fingerprint",
            ),
        )
        trained = _aware(self.trained_through, name="trained_through")
        valid = _aware(self.valid_from, name="valid_from")
        expires = _aware(self.expires_at, name="expires_at")
        if (
            self.schema_version != SIGNAL_EMPIRICAL_ADMISSION_SCHEMA_VERSION
            or role not in _ALLOWED_COHORT_ROLES
            or self.authority != "external_shadow_admission_binding"
            or self.action_authority is not False
            or type(self.sealed_oos_opened) is not bool
            or self.sealed_oos_opened
            or not trained < valid < expires
        ):
            raise ValueError("empirical path admission binding is invalid")
        if window_id in _JUNE_DEVELOPMENT_WINDOWS and role == "rolling_oof":
            raise ValueError("2024-06 W1/W2 cannot be relabelled rolling_oof")
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "cohort_role", role)
        object.__setattr__(self, "trained_through", trained)
        object.__setattr__(self, "valid_from", valid)
        object.__setattr__(self, "expires_at", expires)
        payload = {
            "schema_version": self.schema_version,
            "window_id": window_id,
            "cohort_role": role,
            "source_dataset_sha256": hashes[0],
            "manifest_sha256": hashes[1],
            "cohort_identity_sha256": hashes[2],
            "expected_fit_artifact_id": identities[0],
            "expected_admission_receipt_id": identities[1],
            "expected_path_protocol_fingerprint": hashes[3],
            "expected_path_model_version": identities[2],
            "trained_through": trained.isoformat(),
            "valid_from": valid.isoformat(),
            "expires_at": expires.isoformat(),
            "sealed_oos_opened": False,
            "authority": self.authority,
            "action_authority": False,
        }
        object.__setattr__(
            self,
            "binding_id",
            f"signal-empirical-path-binding:{_canonical_sha256(payload)[:32]}",
        )


@dataclass(frozen=True)
class EmpiricalSignalOutcomeAdmissionBinding:
    """External identity pins and validity clocks for DOL/delivery mappings."""

    expected_dol_fit_artifact_id: str
    expected_dol_admission_receipt_id: str
    expected_delivery_fit_artifact_id: str
    expected_delivery_admission_receipt_id: str
    expected_path_likelihood_artifact_id: str
    valid_from: pd.Timestamp
    expires_at: pd.Timestamp
    sealed_oos_opened: bool = False
    authority: str = "external_shadow_admission_binding"
    action_authority: bool = False
    pins_issued: bool = False
    schema_version: int = SIGNAL_EMPIRICAL_ADMISSION_SCHEMA_VERSION
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        identities = tuple(
            _identity(getattr(self, name), name=name)
            for name in (
                "expected_dol_fit_artifact_id",
                "expected_dol_admission_receipt_id",
                "expected_delivery_fit_artifact_id",
                "expected_delivery_admission_receipt_id",
                "expected_path_likelihood_artifact_id",
            )
        )
        valid = _aware(self.valid_from, name="outcome binding valid_from")
        expires = _aware(self.expires_at, name="outcome binding expires_at")
        if (
            self.schema_version != SIGNAL_EMPIRICAL_ADMISSION_SCHEMA_VERSION
            or self.authority != "external_shadow_admission_binding"
            or self.action_authority is not False
            or self.pins_issued is not False
            or type(self.sealed_oos_opened) is not bool
            or self.sealed_oos_opened
            or valid >= expires
        ):
            raise ValueError("empirical signal outcome admission binding is invalid")
        object.__setattr__(self, "valid_from", valid)
        object.__setattr__(self, "expires_at", expires)
        payload = {
            "schema_version": self.schema_version,
            "expected_dol_fit_artifact_id": identities[0],
            "expected_dol_admission_receipt_id": identities[1],
            "expected_delivery_fit_artifact_id": identities[2],
            "expected_delivery_admission_receipt_id": identities[3],
            "expected_path_likelihood_artifact_id": identities[4],
            "valid_from": valid.isoformat(),
            "expires_at": expires.isoformat(),
            "sealed_oos_opened": False,
            "authority": self.authority,
            "action_authority": False,
            "pins_issued": False,
        }
        object.__setattr__(
            self,
            "binding_id",
            f"signal-empirical-outcome-binding:{_canonical_sha256(payload)[:32]}",
        )


@dataclass(frozen=True)
class SignalEmpiricalAdmissionAssessment:
    """Typed gate result; a partial path conversion is never a full bundle."""

    binding_id: str
    path_conversion_ready: bool
    full_signal_bundle_ready: bool
    blockers: tuple[SignalEmpiricalBlocker, ...]
    status: str
    dol_conversion_ready: bool = False
    delivery_conversion_ready: bool = False
    outcome_binding_id: str | None = None
    authority: str = "shadow_only"
    action_authority: bool = False
    schema_version: int = SIGNAL_EMPIRICAL_ADMISSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        blockers = tuple(self.blockers)
        if (
            self.path_conversion_ready
            and self.dol_conversion_ready
            and self.delivery_conversion_ready
        ):
            expected_status = "all_conversions_ready_external_pins_closed"
        elif self.path_conversion_ready and not (
            self.dol_conversion_ready or self.delivery_conversion_ready
        ):
            expected_status = "path_conversion_ready_full_bundle_closed"
        elif any(
            (
                self.path_conversion_ready,
                self.dol_conversion_ready,
                self.delivery_conversion_ready,
            )
        ):
            expected_status = "partial_conversions_ready_full_bundle_closed"
        else:
            expected_status = "closed"
        if (
            self.schema_version != SIGNAL_EMPIRICAL_ADMISSION_SCHEMA_VERSION
            or not self.binding_id.startswith("signal-empirical-path-binding:")
            or type(self.path_conversion_ready) is not bool
            or type(self.dol_conversion_ready) is not bool
            or type(self.delivery_conversion_ready) is not bool
            or self.full_signal_bundle_ready is not False
            or not blockers
            or len(blockers) != len(set(blockers))
            or blockers != tuple(sorted(blockers, key=lambda item: item.value))
            or self.status != expected_status
            or self.authority != "shadow_only"
            or self.action_authority is not False
            or (
                self.outcome_binding_id is not None
                and not self.outcome_binding_id.startswith(
                    "signal-empirical-outcome-binding:"
                )
            )
            or (
                (self.dol_conversion_ready or self.delivery_conversion_ready)
                and self.outcome_binding_id is None
            )
        ):
            raise ValueError("signal empirical admission assessment is invalid")


@dataclass(frozen=True)
class PathLikelihoodAdmissionConversion:
    """Exact production DTO/payload plus its research-source provenance."""

    artifact: AdmittedPathLikelihoodArtifact
    payload: Mapping[str, Any]
    payload_sha256: str
    source_binding_id: str
    source_fit_artifact_id: str
    source_admission_receipt_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact, AdmittedPathLikelihoodArtifact)
            or not isinstance(self.payload, Mapping)
            or self.payload.get("artifact_id") != self.artifact.artifact_id
            or self.payload_sha256 != _canonical_sha256(self.payload)
            or not self.source_binding_id.startswith("signal-empirical-path-binding:")
            or not self.source_fit_artifact_id
            or not self.source_admission_receipt_id
        ):
            raise ValueError("path likelihood admission conversion is invalid")


@dataclass(frozen=True)
class DOLCalibrationAdmissionConversion:
    artifact: AdmittedDOLCalibrationArtifact
    payload: Mapping[str, Any]
    payload_sha256: str
    source_binding_id: str
    source_fit_artifact_id: str
    source_admission_receipt_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact, AdmittedDOLCalibrationArtifact)
            or not isinstance(self.payload, Mapping)
            or self.payload.get("artifact_id") != self.artifact.artifact_id
            or self.payload_sha256 != _canonical_sha256(self.payload)
            or not self.source_binding_id.startswith(
                "signal-empirical-outcome-binding:"
            )
            or not self.source_fit_artifact_id
            or not self.source_admission_receipt_id
        ):
            raise ValueError("DOL calibration admission conversion is invalid")


@dataclass(frozen=True)
class DeliveryModelAdmissionConversion:
    artifact: TargetBeforeInvalidationArtifact
    payload: Mapping[str, Any]
    payload_sha256: str
    source_binding_id: str
    source_fit_artifact_id: str
    source_admission_receipt_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact, TargetBeforeInvalidationArtifact)
            or not isinstance(self.payload, Mapping)
            or self.payload.get("artifact_id") != self.artifact.artifact_id
            or self.payload_sha256 != _canonical_sha256(self.payload)
            or not self.source_binding_id.startswith(
                "signal-empirical-outcome-binding:"
            )
            or not self.source_fit_artifact_id
            or not self.source_admission_receipt_id
        ):
            raise ValueError("delivery model admission conversion is invalid")


@dataclass(frozen=True)
class SignalOutcomeAdmissionConversion:
    dol: DOLCalibrationAdmissionConversion
    delivery: DeliveryModelAdmissionConversion
    source_binding_id: str
    pins_issued: bool = False
    action_authority: bool = False
    conversion_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dol, DOLCalibrationAdmissionConversion)
            or not isinstance(self.delivery, DeliveryModelAdmissionConversion)
            or self.source_binding_id != self.dol.source_binding_id
            or self.source_binding_id != self.delivery.source_binding_id
            or self.delivery.artifact.dol_calibration_artifact_id
            != self.dol.artifact.artifact_id
            or self.pins_issued is not False
            or self.action_authority is not False
        ):
            raise ValueError("signal outcome admission conversion is invalid")
        payload = {
            "dol_payload_sha256": self.dol.payload_sha256,
            "delivery_payload_sha256": self.delivery.payload_sha256,
            "source_binding_id": self.source_binding_id,
            "pins_issued": False,
            "action_authority": False,
        }
        object.__setattr__(
            self,
            "conversion_id",
            f"signal-outcome-conversion:{_canonical_sha256(payload)[:32]}",
        )


def _path_blockers(
    binding: EmpiricalPathAdmissionBinding,
    *,
    path_fit: PathCalibrationArtifact | None,
    path_receipt: ProbabilityAdmissionReceipt | None,
    path_protocol: PathBeliefProtocol | None,
) -> tuple[SignalEmpiricalBlocker, ...]:
    blockers: set[SignalEmpiricalBlocker] = set()
    if binding.cohort_role != "rolling_oof":
        blockers.add(SignalEmpiricalBlocker.SOURCE_NOT_ROLLING_OOF)
    if path_fit is None:
        blockers.add(SignalEmpiricalBlocker.PATH_FIT_MISSING)
    elif not isinstance(path_fit, PathCalibrationArtifact):
        raise TypeError("path_fit must be PathCalibrationArtifact or None")
    else:
        if path_fit.artifact_id != binding.expected_fit_artifact_id:
            blockers.add(SignalEmpiricalBlocker.PATH_FIT_PIN_MISMATCH)
        if (
            path_fit.source_dataset_sha256 != binding.source_dataset_sha256
            or path_fit.manifest_sha256 != binding.manifest_sha256
        ):
            blockers.add(SignalEmpiricalBlocker.PATH_SOURCE_HASH_MISMATCH)
        if (
            path_fit.gauge_path != "residual_unknown"
            or dict(path_fit.path_log_biases)["residual_unknown"] != 0.0
        ):
            blockers.add(SignalEmpiricalBlocker.PATH_CALIBRATION_GAUGE_MISMATCH)
    if path_receipt is None:
        blockers.add(SignalEmpiricalBlocker.PATH_RECEIPT_MISSING)
    elif not isinstance(path_receipt, ProbabilityAdmissionReceipt):
        raise TypeError("path_receipt must be ProbabilityAdmissionReceipt or None")
    else:
        if path_receipt.receipt_id != binding.expected_admission_receipt_id:
            blockers.add(SignalEmpiricalBlocker.PATH_RECEIPT_PIN_MISMATCH)
        if (
            path_receipt.source_dataset_sha256 != binding.source_dataset_sha256
            or path_receipt.manifest_sha256 != binding.manifest_sha256
            or path_receipt.cohort_identity_sha256 != binding.cohort_identity_sha256
            or (
                path_fit is not None
                and path_receipt.model_artifact_id != path_fit.artifact_id
            )
        ):
            blockers.add(SignalEmpiricalBlocker.PATH_SOURCE_HASH_MISMATCH)
        if (
            path_receipt.artifact_kind != "path_probability"
            or path_receipt.cohort_role != "rolling_oof"
            or not path_receipt.admitted
            or path_receipt.status != "admitted_shadow"
            or path_receipt.authority != "shadow_only"
            or path_receipt.action_authority is not False
        ):
            blockers.add(SignalEmpiricalBlocker.PATH_RECEIPT_NOT_ADMITTED)
    if path_protocol is None:
        blockers.add(SignalEmpiricalBlocker.PATH_PROTOCOL_MISMATCH)
    elif not isinstance(path_protocol, PathBeliefProtocol):
        raise TypeError("path_protocol must be PathBeliefProtocol or None")
    elif (
        path_protocol.fingerprint != binding.expected_path_protocol_fingerprint
        or path_protocol.model_version != binding.expected_path_model_version
    ):
        blockers.add(SignalEmpiricalBlocker.PATH_PROTOCOL_MISMATCH)
    return tuple(sorted(blockers, key=lambda item: item.value))


def _receipt_matches_fit(
    fit: DOLTemperatureFitArtifact | DeliveryModelFitArtifact,
    receipt: SignalOutcomeAdmissionReceipt,
) -> bool:
    return bool(
        receipt.model_artifact_id == fit.artifact_id
        and receipt.fit_cohort_id == fit.source_cohort_id
        and receipt.fit_cohort_role == fit.source_cohort_role
        and receipt.fit_cohort_identity_sha256 == fit.cohort_identity_sha256
        and receipt.fit_window_ids == fit.source_window_ids
        and receipt.fit_fold_ids == fit.source_fold_ids
        and receipt.fit_cluster_ids == fit.source_cluster_ids
        and receipt.split_protocol_sha256 == fit.split_protocol_sha256
        and receipt.cohort_identity_sha256 != fit.cohort_identity_sha256
        and not set(receipt.fit_cluster_ids).intersection(
            receipt.validation_cluster_ids
        )
    )


def _signal_outcome_blockers(
    binding: EmpiricalSignalOutcomeAdmissionBinding | None,
    *,
    dol_fit: DOLTemperatureFitArtifact | None,
    dol_receipt: SignalOutcomeAdmissionReceipt | None,
    delivery_fit: DeliveryModelFitArtifact | None,
    delivery_receipt: SignalOutcomeAdmissionReceipt | None,
    derived_path_artifact: AdmittedPathLikelihoodArtifact | None,
) -> tuple[
    tuple[SignalEmpiricalBlocker, ...],
    tuple[SignalEmpiricalBlocker, ...],
]:
    dol_blockers: set[SignalEmpiricalBlocker] = set()
    delivery_blockers: set[SignalEmpiricalBlocker] = set()
    if binding is None:
        dol_blockers.add(SignalEmpiricalBlocker.OUTCOME_BINDING_MISSING)
        delivery_blockers.add(SignalEmpiricalBlocker.OUTCOME_BINDING_MISSING)
    elif not isinstance(binding, EmpiricalSignalOutcomeAdmissionBinding):
        raise TypeError(
            "outcome_binding must be EmpiricalSignalOutcomeAdmissionBinding or None"
        )

    if dol_fit is None:
        dol_blockers.add(SignalEmpiricalBlocker.DOL_FIT_MISSING)
    elif not isinstance(dol_fit, DOLTemperatureFitArtifact):
        raise TypeError("dol_fit must be DOLTemperatureFitArtifact or None")
    else:
        if (
            binding is not None
            and dol_fit.artifact_id != binding.expected_dol_fit_artifact_id
        ):
            dol_blockers.add(SignalEmpiricalBlocker.DOL_FIT_PIN_MISMATCH)
        if any(
            window in _JUNE_DEVELOPMENT_WINDOWS
            for window in dol_fit.source_window_ids
        ):
            dol_blockers.add(SignalEmpiricalBlocker.DOL_JUNE_SOURCE_CLOSED)
        if (
            binding is not None
            and not dol_fit.trained_through < binding.valid_from
        ):
            dol_blockers.add(
                SignalEmpiricalBlocker.OUTCOME_VALIDITY_CLOCK_MISMATCH
            )

    if dol_receipt is None:
        dol_blockers.add(SignalEmpiricalBlocker.DOL_RECEIPT_MISSING)
    elif not isinstance(dol_receipt, SignalOutcomeAdmissionReceipt):
        raise TypeError("dol_receipt must be SignalOutcomeAdmissionReceipt or None")
    else:
        if (
            binding is not None
            and dol_receipt.receipt_id
            != binding.expected_dol_admission_receipt_id
        ):
            dol_blockers.add(SignalEmpiricalBlocker.DOL_RECEIPT_PIN_MISMATCH)
        if (
            dol_receipt.artifact_kind
            is not SignalOutcomeArtifactKind.DOL_TEMPERATURE
            or dol_receipt.validation_cohort_role != "rolling_oof"
            or not dol_receipt.admitted
            or dol_receipt.status != "admitted_shadow"
            or dol_receipt.authority != "shadow_evidence_only"
            or dol_receipt.action_authority is not False
            or dol_receipt.pins_issued is not False
        ):
            dol_blockers.add(SignalEmpiricalBlocker.DOL_RECEIPT_NOT_ADMITTED)
        if any(
            window in _JUNE_DEVELOPMENT_WINDOWS
            for window in (
                *dol_receipt.fit_window_ids,
                *dol_receipt.validation_window_ids,
            )
        ):
            dol_blockers.add(SignalEmpiricalBlocker.DOL_JUNE_SOURCE_CLOSED)
        if dol_fit is not None and not _receipt_matches_fit(dol_fit, dol_receipt):
            dol_blockers.add(SignalEmpiricalBlocker.DOL_SOURCE_BINDING_MISMATCH)

    if delivery_fit is None:
        delivery_blockers.add(SignalEmpiricalBlocker.DELIVERY_FIT_MISSING)
    elif not isinstance(delivery_fit, DeliveryModelFitArtifact):
        raise TypeError("delivery_fit must be DeliveryModelFitArtifact or None")
    else:
        if (
            binding is not None
            and delivery_fit.artifact_id
            != binding.expected_delivery_fit_artifact_id
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_FIT_PIN_MISMATCH
            )
        if any(
            window in _JUNE_DEVELOPMENT_WINDOWS
            for window in delivery_fit.source_window_ids
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_JUNE_SOURCE_CLOSED
            )
        if (
            binding is not None
            and not delivery_fit.trained_through < binding.valid_from
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.OUTCOME_VALIDITY_CLOCK_MISMATCH
            )
        if (
            binding is not None
            and delivery_fit.lineage.path_likelihood_artifact_id
            != binding.expected_path_likelihood_artifact_id
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_PATH_ARTIFACT_MISMATCH
            )
        if (
            derived_path_artifact is not None
            and binding is not None
            and binding.expected_path_likelihood_artifact_id
            != derived_path_artifact.artifact_id
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_PATH_ARTIFACT_MISMATCH
            )
        if (
            dol_fit is not None
            and delivery_fit.lineage.dol_calibration_fit_artifact_id
            != dol_fit.artifact_id
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_DOL_ARTIFACT_MISMATCH
            )

    if dol_fit is not None and derived_path_artifact is not None:
        if (
            dol_fit.lineage.source_path_protocol_fingerprint
            != derived_path_artifact.source_path_protocol_fingerprint
            or dol_fit.lineage.source_path_model_version
            != derived_path_artifact.source_path_model_version
        ):
            dol_blockers.add(
                SignalEmpiricalBlocker.DOL_SOURCE_BINDING_MISMATCH
            )

    if delivery_receipt is None:
        delivery_blockers.add(SignalEmpiricalBlocker.DELIVERY_RECEIPT_MISSING)
    elif not isinstance(delivery_receipt, SignalOutcomeAdmissionReceipt):
        raise TypeError(
            "delivery_receipt must be SignalOutcomeAdmissionReceipt or None"
        )
    else:
        if (
            binding is not None
            and delivery_receipt.receipt_id
            != binding.expected_delivery_admission_receipt_id
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_RECEIPT_PIN_MISMATCH
            )
        if (
            delivery_receipt.artifact_kind
            is not SignalOutcomeArtifactKind.DELIVERY_MODEL
            or delivery_receipt.validation_cohort_role != "rolling_oof"
            or not delivery_receipt.admitted
            or delivery_receipt.status != "admitted_shadow"
            or delivery_receipt.authority != "shadow_evidence_only"
            or delivery_receipt.action_authority is not False
            or delivery_receipt.pins_issued is not False
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_RECEIPT_NOT_ADMITTED
            )
        if any(
            window in _JUNE_DEVELOPMENT_WINDOWS
            for window in (
                *delivery_receipt.fit_window_ids,
                *delivery_receipt.validation_window_ids,
            )
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_JUNE_SOURCE_CLOSED
            )
        if delivery_fit is not None and not _receipt_matches_fit(
            delivery_fit,
            delivery_receipt,
        ):
            delivery_blockers.add(
                SignalEmpiricalBlocker.DELIVERY_SOURCE_BINDING_MISMATCH
            )

    return (
        tuple(sorted(dol_blockers, key=lambda item: item.value)),
        tuple(sorted(delivery_blockers, key=lambda item: item.value)),
    )


def assess_signal_empirical_admission(
    binding: EmpiricalPathAdmissionBinding,
    *,
    path_fit: PathCalibrationArtifact | None,
    path_receipt: ProbabilityAdmissionReceipt | None,
    path_protocol: PathBeliefProtocol | None,
    outcome_binding: EmpiricalSignalOutcomeAdmissionBinding | None = None,
    dol_fit: DOLTemperatureFitArtifact | None = None,
    dol_receipt: SignalOutcomeAdmissionReceipt | None = None,
    delivery_fit: DeliveryModelFitArtifact | None = None,
    delivery_receipt: SignalOutcomeAdmissionReceipt | None = None,
) -> SignalEmpiricalAdmissionAssessment:
    """Assess all exact conversions while leaving external pins unresolved."""

    if not isinstance(binding, EmpiricalPathAdmissionBinding):
        raise TypeError("binding must be EmpiricalPathAdmissionBinding")
    path_blockers = _path_blockers(
        binding,
        path_fit=path_fit,
        path_receipt=path_receipt,
        path_protocol=path_protocol,
    )
    derived_path_artifact: AdmittedPathLikelihoodArtifact | None = None
    if not path_blockers:
        assert path_fit is not None
        assert path_receipt is not None
        assert path_protocol is not None
        derived_path_artifact = convert_path_calibration_to_admitted(
            binding,
            path_fit=path_fit,
            path_receipt=path_receipt,
            path_protocol=path_protocol,
        ).artifact
    dol_blockers, delivery_blockers = _signal_outcome_blockers(
        outcome_binding,
        dol_fit=dol_fit,
        dol_receipt=dol_receipt,
        delivery_fit=delivery_fit,
        delivery_receipt=delivery_receipt,
        derived_path_artifact=derived_path_artifact,
    )
    blockers = tuple(
        sorted(
            {
                *path_blockers,
                *dol_blockers,
                *delivery_blockers,
                SignalEmpiricalBlocker.EXTERNAL_SIGNAL_ARTIFACT_PINS_MISSING,
            },
            key=lambda item: item.value,
        )
    )
    path_ready = not path_blockers
    dol_ready = not dol_blockers
    delivery_ready = not delivery_blockers and dol_ready
    if path_ready and dol_ready and delivery_ready:
        status = "all_conversions_ready_external_pins_closed"
    elif path_ready and not (dol_ready or delivery_ready):
        status = "path_conversion_ready_full_bundle_closed"
    elif any((path_ready, dol_ready, delivery_ready)):
        status = "partial_conversions_ready_full_bundle_closed"
    else:
        status = "closed"
    return SignalEmpiricalAdmissionAssessment(
        binding_id=binding.binding_id,
        path_conversion_ready=path_ready,
        full_signal_bundle_ready=False,
        blockers=blockers,
        status=status,
        dol_conversion_ready=dol_ready,
        delivery_conversion_ready=delivery_ready,
        outcome_binding_id=(
            None if outcome_binding is None else outcome_binding.binding_id
        ),
    )


def _path_payload(
    artifact: AdmittedPathLikelihoodArtifact,
) -> dict[str, Any]:
    """Serialize exactly the production loader schema in signal_policy.py."""

    return {
        "artifact_kind": "path_likelihood_calibration",
        "schema_version": artifact.schema_version,
        "protocol_id": artifact.protocol_id,
        "model_id": artifact.model_id,
        "model_version": artifact.model_version,
        "calibration_id": artifact.calibration_id,
        "source_dataset_id": artifact.source_dataset_id,
        "coverage_id": artifact.coverage_id,
        "trained_through": artifact.trained_through.isoformat(),
        "valid_from": artifact.valid_from.isoformat(),
        "expires_at": artifact.expires_at.isoformat(),
        "status": artifact.status,
        "authority": artifact.authority,
        "action_authority_ready": artifact.action_authority_ready,
        "artifact_id": artifact.artifact_id,
        "source_path_protocol_fingerprint": (artifact.source_path_protocol_fingerprint),
        "source_path_model_version": artifact.source_path_model_version,
        "temperature": artifact.temperature,
        "path_log_biases": {
            path.value: value for path, value in artifact.path_log_biases
        },
    }


def convert_path_calibration_to_admitted(
    binding: EmpiricalPathAdmissionBinding,
    *,
    path_fit: PathCalibrationArtifact,
    path_receipt: ProbabilityAdmissionReceipt,
    path_protocol: PathBeliefProtocol,
) -> PathLikelihoodAdmissionConversion:
    """Perform the sole exact fitted-to-production conversion currently known."""

    if not isinstance(binding, EmpiricalPathAdmissionBinding):
        raise TypeError("binding must be EmpiricalPathAdmissionBinding")
    blockers = _path_blockers(
        binding,
        path_fit=path_fit,
        path_receipt=path_receipt,
        path_protocol=path_protocol,
    )
    if blockers:
        raise SignalEmpiricalAdmissionError(
            "path empirical admission is closed: "
            + ",".join(item.value for item in blockers)
        )
    expected_paths = tuple(path.value for path in PATH_KINDS)
    fitted_paths = tuple(path for path, _ in path_fit.path_log_biases)
    if fitted_paths != expected_paths:
        raise SignalEmpiricalAdmissionError(
            "path calibration vocabulary differs from signal policy"
        )
    artifact = AdmittedPathLikelihoodArtifact(
        protocol_id=f"path-probability-admission:{path_receipt.receipt_id}",
        model_id=path_fit.source_model_artifact_id,
        model_version=path_fit.model_version,
        calibration_id=path_fit.artifact_id,
        source_dataset_id=f"sha256:{binding.source_dataset_sha256}",
        coverage_id=f"sha256:{binding.cohort_identity_sha256}",
        source_path_protocol_fingerprint=path_protocol.fingerprint,
        source_path_model_version=path_protocol.model_version,
        trained_through=binding.trained_through,
        valid_from=binding.valid_from,
        expires_at=binding.expires_at,
        temperature=path_fit.temperature,
        path_log_biases=tuple(
            (PathKind(path), value) for path, value in path_fit.path_log_biases
        ),
    )
    payload = _path_payload(artifact)
    return PathLikelihoodAdmissionConversion(
        artifact=artifact,
        payload=payload,
        payload_sha256=_canonical_sha256(payload),
        source_binding_id=binding.binding_id,
        source_fit_artifact_id=path_fit.artifact_id,
        source_admission_receipt_id=path_receipt.receipt_id,
    )


def _dol_payload(
    artifact: AdmittedDOLCalibrationArtifact,
) -> dict[str, Any]:
    """Serialize exactly the production DOL loader schema."""

    return {
        "artifact_kind": "dol_probability_calibration",
        "schema_version": artifact.schema_version,
        "protocol_id": artifact.protocol_id,
        "model_id": artifact.model_id,
        "model_version": artifact.model_version,
        "calibration_id": artifact.calibration_id,
        "source_dataset_id": artifact.source_dataset_id,
        "coverage_id": artifact.coverage_id,
        "trained_through": artifact.trained_through.isoformat(),
        "valid_from": artifact.valid_from.isoformat(),
        "expires_at": artifact.expires_at.isoformat(),
        "status": artifact.status,
        "authority": artifact.authority,
        "action_authority_ready": artifact.action_authority_ready,
        "artifact_id": artifact.artifact_id,
        "source_dol_protocol_fingerprint": (
            artifact.source_dol_protocol_fingerprint
        ),
        "source_dol_model_version": artifact.source_dol_model_version,
        "source_dol_model_fingerprint": (
            artifact.source_dol_model_fingerprint
        ),
        "source_path_protocol_fingerprint": (
            artifact.source_path_protocol_fingerprint
        ),
        "source_path_model_version": artifact.source_path_model_version,
        "temperature": artifact.temperature,
    }


def _delivery_payload(
    artifact: TargetBeforeInvalidationArtifact,
) -> dict[str, Any]:
    """Serialize exactly the production outcome-model loader schema."""

    return {
        "artifact_kind": "target_before_invalidation_model",
        "schema_version": artifact.schema_version,
        "protocol_id": artifact.protocol_id,
        "model_id": artifact.model_id,
        "model_version": artifact.model_version,
        "calibration_id": artifact.calibration_id,
        "source_dataset_id": artifact.source_dataset_id,
        "coverage_id": artifact.coverage_id,
        "trained_through": artifact.trained_through.isoformat(),
        "valid_from": artifact.valid_from.isoformat(),
        "expires_at": artifact.expires_at.isoformat(),
        "status": artifact.status,
        "authority": artifact.authority,
        "action_authority_ready": artifact.action_authority_ready,
        "artifact_id": artifact.artifact_id,
        "path_likelihood_artifact_id": artifact.path_likelihood_artifact_id,
        "dol_calibration_artifact_id": artifact.dol_calibration_artifact_id,
        "signal_policy_fingerprint": artifact.signal_policy_fingerprint,
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
            for item in artifact.setup_models
        ],
        "minimum_supported_coverage": artifact.minimum_supported_coverage,
        "estimand": artifact.estimand,
    }


def convert_signal_outcomes_to_admitted(
    binding: EmpiricalSignalOutcomeAdmissionBinding,
    *,
    path_conversion: PathLikelihoodAdmissionConversion,
    dol_fit: DOLTemperatureFitArtifact,
    dol_receipt: SignalOutcomeAdmissionReceipt,
    delivery_fit: DeliveryModelFitArtifact,
    delivery_receipt: SignalOutcomeAdmissionReceipt,
) -> SignalOutcomeAdmissionConversion:
    """Map admitted research evidence to exact shadow-only production DTOs."""

    if not isinstance(binding, EmpiricalSignalOutcomeAdmissionBinding):
        raise TypeError(
            "binding must be EmpiricalSignalOutcomeAdmissionBinding"
        )
    if not isinstance(path_conversion, PathLikelihoodAdmissionConversion):
        raise TypeError("path_conversion must be PathLikelihoodAdmissionConversion")
    dol_blockers, delivery_blockers = _signal_outcome_blockers(
        binding,
        dol_fit=dol_fit,
        dol_receipt=dol_receipt,
        delivery_fit=delivery_fit,
        delivery_receipt=delivery_receipt,
        derived_path_artifact=path_conversion.artifact,
    )
    blockers = tuple(
        sorted(
            {*dol_blockers, *delivery_blockers},
            key=lambda item: item.value,
        )
    )
    if blockers:
        raise SignalEmpiricalAdmissionError(
            "signal outcome empirical admission is closed: "
            + ",".join(item.value for item in blockers)
        )

    dol_artifact = AdmittedDOLCalibrationArtifact(
        protocol_id=f"dol-probability-admission:{dol_receipt.receipt_id}",
        model_id=(
            f"dol-probability-model:"
            f"{dol_fit.lineage.source_dol_model_fingerprint}"
        ),
        model_version=dol_fit.model_version,
        calibration_id=dol_fit.artifact_id,
        source_dataset_id=f"sha256:{dol_receipt.source_dataset_sha256}",
        coverage_id=f"sha256:{dol_receipt.cohort_identity_sha256}",
        source_dol_protocol_fingerprint=(
            dol_fit.lineage.source_dol_protocol_fingerprint
        ),
        source_dol_model_version=dol_fit.lineage.source_dol_model_version,
        source_dol_model_fingerprint=(
            dol_fit.lineage.source_dol_model_fingerprint
        ),
        source_path_protocol_fingerprint=(
            dol_fit.lineage.source_path_protocol_fingerprint
        ),
        source_path_model_version=dol_fit.lineage.source_path_model_version,
        trained_through=dol_fit.trained_through,
        valid_from=binding.valid_from,
        expires_at=binding.expires_at,
        temperature=dol_fit.temperature,
    )
    dol_payload = _dol_payload(dol_artifact)
    dol_conversion = DOLCalibrationAdmissionConversion(
        artifact=dol_artifact,
        payload=dol_payload,
        payload_sha256=_canonical_sha256(dol_payload),
        source_binding_id=binding.binding_id,
        source_fit_artifact_id=dol_fit.artifact_id,
        source_admission_receipt_id=dol_receipt.receipt_id,
    )

    delivery_artifact = TargetBeforeInvalidationArtifact(
        protocol_id=(
            "target-before-invalidation-admission:"
            f"{delivery_receipt.receipt_id}"
        ),
        model_id=delivery_fit.lineage.lineage_id,
        model_version=delivery_fit.model_version,
        calibration_id=delivery_fit.artifact_id,
        source_dataset_id=f"sha256:{delivery_receipt.source_dataset_sha256}",
        coverage_id=f"sha256:{delivery_receipt.cohort_identity_sha256}",
        path_likelihood_artifact_id=path_conversion.artifact.artifact_id,
        dol_calibration_artifact_id=dol_artifact.artifact_id,
        signal_policy_fingerprint=delivery_fit.lineage.signal_policy_fingerprint,
        trained_through=delivery_fit.trained_through,
        valid_from=binding.valid_from,
        expires_at=binding.expires_at,
        setup_models=tuple(
            SetupDeliveryModel(
                setup_family=item.setup_family,
                intercept=item.intercept,
                path_logit_coefficient=item.path_logit_coefficient,
                dol_logit_coefficient=item.dol_logit_coefficient,
                half_life_real_completed_bars=(
                    item.half_life_real_completed_bars
                ),
            )
            for item in delivery_fit.setup_models
        ),
        minimum_supported_coverage=delivery_fit.minimum_supported_coverage,
        estimand=delivery_fit.estimand,
    )
    delivery_payload = _delivery_payload(delivery_artifact)
    delivery_conversion = DeliveryModelAdmissionConversion(
        artifact=delivery_artifact,
        payload=delivery_payload,
        payload_sha256=_canonical_sha256(delivery_payload),
        source_binding_id=binding.binding_id,
        source_fit_artifact_id=delivery_fit.artifact_id,
        source_admission_receipt_id=delivery_receipt.receipt_id,
    )
    return SignalOutcomeAdmissionConversion(
        dol=dol_conversion,
        delivery=delivery_conversion,
        source_binding_id=binding.binding_id,
    )


__all__ = [
    "DOLCalibrationAdmissionConversion",
    "DeliveryModelAdmissionConversion",
    "EmpiricalPathAdmissionBinding",
    "EmpiricalSignalOutcomeAdmissionBinding",
    "PathLikelihoodAdmissionConversion",
    "SIGNAL_EMPIRICAL_ADMISSION_SCHEMA_VERSION",
    "SignalEmpiricalAdmissionAssessment",
    "SignalEmpiricalAdmissionError",
    "SignalEmpiricalBlocker",
    "SignalOutcomeAdmissionConversion",
    "assess_signal_empirical_admission",
    "convert_path_calibration_to_admitted",
    "convert_signal_outcomes_to_admitted",
]
