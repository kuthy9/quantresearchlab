"""Causal development, calibration, OOF, MBO and holdout data splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .model import aware_timestamp


class ValidationProtocolError(ValueError):
    """Raised when a replay crosses a registered data or identity boundary."""


@dataclass(frozen=True)
class ValidationWindow:
    role: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    purpose: str

    def contains(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return self.start <= start and end <= self.end_exclusive


@dataclass(frozen=True)
class CausalSourceIdentity:
    """Manifest-bound identity for the materialized causal OHLCV front."""

    path: str
    sha256: str
    manifest_path: str | None
    manifest_sha256: str | None


@dataclass(frozen=True)
class MBOExecutionArtifactIdentity:
    """Manifest-bound minute execution artifact for a development slice."""

    path: str
    sha256: str
    manifest_path: str
    manifest_sha256: str


@dataclass(frozen=True)
class MBOManifestIdentity:
    """Exact development and sealed-holdout MBO bindings.

    The development identity is the partition manifest, not a directory-name
    proxy.  The holdout identity binds the DBN, its vendor manifest, and the
    seal marker separately so a caller cannot silently treat an unverified
    file as registered execution data.
    """

    development_root: str | None
    development_partition_manifest_path: str | None
    development_partition_manifest_sha256: str | None
    development_execution_artifacts: tuple[MBOExecutionArtifactIdentity, ...]
    sealed_path: str | None
    sealed_sha256: str | None
    sealed_manifest_path: str | None
    sealed_manifest_sha256: str | None
    sealed_marker: str | None


@dataclass(frozen=True)
class BrainCalibrationFitAdmission:
    """Independent causal-unit thresholds that authorize Brain fitting."""

    minimum_dimension_units: int
    minimum_plan_valid_roots: int
    minimum_executable_episodes: int

    @classmethod
    def from_mapping(cls, payload: Any) -> BrainCalibrationFitAdmission:
        if not isinstance(payload, Mapping):
            raise ValidationProtocolError(
                "brain_calibration_fit_admission must be an object"
            )
        expected = {
            "minimum_dimension_units",
            "minimum_plan_valid_roots",
            "minimum_executable_episodes",
        }
        if set(payload) != expected:
            raise ValidationProtocolError(
                "brain_calibration_fit_admission must explicitly contain "
                "minimum_dimension_units, minimum_plan_valid_roots and "
                "minimum_executable_episodes"
            )
        values: dict[str, int] = {}
        for field in sorted(expected):
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationProtocolError(
                    "brain_calibration_fit_admission."
                    f"{field} must be an integer"
                )
            minimum = 2 if field == "minimum_dimension_units" else 1
            if value < minimum:
                raise ValidationProtocolError(
                    "brain_calibration_fit_admission."
                    f"{field} must be at least {minimum}"
                )
            values[field] = value
        return cls(**values)

    def as_dict(self) -> dict[str, int]:
        return {
            "minimum_dimension_units": self.minimum_dimension_units,
            "minimum_plan_valid_roots": self.minimum_plan_valid_roots,
            "minimum_executable_episodes": self.minimum_executable_episodes,
        }


@dataclass(frozen=True)
class ValidationProtocol:
    schema_version: int
    fingerprint: str
    causal_source: CausalSourceIdentity
    mbo_identity: MBOManifestIdentity
    belief_calibration_valid_from: pd.Timestamp
    brain_calibration_fit_admission: BrainCalibrationFitAdmission
    ohlcv_windows: Mapping[str, ValidationWindow]
    mbo_windows: Mapping[str, ValidationWindow]
    fixed_development_windows: Mapping[str, tuple[ValidationWindow, ...]]

    def classify_ohlcv(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> ValidationWindow:
        return self._classify(self.ohlcv_windows, start, end, "OHLCV")

    def classify_mbo(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> ValidationWindow:
        return self._classify(self.mbo_windows, start, end, "MBO")

    @staticmethod
    def _classify(
        windows: Mapping[str, ValidationWindow],
        start: pd.Timestamp,
        end: pd.Timestamp,
        label: str,
    ) -> ValidationWindow:
        start = aware_timestamp(start, name=f"{label}.start")
        end = aware_timestamp(end, name=f"{label}.end")
        if end <= start:
            raise ValidationProtocolError(f"{label} interval must be positive")
        matches = [window for window in windows.values() if window.contains(start, end)]
        if len(matches) != 1:
            raise ValidationProtocolError(
                f"{label} interval {start.isoformat()} -> {end.isoformat()} "
                "crosses or falls outside preregistered windows"
            )
        return matches[0]


def _load_windows(
    payload: Any,
    *,
    name: str,
) -> dict[str, ValidationWindow]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValidationProtocolError(f"{name} must be a non-empty object")
    output: dict[str, ValidationWindow] = {}
    for role, raw in payload.items():
        if not isinstance(raw, Mapping):
            raise ValidationProtocolError(f"{name}.{role} must be an object")
        start = pd.Timestamp(raw.get("start"))
        end = pd.Timestamp(raw.get("end_exclusive"))
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise ValidationProtocolError(
                f"{name}.{role} has an invalid aware interval"
            )
        output[str(role)] = ValidationWindow(
            role=str(role),
            start=start,
            end_exclusive=end,
            purpose=str(raw.get("purpose", "")).strip(),
        )
    ordered = sorted(output.values(), key=lambda item: item.start)
    for left, right in zip(ordered[:-1], ordered[1:]):
        if left.end_exclusive > right.start:
            raise ValidationProtocolError(f"{name} contains overlapping windows")
    return output


def _optional_string(value: Any) -> str | None:
    output = str(value).strip() if value is not None else ""
    return output or None


def _sha256(value: Any, *, name: str, required: bool) -> str | None:
    output = _optional_string(value)
    if output is None:
        if required:
            raise ValidationProtocolError(f"{name} is required")
        return None
    if len(output) != 64 or any(
        character not in "0123456789abcdef" for character in output
    ):
        raise ValidationProtocolError(f"{name} must be a lowercase SHA-256")
    return output


def _verify_manifest_binding(path: str, expected_sha256: str, *, name: str) -> None:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[1] / candidate
    if not candidate.is_file() or candidate.is_symlink():
        raise ValidationProtocolError(f"{name} is missing or is not a regular file")
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValidationProtocolError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _load_source_identities(
    payload: Mapping[str, Any],
) -> tuple[CausalSourceIdentity, MBOManifestIdentity]:
    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        raise ValidationProtocolError("sources must be an object")
    ohlcv = sources.get("ohlcv")
    mbo = sources.get("mbo")
    if not isinstance(ohlcv, Mapping) or not isinstance(mbo, Mapping):
        raise ValidationProtocolError("sources.ohlcv and sources.mbo must be objects")
    development = mbo.get("development")
    sealed = mbo.get("sealed_holdout")
    if not isinstance(development, Mapping) or not isinstance(sealed, Mapping):
        raise ValidationProtocolError(
            "sources.mbo.development and sources.mbo.sealed_holdout must be objects"
        )
    raw_artifacts = development.get("execution_artifacts")
    if (
        not isinstance(raw_artifacts, Sequence)
        or isinstance(raw_artifacts, (str, bytes))
        or not raw_artifacts
        or not all(isinstance(item, Mapping) for item in raw_artifacts)
    ):
        raise ValidationProtocolError(
            "sources.mbo.development.execution_artifacts must be a non-empty "
            "array of objects"
        )
    causal_source = CausalSourceIdentity(
        path=_optional_string(ohlcv.get("path")) or "",
        sha256=_sha256(ohlcv.get("sha256"), name="sources.ohlcv.sha256", required=True)
        or "",
        manifest_path=_optional_string(ohlcv.get("manifest_path")),
        manifest_sha256=_sha256(
            ohlcv.get("manifest_sha256"),
            name="sources.ohlcv.manifest_sha256",
            required=True,
        ),
    )
    if not causal_source.path or causal_source.manifest_path is None:
        raise ValidationProtocolError(
            "sources.ohlcv.path and sources.ohlcv.manifest_path are required"
        )
    mbo_identity = MBOManifestIdentity(
        development_root=_optional_string(development.get("root")),
        development_partition_manifest_path=_optional_string(
            development.get("partition_manifest_path")
        ),
        development_partition_manifest_sha256=_sha256(
            development.get("partition_manifest_sha256"),
            name="sources.mbo.development.partition_manifest_sha256",
            required=True,
        ),
        development_execution_artifacts=tuple(
            MBOExecutionArtifactIdentity(
                path=_optional_string(raw_artifact.get("path")) or "",
                sha256=_sha256(
                    raw_artifact.get("sha256"),
                    name=f"sources.mbo.development.execution_artifacts[{index}].sha256",
                    required=True,
                )
                or "",
                manifest_path=_optional_string(raw_artifact.get("manifest_path")) or "",
                manifest_sha256=_sha256(
                    raw_artifact.get("manifest_sha256"),
                    name=(
                        "sources.mbo.development.execution_artifacts"
                        f"[{index}].manifest_sha256"
                    ),
                    required=True,
                )
                or "",
            )
            for index, raw_artifact in enumerate(raw_artifacts)
        ),
        sealed_path=_optional_string(sealed.get("path")),
        sealed_sha256=_sha256(
            sealed.get("sha256"),
            name="sources.mbo.sealed_holdout.sha256",
            required=True,
        ),
        sealed_manifest_path=_optional_string(sealed.get("manifest_path")),
        sealed_manifest_sha256=_sha256(
            sealed.get("manifest_sha256"),
            name="sources.mbo.sealed_holdout.manifest_sha256",
            required=True,
        ),
        sealed_marker=_optional_string(sealed.get("sealed_marker")),
    )
    if any(
        value is None
        for value in (
            mbo_identity.development_root,
            mbo_identity.development_partition_manifest_path,
            mbo_identity.sealed_path,
            mbo_identity.sealed_manifest_path,
            mbo_identity.sealed_marker,
        )
    ):
        raise ValidationProtocolError(
            "all development-manifest and sealed-holdout MBO paths are required"
        )
    if len(mbo_identity.development_execution_artifacts) != len(raw_artifacts) or any(
        not artifact.path or not artifact.manifest_path
        for artifact in mbo_identity.development_execution_artifacts
    ):
        raise ValidationProtocolError(
            "sources.mbo.development.execution_artifacts must contain "
            "fully manifest-bound artifacts"
        )
    _verify_manifest_binding(
        causal_source.manifest_path,
        causal_source.manifest_sha256 or "",
        name="sources.ohlcv.manifest_path",
    )
    _verify_manifest_binding(
        mbo_identity.development_partition_manifest_path or "",
        mbo_identity.development_partition_manifest_sha256 or "",
        name="sources.mbo.development.partition_manifest_path",
    )
    for index, artifact in enumerate(mbo_identity.development_execution_artifacts):
        _verify_manifest_binding(
            artifact.manifest_path,
            artifact.manifest_sha256,
            name=(
                "sources.mbo.development.execution_artifacts" f"[{index}].manifest_path"
            ),
        )
    _verify_manifest_binding(
        mbo_identity.sealed_manifest_path or "",
        mbo_identity.sealed_manifest_sha256 or "",
        name="sources.mbo.sealed_holdout.manifest_path",
    )
    return causal_source, mbo_identity


def _load_fixed_development_windows(
    payload: Any,
) -> dict[str, tuple[ValidationWindow, ...]]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValidationProtocolError("fixed_development_windows must be an object")
    output: dict[str, tuple[ValidationWindow, ...]] = {}
    for group, raw_group in payload.items():
        if not isinstance(raw_group, Mapping):
            raise ValidationProtocolError(
                f"fixed_development_windows.{group} must be an object"
            )
        raw_windows = raw_group.get("windows")
        if not isinstance(raw_windows, Sequence) or isinstance(
            raw_windows, (str, bytes)
        ):
            raise ValidationProtocolError(
                f"fixed_development_windows.{group}.windows must be an array"
            )
        purpose = str(raw_group.get("purpose", "")).strip()
        windows: list[ValidationWindow] = []
        for index, raw_window in enumerate(raw_windows):
            if not isinstance(raw_window, Mapping):
                raise ValidationProtocolError(
                    f"fixed_development_windows.{group}.windows[{index}] must be an object"
                )
            role = _optional_string(raw_window.get("id"))
            start = pd.Timestamp(raw_window.get("start"))
            end = pd.Timestamp(raw_window.get("end_exclusive"))
            if (
                role is None
                or start.tzinfo is None
                or end.tzinfo is None
                or end <= start
            ):
                raise ValidationProtocolError(
                    f"fixed_development_windows.{group}.windows[{index}] is invalid"
                )
            windows.append(
                ValidationWindow(
                    role=role,
                    start=start,
                    end_exclusive=end,
                    purpose=purpose,
                )
            )
        ordered = sorted(windows, key=lambda item: item.start)
        for left, right in zip(ordered[:-1], ordered[1:]):
            if left.end_exclusive > right.start:
                raise ValidationProtocolError(
                    f"fixed_development_windows.{group} contains overlapping windows"
                )
        output[str(group)] = tuple(ordered)
    return output


def load_validation_protocol(
    path: str | Path = "configs/data_splits.json",
) -> ValidationProtocol:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[1] / source
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValidationProtocolError("validation protocol root must be an object")
    schema_value = payload.get("schema_version")
    if schema_value != 1:
        raise ValidationProtocolError("schema_version must be 1")
    causal_source, mbo_identity = _load_source_identities(payload)
    ohlcv_windows = _load_windows(
        payload.get("ohlcv_windows"),
        name="ohlcv_windows",
    )
    calibration_value = payload.get("belief_calibration_valid_from")
    if calibration_value is None:
        raise ValidationProtocolError("belief_calibration_valid_from is required")
    calibration_valid_from = pd.Timestamp(calibration_value)
    if calibration_valid_from.tzinfo is None:
        raise ValidationProtocolError(
            "belief_calibration_valid_from must be timezone aware"
        )
    return ValidationProtocol(
        schema_version=1,
        fingerprint=hashlib.sha256(raw).hexdigest(),
        causal_source=causal_source,
        mbo_identity=mbo_identity,
        belief_calibration_valid_from=calibration_valid_from,
        brain_calibration_fit_admission=(
            BrainCalibrationFitAdmission.from_mapping(
                payload.get("brain_calibration_fit_admission")
            )
        ),
        ohlcv_windows=ohlcv_windows,
        mbo_windows=_load_windows(
            payload.get("mbo_windows"),
            name="mbo_windows",
        ),
        fixed_development_windows=_load_fixed_development_windows(
            payload.get("fixed_development_windows")
        ),
    )


__all__ = [
    "BrainCalibrationFitAdmission",
    "CausalSourceIdentity",
    "MBOExecutionArtifactIdentity",
    "MBOManifestIdentity",
    "ValidationProtocol",
    "ValidationProtocolError",
    "ValidationWindow",
    "load_validation_protocol",
]
