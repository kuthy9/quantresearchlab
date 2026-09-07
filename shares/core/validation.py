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
class NeutralRepresentationWindow:
    """One preregistered, input-only MarketEpisode capture window."""

    profile_name: str
    representation_split_role: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    warmup_start: pd.Timestamp
    allowed_ohlcv_role: str
    representation_fit_allowed: bool
    expected_symbol: str
    expected_instrument_id: int


@dataclass(frozen=True)
class NeutralRepresentationSplitRegistry:
    """Leakage boundary shared by neutral materialization and representation fit."""

    protocol_version: str
    timezone: str
    warmup_calendar_days: int
    purge_calendar_days: int
    embargo_trading_days: int
    embargo_day_basis: str
    whole_market_episode_single_split: bool
    single_symbol_instrument_replay_frame_required: bool
    episode_split_key: tuple[str, ...]
    smoke_profiles: Mapping[str, str]
    windows: Mapping[str, NeutralRepresentationWindow]


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
    neutral_representation_splits: NeutralRepresentationSplitRegistry

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
        candidate = Path(__file__).resolve().parents[2] / candidate
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


def _after_trading_day_embargo(
    start: pd.Timestamp,
    *,
    trading_days: int,
) -> pd.Timestamp:
    """Return the first local midnight after ``trading_days`` weekdays.

    This is deliberately a deterministic registry check.  The materializer's
    source preflight remains responsible for confirming completed sessions.
    """

    cursor = start.normalize()
    remaining = trading_days
    while remaining:
        if cursor.dayofweek < 5:
            remaining -= 1
        cursor = cursor + pd.DateOffset(days=1)
    return cursor


def _load_neutral_representation_splits(
    payload: Any,
    *,
    market_case_profiles: Any,
    ohlcv_windows: Mapping[str, ValidationWindow],
) -> NeutralRepresentationSplitRegistry:
    name = "neutral_representation_split_registry"
    if not isinstance(payload, Mapping):
        raise ValidationProtocolError(f"{name} must be an object")
    if not isinstance(market_case_profiles, Mapping):
        raise ValidationProtocolError("market_case_input_profiles must be an object")

    protocol_version = _optional_string(payload.get("protocol_version"))
    timezone = _optional_string(payload.get("timezone"))
    profile_names = payload.get("profiles")
    smoke_profiles = payload.get("smoke_profiles")
    episode_split_key = payload.get("market_episode_split_key")
    integer_fields = {
        "warmup_calendar_days": payload.get("warmup_calendar_days"),
        "purge_calendar_days": payload.get("purge_calendar_days"),
        "embargo_trading_days": payload.get("embargo_trading_days"),
    }
    if protocol_version != "neutral-representation-splits-1.0.0":
        raise ValidationProtocolError(f"{name}.protocol_version is incompatible")
    if timezone != "America/New_York":
        raise ValidationProtocolError(f"{name}.timezone must be America/New_York")
    embargo_day_basis = _optional_string(payload.get("embargo_day_basis"))
    if embargo_day_basis != (
        "Monday-Friday local calendar dates; materialization must also confirm "
        "completed source sessions"
    ):
        raise ValidationProtocolError(f"{name}.embargo_day_basis is incompatible")
    if any(type(value) is not int or value <= 0 for value in integer_fields.values()):
        raise ValidationProtocolError(f"{name} day boundaries must be positive integers")
    if integer_fields != {
        "warmup_calendar_days": 14,
        "purge_calendar_days": 14,
        "embargo_trading_days": 5,
    }:
        raise ValidationProtocolError(
            f"{name} must preserve 14-day warmup, 14-day purge and "
            "5-trading-day embargo"
        )
    if (
        payload.get("capture_interval") != "[start,end_exclusive)"
        or payload.get("whole_market_episode_single_split") is not True
        or payload.get("single_symbol_instrument_replay_frame_required") is not True
    ):
        raise ValidationProtocolError(f"{name} has an incompatible split contract")
    if episode_split_key != [
        "run_manifest_sha256",
        "market_epoch_id",
        "market_episode_id",
    ]:
        raise ValidationProtocolError(
            f"{name}.market_episode_split_key must preserve whole MarketEpisodes"
        )
    if (
        not isinstance(profile_names, Sequence)
        or isinstance(profile_names, (str, bytes))
        or not profile_names
        or any(type(value) is not str or not value for value in profile_names)
        or len(set(profile_names)) != len(profile_names)
    ):
        raise ValidationProtocolError(f"{name}.profiles must contain unique profile names")
    if not isinstance(smoke_profiles, Mapping) or set(smoke_profiles) != {
        "train",
        "validation",
        "holdout",
    }:
        raise ValidationProtocolError(
            f"{name}.smoke_profiles must select train, validation and holdout"
        )

    registered = set(profile_names)
    marked = {
        str(profile_name)
        for profile_name, profile in market_case_profiles.items()
        if isinstance(profile, Mapping)
        and profile.get("representation_split_role")
        in {"train", "validation", "holdout"}
    }
    if marked != registered:
        raise ValidationProtocolError(
            f"{name}.profiles disagree with neutral representation profiles"
        )

    windows: dict[str, NeutralRepresentationWindow] = {}
    forbidden_true = (
        "threshold_search",
        "calibration_fit_allowed",
        "future_path_used",
        "outcome_used",
        "pnl_used",
        "mbo_used",
        "brain_output",
        "decision_output",
        "risk_output",
        "execution_output",
        "shadow_output",
        "legacy_case_output",
    )
    for profile_name in profile_names:
        raw = market_case_profiles.get(profile_name)
        if not isinstance(raw, Mapping):
            raise ValidationProtocolError(f"{name}.{profile_name} is missing")
        role = raw.get("representation_split_role")
        fit_allowed = raw.get("representation_fit_allowed")
        if role not in {"train", "validation", "holdout"}:
            raise ValidationProtocolError(
                f"{profile_name}.representation_split_role is invalid"
            )
        if fit_allowed is not (role == "train"):
            raise ValidationProtocolError(
                f"{profile_name}.representation_fit_allowed disagrees with "
                "representation_split_role"
            )
        if any(raw.get(field) is not False for field in forbidden_true):
            raise ValidationProtocolError(
                f"{profile_name} violates the neutral input-only split contract"
            )
        if (
            raw.get("runner_mode") != "market_episode_input_only"
            or raw.get("model_config") != "configs/model.json"
            or raw.get("warmup_calendar_days") != 14
        ):
            raise ValidationProtocolError(f"{profile_name} has incompatible identity")

        start = pd.Timestamp(raw.get("start"))
        end = pd.Timestamp(raw.get("end_exclusive"))
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise ValidationProtocolError(f"{profile_name} has an invalid aware window")
        start_local = start.tz_convert(timezone)
        end_local = end.tz_convert(timezone)
        if (
            start_local != start_local.normalize()
            or end_local != end_local.normalize()
            or end_local != start_local + pd.DateOffset(months=1)
            or start_local.day != 1
        ):
            raise ValidationProtocolError(
                f"{profile_name} must capture exactly one local calendar month"
            )
        expected_name = (
            f"neutral_representation_{role}_{start_local.strftime('%Y_%m')}"
        )
        if profile_name != expected_name:
            raise ValidationProtocolError(
                f"{profile_name} does not match its role/month identity"
            )
        warmup_start = start_local - pd.DateOffset(days=14)
        allowed_role = _optional_string(raw.get("allowed_ohlcv_role"))
        allowed_window = ohlcv_windows.get(allowed_role or "")
        if allowed_window is None or not allowed_window.contains(warmup_start, end_local):
            raise ValidationProtocolError(
                f"{profile_name} replay frame crosses its allowed OHLCV role"
            )
        contract = raw.get("expected_replay_contract")
        if not isinstance(contract, Mapping):
            raise ValidationProtocolError(
                f"{profile_name}.expected_replay_contract is required"
            )
        symbol = contract.get("symbol")
        instrument_id = contract.get("instrument_id")
        if type(symbol) is not str or not symbol or type(instrument_id) is not int:
            raise ValidationProtocolError(
                f"{profile_name}.expected_replay_contract is invalid"
            )
        windows[profile_name] = NeutralRepresentationWindow(
            profile_name=profile_name,
            representation_split_role=role,
            start=start_local,
            end_exclusive=end_local,
            warmup_start=warmup_start,
            allowed_ohlcv_role=allowed_role or "",
            representation_fit_allowed=fit_allowed,
            expected_symbol=symbol,
            expected_instrument_id=instrument_id,
        )

    ordered = sorted(windows.values(), key=lambda item: item.warmup_start)
    purge_days = integer_fields["purge_calendar_days"]
    embargo_days = integer_fields["embargo_trading_days"]
    for left, right in zip(ordered[:-1], ordered[1:]):
        purge_end = left.end_exclusive + pd.DateOffset(days=purge_days)
        earliest_next_replay = _after_trading_day_embargo(
            purge_end,
            trading_days=embargo_days,
        )
        if right.warmup_start < earliest_next_replay:
            raise ValidationProtocolError(
                "neutral representation replay frames violate the 14-calendar-day "
                "purge plus 5-trading-day embargo"
            )

    normalized_smoke = {str(role): str(profile) for role, profile in smoke_profiles.items()}
    for role, profile_name in normalized_smoke.items():
        window = windows.get(profile_name)
        if window is None or window.representation_split_role != role:
            raise ValidationProtocolError(
                f"{name}.smoke_profiles.{role} does not select that split role"
            )
    return NeutralRepresentationSplitRegistry(
        protocol_version=protocol_version,
        timezone=timezone,
        warmup_calendar_days=integer_fields["warmup_calendar_days"],
        purge_calendar_days=purge_days,
        embargo_trading_days=embargo_days,
        embargo_day_basis=embargo_day_basis,
        whole_market_episode_single_split=True,
        single_symbol_instrument_replay_frame_required=True,
        episode_split_key=tuple(episode_split_key),
        smoke_profiles=normalized_smoke,
        windows=windows,
    )


def load_validation_protocol(
    path: str | Path = "configs/data_splits.json",
) -> ValidationProtocol:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[2] / source
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
    neutral_representation_splits = _load_neutral_representation_splits(
        payload.get("neutral_representation_split_registry"),
        market_case_profiles=payload.get("market_case_input_profiles"),
        ohlcv_windows=ohlcv_windows,
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
        neutral_representation_splits=neutral_representation_splits,
    )


__all__ = [
    "BrainCalibrationFitAdmission",
    "CausalSourceIdentity",
    "MBOExecutionArtifactIdentity",
    "MBOManifestIdentity",
    "NeutralRepresentationSplitRegistry",
    "NeutralRepresentationWindow",
    "ValidationProtocol",
    "ValidationProtocolError",
    "ValidationWindow",
    "load_validation_protocol",
]
