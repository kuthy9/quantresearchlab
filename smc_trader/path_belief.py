"""Deterministic hypothesis manager for mutually exclusive market paths.

This module is deliberately independent of ``PlaybookBrain`` and Decision
wiring.  It does not discover market facts, choose typed candidates, or grant
trading authority.  The narrow Brain adapter may feed it only exact canonical
event identities admitted by the frozen Phase 6 evidence boundary.

The Phase 7 surface reuses the original path reducer.  A versioned protocol may
admit conditional likelihoods and the reducer then performs ordinary Bayesian
accounting in log space.  The shipped artifact is admitted for diagnostics
only: its likelihood assumptions have not passed posterior calibration and it
therefore grants no action authority.
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


PATH_STATE_SCHEMA_VERSION = 2
HYPOTHESIS_MANAGER_SCHEMA_VERSION = 1
PATH_RUNTIME_RESOLUTION_SCHEMA_VERSION = 1


class PathBeliefProtocolError(ValueError):
    """Raised when a path-hypothesis protocol is incomplete or unsafe."""


class PathKind(str, Enum):
    CONTINUATION = "continuation"
    DEEPER_RETRACEMENT = "deeper_retracement"
    REVERSAL = "reversal"
    BALANCE = "balance"
    FAILED_BREAKOUT = "failed_breakout"
    RESIDUAL_UNKNOWN = "residual_unknown"


PATH_KINDS = tuple(PathKind)


class PathStatus(str, Enum):
    ACTIVE = "active"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    REALIZED = "realized"


@dataclass(frozen=True)
class PathRuntimeResolutionProtocol:
    """Frozen market-fact boundary for the production shadow adapter.

    The reducer deliberately knows nothing about Eye/Scene-Graph objects.
    This compact protocol records which adapter version may translate those
    objects into reducer terminal/outcome events.  The executable mapping
    remains in ``playbooks`` beside the existing Brain adapter.
    """

    schema_version: int
    protocol_version: str
    status: str
    authority: str
    source_authority: str
    entry_episode_terminal_authority: bool
    realized_winner_rules: tuple[tuple[PathKind, str], ...]
    falsification_rules: tuple[tuple[PathKind, str], ...]
    expiry_rules: tuple[tuple[PathKind, str], ...]

    def __post_init__(self) -> None:
        collections = (
            self.realized_winner_rules,
            self.falsification_rules,
            self.expiry_rules,
        )
        if (
            self.schema_version != PATH_RUNTIME_RESOLUTION_SCHEMA_VERSION
            or not self.protocol_version
            or self.status != "development_unvalidated"
            or self.authority != "shadow_only"
            or self.source_authority
            != "canonical_semantic_atomic_plus_global_market_context"
            or self.entry_episode_terminal_authority is not False
            or any(
                tuple(path for path, _ in values) != PATH_KINDS
                or any(not isinstance(rule, str) or not rule for _, rule in values)
                for values in collections
            )
        ):
            raise ValueError("path runtime resolution protocol is invalid")

    @staticmethod
    def _rule(
        values: tuple[tuple[PathKind, str], ...],
        path: PathKind,
    ) -> str:
        return values[PATH_KINDS.index(PathKind(path))][1]

    def winner_rule(self, path: PathKind) -> str:
        return self._rule(self.realized_winner_rules, path)

    def falsification_rule(self, path: PathKind) -> str:
        return self._rule(self.falsification_rules, path)

    def expiry_rule(self, path: PathKind) -> str:
        return self._rule(self.expiry_rules, path)


def _aware_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identities(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    raw = tuple(values)
    if (
        not raw
        or any(not isinstance(value, str) or not value for value in raw)
        or len(raw) != len(set(raw))
    ):
        raise ValueError(f"{name} must contain exact non-empty identities")
    return tuple(sorted(raw))


def _path_values(
    values: Mapping[str, Any],
    *,
    name: str,
    nonnegative: bool = False,
) -> tuple[tuple[PathKind, float], ...]:
    expected = {path.value for path in PATH_KINDS}
    if not isinstance(values, Mapping) or set(values) != expected:
        raise PathBeliefProtocolError(
            f"{name} must define exactly every registered path"
        )
    output: list[tuple[PathKind, float]] = []
    for path in PATH_KINDS:
        try:
            if isinstance(values[path.value], bool):
                raise TypeError
            value = float(values[path.value])
        except (TypeError, ValueError) as error:
            raise PathBeliefProtocolError(f"{name}.{path.value} is invalid") from error
        if not math.isfinite(value) or (nonnegative and value < 0.0):
            raise PathBeliefProtocolError(f"{name}.{path.value} is invalid")
        output.append((path, value))
    return tuple(output)


def _positive_path_values(
    values: Mapping[str, Any],
    *,
    name: str,
) -> tuple[tuple[PathKind, float], ...]:
    output = _path_values(values, name=name, nonnegative=True)
    if any(value <= 0.0 for _, value in output):
        raise PathBeliefProtocolError(
            f"{name} must contain strictly positive probabilities"
        )
    return output


def _path_text_values(
    values: Any,
    *,
    name: str,
) -> tuple[tuple[PathKind, str], ...]:
    expected = {path.value for path in PATH_KINDS}
    if not isinstance(values, Mapping) or set(values) != expected:
        raise PathBeliefProtocolError(
            f"{name} must define exactly every registered path"
        )
    output: list[tuple[PathKind, str]] = []
    for path in PATH_KINDS:
        value = values[path.value]
        if not isinstance(value, str) or not value.strip():
            raise PathBeliefProtocolError(f"{name}.{path.value} is invalid")
        output.append((path, value.strip()))
    return tuple(output)


def _value_for(
    values: tuple[tuple[PathKind, float], ...],
    path: PathKind,
) -> float:
    return values[PATH_KINDS.index(PathKind(path))][1]


def _validate_path_value_order(
    values: tuple[tuple[PathKind, float], ...],
    *,
    name: str,
) -> None:
    if (
        tuple(path for path, _ in values) != PATH_KINDS
        or any(not math.isfinite(float(value)) for _, value in values)
    ):
        raise ValueError(f"{name} path values are invalid")


@dataclass(frozen=True)
class PathEvidenceRule:
    rule_id: str
    description: str
    log_likelihood_increments: tuple[tuple[PathKind, float], ...]
    evidence_family: str = "legacy_unclassified"
    conditional_likelihoods: tuple[tuple[PathKind, float], ...] | None = None

    def __post_init__(self) -> None:
        _validate_path_value_order(
            self.log_likelihood_increments,
            name="evidence rule",
        )
        if not self.rule_id or not self.description or not self.evidence_family:
            raise ValueError("path evidence rule identity is incomplete")
        if self.conditional_likelihoods is not None:
            _validate_path_value_order(
                self.conditional_likelihoods,
                name="conditional likelihood",
            )
            if any(
                not 0.0 < float(value) <= 1.0
                for _, value in self.conditional_likelihoods
            ):
                raise ValueError(
                    "conditional likelihoods must be finite probabilities"
                )
            expected = tuple(
                (path, math.log(float(value)))
                for path, value in self.conditional_likelihoods
            )
            if any(
                path is not expected_path
                or not math.isclose(
                    float(increment),
                    float(expected_increment),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for (path, increment), (
                    expected_path,
                    expected_increment,
                ) in zip(self.log_likelihood_increments, expected, strict=True)
            ):
                raise ValueError(
                    "Bayesian evidence increments must equal log likelihoods"
                )

    def increment(self, path: PathKind) -> float:
        return _value_for(self.log_likelihood_increments, path)


@dataclass(frozen=True)
class PathEvidenceContribution:
    contribution_id: str
    competition_set_id: str
    rule_id: str
    model_version: str
    protocol_fingerprint: str
    source_event_ids: tuple[str, ...]
    known_at: pd.Timestamp
    log_likelihood_increments: tuple[tuple[PathKind, float], ...]
    evidence_family: str = "legacy_unclassified"
    correlation_key: str = "legacy_unclassified"
    conditional_likelihoods: tuple[tuple[PathKind, float], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "known_at",
            _aware_timestamp(self.known_at, name="path contribution known_at"),
        )
        sources = _identities(
            self.source_event_ids,
            name="path contribution source_event_ids",
        )
        object.__setattr__(self, "source_event_ids", sources)
        _validate_path_value_order(
            self.log_likelihood_increments,
            name="path contribution",
        )
        if self.conditional_likelihoods is not None:
            _validate_path_value_order(
                self.conditional_likelihoods,
                name="path contribution conditional likelihood",
            )
            if any(
                not 0.0 < float(value) <= 1.0
                for _, value in self.conditional_likelihoods
            ):
                raise ValueError("path contribution likelihood is invalid")
        identity_fields = (
            self.contribution_id,
            self.competition_set_id,
            self.rule_id,
            self.model_version,
            self.protocol_fingerprint,
            self.evidence_family,
            self.correlation_key,
        )
        if any(not isinstance(value, str) or not value for value in identity_fields):
            raise ValueError("path contribution identity is incomplete")
        expected = _contribution_id(
            competition_set_id=self.competition_set_id,
            rule_id=self.rule_id,
            model_version=self.model_version,
            protocol_fingerprint=self.protocol_fingerprint,
            source_event_ids=sources,
            known_at=self.known_at,
            evidence_family=self.evidence_family,
            correlation_key=self.correlation_key,
        )
        if self.contribution_id != expected:
            raise ValueError("path contribution identity is not deterministic")

    def increment(self, path: PathKind) -> float:
        return _value_for(self.log_likelihood_increments, path)

    def likelihood(self, path: PathKind) -> float | None:
        if self.conditional_likelihoods is None:
            return None
        return _value_for(self.conditional_likelihoods, path)


@dataclass(frozen=True)
class PathTerminalEvent:
    terminal_event_id: str
    competition_set_id: str
    path: PathKind
    status: PathStatus
    rule_id: str
    reason: str
    model_version: str
    protocol_fingerprint: str
    source_event_ids: tuple[str, ...]
    known_at: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", PathKind(self.path))
        object.__setattr__(self, "status", PathStatus(self.status))
        object.__setattr__(
            self,
            "known_at",
            _aware_timestamp(self.known_at, name="path terminal known_at"),
        )
        sources = _identities(
            self.source_event_ids,
            name="path terminal source_event_ids",
        )
        object.__setattr__(self, "source_event_ids", sources)
        identity_fields = (
            self.terminal_event_id,
            self.competition_set_id,
            self.rule_id,
            self.reason,
            self.model_version,
            self.protocol_fingerprint,
        )
        if (
            self.status not in {PathStatus.INVALIDATED, PathStatus.EXPIRED}
            or any(
                not isinstance(value, str) or not value
                for value in identity_fields
            )
        ):
            raise ValueError("path terminal identity or status is invalid")
        expected = _terminal_event_id(
            competition_set_id=self.competition_set_id,
            path=self.path,
            status=self.status,
            rule_id=self.rule_id,
            reason=self.reason,
            model_version=self.model_version,
            protocol_fingerprint=self.protocol_fingerprint,
            source_event_ids=sources,
            known_at=self.known_at,
        )
        if self.terminal_event_id != expected:
            raise ValueError("path terminal identity is not deterministic")


@dataclass(frozen=True)
class PathOutcomeEvent:
    """One explicit, source-bound realization of a competition-set winner."""

    outcome_event_id: str
    competition_set_id: str
    winner_path: PathKind
    reason: str
    model_version: str
    protocol_fingerprint: str
    source_event_ids: tuple[str, ...]
    known_at: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "winner_path", PathKind(self.winner_path))
        object.__setattr__(
            self,
            "known_at",
            _aware_timestamp(self.known_at, name="path outcome known_at"),
        )
        sources = _identities(
            self.source_event_ids,
            name="path outcome source_event_ids",
        )
        object.__setattr__(self, "source_event_ids", sources)
        if (
            self.winner_path is PathKind.RESIDUAL_UNKNOWN
            or any(
                not isinstance(value, str) or not value
                for value in (
                    self.outcome_event_id,
                    self.competition_set_id,
                    self.reason,
                    self.model_version,
                    self.protocol_fingerprint,
                )
            )
        ):
            raise ValueError("path outcome identity is incomplete")
        expected = _outcome_event_id(
            competition_set_id=self.competition_set_id,
            winner_path=self.winner_path,
            reason=self.reason,
            model_version=self.model_version,
            protocol_fingerprint=self.protocol_fingerprint,
            source_event_ids=sources,
            known_at=self.known_at,
        )
        if self.outcome_event_id != expected:
            raise ValueError("path outcome identity is not deterministic")


@dataclass(frozen=True)
class PathBeliefProtocol:
    schema_version: int
    protocol_version: str
    model_version: str
    status: str
    authority: str
    probability_interpretation: str
    weight_interpretation: str
    common_horizon_rule: str
    common_horizon_expiry_resolution: str
    mutual_exclusivity_rule: str
    exhaustiveness_rule: str
    same_clock_precedence: str
    evidence_dependence_rule: str
    model_admission_status: str
    likelihood_artifact_status: str
    action_authority_ready: bool
    evidence_allowlist: tuple[str, ...]
    phase6_manifest_sha256: str
    phase6_result_sha256: str
    phase6_result_identity: str
    prior_log_weights: tuple[tuple[PathKind, float], ...]
    real_completed_bar_decay: tuple[tuple[PathKind, float], ...]
    terminal_rules: tuple[tuple[str, PathStatus], ...]
    evidence_rules: tuple[PathEvidenceRule, ...]
    runtime_resolution: PathRuntimeResolutionProtocol
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_path_value_order(
            self.prior_log_weights,
            name="path prior",
        )
        _validate_path_value_order(
            self.real_completed_bar_decay,
            name="path decay",
        )
        if any(value < 0.0 for _, value in self.real_completed_bar_decay):
            raise ValueError("path decay must be non-negative")
        if any(value != 0.0 for _, value in self.real_completed_bar_decay):
            raise ValueError(
                "path decay requires a separately admitted temporal artifact"
            )
        if not isinstance(
            self.runtime_resolution,
            PathRuntimeResolutionProtocol,
        ):
            raise ValueError("path runtime resolution protocol is missing")
        allowlist = tuple(self.evidence_allowlist)
        if (
            not allowlist
            or len(allowlist) != len(set(allowlist))
            or any(not isinstance(value, str) or not value for value in allowlist)
        ):
            raise ValueError("path evidence allowlist is invalid")
        object.__setattr__(self, "evidence_allowlist", allowlist)
        if (
            self.schema_version != 2
            or self.status != "development_unvalidated"
            or self.authority != "shadow_only"
            or "not_calibrated" not in self.probability_interpretation
            or "conditional_likelihood" not in self.weight_interpretation
            or self.model_admission_status
            not in {
                "evidence_admission_only",
                "diagnostic_likelihood_admitted",
                "action_model_admitted",
            }
            or self.likelihood_artifact_status
            not in {"missing", "diagnostic_admitted", "action_ready"}
            or self.action_authority_ready is not False
            or not self.protocol_version
            or not self.model_version
            or not self.common_horizon_rule
            or not self.common_horizon_expiry_resolution
            or not self.mutual_exclusivity_rule
            or "residual_unknown" not in self.exhaustiveness_rule
            or not self.same_clock_precedence
            or self.evidence_dependence_rule
            != (
                "correlation_key_is_global_across_evidence_families;"
                "one_dependency_cluster_requires_one_precombined_or_"
                "history_conditioned_likelihood_contribution"
            )
            or len(self.fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.fingerprint)
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.phase6_manifest_sha256,
                    self.phase6_result_sha256,
                    self.phase6_result_identity,
                )
            )
        ):
            raise ValueError("path belief protocol boundary is invalid")
        terminal_ids = tuple(rule_id for rule_id, _ in self.terminal_rules)
        if (
            not terminal_ids
            or len(terminal_ids) != len(set(terminal_ids))
            or any(
                status not in {PathStatus.INVALIDATED, PathStatus.EXPIRED}
                for _, status in self.terminal_rules
            )
        ):
            raise ValueError("path terminal protocol is invalid")
        evidence_ids = tuple(rule.rule_id for rule in self.evidence_rules)
        if (
            not evidence_ids
            or len(evidence_ids) != len(set(evidence_ids))
            or tuple(sorted(evidence_ids)) != tuple(sorted(self.evidence_allowlist))
        ):
            raise ValueError("path evidence protocol is invalid")
        admitted_rules = tuple(
            rule
            for rule in self.evidence_rules
            if rule.rule_id in self.evidence_allowlist
        )
        if (
            tuple(sorted(rule.rule_id for rule in admitted_rules))
            != tuple(sorted(self.evidence_allowlist))
        ):
            raise ValueError("the Phase 6 evidence allowlist is not implemented")
        if self.likelihood_artifact_status == "missing":
            if (
                self.model_admission_status != "evidence_admission_only"
                or any(
                    rule.conditional_likelihoods is not None
                    or any(
                        value != 0.0
                        for _, value in rule.log_likelihood_increments
                    )
                    for rule in admitted_rules
                )
            ):
                raise ValueError(
                    "missing likelihood artifact requires neutral admitted evidence"
                )
        else:
            expected_admission = (
                "diagnostic_likelihood_admitted"
                if self.likelihood_artifact_status == "diagnostic_admitted"
                else "action_model_admitted"
            )
            if (
                self.model_admission_status != expected_admission
                or any(
                    rule.conditional_likelihoods is None
                    for rule in admitted_rules
                )
            ):
                raise ValueError(
                    "an admitted likelihood artifact requires a matching admitted model"
                )

    @property
    def path_kinds(self) -> tuple[PathKind, ...]:
        return PATH_KINDS

    def prior(self, path: PathKind) -> float:
        return _value_for(self.prior_log_weights, path)

    def decay(self, path: PathKind) -> float:
        return _value_for(self.real_completed_bar_decay, path)

    def evidence_rule(self, rule_id: str) -> PathEvidenceRule:
        rule = next(
            (rule for rule in self.evidence_rules if rule.rule_id == rule_id),
            None,
        )
        if rule is None:
            raise PathBeliefProtocolError(
                f"unknown path evidence rule: {rule_id}"
            )
        return rule

    def is_evidence_admitted(self, rule_id: str) -> bool:
        return rule_id in self.evidence_allowlist

    @property
    def can_authorize_action(self) -> bool:
        """Fail closed until a separately admitted action model exists."""

        return bool(
            self.action_authority_ready
            and self.likelihood_artifact_status == "action_ready"
            and self.authority != "shadow_only"
        )

    @property
    def can_apply_bayesian_update(self) -> bool:
        return self.likelihood_artifact_status in {
            "diagnostic_admitted",
            "action_ready",
        }

    def terminal_status(self, rule_id: str) -> PathStatus:
        value = next(
            (status for candidate, status in self.terminal_rules if candidate == rule_id),
            None,
        )
        if value is None:
            raise PathBeliefProtocolError(
                f"unknown path terminal rule: {rule_id}"
            )
        return value

    def make_contribution(
        self,
        *,
        competition_set_id: str,
        rule_id: str,
        source_event_ids: Sequence[str],
        known_at: pd.Timestamp,
        correlation_key: str | None = None,
        require_admitted: bool = False,
    ) -> PathEvidenceContribution:
        rule = self.evidence_rule(rule_id)
        if require_admitted and not self.is_evidence_admitted(rule_id):
            raise PathBeliefProtocolError(
                f"path evidence rule is not admitted: {rule_id}"
            )
        sources = _identities(
            source_event_ids,
            name="path contribution source_event_ids",
        )
        clock = _aware_timestamp(known_at, name="path contribution known_at")
        resolved_correlation_key = (
            f"source-cluster:{_canonical_hash(sources)[:24]}"
            if correlation_key is None
            else str(correlation_key).strip()
        )
        if not resolved_correlation_key:
            raise ValueError("path evidence correlation key is required")
        return PathEvidenceContribution(
            contribution_id=_contribution_id(
                competition_set_id=competition_set_id,
                rule_id=rule_id,
                model_version=self.model_version,
                protocol_fingerprint=self.fingerprint,
                source_event_ids=sources,
                known_at=clock,
                evidence_family=rule.evidence_family,
                correlation_key=resolved_correlation_key,
            ),
            competition_set_id=competition_set_id,
            rule_id=rule_id,
            model_version=self.model_version,
            protocol_fingerprint=self.fingerprint,
            source_event_ids=sources,
            known_at=clock,
            log_likelihood_increments=rule.log_likelihood_increments,
            evidence_family=rule.evidence_family,
            correlation_key=resolved_correlation_key,
            conditional_likelihoods=rule.conditional_likelihoods,
        )

    def make_terminal_event(
        self,
        *,
        competition_set_id: str,
        path: PathKind,
        rule_id: str,
        reason: str,
        source_event_ids: Sequence[str],
        known_at: pd.Timestamp,
    ) -> PathTerminalEvent:
        path = PathKind(path)
        status = self.terminal_status(rule_id)
        sources = _identities(
            source_event_ids,
            name="path terminal source_event_ids",
        )
        clock = _aware_timestamp(known_at, name="path terminal known_at")
        return PathTerminalEvent(
            terminal_event_id=_terminal_event_id(
                competition_set_id=competition_set_id,
                path=path,
                status=status,
                rule_id=rule_id,
                reason=reason,
                model_version=self.model_version,
                protocol_fingerprint=self.fingerprint,
                source_event_ids=sources,
                known_at=clock,
            ),
            competition_set_id=competition_set_id,
            path=path,
            status=status,
            rule_id=rule_id,
            reason=reason,
            model_version=self.model_version,
            protocol_fingerprint=self.fingerprint,
            source_event_ids=sources,
            known_at=clock,
        )

    def make_outcome_event(
        self,
        *,
        competition_set_id: str,
        winner_path: PathKind,
        reason: str,
        source_event_ids: Sequence[str],
        known_at: pd.Timestamp,
    ) -> PathOutcomeEvent:
        winner = PathKind(winner_path)
        sources = _identities(
            source_event_ids,
            name="path outcome source_event_ids",
        )
        clock = _aware_timestamp(known_at, name="path outcome known_at")
        outcome_id = _outcome_event_id(
            competition_set_id=competition_set_id,
            winner_path=winner,
            reason=reason,
            model_version=self.model_version,
            protocol_fingerprint=self.fingerprint,
            source_event_ids=sources,
            known_at=clock,
        )
        return PathOutcomeEvent(
            outcome_event_id=outcome_id,
            competition_set_id=competition_set_id,
            winner_path=winner,
            reason=reason,
            model_version=self.model_version,
            protocol_fingerprint=self.fingerprint,
            source_event_ids=sources,
            known_at=clock,
        )


def _contribution_id(
    *,
    competition_set_id: str,
    rule_id: str,
    model_version: str,
    protocol_fingerprint: str,
    source_event_ids: tuple[str, ...],
    known_at: pd.Timestamp,
    evidence_family: str,
    correlation_key: str,
) -> str:
    digest = _canonical_hash(
        {
            "competition_set_id": competition_set_id,
            "rule_id": rule_id,
            "model_version": model_version,
            "protocol_fingerprint": protocol_fingerprint,
            "source_event_ids": source_event_ids,
            "known_at": known_at.isoformat(),
            "evidence_family": evidence_family,
            "correlation_key": correlation_key,
        }
    )
    return f"path-contribution:{digest[:32]}"


def _terminal_event_id(
    *,
    competition_set_id: str,
    path: PathKind,
    status: PathStatus,
    rule_id: str,
    reason: str,
    model_version: str,
    protocol_fingerprint: str,
    source_event_ids: tuple[str, ...],
    known_at: pd.Timestamp,
) -> str:
    digest = _canonical_hash(
        {
            "competition_set_id": competition_set_id,
            "path": path.value,
            "status": status.value,
            "rule_id": rule_id,
            "reason": reason,
            "model_version": model_version,
            "protocol_fingerprint": protocol_fingerprint,
            "source_event_ids": source_event_ids,
            "known_at": known_at.isoformat(),
        }
    )
    return f"path-terminal:{digest[:32]}"


def _outcome_event_id(
    *,
    competition_set_id: str,
    winner_path: PathKind,
    reason: str,
    model_version: str,
    protocol_fingerprint: str,
    source_event_ids: tuple[str, ...],
    known_at: pd.Timestamp,
) -> str:
    digest = _canonical_hash(
        {
            "competition_set_id": competition_set_id,
            "winner_path": PathKind(winner_path).value,
            "reason": reason,
            "model_version": model_version,
            "protocol_fingerprint": protocol_fingerprint,
            "source_event_ids": source_event_ids,
            "known_at": known_at.isoformat(),
        }
    )
    return f"path-outcome:{digest[:32]}"


def load_path_belief_protocol(
    path: str | Path = "configs/path_hypotheses.json",
) -> PathBeliefProtocol:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[1] / source
    try:
        raw_bytes = source.read_bytes()
        payload = json.loads(raw_bytes)
    except (OSError, ValueError) as error:
        raise PathBeliefProtocolError(
            f"cannot load path hypothesis protocol: {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise PathBeliefProtocolError("path hypothesis protocol root must be an object")
    try:
        schema_version = int(payload.get("schema_version", 0))
    except (TypeError, ValueError) as error:
        raise PathBeliefProtocolError("path protocol schema_version is invalid") from error
    status = str(payload.get("status", ""))
    authority = str(payload.get("authority", ""))
    if status != "development_unvalidated":
        raise PathBeliefProtocolError(
            "path protocol must remain development_unvalidated"
        )
    if authority != "shadow_only":
        raise PathBeliefProtocolError("path protocol must remain shadow_only")
    paths = payload.get("paths")
    if paths != [path.value for path in PATH_KINDS]:
        raise PathBeliefProtocolError(
            "path protocol must declare the exact mutually exclusive exhaustive order"
        )
    horizon = payload.get("common_horizon")
    if not isinstance(horizon, Mapping):
        raise PathBeliefProtocolError("path protocol common_horizon is missing")
    competition_contract = payload.get("competition_contract")
    if not isinstance(competition_contract, Mapping):
        raise PathBeliefProtocolError(
            "path protocol competition_contract is missing"
        )
    if set(competition_contract) != {
        "mutual_exclusivity",
        "exhaustiveness",
        "same_clock_precedence",
        "evidence_dependence",
    }:
        raise PathBeliefProtocolError(
            "path protocol competition_contract fields are not frozen exactly"
        )
    raw_resolution = payload.get("runtime_resolution")
    if not isinstance(raw_resolution, Mapping):
        raise PathBeliefProtocolError(
            "path protocol runtime_resolution is missing"
        )
    required_resolution = {
        "schema_version",
        "protocol_version",
        "status",
        "authority",
        "source_authority",
        "entry_episode_terminal_authority",
        "realized_winner_rules",
        "falsification_rules",
        "expiry_rules",
    }
    if set(raw_resolution) != required_resolution:
        raise PathBeliefProtocolError(
            "path runtime resolution fields are not frozen exactly"
        )
    raw_terminal = payload.get("terminal_rules")
    if not isinstance(raw_terminal, Mapping) or not raw_terminal:
        raise PathBeliefProtocolError("path protocol terminal_rules are missing")
    terminal_rules: list[tuple[str, PathStatus]] = []
    for rule_id, raw_status in sorted(raw_terminal.items()):
        try:
            terminal_rules.append((str(rule_id), PathStatus(raw_status)))
        except ValueError as error:
            raise PathBeliefProtocolError(
                f"path terminal rule {rule_id} has an invalid status"
            ) from error
    raw_rules = payload.get("evidence_rules")
    if not isinstance(raw_rules, Mapping) or not raw_rules:
        raise PathBeliefProtocolError("path protocol evidence_rules are missing")
    evidence_rules: list[PathEvidenceRule] = []
    for rule_id, raw_rule in sorted(raw_rules.items()):
        if not isinstance(raw_rule, Mapping):
            raise PathBeliefProtocolError(
                f"path evidence rule {rule_id} must be an object"
            )
        conditional = raw_rule.get("conditional_likelihood")
        conditional_values = (
            None
            if conditional is None
            else _positive_path_values(
                conditional,
                name=(
                    f"evidence_rules.{rule_id}.conditional_likelihood"
                ),
            )
        )
        if conditional_values is None:
            increments = _path_values(
                raw_rule.get("log_likelihood_increment", {}),
                name=f"evidence_rules.{rule_id}.log_likelihood_increment",
            )
        else:
            increments = tuple(
                (path, math.log(float(value)))
                for path, value in conditional_values
            )
            if "log_likelihood_increment" in raw_rule:
                declared_increments = _path_values(
                    raw_rule["log_likelihood_increment"],
                    name=(
                        f"evidence_rules.{rule_id}.log_likelihood_increment"
                    ),
                )
                if any(
                    path is not declared_path
                    or not math.isclose(
                        value,
                        declared_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for (path, value), (
                        declared_path,
                        declared_value,
                    ) in zip(increments, declared_increments, strict=True)
                ):
                    raise PathBeliefProtocolError(
                        "declared log likelihood increment does not match "
                        f"conditional likelihood for {rule_id}"
                    )
        evidence_rules.append(
            PathEvidenceRule(
                rule_id=str(rule_id),
                description=str(raw_rule.get("description", "")).strip(),
                log_likelihood_increments=increments,
                evidence_family=str(
                    raw_rule.get("evidence_family", rule_id)
                ).strip(),
                conditional_likelihoods=conditional_values,
            )
        )
    try:
        protocol = PathBeliefProtocol(
            schema_version=schema_version,
            protocol_version=str(payload.get("protocol_version", "")).strip(),
            model_version=str(payload.get("model_version", "")).strip(),
            status=status,
            authority=authority,
            probability_interpretation=str(
                payload.get("probability_interpretation", "")
            ).strip(),
            weight_interpretation=str(
                payload.get("weight_interpretation", "")
            ).strip(),
            common_horizon_rule=str(horizon.get("rule", "")).strip(),
            common_horizon_expiry_resolution=str(
                horizon.get("expiry_resolution", "")
            ).strip(),
            mutual_exclusivity_rule=str(
                competition_contract.get("mutual_exclusivity", "")
            ).strip(),
            exhaustiveness_rule=str(
                competition_contract.get("exhaustiveness", "")
            ).strip(),
            same_clock_precedence=str(
                competition_contract.get("same_clock_precedence", "")
            ).strip(),
            evidence_dependence_rule=str(
                competition_contract.get("evidence_dependence", "")
            ).strip(),
            model_admission_status=str(
                payload.get("model_admission_status", "")
            ).strip(),
            likelihood_artifact_status=str(
                payload.get("likelihood_artifact_status", "")
            ).strip(),
            action_authority_ready=payload.get("action_authority_ready"),
            evidence_allowlist=tuple(
                payload.get("phase6_evidence_allowlist", ())
            ),
            phase6_manifest_sha256=str(
                payload.get("phase6_manifest_sha256", "")
            ).strip(),
            phase6_result_sha256=str(
                payload.get("phase6_result_sha256", "")
            ).strip(),
            phase6_result_identity=str(
                payload.get("phase6_result_identity", "")
            ).strip(),
            prior_log_weights=_path_values(
                payload.get("prior_log_weights", {}),
                name="prior_log_weights",
            ),
            real_completed_bar_decay=_path_values(
                payload.get("real_completed_bar_decay", {}),
                name="real_completed_bar_decay",
                nonnegative=True,
            ),
            terminal_rules=tuple(terminal_rules),
            evidence_rules=tuple(evidence_rules),
            runtime_resolution=PathRuntimeResolutionProtocol(
                schema_version=raw_resolution["schema_version"],
                protocol_version=str(
                    raw_resolution["protocol_version"]
                ).strip(),
                status=str(raw_resolution["status"]).strip(),
                authority=str(raw_resolution["authority"]).strip(),
                source_authority=str(
                    raw_resolution["source_authority"]
                ).strip(),
                entry_episode_terminal_authority=raw_resolution[
                    "entry_episode_terminal_authority"
                ],
                realized_winner_rules=_path_text_values(
                    raw_resolution["realized_winner_rules"],
                    name="runtime_resolution.realized_winner_rules",
                ),
                falsification_rules=_path_text_values(
                    raw_resolution["falsification_rules"],
                    name="runtime_resolution.falsification_rules",
                ),
                expiry_rules=_path_text_values(
                    raw_resolution["expiry_rules"],
                    name="runtime_resolution.expiry_rules",
                ),
            ),
            fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
        )
    except ValueError as error:
        raise PathBeliefProtocolError(str(error)) from error
    return protocol


@dataclass(frozen=True)
class PathHypothesisState:
    hypothesis_id: str
    path: PathKind
    status: PathStatus
    log_weight: float | None
    probability: float
    common_expires_at: pd.Timestamp
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None
    terminal_source_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", PathKind(self.path))
        object.__setattr__(self, "status", PathStatus(self.status))
        object.__setattr__(
            self,
            "common_expires_at",
            _aware_timestamp(
                self.common_expires_at,
                name="path hypothesis common_expires_at",
            ),
        )
        terminal_at = self.terminal_at
        if terminal_at is not None:
            terminal_at = _aware_timestamp(
                terminal_at,
                name="path hypothesis terminal_at",
            )
            object.__setattr__(self, "terminal_at", terminal_at)
        raw_sources = tuple(self.terminal_source_event_ids)
        if (
            len(raw_sources) != len(set(raw_sources))
            or any(not isinstance(source, str) or not source for source in raw_sources)
        ):
            raise ValueError("path hypothesis terminal sources are invalid")
        sources = tuple(sorted(raw_sources))
        object.__setattr__(self, "terminal_source_event_ids", sources)
        active = self.status is PathStatus.ACTIVE
        realized = self.status is PathStatus.REALIZED
        if (
            not self.hypothesis_id
            or not math.isfinite(float(self.probability))
            or not 0.0 <= float(self.probability) <= 1.0
            or active != (self.log_weight is not None)
            or (
                self.log_weight is not None
                and not math.isfinite(float(self.log_weight))
            )
            or active != (terminal_at is None)
            or active != (self.terminal_reason is None)
            or (not active and not self.terminal_reason)
            or (active and bool(sources))
            or (self.status is PathStatus.INVALIDATED and not sources)
            or (realized and not sources)
            or (
                self.status is PathStatus.EXPIRED
                and self.terminal_reason == "common_horizon_elapsed"
                and bool(sources)
            )
            or (
                realized
                and not math.isclose(
                    self.probability,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
            or (not active and not realized and self.probability != 0.0)
        ):
            raise ValueError("path hypothesis lifecycle is invalid")


@dataclass(frozen=True)
class PathCompetitionSetState:
    schema_version: int
    competition_set_id: str
    instrument_id: str
    market_epoch_id: str
    authority_structure_id: str
    horizon_id: str
    protocol_version: str
    protocol_fingerprint: str
    model_version: str
    protocol_status: str
    authority: str
    formed_at: pd.Timestamp
    asof: pd.Timestamp
    common_expires_at: pd.Timestamp
    status: PathStatus
    members: tuple[PathHypothesisState, ...]
    applied_contribution_ids: tuple[str, ...] = ()
    applied_terminal_event_ids: tuple[str, ...] = ()
    evidence_ledger: tuple[PathEvidenceContribution, ...] = ()
    terminal_event_ledger: tuple[PathTerminalEvent, ...] = ()
    winner_path: PathKind | None = None
    outcome_event_id: str | None = None
    realized_at: pd.Timestamp | None = None
    outcome_source_event_ids: tuple[str, ...] = ()
    last_real_completed_at: pd.Timestamp | None = None
    real_completed_bar_count: int = 0

    def __post_init__(self) -> None:
        for name in ("formed_at", "asof", "common_expires_at"):
            object.__setattr__(
                self,
                name,
                _aware_timestamp(getattr(self, name), name=f"path set {name}"),
            )
        if self.last_real_completed_at is not None:
            object.__setattr__(
                self,
                "last_real_completed_at",
                _aware_timestamp(
                    self.last_real_completed_at,
                    name="path set last_real_completed_at",
                ),
            )
        if self.realized_at is not None:
            object.__setattr__(
                self,
                "realized_at",
                _aware_timestamp(self.realized_at, name="path set realized_at"),
            )
        object.__setattr__(self, "status", PathStatus(self.status))
        if self.winner_path is not None:
            object.__setattr__(self, "winner_path", PathKind(self.winner_path))
        raw_contributions = tuple(self.applied_contribution_ids)
        raw_terminals = tuple(self.applied_terminal_event_ids)
        if (
            len(raw_contributions) != len(set(raw_contributions))
            or len(raw_terminals) != len(set(raw_terminals))
        ):
            raise ValueError("path set applied event identities are duplicated")
        contributions = tuple(sorted(raw_contributions))
        terminals = tuple(sorted(raw_terminals))
        object.__setattr__(self, "applied_contribution_ids", contributions)
        object.__setattr__(self, "applied_terminal_event_ids", terminals)
        ledger = tuple(self.evidence_ledger)
        ledger_ids = tuple(item.contribution_id for item in ledger)
        ledger_correlation_instances = tuple(
            (item.evidence_family, item.correlation_key) for item in ledger
        )
        terminal_ledger = tuple(self.terminal_event_ledger)
        terminal_ledger_ids = tuple(
            item.terminal_event_id for item in terminal_ledger
        )
        outcome_sources = tuple(sorted(self.outcome_source_event_ids))
        object.__setattr__(self, "evidence_ledger", ledger)
        object.__setattr__(self, "terminal_event_ledger", terminal_ledger)
        object.__setattr__(self, "outcome_source_event_ids", outcome_sources)
        _validate_cross_family_dependency_clusters(ledger)
        identity_fields = (
            self.competition_set_id,
            self.instrument_id,
            self.market_epoch_id,
            self.authority_structure_id,
            self.horizon_id,
            self.protocol_version,
            self.protocol_fingerprint,
            self.model_version,
        )
        active_members = tuple(
            member for member in self.members if member.status is PathStatus.ACTIVE
        )
        expected_id = _competition_set_id(
            instrument_id=self.instrument_id,
            market_epoch_id=self.market_epoch_id,
            authority_structure_id=self.authority_structure_id,
            horizon_id=self.horizon_id,
            formed_at=self.formed_at,
            common_expires_at=self.common_expires_at,
            protocol_fingerprint=self.protocol_fingerprint,
            model_version=self.model_version,
        )
        invalid = bool(
            self.schema_version != PATH_STATE_SCHEMA_VERSION
            or any(not isinstance(value, str) or not value for value in identity_fields)
            or self.competition_set_id != expected_id
            or self.protocol_status != "development_unvalidated"
            or self.authority != "shadow_only"
            or self.formed_at > self.asof
            or self.formed_at >= self.common_expires_at
            or tuple(member.path for member in self.members) != PATH_KINDS
            or any(
                member.common_expires_at != self.common_expires_at
                for member in self.members
            )
            or any(
                member.hypothesis_id
                != _path_hypothesis_id(self.competition_set_id, member.path)
                for member in self.members
            )
            or len({member.hypothesis_id for member in self.members})
            != len(PATH_KINDS)
            or any(not value for value in (*contributions, *terminals))
            or any(
                not isinstance(item, PathEvidenceContribution)
                or item.competition_set_id != self.competition_set_id
                or item.model_version != self.model_version
                or item.protocol_fingerprint != self.protocol_fingerprint
                or item.known_at < self.formed_at
                or item.known_at > self.asof
                or item.known_at >= self.common_expires_at
                for item in ledger
            )
            or tuple(sorted(ledger_ids)) != contributions
            or ledger
            != tuple(
                sorted(
                    ledger,
                    key=lambda item: (item.known_at, item.contribution_id),
                )
            )
            or len(ledger_correlation_instances)
            != len(set(ledger_correlation_instances))
            or any(
                not isinstance(item, PathTerminalEvent)
                or item.competition_set_id != self.competition_set_id
                or item.model_version != self.model_version
                or item.protocol_fingerprint != self.protocol_fingerprint
                or item.known_at <= self.formed_at
                or item.known_at > self.asof
                or item.known_at > self.common_expires_at
                for item in terminal_ledger
            )
            or tuple(sorted(terminal_ledger_ids)) != terminals
            or terminal_ledger
            != tuple(
                sorted(
                    terminal_ledger,
                    key=lambda item: (item.known_at, item.terminal_event_id),
                )
            )
            or len({item.path for item in terminal_ledger})
            != len(terminal_ledger)
            or len(outcome_sources) != len(set(outcome_sources))
            or any(
                not isinstance(source_id, str) or not source_id
                for source_id in outcome_sources
            )
            or type(self.real_completed_bar_count) is not int
            or self.real_completed_bar_count < 0
            or (
                (self.last_real_completed_at is None)
                != (self.real_completed_bar_count == 0)
            )
            or (
                self.last_real_completed_at is not None
                and (
                    self.last_real_completed_at > self.asof
                    or self.last_real_completed_at < self.formed_at
                )
            )
        )
        if self.status is PathStatus.ACTIVE:
            invalid = invalid or bool(
                self.asof >= self.common_expires_at
                or not active_members
                or any(
                    member.status is PathStatus.REALIZED
                    for member in self.members
                )
                or self.member(PathKind.RESIDUAL_UNKNOWN).status
                is not PathStatus.ACTIVE
                or not math.isclose(
                    sum(member.probability for member in active_members),
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or self.winner_path is not None
                or self.outcome_event_id is not None
                or self.realized_at is not None
                or bool(outcome_sources)
            )
        elif self.status is PathStatus.EXPIRED:
            invalid = invalid or bool(
                self.asof < self.common_expires_at
                or active_members
                or any(member.probability != 0.0 for member in self.members)
                or self.winner_path is not None
                or self.outcome_event_id is not None
                or self.realized_at is not None
                or bool(outcome_sources)
            )
        elif self.status is PathStatus.REALIZED:
            realized_members = tuple(
                member
                for member in self.members
                if member.status is PathStatus.REALIZED
            )
            realized_member = (
                None if len(realized_members) != 1 else realized_members[0]
            )
            invalid = invalid or bool(
                self.winner_path is None
                or self.winner_path is PathKind.RESIDUAL_UNKNOWN
                or not self.outcome_event_id
                or self.realized_at is None
                or self.realized_at > self.asof
                or self.realized_at >= self.common_expires_at
                or not outcome_sources
                or len(realized_members) != 1
                or realized_members[0].path is not self.winner_path
                or realized_members[0].probability != 1.0
                or active_members
                or any(
                    member.status
                    not in {
                        PathStatus.REALIZED,
                        PathStatus.INVALIDATED,
                        PathStatus.EXPIRED,
                    }
                    for member in self.members
                )
                or realized_member is None
                or realized_member.terminal_at != self.realized_at
                or realized_member.terminal_source_event_ids
                != outcome_sources
                or self.outcome_event_id
                != _outcome_event_id(
                    competition_set_id=self.competition_set_id,
                    winner_path=self.winner_path,
                    reason=realized_member.terminal_reason or "",
                    model_version=self.model_version,
                    protocol_fingerprint=self.protocol_fingerprint,
                    source_event_ids=outcome_sources,
                    known_at=self.realized_at,
                )
            )
        else:
            invalid = True
        terminal_by_path = {item.path: item for item in terminal_ledger}
        for member in self.members:
            outcome_derived = bool(
                self.status is PathStatus.REALIZED
                and self.winner_path is not None
                and self.realized_at is not None
                and member.terminal_at == self.realized_at
                and member.terminal_source_event_ids == outcome_sources
                and (
                    (
                        member.path is self.winner_path
                        and member.status is PathStatus.REALIZED
                    )
                    or (
                        member.path is not self.winner_path
                        and member.status is PathStatus.INVALIDATED
                        and member.terminal_reason
                        == (
                            "competing_path_realized:"
                            f"{self.winner_path.value}"
                        )
                    )
                )
            )
            common_horizon_derived = bool(
                member.status is PathStatus.EXPIRED
                and member.terminal_at == self.common_expires_at
                and member.terminal_reason == "common_horizon_elapsed"
                and not member.terminal_source_event_ids
            )
            requires_terminal = bool(
                member.status is not PathStatus.ACTIVE
                and not outcome_derived
                and not common_horizon_derived
            )
            terminal = terminal_by_path.get(member.path)
            if requires_terminal:
                invalid = invalid or bool(
                    terminal is None
                    or terminal.status is not member.status
                    or terminal.known_at != member.terminal_at
                    or terminal.reason != member.terminal_reason
                    or terminal.source_event_ids
                    != member.terminal_source_event_ids
                )
            elif terminal is not None:
                invalid = True
        if invalid:
            raise ValueError("path competition set is invalid")

    def member(self, path: PathKind) -> PathHypothesisState:
        return self.members[PATH_KINDS.index(PathKind(path))]

    @property
    def evidence_ledger_fingerprint(self) -> str:
        return _canonical_hash(
            [_contribution_payload(item) for item in self.evidence_ledger]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "competition_set_id": self.competition_set_id,
            "instrument_id": self.instrument_id,
            "market_epoch_id": self.market_epoch_id,
            "authority_structure_id": self.authority_structure_id,
            "horizon_id": self.horizon_id,
            "protocol_version": self.protocol_version,
            "protocol_fingerprint": self.protocol_fingerprint,
            "model_version": self.model_version,
            "protocol_status": self.protocol_status,
            "authority": self.authority,
            "formed_at": self.formed_at.isoformat(),
            "asof": self.asof.isoformat(),
            "common_expires_at": self.common_expires_at.isoformat(),
            "status": self.status.value,
            "members": [
                {
                    "hypothesis_id": member.hypothesis_id,
                    "path": member.path.value,
                    "status": member.status.value,
                    "log_weight": member.log_weight,
                    "probability": member.probability,
                    "common_expires_at": member.common_expires_at.isoformat(),
                    "terminal_at": (
                        None
                        if member.terminal_at is None
                        else member.terminal_at.isoformat()
                    ),
                    "terminal_reason": member.terminal_reason,
                    "terminal_source_event_ids": list(
                        member.terminal_source_event_ids
                    ),
                }
                for member in self.members
            ],
            "applied_contribution_ids": list(self.applied_contribution_ids),
            "applied_terminal_event_ids": list(self.applied_terminal_event_ids),
            "evidence_ledger": [
                _contribution_payload(item) for item in self.evidence_ledger
            ],
            "terminal_event_ledger": [
                _terminal_payload(item) for item in self.terminal_event_ledger
            ],
            "winner_path": (
                None if self.winner_path is None else self.winner_path.value
            ),
            "outcome_event_id": self.outcome_event_id,
            "realized_at": (
                None if self.realized_at is None else self.realized_at.isoformat()
            ),
            "outcome_source_event_ids": list(self.outcome_source_event_ids),
            "last_real_completed_at": (
                None
                if self.last_real_completed_at is None
                else self.last_real_completed_at.isoformat()
            ),
            "real_completed_bar_count": self.real_completed_bar_count,
        }


@dataclass(frozen=True)
class PathBeliefUpdateRecord:
    update_id: str
    competition_set_id: str
    from_asof: pd.Timestamp
    asof: pd.Timestamp
    model_version: str
    protocol_fingerprint: str
    real_completed_bar: bool
    decay_applied: bool
    applied_contributions: tuple[PathEvidenceContribution, ...]
    duplicate_contribution_ids: tuple[str, ...]
    correlated_duplicate_contribution_ids: tuple[str, ...]
    applied_terminal_events: tuple[PathTerminalEvent, ...]
    duplicate_terminal_event_ids: tuple[str, ...]
    applied_outcome_event: PathOutcomeEvent | None
    before_log_weights: tuple[tuple[PathKind, float | None], ...]
    after_log_weights: tuple[tuple[PathKind, float | None], ...]
    after_probabilities: tuple[tuple[PathKind, float], ...]
    log_normalizer: float | None
    common_horizon_expired: bool
    set_status: PathStatus
    bayesian_update_applied: bool
    evidence_admission_only: bool
    initialization: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_asof",
            _aware_timestamp(self.from_asof, name="path update from_asof"),
        )
        object.__setattr__(
            self,
            "asof",
            _aware_timestamp(self.asof, name="path update asof"),
        )
        object.__setattr__(self, "set_status", PathStatus(self.set_status))
        if (
            not self.update_id
            or not self.competition_set_id
            or type(self.initialization) is not bool
            or (
                self.initialization
                and self.from_asof != self.asof
            )
            or (
                not self.initialization
                and self.from_asof >= self.asof
            )
            or type(self.real_completed_bar) is not bool
            or type(self.decay_applied) is not bool
            or type(self.common_horizon_expired) is not bool
            or type(self.bayesian_update_applied) is not bool
            or type(self.evidence_admission_only) is not bool
            or (
                self.bayesian_update_applied
                and self.evidence_admission_only
            )
            or (
                self.applied_outcome_event is not None
                and self.set_status is not PathStatus.REALIZED
            )
            or (
                self.log_normalizer is not None
                and not math.isfinite(float(self.log_normalizer))
            )
        ):
            raise ValueError("path belief update record is invalid")


def _competition_set_id(
    *,
    instrument_id: str,
    market_epoch_id: str,
    authority_structure_id: str,
    horizon_id: str,
    formed_at: pd.Timestamp,
    common_expires_at: pd.Timestamp,
    protocol_fingerprint: str,
    model_version: str,
) -> str:
    digest = _canonical_hash(
        {
            "instrument_id": instrument_id,
            "market_epoch_id": market_epoch_id,
            "authority_structure_id": authority_structure_id,
            "horizon_id": horizon_id,
            "formed_at": formed_at.isoformat(),
            "common_expires_at": common_expires_at.isoformat(),
            "protocol_fingerprint": protocol_fingerprint,
            "model_version": model_version,
        }
    )
    return f"path-competition:{digest[:32]}"


def _path_hypothesis_id(
    competition_set_id: str,
    path: PathKind,
) -> str:
    return (
        "path-hypothesis:"
        f"{_canonical_hash((competition_set_id, PathKind(path).value))[:32]}"
    )


def _contribution_payload(
    contribution: PathEvidenceContribution,
) -> dict[str, Any]:
    return {
        "contribution_id": contribution.contribution_id,
        "competition_set_id": contribution.competition_set_id,
        "rule_id": contribution.rule_id,
        "model_version": contribution.model_version,
        "protocol_fingerprint": contribution.protocol_fingerprint,
        "source_event_ids": list(contribution.source_event_ids),
        "known_at": contribution.known_at.isoformat(),
        "log_likelihood_increments": {
            path.value: value
            for path, value in contribution.log_likelihood_increments
        },
        "evidence_family": contribution.evidence_family,
        "correlation_key": contribution.correlation_key,
        "conditional_likelihoods": (
            None
            if contribution.conditional_likelihoods is None
            else {
                path.value: value
                for path, value in contribution.conditional_likelihoods
            }
        ),
    }


def _restore_contribution(payload: Mapping[str, Any]) -> PathEvidenceContribution:
    if not isinstance(payload, Mapping):
        raise ValueError("path evidence ledger entry must be an object")
    conditional = payload.get("conditional_likelihoods")
    return PathEvidenceContribution(
        contribution_id=str(payload["contribution_id"]),
        competition_set_id=str(payload["competition_set_id"]),
        rule_id=str(payload["rule_id"]),
        model_version=str(payload["model_version"]),
        protocol_fingerprint=str(payload["protocol_fingerprint"]),
        source_event_ids=tuple(payload["source_event_ids"]),
        known_at=pd.Timestamp(payload["known_at"]),
        log_likelihood_increments=_path_values(
            payload["log_likelihood_increments"],
            name="path evidence ledger log likelihoods",
        ),
        evidence_family=str(payload["evidence_family"]),
        correlation_key=str(payload["correlation_key"]),
        conditional_likelihoods=(
            None
            if conditional is None
            else _positive_path_values(
                conditional,
                name="path evidence ledger conditional likelihoods",
            )
        ),
    )


def _terminal_payload(terminal: PathTerminalEvent) -> dict[str, Any]:
    return {
        "terminal_event_id": terminal.terminal_event_id,
        "competition_set_id": terminal.competition_set_id,
        "path": terminal.path.value,
        "status": terminal.status.value,
        "rule_id": terminal.rule_id,
        "reason": terminal.reason,
        "model_version": terminal.model_version,
        "protocol_fingerprint": terminal.protocol_fingerprint,
        "source_event_ids": list(terminal.source_event_ids),
        "known_at": terminal.known_at.isoformat(),
    }


def _restore_terminal(payload: Mapping[str, Any]) -> PathTerminalEvent:
    if not isinstance(payload, Mapping):
        raise ValueError("path terminal ledger entry must be an object")
    return PathTerminalEvent(
        terminal_event_id=str(payload["terminal_event_id"]),
        competition_set_id=str(payload["competition_set_id"]),
        path=PathKind(payload["path"]),
        status=PathStatus(payload["status"]),
        rule_id=str(payload["rule_id"]),
        reason=str(payload["reason"]),
        model_version=str(payload["model_version"]),
        protocol_fingerprint=str(payload["protocol_fingerprint"]),
        source_event_ids=tuple(payload["source_event_ids"]),
        known_at=pd.Timestamp(payload["known_at"]),
    )


def _normalized_members(
    members: Sequence[PathHypothesisState],
) -> tuple[tuple[PathHypothesisState, ...], float | None]:
    active = tuple(member for member in members if member.status is PathStatus.ACTIVE)
    if not active:
        return (
            tuple(replace(member, probability=0.0) for member in members),
            None,
        )
    weights = tuple(float(member.log_weight) for member in active)
    maximum = max(weights)
    total = math.fsum(math.exp(value - maximum) for value in weights)
    log_normalizer = maximum + math.log(total)
    probabilities = {
        member.path: math.exp(float(member.log_weight) - log_normalizer)
        for member in active
    }
    normalized = tuple(
        replace(
            member,
            probability=(
                probabilities[member.path]
                if member.status is PathStatus.ACTIVE
                else 0.0
            ),
        )
        for member in members
    )
    return normalized, log_normalizer


def _validate_cross_family_dependency_clusters(
    contributions: Sequence[PathEvidenceContribution],
) -> None:
    """Reject independent multipliers for one declared evidence cluster.

    ``correlation_key`` is a global upstream dependency identity, not a
    family-local de-duplication token.  Different families may share a cluster
    in the neutral source ledger, but an admitted likelihood model must
    represent them as one precombined contribution (or a separately
    registered history-conditioned contribution with a distinct key).
    """

    by_key: dict[str, PathEvidenceContribution] = {}
    fitted_by_source: dict[str, PathEvidenceContribution] = {}
    for contribution in sorted(
        contributions,
        key=lambda value: (value.known_at, value.contribution_id),
    ):
        if contribution.conditional_likelihoods is not None:
            for source_event_id in contribution.source_event_ids:
                prior_source = fitted_by_source.get(source_event_id)
                if (
                    prior_source is not None
                    and prior_source.correlation_key
                    != contribution.correlation_key
                ):
                    raise ValueError(
                        "one source event cannot declare multiple dependency "
                        "clusters"
                    )
                fitted_by_source[source_event_id] = contribution
        prior = by_key.get(contribution.correlation_key)
        if prior is None:
            by_key[contribution.correlation_key] = contribution
            continue
        if (
            prior.evidence_family != contribution.evidence_family
            and (
                prior.conditional_likelihoods is not None
                or contribution.conditional_likelihoods is not None
            )
        ):
            raise ValueError(
                "cross-family evidence in one dependency cluster requires "
                "one registered joint or history-conditioned contribution"
            )


def create_path_competition_set(
    protocol: PathBeliefProtocol,
    *,
    instrument_id: str,
    market_epoch_id: str,
    authority_structure_id: str,
    horizon_id: str,
    formed_at: pd.Timestamp,
    common_expires_at: pd.Timestamp,
) -> PathCompetitionSetState:
    if not isinstance(protocol, PathBeliefProtocol):
        raise TypeError("path competition set requires PathBeliefProtocol")
    clock = _aware_timestamp(formed_at, name="path competition formed_at")
    expiry = _aware_timestamp(
        common_expires_at,
        name="path competition common_expires_at",
    )
    identities = (
        instrument_id,
        market_epoch_id,
        authority_structure_id,
        horizon_id,
    )
    if (
        any(not isinstance(value, str) or not value for value in identities)
        or clock >= expiry
    ):
        raise ValueError("path competition scope or common horizon is invalid")
    competition_id = _competition_set_id(
        instrument_id=instrument_id,
        market_epoch_id=market_epoch_id,
        authority_structure_id=authority_structure_id,
        horizon_id=horizon_id,
        formed_at=clock,
        common_expires_at=expiry,
        protocol_fingerprint=protocol.fingerprint,
        model_version=protocol.model_version,
    )
    members, _ = _normalized_members(
        tuple(
            PathHypothesisState(
                hypothesis_id=_path_hypothesis_id(competition_id, path),
                path=path,
                status=PathStatus.ACTIVE,
                log_weight=protocol.prior(path),
                probability=0.0,
                common_expires_at=expiry,
            )
            for path in PATH_KINDS
        )
    )
    return PathCompetitionSetState(
        schema_version=PATH_STATE_SCHEMA_VERSION,
        competition_set_id=competition_id,
        instrument_id=instrument_id,
        market_epoch_id=market_epoch_id,
        authority_structure_id=authority_structure_id,
        horizon_id=horizon_id,
        protocol_version=protocol.protocol_version,
        protocol_fingerprint=protocol.fingerprint,
        model_version=protocol.model_version,
        protocol_status=protocol.status,
        authority=protocol.authority,
        formed_at=clock,
        asof=clock,
        common_expires_at=expiry,
        status=PathStatus.ACTIVE,
        members=members,
    )


def initialize_path_competition_set(
    protocol: PathBeliefProtocol,
    state: PathCompetitionSetState,
    *,
    initial_contributions: Sequence[PathEvidenceContribution] = (),
    initial_real_completed_bar: bool,
) -> tuple[PathCompetitionSetState, PathBeliefUpdateRecord]:
    """Assimilate exact evidence at a competition set's formation clock.

    Normal updates remain strictly clock-increasing.  This one-shot helper is
    limited to a pristine freshly-created set so callers never need to forge a
    sub-clock or delay evidence first known at the formation clock.
    """

    if type(initial_real_completed_bar) is not bool:
        raise TypeError("initial_real_completed_bar must be boolean")
    _validate_protocol_binding(protocol, state)
    if (
        state.status is not PathStatus.ACTIVE
        or state.asof != state.formed_at
        or state.applied_contribution_ids
        or state.applied_terminal_event_ids
        or state.last_real_completed_at is not None
        or state.real_completed_bar_count != 0
        or any(
            member.status is not PathStatus.ACTIVE
            or member.log_weight != protocol.prior(member.path)
            for member in state.members
        )
    ):
        raise ValueError("path competition initialization requires a pristine set")

    _validate_cross_family_dependency_clusters(initial_contributions)

    seen: dict[str, PathEvidenceContribution] = {}
    applied: list[PathEvidenceContribution] = []
    duplicate_ids: set[str] = set()
    correlated_duplicate_ids: set[str] = set()
    correlation_instances: set[tuple[str, str]] = set()
    for contribution in sorted(
        initial_contributions,
        key=lambda value: (value.known_at, value.contribution_id),
    ):
        prior = seen.get(contribution.contribution_id)
        if prior is not None and prior != contribution:
            raise ValueError("duplicate path contribution identity has two payloads")
        seen[contribution.contribution_id] = contribution
        if (
            contribution.competition_set_id != state.competition_set_id
            or contribution.model_version != protocol.model_version
            or contribution.protocol_fingerprint != protocol.fingerprint
        ):
            raise ValueError("path contribution model or protocol binding is invalid")
        if contribution.contribution_id != _contribution_id(
            competition_set_id=contribution.competition_set_id,
            rule_id=contribution.rule_id,
            model_version=contribution.model_version,
            protocol_fingerprint=contribution.protocol_fingerprint,
            source_event_ids=contribution.source_event_ids,
            known_at=contribution.known_at,
            evidence_family=contribution.evidence_family,
            correlation_key=contribution.correlation_key,
        ):
            raise ValueError("path contribution identity is not deterministic")
        rule = protocol.evidence_rule(contribution.rule_id)
        if (
            contribution.log_likelihood_increments
            != rule.log_likelihood_increments
            or contribution.evidence_family != rule.evidence_family
            or contribution.conditional_likelihoods
            != rule.conditional_likelihoods
        ):
            raise ValueError("path contribution differs from its registered rule")
        if contribution.known_at != state.formed_at:
            raise ValueError(
                "initial path contribution must use the exact formation clock"
            )
        if contribution.contribution_id in {
            item.contribution_id for item in applied
        }:
            duplicate_ids.add(contribution.contribution_id)
            continue
        correlation_instance = (
            contribution.evidence_family,
            contribution.correlation_key,
        )
        if correlation_instance in correlation_instances:
            correlated_duplicate_ids.add(contribution.contribution_id)
            continue
        correlation_instances.add(correlation_instance)
        applied.append(contribution)

    before_logs = tuple(
        (member.path, member.log_weight) for member in state.members
    )
    members = list(state.members)
    initial_decay_applied = bool(
        initial_real_completed_bar
        and any(protocol.decay(path) > 0.0 for path in PATH_KINDS)
    )
    if initial_decay_applied:
        members = [
            replace(
                member,
                log_weight=(
                    float(member.log_weight) - protocol.decay(member.path)
                ),
            )
            for member in members
        ]
    for contribution in applied:
        members = [
            replace(
                member,
                log_weight=(
                    float(member.log_weight)
                    + contribution.increment(member.path)
                ),
            )
            for member in members
        ]
    normalized, log_normalizer = _normalized_members(members)
    initialized = replace(
        state,
        members=normalized,
        applied_contribution_ids=tuple(
            sorted(item.contribution_id for item in applied)
        ),
        evidence_ledger=tuple(applied),
        last_real_completed_at=(
            state.formed_at if initial_real_completed_bar else None
        ),
        real_completed_bar_count=int(initial_real_completed_bar),
    )
    after_logs = tuple(
        (member.path, member.log_weight) for member in initialized.members
    )
    after_probabilities = tuple(
        (member.path, member.probability) for member in initialized.members
    )
    record_payload = {
        "initialization": True,
        "competition_set_id": state.competition_set_id,
        "asof": state.formed_at.isoformat(),
        "real_completed_bar": initial_real_completed_bar,
        "applied_contribution_ids": [
            item.contribution_id for item in applied
        ],
        "duplicate_contribution_ids": sorted(duplicate_ids),
        "correlated_duplicate_contribution_ids": sorted(
            correlated_duplicate_ids
        ),
        "after_log_weights": [
            (path.value, value) for path, value in after_logs
        ],
        "after_probabilities": [
            (path.value, value) for path, value in after_probabilities
        ],
    }
    record = PathBeliefUpdateRecord(
        update_id=f"path-update:{_canonical_hash(record_payload)[:32]}",
        competition_set_id=state.competition_set_id,
        from_asof=state.formed_at,
        asof=state.formed_at,
        model_version=protocol.model_version,
        protocol_fingerprint=protocol.fingerprint,
        real_completed_bar=initial_real_completed_bar,
        decay_applied=initial_decay_applied,
        applied_contributions=tuple(applied),
        duplicate_contribution_ids=tuple(sorted(duplicate_ids)),
        correlated_duplicate_contribution_ids=tuple(
            sorted(correlated_duplicate_ids)
        ),
        applied_terminal_events=(),
        duplicate_terminal_event_ids=(),
        applied_outcome_event=None,
        before_log_weights=before_logs,
        after_log_weights=after_logs,
        after_probabilities=after_probabilities,
        log_normalizer=log_normalizer,
        common_horizon_expired=False,
        set_status=PathStatus.ACTIVE,
        bayesian_update_applied=bool(
            applied
            and all(item.conditional_likelihoods is not None for item in applied)
        ),
        evidence_admission_only=bool(
            applied
            and all(item.conditional_likelihoods is None for item in applied)
        ),
        initialization=True,
    )
    return initialized, record


def _validate_protocol_binding(
    protocol: PathBeliefProtocol,
    state: PathCompetitionSetState,
) -> None:
    if (
        state.protocol_version != protocol.protocol_version
        or state.protocol_fingerprint != protocol.fingerprint
        or state.model_version != protocol.model_version
        or state.protocol_status != protocol.status
        or state.authority != protocol.authority
    ):
        raise ValueError("path state model or protocol binding is stale")


def reduce_path_competition_set(
    protocol: PathBeliefProtocol,
    state: PathCompetitionSetState,
    *,
    asof: pd.Timestamp,
    contributions: Sequence[PathEvidenceContribution] = (),
    terminal_events: Sequence[PathTerminalEvent] = (),
    outcome_events: Sequence[PathOutcomeEvent] = (),
    real_completed_bar: bool,
) -> tuple[PathCompetitionSetState, PathBeliefUpdateRecord]:
    """Apply one clock of evidence to an existing competition set.

    The caller must invoke this reducer once per normalized clock.  A real
    completed bar applies exactly one preregistered decay step; a synthetic
    clock advances ``asof`` but never changes evidence age.
    """

    if type(real_completed_bar) is not bool:
        raise TypeError("real_completed_bar must be boolean")
    _validate_protocol_binding(protocol, state)
    clock = _aware_timestamp(asof, name="path update asof")
    if state.status is not PathStatus.ACTIVE:
        raise ValueError("a terminal path competition set cannot be updated")
    if clock <= state.asof:
        raise ValueError("path update clock must advance strictly")

    before_members = state.members
    before_logs = tuple((member.path, member.log_weight) for member in before_members)
    members = list(before_members)
    member_index = {member.path: index for index, member in enumerate(members)}

    applied_contribution_ids = set(state.applied_contribution_ids)
    applied_terminal_ids = set(state.applied_terminal_event_ids)
    new_contributions: list[PathEvidenceContribution] = []
    duplicate_contributions: set[str] = set()
    correlated_duplicate_contributions: set[str] = set()
    applied_correlation_instances = {
        (item.evidence_family, item.correlation_key)
        for item in state.evidence_ledger
    }
    seen_contributions: dict[str, PathEvidenceContribution] = {}
    _validate_cross_family_dependency_clusters(
        (*state.evidence_ledger, *contributions)
    )
    for contribution in sorted(
        contributions,
        key=lambda value: (value.known_at, value.contribution_id),
    ):
        existing = seen_contributions.get(contribution.contribution_id)
        if existing is not None and existing != contribution:
            raise ValueError("duplicate path contribution identity has two payloads")
        seen_contributions[contribution.contribution_id] = contribution
        if (
            contribution.competition_set_id != state.competition_set_id
            or contribution.model_version != protocol.model_version
            or contribution.protocol_fingerprint != protocol.fingerprint
        ):
            raise ValueError("path contribution model or protocol binding is invalid")
        if contribution.contribution_id != _contribution_id(
            competition_set_id=contribution.competition_set_id,
            rule_id=contribution.rule_id,
            model_version=contribution.model_version,
            protocol_fingerprint=contribution.protocol_fingerprint,
            source_event_ids=contribution.source_event_ids,
            known_at=contribution.known_at,
            evidence_family=contribution.evidence_family,
            correlation_key=contribution.correlation_key,
        ):
            raise ValueError("path contribution identity is not deterministic")
        rule = protocol.evidence_rule(contribution.rule_id)
        if (
            contribution.log_likelihood_increments
            != rule.log_likelihood_increments
            or contribution.evidence_family != rule.evidence_family
            or contribution.conditional_likelihoods
            != rule.conditional_likelihoods
        ):
            raise ValueError("path contribution differs from its registered rule")
        if contribution.contribution_id in applied_contribution_ids:
            duplicate_contributions.add(contribution.contribution_id)
            continue
        if contribution.known_at > clock:
            raise ValueError("path contribution contains future evidence")
        if contribution.known_at <= state.asof:
            raise ValueError(
                "new path contribution must follow the reducer state clock"
            )
        if contribution.known_at >= state.common_expires_at:
            raise ValueError("path contribution is not known before the common horizon")
        if (
            contribution.contribution_id
            in {item.contribution_id for item in new_contributions}
        ):
            duplicate_contributions.add(contribution.contribution_id)
            continue
        correlation_instance = (
            contribution.evidence_family,
            contribution.correlation_key,
        )
        if correlation_instance in applied_correlation_instances:
            correlated_duplicate_contributions.add(
                contribution.contribution_id
            )
            continue
        applied_correlation_instances.add(correlation_instance)
        new_contributions.append(contribution)

    new_terminals: list[PathTerminalEvent] = []
    duplicate_terminals: set[str] = set()
    seen_terminals: dict[str, PathTerminalEvent] = {}
    for terminal in sorted(
        terminal_events,
        key=lambda value: (value.known_at, value.terminal_event_id),
    ):
        existing = seen_terminals.get(terminal.terminal_event_id)
        if existing is not None and existing != terminal:
            raise ValueError("duplicate path terminal identity has two payloads")
        seen_terminals[terminal.terminal_event_id] = terminal
        if (
            terminal.competition_set_id != state.competition_set_id
            or terminal.model_version != protocol.model_version
            or terminal.protocol_fingerprint != protocol.fingerprint
        ):
            raise ValueError("path terminal model or protocol binding is invalid")
        if terminal.terminal_event_id != _terminal_event_id(
            competition_set_id=terminal.competition_set_id,
            path=terminal.path,
            status=terminal.status,
            rule_id=terminal.rule_id,
            reason=terminal.reason,
            model_version=terminal.model_version,
            protocol_fingerprint=terminal.protocol_fingerprint,
            source_event_ids=terminal.source_event_ids,
            known_at=terminal.known_at,
        ):
            raise ValueError("path terminal identity is not deterministic")
        if terminal.status is not protocol.terminal_status(terminal.rule_id):
            raise ValueError("path terminal differs from its registered rule")
        if terminal.terminal_event_id in applied_terminal_ids:
            duplicate_terminals.add(terminal.terminal_event_id)
            continue
        if terminal.known_at > clock:
            raise ValueError("path terminal contains a future clock")
        if terminal.known_at <= state.asof:
            raise ValueError(
                "new path terminal must follow the reducer state clock"
            )
        if terminal.known_at > state.common_expires_at:
            raise ValueError("path terminal follows the common horizon")
        if terminal.path is PathKind.RESIDUAL_UNKNOWN:
            raise ValueError(
                "residual_unknown cannot terminate before the common horizon"
            )
        if (
            terminal.terminal_event_id
            in {item.terminal_event_id for item in new_terminals}
        ):
            duplicate_terminals.add(terminal.terminal_event_id)
            continue
        if members[member_index[terminal.path]].status is not PathStatus.ACTIVE:
            raise ValueError("a terminal path cannot receive a different terminal event")
        new_terminals.append(terminal)

    outcomes = tuple(sorted(
        outcome_events,
        key=lambda value: (value.known_at, value.outcome_event_id),
    ))
    if len({item.outcome_event_id for item in outcomes}) != len(outcomes):
        unique: dict[str, PathOutcomeEvent] = {}
        for item in outcomes:
            prior = unique.get(item.outcome_event_id)
            if prior is not None and prior != item:
                raise ValueError("one path outcome identity has conflicting payloads")
            unique[item.outcome_event_id] = item
        if len(unique) == 1 and len(outcomes) > 1:
            outcomes = (next(iter(unique.values())),)
    if len(outcomes) > 1:
        raise ValueError("one competition set cannot realize multiple outcomes")
    outcome = None if not outcomes else outcomes[0]
    if outcome is not None:
        if (
            outcome.competition_set_id != state.competition_set_id
            or outcome.model_version != protocol.model_version
            or outcome.protocol_fingerprint != protocol.fingerprint
        ):
            raise ValueError("path outcome model or protocol binding is invalid")
        if outcome.outcome_event_id != _outcome_event_id(
            competition_set_id=outcome.competition_set_id,
            winner_path=outcome.winner_path,
            reason=outcome.reason,
            model_version=outcome.model_version,
            protocol_fingerprint=outcome.protocol_fingerprint,
            source_event_ids=outcome.source_event_ids,
            known_at=outcome.known_at,
        ):
            raise ValueError("path outcome identity is not deterministic")
        if outcome.known_at != clock:
            raise ValueError("path outcome must use the exact reducer clock")
        if outcome.known_at >= state.common_expires_at:
            raise ValueError("path outcome is not known before the common horizon")

    decay_applied = bool(
        real_completed_bar
        and clock < state.common_expires_at
        and any(protocol.decay(path) > 0.0 for path in PATH_KINDS)
    )
    if decay_applied:
        for index, member in enumerate(members):
            if member.status is PathStatus.ACTIVE:
                members[index] = replace(
                    member,
                    log_weight=float(member.log_weight) - protocol.decay(member.path),
                )

    ordered_events: list[tuple[pd.Timestamp, int, str, Any]] = [
        (terminal.known_at, 0, terminal.terminal_event_id, terminal)
        for terminal in new_terminals
    ] + [
        (
            contribution.known_at,
            1,
            contribution.contribution_id,
            contribution,
        )
        for contribution in new_contributions
    ]
    for _, event_priority, _, event in sorted(ordered_events):
        if event_priority == 0:
            terminal = event
            index = member_index[terminal.path]
            member = members[index]
            if member.status is not PathStatus.ACTIVE:
                raise ValueError("same-clock terminal events conflict for one path")
            members[index] = replace(
                member,
                status=terminal.status,
                log_weight=None,
                probability=0.0,
                terminal_at=terminal.known_at,
                terminal_reason=terminal.reason,
                terminal_source_event_ids=terminal.source_event_ids,
            )
            applied_terminal_ids.add(terminal.terminal_event_id)
            continue
        contribution = event
        for index, member in enumerate(members):
            if member.status is PathStatus.ACTIVE:
                members[index] = replace(
                    member,
                    log_weight=(
                        float(member.log_weight)
                        + contribution.increment(member.path)
                    ),
                )
        applied_contribution_ids.add(contribution.contribution_id)

    common_horizon_expired = bool(
        outcome is None and clock >= state.common_expires_at
    )
    if outcome is not None:
        winner = members[member_index[outcome.winner_path]]
        if winner.status is not PathStatus.ACTIVE:
            raise ValueError("a terminal path cannot become the realized winner")
        members = [
            replace(
                member,
                status=(
                    PathStatus.REALIZED
                    if member.path is outcome.winner_path
                    else PathStatus.INVALIDATED
                ),
                log_weight=None,
                probability=(1.0 if member.path is outcome.winner_path else 0.0),
                terminal_at=outcome.known_at,
                terminal_reason=(
                    outcome.reason
                    if member.path is outcome.winner_path
                    else f"competing_path_realized:{outcome.winner_path.value}"
                ),
                terminal_source_event_ids=outcome.source_event_ids,
            )
            if member.status is PathStatus.ACTIVE
            else replace(member, probability=0.0)
            for member in members
        ]
        normalized_members = tuple(members)
        log_normalizer = None
        set_status = PathStatus.REALIZED
    elif common_horizon_expired:
        members = [
            replace(
                member,
                status=PathStatus.EXPIRED,
                log_weight=None,
                probability=0.0,
                terminal_at=state.common_expires_at,
                terminal_reason="common_horizon_elapsed",
                terminal_source_event_ids=(),
            )
            if member.status is PathStatus.ACTIVE
            else replace(member, probability=0.0)
            for member in members
        ]
        normalized_members = tuple(members)
        log_normalizer = None
        set_status = PathStatus.EXPIRED
    else:
        normalized_members, log_normalizer = _normalized_members(members)
        set_status = PathStatus.ACTIVE

    updated = PathCompetitionSetState(
        schema_version=state.schema_version,
        competition_set_id=state.competition_set_id,
        instrument_id=state.instrument_id,
        market_epoch_id=state.market_epoch_id,
        authority_structure_id=state.authority_structure_id,
        horizon_id=state.horizon_id,
        protocol_version=state.protocol_version,
        protocol_fingerprint=state.protocol_fingerprint,
        model_version=state.model_version,
        protocol_status=state.protocol_status,
        authority=state.authority,
        formed_at=state.formed_at,
        asof=clock,
        common_expires_at=state.common_expires_at,
        status=set_status,
        members=normalized_members,
        applied_contribution_ids=tuple(sorted(applied_contribution_ids)),
        applied_terminal_event_ids=tuple(sorted(applied_terminal_ids)),
        evidence_ledger=tuple((*state.evidence_ledger, *new_contributions)),
        terminal_event_ledger=tuple(
            (*state.terminal_event_ledger, *new_terminals)
        ),
        winner_path=(None if outcome is None else outcome.winner_path),
        outcome_event_id=(
            None if outcome is None else outcome.outcome_event_id
        ),
        realized_at=(None if outcome is None else outcome.known_at),
        outcome_source_event_ids=(
            () if outcome is None else outcome.source_event_ids
        ),
        last_real_completed_at=(
            clock if real_completed_bar else state.last_real_completed_at
        ),
        real_completed_bar_count=(
            state.real_completed_bar_count + int(real_completed_bar)
        ),
    )
    after_logs = tuple((member.path, member.log_weight) for member in updated.members)
    after_probabilities = tuple(
        (member.path, member.probability) for member in updated.members
    )
    update_payload = {
        "competition_set_id": state.competition_set_id,
        "from_asof": state.asof.isoformat(),
        "asof": clock.isoformat(),
        "real_completed_bar": real_completed_bar,
        "applied_contribution_ids": [
            item.contribution_id for item in new_contributions
        ],
        "duplicate_contribution_ids": sorted(duplicate_contributions),
        "correlated_duplicate_contribution_ids": sorted(
            correlated_duplicate_contributions
        ),
        "applied_terminal_event_ids": [
            item.terminal_event_id for item in new_terminals
        ],
        "duplicate_terminal_event_ids": sorted(duplicate_terminals),
        "outcome_event_id": (
            None if outcome is None else outcome.outcome_event_id
        ),
        "after_log_weights": [
            (path.value, value) for path, value in after_logs
        ],
        "after_probabilities": [
            (path.value, value) for path, value in after_probabilities
        ],
        "set_status": set_status.value,
    }
    record = PathBeliefUpdateRecord(
        update_id=f"path-update:{_canonical_hash(update_payload)[:32]}",
        competition_set_id=state.competition_set_id,
        from_asof=state.asof,
        asof=clock,
        model_version=protocol.model_version,
        protocol_fingerprint=protocol.fingerprint,
        real_completed_bar=real_completed_bar,
        decay_applied=decay_applied,
        applied_contributions=tuple(new_contributions),
        duplicate_contribution_ids=tuple(sorted(duplicate_contributions)),
        correlated_duplicate_contribution_ids=tuple(
            sorted(correlated_duplicate_contributions)
        ),
        applied_terminal_events=tuple(new_terminals),
        duplicate_terminal_event_ids=tuple(sorted(duplicate_terminals)),
        applied_outcome_event=outcome,
        before_log_weights=before_logs,
        after_log_weights=after_logs,
        after_probabilities=after_probabilities,
        log_normalizer=log_normalizer,
        common_horizon_expired=common_horizon_expired,
        set_status=set_status,
        bayesian_update_applied=bool(
            new_contributions
            and all(
                item.conditional_likelihoods is not None
                for item in new_contributions
            )
        ),
        evidence_admission_only=bool(
            new_contributions
            and all(
                item.conditional_likelihoods is None
                for item in new_contributions
            )
        ),
    )
    return updated, record


def restore_path_competition_set(
    payload: Mapping[str, Any],
    *,
    protocol: PathBeliefProtocol,
) -> PathCompetitionSetState:
    """Restore and validate an exact checkpoint-bound competition set."""

    if not isinstance(payload, Mapping):
        raise ValueError("path competition checkpoint root must be a mapping")
    if payload.get("schema_version") != PATH_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported path competition checkpoint schema")
    if (
        payload.get("protocol_version") != protocol.protocol_version
        or payload.get("protocol_fingerprint") != protocol.fingerprint
        or payload.get("model_version") != protocol.model_version
        or payload.get("protocol_status") != protocol.status
        or payload.get("authority") != protocol.authority
    ):
        raise ValueError("path checkpoint model or protocol binding is stale")
    raw_members = payload.get("members")
    if not isinstance(raw_members, list):
        raise ValueError("path checkpoint members are missing")
    members = tuple(
        PathHypothesisState(
            hypothesis_id=str(item["hypothesis_id"]),
            path=PathKind(item["path"]),
            status=PathStatus(item["status"]),
            log_weight=(
                None
                if item.get("log_weight") is None
                else float(item["log_weight"])
            ),
            probability=float(item["probability"]),
            common_expires_at=pd.Timestamp(item["common_expires_at"]),
            terminal_at=(
                None
                if item.get("terminal_at") is None
                else pd.Timestamp(item["terminal_at"])
            ),
            terminal_reason=item.get("terminal_reason"),
            terminal_source_event_ids=tuple(
                item.get("terminal_source_event_ids", ())
            ),
        )
        for item in raw_members
        if isinstance(item, Mapping)
    )
    if len(members) != len(raw_members):
        raise ValueError("path checkpoint member is not an object")
    raw_ledger = payload.get("evidence_ledger")
    if not isinstance(raw_ledger, list):
        raise ValueError("path checkpoint evidence ledger is missing")
    evidence_ledger = tuple(
        _restore_contribution(item) for item in raw_ledger
    )
    _validate_cross_family_dependency_clusters(evidence_ledger)
    for contribution in evidence_ledger:
        if not protocol.is_evidence_admitted(contribution.rule_id):
            raise ValueError(
                "path checkpoint contains unadmitted evidence"
            )
        rule = protocol.evidence_rule(contribution.rule_id)
        if (
            contribution.log_likelihood_increments
            != rule.log_likelihood_increments
            or contribution.evidence_family != rule.evidence_family
            or contribution.conditional_likelihoods
            != rule.conditional_likelihoods
        ):
            raise ValueError(
                "path checkpoint evidence differs from its registered rule"
            )
    raw_terminal_ledger = payload.get("terminal_event_ledger")
    if not isinstance(raw_terminal_ledger, list):
        raise ValueError("path checkpoint terminal ledger is missing")
    terminal_ledger = tuple(
        _restore_terminal(item) for item in raw_terminal_ledger
    )
    for terminal in terminal_ledger:
        if terminal.status is not protocol.terminal_status(terminal.rule_id):
            raise ValueError(
                "path checkpoint terminal differs from its registered rule"
            )
    state = PathCompetitionSetState(
        schema_version=int(payload["schema_version"]),
        competition_set_id=str(payload["competition_set_id"]),
        instrument_id=str(payload["instrument_id"]),
        market_epoch_id=str(payload["market_epoch_id"]),
        authority_structure_id=str(payload["authority_structure_id"]),
        horizon_id=str(payload["horizon_id"]),
        protocol_version=str(payload["protocol_version"]),
        protocol_fingerprint=str(payload["protocol_fingerprint"]),
        model_version=str(payload["model_version"]),
        protocol_status=str(payload["protocol_status"]),
        authority=str(payload["authority"]),
        formed_at=pd.Timestamp(payload["formed_at"]),
        asof=pd.Timestamp(payload["asof"]),
        common_expires_at=pd.Timestamp(payload["common_expires_at"]),
        status=PathStatus(payload["status"]),
        members=members,
        applied_contribution_ids=tuple(
            payload.get("applied_contribution_ids", ())
        ),
        applied_terminal_event_ids=tuple(
            payload.get("applied_terminal_event_ids", ())
        ),
        evidence_ledger=evidence_ledger,
        terminal_event_ledger=terminal_ledger,
        winner_path=(
            None
            if payload.get("winner_path") is None
            else PathKind(payload["winner_path"])
        ),
        outcome_event_id=payload.get("outcome_event_id"),
        realized_at=(
            None
            if payload.get("realized_at") is None
            else pd.Timestamp(payload["realized_at"])
        ),
        outcome_source_event_ids=tuple(
            payload.get("outcome_source_event_ids", ())
        ),
        last_real_completed_at=(
            None
            if payload.get("last_real_completed_at") is None
            else pd.Timestamp(payload["last_real_completed_at"])
        ),
        real_completed_bar_count=int(payload.get("real_completed_bar_count", 0)),
    )
    _validate_protocol_binding(protocol, state)
    if state.status is PathStatus.ACTIVE:
        active_members = tuple(
            member
            for member in state.members
            if member.status is PathStatus.ACTIVE
        )
        expected_logs = {
            member.path: (
                protocol.prior(member.path)
                - protocol.decay(member.path)
                * state.real_completed_bar_count
                + math.fsum(
                    contribution.increment(member.path)
                    for contribution in state.evidence_ledger
                )
            )
            for member in active_members
        }
        maximum = max(expected_logs.values())
        normalizer = maximum + math.log(
            math.fsum(
                math.exp(value - maximum)
                for value in expected_logs.values()
            )
        )
        if any(
            not math.isclose(
                float(member.log_weight),
                expected_logs[member.path],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                member.probability,
                math.exp(expected_logs[member.path] - normalizer),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for member in active_members
        ):
            raise ValueError(
                "path checkpoint posterior differs from its causal ledger"
            )
    return state


class BayesianBeliefUpdater:
    """Narrow admitted-evidence boundary around the pure path reducer.

    When the protocol has no separately fitted/admitted conditional-likelihood
    artifact, admitted evidence is still journaled with neutral increments but
    no Bayesian-update claim is made.  This is the production configuration at
    the Phase 6 -> Phase 7 boundary.
    """

    def __init__(self, protocol: PathBeliefProtocol) -> None:
        if not isinstance(protocol, PathBeliefProtocol):
            raise TypeError("Bayesian updater requires PathBeliefProtocol")
        self.protocol = protocol

    @property
    def action_authority(self) -> bool:
        return self.protocol.can_authorize_action

    @property
    def update_mode(self) -> str:
        return (
            "bayesian_log_likelihood"
            if self.protocol.can_apply_bayesian_update
            else "evidence_admission_only"
        )

    def _validate_admitted(
        self,
        contributions: Sequence[PathEvidenceContribution],
    ) -> None:
        for contribution in contributions:
            if not self.protocol.is_evidence_admitted(contribution.rule_id):
                raise PathBeliefProtocolError(
                    f"path evidence rule is not admitted: {contribution.rule_id}"
                )
            has_likelihood = contribution.conditional_likelihoods is not None
            if has_likelihood != self.protocol.can_apply_bayesian_update:
                raise PathBeliefProtocolError(
                    "path evidence and likelihood artifact readiness disagree"
                )

    def initialize(
        self,
        state: PathCompetitionSetState,
        *,
        contributions: Sequence[PathEvidenceContribution] = (),
        real_completed_bar: bool,
    ) -> tuple[PathCompetitionSetState, PathBeliefUpdateRecord]:
        self._validate_admitted(contributions)
        return initialize_path_competition_set(
            self.protocol,
            state,
            initial_contributions=contributions,
            initial_real_completed_bar=real_completed_bar,
        )

    def update(
        self,
        state: PathCompetitionSetState,
        *,
        asof: pd.Timestamp,
        contributions: Sequence[PathEvidenceContribution] = (),
        terminal_events: Sequence[PathTerminalEvent] = (),
        outcome_events: Sequence[PathOutcomeEvent] = (),
        real_completed_bar: bool,
    ) -> tuple[PathCompetitionSetState, PathBeliefUpdateRecord]:
        self._validate_admitted(contributions)
        return reduce_path_competition_set(
            self.protocol,
            state,
            asof=asof,
            contributions=contributions,
            terminal_events=terminal_events,
            outcome_events=outcome_events,
            real_completed_bar=real_completed_bar,
        )


class HypothesisManager:
    """Stateful facade used by ``PlaybookBrain`` for one competition set."""

    def __init__(self, protocol: PathBeliefProtocol) -> None:
        self.protocol = protocol
        self.updater = BayesianBeliefUpdater(protocol)
        self.state: PathCompetitionSetState | None = None
        self.update_ledger: tuple[PathBeliefUpdateRecord, ...] = ()
        self._restored_update_ids: tuple[str, ...] = ()

    @property
    def action_authority(self) -> bool:
        return self.updater.action_authority

    def reset(self) -> None:
        self.state = None
        self.update_ledger = ()
        self._restored_update_ids = ()

    def fork(self) -> "HypothesisManager":
        """Create an in-memory transactional branch over immutable state."""

        manager = HypothesisManager(self.protocol)
        manager.state = self.state
        manager.update_ledger = self.update_ledger
        manager._restored_update_ids = self._restored_update_ids
        return manager

    def start(
        self,
        *,
        instrument_id: str,
        market_epoch_id: str,
        authority_structure_id: str,
        horizon_id: str,
        formed_at: pd.Timestamp,
        common_expires_at: pd.Timestamp,
        contributions: Sequence[PathEvidenceContribution] = (),
        real_completed_bar: bool,
    ) -> tuple[PathCompetitionSetState, PathBeliefUpdateRecord]:
        if self.state is not None:
            raise ValueError("hypothesis manager already has a competition set")
        state = create_path_competition_set(
            self.protocol,
            instrument_id=instrument_id,
            market_epoch_id=market_epoch_id,
            authority_structure_id=authority_structure_id,
            horizon_id=horizon_id,
            formed_at=formed_at,
            common_expires_at=common_expires_at,
        )
        return self.initialize_state(
            state,
            contributions=contributions,
            real_completed_bar=real_completed_bar,
        )

    def initialize_state(
        self,
        state: PathCompetitionSetState,
        *,
        contributions: Sequence[PathEvidenceContribution] = (),
        real_completed_bar: bool,
    ) -> tuple[PathCompetitionSetState, PathBeliefUpdateRecord]:
        """Initialize a pristine state after bound contributions are built."""

        if self.state is not None:
            raise ValueError("hypothesis manager already has a competition set")
        state, record = self.updater.initialize(
            state,
            contributions=contributions,
            real_completed_bar=real_completed_bar,
        )
        self.state = state
        self.update_ledger = (record,)
        return state, record

    def advance(
        self,
        *,
        asof: pd.Timestamp,
        contributions: Sequence[PathEvidenceContribution] = (),
        terminal_events: Sequence[PathTerminalEvent] = (),
        outcome_events: Sequence[PathOutcomeEvent] = (),
        real_completed_bar: bool,
    ) -> tuple[PathCompetitionSetState, PathBeliefUpdateRecord]:
        if self.state is None:
            raise ValueError("hypothesis manager has no competition set")
        state, record = self.updater.update(
            self.state,
            asof=asof,
            contributions=contributions,
            terminal_events=terminal_events,
            outcome_events=outcome_events,
            real_completed_bar=real_completed_bar,
        )
        self.state = state
        self.update_ledger = (*self.update_ledger, record)
        return state, record

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": HYPOTHESIS_MANAGER_SCHEMA_VERSION,
            "protocol_fingerprint": self.protocol.fingerprint,
            "model_version": self.protocol.model_version,
            "state": None if self.state is None else self.state.state_dict(),
            "update_ids": [
                *self._restored_update_ids,
                *(item.update_id for item in self.update_ledger),
            ],
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        protocol: PathBeliefProtocol,
    ) -> "HypothesisManager":
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version")
            != HYPOTHESIS_MANAGER_SCHEMA_VERSION
            or payload.get("protocol_fingerprint") != protocol.fingerprint
            or payload.get("model_version") != protocol.model_version
        ):
            raise ValueError("hypothesis manager checkpoint binding is stale")
        manager = cls(protocol)
        raw_state = payload.get("state")
        if raw_state is not None:
            manager.state = restore_path_competition_set(
                raw_state,
                protocol=protocol,
            )
        # Full update payloads are intentionally not duplicated: the exact
        # source-only evidence ledger is persisted inside the state.  IDs are
        # retained as stable audit metadata across checkpoint/resume cycles.
        update_ids = payload.get("update_ids", ())
        if (
            not isinstance(update_ids, list)
            or any(not isinstance(value, str) or not value for value in update_ids)
            or len(update_ids) != len(set(update_ids))
        ):
            raise ValueError("hypothesis manager update identity ledger is invalid")
        manager._restored_update_ids = tuple(update_ids)
        return manager

    def replay_evidence_posterior(self) -> PathCompetitionSetState:
        """Replay the persisted source-only ledger when no time decay exists."""

        current = self.state
        if current is None:
            raise ValueError("hypothesis manager has no competition set")
        if (
            current.status is not PathStatus.ACTIVE
            or current.applied_terminal_event_ids
            or current.outcome_event_id is not None
        ):
            raise ValueError(
                "evidence-posterior replay requires an active unterminated set"
            )
        if any(self.protocol.decay(path) != 0.0 for path in PATH_KINDS):
            raise ValueError("evidence-only replay requires zero time decay")
        replay = create_path_competition_set(
            self.protocol,
            instrument_id=current.instrument_id,
            market_epoch_id=current.market_epoch_id,
            authority_structure_id=current.authority_structure_id,
            horizon_id=current.horizon_id,
            formed_at=current.formed_at,
            common_expires_at=current.common_expires_at,
        )
        at_formation = tuple(
            item
            for item in current.evidence_ledger
            if item.known_at == current.formed_at
        )
        replay, _ = self.updater.initialize(
            replay,
            contributions=at_formation,
            real_completed_bar=False,
        )
        later_clocks = sorted(
            {
                item.known_at
                for item in current.evidence_ledger
                if item.known_at > current.formed_at
            }
        )
        for clock in later_clocks:
            replay, _ = self.updater.update(
                replay,
                asof=clock,
                contributions=tuple(
                    item
                    for item in current.evidence_ledger
                    if item.known_at == clock
                ),
                real_completed_bar=False,
            )
        return replay


__all__ = [
    "BayesianBeliefUpdater",
    "HYPOTHESIS_MANAGER_SCHEMA_VERSION",
    "HypothesisManager",
    "PATH_KINDS",
    "PATH_RUNTIME_RESOLUTION_SCHEMA_VERSION",
    "PATH_STATE_SCHEMA_VERSION",
    "PathBeliefProtocol",
    "PathBeliefProtocolError",
    "PathBeliefUpdateRecord",
    "PathCompetitionSetState",
    "PathEvidenceContribution",
    "PathEvidenceRule",
    "PathHypothesisState",
    "PathKind",
    "PathOutcomeEvent",
    "PathRuntimeResolutionProtocol",
    "PathStatus",
    "PathTerminalEvent",
    "create_path_competition_set",
    "initialize_path_competition_set",
    "load_path_belief_protocol",
    "reduce_path_competition_set",
    "restore_path_competition_set",
]
