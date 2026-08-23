"""Causal, append-only cohorts for the Phase-7 probability research layer.

This module deliberately contains no repository data loader and no fitting
entry point.  It only turns already-visible canonical facts into immutable
research records.  Future outcomes are separate records and every identity is
derived from the complete semantic payload so a replay cannot silently mutate
an earlier generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from smc_trader.path_belief import PATH_KINDS, PathKind
from smc_trader.structural_outcome import OutcomeBar


PROBABILITY_COHORT_SCHEMA_VERSION = 1
PATH_LABELS = tuple(path.value for path in PATH_KINDS)
RESIDUAL_PATH = PathKind.RESIDUAL_UNKNOWN.value
NO_TARGET_OUTCOME = "no_target_before_common_horizon"

_ARCHIVE_STATUSES = frozenset(
    {"realized", "censored", "superseded", "expired", "invalidated"}
)
_PATH_TERMINAL_STATUSES = frozenset(
    {"realized", "falsified", "censored", "superseded", "expired"}
)
_SPLIT_ROLES = frozenset(
    {
        "development_fit",
        "development_cross_fit",
        "calibration",
        "historical_validation",
        "rolling_oof",
        "sealed_oos",
    }
)


def aware_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    """Return one timezone-aware timestamp without changing its clock."""

    try:
        clock = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error
    if clock.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return clock


def _primitive(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, PathKind):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _primitive(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or value == -math.inf:
            raise ValueError("canonical payload contains a non-finite number")
        return "infinity" if value == math.inf else value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _primitive(asdict(value))
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_identity(prefix: str, value: Mapping[str, Any]) -> str:
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("canonical identity prefix is required")
    return f"{prefix}:{canonical_sha256(value)}"


def _identity_values(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if (
        len(normalized) != len(set(normalized))
        or any(not value for value in normalized)
    ):
        raise ValueError(f"{name} must contain unique non-empty identities")
    return tuple(sorted(normalized))


def _path(value: str | PathKind, *, name: str = "path") -> str:
    try:
        resolved = PathKind(value).value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not canonical") from error
    return resolved


def _split_role(value: str) -> str:
    role = str(value).strip()
    if role not in _SPLIT_ROLES:
        raise ValueError(f"unsupported probability split role: {role}")
    return role


def _finite_probability(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


@dataclass(frozen=True)
class PathCompetitionArchive:
    """One immutable terminal record for one path competition generation."""

    competition_set_id: str
    instrument_id: str
    market_epoch_id: str
    authority_structure_id: str
    horizon_id: str
    formed_at: pd.Timestamp
    common_expires_at: pd.Timestamp
    terminal_at: pd.Timestamp
    terminal_status: str
    terminal_cause: str
    realized_path: str | None
    outcome_known_at: pd.Timestamp | None
    source_event_ids: tuple[str, ...]
    censor_reason: str | None
    split_role: str
    fold_id: str
    schema_version: int = PROBABILITY_COHORT_SCHEMA_VERSION
    archive_id: str = field(init=False)

    def __post_init__(self) -> None:
        formed = aware_timestamp(self.formed_at, name="competition formed_at")
        horizon = aware_timestamp(
            self.common_expires_at,
            name="competition common_expires_at",
        )
        terminal = aware_timestamp(self.terminal_at, name="competition terminal_at")
        outcome_clock = (
            None
            if self.outcome_known_at is None
            else aware_timestamp(
                self.outcome_known_at,
                name="competition outcome_known_at",
            )
        )
        status = str(self.terminal_status).strip()
        cause = str(self.terminal_cause).strip()
        realized = (
            None
            if self.realized_path is None
            else _path(self.realized_path, name="realized_path")
        )
        censor = None if self.censor_reason is None else str(self.censor_reason).strip()
        sources = _identity_values(
            self.source_event_ids,
            name="archive source_event_ids",
        )
        role = _split_role(self.split_role)
        identity_fields = (
            self.competition_set_id,
            self.instrument_id,
            self.market_epoch_id,
            self.authority_structure_id,
            self.horizon_id,
            cause,
            self.fold_id,
        )
        if (
            self.schema_version != PROBABILITY_COHORT_SCHEMA_VERSION
            or any(not isinstance(value, str) or not value for value in identity_fields)
            or status not in _ARCHIVE_STATUSES
            or horizon <= formed
            or terminal < formed
            or terminal > horizon
        ):
            raise ValueError("path competition archive boundary is invalid")
        if status == "realized":
            if realized is None or outcome_clock != terminal or censor is not None:
                raise ValueError("realized archive requires one exact uncensored outcome")
        elif realized is not None or outcome_clock is not None or not censor:
            raise ValueError("non-realized archive must preserve an explicit censor")
        if realized == RESIDUAL_PATH and (
            status != "realized"
            or terminal != horizon
            or cause != "common_horizon_without_registered_winner"
            or not sources
        ):
            raise ValueError(
                "residual_unknown requires a clean sourced common-horizon outcome"
            )
        if status == "realized" and realized != RESIDUAL_PATH and not sources:
            raise ValueError("a factual realized path requires source events")
        object.__setattr__(self, "formed_at", formed)
        object.__setattr__(self, "common_expires_at", horizon)
        object.__setattr__(self, "terminal_at", terminal)
        object.__setattr__(self, "outcome_known_at", outcome_clock)
        object.__setattr__(self, "terminal_status", status)
        object.__setattr__(self, "terminal_cause", cause)
        object.__setattr__(self, "realized_path", realized)
        object.__setattr__(self, "censor_reason", censor)
        object.__setattr__(self, "source_event_ids", sources)
        object.__setattr__(self, "split_role", role)
        payload = {
            name: value for name, value in self.__dict__.items() if name != "archive_id"
        }
        object.__setattr__(self, "archive_id", canonical_identity("path-archive", payload))

    @classmethod
    def residual_at_horizon(
        cls,
        *,
        competition_set_id: str,
        instrument_id: str,
        market_epoch_id: str,
        authority_structure_id: str,
        horizon_id: str,
        formed_at: pd.Timestamp,
        common_expires_at: pd.Timestamp,
        source_event_ids: Sequence[str],
        split_role: str,
        fold_id: str,
    ) -> "PathCompetitionArchive":
        horizon = aware_timestamp(common_expires_at, name="common_expires_at")
        return cls(
            competition_set_id=competition_set_id,
            instrument_id=instrument_id,
            market_epoch_id=market_epoch_id,
            authority_structure_id=authority_structure_id,
            horizon_id=horizon_id,
            formed_at=formed_at,
            common_expires_at=horizon,
            terminal_at=horizon,
            terminal_status="realized",
            terminal_cause="common_horizon_without_registered_winner",
            realized_path=RESIDUAL_PATH,
            outcome_known_at=horizon,
            source_event_ids=tuple(source_event_ids),
            censor_reason=None,
            split_role=split_role,
            fold_id=fold_id,
        )


@dataclass(frozen=True)
class PathTerminalDisposition:
    path: str
    status: str
    cause: str
    known_at: pd.Timestamp
    source_event_ids: tuple[str, ...] = ()
    censor_reason: str | None = None

    def __post_init__(self) -> None:
        path = _path(self.path)
        status = str(self.status).strip()
        cause = str(self.cause).strip()
        clock = aware_timestamp(self.known_at, name="path terminal known_at")
        sources = _identity_values(
            self.source_event_ids,
            name="path terminal source_event_ids",
        )
        censor = None if self.censor_reason is None else str(self.censor_reason).strip()
        if status not in _PATH_TERMINAL_STATUSES or not cause:
            raise ValueError("path terminal disposition is invalid")
        if (status in {"censored", "superseded"}) != bool(censor):
            raise ValueError("path censor disposition and reason disagree")
        if status in {"realized", "falsified"} and not sources:
            raise ValueError("factual path terminal requires source events")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "cause", cause)
        object.__setattr__(self, "known_at", clock)
        object.__setattr__(self, "source_event_ids", sources)
        object.__setattr__(self, "censor_reason", censor)


@dataclass(frozen=True)
class PathRiskClock:
    asof: pd.Timestamp
    real_completed_bar: bool
    evidence_history_id: str
    correlation_cluster_id: str
    market_state_id: str
    source_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        clock = aware_timestamp(self.asof, name="risk clock asof")
        if type(self.real_completed_bar) is not bool:
            raise TypeError("real_completed_bar must be boolean")
        identifiers = (
            self.evidence_history_id,
            self.correlation_cluster_id,
            self.market_state_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("risk clock history/state identity is incomplete")
        object.__setattr__(self, "asof", clock)
        object.__setattr__(
            self,
            "source_event_ids",
            _identity_values(self.source_event_ids, name="risk clock source_event_ids"),
        )


@dataclass(frozen=True)
class PathRiskInterval:
    competition_set_id: str
    path: str
    interval_start: pd.Timestamp
    interval_end: pd.Timestamp
    age_real_completed_bars: int
    real_completed_bar: bool
    at_risk: bool
    terminal_status: str | None
    terminal_cause: str | None
    terminal_known_at: pd.Timestamp | None
    common_expires_at: pd.Timestamp
    evidence_history_id: str
    correlation_cluster_id: str
    market_state_id: str
    source_event_ids: tuple[str, ...]
    censor_reason: str | None
    split_role: str
    fold_id: str
    schema_version: int = PROBABILITY_COHORT_SCHEMA_VERSION
    interval_id: str = field(init=False)

    def __post_init__(self) -> None:
        path = _path(self.path)
        start = aware_timestamp(self.interval_start, name="risk interval_start")
        end = aware_timestamp(self.interval_end, name="risk interval_end")
        horizon = aware_timestamp(
            self.common_expires_at,
            name="risk common_expires_at",
        )
        terminal_clock = (
            None
            if self.terminal_known_at is None
            else aware_timestamp(self.terminal_known_at, name="risk terminal_known_at")
        )
        status = (
            None if self.terminal_status is None else str(self.terminal_status).strip()
        )
        cause = None if self.terminal_cause is None else str(self.terminal_cause).strip()
        censor = None if self.censor_reason is None else str(self.censor_reason).strip()
        if (
            self.schema_version != PROBABILITY_COHORT_SCHEMA_VERSION
            or not self.competition_set_id
            or not self.fold_id
            or end <= start
            or end > horizon
            or type(self.age_real_completed_bars) is not int
            or self.age_real_completed_bars < 0
            or type(self.real_completed_bar) is not bool
            or type(self.at_risk) is not bool
            or not self.at_risk
            or any(
                not isinstance(value, str) or not value
                for value in (
                    self.evidence_history_id,
                    self.correlation_cluster_id,
                    self.market_state_id,
                )
            )
        ):
            raise ValueError("path risk interval is invalid")
        terminal_values = (status, cause, terminal_clock)
        if any(value is None for value in terminal_values) != all(
            value is None for value in terminal_values
        ):
            raise ValueError("risk terminal fields must be all present or all absent")
        if status is not None:
            if status not in _PATH_TERMINAL_STATUSES or terminal_clock != end:
                raise ValueError("risk terminal must settle at the interval end")
            if (status in {"censored", "superseded"}) != bool(censor):
                raise ValueError("risk terminal censor reason disagrees")
        elif censor is not None:
            raise ValueError("non-terminal risk interval cannot be censored")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "interval_start", start)
        object.__setattr__(self, "interval_end", end)
        object.__setattr__(self, "common_expires_at", horizon)
        object.__setattr__(self, "terminal_status", status)
        object.__setattr__(self, "terminal_cause", cause)
        object.__setattr__(self, "terminal_known_at", terminal_clock)
        object.__setattr__(self, "censor_reason", censor)
        object.__setattr__(
            self,
            "source_event_ids",
            _identity_values(self.source_event_ids, name="risk source_event_ids"),
        )
        object.__setattr__(self, "split_role", _split_role(self.split_role))
        payload = {
            name: value for name, value in self.__dict__.items() if name != "interval_id"
        }
        object.__setattr__(self, "interval_id", canonical_identity("path-risk", payload))


def _default_terminal_dispositions(
    archive: PathCompetitionArchive,
) -> tuple[PathTerminalDisposition, ...]:
    if archive.terminal_status == "realized":
        assert archive.realized_path is not None
        return tuple(
            PathTerminalDisposition(
                path=path,
                status="realized" if path == archive.realized_path else "falsified",
                cause=(
                    archive.terminal_cause
                    if path == archive.realized_path
                    else f"competing_path_realized:{archive.realized_path}"
                ),
                known_at=archive.terminal_at,
                source_event_ids=archive.source_event_ids,
            )
            for path in PATH_LABELS
        )
    status = (
        "superseded"
        if archive.terminal_status == "superseded"
        else "expired"
        if archive.terminal_status == "expired"
        else "censored"
    )
    return tuple(
        PathTerminalDisposition(
            path=path,
            status=status,
            cause=archive.terminal_cause,
            known_at=archive.terminal_at,
            source_event_ids=archive.source_event_ids,
            censor_reason=(
                archive.censor_reason if status in {"censored", "superseded"} else None
            ),
        )
        for path in PATH_LABELS
    )


def build_path_risk_intervals(
    archive: PathCompetitionArchive,
    clocks: Sequence[PathRiskClock],
    *,
    terminal_dispositions: Sequence[PathTerminalDisposition] | None = None,
) -> tuple[PathRiskInterval, ...]:
    """Build one deterministic at-risk ledger for all six hypotheses.

    A synthetic clock advances event time but leaves ``age_real_completed_bars``
    unchanged.  Every terminal must have an exact clock; the builder never
    invents exposure or silently snaps a terminal onto a later bar.
    """

    if not isinstance(archive, PathCompetitionArchive):
        raise TypeError("archive must be PathCompetitionArchive")
    ordered = tuple(clocks)
    if any(not isinstance(clock, PathRiskClock) for clock in ordered):
        raise TypeError("risk clocks must be PathRiskClock")
    keys = tuple((clock.asof, tuple(clock.source_event_ids)) for clock in ordered)
    if keys != tuple(sorted(keys)) or len({clock.asof for clock in ordered}) != len(
        ordered
    ):
        raise ValueError("risk clocks must be unique and ordered")
    if (
        not ordered
        or ordered[0].asof <= archive.formed_at
        or ordered[-1].asof < archive.terminal_at
        or any(clock.asof > archive.terminal_at for clock in ordered)
    ):
        raise ValueError("risk clocks do not exactly cover the archived generation")
    dispositions = tuple(
        _default_terminal_dispositions(archive)
        if terminal_dispositions is None
        else terminal_dispositions
    )
    if (
        any(not isinstance(item, PathTerminalDisposition) for item in dispositions)
        or len(dispositions) != len(PATH_LABELS)
        or {item.path for item in dispositions} != set(PATH_LABELS)
    ):
        raise ValueError("one terminal disposition per canonical path is required")
    disposition_by_path = {item.path: item for item in dispositions}
    clock_set = {clock.asof for clock in ordered}
    if any(
        item.known_at not in clock_set
        or item.known_at > archive.terminal_at
        or item.known_at <= archive.formed_at
        for item in dispositions
    ):
        raise ValueError("every path terminal requires an exact covered clock")

    intervals: list[PathRiskInterval] = []
    for path in PATH_LABELS:
        disposition = disposition_by_path[path]
        prior_clock = archive.formed_at
        age = 0
        terminal_seen = False
        for clock in ordered:
            if clock.asof > disposition.known_at:
                break
            if clock.real_completed_bar:
                age += 1
            terminal = clock.asof == disposition.known_at
            source_ids = tuple(
                sorted(
                    set(clock.source_event_ids).union(
                        disposition.source_event_ids if terminal else ()
                    )
                )
            )
            intervals.append(
                PathRiskInterval(
                    competition_set_id=archive.competition_set_id,
                    path=path,
                    interval_start=prior_clock,
                    interval_end=clock.asof,
                    age_real_completed_bars=age,
                    real_completed_bar=clock.real_completed_bar,
                    at_risk=True,
                    terminal_status=disposition.status if terminal else None,
                    terminal_cause=disposition.cause if terminal else None,
                    terminal_known_at=clock.asof if terminal else None,
                    common_expires_at=archive.common_expires_at,
                    evidence_history_id=clock.evidence_history_id,
                    correlation_cluster_id=clock.correlation_cluster_id,
                    market_state_id=clock.market_state_id,
                    source_event_ids=source_ids,
                    censor_reason=disposition.censor_reason if terminal else None,
                    split_role=archive.split_role,
                    fold_id=archive.fold_id,
                )
            )
            prior_clock = clock.asof
            if terminal:
                terminal_seen = True
                break
        if not terminal_seen:
            raise ValueError(f"path terminal was not materialized: {path}")
    return tuple(sorted(intervals, key=lambda item: (item.interval_end, item.path)))


@dataclass(frozen=True)
class EvidenceHistoryTransition:
    competition_set_id: str
    asof: pd.Timestamp
    known_at: pd.Timestamp
    common_expires_at: pd.Timestamp
    previous_history_id: str
    evidence_history_id: str
    evidence_rule_ids: tuple[str, ...]
    evidence_observed: bool
    correlation_cluster_id: str
    market_state_id: str
    realized_path: str | None
    outcome_known_at: pd.Timestamp | None
    censor_reason: str | None
    split_role: str
    fold_id: str
    source_event_ids: tuple[str, ...]
    schema_version: int = PROBABILITY_COHORT_SCHEMA_VERSION
    transition_id: str = field(init=False)

    def __post_init__(self) -> None:
        asof = aware_timestamp(self.asof, name="history asof")
        known = aware_timestamp(self.known_at, name="history known_at")
        horizon = aware_timestamp(self.common_expires_at, name="history horizon")
        outcome = (
            None
            if self.outcome_known_at is None
            else aware_timestamp(self.outcome_known_at, name="history outcome_known_at")
        )
        realized = (
            None if self.realized_path is None else _path(self.realized_path)
        )
        censor = None if self.censor_reason is None else str(self.censor_reason).strip()
        if (
            self.schema_version != PROBABILITY_COHORT_SCHEMA_VERSION
            or not self.competition_set_id
            or not self.previous_history_id
            or not self.evidence_history_id
            or not self.correlation_cluster_id
            or not self.market_state_id
            or not self.fold_id
            or type(self.evidence_observed) is not bool
            or known > asof
            or asof >= horizon
        ):
            raise ValueError("evidence history transition is invalid")
        if realized is None:
            if outcome is not None or not censor:
                raise ValueError("unresolved history row requires explicit censor")
        elif outcome is None or outcome <= asof or censor is not None:
            raise ValueError("resolved history row has invalid future label clocks")
        rules = _identity_values(
            self.evidence_rule_ids,
            name="history evidence_rule_ids",
        )
        sources = _identity_values(
            self.source_event_ids,
            name="history source_event_ids",
        )
        if self.evidence_observed and (not rules or not sources):
            raise ValueError("observed evidence history requires exact ancestry")
        object.__setattr__(self, "asof", asof)
        object.__setattr__(self, "known_at", known)
        object.__setattr__(self, "common_expires_at", horizon)
        object.__setattr__(self, "realized_path", realized)
        object.__setattr__(self, "outcome_known_at", outcome)
        object.__setattr__(self, "censor_reason", censor)
        object.__setattr__(self, "evidence_rule_ids", rules)
        object.__setattr__(self, "source_event_ids", sources)
        object.__setattr__(self, "split_role", _split_role(self.split_role))
        payload = {
            name: value
            for name, value in self.__dict__.items()
            if name != "transition_id"
        }
        object.__setattr__(
            self,
            "transition_id",
            canonical_identity("evidence-history", payload),
        )


@dataclass(frozen=True)
class DOLCandidateSnapshot:
    competition_set_id: str
    candidate_set_id: str
    candidate_id: str
    path: str
    symbol: str
    instrument_id: int
    prediction_known_at: pd.Timestamp
    common_expires_at: pd.Timestamp
    candidate_eligible: bool
    candidate_feature_schema_id: str
    raw_candidate_probability: float
    target_price: float
    source_event_ids: tuple[str, ...]
    split_role: str
    fold_id: str
    schema_version: int = PROBABILITY_COHORT_SCHEMA_VERSION
    snapshot_id: str = field(init=False)

    def __post_init__(self) -> None:
        path = _path(self.path)
        prediction = aware_timestamp(
            self.prediction_known_at,
            name="DOL prediction_known_at",
        )
        horizon = aware_timestamp(
            self.common_expires_at,
            name="DOL common_expires_at",
        )
        try:
            target = float(self.target_price)
        except (TypeError, ValueError) as error:
            raise ValueError("DOL target_price must be numeric") from error
        if (
            self.schema_version != PROBABILITY_COHORT_SCHEMA_VERSION
            or any(
                not isinstance(value, str) or not value
                for value in (
                    self.competition_set_id,
                    self.candidate_set_id,
                    self.candidate_id,
                    self.symbol,
                    self.candidate_feature_schema_id,
                    self.fold_id,
                )
            )
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or horizon <= prediction
            or type(self.candidate_eligible) is not bool
            or not math.isfinite(target)
            or target <= 0.0
        ):
            raise ValueError("DOL candidate snapshot is invalid")
        probability = _finite_probability(
            self.raw_candidate_probability,
            name="raw_candidate_probability",
        )
        sources = _identity_values(
            self.source_event_ids,
            name="DOL candidate source_event_ids",
        )
        if self.candidate_eligible and not sources:
            raise ValueError("eligible DOL candidate requires canonical source identity")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "prediction_known_at", prediction)
        object.__setattr__(self, "common_expires_at", horizon)
        object.__setattr__(self, "raw_candidate_probability", probability)
        object.__setattr__(self, "target_price", target)
        object.__setattr__(self, "source_event_ids", sources)
        object.__setattr__(self, "split_role", _split_role(self.split_role))
        payload = {
            name: value for name, value in self.__dict__.items() if name != "snapshot_id"
        }
        object.__setattr__(self, "snapshot_id", canonical_identity("dol-snapshot", payload))


@dataclass(frozen=True)
class DOLCandidateHit:
    candidate_set_id: str
    candidate_id: str
    known_at: pd.Timestamp
    source_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_set_id or not self.candidate_id:
            raise ValueError("DOL candidate hit identity is incomplete")
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(self.known_at, name="DOL hit known_at"),
        )
        sources = _identity_values(
            self.source_event_ids,
            name="DOL hit source_event_ids",
        )
        if not sources:
            raise ValueError("DOL candidate hit requires canonical ancestry")
        object.__setattr__(self, "source_event_ids", sources)


@dataclass(frozen=True)
class DOLCandidateOutcomeRow:
    snapshot: DOLCandidateSnapshot
    first_hit_candidate_id: str | None
    no_target_before_horizon: bool
    outcome_known_at: pd.Timestamp
    censor_reason: str | None
    ambiguous_candidate_ids: tuple[str, ...]
    outcome_source_event_ids: tuple[str, ...]
    schema_version: int = PROBABILITY_COHORT_SCHEMA_VERSION
    label_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, DOLCandidateSnapshot):
            raise TypeError("DOL outcome row requires a candidate snapshot")
        outcome = aware_timestamp(self.outcome_known_at, name="DOL outcome_known_at")
        censor = None if self.censor_reason is None else str(self.censor_reason).strip()
        ambiguous = _identity_values(
            self.ambiguous_candidate_ids,
            name="ambiguous_candidate_ids",
        )
        sources = _identity_values(
            self.outcome_source_event_ids,
            name="DOL outcome source_event_ids",
        )
        first_hit = (
            None
            if self.first_hit_candidate_id is None
            else str(self.first_hit_candidate_id).strip()
        )
        if (
            self.schema_version != PROBABILITY_COHORT_SCHEMA_VERSION
            or type(self.no_target_before_horizon) is not bool
            or outcome <= self.snapshot.prediction_known_at
            or outcome > self.snapshot.common_expires_at
        ):
            raise ValueError("DOL candidate outcome row is invalid")
        modes = sum(
            (
                first_hit is not None,
                self.no_target_before_horizon,
                bool(censor),
            )
        )
        if modes != 1:
            raise ValueError("DOL outcome must be first-hit, no-target, or censored")
        if first_hit is not None and (ambiguous or not sources):
            raise ValueError("first-hit DOL outcome requires exact unambiguous ancestry")
        if self.no_target_before_horizon and (
            outcome != self.snapshot.common_expires_at or ambiguous or not sources
        ):
            raise ValueError(
                "no-target requires a sourced, clean, fully observed horizon"
            )
        if censor == "ambiguous_same_bar":
            if len(ambiguous) < 2 or not sources:
                raise ValueError("same-bar ambiguity requires all candidates and sources")
        elif ambiguous:
            raise ValueError("ambiguous candidate identities require ambiguity censor")
        object.__setattr__(self, "first_hit_candidate_id", first_hit)
        object.__setattr__(self, "outcome_known_at", outcome)
        object.__setattr__(self, "censor_reason", censor)
        object.__setattr__(self, "ambiguous_candidate_ids", ambiguous)
        object.__setattr__(self, "outcome_source_event_ids", sources)
        payload = {
            name: value for name, value in self.__dict__.items() if name != "label_id"
        }
        object.__setattr__(self, "label_id", canonical_identity("dol-label", payload))

    def to_cohort_dict(self) -> dict[str, Any]:
        snapshot = self.snapshot
        return {
            "competition_set_id": snapshot.competition_set_id,
            "candidate_set_id": snapshot.candidate_set_id,
            "candidate_id": snapshot.candidate_id,
            "path": snapshot.path,
            "prediction_known_at": snapshot.prediction_known_at,
            "common_expires_at": snapshot.common_expires_at,
            "candidate_eligible": snapshot.candidate_eligible,
            "candidate_feature_schema_id": snapshot.candidate_feature_schema_id,
            "raw_candidate_probability": snapshot.raw_candidate_probability,
            "first_hit_candidate_id": self.first_hit_candidate_id,
            "no_target_before_horizon": self.no_target_before_horizon,
            "outcome_known_at": self.outcome_known_at,
            "censor_reason": self.censor_reason,
            "split_role": snapshot.split_role,
            "fold_id": snapshot.fold_id,
            "label_id": self.label_id,
        }


def _validate_candidate_group(
    candidates: Sequence[DOLCandidateSnapshot],
) -> tuple[DOLCandidateSnapshot, ...]:
    group = tuple(candidates)
    if not group or any(not isinstance(item, DOLCandidateSnapshot) for item in group):
        raise ValueError("DOL candidate set must be non-empty and typed")
    reference = group[0]
    common = (
        reference.competition_set_id,
        reference.candidate_set_id,
        reference.path,
        reference.symbol,
        reference.instrument_id,
        reference.prediction_known_at,
        reference.common_expires_at,
        reference.candidate_feature_schema_id,
        reference.split_role,
        reference.fold_id,
    )
    if any(
        (
            item.competition_set_id,
            item.candidate_set_id,
            item.path,
            item.symbol,
            item.instrument_id,
            item.prediction_known_at,
            item.common_expires_at,
            item.candidate_feature_schema_id,
            item.split_role,
            item.fold_id,
        )
        != common
        for item in group
    ):
        raise ValueError("DOL candidate set mixes prediction cohorts")
    candidate_ids = tuple(item.candidate_id for item in group)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("DOL candidate set repeats a candidate")
    return tuple(sorted(group, key=lambda item: item.candidate_id))


def derive_dol_candidate_hits(
    candidates: Sequence[DOLCandidateSnapshot],
    bars: Sequence[OutcomeBar],
) -> tuple[DOLCandidateHit, ...]:
    """Project future normalized bars into candidate-hit facts.

    This helper is intentionally geometry-only.  A runtime integration may
    instead pass exact canonical level-touch events directly to
    :func:`label_dol_candidate_set`.
    """

    group = _validate_candidate_group(candidates)
    reference = group[0]
    ordered = tuple(bars)
    if any(not isinstance(bar, OutcomeBar) for bar in ordered):
        raise TypeError("DOL hit projection requires OutcomeBar inputs")
    keys = tuple((bar.known_at, bar.bar_event_id) for bar in ordered)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError("DOL outcome bars are duplicated or out of order")
    hits: list[DOLCandidateHit] = []
    for bar in ordered:
        if bar.known_at <= reference.prediction_known_at:
            continue
        if bar.known_at > reference.common_expires_at:
            break
        if bar.symbol != reference.symbol or bar.instrument_id != reference.instrument_id:
            break
        for candidate in group:
            if (
                candidate.candidate_eligible
                and float(bar.low) <= candidate.target_price <= float(bar.high)
            ):
                hits.append(
                    DOLCandidateHit(
                        candidate_set_id=reference.candidate_set_id,
                        candidate_id=candidate.candidate_id,
                        known_at=bar.known_at,
                        source_event_ids=(bar.bar_event_id,),
                    )
                )
    return tuple(hits)


def label_dol_candidate_set(
    candidates: Sequence[DOLCandidateSnapshot],
    hits: Sequence[DOLCandidateHit],
    *,
    observed_through: pd.Timestamp,
    observation_source_event_ids: Sequence[str],
    censor_reason: str | None = None,
) -> tuple[DOLCandidateOutcomeRow, ...]:
    """Label a frozen DOL candidate set without future-state leakage."""

    group = _validate_candidate_group(candidates)
    reference = group[0]
    observation_clock = aware_timestamp(observed_through, name="DOL observed_through")
    censor = None if censor_reason is None else str(censor_reason).strip()
    observation_sources = _identity_values(
        observation_source_event_ids,
        name="DOL observation_source_event_ids",
    )
    if observation_clock <= reference.prediction_known_at:
        raise ValueError("DOL outcome observation must follow prediction")
    if censor and observation_clock > reference.common_expires_at:
        raise ValueError("DOL censor cannot follow a fully observed horizon")
    if not censor and observation_clock < reference.common_expires_at:
        raise ValueError("an incomplete DOL horizon requires an explicit censor")
    eligible_ids = {
        item.candidate_id for item in group if item.candidate_eligible
    }
    all_ids = {item.candidate_id for item in group}
    normalized_hits: list[DOLCandidateHit] = []
    for hit in hits:
        if not isinstance(hit, DOLCandidateHit):
            raise TypeError("DOL hits must be DOLCandidateHit")
        if hit.candidate_set_id != reference.candidate_set_id:
            raise ValueError("DOL hit belongs to another candidate set")
        if hit.candidate_id not in all_ids:
            raise ValueError("DOL hit references an unknown candidate")
        if hit.known_at <= reference.prediction_known_at:
            raise ValueError("DOL hit leaks a pre-prediction fact")
        if (
            hit.candidate_id in eligible_ids
            and hit.known_at <= reference.common_expires_at
            and hit.known_at <= observation_clock
        ):
            normalized_hits.append(hit)
    normalized_hits.sort(key=lambda item: (item.known_at, item.candidate_id))

    first_hit: str | None = None
    no_target = False
    outcome_clock: pd.Timestamp
    outcome_censor = censor
    ambiguous_ids: tuple[str, ...] = ()
    outcome_sources: tuple[str, ...] = ()
    if normalized_hits:
        outcome_clock = normalized_hits[0].known_at
        earliest = tuple(
            item for item in normalized_hits if item.known_at == outcome_clock
        )
        earliest_ids = tuple(sorted({item.candidate_id for item in earliest}))
        outcome_sources = tuple(
            sorted({source for item in earliest for source in item.source_event_ids})
        )
        if len(earliest_ids) == 1:
            first_hit = earliest_ids[0]
            outcome_censor = None
        else:
            ambiguous_ids = earliest_ids
            outcome_censor = "ambiguous_same_bar"
    elif censor:
        outcome_clock = observation_clock
        outcome_sources = observation_sources
    else:
        outcome_clock = reference.common_expires_at
        no_target = True
        outcome_sources = observation_sources

    if (censor or no_target) and not outcome_sources:
        raise ValueError("DOL observation terminal requires exact source ancestry")

    return tuple(
        DOLCandidateOutcomeRow(
            snapshot=candidate,
            first_hit_candidate_id=first_hit,
            no_target_before_horizon=no_target,
            outcome_known_at=outcome_clock,
            censor_reason=outcome_censor,
            ambiguous_candidate_ids=ambiguous_ids,
            outcome_source_event_ids=outcome_sources,
        )
        for candidate in group
    )


__all__ = [
    "DOLCandidateHit",
    "DOLCandidateOutcomeRow",
    "DOLCandidateSnapshot",
    "EvidenceHistoryTransition",
    "NO_TARGET_OUTCOME",
    "PATH_LABELS",
    "PROBABILITY_COHORT_SCHEMA_VERSION",
    "PathCompetitionArchive",
    "PathRiskClock",
    "PathRiskInterval",
    "PathTerminalDisposition",
    "aware_timestamp",
    "build_path_risk_intervals",
    "canonical_identity",
    "canonical_sha256",
    "derive_dol_candidate_hits",
    "label_dol_candidate_set",
]
