"""Deterministic Phase-7 probability fitting primitives.

The functions here never discover files, choose data windows, or write model
artifacts.  A caller must supply an already materialized development cohort
and exact source/manifest hashes.  This keeps fitting mechanically separate
from cohort authority and from the sealed reveal gate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from smc_trader.probability_cohorts import (
    EvidenceHistoryTransition,
    NO_TARGET_OUTCOME,
    PATH_LABELS,
    PathRiskInterval,
    aware_timestamp,
    canonical_identity,
)


PROBABILITY_MODEL_SCHEMA_VERSION = 1
DEFAULT_HISTORY_TRANSITIONS: Mapping[str, tuple[str, ...]] = {
    "none": (
        "acceptance_only",
        "displacement_only",
        "same_clock_joint",
    ),
    "acceptance_only": ("acceptance_then_displacement",),
    "displacement_only": ("displacement_then_acceptance",),
}
DEFAULT_HAZARD_BIN_ENDS: tuple[int | None, ...] = (
    1,
    5,
    15,
    30,
    60,
    120,
    240,
    None,
)
DEFAULT_HAZARD_CAUSES = (
    "realized",
    "falsified",
    "scope_superseded",
    "expired",
)
DEFAULT_HALF_LIFE_GRID = (5.0, 15.0, 30.0, 60.0, 120.0, 240.0, math.inf)
_FIT_ROLES = frozenset({"development_fit", "development_cross_fit", "calibration"})


def _sha256(value: str, *, name: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _fit_role(value: str) -> str:
    role = str(value).strip()
    if role not in _FIT_ROLES:
        raise ValueError(f"cohort role is not authorized for fitting: {role}")
    return role


def _path_mapping(
    values: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    name: str,
    allow_positive_infinity: bool = False,
) -> tuple[tuple[str, float], ...]:
    source = dict(values)
    if set(source) != set(PATH_LABELS):
        raise ValueError(f"{name} must contain every canonical path exactly")
    normalized = tuple((path, float(source[path])) for path in PATH_LABELS)
    if any(
        not math.isfinite(value)
        and not (allow_positive_infinity and value == math.inf)
        for _, value in normalized
    ):
        raise ValueError(f"{name} contains non-finite values")
    return normalized


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def _unit_weights(identities: Sequence[str]) -> np.ndarray:
    counts = Counter(str(value) for value in identities)
    raw = np.asarray([1.0 / counts[str(value)] for value in identities], dtype=float)
    return raw / np.sum(raw)


@dataclass(frozen=True)
class HistoryLikelihoodCell:
    previous_history_id: str
    evidence_history_id: str
    path: str
    observed_count: int
    denominator_count: int
    conditional_probability: float

    def __post_init__(self) -> None:
        if (
            not self.previous_history_id
            or not self.evidence_history_id
            or self.path not in PATH_LABELS
            or type(self.observed_count) is not int
            or type(self.denominator_count) is not int
            or self.observed_count < 0
            or self.denominator_count < self.observed_count
            or not math.isfinite(float(self.conditional_probability))
            or not 0.0 < float(self.conditional_probability) <= 1.0
        ):
            raise ValueError("history likelihood cell is invalid")


@dataclass(frozen=True)
class HistoryConditionalLikelihoodArtifact:
    model_version: str
    source_dataset_sha256: str
    manifest_sha256: str
    smoothing_alpha: float
    allowed_transitions: tuple[tuple[str, tuple[str, ...]], ...]
    cells: tuple[HistoryLikelihoodCell, ...]
    fit_row_count: int
    schema_version: int = PROBABILITY_MODEL_SCHEMA_VERSION
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_dataset_sha256, name="source_dataset_sha256")
        _sha256(self.manifest_sha256, name="manifest_sha256")
        if (
            self.schema_version != PROBABILITY_MODEL_SCHEMA_VERSION
            or not self.model_version
            or not math.isfinite(float(self.smoothing_alpha))
            or self.smoothing_alpha <= 0.0
            or type(self.fit_row_count) is not int
            or self.fit_row_count < 1
            or not self.cells
        ):
            raise ValueError("history likelihood artifact is invalid")
        keys = tuple(
            (cell.previous_history_id, cell.evidence_history_id, cell.path)
            for cell in self.cells
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("history likelihood cells are not canonical")
        transition_map = dict(self.allowed_transitions)
        for previous, next_states in self.allowed_transitions:
            if (
                not previous
                or not next_states
                or len(next_states) != len(set(next_states))
                or tuple(next_states) != tuple(sorted(next_states))
            ):
                raise ValueError("allowed history transitions are invalid")
            for path in PATH_LABELS:
                probability = math.fsum(
                    cell.conditional_probability
                    for cell in self.cells
                    if cell.previous_history_id == previous and cell.path == path
                )
                if not math.isclose(probability, 1.0, abs_tol=1e-12):
                    raise ValueError("history likelihood probabilities do not sum to one")
        if not transition_map:
            raise ValueError("history likelihood transition map is empty")
        payload = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "source_dataset_sha256": self.source_dataset_sha256,
            "manifest_sha256": self.manifest_sha256,
            "smoothing_alpha": self.smoothing_alpha,
            "allowed_transitions": self.allowed_transitions,
            "cells": self.cells,
            "fit_row_count": self.fit_row_count,
        }
        object.__setattr__(
            self,
            "artifact_id",
            canonical_identity("history-likelihood-artifact", payload),
        )

    def likelihood(self, previous_history_id: str, evidence_history_id: str, path: str) -> float:
        for cell in self.cells:
            if (
                cell.previous_history_id == previous_history_id
                and cell.evidence_history_id == evidence_history_id
                and cell.path == path
            ):
                return float(cell.conditional_probability)
        raise KeyError((previous_history_id, evidence_history_id, path))

    def log_increment(self, previous_history_id: str, evidence_history_id: str, path: str) -> float:
        return math.log(self.likelihood(previous_history_id, evidence_history_id, path))


def fit_history_conditional_likelihood(
    rows: Sequence[EvidenceHistoryTransition],
    *,
    source_dataset_sha256: str,
    manifest_sha256: str,
    model_version: str = "phase7_history_likelihood_v1",
    smoothing_alpha: float = 0.5,
    allowed_transitions: Mapping[str, Sequence[str]] = DEFAULT_HISTORY_TRANSITIONS,
) -> HistoryConditionalLikelihoodArtifact:
    """Fit Jeffreys-smoothed P(new history | old history, realized path)."""

    if not math.isfinite(float(smoothing_alpha)) or smoothing_alpha <= 0.0:
        raise ValueError("history smoothing alpha must be positive")
    normalized_transitions = tuple(
        sorted(
            (
                str(previous),
                tuple(sorted(str(value) for value in next_states)),
            )
            for previous, next_states in allowed_transitions.items()
        )
    )
    transition_map = dict(normalized_transitions)
    typed = tuple(rows)
    if not typed or any(not isinstance(row, EvidenceHistoryTransition) for row in typed):
        raise ValueError("history likelihood fit requires typed cohort rows")
    if len({row.transition_id for row in typed}) != len(typed):
        raise ValueError("history likelihood cohort repeats a transition identity")
    counts: Counter[tuple[str, str, str]] = Counter()
    denominators: Counter[tuple[str, str]] = Counter()
    fit_rows = 0
    for row in typed:
        _fit_role(row.split_role)
        if row.realized_path is None or not row.evidence_observed:
            continue
        allowed = transition_map.get(row.previous_history_id)
        if allowed is None or row.evidence_history_id not in allowed:
            raise ValueError("history transition is outside the frozen state machine")
        counts[(row.previous_history_id, row.evidence_history_id, row.realized_path)] += 1
        denominators[(row.previous_history_id, row.realized_path)] += 1
        fit_rows += 1
    if fit_rows < 1:
        raise ValueError("history likelihood cohort has no resolved observed transitions")
    cells: list[HistoryLikelihoodCell] = []
    for previous, next_states in normalized_transitions:
        width = len(next_states)
        for path in PATH_LABELS:
            denominator = denominators[(previous, path)]
            smoothed_denominator = denominator + smoothing_alpha * width
            for next_state in next_states:
                observed = counts[(previous, next_state, path)]
                cells.append(
                    HistoryLikelihoodCell(
                        previous_history_id=previous,
                        evidence_history_id=next_state,
                        path=path,
                        observed_count=observed,
                        denominator_count=denominator,
                        conditional_probability=(
                            observed + smoothing_alpha
                        ) / smoothed_denominator,
                    )
                )
    return HistoryConditionalLikelihoodArtifact(
        model_version=model_version,
        source_dataset_sha256=_sha256(
            source_dataset_sha256,
            name="source_dataset_sha256",
        ),
        manifest_sha256=_sha256(manifest_sha256, name="manifest_sha256"),
        smoothing_alpha=float(smoothing_alpha),
        allowed_transitions=normalized_transitions,
        cells=tuple(sorted(cells, key=lambda cell: (
            cell.previous_history_id,
            cell.evidence_history_id,
            cell.path,
        ))),
        fit_row_count=fit_rows,
    )


@dataclass(frozen=True)
class CompetingRiskCell:
    path: str
    bin_index: int
    age_start_bar: int
    age_end_bar: int | None
    at_risk_intervals: int
    terminal_counts: tuple[tuple[str, int], ...]
    cause_hazards: tuple[tuple[str, float], ...]
    no_event_probability: float

    def __post_init__(self) -> None:
        if (
            self.path not in PATH_LABELS
            or type(self.bin_index) is not int
            or self.bin_index < 0
            or type(self.age_start_bar) is not int
            or self.age_start_bar < 1
            or (
                self.age_end_bar is not None
                and (
                    type(self.age_end_bar) is not int
                    or self.age_end_bar < self.age_start_bar
                )
            )
            or type(self.at_risk_intervals) is not int
            or self.at_risk_intervals < 1
        ):
            raise ValueError("competing-risk cell is invalid")
        causes = tuple(cause for cause, _ in self.cause_hazards)
        if causes != tuple(cause for cause, _ in self.terminal_counts):
            raise ValueError("competing-risk causes disagree")
        probability = self.no_event_probability + math.fsum(
            value for _, value in self.cause_hazards
        )
        if not math.isclose(probability, 1.0, abs_tol=1e-12):
            raise ValueError("competing-risk probabilities do not sum to one")


@dataclass(frozen=True)
class CompetingRiskLifeTableArtifact:
    model_version: str
    source_dataset_sha256: str
    manifest_sha256: str
    smoothing_alpha: float
    age_bin_ends: tuple[int | None, ...]
    causes: tuple[str, ...]
    cells: tuple[CompetingRiskCell, ...]
    real_at_risk_interval_count: int
    ignored_synthetic_interval_count: int
    schema_version: int = PROBABILITY_MODEL_SCHEMA_VERSION
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_dataset_sha256, name="source_dataset_sha256")
        _sha256(self.manifest_sha256, name="manifest_sha256")
        _validate_hazard_bins(self.age_bin_ends)
        if (
            self.schema_version != PROBABILITY_MODEL_SCHEMA_VERSION
            or not self.model_version
            or not self.causes
            or tuple(self.causes) != tuple(sorted(set(self.causes)))
            or self.real_at_risk_interval_count < 1
            or self.ignored_synthetic_interval_count < 0
            or not self.cells
        ):
            raise ValueError("competing-risk artifact is invalid")
        keys = tuple((cell.path, cell.bin_index) for cell in self.cells)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("competing-risk cells are not canonical")
        payload = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "source_dataset_sha256": self.source_dataset_sha256,
            "manifest_sha256": self.manifest_sha256,
            "smoothing_alpha": self.smoothing_alpha,
            "age_bin_ends": self.age_bin_ends,
            "causes": self.causes,
            "cells": self.cells,
            "real_at_risk_interval_count": self.real_at_risk_interval_count,
            "ignored_synthetic_interval_count": self.ignored_synthetic_interval_count,
        }
        object.__setattr__(
            self,
            "artifact_id",
            canonical_identity("competing-risk-artifact", payload),
        )

    def cell(self, path: str, age_real_completed_bars: int) -> CompetingRiskCell:
        index = _hazard_bin_index(age_real_completed_bars, self.age_bin_ends)
        for cell in self.cells:
            if cell.path == path and cell.bin_index == index:
                return cell
        raise KeyError((path, index))


def _validate_hazard_bins(values: Sequence[int | None]) -> tuple[int | None, ...]:
    bins = tuple(values)
    if (
        len(bins) < 2
        or bins[-1] is not None
        or any(value is None for value in bins[:-1])
        or any(type(value) is not int or value < 1 for value in bins[:-1])
        or tuple(bins[:-1]) != tuple(sorted(set(bins[:-1])))
    ):
        raise ValueError("hazard bins must be increasing and end at session_end")
    return bins


def _hazard_bin_index(age: int, bins: Sequence[int | None]) -> int:
    if type(age) is not int or age < 1:
        raise ValueError("hazard age must be a positive real-bar count")
    for index, end in enumerate(bins):
        if end is None or age <= end:
            return index
    raise AssertionError("open-ended hazard bin was not found")


def fit_competing_risk_life_table(
    intervals: Sequence[PathRiskInterval],
    *,
    source_dataset_sha256: str,
    manifest_sha256: str,
    model_version: str = "phase7_path_competing_risk_v1",
    smoothing_alpha: float = 0.5,
    age_bin_ends: Sequence[int | None] = DEFAULT_HAZARD_BIN_ENDS,
    causes: Sequence[str] = DEFAULT_HAZARD_CAUSES,
) -> CompetingRiskLifeTableArtifact:
    """Fit a real-completed-bar, cause-specific empirical life table."""

    bins = _validate_hazard_bins(age_bin_ends)
    registered_causes = tuple(sorted(set(str(value) for value in causes)))
    if not registered_causes or not math.isfinite(smoothing_alpha) or smoothing_alpha <= 0:
        raise ValueError("competing-risk fit contract is invalid")
    typed = tuple(intervals)
    if not typed or any(not isinstance(row, PathRiskInterval) for row in typed):
        raise ValueError("competing-risk fit requires typed risk intervals")
    if len({row.interval_id for row in typed}) != len(typed):
        raise ValueError("competing-risk cohort repeats an interval identity")
    exposures: Counter[tuple[str, int]] = Counter()
    events: Counter[tuple[str, int, str]] = Counter()
    ignored_synthetic = 0
    for row in typed:
        _fit_role(row.split_role)
        if not row.real_completed_bar:
            ignored_synthetic += 1
            continue
        index = _hazard_bin_index(row.age_real_completed_bars, bins)
        exposures[(row.path, index)] += 1
        cause: str | None = None
        if row.terminal_status == "realized":
            cause = "realized"
        elif row.terminal_status == "falsified":
            cause = "falsified"
        elif row.terminal_status == "superseded":
            cause = "scope_superseded"
        elif row.terminal_status == "expired":
            cause = "expired"
        if cause is not None:
            if cause not in registered_causes:
                raise ValueError(f"unregistered competing-risk cause: {cause}")
            events[(row.path, index, cause)] += 1
    if not exposures:
        raise ValueError("competing-risk cohort has no real completed-bar exposure")
    cells: list[CompetingRiskCell] = []
    lower = 1
    starts: list[int] = []
    for end in bins:
        starts.append(lower)
        if end is not None:
            lower = end + 1
    for path in PATH_LABELS:
        for index, end in enumerate(bins):
            risk = exposures[(path, index)]
            if risk < 1:
                continue
            counts = tuple(
                (cause, events[(path, index, cause)])
                for cause in registered_causes
            )
            event_total = sum(value for _, value in counts)
            if event_total > risk:
                raise ValueError("competing-risk terminal count exceeds exposure")
            denominator = risk + smoothing_alpha * (len(registered_causes) + 1)
            hazards = tuple(
                (cause, (count + smoothing_alpha) / denominator)
                for cause, count in counts
            )
            no_event = (risk - event_total + smoothing_alpha) / denominator
            cells.append(
                CompetingRiskCell(
                    path=path,
                    bin_index=index,
                    age_start_bar=starts[index],
                    age_end_bar=end,
                    at_risk_intervals=risk,
                    terminal_counts=counts,
                    cause_hazards=hazards,
                    no_event_probability=no_event,
                )
            )
    return CompetingRiskLifeTableArtifact(
        model_version=model_version,
        source_dataset_sha256=_sha256(
            source_dataset_sha256,
            name="source_dataset_sha256",
        ),
        manifest_sha256=_sha256(manifest_sha256, name="manifest_sha256"),
        smoothing_alpha=float(smoothing_alpha),
        age_bin_ends=bins,
        causes=registered_causes,
        cells=tuple(sorted(cells, key=lambda cell: (cell.path, cell.bin_index))),
        real_at_risk_interval_count=sum(exposures.values()),
        ignored_synthetic_interval_count=ignored_synthetic,
    )


@dataclass(frozen=True)
class EvidenceContributionAtPrediction:
    contribution_id: str
    age_real_completed_bars: int
    path_log_likelihoods: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if (
            not self.contribution_id
            or type(self.age_real_completed_bars) is not int
            or self.age_real_completed_bars < 0
        ):
            raise ValueError("evidence contribution snapshot is invalid")
        object.__setattr__(
            self,
            "path_log_likelihoods",
            _path_mapping(self.path_log_likelihoods, name="path_log_likelihoods"),
        )


@dataclass(frozen=True)
class PriorReversionSample:
    competition_set_id: str
    prediction_known_at: pd.Timestamp
    outcome_known_at: pd.Timestamp
    realized_path: str
    prior_log_weights: tuple[tuple[str, float], ...]
    contributions: tuple[EvidenceContributionAtPrediction, ...]
    split_role: str
    fold_id: str

    def __post_init__(self) -> None:
        if not self.competition_set_id or not self.fold_id or self.realized_path not in PATH_LABELS:
            raise ValueError("prior-reversion sample identity is invalid")
        prediction = aware_timestamp(
            self.prediction_known_at,
            name="reversion prediction_known_at",
        )
        outcome = aware_timestamp(
            self.outcome_known_at,
            name="reversion outcome_known_at",
        )
        if outcome <= prediction:
            raise ValueError("reversion outcome must be known after prediction")
        object.__setattr__(self, "prediction_known_at", prediction)
        object.__setattr__(self, "outcome_known_at", outcome)
        object.__setattr__(
            self,
            "prior_log_weights",
            _path_mapping(self.prior_log_weights, name="prior_log_weights"),
        )
        contributions = tuple(self.contributions)
        if (
            any(not isinstance(item, EvidenceContributionAtPrediction) for item in contributions)
            or len({item.contribution_id for item in contributions}) != len(contributions)
        ):
            raise ValueError("reversion contributions are duplicated or invalid")
        object.__setattr__(
            self,
            "contributions",
            tuple(sorted(contributions, key=lambda item: item.contribution_id)),
        )
        object.__setattr__(self, "split_role", _fit_role(self.split_role))


def _reversion_factor(age: int, half_life: float) -> float:
    if math.isinf(half_life):
        return 1.0
    return math.exp(-math.log(2.0) * float(age) / float(half_life))


def _reversion_probabilities(
    sample: PriorReversionSample,
    half_lives: Mapping[str, float],
) -> np.ndarray:
    prior = dict(sample.prior_log_weights)
    scores = np.asarray([prior[path] for path in PATH_LABELS], dtype=float)
    for contribution in sample.contributions:
        values = dict(contribution.path_log_likelihoods)
        scores += np.asarray(
            [
                _reversion_factor(
                    contribution.age_real_completed_bars,
                    half_lives[path],
                )
                * values[path]
                for path in PATH_LABELS
            ],
            dtype=float,
        )
    return _softmax(scores[None, :])[0]


@dataclass(frozen=True)
class EvidencePriorReversionArtifact:
    model_version: str
    source_dataset_sha256: str
    manifest_sha256: str
    candidate_half_lives: tuple[float, ...]
    path_half_lives: tuple[tuple[str, float], ...]
    weighted_log_loss: float
    fit_sample_count: int
    schema_version: int = PROBABILITY_MODEL_SCHEMA_VERSION
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_dataset_sha256, name="source_dataset_sha256")
        _sha256(self.manifest_sha256, name="manifest_sha256")
        half_lives = _path_mapping(
            self.path_half_lives,
            name="path_half_lives",
            allow_positive_infinity=True,
        )
        if any(value <= 0.0 for _, value in half_lives):
            raise ValueError("path half-lives must be positive")
        object.__setattr__(self, "path_half_lives", half_lives)
        if (
            self.schema_version != PROBABILITY_MODEL_SCHEMA_VERSION
            or not self.model_version
            or not self.candidate_half_lives
            or any(value <= 0.0 or math.isnan(value) for value in self.candidate_half_lives)
            or not math.isfinite(self.weighted_log_loss)
            or self.fit_sample_count < 1
        ):
            raise ValueError("prior-reversion artifact is invalid")
        payload = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "source_dataset_sha256": self.source_dataset_sha256,
            "manifest_sha256": self.manifest_sha256,
            "candidate_half_lives": self.candidate_half_lives,
            "path_half_lives": self.path_half_lives,
            "weighted_log_loss": self.weighted_log_loss,
            "fit_sample_count": self.fit_sample_count,
        }
        object.__setattr__(
            self,
            "artifact_id",
            canonical_identity("prior-reversion-artifact", payload),
        )

    def probabilities(self, sample: PriorReversionSample) -> Mapping[str, float]:
        values = _reversion_probabilities(sample, dict(self.path_half_lives))
        return {path: float(values[index]) for index, path in enumerate(PATH_LABELS)}


def fit_evidence_prior_reversion(
    samples: Sequence[PriorReversionSample],
    *,
    source_dataset_sha256: str,
    manifest_sha256: str,
    model_version: str = "phase7_evidence_prior_reversion_v1",
    candidate_half_lives: Sequence[float] = DEFAULT_HALF_LIFE_GRID,
    coordinate_rounds: int = 4,
) -> EvidencePriorReversionArtifact:
    """Select path-specific evidence half-lives by deterministic grid search."""

    typed = tuple(samples)
    if not typed or any(not isinstance(item, PriorReversionSample) for item in typed):
        raise ValueError("prior-reversion fit requires typed samples")
    if type(coordinate_rounds) is not int or coordinate_rounds < 1:
        raise ValueError("coordinate_rounds must be positive")
    candidates = tuple(float(value) for value in candidate_half_lives)
    if (
        not candidates
        or len(candidates) != len(set(candidates))
        or any(value <= 0.0 or math.isnan(value) for value in candidates)
    ):
        raise ValueError("half-life candidate grid is invalid")
    weights = _unit_weights([sample.competition_set_id for sample in typed])
    labels = np.asarray([PATH_LABELS.index(sample.realized_path) for sample in typed])

    def loss(values: Mapping[str, float]) -> float:
        probabilities = np.vstack(
            [_reversion_probabilities(sample, values) for sample in typed]
        )
        selected = np.maximum(probabilities[np.arange(len(typed)), labels], 1e-300)
        return float(np.sum(weights * -np.log(selected)))

    selected = {path: math.inf if math.inf in candidates else candidates[-1] for path in PATH_LABELS}
    for _ in range(coordinate_rounds):
        changed = False
        for path in PATH_LABELS:
            choices = []
            for order, candidate in enumerate(candidates):
                trial = dict(selected)
                trial[path] = candidate
                choices.append((loss(trial), order, candidate))
            _, _, best = min(choices, key=lambda item: (item[0], item[1]))
            if best != selected[path]:
                selected[path] = best
                changed = True
        if not changed:
            break
    final_loss = loss(selected)
    return EvidencePriorReversionArtifact(
        model_version=model_version,
        source_dataset_sha256=_sha256(
            source_dataset_sha256,
            name="source_dataset_sha256",
        ),
        manifest_sha256=_sha256(manifest_sha256, name="manifest_sha256"),
        candidate_half_lives=candidates,
        path_half_lives=tuple((path, selected[path]) for path in PATH_LABELS),
        weighted_log_loss=final_loss,
        fit_sample_count=len(typed),
    )


@dataclass(frozen=True)
class PathProbabilitySample:
    competition_set_id: str
    prediction_known_at: pd.Timestamp
    outcome_known_at: pd.Timestamp
    realized_path: str
    raw_probabilities: tuple[tuple[str, float], ...]
    split_role: str
    fold_id: str

    def __post_init__(self) -> None:
        prediction = aware_timestamp(
            self.prediction_known_at,
            name="path prediction_known_at",
        )
        outcome = aware_timestamp(self.outcome_known_at, name="path outcome_known_at")
        probabilities = _path_mapping(
            self.raw_probabilities,
            name="raw_path_probabilities",
        )
        if (
            not self.competition_set_id
            or not self.fold_id
            or self.realized_path not in PATH_LABELS
            or outcome <= prediction
            or any(value < 0.0 for _, value in probabilities)
            or not math.isclose(
                math.fsum(value for _, value in probabilities),
                1.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("path probability sample is invalid")
        object.__setattr__(self, "prediction_known_at", prediction)
        object.__setattr__(self, "outcome_known_at", outcome)
        object.__setattr__(self, "raw_probabilities", probabilities)
        object.__setattr__(self, "split_role", _fit_role(self.split_role))


@dataclass(frozen=True)
class PathCalibrationArtifact:
    model_version: str
    source_dataset_sha256: str
    manifest_sha256: str
    source_model_artifact_id: str
    temperature: float
    path_log_biases: tuple[tuple[str, float], ...]
    gauge_path: str
    weighted_log_loss: float
    fit_sample_count: int
    schema_version: int = PROBABILITY_MODEL_SCHEMA_VERSION
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_dataset_sha256, name="source_dataset_sha256")
        _sha256(self.manifest_sha256, name="manifest_sha256")
        biases = _path_mapping(self.path_log_biases, name="path_log_biases")
        object.__setattr__(self, "path_log_biases", biases)
        if (
            self.schema_version != PROBABILITY_MODEL_SCHEMA_VERSION
            or not self.model_version
            or not self.source_model_artifact_id
            or not math.isfinite(self.temperature)
            or self.temperature <= 0.0
            or self.gauge_path not in PATH_LABELS
            or not math.isclose(dict(biases)[self.gauge_path], 0.0, abs_tol=1e-15)
            or not math.isfinite(self.weighted_log_loss)
            or self.fit_sample_count < 1
        ):
            raise ValueError("path calibration artifact is invalid")
        payload = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "source_dataset_sha256": self.source_dataset_sha256,
            "manifest_sha256": self.manifest_sha256,
            "source_model_artifact_id": self.source_model_artifact_id,
            "temperature": self.temperature,
            "path_log_biases": self.path_log_biases,
            "gauge_path": self.gauge_path,
            "weighted_log_loss": self.weighted_log_loss,
            "fit_sample_count": self.fit_sample_count,
        }
        object.__setattr__(
            self,
            "artifact_id",
            canonical_identity("path-calibration-artifact", payload),
        )

    def probabilities(self, raw_probabilities: Mapping[str, float]) -> Mapping[str, float]:
        values = _path_mapping(raw_probabilities, name="raw_path_probabilities")
        raw = np.asarray([max(value, 1e-300) for _, value in values], dtype=float)
        biases = dict(self.path_log_biases)
        scores = np.log(raw) / self.temperature + np.asarray(
            [biases[path] for path in PATH_LABELS]
        )
        calibrated = _softmax(scores[None, :])[0]
        return {path: float(calibrated[index]) for index, path in enumerate(PATH_LABELS)}


def _calibration_loss_gradient(
    theta: np.ndarray,
    log_probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    free_indices: Sequence[int],
    l2: float,
) -> tuple[float, np.ndarray]:
    log_temperature = float(theta[0])
    inverse_temperature = math.exp(-log_temperature)
    biases = np.zeros(log_probabilities.shape[1], dtype=float)
    biases[list(free_indices)] = theta[1:]
    scores = log_probabilities * inverse_temperature + biases
    probabilities = _softmax(scores)
    selected = np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300)
    loss = float(np.sum(weights * -np.log(selected)))
    residual = probabilities.copy()
    residual[np.arange(len(labels)), labels] -= 1.0
    weighted = residual * weights[:, None]
    gradient_temperature = float(
        np.sum(weighted * (-log_probabilities * inverse_temperature))
    )
    gradient_biases = np.sum(weighted, axis=0)[list(free_indices)]
    regularized = theta.copy()
    regularized[0] = 0.0
    loss += 0.5 * l2 * float(np.dot(regularized, regularized))
    gradient = np.r_[gradient_temperature, gradient_biases] + l2 * regularized
    return loss, gradient


def fit_path_temperature_bias(
    samples: Sequence[PathProbabilitySample],
    *,
    source_dataset_sha256: str,
    manifest_sha256: str,
    source_model_artifact_id: str,
    model_version: str = "phase7_path_temperature_bias_v1",
    gauge_path: str = "residual_unknown",
    l2: float = 1e-4,
    maximum_iterations: int = 1000,
) -> PathCalibrationArtifact:
    """Fit multiclass temperature/bias calibration with one fixed bias gauge."""

    typed = tuple(samples)
    if not typed or any(not isinstance(item, PathProbabilitySample) for item in typed):
        raise ValueError("path calibration requires typed samples")
    if gauge_path not in PATH_LABELS or not source_model_artifact_id:
        raise ValueError("path calibration gauge/model binding is invalid")
    if l2 < 0.0 or maximum_iterations < 1:
        raise ValueError("path calibration optimizer contract is invalid")
    raw = np.asarray(
        [[max(value, 1e-300) for _, value in sample.raw_probabilities] for sample in typed],
        dtype=float,
    )
    logs = np.log(raw)
    labels = np.asarray([PATH_LABELS.index(sample.realized_path) for sample in typed])
    weights = _unit_weights([sample.competition_set_id for sample in typed])
    gauge_index = PATH_LABELS.index(gauge_path)
    free_indices = tuple(index for index in range(len(PATH_LABELS)) if index != gauge_index)
    theta = np.zeros(1 + len(free_indices), dtype=float)
    loss, gradient = _calibration_loss_gradient(
        theta,
        logs,
        labels,
        weights,
        free_indices,
        l2,
    )
    for _ in range(maximum_iterations):
        if float(np.max(np.abs(gradient))) < 1e-10:
            break
        direction = -gradient
        step = 1.0
        accepted = False
        directional = float(np.dot(gradient, direction))
        while step >= 1e-12:
            candidate = theta + step * direction
            candidate[0] = float(np.clip(candidate[0], -4.0, 4.0))
            candidate_loss, candidate_gradient = _calibration_loss_gradient(
                candidate,
                logs,
                labels,
                weights,
                free_indices,
                l2,
            )
            if candidate_loss <= loss + 1e-4 * step * directional:
                theta = candidate
                loss = candidate_loss
                gradient = candidate_gradient
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    biases = np.zeros(len(PATH_LABELS), dtype=float)
    biases[list(free_indices)] = theta[1:]
    return PathCalibrationArtifact(
        model_version=model_version,
        source_dataset_sha256=_sha256(
            source_dataset_sha256,
            name="source_dataset_sha256",
        ),
        manifest_sha256=_sha256(manifest_sha256, name="manifest_sha256"),
        source_model_artifact_id=source_model_artifact_id,
        temperature=math.exp(float(theta[0])),
        path_log_biases=tuple(
            (path, float(biases[index])) for index, path in enumerate(PATH_LABELS)
        ),
        gauge_path=gauge_path,
        weighted_log_loss=float(loss),
        fit_sample_count=len(typed),
    )


@dataclass(frozen=True)
class DOLChoiceSetSample:
    competition_set_id: str
    candidate_set_id: str
    path: str
    prediction_known_at: pd.Timestamp
    outcome_known_at: pd.Timestamp
    candidate_scores: tuple[tuple[str, float], ...]
    outcome_id: str
    split_role: str
    fold_id: str

    def __post_init__(self) -> None:
        if (
            not self.competition_set_id
            or not self.candidate_set_id
            or self.path not in PATH_LABELS
            or not self.fold_id
        ):
            raise ValueError("DOL choice-set sample identity is invalid")
        prediction = aware_timestamp(
            self.prediction_known_at,
            name="DOL prediction_known_at",
        )
        outcome = aware_timestamp(
            self.outcome_known_at,
            name="DOL outcome_known_at",
        )
        if outcome <= prediction:
            raise ValueError("DOL outcome must be known after prediction")
        object.__setattr__(self, "prediction_known_at", prediction)
        object.__setattr__(self, "outcome_known_at", outcome)
        candidates = tuple((str(identity), float(score)) for identity, score in self.candidate_scores)
        if (
            tuple(identity for identity, _ in candidates)
            != tuple(sorted(identity for identity, _ in candidates))
            or len({identity for identity, _ in candidates}) != len(candidates)
            or any(not identity or not math.isfinite(score) for identity, score in candidates)
        ):
            raise ValueError("DOL candidate scores are not canonical")
        if self.outcome_id != NO_TARGET_OUTCOME and self.outcome_id not in {
            identity for identity, _ in candidates
        }:
            raise ValueError("DOL outcome is outside the frozen candidate set")
        if not candidates and self.outcome_id != NO_TARGET_OUTCOME:
            raise ValueError("empty DOL inventory can only realize no-target")
        object.__setattr__(self, "candidate_scores", candidates)
        object.__setattr__(self, "split_role", _fit_role(self.split_role))


@dataclass(frozen=True)
class DOLSoftmaxArtifact:
    model_version: str
    source_dataset_sha256: str
    manifest_sha256: str
    source_ranking_fingerprint: str
    candidate_logit_scale: float
    candidate_path_log_weight_adjustment: tuple[tuple[str, float], ...]
    no_target_log_weight: tuple[tuple[str, float], ...]
    weighted_log_loss: float
    fit_sample_count: int
    schema_version: int = PROBABILITY_MODEL_SCHEMA_VERSION
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_dataset_sha256, name="source_dataset_sha256")
        _sha256(self.manifest_sha256, name="manifest_sha256")
        _sha256(self.source_ranking_fingerprint, name="source_ranking_fingerprint")
        adjustments = _path_mapping(
            self.candidate_path_log_weight_adjustment,
            name="candidate_path_log_weight_adjustment",
        )
        no_target = _path_mapping(
            self.no_target_log_weight,
            name="no_target_log_weight",
        )
        object.__setattr__(self, "candidate_path_log_weight_adjustment", adjustments)
        object.__setattr__(self, "no_target_log_weight", no_target)
        if (
            self.schema_version != PROBABILITY_MODEL_SCHEMA_VERSION
            or not self.model_version
            or not math.isfinite(self.candidate_logit_scale)
            or self.candidate_logit_scale < 0.0
            or any(not math.isclose(value, 0.0, abs_tol=1e-15) for _, value in adjustments)
            or not math.isfinite(self.weighted_log_loss)
            or self.fit_sample_count < 1
        ):
            raise ValueError("DOL softmax artifact is invalid")
        payload = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "source_dataset_sha256": self.source_dataset_sha256,
            "manifest_sha256": self.manifest_sha256,
            "source_ranking_fingerprint": self.source_ranking_fingerprint,
            "candidate_logit_scale": self.candidate_logit_scale,
            "candidate_path_log_weight_adjustment": adjustments,
            "no_target_log_weight": no_target,
            "weighted_log_loss": self.weighted_log_loss,
            "fit_sample_count": self.fit_sample_count,
        }
        object.__setattr__(
            self,
            "artifact_id",
            canonical_identity("dol-softmax-artifact", payload),
        )

    def probabilities(self, path: str, candidate_scores: Sequence[tuple[str, float]]) -> Mapping[str, float]:
        if path not in PATH_LABELS:
            raise ValueError("DOL probability path is not canonical")
        candidates = tuple(candidate_scores)
        if not candidates:
            return {NO_TARGET_OUTCOME: 1.0}
        scores = np.asarray(
            [self.candidate_logit_scale * float(value) for _, value in candidates]
            + [dict(self.no_target_log_weight)[path]],
            dtype=float,
        )
        probabilities = _softmax(scores[None, :])[0]
        return {
            **{
                identity: float(probabilities[index])
                for index, (identity, _) in enumerate(candidates)
            },
            NO_TARGET_OUTCOME: float(probabilities[-1]),
        }


def _dol_loss_gradient(
    parameters: np.ndarray,
    samples: Sequence[DOLChoiceSetSample],
    weights: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    scale = float(parameters[0])
    intercepts = parameters[1:]
    loss = 0.0
    gradient = np.zeros_like(parameters)
    for weight, sample in zip(weights, samples, strict=True):
        scores = np.asarray([value for _, value in sample.candidate_scores], dtype=float)
        logits = np.r_[scale * scores, intercepts[PATH_LABELS.index(sample.path)]]
        probabilities = _softmax(logits[None, :])[0]
        outcome_index = (
            len(sample.candidate_scores)
            if sample.outcome_id == NO_TARGET_OUTCOME
            else tuple(identity for identity, _ in sample.candidate_scores).index(
                sample.outcome_id
            )
        )
        loss += float(weight) * -math.log(max(float(probabilities[outcome_index]), 1e-300))
        gradient[0] += float(weight) * (
            float(np.dot(probabilities[:-1], scores))
            - (0.0 if outcome_index == len(scores) else scores[outcome_index])
        )
        gradient[1 + PATH_LABELS.index(sample.path)] += float(weight) * (
            probabilities[-1] - (1.0 if outcome_index == len(scores) else 0.0)
        )
    regularized = parameters.copy()
    regularized[0] = 0.0
    loss += 0.5 * l2 * float(np.dot(regularized, regularized))
    gradient += l2 * regularized
    return loss, gradient


def fit_dol_softmax(
    samples: Sequence[DOLChoiceSetSample],
    *,
    source_dataset_sha256: str,
    manifest_sha256: str,
    source_ranking_fingerprint: str,
    model_version: str = "phase7_dol_conditional_softmax_v1",
    l2: float = 1e-4,
    maximum_iterations: int = 1000,
) -> DOLSoftmaxArtifact:
    """Fit one non-negative candidate scale and path-specific no-target logits."""

    typed = tuple(samples)
    if not typed or any(not isinstance(item, DOLChoiceSetSample) for item in typed):
        raise ValueError("DOL softmax fit requires typed choice sets")
    keys = tuple((item.candidate_set_id, item.path) for item in typed)
    if len(keys) != len(set(keys)):
        raise ValueError("DOL fit repeats a path-conditioned candidate set")
    if l2 < 0.0 or maximum_iterations < 1:
        raise ValueError("DOL optimizer contract is invalid")
    weights = _unit_weights([item.competition_set_id for item in typed])
    parameters = np.r_[1.0, np.zeros(len(PATH_LABELS), dtype=float)]
    loss, gradient = _dol_loss_gradient(parameters, typed, weights, l2)
    for _ in range(maximum_iterations):
        if float(np.max(np.abs(gradient))) < 1e-10:
            break
        direction = -gradient
        directional = float(np.dot(gradient, direction))
        step = 1.0
        accepted = False
        while step >= 1e-12:
            candidate = parameters + step * direction
            candidate[0] = max(0.0, float(candidate[0]))
            candidate_loss, candidate_gradient = _dol_loss_gradient(
                candidate,
                typed,
                weights,
                l2,
            )
            if candidate_loss <= loss + 1e-4 * step * directional:
                parameters = candidate
                loss = candidate_loss
                gradient = candidate_gradient
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return DOLSoftmaxArtifact(
        model_version=model_version,
        source_dataset_sha256=_sha256(
            source_dataset_sha256,
            name="source_dataset_sha256",
        ),
        manifest_sha256=_sha256(manifest_sha256, name="manifest_sha256"),
        source_ranking_fingerprint=_sha256(
            source_ranking_fingerprint,
            name="source_ranking_fingerprint",
        ),
        candidate_logit_scale=float(parameters[0]),
        candidate_path_log_weight_adjustment=tuple(
            (path, 0.0) for path in PATH_LABELS
        ),
        no_target_log_weight=tuple(
            (path, float(parameters[1 + index]))
            for index, path in enumerate(PATH_LABELS)
        ),
        weighted_log_loss=float(loss),
        fit_sample_count=len(typed),
    )


__all__ = [
    "CompetingRiskCell",
    "CompetingRiskLifeTableArtifact",
    "DEFAULT_HALF_LIFE_GRID",
    "DEFAULT_HAZARD_BIN_ENDS",
    "DEFAULT_HAZARD_CAUSES",
    "DEFAULT_HISTORY_TRANSITIONS",
    "DOLChoiceSetSample",
    "DOLSoftmaxArtifact",
    "EvidenceContributionAtPrediction",
    "EvidencePriorReversionArtifact",
    "HistoryConditionalLikelihoodArtifact",
    "HistoryLikelihoodCell",
    "PROBABILITY_MODEL_SCHEMA_VERSION",
    "PathCalibrationArtifact",
    "PathProbabilitySample",
    "PriorReversionSample",
    "fit_competing_risk_life_table",
    "fit_dol_softmax",
    "fit_evidence_prior_reversion",
    "fit_history_conditional_likelihood",
    "fit_path_temperature_bias",
]
