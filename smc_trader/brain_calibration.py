"""Event-level causal targets for the typed playbook brain.

This module deliberately does not fit a model.  It freezes a small number of
typed belief observations, then resolves them from later completed 1m OHLCV.
It is designed to be called by the existing replay loop::

    recorder.on_bar(bar)        # before the engine consumes this bar
    snapshot = replay.on_bar(...)
    recorder.observe(snapshot)  # after the completed-bar snapshot exists

Only DFP and LSR are observed.  FAVR remains parked.  Sequence progress and
uncertainty are descriptive rows; the other four dimensions carry causal
future outcomes suitable for a separate, simple calibration fitter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import pandas as pd

from .model import (
    BOSLifecycle,
    BOSScope,
    Bar,
    Direction,
    EngineSnapshot,
    EntryLocationLifecycle,
    Playbook,
    PlaybookPhase,
    StructureLifecycle,
    Timeframe,
    aware_timestamp,
)


RECORDER_VERSION = "typed-brain-targets-v5-scene-contract"

SUPPORTED_PLAYBOOKS = frozenset(
    {
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    }
)
FITTED_DIMENSIONS = frozenset(
    {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
)
DESCRIPTIVE_DIMENSIONS = frozenset(
    {"sequence_progress", "uncertainty"}
)
ALL_DIMENSIONS = FITTED_DIMENSIONS | DESCRIPTIVE_DIMENSIONS

_BOUNDARY_ANOMALIES = frozenset(
    {
        "contract_change_history_reset",
        "data_gap_history_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)

_DFP_THESIS_FAILURE_REASONS = frozenset(
    {"opposed_structure", "frozen_h1_bos_missing"}
)
_DFP_LOCAL_EPISODE_TERMINAL_REASONS = frozenset(
    {
        "frozen_entry_location_missing",
        "entry_zone_left_or_failed",
        "trigger_opposed_or_ambiguous",
        "entry_window_expired",
        "frozen_invalidation_breached",
        "entry_zone_beyond_frozen_invalidation",
    }
)
_LSR_THESIS_FAILURE_REASONS = frozenset(
    {
        "accepted_outside_or_failed",
        "frozen_pool_path_missing",
        "source_manipulation_missing",
        "micro_bos_opposed_or_ambiguous",
        "frozen_invalidation_breached",
        "position_invalidation_breached",
        "position_invalidated",
    }
)
_READINESS_FAILURE_REASONS = frozenset(
    {
        "trigger_opposed_or_ambiguous",
        "micro_bos_opposed_or_ambiguous",
        "entry_trigger_contradicted",
        "entry_zone_left_or_failed",
        "entry_zone_left",
    }
)
_CENSOR_TERMINAL_REASONS = frozenset(
    {
        "entry_path_censored",
        "pool_path_censored",
        "data_gap_reset",
        "contract_change_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
_INVALID_TRIGGER_TERMINAL_REASONS = (
    _DFP_THESIS_FAILURE_REASONS
    | _LSR_THESIS_FAILURE_REASONS
    | _READINESS_FAILURE_REASONS
    | _CENSOR_TERMINAL_REASONS
    | frozenset(
        {
            "episode_deadline_elapsed",
            "deadline_elapsed",
            "entry_window_expired",
            "frozen_entry_location_missing",
            "entry_zone_beyond_frozen_invalidation",
            "selected_draw_consumed_or_missing",
        }
    )
)


def _canonical_json(value: Mapping[str, str]) -> str:
    return json.dumps(
        dict(sorted((str(key), str(item)) for key, item in value.items())),
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_identities(values: Any) -> str:
    identities = tuple(dict.fromkeys(str(item) for item in values))
    if any(not item for item in identities):
        raise ValueError("identity lists cannot contain empty values")
    return json.dumps(list(identities), separators=(",", ":"))


def _validate_canonical_identities(value: str, *, name: str) -> None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if (
        not isinstance(parsed, list)
        or any(not isinstance(item, str) or not item for item in parsed)
        or len(parsed) != len(set(parsed))
        or _canonical_identities(parsed) != value
    ):
        raise ValueError(f"{name} identities are invalid")


def _validate_hash(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.isoformat()


def _timestamp(value: Any, *, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    return aware_timestamp(value, name=name)


def _clamped(value: Any, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return parsed


@dataclass(frozen=True)
class BrainCalibrationRecord:
    """One resolved calibration observation written to a light shard."""

    sample_id: str
    hypothesis_key: str
    scene_hypothesis_id: str
    competing_scene_hypothesis_ids: str
    context_root_ids: str
    scene_revision_id: str
    playbook: str
    direction: str
    dimension: str
    setup_id: str
    episode_id: str | None
    context_id: str | None
    evidence_revision_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    sampled_at: pd.Timestamp
    resolved_at: pd.Timestamp
    raw_value: float
    outcome_value: float | None
    resolution: str
    censored: bool
    fit_eligible: bool
    phase: str
    origin_price: float
    trigger_bar_high: float
    trigger_bar_low: float
    invalidation_price: float | None
    invalidation_source_id: str | None
    draw_price: float | None
    draw_id: str | None
    liquidity_route_id: str | None
    context_draw_id: str | None
    intermediate_liquidity_ids: str
    primary_deliverable_target_id: str | None
    terminal_draw_id: str | None
    path_blocker_ids: str
    source_path_ids: str
    dfp_structure_id: str | None
    dfp_structure_confirmed_at: pd.Timestamp | None
    dfp_h1_bos_id: str | None
    deadline: pd.Timestamp | None
    symbol: str
    instrument_id: int
    protocol_version: str
    protocol_hash: str
    registry_hash: str
    model_code_hash: str
    config_hash: str
    primitive_protocol_hashes: str
    brain_input_contract_hash: str

    def __post_init__(self) -> None:
        for name in (
            "sampled_at",
            "resolved_at",
            "dfp_structure_confirmed_at",
            "deadline",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"brain_calibration.{name}"),
                )
        if (
            not self.sample_id
            or not self.hypothesis_key
            or not self.scene_hypothesis_id
            or not self.scene_revision_id
            or self.playbook
            not in {item.value for item in SUPPORTED_PLAYBOOKS}
            or self.direction not in {item.value for item in Direction}
            or self.dimension not in ALL_DIMENSIONS
            or not self.setup_id
            or not self.resolution
            or not self.phase
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or not self.protocol_version
            or not self.protocol_hash
        ):
            raise ValueError("brain calibration record identity is invalid")
        for name in (
            "competing_scene_hypothesis_ids",
            "context_root_ids",
            "intermediate_liquidity_ids",
            "path_blocker_ids",
            "source_path_ids",
        ):
            _validate_canonical_identities(getattr(self, name), name=name)
        for name in (
            "liquidity_route_id",
            "context_draw_id",
            "primary_deliverable_target_id",
            "terminal_draw_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} is invalid")
        _validate_hash(
            self.brain_input_contract_hash,
            name="brain_input_contract_hash",
        )
        _clamped(self.raw_value, name="record.raw_value")
        if self.outcome_value is not None:
            _clamped(self.outcome_value, name="record.outcome_value")
        if (
            self.resolved_at < self.sampled_at
            or (self.deadline is not None and self.deadline < self.sampled_at)
            or not math.isfinite(float(self.origin_price))
            or self.origin_price <= 0.0
            or not math.isfinite(float(self.trigger_bar_high))
            or not math.isfinite(float(self.trigger_bar_low))
            or self.trigger_bar_high < self.trigger_bar_low
            or not self.trigger_bar_low
            <= self.origin_price
            <= self.trigger_bar_high
            or any(
                value is not None
                and (not math.isfinite(float(value)) or float(value) <= 0.0)
                for value in (self.invalidation_price, self.draw_price)
            )
        ):
            raise ValueError("brain calibration record clock or price is invalid")
        if type(self.censored) is not bool or type(self.fit_eligible) is not bool:
            raise ValueError("brain calibration flags must be boolean")
        if self.censored and self.outcome_value is not None:
            raise ValueError("censored calibration rows cannot carry outcomes")
        expected_fit = bool(
            self.dimension in FITTED_DIMENSIONS
            and not self.censored
            and self.outcome_value is not None
        )
        if self.fit_eligible != expected_fit:
            raise ValueError("fit eligibility disagrees with row resolution")
        if self.dimension in DESCRIPTIVE_DIMENSIONS and (
            self.outcome_value is not None or self.censored
        ):
            raise ValueError("descriptive dimensions cannot carry future labels")
        try:
            primitive_hashes = json.loads(self.primitive_protocol_hashes)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("primitive protocol hashes must be canonical JSON") from exc
        if (
            not isinstance(primitive_hashes, dict)
            or not primitive_hashes
            or _canonical_json(primitive_hashes) != self.primitive_protocol_hashes
        ):
            raise ValueError("primitive protocol hashes are invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a parquet/jsonl-friendly mapping without changing clocks."""

        return asdict(self)


@dataclass(frozen=True)
class _OpenSample:
    sample_id: str
    hypothesis_key: str
    scene_hypothesis_id: str
    competing_scene_hypothesis_ids: str
    context_root_ids: str
    scene_revision_id: str
    playbook: str
    direction: str
    dimension: str
    setup_id: str
    episode_id: str | None
    context_id: str | None
    evidence_revision_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    sampled_at: pd.Timestamp
    raw_value: float
    phase: str
    origin_price: float
    trigger_bar_high: float
    trigger_bar_low: float
    invalidation_price: float | None
    invalidation_source_id: str | None
    draw_price: float | None
    draw_id: str | None
    liquidity_route_id: str | None
    context_draw_id: str | None
    intermediate_liquidity_ids: str
    primary_deliverable_target_id: str | None
    terminal_draw_id: str | None
    path_blocker_ids: str
    source_path_ids: str
    dfp_structure_id: str | None
    dfp_structure_confirmed_at: pd.Timestamp | None
    dfp_h1_bos_id: str | None
    deadline: pd.Timestamp | None
    symbol: str
    instrument_id: int
    protocol_version: str
    protocol_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sampled_at",
            aware_timestamp(self.sampled_at, name="open_sample.sampled_at"),
        )
        if self.deadline is not None:
            object.__setattr__(
                self,
                "deadline",
                aware_timestamp(self.deadline, name="open_sample.deadline"),
            )
        if self.dfp_structure_confirmed_at is not None:
            object.__setattr__(
                self,
                "dfp_structure_confirmed_at",
                aware_timestamp(
                    self.dfp_structure_confirmed_at,
                    name="open_sample.dfp_structure_confirmed_at",
                ),
            )


@dataclass(frozen=True)
class _CandidateIdentity:
    hypothesis_key: str
    scene_hypothesis_id: str
    competing_scene_hypothesis_ids: str
    context_root_ids: str
    scene_revision_id: str

    def __post_init__(self) -> None:
        if (
            not self.hypothesis_key
            or not self.scene_hypothesis_id
            or not self.scene_revision_id
        ):
            raise ValueError("candidate scene identity is required")
        _validate_canonical_identities(
            self.competing_scene_hypothesis_ids,
            name="competing_scene_hypothesis_ids",
        )
        _validate_canonical_identities(
            self.context_root_ids,
            name="context_root_ids",
        )


class BrainCalibrationRecorder:
    """Incrementally freeze and causally resolve typed Brain dimensions."""

    def __init__(
        self,
        *,
        registry_hash: str,
        model_code_hash: str,
        config_hash: str,
        primitive_protocol_hashes: Mapping[str, str],
        brain_input_contract_hash: str,
    ) -> None:
        for name, value in (
            ("registry_hash", registry_hash),
            ("model_code_hash", model_code_hash),
            ("config_hash", config_hash),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} is required")
        if not primitive_protocol_hashes:
            raise ValueError("primitive protocol hashes are required")
        _validate_hash(
            brain_input_contract_hash,
            name="brain_input_contract_hash",
        )
        self.registry_hash = registry_hash
        self.model_code_hash = model_code_hash
        self.config_hash = config_hash
        self.primitive_protocol_hashes = _canonical_json(
            primitive_protocol_hashes
        )
        self.brain_input_contract_hash = brain_input_contract_hash
        self._open: dict[str, _OpenSample] = {}
        self._rows: list[BrainCalibrationRecord] = []
        self._seen: set[str] = set()
        self._seen_location_ids: set[tuple[str, str]] = set()
        self._last_bar_start: pd.Timestamp | None = None
        self._last_observation_asof: pd.Timestamp | None = None

    @property
    def open_samples(self) -> tuple[_OpenSample, ...]:
        return tuple(self._open[key] for key in sorted(self._open))

    @property
    def rows(self) -> tuple[BrainCalibrationRecord, ...]:
        return tuple(self._rows)

    def drain_rows(self) -> tuple[BrainCalibrationRecord, ...]:
        rows = tuple(self._rows)
        self._rows.clear()
        return rows

    def on_bar(self, bar: Bar) -> None:
        """Resolve existing samples from one later completed 1m bar.

        This must run before the engine creates the snapshot for ``bar``.  A
        sample created from that snapshot therefore cannot see its own bar.
        """

        if self._last_bar_start is not None and bar.start <= self._last_bar_start:
            raise ValueError("calibration bars must be strictly increasing")
        self._last_bar_start = bar.start
        if not self._open:
            return
        if bar.synthetic_no_trade:
            # A no-trade placeholder contains no executable price path.  It
            # is skipped; only an explicit hard boundary may censor a target.
            return
        if bar.data_gap_before_minutes:
            self._censor_all(bar.end, "data_gap_boundary")
            return
        for sample in tuple(self._open.values()):
            if (bar.symbol, bar.instrument_id) != (
                sample.symbol,
                sample.instrument_id,
            ):
                self._finish(
                    sample,
                    resolved_at=bar.end,
                    outcome=None,
                    resolution="contract_boundary",
                    censored=True,
                )
                continue
            self._resolve_with_bar(sample, bar)

    def observe(
        self,
        snapshot: EngineSnapshot,
        *,
        source_bar: Bar,
    ) -> None:
        """Resolve typed lifecycles, then freeze newly eligible samples."""

        observation = snapshot.observation
        belief = snapshot.belief
        asof = aware_timestamp(
            observation.asof,
            name="brain_calibration.observation.asof",
        )
        if (
            source_bar.end != asof
            or (source_bar.symbol, source_bar.instrument_id)
            != (observation.symbol, observation.instrument_id)
            or not math.isclose(
                float(source_bar.close),
                float(observation.price),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "source bar must be the exact completed bar behind snapshot"
            )
        if (
            self._last_observation_asof is not None
            and asof <= self._last_observation_asof
        ):
            raise ValueError("calibration observations must be strictly increasing")
        self._last_observation_asof = asof
        if set(observation.anomalies) & _BOUNDARY_ANOMALIES:
            self._censor_all(asof, "observation_boundary")
            return
        if source_bar.synthetic_no_trade:
            return

        candidates = tuple(belief.candidates())
        hypotheses = {
            hypothesis.key: hypothesis
            for hypothesis in candidates
            if hypothesis.playbook in SUPPORTED_PLAYBOOKS
        }
        if len(hypotheses) != sum(
            hypothesis.playbook in SUPPORTED_PLAYBOOKS
            for hypothesis in candidates
        ):
            raise ValueError("Brain candidates contain duplicate hypothesis keys")
        identities = {
            key: self._candidate_identity(belief, hypothesis)
            for key, hypothesis in hypotheses.items()
        }
        locations = {
            item.location_id: item
            for item in getattr(observation, "entry_locations", ())
        }
        self._resolve_dfp_context_theses(asof, observation)
        self._resolve_from_snapshot(asof, hypotheses, locations)
        for sample in tuple(self._open.values()):
            if sample.deadline is not None and sample.deadline <= asof:
                self._deadline(sample, sample.deadline)
        for hypothesis in hypotheses.values():
            self._register_hypothesis(
                observation,
                hypothesis,
                identities[hypothesis.key],
                locations,
                source_bar,
            )

    @staticmethod
    def _candidate_identity(
        belief: Any,
        hypothesis: Any,
    ) -> _CandidateIdentity:
        contexts = tuple(
            item
            for item in belief.context_hypotheses.values()
            if item.playbook is hypothesis.playbook
            and item.direction is hypothesis.direction
        )
        if not contexts:
            raise ValueError(
                f"Brain candidate {hypothesis.key} lacks a scene hypothesis"
            )
        primary = contexts[0]
        return _CandidateIdentity(
            hypothesis_key=hypothesis.key,
            scene_hypothesis_id=str(primary.hypothesis_id),
            competing_scene_hypothesis_ids=_canonical_identities(
                item.hypothesis_id for item in contexts[1:]
            ),
            context_root_ids=_canonical_identities(
                primary.context_root_ids
            ),
            scene_revision_id=str(
                getattr(belief, "scene_revision_id", "") or ""
            ),
        )

    def close_unresolved(
        self,
        asof: pd.Timestamp,
        *,
        reason: str = "window_end",
    ) -> None:
        """Right-censor every unresolved sample at a replay boundary."""

        clock = aware_timestamp(asof, name="brain_calibration.window_end")
        self._censor_all(clock, reason)

    def state_dict(self) -> dict[str, Any]:
        """Serialize enough state for exact checkpoint/resume."""

        return {
            "version": RECORDER_VERSION,
            "registry_hash": self.registry_hash,
            "model_code_hash": self.model_code_hash,
            "config_hash": self.config_hash,
            "primitive_protocol_hashes": self.primitive_protocol_hashes,
            "brain_input_contract_hash": self.brain_input_contract_hash,
            "open_samples": [
                self._serialize_open(sample)
                for sample in self.open_samples
            ],
            "queued_rows": [self._serialize_row(row) for row in self._rows],
            "seen_sample_ids": sorted(self._seen),
            "seen_location_ids": [
                list(item) for item in sorted(self._seen_location_ids)
            ],
            "last_bar_start": _iso(self._last_bar_start),
            "last_observation_asof": _iso(self._last_observation_asof),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "BrainCalibrationRecorder":
        """Restore a recorder produced by :meth:`state_dict`."""

        if state.get("version") != RECORDER_VERSION:
            raise ValueError("unsupported brain calibration recorder state")
        primitive_json = str(state.get("primitive_protocol_hashes", ""))
        try:
            primitive_hashes = json.loads(primitive_json)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid primitive hashes in recorder state") from exc
        recorder = cls(
            registry_hash=str(state.get("registry_hash", "")),
            model_code_hash=str(state.get("model_code_hash", "")),
            config_hash=str(state.get("config_hash", "")),
            primitive_protocol_hashes=primitive_hashes,
            brain_input_contract_hash=str(
                state.get("brain_input_contract_hash", "")
            ),
        )
        recorder._seen = {str(item) for item in state.get("seen_sample_ids", ())}
        recorder._seen_location_ids = {
            (str(item[0]), str(item[1]))
            for item in state.get("seen_location_ids", ())
        }
        recorder._last_bar_start = _timestamp(
            state.get("last_bar_start"),
            name="brain_calibration.last_bar_start",
        )
        recorder._last_observation_asof = _timestamp(
            state.get("last_observation_asof"),
            name="brain_calibration.last_observation_asof",
        )
        for payload in state.get("open_samples", ()):
            sample = recorder._deserialize_open(payload)
            if sample.sample_id in recorder._open:
                raise ValueError("duplicate open calibration sample in state")
            recorder._open[sample.sample_id] = sample
        recorder._rows = [
            recorder._deserialize_row(payload)
            for payload in state.get("queued_rows", ())
        ]
        return recorder

    def _resolve_with_bar(self, sample: _OpenSample, bar: Bar) -> None:
        deadline = sample.deadline
        if deadline is not None and bar.start >= deadline:
            self._deadline(sample, deadline)
            return
        invalidated = bool(
            sample.invalidation_price is not None
            and (
                (
                    sample.direction == Direction.LONG.value
                    and bar.low <= sample.invalidation_price
                )
                or (
                    sample.direction == Direction.SHORT.value
                    and bar.high >= sample.invalidation_price
                )
            )
        )
        # Conservative same-bar ordering: invalidation always wins.
        if invalidated:
            self._finish(
                sample,
                resolved_at=bar.end,
                outcome=0.0,
                resolution="invalidation_touched",
            )
            return
        draw_touched = bool(
            sample.draw_price is not None
            and (
                (
                    sample.direction == Direction.LONG.value
                    and bar.high >= sample.draw_price
                )
                or (
                    sample.direction == Direction.SHORT.value
                    and bar.low <= sample.draw_price
                )
            )
        )
        if sample.dimension in {"thesis_strength", "delivery_quality"}:
            if draw_touched:
                self._finish(
                    sample,
                    resolved_at=bar.end,
                    outcome=1.0,
                    resolution="draw_delivered",
                )
                return
        elif sample.dimension == "entry_readiness":
            if bar.start >= sample.sampled_at:
                progress = (
                    1.0
                    if (
                        sample.direction == Direction.LONG.value
                        and bar.close > sample.trigger_bar_high
                    )
                    or (
                        sample.direction == Direction.SHORT.value
                        and bar.close < sample.trigger_bar_low
                    )
                    else 0.0
                )
                self._finish(
                    sample,
                    resolved_at=bar.end,
                    outcome=progress,
                    resolution=(
                        "next_close_advanced_toward_draw"
                        if progress
                        else "next_close_failed_to_advance"
                    ),
                )
                return
    def _deadline(self, sample: _OpenSample, deadline: pd.Timestamp) -> None:
        thesis_intact = sample.dimension == "thesis_strength"
        self._finish(
            sample,
            resolved_at=deadline,
            outcome=1.0 if thesis_intact else 0.0,
            resolution=(
                "thesis_intact_at_deadline"
                if thesis_intact
                else "deadline_without_dimension_delivery"
            ),
        )

    @staticmethod
    def _thesis_failure(sample: _OpenSample, reason: str | None) -> bool:
        if sample.playbook == Playbook.DISPLACEMENT_FIRST_PULLBACK.value:
            return reason in _DFP_THESIS_FAILURE_REASONS
        return reason in _LSR_THESIS_FAILURE_REASONS

    def _resolve_dfp_context_theses(
        self,
        asof: pd.Timestamp,
        observation: Any,
    ) -> None:
        """Resolve frozen DFP context authority independently of entry episodes."""

        frame = getattr(observation, "frame", None)
        if not callable(frame):
            return
        h4 = frame(Timeframe.H4)
        h1 = frame(Timeframe.H1)
        for sample in tuple(self._open.values()):
            if (
                sample.playbook
                != Playbook.DISPLACEMENT_FIRST_PULLBACK.value
                or sample.dimension != "thesis_strength"
                or sample.dfp_structure_id is None
            ):
                continue
            aligned = next(
                (
                    item
                    for item in h4.structures
                    if item.structure_id == sample.dfp_structure_id
                    and item.direction.value == sample.direction
                    and item.lifecycle is StructureLifecycle.CONFIRMED
                    and item.confirmed_at is not None
                ),
                None,
            )
            if aligned is None:
                self._finish(
                    sample,
                    resolved_at=asof,
                    outcome=0.0,
                    resolution="thesis_contradicted:opposed_structure",
                )
                continue
            structure_clock = (
                sample.dfp_structure_confirmed_at
                or aligned.confirmed_at
            )
            opposed = any(
                item.direction.value != sample.direction
                and item.lifecycle is StructureLifecycle.CONFIRMED
                and item.confirmed_at is not None
                and item.confirmed_at >= structure_clock
                for item in h4.structures
            )
            if opposed:
                self._finish(
                    sample,
                    resolved_at=asof,
                    outcome=0.0,
                    resolution="thesis_contradicted:opposed_structure",
                )
                continue
            if sample.dfp_h1_bos_id is None:
                continue
            frozen_h1_present = any(
                item.bos_id == sample.dfp_h1_bos_id
                and item.direction.value == sample.direction
                and item.lifecycle is BOSLifecycle.CONFIRMED
                and item.scope is BOSScope.CONTINUATION
                for item in h1.structure_breaks
            )
            if not frozen_h1_present:
                self._finish(
                    sample,
                    resolved_at=asof,
                    outcome=0.0,
                    resolution="thesis_contradicted:frozen_h1_bos_missing",
                )

    def _resolve_from_snapshot(
        self,
        asof: pd.Timestamp,
        hypotheses: Mapping[str, Any],
        locations: Mapping[str, Any],
    ) -> None:
        for sample in tuple(self._open.values()):
            if sample.dimension == "location_quality":
                location = locations.get(sample.entry_location_id)
                if location is not None:
                    if location.lifecycle is EntryLocationLifecycle.REJECTED:
                        self._finish(
                            sample,
                            resolved_at=asof,
                            outcome=1.0,
                            resolution="frozen_zone_rejected",
                        )
                        continue
                    if location.lifecycle is EntryLocationLifecycle.LEFT:
                        self._finish(
                            sample,
                            resolved_at=asof,
                            outcome=0.0,
                            resolution="frozen_zone_left",
                        )
                        continue
            if sample.dimension in {
                "entry_readiness",
                "delivery_quality",
            }:
                # Trigger-clock targets own frozen price geometry.  A later
                # Brain terminal/rearm cannot rewrite their causal path.
                continue
            hypothesis = hypotheses.get(sample.hypothesis_key)
            if hypothesis is None:
                self._finish(
                    sample,
                    resolved_at=asof,
                    outcome=None,
                    resolution="retention_disappearance_without_terminal",
                    censored=True,
                )
                continue
            phase = hypothesis.phase
            reason = getattr(hypothesis, "terminal_reason", None)
            matching_terminal = self._terminal_matches_sample(
                sample,
                hypothesis,
            )
            if matching_terminal and phase is PlaybookPhase.COMPLETED:
                self._finish(
                    sample,
                    resolved_at=asof,
                    outcome=1.0,
                    resolution="episode_completed",
                )
                continue
            if matching_terminal and phase is PlaybookPhase.INVALIDATED:
                if reason in _CENSOR_TERMINAL_REASONS:
                    self._finish(
                        sample,
                        resolved_at=asof,
                        outcome=None,
                        resolution=f"terminal_censored:{reason}",
                        censored=True,
                    )
                    continue
                elif reason in {"episode_deadline_elapsed", "deadline_elapsed"}:
                    self._deadline(sample, asof)
                    continue
                elif (
                    sample.dimension == "thesis_strength"
                    and self._thesis_failure(sample, reason)
                ):
                    self._finish(
                        sample,
                        resolved_at=asof,
                        outcome=0.0,
                        resolution=f"thesis_contradicted:{reason}",
                    )
                    continue
                elif sample.dimension == "thesis_strength":
                    if (
                        sample.playbook
                        == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
                        and reason in _DFP_LOCAL_EPISODE_TERMINAL_REASONS
                    ):
                        # DFP's H4/H1 context thesis owns a longer lifecycle
                        # than any one entry-zone/trigger attempt.  A local
                        # episode can close and rearm while the frozen
                        # context, draw and deadline remain authoritative.
                        # Leave the target open so only its own contradiction,
                        # delivery, deadline or hard boundary resolves it.
                        pass
                    else:
                        # A terminal outside the explicitly local DFP set
                        # cannot establish whether the frozen thesis survived.
                        # In particular, a missing draw identity must not be
                        # relabelled as successful merely because its price was
                        # not observed after custody was lost.
                        self._finish(
                            sample,
                            resolved_at=asof,
                            outcome=None,
                            resolution="non_thesis_terminal_unresolved",
                            censored=True,
                        )
                        continue
                elif (
                    sample.dimension == "location_quality"
                    and reason in _READINESS_FAILURE_REASONS
                ):
                    self._finish(
                        sample,
                        resolved_at=asof,
                        outcome=0.0,
                        resolution=f"location_failed:{reason}",
                    )
                    continue

            current_owner = self._sample_owner_id(
                hypothesis,
                sample.dimension,
            )
            if current_owner != sample.setup_id:
                self._finish(
                    sample,
                    resolved_at=asof,
                    outcome=None,
                    resolution="setup_replaced_without_matching_terminal",
                    censored=True,
                )

    @staticmethod
    def _terminal_matches_sample(
        sample: _OpenSample,
        hypothesis: Any,
    ) -> bool:
        """Match a frozen target to the exact setup/context being closed."""

        if hypothesis.phase not in {
            PlaybookPhase.COMPLETED,
            PlaybookPhase.INVALIDATED,
        }:
            return False
        sequence = getattr(hypothesis, "sequence", None)
        identities = {
            str(value)
            for value in (
                *tuple(getattr(hypothesis, "terminal_source_ids", ())),
                getattr(hypothesis, "setup_context_id", None),
                getattr(hypothesis, "episode_id", None),
                getattr(hypothesis, "context_id", None),
                None if sequence is None else sequence.setup_id,
            )
            if value is not None
        }
        return sample.setup_id in identities

    def _register_hypothesis(
        self,
        observation: Any,
        hypothesis: Any,
        identity: _CandidateIdentity,
        locations: Mapping[str, Any],
        source_bar: Bar,
    ) -> None:
        sequence = hypothesis.sequence
        if (
            sequence is None
            or sequence.setup_id is None
            or hypothesis.phase
            in {PlaybookPhase.INACTIVE, PlaybookPhase.COMPLETED}
        ):
            return
        path_id = self._entry_path_id(observation, hypothesis)
        raw = self._raw_dimensions(hypothesis)

        valid_trigger = self._valid_trigger_observation(
            hypothesis,
            raw,
        )
        if hypothesis.phase is PlaybookPhase.INVALIDATED:
            if valid_trigger:
                self._register_trigger_targets(
                    observation,
                    hypothesis,
                    identity,
                    raw=raw,
                    path_id=path_id,
                    source_bar=source_bar,
                )
            return

        revision = getattr(hypothesis, "evidence_revision_id", None)
        if revision is not None:
            self._register(
                observation,
                hypothesis,
                identity,
                dimension="thesis_strength",
                raw_value=raw["thesis_strength"],
                revision_key=revision,
                entry_path_id=path_id,
                source_bar=source_bar,
            )
            self._register_descriptive(
                observation,
                hypothesis,
                identity,
                dimension="uncertainty",
                raw_value=raw["uncertainty"],
                revision_key=revision,
                entry_path_id=path_id,
                resolution="contemporaneous_uncertainty_observed",
                source_bar=source_bar,
            )

        sequence_key = (
            f"{sequence.setup_id}:{hypothesis.phase.value}:"
            f"{sequence.completed_steps}/{len(sequence.steps)}"
        )
        self._register_descriptive(
            observation,
            hypothesis,
            identity,
            dimension="sequence_progress",
            raw_value=raw["sequence_progress"],
            revision_key=sequence_key,
            entry_path_id=path_id,
            resolution="sequence_observed",
            source_bar=source_bar,
        )

        location_id = getattr(hypothesis, "entry_location_id", None)
        location = locations.get(location_id)
        if location is not None:
            location_seen_key = (hypothesis.key, location.location_id)
            if location_seen_key not in self._seen_location_ids:
                if location.lifecycle in {
                    EntryLocationLifecycle.APPROACHING,
                    EntryLocationLifecycle.IN_ZONE,
                }:
                    registered = self._register(
                        observation,
                        hypothesis,
                        identity,
                        dimension="location_quality",
                        raw_value=raw["location_quality"],
                        revision_key=f"location:{location.location_id}",
                        entry_path_id=path_id,
                        source_bar=source_bar,
                    )
                    if registered:
                        self._seen_location_ids.add(location_seen_key)
                else:
                    # A terminal location first discovered at a replay/window
                    # boundary is audit context, not a calibration target.
                    self._seen_location_ids.add(location_seen_key)

        if valid_trigger:
            self._register_trigger_targets(
                observation,
                hypothesis,
                identity,
                raw=raw,
                path_id=path_id,
                source_bar=source_bar,
            )

    @staticmethod
    def _valid_trigger_observation(
        hypothesis: Any,
        raw: Mapping[str, float],
    ) -> bool:
        sequence = hypothesis.sequence
        if not sequence.complete or raw["entry_readiness"] <= 0.0:
            return False
        if hypothesis.phase is not PlaybookPhase.INVALIDATED:
            return True
        # A completed, aligned trigger can share its clock with a delivery
        # veto (for example insufficient remaining path).  Explicit trigger,
        # structure, frozen-invalidation, deadline, or custody failures do
        # not describe a valid trigger and must not manufacture targets.
        reason = getattr(hypothesis, "terminal_reason", None)
        return reason not in _INVALID_TRIGGER_TERMINAL_REASONS

    def _register_trigger_targets(
        self,
        observation: Any,
        hypothesis: Any,
        identity: _CandidateIdentity,
        *,
        raw: Mapping[str, float],
        path_id: str | None,
        source_bar: Bar,
    ) -> None:
        sequence = hypothesis.sequence
        revision = getattr(hypothesis, "evidence_revision_id", None)
        trigger_id = path_id or revision or sequence.setup_id
        self._register(
            observation,
            hypothesis,
            identity,
            dimension="entry_readiness",
            raw_value=raw["entry_readiness"],
            revision_key=f"trigger:{trigger_id}",
            entry_path_id=path_id,
            source_bar=source_bar,
        )
        self._register(
            observation,
            hypothesis,
            identity,
            dimension="delivery_quality",
            raw_value=raw["delivery_quality"],
            revision_key=f"delivery-trigger:{trigger_id}",
            entry_path_id=path_id,
            source_bar=source_bar,
        )

    def _register(
        self,
        observation: Any,
        hypothesis: Any,
        identity: _CandidateIdentity,
        *,
        dimension: str,
        raw_value: float,
        revision_key: str,
        entry_path_id: str | None,
        source_bar: Bar,
    ) -> bool:
        sample = self._freeze_sample(
            observation,
            hypothesis,
            identity,
            dimension=dimension,
            raw_value=raw_value,
            revision_key=revision_key,
            entry_path_id=entry_path_id,
            source_bar=source_bar,
        )
        if sample.sample_id in self._seen:
            return True
        required = (
            (sample.draw_price, sample.deadline)
            if dimension == "thesis_strength"
            else (
                (sample.invalidation_price, sample.deadline)
                if dimension == "location_quality"
                else (
                    sample.invalidation_price,
                    sample.draw_price,
                    sample.deadline,
                )
            )
        )
        if any(value is None for value in required):
            # Do not consume the revision identity.  The same causal event may
            # become resolvable after its frozen draw/deadline is published.
            return False
        self._seen.add(sample.sample_id)
        if sample.deadline is not None and sample.deadline <= sample.sampled_at:
            self._finish(
                sample,
                resolved_at=sample.sampled_at,
                outcome=None,
                resolution="non_future_deadline",
                censored=True,
            )
            return True
        self._open[sample.sample_id] = sample
        return True

    def _register_descriptive(
        self,
        observation: Any,
        hypothesis: Any,
        identity: _CandidateIdentity,
        *,
        dimension: str,
        raw_value: float,
        revision_key: str,
        entry_path_id: str | None,
        resolution: str,
        source_bar: Bar,
    ) -> None:
        sample = self._freeze_sample(
            observation,
            hypothesis,
            identity,
            dimension=dimension,
            raw_value=raw_value,
            revision_key=revision_key,
            entry_path_id=entry_path_id,
            source_bar=source_bar,
        )
        if sample.sample_id in self._seen:
            return
        self._seen.add(sample.sample_id)
        self._finish(
            sample,
            resolved_at=sample.sampled_at,
            outcome=None,
            resolution=resolution,
        )

    def _freeze_sample(
        self,
        observation: Any,
        hypothesis: Any,
        identity: _CandidateIdentity,
        *,
        dimension: str,
        raw_value: float,
        revision_key: str,
        entry_path_id: str | None,
        source_bar: Bar,
    ) -> _OpenSample:
        sequence = hypothesis.sequence
        invalidation = hypothesis.invalidation
        plan = hypothesis.plan
        if invalidation is None and plan is not None:
            invalidation = plan.invalidation
        draw = getattr(hypothesis, "draw_selection", None)
        targets = tuple(getattr(hypothesis, "deliverable_targets", ()))
        target = None
        if draw is not None:
            target = next(
                (item for item in targets if item.level_id == draw.draw_id),
                None,
            )
        if target is None and targets:
            target = targets[0]
        draw_id = (
            draw.draw_id
            if draw is not None
            else None if target is None else target.level_id
        )
        draw_price = (
            draw.price
            if draw is not None
            else None if target is None else target.price
        )
        liquidity_route = getattr(hypothesis, "liquidity_route", None)
        if liquidity_route is None and plan is not None:
            liquidity_route = getattr(plan, "liquidity_route", None)
        deadline = getattr(hypothesis, "episode_deadline", None)
        if deadline is None and plan is not None:
            deadline = plan.deadline
        sampled_at = aware_timestamp(
            observation.asof,
            name="brain_calibration.sampled_at",
        )
        if deadline is None:
            execution = getattr(observation, "execution", None)
            minutes = (
                None
                if execution is None
                else getattr(execution, "minutes_to_deadline", None)
            )
            if minutes is not None and int(minutes) >= 0:
                deadline = sampled_at + pd.Timedelta(minutes=int(minutes))
        if (
            dimension == "thesis_strength"
            and hypothesis.playbook
            is Playbook.DISPLACEMENT_FIRST_PULLBACK
        ):
            # DFP's entry-zone protection is not its H4/H1 thesis
            # contradiction.  Opposed structure/BOS is resolved from the
            # typed terminal reason instead.
            invalidation = None
        dfp_structure_id = None
        dfp_structure_confirmed_at = None
        dfp_h1_bos_id = None
        if (
            dimension == "thesis_strength"
            and hypothesis.playbook
            is Playbook.DISPLACEMENT_FIRST_PULLBACK
        ):
            steps = {
                getattr(step, "step_id", None): step
                for step in getattr(sequence, "steps", ())
            }
            context_step = steps.get("h4_structure_and_draw")
            context_sources = tuple(
                getattr(context_step, "source_ids", ())
            )
            if context_sources:
                dfp_structure_id = str(context_sources[0])
                frame = getattr(observation, "frame", None)
                if callable(frame):
                    structure = next(
                        (
                            item
                            for item in frame(Timeframe.H4).structures
                            if item.structure_id == dfp_structure_id
                        ),
                        None,
                    )
                    if structure is not None:
                        dfp_structure_confirmed_at = structure.confirmed_at
                if dfp_structure_confirmed_at is None:
                    dfp_structure_confirmed_at = getattr(
                        context_step,
                        "observed_at",
                        None,
                    )
            h1_step = steps.get("h1_continuation_bos")
            h1_sources = tuple(getattr(h1_step, "source_ids", ()))
            if getattr(h1_step, "satisfied", False) and h1_sources:
                dfp_h1_bos_id = str(h1_sources[0])
        owner_id = self._sample_owner_id(hypothesis, dimension)
        if owner_id is None:
            raise ValueError(
                f"{dimension} calibration sample lacks an owner identity"
            )
        identity_payload = {
            "version": RECORDER_VERSION,
            "playbook": hypothesis.playbook.value,
            "direction": hypothesis.direction.value,
            "dimension": dimension,
            "setup_id": owner_id,
            "revision_key": str(revision_key),
            "scene_hypothesis_id": identity.scene_hypothesis_id,
            "brain_input_contract_hash": self.brain_input_contract_hash,
        }
        sample_id = hashlib.sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return _OpenSample(
            sample_id=sample_id,
            hypothesis_key=identity.hypothesis_key,
            scene_hypothesis_id=identity.scene_hypothesis_id,
            competing_scene_hypothesis_ids=(
                identity.competing_scene_hypothesis_ids
            ),
            context_root_ids=identity.context_root_ids,
            scene_revision_id=identity.scene_revision_id,
            playbook=hypothesis.playbook.value,
            direction=hypothesis.direction.value,
            dimension=dimension,
            setup_id=owner_id,
            episode_id=getattr(hypothesis, "episode_id", None),
            context_id=getattr(hypothesis, "context_id", None),
            evidence_revision_id=getattr(
                hypothesis,
                "evidence_revision_id",
                None,
            ),
            entry_location_id=getattr(
                hypothesis,
                "entry_location_id",
                None,
            ),
            entry_path_id=entry_path_id,
            sampled_at=sampled_at,
            raw_value=_clamped(raw_value, name=f"raw.{dimension}"),
            phase=hypothesis.phase.value,
            origin_price=float(observation.price),
            trigger_bar_high=float(source_bar.high),
            trigger_bar_low=float(source_bar.low),
            invalidation_price=(
                None if invalidation is None else float(invalidation.price)
            ),
            invalidation_source_id=(
                None
                if invalidation is None
                else invalidation.source_level_id
            ),
            draw_price=None if draw_price is None else float(draw_price),
            draw_id=draw_id,
            liquidity_route_id=(
                None
                if liquidity_route is None
                else liquidity_route.route_id
            ),
            context_draw_id=(
                None
                if liquidity_route is None
                else liquidity_route.context_draw_id
            ),
            intermediate_liquidity_ids=_canonical_identities(
                ()
                if liquidity_route is None
                else liquidity_route.intermediate_liquidity_ids
            ),
            primary_deliverable_target_id=(
                None
                if liquidity_route is None
                else liquidity_route.primary_deliverable_target_id
            ),
            terminal_draw_id=(
                None
                if liquidity_route is None
                else liquidity_route.terminal_draw_id
            ),
            path_blocker_ids=_canonical_identities(
                ()
                if liquidity_route is None
                else liquidity_route.path_blocker_ids
            ),
            source_path_ids=_canonical_identities(
                ()
                if liquidity_route is None
                else liquidity_route.source_path_ids
            ),
            dfp_structure_id=dfp_structure_id,
            dfp_structure_confirmed_at=_timestamp(
                dfp_structure_confirmed_at,
                name="brain_calibration.dfp_structure_confirmed_at",
            ),
            dfp_h1_bos_id=dfp_h1_bos_id,
            deadline=_timestamp(deadline, name="brain_calibration.deadline"),
            symbol=str(observation.symbol),
            instrument_id=int(observation.instrument_id),
            protocol_version=sequence.protocol_version,
            protocol_hash=sequence.protocol_hash,
        )

    @staticmethod
    def _sample_owner_id(
        hypothesis: Any,
        dimension: str,
    ) -> str | None:
        if (
            dimension == "thesis_strength"
            and hypothesis.playbook
            is Playbook.DISPLACEMENT_FIRST_PULLBACK
        ):
            context_id = getattr(hypothesis, "context_id", None)
            if context_id is not None:
                return str(context_id)
        setup_id = getattr(hypothesis, "setup_context_id", None)
        if setup_id is not None:
            return str(setup_id)
        sequence = getattr(hypothesis, "sequence", None)
        return (
            None
            if sequence is None or sequence.setup_id is None
            else str(sequence.setup_id)
        )

    @staticmethod
    def _raw_dimensions(hypothesis: Any) -> dict[str, float]:
        supplied = dict(getattr(hypothesis, "raw_quality_dimensions", {}))
        fallback = {
            "thesis_strength": (
                hypothesis.raw_probability
                if getattr(hypothesis, "raw_probability", None) is not None
                else hypothesis.thesis_strength
            ),
            "sequence_progress": hypothesis.sequence_progress,
            "location_quality": hypothesis.location_quality,
            "entry_readiness": hypothesis.entry_readiness,
            "delivery_quality": hypothesis.delivery_quality,
            "uncertainty": hypothesis.uncertainty,
        }
        output: dict[str, float] = {}
        for dimension in ALL_DIMENSIONS:
            value = supplied.get(dimension, fallback[dimension])
            if value is None:
                raise ValueError(
                    f"typed belief lacks raw {dimension} for calibration"
                )
            output[dimension] = _clamped(
                value,
                name=f"raw.{dimension}",
            )
        return output

    @staticmethod
    def _entry_path_id(observation: Any, hypothesis: Any) -> str | None:
        plan = hypothesis.plan
        if plan is not None and getattr(plan, "entry_path_id", None):
            return plan.entry_path_id
        location_id = getattr(hypothesis, "entry_location_id", None)
        matches = [
            item
            for item in getattr(observation, "path_sequences", ())
            if item.context_kind == "zone_return"
            and item.context_id == location_id
            and item.direction is hypothesis.direction
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: item.last_updated_at).sequence_id

    def _finish(
        self,
        sample: _OpenSample,
        *,
        resolved_at: pd.Timestamp,
        outcome: float | None,
        resolution: str,
        censored: bool = False,
    ) -> None:
        self._open.pop(sample.sample_id, None)
        record = BrainCalibrationRecord(
            sample_id=sample.sample_id,
            hypothesis_key=sample.hypothesis_key,
            scene_hypothesis_id=sample.scene_hypothesis_id,
            competing_scene_hypothesis_ids=(
                sample.competing_scene_hypothesis_ids
            ),
            context_root_ids=sample.context_root_ids,
            scene_revision_id=sample.scene_revision_id,
            playbook=sample.playbook,
            direction=sample.direction,
            dimension=sample.dimension,
            setup_id=sample.setup_id,
            episode_id=sample.episode_id,
            context_id=sample.context_id,
            evidence_revision_id=sample.evidence_revision_id,
            entry_location_id=sample.entry_location_id,
            entry_path_id=sample.entry_path_id,
            sampled_at=sample.sampled_at,
            resolved_at=resolved_at,
            raw_value=sample.raw_value,
            outcome_value=outcome,
            resolution=resolution,
            censored=censored,
            fit_eligible=bool(
                sample.dimension in FITTED_DIMENSIONS
                and not censored
                and outcome is not None
            ),
            phase=sample.phase,
            origin_price=sample.origin_price,
            trigger_bar_high=sample.trigger_bar_high,
            trigger_bar_low=sample.trigger_bar_low,
            invalidation_price=sample.invalidation_price,
            invalidation_source_id=sample.invalidation_source_id,
            draw_price=sample.draw_price,
            draw_id=sample.draw_id,
            liquidity_route_id=sample.liquidity_route_id,
            context_draw_id=sample.context_draw_id,
            intermediate_liquidity_ids=(
                sample.intermediate_liquidity_ids
            ),
            primary_deliverable_target_id=(
                sample.primary_deliverable_target_id
            ),
            terminal_draw_id=sample.terminal_draw_id,
            path_blocker_ids=sample.path_blocker_ids,
            source_path_ids=sample.source_path_ids,
            dfp_structure_id=sample.dfp_structure_id,
            dfp_structure_confirmed_at=(
                sample.dfp_structure_confirmed_at
            ),
            dfp_h1_bos_id=sample.dfp_h1_bos_id,
            deadline=sample.deadline,
            symbol=sample.symbol,
            instrument_id=sample.instrument_id,
            protocol_version=sample.protocol_version,
            protocol_hash=sample.protocol_hash,
            registry_hash=self.registry_hash,
            model_code_hash=self.model_code_hash,
            config_hash=self.config_hash,
            primitive_protocol_hashes=self.primitive_protocol_hashes,
            brain_input_contract_hash=self.brain_input_contract_hash,
        )
        self._rows.append(record)

    def _censor_all(self, asof: pd.Timestamp, reason: str) -> None:
        clock = aware_timestamp(asof, name="brain_calibration.censor_at")
        for sample in tuple(self._open.values()):
            self._finish(
                sample,
                resolved_at=max(clock, sample.sampled_at),
                outcome=None,
                resolution=reason,
                censored=True,
            )

    @staticmethod
    def _serialize_open(sample: _OpenSample) -> dict[str, Any]:
        payload = asdict(sample)
        payload["sampled_at"] = _iso(sample.sampled_at)
        payload["dfp_structure_confirmed_at"] = _iso(
            sample.dfp_structure_confirmed_at
        )
        payload["deadline"] = _iso(sample.deadline)
        return payload

    @staticmethod
    def _deserialize_open(payload: Mapping[str, Any]) -> _OpenSample:
        values = dict(payload)
        values["sampled_at"] = _timestamp(
            values.get("sampled_at"),
            name="brain_calibration.sampled_at",
        )
        values["deadline"] = _timestamp(
            values.get("deadline"),
            name="brain_calibration.deadline",
        )
        values["dfp_structure_confirmed_at"] = _timestamp(
            values.get("dfp_structure_confirmed_at"),
            name="brain_calibration.dfp_structure_confirmed_at",
        )
        return _OpenSample(**values)

    @staticmethod
    def _serialize_row(row: BrainCalibrationRecord) -> dict[str, Any]:
        payload = asdict(row)
        payload["sampled_at"] = _iso(row.sampled_at)
        payload["resolved_at"] = _iso(row.resolved_at)
        payload["dfp_structure_confirmed_at"] = _iso(
            row.dfp_structure_confirmed_at
        )
        payload["deadline"] = _iso(row.deadline)
        return payload

    @staticmethod
    def _deserialize_row(payload: Mapping[str, Any]) -> BrainCalibrationRecord:
        values = dict(payload)
        for name in (
            "sampled_at",
            "resolved_at",
            "dfp_structure_confirmed_at",
            "deadline",
        ):
            values[name] = _timestamp(
                values.get(name),
                name=f"brain_calibration.{name}",
            )
        return BrainCalibrationRecord(**values)


__all__ = [
    "ALL_DIMENSIONS",
    "BrainCalibrationRecord",
    "BrainCalibrationRecorder",
    "DESCRIPTIVE_DIMENSIONS",
    "FITTED_DIMENSIONS",
    "RECORDER_VERSION",
    "SUPPORTED_PLAYBOOKS",
]
