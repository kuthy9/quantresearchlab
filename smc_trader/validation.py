"""Preregistered data windows, setup funnels, and frozen future-path tests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .model import (
    Bar,
    Direction,
    EngineSnapshot,
    HypothesisBelief,
    PlaybookPhase,
    TradePlan,
    aware_timestamp,
)


class ValidationProtocolError(ValueError):
    """Raised when a replay would cross a preregistered data boundary."""


@dataclass(frozen=True)
class ValidationWindow:
    role: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    purpose: str

    def contains(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return self.start <= start and end <= self.end_exclusive


@dataclass(frozen=True)
class ValidationProtocol:
    version: str
    fingerprint: str
    causal_front_sha256: str
    belief_calibration_valid_from: pd.Timestamp
    ohlcv_windows: Mapping[str, ValidationWindow]
    mbo_windows: Mapping[str, ValidationWindow]

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
            raise ValidationProtocolError(f"{name}.{role} has an invalid aware interval")
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


def load_validation_protocol(
    path: str | Path = "configs/validation_protocol_v2.json",
) -> ValidationProtocol:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[1] / source
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValidationProtocolError("validation protocol root must be an object")
    version = str(payload.get("protocol_version", "")).strip()
    ohlcv_windows = _load_windows(
        payload.get("ohlcv_windows"),
        name="ohlcv_windows",
    )
    calibration_value = payload.get("belief_calibration_valid_from")
    if calibration_value is None:
        if version.startswith("2.3."):
            raise ValidationProtocolError(
                "v2.3 belief_calibration_valid_from must be explicit"
            )
        calibration_valid_from = min(
            window.start for window in ohlcv_windows.values()
        )
    else:
        calibration_valid_from = pd.Timestamp(calibration_value)
        if calibration_valid_from.tzinfo is None:
            raise ValidationProtocolError(
                "belief_calibration_valid_from must be timezone aware"
            )
    return ValidationProtocol(
        version=version,
        fingerprint=hashlib.sha256(raw).hexdigest(),
        causal_front_sha256=str(payload.get("causal_front_sha256", "")).strip(),
        belief_calibration_valid_from=calibration_valid_from,
        ohlcv_windows=ohlcv_windows,
        mbo_windows=_load_windows(
            payload.get("mbo_windows"),
            name="mbo_windows",
        ),
    )


@dataclass(frozen=True)
class FunnelTransition:
    setup_id: str
    playbook: str
    direction: str
    phase: str
    terminal_at: pd.Timestamp | None
    terminal_reason: str | None
    terminal_source_ids: str
    observed_at: pd.Timestamp
    probability: float
    uncertainty: float
    completed_steps: int
    total_steps: int
    protocol_version: str
    protocol_hash: str
    snapshot_hash: str


class PlaybookFunnelRecorder:
    """Records each setup-phase transition once without creating bar labels."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._rows: list[FunnelTransition] = []

    @property
    def rows(self) -> tuple[FunnelTransition, ...]:
        return tuple(self._rows)

    def drain_rows(self) -> tuple[FunnelTransition, ...]:
        """Move completed transitions to durable output while retaining de-duplication."""

        rows = tuple(self._rows)
        self._rows.clear()
        return rows

    def observe(self, snapshot: EngineSnapshot) -> None:
        for hypothesis in snapshot.belief.hypotheses.values():
            sequence = hypothesis.sequence
            if sequence is None or sequence.setup_id is None:
                continue
            key = (sequence.setup_id, hypothesis.phase.value)
            if key in self._seen:
                continue
            self._seen.add(key)
            self._rows.append(
                FunnelTransition(
                    setup_id=sequence.setup_id,
                    playbook=hypothesis.playbook.value,
                    direction=hypothesis.direction.value,
                    phase=hypothesis.phase.value,
                    terminal_at=hypothesis.terminal_at,
                    terminal_reason=hypothesis.terminal_reason,
                    terminal_source_ids=json.dumps(
                        list(hypothesis.terminal_source_ids),
                        separators=(",", ":"),
                    ),
                    observed_at=snapshot.observation.asof,
                    probability=hypothesis.probability,
                    uncertainty=hypothesis.uncertainty,
                    completed_steps=sequence.completed_steps,
                    total_steps=len(sequence.steps),
                    protocol_version=sequence.protocol_version,
                    protocol_hash=sequence.protocol_hash,
                    snapshot_hash=snapshot.snapshot_hash,
                )
            )


@dataclass(frozen=True)
class FrozenPathTest:
    setup_id: str
    hypothesis_key: str
    entry_location_id: str | None
    entry_path_id: str | None
    playbook: str
    direction: str
    setup_started_at: pd.Timestamp
    sequence_completed_at: pd.Timestamp
    decision_time: pd.Timestamp
    entry: float
    invalidation: float
    invalidation_source_id: str
    target: float
    target_source_id: str
    deadline: pd.Timestamp
    risk_points: float
    probability: float
    raw_probability: float
    uncertainty: float
    phase: str
    calibration_version: str
    calibration_hash: str
    protocol_version: str
    protocol_hash: str
    decision_hash: str
    config_hash: str
    code_hash: str


@dataclass(frozen=True)
class PathTestResult:
    setup_id: str
    hypothesis_key: str
    entry_location_id: str | None
    entry_path_id: str | None
    playbook: str
    direction: str
    setup_started_at: pd.Timestamp
    sequence_completed_at: pd.Timestamp
    decision_time: pd.Timestamp
    resolved_at: pd.Timestamp
    outcome: str
    success: bool
    entry: float
    invalidation: float
    invalidation_source_id: str
    target: float
    target_source_id: str
    deadline: pd.Timestamp
    probability: float
    raw_probability: float
    uncertainty: float
    phase: str
    calibration_version: str
    calibration_hash: str
    mfe_R: float
    mae_R: float
    elapsed_minutes: int
    formation_minutes: int
    entry_touched: bool
    entry_touched_at: pd.Timestamp | None
    time_to_entry_minutes: int | None
    ambiguous_same_bar: bool
    decision_hash: str
    protocol_version: str
    protocol_hash: str
    config_hash: str
    code_hash: str


@dataclass(frozen=True)
class PrimitivePathCaseResult:
    proposal_id: str
    definition_hash: str
    origin_packet_hash: str
    origin_packet_sha256: str
    evaluation_packet_hash: str
    evaluation_packet_sha256: str
    origin_decision_hash: str
    evaluation_decision_hash: str
    origin_hypothesis_key: str | None
    evaluation_hypothesis_key: str
    origin_setup_id: str | None
    evaluation_setup_id: str | None
    evaluation_entry_location_id: str | None
    evaluation_entry_path_id: str | None
    origin_window_role: str
    evaluation_window_role: str
    distinct_case: bool
    non_overlapping_window: bool
    causal_clock_valid: bool
    prefix_invariant: bool
    bounds_valid: bool
    implementation_consistency_passed: bool
    path_evidence_hash: str | None
    path_evidence_verified: bool
    path_property_passed: bool | None
    evaluable: bool
    status: str
    reason: str
    no_pnl_fields_used: bool
    origin_values: Mapping[str, Any]
    evaluation_values: Mapping[str, Any]


@dataclass(frozen=True)
class PrimitiveProposalEvaluation:
    proposal_id: str
    definition_hash: str
    implementation_checked_cases: int
    implementation_failed_cases: int
    path_passed_cases: int
    path_failed_cases: int
    awaiting_path_evidence_cases: int
    not_evaluable_cases: int
    status: str
    reason: str


def _primitive_property(
    issue: str,
    values: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Return fixed bounds and implementation self-consistency only."""

    if issue == "missed_structure_sequence":
        hypothesis_valid = bool(values["hypothesis_prefix_valid"])
        path_valid = bool(values["typed_entry_path_chain_valid"])
        context_valid = bool(values["exact_entry_context"])
        step_ids = tuple(values["typed_entry_path_step_ids"])
        return (
            bool(step_ids) and len(step_ids) == len(set(step_ids)),
            bool(values["implementation_consistent"])
            == (hypothesis_valid and path_valid and context_valid),
        )
    if issue == "pullback_as_invalidation":
        penetration = float(values["zone_penetration"])
        room = float(values["room_R"])
        zone_breach = bool(values["zone_completed_close_breach"])
        stop_touched = bool(values["thesis_stop_touched"])
        zone_expected = (
            values["zone_lifecycle"] != "left"
            and not zone_breach
        )
        return (
            0.0 <= penetration <= 1.0 and math.isfinite(room),
            bool(values["zone_intact"]) == zone_expected
            and bool(values["thesis_intact"]) == (not stop_touched)
            and bool(values["normal_pullback"])
            == (zone_expected and not stop_touched),
        )
    if issue == "late_displacement_chase":
        extension = float(values["impulse_extension_atr"])
        remaining = float(values["net_remaining_draw_atr"])
        raw_ratio = values["extension_to_remaining_draw_ratio"]
        ratio = None if raw_ratio is None else float(raw_ratio)
        return (
            extension >= 0.0
            and remaining >= 0.0
            and (
                ratio is None
                if remaining <= 0.0
                else math.isfinite(ratio) and ratio >= 0.0
            ),
            bool(values["path_exhausted"]) == (remaining <= 0.0)
            and ((ratio is None) == (remaining <= 0.0)),
        )
    if issue == "wrong_liquidity_draw":
        selected = values["selected_draw_id"]
        expected = values["expected_draw_id"]
        target = values["primary_target_id"]
        return (
            all(
                isinstance(item, str) and bool(item)
                for item in (selected, expected, target)
            ),
            bool(values["provenance_matches"])
            == (
                selected == expected
                and selected == target
                and bool(values["delivery_side_valid"])
                and values["draw_lifecycle"] in {"visible", "targeted"}
            ),
        )
    checks = (
        bool(values["source_price_matches"]),
        bool(values["source_clock_causal"]),
        bool(values["plan_clock_matches_source"]),
        bool(values["side_matches"]),
        bool(values["setup_location_path_identities_agree"]),
        bool(values["playbook_source_binding"]),
        bool(values["risk_geometry_matches"]),
    )
    distance = float(values["distance_R"])
    return (
        math.isfinite(distance) and distance > 0.0,
        bool(values["valid"]) == all(checks),
    )


def evaluate_primitive_implementation_case(
    proposal: Any,
    evaluation_packet: Mapping[str, Any],
    *,
    evaluation_hypothesis_key: str,
    origin_window: ValidationWindow,
    evaluation_window: ValidationWindow,
) -> PrimitivePathCaseResult:
    """Run a cross-packet implementation check; no future path is consumed."""

    from .ai_review import compute_primitive
    from .decision_trace import decision_packet_payload_sha256

    value = compute_primitive(
        proposal.issue,
        evaluation_packet,
        hypothesis_key=evaluation_hypothesis_key,
    )
    if value.definition_hash != proposal.definition_hash:
        raise ValueError("primitive evaluation definition changed")
    evaluation_setup_id = value.setup_id
    evaluation_location_id = value.entry_location_id
    evaluation_path_id = value.entry_path_id
    origin_clock = pd.Timestamp(proposal.origin_value.decision_asof)
    evaluation_clock = pd.Timestamp(value.decision_asof)
    origin_clock_bound = (
        origin_window.start
        <= origin_clock
        < origin_window.end_exclusive
    )
    evaluation_clock_bound = (
        evaluation_window.start
        <= evaluation_clock
        < evaluation_window.end_exclusive
    )
    non_overlapping = (
        origin_window.end_exclusive <= evaluation_window.start
        or evaluation_window.end_exclusive <= origin_window.start
    )
    distinct = (
        value.decision_packet_hash != proposal.decision_packet_hash
        and value.decision_hash != proposal.decision_hash
        and evaluation_setup_id is not None
        and evaluation_setup_id != proposal.setup_id
    )
    causal = (
        origin_clock_bound
        and evaluation_clock_bound
        and value.decision_asof
        == pd.Timestamp(evaluation_packet["decision_asof"])
        and evaluation_packet.get("future_path", {}).get("included")
        is False
    )
    bounds_valid = False
    property_passed = False
    if value.evaluable:
        try:
            bounds_valid, property_passed = _primitive_property(
                proposal.issue.value,
                value.values,
            )
        except (KeyError, TypeError, ValueError):
            bounds_valid = False
            property_passed = False
    prefix_invariant = (
        value.decision_packet_hash
        == evaluation_packet.get("packet_hash")
        and value.decision_packet_sha256
        == decision_packet_payload_sha256(evaluation_packet)
    )
    if value.decision_packet_hash == proposal.decision_packet_hash:
        status = "diagnostic_only"
        reason = "origin case cannot approve its own proposed primitive"
    elif not distinct or not non_overlapping:
        status = "ineligible"
        reason = "evaluation requires a different setup and non-overlapping window"
    elif not value.evaluable:
        status = "not_evaluable"
        reason = value.reason
    elif causal and prefix_invariant and bounds_valid and property_passed:
        status = "implementation_checked"
        reason = (
            "cross-packet implementation consistency passed; separate "
            "post-reveal typed path evidence is still required"
        )
    else:
        status = "implementation_failed"
        reason = (
            "causal identity, bounds or implementation consistency failed"
        )
    return PrimitivePathCaseResult(
        proposal_id=proposal.proposal_id,
        definition_hash=proposal.definition_hash,
        origin_packet_hash=proposal.decision_packet_hash,
        origin_packet_sha256=proposal.decision_packet_sha256,
        evaluation_packet_hash=value.decision_packet_hash,
        evaluation_packet_sha256=value.decision_packet_sha256,
        origin_decision_hash=proposal.decision_hash,
        evaluation_decision_hash=value.decision_hash,
        origin_hypothesis_key=proposal.hypothesis_key,
        evaluation_hypothesis_key=evaluation_hypothesis_key,
        origin_setup_id=proposal.setup_id,
        evaluation_setup_id=evaluation_setup_id,
        evaluation_entry_location_id=evaluation_location_id,
        evaluation_entry_path_id=evaluation_path_id,
        origin_window_role=origin_window.role,
        evaluation_window_role=evaluation_window.role,
        distinct_case=distinct,
        non_overlapping_window=non_overlapping,
        causal_clock_valid=causal,
        prefix_invariant=prefix_invariant,
        bounds_valid=bounds_valid,
        implementation_consistency_passed=property_passed,
        path_evidence_hash=None,
        path_evidence_verified=False,
        path_property_passed=None,
        evaluable=value.evaluable,
        status=status,
        reason=reason,
        no_pnl_fields_used=True,
        origin_values=dict(proposal.origin_value.values),
        evaluation_values=dict(value.values),
    )


def finalize_primitive_evaluation(
    proposal: Any,
    cases: Sequence[PrimitivePathCaseResult],
) -> PrimitiveProposalEvaluation:
    bound = [
        item
        for item in cases
        if item.proposal_id == proposal.proposal_id
        and item.definition_hash == proposal.definition_hash
    ]
    if len(bound) != len(cases):
        raise ValueError("primitive evaluation contains another proposal")
    unique_checks: dict[
        tuple[str | None, str],
        PrimitivePathCaseResult,
    ] = {}
    for item in bound:
        if item.status not in {
            "implementation_checked",
            "implementation_failed",
        }:
            continue
        key = (item.evaluation_setup_id, item.evaluation_window_role)
        prior = unique_checks.get(key)
        if prior is not None and prior.status != item.status:
            raise ValueError(
                "one primitive evaluation setup has conflicting results"
            )
        unique_checks.setdefault(key, item)
    implementation_checked = sum(
        item.status == "implementation_checked"
        for item in unique_checks.values()
    )
    implementation_failed = sum(
        item.status == "implementation_failed"
        for item in unique_checks.values()
    )
    # PrimitivePathCaseResult is an implementation-check carrier.  Its legacy
    # path booleans and hash strings are caller-supplied and therefore cannot
    # promote or reject a concept.  A later diagnostic stage must read and
    # verify the physically separate evidence artifact before Agent 3/4 can
    # freeze a semantic assessment.
    path_passed = 0
    path_failed = 0
    awaiting_path = implementation_checked
    not_evaluable = sum(
        item.status in {"not_evaluable", "ineligible", "diagnostic_only"}
        for item in bound
    )
    if implementation_failed:
        status = "implementation_failed"
        reason = "fix the deterministic implementation before path review"
    else:
        status = "awaiting_path_evidence"
        reason = (
            "implementation checks and caller-supplied flags cannot approve "
            "a primitive; read verified independent evidence before semantic "
            "path assessment"
        )
    return PrimitiveProposalEvaluation(
        proposal_id=proposal.proposal_id,
        definition_hash=proposal.definition_hash,
        implementation_checked_cases=implementation_checked,
        implementation_failed_cases=implementation_failed,
        path_passed_cases=path_passed,
        path_failed_cases=path_failed,
        awaiting_path_evidence_cases=awaiting_path,
        not_evaluable_cases=not_evaluable,
        status=status,
        reason=reason,
    )


@dataclass
class _OpenPath:
    frozen: FrozenPathTest
    symbol: str
    instrument_id: int
    maximum_favorable_R: float = 0.0
    maximum_adverse_R: float = 0.0
    entry_touched_at: pd.Timestamp | None = None


def _freeze_path(
    snapshot: EngineSnapshot,
    hypothesis: HypothesisBelief,
    plan: TradePlan,
    config_hash: str,
    code_hash: str,
) -> FrozenPathTest:
    sequence = hypothesis.sequence
    if sequence is None or sequence.setup_id is None:
        raise ValueError("path test requires a registered setup identity")
    completed_clocks = [
        step.observed_at
        for step in sequence.steps
        if step.satisfied and step.observed_at is not None
    ]
    if sequence.started_at is None or not completed_clocks:
        raise ValueError("path test requires complete causal sequence clocks")
    return FrozenPathTest(
        setup_id=sequence.setup_id,
        hypothesis_key=hypothesis.key,
        entry_location_id=plan.entry_location_id,
        entry_path_id=plan.entry_path_id,
        playbook=hypothesis.playbook.value,
        direction=hypothesis.direction.value,
        setup_started_at=sequence.started_at,
        sequence_completed_at=max(completed_clocks),
        decision_time=snapshot.observation.asof,
        entry=plan.planned_entry,
        invalidation=plan.invalidation.price,
        invalidation_source_id=plan.invalidation.source_level_id,
        target=plan.targets[0].price,
        target_source_id=plan.targets[0].level_id,
        deadline=plan.deadline,
        risk_points=plan.risk_points,
        probability=hypothesis.probability,
        raw_probability=(
            hypothesis.probability
            if hypothesis.raw_probability is None
            else hypothesis.raw_probability
        ),
        uncertainty=hypothesis.uncertainty,
        phase=hypothesis.phase.value,
        calibration_version=hypothesis.calibration_version,
        calibration_hash=hypothesis.calibration_hash,
        protocol_version=sequence.protocol_version,
        protocol_hash=sequence.protocol_hash,
        decision_hash=snapshot.snapshot_hash,
        config_hash=config_hash,
        code_hash=code_hash,
    )


class FrozenPathTestRecorder:
    """Freezes complete-sequence plans, then reads only later completed bars."""

    def __init__(self, *, config_hash: str, code_hash: str) -> None:
        if not config_hash or not code_hash:
            raise ValueError("path tests require frozen config and model-code hashes")
        self.config_hash = config_hash
        self.code_hash = code_hash
        self._registered: set[str] = set()
        self._open: dict[str, _OpenPath] = {}
        self._results: list[PathTestResult] = []

    @property
    def open_tests(self) -> tuple[FrozenPathTest, ...]:
        return tuple(item.frozen for item in self._open.values())

    @property
    def results(self) -> tuple[PathTestResult, ...]:
        return tuple(self._results)

    def drain_results(self) -> tuple[PathTestResult, ...]:
        """Move resolved tests to durable output while retaining open/setup state."""

        results = tuple(self._results)
        self._results.clear()
        return results

    def observe(self, snapshot: EngineSnapshot) -> None:
        for hypothesis in snapshot.belief.hypotheses.values():
            sequence = hypothesis.sequence
            if (
                sequence is None
                or sequence.setup_id is None
                or not sequence.complete
                or hypothesis.plan is None
                or sequence.setup_id in self._registered
            ):
                continue
            frozen = _freeze_path(
                snapshot,
                hypothesis,
                hypothesis.plan,
                self.config_hash,
                self.code_hash,
            )
            self._registered.add(frozen.setup_id)
            self._open[frozen.setup_id] = _OpenPath(
                frozen=frozen,
                symbol=snapshot.observation.symbol,
                instrument_id=snapshot.observation.instrument_id,
            )

    def _resolve(
        self,
        setup_id: str,
        *,
        resolved_at: pd.Timestamp,
        outcome: str,
        ambiguous: bool = False,
    ) -> None:
        state = self._open.pop(setup_id)
        frozen = state.frozen
        elapsed = max(
            0,
            int((resolved_at - frozen.decision_time).total_seconds() // 60),
        )
        formation = max(
            0,
            int(
                (
                    frozen.sequence_completed_at - frozen.setup_started_at
                ).total_seconds()
                // 60
            ),
        )
        time_to_entry = (
            None
            if state.entry_touched_at is None
            else max(
                0,
                int(
                    (
                        state.entry_touched_at - frozen.decision_time
                    ).total_seconds()
                    // 60
                ),
            )
        )
        self._results.append(
            PathTestResult(
                setup_id=frozen.setup_id,
                hypothesis_key=frozen.hypothesis_key,
                entry_location_id=frozen.entry_location_id,
                entry_path_id=frozen.entry_path_id,
                playbook=frozen.playbook,
                direction=frozen.direction,
                setup_started_at=frozen.setup_started_at,
                sequence_completed_at=frozen.sequence_completed_at,
                decision_time=frozen.decision_time,
                resolved_at=resolved_at,
                outcome=outcome,
                success=outcome == "target",
                entry=frozen.entry,
                invalidation=frozen.invalidation,
                invalidation_source_id=frozen.invalidation_source_id,
                target=frozen.target,
                target_source_id=frozen.target_source_id,
                deadline=frozen.deadline,
                probability=frozen.probability,
                raw_probability=frozen.raw_probability,
                uncertainty=frozen.uncertainty,
                phase=frozen.phase,
                calibration_version=frozen.calibration_version,
                calibration_hash=frozen.calibration_hash,
                mfe_R=state.maximum_favorable_R,
                mae_R=state.maximum_adverse_R,
                elapsed_minutes=elapsed,
                formation_minutes=formation,
                entry_touched=state.entry_touched_at is not None,
                entry_touched_at=state.entry_touched_at,
                time_to_entry_minutes=time_to_entry,
                ambiguous_same_bar=ambiguous,
                decision_hash=frozen.decision_hash,
                protocol_version=frozen.protocol_version,
                protocol_hash=frozen.protocol_hash,
                config_hash=frozen.config_hash,
                code_hash=frozen.code_hash,
            )
        )

    def on_bar(self, bar: Bar) -> None:
        """Advance tests registered at earlier decision clocks."""

        for setup_id, state in tuple(self._open.items()):
            frozen = state.frozen
            if bar.start < frozen.decision_time:
                raise ValueError("path test received a pre-decision bar")
            if (bar.symbol, bar.instrument_id) != (state.symbol, state.instrument_id):
                self._resolve(
                    setup_id,
                    resolved_at=bar.start,
                    outcome="contract_change",
                )
                continue
            if bar.start >= frozen.deadline:
                self._resolve(
                    setup_id,
                    resolved_at=bar.start,
                    outcome="deadline",
                )
                continue
            direction = Direction(frozen.direction)
            entry_was_already_touched = state.entry_touched_at is not None
            if not entry_was_already_touched:
                entry_touched = (
                    bar.low <= frozen.entry
                    if direction is Direction.LONG
                    else bar.high >= frozen.entry
                )
                if not entry_touched:
                    continue
                state.entry_touched_at = bar.end
            # OHLC cannot establish that a favorable excursion occurred after
            # a first-touch limit entry. Do not credit same-bar MFE; subsequent
            # completed bars may contribute normally.
            favorable_points = (
                0.0
                if not entry_was_already_touched
                else (
                    bar.high - frozen.entry
                    if direction is Direction.LONG
                    else frozen.entry - bar.low
                )
            )
            adverse_points = (
                frozen.entry - bar.low
                if direction is Direction.LONG
                else bar.high - frozen.entry
            )
            state.maximum_favorable_R = max(
                state.maximum_favorable_R,
                favorable_points / frozen.risk_points,
            )
            state.maximum_adverse_R = max(
                state.maximum_adverse_R,
                adverse_points / frozen.risk_points,
            )
            if direction is Direction.LONG:
                invalidation_touched = bar.low <= frozen.invalidation
                target_touched = bar.high >= frozen.target
            else:
                invalidation_touched = bar.high >= frozen.invalidation
                target_touched = bar.low <= frozen.target
            if invalidation_touched:
                self._resolve(
                    setup_id,
                    resolved_at=bar.end,
                    outcome="invalidation",
                    ambiguous=target_touched,
                )
            elif target_touched and entry_was_already_touched:
                # OHLC cannot establish that a favorable target came after a
                # same-bar limit entry. Match conservative_entry_bar(): charge
                # an adverse stop, but never credit a same-bar target.
                self._resolve(
                    setup_id,
                    resolved_at=bar.end,
                    outcome="target",
                )

    def close_unresolved(self, resolved_at: pd.Timestamp) -> None:
        resolved_at = aware_timestamp(resolved_at, name="path_test.close")
        for setup_id in tuple(self._open):
            self._resolve(
                setup_id,
                resolved_at=resolved_at,
                outcome="right_censored",
            )


def records_frame(
    records: tuple[Any, ...],
    *,
    record_type: type[Any] | None = None,
) -> pd.DataFrame:
    if records:
        return pd.DataFrame(asdict(record) for record in records)
    columns = () if record_type is None else tuple(field.name for field in fields(record_type))
    return pd.DataFrame(columns=columns)


__all__ = [
    "FrozenPathTest",
    "FrozenPathTestRecorder",
    "FunnelTransition",
    "PathTestResult",
    "PlaybookFunnelRecorder",
    "PrimitivePathCaseResult",
    "PrimitiveProposalEvaluation",
    "ValidationProtocol",
    "ValidationProtocolError",
    "ValidationWindow",
    "evaluate_primitive_implementation_case",
    "finalize_primitive_evaluation",
    "load_validation_protocol",
    "records_frame",
]
