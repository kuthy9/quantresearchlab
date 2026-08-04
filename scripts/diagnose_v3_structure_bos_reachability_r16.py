#!/usr/bin/env python3
"""One-shot, outcome-free Structure/BOS lifecycle reachability diagnostic.

This harness is deliberately isolated from production decisions, risk, PnL,
real data and MBO.  It expands a preregistered target-timeframe candle motif
onto the causal one-minute clock, then records the exact production
Structure/BOS funnel at the registered terminal candle.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from smc_trader.causal import CausalMarketReader
from smc_trader.model import (
    BOSLifecycle,
    BOSScope,
    Bar,
    Direction,
    Timeframe,
)
from smc_trader.observation import (
    CausalObserver,
    ExecutionRealityInput,
    ObserverConfig,
)
from smc_trader.semantic_audit import (
    classify_bos_case,
    semantic_case_context_complete,
    semantic_event_id,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = (
    "EXP-SMC-3.0.1-001-REGISTERED-SUPPORT-LIFECYCLE-REACHABILITY-R16"
)
PREREGISTRATION_SHA256 = (
    "68904285f0cced00bb0c578e493f4a15edbda9a123675afbd25a5fce8b532107"
)
STRUCTURE_PROTOCOL = (
    ROOT
    / "configs"
    / "smc_primitives_v3_0_1_structure_bos_audit_closure_r3.json"
)

TICK_SIZE = 0.25
REFERENCE_PRICE_TICKS = 400
SYMBOL = "NQH5"
INSTRUMENT_ID = 1
VOLUME_PER_MINUTE = 100.0
START = pd.Timestamp("2025-01-05T18:00:00-05:00")
TARGET_CANDLE_COUNT = 12
BOUNDARY_ORDINAL = 11
MAXIMUM_HISTORY = 4096

STAGE_NAMES = (
    "bos_candidate_observed",
    "selector_clock_eligible",
    "semantic_class_resolved",
    "semantic_context_complete",
    "registered_bucket_admitted",
)
STAGE_PREDICATES = {
    "bos_candidate_observed": (
        "exact prebound BreakOfStructureState identity is visible"
    ),
    "selector_clock_eligible": (
        "registered boundary and (CONFIRMED resolved_at==asof or "
        "PENDING last_attempt_at==asof)"
    ),
    "semantic_class_resolved": (
        "classify_bos_case(exact_bos, case_clock=asof)==expected_bucket"
    ),
    "semantic_context_complete": (
        "semantic_case_context_complete(exact causal observation/history)"
    ),
    "registered_bucket_admitted": (
        "exact prebound semantic event admitted once at registered boundary"
    ),
}


@dataclass(frozen=True)
class CandleTicks:
    open_ticks: int
    high_ticks: int
    low_ticks: int
    close_ticks: int

    def __post_init__(self) -> None:
        if not (
            self.low_ticks
            <= min(self.open_ticks, self.close_ticks)
            <= max(self.open_ticks, self.close_ticks)
            <= self.high_ticks
        ):
            raise ValueError("invalid registered target-candle OHLC")


@dataclass(frozen=True)
class TargetDefinition:
    target_id: str
    timeframe: Timeframe
    direction: Direction
    expected_bucket: str
    swing_role: str
    pivot_ordinal: int
    pending_ordinal: int
    swing_side: str

    @property
    def boundary_rule(self) -> str:
        if self.expected_bucket == "wick_only_no_close":
            return "ordinal_11_pending_last_attempt_at_equals_asof"
        return "ordinal_11_confirmed_resolved_at_equals_asof"


# This is the exact eleven-candle causal structure sequence already used by
# tests/test_v3_structure_bos.py.  Values are represented in registered ticks.
CANONICAL_LONG_PREFIX = (
    CandleTicks(36, 40, 32, 36),
    CandleTicks(40, 44, 36, 40),
    CandleTicks(46, 52, 40, 46),
    CandleTicks(42, 48, 36, 42),
    CandleTicks(38, 44, 32, 38),
    CandleTicks(42, 48, 36, 42),
    CandleTicks(48, 56, 40, 48),
    CandleTicks(46, 52, 40, 46),
    CandleTicks(42, 48, 36, 42),
    CandleTicks(46, 52, 40, 46),
    CandleTicks(52, 60, 44, 52),
)
TERMINAL_CONFIRMED_LONG = CandleTicks(55, 59, 52, 57)
TERMINAL_WICK_LONG = CandleTicks(52, 58, 50, 55)
TERMINAL_OPPOSED_SHORT = CandleTicks(38, 40, 32, 35)

TARGETS = (
    TargetDefinition(
        "r16-4h-long-confirmed",
        Timeframe.H4,
        Direction.LONG,
        "confirmed_bos",
        "second_same_side_hh_target",
        6,
        8,
        "high",
    ),
    TargetDefinition(
        "r16-4h-long-wick",
        Timeframe.H4,
        Direction.LONG,
        "wick_only_no_close",
        "second_same_side_hh_target",
        6,
        8,
        "high",
    ),
    TargetDefinition(
        "r16-4h-long-opposed",
        Timeframe.H4,
        Direction.LONG,
        "broken_or_opposed",
        "mirrored_short_structure_protected_swing",
        8,
        10,
        "high",
    ),
    TargetDefinition(
        "r16-4h-short-confirmed",
        Timeframe.H4,
        Direction.SHORT,
        "confirmed_bos",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r16-4h-short-wick",
        Timeframe.H4,
        Direction.SHORT,
        "wick_only_no_close",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r16-4h-short-opposed",
        Timeframe.H4,
        Direction.SHORT,
        "broken_or_opposed",
        "long_structure_protected_swing",
        8,
        10,
        "low",
    ),
    TargetDefinition(
        "r16-1h-long-opposed",
        Timeframe.H1,
        Direction.LONG,
        "broken_or_opposed",
        "mirrored_short_structure_protected_swing",
        8,
        10,
        "high",
    ),
    TargetDefinition(
        "r16-1h-short-confirmed",
        Timeframe.H1,
        Direction.SHORT,
        "confirmed_bos",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r16-1h-short-wick",
        Timeframe.H1,
        Direction.SHORT,
        "wick_only_no_close",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r16-1h-short-opposed",
        Timeframe.H1,
        Direction.SHORT,
        "broken_or_opposed",
        "long_structure_protected_swing",
        8,
        10,
        "low",
    ),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(
    payload: Any,
    *,
    final_lf: bool = False,
    sort_keys: bool = True,
) -> bytes:
    raw = json.dumps(
        payload,
        sort_keys=sort_keys,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def production_identity(*parts: object) -> str:
    raw = "|".join(str(value) for value in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def mirror_candle(value: CandleTicks) -> CandleTicks:
    twice_reference = 2 * REFERENCE_PRICE_TICKS
    return CandleTicks(
        open_ticks=twice_reference - value.open_ticks,
        high_ticks=twice_reference - value.low_ticks,
        low_ticks=twice_reference - value.high_ticks,
        close_ticks=twice_reference - value.close_ticks,
    )


def template_for_target(target: TargetDefinition) -> tuple[CandleTicks, ...]:
    if target.expected_bucket == "broken_or_opposed":
        values = (*CANONICAL_LONG_PREFIX, TERMINAL_OPPOSED_SHORT)
        # The canonical path confirms a short BOS opposed to a bull structure.
        return (
            tuple(mirror_candle(value) for value in values)
            if target.direction is Direction.LONG
            else tuple(values)
        )
    terminal = (
        TERMINAL_WICK_LONG
        if target.expected_bucket == "wick_only_no_close"
        else TERMINAL_CONFIRMED_LONG
    )
    values = (*CANONICAL_LONG_PREFIX, terminal)
    return (
        tuple(values)
        if target.direction is Direction.LONG
        else tuple(mirror_candle(value) for value in values)
    )


def target_intervals(
    timeframe: Timeframe,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    if timeframe is Timeframe.H1:
        return tuple(
            (
                START + pd.Timedelta(hours=index),
                START + pd.Timedelta(hours=index + 1),
            )
            for index in range(TARGET_CANDLE_COUNT)
        )
    if timeframe is not Timeframe.H4:
        raise ValueError("R16 supports only the registered 1H/4H targets")
    output: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = START
    for session in range(2):
        for duration in (240, 240, 240, 240, 240, 180):
            end = cursor + pd.Timedelta(minutes=duration)
            output.append((cursor, end))
            cursor = end
        if session == 0:
            cursor += pd.Timedelta(minutes=60)
    if len(output) != TARGET_CANDLE_COUNT:
        raise AssertionError("registered 4H schedule length changed")
    return tuple(output)


def _structure_protocol_hash() -> str:
    return sha256_file(STRUCTURE_PROTOCOL)


def support_rows() -> tuple[dict[str, Any], ...]:
    protocol_hash = _structure_protocol_hash()
    output: list[dict[str, Any]] = []
    for target in TARGETS:
        template = template_for_target(target)
        intervals = target_intervals(target.timeframe)
        pivot = template[target.pivot_ordinal]
        price_ticks = (
            pivot.high_ticks
            if target.swing_side == "high"
            else pivot.low_ticks
        )
        pivot_start = intervals[target.pivot_ordinal][0]
        pending_clock = intervals[target.pending_ordinal][1]
        swing_id = production_identity(
            protocol_hash,
            SYMBOL,
            int(INSTRUMENT_ID),
            target.timeframe.value,
            target.swing_side,
            pivot_start.isoformat(),
            int(price_ticks),
        )
        bos_id = production_identity(
            protocol_hash,
            target.timeframe.value,
            target.direction.value,
            swing_id,
            pending_clock.isoformat(),
        )
        output.append(
            {
                "target_id": target.target_id,
                "timeframe": target.timeframe.value,
                "direction": target.direction.value,
                "exact_bucket_enum": target.expected_bucket,
                "exact_stage_vector": list(STAGE_NAMES),
                "boundary_rule": target.boundary_rule,
                "swing_template_role": target.swing_role,
                "pivot_ordinal": int(target.pivot_ordinal),
                "swing_side": target.swing_side,
                "target_price_ticks": int(price_ticks),
                "pivot_start": pivot_start.isoformat(),
                "target_swing_id": swing_id,
                "pending_ordinal": int(target.pending_ordinal),
                "pending_clock": pending_clock.isoformat(),
                "target_bos_id": bos_id,
                "expected_matching_swing_count": 1,
                "expected_matching_bos_count": 1,
                "expected_admission_count": 1,
            }
        )
    return tuple(output)


def support_root(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(rows))).hexdigest()


def _integer_segment(start: int, end: int, steps: int) -> list[int]:
    if steps < 1:
        raise ValueError("minute segment must be positive")
    delta = end - start
    return [start + (delta * index) // steps for index in range(1, steps + 1)]


def expand_candle_to_bars(
    candle: CandleTicks,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[Bar, ...]:
    minutes = int((end - start).total_seconds() // 60)
    if minutes < 3:
        raise ValueError("registered target candle is too short to expand")
    first = minutes // 3
    second = minutes // 3
    third = minutes - first - second
    closes = [
        *_integer_segment(candle.open_ticks, candle.high_ticks, first),
        *_integer_segment(candle.high_ticks, candle.low_ticks, second),
        *_integer_segment(candle.low_ticks, candle.close_ticks, third),
    ]
    if len(closes) != minutes:
        raise AssertionError("minute expansion length changed")
    output: list[Bar] = []
    open_ticks = candle.open_ticks
    for index, close_ticks in enumerate(closes):
        timestamp = start + pd.Timedelta(minutes=index)
        output.append(
            Bar(
                start=timestamp,
                open=open_ticks * TICK_SIZE,
                high=max(open_ticks, close_ticks) * TICK_SIZE,
                low=min(open_ticks, close_ticks) * TICK_SIZE,
                close=close_ticks * TICK_SIZE,
                volume=VOLUME_PER_MINUTE,
                symbol=SYMBOL,
                instrument_id=INSTRUMENT_ID,
            )
        )
        open_ticks = close_ticks
    if output[-1].end != end:
        raise AssertionError("minute expansion does not end at candle boundary")
    return tuple(output)


def target_bars(target: TargetDefinition) -> tuple[Bar, ...]:
    template = template_for_target(target)
    intervals = target_intervals(target.timeframe)
    output: list[Bar] = []
    for candle, (start, end) in zip(template, intervals, strict=True):
        output.extend(
            expand_candle_to_bars(candle, start=start, end=end)
        )
    return tuple(output)


class FutureReadGuard(Iterator[Bar]):
    def __init__(self, bars: Sequence[Bar]) -> None:
        self._bars = tuple(bars)
        self._index = 0
        self._boundary_processed = False
        self.future_reads = 0

    def __iter__(self) -> "FutureReadGuard":
        return self

    def __next__(self) -> Bar:
        if self._boundary_processed:
            self.future_reads += 1
            raise RuntimeError("bar requested after registered resolution boundary")
        if self._index >= len(self._bars):
            raise StopIteration
        value = self._bars[self._index]
        self._index += 1
        return value

    def mark_boundary_processed(self) -> None:
        self._boundary_processed = True


def _bar_commitment(digest: hashlib._Hash, bar: Bar) -> None:
    payload = {
        "start": bar.start.isoformat(),
        "open_ticks": int(round(bar.open / TICK_SIZE)),
        "high_ticks": int(round(bar.high / TICK_SIZE)),
        "low_ticks": int(round(bar.low / TICK_SIZE)),
        "close_ticks": int(round(bar.close / TICK_SIZE)),
        "volume": int(bar.volume),
        "symbol": bar.symbol,
        "instrument_id": int(bar.instrument_id),
    }
    raw = canonical_json_bytes(payload)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def _new_stages() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "reached": False,
            "ordinal": None,
            "clock": None,
            "production_predicate": STAGE_PREDICATES[name],
        }
        for name in STAGE_NAMES
    ]


def _reach_stage(
    stages: list[dict[str, Any]],
    name: str,
    *,
    ordinal: int,
    clock: pd.Timestamp,
) -> None:
    stage = next(value for value in stages if value["name"] == name)
    if not stage["reached"]:
        stage["reached"] = True
        stage["ordinal"] = int(ordinal)
        stage["clock"] = clock.isoformat()


def _ticks(value: float) -> int:
    return int(round(float(value) / TICK_SIZE))


def _assert_emitted_candle(
    emitted,
    expected: CandleTicks,
) -> None:
    actual = (
        _ticks(emitted.open),
        _ticks(emitted.high),
        _ticks(emitted.low),
        _ticks(emitted.close),
    )
    registered = (
        expected.open_ticks,
        expected.high_ticks,
        expected.low_ticks,
        expected.close_ticks,
    )
    if actual != registered:
        raise AssertionError(
            f"target candle expansion changed: actual={actual}, expected={registered}"
        )


def run_target(
    target: TargetDefinition,
    registered_support: Mapping[str, Any],
) -> dict[str, Any]:
    stages = _new_stages()
    template = template_for_target(target)
    bars = target_bars(target)
    guard = FutureReadGuard(bars)
    reader = CausalMarketReader(maximum_history=MAXIMUM_HISTORY)
    observer = CausalObserver(
        ObserverConfig(
            tick_size=TICK_SIZE,
            structure_protocol=str(STRUCTURE_PROTOCOL),
        )
    )
    commitment = hashlib.sha256()
    target_ordinal = -1
    boundary_clock: pd.Timestamp | None = None
    matching_swing_count = 0
    matching_bos_count = 0
    observed_bucket: str | None = None
    admission_count = 0
    rejection_reason: str | None = None

    for bar in guard:
        _bar_commitment(commitment, bar)
        update = reader.on_bar(bar)
        observation = observer.observe(
            update,
            ExecutionRealityInput(
                spread_points=TICK_SIZE,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                deadline=update.asof + pd.Timedelta(days=1),
                size_available=1.0,
                source="r16_synthetic_reachability",
            ),
        )
        completed = update.newly_completed[target.timeframe]
        if not completed:
            continue
        if len(completed) != 1:
            raise AssertionError("target timeframe emitted multiple candles")
        target_ordinal += 1
        _assert_emitted_candle(completed[0], template[target_ordinal])
        frame = observation.frame(target.timeframe)
        swings = [
            item
            for item in frame.swings
            if item.swing_id == registered_support["target_swing_id"]
        ]
        bos_values = [
            item
            for item in frame.structure_breaks
            if item.bos_id == registered_support["target_bos_id"]
        ]
        if len(bos_values) == 1:
            _reach_stage(
                stages,
                "bos_candidate_observed",
                ordinal=target_ordinal,
                clock=observation.asof,
            )
        if target_ordinal != BOUNDARY_ORDINAL:
            continue

        guard.mark_boundary_processed()
        boundary_clock = observation.asof
        matching_swing_count = len(swings)
        matching_bos_count = len(bos_values)
        if matching_swing_count != 1:
            rejection_reason = "prebound_swing_identity_count_not_one"
        elif matching_bos_count != 1:
            rejection_reason = "prebound_bos_identity_count_not_one"
        else:
            focus = bos_values[0]
            eligible = (
                focus.lifecycle is BOSLifecycle.CONFIRMED
                and focus.resolved_at == observation.asof
            ) or (
                focus.lifecycle is BOSLifecycle.PENDING
                and focus.last_attempt_at == observation.asof
            )
            if eligible:
                _reach_stage(
                    stages,
                    "selector_clock_eligible",
                    ordinal=target_ordinal,
                    clock=observation.asof,
                )
                observed_bucket = classify_bos_case(
                    focus,
                    case_clock=observation.asof,
                )
            else:
                rejection_reason = "selector_clock_not_eligible_at_boundary"
            if observed_bucket == target.expected_bucket:
                _reach_stage(
                    stages,
                    "semantic_class_resolved",
                    ordinal=target_ordinal,
                    clock=observation.asof,
                )
            elif rejection_reason is None:
                rejection_reason = "semantic_class_did_not_match_registered_bucket"
            context_complete = semantic_case_context_complete(
                observation,
                update.histories,
                focus,
            )
            if context_complete:
                _reach_stage(
                    stages,
                    "semantic_context_complete",
                    ordinal=target_ordinal,
                    clock=observation.asof,
                )
            elif rejection_reason is None:
                rejection_reason = "semantic_context_incomplete"
            expected_event_id = "|".join(
                (
                    registered_support["target_bos_id"],
                    target.expected_bucket,
                    observation.asof.isoformat(),
                )
            )
            actual_event_id = (
                None
                if observed_bucket is None
                else semantic_event_id(
                    focus,
                    case_class=observed_bucket,
                    case_clock=observation.asof,
                )
            )
            if (
                eligible
                and observed_bucket == target.expected_bucket
                and context_complete
                and actual_event_id == expected_event_id
            ):
                admission_count += 1
                _reach_stage(
                    stages,
                    "registered_bucket_admitted",
                    ordinal=target_ordinal,
                    clock=observation.asof,
                )
            elif rejection_reason is None:
                rejection_reason = "prebound_semantic_event_not_admitted"
        break

    if boundary_clock is None:
        rejection_reason = (
            rejection_reason or "not_reached_by_preregistered_horizon"
        )
    stage_ordinals = [
        int(stage["ordinal"])
        for stage in stages
        if stage["reached"]
    ]
    if stage_ordinals != sorted(stage_ordinals):
        rejection_reason = "stage_ordinals_not_nondecreasing"
    passed = bool(
        boundary_clock is not None
        and matching_swing_count == 1
        and matching_bos_count == 1
        and admission_count == 1
        and guard.future_reads == 0
        and all(stage["reached"] for stage in stages)
        and rejection_reason is None
    )
    return {
        "target_id": target.target_id,
        "timeframe": target.timeframe.value,
        "direction": target.direction.value,
        "expected_bucket": target.expected_bucket,
        "target_swing_id": registered_support["target_swing_id"],
        "target_bos_id": registered_support["target_bos_id"],
        "matching_swing_count": int(matching_swing_count),
        "matching_bos_count": int(matching_bos_count),
        "boundary_ordinal": (
            None if boundary_clock is None else int(BOUNDARY_ORDINAL)
        ),
        "boundary_clock": (
            None if boundary_clock is None else boundary_clock.isoformat()
        ),
        "stages": stages,
        "observed_bucket": observed_bucket,
        "admission_count": int(admission_count),
        "future_bars_read_for_resolution": int(guard.future_reads),
        "prefix_commitment_sha256": commitment.hexdigest(),
        "rejection_reason": rejection_reason,
        "target_status": "pass" if passed else "fail",
    }


def semantic_summary(
    *,
    implementation_manifest_sha256: str,
    registered_support: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results = [
        run_target(target, support)
        for target, support in zip(TARGETS, registered_support, strict=True)
    ]
    admission_count = sum(int(value["admission_count"]) for value in results)
    duplicate_count = sum(
        max(0, int(value["admission_count"]) - 1) for value in results
    )
    future_reads = sum(
        int(value["future_bars_read_for_resolution"]) for value in results
    )
    status = (
        "pass"
        if (
            len(results) == 10
            and admission_count == 10
            and duplicate_count == 0
            and future_reads == 0
            and all(value["target_status"] == "pass" for value in results)
        )
        else "fail"
    )
    return {
        "format_version": 1,
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "support_root_sha256": support_root(registered_support),
        "target_count": int(len(results)),
        "admission_count": int(admission_count),
        "extra_admission_count": 0,
        "duplicate_admission_count": int(duplicate_count),
        "future_bars_read_for_resolution": int(future_reads),
        "targets": results,
        "semantic_status": status,
    }


def semantic_summary_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(payload, final_lf=True, sort_keys=False)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale atomic temporary exists: {temporary}")
    _exclusive_bytes(temporary, payload)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def create_attempt_marker(
    *,
    output_root: Path,
    run_id: str,
    mode: str,
    manifest_sha256: str,
    registered_support_root: str,
) -> Path:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    marker = output_root / "ATTEMPT.json"
    _exclusive_bytes(
        marker,
        canonical_json_bytes(
            {
                "format_version": 1,
                "run_id": run_id,
                "mode": mode,
                "manifest_sha256": manifest_sha256,
                "support_root_sha256": registered_support_root,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            final_lf=True,
        ),
    )
    return marker


def _load_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("implementation manifest SHA-256 changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("preregistration_sha256")
        != PREREGISTRATION_SHA256
    ):
        raise ValueError("implementation manifest protocol binding changed")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("implementation manifest bindings are absent")
    if bindings.get("harness_sha256") != sha256_file(Path(__file__)):
        raise ValueError("implementation harness hash changed")
    registered_files = {
        "test_sha256": bindings.get("test_path"),
        "structure_protocol_sha256": bindings.get("structure_protocol_path"),
        "structure_code_sha256": bindings.get("structure_code_path"),
        "semantic_audit_code_sha256": bindings.get(
            "semantic_audit_code_path"
        ),
        "causal_reader_code_sha256": bindings.get(
            "causal_reader_code_path"
        ),
        "observer_code_sha256": bindings.get("observer_code_path"),
        "model_code_sha256": bindings.get("model_code_path"),
        "market_clock_code_sha256": bindings.get(
            "market_clock_code_path"
        ),
        "canonical_template_source_sha256": bindings.get(
            "canonical_template_source_path"
        ),
    }
    for hash_field, relative in registered_files.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"implementation binding path absent: {hash_field}")
        path_value = ROOT / relative
        if bindings.get(hash_field) != sha256_file(path_value):
            raise ValueError(f"implementation binding changed: {hash_field}")
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("python_executable") != sys.executable
        or runtime.get("python_executable_sha256")
        != sha256_file(sys.executable)
    ):
        raise ValueError("implementation runtime identity changed")
    rows = payload.get("support_rows")
    if not isinstance(rows, list) or len(rows) != len(TARGETS):
        raise ValueError("implementation manifest support payload is invalid")
    if payload.get("support_root_sha256") != support_root(rows):
        raise ValueError("implementation manifest support root changed")
    return payload, actual


def _run_identity(manifest: Mapping[str, Any], mode: str) -> tuple[str, Path]:
    runs = manifest.get("runs")
    if not isinstance(runs, dict) or mode not in runs:
        raise ValueError("implementation manifest run identity is absent")
    value = runs[mode]
    return str(value["run_id"]), ROOT / str(value["output_root"])


def _diagnostic_success_precondition(manifest: Mapping[str, Any]) -> None:
    _, diagnostic_root = _run_identity(manifest, "diagnostic")
    result_path = diagnostic_root / "RESULT.json"
    summary_path = diagnostic_root / "semantic_summary.json"
    if not result_path.is_file() or not summary_path.is_file():
        raise RuntimeError("verification requires completed diagnostic artifacts")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "success"
        or result.get("semantic_summary_sha256")
        != sha256_file(summary_path)
    ):
        raise RuntimeError("verification diagnostic precondition failed")


def execute_run(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    mode: str,
) -> int:
    if mode not in {"diagnostic", "verification"}:
        raise ValueError("R16 mode must be diagnostic or verification")
    manifest, manifest_sha256 = _load_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    if mode == "verification":
        _diagnostic_success_precondition(manifest)
    run_id, output_root = _run_identity(manifest, mode)
    rows = tuple(manifest["support_rows"])
    root_hash = support_root(rows)

    # This marker is intentionally the first mutating action and occurs before
    # target_bars(), run_target(), reader.on_bar(), or observer.observe().
    create_attempt_marker(
        output_root=output_root,
        run_id=run_id,
        mode=mode,
        manifest_sha256=manifest_sha256,
        registered_support_root=root_hash,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        derived_rows = support_rows()
        if list(derived_rows) != list(rows):
            raise ValueError(
                "prebound support differs from the registered derivation"
            )
        summary = semantic_summary(
            implementation_manifest_sha256=manifest_sha256,
            registered_support=derived_rows,
        )
        summary_raw = semantic_summary_bytes(summary)
        summary_path = output_root / "semantic_summary.json"
        _atomic_bytes(summary_path, summary_raw)
        bytes_equal = None
        terminal_success = summary["semantic_status"] == "pass"
        if mode == "verification":
            _, diagnostic_root = _run_identity(manifest, "diagnostic")
            diagnostic_raw = (
                diagnostic_root / "semantic_summary.json"
            ).read_bytes()
            bytes_equal = diagnostic_raw == summary_raw
            terminal_success = bool(terminal_success and bytes_equal)
        envelope = {
            "format_version": 1,
            "run_id": run_id,
            "mode": mode,
            "output_root": str(output_root.relative_to(ROOT)),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "semantic_summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
            "diagnostic_summary_bytes_equal": bytes_equal,
        }
        _atomic_bytes(
            output_root / "execution_envelope.json",
            canonical_json_bytes(envelope, final_lf=True),
        )
        result = {
            "format_version": 1,
            "run_id": run_id,
            "mode": mode,
            "status": "success" if terminal_success else "fail",
            "manifest_sha256": manifest_sha256,
            "support_root_sha256": root_hash,
            "semantic_summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
            "diagnostic_summary_bytes_equal": bytes_equal,
        }
        _atomic_bytes(
            output_root / "RESULT.json",
            canonical_json_bytes(result, final_lf=True),
        )
        return 0 if terminal_success else 2
    except BaseException as exc:
        envelope = {
            "format_version": 1,
            "run_id": run_id,
            "mode": mode,
            "output_root": str(output_root.relative_to(ROOT)),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_bytes(
            output_root / "execution_envelope.json",
            canonical_json_bytes(envelope, final_lf=True),
        )
        result = {
            "format_version": 1,
            "run_id": run_id,
            "mode": mode,
            "status": "fail",
            "manifest_sha256": manifest_sha256,
            "support_root_sha256": root_hash,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_bytes(
            output_root / "RESULT.json",
            canonical_json_bytes(result, final_lf=True),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("diagnostic", "verification"),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return execute_run(
        manifest_path=Path(args.manifest),
        expected_manifest_sha256=args.expected_manifest_sha256,
        mode=args.mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
