#!/usr/bin/env python3
"""Independent stdlib-only R20 support-prefix materializer.

The tool derives the preregistered synthetic support from frozen integer
market-path inputs.  It has no production-model import and no dependency on
host architecture, Rosetta state, Python machine type, Mach-O slices, or a
dyld shared-cache family.  Those runtime properties belong to the separate
R20 runtime-content closure and cannot alter support rows or commitments.

Importing this module has no filesystem side effects.  A later authorized
parent materializer may invoke the CLI only after publishing its own
``ATTEMPT.json``.  This tool then creates exactly ``support_rows.json`` with
create-exclusive semantics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROTOCOL_ID = (
    "EXP-SMC-3.0.1-001-REGISTERED-SUPPORT-LIFECYCLE-REACHABILITY-R20"
)
STRUCTURE_PROTOCOL_SHA256 = (
    "d94c03b71ffdc099089767fe9dfe57668edfd8252e86db014ce3342115def76d"
)
SYMBOL = "NQH5"
INSTRUMENT_ID = 1
VOLUME_PER_MINUTE = 100
REFERENCE_PRICE_TICKS = 400
TARGET_CANDLE_COUNT = 12
BOUNDARY_ORDINAL = 11
START = datetime(
    2025,
    1,
    5,
    18,
    tzinfo=timezone(timedelta(hours=-5)),
)

# Static negative declaration.  These properties are validated by the R20
# runtime closure, but are forbidden as support, path, identity, clock, or
# prefix-commitment inputs.
ARCHITECTURE_EXCLUDED_FROM_SUPPORT = (
    "physical_host_architecture",
    "rosetta_translation_state",
    "python_runtime_machine",
    "python_executable_macho_slice",
    "native_dependency_architecture",
    "dyld_shared_cache_family",
)

SUPPORT_SEMANTIC_INPUTS = (
    "structure_protocol_sha256",
    "symbol",
    "instrument_id",
    "target_timeframe",
    "direction",
    "swing_side",
    "pivot_start",
    "target_price_ticks",
    "pending_clock",
    "integer_ohlc_template",
    "target_intervals",
    "minute_expansion_formula",
    "volume_per_minute",
)

STAGE_NAMES = (
    "bos_candidate_observed",
    "selector_clock_eligible",
    "semantic_class_resolved",
    "semantic_context_complete",
    "registered_bucket_admitted",
)

EXPECTED_STAGE_ORDINALS = {
    "confirmed_bos": (8, 10, 11, 11, 11),
    "wick_only_no_close": (8, 10, 10, 10, 11),
    "broken_or_opposed": (10, 11, 11, 11, 11),
}

BAR_FIELDS = (
    "start",
    "open_ticks",
    "high_ticks",
    "low_ticks",
    "close_ticks",
    "volume",
    "symbol",
    "instrument_id",
)


@dataclass(frozen=True)
class CandleTicks:
    open_ticks: int
    high_ticks: int
    low_ticks: int
    close_ticks: int

    def __post_init__(self) -> None:
        values = (
            self.open_ticks,
            self.high_ticks,
            self.low_ticks,
            self.close_ticks,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("R20 OHLC ticks must be exact integers")
        if not (
            self.low_ticks
            <= min(self.open_ticks, self.close_ticks)
            <= max(self.open_ticks, self.close_ticks)
            <= self.high_ticks
        ):
            raise ValueError("invalid R20 integer OHLC")


@dataclass(frozen=True)
class TargetDefinition:
    target_id: str
    timeframe: str
    direction: str
    expected_bucket: str
    swing_role: str
    pivot_ordinal: int
    pending_ordinal: int
    swing_side: str

    def __post_init__(self) -> None:
        if not self.target_id.startswith("r20-"):
            raise ValueError("R20 target ID must use the r20 prefix")
        if self.timeframe not in {"1H", "4H"}:
            raise ValueError("R20 target timeframe must be 1H or 4H")
        if self.direction not in {"long", "short"}:
            raise ValueError("R20 target direction must be long or short")
        if self.expected_bucket not in EXPECTED_STAGE_ORDINALS:
            raise ValueError("R20 target bucket is not preregistered")
        if self.swing_side not in {"high", "low"}:
            raise ValueError("R20 target swing side must be high or low")
        if not (
            0 <= self.pivot_ordinal < TARGET_CANDLE_COUNT
            and 0 <= self.pending_ordinal < TARGET_CANDLE_COUNT
        ):
            raise ValueError("R20 target ordinal lies outside the prefix")

    @property
    def boundary_rule(self) -> str:
        if self.expected_bucket == "wick_only_no_close":
            return "ordinal_11_pending_last_attempt_at_equals_asof"
        return "ordinal_11_confirmed_resolved_at_equals_asof"


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
        "r20-4h-long-confirmed",
        "4H",
        "long",
        "confirmed_bos",
        "second_same_side_hh_target",
        6,
        8,
        "high",
    ),
    TargetDefinition(
        "r20-4h-long-wick",
        "4H",
        "long",
        "wick_only_no_close",
        "second_same_side_hh_target",
        6,
        8,
        "high",
    ),
    TargetDefinition(
        "r20-4h-long-opposed",
        "4H",
        "long",
        "broken_or_opposed",
        "mirrored_short_structure_protected_swing",
        8,
        10,
        "high",
    ),
    TargetDefinition(
        "r20-4h-short-confirmed",
        "4H",
        "short",
        "confirmed_bos",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r20-4h-short-wick",
        "4H",
        "short",
        "wick_only_no_close",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r20-4h-short-opposed",
        "4H",
        "short",
        "broken_or_opposed",
        "long_structure_protected_swing",
        8,
        10,
        "low",
    ),
    TargetDefinition(
        "r20-1h-long-opposed",
        "1H",
        "long",
        "broken_or_opposed",
        "mirrored_short_structure_protected_swing",
        8,
        10,
        "high",
    ),
    TargetDefinition(
        "r20-1h-short-confirmed",
        "1H",
        "short",
        "confirmed_bos",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r20-1h-short-wick",
        "1H",
        "short",
        "wick_only_no_close",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r20-1h-short-opposed",
        "1H",
        "short",
        "broken_or_opposed",
        "long_structure_protected_swing",
        8,
        10,
        "low",
    ),
)


def canonical_json_bytes(
    payload: Any,
    *,
    final_lf: bool = False,
) -> bytes:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def production_identity(*parts: object) -> str:
    raw = "|".join(str(value) for value in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def mirror_candle(candle: CandleTicks) -> CandleTicks:
    twice_reference = 2 * REFERENCE_PRICE_TICKS
    return CandleTicks(
        open_ticks=twice_reference - candle.open_ticks,
        high_ticks=twice_reference - candle.low_ticks,
        low_ticks=twice_reference - candle.high_ticks,
        close_ticks=twice_reference - candle.close_ticks,
    )


def template_for_target(
    target: TargetDefinition,
) -> tuple[CandleTicks, ...]:
    if target.expected_bucket == "broken_or_opposed":
        base = (*CANONICAL_LONG_PREFIX, TERMINAL_OPPOSED_SHORT)
        if target.direction == "long":
            return tuple(mirror_candle(candle) for candle in base)
        return tuple(base)
    terminal = (
        TERMINAL_WICK_LONG
        if target.expected_bucket == "wick_only_no_close"
        else TERMINAL_CONFIRMED_LONG
    )
    base = (*CANONICAL_LONG_PREFIX, terminal)
    if target.direction == "short":
        return tuple(mirror_candle(candle) for candle in base)
    return tuple(base)


def target_intervals(
    timeframe: str,
) -> tuple[tuple[datetime, datetime], ...]:
    if timeframe == "1H":
        return tuple(
            (
                START + timedelta(hours=index),
                START + timedelta(hours=index + 1),
            )
            for index in range(TARGET_CANDLE_COUNT)
        )
    if timeframe != "4H":
        raise ValueError("R20 supports only 1H and 4H target schedules")
    intervals: list[tuple[datetime, datetime]] = []
    cursor = START
    for session_index in range(2):
        for duration in (240, 240, 240, 240, 240, 180):
            interval_end = cursor + timedelta(minutes=duration)
            intervals.append((cursor, interval_end))
            cursor = interval_end
        if session_index == 0:
            cursor += timedelta(minutes=60)
    if len(intervals) != TARGET_CANDLE_COUNT:
        raise AssertionError("R20 4H interval count changed")
    return tuple(intervals)


def _segment_values(start: int, end: int, steps: int) -> tuple[int, ...]:
    if steps < 1:
        raise ValueError("R20 minute segment must be nonempty")
    delta = end - start
    return tuple(
        start + (delta * step) // steps
        for step in range(1, steps + 1)
    )


def expand_to_minute_bars(
    candle: CandleTicks,
    *,
    start: datetime,
    end: datetime,
) -> tuple[dict[str, object], ...]:
    elapsed_seconds = int((end - start).total_seconds())
    if elapsed_seconds <= 0 or elapsed_seconds % 60:
        raise ValueError("R20 intervals must contain positive whole minutes")
    minute_count = elapsed_seconds // 60
    if minute_count < 3:
        raise ValueError("R20 candle needs three nonempty segments")
    first_count = minute_count // 3
    second_count = minute_count // 3
    final_count = minute_count - first_count - second_count
    closes = (
        *_segment_values(
            candle.open_ticks,
            candle.high_ticks,
            first_count,
        ),
        *_segment_values(
            candle.high_ticks,
            candle.low_ticks,
            second_count,
        ),
        *_segment_values(
            candle.low_ticks,
            candle.close_ticks,
            final_count,
        ),
    )
    if len(closes) != minute_count:
        raise AssertionError("R20 minute-expansion count changed")

    bars: list[dict[str, object]] = []
    open_ticks = candle.open_ticks
    for minute_index, close_ticks in enumerate(closes):
        bars.append(
            {
                "start": (
                    start + timedelta(minutes=minute_index)
                ).isoformat(),
                "open_ticks": int(open_ticks),
                "high_ticks": int(max(open_ticks, close_ticks)),
                "low_ticks": int(min(open_ticks, close_ticks)),
                "close_ticks": int(close_ticks),
                "volume": int(VOLUME_PER_MINUTE),
                "symbol": SYMBOL,
                "instrument_id": int(INSTRUMENT_ID),
            }
        )
        open_ticks = close_ticks
    return tuple(bars)


def iter_target_prefix(
    target: TargetDefinition,
) -> Iterator[dict[str, object]]:
    template = template_for_target(target)
    intervals = target_intervals(target.timeframe)
    if (
        len(template) != TARGET_CANDLE_COUNT
        or len(intervals) != TARGET_CANDLE_COUNT
    ):
        raise AssertionError("R20 prefix does not contain twelve candles")
    for candle, (start, end) in zip(template, intervals, strict=True):
        yield from expand_to_minute_bars(
            candle,
            start=start,
            end=end,
        )


def commitment_for_bars(
    bars: Iterable[Mapping[str, object]],
) -> str:
    digest = hashlib.sha256()
    exact_fields = frozenset(BAR_FIELDS)
    for bar in bars:
        if frozenset(bar) != exact_fields or len(bar) != len(BAR_FIELDS):
            raise ValueError("R20 bar record differs from frozen schema")
        encoded = canonical_json_bytes(dict(bar))
        digest.update(
            len(encoded).to_bytes(8, byteorder="big", signed=False)
        )
        digest.update(encoded)
    return digest.hexdigest()


def expected_prefix_commitment(target: TargetDefinition) -> str:
    return commitment_for_bars(iter_target_prefix(target))


def build_support_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        template = template_for_target(target)
        intervals = target_intervals(target.timeframe)
        pivot = template[target.pivot_ordinal]
        target_price_ticks = (
            pivot.high_ticks
            if target.swing_side == "high"
            else pivot.low_ticks
        )
        pivot_start = intervals[target.pivot_ordinal][0]
        pending_clock = intervals[target.pending_ordinal][1]
        swing_id = production_identity(
            STRUCTURE_PROTOCOL_SHA256,
            SYMBOL,
            int(INSTRUMENT_ID),
            target.timeframe,
            target.swing_side,
            pivot_start.isoformat(),
            int(target_price_ticks),
        )
        bos_id = production_identity(
            STRUCTURE_PROTOCOL_SHA256,
            target.timeframe,
            target.direction,
            swing_id,
            pending_clock.isoformat(),
        )
        stage_ordinals = EXPECTED_STAGE_ORDINALS[target.expected_bucket]
        stage_clocks = tuple(
            intervals[ordinal][1].isoformat()
            for ordinal in stage_ordinals
        )
        rows.append(
            {
                "target_id": target.target_id,
                "timeframe": target.timeframe,
                "direction": target.direction,
                "exact_bucket_enum": target.expected_bucket,
                "exact_stage_vector": list(STAGE_NAMES),
                "boundary_rule": target.boundary_rule,
                "swing_template_role": target.swing_role,
                "pivot_ordinal": int(target.pivot_ordinal),
                "swing_side": target.swing_side,
                "target_price_ticks": int(target_price_ticks),
                "pivot_start": pivot_start.isoformat(),
                "target_swing_id": swing_id,
                "pending_ordinal": int(target.pending_ordinal),
                "pending_clock": pending_clock.isoformat(),
                "target_bos_id": bos_id,
                "expected_matching_swing_count": 1,
                "expected_matching_bos_count": 1,
                "expected_admission_count": 1,
                "expected_first_stage_ordinals": list(stage_ordinals),
                "expected_first_stage_clocks": list(stage_clocks),
                "expected_boundary_clock": intervals[
                    BOUNDARY_ORDINAL
                ][1].isoformat(),
                "expected_prefix_commitment_sha256": (
                    expected_prefix_commitment(target)
                ),
            }
        )
    return tuple(rows)


def support_rows_bytes() -> bytes:
    return canonical_json_bytes(list(build_support_rows()), final_lf=True)


def support_root_sha256(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    frozen_rows = build_support_rows() if rows is None else tuple(rows)
    return hashlib.sha256(
        canonical_json_bytes(list(frozen_rows))
    ).hexdigest()


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is absent: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be regular and non-symlink")


def _require_directory(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is absent: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_support_rows(output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.name != "support_rows.json":
        raise ValueError("R20 output filename must be support_rows.json")
    output_directory = output.parent
    _require_directory(
        output_directory,
        label="parent materialization directory",
    )
    _require_regular_file(
        output_directory / "ATTEMPT.json",
        label="parent materializer ATTEMPT marker",
    )

    payload = support_rows_bytes()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(output_directory)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize architecture-independent R20 support rows."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Existing marked directory's support_rows.json path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    write_support_rows(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
