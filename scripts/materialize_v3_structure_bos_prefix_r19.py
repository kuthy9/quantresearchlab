#!/usr/bin/env python3
"""Stdlib-only R19 support and outcome-free prefix materializer.

This Stage-A tool contains no production imports and has no market-model
inputs.  It deterministically derives the preregistered R19 support rows from
the frozen integer OHLC templates, target-timeframe schedules, identity
formula and one-minute prefix commitment.

Importing this module has no filesystem side effects.  The CLI is reserved
for a later, separately authorized Stage-B total materializer.  It writes
only ``support_rows.json`` into an existing non-symlink directory that
already contains a regular non-symlink ``ATTEMPT.json`` marker.
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
    "EXP-SMC-3.0.1-001-REGISTERED-SUPPORT-LIFECYCLE-REACHABILITY-R19"
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
    0,
    0,
    tzinfo=timezone(timedelta(hours=-5)),
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
    """One preregistered target-timeframe candle in integer ticks."""

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
            raise TypeError("registered OHLC ticks must be integers")
        if not (
            self.low_ticks
            <= min(self.open_ticks, self.close_ticks)
            <= max(self.open_ticks, self.close_ticks)
            <= self.high_ticks
        ):
            raise ValueError("invalid registered target-candle OHLC")


@dataclass(frozen=True)
class TargetDefinition:
    """Outcome-independent identity and support dimension for one target."""

    target_id: str
    timeframe: str
    direction: str
    expected_bucket: str
    swing_role: str
    pivot_ordinal: int
    pending_ordinal: int
    swing_side: str

    def __post_init__(self) -> None:
        if self.timeframe not in {"1H", "4H"}:
            raise ValueError("R19 target timeframe must be 1H or 4H")
        if self.direction not in {"long", "short"}:
            raise ValueError("R19 target direction must be long or short")
        if self.expected_bucket not in EXPECTED_STAGE_ORDINALS:
            raise ValueError("R19 target bucket is not registered")
        if self.swing_side not in {"high", "low"}:
            raise ValueError("R19 swing side must be high or low")
        if not (
            0 <= self.pivot_ordinal < TARGET_CANDLE_COUNT
            and 0 <= self.pending_ordinal < TARGET_CANDLE_COUNT
        ):
            raise ValueError("R19 target ordinal is outside the frozen prefix")

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
        "r19-4h-long-confirmed",
        "4H",
        "long",
        "confirmed_bos",
        "second_same_side_hh_target",
        6,
        8,
        "high",
    ),
    TargetDefinition(
        "r19-4h-long-wick",
        "4H",
        "long",
        "wick_only_no_close",
        "second_same_side_hh_target",
        6,
        8,
        "high",
    ),
    TargetDefinition(
        "r19-4h-long-opposed",
        "4H",
        "long",
        "broken_or_opposed",
        "mirrored_short_structure_protected_swing",
        8,
        10,
        "high",
    ),
    TargetDefinition(
        "r19-4h-short-confirmed",
        "4H",
        "short",
        "confirmed_bos",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r19-4h-short-wick",
        "4H",
        "short",
        "wick_only_no_close",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r19-4h-short-opposed",
        "4H",
        "short",
        "broken_or_opposed",
        "long_structure_protected_swing",
        8,
        10,
        "low",
    ),
    TargetDefinition(
        "r19-1h-long-opposed",
        "1H",
        "long",
        "broken_or_opposed",
        "mirrored_short_structure_protected_swing",
        8,
        10,
        "high",
    ),
    TargetDefinition(
        "r19-1h-short-confirmed",
        "1H",
        "short",
        "confirmed_bos",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r19-1h-short-wick",
        "1H",
        "short",
        "wick_only_no_close",
        "mirrored_second_same_side_ll_target",
        6,
        8,
        "low",
    ),
    TargetDefinition(
        "r19-1h-short-opposed",
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
    """Return the single registered compact, sorted-key JSON encoding."""

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def production_identity(*parts: object) -> str:
    """Reproduce the frozen production ``_identity`` byte formula."""

    raw = "|".join(str(value) for value in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def mirror_candle(candle: CandleTicks) -> CandleTicks:
    """Return the literal tick-exact long/short price mirror."""

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
    """Return the frozen twelve-candle path without observing model output."""

    if target.expected_bucket == "broken_or_opposed":
        values = (*CANONICAL_LONG_PREFIX, TERMINAL_OPPOSED_SHORT)
        if target.direction == "long":
            return tuple(mirror_candle(value) for value in values)
        return tuple(values)
    terminal = (
        TERMINAL_WICK_LONG
        if target.expected_bucket == "wick_only_no_close"
        else TERMINAL_CONFIRMED_LONG
    )
    values = (*CANONICAL_LONG_PREFIX, terminal)
    if target.direction == "short":
        return tuple(mirror_candle(value) for value in values)
    return tuple(values)


def target_intervals(
    timeframe: str,
) -> tuple[tuple[datetime, datetime], ...]:
    """Return the registered target-timeframe schedule."""

    if timeframe == "1H":
        return tuple(
            (
                START + timedelta(hours=index),
                START + timedelta(hours=index + 1),
            )
            for index in range(TARGET_CANDLE_COUNT)
        )
    if timeframe != "4H":
        raise ValueError("R19 supports only the registered 1H/4H targets")
    output: list[tuple[datetime, datetime]] = []
    cursor = START
    for session in range(2):
        for duration_minutes in (240, 240, 240, 240, 240, 180):
            end = cursor + timedelta(minutes=duration_minutes)
            output.append((cursor, end))
            cursor = end
        if session == 0:
            cursor += timedelta(minutes=60)
    if len(output) != TARGET_CANDLE_COUNT:
        raise AssertionError("registered 4H schedule length changed")
    return tuple(output)


def _integer_segment(start: int, end: int, steps: int) -> tuple[int, ...]:
    if steps < 1:
        raise ValueError("minute segment must contain at least one step")
    delta = end - start
    return tuple(
        start + (delta * index) // steps
        for index in range(1, steps + 1)
    )


def expand_candle_to_minute_bars(
    candle: CandleTicks,
    *,
    start: datetime,
    end: datetime,
) -> tuple[dict[str, object], ...]:
    """Expand one frozen candle into deterministic integer-tick 1m bars."""

    elapsed_seconds = int((end - start).total_seconds())
    if elapsed_seconds <= 0 or elapsed_seconds % 60:
        raise ValueError("registered target interval must be positive whole minutes")
    minutes = elapsed_seconds // 60
    if minutes < 3:
        raise ValueError("registered target candle is too short to expand")
    first = minutes // 3
    second = minutes // 3
    third = minutes - first - second
    closes = (
        *_integer_segment(candle.open_ticks, candle.high_ticks, first),
        *_integer_segment(candle.high_ticks, candle.low_ticks, second),
        *_integer_segment(candle.low_ticks, candle.close_ticks, third),
    )
    if len(closes) != minutes:
        raise AssertionError("minute expansion length changed")

    output: list[dict[str, object]] = []
    open_ticks = candle.open_ticks
    for index, close_ticks in enumerate(closes):
        output.append(
            {
                "start": (start + timedelta(minutes=index)).isoformat(),
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
    return tuple(output)


def iter_prefix_bars(
    target: TargetDefinition,
) -> Iterator[dict[str, object]]:
    """Yield exactly the causal 1m prefix through boundary ordinal 11."""

    template = template_for_target(target)
    intervals = target_intervals(target.timeframe)
    if len(template) != TARGET_CANDLE_COUNT or len(intervals) != TARGET_CANDLE_COUNT:
        raise AssertionError("registered target prefix length changed")
    for candle, (start, end) in zip(template, intervals, strict=True):
        yield from expand_candle_to_minute_bars(
            candle,
            start=start,
            end=end,
        )


def commitment_for_bars(
    bars: Iterable[Mapping[str, object]],
) -> str:
    """Hash exact canonical bar records with registered length framing."""

    digest = hashlib.sha256()
    expected_fields = frozenset(BAR_FIELDS)
    for bar in bars:
        if frozenset(bar) != expected_fields or len(bar) != len(BAR_FIELDS):
            raise ValueError("prefix bar fields differ from the frozen schema")
        raw = canonical_json_bytes(dict(bar))
        digest.update(len(raw).to_bytes(8, "big", signed=False))
        digest.update(raw)
    return digest.hexdigest()


def prefix_commitment(target: TargetDefinition) -> str:
    """Return the outcome-free commitment through the registered boundary."""

    return commitment_for_bars(iter_prefix_bars(target))


def build_support_rows() -> tuple[dict[str, Any], ...]:
    """Derive the exact ordered R19 support rows from frozen inputs only."""

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
        target_swing_id = production_identity(
            STRUCTURE_PROTOCOL_SHA256,
            SYMBOL,
            int(INSTRUMENT_ID),
            target.timeframe,
            target.swing_side,
            pivot_start.isoformat(),
            int(price_ticks),
        )
        target_bos_id = production_identity(
            STRUCTURE_PROTOCOL_SHA256,
            target.timeframe,
            target.direction,
            target_swing_id,
            pending_clock.isoformat(),
        )
        first_stage_ordinals = EXPECTED_STAGE_ORDINALS[
            target.expected_bucket
        ]
        first_stage_clocks = tuple(
            intervals[ordinal][1].isoformat()
            for ordinal in first_stage_ordinals
        )
        output.append(
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
                "target_price_ticks": int(price_ticks),
                "pivot_start": pivot_start.isoformat(),
                "target_swing_id": target_swing_id,
                "pending_ordinal": int(target.pending_ordinal),
                "pending_clock": pending_clock.isoformat(),
                "target_bos_id": target_bos_id,
                "expected_matching_swing_count": 1,
                "expected_matching_bos_count": 1,
                "expected_admission_count": 1,
                "expected_first_stage_ordinals": list(
                    first_stage_ordinals
                ),
                "expected_first_stage_clocks": list(first_stage_clocks),
                "expected_boundary_clock": intervals[
                    BOUNDARY_ORDINAL
                ][1].isoformat(),
                "expected_prefix_commitment_sha256": prefix_commitment(
                    target
                ),
            }
        )
    return tuple(output)


def support_rows_bytes() -> bytes:
    """Return the canonical materialized support file including final LF."""

    return canonical_json_bytes(list(build_support_rows()), final_lf=True)


def support_root_sha256(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return the no-final-LF canonical support-root digest."""

    values = build_support_rows() if rows is None else tuple(rows)
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _require_regular_non_symlink(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is absent: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _require_directory_non_symlink(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is absent: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory, not a symlink")


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
    """Create the one allowed output after a parent attempt marker exists."""

    output = Path(output_path)
    if output.name != "support_rows.json":
        raise ValueError("R19 prefix tool output must be support_rows.json")
    parent = output.parent
    _require_directory_non_symlink(
        parent,
        label="materialization output directory",
    )
    _require_regular_non_symlink(
        parent / "ATTEMPT.json",
        label="parent materializer attempt marker",
    )

    payload = support_rows_bytes()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(parent)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize frozen outcome-free R19 support rows."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Existing marked materialization directory's "
            "support_rows.json path"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    write_support_rows(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
