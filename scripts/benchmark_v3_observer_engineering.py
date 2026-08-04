#!/usr/bin/env python3
"""Outcome-free synthetic benchmark for causal reader/observer engineering.

This harness never opens project market data.  It exercises only deterministic
synthetic OHLCV bars and reports opaque causal trace commitments plus operational
timing/resource measurements.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.causal import CausalMarketReader, ReaderUpdate
from smc_trader.model import Bar, MarketObservation, to_primitive
from smc_trader.semantic_discovery_runner import observer_from_model_config


MODEL_CONFIG = ROOT / "configs/model_v3_0_exp001_structure_bos_identity.json"
MARKET_TIMEZONE = "America/New_York"
PROJECTED_SOURCE_ROWS = 1_740_541
TRACE_SEED = b"v3-observer-engineering-trace-v1"


def synthetic_session_bars(
    sessions: int,
    *,
    first_trade_date: str,
    pattern: str,
) -> tuple[Bar, ...]:
    """Return a deterministic causal fixture with no project-data dependency."""

    if sessions <= 0:
        raise ValueError("sessions must be positive")
    if pattern not in {"light", "event_heavy"}:
        raise ValueError("unknown synthetic pattern")
    trade_dates = pd.bdate_range(first_trade_date, periods=sessions)
    output: list[Bar] = []
    counter = 0
    for trade_date in trade_dates:
        start = (
            trade_date
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=18)
        ).tz_localize(MARKET_TIMEZONE)
        end = (
            trade_date + pd.Timedelta(hours=17)
        ).tz_localize(MARKET_TIMEZONE)
        for timestamp in pd.date_range(
            start,
            end,
            freq="1min",
            inclusive="left",
        ):
            if pattern == "event_heavy":
                center = (
                    20_000.0
                    + 0.018 * counter
                    + 11.0 * math.sin(counter / 13.0)
                    + 4.0 * math.sin(counter / 3.0)
                )
                body = 0.55 * math.sin(counter / 2.0)
                wick = 0.9 + 0.25 * abs(math.sin(counter / 5.0))
            else:
                center = (
                    20_000.0
                    + 0.012 * counter
                    + 7.0 * math.sin(counter / 47.0)
                )
                body = 0.12 * math.sin(counter / 9.0)
                wick = 0.5
            open_price = center - body
            close = center + body
            output.append(
                Bar(
                    start=timestamp,
                    open=open_price,
                    high=max(open_price, close) + wick,
                    low=min(open_price, close) - wick,
                    close=close,
                    volume=100 + counter % 31,
                    symbol="NQ-SYNTH",
                    instrument_id=1,
                )
            )
            counter += 1
    return tuple(output)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            to_primitive(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _trace_projection(
    ordinal: int,
    update: ReaderUpdate,
    observation: MarketObservation,
) -> dict[str, Any]:
    return {
        "source_ordinal": ordinal,
        "asof": update.asof,
        "completed_1m": update.completed_1m,
        "newly_completed": update.newly_completed,
        "reader_anomalies": update.anomalies,
        "observation": observation,
    }


def replay(
    bars: tuple[Bar, ...],
    *,
    trace: bool,
) -> tuple[str | None, MarketObservation]:
    reader = CausalMarketReader(maximum_history=1024)
    observer = observer_from_model_config(MODEL_CONFIG)
    root = hashlib.sha256(TRACE_SEED).digest()
    final: MarketObservation | None = None
    for ordinal, bar in enumerate(bars):
        update = reader.on_bar(bar)
        final = observer.observe(update)
        if trace:
            payload = _canonical_bytes(
                _trace_projection(ordinal, update, final)
            )
            root = hashlib.sha256(root + payload).digest()
    if final is None:
        raise AssertionError("synthetic replay produced no observation")
    return (root.hex() if trace else None), final


def _darwin_sysctl_int(name: bytes) -> int | None:
    """Read an integer sysctl in this Python process."""

    if platform.system() != "Darwin":
        return None
    try:
        libc = ctypes.CDLL(
            "/usr/lib/libSystem.B.dylib",
            use_errno=True,
        )
        sysctlbyname = libc.sysctlbyname
        sysctlbyname.argtypes = (
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        sysctlbyname.restype = ctypes.c_int
        value = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        result = sysctlbyname(
            name,
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        )
        if result != 0 or size.value != ctypes.sizeof(value):
            return None
        return value.value
    except (AttributeError, OSError):
        return None


def _rosetta_translated_current_process() -> bool | None:
    """Query Rosetta status in this Python process, not a child process."""

    value = _darwin_sysctl_int(b"sysctl.proc_translated")
    return None if value is None else value == 1


def runtime_fingerprint() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "pyarrow", "psutil", "tzdata"):
        module = __import__(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return {
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": list(platform.python_build()),
        "python_compiler": platform.python_compiler(),
        "platform_machine": platform.machine(),
        "platform": platform.platform(),
        "host_arm64_capable": (
            _darwin_sysctl_int(b"hw.optional.arm64") == 1
            if platform.system() == "Darwin"
            else None
        ),
        "rosetta_translated": _rosetta_translated_current_process(),
        "dependencies": versions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    bars = synthetic_session_bars(
        args.sessions,
        first_trade_date=args.first_trade_date,
        pattern=args.pattern,
    )
    fixture_sha256 = hashlib.sha256(
        b"".join(_canonical_bytes(bar) for bar in bars)
    ).hexdigest()
    if args.mode == "trace":
        trace_root, final = replay(bars, trace=True)
        return {
            "artifact": "v3_observer_engineering_trace",
            "format_version": 1,
            "mode": "synthetic_causal_no_economic",
            "bar_count": len(bars),
            "sessions": args.sessions,
            "first_trade_date": args.first_trade_date,
            "pattern": args.pattern,
            "fixture_sha256": fixture_sha256,
            "trace_root_sha256": trace_root,
            "final_observation_sha256": hashlib.sha256(
                _canonical_bytes(final)
            ).hexdigest(),
            "runtime": runtime_fingerprint(),
        }

    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    final_hashes: list[str] = []
    process = psutil.Process()
    for _ in range(args.repeats):
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        _, final = replay(bars, trace=False)
        wall_seconds.append(time.perf_counter() - wall_start)
        cpu_seconds.append(time.process_time() - cpu_start)
        final_hashes.append(
            hashlib.sha256(_canonical_bytes(final)).hexdigest()
        )
    if len(set(final_hashes)) != 1:
        raise AssertionError("repeated benchmark final states differ")
    median_wall = statistics.median(wall_seconds)
    rows_per_second = len(bars) / median_wall
    return {
        "artifact": "v3_observer_engineering_benchmark",
        "format_version": 1,
        "mode": "synthetic_causal_no_economic",
        "bar_count": len(bars),
        "sessions": args.sessions,
        "first_trade_date": args.first_trade_date,
        "pattern": args.pattern,
        "fixture_sha256": fixture_sha256,
        "repeats": args.repeats,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "median_wall_seconds": median_wall,
        "p95_wall_seconds": max(wall_seconds),
        "median_rows_per_second": rows_per_second,
        "projected_1740541_rows_seconds": (
            PROJECTED_SOURCE_ROWS / rows_per_second
        ),
        "rss_bytes_after_run": process.memory_info().rss,
        "final_observation_sha256": final_hashes[0],
        "runtime": runtime_fingerprint(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("trace", "timing"),
        help="trace emits an exact causal commitment; timing omits per-row hashing",
    )
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--first-trade-date", default="2025-01-06")
    parser.add_argument(
        "--pattern",
        choices=("light", "event_heavy"),
        default="light",
    )
    args = parser.parse_args()
    if args.sessions <= 0 or args.repeats <= 0:
        parser.error("sessions and repeats must be positive")
    if args.mode == "trace" and args.repeats != 1:
        parser.error("trace mode requires exactly one repeat")
    return args


if __name__ == "__main__":
    print(
        json.dumps(
            run(parse_args()),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
