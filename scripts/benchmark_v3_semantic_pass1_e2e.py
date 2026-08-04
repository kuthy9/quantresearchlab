#!/usr/bin/env python3
"""Synthetic-only full Pass1 engineering benchmark.

The harness generates its own deterministic Parquet source and then delegates
the complete scan to ``SemanticDiscoveryRunner``.  It never opens project
market data and grants no Pass2, future, PnL, decision, risk, or MBO authority.

Preparation is deliberately outside the timed execution boundary because a
production Pass1 starts from an already materialized, hash-frozen source.  The
timed ``run`` command includes contract verification, full-file source hashing,
timestamp-only counting, batch decoding, causal replay, both rolling
commitments, selection, checkpoints/fsync, progress/resource monitoring,
terminal publication, and source-free terminal reconciliation.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import sha256_file
from smc_trader import market_clock as registered_market_clock
from smc_trader.market_clock import (
    CLOCK_END,
    CLOCK_START,
    is_registered_trading_minute,
)
from smc_trader.semantic_discovery_runner import (
    DiscoveryCheckpointStore,
    FrozenParquetSourceIdentity,
    ParquetBatchSource,
    SemanticDiscoveryRunner,
    runtime_environment,
)


ARTIFACT_ROOT = ROOT / "artifacts" / "synthetic_semantic_pass1_e2e"
ACTIVE_EXPERIMENT_ID = (
    "EXP-SMC-3.0.1-001-STRUCTURE-BOS-AUDIT-CLOSURE-R3"
)
MODEL_CONFIG = (
    ROOT
    / "configs/model_v3_0_1_exp001_structure_bos_identity_r3.json"
)
PRIMITIVE_PROTOCOL = (
    ROOT
    / "configs/smc_primitives_v3_0_1_structure_bos_audit_closure_r3.json"
)
MARKET_TIMEZONE = "America/New_York"
DEFAULT_PROJECTED_SOURCE_ROWS = 1_740_541
ALLOWED_PATTERNS = {"light", "event_heavy"}
HEX64 = re.compile(r"[0-9a-f]{64}")
SHORT_OPEN_GAP_SESSION_MODULUS = 7
SHORT_OPEN_GAP_SESSION_REMAINDER = 1
SHORT_OPEN_GAP_START_INDEX = 421
LONG_OPEN_GAP_SESSION_MODULUS = 11
LONG_OPEN_GAP_SESSION_REMAINDER = 2
LONG_OPEN_GAP_START_INDEX = 700
LONG_OPEN_GAP_MINUTES = 7
CONFIRMATORY_GATES = {
    "wall_seconds_strictly_less_than": 3_600.0,
    "rss_bytes_strictly_less_than": 4 * 1024**3,
    "disk_free_bytes_at_least": 10 * 1024**3,
    "resource_check_source_rows": 100,
    "checkpoint_stall_timeout_seconds": 300.0,
    "source_rows": DEFAULT_PROJECTED_SOURCE_ROWS,
    "maximum_release_window_seconds": 24 * 60 * 60,
}
IMPLEMENTATION_FILES = {
    "artifact_stream_sha256": ROOT / "smc_trader/artifact_stream.py",
    "model_sha256": ROOT / "smc_trader/model.py",
    "semantic_discovery_runner_sha256": (
        ROOT / "smc_trader/semantic_discovery_runner.py"
    ),
    "semantic_audit_sha256": ROOT / "smc_trader/semantic_audit.py",
    "io_sha256": ROOT / "smc_trader/io.py",
    "causal_sha256": ROOT / "smc_trader/causal.py",
    "market_clock_sha256": ROOT / "smc_trader/market_clock.py",
    "observation_sha256": ROOT / "smc_trader/observation.py",
    "structure_sha256": ROOT / "smc_trader/structure.py",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _bootstrap_sha256_file(path: str | Path) -> str:
    """Independent integrity root used before artifact_stream is trusted."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(
            f"bootstrap hash target is not a regular file: {candidate}"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    require_existing_parent: bool = False,
) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-once artifact already exists: {path}")
    if require_existing_parent:
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise FileNotFoundError(
                "create-once artifact requires its pre-existing parent: "
                f"{path.parent}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(dict(payload)))
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a trusted regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _guard_fixture_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    allowed = ARTIFACT_ROOT.resolve()
    real_data = (ROOT / "data").resolve()
    if not _is_relative_to(candidate, allowed):
        raise PermissionError(
            "synthetic E2E artifacts must stay below "
            f"{allowed}"
        )
    if _is_relative_to(candidate, real_data):
        raise PermissionError("project data/ access is forbidden")
    return candidate


def _guard_fixture_member(path: str | Path, fixture_root: Path) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise PermissionError("synthetic fixture member may not be a symlink")
    candidate = supplied.resolve()
    if not _is_relative_to(candidate, fixture_root.resolve()):
        raise PermissionError("synthetic fixture member escaped its root")
    if _is_relative_to(candidate, (ROOT / "data").resolve()):
        raise PermissionError("project data/ access is forbidden")
    return candidate


def _implementation_hashes() -> dict[str, str]:
    return {
        name: _bootstrap_sha256_file(path)
        for name, path in IMPLEMENTATION_FILES.items()
    }


def _runtime_fingerprint() -> dict[str, Any]:
    import _hashlib
    import numpy
    import pandas
    import pyarrow
    import psutil

    translated = _darwin_sysctl_int(b"sysctl.proc_translated")
    package_roots = {
        "numpy": Path(numpy.__file__).resolve().parent,
        "pandas": Path(pandas.__file__).resolve().parent,
        "pyarrow": Path(pyarrow.__file__).resolve().parent,
        "psutil": Path(psutil.__file__).resolve().parent,
    }
    native_files: set[Path] = {
        Path(sys.executable).resolve(),
        Path(_hashlib.__file__).resolve(),
    }
    for root in package_roots.values():
        native_files.update(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and (
                path.suffix in {".so", ".dylib"}
                or ".so." in path.name
            )
        )
    return {
        "python_executable": sys.executable,
        "python_executable_resolved": str(
            Path(sys.executable).resolve()
        ),
        "python_executable_sha256": _bootstrap_sha256_file(
            Path(sys.executable).resolve()
        ),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": list(platform.python_build()),
        "python_compiler": platform.python_compiler(),
        "platform_machine": platform.machine(),
        "platform": platform.platform(),
        "host_arm64_capable": (
            _darwin_sysctl_int(b"hw.optional.arm64") == 1
            if platform.system() == "Darwin"
            else None
        ),
        "rosetta_translated": (
            None if translated is None else translated == 1
        ),
        "dependencies": {
            **runtime_environment(),
            "numpy_version": str(numpy.__version__),
        },
        "native_binary_sha256": {
            str(path): _bootstrap_sha256_file(path)
            for path in sorted(native_files)
        },
    }


def _darwin_sysctl_int(name: bytes) -> int | None:
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
        if (
            sysctlbyname(
                name,
                ctypes.byref(value),
                ctypes.byref(size),
                None,
                0,
            )
            != 0
            or size.value != ctypes.sizeof(value)
        ):
            return None
        return int(value.value)
    except (AttributeError, OSError):
        return None


def _session_frames(
    *,
    source_rows: int,
    first_trade_date: str,
    pattern: str,
) -> Iterator[pd.DataFrame]:
    """Yield deterministic synthetic sessions until ``source_rows`` exist."""

    if source_rows <= 0:
        raise ValueError("source rows must be positive")
    if pattern not in ALLOWED_PATTERNS:
        raise ValueError("unknown synthetic pattern")
    remaining = int(source_rows)
    emitted = 0
    session_number = 0
    trade_date = pd.Timestamp(first_trade_date).normalize()
    if trade_date.dayofweek >= 5:
        trade_date += pd.offsets.BDay()
    clock_start = pd.Timestamp(CLOCK_START).normalize()
    clock_end = pd.Timestamp(CLOCK_END).normalize()
    if trade_date < clock_start or trade_date > clock_end:
        raise ValueError(
            "synthetic fixture falls outside registered market clock"
        )
    while remaining:
        if trade_date > clock_end:
            raise ValueError(
                "synthetic fixture falls outside registered market clock"
            )
        timestamps = _registered_session_timestamps(
            trade_date=trade_date,
        )
        if timestamps.empty:
            trade_date += pd.offsets.BDay()
            continue
        gap_indices, _ = _session_gap_indices(
            session_number=session_number,
            timestamp_count=len(timestamps),
        )
        if gap_indices:
            timestamps = timestamps.delete(list(gap_indices))
        take = min(remaining, len(timestamps))
        timestamps = timestamps[:take]
        ordinal = np.arange(
            emitted,
            emitted + take,
            dtype=np.float64,
        )
        if pattern == "event_heavy":
            center = (
                20_000.0
                + 0.018 * ordinal
                + 11.0 * np.sin(ordinal / 13.0)
                + 4.0 * np.sin(ordinal / 3.0)
            )
            body = 0.55 * np.sin(ordinal / 2.0)
            wick = 0.9 + 0.25 * np.abs(np.sin(ordinal / 5.0))
        else:
            center = (
                20_000.0
                + 0.012 * ordinal
                + 7.0 * np.sin(ordinal / 47.0)
            )
            body = 0.12 * np.sin(ordinal / 9.0)
            wick = np.full(take, 0.5, dtype=np.float64)
        open_price = center - body
        close = center + body
        yield pd.DataFrame(
            {
                "ts": timestamps,
                "open": open_price,
                "high": np.maximum(open_price, close) + wick,
                "low": np.minimum(open_price, close) - wick,
                "close": close,
                "volume": 100.0 + np.mod(ordinal, 31.0),
                "symbol": np.full(take, "NQ-SYNTH", dtype=object),
                "instrument_id": np.ones(take, dtype=np.int64),
            }
        )
        emitted += take
        remaining -= take
        session_number += 1
        trade_date += pd.offsets.BDay()


def _registered_session_timestamps(
    *,
    trade_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return the exact frozen exchange-clock minutes for one session label."""

    label = pd.Timestamp(trade_date).normalize()
    bounds = registered_market_clock._session_bounds(label.date())
    if bounds is None:
        return pd.DatetimeIndex(
            [],
            name="ts",
            tz=MARKET_TIMEZONE,
        )
    opened, close = bounds
    timestamps = pd.date_range(
        opened,
        close,
        freq="1min",
        inclusive="left",
        name="ts",
    )
    if label.date() < registered_market_clock.SETTLEMENT_PAUSE_END_EXCLUSIVE:
        local = timestamps.tz_convert(MARKET_TIMEZONE)
        settlement_pause = (
            (local.hour == 16)
            & (local.minute >= 15)
            & (local.minute < 30)
        )
        timestamps = timestamps[~settlement_pause]
    return timestamps


def _session_gap_indices(
    *,
    session_number: int,
    timestamp_count: int,
) -> tuple[tuple[int, ...], str | None]:
    """Return deterministic registered-open gap positions for one session."""

    if (
        session_number % LONG_OPEN_GAP_SESSION_MODULUS
        == LONG_OPEN_GAP_SESSION_REMAINDER
        and timestamp_count
        > LONG_OPEN_GAP_START_INDEX + LONG_OPEN_GAP_MINUTES
    ):
        return (
            tuple(
                range(
                    LONG_OPEN_GAP_START_INDEX,
                    LONG_OPEN_GAP_START_INDEX
                    + LONG_OPEN_GAP_MINUTES,
                )
            ),
            "data_gap_reset",
        )
    if (
        session_number % SHORT_OPEN_GAP_SESSION_MODULUS
        == SHORT_OPEN_GAP_SESSION_REMAINDER
        and timestamp_count > SHORT_OPEN_GAP_START_INDEX + 1
    ):
        return (SHORT_OPEN_GAP_START_INDEX,), "synthetic_no_trade"
    return (), None


def _synthetic_clock_preflight(
    *,
    source_rows: int,
    first_trade_date: str,
    require_coverage: bool,
) -> dict[str, Any]:
    """Prove fixture dates and planned gaps against the frozen market clock."""

    if source_rows <= 0:
        raise ValueError("source rows must be positive")
    remaining = int(source_rows)
    session_number = 0
    trade_date = pd.Timestamp(first_trade_date).normalize()
    if trade_date.dayofweek >= 5:
        trade_date += pd.offsets.BDay()
    first_label = pd.Timestamp(trade_date)
    first_session_label: pd.Timestamp | None = None
    last_label = first_label
    clock_start = pd.Timestamp(CLOCK_START).normalize()
    clock_end = pd.Timestamp(CLOCK_END).normalize()
    if first_label < clock_start or first_label > clock_end:
        raise ValueError(
            "synthetic fixture falls outside registered market clock: "
            f"first label {first_label.date()}, "
            f"clock {clock_start.date()}..{clock_end.date()}"
        )
    short_open_gaps = 0
    long_open_gaps = 0
    while remaining:
        if trade_date > clock_end:
            raise ValueError(
                "synthetic fixture falls outside registered market clock: "
                f"source extends beyond {clock_end.date()}"
            )
        timestamps = _registered_session_timestamps(
            trade_date=trade_date,
        )
        timestamp_count = len(timestamps)
        if timestamp_count == 0:
            trade_date += pd.offsets.BDay()
            continue
        if first_session_label is None:
            first_session_label = pd.Timestamp(trade_date)
        gap_indices, gap_kind = _session_gap_indices(
            session_number=session_number,
            timestamp_count=timestamp_count,
        )
        emitted_rows = timestamp_count - len(gap_indices)
        take = min(remaining, emitted_rows)
        if gap_indices and take > min(gap_indices):
            proving_indices = (*gap_indices, max(gap_indices) + 1)
            if all(
                is_registered_trading_minute(timestamps[index])
                for index in proving_indices
            ) and all(
                timestamps[right] - timestamps[left]
                == pd.Timedelta(minutes=1)
                for left, right in zip(
                    proving_indices[:-1],
                    proving_indices[1:],
                )
            ):
                if gap_kind == "synthetic_no_trade":
                    short_open_gaps += 1
                elif gap_kind == "data_gap_reset":
                    long_open_gaps += 1
        remaining -= take
        last_label = pd.Timestamp(trade_date)
        session_number += 1
        trade_date += pd.offsets.BDay()

    if require_coverage and (
        short_open_gaps <= 0 or long_open_gaps <= 0
    ):
        raise ValueError(
            "confirmatory fixture does not prove both registered-open "
            "synthetic and data-gap reset coverage"
        )
    return {
        "clock_start_label": clock_start.date().isoformat(),
        "clock_end_label": clock_end.date().isoformat(),
        "first_session_label": (
            first_label if first_session_label is None else first_session_label
        ).date().isoformat(),
        "last_session_label": last_label.date().isoformat(),
        "planned_short_open_gaps": int(short_open_gaps),
        "planned_long_open_gaps": int(long_open_gaps),
    }


def _write_source(
    path: Path,
    *,
    source_rows: int,
    first_trade_date: str,
    pattern: str,
    row_group_rows: int,
) -> tuple[pd.Timestamp, pd.Timestamp, tuple[int, ...]]:
    if row_group_rows <= 0:
        raise ValueError("row-group rows must be positive")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"synthetic source already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("ns", tz=MARKET_TIMEZONE)),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.float64()),
            pa.field("symbol", pa.string()),
            pa.field("instrument_id", pa.int64()),
        ]
    )
    writer = pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        use_dictionary=["symbol"],
        write_statistics=True,
        version="2.6",
    )
    buffered: list[pd.DataFrame] = []
    buffered_rows = 0
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    years: set[int] = set()
    try:
        for frame in _session_frames(
            source_rows=source_rows,
            first_trade_date=first_trade_date,
            pattern=pattern,
        ):
            if first_timestamp is None:
                first_timestamp = pd.Timestamp(frame["ts"].iloc[0])
            last_timestamp = pd.Timestamp(frame["ts"].iloc[-1])
            years.update(
                int(value)
                for value in pd.DatetimeIndex(frame["ts"]).year.unique()
            )
            buffered.append(frame)
            buffered_rows += len(frame)
            while buffered_rows >= row_group_rows:
                combined = pd.concat(buffered, ignore_index=True)
                head = combined.iloc[:row_group_rows]
                tail = combined.iloc[row_group_rows:]
                writer.write_table(
                    pa.Table.from_pandas(
                        head,
                        schema=schema,
                        preserve_index=False,
                    ),
                    row_group_size=row_group_rows,
                )
                buffered = [] if tail.empty else [tail]
                buffered_rows = len(tail)
        if buffered:
            combined = pd.concat(buffered, ignore_index=True)
            writer.write_table(
                pa.Table.from_pandas(
                    combined,
                    schema=schema,
                    preserve_index=False,
                ),
                row_group_size=row_group_rows,
            )
    finally:
        writer.close()
    if first_timestamp is None or last_timestamp is None:
        raise AssertionError("synthetic source generation produced no rows")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    return (
        first_timestamp,
        last_timestamp + pd.Timedelta(minutes=1),
        tuple(sorted(years)),
    )


def _base_contracts(
    fixture_root: Path,
    *,
    source_path: Path,
    source_sha256: str,
    source_start: pd.Timestamp,
    source_end: pd.Timestamp,
    calendar_years: tuple[int, ...],
    batch_rows: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    implementation = _implementation_hashes()
    protocol_sha256 = sha256_file(PRIMITIVE_PROTOCOL)
    audit = {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_audit_contract",
        "audit_id": (
            f"{ACTIVE_EXPERIMENT_ID}-SYNTHETIC-FULL-E2E-"
            f"{source_sha256[:16]}"
        ),
        "status": "synthetic_test_contract",
        "authorization": "synthetic integration tests only",
        "bindings": {
            "primitive_protocol_sha256": protocol_sha256,
            "causal_source_sha256": source_sha256,
        },
        "source": {
            "path": str(source_path),
            "start": source_start.isoformat(),
            "end_exclusive": source_end.isoformat(),
            "window_role": "synthetic_fixture",
        },
        "selection": {
            "timeframes": ["4H", "1H", "5m", "1m"],
            "directions": ["long", "short"],
            "case_classes": [
                "confirmed_bos",
                "wick_only_no_close",
                "broken_or_opposed",
            ],
            "calendar_years": list(calendar_years),
            "cases_per_bucket": 1,
        },
    }
    audit_path = fixture_root / "contracts" / "audit.json"
    _write_new_json(audit_path, audit)
    runner = {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_runner_contract",
        "runner_id": (
            f"{ACTIVE_EXPERIMENT_ID}-SYNTHETIC-FULL-E2E-RUNNER-"
            f"{source_sha256[:16]}"
        ),
        "status": "implementation_contract_frozen",
        "authorization": (
            "implementation and synthetic tests only; real discovery "
            "execution is forbidden"
        ),
        "bindings": {
            "model_config_sha256": sha256_file(MODEL_CONFIG),
            "blind_audit_contract_sha256": sha256_file(audit_path),
            "primitive_protocol_sha256": protocol_sha256,
            "io_code_sha256": implementation["io_sha256"],
            "causal_reader_code_sha256": implementation["causal_sha256"],
            "market_clock_code_sha256": implementation[
                "market_clock_sha256"
            ],
            "semantic_audit_code_sha256": implementation[
                "semantic_audit_sha256"
            ],
        },
        "source_iterator": {
            "batch_rows": int(batch_rows),
            "maximum_no_trade_gap_minutes": 5,
            "allow_data_gap_reset": True,
        },
        "implementation_hashes_required": list(implementation),
    }
    runner_path = fixture_root / "contracts" / "runner.json"
    _write_new_json(runner_path, runner)
    return audit, runner, implementation


def _release_payload(
    fixture_root: Path,
    *,
    release_name: str,
    output_name: str,
) -> dict[str, Any]:
    manifest = _read_json(fixture_root / "fixture_manifest.json")
    audit_path = fixture_root / "contracts" / "audit.json"
    runner_path = fixture_root / "contracts" / "runner.json"
    audit = _read_json(audit_path)
    runner = _read_json(runner_path)
    implementation = dict(manifest["implementation_hashes"])
    confirmatory_one_time = bool(
        manifest.get("confirmatory_one_time", False)
    )
    output = _guard_fixture_member(
        fixture_root / "outputs" / output_name,
        fixture_root,
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "a new synthetic release requires an absent output path"
        )
    payload = {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_pass1_release",
        "release_id": release_name,
        "status": "synthetic_test_release",
        "authorization": (
            "synthetic pass1 confirmatory one-time"
            if confirmatory_one_time
            else "synthetic pass1 full-E2E only"
        ),
        "confirmatory_one_time": confirmatory_one_time,
        "execution_authorized": True,
        "authorized_phases": ["pass1"],
        "issued_at": str(manifest["issued_at"]),
        "expires_at": str(manifest["expires_at"]),
        "bindings": {
            "audit_contract_sha256": sha256_file(audit_path),
            "engineering_runner_contract_sha256": sha256_file(
                runner_path
            ),
            "semantic_discovery_runner_sha256": implementation[
                "semantic_discovery_runner_sha256"
            ],
            "causal_source_sha256": manifest["source_sha256"],
            "model_config_sha256": runner["bindings"][
                "model_config_sha256"
            ],
            "primitive_protocol_sha256": runner["bindings"][
                "primitive_protocol_sha256"
            ],
            "io_code_sha256": runner["bindings"]["io_code_sha256"],
            "causal_reader_code_sha256": runner["bindings"][
                "causal_reader_code_sha256"
            ],
            "market_clock_code_sha256": runner["bindings"][
                "market_clock_code_sha256"
            ],
            "semantic_audit_code_sha256": runner["bindings"][
                "semantic_audit_code_sha256"
            ],
            "observer_code_sha256": implementation[
                "observation_sha256"
            ],
            "structure_code_sha256": implementation["structure_sha256"],
        },
        "source": {
            "path": audit["source"]["path"],
            "start": audit["source"]["start"],
            "end_exclusive": audit["source"]["end_exclusive"],
            "window_role": "synthetic_fixture",
        },
        "runtime": runtime_environment(),
        "run": {
            "batch_rows": int(manifest["batch_rows"]),
            "maximum_history": int(manifest["maximum_history"]),
            "maximum_no_trade_gap_minutes": 5,
            "allow_data_gap_reset": True,
            "checkpoint_source_rows": int(
                manifest["checkpoint_source_rows"]
            ),
            "real_only_case_anchors": True,
            "diagnostic_stop_allowed": not confirmatory_one_time,
        },
        "resources": dict(manifest["resource_limits"]),
        "output": {"path": str(output)},
    }
    if confirmatory_one_time:
        payload["confirmatory"] = {
            "attempt_id": manifest["confirmatory_attempt_id"],
            "registry_root": manifest[
                "confirmatory_registry_root"
            ],
            "attempt_marker_path": manifest[
                "confirmatory_attempt_marker_path"
            ],
        }
    return payload


def prepare_registry(args: argparse.Namespace) -> dict[str, Any]:
    """Create and durably register an empty one-shot pair directory."""

    registry_root = _guard_fixture_root(args.registry_root)
    parent = registry_root.parent
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or registry_root.exists()
        or registry_root.is_symlink()
    ):
        raise FileExistsError(
            "confirmatory registry requires an existing trusted parent "
            "and an absent target"
        )
    registry_root.mkdir()
    parent_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    registry_descriptor = os.open(registry_root, os.O_RDONLY)
    try:
        os.fsync(registry_descriptor)
    finally:
        os.close(registry_descriptor)
    return {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_registry_prepared",
        "registry_root": str(registry_root),
        "empty": not any(registry_root.iterdir()),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    fixture_root = _guard_fixture_root(args.fixture_root)
    expiry = pd.Timestamp(args.expires_at)
    if expiry.tzinfo is None:
        raise ValueError("synthetic release expiry must be timezone-aware")
    issued_at = pd.Timestamp.now(tz="UTC")
    confirmatory_attempt_id: str | None = None
    confirmatory_registry_root: Path | None = None
    confirmatory_attempt_marker_path: Path | None = None
    if args.confirmatory_one_time:
        remaining_seconds = (
            expiry.tz_convert("UTC") - issued_at
        ).total_seconds()
        if (
            remaining_seconds <= 0
            or remaining_seconds
            > CONFIRMATORY_GATES["maximum_release_window_seconds"]
        ):
            raise ValueError(
                "confirmatory release must expire within 24 hours"
            )
        confirmatory_attempt_id = str(
            args.confirmatory_attempt_id or ""
        )
        if re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            confirmatory_attempt_id,
        ) is None:
            raise ValueError("confirmatory attempt id is invalid")
        if not args.confirmatory_registry_root:
            raise ValueError("confirmatory registry root is required")
        confirmatory_registry_root = _guard_fixture_root(
            args.confirmatory_registry_root
        )
        if (
            confirmatory_registry_root.is_symlink()
            or not confirmatory_registry_root.is_dir()
        ):
            raise ValueError(
                "confirmatory registry root must already exist"
            )
        confirmatory_attempt_marker_path = (
            confirmatory_registry_root
            / "CONFIRMATORY_ATTEMPT_STARTED.json"
        )
    elif (
        args.confirmatory_attempt_id is not None
        or args.confirmatory_registry_root is not None
    ):
        raise ValueError(
            "ordinary fixture may not bind a confirmatory registry"
        )
    if fixture_root.exists() or fixture_root.is_symlink():
        raise FileExistsError(
            "fixture root must be absent before preparation"
        )
    clock_coverage = _synthetic_clock_preflight(
        source_rows=args.source_rows,
        first_trade_date=args.first_trade_date,
        require_coverage=bool(args.confirmatory_one_time),
    )
    fixture_root.mkdir(parents=True)
    source_path = fixture_root / "source" / "synthetic_ohlcv_1m.parquet"
    started = time.perf_counter()
    source_start, source_end, years = _write_source(
        source_path,
        source_rows=args.source_rows,
        first_trade_date=args.first_trade_date,
        pattern=args.pattern,
        row_group_rows=args.row_group_rows,
    )
    source_sha256 = sha256_file(source_path)
    _, _, implementation = _base_contracts(
        fixture_root,
        source_path=source_path,
        source_sha256=source_sha256,
        source_start=source_start,
        source_end=source_end,
        calendar_years=years,
        batch_rows=args.batch_rows,
    )
    resource_limits = {
        "rss_hard_ceiling_bytes": int(args.rss_hard_ceiling_bytes),
        "disk_free_floor_bytes": int(args.disk_free_floor_bytes),
        "checkpoint_stall_timeout_seconds": float(
            args.checkpoint_stall_timeout_seconds
        ),
        "resource_check_source_rows": int(
            args.resource_check_source_rows
        ),
    }
    manifest = {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_fixture_manifest",
        "mode": "synthetic_causal_no_economic",
        "fixture_root": str(fixture_root),
        "source_path": str(source_path),
        "source_rows": int(args.source_rows),
        "source_start": source_start.isoformat(),
        "source_end_exclusive": source_end.isoformat(),
        "source_sha256": source_sha256,
        "source_bytes": source_path.stat().st_size,
        "first_trade_date": args.first_trade_date,
        "pattern": args.pattern,
        "calendar_years": list(years),
        "planned_clock_coverage": clock_coverage,
        "row_group_rows": int(args.row_group_rows),
        "batch_rows": int(args.batch_rows),
        "maximum_history": int(args.maximum_history),
        "checkpoint_source_rows": int(args.checkpoint_source_rows),
        "confirmatory_one_time": bool(args.confirmatory_one_time),
        "issued_at": issued_at.isoformat(),
        "expires_at": args.expires_at,
        "confirmatory_attempt_id": confirmatory_attempt_id,
        "confirmatory_registry_root": (
            None
            if confirmatory_registry_root is None
            else str(confirmatory_registry_root)
        ),
        "confirmatory_attempt_marker_path": (
            None
            if confirmatory_attempt_marker_path is None
            else str(confirmatory_attempt_marker_path)
        ),
        "resource_limits": resource_limits,
        "implementation_hashes": implementation,
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "primitive_protocol_sha256": sha256_file(PRIMITIVE_PROTOCOL),
        "audit_contract_sha256": sha256_file(
            fixture_root / "contracts" / "audit.json"
        ),
        "runner_contract_sha256": sha256_file(
            fixture_root / "contracts" / "runner.json"
        ),
        "preparation_wall_seconds": time.perf_counter() - started,
        "preparation_excluded_from_timed_pass1": True,
    }
    _write_new_json(fixture_root / "fixture_manifest.json", manifest)
    release = _release_payload(
        fixture_root,
        release_name=args.release_name,
        output_name=args.output_name,
    )
    release_path = fixture_root / "releases" / f"{args.release_name}.json"
    _write_new_json(release_path, release)
    return {
        **manifest,
        "release_path": str(release_path),
        "release_sha256": sha256_file(release_path),
        "output_path": release["output"]["path"],
    }


def create_release(args: argparse.Namespace) -> dict[str, Any]:
    fixture_root = _guard_fixture_root(args.fixture_root)
    release = _release_payload(
        fixture_root,
        release_name=args.release_name,
        output_name=args.output_name,
    )
    release_path = fixture_root / "releases" / f"{args.release_name}.json"
    _write_new_json(release_path, release)
    return {
        "artifact": "v3_synthetic_full_e2e_release_created",
        "release_path": str(release_path),
        "release_sha256": sha256_file(release_path),
        "output_path": release["output"]["path"],
    }


class _Monitor:
    def __init__(self, *, output_root: Path, sample_seconds: float) -> None:
        self.output_root = output_root
        self.sample_seconds = float(sample_seconds)
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.rss_peak_bytes = 0
        self.disk_free_min_bytes: int | None = None
        self.progress_statuses: list[str] = []
        self.progress_percentages: list[float] = []
        self._thread = threading.Thread(
            target=self._run,
            name="synthetic-pass1-e2e-monitor",
            daemon=True,
        )

    def start(self) -> None:
        # Establish a resource sample synchronously so source hashing cannot
        # begin in the scheduler gap before the monitor thread first runs.
        self._sample()
        self._thread.start()

    def finish(self) -> None:
        # Capture the final terminal progress even if it was published between
        # the monitor's preceding periodic sample and task completion.
        self._sample()
        self.stop.set()
        self._thread.join(timeout=max(5.0, self.sample_seconds * 4.0))
        if self._thread.is_alive():
            raise TimeoutError("synthetic E2E monitor did not stop")

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        rss = int(process.memory_info().rss)
        anchor = self.output_root.parent
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        filesystem = os.statvfs(anchor)
        free = int(filesystem.f_bavail * filesystem.f_frsize)
        self.rss_peak_bytes = max(self.rss_peak_bytes, rss)
        self.disk_free_min_bytes = (
            free
            if self.disk_free_min_bytes is None
            else min(self.disk_free_min_bytes, free)
        )
        progress_path = self.output_root / "progress.json"
        status = None
        percentage = None
        if progress_path.is_file() and not progress_path.is_symlink():
            try:
                progress = _read_json(progress_path)
            except (OSError, ValueError, json.JSONDecodeError):
                progress = {}
            status = progress.get("status")
            percentage = progress.get("completed_percent")
            if isinstance(status, str) and (
                not self.progress_statuses
                or self.progress_statuses[-1] != status
            ):
                self.progress_statuses.append(status)
            if isinstance(percentage, (int, float)):
                self.progress_percentages.append(float(percentage))
        self.samples.append(
            {
                "monotonic": time.monotonic(),
                "rss_bytes": rss,
                "disk_free_bytes": free,
                "progress_status": status,
                "completed_percent": percentage,
            }
        )

    def _run(self) -> None:
        while not self.stop.is_set():
            self._sample()
            self.stop.wait(self.sample_seconds)


def _load_bound_runner(
    fixture_root: Path,
    release_path: Path,
) -> tuple[SemanticDiscoveryRunner, dict[str, Any], Path]:
    manifest = _read_json(fixture_root / "fixture_manifest.json")
    audit_path = fixture_root / "contracts" / "audit.json"
    runner_path = fixture_root / "contracts" / "runner.json"
    audit = _read_json(audit_path)
    runner_contract = _read_json(runner_path)
    release = _read_json(release_path)
    expected_files = {
        "audit_contract_sha256": audit_path,
        "runner_contract_sha256": runner_path,
        "model_config_sha256": MODEL_CONFIG,
        "primitive_protocol_sha256": PRIMITIVE_PROTOCOL,
    }
    for field, path in expected_files.items():
        guarded = (
            _guard_fixture_member(path, fixture_root)
            if field in {"audit_contract_sha256", "runner_contract_sha256"}
            else path.resolve()
        )
        if sha256_file(guarded) != manifest[field]:
            raise ValueError(f"frozen fixture binding changed: {field}")
    implementation = _implementation_hashes()
    if implementation != manifest["implementation_hashes"]:
        raise ValueError("frozen implementation hashes changed")
    source_path = _guard_fixture_member(
        release["source"]["path"],
        fixture_root,
    )
    output_root = _guard_fixture_member(
        release["output"]["path"],
        fixture_root,
    )
    source = ParquetBatchSource(
        source_path,
        start=release["source"]["start"],
        end_exclusive=release["source"]["end_exclusive"],
        batch_rows=int(release["run"]["batch_rows"]),
        expected_sha256=release["bindings"]["causal_source_sha256"],
    )
    runner = SemanticDiscoveryRunner(
        audit_contract=audit,
        runner_contract=runner_contract,
        source=source,
        model_config_path=MODEL_CONFIG,
        output_root=output_root,
        audit_contract_sha256=sha256_file(audit_path),
        runner_contract_sha256=sha256_file(runner_path),
        implementation_hashes=implementation,
        maximum_history=int(release["run"]["maximum_history"]),
        checkpoint_source_rows=int(
            release["run"]["checkpoint_source_rows"]
        ),
        execution_release=release,
        execution_release_sha256=sha256_file(release_path),
    )
    return runner, manifest, output_root


def _terminal_qa(
    *,
    runner: SemanticDiscoveryRunner,
    output_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    selection_path = output_root / "selection_authority.json"
    progress_path = output_root / "progress.json"
    checkpoint = DiscoveryCheckpointStore(
        output_root / "_checkpoint" / "pass1"
    )
    if checkpoint.status != "valid":
        raise ValueError("terminal checkpoint is not valid")
    source_guard = FrozenParquetSourceIdentity(
        sha256=runner.source.sha256,
        start=runner.source.start,
        end_exclusive=runner.source.end_exclusive,
        batch_rows=runner.source.batch_rows,
    )
    terminal_runner = SemanticDiscoveryRunner(
        audit_contract=runner.audit_contract,
        runner_contract=runner.runner_contract,
        source=source_guard,
        model_config_path=runner.model_config_path,
        output_root=runner.output_root,
        audit_contract_sha256=runner.audit_contract_sha256,
        runner_contract_sha256=runner.runner_contract_sha256,
        implementation_hashes=runner.implementation_hashes,
        maximum_history=runner.maximum_history,
        checkpoint_source_rows=runner.checkpoint_source_rows,
        execution_release=runner.execution_release,
        execution_release_sha256=runner.execution_release_sha256,
    )
    # Both source methods on FrozenParquetSourceIdentity raise immediately.
    # A successful reconciliation therefore proves zero source count/iteration.
    reconciled = terminal_runner.run_pass1(resume=True)
    if reconciled != selection_path:
        raise AssertionError("terminal reconciliation returned another path")
    progress = _read_json(progress_path)
    selection = _read_json(selection_path)
    if (
        float(progress.get("completed_percent", -1.0)) != 100.0
        or progress.get("terminal_reconciled") is not True
        or int(selection.get("source_rows", -1))
        != int(manifest["source_rows"])
    ):
        raise ValueError("terminal Pass1 QA failed")
    cases = selection.get("cases")
    if not isinstance(cases, list):
        raise ValueError("selection cases are not a list")
    selection_contract = runner.audit_contract["selection"]
    expected_buckets = [
        "|".join((timeframe, direction, case_class, str(year)))
        for timeframe in selection_contract["timeframes"]
        for direction in selection_contract["directions"]
        for case_class in selection_contract["case_classes"]
        for year in selection_contract["calendar_years"]
    ]
    buckets: dict[str, list[dict[str, Any]]] = {
        key: [] for key in expected_buckets
    }
    primitive_hash = str(
        runner.audit_contract["bindings"][
            "primitive_protocol_sha256"
        ]
    )
    event_ids: set[str] = set()
    for case in cases:
        event_id = str(case["semantic_event_id"])
        if not event_id or event_id in event_ids:
            raise ValueError("selection event identity is absent/duplicated")
        event_ids.add(event_id)
        clock = pd.Timestamp(case["case_clock"])
        source_start = pd.Timestamp(case["case_source_row_start"])
        if (
            clock.tzinfo is None
            or source_start.tzinfo is None
            or source_start + pd.Timedelta(minutes=1) != clock
            or int(case["calendar_year"])
            != int(clock.tz_convert(MARKET_TIMEZONE).year)
            or case.get("case_bar_synthetic") is not False
            or event_id
            != "|".join(
                (
                    str(case["bos_id"]),
                    str(case["case_class"]),
                    clock.isoformat(),
                )
            )
            or str(case["selection_score"])
            != hashlib.sha256(
                f"{primitive_hash}|{event_id}".encode("utf-8")
            ).hexdigest()
        ):
            raise ValueError(
                "selection event/score/provenance QA failed"
            )
        for digest_field in (
            "selection_score",
            "semantic_state_hash",
            "case_source_row_sha256",
            "case_bar_sha256",
            "source_prefix_root",
            "produced_bar_prefix_root",
        ):
            if HEX64.fullmatch(str(case[digest_field])) is None:
                raise ValueError(
                    "selection case contains an invalid digest"
                )
        key = "|".join(
            (
                str(case["timeframe"]),
                str(case["direction"]),
                str(case["case_class"]),
                str(case["calendar_year"]),
            )
        )
        if key not in buckets:
            raise ValueError("selection contains an unregistered bucket")
        buckets[key].append(case)
    expected_cases = [
        case
        for key in expected_buckets
        for case in sorted(
            buckets[key],
            key=lambda value: (
                value["selection_score"],
                value["semantic_event_id"],
            ),
        )
    ]
    cases_per_bucket = int(selection_contract["cases_per_bucket"])
    missing_buckets = [
        key
        for key in expected_buckets
        if len(buckets[key]) != cases_per_bucket
    ]
    if (
        cases != expected_cases
        or int(selection.get("selected_case_count", -1)) != len(cases)
        or int(selection.get("expected_bucket_count", -1))
        != len(expected_buckets)
        or int(selection.get("expected_case_count", -1))
        != len(expected_buckets) * cases_per_bucket
        or int(selection.get("cases_per_bucket", -1))
        != cases_per_bucket
        or any(
            len(values) > cases_per_bucket
            for values in buckets.values()
        )
        or (
            selection.get("status") == "complete"
            and missing_buckets
        )
        or (
            selection.get("status") == "unavailable"
            and not missing_buckets
        )
        or selection.get("missing_buckets") != missing_buckets
    ):
        raise ValueError("terminal selection ordering is invalid")
    return {
        "selection_status": selection.get("status"),
        "progress_status": progress.get("status"),
        "completed_percent": float(progress["completed_percent"]),
        "source_rows": int(selection["source_rows"]),
        "produced_bars": int(selection["produced_bars"]),
        "real_bars": int(progress["real_bars"]),
        "synthetic_bars": int(progress["synthetic_bars"]),
        "reset_epoch": int(progress["reset_epoch"]),
        "source_prefix_root": selection["source_prefix_root"],
        "produced_bar_prefix_root": selection[
            "produced_bar_prefix_root"
        ],
        "selected_case_count": int(selection["selected_case_count"]),
        "expected_case_count": int(selection["expected_case_count"]),
        "missing_bucket_count": len(selection["missing_buckets"]),
        "selection_sha256": sha256_file(selection_path),
        "progress_sha256": sha256_file(progress_path),
        "checkpoint_state_sha256": checkpoint.current_state_sha256(),
        "terminal_reconciled": True,
        "terminal_source_guard": type(source_guard).__name__,
        "terminal_source_method_calls": 0,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    fixture_root = _guard_fixture_root(args.fixture_root)
    release_path = _guard_fixture_member(args.release, fixture_root)
    if args.monitor_sample_seconds <= 0:
        raise ValueError("monitor sample interval must be positive")
    if args.expected_outcome == "interrupt" and (
        args.diagnostic_stop_after_source_rows <= 0
    ):
        raise ValueError("interrupt outcome requires a positive stop row")
    if args.expected_outcome == "complete" and (
        args.diagnostic_stop_after_source_rows != 0
    ):
        raise ValueError("complete outcome may not request interruption")
    timed_started = time.perf_counter()
    cpu_started = time.process_time()
    release_preview = _read_json(release_path)
    confirmatory_mode = bool(
        getattr(args, "confirmatory_mode", False)
    )
    if (
        release_preview.get("confirmatory_one_time") is True
        and not confirmatory_mode
    ):
        raise PermissionError(
            "one-time confirmatory release requires confirm-pair mode"
        )
    if confirmatory_mode and (
        release_preview.get("confirmatory_one_time") is not True
        or args.resume
        or args.diagnostic_stop_after_source_rows != 0
        or args.expected_outcome != "complete"
    ):
        raise PermissionError(
            "confirmatory mode parameters are not one-time exact"
        )
    output_root = _guard_fixture_member(
        release_preview["output"]["path"],
        fixture_root,
    )
    monitor = _Monitor(
        output_root=output_root,
        sample_seconds=args.monitor_sample_seconds,
    )
    monitor.start()
    starting_source_rows = 0
    observed_outcome = "complete"
    failure: dict[str, str] | None = None
    try:
        runner, manifest, loaded_output_root = _load_bound_runner(
            fixture_root,
            release_path,
        )
        if loaded_output_root != output_root:
            raise ValueError("release output changed during load")
        if args.resume:
            checkpoint_manifest = _read_json(
                output_root
                / "_checkpoint"
                / "pass1"
                / "manifest.json"
            )
            starting_source_rows = int(
                checkpoint_manifest["source_rows_admitted"]
            )
        try:
            runner.run_pass1(
                resume=args.resume,
                diagnostic_stop_after_source_rows=(
                    args.diagnostic_stop_after_source_rows
                ),
            )
        except RuntimeError as exc:
            if (
                args.expected_outcome == "interrupt"
                and str(exc)
                == "intentional semantic pass1 interruption"
            ):
                observed_outcome = "interrupt"
                failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            else:
                raise
        terminal = None
        if observed_outcome == "complete":
            terminal = _terminal_qa(
                runner=runner,
                output_root=output_root,
                manifest=manifest,
            )
    finally:
        monitor.finish()
    wall_seconds = time.perf_counter() - timed_started
    cpu_seconds = time.process_time() - cpu_started
    if observed_outcome != args.expected_outcome:
        raise AssertionError(
            "synthetic E2E observed an unexpected outcome"
        )
    if any(
        right < left
        for left, right in zip(
            monitor.progress_percentages[:-1],
            monitor.progress_percentages[1:],
        )
    ):
        raise ValueError("observed progress percentages regressed")
    checkpoint = DiscoveryCheckpointStore(
        output_root / "_checkpoint" / "pass1"
    )
    final_checkpoint_manifest = _read_json(checkpoint.manifest_path)
    admitted_source_rows = int(
        final_checkpoint_manifest["source_rows_admitted"]
    )
    session_source_rows = admitted_source_rows - starting_source_rows
    session_rate = (
        session_source_rows / wall_seconds
        if wall_seconds > 0
        else 0.0
    )
    result = {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_pass1_result",
        "mode": "synthetic_causal_no_economic",
        "fixture_manifest_sha256": sha256_file(
            fixture_root / "fixture_manifest.json"
        ),
        "release_sha256": sha256_file(release_path),
        "resume": bool(args.resume),
        "diagnostic_stop_after_source_rows": int(
            args.diagnostic_stop_after_source_rows
        ),
        "expected_outcome": args.expected_outcome,
        "observed_outcome": observed_outcome,
        "failure": failure,
        "timed_scope": (
            "contract verification + source full-file hash + timestamp count "
            "+ decode + causal replay + commitments + selector + "
            "checkpoint/fsync + resource/progress monitoring + terminal "
            "publication/reconciliation"
        ),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "source_rows": int(manifest["source_rows"]),
        "starting_source_rows": starting_source_rows,
        "admitted_source_rows": admitted_source_rows,
        "session_source_rows": session_source_rows,
        "source_rows_per_second": session_rate,
        "projected_1740541_rows_seconds": (
            DEFAULT_PROJECTED_SOURCE_ROWS / session_rate
            if (
                session_rate > 0
                and observed_outcome == "complete"
                and not args.resume
            )
            else None
        ),
        "rss_peak_bytes": int(monitor.rss_peak_bytes),
        "disk_free_min_bytes": monitor.disk_free_min_bytes,
        "monitor_sample_count": len(monitor.samples),
        "observed_progress_statuses": monitor.progress_statuses,
        "observed_progress_min_percent": (
            None
            if not monitor.progress_percentages
            else min(monitor.progress_percentages)
        ),
        "observed_progress_max_percent": (
            None
            if not monitor.progress_percentages
            else max(monitor.progress_percentages)
        ),
        "checkpoint_status": checkpoint.status,
        "checkpoint_state_sha256": (
            checkpoint.current_state_sha256()
            if checkpoint.exists
            else None
        ),
        "terminal": terminal,
        "runtime": _runtime_fingerprint(),
    }
    result_path = _guard_fixture_member(args.result, fixture_root)
    _write_new_json(result_path, result)
    return {**result, "result_path": str(result_path)}


def compare_outputs(args: argparse.Namespace) -> dict[str, Any]:
    fixture_root = _guard_fixture_root(args.fixture_root)
    left = _guard_fixture_member(args.left_output, fixture_root)
    right = _guard_fixture_member(args.right_output, fixture_root)
    left_selection = left / "selection_authority.json"
    right_selection = right / "selection_authority.json"
    left_bytes = left_selection.read_bytes()
    right_bytes = right_selection.read_bytes()
    if left_bytes != right_bytes:
        raise AssertionError(
            "uninterrupted and resumed selection authorities differ"
        )
    payload = _read_json(left_selection)
    result = {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_resume_exactness",
        "mode": "synthetic_causal_no_economic",
        "selection_byte_exact": True,
        "selection_sha256": hashlib.sha256(left_bytes).hexdigest(),
        "source_prefix_root": payload["source_prefix_root"],
        "produced_bar_prefix_root": payload["produced_bar_prefix_root"],
        "source_rows": int(payload["source_rows"]),
        "produced_bars": int(payload["produced_bars"]),
        "left_output": str(left),
        "right_output": str(right),
    }
    result_path = _guard_fixture_member(args.result, fixture_root)
    _write_new_json(result_path, result)
    return {**result, "result_path": str(result_path)}


def runtime_fingerprint_command(
    _args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_runtime_fingerprint",
        "runtime": _runtime_fingerprint(),
    }


def _project_regular_file(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"frozen binding is not a regular file: {path}")
    resolved = path.resolve()
    if not _is_relative_to(resolved, ROOT.resolve()):
        raise PermissionError(
            "confirmatory frozen files must stay in the project root"
        )
    if _is_relative_to(resolved, (ROOT / "data").resolve()):
        raise PermissionError("confirmatory binding may not access data/")
    return resolved


def _confirmatory_gap_expectations(
    manifest: Mapping[str, Any],
) -> tuple[int, int]:
    coverage = manifest.get("planned_clock_coverage")
    if not isinstance(coverage, dict):
        raise ValueError(
            "confirmatory planned clock coverage is absent"
        )
    short_open_gaps = coverage.get("planned_short_open_gaps")
    long_open_gaps = coverage.get("planned_long_open_gaps")
    if (
        type(short_open_gaps) is not int
        or short_open_gaps <= 0
        or type(long_open_gaps) is not int
        or long_open_gaps <= 0
    ):
        raise ValueError(
            "confirmatory gap plans must be positive integers"
        )
    return short_open_gaps, long_open_gaps


def _confirmatory_preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if HEX64.fullmatch(args.expected_freeze_sha256) is None:
        raise ValueError("expected freeze digest is invalid")
    if HEX64.fullmatch(args.expected_preregistration_sha256) is None:
        raise ValueError("expected preregistration digest is invalid")
    freeze_path = _project_regular_file(args.freeze)
    preregistration_path = _project_regular_file(args.preregistration)
    if (
        _bootstrap_sha256_file(freeze_path)
        != args.expected_freeze_sha256
    ):
        raise ValueError("confirmatory freeze digest changed")
    if (
        _bootstrap_sha256_file(preregistration_path)
        != args.expected_preregistration_sha256
    ):
        raise ValueError("confirmatory preregistration digest changed")
    freeze = _read_json(freeze_path)
    if (
        freeze.get("status") != "confirmatory_pair_frozen"
        or freeze.get("artifact")
        != "v3_synthetic_full_e2e_confirmatory_pair_freeze"
    ):
        raise ValueError("confirmatory pair freeze is not active")
    if (
        _project_regular_file(
            str(freeze.get("preregistration_path", ""))
        )
        != preregistration_path
    ):
        raise ValueError("confirmatory preregistration path changed")
    if (
        freeze.get("preregistration_sha256")
        != args.expected_preregistration_sha256
    ):
        raise ValueError(
            "confirmatory freeze preregistration digest changed"
        )
    issued_at = pd.Timestamp(freeze.get("issued_at"))
    expiry = pd.Timestamp(freeze.get("expires_at"))
    now = pd.Timestamp.now(tz="UTC")
    if (
        issued_at.tzinfo is None
        or expiry.tzinfo is None
        or issued_at.tz_convert("UTC") > now
        or expiry.tz_convert("UTC") <= now
        or (
            expiry.tz_convert("UTC") - issued_at.tz_convert("UTC")
        ).total_seconds()
        > CONFIRMATORY_GATES["maximum_release_window_seconds"]
    ):
        raise PermissionError(
            "confirmatory pair freeze window is invalid"
        )
    if freeze.get("gates") != CONFIRMATORY_GATES:
        raise ValueError("confirmatory hard gates changed")
    observed_runtime = _runtime_fingerprint()
    if freeze.get("runtime") != observed_runtime:
        raise ValueError("confirmatory runtime fingerprint changed")
    if (
        observed_runtime.get("python_executable")
        != "~/miniconda3/bin/python"
        or observed_runtime.get("platform_machine") != "x86_64"
        or observed_runtime.get("rosetta_translated") is not True
        or observed_runtime.get("host_arm64_capable") is not True
    ):
        raise ValueError("confirmatory x86/Rosetta runtime is not exact")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("confirmatory frozen bindings are absent")
    required_bindings = {
        "scripts/benchmark_v3_semantic_pass1_e2e.py",
        "smc_trader/semantic_discovery_runner.py",
        "smc_trader/artifact_stream.py",
        "smc_trader/model.py",
        "smc_trader/causal.py",
        "smc_trader/structure.py",
        "smc_trader/observation.py",
        "smc_trader/market_clock.py",
        "smc_trader/io.py",
        "smc_trader/semantic_audit.py",
        "configs/model_v3_0_1_exp001_structure_bos_identity_r3.json",
        (
            "configs/"
            "smc_primitives_v3_0_1_structure_bos_audit_closure_r3.json"
        ),
        "tests/test_v3_semantic_discovery_runner.py",
        "tests/test_v3_semantic_discovery_pass1_release.py",
        "tests/test_v3_semantic_pass1_e2e_harness.py",
        (
            "configs/experiments/"
            "EXP-SMC-3.0.1-001-STRUCTURE-BOS-AUDIT-CLOSURE-R3-"
            "TEST-FREEZE-R6.json"
        ),
        (
            "reports/validation_2026-07-29/"
            "v3_exp001_semantic_audit_closure_r6_targeted_"
            "posttest_validation.md"
        ),
        (
            "reports/validation_2026-07-29/"
            "v3_exp001_r6_ledger_clock_erratum.md"
        ),
        (
            "reports/validation_2026-07-29/"
            "v3_exp001_r3_full_e2e_successor_r7_pretest_self_review.md"
        ),
    }
    required_bindings.add(
        str(preregistration_path.relative_to(ROOT))
    )
    fixtures = freeze.get("fixtures")
    if (
        not isinstance(fixtures, list)
        or [item.get("pattern") for item in fixtures]
        != ["light", "event_heavy"]
    ):
        raise ValueError("confirmatory fixture pair changed")
    attempt_id = str(freeze.get("attempt_id", ""))
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
        attempt_id,
    ) is None:
        raise ValueError("confirmatory attempt id is invalid")
    prepared: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    for raw in fixtures:
        spec = dict(raw)
        if HEX64.fullmatch(str(spec.get("source_sha256", ""))) is None:
            raise ValueError("confirmatory source digest is invalid")
        fixture_root = _guard_fixture_root(spec["fixture_root"])
        manifest_path = _guard_fixture_member(
            spec["fixture_manifest_path"],
            fixture_root,
        )
        if manifest_path != (
            fixture_root / "fixture_manifest.json"
        ).resolve():
            raise ValueError(
                "confirmatory fixture manifest path is not canonical"
            )
        release_path = _guard_fixture_member(
            spec["release_path"],
            fixture_root,
        )
        audit_path = _guard_fixture_member(
            fixture_root / "contracts" / "audit.json",
            fixture_root,
        )
        runner_contract_path = _guard_fixture_member(
            fixture_root / "contracts" / "runner.json",
            fixture_root,
        )
        source_path = _guard_fixture_member(
            spec["source_path"],
            fixture_root,
        )
        output_path = _guard_fixture_member(
            spec["output_path"],
            fixture_root,
        )
        result_path = _guard_fixture_member(
            spec["result_path"],
            fixture_root,
        )
        for path in (
            manifest_path,
            release_path,
            audit_path,
            runner_contract_path,
            source_path,
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    "confirmatory fixture input is not a regular file"
                )
        if source_path.stat().st_nlink != 1:
            raise ValueError(
                "confirmatory source may not be a hard-linked file"
            )
        if (
            output_path.exists()
            or output_path.is_symlink()
            or result_path.exists()
            or result_path.is_symlink()
        ):
            raise FileExistsError(
                "confirmatory output/result must both be absent"
            )
        required_bindings.update(
            {
                str(manifest_path.relative_to(ROOT)),
                str(release_path.relative_to(ROOT)),
                str(audit_path.relative_to(ROOT)),
                str(runner_contract_path.relative_to(ROOT)),
            }
        )
        source_paths.add(source_path)
        spec.update(
            {
                "fixture_root": str(fixture_root),
                "fixture_manifest_path": str(manifest_path),
                "release_path": str(release_path),
                "source_path": str(source_path),
                "output_path": str(output_path),
                "result_path": str(result_path),
                "_audit_path": str(audit_path),
                "_runner_contract_path": str(runner_contract_path),
            }
        )
        prepared.append(spec)
    identity_families = {
        "fixture roots": [
            spec["fixture_root"] for spec in prepared
        ],
        "source paths": [
            spec["source_path"] for spec in prepared
        ],
        "source hashes": [
            spec["source_sha256"] for spec in prepared
        ],
        "release paths": [
            spec["release_path"] for spec in prepared
        ],
        "output paths": [
            spec["output_path"] for spec in prepared
        ],
        "result paths": [
            spec["result_path"] for spec in prepared
        ],
    }
    for name, values in identity_families.items():
        if len(set(values)) != len(values):
            raise ValueError(
                f"confirmatory pair has duplicate {name}"
            )
    fixture_roots = [
        Path(spec["fixture_root"]) for spec in prepared
    ]
    if any(
        _is_relative_to(left, right)
        or _is_relative_to(right, left)
        for index, left in enumerate(fixture_roots)
        for right in fixture_roots[index + 1 :]
    ):
        raise ValueError("confirmatory fixture roots may not be nested")
    pair_root = _guard_fixture_root(freeze["pair_result_root"])
    if (
        pair_root.is_symlink()
        or not pair_root.is_dir()
        or any(pair_root.iterdir())
    ):
        raise ValueError(
            "confirmatory registry root must preexist and be empty"
        )
    if any(
        _is_relative_to(pair_root, fixture_root)
        or _is_relative_to(fixture_root, pair_root)
        for fixture_root in fixture_roots
    ):
        raise ValueError(
            "confirmatory registry and fixture roots must be separate"
        )
    attempt_path = pair_root / "CONFIRMATORY_ATTEMPT_STARTED.json"
    pair_result_path = pair_root / "confirmatory_pair_result.json"
    completion_path = pair_root / "CONFIRMATORY_ATTEMPT_COMPLETED.json"
    if set(bindings) != required_bindings:
        raise ValueError(
            "confirmatory binding family must be exact"
        )
    verified_paths: dict[str, Path] = {}
    for path_value in sorted(required_bindings):
        expected = bindings.get(path_value)
        if HEX64.fullmatch(str(expected)) is None:
            raise ValueError("confirmatory binding digest is invalid")
        path = _project_regular_file(path_value)
        if path in source_paths:
            raise PermissionError(
                "confirmatory source may not be a preflight binding"
            )
        if _bootstrap_sha256_file(path) != expected:
            raise ValueError(
                f"confirmatory frozen binding changed: {path_value}"
            )
        verified_paths[path_value] = path
    for path_value in (
        "smc_trader/artifact_stream.py",
        "scripts/benchmark_v3_semantic_pass1_e2e.py",
        "smc_trader/semantic_discovery_runner.py",
    ):
        path = verified_paths[path_value]
        if sha256_file(path) != _bootstrap_sha256_file(path):
            raise ValueError(
                "artifact_stream/bootstrap hash implementations disagree"
            )
    checked: list[dict[str, Any]] = []
    for spec in prepared:
        fixture_root = Path(spec["fixture_root"])
        manifest_path = Path(spec["fixture_manifest_path"])
        release_path = Path(spec["release_path"])
        source_path = Path(spec["source_path"])
        output_path = Path(spec["output_path"])
        result_path = Path(spec["result_path"])
        audit_path = Path(spec.pop("_audit_path"))
        runner_contract_path = Path(spec.pop("_runner_contract_path"))
        spec["release_sha256"] = bindings[
            str(release_path.relative_to(ROOT))
        ]
        manifest = _read_json(manifest_path)
        release = _read_json(release_path)
        audit = _read_json(audit_path)
        runner_contract = _read_json(runner_contract_path)
        expected_synthetic_bars, expected_reset_epochs = (
            _confirmatory_gap_expectations(manifest)
        )
        fixture_issued_at = pd.Timestamp(manifest.get("issued_at"))
        if (
            manifest.get("pattern") != spec["pattern"]
            or int(manifest.get("source_rows", 0))
            != CONFIRMATORY_GATES["source_rows"]
            or manifest.get("source_path") != str(source_path)
            or manifest.get("source_sha256")
            != spec["source_sha256"]
            or manifest.get("confirmatory_one_time") is not True
            or fixture_issued_at.tzinfo is None
            or fixture_issued_at.tz_convert("UTC")
            > issued_at.tz_convert("UTC")
            or (
                expiry.tz_convert("UTC")
                - fixture_issued_at.tz_convert("UTC")
            ).total_seconds()
            > CONFIRMATORY_GATES[
                "maximum_release_window_seconds"
            ]
            or manifest.get("expires_at") != freeze["expires_at"]
            or manifest.get("confirmatory_attempt_id") != attempt_id
            or manifest.get("confirmatory_registry_root")
            != str(pair_root)
            or manifest.get("confirmatory_attempt_marker_path")
            != str(attempt_path)
        ):
            raise ValueError("confirmatory fixture manifest changed")
        if (
            audit.get("artifact")
            != "v3_synthetic_full_e2e_audit_contract"
            or audit.get("status") != "synthetic_test_contract"
            or audit.get("authorization")
            != "synthetic integration tests only"
            or audit.get("source", {}).get("window_role")
            != "synthetic_fixture"
            or audit.get("source", {}).get("path") != str(source_path)
            or audit.get("bindings", {}).get("causal_source_sha256")
            != spec["source_sha256"]
            or runner_contract.get("artifact")
            != "v3_synthetic_full_e2e_runner_contract"
            or runner_contract.get("status")
            != "implementation_contract_frozen"
            or runner_contract.get("authorization")
            != (
                "implementation and synthetic tests only; real discovery "
                "execution is forbidden"
            )
            or runner_contract.get("bindings", {}).get(
                "blind_audit_contract_sha256"
            )
            != sha256_file(audit_path)
            or set(
                runner_contract.get(
                    "implementation_hashes_required", []
                )
            )
            != set(IMPLEMENTATION_FILES)
        ):
            raise ValueError(
                "confirmatory synthetic contract family changed"
            )
        if (
            release.get("artifact")
            != "v3_synthetic_full_e2e_pass1_release"
            or release.get("authorization")
            != "synthetic pass1 confirmatory one-time"
            or release.get("status") != "synthetic_test_release"
            or release.get("confirmatory_one_time") is not True
            or release.get("execution_authorized") is not True
            or release.get("authorized_phases") != ["pass1"]
            or release.get("issued_at") != manifest.get("issued_at")
            or release.get("expires_at") != freeze["expires_at"]
            or release.get("confirmatory")
            != {
                "attempt_id": attempt_id,
                "registry_root": str(pair_root),
                "attempt_marker_path": str(attempt_path),
            }
            or release.get("source", {}).get("window_role")
            != "synthetic_fixture"
            or release.get("source", {}).get("path") != str(source_path)
            or release.get("bindings", {}).get(
                "causal_source_sha256"
            )
            != spec["source_sha256"]
            or release.get("output", {}).get("path")
            != str(output_path)
            or release.get("run", {}).get(
                "diagnostic_stop_allowed"
            )
            is not False
            or int(
                release.get("run", {}).get(
                    "checkpoint_source_rows", 0
                )
            )
            != 25_000
            or release.get("resources")
            != {
                "rss_hard_ceiling_bytes": (
                    CONFIRMATORY_GATES[
                        "rss_bytes_strictly_less_than"
                    ]
                ),
                "disk_free_floor_bytes": (
                    CONFIRMATORY_GATES["disk_free_bytes_at_least"]
                ),
                "checkpoint_stall_timeout_seconds": (
                    CONFIRMATORY_GATES[
                        "checkpoint_stall_timeout_seconds"
                    ]
                ),
                "resource_check_source_rows": (
                    CONFIRMATORY_GATES[
                        "resource_check_source_rows"
                    ]
                ),
            }
        ):
            raise ValueError("confirmatory one-time release changed")
        spec.update(
            {
                "expected_synthetic_bars": (
                    expected_synthetic_bars
                ),
                "expected_reset_epochs": expected_reset_epochs,
                "expected_produced_bars": (
                    CONFIRMATORY_GATES["source_rows"]
                    + expected_synthetic_bars
                ),
            }
        )
        checked.append(spec)
    freeze["_verified_freeze_sha256"] = args.expected_freeze_sha256
    freeze["_verified_preregistration_sha256"] = (
        args.expected_preregistration_sha256
    )
    freeze["_pair_result_path"] = str(pair_result_path)
    freeze["_registry_root"] = str(pair_root)
    freeze["_attempt_path"] = str(attempt_path)
    freeze["_completion_path"] = str(completion_path)
    return freeze, tuple(checked)


def _confirmatory_result_errors(
    result: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    effective_wall_seconds: float,
) -> list[str]:
    errors: list[str] = []
    terminal = result.get("terminal")
    if not isinstance(terminal, dict):
        return ["terminal QA result is absent"]
    if result.get("observed_outcome") != "complete":
        errors.append("Pass1 outcome is not complete")
    if result.get("resume") is not False:
        errors.append("confirmatory run used resume")
    if int(result.get("diagnostic_stop_after_source_rows", -1)) != 0:
        errors.append("confirmatory run used a diagnostic stop")
    if effective_wall_seconds >= CONFIRMATORY_GATES[
        "wall_seconds_strictly_less_than"
    ]:
        errors.append("effective wall-time gate failed")
    if int(result.get("rss_peak_bytes", 0)) >= CONFIRMATORY_GATES[
        "rss_bytes_strictly_less_than"
    ]:
        errors.append("RSS gate failed")
    if int(result.get("disk_free_min_bytes", 0)) < CONFIRMATORY_GATES[
        "disk_free_bytes_at_least"
    ]:
        errors.append("disk-free gate failed")
    expected_rows = CONFIRMATORY_GATES["source_rows"]
    if (
        int(result.get("source_rows", -1)) != expected_rows
        or int(result.get("admitted_source_rows", -1)) != expected_rows
        or int(result.get("session_source_rows", -1)) != expected_rows
        or int(terminal.get("source_rows", -1)) != expected_rows
        or int(terminal.get("real_bars", -1)) != expected_rows
    ):
        errors.append("source-row gate failed")
    expected_synthetic_bars = spec.get("expected_synthetic_bars")
    expected_reset_epochs = spec.get("expected_reset_epochs")
    expected_produced_bars = spec.get("expected_produced_bars")
    if (
        type(expected_synthetic_bars) is not int
        or expected_synthetic_bars <= 0
        or type(expected_reset_epochs) is not int
        or expected_reset_epochs <= 0
        or type(expected_produced_bars) is not int
        or expected_produced_bars
        != expected_rows + expected_synthetic_bars
    ):
        errors.append("confirmatory gap plan gate failed")
    if (
        type(terminal.get("synthetic_bars")) is not int
        or terminal.get("synthetic_bars")
        != expected_synthetic_bars
    ):
        errors.append("synthetic no-trade coverage gate failed")
    if (
        type(terminal.get("reset_epoch")) is not int
        or terminal.get("reset_epoch") != expected_reset_epochs
    ):
        errors.append("data-gap reset coverage gate failed")
    if (
        type(terminal.get("produced_bars")) is not int
        or terminal.get("produced_bars") != expected_produced_bars
    ):
        errors.append("produced-bar count gate failed")
    if (
        result.get("checkpoint_status") != "valid"
        or HEX64.fullmatch(
            str(result.get("checkpoint_state_sha256", ""))
        )
        is None
    ):
        errors.append("checkpoint gate failed")
    statuses = result.get("observed_progress_statuses", [])
    terminal_status = terminal.get("progress_status")
    minimum_percent = result.get("observed_progress_min_percent")
    maximum_percent = result.get("observed_progress_max_percent")
    if (
        not isinstance(statuses, list)
        or "running" not in statuses
        or terminal_status not in statuses
        or int(result.get("monitor_sample_count", 0)) <= 0
        or not isinstance(minimum_percent, (int, float))
        or float(minimum_percent) < 0.0
        or not isinstance(maximum_percent, (int, float))
        or float(maximum_percent) != 100.0
        or float(terminal.get("completed_percent", -1.0)) != 100.0
    ):
        errors.append("progress gate failed")
    selected_case_count = terminal.get("selected_case_count")
    expected_case_count = terminal.get("expected_case_count")
    missing_bucket_count = terminal.get("missing_bucket_count")
    if (
        terminal.get("selection_status") != "complete"
        or terminal.get("progress_status") != "complete"
        or type(expected_case_count) is not int
        or expected_case_count <= 0
        or type(selected_case_count) is not int
        or selected_case_count != expected_case_count
        or type(missing_bucket_count) is not int
        or missing_bucket_count != 0
        or terminal.get("terminal_reconciled") is not True
        or terminal.get("terminal_source_guard")
        != "FrozenParquetSourceIdentity"
        or int(terminal.get("terminal_source_method_calls", -1)) != 0
    ):
        errors.append("terminal selection/source-free QA gate failed")
    if (
        HEX64.fullmatch(str(terminal.get("source_prefix_root", "")))
        is None
        or HEX64.fullmatch(
            str(terminal.get("produced_bar_prefix_root", ""))
        )
        is None
    ):
        errors.append("rolling commitment gate failed")
    output = Path(str(spec["output_path"]))
    if (
        not output.is_dir()
        or {path.name for path in output.iterdir()}
        != {
            "_checkpoint",
            "progress.json",
            "selection_authority.json",
        }
    ):
        errors.append("phase-exact output-tree gate failed")
    return errors


def confirm_pair(args: argparse.Namespace) -> dict[str, Any]:
    pair_started = time.perf_counter()
    freeze, fixtures = _confirmatory_preflight(args)
    _write_new_json(
        Path(freeze["_attempt_path"]),
        {
            "format_version": 1,
            "artifact": "v3_synthetic_full_e2e_attempt_started",
            "attempt_id": freeze["attempt_id"],
            "freeze_sha256": freeze["_verified_freeze_sha256"],
            "preregistration_sha256": freeze[
                "_verified_preregistration_sha256"
            ],
            "patterns": ["light", "event_heavy"],
            "fixtures": [
                {
                    "pattern": spec["pattern"],
                    "release_sha256": spec["release_sha256"],
                    "source_sha256": spec["source_sha256"],
                    "output_path": spec["output_path"],
                }
                for spec in fixtures
            ],
            "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "one_time_attempt_consumed": True,
        },
        require_existing_parent=True,
    )
    preflight_seconds = time.perf_counter() - pair_started
    results: list[dict[str, Any]] = []
    all_errors: list[str] = []
    for spec in fixtures:
        run_args = argparse.Namespace(
            fixture_root=spec["fixture_root"],
            release=spec["release_path"],
            result=spec["result_path"],
            resume=False,
            diagnostic_stop_after_source_rows=0,
            expected_outcome="complete",
            monitor_sample_seconds=0.05,
            confirmatory_mode=True,
        )
        result = run_benchmark(run_args)
        effective_wall = preflight_seconds + float(
            result["wall_seconds"]
        )
        errors = _confirmatory_result_errors(
            result,
            spec=spec,
            effective_wall_seconds=effective_wall,
        )
        result_summary = {
            "pattern": spec["pattern"],
            "result_path": result["result_path"],
            "result_sha256": sha256_file(result["result_path"]),
            "run_wall_seconds": result["wall_seconds"],
            "effective_wall_seconds_including_pair_preflight": (
                effective_wall
            ),
            "rss_peak_bytes": result["rss_peak_bytes"],
            "disk_free_min_bytes": result["disk_free_min_bytes"],
            "source_prefix_root": result["terminal"][
                "source_prefix_root"
            ],
            "produced_bar_prefix_root": result["terminal"][
                "produced_bar_prefix_root"
            ],
            "selection_status": result["terminal"][
                "selection_status"
            ],
            "selected_case_count": result["terminal"][
                "selected_case_count"
            ],
            "expected_case_count": result["terminal"][
                "expected_case_count"
            ],
            "missing_bucket_count": result["terminal"][
                "missing_bucket_count"
            ],
            "planned_short_open_gaps": spec[
                "expected_synthetic_bars"
            ],
            "actual_synthetic_bars": result["terminal"][
                "synthetic_bars"
            ],
            "planned_long_open_gaps": spec[
                "expected_reset_epochs"
            ],
            "actual_reset_epochs": result["terminal"][
                "reset_epoch"
            ],
            "expected_produced_bars": spec[
                "expected_produced_bars"
            ],
            "actual_produced_bars": result["terminal"][
                "produced_bars"
            ],
            "errors": errors,
        }
        results.append(result_summary)
        all_errors.extend(
            f"{spec['pattern']}: {value}" for value in errors
        )
    maximum_wall = max(
        float(item["effective_wall_seconds_including_pair_preflight"])
        for item in results
    )
    pair_result = {
        "format_version": 1,
        "artifact": "v3_synthetic_full_e2e_confirmatory_pair_result",
        "mode": "synthetic_causal_no_economic",
        "status": "pass" if not all_errors else "fail",
        "freeze_sha256": freeze["_verified_freeze_sha256"],
        "preregistration_sha256": freeze[
            "_verified_preregistration_sha256"
        ],
        "preflight_seconds": preflight_seconds,
        "patterns": results,
        "maximum_effective_wall_seconds": maximum_wall,
        "hard_gates": CONFIRMATORY_GATES,
        "errors": all_errors,
        "runtime": _runtime_fingerprint(),
        "real_data_authority": False,
        "pass2_authority": False,
        "economic_authority": False,
        "mbo_authority": False,
    }
    pair_result_path = Path(freeze["_pair_result_path"])
    _write_new_json(
        pair_result_path,
        pair_result,
        require_existing_parent=True,
    )
    _write_new_json(
        Path(freeze["_completion_path"]),
        {
            "format_version": 1,
            "artifact": "v3_synthetic_full_e2e_attempt_completed",
            "status": pair_result["status"],
            "pair_result_sha256": sha256_file(pair_result_path),
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        require_existing_parent=True,
    )
    if all_errors:
        raise RuntimeError(
            "synthetic full-E2E confirmatory pair failed hard gates: "
            + "; ".join(all_errors)
        )
    return {
        **pair_result,
        "pair_result_path": str(pair_result_path),
        "pair_result_sha256": sha256_file(pair_result_path),
    }


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry_parser = subparsers.add_parser("prepare-registry")
    registry_parser.add_argument("--registry-root", required=True)
    registry_parser.set_defaults(handler=prepare_registry)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--fixture-root", required=True)
    prepare_parser.add_argument("--source-rows", type=_positive, required=True)
    prepare_parser.add_argument("--first-trade-date", required=True)
    prepare_parser.add_argument(
        "--pattern",
        choices=sorted(ALLOWED_PATTERNS),
        required=True,
    )
    prepare_parser.add_argument(
        "--row-group-rows",
        type=_positive,
        default=65_536,
    )
    prepare_parser.add_argument(
        "--batch-rows",
        type=_positive,
        default=65_536,
    )
    prepare_parser.add_argument(
        "--maximum-history",
        type=_positive,
        default=1_024,
    )
    prepare_parser.add_argument(
        "--checkpoint-source-rows",
        type=_positive,
        default=25_000,
    )
    prepare_parser.add_argument(
        "--rss-hard-ceiling-bytes",
        type=_positive,
        default=4 * 1024**3,
    )
    prepare_parser.add_argument(
        "--disk-free-floor-bytes",
        type=_positive,
        default=1024**3,
    )
    prepare_parser.add_argument(
        "--checkpoint-stall-timeout-seconds",
        type=float,
        default=300.0,
    )
    prepare_parser.add_argument(
        "--resource-check-source-rows",
        type=_positive,
        default=5_000,
    )
    prepare_parser.add_argument("--release-name", required=True)
    prepare_parser.add_argument("--output-name", required=True)
    prepare_parser.add_argument(
        "--confirmatory-one-time",
        action="store_true",
    )
    prepare_parser.add_argument("--confirmatory-attempt-id")
    prepare_parser.add_argument("--confirmatory-registry-root")
    prepare_parser.add_argument(
        "--expires-at",
        default="2099-01-01T00:00:00+00:00",
    )
    prepare_parser.set_defaults(handler=prepare)

    release_parser = subparsers.add_parser("create-release")
    release_parser.add_argument("--fixture-root", required=True)
    release_parser.add_argument("--release-name", required=True)
    release_parser.add_argument("--output-name", required=True)
    release_parser.set_defaults(handler=create_release)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--fixture-root", required=True)
    run_parser.add_argument("--release", required=True)
    run_parser.add_argument("--result", required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument(
        "--diagnostic-stop-after-source-rows",
        type=int,
        default=0,
    )
    run_parser.add_argument(
        "--expected-outcome",
        choices=("complete", "interrupt"),
        required=True,
    )
    run_parser.add_argument(
        "--monitor-sample-seconds",
        type=float,
        default=0.25,
    )
    run_parser.set_defaults(handler=run_benchmark)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--fixture-root", required=True)
    compare_parser.add_argument("--left-output", required=True)
    compare_parser.add_argument("--right-output", required=True)
    compare_parser.add_argument("--result", required=True)
    compare_parser.set_defaults(handler=compare_outputs)

    runtime_parser = subparsers.add_parser("runtime-fingerprint")
    runtime_parser.set_defaults(handler=runtime_fingerprint_command)

    confirm_parser = subparsers.add_parser("confirm-pair")
    confirm_parser.add_argument("--freeze", required=True)
    confirm_parser.add_argument(
        "--expected-freeze-sha256",
        required=True,
    )
    confirm_parser.add_argument("--preregistration", required=True)
    confirm_parser.add_argument(
        "--expected-preregistration-sha256",
        required=True,
    )
    confirm_parser.set_defaults(handler=confirm_pair)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    print(
        json.dumps(
            parsed.handler(parsed),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
