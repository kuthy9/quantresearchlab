"""Resumable two-pass discovery for the frozen v3 structure/BOS audit.

The runner intentionally imports no brain, decision, risk, portfolio, shadow
replay, outcome, PnL, or MBO code.  Its causal guarantee is *logical row
admission*: only completed bars yielded in order are admitted to the reader and
observer.  It does not claim that a Parquet implementation never decodes later
bytes from the same row group.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import pickle
import re
import shutil
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from .artifact_stream import sha256_file
from .causal import CausalMarketReader
from .io import iter_completed_bars
from .model import (
    BOSLifecycle,
    Bar,
    Candle,
    CORE_TIMEFRAMES,
    MarketObservation,
    Timeframe,
    content_hash,
    to_primitive,
)
from .observation import CausalObserver, ExecutionRealityInput, ObserverConfig
from .semantic_audit import (
    SemanticCase,
    classify_bos_case,
    materialize_blind_review_export,
    materialize_blind_unit,
    selection_score,
    semantic_case_context_complete,
    semantic_event_id,
)


RUNNER_CONTRACT = Path(
    "configs/experiments/"
    "EXP-SMC-3.0.1-001-SEMANTIC-DISCOVERY-RUNNER-R3.json"
)
PASS1_EXECUTION_RELEASE = Path(
    "configs/experiments/"
    "EXP-SMC-3.0.1-001-SEMANTIC-DISCOVERY-PASS1-EXECUTION-R3.json"
)
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_FORMAT_VERSION = 1
HEX64 = re.compile(r"[0-9a-f]{64}")
PROVENANCE_REAL = "real_source_bar"
PROVENANCE_SYNTHETIC = "synthetic_backfill"
CONFIRMATORY_MAX_RELEASE_SECONDS = 24 * 60 * 60


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _chain(previous: str, value: Any) -> str:
    if HEX64.fullmatch(previous) is None:
        raise ValueError("rolling-root parent is not SHA-256")
    payload = _canonical_json(value)
    digest = hashlib.sha256()
    parent = previous.encode("ascii")
    digest.update(len(parent).to_bytes(8, "big"))
    digest.update(parent)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _chain_primitive_mapping(
    previous: str,
    value: Mapping[str, Any],
) -> str:
    """Strictly equivalent fast path for an already primitive mapping."""

    if HEX64.fullmatch(previous) is None:
        raise ValueError("rolling-root parent is not SHA-256")
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    parent = previous.encode("ascii")
    digest.update(len(parent).to_bytes(8, "big"))
    digest.update(parent)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale atomic-write temporary exists: {temporary}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a complete create-once JSON file.

    The hard-link step is an atomic no-replace operation on the same
    filesystem.  A crash can leave an unreferenced temporary, but can never
    expose a partially written authority/global manifest at ``path``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def load_runner_contract(
    path: str | Path = RUNNER_CONTRACT,
    *,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "implementation_contract_frozen":
        raise ValueError("semantic discovery runner contract is not frozen")
    if "implementation and synthetic tests only" not in str(
        payload.get("authorization", "")
    ):
        raise ValueError("runner contract lacks the implementation-only gate")
    iterator = payload.get("source_iterator", {})
    if (
        int(iterator.get("maximum_no_trade_gap_minutes", -1)) != 5
        or iterator.get("allow_data_gap_reset") is not True
    ):
        raise ValueError("runner gap policy changed")
    bindings = payload.get("bindings", {})
    if any(HEX64.fullmatch(str(value)) is None for value in bindings.values()):
        raise ValueError("runner binding is not a SHA-256")
    if verify_bound_files:
        required_file_keys = {
            "blind_audit_contract_sha256",
            "model_config_sha256",
            "primitive_protocol_sha256",
            "io_code_sha256",
            "causal_reader_code_sha256",
            "market_clock_code_sha256",
            "semantic_audit_code_sha256",
        }
        files = payload.get("bound_files")
        if (
            not isinstance(files, Mapping)
            or set(files) != required_file_keys
        ):
            raise ValueError(
                "semantic runner bound-file family changed"
            )
        for name, relative in files.items():
            if bindings.get(name) != sha256_file(ROOT / relative):
                raise ValueError(f"runner frozen binding changed: {name}")
    return payload


def runtime_environment() -> dict[str, str]:
    """Return the exact runtime family frozen by a Pass1 release."""

    import pyarrow
    import psutil

    try:
        import tzdata

        timezone_data = str(tzdata.__version__)
    except ImportError:
        timezone_data = "system-zoneinfo"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pandas_version": str(pd.__version__),
        "pyarrow_version": str(pyarrow.__version__),
        "tzdata_version": timezone_data,
        "psutil_version": str(psutil.__version__),
    }


def load_pass1_execution_release(
    path: str | Path = PASS1_EXECUTION_RELEASE,
    *,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    """Load a phase-scoped release without opening its bound market source."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") not in {
        "draft_no_execution_authority",
        "execution_contract_frozen",
    }:
        raise ValueError("unsupported Pass1 execution-release status")
    if payload.get("authorized_phases") != ["pass1"]:
        raise ValueError("Pass1 release may authorize only pass1")
    if payload.get("execution_authorized") is not (
        payload.get("status") == "execution_contract_frozen"
    ):
        raise ValueError("Pass1 release status/authority is inconsistent")
    if not str(payload.get("release_id", "")).strip():
        raise ValueError("Pass1 release id is absent")
    bindings = payload.get("bindings", {})
    if any(
        HEX64.fullmatch(str(value)) is None
        for value in bindings.values()
    ):
        raise ValueError("Pass1 release binding is not a SHA-256")
    required_bound_files = {
        "audit_contract_sha256",
        "engineering_runner_contract_sha256",
        "semantic_discovery_runner_sha256",
        "pass1_cli_sha256",
        "engineering_cli_sha256",
        "model_config_sha256",
        "primitive_protocol_sha256",
        "io_code_sha256",
        "causal_reader_code_sha256",
        "market_clock_code_sha256",
        "semantic_audit_code_sha256",
        "observer_code_sha256",
        "structure_code_sha256",
        "pyproject_sha256",
        "previous_runner_validation_report_sha256",
    }
    if set(bindings) != required_bound_files | {"causal_source_sha256"}:
        raise ValueError("Pass1 release binding family changed")
    bound_files = payload.get("bound_files", {})
    if set(bound_files) != required_bound_files:
        raise ValueError("Pass1 release bound-file family changed")
    if "causal_source_sha256" in bound_files:
        raise ValueError("Pass1 release loader may not bind source as a file")
    source_binding = payload.get("source", {})
    if (
        not str(source_binding.get("path", "")).strip()
        or source_binding.get("window_role") != "semantic_discovery"
        or _aware(
            source_binding.get("end_exclusive"),
            name="release.source.end_exclusive",
        )
        <= _aware(
            source_binding.get("start"),
            name="release.source.start",
        )
    ):
        raise ValueError("Pass1 release source window is invalid")
    run = payload.get("run", {})
    if (
        int(run.get("batch_rows", 0)) != 65_536
        or int(run.get("maximum_history", 0)) != 1_024
        or int(run.get("maximum_no_trade_gap_minutes", -1)) != 5
        or run.get("allow_data_gap_reset") is not True
        or int(run.get("checkpoint_source_rows", 0)) != 25_000
        or run.get("real_only_case_anchors") is not True
        or run.get("diagnostic_stop_allowed") is not False
    ):
        raise ValueError("Pass1 release run parameters changed")
    limits = payload.get("resources", {})
    ExecutionResourceLimits.from_mapping(limits)
    if payload.get("runtime") != runtime_environment():
        raise ValueError("Pass1 release runtime environment changed")
    release_id = str(payload["release_id"])
    parent_experiment_id = str(
        payload.get("parent_experiment_id", "")
    ).strip()
    if not parent_experiment_id:
        raise ValueError("Pass1 release parent experiment is absent")
    expected_output = (
        f"artifacts/semantic_discovery/{parent_experiment_id}/pass1/"
        f"{release_id}"
    )
    output = payload.get("output", {})
    if (
        output.get("path") != expected_output
        or output.get("new_run_requires_absent_non_symlink_path")
        is not True
        or output.get("allowed_top_level_entries")
        != [
            "_checkpoint",
            "progress.json",
            "selection_authority.json",
        ]
    ):
        raise ValueError("Pass1 release output contract changed")
    permissions = payload.get("permissions", {})
    required_permissions = {
        "real_pass1_authorized",
        "real_pass2_authorized",
        "case_materialization_authorized",
        "discovery_data_authorized",
        "sealed_data_authorized",
        "future_reveal_authorized",
        "economic_tests_authorized",
        "mbo_authorized",
    }
    if set(permissions) != required_permissions:
        raise ValueError("Pass1 release permission family changed")
    forbidden_permissions = {
        "real_pass2_authorized",
        "case_materialization_authorized",
        "sealed_data_authorized",
        "future_reveal_authorized",
        "economic_tests_authorized",
        "mbo_authorized",
    }
    if any(permissions.get(name) is not False for name in forbidden_permissions):
        raise ValueError("Pass1 release grants a forbidden permission")
    if payload["status"] == "draft_no_execution_authority":
        if any(value is not False for value in permissions.values()):
            raise ValueError("draft Pass1 release grants data authority")
    elif (
        permissions.get("real_pass1_authorized") is not True
        or permissions.get("discovery_data_authorized") is not True
        or payload.get("authorization")
        != "one-time real semantic discovery pass1 authorized"
    ):
        raise ValueError("frozen Pass1 release lacks exact data authority")
    expiry = _aware(payload.get("expires_at"), name="release.expires_at")
    if payload.get("status") == "execution_contract_frozen":
        now = pd.Timestamp.now(tz="UTC")
        if expiry.tz_convert("UTC") <= now:
            raise ValueError("Pass1 execution release is expired")
    if verify_bound_files:
        for name, relative in bound_files.items():
            expected = bindings.get(name)
            if expected != sha256_file(ROOT / str(relative)):
                raise ValueError(
                    f"Pass1 release bound file changed: {name}"
                )
    return payload


@dataclass(frozen=True)
class ExecutionResourceLimits:
    rss_hard_ceiling_bytes: int
    disk_free_floor_bytes: int
    checkpoint_stall_timeout_seconds: float
    resource_check_source_rows: int

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "ExecutionResourceLimits":
        value = cls(
            rss_hard_ceiling_bytes=int(
                payload.get("rss_hard_ceiling_bytes", 0)
            ),
            disk_free_floor_bytes=int(
                payload.get("disk_free_floor_bytes", 0)
            ),
            checkpoint_stall_timeout_seconds=float(
                payload.get("checkpoint_stall_timeout_seconds", 0)
            ),
            resource_check_source_rows=int(
                payload.get("resource_check_source_rows", 0)
            ),
        )
        if (
            value.rss_hard_ceiling_bytes <= 0
            or value.disk_free_floor_bytes <= 0
            or value.checkpoint_stall_timeout_seconds <= 0
            or value.resource_check_source_rows <= 0
        ):
            raise ValueError("Pass1 resource limits must be positive")
        return value


def validate_pass1_release_output_tree(
    output_root: str | Path,
    *,
    resume: bool,
) -> None:
    """Reject any output tree outside the one-phase release protocol."""

    root = Path(output_root)
    anchor = root
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if anchor.is_symlink() or not anchor.is_dir():
        raise ValueError(
            "execution output has an untrusted filesystem ancestor"
        )
    if root.is_symlink():
        raise ValueError("execution output root may not be a symlink")
    if not resume:
        if root.exists():
            raise FileExistsError(
                "execution output must be absent for a new run"
            )
        return
    if not root.is_dir():
        raise ValueError(
            "resume output must be an existing regular directory"
        )
    observed = {path.name for path in root.iterdir()}
    running = {"_checkpoint", "progress.json"}
    completed = running | {"selection_authority.json"}
    if frozenset(observed) not in {
        frozenset(running),
        frozenset(completed),
    }:
        raise ValueError("resume output contains an unregistered entry")
    checkpoint_root = root / "_checkpoint"
    if (
        checkpoint_root.is_symlink()
        or not checkpoint_root.is_dir()
        or {path.name for path in checkpoint_root.iterdir()}
        != {"pass1"}
    ):
        raise ValueError("resume checkpoint tree is not phase-exact")
    forbidden_names = {
        "packets",
        "PASS2_MANIFEST.json",
        "authority",
        "blind",
    }
    if any(
        path.name in forbidden_names
        for path in root.rglob("*")
    ):
        raise ValueError("Pass1 output contains a forbidden artifact")


@dataclass(frozen=True)
class SourceChunk:
    """A bounded normalized source frame plus immutable physical row ordinals."""

    frame: pd.DataFrame
    ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.frame) != len(self.ordinals):
            raise ValueError("source chunk row/ordinal counts differ")
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise ValueError("source chunk requires a DatetimeIndex")
        if self.frame.index.tz is None:
            raise ValueError("source chunk timestamps must be timezone-aware")
        if not self.frame.index.is_monotonic_increasing:
            raise ValueError("source chunk is not chronological")
        if self.frame.index.has_duplicates:
            raise ValueError("source chunk has duplicate timestamps")
        if any(
            right <= left
            for left, right in zip(self.ordinals[:-1], self.ordinals[1:])
        ):
            raise ValueError("source ordinals are not strictly increasing")


@dataclass(frozen=True)
class CompletedBarReceipt:
    """A completed bar with internally derived source-row provenance."""

    bar: Bar
    provenance: str
    source_row_start: pd.Timestamp
    source_row_ordinal: int
    source_row_sha256: str
    bar_sha256: str
    max_source_row_admitted: pd.Timestamp

    @property
    def real_source_bar(self) -> bool:
        return self.provenance == PROVENANCE_REAL

    def __post_init__(self) -> None:
        if self.provenance not in {
            PROVENANCE_REAL,
            PROVENANCE_SYNTHETIC,
        }:
            raise ValueError("unknown completed-bar provenance")
        source_start = _aware(
            self.source_row_start,
            name="source_row_start",
        )
        admitted = _aware(
            self.max_source_row_admitted,
            name="max_source_row_admitted",
        )
        if admitted != source_start:
            raise ValueError("receipt source cutoff is not internally exact")
        if self.source_row_ordinal < 0:
            raise ValueError("source row ordinal must be nonnegative")
        if (
            HEX64.fullmatch(self.source_row_sha256) is None
            or HEX64.fullmatch(self.bar_sha256) is None
        ):
            raise ValueError("receipt hashes must be SHA-256")
        if self.real_source_bar:
            if self.bar.synthetic_no_trade or self.bar.start != source_start:
                raise ValueError("real provenance disagrees with completed bar")
            if source_start >= self.bar.end:
                raise ValueError("real row is not before its observation clock")
        elif not self.bar.synthetic_no_trade:
            raise ValueError("synthetic provenance requires a synthetic bar")


def _source_bar(frame: pd.DataFrame, position: int) -> Bar:
    timestamp = frame.index[position]
    row = frame.iloc[position]
    return Bar(
        start=timestamp,
        open=float(row.open),
        high=float(row.high),
        low=float(row.low),
        close=float(row.close),
        volume=float(row.volume),
        symbol=str(row.symbol),
        instrument_id=int(row.instrument_id),
    )


def _source_bars(frame: pd.DataFrame) -> tuple[Bar, ...]:
    """Construct source bars without allocating one pandas Series per row."""

    columns = frame[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
            "instrument_id",
        ]
    ]
    return tuple(
        Bar(
            start=timestamp,
            open=float(open_price),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
            symbol=str(symbol),
            instrument_id=int(instrument_id),
        )
        for timestamp, (
            open_price,
            high,
            low,
            close,
            volume,
            symbol,
            instrument_id,
        ) in zip(
            frame.index,
            columns.itertuples(index=False, name=None),
        )
    )


def source_row_sha256(bar: Bar, ordinal: int) -> str:
    if bar.synthetic_no_trade or bar.data_gap_before_minutes:
        raise ValueError("source-row hash requires an unmodified real row")
    payload = json.dumps(
        {
            "source_row_ordinal": int(ordinal),
            "start": bar.start.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "symbol": bar.symbol,
            "instrument_id": bar.instrument_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bar_sha256(bar: Bar) -> str:
    """Return the exact ``content_hash(Bar)`` without dataclass recursion."""

    payload = json.dumps(
        {
            "start": bar.start.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "symbol": bar.symbol,
            "instrument_id": bar.instrument_id,
            "synthetic_no_trade": bar.synthetic_no_trade,
            "data_gap_before_minutes": bar.data_gap_before_minutes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _single_source_frame(bar: Bar) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [bar.open],
            "high": [bar.high],
            "low": [bar.low],
            "close": [bar.close],
            "volume": [bar.volume],
            "symbol": [bar.symbol],
            "instrument_id": [bar.instrument_id],
        },
        index=pd.DatetimeIndex([bar.start], name="ts"),
    )


def iter_provenanced_completed_bars(
    chunks: Iterable[SourceChunk],
    *,
    prior_source_bar: Bar | None = None,
    prior_source_ordinal: int | None = None,
    maximum_no_trade_gap_minutes: int = 5,
    allow_data_gap_reset: bool = True,
) -> Iterator[CompletedBarReceipt]:
    """Preserve ``iter_completed_bars`` exactly across bounded source chunks."""

    prior_bar = prior_source_bar
    prior_ordinal = prior_source_ordinal
    if (prior_bar is None) != (prior_ordinal is None):
        raise ValueError("resume prior row and ordinal must be supplied together")
    for chunk in chunks:
        if chunk.frame.empty:
            continue
        if prior_bar is not None:
            if chunk.frame.index[0] <= prior_bar.start:
                raise ValueError("resumed source did not advance beyond checkpoint")
            combined = pd.concat(
                [_single_source_frame(prior_bar), chunk.frame],
                axis=0,
            )
            combined_ordinals = (int(prior_ordinal), *chunk.ordinals)
            skip_start = prior_bar.start
        else:
            combined = chunk.frame
            combined_ordinals = chunk.ordinals
            skip_start = None
        starts = tuple(pd.Timestamp(value) for value in combined.index)
        real_bars = _source_bars(combined)
        ordinal_by_start = dict(zip(starts, combined_ordinals))
        source_by_start = dict(zip(starts, real_bars))
        iterator = iter_completed_bars(
            combined,
            maximum_no_trade_gap_minutes=maximum_no_trade_gap_minutes,
            allow_data_gap_reset=allow_data_gap_reset,
        )
        for bar in iterator:
            if skip_start is not None and bar.start == skip_start:
                continue
            if bar.synthetic_no_trade:
                next_position = bisect_left(starts, bar.end)
                if next_position >= len(starts):
                    raise AssertionError("synthetic bar lacks its proving source row")
                backing_start = starts[next_position]
                provenance = PROVENANCE_SYNTHETIC
            else:
                backing_start = bar.start
                provenance = PROVENANCE_REAL
            backing = source_by_start.get(backing_start)
            ordinal = ordinal_by_start.get(backing_start)
            if backing is None or ordinal is None:
                raise AssertionError("completed bar provenance is unresolved")
            yield CompletedBarReceipt(
                bar=bar,
                provenance=provenance,
                source_row_start=backing_start,
                source_row_ordinal=int(ordinal),
                source_row_sha256=source_row_sha256(
                    backing,
                    int(ordinal),
                ),
                bar_sha256=_bar_sha256(bar),
                max_source_row_admitted=backing_start,
            )
        prior_bar = real_bars[-1]
        prior_ordinal = int(combined_ordinals[-1])


class ParquetBatchSource:
    """Bounded Parquet batches with stable raw-file source row ordinals."""

    REQUIRED = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        path: str | Path,
        *,
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
        batch_rows: int = 65_536,
        expected_sha256: str | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("Parquet source must be a trusted regular file")
        self.start = _aware(start, name="source_start")
        self.end_exclusive = _aware(
            end_exclusive,
            name="source_end_exclusive",
        )
        if self.end_exclusive <= self.start:
            raise ValueError("source interval must be positive")
        self.batch_rows = int(batch_rows)
        if self.batch_rows <= 0:
            raise ValueError("Parquet batch size must be positive")
        self.sha256 = sha256_file(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("Parquet source hash differs from frozen binding")

    @staticmethod
    def _timestamps(raw: pd.DataFrame) -> pd.DatetimeIndex:
        values = pd.DataFrame(raw).copy()
        if "ts" in values.columns:
            timestamp = pd.to_datetime(
                values.pop("ts"),
                errors="coerce",
                utc=True,
            )
        elif "ts_event" in values.columns:
            timestamp = pd.to_datetime(
                values.pop("ts_event"),
                errors="coerce",
                utc=True,
            )
        elif isinstance(values.index, pd.DatetimeIndex):
            timestamp = pd.to_datetime(
                values.index,
                errors="coerce",
                utc=True,
            )
        else:
            raise ValueError("Parquet OHLCV batch has no timestamp")
        result = pd.DatetimeIndex(timestamp).tz_convert(
            "America/New_York"
        )
        if result.isna().any():
            raise ValueError("Parquet OHLCV batch has an invalid timestamp")
        return result

    @staticmethod
    def _normalize(
        raw: pd.DataFrame,
        raw_ordinals: Sequence[int],
    ) -> SourceChunk:
        values = pd.DataFrame(raw).copy()
        timestamp = ParquetBatchSource._timestamps(values)
        if "ts" in values.columns:
            values.pop("ts")
        elif "ts_event" in values.columns:
            values.pop("ts_event")
        if len(values) != len(raw_ordinals):
            raise ValueError("Parquet row ordinal count changed")
        values["__source_ordinal"] = tuple(int(x) for x in raw_ordinals)
        values.index = timestamp
        values.index.name = "ts"
        missing = sorted(set(ParquetBatchSource.REQUIRED) - set(values))
        if missing:
            raise ValueError(f"Parquet OHLCV fields missing: {missing}")
        for column in ParquetBatchSource.REQUIRED:
            values[column] = pd.to_numeric(values[column], errors="raise")
        if "symbol" not in values:
            values["symbol"] = "NQ"
        if "instrument_id" not in values:
            values["instrument_id"] = 0
        values["symbol"] = values["symbol"].astype(str)
        values["instrument_id"] = pd.to_numeric(
            values["instrument_id"],
            errors="raise",
        ).astype("int64")
        values = values.sort_index(kind="stable")
        if values.index.has_duplicates:
            raise ValueError("Parquet OHLCV batch has duplicate timestamps")
        invalid = (
            (values["high"] < values[["open", "close"]].max(axis=1))
            | (values["low"] > values[["open", "close"]].min(axis=1))
            | (values["high"] < values["low"])
            | (values["volume"] < 0)
        )
        if invalid.any():
            raise ValueError("Parquet OHLCV batch has invalid bars")
        ordinals = tuple(int(x) for x in values.pop("__source_ordinal"))
        return SourceChunk(
            frame=values[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                    "instrument_id",
                ]
            ],
            ordinals=ordinals,
        )

    def iter_chunks(
        self,
        *,
        after_source_ordinal: int | None = None,
    ) -> Iterator[SourceChunk]:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(self.path)
        raw_offset = 0
        prior_start: pd.Timestamp | None = None
        for row_group_index in range(parquet.num_row_groups):
            row_count = parquet.metadata.row_group(row_group_index).num_rows
            group_end = raw_offset + row_count
            if (
                after_source_ordinal is not None
                and group_end - 1 <= after_source_ordinal
            ):
                raw_offset = group_end
                continue
            batch_offset = raw_offset
            for batch in parquet.iter_batches(
                batch_size=self.batch_rows,
                row_groups=[row_group_index],
            ):
                ordinals = tuple(
                    range(batch_offset, batch_offset + batch.num_rows)
                )
                batch_offset += batch.num_rows
                chunk = self._normalize(batch.to_pandas(), ordinals)
                mask = (
                    (chunk.frame.index >= self.start)
                    & (chunk.frame.index < self.end_exclusive)
                )
                if after_source_ordinal is not None:
                    mask &= pd.Series(
                        [
                            value > after_source_ordinal
                            for value in chunk.ordinals
                        ],
                        index=chunk.frame.index,
                    ).to_numpy()
                if not mask.any():
                    continue
                positions = [
                    index
                    for index, keep in enumerate(mask)
                    if bool(keep)
                ]
                frame = chunk.frame.iloc[positions]
                filtered_ordinals = tuple(
                    chunk.ordinals[index] for index in positions
                )
                if (
                    prior_start is not None
                    and frame.index[0] <= prior_start
                ):
                    raise ValueError("Parquet source is not globally chronological")
                prior_start = frame.index[-1]
                yield SourceChunk(frame=frame, ordinals=filtered_ordinals)
            raw_offset = group_end

    def count_rows(
        self,
        *,
        on_batch: Callable[[int], None] | None = None,
    ) -> int:
        """Count the bound interval from only the physical timestamp column."""

        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(self.path)
        names = tuple(parquet.schema_arrow.names)
        timestamp_column = next(
            (
                candidate
                for candidate in ("ts", "ts_event")
                if candidate in names
            ),
            None,
        )
        if timestamp_column is None:
            metadata = parquet.schema_arrow.pandas_metadata or {}
            index_columns = metadata.get("index_columns", ())
            timestamp_column = next(
                (
                    value
                    for value in index_columns
                    if isinstance(value, str) and value in names
                ),
                None,
            )
        if timestamp_column is None:
            raise ValueError(
                "Parquet source has no physical timestamp column"
            )
        count = 0
        scanned = 0
        for batch in parquet.iter_batches(
            batch_size=self.batch_rows,
            columns=[timestamp_column],
        ):
            timestamp = self._timestamps(batch.to_pandas())
            count += int(
                (
                    (timestamp >= self.start)
                    & (timestamp < self.end_exclusive)
                ).sum()
            )
            scanned += int(batch.num_rows)
            if on_batch is not None:
                on_batch(scanned)
        return count


@dataclass(frozen=True)
class FrozenParquetSourceIdentity:
    """Hash/window identity used only to finalize an already complete Pass1."""

    sha256: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    batch_rows: int

    def __post_init__(self) -> None:
        if HEX64.fullmatch(self.sha256) is None:
            raise ValueError("frozen source identity hash is invalid")
        start = _aware(self.start, name="source_identity.start")
        end = _aware(
            self.end_exclusive,
            name="source_identity.end_exclusive",
        )
        if end <= start or int(self.batch_rows) <= 0:
            raise ValueError("frozen source identity is invalid")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)
        object.__setattr__(self, "batch_rows", int(self.batch_rows))

    def iter_chunks(self, **_kwargs) -> Iterator[SourceChunk]:
        raise PermissionError(
            "frozen source identity cannot iterate market rows"
        )

    def count_rows(self, **_kwargs) -> int:
        raise PermissionError(
            "frozen source identity cannot count market rows"
        )


def source_for_pass1_release(
    release: Mapping[str, Any],
    *,
    resume: bool,
) -> ParquetBatchSource | FrozenParquetSourceIdentity:
    """Select a real source or a zero-I/O terminal identity after output QA."""

    permissions = release.get("permissions")
    if (
        release.get("status") != "execution_contract_frozen"
        or release.get("execution_authorized") is not True
        or release.get("authorization")
        != "one-time real semantic discovery pass1 authorized"
        or release.get("authorized_phases") != ["pass1"]
        or not isinstance(permissions, Mapping)
        or permissions.get("real_pass1_authorized") is not True
        or permissions.get("discovery_data_authorized") is not True
    ):
        raise PermissionError(
            "Pass1 source access requires an active exact execution release"
        )
    output_value = Path(str(release.get("output", {}).get("path", "")))
    output_root = (
        output_value
        if output_value.is_absolute()
        else ROOT / output_value
    )
    validate_pass1_release_output_tree(
        output_root,
        resume=resume,
    )
    source = release.get("source", {})
    run = release.get("run", {})
    selection_path = output_root / "selection_authority.json"
    if resume and selection_path.is_symlink():
        raise ValueError(
            "terminal selection may not be a symlink"
        )
    if resume and selection_path.is_file():
        return FrozenParquetSourceIdentity(
            sha256=str(
                release.get("bindings", {}).get(
                    "causal_source_sha256",
                    "",
                )
            ),
            start=_aware(
                source.get("start"),
                name="release.source.start",
            ),
            end_exclusive=_aware(
                source.get("end_exclusive"),
                name="release.source.end_exclusive",
            ),
            batch_rows=int(run.get("batch_rows", 0)),
        )
    source_path = Path(str(source.get("path", "")))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    return ParquetBatchSource(
        source_path,
        start=source.get("start"),
        end_exclusive=source.get("end_exclusive"),
        batch_rows=int(run.get("batch_rows", 0)),
        expected_sha256=str(
            release.get("bindings", {}).get(
                "causal_source_sha256",
                "",
            )
        ),
    )


class BoundedSemanticSelector:
    """Exact selector whose memory is bounded by bucket_count * retain_count.

    A qualifying structure event is defined at exactly one observation clock:
    ``resolved_at == asof`` or ``last_attempt_at == asof``.  Since that clock is
    part of ``semantic_event_id``, the same event cannot qualify on a later
    observation.  Duplicate IDs within one observation fail closed; therefore
    an unbounded global ``seen`` set is neither necessary nor permitted.
    """

    def __init__(self, audit_contract: Mapping[str, Any]) -> None:
        self.contract = dict(audit_contract)
        self.primitive_hash = str(
            self.contract["bindings"]["primitive_protocol_sha256"]
        )
        selection = self.contract["selection"]
        self.allowed_years = frozenset(
            int(value) for value in selection["calendar_years"]
        )
        self.cases_per_bucket = int(selection["cases_per_bucket"])
        self._buckets: dict[str, list[dict[str, Any]]] = {}
        self._last_eligible_clock: pd.Timestamp | None = None

    def expected_bucket_keys(self) -> tuple[str, ...]:
        selection = self.contract["selection"]
        return tuple(
            "|".join((timeframe, direction, case_class, str(year)))
            for timeframe in selection["timeframes"]
            for direction in selection["directions"]
            for case_class in selection["case_classes"]
            for year in selection["calendar_years"]
        )

    def bucket_progress(self) -> dict[str, Any]:
        expected = self.expected_bucket_keys()
        counts = {
            key: len(self._buckets.get(key, ()))
            for key in expected
        }
        missing = [
            key
            for key, count in counts.items()
            if count != self.cases_per_bucket
        ]
        return {
            "filled_bucket_count": len(expected) - len(missing),
            "expected_bucket_count": len(expected),
            "selected_case_count": sum(counts.values()),
            "expected_case_count": len(expected) * self.cases_per_bucket,
            "missing_buckets": missing,
        }

    def observe(
        self,
        observation: MarketObservation,
        receipt: CompletedBarReceipt,
        *,
        histories: Mapping[Timeframe, Sequence[Candle]],
        source_prefix_root: str,
        produced_bar_prefix_root: str,
        source_rows_admitted: int,
        produced_bars: int,
        reset_epoch: int,
    ) -> None:
        if not receipt.real_source_bar:
            raise ValueError("synthetic transitions are ineligible case anchors")
        if observation.asof != receipt.bar.end:
            raise ValueError("selector observation and bar clocks differ")
        if receipt.source_row_start >= observation.asof:
            raise ValueError("case source row is not before the case clock")
        if self._last_eligible_clock is not None and (
            observation.asof <= self._last_eligible_clock
        ):
            raise ValueError("eligible case clocks are not strictly increasing")
        self._last_eligible_clock = observation.asof
        year = int(
            observation.asof.tz_convert("America/New_York").year
        )
        if year not in self.allowed_years:
            return
        if (
            HEX64.fullmatch(source_prefix_root) is None
            or HEX64.fullmatch(produced_bar_prefix_root) is None
        ):
            raise ValueError("selector prefix roots must be SHA-256")
        seen_this_clock: set[str] = set()
        for timeframe in CORE_TIMEFRAMES:
            for item in observation.frame(timeframe).structure_breaks:
                if not (
                    (
                        item.lifecycle is BOSLifecycle.CONFIRMED
                        and item.resolved_at == observation.asof
                    )
                    or (
                        item.lifecycle is BOSLifecycle.PENDING
                        and item.last_attempt_at == observation.asof
                    )
                ):
                    continue
                case_class = classify_bos_case(
                    item,
                    case_clock=observation.asof,
                )
                if case_class is None:
                    continue
                if not semantic_case_context_complete(
                    observation,
                    histories,
                    item,
                ):
                    continue
                event_id = semantic_event_id(
                    item,
                    case_class=case_class,
                    case_clock=observation.asof,
                )
                if event_id in seen_this_clock:
                    raise ValueError(
                        "duplicate semantic event id in one observation"
                    )
                seen_this_clock.add(event_id)
                base = SemanticCase(
                    semantic_event_id=event_id,
                    bos_id=item.bos_id,
                    timeframe=timeframe,
                    direction=item.direction,
                    case_class=case_class,
                    calendar_year=year,
                    case_clock=observation.asof,
                    selection_score=selection_score(
                        self.primitive_hash,
                        event_id,
                    ),
                    semantic_state_hash=content_hash(item),
                    target_swing_id=item.target_swing_id,
                    source_structure_id=item.source_structure_id,
                )
                candidate = {
                    **to_primitive(base),
                    "case_bar_synthetic": False,
                    "case_source_row_start": receipt.source_row_start,
                    "case_source_row_ordinal": receipt.source_row_ordinal,
                    "case_source_row_sha256": receipt.source_row_sha256,
                    "case_bar_sha256": receipt.bar_sha256,
                    "source_prefix_root": source_prefix_root,
                    "produced_bar_prefix_root": produced_bar_prefix_root,
                    "source_rows_admitted": int(source_rows_admitted),
                    "produced_bars": int(produced_bars),
                    "reset_epoch": int(reset_epoch),
                }
                key = base.bucket_key
                bucket = self._buckets.setdefault(key, [])
                bucket.append(candidate)
                bucket.sort(
                    key=lambda value: (
                        value["selection_score"],
                        value["semantic_event_id"],
                    )
                )
                del bucket[self.cases_per_bucket :]

    def manifest(
        self,
        *,
        audit_contract_sha256: str,
        runner_contract_sha256: str,
        engine_output_root: str | Path,
        source_sha256: str,
        source_start: pd.Timestamp,
        source_end_exclusive: pd.Timestamp,
        source_rows: int,
        produced_bars: int,
        source_prefix_root: str,
        produced_bar_prefix_root: str,
        implementation_hashes: Mapping[str, str],
        iterator_bindings: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected = self.expected_bucket_keys()
        missing = [
            key
            for key in expected
            if len(self._buckets.get(key, ())) != self.cases_per_bucket
        ]
        cases = [
            item
            for key in expected
            for item in self._buckets.get(key, ())
        ]
        return {
            "format_version": 1,
            "artifact": "v3_structure_bos_selection_authority",
            "audit_id": self.contract["audit_id"],
            "status": "unavailable" if missing else "complete",
            "audit_contract_sha256": audit_contract_sha256,
            "runner_contract_sha256": runner_contract_sha256,
            "engine_output_root": str(Path(engine_output_root).resolve()),
            "source_sha256": source_sha256,
            "source_start": _aware(source_start, name="source_start"),
            "source_end_exclusive": _aware(
                source_end_exclusive,
                name="source_end_exclusive",
            ),
            "source_rows": int(source_rows),
            "produced_bars": int(produced_bars),
            "source_prefix_root": source_prefix_root,
            "produced_bar_prefix_root": produced_bar_prefix_root,
            "iterator_bindings": dict(sorted(iterator_bindings.items())),
            "implementation_hashes": dict(
                sorted(implementation_hashes.items())
            ),
            "expected_bucket_count": len(expected),
            "cases_per_bucket": self.cases_per_bucket,
            "expected_case_count": len(expected) * self.cases_per_bucket,
            "selected_case_count": len(cases),
            "missing_buckets": missing,
            "cases": cases,
        }


def observer_from_model_config(
    path: str | Path,
) -> CausalObserver:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("observer", {})
    minimum = raw.get("minimum_bars", {})
    return CausalObserver(
        ObserverConfig(
            atr_period=int(raw.get("atr_period", 14)),
            swing_k=int(raw.get("swing_k", 2)),
            external_liquidity_lookback=int(
                raw.get("external_liquidity_lookback", 80)
            ),
            memory_events=int(raw.get("memory_events", 512)),
            minimum_bars={
                timeframe: int(minimum.get(timeframe.value, default))
                for timeframe, default in {
                    Timeframe.H4: 16,
                    Timeframe.H1: 24,
                    Timeframe.M5: 24,
                    Timeframe.M1: 30,
                }.items()
            },
            tick_size=float(payload.get("tick_size", 0.25)),
            point_value=float(payload.get("point_value", 20.0)),
            structure_protocol=raw.get("structure_protocol"),
            liquidity_protocol=raw.get("liquidity_protocol"),
            displacement_protocol=raw.get("displacement_protocol"),
            group3_protocol=raw.get("group3_protocol"),
            group4_protocol=raw.get("group4_protocol"),
            group5_protocol=raw.get("group5_protocol"),
        )
    )


class SemanticDiscoveryReplay:
    """Reader + observer semantic replay and internally derived prefix roots."""

    def __init__(
        self,
        *,
        reader: CausalMarketReader,
        observer: CausalObserver,
        selector: BoundedSemanticSelector | None,
    ) -> None:
        self.reader = reader
        self.observer = observer
        self.selector = selector
        self.source_prefix_root = "0" * 64
        self.produced_bar_prefix_root = "0" * 64
        self.source_rows_admitted = 0
        self.real_bars = 0
        self.synthetic_bars = 0
        self.produced_bars = 0
        self.reset_epoch = 0
        self.last_reset_at: pd.Timestamp | None = None
        self.last_reset_reason: str | None = None
        self.last_source_bar: Bar | None = None
        self.last_source_row_ordinal: int | None = None
        self.last_source_row_sha256: str | None = None
        self.last_observation: MarketObservation | None = None

    def on_receipt(
        self,
        receipt: CompletedBarReceipt,
    ) -> MarketObservation:
        next_source_root = self.source_prefix_root
        next_source_rows = self.source_rows_admitted
        next_real_bars = self.real_bars
        next_synthetic_bars = self.synthetic_bars
        if receipt.real_source_bar:
            if (
                self.last_source_row_ordinal is not None
                and receipt.source_row_ordinal
                <= self.last_source_row_ordinal
            ):
                raise ValueError("source row ordinal did not advance")
            next_source_root = _chain_primitive_mapping(
                self.source_prefix_root,
                {
                    "source_row_ordinal": receipt.source_row_ordinal,
                    "source_row_start": (
                        receipt.source_row_start.isoformat()
                    ),
                    "source_row_sha256": receipt.source_row_sha256,
                },
            )
            next_source_rows += 1
            next_real_bars += 1
        else:
            next_synthetic_bars += 1
        next_produced_bars = self.produced_bars + 1
        next_produced_root = _chain_primitive_mapping(
            self.produced_bar_prefix_root,
            {
                "produced_bar_number": next_produced_bars,
                "provenance": receipt.provenance,
                "source_row_start": receipt.source_row_start.isoformat(),
                "source_row_ordinal": receipt.source_row_ordinal,
                "source_row_sha256": receipt.source_row_sha256,
                "bar_sha256": receipt.bar_sha256,
                "synthetic_no_trade": receipt.bar.synthetic_no_trade,
                "data_gap_before_minutes": (
                    receipt.bar.data_gap_before_minutes
                ),
            },
        )
        update = self.reader.on_bar(receipt.bar)
        observation = self.observer.observe(
            update,
            ExecutionRealityInput(
                spread_points=None,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                source="semantic_audit_missing_execution",
            ),
        )
        next_reset_epoch = self.reset_epoch
        reset_anomalies = {
            "contract_change_history_reset",
            "data_gap_history_reset",
        }.intersection(observation.anomalies)
        next_last_reset_at = self.last_reset_at
        next_last_reset_reason = self.last_reset_reason
        if reset_anomalies:
            next_reset_epoch += 1
            next_last_reset_at = observation.asof
            next_last_reset_reason = (
                "contract_change_reset"
                if "contract_change_history_reset" in reset_anomalies
                else "data_gap_reset"
            )
        # Commit counters and roots only after the complete reader/observer
        # transition succeeds.  Checkpoints can therefore never name a
        # half-applied semantic transition.
        self.source_prefix_root = next_source_root
        self.produced_bar_prefix_root = next_produced_root
        self.source_rows_admitted = next_source_rows
        self.real_bars = next_real_bars
        self.synthetic_bars = next_synthetic_bars
        self.produced_bars = next_produced_bars
        self.reset_epoch = next_reset_epoch
        self.last_reset_at = next_last_reset_at
        self.last_reset_reason = next_last_reset_reason
        if receipt.real_source_bar:
            self.last_source_bar = Bar(
                start=receipt.bar.start,
                open=receipt.bar.open,
                high=receipt.bar.high,
                low=receipt.bar.low,
                close=receipt.bar.close,
                volume=receipt.bar.volume,
                symbol=receipt.bar.symbol,
                instrument_id=receipt.bar.instrument_id,
            )
            self.last_source_row_ordinal = receipt.source_row_ordinal
            self.last_source_row_sha256 = receipt.source_row_sha256
            if self.selector is not None:
                self.selector.observe(
                    observation,
                    receipt,
                    histories={
                        timeframe: self.reader.window(
                            timeframe,
                            self.reader.maximum_history,
                        )
                        for timeframe in CORE_TIMEFRAMES
                    },
                    source_prefix_root=self.source_prefix_root,
                    produced_bar_prefix_root=(
                        self.produced_bar_prefix_root
                    ),
                    source_rows_admitted=self.source_rows_admitted,
                    produced_bars=self.produced_bars,
                    reset_epoch=self.reset_epoch,
                )
        self.last_observation = observation
        return observation


class DiscoveryCheckpointStore:
    """Hash-bound, fsynced, trusted-local resumable state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    @property
    def status(self) -> str:
        if not self.root.exists() and not self.root.is_symlink():
            return "absent"
        if self.root.is_symlink() or not self.root.is_dir():
            return "invalid"
        if (
            not self.manifest_path.exists()
            and not self.manifest_path.is_symlink()
        ):
            return "invalid"
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            return "invalid"
        try:
            manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            return "invalid"
        filename = str(manifest.get("state_file", ""))
        if re.fullmatch(r"state-[0-9a-f]{64}\.pkl", filename) is None:
            return "invalid"
        path = self.root / filename
        if path.is_symlink() or not path.is_file():
            return "invalid"
        try:
            if sha256_file(path) != manifest.get("state_sha256"):
                return "invalid"
        except OSError:
            return "invalid"
        return "valid"

    @property
    def exists(self) -> bool:
        return self.status == "valid"

    def require_absent(self) -> None:
        status = self.status
        if status == "valid":
            raise FileExistsError(
                "checkpoint exists; use explicit resume"
            )
        if status == "invalid":
            raise ValueError(
                "checkpoint path exists but is invalid; refusing overwrite"
            )

    def current_state_sha256(self) -> str | None:
        if self.status == "absent":
            return None
        if self.status != "valid":
            raise ValueError("checkpoint is invalid")
        manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        digest = str(manifest["state_sha256"])
        if HEX64.fullmatch(digest) is None:
            raise ValueError("checkpoint state digest is invalid")
        return digest

    def save(
        self,
        state: Mapping[str, Any],
        *,
        bindings: Mapping[str, Any],
        phase: str,
    ) -> dict[str, Any]:
        if phase not in {"pass1", "pass2"}:
            raise ValueError("unknown discovery checkpoint phase")
        raw = pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
        state_hash = hashlib.sha256(raw).hexdigest()
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "phase": phase,
            "state_sha256": state_hash,
            "state_file": f"state-{state_hash}.pkl",
            "bindings": dict(bindings),
            "source_rows_admitted": int(
                state["replay"].source_rows_admitted
            ),
            "produced_bars": int(state["replay"].produced_bars),
            "last_source_row_ordinal": (
                state["replay"].last_source_row_ordinal
            ),
            "committed_cases": sorted(
                str(value)
                for value in state.get("committed_cases", ())
            ),
            "selection_manifest_sha256": state.get(
                "selection_manifest_sha256"
            ),
            "verified_source_row_count": state.get(
                "verified_source_row_count"
            ),
        }
        state_path = self.root / manifest["state_file"]
        if self.root.is_symlink() or (
            self.root.exists() and not self.root.is_dir()
        ):
            raise ValueError("checkpoint root is not a regular directory")
        if state_path.exists() or state_path.is_symlink():
            if (
                state_path.is_symlink()
                or not state_path.is_file()
                or sha256_file(state_path) != state_hash
            ):
                raise ValueError(
                    "content-addressed checkpoint state path is invalid"
                )
        else:
            _atomic_bytes(state_path, raw)
        if self.manifest_path.is_symlink() or (
            self.manifest_path.exists()
            and not self.manifest_path.is_file()
        ):
            raise ValueError("checkpoint manifest path is invalid")
        _atomic_bytes(self.manifest_path, _canonical_json(manifest))
        for stale in self.root.glob("state-*.pkl"):
            if (
                stale != state_path
                and stale.is_file()
                and not stale.is_symlink()
                and re.fullmatch(r"state-[0-9a-f]{64}\.pkl", stale.name)
            ):
                stale.unlink()
        return manifest

    def load(
        self,
        *,
        expected_bindings: Mapping[str, Any],
        phase: str,
    ) -> dict[str, Any]:
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise ValueError("checkpoint manifest is not a trusted regular file")
        manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if (
            int(manifest.get("format_version", 0))
            != CHECKPOINT_FORMAT_VERSION
            or manifest.get("phase") != phase
        ):
            raise ValueError("checkpoint phase or format changed")
        if manifest.get("bindings") != dict(expected_bindings):
            raise ValueError("checkpoint bindings changed")
        filename = str(manifest.get("state_file", ""))
        if re.fullmatch(r"state-[0-9a-f]{64}\.pkl", filename) is None:
            raise ValueError("checkpoint state filename is unsafe")
        state_path = self.root / filename
        if state_path.is_symlink() or not state_path.is_file():
            raise ValueError("checkpoint state is not a trusted regular file")
        raw = state_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["state_sha256"]:
            raise ValueError("checkpoint state digest is invalid")
        state = pickle.loads(raw)
        if not isinstance(state, dict):
            raise ValueError("checkpoint state root must be a mapping")
        replay = state.get("replay")
        if not isinstance(replay, SemanticDiscoveryReplay):
            raise ValueError("checkpoint replay type changed")
        comparisons = {
            "source_rows_admitted": replay.source_rows_admitted,
            "produced_bars": replay.produced_bars,
            "last_source_row_ordinal": replay.last_source_row_ordinal,
            "committed_cases": sorted(
                str(value)
                for value in state.get("committed_cases", ())
            ),
            "selection_manifest_sha256": state.get(
                "selection_manifest_sha256"
            ),
            "verified_source_row_count": state.get(
                "verified_source_row_count"
            ),
        }
        for field, value in comparisons.items():
            if manifest.get(field) != value:
                raise ValueError(f"checkpoint manifest disagrees on {field}")
        return state


class CasePacketTransaction:
    """Publish authority + blind packet as one immutable case transaction."""

    REQUIRED_RELATIVE = (
        "authority/authority.json",
        "blind/case.png",
        "blind/blind_raw_evidence.json",
        "blind/blind_manifest.json",
        "blind/review_template.json",
        "blind/BLIND_PACKET.json",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.staging_root = self.root / "._staging"
        self.cases_root = self.root / "cases"

    @staticmethod
    def case_id(semantic_event_id_value: str) -> str:
        return hashlib.sha256(
            semantic_event_id_value.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _verify_complete(root: Path) -> dict[str, Any]:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("case transaction root is not a regular directory")
        marker = root / "COMPLETED.json"
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("case transaction is partial")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        file_hashes = payload.get("file_hashes")
        if (
            not isinstance(file_hashes, dict)
            or set(file_hashes) != set(CasePacketTransaction.REQUIRED_RELATIVE)
        ):
            raise ValueError("case transaction file set changed")
        for relative, expected in file_hashes.items():
            if HEX64.fullmatch(str(expected)) is None:
                raise ValueError("case transaction digest is invalid")
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError("case transaction file is not regular")
            if sha256_file(path) != expected:
                raise ValueError("case transaction file digest changed")
        allowed_files = {
            *CasePacketTransaction.REQUIRED_RELATIVE,
            "COMPLETED.json",
        }
        allowed_directories = {"authority", "blind"}
        observed_files: set[str] = set()
        observed_directories: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("case transaction tree contains a symlink")
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                observed_directories.add(relative)
            elif path.is_file():
                observed_files.add(relative)
            else:
                raise ValueError(
                    "case transaction tree contains a non-regular entry"
                )
        if (
            observed_files != allowed_files
            or observed_directories != allowed_directories
        ):
            raise ValueError("case transaction tree has unbound entries")
        return payload

    def publish(
        self,
        *,
        semantic_event_id_value: str,
        completion_metadata: Mapping[str, Any],
        producer: Callable[[Path, Path], None],
    ) -> tuple[Path, str]:
        case_id = self.case_id(semantic_event_id_value)
        final = self.cases_root / case_id
        staging = self.staging_root / case_id
        if final.is_symlink() or staging.is_symlink():
            raise ValueError("case transaction path may not be a symlink")
        if final.exists():
            payload = self._verify_complete(final)
            if (
                payload.get("semantic_event_id") != semantic_event_id_value
                or payload.get("completion_metadata")
                != to_primitive(dict(completion_metadata))
            ):
                raise ValueError("existing case transaction identity differs")
            return final, sha256_file(final / "COMPLETED.json")
        if staging.exists():
            payload = self._verify_complete(staging)
            if (
                payload.get("semantic_event_id") != semantic_event_id_value
                or payload.get("completion_metadata")
                != to_primitive(dict(completion_metadata))
            ):
                raise ValueError("orphan case transaction identity differs")
            self.cases_root.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            _fsync_directory(self.cases_root)
            return final, sha256_file(final / "COMPLETED.json")
        self.staging_root.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        _fsync_directory(self.staging_root)
        try:
            producer(staging / "authority", staging / "blind")
            for relative in self.REQUIRED_RELATIVE:
                path = staging / relative
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        f"case producer omitted required file: {relative}"
                    )
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            file_hashes = {
                relative: sha256_file(staging / relative)
                for relative in self.REQUIRED_RELATIVE
            }
            completed = {
                "format_version": 1,
                "artifact": "v3_semantic_case_transaction",
                "semantic_event_id": semantic_event_id_value,
                "file_hashes": file_hashes,
                "completion_metadata": dict(completion_metadata),
            }
            _write_new_json(staging / "COMPLETED.json", completed)
            _fsync_directory(staging / "authority")
            _fsync_directory(staging / "blind")
            _fsync_directory(staging)
            # Validate the complete staged tree on the *first* publication,
            # not only when a final/orphan is later reused.  A producer cannot
            # smuggle an unhashed file, directory, or symlink into the final
            # immutable case root.
            self._verify_complete(staging)
            self.cases_root.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            _fsync_directory(self.cases_root)
            return final, sha256_file(final / "COMPLETED.json")
        except BaseException:
            # A partial staging directory is intentionally retained and blocks
            # automatic overwrite.  A complete orphan is adopted on resume.
            raise


class SemanticDiscoveryRunner:
    """Orchestrate resumable pass1 selection and pass2 packet materialization."""

    def __init__(
        self,
        *,
        audit_contract: Mapping[str, Any],
        runner_contract: Mapping[str, Any],
        source: ParquetBatchSource | FrozenParquetSourceIdentity,
        model_config_path: str | Path,
        output_root: str | Path,
        audit_contract_sha256: str,
        runner_contract_sha256: str,
        implementation_hashes: Mapping[str, str],
        maximum_history: int = 1024,
        checkpoint_source_rows: int = 25_000,
        execution_release: Mapping[str, Any] | None = None,
        execution_release_sha256: str | None = None,
    ) -> None:
        self.audit_contract = dict(audit_contract)
        self.runner_contract = dict(runner_contract)
        self.source = source
        self.model_config_path = Path(model_config_path)
        self.output_root = Path(output_root)
        self.audit_contract_sha256 = audit_contract_sha256
        self.runner_contract_sha256 = runner_contract_sha256
        self.implementation_hashes = dict(implementation_hashes)
        self.execution_release = (
            None
            if execution_release is None
            else dict(execution_release)
        )
        self.execution_release_sha256 = execution_release_sha256
        self.resource_limits: ExecutionResourceLimits | None = None
        self.maximum_history = int(maximum_history)
        self.checkpoint_source_rows = int(checkpoint_source_rows)
        if self.maximum_history <= 0 or self.checkpoint_source_rows <= 0:
            raise ValueError("history and checkpoint sizes must be positive")
        iterator = self.runner_contract["source_iterator"]
        self.maximum_no_trade_gap_minutes = int(
            iterator["maximum_no_trade_gap_minutes"]
        )
        self.allow_data_gap_reset = bool(
            iterator["allow_data_gap_reset"]
        )
        if self.source.batch_rows != int(iterator["batch_rows"]):
            raise ValueError("source batch size differs from runner contract")
        actual_model_hash = sha256_file(self.model_config_path)
        if actual_model_hash != self.runner_contract["bindings"][
            "model_config_sha256"
        ]:
            raise ValueError("actual model config differs from frozen binding")
        required_implementation_hashes = set(
            self.runner_contract["implementation_hashes_required"]
        )
        if set(self.implementation_hashes) != required_implementation_hashes:
            raise ValueError("implementation hash family differs from contract")
        if any(
            HEX64.fullmatch(str(value)) is None
            for value in self.implementation_hashes.values()
        ):
            raise ValueError("implementation binding is not a SHA-256")
        if self.audit_contract_sha256 != self.runner_contract["bindings"][
            "blind_audit_contract_sha256"
        ]:
            raise ValueError("actual audit contract differs from runner binding")
        frozen_source = self.audit_contract["source"]
        if self.source.sha256 != self.audit_contract["bindings"][
            "causal_source_sha256"
        ]:
            raise ValueError("runner source differs from audit contract")
        if self.source.start != _aware(
            frozen_source["start"],
            name="frozen_source_start",
        ) or self.source.end_exclusive != _aware(
            frozen_source["end_exclusive"],
            name="frozen_source_end",
        ):
            raise ValueError("runner interval differs from audit contract")
        if (self.execution_release is None) != (
            self.execution_release_sha256 is None
        ):
            raise ValueError(
                "execution release and its digest must be supplied together"
            )
        if self.execution_release is not None:
            self._validate_execution_release_bindings()

    def _validate_execution_release_bindings(self) -> None:
        release = self.execution_release
        if release is None:
            raise AssertionError("execution release is absent")
        if HEX64.fullmatch(str(self.execution_release_sha256)) is None:
            raise ValueError("execution release digest is invalid")
        status = str(release.get("status", ""))
        if status not in {
            "draft_no_execution_authority",
            "execution_contract_frozen",
            "synthetic_test_release",
        }:
            raise ValueError("execution release status is unsupported")
        if release.get("authorized_phases") != ["pass1"]:
            raise ValueError("execution release may authorize only pass1")
        if bool(release.get("execution_authorized")) != (
            status
            in {"execution_contract_frozen", "synthetic_test_release"}
        ):
            raise ValueError("execution release authority is inconsistent")
        confirmatory_one_time = (
            status == "synthetic_test_release"
            and release.get("confirmatory_one_time") is True
        )
        confirmatory_flag = release.get("confirmatory_one_time")
        if (
            confirmatory_flag is not None
            and not isinstance(confirmatory_flag, bool)
        ):
            raise ValueError(
                "confirmatory one-time release flag is invalid"
            )
        if confirmatory_one_time:
            confirmatory = release.get("confirmatory")
            if (
                not isinstance(confirmatory, dict)
                or set(confirmatory)
                != {
                    "attempt_id",
                    "registry_root",
                    "attempt_marker_path",
                }
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
                    str(confirmatory.get("attempt_id", "")),
                )
                is None
            ):
                raise ValueError(
                    "confirmatory one-time registry binding is invalid"
                )
            registry_root = Path(
                str(confirmatory["registry_root"])
            )
            marker_path = Path(
                str(confirmatory["attempt_marker_path"])
            )
            if (
                not registry_root.is_absolute()
                or not marker_path.is_absolute()
                or marker_path.parent != registry_root
                or marker_path.name
                != "CONFIRMATORY_ATTEMPT_STARTED.json"
            ):
                raise ValueError(
                    "confirmatory one-time registry path is invalid"
                )
        elif release.get("confirmatory") is not None:
            raise ValueError(
                "ordinary release may not bind a confirmatory registry"
            )
        bindings = release.get("bindings", {})
        required = {
            "audit_contract_sha256": self.audit_contract_sha256,
            "engineering_runner_contract_sha256": (
                self.runner_contract_sha256
            ),
            "semantic_discovery_runner_sha256": (
                self.implementation_hashes[
                    "semantic_discovery_runner_sha256"
                ]
            ),
            "causal_source_sha256": self.source.sha256,
            "model_config_sha256": self.runner_contract["bindings"][
                "model_config_sha256"
            ],
            "primitive_protocol_sha256": self.runner_contract["bindings"][
                "primitive_protocol_sha256"
            ],
            "io_code_sha256": self.runner_contract["bindings"][
                "io_code_sha256"
            ],
            "causal_reader_code_sha256": self.runner_contract[
                "bindings"
            ]["causal_reader_code_sha256"],
            "market_clock_code_sha256": self.runner_contract["bindings"][
                "market_clock_code_sha256"
            ],
            "semantic_audit_code_sha256": self.runner_contract[
                "bindings"
            ]["semantic_audit_code_sha256"],
            "observer_code_sha256": self.implementation_hashes[
                "observation_sha256"
            ],
            "structure_code_sha256": self.implementation_hashes[
                "structure_sha256"
            ],
        }
        for name, expected in required.items():
            if bindings.get(name) != expected:
                raise ValueError(
                    f"execution release binding changed: {name}"
                )
        release_source = release.get("source", {})
        frozen_source = self.audit_contract["source"]
        expected_source_path = str(frozen_source["path"])
        if release_source != {
            "path": expected_source_path,
            "start": str(frozen_source["start"]),
            "end_exclusive": str(frozen_source["end_exclusive"]),
            "window_role": str(frozen_source["window_role"]),
        }:
            raise ValueError("execution release source window changed")
        run = release.get("run", {})
        expected_run = {
            "batch_rows": self.source.batch_rows,
            "maximum_history": self.maximum_history,
            "maximum_no_trade_gap_minutes": (
                self.maximum_no_trade_gap_minutes
            ),
            "allow_data_gap_reset": self.allow_data_gap_reset,
            "checkpoint_source_rows": self.checkpoint_source_rows,
            "real_only_case_anchors": True,
            "diagnostic_stop_allowed": (
                status == "synthetic_test_release"
                and not confirmatory_one_time
            ),
        }
        if run != expected_run:
            raise ValueError("execution release run parameters changed")
        if release.get("runtime") != runtime_environment():
            raise ValueError("execution release runtime environment changed")
        self.resource_limits = ExecutionResourceLimits.from_mapping(
            release.get("resources", {})
        )
        output_value = str(release.get("output", {}).get("path", ""))
        if not output_value:
            raise ValueError("execution release output path is absent")
        expected_output = Path(output_value)
        if not expected_output.is_absolute():
            expected_output = ROOT / expected_output
        if self.output_root.absolute() != expected_output.absolute():
            raise ValueError("runner output differs from execution release")

    @property
    def iterator_bindings(self) -> dict[str, Any]:
        return {
            "maximum_history": self.maximum_history,
            "maximum_no_trade_gap_minutes": (
                self.maximum_no_trade_gap_minutes
            ),
            "allow_data_gap_reset": self.allow_data_gap_reset,
            "source_batch_rows": self.source.batch_rows,
        }

    def _bindings(
        self,
        *,
        phase: str,
        selection_manifest_sha256: str | None = None,
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "audit_contract_sha256": self.audit_contract_sha256,
            "runner_contract_sha256": self.runner_contract_sha256,
            "source_sha256": self.source.sha256,
            "source_start": self.source.start.isoformat(),
            "source_end_exclusive": self.source.end_exclusive.isoformat(),
            "model_config_sha256": sha256_file(
                self.model_config_path
            ),
            "implementation_hashes": dict(
                sorted(self.implementation_hashes.items())
            ),
            "iterator_bindings": self.iterator_bindings,
            "checkpoint_source_rows": self.checkpoint_source_rows,
            "execution_release_sha256": self.execution_release_sha256,
            "execution_release_id": (
                None
                if self.execution_release is None
                else self.execution_release.get("release_id")
            ),
            "resource_limits": (
                None
                if self.resource_limits is None
                else to_primitive(self.resource_limits)
            ),
            "selection_manifest_sha256": selection_manifest_sha256,
        }

    def _assert_execution_authorized(self, phase: str) -> None:
        if phase not in {"pass1", "pass2"}:
            raise ValueError("unknown semantic discovery phase")
        authorization = str(
            self.runner_contract.get("authorization", "")
        )
        status = str(self.runner_contract.get("status", ""))
        source_role = str(
            self.audit_contract.get("source", {}).get("window_role", "")
        )
        synthetic = (
            self.execution_release is None
            and
            status == "synthetic_test_contract"
            and authorization == "synthetic integration tests only"
            and source_role == "synthetic_fixture"
        )
        if synthetic:
            return
        release = self.execution_release
        if release is None:
            raise PermissionError(
                "semantic discovery execution is not authorized by the "
                "runner contract and source role"
            )
        if phase not in release.get("authorized_phases", ()):
            raise PermissionError(
                f"execution release does not authorize {phase}"
            )
        release_status = str(release.get("status", ""))
        release_authorized = release.get("execution_authorized") is True
        expiry = _aware(
            release.get("expires_at"),
            name="execution_release.expires_at",
        )
        if expiry.tz_convert("UTC") <= pd.Timestamp.now(tz="UTC"):
            raise PermissionError("execution release is expired")
        test_release = (
            release_status == "synthetic_test_release"
            and release_authorized
            and source_role == "synthetic_fixture"
        )
        confirmatory_one_time = (
            test_release
            and release.get("confirmatory_one_time") is True
        )
        if confirmatory_one_time:
            issued_at = _aware(
                release.get("issued_at"),
                name="execution_release.issued_at",
            )
            now = pd.Timestamp.now(tz="UTC")
            if (
                issued_at.tz_convert("UTC") > now
                or (
                    expiry.tz_convert("UTC")
                    - issued_at.tz_convert("UTC")
                ).total_seconds()
                > CONFIRMATORY_MAX_RELEASE_SECONDS
            ):
                raise PermissionError(
                    "one-time synthetic release window is invalid"
                )
        if confirmatory_one_time and release.get("authorization") != (
            "synthetic pass1 confirmatory one-time"
        ):
            raise PermissionError(
                "one-time synthetic authorization text changed"
            )
        production_release = (
            release_status == "execution_contract_frozen"
            and release_authorized
            and source_role == "semantic_discovery"
            and release.get("authorization")
            == "one-time real semantic discovery pass1 authorized"
        )
        if not (test_release or production_release):
            raise PermissionError(
                "execution release is not active for this source role"
            )
        if confirmatory_one_time:
            self._assert_confirmatory_attempt_registered()

    def _assert_confirmatory_attempt_registered(self) -> None:
        release = self.execution_release
        if release is None:
            raise AssertionError("execution release is absent")
        confirmatory = release.get("confirmatory")
        if not isinstance(confirmatory, dict):
            raise PermissionError(
                "confirmatory attempt registry binding is absent"
            )
        registry_root = Path(str(confirmatory["registry_root"]))
        marker_path = Path(str(confirmatory["attempt_marker_path"]))
        if (
            registry_root.is_symlink()
            or not registry_root.is_dir()
            or marker_path.is_symlink()
            or not marker_path.is_file()
            or marker_path.parent != registry_root
        ):
            raise PermissionError(
                "confirmatory attempt marker is not durably registered"
            )
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PermissionError(
                "confirmatory attempt marker is unreadable"
            ) from exc
        if not isinstance(marker, dict):
            raise PermissionError(
                "confirmatory attempt marker identity changed"
            )
        entries = marker.get("fixtures")
        if (
            marker.get("artifact")
            != "v3_synthetic_full_e2e_attempt_started"
            or marker.get("one_time_attempt_consumed") is not True
            or marker.get("attempt_id")
            != confirmatory.get("attempt_id")
            or not isinstance(entries, list)
        ):
            raise PermissionError(
                "confirmatory attempt marker identity changed"
            )
        output_path = str(self.output_root)
        matching = [
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("release_sha256")
            == self.execution_release_sha256
            and item.get("source_sha256") == self.source.sha256
            and item.get("output_path") == output_path
        ]
        if len(matching) != 1:
            raise PermissionError(
                "confirmatory release is absent from the attempt marker"
            )

    def _validate_release_output(
        self,
        *,
        phase: str,
        resume: bool,
    ) -> None:
        if self.execution_release is None:
            return
        if phase != "pass1":
            raise PermissionError(
                "phase-scoped execution release may not touch pass2 output"
            )
        validate_pass1_release_output_tree(
            self.output_root,
            resume=resume,
        )

    def _enforce_resource_limits(
        self,
        *,
        last_checkpoint_monotonic: float,
        source_rows_admitted: int,
        force: bool = False,
    ) -> None:
        limits = self.resource_limits
        if limits is None:
            return
        if (
            time.monotonic() - last_checkpoint_monotonic
            > limits.checkpoint_stall_timeout_seconds
        ):
            raise TimeoutError(
                "Pass1 durable-checkpoint stall timeout exceeded"
            )
        if (
            not force
            and source_rows_admitted
            % limits.resource_check_source_rows
            != 0
        ):
            return
        import psutil

        rss = int(psutil.Process(os.getpid()).memory_info().rss)
        if rss >= limits.rss_hard_ceiling_bytes:
            raise MemoryError(
                "Pass1 RSS hard ceiling exceeded"
            )
        anchor = self.output_root.parent
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        if anchor.is_symlink() or not anchor.is_dir():
            raise ValueError(
                "Pass1 output filesystem anchor is not trusted"
            )
        free = int(shutil.disk_usage(anchor).free)
        if free < limits.disk_free_floor_bytes:
            raise OSError("Pass1 disk free-space floor breached")

    def _validate_selection_authority(
        self,
        selection: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        selector = BoundedSemanticSelector(self.audit_contract)
        expected_buckets = selector.expected_bucket_keys()
        cases_per_bucket = selector.cases_per_bucket
        expected_case_count = len(expected_buckets) * cases_per_bucket
        required_top = {
            "format_version",
            "artifact",
            "audit_id",
            "status",
            "audit_contract_sha256",
            "runner_contract_sha256",
            "engine_output_root",
            "source_sha256",
            "source_start",
            "source_end_exclusive",
            "source_rows",
            "produced_bars",
            "source_prefix_root",
            "produced_bar_prefix_root",
            "iterator_bindings",
            "implementation_hashes",
            "expected_bucket_count",
            "cases_per_bucket",
            "expected_case_count",
            "selected_case_count",
            "missing_buckets",
            "cases",
        }
        if set(selection) != required_top:
            raise ValueError("selection authority schema changed")
        expected_identity = {
            "format_version": 1,
            "artifact": "v3_structure_bos_selection_authority",
            "audit_id": self.audit_contract["audit_id"],
            "status": "complete",
            "audit_contract_sha256": self.audit_contract_sha256,
            "runner_contract_sha256": self.runner_contract_sha256,
            "engine_output_root": str(self.output_root.resolve()),
            "source_sha256": self.source.sha256,
            "source_start": self.source.start.isoformat(),
            "source_end_exclusive": self.source.end_exclusive.isoformat(),
            "iterator_bindings": self.iterator_bindings,
            "implementation_hashes": dict(
                sorted(self.implementation_hashes.items())
            ),
            "expected_bucket_count": len(expected_buckets),
            "cases_per_bucket": cases_per_bucket,
            "expected_case_count": expected_case_count,
            "selected_case_count": expected_case_count,
            "missing_buckets": [],
        }
        primitive_selection = to_primitive(dict(selection))
        for field, expected in expected_identity.items():
            if primitive_selection.get(field) != expected:
                raise ValueError(
                    f"selection authority binding changed: {field}"
                )
        for field in (
            "source_prefix_root",
            "produced_bar_prefix_root",
        ):
            if HEX64.fullmatch(str(selection[field])) is None:
                raise ValueError("selection final prefix root is invalid")
        source_rows = int(selection["source_rows"])
        produced_bars = int(selection["produced_bars"])
        if source_rows <= 0 or produced_bars < source_rows:
            raise ValueError("selection final source counts are invalid")
        raw_cases = selection.get("cases")
        if not isinstance(raw_cases, list) or len(raw_cases) != (
            expected_case_count
        ):
            raise ValueError("selection authority case count changed")
        allowed_timeframes = set(
            self.audit_contract["selection"]["timeframes"]
        )
        allowed_directions = set(
            self.audit_contract["selection"]["directions"]
        )
        allowed_classes = set(
            self.audit_contract["selection"]["case_classes"]
        )
        allowed_years = {
            int(value)
            for value in self.audit_contract["selection"][
                "calendar_years"
            ]
        }
        required_case = {
            "semantic_event_id",
            "bos_id",
            "timeframe",
            "direction",
            "case_class",
            "calendar_year",
            "case_clock",
            "selection_score",
            "semantic_state_hash",
            "target_swing_id",
            "source_structure_id",
            "case_bar_synthetic",
            "case_source_row_start",
            "case_source_row_ordinal",
            "case_source_row_sha256",
            "case_bar_sha256",
            "source_prefix_root",
            "produced_bar_prefix_root",
            "source_rows_admitted",
            "produced_bars",
            "reset_epoch",
        }
        buckets: dict[str, list[dict[str, Any]]] = {
            key: [] for key in expected_buckets
        }
        event_ids: set[str] = set()
        primitive_hash = str(
            self.audit_contract["bindings"][
                "primitive_protocol_sha256"
            ]
        )
        for raw in raw_cases:
            if not isinstance(raw, dict) or set(raw) != required_case:
                raise ValueError("selection case schema changed")
            case = dict(raw)
            event_id = str(case["semantic_event_id"])
            if not event_id or event_id in event_ids:
                raise ValueError("selection semantic event id is duplicated")
            event_ids.add(event_id)
            timeframe = str(case["timeframe"])
            direction = str(case["direction"])
            case_class = str(case["case_class"])
            year = int(case["calendar_year"])
            if (
                timeframe not in allowed_timeframes
                or direction not in allowed_directions
                or case_class not in allowed_classes
                or year not in allowed_years
            ):
                raise ValueError("selection case bucket is unregistered")
            clock = _aware(case["case_clock"], name="case_clock")
            source_start = _aware(
                case["case_source_row_start"],
                name="case_source_row_start",
            )
            if (
                source_start + pd.Timedelta(minutes=1) != clock
                or source_start < self.source.start
                or source_start >= self.source.end_exclusive
                or case.get("case_bar_synthetic") is not False
            ):
                raise ValueError("selection case real-row clock is invalid")
            if (
                year
                != int(clock.tz_convert("America/New_York").year)
            ):
                raise ValueError(
                    "selection calendar year differs from case clock"
                )
            expected_event_id = "|".join(
                (str(case["bos_id"]), case_class, clock.isoformat())
            )
            if event_id != expected_event_id:
                raise ValueError("selection semantic event identity changed")
            if case["selection_score"] != selection_score(
                primitive_hash,
                event_id,
            ):
                raise ValueError("selection score is invalid")
            for field in (
                "selection_score",
                "semantic_state_hash",
                "case_source_row_sha256",
                "case_bar_sha256",
                "source_prefix_root",
                "produced_bar_prefix_root",
            ):
                if HEX64.fullmatch(str(case[field])) is None:
                    raise ValueError(
                        f"selection case digest is invalid: {field}"
                    )
            if (
                int(case["case_source_row_ordinal"]) < 0
                or int(case["source_rows_admitted"]) <= 0
                or int(case["produced_bars"])
                < int(case["source_rows_admitted"])
                or int(case["source_rows_admitted"]) > source_rows
                or int(case["produced_bars"]) > produced_bars
                or int(case["reset_epoch"]) < 0
            ):
                raise ValueError("selection case counters are invalid")
            key = "|".join(
                (timeframe, direction, case_class, str(year))
            )
            buckets[key].append(case)
        for key, values in buckets.items():
            if len(values) != cases_per_bucket:
                raise ValueError(
                    f"selection bucket count changed: {key}"
                )
            values.sort(
                key=lambda value: (
                    value["selection_score"],
                    value["semantic_event_id"],
                )
            )
        expected_order = [
            case["semantic_event_id"]
            for key in expected_buckets
            for case in buckets[key]
        ]
        if [
            str(case["semantic_event_id"]) for case in raw_cases
        ] != expected_order:
            raise ValueError("selection case ordering changed")
        return tuple(
            case for key in expected_buckets for case in buckets[key]
        )

    def _new_replay(
        self,
        *,
        selector: BoundedSemanticSelector | None,
    ) -> SemanticDiscoveryReplay:
        return SemanticDiscoveryReplay(
            reader=CausalMarketReader(maximum_history=self.maximum_history),
            observer=observer_from_model_config(self.model_config_path),
            selector=selector,
        )

    def _receipts(
        self,
        replay: SemanticDiscoveryReplay,
    ) -> Iterator[CompletedBarReceipt]:
        return iter_provenanced_completed_bars(
            self.source.iter_chunks(
                after_source_ordinal=replay.last_source_row_ordinal
            ),
            prior_source_bar=replay.last_source_bar,
            prior_source_ordinal=replay.last_source_row_ordinal,
            maximum_no_trade_gap_minutes=(
                self.maximum_no_trade_gap_minutes
            ),
            allow_data_gap_reset=self.allow_data_gap_reset,
        )

    def _progress(
        self,
        *,
        phase: str,
        replay: SemanticDiscoveryReplay,
        total_source_rows: int | None,
        started: float,
        starting_rows: int,
        status: str,
        durable_checkpoint_state_sha256: str | None = None,
    ) -> dict[str, Any]:
        completed = int(replay.source_rows_admitted)
        fraction = (
            0.0
            if total_source_rows is None
            else (
                1.0
                if total_source_rows == 0
                else min(1.0, completed / total_source_rows)
            )
        )
        elapsed = max(0.0, time.monotonic() - started)
        session_rows = max(0, completed - starting_rows)
        rate = session_rows / elapsed if elapsed > 0 else 0.0
        remaining = (
            None
            if total_source_rows is None
            else max(0, total_source_rows - completed)
        )
        payload = {
            "format_version": 1,
            "artifact": "v3_semantic_discovery_progress",
            "phase": phase,
            "status": status,
            "durable_checkpoint_only": True,
            "resume_supported": (
                durable_checkpoint_state_sha256 is not None
            ),
            "source_rows_admitted": completed,
            "total_source_rows": (
                None
                if total_source_rows is None
                else int(total_source_rows)
            ),
            "completed_percent": round(100.0 * fraction, 6),
            "produced_bars": int(replay.produced_bars),
            "real_bars": int(replay.real_bars),
            "synthetic_bars": int(replay.synthetic_bars),
            "reset_epoch": int(replay.reset_epoch),
            "source_rows_per_second": rate,
            "eta_seconds": (
                None
                if rate <= 0 or remaining is None
                else remaining / rate
            ),
            "causal_clock": (
                None
                if replay.last_observation is None
                else replay.last_observation.asof
            ),
            "durable_checkpoint_state_sha256": (
                durable_checkpoint_state_sha256
            ),
            "last_source_row_ordinal": replay.last_source_row_ordinal,
            "last_source_start": (
                None
                if replay.last_source_bar is None
                else replay.last_source_bar.start
            ),
        }
        if replay.selector is not None:
            payload["selection_progress"] = (
                replay.selector.bucket_progress()
            )
        return payload

    def _reconcile_pass1_terminal(
        self,
        *,
        selection_path: Path,
        checkpoint: DiscoveryCheckpointStore,
        state: Mapping[str, Any],
    ) -> Path:
        """Finalize progress after a crash following selection publication."""

        if selection_path.is_symlink() or not selection_path.is_file():
            raise ValueError(
                "Pass1 terminal selection is not a trusted regular file"
            )
        replay = state.get("replay")
        if (
            not isinstance(replay, SemanticDiscoveryReplay)
            or replay.selector is None
        ):
            raise ValueError("Pass1 terminal checkpoint lacks its selector")
        verified_count = state.get("verified_source_row_count")
        if (
            isinstance(verified_count, bool)
            or not isinstance(verified_count, int)
            or verified_count < 0
            or replay.source_rows_admitted != verified_count
        ):
            raise ValueError(
                "Pass1 terminal checkpoint is not a complete source scan"
            )
        expected = replay.selector.manifest(
            audit_contract_sha256=self.audit_contract_sha256,
            runner_contract_sha256=self.runner_contract_sha256,
            engine_output_root=self.output_root,
            source_sha256=self.source.sha256,
            source_start=self.source.start,
            source_end_exclusive=self.source.end_exclusive,
            source_rows=replay.source_rows_admitted,
            produced_bars=replay.produced_bars,
            source_prefix_root=replay.source_prefix_root,
            produced_bar_prefix_root=replay.produced_bar_prefix_root,
            implementation_hashes=self.implementation_hashes,
            iterator_bindings=self.iterator_bindings,
        )
        observed = json.loads(
            selection_path.read_text(encoding="utf-8")
        )
        if _canonical_json(observed) != _canonical_json(expected):
            raise ValueError(
                "Pass1 terminal selection differs from checkpoint state"
            )
        progress_path = self.output_root / "progress.json"
        if progress_path.is_symlink() or not progress_path.is_file():
            raise ValueError(
                "Pass1 terminal progress is not a trusted regular file"
            )
        checkpoint_sha256 = checkpoint.current_state_sha256()
        terminal_progress = self._progress(
            phase="pass1",
            replay=replay,
            total_source_rows=verified_count,
            started=time.monotonic(),
            starting_rows=verified_count,
            status=(
                "complete"
                if expected["status"] == "complete"
                else "unavailable"
            ),
            durable_checkpoint_state_sha256=checkpoint_sha256,
        )
        _atomic_bytes(
            progress_path,
            _canonical_json(
                {
                    **terminal_progress,
                    "completed_percent": 100.0,
                    "eta_seconds": 0.0,
                    "terminal_reconciled": True,
                }
            ),
        )
        return selection_path

    def run_pass1(
        self,
        *,
        resume: bool = False,
        diagnostic_stop_after_source_rows: int = 0,
    ) -> Path:
        self._assert_execution_authorized("pass1")
        selection_path = self.output_root / "selection_authority.json"
        if (
            resume
            and self.execution_release is not None
            and self.execution_release.get("confirmatory_one_time") is True
            and (
                not isinstance(self.source, FrozenParquetSourceIdentity)
                or selection_path.is_symlink()
                or not selection_path.is_file()
            )
        ):
            raise PermissionError(
                "one-time confirmatory resume is terminal-only"
            )
        if diagnostic_stop_after_source_rows < 0:
            raise ValueError(
                "diagnostic stop source-row count may not be negative"
            )
        if (
            diagnostic_stop_after_source_rows > 0
            and self.execution_release is not None
            and self.execution_release.get("run", {}).get(
                "diagnostic_stop_allowed"
            )
            is not True
        ):
            raise PermissionError(
                "production Pass1 release forbids diagnostic interruption"
            )
        # A phase-scoped release validates its exact, absent-or-resumable
        # output before source selection, row counting, or iteration.
        self._validate_release_output(
            phase="pass1",
            resume=resume,
        )
        checkpoint = DiscoveryCheckpointStore(
            self.output_root / "_checkpoint" / "pass1"
        )
        bindings = self._bindings(phase="pass1")
        if selection_path.exists() or selection_path.is_symlink():
            if not resume:
                raise FileExistsError("selection authority is immutable")
            state = checkpoint.load(
                expected_bindings=bindings,
                phase="pass1",
            )
            return self._reconcile_pass1_terminal(
                selection_path=selection_path,
                checkpoint=checkpoint,
                state=state,
            )
        if resume:
            state = checkpoint.load(
                expected_bindings=bindings,
                phase="pass1",
            )
        else:
            checkpoint.require_absent()
            state = {
                "replay": self._new_replay(
                    selector=BoundedSemanticSelector(
                        self.audit_contract
                    )
                ),
                "committed_cases": set(),
                "selection_manifest_sha256": None,
                "last_checkpoint_source_rows": 0,
                "verified_source_row_count": None,
            }
        replay: SemanticDiscoveryReplay = state["replay"]
        if replay.selector is None:
            raise ValueError("pass1 checkpoint lacks selector")
        preflight_started = time.monotonic()
        last_checkpoint_monotonic = preflight_started
        if not resume:
            checkpoint_manifest = checkpoint.save(
                state,
                bindings=bindings,
                phase="pass1",
            )
            durable_checkpoint_state_sha256 = str(
                checkpoint_manifest["state_sha256"]
            )
            last_checkpoint_monotonic = time.monotonic()
        else:
            durable_checkpoint_state_sha256 = (
                checkpoint.current_state_sha256()
            )
        durable_progress = self._progress(
            phase="pass1",
            replay=replay,
            total_source_rows=None,
            started=preflight_started,
            starting_rows=replay.source_rows_admitted,
            status="verifying_source",
            durable_checkpoint_state_sha256=(
                durable_checkpoint_state_sha256
            ),
        )
        _atomic_bytes(
            self.output_root / "progress.json",
            _canonical_json(durable_progress),
        )
        self._enforce_resource_limits(
            last_checkpoint_monotonic=last_checkpoint_monotonic,
            source_rows_admitted=replay.source_rows_admitted,
            force=True,
        )
        try:
            total_source_rows = self.source.count_rows(
                on_batch=lambda _rows_scanned: (
                    self._enforce_resource_limits(
                        last_checkpoint_monotonic=(
                            last_checkpoint_monotonic
                        ),
                        source_rows_admitted=(
                            replay.source_rows_admitted
                        ),
                        force=True,
                    )
                )
            )
        except BaseException as exc:
            try:
                _atomic_bytes(
                    self.output_root / "progress.json",
                    _canonical_json(
                        {
                            **durable_progress,
                            "status": "failed",
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                        }
                    ),
                )
            except OSError:
                pass
            raise
        verified_count = state.get("verified_source_row_count")
        if verified_count is None:
            if (
                replay.source_rows_admitted != 0
                or replay.last_source_row_ordinal is not None
            ):
                raise ValueError(
                    "Pass1 admitted rows before source-count verification"
                )
            state["verified_source_row_count"] = total_source_rows
        elif (
            isinstance(verified_count, bool)
            or not isinstance(verified_count, int)
            or verified_count < 0
        ):
            raise ValueError(
                "Pass1 checkpoint has an invalid verified source-row count"
            )
        elif verified_count != total_source_rows:
            raise ValueError(
                "Pass1 source-row count changed since checkpoint"
            )
        if replay.source_rows_admitted > total_source_rows:
            raise ValueError(
                "Pass1 checkpoint cursor exceeds verified source rows"
            )
        started = time.monotonic()
        last_checkpoint_monotonic = started
        starting_rows = replay.source_rows_admitted
        checkpoint_manifest = checkpoint.save(
            state,
            bindings=bindings,
            phase="pass1",
        )
        durable_checkpoint_state_sha256 = str(
            checkpoint_manifest["state_sha256"]
        )
        last_checkpoint_monotonic = time.monotonic()
        durable_progress = self._progress(
            phase="pass1",
            replay=replay,
            total_source_rows=total_source_rows,
            started=started,
            starting_rows=starting_rows,
            status="running",
            durable_checkpoint_state_sha256=(
                durable_checkpoint_state_sha256
            ),
        )
        _atomic_bytes(
            self.output_root / "progress.json",
            _canonical_json(durable_progress),
        )

        def commit(status: str = "running") -> None:
            nonlocal durable_progress
            nonlocal durable_checkpoint_state_sha256
            nonlocal last_checkpoint_monotonic
            self._enforce_resource_limits(
                last_checkpoint_monotonic=last_checkpoint_monotonic,
                source_rows_admitted=replay.source_rows_admitted,
                force=True,
            )
            state["last_checkpoint_source_rows"] = (
                replay.source_rows_admitted
            )
            checkpoint_manifest = checkpoint.save(
                state,
                bindings=bindings,
                phase="pass1",
            )
            durable_checkpoint_state_sha256 = str(
                checkpoint_manifest["state_sha256"]
            )
            last_checkpoint_monotonic = time.monotonic()
            durable_progress = self._progress(
                phase="pass1",
                replay=replay,
                total_source_rows=total_source_rows,
                started=started,
                starting_rows=starting_rows,
                status=status,
                durable_checkpoint_state_sha256=(
                    durable_checkpoint_state_sha256
                ),
            )
            _atomic_bytes(
                self.output_root / "progress.json",
                _canonical_json(durable_progress),
            )

        try:
            for receipt in self._receipts(replay):
                replay.on_receipt(receipt)
                if not receipt.real_source_bar:
                    continue
                self._enforce_resource_limits(
                    last_checkpoint_monotonic=(
                        last_checkpoint_monotonic
                    ),
                    source_rows_admitted=replay.source_rows_admitted,
                )
                due = (
                    replay.source_rows_admitted
                    - int(state["last_checkpoint_source_rows"])
                    >= self.checkpoint_source_rows
                )
                stop = (
                    diagnostic_stop_after_source_rows > 0
                    and replay.source_rows_admitted
                    >= diagnostic_stop_after_source_rows
                )
                if due or stop:
                    commit()
                if stop:
                    raise RuntimeError(
                        "intentional semantic pass1 interruption"
                    )
        except BaseException as exc:
            try:
                _atomic_bytes(
                    self.output_root / "progress.json",
                    _canonical_json(
                        {
                            **durable_progress,
                            "status": "failed",
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                        }
                    ),
                )
            except OSError:
                # Preserve the causal/resource failure when the same storage
                # incident also prevents a best-effort progress update.
                pass
            raise
        if replay.source_rows_admitted != total_source_rows:
            raise RuntimeError("pass1 did not admit every bound source row")
        commit()
        manifest = replay.selector.manifest(
            audit_contract_sha256=self.audit_contract_sha256,
            runner_contract_sha256=self.runner_contract_sha256,
            engine_output_root=self.output_root,
            source_sha256=self.source.sha256,
            source_start=self.source.start,
            source_end_exclusive=self.source.end_exclusive,
            source_rows=replay.source_rows_admitted,
            produced_bars=replay.produced_bars,
            source_prefix_root=replay.source_prefix_root,
            produced_bar_prefix_root=replay.produced_bar_prefix_root,
            implementation_hashes=self.implementation_hashes,
            iterator_bindings=self.iterator_bindings,
        )
        _write_new_json(selection_path, manifest)
        _atomic_bytes(
            self.output_root / "progress.json",
            _canonical_json(
                self._progress(
                    phase="pass1",
                    replay=replay,
                    total_source_rows=total_source_rows,
                    started=started,
                    starting_rows=starting_rows,
                    status=(
                        "complete"
                        if manifest["status"] == "complete"
                        else "unavailable"
                    ),
                    durable_checkpoint_state_sha256=(
                        durable_checkpoint_state_sha256
                    ),
                )
            ),
        )
        if self.execution_release is not None:
            observed = {path.name for path in self.output_root.iterdir()}
            if observed != {
                "_checkpoint",
                "progress.json",
                "selection_authority.json",
            }:
                raise ValueError(
                    "completed Pass1 output tree is not phase-exact"
                )
        return selection_path

    @staticmethod
    def _case_matches_receipt(
        selected: Mapping[str, Any],
        receipt: CompletedBarReceipt,
        replay: SemanticDiscoveryReplay,
    ) -> None:
        expected = {
            "case_bar_synthetic": False,
            "case_source_row_start": receipt.source_row_start.isoformat(),
            "case_source_row_ordinal": receipt.source_row_ordinal,
            "case_source_row_sha256": receipt.source_row_sha256,
            "case_bar_sha256": receipt.bar_sha256,
            "source_prefix_root": replay.source_prefix_root,
            "produced_bar_prefix_root": replay.produced_bar_prefix_root,
            "source_rows_admitted": replay.source_rows_admitted,
            "produced_bars": replay.produced_bars,
            "reset_epoch": replay.reset_epoch,
        }
        primitive = to_primitive(dict(selected))
        for field, value in expected.items():
            if primitive.get(field) != value:
                raise ValueError(
                    f"pass2 case provenance mismatch: {field}"
                )

    def run_pass2(
        self,
        *,
        resume: bool = False,
        diagnostic_stop_after_source_rows: int = 0,
    ) -> Path:
        self._assert_execution_authorized("pass2")
        selection_path = self.output_root / "selection_authority.json"
        if selection_path.is_symlink() or not selection_path.is_file():
            raise ValueError("pass2 requires a trusted selection manifest")
        selection_hash = sha256_file(selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        cases = self._validate_selection_authority(selection)
        by_clock: dict[pd.Timestamp, list[dict[str, Any]]] = {}
        for selected in cases:
            clock = _aware(selected["case_clock"], name="case_clock")
            by_clock.setdefault(clock, []).append(dict(selected))
        global_path = self.output_root / "PASS2_MANIFEST.json"
        if global_path.exists():
            raise FileExistsError("global pass2 manifest is immutable")
        checkpoint = DiscoveryCheckpointStore(
            self.output_root / "_checkpoint" / "pass2"
        )
        bindings = self._bindings(
            phase="pass2",
            selection_manifest_sha256=selection_hash,
        )
        if resume:
            state = checkpoint.load(
                expected_bindings=bindings,
                phase="pass2",
            )
        else:
            checkpoint.require_absent()
            state = {
                "replay": self._new_replay(selector=None),
                "committed_cases": set(),
                "case_commit_hashes": {},
                "selection_manifest_sha256": selection_hash,
                "last_checkpoint_source_rows": 0,
            }
        replay: SemanticDiscoveryReplay = state["replay"]
        if replay.selector is not None:
            raise ValueError("pass2 must not run a selector")
        if state.get("selection_manifest_sha256") != selection_hash:
            raise ValueError("pass2 checkpoint selection changed")
        transaction = CasePacketTransaction(self.output_root / "packets")
        for event_id in sorted(state["committed_cases"]):
            final = (
                transaction.cases_root
                / transaction.case_id(str(event_id))
            )
            payload = transaction._verify_complete(final)
            marker_hash = sha256_file(final / "COMPLETED.json")
            if (
                payload.get("semantic_event_id") != event_id
                or state["case_commit_hashes"].get(event_id)
                != marker_hash
            ):
                raise ValueError(
                    "pass2 checkpoint case transaction changed"
                )
        total_source_rows = self.source.count_rows()
        if int(selection["source_rows"]) != total_source_rows:
            raise ValueError(
                "pass2 source row count differs from selection authority"
            )
        started = time.monotonic()
        starting_rows = replay.source_rows_admitted
        durable_checkpoint_state_sha256 = (
            checkpoint.current_state_sha256()
        )
        durable_progress = {
            **self._progress(
                phase="pass2",
                replay=replay,
                total_source_rows=total_source_rows,
                started=started,
                starting_rows=starting_rows,
                status="running",
                durable_checkpoint_state_sha256=(
                    durable_checkpoint_state_sha256
                ),
            ),
            "committed_cases": len(state["committed_cases"]),
            "total_cases": len(cases),
            "case_completed_percent": (
                100.0
                if not cases
                else round(
                    100.0
                    * len(state["committed_cases"])
                    / len(cases),
                    6,
                )
            ),
        }

        def commit(status: str = "running") -> None:
            nonlocal durable_progress
            nonlocal durable_checkpoint_state_sha256
            state["last_checkpoint_source_rows"] = (
                replay.source_rows_admitted
            )
            checkpoint_manifest = checkpoint.save(
                state,
                bindings=bindings,
                phase="pass2",
            )
            durable_checkpoint_state_sha256 = str(
                checkpoint_manifest["state_sha256"]
            )
            durable_progress = {
                **self._progress(
                    phase="pass2",
                    replay=replay,
                    total_source_rows=total_source_rows,
                    started=started,
                    starting_rows=starting_rows,
                    status=status,
                    durable_checkpoint_state_sha256=(
                        durable_checkpoint_state_sha256
                    ),
                ),
                "committed_cases": len(state["committed_cases"]),
                "total_cases": len(cases),
                "case_completed_percent": (
                    100.0
                    if not cases
                    else round(
                        100.0
                        * len(state["committed_cases"])
                        / len(cases),
                        6,
                    )
                ),
            }
            _atomic_bytes(
                self.output_root / "progress.json",
                _canonical_json(durable_progress),
            )

        try:
            for receipt in self._receipts(replay):
                observation = replay.on_receipt(receipt)
                clock_cases = by_clock.get(observation.asof, ())
                if clock_cases:
                    if not receipt.real_source_bar:
                        raise ValueError(
                            "frozen selection contains a synthetic anchor"
                        )
                    for selected in clock_cases:
                        event_id = str(selected["semantic_event_id"])
                        self._case_matches_receipt(
                            selected,
                            receipt,
                            replay,
                        )
                        prefix_commitment = content_hash(
                            {
                                "source_sha256": self.source.sha256,
                                "source_start": self.source.start,
                                "case_clock": observation.asof,
                                "source_rows": replay.source_rows_admitted,
                                "last_source_row_start": (
                                    receipt.source_row_start
                                ),
                                "last_source_row_sha256": (
                                    receipt.source_row_sha256
                                ),
                                "source_prefix_root": (
                                    replay.source_prefix_root
                                ),
                                "produced_bar_prefix_root": (
                                    replay.produced_bar_prefix_root
                                ),
                                "iterator_bindings": (
                                    self.iterator_bindings
                                ),
                            }
                        )

                        def produce(
                            authority: Path,
                            blind: Path,
                            *,
                            chosen: Mapping[str, Any] = selected,
                            prefix: str = prefix_commitment,
                        ) -> None:
                            materialize_blind_unit(
                                contract=self.audit_contract,
                                selected_case=chosen,
                                observation=observation,
                                histories={
                                    timeframe: replay.reader.window(
                                        timeframe,
                                        self.maximum_history,
                                    )
                                    for timeframe in CORE_TIMEFRAMES
                                },
                                authority_root=authority,
                                blind_root=blind,
                                max_source_time_loaded=(
                                    receipt.source_row_start
                                ),
                                max_engine_time_processed=(
                                    observation.asof
                                ),
                                prefix_source_sha256=prefix,
                                implementation_hashes=(
                                    self.implementation_hashes
                                ),
                                tick_size=(
                                    replay.observer.config.tick_size
                                ),
                                history_capacity=self.maximum_history,
                                reset_epoch=replay.reset_epoch,
                                last_reset_at=replay.last_reset_at,
                                last_reset_reason=(
                                    replay.last_reset_reason
                                ),
                            )

                        _, marker_hash = transaction.publish(
                            semantic_event_id_value=event_id,
                            completion_metadata={
                                "selection_manifest_sha256": (
                                    selection_hash
                                ),
                                "case_clock": observation.asof,
                                "case_source_row_start": (
                                    receipt.source_row_start
                                ),
                                "case_source_row_ordinal": (
                                    receipt.source_row_ordinal
                                ),
                                "case_source_row_sha256": (
                                    receipt.source_row_sha256
                                ),
                                "case_bar_sha256": receipt.bar_sha256,
                                "source_prefix_root": (
                                    replay.source_prefix_root
                                ),
                                "produced_bar_prefix_root": (
                                    replay.produced_bar_prefix_root
                                ),
                                "prefix_commitment": prefix_commitment,
                                "reset_epoch": replay.reset_epoch,
                            },
                            producer=produce,
                        )
                        state["committed_cases"].add(event_id)
                        state["case_commit_hashes"][event_id] = (
                            marker_hash
                        )
                if not receipt.real_source_bar:
                    continue
                due = (
                    replay.source_rows_admitted
                    - int(state["last_checkpoint_source_rows"])
                    >= self.checkpoint_source_rows
                )
                stop = (
                    diagnostic_stop_after_source_rows > 0
                    and replay.source_rows_admitted
                    >= diagnostic_stop_after_source_rows
                )
                if due or clock_cases or stop:
                    commit()
                if stop:
                    raise RuntimeError(
                        "intentional semantic pass2 interruption"
                    )
                if len(state["committed_cases"]) == len(cases):
                    break
        except BaseException as exc:
            _atomic_bytes(
                self.output_root / "progress.json",
                _canonical_json(
                    {
                        **durable_progress,
                        "status": "failed",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    }
                ),
            )
            raise
        expected_ids = {
            str(selected["semantic_event_id"]) for selected in cases
        }
        if state["committed_cases"] != expected_ids:
            raise RuntimeError("pass2 source ended before every case committed")
        commit()
        global_manifest = {
            "format_version": 1,
            "artifact": "v3_semantic_pass2_manifest",
            "audit_id": self.audit_contract["audit_id"],
            "engine_output_root": str(self.output_root.resolve()),
            "selection_manifest_sha256": selection_hash,
            "source_sha256": self.source.sha256,
            "audit_contract_sha256": self.audit_contract_sha256,
            "runner_contract_sha256": self.runner_contract_sha256,
            "implementation_hashes": dict(
                sorted(self.implementation_hashes.items())
            ),
            "iterator_bindings": self.iterator_bindings,
            "case_count": len(cases),
            "final_source_rows_admitted": replay.source_rows_admitted,
            "final_produced_bars": replay.produced_bars,
            "final_source_prefix_root": replay.source_prefix_root,
            "final_produced_bar_prefix_root": (
                replay.produced_bar_prefix_root
            ),
            "final_source_row_ordinal": replay.last_source_row_ordinal,
            "final_source_start": (
                None
                if replay.last_source_bar is None
                else replay.last_source_bar.start
            ),
            "final_causal_clock": (
                None
                if replay.last_observation is None
                else replay.last_observation.asof
            ),
            "final_checkpoint_state_sha256": (
                durable_checkpoint_state_sha256
            ),
            "case_commit_hashes": dict(
                sorted(state["case_commit_hashes"].items())
            ),
        }
        _write_new_json(global_path, global_manifest)
        _atomic_bytes(
            self.output_root / "progress.json",
            _canonical_json(
                {
                    **self._progress(
                        phase="pass2",
                        replay=replay,
                        total_source_rows=total_source_rows,
                        started=started,
                        starting_rows=starting_rows,
                        status="complete",
                        durable_checkpoint_state_sha256=(
                            durable_checkpoint_state_sha256
                        ),
                    ),
                    "committed_cases": len(cases),
                    "total_cases": len(cases),
                    "case_completed_percent": 100.0,
                }
            ),
        )
        return global_path

    def export_blind_review_set(
        self,
        destination: str | Path,
    ) -> Path:
        """Publish into an explicit root outside the engine output tree."""

        pass2_path = self.output_root / "PASS2_MANIFEST.json"
        if pass2_path.is_symlink() or not pass2_path.is_file():
            raise ValueError(
                "blind review export requires completed Pass2"
            )
        return materialize_blind_review_export(
            pass2_manifest_path=pass2_path,
            selection_manifest_path=(
                self.output_root / "selection_authority.json"
            ),
            packets_root=self.output_root / "packets",
            destination=Path(destination),
            contract=self.audit_contract,
        )


__all__ = [
    "BoundedSemanticSelector",
    "CasePacketTransaction",
    "CompletedBarReceipt",
    "DiscoveryCheckpointStore",
    "ParquetBatchSource",
    "PROVENANCE_REAL",
    "PROVENANCE_SYNTHETIC",
    "RUNNER_CONTRACT",
    "SemanticDiscoveryReplay",
    "SemanticDiscoveryRunner",
    "SourceChunk",
    "iter_provenanced_completed_bars",
    "load_runner_contract",
    "observer_from_model_config",
    "source_row_sha256",
]
