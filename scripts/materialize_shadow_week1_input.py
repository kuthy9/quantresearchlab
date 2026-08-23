#!/usr/bin/env python3
"""Materialize the frozen Phase-6 Week-1 clocks for a Phase-9 file rehearsal.

The artifact is a deterministic historical input journal for the existing
``run_shadow_file_pilot.py`` engineering path.  It is deliberately cold-start,
flat-account, zero-order, and non-live.  It reads only the hash-bound Week-1
MBO minute feature artifact and causal OHLCV source; raw MBO and sealed OOS
data are never opened.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat as stat_module
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shadow_file_pilot import (  # noqa: E402
    INPUT_SCHEMA_VERSION,
    shadow_clock_input_from_payload,
    shadow_clock_input_payload,
)
from smc_trader.artifact_stream import (  # noqa: E402
    canonical_json,
    sha256_file,
)
from smc_trader.execution import TopOfBook, TopOfBookExecutionProvider  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.mbo_mechanism import validate_mbo_mechanism_frame  # noqa: E402
from smc_trader.model import AccountState, Bar  # noqa: E402
from smc_trader.shadow_live import (  # noqa: E402
    ShadowClockInput,
    load_shadow_live_protocol,
    shadow_runtime_bindings_from_model_config,
)


MATERIALIZER_SCHEMA_VERSION = "phase9_shadow_week1_materialization_v1"
MATERIALIZER_STATUS = "complete_historical_cold_start_file_input"
BUNDLE_PUBLISH_PROTOCOL = "same_directory_exclusive_output_commit_v1"
WEEK1_MANIFEST = (
    ROOT
    / "experiments/manifests/"
    "smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.yaml"
)
WEEK1_MANIFEST_SHA256 = (
    "77e5c43a157fc9aa5ac27a1f209daa869e07d649284630a186b3a50b71c689c4"
)
MODEL_CONFIG = ROOT / "configs/model.json"
SHADOW_PROTOCOL = ROOT / "configs/shadow_live_v1.json"
SHADOW_FILE_RUNNER = ROOT / "scripts/run_shadow_file_pilot.py"

WINDOW_ID = "2024-06-week-1"
WINDOW_START = pd.Timestamp("2024-06-02T22:00:00Z")
WINDOW_END = pd.Timestamp("2024-06-07T21:01:00Z")
SYMBOL = "NQM4"
INSTRUMENT_ID = 13743
EXPECTED_ROWS = 6900
EXPECTED_REAL_ROWS = 6899
EXPECTED_SYNTHETIC_CLOCKS = (
    pd.Timestamp("2024-06-07T03:10:00Z"),
)
TICK_SIZE = 0.25
POINT_VALUE = 20.0
ACCOUNT_EQUITY = 100_000.0
RECEIVED_DELAY = pd.Timedelta(5, unit="ms")
EXECUTION_DEADLINE = pd.Timedelta(1, unit="h")

_FEATURE_BINDING = "mbo_feature_artifact"
_FEATURE_MANIFEST_BINDING = "mbo_feature_manifest"
_OHLCV_BINDING = "ohlcv_artifact"
_OHLCV_MANIFEST_BINDING = "ohlcv_manifest"
_REQUIRED_FEATURE_COLUMNS = frozenset(
    {
        "decision_time",
        "symbol",
        "instrument_id",
        "book_observed_at",
        "publisher_id",
        "sequence",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "top5_bid_size",
        "top5_ask_size",
        "depth_imbalance",
        "book_valid",
        "book_age_seconds",
    }
)


class ShadowWeek1MaterializationError(ValueError):
    """Raised when the bounded Week-1 input contract fails closed."""


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ShadowWeek1MaterializationError(
                f"JSON contains a duplicate key: {key!r}"
            )
        result[key] = value
    return result


def _regular_file(path: Path, *, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ShadowWeek1MaterializationError(
            f"{name} must be a trusted regular file: {path}"
        )
    return path


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    _regular_file(path, name=name)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_guard,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowWeek1MaterializationError(
            f"{name} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ShadowWeek1MaterializationError(f"{name} root must be an object")
    return payload


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ShadowWeek1MaterializationError(f"{name} is not a timestamp") from exc
    if timestamp.tzinfo is None:
        raise ShadowWeek1MaterializationError(f"{name} must be timezone aware")
    return timestamp.tz_convert("UTC")


def _bound_source(
    bindings: Mapping[str, Any],
    key: str,
) -> tuple[Path, str]:
    raw = bindings.get(key)
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
        raise ShadowWeek1MaterializationError(
            f"Week-1 manifest {key} binding changed"
        )
    relative = raw.get("path")
    expected_hash = raw.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise ShadowWeek1MaterializationError(
            f"Week-1 manifest {key} binding is invalid"
        )
    path = ROOT / relative
    _regular_file(path, name=key)
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ShadowWeek1MaterializationError(
            f"Week-1 manifest {key} SHA-256 differs"
        )
    return path, actual_hash


def _registered_sources(
    manifest_path: str | Path = WEEK1_MANIFEST,
) -> tuple[Mapping[str, Any], dict[str, tuple[Path, str]]]:
    path = Path(manifest_path)
    _regular_file(path, name="Week-1 v5 manifest")
    if sha256_file(path) != WEEK1_MANIFEST_SHA256:
        raise ShadowWeek1MaterializationError(
            "Week-1 v5 manifest SHA-256 differs from the frozen gate"
        )
    manifest = _read_json(path, name="Week-1 v5 manifest")
    primary = manifest.get("windows", {}).get("primary", {})
    census = manifest.get("reader_census_contract", {})
    authority = manifest.get("authority", {})
    contract = manifest.get("contract", {})
    if (
        manifest.get("experiment_id")
        != "smc_semantics_v1_2_phase6_mbo_2024_06_week1_v5"
        or manifest.get("study_mode") != "primary_week_only"
        or manifest.get("frozen_before_run") is not True
        or authority.get("sealed_holdout_opened") is not False
        or primary.get("id") != WINDOW_ID
        or _timestamp(primary.get("start"), name="Week-1 start") != WINDOW_START
        or _timestamp(primary.get("end_exclusive"), name="Week-1 end")
        != WINDOW_END
        or primary.get("expected_rows") != EXPECTED_ROWS
        or contract != {"symbol": SYMBOL, "instrument_id": INSTRUMENT_ID}
        or census.get("window_id") != WINDOW_ID
        or census.get("completed_clocks") != EXPECTED_ROWS
        or census.get("real_completed") != EXPECTED_REAL_ROWS
        or census.get("synthetic_no_trade") != len(EXPECTED_SYNTHETIC_CLOCKS)
        or tuple(
            _timestamp(value, name="registered synthetic clock")
            for value in census.get("synthetic_decision_clocks", ())
        )
        != EXPECTED_SYNTHETIC_CLOCKS
    ):
        raise ShadowWeek1MaterializationError(
            "Week-1 v5 window, contract, authority, or census changed"
        )
    bindings = manifest.get("identity_bindings")
    if not isinstance(bindings, Mapping):
        raise ShadowWeek1MaterializationError(
            "Week-1 v5 identity bindings are missing"
        )
    sources = {
        key: _bound_source(bindings, key)
        for key in (
            _FEATURE_BINDING,
            _FEATURE_MANIFEST_BINDING,
            _OHLCV_BINDING,
            _OHLCV_MANIFEST_BINDING,
        )
    }

    feature_path, feature_hash = sources[_FEATURE_BINDING]
    feature_manifest_path, _ = sources[_FEATURE_MANIFEST_BINDING]
    feature_manifest = _read_json(
        feature_manifest_path,
        name="Week-1 feature manifest",
    )
    output = feature_manifest.get("output", {})
    if (
        feature_manifest.get("artifact_kind")
        != "mbo_minute_mechanism_features"
        or feature_manifest.get("sealed_holdout_read") is not False
        or feature_manifest.get("contract_selection_causal") is not True
        or _timestamp(feature_manifest.get("start"), name="feature start")
        != WINDOW_START
        or _timestamp(
            feature_manifest.get("end_exclusive"),
            name="feature end",
        )
        != WINDOW_END
        or feature_manifest.get("symbol") != SYMBOL
        or feature_manifest.get("instrument_id") != INSTRUMENT_ID
        or output.get("path") != str(feature_path.relative_to(ROOT))
        or output.get("sha256") != feature_hash
        or output.get("rows") != EXPECTED_ROWS
        or output.get("valid_book_rows") != EXPECTED_ROWS
    ):
        raise ShadowWeek1MaterializationError(
            "Week-1 feature manifest contract changed"
        )

    ohlcv_path, _ = sources[_OHLCV_BINDING]
    ohlcv_manifest_path, _ = sources[_OHLCV_MANIFEST_BINDING]
    ohlcv_manifest = _read_json(
        ohlcv_manifest_path,
        name="causal OHLCV manifest",
    )
    if (
        ohlcv_manifest.get("output") != str(ohlcv_path.relative_to(ROOT))
        or ohlcv_manifest.get("current_session_volume_used") is not False
        or ohlcv_manifest.get("selection")
        != "highest total volume from the strictly prior completed Globex session"
    ):
        raise ShadowWeek1MaterializationError(
            "causal OHLCV manifest selection contract changed"
        )
    return manifest, sources


def _week1_bars(path: Path) -> tuple[Bar, ...]:
    loaded = load_ohlcv(path, start=WINDOW_START, end=WINDOW_END)
    if loaded.warnings or not loaded.contract_selection_causal:
        raise ShadowWeek1MaterializationError(
            "Week-1 OHLCV is not the clean causal previous-session front"
        )
    bars = tuple(
        bar
        for bar in iter_completed_bars(loaded.frame)
        if WINDOW_START <= bar.end.tz_convert("UTC") < WINDOW_END
    )
    synthetic = tuple(
        bar.end.tz_convert("UTC") for bar in bars if bar.synthetic_no_trade
    )
    if (
        len(bars) != EXPECTED_ROWS
        or sum(not bar.synthetic_no_trade for bar in bars) != EXPECTED_REAL_ROWS
        or synthetic != EXPECTED_SYNTHETIC_CLOCKS
        or any(bar.data_gap_before_minutes != 0 for bar in bars)
        or {bar.symbol for bar in bars} != {SYMBOL}
        or {bar.instrument_id for bar in bars} != {INSTRUMENT_ID}
        or any(right.end <= left.end for left, right in zip(bars, bars[1:]))
    ):
        raise ShadowWeek1MaterializationError(
            "causal OHLCV replay differs from the frozen Week-1 census"
        )
    return bars


def _week1_features(path: Path) -> pd.DataFrame:
    values = validate_mbo_mechanism_frame(
        pd.read_parquet(path),
        expected_start=WINDOW_START,
        expected_end=WINDOW_END,
        expected_symbol=SYMBOL,
        expected_instrument_id=INSTRUMENT_ID,
        expected_rows=EXPECTED_ROWS,
    )
    if not values["book_valid"].all():
        raise ShadowWeek1MaterializationError(
            "Week-1 shadow input requires one valid BBO for every clock"
        )
    return values


def _finite_number(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ShadowWeek1MaterializationError(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise ShadowWeek1MaterializationError(f"{name} is not finite")
    return result


def _whole_size(value: Any, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if result < 0.0 or not result.is_integer():
        raise ShadowWeek1MaterializationError(
            f"{name} must be a non-negative whole-contract size"
        )
    return result


def _clock_input(bar: Bar, row: Mapping[str, Any]) -> ShadowClockInput:
    decision_clock = bar.end.tz_convert("UTC")
    row_clock = _timestamp(row["decision_time"], name="feature decision_time")
    if (
        row_clock != decision_clock
        or row["symbol"] != bar.symbol
        or int(row["instrument_id"]) != bar.instrument_id
        or bool(row["book_valid"]) is not True
    ):
        raise ShadowWeek1MaterializationError(
            "feature and OHLCV clock/contract join differs"
        )
    observed_at = _timestamp(
        row["book_observed_at"],
        name="book_observed_at",
    )
    if observed_at > decision_clock:
        raise ShadowWeek1MaterializationError("BBO is future-known")
    bid = _finite_number(row["bid"], name="bid")
    ask = _finite_number(row["ask"], name="ask")
    bid_size = _whole_size(row["bid_size"], name="bid_size")
    ask_size = _whole_size(row["ask_size"], name="ask_size")
    top5_bid = _whole_size(row["top5_bid_size"], name="top5_bid_size")
    top5_ask = _whole_size(row["top5_ask_size"], name="top5_ask_size")
    if top5_bid < bid_size or top5_ask < ask_size:
        raise ShadowWeek1MaterializationError(
            "top-5 depth cannot be smaller than the best level"
        )
    top5_total = top5_bid + top5_ask
    if top5_total <= 0.0:
        raise ShadowWeek1MaterializationError("top-5 depth is empty")
    expected_top5_imbalance = (top5_bid - top5_ask) / top5_total
    recorded_top5_imbalance = _finite_number(
        row["depth_imbalance"],
        name="depth_imbalance",
    )
    if not math.isclose(
        recorded_top5_imbalance,
        expected_top5_imbalance,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ShadowWeek1MaterializationError(
            "top-5 depth imbalance differs from its displayed sizes"
        )
    source_age_seconds = float((decision_clock.value - observed_at.value) / 1e9)
    recorded_age = _finite_number(row["book_age_seconds"], name="book_age_seconds")
    if source_age_seconds < 0.0 or not math.isclose(
        recorded_age,
        source_age_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ShadowWeek1MaterializationError(
            "book age differs from the exact observed and decision clocks"
        )
    # ShadowClockInput and TopOfBookExecutionProvider intentionally share the
    # scalar Timedelta API. Validate the nanosecond-exact source field above,
    # then use this same scalar projection on both sides of the live seam.
    shadow_age_seconds = float((decision_clock - observed_at).total_seconds())

    provider = TopOfBookExecutionProvider(tick_size=TICK_SIZE)
    execution = provider.observe(
        TopOfBook(
            observed_at=observed_at,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
        ),
        decision_clock=decision_clock,
        deadline=decision_clock + EXECUTION_DEADLINE,
        direction=None,
        quantity=1,
    )
    # The Phase-6 field is top-five depth imbalance. ShadowClockInput requires
    # the BBO-redundant best-level value, so never copy the top-five scalar.
    expected_best_imbalance = (bid_size - ask_size) / max(
        1.0,
        bid_size + ask_size,
    )
    if (
        not math.isclose(
            float(execution.depth_imbalance),
            expected_best_imbalance,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(execution.data_age_seconds),
            shadow_age_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ShadowWeek1MaterializationError(
            "execution provider did not recompute best-level imbalance and age"
        )

    clock_text = decision_clock.isoformat()
    publisher_id = int(_whole_size(row["publisher_id"], name="publisher_id"))
    sequence = int(_whole_size(row["sequence"], name="sequence"))
    feed_event_id = (
        f"phase9-week1-feed:{SYMBOL}:{INSTRUMENT_ID}:{clock_text}"
    )
    execution_event_id = (
        f"phase9-week1-bbo:{publisher_id}:{sequence}:{clock_text}"
    )
    account_event_id = f"phase9-week1-flat-account:{clock_text}"
    value = ShadowClockInput(
        feed_event_id=feed_event_id,
        received_at=decision_clock + RECEIVED_DELAY,
        bar=bar,
        execution=execution,
        execution_observed_at=observed_at,
        execution_known_at=decision_clock,
        execution_source_event_id=execution_event_id,
        account=AccountState(
            equity=ACCOUNT_EQUITY,
            open_risk_fraction=0.0,
            requested_risk_fraction=0.0,
            quantity=1,
            point_value=POINT_VALUE,
            position=None,
        ),
        account_observed_at=decision_clock,
        account_known_at=decision_clock,
        account_snapshot_id=account_event_id,
        source_event_ids=(
            feed_event_id,
            execution_event_id,
            account_event_id,
        ),
        approved_intents=(),
        execution_events=(),
    )
    if not math.isclose(
        float(value.execution.data_age_seconds),
        shadow_age_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ShadowWeek1MaterializationError("shadow input age is not scalar-exact")
    return value


def build_shadow_week1_payloads(
    bars: Sequence[Bar],
    features: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_ROWS,
    expected_synthetic_clocks: Sequence[pd.Timestamp] = EXPECTED_SYNTHETIC_CLOCKS,
) -> tuple[dict[str, Any], ...]:
    """Build and parse-back the deterministic zero-order clock payloads."""

    values = pd.DataFrame(features).copy()
    missing = sorted(_REQUIRED_FEATURE_COLUMNS - set(values))
    if missing:
        raise ShadowWeek1MaterializationError(
            f"feature projection is missing fields: {missing}"
        )
    values["decision_time"] = pd.to_datetime(
        values["decision_time"],
        errors="coerce",
        utc=True,
    )
    if (
        len(bars) != int(expected_rows)
        or len(values) != int(expected_rows)
        or values["decision_time"].isna().any()
        or values["decision_time"].duplicated().any()
        or not values["decision_time"].is_monotonic_increasing
    ):
        raise ShadowWeek1MaterializationError(
            "Week-1 payload census or feature clock order differs"
        )
    bar_clocks = tuple(bar.end.tz_convert("UTC") for bar in bars)
    feature_clocks = tuple(pd.Timestamp(value) for value in values["decision_time"])
    if bar_clocks != feature_clocks:
        raise ShadowWeek1MaterializationError(
            "feature clocks do not exactly join the causal OHLCV clocks"
        )
    synthetic = tuple(
        bar.end.tz_convert("UTC") for bar in bars if bar.synthetic_no_trade
    )
    expected_synthetic = tuple(
        _timestamp(value, name="expected synthetic clock")
        for value in expected_synthetic_clocks
    )
    if synthetic != expected_synthetic:
        raise ShadowWeek1MaterializationError(
            "synthetic no-trade clock census differs"
        )

    payloads: list[dict[str, Any]] = []
    for bar, row in zip(bars, values.to_dict("records"), strict=True):
        value = _clock_input(bar, row)
        payload = shadow_clock_input_payload(value)
        parsed = shadow_clock_input_from_payload(payload)
        if parsed.input_digest != value.input_digest:
            raise ShadowWeek1MaterializationError(
                "shadow payload parse-back changed the input identity"
            )
        payloads.append(payload)
    if (
        len({item["feed_event_id"] for item in payloads}) != len(payloads)
        or any(
            item["approved_intents"] or item["execution_events"]
            for item in payloads
        )
    ):
        raise ShadowWeek1MaterializationError(
            "materialized shadow identities or zero-order policy changed"
        )
    return tuple(payloads)


def _jsonl(payloads: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(payload) + b"\n" for payload in payloads)


def _sidecar_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + ".manifest.json")


def _bundle_transaction_id() -> str:
    return secrets.token_hex(32)


def _bundle_stage_paths(
    destination: Path,
    transaction_id: str,
) -> tuple[Path, Path]:
    prefix = f".{destination.name}.{transaction_id}"
    return (
        destination.with_name(f"{prefix}.output.staging"),
        destination.with_name(f"{prefix}.manifest.staging"),
    )


@dataclass(frozen=True)
class _OwnedFile:
    path: Path
    device: int
    inode: int


def _owned_file(path: Path) -> _OwnedFile | None:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except (FileNotFoundError, OSError):
        return None
    if not stat_module.S_ISREG(file_stat.st_mode):
        return None
    return _OwnedFile(
        path=path,
        device=int(file_stat.st_dev),
        inode=int(file_stat.st_ino),
    )


def _matches_owned(path: Path, owned: _OwnedFile) -> bool:
    current = _owned_file(path)
    return bool(
        current is not None
        and current.device == owned.device
        and current.inode == owned.inode
    )


def _unlink_owned(path: Path, owned: _OwnedFile) -> bool:
    if not _matches_owned(path, owned):
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _write_exclusive(path: Path, payload: bytes) -> _OwnedFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("xb")
    stat = os.fstat(handle.fileno())
    owned = _OwnedFile(
        path=path,
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
    )
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    except BaseException:
        handle.close()
        _unlink_owned(path, owned)
        raise
    handle.close()
    return owned


def _lock_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.publish.lock")


@contextmanager
def _destination_lock(destination: Path):
    """Serialize recovery, staging and publication across local processes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(destination)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ShadowWeek1MaterializationError(
            "shadow Week-1 destination lock is unavailable"
        ) from exc
    try:
        lock_stat = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_nlink < 1
        ):
            raise ShadowWeek1MaterializationError(
                "shadow Week-1 destination lock is not a regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _commit_metadata(
    destination: Path,
    sidecar: Path,
    transaction_id: str,
) -> dict[str, Any]:
    return {
        "protocol": BUNDLE_PUBLISH_PROTOCOL,
        "transaction_id": transaction_id,
        "same_directory_staging": True,
        "exclusive_no_replace": True,
        "sidecar_published_first": True,
        "output_published_last_commit_marker": True,
        "output_path": str(destination.resolve()),
        "sidecar_path": str(sidecar.resolve()),
    }


def _valid_commit_metadata(
    payload: Mapping[str, Any],
    *,
    destination: Path,
    sidecar: Path,
    transaction_id: str,
) -> bool:
    return payload.get("bundle_commit") == _commit_metadata(
        destination,
        sidecar,
        transaction_id,
    )


def _validate_stage_output(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> bool:
    return bool(
        not path.is_symlink()
        and path.is_file()
        and path.stat().st_size == expected_bytes
        and sha256_file(path) == expected_sha256
    )


def _stage_bundle(
    destination: Path,
    output_bytes: bytes,
    sidecar_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], _OwnedFile, _OwnedFile]:
    sidecar = _sidecar_path(destination)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    output = sidecar_payload.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("path") != str(destination.resolve())
        or output.get("sha256") != output_sha256
        or output.get("bytes") != len(output_bytes)
    ):
        raise ShadowWeek1MaterializationError(
            "shadow Week-1 sidecar does not bind the staged output"
        )
    transaction_id = _bundle_transaction_id()
    output_stage, manifest_stage = _bundle_stage_paths(
        destination,
        transaction_id,
    )
    committed_payload = {
        **dict(sidecar_payload),
        "bundle_commit": _commit_metadata(
            destination,
            sidecar,
            transaction_id,
        ),
    }
    output_owned: _OwnedFile | None = None
    manifest_owned: _OwnedFile | None = None
    try:
        output_owned = _write_exclusive(output_stage, output_bytes)
        manifest_owned = _write_exclusive(
            manifest_stage,
            canonical_json(committed_payload),
        )
    except BaseException:
        if manifest_owned is not None:
            _unlink_owned(manifest_stage, manifest_owned)
        if output_owned is not None:
            _unlink_owned(output_stage, output_owned)
        raise
    return committed_payload, output_owned, manifest_owned


def _recover_interrupted_bundle_locked(destination: Path) -> bool:
    """Recover only an orphan provably hard-linked to our staging files."""

    sidecar = _sidecar_path(destination)
    if not sidecar.exists() or sidecar.is_symlink():
        return False
    try:
        payload = _read_json(sidecar, name="shadow Week-1 sidecar")
    except ShadowWeek1MaterializationError:
        return False
    commit = payload.get("bundle_commit")
    output = payload.get("output")
    if not isinstance(commit, Mapping) or not isinstance(output, Mapping):
        return False
    transaction_id = commit.get("transaction_id")
    expected_sha256 = output.get("sha256")
    expected_bytes = output.get("bytes")
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 64
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or type(expected_bytes) is not int
        or expected_bytes < 0
        or not _valid_commit_metadata(
            payload,
            destination=destination,
            sidecar=sidecar,
            transaction_id=transaction_id,
        )
    ):
        return False
    output_stage, manifest_stage = _bundle_stage_paths(
        destination,
        transaction_id,
    )
    output_owned = _owned_file(output_stage)
    manifest_owned = _owned_file(manifest_stage)
    if (
        output_owned is None
        or manifest_owned is None
        or not _matches_owned(sidecar, manifest_owned)
        or not _validate_stage_output(
            output_stage,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
    ):
        return False
    if destination.exists() or destination.is_symlink():
        if not _matches_owned(destination, output_owned):
            return False
        # The output is the last-published commit marker. Both final files are
        # authoritative; only crash-left staging links need cleanup.
        _unlink_owned(manifest_stage, manifest_owned)
        _unlink_owned(output_stage, output_owned)
        return False
    # A sidecar without the output commit marker is not a published bundle.
    # The shared inode proves this sidecar came from this transaction, so it
    # is safe to remove and retry without touching an unrelated user file.
    if not _unlink_owned(sidecar, manifest_owned):
        return False
    _unlink_owned(manifest_stage, manifest_owned)
    _unlink_owned(output_stage, output_owned)
    return True


def _recover_interrupted_bundle(destination: Path) -> bool:
    with _destination_lock(destination):
        return _recover_interrupted_bundle_locked(destination)


def _publish_bundle_exclusive(
    destination: Path,
    output_bytes: bytes,
    sidecar_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    sidecar = _sidecar_path(destination)
    with _destination_lock(destination):
        _recover_interrupted_bundle_locked(destination)
        if (
            destination.exists()
            or destination.is_symlink()
            or sidecar.exists()
            or sidecar.is_symlink()
        ):
            raise FileExistsError(
                "refusing to overwrite the shadow Week-1 input or sidecar"
            )
        committed, output_owned, manifest_owned = _stage_bundle(
            destination,
            output_bytes,
            sidecar_payload,
        )
        publish_error: BaseException | None = None
        try:
            os.link(
                manifest_owned.path,
                sidecar,
                follow_symlinks=False,
            )
            os.link(
                output_owned.path,
                destination,
                follow_symlinks=False,
            )
        except BaseException as exc:
            publish_error = exc

        # A filesystem wrapper may link successfully and then raise. Re-read
        # both final names by recorded dev+inode before deciding rollback.
        sidecar_linked = _matches_owned(sidecar, manifest_owned)
        output_linked = _matches_owned(destination, output_owned)
        if not (sidecar_linked and output_linked):
            if output_linked:
                _unlink_owned(destination, output_owned)
            if sidecar_linked:
                _unlink_owned(sidecar, manifest_owned)
            _unlink_owned(manifest_owned.path, manifest_owned)
            _unlink_owned(output_owned.path, output_owned)
            if publish_error is not None:
                raise publish_error
            raise ShadowWeek1MaterializationError(
                "shadow Week-1 bundle publication was incomplete"
            )

        _unlink_owned(manifest_owned.path, manifest_owned)
        _unlink_owned(output_owned.path, output_owned)
        return committed


def materialize_shadow_week1_input(
    output_path: str | Path,
    *,
    manifest_path: str | Path = WEEK1_MANIFEST,
) -> Mapping[str, Any]:
    destination = Path(output_path)
    sidecar = _sidecar_path(destination)
    _recover_interrupted_bundle(destination)
    if (
        destination.exists()
        or destination.is_symlink()
        or sidecar.exists()
        or sidecar.is_symlink()
    ):
        raise FileExistsError(
            "refusing to overwrite the shadow Week-1 input or sidecar"
        )

    _, sources = _registered_sources(manifest_path)
    bars = _week1_bars(sources[_OHLCV_BINDING][0])
    features = _week1_features(sources[_FEATURE_BINDING][0])
    payloads = build_shadow_week1_payloads(bars, features)
    output_bytes = _jsonl(payloads)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    parsed_digests = tuple(
        shadow_clock_input_from_payload(payload).input_digest
        for payload in payloads
    )

    for path, name in (
        (MODEL_CONFIG, "current model config"),
        (SHADOW_PROTOCOL, "current shadow protocol"),
        (SHADOW_FILE_RUNNER, "current shadow file runner"),
        (Path(__file__).resolve(), "current Week-1 materializer"),
    ):
        _regular_file(path, name=name)
    shadow_protocol_hash = sha256_file(SHADOW_PROTOCOL)
    protocol = load_shadow_live_protocol(
        SHADOW_PROTOCOL,
        expected_sha256=shadow_protocol_hash,
    )
    mapping = dict(protocol.instrument_mapping)
    if (
        mapping.get("vendor_symbol") != SYMBOL
        or mapping.get("vendor_instrument_id") != INSTRUMENT_ID
        or float(mapping.get("tick_size", 0.0)) != TICK_SIZE
        or float(mapping.get("point_value", 0.0)) != POINT_VALUE
    ):
        raise ShadowWeek1MaterializationError(
            "current shadow instrument mapping differs from Week-1"
        )
    runtime_bindings = dict(
        shadow_runtime_bindings_from_model_config(MODEL_CONFIG)
    )
    source_bindings = {
        key: {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
        }
        for key, (path, digest) in sources.items()
    }
    sidecar_payload = {
        "format_version": 1,
        "schema_version": MATERIALIZER_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "status": MATERIALIZER_STATUS,
        "authority": "historical_engineering_file_input_only",
        "contract": {
            "symbol": SYMBOL,
            "instrument_id": INSTRUMENT_ID,
            "tick_size": TICK_SIZE,
            "point_value": POINT_VALUE,
        },
        "window": {
            "id": WINDOW_ID,
            "start": WINDOW_START.isoformat(),
            "end_exclusive": WINDOW_END.isoformat(),
        },
        "census": {
            "rows": len(payloads),
            "real_completed": sum(not bar.synthetic_no_trade for bar in bars),
            "synthetic_no_trade": sum(bar.synthetic_no_trade for bar in bars),
            "synthetic_decision_clocks": [
                value.isoformat() for value in EXPECTED_SYNTHETIC_CLOCKS
            ],
            "first_decision_clock": bars[0].end.tz_convert("UTC").isoformat(),
            "last_decision_clock": bars[-1].end.tz_convert("UTC").isoformat(),
            "data_gap_resets": 0,
            "contract_changes": 0,
        },
        "source_bindings": {
            "week1_v5_manifest": {
                "path": str(Path(manifest_path).resolve().relative_to(ROOT)),
                "sha256": WEEK1_MANIFEST_SHA256,
            },
            **source_bindings,
        },
        "current_runtime_bindings": {
            "model_config": {
                "path": str(MODEL_CONFIG.relative_to(ROOT)),
                "sha256": sha256_file(MODEL_CONFIG),
            },
            "shadow_protocol": {
                "path": str(SHADOW_PROTOCOL.relative_to(ROOT)),
                "sha256": shadow_protocol_hash,
                "protocol_id": protocol.protocol_id,
            },
            "shadow_file_runner": {
                "path": str(SHADOW_FILE_RUNNER.relative_to(ROOT)),
                "sha256": sha256_file(SHADOW_FILE_RUNNER),
            },
            "input_materializer": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "model_runtime_protocols": runtime_bindings,
        },
        "projection_contract": {
            "execution_provider": "TopOfBookExecutionProvider",
            "source_depth_imbalance": "validated_top5_not_serialized",
            "serialized_depth_imbalance": "recomputed_best_level",
            "data_age_seconds": "scalar_decision_clock_minus_book_observed_at",
            "received_delay_milliseconds": 5,
            "execution_deadline_minutes": 60,
            "account": {
                "equity": ACCOUNT_EQUITY,
                "flat": True,
                "open_risk_fraction": 0.0,
                "requested_risk_fraction": 0.0,
            },
            "approved_intents": 0,
            "execution_events": 0,
        },
        "limits": {
            "cold_start": True,
            "warm_state_restored": False,
            "warmup_history_included": False,
            "real_time_live": False,
            "multi_day_live_pilot": False,
            "broker_submission": False,
            "live_account_state": False,
            "raw_mbo_read": False,
            "sealed_holdout_read": False,
            "phase9_gate_closed": True,
        },
        "output": {
            "path": str(destination.resolve()),
            "sha256": output_sha256,
            "rows": len(payloads),
            "bytes": len(output_bytes),
            "input_digest_sequence_sha256": hashlib.sha256(
                json.dumps(parsed_digests, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
    }
    return _publish_bundle_exclusive(
        destination,
        output_bytes,
        sidecar_payload,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the frozen Week-1 Phase-9 file input",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--week1-manifest",
        default=str(WEEK1_MANIFEST.relative_to(ROOT)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_shadow_week1_input(
        args.output,
        manifest_path=args.week1_manifest,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
