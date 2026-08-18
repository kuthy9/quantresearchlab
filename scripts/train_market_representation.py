#!/usr/bin/env python3
"""Train the optional outcome-blind multi-timeframe representation model.

This script consumes finalized case *input* revisions and builds observable
self-supervised targets from their complete typed-transition ledger.  An
external target file is only an exactly bound cache.  It is not a replay engine
and never imports shadow outcomes, trade results, Decision, Risk, Eye, or
playbooks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.market_representation import (  # noqa: E402
    CAUSAL_CANDLE_FEATURES,
    BAR_END_INDEX_BINDING,
    FEATURE_SCHEMA_VERSION,
    INFERENCE_INPUT_PROTOCOL,
    MODEL_VERSION,
    NEUTRAL_MARKET_TRANSITION_KINDS,
    NEUTRAL_INFERENCE_INPUT_PROTOCOL,
    NEUTRAL_REPRESENTATION_LOSS_WEIGHTS,
    NEUTRAL_SPARSE_ACTIVE_TARGETS,
    NEUTRAL_SPARSE_DISABLED_TARGETS,
    OUTCOME_BLIND_HEAD_WIDTHS,
    TIMEFRAMES,
    CanonicalOHLCVStore,
    CanonicalSourceKey,
    EmbeddingEvaluationSample,
    MarketRepresentationModel,
    ObservableTargetRecord,
    PrefixIndexRange,
    PreparedRepresentationCase,
    RepresentationCase,
    RepresentationDataError,
    SelfSupervisedTarget,
    assign_leakage_safe_splits,
    build_observable_revision_targets,
    build_neutral_market_revision_targets,
    collate_representation_cases,
    compare_reconstruction_to_baselines,
    compare_validation_to_baseline,
    deduplicate_causal_inputs,
    encode_decision_time_head_records,
    encode_decision_time_records,
    encode_market_episode_active_head_records,
    encode_market_episode_records,
    evaluate_outcome_blind_embedding_space,
    majority_class_baselines,
    mask_direct_label_source_tokens,
    neutral_representation_multitask_loss,
    neutral_direct_source_preprocessing_identity,
    prepare_representation_case,
    prepare_neutral_representation_case,
    representation_case_from_case_input_row,
    representation_case_from_market_case_input_row,
    representation_checkpoint_id,
    representation_multitask_loss,
    representation_task_metrics,
    require_torch,
    require_neutral_preprocessed_examples,
    save_neutral_representation_checkpoint,
    save_representation_checkpoint,
    select_first_causal_stage_revisions,
    zero_reconstruction_baselines,
)


LINEAGE_SCHEMA = "smc-canonical-mtf-lineage-v1"
AGGREGATION_PROTOCOL = "smc-existing-causal-aggregation-v1"
EMBEDDING_ARTIFACT_SCHEMA = "smc-decision-time-embeddings-v1"
HEAD_ARTIFACT_SCHEMA = "smc-decision-time-self-supervised-heads-v1"
NEUTRAL_EMBEDDING_ARTIFACT_SCHEMA = "smc-neutral-market-episode-embeddings-v2"
NEUTRAL_HEAD_ARTIFACT_SCHEMA = "smc-neutral-market-episode-active-heads-v2"
NEUTRAL_MATERIAL_SELECTION_CONTRACT = "first_online_market_episode_material_kind_by_revision_index_v1"
DATA_SPLITS = ROOT / "configs/data_splits.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="run a deterministic one-purpose synthetic optimization smoke test",
    )
    parser.add_argument(
        "--case-input-shard",
        action="append",
        default=[],
        help="committed causal_case_input_shards Parquet shard (repeatable)",
    )
    parser.add_argument(
        "--case-input-manifest",
        help="complete causal_case_input_shards manifest binding the supplied shards",
    )
    parser.add_argument(
        "--case-input-manifest-sha",
        help="pre-registered SHA-256 of the case input stream manifest",
    )
    parser.add_argument(
        "--case-run-manifest-sha",
        help="pre-registered SHA-256 of the run manifest bound by the input stream",
    )
    parser.add_argument(
        "--case-library-manifest",
        help="final complete causal_case_library manifest (input metadata only is opened)",
    )
    parser.add_argument(
        "--case-library-manifest-sha",
        help="pre-registered SHA-256 of the finalized causal-case library manifest",
    )
    parser.add_argument(
        "--self-supervised-targets",
        help="separate Parquet/JSONL labels keyed by revision_id",
    )
    parser.add_argument(
        "--market-case-input-manifest", action="append", default=[],
        help="input-only shard manifest; repeat in run-manifest order",
    )
    parser.add_argument(
        "--market-case-run-manifest", action="append", default=[],
        help="bound run manifest; repeat in input-manifest order",
    )
    parser.add_argument("--neutral-dataset-audit-only", action="store_true")
    parser.add_argument(
        "--neutral-single-batch-smoke",
        action="store_true",
        help=(
            "load neutral rows and their manifest-bound source, construct the "
            "canonical five-timeframe features, and run one forward/loss batch "
            "without fitting or exporting"
        ),
    )
    parser.add_argument(
        "--market-embedding-kind", choices=NEUTRAL_MARKET_TRANSITION_KINDS
    )
    parser.add_argument(
        "--neutral-fit", action="store_true",
        help="fit the preregistered three-window smoke or ten-window population",
    )
    parser.add_argument(
        "--canonical-view",
        action="append",
        default=[],
        metavar="MARKET_EPOCH_ID:TIMEFRAME:SOURCE_SHA256=PATH",
        help="epoch/source-qualified canonical completed-bar view (repeatable)",
    )
    parser.add_argument(
        "--canonical-view-sha",
        action="append",
        default=[],
        metavar="MARKET_EPOCH_ID:TIMEFRAME:SOURCE_SHA256=VIEW_SHA256",
        help="trusted manifest content hash for each canonical view",
    )
    parser.add_argument(
        "--canonical-lineage-manifest",
        help="trusted JSON manifest binding source, epoch and derived views",
    )
    parser.add_argument(
        "--canonical-lineage-manifest-sha",
        help="pre-registered SHA-256 of the lineage manifest",
    )
    parser.add_argument(
        "--tick-size",
        action="append",
        default=[],
        metavar="TIMEFRAME=VALUE",
        help="tick size per canonical timeframe",
    )
    parser.add_argument(
        "--availability-time",
        action="append",
        default=[],
        metavar="TIMEFRAME=COLUMN_OR_EXPLICIT_INDEX_TOKEN",
        help=(
            "availability column per timeframe; use "
            f"{BAR_END_INDEX_BINDING!r} only when the index is certified bar-end"
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="independently initialized members; use >=3 for OOD disagreement",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--metrics-output")
    parser.add_argument(
        "--embedding-output",
        help="atomic JSONL export from the first ensemble member/reference encoder",
    )
    parser.add_argument(
        "--head-output",
        help="atomic JSONL export of all independent ensemble-member heads",
    )
    parser.add_argument(
        "--embedding-stage",
        choices=(
            "context_formed",
            "context_changed",
            "episode_created",
            "zone_registered",
            "first_pullback",
            "trigger",
            "plan_formed",
            "terminal",
        ),
        help="explicit online-known revision stage used for export",
    )
    return parser


def _read_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"input shard is absent or not a regular file: {path}")
        if path.suffix.lower() == ".parquet":
            rows.extend(pd.read_parquet(path).to_dict(orient="records"))
        elif path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise RepresentationDataError(
                            f"{path}:{line_number} must contain a JSON object"
                        )
                    rows.append(payload)
        else:
            raise RepresentationDataError(
                f"unsupported shard type {path.suffix!r}; use Parquet or JSONL"
            )
    if not rows:
        raise RepresentationDataError("input shards contain no rows")
    return rows


def _assignment(value: str, *, name: str) -> tuple[str, str]:
    left, separator, right = value.partition("=")
    if not separator or not left.strip() or not right.strip():
        raise RepresentationDataError(f"{name} must use KEY=VALUE syntax")
    return left.strip().lower(), right.strip()


def _identity_assignment(value: str, *, name: str) -> tuple[str, str]:
    left, separator, right = value.partition("=")
    if not separator or not left.strip() or not right.strip():
        raise RepresentationDataError(f"{name} must use KEY=VALUE syntax")
    return left.strip(), right.strip()


def _load_frame(path: str) -> pd.DataFrame:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"canonical view is absent or not a regular file: {source}")
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() in {".csv", ".gz"}:
        frame = pd.read_csv(source, index_col=0)
        frame.index = pd.to_datetime(frame.index)
        return frame
    raise RepresentationDataError("canonical views must be Parquet or CSV")


def _sha256_file(path: str) -> str:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(
            f"canonical view is absent or not a regular file: {source}"
        )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_sha256(value: str | None, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RepresentationDataError(f"{name} must be a lowercase SHA-256")
    return normalized


def _bound_manifest_path(root: Path, value: Any, *, name: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RepresentationDataError(f"{name} must be a relative manifest binding")
    resolved = (root / relative).resolve()
    if root != resolved.parent and root not in resolved.parents:
        raise RepresentationDataError(f"{name} escaped the case run root")
    if resolved.is_symlink() or not resolved.is_file():
        raise RepresentationDataError(f"{name} is missing or not a regular file")
    return resolved


def _canonical_manifest(
    path: str | None,
    *,
    option: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    if not path:
        raise RepresentationDataError(f"neutral dataset mode requires {option}")
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise RepresentationDataError(f"{option} is missing or not a regular file")
    raw = supplied.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError(f"{option} is not valid JSON") from exc
    if not isinstance(payload, Mapping) or raw != json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8"):
        raise RepresentationDataError(f"{option} is not canonical JSON")
    return supplied.resolve(), actual_sha, dict(payload)


def _neutral_bound_file(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RepresentationDataError(f"{name} must be a relative path")
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RepresentationDataError(f"{name} must be a relative path")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise RepresentationDataError(f"{name} is missing")
    resolved = path.resolve()
    if root != resolved.parent and root not in resolved.parents:
        raise RepresentationDataError(f"{name} escaped its run root")
    return resolved


def _neutral_run_identity(
    run: Mapping[str, Any], *,
    identity_cache: dict[tuple[str, str, str], Any] | None = None,
) -> Mapping[str, Any]:
    """Verify each run-constant source/config identity exactly once."""

    from smc_trader.market_representation import _validated_market_case_run_manifest

    normalized = _validated_market_case_run_manifest(run)
    repository = run.get("repository")
    cache = identity_cache if identity_cache is not None else {}
    config_path = Path(normalized["model_config_path"])
    config_key = ("config", str(config_path), normalized["model_config_sha256"])
    try:
        cached_config = cache.get(config_key)
        if cached_config is None:
            config_raw = config_path.read_bytes()
            if hashlib.sha256(config_raw).hexdigest() != normalized[
                "model_config_sha256"
            ]:
                raise RepresentationDataError(
                    "neutral run model_config hash mismatch"
                )
            cached_config = json.loads(config_raw.decode("utf-8"))
            cache[config_key] = cached_config
        config_payload = cached_config
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError("neutral run model config is unreadable") from exc
    source_path = Path(normalized["source_path"])
    if source_path.is_symlink() or not source_path.is_file():
        raise RepresentationDataError("neutral run source is missing")
    source_key = ("source", str(source_path), normalized["source_sha256"])
    if source_key not in cache:
        if _sha256_file(str(source_path)) != normalized["source_sha256"]:
            raise RepresentationDataError("neutral run source hash mismatch")
        cache[source_key] = True
    if not isinstance(config_payload, Mapping):
        raise RepresentationDataError("neutral run model config must be an object")
    try:
        config_matches = (
            int(config_payload["schema_version"])
            == normalized["model_config_schema_version"]
            and float(config_payload["tick_size"]) == normalized["tick_size"]
            and str(config_payload["timezone"]) == normalized["timezone"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RepresentationDataError(
            "neutral run model config identity is incomplete"
        ) from exc
    if not config_matches:
        raise RepresentationDataError(
            "neutral run model config values differ from its manifest identity"
        )
    return {
        "profile_identity": str(run["profile"]["identity"]),
        "source_sha256": normalized["source_sha256"],
        "model_config_sha256": normalized["model_config_sha256"],
        "source_identity_verified_once": True,
        "model_config_identity_verified_once": True,
        "calibration_fit_allowed": False,
        "normalized": normalized,
        "model_config": dict(config_payload),
        "repository": None if repository is None else dict(repository),
    }


def _load_neutral_market_dataset(
    *, input_manifest_path: str | None, run_manifest_path: str | None,
    identity_cache: dict[tuple[str, str, str], Any] | None = None,
) -> Mapping[str, Any]:
    """Load one finalized neutral stream; no library or outcome is consulted."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    from smc_trader.artifact_stream import _arrow_schema
    from smc_trader.market_cases import (
        MARKET_CASE_INPUT_FIELD_TYPES,
        validate_market_case_rows,
    )

    input_path, input_sha, manifest = _canonical_manifest(
        input_manifest_path,
        option="--market-case-input-manifest",
    )
    run_path, run_sha, run = _canonical_manifest(
        run_manifest_path,
        option="--market-case-run-manifest",
    )
    run_identity = _neutral_run_identity(run, identity_cache=identity_cache)
    keys = {
        "format_version", "artifact", "status", "stream", "rows", "shards",
        "bindings", "schema_fingerprint", "field_types",
    }
    fingerprint = hashlib.sha256(
        json.dumps(dict(MARKET_CASE_INPUT_FIELD_TYPES), sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()
    bindings = manifest.get("bindings")
    if (
        set(manifest) != keys or manifest.get("format_version") != 1
        or manifest.get("artifact")
        != "continuous_development_market_case_input_shards"
        or manifest.get("status") != "complete"
        or manifest.get("stream") != "market_case_input_shards"
        or manifest.get("field_types") != dict(MARKET_CASE_INPUT_FIELD_TYPES)
        or manifest.get("schema_fingerprint") != fingerprint
        or not isinstance(bindings, Mapping) or set(bindings) != {"run_manifest"}
    ):
        raise RepresentationDataError("neutral input manifest identity is invalid")
    if _neutral_bound_file(input_path.parent, bindings["run_manifest"],
                           name="run binding") != run_path:
        raise RepresentationDataError("neutral input binds another run manifest")
    shards, declared = manifest.get("shards"), manifest.get("rows")
    if type(declared) is not int or declared < 1 or not isinstance(shards, list) or not shards:
        raise RepresentationDataError("neutral input manifest contains no rows")

    expected_schema = _arrow_schema(MARKET_CASE_INPUT_FIELD_TYPES)
    expected_columns = list(MARKET_CASE_INPUT_FIELD_TYPES)
    rows: list[dict[str, Any]] = []
    shard_keys = {"index", "path", "rows", "first_key", "last_key", "sha256"}
    for index, shard in enumerate(shards):
        expected_path = Path("market_case_input_shards") / f"part-{index:05d}.parquet"
        if (
            not isinstance(shard, Mapping) or set(shard) != shard_keys
            or type(shard["index"]) is not int or shard["index"] != index
            or type(shard["rows"]) is not int or shard["rows"] < 1
            or Path(str(shard["path"])) != expected_path
        ):
            raise RepresentationDataError("neutral input shard metadata is invalid")
        path = _neutral_bound_file(input_path.parent, shard["path"], name="input shard")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != _normalized_sha256(
            shard["sha256"], name="input shard sha256"
        ):
            raise RepresentationDataError("neutral input shard hash mismatch")
        try:
            table = pq.read_table(pa.BufferReader(raw))
        except Exception as exc:
            raise RepresentationDataError("neutral input shard is invalid Parquet") from exc
        part = table.to_pylist()
        if (
            table.schema.remove_metadata() != expected_schema
            or table.column_names != expected_columns or table.num_rows != shard["rows"]
            or part[0]["revision_id"] != shard["first_key"]
            or part[-1]["revision_id"] != shard["last_key"]
        ):
            raise RepresentationDataError("neutral input shard schema/binding changed")
        rows.extend(part)
    if len(rows) != declared or sum(shard["rows"] for shard in shards) != declared:
        raise RepresentationDataError("neutral input row counts are inconsistent")
    try:
        validate_market_case_rows(rows)
    except (TypeError, ValueError) as exc:
        raise RepresentationDataError("neutral input continuity is invalid") from exc
    return {
        "rows": tuple(rows), "input_path": input_path, "input_sha": input_sha,
        "input_manifest": manifest, "run_path": run_path, "run_sha": run_sha,
        "run_manifest": run,
        "run_identity": run_identity,
    }


def _neutral_profile_sha256(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(profile), sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _neutral_split_registry() -> tuple[Any, Mapping[str, Any], str]:
    from smc_trader.validation import ValidationProtocolError, load_validation_protocol

    try:
        raw = DATA_SPLITS.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        protocol = load_validation_protocol(DATA_SPLITS)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            ValidationProtocolError) as exc:
        raise RepresentationDataError("neutral split registry is invalid") from exc
    profiles = payload.get("market_case_input_profiles")
    if not isinstance(profiles, Mapping):
        raise RepresentationDataError("neutral profiles are missing")
    return (protocol.neutral_representation_splits, profiles,
            hashlib.sha256(raw).hexdigest())


def _neutral_episode_prefix_exposure(
    dataset: Mapping[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    from smc_trader.io import load_ohlcv

    normalized = dataset["run_identity"]["normalized"]
    try:
        loaded = load_ohlcv(
            normalized["source_path"],
            start=normalized["source_first"],
            end=normalized["source_last"] + pd.Timedelta(minutes=1),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RepresentationDataError("neutral prefix source cannot be loaded") from exc
    frame = loaded.frame
    if len(frame) != normalized["source_rows"] or not frame.index.is_monotonic_increasing:
        raise RepresentationDataError("neutral prefix source identity changed")
    contracts = frame.loc[:, ["symbol", "instrument_id"]].drop_duplicates()
    expected_contract = (normalized["symbol"], normalized["instrument_id"])
    if len(contracts) != 1 or (
        str(contracts.iloc[0]["symbol"]),
        int(contracts.iloc[0]["instrument_id"]),
    ) != expected_contract:
        raise RepresentationDataError("neutral replay frame must contain one contract")

    run_sha = str(dataset["run_sha"])
    episodes: dict[
        tuple[str, str, str], tuple[pd.Timestamp, pd.Timestamp]
    ] = {}
    for row in dataset["rows"]:
        try:
            raw_prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
            ranges = {
                (
                    int(prefix["replay_view_1m_row_start"]),
                    int(prefix["replay_view_1m_row_end_exclusive"]),
                )
                for prefix in raw_prefixes
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RepresentationDataError("neutral prefix binding is invalid") from exc
        if len(ranges) != 1:
            raise RepresentationDataError("neutral prefix differs across timeframes")
        start_index, end_index = next(iter(ranges))
        if start_index < 0 or end_index <= start_index or end_index > len(frame):
            raise RepresentationDataError("neutral prefix exceeds its replay frame")
        start_at = pd.Timestamp(frame.index[start_index])
        end_at = pd.Timestamp(frame.index[end_index - 1]) + pd.Timedelta(minutes=1)
        key = (
            run_sha,
            str(row["market_epoch_id"]),
            str(row["market_episode_id"]),
        )
        prior = episodes.get(key)
        episodes[key] = (
            start_at if prior is None else min(start_at, prior[0]),
            end_at if prior is None else max(end_at, prior[1]),
        )
    if not episodes:
        raise RepresentationDataError("neutral fit dataset contains no MarketEpisode")
    return (min(value[0] for value in episodes.values()),
            max(value[1] for value in episodes.values()))


def _observed_completed_sessions(
    source_path: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[str, ...]:
    from smc_trader.io import load_ohlcv
    from smc_trader.market_clock import _session_bounds, is_registered_trading_minute

    if end <= start:
        return ()
    try:
        frame = load_ohlcv(source_path, start=start, end=end).frame
    except (OSError, TypeError, ValueError) as exc:
        raise RepresentationDataError("neutral embargo source cannot be loaded") from exc
    if frame.empty:
        return ()
    local = pd.DatetimeIndex(frame.index).tz_convert("America/New_York")
    # Shift Globex 18:00-17:00 bars onto one local session label.
    labels = (local + pd.Timedelta(hours=6)).normalize()
    complete: list[str] = []
    for raw_label in sorted(set(labels)):
        label = pd.Timestamp(raw_label).date()
        bounds = _session_bounds(label)
        if bounds is None:
            continue
        opened, closed = bounds
        if opened < start or closed > end:
            continue
        actual = local[labels == raw_label]
        expected = pd.DatetimeIndex(
            stamp
            for stamp in pd.date_range(
                opened,
                closed - pd.Timedelta(minutes=1),
                freq="1min",
            )
            if is_registered_trading_minute(stamp)
        )
        if actual.equals(expected):
            complete.append(label.isoformat())
    return tuple(complete)


def _validate_neutral_actual_exposure_boundaries(
    datasets: Sequence[Mapping[str, Any]],
    registry: Any,
) -> tuple[Mapping[str, Any], ...]:
    ordered = sorted(datasets, key=lambda item: item["registered_window"].warmup_start)
    observed: list[Mapping[str, Any]] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        purge_end = left["prefix_exposure_end"] + pd.DateOffset(
            days=registry.purge_calendar_days
        )
        if right["prefix_exposure_start"] <= purge_end:
            raise RepresentationDataError("neutral prefix violates 14-day purge/embargo")
        sessions = _observed_completed_sessions(
            left["run_identity"]["normalized"]["source_path"],
            start=purge_end,
            end=right["prefix_exposure_start"],
        )
        if len(sessions) < registry.embargo_trading_days:
            raise RepresentationDataError(
                "neutral prefix embargo has fewer than five completed sessions"
            )
        observed.append(
            {
                "left_profile": left["profile_name"],
                "right_profile": right["profile_name"],
                "purge_end": purge_end.isoformat(),
                "next_prefix_start": right["prefix_exposure_start"].isoformat(),
                "observed_session_count": len(sessions),
                "first_observed_session": sessions[0],
                "last_observed_session": sessions[-1],
            }
        )
    return tuple(observed)


def _load_neutral_fit_collection(
    *,
    input_manifest_paths: Sequence[str],
    run_manifest_paths: Sequence[str],
) -> Mapping[str, Any]:
    if len(input_manifest_paths) != len(run_manifest_paths) or len(input_manifest_paths) < 3:
        raise RepresentationDataError(
            "neutral fit requires paired input/run manifests for all three roles"
        )
    registry, raw_profiles, registry_sha = _neutral_split_registry()
    identity_cache: dict[tuple[str, str, str], Any] = {}
    datasets: list[dict[str, Any]] = []
    selected_profiles: set[str] = set()
    revision_origins: dict[str, str] = {}
    for input_path, run_path in zip(input_manifest_paths, run_manifest_paths, strict=True):
        loaded = dict(
            _load_neutral_market_dataset(
                input_manifest_path=input_path,
                run_manifest_path=run_path,
                identity_cache=identity_cache,
            )
        )
        run = loaded["run_manifest"]
        profile_name = str(run["profile"]["name"])
        profile = raw_profiles.get(profile_name)
        window = registry.windows.get(profile_name)
        if not isinstance(profile, Mapping) or window is None:
            raise RepresentationDataError(
                "neutral fit run uses an unregistered representation profile"
            )
        if run["profile"]["identity"] != _neutral_profile_sha256(profile) or (
            profile.get("representation_split_role") != window.representation_split_role
        ):
            raise RepresentationDataError(
                "neutral fit profile identity or role was tampered"
            )
        normalized = loaded["run_identity"]["normalized"]
        if (
            normalized["window_start"] != window.start
            or normalized["window_end_exclusive"] != window.end_exclusive
            or normalized["window_role"] != window.allowed_ohlcv_role
            or run["window"]["warmup_days"] != registry.warmup_calendar_days
            or normalized["symbol"] != window.expected_symbol
            or normalized["instrument_id"] != window.expected_instrument_id
        ):
            raise RepresentationDataError("neutral fit run differs from its registry")
        if loaded["run_identity"]["repository"] is None:
            raise RepresentationDataError("neutral fit run must bind one commit")
        if profile_name in selected_profiles:
            raise RepresentationDataError("neutral fit profile is duplicated")
        selected_profiles.add(profile_name)
        role = window.representation_split_role
        for row in loaded["rows"]:
            revision_id = str(row["revision_id"])
            if revision_id in revision_origins:
                raise RepresentationDataError(
                    "neutral fit revision identity is duplicated across runs"
                )
            revision_origins[revision_id] = loaded["run_sha"]
        loaded.update(profile_name=profile_name,
                      representation_split_role=role, registered_window=window)
        datasets.append(loaded)

    smoke_profiles = set(registry.smoke_profiles.values())
    full_profiles = set(registry.windows)
    if selected_profiles == smoke_profiles:
        scope = "three_window_pipeline_smoke"
    elif selected_profiles == full_profiles:
        scope = "ten_window_registered_fit"
    else:
        raise RepresentationDataError("neutral fit population is not preregistered")
    if {item["representation_split_role"] for item in datasets} != {
        "train", "validation", "holdout"
    }:
        raise RepresentationDataError("neutral fit requires all three roles")
    compatibility = {
        "source_sha256": lambda item: item["run_identity"]["normalized"]["source_sha256"],
        "model_config_sha256": lambda item: item["run_identity"]["normalized"]["model_config_sha256"],
        "market_case_protocol": lambda item: json.dumps(
            item["run_manifest"]["market_case_input_identity"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        ),
    }
    identities = {name: {getter(item) for item in datasets}
                  for name, getter in compatibility.items()}
    if any(len(values) != 1 for values in identities.values()):
        raise RepresentationDataError(
            "neutral fit runs have incompatible source/config/protocol identity"
        )

    for dataset in datasets:
        start, end = _neutral_episode_prefix_exposure(dataset)
        window = dataset["registered_window"]
        if start < window.warmup_start or end > window.end_exclusive:
            raise RepresentationDataError("neutral prefix escaped its replay window")
        dataset.update(prefix_exposure_start=start, prefix_exposure_end=end)
    observed_embargo_sessions = _validate_neutral_actual_exposure_boundaries(
        datasets, registry
    )
    return {
        "datasets": tuple(datasets),
        "scope": scope,
        "registry": registry,
        "registry_sha256": registry_sha,
        "source_sha256": next(iter(identities["source_sha256"])),
        "model_config_sha256": next(iter(identities["model_config_sha256"])),
        "timezone": datasets[0]["run_identity"]["normalized"]["timezone"],
        "observed_embargo_sessions": observed_embargo_sessions,
    }


def _neutral_dataset_audit_report(
    dataset: Mapping[str, Any], *, embedding_kind: str | None,
) -> Mapping[str, Any]:
    from smc_trader.market_representation import (
        NEUTRAL_SPARSE_ACTIVE_TARGETS,
        NEUTRAL_SPARSE_DISABLED_TARGETS,
        build_neutral_market_revision_targets,
    )

    rows, run = dataset["rows"], dataset["run_manifest"]
    targets = build_neutral_market_revision_targets(rows, run)
    if set(targets) != {str(row["revision_id"]) for row in rows}:
        raise RepresentationDataError("neutral targets do not cover revisions")
    kinds = Counter({kind: 0 for kind in NEUTRAL_MARKET_TRANSITION_KINDS})
    directions: Counter[str] = Counter()
    cases: dict[tuple[str, str], str] = {}
    scales: Counter[str] = Counter()
    scale_state = {scale: Counter() for scale in ("4H", "1H", "15m", "5m", "1m")}
    active = Counter()
    eligible = 0
    for row in rows:
        row_kinds = json.loads(str(row["transition_kinds_json"]))
        prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
        context = json.loads(str(row["neutral_global_context_json"]))
        kinds.update(row_kinds)
        eligible += int(embedding_kind is not None and embedding_kind in row_kinds)
        direction = str(row["direction"])
        directions[direction] += 1
        key = (str(row["market_epoch_id"]), str(row["market_episode_id"]))
        if cases.setdefault(key, direction) != direction:
            raise RepresentationDataError("neutral episode direction changed")
        scales.update(prefix["timeframe"] for prefix in prefixes)
        details = context["scale_relation_details"]
        row_ambiguous = row_unknown = False
        for scale, counts in scale_state.items():
            detail = details.get(scale)
            present = isinstance(detail, Mapping)
            ambiguous = present and detail.get("ambiguous") is True
            unknown = present and (
                detail.get("relation") == "unknown"
                or detail.get("graph_connected") is False
            )
            counts.update(present=int(present), ambiguous=int(ambiguous), unknown=int(unknown))
            row_ambiguous, row_unknown = row_ambiguous or ambiguous, row_unknown or unknown
        active.update(ambiguous=int(row_ambiguous), unknown=int(row_unknown))
    manifest = dataset["input_manifest"]
    return {
        "schema_version": 1, "mode": "neutral_market_dataset_audit",
        "audit_only": True, "calibration_fit_allowed": False,
        "training_performed": False, "rows": len(rows), "cases": len(cases),
        "kinds": dict(kinds),
        "directions": {
            "rows": dict(directions), "cases": dict(Counter(cases.values()))
        },
        "scales": dict(scales),
        "active_scale_state": {
            "denominator_rows": len(rows), "ambiguous_rows": active["ambiguous"],
            "unknown_or_disconnected_rows": active["unknown"],
            "by_timeframe": {
                scale: {
                    "rows_with_detail": counts["present"],
                    "ambiguous_rows": counts["ambiguous"],
                    "ambiguous_ratio": counts["ambiguous"] / len(rows),
                    "unknown_or_disconnected_rows": counts["unknown"],
                    "unknown_or_disconnected_ratio": counts["unknown"] / len(rows),
                }
                for scale, counts in scale_state.items()
            },
        },
        "embedding_kind": embedding_kind,
        "embedding_kind_rows": eligible if embedding_kind is not None else None,
        "target_coverage": {
            "records": len(targets),
            "next_lifecycle": sum(record.next_revision_id is not None for record in targets.values()),
            "scale_direction_alignment": sum(
                record.target.scale_direction_alignment >= 0 for record in targets.values()
            ),
            "active": list(NEUTRAL_SPARSE_ACTIVE_TARGETS),
            "disabled": list(NEUTRAL_SPARSE_DISABLED_TARGETS),
        },
        "stream_lineage": {
            "manifest": str(dataset["input_path"]), "sha256": dataset["input_sha"],
            "rows": manifest["rows"], "shards": len(manifest["shards"]),
            "schema_fingerprint": manifest["schema_fingerprint"],
        },
        "run_lineage": {
            "manifest": str(dataset["run_path"]), "sha256": dataset["run_sha"],
            "profile": dict(run["profile"]), "window_role": run["window"]["role"],
            "source_sha256": run["source"]["sha256"],
            "model_config_sha256": run["model_config"]["sha256"],
            "identity_verification": {
                key: value
                for key, value in dataset["run_identity"].items()
                if key not in {"normalized", "model_config"}
            },
        },
    }


def _neutral_canonical_store(
    dataset: Mapping[str, Any],
    cases: Sequence[RepresentationCase],
) -> tuple[CanonicalOHLCVStore, Mapping[str, Any]]:
    """Rebuild the runner's completed-bar views from its one bound 1m source."""

    from smc_trader.causal import CausalMarketReader
    from smc_trader.io import iter_completed_bars, load_ohlcv
    from smc_trader.market_representation import normalize_timeframe
    from smc_trader.scene_graph import parse_scale_specs

    if not cases:
        raise RepresentationDataError("neutral dataset requires at least one case")
    epochs = {case.market_epoch_id for case in cases}
    identity = dataset["run_identity"]
    normalized, config = identity["normalized"], identity["model_config"]
    scales = config.get("scales")
    if not isinstance(scales, list):
        raise RepresentationDataError("neutral model config omits scale registry")
    try:
        expanded_scales = [
            {**item, "history_limit": max(1024, normalized["source_rows"] * 2)}
            for item in scales
            if isinstance(item, Mapping)
        ]
        specs = parse_scale_specs(expanded_scales)
    except (TypeError, ValueError) as exc:
        raise RepresentationDataError("neutral model scale registry is invalid") from exc
    enabled = {
        normalize_timeframe(spec.native_timeframe.value)
        for spec in specs
        if spec.enabled and spec.native_timeframe is not None
    }
    if enabled != set(TIMEFRAMES):
        raise RepresentationDataError(
            "neutral model config must enable exactly the canonical five timeframes"
        )

    source_last_exclusive = normalized["source_last"] + pd.Timedelta(minutes=1)
    try:
        loaded = load_ohlcv(
            normalized["source_path"],
            start=normalized["source_first"],
            end=source_last_exclusive,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RepresentationDataError("neutral source window cannot be loaded") from exc
    if len(loaded.frame) != normalized["source_rows"]:
        raise RepresentationDataError("neutral source row identity changed")
    contracts = loaded.frame.loc[:, ["symbol", "instrument_id"]].drop_duplicates()
    if len(contracts) != 1 or (
        str(contracts.iloc[0]["symbol"]), int(contracts.iloc[0]["instrument_id"])
    ) != (normalized["symbol"], normalized["instrument_id"]):
        raise RepresentationDataError("neutral source contract identity changed")

    reader = CausalMarketReader(scale_specs=specs)
    completed: dict[str, list[Any]] = {timeframe: [] for timeframe in TIMEFRAMES}
    completed_1m = 0
    try:
        for bar in iter_completed_bars(loaded.frame, allow_data_gap_reset=False):
            update = reader.on_bar(bar)
            completed_1m += 1
            for timeframe, candles in update.newly_completed.items():
                completed[normalize_timeframe(timeframe.value)].extend(candles)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RepresentationDataError(
            "neutral source cannot reproduce the completed-bar clock"
        ) from exc
    if any(not completed[timeframe] for timeframe in TIMEFRAMES):
        raise RepresentationDataError("neutral source has an empty canonical timeframe")

    source_sha = normalized["source_sha256"]
    frames: dict[CanonicalSourceKey, pd.DataFrame] = {}
    ticks: dict[CanonicalSourceKey, float] = {}
    availability: dict[CanonicalSourceKey, str] = {}
    canonical_frames: dict[str, pd.DataFrame] = {}
    for timeframe in TIMEFRAMES:
        candles = completed[timeframe]
        canonical_frames[timeframe] = pd.DataFrame(
            {
                "open": [float(candle.open) for candle in candles],
                "high": [float(candle.high) for candle in candles],
                "low": [float(candle.low) for candle in candles],
                "close": [float(candle.close) for candle in candles],
                "volume": [float(candle.volume) for candle in candles],
            },
            index=pd.DatetimeIndex([candle.end for candle in candles]),
        )
    for epoch in epochs:
        for timeframe, frame in canonical_frames.items():
            key = CanonicalSourceKey(epoch, timeframe, source_sha)
            frames[key] = frame
            ticks[key] = float(normalized["tick_size"])
            availability[key] = BAR_END_INDEX_BINDING
    store = CanonicalOHLCVStore(
        frames,
        tick_sizes=ticks,
        availability_bindings=availability,
    )
    return store, {
        "source_rows": len(loaded.frame),
        "completed_1m_bars": completed_1m,
        "market_epochs": len(epochs),
        "canonical_rows": {
            timeframe: len(completed[timeframe]) for timeframe in TIMEFRAMES
        },
        "availability_binding": BAR_END_INDEX_BINDING,
    }


def _neutral_single_batch_smoke_report(
    dataset: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Exercise one neutral forward/loss batch without fitting or exporting."""

    require_torch()
    import torch
    from smc_trader.market_representation import (
        NEUTRAL_SPARSE_ACTIVE_TARGETS,
        NEUTRAL_SPARSE_DISABLED_TARGETS,
        build_neutral_market_revision_targets,
        representation_case_from_market_case_input_row,
    )

    rows, run = dataset["rows"], dataset["run_manifest"]
    cases = tuple(
        representation_case_from_market_case_input_row(row, run) for row in rows
    )
    target_records = build_neutral_market_revision_targets(rows, run)
    if set(target_records) != {case.revision_id for case in cases}:
        raise RepresentationDataError("neutral smoke target coverage changed")
    store, source_summary = _neutral_canonical_store(dataset, cases)
    examples = tuple(
        prepare_neutral_representation_case(case, store) for case in cases
    )
    targets = tuple(target_records[case.revision_id].target for case in cases)

    seed = 17
    torch.manual_seed(seed)
    model = MarketRepresentationModel()
    model.eval()
    batch, target_batch = collate_representation_cases(
        examples,
        targets=targets,
        mask_probability=0.15,
        seed=seed,
    )
    assert target_batch is not None
    with torch.no_grad():
        output = model(batch)
        breakdown = representation_multitask_loss(output, batch, target_batch)
    if not bool(torch.isfinite(breakdown.total)):
        raise RepresentationDataError("neutral single-batch smoke loss is non-finite")
    component_losses = {
        name: float(value.detach().cpu())
        for name, value in breakdown.components.items()
    }
    if any(
        component_losses[name] != 0.0
        for name in ("next_event", "next_event_time", "displacement", "draw_consumed")
    ):
        raise RepresentationDataError("neutral smoke enabled a disabled legacy target")
    feature_max = max(example.feature_max_at for example in examples)
    input_max = max(case.asof for case in cases)
    if feature_max > input_max:
        raise RepresentationDataError("neutral smoke constructed future features")
    return {
        "schema_version": 1,
        "mode": "neutral_market_single_batch_smoke",
        "rows": len(rows),
        "cases": len(cases),
        "batches": 1,
        "batch_rows": len(cases),
        "canonical_timeframes": list(TIMEFRAMES),
        "candle_feature_width": len(CAUSAL_CANDLE_FEATURES),
        "feature_max_at": feature_max.isoformat(),
        "input_max_asof": input_max.isoformat(),
        "target_contract": {
            "active": list(NEUTRAL_SPARSE_ACTIVE_TARGETS),
            "disabled": list(NEUTRAL_SPARSE_DISABLED_TARGETS),
            "next_lifecycle_labelled": sum(
                target.next_lifecycle >= 0 for target in targets
            ),
            "scale_direction_alignment_labelled": sum(
                target.scale_direction_alignment >= 0 for target in targets
            ),
        },
        "embedding_shape": list(output.embedding.shape),
        "loss": float(breakdown.total.detach().cpu()),
        "loss_components": component_losses,
        "model_parameter_count": model.parameter_count(),
        "source": source_summary,
        "optimizer_created": False,
        "backward_performed": False,
        "training_performed": False,
        "checkpoint_written": False,
        "artifacts_exported": False,
        "outcome_fields_used": False,
        "brain_used": False,
        "shadow_used": False,
    }


def _prepare_neutral_fit_collection(
    collection: Mapping[str, Any],
) -> Mapping[str, Any]:
    examples: list[PreparedRepresentationCase] = []
    targets: list[SelfSupervisedTarget] = []
    splits: dict[str, str] = {}
    origins: dict[str, str] = {}
    for dataset in collection["datasets"]:
        rows, run = dataset["rows"], dataset["run_manifest"]
        cases = tuple(representation_case_from_market_case_input_row(row, run)
                      for row in rows)
        target_records = build_neutral_market_revision_targets(rows, run)
        if set(target_records) != {case.revision_id for case in cases}:
            raise RepresentationDataError("neutral targets omit input revisions")
        store, _ = _neutral_canonical_store(dataset, cases)
        prepared = tuple(
            prepare_neutral_representation_case(case, store) for case in cases
        )
        role = str(dataset["representation_split_role"])
        for example in prepared:
            revision_id = example.case.revision_id
            if revision_id in splits:
                raise RepresentationDataError("neutral fit duplicate revision")
            splits[revision_id] = role
            origins[revision_id] = str(dataset["run_sha"])
            examples.append(example)
            targets.append(target_records[revision_id].target)
    if set(splits.values()) != {"train", "validation", "holdout"}:
        raise RepresentationDataError("neutral fit prepared incomplete split roles")
    return {"examples": tuple(examples), "targets": tuple(targets),
            "splits": splits, "origins": origins}


def _neutral_role_metrics(
    model: MarketRepresentationModel,
    examples: Sequence[PreparedRepresentationCase],
    targets: Sequence[SelfSupervisedTarget],
    indices: Sequence[int],
    *, batch_size: int, seed: int, device: Any,
) -> Mapping[str, Any]:
    """Measure one split without gradients.

    ``objective_losses`` are row-weighted means of the per-batch objective
    values.  They are diagnostic objective means, not token-level NLLs.
    """

    import torch

    require_neutral_preprocessed_examples(examples)
    sums: Counter[str] = Counter()
    head_sums: Counter[str] = Counter()
    head_counts: Counter[str] = Counter()
    embeddings: list[np.ndarray] = []
    rows = 0
    model.eval()
    for batch_index, selected in enumerate(_batches(indices, batch_size)):
        selected_examples = tuple(examples[index] for index in selected)
        selected_targets = tuple(targets[index] for index in selected)
        objective_batch, objective_targets = collate_representation_cases(
            selected_examples,
            targets=selected_targets,
            mask_probability=0.15, seed=seed + batch_index,
        )
        inference_batch, inference_targets = collate_representation_cases(
            selected_examples,
            targets=selected_targets,
            mask_probability=0.0, seed=0,
        )
        assert objective_targets is not None and inference_targets is not None
        objective_batch = objective_batch.to(device)
        objective_targets = objective_targets.to(device)
        inference_batch = inference_batch.to(device)
        inference_targets = inference_targets.to(device)
        with torch.no_grad():
            objective_output = model(objective_batch)
            breakdown = neutral_representation_multitask_loss(
                objective_output, objective_batch, objective_targets
            )
            inference_output = model(inference_batch)
            task_metrics = representation_task_metrics(
                inference_output, inference_targets
            )
        embeddings.append(inference_output.embedding.detach().cpu().numpy())
        weight = len(selected)
        rows += weight
        sums["total"] += float(breakdown.total.detach().cpu()) * weight
        for name, value in breakdown.components.items():
            sums[name] += float(value.detach().cpu()) * weight
        for target_name, target_values in (
            ("next_lifecycle", inference_targets.next_lifecycle),
            (
                "scale_direction_alignment",
                inference_targets.scale_direction_alignment,
            ),
        ):
            labelled = int((target_values != -100).sum().detach().cpu())
            if labelled:
                head_counts[target_name] += labelled
                for suffix in ("nll", "accuracy"):
                    key = f"{target_name}_{suffix}"
                    head_sums[key] += task_metrics[key] * labelled
    if rows < 1:
        raise RepresentationDataError("neutral role evaluation is empty")
    disabled = ("next_event", "next_event_time", "displacement", "draw_consumed")
    if any(sums[name] != 0.0 for name in disabled):
        raise RepresentationDataError("neutral evaluation enabled a disabled head")
    matrix = np.concatenate(embeddings, axis=0).astype(np.float64, copy=False)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    spectrum = np.linalg.eigvalsh(centered.T @ centered)
    spectrum = np.clip(spectrum, 0.0, None)
    positive = spectrum[spectrum > np.finfo(np.float64).eps]
    weights = positive / positive.sum() if positive.size else positive
    effective_rank = (
        0.0
        if not weights.size
        else float(np.exp(-np.sum(weights * np.log(weights))))
    )
    active_heads = {}
    for name in ("next_lifecycle", "scale_direction_alignment"):
        count = int(head_counts[name])
        active_heads[name] = {
            "labelled_rows": count,
            "nll": None if not count else head_sums[f"{name}_nll"] / count,
            "accuracy": (
                None if not count else head_sums[f"{name}_accuracy"] / count
            ),
        }
    return {
        "rows": rows,
        "total_loss": sums["total"] / rows,
        "objective_losses": {name: sums[name] / rows for name in (
            "candle_reconstruction", "event_reconstruction",
            "next_lifecycle", "scale_alignment",
        )},
        "active_head_metrics": active_heads,
        "geometry": {
            "embedding_dim": int(matrix.shape[1]),
            "effective_rank": effective_rank,
            "mean_feature_std": float(matrix.std(axis=0).mean()),
            "centroid_norm": float(np.linalg.norm(matrix.mean(axis=0))),
        },
        "label_sources_masked": True,
        "direct_source_preprocessing": (
            neutral_direct_source_preprocessing_identity()
        ),
        "gradient_enabled": False,
    }


def _neutral_exact_mapping(
    value: Any, keys: Sequence[str], label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise RepresentationDataError(f"neutral {label} key contract differs")
    return value


def _neutral_metric_number(
    value: Any, label: str, *, minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise RepresentationDataError(f"neutral {label} must be finite numeric")
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        raise RepresentationDataError(f"neutral {label} is outside its range")
    return number


def _neutral_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise RepresentationDataError(f"neutral {label} must be a positive integer")
    return value


def _validate_neutral_role_metrics(
    value: Any, *, expected_rows: int, expected_dim: int,
) -> None:
    measured = _neutral_exact_mapping(value, (
        "rows", "total_loss", "objective_losses", "active_head_metrics",
        "geometry", "label_sources_masked", "direct_source_preprocessing",
        "gradient_enabled",
    ), "role metrics")
    if (
        _neutral_positive_int(measured["rows"], "role rows") != expected_rows
        or measured["label_sources_masked"] is not True
        or measured["gradient_enabled"] is not False
        or measured["direct_source_preprocessing"]
        != neutral_direct_source_preprocessing_identity()
    ):
        raise RepresentationDataError("neutral role execution contract differs")
    total = _neutral_metric_number(measured["total_loss"], "total loss")
    losses = _neutral_exact_mapping(measured["objective_losses"], (
        "candle_reconstruction", "event_reconstruction", "next_lifecycle",
        "scale_alignment",
    ), "objective losses")
    numeric_losses = {
        name: _neutral_metric_number(losses[name], f"{name} objective loss")
        for name in losses
    }
    weighted = sum(
        numeric_losses[name] * NEUTRAL_REPRESENTATION_LOSS_WEIGHTS[name]
        for name in numeric_losses
    )
    if not math.isclose(total, weighted, rel_tol=1e-6, abs_tol=1e-8):
        raise RepresentationDataError("neutral total/objective loss contract differs")

    heads = _neutral_exact_mapping(measured["active_head_metrics"],
        ("next_lifecycle", "scale_direction_alignment"), "active heads")
    for name, raw_head in heads.items():
        head = _neutral_exact_mapping(
            raw_head, ("labelled_rows", "nll", "accuracy"), f"{name} head")
        labelled = head["labelled_rows"]
        if type(labelled) is not int or not 0 <= labelled <= expected_rows:
            raise RepresentationDataError("neutral labelled-row count is invalid")
        if labelled == 0:
            if head["nll"] is not None or head["accuracy"] is not None:
                raise RepresentationDataError("neutral empty head metrics must be null")
        else:
            _neutral_metric_number(head["nll"], f"{name} NLL")
            _neutral_metric_number(
                head["accuracy"], f"{name} accuracy", maximum=1.0
            )

    geometry = _neutral_exact_mapping(measured["geometry"], (
        "embedding_dim", "effective_rank", "mean_feature_std", "centroid_norm",
    ), "geometry")
    if geometry["embedding_dim"] != expected_dim:
        raise RepresentationDataError("neutral geometry dimension differs")
    _neutral_metric_number(
        geometry["effective_rank"], "effective rank", maximum=float(expected_dim)
    )
    _neutral_metric_number(geometry["mean_feature_std"], "feature std")
    _neutral_metric_number(geometry["centroid_norm"], "centroid norm")


def _validate_neutral_member_metrics_protocol(value: Mapping[str, Any]) -> None:
    """Require the exact outcome-blind B0 member-metrics contract."""

    metrics = _neutral_exact_mapping(value, (
        "model_version", "parameter_count", "embedding_dim", "epochs",
        "epoch_training_loss", "epoch_metrics", "split_counts", "objectives",
        "train", "validation", "holdout", "direct_source_preprocessing",
        "optimized_roles", "validation_used_for_optimization",
        "holdout_used_for_optimization", "holdout_used_for_selection",
        "model_selection_performed", "threshold_search_performed",
        "outcome_fields_used", "model_capability_validated",
        "trading_edge_claimed",
    ), "member metrics")
    if (
        metrics["model_version"] != MODEL_VERSION
        or metrics["direct_source_preprocessing"]
        != neutral_direct_source_preprocessing_identity()
        or metrics["optimized_roles"] != ["train"]
        or any(metrics[name] is not False for name in (
            "validation_used_for_optimization", "holdout_used_for_optimization",
            "holdout_used_for_selection", "model_selection_performed",
            "threshold_search_performed", "outcome_fields_used",
            "model_capability_validated", "trading_edge_claimed",
        ))
    ):
        raise RepresentationDataError("neutral member execution contract differs")
    _neutral_positive_int(metrics["parameter_count"], "parameter count")
    embedding_dim = _neutral_positive_int(metrics["embedding_dim"], "embedding dim")
    epochs = _neutral_positive_int(metrics["epochs"], "epoch count")
    epoch_losses = metrics["epoch_training_loss"]
    epoch_metrics = metrics["epoch_metrics"]
    if (
        not isinstance(epoch_losses, list) or len(epoch_losses) != epochs
        or not isinstance(epoch_metrics, list) or len(epoch_metrics) != epochs
    ):
        raise RepresentationDataError("neutral epoch sequence contract differs")
    for loss in epoch_losses:
        _neutral_metric_number(loss, "epoch training loss")

    split_counts = _neutral_exact_mapping(
        metrics["split_counts"], ("train", "validation", "holdout"),
        "split counts",
    )
    for role in split_counts:
        _neutral_positive_int(split_counts[role], f"{role} split rows")
    objectives = _neutral_exact_mapping(
        metrics["objectives"], ("loss_weights", "active_heads", "disabled_heads"),
        "objectives",
    )
    weights = _neutral_exact_mapping(
        objectives["loss_weights"], tuple(NEUTRAL_REPRESENTATION_LOSS_WEIGHTS),
        "loss weights",
    )
    if (
        objectives["active_heads"] != list(NEUTRAL_SPARSE_ACTIVE_TARGETS)
        or objectives["disabled_heads"] != list(NEUTRAL_SPARSE_DISABLED_TARGETS)
        or any(
            _neutral_metric_number(weights[name], f"{name} loss weight")
            != expected
            for name, expected in NEUTRAL_REPRESENTATION_LOSS_WEIGHTS.items()
        )
    ):
        raise RepresentationDataError("neutral objective configuration differs")

    for expected_epoch, raw_epoch in enumerate(epoch_metrics, start=1):
        epoch = _neutral_exact_mapping(
            raw_epoch, ("epoch", "train", "validation"), "epoch metrics"
        )
        if type(epoch["epoch"]) is not int or epoch["epoch"] != expected_epoch:
            raise RepresentationDataError("neutral epoch ordinal differs")
        for role in ("train", "validation"):
            _validate_neutral_role_metrics(
                epoch[role], expected_rows=split_counts[role],
                expected_dim=embedding_dim,
            )
    if (
        metrics["train"] != epoch_metrics[-1]["train"]
        or metrics["validation"] != epoch_metrics[-1]["validation"]
    ):
        raise RepresentationDataError("neutral final role metrics differ")
    _validate_neutral_role_metrics(
        metrics["holdout"], expected_rows=split_counts["holdout"],
        expected_dim=embedding_dim,
    )


def _train_neutral_member(
    examples: Sequence[PreparedRepresentationCase],
    targets: Sequence[SelfSupervisedTarget],
    splits: Mapping[str, str],
    args: argparse.Namespace,
) -> tuple[MarketRepresentationModel, Mapping[str, Any]]:
    require_torch()
    import torch

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise RepresentationDataError("training hyperparameters must be positive")
    require_neutral_preprocessed_examples(examples)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RepresentationDataError("requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RepresentationDataError("requested MPS device is unavailable")
    split_indices = {role: [index for index, example in enumerate(examples)
                            if splits[example.case.revision_id] == role]
                     for role in ("train", "validation", "holdout")}
    if any(not indices for indices in split_indices.values()):
        raise RepresentationDataError("neutral fit has an empty split")
    model = MarketRepresentationModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    epoch_losses: list[float] = []
    epoch_metrics: list[Mapping[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        order = list(split_indices["train"])
        random.Random(args.seed + epoch).shuffle(order)
        losses: list[float] = []
        for batch_index, selected in enumerate(_batches(order, args.batch_size)):
            selected_examples = tuple(examples[index] for index in selected)
            batch, target_batch = collate_representation_cases(
                selected_examples,
                targets=tuple(targets[index] for index in selected),
                mask_probability=0.15,
                seed=args.seed + epoch * 10_000 + batch_index,
            )
            assert target_batch is not None
            batch = batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            breakdown = neutral_representation_multitask_loss(model(batch), batch,
                                                               target_batch)
            if not bool(torch.isfinite(breakdown.total)):
                raise RepresentationDataError("non-finite neutral training loss")
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(breakdown.total.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train": _neutral_role_metrics(
                model, examples, targets, split_indices["train"],
                batch_size=args.batch_size,
                seed=args.seed + 800_000,
                device=device,
            ),
            "validation": _neutral_role_metrics(
                model, examples, targets, split_indices["validation"],
                batch_size=args.batch_size,
                seed=args.seed + 900_000,
                device=device,
            ),
        })
    holdout_metrics = _neutral_role_metrics(
        model, examples, targets, split_indices["holdout"],
        batch_size=args.batch_size, seed=args.seed + 950_000, device=device,
    )
    metrics = {
        "model_version": MODEL_VERSION,
        "parameter_count": model.parameter_count(),
        "embedding_dim": model.config.embedding_dim,
        "epochs": args.epochs,
        "epoch_training_loss": epoch_losses,
        "epoch_metrics": epoch_metrics,
        "split_counts": {role: len(indices) for role, indices in split_indices.items()},
        "objectives": {
            "loss_weights": dict(NEUTRAL_REPRESENTATION_LOSS_WEIGHTS),
            "active_heads": list(NEUTRAL_SPARSE_ACTIVE_TARGETS),
            "disabled_heads": list(NEUTRAL_SPARSE_DISABLED_TARGETS),
        },
        "train": epoch_metrics[-1]["train"],
        "validation": epoch_metrics[-1]["validation"],
        "holdout": holdout_metrics,
        "direct_source_preprocessing": (
            neutral_direct_source_preprocessing_identity()
        ),
        "optimized_roles": ["train"],
        "validation_used_for_optimization": False,
        "holdout_used_for_optimization": False,
        "holdout_used_for_selection": False,
        "model_selection_performed": False,
        "threshold_search_performed": False,
        "outcome_fields_used": False,
        "model_capability_validated": False,
        "trading_edge_claimed": False,
    }
    _validate_neutral_member_metrics_protocol(metrics)
    return model, metrics


def _neutral_fit_lineage(collection: Mapping[str, Any]) -> Mapping[str, Any]:
    registry = collection["registry"]
    datasets = sorted(collection["datasets"], key=lambda item: item["profile_name"])
    protocol_identity = datasets[0]["run_manifest"]["market_case_input_identity"]
    return {
        "input_runs": [
            {
                "profile_name": item["profile_name"],
                "split_role": item["representation_split_role"],
                "input_manifest_path": str(item["input_path"]),
                "input_manifest_sha256": item["input_sha"],
                "run_manifest_path": str(item["run_path"]),
                "run_manifest_sha256": item["run_sha"],
                "repository_commit": item["run_identity"]["repository"]["commit"],
            }
            for item in datasets
        ],
        "source_identity": {"sha256": collection["source_sha256"]},
        "model_config_identity": {"sha256": collection["model_config_sha256"], "timezone": collection["timezone"]},
        "market_case_protocol": protocol_identity,
        "representation_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "split_protocol": {
            "registry_sha256": collection["registry_sha256"],
            "protocol_version": registry.protocol_version,
            "warmup_calendar_days": registry.warmup_calendar_days,
            "purge_calendar_days": registry.purge_calendar_days,
            "embargo_trading_days": registry.embargo_trading_days,
            "market_episode_split_key": list(registry.episode_split_key),
            "actual_prefix_exposure_verified": True,
            "observed_completed_session_embargo": list(collection["observed_embargo_sessions"]),
        },
    }


def _neutral_export_records(
    model: MarketRepresentationModel,
    examples: Sequence[PreparedRepresentationCase],
    splits: Mapping[str, str],
    origins: Mapping[str, str],
    *,
    member_id: str,
    batch_size: int,
    device: str,
    embeddings: bool,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    require_neutral_preprocessed_examples(examples)
    embedding_rows: list[Mapping[str, Any]] = []
    head_rows: list[Mapping[str, Any]] = []
    run_groups: dict[str, list[PreparedRepresentationCase]] = {}
    for example in examples:
        run_groups.setdefault(origins[example.case.revision_id], []).append(example)
    for run_sha, run_examples in sorted(run_groups.items()):
        if embeddings:
            for kind in NEUTRAL_MARKET_TRANSITION_KINDS:
                selected = [example for example in run_examples
                            if kind in example.case.transition_kinds]
                grains = [(item.case.market_epoch_id, item.case.market_episode_id, kind)
                          for item in selected]
                if len(grains) != len(set(grains)):
                    raise RepresentationDataError("duplicate material occurrence")
                for batch_examples in _batches(selected, batch_size):
                    batch, _ = collate_representation_cases(
                        tuple(batch_examples), mask_probability=0.0, seed=0
                    )
                    records = encode_market_episode_records(
                        model, batch.to(device), tuple(batch_examples),
                        split_roles=splits, material_kind=kind)
                    for record in records:
                        embedding_rows.append({**record, "run_manifest_sha256": run_sha})
        for batch_examples in _batches(run_examples, batch_size):
            batch, _ = collate_representation_cases(
                tuple(batch_examples), mask_probability=0.0, seed=0
            )
            records = encode_market_episode_active_head_records(
                model, batch.to(device), tuple(batch_examples), member_id=member_id)
            for record in records:
                head_rows.append({
                    **record, "run_manifest_sha256": run_sha,
                    "data_split": splits[str(record["revision_id"])],
                })
    return embedding_rows, head_rows


def _write_neutral_artifact(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: str,
    checkpoint_ids: Sequence[str],
    lineage: Mapping[str, Any],
) -> Path:
    if not rows:
        raise RepresentationDataError("neutral export produced no records")
    for row in rows:
        if (
            row.get("outcome_fields_used") is not False
            or "outcome" in row
            or "frozen_outcome" in row
        ):
            raise RepresentationDataError("neutral export contains outcome data")
    destination = _atomic_jsonl(path, rows)
    manifest = {
        "schema": schema,
        "status": "complete",
        "records": len(rows),
        "artifact_path": destination.name,
        "artifact_sha256": _sha256_file(str(destination)),
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_protocol": NEUTRAL_INFERENCE_INPUT_PROTOCOL,
        "direct_source_preprocessing": (
            neutral_direct_source_preprocessing_identity()
        ),
        "selection_contract": NEUTRAL_MATERIAL_SELECTION_CONTRACT,
        "checkpoint_ids": sorted(checkpoint_ids),
        "split_roles": sorted({str(row["data_split"]) for row in rows}),
        "material_kinds": list(NEUTRAL_MARKET_TRANSITION_KINDS),
        "head_schema": (
            {name: OUTCOME_BLIND_HEAD_WIDTHS[name]
             for name in NEUTRAL_SPARSE_ACTIVE_TARGETS}
            if schema == NEUTRAL_HEAD_ARTIFACT_SCHEMA
            else None
        ),
        "lineage": lineage,
        "outcome_fields_used": False,
        "model_capability_validated": False,
    }
    manifest_path = _artifact_manifest_path(destination)
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _neutral_fit_report(
    collection: Mapping[str, Any],
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    prepared = _prepare_neutral_fit_collection(collection)
    examples = prepared["examples"]
    targets = prepared["targets"]
    splits = prepared["splits"]
    origins = prepared["origins"]
    lineage = _neutral_fit_lineage(collection)
    members: list[Mapping[str, Any]] = []
    checkpoint_ids: set[str] = set()
    embedding_rows: list[Mapping[str, Any]] = []
    head_rows: list[Mapping[str, Any]] = []
    for member_index in range(args.ensemble_size):
        member_id = f"member-{member_index:03d}"
        member_args = argparse.Namespace(**vars(args))
        member_args.seed = args.seed + member_index * 100_003
        model, metrics = _train_neutral_member(examples, targets, splits, member_args)
        checkpoint_id = representation_checkpoint_id(model)
        if checkpoint_id in checkpoint_ids:
            raise RepresentationDataError("neutral member checkpoint is duplicated")
        checkpoint_ids.add(checkpoint_id)
        base = Path(args.checkpoint).resolve()
        checkpoint_path = base.with_name(
            f"{base.stem}.{member_id}{base.suffix or '.pt'}")
        save_neutral_representation_checkpoint(
            checkpoint_path,
            model,
            metadata={
                "member_id": member_id,
                "seed": member_args.seed,
                "split_counts": metrics["split_counts"],
                "lineage": lineage,
                "outcome_fields_used": False,
                "model_capability_validated": False,
            },
        )
        member_embeddings, member_heads = _neutral_export_records(
            model, examples, splits, origins, member_id=member_id,
            batch_size=args.batch_size, device=args.device,
            embeddings=member_index == 0)
        embedding_rows.extend(member_embeddings)
        head_rows.extend(member_heads)
        members.append({
            "member_id": member_id, "seed": member_args.seed,
            "checkpoint_id": checkpoint_id, "checkpoint_path": str(checkpoint_path),
            "metrics": metrics,
        })
    embedding_manifest = _write_neutral_artifact(
        args.embedding_output, embedding_rows, schema=NEUTRAL_EMBEDDING_ARTIFACT_SCHEMA,
        checkpoint_ids=(members[0]["checkpoint_id"],), lineage=lineage)
    head_manifest = _write_neutral_artifact(
        args.head_output, head_rows, schema=NEUTRAL_HEAD_ARTIFACT_SCHEMA,
        checkpoint_ids=tuple(item["checkpoint_id"] for item in members),
        lineage=lineage)
    return {
        "schema_version": 2,
        "mode": "neutral_market_representation_fit",
        "pipeline_scope": collection["scope"],
        "rows": len(examples),
        "split_rows": dict(Counter(splits.values())),
        "ensemble_members": members,
        "artifacts": {
            "embeddings": {
                "path": str(Path(args.embedding_output).resolve()),
                "manifest": str(embedding_manifest), "records": len(embedding_rows)},
            "active_heads": {
                "path": str(Path(args.head_output).resolve()),
                "manifest": str(head_manifest), "records": len(head_rows)},
        },
        "lineage": lineage,
        "direct_source_preprocessing": (
            neutral_direct_source_preprocessing_identity()
        ),
        "training_performed": True,
        "validation_used_for_optimization": False,
        "holdout_used_for_optimization": False,
        "holdout_used_for_selection": False,
        "model_selection_performed": False,
        "threshold_search_performed": False,
        "outcome_fields_used": False,
        "model_capability_validated": False,
        "retrieval_quality_validated": False,
        "ood_capability_validated": False,
        "trading_edge_claimed": False,
        "action_value_claimed": False,
    }


def _neutral_mode_requested(args: argparse.Namespace) -> bool:
    return any((
        args.neutral_dataset_audit_only, args.neutral_single_batch_smoke,
        args.neutral_fit,
        args.market_case_input_manifest, args.market_case_run_manifest,
        args.market_embedding_kind,
    ))


def _validate_neutral_cli(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    modes = int(args.neutral_dataset_audit_only) + int(
        args.neutral_single_batch_smoke
    ) + int(args.neutral_fit)
    if modes != 1:
        raise RepresentationDataError(
            "neutral datasets require exactly one audit, single-batch smoke or fit mode"
        )
    if (
        not args.market_case_input_manifest
        or not args.market_case_run_manifest
        or len(args.market_case_input_manifest)
        != len(args.market_case_run_manifest)
    ):
        raise RepresentationDataError(
            "neutral mode requires paired --market-case-input-manifest and "
            "--market-case-run-manifest"
        )
    if not args.neutral_fit and len(args.market_case_input_manifest) != 1:
        raise RepresentationDataError(
            "neutral audit/smoke accepts exactly one input/run manifest pair"
        )
    legacy = any((
        args.synthetic_smoke, args.case_input_shard, args.case_input_manifest,
        args.case_input_manifest_sha, args.case_run_manifest_sha,
        args.case_library_manifest, args.case_library_manifest_sha,
        args.self_supervised_targets,
    ))
    explicit = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    fit_options = {
        "--canonical-view", "--canonical-view-sha", "--canonical-lineage-manifest",
        "--canonical-lineage-manifest-sha", "--tick-size", "--availability-time",
        "--epochs", "--ensemble-size", "--batch-size", "--learning-rate", "--seed",
        "--device", "--checkpoint", "--embedding-output", "--head-output",
        "--embedding-stage",
    }
    if legacy:
        raise RepresentationDataError("neutral and legacy inputs are mutually exclusive")
    if (args.neutral_single_batch_smoke or args.neutral_fit) and args.market_embedding_kind:
        raise RepresentationDataError(
            "neutral embedding-kind selection is audit-only"
        )
    if not args.neutral_fit and (forbidden := sorted(explicit & fit_options)):
        raise RepresentationDataError(
            "neutral modes forbid training/export options: " + ", ".join(forbidden)
        )
    if args.neutral_fit:
        forbidden_fit = sorted(
            explicit
            & {
                "--canonical-view", "--canonical-view-sha",
                "--canonical-lineage-manifest", "--canonical-lineage-manifest-sha",
                "--tick-size", "--availability-time", "--embedding-stage",
            }
        )
        if forbidden_fit:
            raise RepresentationDataError(
                "neutral fit forbids legacy feature/export options: "
                + ", ".join(forbidden_fit)
            )
        if "--ensemble-size" not in explicit:
            args.ensemble_size = 3
        if args.ensemble_size < 3:
            raise RepresentationDataError(
                "neutral fit requires at least three independent ensemble members"
            )
        required_outputs = {
            "--checkpoint": args.checkpoint,
            "--embedding-output": args.embedding_output,
            "--head-output": args.head_output,
        }
        missing_outputs = sorted(
            name for name, value in required_outputs.items() if not value
        )
        if missing_outputs:
            raise RepresentationDataError(
                "neutral fit requires: " + ", ".join(missing_outputs)
            )


def _validate_case_input_stream_manifest(
    shard_paths: Sequence[str | Path],
    manifest_path: str | None,
    expected_manifest_sha256: str | None,
    expected_run_sha256: str | None,
) -> tuple[str, int, Path]:
    """Hash-bind real training to the committed outcome-free case stream."""

    if not manifest_path:
        raise RepresentationDataError(
            "real training requires --case-input-manifest"
        )
    manifest_sha = _normalized_sha256(
        expected_manifest_sha256, name="--case-input-manifest-sha"
    )
    run_sha = _normalized_sha256(
        expected_run_sha256, name="--case-run-manifest-sha"
    )
    manifest = Path(manifest_path).resolve()
    if manifest.is_symlink() or not manifest.is_file():
        raise RepresentationDataError("case input manifest is missing")
    if _sha256_file(str(manifest)) != manifest_sha:
        raise RepresentationDataError("case input manifest content hash mismatch")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError("case input manifest is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RepresentationDataError("case input manifest must be an object")
    from smc_trader.causal_cases import CAUSAL_CASE_INPUT_FIELD_TYPES

    if (
        payload.get("format_version") != 1
        or payload.get("status") != "complete"
        or payload.get("stream") != "causal_case_input_shards"
        or payload.get("field_types") != dict(CAUSAL_CASE_INPUT_FIELD_TYPES)
    ):
        raise RepresentationDataError(
            "case input manifest identity, status or exact schema is invalid"
        )
    root = manifest.parent
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings.get("run_manifest"):
        raise RepresentationDataError("case input manifest lacks run binding")
    run_manifest = _bound_manifest_path(
        root, bindings["run_manifest"], name="case run manifest"
    )
    if _sha256_file(str(run_manifest)) != run_sha:
        raise RepresentationDataError("case run manifest content hash mismatch")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise RepresentationDataError("case input manifest contains no shards")
    committed: list[Path] = []
    total_rows = 0
    for expected_index, shard in enumerate(raw_shards):
        if not isinstance(shard, Mapping) or int(shard.get("index", -1)) != expected_index:
            raise RepresentationDataError("case input shard order is not contiguous")
        path = _bound_manifest_path(root, shard.get("path"), name="case input shard")
        if path.suffix.lower() != ".parquet":
            raise RepresentationDataError(
                "real training accepts only committed causal-case Parquet shards"
            )
        expected_hash = _normalized_sha256(
            str(shard.get("sha256", "")), name="case input shard sha256"
        )
        if _sha256_file(str(path)) != expected_hash:
            raise RepresentationDataError("case input shard content hash mismatch")
        rows = int(shard.get("rows", -1))
        if rows < 0:
            raise RepresentationDataError("case input shard row count is invalid")
        total_rows += rows
        committed.append(path)
    supplied = [Path(path).resolve() for path in shard_paths]
    if supplied != committed:
        raise RepresentationDataError(
            "--case-input-shard paths/order must exactly match the committed manifest"
        )
    if total_rows != int(payload.get("rows", -1)):
        raise RepresentationDataError("case input manifest row counts are inconsistent")
    return manifest_sha, total_rows, run_manifest


def _validate_case_library_manifest(
    path: str | None,
    expected_sha256: str | None,
    *,
    input_manifest_path: str,
    input_manifest_sha256: str,
    input_rows: int,
    run_manifest_path: Path,
    run_manifest_sha256: str,
) -> str:
    """Require the post-finalization manifest without opening outcome rows."""

    if not path:
        raise RepresentationDataError(
            "real training requires --case-library-manifest"
        )
    expected_sha = _normalized_sha256(
        expected_sha256, name="--case-library-manifest-sha"
    )
    manifest = Path(path).resolve()
    if manifest.is_symlink() or not manifest.is_file():
        raise RepresentationDataError("causal-case library manifest is missing")
    if _sha256_file(str(manifest)) != expected_sha:
        raise RepresentationDataError("causal-case library manifest hash mismatch")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError(
            "causal-case library manifest is not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RepresentationDataError("causal-case library manifest must be an object")
    from smc_trader.causal_cases import (
        CAUSAL_CASE_OUTCOME_FIELD_TYPES,
        CAUSAL_CASE_PROTOCOL,
        CAUSAL_CASE_RECORDER_SCHEMA_VERSION,
        expected_causal_case_run_identity,
    )

    if (
        payload.get("format_version") != 1
        or payload.get("artifact") != "entry_episode_causal_case_library"
        or payload.get("status") != "complete"
        or payload.get("grain") != "entry_episode"
        or payload.get("recorder_schema_version")
        != CAUSAL_CASE_RECORDER_SCHEMA_VERSION
        or payload.get("protocol") != dict(CAUSAL_CASE_PROTOCOL)
    ):
        raise RepresentationDataError(
            "causal-case library finalization identity/protocol is invalid"
        )
    root = manifest.parent
    input_binding = payload.get("input_stream")
    outcome_binding = payload.get("future_outcome_stream")
    run_binding = payload.get("bindings")
    leakage = payload.get("leakage_contract")
    if (
        not isinstance(input_binding, Mapping)
        or not isinstance(outcome_binding, Mapping)
        or not isinstance(run_binding, Mapping)
    ):
        raise RepresentationDataError("causal-case library bindings are invalid")
    bound_input = _bound_manifest_path(
        root, input_binding.get("manifest"), name="library input manifest"
    )
    bound_run = _bound_manifest_path(
        root, run_binding.get("run_manifest"), name="library run manifest"
    )
    bound_outcome = _bound_manifest_path(
        root,
        outcome_binding.get("manifest"),
        name="library future outcome stream manifest",
    )
    if bound_run != run_manifest_path.resolve():
        raise RepresentationDataError(
            "causal-case run manifest path mismatch"
        )
    try:
        run_bytes = bound_run.read_bytes()
        run_payload = json.loads(run_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError(
            "causal-case run manifest is not valid JSON"
        ) from exc
    if hashlib.sha256(run_bytes).hexdigest() != run_manifest_sha256:
        raise RepresentationDataError(
            "causal-case run manifest content hash mismatch"
        )
    if not isinstance(run_payload, Mapping):
        raise RepresentationDataError(
            "causal-case run manifest must be an object"
        )
    canonical_run_bytes = json.dumps(
        dict(run_payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if run_bytes != canonical_run_bytes:
        raise RepresentationDataError(
            "causal-case run manifest is not canonical JSON"
        )
    run_output = run_payload.get("output")
    stream_families = (
        None
        if not isinstance(run_output, Mapping)
        else run_output.get("stream_families")
    )
    required_case_streams = {
        "causal_case_input_shards",
        "causal_case_outcome_shards",
    }
    if (
        run_payload.get("schema_version") != 1
        or run_payload.get("runner") != "continuous_replay"
        or run_payload.get("causal_case_identity")
        != expected_causal_case_run_identity()
        or not isinstance(run_output, Mapping)
        or run_output.get("causal_case_library") is not True
        or not isinstance(stream_families, list)
        or any(not isinstance(value, str) for value in stream_families)
        or len(stream_families) != len(set(stream_families))
        or not required_case_streams.issubset(set(stream_families))
    ):
        raise RepresentationDataError(
            "causal-case run manifest identity is invalid"
        )
    expected_outcome_sha = _normalized_sha256(
        str(outcome_binding.get("manifest_sha256", "")),
        name="library future outcome stream manifest_sha256",
    )
    if _sha256_file(str(bound_outcome)) != expected_outcome_sha:
        raise RepresentationDataError(
            "causal-case future outcome stream manifest hash mismatch"
        )
    try:
        outcome_manifest = json.loads(bound_outcome.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError(
            "causal-case future outcome stream manifest is not valid JSON"
        ) from exc
    if not isinstance(outcome_manifest, Mapping):
        raise RepresentationDataError(
            "causal-case future outcome stream manifest must be an object"
        )
    outcome_rows = int(outcome_binding.get("rows", -1))
    outcome_shards = outcome_manifest.get("shards")
    shard_rows = 0
    shard_metadata_valid = isinstance(outcome_shards, list)
    if shard_metadata_valid:
        for expected_index, shard in enumerate(outcome_shards):
            if not isinstance(shard, Mapping):
                shard_metadata_valid = False
                break
            relative = Path(str(shard.get("path", "")))
            rows = int(shard.get("rows", -1))
            digest = str(shard.get("sha256", "")).lower()
            if (
                int(shard.get("index", -1)) != expected_index
                or relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or rows < 0
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                shard_metadata_valid = False
                break
            shard_rows += rows
    if (
        outcome_manifest.get("format_version") != 1
        or outcome_manifest.get("status") != "complete"
        or outcome_manifest.get("stream") != "causal_case_outcome_shards"
        or outcome_manifest.get("field_types")
        != dict(CAUSAL_CASE_OUTCOME_FIELD_TYPES)
        or int(outcome_manifest.get("rows", -1)) != outcome_rows
        or outcome_rows < 0
        or not shard_metadata_valid
        or shard_rows != outcome_rows
        or not isinstance(outcome_manifest.get("bindings"), Mapping)
        or _bound_manifest_path(
            root,
            outcome_manifest["bindings"].get("run_manifest"),
            name="future outcome stream run manifest",
        )
        != run_manifest_path.resolve()
    ):
        raise RepresentationDataError(
            "causal-case future outcome stream finalization metadata is invalid"
        )
    if (
        bound_input != Path(input_manifest_path).resolve()
        or str(input_binding.get("manifest_sha256", "")).lower()
        != input_manifest_sha256
        or int(input_binding.get("rows", -1)) != input_rows
        or bound_run != run_manifest_path.resolve()
        or str(run_binding.get("run_manifest_sha256", "")).lower()
        != run_manifest_sha256
        or not isinstance(leakage, Mapping)
        or leakage.get("outcome_fields_in_input_schema") is not False
        or leakage.get("embedding_source") != "input_stream_only"
        or leakage.get("episode_split_disjoint_required") is not True
        or leakage.get("normalization_prefix_only") is not True
    ):
        raise RepresentationDataError(
            "causal-case library does not bind the exact input/run leakage contract"
        )
    return expected_sha


def _canonical_view_key(raw_key: str, *, option: str) -> tuple[str, str, str]:
    parts = raw_key.rsplit(":", 2)
    if len(parts) != 3 or not parts[0]:
        raise RepresentationDataError(
            f"{option} must bind MARKET_EPOCH_ID:TIMEFRAME:SOURCE_SHA256 explicitly"
        )
    market_epoch_id, timeframe, source_id = parts
    timeframe = timeframe.lower()
    source_id = source_id.lower()
    if timeframe not in TIMEFRAMES:
        raise RepresentationDataError(f"unknown canonical timeframe {timeframe!r}")
    if len(source_id) != 64 or any(
        character not in "0123456789abcdef" for character in source_id
    ):
        raise RepresentationDataError(
            f"{option} must bind MARKET_EPOCH_ID:TIMEFRAME:SOURCE_SHA256 explicitly"
        )
    return market_epoch_id, timeframe, source_id


def _load_lineage_manifest(
    path: str | None,
    expected_sha256: str | None,
) -> Mapping[str, Any]:
    if not path or not expected_sha256:
        raise RepresentationDataError(
            "real training requires --canonical-lineage-manifest and its "
            "pre-registered --canonical-lineage-manifest-sha"
        )
    normalized_hash = expected_sha256.strip().lower()
    if len(normalized_hash) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_hash
    ):
        raise RepresentationDataError("canonical lineage manifest SHA-256 is invalid")
    if _sha256_file(path) != normalized_hash:
        raise RepresentationDataError("canonical lineage manifest content hash mismatch")
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError("canonical lineage manifest is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RepresentationDataError("canonical lineage manifest must be an object")
    if payload.get("schema") != LINEAGE_SCHEMA:
        raise RepresentationDataError("canonical lineage manifest schema mismatch")
    if payload.get("aggregation_protocol") != AGGREGATION_PROTOCOL:
        raise RepresentationDataError("canonical aggregation protocol mismatch")
    if not isinstance(payload.get("views"), list):
        raise RepresentationDataError("canonical lineage manifest views are invalid")
    return payload


def _build_store(
    cases: Sequence[RepresentationCase],
    canonical_args: Sequence[str],
    canonical_hash_args: Sequence[str],
    tick_args: Sequence[str],
    availability_args: Sequence[str],
    lineage_manifest_path: str | None,
    lineage_manifest_sha256: str | None,
) -> CanonicalOHLCVStore:
    lineage_manifest = _load_lineage_manifest(
        lineage_manifest_path, lineage_manifest_sha256
    )
    view_paths: dict[tuple[str, str, str], str] = {}
    for value in canonical_args:
        raw_key, path = _identity_assignment(value, name="--canonical-view")
        key = _canonical_view_key(raw_key, option="--canonical-view")
        if key in view_paths:
            raise RepresentationDataError(f"duplicate canonical view binding {raw_key!r}")
        view_paths[key] = path
    expected_hashes: dict[tuple[str, str, str], str] = {}
    for value in canonical_hash_args:
        raw_key, expected_hash = _identity_assignment(
            value, name="--canonical-view-sha"
        )
        key = _canonical_view_key(raw_key, option="--canonical-view-sha")
        normalized_hash = expected_hash.lower()
        if len(normalized_hash) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_hash
        ):
            raise RepresentationDataError(
                "--canonical-view-sha values must be SHA-256 hex digests"
            )
        if key in expected_hashes:
            raise RepresentationDataError(
                f"duplicate canonical view hash binding {raw_key!r}"
            )
        expected_hashes[key] = normalized_hash
    if set(view_paths) != set(expected_hashes):
        raise RepresentationDataError(
            "--canonical-view and --canonical-view-sha must bind identical "
            "source-qualified five-timeframe keys"
        )
    resolved_view_owners: dict[Path, tuple[str, str, str]] = {}
    for key, raw_path in view_paths.items():
        resolved_path = Path(raw_path).resolve()
        prior = resolved_view_owners.setdefault(resolved_path, key)
        if prior != key:
            raise RepresentationDataError(
                "each epoch/timeframe/source binding must use a distinct view file"
            )
    views: dict[tuple[str, str, str], pd.DataFrame] = {}
    for key, path in view_paths.items():
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hashes[key]:
            raise RepresentationDataError(
                "canonical view content hash mismatch for " + ":".join(key)
            )
        views[key] = _load_frame(path)
    ticks = {
        timeframe: float(raw_value)
        for timeframe, raw_value in (
            _assignment(value, name="--tick-size") for value in tick_args
        )
    }
    if set(ticks) != set(TIMEFRAMES):
        raise RepresentationDataError(
            f"--tick-size must bind exactly {TIMEFRAMES}"
        )
    availability_by_timeframe = dict(
        _assignment(value, name="--availability-time")
        for value in availability_args
    )
    if set(availability_by_timeframe) != set(TIMEFRAMES):
        raise RepresentationDataError(
            f"--availability-time must bind exactly {TIMEFRAMES}; raw bar-start "
            "indices are never trusted implicitly"
        )

    case_lineage: dict[tuple[str, str], RepresentationCase] = {}
    verified_parent_sources: set[tuple[Path, str]] = set()
    for case in cases:
        if not case.canonical_source_path or not case.symbol or case.instrument_id < 0:
            raise RepresentationDataError(
                "case adapter must provide source_path, symbol and instrument_id "
                "for canonical lineage validation"
            )
        source_ids = {
            case.prefixes[timeframe].canonical_source_id for timeframe in TIMEFRAMES
        }
        if len(source_ids) != 1:
            raise RepresentationDataError(
                "case five-timeframe prefixes disagree on canonical source identity"
            )
        source_id = next(iter(source_ids))
        resolved_source_path = Path(case.canonical_source_path).resolve()
        source_verification_key = (resolved_source_path, source_id)
        if source_verification_key not in verified_parent_sources:
            if _sha256_file(case.canonical_source_path) != source_id:
                raise RepresentationDataError(
                    "case canonical source_path content does not match source_sha256"
                )
            verified_parent_sources.add(source_verification_key)
        key = (case.market_epoch_id, source_id)
        prior = case_lineage.setdefault(key, case)
        if (
            prior.symbol != case.symbol
            or prior.instrument_id != case.instrument_id
            or Path(prior.canonical_source_path).resolve()
            != Path(case.canonical_source_path).resolve()
        ):
            raise RepresentationDataError(
                "case rows disagree on source/symbol/instrument lineage"
            )

    manifest_entries: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for raw_entry in lineage_manifest["views"]:
        if not isinstance(raw_entry, Mapping):
            raise RepresentationDataError("canonical lineage view entry is invalid")
        key = (
            str(raw_entry.get("market_epoch_id", "")),
            str(raw_entry.get("parent_source_sha256", "")).lower(),
            str(raw_entry.get("timeframe", "")).lower(),
        )
        if key in manifest_entries:
            raise RepresentationDataError("canonical lineage manifest has duplicate view")
        manifest_entries[key] = raw_entry

    expected_manifest_keys = {
        (epoch_id, source_id, timeframe)
        for epoch_id, source_id in case_lineage
        for timeframe in TIMEFRAMES
    }
    if set(manifest_entries) != expected_manifest_keys:
        raise RepresentationDataError(
            "canonical lineage manifest does not exactly cover case epoch/source/TF keys"
        )
    for (epoch_id, source_id, timeframe), entry in manifest_entries.items():
        case = case_lineage[(epoch_id, source_id)]
        path = view_paths.get((epoch_id, timeframe, source_id))
        expected_view_hash = expected_hashes.get((epoch_id, timeframe, source_id))
        if path is None or expected_view_hash is None:
            raise RepresentationDataError("canonical lineage view binding is missing")
        manifest_view_hash = str(entry.get("view_sha256", "")).lower()
        if (
            entry.get("aggregation_protocol") != AGGREGATION_PROTOCOL
            or str(entry.get("symbol", "")) != case.symbol
            or int(entry.get("instrument_id", -1)) != case.instrument_id
            or Path(str(entry.get("parent_source_path", ""))).resolve()
            != Path(case.canonical_source_path).resolve()
            or Path(str(entry.get("view_path", ""))).resolve() != Path(path).resolve()
            or manifest_view_hash != expected_view_hash
            or str(entry.get("availability_binding", ""))
            != availability_by_timeframe[timeframe]
        ):
            raise RepresentationDataError(
                f"canonical lineage mismatch for {epoch_id}:{source_id}:{timeframe}"
            )

    frames: dict[CanonicalSourceKey, pd.DataFrame] = {}
    tick_bindings: dict[CanonicalSourceKey, float] = {}
    availability_bindings: dict[CanonicalSourceKey, str] = {}
    for case in cases:
        for timeframe in TIMEFRAMES:
            prefix = case.prefixes[timeframe]
            frame = views.get(
                (case.market_epoch_id, timeframe, prefix.canonical_source_id)
            )
            if frame is None:
                raise RepresentationDataError(
                    "missing --canonical-view binding for "
                    f"{timeframe}:{prefix.canonical_source_id}"
                )
            key = CanonicalSourceKey(
                case.market_epoch_id, timeframe, prefix.canonical_source_id
            )
            frames[key] = frame
            tick_bindings[key] = ticks[timeframe]
            availability_bindings[key] = availability_by_timeframe[timeframe]
    return CanonicalOHLCVStore(
        frames,
        tick_sizes=tick_bindings,
        availability_bindings=availability_bindings,
    )


def _target_map(path: str | Path) -> dict[str, ObservableTargetRecord]:
    output: dict[str, ObservableTargetRecord] = {}
    required_metadata = {
        "revision_id",
        "market_epoch_id",
        "entry_episode_id",
        "input_asof",
        "label_max_observed_at",
        "next_revision_id",
        "label_source",
    }
    expected_fields = required_metadata | {
        "next_event_type",
        "next_lifecycle",
        "next_event_time_bucket",
        "displacement_state",
        "draw_consumed",
        "scale_direction_alignment",
    }
    for row in _read_records((path,)):
        if set(row) != expected_fields:
            raise RepresentationDataError(
                "external target cache must contain the exact "
                "ObservableTargetRecord schema"
            )
        revision_id = str(row.get("revision_id", "")).strip()
        if not revision_id or revision_id in output:
            raise RepresentationDataError(
                "self-supervised target rows require unique revision_id"
            )
        record = ObservableTargetRecord(
            revision_id=revision_id,
            market_epoch_id=str(row["market_epoch_id"]),
            entry_episode_id=str(row["entry_episode_id"]),
            input_asof=pd.Timestamp(row["input_asof"]),
            label_max_observed_at=pd.Timestamp(row["label_max_observed_at"]),
            next_revision_id=(
                None
                if row["next_revision_id"] is None
                else str(row["next_revision_id"])
            ),
            label_source=str(row["label_source"]),
            target=SelfSupervisedTarget.from_mapping(row),
        )
        if record.label_source != (
            "same_episode_observable_revisions_with_complete_transition_coverage_v2"
        ):
            raise RepresentationDataError("external target label_source is unsupported")
        output[revision_id] = record
    return output


def _validate_external_target_cache(
    external: Mapping[str, ObservableTargetRecord],
    internally_built: Mapping[str, ObservableTargetRecord],
) -> None:
    """External rows are only a cache; they must equal the causal builder."""

    if set(external) != set(internally_built):
        raise RepresentationDataError(
            "external target cache does not cover internal revisions exactly"
        )
    for revision_id, expected in internally_built.items():
        observed = external[revision_id]
        if observed != expected:
            raise RepresentationDataError(
                f"external target cache disagrees with causal builder: {revision_id}"
            )


def _synthetic_examples(seed: int) -> tuple[
    tuple[PreparedRepresentationCase, ...], tuple[SelfSupervisedTarget, ...]
]:
    rng = np.random.default_rng(seed)
    examples: list[PreparedRepresentationCase] = []
    targets: list[SelfSupervisedTarget] = []
    asof = pd.Timestamp("2022-01-03 16:00", tz="UTC")
    for index in range(12):
        epoch = f"epoch-{index // 6}"
        context = f"context-{index // 2}"
        episode = f"episode-{index}"
        prefixes = {
            timeframe: PrefixIndexRange(
                market_epoch_id=epoch,
                timeframe=timeframe,
                canonical_source_id="synthetic",
                row_start=0,
                row_end_exclusive=8,
            )
            for timeframe in TIMEFRAMES
        }
        # RepresentationCase requires contemporaneous event semantics; the
        # prepared smoke data below uses a deterministic synthetic token.
        from smc_trader.market_representation import EventGraphObservation

        case = RepresentationCase(
            case_id=f"case-{index}",
            revision_id=f"revision-{index}",
            market_epoch_id=epoch,
            context_thesis_id=context,
            entry_episode_id=episode,
            asof=asof,
            direction=1 if index % 2 == 0 else -1,
            regime=("continuation", "sweep_failure", "balance", "unknown")[index % 4],
            prefixes=prefixes,
            events=(
                EventGraphObservation(
                    event_id=f"event-{index}",
                    event_type=("displacement", "sweep", "range")[index % 3],
                    lifecycle="active",
                    observed_at=asof,
                    active_since=asof,
                    duration_seconds=0.0,
                    relation_types=("connected_to",),
                    direction=1 if index % 2 == 0 else -1,
                    scale="1m",
                    market_epoch_id=epoch,
                ),
            ),
        )
        timeframe_features = {
            timeframe: rng.normal(
                0.0,
                0.5,
                size=(8 + timeframe_index * 2, len(CAUSAL_CANDLE_FEATURES)),
            ).astype(np.float32)
            for timeframe_index, timeframe in enumerate(TIMEFRAMES)
        }
        examples.append(
            PreparedRepresentationCase(
                case=case,
                timeframe_features=timeframe_features,
                feature_max_at=asof,
                event_type_ids=np.asarray([2 + index % 3], dtype=np.int64),
                lifecycle_ids=np.asarray([2], dtype=np.int64),
                relation_ids=np.asarray([2], dtype=np.int64),
                scale_ids=np.asarray([2 + index % 5], dtype=np.int64),
                event_numeric=np.asarray([[0.0, 0.0, 1.0, case.direction]], dtype=np.float32),
            )
        )
        targets.append(
            SelfSupervisedTarget(
                next_event_type=2 + ((index + 1) % 3),
                next_lifecycle=2 + index % 2,
                next_event_time_bucket=index % 4,
                displacement_state=index % 2,
                draw_consumed=(index // 2) % 2,
                scale_direction_alignment=index % 3,
            )
        )
    return tuple(examples), tuple(targets)


def _batches(indices: Sequence[int], batch_size: int) -> Sequence[Sequence[int]]:
    return tuple(indices[start : start + batch_size] for start in range(0, len(indices), batch_size))


def _train(
    examples: Sequence[PreparedRepresentationCase],
    targets: Sequence[SelfSupervisedTarget],
    splits: Mapping[str, str],
    args: argparse.Namespace,
) -> tuple[MarketRepresentationModel, dict[str, Any]]:
    require_torch()
    import torch

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise RepresentationDataError("epochs, batch-size and learning-rate must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RepresentationDataError("requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RepresentationDataError("requested MPS device is unavailable")

    split_indices = {
        role: [
            index
            for index, example in enumerate(examples)
            if splits[example.case.revision_id] == role
        ]
        for role in ("train", "validation", "test")
    }
    if not split_indices["train"] or not split_indices["validation"]:
        raise RepresentationDataError(
            "training requires non-empty leakage-safe train and validation splits"
        )
    model = MarketRepresentationModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    epoch_losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        order = list(split_indices["train"])
        random.Random(args.seed + epoch).shuffle(order)
        losses: list[float] = []
        for batch_index, selected in enumerate(_batches(order, args.batch_size)):
            batch, target_batch = collate_representation_cases(
                tuple(examples[index] for index in selected),
                targets=tuple(targets[index] for index in selected),
                mask_probability=0.15,
                seed=args.seed + epoch * 10_000 + batch_index,
            )
            assert target_batch is not None
            batch = batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            breakdown = representation_multitask_loss(model(batch), batch, target_batch)
            if not bool(torch.isfinite(breakdown.total)):
                raise RepresentationDataError("non-finite representation training loss")
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(breakdown.total.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))

    model.eval()
    selected = split_indices["validation"]
    masked_validation_examples = tuple(
        mask_direct_label_source_tokens(examples[index]) for index in selected
    )
    validation_batch, validation_targets = collate_representation_cases(
        masked_validation_examples,
        targets=tuple(targets[index] for index in selected),
        mask_probability=0.15,
        seed=args.seed + 999_999,
    )
    assert validation_targets is not None
    validation_batch = validation_batch.to(device)
    validation_targets = validation_targets.to(device)
    with torch.no_grad():
        validation_output = model(validation_batch)
        validation_loss = representation_multitask_loss(
            validation_output, validation_batch, validation_targets
        )
        task_metrics = representation_task_metrics(
            validation_output, validation_targets
        )
        reconstruction_baselines = zero_reconstruction_baselines(validation_batch)
    train_target_rows = tuple(targets[index] for index in split_indices["train"])
    validation_target_rows = tuple(targets[index] for index in selected)
    baselines = majority_class_baselines(train_target_rows, validation_target_rows)
    comparison = compare_validation_to_baseline(
        task_metrics,
        baselines,
        shortcut_sensitive_tasks_masked=True,
    )
    validation_component_values = {
        name: float(value.detach().cpu())
        for name, value in validation_loss.components.items()
    }
    reconstruction_comparison = compare_reconstruction_to_baselines(
        validation_component_values,
        reconstruction_baselines,
    )
    evaluation_samples: dict[str, tuple[EmbeddingEvaluationSample, ...]] = {}
    for role in ("train", "validation"):
        role_indices = split_indices[role]
        masked_role_examples = tuple(
            mask_direct_label_source_tokens(examples[index])
            for index in role_indices
        )
        role_batch, _ = collate_representation_cases(
            masked_role_examples,
            mask_probability=0.0,
            seed=args.seed,
        )
        role_batch = role_batch.to(device)
        with torch.no_grad():
            role_embeddings = model.encode(role_batch).detach().cpu().numpy()
        evaluation_samples[role] = tuple(
            EmbeddingEvaluationSample(
                revision_id=examples[source_index].case.revision_id,
                entry_episode_id=examples[source_index].case.entry_episode_id,
                asof=examples[source_index].case.asof,
                direction=examples[source_index].case.direction,
                regime=examples[source_index].case.regime,
                mechanism_label=examples[source_index].case.mechanism_label,
                embedding=tuple(float(value) for value in role_embeddings[position]),
                label_sources_masked=True,
            )
            for position, source_index in enumerate(role_indices)
        )
    embedding_evaluation = evaluate_outcome_blind_embedding_space(
        evaluation_samples["train"],
        evaluation_samples["validation"],
        k=min(5, max(1, len(evaluation_samples["train"]) - 1)),
    )
    metrics: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "parameter_count": model.parameter_count(),
        "embedding_dim": model.config.embedding_dim,
        "epochs": args.epochs,
        "epoch_training_loss": epoch_losses,
        "validation_total_loss": float(validation_loss.total.detach().cpu()),
        "validation_components": validation_component_values,
        "validation_task_metrics": task_metrics,
        "majority_baselines": baselines,
        "zero_reconstruction_baselines": reconstruction_baselines,
        "baseline_comparison": comparison,
        "reconstruction_baseline_comparison": reconstruction_comparison,
        "embedding_evaluation": embedding_evaluation,
        "completion_criteria_met": bool(
            comparison["pre_registered_criteria_met"]
            and reconstruction_comparison["pre_registered_criteria_met"]
            and embedding_evaluation["criteria_met"]
        ),
        "split_counts": {role: len(indices) for role, indices in split_indices.items()},
        "outcome_fields_used": False,
        "trading_edge_claimed": False,
    }
    return model, metrics


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            _strict_jsonable(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _atomic_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    serialized = "".join(
        json.dumps(
            _strict_jsonable(row),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(destination)
    return destination


def _artifact_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def _write_decision_artifact(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: str,
    decision_stage: str,
    checkpoint_ids: Sequence[str],
    case_input_manifest_sha256: str | None,
    case_library_manifest_sha256: str | None,
) -> Path:
    if not rows:
        raise RepresentationDataError("decision-time export produced no records")
    destination = _atomic_jsonl(path, rows)
    identity_rows = [
        {
            "case_id": row.get("case_id"),
            "revision_id": row.get("revision_id"),
            "entry_episode_id": row.get("entry_episode_id"),
            "decision_at": row.get("embedding_asof", row.get("decision_at")),
            "feature_max_at": row.get("feature_max_at"),
            "checkpoint_id": row.get(
                "embedding_checkpoint_id", row.get("checkpoint_id")
            ),
            "member_id": row.get("member_id"),
        }
        for row in rows
    ]
    identity_sha256 = hashlib.sha256(
        json.dumps(
            _strict_jsonable(identity_rows),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": schema,
        "status": "complete",
        "records": len(rows),
        "artifact_path": destination.name,
        "artifact_sha256": _sha256_file(str(destination)),
        "record_identity_sha256": identity_sha256,
        "record_identity_fields": (
            "case_id",
            "revision_id",
            "entry_episode_id",
            "decision_at",
            "feature_max_at",
            "checkpoint_id",
            "member_id",
        ),
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "decision_stage": decision_stage,
        "selection_contract": "first_online_stage_occurrence_by_revision_index_v1",
        "input_protocol": INFERENCE_INPUT_PROTOCOL,
        "outcome_fields_used": False,
        "checkpoint_ids": sorted(checkpoint_ids),
        "case_input_manifest_sha256": case_input_manifest_sha256,
        "case_library_manifest_sha256": case_library_manifest_sha256,
        "head_schema": (
            dict(OUTCOME_BLIND_HEAD_WIDTHS)
            if schema == HEAD_ARTIFACT_SCHEMA
            else None
        ),
    }
    manifest_path = _artifact_manifest_path(destination)
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _strict_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strict_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    raise RepresentationDataError(
        f"metrics contain unsupported JSON value {type(value).__name__}"
    )


def _validate_neutral_fit_metrics_protocol(value: Mapping[str, Any]) -> None:
    report = _neutral_exact_mapping(value, (
        "schema_version", "mode", "pipeline_scope", "rows", "split_rows",
        "ensemble_members", "artifacts", "lineage", "direct_source_preprocessing",
        "training_performed", "validation_used_for_optimization",
        "holdout_used_for_optimization", "holdout_used_for_selection",
        "model_selection_performed", "threshold_search_performed",
        "outcome_fields_used", "model_capability_validated",
        "retrieval_quality_validated", "ood_capability_validated",
        "trading_edge_claimed", "action_value_claimed",
    ), "fit report")
    members = report["ensemble_members"]
    if (
        report["schema_version"] != 2
        or report["mode"] != "neutral_market_representation_fit"
        or report["pipeline_scope"] not in {
            "three_window_pipeline_smoke", "ten_window_registered_fit",
        }
        or report["direct_source_preprocessing"]
        != neutral_direct_source_preprocessing_identity()
        or report["training_performed"] is not True
        or any(report[name] is not False for name in (
            "validation_used_for_optimization", "holdout_used_for_optimization",
            "holdout_used_for_selection", "model_selection_performed",
            "threshold_search_performed", "outcome_fields_used",
            "model_capability_validated", "retrieval_quality_validated",
            "ood_capability_validated", "trading_edge_claimed",
            "action_value_claimed",
        ))
        or not isinstance(members, list)
        or len(members) < 3
        or not isinstance(report["lineage"], Mapping)
    ):
        raise RepresentationDataError(
            "neutral fit metrics B0 protocol is missing or differs"
        )
    rows = _neutral_positive_int(report["rows"], "fit rows")
    split_rows = _neutral_exact_mapping(
        report["split_rows"], ("train", "validation", "holdout"),
        "fit split rows",
    )
    for role in split_rows:
        _neutral_positive_int(split_rows[role], f"fit {role} rows")
    if sum(split_rows.values()) != rows:
        raise RepresentationDataError("neutral fit row totals differ")
    artifacts = _neutral_exact_mapping(
        report["artifacts"], ("embeddings", "active_heads"), "fit artifacts"
    )
    artifact_records: dict[str, int] = {}
    for name, raw_artifact in artifacts.items():
        artifact = _neutral_exact_mapping(
            raw_artifact, ("path", "manifest", "records"), f"{name} artifact"
        )
        if not all(isinstance(artifact[key], str) and artifact[key]
                   for key in ("path", "manifest")):
            raise RepresentationDataError("neutral artifact path contract differs")
        artifact_records[name] = _neutral_positive_int(
            artifact["records"], f"{name} artifact rows"
        )

    seeds: set[int] = set()
    checkpoint_ids: set[str] = set()
    checkpoint_paths: set[str] = set()
    for index, raw_member in enumerate(members):
        member = _neutral_exact_mapping(raw_member, (
            "member_id", "seed", "checkpoint_id", "checkpoint_path", "metrics",
        ), "ensemble member")
        checkpoint_id = member["checkpoint_id"]
        if (
            member["member_id"] != f"member-{index:03d}"
            or type(member["seed"]) is not int
            or not isinstance(checkpoint_id, str)
            or len(checkpoint_id) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint_id)
            or not isinstance(member["checkpoint_path"], str)
            or not member["checkpoint_path"]
            or not isinstance(member["metrics"], Mapping)
        ):
            raise RepresentationDataError("neutral ensemble member contract differs")
        _validate_neutral_member_metrics_protocol(member["metrics"])
        if member["metrics"]["split_counts"] != split_rows:
            raise RepresentationDataError("neutral member split counts differ")
        seeds.add(member["seed"])
        checkpoint_ids.add(checkpoint_id)
        checkpoint_paths.add(member["checkpoint_path"])
    if not all(len(values) == len(members) for values in (
        seeds, checkpoint_ids, checkpoint_paths,
    )):
        raise RepresentationDataError("neutral ensemble members are not independent")
    if (
        artifact_records["embeddings"] < rows
        or artifact_records["active_heads"] != rows * len(members)
    ):
        raise RepresentationDataError("neutral fit artifact row counts differ")


def _neutral_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RepresentationDataError("neutral fit metrics contain duplicate keys")
        value[key] = item
    return value


def _reject_nonstandard_json_number(token: str) -> None:
    raise RepresentationDataError(
        f"neutral fit metrics contain non-standard number {token}"
    )


def load_neutral_fit_metrics(path: str | Path) -> Mapping[str, Any]:
    """Load a B0 metrics artifact, rejecting missing/tampered protocol data."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RepresentationDataError("neutral fit metrics are not a regular file")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_neutral_json_object,
            parse_constant=_reject_nonstandard_json_number,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError("neutral fit metrics are invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RepresentationDataError("neutral fit metrics must be an object")
    _validate_neutral_fit_metrics_protocol(payload)
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    if _neutral_mode_requested(args):
        _validate_neutral_cli(args, argv=raw_argv)
        if args.neutral_fit:
            collection = _load_neutral_fit_collection(
                input_manifest_paths=args.market_case_input_manifest,
                run_manifest_paths=args.market_case_run_manifest,
            )
            raw_report = _neutral_fit_report(collection, args)
        else:
            dataset = _load_neutral_market_dataset(
                input_manifest_path=args.market_case_input_manifest[0],
                run_manifest_path=args.market_case_run_manifest[0],
            )
            raw_report = (
                _neutral_dataset_audit_report(
                    dataset,
                    embedding_kind=args.market_embedding_kind,
                )
                if args.neutral_dataset_audit_only
                else _neutral_single_batch_smoke_report(dataset)
            )
        report = _strict_jsonable(raw_report)
        if args.neutral_fit:
            _validate_neutral_fit_metrics_protocol(report)
        if args.metrics_output:
            _atomic_json(args.metrics_output, report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    require_torch()
    export_requested = bool(args.embedding_output or args.head_output)
    if export_requested and not args.embedding_stage:
        raise RepresentationDataError(
            "decision-time export requires an explicit --embedding-stage"
        )
    case_input_manifest_sha256: str | None = None
    case_library_manifest_sha256: str | None = None
    if args.synthetic_smoke:
        examples, targets = _synthetic_examples(args.seed)
        # Fixed roles make the smoke test deterministic and ensure both sets
        # exist; real data always uses connected episode/input grouping below.
        splits = {
            example.case.revision_id: (
                "validation" if index in {2, 7} else "test" if index == 11 else "train"
            )
            for index, example in enumerate(examples)
        }
        export_case_population = tuple(example.case for example in examples)
        export_prepared_lookup = {
            example.case.revision_id: example for example in examples
        }
        export_splits = splits
    else:
        if not args.case_input_shard:
            raise RepresentationDataError(
                "real training requires --case-input-shard"
            )
        (
            case_input_manifest_sha256,
            expected_input_rows,
            case_run_manifest_path,
        ) = (
            _validate_case_input_stream_manifest(
                args.case_input_shard,
                args.case_input_manifest,
                args.case_input_manifest_sha,
                args.case_run_manifest_sha,
            )
        )
        case_library_manifest_sha256 = _validate_case_library_manifest(
            args.case_library_manifest,
            args.case_library_manifest_sha,
            input_manifest_path=args.case_input_manifest,
            input_manifest_sha256=case_input_manifest_sha256,
            input_rows=expected_input_rows,
            run_manifest_path=case_run_manifest_path,
            run_manifest_sha256=_normalized_sha256(
                args.case_run_manifest_sha, name="--case-run-manifest-sha"
            ),
        )
        input_rows = _read_records(args.case_input_shard)
        if len(input_rows) != expected_input_rows:
            raise RepresentationDataError(
                "materialized case input rows disagree with committed manifest"
            )
        raw_cases = tuple(
            representation_case_from_case_input_row(row) for row in input_rows
        )
        cases = deduplicate_causal_inputs(raw_cases)
        built = build_observable_revision_targets(input_rows)
        if args.self_supervised_targets:
            external_targets = _target_map(args.self_supervised_targets)
            _validate_external_target_cache(external_targets, built)
        target_map = {
            revision_id: record.target for revision_id, record in built.items()
        }
        missing = sorted(
            case.revision_id for case in cases if case.revision_id not in target_map
        )
        if missing:
            raise RepresentationDataError(
                f"self-supervised target stream omits revisions: {missing[:10]}"
            )
        store = _build_store(
            cases,
            args.canonical_view,
            args.canonical_view_sha,
            args.tick_size,
            args.availability_time,
            args.canonical_lineage_manifest,
            args.canonical_lineage_manifest_sha,
        )
        examples = tuple(prepare_representation_case(case, store) for case in cases)
        targets = tuple(target_map[case.revision_id] for case in cases)
        raw_splits = assign_leakage_safe_splits(raw_cases, seed=args.seed)
        splits = {
            case.revision_id: raw_splits[case.revision_id] for case in cases
        }
        export_case_population = raw_cases
        export_prepared_lookup = {}
        export_splits = raw_splits

    export_examples: tuple[PreparedRepresentationCase, ...] = ()
    if export_requested:
        selected_cases = select_first_causal_stage_revisions(
            export_case_population,
            decision_stage=args.embedding_stage,
        )
        if not selected_cases:
            raise RepresentationDataError(
                "no causal revisions exist at the requested embedding stage"
            )
        export_examples = tuple(
            (
                export_prepared_lookup[case.revision_id]
                if case.revision_id in export_prepared_lookup
                else prepare_representation_case(case, store)
            )
            for case in selected_cases
        )

    if args.ensemble_size < 1:
        raise RepresentationDataError("ensemble-size must be positive")
    members: list[dict[str, Any]] = []
    checkpoint_ids: set[str] = set()
    embedding_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for member_index in range(args.ensemble_size):
        member_args = argparse.Namespace(**vars(args))
        member_args.seed = args.seed + member_index * 100_003
        model, member_metrics = _train(examples, targets, splits, member_args)
        checkpoint_id = representation_checkpoint_id(model)
        if checkpoint_id in checkpoint_ids:
            raise RepresentationDataError(
                "independent ensemble members produced duplicate checkpoint identity"
            )
        checkpoint_ids.add(checkpoint_id)
        if export_requested:
            inference_batch, _ = collate_representation_cases(
                export_examples,
                mask_probability=0.0,
                seed=args.seed,
            )
            inference_batch = inference_batch.to(args.device)
            if args.embedding_output and member_index == 0:
                embedding_rows.extend(
                    record.as_dict()
                    for record in encode_decision_time_records(
                        model,
                        inference_batch,
                        export_examples,
                        split_roles=export_splits,
                        decision_stage=args.embedding_stage,
                    )
                )
            if args.head_output:
                head_rows.extend(
                    record.as_dict()
                    for record in encode_decision_time_head_records(
                        model,
                        inference_batch,
                        export_examples,
                        member_id=f"member-{member_index:03d}",
                    )
                )
        checkpoint_path: Path | None = None
        if args.checkpoint:
            base = Path(args.checkpoint).resolve()
            checkpoint_path = (
                base
                if args.ensemble_size == 1
                else base.with_name(
                    f"{base.stem}.member-{member_index:03d}{base.suffix or '.pt'}"
                )
            )
            save_representation_checkpoint(
                checkpoint_path,
                model,
                metadata={
                    "member_id": f"member-{member_index:03d}",
                    "seed": member_args.seed,
                    "split_counts": member_metrics["split_counts"],
                    "outcome_fields_used": False,
                },
            )
        members.append(
            {
                "member_id": f"member-{member_index:03d}",
                "seed": member_args.seed,
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": (
                    None if checkpoint_path is None else str(checkpoint_path)
                ),
                "metrics": member_metrics,
            }
        )
    exported_artifacts: dict[str, Any] = {}
    if args.embedding_output:
        embedding_manifest = _write_decision_artifact(
            args.embedding_output,
            embedding_rows,
            schema=EMBEDDING_ARTIFACT_SCHEMA,
            decision_stage=args.embedding_stage,
            checkpoint_ids=(members[0]["checkpoint_id"],),
            case_input_manifest_sha256=case_input_manifest_sha256,
            case_library_manifest_sha256=case_library_manifest_sha256,
        )
        exported_artifacts["embeddings"] = {
            "path": str(Path(args.embedding_output).resolve()),
            "manifest": str(embedding_manifest),
            "manifest_sha256": _sha256_file(str(embedding_manifest)),
            "reference_checkpoint_id": members[0]["checkpoint_id"],
        }
    if args.head_output:
        head_manifest = _write_decision_artifact(
            args.head_output,
            head_rows,
            schema=HEAD_ARTIFACT_SCHEMA,
            decision_stage=args.embedding_stage,
            checkpoint_ids=tuple(member["checkpoint_id"] for member in members),
            case_input_manifest_sha256=case_input_manifest_sha256,
            case_library_manifest_sha256=case_library_manifest_sha256,
        )
        exported_artifacts["heads"] = {
            "path": str(Path(args.head_output).resolve()),
            "manifest": str(head_manifest),
            "manifest_sha256": _sha256_file(str(head_manifest)),
            "member_checkpoint_ids": [
                member["checkpoint_id"] for member in members
            ],
        }
    stability_metric_paths = {
        "regime_centroid_accuracy": (
            "embedding_evaluation",
            "regime_centroid_accuracy",
        ),
        "cross_date_mechanism_retrieval_at_k": (
            "embedding_evaluation",
            "cross_date_mechanism_retrieval_at_k",
        ),
        "cross_date_mechanism_lift_over_chance": (
            "embedding_evaluation",
            "cross_date_mechanism_lift_over_chance",
        ),
        "classification_mean_relative_improvement": (
            "baseline_comparison",
            "mean_relative_improvement",
        ),
    }
    stability_metrics: dict[str, Any] = {}
    for name, (section, key) in stability_metric_paths.items():
        values = [
            float(member["metrics"][section][key])
            for member in members
            if math.isfinite(float(member["metrics"][section][key]))
        ]
        stability_metrics[name] = {
            "measured_members": len(values),
            "mean": float(np.mean(values)) if values else float("nan"),
            "population_std": float(np.std(values)) if values else float("nan"),
            "minimum": min(values) if values else float("nan"),
            "maximum": max(values) if values else float("nan"),
        }
    embedding_member_passes = [
        bool(member["metrics"]["embedding_evaluation"]["criteria_met"])
        for member in members
    ]
    task_member_passes = [
        bool(member["metrics"]["completion_criteria_met"])
        for member in members
    ]
    ensemble_stability = {
        "minimum_independent_members": 3,
        "member_count": len(members),
        "all_embedding_members_meet_criteria": all(embedding_member_passes),
        "all_task_members_meet_criteria": all(task_member_passes),
        "criteria_met": bool(
            len(members) >= 3
            and all(embedding_member_passes)
            and all(task_member_passes)
        ),
        "metrics": stability_metrics,
        "thresholds_are_fixed_development_defaults": True,
    }
    metrics = {
        "model_version": MODEL_VERSION,
        "ensemble_size": args.ensemble_size,
        "independent_checkpoint_ids": sorted(checkpoint_ids),
        "head_probability_interface": "decision_time_head_probabilities(output)",
        "members": members,
        "ensemble_stability": ensemble_stability,
        "exported_artifacts": exported_artifacts,
        "case_input_manifest_sha256": case_input_manifest_sha256,
        "case_library_manifest_sha256": case_library_manifest_sha256,
        "outcome_fields_used": False,
        "trading_edge_claimed": False,
    }
    strict_metrics = _strict_jsonable(metrics)
    if args.metrics_output:
        _atomic_json(args.metrics_output, strict_metrics)
    print(json.dumps(strict_metrics, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
