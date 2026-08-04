#!/usr/bin/env python3
"""Stream/resume governed all-candidate action-clock and shadow replay.

The runner is deliberately memory-bounded and version-aware.  v2.3 remains
the default, while later registered model versions can inherit the same
durability protocol without falling back to the full-snapshot, full-year
in-memory replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.action_clock import (  # noqa: E402
    ActionClockProtocol,
    ActionClockReplay,
    FLAT_ACTIONS,
    PlanLineageStore,
    action_clock_code_fingerprint,
    build_action_clock_engine,
    candidate_action_row,
    candidate_state_row,
    executable_candidate_groups,
)
from smc_trader.action_clock_artifact_schema import (  # noqa: E402
    CALIBRATION_STREAM_FIELD_TYPES,
)
from smc_trader.action_equivalence import (  # noqa: E402
    ActionEquivalenceProtocol,
    action_equivalence_code_fingerprint,
)
from smc_trader.artifact_stream import (  # noqa: E402
    atomic_bytes,
    canonical_json,
    new_stream_state,
    sha256_file,
    verify_stream_shards,
    write_stream_manifest,
    write_stream_shards_bounded,
)
from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.calibration_replay import (  # noqa: E402
    HASH_MODE,
    ReplayCheckpointStore,
    iter_after_source_checkpoint,
)
from smc_trader.io import load_ohlcv  # noqa: E402
from smc_trader.mbo import MinuteExecutionRealityStore  # noqa: E402
from smc_trader.model import Playbook, to_primitive  # noqa: E402
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.shadow_replay import (  # noqa: E402
    FrozenShadowReplay,
    shadow_replay_code_fingerprint,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


STREAM_KEYS = {
    "decision_shards": "asof",
    "candidate_state_shards": "candidate_id",
    "candidate_action_shards": "action_key",
    "flat_outcome_shards": "action_key",
    "position_action_shards": "position_action_key",
    "position_outcome_shards": "position_action_key",
}

POLICY_VARIANTS = {
    "all_three": (),
    "without_failed_auction_value_return": (
        Playbook.FAILED_AUCTION_VALUE_RETURN,
    ),
    "without_liquidity_sweep_reversal": (
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    ),
}


def _aware(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("America/New_York")
    return timestamp


def _deadline(timestamp: pd.Timestamp) -> pd.Timestamp:
    local = timestamp.tz_convert("America/New_York")
    day = local.tz_localize(None).normalize()
    if local.hour >= 18:
        day += pd.Timedelta(days=1)
    return (day + pd.Timedelta(hours=17)).tz_localize(
        "America/New_York",
        ambiguous=True,
        nonexistent="shift_forward",
    )


def _missing_execution(asof: pd.Timestamp) -> ExecutionRealityInput:
    return ExecutionRealityInput(
        spread_points=None,
        expected_slippage_points=0.0,
        commission_per_contract_per_side=0.0,
        deadline=_deadline(asof),
        data_age_seconds=61.0,
        source="missing_execution_authority",
    )


def _load_mbo_store(
    path: str | Path,
    *,
    validation,
) -> tuple[MinuteExecutionRealityStore, dict[str, Any]]:
    source = Path(path)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"MBO minute reality requires a manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("format_version", 0)) != 1:
        raise ValueError("unsupported MBO minute manifest format")
    if manifest.get("validation_protocol_hash") != validation.fingerprint:
        raise ValueError("MBO minute manifest is not bound to this validation protocol")
    if manifest.get("output_sha256") != sha256_file(source):
        raise ValueError("MBO minute file hash differs from its manifest")
    start = pd.Timestamp(manifest.get("start"))
    end = pd.Timestamp(manifest.get("end_exclusive"))
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("MBO minute manifest has an invalid interval")
    window = validation.classify_mbo(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
    )
    if window.role != "revealed_execution_development":
        raise RuntimeError("sealed MBO cannot enter development replay")
    if bool(manifest.get("sealed_holdout_read")):
        raise RuntimeError("MBO development file declares a sealed reveal")
    return MinuteExecutionRealityStore.from_parquet(source), manifest


def _execution_for_bar(
    bar,
    mbo_store: MinuteExecutionRealityStore | None,
) -> ExecutionRealityInput:
    deadline = _deadline(bar.end)
    if mbo_store is None:
        return _missing_execution(bar.end)
    return mbo_store.for_bar(bar, deadline=deadline)


def action_clock_pipeline_fingerprint() -> str:
    """Hash executable pipeline code; dynamic config inputs bind separately."""

    digest = hashlib.sha256()
    for relative in (
        "smc_trader/action_clock.py",
        "smc_trader/shadow_replay.py",
        "smc_trader/artifact_stream.py",
        "smc_trader/action_clock_artifact_schema.py",
        "smc_trader/calibration_replay.py",
        "scripts/run_action_clock_calibration.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolved_dependency_hashes(
    config_payload: dict[str, Any],
) -> dict[str, str | None]:
    """Bind every mutable file referenced by a model config.

    Hashing the config alone is insufficient because it usually contains
    stable paths to registries and protocols whose contents can change.
    """

    raw_paths = {
        "playbook_registry": config_payload.get("playbook_registry"),
        "belief_calibration_artifact": config_payload.get(
            "calibration_artifact"
        ),
        "action_equivalence_protocol": config_payload.get(
            "action_equivalence_protocol"
        ),
        "action_clock_value_protocol": config_payload.get(
            "action_clock_value_protocol"
        ),
        "validation_protocol": config_payload.get("validation_protocol"),
        "structure_protocol": (
            config_payload.get("observer", {}).get("structure_protocol")
            if isinstance(config_payload.get("observer"), dict)
            else None
        ),
    }
    output: dict[str, str | None] = {}
    for name, raw_path in raw_paths.items():
        if raw_path in (None, ""):
            output[f"{name}_sha256"] = None
            continue
        source = Path(str(raw_path))
        if not source.is_absolute():
            source = ROOT / source
        if not source.is_file():
            raise FileNotFoundError(
                f"configured dependency is not a regular file: {source}"
            )
        output[f"{name}_sha256"] = sha256_file(source)
    return output


def _artifact_namespace(config_payload: dict[str, Any]) -> str:
    configured = str(
        config_payload.get("calibration_artifact_namespace", "")
    ).strip()
    raw = configured or str(config_payload.get("version", "")).strip()
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    if not normalized:
        raise ValueError("calibration artifact namespace is empty")
    return normalized


def _sync_checkpoint_aliases(state: dict[str, Any]) -> None:
    decision_stream = state["streams"]["decision_shards"]
    state["decision_rows"] = int(decision_stream["rows"])
    state["next_shard_index"] = int(decision_stream["next_shard_index"])
    state["committed_shards"] = decision_stream["committed_shards"]


def _verify_all_streams(destination: Path, state: dict[str, Any]) -> None:
    if set(state["streams"]) != set(STREAM_KEYS):
        raise ValueError("checkpoint shard stream family changed")
    for stream_state in state["streams"].values():
        verify_stream_shards(destination, stream_state)


def _progress(
    state: dict[str, Any],
    *,
    total_source_rows: int,
    session_started: float,
    session_source_start: int,
) -> dict[str, Any]:
    buffers = state.get("buffers", {})

    def stream_rows(name: str) -> int:
        return int(state["streams"][name]["rows"]) + len(
            buffers.get(name, ())
        )

    consumed = int(state["source_rows_consumed"])
    session_rows = consumed - session_source_start
    elapsed = max(time.monotonic() - session_started, 1e-9)
    rate = session_rows / elapsed
    remaining = max(0, total_source_rows - consumed)
    return {
        "source_rows": consumed,
        "source_rows_total": total_source_rows,
        "complete_percent": round(
            100.0 * consumed / max(total_source_rows, 1),
            3,
        ),
        "rows_per_second_this_process": round(rate, 2),
        "eta_seconds_this_process_rate": (
            None if rate <= 0 else round(remaining / rate, 1)
        ),
        "decision_clocks": int(state["decision_clocks"]),
        "candidate_plans": int(state["candidate_plans"]),
        "candidate_action_rows": int(
            stream_rows("candidate_action_shards")
        ),
        "flat_outcomes": int(
            stream_rows("flat_outcome_shards")
        ),
        "position_action_rows": int(
            stream_rows("position_action_shards")
        ),
        "position_outcomes": int(
            stream_rows("position_outcome_shards")
        ),
        "active_flat_shadow_actions": len(state["shadow"].active),
        "active_position_trials": len(state["shadow"].position_trials),
        "active_position_clock_states": sum(
            len(action.position_states)
            for action in state["shadow"].active.values()
        ),
        "data_gap_resets": int(state.get("data_gap_resets", 0)),
        "data_gap_open_minutes": int(
            state.get("data_gap_open_minutes", 0)
        ),
        "last_asof": (
            None
            if state["last_asof"] is None
            else pd.Timestamp(state["last_asof"]).isoformat()
        ),
    }


def _decision_row(
    snapshot,
    *,
    candidate_count: int,
    shadow: FrozenShadowReplay,
) -> dict[str, Any]:
    best = snapshot.decision.utilities[0] if snapshot.decision.utilities else None
    return {
        "asof": snapshot.observation.asof,
        "calibration_state_commitment": snapshot.snapshot_hash,
        "symbol": snapshot.observation.symbol,
        "instrument_id": snapshot.observation.instrument_id,
        "model_action": snapshot.decision.selected_action.value,
        "risk_action": snapshot.risk.final_action.value,
        "best_variant_action": None if best is None else best.action.value,
        "best_variant_utility_R": None if best is None else best.utility,
        "best_variant_hypothesis_key": (
            None if best is None else best.hypothesis_key
        ),
        "unique_executable_candidate_plans": int(candidate_count),
        "active_flat_shadow_actions": len(shadow.active),
        "active_position_trials": len(shadow.position_trials),
        "execution_source": snapshot.observation.execution.source,
        "snapshot_full_hash_computed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/model_v2_3_action_clock_base.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2_3.json",
    )
    parser.add_argument(
        "--action-clock-protocol",
        default="configs/action_clock_value_protocol_v2_3.json",
    )
    parser.add_argument(
        "--action-equivalence-protocol",
        default="configs/action_equivalence_v2_2.json",
    )
    parser.add_argument("--mbo-execution")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--warmup-days", type=int, default=45)
    parser.add_argument("--shard-rows", type=int, default=25_000)
    parser.add_argument("--checkpoint-bars", type=int, default=25_000)
    parser.add_argument(
        "--policy-variant",
        choices=tuple(POLICY_VARIANTS),
        default="all_three",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostic-stop-after-bars",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.warmup_days < 1
        or args.shard_rows < 1
        or args.checkpoint_bars < 1
    ):
        raise ValueError("warmup, shard and checkpoint limits must be positive")
    start = _aware(args.start)
    end = _aware(args.end)
    if end <= start:
        raise ValueError("action-clock replay interval must be positive")
    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_ohlcv(start, end)
    if window.role not in {"action_clock_development", "rolling_validation"}:
        raise RuntimeError(
            "action-clock sample generation is limited to registered "
            "pre-holdout windows"
        )
    source = Path(args.source)
    source_hash = sha256_file(source)
    if source_hash != validation.causal_front_sha256:
        raise RuntimeError("OHLCV source differs from the registered causal front")
    config = Path(args.config)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    model_version = str(config_payload.get("version", "")).strip()
    if (
        not model_version
        or config_payload.get("action_clock_value_artifact") is not None
    ):
        raise ValueError(
            "sample generation requires a versioned unfitted base config"
        )
    if model_version.startswith("3.0."):
        if (
            config_payload.get("experiment_id")
            != "EXP-SMC-3.0.0-001-SWING-BOS"
        ):
            raise ValueError("v3 sample generation requires the frozen experiment id")
        if (
            config_payload.get("calibration_run_role")
            != "action_clock_sample_generation"
            or config_payload.get("calibration_artifact") in (None, "")
        ):
            raise ValueError(
                "v3 action-clock generation is unavailable until a fitted "
                "belief calibration is frozen in an explicit action-clock config"
            )
    elif not model_version.startswith("2.3."):
        raise ValueError("unsupported action-clock calibration model version")
    action_clock_protocol = ActionClockProtocol.from_file(
        args.action_clock_protocol
    )
    action_protocol = ActionEquivalenceProtocol.from_file(
        args.action_equivalence_protocol
    )
    dependency_hashes = _resolved_dependency_hashes(config_payload)
    expected_protocol_hashes = {
        "action_clock_value_protocol_sha256": (
            action_clock_protocol.fingerprint
        ),
        "action_equivalence_protocol_sha256": action_protocol.fingerprint,
        "validation_protocol_sha256": validation.fingerprint,
    }
    for name, expected in expected_protocol_hashes.items():
        configured = dependency_hashes.get(name)
        if configured is None:
            raise ValueError(
                f"model config must explicitly bind {name.removesuffix('_sha256')}"
            )
        if configured != expected:
            raise ValueError(
                f"CLI and model config disagree on "
                f"{name.removesuffix('_sha256')}"
            )
    disabled_playbooks = tuple(POLICY_VARIANTS[args.policy_variant])
    mbo_store = None
    mbo_manifest = None
    if args.mbo_execution:
        mbo_store, mbo_manifest = _load_mbo_store(
            args.mbo_execution,
            validation=validation,
        )
    load_start = start - pd.Timedelta(days=args.warmup_days)
    loaded = load_ohlcv(source, start=load_start, end=end)
    if not loaded.contract_selection_causal:
        raise RuntimeError(
            "action-clock calibration requires previous-session causal "
            "contract selection"
        )
    replay_frame = loaded.frame.loc[
        loaded.frame.index + pd.Timedelta(minutes=1) < end
    ]
    if replay_frame.empty:
        raise ValueError("action-clock replay interval has no source rows")
    total_source_rows = int(len(replay_frame))
    artifact_namespace = _artifact_namespace(config_payload)
    pipeline_hash = action_clock_pipeline_fingerprint()

    destination = Path(args.output)
    completed_path = destination / "COMPLETED.json"
    if args.resume:
        if completed_path.exists():
            raise FileExistsError("action-clock replay is already complete")
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing non-resume use of non-empty output")
    destination.mkdir(parents=True, exist_ok=True)

    bindings = {
        "runner": "action_clock_calibration_stream_v2",
        "model_version": model_version,
        "experiment_id": config_payload.get("experiment_id"),
        "artifact_namespace": artifact_namespace,
        "source_sha256": source_hash,
        "source_rows": total_source_rows,
        "source_first": replay_frame.index[0].isoformat(),
        "source_last": replay_frame.index[-1].isoformat(),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "validation_window_role": window.role,
        "warmup_days": int(args.warmup_days),
        "config_sha256": sha256_file(config),
        "base_model_code_hash": model_code_fingerprint(),
        "action_equivalence_code_hash": (
            action_equivalence_code_fingerprint()
        ),
        "action_clock_code_hash": action_clock_code_fingerprint(),
        "shadow_replay_code_hash": shadow_replay_code_fingerprint(),
        "pipeline_hash": pipeline_hash,
        "validation_protocol_hash": validation.fingerprint,
        "belief_calibration_valid_from": (
            validation.belief_calibration_valid_from.isoformat()
        ),
        "action_clock_protocol_hash": action_clock_protocol.fingerprint,
        "action_equivalence_protocol_hash": action_protocol.fingerprint,
        "policy_variant": args.policy_variant,
        "disabled_playbooks": sorted(item.value for item in disabled_playbooks),
        "mbo_execution_sha256": (
            None if args.mbo_execution is None else sha256_file(args.mbo_execution)
        ),
        "mbo_execution_manifest_sha256": (
            None
            if args.mbo_execution is None
            else sha256_file(
                Path(args.mbo_execution).with_suffix(
                    Path(args.mbo_execution).suffix + ".manifest.json"
                )
            )
        ),
        "shard_rows": int(args.shard_rows),
        "checkpoint_bars": int(args.checkpoint_bars),
        "hash_mode": HASH_MODE,
        "data_gap_policy": (
            "censor_active_reset_reader_brain_lineage_after_5_open_minutes"
        ),
        **dependency_hashes,
    }
    checkpoint = ReplayCheckpointStore(destination / "_checkpoint")
    if args.resume:
        if not checkpoint.exists:
            raise FileNotFoundError("resume requested without a checkpoint")
        state = checkpoint.load(
            expected_bindings=bindings,
            expected_replay_type=ActionClockReplay,
        )
        _verify_all_streams(destination, state)
        state["resume_count"] = int(state["resume_count"]) + 1
        _sync_checkpoint_aliases(state)
        checkpoint.save(state, bindings=bindings)
    else:
        replay = ActionClockReplay(
            build_action_clock_engine(
                config,
                disabled_playbooks=disabled_playbooks,
            )
        )
        streams = {
            name: new_stream_state(CALIBRATION_STREAM_FIELD_TYPES[name])
            for name in STREAM_KEYS
        }
        state = {
            "replay": replay,
            "lineage": PlanLineageStore(),
            "shadow": FrozenShadowReplay(),
            "streams": streams,
            "buffers": {name: [] for name in STREAM_KEYS},
            "processed_bars": 0,
            "source_rows_consumed": 0,
            "last_checkpoint_processed_bars": 0,
            "decision_clocks": 0,
            "candidate_plans": 0,
            "model_action_counts": {},
            "risk_action_counts": {},
            "last_source_start": None,
            "last_asof": None,
            "resume_count": 0,
            "data_gap_resets": 0,
            "data_gap_open_minutes": 0,
            "peak_buffer_rows_by_stream": {
                name: 0 for name in STREAM_KEYS
            },
            "decision_rows": 0,
            "next_shard_index": 0,
            "committed_shards": streams["decision_shards"][
                "committed_shards"
            ],
        }
    replay = state["replay"]
    lineage = state["lineage"]
    shadow = state["shadow"]
    buffers = state.get("buffers")
    if not isinstance(buffers, dict) or set(buffers) != set(STREAM_KEYS):
        raise ValueError("checkpoint buffer stream family changed")
    if any(
        not isinstance(rows, list) or len(rows) >= args.shard_rows
        for rows in buffers.values()
    ):
        raise ValueError("checkpoint contains an invalid bounded stream buffer")
    peak_buffer_rows = state.setdefault(
        "peak_buffer_rows_by_stream",
        {name: 0 for name in STREAM_KEYS},
    )
    if set(peak_buffer_rows) != set(STREAM_KEYS):
        raise ValueError("checkpoint buffer telemetry stream family changed")
    state.setdefault("data_gap_resets", 0)
    state.setdefault("data_gap_open_minutes", 0)
    session_started = time.monotonic()
    session_source_start = int(state["source_rows_consumed"])
    durable_progress = _progress(
        state,
        total_source_rows=total_source_rows,
        session_started=session_started,
        session_source_start=session_source_start,
    )

    def append_bounded(
        name: str,
        rows: list[dict[str, Any]],
    ) -> None:
        """Append completed rows without ever exceeding the registered cap."""

        if name not in buffers:
            raise KeyError(f"unknown calibration stream: {name}")
        offset = 0
        while offset < len(rows):
            room = args.shard_rows - len(buffers[name])
            if room <= 0:
                write_stream_shards_bounded(
                    destination,
                    name,
                    buffers[name],
                    state["streams"][name],
                    key_column=STREAM_KEYS[name],
                    maximum_rows=args.shard_rows,
                    field_types=CALIBRATION_STREAM_FIELD_TYPES[name],
                )
                room = args.shard_rows
            take = min(room, len(rows) - offset)
            buffers[name].extend(rows[offset : offset + take])
            offset += take
            peak_buffer_rows[name] = max(
                int(peak_buffer_rows[name]),
                len(buffers[name]),
            )
            if len(buffers[name]) == args.shard_rows:
                write_stream_shards_bounded(
                    destination,
                    name,
                    buffers[name],
                    state["streams"][name],
                    key_column=STREAM_KEYS[name],
                    maximum_rows=args.shard_rows,
                    field_types=CALIBRATION_STREAM_FIELD_TYPES[name],
                )

    def drain_shadow() -> None:
        flat, position_samples, position_outcomes = shadow.drain()
        append_bounded("flat_outcome_shards", flat)
        append_bounded("position_action_shards", position_samples)
        append_bounded("position_outcome_shards", position_outcomes)

    def commit_checkpoint(*, flush_partial: bool = False) -> None:
        nonlocal durable_progress
        if state["last_source_start"] is None:
            return
        if flush_partial:
            for name, key_column in STREAM_KEYS.items():
                write_stream_shards_bounded(
                    destination,
                    name,
                    buffers[name],
                    state["streams"][name],
                    key_column=key_column,
                    maximum_rows=args.shard_rows,
                    field_types=CALIBRATION_STREAM_FIELD_TYPES[name],
                )
        state["last_checkpoint_processed_bars"] = int(
            state["processed_bars"]
        )
        _sync_checkpoint_aliases(state)
        checkpoint.save(state, bindings=bindings)
        durable_progress = _progress(
            state,
            total_source_rows=total_source_rows,
            session_started=session_started,
            session_source_start=session_source_start,
        )
        atomic_bytes(
            destination / "progress.json",
            canonical_json(
                {
                    **durable_progress,
                    "status": "running",
                    "resume_supported": True,
                }
            ),
        )
        print(
            json.dumps(
                durable_progress,
                sort_keys=True,
            ),
            flush=True,
        )

    iterator = iter_after_source_checkpoint(
        replay_frame,
        state["last_source_start"],
        allow_data_gap_reset=True,
    )
    last_processed_was_source = False
    safe_source_checkpoint = False
    try:
        for bar in iterator:
            safe_source_checkpoint = False
            if bar.end >= end:
                break
            snapshot = replay.on_bar(
                bar,
                execution=_execution_for_bar(bar, mbo_store),
                belief_enabled=(
                    bar.end >= validation.belief_calibration_valid_from
                ),
            )
            if "data_gap_history_reset" in snapshot.observation.anomalies:
                flat, position_samples, position_outcomes = (
                    shadow.finalize_data_gap(bar.start)
                )
                append_bounded("flat_outcome_shards", flat)
                append_bounded(
                    "position_action_shards",
                    position_samples,
                )
                append_bounded(
                    "position_outcome_shards",
                    position_outcomes,
                )
                state["data_gap_resets"] = (
                    int(state["data_gap_resets"]) + 1
                )
                state["data_gap_open_minutes"] = (
                    int(state["data_gap_open_minutes"])
                    + int(bar.data_gap_before_minutes)
                )
            shadow.advance(bar, snapshot)
            drain_shadow()
            if {
                "contract_change_history_reset",
                "data_gap_history_reset",
            }.intersection(snapshot.observation.anomalies):
                lineage.reset()
            lineage.observe(snapshot.belief)
            state["processed_bars"] = int(state["processed_bars"]) + 1
            last_processed_was_source = not bar.synthetic_no_trade
            if last_processed_was_source:
                state["source_rows_consumed"] = (
                    int(state["source_rows_consumed"]) + 1
                )
                state["last_source_start"] = bar.start
            if snapshot.observation.asof >= start:
                candidates = executable_candidate_groups(
                    snapshot,
                    action_protocol,
                    lineage,
                    engine=replay.engine,
                    disabled_playbooks=disabled_playbooks,
                )
                for candidate in candidates:
                    append_bounded(
                        "candidate_state_shards",
                        [
                            candidate_state_row(
                                candidate,
                                snapshot,
                                tick_size=action_protocol.tick_size,
                            )
                        ],
                    )
                    append_bounded(
                        "candidate_action_shards",
                        [
                            candidate_action_row(
                                candidate,
                                snapshot,
                                action_id=action_id,
                                tick_size=action_protocol.tick_size,
                            )
                            for action_id in FLAT_ACTIONS
                        ],
                    )
                    shadow.register(candidate, snapshot)
                drain_shadow()
                append_bounded(
                    "decision_shards",
                    [
                        _decision_row(
                            snapshot,
                            candidate_count=len(candidates),
                            shadow=shadow,
                        )
                    ],
                )
                state["decision_clocks"] = int(
                    state["decision_clocks"]
                ) + 1
                state["candidate_plans"] = int(
                    state["candidate_plans"]
                ) + len(candidates)
                model_action = snapshot.decision.selected_action.value
                risk_action = snapshot.risk.final_action.value
                state["model_action_counts"][model_action] = (
                    int(state["model_action_counts"].get(model_action, 0)) + 1
                )
                state["risk_action_counts"][risk_action] = (
                    int(state["risk_action_counts"].get(risk_action, 0)) + 1
                )
                state["last_asof"] = snapshot.observation.asof
            safe_source_checkpoint = last_processed_was_source
            due = (
                int(state["processed_bars"])
                - int(state["last_checkpoint_processed_bars"])
                >= args.checkpoint_bars
            )
            if last_processed_was_source and due:
                commit_checkpoint()
            if (
                args.diagnostic_stop_after_bars > 0
                and int(state["processed_bars"])
                >= args.diagnostic_stop_after_bars
                and last_processed_was_source
            ):
                commit_checkpoint()
                raise RuntimeError(
                    "intentional diagnostic interruption after checkpoint"
                )
    except KeyboardInterrupt:
        if safe_source_checkpoint:
            commit_checkpoint()
        raise
    except Exception as exc:
        atomic_bytes(
            destination / "progress.json",
            canonical_json(
                {
                    **durable_progress,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "resume_supported": checkpoint.exists,
                    "durable_checkpoint_only": True,
                }
            ),
        )
        raise
    if state["last_asof"] is None or int(state["decision_clocks"]) == 0:
        raise ValueError("interval produced no completed decision clocks")
    if int(state["source_rows_consumed"]) != total_source_rows:
        raise RuntimeError(
            "source iterator ended before every bound source row "
            "was consumed"
        )
    flat, position_samples, position_outcomes = shadow.finalize_boundary(end)
    append_bounded("flat_outcome_shards", flat)
    append_bounded("position_action_shards", position_samples)
    append_bounded("position_outcome_shards", position_outcomes)
    commit_checkpoint(flush_partial=True)

    manifest_hashes: dict[str, str] = {}
    for name in STREAM_KEYS:
        manifest_path = write_stream_manifest(
            destination,
            name,
            state["streams"][name],
            artifact=f"{artifact_namespace}_{name}",
            bindings=bindings,
        )
        manifest_hashes[name] = sha256_file(manifest_path)
    candidate_action_rows = int(
        state["streams"]["candidate_action_shards"]["rows"]
    )
    if int(state["streams"]["candidate_state_shards"]["rows"]) != int(
        state["candidate_plans"]
    ):
        raise AssertionError("candidate states are not conserved")
    if candidate_action_rows != int(state["candidate_plans"]) * len(
        FLAT_ACTIONS
    ):
        raise AssertionError("candidate action alternatives are not conserved")
    if int(state["streams"]["flat_outcome_shards"]["rows"]) != (
        int(state["candidate_plans"]) * len(FLAT_ACTIONS)
    ):
        raise AssertionError("flat shadow outcomes are not conserved")
    if int(state["streams"]["position_action_shards"]["rows"]) != int(
        state["streams"]["position_outcome_shards"]["rows"]
    ):
        raise AssertionError("position action/outcome rows are not conserved")
    summary = {
        "version": str(config_payload["version"]),
        "source": str(loaded.source),
        "source_sha256": source_hash,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "validation_window_role": window.role,
        "validation_protocol_hash": validation.fingerprint,
        "action_clock_protocol_hash": action_clock_protocol.fingerprint,
        "action_equivalence_protocol_hash": action_protocol.fingerprint,
        "pipeline_hash": pipeline_hash,
        "policy_variant": args.policy_variant,
        "disabled_playbooks": sorted(item.value for item in disabled_playbooks),
        "mbo_execution_manifest": mbo_manifest,
        "decision_clocks": int(state["decision_clocks"]),
        "candidate_plans": int(state["candidate_plans"]),
        "candidate_state_rows": int(
            state["streams"]["candidate_state_shards"]["rows"]
        ),
        "candidate_action_rows": candidate_action_rows,
        "flat_outcomes": int(
            state["streams"]["flat_outcome_shards"]["rows"]
        ),
        "position_action_rows": int(
            state["streams"]["position_action_shards"]["rows"]
        ),
        "position_outcomes": int(
            state["streams"]["position_outcome_shards"]["rows"]
        ),
        "model_action_counts": dict(
            sorted(state["model_action_counts"].items())
        ),
        "risk_action_counts": dict(
            sorted(state["risk_action_counts"].items())
        ),
        "plan_lineages": lineage.size,
        "plan_lineages_total_observed": lineage.total_observed,
        "source_rows_processed": int(state["source_rows_consumed"]),
        "source_rows_total": total_source_rows,
        "data_gap_resets": int(state["data_gap_resets"]),
        "data_gap_open_minutes": int(state["data_gap_open_minutes"]),
        "completed_percent": 100.0,
        "decision_storage": "bounded_parquet_shards",
        "candidate_storage": "bounded_parquet_shards",
        "outcome_storage": "bounded_parquet_shards",
        "checkpoint_resume_supported": True,
        "checkpoint_interval_bars": int(args.checkpoint_bars),
        "maximum_in_memory_completed_rows_per_stream": int(args.shard_rows),
        "maximum_total_completed_buffer_rows_upper_bound": int(
            args.shard_rows * len(STREAM_KEYS)
        ),
        "observed_peak_completed_buffer_rows_by_stream": dict(
            sorted(peak_buffer_rows.items())
        ),
        "resume_count": int(state["resume_count"]),
        "snapshot_hash_mode": HASH_MODE,
        "full_snapshot_hash_per_minute": False,
        "future_path_visible_to_candidate_generation": False,
        "right_boundary_actions_censored": True,
    }
    summary_path = destination / "summary.json"
    atomic_bytes(summary_path, canonical_json(to_primitive(summary)))
    completion = {
        "format_version": 1,
        "artifact": f"{artifact_namespace}_action_clock_calibration",
        "status": "complete",
        "bindings": bindings,
        "summary_sha256": sha256_file(summary_path),
        "stream_manifest_sha256": manifest_hashes,
        "rolling_state_commitment": replay.rolling_commitment,
    }
    atomic_bytes(
        destination / "progress.json",
        canonical_json(
            {
                **_progress(
                    state,
                    total_source_rows=total_source_rows,
                    session_started=session_started,
                    session_source_start=session_source_start,
                ),
                "complete_percent": 100.0,
                "status": "complete",
                "resume_supported": False,
            }
        ),
    )
    atomic_bytes(completed_path, canonical_json(completion))
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
