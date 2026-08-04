#!/usr/bin/env python3
"""Stream and resume the frozen 2023 managed-policy calibration replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.calibration_replay import (  # noqa: E402
    CalibrationSequentialReplay,
    HASH_MODE,
    ReplayCheckpointStore,
    iter_after_source_checkpoint,
)
from smc_trader.engine import ContinuousSMCEngine  # noqa: E402
from smc_trader.io import load_ohlcv  # noqa: E402
from smc_trader.model import Action, to_primitive  # noqa: E402
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.policy_value import (  # noqa: E402
    managed_policy_pipeline_fingerprint,
    path_structural_score,
)
from smc_trader.simulation import SequentialPortfolio  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402


TRADE_COLUMNS = [
    "thesis_hash",
    "playbook",
    "direction",
    "setup_id",
    "entry_location_id",
    "entry_path_id",
    "decision_time",
    "opened_at",
    "closed_at",
    "entry_price",
    "original_invalidation",
    "final_stop",
    "target",
    "exit_price",
    "exit_reason",
    "gross_R",
    "cost_R",
    "net_R",
    "ambiguous_same_bar",
]
ATTEMPT_COLUMNS = [
    "thesis_hash",
    "action_variant_hash",
    "decision_time",
    "playbook",
    "direction",
    "setup_id",
    "entry_location_id",
    "entry_path_id",
    "planned_entry",
    "original_invalidation",
    "invalidation_source_id",
    "primary_target",
    "primary_target_id",
    "deadline",
    "decision_probability",
    "decision_raw_probability",
    "decision_phase",
    "best_variant_utility_R",
    "path_structural_score_R",
    "best_variant_components",
    "filled_at",
    "outcome",
]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


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


def _execution(
    asof: pd.Timestamp,
    *,
    in_calibration_window: bool,
) -> ExecutionRealityInput:
    return ExecutionRealityInput(
        spread_points=0.0,
        expected_slippage_points=0.0,
        commission_per_contract_per_side=0.0,
        deadline=_deadline(asof),
        source=(
            "gross_policy_calibration_zero_cost"
            if in_calibration_window
            else "constant_warmup_no_execution_authority"
        ),
    )


def _structural_score(utility) -> float:
    if "path_structural_score_R" in utility.components:
        return float(utility.components["path_structural_score_R"])
    return float(path_structural_score(utility))


def _decision_row(snapshot) -> dict[str, Any]:
    ranked = snapshot.belief.ranked()
    top = ranked[0] if ranked else None
    best = snapshot.decision.utilities[0] if snapshot.decision.utilities else None
    components = {} if best is None else dict(best.components)
    score = (
        _structural_score(best)
        if best is not None and best.action is Action.ENTER
        else None
    )
    plan = snapshot.decision.plan
    return {
        "asof": snapshot.observation.asof,
        "calibration_state_commitment": snapshot.snapshot_hash,
        "model_action": snapshot.decision.selected_action.value,
        "risk_action": snapshot.risk.final_action.value,
        "utility_advantage_R": snapshot.decision.advantage,
        "best_variant_action": None if best is None else best.action.value,
        "best_variant_utility_R": None if best is None else best.utility,
        "best_variant_hypothesis_key": (
            None if best is None else best.hypothesis_key
        ),
        "best_variant_components": json.dumps(components, sort_keys=True),
        "path_structural_score_R": score,
        "top_playbook": None if top is None else top.playbook.value,
        "top_direction": None if top is None else top.direction.value,
        "top_probability": None if top is None else top.probability,
        "top_raw_probability": None if top is None else top.raw_probability,
        "top_phase": None if top is None else top.phase.value,
        "uncertainty": None if top is None else top.uncertainty,
        "planned_entry": None if plan is None else plan.planned_entry,
        "invalidation": None if plan is None else plan.invalidation.price,
        "primary_target": None if plan is None else plan.targets[0].price,
        "execution_source": snapshot.observation.execution.source,
        "vetoes": json.dumps([item.value for item in snapshot.risk.vetoes]),
    }


def _approval_row(snapshot) -> dict[str, Any]:
    thesis = snapshot.risk.frozen_thesis
    if thesis is None or snapshot.risk.final_action is not Action.ENTER:
        raise AssertionError("approval extraction requires a frozen approved entry")
    if not snapshot.decision.utilities:
        raise AssertionError("approved entry has no compared action variants")
    best = snapshot.decision.utilities[0]
    if best.action is not Action.ENTER:
        raise AssertionError("approved entry is not the best complete action variant")
    plan = snapshot.decision.plan
    if plan is None:
        raise AssertionError("approved entry has no trade plan")
    variant = {
        "verb": Action.ENTER.value,
        "playbook": thesis.playbook.value,
        "direction": thesis.direction.value,
        "setup_id": thesis.setup_id,
        "entry_location_id": thesis.entry_location_id,
        "entry_path_id": thesis.entry_path_id,
        "planned_entry": plan.planned_entry,
        "invalidation_source_id": plan.invalidation.source_level_id,
        "primary_target_id": plan.targets[0].level_id,
        "deadline": plan.deadline.isoformat(),
    }
    selected = snapshot.belief.hypotheses.get(best.hypothesis_key or "")
    return {
        "thesis_hash": thesis.thesis_hash,
        "action_variant_hash": hashlib.sha256(_canonical_json(variant)).hexdigest(),
        "decision_time": snapshot.observation.asof,
        "playbook": thesis.playbook.value,
        "direction": thesis.direction.value,
        "setup_id": thesis.setup_id,
        "entry_location_id": thesis.entry_location_id,
        "entry_path_id": thesis.entry_path_id,
        "planned_entry": thesis.entry,
        "original_invalidation": thesis.original_invalidation.price,
        "invalidation_source_id": thesis.original_invalidation.source_level_id,
        "primary_target": thesis.original_targets[0].price,
        "primary_target_id": thesis.original_targets[0].level_id,
        "deadline": thesis.deadline,
        "decision_probability": None if selected is None else selected.probability,
        "decision_raw_probability": (
            None if selected is None else selected.raw_probability
        ),
        "decision_phase": None if selected is None else selected.phase.value,
        "best_variant_utility_R": best.utility,
        "path_structural_score_R": _structural_score(best),
        "best_variant_components": json.dumps(
            dict(best.components),
            sort_keys=True,
        ),
    }


def _verify_shards(
    destination: Path,
    shards: list[dict[str, Any]],
) -> int:
    expected_index = 0
    total = 0
    for shard in shards:
        if int(shard["index"]) != expected_index:
            raise ValueError("checkpoint decision shard indices are not contiguous")
        path = destination / str(shard["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"committed decision shard is missing: {path}")
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"committed decision shard hash is invalid: {path}")
        rows = len(pd.read_parquet(path, columns=["asof"]))
        if rows != int(shard["rows"]):
            raise ValueError(f"committed decision shard row count is invalid: {path}")
        total += rows
        expected_index += 1
    return total


def _write_shard(
    destination: Path,
    buffer: list[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    if not buffer:
        return
    index = int(state["next_shard_index"])
    relative = Path("decision_shards") / f"part-{index:05d}.parquet"
    path = destination / relative
    frame = pd.DataFrame(buffer)
    _atomic_parquet(frame, path)
    state["committed_shards"].append(
        {
            "index": index,
            "path": str(relative),
            "rows": int(len(frame)),
            "first_asof": pd.Timestamp(frame["asof"].iloc[0]).isoformat(),
            "last_asof": pd.Timestamp(frame["asof"].iloc[-1]).isoformat(),
            "sha256": _sha256(path),
        }
    )
    state["next_shard_index"] = index + 1
    buffer.clear()


def _progress(
    state: dict[str, Any],
    *,
    total_source_rows: int,
    session_started: float,
    session_source_start: int,
) -> dict[str, Any]:
    consumed = int(state["source_rows_consumed"])
    percentage = 100.0 * consumed / max(total_source_rows, 1)
    session_rows = consumed - session_source_start
    elapsed = max(time.monotonic() - session_started, 1e-9)
    rows_per_second = session_rows / elapsed
    eta_seconds = (
        None
        if rows_per_second <= 0
        else (total_source_rows - consumed) / rows_per_second
    )
    return {
        "source_rows": consumed,
        "source_rows_total": total_source_rows,
        "complete_percent": round(percentage, 3),
        "decision_rows": int(state["decision_rows"]),
        "decision_shards": len(state["committed_shards"]),
        "rows_per_second_this_process": round(rows_per_second, 2),
        "eta_seconds_this_process_rate": (
            None if eta_seconds is None else round(eta_seconds, 1)
        ),
        "last_asof": (
            None
            if state["last_asof"] is None
            else pd.Timestamp(state["last_asof"]).isoformat()
        ),
    }


def _publish_progress(
    path: Path, progress: dict[str, Any],
    status: str, resume_supported: bool,
    exc: BaseException | None = None,
) -> None:
    payload = {**progress, "status": status, "resume_supported": resume_supported}
    if exc is not None:
        payload.update(failure_type=type(exc).__name__, failure_message=str(exc),
                       durable_checkpoint_only=True)
    _atomic_bytes(path, _canonical_json(payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/model_v2_1_policy_base.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2_1.json",
    )
    parser.add_argument(
        "--policy-value-protocol",
        default="configs/policy_value_protocol_v2_1.json",
    )
    parser.add_argument("--start", default="2023-01-01T00:00:00-05:00")
    parser.add_argument("--end", default="2024-01-01T00:00:00-05:00")
    parser.add_argument("--warmup-days", type=int, default=45)
    parser.add_argument("--decision-shard-rows", type=int, default=25_000)
    parser.add_argument("--checkpoint-bars", type=int, default=25_000)
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
    if args.warmup_days < 1:
        raise ValueError("managed-policy calibration requires positive warmup days")
    if args.decision_shard_rows < 1 or args.checkpoint_bars < 1:
        raise ValueError("shard and checkpoint row limits must be positive")

    start = _aware(args.start)
    end = _aware(args.end)
    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_ohlcv(start, end)
    registered = validation.ohlcv_windows["managed_policy_calibration"]
    if (
        window.role != "managed_policy_calibration"
        or start != registered.start
        or end != registered.end_exclusive
    ):
        raise RuntimeError(
            "managed-policy calibration must use the complete registered 2023 window"
        )

    source = Path(args.source)
    source_hash = _sha256(source)
    if source_hash != validation.causal_front_sha256:
        raise RuntimeError("OHLCV source hash differs from the preregistered causal front")
    config = Path(args.config)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    if (
        not str(config_payload.get("version", "")).startswith("2.1.")
        or config_payload.get("managed_policy_artifact") is not None
    ):
        raise ValueError("calibration config must be the unfitted v2.1 policy base")
    config_hash = _sha256(config)
    policy_protocol_hash = _sha256(args.policy_value_protocol)
    pipeline_hash = managed_policy_pipeline_fingerprint()
    load_start = start - pd.Timedelta(days=args.warmup_days)
    loaded = load_ohlcv(source, start=load_start, end=end)
    if not loaded.contract_selection_causal:
        raise RuntimeError("managed-policy calibration requires causal contract selection")
    replay_frame = loaded.frame.loc[
        loaded.frame.index + pd.Timedelta(minutes=1) < end
    ]
    total_source_rows = int(len(replay_frame))
    if total_source_rows == 0:
        raise ValueError("managed-policy calibration source window is empty")

    destination = Path(args.output)
    completed_path = destination / "COMPLETED.json"
    progress_path = destination / "progress.json"
    if args.resume:
        if completed_path.exists():
            raise FileExistsError(f"calibration replay is already complete: {destination}")
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing non-resume use of non-empty output: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    bindings = {
        "runner": "managed_policy_calibration_stream_v1",
        "source_sha256": source_hash,
        "source_rows": total_source_rows,
        "source_first": replay_frame.index[0].isoformat(),
        "source_last": replay_frame.index[-1].isoformat(),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "warmup_days": int(args.warmup_days),
        "config_sha256": config_hash,
        "model_code_hash": model_code_fingerprint(),
        "managed_policy_pipeline_hash": pipeline_hash,
        "validation_protocol_hash": validation.fingerprint,
        "policy_value_protocol_hash": policy_protocol_hash,
        "decision_shard_rows": int(args.decision_shard_rows),
        "checkpoint_bars": int(args.checkpoint_bars),
        "hash_mode": HASH_MODE,
    }
    checkpoint = ReplayCheckpointStore(destination / "_checkpoint")
    if args.resume:
        if not checkpoint.exists:
            raise FileNotFoundError("resume requested but no complete checkpoint exists")
        state = checkpoint.load(expected_bindings=bindings)
        committed_rows = _verify_shards(
            destination,
            state["committed_shards"],
        )
        if committed_rows != int(state["decision_rows"]):
            raise ValueError(
                "checkpoint decisions are not conserved by committed shards"
            )
        state["resume_count"] = int(state["resume_count"]) + 1
        checkpoint.save(state, bindings=bindings)
    else:
        replay = CalibrationSequentialReplay(
            engine=ContinuousSMCEngine.from_config(config),
            portfolio=SequentialPortfolio(),
        )
        state = {
            "replay": replay,
            "processed_bars": 0,
            "source_rows_consumed": 0,
            "last_checkpoint_processed_bars": 0,
            "decision_rows": 0,
            "model_action_counts": {},
            "risk_action_counts": {},
            "entry_approvals": {},
            "filled_entries": {},
            "last_source_start": None,
            "last_asof": None,
            "next_shard_index": 0,
            "committed_shards": [],
            "resume_count": 0,
        }
    replay = state["replay"]
    buffer: list[dict[str, Any]] = []
    session_started = time.monotonic()
    session_source_start = int(state["source_rows_consumed"])
    durable_progress = _progress(
        state, total_source_rows=total_source_rows,
        session_started=session_started, session_source_start=session_source_start
    )

    def commit_checkpoint() -> None:
        nonlocal durable_progress, safe_source_checkpoint
        safe_source_checkpoint = False
        if state["last_source_start"] is None:
            return
        _write_shard(destination, buffer, state)
        state["last_checkpoint_processed_bars"] = int(state["processed_bars"])
        checkpoint.save(state, bindings=bindings)
        durable_progress = _progress(
            state, total_source_rows=total_source_rows,
            session_started=session_started, session_source_start=session_source_start
        )
        _publish_progress(progress_path, durable_progress, "running", True)
        print(
            json.dumps(
                _progress(
                    state,
                    total_source_rows=total_source_rows,
                    session_started=session_started,
                    session_source_start=session_source_start,
                ),
                sort_keys=True,
            ),
            flush=True,
        )

    iterator = iter_after_source_checkpoint(
        replay_frame,
        state["last_source_start"],
    )
    last_processed_was_source = False
    safe_source_checkpoint = False
    try:
        for bar in iterator:
            safe_source_checkpoint = False
            if bar.end >= end:
                break
            in_window = bar.end >= start
            step = replay.on_bar(
                bar,
                execution=_execution(
                    bar.end,
                    in_calibration_window=in_window,
                ),
            )
            snapshot = step.snapshot
            state["processed_bars"] = int(state["processed_bars"]) + 1
            last_processed_was_source = not bar.synthetic_no_trade
            if not bar.synthetic_no_trade:
                state["source_rows_consumed"] = (
                    int(state["source_rows_consumed"]) + 1
                )
                state["last_source_start"] = bar.start

            for record in step.closed_trades:
                state["filled_entries"].setdefault(
                    record.thesis_hash,
                    record.opened_at,
                )
            if step.position is not None:
                state["filled_entries"].setdefault(
                    step.position.thesis_hash,
                    step.position.opened_at,
                )

            if snapshot.observation.asof >= start:
                state["last_asof"] = snapshot.observation.asof
                state["decision_rows"] = int(state["decision_rows"]) + 1
                model_action = snapshot.decision.selected_action.value
                risk_action = snapshot.risk.final_action.value
                state["model_action_counts"][model_action] = (
                    int(state["model_action_counts"].get(model_action, 0)) + 1
                )
                state["risk_action_counts"][risk_action] = (
                    int(state["risk_action_counts"].get(risk_action, 0)) + 1
                )
                buffer.append(_decision_row(snapshot))
                if (
                    snapshot.risk.final_action is Action.ENTER
                    and snapshot.risk.frozen_thesis is not None
                ):
                    approval = _approval_row(snapshot)
                    thesis_hash = approval["thesis_hash"]
                    if thesis_hash in state["entry_approvals"]:
                        raise AssertionError(
                            "risk-approved thesis hash was emitted more than once"
                        )
                    state["entry_approvals"][thesis_hash] = approval

            can_checkpoint = not bar.synthetic_no_trade
            safe_source_checkpoint = last_processed_was_source
            due = (
                len(buffer) >= args.decision_shard_rows
                or int(state["processed_bars"])
                - int(state["last_checkpoint_processed_bars"])
                >= args.checkpoint_bars
            )
            if can_checkpoint and due:
                commit_checkpoint()
            if (
                args.diagnostic_stop_after_bars > 0
                and int(state["processed_bars"]) >= args.diagnostic_stop_after_bars
                and can_checkpoint
            ):
                commit_checkpoint()
                raise RuntimeError("intentional diagnostic interruption after checkpoint")
    except KeyboardInterrupt as exc:
        if safe_source_checkpoint:
            commit_checkpoint()
        _publish_progress(progress_path, durable_progress, "failed",
                          checkpoint.exists, exc)
        raise
    except Exception as exc:
        _publish_progress(progress_path, durable_progress, "failed",
                          checkpoint.exists, exc)
        raise
    if state["last_asof"] is None or int(state["decision_rows"]) == 0:
        raise ValueError("registered interval produced no completed decisions")
    if int(state["source_rows_consumed"]) != total_source_rows:
        raise RuntimeError("source iterator ended before all bound rows were consumed")
    commit_checkpoint()

    trades = [
        record
        for record in replay.portfolio.records
        if record.decision_time >= start
    ]
    trade_rows = [to_primitive(record) for record in trades]
    attempts: list[dict[str, Any]] = []
    for thesis_hash, approval in state["entry_approvals"].items():
        filled_at = state["filled_entries"].get(thesis_hash)
        if filled_at is not None:
            outcome = "filled"
        elif approval["decision_time"] == state["last_asof"]:
            outcome = "pending_right_censored"
        else:
            outcome = "not_filled_or_expired_next_bar"
        attempts.append(
            {
                **approval,
                "filled_at": filled_at,
                "outcome": outcome,
            }
        )
    attempts.sort(key=lambda row: (row["decision_time"], row["thesis_hash"]))

    trades_path = destination / "trades.parquet"
    attempts_path = destination / "entry_attempts.parquet"
    _atomic_parquet(pd.DataFrame(trade_rows, columns=TRADE_COLUMNS), trades_path)
    _atomic_parquet(pd.DataFrame(attempts, columns=ATTEMPT_COLUMNS), attempts_path)

    shard_rows = sum(int(item["rows"]) for item in state["committed_shards"])
    if shard_rows != int(state["decision_rows"]):
        raise AssertionError("decision shard rows do not conserve replay decisions")
    shard_manifest = {
        "format_version": 1,
        "artifact": "managed_policy_calibration_decision_shards",
        "status": "complete",
        "rows": shard_rows,
        "shards": state["committed_shards"],
        "hash_mode": HASH_MODE,
        "full_snapshot_hash_per_minute": False,
        "bindings": bindings,
    }
    shard_manifest_path = destination / "decision_shards.manifest.json"
    _atomic_bytes(shard_manifest_path, _canonical_json(shard_manifest))

    summary = {
        "version": str(config_payload.get("version", "")),
        "source": str(loaded.source),
        "source_sha256": source_hash,
        "source_matches_preregistered_causal_front": True,
        "source_role": loaded.source_role,
        "validation_protocol_version": validation.version,
        "validation_protocol_hash": validation.fingerprint,
        "validation_window_role": window.role,
        "policy_value_protocol_hash": policy_protocol_hash,
        "managed_policy_pipeline_hash": pipeline_hash,
        "config_hash": config_hash,
        "model_code_hash": model_code_fingerprint(),
        "execution_reality_source": "gross_policy_calibration_zero_cost",
        "gross_policy_calibration": True,
        "execution_authority": False,
        "contract_selection_causal": loaded.contract_selection_causal,
        "warnings": list(loaded.warnings),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "decision_clock_interval": "[start, end)",
        "decision_rows": int(state["decision_rows"]),
        "model_action_counts": dict(sorted(state["model_action_counts"].items())),
        "action_counts": dict(sorted(state["risk_action_counts"].items())),
        "approved_entry_attempts": len(attempts),
        "filled_entry_attempts": sum(row["outcome"] == "filled" for row in attempts),
        "unfilled_or_expired_entry_attempts": sum(
            row["outcome"] == "not_filled_or_expired_next_bar"
            for row in attempts
        ),
        "pending_entry_attempts_at_end": sum(
            row["outcome"] == "pending_right_censored" for row in attempts
        ),
        "future_path_loaded": False,
        "future_path_visible_to_model": False,
        "sequential_execution_evaluated": True,
        "profitability_evaluated": False,
        "closed_trades": len(trade_rows),
        "net_R": float(sum(float(row["net_R"]) for row in trade_rows)),
        "calibration_runner": "managed_policy_calibration_stream_v1",
        "causal_operation_order": (
            "portfolio.before_bar -> eyes -> brain -> decision -> risk -> "
            "portfolio.after_decision"
        ),
        "snapshot_hash_mode": HASH_MODE,
        "full_snapshot_hash_per_minute": False,
        "decision_storage": "streamed_parquet_shards",
        "decision_shards": len(state["committed_shards"]),
        "decision_shards_manifest": str(shard_manifest_path),
        "checkpoint_format_version": 1,
        "checkpoint_resume_supported": True,
        "checkpoint_interval_bars": int(args.checkpoint_bars),
        "decision_shard_rows_limit": int(args.decision_shard_rows),
        "resume_count": int(state["resume_count"]),
        "source_rows_processed": int(state["source_rows_consumed"]),
        "source_rows_total": total_source_rows,
        "completed_percent": 100.0,
    }
    summary_path = destination / "summary.json"
    _atomic_bytes(summary_path, _canonical_json(to_primitive(summary)))
    completion = {
        "format_version": 1,
        "artifact": "managed_policy_calibration_replay",
        "status": "complete",
        "bindings": bindings,
        "summary_sha256": _sha256(summary_path),
        "entry_attempts_sha256": _sha256(attempts_path),
        "trades_sha256": _sha256(trades_path),
        "decision_shards_manifest_sha256": _sha256(shard_manifest_path),
        "rolling_state_commitment": replay.rolling_commitment,
    }
    _publish_progress(progress_path, {**durable_progress, "complete_percent": 100.0},
                      "complete", False)
    _atomic_bytes(completed_path, _canonical_json(completion))
    print(json.dumps(summary["action_counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
