#!/usr/bin/env python3
"""Stream/resume the frozen 2023 v2.2 unique-action calibration replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_managed_policy_calibration import (  # noqa: E402
    TRADE_COLUMNS,
    _atomic_bytes,
    _atomic_parquet,
    _aware,
    _canonical_json,
    _deadline,
    _progress,
    _publish_progress,
    _sha256,
    _structural_score,
    _verify_shards,
    _write_shard,
)
from smc_trader.action_equivalence import (  # noqa: E402
    ActionEquivalenceProtocol,
    EquivalentActionGroup,
    action_equivalence_code_fingerprint,
    action_plan_identity,
)
from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.calibration_replay import (  # noqa: E402
    CalibrationSequentialReplay,
    HASH_MODE,
    ReplayCheckpointStore,
    iter_after_source_checkpoint,
)
from smc_trader.io import load_ohlcv  # noqa: E402
from smc_trader.managed_net_value import (  # noqa: E402
    FEATURE_NAMES,
    build_v2_2_policy_base_engine,
    managed_action_features,
    managed_net_value_code_fingerprint,
)
from smc_trader.model import (  # noqa: E402
    Action,
    Playbook,
    PlaybookPhase,
    to_primitive,
)
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.simulation import SequentialPortfolio  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402


ATTEMPT_COLUMNS = [
    "thesis_hash",
    "action_key",
    "action_variant_hash",
    "decision_time",
    "representative_playbook",
    "direction",
    "setup_id",
    "entry_location_id",
    "entry_path_id",
    "evidence_hypothesis_keys",
    "evidence_playbooks",
    "evidence_state_json",
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
    *FEATURE_NAMES,
    "filled_at",
    "outcome",
]

POLICY_VARIANTS = {
    "all_three": (),
    "without_failed_auction_value_return": (
        Playbook.FAILED_AUCTION_VALUE_RETURN,
    ),
    "without_liquidity_sweep_reversal": (
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    ),
}

# This is the hash of the first completed v2.2 replay lineage. That lineage
# over-bound the replay to downstream episode/fitter source files. It remains
# acceptable for read-only migration because the completed artifact separately
# seals its runner/config/model/protocol hashes and every output digest.
LEGACY_CALIBRATION_PIPELINE_HASHES = frozenset(
    {"30c906737e7a575c797524036687d7a02e445470eb422d9d101cbba372354538"}
)


def calibration_pipeline_fingerprint() -> str:
    """Fingerprint only code/configuration that can change the costly replay."""

    digest = hashlib.sha256()
    for relative in (
        "smc_trader/action_equivalence.py",
        "smc_trader/managed_net_value.py",
        "smc_trader/calibration_replay.py",
        "scripts/run_managed_policy_calibration.py",
        "scripts/run_managed_net_calibration.py",
        "configs/action_equivalence_v2_2.json",
        "configs/managed_net_value_protocol_v2_2.json",
        "configs/validation_protocol_v2_2.json",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def calibration_pipeline_hash_is_accepted(value: str) -> bool:
    fingerprint = str(value)
    return (
        fingerprint == calibration_pipeline_fingerprint()
        or fingerprint in LEGACY_CALIBRATION_PIPELINE_HASHES
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
            "managed_net_calibration_zero_cost"
            if in_calibration_window
            else "constant_warmup_no_execution_authority"
        ),
    )


def _selected_group(
    snapshot: Any,
    protocol: ActionEquivalenceProtocol,
    *,
    disabled_playbooks: frozenset[Playbook] = frozenset(),
) -> EquivalentActionGroup:
    if not snapshot.decision.utilities:
        raise AssertionError("approved unique action has no compared utilities")
    representative = snapshot.decision.utilities[0]
    if representative.action is not Action.ENTER:
        raise AssertionError("approved unique action is not the best enter")
    key = representative.hypothesis_key
    hypothesis = snapshot.belief.hypotheses.get(key or "")
    if hypothesis is None or hypothesis.plan is None:
        raise AssertionError("approved unique action has no representative plan")
    identity = action_plan_identity(
        Action.ENTER,
        hypothesis.plan,
        tick_size=protocol.tick_size,
    )
    evidence: list[str] = []
    probabilities: list[float] = []
    for candidate_key, candidate in snapshot.belief.hypotheses.items():
        if (
            candidate.plan is None
            or candidate.phase is not PlaybookPhase.EXECUTABLE
            or candidate.playbook in disabled_playbooks
        ):
            continue
        candidate_identity = action_plan_identity(
            Action.ENTER,
            candidate.plan,
            tick_size=protocol.tick_size,
        )
        if candidate_identity == identity:
            evidence.append(candidate_key)
            probabilities.append(float(candidate.probability))
    if key not in evidence:
        raise AssertionError("representative hypothesis escaped its action group")
    return EquivalentActionGroup(
        identity=identity,
        representative=representative,
        hypothesis_keys=tuple(sorted(evidence)),
        probabilities=tuple(sorted(probabilities, reverse=True)),
    )


def _decision_row(
    snapshot: Any,
    protocol: ActionEquivalenceProtocol,
    *,
    disabled_playbooks: frozenset[Playbook] = frozenset(),
) -> dict[str, Any]:
    best = snapshot.decision.utilities[0] if snapshot.decision.utilities else None
    enter_groups: dict[str, set[str]] = {}
    for utility in snapshot.decision.utilities:
        if utility.action is not Action.ENTER or utility.hypothesis_key is None:
            continue
        hypothesis = snapshot.belief.hypotheses.get(utility.hypothesis_key)
        if hypothesis is None or hypothesis.plan is None:
            continue
        identity = action_plan_identity(
            Action.ENTER,
            hypothesis.plan,
            tick_size=protocol.tick_size,
        )
        same = {
            key
            for key, candidate in snapshot.belief.hypotheses.items()
            if candidate.plan is not None
            and candidate.phase is PlaybookPhase.EXECUTABLE
            and candidate.playbook not in disabled_playbooks
            and action_plan_identity(
                Action.ENTER,
                candidate.plan,
                tick_size=protocol.tick_size,
            )
            == identity
        }
        enter_groups[identity.key] = same
    plan = snapshot.decision.plan
    return {
        "asof": snapshot.observation.asof,
        "calibration_state_commitment": snapshot.snapshot_hash,
        "model_action": snapshot.decision.selected_action.value,
        "risk_action": snapshot.risk.final_action.value,
        "utility_advantage_R": snapshot.decision.advantage,
        "best_unique_action": None if best is None else best.action.value,
        "best_unique_action_utility_R": None if best is None else best.utility,
        "best_representative_hypothesis_key": (
            None if best is None else best.hypothesis_key
        ),
        "best_variant_components": json.dumps(
            {} if best is None else dict(best.components),
            sort_keys=True,
        ),
        "unique_enter_action_count": len(enter_groups),
        "maximum_equivalent_evidence_count": max(
            (len(keys) for keys in enter_groups.values()),
            default=0,
        ),
        "planned_entry": None if plan is None else plan.planned_entry,
        "invalidation": None if plan is None else plan.invalidation.price,
        "primary_target": None if plan is None else plan.targets[0].price,
        "execution_source": snapshot.observation.execution.source,
        "vetoes": json.dumps(
            [item.value for item in snapshot.risk.vetoes],
            sort_keys=True,
        ),
    }


def _approval_row(
    snapshot: Any,
    protocol: ActionEquivalenceProtocol,
    *,
    disabled_playbooks: frozenset[Playbook] = frozenset(),
) -> dict[str, Any]:
    thesis = snapshot.risk.frozen_thesis
    if thesis is None or snapshot.risk.final_action is not Action.ENTER:
        raise AssertionError("approval extraction requires a frozen entry")
    group = _selected_group(
        snapshot,
        protocol,
        disabled_playbooks=disabled_playbooks,
    )
    representative = snapshot.belief.hypotheses[
        group.representative.hypothesis_key or ""
    ]
    plan = representative.plan
    if plan is None:
        raise AssertionError("approved unique action has no trade plan")
    features = managed_action_features(
        group,
        snapshot.belief,
        snapshot.observation,
    )
    evidence_playbooks = sorted(
        {
            snapshot.belief.hypotheses[key].playbook.value
            for key in group.hypothesis_keys
        }
    )
    evidence_state = {
        key: {
            "playbook": snapshot.belief.hypotheses[key].playbook.value,
            "probability": float(
                snapshot.belief.hypotheses[key].probability
            ),
            "raw_probability": float(
                snapshot.belief.hypotheses[key].probability
                if snapshot.belief.hypotheses[key].raw_probability is None
                else snapshot.belief.hypotheses[key].raw_probability
            ),
            "phase": snapshot.belief.hypotheses[key].phase.value,
        }
        for key in group.hypothesis_keys
    }
    return {
        "thesis_hash": thesis.thesis_hash,
        "action_key": features.action_key,
        "action_variant_hash": features.action_key,
        "decision_time": snapshot.observation.asof,
        "representative_playbook": thesis.playbook.value,
        "direction": thesis.direction.value,
        "setup_id": thesis.setup_id,
        "entry_location_id": thesis.entry_location_id,
        "entry_path_id": thesis.entry_path_id,
        "evidence_hypothesis_keys": json.dumps(
            list(features.evidence_hypothesis_keys),
            sort_keys=True,
        ),
        "evidence_playbooks": json.dumps(evidence_playbooks),
        "evidence_state_json": json.dumps(
            evidence_state,
            sort_keys=True,
        ),
        "planned_entry": thesis.entry,
        "original_invalidation": thesis.original_invalidation.price,
        "invalidation_source_id": thesis.original_invalidation.source_level_id,
        "primary_target": thesis.original_targets[0].price,
        "primary_target_id": thesis.original_targets[0].level_id,
        "deadline": thesis.deadline,
        "decision_probability": representative.probability,
        "decision_raw_probability": representative.raw_probability,
        "decision_phase": representative.phase.value,
        "best_variant_utility_R": group.representative.utility,
        "path_structural_score_R": _structural_score(group.representative),
        "best_variant_components": json.dumps(
            dict(group.representative.components),
            sort_keys=True,
        ),
        **features.as_mapping(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/model_v2_2_policy_base.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2_2.json",
    )
    parser.add_argument(
        "--managed-net-value-protocol",
        default="configs/managed_net_value_protocol_v2_2.json",
    )
    parser.add_argument(
        "--action-equivalence-protocol",
        default="configs/action_equivalence_v2_2.json",
    )
    parser.add_argument("--start", default="2023-01-01T00:00:00-05:00")
    parser.add_argument("--end", default="2024-01-01T00:00:00-05:00")
    parser.add_argument("--warmup-days", type=int, default=45)
    parser.add_argument("--decision-shard-rows", type=int, default=25_000)
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
        or args.decision_shard_rows < 1
        or args.checkpoint_bars < 1
    ):
        raise ValueError("warmup, shard and checkpoint limits must be positive")
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
        raise RuntimeError("v2.2 calibration requires the complete 2023 window")
    source = Path(args.source)
    source_hash = _sha256(source)
    if source_hash != validation.causal_front_sha256:
        raise RuntimeError("OHLCV source differs from the preregistered front")
    config = Path(args.config)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    if (
        not str(config_payload.get("version", "")).startswith("2.2.")
        or config_payload.get("managed_net_value_artifact") is not None
    ):
        raise ValueError("calibration config must be the unfitted v2.2 base")
    action_protocol = ActionEquivalenceProtocol.from_file(
        args.action_equivalence_protocol
    )
    disabled_playbooks = frozenset(POLICY_VARIANTS[args.policy_variant])
    load_start = start - pd.Timedelta(days=args.warmup_days)
    loaded = load_ohlcv(source, start=load_start, end=end)
    if not loaded.contract_selection_causal:
        raise RuntimeError("v2.2 calibration requires causal roll selection")
    replay_frame = loaded.frame.loc[
        loaded.frame.index + pd.Timedelta(minutes=1) < end
    ]
    total_source_rows = int(len(replay_frame))
    if total_source_rows == 0:
        raise ValueError("v2.2 calibration source window is empty")

    destination = Path(args.output)
    completed_path = destination / "COMPLETED.json"
    progress_path = destination / "progress.json"
    if args.resume:
        if completed_path.exists():
            raise FileExistsError("v2.2 calibration replay is already complete")
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing non-resume use of non-empty output")
    destination.mkdir(parents=True, exist_ok=True)

    bindings = {
        "runner": "managed_net_calibration_stream_v1",
        "source_sha256": source_hash,
        "source_rows": total_source_rows,
        "source_first": replay_frame.index[0].isoformat(),
        "source_last": replay_frame.index[-1].isoformat(),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "warmup_days": int(args.warmup_days),
        "config_sha256": _sha256(config),
        "base_model_code_hash": model_code_fingerprint(),
        "action_equivalence_code_hash": (
            action_equivalence_code_fingerprint()
        ),
        "managed_net_value_code_hash": managed_net_value_code_fingerprint(),
        "calibration_pipeline_hash": calibration_pipeline_fingerprint(),
        "validation_protocol_hash": validation.fingerprint,
        "managed_net_value_protocol_hash": _sha256(
            args.managed_net_value_protocol
        ),
        "action_equivalence_protocol_hash": action_protocol.fingerprint,
        "policy_variant": args.policy_variant,
        "disabled_playbooks": sorted(
            playbook.value for playbook in disabled_playbooks
        ),
        "decision_shard_rows": int(args.decision_shard_rows),
        "checkpoint_bars": int(args.checkpoint_bars),
        "hash_mode": HASH_MODE,
    }
    checkpoint = ReplayCheckpointStore(destination / "_checkpoint")
    if args.resume:
        if not checkpoint.exists:
            raise FileNotFoundError("resume requested without a checkpoint")
        state = checkpoint.load(expected_bindings=bindings)
        if _verify_shards(
            destination,
            state["committed_shards"],
        ) != int(state["decision_rows"]):
            raise ValueError("checkpoint decision shards are not conserved")
        state["resume_count"] = int(state["resume_count"]) + 1
        checkpoint.save(state, bindings=bindings)
    else:
        replay = CalibrationSequentialReplay(
            engine=build_v2_2_policy_base_engine(
                config,
                disabled_playbooks=tuple(disabled_playbooks),
            ),
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
                buffer.append(
                    _decision_row(
                        snapshot,
                        action_protocol,
                        disabled_playbooks=disabled_playbooks,
                    )
                )
                if (
                    snapshot.risk.final_action is Action.ENTER
                    and snapshot.risk.frozen_thesis is not None
                ):
                    approval = _approval_row(
                        snapshot,
                        action_protocol,
                        disabled_playbooks=disabled_playbooks,
                    )
                    thesis_hash = approval["thesis_hash"]
                    if thesis_hash in state["entry_approvals"]:
                        raise AssertionError("duplicate risk-approved thesis")
                    state["entry_approvals"][thesis_hash] = approval
            due = (
                len(buffer) >= args.decision_shard_rows
                or int(state["processed_bars"])
                - int(state["last_checkpoint_processed_bars"])
                >= args.checkpoint_bars
            )
            safe_source_checkpoint = last_processed_was_source
            if last_processed_was_source and due:
                commit_checkpoint()
            if (
                args.diagnostic_stop_after_bars > 0
                and int(state["processed_bars"]) >= args.diagnostic_stop_after_bars
                and last_processed_was_source
            ):
                commit_checkpoint()
                raise RuntimeError(
                    "intentional v2.2 diagnostic interruption after checkpoint"
                )
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
        raise ValueError("v2.2 interval produced no completed decisions")
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
        outcome = (
            "filled"
            if filled_at is not None
            else (
                "pending_right_censored"
                if approval["decision_time"] == state["last_asof"]
                else "not_filled_or_expired_next_bar"
            )
        )
        attempts.append(
            {**approval, "filled_at": filled_at, "outcome": outcome}
        )
    attempts.sort(key=lambda row: (row["decision_time"], row["thesis_hash"]))
    trades_path = destination / "trades.parquet"
    attempts_path = destination / "entry_attempts.parquet"
    _atomic_parquet(pd.DataFrame(trade_rows, columns=TRADE_COLUMNS), trades_path)
    _atomic_parquet(pd.DataFrame(attempts, columns=ATTEMPT_COLUMNS), attempts_path)

    shard_rows = sum(int(item["rows"]) for item in state["committed_shards"])
    if shard_rows != int(state["decision_rows"]):
        raise AssertionError("v2.2 decision shards are not conserved")
    shard_manifest = {
        "format_version": 1,
        "artifact": "managed_net_calibration_decision_shards",
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
        "validation_protocol_hash": validation.fingerprint,
        "validation_window_role": window.role,
        "action_equivalence_protocol_hash": action_protocol.fingerprint,
        "policy_variant": args.policy_variant,
        "disabled_playbooks": sorted(
            playbook.value for playbook in disabled_playbooks
        ),
        "managed_net_value_protocol_hash": _sha256(
            args.managed_net_value_protocol
        ),
        "calibration_pipeline_hash": calibration_pipeline_fingerprint(),
        "config_hash": _sha256(config),
        "base_model_code_hash": model_code_fingerprint(),
        "managed_net_value_code_hash": managed_net_value_code_fingerprint(),
        "execution_reality_source": "managed_net_calibration_zero_cost",
        "future_path_loaded": False,
        "future_path_visible_to_model": False,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "decision_rows": int(state["decision_rows"]),
        "model_action_counts": dict(sorted(state["model_action_counts"].items())),
        "action_counts": dict(sorted(state["risk_action_counts"].items())),
        "approved_entry_attempts": len(attempts),
        "filled_entry_attempts": sum(
            row["outcome"] == "filled" for row in attempts
        ),
        "unfilled_or_expired_entry_attempts": sum(
            row["outcome"] == "not_filled_or_expired_next_bar"
            for row in attempts
        ),
        "pending_entry_attempts_at_end": sum(
            row["outcome"] == "pending_right_censored" for row in attempts
        ),
        "closed_trades": len(trade_rows),
        "profitability_evaluated": False,
        "snapshot_hash_mode": HASH_MODE,
        "full_snapshot_hash_per_minute": False,
        "decision_storage": "streamed_parquet_shards",
        "decision_shards": len(state["committed_shards"]),
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
        "artifact": "managed_net_calibration_replay",
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
