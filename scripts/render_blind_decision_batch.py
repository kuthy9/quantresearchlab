#!/usr/bin/env python3
"""Generate sampled causal traces, packets and blind images in one replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_decision_audit_batch import (  # noqa: E402
    read_verified_case,
)
from scripts.run_continuous_replay import _deadline  # noqa: E402
from smc_trader.artifact_stream import (  # noqa: E402
    atomic_bytes,
    canonical_json,
    sha256_file,
)
from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.calibration_replay import (  # noqa: E402
    HASH_MODE,
    CalibrationSequentialReplay,
)
from smc_trader.decision_trace import (  # noqa: E402
    TRACE_SCHEMA_VERSION,
    active_causal_timeframes,
    build_decision_trace,
    decision_packet_sha256,
    read_verified_decision_packet,
    validate_causal_histories,
    write_frozen_decision_packet,
)
from smc_trader.engine import ContinuousSMCEngine  # noqa: E402
from smc_trader.io import (  # noqa: E402
    iter_completed_bars,
    load_ohlcv,
    require_materialized,
)
from smc_trader.mbo import (  # noqa: E402
    MinuteExecutionRealityStore,
    assert_mbo_source_allowed,
)
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.visualization import DecisionVisualizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mbo-execution",
        help="bound causal minute execution-reality parquet for an MBO replay",
    )
    return parser.parse_args()


def _read_json(source: Path, label: str) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return result


def _load_bound_mbo_execution(
    path: str | Path,
    bindings: Mapping[str, Any],
) -> MinuteExecutionRealityStore:
    """Load only the exact MBO artifact used by the lightweight replay."""

    source = assert_mbo_source_allowed(path)
    require_materialized(source)
    expected_source_sha256 = bindings.get("mbo_execution_sha256")
    if not isinstance(expected_source_sha256, str):
        raise ValueError("lightweight replay omits its MBO parquet hash")
    actual_source_sha256 = sha256_file(source)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError("MBO execution parquet differs from the lightweight replay")

    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    require_materialized(manifest_path)
    expected_manifest_sha256 = bindings.get(
        "mbo_execution_manifest_sha256"
    )
    if not isinstance(expected_manifest_sha256, str):
        raise ValueError("lightweight replay omits its MBO manifest hash")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("MBO execution manifest differs from the lightweight replay")
    manifest = _read_json(manifest_path, "MBO execution manifest")
    if int(manifest.get("format_version", 0)) != 1:
        raise ValueError("MBO execution manifest format is unsupported")
    if manifest.get("output_sha256") != actual_source_sha256:
        raise ValueError("MBO execution manifest output hash is invalid")
    validation_hash = bindings.get("validation_protocol_hash")
    if (
        not isinstance(validation_hash, str)
        or manifest.get("validation_protocol_hash") != validation_hash
    ):
        raise ValueError("MBO execution manifest validation binding is invalid")
    return MinuteExecutionRealityStore.from_parquet(source)


def _active_targets(
    targets: list[dict[str, Any]],
    ordinal: int,
) -> list[dict[str, Any]]:
    return [
        target
        for target in targets
        if target["trace_start_ordinal"] <= ordinal <= target["anchor_ordinal"]
    ]


def _capture_sampled_trace(
    *,
    ordinal: int,
    targets: list[dict[str, Any]],
    snapshot: Any,
    previous_snapshot: Any | None,
    source_bar: Any | None,
    account_state: Any | None,
    belief_position_input: Any | None,
) -> dict[str, Any] | None:
    """Build one trace only when the clock belongs to a sampled interval."""

    active = _active_targets(targets, ordinal)
    if not active:
        return None
    trace = build_decision_trace(
        snapshot,
        previous_snapshot,
        source_bar=source_bar,
        account_state=account_state,
        belief_position_input=belief_position_input,
    )
    serialized = json.dumps(trace, sort_keys=True, ensure_ascii=False)
    record = {
        "ordinal": ordinal,
        "asof": snapshot.observation.asof.isoformat(),
        "snapshot_hash": snapshot.snapshot_hash,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "decision_trace_sha256": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
        "decision_trace": trace,
    }
    for target in active:
        target["trace_records"].append(record)
    return trace


def _validate_anchor_identity(
    target: Mapping[str, Any],
    snapshot: Any,
    belief_position_input: Any | None,
) -> None:
    if (
        snapshot.observation.asof != target["asof"]
        or snapshot.snapshot_hash != target["snapshot_hash"]
        or snapshot.decision.selected_action.value != target["model_action"]
        or snapshot.risk.final_action.value != target["risk_action"]
        or snapshot.decision.best_hypothesis_key != target["hypothesis_key"]
    ):
        raise ValueError(
            "replayed anchor action or rolling commitment differs for case "
            f"{target['case_id']}"
        )

    hypothesis_key = target["hypothesis_key"]
    scope = target["setup_scope"]
    if hypothesis_key is not None:
        hypothesis = snapshot.belief.hypotheses.get(hypothesis_key)
        sequence = None if hypothesis is None else hypothesis.sequence
        replayed_setup = None if sequence is None else sequence.setup_id
        if (
            hypothesis is None
            or hypothesis.playbook.value != target["playbook"]
            or hypothesis.direction.value != target["direction"]
            or hypothesis.phase.value != target["phase"]
            or replayed_setup != target["setup_id"]
            or scope not in {"setup_bound", "hypothesis_bound"}
        ):
            raise ValueError(
                f"replayed hypothesis identity differs for case {target['case_id']}"
            )
    elif scope == "position_bound":
        position = belief_position_input
        if (
            position is None
            or position.thesis_hash != target["position_thesis_hash"]
            or position.setup_id != target["setup_id"]
            or position.playbook.value != target["playbook"]
            or position.direction.value != target["direction"]
        ):
            raise ValueError(
                f"replayed position identity differs for case {target['case_id']}"
            )
    elif scope != "unbound" or target["setup_id"] is not None:
        raise ValueError("unselected hypothesis anchor has an invalid setup binding")


def _write_sampled_trajectory(
    *,
    target: Mapping[str, Any],
    destination: Path,
    batch_manifest_sha256: str,
    replay_completed_sha256: str,
) -> Path:
    records = list(target["trace_records"])
    expected_rows = (
        target["anchor_ordinal"] - target["trace_start_ordinal"] + 1
    )
    ordinals = [int(item["ordinal"]) for item in records]
    if (
        len(records) != expected_rows
        or ordinals
        != list(
            range(
                target["trace_start_ordinal"],
                target["anchor_ordinal"] + 1,
            )
        )
        or records[-1]["snapshot_hash"] != target["snapshot_hash"]
    ):
        raise RuntimeError("sampled case trajectory is not contiguous to its anchor")
    payload = {
        "format_version": 1,
        "artifact": "sampled_causal_decision_trajectory",
        "status": "complete",
        "case_id": target["case_id"],
        "future_path_included": False,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "start_ordinal": target["trace_start_ordinal"],
        "end_ordinal": target["anchor_ordinal"],
        "maximum_market_time": target["asof"].isoformat(),
        "rows": len(records),
        "records": records,
        "source_binding": {
            "batch_manifest_sha256": batch_manifest_sha256,
            "replay_completed_sha256": replay_completed_sha256,
            "anchor_snapshot_hash": target["snapshot_hash"],
        },
    }
    path = destination / target["trajectory_filename"]
    atomic_bytes(path, canonical_json(payload))
    return path


def main() -> None:
    args = parse_args()
    batch_root = Path(args.batch).resolve()
    batch_manifest_path = batch_root / "decision_audit_batch.manifest.json"
    batch_manifest = _read_json(batch_manifest_path, "batch manifest")
    if (
        batch_manifest.get("format_version") != 2
        or batch_manifest.get("status") != "anchors_selected"
        or batch_manifest.get("blind_first_pass") is not True
        or batch_manifest.get("future_path_included") is not False
    ):
        raise ValueError("decision batch is not a lightweight anchor batch")
    cases = batch_manifest.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or int(batch_manifest.get("case_count", -1)) != len(cases)
    ):
        raise ValueError("decision batch case inventory is invalid")
    batch_manifest_sha256 = sha256_file(batch_manifest_path)

    source_binding = _mapping(batch_manifest.get("source"), "batch source")
    replay_root = Path(str(source_binding.get("replay_root", ""))).resolve()
    completed_path = replay_root / "COMPLETED.json"
    completed = _read_json(completed_path, "replay COMPLETED marker")
    replay_completed_sha256 = sha256_file(completed_path)
    if (
        completed.get("status") != "complete"
        or replay_completed_sha256 != source_binding.get("completed_sha256")
    ):
        raise ValueError("batch replay completion binding is invalid")
    bindings = _mapping(completed.get("bindings"), "replay bindings")
    execution_mode = bindings.get("execution_mode")
    simulate_execution = bool(bindings.get("simulate_execution"))
    if (
        bindings.get("hash_mode") != HASH_MODE
        or bindings.get("include_decision_traces") is not False
        or source_binding.get("include_decision_traces") is not False
        or source_binding.get("runner") != bindings.get("runner")
    ):
        raise ValueError("sampled renderer requires a bound lightweight replay")
    if execution_mode == "mbo_causal_execution":
        if args.mbo_execution is None:
            raise ValueError("MBO lightweight replay requires --mbo-execution")
        execution_store = _load_bound_mbo_execution(
            args.mbo_execution,
            bindings,
        )
    elif execution_mode in {
        "ohlcv_only_execution_unavailable",
        "constant_cli_research_only",
    }:
        if args.mbo_execution is not None:
            raise ValueError("non-MBO lightweight replay rejects --mbo-execution")
        if (
            bindings.get("mbo_execution_sha256") is not None
            or bindings.get("mbo_execution_manifest_sha256") is not None
        ):
            raise ValueError("non-MBO replay contains unexpected MBO bindings")
        execution_store = None
    else:
        raise ValueError("lightweight replay has an unsupported execution mode")

    source = Path(args.source)
    config = Path(args.config)
    if sha256_file(source) != bindings.get("source_sha256"):
        raise ValueError("OHLCV source differs from the lightweight replay")
    if sha256_file(config) != bindings.get("config_sha256"):
        raise ValueError("model config differs from the lightweight replay")
    if model_code_fingerprint() != bindings.get("model_code_hash"):
        raise ValueError("model code differs from the lightweight replay")
    bound_files = {
        replay_root / "decision_shards.manifest.json": source_binding.get(
            "decision_manifest_sha256"
        ),
        ROOT / "scripts" / "run_continuous_replay.py": bindings.get(
            "runner_sha256"
        ),
        ROOT / "smc_trader" / "calibration_replay.py": bindings.get(
            "calibration_replay_sha256"
        ),
        ROOT / "smc_trader" / "artifact_stream.py": bindings.get(
            "artifact_stream_sha256"
        ),
    }
    for bound_path, expected_sha256 in bound_files.items():
        if (
            not isinstance(expected_sha256, str)
            or sha256_file(bound_path) != expected_sha256
        ):
            raise ValueError(
                f"lightweight replay dependency differs: {bound_path.name}"
            )

    targets: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for order, raw_case in enumerate(cases, start=1):
        case_row = _mapping(raw_case, "batch case")
        case_id = str(case_row.get("case_id", ""))
        relative = Path(str(case_row.get("file", "")))
        if (
            not case_id
            or case_id in seen_case_ids
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError("batch case identity or path is invalid")
        seen_case_ids.add(case_id)
        case_path = (batch_root / relative).resolve()
        case_path.relative_to(batch_root)
        if sha256_file(case_path) != case_row.get("sha256"):
            raise ValueError("batch case file differs from its manifest")
        verified = read_verified_case(case_path)
        anchor = _mapping(verified.get("anchor"), "case anchor")
        request = _mapping(
            verified.get("sampled_trajectory_request"),
            "sampled trajectory request",
        )
        asof = _timestamp(anchor.get("asof"), "case anchor")
        anchor_ordinal = int(anchor.get("ordinal", -1))
        trace_start_ordinal = int(request.get("start_ordinal", -1))
        snapshot_hash = str(anchor.get("snapshot_hash", ""))
        if (
            anchor_ordinal != int(case_row.get("anchor_ordinal", -2))
            or asof.isoformat() != case_row.get("anchor_asof")
            or snapshot_hash != case_row.get("anchor_snapshot_hash")
            or trace_start_ordinal
            != int(case_row.get("trace_start_ordinal", -2))
        ):
            raise ValueError("case anchor differs from its batch binding")
        stem = f"{order:02d}-{case_id}-{snapshot_hash[:16]}"
        targets.append(
            {
                "order": order,
                "case_id": case_id,
                "case_file": str(relative),
                "case_sha256": case_row["sha256"],
                "asof": asof,
                "anchor_ordinal": anchor_ordinal,
                "trace_start_ordinal": trace_start_ordinal,
                "snapshot_hash": snapshot_hash,
                "hypothesis_key": anchor.get("hypothesis_key"),
                "setup_id": anchor.get("setup_id"),
                "phase": anchor.get("phase"),
                "model_action": anchor.get("model_action"),
                "risk_action": anchor.get("risk_action"),
                "playbook": anchor.get("playbook"),
                "direction": anchor.get("direction"),
                "setup_scope": anchor.get("setup_scope"),
                "position_thesis_hash": anchor.get("position_thesis_hash"),
                "trace_records": [],
                "trajectory_filename": str(
                    Path("sampled_trajectories") / f"{stem}.json"
                ),
                "image_filename": f"{stem}.png",
                "packet_filename": f"{stem}.json",
            }
        )
    target_ordinals = [item["anchor_ordinal"] for item in targets]
    target_clocks = [item["asof"] for item in targets]
    if (
        target_ordinals != sorted(target_ordinals)
        or len(set(target_ordinals)) != len(target_ordinals)
        or target_clocks != sorted(target_clocks)
        or len(set(target_clocks)) != len(target_clocks)
    ):
        raise ValueError("blind review anchors must be unique and ordered")

    start = _timestamp(bindings.get("start"), "replay start")
    end = _timestamp(bindings.get("end_exclusive"), "replay end")
    warmup_days = int(bindings.get("warmup_days", -1))
    if (
        warmup_days < 0
        or min(target_clocks) < start
        or max(target_clocks) >= end
    ):
        raise ValueError("case anchors fall outside the lightweight replay")
    replay_end = max(target_clocks) + pd.Timedelta(minutes=1)
    loaded = load_ohlcv(
        source,
        start=start - pd.Timedelta(days=warmup_days),
        end=replay_end,
    )
    if not loaded.contract_selection_causal:
        raise ValueError("blind image replay requires causal contract selection")

    destination = Path(args.output)
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError("blind image output is not a directory")
        if any(destination.iterdir()):
            raise FileExistsError("blind image output must be empty")
    image_root = destination / "decision_images"
    packet_root = destination / "decision_packets"
    trajectory_root = destination / "sampled_trajectories"
    image_root.mkdir(parents=True, exist_ok=True)
    packet_root.mkdir(parents=True, exist_ok=True)
    trajectory_root.mkdir(parents=True, exist_ok=True)

    targets_by_ordinal = {
        target["anchor_ordinal"]: target for target in targets
    }
    replay = CalibrationSequentialReplay(
        engine=ContinuousSMCEngine.from_config(config),
        simulate_execution=simulate_execution,
    )
    visualizer = DecisionVisualizer()
    rendered: list[dict[str, Any]] = []
    sampled_ordinals: set[int] = set()
    processed_bars = 0
    decision_ordinal = -1
    for bar in iter_completed_bars(loaded.frame):
        if bar.end >= replay_end:
            break
        processed_bars += 1
        previous_snapshot = replay.engine.last_snapshot
        if execution_store is not None and bar.end >= start:
            execution = execution_store.for_bar(
                bar,
                deadline=_deadline(bar.end),
            )
        elif execution_mode == "constant_cli_research_only":
            execution = ExecutionRealityInput(
                spread_points=float(bindings["constant_spread_points"]),
                expected_slippage_points=float(
                    bindings["constant_slippage_points"]
                ),
                deadline=_deadline(bar.end),
                source="constant_cli_research_only",
            )
        else:
            execution = ExecutionRealityInput(
                spread_points=None,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                deadline=_deadline(bar.end),
                source="ohlcv_only_execution_unavailable",
            )
        step = replay.on_bar(
            bar,
            execution=execution,
        )
        snapshot = step.snapshot
        if snapshot.observation.asof < start:
            continue
        decision_ordinal += 1
        trace = _capture_sampled_trace(
            ordinal=decision_ordinal,
            targets=targets,
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
            source_bar=bar,
            account_state=step.account_state,
            belief_position_input=step.belief_position_input,
        )
        if trace is not None:
            sampled_ordinals.add(decision_ordinal)

        target = targets_by_ordinal.get(decision_ordinal)
        if target is None:
            continue
        _validate_anchor_identity(
            target,
            snapshot,
            step.belief_position_input,
        )
        if trace is None:
            raise AssertionError("anchor is outside its sampled trace interval")
        histories = validate_causal_histories(
            snapshot,
            replay.engine.histories(80),
        )
        history_summary = {
            timeframe.value: {
                "rows": len(histories[timeframe]),
                "first": histories[timeframe][0].start.isoformat(),
                "last": histories[timeframe][-1].end.isoformat(),
                "all_complete": all(
                    candle.complete for candle in histories[timeframe]
                ),
                "maximum_market_time": max(
                    candle.end for candle in histories[timeframe]
                ).isoformat(),
            }
            for timeframe in active_causal_timeframes(
                snapshot.observation
            )
        }
        packet_path = write_frozen_decision_packet(
            snapshot,
            histories,
            packet_root / target["packet_filename"],
            previous_snapshot,
            hypothesis_key=target["hypothesis_key"],
            audit_context={
                "scenario": "sampled_decision_batch_anchor",
                "case_id": target["case_id"],
                "batch_manifest_sha256": batch_manifest_sha256,
                "future_path_included": False,
            },
            source_bar=bar,
            account_state=step.account_state,
            belief_position_input=step.belief_position_input,
        )
        packet = read_verified_decision_packet(packet_path)
        if (
            packet.get("maximum_market_time") != target["asof"].isoformat()
            or packet.get("decision_trace") != trace
            or packet.get("audit_hypothesis_key") != target["hypothesis_key"]
        ):
            raise AssertionError("sampled packet differs from its anchor trace")
        trajectory_path = _write_sampled_trajectory(
            target=target,
            destination=destination,
            batch_manifest_sha256=batch_manifest_sha256,
            replay_completed_sha256=replay_completed_sha256,
        )
        sampled_trajectory_rows = len(target["trace_records"])
        artifact = visualizer.render_decision(
            snapshot,
            histories,
            image_root / target["image_filename"],
            ai_proposals=(),
            audit_hypothesis_key=target["hypothesis_key"],
            audit_context={
                "scenario": "sampled_decision_batch_anchor",
                "case_id": target["case_id"],
                "batch_manifest_sha256": batch_manifest_sha256,
                "future_path_included": False,
            },
            suppress_audit_hypothesis=target["hypothesis_key"] is None,
        )
        if (
            artifact.kind != "decision"
            or artifact.decision_hash != target["snapshot_hash"]
            or artifact.hypothesis_key != target["hypothesis_key"]
            or artifact.maximum_market_time != target["asof"]
        ):
            raise AssertionError("blind image identity differs from its anchor")
        rendered.append(
            {
                "order": target["order"],
                "case_id": target["case_id"],
                "case_file": target["case_file"],
                "case_sha256": target["case_sha256"],
                "asof": target["asof"].isoformat(),
                "ordinal": target["anchor_ordinal"],
                "rolling_commitment": target["snapshot_hash"],
                "hypothesis_key": target["hypothesis_key"],
                "setup_id": target["setup_id"],
                "playbook": target["playbook"],
                "direction": target["direction"],
                "phase": target["phase"],
                "model_action": target["model_action"],
                "risk_action": target["risk_action"],
                "histories": history_summary,
                "sampled_trajectory": str(
                    trajectory_path.relative_to(destination)
                ),
                "sampled_trajectory_sha256": sha256_file(trajectory_path),
                "sampled_trajectory_rows": sampled_trajectory_rows,
                "image": str(artifact.path.relative_to(destination)),
                "image_sha256": artifact.sha256,
                "image_kind": artifact.kind,
                "maximum_market_time": artifact.maximum_market_time.isoformat(),
                "decision_packet": str(packet_path.relative_to(destination)),
                "decision_packet_sha256": decision_packet_sha256(packet_path),
                "decision_packet_hash": packet["packet_hash"],
                "future_path_included": False,
            }
        )
        # The completed case is durable; release its detailed in-memory rows.
        target["trace_records"].clear()
        print(
            json.dumps(
                {
                    "case_id": target["case_id"],
                    "rendered_images": len(rendered),
                    "sampled_trace_clocks": len(sampled_ordinals),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(rendered) != len(targets):
        raise RuntimeError("not every sampled case anchor was rendered once")
    rendered.sort(key=lambda item: int(item["order"]))
    output_manifest = {
        "format_version": 2,
        "artifact": "blind_decision_batch_images",
        "status": "complete",
        "batch_manifest": str(batch_manifest_path),
        "batch_manifest_sha256": batch_manifest_sha256,
        "source_replay": str(replay_root),
        "replay_completed_sha256": replay_completed_sha256,
        "decision_manifest_sha256": source_binding[
            "decision_manifest_sha256"
        ],
        "source_sha256": bindings["source_sha256"],
        "config_sha256": bindings["config_sha256"],
        "model_code_hash": bindings["model_code_hash"],
        "execution_mode": execution_mode,
        "mbo_execution_sha256": bindings.get("mbo_execution_sha256"),
        "mbo_execution_manifest_sha256": bindings.get(
            "mbo_execution_manifest_sha256"
        ),
        "decision_trace_schema_version": TRACE_SCHEMA_VERSION,
        "decision_trace_implementation_sha256": sha256_file(
            ROOT / "smc_trader" / "decision_trace.py"
        ),
        "single_causal_replay": True,
        "historical_full_minute_traces_read": False,
        "sampled_trace_union_rows": len(sampled_ordinals),
        "case_count": len(targets),
        "future_path_included": False,
        "future_artifacts_generated": False,
        "images": rendered,
    }
    manifest_path = destination / "blind_decision_images.manifest.json"
    atomic_bytes(manifest_path, canonical_json(output_manifest))
    print(
        json.dumps(
            {
                "images": len(rendered),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
