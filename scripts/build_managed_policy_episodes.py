#!/usr/bin/env python3
"""Materialize the manifest-bound 2023 managed-policy episode ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.playbook_registry import load_playbook_registry  # noqa: E402
from smc_trader.policy_value import (  # noqa: E402
    managed_policy_code_fingerprint,
    managed_policy_pipeline_fingerprint,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--policy-value-protocol",
        default="configs/policy_value_protocol_v2_1.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2_1.json",
    )
    parser.add_argument(
        "--policy-base-config",
        default="configs/model_v2_1_policy_base.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_root = Path(args.replay_output)
    destination = Path(args.output)
    manifest_destination = destination.with_suffix(
        destination.suffix + ".manifest.json"
    )
    if destination.exists() or manifest_destination.exists():
        raise FileExistsError("refusing to overwrite managed-policy episode artifacts")

    summary_path = replay_root / "summary.json"
    attempts_path = replay_root / "entry_attempts.parquet"
    trades_path = replay_root / "trades.parquet"
    completion_path = replay_root / "COMPLETED.json"
    shard_manifest_path = replay_root / "decision_shards.manifest.json"
    for source in (
        summary_path,
        attempts_path,
        trades_path,
        completion_path,
        shard_manifest_path,
    ):
        if not source.is_file():
            raise FileNotFoundError(f"managed-policy replay omits {source.name}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    validation = load_validation_protocol(args.validation_protocol)
    protocol_path = Path(args.policy_value_protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_config_path = Path(args.policy_base_config)
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if (
        not str(base_config.get("version", "")).startswith("2.1.")
        or base_config.get("managed_policy_artifact") is not None
    ):
        raise ValueError("policy-base config is not an unfitted v2.1 behavior policy")
    registry = load_playbook_registry(
        base_config.get("playbook_registry", "configs/playbooks_v2.json")
    )
    base_hash = _sha256(base_config_path)
    window = validation.ohlcv_windows.get("managed_policy_calibration")
    if window is None:
        raise ValueError("validation protocol has no managed-policy window")

    required_summary = {
        "gross_policy_calibration": True,
        "execution_authority": False,
        "profitability_evaluated": False,
        "validation_window_role": "managed_policy_calibration",
        "validation_protocol_hash": validation.fingerprint,
        "config_hash": base_hash,
        "model_code_hash": model_code_fingerprint(),
        "managed_policy_pipeline_hash": managed_policy_pipeline_fingerprint(),
        "execution_reality_source": "gross_policy_calibration_zero_cost",
        "decision_storage": "streamed_parquet_shards",
        "checkpoint_resume_supported": True,
        "full_snapshot_hash_per_minute": False,
    }
    for field, expected in required_summary.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"managed-policy replay summary {field} is not frozen as expected"
            )
    if (
        pd.Timestamp(summary.get("start")) != window.start
        or pd.Timestamp(summary.get("end")) != window.end_exclusive
    ):
        raise ValueError("managed-policy replay did not use the complete window")
    if not bool(summary.get("sequential_execution_evaluated")):
        raise ValueError("managed-policy episode source lacks sequential execution")
    if bool(summary.get("future_path_loaded")) or bool(
        summary.get("future_path_visible_to_model")
    ):
        raise ValueError("managed-policy replay exposed a future path")
    completion_expected = {
        "status": "complete",
        "summary_sha256": _sha256(summary_path),
        "entry_attempts_sha256": _sha256(attempts_path),
        "trades_sha256": _sha256(trades_path),
        "decision_shards_manifest_sha256": _sha256(shard_manifest_path),
    }
    for field, expected in completion_expected.items():
        if completion.get(field) != expected:
            raise ValueError(
                f"managed-policy completion manifest {field} is invalid"
            )
    completion_bindings = completion.get("bindings")
    if not isinstance(completion_bindings, dict):
        raise ValueError("managed-policy completion manifest omits run bindings")
    for field, expected in {
        "source_sha256": summary.get("source_sha256"),
        "config_sha256": base_hash,
        "model_code_hash": model_code_fingerprint(),
        "managed_policy_pipeline_hash": managed_policy_pipeline_fingerprint(),
        "validation_protocol_hash": validation.fingerprint,
        "policy_value_protocol_hash": _sha256(protocol_path),
    }.items():
        if completion_bindings.get(field) != expected:
            raise ValueError(
                f"managed-policy completion binding {field} is stale"
            )

    attempts = pd.read_parquet(attempts_path)
    trades = pd.read_parquet(trades_path)
    required_attempts = {
        "thesis_hash",
        "action_variant_hash",
        "decision_time",
        "playbook",
        "direction",
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
        "filled_at",
        "outcome",
    }
    missing_attempts = sorted(required_attempts - set(attempts))
    if missing_attempts:
        raise ValueError(f"entry attempts omit fields: {missing_attempts}")
    required_trades = {
        "thesis_hash",
        "playbook",
        "direction",
        "decision_time",
        "opened_at",
        "closed_at",
        "exit_reason",
        "gross_R",
        "cost_R",
        "net_R",
        "ambiguous_same_bar",
    }
    missing_trades = sorted(required_trades - set(trades))
    if missing_trades:
        raise ValueError(f"trade ledger omits fields: {missing_trades}")
    if attempts["thesis_hash"].duplicated().any():
        raise ValueError("managed-policy attempts contain duplicate thesis hashes")
    if trades["thesis_hash"].duplicated().any():
        raise ValueError("managed-policy trades contain duplicate thesis hashes")
    if not set(trades["thesis_hash"]).issubset(set(attempts["thesis_hash"])):
        raise ValueError("trade ledger contains a trade without a risk approval")

    for frame, columns in (
        (attempts, ("decision_time", "deadline", "filled_at")),
        (trades, ("decision_time", "opened_at", "closed_at")),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(
                frame[column],
                errors="coerce",
                utc=True,
            )
    if attempts[["decision_time", "deadline"]].isna().any().any():
        raise ValueError("attempt ledger contains invalid causal clocks")
    if not trades.empty and trades[
        ["decision_time", "opened_at", "closed_at"]
    ].isna().any().any():
        raise ValueError("trade ledger contains invalid causal clocks")
    if not attempts["decision_time"].between(
        window.start.tz_convert("UTC"),
        window.end_exclusive.tz_convert("UTC"),
        inclusive="left",
    ).all():
        raise ValueError("attempt decisions escape the calibration window")
    if not trades.empty and not (
        (trades["decision_time"] <= trades["opened_at"])
        & (trades["opened_at"] <= trades["closed_at"])
    ).all():
        raise ValueError("managed-policy trade clocks are not causal")
    filled_outcome = attempts["outcome"].eq("filled")
    if not attempts["filled_at"].notna().eq(filled_outcome).all():
        raise ValueError("attempt fill clocks and outcomes disagree")
    if not trades.empty:
        trade_identity = trades[
            ["thesis_hash", "decision_time", "playbook", "direction"]
        ].merge(
            attempts[
                ["thesis_hash", "decision_time", "playbook", "direction"]
            ],
            on="thesis_hash",
            how="left",
            validate="one_to_one",
            suffixes=("_trade", "_attempt"),
        )
        if not (
            trade_identity["decision_time_trade"].eq(
                trade_identity["decision_time_attempt"]
            )
            & trade_identity["playbook_trade"].eq(
                trade_identity["playbook_attempt"]
            )
            & trade_identity["direction_trade"].eq(
                trade_identity["direction_attempt"]
            )
        ).all():
            raise ValueError("trade and risk-approval identities differ")
    if not trades.empty and (
        pd.to_numeric(trades["cost_R"], errors="raise").abs() > 1e-12
    ).any():
        raise ValueError("gross managed-policy trades unexpectedly contain costs")
    if not trades.empty and not np.allclose(
        pd.to_numeric(trades["net_R"], errors="raise"),
        pd.to_numeric(trades["gross_R"], errors="raise"),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("zero-cost managed-policy net and gross returns differ")

    joined = attempts.merge(
        trades[
            [
                "thesis_hash",
                "opened_at",
                "closed_at",
                "entry_price",
                "final_stop",
                "target",
                "exit_price",
                "exit_reason",
                "gross_R",
                "cost_R",
                "net_R",
                "ambiguous_same_bar",
            ]
        ],
        on="thesis_hash",
        how="left",
        validate="one_to_one",
        suffixes=("_attempt", "_trade"),
    )
    filled = joined["filled_at"].notna()
    closed = joined["closed_at"].notna()
    if (closed & ~filled).any():
        raise ValueError("closed managed-policy trade is not recorded as filled")
    if (
        filled
        & joined["opened_at"].notna()
        & joined["filled_at"].ne(joined["opened_at"])
    ).any():
        raise ValueError("attempt fill clock and trade open clock differ")
    joined["episode_status"] = np.select(
        [
            closed,
            filled & ~closed,
            joined["outcome"].eq("pending_right_censored"),
        ],
        [
            "closed",
            "filled_right_censored",
            "approval_right_censored",
        ],
        default="unfilled",
    )
    clip_low, clip_high = protocol["managed_policy_episode"]["label_clip_R"]
    joined["managed_action_gross_R"] = pd.to_numeric(
        joined["gross_R"],
        errors="coerce",
    )
    joined.loc[
        joined["episode_status"].eq("unfilled"),
        "managed_action_gross_R",
    ] = 0.0
    joined["managed_gross_R_clipped"] = joined[
        "managed_action_gross_R"
    ].clip(float(clip_low), float(clip_high))
    joined["fit_eligible"] = joined["episode_status"].isin(
        ["closed", "unfilled"]
    )
    joined["holding_minutes"] = (
        (joined["closed_at"] - joined["opened_at"]).dt.total_seconds() / 60.0
    )

    ordered_columns = [
        "thesis_hash",
        "action_variant_hash",
        "episode_status",
        "fit_eligible",
        "decision_time",
        "filled_at",
        "opened_at",
        "closed_at",
        "playbook",
        "direction",
        "decision_phase",
        "decision_probability",
        "decision_raw_probability",
        "best_variant_utility_R",
        "path_structural_score_R",
        "planned_entry",
        "original_invalidation",
        "invalidation_source_id",
        "primary_target",
        "primary_target_id",
        "deadline",
        "entry_price",
        "final_stop",
        "target",
        "exit_price",
        "exit_reason",
        "gross_R",
        "managed_action_gross_R",
        "managed_gross_R_clipped",
        "cost_R",
        "net_R",
        "holding_minutes",
        "ambiguous_same_bar",
        "outcome",
    ]
    joined = joined[ordered_columns].sort_values(
        ["decision_time", "thesis_hash"],
        kind="stable",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(destination, index=False)
    episode_hash = _sha256(destination)

    status_counts = (
        joined["episode_status"].value_counts().sort_index().to_dict()
    )
    if int(len(joined)) != int(summary.get("approved_entry_attempts", -1)):
        raise ValueError("episode ledger does not conserve risk-approved attempts")
    if int(filled.sum()) != int(summary.get("filled_entry_attempts", -1)):
        raise ValueError("episode ledger does not conserve filled attempts")
    if int(closed.sum()) != int(summary.get("closed_trades", -1)):
        raise ValueError("episode ledger does not conserve closed trades")

    manifest = {
        "format_version": 1,
        "artifact": "managed_policy_episode_ledger",
        "status": "ready",
        "episode_table": str(destination),
        "episode_table_sha256": episode_hash,
        "episodes": int(len(joined)),
        "status_counts": status_counts,
        "fit_eligible_resolved_attempts": int(joined["fit_eligible"].sum()),
        "replay_summary": str(summary_path),
        "replay_summary_sha256": _sha256(summary_path),
        "replay_completion_manifest": str(completion_path),
        "replay_completion_manifest_sha256": _sha256(completion_path),
        "decision_shards_manifest_sha256": _sha256(shard_manifest_path),
        "entry_attempts_sha256": _sha256(attempts_path),
        "trades_sha256": _sha256(trades_path),
        "source_sha256": summary.get("source_sha256"),
        "window": {
            "role": window.role,
            "start": window.start.isoformat(),
            "end_exclusive": window.end_exclusive.isoformat(),
        },
        "validation_protocol_hash": validation.fingerprint,
        "policy_value_protocol_hash": _sha256(protocol_path),
        "playbook_registry_hash": registry.fingerprint,
        "belief_path_code_hash": model_code_fingerprint(),
        "managed_policy_code_hash": managed_policy_code_fingerprint(),
        "managed_policy_pipeline_hash": managed_policy_pipeline_fingerprint(),
        "policy_base_config_hash": base_hash,
        "future_path_used": False,
        "holdout_used": False,
    }
    manifest_destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(status_counts, sort_keys=True))


if __name__ == "__main__":
    main()
