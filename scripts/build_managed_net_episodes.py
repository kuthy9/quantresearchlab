#!/usr/bin/env python3
"""Build the hash-bound v2.2 unique-action managed-value episode table."""
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

from scripts.run_managed_net_calibration import (  # noqa: E402
    calibration_pipeline_hash_is_accepted,
)
from smc_trader.action_equivalence import (  # noqa: E402
    ActionEquivalenceProtocol,
)
from smc_trader.managed_net_value import (  # noqa: E402
    FEATURE_NAMES,
    managed_net_value_code_fingerprint,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def episode_builder_code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-output", required=True)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay = Path(args.replay_output)
    destination = Path(args.output)
    manifest_path = destination.with_suffix(
        destination.suffix + ".manifest.json"
    )
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite managed-net episodes")
    sources = {
        "summary": replay / "summary.json",
        "completion": replay / "COMPLETED.json",
        "attempts": replay / "entry_attempts.parquet",
        "trades": replay / "trades.parquet",
        "shards": replay / "decision_shards.manifest.json",
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(f"v2.2 replay omits {source.name}")
    summary = json.loads(sources["summary"].read_text(encoding="utf-8"))
    completion = json.loads(
        sources["completion"].read_text(encoding="utf-8")
    )
    if completion.get("status") != "complete":
        raise ValueError("v2.2 replay is not complete")
    for field, source in (
        ("summary_sha256", sources["summary"]),
        ("entry_attempts_sha256", sources["attempts"]),
        ("trades_sha256", sources["trades"]),
        ("decision_shards_manifest_sha256", sources["shards"]),
    ):
        if completion.get(field) != _sha256(source):
            raise ValueError(f"v2.2 completion {field} is stale")

    validation = load_validation_protocol(args.validation_protocol)
    action_protocol = ActionEquivalenceProtocol.from_file(
        args.action_equivalence_protocol
    )
    replay_pipeline_hash = str(summary.get("calibration_pipeline_hash", ""))
    if not calibration_pipeline_hash_is_accepted(replay_pipeline_hash):
        raise ValueError("v2.2 replay calibration pipeline hash is stale")
    policy_base_config_hash = _sha256(args.config)
    if str(summary.get("config_hash", "")) != policy_base_config_hash:
        raise ValueError("v2.2 replay policy-base config hash is stale")
    bindings = {
        "validation_protocol_hash": validation.fingerprint,
        "action_equivalence_protocol_hash": action_protocol.fingerprint,
        "managed_net_value_protocol_hash": _sha256(
            args.managed_net_value_protocol
        ),
        "calibration_pipeline_hash": replay_pipeline_hash,
        "managed_net_value_code_hash": managed_net_value_code_fingerprint(),
        "policy_base_config_hash": policy_base_config_hash,
        "episode_builder_code_hash": episode_builder_code_fingerprint(),
    }
    for field, expected in {
        key: value
        for key, value in bindings.items()
        if key
        not in {
            "policy_base_config_hash",
            "episode_builder_code_hash",
        }
    }.items():
        if str(summary.get(field, "")) != expected:
            raise ValueError(f"v2.2 replay summary {field} is stale")
    if bool(summary.get("future_path_loaded")) or bool(
        summary.get("future_path_visible_to_model")
    ):
        raise ValueError("v2.2 episode source exposed future path")
    policy_variant = str(summary.get("policy_variant", ""))
    if policy_variant not in {
        "all_three",
        "without_failed_auction_value_return",
        "without_liquidity_sweep_reversal",
    }:
        raise ValueError("v2.2 replay has an unregistered policy variant")

    attempts = pd.read_parquet(sources["attempts"])
    trades = pd.read_parquet(sources["trades"])
    required_attempts = {
        "thesis_hash",
        "action_key",
        "decision_time",
        "representative_playbook",
        "direction",
        "evidence_hypothesis_keys",
        "evidence_playbooks",
        "evidence_state_json",
        "filled_at",
        "outcome",
        *FEATURE_NAMES,
    }
    missing = sorted(required_attempts - set(attempts))
    if missing:
        raise ValueError(f"v2.2 attempts omit fields: {missing}")
    if attempts["thesis_hash"].duplicated().any():
        raise ValueError("v2.2 attempts contain duplicate thesis hashes")
    if not trades.empty and trades["thesis_hash"].duplicated().any():
        raise ValueError("v2.2 trades contain duplicate thesis hashes")
    for column in ("decision_time", "filled_at"):
        attempts[column] = pd.to_datetime(
            attempts[column],
            utc=True,
            errors="coerce",
        )
    for column in ("decision_time", "opened_at", "closed_at"):
        if column in trades:
            trades[column] = pd.to_datetime(
                trades[column],
                utc=True,
                errors="coerce",
            )
    if attempts["decision_time"].isna().any():
        raise ValueError("v2.2 attempts contain invalid decision clocks")
    if not trades.empty and trades[
        ["decision_time", "opened_at", "closed_at"]
    ].isna().any().any():
        raise ValueError("v2.2 trades contain invalid causal clocks")
    filled = attempts["outcome"].eq("filled")
    if not attempts["filled_at"].notna().eq(filled).all():
        raise ValueError("v2.2 fill clocks and outcomes disagree")
    if not set(trades.get("thesis_hash", ())).issubset(
        set(attempts["thesis_hash"])
    ):
        raise ValueError("v2.2 trade lacks a risk-approved action")

    trade_columns = [
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
    joined = attempts.merge(
        trades[trade_columns],
        on="thesis_hash",
        how="left",
        validate="one_to_one",
    )
    closed = joined["closed_at"].notna()
    if (closed & ~filled).any():
        raise ValueError("closed v2.2 trade is not marked filled")
    filled_open = filled & ~closed
    if closed.any() and (
        pd.to_numeric(joined.loc[closed, "cost_R"], errors="raise").abs()
        > 1e-12
    ).any():
        raise ValueError("v2.2 structural calibration unexpectedly has costs")
    unfilled = joined["outcome"].eq("not_filled_or_expired_next_bar")
    pending = joined["outcome"].eq("pending_right_censored")
    right_censored = pending | filled_open
    if not (closed | unfilled | right_censored).all():
        raise ValueError("v2.2 episode has an unknown terminal state")
    joined["episode_status"] = np.select(
        [closed, unfilled, right_censored],
        ["closed", "unfilled", "right_censored"],
        default="invalid",
    )
    joined["fit_eligible"] = closed | unfilled
    joined["managed_action_gross_R"] = np.where(
        closed,
        pd.to_numeric(joined["gross_R"], errors="coerce"),
        np.where(unfilled, 0.0, np.nan),
    )
    protocol = json.loads(
        Path(args.managed_net_value_protocol).read_text(encoding="utf-8")
    )
    clip_low, clip_high = protocol["episode_contract"]["gross_label_clip_R"]
    joined["managed_gross_R_clipped"] = joined[
        "managed_action_gross_R"
    ].clip(float(clip_low), float(clip_high))
    if joined.loc[
        joined["fit_eligible"],
        list(FEATURE_NAMES),
    ].isna().any().any():
        raise ValueError("v2.2 eligible episode has missing causal features")

    output_columns = [
        "thesis_hash",
        "action_key",
        "episode_status",
        "fit_eligible",
        "decision_time",
        "filled_at",
        "opened_at",
        "closed_at",
        "representative_playbook",
        "direction",
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
        *FEATURE_NAMES,
        "entry_price",
        "final_stop",
        "target",
        "exit_price",
        "exit_reason",
        "gross_R",
        "managed_action_gross_R",
        "managed_gross_R_clipped",
        "holding_minutes",
        "ambiguous_same_bar",
        "outcome",
    ]
    joined["holding_minutes"] = (
        (joined["closed_at"] - joined["opened_at"]).dt.total_seconds() / 60.0
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    joined[output_columns].to_parquet(destination, index=False)
    eligible = joined["fit_eligible"].astype(bool)
    manifest = {
        "format_version": 1,
        "artifact": "managed_net_unique_action_episodes",
        "status": "ready",
        "policy_variant": policy_variant,
        "disabled_playbooks": list(summary.get("disabled_playbooks", ())),
        **bindings,
        "episode_table": str(destination),
        "episode_table_sha256": _sha256(destination),
        "episodes": int(len(joined)),
        "fit_eligible_resolved_actions": int(eligible.sum()),
        "closed_actions": int((joined["episode_status"] == "closed").sum()),
        "unfilled_actions": int(
            (joined["episode_status"] == "unfilled").sum()
        ),
        "right_censored_actions": int(
            (joined["episode_status"] == "right_censored").sum()
        ),
        "unique_action_keys": int(joined["action_key"].nunique()),
        "future_path_used": False,
        "holdout_used": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
