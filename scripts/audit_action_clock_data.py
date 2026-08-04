#!/usr/bin/env python3
"""Pre-fit causal, grain, conservation and primitive audit for v2.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.action_clock import (  # noqa: E402
    FEATURE_NAMES,
    FLAT_ACTIONS,
    ActionClockProtocol,
    action_clock_code_fingerprint,
)
from smc_trader.artifact_stream import (  # noqa: E402
    atomic_bytes,
    canonical_json,
    sha256_file,
    verify_stream_shards,
)
from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.shadow_replay import (  # noqa: E402
    POSITION_ACTIONS,
    POSITION_FEATURE_NAMES,
    POSITION_SAMPLING_STRIDE_MINUTES,
    shadow_replay_code_fingerprint,
)


PRIMITIVE_DIAGNOSTIC_SAMPLE_LIMIT = 250_000


def _spearman(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman correlation without an undeclared SciPy dependency."""

    paired = pd.concat(
        [
            pd.to_numeric(left, errors="coerce").rename("left"),
            pd.to_numeric(right, errors="coerce").rename("right"),
        ],
        axis=1,
    ).dropna()
    if (
        len(paired) < 2
        or paired["left"].nunique() < 2
        or paired["right"].nunique() < 2
    ):
        return float("nan")
    return float(
        paired["left"].rank(method="average").corr(
            paired["right"].rank(method="average")
        )
    )


class _AuditLedger:
    """Disk-backed global uniqueness and cross-stream conservation ledger."""

    _FLAT_FLAGS = {
        "action": "action_count",
        "outcome": "outcome_count",
        "episode": "episode_count",
    }
    _POSITION_FLAGS = {
        "action": "action_count",
        "outcome": "outcome_count",
        "episode": "episode_count",
    }

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            CREATE TABLE flat_keys (
                key TEXT PRIMARY KEY,
                action_count INTEGER NOT NULL DEFAULT 0,
                outcome_count INTEGER NOT NULL DEFAULT 0,
                episode_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE candidate_states (
                candidate_id TEXT PRIMARY KEY,
                occurrences INTEGER NOT NULL
            );
            CREATE TABLE candidate_actions (
                candidate_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                occurrences INTEGER NOT NULL,
                PRIMARY KEY(candidate_id, action_id)
            );
            CREATE TABLE delta_candidates (
                candidate_id TEXT PRIMARY KEY,
                occurrences INTEGER NOT NULL
            );
            CREATE TABLE position_keys (
                key TEXT PRIMARY KEY,
                action_count INTEGER NOT NULL DEFAULT 0,
                outcome_count INTEGER NOT NULL DEFAULT 0,
                episode_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE position_clocks (
                parent_action_key TEXT NOT NULL,
                position_state_id TEXT NOT NULL,
                decision_time_ns INTEGER NOT NULL,
                PRIMARY KEY(
                    parent_action_key,
                    position_state_id,
                    decision_time_ns
                )
            );
            """
        )

    def _mark_flag(
        self,
        *,
        table: str,
        column: str,
        keys: Iterable[Any],
    ) -> None:
        values = [(str(value),) for value in keys]
        if not values:
            return
        self.connection.executemany(
            f"""
            INSERT INTO {table}(key, {column}) VALUES (?, 1)
            ON CONFLICT(key) DO UPDATE SET {column}={column}+1
            """,
            values,
        )
        self.connection.commit()

    def mark_flat(self, role: str, keys: Iterable[Any]) -> None:
        self._mark_flag(
            table="flat_keys",
            column=self._FLAT_FLAGS[role],
            keys=keys,
        )

    def mark_position(self, role: str, keys: Iterable[Any]) -> None:
        self._mark_flag(
            table="position_keys",
            column=self._POSITION_FLAGS[role],
            keys=keys,
        )

    def mark_candidate_states(self, values: Iterable[Any]) -> None:
        self.connection.executemany(
            """
            INSERT INTO candidate_states(candidate_id, occurrences) VALUES (?, 1)
            ON CONFLICT(candidate_id)
            DO UPDATE SET occurrences=occurrences+1
            """,
            [(str(value),) for value in values],
        )
        self.connection.commit()

    def mark_candidate_actions(
        self,
        values: Iterable[tuple[Any, Any]],
    ) -> None:
        self.connection.executemany(
            """
            INSERT INTO candidate_actions(
                candidate_id, action_id, occurrences
            ) VALUES (?, ?, 1)
            ON CONFLICT(candidate_id, action_id)
            DO UPDATE SET occurrences=occurrences+1
            """,
            [(str(candidate), str(action)) for candidate, action in values],
        )
        self.connection.commit()

    def mark_delta_candidates(self, values: Iterable[Any]) -> None:
        self.connection.executemany(
            """
            INSERT INTO delta_candidates(candidate_id, occurrences) VALUES (?, 1)
            ON CONFLICT(candidate_id)
            DO UPDATE SET occurrences=occurrences+1
            """,
            [(str(value),) for value in values],
        )
        self.connection.commit()

    def mark_position_clocks(self, frame: pd.DataFrame) -> None:
        clocks = pd.to_datetime(
            frame["decision_time"],
            utc=True,
            errors="coerce",
        )
        if clocks.isna().any():
            return
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO position_clocks(
                parent_action_key, position_state_id, decision_time_ns
            ) VALUES (?, ?, ?)
            """,
            [
                (
                    str(parent),
                    str(state),
                    int(clock.value),
                )
                for parent, state, clock in zip(
                    frame["parent_action_key"],
                    frame["position_state_id"],
                    clocks,
                )
            ],
        )
        self.connection.commit()

    def scalar(self, query: str) -> int:
        return int(self.connection.execute(query).fetchone()[0])

    def position_clock_gaps_are_valid(self) -> bool:
        previous_key: tuple[str, str] | None = None
        previous_clock: int | None = None
        minimum_ns = POSITION_SAMPLING_STRIDE_MINUTES * 60 * 1_000_000_000
        cursor = self.connection.execute(
            """
            SELECT parent_action_key, position_state_id, decision_time_ns
            FROM position_clocks
            ORDER BY parent_action_key, position_state_id, decision_time_ns
            """
        )
        for parent, state, clock in cursor:
            key = (str(parent), str(state))
            current = int(clock)
            if (
                key == previous_key
                and previous_clock is not None
                and current - previous_clock < minimum_ns
            ):
                return False
            previous_key = key
            previous_clock = current
        return True

    def close(self) -> None:
        self.connection.close()


def _manifest(root: Path, name: str) -> tuple[dict[str, Any], list[Path]]:
    path = root / f"{name}.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("stream") != name
    ):
        raise ValueError(f"invalid stream manifest: {path}")
    shard_paths = [root / str(item["path"]) for item in payload["shards"]]
    state = {
        "rows": int(payload["rows"]),
        "next_shard_index": len(payload["shards"]),
        "committed_shards": list(payload["shards"]),
    }
    if payload.get("schema_fingerprint") is not None:
        field_types = dict(payload["field_types"])
        if shard_paths:
            import pyarrow.parquet as pq

            physical_order = pq.read_schema(shard_paths[0]).names
            if set(physical_order) != set(field_types):
                raise ValueError(
                    f"manifest/physical schema fields differ: {path}"
                )
            field_types = {
                field: field_types[field] for field in physical_order
            }
        state["schema_fingerprint"] = payload["schema_fingerprint"]
        state["field_types"] = field_types
    verify_stream_shards(root, state)
    return payload, shard_paths


def _completion(root: Path, artifact: str) -> tuple[dict[str, Any], Path]:
    path = root / "COMPLETED.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("artifact") != artifact:
        raise ValueError(f"invalid completion marker: {path}")
    expected = payload.get("stream_manifest_sha256")
    if not isinstance(expected, dict):
        raise ValueError(f"completion marker omits stream hashes: {path}")
    return payload, path


def _causal_json_checks(
    row: Any,
    errors: list[str],
    *,
    root_label: str,
) -> None:
    decision_time = pd.Timestamp(row.decision_time)
    if decision_time.tzinfo is None:
        errors.append(f"{root_label}: candidate decision clock is naive")
        return
    if pd.Timestamp(row.initial_plan_observed_at) > decision_time:
        errors.append(f"{root_label}: initial plan is future-observed")
    if pd.Timestamp(row.phase_started_at) > decision_time:
        errors.append(f"{root_label}: phase starts after its action clock")
    try:
        frames = json.loads(row.observation_state_json)
        for frame in frames.values():
            if pd.Timestamp(frame["cutoff"]) > decision_time:
                errors.append(f"{root_label}: timeframe cutoff is in the future")
        sequence = json.loads(row.sequence_state_json)
        if sequence is not None:
            for step in sequence.get("steps", ()):
                observed = step.get("observed_at")
                if observed is not None and pd.Timestamp(observed) > decision_time:
                    errors.append(f"{root_label}: sequence step is future-observed")
        memory = json.loads(row.event_memory_json)
        for event in memory.get("recent_events", ()):
            if pd.Timestamp(event["observed_at"]) > decision_time:
                errors.append(f"{root_label}: event memory contains a future event")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{root_label}: malformed causal JSON: {error}")


def _flat_outcome_checks(
    frame: pd.DataFrame,
    errors: list[str],
    *,
    root_label: str,
) -> None:
    if frame["action_key"].isna().any() or frame["action_key"].duplicated().any():
        errors.append(f"{root_label}: flat outcome action keys are invalid")
    censored = frame["right_censored"].astype(bool)
    if (
        frame.loc[censored, ["gross_R", "net_R"]]
        .notna()
        .any()
        .any()
    ):
        errors.append(f"{root_label}: right-censored flat labels are populated")
    unfilled = ~frame["filled"].astype(bool) & ~censored
    for column in ("gross_R", "cost_R", "net_R"):
        values = pd.to_numeric(frame.loc[unfilled, column], errors="coerce")
        if values.isna().any() or not np.allclose(values.to_numpy(), 0.0):
            errors.append(
                f"{root_label}: unfilled {column} is not deterministic 0R"
            )
    filled = frame["filled"].astype(bool) & ~censored
    if frame.loc[filled, "gross_R"].isna().any():
        errors.append(f"{root_label}: resolved fill has no gross label")
    decision = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    resolved_at = pd.to_datetime(frame["resolved_at"], utc=True, errors="coerce")
    resolved_rows = ~censored
    if (
        decision.isna().any()
        or resolved_at.loc[resolved_rows].isna().any()
        or (resolved_at.loc[resolved_rows] < decision.loc[resolved_rows]).any()
    ):
        errors.append(f"{root_label}: flat label maturity clock is invalid")


def _primitive_metrics(
    frame: pd.DataFrame,
    *,
    population_rows: int | None = None,
) -> dict[str, Any]:
    enter = frame.loc[
        frame["action_id"].eq("enter_now")
        & frame["status"].eq("resolved")
        & ~frame["right_censored"].astype(bool)
    ].copy()
    if enter.empty:
        return {
            "enter_actions_population": int(population_rows or 0),
            "enter_actions_sampled": 0,
            "sampling": "no_resolved_enter_actions",
        }
    enter["calendar_year"] = pd.to_datetime(
        enter["decision_time"],
        utc=True,
    ).dt.year
    measures = {
        "extension_to_remaining_draw": -1.0,
        "swing_trigger_consistency": 1.0,
        "first_pullback_quality": 1.0,
        "remaining_path_R_capped": 1.0,
    }
    output: dict[str, Any] = {
        "enter_actions_population": int(
            len(enter) if population_rows is None else population_rows
        ),
        "enter_actions_sampled": int(len(enter)),
        "sampling": (
            "complete_population"
            if population_rows is None or population_rows <= len(enter)
            else (
                "deterministic_reservoir_seed_2301_limit_"
                f"{PRIMITIVE_DIAGNOSTIC_SAMPLE_LIMIT}"
            )
        ),
        "overall": {},
    }
    for feature, expected_sign in measures.items():
        correlation = _spearman(
            enter[feature],
            enter["gross_action_utility_R"],
        )
        output["overall"][feature] = {
            "spearman_to_gross_utility": (
                None if pd.isna(correlation) else float(correlation)
            ),
            "preregistered_expected_sign": int(expected_sign),
        }
    output["by_year_direction"] = []
    for (year, direction), group in enter.groupby(
        ["calendar_year", "direction"]
    ):
        row: dict[str, Any] = {
            "year": int(year),
            "direction": str(direction),
            "actions": int(len(group)),
        }
        for feature in measures:
            correlation = _spearman(
                group[feature],
                group["gross_action_utility_R"],
            )
            row[f"{feature}_spearman"] = (
                None if pd.isna(correlation) else float(correlation)
            )
        output["by_year_direction"].append(row)
    return output


def _audit_pair(
    calibration_root: Path,
    episode_root: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    label = str(calibration_root)
    calibration_completion, calibration_completion_path = _completion(
        calibration_root,
        "v2_3_action_clock_calibration",
    )
    episode_completion, episode_completion_path = _completion(
        episode_root,
        "v2_3_action_clock_episodes",
    )
    if (
        episode_completion.get("bindings", {}).get(
            "calibration_completion_sha256"
        )
        != sha256_file(calibration_completion_path)
    ):
        errors.append(f"{label}: episode lineage does not bind calibration")
    calibration_streams = {}
    for name in (
        "candidate_state_shards",
        "candidate_action_shards",
        "flat_outcome_shards",
        "position_action_shards",
        "position_outcome_shards",
    ):
        manifest, paths = _manifest(calibration_root, name)
        calibration_streams[name] = (manifest, paths)
        if calibration_completion["stream_manifest_sha256"].get(
            name
        ) != sha256_file(calibration_root / f"{name}.manifest.json"):
            errors.append(f"{label}: completion does not bind {name}")
    episode_streams = {}
    for name in (
        "flat_episode_shards",
        "delta_episode_shards",
        "position_episode_shards",
    ):
        manifest, paths = _manifest(episode_root, name)
        episode_streams[name] = (manifest, paths)
        if episode_completion["stream_manifest_sha256"].get(
            name
        ) != sha256_file(episode_root / f"{name}.manifest.json"):
            errors.append(f"{label}: completion does not bind {name}")

    candidate_count = int(
        calibration_streams["candidate_state_shards"][0]["rows"]
    )
    current_protocol = ActionClockProtocol.from_file(
        ROOT / "configs/action_clock_value_protocol_v2_3.json"
    )
    calibration_bindings = calibration_completion.get("bindings", {})
    if (
        calibration_bindings.get("action_clock_protocol_hash")
        != current_protocol.fingerprint
    ):
        errors.append(f"{label}: calibration action-clock protocol is stale")
    if (
        calibration_bindings.get("action_clock_code_hash")
        != action_clock_code_fingerprint()
    ):
        errors.append(f"{label}: calibration feature code is stale")
    if (
        calibration_bindings.get("shadow_replay_code_hash")
        != shadow_replay_code_fingerprint()
    ):
        errors.append(f"{label}: calibration shadow replay code is stale")
    if (
        calibration_bindings.get("base_model_code_hash")
        != model_code_fingerprint()
    ):
        errors.append(f"{label}: calibration base-model code is stale")
    if (
        calibration_bindings.get("config_sha256")
        != sha256_file(ROOT / "configs/model_v2_3_action_clock_base.json")
    ):
        errors.append(f"{label}: calibration runtime semantics are stale")
    if (
        episode_completion.get("bindings", {}).get(
            "episode_builder_code_hash"
        )
        != sha256_file(ROOT / "scripts/build_action_clock_episodes.py")
    ):
        errors.append(f"{label}: episode builder code is stale")
    if (
        episode_completion.get("bindings", {}).get(
            "artifact_schema_sha256"
        )
        != sha256_file(
            ROOT / "smc_trader/action_clock_artifact_schema.py"
        )
    ):
        errors.append(f"{label}: episode physical schema binding is stale")

    counts = {
        "candidate_states": 0,
        "candidate_actions": 0,
        "flat_outcomes": 0,
        "flat_episodes": 0,
        "delta_episodes": 0,
        "position_actions": 0,
        "position_outcomes": 0,
        "position_episodes": 0,
        "risk_passed_actions": 0,
        "structural_plan_valid_actions": 0,
        "resolved_flat_actions": 0,
        "right_censored_flat_actions": 0,
        "mbo_candidate_clocks": 0,
    }
    action_ids_seen: set[str] = set()
    position_action_ids_seen: set[str] = set()
    neutral_execution_fields = [
        "spread_ticks",
        "expected_round_trip_cost_R",
        "fillability",
        "depth_imbalance_aligned",
    ]
    primitive_columns = [
        "action_id",
        "status",
        "right_censored",
        "decision_time",
        "direction",
        "gross_action_utility_R",
        "extension_to_remaining_draw",
        "swing_trigger_consistency",
        "first_pullback_quality",
        "remaining_path_R_capped",
    ]
    primitive_sample: list[dict[str, Any]] = []
    primitive_population = 0
    rng = np.random.default_rng(2301)

    with tempfile.TemporaryDirectory(prefix="smc-v23-audit-") as temp_root:
        ledger = _AuditLedger(Path(temp_root) / "audit.sqlite")

        for path in calibration_streams["candidate_state_shards"][1]:
            frame = pd.read_parquet(path)
            counts["candidate_states"] += len(frame)
            if "candidate_id" not in frame or frame["candidate_id"].isna().any():
                errors.append(f"{label}: candidate state key is missing")
                continue
            ledger.mark_candidate_states(frame["candidate_id"])
            if len(errors) < 100:
                for row in frame.itertuples(index=False):
                    _causal_json_checks(row, errors, root_label=label)
                    if len(errors) >= 100:
                        break

        action_columns = [
            "candidate_id",
            "action_key",
            "action_id",
            "decision_time",
            "risk_passed",
            "structural_plan_valid",
            "execution_source",
            *FEATURE_NAMES,
        ]
        for path in calibration_streams["candidate_action_shards"][1]:
            frame = pd.read_parquet(path, columns=action_columns)
            counts["candidate_actions"] += len(frame)
            if (
                frame["candidate_id"].isna().any()
                or frame["action_key"].isna().any()
            ):
                errors.append(f"{label}: candidate action keys are missing")
                continue
            ledger.mark_flat("action", frame["action_key"])
            ledger.mark_candidate_actions(
                zip(frame["candidate_id"], frame["action_id"])
            )
            action_ids_seen.update(frame["action_id"].astype(str).unique())
            numeric = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                errors.append(
                    f"{label}: action features contain missing/non-finite values"
                )
            missing_execution = frame["mbo_available"].lt(0.5)
            if not np.allclose(
                frame.loc[
                    missing_execution,
                    neutral_execution_fields,
                ].to_numpy(dtype=float),
                0.0,
            ):
                errors.append(
                    f"{label}: missing execution authority contains "
                    "pseudo-MBO values"
                )
            counts["risk_passed_actions"] += int(
                frame["risk_passed"].astype(bool).sum()
            )
            counts["structural_plan_valid_actions"] += int(
                frame["structural_plan_valid"].astype(bool).sum()
            )
            counts["mbo_candidate_clocks"] += int(
                (
                    frame["action_id"].eq("enter_now")
                    & frame["mbo_available"].ge(0.5)
                ).sum()
            )

        for path in calibration_streams["flat_outcome_shards"][1]:
            frame = pd.read_parquet(path)
            counts["flat_outcomes"] += len(frame)
            if frame["action_key"].isna().any():
                errors.append(f"{label}: flat outcome key is missing")
                continue
            ledger.mark_flat("outcome", frame["action_key"])
            _flat_outcome_checks(frame, errors, root_label=label)
            counts["resolved_flat_actions"] += int(
                frame["status"].eq("resolved").sum()
            )
            counts["right_censored_flat_actions"] += int(
                frame["right_censored"].astype(bool).sum()
            )

        episode_columns = sorted(
            {
                "action_key",
                "action_id",
                "decision_time",
                "direction",
                "status",
                "right_censored",
                "filled",
                "filled_at",
                "cost_R",
                "cost_source",
                "cost_observed_at",
                "mbo_available",
                *primitive_columns,
            }
        )
        for path in episode_streams["flat_episode_shards"][1]:
            frame = pd.read_parquet(path, columns=episode_columns)
            counts["flat_episodes"] += len(frame)
            if frame["action_key"].isna().any():
                errors.append(f"{label}: flat episode key is missing")
                continue
            ledger.mark_flat("episode", frame["action_key"])
            mbo = frame["mbo_available"].ge(0.5)
            filled = frame["filled"].astype(bool)
            resolved = (
                frame["status"].eq("resolved")
                & ~frame["right_censored"].astype(bool)
            )
            labeled_fills = filled & resolved
            causal_cost = frame["cost_source"].eq("mbo_reconstructed")
            if frame.loc[
                causal_cost & labeled_fills,
                "cost_R",
            ].isna().any():
                errors.append(f"{label}: MBO-resolved fills omit cost labels")
            if frame.loc[
                ~causal_cost & labeled_fills,
                "cost_R",
            ].notna().any():
                errors.append(
                    f"{label}: non-MBO fills claim execution cost labels"
                )
            cost_clock = pd.to_datetime(
                frame["cost_observed_at"],
                utc=True,
                errors="coerce",
            )
            fill_clock = pd.to_datetime(
                frame["filled_at"],
                utc=True,
                errors="coerce",
            )
            causal_fills = causal_cost & labeled_fills
            if (
                cost_clock.loc[causal_fills].isna().any()
                or (
                    cost_clock.loc[causal_fills]
                    > fill_clock.loc[causal_fills]
                ).any()
            ):
                errors.append(f"{label}: execution cost is future-observed")
            immediate = frame["action_id"].isin(
                ["enter_now", "wait_better_price"]
            )
            if (
                causal_cost.loc[immediate & labeled_fills]
                != mbo.loc[immediate & labeled_fills]
            ).any():
                errors.append(
                    f"{label}: immediate-order cost authority differs from "
                    "its action-clock MBO feature"
                )
            selected = frame.loc[
                frame["action_id"].eq("enter_now")
                & frame["status"].eq("resolved")
                & ~frame["right_censored"].astype(bool),
                primitive_columns,
            ]
            for row in selected.to_dict(orient="records"):
                primitive_population += 1
                if len(primitive_sample) < PRIMITIVE_DIAGNOSTIC_SAMPLE_LIMIT:
                    primitive_sample.append(row)
                    continue
                replacement = int(rng.integers(0, primitive_population))
                if replacement < PRIMITIVE_DIAGNOSTIC_SAMPLE_LIMIT:
                    primitive_sample[replacement] = row

        delta_columns = [
            "candidate_id",
            "decision_time",
            "label_resolved_at",
            *[
                f"gross_delta_enter_vs_{action}_R"
                for action in FLAT_ACTIONS
                if action != "enter_now"
            ],
        ]
        for path in episode_streams["delta_episode_shards"][1]:
            frame = pd.read_parquet(path, columns=delta_columns)
            counts["delta_episodes"] += len(frame)
            if frame["candidate_id"].isna().any():
                errors.append(f"{label}: paired delta candidate key is missing")
                continue
            ledger.mark_delta_candidates(frame["candidate_id"])
            delta_decision = pd.to_datetime(
                frame["decision_time"],
                utc=True,
                errors="coerce",
            )
            delta_maturity = pd.to_datetime(
                frame["label_resolved_at"],
                utc=True,
                errors="coerce",
            )
            label_present = frame[
                [
                    f"gross_delta_enter_vs_{action}_R"
                    for action in FLAT_ACTIONS
                    if action != "enter_now"
                ]
            ].notna().all(axis=1)
            if (
                delta_decision.isna().any()
                or delta_maturity.loc[label_present].isna().any()
                or (
                    delta_maturity.loc[label_present]
                    < delta_decision.loc[label_present]
                ).any()
            ):
                errors.append(
                    f"{label}: paired delta maturity clock is invalid"
                )

        position_columns = [
            "position_action_key",
            "position_state_id",
            "parent_action_key",
            "action_id",
            "decision_time",
            "current_stop_activated_at",
            "applied_stop",
            "current_stop",
            "protection_source_id",
            "protection_available",
            *POSITION_FEATURE_NAMES,
        ]
        for path in calibration_streams["position_action_shards"][1]:
            frame = pd.read_parquet(path, columns=position_columns)
            counts["position_actions"] += len(frame)
            if frame.empty:
                continue
            if frame["position_action_key"].isna().any():
                errors.append(f"{label}: position action key is missing")
                continue
            ledger.mark_position("action", frame["position_action_key"])
            ledger.mark_position_clocks(frame)
            position_action_ids_seen.update(
                frame["action_id"].astype(str).unique()
            )
            if not np.isfinite(
                frame[list(POSITION_FEATURE_NAMES)].to_numpy(dtype=float)
            ).all():
                errors.append(f"{label}: position features are non-finite")
            decision = pd.to_datetime(
                frame["decision_time"],
                utc=True,
                errors="coerce",
            )
            activated = pd.to_datetime(
                frame["current_stop_activated_at"],
                utc=True,
                errors="coerce",
            )
            if (
                decision.isna().any()
                or activated.isna().any()
                or (activated > decision).any()
            ):
                errors.append(
                    f"{label}: protected stop state clock is invalid"
                )
            unchanged = frame["action_id"].isin(["hold", "exit"])
            if not np.allclose(
                pd.to_numeric(
                    frame.loc[unchanged, "applied_stop"]
                ).to_numpy(),
                pd.to_numeric(
                    frame.loc[unchanged, "current_stop"]
                ).to_numpy(),
            ):
                errors.append(
                    f"{label}: hold/exit rewrites the current stop"
                )
            protect = frame["action_id"].eq("protect")
            if (
                frame.loc[protect, "protection_source_id"].isna().any()
                or not frame.loc[
                    protect,
                    "protection_available",
                ].astype(bool).all()
            ):
                errors.append(
                    f"{label}: protect lacks causal structure provenance"
                )

        position_outcome_columns = [
            "position_action_key",
            "decision_time",
            "resolved_at",
            "right_censored",
            "gross_R",
            "net_R",
        ]
        for path in calibration_streams["position_outcome_shards"][1]:
            frame = pd.read_parquet(path, columns=position_outcome_columns)
            counts["position_outcomes"] += len(frame)
            if frame.empty:
                continue
            if frame["position_action_key"].isna().any():
                errors.append(f"{label}: position outcome key is missing")
                continue
            ledger.mark_position("outcome", frame["position_action_key"])
            censored = frame["right_censored"].astype(bool)
            if frame.loc[
                censored,
                ["gross_R", "net_R"],
            ].notna().any().any():
                errors.append(
                    f"{label}: right-censored position labels are populated"
                )
            decision = pd.to_datetime(
                frame["decision_time"],
                utc=True,
                errors="coerce",
            )
            resolved_at = pd.to_datetime(
                frame["resolved_at"],
                utc=True,
                errors="coerce",
            )
            resolved_rows = ~censored
            if (
                decision.isna().any()
                or resolved_at.loc[resolved_rows].isna().any()
                or (
                    resolved_at.loc[resolved_rows]
                    < decision.loc[resolved_rows]
                ).any()
            ):
                errors.append(
                    f"{label}: position label maturity clock is invalid"
                )

        for path in episode_streams["position_episode_shards"][1]:
            frame = pd.read_parquet(
                path,
                columns=["position_action_key"],
            )
            counts["position_episodes"] += len(frame)
            if not frame.empty:
                if frame["position_action_key"].isna().any():
                    errors.append(f"{label}: position episode key is missing")
                    continue
                ledger.mark_position(
                    "episode",
                    frame["position_action_key"],
                )

        if counts["candidate_states"] != candidate_count:
            errors.append(
                f"{label}: candidate-state row count is not conserved"
            )
        if ledger.scalar(
            "SELECT COUNT(*) FROM candidate_states WHERE occurrences != 1"
        ):
            errors.append(f"{label}: candidate IDs are duplicated")
        if counts["candidate_actions"] != candidate_count * len(FLAT_ACTIONS):
            errors.append(f"{label}: candidate action family is incomplete")
        if ledger.scalar(
            "SELECT COUNT(*) FROM candidate_actions WHERE occurrences != 1"
        ):
            errors.append(f"{label}: candidate/action pairs are duplicated")
        if ledger.scalar(
            """
            SELECT COUNT(*) FROM (
                SELECT candidate_id
                FROM candidate_actions
                GROUP BY candidate_id
                HAVING COUNT(*) != 5
            )
            """
        ):
            errors.append(f"{label}: a candidate omits an action alternative")
        if ledger.scalar(
            """
            SELECT COUNT(*)
            FROM candidate_actions AS actions
            LEFT JOIN candidate_states AS states
              ON states.candidate_id = actions.candidate_id
            WHERE states.candidate_id IS NULL
            """
        ):
            errors.append(
                f"{label}: action stream contains an unknown candidate"
            )
        if action_ids_seen != set(FLAT_ACTIONS):
            errors.append(f"{label}: flat action family changed")
        if ledger.scalar(
            """
            SELECT COUNT(*) FROM flat_keys
            WHERE action_count != 1
               OR outcome_count != 1
               OR episode_count != 1
            """
        ):
            errors.append(
                f"{label}: flat action/outcome/episode keys are not bijective"
            )
        if (
            counts["flat_outcomes"] != counts["candidate_actions"]
            or counts["flat_episodes"] != counts["candidate_actions"]
        ):
            errors.append(
                f"{label}: flat action/outcome/episode rows are not conserved"
            )
        if (
            counts["delta_episodes"] != candidate_count
            or ledger.scalar(
                """
                SELECT COUNT(*) FROM delta_candidates
                WHERE occurrences != 1
                """
            )
        ):
            errors.append(
                f"{label}: paired delta candidate rows are not conserved"
            )
        if ledger.scalar(
            """
            SELECT COUNT(*)
            FROM delta_candidates AS delta
            LEFT JOIN candidate_states AS states
              ON states.candidate_id = delta.candidate_id
            WHERE states.candidate_id IS NULL
            """
        ):
            errors.append(
                f"{label}: paired delta contains an unknown candidate"
            )
        if counts["position_actions"] > 0:
            if not position_action_ids_seen.issubset(set(POSITION_ACTIONS)):
                errors.append(
                    f"{label}: unregistered position action IDs are present"
                )
            if not {"hold", "exit"}.issubset(position_action_ids_seen):
                errors.append(
                    f"{label}: position action family omits hold or exit"
                )
            if "protect" not in position_action_ids_seen:
                warnings.append(
                    f"{label}: no causal protection opportunity was observed"
                )
        if ledger.scalar(
            """
            SELECT COUNT(*) FROM position_keys
            WHERE action_count != 1
               OR outcome_count != 1
               OR episode_count != 1
            """
        ):
            errors.append(
                f"{label}: position action/outcome/episode keys are not "
                "bijective"
            )
        if not (
            counts["position_actions"]
            == counts["position_outcomes"]
            == counts["position_episodes"]
        ):
            errors.append(
                f"{label}: position action/outcome/episode rows are not "
                "conserved"
            )
        if not ledger.position_clock_gaps_are_valid():
            errors.append(
                f"{label}: a position state violates the five-minute clock"
            )
        ledger.close()

    if counts["mbo_candidate_clocks"] == 0:
        warnings.append(f"{label}: no causal MBO action clocks are covered")
    primitive = _primitive_metrics(
        pd.DataFrame(primitive_sample, columns=primitive_columns),
        population_rows=primitive_population,
    )
    report = {
        "calibration_root": str(calibration_root),
        "episode_root": str(episode_root),
        "calibration_completion_sha256": sha256_file(
            calibration_completion_path
        ),
        "episode_completion_sha256": sha256_file(episode_completion_path),
        "candidate_plans": candidate_count,
        "candidate_actions": counts["candidate_actions"],
        "resolved_flat_actions": counts["resolved_flat_actions"],
        "right_censored_flat_actions": counts[
            "right_censored_flat_actions"
        ],
        "risk_passed_actions": counts["risk_passed_actions"],
        "structural_plan_valid_actions": counts[
            "structural_plan_valid_actions"
        ],
        "mbo_action_clocks": counts["mbo_candidate_clocks"],
        "position_action_rows": counts["position_actions"],
        "position_outcome_rows": counts["position_outcomes"],
        "streaming_shard_audit": True,
        "disk_backed_global_key_ledger": True,
        "maximum_full_streams_concatenated_in_memory": 0,
        "primitive_diagnostics": primitive,
    }
    return report, errors, warnings


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# v2.3 Action-Clock Data Quality Audit",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This audit runs before fitting or profitability gates. Primitive signs "
        "are diagnostics, not labels or permission to trade.",
        "",
    ]
    for item in payload["roots"]:
        lines.extend(
            [
                f"## {item['calibration_root']}",
                "",
                f"- Candidate plans: {item['candidate_plans']}",
                f"- Candidate action rows: {item['candidate_actions']}",
                f"- Resolved / censored: {item['resolved_flat_actions']} / "
                f"{item['right_censored_flat_actions']}",
                f"- Causal MBO action clocks: {item['mbo_action_clocks']}",
                f"- Position actions / outcomes: {item['position_action_rows']} / "
                f"{item['position_outcome_rows']}",
                "",
            ]
        )
    if payload["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {value}" for value in payload["errors"])
        lines.append("")
    if payload["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {value}" for value in payload["warnings"])
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", action="append", required=True)
    parser.add_argument("--episode-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.calibration_root) != len(args.episode_root):
        raise ValueError("calibration and episode roots must be paired")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite data-quality audit")
    roots = []
    errors: list[str] = []
    warnings: list[str] = []
    for calibration, episodes in zip(
        args.calibration_root,
        args.episode_root,
    ):
        report, found_errors, found_warnings = _audit_pair(
            Path(calibration),
            Path(episodes),
        )
        roots.append(report)
        errors.extend(found_errors)
        warnings.extend(found_warnings)
    payload = {
        "format_version": 1,
        "artifact": "v2_3_action_clock_data_quality_audit",
        "status": "passed" if not errors else "failed",
        "action_clock_protocol_hash": ActionClockProtocol.from_file(
            ROOT / "configs/action_clock_value_protocol_v2_3.json"
        ).fingerprint,
        "roots": roots,
        "errors": errors,
        "warnings": warnings,
        "profitability_evaluated": False,
        "future_path_used_for_feature_checks": False,
        "primitive_outcomes_used_for_diagnostics_only": True,
        "audit_code_sha256": sha256_file(Path(__file__)),
    }
    atomic_bytes(output, canonical_json(payload))
    markdown_path = output.with_suffix(".md")
    atomic_bytes(markdown_path, _markdown(payload).encode("utf-8"))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "errors": len(errors),
                "warnings": len(warnings),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if errors:
        raise RuntimeError("v2.3 data-quality audit failed")


if __name__ == "__main__":
    main()
