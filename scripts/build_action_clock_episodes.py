#!/usr/bin/env python3
"""Join v2.3 feature-time shards to separately revealed shadow outcomes."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.action_clock import FEATURE_NAMES, FLAT_ACTIONS  # noqa: E402
from smc_trader.action_clock_artifact_schema import (  # noqa: E402
    EPISODE_STREAM_FIELD_TYPES,
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
from smc_trader.shadow_replay import POSITION_ACTIONS  # noqa: E402


def _stream_manifest(root: Path, name: str) -> dict[str, Any]:
    path = root / f"{name}.manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing completed stream manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("format_version", 0)) != 1
        or payload.get("status") != "complete"
        or payload.get("stream") != name
    ):
        raise ValueError(f"invalid completed stream manifest: {path}")
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
    return payload


def _paths(root: Path, manifest: dict[str, Any]) -> list[Path]:
    return [root / str(item["path"]) for item in manifest["shards"]]


class _DiskOutcomeIndex:
    """Temporary disk-backed exact-key index for bounded shard joins."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        *,
        key: str,
        database: Path,
        allow_empty: bool = False,
    ) -> None:
        self.key = key
        self.connection = sqlite3.connect(database)
        self.connection.execute(
            "CREATE TABLE outcomes (action_key TEXT PRIMARY KEY, payload BLOB NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE consumed (action_key TEXT PRIMARY KEY)"
        )
        rows = 0
        for path in _paths(root, manifest):
            frame = pd.read_parquet(path)
            if key not in frame or frame[key].isna().any():
                raise ValueError("outcome action keys are missing")
            columns = tuple(str(column) for column in frame.columns)
            key_index = columns.index(key)
            payloads = [
                (
                    str(values[key_index]),
                    sqlite3.Binary(
                        pickle.dumps(
                            dict(zip(columns, values)),
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    ),
                )
                for values in frame.itertuples(index=False, name=None)
            ]
            try:
                self.connection.executemany(
                    "INSERT INTO outcomes(action_key, payload) VALUES (?, ?)",
                    payloads,
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("outcome action keys are duplicated") from error
            self.connection.commit()
            rows += len(frame)
        if rows != int(manifest["rows"]):
            raise ValueError("outcome rows differ from their manifest")
        if rows == 0 and not allow_empty:
            raise ValueError("outcome stream is unexpectedly empty")
        self.rows = rows

    def lookup(self, keys: Iterable[Any]) -> pd.DataFrame:
        requested = [str(value) for value in keys]
        if len(requested) != len(set(requested)):
            raise ValueError("feature shard contains duplicate action keys")
        if not requested:
            return pd.DataFrame(index=pd.Index([], name=self.key))
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(requested), 500):
            chunk = requested[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            cursor = self.connection.execute(
                "SELECT action_key, payload FROM outcomes "
                f"WHERE action_key IN ({placeholders})",
                chunk,
            )
            rows.extend(
                pickle.loads(payload)
                for _, payload in cursor.fetchall()
            )
        if len(rows) != len(requested):
            raise ValueError("feature action is missing its frozen outcome")
        frame = pd.DataFrame(rows)
        if frame[self.key].duplicated().any():
            raise ValueError("disk outcome lookup returned duplicate keys")
        try:
            self.connection.executemany(
                "INSERT INTO consumed(action_key) VALUES (?)",
                [(value,) for value in requested],
            )
            self.connection.commit()
        except sqlite3.IntegrityError as error:
            self.connection.rollback()
            raise ValueError(
                "feature streams consume an outcome key more than once"
            ) from error
        indexed = frame.set_index(self.key)
        if not indexed.index.is_unique:
            raise ValueError("disk outcome lookup returned duplicate keys")
        return indexed

    def assert_all_consumed(self) -> None:
        consumed = int(
            self.connection.execute("SELECT COUNT(*) FROM consumed").fetchone()[0]
        )
        if consumed != self.rows:
            raise ValueError(
                f"feature streams consumed {consumed} of {self.rows} outcomes"
            )

    def close(self) -> None:
        self.connection.close()


def _episode_frame(
    action: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    if action["action_key"].duplicated().any():
        raise ValueError("candidate action shard contains duplicate action keys")
    joined = action.join(outcomes, on="action_key", rsuffix="_outcome")
    if joined["status"].isna().any():
        raise ValueError("candidate action is missing its frozen shadow outcome")
    if (
        joined["candidate_id"].astype(str)
        != joined["candidate_id_outcome"].astype(str)
    ).any():
        raise ValueError("candidate/outcome lineage mismatch")
    if (
        joined["action_id"].astype(str)
        != joined["action_id_outcome"].astype(str)
    ).any():
        raise ValueError("action/outcome verb mismatch")
    resolved = (
        joined["status"].eq("resolved")
        & ~joined["right_censored"].astype(bool)
    )
    exposure = ~joined["action_id"].eq("abstain")
    joined["fill_label"] = joined["filled"].astype("boolean")
    joined["conditional_gross_R"] = joined["gross_R"].where(
        joined["filled"].astype(bool)
    )
    joined["loss_label"] = (
        joined["conditional_gross_R"] < 0.0
    ).where(joined["conditional_gross_R"].notna())
    joined["gross_action_utility_R"] = joined["gross_R"]
    joined["net_action_utility_R"] = joined["net_R"]
    joined["fit_eligible_fill"] = resolved & exposure
    joined["fit_eligible_conditional_gross"] = (
        resolved & exposure & joined["filled"].astype(bool)
    )
    joined["fit_eligible_net"] = (
        resolved & exposure & joined["net_R"].notna()
    )
    joined["fit_eligible_cost"] = (
        resolved
        & exposure
        & joined["filled"].astype(bool)
        & joined["cost_R"].notna()
    )
    return joined.drop(
        columns=[
            "candidate_id_outcome",
            "action_id_outcome",
            "decision_time_outcome",
        ],
        errors="ignore",
    )


def _position_episode_frame(
    action: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    if action["position_action_key"].duplicated().any():
        raise ValueError("position action shard contains duplicate keys")
    joined = action.join(outcomes, on="position_action_key", rsuffix="_outcome")
    if joined["status"].isna().any():
        raise ValueError("position action is missing its shadow outcome")
    if (
        joined["action_id"].astype(str)
        != joined["action_id_outcome"].astype(str)
    ).any():
        raise ValueError("position action/outcome verb mismatch")
    resolved = (
        joined["status"].eq("resolved")
        & ~joined["right_censored"].astype(bool)
    )
    joined["fit_eligible_gross"] = resolved & joined["gross_R"].notna()
    joined["fit_eligible_net"] = resolved & joined["net_R"].notna()
    return joined.drop(
        columns=[
            "candidate_id_outcome",
            "parent_action_key_outcome",
            "action_id_outcome",
            "decision_time_outcome",
        ],
        errors="ignore",
    )


def _delta_rows(episodes: pd.DataFrame) -> pd.DataFrame:
    labels = episodes.pivot(
        index="candidate_id",
        columns="action_id",
        values=["gross_action_utility_R", "net_action_utility_R"],
    )
    required = set(FLAT_ACTIONS)
    if set(labels["gross_action_utility_R"].columns) != required:
        raise ValueError("paired delta table omits a registered flat action")
    enter = episodes.loc[
        episodes["action_id"].eq("enter_now"),
        [
            "candidate_id",
            "decision_time",
            "representative_playbook",
            "direction",
            "risk_passed",
            "structural_plan_valid",
            "execution_source",
            *FEATURE_NAMES,
        ],
    ].copy()
    maturity = (
        episodes.assign(
            _resolved_clock=pd.to_datetime(
                episodes["resolved_at"],
                utc=True,
                errors="coerce",
            )
        )
        .groupby("candidate_id", as_index=False)["_resolved_clock"]
        .max()
        .rename(columns={"_resolved_clock": "label_resolved_at"})
    )
    if enter["candidate_id"].duplicated().any():
        raise ValueError("candidate has multiple enter-now feature rows")
    labels.columns = [
        f"{family}__{action}"
        for family, action in labels.columns.to_flat_index()
    ]
    output = enter.merge(
        labels.reset_index(),
        on="candidate_id",
        validate="one_to_one",
    ).merge(
        maturity,
        on="candidate_id",
        validate="one_to_one",
    )
    for alternative in (
        "wait_one_bar",
        "wait_better_price",
        "wait_reacceptance",
        "abstain",
    ):
        output[f"gross_delta_enter_vs_{alternative}_R"] = (
            output["gross_action_utility_R__enter_now"]
            - output[f"gross_action_utility_R__{alternative}"]
        )
        output[f"net_delta_enter_vs_{alternative}_R"] = (
            output["net_action_utility_R__enter_now"]
            - output[f"net_action_utility_R__{alternative}"]
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-rows", type=int, default=25_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_rows < 1:
        raise ValueError("episode shard limit must be positive")
    root = Path(args.calibration_root)
    completion_path = root / "COMPLETED.json"
    if not completion_path.is_file():
        raise FileNotFoundError("action-clock calibration is not complete")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "complete"
        or completion.get("artifact") != "v2_3_action_clock_calibration"
    ):
        raise ValueError("invalid action-clock completion marker")
    manifests = {
        name: _stream_manifest(root, name)
        for name in (
            "candidate_action_shards",
            "flat_outcome_shards",
            "position_action_shards",
            "position_outcome_shards",
        )
    }
    expected_manifest_hashes = completion.get("stream_manifest_sha256")
    if not isinstance(expected_manifest_hashes, dict):
        raise ValueError("calibration completion omits stream manifest hashes")
    for name in manifests:
        if expected_manifest_hashes.get(name) != sha256_file(
            root / f"{name}.manifest.json"
        ):
            raise ValueError(
                f"calibration completion does not bind {name} manifest"
            )
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing to overwrite episode output")
    destination.mkdir(parents=True, exist_ok=True)
    episode_state = new_stream_state(
        EPISODE_STREAM_FIELD_TYPES["flat_episode_shards"]
    )
    delta_state = new_stream_state(
        EPISODE_STREAM_FIELD_TYPES["delta_episode_shards"]
    )
    position_state = new_stream_state(
        EPISODE_STREAM_FIELD_TYPES["position_episode_shards"]
    )
    episode_buffer: list[dict[str, Any]] = []
    delta_buffer: list[dict[str, Any]] = []
    position_buffer: list[dict[str, Any]] = []
    pending_delta_rows: dict[str, list[dict[str, Any]]] = {}
    maximum_pending_delta_candidates = 0
    peak_output_buffers = {
        "flat_episode_shards": 0,
        "delta_episode_shards": 0,
        "position_episode_shards": 0,
    }

    def append_output_bounded(
        stream_name: str,
        rows: list[dict[str, Any]],
        buffer: list[dict[str, Any]],
        stream_state: dict[str, Any],
        *,
        key_column: str,
    ) -> None:
        offset = 0
        while offset < len(rows):
            room = args.shard_rows - len(buffer)
            if room <= 0:
                write_stream_shards_bounded(
                    destination,
                    stream_name,
                    buffer,
                    stream_state,
                    key_column=key_column,
                    maximum_rows=args.shard_rows,
                    field_types=EPISODE_STREAM_FIELD_TYPES[stream_name],
                )
                room = args.shard_rows
            take = min(room, len(rows) - offset)
            buffer.extend(rows[offset : offset + take])
            offset += take
            peak_output_buffers[stream_name] = max(
                peak_output_buffers[stream_name],
                len(buffer),
            )
            if len(buffer) == args.shard_rows:
                write_stream_shards_bounded(
                    destination,
                    stream_name,
                    buffer,
                    stream_state,
                    key_column=key_column,
                    maximum_rows=args.shard_rows,
                    field_types=EPISODE_STREAM_FIELD_TYPES[stream_name],
                )
    counts = {
        "resolved": 0,
        "right_censored": 0,
        "filled": 0,
        "mbo_net_labels": 0,
    }
    with tempfile.TemporaryDirectory(prefix="smc-v23-episode-") as temp_root:
        temporary = Path(temp_root)
        outcome_index = _DiskOutcomeIndex(
            root,
            manifests["flat_outcome_shards"],
            key="action_key",
            database=temporary / "flat-outcomes.sqlite",
        )
        flat_paths = _paths(root, manifests["candidate_action_shards"])
        for shard_number, path in enumerate(flat_paths, start=1):
            action = pd.read_parquet(path)
            outcomes = outcome_index.lookup(action["action_key"])
            episode = _episode_frame(action, outcomes)
            counts["resolved"] += int(episode["status"].eq("resolved").sum())
            counts["right_censored"] += int(
                episode["right_censored"].astype(bool).sum()
            )
            counts["filled"] += int(episode["filled"].astype(bool).sum())
            counts["mbo_net_labels"] += int(episode["net_R"].notna().sum())
            records = episode.to_dict(orient="records")
            append_output_bounded(
                "flat_episode_shards",
                records,
                episode_buffer,
                episode_state,
                key_column="action_key",
            )
            for row in records:
                candidate_id = str(row["candidate_id"])
                pending = pending_delta_rows.setdefault(candidate_id, [])
                pending.append(row)
                if len(pending) > len(FLAT_ACTIONS):
                    raise ValueError(
                        "candidate has more than the registered flat actions"
                    )
                if len(pending) == len(FLAT_ACTIONS):
                    action_ids = {str(item["action_id"]) for item in pending}
                    if action_ids != set(FLAT_ACTIONS):
                        raise ValueError(
                            "candidate action set is duplicated or incomplete"
                        )
                    delta_frame = _delta_rows(pd.DataFrame(pending))
                    if len(delta_frame) != 1:
                        raise AssertionError(
                            "one complete candidate must create one delta row"
                        )
                    append_output_bounded(
                        "delta_episode_shards",
                        delta_frame.to_dict(orient="records"),
                        delta_buffer,
                        delta_state,
                        key_column="candidate_id",
                    )
                    del pending_delta_rows[candidate_id]
            maximum_pending_delta_candidates = max(
                maximum_pending_delta_candidates,
                len(pending_delta_rows),
            )
            if shard_number == len(flat_paths) or shard_number % 10 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "flat_episode_join",
                            "completed_shards": shard_number,
                            "total_shards": len(flat_paths),
                            "percent": round(
                                100.0 * shard_number / max(len(flat_paths), 1),
                                2,
                            ),
                            "episode_rows": int(episode_state["rows"])
                            + len(episode_buffer),
                            "delta_rows": int(delta_state["rows"])
                            + len(delta_buffer),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        outcome_index.assert_all_consumed()
        outcome_index.close()
        if pending_delta_rows:
            raise ValueError(
                "candidate stream ended with incomplete paired action sets"
            )
        write_stream_shards_bounded(
            destination,
            "flat_episode_shards",
            episode_buffer,
            episode_state,
            key_column="action_key",
            maximum_rows=args.shard_rows,
            field_types=EPISODE_STREAM_FIELD_TYPES["flat_episode_shards"],
        )
        write_stream_shards_bounded(
            destination,
            "delta_episode_shards",
            delta_buffer,
            delta_state,
            key_column="candidate_id",
            maximum_rows=args.shard_rows,
            field_types=EPISODE_STREAM_FIELD_TYPES["delta_episode_shards"],
        )

        position_outcome_index = _DiskOutcomeIndex(
            root,
            manifests["position_outcome_shards"],
            key="position_action_key",
            database=temporary / "position-outcomes.sqlite",
            allow_empty=True,
        )
        position_paths = _paths(root, manifests["position_action_shards"])
        for shard_number, path in enumerate(position_paths, start=1):
            action = pd.read_parquet(path)
            outcomes = position_outcome_index.lookup(
                action["position_action_key"]
            )
            episode = _position_episode_frame(action, outcomes)
            append_output_bounded(
                "position_episode_shards",
                episode.to_dict(orient="records"),
                position_buffer,
                position_state,
                key_column="position_action_key",
            )
            if (
                shard_number == len(position_paths)
                or shard_number % 10 == 0
            ):
                print(
                    json.dumps(
                        {
                            "stage": "position_episode_join",
                            "completed_shards": shard_number,
                            "total_shards": len(position_paths),
                            "percent": round(
                                100.0
                                * shard_number
                                / max(len(position_paths), 1),
                                2,
                            ),
                            "episode_rows": int(position_state["rows"])
                            + len(position_buffer),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        position_outcome_index.assert_all_consumed()
        position_outcome_index.close()
        write_stream_shards_bounded(
            destination,
            "position_episode_shards",
            position_buffer,
            position_state,
            key_column="position_action_key",
            maximum_rows=args.shard_rows,
            field_types=EPISODE_STREAM_FIELD_TYPES["position_episode_shards"],
        )
    bindings = {
        "calibration_completion_sha256": sha256_file(completion_path),
        "calibration_bindings": completion.get("bindings", {}),
        "source_stream_manifest_sha256": {
            name: sha256_file(root / f"{name}.manifest.json")
            for name in manifests
        },
        "future_path_used_for_features": False,
        "future_path_used_for_labels_only": True,
        "episode_builder_code_hash": sha256_file(Path(__file__)),
        "artifact_schema_sha256": sha256_file(
            ROOT / "smc_trader/action_clock_artifact_schema.py"
        ),
    }
    output_manifests = {}
    for name, state in (
        ("flat_episode_shards", episode_state),
        ("delta_episode_shards", delta_state),
        ("position_episode_shards", position_state),
    ):
        manifest_path = write_stream_manifest(
            destination,
            name,
            state,
            artifact=f"v2_3_{name}",
            bindings=bindings,
        )
        output_manifests[name] = sha256_file(manifest_path)
    summary = {
        "flat_episode_rows": int(episode_state["rows"]),
        "delta_episode_rows": int(delta_state["rows"]),
        "position_episode_rows": int(position_state["rows"]),
        **counts,
        "disk_backed_outcome_join": True,
        "maximum_in_memory_flat_rows_before_flush": (
            peak_output_buffers["flat_episode_shards"]
        ),
        "maximum_in_memory_delta_rows_before_flush": (
            peak_output_buffers["delta_episode_shards"]
        ),
        "maximum_in_memory_position_rows_before_flush": (
            peak_output_buffers["position_episode_shards"]
        ),
        "maximum_pending_delta_candidates": (
            maximum_pending_delta_candidates
        ),
        "candidate_action_conservation": (
            int(episode_state["rows"])
            == int(manifests["candidate_action_shards"]["rows"])
            == int(manifests["flat_outcome_shards"]["rows"])
        ),
        "position_action_conservation": (
            int(position_state["rows"])
            == int(manifests["position_action_shards"]["rows"])
            == int(manifests["position_outcome_shards"]["rows"])
        ),
    }
    if not (
        summary["candidate_action_conservation"]
        and summary["position_action_conservation"]
    ):
        raise AssertionError("action/outcome/episode rows are not conserved")
    summary_path = destination / "summary.json"
    atomic_bytes(summary_path, canonical_json(summary))
    atomic_bytes(
        destination / "COMPLETED.json",
        canonical_json(
            {
                "format_version": 1,
                "artifact": "v2_3_action_clock_episodes",
                "status": "complete",
                "bindings": bindings,
                "summary_sha256": sha256_file(summary_path),
                "stream_manifest_sha256": output_manifests,
            }
        ),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
