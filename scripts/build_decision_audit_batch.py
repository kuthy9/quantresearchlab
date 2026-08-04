#!/usr/bin/env python3
"""Select a deterministic, stratified blind-audit batch from light replay rows.

Historical replays deliberately contain no full minute decision traces.  This
script selects 20--40 anchor clocks from their lightweight decision shards and
records bounded pre-anchor intervals.  The full sampled trajectories are
created later by one causal replay in ``render_blind_decision_batch.py``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import (  # noqa: E402
    atomic_bytes,
    canonical_json,
    sha256_file,
    verify_stream_shards,
)


ARTIFACT = "blind_decision_trajectory_batch"
CASE_ARTIFACT = "blind_decision_trajectory_case"
ACTION_VALUES = {
    "enter",
    "wait",
    "hold",
    "protect",
    "exit",
    "abstain",
}
PLAYBOOK_VALUES = {
    "displacement_first_pullback",
    "liquidity_sweep_reversal",
    "failed_auction_value_return",
    "none",
}
DIRECTION_VALUES = {"long", "short", "none"}
H4_REGIME_VALUES = {"h4_unready", "h4_up", "h4_down", "h4_flat"}
POSITION_ACTIONS = {"hold", "protect", "exit"}
RESET_ANOMALIES = {
    "contract_change_history_reset",
    "data_gap_history_reset",
}

READ_COLUMNS = (
    "asof",
    "snapshot_hash",
    "model_action",
    "risk_action",
    "top_playbook",
    "top_direction",
    "top_phase",
    "top_setup_id",
    "decision_hypothesis_key",
    "decision_playbook",
    "decision_direction",
    "decision_phase",
    "decision_setup_id",
    "h4_regime",
    "position_open",
    "position_thesis_hash",
    "position_setup_id",
    "position_playbook",
    "position_direction",
    "observation_anomalies",
)
REQUIRED_FIELD_TYPES = {
    "asof": "timestamp_ny",
    "snapshot_hash": "large_string",
    "model_action": "large_string",
    "risk_action": "large_string",
    "top_playbook": "large_string",
    "top_direction": "large_string",
    "top_phase": "large_string",
    "top_setup_id": "large_string",
    "decision_hypothesis_key": "large_string",
    "decision_playbook": "large_string",
    "decision_direction": "large_string",
    "decision_phase": "large_string",
    "decision_setup_id": "large_string",
    "h4_regime": "large_string",
    "position_open": "bool",
    "position_thesis_hash": "large_string",
    "position_setup_id": "large_string",
    "position_playbook": "large_string",
    "position_direction": "large_string",
    "observation_anomalies": "large_string",
}
PRIMARY_DIMENSIONS = (
    "model_action",
    "risk_action",
    "direction",
    "playbook",
    "phase",
    "h4_regime",
    "session",
)


@dataclass(frozen=True)
class Candidate:
    ordinal: int
    shard: str
    shard_row: int
    asof: str
    snapshot_hash: str
    model_action: str
    risk_action: str
    direction: str
    playbook: str
    phase: str
    h4_regime: str
    session: str
    hypothesis_key: str | None
    setup_id: str | None
    setup_scope: str
    position_thesis_hash: str | None
    trace_start_ordinal: int
    trace_start_reason: str
    trace_left_censored: bool
    selection_key: str

    @property
    def stratum(self) -> str:
        return "|".join(
            (
                self.model_action,
                self.risk_action,
                self.direction,
                self.playbook,
                self.phase,
                self.h4_regime,
                self.session,
                self.setup_scope,
            )
        )

    def primary_features(self) -> set[tuple[str, str]]:
        features = {
            ("model_action", self.model_action),
            ("risk_action", self.risk_action),
            ("direction", self.direction),
            ("playbook", self.playbook),
            ("phase", self.phase),
            ("h4_regime", self.h4_regime),
            ("session", self.session),
        }
        if (
            self.setup_scope == "unbound"
            and {self.model_action, self.risk_action}.intersection(
                {"wait", "abstain"}
            )
        ):
            features.add(("ordinary_unbound", "present"))
        return features

    def secondary_features(self) -> set[tuple[str, str]]:
        return {
            ("action_pair", f"{self.model_action}|{self.risk_action}"),
            ("playbook_direction", f"{self.playbook}|{self.direction}"),
            ("regime_session", f"{self.h4_regime}|{self.session}"),
            ("setup_scope", self.setup_scope),
            ("stratum", self.stratum),
        }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is missing or not a regular file: {path}")
    try:
        return _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            name,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_shard(root: Path, relative: Any) -> Path:
    path = Path(str(relative))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("decision shard path escapes replay root")
    resolved_root = root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("decision shard path escapes replay root") from exc
    return resolved


def _load_authority(
    replay_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    completed_path = replay_root / "COMPLETED.json"
    manifest_path = replay_root / "decision_shards.manifest.json"
    completed = _read_json(completed_path, "COMPLETED marker")
    if completed.get("status") != "complete":
        raise ValueError("replay COMPLETED marker is not complete")
    bindings = _mapping(completed.get("bindings"), "COMPLETED.bindings")
    if bindings.get("runner") != "continuous_development_stream_v1":
        raise ValueError("input is not a streamed continuous replay")
    if bindings.get("include_decision_traces") is not False:
        raise ValueError(
            "audit anchors require a lightweight replay without full traces"
        )

    manifest_hashes = _mapping(
        completed.get("stream_manifest_sha256"),
        "COMPLETED.stream_manifest_sha256",
    )
    expected_manifest_hash = manifest_hashes.get("decision_shards")
    if not isinstance(expected_manifest_hash, str):
        raise ValueError("COMPLETED marker omits decision manifest hash")
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError("decision shard manifest differs from COMPLETED")
    manifest = _read_json(manifest_path, "decision shard manifest")
    if (
        manifest.get("status") != "complete"
        or manifest.get("stream") != "decision_shards"
        or manifest.get("bindings") != bindings
    ):
        raise ValueError("decision shard manifest authority is invalid")
    field_types = _mapping(
        manifest.get("field_types"),
        "decision manifest field_types",
    )
    for name, expected_type in REQUIRED_FIELD_TYPES.items():
        if field_types.get(name) != expected_type:
            raise ValueError(
                f"light decision shard field {name} is missing or invalid"
            )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("decision manifest contains no shards")
    for raw_shard in shards:
        _safe_shard(
            replay_root,
            _mapping(raw_shard, "decision shard").get("path"),
        )

    first_schema = pq.read_schema(
        _safe_shard(
            replay_root,
            _mapping(shards[0], "decision shard").get("path"),
        )
    ).remove_metadata()
    if set(first_schema.names) != set(field_types):
        raise ValueError("decision shard columns differ from registered schema")
    ordered_field_types = {
        name: field_types[name] for name in first_schema.names
    }
    stream_state = {
        "rows": int(manifest.get("rows", -1)),
        "next_shard_index": len(shards),
        "committed_shards": shards,
        "schema_fingerprint": manifest.get("schema_fingerprint"),
        "field_types": ordered_field_types,
    }
    if verify_stream_shards(replay_root, stream_state) != int(manifest["rows"]):
        raise ValueError("decision shard rows are not conserved")
    return (
        completed,
        manifest,
        sha256_file(completed_path),
        actual_manifest_hash,
    )


def _iter_decision_rows(
    replay_root: Path,
    manifest: Mapping[str, Any],
) -> Iterable[tuple[int, str, int, dict[str, Any]]]:
    ordinal = 0
    for raw_shard in manifest["shards"]:
        shard = _mapping(raw_shard, "decision shard")
        relative = str(shard["path"])
        table = pq.read_table(
            _safe_shard(replay_root, relative),
            columns=list(READ_COLUMNS),
        )
        columns = {
            name: table[name].to_pylist() for name in READ_COLUMNS
        }
        for row_index in range(table.num_rows):
            yield (
                ordinal,
                relative,
                row_index,
                {name: columns[name][row_index] for name in READ_COLUMNS},
            )
            ordinal += 1


def _session(asof: pd.Timestamp) -> str:
    hour = asof.tz_convert("America/New_York").hour
    if hour < 8:
        return "overnight"
    if hour < 12:
        return "morning"
    return "afternoon"


def _reset_before_row(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, str):
        raise ValueError("observation_anomalies must be serialized JSON")
    try:
        anomalies = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("observation_anomalies is not valid JSON") from exc
    if not isinstance(anomalies, list) or any(
        not isinstance(item, str) for item in anomalies
    ):
        raise ValueError("observation_anomalies must serialize a string list")
    return bool(RESET_ANOMALIES.intersection(anomalies))


def _context(
    row: Mapping[str, Any],
) -> tuple[
    str | None,
    str,
    str,
    str,
    str | None,
    str,
    str | None,
]:
    model_action = str(row.get("model_action"))
    hypothesis_key = _text_or_none(row.get("decision_hypothesis_key"))
    position_open = row.get("position_open") is True
    position_thesis_hash = _text_or_none(row.get("position_thesis_hash"))

    if hypothesis_key is not None:
        playbook = _text_or_none(row.get("decision_playbook")) or "none"
        direction = _text_or_none(row.get("decision_direction")) or "none"
        phase = _text_or_none(row.get("decision_phase")) or "none"
        setup_id = _text_or_none(row.get("decision_setup_id"))
        setup_scope = (
            "setup_bound" if setup_id is not None else "hypothesis_bound"
        )
    elif model_action in POSITION_ACTIONS and position_open:
        playbook = _text_or_none(row.get("position_playbook")) or "none"
        direction = _text_or_none(row.get("position_direction")) or "none"
        phase = "position_open"
        setup_id = _text_or_none(row.get("position_setup_id"))
        setup_scope = "position_bound"
    else:
        # Top belief is useful for stratification, but it is not the decision
        # identity.  In particular, never bind an abstain/wait anchor to the
        # top belief's setup when no hypothesis was selected.
        playbook = _text_or_none(row.get("top_playbook")) or "none"
        direction = _text_or_none(row.get("top_direction")) or "none"
        phase = _text_or_none(row.get("top_phase")) or "none"
        setup_id = None
        setup_scope = "unbound"
        position_thesis_hash = None

    if playbook not in PLAYBOOK_VALUES:
        raise ValueError("light decision row contains an unknown playbook")
    if direction not in DIRECTION_VALUES:
        raise ValueError("light decision row contains an unknown direction")
    return (
        hypothesis_key,
        playbook,
        direction,
        phase,
        setup_id,
        setup_scope,
        position_thesis_hash,
    )


def _offer_smallest(
    bucket: list[Candidate],
    candidate: Candidate,
    limit: int,
) -> None:
    bucket.append(candidate)
    bucket.sort(key=lambda item: (item.selection_key, item.ordinal))
    if len(bucket) > limit:
        bucket.pop()


def _scan_candidates(
    replay_root: Path,
    manifest: Mapping[str, Any],
    *,
    selection_seed: str,
    count: int,
    max_trace_rows: int,
    unbound_context_rows: int,
) -> tuple[list[Candidate], dict[str, set[str]], int]:
    by_stratum: dict[str, list[Candidate]] = {}
    global_heap: list[tuple[int, int, Candidate]] = []
    global_limit = max(80, count * 4)
    available = {name: set() for name in PRIMARY_DIMENSIONS}
    available["ordinary_unbound"] = set()
    epoch_start = 0
    prior_asof: pd.Timestamp | None = None
    row_count = 0

    for ordinal, shard, shard_row, row in _iter_decision_rows(
        replay_root,
        manifest,
    ):
        asof = _timestamp(row.get("asof"), "decision row asof")
        if prior_asof is not None and asof <= prior_asof:
            raise ValueError(
                "decision clocks are not unique and strictly increasing"
            )
        if _reset_before_row(row.get("observation_anomalies")):
            epoch_start = ordinal
        model_action = str(row.get("model_action"))
        risk_action = str(row.get("risk_action"))
        if model_action not in ACTION_VALUES or risk_action not in ACTION_VALUES:
            raise ValueError("light decision row contains an unknown action")
        h4_regime = str(row.get("h4_regime"))
        if h4_regime not in H4_REGIME_VALUES:
            raise ValueError("light decision row contains an unknown H4 regime")
        (
            hypothesis_key,
            playbook,
            direction,
            phase,
            setup_id,
            setup_scope,
            position_thesis_hash,
        ) = _context(row)
        context_rows = (
            unbound_context_rows
            if setup_scope == "unbound"
            else max_trace_rows
        )
        trace_start = max(epoch_start, ordinal - context_rows + 1)
        left_censored = trace_start > epoch_start
        session = _session(asof)
        causal_identity = "|".join(
            (
                str(row["snapshot_hash"]),
                str(hypothesis_key),
                str(setup_id),
                model_action,
                risk_action,
                playbook,
                direction,
                h4_regime,
                session,
            )
        )
        selection_key = hashlib.sha256(
            f"{selection_seed}|{causal_identity}".encode("utf-8")
        ).hexdigest()
        candidate = Candidate(
            ordinal=ordinal,
            shard=shard,
            shard_row=shard_row,
            asof=asof.isoformat(),
            snapshot_hash=str(row["snapshot_hash"]),
            model_action=model_action,
            risk_action=risk_action,
            direction=direction,
            playbook=playbook,
            phase=phase,
            h4_regime=h4_regime,
            session=session,
            hypothesis_key=hypothesis_key,
            setup_id=setup_id,
            setup_scope=setup_scope,
            position_thesis_hash=position_thesis_hash,
            trace_start_ordinal=trace_start,
            trace_start_reason="bounded_pre_anchor_context",
            trace_left_censored=left_censored,
            selection_key=selection_key,
        )
        bucket = by_stratum.setdefault(candidate.stratum, [])
        _offer_smallest(bucket, candidate, 2)
        rank = int(selection_key, 16)
        heap_item = (-rank, -ordinal, candidate)
        if len(global_heap) < global_limit:
            heapq.heappush(global_heap, heap_item)
        elif heap_item > global_heap[0]:
            heapq.heapreplace(global_heap, heap_item)
        for name, value in (
            ("model_action", candidate.model_action),
            ("risk_action", candidate.risk_action),
            ("direction", candidate.direction),
            ("playbook", candidate.playbook),
            ("phase", candidate.phase),
            ("h4_regime", candidate.h4_regime),
            ("session", candidate.session),
        ):
            available[name].add(value)
        if (
            candidate.setup_scope == "unbound"
            and {candidate.model_action, candidate.risk_action}.intersection(
                {"wait", "abstain"}
            )
        ):
            available["ordinary_unbound"].add("present")
        prior_asof = asof
        row_count += 1

    if row_count != int(manifest["rows"]):
        raise ValueError("decision scan row count differs from manifest")
    pooled = {
        candidate.ordinal: candidate
        for bucket in by_stratum.values()
        for candidate in bucket
    }
    pooled.update({item[2].ordinal: item[2] for item in global_heap})
    return list(pooled.values()), available, row_count


def _select_candidates(
    candidates: Iterable[Candidate],
    *,
    count: int,
    available: Mapping[str, set[str]],
) -> list[Candidate]:
    remaining = {item.ordinal: item for item in candidates}
    if len(remaining) < count:
        raise ValueError(
            f"only {len(remaining)} deterministic candidates are available; "
            f"cannot build requested {count}-case batch"
        )
    uncovered_primary = {
        (name, value)
        for name, values in available.items()
        for value in values
    }
    covered_secondary: set[tuple[str, str]] = set()
    selected: list[Candidate] = []
    while len(selected) < count:
        choice = min(
            remaining.values(),
            key=lambda candidate: (
                -len(candidate.primary_features() & uncovered_primary),
                -len(candidate.secondary_features() - covered_secondary),
                candidate.selection_key,
                candidate.ordinal,
            ),
        )
        selected.append(choice)
        uncovered_primary -= choice.primary_features()
        covered_secondary.update(choice.secondary_features())
        remaining.pop(choice.ordinal)
    if uncovered_primary:
        missing = sorted(
            f"{name}={value}" for name, value in uncovered_primary
        )
        raise ValueError(
            "requested batch size cannot cover all lightweight strata: "
            + ", ".join(missing)
        )
    return sorted(selected, key=lambda item: (item.asof, item.snapshot_hash))


def _case_id(completed_hash: str, candidate: Candidate) -> str:
    return hashlib.sha256(
        (
            f"blind-trace-v2|{completed_hash}|{candidate.snapshot_hash}|"
            f"{candidate.hypothesis_key}|{candidate.setup_id}"
        ).encode("utf-8")
    ).hexdigest()[:20]


def _write_cases(
    replay_root: Path,
    output: Path,
    selected: list[Candidate],
    *,
    completed_hash: str,
    decision_manifest_hash: str,
) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    cases_root = output / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        case_id = _case_id(completed_hash, candidate)
        relative = Path("cases") / f"{index:02d}-{case_id}.json"
        payload = {
            "format_version": 2,
            "artifact": CASE_ARTIFACT,
            "status": "anchor_selected",
            "case_id": case_id,
            "blind_first_pass": True,
            "future_path_included": False,
            "anchor": {
                "ordinal": candidate.ordinal,
                "asof": candidate.asof,
                "snapshot_hash": candidate.snapshot_hash,
                "model_action": candidate.model_action,
                "risk_action": candidate.risk_action,
                "hypothesis_key": candidate.hypothesis_key,
                "setup_id": candidate.setup_id,
                "playbook": candidate.playbook,
                "direction": candidate.direction,
                "phase": candidate.phase,
                "h4_regime": candidate.h4_regime,
                "session": candidate.session,
                "setup_scope": candidate.setup_scope,
                "position_thesis_hash": candidate.position_thesis_hash,
            },
            "sampled_trajectory_request": {
                "start_ordinal": candidate.trace_start_ordinal,
                "end_ordinal": candidate.ordinal,
                "maximum_market_time": candidate.asof,
                "rows": (
                    candidate.ordinal - candidate.trace_start_ordinal + 1
                ),
                "start_reason": candidate.trace_start_reason,
                "left_censored": candidate.trace_left_censored,
                "storage": "renderer_local_sampled_trajectory",
            },
            "source_reference": {
                "decision_shard": candidate.shard,
                "shard_row": candidate.shard_row,
                "ordinal": candidate.ordinal,
                "asof": candidate.asof,
                "snapshot_hash": candidate.snapshot_hash,
            },
            "source_binding": {
                "replay_root": str(replay_root.resolve()),
                "replay_completed_sha256": completed_hash,
                "decision_manifest_sha256": decision_manifest_hash,
                "lightweight_decision_shards": True,
                "full_minute_traces_present": False,
            },
        }
        path = output / relative
        atomic_bytes(path, canonical_json(payload))
        written.append(
            {
                "case_id": case_id,
                "file": str(relative),
                "sha256": sha256_file(path),
                "anchor_ordinal": candidate.ordinal,
                "anchor_asof": candidate.asof,
                "anchor_snapshot_hash": candidate.snapshot_hash,
                "trace_start_ordinal": candidate.trace_start_ordinal,
                "trajectory_rows": (
                    candidate.ordinal - candidate.trace_start_ordinal + 1
                ),
                "trace_left_censored": candidate.trace_left_censored,
                "model_action": candidate.model_action,
                "risk_action": candidate.risk_action,
                "playbook": candidate.playbook,
                "direction": candidate.direction,
                "h4_regime": candidate.h4_regime,
                "session": candidate.session,
                "setup_scope": candidate.setup_scope,
            }
        )
    return written


def _coverage(
    available: Mapping[str, set[str]],
    selected: Iterable[Candidate],
) -> dict[str, Any]:
    selected_rows = list(selected)
    output: dict[str, Any] = {}
    for name in PRIMARY_DIMENSIONS:
        selected_values = sorted(
            {str(getattr(item, name)) for item in selected_rows}
        )
        available_values = sorted(available[name])
        output[name] = {
            "available": available_values,
            "selected": selected_values,
            "complete": selected_values == available_values,
        }
    ordinary_selected = any(
        item.setup_scope == "unbound"
        and {item.model_action, item.risk_action}.intersection(
            {"wait", "abstain"}
        )
        for item in selected_rows
    )
    output["ordinary_unbound_wait_or_abstain"] = {
        "available": bool(available["ordinary_unbound"]),
        "selected": ordinary_selected,
        "complete": (
            not available["ordinary_unbound"] or ordinary_selected
        ),
    }
    return output


def read_verified_case(case_path: str | Path) -> dict[str, Any]:
    """Return one hash-bound lightweight anchor request."""

    path = Path(case_path).resolve()
    payload = _read_json(path, "decision audit case")
    if (
        payload.get("format_version") != 2
        or payload.get("artifact") != CASE_ARTIFACT
        or payload.get("status") != "anchor_selected"
        or payload.get("blind_first_pass") is not True
        or payload.get("future_path_included") is not False
    ):
        raise ValueError("decision audit case header is invalid")
    batch_path = path.parent.parent / "decision_audit_batch.manifest.json"
    batch = _read_json(batch_path, "decision audit batch manifest")
    if (
        batch.get("format_version") != 2
        or batch.get("artifact") != ARTIFACT
        or batch.get("status") != "anchors_selected"
    ):
        raise ValueError("decision audit batch manifest is invalid")
    try:
        relative = str(path.relative_to(batch_path.parent))
    except ValueError as exc:
        raise ValueError("decision audit case is outside its batch") from exc
    matching = [
        item
        for item in batch.get("cases", ())
        if isinstance(item, Mapping)
        and item.get("case_id") == payload.get("case_id")
        and item.get("file") == relative
    ]
    if len(matching) != 1 or matching[0].get("sha256") != sha256_file(path):
        raise ValueError("decision audit case differs from its batch binding")
    anchor = _mapping(payload.get("anchor"), "case.anchor")
    request = _mapping(
        payload.get("sampled_trajectory_request"),
        "case.sampled_trajectory_request",
    )
    start = int(request.get("start_ordinal", -1))
    end = int(request.get("end_ordinal", -1))
    if (
        start < 0
        or end < start
        or int(anchor.get("ordinal", -1)) != end
        or int(request.get("rows", -1)) != end - start + 1
        or request.get("maximum_market_time") != anchor.get("asof")
    ):
        raise ValueError("sampled trajectory request is invalid")
    return payload


def build_audit_batch(
    replay_root: str | Path,
    output: str | Path,
    *,
    count: int = 32,
    max_trace_rows: int = 180,
    unbound_context_rows: int = 30,
) -> Path:
    """Select a 20--40 case lightweight batch and return its manifest."""

    if not 20 <= count <= 40:
        raise ValueError("audit batch count must be between 20 and 40")
    if max_trace_rows < 1 or unbound_context_rows < 1:
        raise ValueError("trace row limits must be positive")
    if unbound_context_rows > max_trace_rows:
        raise ValueError("unbound context cannot exceed maximum trace rows")
    source = Path(replay_root)
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("decision audit batch output must be empty")
    completed, decision_manifest, completed_hash, manifest_hash = (
        _load_authority(source)
    )
    selection_seed = hashlib.sha256(
        f"decision-audit-v2|{completed_hash}|{manifest_hash}".encode("utf-8")
    ).hexdigest()
    candidates, available, rows_scanned = _scan_candidates(
        source,
        decision_manifest,
        selection_seed=selection_seed,
        count=count,
        max_trace_rows=max_trace_rows,
        unbound_context_rows=unbound_context_rows,
    )
    selected = _select_candidates(
        candidates,
        count=count,
        available=available,
    )
    case_rows = _write_cases(
        source,
        destination,
        selected,
        completed_hash=completed_hash,
        decision_manifest_hash=manifest_hash,
    )
    coverage = _coverage(available, selected)
    if not all(item["complete"] for item in coverage.values()):
        raise RuntimeError("selected batch failed lightweight stratum coverage")
    bindings = _mapping(completed["bindings"], "COMPLETED.bindings")
    output_manifest = {
        "format_version": 2,
        "artifact": ARTIFACT,
        "status": "anchors_selected",
        "blind_first_pass": True,
        "future_path_included": False,
        "case_count": len(case_rows),
        "selection": {
            "method": "deterministic_lightweight_stratum_coverage_then_hash",
            "selection_seed": selection_seed,
            "requested_count": count,
            "candidate_rows_scanned": rows_scanned,
            "candidate_pool_rows": len(candidates),
            "max_trace_rows": max_trace_rows,
            "unbound_context_rows": unbound_context_rows,
            "full_trace_fields_read": False,
        },
        "coverage": coverage,
        "source": {
            "replay_root": str(source.resolve()),
            "completed_sha256": completed_hash,
            "decision_manifest_sha256": manifest_hash,
            "decision_rows": int(decision_manifest["rows"]),
            "runner": bindings["runner"],
            "include_decision_traces": False,
        },
        "input_validation": {
            "completed_marker_verified": True,
            "decision_manifest_and_shards_verified": True,
            "registered_light_schema_verified": True,
            "decision_clocks_unique_and_strictly_increasing": True,
        },
        "cases": case_rows,
    }
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "decision_audit_batch.manifest.json"
    atomic_bytes(path, canonical_json(output_manifest))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select an outcome-blind 20-40 case batch from light decisions"
        )
    )
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--max-trace-rows", type=int, default=180)
    parser.add_argument("--unbound-context-rows", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_audit_batch(
        args.replay_root,
        args.output,
        count=args.count,
        max_trace_rows=args.max_trace_rows,
        unbound_context_rows=args.unbound_context_rows,
    )
    print(str(manifest))


if __name__ == "__main__":
    main()
