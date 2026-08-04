"""Bounded, resumable EXP016 displacement semantic discovery."""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
import pandas as pd

from .artifact_stream import atomic_bytes, canonical_json, new_stream_state, sha256_file, verify_stream_shards, write_stream_manifest, write_stream_shards_bounded
from .calibration_replay import ReplayCheckpointStore, iter_after_source_checkpoint
from .causal import CausalMarketReader
from .displacement import DisplacementProtocol
from .displacement_observer import CausalDisplacementEye, _CURRENT_METRIC_FIELDS
from .io import load_ohlcv
from .model import Timeframe, to_primitive

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "EXP-SMC-3.0.2-016-DISPLACEMENT-PRODUCTION-IDENTITY-CUSTODY"
PREREGISTRATION_SHA = "6a092dcfbce1845bc4b0eff946a69fc7315bbbcfc7827642a2dceba7c3aaf301"
DEFAULT_CONFIG = ROOT / "configs/experiments/EXP-SMC-3.0.2-016-DISPLACEMENT-PRODUCTION-IDENTITY-CUSTODY.json"
FIELD_TYPES = {
    "record_id": "large_string", "record_kind": "large_string", "input_clock": "timestamp_ny",
    "observation_clock": "timestamp_ny", "source_cursor": "timestamp_ny", "m5_event_clock": "timestamp_ny",
    "same_update_sequence": "int64", "transition_id": "large_string", "entity_id": "large_string",
    "direction": "large_string", "lifecycle": "large_string", "reason": "large_string",
    "prefix_last_admitted_at": "timestamp_ny", "terminal_at": "timestamp_ny",
    "real_episode_bar_count": "int64", "net_points": "float64", "relative_atr": "float64",
    "efficiency": "float64", "speed_atr_per_bar": "float64", "volume_ratio": "float64",
    "state_json": "large_string", "state_sha256": "large_string",
    "reader_anomalies_json": "large_string", "provenance": "large_string",
}

def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

def _safe_path(value, *, leaf=False):
    path = _resolve(value)
    for parent in path.parents:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()): raise PermissionError(f"unsafe path ancestor: {path}")
    if path.is_symlink() or (leaf and not path.is_file()): raise PermissionError(f"unsafe path leaf: {path}")
    return path

def _json(value): return json.dumps(to_primitive(value), sort_keys=True, separators=(",", ":"))
def _sha_text(value): return hashlib.sha256(value.encode("utf-8")).hexdigest()
def _priority(salt, protocol_sha, stratum_id, identity): return _sha_text(f"{salt}{protocol_sha}{stratum_id}{identity}")
def _digest(value): return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)
def _safe_json(path):
    try: return json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() and path.is_file() else {}
    except (OSError, TypeError, ValueError): return {}
def _verify_stream(destination, state):
    rows = verify_stream_shards(destination, state); expected = {str(x["path"]) for x in state["committed_shards"]}; actual = {str(x.relative_to(destination)) for x in (destination / "semantic_shards").glob("part-*.parquet")}
    if actual != expected: raise ValueError("semantic shard directory contains unregistered artifacts")
    return rows

def _consider(slots, stratum_id, priority, packet, maximum=40):
    prior = slots.get(stratum_id)
    if prior is None and len(slots) >= maximum: raise ValueError("audit selector exceeded its frozen bound")
    candidate = {"priority": priority, **dict(packet)}
    if prior is None or priority < str(prior["priority"]): slots[stratum_id] = candidate

def _year_windows(start, end_exclusive):
    start, end_exclusive = pd.Timestamp(start), pd.Timestamp(end_exclusive)
    if start.tzinfo is None or end_exclusive.tzinfo is None or end_exclusive <= start:
        raise ValueError("discovery window must be positive and timezone-aware")
    output, left = [], start
    while left < end_exclusive:
        right = min(pd.Timestamp(f"{left.year + 1}-01-01", tz=left.tz), end_exclusive)
        output.append((left, right)); left = right
    return tuple(output)

def _state_row(update, state, *, kind, transition_id, sequence, source_cursor):
    state_json = _json(state); state_hash = _sha_text(state_json)
    m5 = tuple(update.newly_completed.get(Timeframe.M5, ()))
    record_id = _sha_text(f"{kind}{update.asof.isoformat()}{sequence}{transition_id}{state_hash}")
    return {
        "record_id": record_id, "record_kind": kind, "input_clock": update.completed_1m.end,
        "observation_clock": update.asof, "source_cursor": source_cursor,
        "m5_event_clock": None if not m5 or update.anomalies else m5[-1].end,
        "same_update_sequence": int(sequence), "transition_id": transition_id,
        "entity_id": state.entity_id, "direction": state.direction.value,
        "lifecycle": state.lifecycle.value, "reason": state.terminal_reason or "",
        "prefix_last_admitted_at": state.prefix_last_admitted_at, "terminal_at": state.terminal_at,
        "real_episode_bar_count": int(state.real_episode_bar_count), "net_points": float(state.net_points),
        "relative_atr": float(state.relative_atr), "efficiency": float(state.efficiency),
        "speed_atr_per_bar": float(state.speed_atr_per_bar),
        "volume_ratio": None if state.volume_ratio is None else float(state.volume_ratio),
        "state_json": state_json, "state_sha256": state_hash,
        "reader_anomalies_json": _json(tuple(update.anomalies)), "provenance": "observer_raw_update",
    }

def _semantic_rows(update, observation, raw_update, source_cursor):
    if raw_update is None or update.asof != observation.asof: raise ValueError("eye raw/compact update is missing or stale")
    if tuple(update.anomalies) != tuple(observation.reader_anomalies): raise ValueError("compact reader anomalies disagree")
    if len(tuple(update.newly_completed.get(Timeframe.M5, ()))) > 1: raise ValueError("one M1 update emitted multiple M5 candles")
    transitions = tuple(raw_update.transitions)
    projected = tuple(observation.recent_transitions[-len(transitions):]) if transitions else ()
    expected = tuple((x.transition_id, x.state.entity_id, x.state.lifecycle.value, x.state.terminal_reason, x.state.observed_at, x.state.direction) for x in transitions)
    actual = tuple((x.transition_id, x.entity_id, x.lifecycle, x.reason, x.observed_at, x.direction) for x in projected)
    if expected != actual: raise ValueError("compact and raw displacement transitions disagree")
    if raw_update.state is None and observation.current_metrics is not None: raise ValueError("compact idle metrics disagree")
    if raw_update.state is not None:
        metrics = tuple((name, float(getattr(raw_update.state, name))) for name in _CURRENT_METRIC_FIELDS if getattr(raw_update.state, name) is not None)
        if metrics != observation.current_metrics: raise ValueError("compact and raw displacement metrics disagree")
    rows = [_state_row(update, x.state, kind="transition", transition_id=x.transition_id, sequence=i, source_cursor=source_cursor) for i, x in enumerate(transitions)]
    represented = any(x.state == raw_update.state for x in transitions)
    if raw_update.state is not None and update.newly_completed.get(Timeframe.M5) and not represented:
        rows.append(_state_row(update, raw_update.state, kind="continuation", transition_id="", sequence=len(rows), source_cursor=source_cursor))
    return rows

def _required_strata():
    transitions = {f"{y}:{d}:{life}" for y in range(2017, 2022) for d in ("long", "short") for life in ("started", "active", "exhausted")}
    return transitions | {f"{y}:idle:{window}" for y in range(2017, 2022) for window in ("10-11", "14-15")}

def _case_id(priority): return _sha_text(f"EXP014-case{priority}")

def _transition_packet(update, transition, stratum, priority):
    state = transition.state; m5 = tuple(update.histories.get(Timeframe.M5, ())); m1 = tuple(update.histories.get(Timeframe.M1, ()))
    seed = next((i for i, candle in enumerate(m5) if candle.end == state.started_at), -1)
    target = state.prefix_last_admitted_at
    if state.terminal_evidence_candle_id is not None and state.terminal_at is not None and state.terminal_at > target:
        target = state.terminal_at
    finish = next((i for i, candle in enumerate(m5) if candle.end == target), -1)
    if seed < 15 or finish < seed or len(m1) < 30: return None
    prefix = m5[seed - 15:finish + 1]
    if len(prefix) > 1024 or not all(c.real_completed for c in prefix): return None
    if any((c.symbol, c.instrument_id) != (state.symbol, state.instrument_id) for c in prefix): return None
    case_id = _case_id(priority)
    return {
        "case_id": case_id,
        "blind": {"case_id": case_id, "cutoff": update.asof.isoformat(), "m5_prefix": to_primitive(prefix), "m1_prefix": to_primitive(m1[-30:])},
        "overlay": {"case_id": case_id, "cutoff": update.asof.isoformat(), "stratum": stratum, "priority": priority,
                    "transition_id": transition.transition_id, "state": to_primitive(state), "provenance": "observer_raw_update"},
    }

def _idle_stratum(asof):
    local = pd.Timestamp(asof).tz_convert("America/New_York"); minute = local.hour * 60 + local.minute
    window = "10-11" if 600 <= minute < 660 else ("14-15" if 840 <= minute < 900 else None)
    return None if window is None or not 2017 <= local.year <= 2021 else f"{local.year}:idle:{window}"

def _capture_candidates(state, update, observation, raw, *, source_bar_real, protocol_sha, salt):
    for transition in raw.transitions:
        life = transition.state.lifecycle.value; direction = transition.state.direction.value; year = transition.state.observed_at.tz_convert("America/New_York").year
        if life not in {"started", "active", "exhausted"} or not 2017 <= year <= 2021: continue
        stratum = f"{year}:{direction}:{life}"; score = _priority(salt, protocol_sha, stratum, transition.transition_id)
        packet = _transition_packet(update, transition, stratum, score)
        if packet is None: state["census"]["audit_prefix_overflow"] += int(next((i for i, x in enumerate(update.histories.get(Timeframe.M5, ())) if x.end == transition.state.started_at), -1) < 15)
        else: _consider(state["selector_slots"], stratum, score, packet)
    stratum = _idle_stratum(observation.asof); m5 = tuple(update.histories.get(Timeframe.M5, ())); m1 = tuple(update.histories.get(Timeframe.M1, ()))
    eligible = stratum is not None and source_bar_real and observation.lifecycle == "idle" and not raw.transitions
    eligible = eligible and not update.anomalies and not update.newly_completed.get(Timeframe.M5) and len(m5) >= 24 and len(m1) >= 30
    if not eligible: return
    identity = f"{update.completed_1m.start.isoformat()}{update.completed_1m.symbol}{update.completed_1m.instrument_id}"
    score = _priority(salt, protocol_sha, stratum, identity); prior = state["selector_slots"].get(stratum)
    if prior is not None and score >= prior["priority"]: return
    case_id = _case_id(score)
    _consider(state["selector_slots"], stratum, score, {
        "case_id": case_id,
        "blind": {"case_id": case_id, "cutoff": update.asof.isoformat(), "m5_prefix": to_primitive(m5[-24:]), "m1_prefix": to_primitive(m1[-30:])},
        "overlay": {"case_id": case_id, "cutoff": update.asof.isoformat(), "stratum": stratum, "priority": score,
                    "transition_id": None, "state": None, "provenance": "observer_raw_update"},
    })

def _audit_transitions(state, raw, observation):
    prior_clock = state["last_transition_clock"]; recent = list(state["recent_transition_ids"])
    census = state["census"]
    for anomaly in observation.reader_anomalies:
        census["reader_anomalies"][anomaly] = int(census["reader_anomalies"].get(anomaly, 0)) + 1
    for index, transition in enumerate(raw.transitions):
        clock = transition.state.observed_at
        if prior_clock is not None and clock < prior_clock: raise ValueError("displacement transition clock decreased")
        if clock == prior_clock:
            if index == 0: raise ValueError("equal transition clocks crossed reader updates")
            pair = (raw.transitions[index - 1].state.lifecycle.value, transition.state.lifecycle.value)
            if pair != ("exhausted", "started"): raise ValueError("unregistered equal-clock transition pair")
        if transition.transition_id in recent: raise ValueError("duplicate displacement transition ID")
        recent = (recent + [transition.transition_id])[-64:]; prior_clock = clock
        life = transition.state.lifecycle.value
        month = clock.strftime("%Y-%m"); direction = transition.state.direction.value
        census["months"].add(month); census["calendar_month_counts"][month] = int(census["calendar_month_counts"].get(month, 0)) + 1
        census["direction_counts"][direction] = int(census["direction_counts"].get(direction, 0)) + 1
        key = f"{direction}:{life}"; census["direction_lifecycle_counts"][key] = int(census["direction_lifecycle_counts"].get(key, 0)) + 1
        census[life] = int(census.get(life, 0)) + 1
        if life == "started":
            key = f"{transition.state.direction.value}_started"; census[key] = int(census.get(key, 0)) + 1
        if life == "active":
            key = str(transition.state.real_episode_bar_count)
            census["activation_bars"][key] = int(census["activation_bars"].get(key, 0)) + 1
        if transition.state.terminal_reason:
            reason = transition.state.terminal_reason
            census["terminal_reasons"][reason] = int(census["terminal_reasons"].get(reason, 0)) + 1
            if life == "censored":
                census["boundary_counts"][reason] = int(census["boundary_counts"].get(reason, 0)) + 1
    state["last_transition_clock"] = prior_clock; state["recent_transition_ids"] = recent
    raw_state = raw.state
    expected = ("idle", None, None, None, None, None, None) if raw_state is None else (raw_state.lifecycle.value, raw_state.entity_id,
        raw_state.direction, raw_state.observed_at, raw_state.started_at, raw_state.active_at, raw_state.prefix_last_admitted_at)
    actual = (observation.lifecycle, observation.current_entity_id, observation.current_direction, observation.current_state_observed_at,
              observation.current_started_at, observation.current_active_at, observation.current_last_admitted_at)
    if actual != expected: raise ValueError("raw current state disagrees with compact observation")

def _coverage(state):
    census = state["census"]; missing = sorted(_required_strata() - set(state["selector_slots"]))
    floors = (census.get("started", 0) >= 240, census.get("long_started", 0) >= 80, census.get("short_started", 0) >= 80,
              census.get("active", 0) >= 40, census["terminal_reasons"].get("activation_window_elapsed", 0) >= 20, len(census["months"]) >= 12, census["integrity_violations"] == 0); return ("complete_semantic_available" if all(floors) and not missing else "complete_semantic_unavailable"), missing

def _progress(state, total_rows, started, session_rows):
    rows = int(state["source_rows_consumed"]); elapsed = max(time.monotonic() - started, 1e-9); rate = max(rows - session_rows, 0) / elapsed; remaining = max(total_rows - rows, 0)
    return {"source_rows_processed": rows, "source_rows_total": total_rows, "completed_percent": 100.0 * rows / max(total_rows, 1),
            "rows_per_second": rate, "eta_seconds": None if rate <= 0 else remaining / rate, "processed_bars": int(state["processed_bars"]), "semantic_rows": int(state["stream_state"]["rows"]) + len(state["buffer"]), "resume_count": int(state["resume_count"])}

def _sync(state):
    stream = state["stream_state"]; state["decision_rows"] = int(stream["rows"]); state["next_shard_index"] = int(stream["next_shard_index"]); state["committed_shards"] = stream["committed_shards"]

def _new_state(protocol):
    stream = new_stream_state(FIELD_TYPES)
    return {
        "reader": CausalMarketReader(), "replay": CausalDisplacementEye(protocol), "stream_state": stream,
        "buffer": [], "selector_slots": {}, "census": {"started": 0, "long_started": 0, "short_started": 0,
        "active": 0, "exhausted": 0, "censored": 0, "months": set(), "terminal_reasons": {},
        "activation_bars": {}, "boundary_counts": {}, "reader_anomalies": {},
        "calendar_month_counts": {}, "direction_counts": {"long": 0, "short": 0}, "direction_lifecycle_counts": {},
        "integrity_violations": 0, "audit_prefix_overflow": 0},
        "processed_bars": 0, "source_rows_consumed": 0, "last_checkpoint_source_rows": 0,
        "last_source_start": None, "last_asof": None, "previous_open_state": None, "last_transition_clock": None,
        "recent_transition_ids": [], "resume_count": 0, "peak_buffer_rows": 0,
        "decision_rows": 0, "next_shard_index": 0, "committed_shards": stream["committed_shards"],
    }

def _preflight_fail(destination, bindings, exc, trusted_cursor=None):
    checkpoint_path = destination / "_checkpoint/manifest.json"; checkpoint_meta = _safe_json(checkpoint_path); prior = _safe_json(destination / "progress.json")
    payload = {**prior, "status": "failed_integrity", "failure_type": type(exc).__name__, "failure_message": str(exc), "durable_source_rows": int(prior.get("source_rows_processed", 0)), "durable_source_cursor": None if trusted_cursor is None else pd.Timestamp(trusted_cursor).isoformat()}
    atomic_bytes(destination / "progress.json", canonical_json(payload))
    atomic_bytes(destination / "FAILED.json", canonical_json({"format_version": 1, **payload, "bindings": bindings,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() and not checkpoint_path.is_symlink() else None,
        "checkpoint_state_sha256": checkpoint_meta.get("state_sha256"),
        "progress_sha256": sha256_file(destination / "progress.json")}))

def run(config_path=DEFAULT_CONFIG, *, expected_source_sha256, release_path=None, resume=False, diagnostic_stop_after_source_rows=0, _synthetic=False):
    config_path = _safe_path(config_path, leaf=True); data_root = (ROOT / "data").resolve()
    if _synthetic and (config_path.resolve() == DEFAULT_CONFIG.resolve() or config_path.resolve().is_relative_to(data_root)): raise PermissionError("synthetic bypass cannot read registered data paths")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID or config["bindings"].get("preregistration_sha256") != PREREGISTRATION_SHA: raise ValueError("EXP016 config identity is not frozen")
    source = _resolve(config["runner"]["discovery_source_path"]); source_manifest = _resolve(config["runner"]["discovery_manifest_path"]); destination = _resolve(config["runner"]["output_root"]); protocol_path = _resolve(config["bindings"]["protocol_path"]); dependency_paths = tuple(_resolve(path) for path in config["bindings"]["runtime_dependency_sha256"])
    if _synthetic and (any(path.resolve().is_relative_to(data_root) for path in (source, source_manifest, protocol_path, *dependency_paths)) or destination.resolve().is_relative_to((ROOT / "outputs").resolve())): raise PermissionError("synthetic bypass cannot address registered real paths")
    for path in (source, source_manifest, protocol_path, *dependency_paths): _safe_path(path, leaf=True)
    _safe_path(destination)
    release, release_hash = {}, "synthetic-fixture"
    if not _synthetic:
        if release_path is None: raise PermissionError("Gate 3 real discovery release is absent")
        release_path = _safe_path(release_path, leaf=True)
        release = json.loads(release_path.read_text(encoding="utf-8")); release_hash = sha256_file(release_path)
        valid_release = release.get("status") == "gate3_real_discovery_authorized" and release.get("execution_authorized") is True; valid_release = valid_release and release.get("experiment_id") == EXPERIMENT_ID and release.get("preregistration_sha256") == PREREGISTRATION_SHA
        valid_release = valid_release and release.get("base_config_sha256") == sha256_file(config_path); valid_release = valid_release and _digest(expected_source_sha256) and release.get("source_sha256") == expected_source_sha256
        valid_release = valid_release and release.get("output_root") == config["runner"]["output_root"]; required = ("source_manifest_sha256", "custody_receipt_sha256", "custody_terminal_sha256", "candidate_bundle_sha256", "gate1_candidate_bundle_sha256", "synthetic_result_sha256")
        valid_release = valid_release and all(_digest(release.get(name)) for name in required)
        valid_release = valid_release and release.get("runner_sha256") == sha256_file(Path(__file__))
        valid_release = valid_release and release.get("observer_sha256") == sha256_file(ROOT / "smc_trader/displacement_observer.py")
        valid_release = valid_release and release.get("candidate_bundle_sha256") == config["candidate"]["implementation_component_bundle_sha256"] and release.get("gate1_candidate_bundle_sha256") == config["bindings"]["gate1_attempt002_candidate_bundle_sha256"]
        valid_release = valid_release and release.get("synthetic_result_sha256") == config["bindings"]["gate1_attempt002_terminal_sha256"] and release.get("custody_receipt_sha256") == config["bindings"]["gate2_core_result_sha256"] and release.get("custody_terminal_sha256") == config["bindings"]["gate2_terminal_sha256"]
        valid_release = valid_release and release.get("source_sha256") == config["bindings"]["gate2_output_sha256"] and release.get("source_manifest_sha256") == config["bindings"]["gate2_output_manifest_sha256"]
        valid_release = valid_release and _resolve(release.get("custody_receipt_path", "")) == _resolve(config["custody"]["core_result_path"]) and _resolve(release.get("custody_terminal_path", "")) == _resolve(config["custody"]["terminal_path"])
        valid_release = valid_release and release.get("real_attempt_limit") == 1
        if not valid_release: raise PermissionError("Gate 3 release bindings are invalid")
        custody_path = _safe_path(release["custody_receipt_path"], leaf=True); terminal_path = _safe_path(release["custody_terminal_path"], leaf=True)
    bindings = {"experiment_id": EXPERIMENT_ID, "preregistration_sha256": PREREGISTRATION_SHA,
        "config_sha256": sha256_file(config_path), "protocol_sha256": config["bindings"]["protocol_sha256"],
        "source_sha256": expected_source_sha256,
        "source_manifest_sha256": release.get("source_manifest_sha256", "synthetic-fixture"),
        "runner_sha256": sha256_file(Path(__file__)),
        "observer_sha256": sha256_file(ROOT / "smc_trader/displacement_observer.py"),
        "schema_sha256": hashlib.sha256(canonical_json(FIELD_TYPES)).hexdigest(),
        "custody_receipt_sha256": release.get("custody_receipt_sha256", "synthetic-fixture"),
        "custody_terminal_sha256": release.get("custody_terminal_sha256", "synthetic-fixture"),
        "candidate_bundle_sha256": release.get("candidate_bundle_sha256", "synthetic-fixture"),
        "gate1_candidate_bundle_sha256": release.get("gate1_candidate_bundle_sha256", "synthetic-fixture"),
        "synthetic_result_sha256": release.get("synthetic_result_sha256", "synthetic-fixture"),
        "release_sha256": release_hash,
        "salt": config["audit_selection"]["salt"]}
    completed_path, failed_path = destination / "COMPLETED.json", destination / "FAILED.json"
    if resume and (destination.is_symlink() or not destination.is_dir()): raise ValueError("resume output root is not an ordinary directory")
    if not resume:
        if destination.exists(): raise FileExistsError("EXP016 output root is create-once")
        destination.mkdir(parents=True)
        atomic_bytes(destination / "ATTEMPT.json", canonical_json({"format_version": 1, "status": "IN_PROGRESS", "bindings": bindings}))
    try:
        if source.is_symlink() or not source.is_file() or source_manifest.is_symlink() or not source_manifest.is_file():
            raise ValueError("discovery source and manifest must be ordinary non-symlink files")
        source_hash = sha256_file(source); manifest_hash = sha256_file(source_manifest)
        if source_hash != expected_source_sha256: raise ValueError("physical discovery source hash differs from release")
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        start = pd.Timestamp(config["discovery_window"]["start"]); end = pd.Timestamp(config["discovery_window"]["end_exclusive"])
        valid = manifest.get("output_sha256") == source_hash and int(manifest.get("rows", 0)) >= 1
        valid = valid and pd.Timestamp(manifest.get("start")) == start and pd.Timestamp(manifest.get("end")) < end and pd.Timestamp(manifest.get("end_exclusive")) == end
        valid = valid and manifest.get("current_session_volume_used") is False and manifest.get("sealed_rows_written") is False
        if not valid: raise ValueError("physical discovery manifest violates its frozen source or interval")
        protocol = DisplacementProtocol.from_file(protocol_path)
        if protocol.protocol_hash != bindings["protocol_sha256"]: raise ValueError("frozen displacement protocol hash changed")
        for (path, expected), resolved in zip(config["bindings"]["runtime_dependency_sha256"].items(), dependency_paths):
            if sha256_file(resolved) != expected: raise ValueError(f"frozen runtime dependency hash changed: {path}")
        if not _synthetic:
            custody = json.loads(custody_path.read_text(encoding="utf-8")); terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            identities = custody.get("identities", {}); audits = custody.get("audits", {})
            source_proof, output_proof, equality = audits.get("source", {}), audits.get("output", {}), audits.get("equality", {})
            expected_equalities = {"field_order", "arrow_schema_hex", "rows", "first_timestamp_ns", "last_timestamp_ns", "typed_row_commitment"}
            expected_denials = {"economic_evaluation_authorized", "future_or_action_authorized", "gate3_authorized", "mbo_authorized", "production_release_authorized", "sealed_data_authorized", "semantic_runner_authorized"}
            valid = custody.get("status") == "complete_exact_projection" and custody.get("classification") == "CUSTODY_CORE_EXACT_PROJECTION_VERIFIED"
            valid = valid and terminal.get("status") == terminal.get("classification") == "CUSTODY_GO_GATE3_NOT_AUTHORIZED" and terminal.get("reason") is None and terminal.get("core_returncode") == 0
            valid = valid and custody.get("experiment_id") == terminal.get("experiment_id") == EXPERIMENT_ID and custody.get("preregistration_sha256") == PREREGISTRATION_SHA
            valid = valid and sha256_file(custody_path) == release["custody_receipt_sha256"] == terminal.get("core_result_sha256") and sha256_file(terminal_path) == release["custody_terminal_sha256"]
            valid = valid and identities.get("output_sha256") == terminal.get("output_sha256") == source_hash and identities.get("output_manifest_sha256") == terminal.get("output_manifest_sha256") == manifest_hash
            valid = valid and identities.get("source_registered_sha256") == config["bindings"]["upstream_source_sha256"] and identities.get("materializer_registered_sha256") == config["bindings"]["materializer_sha256"]
            valid = valid and int(source_proof.get("rows", -1)) == int(output_proof.get("rows", -2)) == int(manifest["rows"])
            valid = valid and source_proof.get("first_timestamp_ns") == output_proof.get("first_timestamp_ns") == start.value and source_proof.get("last_timestamp_ns") == output_proof.get("last_timestamp_ns") == pd.Timestamp(manifest["end"]).value
            valid = valid and pd.Timestamp(source_proof.get("first_timestamp")) == pd.Timestamp(output_proof.get("first_timestamp")) == start and pd.Timestamp(source_proof.get("last_timestamp")) == pd.Timestamp(output_proof.get("last_timestamp")) == pd.Timestamp(manifest["end"])
            valid = valid and source_proof.get("field_order") == output_proof.get("field_order") == config["custody"]["required_field_order"] and source_proof.get("arrow_schema_hex") == output_proof.get("arrow_schema_hex")
            commitment = source_proof.get("typed_row_commitment"); valid = valid and _digest(commitment) and commitment == output_proof.get("typed_row_commitment")
            valid = valid and set(equality) == expected_equalities and all(value is True for value in equality.values())
            denials = custody.get("authority_denials", {}); valid = valid and set(denials) == expected_denials and all(value is False for value in denials.values())
            valid = valid and all(custody.get(name) == 0 for name in ("mbo_accesses", "current_session_volume_uses", "action_or_future_fields"))
            if not valid: raise ValueError("custody exact-projection receipt is invalid")
            if release["source_manifest_sha256"] != manifest_hash:
                raise PermissionError("Gate 3 release does not bind source custody")
        elif manifest_hash == "": raise ValueError("synthetic manifest hash is invalid")
    except Exception as exc:
        if destination.is_dir() and not completed_path.exists() and not failed_path.exists(): _preflight_fail(destination, bindings, exc)
        raise
    total_rows = int(manifest["rows"]); checkpoint = ReplayCheckpointStore(destination / "_checkpoint")
    if resume:
        attempt_bindings = bindings; trusted_cursor = None
        try:
            if completed_path.exists() or failed_path.exists(): raise ValueError("attempt is terminal or not resumable")
            attempt = _safe_json(destination / "ATTEMPT.json"); attempt_bindings = attempt.get("bindings")
            if attempt_bindings != bindings: raise ValueError("attempt bindings differ from requested resume")
            if checkpoint.exists:
                state = checkpoint.load(expected_bindings=bindings, expected_replay_type=CausalDisplacementEye); trusted_cursor = state["last_source_start"]; checkpoint_cursor = _safe_json(checkpoint.manifest_path).get("last_source_start")
                if checkpoint_cursor != (None if state["last_source_start"] is None else pd.Timestamp(state["last_source_start"]).isoformat()): raise ValueError("checkpoint manifest cursor disagrees with state")
                verify_stream_shards(destination, state["stream_state"])
            elif (destination / "progress.json").exists() or any((destination / "semantic_shards").glob("part-*.parquet")): raise ValueError("checkpoint-free attempt contains durable artifacts")
            else: state = _new_state(protocol)
            state["resume_count"] = int(state["resume_count"]) + 1
        except Exception as exc:
            if destination.is_dir() and not completed_path.exists() and not failed_path.exists(): _preflight_fail(destination, attempt_bindings, exc, trusted_cursor)
            raise
    else:
        state = _new_state(protocol)
    limit = int(config["runner"]["maximum_shard_rows"])
    if len(state["buffer"]) >= limit: raise ValueError("checkpoint semantic buffer is not bounded")
    started = time.monotonic(); session_rows = int(state["source_rows_consumed"])
    durable = _progress(state, total_rows, started, session_rows)

    def commit(flush=False):
        nonlocal durable
        if flush and state["buffer"]:
            write_stream_shards_bounded(destination, "semantic_shards", state["buffer"], state["stream_state"],
                key_column="record_id", maximum_rows=limit, field_types=FIELD_TYPES)
        state["last_checkpoint_source_rows"] = int(state["source_rows_consumed"]); _sync(state)
        checkpoint.save(state, bindings=bindings); durable = _progress(state, total_rows, started, session_rows)
        atomic_bytes(destination / "progress.json", canonical_json({**durable, "status": "running", "durable_checkpoint_only": True}))

    def fail(exc):
        payload = {**durable, "status": "failed_integrity", "failure_type": type(exc).__name__, "failure_message": str(exc)}
        atomic_bytes(destination / "progress.json", canonical_json(payload))
        checkpoint_manifest = _safe_json(checkpoint.manifest_path) if checkpoint.exists else {}
        checkpoint_hash = sha256_file(checkpoint.manifest_path) if checkpoint_manifest else None
        atomic_bytes(destination / "FAILED.json", canonical_json({"format_version": 1, "status": "failed_integrity",
            "bindings": bindings, "progress_sha256": sha256_file(destination / "progress.json"),
            "checkpoint_manifest_sha256": checkpoint_hash, "checkpoint_state_sha256": checkpoint_manifest.get("state_sha256"),
            "durable_source_rows": durable["source_rows_processed"], "durable_source_cursor": checkpoint_manifest.get("last_source_start"),
            "failure_type": type(exc).__name__, "failure_message": str(exc)}))

    safe = False
    try:
        reader, eye = state["reader"], state["replay"]
        for left, right in _year_windows(start, end):
            cursor = state["last_source_start"]
            if cursor is not None and pd.Timestamp(cursor) >= right: continue
            loaded = load_ohlcv(source, start=left if cursor is None else pd.Timestamp(cursor), end=right)
            if not loaded.contract_selection_causal or loaded.warnings: raise ValueError("discovery source lost causal contract provenance")
            for bar in iter_after_source_checkpoint(loaded.frame, cursor, allow_data_gap_reset=True):
                safe = False; source_real = not bar.synthetic_no_trade
                if source_real:
                    prior = state["last_source_start"]
                    if prior is not None and bar.start <= pd.Timestamp(prior): raise ValueError("real source cursor is not increasing")
                    state["last_source_start"] = bar.start; state["source_rows_consumed"] += 1
                update = reader.on_bar(bar); observation = eye.on_update(update); raw = eye.last_update
                if raw is None: raise ValueError("eye omitted its lossless raw update")
                if not update.newly_completed.get(Timeframe.M5) and not update.anomalies and raw.state != state["previous_open_state"]:
                    raise ValueError("displacement state drifted on intermediate M1")
                _audit_transitions(state, raw, observation)
                for row in _semantic_rows(update, observation, raw, state["last_source_start"]):
                    state["buffer"].append(row); state["peak_buffer_rows"] = max(int(state["peak_buffer_rows"]), len(state["buffer"]))
                    if len(state["buffer"]) == limit:
                        write_stream_shards_bounded(destination, "semantic_shards", state["buffer"], state["stream_state"],
                            key_column="record_id", maximum_rows=limit, field_types=FIELD_TYPES)
                _capture_candidates(state, update, observation, raw, source_bar_real=source_real,
                    protocol_sha=protocol.protocol_hash, salt=config["audit_selection"]["salt"])
                state["processed_bars"] += 1; state["last_asof"] = observation.asof; state["previous_open_state"] = raw.state; safe = source_real
                due = int(state["source_rows_consumed"]) == 1 or int(state["source_rows_consumed"]) - int(state["last_checkpoint_source_rows"]) >= int(config["runner"]["checkpoint_interval_rows"])
                if source_real and due: commit()
                if diagnostic_stop_after_source_rows > 0 and int(state["source_rows_consumed"]) >= diagnostic_stop_after_source_rows and source_real:
                    commit(); safe = False; raise KeyboardInterrupt("intentional EXP016 diagnostic interruption")
            del loaded
        if int(state["source_rows_consumed"]) != total_rows: raise ValueError("not every physical discovery source row was consumed")
        if state["last_asof"] is None or pd.Timestamp(state["last_asof"]) > end: raise ValueError("observation crossed exclusive cutoff")
        commit(flush=True); verified = _verify_stream(destination, state["stream_state"])
        stream_manifest = write_stream_manifest(destination, "semantic_shards", state["stream_state"],
            artifact="exp016_displacement_semantics", bindings=bindings)
        status, missing = _coverage(state); selected = sorted(state["selector_slots"].values(), key=lambda item: item["case_id"])
        census_path = destination / "census.json"; blind_path = destination / "blind_cases.json"; overlay_path = destination / "model_overlay.json"
        census = {**state["census"], "months": sorted(state["census"]["months"]), "missing_strata": missing}
        atomic_bytes(census_path, canonical_json(census))
        atomic_bytes(blind_path, canonical_json({"cases": [x["blind"] for x in selected]}))
        atomic_bytes(overlay_path, canonical_json({"cases": [x["overlay"] for x in selected]}))
        summary = {"status": status, "bindings": bindings, "source_rows": int(state["source_rows_consumed"]),
            "processed_bars": int(state["processed_bars"]), "semantic_rows": verified, "selected_cases": len(selected),
            "missing_strata": missing, "peak_buffer_rows": int(state["peak_buffer_rows"]),
            "resume_count": int(state["resume_count"]), "right_boundary_open_entity": state["previous_open_state"] is not None,
            "full_snapshot_hash_per_m1": False}
        summary_path = destination / "summary.json"; atomic_bytes(summary_path, canonical_json(summary))
        final_progress = {**_progress(state, total_rows, started, session_rows), "status": status, "resume_supported": False}
        atomic_bytes(destination / "progress.json", canonical_json(final_progress))
        progress_hash = sha256_file(destination / "progress.json")
        atomic_bytes(destination / "COMPLETED.json", canonical_json({"format_version": 1, "status": status,
            "bindings": bindings, "stream_manifest_sha256": sha256_file(stream_manifest), "census_sha256": sha256_file(census_path),
            "blind_cases_sha256": sha256_file(blind_path), "model_overlay_sha256": sha256_file(overlay_path),
            "summary_sha256": sha256_file(summary_path), "progress_sha256": progress_hash,
            "checkpoint_manifest_sha256": sha256_file(checkpoint.manifest_path)}))
        return summary
    except KeyboardInterrupt:
        if completed_path.exists(): raise
        if safe: commit()
        raise
    except Exception as exc:
        if completed_path.exists(): raise
        fail(exc); raise

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source-sha256", required=True); parser.add_argument("--release", required=True); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic-stop-after-source-rows", type=int, default=0); args = parser.parse_args()
    result = run(args.config, expected_source_sha256=args.source_sha256, release_path=args.release, resume=args.resume,
        diagnostic_stop_after_source_rows=args.diagnostic_stop_after_source_rows)
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
