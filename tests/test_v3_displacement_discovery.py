from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from smc_trader.calibration_replay import iter_after_source_checkpoint
from smc_trader.causal import ReaderUpdate
from smc_trader.displacement import DisplacementLifecycle, DisplacementProtocol
from smc_trader.displacement_observer import CausalDisplacementEye
from smc_trader.io import load_ohlcv
import smc_trader.displacement_discovery as discovery
from smc_trader.model import Candle, Timeframe


pytestmark = pytest.mark.skip(
    reason="archived EXP016 governance runner is outside the development flow"
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/EXP-SMC-3.0.2-016-DISPLACEMENT-PRODUCTION-IDENTITY-CUSTODY.json"
EXP015_CONTROL = ROOT / "configs/experiments/EXP-SMC-3.0.2-015-PARQUET-NAMED-TIMEZONE-FIXTURE-CLOSURE.json"
PROTOCOL = ROOT / "configs/experiments/EXP-SMC-3.0.2-004-CAUSAL-5M-DISPLACEMENT-DUAL-CLOCK.json"
RUNNER = ROOT / "smc_trader/displacement_discovery.py"
BASE = pd.Timestamp("2020-01-06 09:30:00", tz="America/New_York")


def _config() -> dict: return json.loads(CONFIG.read_text(encoding="utf-8"))


def _m5(index: int, values=(100.0, 101.0, 100.0, 101.0)) -> Candle:
    start = BASE + pd.Timedelta(minutes=5 * index)
    return Candle(Timeframe.M5, start, start + pd.Timedelta(minutes=5), *values, 100.0, "NQH0", 1, 5, 5, True, 5, 0)


def _update(asof: pd.Timestamp, *, m5=()) -> ReaderUpdate:
    minute = Candle(Timeframe.M1, asof - pd.Timedelta(minutes=1), asof, 100.0, 100.25, 99.75, 100.0, 100.0, "NQH0", 1, 1, 1, True, 1, 0)
    newly = {timeframe: () for timeframe in Timeframe}; histories = {timeframe: () for timeframe in Timeframe}
    newly[Timeframe.M1], newly[Timeframe.M5] = (minute,), tuple(m5); histories.update(newly)
    return ReaderUpdate(asof, minute, newly, histories, ())


def _send(eye: CausalDisplacementEye, candle: Candle): return eye.on_update(_update(candle.end, m5=(candle,)))


def _started(*, atr=1.0, seed=(100.0, 101.0, 100.0, 101.0)):
    eye = CausalDisplacementEye(DisplacementProtocol.from_file(PROTOCOL))
    for index in range(15):
        _send(eye, _m5(index, (100.0, 100.0 + atr, 100.0, 100.0)))
    candle = _m5(15, seed); assert _send(eye, candle).lifecycle == "started"; return eye, 16, candle


def _runner_fixture(tmp_path: Path, name: str):
    tmp_path = tmp_path.resolve()
    index = pd.date_range(BASE, periods=90, freq="min"); opens = [100.0 + .5 * (i - 75) if 75 <= i < 80 else (102.5 if i >= 80 else 100.0) for i in range(90)]
    closes = [value + .5 if 75 <= i < 80 else value for i, value in enumerate(opens)]; highs = [max(left, right) + (.25 if i < 75 or i >= 80 else 0.0) for i, (left, right) in enumerate(zip(opens, closes))]
    frame = pd.DataFrame({"open": opens, "high": highs, "low": [min(x) for x in zip(opens, closes)], "close": closes, "volume": [100.0] * 90, "symbol": ["NQH0"] * 90, "instrument_id": [1] * 90}, index=index); frame.index.name = "ts"
    source = tmp_path / "synthetic_source.parquet"; frame.to_parquet(source); source_sha = discovery.sha256_file(source)
    assert pq.read_schema(source).field("ts").type == pa.timestamp("ns", tz="America/New_York")
    end = index[-1] + pd.Timedelta(minutes=1); manifest = source.with_suffix(".parquet.manifest.json")
    manifest.write_text(json.dumps({"output_sha256": source_sha, "rows": 90, "start": index[0].isoformat(), "end": index[-1].isoformat(), "end_exclusive": end.isoformat(), "current_session_volume_used": False, "sealed_rows_written": False}), encoding="utf-8")
    cfg = _config(); cfg["discovery_window"].update({"start": index[0].isoformat(), "end_exclusive": end.isoformat()})
    output = tmp_path / name; cfg["runner"].update({"discovery_source_path": str(source), "discovery_manifest_path": str(manifest), "output_root": str(output), "checkpoint_interval_rows": 13, "maximum_shard_rows": 1})
    config = tmp_path / f"{name}.json"; config.write_text(json.dumps(cfg), encoding="utf-8")
    return config, source_sha, output


def _gate3_release(config: Path, source_sha: str, tmp_path: Path) -> Path:
    tmp_path = tmp_path.resolve(); cfg = json.loads(config.read_text(encoding="utf-8")); manifest_path = Path(cfg["runner"]["discovery_manifest_path"]); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commitment, fields, schema = "c" * 64, cfg["custody"]["required_field_order"], "synthetic-arrow-schema"
    proof = {"rows": manifest["rows"], "first_timestamp": manifest["start"], "first_timestamp_ns": pd.Timestamp(manifest["start"]).value, "last_timestamp": manifest["end"], "last_timestamp_ns": pd.Timestamp(manifest["end"]).value, "field_order": fields, "arrow_schema_hex": schema, "typed_row_commitment": commitment}
    denials = {"economic_evaluation_authorized", "future_or_action_authorized", "gate3_authorized", "mbo_authorized", "production_release_authorized", "sealed_data_authorized", "semantic_runner_authorized"}
    custody = tmp_path / "custody.json"; terminal = tmp_path / "custody-terminal.json"
    receipt = {"status": "complete_exact_projection", "classification": "CUSTODY_CORE_EXACT_PROJECTION_VERIFIED", "experiment_id": cfg["experiment_id"], "preregistration_sha256": cfg["bindings"]["preregistration_sha256"],
               "identities": {"output_sha256": source_sha, "output_manifest_sha256": discovery.sha256_file(manifest_path), "source_registered_sha256": cfg["bindings"]["upstream_source_sha256"], "materializer_registered_sha256": cfg["bindings"]["materializer_sha256"]},
               "audits": {"source": proof, "output": dict(proof), "equality": {name: True for name in ("field_order", "arrow_schema_hex", "rows", "first_timestamp_ns", "last_timestamp_ns", "typed_row_commitment")}},
               "authority_denials": {name: False for name in denials}, "mbo_accesses": 0, "current_session_volume_uses": 0, "action_or_future_fields": 0}
    custody.write_text(json.dumps(receipt), encoding="utf-8")
    terminal_payload = {"status": "CUSTODY_GO_GATE3_NOT_AUTHORIZED", "classification": "CUSTODY_GO_GATE3_NOT_AUTHORIZED", "reason": None, "core_returncode": 0, "experiment_id": cfg["experiment_id"], "core_result_sha256": discovery.sha256_file(custody), "output_sha256": source_sha, "output_manifest_sha256": discovery.sha256_file(manifest_path)}
    terminal.write_text(json.dumps(terminal_payload), encoding="utf-8")
    cfg["custody"].update({"core_result_path": str(custody), "terminal_path": str(terminal)})
    cfg["bindings"].update({"gate2_core_result_sha256": discovery.sha256_file(custody), "gate2_terminal_sha256": discovery.sha256_file(terminal), "gate2_output_sha256": source_sha, "gate2_output_manifest_sha256": discovery.sha256_file(manifest_path)})
    config.write_text(json.dumps(cfg), encoding="utf-8"); release = tmp_path / "gate3-release.json"
    payload = {"status": "gate3_real_discovery_authorized", "execution_authorized": True, "experiment_id": cfg["experiment_id"], "preregistration_sha256": cfg["bindings"]["preregistration_sha256"], "base_config_sha256": discovery.sha256_file(config), "source_sha256": source_sha, "source_manifest_sha256": discovery.sha256_file(manifest_path), "custody_receipt_sha256": discovery.sha256_file(custody), "custody_terminal_sha256": discovery.sha256_file(terminal), "candidate_bundle_sha256": cfg["candidate"]["implementation_component_bundle_sha256"], "gate1_candidate_bundle_sha256": cfg["bindings"]["gate1_attempt002_candidate_bundle_sha256"], "synthetic_result_sha256": cfg["bindings"]["gate1_attempt002_terminal_sha256"], "runner_sha256": discovery.sha256_file(RUNNER), "observer_sha256": discovery.sha256_file(ROOT / "smc_trader/displacement_observer.py"), "real_attempt_limit": 1, "custody_receipt_path": str(custody), "custody_terminal_path": str(terminal), "output_root": cfg["runner"]["output_root"]}
    release.write_text(json.dumps(payload), encoding="utf-8"); return release


def test_exp014_contract_is_single_role_and_data_locked(tmp_path: Path) -> None:
    cfg = _config(); roles, runner = cfg["data_roles"], cfg["runner"]; source = RUNNER.read_text(encoding="utf-8")
    assert cfg["claim_boundary"]["role"] == "development_discovery_not_oof" and runner["discovery_source_path"] == roles["physical_discovery"]["path"]
    assert roles["upstream_multi_role"]["runner_access"] == "denied" and all(roles["denied"].values()) and cfg["authority"]["ohlcv_authorized"] is False
    assert cfg["discovery_window"]["end_exclusive"].startswith("2022-01-01") and all(name in source for name in ("candidate_bundle_sha256", "synthetic_result_sha256", "custody_receipt_path", "runtime_dependency_sha256"))
    config, source_sha, output = _runner_fixture(tmp_path, "confined"); payload = json.loads(config.read_text()); payload["runner"]["discovery_manifest_path"] = str(ROOT / "data/forbidden-exp014-manifest.json"); config.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="synthetic"): discovery.run(config, expected_source_sha256=source_sha, _synthetic=True)
    assert not output.exists()


def test_exp014_eye_exposes_exact_lossless_raw_update() -> None:
    eye = CausalDisplacementEye(DisplacementProtocol.from_file(PROTOCOL)); assert eye.last_update is None
    eye, _, _ = _started()
    raw, observation = eye.last_update, eye.last_observation; assert raw is not None and observation is not None
    assert raw.state == eye.tracker.snapshot() and tuple(item.transition_id for item in raw.transitions) == (observation.latest_transition.transition_id,)


def test_exp014_activation_timeout_terminal_metrics_are_observer_emitted() -> None:
    eye, index, _ = _started(atr=4.0, seed=(100.0, 103.25, 100.0, 103.25))
    values = ((103.25, 103.5, 103.25, 103.5), (103.5, 103.75, 103.5, 103.75), (103.75, 104.0, 103.75, 104.0))
    for offset, value in enumerate(values):
        terminal_candle = _m5(index + offset, value)
        _send(eye, terminal_candle)
    raw = eye.last_update
    assert raw is not None and raw.state is None; terminal = raw.transitions[0].state
    assert (terminal.lifecycle, terminal.terminal_reason, terminal.real_episode_bar_count) == (DisplacementLifecycle.EXHAUSTED, "activation_window_elapsed", 4)
    assert (terminal.prefix_last_admitted_at, terminal.terminal_at, terminal.observed_at) == (terminal_candle.end,) * 3


def test_exp014_same_clock_transition_order_is_lossless() -> None:
    eye, index, _ = _started(); candle = _m5(index, (101.0, 101.0, 99.0, 99.0))
    _send(eye, candle)
    raw = eye.last_update
    assert raw is not None
    exhausted, started = raw.transitions
    assert (exhausted.state.lifecycle, started.state.lifecycle) == (DisplacementLifecycle.EXHAUSTED, DisplacementLifecycle.STARTED)
    assert exhausted.state.observed_at == started.state.observed_at == candle.end
    assert exhausted.state.entity_id != started.state.entity_id


def test_exp014_intermediate_m1_is_continuous_without_state_drift() -> None:
    eye, _, seed = _started()
    state, observation = eye.last_update.state, eye.last_observation
    first = eye.on_update(_update(seed.end + pd.Timedelta(minutes=1)))
    second = eye.on_update(_update(seed.end + pd.Timedelta(minutes=2)))
    assert eye.last_update.state == state and eye.last_update.transitions == ()
    assert replace(observation, asof=first.asof) == first
    assert replace(observation, asof=second.asof) == second


def test_exp014_semantic_stream_is_fixed_bounded_and_future_free() -> None:
    eye, _, seed = _started()
    update, raw = _update(seed.end, m5=(seed,)), eye.last_update
    rows = discovery._semantic_rows(update, eye.last_observation, raw, seed.start); forbidden = set(_config()["runner"]["forbidden_fields"])
    assert 0 < len(rows) <= len(raw.transitions) + 1
    assert all(not forbidden.intersection(row) and row["provenance"] == "observer_raw_update" for row in rows)
    assert all(pd.Timestamp(row["observation_clock"]) <= update.asof and len(row["state_sha256"]) == 64 for row in rows)


def test_exp014_selector_is_deterministic_prefix_only_and_bounded() -> None:
    cfg, stratum = _config(), "2017:long:started"
    salt, protocol = cfg["audit_selection"]["salt"], cfg["bindings"]["protocol_sha256"]
    expected = lambda identity: hashlib.sha256(f"{salt}{protocol}{stratum}{identity}".encode()).hexdigest()
    left, right = {}, {}
    for identity in ("c", "a", "b"):
        discovery._consider(left, stratum, discovery._priority(salt, protocol, stratum, identity), {"case_id": identity}, 40)
    for identity in ("b", "a", "c"):
        discovery._consider(right, stratum, expected(identity), {"case_id": identity}, 40)
    assert left == right
    assert min(("a", "b", "c"), key=expected) in json.dumps(left)
    overflow = {}
    with pytest.raises(ValueError):
        for index in range(41):
            discovery._consider(overflow, str(index), str(index), {"case_id": index}, 40)


def test_exp014_missing_stratum_is_unavailable_without_substitution() -> None:
    selection = _config()["audit_selection"]
    assert (selection["required_cells"], selection["maximum_candidates"]) == (40, 40)
    assert selection["missing_cell_status"] == "complete_semantic_unavailable"
    assert selection["substitution_allowed"] is False
    assert "complete_semantic_unavailable" in RUNNER.read_text(encoding="utf-8")


def test_exp014_year_partition_and_end_exclusive_are_causal() -> None:
    window = _config()["discovery_window"]
    start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end_exclusive"])
    years = discovery._year_windows(start, end)
    assert len(years) == 5 and years[0][0] == start and years[-1][1] == end and all(left < right for left, right in years) and all(left[1] == right[0] for left, right in zip(years, years[1:])) and "del loaded" in RUNNER.read_text()


def test_exp015_named_timezone_schema_year_dst_and_resume_contract(tmp_path: Path) -> None:
    zone = "America/New_York"
    index = pd.DatetimeIndex([pd.Timestamp(value, tz=zone) for value in ("2020-12-31 23:59", "2021-01-01 00:00", "2021-03-12 16:59", "2021-03-14 18:00", "2021-11-05 16:59", "2021-11-07 18:00")])
    frame = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0, "symbol": "NQH1", "instrument_id": 1}, index=index)
    frame.index.name = "ts"; source = tmp_path / "named_timezone_source.parquet"; frame.to_parquet(source)
    assert pq.read_schema(source).field("ts").type == pa.timestamp("ns", tz=zone)
    annual = load_ohlcv(source, start=pd.Timestamp("2021-01-01", tz=zone), end=pd.Timestamp("2022-01-01", tz=zone))
    assert annual.contract_selection_causal and not annual.warnings and tuple(annual.frame.index) == tuple(index[1:])
    assert tuple(int(stamp.utcoffset().total_seconds() // 3600) for stamp in index[2:]) == (-5, -4, -4, -5)
    resumed = tuple(iter_after_source_checkpoint(annual.frame, index[2], allow_data_gap_reset=True))
    assert tuple((bar.start, bar.synthetic_no_trade) for bar in resumed) == tuple((stamp, False) for stamp in index[3:])


@pytest.mark.historical_frozen
def test_exp014_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    direct_cfg, source_sha, direct_root = _runner_fixture(tmp_path, "direct")
    direct = discovery.run(direct_cfg, expected_source_sha256=source_sha, _synthetic=True)
    resume_cfg, source_sha, resume_root = _runner_fixture(tmp_path, "resumed")
    with pytest.raises(KeyboardInterrupt):
        discovery.run(resume_cfg, expected_source_sha256=source_sha,
                      diagnostic_stop_after_source_rows=82, _synthetic=True)
    resumed = discovery.run(resume_cfg, expected_source_sha256=source_sha, resume=True, _synthetic=True)
    keys = ("status", "source_rows", "processed_bars", "semantic_rows", "selected_cases",
            "missing_strata", "peak_buffer_rows", "right_boundary_open_entity")
    assert {key: direct[key] for key in keys} == {key: resumed[key] for key in keys}
    assert (direct["resume_count"], resumed["resume_count"]) == (0, 1)
    direct_shards = sorted((direct_root / "semantic_shards").glob("*.parquet"))
    resumed_shards = sorted((resume_root / "semantic_shards").glob("*.parquet"))
    assert [path.read_bytes() for path in direct_shards] == [path.read_bytes() for path in resumed_shards]
    for name in ("census.json", "blind_cases.json", "model_overlay.json"):
        assert (direct_root / name).read_bytes() == (resume_root / name).read_bytes()


@pytest.mark.historical_frozen
def test_exp014_resume_rejects_binding_or_shard_tamper(tmp_path: Path) -> None:
    config, source_sha, output = _runner_fixture(tmp_path, "tamper")
    with pytest.raises(KeyboardInterrupt):
        discovery.run(config, expected_source_sha256=source_sha,
                      diagnostic_stop_after_source_rows=82, _synthetic=True)
    bindings = json.loads((output / "ATTEMPT.json").read_text(encoding="utf-8"))["bindings"]; shard = next((output / "semantic_shards").glob("*.parquet"))
    with pytest.raises(ValueError, match="bindings"): discovery.ReplayCheckpointStore(output / "_checkpoint").load(expected_bindings={**bindings, "salt": "changed"}, expected_replay_type=CausalDisplacementEye)
    cursor_config, cursor_sha, cursor_output = _runner_fixture(tmp_path, "cursor-tamper")
    with pytest.raises(KeyboardInterrupt): discovery.run(cursor_config, expected_source_sha256=cursor_sha, diagnostic_stop_after_source_rows=82, _synthetic=True)
    cursor_manifest = cursor_output / "_checkpoint/manifest.json"; cursor_payload = json.loads(cursor_manifest.read_text()); trusted_cursor = cursor_payload["last_source_start"]; cursor_payload["last_source_start"] = BASE.isoformat(); cursor_manifest.write_text(json.dumps(cursor_payload))
    with pytest.raises(ValueError, match="cursor"): discovery.run(cursor_config, expected_source_sha256=cursor_sha, resume=True, _synthetic=True)
    assert json.loads((cursor_output / "FAILED.json").read_text())["durable_source_cursor"] == trusted_cursor != BASE.isoformat()
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        discovery.run(config, expected_source_sha256=source_sha, resume=True, _synthetic=True)
    assert (output / "FAILED.json").is_file() and not (output / "COMPLETED.json").exists()


@pytest.mark.historical_frozen
def test_exp014_progress_and_completion_follow_durable_order(tmp_path: Path) -> None:
    config, source_sha, output = _runner_fixture(tmp_path, "complete")
    discovery.run(config, expected_source_sha256=source_sha, release_path=_gate3_release(config, source_sha, tmp_path))
    progress = (output / "progress.json").read_bytes(); completed = json.loads((output / "COMPLETED.json").read_text(encoding="utf-8"))
    assert completed["progress_sha256"] == hashlib.sha256(progress).hexdigest()
    assert json.loads(progress)["completed_percent"] == 100.0 and not (output / "FAILED.json").exists()
    source = RUNNER.read_text(encoding="utf-8")
    assert source.rfind("COMPLETED.json") > source.rfind("progress.json")
    assert source.rfind("FAILED.json") > source.find("progress.json")


@pytest.mark.historical_frozen
def test_exp014_forbidden_import_fields_and_loc_budget() -> None:
    cfg, control, source = _config(), json.loads(EXP015_CONTROL.read_text(encoding="utf-8")), RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {node.module.split(".")[-1] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported.intersection({"decision", "risk", "playbooks", "playbook_registry", "visualization"})
    assert {"run", "main"}.issubset({node.name for node in tree.body if isinstance(node, ast.FunctionDef)})
    effective = lambda path: sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in path.read_text().splitlines())
    assert effective(RUNNER) <= cfg["resources"]["displacement_discovery_effective_line_ceiling"]
    assert control["experiment_id"] == "EXP-SMC-3.0.2-015-PARQUET-NAMED-TIMEZONE-FIXTURE-CLOSURE" and control["resources"]["test_effective_line_ceiling"] == 205 and cfg["resources"]["test_effective_line_ceiling"] == 254 and cfg["resources"]["shared_discovery_test_changed_effective_line_ceiling"] == 29 and effective(Path(__file__)) <= 254
    names = [node.name for node in ast.parse(Path(__file__).read_text()).body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_exp014_")]
    assert cfg["synthetic"]["frozen_exp014_nodes"] == [
        f"tests/test_v3_displacement_discovery.py::{name}" for name in names]


@pytest.mark.historical_frozen
def test_exp016_execution_identity_is_separate_from_frozen_semantics(tmp_path: Path) -> None:
    cfg = _config(); exp014_path = ROOT / "configs/experiments/EXP-SMC-3.0.2-014-CAUSAL-5M-DISPLACEMENT-SEMANTIC-DISCOVERY.json"; exp014 = json.loads(exp014_path.read_text(encoding="utf-8"))
    assert (cfg["semantic_contract_origin"], cfg["audit_case_identity_origin"]) == ("EXP014", "EXP014-case") and "exp016" in cfg["runner"]["output_root"].lower()
    assert cfg["audit_selection"] == exp014["audit_selection"]
    assert (discovery.EXPERIMENT_ID, discovery.PREREGISTRATION_SHA, discovery.DEFAULT_CONFIG) == (cfg["experiment_id"], cfg["bindings"]["preregistration_sha256"], CONFIG)
    assert cfg["bindings"]["semantic_contract_control_path"] == str(exp014_path.relative_to(ROOT)) and cfg["bindings"]["immediate_predecessor_control_path"] == str(EXP015_CONTROL.relative_to(ROOT)) and cfg["bindings"]["predecessor_archive_terminal_sha256"] == "1a84cf9365dd5e244af7464ce0fd6a69d5dcf05d5f4cf6a18d2f681c07cfc876"
    predecessor = (ROOT / "archive/experiments/EXP-SMC-3.0.2-015-exp014-terminal-candidate-001/smc_trader/displacement_discovery.py").read_text(encoding="utf-8"); normalized = RUNNER.read_text(encoding="utf-8")
    substitutions = (("Bounded, resumable EXP016 displacement semantic discovery.", "Bounded, resumable EXP014 displacement semantic discovery."), ("configs/experiments/EXP-SMC-3.0.2-016-DISPLACEMENT-PRODUCTION-IDENTITY-CUSTODY.json", "configs/experiments/EXP-SMC-3.0.2-014-CAUSAL-5M-DISPLACEMENT-SEMANTIC-DISCOVERY.json"), ("EXP-SMC-3.0.2-016-DISPLACEMENT-PRODUCTION-IDENTITY-CUSTODY", exp014["experiment_id"]),
                     (cfg["bindings"]["preregistration_sha256"], exp014["bindings"]["preregistration_sha256"]), ("EXP016 config identity is not frozen", "EXP014 config identity is not frozen"), ("EXP016 output root is create-once", "EXP014 output root is create-once"), ("intentional EXP016 diagnostic interruption", "intentional EXP014 diagnostic interruption"), ("exp016_displacement_semantics", "exp014_displacement_semantics"))
    for current, prior in substitutions:
        assert current in normalized; normalized = normalized.replace(current, prior)
    assert normalized[:normalized.index("def _safe_path")] == predecessor[:predecessor.index("def _json")]
    assert normalized[normalized.index("def _json"):normalized.index("def run")] == predecessor[predecessor.index("def _json"):predecessor.index("def run")]
    assert normalized[normalized.index("    total_rows ="):] == predecessor[predecessor.index("    total_rows ="):]
    assert all(name in normalized[normalized.index("def run"):normalized.index("    total_rows =")] for name in ("custody_terminal_path", "gate1_candidate_bundle_sha256", "authority_denials"))
    salt, protocol, stratum, identity, priority = cfg["audit_selection"]["salt"], cfg["bindings"]["protocol_sha256"], "2017:long:started", "frozen-prefix", "a" * 64
    assert discovery._priority(salt, protocol, stratum, identity) == hashlib.sha256(f"{exp014['audit_selection']['salt']}{protocol}{stratum}{identity}".encode()).hexdigest() and discovery._case_id(priority) == hashlib.sha256(f"EXP014-case{priority}".encode()).hexdigest()
    outside = tmp_path.resolve() / "ancestor-out"; outside.mkdir(); (outside / "leaf.json").write_text("{}"); link = tmp_path.resolve() / "ancestor-link"; link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="ancestor"): discovery._safe_path(link / "leaf.json", leaf=True)
    identities = ((exp014["experiment_id"], exp014["bindings"]["preregistration_sha256"], True), ("WRONG-EXPERIMENT", cfg["bindings"]["preregistration_sha256"], False), (cfg["experiment_id"], "0" * 64, False))
    for index, (experiment_id, preregistration_sha, predecessor_config) in enumerate(identities):
        config, source_sha, output = _runner_fixture(tmp_path, f"identity-reject-{index}"); template = json.loads(config.read_text(encoding="utf-8"))
        payload = json.loads(json.dumps(exp014)) if predecessor_config else template; payload["runner"].update(template["runner"])
        payload["experiment_id"] = experiment_id; payload["bindings"]["preregistration_sha256"] = preregistration_sha; config.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="EXP016 config identity"):
            discovery.run(config, expected_source_sha256=source_sha, _synthetic=True)
        assert not output.exists()
    for index, field in enumerate(("candidate_bundle_sha256", "gate1_candidate_bundle_sha256", "synthetic_result_sha256")):
        config, source_sha, output = _runner_fixture(tmp_path, f"binding-reject-{index}"); release = _gate3_release(config, source_sha, tmp_path); payload = json.loads(release.read_text()); payload[field] = "0" * 64; release.write_text(json.dumps(payload))
        with pytest.raises(PermissionError, match="release bindings"): discovery.run(config, expected_source_sha256=source_sha, release_path=release)
        assert not output.exists()
    for index, field in enumerate(("source_sha256", "source_manifest_sha256")):
        config, source_sha, output = _runner_fixture(tmp_path, f"source-binding-reject-{index}"); release = _gate3_release(config, source_sha, tmp_path); payload = json.loads(release.read_text()); payload[field] = "0" * 64; release.write_text(json.dumps(payload))
        with pytest.raises(PermissionError, match="release bindings"): discovery.run(config, expected_source_sha256=("0" * 64 if field == "source_sha256" else source_sha), release_path=release)
        assert not output.exists()
    config, source_sha, output = _runner_fixture(tmp_path, "ancestor-binding-reject"); release = _gate3_release(config, source_sha, tmp_path); payload = json.loads(release.read_text()); config_payload = json.loads(config.read_text()); run_link = tmp_path.resolve() / "run-ancestor-link"; run_link.symlink_to(tmp_path.resolve(), target_is_directory=True)
    config_payload["custody"].update({"core_result_path": str(run_link / "custody.json"), "terminal_path": str(run_link / "custody-terminal.json")}); config.write_text(json.dumps(config_payload)); payload.update({"base_config_sha256": discovery.sha256_file(config), "custody_receipt_path": config_payload["custody"]["core_result_path"], "custody_terminal_path": config_payload["custody"]["terminal_path"]}); release.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="ancestor"): discovery.run(config, expected_source_sha256=source_sha, release_path=release)
    assert not output.exists()
    config, source_sha, output = _runner_fixture(tmp_path, "identity-valid"); release = _gate3_release(config, source_sha, tmp_path)
    discovery.run(config, expected_source_sha256=source_sha, release_path=release)
    attempt = json.loads((output / "ATTEMPT.json").read_text(encoding="utf-8")); release_payload = json.loads(release.read_text(encoding="utf-8")); custody_path = Path(release_payload["custody_receipt_path"]); terminal_path = Path(release_payload["custody_terminal_path"]); custody = json.loads(custody_path.read_text(encoding="utf-8")); terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    stream = json.loads((output / "semantic_shards.manifest.json").read_text(encoding="utf-8"))
    assert all(payload["experiment_id"] == cfg["experiment_id"] and payload["preregistration_sha256"] == cfg["bindings"]["preregistration_sha256"] for payload in (attempt["bindings"], release_payload, custody))
    assert terminal["experiment_id"] == cfg["experiment_id"] and terminal["core_result_sha256"] == discovery.sha256_file(custody_path)
    assert attempt["bindings"]["release_sha256"] == discovery.sha256_file(release) and attempt["bindings"]["custody_receipt_sha256"] == discovery.sha256_file(custody_path) and attempt["bindings"]["custody_terminal_sha256"] == discovery.sha256_file(terminal_path) and stream["artifact"] == "exp016_displacement_semantics"
