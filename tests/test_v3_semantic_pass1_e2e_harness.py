from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.market_clock import is_registered_trading_minute
from smc_trader.model import Timeframe
from smc_trader.observation import CausalObserver, ObserverConfig
from smc_trader.semantic_discovery_runner import (
    SemanticDiscoveryReplay,
    SourceChunk,
    iter_provenanced_completed_bars,
)


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/benchmark_v3_semantic_pass1_e2e.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "benchmark_v3_semantic_pass1_e2e_under_test",
        HARNESS,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bootstrap_hash_matches_bound_artifact_helper() -> None:
    harness = _load_harness()
    for path in (
        HARNESS,
        ROOT / "smc_trader/artifact_stream.py",
        ROOT / "smc_trader/semantic_discovery_runner.py",
    ):
        assert harness._bootstrap_sha256_file(path) == (
            harness.sha256_file(path)
        )


def test_harness_binds_current_r3_audit_closure_identity(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    assert harness.ACTIVE_EXPERIMENT_ID == (
        "EXP-SMC-3.0.1-001-STRUCTURE-BOS-AUDIT-CLOSURE-R3"
    )
    assert harness.MODEL_CONFIG == (
        ROOT / "configs/model_v3_0_1_exp001_structure_bos_identity_r3.json"
    )
    assert harness.PRIMITIVE_PROTOCOL == (
        ROOT
        / "configs/"
        "smc_primitives_v3_0_1_structure_bos_audit_closure_r3.json"
    )
    source_hash = "a" * 64
    audit, runner, _ = harness._base_contracts(
        tmp_path,
        source_path=tmp_path / "synthetic-source.parquet",
        source_sha256=source_hash,
        source_start=pd.Timestamp(
            "2024-01-08 18:00:00",
            tz="America/New_York",
        ),
        source_end=pd.Timestamp(
            "2024-01-09 17:00:00",
            tz="America/New_York",
        ),
        calendar_years=(2024,),
        batch_rows=64,
    )
    prefix = (
        "EXP-SMC-3.0.1-001-STRUCTURE-BOS-AUDIT-CLOSURE-R3-"
        "SYNTHETIC-FULL-E2E-"
    )
    assert audit["audit_id"] == f"{prefix}{source_hash[:16]}"
    assert runner["runner_id"] == (
        f"{prefix}RUNNER-{source_hash[:16]}"
    )
    assert runner["bindings"]["model_config_sha256"] == _sha(
        harness.MODEL_CONFIG
    )
    assert runner["bindings"]["primitive_protocol_sha256"] == _sha(
        harness.PRIMITIVE_PROTOCOL
    )


def test_confirmatory_prepare_rejects_long_expiry_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    registry = artifact_root / "pair"
    registry.mkdir()
    monkeypatch.setattr(harness, "ARTIFACT_ROOT", artifact_root)
    expiry = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=25)
    args = argparse.Namespace(
        fixture_root=str(artifact_root / "fixture"),
        expires_at=expiry.isoformat(),
        confirmatory_one_time=True,
        confirmatory_attempt_id="TTL-FAIL-R1",
        confirmatory_registry_root=str(registry),
    )
    with pytest.raises(ValueError, match="within 24 hours"):
        harness.prepare(args)
    assert not Path(args.fixture_root).exists()


def test_generic_run_rejects_one_time_release_before_runner_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    artifact_root = tmp_path / "artifacts"
    fixture_root = artifact_root / "fixture"
    fixture_root.mkdir(parents=True)
    release_path = fixture_root / "release.json"
    output = fixture_root / "output"
    release_path.write_text(
        json.dumps(
            {
                "confirmatory_one_time": True,
                "output": {"path": str(output)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(harness, "ARTIFACT_ROOT", artifact_root)
    touched = {"runner": False}

    def forbidden_load(*_args, **_kwargs):
        touched["runner"] = True
        raise AssertionError("one-time generic run loaded a source runner")

    monkeypatch.setattr(harness, "_load_bound_runner", forbidden_load)
    args = argparse.Namespace(
        fixture_root=str(fixture_root),
        release=str(release_path),
        result=str(fixture_root / "result.json"),
        resume=False,
        diagnostic_stop_after_source_rows=0,
        expected_outcome="complete",
        monitor_sample_seconds=0.1,
    )
    with pytest.raises(PermissionError, match="confirm-pair"):
        harness.run_benchmark(args)
    assert touched["runner"] is False
    assert not output.exists()


def test_run_benchmark_starts_monitor_before_source_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    artifact_root = tmp_path / "artifacts"
    fixture_root = artifact_root / "fixture"
    fixture_root.mkdir(parents=True)
    release_path = fixture_root / "release.json"
    output = fixture_root / "output"
    release_path.write_text(
        json.dumps(
            {
                "confirmatory_one_time": False,
                "output": {"path": str(output)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(harness, "ARTIFACT_ROOT", artifact_root)
    events: list[str] = []

    class Monitor:
        progress_percentages: list[float] = []

        def __init__(self, **_kwargs):
            pass

        def start(self):
            events.append("monitor")

        def finish(self):
            events.append("finish")

    def source_loader(*_args, **_kwargs):
        events.append("source_constructor")
        raise RuntimeError("injected source boundary")

    monkeypatch.setattr(harness, "_Monitor", Monitor)
    monkeypatch.setattr(harness, "_load_bound_runner", source_loader)
    args = argparse.Namespace(
        fixture_root=str(fixture_root),
        release=str(release_path),
        result=str(fixture_root / "result.json"),
        resume=False,
        diagnostic_stop_after_source_rows=0,
        expected_outcome="complete",
        monitor_sample_seconds=0.1,
    )
    with pytest.raises(RuntimeError, match="source boundary"):
        harness.run_benchmark(args)
    assert events[:2] == ["monitor", "source_constructor"]
    assert events[-1] == "finish"


def test_monitor_samples_synchronously_before_source_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    monitor = harness._Monitor(
        output_root=tmp_path / "output",
        sample_seconds=1.0,
    )
    events: list[str] = []
    monkeypatch.setattr(
        monitor,
        "_sample",
        lambda: events.append("sample"),
    )
    monitor.start()
    events.append("source_constructor")
    monitor.finish()
    assert events[0] == "sample"
    assert events.index("sample") < events.index("source_constructor")


def _passing_gate_result(harness, output: Path) -> dict:
    output.mkdir()
    (output / "_checkpoint").mkdir()
    (output / "progress.json").write_text("{}", encoding="utf-8")
    (output / "selection_authority.json").write_text(
        "{}",
        encoding="utf-8",
    )
    rows = harness.CONFIRMATORY_GATES["source_rows"]
    return {
        "observed_outcome": "complete",
        "resume": False,
        "diagnostic_stop_after_source_rows": 0,
        "rss_peak_bytes": (
            harness.CONFIRMATORY_GATES[
                "rss_bytes_strictly_less_than"
            ]
            - 1
        ),
        "disk_free_min_bytes": harness.CONFIRMATORY_GATES[
            "disk_free_bytes_at_least"
        ],
        "source_rows": rows,
        "admitted_source_rows": rows,
        "session_source_rows": rows,
        "checkpoint_status": "valid",
        "checkpoint_state_sha256": "a" * 64,
        "observed_progress_statuses": ["running", "complete"],
        "observed_progress_min_percent": 0.0,
        "observed_progress_max_percent": 100.0,
        "monitor_sample_count": 1,
        "terminal": {
            "source_rows": rows,
            "real_bars": rows,
            "synthetic_bars": 1,
            "reset_epoch": 1,
            "produced_bars": rows + 1,
            "progress_status": "complete",
            "selection_status": "complete",
            "selected_case_count": 1,
            "expected_case_count": 1,
            "missing_bucket_count": 0,
            "terminal_reconciled": True,
            "terminal_source_guard": "FrozenParquetSourceIdentity",
            "terminal_source_method_calls": 0,
            "completed_percent": 100.0,
            "source_prefix_root": "b" * 64,
            "produced_bar_prefix_root": "c" * 64,
        },
    }


def _passing_gate_spec(harness, output: Path) -> dict:
    rows = harness.CONFIRMATORY_GATES["source_rows"]
    return {
        "output_path": str(output),
        "expected_synthetic_bars": 1,
        "expected_reset_epochs": 1,
        "expected_produced_bars": rows + 1,
    }


def test_confirmatory_hard_gate_boundaries_are_fail_closed(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    output = tmp_path / "output"
    passing = _passing_gate_result(harness, output)
    spec = _passing_gate_spec(harness, output)
    assert harness._confirmatory_result_errors(
        passing,
        spec=spec,
        effective_wall_seconds=3_599.999,
    ) == []

    rss_boundary = copy.deepcopy(passing)
    rss_boundary["rss_peak_bytes"] = harness.CONFIRMATORY_GATES[
        "rss_bytes_strictly_less_than"
    ]
    assert "RSS gate failed" in harness._confirmatory_result_errors(
        rss_boundary,
        spec=spec,
        effective_wall_seconds=3_599.999,
    )
    assert (
        "effective wall-time gate failed"
        in harness._confirmatory_result_errors(
            passing,
            spec=spec,
            effective_wall_seconds=3_600.0,
        )
    )
    disk_boundary = copy.deepcopy(passing)
    disk_boundary["disk_free_min_bytes"] -= 1
    assert "disk-free gate failed" in harness._confirmatory_result_errors(
        disk_boundary,
        spec=spec,
        effective_wall_seconds=3_599.999,
    )


def test_confirmatory_coverage_gate_diagnostics_are_independent(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    output = tmp_path / "output"
    passing = _passing_gate_result(harness, output)
    spec = _passing_gate_spec(harness, output)

    for value in (0, 2):
        synthetic = copy.deepcopy(passing)
        synthetic["terminal"]["synthetic_bars"] = value
        assert harness._confirmatory_result_errors(
            synthetic,
            spec=spec,
            effective_wall_seconds=3_599.999,
        ) == ["synthetic no-trade coverage gate failed"]

    for value in (0, 2):
        reset = copy.deepcopy(passing)
        reset["terminal"]["reset_epoch"] = value
        assert harness._confirmatory_result_errors(
            reset,
            spec=spec,
            effective_wall_seconds=3_599.999,
        ) == ["data-gap reset coverage gate failed"]

    for delta in (-1, 1):
        produced = copy.deepcopy(passing)
        produced["terminal"]["produced_bars"] = (
            spec["expected_produced_bars"] + delta
        )
        assert harness._confirmatory_result_errors(
            produced,
            spec=spec,
            effective_wall_seconds=3_599.999,
        ) == ["produced-bar count gate failed"]


def test_confirmatory_selection_gate_requires_complete_exact_selection(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    output = tmp_path / "output"
    passing = _passing_gate_result(harness, output)
    spec = _passing_gate_spec(harness, output)

    unavailable = copy.deepcopy(passing)
    unavailable["observed_progress_statuses"] = [
        "running",
        "unavailable",
    ]
    unavailable["terminal"]["progress_status"] = "unavailable"
    unavailable["terminal"]["selection_status"] = "unavailable"
    assert harness._confirmatory_result_errors(
        unavailable,
        spec=spec,
        effective_wall_seconds=3_599.999,
    ) == ["terminal selection/source-free QA gate failed"]

    mismatched = copy.deepcopy(passing)
    mismatched["terminal"]["selected_case_count"] = 2
    assert harness._confirmatory_result_errors(
        mismatched,
        spec=spec,
        effective_wall_seconds=3_599.999,
    ) == ["terminal selection/source-free QA gate failed"]

    missing = copy.deepcopy(passing)
    missing["terminal"]["missing_bucket_count"] = 1
    assert harness._confirmatory_result_errors(
        missing,
        spec=spec,
        effective_wall_seconds=3_599.999,
    ) == ["terminal selection/source-free QA gate failed"]


def test_confirmatory_gap_expectations_parse_exact_positive_integers() -> None:
    harness = _load_harness()
    assert harness._confirmatory_gap_expectations(
        {
            "planned_clock_coverage": {
                "planned_short_open_gaps": 166,
                "planned_long_open_gaps": 116,
            }
        }
    ) == (166, 116)


@pytest.mark.parametrize(
    ("coverage", "message"),
    [
        (None, "planned clock coverage is absent"),
        (
            {"planned_long_open_gaps": 1},
            "gap plans must be positive integers",
        ),
        (
            {
                "planned_short_open_gaps": True,
                "planned_long_open_gaps": 1,
            },
            "gap plans must be positive integers",
        ),
        (
            {
                "planned_short_open_gaps": 1,
                "planned_long_open_gaps": False,
            },
            "gap plans must be positive integers",
        ),
        (
            {
                "planned_short_open_gaps": 0,
                "planned_long_open_gaps": 1,
            },
            "gap plans must be positive integers",
        ),
        (
            {
                "planned_short_open_gaps": 1,
                "planned_long_open_gaps": -1,
            },
            "gap plans must be positive integers",
        ),
    ],
)
def test_confirmatory_gap_expectations_fail_closed(
    coverage: object,
    message: str,
) -> None:
    harness = _load_harness()
    with pytest.raises(ValueError, match=message):
        harness._confirmatory_gap_expectations(
            {"planned_clock_coverage": coverage}
        )


@pytest.mark.parametrize("pattern", ["light", "event_heavy"])
def test_in_range_fixture_exercises_short_and_long_open_gaps(
    pattern: str,
) -> None:
    harness = _load_harness()
    coverage = harness._synthetic_clock_preflight(
        source_rows=5_000,
        first_trade_date="2024-01-08",
        require_coverage=True,
    )
    assert coverage["planned_short_open_gaps"] > 0
    assert coverage["planned_long_open_gaps"] > 0

    frame = pd.concat(
        harness._session_frames(
            source_rows=5_000,
            first_trade_date="2024-01-08",
            pattern=pattern,
        ),
        ignore_index=True,
    ).set_index("ts")
    assert all(
        is_registered_trading_minute(timestamp)
        for timestamp in frame.index
    )
    replay = SemanticDiscoveryReplay(
        reader=CausalMarketReader(maximum_history=64),
        observer=CausalObserver(
            ObserverConfig(
                memory_events=32,
                minimum_bars={
                    timeframe: 1 for timeframe in Timeframe
                },
            )
        ),
        selector=None,
    )
    receipts = iter_provenanced_completed_bars(
        [
            SourceChunk(
                frame=frame,
                ordinals=tuple(range(len(frame))),
            )
        ],
        maximum_no_trade_gap_minutes=5,
        allow_data_gap_reset=True,
    )
    for receipt in receipts:
        replay.on_receipt(receipt)
    assert replay.source_rows_admitted == len(frame)
    assert replay.synthetic_bars > 0
    assert replay.reset_epoch > 0
    assert replay.produced_bars > len(frame)


def test_session_frames_omit_holiday_and_early_close_minutes() -> None:
    harness = _load_harness()
    holiday_coverage = harness._synthetic_clock_preflight(
        source_rows=5_000,
        first_trade_date="2024-12-23",
        require_coverage=True,
    )
    assert holiday_coverage["planned_short_open_gaps"] > 0
    assert holiday_coverage["planned_long_open_gaps"] > 0
    frame = pd.concat(
        harness._session_frames(
            source_rows=1_200,
            first_trade_date="2024-12-24",
            pattern="light",
        ),
        ignore_index=True,
    ).set_index("ts")
    local = frame.index.tz_convert("America/New_York")
    assert all(
        is_registered_trading_minute(timestamp)
        for timestamp in local
    )
    christmas_eve_close = pd.Timestamp(
        "2024-12-24 13:15",
        tz="America/New_York",
    )
    next_registered_open = pd.Timestamp(
        "2024-12-25 18:00",
        tz="America/New_York",
    )
    assert local[local < christmas_eve_close][-1] == pd.Timestamp(
        "2024-12-24 13:14",
        tz="America/New_York",
    )
    assert not any(
        christmas_eve_close <= timestamp < next_registered_open
        for timestamp in local
    )
    assert local[local >= next_registered_open][0] == next_registered_open


def test_out_of_clock_fixture_fails_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setattr(harness, "ARTIFACT_ROOT", artifact_root)
    fixture_root = artifact_root / "fixture"
    args = argparse.Namespace(
        fixture_root=str(fixture_root),
        expires_at="2099-01-01T00:00:00+00:00",
        confirmatory_one_time=False,
        confirmatory_attempt_id=None,
        confirmatory_registry_root=None,
        source_rows=5_000,
        first_trade_date="2031-03-10",
        pattern="light",
    )
    with pytest.raises(
        ValueError,
        match="outside registered market clock",
    ):
        harness.prepare(args)
    assert not fixture_root.exists()


def _minimal_preflight_tree(
    tmp_path: Path,
    harness,
    *,
    extra_source_binding: bool,
) -> tuple[argparse.Namespace, tuple[Path, Path]]:
    project = tmp_path / "project"
    artifact_root = (
        project / "artifacts/synthetic_semantic_pass1_e2e"
    )
    artifact_root.mkdir(parents=True)
    fixed = {
        "scripts/benchmark_v3_semantic_pass1_e2e.py",
        "smc_trader/semantic_discovery_runner.py",
        "smc_trader/artifact_stream.py",
        "smc_trader/model.py",
        "smc_trader/causal.py",
        "smc_trader/structure.py",
        "smc_trader/observation.py",
        "smc_trader/market_clock.py",
        "smc_trader/io.py",
        "smc_trader/semantic_audit.py",
        "configs/model_v3_0_1_exp001_structure_bos_identity_r3.json",
        (
            "configs/"
            "smc_primitives_v3_0_1_structure_bos_audit_closure_r3.json"
        ),
        "tests/test_v3_semantic_discovery_runner.py",
        "tests/test_v3_semantic_discovery_pass1_release.py",
        "tests/test_v3_semantic_pass1_e2e_harness.py",
        (
            "configs/experiments/"
            "EXP-SMC-3.0.1-001-STRUCTURE-BOS-AUDIT-CLOSURE-R3-"
            "TEST-FREEZE-R6.json"
        ),
        (
            "reports/validation_2026-07-29/"
            "v3_exp001_semantic_audit_closure_r6_targeted_"
            "posttest_validation.md"
        ),
        (
            "reports/validation_2026-07-29/"
            "v3_exp001_r6_ledger_clock_erratum.md"
        ),
        (
            "reports/validation_2026-07-29/"
            "v3_exp001_r3_full_e2e_successor_r7_pretest_self_review.md"
        ),
    }
    for relative in fixed:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    prereg = project / "reports/prereg.md"
    prereg.write_text("frozen prereg", encoding="utf-8")
    fixed.add(str(prereg.relative_to(project)))
    fixture_specs = []
    sources: list[Path] = []
    for ordinal, pattern in enumerate(("light", "event_heavy")):
        fixture = artifact_root / f"fixture-{pattern}"
        (fixture / "contracts").mkdir(parents=True)
        (fixture / "releases").mkdir()
        (fixture / "source").mkdir()
        manifest = fixture / "fixture_manifest.json"
        audit = fixture / "contracts/audit.json"
        runner = fixture / "contracts/runner.json"
        release = fixture / "releases/release.json"
        source = fixture / "source/source.parquet"
        for path in (manifest, audit, runner, release):
            path.write_text("{}", encoding="utf-8")
            fixed.add(str(path.relative_to(project)))
        source.write_bytes(
            f"source-{ordinal}-must-not-be-hashed".encode()
        )
        sources.append(source)
        fixture_specs.append(
            {
                "pattern": pattern,
                "fixture_root": str(fixture),
                "fixture_manifest_path": str(manifest),
                "release_path": str(release),
                "source_path": str(source),
                "source_sha256": (
                    "d" * 64 if ordinal == 0 else "e" * 64
                ),
                "output_path": str(fixture / "output"),
                "result_path": str(fixture / "result.json"),
            }
        )
    bindings = {
        relative: _sha(project / relative)
        for relative in fixed
    }
    if extra_source_binding:
        bindings[str(sources[0].relative_to(project))] = _sha(sources[0])
    registry = artifact_root / "pair"
    registry.mkdir()
    runtime = {
        "python_executable": "~/miniconda3/bin/python",
        "platform_machine": "x86_64",
        "rosetta_translated": True,
        "host_arm64_capable": True,
    }
    now = pd.Timestamp.now(tz="UTC")
    freeze = project / "reports/freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "artifact": (
                    "v3_synthetic_full_e2e_confirmatory_pair_freeze"
                ),
                "status": "confirmatory_pair_frozen",
                "attempt_id": "PREFLIGHT-UNIT-R1",
                "issued_at": (
                    now - pd.Timedelta(minutes=1)
                ).isoformat(),
                "expires_at": (
                    now + pd.Timedelta(hours=1)
                ).isoformat(),
                "preregistration_path": str(prereg),
                "preregistration_sha256": _sha(prereg),
                "gates": harness.CONFIRMATORY_GATES,
                "runtime": runtime,
                "bindings": bindings,
                "fixtures": fixture_specs,
                "pair_result_root": str(registry),
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        freeze=str(freeze),
        expected_freeze_sha256=_sha(freeze),
        preregistration=str(prereg),
        expected_preregistration_sha256=_sha(prereg),
    )
    return args, tuple(sources)


def test_preflight_rejects_extra_source_binding_without_reading_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    args, sources = _minimal_preflight_tree(
        tmp_path,
        harness,
        extra_source_binding=True,
    )
    project = tmp_path / "project"
    monkeypatch.setattr(harness, "ROOT", project)
    monkeypatch.setattr(
        harness,
        "ARTIFACT_ROOT",
        project / "artifacts/synthetic_semantic_pass1_e2e",
    )
    runtime = json.loads(Path(args.freeze).read_text())["runtime"]
    monkeypatch.setattr(harness, "_runtime_fingerprint", lambda: runtime)
    original_hash = harness._bootstrap_sha256_file

    def guarded_hash(path):
        if Path(path) in sources:
            raise AssertionError("preflight read source content")
        return original_hash(path)

    monkeypatch.setattr(
        harness,
        "_bootstrap_sha256_file",
        guarded_hash,
    )
    with pytest.raises(ValueError, match="must be exact"):
        harness._confirmatory_preflight(args)


def _configure_minimal_preflight(
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_source_binding: bool = False,
):
    args, sources = _minimal_preflight_tree(
        tmp_path,
        harness,
        extra_source_binding=extra_source_binding,
    )
    project = tmp_path / "project"
    monkeypatch.setattr(harness, "ROOT", project)
    monkeypatch.setattr(
        harness,
        "ARTIFACT_ROOT",
        project / "artifacts/synthetic_semantic_pass1_e2e",
    )
    runtime = json.loads(Path(args.freeze).read_text())["runtime"]
    monkeypatch.setattr(harness, "_runtime_fingerprint", lambda: runtime)
    return args, sources


def _rewrite_freeze(args, mutate) -> None:
    freeze_path = Path(args.freeze)
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    mutate(payload)
    freeze_path.write_text(json.dumps(payload), encoding="utf-8")
    args.expected_freeze_sha256 = _sha(freeze_path)


def test_preflight_requires_canonical_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    args, sources = _configure_minimal_preflight(
        harness,
        tmp_path,
        monkeypatch,
    )
    alternate = (
        Path(json.loads(Path(args.freeze).read_text())["fixtures"][0][
            "fixture_root"
        ])
        / "alternate_manifest.json"
    )
    alternate.write_text("{}", encoding="utf-8")
    _rewrite_freeze(
        args,
        lambda payload: payload["fixtures"][0].update(
            {"fixture_manifest_path": str(alternate)}
        ),
    )
    original_hash = harness._bootstrap_sha256_file

    def guarded_hash(path):
        if Path(path) in sources:
            raise AssertionError("preflight read source content")
        return original_hash(path)

    monkeypatch.setattr(
        harness,
        "_bootstrap_sha256_file",
        guarded_hash,
    )
    with pytest.raises(ValueError, match="not canonical"):
        harness._confirmatory_preflight(args)


def test_preflight_requires_distinct_pair_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    args, _sources = _configure_minimal_preflight(
        harness,
        tmp_path,
        monkeypatch,
    )

    def duplicate_source_hash(payload):
        payload["fixtures"][1]["source_sha256"] = payload["fixtures"][0][
            "source_sha256"
        ]

    _rewrite_freeze(args, duplicate_source_hash)
    with pytest.raises(ValueError, match="duplicate source hashes"):
        harness._confirmatory_preflight(args)


def test_preflight_hash_failure_precedes_parquet_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    project = tmp_path / "project"
    project.mkdir()
    freeze = project / "freeze.json"
    prereg = project / "prereg.md"
    freeze.write_text("{}", encoding="utf-8")
    prereg.write_text("prereg", encoding="utf-8")
    monkeypatch.setattr(harness, "ROOT", project)
    touched = {"source": False}

    def forbidden_source(*_args, **_kwargs):
        touched["source"] = True
        raise AssertionError("Parquet source constructed")

    monkeypatch.setattr(harness, "ParquetBatchSource", forbidden_source)
    args = argparse.Namespace(
        freeze=str(freeze),
        expected_freeze_sha256="0" * 64,
        preregistration=str(prereg),
        expected_preregistration_sha256=_sha(prereg),
    )
    with pytest.raises(ValueError, match="freeze digest changed"):
        harness.confirm_pair(args)
    assert touched["source"] is False


def _fake_confirm_result(result_path: Path) -> dict:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{}", encoding="utf-8")
    return {
        "result_path": str(result_path),
        "wall_seconds": 1.0,
        "rss_peak_bytes": 1,
        "disk_free_min_bytes": 10**12,
        "terminal": {
            "source_prefix_root": "a" * 64,
            "produced_bar_prefix_root": "b" * 64,
            "selection_status": "complete",
            "selected_case_count": 1,
            "expected_case_count": 1,
            "missing_bucket_count": 0,
            "synthetic_bars": 1,
            "reset_epoch": 1,
            "produced_bars": 2,
        },
    }


def test_confirm_pair_marker_precedes_runs_and_is_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    registry = tmp_path / "registry"
    registry.mkdir()
    attempt = registry / "CONFIRMATORY_ATTEMPT_STARTED.json"
    completion = registry / "CONFIRMATORY_ATTEMPT_COMPLETED.json"
    pair_result = registry / "confirmatory_pair_result.json"
    fixtures = []
    for pattern in ("light", "event_heavy"):
        release = tmp_path / f"{pattern}-release.json"
        release.write_text("{}", encoding="utf-8")
        fixtures.append(
            {
                "pattern": pattern,
                "release_path": str(release),
                "release_sha256": _sha(release),
                "source_sha256": "c" * 64,
                "output_path": str(tmp_path / f"{pattern}-output"),
                "result_path": str(tmp_path / f"{pattern}-result.json"),
                "fixture_root": str(tmp_path / pattern),
                "expected_synthetic_bars": 1,
                "expected_reset_epochs": 1,
                "expected_produced_bars": 2,
            }
        )
    freeze = {
        "attempt_id": "PAIR-UNIT-R1",
        "_attempt_path": str(attempt),
        "_completion_path": str(completion),
        "_pair_result_path": str(pair_result),
        "_verified_freeze_sha256": "d" * 64,
        "_verified_preregistration_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        harness,
        "_confirmatory_preflight",
        lambda _args: (dict(freeze), tuple(fixtures)),
    )
    monkeypatch.setattr(
        harness,
        "_confirmatory_result_errors",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(harness, "_runtime_fingerprint", lambda: {})
    order: list[str] = []
    Path(fixtures[0]["release_path"]).write_text(
        '{"tampered":true}',
        encoding="utf-8",
    )

    def fake_run(args):
        assert attempt.is_file()
        marker = json.loads(attempt.read_text(encoding="utf-8"))
        assert marker["one_time_attempt_consumed"] is True
        pattern = Path(args.result).name.split("-")[0]
        marker_entry = next(
            item
            for item in marker["fixtures"]
            if item["pattern"] == pattern
        )
        frozen_entry = next(
            item for item in fixtures if item["pattern"] == pattern
        )
        assert (
            marker_entry["release_sha256"]
            == frozen_entry["release_sha256"]
        )
        if pattern == "light":
            assert marker_entry["release_sha256"] != _sha(
                Path(frozen_entry["release_path"])
            )
        order.append(pattern)
        return _fake_confirm_result(Path(args.result))

    monkeypatch.setattr(harness, "run_benchmark", fake_run)
    result = harness.confirm_pair(argparse.Namespace())
    assert order == ["light", "event_heavy"]
    assert result["status"] == "pass"
    assert pair_result.is_file()
    assert completion.is_file()
    with pytest.raises(FileExistsError, match="create-once"):
        harness.confirm_pair(argparse.Namespace())


def test_failed_confirm_pair_consumes_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    registry = tmp_path / "registry"
    registry.mkdir()
    attempt = registry / "CONFIRMATORY_ATTEMPT_STARTED.json"
    release = tmp_path / "release.json"
    release.write_text("{}", encoding="utf-8")
    freeze = {
        "attempt_id": "PAIR-FAIL-R1",
        "_attempt_path": str(attempt),
        "_completion_path": str(
            registry / "CONFIRMATORY_ATTEMPT_COMPLETED.json"
        ),
        "_pair_result_path": str(
            registry / "confirmatory_pair_result.json"
        ),
        "_verified_freeze_sha256": "d" * 64,
        "_verified_preregistration_sha256": "e" * 64,
    }
    fixture = {
        "pattern": "light",
        "release_path": str(release),
        "release_sha256": _sha(release),
        "source_sha256": "c" * 64,
        "output_path": str(tmp_path / "output"),
        "result_path": str(tmp_path / "result.json"),
        "fixture_root": str(tmp_path / "fixture"),
    }
    monkeypatch.setattr(
        harness,
        "_confirmatory_preflight",
        lambda _args: (dict(freeze), (fixture,)),
    )
    calls = {"count": 0}

    def fail_after_marker(_args):
        calls["count"] += 1
        assert attempt.is_file()
        raise RuntimeError("injected replay failure")

    monkeypatch.setattr(harness, "run_benchmark", fail_after_marker)
    with pytest.raises(RuntimeError, match="injected"):
        harness.confirm_pair(argparse.Namespace())
    assert attempt.is_file()
    with pytest.raises(FileExistsError, match="create-once"):
        harness.confirm_pair(argparse.Namespace())
    assert calls["count"] == 1


def test_confirm_pair_does_not_recreate_missing_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    registry = tmp_path / "registry"
    registry.mkdir()
    attempt = registry / "CONFIRMATORY_ATTEMPT_STARTED.json"
    freeze = {
        "attempt_id": "REGISTRY-RACE-R1",
        "_attempt_path": str(attempt),
        "_completion_path": str(
            registry / "CONFIRMATORY_ATTEMPT_COMPLETED.json"
        ),
        "_pair_result_path": str(
            registry / "confirmatory_pair_result.json"
        ),
        "_verified_freeze_sha256": "d" * 64,
        "_verified_preregistration_sha256": "e" * 64,
    }
    fixtures = (
        {
            "pattern": "light",
            "release_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "output_path": str(tmp_path / "light-output"),
        },
        {
            "pattern": "event_heavy",
            "release_sha256": "c" * 64,
            "source_sha256": "d" * 64,
            "output_path": str(tmp_path / "event-output"),
        },
    )

    def remove_registry(_args):
        registry.rmdir()
        return dict(freeze), fixtures

    monkeypatch.setattr(
        harness,
        "_confirmatory_preflight",
        remove_registry,
    )
    with pytest.raises(
        FileNotFoundError,
        match="pre-existing parent",
    ):
        harness.confirm_pair(argparse.Namespace())
    assert not registry.exists()
    assert not attempt.exists()
