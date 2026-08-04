from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pandas as pd
import psutil
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.model import (
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    Direction,
    StructureLifecycle,
    StructureSequenceState,
    SwingLifecycle,
    SwingPoint,
    SwingRelation,
    SwingSide,
    Timeframe,
)
from smc_trader.observation import CausalObserver, ObserverConfig
import smc_trader.semantic_discovery_runner as discovery_module
from smc_trader.semantic_discovery_runner import (
    PASS1_EXECUTION_RELEASE,
    FrozenParquetSourceIdentity,
    ParquetBatchSource,
    SemanticDiscoveryReplay,
    SemanticDiscoveryRunner,
    load_pass1_execution_release,
    runtime_environment,
    source_for_pass1_release,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame() -> pd.DataFrame:
    index = pd.date_range(
        "2023-01-03 18:00",
        periods=2,
        freq="1min",
        tz="America/New_York",
        name="ts",
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 12.0],
            "symbol": ["NQH3", "NQH3"],
            "instrument_id": [1, 1],
        },
        index=index,
    )


def _bos(asof: pd.Timestamp) -> BreakOfStructureState:
    return BreakOfStructureState(
        bos_id="release-test-bos",
        timeframe=Timeframe.M1,
        direction=Direction.LONG,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
        target_swing_id="release-test-swing",
        source_structure_id="release-test-structure",
        target_price=100.0,
        target_ticks=400,
        pending_at=asof - pd.Timedelta(minutes=1),
        resolved_at=asof,
        age_bars=1,
        attempt_count=0,
        last_attempt_at=None,
    )


def _semantic_context(
    asof: pd.Timestamp,
) -> tuple[
    SwingPoint,
    StructureSequenceState,
    dict[Timeframe, tuple[Candle, ...]],
]:
    swing = SwingPoint(
        swing_id="release-test-swing",
        timeframe=Timeframe.M1,
        symbol="NQH3",
        instrument_id=1,
        side=SwingSide.HIGH,
        price=100.0,
        price_ticks=400,
        pivot_start=asof - pd.Timedelta(minutes=4),
        pivot_end=asof - pd.Timedelta(minutes=3),
        observed_at=asof - pd.Timedelta(minutes=2),
        confirmed_at=asof - pd.Timedelta(minutes=2),
        lifecycle=SwingLifecycle.BROKEN,
        relation=SwingRelation.HH,
        age_bars=2,
        broken_at=asof,
        failure_reason="close_beyond_swing",
    )
    structure = StructureSequenceState(
        structure_id="release-test-structure",
        timeframe=Timeframe.M1,
        direction=Direction.LONG,
        lifecycle=StructureLifecycle.CONFIRMED,
        formed_at=asof - pd.Timedelta(minutes=6),
        confirmed_at=asof - pd.Timedelta(minutes=5),
        broken_at=None,
        high_run=2,
        low_run=2,
        sequence_count=2,
        latest_high_id=swing.swing_id,
        latest_low_id="release-test-low",
        protected_swing_id="release-test-protected",
        protected_price=99.0,
        cumulative_magnitude_atr=2.0,
        age_bars=3,
    )
    durations = {
        Timeframe.H4: 240,
        Timeframe.H1: 60,
        Timeframe.M5: 5,
        Timeframe.M1: 1,
    }
    histories: dict[Timeframe, tuple[Candle, ...]] = {}
    for timeframe, duration in durations.items():
        count = 6 if timeframe is Timeframe.M1 else 1
        histories[timeframe] = tuple(
            Candle(
                timeframe=timeframe,
                start=asof
                - pd.Timedelta(minutes=duration * (count - index)),
                end=asof
                - pd.Timedelta(
                    minutes=duration * (count - index - 1)
                ),
                open=99.75,
                high=100.5,
                low=99.5,
                close=100.0,
                volume=10.0,
                symbol="NQH3",
                instrument_id=1,
                observed_minutes=duration,
                expected_minutes=duration,
                complete=True,
            )
            for index in range(count)
        )
    return swing, structure, histories


class _ReleaseReplay(SemanticDiscoveryReplay):
    def __init__(self, *, selector, event_enabled: bool) -> None:
        super().__init__(
            reader=CausalMarketReader(maximum_history=64),
            observer=CausalObserver(
                ObserverConfig(
                    memory_events=32,
                    minimum_bars={
                        timeframe: 1 for timeframe in Timeframe
                    },
                )
            ),
            selector=selector,
        )
        self.event_enabled = event_enabled

    def on_receipt(self, receipt):
        selector = self.selector
        self.selector = None
        observation = super().on_receipt(receipt)
        self.selector = selector
        if (
            self.event_enabled
            and receipt.real_source_bar
            and receipt.source_row_ordinal == 0
        ):
            frames = dict(observation.frames)
            swing, structure, histories = _semantic_context(
                observation.asof
            )
            frames[Timeframe.M1] = replace(
                observation.frame(Timeframe.M1),
                swings=(swing,),
                structures=(structure,),
                structure_breaks=(_bos(observation.asof),),
            )
            observation = replace(observation, frames=frames)
        else:
            histories = {
                timeframe: self.reader.window(
                    timeframe,
                    self.reader.maximum_history,
                )
                for timeframe in Timeframe
            }
        if receipt.real_source_bar and selector is not None:
            selector.observe(
                observation,
                receipt,
                histories=histories,
                source_prefix_root=self.source_prefix_root,
                produced_bar_prefix_root=self.produced_bar_prefix_root,
                source_rows_admitted=self.source_rows_admitted,
                produced_bars=self.produced_bars,
                reset_epoch=self.reset_epoch,
            )
        self.last_observation = observation
        return observation


class _ReleaseRunner(SemanticDiscoveryRunner):
    def __init__(self, *args, event_enabled: bool, **kwargs) -> None:
        self.event_enabled = event_enabled
        super().__init__(*args, **kwargs)

    def _new_replay(self, *, selector):
        return _ReleaseReplay(
            selector=selector,
            event_enabled=self.event_enabled,
        )


def _contracts(
    tmp_path: Path,
    *,
    source_role: str = "synthetic_fixture",
) -> dict:
    tmp_path.mkdir(parents=True)
    frame = _frame()
    source_path = tmp_path / "fixture.parquet"
    frame.to_parquet(source_path)
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "tick_size": 0.25,
                "point_value": 20.0,
                "observer": {
                    "memory_events": 32,
                    "minimum_bars": {
                        timeframe.value: 1
                        for timeframe in Timeframe
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source = ParquetBatchSource(
        source_path,
        start=frame.index[0],
        end_exclusive=frame.index[-1] + pd.Timedelta(minutes=1),
        batch_rows=2,
        expected_sha256=_sha(source_path),
    )
    audit = {
        "audit_id": "SYNTHETIC-PASS1-RELEASE-AUDIT",
        "bindings": {
            "primitive_protocol_sha256": "a" * 64,
            "causal_source_sha256": source.sha256,
        },
        "source": {
            "path": str(source_path),
            "start": source.start.isoformat(),
            "end_exclusive": source.end_exclusive.isoformat(),
            "window_role": source_role,
        },
        "selection": {
            "timeframes": ["1m"],
            "directions": ["long"],
            "case_classes": ["confirmed_bos"],
            "calendar_years": [2023],
            "cases_per_bucket": 1,
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(audit, sort_keys=True),
        encoding="utf-8",
    )
    implementation_hashes = {
        "semantic_discovery_runner_sha256": "1" * 64,
        "semantic_audit_sha256": "2" * 64,
        "io_sha256": "3" * 64,
        "causal_sha256": "4" * 64,
        "market_clock_sha256": "5" * 64,
        "observation_sha256": "6" * 64,
        "structure_sha256": "7" * 64,
    }
    runner_contract = {
        "runner_id": "SYNTHETIC-PASS1-RELEASE-RUNNER",
        "status": "implementation_contract_frozen",
        "authorization": (
            "implementation and synthetic tests only; real discovery "
            "execution is forbidden until a later governance release"
        ),
        "bindings": {
            "model_config_sha256": _sha(model_path),
            "blind_audit_contract_sha256": _sha(audit_path),
            "primitive_protocol_sha256": "a" * 64,
            "io_code_sha256": "b" * 64,
            "causal_reader_code_sha256": "c" * 64,
            "market_clock_code_sha256": "d" * 64,
            "semantic_audit_code_sha256": "e" * 64,
        },
        "source_iterator": {
            "batch_rows": 2,
            "maximum_no_trade_gap_minutes": 5,
            "allow_data_gap_reset": True,
        },
        "implementation_hashes_required": list(
            implementation_hashes
        ),
    }
    runner_path = tmp_path / "runner.json"
    runner_path.write_text(
        json.dumps(runner_contract, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "audit": audit,
        "audit_sha256": _sha(audit_path),
        "runner_contract": runner_contract,
        "runner_sha256": _sha(runner_path),
        "source": source,
        "model_path": model_path,
        "implementation_hashes": implementation_hashes,
    }


def _release(context: dict, output: Path) -> dict:
    audit = context["audit"]
    runner = context["runner_contract"]
    implementation = context["implementation_hashes"]
    return {
        "release_id": "SYNTHETIC-PASS1-RELEASE-R1",
        "status": "synthetic_test_release",
        "authorization": "synthetic pass1 release test",
        "execution_authorized": True,
        "authorized_phases": ["pass1"],
        "expires_at": "2099-01-01T00:00:00+00:00",
        "bindings": {
            "audit_contract_sha256": context["audit_sha256"],
            "engineering_runner_contract_sha256": (
                context["runner_sha256"]
            ),
            "semantic_discovery_runner_sha256": implementation[
                "semantic_discovery_runner_sha256"
            ],
            "causal_source_sha256": context["source"].sha256,
            "model_config_sha256": runner["bindings"][
                "model_config_sha256"
            ],
            "primitive_protocol_sha256": runner["bindings"][
                "primitive_protocol_sha256"
            ],
            "io_code_sha256": runner["bindings"]["io_code_sha256"],
            "causal_reader_code_sha256": runner["bindings"][
                "causal_reader_code_sha256"
            ],
            "market_clock_code_sha256": runner["bindings"][
                "market_clock_code_sha256"
            ],
            "semantic_audit_code_sha256": runner["bindings"][
                "semantic_audit_code_sha256"
            ],
            "observer_code_sha256": implementation[
                "observation_sha256"
            ],
            "structure_code_sha256": implementation[
                "structure_sha256"
            ],
        },
        "source": {
            "path": audit["source"]["path"],
            "start": audit["source"]["start"],
            "end_exclusive": audit["source"]["end_exclusive"],
            "window_role": audit["source"]["window_role"],
        },
        "runtime": runtime_environment(),
        "run": {
            "batch_rows": 2,
            "maximum_history": 64,
            "maximum_no_trade_gap_minutes": 5,
            "allow_data_gap_reset": True,
            "checkpoint_source_rows": 1,
            "real_only_case_anchors": True,
            "diagnostic_stop_allowed": True,
        },
        "resources": {
            "rss_hard_ceiling_bytes": 10**12,
            "disk_free_floor_bytes": 1,
            "checkpoint_stall_timeout_seconds": 3600,
            "resource_check_source_rows": 1,
        },
        "output": {"path": str(output)},
    }


def _one_time_release(
    context: dict,
    output: Path,
    registry_root: Path,
) -> dict:
    registry_root.mkdir(parents=True)
    release = _release(context, output)
    issued_at = pd.Timestamp.now(tz="UTC")
    release["authorization"] = "synthetic pass1 confirmatory one-time"
    release["confirmatory_one_time"] = True
    release["issued_at"] = issued_at.isoformat()
    release["expires_at"] = (
        issued_at + pd.Timedelta(hours=1)
    ).isoformat()
    release["run"]["diagnostic_stop_allowed"] = False
    release["confirmatory"] = {
        "attempt_id": "UNIT-CONFIRM-R1",
        "registry_root": str(registry_root),
        "attempt_marker_path": str(
            registry_root / "CONFIRMATORY_ATTEMPT_STARTED.json"
        ),
    }
    return release


def _write_attempt_marker(
    *,
    context: dict,
    release: dict,
    output: Path,
    release_sha256: str = "8" * 64,
) -> Path:
    marker_path = Path(
        release["confirmatory"]["attempt_marker_path"]
    )
    marker_path.write_text(
        json.dumps(
            {
                "artifact": (
                    "v3_synthetic_full_e2e_attempt_started"
                ),
                "attempt_id": release["confirmatory"]["attempt_id"],
                "one_time_attempt_consumed": True,
                "fixtures": [
                    {
                        "release_sha256": release_sha256,
                        "source_sha256": context["source"].sha256,
                        "output_path": str(output),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return marker_path


def _runner(
    context: dict,
    *,
    output: Path,
    event_enabled: bool = True,
    release: dict | None = None,
    release_sha256: str | None = "8" * 64,
    source_override=None,
) -> _ReleaseRunner:
    return _ReleaseRunner(
        audit_contract=context["audit"],
        runner_contract=context["runner_contract"],
        source=(
            context["source"]
            if source_override is None
            else source_override
        ),
        model_config_path=context["model_path"],
        output_root=output,
        audit_contract_sha256=context["audit_sha256"],
        runner_contract_sha256=context["runner_sha256"],
        implementation_hashes=context["implementation_hashes"],
        maximum_history=64,
        checkpoint_source_rows=1,
        execution_release=release,
        execution_release_sha256=(
            None if release is None else release_sha256
        ),
        event_enabled=event_enabled,
    )


def test_phase_release_pass1_complete_and_shortage_are_terminal(
    tmp_path: Path,
) -> None:
    complete_context = _contracts(tmp_path / "complete-contracts")
    complete_output = tmp_path / "complete"
    complete = _runner(
        complete_context,
        output=complete_output,
        release=_release(complete_context, complete_output),
    )
    selection = json.loads(
        complete.run_pass1().read_text(encoding="utf-8")
    )
    assert selection["status"] == "complete"
    assert selection["selected_case_count"] == 1
    assert {path.name for path in complete_output.iterdir()} == {
        "_checkpoint",
        "progress.json",
        "selection_authority.json",
    }
    checkpoint = json.loads(
        (
            complete_output
            / "_checkpoint"
            / "pass1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["verified_source_row_count"] == 2

    shortage_context = _contracts(tmp_path / "shortage-contracts")
    shortage_output = tmp_path / "shortage"
    shortage = _runner(
        shortage_context,
        output=shortage_output,
        event_enabled=False,
        release=_release(shortage_context, shortage_output),
    )
    unavailable = json.loads(
        shortage.run_pass1().read_text(encoding="utf-8")
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["selected_case_count"] == 0


def test_phase_release_never_authorizes_pass2_or_old_contract(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "released"
    released = _runner(
        context,
        output=output,
        release=_release(context, output),
    )

    def forbidden_count_rows():
        raise AssertionError("pass2 touched the source")

    released.source.count_rows = forbidden_count_rows
    with pytest.raises(PermissionError, match="does not authorize pass2"):
        released.run_pass2()
    assert not output.exists()

    locked_output = tmp_path / "locked"
    locked = _runner(
        context,
        output=locked_output,
        release=None,
        release_sha256=None,
    )
    with pytest.raises(PermissionError, match="not authorized"):
        locked.run_pass1()
    assert not locked_output.exists()


def test_one_time_synthetic_confirmatory_forbids_diagnostic_stop(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "confirmatory"
    release = _one_time_release(
        context,
        output,
        tmp_path / "attempt-registry",
    )
    _write_attempt_marker(
        context=context,
        release=release,
        output=output,
    )
    runner = _runner(
        context,
        output=output,
        release=release,
    )
    with pytest.raises(PermissionError, match="forbids diagnostic"):
        runner.run_pass1(diagnostic_stop_after_source_rows=1)
    assert not output.exists()


def test_one_time_release_requires_exact_attempt_marker(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "confirmatory"
    release = _one_time_release(
        context,
        output,
        tmp_path / "attempt-registry",
    )
    runner = _runner(
        context,
        output=output,
        release=release,
    )
    with pytest.raises(
        PermissionError,
        match="durably registered",
    ):
        runner.run_pass1()
    assert not output.exists()

    _write_attempt_marker(
        context=context,
        release=release,
        output=tmp_path / "wrong-output",
    )
    with pytest.raises(
        PermissionError,
        match="absent from the attempt marker",
    ):
        runner.run_pass1()
    assert not output.exists()


def test_one_time_release_rejects_partial_resume_but_allows_terminal(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "confirmatory"
    release = _one_time_release(
        context,
        output,
        tmp_path / "attempt-registry",
    )
    _write_attempt_marker(
        context=context,
        release=release,
        output=output,
    )
    runner = _runner(
        context,
        output=output,
        release=release,
    )
    with pytest.raises(
        PermissionError,
        match="terminal-only",
    ):
        runner.run_pass1(resume=True)
    selection = runner.run_pass1()
    assert selection == output / "selection_authority.json"

    source = context["source"]
    frozen = FrozenParquetSourceIdentity(
        sha256=source.sha256,
        start=source.start,
        end_exclusive=source.end_exclusive,
        batch_rows=source.batch_rows,
    )
    terminal = _runner(
        context,
        output=output,
        release=release,
        source_override=frozen,
    )
    assert terminal.run_pass1(resume=True) == selection


def test_explicit_false_confirmatory_flag_keeps_diagnostic_authority(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "ordinary-synthetic"
    release = _release(context, output)
    release["confirmatory_one_time"] = False
    runner = _runner(
        context,
        output=output,
        release=release,
    )
    with pytest.raises(
        RuntimeError,
        match="intentional semantic pass1 interruption",
    ):
        runner.run_pass1(diagnostic_stop_after_source_rows=1)
    assert (output / "_checkpoint" / "pass1" / "manifest.json").is_file()


def test_non_boolean_confirmatory_flag_is_rejected(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "invalid-confirmatory-flag"
    release = _release(context, output)
    release["confirmatory_one_time"] = "false"
    with pytest.raises(
        ValueError,
        match="confirmatory one-time release flag is invalid",
    ):
        _runner(
            context,
            output=output,
            release=release,
        )
    assert not output.exists()


def test_draft_expiry_diagnostic_and_binding_fail_closed(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "output"
    base = _release(context, output)

    draft = copy.deepcopy(base)
    draft.update(
        {
            "status": "draft_no_execution_authority",
            "authorization": "draft only",
            "execution_authorized": False,
        }
    )
    draft["run"]["diagnostic_stop_allowed"] = False
    draft_runner = _runner(
        context,
        output=output,
        release=draft,
    )
    with pytest.raises(PermissionError, match="not active"):
        draft_runner.run_pass1()
    assert not output.exists()
    expired = copy.deepcopy(base)
    expired["expires_at"] = "2020-01-01T00:00:00+00:00"
    expired_runner = _runner(
        context,
        output=output,
        release=expired,
    )
    with pytest.raises(PermissionError, match="expired"):
        expired_runner.run_pass1()
    assert not output.exists()

    wrong_binding = copy.deepcopy(base)
    wrong_binding["bindings"]["causal_source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="binding changed"):
        _runner(
            context,
            output=output,
            release=wrong_binding,
        )

    wrong_runtime = copy.deepcopy(base)
    wrong_runtime["runtime"]["python_version"] = "0.0.0"
    with pytest.raises(ValueError, match="runtime"):
        _runner(
            context,
            output=output,
            release=wrong_runtime,
        )

    with pytest.raises(ValueError, match="output differs"):
        _runner(
            context,
            output=tmp_path / "other",
            release=base,
        )
    with pytest.raises(ValueError, match="digest is invalid"):
        _runner(
            context,
            output=output,
            release=base,
            release_sha256="bad",
        )

    production_context = _contracts(
        tmp_path / "production-contracts",
        source_role="semantic_discovery",
    )
    production = _release(production_context, output)
    production.update(
        {
            "status": "execution_contract_frozen",
            "authorization": (
                "one-time real semantic discovery pass1 authorized"
            ),
        }
    )
    production["run"]["diagnostic_stop_allowed"] = False
    production_runner = _runner(
        production_context,
        output=output,
        release=production,
    )
    with pytest.raises(PermissionError, match="forbids diagnostic"):
        production_runner.run_pass1(
            diagnostic_stop_after_source_rows=1
        )
    assert not output.exists()


def test_source_factory_denies_draft_before_output_or_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "must-not-be-touched"
    draft = {
        "status": "draft_no_execution_authority",
        "authorization": "draft only",
        "execution_authorized": False,
        "authorized_phases": ["pass1"],
        "permissions": {
            "real_pass1_authorized": False,
            "discovery_data_authorized": False,
        },
        "output": {"path": str(output)},
        "source": {"path": str(tmp_path / "sealed.parquet")},
        "run": {"batch_rows": 1},
    }

    def touched(*_args, **_kwargs):
        raise AssertionError("draft release touched output or source")

    monkeypatch.setattr(
        discovery_module,
        "validate_pass1_release_output_tree",
        touched,
    )
    monkeypatch.setattr(
        discovery_module,
        "ParquetBatchSource",
        touched,
    )
    with pytest.raises(PermissionError, match="active exact"):
        source_for_pass1_release(draft, resume=False)
    assert not output.exists()


def test_release_output_and_resume_bindings_are_exact(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    occupied_release = _release(context, occupied)
    occupied_runner = _runner(
        context,
        output=occupied,
        release=occupied_release,
    )

    def forbidden_count_rows():
        raise AssertionError("invalid output touched the source")

    occupied_runner.source.count_rows = forbidden_count_rows
    with pytest.raises(FileExistsError, match="must be absent"):
        occupied_runner.run_pass1()

    context = _contracts(tmp_path / "resume-contracts")
    output = tmp_path / "resume"
    release = _release(context, output)
    interrupted = _runner(
        context,
        output=output,
        release=release,
    )
    with pytest.raises(RuntimeError, match="intentional"):
        interrupted.run_pass1(
            diagnostic_stop_after_source_rows=1
        )

    changed_release = _runner(
        context,
        output=output,
        release=release,
        release_sha256="9" * 64,
    )
    with pytest.raises(ValueError, match="bindings changed"):
        changed_release.run_pass1(resume=True)

    resumed = _runner(
        context,
        output=output,
        release=release,
    )
    original_count = resumed.source.count_rows
    resumed.source.count_rows = (
        lambda **_kwargs: original_count() + 1
    )
    with pytest.raises(ValueError, match="row count changed"):
        resumed.run_pass1(resume=True)


def test_count_preflight_is_checkpointed_and_batch_guarded(
    tmp_path: Path,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "output"
    runner = _runner(
        context,
        output=output,
        release=_release(context, output),
    )
    original_count = runner.source.count_rows
    callbacks: list[int] = []

    def guarded_count(*, on_batch):
        manifest = json.loads(
            (
                output
                / "_checkpoint"
                / "pass1"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        progress = json.loads(
            (output / "progress.json").read_text(encoding="utf-8")
        )
        assert manifest["source_rows_admitted"] == 0
        assert manifest["verified_source_row_count"] is None
        assert progress["status"] == "verifying_source"
        return original_count(
            on_batch=lambda scanned: (
                callbacks.append(scanned),
                on_batch(scanned),
            )
        )

    runner.source.count_rows = guarded_count
    runner.run_pass1()
    assert callbacks
    assert callbacks == sorted(callbacks)


def test_terminal_progress_is_reconciled_without_source_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _contracts(
        tmp_path / "contracts",
        source_role="semantic_discovery",
    )
    output = tmp_path / "output"
    release = _release(context, output)
    release["status"] = "execution_contract_frozen"
    release["authorization"] = (
        "one-time real semantic discovery pass1 authorized"
    )
    release["run"]["diagnostic_stop_allowed"] = False
    release["permissions"] = {
        "real_pass1_authorized": True,
        "discovery_data_authorized": True,
    }
    runner = _runner(
        context,
        output=output,
        release=release,
    )
    original_atomic = discovery_module._atomic_bytes
    injected = {"raised": False}

    def fail_terminal_progress(path, payload):
        destination = Path(path)
        if (
            destination.name == "progress.json"
            and (output / "selection_authority.json").is_file()
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise OSError("fault after selection publication")
        return original_atomic(destination, payload)

    monkeypatch.setattr(
        discovery_module,
        "_atomic_bytes",
        fail_terminal_progress,
    )
    with pytest.raises(OSError, match="after selection"):
        runner.run_pass1()
    monkeypatch.setattr(
        discovery_module,
        "_atomic_bytes",
        original_atomic,
    )
    damaged = json.loads(
        (output / "progress.json").read_text(encoding="utf-8")
    )
    damaged["produced_bars"] = -999
    damaged["causal_clock"] = "2099-01-01T00:00:00+00:00"
    (output / "progress.json").write_text(
        json.dumps(damaged),
        encoding="utf-8",
    )
    source_path = Path(context["audit"]["source"]["path"])
    original_sha256 = discovery_module.sha256_file

    def forbid_source_hash(path):
        if Path(path) == source_path:
            raise AssertionError("terminal resume hashed market source")
        return original_sha256(Path(path))

    monkeypatch.setattr(
        discovery_module,
        "sha256_file",
        forbid_source_hash,
    )
    terminal_identity = source_for_pass1_release(
        release,
        resume=True,
    )
    assert isinstance(
        terminal_identity,
        FrozenParquetSourceIdentity,
    )
    monkeypatch.setattr(
        discovery_module,
        "sha256_file",
        original_sha256,
    )
    resumed = _runner(
        context,
        output=output,
        release=release,
        source_override=terminal_identity,
    )

    result = resumed.run_pass1(resume=True)
    assert result == output / "selection_authority.json"
    progress = json.loads(
        (output / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"
    assert progress["terminal_reconciled"] is True
    assert progress["produced_bars"] >= 0
    assert progress["causal_clock"] != "2099-01-01T00:00:00+00:00"


def test_resource_limits_fail_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _contracts(tmp_path / "contracts")
    output = tmp_path / "output"
    runner = _runner(
        context,
        output=output,
        release=_release(context, output),
    )
    with pytest.raises(TimeoutError, match="stall"):
        runner._enforce_resource_limits(
            last_checkpoint_monotonic=time.monotonic() - 4000,
            source_rows_admitted=0,
            force=True,
        )

    monkeypatch.setattr(
        psutil,
        "Process",
        lambda *_: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(
                rss=runner.resource_limits.rss_hard_ceiling_bytes
            )
        ),
    )
    with pytest.raises(MemoryError, match="RSS"):
        runner._enforce_resource_limits(
            last_checkpoint_monotonic=time.monotonic(),
            source_rows_admitted=0,
            force=True,
        )

    monkeypatch.setattr(
        psutil,
        "Process",
        lambda *_: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=1)
        ),
    )
    monkeypatch.setattr(
        discovery_module.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=0),
    )
    with pytest.raises(OSError, match="free-space"):
        runner._enforce_resource_limits(
            last_checkpoint_monotonic=time.monotonic(),
            source_rows_admitted=0,
            force=True,
        )


@pytest.mark.historical_frozen
def test_draft_loader_and_cli_never_open_market_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = (
        ROOT
        / "data/processed/"
        "nq_1m_previous_session_front_v2_3_pre_holdout_"
        "2017_20260331.parquet"
    )
    observed: list[Path] = []
    original_sha256 = discovery_module.sha256_file

    def recording_sha256(path):
        value = Path(path)
        observed.append(value)
        return original_sha256(value)

    monkeypatch.setattr(
        discovery_module,
        "sha256_file",
        recording_sha256,
    )
    release = load_pass1_execution_release(
        ROOT / PASS1_EXECUTION_RELEASE,
        verify_bound_files=True,
    )
    assert release["status"] == "draft_no_execution_authority"
    assert source_path not in observed

    output = ROOT / release["output"]["path"]
    assert not output.exists()
    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_v3_exp001_semantic_discovery_pass1.py",
            "run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "source access remains locked" in process.stderr
    assert not output.exists()

    override = subprocess.run(
        [
            sys.executable,
            "scripts/run_v3_exp001_semantic_discovery_pass1.py",
            "run",
            "--output",
            str(tmp_path / "override"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert override.returncode != 0
    assert "unrecognized arguments" in override.stderr
    assert not (tmp_path / "override").exists()
