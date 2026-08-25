from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pandas as pd
import numpy as np
import pytest

from scripts import train_market_representation as trainer
from smc_trader import market_cases
from smc_trader.artifact_stream import (
    atomic_bytes,
    canonical_json,
    new_stream_state,
    sha256_file,
    write_stream_manifest,
    write_stream_shard,
)
from smc_trader.market_cases import MARKET_CASE_INPUT_FIELD_TYPES
from smc_trader.market_representation import RepresentationDataError
from smc_trader.market_representation import (
    MarketRepresentationModel,
    NEUTRAL_REPRESENTATION_LOSS_WEIGHTS,
    SelfSupervisedTarget,
    TORCH_AVAILABLE,
    collate_representation_cases,
    neutral_representation_multitask_loss,
    load_neutral_representation_checkpoint,
    neutral_direct_source_preprocessing_identity,
    preprocess_neutral_direct_source_events,
    save_neutral_representation_checkpoint,
)
from smc_trader.model import Direction
from smc_trader.scene_graph import market_episode_id
from tests.test_market_representation import (
    _neutral_market_case_row,
    _with_neutral_scale_details,
)


def _artifact(
    tmp_path: Path,
    *,
    active: bool = False,
    row_count: int = 1,
    ohlcv_source: bool = False,
    repository: object | None = None,
) -> dict[str, object]:
    asof = pd.Timestamp("2024-01-08 18:05", tz="America/New_York")
    row = _neutral_market_case_row(asof=asof)
    context = json.loads(str(row["neutral_global_context_json"]))
    context["ambiguous_evidence"] = ["historical:ambiguous"]
    context["unknown_evidence"] = ["historical:unknown"]
    row["neutral_global_context_json"] = json.dumps(
        context, sort_keys=True, separators=(",", ":")
    )
    row["revision_id"] = market_cases._expected_revision_id(row)
    if active:
        row = _with_neutral_scale_details(
            row, directions=("long",) * 5, ambiguous=True
        )

    source = tmp_path / "source.parquet"
    source_rows = 100
    source_first = "2024-01-01T18:00:00-05:00"
    source_last = "2024-01-09T16:59:00-05:00"
    if ohlcv_source:
        from smc_trader.market_clock import is_registered_trading_minute

        index = pd.date_range(
            pd.Timestamp(source_first).tz_convert("America/New_York"),
            pd.Timestamp(source_last).tz_convert("America/New_York"),
            freq="1min",
        )
        index = index[index.map(is_registered_trading_minute)]
        frame = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.25,
                "volume": 10.0,
                "symbol": "NQH4",
                "instrument_id": 750,
            },
            index=index,
        )
        frame.index.name = "ts"
        frame.to_parquet(source)
        source_rows = len(frame)
        source_first, source_last = frame.index[0].isoformat(), frame.index[-1].isoformat()
        ordinal = int((frame.index < asof).sum()) - 1
        row["source_replay_ordinal"] = ordinal
        prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
        for prefix in prefixes:
            prefix["replay_view_1m_row_end_exclusive"] = ordinal + 1
        row["ohlcv_prefix_refs_json"] = json.dumps(
            prefixes, sort_keys=True, separators=(",", ":")
        )
        row["revision_id"] = market_cases._expected_revision_id(row)
    else:
        source.write_bytes(b"registered neutral source")

    rows = []
    for index in range(row_count):
        item = dict(row)
        if row_count > 1:
            location = f"location:neutral:{index:02d}"
            path = f"path:neutral:{index:02d}"
            item["entry_location_id"], item["entry_path_id"] = location, path
            item["market_episode_id"] = market_episode_id(
                str(item["market_epoch_id"]), location, path, Direction.LONG
            )
            item["revision_id"] = market_cases._expected_revision_id(item)
        rows.append(item)
    config = trainer.ROOT / "configs/model.json"
    model_config = json.loads(config.read_text(encoding="utf-8"))
    registry = json.loads((trainer.ROOT / "configs/data_splits.json").read_text())
    profile_name = "market_episode_input_smoke_2024_01_08"
    profile = registry["market_case_input_profiles"][profile_name]
    profile_identity = hashlib.sha256(canonical_json(profile)).hexdigest()
    run = {
        "schema_version": 1,
        "runner": "continuous_replay",
        "mode": "market_case_input",
        "runtime_state_schema_version": 5,
        "profile": {"name": profile_name, "identity": profile_identity},
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "rows": source_rows,
            "first": source_first,
            "last": source_last,
            "last_completed_asof": "2024-01-09T17:00:00-05:00",
            "role": "processed_continuous_front",
            "symbol": "NQH4",
            "instrument_id": 750,
        },
        "model_config": {
            "path": str(config.resolve()),
            "sha256": sha256_file(config),
            "schema_version": model_config["schema_version"],
            "tick_size": model_config["tick_size"],
            "timezone": model_config["timezone"],
        },
        "market_case_input_identity": market_cases.expected_market_case_run_identity(),
        "window": {
            "start": profile["start"],
            "end_exclusive": profile["end_exclusive"],
            "role": profile["allowed_ohlcv_role"],
            "warmup_days": profile["warmup_calendar_days"],
            "observation_clock": "completed_1m_bar_end",
            "capture_interval": "[start,end_exclusive)",
        },
        "output": {
            "stream_families": ["market_case_input_shards"],
            "shard_rows": 250,
            "checkpoint_bars": 1000,
        },
    }
    if repository is not None:
        run["runtime_state_schema_version"] = 6
        run["repository"] = repository
    run_path = tmp_path / "run_manifest.json"
    atomic_bytes(run_path, canonical_json(run))
    state = new_stream_state(MARKET_CASE_INPUT_FIELD_TYPES)
    write_stream_shard(
        tmp_path,
        "market_case_input_shards",
        rows,
        state,
        key_column="revision_id",
        field_types=MARKET_CASE_INPUT_FIELD_TYPES,
    )
    manifest_path = write_stream_manifest(
        tmp_path,
        "market_case_input_shards",
        state,
        artifact="continuous_development_market_case_input_shards",
        bindings={"run_manifest": run_path.name},
    )
    args = [
        "--neutral-dataset-audit-only",
        "--market-case-input-manifest", str(manifest_path),
        "--market-case-run-manifest", str(run_path),
    ]
    return {
        "args": args,
        "manifest": manifest_path,
        "run": run_path,
        "source": source,
        "shard": tmp_path / state["committed_shards"][0]["path"],
    }


def test_neutral_audit_is_fit_free_and_reports_active_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path)
    metrics = tmp_path / "audit.json"
    monkeypatch.setattr(
        trainer, "require_torch", lambda: pytest.fail("neutral audit imported fit")
    )
    assert trainer.main(
        [*artifact["args"], "--market-embedding-kind", "episode_created",
         "--metrics-output", str(metrics)]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(metrics.read_text())
    assert (report["rows"], report["cases"], report["embedding_kind_rows"]) == (1, 1, 1)
    assert report["kinds"]["episode_created"] == 1
    assert report["target_coverage"]["records"] == 1
    assert report["training_performed"] is report["calibration_fit_allowed"] is False
    active = report["active_scale_state"]
    assert active["denominator_rows"] == 1
    assert active["ambiguous_rows"] == active["unknown_or_disconnected_rows"] == 0
    assert set(active["by_timeframe"]) == {"4H", "1H", "15m", "5m", "1m"}


def test_active_scale_state_uses_current_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path, active=True)
    assert trainer.main(artifact["args"]) == 0
    state = json.loads(capsys.readouterr().out)["active_scale_state"]
    assert state["ambiguous_rows"] == state["unknown_or_disconnected_rows"] == 1
    assert all(
        detail["ambiguous_ratio"] == detail["unknown_or_disconnected_ratio"] == 1.0
        for detail in state["by_timeframe"].values()
    )


def test_neutral_cli_uses_only_input_and_run_manifest_identities(
    tmp_path: Path,
) -> None:
    args = _artifact(tmp_path)["args"]
    assert args == [
        "--neutral-dataset-audit-only",
        "--market-case-input-manifest",
        str(tmp_path / "market_case_input_shards.manifest.json"),
        "--market-case-run-manifest",
        str(tmp_path / "run_manifest.json"),
    ]
    option_strings = {
        option
        for action in trainer._parser()._actions
        for option in action.option_strings
    }
    assert "--market-case-input-manifest-sha" not in option_strings
    assert "--market-case-run-manifest-sha" not in option_strings


def test_neutral_single_batch_smoke_constructs_42_real_features_without_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = _artifact(
        tmp_path,
        row_count=42,
        ohlcv_source=True,
    )
    args = list(artifact["args"])
    args[0] = "--neutral-single-batch-smoke"
    monkeypatch.setattr(
        trainer,
        "_train",
        lambda *unused_args, **unused_kwargs: pytest.fail("neutral smoke fit a model"),
    )
    monkeypatch.setattr(
        trainer,
        "_atomic_jsonl",
        lambda *unused_args, **unused_kwargs: pytest.fail("neutral smoke exported rows"),
    )

    assert trainer.main(args) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["mode"] == "neutral_market_single_batch_smoke"
    assert (report["rows"], report["batches"], report["batch_rows"]) == (42, 1, 42)
    assert report["embedding_shape"] == [42, 128]
    assert report["canonical_timeframes"] == ["4h", "1h", "15m", "5m", "1m"]
    assert report["source"]["source_rows"] > 8
    assert report["feature_max_at"] <= report["input_max_asof"]
    assert report["target_contract"]["active"] == [
        "next_lifecycle",
        "scale_direction_alignment",
    ]
    assert all(
        report[name] is False
        for name in (
            "optimizer_created",
            "backward_performed",
            "training_performed",
            "checkpoint_written",
            "artifacts_exported",
            "outcome_fields_used",
            "brain_used",
            "shadow_used",
        )
    )
    assert all(
        report["loss_components"][name] == 0.0
        for name in (
            "next_event",
            "next_event_time",
            "displacement",
            "draw_consumed",
        )
    )


def test_neutral_trainer_accepts_optional_repository_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = _artifact(tmp_path, repository={"commit": "a" * 40})
    assert trainer.main(artifact["args"]) == 0
    verification = json.loads(capsys.readouterr().out)["run_lineage"][
        "identity_verification"
    ]
    assert verification["repository"] == {"commit": "a" * 40}


def test_neutral_trainer_rejects_invalid_optional_repository_identity(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, repository={"commit": "A" * 40})
    with pytest.raises(RepresentationDataError, match="repository identity"):
        trainer.main(artifact["args"])


@pytest.mark.parametrize(
    ("change", "message"),
    (("no_mode", "exactly one"), ("two_modes", "exactly one"),
     ("legacy", "mutually exclusive"),
     ("training", "forbid training/export")),
)
def test_neutral_cli_is_mutually_exclusive_and_audit_only(
    tmp_path: Path, change: str, message: str
) -> None:
    args = list(_artifact(tmp_path)["args"])
    if change == "no_mode":
        args.remove("--neutral-dataset-audit-only")
    elif change == "two_modes":
        args.append("--neutral-single-batch-smoke")
    elif change == "legacy":
        args.append("--synthetic-smoke")
    else:
        args.extend(("--epochs", "0"))
    with pytest.raises(RepresentationDataError, match=message):
        trainer.main(args)


def _b0_cli_args(metrics: Path | None = None) -> list[str]:
    args = ["--neutral-b0-validation"]
    for role in ("train", "validation"):
        args.extend(("--market-case-input-manifest", f"{role}-input.json",
                     "--market-case-run-manifest", f"{role}-run.json"))
    if metrics is not None:
        args.extend(("--metrics-output", str(metrics)))
    return args


def _b1_cli_args(metrics: Path | None = None, parent: Path | None = None) -> list[str]:
    args = ["--neutral-b1-validation"]
    for role in ("train", "validation"):
        args.extend(("--market-case-input-manifest", f"{role}-input.json",
                     "--market-case-run-manifest", f"{role}-run.json"))
    if parent is not None:
        args.extend(("--parent-b0-metrics", str(parent)))
    if metrics is not None:
        args.extend(("--metrics-output", str(metrics)))
    return args


def test_neutral_b0_cli_freezes_training_contract_and_requires_metrics(
    tmp_path: Path,
) -> None:
    args = trainer._parser().parse_args(_b0_cli_args(tmp_path / "metrics.json"))
    trainer._validate_neutral_cli(args, argv=_b0_cli_args(tmp_path / "metrics.json"))
    assert (args.ensemble_size, args.seed, args.batch_size, args.epochs) == (
        3, 17, 16, 10,
    )
    assert args.learning_rate == 3e-4
    with pytest.raises(RepresentationDataError, match="metrics-output"):
        missing = trainer._parser().parse_args(_b0_cli_args())
        trainer._validate_neutral_cli(missing, argv=_b0_cli_args())
    overridden = [*_b0_cli_args(tmp_path / "metrics.json"), "--epochs", "9"]
    with pytest.raises(RepresentationDataError, match="freezes"):
        trainer._validate_neutral_cli(
            trainer._parser().parse_args(overridden), argv=overridden
        )
    non_cpu = [*_b0_cli_args(tmp_path / "metrics.json"), "--device", "mps"]
    with pytest.raises(RepresentationDataError, match="CPU"):
        trainer._validate_neutral_cli(
            trainer._parser().parse_args(non_cpu), argv=non_cpu
        )


def test_neutral_b1_cli_freezes_parent_and_training_contract(tmp_path: Path) -> None:
    argv = _b1_cli_args(tmp_path / "b1.json", tmp_path / "b0.json")
    args = trainer._parser().parse_args(argv)
    trainer._validate_neutral_cli(args, argv=argv)
    assert (args.ensemble_size, args.seed, args.batch_size, args.epochs) == (
        3, 17, 16, 10,
    )
    assert args.learning_rate == 3e-4
    with pytest.raises(RepresentationDataError, match="parent-b0-metrics"):
        missing = _b1_cli_args(tmp_path / "b1.json")
        trainer._validate_neutral_cli(
            trainer._parser().parse_args(missing), argv=missing
        )
    overridden = [*argv, "--epochs", "9"]
    with pytest.raises(RepresentationDataError, match="freezes"):
        trainer._validate_neutral_cli(
            trainer._parser().parse_args(overridden), argv=overridden
        )


def test_neutral_b0_trainer_identity_requires_clean_tracked_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = iter((SimpleNamespace(stdout="a" * 40 + "\n"),
                  SimpleNamespace(stdout="")))
    monkeypatch.setattr(trainer.subprocess, "run", lambda *args, **kwargs: next(clean))
    assert trainer._neutral_b0_trainer_commit() == "a" * 40
    dirty = iter((SimpleNamespace(stdout="a" * 40 + "\n"),
                  SimpleNamespace(stdout=" M scripts/train_market_representation.py\n")))
    monkeypatch.setattr(trainer.subprocess, "run", lambda *args, **kwargs: next(dirty))
    with pytest.raises(RepresentationDataError, match="clean trainer"):
        trainer._neutral_b0_trainer_commit()


@pytest.mark.parametrize(("path", "replacement"), (
    (("max_epochs",), 10.0),
    (("gates", "neighbor_coverage_required"), True),
))
def test_neutral_b0_registry_rejects_equal_but_wrong_json_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...], replacement: object,
) -> None:
    payload = json.loads(trainer.DATA_SPLITS.read_text())
    contract = payload["neutral_representation_b0_validation"]
    parent = contract
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = replacement
    registry = tmp_path / "data_splits.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(trainer, "DATA_SPLITS", registry)
    with pytest.raises(RepresentationDataError, match="frozen contract"):
        trainer._neutral_b0_registry_contract()


def test_neutral_b1_registry_rejects_equal_but_wrong_json_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(trainer.DATA_SPLITS.read_text())
    payload["neutral_representation_b1_validation"][
        "final_embedding_regularizer"
    ]["invariance"]["weight"] = 5
    registry = tmp_path / "data_splits.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(trainer, "DATA_SPLITS", registry)
    with pytest.raises(RepresentationDataError, match="frozen contract"):
        trainer._neutral_b1_registry_contract()


def test_neutral_b0_rejects_holdout_before_loading_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = iter((
        "neutral_representation_train_2021_02",
        "neutral_representation_holdout_2025_02",
    ))
    monkeypatch.setattr(trainer, "_neutral_run_profile_name", lambda path: next(names))
    monkeypatch.setattr(
        trainer, "_load_neutral_market_dataset",
        lambda **kwargs: pytest.fail("B0 opened a dataset before profile preflight"),
    )
    with pytest.raises(RepresentationDataError, match="ordered 2021-02"):
        trainer._load_neutral_fit_collection(
            input_manifest_paths=("train-input", "holdout-input"),
            run_manifest_paths=("train-run", "holdout-run"),
            validation_only=True,
        )


def test_neutral_b0_metrics_are_atomically_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics = tmp_path / "b0.json"
    monkeypatch.setattr(trainer, "_load_neutral_fit_collection", lambda **kwargs: {})
    monkeypatch.setattr(
        trainer, "_neutral_b0_validation_report",
        lambda *args, **kwargs: {"proof": "b0", "criteria_met": True},
    )
    monkeypatch.setattr(trainer, "_neutral_b0_trainer_commit", lambda: "a" * 40)
    monkeypatch.setattr(trainer, "_validate_neutral_b0_metrics_protocol", lambda value: None)
    reads: list[dict[str, object]] = []
    def readback(path: str) -> Mapping[str, object]:
        reads.append(json.loads(Path(path).read_text()))
        return reads[-1]
    monkeypatch.setattr(trainer, "load_neutral_b0_validation_metrics", readback)
    assert trainer.main(_b0_cli_args(metrics)) == 0
    assert reads == [{"criteria_met": True, "proof": "b0"}]
    assert json.loads(capsys.readouterr().out) == reads[0]


def test_neutral_b0_failed_criteria_writes_report_and_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = tmp_path / "b0-failed.json"
    monkeypatch.setattr(trainer, "_load_neutral_fit_collection", lambda **kwargs: {})
    monkeypatch.setattr(
        trainer, "_neutral_b0_validation_report",
        lambda *args, **kwargs: {"proof": "b0", "criteria_met": False},
    )
    monkeypatch.setattr(trainer, "_neutral_b0_trainer_commit", lambda: "a" * 40)
    monkeypatch.setattr(trainer, "_validate_neutral_b0_metrics_protocol", lambda value: None)
    monkeypatch.setattr(
        trainer, "load_neutral_b0_validation_metrics",
        lambda path: json.loads(Path(path).read_text()),
    )
    assert trainer.main(_b0_cli_args(metrics)) == 2
    assert json.loads(metrics.read_text()) == {"criteria_met": False, "proof": "b0"}


def test_neutral_b1_parent_b0_is_exact_failed_run(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _valid_failed_neutral_b0_metrics_payload()
    path = tmp_path / "b0.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    contract = copy.deepcopy(trainer.NEUTRAL_B1_VALIDATION_CONTRACT)
    contract["parent_b0_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract["parent_b0_trainer_commit"] = payload["trainer_repository_commit"]
    contract["parent_b0_inputs"] = {
        item["split_role"]: {
            "rows": payload["split_rows"][item["split_role"]],
            "input_manifest_sha256": item["input_manifest_sha256"],
            "run_manifest_sha256": item["run_manifest_sha256"],
        }
        for item in payload["lineage"]["input_runs"]
    }
    monkeypatch.setattr(trainer, "NEUTRAL_B1_VALIDATION_CONTRACT", contract)
    assert trainer._neutral_b1_parent_b0_metrics(path)["criteria_met"] is False
    payload = _valid_neutral_b0_metrics_payload()
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    contract["parent_b0_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RepresentationDataError, match="frozen failed B0"):
        trainer._neutral_b1_parent_b0_metrics(path)


def test_neutral_b1_manifest_binding_fails_before_dataset_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _valid_failed_neutral_b0_metrics_payload()
    expected = {
        item["input_manifest_sha256"] for item in parent["lineage"]["input_runs"]
    } | {item["run_manifest_sha256"] for item in parent["lineage"]["input_runs"]}
    supplied = iter((*sorted(expected)[:-1], "f" * 64))
    monkeypatch.setattr(trainer, "_sha256_file", lambda path: next(supplied))
    monkeypatch.setattr(
        trainer, "_load_neutral_market_dataset",
        lambda **kwargs: pytest.fail("B1 opened a dataset before parent binding"),
    )
    with pytest.raises(RepresentationDataError, match="differ from parent B0"):
        trainer._load_neutral_fit_collection(
            input_manifest_paths=("train-input", "validation-input"),
            run_manifest_paths=("train-run", "validation-run"),
            validation_protocol="b1", parent_b0_metrics=parent,
        )


@pytest.mark.parametrize(
    "change",
    (
        "noncanonical", "shard", "columns", "path", "rows", "first_key",
        "source", "config_hash", "config_values", "profile",
    ),
)
def test_neutral_artifact_integrity_fails_closed(tmp_path: Path, change: str) -> None:
    artifact = _artifact(tmp_path)
    args = list(artifact["args"])
    manifest = Path(artifact["manifest"])
    shard = Path(artifact["shard"])
    if change == "noncanonical":
        manifest.write_text(json.dumps(json.loads(manifest.read_text()), indent=2))
    elif change == "shard":
        shard.write_bytes(shard.read_bytes() + b"tamper")
    elif change == "columns":
        import pyarrow.parquet as pq

        table = pq.read_table(shard)
        pq.write_table(table.select(list(reversed(table.column_names))), shard)
        payload = json.loads(manifest.read_text())
        payload["shards"][0]["sha256"] = sha256_file(shard)
        atomic_bytes(manifest, canonical_json(payload))
    elif change in {"path", "rows", "first_key"}:
        payload = json.loads(manifest.read_text())
        payload["shards"][0][change] = (
            "wrong.parquet" if change == "path" else 2 if change == "rows" else "wrong"
        )
        atomic_bytes(manifest, canonical_json(payload))
    elif change == "source":
        Path(artifact["source"]).write_bytes(b"tampered source")
    else:
        run = Path(artifact["run"])
        payload = json.loads(run.read_text())
        if change == "config_hash":
            payload["model_config"]["sha256"] = "0" * 64
        elif change == "config_values":
            payload["model_config"]["tick_size"] = 0.5
        else:
            payload["profile"]["identity"] = "not-a-hash"
        atomic_bytes(run, canonical_json(payload))
    with pytest.raises(RepresentationDataError):
        trainer.main(args)


def test_legacy_default_does_not_enter_neutral_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    assert trainer._neutral_mode_requested(trainer._parser().parse_args([])) is False
    monkeypatch.setattr(
        trainer, "require_torch", lambda: (_ for _ in ()).throw(RuntimeError("legacy"))
    )
    with pytest.raises(RuntimeError, match="legacy"):
        trainer.main([])


def _neutral_targets(count: int) -> tuple[SelfSupervisedTarget, ...]:
    return tuple(
        SelfSupervisedTarget(
            next_event_type=-100,
            next_lifecycle=5 if index + 1 < count else -100,
            next_event_time_bucket=-100,
            displacement_state=-100,
            draw_consumed=-100,
            scale_direction_alignment=index % 3,
        )
        for index in range(count)
    )


def _valid_neutral_lineage(roles: tuple[str, ...]) -> dict[str, object]:
    profiles = {
        "train": "neutral_representation_train_2021_02",
        "validation": "neutral_representation_validation_2022_05",
        "holdout": "neutral_representation_holdout_2025_02",
    }
    runs = [{
        "profile_name": profiles[role], "split_role": role,
        "input_manifest_path": f"/inputs/{role}.json",
        "input_manifest_sha256": f"{index + 1:064x}",
        "run_manifest_path": f"/runs/{role}.json",
        "run_manifest_sha256": f"{index + 11:064x}",
        "repository_commit": f"{index + 1:040x}",
    } for index, role in enumerate(roles)]
    observed = [{
        "left_profile": profiles[left], "right_profile": profiles[right],
        "purge_end": "2021-03-15T00:00:00-04:00",
        "next_prefix_start": "2022-04-17T00:00:00-04:00",
        "observed_session_count": 5,
        "first_observed_session": "2021-03-16",
        "last_observed_session": "2021-03-22",
    } for left, right in zip(roles[:-1], roles[1:])]
    return {
        "input_runs": runs,
        "source_identity": {"sha256": "a" * 64},
        "model_config_identity": {
            "sha256": "b" * 64, "timezone": "America/New_York",
        },
        "market_case_protocol": market_cases.expected_market_case_run_identity(),
        "representation_feature_schema_version": trainer.FEATURE_SCHEMA_VERSION,
        "split_protocol": {
            "registry_sha256": "c" * 64,
            "protocol_version": "neutral-representation-splits-1.0.0",
            "warmup_calendar_days": 14, "purge_calendar_days": 14,
            "embargo_trading_days": 5,
            "market_episode_split_key": [
                "run_manifest_sha256", "market_epoch_id", "market_episode_id",
            ],
            "actual_prefix_exposure_verified": True,
            "observed_completed_session_embargo": observed,
        },
    }


def _valid_neutral_fit_metrics_payload() -> dict[str, object]:
    losses = {
        "candle_reconstruction": 0.1,
        "event_reconstruction": 0.2,
        "next_lifecycle": 0.3,
        "scale_alignment": 0.4,
    }
    role = {
        "rows": 1,
        "total_loss": 0.725,
        "objective_losses": losses,
        "active_head_metrics": {
            "next_lifecycle": {"labelled_rows": 1, "nll": 0.5, "accuracy": 1.0},
            "scale_direction_alignment": {
                "labelled_rows": 1, "nll": 0.6, "accuracy": 0.0,
            },
        },
        "geometry": {
            "embedding_dim": 128,
            "effective_rank": 1.0,
            "mean_feature_std": 0.1,
            "centroid_norm": 0.8,
        },
        "label_sources_masked": True,
        "direct_source_preprocessing": neutral_direct_source_preprocessing_identity(),
        "gradient_enabled": False,
    }
    metrics = {
        "model_version": trainer.MODEL_VERSION,
        "parameter_count": 1,
        "embedding_dim": 128,
        "epochs": 1,
        "epoch_training_loss": [0.725],
        "epoch_metrics": [{
            "epoch": 1,
            "train": copy.deepcopy(role),
            "validation": copy.deepcopy(role),
        }],
        "split_counts": {"train": 1, "validation": 1, "holdout": 1},
        "objectives": {
            "loss_weights": dict(NEUTRAL_REPRESENTATION_LOSS_WEIGHTS),
            "active_heads": list(trainer.NEUTRAL_SPARSE_ACTIVE_TARGETS),
            "disabled_heads": list(trainer.NEUTRAL_SPARSE_DISABLED_TARGETS),
        },
        "train": copy.deepcopy(role),
        "validation": copy.deepcopy(role),
        "holdout": copy.deepcopy(role),
        "direct_source_preprocessing": neutral_direct_source_preprocessing_identity(),
        "optimized_roles": ["train"],
        "validation_used_for_optimization": False,
        "holdout_used_for_optimization": False,
        "holdout_used_for_selection": False,
        "model_selection_performed": False,
        "threshold_search_performed": False,
        "outcome_fields_used": False,
        "model_capability_validated": False,
        "trading_edge_claimed": False,
    }
    members = [{
        "member_id": f"member-{index:03d}",
        "seed": 17 + index * 100_003,
        "checkpoint_id": f"{index + 1:064x}",
        "checkpoint_path": f"/models/member-{index:03d}.pt",
        "metrics": copy.deepcopy(metrics),
    } for index in range(3)]
    return {
        "schema_version": 2,
        "mode": "neutral_market_representation_fit",
        "pipeline_scope": "three_window_pipeline_smoke",
        "rows": 3,
        "split_rows": {"train": 1, "validation": 1, "holdout": 1},
        "ensemble_members": members,
        "artifacts": {
            "embeddings": {
                "path": "/artifacts/embeddings.jsonl",
                "manifest": "/artifacts/embeddings.jsonl.manifest.json",
                "records": 3,
            },
            "active_heads": {
                "path": "/artifacts/heads.jsonl",
                "manifest": "/artifacts/heads.jsonl.manifest.json",
                "records": 9,
            },
        },
        "lineage": _valid_neutral_lineage(("train", "validation", "holdout")),
        "direct_source_preprocessing": neutral_direct_source_preprocessing_identity(),
        "training_performed": True,
        "validation_used_for_optimization": False,
        "holdout_used_for_optimization": False,
        "holdout_used_for_selection": False,
        "model_selection_performed": False,
        "threshold_search_performed": False,
        "outcome_fields_used": False,
        "model_capability_validated": False,
        "retrieval_quality_validated": False,
        "ood_capability_validated": False,
        "trading_edge_claimed": False,
        "action_value_claimed": False,
    }


def _valid_neutral_b0_metrics_payload() -> dict[str, object]:
    losses = {
        "candle_reconstruction": 0.1,
        "event_reconstruction": 0.1,
        "next_lifecycle": 0.1,
        "scale_alignment": 0.1,
    }
    role = {
        "rows": 128,
        "total_loss": 0.325,
        "objective_losses": losses,
        "active_head_metrics": {
            name: {
                "labelled_rows": 128, "nll": 0.5,
                "accuracy": 0.8, "balanced_accuracy": 0.8,
            }
            for name in ("next_lifecycle", "scale_direction_alignment")
        },
        "masked_reconstruction": {
            "candle_by_timeframe": {
                timeframe: {
                    "masked_values": 16, "smooth_l1": 0.5,
                    "zero_after_frozen_feature_normalization_smooth_l1": 1.0,
                    "relative_improvement": 0.5,
                }
                for timeframe in trainer.TIMEFRAMES
            },
            "event": {"masked_events": 16, "nll": 0.5},
        },
        "geometry": {
            "embedding_dim": 128, "effective_rank": 20.0,
            "mean_feature_std": 0.1, "centroid_norm": 0.1,
            "unit_centroid_norm": 0.1, "raw_p95_pairwise_cosine": 0.5,
        },
        "material_geometry": {
            kind: {"rows": 128, "effective_rank": 20.0}
            for kind in trainer.NEUTRAL_MARKET_TRANSITION_KINDS
        },
    }
    members = [{
        "member_id": f"member-{index:03d}", "seed": seed,
        "parameter_count": 1, "embedding_dim": 128,
        "epoch_metrics": [{
            "epoch": epoch,
            "train": copy.deepcopy(role),
            "validation": copy.deepcopy(role),
        } for epoch in range(1, 11)],
    } for index, seed in enumerate((17, 100_020, 200_023))]
    neighbor = {
        "eligible_material_queries": 1, "covered_material_queries": 1,
        "neighbors": 10, "k": 10, "purity": 0.8,
        "train_chance": 0.5, "lift": 0.3,
    }
    gates = [{
        "epoch": epoch,
        "members": [{
            "member_id": f"member-{index:03d}",
            "event_relative_improvement": 0.5,
            "event_train_prior_nll": 1.0,
            "neighbor": copy.deepcopy(neighbor),
        } for index in range(3)],
        "active_heads": {
            name: {
                "ensemble_nll": 0.5, "train_prior_nll": 1.0,
                "train_prior_balanced_accuracy": 0.3,
                "member_balanced_accuracy": [0.8, 0.8, 0.8],
            }
            for name in ("next_lifecycle", "scale_direction_alignment")
        },
        "common_pass": True,
    } for epoch in range(1, 11)]
    contract = copy.deepcopy(trainer.NEUTRAL_B0_VALIDATION_CONTRACT)
    contract_sha = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    return {
        "schema_version": 1, "mode": "neutral_b0_validation",
        "pipeline_scope": "2021_02_train_2022_05_validation_only",
        "rows": 256, "split_rows": {"train": 128, "validation": 128},
        "contract": {
            "value": contract, "sha256": contract_sha,
            "registry_sha256": "c" * 64,
        },
        "trainer_repository_commit": "d" * 40,
        "ensemble_members": members, "epoch_gates": gates,
        "selected_epoch": 1,
        "lineage": _valid_neutral_lineage(("train", "validation")),
        "direct_source_preprocessing": neutral_direct_source_preprocessing_identity(),
        "training_performed": True,
        "validation_used_for_selection": True,
        "validation_used_for_optimization": False,
        "holdout_opened": False,
        "holdout_used_for_optimization": False,
        "holdout_used_for_selection": False,
        "outcome_fields_used": False,
        "artifacts_exported": False,
        "threshold_search_performed": False,
        "criteria_met": True,
        "model_capability_validated": False,
        "trading_edge_claimed": False,
        "action_value_claimed": False,
    }


def _valid_failed_neutral_b0_metrics_payload() -> dict[str, object]:
    payload = _valid_neutral_b0_metrics_payload()
    for gate in payload["epoch_gates"]:
        for member in gate["members"]:
            member["neighbor"].update(
                purity=0.0, train_chance=0.5, lift=-0.5
            )
        gate["common_pass"] = False
    payload["selected_epoch"] = None
    payload["criteria_met"] = False
    return payload


def _valid_neutral_b1_metrics_payload() -> dict[str, object]:
    gate = _valid_failed_neutral_b0_metrics_payload()
    gate["rows"] = 2469
    gate["split_rows"] = {"train": 1190, "validation": 1279}
    gate["trainer_repository_commit"] = "e" * 40
    parent_inputs = trainer.NEUTRAL_B1_VALIDATION_CONTRACT["parent_b0_inputs"]
    for run in gate["lineage"]["input_runs"]:
        expected = parent_inputs[run["split_role"]]
        run["input_manifest_sha256"] = expected["input_manifest_sha256"]
        run["run_manifest_sha256"] = expected["run_manifest_sha256"]
    for member in gate["ensemble_members"]:
        for epoch in member["epoch_metrics"]:
            epoch["train"]["rows"] = 1190
            epoch["validation"]["rows"] = 1279
    regularizer_epoch = {
        "epoch": 1,
        "optimizer_updates": 75,
        "task_rows": 1190,
        "vicreg_batches": 74,
        "vicreg_rows": 1184,
        "task_only_batches": 1,
        "task_only_rows": 6,
        "loss_reduction": "sum_of_batch_means",
        "loss_sums": {
            "view_a_task": 10.0, "view_b_task": 12.0, "task_mean": 11.0,
            "invariance": 1.0, "variance": 2.0, "covariance": 3.0,
            "weighted_invariance": 5.0, "weighted_variance": 10.0,
            "weighted_covariance": 0.6, "combined": 26.6,
        },
        "task_component_sums": {name: 0.0 for name in (
            "candle_reconstruction", "event_reconstruction", "next_event",
            "next_lifecycle", "next_event_time", "displacement",
            "draw_consumed", "scale_alignment", "contrastive",
        )},
        "active_batch_diagnostic_means": {
            "first_view_mean_std": 0.8,
            "second_view_mean_std": 0.8,
            "first_view_below_gamma_dim_count": 64.0,
            "second_view_below_gamma_dim_count": 64.0,
            "mean_pair_cosine": 0.7,
        },
    }
    regularizer_members = [{
        "member_id": f"member-{index:03d}", "seed": seed,
        "epoch_metrics": [
            {**copy.deepcopy(regularizer_epoch), "epoch": epoch}
            for epoch in range(1, 11)
        ],
    } for index, seed in enumerate((17, 100_020, 200_023))]
    contract = copy.deepcopy(trainer.NEUTRAL_B1_VALIDATION_CONTRACT)
    contract_sha = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    return {
        "schema_version": 1,
        "mode": "neutral_b1_vicreg_validation",
        "pipeline_scope": "2021_02_train_2022_05_validation_only",
        "parent_b0": {
            "metrics_sha256": contract["parent_b0_metrics_sha256"],
            "contract_sha256": contract["parent_b0_contract_sha256"],
            "trainer_repository_commit": contract["parent_b0_trainer_commit"],
        },
        "contract": {
            "value": contract, "sha256": contract_sha,
            "registry_sha256": "c" * 64,
        },
        "trainer_repository_commit": "e" * 40,
        "b0_gate_evaluation": gate,
        "regularizer_members": regularizer_members,
        "b0_gates_reused_without_change": True,
        "new_validation_thresholds_added": False,
        "regularizer_diagnostics_used_for_selection": False,
        "true_per_scale_alignment_used": False,
        "selected_epoch": None,
        "criteria_met": False,
        "training_performed": True,
        "validation_used_for_selection": True,
        "validation_used_for_optimization": False,
        "holdout_opened": False,
        "holdout_used_for_optimization": False,
        "holdout_used_for_selection": False,
        "outcome_fields_used": False,
        "artifacts_exported": False,
        "threshold_search_performed": False,
        "model_capability_validated": False,
        "trading_edge_claimed": False,
        "action_value_claimed": False,
    }


def test_neutral_loss_has_zero_weight_and_fail_closed_disabled_heads() -> None:
    trainer.require_torch()
    examples, _ = trainer._synthetic_examples(19)
    selected = examples[:3]
    targets = _neutral_targets(len(selected))
    batch, target_batch = collate_representation_cases(
        selected, targets=targets, mask_probability=0.15, seed=3
    )
    assert target_batch is not None
    model = MarketRepresentationModel()
    breakdown = neutral_representation_multitask_loss(
        model(batch), batch, target_batch
    )
    assert {
        name: NEUTRAL_REPRESENTATION_LOSS_WEIGHTS[name]
        for name in (
            "next_event",
            "next_event_time",
            "displacement",
            "draw_consumed",
            "contrastive",
        )
    } == {
        "next_event": 0.0,
        "next_event_time": 0.0,
        "displacement": 0.0,
        "draw_consumed": 0.0,
        "contrastive": 0.0,
    }
    assert all(
        float(breakdown.components[name]) == 0.0
        for name in (
            "next_event",
            "next_event_time",
            "displacement",
            "draw_consumed",
        )
    )

    tampered = list(targets)
    tampered[0] = SelfSupervisedTarget(
        next_event_type=2,
        next_lifecycle=5,
        next_event_time_bucket=-100,
        displacement_state=-100,
        draw_consumed=-100,
        scale_direction_alignment=1,
    )
    _, tampered_batch = collate_representation_cases(
        selected, targets=tampered, mask_probability=0.15, seed=3
    )
    assert tampered_batch is not None
    with pytest.raises(RepresentationDataError, match="legacy target"):
        neutral_representation_multitask_loss(model(batch), batch, tampered_batch)


def test_actual_prefix_overlap_and_short_observed_embargo_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SimpleNamespace(purge_calendar_days=14, embargo_trading_days=5)
    left = {
        "profile_name": "train",
        "registered_window": SimpleNamespace(
            warmup_start=pd.Timestamp("2021-01-18", tz="America/New_York")
        ),
        "prefix_exposure_end": pd.Timestamp(
            "2021-03-01", tz="America/New_York"
        ),
        "prefix_exposure_start": pd.Timestamp(
            "2021-01-18", tz="America/New_York"
        ),
        "run_identity": {"normalized": {"source_path": "/source.parquet"}},
    }
    right = {
        "profile_name": "validation",
        "registered_window": SimpleNamespace(
            warmup_start=pd.Timestamp("2021-03-05", tz="America/New_York")
        ),
        "prefix_exposure_start": pd.Timestamp(
            "2021-03-05", tz="America/New_York"
        ),
        "prefix_exposure_end": pd.Timestamp(
            "2021-04-01", tz="America/New_York"
        ),
        "run_identity": {"normalized": {"source_path": "/source.parquet"}},
    }
    with pytest.raises(RepresentationDataError, match="purge/embargo"):
        trainer._validate_neutral_actual_exposure_boundaries(
            (left, right), registry
        )

    right["registered_window"] = SimpleNamespace(
        warmup_start=pd.Timestamp("2021-04-01", tz="America/New_York")
    )
    right["prefix_exposure_start"] = pd.Timestamp(
        "2021-04-01", tz="America/New_York"
    )
    monkeypatch.setattr(
        trainer,
        "_observed_completed_sessions",
        lambda *args, **kwargs: ("s1", "s2", "s3", "s4"),
    )
    with pytest.raises(RepresentationDataError, match="fewer than five"):
        trainer._validate_neutral_actual_exposure_boundaries(
            (left, right), registry
        )
    monkeypatch.setattr(
        trainer,
        "_observed_completed_sessions",
        lambda *args, **kwargs: ("s1", "s2", "s3", "s4", "s5"),
    )
    report = trainer._validate_neutral_actual_exposure_boundaries(
        (left, right), registry
    )
    assert report[0]["observed_session_count"] == 5


def test_observed_embargo_counts_only_complete_globex_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smc_trader import io
    from smc_trader.market_clock import _session_bounds, is_registered_trading_minute

    bounds = _session_bounds(pd.Timestamp("2021-03-15").date())
    assert bounds is not None
    opened, closed = bounds
    expected = pd.DatetimeIndex(
        stamp
        for stamp in pd.date_range(
            opened,
            closed - pd.Timedelta(minutes=1),
            freq="1min",
        )
        if is_registered_trading_minute(stamp)
    )
    frame = pd.DataFrame({"close": 0.0}, index=expected)
    monkeypatch.setattr(
        io,
        "load_ohlcv",
        lambda *args, **kwargs: SimpleNamespace(frame=frame),
    )
    assert trainer._observed_completed_sessions(
        "/source.parquet", start=opened, end=closed
    ) == ("2021-03-15",)

    frame = pd.DataFrame({"close": 0.0}, index=expected[1:])
    assert trainer._observed_completed_sessions(
        "/source.parquet", start=opened, end=closed
    ) == ()


def test_neutral_fit_rejects_registered_profile_role_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = {
        "train": "neutral_representation_train_2021_02",
        "validation": "neutral_representation_validation_2022_05",
        "holdout": "neutral_representation_holdout_2025_02",
    }
    starts = {
        "train": pd.Timestamp("2021-02-01", tz="America/New_York"),
        "validation": pd.Timestamp("2022-05-01", tz="America/New_York"),
        "holdout": pd.Timestamp("2025-02-01", tz="America/New_York"),
    }
    profiles = {
        name: {"representation_split_role": role}
        for role, name in names.items()
    }
    windows = {
        name: SimpleNamespace(
            representation_split_role=role,
            start=starts[role],
            end_exclusive=starts[role] + pd.DateOffset(months=1),
            warmup_start=starts[role] - pd.DateOffset(days=14),
            allowed_ohlcv_role=f"source-{role}",
            expected_symbol=f"NQ-{role}",
            expected_instrument_id=index,
        )
        for index, (role, name) in enumerate(names.items(), start=1)
    }
    registry = SimpleNamespace(
        windows=windows,
        smoke_profiles=names,
        warmup_calendar_days=14,
        purge_calendar_days=14,
        embargo_trading_days=5,
    )
    loaded: dict[str, Mapping[str, object]] = {}
    for role, name in names.items():
        window = windows[name]
        profile = profiles[name]
        loaded[f"{role}-run"] = {
            "rows": ({"revision_id": f"revision-{role}"},),
            "input_sha": role[0] * 64,
            "run_sha": (role[0] + "1") * 32,
            "run_manifest": {
                "profile": {
                    "name": name,
                    "identity": trainer._neutral_profile_sha256(profile),
                },
                "window": {"warmup_days": 14},
                "market_case_input_identity": {"protocol": "neutral-v1"},
            },
            "run_identity": {
                "normalized": {
                    "window_start": window.start,
                    "window_end_exclusive": window.end_exclusive,
                    "window_role": window.allowed_ohlcv_role,
                    "symbol": window.expected_symbol,
                    "instrument_id": window.expected_instrument_id,
                        "source_sha256": "a" * 64,
                        "model_config_sha256": "b" * 64,
                        "timezone": "America/New_York",
                },
                "repository": {"commit": "c" * 40},
            },
        }
    monkeypatch.setattr(
        trainer,
        "_neutral_split_registry",
        lambda: (registry, profiles, "d" * 64),
    )
    monkeypatch.setattr(
        trainer,
        "_load_neutral_market_dataset",
        lambda *, run_manifest_path, **kwargs: loaded[run_manifest_path],
    )
    monkeypatch.setattr(
        trainer,
        "_neutral_episode_prefix_exposure",
        lambda dataset: (
            dataset["registered_window"].warmup_start,
            dataset["registered_window"].end_exclusive,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_validate_neutral_actual_exposure_boundaries",
        lambda *args, **kwargs: (),
    )
    loaded["validation-run"]["run_identity"]["repository"] = {"commit": "d" * 40}
    loaded["holdout-run"]["run_identity"]["repository"] = {"commit": "e" * 40}
    accepted = trainer._load_neutral_fit_collection(
        input_manifest_paths=("i1", "i2", "i3"),
        run_manifest_paths=("train-run", "validation-run", "holdout-run"),
    )
    assert {
        item["run_identity"]["repository"]["commit"]
        for item in accepted["datasets"]
    } == {"c" * 40, "d" * 40, "e" * 40}

    profiles[names["train"]]["representation_split_role"] = "validation"
    with pytest.raises(RepresentationDataError, match="role was tampered"):
        trainer._load_neutral_fit_collection(
            input_manifest_paths=("i1", "i2", "i3"),
            run_manifest_paths=("train-run", "validation-run", "holdout-run"),
        )


def test_neutral_optimizer_reads_only_train_and_eval_never_backwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse
    import torch

    examples, _ = trainer._synthetic_examples(23)
    selected = tuple(
        preprocess_neutral_direct_source_events(example)
        for example in examples[:9]
    )
    targets = _neutral_targets(len(selected))
    roles = ("train",) * 3 + ("validation",) * 3 + ("holdout",) * 3
    splits = {
        example.case.revision_id: role
        for example, role in zip(selected, roles, strict=True)
    }
    original = trainer.neutral_representation_multitask_loss
    calls: list[tuple[bool, tuple[str, ...]]] = []

    def observed_loss(output: object, batch: object, target_batch: object) -> object:
        calls.append((torch.is_grad_enabled(), tuple(batch.revision_ids)))
        return original(output, batch, target_batch)

    monkeypatch.setattr(
        trainer, "neutral_representation_multitask_loss", observed_loss
    )
    args = argparse.Namespace(
        epochs=1,
        batch_size=3,
        learning_rate=1e-4,
        seed=29,
        device="cpu",
    )
    _, metrics = trainer._train_neutral_member(
        selected, targets, splits, args
    )
    train_ids = {
        revision_id for revision_id, role in splits.items() if role == "train"
    }
    assert calls
    assert all(set(revisions) <= train_ids for grad, revisions in calls if grad)
    evaluation_calls = [
        grad for grad, revisions in calls if not set(revisions) <= train_ids
    ]
    assert evaluation_calls and not any(evaluation_calls)
    assert metrics["optimized_roles"] == ["train"]
    assert metrics["holdout_used_for_optimization"] is False
    assert metrics["holdout_used_for_selection"] is False
    assert metrics["model_selection_performed"] is False
    assert len(metrics["epoch_metrics"]) == 1
    assert set(metrics["epoch_metrics"][0]) == {"epoch", "train", "validation"}
    for role in ("train", "validation"):
        measured = metrics["epoch_metrics"][0][role]
        assert set(measured["objective_losses"]) == {
            "candle_reconstruction", "event_reconstruction",
            "next_lifecycle", "scale_alignment",
        }
        assert set(measured["active_head_metrics"]) == {
            "next_lifecycle", "scale_direction_alignment",
        }
        assert set(measured["geometry"]) == {
            "embedding_dim", "effective_rank", "mean_feature_std", "centroid_norm",
        }


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_b0_event_unigram_uses_every_valid_train_token() -> None:
    import torch

    examples, _ = trainer._synthetic_examples(41)
    selected = tuple(
        preprocess_neutral_direct_source_events(example) for example in examples[:6]
    )
    diagnostics: dict[str, object] = {}
    trainer._neutral_role_metrics(
        MarketRepresentationModel(), selected, _neutral_targets(len(selected)),
        tuple(range(len(selected))), batch_size=3, seed=17,
        device=torch.device("cpu"), diagnostic_sink=diagnostics,
    )
    tokens = diagnostics["all_event_targets"]
    assert len(tokens) == sum(len(example.event_type_ids) for example in selected)
    assert not np.any(tokens == trainer.PAD_TOKEN_ID)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_b1_dropout_rng_is_isolated_and_view_specific() -> None:
    import torch

    class DropoutModel(torch.nn.Module):
        def forward(self, batch: object) -> object:
            del batch
            return torch.nn.functional.dropout(
                torch.ones((16, 128)), p=0.5, training=True
            )

    model = DropoutModel()
    torch.manual_seed(777)
    before = torch.random.get_rng_state().clone()
    first = trainer._neutral_b1_seeded_forward(model, None, seed=11)
    after = torch.random.get_rng_state().clone()
    second = trainer._neutral_b1_seeded_forward(model, None, seed=12)
    repeated = trainer._neutral_b1_seeded_forward(model, None, seed=11)
    assert torch.equal(before, after)
    assert not torch.equal(first, second)
    assert torch.equal(first, repeated)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_b1_trains_two_views_once_and_keeps_short_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    source, _ = trainer._synthetic_examples(43)
    prepared = tuple(
        preprocess_neutral_direct_source_events(example) for example in source[:3]
    )
    examples = tuple([prepared[0]] * 16 + [prepared[1], prepared[2]])
    targets = _neutral_targets(len(examples))
    splits = {
        prepared[0].case.revision_id: "train",
        prepared[1].case.revision_id: "train",
        prepared[2].case.revision_id: "validation",
    }
    forwards: list[int] = []

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(1.0, 2.0, 128))
            self.config = SimpleNamespace(embedding_dim=128)

        def parameter_count(self) -> int:
            return self.weight.numel()

        def forward(self, batch: object) -> object:
            forwards.append(len(batch.case_ids))
            unit = self.weight / torch.linalg.vector_norm(self.weight)
            return SimpleNamespace(embedding=unit.expand(len(batch.case_ids), -1))

    def fake_loss(output: object, batch: object, target: object) -> object:
        del batch, target
        total = output.embedding[:, 0].mean() + 1.0
        components = {name: total * 0.0 for name in (
            "candle_reconstruction", "event_reconstruction", "next_event",
            "next_lifecycle", "next_event_time", "displacement",
            "draw_consumed", "scale_alignment", "contrastive",
        )}
        return SimpleNamespace(total=total, components=components)

    monkeypatch.setattr(trainer, "MarketRepresentationModel", TinyModel)
    monkeypatch.setattr(trainer, "neutral_representation_multitask_loss", fake_loss)
    monkeypatch.setattr(
        trainer, "_neutral_role_metrics",
        lambda model, examples, targets, indices, **kwargs: {"rows": len(indices)},
    )
    _, metrics = trainer._train_neutral_member(
        examples, targets, splits,
        SimpleNamespace(epochs=10, batch_size=16, learning_rate=3e-4,
                        device="cpu", seed=17),
        validation_only=True, b1_validation=True,
    )
    assert forwards == [16, 16, 1, 1] * 10
    for epoch in metrics["b1_regularizer_epoch_metrics"]:
        assert (
            epoch["optimizer_updates"], epoch["task_rows"],
            epoch["vicreg_batches"], epoch["vicreg_rows"],
            epoch["task_only_batches"], epoch["task_only_rows"],
        ) == (2, 17, 1, 16, 1, 1)
        assert epoch["loss_sums"]["combined"] == pytest.approx(
            epoch["loss_sums"]["task_mean"]
            + epoch["loss_sums"]["weighted_invariance"]
            + epoch["loss_sums"]["weighted_variance"]
            + epoch["loss_sums"]["weighted_covariance"]
        )


@pytest.mark.parametrize(("pass_epoch", "selected", "capable"), (
    (4, 4, True), (None, None, False),
))
def test_neutral_b0_selects_earliest_common_pass_without_exports(
    monkeypatch: pytest.MonkeyPatch,
    pass_epoch: int | None,
    selected: int | None,
    capable: bool,
) -> None:
    prepared = {
        "examples": (object(), object()), "targets": (object(), object()),
        "splits": {"train-revision": "train", "validation-revision": "validation"},
        "origins": {},
    }
    seeds: list[int] = []
    def fake_train(*args: object, diagnostic_epochs: list[Mapping[str, object]],
                   **kwargs: object) -> tuple[object, Mapping[str, object]]:
        member_args = args[3]
        seeds.append(member_args.seed)
        diagnostic_epochs.extend({} for _ in range(10))
        role = {
            "rows": 1, "total_loss": 1.0, "objective_losses": {},
            "active_head_metrics": {}, "masked_reconstruction": {},
            "geometry": {}, "material_geometry": {},
        }
        return object(), {
            "parameter_count": 1, "embedding_dim": 128,
            "epoch_metrics": [{"epoch": epoch, "train": role, "validation": role}
                              for epoch in range(1, 11)],
        }
    monkeypatch.setattr(trainer, "_prepare_neutral_fit_collection", lambda value: prepared)
    monkeypatch.setattr(trainer, "_train_neutral_member", fake_train)
    monkeypatch.setattr(
        trainer, "_neutral_b0_epoch_gate",
        lambda members, diagnostics, epoch: {
            "epoch": epoch, "common_pass": pass_epoch is not None and epoch >= pass_epoch,
        },
    )
    monkeypatch.setattr(
        trainer, "_neutral_b0_registry_contract",
        lambda: (trainer.NEUTRAL_B0_VALIDATION_CONTRACT, "d" * 64, "e" * 64),
    )
    monkeypatch.setattr(
        trainer, "_neutral_fit_lineage",
        lambda value: _valid_neutral_lineage(("train", "validation")),
    )
    report = trainer._neutral_b0_validation_report({
        "scope": "2021_02_train_2022_05_validation_only",
        "b0_contract_sha256": "d" * 64,
    }, SimpleNamespace(), trainer_repository_commit="f" * 40)
    assert seeds == [17, 100_020, 200_023]
    assert report["selected_epoch"] == selected
    assert report["criteria_met"] is capable
    assert report["model_capability_validated"] is False
    assert report["holdout_opened"] is report["artifacts_exported"] is False


def test_neutral_b1_report_reuses_b0_gates_and_records_regularizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = {
        "examples": (object(), object()), "targets": (object(), object()),
        "splits": {"train-revision": "train", "validation-revision": "validation"},
        "origins": {},
    }
    calls: list[tuple[int, bool]] = []

    def fake_train(*args: object, diagnostic_epochs: list[Mapping[str, object]],
                   b1_validation: bool, **kwargs: object) -> tuple[object, Mapping[str, object]]:
        seed = args[3].seed
        calls.append((seed, b1_validation))
        diagnostic_epochs.extend({} for _ in range(10))
        role = {
            "rows": 1, "total_loss": 1.0, "objective_losses": {},
            "active_head_metrics": {}, "masked_reconstruction": {},
            "geometry": {}, "material_geometry": {},
        }
        regularizer = [{"epoch": epoch, "proof": seed}
                       for epoch in range(1, 11)]
        return object(), {
            "parameter_count": 1, "embedding_dim": 128,
            "epoch_metrics": [{"epoch": epoch, "train": role, "validation": role}
                              for epoch in range(1, 11)],
            "b1_regularizer_epoch_metrics": regularizer,
        }

    monkeypatch.setattr(trainer, "_prepare_neutral_fit_collection", lambda value: prepared)
    monkeypatch.setattr(trainer, "_train_neutral_member", fake_train)
    monkeypatch.setattr(
        trainer, "_neutral_b0_gate_report",
        lambda *args, **kwargs: {
            "selected_epoch": None, "criteria_met": False,
        },
    )
    monkeypatch.setattr(
        trainer, "_neutral_b1_registry_contract",
        lambda: (trainer.NEUTRAL_B1_VALIDATION_CONTRACT, "d" * 64, "e" * 64),
    )
    report = trainer._neutral_b1_validation_report(
        {"scope": "2021_02_train_2022_05_validation_only",
         "b1_contract_sha256": "d" * 64},
        SimpleNamespace(), trainer_repository_commit="f" * 40,
        parent_b0_metrics=_valid_failed_neutral_b0_metrics_payload(),
    )
    assert calls == [(17, True), (100_020, True), (200_023, True)]
    assert report["criteria_met"] is False
    assert report["b0_gates_reused_without_change"] is True
    assert report["regularizer_diagnostics_used_for_selection"] is False
    assert report["true_per_scale_alignment_used"] is False
    assert report["regularizer_members"][2]["epoch_metrics"][-1] == {
        "epoch": 10, "proof": 200_023,
    }


def test_neutral_b1_failed_criteria_is_atomically_read_back_and_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "b0.json"
    metrics = tmp_path / "b1.json"
    argv = _b1_cli_args(metrics, parent)
    monkeypatch.setattr(trainer, "_neutral_b1_trainer_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        trainer, "_neutral_b1_parent_b0_metrics", lambda path: {"proof": "parent"}
    )
    monkeypatch.setattr(trainer, "_load_neutral_fit_collection", lambda **kwargs: {})
    monkeypatch.setattr(
        trainer, "_neutral_b1_validation_report",
        lambda *args, **kwargs: {"proof": "b1", "criteria_met": False},
    )
    monkeypatch.setattr(trainer, "_validate_neutral_b1_metrics_protocol", lambda value: None)
    monkeypatch.setattr(
        trainer, "load_neutral_b1_validation_metrics",
        lambda path: json.loads(Path(path).read_text()),
    )
    assert trainer.main(argv) == 2
    assert json.loads(metrics.read_text()) == {"criteria_met": False, "proof": "b1"}


def test_neutral_b0_neighbor_purity_is_train_only_strict_prior_cross_day() -> None:
    reference = np.vstack((np.tile([1.0, 0.0], (10, 1)),
                           np.tile([-1.0, 0.0], (10, 1))))
    query = np.array(((1.0, 0.0), (-1.0, 0.0)))
    train_labels = np.array([0] * 10 + [1] * 10)
    report = trainer._b0_neighbor_purity(
        {
            "embeddings": reference,
            "observed_at": tuple(["2021-02-01T10:00:00-05:00"] * 20),
            "material_kinds": tuple([("trigger",)] * 20),
            "head_targets": {
                "next_lifecycle": train_labels,
                "scale_direction_alignment": train_labels,
            },
        },
        {
            "embeddings": query,
            "observed_at": (
                "2022-05-01T10:00:00-04:00", "2022-05-02T10:00:00-04:00",
            ),
            "material_kinds": (("trigger",), ("trigger",)),
            "head_targets": {
                "next_lifecycle": np.array([0, 1]),
                "scale_direction_alignment": np.array([0, 1]),
            },
        },
        10,
    )
    assert report == {
        "eligible_material_queries": 2, "covered_material_queries": 2,
        "neighbors": 20, "k": 10, "purity": 1.0,
        "train_chance": 0.5, "lift": 0.5,
    }


def test_neutral_b0_neighbor_requires_all_queries_to_have_ten_candidates() -> None:
    labels = np.zeros(9, dtype=np.int64)
    report = trainer._b0_neighbor_purity(
        {
            "embeddings": np.tile([1.0, 0.0], (9, 1)),
            "observed_at": tuple(["2021-02-01T10:00:00-05:00"] * 9),
            "material_kinds": tuple([("trigger",)] * 9),
            "head_targets": {
                "next_lifecycle": labels,
                "scale_direction_alignment": labels,
            },
        },
        {
            "embeddings": np.array(((1.0, 0.0),)),
            "observed_at": ("2022-05-01T10:00:00-04:00",),
            "material_kinds": (("trigger",),),
            "head_targets": {
                "next_lifecycle": np.array([0]),
                "scale_direction_alignment": np.array([0]),
            },
        },
        10,
    )
    assert report == {
        "eligible_material_queries": 1, "covered_material_queries": 0,
        "neighbors": 0, "k": 10, "purity": 0.0,
        "train_chance": 1.0, "lift": -1.0,
    }


_DELETE_METRIC_KEY = object()


@pytest.mark.parametrize(("path", "replacement"), (
    (("unexpected",), False),
    (("outcome",), {}),
    (("holdout_used_for_optimization",), True),
    (("holdout_used_for_selection",), True),
    (("threshold_search_performed",), True),
    (("outcome_fields_used",), True),
    (("action_value_claimed",), True),
    (("ensemble_members", 0, "unexpected"), False),
    (("ensemble_members", 0, "metrics", "unexpected"), False),
    (("ensemble_members", 0, "metrics", "optimized_roles"), ["train", "holdout"]),
    (("ensemble_members", 0, "metrics", "validation_used_for_optimization"), True),
    (("ensemble_members", 0, "metrics", "holdout_used_for_optimization"), True),
    (("ensemble_members", 0, "metrics", "holdout_used_for_selection"), True),
    (("ensemble_members", 0, "metrics", "threshold_search_performed"), True),
    (("ensemble_members", 0, "metrics", "outcome_fields_used"), True),
    (("ensemble_members", 0, "metrics", "action_value_claimed"), True),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "holdout"), {}),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "optimized"), True),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "holdout"), {}),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "optimized"), True),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "gradient_enabled"), True),
    (("ensemble_members", 0, "metrics", "holdout", "gradient_enabled"), True),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "geometry", "unexpected"), 0.0),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "geometry", "effective_rank"), None),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "geometry", "effective_rank"), float("inf")),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "geometry", "effective_rank"), 129.0),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "objective_losses", "unexpected"), 0.0),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "objective_losses", "next_lifecycle"), "0.3"),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "objective_losses", "next_lifecycle"), float("nan")),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "objective_losses", "next_lifecycle"), _DELETE_METRIC_KEY),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "active_head_metrics", "unexpected"), {}),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "active_head_metrics", "next_lifecycle", "unexpected"), 0),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "active_head_metrics", "next_lifecycle", "nll"), "0.5"),
    (("ensemble_members", 0, "metrics", "epoch_metrics", 0, "train", "active_head_metrics", "next_lifecycle", "accuracy"), 1.1),
    (("direct_source_preprocessing",), _DELETE_METRIC_KEY),
), ids=(
    "top-extra", "top-outcome", "top-holdout-optimization",
    "top-holdout-selection", "top-threshold", "top-outcome-flag",
    "top-action", "member-extra", "member-metrics-extra",
    "member-optimized-roles", "member-validation-optimization",
    "member-holdout-optimization", "member-holdout-selection",
    "member-threshold", "member-outcome", "member-action-extra",
    "epoch-holdout", "epoch-optimized", "epoch-train-holdout",
    "epoch-train-optimized", "epoch-train-gradient", "holdout-gradient",
    "geometry-extra", "geometry-rank-null", "geometry-rank-infinite",
    "geometry-rank-range", "objective-extra", "objective-nonnumeric",
    "objective-nonfinite", "objective-missing", "heads-extra", "head-extra",
    "head-nonnumeric", "head-range", "top-protocol-missing",
))
def test_neutral_fit_metrics_loader_rejects_nested_tamper(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    payload: object = _valid_neutral_fit_metrics_payload()
    parent = payload
    for key in path[:-1]:
        parent = parent[key]  # type: ignore[index]
    if replacement is _DELETE_METRIC_KEY:
        parent.pop(path[-1])  # type: ignore[union-attr]
    else:
        parent[path[-1]] = replacement  # type: ignore[index]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepresentationDataError):
        trainer.load_neutral_fit_metrics(metrics_path)


def test_neutral_fit_metrics_loader_accepts_exact_v2_and_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    payload = _valid_neutral_fit_metrics_payload()
    encoded = json.dumps(payload, sort_keys=True)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(encoded, encoding="utf-8")
    assert trainer.load_neutral_fit_metrics(metrics_path) == payload
    metrics_path.write_text(
        '{"schema_version":2,' + encoded[1:], encoding="utf-8"
    )
    with pytest.raises(RepresentationDataError, match="duplicate"):
        trainer.load_neutral_fit_metrics(metrics_path)


def test_neutral_b0_metrics_loader_accepts_exact_report_and_credible_failure(
    tmp_path: Path,
) -> None:
    payload = _valid_neutral_b0_metrics_payload()
    metrics_path = tmp_path / "b0.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    assert trainer.load_neutral_b0_validation_metrics(metrics_path) == payload

    for gate in payload["epoch_gates"]:
        neighbor = gate["members"][0]["neighbor"]
        neighbor.update({
            "covered_material_queries": 0, "neighbors": 0,
            "purity": 0.0, "lift": -0.5,
        })
        gate["common_pass"] = False
    payload["selected_epoch"] = None
    payload["criteria_met"] = False
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    assert trainer.load_neutral_b0_validation_metrics(metrics_path) == payload


@pytest.mark.parametrize(("path", "replacement"), (
    (("unexpected",), False),
    (("schema_version",), True),
    (("selected_epoch",), True),
    (("ensemble_members", 0, "seed"), 17.0),
    (("ensemble_members", 0, "epoch_metrics", 0, "epoch"), True),
    (("epoch_gates", 0, "epoch"), True),
    (("epoch_gates", 0, "common_pass"), False),
    (("epoch_gates", 0, "members", 0, "neighbor",
      "covered_material_queries"), 2),
    (("epoch_gates", 0, "members", 0, "neighbor", "k"), 10.0),
    (("epoch_gates", 0, "active_heads", "next_lifecycle",
      "member_balanced_accuracy"), [0.7, 0.8, 0.8]),
    (("ensemble_members", 0, "epoch_metrics", 0, "validation",
      "total_loss"), None),
    (("ensemble_members", 0, "epoch_metrics", 0, "validation",
      "geometry", "effective_rank"), float("nan")),
    (("lineage", "split_protocol", "registry_sha256"), "d" * 64),
))
def test_neutral_b0_metrics_loader_rejects_tamper(
    tmp_path: Path, path: tuple[object, ...], replacement: object,
) -> None:
    payload: object = _valid_neutral_b0_metrics_payload()
    parent = payload
    for key in path[:-1]:
        parent = parent[key]  # type: ignore[index]
    parent[path[-1]] = replacement  # type: ignore[index]
    metrics_path = tmp_path / "b0-tampered.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepresentationDataError):
        trainer.load_neutral_b0_validation_metrics(metrics_path)


@pytest.mark.parametrize(("path", "replacement"), (
    (("max_epochs",), 10.0),
    (("gates", "neighbor_coverage_required"), True),
))
def test_neutral_b0_metrics_loader_rejects_self_hashed_contract_type_tamper(
    tmp_path: Path, path: tuple[object, ...], replacement: object,
) -> None:
    payload = _valid_neutral_b0_metrics_payload()
    contract = payload["contract"]["value"]
    parent = contract
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = replacement
    payload["contract"]["sha256"] = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    metrics_path = tmp_path / "b0-contract-type.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepresentationDataError, match="contract hash"):
        trainer.load_neutral_b0_validation_metrics(metrics_path)


def test_neutral_b0_metrics_loader_rejects_run_order_tamper(tmp_path: Path) -> None:
    payload = _valid_neutral_b0_metrics_payload()
    payload["lineage"]["input_runs"].reverse()
    metrics_path = tmp_path / "b0-run-order.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepresentationDataError, match="lineage binding"):
        trainer.load_neutral_b0_validation_metrics(metrics_path)


def test_neutral_b1_metrics_loader_accepts_exact_credible_failure(
    tmp_path: Path,
) -> None:
    payload = _valid_neutral_b1_metrics_payload()
    path = tmp_path / "b1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = trainer.load_neutral_b1_validation_metrics(path)
    assert loaded["criteria_met"] is False
    assert loaded["selected_epoch"] is None
    assert loaded["true_per_scale_alignment_used"] is False


@pytest.mark.parametrize(("path", "replacement"), (
    (("parent_b0", "metrics_sha256"), "f" * 64),
    (("regularizer_members", 0, "epoch_metrics", 0,
      "loss_sums", "weighted_variance"), 9.0),
    (("regularizer_members", 0, "epoch_metrics", 0, "vicreg_rows"), 1168),
    (("true_per_scale_alignment_used",), True),
    (("b0_gate_evaluation", "lineage", "input_runs", 0,
      "input_manifest_sha256"), "f" * 64),
    (("selected_epoch",), True),
))
def test_neutral_b1_metrics_loader_rejects_tamper(
    tmp_path: Path, path: tuple[object, ...], replacement: object,
) -> None:
    payload = _valid_neutral_b1_metrics_payload()
    parent: object = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = replacement
    source = tmp_path / "b1.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepresentationDataError):
        trainer.load_neutral_b1_validation_metrics(source)


@pytest.mark.parametrize("extra", ("action_authority", "holdout_selected"))
def test_neutral_lineage_is_exact_and_rejects_authority_fields(extra: str) -> None:
    lineage = _valid_neutral_lineage(("train", "validation"))
    trainer._validate_neutral_lineage(lineage, roles={"train", "validation"})
    lineage[extra] = False
    with pytest.raises(RepresentationDataError, match="lineage"):
        trainer._validate_neutral_lineage(lineage, roles={"train", "validation"})


def test_neutral_fit_metrics_rejects_more_than_ten_epochs(tmp_path: Path) -> None:
    payload = _valid_neutral_fit_metrics_payload()
    for member in payload["ensemble_members"]:
        metrics = member["metrics"]
        metrics["epochs"] = 11
        metrics["epoch_training_loss"] *= 11
        metrics["epoch_metrics"] *= 11
    path = tmp_path / "too-many-epochs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepresentationDataError, match="maximum"):
        trainer.load_neutral_fit_metrics(path)


def test_neutral_direct_source_preprocessing_is_deterministic_and_versioned() -> None:
    examples, _ = trainer._synthetic_examples(19)
    original = examples[0]
    marked = replace(
        original,
        direct_label_source_event_mask=np.ones(
            len(original.event_type_ids), dtype=bool
        ),
    )
    first = preprocess_neutral_direct_source_events(marked)
    second = preprocess_neutral_direct_source_events(first)
    identity = neutral_direct_source_preprocessing_identity()
    assert first.label_sources_masked is True
    assert first.neutral_preprocessing_version == (
        identity["protocol"]["protocol_version"]
    )
    assert first.neutral_preprocessing_sha256 == identity["sha256"]
    assert len(first.event_type_ids) == 1
    for name in (
        "event_type_ids", "lifecycle_ids", "relation_ids", "scale_ids",
        "event_numeric", "direct_label_source_event_mask",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="optional PyTorch is not installed"
)
@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_neutral_checkpoint_preprocessing_protocol_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    import torch

    path = tmp_path / "neutral.pt"
    model = MarketRepresentationModel()
    save_neutral_representation_checkpoint(path, model, metadata={
        "member_id": "member-000", "seed": 17,
        "split_counts": {"train": 1, "validation": 1},
        "lineage": _valid_neutral_lineage(("train", "validation")),
        "outcome_fields_used": False, "model_capability_validated": False,
    })
    assert load_neutral_representation_checkpoint(path).config == model.config
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if mutation == "missing":
        payload["metadata"].pop("direct_source_preprocessing")
    else:
        payload["metadata"]["direct_source_preprocessing"]["protocol"][
            "direct_source_action"
        ] = "retain"
    torch.save(payload, path)
    with pytest.raises(RepresentationDataError, match="missing or differs"):
        load_neutral_representation_checkpoint(path)


def test_neutral_fit_exports_three_independent_members_without_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse
    import torch

    examples, _ = trainer._synthetic_examples(31)
    selected = tuple(
        preprocess_neutral_direct_source_events(example)
        for example in examples[:3]
    )
    targets = _neutral_targets(3)
    splits = {
        selected[0].case.revision_id: "train",
        selected[1].case.revision_id: "validation",
        selected[2].case.revision_id: "holdout",
    }
    prepared = {
        "examples": selected,
        "targets": targets,
        "splits": splits,
        "origins": {},
        "source_summaries": (),
    }

    def fake_train(*args: object) -> tuple[MarketRepresentationModel, Mapping[str, object]]:
        member_args = args[-1]
        torch.manual_seed(member_args.seed)
        return MarketRepresentationModel(), {
            "split_counts": {"train": 1, "validation": 1, "holdout": 1}
        }

    def fake_export(
        model: MarketRepresentationModel,
        *args: object,
        member_id: str,
        embeddings: bool,
        **kwargs: object,
    ) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
        checkpoint_id = trainer.representation_checkpoint_id(model)
        common = {
            "run_manifest_sha256": "a" * 64,
            "revision_id": "revision",
            "market_episode_id": "episode",
            "data_split": "train",
            "decision_at": "2025-02-03T10:00:00-05:00",
            "feature_max_at": "2025-02-03T10:00:00-05:00",
            "outcome_fields_used": False,
        }
        embedding_rows = (
            [{**common, "embedding_checkpoint_id": checkpoint_id}]
            if embeddings
            else []
        )
        return embedding_rows, [
            {
                **common,
                "checkpoint_id": checkpoint_id,
                "member_id": member_id,
                "head_predictions": {
                    "next_lifecycle": [1.0],
                    "scale_direction_alignment": [1.0],
                },
            }
        ]

    monkeypatch.setattr(
        trainer, "_prepare_neutral_fit_collection", lambda collection: prepared
    )
    monkeypatch.setattr(trainer, "_neutral_fit_lineage", lambda collection: {})
    monkeypatch.setattr(trainer, "_train_neutral_member", fake_train)
    monkeypatch.setattr(trainer, "_neutral_export_records", fake_export)
    monkeypatch.setattr(
        trainer, "save_neutral_representation_checkpoint", lambda *a, **k: None
    )
    args = argparse.Namespace(
        ensemble_size=3,
        seed=37,
        checkpoint=str(tmp_path / "model.pt"),
        embedding_output=str(tmp_path / "embeddings.jsonl"),
        head_output=str(tmp_path / "heads.jsonl"),
        batch_size=2,
        device="cpu",
    )
    report = trainer._neutral_fit_report({"scope": "three_window_pipeline_smoke"}, args)
    members = report["ensemble_members"]
    assert len({member["seed"] for member in members}) == 3
    assert len({member["checkpoint_id"] for member in members}) == 3
    head_rows = [json.loads(line) for line in (tmp_path / "heads.jsonl").read_text().splitlines()]
    assert len(head_rows) == 3
    assert all("outcome" not in row and row["outcome_fields_used"] is False for row in head_rows)
    head_manifest = json.loads(
        (tmp_path / "heads.jsonl.manifest.json").read_text()
    )
    assert set(head_manifest["head_schema"]) == {
        "next_lifecycle", "scale_direction_alignment"
    }
    assert "case_library_manifest_sha256" not in head_manifest
    assert head_manifest["model_capability_validated"] is False
