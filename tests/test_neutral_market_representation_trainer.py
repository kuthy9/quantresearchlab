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
            "schema_version": 1,
            "tick_size": 0.25,
            "timezone": "America/New_York",
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
        "lineage": {},
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
    save_neutral_representation_checkpoint(path, model, metadata={"seed": 17})
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
