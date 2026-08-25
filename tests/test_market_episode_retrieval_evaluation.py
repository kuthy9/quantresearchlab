from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import evaluate_market_episode_retrieval as evaluator
from scripts import train_market_representation as trainer
from smc_trader.artifact_stream import atomic_bytes, canonical_json, sha256_file
from smc_trader.case_retrieval import (
    MARKET_EPISODE_RETRIEVAL_PROTOCOL,
    CaseRetrievalError,
)
from smc_trader.market_cases import expected_market_case_run_identity
from smc_trader.market_representation import (
    MarketRepresentationModel,
    NEUTRAL_INFERENCE_INPUT_PROTOCOL,
    TORCH_AVAILABLE,
    neutral_direct_source_preprocessing_identity,
    prepare_neutral_representation_case,
    representation_case_from_market_case_input_row,
    representation_checkpoint_id,
)
from tests.test_market_representation import (
    _neutral_revision_row,
    _neutral_run_manifest,
    _neutral_store,
)


def _input_lineage(tmp_path: Path) -> tuple[dict[str, str], dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_sha, config_sha = "a" * 64, "b" * 64
    protocol = expected_market_case_run_identity()
    runs: dict[str, str] = {}
    datasets = []
    for index, role in enumerate(("train", "validation", "holdout"), start=1):
        commit = str(index) * 40
        profile = f"neutral-{role}"
        run = tmp_path / f"{role}-run.json"
        atomic_bytes(run, canonical_json({
            "mode": "market_case_input", "profile": {"name": profile},
            "source": {"sha256": source_sha},
            "model_config": {"sha256": config_sha, "timezone": "America/New_York"},
            "repository": {"commit": commit},
            "market_case_input_identity": protocol,
        }))
        stream = tmp_path / f"{role}-stream.json"
        atomic_bytes(stream, canonical_json({"bindings": {"run_manifest": run.name}}))
        run_sha = sha256_file(run)
        runs[role] = run_sha
        datasets.append({
            "profile_name": profile, "representation_split_role": role,
            "input_path": stream.resolve(), "input_sha": sha256_file(stream),
            "run_path": run.resolve(), "run_sha": run_sha,
            "run_manifest": {"market_case_input_identity": protocol},
            "run_identity": {"repository": {"commit": commit}},
        })
    registry = SimpleNamespace(
        protocol_version="neutral-representation-splits-1.0.0",
        warmup_calendar_days=14, purge_calendar_days=14, embargo_trading_days=5,
        episode_split_key=("run_manifest_sha256", "market_epoch_id", "market_episode_id"),
    )
    collection = {
        "registry": registry, "datasets": datasets, "source_sha256": source_sha,
        "model_config_sha256": config_sha, "timezone": "America/New_York",
        "registry_sha256": "d" * 64,
        "observed_embargo_sessions": tuple({
            "left_profile": f"neutral-{left}",
            "right_profile": f"neutral-{right}",
            "purge_end": "2024-01-20T00:00:00-05:00",
            "next_prefix_start": "2024-02-01T00:00:00-05:00",
            "observed_session_count": 5,
            "first_observed_session": "2024-01-22",
            "last_observed_session": "2024-01-26",
        } for left, right in (("train", "validation"), ("validation", "holdout"))),
    }
    return runs, dict(trainer._neutral_fit_lineage(collection))


def _fit_artifacts(
    tmp_path: Path,
    *,
    lineage_root: Path | None = None,
) -> tuple[Path, Path]:
    import torch

    tmp_path.mkdir(parents=True, exist_ok=True)
    runs, lineage = _input_lineage(lineage_root or tmp_path)
    roles_and_dates = [
        *(('train', f"2024-01-{day:02d} 10:00") for day in range(2, 7)),
        ("validation", "2024-02-01 10:00"),
        ("holdout", "2024-03-01 10:00"),
    ]
    examples, splits, origins = [], {}, {}
    adapter_manifest = _neutral_run_manifest(tmp_path)
    adapter_manifest["window"]["start"] = "2024-01-01T00:00:00-05:00"
    adapter_manifest["window"]["end_exclusive"] = "2024-04-01T00:00:00-04:00"
    adapter_manifest["source"]["first"] = "2024-01-01T00:00:00-05:00"
    adapter_manifest["source"]["last"] = "2024-03-31T23:59:00-04:00"
    adapter_manifest["source"]["last_completed_asof"] = "2024-04-01T00:00:00-04:00"
    for index, (role, raw_date) in enumerate(roles_and_dates):
        case = representation_case_from_market_case_input_row(
            _neutral_revision_row(
                asof=pd.Timestamp(raw_date, tz="America/New_York"),
                revision_index=0, source_replay_ordinal=10 + index,
                replay_update_ordinal=10 + index, lifecycle="registered",
                transition_kinds=("zone_registered",), epoch_id="epoch:shared-source",
                location_id=f"location:{role}:{index}", path_id=f"path:{role}:{index}",
            ),
            adapter_manifest,
        )
        prepared = prepare_neutral_representation_case(
            case, _neutral_store(case)
        )
        examples.append(prepared)
        splits[case.revision_id], origins[case.revision_id] = role, runs[role]
    embedding_rows, head_rows, checkpoints = [], [], []
    for member_index in range(4):
        torch.manual_seed(100 + member_index)
        model = MarketRepresentationModel()
        checkpoint = representation_checkpoint_id(model)
        checkpoints.append(checkpoint)
        embeddings, heads = trainer._neutral_export_records(
            model, tuple(examples), splits, origins,
            member_id=f"member-{member_index:03d}", batch_size=8,
            device="cpu", embeddings=member_index == 0,
        )
        embedding_rows.extend(embeddings)
        head_rows.extend(heads)
    head_rows = [
        row for row in head_rows
        if not (row["member_id"] in {"member-002", "member-003"}
                and row["data_split"] == "holdout")
    ]
    embedding_manifest = trainer._write_neutral_artifact(
        tmp_path / "embeddings.jsonl", embedding_rows,
        schema=trainer.NEUTRAL_EMBEDDING_ARTIFACT_SCHEMA,
        checkpoint_ids=(checkpoints[0],), lineage=lineage,
    )
    head_manifest = trainer._write_neutral_artifact(
        tmp_path / "heads.jsonl", head_rows,
        schema=trainer.NEUTRAL_HEAD_ARTIFACT_SCHEMA,
        checkpoint_ids=tuple(checkpoints), lineage=lineage,
    )
    return embedding_manifest, head_manifest


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_real_fit_export_handoff_reports_only_geometry_and_ood(tmp_path: Path) -> None:
    embedding_manifest, head_manifest = _fit_artifacts(tmp_path)
    output = tmp_path / "evaluation.json"
    assert evaluator.main([
        "--embedding-manifest", str(embedding_manifest),
        "--head-manifest", str(head_manifest), "--output", str(output), "--k", "5",
    ]) == 0
    report = json.loads(output.read_text())
    assert report["labels_used"] is False
    assert report["outcomes_used"] is False
    assert report["prediction_quality_claimed"] is False
    assert report["action_authority"] == "none"
    assert report["schema_version"] == 3
    assert report["retrieval_protocol"] == json.loads(
        canonical_json(dict(MARKET_EPISODE_RETRIEVAL_PROTOCOL))
    )
    assert report["b0_compatible"] is True
    assert report["direct_source_preprocessing"] == (
        neutral_direct_source_preprocessing_identity()
    )
    metrics = report["metrics"]
    retrieval = metrics["cross_day_independent_episode_retrieval"]
    assert retrieval["coverage_with_eligible_neighbour"] == 1.0
    assert retrieval["different_calendar_date_required"] is True
    consistency = metrics["self_supervised_consistency"]
    assert consistency["expected_ensemble_members"] == 4
    assert consistency["complete_ensemble_queries"] == 1
    assert consistency["incomplete_ensemble_queries"] == 1
    assert metrics["ood"]["mean_ensemble_disagreement"] is not None
    assert metrics["ood"]["missing_ensemble_queries"] == 1
    assert metrics["ood"]["missing_ensemble_abstained"] == 1
    assert "effective_rank" in metrics["collapse"]["by_split_material_kind"][
        "train/zone_registered"
    ]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_combined_loader_requires_explicit_record_run_and_verified_input(tmp_path: Path) -> None:
    embedding_manifest, _ = _fit_artifacts(tmp_path)
    payload = json.loads(embedding_manifest.read_text())
    artifact = embedding_manifest.parent / payload["artifact_path"]
    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    rows[0].pop("run_manifest_sha256")
    artifact.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    payload["artifact_sha256"] = sha256_file(artifact)
    atomic_bytes(embedding_manifest, canonical_json(payload))
    with pytest.raises(CaseRetrievalError, match="record run manifest SHA"):
        evaluator.load_artifact_manifest(embedding_manifest)

    embedding_manifest, _ = _fit_artifacts(tmp_path / "other")
    payload = json.loads(embedding_manifest.read_text())
    stream = Path(payload["lineage"]["input_runs"][0]["input_manifest_path"])
    stream.write_bytes(stream.read_bytes() + b"\n")
    with pytest.raises(CaseRetrievalError, match="content lineage differs"):
        evaluator.load_artifact_manifest(embedding_manifest)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_combined_loader_accepts_verified_sibling_external_lineage_only(
    tmp_path: Path,
) -> None:
    embedding_manifest, _ = _fit_artifacts(
        tmp_path / "artifacts",
        lineage_root=tmp_path / "inputs",
    )
    original_manifest = embedding_manifest.read_bytes()
    payload = json.loads(original_manifest)
    raw = payload["lineage"]["input_runs"][0]
    stream = Path(raw["input_manifest_path"])
    original_stream = stream.read_bytes()

    assert stream.parent == tmp_path / "inputs"
    assert evaluator.load_artifact_manifest(embedding_manifest)["rows"]

    relative = json.loads(original_manifest)
    relative["lineage"]["input_runs"][0]["input_manifest_path"] = stream.name
    atomic_bytes(embedding_manifest, canonical_json(relative))
    with pytest.raises(CaseRetrievalError, match="must be absolute"):
        evaluator.load_artifact_manifest(embedding_manifest)
    atomic_bytes(embedding_manifest, original_manifest)

    link = stream.with_name("linked-stream.json")
    link.symlink_to(stream)
    symlinked = json.loads(original_manifest)
    symlinked["lineage"]["input_runs"][0]["input_manifest_path"] = str(link)
    atomic_bytes(embedding_manifest, canonical_json(symlinked))
    with pytest.raises(CaseRetrievalError, match="not a regular file"):
        evaluator.load_artifact_manifest(embedding_manifest)
    link.unlink()
    atomic_bytes(embedding_manifest, original_manifest)

    directory_link = tmp_path / "linked-inputs"
    directory_link.symlink_to(stream.parent, target_is_directory=True)
    aliased = json.loads(original_manifest)
    aliased["lineage"]["input_runs"][0]["input_manifest_path"] = str(
        directory_link / stream.name
    )
    atomic_bytes(embedding_manifest, canonical_json(aliased))
    with pytest.raises(CaseRetrievalError, match="symlink or alias"):
        evaluator.load_artifact_manifest(embedding_manifest)
    directory_link.unlink()
    atomic_bytes(embedding_manifest, original_manifest)

    stream.write_bytes(original_stream + b"\n")
    with pytest.raises(CaseRetrievalError, match="content lineage differs"):
        evaluator.load_artifact_manifest(embedding_manifest)
    stream.write_bytes(original_stream)

    escaped_artifact = json.loads(original_manifest)
    escaped_artifact["artifact_path"] = "../inputs/train-stream.json"
    atomic_bytes(embedding_manifest, canonical_json(escaped_artifact))
    with pytest.raises(CaseRetrievalError, match="must be relative"):
        evaluator.load_artifact_manifest(embedding_manifest)

    artifact = embedding_manifest.parent / payload["artifact_path"]
    artifact_link = artifact.with_name("linked-artifact.jsonl")
    artifact_link.symlink_to(artifact)
    symlinked_artifact = json.loads(original_manifest)
    symlinked_artifact["artifact_path"] = artifact_link.name
    atomic_bytes(embedding_manifest, canonical_json(symlinked_artifact))
    with pytest.raises(CaseRetrievalError, match="binding is missing"):
        evaluator.load_artifact_manifest(embedding_manifest)
    artifact_link.unlink()


def test_evaluator_output_dangling_symlink_is_not_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = tmp_path / "external" / "escaped.json"
    external.parent.mkdir()
    output = tmp_path / "evaluation.json"
    output.symlink_to(external)
    monkeypatch.setattr(evaluator, "load_artifact_manifest", lambda _path: {})
    monkeypatch.setattr(
        evaluator,
        "evaluate",
        lambda *_args, **_kwargs: {"status": "complete"},
    )

    with pytest.raises(SystemExit, match="refused input"):
        evaluator.main([
            "--embedding-manifest", str(tmp_path / "embedding.json"),
            "--head-manifest", str(tmp_path / "heads.json"),
            "--output", str(output),
        ])

    assert output.is_symlink()
    assert not external.exists()
    assert tuple(tmp_path.glob(".evaluation.json.*.tmp")) == ()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_real_writer_manifests_reject_synchronized_split_protocol_tamper(
    tmp_path: Path,
) -> None:
    manifests = _fit_artifacts(tmp_path)
    for manifest in manifests:
        payload = json.loads(manifest.read_text())
        payload["lineage"]["split_protocol"] = {}
        atomic_bytes(manifest, canonical_json(payload))

    for manifest in manifests:
        with pytest.raises(CaseRetrievalError, match="split protocol lineage"):
            evaluator.load_artifact_manifest(manifest)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_b0_manifest_preprocessing_protocol_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    embedding_manifest, _ = _fit_artifacts(tmp_path)
    payload = json.loads(embedding_manifest.read_text())
    if mutation == "missing":
        payload.pop("direct_source_preprocessing")
    else:
        payload["direct_source_preprocessing"]["protocol"][
            "direct_source_action"
        ] = "retain"
    atomic_bytes(embedding_manifest, canonical_json(payload))
    with pytest.raises(CaseRetrievalError, match="manifest contract"):
        evaluator.load_artifact_manifest(embedding_manifest)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_b0_manifest_uses_processed_input_protocol(tmp_path: Path) -> None:
    embedding_manifest, head_manifest = _fit_artifacts(tmp_path)
    for path in (embedding_manifest, head_manifest):
        payload = json.loads(path.read_text())
        assert payload["schema"].endswith("-v2")
        assert payload["input_protocol"] == NEUTRAL_INFERENCE_INPUT_PROTOCOL
        assert payload["direct_source_preprocessing"] == (
            neutral_direct_source_preprocessing_identity()
        )
    legacy = json.loads(embedding_manifest.read_text())
    legacy["schema"] = "smc-neutral-market-episode-embeddings-v1"
    atomic_bytes(embedding_manifest, canonical_json(legacy))
    with pytest.raises(CaseRetrievalError, match="manifest contract"):
        evaluator.load_artifact_manifest(embedding_manifest)
