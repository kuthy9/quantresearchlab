#!/usr/bin/env python3
"""Outcome-free geometry/retrieval/OOD smoke for combined neutral artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import atomic_bytes, canonical_json  # noqa: E402
from smc_trader.case_retrieval import (  # noqa: E402
    MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS,
    MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT,
    MARKET_EPISODE_MATERIAL_KINDS,
    CaseRetrievalError,
    MarketEpisodeCaseIndex,
    MarketEpisodeEmbeddingQuery,
    OODThresholds,
    normalise_market_episode_dataset_contract,
)
from smc_trader.market_representation import (  # noqa: E402
    NEUTRAL_INFERENCE_INPUT_PROTOCOL,
    neutral_direct_source_preprocessing_identity,
)
EMBEDDING_SCHEMA = "smc-neutral-market-episode-embeddings-v2"
HEAD_SCHEMA = "smc-neutral-market-episode-active-heads-v2"
SPLITS = ("train", "validation", "holdout")
SELECTION = MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT
HEAD_IDENTITY_FIELDS = (
    "run_manifest_sha256", "market_epoch_id", "market_episode_id", "revision_id",
)
MANIFEST_FIELDS = {
    "schema", "status", "records", "artifact_path", "artifact_sha256",
    "model_version", "feature_schema_version", "input_protocol",
    "direct_source_preprocessing",
    "selection_contract", "checkpoint_ids", "split_roles", "material_kinds",
    "head_schema", "lineage", "outcome_fields_used", "model_capability_validated",
}
LINEAGE_FIELDS = {
    "input_runs", "source_identity", "model_config_identity",
    "market_case_protocol", "representation_feature_schema_version", "split_protocol",
}
SPLIT_PROTOCOL_FIELDS = {
    "registry_sha256", "protocol_version", "warmup_calendar_days",
    "purge_calendar_days", "embargo_trading_days", "market_episode_split_key",
    "actual_prefix_exposure_verified", "observed_completed_session_embargo",
}
OBSERVED_EMBARGO_FIELDS = {
    "left_profile", "right_profile", "purge_end", "next_prefix_start",
    "observed_session_count", "first_observed_session", "last_observed_session",
}
INPUT_RUN_FIELDS = {
    "profile_name", "split_role", "input_manifest_path",
    "input_manifest_sha256", "run_manifest_path", "run_manifest_sha256", "repository_commit",
}
EMBEDDING_FIELDS = {
    "revision_id", "revision_index", "revision_stage", "material_kind",
    "transition_kinds", "market_epoch_id", "market_episode_id",
    "entry_location_id", "entry_path_id", "decision_at", "direction",
    "data_split", "embedding_model_version", "embedding_checkpoint_id",
    "embedding_dim", "embedding_clock", "embedding_asof", "feature_max_at",
    "embedding_input_protocol", "outcome_fields_used", "decision_embedding",
    "run_manifest_sha256",
}
HEAD_FIELDS = {
    "member_id", "checkpoint_id", "model_version", "revision_id",
    "market_epoch_id", "market_episode_id", "decision_at", "feature_max_at", "outcome_fields_used",
    "input_protocol", "head_predictions", "run_manifest_sha256", "data_split",
}

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _digest(value: Any, name: str) -> str:
    token = str(value or "").strip().lower()
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        raise CaseRetrievalError(f"{name} is not a SHA-256")
    return token

def _object(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaseRetrievalError(f"{name} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise CaseRetrievalError(f"{name} must be an object")
    return value

def _bound(owner: Path, raw: Any, name: str) -> Path:
    path = Path(str(raw or ""))
    path = path if path.is_absolute() else owner.parent / path
    if path.is_symlink() or not path.is_file():
        raise CaseRetrievalError(f"{name} is absent or not a regular file")
    return path.resolve()

def _rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CaseRetrievalError("artifact is not UTF-8 JSONL") from exc
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaseRetrievalError(f"artifact line {number} is invalid") from exc
        if not line or not isinstance(row, Mapping):
            raise CaseRetrievalError(f"artifact line {number} is not an object")
        output.append(dict(row))
    return output

def _clock(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaseRetrievalError(f"{name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CaseRetrievalError(f"{name} is not timezone-aware")
    return parsed

def _lineage(owner: Path, value: Any, feature_schema: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != LINEAGE_FIELDS:
        raise CaseRetrievalError("combined artifact lineage is invalid")
    source, config = value["source_identity"], value["model_config_identity"]
    if not isinstance(source, Mapping) or set(source) != {"sha256"}:
        raise CaseRetrievalError("source identity is invalid")
    if not isinstance(config, Mapping) or set(config) != {"sha256", "timezone"}:
        raise CaseRetrievalError("model config identity is invalid")
    source_sha = _digest(source["sha256"], "source identity")
    config_sha = _digest(config["sha256"], "model config identity")
    if value["representation_feature_schema_version"] != feature_schema:
        raise CaseRetrievalError("feature schema lineage differs")
    protocol, split_protocol = value["market_case_protocol"], value["split_protocol"]
    if not isinstance(protocol, Mapping) or not protocol or not isinstance(split_protocol, Mapping):
        raise CaseRetrievalError("dataset protocol lineage is invalid")
    observed = split_protocol.get("observed_completed_session_embargo")
    if (
        set(split_protocol) != SPLIT_PROTOCOL_FIELDS
        or _digest(split_protocol.get("registry_sha256"), "split registry SHA")
        != split_protocol["registry_sha256"]
        or split_protocol.get("protocol_version")
        != "neutral-representation-splits-1.0.0"
        or split_protocol.get("warmup_calendar_days") != 14
        or split_protocol.get("purge_calendar_days") != 14
        or split_protocol.get("embargo_trading_days") != 5
        or split_protocol.get("market_episode_split_key") != [
            "run_manifest_sha256", "market_epoch_id", "market_episode_id",
        ]
        or split_protocol.get("actual_prefix_exposure_verified") is not True
        or not isinstance(observed, list)
        or not observed
        or any(
            not isinstance(item, Mapping)
            or set(item) != OBSERVED_EMBARGO_FIELDS
            or type(item.get("observed_session_count")) is not int
            or item["observed_session_count"] < 5
            or any(not isinstance(item.get(name), str) or not item[name] for name in (
                "left_profile", "right_profile", "purge_end", "next_prefix_start",
                "first_observed_session", "last_observed_session",
            ))
            for item in observed
        )
    ):
        raise CaseRetrievalError("neutral split protocol lineage is invalid")
    raw_runs = value["input_runs"]
    if isinstance(raw_runs, (str, bytes)) or not isinstance(raw_runs, Sequence):
        raise CaseRetrievalError("input run lineage is invalid")
    runs: dict[str, Any] = {}
    for raw in raw_runs:
        if not isinstance(raw, Mapping) or set(raw) != INPUT_RUN_FIELDS:
            raise CaseRetrievalError("input run lineage is invalid")
        role, profile = str(raw["split_role"]), str(raw["profile_name"])
        stream = _bound(owner, raw["input_manifest_path"], "input manifest")
        run = _bound(owner, raw["run_manifest_path"], "run manifest")
        stream_sha = _digest(raw["input_manifest_sha256"], "input manifest SHA")
        run_sha = _digest(raw["run_manifest_sha256"], "run manifest SHA")
        commit = str(raw["repository_commit"])
        if role not in SPLITS or not profile or _sha(stream) != stream_sha or _sha(run) != run_sha:
            raise CaseRetrievalError("input run content lineage differs")
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise CaseRetrievalError("input run repository commit is invalid")
        stream_value, run_value = _object(stream, "input manifest"), _object(run, "run manifest")
        if (
            not isinstance(stream_value.get("bindings"), Mapping)
            or stream_value["bindings"].get("run_manifest") != run.name
            or run_value.get("mode") != "market_case_input"
            or run_value.get("profile", {}).get("name") != profile
            or run_value.get("source", {}).get("sha256") != source_sha
            or run_value.get("model_config", {}).get("sha256") != config_sha
            or run_value.get("model_config", {}).get("timezone") != config["timezone"]
            or run_value.get("repository") != {"commit": commit}
            or run_value.get("market_case_input_identity") != protocol
        ):
            raise CaseRetrievalError("input run contract differs from combined lineage")
        if run_sha in runs:
            raise CaseRetrievalError("input run lineage is duplicated")
        runs[run_sha] = {"split_role": role, "stream_manifest_sha256": stream_sha}
    if {item["split_role"] for item in runs.values()} != set(SPLITS):
        raise CaseRetrievalError("combined artifact must cover train/validation/holdout")
    common = {
        "source_sha256": source_sha, "model_config_sha256": config_sha,
        "market_case_protocol": protocol,
        "representation_feature_schema_version": feature_schema,
        "calendar_timezone": config["timezone"],
    }
    return runs, common

def load_artifact_manifest(path: str | Path) -> dict[str, Any]:
    """Validate one combined fit artifact and every referenced neutral input run."""
    unresolved = Path(path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise CaseRetrievalError("artifact manifest is not a regular file")
    owner, manifest = unresolved.resolve(), _object(unresolved.resolve(), "artifact manifest")
    if (
        set(manifest) != MANIFEST_FIELDS or manifest.get("status") != "complete"
        or manifest.get("schema") not in {EMBEDDING_SCHEMA, HEAD_SCHEMA}
        or manifest.get("outcome_fields_used") is not False
        or manifest.get("model_capability_validated") is not False
        or manifest.get("input_protocol") != NEUTRAL_INFERENCE_INPUT_PROTOCOL
        or manifest.get("direct_source_preprocessing")
        != neutral_direct_source_preprocessing_identity()
        or manifest.get("selection_contract") != MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT
        or manifest.get("split_roles") != sorted(SPLITS)
        or manifest.get("material_kinds") != list(MARKET_EPISODE_MATERIAL_KINDS)
    ):
        raise CaseRetrievalError("combined artifact manifest contract is invalid")
    schema = manifest["schema"]
    expected_heads = dict(MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS)
    if manifest["head_schema"] != (None if schema == EMBEDDING_SCHEMA else expected_heads):
        raise CaseRetrievalError("combined artifact head schema is invalid")
    checkpoints = tuple(_digest(item, "checkpoint") for item in manifest["checkpoint_ids"])
    if (
        len(checkpoints) != len(set(checkpoints))
        or len(checkpoints) < (1 if schema == EMBEDDING_SCHEMA else 3)
        or (schema == EMBEDDING_SCHEMA and len(checkpoints) != 1)
    ):
        raise CaseRetrievalError("combined artifact checkpoint set is invalid")
    artifact = _bound(owner, manifest["artifact_path"], "artifact")
    if _sha(artifact) != _digest(manifest["artifact_sha256"], "artifact SHA"):
        raise CaseRetrievalError("combined artifact content hash differs")
    rows = _rows(artifact)
    if manifest["records"] != len(rows) or not rows:
        raise CaseRetrievalError("combined artifact record count is invalid")
    runs, common = _lineage(owner, manifest["lineage"], manifest["feature_schema_version"])
    dimension: int | None = None
    seen: set[tuple[Any, ...]] = set()
    member_checkpoints: dict[str, str] = {}
    for row in rows:
        run_sha = _digest(row.get("run_manifest_sha256"), "record run manifest SHA")
        if (
            row.get("run_manifest_sha256") != run_sha
            or run_sha not in runs
            or row.get("data_split") != runs[run_sha]["split_role"]
        ):
            raise CaseRetrievalError("record run/split lineage differs")
        decision, feature = _clock(row.get("decision_at"), "decision_at"), _clock(row.get("feature_max_at"), "feature_max_at")
        if feature > decision or row.get("outcome_fields_used") is not False:
            raise CaseRetrievalError("record clock/outcome contract is invalid")
        if schema == EMBEDDING_SCHEMA:
            if set(row) != EMBEDDING_FIELDS:
                raise CaseRetrievalError("embedding row fields are invalid")
            current_dim = row.get("embedding_dim")
            if (
                isinstance(current_dim, bool) or not isinstance(current_dim, int)
                or current_dim < 1 or row.get("embedding_model_version") != manifest["model_version"]
                or row.get("embedding_checkpoint_id") != checkpoints[0]
                or row.get("embedding_input_protocol") != manifest["input_protocol"]
                or row.get("material_kind") not in MARKET_EPISODE_MATERIAL_KINDS
            ):
                raise CaseRetrievalError("embedding row contract differs")
            dimension = current_dim if dimension is None else dimension
            if current_dim != dimension:
                raise CaseRetrievalError("embedding dimensions differ")
            parsed = MarketEpisodeEmbeddingQuery.from_mapping(
                row, material_kind=str(row["material_kind"]), embedding_dim=dimension,
            )
            grain = (run_sha, parsed.market_epoch_id, parsed.market_episode_id, parsed.material_kind)
        else:
            if (
                set(row) != HEAD_FIELDS
                or row.get("model_version") != manifest["model_version"]
                or row.get("input_protocol") != manifest["input_protocol"]
            ):
                raise CaseRetrievalError("active-head row contract differs")
            member, checkpoint = str(row.get("member_id", "")), _digest(row.get("checkpoint_id"), "head checkpoint")
            if not member or checkpoint not in checkpoints or member_checkpoints.setdefault(member, checkpoint) != checkpoint:
                raise CaseRetrievalError("active-head member/checkpoint differs")
            heads = row.get("head_predictions")
            if not isinstance(heads, Mapping) or set(heads) != set(expected_heads):
                raise CaseRetrievalError("active-head predictions are invalid")
            for name, width in expected_heads.items():
                probabilities = np.asarray(heads[name], dtype=float)
                if (
                    probabilities.shape != (width,)
                    or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0)
                    or not np.isclose(probabilities.sum(), 1.0)
                ):
                    raise CaseRetrievalError("active-head probabilities are invalid")
            grain = (*tuple(str(row[name]) for name in HEAD_IDENTITY_FIELDS), member)
        if grain in seen:
            raise CaseRetrievalError("combined artifact record grain is duplicated")
        seen.add(grain)
    if schema == HEAD_SCHEMA and (
        len(member_checkpoints) != len(checkpoints)
        or set(member_checkpoints.values()) != set(checkpoints)
    ):
        raise CaseRetrievalError("combined artifact does not bind independent ensemble members")
    common.update({
        "embedding_input_protocol": manifest["input_protocol"],
        "selection_contract": manifest["selection_contract"],
        "embedding_model_version": manifest["model_version"],
    })
    return {
        "schema": schema, "rows": rows, "runs": runs, "common_contract": common,
        "checkpoints": checkpoints, "embedding_dim": dimension,
        "artifact_sha256": manifest["artifact_sha256"], "lineage": manifest["lineage"],
    }

def _geometry(records: Sequence[Any]) -> dict[str, Any]:
    matrix = np.asarray([row.decision_embedding for row in records], dtype=float)
    spectrum = np.linalg.svd(matrix - matrix.mean(0), compute_uv=False) ** 2
    total = float(spectrum.sum())
    weights = spectrum[spectrum > 0] / total if total > np.finfo(float).eps else []
    rank = 0.0 if not len(weights) else float(np.exp(-np.sum(weights * np.log(weights))))
    pairs = np.empty(0) if len(matrix) < 2 else (matrix @ matrix.T)[np.triu_indices(len(matrix), 1)]
    return {
        "records": len(records), "effective_rank": rank,
        "p95_pairwise_cosine": None if not len(pairs) else float(np.quantile(pairs, 0.95)),
    }

def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))

def evaluate(
    embeddings: Mapping[str, Any],
    heads: Mapping[str, Any],
    *,
    k: int = 10,
    thresholds: OODThresholds | None = None,
) -> dict[str, Any]:
    """Use train references and validation/holdout queries from one encoder."""
    if embeddings["schema"] != EMBEDDING_SCHEMA or heads["schema"] != HEAD_SCHEMA or k < 1:
        raise CaseRetrievalError("combined evaluator artifact roles are invalid")
    if embeddings["lineage"] != heads["lineage"] or embeddings["common_contract"] != heads["common_contract"]:
        raise CaseRetrievalError("embedding/head dataset lineage differs")
    if embeddings["checkpoints"][0] not in heads["checkpoints"]:
        raise CaseRetrievalError("reference encoder is absent from ensemble checkpoints")
    expected_members = len(heads["checkpoints"])
    thresholds = replace(thresholds or OODThresholds(), minimum_ensemble_members=expected_members)
    contract = normalise_market_episode_dataset_contract(
        {**embeddings["common_contract"], "embedding_checkpoint_id": embeddings["checkpoints"][0]})
    dimension, runs = embeddings["embedding_dim"], embeddings["runs"]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in embeddings["rows"]:
        grouped.setdefault(str(row["run_manifest_sha256"]), []).append(row)
    artifacts = []
    for run_sha, rows in grouped.items():
        if rows[0]["data_split"] != "train":
            continue
        artifacts.append({
            "records": rows,
            "artifact_lineage": {"stream_manifest_sha256": runs[run_sha]["stream_manifest_sha256"],
                                 "run_manifest_sha256": run_sha, "selection_contract": SELECTION},
            "dataset_contract": contract,
        })
    index = MarketEpisodeCaseIndex.from_artifacts(artifacts, embedding_dim=dimension)
    parsed = [MarketEpisodeEmbeddingQuery.from_mapping(
        row, material_kind=str(row["material_kind"]), embedding_dim=dimension) for row in embeddings["rows"]]
    head_map: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in heads["rows"]:
        key = tuple(str(row[name]) for name in HEAD_IDENTITY_FIELDS)
        head_map.setdefault(key, []).append(row)
    embedding_bindings = {
        tuple(str(row[name]) for name in HEAD_IDENTITY_FIELDS):
        (_clock(row["decision_at"], "embedding decision_at"), _clock(row["feature_max_at"], "embedding feature_max_at")) for row in embeddings["rows"]}
    if any(
        key not in embedding_bindings
        or any(
            (_clock(row["decision_at"], "head decision_at"),
             _clock(row["feature_max_at"], "head feature_max_at"))
            != embedding_bindings[key] for row in rows
        )
        for key, rows in head_map.items()
    ):
        raise CaseRetrievalError("active-head artifact contains an orphan or misbound revision")
    geometry = {}
    for split in SPLITS:
        for kind in MARKET_EPISODE_MATERIAL_KINDS:
            selected = [row for row in parsed if row.data_split == split and row.material_kind == kind]
            if selected:
                geometry[f"{split}/{kind}"] = _geometry(selected)
    assessments, unique = [], {}
    for query in parsed:
        if query.data_split == "train":
            continue
        ensemble = head_map.get((query.run_manifest_sha256, query.market_epoch_id, query.market_episode_id, query.revision_id), ())
        result = index.query(
            query,
            artifact_lineage={"stream_manifest_sha256": runs[query.run_manifest_sha256]["stream_manifest_sha256"],
                              "run_manifest_sha256": query.run_manifest_sha256, "selection_contract": SELECTION},
            dataset_contract=contract, ensemble=ensemble, thresholds=thresholds,
            require_different_calendar_date=True, k=k,
        )
        assessments.append(result.ood)
        scope = (query.run_manifest_sha256, query.market_epoch_id, query.market_episode_id)
        if scope not in unique or result.ood.ensemble_members < unique[scope].ensemble_members:
            unique[scope] = result.ood
    if not assessments:
        raise CaseRetrievalError("combined evaluator has no validation/holdout queries")
    eligible = [float(item.eligible_neighbours) for item in assessments]
    disagreements = [item.ensemble_disagreement for item in unique.values() if item.ensemble_disagreement is not None]
    missing = [item for item in unique.values() if item.ensemble_members < expected_members]
    return {
        "schema_version": 2, "evaluation": "neutral_market_episode_geometry_retrieval_ood",
        "input_artifact_sha256": {"embeddings": embeddings["artifact_sha256"], "active_heads": heads["artifact_sha256"]},
        "direct_source_preprocessing": (
            neutral_direct_source_preprocessing_identity()
        ),
        "b0_compatible": True,
        "metrics": {
            "collapse": {"by_split_material_kind": geometry},
            "self_supervised_consistency": {
                "expected_ensemble_members": expected_members,
                "complete_ensemble_queries": sum(item.ensemble_members == expected_members for item in unique.values()),
                "incomplete_ensemble_queries": len(missing),
            },
            "cross_day_independent_episode_retrieval": {
                "coverage_with_eligible_neighbour": sum(value > 0 for value in eligible) / len(eligible),
                "mean_local_density": _mean([item.local_density for item in assessments]),
                "different_calendar_date_required": True,
                "scoped_episode_identity": "run_manifest_sha256+market_epoch_id+market_episode_id",
            },
            "ood": {
                "policy_counts": dict(sorted(Counter(item.policy.value for item in assessments).items())),
                "mean_nearest_cosine_distance": _mean([
                    item.nearest_cosine_distance for item in assessments
                    if item.nearest_cosine_distance is not None
                ]),
                "mean_ensemble_disagreement": _mean(disagreements),
                "missing_ensemble_queries": len(missing),
                "missing_ensemble_abstained": sum(item.policy.value == "abstain" for item in missing),
            },
        },
        "labels_used": False, "outcomes_used": False, "prediction_quality_claimed": False, "action_authority": "none",
    }

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--head-manifest", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        thresholds = (
            OODThresholds() if args.thresholds is None
            else OODThresholds(**_object(args.thresholds.resolve(), "thresholds"))
        )
        report = evaluate(
            load_artifact_manifest(args.embedding_manifest),
            load_artifact_manifest(args.head_manifest),
            k=args.k,
            thresholds=thresholds,
        )
        atomic_bytes(args.output.resolve(), canonical_json(report))
    except (CaseRetrievalError, TypeError, ValueError) as exc:
        raise SystemExit(f"neutral retrieval evaluation refused input: {exc}") from exc
    return 0

if __name__ == "__main__":
    main()
