#!/usr/bin/env python3
"""Build or query the episode-level causal-case similarity index.

This script consumes already materialised case/embedding artifacts.  It is not
a replay entry point and never calls the trading engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shares.core.artifact_stream import (  # noqa: E402
    canonical_json,
    normalise_sha256,
    publish_canonical_manifest,
    read_json_object,
    sha256_file,
)
from shares.core.case_retrieval import (  # noqa: E402
    DEFAULT_MARKET_EMBEDDING_DIM,
    CaseRetrievalError,
    EnsembleMemberPrediction,
    EpisodeCaseIndex,
    EpisodeEmbeddingQuery,
    OODThresholds,
    FIRST_STAGE_SELECTION_CONTRACT,
)
from shares.core.market_representation import (  # noqa: E402
    INFERENCE_INPUT_PROTOCOL,
    OUTCOME_BLIND_HEAD_WIDTHS,
)


_EMBEDDING_ARTIFACT_SCHEMA = "smc-decision-time-embeddings-v1"
_HEAD_ARTIFACT_SCHEMA = "smc-decision-time-self-supervised-heads-v1"
_IDENTITY_FIELDS = (
    "case_id",
    "revision_id",
    "entry_episode_id",
    "decision_at",
    "feature_max_at",
    "checkpoint_id",
    "member_id",
)


class _OutputNotPublishedError(CaseRetrievalError):
    """The destination CAS failed before this process published any bytes."""


_FileIdentity = tuple[int, int, int, int, int]


def _regular_file_identity(path: Path) -> _FileIdentity:
    """Capture enough identity to guard a later best-effort owned rollback."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CaseRetrievalError(
            f"published checkpoint identity is unavailable: {path}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CaseRetrievalError(
            f"published checkpoint is not a regular file: {path}"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _unlink_owned_regular_file(path: Path, identity: _FileIdentity) -> bool:
    """Remove only the unchanged regular-file inode published by this build."""

    try:
        metadata = path.lstat()
    except OSError:
        return False
    current = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if not stat.S_ISREG(metadata.st_mode) or current != identity:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


_JSON_FIELDS = frozenset(
    {
        "decision_embedding",
        "market_embedding",
        "embedding",
        "embedding_feature_names",
        "embedding_inputs",
        "frozen_outcome",
        "outcome",
        "head_predictions",
        "predictions",
    }
)


def _decode_json_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(record)
    for key in _JSON_FIELDS:
        value = output.get(key)
        if isinstance(value, str) and value.strip().startswith(("[", "{")):
            try:
                output[key] = json.loads(value)
            except json.JSONDecodeError as exc:
                raise CaseRetrievalError(f"{key} contains invalid JSON") from exc
    return output


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise CaseRetrievalError(f"input is not a regular file: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return [
            _decode_json_fields(record)
            for record in pd.read_parquet(path).to_dict(orient="records")
        ]
    if suffix == ".jsonl":
        values: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CaseRetrievalError(
                    f"invalid JSONL at line {line_number}"
                ) from exc
            if not isinstance(value, Mapping):
                raise CaseRetrievalError(
                    f"JSONL line {line_number} is not an object"
                )
            values.append(_decode_json_fields(value))
        return values
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaseRetrievalError(f"invalid JSON input: {path}") from exc
    if isinstance(payload, Mapping):
        for key in ("records", "cases", "members"):
            if key in payload:
                payload = payload[key]
                break
        else:
            return [_decode_json_fields(payload)]
    if not isinstance(payload, list) or any(
        not isinstance(item, Mapping) for item in payload
    ):
        raise CaseRetrievalError("JSON input must be an object or object list")
    return [_decode_json_fields(item) for item in payload]


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _manifest_sha(value: str | None, *, name: str) -> str:
    try:
        return normalise_sha256(value, name=name)
    except ValueError as exc:
        raise CaseRetrievalError(str(exc)) from exc


def _artifact_lineage(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_input_manifest_sha256": payload.get(
            "case_input_manifest_sha256"
        ),
        "case_library_manifest_sha256": payload.get(
            "case_library_manifest_sha256"
        ),
        "decision_stage": payload.get("decision_stage"),
        "selection_contract": payload.get("selection_contract"),
    }


def _record_identity_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    identities = [
        {
            "case_id": row.get("case_id"),
            "revision_id": row.get("revision_id"),
            "entry_episode_id": row.get("entry_episode_id"),
            "decision_at": row.get("embedding_asof", row.get("decision_at")),
            "feature_max_at": row.get("feature_max_at"),
            "checkpoint_id": row.get(
                "embedding_checkpoint_id", row.get("checkpoint_id")
            ),
            "member_id": row.get("member_id"),
        }
        for row in records
    ]
    encoded = json.dumps(
        identities,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_export_artifact(
    artifact: Path,
    manifest: Path,
    expected_manifest_sha256: str | None,
    *,
    expected_schema: str,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    expected_sha = _manifest_sha(
        expected_manifest_sha256,
        name=f"{expected_schema} manifest SHA",
    )
    if artifact.is_symlink() or not artifact.is_file():
        raise CaseRetrievalError(f"artifact is missing or not a regular file: {artifact}")
    if manifest.is_symlink() or not manifest.is_file():
        raise CaseRetrievalError(f"artifact manifest is missing: {manifest}")
    if _sha256_file(manifest) != expected_sha:
        raise CaseRetrievalError("artifact manifest content hash mismatch")
    try:
        payload = read_json_object(manifest, name="artifact manifest")
    except (OSError, ValueError) as exc:
        raise CaseRetrievalError(str(exc)) from exc
    if (
        payload.get("schema") != expected_schema
        or payload.get("status") != "complete"
        or payload.get("artifact_path") != artifact.name
        or payload.get("artifact_sha256") != _sha256_file(artifact)
        or payload.get("input_protocol") != INFERENCE_INPUT_PROTOCOL
        or payload.get("outcome_fields_used") is not False
        or payload.get("selection_contract")
        != FIRST_STAGE_SELECTION_CONTRACT
    ):
        raise CaseRetrievalError("artifact manifest identity or hash is invalid")
    records = _read_records(artifact)
    if payload.get("records") != len(records):
        raise CaseRetrievalError("artifact manifest row count is invalid")
    if (
        tuple(payload.get("record_identity_fields", ())) != _IDENTITY_FIELDS
        or payload.get("record_identity_sha256")
        != _record_identity_sha256(records)
    ):
        raise CaseRetrievalError("artifact record identity binding is invalid")
    if expected_schema == _HEAD_ARTIFACT_SCHEMA and payload.get(
        "head_schema"
    ) != dict(OUTCOME_BLIND_HEAD_WIDTHS):
        raise CaseRetrievalError("head artifact manifest schema is invalid")
    if expected_schema == _EMBEDDING_ARTIFACT_SCHEMA:
        stages = {str(record.get("revision_stage", "")) for record in records}
        model_versions = {
            str(record.get("embedding_model_version", "")) for record in records
        }
        checkpoint_ids = {
            str(record.get("embedding_checkpoint_id", "")) for record in records
        }
        if (
            len(stages) != 1
            or payload.get("decision_stage") not in stages
            or len(model_versions) != 1
            or payload.get("model_version") not in model_versions
            or sorted(checkpoint_ids) != payload.get("checkpoint_ids")
        ):
            raise CaseRetrievalError(
                "embedding artifact rows disagree with their sidecar identity"
            )
    return records, payload


def _outcomes_by_case_id(paths: Iterable[Path]) -> dict[str, Mapping[str, Any]]:
    outcomes: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        for record in _read_records(path):
            case_id = str(record.get("case_id", "")).strip()
            if not case_id:
                raise CaseRetrievalError("outcome record omits case_id")
            # A causal-case outcome shard stores outcome columns at top level;
            # nested outcome payloads are accepted for compact JSON fixtures.
            raw = record.get("frozen_outcome", record.get("outcome"))
            if raw is None:
                raw = {
                    str(key): value
                    for key, value in record.items()
                    if key not in {"case_id", "outcome_id"}
                }
            if not isinstance(raw, Mapping):
                raise CaseRetrievalError("outcome payload must be an object")
            prior = outcomes.get(case_id)
            if prior is not None and canonical_json(prior) != canonical_json(raw):
                raise CaseRetrievalError(
                    f"conflicting frozen outcomes for case {case_id}"
                )
            outcomes[case_id] = dict(raw)
    return outcomes


def _single_record(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(values) != 1:
        raise CaseRetrievalError("query input must contain exactly one object")
    return dict(values[0])


def _query_record(
    values: Sequence[Mapping[str, Any]],
    *,
    case_id: str | None,
    revision_id: str | None,
) -> dict[str, Any]:
    if case_id is None and revision_id is None:
        return _single_record(values)
    if not case_id or not revision_id:
        raise CaseRetrievalError(
            "bulk query selection requires both --query-case-id and "
            "--query-revision-id"
        )
    matches = [
        value
        for value in values
        if str(value.get("case_id", "")) == case_id
        and str(value.get("revision_id", "")) == revision_id
    ]
    if len(matches) != 1:
        raise CaseRetrievalError(
            "bulk embedding artifact does not contain one exact query revision"
        )
    return dict(matches[0])


def _ensemble(
    path: Path | None,
    *,
    manifest: Path | None,
    manifest_sha256: str | None,
    query: EpisodeEmbeddingQuery,
    expected_lineage: Mapping[str, Any],
) -> list[EnsembleMemberPrediction] | None:
    if path is None:
        return None
    if manifest is None:
        raise CaseRetrievalError("ensemble artifact requires its sidecar manifest")
    raw, payload = _validate_export_artifact(
        path,
        manifest,
        manifest_sha256,
        expected_schema=_HEAD_ARTIFACT_SCHEMA,
    )
    all_members = [EnsembleMemberPrediction.from_mapping(item) for item in raw]
    if _artifact_lineage(payload) != dict(expected_lineage):
        raise CaseRetrievalError(
            "head/query causal-case artifact lineages differ"
        )
    expected_checkpoint_ids = sorted(
        {member.checkpoint_id for member in all_members}
    )
    if payload.get("checkpoint_ids") != expected_checkpoint_ids:
        raise CaseRetrievalError("head artifact checkpoint identities are invalid")
    exact = [
        member
        for member in all_members
        if (
            member.case_id,
            member.revision_id,
            member.entry_episode_id,
            member.decision_at,
            member.feature_max_at,
            member.model_version,
        )
        == (
            query.case_id,
            query.revision_id,
            query.entry_episode_id,
            query.decision_at,
            query.feature_max_at,
            query.embedding_model_version,
        )
    ]
    return exact


def _thresholds(path: Path | None) -> OODThresholds:
    if path is None:
        return OODThresholds()
    value = _single_record(_read_records(path))
    try:
        return OODThresholds(**value)
    except TypeError as exc:
        raise CaseRetrievalError("OOD threshold keys are invalid") from exc


def _write_or_print(payload: Mapping[str, Any], destination: Path | None) -> None:
    encoded = canonical_json(payload)
    if destination is None:
        sys.stdout.write(encoded.decode("utf-8") + "\n")
    else:
        try:
            publish_canonical_manifest(destination, payload)
        except FileExistsError as exc:
            raise _OutputNotPublishedError(
                f"output already exists and will not be replaced: {destination}"
            ) from exc


def _build(args: argparse.Namespace) -> None:
    if args.report is not None and (
        args.report.is_symlink() or args.report.exists()
    ):
        raise _OutputNotPublishedError(
            f"output already exists and will not be replaced: {args.report}"
        )
    records: list[dict[str, Any]] = []
    if not (
        len(args.cases)
        == len(args.case_manifests)
        == len(args.case_manifest_shas)
    ):
        raise CaseRetrievalError(
            "each embedding artifact requires one manifest and pre-registered SHA"
        )
    checkpoint_ids: set[str] = set()
    artifact_lineage: dict[str, Any] | None = None
    for path, manifest, manifest_sha in zip(
        args.cases,
        args.case_manifests,
        args.case_manifest_shas,
        strict=True,
    ):
        artifact_records, payload = _validate_export_artifact(
            path,
            manifest,
            manifest_sha,
            expected_schema=_EMBEDDING_ARTIFACT_SCHEMA,
        )
        records.extend(artifact_records)
        checkpoint_ids.update(str(item) for item in payload.get("checkpoint_ids", ()))
        current_lineage = _artifact_lineage(payload)
        if artifact_lineage is None:
            artifact_lineage = current_lineage
        elif current_lineage != artifact_lineage:
            raise CaseRetrievalError(
                "embedding artifacts come from different causal-case lineages"
            )
    index = EpisodeCaseIndex.from_mappings(
        records,
        embedding_dim=args.embedding_dim,
        artifact_lineage=artifact_lineage,
    )
    if checkpoint_ids != {index.embedding_checkpoint_id}:
        raise CaseRetrievalError("embedding artifact checkpoint identity is invalid")
    try:
        checkpoint = index.save_checkpoint(args.output)
    except FileExistsError as exc:
        raise _OutputNotPublishedError(
            f"output already exists and will not be replaced: {args.output}"
        ) from exc
    checkpoint_identity = _regular_file_identity(checkpoint)
    try:
        _write_or_print(
            {
                "status": "complete",
                "checkpoint": str(checkpoint),
                "indexed_independent_episodes": len(index.records),
                "ignored_non_decision_revisions": (
                    index.ignored_non_decision_revisions
                ),
                "embedding_dim": index.embedding_dim,
                "outcomes_persisted": False,
                "action_authority": "none",
            },
            args.report,
        )
    except _OutputNotPublishedError:
        # Only this exception proves the report CAS never committed.  Other
        # exceptions have an unknown commit state (for example an interrupt
        # immediately after its link) and must preserve the checkpoint.
        _unlink_owned_regular_file(checkpoint, checkpoint_identity)
        raise


def _query(args: argparse.Namespace) -> None:
    index = EpisodeCaseIndex.load_checkpoint(args.index)
    query_records, query_manifest = _validate_export_artifact(
        args.query,
        args.query_manifest,
        args.query_manifest_sha,
        expected_schema=_EMBEDDING_ARTIFACT_SCHEMA,
    )
    query = EpisodeEmbeddingQuery.from_mapping(
        _query_record(
            query_records,
            case_id=args.query_case_id,
            revision_id=args.query_revision_id,
        ),
        embedding_dim=index.embedding_dim,
    )
    if query_manifest.get("checkpoint_ids") != [query.embedding_checkpoint_id]:
        raise CaseRetrievalError("query embedding checkpoint identity is invalid")
    if query_manifest.get("decision_stage") != query.revision_stage:
        raise CaseRetrievalError("query revision stage differs from its manifest")
    query_lineage = _artifact_lineage(query_manifest)
    result = index.query(
        query,
        k=args.k,
        ensemble=_ensemble(
            args.ensemble,
            manifest=args.ensemble_manifest,
            manifest_sha256=args.ensemble_manifest_sha,
            query=query,
            expected_lineage=query_lineage,
        ),
        thresholds=_thresholds(args.thresholds),
        frozen_outcomes=_outcomes_by_case_id(args.outcomes),
        artifact_lineage=query_lineage,
    )
    _write_or_print(result.as_dict(), args.output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build/query an outcome-blind EntryEpisode similarity and OOD index"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build an input-only index")
    build.add_argument(
        "--cases",
        type=Path,
        action="append",
        required=True,
        help="case+decision-embedding JSON/JSONL/Parquet (repeatable)",
    )
    build.add_argument(
        "--case-manifest",
        dest="case_manifests",
        type=Path,
        action="append",
        required=True,
        help="sidecar manifest for each --cases artifact",
    )
    build.add_argument(
        "--case-manifest-sha",
        dest="case_manifest_shas",
        action="append",
        required=True,
        help="pre-registered SHA-256 for each --case-manifest",
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path)
    build.add_argument(
        "--embedding-dim", type=int, default=DEFAULT_MARKET_EMBEDDING_DIM
    )
    build.set_defaults(run=_build)

    query = subparsers.add_parser("query", help="query one current episode")
    query.add_argument("--index", type=Path, required=True)
    query.add_argument("--query", type=Path, required=True)
    query.add_argument("--query-case-id")
    query.add_argument("--query-revision-id")
    query.add_argument("--query-manifest", type=Path, required=True)
    query.add_argument("--query-manifest-sha", required=True)
    query.add_argument(
        "--ensemble",
        type=Path,
        help="decision-time probability heads from independent checkpoints",
    )
    query.add_argument("--ensemble-manifest", type=Path)
    query.add_argument("--ensemble-manifest-sha")
    query.add_argument(
        "--outcomes",
        type=Path,
        action="append",
        default=[],
        help="separate frozen outcome JSON/JSONL/Parquet (repeatable)",
    )
    query.add_argument("--thresholds", type=Path)
    query.add_argument("--k", type=int, default=10)
    query.add_argument("--output", type=Path)
    query.set_defaults(run=_query)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        args.run(args)
    except CaseRetrievalError as exc:
        raise SystemExit(f"case retrieval refused input: {exc}") from exc


if __name__ == "__main__":
    main()
