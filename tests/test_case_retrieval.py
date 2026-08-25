from __future__ import annotations

from datetime import timedelta
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from smc_trader.case_retrieval import (
    CASE_RETRIEVAL_PROTOCOL,
    CASE_RETRIEVAL_SCHEMA_VERSION,
    MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS,
    MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT,
    MARKET_EPISODE_MATERIAL_KINDS,
    MARKET_EPISODE_RETRIEVAL_PROTOCOL,
    CaseRetrievalError,
    EnsembleMemberPrediction,
    EpisodeCaseIndex,
    EpisodeEmbeddingQuery,
    MarketEpisodeCaseIndex,
    MarketEpisodeEmbeddingQuery,
    MarketEpisodeEmbeddingRecord,
    OODThresholds,
    RetrievalPolicy,
)
from smc_trader.market_representation import (
    EMBEDDING_DIM,
    INFERENCE_INPUT_PROTOCOL,
    NEUTRAL_INFERENCE_INPUT_PROTOCOL,
    OUTCOME_BLIND_HEAD_WIDTHS,
    DecisionTimeEmbeddingRecord,
    checkpoint_embedding_contract,
    neutral_direct_source_preprocessing_identity,
)
from smc_trader.scene_graph import market_episode_id


DIM = 4
BASE = pd.Timestamp("2022-01-03T10:00:00-05:00")
EMBEDDING_CHECKPOINT_ID = "a" * 64
MARKET_EPISODE_LINEAGE = {
    "stream_manifest_sha256": "b" * 64,
    "run_manifest_sha256": "c" * 64,
    "selection_contract": MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT,
}
MARKET_EPISODE_DATASET_CONTRACT = {
    "source_sha256": "d" * 64,
    "model_config_sha256": "e" * 64,
    "market_case_protocol": {"protocol_version": "neutral-test-v1"},
    "representation_feature_schema_version": "feature-test-v1",
    "embedding_input_protocol": NEUTRAL_INFERENCE_INPUT_PROTOCOL,
    "selection_contract": MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT,
    "embedding_model_version": "representation:test-v1",
    "embedding_checkpoint_id": EMBEDDING_CHECKPOINT_ID,
    "calendar_timezone": "America/New_York",
}


def _case(
    episode: str,
    minute: int,
    embedding: Sequence[float],
    *,
    case_id: str | None = None,
    context: str | None = None,
    epoch: str = "epoch:1",
    split: str = "train",
    direction: str = "long",
    regime: str = "continuation",
    embedding_clock: str = "decision_time",
    revision_stage: str = "plan_formed",
    revision_index: int = 1,
    stage_occurrence: int = 0,
    outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    decision_at = BASE + timedelta(minutes=int(minute))
    return {
        "case_id": case_id or f"case:{episode}",
        "revision_id": f"revision:{episode}:{revision_stage}",
        "revision_stage": revision_stage,
        "revision_index": revision_index,
        "stage_identity": f"{revision_stage}:{episode}:{revision_index}",
        "stage_occurrence": stage_occurrence,
        "market_epoch_id": epoch,
        "context_thesis_id": context or f"context:{episode}",
        "entry_episode_id": episode,
        "decision_at": decision_at,
        "revision_at": decision_at,
        "direction": direction,
        "regime": regime,
        "split_role": split,
        "embedding_model_version": "representation:test-v1",
        "embedding_checkpoint_id": EMBEDDING_CHECKPOINT_ID,
        "embedding_input_protocol": INFERENCE_INPUT_PROTOCOL,
        "decision_embedding": list(embedding),
        "embedding_clock": embedding_clock,
        "embedding_asof": decision_at,
        "feature_max_at": decision_at,
        "outcome_fields_used": False,
        "embedding_feature_names": [
            "return_1m",
            "atr_normalized_distance",
            "typed_event_token",
        ],
        "frozen_outcome": dict(outcome or {}),
    }


def _query(
    minute: int,
    embedding: Sequence[float],
    *,
    episode: str = "episode:query",
    epoch: str = "epoch:1",
    split: str = "train",
    reference_splits: Sequence[str] | None = None,
) -> EpisodeEmbeddingQuery:
    raw = _case(
        episode,
        minute,
        embedding,
        epoch=epoch,
        split=split,
    )
    raw.pop("frozen_outcome")
    if reference_splits is not None:
        raw["allowed_reference_splits"] = list(reference_splits)
    return EpisodeEmbeddingQuery.from_mapping(raw, embedding_dim=DIM)


def _ensemble(
    values: Sequence[float],
    *,
    query: EpisodeEmbeddingQuery | None = None,
) -> list[EnsembleMemberPrediction]:
    bound = query or _query(20, [1.0, 0.0, 0.0, 0.0])
    return [
        EnsembleMemberPrediction(
            member_id=f"member:{index}",
            checkpoint_id=hashlib.sha256(
                f"checkpoint:{index}".encode("utf-8")
            ).hexdigest(),
            model_version=bound.embedding_model_version,
            case_id=bound.case_id,
            revision_id=bound.revision_id,
            entry_episode_id=bound.entry_episode_id,
            decision_at=bound.decision_at,
            feature_max_at=bound.feature_max_at,
            outcome_fields_used=False,
            head_predictions={
                name: tuple(
                    [value, 1.0 - value]
                    + [0.0] * (width - 2)
                )
                for name, width in OUTCOME_BLIND_HEAD_WIDTHS.items()
            },
            input_protocol=INFERENCE_INPUT_PROTOCOL,
        )
        for index, value in enumerate(values)
    ]


def _market_episode_case(
    episode: str,
    minute: int,
    embedding: Sequence[float],
    *,
    transition_kinds: Sequence[str] = ("trigger",),
    revision_index: int = 1,
    epoch: str = "epoch:neutral",
    direction: str = "long",
    split: str = "train",
    run_manifest_sha256: str = "c" * 64,
) -> dict[str, Any]:
    location_id = f"location:{episode}"
    path_id = f"path:{episode}"
    episode_id = market_episode_id(
        epoch,
        location_id,
        path_id,
        direction,
    )
    decision_at = BASE + timedelta(minutes=int(minute))
    return {
        "revision_id": f"market-revision:{episode}:{revision_index}",
        "revision_index": revision_index,
        "revision_stage": "market_episode_transition",
        "run_manifest_sha256": run_manifest_sha256,
        "market_epoch_id": epoch,
        "market_episode_id": episode_id,
        "entry_location_id": location_id,
        "entry_path_id": path_id,
        "direction": direction,
        "data_split": split,
        "transition_kinds_json": json.dumps(list(transition_kinds)),
        "decision_at": decision_at,
        "embedding_clock": "decision_time",
        "embedding_asof": decision_at,
        "feature_max_at": decision_at,
        "embedding_model_version": "representation:test-v1",
        "embedding_checkpoint_id": EMBEDDING_CHECKPOINT_ID,
        "embedding_input_protocol": NEUTRAL_INFERENCE_INPUT_PROTOCOL,
        "decision_embedding": list(embedding),
        "outcome_fields_used": False,
        "embedding_feature_names": ("return_1m", "atr_normalized_distance"),
    }


def _market_episode_query(
    minute: int,
    embedding: Sequence[float],
    *,
    episode: str = "query",
    material_kind: str = "trigger",
    epoch: str = "epoch:neutral",
    split: str = "train",
    run_manifest_sha256: str = "c" * 64,
) -> MarketEpisodeEmbeddingQuery:
    return MarketEpisodeEmbeddingQuery.from_mapping(
        _market_episode_case(
            episode,
            minute,
            embedding,
            transition_kinds=(material_kind,),
            split=split,
            run_manifest_sha256=run_manifest_sha256,
        ),
        material_kind=material_kind,
        embedding_dim=DIM,
    )


def _market_episode_ensemble(
    values: Sequence[float],
    *,
    query: MarketEpisodeEmbeddingQuery,
) -> list[dict[str, Any]]:
    return [
        {
            "member_id": f"neutral-member:{index}",
            "checkpoint_id": hashlib.sha256(
                f"neutral-checkpoint:{index}".encode("utf-8")
            ).hexdigest(),
            "model_version": query.embedding_model_version,
            "revision_id": query.revision_id,
            "run_manifest_sha256": query.run_manifest_sha256,
            "market_epoch_id": query.market_epoch_id,
            "market_episode_id": query.market_episode_id,
            "decision_at": query.decision_at,
            "feature_max_at": query.feature_max_at,
            "outcome_fields_used": False,
            "head_predictions": {
                name: tuple(
                    [value, 1.0 - value]
                    + [0.0] * (width - 2)
                )
                for name, width in MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS.items()
            },
            "input_protocol": NEUTRAL_INFERENCE_INPUT_PROTOCOL,
        }
        for index, value in enumerate(values)
    ]


def _close_cases(count: int = 5) -> list[dict[str, Any]]:
    return [
        _case(
            f"episode:{index}",
            index,
            [1.0, index * 0.01, 0.0, 0.0],
            direction="long" if index % 2 == 0 else "short",
            regime="continuation" if index < 3 else "balance",
        )
        for index in range(count)
    ]


def _write_export_manifest(
    artifact: Path,
    *,
    schema: str,
    records: int,
    checkpoint_ids: Sequence[str],
    head_schema: Mapping[str, int] | None = None,
) -> tuple[Path, str]:
    manifest = artifact.with_name(f"{artifact.name}.manifest.json")
    materialized = json.loads(artifact.read_text(encoding="utf-8"))
    materialized_rows = (
        materialized if isinstance(materialized, list) else [materialized]
    )
    identity_rows = [
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
        for row in materialized_rows
    ]
    identity_sha = hashlib.sha256(
        json.dumps(
            identity_rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": schema,
        "status": "complete",
        "records": records,
        "artifact_path": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "record_identity_sha256": identity_sha,
        "record_identity_fields": [
            "case_id",
            "revision_id",
            "entry_episode_id",
            "decision_at",
            "feature_max_at",
            "checkpoint_id",
            "member_id",
        ],
        "model_version": "representation:test-v1",
        "feature_schema_version": 1,
        "decision_stage": "plan_formed",
        "selection_contract": "first_online_stage_occurrence_by_revision_index_v1",
        "input_protocol": INFERENCE_INPUT_PROTOCOL,
        "outcome_fields_used": False,
        "checkpoint_ids": sorted(checkpoint_ids),
        "case_input_manifest_sha256": None,
        "case_library_manifest_sha256": None,
        "head_schema": None if head_schema is None else dict(head_schema),
    }
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _bound_outcome(
    record: Mapping[str, Any],
    *,
    resolved_at: pd.Timestamp,
    **payload: Any,
) -> dict[str, Any]:
    return {
        "market_epoch_id": record["market_epoch_id"],
        "context_thesis_id": record["context_thesis_id"],
        "entry_episode_id": record["entry_episode_id"],
        "resolved_at": resolved_at,
        **payload,
    }


def test_protocol_is_episode_grain_outcome_blind_and_non_actionable() -> None:
    assert CASE_RETRIEVAL_PROTOCOL["case_grain"] == (
        "one_independent_entry_episode"
    )
    assert CASE_RETRIEVAL_PROTOCOL["outcome_in_index_vector"] is False
    assert CASE_RETRIEVAL_PROTOCOL["action_authority"] == (
        "none_empirical_prior_only"
    )
    assert tuple(CASE_RETRIEVAL_PROTOCOL["policy_outputs"]) == (
        "continue_evaluation",
        "increase_uncertainty",
        "abstain",
    )


def test_future_outcome_changes_do_not_change_vectors_or_neighbours() -> None:
    cases_a = _close_cases(5)
    cases_b = _close_cases(5)
    outcome_a = {
        cases_a[0]["case_id"]: _bound_outcome(
            cases_a[0],
            resolved_at=BASE + timedelta(minutes=10),
            first_event="target",
            mfe_r=2.2,
        )
    }
    outcome_b = {
        cases_b[0]["case_id"]: _bound_outcome(
            cases_b[0],
            resolved_at=BASE + timedelta(minutes=10),
            first_event="invalidation",
            mfe_r=0.1,
        )
    }
    index_a = EpisodeCaseIndex.from_mappings(cases_a, embedding_dim=DIM)
    index_b = EpisodeCaseIndex.from_mappings(cases_b, embedding_dim=DIM)
    query = _query(20, [1.0, 0.0, 0.0, 0.0])

    result_a = index_a.query(
        query,
        k=5,
        ensemble=_ensemble([0.49, 0.50, 0.51]),
        frozen_outcomes=outcome_a,
    )
    result_b = index_b.query(
        query,
        k=5,
        ensemble=_ensemble([0.49, 0.50, 0.51]),
        frozen_outcomes=outcome_b,
    )

    assert np.array_equal(index_a.vectors, index_b.vectors)
    assert [item.entry_episode_id for item in result_a.neighbours] == [
        item.entry_episode_id for item in result_b.neighbours
    ]
    assert [item.cosine_distance for item in result_a.neighbours] == [
        item.cosine_distance for item in result_b.neighbours
    ]
    assert (
        result_a.frozen_outcome_distribution
        != result_b.frozen_outcome_distribution
    )
    assert result_a.ood == result_b.ood


def _representation_export_row(
    episode: str,
    minute: int,
    *,
    split: str,
    offset: float,
) -> dict[str, Any]:
    values = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    values[0] = 1.0
    values[1] = offset
    values /= np.linalg.norm(values)
    timestamp = BASE + timedelta(minutes=int(minute))
    return DecisionTimeEmbeddingRecord(
        case_id=f"representation-case:{episode}",
        revision_id=f"representation-revision:{episode}",
        revision_stage="plan_formed",
        revision_index=1,
        stage_identity=f"plan_formed:{episode}:1",
        stage_occurrence=0,
        context_thesis_id=f"representation-context:{episode}",
        entry_episode_id=episode,
        market_epoch_id="epoch:representation",
        direction=1,
        regime="continuation",
        embedding_model_version=checkpoint_embedding_contract()["model_version"],
        embedding_checkpoint_id=EMBEDDING_CHECKPOINT_ID,
        embedding_dim=EMBEDDING_DIM,
        embedding_clock="decision_time",
        embedding_asof=timestamp,
        feature_max_at=timestamp,
        data_split=split,
        split_role=split,
        outcome_fields_used=False,
        decision_embedding=tuple(float(item) for item in values),
    ).as_dict()


def test_representation_export_contract_is_consumed_without_outcome_leakage() -> None:
    exported = [
        _representation_export_row(
            f"episode:representation:{index}",
            index,
            split="train",
            offset=index * 0.01,
        )
        for index in range(5)
    ]
    non_decision_revisions = []
    for stage, minute in (("zone_registered", 7), ("terminal", 30)):
        revision = dict(exported[0])
        revision.update(
            {
                "revision_id": f"revision:{stage}",
                "revision_at": BASE + timedelta(minutes=int(minute)),
                "embedding_clock": "case_revision",
            }
        )
        non_decision_revisions.append(revision)
    index = EpisodeCaseIndex.from_mappings(
        [*exported, *non_decision_revisions], embedding_dim=EMBEDDING_DIM
    )
    query_raw = _representation_export_row(
        "episode:representation:query",
        20,
        split="validation",
        offset=0.0,
    )
    query_raw["allowed_reference_splits"] = ["train"]
    query = EpisodeEmbeddingQuery.from_mapping(
        query_raw, embedding_dim=EMBEDDING_DIM
    )
    outcome_a = {
        exported[0]["case_id"]: _bound_outcome(
            exported[0],
            resolved_at=BASE + timedelta(minutes=10),
            first_event="target",
            mfe_r=2.0,
        )
    }
    outcome_b = {
        exported[0]["case_id"]: _bound_outcome(
            exported[0],
            resolved_at=BASE + timedelta(minutes=10),
            first_event="invalidation",
            mfe_r=0.0,
        )
    }
    ensemble = _ensemble([0.49, 0.50, 0.51], query=query)

    first = index.query(
        query,
        k=5,
        ensemble=ensemble,
        frozen_outcomes=outcome_a,
    )
    second = index.query(
        query,
        k=5,
        ensemble=ensemble,
        frozen_outcomes=outcome_b,
    )

    assert index.ignored_non_decision_revisions == 2
    assert len(index.records) == 5
    assert [item.entry_episode_id for item in first.neighbours] == [
        item.entry_episode_id for item in second.neighbours
    ]
    assert [item.similarity for item in first.neighbours] == [
        item.similarity for item in second.neighbours
    ]
    assert first.ood == second.ood
    assert first.sufficient_neighbours is True
    assert first.time_distribution == {"2022-01": 5}
    assert first.direction_distribution == {"1": 5}
    assert first.regime_distribution == {"continuation": 5}
    assert first.frozen_outcome_distribution["first_terminal"] == {
        "target": 1
    }
    assert second.frozen_outcome_distribution["first_terminal"] == {
        "invalidation": 1
    }
    assert first.ood.policy is RetrievalPolicy.CONTINUE_EVALUATION


def test_same_episode_revisions_keep_only_explicit_decision_clock() -> None:
    decision = _case("episode:a", 5, [1.0, 0.0, 0.0, 0.0])
    context_revision = dict(decision)
    context_revision.update(
        {
            "embedding_clock": "case_revision",
            "revision_at": BASE + timedelta(minutes=1),
        }
    )
    terminal_revision = dict(decision)
    terminal_revision.update(
        {
            "embedding_clock": "case_revision",
            "revision_at": BASE + timedelta(minutes=20),
        }
    )

    index = EpisodeCaseIndex.from_mappings(
        [context_revision, decision, terminal_revision], embedding_dim=DIM
    )

    assert len(index.records) == 1
    assert index.records[0].entry_episode_id == "episode:a"
    assert index.ignored_non_decision_revisions == 2


def test_multi_stage_episode_is_compared_only_at_query_stage() -> None:
    episode_a_trigger = _case(
        "episode:a",
        1,
        [0.0, 1.0, 0.0, 0.0],
        revision_stage="trigger",
    )
    episode_a_plan = _case(
        "episode:a",
        2,
        [1.0, 0.0, 0.0, 0.0],
        revision_stage="plan_formed",
    )
    episode_b_plan = _case(
        "episode:b",
        3,
        [0.9, 0.1, 0.0, 0.0],
        revision_stage="plan_formed",
    )
    index = EpisodeCaseIndex.from_mappings(
        [episode_a_trigger, episode_a_plan, episode_b_plan],
        embedding_dim=DIM,
    )

    query = _query(10, [1.0, 0.0, 0.0, 0.0])
    result = index.query(
        query,
        k=5,
        ensemble=_ensemble([0.49, 0.50, 0.51], query=query),
    )

    assert len(index.records) == 3
    assert result.query_revision_stage == "plan_formed"
    assert [item.entry_episode_id for item in result.neighbours] == [
        "episode:a",
        "episode:b",
    ]
    assert all(item.revision_stage == "plan_formed" for item in result.neighbours)


def test_context_changed_corpus_ignores_explicit_later_stage_occurrence() -> None:
    first = _case(
        "episode:context",
        1,
        [1.0, 0.0, 0.0, 0.0],
        revision_stage="context_changed",
        revision_index=2,
    )
    later = _case(
        "episode:context",
        2,
        [0.0, 1.0, 0.0, 0.0],
        revision_stage="context_changed",
        revision_index=3,
        stage_occurrence=1,
    )

    index = EpisodeCaseIndex.from_mappings(
        [later, first], embedding_dim=DIM
    )

    assert len(index.records) == 1
    assert index.records[0].revision_index == 2
    assert index.ignored_non_decision_revisions == 1


def test_multiple_material_revisions_keep_only_first_stage_occurrence() -> None:
    first = _case(
        "episode:context-first",
        1,
        [1.0, 0.0, 0.0, 0.0],
        revision_stage="context_changed",
        revision_index=2,
    )
    later = _case(
        "episode:context-first",
        2,
        [0.0, 1.0, 0.0, 0.0],
        revision_stage="context_changed",
        revision_index=3,
    )

    index = EpisodeCaseIndex.from_mappings(
        [later, first], embedding_dim=DIM
    )

    assert len(index.records) == 1
    assert index.records[0].revision_id == first["revision_id"]
    assert index.records[0].revision_index == 2
    assert index.ignored_non_decision_revisions == 1


def test_conflicting_decision_rows_for_one_episode_fail_closed() -> None:
    first = _case("episode:a", 1, [1.0, 0.0, 0.0, 0.0])
    conflicting = dict(first)
    conflicting["decision_embedding"] = [0.0, 1.0, 0.0, 0.0]
    with pytest.raises(CaseRetrievalError, match="conflicting first-occurrence"):
        EpisodeCaseIndex.from_mappings(
            [first, conflicting], embedding_dim=DIM
        )

    conflicting_identity = dict(first)
    conflicting_identity["revision_id"] = "revision:duplicate-online-index"
    conflicting_identity["decision_at"] = first["decision_at"] + timedelta(minutes=1)
    conflicting_identity["revision_at"] = conflicting_identity["decision_at"]
    conflicting_identity["embedding_asof"] = conflicting_identity["decision_at"]
    conflicting_identity["feature_max_at"] = conflicting_identity["decision_at"]
    with pytest.raises(CaseRetrievalError, match="conflicting first-occurrence identity"):
        EpisodeCaseIndex.from_mappings(
            [first, conflicting_identity], embedding_dim=DIM
        )


def test_index_and_query_must_share_exact_encoder_checkpoint() -> None:
    index = EpisodeCaseIndex.from_mappings(_close_cases(5), embedding_dim=DIM)
    raw = _case(
        "episode:different-encoder",
        20,
        [1.0, 0.0, 0.0, 0.0],
    )
    raw["embedding_checkpoint_id"] = "b" * 64
    raw.pop("frozen_outcome")
    query = EpisodeEmbeddingQuery.from_mapping(raw, embedding_dim=DIM)

    with pytest.raises(CaseRetrievalError, match="checkpoint embedding spaces"):
        index.query(query)

    mixed = _close_cases(2)
    mixed[1]["embedding_checkpoint_id"] = "b" * 64
    with pytest.raises(CaseRetrievalError, match="mix encoder checkpoint"):
        EpisodeCaseIndex.from_mappings(mixed, embedding_dim=DIM)

    wrong_model = _case(
        "episode:different-model", 20, [1.0, 0.0, 0.0, 0.0]
    )
    wrong_model["embedding_model_version"] = "representation:other"
    wrong_model.pop("frozen_outcome")
    with pytest.raises(CaseRetrievalError, match="model versions differ"):
        index.query(EpisodeEmbeddingQuery.from_mapping(wrong_model, embedding_dim=DIM))


def test_same_episode_cannot_be_shared_across_train_and_validation() -> None:
    train = _case(
        "episode:shared", 1, [1.0, 0.0, 0.0, 0.0], split="train"
    )
    validation = _case(
        "episode:shared",
        1,
        [1.0, 0.0, 0.0, 0.0],
        split="validation",
    )
    with pytest.raises(CaseRetrievalError, match="shared across data splits"):
        EpisodeCaseIndex.from_mappings(
            [train, validation], embedding_dim=DIM
        )


def test_query_excludes_self_future_other_split_and_other_epoch() -> None:
    records = [
        _case("episode:eligible", 1, [0.7, 0.3, 0.0, 0.0]),
        _case(
            "episode:other-split",
            2,
            [1.0, 0.0, 0.0, 0.0],
            split="validation",
        ),
        _case(
            "episode:other-epoch",
            2,
            [1.0, 0.0, 0.0, 0.0],
            epoch="epoch:2",
        ),
        _case("episode:query", 2, [1.0, 0.0, 0.0, 0.0]),
        _case("episode:same-clock", 5, [1.0, 0.0, 0.0, 0.0]),
        _case("episode:future", 6, [1.0, 0.0, 0.0, 0.0]),
    ]
    index = EpisodeCaseIndex.from_mappings(records, embedding_dim=DIM)

    query = _query(5, [1.0, 0.0, 0.0, 0.0])
    result = index.query(
        query,
        k=10,
        ensemble=_ensemble([0.49, 0.50, 0.51], query=query),
    )

    assert [item.entry_episode_id for item in result.neighbours] == [
        "episode:eligible"
    ]
    assert result.ood.policy is RetrievalPolicy.INCREASE_UNCERTAINTY


def test_validation_query_reads_train_reference_only() -> None:
    records = [
        _case("episode:train", 1, [0.7, 0.3, 0.0, 0.0], split="train"),
        _case(
            "episode:validation",
            2,
            [1.0, 0.0, 0.0, 0.0],
            split="validation",
        ),
        _case(
            "episode:test",
            2,
            [1.0, 0.0, 0.0, 0.0],
            split="test",
        ),
    ]
    index = EpisodeCaseIndex.from_mappings(records, embedding_dim=DIM)

    query = _query(
        5,
        [1.0, 0.0, 0.0, 0.0],
        split="validation",
    )
    result = index.query(
        query,
        k=10,
        ensemble=_ensemble([0.49, 0.50, 0.51], query=query),
    )

    assert [item.entry_episode_id for item in result.neighbours] == [
        "episode:train"
    ]
    assert all(item.data_split == "train" for item in result.neighbours)
    with pytest.raises(CaseRetrievalError, match="evaluation-role"):
        _query(
            5,
            [1.0, 0.0, 0.0, 0.0],
            split="validation",
            reference_splits=("validation",),
        )


def test_leakage_guard_rejects_future_fields_and_post_decision_clocks() -> None:
    leaked_name = _case("episode:a", 1, [1.0, 0.0, 0.0, 0.0])
    leaked_name["embedding_feature_names"] = ["return_1m", "mfe_r"]
    with pytest.raises(CaseRetrievalError, match="outcome/future"):
        EpisodeCaseIndex.from_mappings([leaked_name], embedding_dim=DIM)

    leaked_mapping = _case("episode:a", 1, [1.0, 0.0, 0.0, 0.0])
    leaked_mapping["embedding_inputs"] = {
        "causal": {"return": 0.1},
        "outcome": {"target_hit": True},
    }
    with pytest.raises(CaseRetrievalError, match="outcome/future"):
        EpisodeCaseIndex.from_mappings([leaked_mapping], embedding_dim=DIM)

    future_clock = _case("episode:a", 1, [1.0, 0.0, 0.0, 0.0])
    future_clock["feature_max_at"] = (
        future_clock["decision_at"] + timedelta(minutes=1)
    )
    with pytest.raises(CaseRetrievalError, match="after the decision clock"):
        EpisodeCaseIndex.from_mappings([future_clock], embedding_dim=DIM)

    unspecified = _case("episode:a", 1, [1.0, 0.0, 0.0, 0.0])
    unspecified.pop("outcome_fields_used")
    with pytest.raises(CaseRetrievalError, match="explicitly false"):
        EpisodeCaseIndex.from_mappings([unspecified], embedding_dim=DIM)


def test_sparse_cases_raise_uncertainty_and_missing_ensemble_abstains() -> None:
    index = EpisodeCaseIndex.from_mappings(
        [_case("episode:a", 1, [1.0, 0.0, 0.0, 0.0])],
        embedding_dim=DIM,
    )
    query = _query(5, [1.0, 0.0, 0.0, 0.0])

    sparse = index.query(
        query,
        ensemble=_ensemble([0.49, 0.50, 0.51], query=query),
    )
    missing_model = index.query(query, ensemble=None)

    assert sparse.sufficient_neighbours is False
    assert sparse.ood.policy is RetrievalPolicy.INCREASE_UNCERTAINTY
    assert "insufficient_independent_neighbours" in sparse.ood.reasons
    assert missing_model.ood.policy is RetrievalPolicy.ABSTAIN
    assert "deep_ensemble_unavailable" in missing_model.ood.reasons


def test_high_disagreement_abstains_and_moderate_disagreement_is_uncertain() -> None:
    index = EpisodeCaseIndex.from_mappings(_close_cases(6), embedding_dim=DIM)
    query = _query(20, [1.0, 0.0, 0.0, 0.0])

    high = index.query(query, ensemble=_ensemble([0.0, 0.0, 1.0]))
    moderate = index.query(query, ensemble=_ensemble([0.2, 0.5, 0.8]))

    assert high.ood.policy is RetrievalPolicy.ABSTAIN
    assert "ensemble_disagreement_out_of_distribution" in high.ood.reasons
    assert moderate.ood.policy is RetrievalPolicy.INCREASE_UNCERTAINTY
    assert "elevated_ensemble_disagreement" in moderate.ood.reasons


def test_ensemble_heads_are_bound_to_exact_query_revision_and_clock() -> None:
    index = EpisodeCaseIndex.from_mappings(_close_cases(6), embedding_dim=DIM)
    query = _query(20, [1.0, 0.0, 0.0, 0.0])
    members = _ensemble([0.49, 0.50, 0.51], query=query)

    with pytest.raises(CaseRetrievalError, match="revision binding"):
        index.query(
            query,
            ensemble=[replace(members[0], revision_id="revision:other"), *members[1:]],
        )
    with pytest.raises(CaseRetrievalError, match="decision clock binding"):
        index.query(
            query,
            ensemble=[
                replace(
                    members[0],
                    decision_at=query.decision_at - timedelta(minutes=1),
                    feature_max_at=query.feature_max_at - timedelta(minutes=1),
                ),
                *members[1:],
            ],
        )


def test_ensemble_refuses_unregistered_or_masked_prediction_heads() -> None:
    member = _ensemble([0.5])[0]
    with pytest.raises(CaseRetrievalError, match="fixed outcome-blind head schema"):
        replace(
            member,
            head_predictions={
                **member.head_predictions,
                "future_profit": (0.5, 0.5),
            },
        )
    with pytest.raises(CaseRetrievalError, match="deterministic unmasked"):
        replace(member, input_protocol="training_random_mask_v1")


def test_sufficient_close_cases_and_consistent_ensemble_continue_evaluation() -> None:
    index = EpisodeCaseIndex.from_mappings(_close_cases(6), embedding_dim=DIM)

    result = index.query(
        _query(20, [1.0, 0.0, 0.0, 0.0]),
        k=5,
        ensemble=_ensemble([0.49, 0.50, 0.51]),
    )

    assert len(result.neighbours) == 5
    assert len({item.entry_episode_id for item in result.neighbours}) == 5
    assert result.sufficient_neighbours is True
    assert result.ood.policy is RetrievalPolicy.CONTINUE_EVALUATION
    assert result.direction_distribution == {"long": 3, "short": 2}
    assert result.regime_distribution == {"balance": 2, "continuation": 3}


def test_obvious_embedding_ood_abstains() -> None:
    index = EpisodeCaseIndex.from_mappings(_close_cases(6), embedding_dim=DIM)

    result = index.query(
        _query(20, [0.0, 0.0, 1.0, 0.0]),
        k=5,
        ensemble=_ensemble([0.49, 0.50, 0.51]),
    )

    assert result.ood.policy is RetrievalPolicy.ABSTAIN
    assert "nearest_embedding_out_of_distribution" in result.ood.reasons


def test_outcomes_are_joined_after_selection_and_only_as_distribution() -> None:
    records = _close_cases(5)
    index = EpisodeCaseIndex.from_mappings(records, embedding_dim=DIM)
    outcomes = {
        "case:episode:0": _bound_outcome(
            records[0],
            resolved_at=BASE + timedelta(minutes=10),
            first_event="target",
            filled=True,
            draw_delivered=True,
            reached_1r=True,
            mfe_r=1.5,
            mae_r=0.25,
        ),
        "case:episode:1": _bound_outcome(
            records[1],
            resolved_at=BASE + timedelta(minutes=11),
            first_event="deadline",
            filled=False,
            draw_delivered=False,
            reached_1r=False,
            mfe_r=0.2,
            mae_r=0.1,
        ),
    }

    result = index.query(
        _query(20, [1.0, 0.0, 0.0, 0.0]),
        k=5,
        ensemble=_ensemble([0.49, 0.50, 0.51]),
        frozen_outcomes=outcomes,
    )

    distribution = result.frozen_outcome_distribution
    assert distribution["independent_episode_count"] == 5
    assert distribution["resolved_episode_count"] == 2
    assert distribution["first_terminal"] == {"deadline": 1, "target": 1}
    assert distribution["mfe_r"]["mean"] == pytest.approx(0.85)
    assert all(not hasattr(item, "frozen_outcome") for item in result.neighbours)


def test_outcome_resolved_after_query_clock_is_not_available_to_prior() -> None:
    records = _close_cases(5)
    index = EpisodeCaseIndex.from_mappings(records, embedding_dim=DIM)
    query = _query(20, [1.0, 0.0, 0.0, 0.0])
    outcomes = {
        "case:episode:0": _bound_outcome(
            records[0],
            resolved_at=BASE + timedelta(minutes=21),
            first_event="target",
            mfe_r=2.0,
        )
    }

    result = index.query(
        query,
        k=5,
        ensemble=_ensemble([0.49, 0.50, 0.51]),
        frozen_outcomes=outcomes,
    )

    distribution = result.frozen_outcome_distribution
    assert distribution["resolved_episode_count"] == 0
    assert distribution["unavailable_future_outcome_count"] == 1
    assert distribution["first_terminal"] == {}


def test_checkpoint_is_schema_bound_pickle_free_and_drops_outcomes(
    tmp_path: Path,
) -> None:
    records = _close_cases(5)
    index = EpisodeCaseIndex.from_mappings(records, embedding_dim=DIM)
    path = index.save_checkpoint(tmp_path / "case-index.npz")

    loaded = EpisodeCaseIndex.load_checkpoint(path)

    assert len(loaded.records) == 5
    assert np.allclose(loaded.vectors, index.vectors, atol=1e-6)
    assert all(not hasattr(record, "frozen_outcome") for record in loaded.records)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(bytes(archive["manifest"]).decode("utf-8"))
        assert manifest["schema_version"] == CASE_RETRIEVAL_SCHEMA_VERSION
        assert manifest["embedding_checkpoint_ids"] == [
            EMBEDDING_CHECKPOINT_ID
        ]
        assert manifest["outcome_in_index_vector"] is False
        assert manifest["frozen_outcomes_persisted"] is False


def test_index_checkpoint_binds_finalized_case_library_lineage(
    tmp_path: Path,
) -> None:
    lineage = {
        "case_input_manifest_sha256": "1" * 64,
        "case_library_manifest_sha256": "2" * 64,
        "decision_stage": "plan_formed",
        "selection_contract": (
            "first_online_stage_occurrence_by_revision_index_v1"
        ),
    }
    index = EpisodeCaseIndex.from_mappings(
        _close_cases(5),
        embedding_dim=DIM,
        artifact_lineage=lineage,
    )
    loaded = EpisodeCaseIndex.load_checkpoint(
        index.save_checkpoint(tmp_path / "lineage-index.npz")
    )
    assert loaded.artifact_lineage == lineage
    query = _query(20, [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(CaseRetrievalError, match="omits the index"):
        loaded.query(query)
    with pytest.raises(CaseRetrievalError, match="lineages differ"):
        loaded.query(
            query,
            artifact_lineage={
                **lineage,
                "case_library_manifest_sha256": "3" * 64,
            },
        )


def test_index_strictly_rejects_inline_outcome_even_on_ignored_revision() -> None:
    leaked = _case(
        "episode:inline",
        1,
        [1.0, 0.0, 0.0, 0.0],
        embedding_clock="case_revision",
        outcome={"first_event": "target"},
    )
    with pytest.raises(CaseRetrievalError, match="inline outcomes are forbidden"):
        EpisodeCaseIndex.from_mappings([leaked], embedding_dim=DIM)

    top_level = _case(
        "episode:inline-columns", 1, [1.0, 0.0, 0.0, 0.0]
    )
    top_level["frozen_outcome"] = {}
    top_level["mfe_R"] = 2.0
    with pytest.raises(CaseRetrievalError, match="inline outcome columns"):
        EpisodeCaseIndex.from_mappings([top_level], embedding_dim=DIM)


def test_external_outcome_identity_must_match_selected_episode() -> None:
    index = EpisodeCaseIndex.from_mappings(_close_cases(5), embedding_dim=DIM)
    with pytest.raises(CaseRetrievalError, match="identity differs"):
        index.query(
            _query(20, [1.0, 0.0, 0.0, 0.0]),
            k=5,
            ensemble=_ensemble([0.49, 0.50, 0.51]),
            frozen_outcomes={
                "case:episode:0": {
                    "market_epoch_id": "epoch:other",
                    "entry_episode_id": "episode:0",
                    "resolved_at": BASE + timedelta(minutes=10),
                    "first_event": "target",
                }
            },
        )

    with pytest.raises(CaseRetrievalError, match="identity differs"):
        index.query(
            _query(20, [1.0, 0.0, 0.0, 0.0]),
            k=5,
            ensemble=_ensemble([0.49, 0.50, 0.51]),
            frozen_outcomes={
                "case:episode:0": {
                    "resolved_at": BASE + timedelta(minutes=10),
                    "first_event": "target",
                }
            },
        )


def test_output_has_no_trade_recommendation_or_trade_action() -> None:
    index = EpisodeCaseIndex.from_mappings(_close_cases(5), embedding_dim=DIM)
    result = index.query(
        _query(20, [1.0, 0.0, 0.0, 0.0]),
        ensemble=_ensemble([0.49, 0.50, 0.51]),
    ).as_dict()
    encoded = json.dumps(result, sort_keys=True).lower()

    assert '"buy"' not in encoded
    assert '"sell"' not in encoded
    assert "suggested_trade" not in encoded
    assert "trade_action" not in encoded
    assert result["action_authority"] == "none"
    assert result["ood"]["policy"] == "continue_evaluation"


def test_cli_build_and_query_keep_outcomes_in_separate_artifact(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.json"
    query_path = tmp_path / "query.json"
    ensemble_path = tmp_path / "ensemble.json"
    outcomes_path = tmp_path / "outcomes.json"
    checkpoint = tmp_path / "index.npz"
    cases.write_text(json.dumps(_close_cases(5), default=str), encoding="utf-8")
    cases_manifest, cases_manifest_sha = _write_export_manifest(
        cases,
        schema="smc-decision-time-embeddings-v1",
        records=5,
        checkpoint_ids=(EMBEDDING_CHECKPOINT_ID,),
    )
    query_record = _case(
        "episode:cli-query", 20, [1.0, 0.0, 0.0, 0.0]
    )
    query_record.pop("frozen_outcome")
    other_query_record = _case(
        "episode:other-query", 19, [1.0, 0.0, 0.0, 0.0]
    )
    other_query_record.pop("frozen_outcome")
    query_path.write_text(
        json.dumps([query_record, other_query_record], default=str),
        encoding="utf-8",
    )
    query_manifest, query_manifest_sha = _write_export_manifest(
        query_path,
        schema="smc-decision-time-embeddings-v1",
        records=2,
        checkpoint_ids=(EMBEDDING_CHECKPOINT_ID,),
    )
    bound_query = EpisodeEmbeddingQuery.from_mapping(
        query_record, embedding_dim=DIM
    )
    query_members = _ensemble([0.49, 0.50, 0.51], query=bound_query)
    other_members = _ensemble(
        [0.48, 0.50, 0.52],
        query=_query(19, [1.0, 0.0, 0.0, 0.0], episode="episode:other-query"),
    )
    all_members = [*query_members, *other_members]
    ensemble_path.write_text(
        json.dumps(
            [
                    {
                        "member_id": member.member_id,
                        "checkpoint_id": member.checkpoint_id,
                        "model_version": member.model_version,
                        "case_id": member.case_id,
                        "revision_id": member.revision_id,
                        "entry_episode_id": member.entry_episode_id,
                        "decision_at": member.decision_at.isoformat(),
                        "feature_max_at": member.feature_max_at.isoformat(),
                        "outcome_fields_used": member.outcome_fields_used,
                        "input_protocol": member.input_protocol,
                        "head_predictions": member.head_predictions,
                    }
                for member in all_members
            ]
        ),
        encoding="utf-8",
    )
    ensemble_manifest, ensemble_manifest_sha = _write_export_manifest(
        ensemble_path,
        schema="smc-decision-time-self-supervised-heads-v1",
        records=len(all_members),
        checkpoint_ids=tuple({member.checkpoint_id for member in all_members}),
        head_schema=OUTCOME_BLIND_HEAD_WIDTHS,
    )
    outcomes_path.write_text(
        json.dumps(
            [
                {
                    "outcome_id": "outcome:cli",
                    "case_id": "case:episode:0",
                    "market_epoch_id": "epoch:1",
                    "context_thesis_id": "context:episode:0",
                    "entry_episode_id": "episode:0",
                    "resolved_at": (
                        BASE + timedelta(minutes=10)
                    ).isoformat(),
                    "first_event": "target",
                    "mfe_R": 1.25,
                    "hit_1R": True,
                    "filled": True,
                    "expired": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts/query_causal_cases.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "build",
            "--cases",
            str(cases),
            "--case-manifest",
            str(cases_manifest),
            "--case-manifest-sha",
            cases_manifest_sha,
            "--output",
            str(checkpoint),
            "--embedding-dim",
            str(DIM),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "query",
            "--index",
            str(checkpoint),
            "--query",
            str(query_path),
            "--query-case-id",
            str(query_record["case_id"]),
            "--query-revision-id",
            str(query_record["revision_id"]),
            "--query-manifest",
            str(query_manifest),
            "--query-manifest-sha",
            query_manifest_sha,
            "--ensemble",
            str(ensemble_path),
            "--ensemble-manifest",
            str(ensemble_manifest),
            "--ensemble-manifest-sha",
            ensemble_manifest_sha,
            "--outcomes",
            str(outcomes_path),
            "--k",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["sufficient_neighbours"] is True
    assert payload["ood"]["policy"] == "continue_evaluation"
    assert payload["frozen_outcome_distribution"]["first_terminal"] == {
        "target": 1
    }
    assert payload["frozen_outcome_distribution"]["expire"] == {"false": 1}
    assert payload["frozen_outcome_distribution"]["mfe_r"]["mean"] == 1.25

    mismatched_manifest = json.loads(ensemble_manifest.read_text(encoding="utf-8"))
    mismatched_manifest["case_input_manifest_sha256"] = "1" * 64
    mismatched_manifest["case_library_manifest_sha256"] = "2" * 64
    ensemble_manifest.write_text(
        json.dumps(mismatched_manifest, sort_keys=True), encoding="utf-8"
    )
    mismatched_sha = hashlib.sha256(ensemble_manifest.read_bytes()).hexdigest()
    mismatched_command = list(completed.args)
    manifest_sha_index = mismatched_command.index("--ensemble-manifest-sha") + 1
    mismatched_command[manifest_sha_index] = mismatched_sha
    refused = subprocess.run(
        mismatched_command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "artifact lineages differ" in refused.stderr


def test_market_episode_selector_keeps_first_of_all_six_material_kinds() -> None:
    revisions = [
        _market_episode_case(
            "selector",
            1,
            [1.0, 0.0, 0.0, 0.0],
            transition_kinds=("episode_created", "zone_registered"),
            revision_index=1,
        ),
        *[
            _market_episode_case(
                "selector",
                index,
                [1.0, 0.0, 0.0, 0.0],
                transition_kinds=(kind,),
                revision_index=index,
            )
            for index, kind in enumerate(
                ("first_pullback", "trigger", "successful_pulse", "terminal"),
                start=2,
            )
        ],
        _market_episode_case(
            "selector",
            9,
            [0.0, 1.0, 0.0, 0.0],
            transition_kinds=("trigger",),
            revision_index=9,
        ),
    ]
    assert all("material_kind" not in revision for revision in revisions)
    index = MarketEpisodeCaseIndex.from_mappings(
        revisions,
        artifact_lineage=MARKET_EPISODE_LINEAGE,
        embedding_dim=DIM,
    )

    assert tuple(record.material_kind for record in index.records) == (
        MARKET_EPISODE_MATERIAL_KINDS
    )
    assert next(
        record.revision_index
        for record in index.records
        if record.material_kind == "trigger"
    ) == 3

    revisions[0]["market_episode_id"] = "market-episode:not-canonical"
    with pytest.raises(CaseRetrievalError, match="not canonical"):
        MarketEpisodeCaseIndex.from_mappings(
            revisions,
            artifact_lineage=MARKET_EPISODE_LINEAGE,
            embedding_dim=DIM,
        )


def test_market_episode_selector_consumes_explicit_material_grains_once() -> None:
    first_pullback = _market_episode_case(
        "simultaneous",
        1,
        [1.0, 0.0, 0.0, 0.0],
        transition_kinds=("first_pullback", "terminal"),
        revision_index=1,
    )
    first_pullback["material_kind"] = "first_pullback"
    terminal = {
        **first_pullback,
        "material_kind": "terminal",
        "decision_embedding": [1.0, 1e-8, 0.0, 0.0],
    }

    index = MarketEpisodeCaseIndex.from_mappings(
        [first_pullback, terminal],
        artifact_lineage=MARKET_EPISODE_LINEAGE,
        embedding_dim=DIM,
    )

    assert tuple(record.material_kind for record in index.records) == (
        "first_pullback",
        "terminal",
    )
    assert index.records[0].decision_embedding != index.records[1].decision_embedding


def test_market_episode_selector_rejects_unbound_explicit_material_kind() -> None:
    revision = _market_episode_case(
        "explicit-mismatch",
        1,
        [1.0, 0.0, 0.0, 0.0],
        transition_kinds=("first_pullback", "terminal"),
    )
    revision["material_kind"] = "trigger"

    with pytest.raises(CaseRetrievalError, match="explicit material kind"):
        MarketEpisodeCaseIndex.from_mappings(
            [revision],
            artifact_lineage=MARKET_EPISODE_LINEAGE,
            embedding_dim=DIM,
        )


def test_market_episode_direct_constructor_requires_multi_run_source_contract() -> None:
    first_lineage = {
        **MARKET_EPISODE_LINEAGE,
        "run_manifest_sha256": "1" * 64,
    }
    second_lineage = {
        **MARKET_EPISODE_LINEAGE,
        "stream_manifest_sha256": "4" * 64,
        "run_manifest_sha256": "2" * 64,
    }
    records = tuple(
        MarketEpisodeEmbeddingRecord.from_mapping(
            _market_episode_case(
                f"direct-{index}",
                index,
                [1.0, index * 0.01, 0.0, 0.0],
                run_manifest_sha256=str(index) * 64,
            ),
            material_kind="trigger",
            embedding_dim=DIM,
        )
        for index in (1, 2)
    )

    with pytest.raises(CaseRetrievalError, match="requires a dataset contract"):
        MarketEpisodeCaseIndex(
            records,
            artifact_lineages=(first_lineage, second_lineage),
            embedding_dim=DIM,
        )


def test_market_episode_direct_constructor_rejects_duplicate_and_cross_split() -> None:
    first = MarketEpisodeEmbeddingRecord.from_mapping(
        _market_episode_case("direct", 1, [1.0, 0.0, 0.0, 0.0]),
        material_kind="trigger",
        embedding_dim=DIM,
    )
    later = MarketEpisodeEmbeddingRecord.from_mapping(
        _market_episode_case(
            "direct",
            2,
            [1.0, 0.01, 0.0, 0.0],
            revision_index=2,
        ),
        material_kind="trigger",
        embedding_dim=DIM,
    )
    with pytest.raises(CaseRetrievalError, match="repeats a selected"):
        MarketEpisodeCaseIndex(
            (first, later),
            artifact_lineages=(MARKET_EPISODE_LINEAGE,),
            embedding_dim=DIM,
        )

    other_split = MarketEpisodeEmbeddingRecord.from_mapping(
        _market_episode_case(
            "direct",
            2,
            [1.0, 0.01, 0.0, 0.0],
            revision_index=2,
            split="validation",
        ),
        material_kind="trigger",
        embedding_dim=DIM,
    )
    with pytest.raises(CaseRetrievalError, match="shared across splits"):
        MarketEpisodeCaseIndex(
            (first, other_split),
            artifact_lineages=(MARKET_EPISODE_LINEAGE,),
            embedding_dim=DIM,
        )


def test_market_episode_query_is_strictly_prior_and_outcome_free() -> None:
    eligible = [
        _market_episode_case(
            f"prior-{index}",
            index,
            [1.0, index * 0.01, 0.0, 0.0],
        )
        for index in range(1, 6)
    ]
    records = [
        *eligible,
        _market_episode_case("query", 6, [1.0, 0.0, 0.0, 0.0]),
        _market_episode_case("same-clock", 20, [1.0, 0.0, 0.0, 0.0]),
        _market_episode_case("future", 21, [1.0, 0.0, 0.0, 0.0]),
        _market_episode_case(
            "other-epoch",
            1,
            [1.0, 0.0, 0.0, 0.0],
            epoch="epoch:other",
        ),
        _market_episode_case(
            "other-kind",
            1,
            [1.0, 0.0, 0.0, 0.0],
            transition_kinds=("terminal",),
        ),
    ]
    index = MarketEpisodeCaseIndex.from_mappings(
        records,
        artifact_lineage=MARKET_EPISODE_LINEAGE,
        embedding_dim=DIM,
    )
    query = _market_episode_query(20, [1.0, 0.0, 0.0, 0.0])
    result = index.query(
        query,
        artifact_lineage=MARKET_EPISODE_LINEAGE,
        ensemble=_market_episode_ensemble([0.49, 0.50, 0.51], query=query),
    )

    assert result.ood.policy is RetrievalPolicy.CONTINUE_EVALUATION
    assert result.ood.eligible_neighbours == 5
    assert {item["market_episode_id"] for item in result.neighbours} == {
        item["market_episode_id"] for item in eligible
    }
    assert set(result.ood.head_disagreement) == set(
        MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS
    )
    assert "frozen_outcome_distribution" not in result.as_dict()

    no_ensemble = index.query(
        query,
        artifact_lineage=MARKET_EPISODE_LINEAGE,
    )
    assert no_ensemble.ood.policy is RetrievalPolicy.ABSTAIN
    assert "deep_ensemble_unavailable" in no_ensemble.ood.reasons


def test_market_episode_query_preserves_exact_neutral_preprocessing_contract() -> None:
    raw = _market_episode_case(
        "real-mapping",
        20,
        [1.0, 0.0, 0.0, 0.0],
    )
    query = MarketEpisodeEmbeddingQuery.from_mapping(
        raw,
        material_kind="trigger",
        embedding_dim=DIM,
    )
    identity = neutral_direct_source_preprocessing_identity()

    assert query.embedding_input_protocol == NEUTRAL_INFERENCE_INPUT_PROTOCOL
    assert (
        query.neutral_preprocessing_version
        == identity["protocol"]["protocol_version"]
    )
    assert query.neutral_preprocessing_sha256 == identity["sha256"]

    wrong_protocol = dict(raw)
    wrong_protocol["embedding_input_protocol"] = INFERENCE_INPUT_PROTOCOL
    with pytest.raises(CaseRetrievalError, match="preprocessing contract"):
        MarketEpisodeEmbeddingQuery.from_mapping(
            wrong_protocol,
            material_kind="trigger",
            embedding_dim=DIM,
        )


def test_market_episode_query_direct_construction_fails_closed() -> None:
    query = _market_episode_query(20, [1.0, 0.0, 0.0, 0.0])
    direct = dict(query.__dict__)

    missing = dict(direct)
    missing.pop("neutral_preprocessing_sha256")
    with pytest.raises(TypeError, match="neutral_preprocessing_sha256"):
        MarketEpisodeEmbeddingQuery(**missing)

    tampered = {
        **direct,
        "neutral_preprocessing_sha256": "0" * 64,
    }
    with pytest.raises(CaseRetrievalError, match="preprocessing contract"):
        MarketEpisodeEmbeddingQuery(**tampered)


def test_market_episode_index_rechecks_query_preprocessing_contract() -> None:
    index = MarketEpisodeCaseIndex.from_mappings(
        [_market_episode_case("prior", 1, [1.0, 0.0, 0.0, 0.0])],
        artifact_lineage=MARKET_EPISODE_LINEAGE,
        embedding_dim=DIM,
    )
    query = _market_episode_query(20, [1.0, 0.0, 0.0, 0.0])
    object.__setattr__(query, "neutral_preprocessing_sha256", "0" * 64)

    with pytest.raises(CaseRetrievalError, match="preprocessing contract"):
        index.query(query, artifact_lineage=MARKET_EPISODE_LINEAGE)


def test_market_episode_cross_run_self_neighbour_is_rejected() -> None:
    reference = _market_episode_case(
        "shared-local-identity",
        1,
        [1.0, 0.0, 0.0, 0.0],
        run_manifest_sha256="1" * 64,
    )
    reference_lineage = {
        **MARKET_EPISODE_LINEAGE,
        "run_manifest_sha256": "1" * 64,
    }
    second_reference = _market_episode_case(
        "shared-local-identity",
        24 * 60 + 1,
        [1.0, 0.01, 0.0, 0.0],
        run_manifest_sha256="2" * 64,
    )
    second_reference_lineage = {
        **MARKET_EPISODE_LINEAGE,
        "stream_manifest_sha256": "4" * 64,
        "run_manifest_sha256": "2" * 64,
    }
    independent_reference = _market_episode_case(
        "independent-physical-episode",
        24 * 60 + 2,
        [1.0, 0.02, 0.0, 0.0],
        run_manifest_sha256="2" * 64,
    )
    query_lineage = {
        **MARKET_EPISODE_LINEAGE,
        "stream_manifest_sha256": "5" * 64,
        "run_manifest_sha256": "3" * 64,
    }
    index = MarketEpisodeCaseIndex.from_artifacts(
        (
            {
                "records": [reference],
                "artifact_lineage": reference_lineage,
                "dataset_contract": MARKET_EPISODE_DATASET_CONTRACT,
            },
            {
                "records": [second_reference, independent_reference],
                "artifact_lineage": second_reference_lineage,
                "dataset_contract": MARKET_EPISODE_DATASET_CONTRACT,
            },
        ),
        embedding_dim=DIM,
    )
    query = _market_episode_query(
        2 * 24 * 60 + 1,
        [1.0, 0.0, 0.0, 0.0],
        episode="shared-local-identity",
        split="validation",
        run_manifest_sha256="3" * 64,
    )
    result = index.query(
        query,
        artifact_lineage=query_lineage,
        dataset_contract=MARKET_EPISODE_DATASET_CONTRACT,
        require_different_calendar_date=True,
        thresholds=OODThresholds(minimum_neighbours=1),
        ensemble=_market_episode_ensemble([0.49, 0.50, 0.51], query=query),
    )

    assert result.ood.eligible_neighbours == 1
    assert {row["market_episode_id"] for row in result.neighbours} == {
        independent_reference["market_episode_id"]
    }
    assert result.as_dict()["query_run_manifest_sha256"] == "3" * 64
    assert (
        result.neighbours[0]["market_episode_id"]
        != result.query_market_episode_id
    )

    with pytest.raises(CaseRetrievalError, match="dataset contracts differ"):
        index.query(
            query,
            artifact_lineage=query_lineage,
            dataset_contract={
                **MARKET_EPISODE_DATASET_CONTRACT,
                "source_sha256": "4" * 64,
            },
        )


def test_case_protocols_cannot_be_swapped_or_receive_neutral_outcomes() -> None:
    neutral = _market_episode_case("neutral", 1, [1.0, 0.0, 0.0, 0.0])
    assert MARKET_EPISODE_RETRIEVAL_PROTOCOL["outcome_channel"] == "forbidden"
    with pytest.raises(CaseRetrievalError):
        EpisodeCaseIndex.from_mappings([neutral], embedding_dim=DIM)

    causal = _case("causal", 1, [1.0, 0.0, 0.0, 0.0])
    causal.pop("frozen_outcome")
    with pytest.raises(CaseRetrievalError):
        MarketEpisodeCaseIndex.from_mappings(
            [causal],
            artifact_lineage=MARKET_EPISODE_LINEAGE,
            embedding_dim=DIM,
        )

    injected = dict(neutral)
    injected["outcome"] = {}
    with pytest.raises(CaseRetrievalError, match="not outcome-blind"):
        MarketEpisodeCaseIndex.from_mappings(
            [injected],
            artifact_lineage=MARKET_EPISODE_LINEAGE,
            embedding_dim=DIM,
        )


def test_both_case_indexes_expose_read_only_vector_views() -> None:
    causal = EpisodeCaseIndex.from_mappings(
        [_case("causal", 1, [1.0, 0.0, 0.0, 0.0])],
        embedding_dim=DIM,
    )
    neutral = MarketEpisodeCaseIndex.from_mappings(
        [_market_episode_case("neutral", 1, [1.0, 0.0, 0.0, 0.0])],
        artifact_lineage=MARKET_EPISODE_LINEAGE,
        embedding_dim=DIM,
    )

    assert causal.vectors.flags.writeable is False
    assert neutral.vectors.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        causal.vectors[0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        neutral.vectors[0, 0] = 0.0


def test_market_episode_lineage_and_active_heads_fail_closed() -> None:
    case = _market_episode_case("prior", 1, [1.0, 0.0, 0.0, 0.0])
    extra_lineage = {
        **MARKET_EPISODE_LINEAGE,
        "case_library_sha256": "d" * 64,
    }
    with pytest.raises(CaseRetrievalError, match="lineage keys"):
        MarketEpisodeCaseIndex.from_mappings(
            [case],
            artifact_lineage=extra_lineage,
            embedding_dim=DIM,
        )

    index = MarketEpisodeCaseIndex.from_mappings(
        [case],
        artifact_lineage=MARKET_EPISODE_LINEAGE,
        embedding_dim=DIM,
    )
    query = _market_episode_query(20, [1.0, 0.0, 0.0, 0.0])
    ensemble = _market_episode_ensemble([0.49, 0.50, 0.51], query=query)
    wrong_epoch = [dict(row) for row in ensemble]
    wrong_epoch[0]["market_epoch_id"] = "epoch:other"
    with pytest.raises(CaseRetrievalError, match="ensemble binding"):
        index.query(
            query,
            artifact_lineage=MARKET_EPISODE_LINEAGE,
            ensemble=wrong_epoch,
        )

    missing_run = [dict(row) for row in ensemble]
    missing_run[0].pop("run_manifest_sha256")
    with pytest.raises(CaseRetrievalError, match="ensemble run lineage"):
        index.query(
            query,
            artifact_lineage=MARKET_EPISODE_LINEAGE,
            ensemble=missing_run,
        )

    ensemble[0]["head_predictions"] = {
        **ensemble[0]["head_predictions"],
        "masked_reconstruction": (0.5, 0.5),
    }
    with pytest.raises(CaseRetrievalError, match="active heads"):
        index.query(
            query,
            artifact_lineage=MARKET_EPISODE_LINEAGE,
            ensemble=ensemble,
        )


def test_threshold_contract_rejects_reversed_or_single_model_configuration() -> None:
    with pytest.raises(CaseRetrievalError, match="at least two"):
        OODThresholds(minimum_ensemble_members=1)
    with pytest.raises(CaseRetrievalError, match="reversed"):
        OODThresholds(
            maximum_disagreement_continue=0.5,
            maximum_disagreement_abstain=0.1,
        )
