"""Outcome-blind episode retrieval and novelty/OOD diagnostics.

The index consumes one 128-dimensional representation frozen at an
``EntryEpisode`` decision clock.  Frozen future outcomes are carried in a
separate metadata channel and are used only after neighbours have been
selected.  They can therefore change an empirical path distribution, but can
never change an indexed vector, distance, or neighbour ordering.

This module is deliberately downstream of the causal case library and the
representation model.  It has no execution, Decision, Risk, playbook, or Eye
authority.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .market_representation import (
    INFERENCE_INPUT_PROTOCOL,
    OUTCOME_BLIND_HEAD_WIDTHS,
)
from .scene_graph import market_episode_id as canonical_market_episode_id


CASE_RETRIEVAL_SCHEMA_VERSION = 3
CASE_RETRIEVAL_PROTOCOL_VERSION = "episode-case-retrieval-1.2.0"
DEFAULT_MARKET_EMBEDDING_DIM = 128
CASE_REVISION_STAGES = frozenset(
    {
        "context_formed",
        "episode_created",
        "context_changed",
        "zone_registered",
        "first_pullback",
        "trigger",
        "plan_formed",
        "terminal",
    }
)
FIRST_STAGE_SELECTION_CONTRACT = (
    "first_online_stage_occurrence_by_revision_index_v1"
)
_ARTIFACT_LINEAGE_KEYS = frozenset(
    {
        "case_input_manifest_sha256",
        "case_library_manifest_sha256",
        "decision_stage",
        "selection_contract",
    }
)

MARKET_EPISODE_MATERIAL_KINDS = (
    "episode_created",
    "zone_registered",
    "first_pullback",
    "trigger",
    "successful_pulse",
    "terminal",
)
MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT = (
    "first_online_market_episode_material_kind_by_revision_index_v1"
)
MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS: Mapping[str, int] = {
    name: OUTCOME_BLIND_HEAD_WIDTHS[name]
    for name in ("next_lifecycle", "scale_direction_alignment")
}
MARKET_EPISODE_DATASET_CONTRACT_KEYS = frozenset(
    {
        "source_sha256",
        "model_config_sha256",
        "market_case_protocol",
        "representation_feature_schema_version",
        "embedding_input_protocol",
        "selection_contract",
        "embedding_model_version",
        "embedding_checkpoint_id",
        "calendar_timezone",
    }
)
_MARKET_EPISODE_MATERIAL_KIND_ORDER = {
    name: index for index, name in enumerate(MARKET_EPISODE_MATERIAL_KINDS)
}

CASE_RETRIEVAL_PROTOCOL: Mapping[str, Any] = {
    "schema_version": CASE_RETRIEVAL_SCHEMA_VERSION,
    "protocol_version": CASE_RETRIEVAL_PROTOCOL_VERSION,
    "case_grain": "one_independent_entry_episode",
    "embedding_clock": "decision_time",
    "revision_comparability": (
        "exact_same_revision_stage_and_first_causal_stage_occurrence"
    ),
    "distance": "cosine_on_l2_normalized_market_embedding",
    "embedding_space_binding": "single_encoder_checkpoint_content_sha256",
    "artifact_lineage_binding": (
        "exact_case_input_and_finalized_library_manifest_sha256"
    ),
    "embedding_input_protocol": INFERENCE_INPUT_PROTOCOL,
    "ensemble_head_binding": (
        "exact_case_revision_episode_model_decision_and_feature_clocks_"
        "with_fixed_outcome_blind_head_schema"
    ),
    "ensemble_head_schema": dict(OUTCOME_BLIND_HEAD_WIDTHS),
    "outcome_in_index_vector": False,
    "outcome_availability": "resolved_at_no_later_than_query_decision_at",
    "neighbour_guards": (
        "explicit_prior_reference_split",
        "same_market_epoch",
        "strictly_prior_decision_clock",
        "different_entry_episode",
    ),
    "policy_outputs": (
        "continue_evaluation",
        "increase_uncertainty",
        "abstain",
    ),
    "action_authority": "none_empirical_prior_only",
}


class CaseRetrievalError(ValueError):
    """Raised when an index or query would violate its causal contract."""


def _normalise_artifact_lineage(
    value: Mapping[str, Any] | None,
) -> dict[str, str | None]:
    if value is None:
        return {
            "case_input_manifest_sha256": None,
            "case_library_manifest_sha256": None,
            "decision_stage": None,
            "selection_contract": None,
        }
    if set(value) != set(_ARTIFACT_LINEAGE_KEYS):
        raise CaseRetrievalError("embedding artifact lineage keys are invalid")
    input_sha = value.get("case_input_manifest_sha256")
    library_sha = value.get("case_library_manifest_sha256")
    if (input_sha is None) != (library_sha is None):
        raise CaseRetrievalError(
            "case input and finalized-library lineage must both be bound or absent"
        )
    normalised_hashes: list[str | None] = []
    for name, raw in (
        ("case input manifest", input_sha),
        ("case library manifest", library_sha),
    ):
        if raw is None:
            normalised_hashes.append(None)
            continue
        digest = str(raw).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise CaseRetrievalError(f"{name} lineage is not a SHA-256")
        normalised_hashes.append(digest)
    stage_raw = value.get("decision_stage")
    contract_raw = value.get("selection_contract")
    if stage_raw is None and contract_raw is None and input_sha is None:
        return {
            "case_input_manifest_sha256": None,
            "case_library_manifest_sha256": None,
            "decision_stage": None,
            "selection_contract": None,
        }
    stage = str(stage_raw or "").strip().lower()
    if stage not in CASE_REVISION_STAGES:
        raise CaseRetrievalError("embedding artifact decision stage is invalid")
    if contract_raw != FIRST_STAGE_SELECTION_CONTRACT:
        raise CaseRetrievalError("embedding artifact selection contract is invalid")
    return {
        "case_input_manifest_sha256": normalised_hashes[0],
        "case_library_manifest_sha256": normalised_hashes[1],
        "decision_stage": stage,
        "selection_contract": FIRST_STAGE_SELECTION_CONTRACT,
    }


def _normalise_market_episode_artifact_lineage(
    value: Mapping[str, Any] | None,
) -> dict[str, str]:
    keys = {
        "stream_manifest_sha256",
        "run_manifest_sha256",
        "selection_contract",
    }
    if value is None or set(value) != keys:
        raise CaseRetrievalError("MarketEpisode lineage keys are invalid")
    output: dict[str, str] = {}
    for name in ("stream_manifest_sha256", "run_manifest_sha256"):
        digest = str(value.get(name, "")).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise CaseRetrievalError(f"MarketEpisode {name} is not a SHA-256")
        output[name] = digest
    if (
        value.get("selection_contract")
        != MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT
    ):
        raise CaseRetrievalError("MarketEpisode selector contract is invalid")
    output["selection_contract"] = (
        MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT
    )
    return output


def normalise_market_episode_dataset_contract(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Canonical compatibility identity shared by independent time splits."""

    if value is None or set(value) != MARKET_EPISODE_DATASET_CONTRACT_KEYS:
        raise CaseRetrievalError("MarketEpisode dataset contract keys are invalid")
    output: dict[str, Any] = {}
    for name in (
        "source_sha256",
        "model_config_sha256",
        "embedding_checkpoint_id",
    ):
        digest = str(value.get(name, "")).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise CaseRetrievalError(f"MarketEpisode {name} is not a SHA-256")
        output[name] = digest
    protocol = value.get("market_case_protocol")
    if not isinstance(protocol, Mapping) or not protocol:
        raise CaseRetrievalError("MarketEpisode protocol contract is invalid")
    output["market_case_protocol"] = _jsonable(protocol)
    for name in (
        "representation_feature_schema_version",
        "embedding_model_version",
        "calendar_timezone",
    ):
        output[name] = _nonempty(value.get(name), name)
    if value.get("embedding_input_protocol") != INFERENCE_INPUT_PROTOCOL:
        raise CaseRetrievalError("MarketEpisode input protocol is invalid")
    output["embedding_input_protocol"] = INFERENCE_INPUT_PROTOCOL
    if (
        value.get("selection_contract")
        != MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT
    ):
        raise CaseRetrievalError("MarketEpisode selector contract is invalid")
    output["selection_contract"] = (
        MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT
    )
    return output


class RetrievalPolicy(str, Enum):
    """The only permitted downstream routing suggestions.

    These values change confidence routing only.  They are not trading
    actions and do not encode a direction.
    """

    CONTINUE_EVALUATION = "continue_evaluation"
    INCREASE_UNCERTAINTY = "increase_uncertainty"
    ABSTAIN = "abstain"


_FORBIDDEN_EMBEDDING_FIELD_TOKENS = frozenset(
    {
        "future",
        "future_path",
        "frozen_outcome",
        "outcome",
        "outcome_value",
        "first_reached",
        "first_event",
        "first_terminal",
        "target_first",
        "invalidation_first",
        "deadline_first",
        "same_bar_collision",
        "target_hit",
        "invalidation_hit",
        "deadline_hit",
        "mfe",
        "mfe_r",
        "mfe_points",
        "mae",
        "mae_r",
        "mae_points",
        "reached_0_5r",
        "hit_0_5r",
        "reached_1r",
        "hit_1r",
        "reached_2r",
        "hit_2r",
        "draw_delivered",
        "draw_delivery",
        "filled",
        "expired",
        "censored",
        "resolved_at",
        "outcome_id",
        "source_shadow_candidate_id",
        "terminal_reason",
        "resolution",
    }
)

_EVALUATION_SPLITS = frozenset(
    {
        "validation",
        "test",
        "brain_validation",
        "brain_calibration_trial",
        "rolling_oof",
        "sealed_holdout",
        "holdout",
    }
)
_DEFAULT_REFERENCE_SPLITS: Mapping[str, tuple[str, ...]] = {
    "validation": ("train",),
    "test": ("train",),
    "brain_validation": ("calibration",),
    "brain_calibration_trial": ("calibration",),
    "rolling_oof": ("calibration",),
    "sealed_holdout": ("calibration",),
    "holdout": ("train",),
    "production": ("historical", "reference"),
    "live": ("historical", "reference"),
}


def _nonempty(value: Any, name: str) -> str:
    output = str(value).strip()
    if not output:
        raise CaseRetrievalError(f"{name} must be non-empty")
    return output


def _aware_utc(value: Any, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise CaseRetrievalError(f"{name} is not a timestamp") from exc
    if timestamp.tzinfo is None:
        raise CaseRetrievalError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _jsonable(value: Any) -> Any:
    """Return a stable JSON value without accepting non-finite numbers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        output = float(value)
        if not math.isfinite(output):
            raise CaseRetrievalError("metadata contains a non-finite number")
        return output
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return _aware_utc(value, "metadata timestamp").isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise CaseRetrievalError(
        f"metadata value of type {type(value).__name__} is not JSON-safe"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalise_field_name(name: Any) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def _looks_like_outcome_field(name: Any) -> bool:
    normalised = _normalise_field_name(name)
    return normalised in _FORBIDDEN_EMBEDDING_FIELD_TOKENS or any(
        normalised.startswith(f"{token}_")
        or normalised.endswith(f"_{token}")
        for token in _FORBIDDEN_EMBEDDING_FIELD_TOKENS
    )


def _mapping_leaf_paths(
    value: Any,
    *,
    prefix: str = "",
) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _mapping_leaf_paths(item, prefix=path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _mapping_leaf_paths(item, prefix=f"{prefix}[{index}]")
    else:
        yield prefix


def _validate_embedding_inputs(record: Mapping[str, Any]) -> tuple[str, ...]:
    raw_names = record.get("embedding_feature_names", ())
    if raw_names is None:
        raw_names = ()
    if isinstance(raw_names, (str, bytes)) or not isinstance(raw_names, Sequence):
        raise CaseRetrievalError("embedding_feature_names must be a sequence")
    names = tuple(_nonempty(item, "embedding feature name") for item in raw_names)
    inspected = list(names)
    if "embedding_inputs" in record:
        inspected.extend(_mapping_leaf_paths(record["embedding_inputs"]))
    forbidden = sorted(
        name for name in inspected if any(
            _looks_like_outcome_field(part)
            for part in str(name).replace("[", ".").split(".")
            if part and not part.rstrip("]").isdigit()
        )
    )
    if forbidden:
        raise CaseRetrievalError(
            "outcome/future fields are forbidden from embedding inputs: "
            f"{forbidden}"
        )
    return names


def _inline_outcome_columns(record: Mapping[str, Any]) -> tuple[str, ...]:
    allowed_contract_markers = {
        "outcome_fields_used",
        "frozen_outcome",
        "outcome",
        "observed_terminal_reason",
    }
    return tuple(
        sorted(
            str(key)
            for key in record
            if str(key) not in allowed_contract_markers
            and _looks_like_outcome_field(key)
        )
    )


def _normalise_embedding(
    values: Sequence[float] | np.ndarray,
    *,
    expected_dim: int,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CaseRetrievalError("decision_embedding must be numeric") from exc
    if vector.ndim != 1 or vector.shape[0] != expected_dim:
        raise CaseRetrievalError(
            "decision_embedding has the wrong shape: "
            f"expected ({expected_dim},), got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise CaseRetrievalError("decision_embedding must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise CaseRetrievalError("decision_embedding must have non-zero norm")
    return vector / norm


def _first_present(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise CaseRetrievalError(f"missing required field {names[0]}")


def _decision_record(record: Mapping[str, Any]) -> bool:
    """Identify the one row that owns an episode's decision embedding.

    Explicit ``embedding_clock``/``embedding_role`` is preferred.  For a
    compact already-materialised record, the dedicated ``decision_embedding``
    field is also sufficient.  Revision rows with a different ``revision_at``
    are never silently promoted to decision rows.
    """

    role = record.get("embedding_clock", record.get("embedding_role"))
    if role is not None:
        return str(role).strip().lower() in {"decision", "decision_time"}
    if "decision_embedding" not in record:
        return False
    if "revision_at" in record and "decision_at" in record:
        return _aware_utc(record["revision_at"], "revision_at") == _aware_utc(
            record["decision_at"], "decision_at"
        )
    return True


def _market_episode_direction(value: Any) -> str:
    raw = value.value if isinstance(value, Enum) else value
    token = str(int(raw) if isinstance(raw, np.integer) else raw).lower()
    if token in {"long", "1", "+1"}:
        return "long"
    if token in {"short", "-1"}:
        return "short"
    raise CaseRetrievalError("MarketEpisode direction is invalid")


def _market_episode_material_kinds(
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    raw: Any = record.get("transition_kinds", record.get("transition_kinds_json"))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CaseRetrievalError("MarketEpisode transition kinds are invalid") from exc
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise CaseRetrievalError("MarketEpisode transition kinds are invalid")
    kinds = tuple(str(item).strip().lower() for item in raw)
    if not kinds or any(kind not in MARKET_EPISODE_MATERIAL_KINDS for kind in kinds):
        raise CaseRetrievalError("MarketEpisode transition kinds are invalid")
    canonical = tuple(sorted(set(kinds), key=_MARKET_EPISODE_MATERIAL_KIND_ORDER.__getitem__))
    if kinds != canonical:
        raise CaseRetrievalError("MarketEpisode transition kinds are invalid")
    return kinds


@dataclass(frozen=True)
class EpisodeEmbeddingRecord:
    """One causally admissible decision embedding at episode grain."""

    case_id: str
    revision_id: str
    revision_stage: str
    revision_index: int
    stage_identity: str
    stage_occurrence: int
    market_epoch_id: str
    context_thesis_id: str
    entry_episode_id: str
    decision_at: pd.Timestamp
    direction: str
    regime: str
    data_split: str
    embedding_model_version: str
    embedding_checkpoint_id: str
    embedding_input_protocol: str
    decision_embedding: tuple[float, ...]
    embedding_asof: pd.Timestamp
    feature_max_at: pd.Timestamp
    outcome_fields_used: bool = False
    embedding_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "revision_id",
            "stage_identity",
            "market_epoch_id",
            "context_thesis_id",
            "entry_episode_id",
            "direction",
            "regime",
            "data_split",
            "embedding_model_version",
            "embedding_checkpoint_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if len(self.embedding_checkpoint_id) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.embedding_checkpoint_id
        ):
            raise CaseRetrievalError(
                "embedding_checkpoint_id must be a lowercase content SHA-256"
            )
        if self.embedding_input_protocol != INFERENCE_INPUT_PROTOCOL:
            raise CaseRetrievalError(
                "decision embeddings must use the deterministic unmasked "
                "input protocol"
            )
        stage = _nonempty(self.revision_stage, "revision_stage").lower()
        if stage not in CASE_REVISION_STAGES:
            raise CaseRetrievalError(
                f"unsupported decision revision stage {self.revision_stage!r}"
            )
        object.__setattr__(self, "revision_stage", stage)
        if isinstance(self.revision_index, bool) or not isinstance(
            self.revision_index, (int, np.integer)
        ) or int(self.revision_index) < 0:
            raise CaseRetrievalError("revision_index must be a non-negative integer")
        object.__setattr__(self, "revision_index", int(self.revision_index))
        if isinstance(self.stage_occurrence, bool) or not isinstance(
            self.stage_occurrence, (int, np.integer)
        ) or int(self.stage_occurrence) != 0:
            raise CaseRetrievalError(
                "v1 index admits only the first causal stage occurrence"
            )
        object.__setattr__(self, "stage_occurrence", 0)
        object.__setattr__(self, "data_split", self.data_split.lower())
        decision_at = _aware_utc(self.decision_at, "decision_at")
        embedding_asof = _aware_utc(self.embedding_asof, "embedding_asof")
        feature_max_at = _aware_utc(self.feature_max_at, "feature_max_at")
        if embedding_asof != decision_at:
            raise CaseRetrievalError(
                "decision embedding must be frozen at the exact decision clock"
            )
        if feature_max_at > decision_at:
            raise CaseRetrievalError(
                "embedding features include information after the decision clock"
            )
        if self.outcome_fields_used is not False:
            raise CaseRetrievalError(
                "outcome_fields_used must be explicitly false"
            )
        forbidden_names = sorted(
            name
            for name in self.embedding_feature_names
            if _looks_like_outcome_field(name)
        )
        if forbidden_names:
            raise CaseRetrievalError(
                "outcome/future fields are forbidden from embedding inputs: "
                f"{forbidden_names}"
            )
        object.__setattr__(self, "decision_at", decision_at)
        object.__setattr__(self, "embedding_asof", embedding_asof)
        object.__setattr__(self, "feature_max_at", feature_max_at)
        object.__setattr__(
            self,
            "decision_embedding",
            tuple(float(item) for item in self.decision_embedding),
        )
        object.__setattr__(
            self,
            "embedding_feature_names",
            tuple(str(item) for item in self.embedding_feature_names),
        )

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        embedding_dim: int = DEFAULT_MARKET_EMBEDDING_DIM,
    ) -> "EpisodeEmbeddingRecord":
        decision_at = _aware_utc(
            _first_present(record, "decision_at", "asof", "embedding_asof"),
            "decision_at",
        )
        embedding = _normalise_embedding(
            _first_present(
                record,
                "decision_embedding",
                "market_embedding",
                "embedding",
            ),
            expected_dim=embedding_dim,
        )
        names = _validate_embedding_inputs(record)
        inline_columns = _inline_outcome_columns(record)
        if inline_columns:
            raise CaseRetrievalError(
                "inline outcome columns are forbidden: "
                f"{list(inline_columns)}"
            )
        outcome_fields_used = record.get("outcome_fields_used")
        if outcome_fields_used is not False:
            raise CaseRetrievalError(
                "outcome_fields_used must be present and explicitly false"
            )
        outcome = record.get("frozen_outcome", record.get("outcome", {}))
        if outcome is None:
            outcome = {}
        if not isinstance(outcome, Mapping):
            raise CaseRetrievalError("frozen_outcome must be a mapping")
        if outcome:
            raise CaseRetrievalError(
                "inline outcomes are forbidden; join the independent outcome "
                "artifact only after neighbour selection"
            )
        return cls(
            case_id=_first_present(record, "case_id"),
            revision_id=_first_present(record, "revision_id"),
            revision_stage=_first_present(record, "revision_stage"),
            revision_index=_first_present(record, "revision_index"),
            stage_identity=_first_present(record, "stage_identity"),
            stage_occurrence=_first_present(record, "stage_occurrence"),
            market_epoch_id=_first_present(record, "market_epoch_id"),
            context_thesis_id=_first_present(record, "context_thesis_id"),
            entry_episode_id=_first_present(record, "entry_episode_id"),
            decision_at=decision_at,
            direction=_first_present(record, "direction"),
            regime=record.get("regime", "unknown"),
            data_split=_first_present(
                record, "data_split", "split", "split_role"
            ),
            embedding_model_version=_first_present(
                record, "embedding_model_version", "model_version"
            ),
            embedding_checkpoint_id=_first_present(
                record, "embedding_checkpoint_id"
            ),
            embedding_input_protocol=_first_present(
                record, "embedding_input_protocol"
            ),
            decision_embedding=tuple(float(item) for item in embedding),
            embedding_asof=_aware_utc(
                record.get("embedding_asof", decision_at), "embedding_asof"
            ),
            feature_max_at=_aware_utc(
                record.get("feature_max_at", decision_at), "feature_max_at"
            ),
            outcome_fields_used=False,
            embedding_feature_names=names,
        )

    @property
    def episode_key(self) -> tuple[str, str, str, str]:
        return (
            self.data_split,
            self.market_epoch_id,
            self.entry_episode_id,
            self.revision_stage,
        )

    def input_fingerprint(self) -> str:
        """Hash only fields allowed to determine neighbours."""

        return _sha256(
            _canonical_json(
                {
                    "case_id": self.case_id,
                    "revision_id": self.revision_id,
                    "revision_stage": self.revision_stage,
                    "revision_index": self.revision_index,
                    "stage_identity": self.stage_identity,
                    "stage_occurrence": self.stage_occurrence,
                    "market_epoch_id": self.market_epoch_id,
                    "context_thesis_id": self.context_thesis_id,
                    "entry_episode_id": self.entry_episode_id,
                    "decision_at": self.decision_at.isoformat(),
                    "direction": self.direction,
                    "regime": self.regime,
                    "data_split": self.data_split,
                    "embedding_model_version": self.embedding_model_version,
                    "embedding_checkpoint_id": self.embedding_checkpoint_id,
                    "embedding_input_protocol": self.embedding_input_protocol,
                    "embedding_asof": self.embedding_asof.isoformat(),
                    "feature_max_at": self.feature_max_at.isoformat(),
                    "embedding_feature_names": self.embedding_feature_names,
                    "decision_embedding": self.decision_embedding,
                }
            )
        )

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return input-side metadata only.

        Frozen outcomes intentionally do not survive an index checkpoint.
        Callers join the independently stored outcome artifact at query time.
        """

        return {
            "case_id": self.case_id,
            "revision_id": self.revision_id,
            "revision_stage": self.revision_stage,
            "revision_index": self.revision_index,
            "stage_identity": self.stage_identity,
            "stage_occurrence": self.stage_occurrence,
            "market_epoch_id": self.market_epoch_id,
            "context_thesis_id": self.context_thesis_id,
            "entry_episode_id": self.entry_episode_id,
            "decision_at": self.decision_at.isoformat(),
            "direction": self.direction,
            "regime": self.regime,
            "data_split": self.data_split,
            "embedding_model_version": self.embedding_model_version,
            "embedding_checkpoint_id": self.embedding_checkpoint_id,
            "embedding_input_protocol": self.embedding_input_protocol,
            "embedding_asof": self.embedding_asof.isoformat(),
            "feature_max_at": self.feature_max_at.isoformat(),
            "outcome_fields_used": False,
            "embedding_feature_names": list(self.embedding_feature_names),
        }


@dataclass(frozen=True)
class EpisodeEmbeddingQuery:
    """A current episode embedding with no future-result channel."""

    case_id: str
    market_epoch_id: str
    revision_id: str
    revision_stage: str
    revision_index: int
    stage_identity: str
    stage_occurrence: int
    context_thesis_id: str
    entry_episode_id: str
    decision_at: pd.Timestamp
    direction: str
    regime: str
    data_split: str
    embedding_model_version: str
    embedding_checkpoint_id: str
    embedding_input_protocol: str
    decision_embedding: tuple[float, ...]
    embedding_asof: pd.Timestamp
    feature_max_at: pd.Timestamp
    reference_splits: tuple[str, ...] = ()
    outcome_fields_used: bool = False
    embedding_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        probe = EpisodeEmbeddingRecord(
            case_id=self.case_id,
            revision_id=self.revision_id,
            revision_stage=self.revision_stage,
            revision_index=self.revision_index,
            stage_identity=self.stage_identity,
            stage_occurrence=self.stage_occurrence,
            market_epoch_id=self.market_epoch_id,
            context_thesis_id=self.context_thesis_id,
            entry_episode_id=self.entry_episode_id,
            decision_at=self.decision_at,
            direction=self.direction,
            regime=self.regime,
            data_split=self.data_split,
            embedding_model_version=self.embedding_model_version,
            embedding_checkpoint_id=self.embedding_checkpoint_id,
            embedding_input_protocol=self.embedding_input_protocol,
            decision_embedding=self.decision_embedding,
            embedding_asof=self.embedding_asof,
            feature_max_at=self.feature_max_at,
            outcome_fields_used=self.outcome_fields_used,
            embedding_feature_names=self.embedding_feature_names,
        )
        for name in (
            "case_id",
            "market_epoch_id",
            "revision_id",
            "revision_stage",
            "revision_index",
            "stage_identity",
            "stage_occurrence",
            "context_thesis_id",
            "entry_episode_id",
            "decision_at",
            "direction",
            "regime",
            "data_split",
            "embedding_model_version",
            "embedding_checkpoint_id",
            "embedding_input_protocol",
            "decision_embedding",
            "embedding_asof",
            "feature_max_at",
            "embedding_feature_names",
        ):
            object.__setattr__(self, name, getattr(probe, name))
        references = self.reference_splits or _DEFAULT_REFERENCE_SPLITS.get(
            probe.data_split, (probe.data_split,)
        )
        references = tuple(
            dict.fromkeys(
                _nonempty(item, "reference split").lower()
                for item in references
            )
        )
        if not references:
            raise CaseRetrievalError("at least one reference split is required")
        forbidden = sorted(set(references) & _EVALUATION_SPLITS)
        if forbidden:
            raise CaseRetrievalError(
                "queries cannot use evaluation-role reference corpora: "
                f"{forbidden}"
            )
        object.__setattr__(self, "reference_splits", references)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        embedding_dim: int = DEFAULT_MARKET_EMBEDDING_DIM,
    ) -> "EpisodeEmbeddingQuery":
        copied = dict(record)
        copied.setdefault("case_id", "query")
        parsed = EpisodeEmbeddingRecord.from_mapping(
            copied, embedding_dim=embedding_dim
        )
        raw_references = record.get(
            "allowed_reference_splits",
            record.get("reference_splits", record.get("reference_split", ())),
        )
        if isinstance(raw_references, str):
            references = (raw_references,)
        elif raw_references is None:
            references = ()
        elif isinstance(raw_references, Sequence):
            references = tuple(str(item) for item in raw_references)
        else:
            raise CaseRetrievalError("reference_splits must be a string or sequence")
        return cls(
            case_id=parsed.case_id,
            market_epoch_id=parsed.market_epoch_id,
            revision_id=parsed.revision_id,
            revision_stage=parsed.revision_stage,
            revision_index=parsed.revision_index,
            stage_identity=parsed.stage_identity,
            stage_occurrence=parsed.stage_occurrence,
            context_thesis_id=parsed.context_thesis_id,
            entry_episode_id=parsed.entry_episode_id,
            decision_at=parsed.decision_at,
            direction=parsed.direction,
            regime=parsed.regime,
            data_split=parsed.data_split,
            embedding_model_version=parsed.embedding_model_version,
            embedding_checkpoint_id=parsed.embedding_checkpoint_id,
            embedding_input_protocol=parsed.embedding_input_protocol,
            decision_embedding=parsed.decision_embedding,
            embedding_asof=parsed.embedding_asof,
            feature_max_at=parsed.feature_max_at,
            reference_splits=references,
            outcome_fields_used=False,
            embedding_feature_names=parsed.embedding_feature_names,
        )


@dataclass(frozen=True)
class MarketEpisodeEmbeddingRecord:
    revision_id: str
    revision_index: int
    material_kind: str
    run_manifest_sha256: str
    market_epoch_id: str
    market_episode_id: str
    entry_location_id: str
    entry_path_id: str
    decision_at: pd.Timestamp
    direction: str
    data_split: str
    embedding_model_version: str
    embedding_checkpoint_id: str
    decision_embedding: tuple[float, ...]
    feature_max_at: pd.Timestamp

    def __post_init__(self) -> None:
        for name in (
            "revision_id",
            "run_manifest_sha256",
            "market_epoch_id",
            "market_episode_id",
            "entry_location_id",
            "entry_path_id",
            "embedding_model_version",
            "embedding_checkpoint_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if len(self.run_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.run_manifest_sha256
        ):
            raise CaseRetrievalError("MarketEpisode run manifest SHA is invalid")
        if isinstance(self.revision_index, bool) or not isinstance(
            self.revision_index, (int, np.integer)
        ) or int(self.revision_index) < 0:
            raise CaseRetrievalError("MarketEpisode revision_index is invalid")
        object.__setattr__(self, "revision_index", int(self.revision_index))
        material_kind = str(self.material_kind).strip().lower()
        if material_kind not in MARKET_EPISODE_MATERIAL_KINDS:
            raise CaseRetrievalError("MarketEpisode material kind is invalid")
        object.__setattr__(self, "material_kind", material_kind)
        direction = _market_episode_direction(self.direction)
        object.__setattr__(self, "direction", direction)
        expected = canonical_market_episode_id(
            self.market_epoch_id,
            self.entry_location_id,
            self.entry_path_id,
            direction,
        )
        if self.market_episode_id != expected:
            raise CaseRetrievalError("market_episode_id is not canonical")
        decision_at = _aware_utc(self.decision_at, "decision_at")
        feature_max_at = _aware_utc(self.feature_max_at, "feature_max_at")
        if feature_max_at > decision_at:
            raise CaseRetrievalError("MarketEpisode feature clock is in the future")
        data_split = _nonempty(self.data_split, "data_split").lower()
        object.__setattr__(self, "data_split", data_split)
        object.__setattr__(self, "decision_at", decision_at)
        object.__setattr__(self, "feature_max_at", feature_max_at)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        material_kind: str,
        embedding_dim: int = DEFAULT_MARKET_EMBEDDING_DIM,
    ) -> "MarketEpisodeEmbeddingRecord":
        selected_kind = str(material_kind).strip().lower()
        if selected_kind not in _market_episode_material_kinds(record):
            raise CaseRetrievalError("MarketEpisode material kind is absent")
        decision_at = _aware_utc(
            _first_present(record, "decision_at", "asof", "embedding_asof"),
            "decision_at",
        )
        if (
            record.get("outcome_fields_used") is not False
            or "frozen_outcome" in record or "outcome" in record
            or _inline_outcome_columns(record)
            or record.get("revision_stage", "market_episode_transition")
            != "market_episode_transition"
            or record.get("embedding_clock", "decision_time")
            not in {"decision", "decision_time"}
            or str(record.get("embedding_input_protocol", ""))
            != INFERENCE_INPUT_PROTOCOL
            or _aware_utc(record.get("embedding_asof", decision_at), "embedding_asof")
            != decision_at
        ):
            raise CaseRetrievalError("MarketEpisode embedding input is not outcome-blind")
        _validate_embedding_inputs(record)
        embedding = _normalise_embedding(
            _first_present(
                record,
                "decision_embedding",
                "market_embedding",
                "embedding",
            ),
            expected_dim=embedding_dim,
        )
        return cls(
            revision_id=_first_present(record, "revision_id"),
            revision_index=_first_present(record, "revision_index"),
            material_kind=selected_kind,
            run_manifest_sha256=_first_present(
                record, "run_manifest_sha256"
            ),
            market_epoch_id=_first_present(record, "market_epoch_id"),
            market_episode_id=_first_present(record, "market_episode_id"),
            entry_location_id=_first_present(record, "entry_location_id"),
            entry_path_id=_first_present(record, "entry_path_id"),
            decision_at=decision_at,
            direction=_first_present(record, "direction"),
            data_split=_first_present(record, "data_split", "split_role", "split"),
            embedding_model_version=_first_present(
                record,
                "embedding_model_version",
                "model_version",
            ),
            embedding_checkpoint_id=_first_present(
                record,
                "embedding_checkpoint_id",
            ),
            decision_embedding=tuple(float(item) for item in embedding),
            feature_max_at=_aware_utc(
                record.get("feature_max_at", decision_at),
                "feature_max_at",
            ),
        )


@dataclass(frozen=True)
class MarketEpisodeEmbeddingQuery(MarketEpisodeEmbeddingRecord):
    reference_splits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        refs = self.reference_splits or _DEFAULT_REFERENCE_SPLITS.get(
            self.data_split,
            (self.data_split,),
        )
        refs = tuple(dict.fromkeys(str(item).strip().lower() for item in refs))
        if not refs or set(refs) & _EVALUATION_SPLITS:
            raise CaseRetrievalError("MarketEpisode reference splits are invalid")
        object.__setattr__(self, "reference_splits", refs)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        material_kind: str | None = None,
        embedding_dim: int = DEFAULT_MARKET_EMBEDDING_DIM,
    ) -> "MarketEpisodeEmbeddingQuery":
        kinds = _market_episode_material_kinds(record)
        if material_kind is None:
            if len(kinds) != 1:
                raise CaseRetrievalError(
                    "query material_kind is required for a multi-kind revision"
                )
            selected = kinds[0]
        else:
            selected = str(material_kind).strip().lower()
            if selected not in kinds:
                raise CaseRetrievalError(
                    "query material kind is absent from its revision"
                )
        parsed = MarketEpisodeEmbeddingRecord.from_mapping(
            record,
            material_kind=selected,
            embedding_dim=embedding_dim,
        )
        raw_refs = record.get(
            "reference_splits", record.get("allowed_reference_splits", ())
        )
        refs = (raw_refs,) if isinstance(raw_refs, str) else tuple(raw_refs or ())
        return cls(**parsed.__dict__, reference_splits=refs)


@dataclass(frozen=True)
class EnsembleMemberPrediction:
    """Decision-clock probability heads emitted by one independent model."""

    member_id: str
    checkpoint_id: str
    model_version: str
    case_id: str
    revision_id: str
    entry_episode_id: str
    decision_at: pd.Timestamp
    feature_max_at: pd.Timestamp
    outcome_fields_used: bool
    head_predictions: Mapping[str, tuple[float, ...]]
    input_protocol: str = INFERENCE_INPUT_PROTOCOL

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_id", _nonempty(self.member_id, "member_id"))
        object.__setattr__(
            self, "checkpoint_id", _nonempty(self.checkpoint_id, "checkpoint_id")
        )
        if len(self.checkpoint_id) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.checkpoint_id
        ):
            raise CaseRetrievalError(
                "ensemble checkpoint_id must be a lowercase content SHA-256"
            )
        for name in (
            "model_version",
            "case_id",
            "revision_id",
            "entry_episode_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        decision_at = _aware_utc(self.decision_at, "ensemble decision_at")
        feature_max_at = _aware_utc(
            self.feature_max_at, "ensemble feature_max_at"
        )
        if feature_max_at > decision_at:
            raise CaseRetrievalError(
                "ensemble head features exceed the decision clock"
            )
        if self.outcome_fields_used is not False:
            raise CaseRetrievalError(
                "ensemble heads must explicitly exclude outcome fields"
            )
        if self.input_protocol != INFERENCE_INPUT_PROTOCOL:
            raise CaseRetrievalError(
                "ensemble heads must use the deterministic unmasked input protocol"
            )
        object.__setattr__(self, "decision_at", decision_at)
        object.__setattr__(self, "feature_max_at", feature_max_at)
        if set(self.head_predictions) != set(OUTCOME_BLIND_HEAD_WIDTHS):
            raise CaseRetrievalError(
                "ensemble members must expose the fixed outcome-blind head schema"
            )
        normalised: dict[str, tuple[float, ...]] = {}
        for raw_name, raw_values in self.head_predictions.items():
            name = _nonempty(raw_name, "prediction head name")
            values = (raw_values,) if isinstance(raw_values, (int, float)) else raw_values
            try:
                vector = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise CaseRetrievalError(
                    f"ensemble head {name} is not numeric"
                ) from exc
            if (
                vector.ndim != 1
                or vector.size != OUTCOME_BLIND_HEAD_WIDTHS[name]
                or not np.isfinite(vector).all()
            ):
                raise CaseRetrievalError(
                    f"ensemble head {name} has the wrong finite probability shape"
                )
            if np.any(vector < 0.0) or np.any(vector > 1.0):
                raise CaseRetrievalError(
                    f"ensemble head {name} must contain probabilities in [0, 1]"
                )
            if not math.isclose(
                float(np.sum(vector)), 1.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise CaseRetrievalError(
                    f"ensemble categorical head {name} must sum to one"
                )
            normalised[name] = tuple(float(item) for item in vector)
        object.__setattr__(self, "head_predictions", normalised)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EnsembleMemberPrediction":
        heads = value.get("head_predictions", value.get("predictions"))
        if not isinstance(heads, Mapping):
            raise CaseRetrievalError("ensemble predictions must be a mapping of heads")
        return cls(
            member_id=_first_present(value, "member_id"),
            checkpoint_id=_first_present(value, "checkpoint_id", "model_version"),
            model_version=_first_present(value, "model_version"),
            case_id=_first_present(value, "case_id"),
            revision_id=_first_present(value, "revision_id"),
            entry_episode_id=_first_present(value, "entry_episode_id"),
            decision_at=_first_present(value, "decision_at"),
            feature_max_at=_first_present(value, "feature_max_at"),
            outcome_fields_used=value.get("outcome_fields_used"),
            head_predictions={str(key): item for key, item in heads.items()},
            input_protocol=_first_present(value, "input_protocol"),
        )


@dataclass(frozen=True)
class OODThresholds:
    """Pre-registered confidence-routing thresholds.

    They never widen an Eye/playbook gate and never create an entry signal.
    """

    minimum_neighbours: int = 5
    density_radius: float = 0.35
    minimum_density_continue: float = 1.0
    minimum_density_abstain: float = 0.2
    maximum_mean_distance_continue: float = 0.25
    maximum_mean_distance_abstain: float = 0.60
    maximum_nearest_distance_abstain: float = 0.80
    minimum_ensemble_members: int = 3
    maximum_disagreement_continue: float = 0.08
    maximum_disagreement_abstain: float = 0.25

    def __post_init__(self) -> None:
        if self.minimum_neighbours < 1:
            raise CaseRetrievalError("minimum_neighbours must be positive")
        if self.minimum_ensemble_members < 2:
            raise CaseRetrievalError(
                "minimum_ensemble_members must be at least two"
            )
        for name in (
            "density_radius",
            "minimum_density_continue",
            "minimum_density_abstain",
            "maximum_mean_distance_continue",
            "maximum_mean_distance_abstain",
            "maximum_nearest_distance_abstain",
            "maximum_disagreement_continue",
            "maximum_disagreement_abstain",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise CaseRetrievalError(f"{name} must be finite and non-negative")
        if self.minimum_density_abstain > self.minimum_density_continue:
            raise CaseRetrievalError("density abstain threshold exceeds continue threshold")
        if self.minimum_density_continue > 1.0:
            raise CaseRetrievalError("density thresholds cannot exceed one")
        if self.maximum_mean_distance_continue > self.maximum_mean_distance_abstain:
            raise CaseRetrievalError("mean-distance thresholds are reversed")
        if (
            self.density_radius > 2.0
            or self.maximum_mean_distance_abstain > 2.0
            or self.maximum_nearest_distance_abstain > 2.0
        ):
            raise CaseRetrievalError("cosine-distance thresholds cannot exceed two")
        if self.maximum_disagreement_continue > self.maximum_disagreement_abstain:
            raise CaseRetrievalError("disagreement thresholds are reversed")
        if self.maximum_disagreement_abstain > 1.0:
            raise CaseRetrievalError("probability disagreement cannot exceed one")


@dataclass(frozen=True)
class NeighbourMatch:
    case_id: str
    revision_id: str
    revision_stage: str
    revision_index: int
    stage_identity: str
    stage_occurrence: int
    market_epoch_id: str
    context_thesis_id: str
    entry_episode_id: str
    decision_at: pd.Timestamp
    direction: str
    regime: str
    data_split: str
    similarity: float
    cosine_distance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "revision_id": self.revision_id,
            "revision_stage": self.revision_stage,
            "revision_index": self.revision_index,
            "stage_identity": self.stage_identity,
            "stage_occurrence": self.stage_occurrence,
            "market_epoch_id": self.market_epoch_id,
            "context_thesis_id": self.context_thesis_id,
            "entry_episode_id": self.entry_episode_id,
            "decision_at": self.decision_at.isoformat(),
            "direction": self.direction,
            "regime": self.regime,
            "data_split": self.data_split,
            "similarity": self.similarity,
            "cosine_distance": self.cosine_distance,
        }


@dataclass(frozen=True)
class OODAssessment:
    policy: RetrievalPolicy
    reasons: tuple[str, ...]
    eligible_neighbours: int
    neighbours_within_density_radius: int
    local_density: float
    nearest_cosine_distance: float | None
    mean_cosine_distance: float | None
    ensemble_members: int
    ensemble_disagreement: float | None
    head_disagreement: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "reasons": list(self.reasons),
            "eligible_neighbours": self.eligible_neighbours,
            "neighbours_within_density_radius": self.neighbours_within_density_radius,
            "local_density": self.local_density,
            "nearest_cosine_distance": self.nearest_cosine_distance,
            "mean_cosine_distance": self.mean_cosine_distance,
            "ensemble_members": self.ensemble_members,
            "ensemble_disagreement": self.ensemble_disagreement,
            "head_disagreement": dict(self.head_disagreement),
        }


@dataclass(frozen=True)
class CaseRetrievalResult:
    query_entry_episode_id: str
    query_revision_stage: str
    neighbours: tuple[NeighbourMatch, ...]
    sufficient_neighbours: bool
    time_distribution: Mapping[str, int]
    direction_distribution: Mapping[str, int]
    regime_distribution: Mapping[str, int]
    frozen_outcome_distribution: Mapping[str, Any]
    ood: OODAssessment

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CASE_RETRIEVAL_SCHEMA_VERSION,
            "protocol_version": CASE_RETRIEVAL_PROTOCOL_VERSION,
            "query_entry_episode_id": self.query_entry_episode_id,
            "query_revision_stage": self.query_revision_stage,
            "neighbours": [item.as_dict() for item in self.neighbours],
            "sufficient_neighbours": self.sufficient_neighbours,
            "time_distribution": dict(self.time_distribution),
            "direction_distribution": dict(self.direction_distribution),
            "regime_distribution": dict(self.regime_distribution),
            "frozen_outcome_distribution": _jsonable(
                self.frozen_outcome_distribution
            ),
            "ood": self.ood.as_dict(),
            "empirical_prior_only": True,
            "action_authority": "none",
        }


@dataclass(frozen=True)
class MarketEpisodeRetrievalResult:
    query_run_manifest_sha256: str
    query_market_epoch_id: str
    query_market_episode_id: str
    query_material_kind: str
    neighbours: tuple[Mapping[str, Any], ...]
    sufficient_neighbours: bool
    ood: OODAssessment

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_run_manifest_sha256": self.query_run_manifest_sha256,
            "query_market_epoch_id": self.query_market_epoch_id,
            "query_market_episode_id": self.query_market_episode_id,
            "query_material_kind": self.query_material_kind,
            "neighbours": [dict(item) for item in self.neighbours],
            "sufficient_neighbours": self.sufficient_neighbours,
            "ood": self.ood.as_dict(),
            "empirical_prior_only": True,
            "action_authority": "none",
        }


def _ensemble_disagreement(
    members: Sequence[Any] | None,
) -> tuple[int, float | None, dict[str, float]]:
    if not members:
        return 0, None, {}
    field = lambda item, name: (
        item[name] if isinstance(item, Mapping) else getattr(item, name)
    )
    member_ids = [field(item, "member_id") for item in members]
    checkpoint_ids = [field(item, "checkpoint_id") for item in members]
    if (
        len(member_ids) != len(set(member_ids))
        or len(checkpoint_ids) != len(set(checkpoint_ids))
    ):
        raise CaseRetrievalError("ensemble members/checkpoints must be independent")
    head_names = tuple(sorted(field(members[0], "head_predictions")))
    expected_shapes = {
        name: len(field(members[0], "head_predictions")[name])
        for name in head_names
    }
    for member in members[1:]:
        if tuple(sorted(field(member, "head_predictions"))) != head_names:
            raise CaseRetrievalError("ensemble members expose different heads")
        if any(
            len(field(member, "head_predictions")[name]) != expected_shapes[name]
            for name in head_names
        ):
            raise CaseRetrievalError("ensemble head shapes differ across members")
    by_head: dict[str, float] = {}
    for name in head_names:
        values = np.asarray(
            [field(member, "head_predictions")[name] for member in members],
            dtype=np.float64,
        )
        # Population standard deviation is stable for any ensemble size; RMS
        # across head components gives equal per-head rather than per-logit mass.
        by_head[name] = float(np.sqrt(np.mean(np.var(values, axis=0))))
    return len(members), max(by_head.values()), by_head


def _categorical(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _first_outcome_value(outcome: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in outcome and outcome[name] is not None:
            return outcome[name]
    return None


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _outcome_distribution(
    records: Sequence[EpisodeEmbeddingRecord],
    external_outcomes: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    available_at: pd.Timestamp,
) -> dict[str, Any]:
    external_outcomes = external_outcomes or {}
    query_clock = _aware_utc(available_at, "outcome availability clock")
    resolved: list[Mapping[str, Any]] = []
    unavailable_future = 0
    for record in records:
        outcome = external_outcomes.get(record.case_id)
        if not outcome:
            continue
        if not isinstance(outcome, Mapping):
            raise CaseRetrievalError(
                f"frozen outcome for {record.case_id} must be a mapping"
            )
        for name, expected in (
            ("market_epoch_id", record.market_epoch_id),
            ("context_thesis_id", record.context_thesis_id),
            ("entry_episode_id", record.entry_episode_id),
        ):
            if name not in outcome or str(outcome[name]) != expected:
                raise CaseRetrievalError(
                    "frozen outcome identity differs from selected case: "
                    f"{record.case_id}/{name}"
                )
        if "resolved_at" not in outcome:
            raise CaseRetrievalError(
                f"frozen outcome for {record.case_id} lacks resolved_at"
            )
        resolved_at = _aware_utc(
            outcome["resolved_at"],
            f"frozen outcome {record.case_id} resolved_at",
        )
        if resolved_at < record.decision_at:
            raise CaseRetrievalError(
                f"frozen outcome for {record.case_id} predates its decision"
            )
        if resolved_at > query_clock:
            unavailable_future += 1
            continue
        resolved.append(outcome)
    first_terminal: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    fill: Counter[str] = Counter()
    expire: Counter[str] = Counter()
    draw: Counter[str] = Counter()
    terminal_reasons: Counter[str] = Counter()
    milestones: dict[str, Counter[str]] = {
        "0.5R": Counter(),
        "1R": Counter(),
        "2R": Counter(),
    }
    mfe: list[float] = []
    mae: list[float] = []
    for outcome in resolved:
        first_terminal[_categorical(
            _first_outcome_value(
                outcome,
                "first_reached",
                "first_terminal",
                "path_first",
                "first_event",
            )
        )] += 1
        resolutions[_categorical(
            _first_outcome_value(outcome, "resolution", "path_resolution")
        )] += 1
        fill[_categorical(
            _first_outcome_value(outcome, "filled", "fill_status")
        )] += 1
        expire[_categorical(
            _first_outcome_value(outcome, "expired", "expire_status")
        )] += 1
        draw[_categorical(
            _first_outcome_value(outcome, "draw_delivered", "draw_delivery")
        )] += 1
        terminal_reasons[_categorical(
            _first_outcome_value(outcome, "terminal_reason")
        )] += 1
        for label, names in (
            ("0.5R", ("reached_0_5r", "hit_0_5r", "hit_0_5R")),
            ("1R", ("reached_1r", "hit_1r", "hit_1R")),
            ("2R", ("reached_2r", "hit_2r", "hit_2R")),
        ):
            milestones[label][
                _categorical(_first_outcome_value(outcome, *names))
            ] += 1
        for destination, names in (
            (mfe, ("mfe_r", "mfe_R", "mfe", "mfe_points")),
            (mae, ("mae_r", "mae_R", "mae", "mae_points")),
        ):
            raw = _first_outcome_value(outcome, *names)
            if raw is not None:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    destination.append(value)
    return {
        "independent_episode_count": len(records),
        "resolved_episode_count": len(resolved),
        "unavailable_future_outcome_count": unavailable_future,
        "first_terminal": dict(sorted(first_terminal.items())),
        "resolution": dict(sorted(resolutions.items())),
        "fill": dict(sorted(fill.items())),
        "expire": dict(sorted(expire.items())),
        "draw_delivery": dict(sorted(draw.items())),
        "terminal_reason": dict(sorted(terminal_reasons.items())),
        "r_milestones": {
            key: dict(sorted(value.items())) for key, value in milestones.items()
        },
        "mfe_r": _numeric_summary(mfe),
        "mae_r": _numeric_summary(mae),
    }


def _cosine_density_snapshot(
    vectors: np.ndarray,
    eligible_indices: Sequence[int],
    query_vector: np.ndarray,
    rank_keys: Sequence[tuple[Any, ...]],
    *,
    k: int,
    thresholds: OODThresholds,
) -> tuple[Any, ...]:
    similarities = (
        vectors[list(eligible_indices)] @ query_vector
        if eligible_indices else np.empty(0, dtype=np.float64)
    )
    ranked = sorted(
        zip(eligible_indices, similarities, strict=True),
        key=lambda item: (-float(item[1]), *rank_keys[item[0]]),
    )
    distances = np.asarray(
        [1.0 - np.clip(similarity, -1.0, 1.0) for _, similarity in ranked],
        dtype=np.float64,
    )
    density = int(np.count_nonzero(distances <= thresholds.density_radius))
    sample = distances[: thresholds.minimum_neighbours]
    return (
        ranked,
        ranked[:k],
        density,
        min(1.0, density / float(thresholds.minimum_neighbours)),
        None if not distances.size else float(distances[0]),
        None if not sample.size else float(np.mean(sample)),
    )


class EpisodeCaseIndex:
    """Immutable, episode-deduplicated cosine index."""

    def __init__(
        self,
        records: Sequence[EpisodeEmbeddingRecord],
        *,
        embedding_dim: int = DEFAULT_MARKET_EMBEDDING_DIM,
        ignored_non_decision_revisions: int = 0,
        artifact_lineage: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_dim < 1:
            raise CaseRetrievalError("embedding_dim must be positive")
        ordered = tuple(
            sorted(
                records,
                key=lambda item: (
                    item.data_split,
                    item.market_epoch_id,
                    item.decision_at.value,
                    item.entry_episode_id,
                    item.revision_stage,
                    item.context_thesis_id,
                    item.case_id,
                ),
            )
        )
        keys = [item.episode_key for item in ordered]
        if len(keys) != len(set(keys)):
            raise CaseRetrievalError(
                "index contains duplicate EntryEpisode revision stages"
            )
        identity_splits: dict[tuple[str, str], str] = {}
        for record in ordered:
            identity = (record.market_epoch_id, record.entry_episode_id)
            prior_split = identity_splits.setdefault(identity, record.data_split)
            if prior_split != record.data_split:
                raise CaseRetrievalError(
                    "one entry episode cannot be shared across data splits"
                )
        if ordered:
            versions = {item.embedding_model_version for item in ordered}
            if len(versions) != 1:
                raise CaseRetrievalError(
                    "one index cannot mix embedding model versions"
                )
            checkpoint_ids = {
                item.embedding_checkpoint_id for item in ordered
            }
            if len(checkpoint_ids) != 1:
                raise CaseRetrievalError(
                    "one index cannot mix encoder checkpoint embedding spaces"
                )
            matrix = np.asarray(
                [
                    _normalise_embedding(
                        item.decision_embedding, expected_dim=embedding_dim
                    )
                    for item in ordered
                ],
                dtype=np.float64,
            )
        else:
            matrix = np.empty((0, embedding_dim), dtype=np.float64)
        matrix.setflags(write=False)
        self._records = ordered
        self._vectors = matrix
        self.embedding_dim = int(embedding_dim)
        self.embedding_checkpoint_id = (
            None if not ordered else ordered[0].embedding_checkpoint_id
        )
        self.ignored_non_decision_revisions = int(ignored_non_decision_revisions)
        self.artifact_lineage = _normalise_artifact_lineage(artifact_lineage)

    @classmethod
    def from_mappings(
        cls,
        values: Iterable[Mapping[str, Any]],
        *,
        embedding_dim: int = DEFAULT_MARKET_EMBEDDING_DIM,
        artifact_lineage: Mapping[str, Any] | None = None,
    ) -> "EpisodeCaseIndex":
        accepted: dict[tuple[str, str, str, str], EpisodeEmbeddingRecord] = {}
        ignored = 0
        for raw in values:
            if not isinstance(raw, Mapping):
                raise CaseRetrievalError("case records must be mappings")
            inline_columns = _inline_outcome_columns(raw)
            if inline_columns:
                raise CaseRetrievalError(
                    "inline outcome columns are forbidden: "
                    f"{list(inline_columns)}"
                )
            inline_outcome = raw.get("frozen_outcome", raw.get("outcome", {}))
            if inline_outcome is not None and not isinstance(
                inline_outcome, Mapping
            ):
                raise CaseRetrievalError("inline outcome payload must be a mapping")
            if inline_outcome:
                raise CaseRetrievalError(
                    "inline outcomes are forbidden; use the independent "
                    "outcome artifact"
                )
            if not _decision_record(raw):
                ignored += 1
                continue
            raw_occurrence = raw.get("stage_occurrence")
            if isinstance(raw_occurrence, bool) or not isinstance(
                raw_occurrence, (int, np.integer)
            ) or int(raw_occurrence) < 0:
                raise CaseRetrievalError(
                    "stage_occurrence must be a non-negative integer"
                )
            if int(raw_occurrence) > 0:
                # Occurrence is assigned online by the sparse case exporter.
                # Ignoring a later occurrence is therefore causal and invariant
                # to input ordering; it never asks which later row looks best.
                ignored += 1
                continue
            record = EpisodeEmbeddingRecord.from_mapping(
                raw, embedding_dim=embedding_dim
            )
            prior = accepted.get(record.episode_key)
            if prior is None:
                accepted[record.episode_key] = record
                continue
            prior_rank = (
                prior.revision_index,
                prior.decision_at.value,
                prior.revision_id,
            )
            record_rank = (
                record.revision_index,
                record.decision_at.value,
                record.revision_id,
            )
            if record.revision_index == prior.revision_index:
                if record.input_fingerprint() != prior.input_fingerprint():
                    raise CaseRetrievalError(
                        "one episode stage has conflicting first-occurrence identity"
                    )
                ignored += 1
                continue
            if record_rank < prior_rank:
                accepted[record.episode_key] = record
            ignored += 1
        return cls(
            tuple(accepted.values()),
            embedding_dim=embedding_dim,
            ignored_non_decision_revisions=ignored,
            artifact_lineage=artifact_lineage,
        )

    @property
    def records(self) -> tuple[EpisodeEmbeddingRecord, ...]:
        return self._records

    @property
    def vectors(self) -> np.ndarray:
        view = self._vectors.view()
        view.setflags(write=False)
        return view

    def query(
        self,
        query: EpisodeEmbeddingQuery,
        *,
        k: int = 10,
        ensemble: Sequence[EnsembleMemberPrediction] | None = None,
        thresholds: OODThresholds | None = None,
        frozen_outcomes: Mapping[str, Mapping[str, Any]] | None = None,
        artifact_lineage: Mapping[str, Any] | None = None,
    ) -> CaseRetrievalResult:
        if k < 1:
            raise CaseRetrievalError("k must be positive")
        thresholds = thresholds or OODThresholds()
        supplied_lineage = _normalise_artifact_lineage(artifact_lineage)
        if any(self.artifact_lineage.values()):
            if not any(supplied_lineage.values()):
                raise CaseRetrievalError(
                    "query omits the index causal-case artifact lineage"
                )
            if supplied_lineage != self.artifact_lineage:
                raise CaseRetrievalError(
                    "query/index causal-case artifact lineages differ"
                )
        if (
            self._records
            and query.embedding_model_version
            != self._records[0].embedding_model_version
        ):
            raise CaseRetrievalError("query/index embedding model versions differ")
        if (
            self.embedding_checkpoint_id is not None
            and query.embedding_checkpoint_id
            != self.embedding_checkpoint_id
        ):
            raise CaseRetrievalError(
                "query/index encoder checkpoint embedding spaces differ"
            )
        query_vector = _normalise_embedding(
            query.decision_embedding, expected_dim=self.embedding_dim
        )
        eligible_indices = [
            index
            for index, record in enumerate(self._records)
            if record.data_split in query.reference_splits
            and record.market_epoch_id == query.market_epoch_id
            and record.revision_stage == query.revision_stage
            and record.entry_episode_id != query.entry_episode_id
            and record.decision_at < query.decision_at
        ]
        (
            ranked, selected, density_count, local_density,
            nearest_distance, mean_distance,
        ) = _cosine_density_snapshot(
            self._vectors, eligible_indices, query_vector,
            tuple(
                (-record.decision_at.value, record.entry_episode_id)
                for record in self._records
            ),
            k=k, thresholds=thresholds,
        )
        selected_records = [self._records[index] for index, _ in selected]
        neighbours = tuple(
            NeighbourMatch(
                case_id=record.case_id,
                revision_id=record.revision_id,
                revision_stage=record.revision_stage,
                revision_index=record.revision_index,
                stage_identity=record.stage_identity,
                stage_occurrence=record.stage_occurrence,
                market_epoch_id=record.market_epoch_id,
                context_thesis_id=record.context_thesis_id,
                entry_episode_id=record.entry_episode_id,
                decision_at=record.decision_at,
                direction=record.direction,
                regime=record.regime,
                data_split=record.data_split,
                similarity=float(np.clip(similarity, -1.0, 1.0)),
                cosine_distance=float(
                    1.0 - np.clip(similarity, -1.0, 1.0)
                ),
            )
            for (index, similarity), record in zip(
                selected, selected_records, strict=True
            )
        )
        if ensemble:
            for member in ensemble:
                for actual, expected, name in (
                    (member.model_version, query.embedding_model_version, "model"),
                    (member.case_id, query.case_id, "case"),
                    (member.revision_id, query.revision_id, "revision"),
                    (
                        member.entry_episode_id,
                        query.entry_episode_id,
                        "EntryEpisode",
                    ),
                    (member.decision_at, query.decision_at, "decision clock"),
                    (
                        member.feature_max_at,
                        query.feature_max_at,
                        "feature clock",
                    ),
                ):
                    if actual != expected:
                        raise CaseRetrievalError(
                            f"ensemble {name} binding differs from query"
                        )
        ensemble_members, disagreement, head_disagreement = (
            _ensemble_disagreement(ensemble)
        )
        policy, reasons = self._route_policy(
            eligible_neighbours=len(ranked),
            density_count=density_count,
            local_density=local_density,
            nearest_distance=nearest_distance,
            mean_distance=mean_distance,
            ensemble_members=ensemble_members,
            disagreement=disagreement,
            thresholds=thresholds,
        )
        sufficient = bool(
            len(ranked) >= thresholds.minimum_neighbours
            and density_count >= thresholds.minimum_neighbours
        )
        return CaseRetrievalResult(
            query_entry_episode_id=query.entry_episode_id,
            query_revision_stage=query.revision_stage,
            neighbours=neighbours,
            sufficient_neighbours=sufficient,
            time_distribution=dict(
                sorted(
                    Counter(
                        record.decision_at.strftime("%Y-%m")
                        for record in selected_records
                    ).items()
                )
            ),
            direction_distribution=dict(
                sorted(Counter(record.direction for record in selected_records).items())
            ),
            regime_distribution=dict(
                sorted(Counter(record.regime for record in selected_records).items())
            ),
            frozen_outcome_distribution=_outcome_distribution(
                selected_records,
                frozen_outcomes,
                available_at=query.decision_at,
            ),
            ood=OODAssessment(
                policy=policy,
                reasons=tuple(reasons),
                eligible_neighbours=len(ranked),
                neighbours_within_density_radius=density_count,
                local_density=local_density,
                nearest_cosine_distance=nearest_distance,
                mean_cosine_distance=mean_distance,
                ensemble_members=ensemble_members,
                ensemble_disagreement=disagreement,
                head_disagreement=head_disagreement,
            ),
        )

    @staticmethod
    def _route_policy(
        *,
        eligible_neighbours: int,
        density_count: int,
        local_density: float,
        nearest_distance: float | None,
        mean_distance: float | None,
        ensemble_members: int,
        disagreement: float | None,
        thresholds: OODThresholds,
    ) -> tuple[RetrievalPolicy, list[str]]:
        abstain: list[str] = []
        uncertain: list[str] = []
        if ensemble_members < thresholds.minimum_ensemble_members or disagreement is None:
            abstain.append("deep_ensemble_unavailable")
        if eligible_neighbours == 0 or nearest_distance is None or mean_distance is None:
            abstain.append("no_causal_neighbours")
        else:
            if nearest_distance > thresholds.maximum_nearest_distance_abstain:
                abstain.append("nearest_embedding_out_of_distribution")
            if mean_distance > thresholds.maximum_mean_distance_abstain:
                abstain.append("mean_embedding_distance_out_of_distribution")
            if local_density < thresholds.minimum_density_abstain:
                abstain.append("neighbour_density_out_of_distribution")
            if eligible_neighbours < thresholds.minimum_neighbours:
                uncertain.append("insufficient_independent_neighbours")
            if density_count < thresholds.minimum_neighbours:
                uncertain.append("sparse_local_neighbourhood")
            if mean_distance > thresholds.maximum_mean_distance_continue:
                uncertain.append("elevated_embedding_distance")
            if local_density < thresholds.minimum_density_continue:
                uncertain.append("low_neighbour_density")
        if disagreement is not None:
            if disagreement > thresholds.maximum_disagreement_abstain:
                abstain.append("ensemble_disagreement_out_of_distribution")
            elif disagreement > thresholds.maximum_disagreement_continue:
                uncertain.append("elevated_ensemble_disagreement")
        if abstain:
            return RetrievalPolicy.ABSTAIN, sorted(set(abstain + uncertain))
        if uncertain:
            return RetrievalPolicy.INCREASE_UNCERTAINTY, sorted(set(uncertain))
        return RetrievalPolicy.CONTINUE_EVALUATION, [
            "sufficient_similar_cases_and_consistent_ensemble"
        ]

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically save one hash-bound, pickle-free index checkpoint."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        vectors = np.asarray(self._vectors, dtype="<f4")
        metadata_payload = _canonical_json(
            [record.checkpoint_metadata() for record in self._records]
        )
        vector_payload = vectors.tobytes(order="C")
        model_versions = sorted(
            {record.embedding_model_version for record in self._records}
        )
        checkpoint_ids = sorted(
            {record.embedding_checkpoint_id for record in self._records}
        )
        manifest = {
            "schema_version": CASE_RETRIEVAL_SCHEMA_VERSION,
            "protocol_version": CASE_RETRIEVAL_PROTOCOL_VERSION,
            "status": "complete",
            "rows": len(self._records),
            "embedding_dim": self.embedding_dim,
            "embedding_model_versions": model_versions,
            "embedding_checkpoint_ids": checkpoint_ids,
            "artifact_lineage": self.artifact_lineage,
            "ignored_non_decision_revisions": self.ignored_non_decision_revisions,
            "vectors_dtype": "float32_little_endian",
            "vectors_sha256": _sha256(vector_payload),
            "metadata_sha256": _sha256(metadata_payload),
            "outcome_in_index_vector": False,
            "frozen_outcomes_persisted": False,
            "action_authority": "none_empirical_prior_only",
        }
        manifest_payload = _canonical_json(manifest)
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    vectors=vectors,
                    metadata=np.frombuffer(metadata_payload, dtype=np.uint8),
                    manifest=np.frombuffer(manifest_payload, dtype=np.uint8),
                )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "EpisodeCaseIndex":
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise CaseRetrievalError("index checkpoint is not a regular file")
        try:
            with np.load(source, allow_pickle=False) as archive:
                if set(archive.files) != {"vectors", "metadata", "manifest"}:
                    raise CaseRetrievalError("index checkpoint members are invalid")
                vectors = np.asarray(archive["vectors"], dtype="<f4")
                metadata_payload = bytes(
                    np.asarray(archive["metadata"], dtype=np.uint8)
                )
                manifest_payload = bytes(
                    np.asarray(archive["manifest"], dtype=np.uint8)
                )
        except (OSError, ValueError, TypeError) as exc:
            raise CaseRetrievalError("index checkpoint cannot be read") from exc
        try:
            manifest = json.loads(manifest_payload.decode("utf-8"))
            metadata = json.loads(metadata_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CaseRetrievalError("index checkpoint JSON is invalid") from exc
        if manifest.get("schema_version") != CASE_RETRIEVAL_SCHEMA_VERSION:
            raise CaseRetrievalError("unsupported case retrieval schema")
        if manifest.get("protocol_version") != CASE_RETRIEVAL_PROTOCOL_VERSION:
            raise CaseRetrievalError("case retrieval protocol mismatch")
        if manifest.get("status") != "complete":
            raise CaseRetrievalError("index checkpoint is incomplete")
        artifact_lineage = _normalise_artifact_lineage(
            manifest.get("artifact_lineage")
        )
        embedding_dim = int(manifest.get("embedding_dim", -1))
        rows = int(manifest.get("rows", -1))
        if vectors.shape != (rows, embedding_dim):
            raise CaseRetrievalError("checkpoint vector shape is invalid")
        if _sha256(vectors.tobytes(order="C")) != manifest.get("vectors_sha256"):
            raise CaseRetrievalError("checkpoint vector hash is invalid")
        if _sha256(metadata_payload) != manifest.get("metadata_sha256"):
            raise CaseRetrievalError("checkpoint metadata hash is invalid")
        if not isinstance(metadata, list) or len(metadata) != rows:
            raise CaseRetrievalError("checkpoint record cardinality is invalid")
        records: list[EpisodeEmbeddingRecord] = []
        for raw, vector in zip(metadata, vectors, strict=True):
            if not isinstance(raw, Mapping):
                raise CaseRetrievalError("checkpoint record is not a mapping")
            enriched = dict(raw)
            enriched["decision_embedding"] = vector.tolist()
            enriched["embedding_clock"] = "decision_time"
            records.append(
                EpisodeEmbeddingRecord.from_mapping(
                    enriched, embedding_dim=embedding_dim
                )
            )
        index = cls(
            records,
            embedding_dim=embedding_dim,
            ignored_non_decision_revisions=int(
                manifest.get("ignored_non_decision_revisions", 0)
            ),
            artifact_lineage=artifact_lineage,
        )
        expected_versions = sorted(
            {record.embedding_model_version for record in records}
        )
        if expected_versions != manifest.get("embedding_model_versions"):
            raise CaseRetrievalError("checkpoint model binding is invalid")
        expected_checkpoint_ids = sorted(
            {record.embedding_checkpoint_id for record in records}
        )
        if expected_checkpoint_ids != manifest.get("embedding_checkpoint_ids"):
            raise CaseRetrievalError(
                "checkpoint encoder content binding is invalid"
            )
        return index


class MarketEpisodeCaseIndex:
    """Thin neutral selector over the shared cosine/OOD core."""

    def __init__(
        self,
        records: Sequence[MarketEpisodeEmbeddingRecord],
        *,
        artifact_lineages: Sequence[Mapping[str, Any]],
        embedding_dim: int,
        dataset_contract: Mapping[str, Any] | None = None,
    ) -> None:
        self.artifact_lineages = tuple(
            _normalise_market_episode_artifact_lineage(value)
            for value in artifact_lineages
        )
        if not self.artifact_lineages:
            raise CaseRetrievalError("neutral index has no artifact lineage")
        if len({value["run_manifest_sha256"] for value in self.artifact_lineages}) != len(
            self.artifact_lineages
        ):
            raise CaseRetrievalError("neutral index repeats an artifact run lineage")
        known_runs = {
            value["run_manifest_sha256"] for value in self.artifact_lineages
        }
        if any(record.run_manifest_sha256 not in known_runs for record in records):
            raise CaseRetrievalError("neutral record has an unbound run lineage")
        self.artifact_lineage = self.artifact_lineages[0]
        self.dataset_contract = (
            None
            if dataset_contract is None
            else normalise_market_episode_dataset_contract(dataset_contract)
        )
        self.records = tuple(sorted(records, key=lambda item: (
            item.run_manifest_sha256,
            item.market_epoch_id,
            item.decision_at.value,
            item.market_episode_id,
            item.material_kind,
        )))
        spaces = {
            (item.embedding_model_version, item.embedding_checkpoint_id)
            for item in self.records
        }
        if len(spaces) > 1:
            raise CaseRetrievalError("neutral index mixes embedding spaces")
        self.embedding_space = next(iter(spaces), None)
        self.embedding_dim = embedding_dim
        self.vectors = np.asarray([
            _normalise_embedding(
                item.decision_embedding,
                expected_dim=embedding_dim,
            )
            for item in self.records
        ], dtype=np.float64).reshape((-1, embedding_dim))
        self.vectors.setflags(write=False)

    @classmethod
    def from_mappings(
        cls,
        values: Iterable[Mapping[str, Any]],
        *,
        artifact_lineage: Mapping[str, Any],
        embedding_dim: int = DEFAULT_MARKET_EMBEDDING_DIM,
        dataset_contract: Mapping[str, Any] | None = None,
    ) -> "MarketEpisodeCaseIndex":
        lineage = _normalise_market_episode_artifact_lineage(artifact_lineage)
        return cls.from_artifacts(
            (
                {
                    "records": tuple(values),
                    "artifact_lineage": lineage,
                    "dataset_contract": dataset_contract,
                },
            ),
            embedding_dim=embedding_dim,
        )

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Sequence[Mapping[str, Any]],
        *,
        embedding_dim: int = DEFAULT_MARKET_EMBEDDING_DIM,
    ) -> "MarketEpisodeCaseIndex":
        if not artifacts:
            raise CaseRetrievalError("neutral index has no reference artifacts")
        accepted: dict[
            tuple[str, str, str, str, str], MarketEpisodeEmbeddingRecord
        ] = {}
        lineages: list[dict[str, str]] = []
        contracts: list[dict[str, Any] | None] = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or set(artifact) != {
                "records", "artifact_lineage", "dataset_contract",
            }:
                raise CaseRetrievalError("neutral reference artifact is invalid")
            lineage = _normalise_market_episode_artifact_lineage(
                artifact["artifact_lineage"]
            )
            contract = (
                None
                if artifact["dataset_contract"] is None
                else normalise_market_episode_dataset_contract(
                    artifact["dataset_contract"]
                )
            )
            lineages.append(lineage)
            contracts.append(contract)
            for raw in artifact["records"]:
                if not isinstance(raw, Mapping):
                    raise CaseRetrievalError("MarketEpisode records must be mappings")
                bound = dict(raw)
                declared_run = bound.setdefault(
                    "run_manifest_sha256", lineage["run_manifest_sha256"]
                )
                if declared_run != lineage["run_manifest_sha256"]:
                    raise CaseRetrievalError(
                        "MarketEpisode record run lineage differs from its artifact"
                    )
                for kind in _market_episode_material_kinds(bound):
                    record = MarketEpisodeEmbeddingRecord.from_mapping(
                        bound,
                        material_kind=kind,
                        embedding_dim=embedding_dim,
                    )
                    key = (
                        record.data_split,
                        record.run_manifest_sha256,
                        record.market_epoch_id,
                        record.market_episode_id,
                        record.material_kind,
                    )
                    prior = accepted.get(key)
                    if prior is None or record.revision_index < prior.revision_index:
                        accepted[key] = record
                    elif record.revision_index == prior.revision_index and record != prior:
                        raise CaseRetrievalError(
                            "conflicting first material occurrence"
                        )
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise CaseRetrievalError("neutral reference dataset contracts differ")
        if len(contracts) > 1 and contracts[0] is None:
            raise CaseRetrievalError(
                "multi-artifact neutral index requires a dataset contract"
            )
        episode_splits: dict[tuple[str, str, str], str] = {}
        for record in accepted.values():
            identity = (
                record.run_manifest_sha256,
                record.market_epoch_id,
                record.market_episode_id,
            )
            if (
                episode_splits.setdefault(identity, record.data_split)
                != record.data_split
            ):
                raise CaseRetrievalError("MarketEpisode is shared across splits")
        return cls(
            tuple(accepted.values()),
            artifact_lineages=lineages,
            embedding_dim=embedding_dim,
            dataset_contract=contracts[0],
        )

    @staticmethod
    def _ensemble(
        query: MarketEpisodeEmbeddingQuery,
        members: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[int, float | None, dict[str, float]]:
        if not members:
            return 0, None, {}
        for member in members:
            required = (
                "member_id", "checkpoint_id", "model_version", "revision_id",
                "market_epoch_id", "market_episode_id", "decision_at", "feature_max_at",
                "outcome_fields_used", "input_protocol", "head_predictions",
            )
            if not isinstance(member, Mapping) or any(
                name not in member for name in required
            ):
                raise CaseRetrievalError("neutral ensemble active heads are invalid")
            heads = member["head_predictions"]
            if (
                not isinstance(heads, Mapping)
                or set(heads)
                != set(MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS)
            ):
                raise CaseRetrievalError("neutral ensemble active heads are invalid")
            bindings = (
                (member["model_version"], query.embedding_model_version),
                (member["revision_id"], query.revision_id),
                (member["market_epoch_id"], query.market_epoch_id),
                (member["market_episode_id"], query.market_episode_id),
                (
                    _aware_utc(member["decision_at"], "ensemble decision_at"),
                    query.decision_at,
                ),
                (
                    _aware_utc(
                        member["feature_max_at"],
                        "ensemble feature_max_at",
                    ),
                    query.feature_max_at,
                ),
            )
            if (
                member["outcome_fields_used"] is not False
                or member["input_protocol"] != INFERENCE_INPUT_PROTOCOL
                or any(actual != expected for actual, expected in bindings)
            ):
                raise CaseRetrievalError("neutral ensemble binding is invalid")
            for name, width in MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS.items():
                values = np.asarray(heads[name], dtype=np.float64)
                if (
                    values.shape != (width,)
                    or not np.isfinite(values).all()
                    or np.any(values < 0)
                    or np.any(values > 1)
                    or not math.isclose(
                        float(values.sum()),
                        1.0,
                        abs_tol=1e-6,
                    )
                ):
                    raise CaseRetrievalError("neutral ensemble probabilities are invalid")
        return _ensemble_disagreement(members)

    def query(
        self,
        query: MarketEpisodeEmbeddingQuery,
        *,
        artifact_lineage: Mapping[str, Any],
        k: int = 10,
        ensemble: Sequence[Mapping[str, Any]] | None = None,
        thresholds: OODThresholds | None = None,
        dataset_contract: Mapping[str, Any] | None = None,
        require_different_calendar_date: bool = False,
    ) -> MarketEpisodeRetrievalResult:
        query_lineage = _normalise_market_episode_artifact_lineage(
            artifact_lineage
        )
        if k < 1:
            raise CaseRetrievalError("neutral query contract is invalid")
        if self.dataset_contract is None:
            if dataset_contract is not None or query_lineage != self.artifact_lineage:
                raise CaseRetrievalError("neutral query contract is invalid")
        elif (
            dataset_contract is None
            or normalise_market_episode_dataset_contract(dataset_contract)
            != self.dataset_contract
        ):
            raise CaseRetrievalError("neutral query dataset contracts differ")
        if self.embedding_space and self.embedding_space != (
            query.embedding_model_version,
            query.embedding_checkpoint_id,
        ):
            raise CaseRetrievalError("neutral query embedding space differs")
        thresholds = thresholds or OODThresholds()
        query_run = query_lineage["run_manifest_sha256"]
        if query.run_manifest_sha256 != query_run:
            raise CaseRetrievalError("neutral query record run lineage differs")
        if ensemble and any(
            not isinstance(member, Mapping)
            or member.get("run_manifest_sha256") != query_run
            for member in ensemble
        ):
            raise CaseRetrievalError("neutral ensemble run lineage is invalid")
        query_scope = (
            query_run,
            query.market_epoch_id,
            query.market_episode_id,
        )
        timezone = (
            "UTC"
            if self.dataset_contract is None
            else str(self.dataset_contract["calendar_timezone"])
        )
        try:
            query_date = query.decision_at.tz_convert(timezone).date()
        except (KeyError, TypeError, ValueError) as exc:
            raise CaseRetrievalError(
                "MarketEpisode calendar timezone is invalid"
            ) from exc
        eligible = [
            index
            for index, record in enumerate(self.records)
            if record.data_split in query.reference_splits
            and record.material_kind == query.material_kind
            and (
                record.run_manifest_sha256,
                record.market_epoch_id,
                record.market_episode_id,
            )
            != query_scope
            and record.decision_at < query.decision_at
            and (
                not require_different_calendar_date
                or record.decision_at.tz_convert(timezone).date() != query_date
            )
        ]
        ranked, selected, density, local, nearest, mean = _cosine_density_snapshot(
            self.vectors,
            eligible,
            _normalise_embedding(
                query.decision_embedding,
                expected_dim=self.embedding_dim,
            ),
            tuple(
                (
                    -item.decision_at.value,
                    item.run_manifest_sha256,
                    item.market_episode_id,
                )
                for item in self.records
            ),
            k=k,
            thresholds=thresholds,
        )
        count, disagreement, by_head = self._ensemble(query, ensemble)
        policy, reasons = EpisodeCaseIndex._route_policy(
            eligible_neighbours=len(ranked),
            density_count=density,
            local_density=local,
            nearest_distance=nearest,
            mean_distance=mean,
            ensemble_members=count,
            disagreement=disagreement,
            thresholds=thresholds,
        )
        neighbours = tuple(
            {
                "run_manifest_sha256": self.records[index].run_manifest_sha256,
                "market_epoch_id": self.records[index].market_epoch_id,
                "market_episode_id": self.records[index].market_episode_id,
                "revision_id": self.records[index].revision_id,
                "decision_at": self.records[index].decision_at.isoformat(),
                "direction": self.records[index].direction,
                "similarity": float(np.clip(similarity, -1.0, 1.0)),
            }
            for index, similarity in selected
        )
        return MarketEpisodeRetrievalResult(
            query_run_manifest_sha256=query_run,
            query_market_epoch_id=query.market_epoch_id,
            query_market_episode_id=query.market_episode_id,
            query_material_kind=query.material_kind,
            neighbours=neighbours,
            sufficient_neighbours=(
                len(ranked) >= thresholds.minimum_neighbours
                and density >= thresholds.minimum_neighbours
            ),
            ood=OODAssessment(
                policy=policy,
                reasons=tuple(reasons),
                eligible_neighbours=len(ranked),
                neighbours_within_density_radius=density,
                local_density=local,
                nearest_cosine_distance=nearest,
                mean_cosine_distance=mean,
                ensemble_members=count,
                ensemble_disagreement=disagreement,
                head_disagreement=by_head,
            ),
        )


__all__ = [
    "CASE_RETRIEVAL_PROTOCOL",
    "CASE_RETRIEVAL_PROTOCOL_VERSION",
    "CASE_RETRIEVAL_SCHEMA_VERSION",
    "DEFAULT_MARKET_EMBEDDING_DIM",
    "MARKET_EPISODE_ACTIVE_ENSEMBLE_HEAD_WIDTHS",
    "MARKET_EPISODE_DATASET_CONTRACT_KEYS",
    "MARKET_EPISODE_FIRST_OCCURRENCE_SELECTION_CONTRACT",
    "MARKET_EPISODE_MATERIAL_KINDS",
    "CaseRetrievalError",
    "CaseRetrievalResult",
    "EnsembleMemberPrediction",
    "EpisodeCaseIndex",
    "EpisodeEmbeddingQuery",
    "EpisodeEmbeddingRecord",
    "MarketEpisodeCaseIndex",
    "MarketEpisodeEmbeddingQuery",
    "MarketEpisodeEmbeddingRecord",
    "MarketEpisodeRetrievalResult",
    "NeighbourMatch",
    "OODAssessment",
    "OODThresholds",
    "normalise_market_episode_dataset_contract",
    "RetrievalPolicy",
]
