"""Outcome-blind multi-timeframe market representation primitives.

This module is deliberately outside the observation/playbook/decision path.  It
reads immutable canonical OHLCV *prefix references* and contemporaneous event
snapshots, then (when the optional PyTorch dependency is installed) encodes the
result into a 128-dimensional decision-time embedding.

The preprocessing layer has no PyTorch dependency so its causal contracts can
be checked in the production replay environment.  Model construction and
training fail closed when PyTorch is unavailable.
"""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass, field, replace
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


try:  # Optional by design; do not make replay import depend on PyTorch.
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    TORCH_AVAILABLE = True
except ModuleNotFoundError as exc:  # pragma: no cover - branch depends on env.
    if exc.name != "torch":
        raise
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    pack_padded_sequence = None  # type: ignore[assignment]
    pad_packed_sequence = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


MODEL_VERSION = "1.0.0-causal-multiscale-gru"
FEATURE_SCHEMA_VERSION = 1
TIMEFRAMES = ("4h", "1h", "15m", "5m", "1m")
EMBEDDING_DIM = 128
PARAMETER_BUDGET = 5_000_000

# Development defaults are fixed in code so a validation run cannot tune the
# acceptance bar after seeing its own result.  They are representation-quality
# gates, not profitability thresholds.
MIN_TASK_RELATIVE_IMPROVEMENT = 0.02
MIN_MEAN_RELATIVE_IMPROVEMENT = 0.05
MIN_RECONSTRUCTION_RELATIVE_IMPROVEMENT = 0.02
MIN_REGIME_ACCURACY_MARGIN = 0.05
MIN_CROSS_DATE_RETRIEVAL_AT_K = 0.60
MIN_CROSS_DATE_RETRIEVAL_LIFT = 0.10
MIN_DIRECTION_REGIME_SAMPLES = 2
MIN_GROUP_FEATURE_STD = 1e-6

# Case artifacts retain these fields for audit, but they are not market
# observations and therefore may not become encoder tokens.
CONTROL_TOKEN_KEY_MARKERS = (
    "action",
    "decision",
    "risk",
    "hard_gate",
    "playbook",
    "observable_regime",
    "mechanism_label",
)
DIRECT_LABEL_SOURCE_AUDIT = {
    "regime_and_mechanism": (
        "entry_episode.playbook",
        "context_thesis.playbook",
        "observable_regime",
        "mechanism_label",
    ),
    "scale_direction_alignment": ("scale_relations",),
}
ISO_TIMESTAMP_TOKEN_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
BAR_END_INDEX_BINDING = "__index_is_completed_bar_end__"
INFERENCE_INPUT_PROTOCOL = "inference_unmasked_v1"
NEUTRAL_INFERENCE_INPUT_PROTOCOL = "neutral_direct_source_filtered_v1"
NEUTRAL_TRAINING_CONTRACT = "neutral-market-representation-b0-v2"
_NEUTRAL_DIRECT_SOURCE_EVENT_PREPROCESSING_PROTOCOL = {
    "protocol_version": "neutral-direct-source-event-preprocessing-1.0.0",
    "scope": "neutral_market_episode",
    "direct_source_detector": "prepared_direct_label_source_event_mask_v1",
    "direct_source_action": "drop",
    "retained_event_order": "source_order",
    "empty_event_fallback": {
        "event_count": 1,
        "token_id": MASK_TOKEN_ID,
        "numeric": [0.0, 0.0, 0.0, 0.0],
    },
    "candle_features": "unchanged",
    "objective_random_masks": "applied_after_this_protocol_not_persisted",
    "deterministic": True,
}


def neutral_direct_source_preprocessing_identity() -> dict[str, Any]:
    """Return the immutable-by-copy identity for neutral event preprocessing."""

    protocol = copy.deepcopy(_NEUTRAL_DIRECT_SOURCE_EVENT_PREPROCESSING_PROTOCOL)
    encoded = json.dumps(
        protocol, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "protocol": protocol,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


CASE_REVISION_STAGES = frozenset(
    {
        "context_formed",
        "context_changed",
        "episode_created",
        "zone_registered",
        "first_pullback",
        "trigger",
        "plan_formed",
        "terminal",
        "market_episode_transition",
    }
)
NEXT_EVENT_TYPE_VOCAB = {
    "<pad>": 0,
    "<mask>": 1,
    "<none>": 2,
    "<ambiguous>": 3,
    "<unknown>": 4,
    "liquidity_inventory_transitions_this_update": 5,
    "liquidity_pool_transitions_this_update": 6,
    "group3_fvg_transitions_this_update": 7,
    "group3_order_block_transitions_this_update": 8,
    "group4_range_transitions_this_update": 9,
    "group4_manipulation_transitions_this_update": 10,
    "group5_entry_location_transitions_this_update": 11,
    "group5_reacceptance_transitions_this_update": 12,
    "group5_micro_bos_transitions_this_update": 13,
    "group5_path_transitions_this_update": 14,
    "group5_step_transitions_this_update": 15,
    "scene_node_delta": 16,
    "scene_edge_delta": 17,
    "scene_resolution": 18,
}
NEXT_LIFECYCLE_VOCAB = {
    "<pad>": 0,
    "<mask>": 1,
    "<none>": 2,
    "<ambiguous>": 3,
    "unknown": 4,
    "formed": 5,
    "candidate": 6,
    "active": 7,
    "confirmed": 8,
    "revised": 9,
    "mitigated": 10,
    "consumed": 11,
    "completed": 12,
    "closed": 13,
    "invalidated": 14,
    "failed": 15,
    "left": 16,
    "expired": 17,
    "censored": 18,
}
OUTCOME_BLIND_HEAD_WIDTHS = {
    "next_event_type": len(NEXT_EVENT_TYPE_VOCAB),
    "next_lifecycle": len(NEXT_LIFECYCLE_VOCAB),
    "next_event_time_bucket": 8,
    "displacement_state": 2,
    "draw_consumed": 2,
    "scale_direction_alignment": 3,
}

# No absolute OHLC value is exposed by the feature builder.
CAUSAL_CANDLE_FEATURES = (
    "log_return",
    "open_gap_atr",
    "body_atr",
    "range_atr",
    "high_from_previous_close_atr",
    "low_from_previous_close_atr",
    "body_ticks",
    "range_ticks",
    "body_to_range",
    "upper_wick_to_range",
    "lower_wick_to_range",
    "close_in_range",
    "relative_volume",
    "log_time_delta",
    "atr_baseline_valid",
    "volume_baseline_valid",
)

FORBIDDEN_MODEL_INPUT_KEYS = frozenset(
    {
        "outcome",
        "future_outcome",
        "frozen_outcome",
        "target_hit_first",
        "invalidation_hit_first",
        "deadline_hit_first",
        "mfe",
        "mae",
        "mfe_r",
        "mae_r",
        "hit_0_5r",
        "hit_1r",
        "hit_2r",
        "draw_delivered",
        "fill_outcome",
        "expire_outcome",
        "terminal_outcome_reason",
        "pnl",
        "profit",
        "realized_r",
    }
)

# These are training targets, never fields of RepresentationBatch.
ALLOWED_SELF_SUPERVISED_TARGETS = frozenset(
    {
        "next_event_type",
        "next_lifecycle",
        "next_event_time_bucket",
        "displacement_state",
        "draw_consumed",
        "scale_direction_alignment",
    }
)
NEUTRAL_MARKET_TRANSITION_KINDS = (
    "episode_created",
    "zone_registered",
    "first_pullback",
    "trigger",
    "successful_pulse",
    "terminal",
)
NEUTRAL_MARKET_LIFECYCLE_TARGETS = {
    "registered": NEXT_LIFECYCLE_VOCAB["formed"],
    "pullback": NEXT_LIFECYCLE_VOCAB["active"],
    "triggered": NEXT_LIFECYCLE_VOCAB["confirmed"],
    "terminal": NEXT_LIFECYCLE_VOCAB["completed"],
}
NEUTRAL_SPARSE_ACTIVE_TARGETS = (
    "next_lifecycle",
    "scale_direction_alignment",
)
NEUTRAL_SPARSE_DISABLED_TARGETS = (
    "next_event_type",
    "next_event_time_bucket",
    "displacement_state",
    "draw_consumed",
)
NEUTRAL_REPRESENTATION_LOSS_WEIGHTS: Mapping[str, float] = {
    "candle_reconstruction": 1.0,
    "event_reconstruction": 1.0,
    "next_event": 0.0,
    "next_lifecycle": 0.75,
    "next_event_time": 0.0,
    "displacement": 0.0,
    "draw_consumed": 0.0,
    "scale_alignment": 0.5,
    "contrastive": 0.0,
}
NEUTRAL_SPARSE_TARGET_LABEL_SOURCE = (
    "same_market_episode_next_sparse_revision_and_same_clock_global_context_v1"
)
TARGET_METADATA_KEYS = frozenset(
    {
        "case_id",
        "revision_id",
        "market_epoch_id",
        "entry_episode_id",
        "input_asof",
        "label_max_observed_at",
        "next_revision_id",
        "label_source",
    }
)


class RepresentationDataError(ValueError):
    """Raised when decision-time representation data violates causality."""


class TorchUnavailableError(RuntimeError):
    """Raised when an optional model operation is requested without PyTorch."""


def require_torch() -> None:
    """Fail closed instead of silently changing the representation model."""

    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "market representation training requires optional dependency "
            "PyTorch; install it in a separate training environment"
        )


def _aware_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RepresentationDataError(f"{name} must be timezone-aware")
    return timestamp


def _nonempty(value: Any, *, name: str) -> str:
    output = str(value).strip() if value is not None else ""
    if not output:
        raise RepresentationDataError(f"{name} must be non-empty")
    return output


def normalize_timeframe(value: Any) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "240m": "4h",
        "60m": "1h",
        "0.25h": "15m",
        "15min": "15m",
        "5min": "5m",
        "1min": "1m",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TIMEFRAMES:
        raise RepresentationDataError(
            f"unsupported timeframe {value!r}; expected one of {TIMEFRAMES}"
        )
    return normalized


def _direction_code(value: Any) -> int:
    if isinstance(value, bool):
        raise RepresentationDataError("direction cannot be boolean")
    if isinstance(value, (int, np.integer)) and int(value) in {-1, 0, 1}:
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"long", "bullish", "up", "buy", "1", "+1"}:
        return 1
    if normalized in {"short", "bearish", "down", "sell", "-1"}:
        return -1
    if normalized in {"neutral", "unknown", "none", "0"}:
        return 0
    raise RepresentationDataError(f"unsupported direction {value!r}")


def _nested_forbidden_input_paths(value: Any, *, path: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().lower()
            child = f"{path}.{key}" if path else key
            if key in FORBIDDEN_MODEL_INPUT_KEYS:
                found.append(child)
                continue
            found.extend(_nested_forbidden_input_paths(nested, path=child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            found.extend(
                _nested_forbidden_input_paths(nested, path=f"{path}[{index}]")
            )
    return tuple(found)


@dataclass(frozen=True)
class PrefixIndexRange:
    """A causal range into one epoch-bound canonical OHLCV source.

    ``tail_at_or_before`` binds snapshot-local ``0..bars`` coordinates to the
    last completed rows at a clock.  This avoids interpreting an Eye frame's
    local row zero as row zero of a longer external canonical view.
    """

    market_epoch_id: str
    timeframe: str
    canonical_source_id: str
    row_start: int
    row_end_exclusive: int
    available_at_column: str | None = None
    start_at: pd.Timestamp | None = None
    end_at: pd.Timestamp | None = None
    tail_at_or_before: pd.Timestamp | None = None
    resolve_external_rows_by_time: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "market_epoch_id",
            _nonempty(self.market_epoch_id, name="prefix.market_epoch_id"),
        )
        object.__setattr__(self, "timeframe", normalize_timeframe(self.timeframe))
        object.__setattr__(
            self,
            "canonical_source_id",
            _nonempty(self.canonical_source_id, name="prefix.canonical_source_id"),
        )
        if isinstance(self.row_start, bool) or not isinstance(
            self.row_start, (int, np.integer)
        ):
            raise RepresentationDataError("prefix.row_start must be an integer")
        if isinstance(self.row_end_exclusive, bool) or not isinstance(
            self.row_end_exclusive, (int, np.integer)
        ):
            raise RepresentationDataError(
                "prefix.row_end_exclusive must be an integer"
            )
        if self.row_start < 0 or self.row_end_exclusive <= self.row_start:
            raise RepresentationDataError("prefix row range must be non-empty")
        if self.available_at_column is not None:
            object.__setattr__(
                self,
                "available_at_column",
                _nonempty(
                    self.available_at_column,
                    name="prefix.available_at_column",
                ),
            )
        if (self.start_at is None) != (self.end_at is None):
            raise RepresentationDataError(
                "prefix start_at and end_at must be supplied together"
            )
        if self.start_at is not None and self.end_at is not None:
            start = _aware_timestamp(self.start_at, name="prefix.start_at")
            end = _aware_timestamp(self.end_at, name="prefix.end_at")
            if end < start:
                raise RepresentationDataError("prefix time boundary is reversed")
            object.__setattr__(self, "start_at", start)
            object.__setattr__(self, "end_at", end)
        if self.tail_at_or_before is not None:
            tail = _aware_timestamp(
                self.tail_at_or_before,
                name="prefix.tail_at_or_before",
            )
            if (
                self.start_at is not None
                or self.end_at is not None
                or self.resolve_external_rows_by_time
                or self.row_start != 0
            ):
                raise RepresentationDataError(
                    "tail-resolved prefix must use local row zero and no time range"
                )
            object.__setattr__(self, "tail_at_or_before", tail)
        if self.resolve_external_rows_by_time and self.start_at is None:
            raise RepresentationDataError(
                "time-resolved external prefix requires time boundaries"
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        market_epoch_id: str,
        timeframe: str | None = None,
        canonical_source_id: str | None = None,
    ) -> "PrefixIndexRange":
        return cls(
            market_epoch_id=str(payload.get("market_epoch_id", market_epoch_id)),
            timeframe=str(payload.get("timeframe", timeframe or "")),
            canonical_source_id=str(
                payload.get(
                    "canonical_source_id",
                    payload.get("source_id", canonical_source_id or ""),
                )
            ),
            row_start=payload.get("row_start", payload.get("frame_row_start")),
            row_end_exclusive=payload.get(
                "row_end_exclusive",
                payload.get("row_end", payload.get("frame_row_end_exclusive")),
            ),
            available_at_column=payload.get("available_at_column"),
            start_at=payload.get("start_at"),
            end_at=payload.get("end_at"),
            tail_at_or_before=payload.get("tail_at_or_before"),
            # Recorder frame rows are bounded snapshot-local coordinates.  They
            # must never be treated as rows in an external canonical view.
            resolve_external_rows_by_time=bool(
                "frame_row_start" in payload
                or payload.get("boundary_semantics")
                == "[start_at,end_at]_completed_prefix"
            ),
        )


@dataclass(frozen=True)
class EventGraphObservation:
    """Contemporaneous event node plus its typed graph relations."""

    event_id: str
    event_type: str
    lifecycle: str
    observed_at: pd.Timestamp
    active_since: pd.Timestamp
    duration_seconds: float
    relation_types: tuple[str, ...] = ()
    direction: int = 0
    scale: str = "unknown"
    market_epoch_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _nonempty(self.event_id, name="event_id"))
        object.__setattr__(
            self, "event_type", _nonempty(self.event_type, name="event_type")
        )
        object.__setattr__(
            self, "lifecycle", _nonempty(self.lifecycle, name="event.lifecycle")
        )
        object.__setattr__(
            self,
            "observed_at",
            _aware_timestamp(self.observed_at, name="event.observed_at"),
        )
        object.__setattr__(
            self,
            "active_since",
            _aware_timestamp(self.active_since, name="event.active_since"),
        )
        if self.active_since > self.observed_at:
            raise RepresentationDataError(
                "event.active_since cannot be after event.observed_at"
            )
        if not math.isfinite(float(self.duration_seconds)) or self.duration_seconds < 0:
            raise RepresentationDataError(
                "event.duration_seconds must be finite and non-negative"
            )
        object.__setattr__(
            self,
            "relation_types",
            tuple(sorted({_nonempty(item, name="event.relation") for item in self.relation_types})),
        )
        object.__setattr__(self, "direction", _direction_code(self.direction))
        object.__setattr__(self, "scale", str(self.scale).strip().lower() or "unknown")
        if self.market_epoch_id:
            object.__setattr__(
                self,
                "market_epoch_id",
                _nonempty(self.market_epoch_id, name="event.market_epoch_id"),
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        asof: pd.Timestamp,
        market_epoch_id: str,
    ) -> "EventGraphObservation":
        observed_at = payload.get("observed_at", payload.get("active_at", asof))
        active_since = payload.get(
            "active_since", payload.get("formed_at", observed_at)
        )
        observed = _aware_timestamp(observed_at, name="event.observed_at")
        active = _aware_timestamp(active_since, name="event.active_since")
        raw_duration = payload.get("duration_seconds")
        duration = (
            float(raw_duration)
            if raw_duration is not None
            else max(0.0, float((observed - active).total_seconds()))
        )
        relations = payload.get("relation_types", payload.get("relations", ()))
        if isinstance(relations, str):
            relations = (relations,)
        return cls(
            event_id=str(payload.get("event_id", payload.get("id", ""))),
            event_type=str(payload.get("event_type", payload.get("kind", ""))),
            lifecycle=str(payload.get("lifecycle", "unknown")),
            observed_at=observed,
            active_since=active,
            duration_seconds=duration,
            relation_types=tuple(str(item) for item in relations),
            direction=_direction_code(payload.get("direction", 0)),
            scale=str(payload.get("scale", payload.get("timeframe", "unknown"))),
            market_epoch_id=str(payload.get("market_epoch_id", market_epoch_id)),
        )


def _event_identity(payload: Mapping[str, Any], *, fallback: str) -> str:
    for name in (
        "event_id",
        "location_id",
        "path_id",
        "step_id",
        "fvg_id",
        "order_block_id",
        "manipulation_id",
        "range_id",
        "reacceptance_id",
        "reference_id",
        "pool_id",
        "displacement_id",
        "source_displacement_id",
        "entity_id",
        "id",
    ):
        value = payload.get(name)
        if isinstance(value, str) and value:
            return value
    return fallback


def _typed_transition_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        marker_keys = {
            "event_id",
            "kind",
            "event_type",
            "lifecycle",
            "observed_at",
            "active_at",
            "formed_at",
            "location_id",
            "path_id",
            "step_id",
        }
        if marker_keys.intersection(value):
            yield value
            return
        for nested in value.values():
            yield from _typed_transition_mappings(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _typed_transition_mappings(nested)


def _categorical_case_tokens(
    value: Any,
    *,
    section: str,
    path: str = "",
) -> Iterable[str]:
    """Yield schema/value categories while excluding prices and identities."""

    if isinstance(value, Mapping):
        for key, nested in sorted(value.items(), key=lambda item: str(item[0])):
            normalized = str(key).lower()
            child = f"{path}.{normalized}" if path else normalized
            if (
                normalized in {"asof", "timestamp", "datetime"}
                or normalized.endswith("_at")
                or normalized.endswith("_timestamp")
            ):
                continue
            if normalized.endswith("_id") or normalized.endswith("_ids"):
                continue
            if any(part in normalized for part in ("price", "entry", "target", "stop")):
                continue
            if any(marker in normalized for marker in CONTROL_TOKEN_KEY_MARKERS):
                continue
            yield from _categorical_case_tokens(nested, section=section, path=child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        yield f"{section}:{path}:count={min(len(value), 32)}"
    elif isinstance(value, (bool, str)) or value is None:
        rendered = "none" if value is None else str(value).strip().lower()
        if isinstance(value, str) and ISO_TIMESTAMP_TOKEN_RE.fullmatch(rendered):
            return
        if rendered and len(rendered) <= 80:
            yield f"{section}:{path}={rendered}"


def _case_event_observations(
    raw_events: Any,
    *,
    payload: Mapping[str, Any],
    asof: pd.Timestamp,
    market_epoch_id: str,
    case_direction: int,
) -> tuple[EventGraphObservation, ...]:
    """Adapt direct event arrays or the causal-case-library event bundle."""

    if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):
        if not all(isinstance(item, Mapping) for item in raw_events):
            raise RepresentationDataError("every case event must be an object")
        direct = tuple(
            EventGraphObservation.from_mapping(
                item,
                asof=asof,
                market_epoch_id=market_epoch_id,
            )
            for item in raw_events
        )
        if direct:
            return direct
    elif not isinstance(raw_events, Mapping):
        raise RepresentationDataError("case.events must be an array or case-library bundle")

    output: list[EventGraphObservation] = []
    if isinstance(raw_events, Mapping):
        transition_kinds = raw_events.get(
            "market_episode_transition_kinds",
            (),
        )
        if (
            not isinstance(transition_kinds, Sequence)
            or isinstance(transition_kinds, (str, bytes))
            or any(not isinstance(item, str) or not item for item in transition_kinds)
        ):
            raise RepresentationDataError(
                "market episode transition kinds must be an array of identities"
            )
        market_episode = payload.get("market_episode", {})
        episode_lifecycle = (
            str(market_episode.get("lifecycle", "observed"))
            if isinstance(market_episode, Mapping)
            else "observed"
        )
        for index, transition_kind in enumerate(transition_kinds):
            output.append(
                EventGraphObservation(
                    event_id=(
                        f"market-episode-transition:{index}:{transition_kind}"
                    ),
                    event_type=f"market_episode_transition:{transition_kind}",
                    lifecycle=episode_lifecycle,
                    observed_at=asof,
                    active_since=asof,
                    duration_seconds=0.0,
                    relation_types=(
                        "neutral_market_episode",
                        f"transition_kind:{transition_kind}",
                    ),
                    direction=case_direction,
                    scale="cross_scale",
                    market_epoch_id=market_epoch_id,
                )
            )
        transition = raw_events.get("observation_transition", {})
        collections = transition.get("collections", {}) if isinstance(transition, Mapping) else {}
        if isinstance(collections, Mapping):
            for collection_name, values in sorted(
                collections.items(), key=lambda item: str(item[0])
            ):
                for index, event in enumerate(_typed_transition_mappings(values)):
                    event_payload = dict(event)
                    event_payload.setdefault(
                        "event_id", f"{collection_name}:{index}:{_event_identity(event, fallback='event')}"
                    )
                    event_payload.setdefault(
                        "event_type",
                        event_payload.get("kind", str(collection_name)),
                    )
                    event_payload.setdefault("lifecycle", "active")
                    event_payload.setdefault("observed_at", asof)
                    event_payload.setdefault("active_since", event_payload.get("formed_at", asof))
                    event_payload.setdefault("direction", case_direction)
                    relations = event_payload.get("relation_types", ())
                    if isinstance(relations, str):
                        relations = (relations,)
                    event_payload["relation_types"] = tuple(relations) + (
                        f"transition_collection:{collection_name}",
                    )
                    output.append(
                        EventGraphObservation.from_mapping(
                            event_payload,
                            asof=asof,
                            market_epoch_id=market_epoch_id,
                        )
                    )

        for delta_name, lifecycle in (
            ("added_event_ids", "added"),
            ("invalidated_event_ids", "invalidated"),
        ):
            values = raw_events.get(delta_name, ())
            count = len(values) if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else 0
            if count:
                output.append(
                    EventGraphObservation(
                        event_id=f"case-delta:{delta_name}",
                        event_type=f"case_delta:{delta_name}:count={min(count, 64)}",
                        lifecycle=lifecycle,
                        observed_at=asof,
                        active_since=asof,
                        duration_seconds=0.0,
                        relation_types=("scene_graph_delta",),
                        direction=case_direction,
                        scale="cross_scale",
                        market_epoch_id=market_epoch_id,
                    )
                )

    graph = payload.get("graph", {})
    if isinstance(graph, Mapping):
        descriptors = graph.get("relation_descriptors", ())
        descriptors_complete = graph.get("relation_descriptors_complete")
        changed_edge_ids = tuple(graph.get("added_edge_ids", ())) + tuple(
            graph.get("revised_edge_ids", ())
        )
        if descriptors_complete is False or (
            changed_edge_ids and descriptors_complete is not True
        ):
            raise RepresentationDataError(
                "Scene Graph relation descriptors are incomplete at case asof"
            )
        if descriptors is not None and (
            not isinstance(descriptors, Sequence)
            or isinstance(descriptors, (str, bytes))
        ):
            raise RepresentationDataError(
                "Scene Graph relation descriptors must be an array"
            )
        for index, descriptor in enumerate(descriptors or ()):
            if not isinstance(descriptor, Mapping):
                raise RepresentationDataError(
                    "Scene Graph relation descriptor must be an object"
                )
            source = descriptor.get("source", {})
            target = descriptor.get("target", {})
            if not isinstance(source, Mapping) or not isinstance(target, Mapping):
                raise RepresentationDataError(
                    "Scene Graph relation endpoints must be typed objects"
                )
            relation = str(descriptor.get("relation", "unknown")).strip().lower()
            source_kind = str(source.get("kind", "unknown")).strip().lower()
            source_role = str(source.get("role", "unknown")).strip().lower()
            target_kind = str(target.get("kind", "unknown")).strip().lower()
            target_role = str(target.get("role", "unknown")).strip().lower()
            source_scale = str(
                source.get("structural_scale", source.get("timeframe", "unknown"))
            ).strip().lower()
            target_scale = str(
                target.get("structural_scale", target.get("timeframe", "unknown"))
            ).strip().lower()
            observed_at = _aware_timestamp(
                descriptor.get("observed_at", asof),
                name="case.graph_relation.observed_at",
            )
            output.append(
                EventGraphObservation(
                    event_id=f"graph-relation:{index}:{descriptor.get('edge_id', 'audit')}",
                    event_type=(
                        f"scene_relation:{relation}|source_kind:{source_kind}"
                        f"|source_role:{source_role}|target_kind:{target_kind}"
                        f"|target_role:{target_role}"
                    ),
                    lifecycle=str(descriptor.get("lifecycle", "unknown")),
                    observed_at=observed_at,
                    active_since=observed_at,
                    duration_seconds=0.0,
                    relation_types=(
                        f"relation:{relation}",
                        f"source:{source_kind}:{source_role}:{source_scale}",
                        f"target:{target_kind}:{target_role}:{target_scale}",
                    ),
                    direction=case_direction,
                    scale=f"{source_scale}->{target_scale}",
                    market_epoch_id=market_epoch_id,
                )
            )
        for field_name in (
            "added_node_ids",
            "revised_node_ids",
            "added_edge_ids",
            "revised_edge_ids",
            "resolution_event_ids",
        ):
            values = graph.get(field_name, ())
            count = len(values) if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else 0
            if count:
                output.append(
                    EventGraphObservation(
                        event_id=f"graph-delta:{field_name}",
                        event_type=f"graph_delta:{field_name}:count={min(count, 64)}",
                        lifecycle="active",
                        observed_at=asof,
                        active_since=asof,
                        duration_seconds=0.0,
                        relation_types=(field_name, "scene_graph"),
                        direction=case_direction,
                        scale="cross_scale",
                        market_epoch_id=market_epoch_id,
                    )
                )

    for section in (
        "context_thesis",
        "entry_episode",
        "neutral_global_context",
        "market_episode",
        "authority",
        "scale_relations",
        "draw",
        "blockers",
        "ambiguities",
        "supporting_evidence",
        "opposing_evidence",
    ):
        for index, token in enumerate(_categorical_case_tokens(payload.get(section), section=section)):
            output.append(
                EventGraphObservation(
                    event_id=f"case-state:{section}:{index}",
                    event_type=token,
                    lifecycle="observed",
                    observed_at=asof,
                    active_since=asof,
                    duration_seconds=0.0,
                    relation_types=(f"case_state:{section}",),
                    direction=case_direction,
                    scale="cross_scale",
                    market_epoch_id=market_epoch_id,
                )
            )

    if not output:
        output.append(
            EventGraphObservation(
                event_id="case-revision-observed",
                event_type=f"case_revision:{payload.get('revision_stage', 'unknown')}",
                lifecycle="observed",
                observed_at=asof,
                active_since=asof,
                duration_seconds=0.0,
                relation_types=("entry_episode_revision",),
                direction=case_direction,
                scale="cross_scale",
                market_epoch_id=market_epoch_id,
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class RepresentationCase:
    """Outcome-free input view of one causal episode revision.

    ``entry_episode_id`` remains the legacy compatibility identity.  Neutral
    input rows also bind ``market_episode_id`` explicitly; when omitted by an
    old caller it deterministically aliases the EntryEpisode identity.
    """

    case_id: str
    revision_id: str
    market_epoch_id: str
    context_thesis_id: str
    entry_episode_id: str
    asof: pd.Timestamp
    direction: int
    regime: str
    prefixes: Mapping[str, PrefixIndexRange]
    events: tuple[EventGraphObservation, ...]
    market_episode_id: str = ""
    revision_stage: str = "episode_created"
    revision_index: int = 0
    stage_identity: str = ""
    mechanism_label: str = "unknown"
    authority_direction: int = 0
    canonical_source_path: str = ""
    symbol: str = ""
    instrument_id: int = -1
    entry_location_id: str = ""
    entry_path_id: str = ""
    transition_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "revision_id",
            "market_epoch_id",
            "context_thesis_id",
            "entry_episode_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "market_episode_id",
            _nonempty(
                self.market_episode_id or self.entry_episode_id,
                name="market_episode_id",
            ),
        )
        object.__setattr__(self, "asof", _aware_timestamp(self.asof, name="case.asof"))
        object.__setattr__(self, "direction", _direction_code(self.direction))
        object.__setattr__(self, "regime", str(self.regime).strip().lower() or "unknown")
        stage = str(self.revision_stage).strip().lower()
        if stage not in CASE_REVISION_STAGES:
            raise RepresentationDataError(
                f"unsupported case revision_stage {self.revision_stage!r}"
            )
        object.__setattr__(self, "revision_stage", stage)
        if isinstance(self.revision_index, bool) or not isinstance(
            self.revision_index, (int, np.integer)
        ) or self.revision_index < 0:
            raise RepresentationDataError("case revision_index must be non-negative")
        object.__setattr__(self, "revision_index", int(self.revision_index))
        stage_identity = str(self.stage_identity).strip()
        object.__setattr__(
            self,
            "stage_identity",
            stage_identity or f"{stage}:{self.revision_id}",
        )
        object.__setattr__(
            self,
            "mechanism_label",
            str(self.mechanism_label).strip().lower() or "unknown",
        )
        object.__setattr__(
            self,
            "authority_direction",
            _direction_code(self.authority_direction),
        )
        source_path = str(self.canonical_source_path).strip()
        if source_path and not Path(source_path).is_absolute():
            raise RepresentationDataError("case canonical_source_path must be absolute")
        object.__setattr__(self, "canonical_source_path", source_path)
        object.__setattr__(self, "symbol", str(self.symbol).strip())
        if isinstance(self.instrument_id, bool) or not isinstance(
            self.instrument_id, (int, np.integer)
        ):
            raise RepresentationDataError("case instrument_id must be an integer")
        object.__setattr__(self, "instrument_id", int(self.instrument_id))
        location_id = str(self.entry_location_id).strip()
        path_id = str(self.entry_path_id).strip()
        if bool(location_id) != bool(path_id):
            raise RepresentationDataError(
                "neutral case physical location/path must be supplied together"
            )
        raw_kinds = self.transition_kinds
        if isinstance(raw_kinds, (str, bytes)) or not isinstance(
            raw_kinds, Sequence
        ):
            raise RepresentationDataError("neutral transition_kinds must be an array")
        kinds = tuple(str(value).strip().lower() for value in raw_kinds)
        expected_kinds = tuple(
            kind for kind in NEUTRAL_MARKET_TRANSITION_KINDS if kind in kinds
        )
        if kinds and (kinds != expected_kinds or len(kinds) != len(set(kinds))):
            raise RepresentationDataError("neutral transition_kinds are invalid")
        object.__setattr__(self, "entry_location_id", location_id)
        object.__setattr__(self, "entry_path_id", path_id)
        object.__setattr__(self, "transition_kinds", kinds)

        normalized: dict[str, PrefixIndexRange] = {}
        for raw_timeframe, prefix in self.prefixes.items():
            timeframe = normalize_timeframe(raw_timeframe)
            if prefix.timeframe != timeframe:
                raise RepresentationDataError(
                    "prefix mapping key and prefix.timeframe disagree"
                )
            if prefix.market_epoch_id != self.market_epoch_id:
                raise RepresentationDataError(
                    "canonical prefix cannot cross the case market epoch"
                )
            normalized[timeframe] = prefix
        missing = sorted(set(TIMEFRAMES) - set(normalized))
        extra = sorted(set(normalized) - set(TIMEFRAMES))
        if missing or extra:
            raise RepresentationDataError(
                f"case must bind exactly five timeframes; missing={missing}, extra={extra}"
            )
        object.__setattr__(self, "prefixes", normalized)

        ordered_events = tuple(sorted(self.events, key=lambda item: (item.observed_at, item.event_id)))
        for event in ordered_events:
            if event.observed_at > self.asof or event.active_since > self.asof:
                raise RepresentationDataError(
                    "all event/graph input timestamps must be <= case.asof"
                )
            if event.market_epoch_id and event.market_epoch_id != self.market_epoch_id:
                raise RepresentationDataError(
                    "event/graph input cannot cross the case market epoch"
                )
        object.__setattr__(self, "events", ordered_events)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RepresentationCase":
        forbidden = sorted(_nested_forbidden_input_paths(payload))
        if forbidden:
            raise RepresentationDataError(
                "pass only the case input/revision view to the representation model; "
                f"future outcome fields are forbidden: {forbidden}"
            )
        market_epoch_id = str(payload.get("market_epoch_id", ""))
        asof = _aware_timestamp(
            payload.get("asof", payload.get("decision_at")), name="case.asof"
        )
        if "decision_at" in payload and _aware_timestamp(
            payload["decision_at"], name="case.decision_at"
        ) != asof:
            raise RepresentationDataError("case decision_at must equal asof")
        if "normalization_cutoff_at" in payload:
            cutoff = _aware_timestamp(
                payload["normalization_cutoff_at"],
                name="case.normalization_cutoff_at",
            )
            if cutoff >= asof:
                raise RepresentationDataError(
                    "normalization cutoff must be strictly before decision asof"
                )
        if "normalization_policy" in payload and payload["normalization_policy"] != (
            "bars_with_end_strictly_before_decision_asof"
        ):
            raise RepresentationDataError("case normalization policy is unsupported")
        raw_prefixes = payload.get("prefixes", payload.get("ohlcv_prefixes"))
        source_id = str(payload.get("canonical_source_id", ""))
        if isinstance(raw_prefixes, Mapping):
            prefix_items = tuple(raw_prefixes.items())
        elif isinstance(raw_prefixes, Sequence) and not isinstance(
            raw_prefixes, (str, bytes)
        ):
            prefix_items = tuple(
                (raw.get("timeframe", ""), raw)
                for raw in raw_prefixes
                if isinstance(raw, Mapping)
            )
            if len(prefix_items) != len(raw_prefixes):
                raise RepresentationDataError("every case prefix must be an object")
        else:
            raise RepresentationDataError(
                "case.prefixes must be a timeframe mapping or array"
            )
        prefixes: dict[str, PrefixIndexRange] = {}
        for timeframe, raw in prefix_items:
            if not isinstance(raw, Mapping):
                raise RepresentationDataError("every case prefix must be an object")
            normalized_timeframe = normalize_timeframe(timeframe)
            if normalized_timeframe in prefixes:
                raise RepresentationDataError("case contains a duplicate timeframe prefix")
            if "end_at" in raw and _aware_timestamp(
                raw["end_at"], name="case.prefix.end_at"
            ) > asof:
                raise RepresentationDataError("case prefix end exceeds decision asof")
            if "tail_at_or_before" in raw and _aware_timestamp(
                raw["tail_at_or_before"],
                name="case.prefix.tail_at_or_before",
            ) > asof:
                raise RepresentationDataError(
                    "case prefix tail clock exceeds decision asof"
                )
            prefixes[normalized_timeframe] = PrefixIndexRange.from_mapping(
                raw,
                market_epoch_id=market_epoch_id,
                timeframe=str(timeframe),
                canonical_source_id=source_id,
            )
        if len(prefixes) != len(prefix_items):
            raise RepresentationDataError("every case prefix must be an object")
        raw_events = payload.get("events", payload.get("event_graph", ()))
        events = _case_event_observations(
            raw_events,
            payload=payload,
            asof=asof,
            market_epoch_id=market_epoch_id,
            case_direction=_direction_code(payload.get("direction", 0)),
        )
        authority_payload = payload.get("authority")
        raw_authority_direction: Any = payload.get("authority_direction", 0)
        if isinstance(authority_payload, Mapping):
            raw_authority_direction = authority_payload.get(
                "authority_direction", raw_authority_direction
            )
        raw_market_episode = payload.get("market_episode", {})
        market_episode = (
            raw_market_episode if isinstance(raw_market_episode, Mapping) else {}
        )
        return cls(
            case_id=str(payload.get("case_id", "")),
            revision_id=str(
                payload.get("revision_id", payload.get("case_revision_id", ""))
            ),
            market_epoch_id=market_epoch_id,
            context_thesis_id=str(payload.get("context_thesis_id", "")),
            entry_episode_id=str(
                payload.get(
                    "entry_episode_id",
                    payload.get("market_episode_id", ""),
                )
            ),
            asof=asof,
            direction=_direction_code(payload.get("direction", 0)),
            regime=str(payload.get("regime", "unknown")),
            prefixes=prefixes,
            events=events,
            market_episode_id=str(
                payload.get(
                    "market_episode_id",
                    payload.get("entry_episode_id", ""),
                )
            ),
            revision_stage=str(payload.get("revision_stage", "episode_created")),
            revision_index=int(payload.get("revision_index", 0)),
            stage_identity=str(payload.get("stage_identity", "")),
            mechanism_label=str(payload.get("mechanism_label", "unknown")),
            authority_direction=_direction_code(raw_authority_direction),
            canonical_source_path=str(payload.get("source_path", "")),
            symbol=str(payload.get("symbol", "")),
            instrument_id=int(payload.get("instrument_id", -1)),
            entry_location_id=str(
                payload.get(
                    "entry_location_id",
                    market_episode.get("entry_location_id", ""),
                )
            ),
            entry_path_id=str(
                payload.get(
                    "entry_path_id",
                    market_episode.get("entry_path_id", ""),
                )
            ),
            transition_kinds=tuple(
                payload.get(
                    "transition_kinds",
                    market_episode.get("transition_kinds", ()),
                )
            ),
        )


@dataclass(frozen=True)
class CanonicalSourceKey:
    market_epoch_id: str
    timeframe: str
    canonical_source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "market_epoch_id",
            _nonempty(self.market_epoch_id, name="source.market_epoch_id"),
        )
        object.__setattr__(self, "timeframe", normalize_timeframe(self.timeframe))
        object.__setattr__(
            self,
            "canonical_source_id",
            _nonempty(self.canonical_source_id, name="source.canonical_source_id"),
        )


@dataclass(frozen=True)
class CausalPrefixFeatures:
    values: np.ndarray
    feature_names: tuple[str, ...]
    row_start: int
    row_end_exclusive: int
    last_available_at: pd.Timestamp

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise RepresentationDataError("prefix feature matrix has invalid shape")
        if not np.isfinite(values).all():
            raise RepresentationDataError("prefix feature matrix must be finite")
        object.__setattr__(self, "values", values)


class CanonicalOHLCVStore:
    """Read-only resolver for epoch-bound canonical OHLCV frames.

    Frame indices must be causal availability/completion timestamps.  If a
    source index is bar-open time, each PrefixIndexRange must name an explicit
    timezone-aware ``available_at_column`` instead.
    """

    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        frames: Mapping[CanonicalSourceKey | tuple[str, str, str], pd.DataFrame],
        *,
        tick_sizes: Mapping[
            CanonicalSourceKey | tuple[str, str, str], float
        ],
        availability_bindings: Mapping[
            CanonicalSourceKey | tuple[str, str, str], str
        ],
        normalization_window: int = 20,
    ) -> None:
        if normalization_window < 2:
            raise RepresentationDataError("normalization_window must be at least 2")
        normalized_frames: dict[CanonicalSourceKey, pd.DataFrame] = {}
        normalized_ticks: dict[CanonicalSourceKey, float] = {}
        normalized_availability: dict[CanonicalSourceKey, str] = {}
        for raw_key, frame in frames.items():
            key = raw_key if isinstance(raw_key, CanonicalSourceKey) else CanonicalSourceKey(*raw_key)
            if not isinstance(frame, pd.DataFrame):
                raise RepresentationDataError("canonical source must be a DataFrame")
            normalized_frames[key] = frame
        for raw_key, raw_tick in tick_sizes.items():
            key = raw_key if isinstance(raw_key, CanonicalSourceKey) else CanonicalSourceKey(*raw_key)
            tick = float(raw_tick)
            if not math.isfinite(tick) or tick <= 0:
                raise RepresentationDataError("tick size must be finite and positive")
            normalized_ticks[key] = tick
        for raw_key, raw_binding in availability_bindings.items():
            key = raw_key if isinstance(raw_key, CanonicalSourceKey) else CanonicalSourceKey(*raw_key)
            binding = _nonempty(raw_binding, name="canonical.availability_binding")
            normalized_availability[key] = binding
        if not (
            set(normalized_frames)
            == set(normalized_ticks)
            == set(normalized_availability)
        ):
            raise RepresentationDataError(
                "canonical frames, ticks and explicit availability bindings "
                "must have identical keys"
            )
        self._frames = normalized_frames
        self._tick_sizes = normalized_ticks
        self._availability_bindings = normalized_availability
        self.normalization_window = int(normalization_window)

    @staticmethod
    def _availability(
        frame: pd.DataFrame,
        *,
        binding: str,
        source: CanonicalSourceKey,
    ) -> pd.DatetimeIndex:
        raw: Any
        if binding == BAR_END_INDEX_BINDING:
            raw = frame.index
        else:
            if binding not in frame:
                raise RepresentationDataError(
                    f"canonical source {source} omits available-at column {binding!r}"
                )
            raw = frame[binding]
        try:
            index = pd.DatetimeIndex(pd.to_datetime(raw))
        except (TypeError, ValueError) as exc:
            raise RepresentationDataError(
                f"canonical source {source} has invalid availability timestamps"
            ) from exc
        if index.tz is None:
            raise RepresentationDataError(
                f"canonical source {source} availability timestamps must be aware"
            )
        if not index.is_monotonic_increasing or not index.is_unique:
            raise RepresentationDataError(
                f"canonical source {source} availability timestamps must be sorted and unique"
            )
        return index

    def read_prefix(
        self,
        prefix: PrefixIndexRange,
        *,
        asof: pd.Timestamp,
    ) -> CausalPrefixFeatures:
        key = CanonicalSourceKey(
            prefix.market_epoch_id,
            prefix.timeframe,
            prefix.canonical_source_id,
        )
        frame = self._frames.get(key)
        if frame is None:
            raise RepresentationDataError(f"canonical prefix source is not registered: {key}")
        missing = sorted(set(self.REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise RepresentationDataError(
                f"canonical source {key} omits OHLCV columns: {missing}"
            )
        availability = self._availability(
            frame,
            binding=self._availability_bindings[key],
            source=key,
        )
        if (
            prefix.available_at_column is not None
            and prefix.available_at_column != self._availability_bindings[key]
        ):
            raise RepresentationDataError(
                "case prefix and canonical availability binding disagree"
            )
        causal_asof = _aware_timestamp(asof, name="prefix.asof")
        if prefix.tail_at_or_before is not None:
            if prefix.tail_at_or_before > causal_asof:
                raise RepresentationDataError(
                    "canonical prefix tail clock exceeds case.asof"
                )
            row_end_exclusive = int(
                availability.searchsorted(
                    prefix.tail_at_or_before,
                    side="right",
                )
            )
            local_width = prefix.row_end_exclusive - prefix.row_start
            row_start = row_end_exclusive - local_width
            if row_start < 0:
                raise RepresentationDataError(
                    "canonical view has fewer completed rows than the Eye prefix"
                )
        elif prefix.resolve_external_rows_by_time:
            assert prefix.start_at is not None and prefix.end_at is not None
            row_start = int(availability.searchsorted(prefix.start_at, side="left"))
            row_end_exclusive = int(
                availability.searchsorted(prefix.end_at, side="right")
            )
        else:
            row_start = prefix.row_start
            row_end_exclusive = prefix.row_end_exclusive
        if row_end_exclusive > len(frame):
            raise RepresentationDataError(
                "canonical prefix row_end_exclusive exceeds source length"
            )
        if row_end_exclusive <= row_start:
            raise RepresentationDataError(
                "canonical time boundary resolves to an empty prefix"
            )
        selected_times = availability[row_start:row_end_exclusive]
        if len(selected_times) == 0 or bool((selected_times > causal_asof).any()):
            raise RepresentationDataError(
                "all canonical OHLCV input availability times must be <= case.asof"
            )
        selected = frame.iloc[row_start:row_end_exclusive]
        values = self._causal_features(
            selected,
            availability=selected_times,
            tick_size=self._tick_sizes[key],
        )
        return CausalPrefixFeatures(
            values=values,
            feature_names=CAUSAL_CANDLE_FEATURES,
            row_start=row_start,
            row_end_exclusive=row_end_exclusive,
            last_available_at=selected_times[-1],
        )

    def _causal_features(
        self,
        frame: pd.DataFrame,
        *,
        availability: pd.DatetimeIndex,
        tick_size: float,
    ) -> np.ndarray:
        numeric = frame.loc[:, self.REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise RepresentationDataError("canonical OHLCV values must be finite")
        open_ = numeric["open"].astype(float)
        high = numeric["high"].astype(float)
        low = numeric["low"].astype(float)
        close = numeric["close"].astype(float)
        volume = numeric["volume"].astype(float)
        if bool((low > high).any()) or bool(((open_ < low) | (open_ > high)).any()):
            raise RepresentationDataError("canonical OHLC geometry is invalid")
        if bool(((close < low) | (close > high)).any()) or bool((close <= 0).any()):
            raise RepresentationDataError("canonical OHLC geometry is invalid")
        if bool((volume < 0).any()):
            raise RepresentationDataError("canonical volume cannot be negative")

        previous_close = close.shift(1)
        true_range = pd.concat(
            (
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ),
            axis=1,
        ).max(axis=1)
        # The current candle is deliberately excluded from both baselines.
        prior_atr = (
            true_range.rolling(self.normalization_window, min_periods=2)
            .mean()
            .shift(1)
        )
        prior_volume = (
            volume.rolling(self.normalization_window, min_periods=2)
            .median()
            .shift(1)
        )
        atr_valid = prior_atr.notna() & (prior_atr > 0)
        volume_valid = prior_volume.notna() & (prior_volume > 0)
        safe_atr = prior_atr.where(atr_valid)
        safe_volume = prior_volume.where(volume_valid)

        candle_range = high - low
        body = close - open_
        body_abs = body.abs()
        upper_wick = high - pd.concat((open_, close), axis=1).max(axis=1)
        lower_wick = pd.concat((open_, close), axis=1).min(axis=1) - low
        safe_range = candle_range.where(candle_range > 0)
        safe_previous = previous_close.where(previous_close > 0)
        time_seconds = pd.Series(availability.view("i8"), index=frame.index).diff() / 1e9
        positive_deltas = time_seconds.where(time_seconds > 0)

        columns = (
            np.log(close / safe_previous),
            (open_ - previous_close) / safe_atr,
            body / safe_atr,
            candle_range / safe_atr,
            (high - previous_close) / safe_atr,
            (low - previous_close) / safe_atr,
            body / tick_size,
            candle_range / tick_size,
            body_abs / safe_range,
            upper_wick / safe_range,
            lower_wick / safe_range,
            (close - low) / safe_range,
            volume / safe_volume,
            np.log1p(positive_deltas),
            atr_valid.astype(float),
            volume_valid.astype(float),
        )
        output = pd.concat(columns, axis=1).to_numpy(dtype=np.float64)
        output = np.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
        # Heavy tails should not let one malformed-but-finite bar dominate.
        output = np.clip(output, -50.0, 50.0).astype(np.float32, copy=False)
        return output


def _stable_token(value: str, *, buckets: int) -> int:
    if buckets <= 2:
        raise RepresentationDataError("token bucket count must exceed reserved IDs")
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return 2 + (int.from_bytes(digest[:8], "big") % (buckets - 2))


def _stable_group(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:7], "big")  # exact in signed int64


def _is_direct_label_source_event(event: EventGraphObservation) -> bool:
    """Identify encoder tokens that can trivially reveal an evaluation label."""

    event_type = event.event_type.strip().lower()
    relations = {value.strip().lower() for value in event.relation_types}
    return bool(
        "case_state:scale_relations" in relations
        or (
            "case_state:neutral_global_context" in relations
            and event_type.startswith(
                "neutral_global_context:scale_relation_details."
            )
        )
        or any(
            marker in event_type
            for marker in (
                "playbook=",
                "observable_regime=",
                "mechanism_label=",
            )
        )
    )


@dataclass(frozen=True)
class PreparedRepresentationCase:
    case: RepresentationCase
    timeframe_features: Mapping[str, np.ndarray]
    feature_max_at: pd.Timestamp
    event_type_ids: np.ndarray
    lifecycle_ids: np.ndarray
    relation_ids: np.ndarray
    scale_ids: np.ndarray
    event_numeric: np.ndarray
    direct_label_source_event_mask: np.ndarray | None = None
    label_sources_masked: bool = False
    neutral_preprocessing_version: str | None = None
    neutral_preprocessing_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_max_at",
            _aware_timestamp(self.feature_max_at, name="prepared.feature_max_at"),
        )
        if self.feature_max_at > self.case.asof:
            raise RepresentationDataError(
                "prepared feature_max_at cannot be after decision asof"
            )
        if set(self.timeframe_features) != set(TIMEFRAMES):
            raise RepresentationDataError("prepared case must contain all five timeframes")
        for timeframe, values in self.timeframe_features.items():
            matrix = np.asarray(values, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] != len(CAUSAL_CANDLE_FEATURES):
                raise RepresentationDataError(
                    f"prepared {timeframe} features have invalid shape"
                )
            if not np.isfinite(matrix).all():
                raise RepresentationDataError("prepared candle features must be finite")
        event_count = len(self.event_type_ids)
        if event_count < 1:
            raise RepresentationDataError(
                "prepared case requires at least one contemporaneous event token"
            )
        if any(
            len(values) != event_count
            for values in (self.lifecycle_ids, self.relation_ids, self.scale_ids)
        ):
            raise RepresentationDataError("prepared event token arrays disagree")
        if np.asarray(self.event_numeric).shape != (event_count, 4):
            raise RepresentationDataError("prepared event numeric features are invalid")
        source_mask = (
            np.zeros(event_count, dtype=bool)
            if self.direct_label_source_event_mask is None
            else np.asarray(self.direct_label_source_event_mask, dtype=bool)
        )
        if source_mask.shape != (event_count,):
            raise RepresentationDataError(
                "prepared direct-label-source mask has invalid shape"
            )
        object.__setattr__(self, "direct_label_source_event_mask", source_mask)
        object.__setattr__(self, "label_sources_masked", bool(self.label_sources_masked))
        version = self.neutral_preprocessing_version
        digest = self.neutral_preprocessing_sha256
        if (version is None) != (digest is None):
            raise RepresentationDataError(
                "prepared neutral preprocessing identity is incomplete"
            )
        if version is not None:
            expected = neutral_direct_source_preprocessing_identity()
            if (
                version != expected["protocol"]["protocol_version"]
                or digest != expected["sha256"]
                or not self.label_sources_masked
                or bool(source_mask.any())
            ):
                raise RepresentationDataError(
                    "prepared neutral preprocessing identity is invalid"
                )


def prepare_representation_case(
    case: RepresentationCase,
    store: CanonicalOHLCVStore,
    *,
    event_type_buckets: int = 512,
    lifecycle_buckets: int = 64,
    relation_buckets: int = 256,
    scale_buckets: int = 32,
) -> PreparedRepresentationCase:
    """Resolve references without reading one row after the decision time."""

    resolved = {
        timeframe: store.read_prefix(case.prefixes[timeframe], asof=case.asof)
        for timeframe in TIMEFRAMES
    }
    features = {timeframe: item.values for timeframe, item in resolved.items()}
    type_ids: list[int] = []
    lifecycle_ids: list[int] = []
    relation_ids: list[int] = []
    scale_ids: list[int] = []
    numeric: list[tuple[float, float, float, float]] = []
    direct_label_source_mask: list[bool] = []
    for event in case.events:
        age = max(0.0, float((case.asof - event.observed_at).total_seconds()))
        type_ids.append(_stable_token(event.event_type, buckets=event_type_buckets))
        lifecycle_ids.append(_stable_token(event.lifecycle, buckets=lifecycle_buckets))
        relation_ids.append(
            _stable_token("|".join(event.relation_types) or "none", buckets=relation_buckets)
        )
        scale_ids.append(_stable_token(event.scale, buckets=scale_buckets))
        numeric.append(
            (
                math.log1p(age),
                math.log1p(float(event.duration_seconds)),
                float(len(event.relation_types)),
                float(event.direction),
            )
        )
        direct_label_source_mask.append(_is_direct_label_source_event(event))
    return PreparedRepresentationCase(
        case=case,
        timeframe_features=features,
        feature_max_at=max(item.last_available_at for item in resolved.values()),
        event_type_ids=np.asarray(type_ids, dtype=np.int64),
        lifecycle_ids=np.asarray(lifecycle_ids, dtype=np.int64),
        relation_ids=np.asarray(relation_ids, dtype=np.int64),
        scale_ids=np.asarray(scale_ids, dtype=np.int64),
        event_numeric=np.asarray(numeric, dtype=np.float32),
        direct_label_source_event_mask=np.asarray(
            direct_label_source_mask, dtype=bool
        ),
    )


def mask_direct_label_source_tokens(
    example: PreparedRepresentationCase,
) -> PreparedRepresentationCase:
    """Build the inference-time shortcut probe used by quality evaluation.

    The market embedding is recomputed after removing case-derived label
    sources.  No target or future outcome is consulted.  If every event is a
    direct source, a single fixed UNK-like token remains so the encoder still
    receives a valid sequence without leaking how many source tokens existed.
    """

    source_mask = np.asarray(example.direct_label_source_event_mask, dtype=bool)
    keep = ~source_mask
    if bool(keep.any()):
        return replace(
            example,
            event_type_ids=np.asarray(example.event_type_ids[keep], dtype=np.int64),
            lifecycle_ids=np.asarray(example.lifecycle_ids[keep], dtype=np.int64),
            relation_ids=np.asarray(example.relation_ids[keep], dtype=np.int64),
            scale_ids=np.asarray(example.scale_ids[keep], dtype=np.int64),
            event_numeric=np.asarray(example.event_numeric[keep], dtype=np.float32),
            direct_label_source_event_mask=np.zeros(int(keep.sum()), dtype=bool),
            label_sources_masked=True,
        )
    return replace(
        example,
        event_type_ids=np.asarray([MASK_TOKEN_ID], dtype=np.int64),
        lifecycle_ids=np.asarray([MASK_TOKEN_ID], dtype=np.int64),
        relation_ids=np.asarray([MASK_TOKEN_ID], dtype=np.int64),
        scale_ids=np.asarray([MASK_TOKEN_ID], dtype=np.int64),
        event_numeric=np.zeros((1, 4), dtype=np.float32),
        direct_label_source_event_mask=np.zeros(1, dtype=bool),
        label_sources_masked=True,
    )


def preprocess_neutral_direct_source_events(
    example: PreparedRepresentationCase,
) -> PreparedRepresentationCase:
    """Apply the single deterministic B0 event-input protocol.

    Training, validation, export, and future online queries must all start from
    this processed ``PreparedRepresentationCase``.  Objective masking remains
    a later, stochastic training-only operation and is not part of this step.
    """

    identity = neutral_direct_source_preprocessing_identity()
    if (
        example.neutral_preprocessing_version is not None
        and (
            example.neutral_preprocessing_version
            != identity["protocol"]["protocol_version"]
            or example.neutral_preprocessing_sha256 != identity["sha256"]
        )
    ):
        raise RepresentationDataError(
            "prepared case uses a different neutral preprocessing protocol"
        )
    processed = mask_direct_label_source_tokens(example)
    return replace(
        processed,
        neutral_preprocessing_version=identity["protocol"]["protocol_version"],
        neutral_preprocessing_sha256=identity["sha256"],
    )


def prepare_neutral_representation_case(
    case: RepresentationCase,
    store: CanonicalOHLCVStore,
) -> PreparedRepresentationCase:
    """Resolve causal prefixes and apply the registered neutral B0 protocol."""

    return preprocess_neutral_direct_source_events(
        prepare_representation_case(case, store)
    )


def require_neutral_preprocessed_examples(
    examples: Sequence[PreparedRepresentationCase],
) -> None:
    """Fail closed unless every example carries the current processed input."""

    if not examples:
        raise RepresentationDataError("neutral preprocessing received no examples")
    identity = neutral_direct_source_preprocessing_identity()
    for example in examples:
        if (
            example.neutral_preprocessing_version
            != identity["protocol"]["protocol_version"]
            or example.neutral_preprocessing_sha256 != identity["sha256"]
            or not example.label_sources_masked
            or bool(np.asarray(example.direct_label_source_event_mask).any())
        ):
            raise RepresentationDataError(
                "neutral representation input was not processed by the B0 protocol"
            )


def causal_input_fingerprint(case: RepresentationCase) -> str:
    """Hash only information available at decision time, excluding identities/outcomes."""

    payload = {
        "schema": FEATURE_SCHEMA_VERSION,
        "asof": case.asof.isoformat(),
        "direction": case.direction,
        "authority_direction": case.authority_direction,
        "prefixes": {
            timeframe: {
                "market_epoch_id": prefix.market_epoch_id,
                "source": prefix.canonical_source_id,
                "row_start": prefix.row_start,
                "row_end_exclusive": prefix.row_end_exclusive,
                "start_at": (
                    None if prefix.start_at is None else prefix.start_at.isoformat()
                ),
                "end_at": None if prefix.end_at is None else prefix.end_at.isoformat(),
                "tail_at_or_before": (
                    None
                    if prefix.tail_at_or_before is None
                    else prefix.tail_at_or_before.isoformat()
                ),
                "resolve_external_rows_by_time": prefix.resolve_external_rows_by_time,
                "available_at_column": prefix.available_at_column,
            }
            for timeframe, prefix in sorted(case.prefixes.items())
        },
        "events": [
            {
                "event_type": event.event_type,
                "lifecycle": event.lifecycle,
                "observed_at": event.observed_at.isoformat(),
                "active_since": event.active_since.isoformat(),
                "duration_seconds": event.duration_seconds,
                "relations": event.relation_types,
                "direction": event.direction,
                "scale": event.scale,
            }
            for event in case.events
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deduplicate_causal_inputs(
    cases: Sequence[RepresentationCase],
) -> tuple[RepresentationCase, ...]:
    """Keep one deterministic revision for an identical decision-time input."""

    selected: dict[str, RepresentationCase] = {}
    for case in sorted(cases, key=lambda item: (item.asof, item.case_id, item.revision_id)):
        selected.setdefault(causal_input_fingerprint(case), case)
    return tuple(selected[key] for key in sorted(selected))


def _market_episode_grain(case: RepresentationCase) -> tuple[str, str]:
    return case.market_epoch_id, case.market_episode_id


def assign_leakage_safe_splits(
    cases: Sequence[RepresentationCase],
    *,
    seed: int = 17,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, str]:
    """Split connected episode/input groups, preventing cross-split duplicates."""

    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise RepresentationDataError("invalid train/validation fractions")
    if train_fraction + validation_fraction >= 1:
        raise RepresentationDataError("train + validation fraction must be below 1")
    if len({case.revision_id for case in cases}) != len(cases):
        raise RepresentationDataError("revision_id must be unique before splitting")

    parents = list(range(len(cases)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    by_episode: dict[tuple[str, str], int] = {}
    by_input: dict[str, int] = {}
    for index, case in enumerate(cases):
        for mapping, key in (
            (by_episode, _market_episode_grain(case)),
            (by_input, causal_input_fingerprint(case)),
        ):
            prior = mapping.setdefault(key, index)
            union(index, prior)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(cases)):
        groups[find(index)].append(index)
    output: dict[str, str] = {}
    for indices in groups.values():
        component = "|".join(
            sorted(
                {
                    "episode:"
                    f"{cases[index].market_epoch_id}:"
                    f"{cases[index].market_episode_id}"
                    for index in indices
                }
                | {
                    f"input:{causal_input_fingerprint(cases[index])}"
                    for index in indices
                }
            )
        )
        digest = hashlib.sha256(f"{seed}|{component}".encode("utf-8")).digest()
        position = int.from_bytes(digest[:8], "big") / float(2**64)
        split = (
            "train"
            if position < train_fraction
            else "validation"
            if position < train_fraction + validation_fraction
            else "test"
        )
        for index in indices:
            output[cases[index].revision_id] = split
    validate_split_integrity(cases, output)
    return output


def validate_split_integrity(
    cases: Sequence[RepresentationCase], assignments: Mapping[str, str]
) -> None:
    expected = {case.revision_id for case in cases}
    if set(assignments) != expected:
        raise RepresentationDataError("split assignments do not cover revisions exactly")
    valid = {"train", "validation", "test"}
    if not set(assignments.values()) <= valid:
        raise RepresentationDataError("unknown dataset split label")
    episode_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    input_splits: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        split = assignments[case.revision_id]
        episode_splits[_market_episode_grain(case)].add(split)
        input_splits[causal_input_fingerprint(case)].add(split)
    if any(len(values) != 1 for values in episode_splits.values()):
        raise RepresentationDataError(
            "train/validation/test share a market-epoch/MarketEpisode pair"
        )
    if any(len(values) != 1 for values in input_splits.values()):
        raise RepresentationDataError(
            "identical decision-time inputs appear in different dataset splits"
        )


@dataclass(frozen=True)
class SelfSupervisedTarget:
    """Labels are passed only to loss computation, never model encoding."""

    next_event_type: int = -100
    next_lifecycle: int = -100
    next_event_time_bucket: int = -100
    displacement_state: int = -100
    draw_consumed: int = -100
    scale_direction_alignment: int = -100

    def __post_init__(self) -> None:
        class_counts = {
            "next_event_type": len(NEXT_EVENT_TYPE_VOCAB),
            "next_lifecycle": len(NEXT_LIFECYCLE_VOCAB),
            "next_event_time_bucket": 8,
            "displacement_state": 2,
            "draw_consumed": 2,
            "scale_direction_alignment": 3,
        }
        for name, count in class_counts.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise RepresentationDataError(f"target {name} must be an integer class")
            if value != -100 and not 0 <= int(value) < count:
                raise RepresentationDataError(
                    f"target {name} is outside registered class range"
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SelfSupervisedTarget":
        forbidden = sorted(
            key
            for key in payload
            if str(key).strip().lower() not in ALLOWED_SELF_SUPERVISED_TARGETS
            and str(key).strip().lower() not in TARGET_METADATA_KEYS
        )
        if forbidden:
            raise RepresentationDataError(
                f"unsupported or economic training target fields: {forbidden}"
            )
        values: dict[str, int] = {}
        for name in ALLOWED_SELF_SUPERVISED_TARGETS:
            value = payload.get(name, -100)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise RepresentationDataError(f"target {name} must be an integer class")
            values[name] = int(value)
        return cls(**values)


@dataclass(frozen=True)
class ObservableTargetRecord:
    """Separated label record derived only from later observable revisions."""

    revision_id: str
    market_epoch_id: str
    entry_episode_id: str
    input_asof: pd.Timestamp
    label_max_observed_at: pd.Timestamp
    next_revision_id: str | None
    target: SelfSupervisedTarget
    label_source: str = (
        "same_episode_observable_revisions_with_complete_transition_coverage_v2"
    )

    def __post_init__(self) -> None:
        input_asof = _aware_timestamp(self.input_asof, name="target.input_asof")
        label_clock = _aware_timestamp(
            self.label_max_observed_at, name="target.label_max_observed_at"
        )
        if label_clock < input_asof:
            raise RepresentationDataError("target label clock precedes model input")

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "market_epoch_id": self.market_epoch_id,
            "entry_episode_id": self.entry_episode_id,
            "input_asof": self.input_asof.isoformat(),
            "label_max_observed_at": self.label_max_observed_at.isoformat(),
            "next_revision_id": self.next_revision_id,
            "label_source": self.label_source,
            **vars(self.target),
        }


@dataclass(frozen=True)
class NeutralSparseTargetRecord:
    """Input-only labels at canonical MarketEpisode grain.

    The next lifecycle may use one later sparse revision of the same physical
    episode.  Scale alignment is contemporaneous.  The four legacy targets
    that require a complete transition ledger, a frozen draw, or displacement
    custody remain explicitly disabled rather than being guessed.
    """

    revision_id: str
    market_epoch_id: str
    market_episode_id: str
    input_asof: pd.Timestamp
    label_max_observed_at: pd.Timestamp
    next_revision_id: str | None
    target: SelfSupervisedTarget
    label_source: str = NEUTRAL_SPARSE_TARGET_LABEL_SOURCE

    def __post_init__(self) -> None:
        for name in ("revision_id", "market_epoch_id", "market_episode_id"):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=f"neutral_target.{name}"),
            )
        input_asof = _aware_timestamp(
            self.input_asof,
            name="neutral_target.input_asof",
        )
        label_clock = _aware_timestamp(
            self.label_max_observed_at,
            name="neutral_target.label_max_observed_at",
        )
        if label_clock < input_asof:
            raise RepresentationDataError(
                "neutral target label clock precedes model input"
            )
        object.__setattr__(self, "input_asof", input_asof)
        object.__setattr__(self, "label_max_observed_at", label_clock)
        if self.next_revision_id is not None:
            object.__setattr__(
                self,
                "next_revision_id",
                _nonempty(
                    self.next_revision_id,
                    name="neutral_target.next_revision_id",
                ),
            )
        if not isinstance(self.target, SelfSupervisedTarget):
            raise RepresentationDataError(
                "neutral target payload must be SelfSupervisedTarget"
            )
        if self.label_source != NEUTRAL_SPARSE_TARGET_LABEL_SOURCE:
            raise RepresentationDataError("neutral target label source changed")
        if any(
            getattr(self.target, name) != -100
            for name in NEUTRAL_SPARSE_DISABLED_TARGETS
        ):
            raise RepresentationDataError(
                "neutral sparse target enabled a coverage-dependent legacy task"
            )
        if self.target.scale_direction_alignment < 0:
            raise RepresentationDataError(
                "neutral sparse target must classify same-clock scale alignment"
            )
        has_next = self.next_revision_id is not None
        if has_next != (self.target.next_lifecycle >= 0):
            raise RepresentationDataError(
                "neutral next-lifecycle label differs from next revision custody"
            )

    @property
    def entry_episode_id(self) -> str:
        """Legacy read alias; neutral code must use ``market_episode_id``."""

        return self.market_episode_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "market_epoch_id": self.market_epoch_id,
            "market_episode_id": self.market_episode_id,
            "input_asof": self.input_asof.isoformat(),
            "label_max_observed_at": self.label_max_observed_at.isoformat(),
            "next_revision_id": self.next_revision_id,
            "label_source": self.label_source,
            "active_tasks": list(NEUTRAL_SPARSE_ACTIVE_TARGETS),
            "disabled_tasks": list(NEUTRAL_SPARSE_DISABLED_TARGETS),
            **vars(self.target),
        }


_MARKET_CASE_RUN_KEYS = frozenset(
    {
        "schema_version",
        "runner",
        "mode",
        "runtime_state_schema_version",
        "profile",
        "source",
        "model_config",
        "market_case_input_identity",
        "window",
        "output",
    }
)
_MARKET_CASE_SOURCE_KEYS = frozenset(
    {
        "path",
        "sha256",
        "rows",
        "first",
        "last",
        "last_completed_asof",
        "role",
        "symbol",
        "instrument_id",
    }
)
_MARKET_CASE_MODEL_CONFIG_KEYS = frozenset(
    {
        "path",
        "sha256",
        "schema_version",
        "tick_size",
        "timezone",
    }
)
_MARKET_CASE_WINDOW_KEYS = frozenset(
    {
        "start",
        "end_exclusive",
        "role",
        "warmup_days",
        "observation_clock",
        "capture_interval",
    }
)
_NEUTRAL_INPUT_FORBIDDEN_KEY_MARKERS = (
    "playbook",
    "shadow",
    "brain_response",
    "selected_action",
    "decision",
    "risk",
    "outcome",
    "profit",
    "pnl",
    "mfe",
    "mae",
)
_NEUTRAL_MICRO_BOS_REFERENCE_ALIGNMENTS = frozenset(
    {
        "aligned",
        "opposed",
        "simultaneous_unknown",
        "ambiguous_same_clock",
    }
)


def _strict_manifest_section(
    value: Any,
    *,
    keys: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise RepresentationDataError(f"market case run {name} schema changed")
    return value


def _sha256_identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise RepresentationDataError(f"market case run {name} must be text")
    identity = value
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise RepresentationDataError(f"market case run {name} is not a SHA-256")
    return identity


def _nonnegative_manifest_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RepresentationDataError(f"market case run {name} must be an integer")
    output = int(value)
    if output < 0:
        raise RepresentationDataError(
            f"market case run {name} must be non-negative"
        )
    return output


def _manifest_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise RepresentationDataError(f"market case run {name} must be text")
    return _nonempty(value, name=f"market case run {name}")


def _neutral_forbidden_input_paths(
    value: Any,
    *,
    path: str,
) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().lower()
            child = f"{path}.{key}" if path else key
            if any(
                marker in key
                for marker in _NEUTRAL_INPUT_FORBIDDEN_KEY_MARKERS
            ):
                found.append(child)
                continue
            found.extend(_neutral_forbidden_input_paths(nested, path=child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            found.extend(
                _neutral_forbidden_input_paths(
                    nested,
                    path=f"{path}[{index}]",
                )
            )
    return tuple(found)


def _normalise_neutral_observation_input(
    observation: Any,
) -> Mapping[str, Any]:
    """Turn the sole causal ``outcome`` enum into a typed relation token.

    ``validate_market_case_input_row`` has already enforced the exact nested
    path and four-value enum.  Removing the overloaded key here lets the
    generic representation guard remain strict for every economic/future
    outcome while retaining this same-clock MicroBOS reference fact.
    """

    if not isinstance(observation, Mapping):
        raise RepresentationDataError(
            "neutral Observation transition must be an object"
        )
    normalised = copy.deepcopy(observation)
    collections = normalised.get("collections")
    if not isinstance(collections, Mapping):
        raise RepresentationDataError(
            "neutral Observation transition collections must be an object"
        )
    micro_bos_events = collections.get(
        "group5_micro_bos_transitions_this_update"
    )
    if not isinstance(micro_bos_events, list):
        raise RepresentationDataError(
            "neutral MicroBOS transition collection must be an array"
        )
    for event in micro_bos_events:
        if not isinstance(event, dict):
            raise RepresentationDataError(
                "neutral MicroBOS transition must be an object"
            )
        if "outcome" not in event:
            continue
        alignment = event.pop("outcome")
        if (
            not isinstance(alignment, str)
            or alignment not in _NEUTRAL_MICRO_BOS_REFERENCE_ALIGNMENTS
        ):
            raise RepresentationDataError(
                "neutral MicroBOS reference alignment changed"
            )
        raw_relations = event.get("relation_types", ())
        if raw_relations is None:
            relations: tuple[str, ...] = ()
        elif isinstance(raw_relations, str):
            relations = (raw_relations,)
        elif isinstance(raw_relations, Sequence) and not isinstance(
            raw_relations,
            (str, bytes),
        ):
            if any(not isinstance(value, str) for value in raw_relations):
                raise RepresentationDataError(
                    "neutral MicroBOS relations must be strings"
                )
            relations = tuple(raw_relations)
        else:
            raise RepresentationDataError(
                "neutral MicroBOS relations must be an array"
            )
        event["relation_types"] = list(
            dict.fromkeys(
                (
                    *relations,
                    f"micro_bos_reference_alignment:{alignment}",
                )
            )
        )
    return normalised


def _validated_market_case_run_manifest(
    run_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    keys = set(run_manifest) if isinstance(run_manifest, Mapping) else set()
    if not isinstance(run_manifest, Mapping) or keys not in (
        set(_MARKET_CASE_RUN_KEYS),
        set(_MARKET_CASE_RUN_KEYS) | {"repository"},
    ):
        raise RepresentationDataError("market case run manifest schema changed")
    if "repository" in run_manifest:
        repository = run_manifest["repository"]
        commit = repository.get("commit") if isinstance(repository, Mapping) else None
        if (
            not isinstance(repository, Mapping)
            or set(repository) != {"commit"}
            or not isinstance(commit, str)
            or len(commit) != 40
            or commit != commit.lower()
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise RepresentationDataError(
                "market case run repository identity is invalid"
            )
    if (
        _nonnegative_manifest_integer(
            run_manifest["schema_version"],
            name="schema_version",
        )
        != 1
        or run_manifest["runner"] != "continuous_replay"
        or run_manifest["mode"] != "market_case_input"
    ):
        raise RepresentationDataError("market case run manifest identity changed")
    runtime_schema = _nonnegative_manifest_integer(
        run_manifest["runtime_state_schema_version"],
        name="runtime_state_schema_version",
    )
    has_repository = "repository" in run_manifest
    if not (
        (runtime_schema == 5 and not has_repository)
        or (runtime_schema == 6 and has_repository)
    ):
        raise RepresentationDataError(
            "market case run runtime/repository version binding changed"
        )

    profile = _strict_manifest_section(
        run_manifest["profile"],
        keys=frozenset({"name", "identity"}),
        name="profile",
    )
    _manifest_text(profile["name"], name="profile.name")
    _sha256_identity(profile["identity"], name="profile.identity")

    source = _strict_manifest_section(
        run_manifest["source"],
        keys=_MARKET_CASE_SOURCE_KEYS,
        name="source",
    )
    source_path = Path(_manifest_text(source["path"], name="source.path"))
    if not source_path.is_absolute():
        raise RepresentationDataError("market case run source path must be absolute")
    source_sha256 = _sha256_identity(source["sha256"], name="source.sha256")
    source_rows = _nonnegative_manifest_integer(source["rows"], name="source.rows")
    if source_rows < 1:
        raise RepresentationDataError("market case run source must contain rows")
    source_first = _aware_timestamp(
        _manifest_text(source["first"], name="source.first"),
        name="run.source.first",
    )
    source_last = _aware_timestamp(
        _manifest_text(source["last"], name="source.last"),
        name="run.source.last",
    )
    source_completed = _aware_timestamp(
        _manifest_text(
            source["last_completed_asof"],
            name="source.last_completed_asof",
        ),
        name="run.source.last_completed_asof",
    )
    if source_first > source_last or source_last > source_completed:
        raise RepresentationDataError("market case run source clocks are reversed")
    source_role = _manifest_text(source["role"], name="source.role")
    source_symbol = _manifest_text(source["symbol"], name="source.symbol")
    source_instrument_id = _nonnegative_manifest_integer(
        source["instrument_id"],
        name="market case run source.instrument_id",
    )

    model_config = _strict_manifest_section(
        run_manifest["model_config"],
        keys=_MARKET_CASE_MODEL_CONFIG_KEYS,
        name="model_config",
    )
    config_path = Path(
        _manifest_text(model_config["path"], name="model_config.path")
    )
    if not config_path.is_absolute():
        raise RepresentationDataError(
            "market case run model_config path must be absolute"
        )
    config_sha256 = _sha256_identity(
        model_config["sha256"],
        name="model_config.sha256",
    )
    config_schema = _nonnegative_manifest_integer(
        model_config["schema_version"],
        name="model_config.schema_version",
    )
    raw_tick_size = model_config["tick_size"]
    if isinstance(raw_tick_size, bool) or not isinstance(
        raw_tick_size,
        (int, float, np.integer, np.floating),
    ):
        raise RepresentationDataError(
            "market case run model_config tick_size is invalid"
        )
    tick_size = float(raw_tick_size)
    if config_schema < 1 or not math.isfinite(tick_size) or tick_size <= 0:
        raise RepresentationDataError(
            "market case run model_config values are invalid"
        )
    timezone = _manifest_text(
        model_config["timezone"],
        name="model_config.timezone",
    )

    window = _strict_manifest_section(
        run_manifest["window"],
        keys=_MARKET_CASE_WINDOW_KEYS,
        name="window",
    )
    window_start = _aware_timestamp(
        _manifest_text(window["start"], name="window.start"),
        name="run.window.start",
    )
    window_end = _aware_timestamp(
        _manifest_text(window["end_exclusive"], name="window.end_exclusive"),
        name="run.window.end_exclusive",
    )
    if window_start >= window_end:
        raise RepresentationDataError("market case run window is empty")
    window_role = _manifest_text(window["role"], name="window.role")
    _nonnegative_manifest_integer(window["warmup_days"], name="window.warmup_days")
    if (
        window["observation_clock"] != "completed_1m_bar_end"
        or window["capture_interval"] != "[start,end_exclusive)"
    ):
        raise RepresentationDataError("market case run window contract changed")

    output = _strict_manifest_section(
        run_manifest["output"],
        keys=frozenset({"stream_families", "shard_rows", "checkpoint_bars"}),
        name="output",
    )
    if output["stream_families"] != ["market_case_input_shards"]:
        raise RepresentationDataError("market case run output stream changed")
    if (
        _nonnegative_manifest_integer(output["shard_rows"], name="output.shard_rows")
        < 1
        or _nonnegative_manifest_integer(
            output["checkpoint_bars"],
            name="output.checkpoint_bars",
        )
        < 1
    ):
        raise RepresentationDataError("market case run output bounds are invalid")

    from .market_cases import expected_market_case_run_identity

    if run_manifest["market_case_input_identity"] != (
        expected_market_case_run_identity()
    ):
        raise RepresentationDataError("market case run input identity changed")
    return {
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_rows": source_rows,
        "source_first": source_first,
        "source_last": source_last,
        "source_last_completed_asof": source_completed,
        "source_role": source_role,
        "symbol": source_symbol,
        "instrument_id": source_instrument_id,
        "model_config_path": str(config_path),
        "model_config_sha256": config_sha256,
        "model_config_schema_version": config_schema,
        "tick_size": tick_size,
        "timezone": timezone,
        "window_start": window_start,
        "window_end_exclusive": window_end,
        "window_role": window_role,
    }


def _neutral_authority_direction(context: Mapping[str, Any]) -> int:
    authority_stack = context.get("authority_stack", ())
    if not isinstance(authority_stack, Sequence) or isinstance(
        authority_stack,
        (str, bytes),
    ):
        raise RepresentationDataError(
            "neutral global context authority_stack must be an array"
        )
    for layer in authority_stack:
        if not isinstance(layer, Mapping):
            raise RepresentationDataError(
                "neutral global context authority layer must be an object"
            )
        if str(layer.get("status", "")).strip().lower() == "intact":
            return _direction_code(layer.get("direction", 0))
    return 0


def _neutral_scale_direction_alignment_class(
    context: Mapping[str, Any],
) -> int:
    """Classify only active, same-clock neutral scale facts.

    Unknown, disconnected, ambiguous, or directionless detail states do not
    become evidence.  Two or more distinct scales are required, so an absent
    lower-timeframe fact cannot be inferred from the authority stack.
    """

    details = context.get("scale_relation_details")
    if not isinstance(details, Mapping):
        raise RepresentationDataError(
            "neutral global context scale_relation_details must be an object"
        )
    directions_by_scale: dict[str, int] = {}
    conflicted_scales: set[str] = set()
    for raw_scale, raw_detail in details.items():
        scale = normalize_timeframe(raw_scale)
        if not isinstance(raw_detail, Mapping):
            raise RepresentationDataError(
                "neutral scale relation detail must be an object"
            )
        detail_scale = raw_detail.get("timeframe", raw_scale)
        if normalize_timeframe(detail_scale) != scale:
            raise RepresentationDataError(
                "neutral scale relation key and timeframe disagree"
            )
        relation = str(raw_detail.get("relation", "unknown")).strip().lower()
        if relation not in {
            "aligned",
            "normal_pullback",
            "material_opposition",
            "unknown",
        }:
            raise RepresentationDataError(
                "neutral scale relation kind is unsupported"
            )
        ambiguous = raw_detail.get("ambiguous", False)
        graph_connected = raw_detail.get("graph_connected", False)
        if type(ambiguous) is not bool or type(graph_connected) is not bool:
            raise RepresentationDataError(
                "neutral scale relation flags must be booleans"
            )
        if relation == "unknown" or ambiguous or not graph_connected:
            continue
        try:
            direction = _direction_code(raw_detail.get("direction", 0))
        except RepresentationDataError:
            direction = 0
        if direction == 0:
            continue
        previous = directions_by_scale.get(scale)
        if previous is not None and previous != direction:
            conflicted_scales.add(scale)
        directions_by_scale[scale] = direction
    if conflicted_scales or len(directions_by_scale) < 2:
        return 0
    return 1 if len(set(directions_by_scale.values())) == 1 else 2


def market_case_input_to_representation_mapping(
    row: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Adapt one neutral MarketEpisode row without consulting playbooks.

    Runtime source/config lineage is injected only from the already-loaded,
    input-only run manifest.  The row contributes only its canonical physical
    identity, same-clock Eye/Scene/neutral context and OHLCV prefix bounds.
    """

    from .market_cases import validate_market_case_input_row

    try:
        validate_market_case_input_row(row)
    except (TypeError, ValueError) as exc:
        raise RepresentationDataError(
            "neutral market case input row is invalid"
        ) from exc
    manifest = _validated_market_case_run_manifest(run_manifest)
    asof = _aware_timestamp(row["asof"], name="market_case.asof")
    if not manifest["window_start"] <= asof < manifest["window_end_exclusive"]:
        raise RepresentationDataError("market case row is outside the run window")
    if not manifest["source_first"] <= asof <= manifest[
        "source_last_completed_asof"
    ]:
        raise RepresentationDataError("market case row exceeds the run source clock")
    source_ordinal = int(row["source_replay_ordinal"])
    if source_ordinal >= manifest["source_rows"]:
        raise RepresentationDataError("market case row exceeds the run source rows")

    transition_kinds = json.loads(str(row["transition_kinds_json"]))
    observation = _normalise_neutral_observation_input(
        json.loads(str(row["observation_transition_json"]))
    )
    scene = json.loads(str(row["scene_graph_delta_json"]))
    context = json.loads(str(row["neutral_global_context_json"]))
    prefix_rows = json.loads(str(row["ohlcv_prefix_refs_json"]))
    replay_view_ranges = {
        (
            int(prefix["replay_view_1m_row_start"]),
            int(prefix["replay_view_1m_row_end_exclusive"]),
        )
        for prefix in prefix_rows
    }
    if len(replay_view_ranges) != 1:
        raise RepresentationDataError(
            "neutral prefix replay-view lineage differs across timeframes"
        )
    replay_view_start, replay_view_end = next(iter(replay_view_ranges))
    if replay_view_start > replay_view_end or replay_view_end > manifest[
        "source_rows"
    ]:
        raise RepresentationDataError(
            "neutral prefix replay-view lineage exceeds the run source"
        )
    selected_inputs = {
        "transition_kinds": transition_kinds,
        "observation_transition": observation,
        "scene_graph_delta": scene,
        "neutral_global_context": context,
        "ohlcv_prefix_refs": prefix_rows,
    }
    forbidden = sorted(
        _neutral_forbidden_input_paths(selected_inputs, path="")
    )
    if forbidden:
        raise RepresentationDataError(
            "neutral representation input contains control, Shadow or economic keys: "
            f"{forbidden}"
        )
    if not isinstance(context, Mapping):
        raise RepresentationDataError("neutral global context must be an object")

    market_epoch_id = str(row["market_epoch_id"])
    market_episode_id = str(row["market_episode_id"])
    prefixes = [
        {
            "market_epoch_id": market_epoch_id,
            "timeframe": prefix["timeframe"],
            "canonical_source_id": manifest["source_sha256"],
            "row_start": prefix["frame_row_start"],
            "row_end_exclusive": prefix["frame_row_end_exclusive"],
            "tail_at_or_before": prefix["cutoff"],
        }
        for prefix in prefix_rows
    ]
    market_episode = {
        "market_episode_id": market_episode_id,
        "lifecycle": str(row["lifecycle"]),
        "direction": str(row["direction"]),
        "entry_location_id": str(row["entry_location_id"]),
        "entry_path_id": str(row["entry_path_id"]),
        "transition_kinds": tuple(str(value) for value in transition_kinds),
    }
    regime = (
        "balance"
        if str(context.get("market_mode", "")).strip().lower() == "balanced"
        else "unknown"
    )
    return {
        "case_id": f"market-case:{market_episode_id}",
        "revision_id": str(row["revision_id"]),
        "revision_index": int(row["revision_index"]),
        "revision_stage": str(row["revision_stage"]),
        "stage_identity": (
            f"{row['revision_stage']}:{row['revision_id']}"
        ),
        "split_role": manifest["window_role"],
        "market_epoch_id": market_epoch_id,
        "market_episode_id": market_episode_id,
        # Compatibility-only aliases; no EntryEpisode or ContextThesis object
        # is read or fabricated by the neutral adapter.
        "entry_episode_id": market_episode_id,
        "context_thesis_id": f"neutral-context:{market_episode_id}",
        "direction": str(row["direction"]),
        "regime": regime,
        "mechanism_label": "unknown",
        "asof": asof,
        "decision_at": asof,
        "canonical_source_id": manifest["source_sha256"],
        "source_path": manifest["source_path"],
        "source_role": manifest["source_role"],
        "symbol": manifest["symbol"],
        "instrument_id": manifest["instrument_id"],
        "entry_location_id": str(row["entry_location_id"]),
        "entry_path_id": str(row["entry_path_id"]),
        "transition_kinds": tuple(str(value) for value in transition_kinds),
        "model_config_path": manifest["model_config_path"],
        "model_config_sha256": manifest["model_config_sha256"],
        "model_config_schema_version": manifest[
            "model_config_schema_version"
        ],
        "tick_size": manifest["tick_size"],
        "timezone": manifest["timezone"],
        "prefixes": prefixes,
        "events": {
            "market_episode_transition_kinds": transition_kinds,
            "observation_transition": observation,
        },
        "graph": scene,
        "neutral_global_context": context,
        "market_episode": market_episode,
        "authority_direction": _neutral_authority_direction(context),
    }


def representation_case_from_market_case_input_row(
    row: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> RepresentationCase:
    """Build a strict outcome-blind case from one neutral input row."""

    return RepresentationCase.from_mapping(
        market_case_input_to_representation_mapping(row, run_manifest)
    )


def build_neutral_market_revision_targets(
    rows: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
) -> dict[str, NeutralSparseTargetRecord]:
    """Build the two causally supported targets from neutral sparse rows.

    The next lifecycle is taken only from the next revision of the same
    ``(market_epoch_id, market_episode_id)``.  Scale alignment uses only the
    current row's neutral global context.  Coverage-dependent legacy target
    builders are intentionally not involved.
    """

    from .market_cases import validate_market_case_rows

    _validated_market_case_run_manifest(run_manifest)
    try:
        validate_market_case_rows(rows)
    except (TypeError, ValueError) as exc:
        raise RepresentationDataError(
            "neutral market case target rows are invalid"
        ) from exc
    adapted: list[
        tuple[RepresentationCase, Mapping[str, Any], str]
    ] = []
    for row in rows:
        mapping = market_case_input_to_representation_mapping(row, run_manifest)
        case = RepresentationCase.from_mapping(mapping)
        market_episode = mapping.get("market_episode")
        context = mapping.get("neutral_global_context")
        if not isinstance(market_episode, Mapping) or not isinstance(
            context,
            Mapping,
        ):
            raise RepresentationDataError(
                "neutral target source mapping is incomplete"
            )
        lifecycle = str(market_episode.get("lifecycle", ""))
        if lifecycle not in NEUTRAL_MARKET_LIFECYCLE_TARGETS:
            raise RepresentationDataError(
                "neutral target source lifecycle changed"
            )
        adapted.append((case, context, lifecycle))

    groups: dict[
        tuple[str, str],
        list[tuple[RepresentationCase, Mapping[str, Any], str]],
    ] = defaultdict(list)
    for item in adapted:
        case = item[0]
        groups[(case.market_epoch_id, case.market_episode_id)].append(item)

    output: dict[str, NeutralSparseTargetRecord] = {}
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda item: (
                item[0].revision_index,
                item[0].asof,
                item[0].revision_id,
            ),
        )
        if [item[0].revision_index for item in ordered] != list(
            range(len(ordered))
        ):
            raise RepresentationDataError(
                "neutral sparse revision indexes are discontinuous"
            )
        for position, (case, context, _) in enumerate(ordered):
            next_item = (
                ordered[position + 1]
                if position + 1 < len(ordered)
                else None
            )
            if next_item is None:
                next_revision_id = None
                next_lifecycle = -100
                label_clock = case.asof
            else:
                next_case, _, next_lifecycle_name = next_item
                next_revision_id = next_case.revision_id
                next_lifecycle = NEUTRAL_MARKET_LIFECYCLE_TARGETS[
                    next_lifecycle_name
                ]
                label_clock = next_case.asof
            target = SelfSupervisedTarget(
                next_event_type=-100,
                next_lifecycle=next_lifecycle,
                next_event_time_bucket=-100,
                displacement_state=-100,
                draw_consumed=-100,
                scale_direction_alignment=(
                    _neutral_scale_direction_alignment_class(context)
                ),
            )
            output[case.revision_id] = NeutralSparseTargetRecord(
                revision_id=case.revision_id,
                market_epoch_id=case.market_epoch_id,
                market_episode_id=case.market_episode_id,
                input_asof=case.asof,
                label_max_observed_at=label_clock,
                next_revision_id=next_revision_id,
                target=target,
            )
    return output


def _adapt_case_input_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    if "prefix_refs_json" in row:
        # Local import avoids making normal replay/module import depend on the
        # optional representation layer in the opposite direction.
        from .causal_cases import case_input_to_representation_mapping

        adapted = dict(case_input_to_representation_mapping(row))
        adapted["revision_index"] = int(row["revision_index"])
        adapted["split_role"] = str(row["split_role"])
        adapted["stage_identity"] = str(row["stage_identity"])
        adapted["symbol"] = str(row["symbol"])
        adapted["instrument_id"] = int(row["instrument_id"])
        return adapted
    return dict(row)


def representation_case_from_case_input_row(
    row: Mapping[str, Any],
) -> RepresentationCase:
    """Consume a recorder row through its strict future-free public adapter."""

    return RepresentationCase.from_mapping(_adapt_case_input_mapping(row))


def _nested_values(value: Any, *, key_names: frozenset[str]) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in key_names:
                yield nested
            yield from _nested_values(nested, key_names=key_names)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _nested_values(nested, key_names=key_names)


def _episode_lifecycle(mapping: Mapping[str, Any]) -> str:
    episode = mapping.get("entry_episode")
    if isinstance(episode, Mapping):
        value = episode.get("lifecycle", episode.get("phase"))
        if value is not None:
            return str(value).strip().lower()
        terminal = episode.get("terminal_reason")
        if terminal:
            return "terminal"
    return "unknown"


def _source_displacement_id(mapping: Mapping[str, Any]) -> str | None:
    for section in (mapping.get("entry_episode"), mapping.get("context_thesis")):
        for value in _nested_values(
            section,
            key_names=frozenset(
                {"source_displacement_id", "displacement_id", "source_entity_id"}
            ),
        ):
            if isinstance(value, str) and value:
                return value
    return None


def _terminal_or_exhausted(mapping: Mapping[str, Any]) -> bool:
    lifecycle = _episode_lifecycle(mapping)
    terminal_words = (
        "terminal",
        "closed",
        "failed",
        "invalidated",
        "expired",
        "exhausted",
        "left",
        "consumed",
    )
    if any(word in lifecycle for word in terminal_words):
        return True
    episode = mapping.get("entry_episode")
    reasons = " ".join(
        str(value).lower()
        for value in _nested_values(
            episode,
            key_names=frozenset({"terminal_reason", "observed_terminal_reason"}),
        )
        if value is not None
    )
    return any(word in reasons for word in terminal_words)


def _draw_present(mapping: Mapping[str, Any]) -> bool:
    draw = mapping.get("draw")
    if not isinstance(draw, Mapping):
        return False
    for value in draw.values():
        if value is None or value == "" or value == () or value == [] or value == {}:
            continue
        return True
    return False


def _draw_identity(mapping: Mapping[str, Any]) -> str | None:
    draw = mapping.get("draw")
    for value in _nested_values(
        draw,
        key_names=frozenset(
            {
                "draw_id",
                "target_id",
                "pool_id",
                "entity_id",
                "reference_id",
                "liquidity_pool_id",
            }
        ),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def _draw_consumed(mapping: Mapping[str, Any]) -> bool:
    draw = mapping.get("draw")
    for value in _nested_values(
        draw,
        key_names=frozenset(
            {"consumed", "delivered", "draw_delivered", "lifecycle", "status"}
        ),
    ):
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in {
            "consumed",
            "delivered",
            "reached",
        }:
            return True
    return False


def _scale_alignment_class(mapping: Mapping[str, Any]) -> int:
    directions: list[int] = []
    for value in _nested_values(
        mapping.get("scale_relations"),
        key_names=frozenset({"direction", "bias", "trend_direction"}),
    ):
        try:
            direction = _direction_code(value)
        except RepresentationDataError:
            continue
        if direction:
            directions.append(direction)
    if len(directions) < 2:
        return 0  # unknown/insufficient scales
    return 1 if len(set(directions)) == 1 else 2


def _event_time_bucket(delta: pd.Timedelta) -> int:
    seconds = max(0.0, float(delta.total_seconds()))
    for index, boundary in enumerate((0.0, 60.0, 300.0, 900.0, 3600.0, 14_400.0)):
        if seconds <= boundary:
            return index
    return 6


def _next_typed_event_and_lifecycle(
    mapping: Mapping[str, Any],
    *,
    after: pd.Timestamp,
) -> tuple[int, int, pd.Timestamp | None]:
    observations: list[tuple[pd.Timestamp, str, str]] = []
    events = mapping.get("events")
    transition = (
        events.get("observation_transition", {})
        if isinstance(events, Mapping)
        else {}
    )
    transition_updates = (
        transition.get("updates", ()) if isinstance(transition, Mapping) else ()
    )
    transition_units = (
        tuple(
            update for update in transition_updates if isinstance(update, Mapping)
        )
        if isinstance(transition_updates, Sequence)
        and not isinstance(transition_updates, (str, bytes))
        and transition_updates
        else (transition,)
    )
    for update in transition_units:
        collections = update.get("collections", {})
        if not isinstance(collections, Mapping):
            continue
        update_clock = update.get("asof")
        for raw_name, values in collections.items():
            typed = tuple(_typed_transition_mappings(values))
            if not typed:
                continue
            name = str(raw_name)
            for event in typed:
                raw_clock = event.get("observed_at", update_clock)
                if raw_clock is None:
                    continue
                clock = _aware_timestamp(
                    raw_clock, name="target.typed_event.observed_at"
                )
                if clock <= after:
                    continue
                raw_lifecycle = event.get("lifecycle", "unknown")
                lifecycle = str(raw_lifecycle).strip().lower()
                observations.append(
                    (
                        clock,
                        name if name in NEXT_EVENT_TYPE_VOCAB else "<unknown>",
                        (
                            lifecycle
                            if lifecycle in NEXT_LIFECYCLE_VOCAB
                            else "unknown"
                        ),
                    )
                )
    graph = mapping.get("graph")
    if isinstance(graph, Mapping):
        raw_graph_coverage = graph.get("coverage", {})
        graph_coverage = (
            raw_graph_coverage
            if isinstance(raw_graph_coverage, Mapping)
            else {}
        )
        graph_coverage_start = graph_coverage.get("coverage_start_at")
        graph_is_complete = bool(
            graph_coverage.get("complete") is True
            and graph_coverage.get("all_typed_deltas_available") is True
            and graph_coverage.get("gap_free") is True
            and graph_coverage_start is not None
            and _aware_timestamp(
                graph_coverage_start, name="target.graph.coverage_start_at"
            )
            == after
        )
        if not graph_is_complete:
            graph = {}
    if isinstance(graph, Mapping) and graph:
        graph_updates = graph.get("updates", ())
        graph_units = (
            tuple(update for update in graph_updates if isinstance(update, Mapping))
            if isinstance(graph_updates, Sequence)
            and not isinstance(graph_updates, (str, bytes))
            and graph_updates
            else (graph,)
        )
        for update in graph_units:
            raw_graph_clock = update.get("asof")
            graph_clock = (
                None
                if raw_graph_clock is None
                else _aware_timestamp(raw_graph_clock, name="target.graph.asof")
            )
            if graph_clock is None or graph_clock <= after:
                continue
            if update.get("added_node_ids") or update.get("revised_node_ids"):
                observations.append((graph_clock, "scene_node_delta", "active"))
            if update.get("added_edge_ids") or update.get("revised_edge_ids"):
                observations.append((graph_clock, "scene_edge_delta", "active"))
            if update.get("resolution_event_ids"):
                observations.append(
                    (graph_clock, "scene_resolution", "completed")
                )

    if not observations:
        return -100, -100, None
    earliest_clock = min(item[0] for item in observations)
    earliest = [item for item in observations if item[0] == earliest_clock]
    event_families = {item[1] for item in earliest}
    lifecycles = {item[2] for item in earliest}
    event_label = (
        NEXT_EVENT_TYPE_VOCAB[next(iter(event_families))]
        if len(event_families) == 1
        else NEXT_EVENT_TYPE_VOCAB["<ambiguous>"]
    )
    lifecycle_label = (
        NEXT_LIFECYCLE_VOCAB[next(iter(lifecycles))]
        if len(lifecycles) == 1
        else NEXT_LIFECYCLE_VOCAB["<ambiguous>"]
    )
    return event_label, lifecycle_label, earliest_clock


def _has_complete_transition_coverage(
    mapping: Mapping[str, Any],
    *,
    previous_asof: pd.Timestamp,
    current_asof: pd.Timestamp,
) -> bool:
    events = mapping.get("events")
    transition = (
        events.get("observation_transition", {})
        if isinstance(events, Mapping)
        else {}
    )
    if not isinstance(transition, Mapping):
        return False
    if transition.get("typed_transition_delta_available") is not True:
        return False
    raw_coverage = transition.get("coverage", {})
    coverage = raw_coverage if isinstance(raw_coverage, Mapping) else {}
    complete = coverage.get("complete")
    if (
        complete is not True
        or coverage.get("all_typed_deltas_available") is not True
        or coverage.get("gap_free") is not True
        or coverage.get("coverage_start_exclusive") is not True
    ):
        return False
    raw_start = coverage.get("coverage_start_at")
    raw_end = coverage.get("coverage_end_at")
    if raw_start is None or raw_end is None:
        return False
    start = _aware_timestamp(raw_start, name="target.coverage_start_at")
    end = _aware_timestamp(raw_end, name="target.coverage_end_at")
    return start == previous_asof and end == current_asof


def _displacement_state_from_coverage(
    mapping: Mapping[str, Any],
    *,
    displacement_id: str | None,
    previous_asof: pd.Timestamp,
    current_asof: pd.Timestamp,
) -> int:
    """Read only exact source-entity evidence from a complete interval ledger."""

    if not displacement_id or not _has_complete_transition_coverage(
        mapping,
        previous_asof=previous_asof,
        current_asof=current_asof,
    ):
        return -100
    observed_states: set[int] = set()
    events = mapping.get("events")
    if isinstance(events, Mapping):
        invalidated = events.get("invalidated_event_ids", ())
        if isinstance(invalidated, Sequence) and not isinstance(
            invalidated, (str, bytes)
        ) and displacement_id in {str(value) for value in invalidated}:
            observed_states.add(0)
        transition = events.get("observation_transition", {})
        updates = transition.get("updates", ()) if isinstance(transition, Mapping) else ()
        if isinstance(updates, Sequence) and not isinstance(updates, (str, bytes)):
            for update in updates:
                if not isinstance(update, Mapping):
                    continue
                collections = update.get("collections", {})
                if not isinstance(collections, Mapping):
                    continue
                for values in collections.values():
                    for event in _typed_transition_mappings(values):
                        if _event_identity(event, fallback="") != displacement_id:
                            continue
                        lifecycle = str(event.get("lifecycle", "")).strip().lower()
                        if any(
                            word in lifecycle
                            for word in (
                                "terminal",
                                "closed",
                                "failed",
                                "invalidated",
                                "expired",
                                "exhausted",
                                "consumed",
                            )
                        ):
                            observed_states.add(0)
                        elif lifecycle in {
                            "active",
                            "added",
                            "formed",
                            "revised",
                            "confirmed",
                            "continued",
                        }:
                            observed_states.add(1)
    graph = mapping.get("graph")
    if isinstance(graph, Mapping):
        coverage = graph.get("coverage", {})
        if isinstance(coverage, Mapping) and coverage.get("complete") is True:
            updates = graph.get("updates", ())
            if isinstance(updates, Sequence) and not isinstance(updates, (str, bytes)):
                for update in updates:
                    if not isinstance(update, Mapping):
                        continue
                    if displacement_id in {
                        str(value) for value in update.get("resolution_event_ids", ())
                    }:
                        observed_states.add(0)
                    if displacement_id in {
                        str(value)
                        for field_name in (
                            "added_node_ids",
                            "revised_node_ids",
                        )
                        for value in update.get(field_name, ())
                    }:
                        observed_states.add(1)
    return next(iter(observed_states)) if len(observed_states) == 1 else -100


def _draw_consumption_from_coverage(
    ordered: Sequence[tuple[RepresentationCase, Mapping[str, Any]]],
    *,
    position: int,
    draw_horizon: pd.Timedelta,
) -> tuple[int, pd.Timestamp]:
    case, mapping = ordered[position]
    draw_id = _draw_identity(mapping)
    if draw_id is None:
        return -100, case.asof
    deadline = case.asof + draw_horizon
    cursor = case.asof
    label_clock = case.asof
    for future_case, future_mapping in ordered[position + 1 :]:
        if not _has_complete_transition_coverage(
            future_mapping,
            previous_asof=cursor,
            current_asof=future_case.asof,
        ):
            return -100, label_clock
        label_clock = max(label_clock, future_case.asof)
        exact_events: list[tuple[pd.Timestamp, int]] = []
        events = future_mapping.get("events")
        if isinstance(events, Mapping):
            transition = events.get("observation_transition", {})
            updates = (
                transition.get("updates", ())
                if isinstance(transition, Mapping)
                else ()
            )
            if isinstance(updates, Sequence) and not isinstance(updates, (str, bytes)):
                for update in updates:
                    if not isinstance(update, Mapping):
                        continue
                    update_clock = update.get("asof")
                    collections = update.get("collections", {})
                    if not isinstance(collections, Mapping):
                        continue
                    for values in collections.values():
                        for event in _typed_transition_mappings(values):
                            if _event_identity(event, fallback="") != draw_id:
                                continue
                            raw_clock = event.get("observed_at", update_clock)
                            if raw_clock is None:
                                continue
                            clock = _aware_timestamp(
                                raw_clock, name="target.draw_event.observed_at"
                            )
                            lifecycle = str(event.get("lifecycle", "")).lower()
                            if any(
                                word in lifecycle
                                for word in ("consumed", "delivered", "reached")
                            ):
                                exact_events.append((clock, 1))
                            elif any(
                                word in lifecycle
                                for word in (
                                    "invalidated",
                                    "expired",
                                    "failed",
                                    "closed",
                                    "terminal",
                                )
                            ):
                                exact_events.append((clock, 0))
            invalidated = events.get("invalidated_event_ids", ())
            if isinstance(invalidated, Sequence) and not isinstance(
                invalidated, (str, bytes)
            ) and draw_id in {str(value) for value in invalidated}:
                exact_events.append((future_case.asof, 0))
        for clock, state in sorted(exact_events, key=lambda item: (item[0], item[1])):
            if clock <= deadline:
                return state, label_clock
        if future_case.asof >= deadline:
            return 0, label_clock
        cursor = future_case.asof
    return -100, label_clock


def build_observable_revision_targets(
    rows: Sequence[Mapping[str, Any]],
    *,
    draw_horizon: pd.Timedelta | timedelta = timedelta(hours=1),
) -> dict[str, ObservableTargetRecord]:
    """Build separated labels from later revisions of the same epoch/episode.

    No Shadow outcome or economic result is accepted.  Later timestamps are
    recorded in ``label_max_observed_at`` and the resulting target object is
    never part of :class:`RepresentationBatch`.
    """

    draw_horizon = pd.Timedelta(draw_horizon)
    if draw_horizon <= pd.Timedelta(0, unit="s"):
        raise RepresentationDataError("draw_horizon must be positive")
    adapted = [_adapt_case_input_mapping(row) for row in rows]
    cases = [RepresentationCase.from_mapping(row) for row in adapted]
    if len({case.revision_id for case in cases}) != len(cases):
        raise RepresentationDataError("target source revision_id must be unique")
    groups: dict[
        tuple[str, str],
        list[tuple[RepresentationCase, Mapping[str, Any]]],
    ] = defaultdict(list)
    for case, mapping in zip(cases, adapted):
        groups[(case.market_epoch_id, case.entry_episode_id)].append((case, mapping))
    output: dict[str, ObservableTargetRecord] = {}
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda item: (
                item[0].revision_index,
                item[0].asof,
                item[0].revision_id,
            ),
        )
        indices = [item[0].revision_index for item in ordered]
        if len(indices) != len(set(indices)) or indices != sorted(indices):
            raise RepresentationDataError(
                "same-episode revision_index must be unique and monotone"
            )
        clocks = [item[0].asof for item in ordered]
        if clocks != sorted(clocks):
            raise RepresentationDataError(
                "same-episode observable revision clocks are not monotone"
            )
        for position, (case, mapping) in enumerate(ordered):
            next_item = ordered[position + 1] if position + 1 < len(ordered) else None
            next_case = None if next_item is None else next_item[0]
            next_mapping = None if next_item is None else next_item[1]
            label_clock = case.asof
            if next_case is None or next_mapping is None:
                next_event = next_lifecycle = next_time = displacement = -100
            else:
                if _has_complete_transition_coverage(
                    next_mapping,
                    previous_asof=case.asof,
                    current_asof=next_case.asof,
                ):
                    next_event, next_lifecycle, next_event_at = (
                        _next_typed_event_and_lifecycle(
                            next_mapping,
                            after=case.asof,
                        )
                    )
                    next_time = (
                        -100
                        if next_event_at is None
                        else _event_time_bucket(next_event_at - case.asof)
                    )
                else:
                    next_event = next_lifecycle = next_time = -100
                displacement = _displacement_state_from_coverage(
                    next_mapping,
                    displacement_id=_source_displacement_id(mapping),
                    previous_asof=case.asof,
                    current_asof=next_case.asof,
                )
                label_clock = max(label_clock, next_case.asof)

            draw_label, draw_label_clock = _draw_consumption_from_coverage(
                ordered,
                position=position,
                draw_horizon=draw_horizon,
            )
            label_clock = max(label_clock, draw_label_clock)
            target = SelfSupervisedTarget(
                next_event_type=next_event,
                next_lifecycle=next_lifecycle,
                next_event_time_bucket=next_time,
                displacement_state=displacement,
                draw_consumed=draw_label,
                scale_direction_alignment=_scale_alignment_class(mapping),
            )
            output[case.revision_id] = ObservableTargetRecord(
                revision_id=case.revision_id,
                market_epoch_id=case.market_epoch_id,
                entry_episode_id=case.entry_episode_id,
                input_asof=case.asof,
                label_max_observed_at=label_clock,
                next_revision_id=None if next_case is None else next_case.revision_id,
                target=target,
            )
    return output


@dataclass
class RepresentationBatch:
    timeframe_features: Mapping[str, Tensor]
    timeframe_padding_masks: Mapping[str, Tensor]
    candle_reconstruction_masks: Mapping[str, Tensor]
    candle_reconstruction_targets: Mapping[str, Tensor]
    event_type_ids: Tensor
    lifecycle_ids: Tensor
    relation_ids: Tensor
    scale_ids: Tensor
    event_numeric: Tensor
    event_padding_mask: Tensor
    event_reconstruction_mask: Tensor
    event_reconstruction_target: Tensor
    context_group: Tensor
    episode_group: Tensor
    epoch_group: Tensor
    authority_direction: Tensor
    case_ids: tuple[str, ...]
    revision_ids: tuple[str, ...]

    def to(self, device: Any) -> "RepresentationBatch":
        require_torch()
        return RepresentationBatch(
            timeframe_features={key: value.to(device) for key, value in self.timeframe_features.items()},
            timeframe_padding_masks={key: value.to(device) for key, value in self.timeframe_padding_masks.items()},
            candle_reconstruction_masks={key: value.to(device) for key, value in self.candle_reconstruction_masks.items()},
            candle_reconstruction_targets={key: value.to(device) for key, value in self.candle_reconstruction_targets.items()},
            event_type_ids=self.event_type_ids.to(device),
            lifecycle_ids=self.lifecycle_ids.to(device),
            relation_ids=self.relation_ids.to(device),
            scale_ids=self.scale_ids.to(device),
            event_numeric=self.event_numeric.to(device),
            event_padding_mask=self.event_padding_mask.to(device),
            event_reconstruction_mask=self.event_reconstruction_mask.to(device),
            event_reconstruction_target=self.event_reconstruction_target.to(device),
            context_group=self.context_group.to(device),
            episode_group=self.episode_group.to(device),
            epoch_group=self.epoch_group.to(device),
            authority_direction=self.authority_direction.to(device),
            case_ids=self.case_ids,
            revision_ids=self.revision_ids,
        )


@dataclass
class TargetBatch:
    next_event_type: Tensor
    next_lifecycle: Tensor
    next_event_time_bucket: Tensor
    displacement_state: Tensor
    draw_consumed: Tensor
    scale_direction_alignment: Tensor

    def to(self, device: Any) -> "TargetBatch":
        require_torch()
        return TargetBatch(**{name: value.to(device) for name, value in vars(self).items()})


def collate_representation_cases(
    examples: Sequence[PreparedRepresentationCase],
    *,
    targets: Sequence[SelfSupervisedTarget] | None = None,
    max_lengths: Mapping[str, int] | None = None,
    max_events: int = 128,
    mask_probability: float = 0.15,
    seed: int = 17,
) -> tuple[RepresentationBatch, TargetBatch | None]:
    """Pad, truncate from the left, and mask without introducing future rows."""

    require_torch()
    if not examples:
        raise RepresentationDataError("cannot collate an empty batch")
    if targets is not None and len(targets) != len(examples):
        raise RepresentationDataError("target count does not match example count")
    if not 0 <= mask_probability < 1:
        raise RepresentationDataError("mask_probability must be in [0, 1)")
    lengths = dict(max_lengths or {"4h": 64, "1h": 96, "15m": 128, "5m": 192, "1m": 256})
    if set(lengths) != set(TIMEFRAMES) or any(int(value) < 1 for value in lengths.values()):
        raise RepresentationDataError("max_lengths must bind every timeframe positively")
    if max_events < 1:
        raise RepresentationDataError("max_events must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    batch_size = len(examples)
    feature_tensors: dict[str, Tensor] = {}
    padding_masks: dict[str, Tensor] = {}
    reconstruction_masks: dict[str, Tensor] = {}
    reconstruction_targets: dict[str, Tensor] = {}
    for timeframe in TIMEFRAMES:
        width = int(lengths[timeframe])
        tensor = torch.zeros((batch_size, width, len(CAUSAL_CANDLE_FEATURES)), dtype=torch.float32)
        valid = torch.zeros((batch_size, width), dtype=torch.bool)
        for index, example in enumerate(examples):
            values = example.timeframe_features[timeframe][-width:]
            count = len(values)
            tensor[index, :count] = torch.from_numpy(np.asarray(values, dtype=np.float32))
            valid[index, :count] = True
        target = tensor.clone()
        masked = (torch.rand((batch_size, width), generator=generator) < mask_probability) & valid
        tensor[masked] = 0.0
        feature_tensors[timeframe] = tensor
        padding_masks[timeframe] = valid
        reconstruction_masks[timeframe] = masked
        reconstruction_targets[timeframe] = target

    event_width = min(max_events, max(len(example.event_type_ids) for example in examples))
    type_ids = torch.full((batch_size, event_width), PAD_TOKEN_ID, dtype=torch.long)
    lifecycle_ids = torch.full_like(type_ids, PAD_TOKEN_ID)
    relation_ids = torch.full_like(type_ids, PAD_TOKEN_ID)
    scale_ids = torch.full_like(type_ids, PAD_TOKEN_ID)
    event_numeric = torch.zeros((batch_size, event_width, 4), dtype=torch.float32)
    event_valid = torch.zeros((batch_size, event_width), dtype=torch.bool)
    for index, example in enumerate(examples):
        count = min(event_width, len(example.event_type_ids))
        event_slice = slice(len(example.event_type_ids) - count, None)
        type_ids[index, :count] = torch.from_numpy(example.event_type_ids[event_slice])
        lifecycle_ids[index, :count] = torch.from_numpy(example.lifecycle_ids[event_slice])
        relation_ids[index, :count] = torch.from_numpy(example.relation_ids[event_slice])
        scale_ids[index, :count] = torch.from_numpy(example.scale_ids[event_slice])
        event_numeric[index, :count] = torch.from_numpy(example.event_numeric[event_slice])
        event_valid[index, :count] = True
    event_target = type_ids.clone()
    event_masked = (torch.rand((batch_size, event_width), generator=generator) < mask_probability) & event_valid
    type_ids[event_masked] = MASK_TOKEN_ID

    batch = RepresentationBatch(
        timeframe_features=feature_tensors,
        timeframe_padding_masks=padding_masks,
        candle_reconstruction_masks=reconstruction_masks,
        candle_reconstruction_targets=reconstruction_targets,
        event_type_ids=type_ids,
        lifecycle_ids=lifecycle_ids,
        relation_ids=relation_ids,
        scale_ids=scale_ids,
        event_numeric=event_numeric,
        event_padding_mask=event_valid,
        event_reconstruction_mask=event_masked,
        event_reconstruction_target=event_target,
        context_group=torch.tensor([_stable_group(item.case.context_thesis_id) for item in examples], dtype=torch.long),
        episode_group=torch.tensor(
            [
                _stable_group(
                    f"{item.case.market_epoch_id}\x1f{item.case.entry_episode_id}"
                )
                for item in examples
            ],
            dtype=torch.long,
        ),
        epoch_group=torch.tensor([_stable_group(item.case.market_epoch_id) for item in examples], dtype=torch.long),
        authority_direction=torch.tensor(
            [item.case.authority_direction for item in examples], dtype=torch.long
        ),
        case_ids=tuple(item.case.case_id for item in examples),
        revision_ids=tuple(item.case.revision_id for item in examples),
    )
    target_batch = None
    if targets is not None:
        target_batch = TargetBatch(
            **{
                name: torch.tensor([getattr(item, name) for item in targets], dtype=torch.long)
                for name in sorted(ALLOWED_SELF_SUPERVISED_TARGETS)
            }
        )
    return batch, target_batch


@dataclass(frozen=True)
class MarketRepresentationConfig:
    candle_feature_dim: int = len(CAUSAL_CANDLE_FEATURES)
    temporal_hidden_dim: int = 64
    temporal_layers: int = 2
    event_hidden_dim: int = 64
    event_layers: int = 2
    fusion_hidden_dim: int = 256
    embedding_dim: int = EMBEDDING_DIM
    event_type_vocab: int = 512
    lifecycle_vocab: int = 64
    relation_vocab: int = 256
    scale_vocab: int = 32
    next_event_classes: int = len(NEXT_EVENT_TYPE_VOCAB)
    next_lifecycle_classes: int = len(NEXT_LIFECYCLE_VOCAB)
    time_bucket_classes: int = 8
    displacement_classes: int = 2
    draw_classes: int = 2
    scale_alignment_classes: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.candle_feature_dim != len(CAUSAL_CANDLE_FEATURES):
            raise RepresentationDataError("model candle feature schema mismatch")
        if self.embedding_dim != EMBEDDING_DIM:
            raise RepresentationDataError("v1 embedding dimension is fixed at 128")
        if self.temporal_layers < 1 or self.event_layers < 1:
            raise RepresentationDataError("encoder layer counts must be positive")
        for name in (
            "event_type_vocab",
            "lifecycle_vocab",
            "relation_vocab",
            "scale_vocab",
        ):
            if getattr(self, name) <= MASK_TOKEN_ID:
                raise RepresentationDataError(f"{name} must include reserved tokens")
        if not 0 <= self.dropout < 1:
            raise RepresentationDataError("dropout must be in [0, 1)")


@dataclass
class RepresentationOutput:
    embedding: Tensor
    candle_reconstruction: Mapping[str, Tensor]
    event_reconstruction_logits: Tensor
    next_event_logits: Tensor
    next_lifecycle_logits: Tensor
    next_event_time_logits: Tensor
    displacement_logits: Tensor
    draw_consumed_logits: Tensor
    scale_alignment_logits: Tensor


@dataclass
class LossBreakdown:
    total: Tensor
    components: Mapping[str, Tensor]


@dataclass(frozen=True)
class DecisionTimeEmbeddingRecord:
    """Outcome-free handoff consumed by similarity/OOD infrastructure."""

    case_id: str
    revision_id: str
    revision_stage: str
    revision_index: int
    stage_identity: str
    stage_occurrence: int
    context_thesis_id: str
    entry_episode_id: str
    market_epoch_id: str
    direction: int
    regime: str
    embedding_model_version: str
    embedding_dim: int
    embedding_clock: str
    embedding_asof: pd.Timestamp
    feature_max_at: pd.Timestamp
    data_split: str
    split_role: str
    outcome_fields_used: bool
    embedding_checkpoint_id: str
    decision_embedding: tuple[float, ...]
    mechanism_label: str = "unknown"
    embedding_input_protocol: str = INFERENCE_INPUT_PROTOCOL

    def __post_init__(self) -> None:
        if self.revision_stage not in CASE_REVISION_STAGES:
            raise RepresentationDataError("embedding revision_stage is invalid")
        if isinstance(self.revision_index, bool) or self.revision_index < 0:
            raise RepresentationDataError("embedding revision_index is invalid")
        _nonempty(self.stage_identity, name="embedding.stage_identity")
        if self.stage_occurrence != 0:
            raise RepresentationDataError(
                "v1 retrieval corpus admits only the first causal stage occurrence"
            )
        if self.embedding_dim != EMBEDDING_DIM or len(self.decision_embedding) != EMBEDDING_DIM:
            raise RepresentationDataError("decision embedding must contain exactly 128 values")
        if len(self.embedding_checkpoint_id) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.embedding_checkpoint_id
        ):
            raise RepresentationDataError(
                "embedding_checkpoint_id must be a lowercase content SHA-256"
            )
        if self.embedding_clock != "decision_time":
            raise RepresentationDataError("embedding clock must be decision_time")
        if self.embedding_input_protocol != INFERENCE_INPUT_PROTOCOL:
            raise RepresentationDataError(
                "decision embedding must use the deterministic unmasked input protocol"
            )
        if self.outcome_fields_used:
            raise RepresentationDataError("decision embedding cannot use outcome fields")
        asof = _aware_timestamp(self.embedding_asof, name="embedding_asof")
        feature_max = _aware_timestamp(self.feature_max_at, name="feature_max_at")
        if feature_max > asof:
            raise RepresentationDataError("embedding feature clock exceeds decision time")
        values = np.asarray(self.decision_embedding, dtype=np.float64)
        if not np.isfinite(values).all():
            raise RepresentationDataError("decision embedding must be finite")
        if not math.isclose(float(np.linalg.norm(values)), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RepresentationDataError("decision embedding must be L2-normalized")

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "revision_id": self.revision_id,
            "revision_stage": self.revision_stage,
            "revision_index": self.revision_index,
            "stage_identity": self.stage_identity,
            "stage_occurrence": self.stage_occurrence,
            "context_thesis_id": self.context_thesis_id,
            "entry_episode_id": self.entry_episode_id,
            "market_epoch_id": self.market_epoch_id,
            "direction": self.direction,
            "regime": self.regime,
            "mechanism_label": self.mechanism_label,
            "embedding_model_version": self.embedding_model_version,
            "embedding_dim": self.embedding_dim,
            "embedding_clock": self.embedding_clock,
            "embedding_asof": self.embedding_asof.isoformat(),
            "feature_max_at": self.feature_max_at.isoformat(),
            "data_split": self.data_split,
            "split_role": self.split_role,
            "outcome_fields_used": self.outcome_fields_used,
            "embedding_checkpoint_id": self.embedding_checkpoint_id,
            "embedding_input_protocol": self.embedding_input_protocol,
            "decision_embedding": list(self.decision_embedding),
        }


@dataclass(frozen=True)
class EmbeddingEvaluationSample:
    """Outcome-free metadata used only to evaluate representation geometry."""

    revision_id: str
    entry_episode_id: str
    asof: pd.Timestamp
    direction: int
    regime: str
    mechanism_label: str
    embedding: tuple[float, ...]
    label_sources_masked: bool = False

    def __post_init__(self) -> None:
        _nonempty(self.revision_id, name="evaluation.revision_id")
        _nonempty(self.entry_episode_id, name="evaluation.entry_episode_id")
        object.__setattr__(self, "asof", _aware_timestamp(self.asof, name="evaluation.asof"))
        object.__setattr__(self, "direction", _direction_code(self.direction))
        object.__setattr__(self, "regime", str(self.regime).strip().lower() or "unknown")
        object.__setattr__(
            self,
            "mechanism_label",
            str(self.mechanism_label).strip().lower() or "unknown",
        )
        object.__setattr__(
            self, "label_sources_masked", bool(self.label_sources_masked)
        )
        vector = np.asarray(self.embedding, dtype=np.float64)
        if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
            raise RepresentationDataError("evaluation embedding must be finite 128d")
        if not math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RepresentationDataError("evaluation embedding must be L2-normalized")


@dataclass(frozen=True)
class DecisionTimeHeadPredictionRecord:
    """One ensemble member's heads bound to an exact decision revision clock."""

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
        _nonempty(self.member_id, name="head_prediction.member_id")
        if len(self.checkpoint_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.checkpoint_id
        ):
            raise RepresentationDataError("head prediction checkpoint_id is invalid")
        decision = _aware_timestamp(self.decision_at, name="head_prediction.decision_at")
        feature_max = _aware_timestamp(
            self.feature_max_at, name="head_prediction.feature_max_at"
        )
        if feature_max > decision:
            raise RepresentationDataError("head prediction feature clock exceeds decision")
        if self.outcome_fields_used:
            raise RepresentationDataError("ensemble heads cannot consume outcomes")
        if self.input_protocol != INFERENCE_INPUT_PROTOCOL:
            raise RepresentationDataError(
                "ensemble heads must use the deterministic unmasked input protocol"
            )
        expected = OUTCOME_BLIND_HEAD_WIDTHS
        if set(self.head_predictions) != set(expected):
            raise RepresentationDataError("ensemble head prediction schema mismatch")
        for name, width in expected.items():
            values = np.asarray(self.head_predictions[name], dtype=np.float64)
            if values.shape != (width,) or not np.isfinite(values).all():
                raise RepresentationDataError(f"ensemble head {name} is invalid")
            if bool((values < 0).any()) or not math.isclose(
                float(values.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5
            ):
                raise RepresentationDataError(
                    f"ensemble head {name} must be a probability vector"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "checkpoint_id": self.checkpoint_id,
            "model_version": self.model_version,
            "case_id": self.case_id,
            "revision_id": self.revision_id,
            "entry_episode_id": self.entry_episode_id,
            "decision_at": self.decision_at.isoformat(),
            "feature_max_at": self.feature_max_at.isoformat(),
            "outcome_fields_used": self.outcome_fields_used,
            "input_protocol": self.input_protocol,
            "head_predictions": {
                name: list(values)
                for name, values in sorted(self.head_predictions.items())
            },
        }


def validate_single_embedding_checkpoint(
    records: Sequence[DecisionTimeEmbeddingRecord | Mapping[str, Any]],
) -> str:
    """Reject a vector index assembled from incompatible encoder weights."""

    if not records:
        raise RepresentationDataError("embedding checkpoint audit needs records")
    identities = {
        (
            record.embedding_checkpoint_id
            if isinstance(record, DecisionTimeEmbeddingRecord)
            else str(record.get("embedding_checkpoint_id", ""))
        )
        for record in records
    }
    if any(
        len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        for identity in identities
    ):
        raise RepresentationDataError("embedding record has invalid checkpoint identity")
    if len(identities) != 1:
        raise RepresentationDataError(
            "embedding vectors from different encoder checkpoints cannot share an index"
        )
    return next(iter(identities))


def evaluate_outcome_blind_embedding_space(
    reference: Sequence[EmbeddingEvaluationSample],
    queries: Sequence[EmbeddingEvaluationSample],
    *,
    k: int = 5,
    minimum_regime_samples: int = 5,
    minimum_direction_regime_samples: int = MIN_DIRECTION_REGIME_SAMPLES,
    minimum_group_feature_std: float = MIN_GROUP_FEATURE_STD,
) -> dict[str, Any]:
    """Measure regime separation and cross-date causal-mechanism retrieval.

    The function reports insufficiency instead of manufacturing a pass.  It
    accepts no outcome/result field and makes no profitability claim.
    """

    if not reference or not queries:
        raise RepresentationDataError("embedding evaluation needs reference and queries")
    if (
        k < 1
        or minimum_regime_samples < 1
        or minimum_direction_regime_samples < 2
        or minimum_group_feature_std <= 0
    ):
        raise RepresentationDataError(
            "evaluation k, coverage and non-collapse thresholds must be positive"
        )
    reference_matrix = np.asarray([item.embedding for item in reference], dtype=np.float64)
    query_matrix = np.asarray([item.embedding for item in queries], dtype=np.float64)
    canonical_regimes = ("continuation", "sweep_failure", "balance", "unknown")
    regime_counts = {
        regime: sum(item.regime == regime for item in reference)
        for regime in canonical_regimes
    }
    centroids: dict[str, np.ndarray] = {}
    for regime in canonical_regimes:
        selected = reference_matrix[
            np.asarray([item.regime == regime for item in reference], dtype=bool)
        ]
        if len(selected):
            centroid = selected.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroids[regime] = centroid / norm
    evaluated_queries = [
        (index, item)
        for index, item in enumerate(queries)
        if item.regime in centroids
    ]
    centroid_hits: list[bool] = []
    for index, item in evaluated_queries:
        predicted = max(
            centroids,
            key=lambda regime: (
                float(query_matrix[index] @ centroids[regime]),
                regime,
            ),
        )
        centroid_hits.append(predicted == item.regime)
    centroid_accuracy = float(np.mean(centroid_hits)) if centroid_hits else float("nan")
    majority_accuracy = max(regime_counts.values()) / len(reference)

    retrieval_hits: list[bool] = []
    retrieval_precisions: list[float] = []
    retrieval_chance_hits: list[float] = []
    sufficient_neighbours = 0
    for query_index, query in enumerate(queries):
        candidates = [
            (reference_index, candidate)
            for reference_index, candidate in enumerate(reference)
            if candidate.entry_episode_id != query.entry_episode_id
            and candidate.asof.date() != query.asof.date()
        ]
        if len(candidates) < k or query.mechanism_label == "unknown":
            continue
        sufficient_neighbours += 1
        matching_candidates = sum(
            candidate.mechanism_label == query.mechanism_label
            for _, candidate in candidates
        )
        nonmatching_candidates = len(candidates) - matching_candidates
        chance_miss = (
            math.comb(nonmatching_candidates, k) / math.comb(len(candidates), k)
            if nonmatching_candidates >= k
            else 0.0
        )
        retrieval_chance_hits.append(1.0 - chance_miss)
        ranked = sorted(
            candidates,
            key=lambda pair: (
                -float(query_matrix[query_index] @ reference_matrix[pair[0]]),
                pair[1].revision_id,
            ),
        )[:k]
        matches = [
            candidate.mechanism_label == query.mechanism_label
            for _, candidate in ranked
        ]
        retrieval_hits.append(any(matches))
        retrieval_precisions.append(float(np.mean(matches)))

    group_diagnostics: dict[str, Any] = {}
    query_group_diagnostics: dict[str, Any] = {}
    for direction in (-1, 0, 1):
        for regime in canonical_regimes:
            selected = reference_matrix[
                np.asarray(
                    [
                        item.direction == direction and item.regime == regime
                        for item in reference
                    ],
                    dtype=bool,
                )
            ]
            key = f"direction={direction}|regime={regime}"
            group_diagnostics[key] = {
                "count": int(len(selected)),
                "mean_feature_std": (
                    float(np.mean(np.std(selected, axis=0)))
                    if len(selected) >= 2
                    else float("nan")
                ),
            }
            query_selected_indices = [
                index
                for index, item in enumerate(queries)
                if item.direction == direction and item.regime == regime
            ]
            query_selected = query_matrix[query_selected_indices]
            group_hits = []
            for index in query_selected_indices:
                predicted = max(
                    centroids,
                    key=lambda candidate_regime: (
                        float(query_matrix[index] @ centroids[candidate_regime]),
                        candidate_regime,
                    ),
                )
                group_hits.append(predicted == queries[index].regime)
            query_group_diagnostics[key] = {
                "count": int(len(query_selected)),
                "mean_feature_std": (
                    float(np.mean(np.std(query_selected, axis=0)))
                    if len(query_selected) >= 2
                    else float("nan")
                ),
                "centroid_accuracy": (
                    float(np.mean(group_hits)) if group_hits else float("nan")
                ),
            }
    mean_feature_std = float(np.mean(np.std(reference_matrix, axis=0)))
    regime_evidence_sufficient = all(
        regime_counts[regime] >= minimum_regime_samples
        for regime in canonical_regimes
    )
    retrieval_evidence_sufficient = sufficient_neighbours == len(queries)
    required_group_keys = tuple(
        f"direction={direction}|regime={regime}"
        for direction in (-1, 1)
        for regime in canonical_regimes
    )
    reference_direction_regime_coverage_sufficient = all(
        group_diagnostics[key]["count"] >= minimum_direction_regime_samples
        for key in required_group_keys
    )
    query_direction_regime_coverage_sufficient = all(
        query_group_diagnostics[key]["count"] >= minimum_direction_regime_samples
        for key in required_group_keys
    )
    direction_regime_coverage_sufficient = bool(
        reference_direction_regime_coverage_sufficient
        and query_direction_regime_coverage_sufficient
    )
    reference_direction_regime_noncollapse = bool(
        reference_direction_regime_coverage_sufficient
    ) and all(
        math.isfinite(group_diagnostics[key]["mean_feature_std"])
        and group_diagnostics[key]["mean_feature_std"] >= minimum_group_feature_std
        for key in required_group_keys
    )
    query_direction_regime_noncollapse = bool(
        query_direction_regime_coverage_sufficient
    ) and all(
        math.isfinite(query_group_diagnostics[key]["mean_feature_std"])
        and query_group_diagnostics[key]["mean_feature_std"]
        >= minimum_group_feature_std
        for key in required_group_keys
    )
    direction_regime_noncollapse = bool(
        reference_direction_regime_noncollapse
        and query_direction_regime_noncollapse
    )
    evidence_sufficient = bool(
        regime_evidence_sufficient
        and retrieval_evidence_sufficient
        and direction_regime_coverage_sufficient
        and direction_regime_noncollapse
    )
    retrieval_at_k = float(np.mean(retrieval_hits)) if retrieval_hits else float("nan")
    precision_at_k = (
        float(np.mean(retrieval_precisions)) if retrieval_precisions else float("nan")
    )
    retrieval_chance_at_k = (
        float(np.mean(retrieval_chance_hits))
        if retrieval_chance_hits
        else float("nan")
    )
    shortcut_probe_safe = all(
        item.label_sources_masked for item in (*reference, *queries)
    )
    geometry_criteria_met = bool(
        evidence_sufficient
        and math.isfinite(centroid_accuracy)
        and centroid_accuracy
        >= majority_accuracy + MIN_REGIME_ACCURACY_MARGIN
        and math.isfinite(retrieval_at_k)
        and retrieval_at_k >= MIN_CROSS_DATE_RETRIEVAL_AT_K
        and math.isfinite(retrieval_chance_at_k)
        and retrieval_at_k
        >= retrieval_chance_at_k + MIN_CROSS_DATE_RETRIEVAL_LIFT
        and mean_feature_std > 1e-5
    )
    criteria_met = bool(geometry_criteria_met and shortcut_probe_safe)
    return {
        "status": "measured" if evidence_sufficient else "insufficient_evidence",
        "criteria_met": criteria_met,
        "geometry_criteria_met": geometry_criteria_met,
        "reference_count": len(reference),
        "query_count": len(queries),
        "regime_counts": regime_counts,
        "minimum_regime_samples": minimum_regime_samples,
        "regime_centroid_accuracy": centroid_accuracy,
        "regime_majority_baseline_accuracy": majority_accuracy,
        "cross_date_mechanism_retrieval_at_k": retrieval_at_k,
        "cross_date_mechanism_precision_at_k": precision_at_k,
        "cross_date_mechanism_chance_at_k": retrieval_chance_at_k,
        "cross_date_mechanism_lift_over_chance": (
            retrieval_at_k - retrieval_chance_at_k
            if math.isfinite(retrieval_at_k)
            and math.isfinite(retrieval_chance_at_k)
            else float("nan")
        ),
        "queries_with_sufficient_independent_neighbours": sufficient_neighbours,
        "k": k,
        "mean_feature_std": mean_feature_std,
        "direction_regime_groups": group_diagnostics,
        "held_out_direction_regime_groups": query_group_diagnostics,
        "minimum_direction_regime_samples": minimum_direction_regime_samples,
        "minimum_group_feature_std": minimum_group_feature_std,
        "direction_regime_coverage_sufficient": direction_regime_coverage_sufficient,
        "direction_regime_noncollapse": direction_regime_noncollapse,
        "reference_direction_regime_coverage_sufficient": reference_direction_regime_coverage_sufficient,
        "held_out_direction_regime_coverage_sufficient": query_direction_regime_coverage_sufficient,
        "reference_direction_regime_noncollapse": reference_direction_regime_noncollapse,
        "held_out_direction_regime_noncollapse": query_direction_regime_noncollapse,
        "pre_registered_thresholds": {
            "minimum_regime_accuracy_margin_over_majority": MIN_REGIME_ACCURACY_MARGIN,
            "minimum_cross_date_retrieval_at_k": MIN_CROSS_DATE_RETRIEVAL_AT_K,
            "minimum_cross_date_retrieval_lift_over_chance": MIN_CROSS_DATE_RETRIEVAL_LIFT,
        },
        "label_source_shortcut_audit": {
            "known_direct_sources": DIRECT_LABEL_SOURCE_AUDIT,
            "masked_label_probe_used": shortcut_probe_safe,
            "criteria_blocked_by_uncontrolled_shortcut": not shortcut_probe_safe,
        },
        "outcome_fields_used": False,
        "trading_edge_claimed": False,
    }


if TORCH_AVAILABLE:

    class _TemporalGRUEncoder(nn.Module):
        def __init__(self, config: MarketRepresentationConfig) -> None:
            super().__init__()
            dropout = config.dropout if config.temporal_layers > 1 else 0.0
            self.gru = nn.GRU(
                config.candle_feature_dim,
                config.temporal_hidden_dim,
                num_layers=config.temporal_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.reconstruction = nn.Linear(
                config.temporal_hidden_dim, config.candle_feature_dim
            )

        def forward(self, values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            lengths = valid.sum(dim=1)
            if bool((lengths < 1).any()):
                raise RepresentationDataError("each timeframe needs at least one prefix row")
            packed = pack_padded_sequence(
                values,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, hidden = self.gru(packed)
            sequence, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=values.shape[1]
            )
            pooled = hidden[-1]
            return pooled, sequence, self.reconstruction(sequence)


    class _EventGraphGRUEncoder(nn.Module):
        def __init__(self, config: MarketRepresentationConfig) -> None:
            super().__init__()
            self.event_type = nn.Embedding(config.event_type_vocab, 24, padding_idx=PAD_TOKEN_ID)
            self.lifecycle = nn.Embedding(config.lifecycle_vocab, 12, padding_idx=PAD_TOKEN_ID)
            self.relation = nn.Embedding(config.relation_vocab, 16, padding_idx=PAD_TOKEN_ID)
            self.scale = nn.Embedding(config.scale_vocab, 8, padding_idx=PAD_TOKEN_ID)
            self.numeric = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.LayerNorm(16))
            input_dim = 24 + 12 + 16 + 8 + 16
            dropout = config.dropout if config.event_layers > 1 else 0.0
            self.gru = nn.GRU(
                input_dim,
                config.event_hidden_dim,
                num_layers=config.event_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.reconstruction = nn.Linear(config.event_hidden_dim, config.event_type_vocab)

        def forward(self, batch: RepresentationBatch) -> tuple[Tensor, Tensor, Tensor]:
            values = torch.cat(
                (
                    self.event_type(batch.event_type_ids),
                    self.lifecycle(batch.lifecycle_ids),
                    self.relation(batch.relation_ids),
                    self.scale(batch.scale_ids),
                    self.numeric(batch.event_numeric),
                ),
                dim=-1,
            )
            lengths = batch.event_padding_mask.sum(dim=1)
            if bool((lengths < 1).any()):
                raise RepresentationDataError("each case needs at least one event token")
            packed = pack_padded_sequence(
                values,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, hidden = self.gru(packed)
            sequence, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=values.shape[1],
            )
            return hidden[-1], sequence, self.reconstruction(sequence)


    class MarketRepresentationModel(nn.Module):
        """Five GRU candle encoders plus one event/graph GRU encoder."""

        def __init__(self, config: MarketRepresentationConfig | None = None) -> None:
            super().__init__()
            self.config = config or MarketRepresentationConfig()
            self.temporal_encoders = nn.ModuleDict(
                {timeframe: _TemporalGRUEncoder(self.config) for timeframe in TIMEFRAMES}
            )
            self.event_encoder = _EventGraphGRUEncoder(self.config)
            fused_width = len(TIMEFRAMES) * self.config.temporal_hidden_dim + self.config.event_hidden_dim
            self.fusion = nn.Sequential(
                nn.LayerNorm(fused_width),
                nn.Linear(fused_width, self.config.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.fusion_hidden_dim, self.config.embedding_dim),
                nn.LayerNorm(self.config.embedding_dim),
            )
            self.next_event = nn.Linear(self.config.embedding_dim, self.config.next_event_classes)
            self.next_lifecycle = nn.Linear(self.config.embedding_dim, self.config.next_lifecycle_classes)
            self.next_event_time = nn.Linear(self.config.embedding_dim, self.config.time_bucket_classes)
            self.displacement = nn.Linear(self.config.embedding_dim, self.config.displacement_classes)
            self.draw_consumed = nn.Linear(self.config.embedding_dim, self.config.draw_classes)
            self.scale_alignment = nn.Linear(self.config.embedding_dim, self.config.scale_alignment_classes)
            if self.parameter_count() >= PARAMETER_BUDGET:
                raise RepresentationDataError(
                    f"model has {self.parameter_count():,} parameters; budget is <{PARAMETER_BUDGET:,}"
                )

        def parameter_count(self, *, trainable_only: bool = True) -> int:
            parameters = (
                (parameter for parameter in self.parameters() if parameter.requires_grad)
                if trainable_only
                else self.parameters()
            )
            return sum(parameter.numel() for parameter in parameters)

        def _encode_with_sequences(
            self, batch: RepresentationBatch
        ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
            pooled: list[Tensor] = []
            reconstructions: dict[str, Tensor] = {}
            for timeframe in TIMEFRAMES:
                encoded, _, reconstructed = self.temporal_encoders[timeframe](
                    batch.timeframe_features[timeframe],
                    batch.timeframe_padding_masks[timeframe],
                )
                pooled.append(encoded)
                reconstructions[timeframe] = reconstructed
            event_pooled, _, event_reconstruction = self.event_encoder(batch)
            pooled.append(event_pooled)
            embedding = F.normalize(self.fusion(torch.cat(pooled, dim=-1)), p=2, dim=-1)
            return embedding, reconstructions, event_reconstruction

        def encode(self, batch: RepresentationBatch) -> Tensor:
            """Return only a decision-time embedding; targets/outcomes are inaccessible."""

            embedding, _, _ = self._encode_with_sequences(batch)
            return embedding

        def forward(self, batch: RepresentationBatch) -> RepresentationOutput:
            embedding, candle_reconstruction, event_reconstruction = self._encode_with_sequences(batch)
            return RepresentationOutput(
                embedding=embedding,
                candle_reconstruction=candle_reconstruction,
                event_reconstruction_logits=event_reconstruction,
                next_event_logits=self.next_event(embedding),
                next_lifecycle_logits=self.next_lifecycle(embedding),
                next_event_time_logits=self.next_event_time(embedding),
                displacement_logits=self.displacement(embedding),
                draw_consumed_logits=self.draw_consumed(embedding),
                scale_alignment_logits=self.scale_alignment(embedding),
            )


else:

    class MarketRepresentationModel:  # pragma: no cover - environment branch.
        """Fail-closed placeholder when the optional dependency is absent."""

        def __init__(self, config: MarketRepresentationConfig | None = None) -> None:
            del config
            require_torch()


def supervised_causal_contrastive_loss(
    embedding: Tensor,
    batch: RepresentationBatch,
    *,
    negative_margin: float = 0.2,
) -> Tensor:
    """Pull sibling episodes together; separate epochs/opposite authorities."""

    require_torch()
    similarity = embedding @ embedding.transpose(0, 1)
    size = embedding.shape[0]
    off_diagonal = ~torch.eye(size, dtype=torch.bool, device=embedding.device)
    positive = (
        (batch.context_group[:, None] == batch.context_group[None, :])
        & (batch.episode_group[:, None] != batch.episode_group[None, :])
        & (batch.epoch_group[:, None] == batch.epoch_group[None, :])
        & (
            (batch.authority_direction[:, None] * batch.authority_direction[None, :])
            >= 0
        )
        & off_diagonal
    )
    opposite_authority = (
        (batch.authority_direction[:, None] * batch.authority_direction[None, :]) < 0
    )
    negative = (
        (
            (batch.epoch_group[:, None] != batch.epoch_group[None, :])
            | opposite_authority
        )
        & off_diagonal
    )
    zero = embedding.sum() * 0.0
    positive_loss = (1.0 - similarity[positive]).mean() if bool(positive.any()) else zero
    negative_loss = (
        torch.relu(similarity[negative] - negative_margin).mean()
        if bool(negative.any())
        else zero
    )
    return positive_loss + negative_loss


def representation_multitask_loss(
    output: RepresentationOutput,
    batch: RepresentationBatch,
    targets: TargetBatch,
    *,
    weights: Mapping[str, float] | None = None,
) -> LossBreakdown:
    """Compute self-supervised losses; outcome fields have no parameter path."""

    require_torch()
    configured = {
        "candle_reconstruction": 1.0,
        "event_reconstruction": 1.0,
        "next_event": 1.0,
        "next_lifecycle": 0.75,
        "next_event_time": 0.5,
        "displacement": 0.5,
        "draw_consumed": 0.5,
        "scale_alignment": 0.5,
        "contrastive": 0.25,
    }
    if weights is not None:
        unknown = sorted(set(weights) - set(configured))
        if unknown:
            raise RepresentationDataError(f"unknown loss weights: {unknown}")
        configured.update({key: float(value) for key, value in weights.items()})
    zero = output.embedding.sum() * 0.0
    candle_terms: list[Tensor] = []
    for timeframe in TIMEFRAMES:
        mask = batch.candle_reconstruction_masks[timeframe]
        if bool(mask.any()):
            candle_terms.append(
                F.smooth_l1_loss(
                    output.candle_reconstruction[timeframe][mask],
                    batch.candle_reconstruction_targets[timeframe][mask],
                )
            )
    components: dict[str, Tensor] = {
        "candle_reconstruction": torch.stack(candle_terms).mean() if candle_terms else zero,
    }
    event_mask = batch.event_reconstruction_mask
    components["event_reconstruction"] = (
        F.cross_entropy(
            output.event_reconstruction_logits[event_mask],
            batch.event_reconstruction_target[event_mask],
        )
        if bool(event_mask.any())
        else zero
    )
    for name, logits, target in (
        ("next_event", output.next_event_logits, targets.next_event_type),
        ("next_lifecycle", output.next_lifecycle_logits, targets.next_lifecycle),
        ("next_event_time", output.next_event_time_logits, targets.next_event_time_bucket),
        ("displacement", output.displacement_logits, targets.displacement_state),
        ("draw_consumed", output.draw_consumed_logits, targets.draw_consumed),
        ("scale_alignment", output.scale_alignment_logits, targets.scale_direction_alignment),
    ):
        labelled = target != -100
        components[name] = (
            F.cross_entropy(logits[labelled], target[labelled])
            if bool(labelled.any())
            else zero
        )
    components["contrastive"] = supervised_causal_contrastive_loss(
        output.embedding, batch
    )
    total = sum(configured[name] * value for name, value in components.items())
    return LossBreakdown(total=total, components=components)


def neutral_representation_multitask_loss(
    output: RepresentationOutput,
    batch: RepresentationBatch,
    targets: TargetBatch,
) -> LossBreakdown:
    """Compute only the preregistered neutral V1 representation objectives."""

    require_torch()
    disabled = {
        "next_event_type": targets.next_event_type,
        "next_event_time_bucket": targets.next_event_time_bucket,
        "displacement_state": targets.displacement_state,
        "draw_consumed": targets.draw_consumed,
    }
    if any(bool((values != -100).any()) for values in disabled.values()):
        raise RepresentationDataError(
            "neutral fit enabled a coverage-dependent legacy target"
        )
    return representation_multitask_loss(
        output,
        batch,
        targets,
        weights=NEUTRAL_REPRESENTATION_LOSS_WEIGHTS,
    )


def majority_class_baselines(
    training_targets: Sequence[SelfSupervisedTarget],
    validation_targets: Sequence[SelfSupervisedTarget],
) -> dict[str, float]:
    """Smoothed majority negative log likelihood for validation comparison."""

    if not training_targets or not validation_targets:
        raise RepresentationDataError("baseline evaluation needs train and validation targets")
    output: dict[str, float] = {}
    for name in sorted(ALLOWED_SELF_SUPERVISED_TARGETS):
        train = [getattr(item, name) for item in training_targets if getattr(item, name) >= 0]
        validation = [getattr(item, name) for item in validation_targets if getattr(item, name) >= 0]
        if not train or not validation:
            output[f"{name}_nll"] = float("nan")
            output[f"{name}_accuracy"] = float("nan")
            continue
        classes = sorted(set(train) | set(validation))
        counts = {label: 1 for label in classes}
        for label in train:
            counts[label] += 1
        denominator = float(sum(counts.values()))
        probabilities = {label: count / denominator for label, count in counts.items()}
        majority = max(classes, key=lambda label: (counts[label], -label))
        output[f"{name}_nll"] = float(
            -np.mean([math.log(probabilities[label]) for label in validation])
        )
        output[f"{name}_accuracy"] = float(
            np.mean([label == majority for label in validation])
        )
    return output


def representation_task_metrics(
    output: RepresentationOutput,
    targets: TargetBatch,
) -> dict[str, float]:
    """Outcome-free classification NLL/accuracy metrics for model selection."""

    require_torch()
    metrics: dict[str, float] = {}
    for name, logits, target in (
        ("next_event_type", output.next_event_logits, targets.next_event_type),
        ("next_lifecycle", output.next_lifecycle_logits, targets.next_lifecycle),
        ("next_event_time_bucket", output.next_event_time_logits, targets.next_event_time_bucket),
        ("displacement_state", output.displacement_logits, targets.displacement_state),
        ("draw_consumed", output.draw_consumed_logits, targets.draw_consumed),
        ("scale_direction_alignment", output.scale_alignment_logits, targets.scale_direction_alignment),
    ):
        labelled = target != -100
        if not bool(labelled.any()):
            metrics[f"{name}_nll"] = float("nan")
            metrics[f"{name}_accuracy"] = float("nan")
            continue
        selected_logits = logits[labelled]
        selected_target = target[labelled]
        metrics[f"{name}_nll"] = float(
            F.cross_entropy(selected_logits, selected_target).detach().cpu()
        )
        metrics[f"{name}_accuracy"] = float(
            (selected_logits.argmax(dim=-1) == selected_target)
            .float()
            .mean()
            .detach()
            .cpu()
        )
    return metrics


def zero_reconstruction_baselines(batch: RepresentationBatch) -> dict[str, float]:
    """Zero-feature baseline for normalized masked candle reconstruction."""

    require_torch()
    values: list[float] = []
    output: dict[str, float] = {}
    for timeframe in TIMEFRAMES:
        mask = batch.candle_reconstruction_masks[timeframe]
        value = (
            float(
                F.smooth_l1_loss(
                    torch.zeros_like(batch.candle_reconstruction_targets[timeframe][mask]),
                    batch.candle_reconstruction_targets[timeframe][mask],
                ).detach().cpu()
            )
            if bool(mask.any())
            else float("nan")
        )
        output[f"{timeframe}_masked_candle_zero_loss"] = value
        if math.isfinite(value):
            values.append(value)
    output["masked_candle_zero_loss"] = float(np.mean(values)) if values else float("nan")
    output["masked_event_uniform_nll"] = (
        math.log(512.0)
        if bool(batch.event_reconstruction_mask.any())
        else float("nan")
    )
    return output


def compare_reconstruction_to_baselines(
    validation_components: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Gate both registered reconstruction objectives against simple baselines."""

    candle_loss = float(validation_components.get("candle_reconstruction", float("nan")))
    event_loss = float(validation_components.get("event_reconstruction", float("nan")))
    candle_baseline = float(
        baseline_metrics.get("masked_candle_zero_loss", float("nan"))
    )
    event_baseline = float(
        baseline_metrics.get("masked_event_uniform_nll", float("nan"))
    )
    comparisons = {
        "candle_reconstruction_relative_improvement": (
            (candle_baseline - candle_loss) / candle_baseline
            if math.isfinite(candle_loss)
            and math.isfinite(candle_baseline)
            and candle_baseline > 0
            else float("nan")
        ),
        "event_reconstruction_relative_improvement": (
            (event_baseline - event_loss) / event_baseline
            if math.isfinite(event_loss)
            and math.isfinite(event_baseline)
            and event_baseline > 0
            else float("nan")
        ),
    }
    improvements = tuple(comparisons.values())
    comparisons["minimum_relative_improvement"] = (
        MIN_RECONSTRUCTION_RELATIVE_IMPROVEMENT
    )
    comparisons["all_reconstruction_tasks_measured"] = all(
        math.isfinite(value) for value in improvements
    )
    comparisons["pre_registered_criteria_met"] = bool(
        all(math.isfinite(value) for value in improvements)
        and all(
            value >= MIN_RECONSTRUCTION_RELATIVE_IMPROVEMENT
            for value in improvements
        )
    )
    return comparisons


def compare_validation_to_baseline(
    validation_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    *,
    shortcut_sensitive_tasks_masked: bool = False,
) -> dict[str, Any]:
    """Apply fixed development thresholds to outcome-free validation tasks."""

    required = tuple(
        f"{name}_nll" for name in sorted(ALLOWED_SELF_SUPERVISED_TARGETS)
    )
    result: dict[str, Any] = {}
    improvements: list[float] = []
    for key in required:
        model_value = float(validation_metrics.get(key, float("nan")))
        baseline_value = float(baseline_metrics.get(key, float("nan")))
        if not math.isfinite(model_value) or not math.isfinite(baseline_value) or baseline_value <= 0:
            continue
        improvement = (baseline_value - model_value) / baseline_value
        result[f"{key}_relative_improvement"] = improvement
        improvements.append(improvement)
    mean_improvement = (
        float(np.mean(improvements)) if improvements else float("nan")
    )
    all_tasks_measured = len(improvements) == len(required)
    result["mean_relative_improvement"] = mean_improvement
    result["all_measured_tasks_better"] = bool(improvements) and all(
        value > 0 for value in improvements
    )
    result["all_tasks_measured"] = all_tasks_measured
    result["minimum_task_relative_improvement"] = MIN_TASK_RELATIVE_IMPROVEMENT
    result["minimum_mean_relative_improvement"] = MIN_MEAN_RELATIVE_IMPROVEMENT
    result["shortcut_sensitive_tasks_masked"] = bool(
        shortcut_sensitive_tasks_masked
    )
    result["pre_registered_criteria_met"] = bool(
        all_tasks_measured
        and improvements
        and all(value >= MIN_TASK_RELATIVE_IMPROVEMENT for value in improvements)
        and math.isfinite(mean_improvement)
        and mean_improvement >= MIN_MEAN_RELATIVE_IMPROVEMENT
        and shortcut_sensitive_tasks_masked
    )
    result["label_source_shortcut_audit"] = {
        "scale_direction_alignment_direct_source": "scale_relations",
        "criteria_blocked_if_unmasked": True,
    }
    return result


def save_representation_checkpoint(
    path: str | Path,
    model: MarketRepresentationModel,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist architecture and weights only; training outcomes are excluded."""

    require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    supplied = dict(metadata or {})
    forbidden = sorted(_nested_forbidden_input_paths(supplied))
    if forbidden:
        raise RepresentationDataError(
            f"checkpoint metadata cannot contain future outcomes: {forbidden}"
        )
    checkpoint_id = representation_checkpoint_id(model)
    payload = {
        "checkpoint_id": checkpoint_id,
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": CAUSAL_CANDLE_FEATURES,
        "timeframes": TIMEFRAMES,
        "embedding_dim": EMBEDDING_DIM,
        "config": vars(model.config),
        "parameter_count": model.parameter_count(),
        "model_state": model.state_dict(),
        "metadata": supplied,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def representation_checkpoint_id(model: MarketRepresentationModel) -> str:
    """Content identity for one independently initialized ensemble member."""

    require_torch()
    digest = hashlib.sha256()
    digest.update(MODEL_VERSION.encode("utf-8"))
    digest.update(
        json.dumps(vars(model.config), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _model_from_representation_checkpoint_payload(
    payload: Mapping[str, Any],
) -> MarketRepresentationModel:
    if not isinstance(payload, Mapping):
        raise RepresentationDataError("representation checkpoint must be an object")
    if payload.get("model_version") != MODEL_VERSION:
        raise RepresentationDataError("representation checkpoint model version mismatch")
    if tuple(payload.get("feature_names", ())) != CAUSAL_CANDLE_FEATURES:
        raise RepresentationDataError("representation checkpoint feature schema mismatch")
    if int(payload.get("embedding_dim", -1)) != EMBEDDING_DIM:
        raise RepresentationDataError("representation checkpoint embedding size mismatch")
    model = MarketRepresentationModel(MarketRepresentationConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    if payload.get("checkpoint_id") != representation_checkpoint_id(model):
        raise RepresentationDataError("representation checkpoint content identity mismatch")
    return model


def load_representation_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
) -> MarketRepresentationModel:
    require_torch()
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    return _model_from_representation_checkpoint_payload(payload)


def save_neutral_representation_checkpoint(
    path: str | Path,
    model: MarketRepresentationModel,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist a checkpoint that is explicitly bound to the neutral B0 input."""

    supplied = dict(metadata or {})
    required = {
        "training_contract": NEUTRAL_TRAINING_CONTRACT,
        "direct_source_preprocessing": (
            neutral_direct_source_preprocessing_identity()
        ),
        "inference_input_protocol": NEUTRAL_INFERENCE_INPUT_PROTOCOL,
    }
    for name, value in required.items():
        if name in supplied and supplied[name] != value:
            raise RepresentationDataError(
                f"neutral checkpoint {name} conflicts with the B0 protocol"
            )
    supplied.update(required)
    save_representation_checkpoint(path, model, metadata=supplied)


def load_neutral_representation_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
) -> MarketRepresentationModel:
    """Load only a checkpoint carrying the exact current neutral B0 protocol."""

    require_torch()
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    metadata = payload.get("metadata")
    expected = {
        "training_contract": NEUTRAL_TRAINING_CONTRACT,
        "direct_source_preprocessing": (
            neutral_direct_source_preprocessing_identity()
        ),
        "inference_input_protocol": NEUTRAL_INFERENCE_INPUT_PROTOCOL,
    }
    if (
        not isinstance(metadata, Mapping)
        or any(metadata.get(name) != value for name, value in expected.items())
    ):
        raise RepresentationDataError(
            "neutral checkpoint preprocessing protocol is missing or differs"
        )
    return _model_from_representation_checkpoint_payload(payload)


def select_first_causal_stage_revisions(
    cases: Sequence[RepresentationCase],
    *,
    decision_stage: str,
) -> tuple[RepresentationCase, ...]:
    """Select the first observed stage occurrence without looking at outcomes.

    ``revision_index`` is assigned online by the sparse case recorder.  Taking
    its minimum is therefore invariant to input order and does not depend on a
    later/deeper stage, terminal result, or future path quality.
    """

    stage = str(decision_stage).strip().lower()
    if stage not in CASE_REVISION_STAGES:
        raise RepresentationDataError("decision_stage is invalid")
    grouped: dict[tuple[str, str], list[RepresentationCase]] = defaultdict(list)
    for case in cases:
        if case.revision_stage == stage:
            grouped[(case.market_epoch_id, case.entry_episode_id)].append(case)
    selected: list[RepresentationCase] = []
    for key, revisions in grouped.items():
        indices = [case.revision_index for case in revisions]
        if len(indices) != len(set(indices)):
            raise RepresentationDataError(
                f"same-stage revisions have duplicate revision_index: {key}"
            )
        selected.append(
            min(
                revisions,
                key=lambda case: (case.revision_index, case.asof, case.revision_id),
            )
        )
    return tuple(
        sorted(
            selected,
            key=lambda case: (
                case.market_epoch_id,
                case.entry_episode_id,
                case.revision_index,
            ),
        )
    )


def _require_unmasked_inference_batch(batch: RepresentationBatch) -> None:
    """Reject stochastic training masks at the retrieval/export boundary."""

    require_torch()
    for timeframe, mask in batch.candle_reconstruction_masks.items():
        if bool(mask.detach().any().cpu()):
            raise RepresentationDataError(
                f"decision-time export requires mask_probability=0.0; "
                f"{timeframe} contains a reconstruction mask"
            )
    if bool(batch.event_reconstruction_mask.detach().any().cpu()):
        raise RepresentationDataError(
            "decision-time export requires mask_probability=0.0; "
            "event tokens contain a reconstruction mask"
        )


def _market_episode_export_cases(
    batch: RepresentationBatch,
    examples: Sequence[PreparedRepresentationCase],
) -> tuple[RepresentationCase, ...]:
    from .model import Direction
    from .scene_graph import market_episode_id as canonical_market_episode_id

    require_torch()
    _require_unmasked_inference_batch(batch)
    require_neutral_preprocessed_examples(examples)
    if len(examples) != len(batch.case_ids):
        raise RepresentationDataError("MarketEpisode examples do not match batch")
    cases: list[RepresentationCase] = []
    for index, example in enumerate(examples):
        case = example.case
        if (
            case.case_id != batch.case_ids[index]
            or case.revision_id != batch.revision_ids[index]
        ):
            raise RepresentationDataError("MarketEpisode batch/example order mismatch")
        if (
            case.revision_stage != "market_episode_transition"
            or not case.entry_location_id
            or not case.entry_path_id
            or not case.transition_kinds
            or case.direction not in {-1, 1}
        ):
            raise RepresentationDataError("MarketEpisode physical fields are missing")
        direction = Direction.LONG if case.direction == 1 else Direction.SHORT
        if case.market_episode_id != canonical_market_episode_id(
            case.market_epoch_id,
            case.entry_location_id,
            case.entry_path_id,
            direction,
        ):
            raise RepresentationDataError("MarketEpisode physical identity is not canonical")
        cases.append(case)
    return tuple(cases)


def encode_market_episode_records(
    model: MarketRepresentationModel,
    batch: RepresentationBatch,
    examples: Sequence[PreparedRepresentationCase],
    *,
    split_roles: Mapping[str, str],
    material_kind: str,
) -> tuple[Mapping[str, Any], ...]:
    """Export one outcome-free unmasked embedding per explicit material kind."""

    kind = str(material_kind).strip().lower()
    if kind not in NEUTRAL_MARKET_TRANSITION_KINDS:
        raise RepresentationDataError("MarketEpisode material_kind is invalid")
    cases = _market_episode_export_cases(batch, examples)
    grains = [(case.market_epoch_id, case.market_episode_id, kind) for case in cases]
    if len(grains) != len(set(grains)):
        raise RepresentationDataError("duplicate MarketEpisode material occurrence")
    for case in cases:
        if kind not in case.transition_kinds:
            raise RepresentationDataError("batch mixes MarketEpisode material kinds")
    records = encode_decision_time_records(
        model,
        batch,
        examples,
        split_roles=split_roles,
        decision_stage="market_episode_transition",
    )
    return tuple(
        {
            "revision_id": case.revision_id,
            "revision_index": case.revision_index,
            "revision_stage": "market_episode_transition",
            "material_kind": kind,
            "transition_kinds": list(case.transition_kinds),
            "market_epoch_id": case.market_epoch_id,
            "market_episode_id": case.market_episode_id,
            "entry_location_id": case.entry_location_id,
            "entry_path_id": case.entry_path_id,
            "decision_at": case.asof.isoformat(),
            "direction": case.direction,
            "data_split": record.data_split,
            "embedding_model_version": record.embedding_model_version,
            "embedding_checkpoint_id": record.embedding_checkpoint_id,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_clock": "decision_time",
            "embedding_asof": case.asof.isoformat(),
            "feature_max_at": record.feature_max_at.isoformat(),
            "embedding_input_protocol": NEUTRAL_INFERENCE_INPUT_PROTOCOL,
            "outcome_fields_used": False,
            "decision_embedding": list(record.decision_embedding),
        }
        for case, record in zip(cases, records, strict=True)
    )


def encode_market_episode_active_head_records(
    model: MarketRepresentationModel,
    batch: RepresentationBatch,
    examples: Sequence[PreparedRepresentationCase],
    *,
    member_id: str,
) -> tuple[Mapping[str, Any], ...]:
    """Export only the two neutral active heads for retrieval disagreement."""

    cases = _market_episode_export_cases(batch, examples)
    records = encode_decision_time_head_records(
        model, batch, examples, member_id=member_id
    )
    return tuple(
        {
            "member_id": record.member_id,
            "checkpoint_id": record.checkpoint_id,
            "model_version": record.model_version,
            "revision_id": case.revision_id,
            "market_epoch_id": case.market_epoch_id,
            "market_episode_id": case.market_episode_id,
            "decision_at": case.asof.isoformat(),
            "feature_max_at": record.feature_max_at.isoformat(),
            "outcome_fields_used": False,
            "input_protocol": NEUTRAL_INFERENCE_INPUT_PROTOCOL,
            "head_predictions": {
                name: list(record.head_predictions[name])
                for name in NEUTRAL_SPARSE_ACTIVE_TARGETS
            },
        }
        for case, record in zip(cases, records, strict=True)
    )


def encode_decision_time_records(
    model: MarketRepresentationModel,
    batch: RepresentationBatch,
    examples: Sequence[PreparedRepresentationCase],
    *,
    split_roles: Mapping[str, str],
    decision_stage: str,
) -> tuple[DecisionTimeEmbeddingRecord, ...]:
    """Export one same-stage decision row per independent EntryEpisode.

    Callers must first use :func:`select_first_causal_stage_revisions`.  The
    exporter never searches later revisions for a "best" stage and therefore
    cannot select by hindsight.  Any duplicate fails closed here.
    """

    require_torch()
    _require_unmasked_inference_batch(batch)
    selected_stage = str(decision_stage).strip().lower()
    if selected_stage not in CASE_REVISION_STAGES:
        raise RepresentationDataError("embedding export decision_stage is invalid")
    if len(examples) != len(batch.case_ids):
        raise RepresentationDataError("embedding examples do not match batch size")
    episode_grains = [
        (example.case.market_epoch_id, example.case.entry_episode_id)
        for example in examples
    ]
    if len(set(episode_grains)) != len(episode_grains):
        raise RepresentationDataError(
            "embedding export accepts at most one same-stage revision per "
            "market-epoch/EntryEpisode pair"
        )
    for index, example in enumerate(examples):
        if (
            example.case.case_id != batch.case_ids[index]
            or example.case.revision_id != batch.revision_ids[index]
        ):
            raise RepresentationDataError("embedding batch/example order mismatch")
        if example.case.revision_stage != selected_stage:
            raise RepresentationDataError(
                "embedding batch mixes revision stages or differs from decision_stage"
            )
        if example.case.revision_id not in split_roles:
            raise RepresentationDataError("embedding export requires a split role")
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            encoded = model.encode(batch).detach().cpu().numpy()
    finally:
        model.train(was_training)
    output: list[DecisionTimeEmbeddingRecord] = []
    checkpoint_id = representation_checkpoint_id(model)
    for index, example in enumerate(examples):
        case = example.case
        split = _nonempty(split_roles[case.revision_id], name="embedding.split_role")
        output.append(
            DecisionTimeEmbeddingRecord(
                case_id=case.case_id,
                revision_id=case.revision_id,
                revision_stage=case.revision_stage,
                revision_index=case.revision_index,
                stage_identity=case.stage_identity,
                stage_occurrence=0,
                context_thesis_id=case.context_thesis_id,
                entry_episode_id=case.entry_episode_id,
                market_epoch_id=case.market_epoch_id,
                direction=case.direction,
                regime=case.regime,
                mechanism_label=case.mechanism_label,
                embedding_model_version=MODEL_VERSION,
                embedding_dim=EMBEDDING_DIM,
                embedding_clock="decision_time",
                embedding_asof=case.asof,
                feature_max_at=example.feature_max_at,
                data_split=split,
                split_role=split,
                outcome_fields_used=False,
                embedding_checkpoint_id=checkpoint_id,
                decision_embedding=tuple(float(value) for value in encoded[index]),
            )
        )
    return tuple(output)


def decision_time_head_probabilities(
    output: RepresentationOutput,
) -> Mapping[str, Tensor]:
    """Probability heads for ensemble disagreement; never an action opinion."""

    require_torch()
    return {
        "next_event_type": torch.softmax(output.next_event_logits, dim=-1),
        "next_lifecycle": torch.softmax(output.next_lifecycle_logits, dim=-1),
        "next_event_time_bucket": torch.softmax(output.next_event_time_logits, dim=-1),
        "displacement_state": torch.softmax(output.displacement_logits, dim=-1),
        "draw_consumed": torch.softmax(output.draw_consumed_logits, dim=-1),
        "scale_direction_alignment": torch.softmax(output.scale_alignment_logits, dim=-1),
    }


def encode_decision_time_head_records(
    model: MarketRepresentationModel,
    batch: RepresentationBatch,
    examples: Sequence[PreparedRepresentationCase],
    *,
    member_id: str,
) -> tuple[DecisionTimeHeadPredictionRecord, ...]:
    """Export ensemble heads with exact query revision and feature clocks."""

    require_torch()
    _require_unmasked_inference_batch(batch)
    if len(examples) != len(batch.case_ids):
        raise RepresentationDataError("head export examples do not match batch")
    for index, example in enumerate(examples):
        if (
            example.case.case_id != batch.case_ids[index]
            or example.case.revision_id != batch.revision_ids[index]
        ):
            raise RepresentationDataError("head export batch/example order mismatch")
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            probabilities = decision_time_head_probabilities(model(batch))
            detached = {
                name: values.detach().cpu().numpy()
                for name, values in probabilities.items()
            }
    finally:
        model.train(was_training)
    checkpoint_id = representation_checkpoint_id(model)
    return tuple(
        DecisionTimeHeadPredictionRecord(
            member_id=member_id,
            checkpoint_id=checkpoint_id,
            model_version=MODEL_VERSION,
            case_id=example.case.case_id,
            revision_id=example.case.revision_id,
            entry_episode_id=example.case.entry_episode_id,
            decision_at=example.case.asof,
            feature_max_at=example.feature_max_at,
            outcome_fields_used=False,
            head_predictions={
                name: tuple(float(value) for value in values[index])
                for name, values in detached.items()
            },
        )
        for index, example in enumerate(examples)
    )


def checkpoint_embedding_contract() -> dict[str, Any]:
    """Stable discovery contract for retrieval/OOD consumers."""

    return {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "embedding_dim": EMBEDDING_DIM,
        "normalized": True,
        "embedding_clock": "decision_time",
        "embedding_input_protocol": INFERENCE_INPUT_PROTOCOL,
        "asof_only": True,
        "outcome_fields_consumed": False,
        "required_export_fields": (
            "embedding_model_version",
            "decision_embedding",
            "embedding_asof",
            "revision_stage",
            "revision_index",
            "stage_identity",
            "stage_occurrence",
            "feature_max_at",
            "data_split",
            "split_role",
            "outcome_fields_used",
            "embedding_checkpoint_id",
            "embedding_input_protocol",
        ),
        "head_schema": dict(OUTCOME_BLIND_HEAD_WIDTHS),
        "ensemble_member_interface": "MarketRepresentationModel.encode(batch)",
    }
