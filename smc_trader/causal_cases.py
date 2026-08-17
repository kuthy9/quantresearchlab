"""Leakage-safe EntryEpisode case records for the existing replay.

The case library is a downstream recorder.  It never feeds the Brain, Decision,
Risk, Eye, or playbook runtime and it never owns a second replay loop.  Sparse
decision-time revisions and later Shadow labels are intentionally represented
by two disjoint Arrow schemas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .artifact_stream import atomic_bytes, canonical_json, sha256_file
from .model import PlaybookPhase, Timeframe, aware_timestamp, to_primitive


CAUSAL_CASE_RECORDER_SCHEMA_VERSION = 7
CAUSAL_CASE_PROTOCOL_VERSION = "entry-episode-causal-case-1.6.0"

CAUSAL_CASE_PROTOCOL: Mapping[str, Any] = {
    "protocol_version": CAUSAL_CASE_PROTOCOL_VERSION,
    "grain": "one_independent_entry_episode",
    "input_clock": "decision_time_completed_information_only",
    "admission_clock": "recorder_first_observed_snapshot_not_playbook_formed_at",
    "revision_rule": (
        "context_material_revision_or_episode_created_or_zone_registered_or_"
        "first_pullback_or_trigger_or_plan_or_terminal"
    ),
    "plan_formed_identity": (
        "first_nonempty_plan_stable_episode_custody_only;pre_risk_provisional_"
        "target_draw_route_and_live_r_re_evaluation_emit_no_revision"
    ),
    "lsr_execution_plan_identity": (
        "first_executable_episode_freezes_separate_full_plan_and_trigger_at_"
        "one_lock_clock;retained_fields_must_match_exactly;terminal_bounded_"
        "absence_does_not_clear_frozen_custody;trigger_available_kinds_are_"
        "append_only_evidence_family_diagnostics_outside_custody"
    ),
    "unchanged_evidence_heartbeat_rows": False,
    "ohlcv_storage": "canonical_source_identity_and_prefix_bounds_only",
    "prefix_index_semantics": (
        "timezone_aware_start_end_are_canonical_reload_boundaries;replay_view_"
        "1m_rows_are_run_bound_ordinals;frame_rows_are_local_observer_counters_"
        "and_must_not_index_an_external_multitimeframe_table"
    ),
    "normalization": "strictly_prior_completed_source_bars",
    "transition_coverage_continuity": (
        "recorder_requires_contiguous_replay_update_ordinals_and_one_observe_"
        "or_prime_call_per_existing_replay_update;canonical_source_ordinals_"
        "are_separate_and_may_repeat_across_synthetic_no_trade_updates"
    ),
    "graph_relation_tokens": (
        "outcome_blind_changed_edge_relation_and_endpoint_kind_role_scale_"
        "lifecycle_descriptors_from_the_same_asof_scene_graph"
    ),
    "scene_graph_case_canonicalization": (
        "causal_update_order_preserved;per_update_node_edge_resolution_id_"
        "sets_sorted_unique;aggregate_id_sets_sorted_union;relation_"
        "descriptors_added_then_revised_each_sorted_by_edge_id_and_full_"
        "payload;aggregate_descriptors_preserve_update_order;case_added_and_"
        "invalidated_event_id_summaries_sorted_unique"
    ),
    "market_epoch_identity": (
        "sha256(canonical_source_sha256,symbol,instrument_id,runtime_epoch_id);"
        "raw_runtime_epoch_remains_in_typed_context_json"
    ),
    "observable_regime_rule": (
        "asof_only_v1:balanced_market_or_favr=balance;lsr=sweep_failure;"
        "dfp=continuation;otherwise=unknown"
    ),
    "future_label_source": "existing_shadow_frozen_outcome_only",
    "shadow_outcome_selection": (
        "highest_causal_event_specificity_then_earliest_observed_at_then_"
        "candidate_id;never_resolved_at_resolution_or_outcome"
    ),
    "censored_first_event": (
        "strictly_validated_shadow_censoring_maps_to_right_censored_while_"
        "the_exact_terminal_cause_remains_in_resolution"
    ),
    "same_bar_collision": "invalidation_before_target_stop_first",
    "expired_resolution_rule": "explicit_frozen_shadow_deadline_resolutions_only",
    "shadow_resolution_families": (
        "exact_bidirectional_resolution_first_event_pairwise_fill_censor_and_"
        "collision_validation_without_substring_inference"
    ),
    "input_outcome_storage": "separate_arrow_streams_and_manifests",
    "model_feedback": "none_dataset_only",
}


def expected_causal_case_run_identity() -> Mapping[str, Any]:
    """Return the exact current run-manifest identity for case capture."""

    return {
        "recorder_schema_version": CAUSAL_CASE_RECORDER_SCHEMA_VERSION,
        "protocol": dict(CAUSAL_CASE_PROTOCOL),
        "input_stream": "causal_case_input_shards",
        "future_outcome_stream": "causal_case_outcome_shards",
        "future_visible_to_input": False,
        "output_affects_model": False,
    }


CAUSAL_CASE_INPUT_FIELD_TYPES: Mapping[str, str] = {
    "revision_id": "large_string",
    "case_id": "large_string",
    "revision_index": "int64",
    "revision_stage": "large_string",
    "stage_identity": "large_string",
    "stage_observed_at": "timestamp_ny",
    "input_fingerprint": "large_string",
    "market_epoch_id": "large_string",
    "context_thesis_id": "large_string",
    "entry_episode_id": "large_string",
    "candidate_id": "large_string",
    "playbook": "large_string",
    "direction": "large_string",
    "observable_regime": "large_string",
    "mechanism_label": "large_string",
    "asof": "timestamp_ny",
    "decision_at": "timestamp_ny",
    "admitted_at": "timestamp_ny",
    "symbol": "large_string",
    "instrument_id": "int64",
    "decision_price": "float64",
    "source_path": "large_string",
    "source_sha256": "large_string",
    "source_role": "large_string",
    "split_role": "large_string",
    "prefix_refs_json": "large_string",
    "normalization_cutoff_at": "timestamp_ny",
    "normalization_policy": "large_string",
    "added_event_ids_json": "large_string",
    "invalidated_event_ids_json": "large_string",
    "observation_transition_json": "large_string",
    "scene_graph_delta_json": "large_string",
    "context_thesis_json": "large_string",
    "entry_episode_json": "large_string",
    "brain_response_json": "large_string",
    "authority_json": "large_string",
    "scale_relations_json": "large_string",
    "draw_json": "large_string",
    "blockers_json": "large_string",
    "ambiguities_json": "large_string",
    "supporting_evidence_json": "large_string",
    "opposing_evidence_json": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "planned_entry": "float64",
    "entry_zone_lower": "float64",
    "entry_zone_upper": "float64",
    "invalidation_price": "float64",
    "invalidation_source_id": "large_string",
    "primary_target_price": "float64",
    "primary_target_id": "large_string",
    "planned_deadline_at": "timestamp_ny",
    "selected_trigger_id": "large_string",
    "selected_trigger_kind": "large_string",
    "selected_trigger_at": "timestamp_ny",
    "observed_terminal_reason": "large_string",
    "model_versions_json": "large_string",
}


CAUSAL_CASE_OUTCOME_FIELD_TYPES: Mapping[str, str] = {
    "outcome_id": "large_string",
    "case_id": "large_string",
    "market_epoch_id": "large_string",
    "context_thesis_id": "large_string",
    "entry_episode_id": "large_string",
    "resolved_at": "timestamp_ny",
    "first_event": "large_string",
    "target_first": "bool",
    "invalidation_first": "bool",
    "deadline_first": "bool",
    "same_bar_collision": "bool",
    "mfe_points": "float64",
    "mae_points": "float64",
    "mfe_R": "float64",
    "mae_R": "float64",
    "hit_0_5R": "bool",
    "hit_1R": "bool",
    "hit_2R": "bool",
    "draw_delivered": "bool",
    "time_to_draw_real_1m_bars": "int64",
    "filled": "bool",
    "expired": "bool",
    "terminal_reason": "large_string",
    "resolution": "large_string",
    "censored": "bool",
    "source_shadow_candidate_id": "large_string",
}


_INPUT_JSON_FIELDS = frozenset(
    name for name in CAUSAL_CASE_INPUT_FIELD_TYPES if name.endswith("_json")
)
_OUTCOME_ONLY_FIELDS = frozenset(
    set(CAUSAL_CASE_OUTCOME_FIELD_TYPES)
    - {
        "case_id",
        "market_epoch_id",
        "context_thesis_id",
        "entry_episode_id",
    }
)
_BOUNDARY_ANOMALIES = frozenset(
    {
        "contract_change_history_reset",
        "data_gap_history_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
_TERMINAL_LIFECYCLES = frozenset(
    {
        "closed",
        "completed",
        "invalidated",
        "censored",
        "failed",
        "left",
        "mitigated",
        "expired",
        "consumed",
    }
)
_REVISION_STAGES = frozenset(
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
_OBSERVABLE_REGIMES = frozenset(
    {"continuation", "sweep_failure", "balance", "unknown"}
)
_SHADOW_PRIORITY = {
    "playbook_executable": 50,
    "first_entry_fvg": 40,
    "first_entry_order_block": 40,
    "eligible_entry_fvg": 30,
    "eligible_entry_order_block": 30,
    "open_market_thesis_revision": 10,
}
_SHADOW_EXPIRED_RESOLUTIONS = frozenset(
    {
        "deadline_no_delivery",
        "entry_unfilled_deadline",
        "deadline_inside_completed_bar_censored",
    }
)
_SHADOW_NON_CENSORED_FIRST_EVENT_BY_RESOLUTION = {
    "target_first": "target",
    "invalidation_first": "invalidation",
    "same_bar_invalidation_priority": "invalidation",
    "source_identity_invalidated": "invalidation",
    "deadline_no_delivery": "deadline",
    "entry_unfilled_deadline": "deadline",
    "geometry_incomplete": "unresolved",
}
_SHADOW_CENSORED_RESOLUTIONS = frozenset(
    {
        "observation_boundary",
        "data_gap_boundary",
        "window_right_censored",
        "contract_boundary",
        "deadline_inside_completed_bar_censored",
        "activation_geometry_invalid",
        "invalidation_before_entry",
        "draw_consumed_before_entry",
        "entry_target_same_bar_order_unknown",
    }
)
_SCENE_SET_ID_FIELDS = (
    "added_node_ids",
    "revised_node_ids",
    "added_edge_ids",
    "revised_edge_ids",
    "resolution_event_ids",
)
_FUTURE_CLOCK_KEYS = frozenset(
    {
        "deadline",
        "deadline_at",
        "episode_deadline",
        "thesis_deadline",
        "planned_deadline_at",
    }
)


def _project_shadow_first_event(
    *,
    resolution: str,
    filled: bool,
    censored: bool,
    target_first: bool | None,
    invalidation_first: bool | None,
    same_bar_collision: bool,
) -> str:
    """Validate one frozen Shadow terminal tuple and project its case label."""

    if not isinstance(resolution, str) or not resolution:
        raise ValueError("causal case Shadow resolution is missing")
    for name, value in (
        ("target_before_invalidation", target_first),
        ("invalidation_before_target", invalidation_first),
    ):
        if value is not None and type(value) is not bool:
            raise ValueError(
                f"causal case Shadow {name} is not nullable boolean"
            )
    pairwise_first = (target_first, invalidation_first)
    if pairwise_first not in {
        (None, None),
        (True, False),
        (False, True),
    }:
        raise ValueError(
            "causal case Shadow pairwise first-event flags are inconsistent"
        )
    for name, value in (
        ("censored", censored),
        ("same_bar_collision", same_bar_collision),
        ("filled", filled),
    ):
        if type(value) is not bool:
            raise ValueError(f"causal case Shadow {name} is not boolean")
    if not filled and pairwise_first != (None, None):
        raise ValueError(
            "causal case unfilled Shadow has pairwise first-event flags"
        )
    if censored:
        if pairwise_first != (None, None) or same_bar_collision:
            raise ValueError(
                "causal case censored Shadow has resolved first-event flags"
            )
        if resolution in _SHADOW_NON_CENSORED_FIRST_EVENT_BY_RESOLUTION:
            raise ValueError(
                "causal case censored Shadow uses a non-censored resolution family"
            )
        if resolution not in _SHADOW_CENSORED_RESOLUTIONS:
            raise ValueError(
                "causal case Shadow censored resolution family is unsupported"
            )
        return "right_censored"

    first_event = _SHADOW_NON_CENSORED_FIRST_EVENT_BY_RESOLUTION.get(
        resolution
    )
    if first_event is None:
        raise ValueError(
            "causal case Shadow non-censored resolution family is unsupported"
        )
    expected_pair = (
        (True, False)
        if first_event == "target"
        else (False, True)
        if first_event == "invalidation" and filled
        else (None, None)
    )
    if pairwise_first != expected_pair:
        raise ValueError(
            "causal case Shadow resolution family disagrees with first-event flags"
        )
    expected_collision = resolution == "same_bar_invalidation_priority"
    if same_bar_collision is not expected_collision:
        raise ValueError(
            "causal case Shadow resolution family disagrees with collision flag"
        )
    if first_event == "target" and not filled:
        raise ValueError(
            "causal case Shadow target resolution is not a filled path"
        )
    if resolution == "deadline_no_delivery" and not filled:
        raise ValueError(
            "causal case Shadow filled deadline resolution is unfilled"
        )
    if resolution == "entry_unfilled_deadline" and filled:
        raise ValueError(
            "causal case Shadow unfilled deadline resolution is filled"
        )
    if resolution == "geometry_incomplete" and filled:
        raise ValueError(
            "causal case Shadow incomplete geometry is unexpectedly filled"
        )
    return first_event


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("causal case contains a non-finite number")
    return number


def _identity(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _canonical_identity_set(
    values: Iterable[Any],
    *,
    name: str,
) -> tuple[str, ...]:
    """Return the deterministic representation of one set-valued ID field."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"causal case {name} must be an ID collection")
    try:
        identities = tuple(values)
    except TypeError as exc:
        raise ValueError(
            f"causal case {name} must be an ID collection"
        ) from exc
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError(f"causal case {name} contains an invalid identity")
    return tuple(sorted(set(identities)))


def _validate_canonical_identity_list(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"causal case {name} must be an ID array")
    expected = _canonical_identity_set(value, name=name)
    if tuple(value) != expected:
        raise ValueError(f"causal case {name} is not sorted unique")
    return expected


def _ordered_scene_changed_edge_ids(
    added_edge_ids: Sequence[str],
    revised_edge_ids: Sequence[str],
) -> tuple[str, ...]:
    overlap = set(added_edge_ids).intersection(revised_edge_ids)
    if overlap:
        raise ValueError(
            "causal case Scene edge is both added and revised in one update"
        )
    return (*added_edge_ids, *revised_edge_ids)


def _scene_relation_descriptor_sort_key(
    descriptor: Mapping[str, Any],
) -> tuple[str, str]:
    """Sort by identity and then the complete canonical descriptor payload."""

    return str(descriptor.get("edge_id", "")), _json(descriptor)


def _primary_identity(value: Any) -> str | None:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        value = value[1]
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
        "entity_id",
    ):
        identity = _identity(getattr(value, name, None))
        if identity is not None:
            return identity
    return None


def _lifecycle(value: Any) -> str | None:
    raw = getattr(value, "lifecycle", None)
    return None if raw is None else str(_enum_value(raw))


def _transition_collections(observation: Any) -> tuple[tuple[str, Sequence[Any]], ...]:
    names = (
        "liquidity_inventory_transitions_this_update",
        "liquidity_pool_transitions_this_update",
        "group3_fvg_transitions_this_update",
        "group3_order_block_transitions_this_update",
        "group4_range_transitions_this_update",
        "group4_manipulation_transitions_this_update",
        "group5_entry_location_transitions_this_update",
        "group5_reacceptance_transitions_this_update",
        "group5_micro_bos_transitions_this_update",
        "group5_path_transitions_this_update",
        "group5_step_transitions_this_update",
    )
    return tuple((name, tuple(getattr(observation, name, ()))) for name in names)


def _observation_transition_payload(observation: Any) -> Mapping[str, Any]:
    return {
        "typed_transition_delta_available": bool(
            getattr(observation, "typed_transition_delta_available", False)
        ),
        "collections": {
            name: to_primitive(values) for name, values in _transition_collections(observation)
        },
    }


def _scene_node_relation_role(node: Any) -> str:
    liquidity_role = _enum_value(getattr(node, "liquidity_role", None))
    if isinstance(liquidity_role, str) and liquidity_role:
        return liquidity_role
    for name, value in getattr(node, "semantic_attributes", ()):
        if str(name) in {"role", "authority_role", "location_role"} and value:
            return str(value)
    return str(getattr(node, "kind", "unknown"))


def _scene_endpoint_descriptor(node: Any) -> Mapping[str, Any]:
    return {
        "node_id": str(getattr(node, "node_id")),
        "kind": str(getattr(node, "kind", "unknown")),
        "role": _scene_node_relation_role(node),
        "timeframe": str(_enum_value(getattr(node, "timeframe", "unknown"))),
        "structural_scale": str(
            _enum_value(getattr(node, "structural_scale", "unknown"))
        ),
        "lifecycle": str(_enum_value(getattr(node, "lifecycle", "unknown"))),
    }


def _scene_relation_descriptors(
    observation: Any,
    scene_graph: Any | None,
    *,
    added_edge_ids: Sequence[str],
    revised_edge_ids: Sequence[str],
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    """Freeze typed changed-edge topology from the same observable graph.

    The absolute identities remain for audit/custody joins.  Representation
    code can tokenize the relation and endpoint descriptors without treating
    those identities as semantic features.
    """

    edge_ids = _ordered_scene_changed_edge_ids(
        added_edge_ids,
        revised_edge_ids,
    )
    if not edge_ids:
        return (), True
    if scene_graph is None:
        return (), False
    graph_asof = getattr(scene_graph, "last_asof", None)
    if graph_asof is not None and _clock(
        graph_asof,
        name="causal_case.scene_graph.last_asof",
    ) != _clock(observation.asof, name="causal_case.scene_graph.observation_asof"):
        raise ValueError("causal case Scene Graph is not from the decision asof")
    graph_revision = getattr(scene_graph, "revision_id", None)
    observation_revision = getattr(observation, "scene_revision_id", None)
    if (
        graph_revision is not None
        and observation_revision is not None
        and str(graph_revision) != str(observation_revision)
    ):
        raise ValueError("causal case Scene Graph revision differs from observation")
    permanent_edges = getattr(scene_graph, "_edges", {})
    current_edges = getattr(scene_graph, "_current_path_block_edges", {})
    nodes = getattr(scene_graph, "_nodes", {})
    if not all(isinstance(value, Mapping) for value in (permanent_edges, current_edges, nodes)):
        raise TypeError("causal case Scene Graph does not expose typed current indexes")
    descriptors: list[Mapping[str, Any]] = []
    for edge_id in edge_ids:
        edge = permanent_edges.get(edge_id, current_edges.get(edge_id))
        if edge is None:
            raise ValueError("causal case changed Scene edge is missing from current graph")
        source = nodes.get(getattr(edge, "source_node_id", None))
        target = nodes.get(getattr(edge, "target_node_id", None))
        if source is None or target is None:
            raise ValueError("causal case changed Scene edge has a missing endpoint")
        observed_at = _clock(
            getattr(edge, "observed_at", None),
            name="causal_case.scene_edge.observed_at",
        )
        if observed_at > _clock(observation.asof, name="causal_case.observation.asof"):
            raise ValueError("causal case changed Scene edge is future-dated")
        descriptors.append(
            {
                "edge_id": str(edge_id),
                "relation": str(_enum_value(getattr(edge, "relation", "unknown"))),
                "lifecycle": str(_enum_value(getattr(edge, "lifecycle", "unknown"))),
                "observed_at": observed_at,
                "source": _scene_endpoint_descriptor(source),
                "target": _scene_endpoint_descriptor(target),
            }
        )
    added = tuple(
        sorted(
            descriptors[: len(added_edge_ids)],
            key=_scene_relation_descriptor_sort_key,
        )
    )
    revised = tuple(
        sorted(
            descriptors[len(added_edge_ids) :],
            key=_scene_relation_descriptor_sort_key,
        )
    )
    return (*added, *revised), True


def _scene_delta_payload(
    observation: Any,
    *,
    scene_graph: Any | None,
) -> Mapping[str, Any]:
    scene_id_sets = {
        name: _canonical_identity_set(
            getattr(observation, f"scene_{name}", ()),
            name=f"Scene update {name}",
        )
        for name in _SCENE_SET_ID_FIELDS
    }
    relation_descriptors, relation_descriptors_complete = (
        _scene_relation_descriptors(
            observation,
            scene_graph,
            added_edge_ids=scene_id_sets["added_edge_ids"],
            revised_edge_ids=scene_id_sets["revised_edge_ids"],
        )
    )
    return {
        "asof": observation.asof,
        "revision_id": getattr(observation, "scene_revision_id", None),
        **scene_id_sets,
        "relation_descriptors": relation_descriptors,
        "relation_descriptors_complete": relation_descriptors_complete,
    }


def _event_delta_ids(observation: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    added = list(getattr(observation, "scene_added_node_ids", ()))
    added.extend(getattr(observation, "scene_added_edge_ids", ()))
    invalidated = list(getattr(observation, "scene_resolution_event_ids", ()))
    for _name, values in _transition_collections(observation):
        for value in values:
            identity = _primary_identity(value)
            if identity is None:
                continue
            lifecycle = _lifecycle(value)
            if lifecycle in _TERMINAL_LIFECYCLES:
                invalidated.append(identity)
            else:
                added.append(identity)
    context = getattr(getattr(observation, "belief", None), "global_context", None)
    if context is not None:
        invalidated.extend(getattr(context, "invalidated_source_ids", ()))
    return (
        _canonical_identity_set(added, name="added event ID summary"),
        _canonical_identity_set(
            invalidated,
            name="invalidated event ID summary",
        ),
    )


def _case_transition_update(
    snapshot: Any,
    *,
    replay_update_ordinal: int,
    scene_graph: Any | None,
) -> Mapping[str, Any]:
    """Capture one outcome-blind replay update for sparse interval coverage."""

    observation = snapshot.observation
    added, invalidated = _event_delta_ids(observation)
    global_context = getattr(snapshot.belief, "global_context", None)
    if global_context is not None:
        invalidated = _canonical_identity_set(
            (
                *invalidated,
                *getattr(global_context, "invalidated_source_ids", ()),
            ),
            name="invalidated event ID summary",
        )
    return {
        "asof": observation.asof,
        "replay_update_ordinal": int(replay_update_ordinal),
        "observation_transition": _observation_transition_payload(observation),
        "scene_graph_delta": _scene_delta_payload(
            observation,
            scene_graph=scene_graph,
        ),
        "added_event_ids": added,
        "invalidated_event_ids": invalidated,
    }


def _transition_update_is_eventful(update: Mapping[str, Any]) -> bool:
    observation = update["observation_transition"]
    scene = update["scene_graph_delta"]
    return bool(
        any(tuple(values) for values in observation.get("collections", {}).values())
        or any(
            tuple(scene.get(name, ()))
            for name in (
                "added_node_ids",
                "revised_node_ids",
                "added_edge_ids",
                "revised_edge_ids",
                "resolution_event_ids",
            )
        )
        or tuple(update.get("added_event_ids", ()))
        or tuple(update.get("invalidated_event_ids", ()))
    )


def _compact_candidate(identity: str, candidate: Any) -> Mapping[str, Any]:
    trigger = getattr(candidate, "selected_trigger", None)
    plan = getattr(candidate, "plan", None)
    return {
        "candidate_id": identity,
        "key": getattr(candidate, "key", None),
        "record_kind": getattr(candidate, "record_kind", None),
        "playbook": _enum_value(getattr(candidate, "playbook", None)),
        "direction": _enum_value(getattr(candidate, "direction", None)),
        "phase": _enum_value(getattr(candidate, "phase", None)),
        "context_thesis_id": getattr(candidate, "context_thesis_id", None),
        "episode_id": getattr(candidate, "episode_id", None),
        "setup_id": getattr(candidate, "setup_context_id", None),
        "entry_location_id": getattr(candidate, "entry_location_id", None),
        "entry_path_id": getattr(candidate, "entry_path_id", None),
        "probability": _finite(getattr(candidate, "probability", None)),
        "raw_probability": _finite(getattr(candidate, "raw_probability", None)),
        "phase_started_at": getattr(candidate, "phase_started_at", None),
        "evidence_revision_id": getattr(candidate, "evidence_revision_id", None),
        "hard_gate_results": dict(getattr(candidate, "hard_gate_results", {})),
        "scores": {
            name: _finite(getattr(candidate, name, None))
            for name in (
                "thesis_strength",
                "sequence_progress",
                "location_quality",
                "entry_readiness",
                "delivery_quality",
                "uncertainty",
            )
        },
        "selected_trigger": None if trigger is None else to_primitive(trigger),
        "plan": None if plan is None else to_primitive(plan),
    }


def _brain_response_payload(snapshot: Any) -> Mapping[str, Any]:
    belief = snapshot.belief
    candidates: dict[str, Any] = {}
    for collection_name in (
        "thesis_candidates",
        "retained_episode_candidates",
        "position_management_candidates",
        "hypotheses",
    ):
        collection = getattr(belief, collection_name, {})
        if not isinstance(collection, Mapping):
            continue
        for identity, candidate in sorted(collection.items(), key=lambda item: str(item[0])):
            key = str(identity)
            candidates.setdefault(key, _compact_candidate(key, candidate))
    decision = snapshot.decision
    risk = snapshot.risk
    return {
        "candidate_count": len(candidates),
        "candidates": tuple(candidates[key] for key in sorted(candidates)),
        "dominant_hypothesis_id": getattr(belief, "dominant_hypothesis_id", None),
        "competing_hypothesis_ids": tuple(
            getattr(belief, "competing_hypothesis_ids", ())
        ),
        "selected_action": _enum_value(getattr(decision, "selected_action", None)),
        "best_hypothesis_key": getattr(decision, "best_hypothesis_key", None),
        "advantage": _finite(getattr(decision, "advantage", None)),
        "decision_reasons": tuple(getattr(decision, "reasons", ())),
        "decision_plan": (
            None
            if getattr(decision, "plan", None) is None
            else to_primitive(decision.plan)
        ),
        "risk_action": _enum_value(getattr(risk, "final_action", None)),
        "risk_passed": bool(getattr(risk, "passed", False)),
        "risk_vetoes": tuple(_enum_value(item) for item in getattr(risk, "vetoes", ())),
        "risk_reasons": tuple(getattr(risk, "reasons", ())),
    }


def _context_signature(context: Any) -> str:
    payload = {
        "market_epoch_id": context.market_epoch_id,
        "direction": _enum_value(context.direction),
        "authority_ids": tuple(sorted(set(context.authority_ids))),
        "context_draw": to_primitive(context.context_draw),
        "structural_invalidation": to_primitive(context.structural_invalidation),
        "supporting_event_ids": tuple(sorted(set(context.supporting_event_ids))),
        "opposing_event_ids": tuple(sorted(set(context.opposing_event_ids))),
        "lifecycle": context.lifecycle,
        "thesis_deadline": context.thesis_deadline,
        "terminal_at": context.terminal_at,
        "terminal_reason": context.terminal_reason,
    }
    return _hash_payload(payload)


def _stable_plan_custody_payload(plan: Any) -> Mapping[str, Any]:
    return {
        "playbook": _enum_value(plan.playbook),
        "direction": _enum_value(plan.direction),
        "setup_id": plan.setup_id,
        "entry_location_id": plan.entry_location_id,
        "entry_path_id": plan.entry_path_id,
        "planned_entry": _finite(plan.planned_entry),
        "entry_zone_lower": _finite(plan.entry_zone_lower),
        "entry_zone_upper": _finite(plan.entry_zone_upper),
        "invalidation": (
            plan.invalidation.source_level_id,
            _finite(plan.invalidation.price),
            plan.invalidation.side,
            plan.invalidation.observed_at,
            plan.invalidation.rationale,
        ),
        "risk_points": _finite(plan.risk_points),
        "deadline": plan.deadline,
        "lsr_context": to_primitive(getattr(plan, "lsr_context", None)),
        "range_auction": to_primitive(getattr(plan, "range_auction", None)),
    }


def _plan_custody_identity(plan: Any) -> str:
    """Hash stable Episode custody, not a provisional pre-Risk plan view.

    A typed plan can exist before Risk approval while its visible target,
    draw, route and derived R diagnostics are still being re-evaluated.  Those
    fields remain captured in the first ``plan_formed`` row, but they cannot
    create another milestone or change its identity.  Entry geometry, the
    frozen invalidation/deadline and playbook-specific parent provenance are
    stable Episode custody and therefore remain fail-closed here.
    """

    return _hash_payload(_stable_plan_custody_payload(plan))


def _lsr_execution_plan_identity(plan: Any) -> str:
    """Hash exact LSR owner plan, excluding only its live path-R view."""

    plan_payload = dict(to_primitive(plan))
    plan_payload.pop("remaining_path_R", None)
    return _hash_payload(plan_payload)


def _lsr_execution_trigger_identity(selected_trigger: Any) -> str:
    """Hash frozen trigger custody, excluding evidence-family diagnostics."""

    payload = dict(to_primitive(selected_trigger))
    payload.pop("available_trigger_kinds", None)
    return _hash_payload(payload)


def _observable_regime(snapshot: Any, episode: Any) -> str:
    global_context = getattr(snapshot.belief, "global_context", None)
    market_mode = _enum_value(getattr(global_context, "market_mode", None))
    playbook = _enum_value(getattr(episode, "playbook", None))
    if market_mode == "balanced" or playbook == "failed_auction_value_return":
        return "balance"
    if playbook == "liquidity_sweep_reversal":
        return "sweep_failure"
    if playbook == "displacement_first_pullback":
        return "continuation"
    return "unknown"


def _case_id(epoch_id: str, context_id: str, episode_id: str) -> str:
    digest = hashlib.sha256(
        f"{epoch_id}|{context_id}|{episode_id}".encode("utf-8")
    ).hexdigest()
    return f"entry-case:{digest[:32]}"


def _stage_identity(stage: str, identity: str) -> str:
    return f"{stage}:{identity}"


def _clock(value: Any, *, name: str) -> pd.Timestamp:
    return aware_timestamp(value, name=name)


def _case_feature_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    excluded = {
        "revision_id",
        "case_id",
        "revision_index",
        "stage_identity",
        "input_fingerprint",
    }
    return {name: row[name] for name in CAUSAL_CASE_INPUT_FIELD_TYPES if name not in excluded}


def case_input_fingerprint(row: Mapping[str, Any]) -> str:
    """Return the future-blind fingerprint registered in one input row."""

    return _hash_payload(to_primitive(_case_feature_payload(row)))


@dataclass(frozen=True)
class CausalCaseInputRecord:
    revision_id: str
    case_id: str
    revision_index: int
    revision_stage: str
    stage_identity: str
    stage_observed_at: pd.Timestamp
    input_fingerprint: str
    market_epoch_id: str
    context_thesis_id: str
    entry_episode_id: str
    candidate_id: str
    playbook: str
    direction: str
    observable_regime: str
    mechanism_label: str
    asof: pd.Timestamp
    decision_at: pd.Timestamp
    admitted_at: pd.Timestamp
    symbol: str
    instrument_id: int
    decision_price: float
    source_path: str
    source_sha256: str
    source_role: str
    split_role: str
    prefix_refs_json: str
    normalization_cutoff_at: pd.Timestamp
    normalization_policy: str
    added_event_ids_json: str
    invalidated_event_ids_json: str
    observation_transition_json: str
    scene_graph_delta_json: str
    context_thesis_json: str
    entry_episode_json: str
    brain_response_json: str
    authority_json: str
    scale_relations_json: str
    draw_json: str
    blockers_json: str
    ambiguities_json: str
    supporting_evidence_json: str
    opposing_evidence_json: str
    entry_location_id: str | None
    entry_path_id: str | None
    planned_entry: float | None
    entry_zone_lower: float | None
    entry_zone_upper: float | None
    invalidation_price: float | None
    invalidation_source_id: str | None
    primary_target_price: float | None
    primary_target_id: str | None
    planned_deadline_at: pd.Timestamp
    selected_trigger_id: str | None
    selected_trigger_kind: str | None
    selected_trigger_at: pd.Timestamp | None
    observed_terminal_reason: str | None
    model_versions_json: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalCaseOutcomeRecord:
    outcome_id: str
    case_id: str
    market_epoch_id: str
    context_thesis_id: str
    entry_episode_id: str
    resolved_at: pd.Timestamp
    first_event: str
    target_first: bool | None
    invalidation_first: bool | None
    deadline_first: bool
    same_bar_collision: bool
    mfe_points: float | None
    mae_points: float | None
    mfe_R: float | None
    mae_R: float | None
    hit_0_5R: bool | None
    hit_1R: bool | None
    hit_2R: bool | None
    draw_delivered: bool | None
    time_to_draw_real_1m_bars: int | None
    filled: bool
    expired: bool
    terminal_reason: str | None
    resolution: str
    censored: bool
    source_shadow_candidate_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _CaseState:
    case_id: str
    market_epoch_id: str
    context_thesis_id: str
    entry_episode_id: str
    candidate_id: str
    playbook: str
    admitted_at: pd.Timestamp
    admitted_replay_update_ordinal: int
    next_revision_index: int = 0
    stage_identities: set[str] = field(default_factory=set)
    context_signature: str | None = None
    entry_location_id: str | None = None
    entry_path_id: str | None = None
    source_displacement_id: str | None = None
    source_zone_id: str | None = None
    frozen_execution_plan_identity: str | None = None
    frozen_execution_trigger_identity: str | None = None
    frozen_execution_lock_at: pd.Timestamp | None = None
    observed_execution_trigger_kinds: tuple[str, ...] = ()
    last_context: Any | None = None
    last_episode: Any | None = None
    last_revision_at: pd.Timestamp | None = None
    last_revision_replay_update_ordinal: int | None = None
    transition_updates: list[Mapping[str, Any]] = field(default_factory=list)
    coverage_observed_update_count: int = 0
    coverage_last_observed_at: pd.Timestamp | None = None
    coverage_last_replay_update_ordinal: int | None = None
    coverage_all_typed_available: bool = True
    coverage_gap_free: bool = True
    terminal_at: pd.Timestamp | None = None
    terminal_reason: str | None = None
    shadow_payload: dict[str, Any] | None = None
    shadow_priority: int = -1


class CausalCaseRecorder:
    """Sparse decision-time case capture joined only to later Shadow rows."""

    def __init__(
        self,
        *,
        source_path: str | Path,
        source_sha256: str,
        source_role: str,
        split_role: str,
        capture_start: pd.Timestamp,
        model_versions: Mapping[str, Any],
    ) -> None:
        resolved = Path(source_path).resolve()
        if len(source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_sha256.lower()
        ):
            raise ValueError("causal case source SHA-256 is invalid")
        if not source_role or not split_role:
            raise ValueError("causal case source and split roles are required")
        self._recorder_schema_version = CAUSAL_CASE_RECORDER_SCHEMA_VERSION
        self._source_path = str(resolved)
        self._source_sha256 = source_sha256.lower()
        self._source_role = str(source_role)
        self._split_role = str(split_role)
        self._capture_start = _clock(capture_start, name="causal_case.capture_start")
        self._model_versions = to_primitive(dict(model_versions))
        self._input_rows: list[CausalCaseInputRecord] = []
        self._outcome_rows: list[CausalCaseOutcomeRecord] = []
        self._cases: dict[str, _CaseState] = {}
        self._episode_to_case: dict[str, str] = {}
        self._episode_epoch: dict[str, str] = {}
        self._ignored_episode_ids: set[str] = set()
        self._rejected_episode_ids: set[str] = set()
        self._pending_shadow_by_episode: dict[str, list[dict[str, Any]]] = {}
        self._input_fingerprints: set[str] = set()
        self._revision_ids: set[str] = set()
        self._outcome_count = 0
        self._current_epoch_id: str | None = None
        self._epoch_source_row_start: int | None = None
        self._epoch_started_at: pd.Timestamp | None = None
        self._last_observation_asof: pd.Timestamp | None = None
        self._last_source_row_ordinal: int | None = None
        self._last_source_bar_was_synthetic: bool | None = None
        self._last_replay_update_ordinal: int | None = None
        self._last_epoch_snapshot: Any | None = None
        self._last_epoch_source_bar: Any | None = None
        self._last_epoch_source_row_ordinal: int | None = None
        self._last_epoch_replay_update_ordinal: int | None = None
        self._closed = False
        self._skipped_quality: dict[str, int] = {}

    @property
    def recorder_schema_version(self) -> int:
        return self._recorder_schema_version

    @property
    def summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": self._recorder_schema_version,
            "protocol_version": CAUSAL_CASE_PROTOCOL_VERSION,
            "case_count": len(self._cases),
            "input_revision_count": len(self._revision_ids),
            "outcome_count": self._outcome_count,
            "ignored_left_censored_episode_count": len(self._ignored_episode_ids),
            "rejected_episode_count": len(self._rejected_episode_ids),
            "skipped_quality": dict(sorted(self._skipped_quality.items())),
            "future_visible_to_input": False,
            "output_affects_model": False,
        }

    def drain_input_rows(self) -> tuple[CausalCaseInputRecord, ...]:
        rows = tuple(self._input_rows)
        self._input_rows.clear()
        return rows

    def drain_outcome_rows(self) -> tuple[CausalCaseOutcomeRecord, ...]:
        rows = tuple(self._outcome_rows)
        self._outcome_rows.clear()
        return rows

    def _skip(self, episode_id: str, reason: str) -> None:
        self._rejected_episode_ids.add(episode_id)
        self._skipped_quality[reason] = int(self._skipped_quality.get(reason, 0)) + 1
        pending = self._pending_shadow_by_episode.pop(episode_id, ())
        if pending:
            self._count_quality(
                "shadow_pending_discarded_on_episode_rejection",
                len(pending),
            )

    def _count_quality(self, reason: str, count: int = 1) -> None:
        if count < 1:
            return
        self._skipped_quality[reason] = int(self._skipped_quality.get(reason, 0)) + int(
            count
        )

    @staticmethod
    def _snapshot_runtime_epoch(snapshot: Any) -> str:
        context = getattr(snapshot.belief, "global_context", None)
        epoch_id = _identity(getattr(context, "market_epoch_id", None))
        if epoch_id is not None:
            return epoch_id
        epochs = {
            _identity(getattr(item, "market_epoch_id", None))
            for item in getattr(snapshot.belief, "context_theses", {}).values()
        }
        epochs.discard(None)
        if len(epochs) != 1:
            raise ValueError("causal case snapshot lacks one market epoch identity")
        return str(next(iter(epochs)))

    def _snapshot_epoch(self, snapshot: Any, runtime_epoch_id: str) -> str:
        payload = (
            f"{self._source_sha256}|{snapshot.observation.symbol}|"
            f"{int(snapshot.observation.instrument_id)}|{runtime_epoch_id}"
        )
        return "market-epoch:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _validate_snapshot(self, snapshot: Any, source_bar: Any) -> pd.Timestamp:
        if self._closed:
            raise RuntimeError("causal case recorder is already closed")
        asof = _clock(snapshot.observation.asof, name="causal_case.observation.asof")
        if _clock(source_bar.end, name="causal_case.source_bar.end") != asof:
            raise ValueError("causal case source bar is not the snapshot bar")
        if (source_bar.symbol, int(source_bar.instrument_id)) != (
            snapshot.observation.symbol,
            int(snapshot.observation.instrument_id),
        ):
            raise ValueError("causal case source contract differs from snapshot")
        if not math.isclose(
            float(source_bar.close),
            float(snapshot.observation.price),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("causal case source close differs from decision price")
        if self._last_observation_asof is not None and asof <= self._last_observation_asof:
            raise ValueError("causal case observations must be strictly increasing")
        self._last_observation_asof = asof
        return asof

    def _validate_replay_coordinates(
        self,
        *,
        source_row_ordinal: int,
        replay_update_ordinal: int,
        source_bar: Any,
    ) -> None:
        if isinstance(source_row_ordinal, bool) or not isinstance(
            source_row_ordinal,
            int,
        ):
            raise TypeError("causal case source row ordinal must be an integer")
        if source_row_ordinal < 0:
            raise ValueError("causal case source row ordinal cannot be negative")
        if isinstance(replay_update_ordinal, bool) or not isinstance(
            replay_update_ordinal,
            int,
        ):
            raise TypeError("causal case replay update ordinal must be an integer")
        if replay_update_ordinal < 0:
            raise ValueError("causal case replay update ordinal cannot be negative")
        if (
            self._last_replay_update_ordinal is not None
            and replay_update_ordinal != self._last_replay_update_ordinal + 1
        ):
            raise ValueError(
                "causal case recorder did not receive a contiguous replay update"
            )
        if self._last_source_row_ordinal is not None:
            expected_source_ordinal = self._last_source_row_ordinal + int(
                not bool(self._last_source_bar_was_synthetic)
            )
            if source_row_ordinal != expected_source_ordinal:
                raise ValueError(
                    "causal case canonical source ordinal disagrees with replay bars"
                )
        self._last_source_row_ordinal = source_row_ordinal
        self._last_source_bar_was_synthetic = bool(source_bar.synthetic_no_trade)
        self._last_replay_update_ordinal = replay_update_ordinal

    def _advance_epoch(
        self,
        epoch_id: str,
        *,
        asof: pd.Timestamp,
        source_row_ordinal: int,
        replay_update_ordinal: int,
        source_bar_start: pd.Timestamp,
        reason: str,
    ) -> None:
        if self._current_epoch_id == epoch_id:
            return
        if self._current_epoch_id is not None:
            self._emit_epoch_terminal_revisions(
                asof=asof,
                replay_update_ordinal=replay_update_ordinal,
                reason=reason,
            )
            pending_count = sum(
                len(values) for values in self._pending_shadow_by_episode.values()
            )
            if pending_count:
                self._count_quality(
                    "shadow_pending_discarded_on_epoch_reset",
                    pending_count,
                )
                self._pending_shadow_by_episode.clear()
        self._current_epoch_id = epoch_id
        self._epoch_source_row_start = int(source_row_ordinal)
        self._epoch_started_at = _clock(
            source_bar_start,
            name="causal_case.epoch_started_at",
        )

    def _emit_epoch_terminal_revisions(
        self,
        *,
        asof: pd.Timestamp,
        replay_update_ordinal: int,
        reason: str,
    ) -> None:
        """Close the old epoch at the observable boundary without new-epoch bars."""

        snapshot = self._last_epoch_snapshot
        source_bar = self._last_epoch_source_bar
        source_row_ordinal = self._last_epoch_source_row_ordinal
        open_cases = tuple(
            case
            for case in self._cases.values()
            if case.market_epoch_id == self._current_epoch_id
            and case.terminal_at is None
        )
        if not open_cases:
            return
        if snapshot is None or source_bar is None or source_row_ordinal is None:
            raise RuntimeError("causal case epoch reset lacks its last bounded prefix")
        for case in sorted(open_cases, key=lambda item: item.case_id):
            if case.last_context is None or case.last_episode is None:
                raise RuntimeError("causal case epoch reset lacks prior typed custody")
            stage_identity = _stage_identity(
                "terminal",
                f"{asof.isoformat()}|{reason}",
            )
            if stage_identity in case.stage_identities:
                continue
            if any(
                identity.startswith("terminal:")
                for identity in case.stage_identities
            ):
                raise ValueError("causal case terminal identity changed at epoch reset")
            self._emit_input(
                case,
                snapshot,
                context=case.last_context,
                episode=case.last_episode,
                source_bar=source_bar,
                source_row_ordinal=source_row_ordinal,
                stage="terminal",
                stage_identity=stage_identity,
                stage_observed_at=asof,
                decision_asof=asof,
                boundary_terminal_reason=reason,
                replay_update_ordinal=replay_update_ordinal,
            )
            case.terminal_at = asof
            case.terminal_reason = reason

    def _remember_epoch_snapshot(
        self,
        snapshot: Any,
        *,
        source_bar: Any,
        source_row_ordinal: int,
        replay_update_ordinal: int,
    ) -> None:
        self._last_epoch_snapshot = snapshot
        self._last_epoch_source_bar = source_bar
        self._last_epoch_source_row_ordinal = int(source_row_ordinal)
        self._last_epoch_replay_update_ordinal = int(replay_update_ordinal)

    @staticmethod
    def _accumulate_transition_update(
        case: _CaseState,
        update: Mapping[str, Any],
    ) -> None:
        asof = _clock(update["asof"], name="causal_case.transition_update.asof")
        replay_update_ordinal = int(update["replay_update_ordinal"])
        if (
            case.coverage_last_observed_at is not None
            and asof <= case.coverage_last_observed_at
        ):
            raise ValueError("causal case transition update was duplicated")
        if case.last_revision_at is not None and asof <= case.last_revision_at:
            raise ValueError("causal case transition update did not follow its revision")
        prior_replay_update_ordinal = (
            case.last_revision_replay_update_ordinal
            if case.coverage_last_replay_update_ordinal is None
            else case.coverage_last_replay_update_ordinal
        )
        if prior_replay_update_ordinal is not None and (
            replay_update_ordinal != prior_replay_update_ordinal + 1
        ):
            case.coverage_gap_free = False
        case.coverage_observed_update_count += 1
        case.coverage_last_observed_at = asof
        case.coverage_last_replay_update_ordinal = replay_update_ordinal
        case.coverage_all_typed_available = bool(
            case.coverage_all_typed_available
            and update["observation_transition"].get(
                "typed_transition_delta_available",
                False,
            )
        )
        if _transition_update_is_eventful(update):
            case.transition_updates.append(update)

    def prime(
        self,
        snapshot: Any,
        *,
        source_bar: Any,
        source_row_ordinal: int,
        replay_update_ordinal: int | None = None,
        scene_graph: Any | None = None,
    ) -> None:
        """Mark warmup episodes as left-censored without emitting a case."""

        replay_ordinal = (
            0
            if replay_update_ordinal is None and self._last_replay_update_ordinal is None
            else (
                int(self._last_replay_update_ordinal) + 1
                if replay_update_ordinal is None
                else int(replay_update_ordinal)
            )
        )
        self._validate_replay_coordinates(
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_ordinal,
            source_bar=source_bar,
        )
        asof = self._validate_snapshot(snapshot, source_bar)
        runtime_epoch_id = self._snapshot_runtime_epoch(snapshot)
        epoch_id = self._snapshot_epoch(snapshot, runtime_epoch_id)
        self._advance_epoch(
            epoch_id,
            asof=asof,
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_ordinal,
            source_bar_start=source_bar.start,
            reason="warmup_epoch_reset",
        )
        for episode in getattr(snapshot.belief, "entry_episodes", {}).values():
            episode_id = _identity(getattr(episode, "episode_id", None))
            if episode_id is not None:
                self._ignored_episode_ids.add(episode_id)
                pending = self._pending_shadow_by_episode.pop(episode_id, ())
                if pending:
                    self._count_quality(
                        "shadow_pending_discarded_on_warmup_prime",
                        len(pending),
                    )
        self._remember_epoch_snapshot(
            snapshot,
            source_bar=source_bar,
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_ordinal,
        )

    def _prefix_refs(
        self,
        snapshot: Any,
        *,
        asof: pd.Timestamp,
        source_row_ordinal: int,
    ) -> tuple[Mapping[str, Any], ...]:
        if self._epoch_source_row_start is None or self._epoch_started_at is None:
            raise RuntimeError("causal case epoch prefix is not initialized")
        frames = snapshot.observation.frames
        refs: list[Mapping[str, Any]] = []
        for timeframe in (
            Timeframe.H4,
            Timeframe.H1,
            Timeframe.M15,
            Timeframe.M5,
            Timeframe.M1,
        ):
            frame = frames.get(timeframe)
            if frame is None:
                continue
            cutoff = _clock(frame.cutoff, name=f"causal_case.{timeframe.value}.cutoff")
            if cutoff > asof:
                raise ValueError("causal case OHLCV prefix contains the future")
            refs.append(
                {
                    "timeframe": timeframe.value,
                    "frame_row_start": 0,
                    "frame_row_end_exclusive": int(frame.bars),
                    "replay_view_1m_row_start": int(self._epoch_source_row_start),
                    "replay_view_1m_row_end_exclusive": int(source_row_ordinal) + 1,
                    "start_at": self._epoch_started_at,
                    "end_at": cutoff,
                    "asof": asof,
                    "boundary_semantics": "[start_at,end_at]_completed_prefix",
                    "reload_rule": (
                        "filter_canonical_source_by_timezone_aware_time_bounds;"
                        "aggregate_to_timeframe;never_use_frame_row_as_global_index"
                    ),
                }
            )
        return tuple(refs)

    @staticmethod
    def _admission_quality_reason(
        episode: Any,
        context: Any,
        admitted_at: pd.Timestamp,
        observation: Any,
    ) -> str | None:
        updated_at = _clock(episode.updated_at, name="causal_case.episode.updated_at")
        if updated_at != admitted_at:
            return "episode_not_admitted_on_current_revision"
        for name, raw in (
            ("first_pullback", getattr(episode, "first_pullback_at", None)),
            (
                "trigger",
                None
                if getattr(episode, "selected_trigger", None) is None
                else episode.selected_trigger.observed_at,
            ),
            ("terminal", getattr(episode, "terminal_at", None)),
        ):
            if raw is not None and _clock(raw, name=f"causal_case.{name}_at") < admitted_at:
                return f"late_admission_with_prior_{name}"
        location_id = _identity(getattr(episode, "entry_location_id", None))
        path_id = _identity(getattr(episode, "entry_path_id", None))
        if (
            location_id is not None
            and _enum_value(getattr(episode, "playbook", None))
            == "liquidity_sweep_reversal"
        ):
            locations = tuple(
                item
                for item in getattr(observation, "entry_locations", ())
                if getattr(item, "location_id", None) == location_id
            )
            if len(locations) != 1:
                return "admission_missing_exact_physical_zone"
            location = locations[0]
            authority_ids = frozenset(getattr(context, "authority_ids", ()))
            if (
                getattr(episode, "direction", None)
                is not getattr(context, "direction", None)
            ):
                return "admission_lsr_direction_mismatch"
            if getattr(episode, "initiating_event_id", None) not in authority_ids:
                return "admission_lsr_root_not_owned_by_context"
            if getattr(location, "source_displacement_id", None) not in authority_ids:
                return "admission_lsr_displacement_not_owned_by_context"
            if (
                getattr(location, "direction", None)
                is not getattr(episode, "direction", None)
                or getattr(location, "symbol", None)
                != getattr(observation, "symbol", None)
                or int(getattr(location, "instrument_id", -1))
                != int(getattr(observation, "instrument_id", -2))
            ):
                return "admission_lsr_zone_contract_mismatch"
            if _clock(
                location.formed_at,
                name="causal_case.physical_zone.formed_at",
            ) < admitted_at:
                return "late_admission_with_prior_zone_formation"
            first_entered_at = getattr(location, "first_entered_at", None)
            if first_entered_at is not None and _clock(
                first_entered_at,
                name="causal_case.physical_zone.first_entered_at",
            ) < admitted_at:
                return "late_admission_with_prior_physical_first_entry"
            paths = tuple(
                item
                for item in getattr(observation, "path_sequences", ())
                if getattr(item, "sequence_id", None) == path_id
            )
            if path_id is None or len(paths) != 1:
                return "admission_missing_exact_physical_path"
            path = paths[0]
            if (
                getattr(path, "context_kind", None) != "zone_return"
                or getattr(path, "context_id", None) != location_id
                or getattr(path, "direction", None)
                is not getattr(episode, "direction", None)
                or getattr(path, "symbol", None)
                != getattr(observation, "symbol", None)
                or int(getattr(path, "instrument_id", -1))
                != int(getattr(observation, "instrument_id", -2))
            ):
                return "admission_lsr_path_ownership_mismatch"
            if _clock(
                path.formed_at,
                name="causal_case.physical_path.formed_at",
            ) < admitted_at:
                return "late_admission_with_prior_path_formation"
        return None

    @staticmethod
    def _validate_case_clocks(context: Any, episode: Any, asof: pd.Timestamp) -> None:
        observed = (
            ("context.formed_at", context.formed_at),
            ("context.updated_at", context.updated_at),
            ("context.terminal_at", context.terminal_at),
            ("episode.formed_at", episode.formed_at),
            ("episode.updated_at", episode.updated_at),
            ("episode.first_pullback_at", episode.first_pullback_at),
            ("episode.terminal_at", episode.terminal_at),
            (
                "episode.selected_trigger_at",
                None
                if episode.selected_trigger is None
                else episode.selected_trigger.observed_at,
            ),
        )
        for name, raw in observed:
            if raw is not None and _clock(raw, name=f"causal_case.{name}") > asof:
                raise ValueError(f"causal case observed clock is in the future: {name}")

    def observe(
        self,
        snapshot: Any,
        *,
        source_bar: Any,
        source_row_ordinal: int,
        replay_update_ordinal: int | None = None,
        scene_graph: Any | None = None,
    ) -> None:
        """Emit only newly reached sparse revisions for current episodes."""

        replay_ordinal = (
            0
            if replay_update_ordinal is None and self._last_replay_update_ordinal is None
            else (
                int(self._last_replay_update_ordinal) + 1
                if replay_update_ordinal is None
                else int(replay_update_ordinal)
            )
        )
        self._validate_replay_coordinates(
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_ordinal,
            source_bar=source_bar,
        )
        asof = self._validate_snapshot(snapshot, source_bar)
        runtime_epoch_id = self._snapshot_runtime_epoch(snapshot)
        epoch_id = self._snapshot_epoch(snapshot, runtime_epoch_id)
        anomalies = set(getattr(snapshot.observation, "anomalies", ()))
        boundary_anomalies = sorted(anomalies.intersection(_BOUNDARY_ANOMALIES))
        epoch_reason = (
            boundary_anomalies[0]
            if boundary_anomalies
            else "market_epoch_changed"
        )
        self._advance_epoch(
            epoch_id,
            asof=asof,
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_ordinal,
            source_bar_start=source_bar.start,
            reason=epoch_reason,
        )
        transition_update = _case_transition_update(
            snapshot,
            replay_update_ordinal=replay_ordinal,
            scene_graph=scene_graph,
        )
        for case in self._cases.values():
            if case.market_epoch_id != epoch_id or case.terminal_at is not None:
                continue
            self._accumulate_transition_update(case, transition_update)
        if source_bar.synthetic_no_trade or asof < self._capture_start:
            self._remember_epoch_snapshot(
                snapshot,
                source_bar=source_bar,
                source_row_ordinal=source_row_ordinal,
                replay_update_ordinal=replay_ordinal,
            )
            return
        contexts = getattr(snapshot.belief, "context_theses", {})
        episodes = getattr(snapshot.belief, "entry_episodes", {})
        if not isinstance(contexts, Mapping) or not isinstance(episodes, Mapping):
            raise TypeError("causal case recorder requires typed Context/Episode mappings")
        seen_episode_ids: set[str] = set()
        for candidate_key, episode in sorted(episodes.items(), key=lambda item: str(item[0])):
            episode_id = _identity(getattr(episode, "episode_id", None))
            context_id = _identity(getattr(episode, "parent_context_thesis_id", None))
            if episode_id is None or context_id is None or episode_id in seen_episode_ids:
                raise ValueError("causal case EntryEpisode identity is missing or duplicated")
            seen_episode_ids.add(episode_id)
            if episode_id in self._ignored_episode_ids or episode_id in self._rejected_episode_ids:
                continue
            context = contexts.get(context_id)
            if context is None or getattr(context, "context_thesis_id", None) != context_id:
                self._skip(episode_id, "missing_exact_parent_context")
                continue
            if context.market_epoch_id != runtime_epoch_id:
                self._skip(episode_id, "context_epoch_mismatch")
                continue
            self._validate_case_clocks(context, episode, asof)
            existing_case_id = self._episode_to_case.get(episode_id)
            new_case = existing_case_id is None
            if existing_case_id is not None:
                prior_case = self._cases[existing_case_id]
                if prior_case.market_epoch_id != epoch_id:
                    self._skip(episode_id, "episode_identity_crossed_epoch")
                    continue
                if prior_case.terminal_at is not None:
                    continue
            if new_case:
                quality_reason = self._admission_quality_reason(
                    episode,
                    context,
                    asof,
                    snapshot.observation,
                )
                if quality_reason is not None:
                    self._skip(episode_id, quality_reason)
                    continue
                prior_epoch = self._episode_epoch.get(episode_id)
                if prior_epoch is not None and prior_epoch != epoch_id:
                    self._skip(episode_id, "episode_identity_crossed_epoch")
                    continue
                case_id = _case_id(epoch_id, context_id, episode_id)
                case = _CaseState(
                    case_id=case_id,
                    market_epoch_id=epoch_id,
                    context_thesis_id=context_id,
                    entry_episode_id=episode_id,
                    candidate_id=str(getattr(episode, "candidate_id", candidate_key)),
                    playbook=str(_enum_value(getattr(episode, "playbook", None))),
                    admitted_at=asof,
                    admitted_replay_update_ordinal=replay_ordinal,
                    context_signature=_context_signature(context),
                    entry_location_id=_identity(
                        getattr(episode, "entry_location_id", None)
                    ),
                    entry_path_id=_identity(
                        getattr(episode, "entry_path_id", None)
                    ),
                    last_context=context,
                    last_episode=episode,
                )
                self._accumulate_transition_update(case, transition_update)
                self._cases[case_id] = case
                self._episode_to_case[episode_id] = case_id
                self._episode_epoch[episode_id] = epoch_id
            else:
                case = self._cases[existing_case_id]
                if (
                    case.market_epoch_id != epoch_id
                    or case.context_thesis_id != context_id
                    or case.candidate_id != str(getattr(episode, "candidate_id", candidate_key))
                ):
                    raise ValueError("causal case episode identity mutated")
                case.last_context = context
                case.last_episode = episode
            stages: list[tuple[str, str, pd.Timestamp]] = []
            context_signature = _context_signature(context)
            if new_case:
                if _clock(
                    context.formed_at,
                    name="causal_case.context.formed_at",
                ) == asof:
                    stages.append(
                        (
                            "context_formed",
                            _stage_identity("context_formed", context_signature),
                            asof,
                        )
                    )
                stages.append(
                    (
                        "episode_created",
                        _stage_identity("episode_created", episode_id),
                        asof,
                    )
                )
            elif context_signature != case.context_signature:
                if _clock(context.updated_at, name="causal_case.context.updated_at") != asof:
                    raise ValueError("causal case Context material change arrived late")
                stages.append(
                    (
                        "context_changed",
                        _stage_identity("context_changed", context_signature),
                        asof,
                    )
                )
                case.context_signature = context_signature

            entry_location_id = _identity(getattr(episode, "entry_location_id", None))
            entry_path_id = _identity(getattr(episode, "entry_path_id", None))
            if case.entry_location_id is not None and entry_location_id != case.entry_location_id:
                raise ValueError("causal case EntryEpisode changed its frozen zone")
            if case.entry_path_id is not None and entry_path_id != case.entry_path_id:
                raise ValueError("causal case EntryEpisode changed its frozen path")
            if entry_location_id is not None:
                case.entry_location_id = entry_location_id
                case.entry_path_id = entry_path_id
                matching_locations = tuple(
                    item
                    for item in getattr(snapshot.observation, "entry_locations", ())
                    if getattr(item, "location_id", None) == entry_location_id
                )
                if len(matching_locations) == 1:
                    location = matching_locations[0]
                    source_displacement_id = _identity(
                        getattr(location, "source_displacement_id", None)
                    )
                    source_zone_id = _identity(
                        getattr(location, "source_zone_id", None)
                    )
                    if (
                        case.source_displacement_id is not None
                        and source_displacement_id != case.source_displacement_id
                    ) or (
                        case.source_zone_id is not None
                        and source_zone_id != case.source_zone_id
                    ):
                        raise ValueError("causal case physical zone custody mutated")
                    case.source_displacement_id = source_displacement_id
                    case.source_zone_id = source_zone_id
                stages.append(
                    (
                        "zone_registered",
                        _stage_identity("zone_registered", entry_location_id),
                        asof,
                    )
                )
            if case.playbook == "liquidity_sweep_reversal":
                execution_owner_phase = episode.phase in {
                    PlaybookPhase.EXECUTABLE,
                    PlaybookPhase.ENTERED,
                    PlaybookPhase.DELIVERING,
                }
                execution_locked = any(
                    value is not None
                    for value in (
                        case.frozen_execution_plan_identity,
                        case.frozen_execution_trigger_identity,
                        case.frozen_execution_lock_at,
                    )
                )
                if execution_locked:
                    if any(
                        value is None
                        for value in (
                            case.frozen_execution_plan_identity,
                            case.frozen_execution_trigger_identity,
                            case.frozen_execution_lock_at,
                        )
                    ):
                        raise ValueError(
                            "causal case frozen LSR execution lock is partial"
                        )
                    terminal_episode = episode.terminal_at is not None
                    if not terminal_episode and (
                        episode.plan is None or episode.selected_trigger is None
                    ):
                        raise ValueError(
                            "causal case frozen LSR execution custody disappeared"
                        )
                    if (
                        episode.plan is not None
                        and _lsr_execution_plan_identity(episode.plan)
                        != case.frozen_execution_plan_identity
                    ):
                        raise ValueError(
                            "causal case frozen LSR execution plan mutated"
                        )
                    if (
                        episode.selected_trigger is not None
                        and _lsr_execution_trigger_identity(
                            episode.selected_trigger
                        )
                        != case.frozen_execution_trigger_identity
                    ):
                        raise ValueError(
                            "causal case frozen LSR execution trigger mutated"
                        )
                    if episode.selected_trigger is not None:
                        observed_kinds = tuple(
                            episode.selected_trigger.available_trigger_kinds
                        )
                        prior_kinds = case.observed_execution_trigger_kinds
                        if observed_kinds[: len(prior_kinds)] != prior_kinds:
                            raise ValueError(
                                "causal case LSR trigger evidence families are not append-only"
                            )
                        case.observed_execution_trigger_kinds = observed_kinds
                elif execution_owner_phase:
                    if episode.plan is None or episode.selected_trigger is None:
                        raise ValueError(
                            "causal case LSR execution owner lacks plan or trigger"
                        )
                    case.frozen_execution_lock_at = asof
                    case.frozen_execution_plan_identity = (
                        _lsr_execution_plan_identity(episode.plan)
                    )
                    case.frozen_execution_trigger_identity = (
                        _lsr_execution_trigger_identity(
                            episode.selected_trigger
                        )
                    )
                    case.observed_execution_trigger_kinds = tuple(
                        episode.selected_trigger.available_trigger_kinds
                    )
            if episode.first_pullback_at is not None:
                pullback_at = _clock(
                    episode.first_pullback_at,
                    name="causal_case.first_pullback_at",
                )
                stages.append(
                    (
                        "first_pullback",
                        _stage_identity("first_pullback", pullback_at.isoformat()),
                        pullback_at,
                    )
                )
            trigger = episode.selected_trigger
            if trigger is not None:
                trigger_at = _clock(trigger.observed_at, name="causal_case.trigger_at")
                stages.append(
                    (
                        "trigger",
                        _stage_identity("trigger", trigger.trigger_id),
                        trigger_at,
                    )
                )
            if episode.plan is not None:
                plan_identity = _plan_custody_identity(episode.plan)
                stages.append(
                    (
                        "plan_formed",
                        _stage_identity("plan_formed", plan_identity),
                        asof,
                    )
                )
            if episode.terminal_at is not None:
                terminal_at = _clock(episode.terminal_at, name="causal_case.terminal_at")
                stages.append(
                    (
                        "terminal",
                        _stage_identity(
                            "terminal",
                            f"{terminal_at.isoformat()}|{episode.terminal_reason}",
                        ),
                        terminal_at,
                    )
                )
                case.terminal_at = terminal_at
                case.terminal_reason = episode.terminal_reason
            for stage, stage_key, stage_at in stages:
                if stage_key in case.stage_identities:
                    continue
                if stage in {"first_pullback", "trigger", "terminal"} and (
                    stage_at != asof
                    or _clock(
                        episode.updated_at,
                        name="causal_case.episode.updated_at",
                    )
                    != asof
                ):
                    raise ValueError(
                        "causal case milestone was first observed as a late backfill: "
                        f"{stage}"
                    )
                prior_same_stage = tuple(
                    item
                    for item in case.stage_identities
                    if item.startswith(f"{stage}:")
                )
                if prior_same_stage and stage != "context_changed":
                    raise ValueError(
                        "causal case immutable episode stage changed identity: "
                        f"{stage}"
                    )
                self._emit_input(
                    case,
                    snapshot,
                    context=context,
                    episode=episode,
                    source_bar=source_bar,
                    source_row_ordinal=source_row_ordinal,
                    stage=stage,
                    stage_identity=stage_key,
                    stage_observed_at=stage_at,
                    replay_update_ordinal=replay_ordinal,
                )
            if new_case:
                for payload in self._pending_shadow_by_episode.pop(episode_id, ()):
                    self._consider_shadow(case, payload)
        self._remember_epoch_snapshot(
            snapshot,
            source_bar=source_bar,
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_ordinal,
        )

    @staticmethod
    def _transition_views(
        case: _CaseState,
        *,
        asof: pd.Timestamp,
        replay_update_ordinal: int,
        boundary_terminal_reason: str | None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, ...], tuple[str, ...]]:
        updates = list(case.transition_updates)
        observed_update_count = int(case.coverage_observed_update_count)
        coverage_last_observed_at = case.coverage_last_observed_at
        coverage_last_replay_update_ordinal = (
            case.coverage_last_replay_update_ordinal
        )
        typed_complete = bool(case.coverage_all_typed_available)
        gap_free = bool(case.coverage_gap_free)
        if boundary_terminal_reason is not None:
            updates.append(
                {
                    "asof": asof,
                    "replay_update_ordinal": replay_update_ordinal,
                    "observation_transition": {
                        "typed_transition_delta_available": True,
                        "collections": {},
                        "case_boundary_terminal": {
                            "reason": boundary_terminal_reason,
                            "observed_at": asof,
                            "last_old_epoch_observation_at": (
                                coverage_last_observed_at
                            ),
                        },
                    },
                    "scene_graph_delta": {
                        "asof": asof,
                        "revision_id": None,
                        "added_node_ids": (),
                        "revised_node_ids": (),
                        "added_edge_ids": (),
                        "revised_edge_ids": (),
                        "resolution_event_ids": (),
                        "relation_descriptors": (),
                        "relation_descriptors_complete": True,
                    },
                    "added_event_ids": (),
                    "invalidated_event_ids": _canonical_identity_set(
                        (
                            case.context_thesis_id,
                            case.entry_episode_id,
                        ),
                        name="invalidated event ID summary",
                    ),
                }
            )
            observed_update_count += 1
            coverage_last_observed_at = asof
            coverage_last_replay_update_ordinal = replay_update_ordinal
        update_clocks = tuple(
            _clock(update["asof"], name="causal_case.transition_update.asof")
            for update in updates
        )
        if any(
            left >= right
            for left, right in zip(update_clocks, update_clocks[1:])
        ):
            raise ValueError("causal case transition coverage clocks are not increasing")
        if update_clocks and update_clocks[-1] > asof:
            raise ValueError("causal case transition coverage exceeds revision asof")
        interval_complete = bool(
            (
                observed_update_count > 0
                and coverage_last_observed_at == asof
            )
            or (
                observed_update_count == 0
                and case.last_revision_at is not None
                and case.last_revision_at == asof
            )
        )
        coverage_start_replay_update_ordinal = (
            case.admitted_replay_update_ordinal
            if case.last_revision_replay_update_ordinal is None
            else case.last_revision_replay_update_ordinal
        )
        expected_observed_update_count = (
            replay_update_ordinal
            - coverage_start_replay_update_ordinal
            + int(case.last_revision_replay_update_ordinal is None)
        )
        ordinal_complete = bool(
            expected_observed_update_count >= 0
            and observed_update_count == expected_observed_update_count
            and (
                (
                    observed_update_count > 0
                    and coverage_last_replay_update_ordinal
                    == replay_update_ordinal
                )
                or (
                    observed_update_count == 0
                    and coverage_last_replay_update_ordinal is None
                )
            )
        )
        coverage = {
            "coverage_start_at": (
                case.admitted_at
                if case.last_revision_at is None
                else case.last_revision_at
            ),
            "coverage_start_exclusive": case.last_revision_at is not None,
            "coverage_end_at": asof,
            "coverage_start_replay_update_ordinal": (
                coverage_start_replay_update_ordinal
            ),
            "coverage_end_replay_update_ordinal": replay_update_ordinal,
            "complete": bool(
                typed_complete
                and gap_free
                and interval_complete
                and ordinal_complete
            ),
            "observed_update_count": observed_update_count,
            "eventful_update_count": len(updates),
            "last_observed_update_at": coverage_last_observed_at,
            "last_observed_replay_update_ordinal": (
                coverage_last_replay_update_ordinal
            ),
            "all_typed_deltas_available": typed_complete,
            "gap_free": gap_free,
        }
        collections: dict[str, list[Any]] = {}
        scene_values: dict[str, list[Any]] = {
            "added_node_ids": [],
            "revised_node_ids": [],
            "added_edge_ids": [],
            "revised_edge_ids": [],
            "resolution_event_ids": [],
        }
        relation_descriptors: list[Mapping[str, Any]] = []
        relation_descriptors_complete = True
        added_ids: list[str] = []
        invalidated_ids: list[str] = []
        observation_updates: list[Mapping[str, Any]] = []
        scene_updates: list[Mapping[str, Any]] = []
        scene_revision_id: str | None = None
        for update in updates:
            observation = update["observation_transition"]
            observation_updates.append(
                {
                    "asof": update["asof"],
                    "replay_update_ordinal": update["replay_update_ordinal"],
                    **dict(observation),
                }
            )
            for name, values in observation.get("collections", {}).items():
                collections.setdefault(str(name), []).extend(tuple(values))
            scene = update["scene_graph_delta"]
            scene_updates.append(
                {
                    **dict(scene),
                    "replay_update_ordinal": update["replay_update_ordinal"],
                }
            )
            if scene.get("revision_id") is not None:
                scene_revision_id = str(scene["revision_id"])
            for name in scene_values:
                scene_values[name].extend(tuple(scene.get(name, ())))
            relation_descriptors.extend(
                tuple(scene.get("relation_descriptors", ()))
            )
            relation_descriptors_complete = bool(
                relation_descriptors_complete
                and scene.get("relation_descriptors_complete", False)
            )
            added_ids.extend(str(value) for value in update["added_event_ids"])
            invalidated_ids.extend(
                str(value) for value in update["invalidated_event_ids"]
            )
        observation_payload: dict[str, Any] = {
            "typed_transition_delta_available": typed_complete,
            "coverage": coverage,
            "collections": {
                name: tuple(values)
                for name, values in sorted(collections.items())
            },
            "updates": tuple(observation_updates),
        }
        if boundary_terminal_reason is not None:
            observation_payload["case_boundary_terminal"] = {
                "reason": boundary_terminal_reason,
                "observed_at": asof,
            }
        scene_payload = {
            "asof": asof,
            "revision_id": scene_revision_id,
            "coverage": coverage,
            **{
                name: _canonical_identity_set(
                    values,
                    name=f"aggregate Scene {name}",
                )
                for name, values in scene_values.items()
            },
            "relation_descriptors": tuple(relation_descriptors),
            "relation_descriptors_complete": relation_descriptors_complete,
            "updates": tuple(scene_updates),
        }
        return (
            observation_payload,
            scene_payload,
            _canonical_identity_set(
                added_ids,
                name="aggregate added event ID summary",
            ),
            _canonical_identity_set(
                invalidated_ids,
                name="aggregate invalidated event ID summary",
            ),
        )

    def _emit_input(
        self,
        case: _CaseState,
        snapshot: Any,
        *,
        context: Any,
        episode: Any,
        source_bar: Any,
        source_row_ordinal: int,
        replay_update_ordinal: int,
        stage: str,
        stage_identity: str,
        stage_observed_at: pd.Timestamp,
        decision_asof: pd.Timestamp | None = None,
        boundary_terminal_reason: str | None = None,
    ) -> None:
        snapshot_asof = _clock(
            snapshot.observation.asof,
            name="causal_case.snapshot_asof",
        )
        asof = (
            snapshot_asof
            if decision_asof is None
            else _clock(decision_asof, name="causal_case.asof")
        )
        if snapshot_asof > asof:
            raise ValueError("causal case boundary prefix is after its decision clock")
        if stage_observed_at > asof:
            raise ValueError("causal case revision stage is future-dated")
        plan = episode.plan
        trigger = episode.selected_trigger
        global_context = getattr(snapshot.belief, "global_context", None)
        (
            observation_transition,
            scene_graph_delta,
            added_ids,
            invalidated_ids,
        ) = self._transition_views(
            case,
            asof=asof,
            replay_update_ordinal=replay_update_ordinal,
            boundary_terminal_reason=boundary_terminal_reason,
        )
        brain_response = dict(_brain_response_payload(snapshot))
        if boundary_terminal_reason is not None:
            brain_response.update(
                {
                    "selected_action": "abstain",
                    "best_hypothesis_key": None,
                    "decision_reasons": (
                        f"case_boundary_terminal:{boundary_terminal_reason}",
                    ),
                    "decision_plan": None,
                    "risk_action": "abstain",
                    "risk_passed": False,
                    "risk_reasons": (
                        f"case_boundary_terminal:{boundary_terminal_reason}",
                    ),
                }
            )
        invalidation = (
            episode.invalidation
            if episode.invalidation is not None
            else None if plan is None else plan.invalidation
        )
        primary_target = None if plan is None or not plan.targets else plan.targets[0]
        prefix_refs = self._prefix_refs(
            snapshot,
            asof=asof,
            source_row_ordinal=source_row_ordinal,
        )
        normalization_cutoff = _clock(
            source_bar.start,
            name="causal_case.normalization_cutoff",
        )
        if normalization_cutoff >= asof:
            raise ValueError("causal case normalization cutoff is not strictly prior")
        authority = {
            "context_authority_ids": tuple(context.authority_ids),
            "authority_stack": (
                ()
                if global_context is None
                else to_primitive(global_context.authority_stack)
            ),
            "authority_direction": (
                None
                if global_context is None
                else _enum_value(global_context.authority_direction)
            ),
            "authority_timeframe": (
                None
                if global_context is None
                else _enum_value(global_context.authority_timeframe)
            ),
        }
        scale_relations = (
            {} if global_context is None else global_context.scale_relation_details
        )
        draw = {
            "context_draw": to_primitive(context.context_draw),
            "episode_primary_target": to_primitive(primary_target),
            "selected_draw_id": None if plan is None else plan.selected_draw_id,
        }
        blockers = {
            "obstruction_views": (
                {} if global_context is None else to_primitive(global_context.obstruction_views)
            ),
            "episode_path_blocker_ids": (
                ()
                if plan is None or plan.liquidity_route is None
                else tuple(plan.liquidity_route.path_blocker_ids)
            ),
        }
        ambiguities = {
            "belief_unresolved": tuple(snapshot.belief.unresolved_ambiguities),
            "global_unknown": (
                () if global_context is None else tuple(global_context.unknown_evidence)
            ),
            "global_ambiguous": (
                () if global_context is None else tuple(global_context.ambiguous_evidence)
            ),
            "material_conflicts": (
                () if global_context is None else to_primitive(global_context.material_conflicts)
            ),
        }
        base = {
            "revision_id": "",
            "case_id": case.case_id,
            "revision_index": case.next_revision_index,
            "revision_stage": stage,
            "stage_identity": stage_identity,
            "stage_observed_at": stage_observed_at,
            "input_fingerprint": "",
            "market_epoch_id": case.market_epoch_id,
            "context_thesis_id": case.context_thesis_id,
            "entry_episode_id": case.entry_episode_id,
            "candidate_id": case.candidate_id,
            "playbook": str(_enum_value(episode.playbook)),
            "direction": str(_enum_value(episode.direction)),
            "observable_regime": _observable_regime(snapshot, episode),
            "mechanism_label": str(_enum_value(episode.playbook)),
            "asof": asof,
            "decision_at": asof,
            "admitted_at": case.admitted_at,
            "symbol": snapshot.observation.symbol,
            "instrument_id": int(snapshot.observation.instrument_id),
            "decision_price": _finite(snapshot.observation.price),
            "source_path": self._source_path,
            "source_sha256": self._source_sha256,
            "source_role": self._source_role,
            "split_role": self._split_role,
            "prefix_refs_json": _json(prefix_refs),
            "normalization_cutoff_at": normalization_cutoff,
            "normalization_policy": "bars_with_end_strictly_before_decision_asof",
            "added_event_ids_json": _json(added_ids),
            "invalidated_event_ids_json": _json(invalidated_ids),
            "observation_transition_json": _json(
                observation_transition
            ),
            "scene_graph_delta_json": _json(scene_graph_delta),
            "context_thesis_json": _json(context),
            "entry_episode_json": _json(episode),
            "brain_response_json": _json(brain_response),
            "authority_json": _json(authority),
            "scale_relations_json": _json(scale_relations),
            "draw_json": _json(draw),
            "blockers_json": _json(blockers),
            "ambiguities_json": _json(ambiguities),
            "supporting_evidence_json": _json(context.supporting_event_ids),
            "opposing_evidence_json": _json(context.opposing_event_ids),
            "entry_location_id": episode.entry_location_id,
            "entry_path_id": episode.entry_path_id,
            "planned_entry": None if plan is None else _finite(plan.planned_entry),
            "entry_zone_lower": None if plan is None else _finite(plan.entry_zone_lower),
            "entry_zone_upper": None if plan is None else _finite(plan.entry_zone_upper),
            "invalidation_price": (
                None if invalidation is None else _finite(invalidation.price)
            ),
            "invalidation_source_id": (
                None if invalidation is None else invalidation.source_level_id
            ),
            "primary_target_price": (
                None if primary_target is None else _finite(primary_target.price)
            ),
            "primary_target_id": (
                None if primary_target is None else primary_target.level_id
            ),
            "planned_deadline_at": episode.deadline,
            "selected_trigger_id": None if trigger is None else trigger.trigger_id,
            "selected_trigger_kind": None if trigger is None else trigger.trigger_kind,
            "selected_trigger_at": None if trigger is None else trigger.observed_at,
            "observed_terminal_reason": (
                boundary_terminal_reason
                if boundary_terminal_reason is not None
                else episode.terminal_reason
            ),
            "model_versions_json": _json(self._model_versions),
        }
        fingerprint = case_input_fingerprint(base)
        if fingerprint in self._input_fingerprints:
            raise ValueError("causal case input features were duplicated")
        base["input_fingerprint"] = fingerprint
        revision_id = "case-revision:" + hashlib.sha256(
            f"{case.case_id}|{stage_identity}|{fingerprint}".encode("utf-8")
        ).hexdigest()[:32]
        if revision_id in self._revision_ids:
            raise ValueError("causal case revision identity was duplicated")
        base["revision_id"] = revision_id
        record = CausalCaseInputRecord(**base)
        validate_case_input_row(record.to_dict())
        case.stage_identities.add(stage_identity)
        case.next_revision_index += 1
        self._input_fingerprints.add(fingerprint)
        self._revision_ids.add(revision_id)
        self._input_rows.append(record)
        case.last_revision_at = asof
        case.last_revision_replay_update_ordinal = replay_update_ordinal
        case.transition_updates.clear()
        case.coverage_observed_update_count = 0
        case.coverage_last_observed_at = None
        case.coverage_last_replay_update_ordinal = None
        case.coverage_all_typed_available = True
        case.coverage_gap_free = True

    def consume_shadow_records(self, records: Iterable[Any]) -> None:
        """Consume already-resolved Shadow rows without exposing them to inputs."""

        for item in records:
            payload = (
                dict(item)
                if isinstance(item, Mapping)
                else item.to_dict()
                if hasattr(item, "to_dict")
                else asdict(item)
                if is_dataclass(item)
                else None
            )
            if payload is None:
                raise TypeError("causal case Shadow record is not serializable")
            episode_id = _identity(payload.get("source_episode_id"))
            if episode_id is None:
                continue
            if (
                episode_id in self._ignored_episode_ids
                or episode_id in self._rejected_episode_ids
            ):
                self._count_quality("shadow_for_ignored_or_rejected_episode")
                continue
            case_id = self._episode_to_case.get(episode_id)
            if case_id is None:
                # Runner order is case.observe -> Shadow observe/drain on the
                # same update, so a causally admissible episode is already
                # mapped before its Shadow row can arrive.  Unknown episodes
                # are never retained across a long replay window.
                self._count_quality("shadow_for_unadmitted_episode")
                continue
            self._consider_shadow(self._cases[case_id], payload)

    def _consider_shadow(self, case: _CaseState, payload: Mapping[str, Any]) -> None:
        context_id = _identity(payload.get("source_context_thesis_id"))
        if context_id != case.context_thesis_id:
            self._count_quality("shadow_context_custody_missing_or_mismatch")
            return
        binding_status = str(payload.get("entry_episode_binding_status") or "")
        payload_location_id = _identity(payload.get("entry_location_id"))
        payload_path_id = _identity(payload.get("entry_path_id"))
        if (
            "ambiguous" in binding_status
            or case.entry_location_id is None
            or case.entry_path_id is None
            or payload_location_id != case.entry_location_id
            or payload_path_id != case.entry_path_id
        ):
            self._count_quality("shadow_entry_custody_missing_or_mismatch")
            return
        if case.playbook == "liquidity_sweep_reversal":
            payload_displacement_id = _identity(payload.get("lsr_displacement_id"))
            payload_zone_id = _identity(payload.get("lsr_entry_zone_id"))
            if (
                case.source_displacement_id is None
                or case.source_zone_id is None
                or payload_displacement_id != case.source_displacement_id
                or payload_zone_id != case.source_zone_id
            ):
                self._count_quality("shadow_lsr_zone_custody_missing_or_mismatch")
                return
        observed_at = _clock(payload["observed_at"], name="causal_case.shadow.observed_at")
        resolved_at = _clock(payload["resolved_at"], name="causal_case.shadow.resolved_at")
        if resolved_at < observed_at or observed_at < case.admitted_at:
            return
        event_kind = str(payload.get("event_kind") or payload.get("candidate_origin") or "")
        priority = int(_SHADOW_PRIORITY.get(event_kind, 0))
        candidate_id = str(payload.get("candidate_id") or "")
        if priority < case.shadow_priority:
            return
        if priority == case.shadow_priority and case.shadow_payload is not None:
            prior_id = str(case.shadow_payload.get("candidate_id") or "")
            if candidate_id == prior_id:
                return
            prior_observed = _clock(
                case.shadow_payload["observed_at"],
                name="causal_case.prior_shadow.observed_at",
            )
            if (observed_at, candidate_id) >= (prior_observed, prior_id):
                return
        case.shadow_payload = dict(payload)
        case.shadow_priority = priority

    @staticmethod
    def _first_event(payload: Mapping[str, Any]) -> str:
        return _project_shadow_first_event(
            resolution=payload.get("resolution"),
            filled=payload.get("filled"),
            censored=payload.get("censored"),
            target_first=payload.get("target_before_invalidation"),
            invalidation_first=payload.get("invalidation_before_target"),
            same_bar_collision=payload.get("same_bar_collision"),
        )

    def close_unresolved(self, asof: pd.Timestamp) -> None:
        """Write exactly one future-label row per admitted case."""

        if self._closed:
            raise RuntimeError("causal case recorder was closed twice")
        clock = _clock(asof, name="causal_case.close_asof")
        if self._last_observation_asof is not None and clock < self._last_observation_asof:
            raise ValueError("causal case close clock precedes observations")
        pending_count = sum(
            len(values) for values in self._pending_shadow_by_episode.values()
        )
        if pending_count:
            self._count_quality(
                "shadow_pending_discarded_at_close",
                pending_count,
            )
            self._pending_shadow_by_episode.clear()
        for case in sorted(self._cases.values(), key=lambda item: item.case_id):
            payload = case.shadow_payload
            if payload is None:
                terminal = case.terminal_at is not None
                resolution = (
                    "episode_terminal_without_shadow_path"
                    if terminal
                    else "window_end_right_censored"
                )
                first_event = "episode_terminal" if terminal else "right_censored"
                resolved_at = case.terminal_at or clock
                record = CausalCaseOutcomeRecord(
                    outcome_id=f"case-outcome:{case.case_id.split(':', 1)[-1]}",
                    case_id=case.case_id,
                    market_epoch_id=case.market_epoch_id,
                    context_thesis_id=case.context_thesis_id,
                    entry_episode_id=case.entry_episode_id,
                    resolved_at=resolved_at,
                    first_event=first_event,
                    target_first=None,
                    invalidation_first=None,
                    deadline_first=False,
                    same_bar_collision=False,
                    mfe_points=None,
                    mae_points=None,
                    mfe_R=None,
                    mae_R=None,
                    hit_0_5R=None,
                    hit_1R=None,
                    hit_2R=None,
                    draw_delivered=None,
                    time_to_draw_real_1m_bars=None,
                    filled=False,
                    expired=False,
                    terminal_reason=case.terminal_reason,
                    resolution=resolution,
                    censored=not terminal,
                    source_shadow_candidate_id=None,
                )
            else:
                first_event = self._first_event(payload)
                resolution = str(payload.get("resolution") or first_event)
                deadline_first = first_event == "deadline"
                time_to_draw = payload.get("time_to_draw_real_bars")
                shadow_resolved_at = _clock(
                    payload["resolved_at"],
                    name="causal_case.resolved_at",
                )
                # ``resolved_at`` is the clock at which the complete outcome
                # row is safe to expose, not an encoder feature or the hidden
                # first-hit timestamp.  It therefore cannot precede a later
                # observable sparse input revision for the same Episode.
                outcome_available_at = max(
                    shadow_resolved_at,
                    case.last_revision_at or case.admitted_at,
                )
                record = CausalCaseOutcomeRecord(
                    outcome_id=f"case-outcome:{case.case_id.split(':', 1)[-1]}",
                    case_id=case.case_id,
                    market_epoch_id=case.market_epoch_id,
                    context_thesis_id=case.context_thesis_id,
                    entry_episode_id=case.entry_episode_id,
                    resolved_at=outcome_available_at,
                    first_event=first_event,
                    target_first=(
                        None
                        if payload.get("target_before_invalidation") is None
                        else bool(payload.get("target_before_invalidation"))
                    ),
                    invalidation_first=(
                        None
                        if payload.get("invalidation_before_target") is None
                        else bool(payload.get("invalidation_before_target"))
                    ),
                    deadline_first=deadline_first,
                    same_bar_collision=bool(payload.get("same_bar_collision", False)),
                    mfe_points=_finite(payload.get("mfe_points")),
                    mae_points=_finite(payload.get("mae_points")),
                    mfe_R=_finite(payload.get("mfe_R")),
                    mae_R=_finite(payload.get("mae_R")),
                    hit_0_5R=(
                        None
                        if payload.get("hit_0_5R") is None
                        else bool(payload.get("hit_0_5R"))
                    ),
                    hit_1R=(
                        None if payload.get("hit_1R") is None else bool(payload.get("hit_1R"))
                    ),
                    hit_2R=(
                        None if payload.get("hit_2R") is None else bool(payload.get("hit_2R"))
                    ),
                    draw_delivered=(
                        None
                        if payload.get("draw_id") is None
                        else time_to_draw is not None
                    ),
                    time_to_draw_real_1m_bars=(
                        None if time_to_draw is None else int(time_to_draw)
                    ),
                    filled=bool(payload.get("filled", False)),
                    expired=resolution in _SHADOW_EXPIRED_RESOLUTIONS,
                    terminal_reason=(
                        case.terminal_reason
                        or _identity(payload.get("entry_episode_terminal_reason"))
                    ),
                    resolution=resolution,
                    censored=bool(payload.get("censored", False)),
                    source_shadow_candidate_id=_identity(payload.get("candidate_id")),
                )
            self._outcome_rows.append(record)
            self._outcome_count += 1
        self._closed = True


def _parse_json_field(row: Mapping[str, Any], name: str) -> Any:
    raw = row.get(name)
    if not isinstance(raw, str):
        raise ValueError(f"causal case input {name} is not JSON text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"causal case input {name} is invalid JSON") from exc


def _validate_observed_clocks(value: Any, *, asof: pd.Timestamp, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            key_text = str(key)
            named_clock = bool(
                key_text.endswith("_at")
                or key_text
                in {
                    "asof",
                    "timestamp",
                    "datetime",
                    "cutoff",
                    "start_at",
                    "end_at",
                }
                or key_text in _FUTURE_CLOCK_KEYS
            )
            # Only a scalar, explicitly named planning deadline may be after
            # asof.  Derived/result clocks such as ``deadline_result_at`` are
            # observations and remain subject to the normal causal clock gate.
            if named_clock and item is not None:
                if not isinstance(item, str):
                    raise ValueError(
                        f"causal case named JSON clock is not ISO text: {child}"
                    )
                try:
                    clock = pd.Timestamp(item)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"causal case named JSON clock is invalid: {child}"
                    ) from exc
                if clock.tzinfo is None:
                    raise ValueError(
                        f"causal case named JSON clock is timezone-naive: {child}"
                    )
                if key_text not in _FUTURE_CLOCK_KEYS and clock > asof:
                    raise ValueError(f"causal case input clock exceeds asof: {child}")
            _validate_observed_clocks(item, asof=asof, path=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_observed_clocks(item, asof=asof, path=f"{path}[{index}]")


def validate_case_input_row(row: Mapping[str, Any]) -> None:
    """Fail closed on schema drift, future clocks, or non-prefix normalization."""

    missing = sorted(set(CAUSAL_CASE_INPUT_FIELD_TYPES) - set(row))
    extra = sorted(set(row) - set(CAUSAL_CASE_INPUT_FIELD_TYPES))
    if missing or extra:
        raise ValueError(f"causal case input schema differs: missing={missing}, extra={extra}")
    forbidden = sorted(set(row).intersection(_OUTCOME_ONLY_FIELDS))
    if forbidden:
        raise ValueError(f"future outcome fields entered case input: {forbidden}")
    asof = _clock(row["asof"], name="causal_case.input.asof")
    if _clock(row["decision_at"], name="causal_case.input.decision_at") != asof:
        raise ValueError("causal case decision clock differs from asof")
    if _clock(row["stage_observed_at"], name="causal_case.stage_observed_at") > asof:
        raise ValueError("causal case stage is future-dated")
    if _clock(row["admitted_at"], name="causal_case.admitted_at") > asof:
        raise ValueError("causal case admission is future-dated")
    if _clock(row["normalization_cutoff_at"], name="causal_case.normalization") >= asof:
        raise ValueError("normalization cutoff must be strictly before asof")
    planned_deadline_at = row["planned_deadline_at"]
    if planned_deadline_at is not None:
        _clock(planned_deadline_at, name="causal_case.planned_deadline_at")
    selected_trigger_at = row["selected_trigger_at"]
    if selected_trigger_at is not None and _clock(
        selected_trigger_at,
        name="causal_case.selected_trigger_at",
    ) > asof:
        raise ValueError("causal case selected trigger is future-dated")
    if row["normalization_policy"] != "bars_with_end_strictly_before_decision_asof":
        raise ValueError("causal case normalization policy changed")
    source_sha256 = str(row["source_sha256"]).lower()
    if (
        len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
        or not Path(str(row["source_path"])).is_absolute()
    ):
        raise ValueError("causal case canonical source identity is invalid")
    stage = str(row["revision_stage"])
    if stage not in _REVISION_STAGES:
        raise ValueError("causal case revision stage is unsupported")
    if not str(row["stage_identity"]).startswith(f"{stage}:"):
        raise ValueError("causal case stage identity disagrees with its stage")
    revision_index = row["revision_index"]
    if (
        isinstance(revision_index, bool)
        or not isinstance(revision_index, int)
        or revision_index < 0
    ):
        raise ValueError("causal case revision index is invalid")
    if str(row["observable_regime"]) not in _OBSERVABLE_REGIMES:
        raise ValueError("causal case observable regime is unsupported")
    if not str(row["mechanism_label"]):
        raise ValueError("causal case mechanism label is missing")
    for field_name in _INPUT_JSON_FIELDS:
        payload = _parse_json_field(row, field_name)
        _validate_observed_clocks(payload, asof=asof, path=field_name)
    transition = _parse_json_field(row, "observation_transition_json")
    scene_delta = _parse_json_field(row, "scene_graph_delta_json")
    transition_coverage = transition.get("coverage")
    scene_coverage = scene_delta.get("coverage")
    if (
        not isinstance(transition_coverage, Mapping)
        or transition_coverage != scene_coverage
        or type(transition_coverage.get("complete")) is not bool
    ):
        raise ValueError("causal case transition coverage contract is invalid")
    coverage_start = _clock(
        transition_coverage["coverage_start_at"],
        name="causal_case.transition_coverage.start_at",
    )
    coverage_end = _clock(
        transition_coverage["coverage_end_at"],
        name="causal_case.transition_coverage.end_at",
    )
    if coverage_start > coverage_end or coverage_end != asof:
        raise ValueError("causal case transition coverage clock is invalid")
    updates = transition.get("updates")
    scene_updates = scene_delta.get("updates")
    if (
        not isinstance(updates, list)
        or not isinstance(scene_updates, list)
        or len(updates) != len(scene_updates)
    ):
        raise ValueError("causal case transition coverage update count is invalid")
    raw_observed_update_count = transition_coverage.get("observed_update_count")
    raw_eventful_update_count = transition_coverage.get("eventful_update_count")
    if (
        isinstance(raw_observed_update_count, bool)
        or not isinstance(raw_observed_update_count, int)
        or raw_observed_update_count < 0
        or isinstance(raw_eventful_update_count, bool)
        or not isinstance(raw_eventful_update_count, int)
        or raw_eventful_update_count < 0
    ):
        raise ValueError("causal case transition coverage update count is invalid")
    observed_update_count = raw_observed_update_count
    eventful_update_count = raw_eventful_update_count
    if (
        eventful_update_count != len(updates)
        or observed_update_count < eventful_update_count
        or type(transition_coverage.get("all_typed_deltas_available")) is not bool
        or type(transition_coverage.get("gap_free")) is not bool
    ):
        raise ValueError("causal case transition coverage update count is invalid")
    start_replay_update_ordinal = transition_coverage.get(
        "coverage_start_replay_update_ordinal"
    )
    end_replay_update_ordinal = transition_coverage.get(
        "coverage_end_replay_update_ordinal"
    )
    last_replay_update_ordinal = transition_coverage.get(
        "last_observed_replay_update_ordinal"
    )
    for name, ordinal in (
        ("start", start_replay_update_ordinal),
        ("end", end_replay_update_ordinal),
    ):
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError(
                f"causal case transition coverage {name} replay ordinal is invalid"
            )
    if end_replay_update_ordinal < start_replay_update_ordinal:
        raise ValueError("causal case transition coverage replay interval is invalid")
    update_clocks = tuple(
        _clock(update["asof"], name="causal_case.transition_update.asof")
        for update in updates
        if isinstance(update, Mapping)
    )
    if len(update_clocks) != len(updates) or any(
        left >= right for left, right in zip(update_clocks, update_clocks[1:])
    ):
        raise ValueError("causal case transition update clocks are invalid")
    update_ordinals = tuple(
        update.get("replay_update_ordinal")
        for update in updates
        if isinstance(update, Mapping)
    )
    scene_update_ordinals = tuple(
        update.get("replay_update_ordinal")
        for update in scene_updates
        if isinstance(update, Mapping)
    )
    scene_update_clocks = tuple(
        _clock(update.get("asof"), name="causal_case.scene_transition_update.asof")
        for update in scene_updates
        if isinstance(update, Mapping)
    )
    if (
        len(update_ordinals) != len(updates)
        or update_ordinals != scene_update_ordinals
        or update_clocks != scene_update_clocks
        or any(
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            for ordinal in update_ordinals
        )
        or any(
            left >= right
            for left, right in zip(update_ordinals, update_ordinals[1:])
        )
    ):
        raise ValueError("causal case transition update ordinals are invalid")
    raw_start_exclusive = transition_coverage.get("coverage_start_exclusive")
    if type(raw_start_exclusive) is not bool:
        raise ValueError("causal case transition coverage boundary type is invalid")
    start_exclusive = raw_start_exclusive
    expected_observed_update_count = (
        end_replay_update_ordinal
        - start_replay_update_ordinal
        + int(not start_exclusive)
    )
    if observed_update_count != expected_observed_update_count:
        raise ValueError("causal case transition replay coverage is incomplete")
    if any(
        ordinal <= start_replay_update_ordinal
        if start_exclusive
        else ordinal < start_replay_update_ordinal
        for ordinal in update_ordinals
    ) or any(ordinal > end_replay_update_ordinal for ordinal in update_ordinals):
        raise ValueError("causal case transition update ordinal is outside coverage")
    if any(
        clock <= coverage_start if start_exclusive else clock < coverage_start
        for clock in update_clocks
    ) or any(clock > coverage_end for clock in update_clocks):
        raise ValueError("causal case transition update is outside coverage")
    raw_last_observed = transition_coverage.get("last_observed_update_at")
    last_observed = (
        None
        if raw_last_observed is None
        else _clock(
            raw_last_observed,
            name="causal_case.transition_coverage.last_observed_update_at",
        )
    )
    if (observed_update_count == 0) != (last_observed is None):
        raise ValueError("causal case transition coverage last clock is invalid")
    if observed_update_count == 0:
        if last_replay_update_ordinal is not None:
            raise ValueError("causal case transition coverage last ordinal is invalid")
    elif (
        isinstance(last_replay_update_ordinal, bool)
        or not isinstance(last_replay_update_ordinal, int)
        or last_replay_update_ordinal != end_replay_update_ordinal
    ):
        raise ValueError("causal case transition coverage last ordinal is invalid")
    if last_observed is not None and (
        last_observed < coverage_start or last_observed > coverage_end
    ):
        raise ValueError("causal case transition coverage last clock is invalid")
    expected_complete = bool(
        transition_coverage["all_typed_deltas_available"]
        and transition_coverage["gap_free"]
        and (
            (observed_update_count > 0 and last_observed == coverage_end)
            or (
                observed_update_count == 0
                and start_exclusive
                and coverage_start == coverage_end
            )
        )
    )
    if transition_coverage["complete"] != expected_complete:
        raise ValueError("causal case transition coverage completeness is inconsistent")
    if bool(transition.get("typed_transition_delta_available")) != bool(
        transition_coverage["all_typed_deltas_available"]
    ):
        raise ValueError("causal case typed-delta coverage flag is inconsistent")
    aggregate_scene_ids = {
        name: _validate_canonical_identity_list(
            scene_delta.get(name),
            name=f"aggregate Scene {name}",
        )
        for name in _SCENE_SET_ID_FIELDS
    }
    aggregate_update_scene_ids: dict[str, set[str]] = {
        name: set() for name in _SCENE_SET_ID_FIELDS
    }
    _validate_canonical_identity_list(
        _parse_json_field(row, "added_event_ids_json"),
        name="aggregate added event ID summary",
    )
    _validate_canonical_identity_list(
        _parse_json_field(row, "invalidated_event_ids_json"),
        name="aggregate invalidated event ID summary",
    )
    relation_descriptors = scene_delta.get("relation_descriptors")
    relation_descriptors_complete = scene_delta.get(
        "relation_descriptors_complete"
    )
    if not isinstance(relation_descriptors, list) or type(
        relation_descriptors_complete
    ) is not bool:
        raise ValueError("causal case Scene relation descriptor contract is invalid")
    flattened_relation_descriptors: list[Mapping[str, Any]] = []
    all_relation_descriptors_complete = True
    for scene_update, scene_update_clock in zip(
        scene_updates,
        scene_update_clocks,
        strict=True,
    ):
        scene_update_ids = {
            name: _validate_canonical_identity_list(
                scene_update.get(name),
                name=f"Scene update {name}",
            )
            for name in _SCENE_SET_ID_FIELDS
        }
        for name, values in scene_update_ids.items():
            aggregate_update_scene_ids[name].update(values)
        descriptors = scene_update.get("relation_descriptors")
        descriptors_complete = scene_update.get("relation_descriptors_complete")
        if not isinstance(descriptors, list) or type(descriptors_complete) is not bool:
            raise ValueError("causal case Scene relation update contract is invalid")
        added_edge_ids = scene_update_ids["added_edge_ids"]
        revised_edge_ids = scene_update_ids["revised_edge_ids"]
        changed_edge_ids = _ordered_scene_changed_edge_ids(
            added_edge_ids,
            revised_edge_ids,
        )
        descriptor_ids: list[str] = []
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise ValueError("causal case Scene relation descriptor is invalid")
            edge_id = descriptor.get("edge_id")
            if not isinstance(edge_id, str) or not edge_id:
                raise ValueError("causal case Scene relation identity is invalid")
            descriptor_ids.append(edge_id)
            if not all(
                isinstance(descriptor.get(name), str) and descriptor.get(name)
                for name in ("relation", "lifecycle", "observed_at")
            ):
                raise ValueError("causal case Scene relation semantics are invalid")
            if _clock(
                descriptor["observed_at"],
                name="causal_case.scene_relation.observed_at",
            ) > scene_update_clock:
                raise ValueError("causal case Scene relation is future-dated")
            for endpoint_name in ("source", "target"):
                endpoint = descriptor.get(endpoint_name)
                if not isinstance(endpoint, Mapping) or not all(
                    isinstance(endpoint.get(name), str) and endpoint.get(name)
                    for name in (
                        "node_id",
                        "kind",
                        "role",
                        "timeframe",
                        "structural_scale",
                        "lifecycle",
                    )
                ):
                    raise ValueError("causal case Scene endpoint descriptor is invalid")
        if len(descriptor_ids) != len(set(descriptor_ids)):
            raise ValueError("causal case Scene relation descriptor is duplicated")
        added_descriptors = sorted(
            (
                descriptor
                for descriptor in descriptors
                if descriptor["edge_id"] in set(added_edge_ids)
            ),
            key=_scene_relation_descriptor_sort_key,
        )
        revised_descriptors = sorted(
            (
                descriptor
                for descriptor in descriptors
                if descriptor["edge_id"] in set(revised_edge_ids)
            ),
            key=_scene_relation_descriptor_sort_key,
        )
        expected_descriptors = [*added_descriptors, *revised_descriptors]
        if descriptors != expected_descriptors:
            raise ValueError(
                "causal case Scene relation descriptors are not canonical"
            )
        if descriptors_complete and set(descriptor_ids) != set(changed_edge_ids):
            raise ValueError("causal case Scene relation descriptor coverage is incomplete")
        if not descriptors_complete and not set(descriptor_ids).issubset(
            changed_edge_ids
        ):
            raise ValueError("causal case Scene relation descriptor is unbound")
        flattened_relation_descriptors.extend(descriptors)
        all_relation_descriptors_complete = bool(
            all_relation_descriptors_complete and descriptors_complete
        )
    for name, aggregate_values in aggregate_scene_ids.items():
        expected_aggregate = tuple(sorted(aggregate_update_scene_ids[name]))
        if aggregate_values != expected_aggregate:
            raise ValueError(
                f"causal case aggregate Scene {name} differs from update union"
            )
    if (
        relation_descriptors != flattened_relation_descriptors
        or relation_descriptors_complete != all_relation_descriptors_complete
    ):
        raise ValueError("causal case aggregate Scene relation descriptors changed")
    prefixes = _parse_json_field(row, "prefix_refs_json")
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("causal case has no OHLCV prefix references")
    expected_timeframes = {
        Timeframe.H4.value,
        Timeframe.H1.value,
        Timeframe.M15.value,
        Timeframe.M5.value,
        Timeframe.M1.value,
    }
    observed_timeframes: set[str] = set()
    replay_bounds: set[tuple[int, int]] = set()
    for prefix in prefixes:
        if not isinstance(prefix, Mapping):
            raise ValueError("causal case OHLCV prefix is invalid")
        timeframe = str(prefix.get("timeframe", ""))
        if timeframe in observed_timeframes:
            raise ValueError("causal case has a duplicate timeframe prefix")
        observed_timeframes.add(timeframe)
        start_at = _clock(prefix["start_at"], name="causal_case.prefix.start_at")
        end_at = _clock(prefix["end_at"], name="causal_case.prefix.end_at")
        prefix_asof = _clock(prefix["asof"], name="causal_case.prefix.asof")
        if start_at > end_at or end_at > asof or prefix_asof != asof:
            raise ValueError("causal case OHLCV prefix reveals future data")
        frame_start = prefix.get("frame_row_start")
        frame_end = prefix.get("frame_row_end_exclusive")
        replay_start = prefix.get("replay_view_1m_row_start")
        replay_end = prefix.get("replay_view_1m_row_end_exclusive")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (frame_start, frame_end, replay_start, replay_end)
        ):
            raise ValueError("causal case prefix row coordinates are not integers")
        if frame_start != 0 or frame_end <= frame_start:
            raise ValueError("causal case frame prefix row bounds are invalid")
        if replay_start < 0 or replay_end <= replay_start:
            raise ValueError("causal case source prefix row bounds are invalid")
        replay_bounds.add((replay_start, replay_end))
        if (
            prefix.get("boundary_semantics")
            != "[start_at,end_at]_completed_prefix"
            or "never_use_frame_row_as_global_index"
            not in str(prefix.get("reload_rule", ""))
        ):
            raise ValueError("causal case prefix coordinate semantics changed")
    if observed_timeframes != expected_timeframes:
        raise ValueError("causal case must bind exactly five timeframe prefixes")
    if len(replay_bounds) != 1:
        raise ValueError("causal case timeframe prefixes disagree on replay view")
    expected_fingerprint = case_input_fingerprint(row)
    if row["input_fingerprint"] and row["input_fingerprint"] != expected_fingerprint:
        raise ValueError("causal case input fingerprint is invalid")


def validate_episode_disjoint_splits(
    split_rows: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    """Assert that no EntryEpisode is shared by train/validation/test."""

    owners: dict[tuple[str, str], str] = {}
    for split, rows in split_rows.items():
        for row in rows:
            key = (str(row["market_epoch_id"]), str(row["entry_episode_id"]))
            prior = owners.setdefault(key, str(split))
            if prior != str(split):
                raise ValueError(
                    f"EntryEpisode is shared by data splits: {key} in {prior} and {split}"
                )


def case_input_to_representation_mapping(
    row: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the only supported input mapping for representation learning.

    The adapter accepts an exact case-input row and never accepts or returns a
    future-label field.  Model code can therefore consume prefixes/events
    without knowing the recorder or joining the outcome stream.
    """

    validate_case_input_row(row)
    if set(row).intersection(_OUTCOME_ONLY_FIELDS):
        raise ValueError("future outcome mapping cannot become representation input")
    return {
        "case_id": str(row["case_id"]),
        "revision_id": str(row["revision_id"]),
        "revision_index": int(row["revision_index"]),
        "revision_stage": str(row["revision_stage"]),
        "stage_identity": str(row["stage_identity"]),
        "split_role": str(row["split_role"]),
        "market_epoch_id": str(row["market_epoch_id"]),
        "context_thesis_id": str(row["context_thesis_id"]),
        "entry_episode_id": str(row["entry_episode_id"]),
        "symbol": str(row["symbol"]),
        "instrument_id": int(row["instrument_id"]),
        "direction": str(row["direction"]),
        "regime": str(row["observable_regime"]),
        "observable_regime": str(row["observable_regime"]),
        "mechanism_label": str(row["mechanism_label"]),
        "asof": row["asof"],
        "decision_at": row["decision_at"],
        "canonical_source_id": str(row["source_sha256"]),
        "source_path": str(row["source_path"]),
        "prefixes": _parse_json_field(row, "prefix_refs_json"),
        "normalization_cutoff_at": row["normalization_cutoff_at"],
        "normalization_policy": str(row["normalization_policy"]),
        "events": {
            "added_event_ids": _parse_json_field(row, "added_event_ids_json"),
            "invalidated_event_ids": _parse_json_field(
                row,
                "invalidated_event_ids_json",
            ),
            "observation_transition": _parse_json_field(
                row,
                "observation_transition_json",
            ),
        },
        "graph": _parse_json_field(row, "scene_graph_delta_json"),
        "context_thesis": _parse_json_field(row, "context_thesis_json"),
        "entry_episode": _parse_json_field(row, "entry_episode_json"),
        "brain_response": _parse_json_field(row, "brain_response_json"),
        "authority": _parse_json_field(row, "authority_json"),
        "scale_relations": _parse_json_field(row, "scale_relations_json"),
        "draw": _parse_json_field(row, "draw_json"),
        "blockers": _parse_json_field(row, "blockers_json"),
        "ambiguities": _parse_json_field(row, "ambiguities_json"),
        "supporting_evidence": _parse_json_field(
            row,
            "supporting_evidence_json",
        ),
        "opposing_evidence": _parse_json_field(row, "opposing_evidence_json"),
        "model_versions": _parse_json_field(row, "model_versions_json"),
    }


def validate_case_library_rows(
    input_rows: Iterable[Mapping[str, Any]],
    outcome_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Audit key/cardinality/separation invariants for a materialized library."""

    inputs = tuple(input_rows)
    outcomes = tuple(outcome_rows)
    revision_ids: set[str] = set()
    stages: set[tuple[str, str]] = set()
    immutable_stage_names: set[tuple[str, str]] = set()
    fingerprints: set[str] = set()
    cases: dict[str, tuple[str, str, str]] = {}
    episode_cases: dict[tuple[str, str], str] = {}
    revisions_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in inputs:
        validate_case_input_row(row)
        revision_id = str(row["revision_id"])
        stage_key = (str(row["case_id"]), str(row["stage_identity"]))
        stage_name_key = (str(row["case_id"]), str(row["revision_stage"]))
        fingerprint = str(row["input_fingerprint"])
        if not all(
            isinstance(row[name], str) and bool(row[name])
            for name in (
                "revision_id",
                "case_id",
                "stage_identity",
                "input_fingerprint",
                "market_epoch_id",
                "context_thesis_id",
                "entry_episode_id",
            )
        ):
            raise ValueError("causal case materialized input identity is missing")
        if (
            str(row["revision_stage"]) != "context_changed"
            and stage_name_key in immutable_stage_names
        ):
            raise ValueError(
                "causal case immutable revision stage was materialized more than once"
            )
        if revision_id in revision_ids or stage_key in stages or fingerprint in fingerprints:
            raise ValueError("causal case input revision/stage/features are duplicated")
        revision_ids.add(revision_id)
        stages.add(stage_key)
        immutable_stage_names.add(stage_name_key)
        fingerprints.add(fingerprint)
        identity = (
            str(row["market_epoch_id"]),
            str(row["context_thesis_id"]),
            str(row["entry_episode_id"]),
        )
        if str(row["case_id"]) != _case_id(*identity):
            raise ValueError("causal case deterministic identity is invalid")
        expected_revision_id = "case-revision:" + hashlib.sha256(
            (
                f"{row['case_id']}|{row['stage_identity']}|"
                f"{row['input_fingerprint']}"
            ).encode("utf-8")
        ).hexdigest()[:32]
        if revision_id != expected_revision_id:
            raise ValueError("causal case deterministic revision identity is invalid")
        prior = cases.setdefault(str(row["case_id"]), identity)
        if prior != identity:
            raise ValueError("causal case identity changed across revisions")
        episode_key = (identity[0], identity[2])
        prior_case_id = episode_cases.setdefault(episode_key, str(row["case_id"]))
        if prior_case_id != str(row["case_id"]):
            raise ValueError("EntryEpisode was materialized as multiple cases")
        revisions_by_case.setdefault(str(row["case_id"]), []).append(row)
    for case_id, revisions in revisions_by_case.items():
        ordered = sorted(revisions, key=lambda item: int(item["revision_index"]))
        previous: Mapping[str, Any] | None = None
        for expected_index, row in enumerate(ordered):
            if int(row["revision_index"]) != expected_index:
                raise ValueError("causal case revision indices are not contiguous")
            transition = _parse_json_field(row, "observation_transition_json")
            coverage = transition["coverage"]
            if coverage.get("gap_free") is not True:
                raise ValueError("causal case finalized coverage is not gap-free")
            row_asof = _clock(row["asof"], name="causal_case.library.asof")
            if previous is None:
                if (
                    _clock(
                        coverage["coverage_start_at"],
                        name="causal_case.library.coverage_start_at",
                    )
                    != _clock(
                        row["admitted_at"],
                        name="causal_case.library.admitted_at",
                    )
                    or coverage["coverage_start_exclusive"] is not False
                ):
                    raise ValueError("causal case admission coverage is incomplete")
            else:
                previous_transition = _parse_json_field(
                    previous,
                    "observation_transition_json",
                )
                previous_coverage = previous_transition["coverage"]
                if (
                    _clock(
                        coverage["coverage_start_at"],
                        name="causal_case.library.coverage_start_at",
                    )
                    != _clock(
                        previous["asof"],
                        name="causal_case.library.previous_asof",
                    )
                    or int(coverage["coverage_start_replay_update_ordinal"])
                    != int(
                        previous_coverage[
                            "coverage_end_replay_update_ordinal"
                        ]
                    )
                    or coverage["coverage_start_exclusive"] is not True
                    or row_asof
                    < _clock(
                        previous["asof"],
                        name="causal_case.library.previous_asof",
                    )
                ):
                    raise ValueError("causal case revision coverage boundary changed")
            previous = row
    outcome_ids: set[str] = set()
    outcome_cases: set[str] = set()
    last_input_asof_by_case = {
        case_id: max(
            _clock(row["asof"], name="causal_case.library.input_asof")
            for row in revisions
        )
        for case_id, revisions in revisions_by_case.items()
    }
    for row in outcomes:
        missing = set(CAUSAL_CASE_OUTCOME_FIELD_TYPES) - set(row)
        extra = set(row) - set(CAUSAL_CASE_OUTCOME_FIELD_TYPES)
        if missing or extra:
            raise ValueError("causal case future outcome schema differs")
        outcome_id = str(row["outcome_id"])
        case_id = str(row["case_id"])
        if not all(
            isinstance(row[name], str) and bool(row[name])
            for name in (
                "outcome_id",
                "case_id",
                "market_epoch_id",
                "context_thesis_id",
                "entry_episode_id",
            )
        ):
            raise ValueError("causal case materialized outcome identity is missing")
        if outcome_id in outcome_ids or case_id in outcome_cases:
            raise ValueError("causal case future outcome is duplicated")
        if case_id not in cases:
            raise ValueError("causal case future outcome has no input case")
        identity = cases[case_id]
        if identity != (
            str(row["market_epoch_id"]),
            str(row["context_thesis_id"]),
            str(row["entry_episode_id"]),
        ):
            raise ValueError("causal case future outcome identity changed")
        expected_outcome_id = f"case-outcome:{case_id.split(':', 1)[-1]}"
        if outcome_id != expected_outcome_id:
            raise ValueError("causal case deterministic outcome identity is invalid")
        resolved_at = _clock(
            row["resolved_at"],
            name="causal_case.outcome.resolved_at",
        )
        if resolved_at < last_input_asof_by_case[case_id]:
            raise ValueError("causal case future outcome precedes its last input")
        for name in (
            "deadline_first",
            "same_bar_collision",
            "filled",
            "expired",
            "censored",
        ):
            if type(row[name]) is not bool:
                raise ValueError(f"causal case future outcome {name} is not boolean")
        for name in (
            "target_first",
            "invalidation_first",
            "hit_0_5R",
            "hit_1R",
            "hit_2R",
            "draw_delivered",
        ):
            if row[name] is not None and type(row[name]) is not bool:
                raise ValueError(
                    f"causal case future outcome {name} is not nullable boolean"
                )
        for name in ("mfe_points", "mae_points", "mfe_R", "mae_R"):
            value = row[name]
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"causal case future outcome {name} is not finite"
                )
        time_to_draw = row["time_to_draw_real_1m_bars"]
        if time_to_draw is not None and (
            isinstance(time_to_draw, bool)
            or not isinstance(time_to_draw, int)
            or time_to_draw < 0
        ):
            raise ValueError("causal case future draw time is invalid")
        first_event = str(row["first_event"])
        if not isinstance(row["resolution"], str) or not row["resolution"]:
            raise ValueError("causal case future resolution is missing")
        if first_event not in {
            "target",
            "invalidation",
            "deadline",
            "episode_terminal",
            "right_censored",
            "unresolved",
        }:
            raise ValueError("causal case future first-event kind is invalid")
        source_shadow_candidate_id = row["source_shadow_candidate_id"]
        if source_shadow_candidate_id is not None:
            if (
                not isinstance(source_shadow_candidate_id, str)
                or not source_shadow_candidate_id
            ):
                raise ValueError(
                    "causal case future Shadow candidate identity is invalid"
                )
            expected_first_event = _project_shadow_first_event(
                resolution=row["resolution"],
                filled=row["filled"],
                censored=row["censored"],
                target_first=row["target_first"],
                invalidation_first=row["invalidation_first"],
                same_bar_collision=row["same_bar_collision"],
            )
            if first_event != expected_first_event:
                raise ValueError(
                    "causal case future first event disagrees with its Shadow "
                    "resolution family"
                )
        if bool(row["deadline_first"]) != (first_event == "deadline"):
            raise ValueError("causal case future deadline-first flag is inconsistent")
        pairwise_first = (row["target_first"], row["invalidation_first"])
        if pairwise_first not in {
            (None, None),
            (True, False),
            (False, True),
        }:
            raise ValueError(
                "causal case future pairwise first-event flags are inconsistent"
            )
        if not row["filled"] and pairwise_first != (None, None):
            raise ValueError(
                "causal case unfilled future outcome has pairwise first-event flags"
            )
        if first_event == "target" and (
            pairwise_first != (True, False) or not row["filled"]
        ):
            raise ValueError("causal case future target-first flags are inconsistent")
        if first_event == "invalidation":
            expected_pair = (False, True) if row["filled"] else (None, None)
            if pairwise_first != expected_pair:
                raise ValueError(
                    "causal case future invalidation-first flags are inconsistent"
                )
        if first_event not in {"target", "invalidation"} and pairwise_first != (
            None,
            None,
        ):
            raise ValueError(
                "causal case non-path future event has pairwise first-event flags"
            )
        if bool(row["censored"]) != (first_event == "right_censored"):
            raise ValueError("causal case future censoring flag is inconsistent")
        expected_expired = str(row["resolution"]) in _SHADOW_EXPIRED_RESOLUTIONS
        if row["expired"] is not expected_expired:
            raise ValueError("causal case future expiry semantics conflict")
        if first_event == "deadline" and row["expired"] is not True:
            raise ValueError("causal case future deadline did not expire")
        if row["same_bar_collision"] and not (
            first_event == "invalidation"
            and pairwise_first == (False, True)
            and row["filled"]
            and not row["censored"]
            and str(row["resolution"]) == "same_bar_invalidation_priority"
        ):
            raise ValueError("causal case future collision semantics conflict")
        if (
            str(row["resolution"]) == "same_bar_invalidation_priority"
            and row["same_bar_collision"] is not True
        ):
            raise ValueError("causal case future collision flag is missing")
        hit_half = row["hit_0_5R"]
        hit_one = row["hit_1R"]
        hit_two = row["hit_2R"]
        if (hit_two is True and hit_one is not True) or (
            hit_one is True and hit_half is not True
        ):
            raise ValueError("causal case future R-hit ladder is inconsistent")
        if row["draw_delivered"] is False and time_to_draw is not None:
            raise ValueError("causal case future draw delivery clock is inconsistent")
        if row["draw_delivered"] is True and time_to_draw is None:
            raise ValueError("causal case future draw delivery clock is missing")
        if row["draw_delivered"] is None and time_to_draw is not None:
            raise ValueError("causal case future draw delivery identity is missing")
        outcome_ids.add(outcome_id)
        outcome_cases.add(case_id)
    if outcome_cases != set(cases):
        missing = sorted(set(cases) - outcome_cases)
        raise ValueError(
            "causal case future outcome cardinality is incomplete: "
            f"missing={missing}"
        )


def write_causal_case_library_manifest(
    destination: str | Path,
    *,
    input_stream_manifest: str | Path,
    outcome_stream_manifest: str | Path,
    run_manifest: str | Path,
) -> Path:
    """Bind the two separate stream manifests without merging their schemas."""

    root = Path(destination)
    root_resolved = root.resolve()

    def bound_path(value: str | Path) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("causal case manifest binding must be relative")
        resolved = (root / relative).resolve()
        if resolved.parent != root_resolved:
            raise ValueError("causal case manifest binding escaped its run root")
        return resolved

    inputs = bound_path(input_stream_manifest)
    outcomes = bound_path(outcome_stream_manifest)
    run = bound_path(run_manifest)
    for path in (inputs, outcomes, run):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"causal case manifest binding is missing: {path}")
    input_payload = json.loads(inputs.read_text(encoding="utf-8"))
    outcome_payload = json.loads(outcomes.read_text(encoding="utf-8"))
    try:
        run_bytes = run.read_bytes()
        run_payload = json.loads(run_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("causal case run manifest is not valid JSON") from exc
    if not isinstance(run_payload, Mapping):
        raise ValueError("causal case run manifest must be an object")
    if run_bytes != canonical_json(run_payload):
        raise ValueError("causal case run manifest is not canonical JSON")
    validated_run_sha256 = hashlib.sha256(run_bytes).hexdigest()
    run_output = run_payload.get("output")
    stream_families = (
        None
        if not isinstance(run_output, Mapping)
        else run_output.get("stream_families")
    )
    required_case_streams = {
        "causal_case_input_shards",
        "causal_case_outcome_shards",
    }
    if (
        run_payload.get("schema_version") != 1
        or run_payload.get("runner") != "continuous_replay"
        or run_payload.get("causal_case_identity")
        != expected_causal_case_run_identity()
        or not isinstance(run_output, Mapping)
        or run_output.get("causal_case_library") is not True
        or not isinstance(stream_families, list)
        or any(not isinstance(value, str) for value in stream_families)
        or len(stream_families) != len(set(stream_families))
        or not required_case_streams.issubset(set(stream_families))
    ):
        raise ValueError("causal case run manifest identity is incompatible")
    if input_payload.get("field_types") != dict(CAUSAL_CASE_INPUT_FIELD_TYPES):
        raise ValueError("causal case input stream manifest schema changed")
    if outcome_payload.get("field_types") != dict(CAUSAL_CASE_OUTCOME_FIELD_TYPES):
        raise ValueError("causal case outcome stream manifest schema changed")
    if input_payload.get("status") != "complete" or outcome_payload.get("status") != "complete":
        raise ValueError("causal case stream manifest is incomplete")
    if input_payload.get("stream") != "causal_case_input_shards":
        raise ValueError("causal case input stream identity changed")
    if outcome_payload.get("stream") != "causal_case_outcome_shards":
        raise ValueError("causal case outcome stream identity changed")
    expected_run_binding = Path(run_manifest).name
    if input_payload.get("bindings", {}).get("run_manifest") != expected_run_binding:
        raise ValueError("causal case input stream run binding changed")
    if outcome_payload.get("bindings", {}).get("run_manifest") != expected_run_binding:
        raise ValueError("causal case outcome stream run binding changed")

    def materialized_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        import pyarrow.parquet as pq

        rows: list[Mapping[str, Any]] = []
        expected_index = 0
        total = 0
        for shard in payload.get("shards", ()):  # sparse episode revisions only
            if int(shard.get("index", -1)) != expected_index:
                raise ValueError("causal case stream shard order is not contiguous")
            relative = Path(str(shard.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("causal case shard binding escaped its run root")
            path = (root / relative).resolve()
            if root_resolved not in path.parents:
                raise ValueError("causal case shard binding escaped its run root")
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"causal case stream shard is missing: {path}")
            if sha256_file(path) != str(shard.get("sha256", "")):
                raise ValueError("causal case stream shard hash changed")
            values = pq.read_table(path).to_pylist()
            if len(values) != int(shard.get("rows", -1)):
                raise ValueError("causal case stream shard row count changed")
            rows.extend(values)
            total += len(values)
            expected_index += 1
        if total != int(payload.get("rows", -1)):
            raise ValueError("causal case stream rows are not conserved")
        return rows

    # Re-open the just-finalized sparse streams through Arrow.  This validates
    # the serialized representation itself (not merely recorder memory),
    # including one outcome per case and complete input/outcome separation.
    validate_case_library_rows(
        materialized_rows(input_payload),
        materialized_rows(outcome_payload),
    )
    payload = {
        "format_version": 1,
        "artifact": "entry_episode_causal_case_library",
        "status": "complete",
        "recorder_schema_version": CAUSAL_CASE_RECORDER_SCHEMA_VERSION,
        "protocol": dict(CAUSAL_CASE_PROTOCOL),
        "grain": "entry_episode",
        "input_stream": {
            "manifest": str(Path(input_stream_manifest)),
            "manifest_sha256": sha256_file(inputs),
            "rows": int(input_payload["rows"]),
        },
        "future_outcome_stream": {
            "manifest": str(Path(outcome_stream_manifest)),
            "manifest_sha256": sha256_file(outcomes),
            "rows": int(outcome_payload["rows"]),
        },
        "bindings": {
            "run_manifest": str(Path(run_manifest)),
            "run_manifest_sha256": validated_run_sha256,
        },
        "leakage_contract": {
            "outcome_fields_in_input_schema": False,
            "embedding_source": "input_stream_only",
            "episode_split_disjoint_required": True,
            "normalization_prefix_only": True,
        },
    }
    path = root / "causal_case_library.manifest.json"
    atomic_bytes(path, canonical_json(payload))
    return path


__all__ = [
    "CAUSAL_CASE_INPUT_FIELD_TYPES",
    "CAUSAL_CASE_OUTCOME_FIELD_TYPES",
    "CAUSAL_CASE_PROTOCOL",
    "CAUSAL_CASE_PROTOCOL_VERSION",
    "CAUSAL_CASE_RECORDER_SCHEMA_VERSION",
    "CausalCaseInputRecord",
    "CausalCaseOutcomeRecord",
    "CausalCaseRecorder",
    "case_input_to_representation_mapping",
    "case_input_fingerprint",
    "expected_causal_case_run_identity",
    "validate_case_input_row",
    "validate_case_library_rows",
    "validate_episode_disjoint_splits",
    "write_causal_case_library_manifest",
]
