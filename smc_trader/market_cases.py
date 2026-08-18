"""Input-only neutral MarketEpisode records for the existing replay.

The recorder consumes the ``NeutralEngineSnapshot`` already produced by the
single continuous replay loop.  It has no outcome, Shadow, action,
finalization, or playbook dependency.  One row represents one physical
MarketEpisode milestone at one observable clock; relation-only upstream
transitions update custody without producing rows.  Multiple physical facts
reached at the same clock are carried in the ordered
``transition_kinds_json`` list.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable

import pandas as pd

from .artifact_stream import canonical_json
from .model import (
    Direction,
    NEUTRAL_MARKET_STATE_SCHEMA_VERSION,
    Timeframe,
    aware_timestamp,
)
from .scene_graph import market_episode_id as neutral_market_episode_id


MARKET_CASE_RECORDER_SCHEMA_VERSION = 1
MARKET_CASE_PROTOCOL_VERSION = "market-episode-input-only-1.2.0"

# Run-level replay policy, intentionally separate from the recorder protocol:
# the row grain and validation contract are unchanged, while runner schema 7
# now binds how a large same-contract source discontinuity reaches the already
# supported data-gap epoch boundary.
MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY: Mapping[str, Any] = {
    "maximum_no_trade_gap_minutes": 5,
    "allow_same_contract_data_gap_reset": True,
    "data_gap_reset_anomaly": "data_gap_history_reset",
    "allow_cross_contract_data_gap_reset": False,
    "synthesize_over_cap_missing_minutes": False,
}

_TRANSITION_KIND_ORDER = {
    "episode_created": 0,
    "zone_registered": 1,
    "first_pullback": 2,
    "trigger": 3,
    "successful_pulse": 4,
    "terminal": 5,
}
_LIFECYCLE_ORDER = {
    "registered": 0,
    "pullback": 1,
    "triggered": 2,
    "terminal": 3,
}
_BOUNDARY_ANOMALIES = frozenset(
    {
        "contract_change_history_reset",
        "data_gap_history_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
_TRANSITION_COLLECTION_NAMES = (
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
_SCENE_ID_FIELDS = (
    "added_node_ids",
    "revised_node_ids",
    "added_edge_ids",
    "revised_edge_ids",
    "resolution_event_ids",
)
_TIMEFRAMES = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)
_RETIREMENT_REASONS = frozenset(
    {
        "upstream_compacted_after_success",
        "upstream_compacted_after_terminal",
    }
)
_FORBIDDEN_INPUT_KEY_MARKERS = (
    "future",
    "outcome",
    "future_profit",
    "profit",
    "pnl",
    "target_first",
    "invalidation_first",
    "deadline_first",
    "same_bar_collision",
    "mfe",
    "mae",
    "hit_0_5r",
    "hit_1r",
    "hit_2r",
    "draw_delivered",
)
_MICRO_BOS_REFERENCE_OUTCOMES = frozenset(
    {
        "aligned",
        "opposed",
        "simultaneous_unknown",
        "ambiguous_same_clock",
    }
)

MARKET_CASE_PROTOCOL: Mapping[str, Any] = {
    "protocol_version": MARKET_CASE_PROTOCOL_VERSION,
    "grain": "one_market_episode_physical_milestone_at_one_observable_clock",
    "runtime_source": "NeutralEngineSnapshot.neutral_market_state",
    "neutral_runtime_schema_version": NEUTRAL_MARKET_STATE_SCHEMA_VERSION,
    "replay_ownership": "same_existing_continuous_replay_no_second_loop",
    "input_only": True,
    "outcome_stream": False,
    "shadow_join": False,
    "playbook_dependency": False,
    "row_accumulation": False,
    "revision_rule": (
        "one_row_per_physical_episode_milestone;relation_only_transitions_"
        "update_custody_without_rows;multiple_same_clock_facts_are_one_"
        "ordered_transition_kinds_list;no_heartbeat"
    ),
    "memory": "active_episodes_lightweight_closed_identities_pending_rows_only",
    "ohlcv_storage": "same_clock_prefix_boundaries_only_no_copied_bars",
}


def expected_market_case_run_identity() -> Mapping[str, Any]:
    """Return the exact input-only identity a run manifest must bind."""

    return {
        "recorder_schema_version": MARKET_CASE_RECORDER_SCHEMA_VERSION,
        "protocol": dict(MARKET_CASE_PROTOCOL),
        "input_stream": "market_case_input_shards",
        "input_only": True,
        "output_affects_model": False,
    }


MARKET_CASE_INPUT_FIELD_TYPES: Mapping[str, str] = {
    "revision_id": "large_string",
    "revision_index": "int64",
    "market_epoch_id": "large_string",
    "market_episode_id": "large_string",
    "asof": "timestamp_utc",
    "direction": "large_string",
    "lifecycle": "large_string",
    "entry_location_id": "large_string",
    "entry_path_id": "large_string",
    "revision_stage": "large_string",
    "transition_kinds_json": "large_string",
    "observation_transition_json": "large_string",
    "scene_graph_delta_json": "large_string",
    "neutral_global_context_json": "large_string",
    "ohlcv_prefix_refs_json": "large_string",
    "source_replay_ordinal": "int64",
    "replay_update_ordinal": "int64",
    "source_bar_synthetic": "bool",
}


def _clock(value: Any, *, name: str) -> pd.Timestamp:
    return aware_timestamp(value, name=name)


def _identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"market case {name} is missing")
    return value


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return _clock(value, name="market_case.timestamp").isoformat()
    if isinstance(value, Mapping):
        return {str(_enum_value(key)): _primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_primitive(item) for item in value]
        return sorted(items, key=lambda item: canonical_json({"value": item}))
    if hasattr(value, "__dict__"):
        return {
            str(key): _primitive(item)
            for key, item in vars(value).items()
        }
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("market case input contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"market case input value is not serializable: {type(value)!r}")


def _json(value: Any) -> str:
    return json.dumps(
        _primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _parse_json(raw: Any, *, name: str) -> Any:
    if not isinstance(raw, str):
        raise ValueError(f"market case {name} must be JSON text")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"market case {name} is invalid JSON") from exc
    expected = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if raw != expected:
        raise ValueError(f"market case {name} is not canonical JSON")
    return value


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(_primitive(value))).hexdigest()


def _canonical_id_set(values: Iterable[Any], *, name: str) -> tuple[str, ...]:
    raw_values = tuple(values)
    if any(not isinstance(value, str) or not value for value in raw_values):
        raise ValueError(f"market case {name} contains an invalid identity")
    output = tuple(sorted(set(raw_values)))
    if len(output) != len(raw_values):
        raise ValueError(f"market case {name} contains duplicate identities")
    return output


def _is_fact_clock_key(key: str) -> bool:
    lowered = key.lower()
    if "deadline" in lowered:
        return False
    return (
        lowered in {"asof", "cutoff"}
        or lowered.endswith("_at")
        or lowered.endswith("_asof")
    )


def _validate_input_json_tree(
    value: Any,
    *,
    asof: pd.Timestamp,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError("market case JSON key is invalid")
            lowered = raw_key.lower()
            allowed_micro_bos_outcome = (
                raw_key == "outcome"
                and len(path) == 4
                and path[:3]
                == (
                    "observation_transition_json",
                    "collections",
                    "group5_micro_bos_transitions_this_update",
                )
                and path[3].isdigit()
                and isinstance(item, str)
                and item in _MICRO_BOS_REFERENCE_OUTCOMES
            )
            if (
                any(marker in lowered for marker in _FORBIDDEN_INPUT_KEY_MARKERS)
                and not allowed_micro_bos_outcome
            ):
                raise ValueError("market case input JSON contains a future/outcome key")
            if item is not None and _is_fact_clock_key(raw_key):
                clock = _clock(item, name=f"market_case.{'.'.join((*path, raw_key))}")
                if clock > asof:
                    raise ValueError("market case input JSON contains a future clock")
            _validate_input_json_tree(item, asof=asof, path=(*path, raw_key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_input_json_tree(item, asof=asof, path=(*path, str(index)))
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("market case input JSON contains a non-finite number")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("market case input JSON contains an unsupported value")


def _episode_direction(episode: Any) -> str:
    return Direction(getattr(episode, "direction")).value


def _episode_lifecycle(episode: Any) -> str:
    lifecycle = str(_enum_value(getattr(episode, "lifecycle", "")))
    if lifecycle not in _LIFECYCLE_ORDER:
        raise ValueError("market case MarketEpisode lifecycle is invalid")
    return lifecycle


def _episode_physical_payload(episode: Any) -> Mapping[str, Any]:
    names = (
        "episode_id",
        "market_epoch_id",
        "symbol",
        "instrument_id",
        "direction",
        "entry_location_id",
        "entry_path_id",
        "source_zone_id",
        "source_displacement_id",
        "entry_location_protocol_hash",
        "source_group3_protocol_hash",
        "source_zone_kind",
        "source_zone_protocol_hash",
        "source_bos_id",
        "lower_bound",
        "upper_bound",
        "midpoint",
        "near_edge",
        "far_edge",
        "failure_boundary",
        "formed_at",
    )
    return {name: _primitive(getattr(episode, name)) for name in names}


def _episode_state_signature(episode: Any) -> str:
    return _hash_payload(_primitive(episode))


def _milestone_tuple(episode: Any, name: str) -> tuple[Any, ...]:
    fields_by_name = {
        "first_pullback": ("first_pullback_step_id", "first_pullback_at"),
        "trigger": ("trigger_step_id", "trigger_event_id", "trigger_at"),
        "successful_pulse": (
            "successful_pulse_at",
            "successful_pulse_reason",
        ),
        "terminal": ("terminal_at", "terminal_reason"),
    }
    return tuple(getattr(episode, field) for field in fields_by_name[name])


def _validate_episode(episode: Any, *, asof: pd.Timestamp, epoch_id: str) -> None:
    episode_id = _identity(getattr(episode, "episode_id", None), name="MarketEpisode")
    if getattr(episode, "market_epoch_id", None) != epoch_id:
        raise ValueError("market case MarketEpisode crossed market epoch")
    direction = Direction(getattr(episode, "direction"))
    location_id = _identity(
        getattr(episode, "entry_location_id", None),
        name="EntryLocation",
    )
    path_id = _identity(getattr(episode, "entry_path_id", None), name="EntryPath")
    expected_episode_id = neutral_market_episode_id(
        epoch_id,
        location_id,
        path_id,
        direction,
    )
    if episode_id != expected_episode_id:
        raise ValueError("market case MarketEpisode physical identity changed")
    formed_at = _clock(getattr(episode, "formed_at", None), name="episode.formed_at")
    updated_at = _clock(getattr(episode, "updated_at", None), name="episode.updated_at")
    if formed_at > updated_at or updated_at > asof:
        raise ValueError("market case MarketEpisode clock is future-dated")
    for name in (
        "first_pullback_at",
        "trigger_at",
        "successful_pulse_at",
        "terminal_at",
    ):
        value = getattr(episode, name, None)
        if value is not None and _clock(value, name=f"episode.{name}") > asof:
            raise ValueError("market case MarketEpisode milestone is future-dated")
    pullback = _milestone_tuple(episode, "first_pullback")
    trigger = _milestone_tuple(episode, "trigger")
    pulse = _milestone_tuple(episode, "successful_pulse")
    terminal = _milestone_tuple(episode, "terminal")
    for milestone in (pullback, trigger, pulse, terminal):
        if any(value is None for value in milestone) != all(
            value is None for value in milestone
        ):
            raise ValueError("market case MarketEpisode milestone is partial")
    expected_lifecycle = (
        "terminal"
        if terminal[0] is not None
        else "triggered"
        if trigger[0] is not None
        else "pullback"
        if pullback[0] is not None
        else "registered"
    )
    if _episode_lifecycle(episode) != expected_lifecycle:
        raise ValueError("market case MarketEpisode lifecycle differs from milestones")
    if pulse[0] is not None and trigger[0] is None:
        raise ValueError("market case successful pulse lacks a trigger")
    if pullback[0] is not None:
        _identity(pullback[0], name="first pullback step")
    if trigger[0] is not None:
        _identity(trigger[0], name="trigger step")
        _identity(trigger[1], name="trigger event")
    if pulse[0] is not None:
        _identity(pulse[1], name="successful pulse reason")
    if terminal[0] is not None:
        _identity(terminal[1], name="terminal reason")
    if getattr(episode, "binding_status", None) not in {
        "unique",
        "unbound",
        "ambiguous",
    } or getattr(episode, "claim_status", None) not in {
        "unique",
        "unbound",
        "ambiguous",
    }:
        raise ValueError("market case MarketEpisode relation status is invalid")
    _episode_physical_payload(episode)


def _validate_episode_transition(previous: Any, current: Any) -> None:
    if _episode_physical_payload(previous) != _episode_physical_payload(current):
        raise ValueError("market case MarketEpisode physical custody mutated")
    if _LIFECYCLE_ORDER[_episode_lifecycle(current)] < _LIFECYCLE_ORDER[
        _episode_lifecycle(previous)
    ]:
        raise ValueError("market case MarketEpisode lifecycle moved backward")
    for name in (
        "first_pullback",
        "trigger",
        "successful_pulse",
        "terminal",
    ):
        before = _milestone_tuple(previous, name)
        after = _milestone_tuple(current, name)
        if before[0] is not None and after != before:
            raise ValueError(f"market case {name} custody mutated")


def _transition_kinds(previous: Any | None, current: Any) -> tuple[str, ...]:
    kinds: list[str] = []
    if previous is None:
        kinds.extend(("episode_created", "zone_registered"))
    for name in (
        "first_pullback",
        "trigger",
        "successful_pulse",
        "terminal",
    ):
        before = None if previous is None else _milestone_tuple(previous, name)[0]
        after = _milestone_tuple(current, name)[0]
        if before is None and after is not None:
            kinds.append(name)
    return tuple(sorted(set(kinds), key=_TRANSITION_KIND_ORDER.__getitem__))


def _observation_transition_payload(
    observation: Any,
    *,
    asof: pd.Timestamp,
    replay_update_ordinal: int,
) -> Mapping[str, Any]:
    return {
        "asof": asof,
        "replay_update_ordinal": replay_update_ordinal,
        "typed_transition_delta_available": bool(
            getattr(observation, "typed_transition_delta_available", False)
        ),
        "collections": {
            name: _primitive(tuple(getattr(observation, name, ())))
            for name in _TRANSITION_COLLECTION_NAMES
        },
    }


def _scene_node_descriptor(node: Any) -> Mapping[str, Any]:
    role = None
    for key, value in getattr(node, "semantic_attributes", ()):
        if str(key) in {"role", "authority_role", "location_role"} and value:
            role = str(value)
            break
    if role is None:
        role = str(_enum_value(getattr(node, "liquidity_role", None)) or getattr(node, "kind", "unknown"))
    return {
        "node_id": str(getattr(node, "node_id")),
        "kind": str(getattr(node, "kind", "unknown")),
        "role": role,
        "timeframe": str(_enum_value(getattr(node, "timeframe", "unknown"))),
        "structural_scale": str(
            _enum_value(getattr(node, "structural_scale", "unknown"))
        ),
        "lifecycle": str(_enum_value(getattr(node, "lifecycle", "unknown"))),
    }


def _scene_relation_descriptors(
    observation: Any,
    *,
    scene_graph: Any | None,
    added_edge_ids: tuple[str, ...],
    revised_edge_ids: tuple[str, ...],
    asof: pd.Timestamp,
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    edge_keys = tuple(("added", value) for value in added_edge_ids) + tuple(
        ("revised", value) for value in revised_edge_ids
    )
    if not edge_keys:
        return (), True
    if scene_graph is None:
        return (), False
    graph_asof = getattr(scene_graph, "last_asof", None)
    if graph_asof is not None and _clock(graph_asof, name="scene_graph.last_asof") != asof:
        raise ValueError("market case Scene Graph clock differs from observation")
    graph_revision = getattr(scene_graph, "revision_id", None)
    observation_revision = getattr(observation, "scene_revision_id", None)
    if graph_revision is not None and str(graph_revision) != str(observation_revision):
        raise ValueError("market case Scene Graph revision differs from observation")
    permanent_edges = getattr(scene_graph, "_edges", None)
    current_edges = getattr(scene_graph, "_current_path_block_edges", None)
    nodes = getattr(scene_graph, "_nodes", None)
    if not all(isinstance(value, Mapping) for value in (permanent_edges, current_edges, nodes)):
        raise TypeError("market case Scene Graph lacks current typed indexes")
    descriptors: list[Mapping[str, Any]] = []
    for change_kind, edge_id in edge_keys:
        edge = permanent_edges.get(edge_id, current_edges.get(edge_id))
        if edge is None:
            raise ValueError("market case changed Scene edge is missing")
        source = nodes.get(getattr(edge, "source_node_id", None))
        target = nodes.get(getattr(edge, "target_node_id", None))
        if source is None or target is None:
            raise ValueError("market case changed Scene edge endpoint is missing")
        observed_at = _clock(getattr(edge, "observed_at", None), name="scene_edge.observed_at")
        if observed_at > asof:
            raise ValueError("market case changed Scene edge is future-dated")
        descriptors.append(
            {
                "change_kind": change_kind,
                "edge_id": edge_id,
                "relation": str(_enum_value(getattr(edge, "relation", "unknown"))),
                "lifecycle": str(_enum_value(getattr(edge, "lifecycle", "unknown"))),
                "observed_at": observed_at,
                "source": _scene_node_descriptor(source),
                "target": _scene_node_descriptor(target),
            }
        )
    return tuple(descriptors), True


def _scene_graph_delta_payload(
    observation: Any,
    *,
    scene_graph: Any | None,
    asof: pd.Timestamp,
    replay_update_ordinal: int,
) -> Mapping[str, Any]:
    identities = {
        name: _canonical_id_set(
            getattr(observation, f"scene_{name}", ()),
            name=f"Scene {name}",
        )
        for name in _SCENE_ID_FIELDS
    }
    if set(identities["added_edge_ids"]) & set(identities["revised_edge_ids"]):
        raise ValueError("market case Scene edge is both added and revised")
    descriptors, complete = _scene_relation_descriptors(
        observation,
        scene_graph=scene_graph,
        added_edge_ids=identities["added_edge_ids"],
        revised_edge_ids=identities["revised_edge_ids"],
        asof=asof,
    )
    return {
        "asof": asof,
        "replay_update_ordinal": replay_update_ordinal,
        "revision_id": _identity(
            getattr(observation, "scene_revision_id", None),
            name="Scene revision",
        ),
        **identities,
        "relation_descriptors": descriptors,
        "relation_descriptors_complete": complete,
    }


@dataclass(frozen=True)
class MarketCaseInputRecord:
    revision_id: str
    revision_index: int
    market_epoch_id: str
    market_episode_id: str
    asof: pd.Timestamp
    direction: str
    lifecycle: str
    entry_location_id: str
    entry_path_id: str
    revision_stage: str
    transition_kinds_json: str
    observation_transition_json: str
    scene_graph_delta_json: str
    neutral_global_context_json: str
    ohlcv_prefix_refs_json: str
    source_replay_ordinal: int
    replay_update_ordinal: int
    source_bar_synthetic: bool

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in MARKET_CASE_INPUT_FIELD_TYPES}


@dataclass(frozen=True)
class _EpisodeMemory:
    episode: Any
    physical_signature: str
    state_signature: str
    next_revision_index: int


@dataclass(frozen=True)
class _ClosedEpisodeIdentity:
    physical_signature: str
    state_signature: str
    lifecycle: str
    terminal_at: pd.Timestamp | None
    terminal_reason: str | None
    status: str


@dataclass(frozen=True)
class _SnapshotView:
    asof: pd.Timestamp
    epoch_id: str
    scene_revision_id: str
    symbol: str
    instrument_id: int
    global_context: Any
    episodes: Mapping[str, Any]
    transitions: tuple[Any, ...]
    prior_epoch_terminal_transitions: tuple[Any, ...]
    retired_episode_ids: tuple[str, ...]
    retirement_reasons: Mapping[str, str]


def _snapshot_view(snapshot: Any) -> _SnapshotView:
    state = getattr(snapshot, "neutral_market_state", None)
    observation = getattr(snapshot, "observation", None)
    if state is None or observation is None:
        raise ValueError("market case snapshot lacks neutral state or observation")
    if getattr(state, "schema_version", None) != NEUTRAL_MARKET_STATE_SCHEMA_VERSION:
        raise ValueError("market case neutral runtime schema version differs")
    asof = _clock(getattr(state, "asof", None), name="neutral_state.asof")
    if _clock(getattr(observation, "asof", None), name="observation.asof") != asof:
        raise ValueError("market case neutral and observation clocks differ")
    epoch_id = _identity(getattr(state, "market_epoch_id", None), name="market epoch")
    scene_revision_id = _identity(
        getattr(state, "scene_revision_id", None),
        name="neutral Scene revision",
    )
    if getattr(observation, "scene_revision_id", None) != scene_revision_id:
        raise ValueError("market case neutral and observation Scene revisions differ")
    symbol = _identity(getattr(observation, "symbol", None), name="symbol")
    instrument_id = getattr(observation, "instrument_id", None)
    if isinstance(instrument_id, bool) or not isinstance(instrument_id, int):
        raise ValueError("market case observation instrument is invalid")
    global_context = getattr(state, "global_context", None)
    if (
        global_context is None
        or getattr(global_context, "market_epoch_id", None) != epoch_id
        or getattr(global_context, "scene_revision_id", None) != scene_revision_id
        or _clock(getattr(global_context, "updated_at", None), name="global_context.updated_at")
        != asof
    ):
        raise ValueError("market case neutral GlobalMarketContext identity differs")
    episodes: dict[str, Any] = {}
    for episode in tuple(getattr(state, "market_episodes", ())):
        _validate_episode(episode, asof=asof, epoch_id=epoch_id)
        if (
            getattr(episode, "symbol", None) != symbol
            or getattr(episode, "instrument_id", None) != instrument_id
        ):
            raise ValueError("market case episode instrument differs from observation")
        episode_id = str(episode.episode_id)
        if episode_id in episodes:
            raise ValueError("market case neutral state duplicates MarketEpisode")
        episodes[episode_id] = episode
    transitions: list[Any] = []
    prior_epoch: list[Any] = []
    transition_ids: set[str] = set()
    current_transition_ids: set[str] = set()
    for episode in tuple(getattr(state, "episode_transitions_this_update", ())):
        transition_id = _identity(
            getattr(episode, "episode_id", None),
            name="MarketEpisode transition",
        )
        if transition_id in transition_ids:
            raise ValueError("market case duplicates MarketEpisode transition")
        transition_ids.add(transition_id)
        if getattr(episode, "market_epoch_id", None) == epoch_id:
            current_transition_ids.add(transition_id)
            _validate_episode(episode, asof=asof, epoch_id=epoch_id)
            if episodes.get(transition_id) is not episode and _episode_state_signature(
                episodes.get(transition_id)
            ) != _episode_state_signature(episode):
                raise ValueError("market case transition differs from current episode")
            if _clock(getattr(episode, "updated_at", None), name="episode.updated_at") != asof:
                raise ValueError("market case transition is not from the current clock")
            transitions.append(episode)
        elif (
            _episode_lifecycle(episode) == "terminal"
            and _clock(getattr(episode, "terminal_at", None), name="episode.terminal_at")
            <= asof
        ):
            prior_epoch.append(episode)
        else:
            raise ValueError("market case transition crossed epoch without terminal")
    expected_transition_ids = {
        episode_id
        for episode_id, episode in episodes.items()
        if _clock(getattr(episode, "updated_at", None), name="episode.updated_at") == asof
    }
    if expected_transition_ids != current_transition_ids:
        raise ValueError("market case current MarketEpisode transitions are incomplete")
    if tuple(str(item.episode_id) for item in transitions) != tuple(
        sorted(str(item.episode_id) for item in transitions)
    ):
        raise ValueError("market case transitions are not deterministically ordered")
    retired_ids = tuple(getattr(state, "retired_episode_ids_this_update", ()))
    retirement_items = tuple(getattr(state, "retirement_reasons_this_update", ()))
    retirement_reasons: dict[str, str] = {}
    for item in retirement_items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("market case retirement reason is malformed")
        episode_id = _identity(item[0], name="retired MarketEpisode")
        reason = str(item[1])
        if reason not in _RETIREMENT_REASONS or episode_id in retirement_reasons:
            raise ValueError("market case retirement reason is invalid")
        retirement_reasons[episode_id] = reason
    if retired_ids != tuple(sorted(retired_ids)) or tuple(retirement_reasons) != retired_ids:
        raise ValueError("market case retirement identities differ")
    return _SnapshotView(
        asof=asof,
        epoch_id=epoch_id,
        scene_revision_id=scene_revision_id,
        symbol=symbol,
        instrument_id=instrument_id,
        global_context=global_context,
        episodes=episodes,
        transitions=tuple(transitions),
        prior_epoch_terminal_transitions=tuple(prior_epoch),
        retired_episode_ids=retired_ids,
        retirement_reasons=retirement_reasons,
    )


def _expected_revision_id(row: Mapping[str, Any]) -> str:
    payload = {
        name: (
            _clock(row[name], name="market_case.revision.asof")
            .tz_convert("UTC")
            .isoformat()
            if name == "asof"
            else _primitive(row[name])
        )
        for name in MARKET_CASE_INPUT_FIELD_TYPES
        if name != "revision_id"
    }
    return "market-input-revision:" + _hash_payload(payload)[:32]


def validate_market_case_input_row(row: Mapping[str, Any]) -> None:
    """Fail closed on one materialized input-only row."""

    if set(row) != set(MARKET_CASE_INPUT_FIELD_TYPES):
        raise ValueError("market case input row schema changed")
    for name in (
        "revision_id",
        "market_epoch_id",
        "market_episode_id",
        "direction",
        "lifecycle",
        "entry_location_id",
        "entry_path_id",
    ):
        _identity(row[name], name=name)
    if row["revision_stage"] != "market_episode_transition":
        raise ValueError("market case revision stage changed")
    for name in ("revision_index", "source_replay_ordinal", "replay_update_ordinal"):
        if isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0:
            raise ValueError(f"market case {name} is invalid")
    if type(row["source_bar_synthetic"]) is not bool:
        raise ValueError("market case source_bar_synthetic is invalid")
    asof = _clock(row["asof"], name="market_case.asof")
    direction = Direction(row["direction"])
    lifecycle = str(row["lifecycle"])
    if lifecycle not in _LIFECYCLE_ORDER:
        raise ValueError("market case lifecycle is invalid")
    if row["market_episode_id"] != neutral_market_episode_id(
        str(row["market_epoch_id"]),
        str(row["entry_location_id"]),
        str(row["entry_path_id"]),
        direction,
    ):
        raise ValueError("market case row physical identity changed")
    transition_kinds = _parse_json(
        row["transition_kinds_json"],
        name="transition_kinds_json",
    )
    if (
        not isinstance(transition_kinds, list)
        or not transition_kinds
        or any(value not in _TRANSITION_KIND_ORDER for value in transition_kinds)
        or transition_kinds
        != sorted(set(transition_kinds), key=_TRANSITION_KIND_ORDER.__getitem__)
    ):
        raise ValueError("market case transition kinds are invalid")
    if ("terminal" in transition_kinds) != (lifecycle == "terminal"):
        raise ValueError("market case terminal transition differs from lifecycle")
    observation = _parse_json(
        row["observation_transition_json"],
        name="observation_transition_json",
    )
    scene = _parse_json(row["scene_graph_delta_json"], name="scene_graph_delta_json")
    context = _parse_json(
        row["neutral_global_context_json"],
        name="neutral_global_context_json",
    )
    prefixes = _parse_json(
        row["ohlcv_prefix_refs_json"],
        name="ohlcv_prefix_refs_json",
    )
    expected_observation_keys = {
        "asof",
        "replay_update_ordinal",
        "typed_transition_delta_available",
        "collections",
    }
    if (
        not isinstance(observation, Mapping)
        or set(observation) != expected_observation_keys
        or _clock(observation["asof"], name="market_case.observation.asof") != asof
        or observation["replay_update_ordinal"] != row["replay_update_ordinal"]
        or type(observation["typed_transition_delta_available"]) is not bool
        or not isinstance(observation["collections"], Mapping)
        or set(observation["collections"]) != set(_TRANSITION_COLLECTION_NAMES)
        or any(not isinstance(value, list) for value in observation["collections"].values())
    ):
        raise ValueError("market case Observation transition shape changed")
    expected_scene_keys = {
        "asof",
        "replay_update_ordinal",
        "revision_id",
        *_SCENE_ID_FIELDS,
        "relation_descriptors",
        "relation_descriptors_complete",
    }
    if (
        not isinstance(scene, Mapping)
        or set(scene) != expected_scene_keys
        or _clock(scene["asof"], name="market_case.scene.asof") != asof
        or scene["replay_update_ordinal"] != row["replay_update_ordinal"]
        or not isinstance(scene["revision_id"], str)
        or not scene["revision_id"]
        or any(
            not isinstance(scene[name], list)
            or scene[name] != sorted(set(scene[name]))
            for name in _SCENE_ID_FIELDS
        )
        or set(scene["added_edge_ids"]) & set(scene["revised_edge_ids"])
        or not isinstance(scene["relation_descriptors"], list)
        or type(scene["relation_descriptors_complete"]) is not bool
    ):
        raise ValueError("market case Scene delta shape changed")
    if not isinstance(context, Mapping) or (
        context.get("market_epoch_id") != row["market_epoch_id"]
        or context.get("scene_revision_id") != scene["revision_id"]
        or _clock(
            context.get("updated_at"),
            name="market_case.context.updated_at",
        )
        != asof
    ):
        raise ValueError("market case neutral context identity changed")
    if not isinstance(prefixes, list) or len(prefixes) != len(_TIMEFRAMES):
        raise ValueError("market case OHLCV prefix count changed")
    expected_prefix_keys = {
        "timeframe",
        "frame_row_start",
        "frame_row_end_exclusive",
        "replay_view_1m_row_start",
        "replay_view_1m_row_end_exclusive",
        "cutoff",
    }
    expected_replay_end = int(row["source_replay_ordinal"]) + int(
        not row["source_bar_synthetic"]
    )
    for timeframe, prefix in zip(_TIMEFRAMES, prefixes, strict=True):
        if (
            not isinstance(prefix, Mapping)
            or set(prefix) != expected_prefix_keys
            or prefix["timeframe"] != timeframe.value
            or prefix["frame_row_start"] != 0
            or isinstance(prefix["frame_row_end_exclusive"], bool)
            or not isinstance(prefix["frame_row_end_exclusive"], int)
            or prefix["frame_row_end_exclusive"] < 0
            or isinstance(prefix["replay_view_1m_row_start"], bool)
            or not isinstance(prefix["replay_view_1m_row_start"], int)
            or prefix["replay_view_1m_row_start"] < 0
            or prefix["replay_view_1m_row_end_exclusive"] != expected_replay_end
            or prefix["replay_view_1m_row_start"] > expected_replay_end
            or _clock(prefix["cutoff"], name="market_case.prefix.cutoff") > asof
        ):
            raise ValueError("market case OHLCV prefix identity changed")
    for name, value in (
        ("observation_transition_json", observation),
        ("scene_graph_delta_json", scene),
        ("neutral_global_context_json", context),
        ("ohlcv_prefix_refs_json", prefixes),
    ):
        _validate_input_json_tree(value, asof=asof, path=(name,))
    if row["revision_id"] != _expected_revision_id(row):
        raise ValueError("market case stable revision identity changed")


def validate_market_case_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate sparse continuity without introducing a library finalizer."""

    revision_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        validate_market_case_input_row(row)
        revision_id = str(row["revision_id"])
        if revision_id in revision_ids:
            raise ValueError("market case revision identity is duplicated")
        revision_ids.add(revision_id)
        grouped.setdefault(
            (str(row["market_epoch_id"]), str(row["market_episode_id"])),
            [],
        ).append(row)
    for case_rows in grouped.values():
        ordered = sorted(case_rows, key=lambda item: int(item["revision_index"]))
        if [int(item["revision_index"]) for item in ordered] != list(
            range(len(ordered))
        ):
            raise ValueError("market case sparse revision indexes are discontinuous")
        previous: Mapping[str, Any] | None = None
        for row in ordered:
            if previous is not None:
                if (
                    _clock(row["asof"], name="market_case.asof")
                    < _clock(previous["asof"], name="market_case.previous_asof")
                    or int(row["replay_update_ordinal"])
                    <= int(previous["replay_update_ordinal"])
                    or _LIFECYCLE_ORDER[str(row["lifecycle"])]
                    < _LIFECYCLE_ORDER[str(previous["lifecycle"])]
                ):
                    raise ValueError("market case sparse revisions moved backward")
                if previous["lifecycle"] == "terminal":
                    raise ValueError("market case terminal episode revived")
            previous = row


class MarketEpisodeCaseRecorder:
    """Bounded input-only recorder over one existing replay snapshot stream."""

    recorder_schema_version = MARKET_CASE_RECORDER_SCHEMA_VERSION
    protocol_version = MARKET_CASE_PROTOCOL_VERSION

    def __init__(self, *, capture_start: pd.Timestamp) -> None:
        self._capture_start = _clock(capture_start, name="market_case.capture_start")
        self._current_epoch_id: str | None = None
        self._epoch_source_row_start: int | None = None
        self._active: dict[str, _EpisodeMemory] = {}
        self._closed: dict[str, _ClosedEpisodeIdentity] = {}
        self._pending_rows: list[MarketCaseInputRecord] = []
        self._last_asof: pd.Timestamp | None = None
        self._last_source_replay_ordinal: int | None = None
        self._last_replay_update_ordinal: int | None = None
        self._rows_emitted = 0
        self._episodes_recorded = 0
        self._terminal_rows = 0
        self._epoch_resets = 0
        self._primed_snapshots = 0
        self._left_censored_seeded = 0
        self._transition_kind_counts = {
            name: 0 for name in _TRANSITION_KIND_ORDER
        }

    @property
    def summary(self) -> Mapping[str, Any]:
        return {
            "input_only": True,
            "rows_emitted": self._rows_emitted,
            "episodes_recorded": self._episodes_recorded,
            "terminal_rows": self._terminal_rows,
            "epoch_resets": self._epoch_resets,
            "primed_snapshots": self._primed_snapshots,
            "left_censored_seeded": self._left_censored_seeded,
            "active_episode_count": len(self._active),
            "closed_identity_count": len(self._closed),
            "pending_row_count": len(self._pending_rows),
            "transition_kind_counts": dict(self._transition_kind_counts),
        }

    def _validate_coordinates(
        self,
        *,
        view: _SnapshotView,
        source_bar: Any,
        source_row_ordinal: int,
        replay_update_ordinal: int,
    ) -> bool:
        if (
            isinstance(source_row_ordinal, bool)
            or not isinstance(source_row_ordinal, int)
            or source_row_ordinal < 0
            or isinstance(replay_update_ordinal, bool)
            or not isinstance(replay_update_ordinal, int)
            or replay_update_ordinal < 0
        ):
            raise ValueError("market case replay coordinate is invalid")
        synthetic = getattr(source_bar, "synthetic_no_trade", None)
        if type(synthetic) is not bool:
            raise ValueError("market case source synthetic flag is invalid")
        start = _clock(getattr(source_bar, "start", None), name="source_bar.start")
        end = _clock(getattr(source_bar, "end", None), name="source_bar.end")
        if start >= end or end != view.asof:
            raise ValueError("market case source bar clock differs from snapshot")
        if (
            getattr(source_bar, "symbol", None) != view.symbol
            or getattr(source_bar, "instrument_id", None) != view.instrument_id
        ):
            raise ValueError("market case source bar instrument differs from snapshot")
        if (
            self._last_replay_update_ordinal is not None
            and replay_update_ordinal != self._last_replay_update_ordinal + 1
        ):
            raise ValueError("market case replay update ordinal is discontinuous")
        if (
            self._last_source_replay_ordinal is not None
            and source_row_ordinal < self._last_source_replay_ordinal
        ):
            raise ValueError("market case source replay ordinal moved backward")
        if self._last_asof is not None and view.asof < self._last_asof:
            raise ValueError("market case observation clock moved backward")
        return synthetic

    @staticmethod
    def _prefix_refs(
        snapshot: Any,
        *,
        asof: pd.Timestamp,
        epoch_source_row_start: int,
        source_row_ordinal: int,
        synthetic: bool,
    ) -> tuple[Mapping[str, Any], ...]:
        frames_by_timeframe = getattr(snapshot.observation, "frames", None)
        if not isinstance(frames_by_timeframe, Mapping):
            raise ValueError("market case observation frames are missing")
        refs: list[Mapping[str, Any]] = []
        for timeframe in _TIMEFRAMES:
            frame = frames_by_timeframe.get(timeframe)
            if frame is None:
                frame = frames_by_timeframe.get(timeframe.value)
            if frame is None:
                raise ValueError("market case requires five OHLCV prefixes")
            cutoff = _clock(getattr(frame, "cutoff", None), name="frame.cutoff")
            bars = getattr(frame, "bars", None)
            if cutoff > asof or isinstance(bars, bool) or not isinstance(bars, int) or bars < 0:
                raise ValueError("market case OHLCV prefix is invalid or future-dated")
            refs.append(
                {
                    "timeframe": timeframe.value,
                    "frame_row_start": 0,
                    "frame_row_end_exclusive": bars,
                    "replay_view_1m_row_start": epoch_source_row_start,
                    "replay_view_1m_row_end_exclusive": source_row_ordinal
                    + int(not synthetic),
                    "cutoff": cutoff,
                }
            )
        return tuple(refs)

    @staticmethod
    def _memory(episode: Any, *, next_revision_index: int) -> _EpisodeMemory:
        return _EpisodeMemory(
            episode=episode,
            physical_signature=_hash_payload(_episode_physical_payload(episode)),
            state_signature=_episode_state_signature(episode),
            next_revision_index=next_revision_index,
        )

    @staticmethod
    def _closed_identity(episode: Any, *, status: str) -> _ClosedEpisodeIdentity:
        terminal_at = getattr(episode, "terminal_at", None)
        return _ClosedEpisodeIdentity(
            physical_signature=_hash_payload(_episode_physical_payload(episode)),
            state_signature=_episode_state_signature(episode),
            lifecycle=_episode_lifecycle(episode),
            terminal_at=(
                None
                if terminal_at is None
                else _clock(terminal_at, name="episode.terminal_at")
            ),
            terminal_reason=getattr(episode, "terminal_reason", None),
            status=status,
        )

    @staticmethod
    def _build_row(
        snapshot: Any,
        view: _SnapshotView,
        episode: Any,
        *,
        revision_index: int,
        transition_kinds: tuple[str, ...],
        source_row_ordinal: int,
        replay_update_ordinal: int,
        synthetic: bool,
        scene_graph: Any | None,
        epoch_source_row_start: int,
    ) -> MarketCaseInputRecord:
        observation_payload = _observation_transition_payload(
            snapshot.observation,
            asof=view.asof,
            replay_update_ordinal=replay_update_ordinal,
        )
        scene_payload = _scene_graph_delta_payload(
            snapshot.observation,
            scene_graph=scene_graph,
            asof=view.asof,
            replay_update_ordinal=replay_update_ordinal,
        )
        prefix_payload = MarketEpisodeCaseRecorder._prefix_refs(
            snapshot,
            asof=view.asof,
            epoch_source_row_start=epoch_source_row_start,
            source_row_ordinal=source_row_ordinal,
            synthetic=synthetic,
        )
        values: dict[str, Any] = {
            "revision_id": "",
            "revision_index": revision_index,
            "market_epoch_id": view.epoch_id,
            "market_episode_id": str(episode.episode_id),
            "asof": view.asof,
            "direction": _episode_direction(episode),
            "lifecycle": _episode_lifecycle(episode),
            "entry_location_id": str(episode.entry_location_id),
            "entry_path_id": str(episode.entry_path_id),
            "revision_stage": "market_episode_transition",
            "transition_kinds_json": _json(transition_kinds),
            "observation_transition_json": _json(observation_payload),
            "scene_graph_delta_json": _json(scene_payload),
            "neutral_global_context_json": _json(view.global_context),
            "ohlcv_prefix_refs_json": _json(prefix_payload),
            "source_replay_ordinal": source_row_ordinal,
            "replay_update_ordinal": replay_update_ordinal,
            "source_bar_synthetic": synthetic,
        }
        values["revision_id"] = _expected_revision_id(values)
        validate_market_case_input_row(values)
        return MarketCaseInputRecord(**values)

    def _ingest(
        self,
        snapshot: Any,
        *,
        source_bar: Any,
        source_row_ordinal: int,
        replay_update_ordinal: int,
        scene_graph: Any | None,
        emit: bool,
    ) -> None:
        view = _snapshot_view(snapshot)
        synthetic = self._validate_coordinates(
            view=view,
            source_bar=source_bar,
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_update_ordinal,
        )
        anomalies = {
            str(_enum_value(value))
            for value in getattr(snapshot.observation, "anomalies", ())
        }
        boundary = anomalies.intersection(_BOUNDARY_ANOMALIES)
        had_epoch = self._current_epoch_id is not None
        epoch_changed = self._current_epoch_id != view.epoch_id
        if boundary and self._current_epoch_id is not None and not epoch_changed:
            raise ValueError("market case reset boundary did not advance market epoch")
        active = {} if epoch_changed else dict(self._active)
        closed = {} if epoch_changed else dict(self._closed)
        epoch_start = (
            source_row_ordinal
            if epoch_changed or self._epoch_source_row_start is None
            else self._epoch_source_row_start
        )
        pending: list[MarketCaseInputRecord] = []
        transition_kind_counts = dict(self._transition_kind_counts)
        episodes_recorded = self._episodes_recorded
        terminal_rows = self._terminal_rows
        left_censored_seeded = self._left_censored_seeded

        transition_ids = {str(episode.episode_id) for episode in view.transitions}
        for episode in view.transitions:
            episode_id = str(episode.episode_id)
            if episode_id in closed:
                raise ValueError("market case closed MarketEpisode transitioned again")
            previous_memory = active.get(episode_id)
            previous = None if previous_memory is None else previous_memory.episode
            if previous is not None:
                _validate_episode_transition(previous, episode)
            kinds = _transition_kinds(previous, episode)
            next_index = 0 if previous_memory is None else previous_memory.next_revision_index
            if emit and kinds:
                first_emitted_revision = next_index == 0
                row = self._build_row(
                    snapshot,
                    view,
                    episode,
                    revision_index=next_index,
                    transition_kinds=kinds,
                    source_row_ordinal=source_row_ordinal,
                    replay_update_ordinal=replay_update_ordinal,
                    synthetic=synthetic,
                    scene_graph=scene_graph,
                    epoch_source_row_start=epoch_start,
                )
                pending.append(row)
                next_index += 1
                if first_emitted_revision:
                    episodes_recorded += 1
                if "terminal" in kinds:
                    terminal_rows += 1
                for kind in kinds:
                    transition_kind_counts[kind] += 1
            memory = self._memory(episode, next_revision_index=next_index)
            if _episode_lifecycle(episode) == "terminal":
                active.pop(episode_id, None)
                closed[episode_id] = self._closed_identity(episode, status="terminal")
            else:
                active[episode_id] = memory

        for episode_id, episode in view.episodes.items():
            if episode_id in transition_ids:
                continue
            state_signature = _episode_state_signature(episode)
            if episode_id in active:
                if active[episode_id].state_signature != state_signature:
                    raise ValueError("market case episode changed without a transition")
                continue
            if episode_id in closed:
                identity = closed[episode_id]
                if identity.status == "retired":
                    raise ValueError("market case retired MarketEpisode reappeared")
                if identity.state_signature != state_signature or _episode_lifecycle(episode) != "terminal":
                    raise ValueError("market case terminal MarketEpisode custody changed")
                continue
            if _clock(episode.formed_at, name="episode.formed_at") == view.asof:
                raise ValueError("market case newly formed episode lacks a transition")
            if _episode_lifecycle(episode) == "terminal":
                closed[episode_id] = self._closed_identity(episode, status="terminal")
            else:
                active[episode_id] = self._memory(episode, next_revision_index=0)
            left_censored_seeded += int(emit)

        for episode_id in view.retired_episode_ids:
            reason = view.retirement_reasons[episode_id]
            if episode_id in view.episodes or episode_id in transition_ids:
                raise ValueError("market case retired episode remains in current state")
            if episode_id in active:
                episode = active[episode_id].episode
                if (
                    reason != "upstream_compacted_after_success"
                    or getattr(episode, "successful_pulse_at", None) is None
                ):
                    raise ValueError("market case active retirement lacks successful pulse")
                active.pop(episode_id)
                closed[episode_id] = self._closed_identity(episode, status="retired")
            elif episode_id in closed:
                identity = closed[episode_id]
                if reason != "upstream_compacted_after_terminal" or identity.lifecycle != "terminal":
                    raise ValueError("market case terminal retirement reason changed")
                closed[episode_id] = replace(identity, status="retired")
            else:
                raise ValueError("market case retirement references an unknown episode")

        missing_active = set(active).difference(view.episodes)
        if missing_active:
            raise ValueError("market case active episode disappeared without retirement")

        self._active = active
        self._closed = closed
        self._pending_rows.extend(pending)
        self._current_epoch_id = view.epoch_id
        self._epoch_source_row_start = epoch_start
        self._last_asof = view.asof
        self._last_source_replay_ordinal = source_row_ordinal
        self._last_replay_update_ordinal = replay_update_ordinal
        self._rows_emitted += len(pending)
        self._episodes_recorded = episodes_recorded
        self._terminal_rows = terminal_rows
        self._left_censored_seeded = left_censored_seeded
        self._transition_kind_counts = transition_kind_counts
        if epoch_changed and had_epoch:
            self._epoch_resets += 1

    def prime(
        self,
        snapshot: Any,
        *,
        source_bar: Any,
        source_row_ordinal: int,
        replay_update_ordinal: int,
        scene_graph: Any | None = None,
    ) -> None:
        view = _snapshot_view(snapshot)
        if view.asof >= self._capture_start:
            raise ValueError("market case prime clock reached capture start")
        self._ingest(
            snapshot,
            source_bar=source_bar,
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_update_ordinal,
            scene_graph=scene_graph,
            emit=False,
        )
        self._primed_snapshots += 1

    def observe(
        self,
        snapshot: Any,
        *,
        source_bar: Any,
        source_row_ordinal: int,
        replay_update_ordinal: int,
        scene_graph: Any | None = None,
    ) -> None:
        view = _snapshot_view(snapshot)
        if view.asof < self._capture_start:
            raise ValueError("market case observe clock precedes capture start")
        self._ingest(
            snapshot,
            source_bar=source_bar,
            source_row_ordinal=source_row_ordinal,
            replay_update_ordinal=replay_update_ordinal,
            scene_graph=scene_graph,
            emit=True,
        )

    def drain_input_rows(self) -> tuple[MarketCaseInputRecord, ...]:
        rows = tuple(self._pending_rows)
        self._pending_rows.clear()
        return rows


__all__ = [
    "MARKET_CASE_INPUT_FIELD_TYPES",
    "MARKET_CASE_PROTOCOL",
    "MARKET_CASE_PROTOCOL_VERSION",
    "MARKET_CASE_RECORDER_SCHEMA_VERSION",
    "MarketCaseInputRecord",
    "MarketEpisodeCaseRecorder",
    "expected_market_case_run_identity",
    "validate_market_case_input_row",
    "validate_market_case_rows",
]
