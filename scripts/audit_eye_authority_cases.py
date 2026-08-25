#!/usr/bin/env python3
"""Replay sampled Eye cases with EventMemory, Scene Graph and images enabled."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eye_authority_scan import (  # noqa: E402
    CANONICAL_PROFILE,
    ROOT as SCAN_ROOT,
    _build_eye,
    _json,
    _registered_payload,
    _validate_loaded_source,
    _validate_registered_payload,
    _write_json,
)
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.observation import CausalObserver  # noqa: E402
from smc_trader.visualization import (  # noqa: E402
    DecisionVisualizer,
    VisualArtifact,
)


DEFAULT_OUTPUT = ROOT / "outputs/development/eye_authority_case_audit"
CASE_KEYS = frozenset(
    {
        "case_id",
        "stratum",
        "group",
        "primitive",
        "entity_id",
        "lifecycle_or_outcome",
        "event_clock",
        "strata",
        "source_ids",
    }
)
IDENTITY_FIELDS = {
    "swing": "swing_id",
    "structure": "structure_id",
    "bos": "bos_id",
    "support_resistance": "zone_id",
    "liquidity_pool": "pool_id",
    "liquidity_inventory": "item_id",
    "fvg": "fvg_id",
    "order_block": "order_block_id",
    "dealing_range": "range_id",
    "manipulation": "manipulation_id",
    "entry_location": "location_id",
    "qualified_reacceptance": "reacceptance_id",
    "micro_bos": "reference_id",
    "path_sequence": "sequence_id",
}
CASE_PRIMITIVE_TO_IDENTITY = {
    **{name: name for name in IDENTITY_FIELDS},
    "bos_post_break": "bos",
    "displacement": "displacement",
    "range_maturity_evaluation": "dealing_range",
}
KEY_RELATIONS = {
    "swing": ("ANCHORS", "BREAKS"),
    "structure": ("BREAKS",),
    "bos": ("BREAKS",),
    "bos_post_break": ("BREAKS",),
    "support_resistance": ("ANCHORS", "LOCATED_AT"),
    "liquidity_pool": ("LOCATED_AT", "SWEEPS"),
    "liquidity_inventory": ("LOCATED_AT", "SWEEPS"),
    "displacement": ("CREATES", "BREAKS", "PRECEDES"),
    "fvg": ("CREATES",),
    "order_block": ("CREATES",),
    "dealing_range": ("SOURCED_FROM",),
    "range_maturity_evaluation": (),
    "manipulation": ("SWEEPS",),
    "entry_location": ("RETURNS_TO",),
    "qualified_reacceptance": ("CONFIRMS", "PRECEDES"),
    "micro_bos": ("CONFIRMS",),
    # Path order is carried by PathSequenceState.steps and path-step nodes.
    # Scene Graph does not create PRECEDES/CONFIRMS edges on the sequence node.
    "path_sequence": (),
}

# Direction is expressed from the sampled entity toward its frozen source.
# Only relations that are causal requirements of the sampled primitive belong
# here; broad contextual graph connectivity remains descriptive.
CASE_RELATION_DIRECTIONS = {
    ("dealing_range", "SOURCED_FROM"): "outbound",
    ("manipulation", "SWEEPS"): "outbound",
    ("fvg", "CREATES"): "inbound",
    ("order_block", "CREATES"): "inbound",
    ("entry_location", "RETURNS_TO"): "outbound",
}
CASE_EVENT_KINDS = {
    "dealing_range": "dealing_range_state",
    "range_maturity_evaluation": "dealing_range_state",
    "manipulation": "manipulation_state",
    "path_sequence": "entry_path_state",
}
CASE_NODE_KINDS = {
    "dealing_range": "range",
    "range_maturity_evaluation": "range",
    "manipulation": "manipulation",
    "path_sequence": "path_sequence",
}
RANGE_REASONABLY_BROKEN_REASONS = frozenset(
    {"close_beyond_frozen_range", "maturity_deadline_elapsed"}
)
RANGE_SOURCE_OR_RESET_REASONS = frozenset(
    {
        "forming_source_invalidated",
        "source_identity_changed",
        "data_gap_reset",
        "contract_change_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
GROUP5_COMPLETE_REASONS = frozenset(
    {"micro_bos_aligned", "pool_reversal_sequence_observed"}
)
HARD_BOUNDARY_ANOMALIES = frozenset(
    {"contract_change_history_reset", "data_gap_history_reset"}
)
CASE_STRATUM_CONTRACTS = {
    "all_recognized_mature": ("group4", "dealing_range", {"mature"}),
    "obvious_mature_looking_but_rejected": (
        "group4",
        "range_maturity_evaluation",
        None,
    ),
    "near_mature_single_gate": (
        "group4",
        "range_maturity_evaluation",
        None,
    ),
    "multiple_gate_rejected": (
        "group4",
        "range_maturity_evaluation",
        None,
    ),
    "forming_reasonably_broken": ("group4", "dealing_range", {"broken"}),
    "source_identity_or_reset_failure": (
        "group4",
        "dealing_range",
        {"broken"},
    ),
    "mature_range_manipulation": (
        "group4",
        "manipulation",
        {
            "swept",
            "reaccepted",
            "accepted_outside",
            "deadline_censored",
        },
    ),
    "group5_complete_path": ("group5", "path_sequence", {"closed"}),
    "group5_interrupted_path": ("group5", "path_sequence", {"closed"}),
}
CASE_STRATA_FIELDS = {
    "all_recognized_mature": frozenset({"timeframe"}),
    "obvious_mature_looking_but_rejected": frozenset({"timeframe"}),
    "near_mature_single_gate": frozenset({"timeframe"}),
    "multiple_gate_rejected": frozenset({"timeframe"}),
    "forming_reasonably_broken": frozenset({"timeframe"}),
    "source_identity_or_reset_failure": frozenset({"timeframe"}),
    "mature_range_manipulation": frozenset(
        {
            "timeframe",
            "source_kind",
            "source_timeframe",
            "side",
            "structural_rank",
            "internal_external",
        }
    ),
    "group5_complete_path": frozenset({"context_kind", "direction"}),
    "group5_interrupted_path": frozenset({"context_kind", "direction"}),
}


def _case_transport_contract(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return only relations and identities applicable at the case clock."""

    primitive = str(case["primitive"])
    stratum = str(case["stratum"])
    if primitive == "range_maturity_evaluation":
        return {
            "identity_role": "range_context_for_diagnostic",
            "event_required": True,
            "event_semantics": "range_context_timeline_not_diagnostic_event",
            "node_required": True,
            "node_semantics": "range_context_node_not_diagnostic_node",
            "relation_applicability": "not_applicable_diagnostic",
            "relations": (),
        }
    if (
        primitive == "dealing_range"
        and stratum == "source_identity_or_reset_failure"
    ):
        return {
            "identity_role": "terminal_entity",
            "event_required": True,
            "event_semantics": "case_entity_timeline",
            "node_required": True,
            "node_semantics": "case_entity_node",
            "relation_applicability": "not_applicable_source_or_reset_terminal",
            "relations": (),
        }
    relations = KEY_RELATIONS.get(primitive, ())
    return {
        "identity_role": "case_entity",
        "event_required": True,
        "event_semantics": "case_entity_timeline",
        "node_required": True,
        "node_semantics": "case_entity_node",
        "relation_applicability": "required_at_case_clock",
        "relations": tuple(relations),
    }


def _case_expected_lifecycle(case: Mapping[str, Any]) -> str | None:
    if str(case["primitive"]) == "range_maturity_evaluation":
        return None
    outcome = str(case["lifecycle_or_outcome"])
    if outcome == "deadline_censored":
        return "censored"
    return outcome


def _case_shape_valid(case: Mapping[str, Any]) -> bool:
    contract = CASE_STRATUM_CONTRACTS.get(str(case.get("stratum")))
    if contract is None:
        return False
    group, primitive, outcomes = contract
    outcome = str(case.get("lifecycle_or_outcome"))
    if str(case.get("group")) != group or str(case.get("primitive")) != primitive:
        return False
    if outcomes is not None and outcome not in outcomes:
        return False
    gates = tuple(value for value in outcome.split("+") if value)
    if primitive != "range_maturity_evaluation":
        return True
    if str(case.get("stratum")) == "near_mature_single_gate":
        return len(gates) == 1
    if str(case.get("stratum")) == "multiple_gate_rejected":
        return len(gates) >= 2
    return bool(gates)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _current_epoch_id(scene_graph: Any, observation: Any) -> str | None:
    try:
        candles = scene_graph.current_candle_structure_nodes(
            observation,
            timeframes=("1m",),
            include_non_real=True,
        )
    except (AttributeError, ValueError):
        return None
    epochs = {
        str(node.market_epoch_id)
        for node in candles
        if (
            node.kind == "candle_structure"
            and node.timeframe == "1m"
            and node.lifecycle == "observed"
            and node.observed_at == observation.asof
        )
    }
    return next(iter(epochs)) if len(epochs) == 1 else None


def _case_reason_allowed(case: Mapping[str, Any], reason: str | None) -> bool:
    stratum = str(case["stratum"])
    primitive = str(case["primitive"])
    lifecycle = str(case["lifecycle_or_outcome"])
    if primitive == "dealing_range" and stratum == "all_recognized_mature":
        return reason == "maturity_conditions_met"
    if primitive == "dealing_range" and stratum == "forming_reasonably_broken":
        return reason in RANGE_REASONABLY_BROKEN_REASONS
    if (
        primitive == "dealing_range"
        and stratum == "source_identity_or_reset_failure"
    ):
        return reason in RANGE_SOURCE_OR_RESET_REASONS
    if primitive == "manipulation" and stratum == "mature_range_manipulation":
        expected_reason = {
            "swept": "source_swept",
            "reaccepted": "reentry_held_inside_swept_boundary",
            "accepted_outside": "consecutive_closes_held_outside",
            "deadline_censored": "deadline_elapsed",
        }.get(lifecycle)
        return expected_reason is not None and reason == expected_reason
    if primitive == "path_sequence" and stratum == "group5_complete_path":
        return reason in GROUP5_COMPLETE_REASONS
    if primitive == "path_sequence" and stratum == "group5_interrupted_path":
        return bool(reason) and reason not in GROUP5_COMPLETE_REASONS
    return True


def _event_matches_case(
    event: Any,
    case: Mapping[str, Any],
    case_clock: pd.Timestamp,
) -> bool:
    primitive = str(case["primitive"])
    expected_kind = CASE_EVENT_KINDS.get(primitive)
    if expected_kind is not None and _enum_value(event.kind) != expected_kind:
        return False
    if primitive == "range_maturity_evaluation":
        return (
            event.lifecycle == "forming"
            and event.observed_at <= case_clock
            and event.ended_at is None
        )
    expected_lifecycle = _case_expected_lifecycle(case)
    if (
        event.lifecycle != expected_lifecycle
        or event.observed_at != case_clock
        or not _case_reason_allowed(case, event.transition_reason)
    ):
        return False
    stratum = str(case["stratum"])
    terminal = (
        (
            primitive == "dealing_range"
            and stratum
            in {
                "forming_reasonably_broken",
                "source_identity_or_reset_failure",
            }
        )
        or (
            primitive == "path_sequence"
            and stratum
            in {"group5_complete_path", "group5_interrupted_path"}
        )
        or expected_lifecycle
        in {"reaccepted", "accepted_outside", "censored"}
    )
    return (event.ended_at == case_clock) if terminal else event.ended_at is None


def _node_matches_case(
    node: Any,
    case: Mapping[str, Any],
    case_clock: pd.Timestamp,
    current_epoch_id: str | None,
    event: Any | None,
) -> bool:
    primitive = str(case["primitive"])
    expected_kind = CASE_NODE_KINDS.get(primitive, primitive)
    if node.kind != expected_kind or node.market_epoch_id != current_epoch_id:
        return False
    if primitive == "range_maturity_evaluation":
        return node.lifecycle == "forming" and node.observed_at <= case_clock
    if (
        node.lifecycle != _case_expected_lifecycle(case)
        or node.observed_at != case_clock
    ):
        return False
    if event is None:
        return True
    return node.resolution_reason == event.transition_reason


def _diagnostic_gate_snapshot_present(
    observation: Any,
    case: Mapping[str, Any],
    case_clock: pd.Timestamp,
) -> bool | None:
    if str(case["primitive"]) != "range_maturity_evaluation":
        return None
    if not _case_shape_valid(case):
        return False
    expected_gates = tuple(str(case["lifecycle_or_outcome"]).split("+"))
    return any(
        snapshot.observed_at == case_clock
        and snapshot.maturity_range_id == str(case["entity_id"])
        and tuple(snapshot.unmet_maturity_gates) == expected_gates
        for snapshot in observation.group4_range_funnel
    )


def _source_identity_matches(node: Any, source_id: str) -> bool:
    return (
        getattr(node, "entity_id", None) == source_id
        or source_id in tuple(getattr(node, "source_ids", ()))
    )


def _exact_relation_evidence(
    *,
    primitive: str,
    expected_relations: Sequence[str],
    source_ids: set[str],
    case_nodes: Sequence[Any],
    nodes: Sequence[Any],
    edges: Sequence[Any],
    current_epoch_id: str | None,
    case_clock: pd.Timestamp,
) -> tuple[Counter[str], dict[str, set[str]], tuple[Any, ...]]:
    case_node_ids = {node.node_id for node in case_nodes}
    node_by_id = {node.node_id: node for node in nodes}
    counts: Counter[str] = Counter()
    covered_sources = {relation: set() for relation in expected_relations}
    matched: list[Any] = []
    for edge in edges:
        relation = edge.relation.value
        if relation not in covered_sources:
            continue
        if edge.lifecycle != "active" or edge.first_observed_at > case_clock:
            continue
        source_node = node_by_id.get(edge.source_node_id)
        target_node = node_by_id.get(edge.target_node_id)
        if source_node is None or target_node is None:
            continue
        if (
            source_node.market_epoch_id != current_epoch_id
            or target_node.market_epoch_id != current_epoch_id
        ):
            continue
        direction = CASE_RELATION_DIRECTIONS.get((primitive, relation))
        if direction == "outbound":
            if source_node.node_id not in case_node_ids:
                continue
            context_node = target_node
        elif direction == "inbound":
            if target_node.node_id not in case_node_ids:
                continue
            context_node = source_node
        else:
            continue
        matched_source_ids = {
            source_id
            for source_id in source_ids
            if _source_identity_matches(context_node, source_id)
        }
        if source_ids and not matched_source_ids:
            continue
        counts[relation] += 1
        covered_sources[relation].update(matched_source_ids)
        matched.append(edge)
    return counts, covered_sources, tuple(matched)


@dataclass(frozen=True)
class AuditWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    cases: tuple[Mapping[str, Any], ...]


def _clock(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return result


def _load_case_index(
    path: Path,
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    value = _json(path)
    cases = value.get("cases")
    if (
        value.get("schema_version") != 1
        or value.get("profile") != CANONICAL_PROFILE
        or not isinstance(value.get("scan_identity"), str)
        or len(value["scan_identity"]) != 64
        or value.get("future_hidden_on_first_review") is not True
        or not isinstance(cases, list)
        or not 20 <= len(cases) <= 40
    ):
        raise ValueError("case index is not a frozen 20-40 case Eye sample")
    start = _clock(
        profile["windows"][0]["start"],
        name="registered window start",
    )
    end = _clock(
        profile["windows"][0]["end_exclusive"],
        name="registered window end",
    )
    sampling = profile.get("blind_case_sampling")
    allowed_strata = (
        set(sampling.get("categories", ()))
        if isinstance(sampling, Mapping)
        else set()
    )
    if not allowed_strata:
        raise ValueError("profile has no frozen Eye case categories")
    identities: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != CASE_KEYS:
            raise ValueError("case index contains an unexpected field contract")
        case_id = str(case.get("case_id", ""))
        entity_id = str(case.get("entity_id", ""))
        event_clock = _clock(case.get("event_clock"), name="case event clock")
        strata = case.get("strata")
        expected_strata = CASE_STRATA_FIELDS.get(str(case.get("stratum")))
        if (
            not case_id
            or case_id in identities
            or not entity_id
            or str(case.get("stratum")) not in allowed_strata
            or not _case_shape_valid(case)
            or not isinstance(strata, Mapping)
            or set(strata) != expected_strata
            or any(
                not isinstance(value, (str, int, float, bool))
                or isinstance(value, float) and not pd.notna(value)
                for value in strata.values()
            )
            or not start <= event_clock < end
            or not isinstance(case.get("source_ids"), list)
            or any(not str(item) for item in case["source_ids"])
        ):
            raise ValueError("case index identity, source or clock is invalid")
        identities.add(case_id)
    return value


def merge_case_windows(
    cases: Sequence[Mapping[str, Any]],
    *,
    warmup_calendar_days: int,
    timezone: str,
) -> tuple[AuditWindow, ...]:
    if type(warmup_calendar_days) is not int or warmup_calendar_days < 1:
        raise ValueError("case audit warmup must be a positive day count")
    ordered = sorted(
        cases,
        key=lambda item: (
            _clock(item["event_clock"], name="case clock"),
            str(item["case_id"]),
        ),
    )
    windows: list[AuditWindow] = []
    for case in ordered:
        end = _clock(case["event_clock"], name="case clock").tz_convert(timezone)
        start = end - pd.DateOffset(days=warmup_calendar_days)
        if windows and start <= windows[-1].end:
            prior = windows[-1]
            windows[-1] = AuditWindow(
                start=prior.start,
                end=max(prior.end, end),
                cases=(*prior.cases, case),
            )
        else:
            windows.append(AuditWindow(start=start, end=end, cases=(case,)))
    return tuple(windows)


def _add_identities(
    target: dict[str, set[str]],
    kind: str,
    states: Iterable[Any],
) -> None:
    field = IDENTITY_FIELDS[kind]
    for state in states:
        identity = getattr(state, field, None)
        if identity:
            target.setdefault(kind, set()).add(str(identity))


def typed_identity_sets(observation: Any) -> dict[str, set[str]]:
    identities: dict[str, set[str]] = {
        kind: set() for kind in (*IDENTITY_FIELDS, "displacement")
    }
    for frame in observation.frames.values():
        _add_identities(identities, "swing", frame.swings)
        _add_identities(identities, "structure", frame.structures)
        _add_identities(identities, "bos", frame.structure_breaks)
        _add_identities(
            identities,
            "support_resistance",
            frame.support_resistance,
        )
        _add_identities(identities, "liquidity_pool", frame.liquidity_pools)
        _add_identities(identities, "fvg", frame.fair_value_gaps)
        _add_identities(identities, "order_block", frame.order_blocks)
        _add_identities(identities, "dealing_range", frame.dealing_ranges)
    _add_identities(
        identities,
        "liquidity_inventory",
        observation.liquidity_inventory,
    )
    _add_identities(identities, "liquidity_pool", observation.liquidity_pool_states)
    _add_identities(identities, "fvg", observation.group3_boundary_fvg_transitions)
    _add_identities(
        identities,
        "order_block",
        observation.group3_boundary_order_block_transitions,
    )
    _add_identities(
        identities,
        "dealing_range",
        observation.group4_boundary_range_transitions,
    )
    _add_identities(identities, "manipulation", observation.manipulations)
    _add_identities(
        identities,
        "manipulation",
        observation.group4_boundary_manipulation_transitions,
    )
    _add_identities(identities, "entry_location", observation.entry_locations)
    _add_identities(
        identities,
        "qualified_reacceptance",
        observation.qualified_reacceptances,
    )
    _add_identities(identities, "micro_bos", observation.micro_bos_references)
    _add_identities(identities, "path_sequence", observation.path_sequences)
    _add_identities(
        identities,
        "path_sequence",
        observation.group5_boundary_path_transitions,
    )
    _add_identities(
        identities,
        "qualified_reacceptance",
        observation.group5_boundary_reacceptance_transitions,
    )
    displacement = observation.displacement
    if displacement is not None:
        if displacement.current_entity_id:
            identities["displacement"].add(displacement.current_entity_id)
        identities["displacement"].update(
            transition.entity_id
            for transition in displacement.recent_transitions
        )
    return identities


def _memory_events(
    observation: Any,
    event_memory: Any | None = None,
) -> tuple[Any, ...]:
    events = {event.event_id: event for event in observation.recent_events}
    if event_memory is not None:
        events.update(
            (event.event_id, event)
            for event in event_memory.recent(limit=1_000_000)
        )
    for timeline in observation.retained_entity_timelines.values():
        events.update((event.event_id, event) for event in timeline)
    return tuple(events.values())


def _path_step_transport(
    *,
    observation: Any,
    events: Sequence[Any],
    nodes: Sequence[Any],
    current_epoch_id: str | None,
    sequence_id: str,
) -> dict[str, Any] | None:
    paths = tuple(observation.path_sequences) + tuple(
        observation.group5_boundary_path_transitions
    )
    path = next(
        (item for item in reversed(paths) if item.sequence_id == sequence_id),
        None,
    )
    if path is None:
        return {
            "typed_path_present": False,
            "typed_step_order_valid": False,
            "expected_step_ids": [],
            "event_step_ids": [],
            "scene_step_ids": [],
            "event_steps_complete": False,
            "scene_steps_complete": False,
            "complete": False,
        }
    steps = tuple(path.steps)
    step_ids = tuple(step.step_id for step in steps)
    order_valid = all(
        (
            not step.predecessor_step_ids
            if index == 0
            else step.predecessor_step_ids == (step_ids[index - 1],)
        )
        and (
            index == 0
            or steps[index - 1].observed_at <= step.observed_at
        )
        for index, step in enumerate(steps)
    )
    step_events = tuple(
        event
        for event in events
        if (
            _enum_value(event.kind) == "entry_path_step"
            and event.details.get("sequence_id") == sequence_id
        )
    )
    event_step_ids = {
        str(event.details.get("step_id"))
        for event in step_events
        if event.details.get("step_id")
    }
    event_steps_complete = all(
        any(
            event.observed_at == step.observed_at
            and event.details.get("kind") == step.kind
            and tuple(event.details.get("predecessor_step_ids", ()))
            == step.predecessor_step_ids
            and event.details.get("source_entity_id")
            == step.source_entity_id
            for event in step_events
            if event.details.get("step_id") == step.step_id
        )
        for step in steps
    )
    step_nodes = tuple(
        node
        for node in nodes
        if (
            node.kind == "path_step"
            and node.market_epoch_id == current_epoch_id
            and sequence_id in node.source_ids
        )
    )
    scene_step_ids = {
        step_id
        for node in step_nodes
        for step_id in step_ids
        if step_id in node.source_ids
    }
    scene_steps_complete = all(
        any(
            step.step_id in node.source_ids
            and step.source_entity_id in node.source_ids
            and all(
                predecessor_id in node.source_ids
                for predecessor_id in step.predecessor_step_ids
            )
            and node.observed_at == step.observed_at
            and dict(node.semantic_attributes).get("path_step_kind")
            == step.kind
            for node in step_nodes
        )
        for step in steps
    )
    complete = bool(steps) and all(
        (order_valid, event_steps_complete, scene_steps_complete)
    )
    return {
        "typed_path_present": True,
        "typed_step_order_valid": order_valid,
        "expected_step_ids": list(step_ids),
        "event_step_ids": sorted(event_step_ids),
        "scene_step_ids": sorted(scene_step_ids),
        "event_steps_complete": event_steps_complete,
        "scene_steps_complete": scene_steps_complete,
        "complete": complete,
    }


def transmission_record(
    case: Mapping[str, Any],
    observation: Any,
    scene_graph: Any,
    artifact: VisualArtifact | None,
    *,
    event_memory: Any | None = None,
    visualization_error: str | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    entity_id = str(case["entity_id"])
    primitive = str(case["primitive"])
    identity_kind = CASE_PRIMITIVE_TO_IDENTITY.get(primitive, primitive)
    identities = typed_identity_sets(observation)
    events = _memory_events(observation, event_memory)
    memory_entity_ids = {
        str(event.entity_id)
        for event in events
        if getattr(event, "entity_id", None)
    }
    matching_events = tuple(
        event
        for event in events
        if getattr(event, "entity_id", None) == entity_id
    )
    case_clock = _clock(case["event_clock"], name="case event clock")
    observation_clock_matches = observation.asof == case_clock
    source_ids = {str(value) for value in case.get("source_ids", ())}
    expected_lifecycle = _case_expected_lifecycle(case)
    exact_events = tuple(
        event
        for event in matching_events
        if _event_matches_case(event, case, case_clock)
    )
    exact_event = exact_events[-1] if exact_events else None
    nodes = scene_graph.nodes_asof(observation.asof)
    current_epoch_id = _current_epoch_id(scene_graph, observation)
    graph_entity_ids = {
        str(value)
        for node in nodes
        for value in (
            node.entity_id,
            *node.source_ids,
        )
        if value
    }
    historical_case_entity_nodes = tuple(
        node for node in nodes if node.entity_id == entity_id
    )
    case_entity_nodes = tuple(
        node
        for node in historical_case_entity_nodes
        if node.market_epoch_id == current_epoch_id
    )
    exact_case_nodes = tuple(
        node
        for node in case_entity_nodes
        if _node_matches_case(
            node,
            case,
            case_clock,
            current_epoch_id,
            exact_event,
        )
    )
    boundary_anomalies = HARD_BOUNDARY_ANOMALIES.intersection(
        observation.anomalies
    )
    boundary_event_reason = (
        None if exact_event is None else exact_event.transition_reason
    )
    expected_boundary_anomaly = {
        "contract_change_reset": "contract_change_history_reset",
        "data_gap_reset": "data_gap_history_reset",
    }.get(boundary_event_reason)
    hard_boundary_case = (
        str(case["stratum"]) == "source_identity_or_reset_failure"
        and expected_boundary_anomaly in boundary_anomalies
    )
    boundary_terminal_nodes = tuple(
        node
        for node in historical_case_entity_nodes
        if (
            hard_boundary_case
            and node.kind == "range"
            and node.market_epoch_id != current_epoch_id
            and node.lifecycle == "censored"
            and node.observed_at == case_clock
            and node.resolution_reason == expected_boundary_anomaly
        )
    )
    boundary_no_revival = not case_entity_nodes
    graph_case_state_met = (
        current_epoch_id is not None
        and bool(boundary_terminal_nodes)
        and boundary_no_revival
        if hard_boundary_case
        else current_epoch_id is not None and bool(exact_case_nodes)
    )
    node_source_carriers = (
        boundary_terminal_nodes if hard_boundary_case else exact_case_nodes
    )
    node_frozen_source_ids = {
        str(value)
        for node in node_source_carriers
        for value in (node.entity_id, *node.source_ids)
        if value
    }
    direct_node_source_coverage_met = source_ids <= node_frozen_source_ids
    matching_nodes = tuple(
        node
        for node in nodes
        if node.entity_id == entity_id or entity_id in node.source_ids
    )
    context_nodes = tuple(
        node
        for node in nodes
        if node.entity_id in source_ids
        or bool(source_ids.intersection(node.source_ids))
    )
    relevant_node_ids = {
        node.node_id for node in (*matching_nodes, *context_nodes)
    }
    edges = scene_graph.edges_asof(observation.asof)
    connected_edges = tuple(
        edge
        for edge in edges
        if edge.source_node_id in relevant_node_ids
        or edge.target_node_id in relevant_node_ids
    )
    relation_counts = Counter(edge.relation.value for edge in connected_edges)
    case_entity_node_ids = {node.node_id for node in case_entity_nodes}
    case_edges = tuple(
        edge
        for edge in edges
        if edge.source_node_id in case_entity_node_ids
        or edge.target_node_id in case_entity_node_ids
    )
    case_relation_counts = Counter(edge.relation.value for edge in case_edges)
    contract = _case_transport_contract(case)
    expected_relations = contract["relations"]
    event_required = bool(contract["event_required"])
    node_required = bool(contract["node_required"])
    graph_revision_matches = (
        observation.scene_revision_id == scene_graph.revision_id
    )
    exact_relation_counts, covered_relation_sources, exact_edges = (
        _exact_relation_evidence(
            primitive=primitive,
            expected_relations=expected_relations,
            source_ids=source_ids,
            case_nodes=exact_case_nodes,
            nodes=nodes,
            edges=edges,
            current_epoch_id=current_epoch_id,
            case_clock=case_clock,
        )
    )
    exact_relations_met = all(
        exact_relation_counts[relation] > 0
        and (
            not source_ids
            or covered_relation_sources[relation] == source_ids
        )
        for relation in expected_relations
    )
    node_source_coverage_met = (
        direct_node_source_coverage_met
        or (
            bool(expected_relations)
            and all(
                covered_relation_sources[relation] == source_ids
                for relation in expected_relations
            )
        )
    )
    transport_by_kind = {
        kind: {
            "typed": len(values),
            "event_memory": len(values & memory_entity_ids),
            "scene_graph": len(values & graph_entity_ids),
            "missing_from_event_memory": sorted(values - memory_entity_ids)[:8],
            "missing_from_scene_graph": sorted(values - graph_entity_ids)[:8],
        }
        for kind, values in sorted(identities.items())
    }
    image_path = None
    if artifact is not None:
        image_path = (
            str(artifact.path)
            if output is None
            else str(artifact.path.relative_to(output))
        )
    typed_case_present = entity_id in identities.get(identity_kind, set())
    diagnostic_gate_snapshot_present = _diagnostic_gate_snapshot_present(
        observation,
        case,
        case_clock,
    )
    incomplete_entity_timeline = any(
        key.endswith(f":{entity_id}")
        for key in observation.incomplete_entity_timeline_keys
    )
    path_step_transport = (
        _path_step_transport(
            observation=observation,
            events=events,
            nodes=nodes,
            current_epoch_id=current_epoch_id,
            sequence_id=entity_id,
        )
        if primitive == "path_sequence"
        else None
    )
    event_requirement_met = not event_required or bool(exact_events)
    node_requirement_met = not node_required or (
        graph_case_state_met and node_source_coverage_met
    )
    complete_transport_contract_met = (
        observation_clock_matches
        and typed_case_present
        and not incomplete_entity_timeline
        and (
            path_step_transport is None
            or path_step_transport["complete"]
        )
        and diagnostic_gate_snapshot_present is not False
        and event_requirement_met
        and node_requirement_met
        and exact_relations_met
        and graph_revision_matches
    )
    return {
        "case_id": str(case["case_id"]),
        "asof": observation.asof.isoformat(),
        "group": str(case["group"]),
        "primitive": primitive,
        "entity_id": entity_id,
        "lifecycle_or_outcome": str(case["lifecycle_or_outcome"]),
        "future_hidden": True,
        "observation": {
            "case_clock_matches_observation": observation_clock_matches,
            "execution_reality_status": observation.execution.source,
            "case_identity_role": contract["identity_role"],
            "diagnostic_gate_snapshot_present": diagnostic_gate_snapshot_present,
            "typed_identity_counts": {
                kind: len(values) for kind, values in sorted(identities.items())
            },
            "case_identity_present": typed_case_present,
            "transport_by_typed_kind": transport_by_kind,
        },
        "event_memory": {
            "materialized": True,
            "case_event_present": bool(matching_events),
            "case_event_state_present": bool(exact_events),
            "case_event_required": event_required,
            "case_event_semantics": contract["event_semantics"],
            "expected_lifecycle": expected_lifecycle,
            "expected_transition_clock": (
                None if expected_lifecycle is None else case_clock.isoformat()
            ),
            "case_event_requirement_met": (
                event_requirement_met
            ),
            "matching_event_ids": sorted(event.event_id for event in matching_events),
            "matching_event_states": [
                {
                    "lifecycle": getattr(event, "lifecycle", None),
                    "observed_at": (
                        None
                        if getattr(event, "observed_at", None) is None
                        else event.observed_at.isoformat()
                    ),
                    "ended_at": (
                        None
                        if getattr(event, "ended_at", None) is None
                        else event.ended_at.isoformat()
                    ),
                    "transition_reason": getattr(
                        event,
                        "transition_reason",
                        None,
                    ),
                }
                for event in matching_events
            ],
            "incomplete_entity_timeline": incomplete_entity_timeline,
            "path_step_transport": path_step_transport,
        },
        "scene_graph": {
            "revision_matches_observation": graph_revision_matches,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "current_market_epoch_id": current_epoch_id,
            "hard_boundary_case": hard_boundary_case,
            "hard_boundary_anomalies": sorted(boundary_anomalies),
            "boundary_no_revival": boundary_no_revival,
            "boundary_terminal_node_ids": sorted(
                node.node_id for node in boundary_terminal_nodes
            ),
            "historical_case_node_count": len(historical_case_entity_nodes),
            "case_node_present": bool(case_entity_nodes),
            "case_node_state_present": graph_case_state_met,
            "case_node_required": node_required,
            "case_node_semantics": contract["node_semantics"],
            "case_node_requirement_met": (
                node_requirement_met
            ),
            "case_node_direct_frozen_source_coverage_met": (
                direct_node_source_coverage_met
            ),
            "case_node_or_exact_edge_source_coverage_met": (
                node_source_coverage_met
            ),
            "case_node_frozen_source_ids": sorted(node_frozen_source_ids),
            "case_entity_node_ids": sorted(
                node.node_id for node in case_entity_nodes
            ),
            "case_entity_node_states": [
                {
                    "node_id": node.node_id,
                    "lifecycle": node.lifecycle,
                    "observed_at": node.observed_at.isoformat(),
                    "market_epoch_id": node.market_epoch_id,
                    "resolution_reason": node.resolution_reason,
                }
                for node in historical_case_entity_nodes
            ],
            "matching_node_ids": sorted(node.node_id for node in matching_nodes),
            "context_node_ids": sorted(node.node_id for node in context_nodes),
            "connected_edge_count": len(connected_edges),
            "connected_relation_counts": dict(sorted(relation_counts.items())),
            "case_edge_count": len(case_edges),
            "case_relation_counts": dict(sorted(case_relation_counts.items())),
            "key_edge_applicability": contract["relation_applicability"],
            "key_edge_coverage": {
                relation: exact_relation_counts[relation] > 0
                for relation in expected_relations
            },
            "key_edge_exact_relation_counts": dict(
                sorted(exact_relation_counts.items())
            ),
            "key_edge_frozen_source_coverage": {
                relation: sorted(covered_relation_sources[relation])
                for relation in expected_relations
            },
            "key_edge_expected_source_ids": sorted(source_ids),
            "key_edge_matched_edge_ids": sorted(edge.edge_id for edge in exact_edges),
            "key_edge_requirement_met": exact_relations_met,
        },
        "image_path": image_path,
        "visualization_error": visualization_error,
        "complete_transport_contract_met": complete_transport_contract_met,
    }


def _full_observer_config(payload: Mapping[str, Any]) -> Any:
    _, lightweight = _build_eye(payload)
    return replace(
        lightweight.config,
        project_scene_graph=True,
        materialize_event_view=True,
        range_auction_projection_only=False,
        eye_authority_mode=True,
    )


def _replay_selected_cases(
    *,
    payload: Mapping[str, Any],
    profile: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    output: Path,
) -> tuple[dict[str, dict[str, Any]], tuple[AuditWindow, ...]]:
    """Replay only the requested frozen cases through the public Eye path."""

    config = _full_observer_config(payload)
    windows = merge_case_windows(
        cases,
        warmup_calendar_days=int(profile["warmup_calendar_days"]),
        timezone=str(profile["timezone"]),
    )
    records: dict[str, dict[str, Any]] = {}
    visualizer = DecisionVisualizer()
    source_path = SCAN_ROOT / str(payload["source"]["path"])
    for window in windows:
        reader = CausalMarketReader(scale_specs=config.scale_specs)
        observer = CausalObserver(config)
        loaded = load_ohlcv(source_path, start=window.start, end=window.end)
        _validate_loaded_source(loaded, expected_path=source_path)
        by_clock: dict[pd.Timestamp, list[Mapping[str, Any]]] = {}
        for case in window.cases:
            by_clock.setdefault(
                _clock(case["event_clock"], name="case event clock"),
                [],
            ).append(case)
        for bar in iter_completed_bars(
            loaded.frame,
            allow_data_gap_reset=bool(profile["allow_data_gap_reset"]),
        ):
            update = reader.on_bar(bar)
            observation = observer.observe(update)
            selected = by_clock.get(observation.asof, ())
            for case in selected:
                case_id = str(case["case_id"])
                image_path = output / "images" / f"{case_id}.png"
                artifact = None
                visual_error = None
                try:
                    artifact = visualizer.render_observation(
                        observation,
                        update.histories,
                        image_path,
                        case_id=case_id,
                        case_entity_id=str(case["entity_id"]),
                        scene_graph=observer.scene_graph,
                    )
                except Exception as exc:  # per-case result, not a silent pass
                    image_path.unlink(missing_ok=True)
                    visual_error = f"{type(exc).__name__}: {exc}"
                records[case_id] = transmission_record(
                    case,
                    observation,
                    observer.scene_graph,
                    artifact,
                    event_memory=observer.memory,
                    visualization_error=visual_error,
                    output=output,
                )
        if len(records) == len(cases):
            break
    return records, windows


def _audit_summary(
    records: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if len(records) != len(cases):
        raise ValueError("audit summary records and frozen cases disagree")
    event_required = event_met = 0
    node_required = node_met = 0
    edge_required = edge_evaluated = edge_met = 0
    for record, case in zip(records, cases):
        contract = _case_transport_contract(case)
        event = record.get("event_memory", {})
        graph = record.get("scene_graph", {})
        if contract["event_required"]:
            event_required += 1
            event_met += int(bool(event.get("case_event_requirement_met")))
        if contract["node_required"]:
            node_required += 1
            node_met += int(bool(graph.get("case_node_requirement_met")))
        relations = tuple(contract["relations"])
        if relations:
            edge_required += 1
            exact_counts = graph.get("key_edge_exact_relation_counts")
            exact_result = graph.get("key_edge_requirement_met")
            if isinstance(exact_counts, Mapping) and isinstance(
                exact_result,
                bool,
            ):
                edge_evaluated += 1
                edge_met += int(exact_result)
    return {
        "requested_cases": len(cases),
        "observed_cases": sum("status" not in record for record in records),
        "images_rendered": sum(bool(record.get("image_path")) for record in records),
        "complete_transport_contract_met": sum(
            bool(record.get("complete_transport_contract_met"))
            for record in records
        ),
        "event_memory_identity_covered": sum(
            bool(record.get("event_memory", {}).get("case_event_present"))
            for record in records
        ),
        "event_memory_state_and_clock_covered": sum(
            bool(
                record.get("event_memory", {}).get(
                    "case_event_requirement_met"
                )
            )
            for record in records
        ),
        "event_memory_identity_required": event_required,
        "event_memory_identity_requirement_met": event_met,
        "scene_graph_identity_covered": sum(
            bool(record.get("scene_graph", {}).get("case_node_present"))
            for record in records
        ),
        "scene_graph_state_clock_epoch_covered": sum(
            bool(
                record.get("scene_graph", {}).get(
                    "case_node_requirement_met"
                )
            )
            for record in records
        ),
        "scene_graph_identity_required": node_required,
        "scene_graph_identity_requirement_met": node_met,
        "key_edge_required_cases": edge_required,
        "key_edge_evaluated_cases": edge_evaluated,
        "key_edge_requirement_met": edge_met,
    }


def run_audit(
    *,
    case_index_path: Path,
    output: Path,
    force: bool = False,
) -> dict[str, Any]:
    payload = _registered_payload()
    _validate_registered_payload(payload)
    profile = payload["profile"]
    index = _load_case_index(case_index_path, profile=profile)
    output = output.resolve()
    destination = output / "transmission_audit.json"
    images = output / "images"
    prior_images = tuple(images.glob("*.png")) if images.is_dir() else ()
    if (destination.exists() or prior_images) and not force:
        raise FileExistsError("transmission audit exists; use --force to replace it")
    if force:
        for path in prior_images:
            path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    records, windows = _replay_selected_cases(
        payload=payload,
        profile=profile,
        cases=index["cases"],
        output=output,
    )
    for case in index["cases"]:
        case_id = str(case["case_id"])
        if case_id not in records:
            records[case_id] = {
                "case_id": case_id,
                "asof": str(case["event_clock"]),
                "group": str(case["group"]),
                "primitive": str(case["primitive"]),
                "entity_id": str(case["entity_id"]),
                "lifecycle_or_outcome": str(case["lifecycle_or_outcome"]),
                "future_hidden": True,
                "status": "case_clock_not_observed",
                "image_path": None,
            }
    ordered = [records[str(case["case_id"])] for case in index["cases"]]
    result = {
        "schema_version": 1,
        "profile": CANONICAL_PROFILE,
        "annual_scan_identity": index["scan_identity"],
        "future_hidden": True,
        "brain_decision_risk_execution_used": False,
        "execution_reality_status": "not_evaluated",
        "warmup_calendar_days": profile["warmup_calendar_days"],
        "merged_replay_windows": [
            {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "case_count": len(window.cases),
            }
            for window in windows
        ],
        "summary": _audit_summary(ordered, index["cases"]),
        "cases": ordered,
    }
    _write_json(destination, result)
    return result


def main() -> None:
    payload = _registered_payload()
    profile = payload["profile"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-index",
        type=Path,
        default=ROOT / str(profile["permanent_case_index_path"]),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_audit(
        case_index_path=args.case_index,
        output=args.output,
        force=args.force,
    )
    print(
        json.dumps(
            {
                **result["summary"],
                "path": str(args.output / "transmission_audit.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
