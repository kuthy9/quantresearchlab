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
    "fvg": ("CREATES", "RETURNS_TO"),
    "order_block": ("CREATES", "RETURNS_TO"),
    "dealing_range": ("LOCATED_AT", "SWEEPS"),
    "range_maturity_evaluation": ("LOCATED_AT",),
    "manipulation": ("SWEEPS", "PRECEDES"),
    "entry_location": ("RETURNS_TO",),
    "qualified_reacceptance": ("CONFIRMS", "PRECEDES"),
    "micro_bos": ("CONFIRMS",),
    "path_sequence": ("PRECEDES", "CONFIRMS"),
}


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
        if (
            not case_id
            or case_id in identities
            or not entity_id
            or str(case.get("stratum")) not in allowed_strata
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


def _memory_events(observation: Any) -> tuple[Any, ...]:
    events = {event.event_id: event for event in observation.recent_events}
    for timeline in observation.retained_entity_timelines.values():
        events.update((event.event_id, event) for event in timeline)
    return tuple(events.values())


def transmission_record(
    case: Mapping[str, Any],
    observation: Any,
    scene_graph: Any,
    artifact: VisualArtifact | None,
    *,
    visualization_error: str | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    entity_id = str(case["entity_id"])
    primitive = str(case["primitive"])
    identity_kind = CASE_PRIMITIVE_TO_IDENTITY.get(primitive, primitive)
    identities = typed_identity_sets(observation)
    events = _memory_events(observation)
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
    nodes = scene_graph.nodes_asof(observation.asof)
    graph_entity_ids = {
        str(value)
        for node in nodes
        for value in (
            node.entity_id,
            *node.source_ids,
        )
        if value
    }
    matching_nodes = tuple(
        node
        for node in nodes
        if node.entity_id == entity_id or entity_id in node.source_ids
    )
    source_ids = {str(value) for value in case.get("source_ids", ())}
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
    expected_relations = KEY_RELATIONS.get(primitive, ())
    graph_revision_matches = (
        observation.scene_revision_id == scene_graph.revision_id
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
    return {
        "case_id": str(case["case_id"]),
        "asof": observation.asof.isoformat(),
        "group": str(case["group"]),
        "primitive": primitive,
        "entity_id": entity_id,
        "lifecycle_or_outcome": str(case["lifecycle_or_outcome"]),
        "future_hidden": True,
        "observation": {
            "typed_identity_counts": {
                kind: len(values) for kind, values in sorted(identities.items())
            },
            "case_identity_present": entity_id
            in identities.get(identity_kind, set()),
            "transport_by_typed_kind": transport_by_kind,
        },
        "event_memory": {
            "materialized": True,
            "case_event_present": bool(matching_events),
            "matching_event_ids": sorted(event.event_id for event in matching_events),
            "incomplete_entity_timeline": any(
                key.endswith(f":{entity_id}")
                for key in observation.incomplete_entity_timeline_keys
            ),
        },
        "scene_graph": {
            "revision_matches_observation": graph_revision_matches,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "case_node_present": bool(matching_nodes),
            "matching_node_ids": sorted(node.node_id for node in matching_nodes),
            "context_node_ids": sorted(node.node_id for node in context_nodes),
            "connected_edge_count": len(connected_edges),
            "connected_relation_counts": dict(sorted(relation_counts.items())),
            "key_edge_coverage": {
                relation: relation_counts[relation] > 0
                for relation in expected_relations
            },
        },
        "image_path": image_path,
        "visualization_error": visualization_error,
    }


def _full_observer_config(payload: Mapping[str, Any]) -> Any:
    _, lightweight = _build_eye(payload)
    return replace(
        lightweight.config,
        project_scene_graph=True,
        materialize_event_view=True,
        group4_projection_only=False,
        eye_authority_mode=False,
    )


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
    config = _full_observer_config(payload)
    windows = merge_case_windows(
        index["cases"],
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
            cases = by_clock.get(observation.asof, ())
            for case in cases:
                case_id = str(case["case_id"])
                image_path = images / f"{case_id}.png"
                artifact = None
                visual_error = None
                try:
                    artifact = visualizer.render_observation(
                        observation,
                        update.histories,
                        image_path,
                        case_id=case_id,
                        scene_graph=observer.scene_graph,
                    )
                except Exception as exc:  # per-case audit result, not a silent pass
                    image_path.unlink(missing_ok=True)
                    visual_error = f"{type(exc).__name__}: {exc}"
                records[case_id] = transmission_record(
                    case,
                    observation,
                    observer.scene_graph,
                    artifact,
                    visualization_error=visual_error,
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
        "warmup_calendar_days": profile["warmup_calendar_days"],
        "merged_replay_windows": [
            {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "case_count": len(window.cases),
            }
            for window in windows
        ],
        "summary": {
            "requested_cases": len(index["cases"]),
            "observed_cases": sum("status" not in record for record in ordered),
            "images_rendered": sum(
                bool(record.get("image_path")) for record in ordered
            ),
            "event_memory_identity_covered": sum(
                bool(record.get("event_memory", {}).get("case_event_present"))
                for record in ordered
            ),
            "scene_graph_identity_covered": sum(
                bool(record.get("scene_graph", {}).get("case_node_present"))
                for record in ordered
            ),
        },
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
