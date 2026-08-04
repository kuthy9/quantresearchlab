"""Compact causal decision traces and full frozen packets for sampled audits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .market_clock import scheduled_gap_kind
from .model import (
    Bar,
    Candle,
    EngineSnapshot,
    EventKind,
    HypothesisBelief,
    MarketEvent,
    MarketObservation,
    Timeframe,
    content_hash,
    to_primitive,
)


# The temporal-reading projection is additive.  Keep the existing trace
# envelope version so archived four-scale calibration artifacts and their
# readers remain compatible.
TRACE_SCHEMA_VERSION = 1
QUALITY_FIELDS = (
    "thesis_strength",
    "sequence_progress",
    "location_quality",
    "entry_readiness",
    "delivery_quality",
    "uncertainty",
)
FRAME_METRICS = {
    Timeframe.H4: (
        "directional_displacement",
        "path_efficiency",
        "structure_direction",
        "structure_age_bars",
        "range_position",
        "external_above_distance_atr",
        "external_below_distance_atr",
    ),
    Timeframe.H1: (
        "swing_progression",
        "acceptance_direction",
        "rejection_direction",
        "dealing_range_position",
        "up_path_obstruction_atr",
        "down_path_obstruction_atr",
    ),
    Timeframe.M15: (
        "swing_progression",
        "acceptance_direction",
        "rejection_direction",
        "dealing_range_position",
        "up_path_obstruction_atr",
        "down_path_obstruction_atr",
    ),
    Timeframe.M5: (
        "impulse_direction",
        "impulse_strength",
        "pullback_depth",
        "pullback_completeness",
        "reacceptance_direction",
        "compression",
    ),
    Timeframe.M1: (
        "acceleration",
        "counter_pressure",
        "trigger_hold_direction",
        "trigger_age_bars",
    ),
}
INVALIDATING_LIFECYCLES = {
    "accepted",
    "accepted_outside",
    "broken",
    "censored",
    "failed",
    "formation_failed",
    "invalidated",
    "left",
    "lost",
    "retired",
}
SUCCESSFUL_PATH_CLOSE_REASONS = {
    "micro_bos_aligned",
    "pool_reversal_sequence_observed",
    "qualified_reacceptance_held",
    "zone_rejection_observed",
}
FAILED_PATH_CLOSE_REASONS = {
    "accepted_outside",
    "location_left",
    "micro_bos_ambiguous_same_clock",
    "micro_bos_opposed",
    "reacceptance_failed",
}


def active_causal_timeframes(
    observation: MarketObservation,
) -> tuple[Timeframe, ...]:
    """Return the ordered, snapshot-bound scale registry.

    Archived snapshots do not carry ``active_timeframes``.  For those
    snapshots the insertion order of ``frames`` is the four-scale authority.
    New snapshots carry the exact enabled ScaleSpec order, including M15.
    """

    active = tuple(
        getattr(observation, "active_timeframes", ())
        or tuple(observation.frames)
    )
    if (
        not active
        or len(active) != len(set(active))
        or set(active) != set(observation.frames)
        or Timeframe.M1 not in active
    ):
        raise ValueError(
            "observation active timeframes disagree with its frame mapping"
        )
    return active


def _recent_event_index(
    observation: MarketObservation,
) -> dict[str, MarketEvent]:
    """Index the bounded recent stream and retained typed lifecycles."""

    return {
        event.event_id: event
        for event in (
            *observation.recent_events,
            *(
                event
                for timeline
                in observation.retained_entity_timelines.values()
                for event in timeline
            ),
        )
    }


def _invalidating_transition(
    *,
    lifecycle: str | None,
    reason: str | None,
) -> bool:
    normalized_lifecycle = (lifecycle or "").lower()
    normalized_reason = (reason or "").lower()
    if normalized_lifecycle in INVALIDATING_LIFECYCLES:
        return True
    if normalized_lifecycle != "closed":
        return False
    if normalized_reason in SUCCESSFUL_PATH_CLOSE_REASONS:
        return False
    return normalized_reason in FAILED_PATH_CLOSE_REASONS


def _event_record(
    event: MarketEvent,
    observation: MarketObservation,
) -> dict[str, Any]:
    return {
        **to_primitive(event),
        "source": "event_memory",
        "duration_minutes": int(
            observation.event_durations_minutes.get(event.event_id, 0)
        ),
        "age_minutes": int(
            observation.event_ages_minutes.get(event.event_id, 0)
        ),
    }


def _stateful_condition(
    observation: MarketObservation,
    kind: EventKind,
    timeframe: Timeframe,
) -> bool:
    """Mirror the observer's registered state-open conditions."""

    metrics = observation.frame(timeframe).metrics
    if kind is EventKind.IMPULSE and timeframe is Timeframe.M5:
        return float(metrics.get("impulse_strength", 0.0)) >= 0.55
    if kind is EventKind.REACCEPTANCE and timeframe is Timeframe.M5:
        return abs(float(metrics.get("reacceptance_direction", 0.0))) >= 0.35
    if kind is EventKind.COMPRESSION and timeframe is Timeframe.M5:
        return float(metrics.get("compression", 0.0)) >= 0.60
    if kind is EventKind.TRIGGER_HELD and timeframe is Timeframe.M1:
        return float(metrics.get("trigger_hold_direction", 0.0)) != 0.0
    return False


def _event_delta(
    snapshot: EngineSnapshot,
    previous_snapshot: EngineSnapshot | None,
) -> dict[str, Any]:
    current = _recent_event_index(snapshot.observation)
    previous_ids = (
        set()
        if previous_snapshot is None
        else set(_recent_event_index(previous_snapshot.observation))
    )
    added = sorted(
        (
            event
            for event_id, event in current.items()
            if event_id not in previous_ids
        ),
        key=lambda event: (
            event.observed_at,
            event.sequence_no,
            event.event_id,
        ),
    )
    previous_observation = (
        None if previous_snapshot is None else previous_snapshot.observation
    )
    previous_entity_events: dict[str, MarketEvent] = {}
    previous_state_events: dict[tuple[EventKind, Timeframe], MarketEvent] = {}
    if previous_observation is not None:
        for timeline in previous_observation.retained_entity_timelines.values():
            if timeline:
                event = timeline[-1]
                if event.entity_id is not None:
                    previous_entity_events[event.entity_id] = event
        for event in previous_observation.recent_events:
            if event.kind in {
                EventKind.IMPULSE,
                EventKind.REACCEPTANCE,
                EventKind.COMPRESSION,
                EventKind.TRIGGER_HELD,
            } and event.ended_at is None:
                state_key = (event.kind, event.timeframe)
                prior_state = previous_state_events.get(state_key)
                if prior_state is None or (
                    event.observed_at,
                    event.sequence_no,
                    event.event_id,
                ) > (
                    prior_state.observed_at,
                    prior_state.sequence_no,
                    prior_state.event_id,
                ):
                    previous_state_events[state_key] = event
            if event.entity_id is not None:
                prior = previous_entity_events.get(event.entity_id)
                if prior is None or (
                    event.observed_at,
                    event.sequence_no,
                    event.event_id,
                ) > (
                    prior.observed_at,
                    prior.sequence_no,
                    prior.event_id,
                ):
                    previous_entity_events[event.entity_id] = event
    ended_records: list[dict[str, Any]] = []
    ended_ids: set[str] = set()
    for event in added:
        prior: MarketEvent | None = None
        if event.entity_id is not None:
            prior = previous_entity_events.get(event.entity_id)
        elif event.kind in {
            EventKind.IMPULSE,
            EventKind.REACCEPTANCE,
            EventKind.COMPRESSION,
            EventKind.TRIGGER_HELD,
        }:
            candidate = previous_state_events.get(
                (event.kind, event.timeframe)
            )
            if (
                candidate is not None
                and previous_observation is not None
                and _stateful_condition(
                    previous_observation,
                    event.kind,
                    event.timeframe,
                )
            ):
                prior = candidate
        elif (
            previous_observation is not None
            and event.kind.value == "trigger_lost"
            and _stateful_condition(
                previous_observation,
                EventKind.TRIGGER_HELD,
                event.timeframe,
            )
        ):
            prior = next(
                (
                    candidate
                    for candidate in reversed(
                        previous_observation.recent_events
                    )
                    if candidate.kind.value == "trigger_held"
                    and candidate.timeframe is event.timeframe
                    and candidate.ended_at is None
                ),
                None,
            )
        if prior is None:
            continue
        terminal = current.get(prior.event_id)
        terminal_is_observed = bool(
            terminal is not None and terminal.ended_at is not None
        )
        record = _event_record(
            terminal if terminal_is_observed else prior,
            snapshot.observation,
        )
        record.update(
            {
                "ended_at": (
                    terminal.ended_at
                    if terminal_is_observed
                    else event.observed_at
                ),
                "ended_by_event_id": event.event_id,
                "end_reason": (
                    terminal.transition_reason
                    if terminal_is_observed
                    else event.transition_reason or event.kind.value
                ),
            }
        )
        ended_records.append(record)
        ended_ids.add(prior.event_id)
    if previous_observation is not None:
        for (kind, timeframe), prior in previous_state_events.items():
            if (
                prior.event_id in ended_ids
                or not _stateful_condition(
                    previous_observation,
                    kind,
                    timeframe,
                )
                or _stateful_condition(
                    snapshot.observation,
                    kind,
                    timeframe,
                )
            ):
                continue
            terminal = current.get(prior.event_id)
            terminal_is_observed = bool(
                terminal is not None and terminal.ended_at is not None
            )
            ended_at = (
                terminal.ended_at
                if terminal_is_observed
                else snapshot.observation.frame(timeframe).cutoff
            )
            record = _event_record(
                terminal if terminal_is_observed else prior,
                snapshot.observation,
            )
            record.update(
                {
                    "ended_at": ended_at,
                    "ended_by_event_id": None,
                    "end_reason": (
                        terminal.transition_reason
                        if terminal_is_observed
                        else "state_condition_no_longer_held"
                    ),
                }
            )
            ended_records.append(record)
            ended_ids.add(prior.event_id)
    ended_records.sort(
        key=lambda item: (
            pd.Timestamp(item["ended_at"]),
            item["event_id"],
        )
    )
    invalidated = [
        event
        for event in added
        if (
            _invalidating_transition(
                lifecycle=event.lifecycle,
                reason=event.transition_reason,
            )
            or event.kind
            in {
                EventKind.TRIGGER_LOST,
                EventKind.LIQUIDITY_CONSUMED,
            }
        )
    ]
    return {
        "initialized_without_prior": previous_snapshot is None,
        "added": [
            _event_record(event, snapshot.observation) for event in added
        ],
        "ended": ended_records,
        "invalidated": [
            _event_record(event, snapshot.observation)
            for event in invalidated
        ],
    }


def _typed_transition_delta(
    snapshot: EngineSnapshot,
    previous_snapshot: EngineSnapshot | None,
) -> list[dict[str, Any]]:
    current = snapshot.observation
    previous = (
        None if previous_snapshot is None else previous_snapshot.observation
    )
    rows: list[dict[str, Any]] = []

    prior_displacement_ids = (
        set()
        if previous is None or previous.displacement is None
        else {
            item.transition_id
            for item in previous.displacement.recent_transitions
        }
    )
    if current.displacement is not None:
        rows.extend(
            {
                "family": "displacement",
                "entity_id": item.entity_id,
                "revision_id": item.transition_id,
                "lifecycle": item.lifecycle,
                "observed_at": item.observed_at,
                "reason": item.reason,
                "state": to_primitive(item),
            }
            for item in current.displacement.recent_transitions
            if item.transition_id not in prior_displacement_ids
        )

    boundary_families = (
        (
            "fvg_boundary",
            "fvg_id",
            current.group3_boundary_fvg_transitions,
            ()
            if previous is None
            else previous.group3_boundary_fvg_transitions,
        ),
        (
            "order_block_boundary",
            "order_block_id",
            current.group3_boundary_order_block_transitions,
            ()
            if previous is None
            else previous.group3_boundary_order_block_transitions,
        ),
        (
            "range_boundary",
            "range_id",
            current.group4_boundary_range_transitions,
            ()
            if previous is None
            else previous.group4_boundary_range_transitions,
        ),
        (
            "manipulation_boundary",
            "manipulation_id",
            current.group4_boundary_manipulation_transitions,
            ()
            if previous is None
            else previous.group4_boundary_manipulation_transitions,
        ),
        (
            "entry_path_boundary",
            "sequence_id",
            current.group5_boundary_path_transitions,
            ()
            if previous is None
            else previous.group5_boundary_path_transitions,
        ),
        (
            "reacceptance_boundary",
            "reacceptance_id",
            current.group5_boundary_reacceptance_transitions,
            ()
            if previous is None
            else previous.group5_boundary_reacceptance_transitions,
        ),
    )
    for family, identity_name, states, prior_states in boundary_families:
        prior_ids = {
            (
                getattr(item, identity_name),
                getattr(item, "lifecycle").value,
                getattr(item, "state_started_at", None),
            )
            for item in prior_states
        }
        for item in states:
            identity = (
                getattr(item, identity_name),
                getattr(item, "lifecycle").value,
                getattr(item, "state_started_at", None),
            )
            if identity in prior_ids:
                continue
            rows.append(
                {
                    "family": family,
                    "entity_id": getattr(item, identity_name),
                    "revision_id": (
                        f"{family}:{getattr(item, identity_name)}:"
                        f"{getattr(item, 'lifecycle').value}:"
                        f"{getattr(item, 'last_updated_at', current.asof)}"
                    ),
                    "lifecycle": getattr(item, "lifecycle").value,
                    "observed_at": getattr(
                        item,
                        "last_updated_at",
                        current.asof,
                    ),
                    "reason": getattr(item, "transition_reason", None),
                    "state": to_primitive(item),
                }
            )

    state_families = (
        (
            "entry_location",
            "location_id",
            current.entry_locations,
            () if previous is None else previous.entry_locations,
        ),
        (
            "qualified_reacceptance",
            "reacceptance_id",
            current.qualified_reacceptances,
            ()
            if previous is None
            else previous.qualified_reacceptances,
        ),
        (
            "entry_path",
            "sequence_id",
            current.path_sequences,
            () if previous is None else previous.path_sequences,
        ),
    )
    for family, identity_name, states, prior_states in state_families:
        prior_by_id = {
            getattr(item, identity_name): item for item in prior_states
        }
        for item in states:
            prior = prior_by_id.get(getattr(item, identity_name))
            revision = (
                item.lifecycle.value,
                item.state_started_at,
                len(getattr(item, "steps", ())),
                getattr(item, "first_entered_at", None),
                getattr(item, "reclaimed_at", None),
                getattr(item, "held_at", None),
            )
            prior_revision = (
                None
                if prior is None
                else (
                    prior.lifecycle.value,
                    prior.state_started_at,
                    len(getattr(prior, "steps", ())),
                    getattr(prior, "first_entered_at", None),
                    getattr(prior, "reclaimed_at", None),
                    getattr(prior, "held_at", None),
                )
            )
            if revision == prior_revision:
                continue
            rows.append(
                {
                    "family": family,
                    "entity_id": getattr(item, identity_name),
                    "revision_id": (
                        f"{family}:{getattr(item, identity_name)}:"
                        f"{item.lifecycle.value}:{item.last_updated_at}"
                    ),
                    "lifecycle": item.lifecycle.value,
                    "observed_at": item.last_updated_at,
                    "reason": item.transition_reason,
                    "state": to_primitive(item),
                }
            )

    prior_micro_ids = (
        set()
        if previous is None
        else {item.reference_id for item in previous.micro_bos_references}
    )
    rows.extend(
        {
            "family": "micro_bos",
            "entity_id": item.reference_id,
            "revision_id": f"micro_bos:{item.reference_id}",
            "lifecycle": "qualified" if item.qualified else "observed",
            "observed_at": item.resolved_at,
            "reason": item.outcome,
            "state": to_primitive(item),
        }
        for item in current.micro_bos_references
        if item.reference_id not in prior_micro_ids
    )
    return to_primitive(
        sorted(
            rows,
            key=lambda item: (
                pd.Timestamp(item["observed_at"]),
                item["family"],
                item["revision_id"],
            ),
        )
    )


def _evidence_rows(items: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "primitive": item.primitive,
            "value": float(item.value),
            "observed_at": item.observed_at,
            "reason": item.explanation,
        }
        for item in items
    ]


def _hypothesis_summary(
    hypothesis: HypothesisBelief,
    *,
    selected: bool,
    asof: pd.Timestamp,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "key": hypothesis.key,
        "selected": bool(selected),
        "playbook": hypothesis.playbook.value,
        "direction": hypothesis.direction.value,
        "phase": hypothesis.phase.value,
        "phase_started_at": hypothesis.phase_started_at,
        "phase_age_minutes": max(
            0,
            int((asof - hypothesis.phase_started_at).total_seconds() // 60),
        ),
        "probability": float(hypothesis.probability),
        "raw_probability": hypothesis.raw_probability,
        "eligible": hypothesis.eligible,
        "effective_probability": hypothesis.effective_probability,
        "uncertainty": float(hypothesis.uncertainty),
        "qualities": {
            name: getattr(hypothesis, name) for name in QUALITY_FIELDS
        },
        "raw_qualities": dict(hypothesis.raw_quality_dimensions),
        "evidence_groups": dict(hypothesis.evidence_group_scores),
        "hard_gates": dict(hypothesis.hard_gate_results),
        "setup_context_id": hypothesis.setup_context_id,
        "entry_location_id": hypothesis.entry_location_id,
        "context_id": hypothesis.context_id,
        "episode_id": hypothesis.episode_id,
        "episode_deadline": hypothesis.episode_deadline,
        "initiating_event_id": hypothesis.initiating_event_id,
        "evidence_revision_id": hypothesis.evidence_revision_id,
        "terminal_at": hypothesis.terminal_at,
        "terminal_reason": hypothesis.terminal_reason,
        "terminal_source_ids": hypothesis.terminal_source_ids,
        "remaining_path_R": hypothesis.remaining_path_R,
        "supporting": _evidence_rows(hypothesis.supporting),
        "contradicting": _evidence_rows(hypothesis.contradicting),
        "invalidation": to_primitive(hypothesis.invalidation),
        "deliverable_targets": to_primitive(
            hypothesis.deliverable_targets
        ),
        "draw_selection": to_primitive(hypothesis.draw_selection),
        "liquidity_route": to_primitive(hypothesis.liquidity_route),
        "sequence": to_primitive(hypothesis.sequence),
        "plan": to_primitive(hypothesis.plan),
    }
    return output


def _enum_text(value: Any) -> Any:
    return getattr(value, "value", value)


def _focus_summary(focus: Any | None) -> dict[str, Any] | None:
    if focus is None:
        return None
    return {
        "asof": getattr(focus, "asof", None),
        "primary_timeframes": tuple(
            getattr(focus, "primary_timeframes", ())
        ),
        "supplemental_timeframes": tuple(
            getattr(focus, "supplemental_timeframes", ())
        ),
        "reason_codes": tuple(getattr(focus, "reason_codes", ())),
        "trigger_event_ids": tuple(
            getattr(focus, "trigger_event_ids", ())
        ),
        "question": getattr(focus, "question", None),
        "resolution_status": _enum_text(
            getattr(focus, "resolution_status", None)
        ),
        "switched": bool(getattr(focus, "switched", False)),
        "switched_at": getattr(focus, "switched_at", None),
        "prior_timeframes": tuple(
            getattr(focus, "prior_timeframes", ())
        ),
        "hypothesis_id": getattr(focus, "hypothesis_id", None),
        "phase_at_selection": getattr(
            focus,
            "phase_at_selection",
            None,
        ),
        "supplemental_query_used": bool(
            getattr(focus, "supplemental_query_used", False)
        ),
        "focus_revision_id": getattr(
            focus,
            "focus_revision_id",
            None,
        ),
    }


def _context_hypothesis_summary(context: Any) -> dict[str, Any]:
    def evidence_states(name: str) -> dict[str, Any]:
        return {
            str(key): _enum_text(value)
            for key, value in dict(getattr(context, name, {})).items()
        }

    def graph_paths(name: str) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(str(value) for value in path)
            for path in getattr(context, name, ())
        )

    return {
        "hypothesis_id": getattr(context, "hypothesis_id", None),
        "playbook": _enum_text(getattr(context, "playbook", None)),
        "direction": _enum_text(getattr(context, "direction", None)),
        "context_root_ids": tuple(
            getattr(context, "context_root_ids", ())
        ),
        "context_timeframe": getattr(
            context,
            "context_timeframe",
            None,
        ),
        "setup_timeframe": getattr(context, "setup_timeframe", None),
        "trigger_timeframe": getattr(
            context,
            "trigger_timeframe",
            None,
        ),
        "sequence_stage": getattr(context, "sequence_stage", None),
        "next_expected_event": getattr(
            context,
            "next_expected_event",
            None,
        ),
        "supporting_graph_paths": graph_paths(
            "supporting_graph_paths"
        ),
        "contradicting_graph_paths": graph_paths(
            "contradicting_graph_paths"
        ),
        # Preserve UNKNOWN/AMBIGUOUS as explicit states.  They are not
        # flattened into a failed boolean hard gate.
        "missing_evidence": evidence_states("missing_evidence"),
        "ambiguous_evidence": evidence_states("ambiguous_evidence"),
        "context_draw_id": getattr(context, "context_draw_id", None),
        "primary_target_id": getattr(
            context,
            "primary_target_id",
            None,
        ),
        "invalidation_id": getattr(context, "invalidation_id", None),
        "evidence_revision_id": getattr(
            context,
            "evidence_revision_id",
            None,
        ),
    }


def _market_reading_projection(
    snapshot: EngineSnapshot,
) -> dict[str, Any]:
    observation = snapshot.observation
    belief = snapshot.belief
    contexts = {
        str(identity): _context_hypothesis_summary(context)
        for identity, context in dict(
            getattr(belief, "context_hypotheses", {})
        ).items()
    }
    return {
        "active_timeframes": tuple(
            timeframe.value
            for timeframe in active_causal_timeframes(observation)
        ),
        "scale_registry_id": getattr(
            observation,
            "scale_registry_id",
            "legacy-four-scale",
        ),
        "scene_revision_id": (
            getattr(belief, "scene_revision_id", None)
            or getattr(observation, "scene_revision_id", None)
        ),
        "scene_delta": {
            "added_node_ids": tuple(
                getattr(observation, "scene_added_node_ids", ())
            ),
            "revised_node_ids": tuple(
                getattr(observation, "scene_revised_node_ids", ())
            ),
            "added_edge_ids": tuple(
                getattr(observation, "scene_added_edge_ids", ())
            ),
            "revised_edge_ids": tuple(
                getattr(observation, "scene_revised_edge_ids", ())
            ),
            "resolution_event_ids": tuple(
                getattr(
                    observation,
                    "scene_resolution_event_ids",
                    (),
                )
            ),
        },
        "focus_state": _focus_summary(
            getattr(belief, "focus_state", None)
        ),
        "dominant_hypothesis_id": getattr(
            belief,
            "dominant_hypothesis_id",
            None,
        ),
        "competing_hypothesis_ids": tuple(
            getattr(belief, "competing_hypothesis_ids", ())
        ),
        "context_hypotheses": contexts,
        "cross_scale_conflicts": tuple(
            getattr(belief, "cross_scale_conflicts", ())
        ),
        "unresolved_ambiguities": tuple(
            getattr(belief, "unresolved_ambiguities", ())
        ),
    }


def _market_reading_delta(
    snapshot: EngineSnapshot,
    previous_snapshot: EngineSnapshot | None,
) -> dict[str, Any]:
    current = _market_reading_projection(snapshot)
    previous = (
        None
        if previous_snapshot is None
        else _market_reading_projection(previous_snapshot)
    )
    current_contexts = current["context_hypotheses"]
    previous_contexts = (
        {} if previous is None else previous["context_hypotheses"]
    )
    current_ids = set(current_contexts)
    previous_ids = set(previous_contexts)
    return {
        "initialized_without_prior": previous is None,
        "scene_revision_from": (
            None if previous is None else previous["scene_revision_id"]
        ),
        "scene_revision_to": current["scene_revision_id"],
        "scene_changed": (
            previous is None
            or previous["scene_revision_id"]
            != current["scene_revision_id"]
        ),
        "focus_revision_from": (
            None
            if previous is None or previous["focus_state"] is None
            else previous["focus_state"].get("focus_revision_id")
        ),
        "focus_revision_to": (
            None
            if current["focus_state"] is None
            else current["focus_state"].get("focus_revision_id")
        ),
        "focus_changed": (
            previous is None
            or previous["focus_state"] != current["focus_state"]
        ),
        "contexts_added": tuple(sorted(current_ids - previous_ids)),
        "contexts_retired": tuple(sorted(previous_ids - current_ids)),
        "contexts_revised": tuple(
            sorted(
                identity
                for identity in current_ids & previous_ids
                if current_contexts[identity]
                != previous_contexts[identity]
            )
        ),
        "dominant_hypothesis_from": (
            None
            if previous is None
            else previous["dominant_hypothesis_id"]
        ),
        "dominant_hypothesis_to": current[
            "dominant_hypothesis_id"
        ],
        "new_cross_scale_conflicts": tuple(
            sorted(
                set(current["cross_scale_conflicts"])
                - (
                    set()
                    if previous is None
                    else set(previous["cross_scale_conflicts"])
                )
            )
        ),
        "new_unresolved_ambiguities": tuple(
            sorted(
                set(current["unresolved_ambiguities"])
                - (
                    set()
                    if previous is None
                    else set(previous["unresolved_ambiguities"])
                )
            )
        ),
    }


def _number_delta(current: Any, previous: Any) -> float | None:
    if current is None or previous is None:
        return None
    return float(current) - float(previous)


def _terminal_summary(
    hypothesis: HypothesisBelief,
) -> dict[str, Any] | None:
    if hypothesis.terminal_at is None:
        return None
    return {
        "at": hypothesis.terminal_at,
        "reason": hypothesis.terminal_reason,
        "source_ids": hypothesis.terminal_source_ids,
    }


def _same_completed_m1_bar(left: Any, right: Any) -> bool:
    fields = (
        "start",
        "end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "instrument_id",
    )
    return all(getattr(left, name, None) == getattr(right, name, None) for name in fields)


def _entity_summary(item: Any) -> dict[str, Any]:
    identity = next(
        (
            getattr(item, name)
            for name in (
                "swing_id",
                "structure_id",
                "bos_id",
                "zone_id",
                "pool_id",
                "fvg_id",
                "order_block_id",
                "range_id",
            )
            if getattr(item, name, None) is not None
        ),
        None,
    )
    lifecycle = getattr(item, "lifecycle", None)
    direction = getattr(item, "direction", None)
    return to_primitive(
        {
            "id": identity,
            "lifecycle": (
                None
                if lifecycle is None
                else getattr(lifecycle, "value", lifecycle)
            ),
            "direction": (
                None
                if direction is None
                else getattr(direction, "value", direction)
            ),
            "side": getattr(
                getattr(item, "side", None),
                "value",
                getattr(item, "side", None),
            ),
            "relation": getattr(
                getattr(item, "relation", None),
                "value",
                getattr(item, "relation", None),
            ),
            "price": getattr(item, "price", None),
            "target_price": getattr(item, "target_price", None),
            "lower_bound": getattr(item, "lower_bound", None),
            "upper_bound": getattr(item, "upper_bound", None),
            "formed_at": getattr(item, "formed_at", None),
            "confirmed_at": getattr(item, "confirmed_at", None),
            "state_started_at": getattr(item, "state_started_at", None),
            "last_updated_at": getattr(item, "last_updated_at", None),
            "resolved_at": getattr(item, "resolved_at", None),
            "age_bars": getattr(
                item,
                "age_bars",
                getattr(item, "age_h1_bars", None),
            ),
            "strength": getattr(item, "strength", None),
            "transition_reason": getattr(
                item,
                "transition_reason",
                getattr(item, "failure_reason", None),
            ),
        }
    )


def _belief_delta(
    snapshot: EngineSnapshot,
    previous_snapshot: EngineSnapshot | None,
) -> dict[str, Any]:
    prior_hypotheses = (
        {}
        if previous_snapshot is None
        else previous_snapshot.belief.hypotheses
    )
    output: dict[str, Any] = {}
    for key, current in snapshot.belief.hypotheses.items():
        previous = prior_hypotheses.get(key)
        if previous is None:
            output[key] = {
                "initialized": True,
                "phase_from": None,
                "phase_to": current.phase.value,
                "quality_delta": {
                    name: None for name in QUALITY_FIELDS
                },
                "evidence_group_delta": {
                    name: None
                    for name in current.evidence_group_scores
                },
                "hard_gate_changes": dict(current.hard_gate_results),
                "setup_changed": current.setup_context_id is not None,
                "context_id_from": None,
                "context_id_to": current.context_id,
                "context_changed": current.context_id is not None,
                "episode_id_from": None,
                "episode_id_to": current.episode_id,
                "episode_changed": current.episode_id is not None,
                "episode_deadline_from": None,
                "episode_deadline_to": current.episode_deadline,
                "episode_deadline_changed": (
                    current.episode_deadline is not None
                ),
                "terminal_from": None,
                "terminal_to": _terminal_summary(current),
                "terminal_changed": current.terminal_at is not None,
                "eligible_from": None,
                "eligible_to": current.eligible,
                "effective_probability_delta": None,
            }
            continue
        gate_names = set(previous.hard_gate_results) | set(
            current.hard_gate_results
        )
        output[key] = {
            "initialized": False,
            "phase_from": previous.phase.value,
            "phase_to": current.phase.value,
            "phase_changed": previous.phase is not current.phase,
            "quality_delta": {
                name: _number_delta(
                    getattr(current, name),
                    getattr(previous, name),
                )
                for name in QUALITY_FIELDS
            },
            "evidence_group_delta": {
                name: _number_delta(
                    current.evidence_group_scores.get(name),
                    previous.evidence_group_scores.get(name),
                )
                for name in (
                    set(previous.evidence_group_scores)
                    | set(current.evidence_group_scores)
                )
            },
            "hard_gate_changes": {
                name: current.hard_gate_results.get(name)
                for name in gate_names
                if previous.hard_gate_results.get(name)
                != current.hard_gate_results.get(name)
            },
            "setup_changed": (
                previous.setup_context_id != current.setup_context_id
            ),
            "entry_location_changed": (
                previous.entry_location_id != current.entry_location_id
            ),
            "context_id_from": previous.context_id,
            "context_id_to": current.context_id,
            "context_changed": previous.context_id != current.context_id,
            "episode_id_from": previous.episode_id,
            "episode_id_to": current.episode_id,
            "episode_changed": previous.episode_id != current.episode_id,
            "episode_deadline_from": previous.episode_deadline,
            "episode_deadline_to": current.episode_deadline,
            "episode_deadline_changed": (
                previous.episode_deadline != current.episode_deadline
            ),
            "terminal_from": _terminal_summary(previous),
            "terminal_to": _terminal_summary(current),
            "terminal_changed": (
                _terminal_summary(previous) != _terminal_summary(current)
            ),
            "eligible_from": previous.eligible,
            "eligible_to": current.eligible,
            "effective_probability_delta": _number_delta(
                current.effective_probability,
                previous.effective_probability,
            ),
        }
    return output


def _observation_summary(
    observation: MarketObservation,
    previous_observation: MarketObservation | None,
    source_bar: Any | None,
) -> dict[str, Any]:
    frames = {}
    active_timeframes = active_causal_timeframes(observation)
    for timeframe in active_timeframes:
        frame = observation.frame(timeframe)
        previous_frame = (
            None
            if previous_observation is None
            else previous_observation.frames.get(timeframe)
        )
        frames[timeframe.value] = {
            "cutoff": frame.cutoff,
            "bars": int(frame.bars),
            "ready": bool(frame.ready),
            "updated_this_minute": (
                previous_frame is None
                or previous_frame.cutoff != frame.cutoff
            ),
            "metrics": {
                name: float(frame.metrics[name])
                for name in FRAME_METRICS.get(timeframe, ())
                if name in frame.metrics
            },
            "primitive_counts": {
                "swings": len(frame.swings),
                "structures": len(frame.structures),
                "bos": len(frame.structure_breaks),
                "support_resistance": len(frame.support_resistance),
                "liquidity_pools": len(frame.liquidity_pools),
                "fvg": len(frame.fair_value_gaps),
                "order_blocks": len(frame.order_blocks),
                "dealing_ranges": len(frame.dealing_ranges),
            },
            "typed_states": {
                "swings": [
                    _entity_summary(item) for item in frame.swings[-4:]
                ],
                "structures": [
                    _entity_summary(item) for item in frame.structures[-2:]
                ],
                "bos": [
                    _entity_summary(item)
                    for item in frame.structure_breaks[-3:]
                ],
                "support_resistance": [
                    _entity_summary(item)
                    for item in frame.support_resistance[-2:]
                ],
                "liquidity_pools": [
                    _entity_summary(item)
                    for item in frame.liquidity_pools[-2:]
                ],
                "fvg": [
                    _entity_summary(item)
                    for item in frame.fair_value_gaps[-2:]
                ],
                "order_blocks": [
                    _entity_summary(item)
                    for item in frame.order_blocks[-2:]
                ],
                "dealing_ranges": [
                    _entity_summary(item)
                    for item in frame.dealing_ranges[-2:]
                ],
                "candle_structure": to_primitive(frame.candle_structure),
            },
        }
    return {
        "asof": observation.asof,
        "symbol": observation.symbol,
        "instrument_id": observation.instrument_id,
        "price": float(observation.price),
        "active_timeframes": tuple(
            timeframe.value for timeframe in active_timeframes
        ),
        "scale_registry_id": getattr(
            observation,
            "scale_registry_id",
            "legacy-four-scale",
        ),
        "scene_revision_id": getattr(
            observation,
            "scene_revision_id",
            None,
        ),
        "scene_delta": {
            "added_node_ids": tuple(
                getattr(observation, "scene_added_node_ids", ())
            ),
            "revised_node_ids": tuple(
                getattr(observation, "scene_revised_node_ids", ())
            ),
            "added_edge_ids": tuple(
                getattr(observation, "scene_added_edge_ids", ())
            ),
            "resolution_event_ids": tuple(
                getattr(
                    observation,
                    "scene_resolution_event_ids",
                    (),
                )
            ),
        },
        "source_bar": to_primitive(source_bar),
        "frames": frames,
        "displacement": (
            None
            if observation.displacement is None
            else {
                "lifecycle": observation.displacement.lifecycle,
                "entity_id": observation.displacement.current_entity_id,
                "direction": (
                    None
                    if observation.displacement.current_direction is None
                    else observation.displacement.current_direction.value
                ),
                "started_at": observation.displacement.current_started_at,
                "active_at": observation.displacement.current_active_at,
                "metrics": dict(
                    observation.displacement.current_metrics or ()
                ),
                "latest_transition": to_primitive(
                    observation.displacement.latest_transition
                ),
                "transitions_this_update": to_primitive(
                    observation.displacement.transitions_this_update
                ),
            }
        ),
        "group5": {
            "entry_locations": [
                {
                    "id": item.location_id,
                    "lifecycle": item.lifecycle.value,
                    "direction": item.direction.value,
                    "zone_id": item.source_zone_id,
                    "first_entered_at": item.first_entered_at,
                    "last_updated_at": item.last_updated_at,
                }
                for item in observation.entry_locations
            ],
            "reacceptances": [
                {
                    "id": item.reacceptance_id,
                    "lifecycle": item.lifecycle.value,
                    "context_id": item.context_id,
                    "direction": item.direction.value,
                    "last_updated_at": item.last_updated_at,
                }
                for item in observation.qualified_reacceptances
            ],
            "micro_bos": [
                {
                    "id": item.reference_id,
                    "context_id": item.context_id,
                    "bos_id": item.bos_id,
                    "qualified": item.qualified,
                    "outcome": item.outcome,
                    "resolved_at": item.resolved_at,
                }
                for item in observation.micro_bos_references
            ],
            "path_sequences": [
                {
                    "id": item.sequence_id,
                    "context_id": item.context_id,
                    "lifecycle": item.lifecycle.value,
                    "direction": item.direction.value,
                    "steps": [
                        {
                            "kind": step.kind,
                            "observed_at": step.observed_at,
                            "source_entity_id": step.source_entity_id,
                        }
                        for step in item.steps
                    ],
                }
                for item in observation.path_sequences
            ],
        },
        "execution": to_primitive(observation.execution),
        "anomalies": list(observation.anomalies),
    }


def build_decision_trace(
    snapshot: EngineSnapshot,
    previous_snapshot: EngineSnapshot | None = None,
    *,
    hypothesis_key: str | None = None,
    source_bar: Any | None = None,
    account_state: Any | None = None,
    belief_position_input: Any | None = None,
) -> dict[str, Any]:
    """Return one compact, causal minute trace without any future path."""

    reset_anomalies = {
        "contract_change_history_reset",
        "data_gap_history_reset",
    }
    brain_reset_before_update = bool(
        reset_anomalies.intersection(snapshot.observation.anomalies)
    )
    if (
        previous_snapshot is not None
        and previous_snapshot.observation.asof >= snapshot.observation.asof
    ):
        raise ValueError("decision trace prior must strictly predate the decision")
    if source_bar is not None and (
        not isinstance(source_bar, (Bar, Candle))
        or source_bar.end != snapshot.observation.asof
        or (
            isinstance(source_bar, Candle)
            and (
                source_bar.timeframe is not Timeframe.M1
                or not source_bar.complete
            )
        )
    ):
        raise ValueError(
            "decision trace source bar type or observation clock differs"
        )
    if source_bar is not None and (
        source_bar.symbol,
        source_bar.instrument_id,
    ) != (
        snapshot.observation.symbol,
        snapshot.observation.instrument_id,
    ):
        raise ValueError(
            "decision trace source bar and observation contracts differ"
        )
    if source_bar is not None and (
        source_bar.start
        != snapshot.observation.asof - pd.Timedelta(minutes=1)
        or float(source_bar.close) != float(snapshot.observation.price)
    ):
        raise ValueError(
            "decision trace source bar does not match the observed M1 close"
        )
    selected_key = hypothesis_key or snapshot.decision.best_hypothesis_key
    if selected_key is not None and selected_key not in snapshot.belief.hypotheses:
        raise ValueError("decision trace selected hypothesis is absent")
    if (
        previous_snapshot is not None
        and not brain_reset_before_update
        and (
            previous_snapshot.observation.symbol,
            previous_snapshot.observation.instrument_id,
        )
        != (
            snapshot.observation.symbol,
            snapshot.observation.instrument_id,
        )
    ):
        raise ValueError(
            "decision trace changed contract without a causal reset"
        )
    causal_previous_snapshot = (
        None if brain_reset_before_update else previous_snapshot
    )
    events = _event_delta(snapshot, causal_previous_snapshot)
    typed_transitions = _typed_transition_delta(
        snapshot,
        causal_previous_snapshot,
    )
    typed_invalidations = [
        {
            **item,
            "source": "typed_state_transition",
        }
        for item in typed_transitions
        if _invalidating_transition(
            lifecycle=item.get("lifecycle"),
            reason=item.get("reason"),
        )
    ]
    current_beliefs = {
        key: _hypothesis_summary(
            hypothesis,
            selected=key == selected_key,
            asof=snapshot.observation.asof,
        )
        for key, hypothesis in snapshot.belief.hypotheses.items()
    }
    previous_beliefs = (
        None
        if causal_previous_snapshot is None
        else {
            key: _hypothesis_summary(
                hypothesis,
                selected=key == selected_key,
                asof=causal_previous_snapshot.belief.asof,
            )
            for key, hypothesis
            in causal_previous_snapshot.belief.hypotheses.items()
        }
    )
    return to_primitive(
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "decision_hash": snapshot.snapshot_hash,
            "decision_asof": snapshot.observation.asof,
            "prior_belief_asof": (
                None
                if causal_previous_snapshot is None
                else causal_previous_snapshot.belief.asof
            ),
            "discarded_pre_reset_belief_asof": (
                previous_snapshot.belief.asof
                if brain_reset_before_update
                and previous_snapshot is not None
                else None
            ),
            "observation": _observation_summary(
                snapshot.observation,
                (
                    None
                    if causal_previous_snapshot is None
                    else causal_previous_snapshot.observation
                ),
                source_bar,
            ),
            "account_before_decision": to_primitive(account_state),
            "belief_position_input": to_primitive(belief_position_input),
            "events_added": events["added"],
            "events_ended": events["ended"],
            "events_invalidated": [
                *events["invalidated"],
                *typed_invalidations,
            ],
            "typed_state_transitions": typed_transitions,
            "event_delta_initialized_without_prior": events[
                "initialized_without_prior"
            ],
            "primitive_boundary": bool(
                {
                    "contract_change_history_reset",
                    "data_gap_history_reset",
                    "data_anomaly",
                    "tick_size_mismatch",
                }.intersection(snapshot.observation.anomalies)
            ),
            "reader_history_reset": bool(
                reset_anomalies.intersection(
                    snapshot.observation.anomalies
                )
            ),
            "brain_reset_before_update": brain_reset_before_update,
            "belief_t_minus_1": previous_beliefs,
            "belief_t": current_beliefs,
            "belief_delta": _belief_delta(
                snapshot,
                causal_previous_snapshot,
            ),
            "market_reading_t": _market_reading_projection(snapshot),
            "market_reading_delta": _market_reading_delta(
                snapshot,
                causal_previous_snapshot,
            ),
            "selected_hypothesis_key": selected_key,
            "decision": to_primitive(snapshot.decision),
            "risk": to_primitive(snapshot.risk),
            "future_path_included": False,
        }
    )


def build_frozen_decision_packet(
    snapshot: EngineSnapshot,
    histories: Mapping[Timeframe, Sequence[Any]],
    previous_snapshot: EngineSnapshot | None = None,
    *,
    hypothesis_key: str | None = None,
    audit_context: Mapping[str, Any] | None = None,
    source_bar: Any | None = None,
    account_state: Any | None = None,
    belief_position_input: Any | None = None,
) -> dict[str, Any]:
    """Freeze one sampled decision information set for blind multi-agent audit."""

    validated_histories = validate_causal_histories(snapshot, histories)
    if (
        source_bar is not None
        and not _same_completed_m1_bar(
            source_bar,
            validated_histories[Timeframe.M1][-1],
        )
    ):
        raise ValueError(
            "frozen packet source bar differs from the causal M1 history"
        )
    brain_reset_before_update = bool(
        {
            "contract_change_history_reset",
            "data_gap_history_reset",
        }.intersection(snapshot.observation.anomalies)
    )
    causal_histories = {
        timeframe.value: to_primitive(validated_histories[timeframe])
        for timeframe in active_causal_timeframes(snapshot.observation)
    }
    maximum_market_time = max(
        item.end
        for values in validated_histories.values()
        for item in values
    )
    payload = {
        "format_version": 1,
        "artifact": "frozen_causal_decision_packet",
        "decision_hash": snapshot.snapshot_hash,
        "decision_asof": snapshot.observation.asof,
        "maximum_market_time": maximum_market_time,
        "histories": causal_histories,
        "observation_t": to_primitive(snapshot.observation),
        "belief_t_minus_1": (
            None
            if previous_snapshot is None or brain_reset_before_update
            else to_primitive(previous_snapshot.belief)
        ),
        "belief_discarded_before_reset": (
            to_primitive(previous_snapshot.belief)
            if previous_snapshot is not None and brain_reset_before_update
            else None
        ),
        "brain_reset_before_update": brain_reset_before_update,
        "belief_t": to_primitive(snapshot.belief),
        "market_reading_t": _market_reading_projection(snapshot),
        "decision_t": to_primitive(snapshot.decision),
        "risk_t": to_primitive(snapshot.risk),
        "source_bar_t": to_primitive(source_bar),
        "account_before_decision": to_primitive(account_state),
        "belief_position_input": to_primitive(belief_position_input),
        "decision_trace": build_decision_trace(
            snapshot,
            previous_snapshot,
            hypothesis_key=hypothesis_key,
            source_bar=source_bar,
            account_state=account_state,
            belief_position_input=belief_position_input,
        ),
        "audit_hypothesis_key": hypothesis_key,
        "audit_context": dict(audit_context or {}),
        "future_path": {
            "included": False,
            "revealed": False,
            "storage": "physically_separate_artifact",
        },
    }
    primitive = to_primitive(payload)
    primitive["packet_hash"] = content_hash(primitive)
    return primitive


def sealed_path_audit_context(setup_id: str) -> dict[str, Any]:
    if not isinstance(setup_id, str) or not setup_id:
        raise ValueError("sealed path audit requires a setup identity")
    return {
        "scenario": "sealed_path_audit",
        "setup_id": setup_id,
        "future_present": False,
    }


def validate_causal_histories(
    snapshot: EngineSnapshot,
    histories: Mapping[Timeframe, Sequence[Any]],
) -> dict[Timeframe, tuple[Candle, ...]]:
    """Fail closed when a visual/audit history differs from model cutoffs."""

    active_timeframes = active_causal_timeframes(snapshot.observation)
    if set(histories) != set(active_timeframes):
        expected = ", ".join(
            timeframe.value for timeframe in active_timeframes
        )
        raise ValueError(
            "causal history must match the enabled scale registry: "
            f"{expected}"
        )
    output: dict[Timeframe, tuple[Candle, ...]] = {}
    for timeframe in active_timeframes:
        values = tuple(histories[timeframe])
        frame = snapshot.observation.frame(timeframe)
        if any(
            not isinstance(item, Candle)
            or item.timeframe is not timeframe
            or not item.complete
            or item.end > snapshot.observation.asof
            or (item.symbol, item.instrument_id)
            != (
                snapshot.observation.symbol,
                snapshot.observation.instrument_id,
            )
            for item in values
        ):
            raise ValueError(
                "causal history contains a wrong, incomplete or future candle"
            )
        if (
            len({(item.start, item.end) for item in values}) != len(values)
            or tuple(
                sorted(
                    values,
                    key=lambda item: (item.start, item.end),
                )
            )
            != values
            or any(
                right.start < left.end
                for left, right in zip(values[:-1], values[1:])
            )
            or any(
                right.start != left.end
                and scheduled_gap_kind(left.end, right.start) is None
                for left, right in zip(values[:-1], values[1:])
            )
        ):
            raise ValueError(
                "causal history is duplicated, unordered or has an "
                "unexplained gap"
            )
        if (
            not values
            or (values and values[-1].end != frame.cutoff)
        ):
            raise ValueError(
                "causal history cutoff disagrees with observation"
            )
        output[timeframe] = values
    if output[Timeframe.M1][-1].end != snapshot.observation.asof:
        raise ValueError("causal M1 history does not reach the decision clock")
    return output


def write_frozen_decision_packet(
    snapshot: EngineSnapshot,
    histories: Mapping[Timeframe, Sequence[Any]],
    destination: str | Path,
    previous_snapshot: EngineSnapshot | None = None,
    *,
    hypothesis_key: str | None = None,
    audit_context: Mapping[str, Any] | None = None,
    source_bar: Any | None = None,
    account_state: Any | None = None,
    belief_position_input: Any | None = None,
) -> Path:
    packet = build_frozen_decision_packet(
        snapshot,
        histories,
        previous_snapshot,
        hypothesis_key=hypothesis_key,
        audit_context=audit_context,
        source_bar=source_bar,
        account_state=account_state,
        belief_position_input=belief_position_input,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(decision_packet_bytes(packet))
    return output


def decision_packet_bytes(packet: Mapping[str, Any]) -> bytes:
    """Return the one registered on-disk serialization for a frozen packet."""

    return json.dumps(
        packet,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")


def decision_packet_payload_sha256(packet: Mapping[str, Any]) -> str:
    return hashlib.sha256(decision_packet_bytes(packet)).hexdigest()


def decision_packet_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_verified_decision_packet(path: str | Path) -> dict[str, Any]:
    packet = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError("frozen decision packet root must be an object")
    claimed_hash = packet.get("packet_hash")
    unhashed = dict(packet)
    unhashed.pop("packet_hash", None)
    if (
        not isinstance(claimed_hash, str)
        or content_hash(unhashed) != claimed_hash
    ):
        raise ValueError("frozen decision packet content hash is invalid")
    return packet


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "active_causal_timeframes",
    "build_decision_trace",
    "build_frozen_decision_packet",
    "decision_packet_bytes",
    "decision_packet_payload_sha256",
    "decision_packet_sha256",
    "read_verified_decision_packet",
    "sealed_path_audit_context",
    "validate_causal_histories",
    "write_frozen_decision_packet",
]
