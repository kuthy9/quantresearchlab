"""AI visual feedback adapter: issue flags become unvalidated causal primitives."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from .decision_trace import decision_packet_payload_sha256
from .model import EngineSnapshot, clamp, content_hash


class ReviewIssue(str, Enum):
    MISSED_STRUCTURE_SEQUENCE = "missed_structure_sequence"
    PULLBACK_AS_INVALIDATION = "pullback_as_invalidation"
    LATE_DISPLACEMENT_CHASE = "late_displacement_chase"
    WRONG_LIQUIDITY_DRAW = "wrong_liquidity_draw"
    INVALID_STRUCTURAL_STOP = "invalid_structural_stop"


@dataclass(frozen=True)
class PrimitiveProposal:
    proposal_id: str
    issue: ReviewIssue
    confidence: float
    primitive_name: str
    formula_version: str
    definition_hash: str
    required_inputs: tuple[str, ...]
    formula: str
    clock_rule: str
    path_test: str
    status: str
    decision_hash: str
    hypothesis_key: str | None
    setup_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    decision_packet_hash: str
    decision_packet_sha256: str
    origin_value: "PrimitiveValue"
    note: str


@dataclass(frozen=True)
class PrimitiveValue:
    primitive_name: str
    formula_version: str
    definition_hash: str
    decision_packet_hash: str
    decision_packet_sha256: str
    decision_hash: str
    decision_asof: pd.Timestamp
    hypothesis_key: str
    setup_id: str
    entry_location_id: str
    entry_path_id: str
    evaluable: bool
    reason: str
    values: Mapping[str, Any]


PRIMITIVE_MAP: dict[ReviewIssue, dict[str, Any]] = {
    ReviewIssue.MISSED_STRUCTURE_SEQUENCE: {
        "primitive_name": "typed_causal_sequence_prefix_implementation_check",
        "formula_version": "2.0.0",
        "required_inputs": (
            "hypothesis_sequence_steps",
            "typed_entry_path_steps",
            "step_observation_times",
            "predecessor_step_ids",
        ),
        "formula": (
            "implementation_consistent=(satisfied hypothesis steps form one "
            "unbroken time-ordered "
            "prefix and the exact typed entry path forms one predecessor-bound "
            "causal chain); typed-source alignment requires separate blind "
            "trace review; this is never swing progression"
        ),
        "clock_rule": "all satisfied step clocks <= decision_time",
        "path_test": (
            "implementation consistency alone cannot approve this primitive; "
            "a separate blind trace must verify the registered event order"
        ),
    },
    ReviewIssue.PULLBACK_AS_INVALIDATION: {
        "primitive_name": "pullback_room_before_frozen_invalidation",
        "formula_version": "2.0.0",
        "required_inputs": (
            "exact_entry_path_first_pullback",
            "frozen_entry_zone_failure_boundary",
            "frozen_invalidation",
            "completed_pullback_path",
        ),
        "formula": (
            "room_R=d*(completed-path extreme-frozen thesis invalidation)/risk; "
            "penetration=clip((near_edge-extreme)/(near_edge-far_edge),0,1); "
            "zone_intact=(no completed close beyond frozen zone failure); "
            "thesis_intact=(frozen thesis invalidation not touched)"
        ),
        "clock_rule": "current completed bars only; frozen invalidation cannot be rewritten",
        "path_test": (
            "implementation checks cannot decide whether a normal pullback was "
            "misread; a separate blind sequence/path audit is required"
        ),
    },
    ReviewIssue.LATE_DISPLACEMENT_CHASE: {
        "primitive_name": "extension_to_remaining_draw_ratio",
        "formula_version": "2.0.0",
        "required_inputs": (
            "entry_location_source_displacement_id",
            "frozen_source_zone",
            "source_displacement_completed_m5_prefix",
            "nearest_visible_target_distance",
            "current_spread_cost",
        ),
        "formula": (
            "net_remaining_atr=max(0,(d*(draw-planned_entry)"
            "-round_trip_cost)/formation_ATR); "
            "source_extension_atr=d*(prefix_last_close-seed_open)/formation_ATR; "
            "ratio=null if remaining==0 else source_extension_atr/remaining"
        ),
        "clock_rule": (
            "zone, exact source displacement prefix and target must all be "
            "visible no later than decision_time"
        ),
        "path_test": (
            "implementation consistency cannot approve chase semantics; a "
            "separate revealed path must test delivery from the frozen location"
        ),
    },
    ReviewIssue.WRONG_LIQUIDITY_DRAW: {
        "primitive_name": "registered_draw_provenance_integrity",
        "formula_version": "3.0.0",
        "required_inputs": (
            "plan_selected_draw_id",
            "plan_primary_target_id",
            "playbook_sequence_sources",
            "prior_same_setup_plan",
            "entry_location_nearest_draw",
            "causal_liquidity_inventory",
        ),
        "formula": (
            "selected draw must equal the registered playbook source: DFP uses "
            "its frozen H4 structure/draw step; LSR preserves the prior exact "
            "setup draw or uses the exact entry location nearest eligible draw"
        ),
        "clock_rule": "each level formed and confirmed no later than decision_time",
        "path_test": (
            "future acceptance compares the frozen eligible inventory with "
            "typed liquidity-consumption order, never PnL"
        ),
    },
    ReviewIssue.INVALID_STRUCTURAL_STOP: {
        "primitive_name": "structural_invalidation_origin_integrity",
        "formula_version": "2.0.0",
        "required_inputs": (
            "frozen_plan_invalidation",
            "exact_playbook_setup_path",
            "dfp_entry_location_failure_boundary",
            "lsr_manipulation_sweep_extreme",
            "setup_location_path_identities",
        ),
        "formula": (
            "valid=unique registered playbook source "
            "& source_clock==plan_stop_clock<=decision "
            "& semantic_source_price==stop & side_matches "
            "& risk_geometry and setup/location/path identities agree; "
            "distance_R=abs(entry-stop)/risk"
        ),
        "clock_rule": "source observed_at <= decision_time and immutable after entry",
        "path_test": (
            "later audit must retain the same frozen source and price; no "
            "target, PnL or best-price criterion"
        ),
    },
}


def _definition_hash(definition: Mapping[str, Any]) -> str:
    payload = {
        name: definition[name]
        for name in (
            "primitive_name",
            "formula_version",
            "required_inputs",
            "formula",
            "clock_rule",
            "path_test",
        )
    }
    return content_hash(payload)


def _packet_identity(
    packet: Mapping[str, Any],
) -> tuple[str, str, pd.Timestamp]:
    claimed = packet.get("packet_hash")
    unhashed = dict(packet)
    unhashed.pop("packet_hash", None)
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or content_hash(unhashed) != claimed
        or packet.get("future_path", {}).get("included") is not False
    ):
        raise ValueError("primitive computation requires a verified causal packet")
    decision_hash = packet.get("decision_hash")
    decision_asof = pd.Timestamp(packet.get("decision_asof"))
    if (
        not isinstance(decision_hash, str)
        or not decision_hash
        or decision_asof.tzinfo is None
    ):
        raise ValueError("primitive packet decision identity is invalid")
    return claimed, decision_hash, decision_asof


def _packet_hypothesis(
    packet: Mapping[str, Any],
    hypothesis_key: str,
) -> Mapping[str, Any]:
    if packet.get("audit_hypothesis_key") != hypothesis_key:
        raise ValueError(
            "primitive packet is sealed for another hypothesis"
        )
    hypotheses = packet.get("belief_t", {}).get("hypotheses", {})
    hypothesis = hypotheses.get(hypothesis_key)
    if not isinstance(hypothesis, Mapping):
        raise ValueError("primitive packet lacks its reviewed hypothesis")
    return hypothesis


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _packet_review_identity(
    packet: Mapping[str, Any],
    hypothesis_key: str,
) -> dict[str, str | None]:
    """Derive the exact sealed setup identity and reject internal drift."""

    hypothesis = _packet_hypothesis(packet, hypothesis_key)
    sequence = _mapping_or_empty(hypothesis.get("sequence"))
    plan = _mapping_or_empty(hypothesis.get("plan"))
    setup_values = {
        value
        for value in (
            hypothesis.get("setup_context_id"),
            sequence.get("setup_id"),
            plan.get("setup_id"),
        )
        if value is not None
    }
    location_values = {
        value
        for value in (
            hypothesis.get("entry_location_id"),
            plan.get("entry_location_id"),
        )
        if value is not None
    }
    if len(setup_values) > 1 or len(location_values) > 1:
        raise ValueError(
            "primitive packet hypothesis identities disagree internally"
        )
    setup_id = next(iter(setup_values), None)
    entry_location_id = next(iter(location_values), None)
    entry_path_id = plan.get("entry_path_id")
    if any(
        not isinstance(value, str) or not value
        for value in (
            setup_id,
            entry_location_id,
            entry_path_id,
        )
    ):
        raise ValueError(
            "primitive packet requires complete setup/location/path identity"
        )
    audit_context = _mapping_or_empty(packet.get("audit_context"))
    if (
        audit_context.get("scenario") != "sealed_path_audit"
        or audit_context.get("setup_id") != setup_id
        or audit_context.get("future_present") is not False
    ):
        raise ValueError(
            "primitive packet audit context is sealed for another setup"
        )
    decision_trace = _mapping_or_empty(packet.get("decision_trace"))
    if decision_trace.get("selected_hypothesis_key") != hypothesis_key:
        raise ValueError(
            "primitive packet trace selected another hypothesis"
        )
    observation = _mapping_or_empty(packet.get("observation_t"))
    locations = [
        item
        for item in observation.get("entry_locations", ())
        if (
            isinstance(item, Mapping)
            and item.get("location_id") == entry_location_id
        )
    ]
    paths = (
        *observation.get("path_sequences", ()),
        *observation.get("group5_boundary_path_transitions", ()),
    )
    entry_paths = [
        item
        for item in paths
        if (
            isinstance(item, Mapping)
            and item.get("sequence_id") == entry_path_id
        )
    ]
    setup_paths = [
        item
        for item in paths
        if (
            isinstance(item, Mapping)
            and item.get("sequence_id") == setup_id
        )
    ]
    location = locations[0] if len(locations) == 1 else None
    entry_path = entry_paths[0] if len(entry_paths) == 1 else None
    setup_path = setup_paths[0] if len(setup_paths) == 1 else None
    direction = hypothesis.get("direction")
    playbook = hypothesis.get("playbook")
    if playbook == "displacement_first_pullback":
        playbook_context_valid = bool(
            setup_id == entry_path_id
            and isinstance(setup_path, Mapping)
            and setup_path.get("context_kind") == "zone_return"
        )
    elif playbook == "liquidity_sweep_reversal":
        playbook_context_valid = bool(
            isinstance(setup_path, Mapping)
            and setup_path.get("context_kind") == "pool_reversal"
        )
    elif playbook == "failed_auction_value_return":
        playbook_context_valid = True
    else:
        playbook_context_valid = False
    if (
        not isinstance(location, Mapping)
        or not isinstance(entry_path, Mapping)
        or not isinstance(setup_path, Mapping)
        or location.get("direction") != direction
        or entry_path.get("direction") != direction
        or entry_path.get("context_kind") != "zone_return"
        or entry_path.get("context_id") != entry_location_id
        or not playbook_context_valid
        or any(
            not _clock_is_causal(value, pd.Timestamp(packet["decision_asof"]))
            for value in (
                location.get("formed_at"),
                location.get("last_updated_at"),
                setup_path.get("formed_at"),
                setup_path.get("last_updated_at"),
                entry_path.get("formed_at"),
                entry_path.get("last_updated_at"),
            )
        )
        or any(
            not _clock_is_causal(
                step.get("observed_at"),
                pd.Timestamp(packet["decision_asof"]),
            )
            for path in (setup_path, entry_path)
            for step in path.get("steps", ())
            if isinstance(step, Mapping)
        )
    ):
        raise ValueError(
            "primitive packet lacks its exact typed setup/location/path"
        )
    return {
        "decision_hash": str(packet.get("decision_hash")),
        "hypothesis_key": hypothesis_key,
        "setup_id": setup_id,
        "entry_location_id": entry_location_id,
        "entry_path_id": entry_path_id,
    }


def _frame(
    packet: Mapping[str, Any],
    timeframe: str,
) -> Mapping[str, Any]:
    value = packet.get("observation_t", {}).get("frames", {}).get(timeframe)
    if not isinstance(value, Mapping):
        raise KeyError(f"missing {timeframe} observation frame")
    return value


def _clock_is_causal(value: Any, decision_asof: pd.Timestamp) -> bool:
    if value is None:
        return False
    timestamp = pd.Timestamp(value)
    return timestamp.tzinfo is not None and timestamp <= decision_asof


def _primitive_not_evaluable(
    *,
    definition: Mapping[str, Any],
    packet_hash: str,
    packet_sha256: str,
    decision_hash: str,
    decision_asof: pd.Timestamp,
    hypothesis_key: str,
    setup_id: str,
    entry_location_id: str,
    entry_path_id: str,
    reason: str,
) -> PrimitiveValue:
    return PrimitiveValue(
        primitive_name=str(definition["primitive_name"]),
        formula_version=str(definition["formula_version"]),
        definition_hash=_definition_hash(definition),
        decision_packet_hash=packet_hash,
        decision_packet_sha256=packet_sha256,
        decision_hash=decision_hash,
        decision_asof=decision_asof,
        hypothesis_key=hypothesis_key,
        setup_id=setup_id,
        entry_location_id=entry_location_id,
        entry_path_id=entry_path_id,
        evaluable=False,
        reason=reason,
        values={},
    )


def compute_primitive(
    issue: ReviewIssue,
    packet: Mapping[str, Any],
    *,
    hypothesis_key: str,
) -> PrimitiveValue:
    """Compute one diagnostic primitive from a verified decision packet only."""

    definition = PRIMITIVE_MAP[issue]
    packet_hash, decision_hash, decision_asof = _packet_identity(packet)
    packet_sha256 = decision_packet_payload_sha256(packet)
    packet_review_identity = _packet_review_identity(
        packet,
        hypothesis_key,
    )
    if packet_review_identity["decision_hash"] != decision_hash:
        raise ValueError("primitive packet decision identities disagree")
    hypothesis = _packet_hypothesis(packet, hypothesis_key)
    direction = str(hypothesis.get("direction"))
    if direction not in {"long", "short"}:
        raise ValueError("primitive hypothesis direction is invalid")
    direction_sign = 1 if direction == "long" else -1

    def unavailable(reason: str) -> PrimitiveValue:
        return _primitive_not_evaluable(
            definition=definition,
            packet_hash=packet_hash,
            packet_sha256=packet_sha256,
            decision_hash=decision_hash,
            decision_asof=decision_asof,
            hypothesis_key=hypothesis_key,
            setup_id=str(packet_review_identity["setup_id"]),
            entry_location_id=str(
                packet_review_identity["entry_location_id"]
            ),
            entry_path_id=str(
                packet_review_identity["entry_path_id"]
            ),
            reason=reason,
        )

    values: dict[str, Any]
    try:
        if issue is ReviewIssue.MISSED_STRUCTURE_SEQUENCE:
            sequence = hypothesis.get("sequence")
            plan = hypothesis.get("plan")
            if (
                not isinstance(sequence, Mapping)
                or not isinstance(plan, Mapping)
            ):
                return unavailable("missing_frozen_sequence_or_plan")
            hypothesis_steps = sequence.get("steps")
            if (
                not isinstance(hypothesis_steps, Sequence)
                or isinstance(hypothesis_steps, (str, bytes))
                or not hypothesis_steps
            ):
                return unavailable("missing_hypothesis_sequence_steps")
            satisfied_prefix: list[Mapping[str, Any]] = []
            unsatisfied_seen = False
            hypothesis_prefix_valid = True
            prior_clock: pd.Timestamp | None = None
            for raw_step in hypothesis_steps:
                if not isinstance(raw_step, Mapping):
                    hypothesis_prefix_valid = False
                    continue
                satisfied = bool(raw_step.get("satisfied"))
                if not satisfied:
                    unsatisfied_seen = True
                    continue
                clock_value = raw_step.get("observed_at")
                clock = (
                    None
                    if clock_value is None
                    else pd.Timestamp(clock_value)
                )
                if (
                    unsatisfied_seen
                    or clock is None
                    or clock.tzinfo is None
                    or clock > decision_asof
                    or (
                        prior_clock is not None
                        and clock < prior_clock
                    )
                ):
                    hypothesis_prefix_valid = False
                else:
                    satisfied_prefix.append(raw_step)
                    prior_clock = clock

            entry_path_id = plan.get("entry_path_id")
            paths = (
                *packet.get("observation_t", {}).get(
                    "path_sequences",
                    (),
                ),
                *packet.get("observation_t", {}).get(
                    "group5_boundary_path_transitions",
                    (),
                ),
            )
            entry_path = next(
                (
                    item
                    for item in paths
                    if isinstance(item, Mapping)
                    and item.get("sequence_id") == entry_path_id
                ),
                None,
            )
            if not isinstance(entry_path, Mapping):
                return unavailable("exact_typed_entry_path_not_visible")
            path_steps = entry_path.get("steps")
            if (
                not isinstance(path_steps, Sequence)
                or isinstance(path_steps, (str, bytes))
                or not path_steps
            ):
                return unavailable("typed_entry_path_has_no_steps")
            path_chain_valid = True
            prior_step_id: str | None = None
            prior_step_clock: pd.Timestamp | None = None
            path_step_ids: list[str] = []
            for index, raw_step in enumerate(path_steps):
                if not isinstance(raw_step, Mapping):
                    path_chain_valid = False
                    continue
                step_id = str(raw_step.get("step_id", ""))
                clock = pd.Timestamp(raw_step.get("observed_at"))
                predecessors = tuple(
                    raw_step.get("predecessor_step_ids") or ()
                )
                relation = raw_step.get("same_clock_relation")
                expected_relation = (
                    "origin"
                    if index == 0
                    else (
                        "strictly_after"
                        if (
                            prior_step_clock is not None
                            and clock > prior_step_clock
                        )
                        else relation
                    )
                )
                if (
                    not step_id
                    or step_id in path_step_ids
                    or clock.tzinfo is None
                    or clock > decision_asof
                    or (
                        index == 0
                        and (
                            predecessors
                            or relation != "origin"
                        )
                    )
                    or (
                        index > 0
                        and predecessors != (prior_step_id,)
                    )
                    or (
                        index > 0
                        and (
                            prior_step_clock is None
                            or clock < prior_step_clock
                            or (
                                clock > prior_step_clock
                                and relation != "strictly_after"
                            )
                            or (
                                clock == prior_step_clock
                                and relation
                                not in {
                                    "same_clock_known",
                                    "same_clock_unknown",
                                }
                            )
                        )
                    )
                    or raw_step.get("direction") != direction
                    or expected_relation != relation
                ):
                    path_chain_valid = False
                path_step_ids.append(step_id)
                prior_step_id = step_id
                prior_step_clock = clock
            exact_context = (
                entry_path.get("context_kind") == "zone_return"
                and entry_path.get("context_id")
                == plan.get("entry_location_id")
                and entry_path.get("direction") == direction
            )
            values = {
                "implementation_consistent": (
                    hypothesis_prefix_valid
                    and path_chain_valid
                    and exact_context
                ),
                "hypothesis_prefix_valid": hypothesis_prefix_valid,
                "typed_entry_path_chain_valid": path_chain_valid,
                "exact_entry_context": exact_context,
                "satisfied_hypothesis_step_ids": [
                    str(item.get("step_id"))
                    for item in satisfied_prefix
                ],
                "satisfied_hypothesis_step_times": [
                    item.get("observed_at")
                    for item in satisfied_prefix
                ],
                "typed_entry_path_id": entry_path_id,
                "typed_entry_path_step_ids": path_step_ids,
                "typed_entry_path_step_kinds": [
                    str(item.get("kind"))
                    for item in path_steps
                    if isinstance(item, Mapping)
                ],
                "typed_entry_path_step_times": [
                    item.get("observed_at")
                    for item in path_steps
                    if isinstance(item, Mapping)
                ],
                "path_lifecycle": entry_path.get("lifecycle"),
                "path_transition_reason": entry_path.get(
                    "transition_reason"
                ),
            }
        elif issue is ReviewIssue.PULLBACK_AS_INVALIDATION:
            plan = hypothesis.get("plan")
            if not isinstance(plan, Mapping):
                return unavailable("missing_frozen_plan")
            location_id = plan.get("entry_location_id")
            location = next(
                (
                    item
                    for item in packet.get("observation_t", {}).get(
                        "entry_locations",
                        (),
                    )
                    if item.get("location_id") == location_id
                ),
                None,
            )
            if (
                not isinstance(location, Mapping)
                or location.get("first_entered_at") is None
            ):
                return unavailable("entry_location_not_first_entered")
            entry_path_id = plan.get("entry_path_id")
            entry_path = next(
                (
                    item
                    for item in packet.get("observation_t", {}).get(
                        "path_sequences",
                        (),
                    )
                    if (
                        isinstance(item, Mapping)
                        and item.get("sequence_id") == entry_path_id
                    )
                ),
                None,
            )
            first_entered = pd.Timestamp(location["first_entered_at"])
            first_pullbacks = (
                ()
                if not isinstance(entry_path, Mapping)
                else tuple(
                    item
                    for item in entry_path.get("steps", ())
                    if (
                        isinstance(item, Mapping)
                        and item.get("kind") == "first_pullback"
                    )
                )
            )
            if (
                first_entered.tzinfo is None
                or first_entered > decision_asof
                or not isinstance(entry_path, Mapping)
                or entry_path.get("context_kind") != "zone_return"
                or entry_path.get("context_id") != location_id
                or entry_path.get("direction") != direction
                or len(first_pullbacks) != 1
                or first_pullbacks[0].get("source_entity_id")
                != location_id
                or pd.Timestamp(first_pullbacks[0].get("observed_at"))
                != first_entered
            ):
                return unavailable(
                    "exact_first_pullback_path_binding_invalid"
                )
            bars = [
                item
                for item in packet.get("histories", {}).get("1m", ())
                if pd.Timestamp(item["end"]) >= first_entered
                and pd.Timestamp(item["end"]) <= decision_asof
            ]
            if (
                not bars
                or pd.Timestamp(bars[0]["end"]) != first_entered
                or pd.Timestamp(bars[-1]["end"]) != decision_asof
                or any(
                    pd.Timestamp(right["start"])
                    != pd.Timestamp(left["end"])
                    for left, right in zip(bars[:-1], bars[1:])
                )
            ):
                return unavailable("first_entry_path_not_in_packet_history")
            invalidation = float(plan["invalidation"]["price"])
            risk_points = float(plan["risk_points"])
            if (
                risk_points <= 0
                or not math.isclose(
                    risk_points,
                    abs(float(plan["planned_entry"]) - invalidation),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                return unavailable("invalid_frozen_risk")
            extreme = (
                min(float(item["low"]) for item in bars)
                if direction_sign > 0
                else max(float(item["high"]) for item in bars)
            )
            near_edge = float(location["near_edge"])
            far_edge = float(location["far_edge"])
            denominator = near_edge - far_edge
            if abs(denominator) <= 1e-12:
                return unavailable("degenerate_entry_zone")
            penetration = clamp((near_edge - extreme) / denominator)
            failure_boundary = float(location["failure_boundary"])
            zone_close_breach = any(
                (
                    float(item["close"]) <= failure_boundary
                    if direction_sign > 0
                    else float(item["close"]) >= failure_boundary
                )
                for item in bars
            )
            thesis_stop_touched = (
                extreme <= invalidation
                if direction_sign > 0
                else extreme >= invalidation
            )
            room_R = (
                direction_sign * (extreme - invalidation) / risk_points
            )
            zone_lifecycle_intact = (
                location.get("lifecycle") != "left"
                and not zone_close_breach
            )
            execution_stop_untouched = not thesis_stop_touched
            values = {
                "room_R": room_R,
                "zone_penetration": penetration,
                "zone_intact": zone_lifecycle_intact,
                "thesis_intact": execution_stop_untouched,
                "zone_lifecycle_intact": zone_lifecycle_intact,
                "execution_stop_untouched": execution_stop_untouched,
                "normal_pullback": (
                    zone_lifecycle_intact
                    and execution_stop_untouched
                ),
                "zone_completed_close_breach": zone_close_breach,
                "thesis_stop_touched": thesis_stop_touched,
                "zone_lifecycle": location.get("lifecycle"),
                "entry_location_id": location_id,
                "entry_path_id": entry_path_id,
                "first_pullback_step_id": first_pullbacks[0].get(
                    "step_id"
                ),
                "source_zone_id": location.get("source_zone_id"),
                "frozen_zone_failure_boundary": failure_boundary,
                "frozen_invalidation_source_id": plan["invalidation"].get(
                    "source_level_id"
                ),
                "first_entered_at": location["first_entered_at"],
            }
        elif issue is ReviewIssue.LATE_DISPLACEMENT_CHASE:
            plan = hypothesis.get("plan")
            observation = packet.get("observation_t", {})
            if not isinstance(plan, Mapping):
                return unavailable("missing_frozen_plan")
            draw_id = plan.get("selected_draw_id")
            draw = next(
                (
                    item
                    for item in observation.get("liquidity_inventory", ())
                    if item.get("item_id") == draw_id
                    and item.get("lifecycle") in {"visible", "targeted"}
                ),
                None,
            )
            location = next(
                (
                    item
                    for item in observation.get("entry_locations", ())
                    if (
                        isinstance(item, Mapping)
                        and item.get("location_id")
                        == plan.get("entry_location_id")
                    )
                ),
                None,
            )
            if not isinstance(location, Mapping):
                return unavailable("exact_entry_location_not_visible")
            source_zone_id = location.get("source_zone_id")
            source_displacement_id = location.get(
                "source_displacement_id"
            )
            source_zone = next(
                (
                    item
                    for collection in (
                        "fair_value_gaps",
                        "order_blocks",
                    )
                    for item in _frame(packet, "5m").get(
                        collection,
                        (),
                    )
                    if (
                        isinstance(item, Mapping)
                        and item.get(
                            "fvg_id",
                            item.get("order_block_id"),
                        )
                        == source_zone_id
                    )
                ),
                None,
            )
            if (
                not isinstance(source_zone, Mapping)
                or source_zone.get("source_displacement_id")
                != source_displacement_id
                or source_zone.get("direction") != direction
            ):
                return unavailable(
                    "entry_zone_source_displacement_not_visible"
                )
            started_at = pd.Timestamp(
                source_zone.get("source_displacement_started_at")
            )
            active_at = pd.Timestamp(
                source_zone.get("source_displacement_active_at")
            )
            prefix_end = pd.Timestamp(
                source_zone.get("confirmed_at")
            )
            m5_prefix = [
                item
                for item in packet.get("histories", {}).get("5m", ())
                if (
                    pd.Timestamp(item["end"]) >= started_at
                    and pd.Timestamp(item["end"]) <= prefix_end
                )
            ]
            formation_atr = (
                float(source_zone["width_points"])
                / float(source_zone["width_atr"])
            )
            if (
                not isinstance(draw, Mapping)
                or formation_atr <= 0
                or not m5_prefix
                or pd.Timestamp(m5_prefix[0]["end"]) != started_at
                or pd.Timestamp(m5_prefix[-1]["end"]) != prefix_end
                or any(
                    pd.Timestamp(right["start"])
                    != pd.Timestamp(left["end"])
                    for left, right in zip(
                        m5_prefix[:-1],
                        m5_prefix[1:],
                    )
                )
                or not _clock_is_causal(started_at, decision_asof)
                or not _clock_is_causal(active_at, decision_asof)
                or not _clock_is_causal(prefix_end, decision_asof)
                or not started_at <= active_at <= prefix_end
                or draw.get("side")
                != ("above" if direction_sign > 0 else "below")
                or draw.get("lifecycle") not in {"visible", "targeted"}
                or not _clock_is_causal(
                    draw.get("confirmed_at"),
                    decision_asof,
                )
            ):
                return unavailable(
                    "draw_or_source_displacement_prefix_unavailable"
                )
            planned_entry = float(plan["planned_entry"])
            cost = float(
                observation.get("execution", {}).get(
                    "expected_round_trip_cost_points",
                    0.0,
                )
            )
            net_remaining_atr = max(
                0.0,
                (
                    direction_sign
                    * (float(draw["price"]) - planned_entry)
                    - cost
                )
                / formation_atr,
            )
            extension_atr = max(
                0.0,
                direction_sign
                * (
                    float(m5_prefix[-1]["close"])
                    - float(m5_prefix[0]["open"])
                )
                / formation_atr,
            )
            values = {
                "impulse_extension_atr": extension_atr,
                "net_remaining_draw_atr": net_remaining_atr,
                "extension_to_remaining_draw_ratio": (
                    None
                    if net_remaining_atr <= 0.0
                    else extension_atr / net_remaining_atr
                ),
                "path_exhausted": net_remaining_atr <= 0.0,
                "selected_draw_id": draw_id,
                "entry_location_id": location.get("location_id"),
                "source_zone_id": source_zone_id,
                "displacement_id": source_displacement_id,
                "source_displacement_started_at": started_at,
                "source_displacement_active_at": active_at,
                "source_displacement_prefix_end": prefix_end,
                "source_displacement_prefix_bars": len(m5_prefix),
                "formation_atr": formation_atr,
                "planned_entry": planned_entry,
                "round_trip_cost_points": cost,
            }
        elif issue is ReviewIssue.WRONG_LIQUIDITY_DRAW:
            plan = hypothesis.get("plan")
            observation = packet.get("observation_t", {})
            if not isinstance(plan, Mapping):
                return unavailable("missing_frozen_plan")
            expected_side = "above" if direction_sign > 0 else "below"
            selected_draw_id = plan.get("selected_draw_id")
            targets = plan.get("targets") or ()
            if not targets or not isinstance(targets[0], Mapping):
                return unavailable("missing_primary_target")
            primary_target_id = targets[0].get("level_id")
            draws = [
                item
                for item in observation.get("liquidity_inventory", ())
                if (
                    isinstance(item, Mapping)
                    and item.get("item_id") == selected_draw_id
                )
            ]
            if len(draws) != 1:
                return unavailable("selected_draw_not_unique_in_inventory")
            draw = draws[0]
            location = next(
                (
                    item
                    for item in observation.get("entry_locations", ())
                    if (
                        isinstance(item, Mapping)
                        and item.get("location_id")
                        == plan.get("entry_location_id")
                    )
                ),
                None,
            )
            if not isinstance(location, Mapping):
                return unavailable("exact_entry_location_not_visible")
            prior_hypothesis = (
                packet.get("belief_t_minus_1", {})
                .get("hypotheses", {})
                .get(hypothesis_key)
                if isinstance(
                    packet.get("belief_t_minus_1"),
                    Mapping,
                )
                else None
            )
            prior_plan = (
                prior_hypothesis.get("plan")
                if isinstance(prior_hypothesis, Mapping)
                else None
            )
            playbook = str(hypothesis.get("playbook"))
            expected_draw_id: str | None = None
            selection_reason: str
            if playbook == "displacement_first_pullback":
                sequence = _mapping_or_empty(
                    hypothesis.get("sequence")
                )
                background_step = next(
                    (
                        item
                        for item in sequence.get("steps", ())
                        if (
                            isinstance(item, Mapping)
                            and item.get("step_id")
                            == "h4_structure_and_draw"
                            and bool(item.get("satisfied"))
                        )
                    ),
                    None,
                )
                source_ids = (
                    ()
                    if not isinstance(background_step, Mapping)
                    else tuple(background_step.get("source_ids") or ())
                )
                matching_draw_ids = [
                    source_id
                    for source_id in source_ids
                    if any(
                        isinstance(item, Mapping)
                        and item.get("item_id") == source_id
                        and item.get("timeframe") == "4H"
                        for item in observation.get(
                            "liquidity_inventory",
                            (),
                        )
                    )
                ]
                if len(matching_draw_ids) != 1:
                    return unavailable(
                        "dfp_frozen_h4_draw_source_not_unique"
                    )
                expected_draw_id = str(matching_draw_ids[0])
                selection_reason = "dfp_frozen_h4_sequence_draw"
            elif playbook == "liquidity_sweep_reversal":
                prior_same_setup = bool(
                    isinstance(prior_plan, Mapping)
                    and prior_plan.get("playbook")
                    == "liquidity_sweep_reversal"
                    and prior_plan.get("setup_id")
                    == plan.get("setup_id")
                    and prior_plan.get("entry_location_id")
                    == plan.get("entry_location_id")
                    and prior_plan.get("selected_draw_id")
                    is not None
                )
                expected_draw_id = (
                    str(prior_plan["selected_draw_id"])
                    if prior_same_setup
                    else location.get("nearest_visible_draw_id")
                )
                selection_reason = (
                    "lsr_prior_exact_setup_draw"
                    if prior_same_setup
                    else "lsr_entry_location_nearest_draw"
                )
                if draw.get("kind") == "range_boundary":
                    return unavailable(
                        "lsr_range_boundary_draw_is_ineligible"
                    )
            else:
                return unavailable("favr_is_parked")
            planned_entry = float(plan["planned_entry"])
            draw_price = float(draw["price"])
            delivery_side_valid = (
                draw_price > planned_entry
                if direction_sign > 0
                else draw_price < planned_entry
            )
            provenance_matches = bool(
                expected_draw_id is not None
                and selected_draw_id == expected_draw_id
                and selected_draw_id == primary_target_id
                and draw.get("lifecycle") in {"visible", "targeted"}
                and draw.get("side") == expected_side
                and delivery_side_valid
                and _clock_is_causal(
                    draw.get("confirmed_at"),
                    decision_asof,
                )
            )
            values = {
                "selected_draw_id": selected_draw_id,
                "primary_target_id": primary_target_id,
                "expected_draw_id": expected_draw_id,
                "selection_reason": selection_reason,
                "provenance_matches": provenance_matches,
                "draw_lifecycle": draw.get("lifecycle"),
                "draw_timeframe": draw.get("timeframe"),
                "draw_side": draw.get("side"),
                "draw_confirmed_at": draw.get("confirmed_at"),
                "delivery_side_valid": delivery_side_valid,
                "entry_location_id": location.get("location_id"),
            }
        else:
            plan = hypothesis.get("plan")
            observation = packet.get("observation_t", {})
            sequence = hypothesis.get("sequence")
            if not isinstance(plan, Mapping):
                return unavailable("missing_frozen_plan")
            source_id = plan.get("invalidation", {}).get(
                "source_level_id"
            )
            playbook = str(hypothesis.get("playbook"))
            paths = (
                *observation.get("path_sequences", ()),
                *observation.get(
                    "group5_boundary_path_transitions",
                    (),
                ),
            )
            setup_path = next(
                (
                    item
                    for item in paths
                    if (
                        isinstance(item, Mapping)
                        and item.get("sequence_id")
                        == plan.get("setup_id")
                    )
                ),
                None,
            )
            location = next(
                (
                    item
                    for item in observation.get("entry_locations", ())
                    if (
                        isinstance(item, Mapping)
                        and item.get("location_id")
                        == plan.get("entry_location_id")
                    )
                ),
                None,
            )
            source: Mapping[str, Any]
            source_type: str
            semantic_price: Any
            semantic_clock: Any
            playbook_source_binding: bool
            if playbook == "displacement_first_pullback":
                matches = [
                    item
                    for item in observation.get("entry_locations", ())
                    if (
                        isinstance(item, Mapping)
                        and item.get("location_id") == source_id
                    )
                ]
                if len(matches) != 1:
                    return unavailable(
                        "dfp_invalidation_location_not_unique"
                    )
                source = matches[0]
                source_type = "entry_location"
                semantic_price = source.get("failure_boundary")
                semantic_clock = source.get("formed_at")
                playbook_source_binding = bool(
                    source_id == plan.get("entry_location_id")
                    and plan.get("setup_id")
                    == plan.get("entry_path_id")
                    and isinstance(setup_path, Mapping)
                    and setup_path.get("context_kind")
                    == "zone_return"
                )
            elif playbook == "liquidity_sweep_reversal":
                if (
                    not isinstance(setup_path, Mapping)
                    or setup_path.get("context_kind")
                    != "pool_reversal"
                ):
                    return unavailable(
                        "lsr_setup_pool_path_not_visible"
                    )
                manipulation_id = setup_path.get("context_id")
                matches = [
                    item
                    for item in (
                        *observation.get("manipulations", ()),
                        *observation.get(
                            "group4_boundary_manipulation_transitions",
                            (),
                        ),
                    )
                    if (
                        isinstance(item, Mapping)
                        and item.get("manipulation_id")
                        == manipulation_id
                    )
                ]
                if len(matches) != 1:
                    return unavailable(
                        "lsr_invalidation_manipulation_not_unique"
                    )
                source = matches[0]
                source_type = "manipulation"
                semantic_price = source.get("sweep_extreme")
                semantic_clock = source.get("swept_at")
                playbook_source_binding = bool(
                    source_id == manipulation_id
                    and source.get("source_kind")
                    == "formed_liquidity_pool"
                )
            else:
                return unavailable("favr_is_parked")
            invalidation = float(plan["invalidation"]["price"])
            source_price_matches = (
                semantic_price is not None
                and math.isclose(
                    float(semantic_price),
                    invalidation,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            )
            expected_side = "below" if direction_sign > 0 else "above"
            identities_agree = (
                isinstance(sequence, Mapping)
                and plan.get("setup_id") == sequence.get("setup_id")
                and plan.get("setup_id")
                == hypothesis.get("setup_context_id")
                and plan.get("entry_location_id")
                == hypothesis.get("entry_location_id")
                and plan.get("entry_path_id")
                == packet_review_identity.get("entry_path_id")
                and plan.get("entry_path_id") is not None
            )
            clock_causal = _clock_is_causal(
                semantic_clock,
                decision_asof,
            )
            plan_clock_matches = bool(
                semantic_clock is not None
                and pd.Timestamp(
                    plan.get("invalidation", {}).get("observed_at")
                )
                == pd.Timestamp(semantic_clock)
            )
            side_matches = (
                plan.get("invalidation", {}).get("side")
                == expected_side
            )
            risk_points = float(plan["risk_points"])
            risk_matches = bool(
                risk_points > 0.0
                and math.isclose(
                    risk_points,
                    abs(
                        float(plan["planned_entry"])
                        - invalidation
                    ),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            )
            values = {
                "valid": (
                    source_price_matches
                    and clock_causal
                    and plan_clock_matches
                    and side_matches
                    and identities_agree
                    and playbook_source_binding
                    and risk_matches
                ),
                "source_type": source_type,
                "source_id": source_id,
                "source_clock": semantic_clock,
                "source_stop_price": semantic_price,
                "source_price_matches": source_price_matches,
                "source_clock_causal": clock_causal,
                "plan_clock_matches_source": plan_clock_matches,
                "side_matches": side_matches,
                "setup_location_path_identities_agree": identities_agree,
                "playbook_source_binding": playbook_source_binding,
                "risk_geometry_matches": risk_matches,
                "distance_R": (
                    abs(float(plan["planned_entry"]) - invalidation)
                    / risk_points
                ),
            }
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        return unavailable(f"missing_or_invalid_input:{type(exc).__name__}")
    return PrimitiveValue(
        primitive_name=str(definition["primitive_name"]),
        formula_version=str(definition["formula_version"]),
        definition_hash=_definition_hash(definition),
        decision_packet_hash=packet_hash,
        decision_packet_sha256=packet_sha256,
        decision_hash=decision_hash,
        decision_asof=decision_asof,
        hypothesis_key=hypothesis_key,
        setup_id=str(packet_review_identity["setup_id"]),
        entry_location_id=str(
            packet_review_identity["entry_location_id"]
        ),
        entry_path_id=str(packet_review_identity["entry_path_id"]),
        evaluable=True,
        reason="computed_from_completed_decision_packet",
        values=values,
    )


FORBIDDEN_REVIEW_KEYS = {
    "action",
    "recommended_action",
    "trade_label",
    "buy_label",
    "sell_label",
    "should_enter",
    "should_exit",
    "profit_label",
    "pnl",
    "future_path",
}


def _scan_forbidden(value: Any, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            child = f"{path}.{name}" if path else name
            if name in FORBIDDEN_REVIEW_KEYS:
                failures.append(child)
            failures.extend(_scan_forbidden(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            failures.extend(_scan_forbidden(item, f"{path}[{index}]"))
    return failures


def ai_review_identity(
    snapshot: EngineSnapshot,
    hypothesis_key: str | None = None,
) -> dict[str, str | None]:
    """Return the exact pre-reveal setup identity an AI review must echo."""

    selected_key = (
        hypothesis_key
        if hypothesis_key is not None
        else snapshot.decision.best_hypothesis_key
    )
    if selected_key is None:
        return {
            "decision_hash": snapshot.snapshot_hash,
            "hypothesis_key": None,
            "setup_id": None,
            "entry_location_id": None,
            "entry_path_id": None,
        }
    belief = snapshot.belief.hypotheses.get(selected_key)
    if belief is None:
        raise ValueError("AI review hypothesis identity is absent")
    plan = belief.plan
    sequence_setup_id = (
        None if belief.sequence is None else belief.sequence.setup_id
    )
    plan_setup_id = None if plan is None else plan.setup_id
    if (
        sequence_setup_id is not None
        and plan_setup_id is not None
        and sequence_setup_id != plan_setup_id
    ):
        raise ValueError("AI review setup identities disagree")
    entry_location_id = (
        belief.entry_location_id
        if belief.entry_location_id is not None
        else None if plan is None else plan.entry_location_id
    )
    if (
        plan is not None
        and belief.entry_location_id is not None
        and plan.entry_location_id is not None
        and belief.entry_location_id != plan.entry_location_id
    ):
        raise ValueError("AI review entry-location identities disagree")
    return {
        "decision_hash": snapshot.snapshot_hash,
        "hypothesis_key": selected_key,
        "setup_id": sequence_setup_id or plan_setup_id,
        "entry_location_id": entry_location_id,
        "entry_path_id": None if plan is None else plan.entry_path_id,
    }


class AIReviewAdapter:
    """Accepts diagnostic issue codes, never an AI-selected trading action."""

    def convert(
        self,
        review: Mapping[str, Any],
        snapshot: EngineSnapshot,
        *,
        hypothesis_key: str | None = None,
        decision_packet_hash: str,
        decision_packet_sha256: str,
        decision_packet: Mapping[str, Any],
    ) -> tuple[PrimitiveProposal, ...]:
        forbidden = _scan_forbidden(review)
        if forbidden:
            raise ValueError(
                "AI review contains forbidden action/outcome fields: "
                + ", ".join(forbidden)
            )
        identity = ai_review_identity(snapshot, hypothesis_key)
        if identity["hypothesis_key"] is None:
            raise ValueError(
                "AI primitive review requires an explicit hypothesis"
            )
        if any(
            not isinstance(identity.get(name), str)
            or not identity.get(name)
            for name in (
                "setup_id",
                "entry_location_id",
                "entry_path_id",
            )
        ):
            raise ValueError(
                "AI primitive review requires a complete frozen setup"
            )
        identity_fields = set(identity)
        packet_identity = {
            "decision_packet_hash": decision_packet_hash,
            "decision_packet_sha256": decision_packet_sha256,
        }
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in packet_identity.values()
        ):
            raise ValueError(
                "AI review requires valid frozen decision-packet hashes"
            )
        verified_packet_hash, packet_decision_hash, _ = _packet_identity(
            decision_packet
        )
        verified_packet_sha256 = decision_packet_payload_sha256(
            decision_packet
        )
        sealed_identity = _packet_review_identity(
            decision_packet,
            str(identity["hypothesis_key"]),
        )
        if (
            verified_packet_hash != decision_packet_hash
            or verified_packet_sha256 != decision_packet_sha256
            or packet_decision_hash != snapshot.snapshot_hash
            or sealed_identity != identity
        ):
            raise ValueError(
                "AI review packet differs from the exact reviewed hypothesis"
            )
        if set(review) - {
            "issues",
            "reviewer_id",
            *identity_fields,
            *packet_identity,
        }:
            raise ValueError("AI review contains unregistered top-level fields")
        missing_identity = (
            identity_fields | set(packet_identity)
        ) - set(review)
        if missing_identity:
            raise ValueError(
                "AI review lacks frozen identity fields: "
                + ", ".join(sorted(missing_identity))
            )
        if any(review.get(name) != expected for name, expected in identity.items()):
            raise ValueError(
                "AI review is not bound to the sealed decision and setup"
            )
        if any(
            review.get(name) != expected
            for name, expected in packet_identity.items()
        ):
            raise ValueError(
                "AI review is not bound to the reviewed decision packet"
            )
        issues = review.get("issues")
        if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)):
            raise ValueError("AI review issues must be a list")
        proposals: list[PrimitiveProposal] = []
        seen: set[ReviewIssue] = set()
        for item in issues:
            if not isinstance(item, Mapping):
                raise ValueError("each AI issue must be an object")
            if set(item) - {"code", "confidence", "note"}:
                raise ValueError("AI issue contains an unregistered field")
            issue = ReviewIssue(str(item.get("code")))
            if issue in seen:
                raise ValueError(f"duplicate AI review issue: {issue.value}")
            seen.add(issue)
            confidence = clamp(float(item.get("confidence", 0.0)))
            note = str(item.get("note", "")).strip()
            forbidden_phrases = (
                "should buy",
                "should sell",
                "should enter",
                "should exit",
                "go long",
                "go short",
                "buy now",
                "sell now",
            )
            if any(phrase in note.lower() for phrase in forbidden_phrases):
                raise ValueError("AI issue note contains a direct trading recommendation")
            definition = PRIMITIVE_MAP[issue]
            definition_hash = _definition_hash(definition)
            origin_value = compute_primitive(
                issue,
                decision_packet,
                hypothesis_key=identity["hypothesis_key"],
            )
            raw = (
                f"{snapshot.snapshot_hash}|{identity['hypothesis_key']}|"
                f"{identity['setup_id']}|{identity['entry_location_id']}|"
                f"{identity['entry_path_id']}|{decision_packet_hash}|"
                f"{decision_packet_sha256}|{issue.value}|"
                f"{definition['primitive_name']}|{definition_hash}"
            )
            proposals.append(
                PrimitiveProposal(
                    proposal_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
                    issue=issue,
                    confidence=confidence,
                    primitive_name=definition["primitive_name"],
                    formula_version=definition["formula_version"],
                    definition_hash=definition_hash,
                    required_inputs=definition["required_inputs"],
                    formula=definition["formula"],
                    clock_rule=definition["clock_rule"],
                    path_test=definition["path_test"],
                    status="unvalidated",
                    decision_hash=snapshot.snapshot_hash,
                    hypothesis_key=identity["hypothesis_key"],
                    setup_id=identity["setup_id"],
                    entry_location_id=identity["entry_location_id"],
                    entry_path_id=identity["entry_path_id"],
                    decision_packet_hash=decision_packet_hash,
                    decision_packet_sha256=decision_packet_sha256,
                    origin_value=origin_value,
                    note=note,
                )
            )
        return tuple(proposals)


class CausalPrimitiveRegistry:
    """Outcome-free registry; registration grants no model or trading authority."""

    def __init__(self) -> None:
        self._proposals: dict[str, PrimitiveProposal] = {}

    def register(self, proposals: Sequence[PrimitiveProposal]) -> None:
        forbidden_markers = ("pnl", "profit", "future", "winner", "buy", "sell")
        for proposal in proposals:
            searchable = " ".join(
                (
                    proposal.primitive_name,
                    proposal.formula,
                    proposal.clock_rule,
                )
            ).lower()
            if any(marker in searchable for marker in forbidden_markers):
                raise ValueError("primitive proposal contains an outcome/action marker")
            if proposal.status != "unvalidated":
                raise ValueError("AI proposals can only enter as unvalidated")
            if (
                proposal.definition_hash
                != _definition_hash(PRIMITIVE_MAP[proposal.issue])
                or proposal.origin_value.definition_hash
                != proposal.definition_hash
                or proposal.origin_value.formula_version
                != proposal.formula_version
                or proposal.origin_value.primitive_name
                != proposal.primitive_name
                or proposal.origin_value.decision_packet_hash
                != proposal.decision_packet_hash
                or proposal.origin_value.decision_packet_sha256
                != proposal.decision_packet_sha256
                or proposal.origin_value.hypothesis_key
                != proposal.hypothesis_key
                or proposal.origin_value.setup_id
                != proposal.setup_id
                or proposal.origin_value.entry_location_id
                != proposal.entry_location_id
                or proposal.origin_value.entry_path_id
                != proposal.entry_path_id
            ):
                raise ValueError(
                    "AI proposal definition or origin computation is not bound"
                )
            self._proposals[proposal.proposal_id] = proposal

    def pending(self) -> tuple[PrimitiveProposal, ...]:
        return tuple(self._proposals.values())


__all__ = [
    "AIReviewAdapter",
    "CausalPrimitiveRegistry",
    "PrimitiveValue",
    "PrimitiveProposal",
    "ReviewIssue",
    "ai_review_identity",
    "compute_primitive",
]
