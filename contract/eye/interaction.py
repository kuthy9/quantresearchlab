"""Eye interaction facts: physical zone contact, reacceptance and micro breaks.

``InteractionUpdate`` is the canonical physical payload; the Brain
interprets it rather than re-detecting it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
import weakref
from typing import Any, ClassVar
import pandas as pd

from contract.market.primitives import Direction, _exact_dataclass_pickle_state, _restore_exact_dataclass_pickle_state, aware_timestamp
from contract.eye.vocabulary import BOSScope, GROUP5_CONTEXT_KINDS, GROUP5_HARD_BOUNDARY_REASONS, GROUP5_PATH_STEP_KINDS, GROUP5_SAME_CLOCK_RELATIONS, INTERACTION_PHYSICAL_PATH_STEP_KINDS, INTERACTION_UPDATE_SCHEMA_VERSION, PathSequenceLifecycle, ReacceptanceLifecycle, _INTERACTION_PHYSICAL_PATH_REASONS, _INTERACTION_PHYSICAL_PATH_STEP_REASONS
from contract.eye.entities import EntryLocationState, ReacceptanceState


@dataclass(frozen=True)
class MicroBreakFact:
    """Exact confirmed M1 break bound to an interaction clock.

    This is an Eye fact.  It deliberately carries neither setup
    qualification nor an aligned/opposed outcome; those are Brain-owned
    interpretations of the raw direction and ordering fields below.
    ``reference_id`` retains the frozen Group 5 identity preimage.
    """

    reference_id: str
    protocol_hash: str
    context_kind: str
    context_id: str
    context_direction: Direction
    anchor_at: pd.Timestamp
    bos_id: str
    bos_direction: Direction
    target_swing_id: str
    scope: BOSScope
    pending_at: pd.Timestamp
    resolved_at: pd.Timestamp
    relation: str
    strength: float

    def __post_init__(self) -> None:
        if (
            not self.reference_id
            or not self.protocol_hash
            or self.context_kind not in GROUP5_CONTEXT_KINDS
            or not self.context_id
            or not isinstance(self.context_direction, Direction)
            or not self.bos_id
            or not isinstance(self.bos_direction, Direction)
            or not self.target_swing_id
            or not isinstance(self.scope, BOSScope)
            or self.relation not in {"strictly_after", "same_clock_unknown"}
            or not math.isfinite(float(self.strength))
            or not 0.0 <= self.strength <= 1.0
        ):
            raise ValueError("micro-break fact is invalid")
        for name in ("anchor_at", "pending_at", "resolved_at"):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"micro_break.{name}",
                ),
            )
        if (
            self.pending_at >= self.resolved_at
            or self.resolved_at < self.anchor_at
            or (
                self.relation == "strictly_after"
                and self.resolved_at <= self.anchor_at
            )
            or (
                self.relation == "same_clock_unknown"
                and self.resolved_at != self.anchor_at
            )
        ):
            raise ValueError("micro-break clocks are invalid")


@dataclass(frozen=True)
class MicroBOSReference:
    """A non-recomputed reference to an exact upstream confirmed M1 BOS."""

    reference_id: str
    protocol_hash: str
    context_kind: str
    context_id: str
    expected_direction: Direction
    anchor_at: pd.Timestamp
    bos_id: str
    bos_direction: Direction
    target_swing_id: str
    scope: BOSScope
    pending_at: pd.Timestamp
    resolved_at: pd.Timestamp
    relation: str
    outcome: str
    qualified: bool
    strength: float

    def __post_init__(self) -> None:
        if (
            not self.reference_id
            or not self.protocol_hash
            or self.context_kind not in GROUP5_CONTEXT_KINDS
            or not self.context_id
            or not isinstance(self.expected_direction, Direction)
            or not self.bos_id
            or not isinstance(self.bos_direction, Direction)
            or not self.target_swing_id
            or not isinstance(self.scope, BOSScope)
            or self.relation not in {"strictly_after", "same_clock_unknown"}
            or self.outcome
            not in {
                "aligned",
                "opposed",
                "simultaneous_unknown",
                "ambiguous_same_clock",
            }
            or type(self.qualified) is not bool
            or not math.isfinite(float(self.strength))
            or not 0.0 <= self.strength <= 1.0
        ):
            raise ValueError("micro-BOS reference is invalid")
        for name in ("anchor_at", "pending_at", "resolved_at"):
            object.__setattr__(
                self,
                name,
                aware_timestamp(
                    getattr(self, name),
                    name=f"micro_bos.{name}",
                ),
            )
        if (
            self.pending_at >= self.resolved_at
            or self.resolved_at < self.anchor_at
            or (
                self.relation == "strictly_after"
                and self.resolved_at <= self.anchor_at
            )
            or (
                self.relation == "same_clock_unknown"
                and self.resolved_at != self.anchor_at
            )
            or self.qualified
            != (
                self.relation == "strictly_after"
                and self.outcome == "aligned"
            )
            or (
                self.outcome == "aligned"
                and self.bos_direction is not self.expected_direction
            )
            or (
                self.outcome == "opposed"
                and self.bos_direction is self.expected_direction
            )
            or (
                self.outcome == "simultaneous_unknown"
                and self.relation != "same_clock_unknown"
            )
            or (
                self.outcome != "simultaneous_unknown"
                and self.relation == "same_clock_unknown"
            )
            or (
                self.outcome == "ambiguous_same_clock"
                and self.relation != "strictly_after"
            )
        ):
            raise ValueError("micro-BOS reference clocks or outcome are invalid")


@dataclass(frozen=True)
class PathSequenceStep:
    """One immutable, source-bound milestone in a Group 5 context."""

    step_id: str
    kind: str
    observed_at: pd.Timestamp
    source_event_id: str | None
    source_entity_id: str
    predecessor_step_ids: tuple[str, ...]
    same_clock_relation: str
    direction: Direction
    strength: float
    reason: str
    source_active_at: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(
                self.observed_at,
                name="path_step.observed_at",
            ),
        )
        object.__setattr__(
            self,
            "predecessor_step_ids",
            tuple(self.predecessor_step_ids),
        )
        if self.source_active_at is not None:
            object.__setattr__(
                self,
                "source_active_at",
                aware_timestamp(
                    self.source_active_at,
                    name="path_step.source_active_at",
                ),
            )
        if (
            not self.step_id
            or self.kind not in GROUP5_PATH_STEP_KINDS
            or self.source_event_id == ""
            or not self.source_entity_id
            or len(self.predecessor_step_ids)
            != len(set(self.predecessor_step_ids))
            or any(not value for value in self.predecessor_step_ids)
            or self.same_clock_relation
            not in GROUP5_SAME_CLOCK_RELATIONS
            or not isinstance(self.direction, Direction)
            or not math.isfinite(float(self.strength))
            or not 0.0 <= self.strength <= 1.0
            or not self.reason
            or (
                self.kind == "opposite_displacement"
                and (
                    self.source_event_id is None
                    or self.source_active_at is None
                )
            )
            or (
                self.kind != "opposite_displacement"
                and self.source_active_at is not None
            )
            or (
                self.source_active_at is not None
                and self.source_active_at > self.observed_at
            )
        ):
            raise ValueError("path-sequence step is invalid")


@dataclass(frozen=True)
class PathSequenceState:
    """A bounded ordered path; it is not a scalar score or action label."""

    sequence_id: str
    protocol_hash: str
    symbol: str
    instrument_id: int
    context_kind: str
    context_id: str
    direction: Direction
    lifecycle: PathSequenceLifecycle
    formed_at: pd.Timestamp
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    age_real_1m_bars: int
    state_duration_real_1m_bars: int
    steps: tuple[PathSequenceStep, ...]
    ended_at: pd.Timestamp | None = None
    transition_reason: str = "context_registered"

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if (
            not self.sequence_id
            or not self.protocol_hash
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.context_kind not in GROUP5_CONTEXT_KINDS
            or not self.context_id
            or not isinstance(self.direction, Direction)
            or not isinstance(self.lifecycle, PathSequenceLifecycle)
            or type(self.age_real_1m_bars) is not int
            or self.age_real_1m_bars < 0
            or type(self.state_duration_real_1m_bars) is not int
            or self.state_duration_real_1m_bars < 0
            or not self.steps
            or not self.transition_reason
        ):
            raise ValueError("path-sequence state identity is invalid")
        for name in (
            "formed_at",
            "state_started_at",
            "last_updated_at",
            "ended_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"path_sequence.{name}"),
                )
        step_ids = tuple(step.step_id for step in self.steps)
        if (
            self.state_started_at < self.formed_at
            or self.last_updated_at < self.state_started_at
            or len(step_ids) != len(set(step_ids))
            or self.steps[0].observed_at != self.formed_at
            or any(
                left.observed_at > right.observed_at
                for left, right in zip(self.steps, self.steps[1:])
            )
            or any(
                step.observed_at > self.last_updated_at
                or (
                    index == 0
                    and (
                        step.predecessor_step_ids
                        or step.same_clock_relation != "origin"
                    )
                )
                or (
                    index > 0
                    and step.predecessor_step_ids
                    != (step_ids[index - 1],)
                )
                or (
                    index > 0
                    and step.observed_at
                    > self.steps[index - 1].observed_at
                    and step.same_clock_relation != "strictly_after"
                )
                or (
                    index > 0
                    and step.observed_at
                    == self.steps[index - 1].observed_at
                    and step.same_clock_relation
                    not in {"same_clock_known", "same_clock_unknown"}
                )
                for index, step in enumerate(self.steps)
            )
        ):
            raise ValueError("path-sequence history is invalid")
        if self.lifecycle is PathSequenceLifecycle.ACTIVE:
            if self.ended_at is not None:
                raise ValueError("active path sequence cannot be ended")
        elif (
            self.ended_at is None
            or self.ended_at != self.state_started_at
            or self.last_updated_at != self.ended_at
        ):
            raise ValueError("terminal path sequence clocks are invalid")
        if (
            self.lifecycle is PathSequenceLifecycle.CENSORED
            and self.transition_reason not in GROUP5_HARD_BOUNDARY_REASONS
        ):
            raise ValueError("censored path sequence reason is invalid")


# DTO instances that have passed ``exact_values`` once, with the field values
# they were admitted with.  Every nested DTO is a frozen dataclass, so an
# instance that was canonical stays canonical unless something bypasses the
# freeze; comparing the remembered values catches that at a fraction of the
# cost of reconstructing the object through its validators.  Keyed by ``id``
# with a weak reference so a freed object cannot leave an entry a recycled id
# could hit, and the callback removes the entry when the object is collected.
_ADMITTED: dict[int, tuple["weakref.ReferenceType[Any]", tuple[Any, ...]]] = {}


def _is_admitted(value: Any, names: tuple[str, ...] = ()) -> bool:
    entry = _ADMITTED.get(id(value))
    if entry is None or entry[0]() is not value:
        return False
    remembered = entry[1]
    names = names or tuple(item.name for item in fields(type(value)))
    return all(
        current is expected
        or (
            type(current) is type(expected)
            and current == expected
            # A clock that lost or changed its zone compares equal on the
            # wall time; the admitted value's zone is part of what was admitted.
            and getattr(current, "tzinfo", None) == getattr(expected, "tzinfo", None)
        )
        for current, expected in zip(
            (getattr(value, name) for name in names), remembered, strict=True
        )
    )


def _remember_admitted(value: Any, names: tuple[str, ...]) -> None:
    key = id(value)

    def _forget(_reference: Any, key: int = key) -> None:
        _ADMITTED.pop(key, None)

    _ADMITTED[key] = (
        weakref.ref(value, _forget),
        tuple(getattr(value, name) for name in names),
    )


@dataclass(frozen=True)
class InteractionUpdate:
    """Canonical Eye output for physical zone/pool interaction facts.

    The reducer owns ordering and source binding only.  This canonical DTO has
    no Brain interpretation or legacy entry-qualification properties.
    """

    zone_interactions: tuple[EntryLocationState, ...]
    reacceptance_interactions: tuple[ReacceptanceState, ...]
    micro_break_facts: tuple[MicroBreakFact, ...]
    interaction_paths: tuple[PathSequenceState, ...]
    interaction_path_transitions: tuple[PathSequenceState, ...] = ()
    reacceptance_interaction_transitions: tuple[
        ReacceptanceState,
        ...,
    ] = ()
    milestone_transitions: tuple[tuple[str, PathSequenceStep], ...] = ()
    cold_source_ids: tuple[str, ...] = ()
    boundary_reason: str | None = None

    schema_version: ClassVar[int] = INTERACTION_UPDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "zone_interactions",
            "reacceptance_interactions",
            "micro_break_facts",
            "interaction_paths",
            "interaction_path_transitions",
            "reacceptance_interaction_transitions",
            "milestone_transitions",
            "cold_source_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self.validate_canonical_bindings()

    def validate_canonical_bindings(self) -> None:
        """Fail closed unless every physical fact has one exact owner.

        This is the single cross-record validator used both at DTO admission
        and immediately before Brain interpretation.  It deliberately checks
        only the bounded current interaction graph (or one boundary batch),
        never retained reducer history.
        """

        if set(self.__dict__) != {
            item.name for item in fields(InteractionUpdate)
        }:
            raise ValueError("interaction update shape changed")

        def exact_values(values: tuple[Any, ...], kind: type, label: str) -> None:
            """Re-admit every nested DTO through its sole canonical contract."""

            names = tuple(item.name for item in fields(kind))
            expected = set(names)
            for value in values:
                if _is_admitted(value, names):
                    continue
                if (
                    type(value) is not kind
                    or set(getattr(value, "__dict__", ())) != expected
                ):
                    raise ValueError(f"interaction {label} shape changed")
                state = {name: getattr(value, name) for name in names}
                try:
                    canonical = kind(**state)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"interaction {label} canonical state changed"
                    ) from error
                if any(
                    type(state[name]) is not type(getattr(canonical, name))
                    or state[name] != getattr(canonical, name)
                    for name in names
                ):
                    raise ValueError(
                        f"interaction {label} canonical state changed"
                    )
                _remember_admitted(value, names)

        typed_collections = (
            (self.zone_interactions, EntryLocationState, "zone"),
            (
                self.reacceptance_interactions,
                ReacceptanceState,
                "reacceptance",
            ),
            (self.micro_break_facts, MicroBreakFact, "micro-break"),
            (self.interaction_paths, PathSequenceState, "path"),
            (
                self.interaction_path_transitions,
                PathSequenceState,
                "path transition",
            ),
            (
                self.reacceptance_interaction_transitions,
                ReacceptanceState,
                "reacceptance transition",
            ),
        )
        for values, kind, label in typed_collections:
            exact_values(values, kind, label)
        all_paths = (*self.interaction_paths, *self.interaction_path_transitions)
        exact_values(
            tuple(step for path in all_paths for step in path.steps),
            PathSequenceStep,
            "path step",
        )
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or type(pair[1]) is not PathSequenceStep
            for pair in self.milestone_transitions
        ):
            raise ValueError("interaction milestone transition shape changed")
        exact_values(
            tuple(pair[1] for pair in self.milestone_transitions),
            PathSequenceStep,
            "milestone",
        )
        if any(type(value) is not str or not value for value in self.cold_source_ids):
            raise ValueError("interaction cold source identity changed")
        if self.cold_source_ids != tuple(sorted(set(self.cold_source_ids))):
            raise ValueError(
                "interaction cold source identities must be unique and sorted"
            )
        if (
            self.boundary_reason is not None
            and self.boundary_reason not in GROUP5_HARD_BOUNDARY_REASONS
        ):
            raise ValueError("interaction update has an unregistered boundary")
        if self.boundary_reason is None and self.reacceptance_interaction_transitions:
            raise ValueError(
                "interaction reacceptance censor requires a hard boundary"
            )
        if any(
            state.lifecycle is not ReacceptanceLifecycle.CENSORED
            or state.censored_at is None
            or state.transition_reason != "hard_boundary_censored"
            for state in self.reacceptance_interaction_transitions
        ):
            raise ValueError(
                "interaction boundary reacceptance transition is invalid"
            )
        if self.boundary_reason is not None and any(
            (
                self.zone_interactions,
                self.reacceptance_interactions,
                self.micro_break_facts,
                self.interaction_paths,
                self.milestone_transitions,
                self.cold_source_ids,
            )
        ):
            raise ValueError("interaction boundary cannot publish current entities")

        def index_by(
            values: tuple[Any, ...], attribute: str, label: str
        ) -> dict[Any, Any]:
            output = {getattr(value, attribute): value for value in values}
            if len(output) != len(values):
                raise ValueError(f"interaction {label} identities repeat")
            return output

        def index_by_context(
            values: tuple[Any, ...], label: str
        ) -> dict[tuple[str, str], Any]:
            output = {
                (value.context_kind, value.context_id): value for value in values
            }
            if len(output) != len(values):
                raise ValueError(f"interaction {label} contexts repeat")
            return output

        locations = index_by(self.zone_interactions, "location_id", "zone")
        index_by(self.reacceptance_interactions, "reacceptance_id", "reacceptance")
        index_by(self.micro_break_facts, "reference_id", "micro-break")
        paths = index_by(self.interaction_paths, "sequence_id", "path")
        index_by(
            self.interaction_path_transitions, "sequence_id", "path transition"
        )
        index_by(
            self.reacceptance_interaction_transitions,
            "reacceptance_id",
            "reacceptance transition",
        )
        paths_by_context = index_by_context(self.interaction_paths, "path")
        transition_paths_by_context = index_by_context(
            self.interaction_path_transitions,
            "path transition",
        )
        index_by_context(self.reacceptance_interactions, "reacceptance")
        index_by_context(
            self.reacceptance_interaction_transitions,
            "reacceptance transition",
        )
        milestone_keys = tuple(
            (sequence_id, step.step_id)
            for sequence_id, step in self.milestone_transitions
        )
        if len(milestone_keys) != len(set(milestone_keys)):
            raise ValueError("interaction milestone transitions repeat")
        steps_by_path = {
            path.sequence_id: {step.step_id: step for step in path.steps}
            for path in self.interaction_paths
        }
        if any(
            steps_by_path.get(sequence_id, {}).get(step.step_id) != step
            for sequence_id, step in self.milestone_transitions
        ):
            raise ValueError("interaction milestone lacks its exact current path")
        step_ordinals_by_path = {
            path.sequence_id: {
                step.step_id: ordinal
                for ordinal, step in enumerate(path.steps)
            }
            for path in self.interaction_paths
        }
        last_milestone_ordinal: dict[str, int] = {}
        for sequence_id, step in self.milestone_transitions:
            ordinal = step_ordinals_by_path[sequence_id][step.step_id]
            if ordinal <= last_milestone_ordinal.get(sequence_id, -1):
                raise ValueError(
                    "interaction milestones are not in path predecessor order"
                )
            last_milestone_ordinal[sequence_id] = ordinal

        def validate_path(path: PathSequenceState, *, current: bool) -> None:
            if (
                path.transition_reason not in _INTERACTION_PHYSICAL_PATH_REASONS
                or any(
                    step.direction is not path.direction
                    or step.kind not in INTERACTION_PHYSICAL_PATH_STEP_KINDS
                    or step.reason
                    not in _INTERACTION_PHYSICAL_PATH_STEP_REASONS[step.kind]
                    for step in path.steps
                )
            ):
                raise ValueError(
                    "interaction path physical vocabulary or direction changed"
                )
            if path.context_kind == "zone_return":
                location = locations.get(path.context_id)
                if current and location is None:
                    raise ValueError("zone-return path lacks its exact zone")
                if (
                    path.steps[0].kind != "zone_visible"
                    or path.steps[0].source_event_id is None
                    or path.steps[0].source_event_id
                    != path.steps[0].source_entity_id
                    or (
                        current
                        and (
                            path.protocol_hash != location.protocol_hash
                            or path.symbol != location.symbol
                            or path.instrument_id != location.instrument_id
                            or path.direction is not location.direction
                            or path.formed_at != location.formed_at
                            or path.steps[0].source_event_id
                            != location.source_zone_id
                        )
                    )
                ):
                    raise ValueError("zone-return path custody changed")
            elif (
                path.steps[0].kind != "pool_swept"
                or path.steps[0].source_event_id != path.context_id
                or path.steps[0].source_entity_id != path.context_id
            ):
                raise ValueError("pool-reversal path custody changed")

        for path in self.interaction_paths:
            validate_path(path, current=True)
        if {
            path.context_id
            for path in self.interaction_paths
            if path.context_kind == "zone_return"
        } != set(locations):
            raise ValueError("interaction zones and paths differ")

        reference_steps: dict[tuple[str, str], PathSequenceStep] = {}
        micro_steps: dict[
            tuple[str, str, str, pd.Timestamp],
            PathSequenceStep,
        ] = {}
        pool_anchors: dict[tuple[str, str], pd.Timestamp] = {}
        for path in self.interaction_paths:
            context = (path.context_kind, path.context_id)
            opposite_steps = 0
            for step in path.steps:
                if step.kind == "reference_left":
                    key = (path.context_id, step.source_entity_id)
                    if key in reference_steps:
                        raise ValueError("interaction reference milestone repeats")
                    reference_steps[key] = step
                elif step.kind == "micro_break_observed":
                    if step.source_event_id is None:
                        raise ValueError("physical micro-break lacks its BOS source")
                    key = (*context, step.source_event_id, step.observed_at)
                    if key in micro_steps:
                        raise ValueError("physical micro-break source repeats")
                    micro_steps[key] = step
                elif step.kind == "opposite_displacement":
                    opposite_steps += 1
                    pool_anchors[context] = step.observed_at
            if path.context_kind == "pool_reversal" and opposite_steps != 1:
                pool_anchors.pop(context, None)

        reacceptance_keys: set[tuple[str, str]] = set()
        for state in self.reacceptance_interactions:
            path = paths_by_context.get(("zone_return", state.context_id))
            location = locations.get(state.context_id)
            reference = reference_steps.get(
                (state.context_id, state.reacceptance_id)
            )
            if (
                path is None
                or location is None
                or state.protocol_hash != path.protocol_hash
                or state.symbol != path.symbol
                or state.instrument_id != path.instrument_id
                or state.direction is not path.direction
                or state.source_entity_id != location.source_zone_id
                or state.reference_price != location.near_edge
                or state.failure_boundary != location.failure_boundary
                or reference is None
                or reference.observed_at != state.left_at
                or reference.source_event_id is not None
            ):
                raise ValueError("interaction reacceptance custody changed")
            reacceptance_keys.add((state.context_id, state.reacceptance_id))
        if set(reference_steps) != reacceptance_keys:
            raise ValueError("interaction reference milestones lack reacceptances")

        fact_keys: set[tuple[str, str, str, pd.Timestamp]] = set()
        strict_fact_clocks: dict[
            tuple[str, str], set[pd.Timestamp]
        ] = {}
        for fact in self.micro_break_facts:
            context = (fact.context_kind, fact.context_id)
            path = paths_by_context.get(context)
            key = (
                *context,
                fact.bos_id,
                fact.resolved_at,
            )
            step = micro_steps.get(key)
            if path is None:
                anchor = None
            elif path.context_kind == "zone_return":
                anchor = locations[path.context_id].first_entered_at
            else:
                anchor = pool_anchors.get(context)
            expected_reason = (
                "confirmed_m1_break_at_anchor_clock"
                if fact.relation == "same_clock_unknown"
                else "first_strictly_later_confirmed_m1_break"
            )
            if (
                key in fact_keys
                or path is None
                or fact.protocol_hash != path.protocol_hash
                or fact.context_direction is not path.direction
                or anchor is None
                or fact.anchor_at != anchor
                or step is None
                or step.direction is not fact.context_direction
                or step.source_entity_id != fact.target_swing_id
                or step.strength != fact.strength
                or step.reason != expected_reason
            ):
                raise ValueError("micro-break fact custody changed")
            fact_keys.add(key)
            if fact.relation == "strictly_after":
                strict_fact_clocks.setdefault(context, set()).add(
                    fact.resolved_at
                )
        if fact_keys != set(micro_steps):
            raise ValueError("micro-break facts and path milestones differ")
        if any(
            (
                path.transition_reason
                == "first_strict_micro_break_observed"
                and strict_fact_clocks.get(
                    (path.context_kind, path.context_id)
                )
                != {path.ended_at}
            )
            for path in self.interaction_paths
        ):
            raise ValueError(
                "strict micro-break path lacks its exact closing fact"
            )
        dominance_steps = {
            "location_left": "location_left",
            "reacceptance_failed": "reacceptance_failed",
            "accepted_outside": "accepted_outside",
            "opposite_displacement_ambiguous_same_clock": (
                "opposite_displacement_ambiguous"
            ),
        }
        for path in self.interaction_paths:
            clocks = strict_fact_clocks.get(
                (path.context_kind, path.context_id),
                set(),
            )
            if not clocks:
                continue
            dominance_step = dominance_steps.get(path.transition_reason)
            if (
                path.lifecycle is not PathSequenceLifecycle.CLOSED
                or clocks != {path.ended_at}
                or (
                    path.transition_reason
                    != "first_strict_micro_break_observed"
                    and (
                        dominance_step is None
                        or not any(
                            step.kind == dominance_step
                            and step.observed_at == path.ended_at
                            for step in path.steps
                        )
                    )
                )
            ):
                raise ValueError(
                    "strict micro-break path closing reason changed"
                )

        if self.boundary_reason is None:
            for transition in self.interaction_path_transitions:
                current = paths.get(transition.sequence_id)
                if (
                    current is None
                    or transition.protocol_hash != current.protocol_hash
                    or transition.symbol != current.symbol
                    or transition.instrument_id != current.instrument_id
                    or transition.context_kind != current.context_kind
                    or transition.context_id != current.context_id
                    or transition.direction is not current.direction
                    or transition.formed_at != current.formed_at
                    or transition.steps
                    != current.steps[: len(transition.steps)]
                ):
                    raise ValueError("interaction path transition changed custody")
        else:
            for path in self.interaction_path_transitions:
                validate_path(path, current=False)
            if any(
                path.lifecycle is not PathSequenceLifecycle.CENSORED
                or path.transition_reason != self.boundary_reason
                for path in self.interaction_path_transitions
            ):
                raise ValueError("interaction boundary path is invalid")
            for state in self.reacceptance_interaction_transitions:
                path = transition_paths_by_context.get(
                    ("zone_return", state.context_id)
                )
                if (
                    path is None
                    or state.protocol_hash != path.protocol_hash
                    or state.symbol != path.symbol
                    or state.instrument_id != path.instrument_id
                    or state.direction is not path.direction
                    or state.source_entity_id
                    != path.steps[0].source_entity_id
                    or not any(
                        step.kind == "reference_left"
                        and step.source_entity_id == state.reacceptance_id
                        and step.observed_at == state.left_at
                        for step in path.steps
                    )
                ):
                    raise ValueError(
                        "interaction boundary reacceptance changed custody"
                    )
            reference_keys: list[tuple[str, str]] = []
            terminal_reference_keys: set[tuple[str, str]] = set()
            for path in self.interaction_path_transitions:
                for step in path.steps:
                    key = (path.context_id, step.source_entity_id)
                    if step.kind == "reference_left":
                        reference_keys.append(key)
                    elif step.kind in {
                        "reacceptance_held",
                        "reacceptance_failed",
                    }:
                        terminal_reference_keys.add(key)
            if len(reference_keys) != len(set(reference_keys)):
                raise ValueError(
                    "interaction boundary reference milestones repeat"
                )
            live_reference_keys = set(reference_keys) - terminal_reference_keys
            transition_keys = {
                (state.context_id, state.reacceptance_id)
                for state in self.reacceptance_interaction_transitions
            }
            if live_reference_keys != transition_keys:
                raise ValueError(
                    "interaction boundary live reacceptance lineage differs"
                )

    def __getstate__(self) -> Mapping[str, Any]:
        self.validate_canonical_bindings()
        return _exact_dataclass_pickle_state(
            self,
            schema_version=INTERACTION_UPDATE_SCHEMA_VERSION,
            label="InteractionUpdate",
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _restore_exact_dataclass_pickle_state(
            self,
            state,
            schema_version=INTERACTION_UPDATE_SCHEMA_VERSION,
            label="InteractionUpdate",
        )


__all__ = [
    "InteractionUpdate",
    "MicroBOSReference",
    "MicroBreakFact",
    "PathSequenceState",
    "PathSequenceStep",
]
