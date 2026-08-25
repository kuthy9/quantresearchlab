"""Frozen import and source-shape adapter for historical Group 5 readers.

Nothing in this module is a production authority.  New code imports
``smc_trader.interaction``; wildcard imports intentionally expose nothing.
Old mutable reducer checkpoints are not resumed through this shim: replay
their canonical events into the current Interaction semantics instead.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from .brain_entry_sequence import (
    interpret_interaction_paths,
    interpret_micro_break_facts,
    interpret_milestone_transitions,
)
from .interaction import InteractionProtocol, InteractionSemantics
from .model import (
    EntryLocationState,
    InteractionUpdate,
    MicroBOSReference,
    PathSequenceState,
    PathSequenceStep,
    ReacceptanceState,
)


Group5Protocol = InteractionProtocol


@dataclass(frozen=True)
class Group5Update:
    """Historical interpreted shape; never embedded in new observations."""

    entry_locations: tuple[EntryLocationState, ...]
    qualified_reacceptances: tuple[ReacceptanceState, ...]
    micro_bos_references: tuple[MicroBOSReference, ...]
    path_sequences: tuple[PathSequenceState, ...]
    path_transitions: tuple[PathSequenceState, ...] = ()
    reacceptance_transitions: tuple[ReacceptanceState, ...] = ()
    step_transitions: tuple[tuple[str, PathSequenceStep], ...] = ()
    cold_source_ids: tuple[str, ...] = ()
    boundary_reason: str | None = None


def _legacy_update(update: InteractionUpdate) -> Group5Update:
    references = interpret_micro_break_facts(update.micro_break_facts)
    paths = interpret_interaction_paths(
        update.interaction_paths,
        references,
    )
    return Group5Update(
        entry_locations=update.zone_interactions,
        qualified_reacceptances=update.reacceptance_interactions,
        micro_bos_references=references,
        path_sequences=paths,
        path_transitions=interpret_interaction_paths(
            update.interaction_path_transitions,
            references,
            allow_censored_without_facts=(
                update.boundary_reason is not None
            ),
        ),
        reacceptance_transitions=(
            update.reacceptance_interaction_transitions
        ),
        step_transitions=interpret_milestone_transitions(
            update.interaction_paths,
            references,
            update.milestone_transitions,
        ),
        cold_source_ids=update.cold_source_ids,
        boundary_reason=update.boundary_reason,
    )


class CausalGroup5Reducer:
    """Frozen API adapter over the sole ``InteractionSemantics`` reducer."""

    def __init__(self, protocol: Group5Protocol) -> None:
        self._semantics = InteractionSemantics(protocol)
        self._last_canonical: InteractionUpdate | None = None
        self._last_legacy: Group5Update | None = None

    def __getstate__(self) -> Mapping[str, object]:
        if (
            (self._last_canonical is None)
            != (self._last_legacy is None)
            or (
                self._last_canonical is not None
                and (
                    type(self._last_canonical) is not InteractionUpdate
                    or type(self._last_legacy) is not Group5Update
                    or set(self._last_legacy.__dict__)
                    != {item.name for item in fields(Group5Update)}
                    or _legacy_update(self._last_canonical)
                    != self._last_legacy
                )
            )
        ):
            raise ValueError("legacy Group 5 reducer state is not exact")
        return {
            "schema_version": 1,
            "_semantics": self._semantics,
            "_last_canonical": self._last_canonical,
            "_last_legacy": self._last_legacy,
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or set(state)
            != {
                "schema_version",
                "_semantics",
                "_last_canonical",
                "_last_legacy",
            }
            or state.get("schema_version") != 1
            or not isinstance(state.get("_semantics"), InteractionSemantics)
            or (
                (state["_last_canonical"] is None)
                != (state["_last_legacy"] is None)
            )
            or (
                state["_last_canonical"] is not None
                and (
                    type(state["_last_canonical"])
                    is not InteractionUpdate
                    or type(state["_last_legacy"]) is not Group5Update
                    or set(state["_last_legacy"].__dict__)
                    != {item.name for item in fields(Group5Update)}
                    or _legacy_update(state["_last_canonical"])
                    != state["_last_legacy"]
                )
            )
        ):
            raise ValueError(
                "legacy Group 5 reducer state cannot resume; replay "
                "canonical events into InteractionSemantics"
            )
        self._semantics = state["_semantics"]
        self._last_canonical = state["_last_canonical"]
        self._last_legacy = state["_last_legacy"]

    @property
    def protocol(self) -> InteractionProtocol:
        return self._semantics.protocol

    def _adapt(self, update: InteractionUpdate) -> Group5Update:
        if update is self._last_canonical and self._last_legacy is not None:
            return self._last_legacy
        legacy = _legacy_update(update)
        self._last_canonical = update
        self._last_legacy = legacy
        return legacy

    def snapshot(self) -> Group5Update:
        return self._adapt(self._semantics.snapshot())

    def on_completed_1m(self, *args: Any, **kwargs: Any) -> Group5Update:
        return self._adapt(self._semantics.on_completed_1m(*args, **kwargs))

    def on_boundary(self, *args: Any, **kwargs: Any) -> Group5Update:
        return self._adapt(self._semantics.on_boundary(*args, **kwargs))


__all__: tuple[str, ...] = ()
