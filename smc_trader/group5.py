"""Causal completed-1m entry-location and ordered-path primitives."""
from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .model import (
    BOSLifecycle,
    BreakOfStructureState,
    Candle,
    Direction,
    EntryLocationLifecycle,
    EntryLocationState,
    FVGQualification,
    FairValueGapLifecycle,
    FairValueGapState,
    GROUP5_HARD_BOUNDARY_REASONS,
    GROUP5_SAME_CLOCK_REACCEPTANCE_FAILURE_REASONS,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    ManipulationState,
    MicroBOSReference,
    OrderBlockLifecycle,
    OrderBlockState,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
    QualifiedReacceptanceLifecycle,
    QualifiedReacceptanceState,
    Timeframe,
    aware_timestamp,
    clamp,
)


@dataclass(frozen=True)
class Group5Protocol:
    """Executable mirror of the frozen Group 5 descriptive contract."""

    protocol_hash: str
    source_group12_protocol_hash: str
    source_group3_protocol_hash: str
    source_group4_protocol_hash: str
    tick_size: float
    m1_atr_period: int
    later_hold_bars: int
    maximum_contexts: int
    maximum_steps_per_path: int
    protocol_version: str
    typed_state_available: bool
    brain_input_allowed: bool
    natural_authority_validated: bool
    independent_action_authority: bool
    favr_enabled: bool

    def __post_init__(self) -> None:
        hashes = (
            self.protocol_hash,
            self.source_group12_protocol_hash,
            self.source_group3_protocol_hash,
            self.source_group4_protocol_hash,
        )
        if (
            any(
                len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in hashes
            )
            or not isinstance(self.protocol_version, str)
            or not self.protocol_version.strip()
            or any(
                type(value) is not bool
                for value in (
                    self.typed_state_available,
                    self.brain_input_allowed,
                    self.natural_authority_validated,
                    self.independent_action_authority,
                    self.favr_enabled,
                )
            )
            or not self.typed_state_available
            or not self.brain_input_allowed
            or self.independent_action_authority
            or (
                self.favr_enabled
                and not self.natural_authority_validated
            )
            or isinstance(self.tick_size, bool)
            or not isinstance(self.tick_size, (int, float))
            or not math.isfinite(float(self.tick_size))
            or float(self.tick_size) <= 0.0
            or any(
                type(value) is not int or value < 1
                for value in (
                    self.m1_atr_period,
                    self.later_hold_bars,
                    self.maximum_contexts,
                    self.maximum_steps_per_path,
                )
            )
        ):
            raise ValueError("Group 5 protocol is invalid")

    @classmethod
    def from_file(cls, path: str | Path) -> "Group5Protocol":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        parameters = payload["engineering_parameters"]
        authority = payload["authority"]
        return cls(
            protocol_hash=hashlib.sha256(raw).hexdigest(),
            source_group12_protocol_hash=payload["upstream"][
                "group12_protocol_sha256"
            ],
            source_group3_protocol_hash=payload["upstream"][
                "group3_protocol_sha256"
            ],
            source_group4_protocol_hash=payload["upstream"][
                "group4_protocol_sha256"
            ],
            tick_size=payload["tick_size"],
            m1_atr_period=parameters["m1_atr_period"],
            later_hold_bars=parameters[
                "qualified_reacceptance_later_hold_bars"
            ],
            maximum_contexts=parameters["maximum_context_states"],
            maximum_steps_per_path=parameters[
                "maximum_steps_per_path"
            ],
            protocol_version=payload["protocol_version"],
            typed_state_available=authority[
                "typed_state_available"
            ],
            brain_input_allowed=authority["brain_input_allowed"],
            natural_authority_validated=authority[
                "natural_authority_validated"
            ],
            independent_action_authority=authority[
                "independent_action_authority"
            ],
            favr_enabled=authority["favr_enabled"],
        )


@dataclass(frozen=True)
class Group5Update:
    entry_locations: tuple[EntryLocationState, ...]
    qualified_reacceptances: tuple[
        QualifiedReacceptanceState,
        ...,
    ]
    micro_bos_references: tuple[MicroBOSReference, ...]
    path_sequences: tuple[PathSequenceState, ...]
    path_transitions: tuple[PathSequenceState, ...] = ()
    reacceptance_transitions: tuple[
        QualifiedReacceptanceState,
        ...,
    ] = ()
    step_transitions: tuple[tuple[str, PathSequenceStep], ...] = ()
    cold_source_ids: tuple[str, ...] = ()
    boundary_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "entry_locations",
            "qualified_reacceptances",
            "micro_bos_references",
            "path_sequences",
            "path_transitions",
            "reacceptance_transitions",
            "step_transitions",
            "cold_source_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            self.boundary_reason is not None
            and self.boundary_reason
            not in GROUP5_HARD_BOUNDARY_REASONS
        ):
            raise ValueError("Group 5 update has an unregistered boundary")
        if (
            self.boundary_reason is None
            and self.reacceptance_transitions
        ):
            raise ValueError(
                "Group 5 reacceptance censor requires a hard boundary"
            )
        if any(
            state.lifecycle
            is not QualifiedReacceptanceLifecycle.CENSORED
            or state.censored_at is None
            or state.transition_reason != "hard_boundary_censored"
            for state in self.reacceptance_transitions
        ):
            raise ValueError(
                "Group 5 boundary reacceptance transition is invalid"
            )
        if len(self.cold_source_ids) != len(set(self.cold_source_ids)):
            raise ValueError("Group 5 cold source identities repeat")
        path_ids = {
            state.sequence_id for state in self.path_sequences
        }
        if any(
            sequence_id not in path_ids
            or step not in next(
                state.steps
                for state in self.path_sequences
                if state.sequence_id == sequence_id
            )
            for sequence_id, step in self.step_transitions
        ):
            raise ValueError(
                "Group 5 step transition lacks its current path"
            )


@dataclass(frozen=True)
class _ZoneSource:
    kind: str
    source_id: str
    protocol_hash: str
    symbol: str
    instrument_id: int
    direction: Direction
    source_displacement_id: str | None
    source_displacement_active_at: pd.Timestamp | None
    source_bos_id: str | None
    fvg_qualification: FVGQualification | None
    lower_bound: float
    upper_bound: float
    midpoint: float
    invalidation_price: float
    confirmed_at: pd.Timestamp
    strength: float
    lifecycle: str

    @property
    def can_register(self) -> bool:
        return (
            self.kind == "fvg"
            and self.fvg_qualification
            is FVGQualification.DISPLACEMENT_LINKED
            and self.lifecycle == FairValueGapLifecycle.OPEN.value
        ) or (
            self.kind == "order_block"
            and self.lifecycle == OrderBlockLifecycle.CREATED.value
        )

    @property
    def failed(self) -> bool:
        return (
            self.kind == "fvg"
            and self.lifecycle
            == FairValueGapLifecycle.INVALIDATED.value
        ) or (
            self.kind == "order_block"
            and self.lifecycle == OrderBlockLifecycle.FAILED.value
        )


def _identity(*parts: object) -> str:
    return hashlib.sha256(
        "|".join(
            value.isoformat()
            if isinstance(value, pd.Timestamp)
            else str(value)
            for value in parts
        ).encode("utf-8")
    ).hexdigest()


class CausalGroup5Reducer:
    """One transactional reducer for Group 5's two causal contexts."""

    def __init__(self, protocol: Group5Protocol) -> None:
        self.protocol = protocol
        self._locations: dict[str, EntryLocationState] = {}
        self._reacceptances: dict[
            str,
            QualifiedReacceptanceState,
        ] = {}
        self._micro_references: dict[str, MicroBOSReference] = {}
        self._paths: dict[str, PathSequenceState] = {}
        self._path_order: deque[str] = deque()
        self._exposed_terminal_path_ids: set[str] = set()
        self._identity: tuple[str, int] | None = None
        self._last_raw_end: pd.Timestamp | None = None
        self._last_input: tuple[object, ...] | None = None
        self._last_output: Group5Update | None = None
        self._last_boundary_input: tuple[object, ...] | None = None
        self._last_boundary_output: Group5Update | None = None

    def _transaction_clone(self) -> "CausalGroup5Reducer":
        candidate = copy(self)
        candidate._locations = dict(self._locations)
        candidate._reacceptances = dict(self._reacceptances)
        candidate._micro_references = dict(self._micro_references)
        candidate._paths = dict(self._paths)
        candidate._path_order = deque(self._path_order)
        candidate._exposed_terminal_path_ids = set(
            self._exposed_terminal_path_ids
        )
        return candidate

    def _commit(self, candidate: "CausalGroup5Reducer") -> None:
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    @staticmethod
    def _price_identity(price: float) -> str:
        """Bind derived zone prices without pretending they are trade ticks."""

        value = float(price)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("invalid Group 5 reference price")
        # Support/resistance bounds may be ATR-derived and therefore lie
        # between executable ticks. They remain valid descriptive reference
        # levels; the source context and frozen decimal bind their identity.
        return f"{value:.10f}"

    @staticmethod
    def _location_terminal(state: EntryLocationState) -> bool:
        return state.lifecycle is EntryLocationLifecycle.LEFT

    @staticmethod
    def _reacceptance_terminal(
        state: QualifiedReacceptanceState,
    ) -> bool:
        return state.lifecycle in {
            QualifiedReacceptanceLifecycle.HELD,
            QualifiedReacceptanceLifecycle.FAILED,
            QualifiedReacceptanceLifecycle.CENSORED,
        }

    def _snapshot(self) -> Group5Update:
        return Group5Update(
            entry_locations=tuple(
                self._locations[key]
                for key in sorted(
                    self._locations,
                    key=lambda value: (
                        self._locations[value].formed_at,
                        value,
                    ),
                )
            ),
            qualified_reacceptances=tuple(
                self._reacceptances[key]
                for key in sorted(
                    self._reacceptances,
                    key=lambda value: (
                        self._reacceptances[value].formed_at,
                        value,
                    ),
                )
            ),
            micro_bos_references=tuple(
                self._micro_references[key]
                for key in sorted(
                    self._micro_references,
                    key=lambda value: (
                        self._micro_references[value].resolved_at,
                        value,
                    ),
                )
            ),
            path_sequences=tuple(
                self._paths[key]
                for key in self._path_order
                if key in self._paths
            ),
        )

    def snapshot(self) -> Group5Update:
        return self._snapshot()

    @staticmethod
    def _zone_source(
        state: FairValueGapState | OrderBlockState,
    ) -> _ZoneSource:
        if isinstance(state, FairValueGapState):
            return _ZoneSource(
                kind="fvg",
                source_id=state.fvg_id,
                protocol_hash=state.protocol_hash,
                symbol=state.symbol,
                instrument_id=state.instrument_id,
                direction=state.direction,
                source_displacement_id=state.source_displacement_id,
                source_displacement_active_at=(
                    state.source_displacement_active_at
                ),
                source_bos_id=None,
                fvg_qualification=state.qualification,
                lower_bound=state.lower_bound,
                upper_bound=state.upper_bound,
                midpoint=state.midpoint,
                invalidation_price=state.invalidation_price,
                confirmed_at=state.confirmed_at,
                strength=state.strength,
                lifecycle=state.lifecycle.value,
            )
        if isinstance(state, OrderBlockState):
            return _ZoneSource(
                kind="order_block",
                source_id=state.order_block_id,
                protocol_hash=state.protocol_hash,
                symbol=state.symbol,
                instrument_id=state.instrument_id,
                direction=state.direction,
                source_displacement_id=state.source_displacement_id,
                source_displacement_active_at=(
                    state.source_displacement_active_at
                ),
                source_bos_id=state.source_bos_id,
                fvg_qualification=None,
                lower_bound=state.lower_bound,
                upper_bound=state.upper_bound,
                midpoint=state.midpoint,
                invalidation_price=state.invalidation_price,
                confirmed_at=state.confirmed_at,
                strength=state.strength,
                lifecycle=state.lifecycle.value,
            )
        raise TypeError("Group 5 received an untyped entry-zone source")

    def _validate_sources(
        self,
        candle: Candle,
        sources: Sequence[_ZoneSource],
        manipulations: Sequence[ManipulationState],
        m1_bos: Sequence[BreakOfStructureState],
        inventory: Sequence[LiquidityInventoryItem],
    ) -> None:
        identity = (candle.symbol, int(candle.instrument_id))
        if self._identity is not None and identity != self._identity:
            raise ValueError(
                "Group 5 contract changed without a hard boundary"
            )
        if len({source.source_id for source in sources}) != len(sources):
            raise ValueError("Group 5 entry-zone source identities repeat")
        if any(
            source.protocol_hash
            != self.protocol.source_group3_protocol_hash
            or (source.symbol, source.instrument_id) != identity
            or source.confirmed_at > candle.end
            for source in sources
        ):
            raise ValueError(
                "Group 5 entry-zone source binding is invalid"
            )
        if len(
            {state.manipulation_id for state in manipulations}
        ) != len(manipulations):
            raise ValueError(
                "Group 5 manipulation source identities repeat"
            )
        if any(
            state.protocol_hash
            != self.protocol.source_group4_protocol_hash
            or state.source_kind != "formed_liquidity_pool"
            or (state.symbol, state.instrument_id) != identity
            or state.last_updated_at > candle.end
            for state in manipulations
        ):
            raise ValueError(
                "Group 5 manipulation source binding is invalid"
            )
        if len({state.bos_id for state in m1_bos}) != len(m1_bos):
            raise ValueError("Group 5 M1 BOS identities repeat")
        if any(
            state.timeframe is not Timeframe.M1
            or (
                state.resolved_at is not None
                and state.resolved_at > candle.end
            )
            for state in m1_bos
        ):
            raise ValueError("Group 5 M1 BOS source clock is invalid")
        if len({item.item_id for item in inventory}) != len(inventory):
            raise ValueError("Group 5 inventory identities repeat")
        if any(
            item.confirmed_at > candle.end
            or (
                item.targeted_at is not None
                and item.targeted_at > candle.end
            )
            or (
                item.consumed_at is not None
                and item.consumed_at > candle.end
            )
            for item in inventory
        ):
            raise ValueError("Group 5 inventory source clock is invalid")
        self._identity = identity

    def _nearest_draw(
        self,
        direction: Direction,
        price: float,
        inventory: Sequence[LiquidityInventoryItem],
    ) -> tuple[str | None, float | None]:
        candidates: list[tuple[float, str]] = []
        for item in inventory:
            if (
                item.lifecycle is not LiquidityInventoryLifecycle.VISIBLE
                or item.kind == "range_boundary"
            ):
                continue
            if (
                direction is Direction.LONG
                and item.side == "above"
                and item.price > price
            ):
                candidates.append((item.price - price, item.item_id))
            elif (
                direction is Direction.SHORT
                and item.side == "below"
                and item.price < price
            ):
                candidates.append((price - item.price, item.item_id))
        if not candidates:
            return None, None
        distance, item_id = min(candidates, key=lambda value: (value[0], value[1]))
        return item_id, float(distance)

    @staticmethod
    def _distance_to_zone(
        price: float,
        lower: float,
        upper: float,
    ) -> float:
        if lower <= price <= upper:
            return 0.0
        return min(abs(price - lower), abs(price - upper))

    @staticmethod
    def _delivery_side(
        direction: Direction,
        price: float,
        reference: float,
    ) -> bool:
        return (
            price > reference
            if direction is Direction.LONG
            else price < reference
        )

    @staticmethod
    def _adverse_side(
        direction: Direction,
        price: float,
        reference: float,
    ) -> bool:
        return (
            price < reference
            if direction is Direction.LONG
            else price > reference
        )

    @staticmethod
    def _intersects(
        candle: Candle,
        lower: float,
        upper: float,
    ) -> bool:
        return candle.low <= upper and candle.high >= lower

    def _path_id(
        self,
        *,
        symbol: str,
        instrument_id: int,
        context_kind: str,
        context_id: str,
    ) -> str:
        return _identity(
            "group5-path-v1",
            self.protocol.protocol_hash,
            symbol,
            instrument_id,
            context_kind,
            context_id,
        )

    def _location_id(self, source: _ZoneSource) -> str:
        return _identity(
            "group5-location-v1",
            self.protocol.protocol_hash,
            source.symbol,
            source.instrument_id,
            source.kind,
            source.source_id,
        )

    def _reacceptance_id(
        self,
        *,
        context_kind: str,
        context_id: str,
        reference_price: float,
        left_at: pd.Timestamp,
    ) -> str:
        return _identity(
            "group5-reacceptance-v1",
            self.protocol.protocol_hash,
            context_kind,
            context_id,
            self._price_identity(reference_price),
            left_at,
        )

    def _step(
        self,
        path: PathSequenceState,
        *,
        kind: str,
        observed_at: pd.Timestamp,
        source_event_id: str | None,
        source_entity_id: str,
        strength: float,
        reason: str,
        same_clock_relation: str | None = None,
    ) -> PathSequenceStep:
        if len(path.steps) >= self.protocol.maximum_steps_per_path:
            raise RuntimeError("Group 5 path step capacity is exhausted")
        previous = path.steps[-1] if path.steps else None
        if previous is None:
            relation = "origin"
            predecessors: tuple[str, ...] = ()
        elif observed_at > previous.observed_at:
            relation = "strictly_after"
            predecessors = (previous.step_id,)
        elif observed_at == previous.observed_at:
            relation = same_clock_relation or "same_clock_unknown"
            if relation not in {"same_clock_known", "same_clock_unknown"}:
                raise ValueError("same-clock Group 5 relation is invalid")
            predecessors = (previous.step_id,)
        else:
            raise ValueError("Group 5 path step is out of order")
        ordinal = len(path.steps)
        return PathSequenceStep(
            step_id=_identity(
                "group5-step-v1",
                self.protocol.protocol_hash,
                path.sequence_id,
                ordinal,
                kind,
                observed_at,
                source_entity_id,
            ),
            kind=kind,
            observed_at=observed_at,
            source_event_id=source_event_id,
            source_entity_id=source_entity_id,
            predecessor_step_ids=predecessors,
            same_clock_relation=relation,
            direction=path.direction,
            strength=clamp(strength),
            reason=reason,
        )

    def _append_step(
        self,
        path_id: str,
        *,
        kind: str,
        observed_at: pd.Timestamp,
        source_event_id: str | None,
        source_entity_id: str,
        strength: float,
        reason: str,
        step_transitions: list[tuple[str, PathSequenceStep]],
        same_clock_relation: str | None = None,
    ) -> PathSequenceStep:
        path = self._paths[path_id]
        if path.lifecycle is not PathSequenceLifecycle.ACTIVE:
            raise RuntimeError("cannot append to a terminal Group 5 path")
        step = self._step(
            path,
            kind=kind,
            observed_at=observed_at,
            source_event_id=source_event_id,
            source_entity_id=source_entity_id,
            strength=strength,
            reason=reason,
            same_clock_relation=same_clock_relation,
        )
        self._paths[path_id] = replace(
            path,
            last_updated_at=observed_at,
            steps=(*path.steps, step),
        )
        step_transitions.append((path_id, step))
        return step

    def _close_path(
        self,
        path_id: str,
        observed_at: pd.Timestamp,
        reason: str,
        transitions: list[PathSequenceState],
    ) -> None:
        path = self._paths[path_id]
        if path.lifecycle is not PathSequenceLifecycle.ACTIVE:
            return
        terminal = replace(
            path,
            lifecycle=PathSequenceLifecycle.CLOSED,
            state_started_at=observed_at,
            last_updated_at=observed_at,
            state_duration_real_1m_bars=0,
            ended_at=observed_at,
            transition_reason=reason,
        )
        self._paths[path_id] = terminal
        transitions.append(terminal)

    def _mark_active_path_milestone(
        self,
        path_id: str,
        observed_at: pd.Timestamp,
        reason: str,
    ) -> None:
        path = self._paths[path_id]
        if path.lifecycle is not PathSequenceLifecycle.ACTIVE:
            raise RuntimeError("cannot mark a terminal Group 5 path")
        if path.transition_reason == reason:
            return
        self._paths[path_id] = replace(
            path,
            state_started_at=observed_at,
            last_updated_at=max(path.last_updated_at, observed_at),
            state_duration_real_1m_bars=0,
            transition_reason=reason,
        )

    def _admit_context_capacity(self) -> None:
        if len(self._paths) < self.protocol.maximum_contexts:
            return
        candidates = tuple(
            (
                self._paths[path_id].ended_at,
                path_id,
            )
            for path_id in self._exposed_terminal_path_ids
            if (
                path_id in self._paths
                and self._paths[path_id].lifecycle
                in {
                    PathSequenceLifecycle.CLOSED,
                    PathSequenceLifecycle.CENSORED,
                }
                and self._paths[path_id].ended_at is not None
            )
        )
        if not candidates:
            raise RuntimeError(
                "Group 5 context capacity cannot evict live or same-bar state"
            )
        _, path_id = min(candidates)
        path = self._paths.pop(path_id)
        self._path_order.remove(path_id)
        self._exposed_terminal_path_ids.discard(path_id)
        if path.context_kind == "zone_return":
            self._locations.pop(path.context_id, None)
        for key, state in tuple(self._reacceptances.items()):
            mapped_kind = (
                "zone_return"
                if state.context_kind == "entry_zone"
                else "pool_reversal"
            )
            if (mapped_kind, state.context_id) == (
                path.context_kind,
                path.context_id,
            ):
                self._reacceptances.pop(key, None)
        for key, reference in tuple(self._micro_references.items()):
            if (
                reference.context_kind,
                reference.context_id,
            ) == (path.context_kind, path.context_id):
                self._micro_references.pop(key, None)

    def _new_path(
        self,
        *,
        symbol: str,
        instrument_id: int,
        context_kind: str,
        context_id: str,
        direction: Direction,
        formed_at: pd.Timestamp,
        first_kind: str,
        source_event_id: str | None,
        source_entity_id: str,
        strength: float,
        reason: str,
    ) -> tuple[PathSequenceState, PathSequenceStep]:
        self._admit_context_capacity()
        path_id = self._path_id(
            symbol=symbol,
            instrument_id=instrument_id,
            context_kind=context_kind,
            context_id=context_id,
        )
        if path_id in self._paths:
            raise RuntimeError("Group 5 context identity was registered twice")
        placeholder = PathSequenceState(
            sequence_id=path_id,
            protocol_hash=self.protocol.protocol_hash,
            symbol=symbol,
            instrument_id=instrument_id,
            context_kind=context_kind,
            context_id=context_id,
            direction=direction,
            lifecycle=PathSequenceLifecycle.ACTIVE,
            formed_at=formed_at,
            state_started_at=formed_at,
            last_updated_at=formed_at,
            age_real_1m_bars=0,
            state_duration_real_1m_bars=0,
            steps=(
                PathSequenceStep(
                    step_id=_identity(
                        "group5-step-v1",
                        self.protocol.protocol_hash,
                        path_id,
                        0,
                        first_kind,
                        formed_at,
                        source_entity_id,
                    ),
                    kind=first_kind,
                    observed_at=formed_at,
                    source_event_id=source_event_id,
                    source_entity_id=source_entity_id,
                    predecessor_step_ids=(),
                    same_clock_relation="origin",
                    direction=direction,
                    strength=clamp(strength),
                    reason=reason,
                ),
            ),
            transition_reason="context_registered",
        )
        self._paths[path_id] = placeholder
        self._path_order.append(path_id)
        return placeholder, placeholder.steps[0]

    def _location_view(
        self,
        state: EntryLocationState,
        candle: Candle,
        inventory: Sequence[LiquidityInventoryItem],
    ) -> dict[str, object]:
        draw_id, draw_distance = self._nearest_draw(
            state.direction,
            float(candle.close),
            inventory,
        )
        return {
            "current_price": float(candle.close),
            "distance_to_zone_points": self._distance_to_zone(
                float(candle.close),
                state.lower_bound,
                state.upper_bound,
            ),
            "distance_to_failure_points": (
                state.direction.sign
                * (float(candle.close) - state.failure_boundary)
            ),
            "nearest_visible_draw_id": draw_id,
            "nearest_visible_draw_distance_points": draw_distance,
        }

    def _new_zone_context(
        self,
        source: _ZoneSource,
        candle: Candle,
        inventory: Sequence[LiquidityInventoryItem],
        path_transitions: list[PathSequenceState],
        step_transitions: list[tuple[str, PathSequenceStep]],
    ) -> None:
        if not source.can_register or source.source_displacement_id is None:
            raise ValueError(
                "Group 5 registers only displacement-linked FVGs or strict OBs"
            )
        location_id = self._location_id(source)
        if location_id in self._locations:
            raise RuntimeError("Group 5 location identity was registered twice")
        near = (
            source.upper_bound
            if source.direction is Direction.LONG
            else source.lower_bound
        )
        far = (
            source.lower_bound
            if source.direction is Direction.LONG
            else source.upper_bound
        )
        draw_id, draw_distance = self._nearest_draw(
            source.direction,
            float(candle.close),
            inventory,
        )
        departure = (
            candle.end
            if self._delivery_side(
                source.direction,
                float(candle.close),
                near,
            )
            else None
        )
        location = EntryLocationState(
            location_id=location_id,
            protocol_hash=self.protocol.protocol_hash,
            source_group3_protocol_hash=(
                self.protocol.source_group3_protocol_hash
            ),
            symbol=source.symbol,
            instrument_id=source.instrument_id,
            direction=source.direction,
            source_zone_kind=source.kind,
            source_zone_id=source.source_id,
            source_zone_protocol_hash=source.protocol_hash,
            source_displacement_id=source.source_displacement_id,
            source_bos_id=source.source_bos_id,
            lower_bound=source.lower_bound,
            upper_bound=source.upper_bound,
            midpoint=source.midpoint,
            near_edge=near,
            far_edge=far,
            failure_boundary=source.invalidation_price,
            formed_at=source.confirmed_at,
            lifecycle=EntryLocationLifecycle.APPROACHING,
            state_started_at=source.confirmed_at,
            last_updated_at=source.confirmed_at,
            age_real_1m_bars=0,
            state_duration_real_1m_bars=0,
            current_price=float(candle.close),
            distance_to_zone_points=self._distance_to_zone(
                float(candle.close),
                source.lower_bound,
                source.upper_bound,
            ),
            distance_to_failure_points=(
                source.direction.sign
                * (float(candle.close) - source.invalidation_price)
            ),
            nearest_visible_draw_id=draw_id,
            nearest_visible_draw_distance_points=draw_distance,
            departure_confirmed_at=departure,
        )
        self._locations[location_id] = location
        path, first_step = self._new_path(
            symbol=source.symbol,
            instrument_id=source.instrument_id,
            context_kind="zone_return",
            context_id=location_id,
            direction=source.direction,
            formed_at=source.confirmed_at,
            first_kind="zone_visible",
            source_event_id=source.source_id,
            source_entity_id=source.source_id,
            strength=source.strength,
            reason="typed_entry_zone_registered",
        )
        path_transitions.append(path)
        step_transitions.append((path.sequence_id, first_step))
        if departure is not None:
            self._append_step(
                path.sequence_id,
                kind="departure_confirmed",
                observed_at=departure,
                source_event_id=None,
                source_entity_id=location_id,
                strength=0.0,
                reason="formation_close_on_delivery_side",
                step_transitions=step_transitions,
                same_clock_relation="same_clock_known",
            )

    def _new_reacceptance(
        self,
        *,
        symbol: str,
        instrument_id: int,
        context_kind: str,
        context_id: str,
        source_entity_id: str,
        direction: Direction,
        reference_price: float,
        failure_boundary: float,
        candle: Candle,
        atr: float,
        path_id: str,
        step_transitions: list[tuple[str, PathSequenceStep]],
    ) -> QualifiedReacceptanceState:
        identity = self._reacceptance_id(
            context_kind=context_kind,
            context_id=context_id,
            reference_price=reference_price,
            left_at=candle.end,
        )
        margin = clamp(
            abs(float(candle.close) - reference_price) / atr
        )
        state = QualifiedReacceptanceState(
            reacceptance_id=identity,
            protocol_hash=self.protocol.protocol_hash,
            symbol=symbol,
            instrument_id=instrument_id,
            context_kind=context_kind,
            context_id=context_id,
            source_entity_id=source_entity_id,
            direction=direction,
            reference_price=reference_price,
            failure_boundary=failure_boundary,
            lifecycle=QualifiedReacceptanceLifecycle.LEFT,
            formed_at=candle.end,
            state_started_at=candle.end,
            last_updated_at=candle.end,
            left_at=candle.end,
            age_real_1m_bars=0,
            state_duration_real_1m_bars=0,
            required_later_hold_bars=self.protocol.later_hold_bars,
            transition_reason="reference_left",
        )
        self._reacceptances[identity] = state
        self._append_step(
            path_id,
            kind="reference_left",
            observed_at=candle.end,
            source_event_id=None,
            source_entity_id=identity,
            strength=margin,
            reason="completed_close_on_adverse_side",
            step_transitions=step_transitions,
            same_clock_relation="same_clock_known",
        )
        return state

    def _context_reacceptance(
        self,
        context_kind: str,
        context_id: str,
    ) -> tuple[str, QualifiedReacceptanceState] | None:
        if context_kind != "zone_return":
            raise ValueError(
                "qualified reacceptance is only defined for entry zones"
            )
        matches = tuple(
            (key, state)
            for key, state in self._reacceptances.items()
            if (
                state.context_kind == "entry_zone"
                and state.context_id == context_id
            )
        )
        if len(matches) > 1:
            raise RuntimeError(
                "Group 5 context retained multiple reacceptances"
            )
        return matches[0] if matches else None

    def _fail_reacceptance(
        self,
        key: str,
        candle: Candle,
        reason: str,
        path_id: str,
        step_transitions: list[tuple[str, PathSequenceStep]],
    ) -> QualifiedReacceptanceState:
        state = self._reacceptances[key]
        if self._reacceptance_terminal(state):
            return state
        failed = replace(
            state,
            lifecycle=QualifiedReacceptanceLifecycle.FAILED,
            state_started_at=candle.end,
            last_updated_at=candle.end,
            state_duration_real_1m_bars=0,
            failed_at=candle.end,
            transition_reason=reason,
        )
        self._reacceptances[key] = failed
        self._append_step(
            path_id,
            kind="reacceptance_failed",
            observed_at=candle.end,
            source_event_id=None,
            source_entity_id=state.reacceptance_id,
            strength=0.0,
            reason=reason,
            step_transitions=step_transitions,
            same_clock_relation=(
                "same_clock_known"
                if reason
                in GROUP5_SAME_CLOCK_REACCEPTANCE_FAILURE_REASONS
                else None
            ),
        )
        return failed

    def _advance_reacceptance(
        self,
        key: str,
        candle: Candle,
        atr: float,
        *,
        path_id: str,
        step_transitions: list[tuple[str, PathSequenceStep]],
    ) -> QualifiedReacceptanceState:
        state = self._reacceptances[key]
        if self._reacceptance_terminal(state):
            return state
        aged = replace(
            state,
            last_updated_at=candle.end,
            age_real_1m_bars=state.age_real_1m_bars + 1,
            state_duration_real_1m_bars=(
                state.state_duration_real_1m_bars + 1
            ),
        )
        self._reacceptances[key] = aged
        if self._adverse_side(
            state.direction,
            float(candle.close),
            state.failure_boundary,
        ):
            return self._fail_reacceptance(
                key,
                candle,
                "close_beyond_failure_boundary",
                path_id,
                step_transitions,
            )
        if state.lifecycle is QualifiedReacceptanceLifecycle.LEFT:
            if (
                candle.end > state.left_at
                and self._delivery_side(
                    state.direction,
                    float(candle.close),
                    state.reference_price,
                )
            ):
                margin = clamp(
                    abs(float(candle.close) - state.reference_price)
                    / atr
                )
                reclaimed = replace(
                    aged,
                    lifecycle=(
                        QualifiedReacceptanceLifecycle.RECLAIMED
                    ),
                    state_started_at=candle.end,
                    state_duration_real_1m_bars=0,
                    reclaimed_at=candle.end,
                    reclaim_margin_atr=margin,
                    transition_reason="strict_reference_reclaim",
                )
                self._reacceptances[key] = reclaimed
                self._append_step(
                    path_id,
                    kind="reference_reclaimed",
                    observed_at=candle.end,
                    source_event_id=None,
                    source_entity_id=state.reacceptance_id,
                    strength=margin,
                    reason="strict_completed_close_reclaim",
                    step_transitions=step_transitions,
                )
                return reclaimed
            return aged
        if self._adverse_side(
            state.direction,
            float(candle.close),
            state.reference_price,
        ):
            return self._fail_reacceptance(
                key,
                candle,
                "reclaim_lost_before_hold",
                path_id,
                step_transitions,
            )
        if self._delivery_side(
            state.direction,
            float(candle.close),
            state.reference_price,
        ):
            hold_count = min(
                state.required_later_hold_bars,
                state.hold_real_1m_bars + 1,
            )
            if hold_count >= state.required_later_hold_bars:
                hold_margin = clamp(
                    abs(float(candle.close) - state.reference_price)
                    / atr
                )
                held = replace(
                    aged,
                    lifecycle=QualifiedReacceptanceLifecycle.HELD,
                    state_started_at=candle.end,
                    state_duration_real_1m_bars=0,
                    hold_real_1m_bars=hold_count,
                    held_at=candle.end,
                    hold_margin_atr=hold_margin,
                    strength=min(
                        state.reclaim_margin_atr,
                        hold_margin,
                    ),
                    transition_reason="later_completed_bar_held",
                )
                self._reacceptances[key] = held
                self._append_step(
                    path_id,
                    kind="reacceptance_held",
                    observed_at=candle.end,
                    source_event_id=None,
                    source_entity_id=state.reacceptance_id,
                    strength=held.strength,
                    reason="later_real_completed_hold",
                    step_transitions=step_transitions,
                )
                return held
        return aged

    def _advance_location(
        self,
        location_id: str,
        candle: Candle,
        atr: float,
        inventory: Sequence[LiquidityInventoryItem],
        source: _ZoneSource | None,
        path_transitions: list[PathSequenceState],
        step_transitions: list[tuple[str, PathSequenceStep]],
    ) -> None:
        state = self._locations[location_id]
        if self._location_terminal(state):
            return
        path_id = self._path_id(
            symbol=state.symbol,
            instrument_id=state.instrument_id,
            context_kind="zone_return",
            context_id=location_id,
        )
        aged = replace(
            state,
            **self._location_view(state, candle, inventory),
            last_updated_at=candle.end,
            age_real_1m_bars=state.age_real_1m_bars + 1,
            state_duration_real_1m_bars=(
                state.state_duration_real_1m_bars + 1
            ),
        )
        self._locations[location_id] = aged
        source_failed = bool(source is not None and source.failed)

        if state.lifecycle is EntryLocationLifecycle.REJECTED:
            if self._adverse_side(
                state.direction,
                float(candle.close),
                state.far_edge,
            ) or source_failed:
                reason = (
                    (
                        "fvg_invalidated"
                        if state.source_zone_kind == "fvg"
                        else "order_block_failed"
                    )
                    if source_failed
                    else "close_beyond_far_edge"
                )
                left = replace(
                    aged,
                    lifecycle=EntryLocationLifecycle.LEFT,
                    state_started_at=candle.end,
                    state_duration_real_1m_bars=0,
                    rejected_at=None,
                    left_at=candle.end,
                    transition_reason=reason,
                )
                self._locations[location_id] = left
                self._append_step(
                    path_id,
                    kind="location_left",
                    observed_at=candle.end,
                    source_event_id=(
                        state.source_zone_id if source_failed else None
                    ),
                    source_entity_id=location_id,
                    strength=0.0,
                    reason=reason,
                    step_transitions=step_transitions,
                )
            return

        if state.departure_confirmed_at is None:
            if source_failed:
                left = replace(
                    aged,
                    lifecycle=EntryLocationLifecycle.LEFT,
                    state_started_at=candle.end,
                    state_duration_real_1m_bars=0,
                    left_at=candle.end,
                    transition_reason=(
                        "fvg_invalidated"
                        if state.source_zone_kind == "fvg"
                        else "order_block_failed"
                    ),
                )
                self._locations[location_id] = left
                self._append_step(
                    path_id,
                    kind="location_left",
                    observed_at=candle.end,
                    source_event_id=state.source_zone_id,
                    source_entity_id=location_id,
                    strength=0.0,
                    reason=left.transition_reason,
                    step_transitions=step_transitions,
                )
            elif self._delivery_side(
                state.direction,
                float(candle.close),
                state.near_edge,
            ):
                self._locations[location_id] = replace(
                    aged,
                    departure_confirmed_at=candle.end,
                )
                self._append_step(
                    path_id,
                    kind="departure_confirmed",
                    observed_at=candle.end,
                    source_event_id=None,
                    source_entity_id=location_id,
                    strength=0.0,
                    reason="later_close_on_delivery_side",
                    step_transitions=step_transitions,
                )
            return

        if state.first_entered_at is None:
            if self._adverse_side(
                state.direction,
                float(candle.open),
                state.far_edge,
            ):
                reason = (
                    (
                        "fvg_invalidated"
                        if state.source_zone_kind == "fvg"
                        else "order_block_failed"
                    )
                    if source_failed
                    else "gap_through_frozen_zone"
                )
                left = replace(
                    aged,
                    lifecycle=EntryLocationLifecycle.LEFT,
                    state_started_at=candle.end,
                    state_duration_real_1m_bars=0,
                    left_at=candle.end,
                    transition_reason=reason,
                )
                self._locations[location_id] = left
                self._append_step(
                    path_id,
                    kind="location_left",
                    observed_at=candle.end,
                    source_event_id=(
                        state.source_zone_id
                        if source_failed
                        else None
                    ),
                    source_entity_id=location_id,
                    strength=0.0,
                    reason=reason,
                    step_transitions=step_transitions,
                )
                return
            if self._intersects(
                candle,
                state.lower_bound,
                state.upper_bound,
            ):
                width = state.upper_bound - state.lower_bound
                penetration = (
                    (
                        state.upper_bound
                        - max(float(candle.low), state.lower_bound)
                    )
                    / width
                    if state.direction is Direction.LONG
                    else (
                        min(float(candle.high), state.upper_bound)
                        - state.lower_bound
                    )
                    / width
                )
                open_inside = (
                    state.lower_bound
                    <= candle.open
                    <= state.upper_bound
                )
                mode = (
                    "gap_opened_inside"
                    if open_inside
                    else "crossed_near_edge"
                )
                contact = (
                    float(candle.open)
                    if open_inside
                    else state.near_edge
                )
                visited = replace(
                    aged,
                    lifecycle=EntryLocationLifecycle.IN_ZONE,
                    state_started_at=candle.end,
                    state_duration_real_1m_bars=0,
                    first_entered_at=candle.end,
                    entry_mode=mode,
                    contact_reference_price=contact,
                    first_penetration_fraction=clamp(penetration),
                    transition_reason="first_pullback",
                )
                self._locations[location_id] = visited
                self._append_step(
                    path_id,
                    kind="first_pullback",
                    observed_at=candle.end,
                    source_event_id=None,
                    source_entity_id=location_id,
                    strength=visited.first_penetration_fraction,
                    reason=mode,
                    step_transitions=step_transitions,
                )
                if source_failed or self._adverse_side(
                    state.direction,
                    float(candle.close),
                    state.far_edge,
                ):
                    reason = (
                        (
                            "fvg_invalidated"
                            if state.source_zone_kind == "fvg"
                            else "order_block_failed"
                        )
                        if source_failed
                        else "close_beyond_far_edge"
                    )
                    left = replace(
                        visited,
                        lifecycle=EntryLocationLifecycle.LEFT,
                        state_started_at=candle.end,
                        state_duration_real_1m_bars=0,
                        left_at=candle.end,
                        transition_reason=reason,
                    )
                    self._locations[location_id] = left
                    self._append_step(
                        path_id,
                        kind="location_left",
                        observed_at=candle.end,
                        source_event_id=(
                            state.source_zone_id
                            if source_failed
                            else None
                        ),
                        source_entity_id=location_id,
                        strength=0.0,
                        reason=reason,
                        step_transitions=step_transitions,
                        same_clock_relation="same_clock_known",
                    )
                elif self._delivery_side(
                    state.direction,
                    float(candle.close),
                    state.near_edge,
                ):
                    reaction = clamp(
                        abs(float(candle.close) - state.near_edge)
                        / atr
                    )
                    reason = (
                        "gap_inside_recovery"
                        if open_inside
                        else "same_bar_wick_rejection"
                    )
                    rejected = replace(
                        visited,
                        lifecycle=EntryLocationLifecycle.REJECTED,
                        state_started_at=candle.end,
                        state_duration_real_1m_bars=0,
                        rejected_at=candle.end,
                        reaction_atr=reaction,
                        transition_reason=reason,
                    )
                    self._locations[location_id] = rejected
                    self._append_step(
                        path_id,
                        kind="wick_rejection",
                        observed_at=candle.end,
                        source_event_id=None,
                        source_entity_id=location_id,
                        strength=reaction,
                        reason=reason,
                        step_transitions=step_transitions,
                        same_clock_relation="same_clock_known",
                    )
                elif self._adverse_side(
                    state.direction,
                    float(candle.close),
                    state.near_edge,
                ):
                    self._new_reacceptance(
                        symbol=state.symbol,
                        instrument_id=state.instrument_id,
                        context_kind="entry_zone",
                        context_id=location_id,
                        source_entity_id=state.source_zone_id,
                        direction=state.direction,
                        reference_price=state.near_edge,
                        failure_boundary=state.failure_boundary,
                        candle=candle,
                        atr=atr,
                        path_id=path_id,
                        step_transitions=step_transitions,
                    )
            elif source_failed:
                left = replace(
                    aged,
                    lifecycle=EntryLocationLifecycle.LEFT,
                    state_started_at=candle.end,
                    state_duration_real_1m_bars=0,
                    left_at=candle.end,
                    transition_reason=(
                        "fvg_invalidated"
                        if state.source_zone_kind == "fvg"
                        else "order_block_failed"
                    ),
                )
                self._locations[location_id] = left
                self._append_step(
                    path_id,
                    kind="location_left",
                    observed_at=candle.end,
                    source_event_id=state.source_zone_id,
                    source_entity_id=location_id,
                    strength=0.0,
                    reason=left.transition_reason,
                    step_transitions=step_transitions,
                )
            return

        if self._adverse_side(
            state.direction,
            float(candle.close),
            state.far_edge,
        ) or source_failed:
            reason = (
                (
                    "fvg_invalidated"
                    if state.source_zone_kind == "fvg"
                    else "order_block_failed"
                )
                if source_failed
                else "close_beyond_far_edge"
            )
            left = replace(
                aged,
                lifecycle=EntryLocationLifecycle.LEFT,
                state_started_at=candle.end,
                state_duration_real_1m_bars=0,
                left_at=candle.end,
                transition_reason=reason,
            )
            self._locations[location_id] = left
            self._append_step(
                path_id,
                kind="location_left",
                observed_at=candle.end,
                source_event_id=(
                    state.source_zone_id if source_failed else None
                ),
                source_entity_id=location_id,
                strength=0.0,
                reason=reason,
                step_transitions=step_transitions,
            )
            return

        reacceptance = self._context_reacceptance(
            "zone_return",
            location_id,
        )
        if reacceptance is not None:
            _, reacceptance_state = reacceptance
            if (
                reacceptance_state.lifecycle
                is QualifiedReacceptanceLifecycle.HELD
            ):
                rejected = replace(
                    aged,
                    lifecycle=EntryLocationLifecycle.REJECTED,
                    state_started_at=candle.end,
                    state_duration_real_1m_bars=0,
                    rejected_at=candle.end,
                    reaction_atr=reacceptance_state.strength,
                    transition_reason="qualified_reacceptance_held",
                )
                self._locations[location_id] = rejected
            return
        if self._delivery_side(
            state.direction,
            float(candle.close),
            state.near_edge,
        ):
            reaction = clamp(
                abs(float(candle.close) - state.near_edge) / atr
            )
            rejected = replace(
                aged,
                lifecycle=EntryLocationLifecycle.REJECTED,
                state_started_at=candle.end,
                state_duration_real_1m_bars=0,
                rejected_at=candle.end,
                reaction_atr=reaction,
                transition_reason="later_zone_rejection",
            )
            self._locations[location_id] = rejected
            self._append_step(
                path_id,
                kind="wick_rejection",
                observed_at=candle.end,
                source_event_id=None,
                source_entity_id=location_id,
                strength=reaction,
                reason="later_zone_rejection",
                step_transitions=step_transitions,
            )
        elif self._adverse_side(
            state.direction,
            float(candle.close),
            state.near_edge,
        ):
            self._new_reacceptance(
                symbol=state.symbol,
                instrument_id=state.instrument_id,
                context_kind="entry_zone",
                context_id=location_id,
                source_entity_id=state.source_zone_id,
                direction=state.direction,
                reference_price=state.near_edge,
                failure_boundary=state.failure_boundary,
                candle=candle,
                atr=atr,
                path_id=path_id,
                step_transitions=step_transitions,
            )

    def _pool_direction(
        self,
        state: ManipulationState,
    ) -> Direction:
        return (
            Direction.SHORT
            if state.side == "above"
            else Direction.LONG
        )

    def _new_pool_context(
        self,
        state: ManipulationState,
        candle: Candle,
        atr: float,
        m1_bos: Sequence[BreakOfStructureState],
        path_transitions: list[PathSequenceState],
        step_transitions: list[tuple[str, PathSequenceStep]],
    ) -> None:
        direction = self._pool_direction(state)
        path, first_step = self._new_path(
            symbol=state.symbol,
            instrument_id=state.instrument_id,
            context_kind="pool_reversal",
            context_id=state.manipulation_id,
            direction=direction,
            formed_at=state.swept_at,
            first_kind="pool_swept",
            source_event_id=state.manipulation_id,
            source_entity_id=state.manipulation_id,
            strength=clamp(state.penetration_atr),
            reason="typed_pool_manipulation_swept",
        )
        path_transitions.append(path)
        step_transitions.append((path.sequence_id, first_step))
        # Group 4 owns the pool reclaim/hold lifecycle. Group 5 records the
        # sweep context only and waits for the authoritative REACCEPTED state;
        # entry-zone reacceptance remains a separate Group 5 primitive.

    def _reference_for_context(
        self,
        context_kind: str,
        context_id: str,
        bos_id: str,
    ) -> MicroBOSReference | None:
        return next(
            (
                reference
                for reference in self._micro_references.values()
                if (
                    reference.context_kind == context_kind
                    and reference.context_id == context_id
                    and reference.bos_id == bos_id
                )
            ),
            None,
        )

    def _strict_reference_for_context(
        self,
        context_kind: str,
        context_id: str,
    ) -> tuple[MicroBOSReference, ...]:
        return tuple(
            reference
            for reference in self._micro_references.values()
            if (
                reference.context_kind == context_kind
                and reference.context_id == context_id
                and reference.relation == "strictly_after"
            )
        )

    def _bind_micro_bos(
        self,
        path_id: str,
        anchor_at: pd.Timestamp | None,
        current_end: pd.Timestamp,
        m1_bos: Sequence[BreakOfStructureState],
        step_transitions: list[tuple[str, PathSequenceStep]],
    ) -> tuple[MicroBOSReference, ...]:
        if anchor_at is None:
            return ()
        path = self._paths[path_id]
        strict_references = self._strict_reference_for_context(
            path.context_kind,
            path.context_id,
        )
        confirmed = tuple(
            state
            for state in m1_bos
            if (
                state.lifecycle is BOSLifecycle.CONFIRMED
                and state.resolved_at is not None
                and state.resolved_at >= anchor_at
            )
        )
        if strict_references:
            earliest_bound_clock = min(
                reference.resolved_at
                for reference in strict_references
            )
            unbound_prior = tuple(
                state
                for state in confirmed
                if (
                    state.resolved_at <= earliest_bound_clock
                    and self._reference_for_context(
                        path.context_kind,
                        path.context_id,
                        state.bos_id,
                    )
                    is None
                )
            )
            if unbound_prior:
                raise RuntimeError(
                    "Group 5 cannot revise its first bound M1 BOS clock"
                )
            return ()
        same_clock = tuple(
            state
            for state in confirmed
            if state.resolved_at == anchor_at
            and self._reference_for_context(
                path.context_kind,
                path.context_id,
                state.bos_id,
            )
            is None
        )
        if current_end > anchor_at and same_clock:
            raise RuntimeError(
                "Group 5 cannot backfill an anchor-clock M1 BOS"
            )
        created: list[MicroBOSReference] = []
        for state in sorted(same_clock, key=lambda item: item.bos_id):
            reference = MicroBOSReference(
                reference_id=_identity(
                    "group5-micro-bos-v1",
                    self.protocol.protocol_hash,
                    path.sequence_id,
                    state.bos_id,
                    "same_clock_unknown",
                ),
                protocol_hash=self.protocol.protocol_hash,
                context_kind=path.context_kind,
                context_id=path.context_id,
                expected_direction=path.direction,
                anchor_at=anchor_at,
                bos_id=state.bos_id,
                bos_direction=state.direction,
                target_swing_id=state.target_swing_id,
                scope=state.scope,
                pending_at=state.pending_at,
                resolved_at=state.resolved_at,
                relation="same_clock_unknown",
                outcome="simultaneous_unknown",
                qualified=False,
                strength=state.strength,
            )
            self._micro_references[reference.reference_id] = reference
            created.append(reference)
            self._append_step(
                path_id,
                kind="micro_bos_simultaneous",
                observed_at=state.resolved_at,
                source_event_id=state.bos_id,
                source_entity_id=state.target_swing_id,
                strength=state.strength,
                reason="anchor_clock_order_unknown",
                step_transitions=step_transitions,
                same_clock_relation="same_clock_unknown",
            )
        later = tuple(
            state
            for state in confirmed
            if state.resolved_at > anchor_at
        )
        late = tuple(
            state
            for state in later
            if state.resolved_at < current_end
        )
        if late:
            raise RuntimeError(
                "Group 5 cannot backfill a strictly-later M1 BOS"
            )
        later = tuple(
            state
            for state in later
            if state.resolved_at == current_end
        )
        if not later:
            return tuple(created)
        first_clock = min(state.resolved_at for state in later)
        first = tuple(
            sorted(
                (
                    state
                    for state in later
                    if state.resolved_at == first_clock
                ),
                key=lambda item: item.bos_id,
            )
        )
        ambiguous = len(first) != 1
        for state in first:
            aligned = state.direction is path.direction
            outcome = (
                "ambiguous_same_clock"
                if ambiguous
                else ("aligned" if aligned else "opposed")
            )
            reference = MicroBOSReference(
                reference_id=_identity(
                    "group5-micro-bos-v1",
                    self.protocol.protocol_hash,
                    path.sequence_id,
                    state.bos_id,
                    "strictly_after",
                ),
                protocol_hash=self.protocol.protocol_hash,
                context_kind=path.context_kind,
                context_id=path.context_id,
                expected_direction=path.direction,
                anchor_at=anchor_at,
                bos_id=state.bos_id,
                bos_direction=state.direction,
                target_swing_id=state.target_swing_id,
                scope=state.scope,
                pending_at=state.pending_at,
                resolved_at=state.resolved_at,
                relation="strictly_after",
                outcome=outcome,
                qualified=bool(not ambiguous and aligned),
                strength=state.strength,
            )
            self._micro_references[reference.reference_id] = reference
            created.append(reference)
            kind = (
                "micro_bos_ambiguous"
                if ambiguous
                else (
                    "micro_bos_confirmed"
                    if aligned
                    else "micro_bos_opposed"
                )
            )
            self._append_step(
                path_id,
                kind=kind,
                observed_at=state.resolved_at,
                source_event_id=state.bos_id,
                source_entity_id=state.target_swing_id,
                strength=state.strength,
                reason=outcome,
                step_transitions=step_transitions,
            )
        return tuple(created)

    def _bind_pool_opposite_displacement(
        self,
        path_id: str,
        sources: Sequence[_ZoneSource],
        candle: Candle,
        step_transitions: list[tuple[str, PathSequenceStep]],
    ) -> PathSequenceStep | None:
        path = self._paths[path_id]
        existing = next(
            (
                step
                for step in path.steps
                if step.kind == "opposite_displacement"
            ),
            None,
        )
        if existing is not None:
            return existing
        reaccepted_step = next(
            (
                step
                for step in path.steps
                if step.kind == "reacceptance_held"
            ),
            None,
        )
        if reaccepted_step is None:
            return None
        reaccepted_at = reaccepted_step.observed_at
        eligible = tuple(
            source
            for source in sources
            if (
                source.can_register
                and source.direction is path.direction
                and source.source_displacement_id is not None
                and source.source_displacement_active_at is not None
                and source.source_displacement_active_at > reaccepted_at
                and source.confirmed_at > reaccepted_at
            )
        )
        late = tuple(
            source
            for source in eligible
            if source.confirmed_at < candle.end
        )
        if late:
            raise RuntimeError(
                "Group 5 cannot backfill opposite displacement evidence"
            )
        current = tuple(
            source
            for source in eligible
            if source.confirmed_at == candle.end
        )
        displacement_ids = tuple(
            dict.fromkeys(
                source.source_displacement_id for source in current
            )
        )
        if len(displacement_ids) != 1:
            if len(displacement_ids) > 1:
                source = min(
                    current,
                    key=lambda value: (value.kind, value.source_id),
                )
                self._append_step(
                    path_id,
                    kind="opposite_displacement_ambiguous",
                    observed_at=candle.end,
                    source_event_id=source.source_id,
                    source_entity_id=_identity(
                        "ambiguous-displacement",
                        *sorted(displacement_ids),
                    ),
                    strength=0.0,
                    reason="multiple_opposite_displacements_same_clock",
                    step_transitions=step_transitions,
                )
            return None
        displacement_id = displacement_ids[0]
        source = min(
            (
                value
                for value in current
                if value.source_displacement_id == displacement_id
            ),
            key=lambda value: (value.kind, value.source_id),
        )
        self._append_step(
            path_id,
            kind="opposite_displacement",
            observed_at=source.confirmed_at,
            source_event_id=source.source_id,
            source_entity_id=displacement_id,
            strength=0.0,
            reason="displacement_linked_zone_after_reacceptance",
            step_transitions=step_transitions,
        )
        return self._paths[path_id].steps[-1]

    def _path_has_step(
        self,
        path_id: str,
        kinds: set[str],
    ) -> bool:
        return any(
            step.kind in kinds for step in self._paths[path_id].steps
        )

    def _age_paths(self, candle: Candle) -> None:
        for path_id, state in tuple(self._paths.items()):
            if state.lifecycle is not PathSequenceLifecycle.ACTIVE:
                continue
            self._paths[path_id] = replace(
                state,
                last_updated_at=candle.end,
                age_real_1m_bars=state.age_real_1m_bars + 1,
                state_duration_real_1m_bars=(
                    state.state_duration_real_1m_bars + 1
                ),
            )

    def _apply_real(
        self,
        candle: Candle,
        sources: Sequence[_ZoneSource],
        manipulations: Sequence[ManipulationState],
        m1_bos: Sequence[BreakOfStructureState],
        inventory: Sequence[LiquidityInventoryItem],
        atr: float,
    ) -> Group5Update:
        path_transitions: list[PathSequenceState] = []
        step_transitions: list[tuple[str, PathSequenceStep]] = []
        source_by_id = {source.source_id: source for source in sources}
        manipulation_by_id = {
            state.manipulation_id: state for state in manipulations
        }
        deferred_source_failures: list[
            tuple[str, str, str]
        ] = []
        cold_source_ids: list[str] = []
        self._age_paths(candle)

        # Existing reacceptances advance before their locations consume the
        # resulting held/failed state.
        for key, state in tuple(self._reacceptances.items()):
            path_kind = "zone_return"
            path_id = self._path_id(
                symbol=state.symbol,
                instrument_id=state.instrument_id,
                context_kind=path_kind,
                context_id=state.context_id,
            )
            if (
                path_id not in self._paths
                or self._paths[path_id].lifecycle
                is not PathSequenceLifecycle.ACTIVE
            ):
                if not self._reacceptance_terminal(state):
                    raise RuntimeError(
                        "closed Group 5 path retained a live reacceptance"
                    )
                continue
            location = self._locations[state.context_id]
            source = source_by_id.get(location.source_zone_id)
            source_failed = bool(source is not None and source.failed)
            if source_failed:
                self._reacceptances[key] = replace(
                    state,
                    last_updated_at=candle.end,
                    age_real_1m_bars=state.age_real_1m_bars + 1,
                    state_duration_real_1m_bars=(
                        state.state_duration_real_1m_bars + 1
                    ),
                )
                deferred_source_failures.append(
                    (
                        key,
                        "source_invalidated",
                        path_id,
                    )
                )
            else:
                self._advance_reacceptance(
                    key,
                    candle,
                    atr,
                    path_id=path_id,
                    step_transitions=step_transitions,
                )

        for location_id, location in tuple(self._locations.items()):
            path_id = self._path_id(
                symbol=location.symbol,
                instrument_id=location.instrument_id,
                context_kind="zone_return",
                context_id=location_id,
            )
            if (
                path_id in self._paths
                and self._paths[path_id].lifecycle
                is PathSequenceLifecycle.ACTIVE
            ):
                self._advance_location(
                    location_id,
                    candle,
                    atr,
                    inventory,
                    source_by_id.get(location.source_zone_id),
                    path_transitions,
                    step_transitions,
                )

        # Project the one authoritative Group 4 resolution into the path.
        for path_id, path in tuple(self._paths.items()):
            if (
                path.lifecycle is not PathSequenceLifecycle.ACTIVE
                or path.context_kind != "pool_reversal"
            ):
                continue
            manipulation = manipulation_by_id.get(path.context_id)
            if (
                manipulation is not None
                and manipulation.lifecycle
                is ManipulationLifecycle.ACCEPTED_OUTSIDE
            ):
                if manipulation.accepted_outside_at < candle.end and not (
                    self._path_has_step(path_id, {"accepted_outside"})
                ):
                    raise RuntimeError(
                        "Group 5 cannot backfill accepted-outside state"
                    )
                if not self._path_has_step(path_id, {"accepted_outside"}):
                    self._append_step(
                        path_id,
                        kind="accepted_outside",
                        observed_at=manipulation.accepted_outside_at,
                        source_event_id=manipulation.manipulation_id,
                        source_entity_id=manipulation.manipulation_id,
                        strength=0.0,
                        reason="group4_accepted_outside",
                        step_transitions=step_transitions,
                    )
            elif (
                manipulation is not None
                and manipulation.lifecycle
                is ManipulationLifecycle.REACCEPTED
                and not self._path_has_step(
                    path_id,
                    {"reacceptance_held"},
                )
            ):
                if manipulation.reaccepted_at < candle.end:
                    raise RuntimeError(
                        "Group 5 cannot backfill manipulation reacceptance"
                    )
                self._append_step(
                    path_id,
                    kind="reacceptance_held",
                    observed_at=manipulation.reaccepted_at,
                    source_event_id=manipulation.manipulation_id,
                    source_entity_id=manipulation.manipulation_id,
                    strength=clamp(manipulation.penetration_atr),
                    reason="group4_reentry_held",
                    step_transitions=step_transitions,
                )
            elif (
                manipulation is not None
                and manipulation.lifecycle
                is ManipulationLifecycle.SWEPT
                and manipulation.deadline_elapsed
            ):
                if manipulation.censored_at < candle.end:
                    raise RuntimeError(
                        "Group 5 cannot backfill manipulation deadline"
                    )
                self._close_path(
                    path_id,
                    manipulation.censored_at,
                    "manipulation_resolution_deadline",
                    path_transitions,
                )

        for path_id, path in tuple(self._paths.items()):
            if (
                path.lifecycle is not PathSequenceLifecycle.ACTIVE
                or path.context_kind != "pool_reversal"
            ):
                continue
            self._bind_pool_opposite_displacement(
                path_id,
                sources,
                candle,
                step_transitions,
            )

        # Typed source transitions are recorded before their derived
        # reacceptance failure at the same completed clock.
        for key, reason, path_id in deferred_source_failures:
            if not self._reacceptance_terminal(
                self._reacceptances[key]
            ):
                self._fail_reacceptance(
                    key,
                    candle,
                    reason,
                    path_id,
                    step_transitions,
                )

        # Bind BOS after all same-clock price and typed-source changes, but
        # before closure so failure and BOS can both remain visible.
        for path_id, path in tuple(self._paths.items()):
            if path.lifecycle is not PathSequenceLifecycle.ACTIVE:
                continue
            anchor = None
            if path.context_kind == "zone_return":
                anchor = self._locations[path.context_id].first_entered_at
            else:
                manipulation = manipulation_by_id.get(path.context_id)
                displacement_step = next(
                    (
                        step
                        for step in self._paths[path_id].steps
                        if step.kind == "opposite_displacement"
                    ),
                    None,
                )
                anchor = (
                    None
                    if displacement_step is None
                    else displacement_step.observed_at
                )
            self._bind_micro_bos(
                path_id=path_id,
                anchor_at=anchor,
                current_end=candle.end,
                m1_bos=m1_bos,
                step_transitions=step_transitions,
            )

        # Failure dominates a same-clock BOS; otherwise close only after the
        # complete context-specific first trigger/contradiction is recorded.
        for path_id, path in tuple(self._paths.items()):
            if path.lifecycle is not PathSequenceLifecycle.ACTIVE:
                continue
            kinds = {step.kind for step in self._paths[path_id].steps}
            if path.context_kind == "zone_return":
                location = self._locations[path.context_id]
                reacceptance = self._context_reacceptance(
                    "zone_return",
                    path.context_id,
                )
                if location.lifecycle is EntryLocationLifecycle.LEFT:
                    reason = "location_left"
                elif (
                    reacceptance is not None
                    and reacceptance[1].lifecycle
                    is QualifiedReacceptanceLifecycle.FAILED
                ):
                    reason = "reacceptance_failed"
                elif "micro_bos_ambiguous" in kinds:
                    reason = "micro_bos_ambiguous_same_clock"
                elif "micro_bos_opposed" in kinds:
                    reason = "micro_bos_opposed"
                elif "micro_bos_confirmed" in kinds:
                    reason = "micro_bos_aligned"
                elif "reacceptance_held" in kinds:
                    milestone = next(
                        step
                        for step in reversed(self._paths[path_id].steps)
                        if step.kind == "reacceptance_held"
                    )
                    self._mark_active_path_milestone(
                        path_id,
                        milestone.observed_at,
                        "qualified_reacceptance_held",
                    )
                    continue
                elif "wick_rejection" in kinds:
                    milestone = next(
                        step
                        for step in reversed(self._paths[path_id].steps)
                        if step.kind == "wick_rejection"
                    )
                    self._mark_active_path_milestone(
                        path_id,
                        milestone.observed_at,
                        "zone_rejection_observed",
                    )
                    continue
                else:
                    continue
                if (
                    reacceptance is not None
                    and not self._reacceptance_terminal(reacceptance[1])
                ):
                    self._fail_reacceptance(
                        reacceptance[0],
                        candle,
                        "context_closed_before_hold",
                        path_id,
                        step_transitions,
                    )
            else:
                if "accepted_outside" in kinds:
                    reason = "accepted_outside"
                elif "opposite_displacement_ambiguous" in kinds:
                    reason = "opposite_displacement_ambiguous_same_clock"
                elif "micro_bos_ambiguous" in kinds:
                    reason = "micro_bos_ambiguous_same_clock"
                elif "micro_bos_opposed" in kinds:
                    reason = "micro_bos_opposed"
                else:
                    has_bos = "micro_bos_confirmed" in kinds
                    returned = bool(
                        "reacceptance_held" in kinds
                    )
                    has_displacement = (
                        "opposite_displacement" in kinds
                    )
                    if not (
                        has_bos and returned and has_displacement
                    ):
                        continue
                    reason = "pool_reversal_sequence_observed"
            self._close_path(
                path_id,
                candle.end,
                reason,
                path_transitions,
            )

        # Register current-clock sources last. Zones cannot self-visit or
        # resolve. A new swept-pool context may record only anchor-clock BOS
        # as simultaneous/unknown; it cannot bind an ordered BOS.
        registered_zone_ids = {
            state.source_zone_id for state in self._locations.values()
        }
        for source in sources:
            if (
                source.confirmed_at == candle.end
                and source.can_register
            ):
                self._new_zone_context(
                    source,
                    candle,
                    inventory,
                    path_transitions,
                    step_transitions,
                )
            elif source.source_id not in registered_zone_ids:
                cold_source_ids.append(source.source_id)
        registered_pool_ids = {
            path.context_id
            for path in self._paths.values()
            if path.context_kind == "pool_reversal"
        }
        for state in manipulations:
            if (
                state.source_kind == "formed_liquidity_pool"
                and state.lifecycle is ManipulationLifecycle.SWEPT
                and state.censored_at is None
                and state.swept_at == candle.end
            ):
                self._new_pool_context(
                    state,
                    candle,
                    atr,
                    m1_bos,
                    path_transitions,
                    step_transitions,
                )
            elif state.manipulation_id not in registered_pool_ids:
                cold_source_ids.append(state.manipulation_id)

        output = self._snapshot()
        output = replace(
            output,
            path_transitions=tuple(path_transitions),
            step_transitions=tuple(step_transitions),
            cold_source_ids=tuple(sorted(set(cold_source_ids))),
        )
        # Only terminal contexts from a prior successful output are eligible
        # for next-bar compaction.
        self._exposed_terminal_path_ids.update(
            path_id
            for path_id, state in self._paths.items()
            if state.lifecycle
            in {
                PathSequenceLifecycle.CLOSED,
                PathSequenceLifecycle.CENSORED,
            }
        )
        return output

    def on_completed_1m(
        self,
        candle: Candle,
        *,
        fair_value_gaps: Iterable[FairValueGapState] = (),
        order_blocks: Iterable[OrderBlockState] = (),
        manipulations: Iterable[ManipulationState] = (),
        m1_bos: Iterable[BreakOfStructureState] = (),
        liquidity_inventory: Iterable[LiquidityInventoryItem] = (),
        m1_atr: float,
    ) -> Group5Update:
        if (
            not isinstance(candle, Candle)
            or candle.timeframe is not Timeframe.M1
            or not candle.complete
            or candle.expected_minutes != 1
            or candle.observed_minutes != 1
        ):
            raise ValueError("Group 5 requires one completed 1m candle")
        atr = float(m1_atr)
        if not math.isfinite(atr) or atr <= 0:
            raise ValueError("Group 5 requires a positive causal M1 ATR")
        fvg_states = tuple(fair_value_gaps)
        order_block_states = tuple(order_blocks)
        manipulation_states = tuple(manipulations)
        bos_states = tuple(m1_bos)
        inventory = tuple(liquidity_inventory)
        sources = tuple(
            self._zone_source(state)
            for state in (*fvg_states, *order_block_states)
        )
        input_value = (
            candle,
            fvg_states,
            order_block_states,
            manipulation_states,
            bos_states,
            inventory,
            atr,
        )
        if (
            self._last_input == input_value
            and self._last_output is not None
        ):
            return self._last_output
        if (
            self._last_raw_end is not None
            and candle.end <= self._last_raw_end
        ):
            raise ValueError(
                "duplicate or out-of-order Group 5 completed candle"
            )
        candidate = self._transaction_clone()
        candidate._validate_sources(
            candle,
            sources,
            manipulation_states,
            bos_states,
            inventory,
        )
        if candle.real_completed:
            output = candidate._apply_real(
                candle,
                sources,
                manipulation_states,
                bos_states,
                inventory,
                atr,
            )
        else:
            output = candidate._snapshot()
        candidate._last_raw_end = candle.end
        candidate._last_input = input_value
        candidate._last_output = output
        candidate._last_boundary_input = None
        candidate._last_boundary_output = None
        self._commit(candidate)
        return output

    def on_boundary(
        self,
        reason: str,
        observed_at: pd.Timestamp,
        *,
        symbol: str,
        instrument_id: int,
    ) -> Group5Update:
        clock = aware_timestamp(
            observed_at,
            name="group5.boundary.observed_at",
        )
        if (
            not symbol
            or type(instrument_id) is not int
            or instrument_id < 0
        ):
            raise ValueError("Group 5 boundary identity is invalid")
        boundary_identity = (symbol, instrument_id)
        input_value = (reason, clock, boundary_identity)
        if (
            input_value == self._last_boundary_input
            and self._last_boundary_output is not None
        ):
            return self._last_boundary_output
        if reason not in GROUP5_HARD_BOUNDARY_REASONS:
            raise ValueError("Group 5 boundary reason is not hard")
        if self._last_raw_end is not None and clock <= self._last_raw_end:
            raise ValueError("Group 5 boundary is duplicate or out of order")
        if self._identity is not None and (
            (
                reason == "contract_change_reset"
                and boundary_identity == self._identity
            )
            or (
                reason != "contract_change_reset"
                and boundary_identity != self._identity
            )
        ):
            raise ValueError(
                "Group 5 boundary identity contradicts its reason"
            )
        candidate = self._transaction_clone()
        transitions: list[PathSequenceState] = []
        reacceptance_transitions: list[
            QualifiedReacceptanceState
        ] = []
        for state in candidate._reacceptances.values():
            if candidate._reacceptance_terminal(state):
                continue
            path_kind = (
                "zone_return"
                if state.context_kind == "entry_zone"
                else "pool_reversal"
            )
            path_id = candidate._path_id(
                symbol=state.symbol,
                instrument_id=state.instrument_id,
                context_kind=path_kind,
                context_id=state.context_id,
            )
            path = candidate._paths.get(path_id)
            if (
                path is None
                or path.lifecycle is not PathSequenceLifecycle.ACTIVE
                or (path.symbol, path.instrument_id)
                != (state.symbol, state.instrument_id)
            ):
                raise RuntimeError(
                    "live Group 5 reacceptance lacks its paired path"
                )
        for path_id, state in tuple(candidate._paths.items()):
            if state.lifecycle is not PathSequenceLifecycle.ACTIVE:
                continue
            censored = replace(
                state,
                lifecycle=PathSequenceLifecycle.CENSORED,
                state_started_at=clock,
                last_updated_at=clock,
                state_duration_real_1m_bars=0,
                ended_at=clock,
                transition_reason=reason,
            )
            candidate._paths[path_id] = censored
            transitions.append(censored)
        for key, state in tuple(candidate._reacceptances.items()):
            if candidate._reacceptance_terminal(state):
                continue
            censored = replace(
                state,
                lifecycle=QualifiedReacceptanceLifecycle.CENSORED,
                state_started_at=clock,
                last_updated_at=clock,
                state_duration_real_1m_bars=0,
                censored_at=clock,
                transition_reason="hard_boundary_censored",
            )
            candidate._reacceptances[key] = censored
            reacceptance_transitions.append(censored)
        output = Group5Update(
            entry_locations=(),
            qualified_reacceptances=(),
            micro_bos_references=(),
            path_sequences=(),
            path_transitions=tuple(transitions),
            reacceptance_transitions=tuple(
                reacceptance_transitions
            ),
            boundary_reason=reason,
        )
        candidate._locations.clear()
        candidate._reacceptances.clear()
        candidate._micro_references.clear()
        candidate._paths.clear()
        candidate._path_order.clear()
        candidate._exposed_terminal_path_ids.clear()
        candidate._identity = None
        candidate._last_raw_end = clock
        candidate._last_input = None
        candidate._last_output = None
        candidate._last_boundary_input = input_value
        candidate._last_boundary_output = output
        self._commit(candidate)
        return output


__all__ = [
    "CausalGroup5Reducer",
    "Group5Protocol",
    "Group5Update",
]
