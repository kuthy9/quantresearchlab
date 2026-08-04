"""Preregistered unique-action comparison across correlated playbook evidence."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .decision import DecisionConfig, UtilityDecisionLayer
from .model import (
    Action,
    ActionUtility,
    HypothesisBelief,
    MarketBelief,
    MarketObservation,
    Playbook,
    PlaybookPhase,
    TradePlan,
)


class PlanRelation(str, Enum):
    EQUIVALENT = "equivalent_action"
    COMPETING_STRUCTURAL_STOP = "competing_structural_stop"
    COMPETING_STOP_PROVENANCE = "competing_stop_provenance"
    COMPETING_TARGET = "competing_target"
    COMPETING_DEADLINE = "competing_deadline"
    DIFFERENT_ENTRY = "different_entry"
    OPPOSITE_DIRECTION = "opposite_direction"


@dataclass(frozen=True)
class ActionEquivalenceProtocol:
    version: str
    fingerprint: str
    tick_size: float
    status: str

    @classmethod
    def from_file(
        cls,
        path: str | Path = "configs/action_equivalence_v2_2.json",
    ) -> "ActionEquivalenceProtocol":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("action-equivalence protocol root must be an object")
        version = str(payload.get("protocol_version", ""))
        status = str(payload.get("status", ""))
        if not version.startswith("2.2.") or status != "preregistered_before_v2_2_fit":
            raise ValueError("action-equivalence protocol is not frozen for v2.2")
        identity = payload.get("action_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("action-equivalence protocol omits action identity")
        fields = identity.get("fields")
        expected = [
            "verb",
            "direction",
            "planned_entry_tick",
            "invalidation_tick",
            "invalidation_source_level_id",
            "primary_target_tick",
            "primary_target_level_id",
            "deadline_minute",
        ]
        if fields != expected or bool(identity.get("playbook_is_part_of_identity")):
            raise ValueError("action-equivalence identity fields were changed")
        tick_size = float(identity.get("tick_size", 0.0))
        if not math.isfinite(tick_size) or tick_size <= 0:
            raise ValueError("action-equivalence tick size must be positive")
        return cls(
            version=version,
            fingerprint=hashlib.sha256(raw).hexdigest(),
            tick_size=tick_size,
            status=status,
        )


@dataclass(frozen=True)
class ActionPlanIdentity:
    verb: Action
    direction: str
    entry_tick: int
    invalidation_tick: int
    invalidation_source_level_id: str
    primary_target_tick: int
    primary_target_level_id: str
    deadline_minute_utc: str

    @property
    def key(self) -> str:
        payload = {
            "verb": self.verb.value,
            "direction": self.direction,
            "entry_tick": self.entry_tick,
            "invalidation_tick": self.invalidation_tick,
            "invalidation_source_level_id": self.invalidation_source_level_id,
            "primary_target_tick": self.primary_target_tick,
            "primary_target_level_id": self.primary_target_level_id,
            "deadline_minute_utc": self.deadline_minute_utc,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


@dataclass(frozen=True)
class EquivalentActionGroup:
    identity: ActionPlanIdentity
    representative: ActionUtility
    hypothesis_keys: tuple[str, ...]
    probabilities: tuple[float, ...]

    @property
    def evidence_count(self) -> int:
        return len(self.hypothesis_keys)


def action_utility_is_ready(
    utility: ActionUtility,
    belief: MarketBelief,
) -> bool:
    if utility.action not in {Action.ENTER, Action.WAIT}:
        return True
    if utility.hypothesis_key is None:
        return False
    hypothesis = belief.hypotheses.get(utility.hypothesis_key)
    if hypothesis is None:
        return False
    if utility.action is Action.ENTER:
        return bool(
            hypothesis.plan is not None
            and hypothesis.phase is PlaybookPhase.EXECUTABLE
            and (
                not hypothesis.hard_gate_results
                or all(hypothesis.hard_gate_results.values())
            )
        )
    return hypothesis.phase in {
        PlaybookPhase.FORMING,
        PlaybookPhase.ARMED,
        PlaybookPhase.WAITING_LOCATION,
        PlaybookPhase.WAITING_TRIGGER,
        PlaybookPhase.WAITING_PULLBACK,
        PlaybookPhase.EXECUTABLE,
    }


def _tick(price: float, tick_size: float) -> int:
    value = float(price)
    if not math.isfinite(value):
        raise ValueError("action-equivalence plan price is not finite")
    ticks = int(round(value / tick_size))
    if abs(value - ticks * tick_size) > max(1e-9, tick_size * 1e-7):
        raise ValueError("action-equivalence plan price is not tick aligned")
    return ticks


def _deadline_minute_utc(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("action-equivalence deadline must be timezone aware")
    return timestamp.tz_convert("UTC").floor("min").isoformat()


def action_plan_identity(
    action: Action,
    plan: TradePlan,
    *,
    tick_size: float,
) -> ActionPlanIdentity:
    if action not in {Action.ENTER, Action.WAIT}:
        raise ValueError("only enter/wait plans have a flat action identity")
    return ActionPlanIdentity(
        verb=action,
        direction=plan.direction.value,
        entry_tick=_tick(plan.planned_entry, tick_size),
        invalidation_tick=_tick(plan.invalidation.price, tick_size),
        invalidation_source_level_id=plan.invalidation.source_level_id,
        primary_target_tick=_tick(plan.targets[0].price, tick_size),
        primary_target_level_id=plan.targets[0].level_id,
        deadline_minute_utc=_deadline_minute_utc(plan.deadline),
    )


def plan_relation(
    left: TradePlan,
    right: TradePlan,
    *,
    tick_size: float,
) -> PlanRelation:
    if left.direction is not right.direction:
        return PlanRelation.OPPOSITE_DIRECTION
    if _tick(left.planned_entry, tick_size) != _tick(
        right.planned_entry,
        tick_size,
    ):
        return PlanRelation.DIFFERENT_ENTRY
    if _deadline_minute_utc(left.deadline) != _deadline_minute_utc(
        right.deadline
    ):
        return PlanRelation.COMPETING_DEADLINE
    if (
        _tick(left.targets[0].price, tick_size)
        != _tick(right.targets[0].price, tick_size)
        or left.targets[0].level_id != right.targets[0].level_id
    ):
        return PlanRelation.COMPETING_TARGET
    if _tick(left.invalidation.price, tick_size) != _tick(
        right.invalidation.price,
        tick_size,
    ):
        return PlanRelation.COMPETING_STRUCTURAL_STOP
    if (
        left.invalidation.source_level_id
        != right.invalidation.source_level_id
    ):
        return PlanRelation.COMPETING_STOP_PROVENANCE
    return PlanRelation.EQUIVALENT


def equivalent_action_groups(
    utilities: Sequence[ActionUtility],
    belief: MarketBelief,
    *,
    tick_size: float,
) -> tuple[EquivalentActionGroup, ...]:
    grouped: dict[ActionPlanIdentity, list[ActionUtility]] = {}
    for utility in utilities:
        if (
            utility.action not in {Action.ENTER, Action.WAIT}
            or utility.hypothesis_key is None
        ):
            continue
        hypothesis = belief.hypotheses.get(utility.hypothesis_key)
        if hypothesis is None or hypothesis.plan is None:
            continue
        identity = action_plan_identity(
            utility.action,
            hypothesis.plan,
            tick_size=tick_size,
        )
        grouped.setdefault(identity, []).append(utility)
    output: list[EquivalentActionGroup] = []
    for identity, members in grouped.items():
        ordered = sorted(
            members,
            key=lambda item: (
                -float(item.utility),
                item.hypothesis_key or "",
            ),
        )
        keys = tuple(
            sorted(
                item.hypothesis_key
                for item in members
                if item.hypothesis_key is not None
            )
        )
        probabilities = tuple(
            sorted(
                (
                    float(belief.hypotheses[key].probability)
                    for key in keys
                ),
                reverse=True,
            )
        )
        output.append(
            EquivalentActionGroup(
                identity=identity,
                representative=ordered[0],
                hypothesis_keys=keys,
                probabilities=probabilities,
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                item.identity.verb.value,
                item.identity.key,
            ),
        )
    )


def collapse_equivalent_action_utilities(
    utilities: Sequence[ActionUtility],
    belief: MarketBelief,
    *,
    tick_size: float,
) -> list[ActionUtility]:
    groups = equivalent_action_groups(
        utilities,
        belief,
        tick_size=tick_size,
    )
    grouped_members = {
        (group.identity.verb, key)
        for group in groups
        for key in group.hypothesis_keys
    }
    output = [
        utility
        for utility in utilities
        if (utility.action, utility.hypothesis_key) not in grouped_members
    ]
    for group in groups:
        representative = group.representative
        probabilities = group.probabilities
        maximum = probabilities[0]
        second = probabilities[1] if len(probabilities) > 1 else 0.0
        minimum = probabilities[-1]
        components = dict(representative.components)
        components.update(
            {
                "equivalent_hypothesis_count": float(group.evidence_count),
                "equivalent_probability_max": maximum,
                "equivalent_probability_second": second,
                "equivalent_probability_dispersion": maximum - minimum,
            }
        )
        output.append(
            ActionUtility(
                action=representative.action,
                utility=float(representative.utility),
                components=components,
                hypothesis_key=representative.hypothesis_key,
                reason=(
                    "unique executable action; correlated evidence "
                    f"[{', '.join(group.hypothesis_keys)}]; conservative "
                    f"utility=max constituent={representative.utility:.3f}R; "
                    f"{representative.reason}"
                ),
            )
        )
    return output


class ActionEquivalenceDecisionLayer(UtilityDecisionLayer):
    """Compare unique action plans while retaining all playbook beliefs."""

    def __init__(
        self,
        protocol: ActionEquivalenceProtocol,
        config: DecisionConfig | None = None,
        *,
        disabled_playbooks: Sequence[Playbook] = (),
    ) -> None:
        super().__init__(config)
        self.protocol = protocol
        self.disabled_playbooks = frozenset(disabled_playbooks)
        if len(self.disabled_playbooks) >= len(Playbook):
            raise ValueError("at least one playbook must remain enabled")

    def _flat_utilities(
        self,
        observation: MarketObservation,
        belief: MarketBelief,
    ) -> list[ActionUtility]:
        enabled = {
            key: hypothesis
            for key, hypothesis in belief.hypotheses.items()
            if hypothesis.playbook not in self.disabled_playbooks
        }
        filtered = MarketBelief(asof=belief.asof, hypotheses=enabled)
        base = [
            utility
            for utility in super()._flat_utilities(observation, filtered)
            if action_utility_is_ready(utility, filtered)
        ]
        return collapse_equivalent_action_utilities(
            base,
            filtered,
            tick_size=self.protocol.tick_size,
        )


def action_equivalence_code_fingerprint() -> str:
    source = Path(__file__)
    return hashlib.sha256(source.read_bytes()).hexdigest()


__all__ = [
    "ActionEquivalenceDecisionLayer",
    "ActionEquivalenceProtocol",
    "ActionPlanIdentity",
    "EquivalentActionGroup",
    "PlanRelation",
    "action_equivalence_code_fingerprint",
    "action_plan_identity",
    "action_utility_is_ready",
    "collapse_equivalent_action_utilities",
    "equivalent_action_groups",
    "plan_relation",
]
