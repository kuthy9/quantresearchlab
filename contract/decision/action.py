"""Decision contracts: the action vocabulary and the utility comparison result."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import pandas as pd

from contract.market.primitives import aware_timestamp
from contract.brain.plan import TradePlan


class Action(str, Enum):
    ENTER = "enter"
    WAIT = "wait"
    HOLD = "hold"
    PROTECT = "protect"
    EXIT = "exit"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class ActionUtility:
    action: Action
    utility: float
    components: Mapping[str, float]
    hypothesis_key: str | None
    reason: str

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.utility)):
            raise ValueError("action utility must be finite")
        if any(not math.isfinite(float(value)) for value in self.components.values()):
            raise ValueError("action utility components must be finite")


@dataclass(frozen=True)
class Decision:
    asof: pd.Timestamp
    selected_action: Action
    utilities: tuple[ActionUtility, ...]
    best_hypothesis_key: str | None
    advantage: float
    reasons: tuple[str, ...]
    plan: TradePlan | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="decision.asof"))
        if not self.utilities or not math.isfinite(float(self.advantage)):
            raise ValueError("decision requires finite compared utilities")


__all__ = [
    "Action",
    "ActionUtility",
    "Decision",
]
