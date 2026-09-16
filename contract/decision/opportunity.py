"""The opportunity boundary between the LLM Brain and Risk / Execution.

The Brain proposes an ``Opportunity`` in aliases only; ``OpportunityGeometry``
is what deterministic code derives from the Eye's object geometry — the first
thing a Risk engine reads and the last thing the LLM never writes."""
from __future__ import annotations

from dataclasses import dataclass
import math

from contract.brain.state import Opportunity

OpportunityProposal = Opportunity


class GeometryError(ValueError):
    """An opportunity cannot be resolved to prices on the current objects."""


@dataclass(frozen=True)
class OpportunityGeometry:
    entry_price: float
    stop_price: float
    target_price: float
    reward_risk: float
    rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("entry_price", "stop_price", "target_price", "reward_risk"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"opportunity geometry {name} must be finite")
            object.__setattr__(self, name, value)
        if self.reward_risk <= 0.0:
            raise ValueError("opportunity geometry reward_risk must be positive")
        object.__setattr__(self, "rule_ids", tuple(self.rule_ids))

    def to_dict(self) -> dict:
        return {
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "reward_risk": self.reward_risk,
            "rule_ids": list(self.rule_ids),
        }


__all__ = ["GeometryError", "OpportunityGeometry", "OpportunityProposal"]
