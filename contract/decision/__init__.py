"""Decision-layer contracts.

``opportunity`` is the LLM Brain's boundary (aliases in, prices out);
``action`` is the retired typed vertical's action vocabulary, kept for the
neutral-projection consumers in ``shares/``."""
from __future__ import annotations

from .action import (
    Action,
    ActionUtility,
    Decision,
)
from .opportunity import (
    GeometryError,
    OpportunityGeometry,
    OpportunityProposal,
)

__all__ = [
    "Action",
    "ActionUtility",
    "Decision",
    "GeometryError",
    "OpportunityGeometry",
    "OpportunityProposal",
]
