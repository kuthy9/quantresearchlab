"""Causal continuous SMC trader, product version 1."""

from .engine import ContinuousSMCEngine
from .model import (
    Action,
    Bar,
    Direction,
    Playbook,
    PlaybookPhase,
)

__all__ = [
    "Action",
    "Bar",
    "ContinuousSMCEngine",
    "Direction",
    "Playbook",
    "PlaybookPhase",
]

__version__ = "1.0.0"
