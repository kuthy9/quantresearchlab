"""Causal continuous SMC trader, product version 3."""

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

__version__ = "3.0.0"
