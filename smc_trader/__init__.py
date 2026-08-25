"""Causal continuous SMC trader, product version 1."""

from .engine import ContinuousSMCEngine
from .execution_fsm import ExecutionFSM, RiskApprovedTradeIntent
from .event_store import EventStore
from .market_state import (
    MarketSnapshot,
    MarketSnapshotAuthority,
    RelationState,
    SessionState,
    TimeframeState,
)
from .model import (
    Action,
    Bar,
    Direction,
    Playbook,
    PlaybookPhase,
    SMC_SEMANTIC_VERSION,
)
from .semantics import SemanticRegistry

__all__ = [
    "Action",
    "Bar",
    "ContinuousSMCEngine",
    "Direction",
    "ExecutionFSM",
    "EventStore",
    "MarketSnapshot",
    "MarketSnapshotAuthority",
    "Playbook",
    "PlaybookPhase",
    "RelationState",
    "RiskApprovedTradeIntent",
    "SMC_SEMANTIC_VERSION",
    "SemanticRegistry",
    "SessionState",
    "TimeframeState",
]

__version__ = "1.0.0"
