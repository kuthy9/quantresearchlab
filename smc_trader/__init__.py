"""Causal continuous SMC trader, product version 1.

The public surface is the Trading Eye's published market view plus the
runtime entry points that consume it.  ``EventStore`` is the Eye's internal
history authority and is deliberately not exported here; import
``smc_trader.event_store`` directly when a research or replay tool needs
read-only access to event lineage."""

from .engine import ContinuousSMCEngine
from .execution_fsm import ExecutionFSM, RiskApprovedTradeIntent
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
