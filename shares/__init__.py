"""Causal continuous SMC trader, product version 1.

The public surface is the Trading Eye's published market view plus the runtime
entry points that consume it.  It spans all four subsystems, so it lives here in
``shares`` rather than inside any one of them.  ``EventStore`` is the Eye's
internal history authority and is deliberately not exported; import
``eyes.core.event_store`` directly when a research or replay tool needs
read-only access to event lineage.

Every name is resolved lazily.  ``shares.core.model`` is imported by all four
subsystems, so binding the facade eagerly would make that import pull the whole
engine in and would turn the existing module-level cycles into import errors.
"""
from __future__ import annotations

from typing import Any

_EXPORTS = {
    "Action": "shares.core.model",
    "Bar": "shares.core.model",
    "ContinuousSMCEngine": "shares.core.engine",
    "Direction": "shares.core.model",
    "ExecutionFSM": "execution.core.execution_fsm",
    "MarketSnapshot": "eyes.core.market_state",
    "MarketSnapshotAuthority": "eyes.core.market_state",
    "Playbook": "shares.core.model",
    "PlaybookPhase": "shares.core.model",
    "RelationState": "eyes.core.market_state",
    "RiskApprovedTradeIntent": "execution.core.execution_fsm",
    "SMC_SEMANTIC_VERSION": "shares.core.model",
    "SemanticRegistry": "eyes.core.semantics",
    "SessionState": "eyes.core.market_state",
    "TimeframeState": "eyes.core.market_state",
}

__all__ = sorted(_EXPORTS)

__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
