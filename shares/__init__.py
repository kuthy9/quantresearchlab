"""Causal continuous SMC trader, product version 1.

The public surface is the Trading Eye's published market view plus the runtime
entry points that consume it.  It spans all four subsystems, so it lives here in
``shares`` rather than inside any one of them.  ``EventStore`` is the Eye's
internal history authority and is deliberately not exported; import
``eyes.core.event_store`` directly when a research or replay tool needs
read-only access to event lineage.

``ContinuousSMCEngine`` is deliberately absent while the typed Brain is being
rebuilt: ``shares.core.engine`` still imports the retired playbook, DOL and
Signal Policy modules and cannot be imported.  Import it directly once that
orchestration is rebound to a new belief producer.

Every name is resolved lazily so this facade never pulls the engine in.  The
type contracts themselves now live in the ``contract`` package, one module per
layer; import from there rather than through this facade.
"""
from __future__ import annotations

from typing import Any

_EXPORTS = {
    "Action": "contract.decision",
    "Bar": "contract.market",
    "Direction": "contract.market",
    "MarketSnapshot": "eyes.core.market_state",
    "MarketSnapshotAuthority": "eyes.core.market_state",
    "Playbook": "contract.market",
    "PlaybookPhase": "contract.market",
    "RelationState": "eyes.core.market_state",
    "SMC_SEMANTIC_VERSION": "contract.market",
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
