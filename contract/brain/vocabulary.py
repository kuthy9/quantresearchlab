"""Brain vocabulary: conflict roles and the neutral-state schema version."""
from __future__ import annotations

from enum import Enum


class GlobalConflictRole(str, Enum):
    """Deterministic market-level meaning of one cross-scale opposition."""

    LOCAL_COUNTERTREND_DELIVERY = "local_countertrend_delivery"
    AUTHORITY_TRANSITION_CANDIDATE = "authority_transition_candidate"
    AUTHORITY_INVALIDATION = "authority_invalidation"


class HypothesisConflictRelation(str, Enum):
    """Direction-aware meaning of global evidence for one hypothesis."""

    CHALLENGES_INCUMBENT = "challenges_incumbent"
    SUPPORTS_CHALLENGER = "supports_challenger"
    LOCAL_COUNTERTREND = "local_countertrend"
    UNRELATED = "unrelated"


NEUTRAL_MARKET_STATE_SCHEMA_VERSION = 2


__all__ = [
    "GlobalConflictRole",
    "HypothesisConflictRelation",
    "NEUTRAL_MARKET_STATE_SCHEMA_VERSION",
]
