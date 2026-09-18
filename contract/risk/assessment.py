"""Risk contracts: the veto vocabulary and the independent risk verdict.

Risk is a separate hard boundary; a veto code names why an otherwise
approved candidate may not act."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contract.decision.action import Action


class VetoCode(str, Enum):
    NONE = "none"
    DATA_ANOMALY = "data_anomaly"
    STALE_DATA = "stale_data"
    SPREAD = "spread"
    COST = "cost"
    DEADLINE = "deadline"
    FILLABILITY = "fillability"
    ACCOUNT_RISK = "account_risk"
    INVALID_STOP = "invalid_stop"
    INVALID_TARGET = "invalid_target"
    REWARD_RISK = "reward_risk"
    NO_PLAN = "no_plan"
    PROTECTION_NOT_TIGHTER = "protection_not_tighter"
    # Added 2026-09-16 with the Risk gate: exposure and sizing vetoes.
    EXPOSURE = "exposure"
    WORKING_ORDER = "working_order"
    POSITION_SIZE = "position_size"
    # Added 2026-09-17 with Risk v2: the session's loss limit, the run's
    # drawdown halt, the notional cap.
    DAILY_STOP = "daily_stop"
    HALTED = "halted"
    LEVERAGE = "leverage"


@dataclass(frozen=True)
class RiskAssessment:
    requested_action: Action
    final_action: Action
    passed: bool
    vetoes: tuple[VetoCode, ...]
    reasons: tuple[str, ...]
    frozen_thesis: "FrozenThesis | None" = None
    protected_stop: float | None = None


__all__ = [
    "RiskAssessment",
    "VetoCode",
]
