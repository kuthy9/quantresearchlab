"""Deterministic, immutable, never-submit Trade Intent construction.

This is a shadow boundary object, not an order API.  It freezes an already
eligible :class:`~brain.core.signal_policy.SignalAssessment` together with the
existing typed entry episode, trade plan, and account risk quantities.  No
Decision, RiskManager, broker, or execution state is called from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Sequence

import pandas as pd

from shares.core.model import (
    AccountState,
    Direction,
    EntryEpisodeState,
    HypothesisBelief,
    LiquidityLevel,
    PlaybookPhase,
    StructuralLevel,
)
from brain.core.signal_policy import (
    CancelCondition,
    SHADOW_AUTHORITY,
    SignalAssessment,
    SetupFamily,
    trade_plan_identity,
)


TRADE_INTENT_SCHEMA_VERSION = 1


class EntryMethod(str, Enum):
    FVG_50_LIMIT = "fvg_50_limit"
    OB_50_LIMIT = "ob_50_limit"
    RECLAIM_ENTRY = "reclaim_entry"
    MARKET_ENTRY = "market_entry"


class TimeInForce(str, Enum):
    GOOD_TIL_TIME = "good_til_time"


class TradeIntentError(ValueError):
    """Raised when an immutable intent cannot be formed safely."""


def _aware_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise TradeIntentError(f"{name} must be timezone aware")
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _exact_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    raw = tuple(values)
    if (
        not raw
        or len(raw) != len(set(raw))
        or any(not isinstance(value, str) or not value for value in raw)
    ):
        raise TradeIntentError(f"{name} must contain unique non-empty identities")
    return tuple(sorted(raw))


@dataclass(frozen=True)
class TradeIntent:
    """One frozen shadow instruction that cannot submit an order."""

    created_at: pd.Timestamp
    expires_at: pd.Timestamp
    signal_id: str
    candidate_id: str
    episode_id: str
    setup_id: str
    setup_family: SetupFamily
    competition_set_id: str
    path_hypothesis_id: str
    dol_ranking_id: str
    dol_candidate_id: str
    source_event_ids: tuple[str, ...]
    source_identity_ids: tuple[str, ...]
    policy_protocol_id: str
    policy_protocol_version: str
    policy_protocol_fingerprint: str
    path_likelihood_artifact_id: str
    path_model_id: str
    path_model_version: str
    path_calibration_id: str
    dol_calibration_artifact_id: str
    dol_model_id: str
    dol_model_version: str
    dol_calibration_id: str
    outcome_model_artifact_id: str
    outcome_model_id: str
    outcome_model_version: str
    outcome_calibration_id: str
    symbol: str
    instrument_id: str
    side: Direction
    account_snapshot_id: str
    risk_budget_id: str
    quantity: int
    point_value: float
    risk_budget_fraction: float
    risk_budget_amount: float
    position_risk_amount: float
    entry_method_preferences: tuple[EntryMethod, ...]
    planned_entry: float
    invalidation: StructuralLevel
    targets: tuple[LiquidityLevel, ...]
    trade_plan_id: str
    max_wait_seconds: float
    time_in_force: TimeInForce
    cancel_conditions: tuple[CancelCondition, ...]
    schema_version: int = TRADE_INTENT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    submission_allowed: bool = False
    intent_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "created_at",
            _aware_timestamp(self.created_at, name="intent created_at"),
        )
        object.__setattr__(
            self,
            "expires_at",
            _aware_timestamp(self.expires_at, name="intent expires_at"),
        )
        object.__setattr__(self, "setup_family", SetupFamily(self.setup_family))
        object.__setattr__(self, "side", Direction(self.side))
        object.__setattr__(
            self,
            "time_in_force",
            TimeInForce(self.time_in_force),
        )
        source_events = _exact_ids(
            self.source_event_ids,
            name="intent source_event_ids",
        )
        source_identities = _exact_ids(
            self.source_identity_ids,
            name="intent source_identity_ids",
        )
        object.__setattr__(self, "source_event_ids", source_events)
        object.__setattr__(self, "source_identity_ids", source_identities)
        methods = tuple(EntryMethod(value) for value in self.entry_method_preferences)
        object.__setattr__(self, "entry_method_preferences", methods)
        conditions = tuple(self.cancel_conditions)
        object.__setattr__(self, "cancel_conditions", conditions)
        targets = tuple(self.targets)
        object.__setattr__(self, "targets", targets)
        identities = (
            self.signal_id,
            self.candidate_id,
            self.episode_id,
            self.setup_id,
            self.competition_set_id,
            self.path_hypothesis_id,
            self.dol_ranking_id,
            self.dol_candidate_id,
            self.policy_protocol_id,
            self.policy_protocol_version,
            self.policy_protocol_fingerprint,
            self.path_likelihood_artifact_id,
            self.path_model_id,
            self.path_model_version,
            self.path_calibration_id,
            self.dol_calibration_artifact_id,
            self.dol_model_id,
            self.dol_model_version,
            self.dol_calibration_id,
            self.outcome_model_artifact_id,
            self.outcome_model_id,
            self.outcome_model_version,
            self.outcome_calibration_id,
            self.symbol,
            self.instrument_id,
            self.account_snapshot_id,
            self.risk_budget_id,
            self.trade_plan_id,
        )
        numeric = (
            self.point_value,
            self.risk_budget_fraction,
            self.risk_budget_amount,
            self.position_risk_amount,
            self.planned_entry,
            self.max_wait_seconds,
        )
        if (
            self.schema_version != TRADE_INTENT_SCHEMA_VERSION
            or self.authority != SHADOW_AUTHORITY
            or type(self.submission_allowed) is not bool
            or self.submission_allowed
            or any(not isinstance(value, str) or not value for value in identities)
            or len(self.policy_protocol_fingerprint) != 64
            or self.expires_at <= self.created_at
            or type(self.quantity) is not int
            or self.quantity <= 0
            or any(not math.isfinite(float(value)) for value in numeric)
            or self.point_value <= 0.0
            or not 0.0 < self.risk_budget_fraction <= 1.0
            or self.risk_budget_amount <= 0.0
            or self.position_risk_amount <= 0.0
            or self.position_risk_amount > self.risk_budget_amount + 1e-9
            or self.planned_entry <= 0.0
            or self.max_wait_seconds <= 0.0
            or not methods
            or len(methods) != len(set(methods))
            or not isinstance(self.invalidation, StructuralLevel)
            or not targets
            or any(not isinstance(target, LiquidityLevel) for target in targets)
            or not conditions
            or any(not isinstance(item, CancelCondition) for item in conditions)
            or len({item.condition_id for item in conditions}) != len(conditions)
            or self.time_in_force is not TimeInForce.GOOD_TIL_TIME
        ):
            raise TradeIntentError("trade intent contract is invalid")
        payload = {
            name: (
                value.value
                if isinstance(value, Enum)
                else (
                    value.isoformat()
                    if isinstance(value, pd.Timestamp)
                    else (
                        [item.value for item in value]
                        if name == "entry_method_preferences"
                        else (
                            [item.condition_id for item in value]
                            if name == "cancel_conditions"
                            else (
                                [item.level_id for item in value]
                                if name == "targets"
                                else (
                                    {
                                        "price": value.price,
                                        "side": value.side,
                                        "source_level_id": value.source_level_id,
                                        "observed_at": value.observed_at.isoformat(),
                                        "rationale": value.rationale,
                                    }
                                    if name == "invalidation"
                                    else (
                                        list(value)
                                        if isinstance(value, tuple)
                                        else value
                                    )
                                )
                            )
                        )
                    )
                )
            )
            for name, value in self.__dict__.items()
            if name != "intent_id"
        }
        object.__setattr__(
            self,
            "intent_id",
            f"trade-intent:{_canonical_hash(payload)[:32]}",
        )


def build_trade_intent(
    assessment: SignalAssessment,
    *,
    asof: pd.Timestamp,
    setup_candidate: HypothesisBelief,
    entry_episode: EntryEpisodeState,
    account: AccountState,
    account_snapshot_id: str,
    risk_budget_id: str,
    entry_method_preferences: Sequence[EntryMethod],
) -> TradeIntent:
    """Freeze an eligible shadow assessment; never submit or mutate state."""

    if not isinstance(assessment, SignalAssessment):
        raise TypeError("trade intent requires SignalAssessment")
    if not isinstance(setup_candidate, HypothesisBelief):
        raise TypeError("trade intent requires typed HypothesisBelief")
    if not isinstance(entry_episode, EntryEpisodeState):
        raise TypeError("trade intent requires EntryEpisodeState")
    if not isinstance(account, AccountState):
        raise TypeError("trade intent requires AccountState")
    clock = _aware_timestamp(asof, name="trade intent asof")
    if not assessment.eligible:
        raise TradeIntentError("rejected signal cannot form a Trade Intent")
    if clock < assessment.assessed_at or clock >= assessment.expires_at:
        raise TradeIntentError("signal assessment is stale or expired")
    if not account_snapshot_id or not risk_budget_id:
        raise TradeIntentError("account and risk-budget identities are required")

    plan = setup_candidate.plan
    if (
        plan is None
        or plan != entry_episode.plan
        or assessment.trade_plan_id != trade_plan_identity(plan)
        or assessment.candidate_id != setup_candidate.candidate_id
        or assessment.candidate_id != entry_episode.candidate_id
        or assessment.episode_id != setup_candidate.episode_id
        or assessment.episode_id != entry_episode.episode_id
        or assessment.setup_id != plan.setup_id
        or assessment.setup_family is None
        or setup_candidate.phase is not PlaybookPhase.EXECUTABLE
        or entry_episode.phase is not PlaybookPhase.EXECUTABLE
        or setup_candidate.direction is not assessment.direction
        or setup_candidate.playbook is not entry_episode.playbook
        or plan.playbook is not setup_candidate.playbook
        or plan.direction is not setup_candidate.direction
    ):
        raise TradeIntentError("signal and typed setup identities disagree")

    methods = tuple(EntryMethod(value) for value in entry_method_preferences)
    if not methods or len(methods) != len(set(methods)):
        raise TradeIntentError("entry method preferences must be ordered and unique")

    risk_budget_fraction = float(account.requested_risk_fraction)
    risk_budget_amount = float(account.equity) * risk_budget_fraction
    position_risk_amount = (
        float(plan.risk_points) * float(account.point_value) * int(account.quantity)
    )
    if (
        risk_budget_fraction <= 0.0
        or account.open_risk_fraction + risk_budget_fraction > 1.0
        or position_risk_amount > risk_budget_amount + 1e-9
    ):
        raise TradeIntentError("frozen position risk exceeds the risk budget")

    required_assessment_ids = (
        assessment.path_hypothesis_id,
        assessment.path_likelihood_artifact_id,
        assessment.path_model_id,
        assessment.path_model_version,
        assessment.path_calibration_id,
        assessment.dol_calibration_artifact_id,
        assessment.dol_model_id,
        assessment.dol_model_version,
        assessment.dol_calibration_id,
        assessment.outcome_model_artifact_id,
        assessment.outcome_model_id,
        assessment.outcome_model_version,
        assessment.outcome_calibration_id,
    )
    if any(value is None for value in required_assessment_ids):
        raise TradeIntentError("eligible assessment lacks model identities")

    return TradeIntent(
        created_at=clock,
        expires_at=assessment.expires_at,
        signal_id=assessment.signal_id,
        candidate_id=assessment.candidate_id,
        episode_id=assessment.episode_id,
        setup_id=assessment.setup_id,
        setup_family=assessment.setup_family,
        competition_set_id=assessment.competition_set_id,
        path_hypothesis_id=assessment.path_hypothesis_id,
        dol_ranking_id=assessment.dol_ranking_id,
        dol_candidate_id=assessment.dol_candidate_id,
        source_event_ids=assessment.source_event_ids,
        source_identity_ids=assessment.source_identity_ids,
        policy_protocol_id=assessment.policy_protocol_id,
        policy_protocol_version=assessment.policy_protocol_version,
        policy_protocol_fingerprint=assessment.policy_protocol_fingerprint,
        path_likelihood_artifact_id=assessment.path_likelihood_artifact_id,
        path_model_id=assessment.path_model_id,
        path_model_version=assessment.path_model_version,
        path_calibration_id=assessment.path_calibration_id,
        dol_calibration_artifact_id=assessment.dol_calibration_artifact_id,
        dol_model_id=assessment.dol_model_id,
        dol_model_version=assessment.dol_model_version,
        dol_calibration_id=assessment.dol_calibration_id,
        outcome_model_artifact_id=assessment.outcome_model_artifact_id,
        outcome_model_id=assessment.outcome_model_id,
        outcome_model_version=assessment.outcome_model_version,
        outcome_calibration_id=assessment.outcome_calibration_id,
        symbol=assessment.symbol,
        instrument_id=assessment.instrument_id,
        side=assessment.direction,
        account_snapshot_id=account_snapshot_id,
        risk_budget_id=risk_budget_id,
        quantity=account.quantity,
        point_value=account.point_value,
        risk_budget_fraction=risk_budget_fraction,
        risk_budget_amount=risk_budget_amount,
        position_risk_amount=position_risk_amount,
        entry_method_preferences=methods,
        planned_entry=plan.planned_entry,
        invalidation=plan.invalidation,
        targets=plan.targets,
        trade_plan_id=assessment.trade_plan_id,
        max_wait_seconds=(assessment.expires_at - clock).total_seconds(),
        time_in_force=TimeInForce.GOOD_TIL_TIME,
        cancel_conditions=assessment.cancel_conditions,
    )


__all__ = [
    "EntryMethod",
    "TRADE_INTENT_SCHEMA_VERSION",
    "TimeInForce",
    "TradeIntent",
    "TradeIntentError",
    "build_trade_intent",
]
