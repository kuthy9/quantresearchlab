"""The trade plan the Brain's opportunity becomes, and the Risk gate's verdict.

``TradePlan`` carries the three objects the LLM named as aliases *and* as
the Eye entity ids the episode's registry maps them to — the ``data_id`` a
downstream reader resolves back to the Eye — together with the geometry code
derived from them on the same bar.  Its ``signature`` is the identity of an
intent across bars and revisions: the direction and the three entity ids,
never the aliases (which are episode-local) and never the prices (which move
with the objects).  ``RiskVerdict`` is what the gate returns: sized and
priced when it passes, named vetoes when it does not."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import pandas as pd

from contract.brain.state import InvalidationMode, ThesisGrade, TradeDirection, isoformat_utc
from contract.decision import OpportunityGeometry
from contract.market.primitives import aware_timestamp
from contract.risk.assessment import VetoCode


@dataclass(frozen=True)
class ObjectRef:
    alias: str
    entity_id: str
    kind: str
    timeframe: str

    def __post_init__(self) -> None:
        for name in ("alias", "entity_id", "kind", "timeframe"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"object ref {name} must be non-empty text")

    def to_dict(self) -> dict[str, str]:
        return {"alias": self.alias, "entity_id": self.entity_id, "kind": self.kind, "timeframe": self.timeframe}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectRef":
        return cls(str(payload["alias"]), str(payload["entity_id"]), str(payload["kind"]), str(payload["timeframe"]))


@dataclass(frozen=True)
class TradePlan:
    episode_id: str
    revision: int
    known_at: pd.Timestamp
    direction: TradeDirection
    entry: ObjectRef
    invalidation: ObjectRef
    target: ObjectRef
    geometry: OpportunityGeometry
    close: float
    # The thesis the plan expresses (2026-09-17): its id, the scale it rests
    # on, the Brain's grade and how the invalidation object falsifies it.
    # Empty / default for plans journaled before they existed.
    thesis_id: str = ""
    governing_timeframe: str = ""
    grade: ThesisGrade = ThesisGrade.BASE
    invalidation_mode: InvalidationMode = InvalidationMode.TOUCH
    # The invalidation object's far edge (the level a CLOSE_BEYOND exit
    # watches a scale close against); None for plans before 2026-09-17.
    invalidation_level: float | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or type(self.revision) is not int or self.revision < 0:
            raise ValueError("trade plan needs an episode_id and a non-negative revision")
        object.__setattr__(self, "known_at", aware_timestamp(self.known_at, name="plan.known_at"))
        object.__setattr__(self, "direction", TradeDirection(self.direction))
        object.__setattr__(self, "grade", ThesisGrade(self.grade))
        object.__setattr__(self, "invalidation_mode", InvalidationMode(self.invalidation_mode))
        for name in ("thesis_id", "governing_timeframe"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"trade plan {name} must be text")
        if self.invalidation_level is not None:
            level = float(self.invalidation_level)
            if not math.isfinite(level) or level <= 0.0:
                raise ValueError("trade plan invalidation_level must be a positive finite price")
            object.__setattr__(self, "invalidation_level", level)
        close = float(self.close)
        if not math.isfinite(close) or close <= 0.0:
            raise ValueError("trade plan close must be a positive finite price")
        object.__setattr__(self, "close", close)

    @property
    def signature(self) -> str:
        text = "|".join((self.direction.value, self.entry.entity_id, self.invalidation.entity_id, self.target.entity_id))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "revision": self.revision,
            "known_at": isoformat_utc(self.known_at),
            "direction": self.direction.value,
            "entry": self.entry.to_dict(),
            "invalidation": self.invalidation.to_dict(),
            "target": self.target.to_dict(),
            "geometry": self.geometry.to_dict(),
            "close": self.close,
            "signature": self.signature,
            "thesis_id": self.thesis_id,
            "governing_timeframe": self.governing_timeframe,
            "grade": self.grade.value,
            "invalidation_mode": self.invalidation_mode.value,
            "invalidation_level": self.invalidation_level,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradePlan":
        geometry = payload["geometry"]
        return cls(
            episode_id=str(payload["episode_id"]), revision=int(payload["revision"]), known_at=pd.Timestamp(payload["known_at"]),
            direction=payload["direction"], entry=ObjectRef.from_dict(payload["entry"]),
            invalidation=ObjectRef.from_dict(payload["invalidation"]), target=ObjectRef.from_dict(payload["target"]),
            geometry=OpportunityGeometry(
                geometry["entry_price"], geometry["stop_price"], geometry["target_price"], geometry["reward_risk"], tuple(geometry["rule_ids"]),
            ),
            close=float(payload["close"]),
            thesis_id=str(payload.get("thesis_id", "")), governing_timeframe=str(payload.get("governing_timeframe", "")),
            grade=payload.get("grade") or ThesisGrade.BASE, invalidation_mode=payload.get("invalidation_mode") or InvalidationMode.TOUCH,
            invalidation_level=payload.get("invalidation_level"),
        )


@dataclass(frozen=True)
class RiskVerdict:
    passed: bool
    vetoes: tuple[VetoCode, ...]
    reasons: tuple[str, ...]
    quantity: int = 0
    limit_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    risk_amount: float | None = None
    reward_risk: float | None = None
    equity: float | None = None
    # The grade the quantity was sized with and its equity fraction (2026-09-17).
    grade_applied: str | None = None
    risk_fraction: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vetoes", tuple(VetoCode(item) for item in self.vetoes))
        if self.risk_fraction is not None:
            object.__setattr__(self, "risk_fraction", float(self.risk_fraction))
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))
        if self.passed:
            if self.vetoes:
                raise ValueError("a passed verdict carries no vetoes")
            if type(self.quantity) is not int or self.quantity < 1:
                raise ValueError("a passed verdict needs a positive quantity")
            for name in ("limit_price", "stop_price", "target_price", "risk_amount", "reward_risk", "equity"):
                value = getattr(self, name)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"a passed verdict needs a finite {name}")
                object.__setattr__(self, name, float(value))
        elif not self.vetoes:
            raise ValueError("a refused verdict names its vetoes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "vetoes": [item.value for item in self.vetoes],
            "reasons": list(self.reasons),
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "risk_amount": self.risk_amount,
            "reward_risk": self.reward_risk,
            "equity": self.equity,
            "grade_applied": self.grade_applied,
            "risk_fraction": self.risk_fraction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RiskVerdict":
        return cls(
            passed=bool(payload["passed"]), vetoes=tuple(payload.get("vetoes", ())), reasons=tuple(payload.get("reasons", ())),
            quantity=int(payload.get("quantity", 0)), limit_price=payload.get("limit_price"), stop_price=payload.get("stop_price"),
            target_price=payload.get("target_price"), risk_amount=payload.get("risk_amount"), reward_risk=payload.get("reward_risk"),
            equity=payload.get("equity"), grade_applied=payload.get("grade_applied"), risk_fraction=payload.get("risk_fraction"),
        )


__all__ = ["ObjectRef", "RiskVerdict", "TradePlan"]
