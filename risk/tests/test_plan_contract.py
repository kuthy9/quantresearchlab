from __future__ import annotations

import json

import pandas as pd
import pytest

from contract.brain.state import InvalidationMode, ThesisGrade, TradeDirection
from contract.decision import OpportunityGeometry
from contract.execution import OrderRole
from contract.risk import ObjectRef, RiskVerdict, TradePlan, VetoCode

T = pd.Timestamp("2022-01-03T14:12:00Z")
GEOMETRY = OpportunityGeometry(16387.5, 16411.5, 16350.75, 1.53125, ("entry.zone.near_edge", "stop.pool.far_edge", "target.pool.midpoint"))


def plan(**over) -> TradePlan:
    base = dict(
        episode_id="EP_20220103_001", revision=12, known_at=T, direction=TradeDirection.SHORT,
        entry=ObjectRef("FVG_5m_10", "fvg:aaaa", "fvg", "5m"),
        invalidation=ObjectRef("BSL_5m_3", "swing:bbbb", "bsl", "5m"),
        target=ObjectRef("SSL_4H_2", "swing:cccc", "ssl", "4H"),
        geometry=GEOMETRY, close=16390.0, thesis_id="T1", governing_timeframe="1H",
    )
    base.update(over)
    return TradePlan(**base)


def test_signature_is_the_direction_and_the_three_entity_ids() -> None:
    a = plan()
    assert len(a.signature) == 16
    assert plan(revision=13, known_at=T + pd.Timedelta(minutes=1)).signature == a.signature
    assert plan(target=ObjectRef("SSL_15m_2", "swing:dddd", "ssl", "15m")).signature != a.signature
    assert plan(direction=TradeDirection.LONG).signature != a.signature
    assert plan(entry=ObjectRef("FVG_5m_11", "fvg:aaaa", "fvg", "5m")).signature == a.signature, "the alias is episode-local; the entity id is the identity"


def test_round_trips() -> None:
    a = plan()
    assert TradePlan.from_dict(json.loads(json.dumps(a.to_dict()))) == a
    passed = RiskVerdict(True, (), (), quantity=2, limit_price=16387.5, stop_price=16411.5, target_price=16350.75, risk_amount=960.0, reward_risk=1.53125, equity=250_000.0)
    assert RiskVerdict.from_dict(json.loads(json.dumps(passed.to_dict()))) == passed
    vetoed = RiskVerdict(False, (VetoCode.REWARD_RISK, VetoCode.POSITION_SIZE), ("rr 1.1 < 1.5", "0 contracts"))
    assert RiskVerdict.from_dict(vetoed.to_dict()) == vetoed


def test_verdict_invariants() -> None:
    with pytest.raises(ValueError, match="quantity"):
        RiskVerdict(True, (), ())
    with pytest.raises(ValueError, match="vetoes"):
        RiskVerdict(False, (), ("no reason",))
    assert VetoCode.EXPOSURE.value == "exposure" and VetoCode.WORKING_ORDER.value == "working_order" and VetoCode.POSITION_SIZE.value == "position_size"


def test_plan_carries_the_thesis_fields_and_the_signature_ignores_them() -> None:
    a = plan(thesis_id="T1", grade=ThesisGrade.BASE)
    b = plan(thesis_id="T2", grade="A_PLUS", invalidation_mode="CLOSE_BEYOND")
    assert a.signature == b.signature
    assert b.grade is ThesisGrade.A_PLUS and b.invalidation_mode is InvalidationMode.CLOSE_BEYOND
    assert TradePlan.from_dict(json.loads(json.dumps(b.to_dict()))) == b and b.to_dict()["invalidation_mode"] == "CLOSE_BEYOND"
    old = {k: v for k, v in a.to_dict().items() if k not in ("thesis_id", "governing_timeframe", "grade", "invalidation_mode")}
    assert TradePlan.from_dict(old).thesis_id == "" and TradePlan.from_dict(old).grade is ThesisGrade.BASE


def test_verdict_carries_the_grade_it_sized_with() -> None:
    passed = RiskVerdict(True, (), (), quantity=1, limit_price=1.0, stop_price=0.5, target_price=2.0, risk_amount=10.0, reward_risk=2.0, equity=1.0, grade_applied="A_PLUS", risk_fraction=0.02)
    assert RiskVerdict.from_dict(json.loads(json.dumps(passed.to_dict()))) == passed and passed.to_dict()["grade_applied"] == "A_PLUS"


def test_vetoes_and_roles_gained_the_v2_names() -> None:
    assert {VetoCode.DAILY_STOP.value, VetoCode.HALTED.value, VetoCode.LEVERAGE.value} == {"daily_stop", "halted", "leverage"}
    assert OrderRole.FLATTEN.value == "flatten"
