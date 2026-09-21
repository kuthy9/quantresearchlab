from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from brain.core.position_ledger import PositionRecord
from contract.brain.state import TradeDirection
from contract.decision import OpportunityGeometry
from contract.execution import AccountSnapshot, OrderRole, OrderState, OrderStatus, Position
from contract.risk import ObjectRef, TradePlan, VetoCode
from risk.core.gate import RiskConfig, RiskGate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = RiskConfig.from_json(ROOT / "risk" / "configs" / "risk.json")
T = pd.Timestamp("2022-01-03T14:12:00Z")


def plan(direction=TradeDirection.SHORT, entry=16387.5, stop=16411.5, target=16330.0, grade="BASE") -> TradePlan:
    """A SHORT with a 24-point stop and a 57.5-point target: 2.4 R, 480 USD a contract."""
    if direction is TradeDirection.LONG and entry == 16387.5 and stop == 16411.5 and target == 16330.0:
        entry, stop, target = 16350.0, 16330.0, 16400.0
    rr = abs(target - entry) / abs(entry - stop)
    return TradePlan(
        "EP_20220103_001", 12, T, direction,
        ObjectRef("FVG_5m_10", "fvg:a", "fvg", "5m"), ObjectRef("BSL_5m_3", "swing:b", "bsl", "5m"), ObjectRef("SSL_4H_2", "swing:c", "ssl", "4H"),
        OpportunityGeometry(entry, stop, target, rr, ("e", "s", "t")), close=16390.0, thesis_id="T1", governing_timeframe="15m", grade=grade,
    )


def account(equity=100_000.0, asof=T, positions=(), open_orders=(), available=None) -> AccountSnapshot:
    available = equity * 0.9 if available is None else available
    return AccountSnapshot("DU1", asof, equity, available, equity * 4, tuple(positions), tuple(open_orders), source="test")


def working_entry() -> OrderState:
    return OrderState("o1", "EP:sig", OrderRole.ENTRY, "SELL", 1, 16387.5, None, 0, None, OrderStatus.WORKING, T, T)


def positions(count: int, direction=TradeDirection.SHORT) -> tuple[PositionRecord, ...]:
    return tuple(PositionRecord(f"p{i}", direction, T, "FVG_5m_1", thesis_id=f"T{i}") for i in range(count))


def test_config_loads_the_v2_defaults_and_hashes() -> None:
    assert CONFIG.risk_fraction == {"BASE": 0.015, "A_PLUS": 0.02} and CONFIG.max_open_positions == 3
    assert CONFIG.min_reward_risk == 2.0 and CONFIG.preferred_reward_risk == 3.0
    assert CONFIG.daily_loss_fraction == 0.025 and CONFIG.max_drawdown_fraction == 0.065 and CONFIG.max_leverage == 8.0
    assert CONFIG.max_quantity == 5 and CONFIG.order_ttl_bars == 15 and CONFIG.account_max_age_s == 120 and CONFIG.margin_per_contract == 20_000.0
    assert CONFIG.thesis.max_expressions == 2 and CONFIG.thesis.stop_cooldown_bars == 30
    assert CONFIG.contract.symbol == "NQ" and CONFIG.contract.point_value == 20.0 and CONFIG.contract.tick_size == 0.25
    assert len(CONFIG.sha256) == 64


def test_v1_config_is_refused(tmp_path: Path) -> None:
    import json

    payload = json.loads((ROOT / "risk" / "configs" / "risk.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    old = tmp_path / "risk_v1.json"
    old.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        RiskConfig.from_json(old)


def test_base_grade_risks_one_and_a_half_percent() -> None:
    gate = RiskGate(CONFIG)
    # 24 points × 20 USD = 480 USD a contract; 1.5 % of 100 000 = 1 500 → 3, and the leverage cap (800 000 / 327 750) → 2
    verdict = gate.assess(plan(), account(100_000.0), asof=T)
    assert verdict.passed and verdict.quantity == 2 and verdict.risk_amount == 960.0
    assert verdict.grade_applied == "BASE" and verdict.risk_fraction == 0.015
    assert (verdict.limit_price, verdict.stop_price, verdict.target_price) == (16387.5, 16411.5, 16330.0)
    assert verdict.reward_risk == pytest.approx(57.5 / 24) and verdict.equity == 100_000.0
    # a 50-point stop (1 000 USD a contract): 1 500 → 1 contract
    assert gate.assess(plan(stop=16437.5, target=16287.5), account(100_000.0), asof=T).quantity == 1
    # the cap: 10 M of equity and a 50-point stop would buy 150; max_quantity 5
    assert gate.assess(plan(stop=16437.5, target=16287.5), account(10_000_000.0), asof=T).quantity == CONFIG.max_quantity


def test_a_plus_sizes_two_percent_only_at_the_preferred_ratio() -> None:
    gate = RiskGate(CONFIG)
    demoted = gate.assess(plan(stop=16437.5, target=16287.5, grade="A_PLUS"), account(), asof=T)  # 2 R < preferred 3
    assert demoted.passed and demoted.grade_applied == "BASE" and demoted.risk_fraction == 0.015 and demoted.quantity == 1
    a_plus = gate.assess(plan(stop=16437.5, target=16237.5, grade="A_PLUS"), account(), asof=T)  # 3 R
    assert a_plus.passed and a_plus.grade_applied == "A_PLUS" and a_plus.risk_fraction == 0.02 and a_plus.quantity == 2  # 2 000 / 1 000


def test_reward_risk_below_two_is_vetoed() -> None:
    verdict = RiskGate(CONFIG).assess(plan(target=16340.0), account(), asof=T)  # 47.5 / 24 = 1.98
    assert verdict.vetoes == (VetoCode.REWARD_RISK,) and "1.98 < 2.0" in verdict.reasons[0]


def test_leverage_caps_the_contracts_by_notional() -> None:
    tight = plan(stop=16389.5, target=16380.0)  # a 2-point stop: the budget alone buys 37, max_quantity 5, margin 4
    assert RiskGate(CONFIG).assess(tight, account(100_000.0), asof=T).quantity == 2  # 800 000 / (16 387.5 × 20)
    verdict = RiskGate(CONFIG).assess(tight, account(30_000.0), asof=T)  # 240 000 / 327 750 < 1
    assert verdict.vetoes == (VetoCode.LEVERAGE,) and "leverage" in verdict.reasons[0]


def test_position_size_reason_says_the_contract_is_too_large_for_this_stop() -> None:
    verdict = RiskGate(CONFIG).assess(plan(stop=16487.5, target=16187.5), account(), asof=T)  # 100 points = 2 000 > 1 500
    assert verdict.vetoes == (VetoCode.POSITION_SIZE,) and "too large for this stop" in verdict.reasons[0]
    assert "nearer" not in verdict.reasons[0]


def test_prices_are_rounded_to_the_tick() -> None:
    verdict = RiskGate(CONFIG).assess(plan(entry=16387.6, stop=16411.4, target=16330.1), account(), asof=T)
    assert verdict.passed and (verdict.limit_price, verdict.stop_price, verdict.target_price) == (16387.5, 16411.5, 16330.0)


@pytest.mark.parametrize("case,veto", [
    ("no_plan", VetoCode.NO_PLAN),
    ("stale", VetoCode.STALE_DATA),
    ("foreign_position", VetoCode.EXPOSURE),
    ("working", VetoCode.WORKING_ORDER),
    ("bad_stop", VetoCode.INVALID_STOP),
    ("bad_target", VetoCode.INVALID_TARGET),
    ("rr", VetoCode.REWARD_RISK),
    ("size", VetoCode.POSITION_SIZE),
    ("margin", VetoCode.POSITION_SIZE),
])
def test_each_veto(case, veto) -> None:
    gate = RiskGate(CONFIG)
    the_plan, the_account = plan(), account()
    if case == "no_plan":
        the_plan = None
    elif case == "stale":
        the_account = account(asof=T - pd.Timedelta(seconds=CONFIG.account_max_age_s + 1))
    elif case == "foreign_position":
        the_account = account(positions=(Position("NQ", -1, 16380.0),))  # a position in the contract that is not ours
    elif case == "working":
        the_account = account(open_orders=(working_entry(),))
    elif case == "bad_stop":
        the_plan = plan(stop=16380.0)  # a SHORT with the stop below the entry
    elif case == "bad_target":
        the_plan = plan(target=16400.0, stop=16411.5)  # a SHORT with the target above the entry
    elif case == "rr":
        the_plan = plan(target=16370.0)  # 17.5 / 24 < 2
    elif case == "size":
        the_account = account(equity=5_000.0)  # 75 USD of budget buys no contract at 480
    elif case == "margin":
        the_account = account(available=15_000.0)
    verdict = gate.assess(the_plan, the_account, asof=T)
    assert verdict.passed is False and veto in verdict.vetoes and verdict.reasons


def test_exposure_counts_the_machines_positions_and_refuses_the_opposite_direction() -> None:
    gate = RiskGate(CONFIG)
    assert gate.assess(plan(), account(), asof=T, positions=positions(3)).vetoes == (VetoCode.EXPOSURE,)
    assert gate.assess(plan(), account(), asof=T, positions=positions(2)).passed
    opposite = gate.assess(plan(TradeDirection.LONG), account(), asof=T, positions=positions(1))
    assert opposite.vetoes == (VetoCode.EXPOSURE,) and "opposite" in opposite.reasons[0]
    # the account's net position is only counted as foreign when none of the positions is ours
    ours = account(positions=(Position("NQ", -1, 16380.0),))
    assert gate.assess(plan(), ours, asof=T, positions=positions(1)).passed


def test_a_long_plan_mirrors() -> None:
    verdict = RiskGate(CONFIG).assess(plan(TradeDirection.LONG), account(), asof=T)  # 20-point stop, 50-point target
    assert verdict.passed and verdict.quantity == 2 and verdict.reward_risk == 2.5


def test_config_override_changes_the_answer() -> None:
    looser = dataclasses.replace(CONFIG, min_reward_risk=1.0)
    assert RiskGate(looser).assess(plan(target=16370.0), account(), asof=T).passed is False  # rr 0.73
    assert RiskGate(looser).assess(plan(target=16360.0), account(), asof=T).passed is True  # rr 1.15


def test_quantity_is_capped_by_the_margin_the_available_funds_can_hold() -> None:
    gate = RiskGate(CONFIG)
    tight = plan(entry=16387.5, stop=16389.5, target=16380.0)
    # 2 contracts by leverage at 100 000; with 30 000 of available funds only one contract's margin is held
    verdict = gate.assess(tight, account(available=30_000.0), asof=T)
    assert verdict.passed and verdict.quantity == 1
    verdict = gate.assess(tight, account(available=15_000.0), asof=T)
    assert not verdict.passed and verdict.vetoes == (VetoCode.POSITION_SIZE,) and "margin" in verdict.reasons[0]


def test_daily_stop_latches_for_the_session_date_and_resets_at_the_next() -> None:
    gate = RiskGate(CONFIG)
    monday_open = pd.Timestamp("2022-01-02T23:00:00Z")  # 18:00 New York on Sunday opens Monday's session
    assert gate.session_date(monday_open).isoformat() == "2022-01-03"
    gate.observe(account(100_000.0, asof=monday_open), monday_open)
    later = monday_open + pd.Timedelta(hours=12)
    gate.observe(account(97_400.0, asof=later), later)
    assert gate.daily_stopped(later)
    assert gate.assess(plan(), account(97_400.0, asof=later), asof=later).vetoes == (VetoCode.DAILY_STOP,)
    recovered = later + pd.Timedelta(hours=1)
    gate.observe(account(99_000.0, asof=recovered), recovered)
    assert gate.daily_stopped(recovered), "the daily stop holds for the rest of the session even if equity recovers"
    tuesday = pd.Timestamp("2022-01-03T23:30:00Z")
    gate.observe(account(97_400.0, asof=tuesday), tuesday)
    assert not gate.daily_stopped(tuesday) and gate.assess(plan(), account(97_400.0, asof=tuesday), asof=tuesday).passed


def test_drawdown_from_the_peak_halts_and_stays_halted() -> None:
    gate = RiskGate(CONFIG)
    for i, equity in enumerate((100_000.0, 104_000.0, 98_000.0, 97_240.0, 120_000.0)):
        at = T + pd.Timedelta(minutes=i)
        gate.observe(account(equity, asof=at), at)
        if equity == 98_000.0:
            assert not gate.halted  # 5.8 % below the 104 000 peak
    assert gate.halted and gate.halt_record == {"at": "2022-01-03T14:15:00Z", "equity": 97_240.0, "peak": 104_000.0, "drawdown": 0.065}
    at = T + pd.Timedelta(minutes=5)
    assert gate.assess(plan(), account(120_000.0, asof=at), asof=at).vetoes == (VetoCode.HALTED,)


def test_leverage_counts_the_contracts_already_open() -> None:
    gate = RiskGate(CONFIG)
    tight = plan(stop=16389.5, target=16380.0)
    # two contracts already short in the account (ours): the 8× cap (two contracts at 100 000) is used up
    held = account(positions=(Position("NQ", -2, 16380.0),))
    verdict = gate.assess(tight, held, asof=T, positions=positions(1))
    assert verdict.vetoes == (VetoCode.LEVERAGE,) and "already open" in verdict.reasons[0]
    one_held = account(positions=(Position("NQ", -1, 16380.0),))
    assert gate.assess(tight, one_held, asof=T, positions=positions(1)).quantity == 1


def test_the_config_is_schema_3_and_the_ttl_is_fifteen_bars_of_the_entry_scale() -> None:
    from risk.core.gate import RISK_SCHEMA_VERSION

    assert RISK_SCHEMA_VERSION == 3
    assert RiskConfig.from_json(Path(__file__).resolve().parents[2] / "risk" / "configs" / "risk.json").order_ttl_bars == 15
