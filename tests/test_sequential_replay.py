from __future__ import annotations

from dataclasses import replace

import pandas as pd

from smc_trader.model import (
    Action,
    Bar,
    EngineSnapshot,
    RiskAssessment,
)
from smc_trader.risk import RiskLimits, StructuralRiskEngine
from smc_trader.simulation import SequentialPortfolio
from smc_trader.observation import ExecutionRealityInput

from .helpers import engine_snapshot, flat_account


def _approved_entry_snapshot() -> EngineSnapshot:
    base = engine_snapshot()
    risk = StructuralRiskEngine(RiskLimits(maximum_cost_R=0.50)).review(
        base.decision,
        base.observation,
        flat_account(),
    )
    assert risk.passed and risk.frozen_thesis is not None
    return replace(base, risk=risk)


def test_sequential_entry_waits_for_next_bar_and_preserves_original_stop() -> None:
    snapshot = _approved_entry_snapshot()
    portfolio = SequentialPortfolio()
    portfolio.after_decision(snapshot)
    entry_bar = Bar(
        snapshot.observation.asof,
        101.0,
        103.5,
        99.5,
        103.0,
        100,
        "NQH5",
        1,
    )
    closed = portfolio.before_bar(
        entry_bar,
        execution=ExecutionRealityInput(expected_slippage_points=0.25),
    )
    assert not closed
    position = portfolio.account(entry_bar.end).position
    assert position is not None
    assert position.opened_at == entry_bar.end
    assert position.entry_price == snapshot.decision.plan.planned_entry
    assert position.original_invalidation.price == 98.0
    assert position.current_stop == 98.0
    assert position.mfe_R == 0.0
    assert position.mae_R == (
        entry_bar.low - position.entry_price
    ) / snapshot.decision.plan.risk_points

    protect = replace(
        snapshot,
        risk=RiskAssessment(
            Action.PROTECT,
            Action.PROTECT,
            True,
            (),
            ("confirmed structure",),
            protected_stop=99.0,
        ),
    )
    portfolio.after_decision(protect)
    position = portfolio.account(entry_bar.end).position
    assert position.original_invalidation.price == 98.0
    assert position.current_stop == 99.0


def test_pending_limit_cannot_fill_at_frozen_deadline() -> None:
    snapshot = _approved_entry_snapshot()
    deadline = snapshot.observation.asof + pd.Timedelta(minutes=1)
    assert snapshot.risk.frozen_thesis is not None
    assert snapshot.decision.plan is not None
    snapshot = replace(
        snapshot,
        decision=replace(
            snapshot.decision,
            plan=replace(snapshot.decision.plan, deadline=deadline),
        ),
        risk=replace(
            snapshot.risk,
            frozen_thesis=replace(
                snapshot.risk.frozen_thesis,
                deadline=deadline,
            ),
        ),
    )
    portfolio = SequentialPortfolio()
    portfolio.after_decision(snapshot)
    deadline_bar = Bar(
        snapshot.observation.asof,
        101.0,
        103.5,
        99.5,
        103.0,
        100,
        "NQH5",
        1,
    )

    assert not portfolio.before_bar(deadline_bar)
    assert portfolio.account(deadline_bar.end).position is None
    assert portfolio.records == ()


def test_sequential_target_closes_only_on_a_later_bar() -> None:
    snapshot = _approved_entry_snapshot()
    portfolio = SequentialPortfolio()
    portfolio.after_decision(snapshot)
    entry_bar = Bar(
        snapshot.observation.asof,
        101.0,
        103.5,
        99.5,
        103.0,
        100,
        "NQH5",
        1,
    )
    execution = ExecutionRealityInput(expected_slippage_points=0.25)
    assert not portfolio.before_bar(entry_bar, execution)
    target_bar = Bar(
        entry_bar.end,
        102.5,
        103.5,
        100.5,
        103.25,
        100,
        "NQH5",
        1,
    )
    closed = portfolio.before_bar(target_bar, execution)
    assert len(closed) == 1
    assert closed[0].exit_reason == "target"
    assert closed[0].gross_R == 1.5
    assert closed[0].original_invalidation == 98.0
    lifecycle = portfolio.lifecycle_position
    assert lifecycle is not None
    assert lifecycle.status == "completed"
    assert portfolio.account(target_bar.end).position is None


def test_stop_close_exposes_invalidated_lifecycle_for_one_decision() -> None:
    snapshot = _approved_entry_snapshot()
    portfolio = SequentialPortfolio()
    portfolio.after_decision(snapshot)
    execution = ExecutionRealityInput(expected_slippage_points=0.25)
    entry_bar = Bar(
        snapshot.observation.asof,
        101.0,
        101.5,
        99.5,
        100.5,
        100,
        "NQH5",
        1,
    )
    assert not portfolio.before_bar(entry_bar, execution)
    stop_bar = Bar(
        entry_bar.end,
        99.0,
        99.5,
        97.5,
        98.0,
        100,
        "NQH5",
        1,
    )
    closed = portfolio.before_bar(stop_bar, execution)
    assert len(closed) == 1
    lifecycle = portfolio.lifecycle_position
    assert lifecycle is not None
    assert lifecycle.status == "invalidated"
    portfolio.clear_lifecycle_position()
    assert portfolio.lifecycle_position is None
