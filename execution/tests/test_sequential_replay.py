from __future__ import annotations

from dataclasses import replace

import pandas as pd

from shares.core.model import (
    Action,
    Bar,
    EngineSnapshot,
    RiskAssessment,
)
from brain.core.risk import StructuralRiskEngine
from execution.core.simulation import SequentialPortfolio, SequentialReplay
from execution.core.execution import ExecutionRealityInput

from brain.tests.test_v4_typed_vertical import (
    _dfp_fixture,
    _mapped_brain,
    _ready_decision_layer,
)
from shares.tests.helpers import graph_free_action_belief


def _approved_entry_snapshot() -> EngineSnapshot:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _mapped_brain()
    brain.update(forming)
    belief = brain.update(triggered)
    decision = _ready_decision_layer(brain).decide(
        triggered,
        graph_free_action_belief(belief),
    )
    snapshot = EngineSnapshot(
        observation=triggered,
        belief=belief,
        decision=decision,
        risk=StructuralRiskEngine().review(decision, triggered),
    )
    assert snapshot.risk.passed and snapshot.risk.frozen_thesis is not None
    return snapshot


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
    assert position.original_invalidation.price == (
        snapshot.decision.plan.invalidation.price
    )
    assert position.current_stop == snapshot.decision.plan.invalidation.price
    assert position.mfe_R == 0.0
    assert (
        position.mae_R
        == (entry_bar.low - position.entry_price) / snapshot.decision.plan.risk_points
    )

    protect = replace(
        snapshot,
        risk=RiskAssessment(
            Action.PROTECT,
            Action.PROTECT,
            True,
            (),
            ("confirmed structure",),
            protected_stop=99.5,
        ),
    )
    portfolio.after_decision(protect)
    position = portfolio.account(entry_bar.end).position
    assert position.original_invalidation.price == (
        snapshot.decision.plan.invalidation.price
    )
    assert position.current_stop == 99.5


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
    assert closed[0].gross_R == snapshot.decision.plan.primary_target_R
    assert closed[0].original_invalidation == (
        snapshot.decision.plan.invalidation.price
    )
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


def test_contract_boundary_exits_before_brain_receives_lifecycle_position() -> None:
    """A Brain epoch reset follows, rather than causes, boundary exit."""

    snapshot = _approved_entry_snapshot()

    class BoundaryPortfolio(SequentialPortfolio):
        def after_decision(self, _snapshot: EngineSnapshot) -> None:
            # The assertion concerns before-bar ordering only.  Do not let the
            # fixture snapshot create another pending entry after the exit.
            return None

    class CapturingEngine:
        def __init__(self) -> None:
            self.account = None
            self.belief_position = None

        def on_bar(self, _bar, *, execution, account, belief_position):
            self.account = account
            self.belief_position = belief_position
            return snapshot

    portfolio = BoundaryPortfolio()
    SequentialPortfolio.after_decision(portfolio, snapshot)
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
    assert not portfolio.before_bar(entry_bar)
    assert portfolio.account(entry_bar.end).position is not None

    engine = CapturingEngine()
    replay = SequentialReplay(engine=engine, portfolio=portfolio)
    next_contract = Bar(
        entry_bar.end,
        100.5,
        101.0,
        100.0,
        100.75,
        100,
        "NQM5",
        2,
    )
    step = replay.on_bar(
        next_contract,
        execution=ExecutionRealityInput(),
    )

    assert len(step.closed_trades) == 1
    assert step.closed_trades[0].exit_reason == "contract_change_gap_exit"
    assert engine.account.position is None
    assert engine.belief_position is not None
    assert engine.belief_position.status == "invalidated"
    assert step.position is None
