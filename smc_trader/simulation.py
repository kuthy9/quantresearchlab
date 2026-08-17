"""Causal sequential execution adapter for historical and shadow replay."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import ContinuousSMCEngine
from .model import (
    AccountState,
    Action,
    Bar,
    Direction,
    EngineSnapshot,
    FrozenThesis,
    LiquidityLevel,
    PositionSnapshot,
    StructuralLevel,
    TradePlan,
)
from .observation import ExecutionRealityInput
from .risk import (
    causal_protection_candidate,
    conservative_entry_bar,
    conservative_position_bar,
)


@dataclass(frozen=True)
class TradeRecord:
    thesis_hash: str
    playbook: str
    direction: str
    setup_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    decision_time: pd.Timestamp
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    entry_price: float
    original_invalidation: float
    final_stop: float
    target: float
    exit_price: float
    exit_reason: str
    gross_R: float
    cost_R: float
    net_R: float
    ambiguous_same_bar: bool


@dataclass(frozen=True)
class ReplayStep:
    snapshot: EngineSnapshot
    closed_trades: tuple[TradeRecord, ...]
    position: PositionSnapshot | None
    account_state: AccountState
    belief_position_input: PositionSnapshot | None


@dataclass
class _PendingEntry:
    plan: TradePlan
    thesis: FrozenThesis
    decision_time: pd.Timestamp
    cost_points: float
    symbol: str
    instrument_id: int


@dataclass
class _OpenTrade:
    thesis: FrozenThesis
    decision_time: pd.Timestamp
    opened_at: pd.Timestamp
    entry_price: float
    current_stop: float
    target: LiquidityLevel
    risk_points: float
    cost_points: float
    symbol: str
    instrument_id: int
    mark_R: float = 0.0
    mfe_R: float = 0.0
    mae_R: float = 0.0
    protection_candidate: StructuralLevel | None = None


class SequentialPortfolio:
    """Owns execution state; it cannot alter beliefs or risk decisions."""

    def __init__(
        self,
        *,
        starting_equity: float = 100_000.0,
        point_value: float = 20.0,
        quantity: int = 1,
        requested_risk_fraction: float = 0.005,
    ) -> None:
        if starting_equity <= 0 or point_value <= 0 or quantity <= 0:
            raise ValueError("portfolio capital, point value, and quantity must be positive")
        self.starting_equity = float(starting_equity)
        self.equity = float(starting_equity)
        self.point_value = float(point_value)
        self.quantity = int(quantity)
        self.requested_risk_fraction = float(requested_risk_fraction)
        self._pending_entry: _PendingEntry | None = None
        self._pending_exit_reason: str | None = None
        self._open: _OpenTrade | None = None
        self._lifecycle_position: PositionSnapshot | None = None
        self._records: list[TradeRecord] = []
        self._next_open_slippage_points = 0.0

    @property
    def records(self) -> tuple[TradeRecord, ...]:
        return tuple(self._records)

    @property
    def lifecycle_position(self) -> PositionSnapshot | None:
        """One-bar terminal position context for the playbook state machine."""

        return self._lifecycle_position

    def clear_lifecycle_position(self) -> None:
        self._lifecycle_position = None

    def _position_snapshot(self, asof: pd.Timestamp) -> PositionSnapshot | None:
        trade = self._open
        if trade is None:
            return None
        elapsed = max(0, int((asof - trade.opened_at).total_seconds() // 60))
        return PositionSnapshot(
            thesis_hash=trade.thesis.thesis_hash,
            symbol=trade.symbol,
            instrument_id=trade.instrument_id,
            playbook=trade.thesis.playbook,
            direction=trade.thesis.direction,
            entry_price=trade.entry_price,
            original_invalidation=trade.thesis.original_invalidation,
            current_stop=trade.current_stop,
            primary_target=trade.target,
            opened_at=trade.opened_at,
            deadline=trade.thesis.deadline,
            quantity=self.quantity,
            unrealized_R=trade.mark_R,
            elapsed_minutes=elapsed,
            mfe_R=trade.mfe_R,
            mae_R=trade.mae_R,
            protection_candidate=trade.protection_candidate,
            setup_id=trade.thesis.setup_id,
            entry_location_id=trade.thesis.entry_location_id,
            entry_path_id=trade.thesis.entry_path_id,
        )

    def account(self, asof: pd.Timestamp) -> AccountState:
        position = self._position_snapshot(asof)
        open_risk = 0.0
        if self._open is not None:
            open_risk = (
                abs(self._open.entry_price - self._open.current_stop)
                * self.point_value
                * self.quantity
                / max(self.equity, 1.0)
            )
        return AccountState(
            equity=self.equity,
            open_risk_fraction=float(open_risk),
            requested_risk_fraction=self.requested_risk_fraction,
            quantity=self.quantity,
            point_value=self.point_value,
            position=position,
        )

    def _close(
        self,
        *,
        time: pd.Timestamp,
        price: float,
        reason: str,
        ambiguous: bool = False,
    ) -> TradeRecord:
        trade = self._open
        if trade is None:
            raise RuntimeError("cannot close a missing position")
        gross_R = (
            trade.thesis.direction.sign * (float(price) - trade.entry_price)
            / trade.risk_points
        )
        cost_R = trade.cost_points / trade.risk_points
        net_R = gross_R - cost_R
        self.equity += (
            net_R * trade.risk_points * self.point_value * self.quantity
        )
        status = "completed" if reason == "target" else "invalidated"
        self._lifecycle_position = PositionSnapshot(
            thesis_hash=trade.thesis.thesis_hash,
            symbol=trade.symbol,
            instrument_id=trade.instrument_id,
            playbook=trade.thesis.playbook,
            direction=trade.thesis.direction,
            entry_price=trade.entry_price,
            original_invalidation=trade.thesis.original_invalidation,
            current_stop=trade.current_stop,
            primary_target=trade.target,
            opened_at=trade.opened_at,
            deadline=trade.thesis.deadline,
            quantity=self.quantity,
            unrealized_R=float(gross_R),
            elapsed_minutes=max(
                0, int((time - trade.opened_at).total_seconds() // 60)
            ),
            mfe_R=trade.mfe_R,
            mae_R=trade.mae_R,
            protection_candidate=trade.protection_candidate,
            status=status,
            setup_id=trade.thesis.setup_id,
            entry_location_id=trade.thesis.entry_location_id,
            entry_path_id=trade.thesis.entry_path_id,
        )
        record = TradeRecord(
            thesis_hash=trade.thesis.thesis_hash,
            playbook=trade.thesis.playbook.value,
            direction=trade.thesis.direction.value,
            setup_id=trade.thesis.setup_id,
            entry_location_id=trade.thesis.entry_location_id,
            entry_path_id=trade.thesis.entry_path_id,
            decision_time=trade.decision_time,
            opened_at=trade.opened_at,
            closed_at=time,
            entry_price=trade.entry_price,
            original_invalidation=trade.thesis.original_invalidation.price,
            final_stop=trade.current_stop,
            target=trade.target.price,
            exit_price=float(price),
            exit_reason=reason,
            gross_R=float(gross_R),
            cost_R=float(cost_R),
            net_R=float(net_R),
            ambiguous_same_bar=bool(ambiguous),
        )
        self._records.append(record)
        self._open = None
        self._pending_exit_reason = None
        return record

    def before_bar(
        self,
        bar: Bar,
        execution: ExecutionRealityInput | None = None,
    ) -> tuple[TradeRecord, ...]:
        """Apply prior instructions without reading this bar's closing execution data."""

        self._lifecycle_position = None
        closed: list[TradeRecord] = []
        slippage = max(0.0, self._next_open_slippage_points)
        current_contract = (bar.symbol, int(bar.instrument_id))
        if self._open is not None and current_contract != (
            self._open.symbol,
            self._open.instrument_id,
        ):
            price = (
                bar.open - slippage
                if self._open.thesis.direction is Direction.LONG
                else bar.open + slippage
            )
            closed.append(
                self._close(
                    time=bar.start,
                    price=price,
                    reason="contract_change_gap_exit",
                )
            )
        if self._open is not None and self._pending_exit_reason is not None:
            price = (
                bar.open - slippage
                if self._open.thesis.direction is Direction.LONG
                else bar.open + slippage
            )
            closed.append(
                self._close(
                    time=bar.start,
                    price=price,
                    reason=self._pending_exit_reason,
                )
            )

        if self._open is not None and bar.start >= self._open.thesis.deadline:
            price = (
                bar.open - slippage
                if self._open.thesis.direction is Direction.LONG
                else bar.open + slippage
            )
            closed.append(
                self._close(time=bar.start, price=price, reason="hard_deadline")
            )

        if self._open is not None:
            favorable_price = (
                bar.high
                if self._open.thesis.direction is Direction.LONG
                else bar.low
            )
            adverse_price = (
                bar.low
                if self._open.thesis.direction is Direction.LONG
                else bar.high
            )
            favorable_R = (
                self._open.thesis.direction.sign
                * (favorable_price - self._open.entry_price)
                / self._open.risk_points
            )
            adverse_R = (
                self._open.thesis.direction.sign
                * (adverse_price - self._open.entry_price)
                / self._open.risk_points
            )
            self._open.mfe_R = max(self._open.mfe_R, float(favorable_R))
            self._open.mae_R = min(self._open.mae_R, float(adverse_R))
            result = conservative_position_bar(
                self._open.thesis.direction,
                bar,
                current_stop=self._open.current_stop,
                target=self._open.target.price,
            )
            if result.closed:
                assert result.exit_price is not None
                closed.append(
                    self._close(
                        time=bar.end,
                        price=result.exit_price,
                        reason=result.reason,
                        ambiguous=result.ambiguous_same_bar,
                    )
                )
            else:
                self._open.mark_R = (
                    self._open.thesis.direction.sign
                    * (bar.close - self._open.entry_price)
                    / self._open.risk_points
                )

        if self._open is None and self._pending_entry is not None:
            pending = self._pending_entry
            if (
                current_contract != (pending.symbol, pending.instrument_id)
                # A limit touch is only knowable at the completed-bar clock.
                # Do not create a position at or beyond its frozen deadline.
                or bar.end >= pending.thesis.deadline
            ):
                self._pending_entry = None
                return tuple(closed)
            result = conservative_entry_bar(pending.plan, bar)
            self._pending_entry = None
            if result.filled:
                self._open = _OpenTrade(
                    thesis=pending.thesis,
                    decision_time=pending.decision_time,
                    # The touch is only knowable after this completed bar.
                    # Align the fill clock with the completed-bar execution
                    # contract; the intrabar touch was not knowable earlier.
                    opened_at=bar.end,
                    entry_price=float(result.entry_price),
                    current_stop=pending.thesis.original_invalidation.price,
                    target=pending.thesis.original_targets[0],
                    risk_points=pending.plan.risk_points,
                    cost_points=pending.cost_points,
                    symbol=pending.symbol,
                    instrument_id=pending.instrument_id,
                )
                adverse_price = (
                    bar.low
                    if pending.thesis.direction is Direction.LONG
                    else bar.high
                )
                # The intrabar order between the limit touch and the favorable
                # extreme is unknown. Do not feed that optimistic suffix into
                # the next belief update; adverse excursion remains
                # conservative because it can only reduce confidence.
                self._open.mfe_R = 0.0
                self._open.mae_R = min(
                    0.0,
                    float(
                        pending.thesis.direction.sign
                        * (adverse_price - self._open.entry_price)
                        / self._open.risk_points
                    ),
                )
                if result.closed:
                    assert result.exit_price is not None
                    closed.append(
                        self._close(
                            time=bar.end,
                            price=result.exit_price,
                            reason=result.reason,
                            ambiguous=result.ambiguous_same_bar,
                        )
                    )
                else:
                    self._open.mark_R = (
                        pending.thesis.direction.sign
                        * (bar.close - self._open.entry_price)
                        / self._open.risk_points
                    )
        return tuple(closed)

    def _new_protection_candidate(
        self,
        snapshot: EngineSnapshot,
    ) -> StructuralLevel | None:
        position = self._position_snapshot(snapshot.observation.asof)
        if position is None:
            return None
        return causal_protection_candidate(position, snapshot.observation)

    def after_decision(self, snapshot: EngineSnapshot) -> None:
        action = snapshot.risk.final_action
        if action is Action.ENTER:
            if self._open is not None or self._pending_entry is not None:
                raise RuntimeError("risk approved a new entry while exposure already exists")
            if snapshot.decision.plan is None or snapshot.risk.frozen_thesis is None:
                raise RuntimeError("approved entry lacks plan or frozen thesis")
            self._pending_entry = _PendingEntry(
                plan=snapshot.decision.plan,
                thesis=snapshot.risk.frozen_thesis,
                decision_time=snapshot.observation.asof,
                cost_points=snapshot.observation.execution.expected_round_trip_cost_points,
                symbol=snapshot.observation.symbol,
                instrument_id=snapshot.observation.instrument_id,
            )
        elif action is Action.EXIT and self._open is not None:
            self._pending_exit_reason = "model_or_risk_exit"
        elif action is Action.PROTECT and self._open is not None:
            if snapshot.risk.protected_stop is None:
                raise RuntimeError("approved protection lacks a protected stop")
            if self._open.thesis.direction is Direction.LONG:
                self._open.current_stop = max(
                    self._open.current_stop, snapshot.risk.protected_stop
                )
            else:
                self._open.current_stop = min(
                    self._open.current_stop, snapshot.risk.protected_stop
                )
        if self._open is not None:
            self._open.protection_candidate = self._new_protection_candidate(snapshot)
        self._next_open_slippage_points = float(
            snapshot.observation.execution.expected_slippage_points
        )


class SequentialReplay:
    """Feeds each completed bar through execution, eyes, brain, decision, risk."""

    def __init__(
        self,
        engine: ContinuousSMCEngine | None = None,
        portfolio: SequentialPortfolio | None = None,
    ) -> None:
        self.engine = engine or ContinuousSMCEngine.from_config(
            runtime_mode="development"
        )
        self.portfolio = portfolio or SequentialPortfolio()

    def on_bar(
        self,
        bar: Bar,
        *,
        execution: ExecutionRealityInput,
    ) -> ReplayStep:
        closed = self.portfolio.before_bar(bar)
        account = self.portfolio.account(bar.end)
        belief_position = (
            self.portfolio.lifecycle_position
            if self.portfolio.lifecycle_position is not None
            else account.position
        )
        snapshot = self.engine.on_bar(
            bar,
            execution=execution,
            account=account,
            belief_position=belief_position,
        )
        self.portfolio.after_decision(snapshot)
        self.portfolio.clear_lifecycle_position()
        return ReplayStep(
            snapshot=snapshot,
            closed_trades=closed,
            position=self.portfolio.account(bar.end).position,
            account_state=account,
            belief_position_input=belief_position,
        )


__all__ = [
    "ReplayStep",
    "SequentialPortfolio",
    "SequentialReplay",
    "TradeRecord",
]
