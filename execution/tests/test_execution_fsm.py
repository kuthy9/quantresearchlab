from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from execution.core.execution import TopOfBook
from execution.core.execution_fsm import (
    CancelOrderCommand,
    EXECUTION_AUTHORITY,
    EXECUTION_PROTOCOL_FINGERPRINT,
    EXECUTION_PROTOCOL_VERSION,
    ExecutionContractError,
    ExecutionFSM,
    ExecutionFact,
    ExecutionFactKind,
    ExitPositionCommand,
    ImmutableExecutionEventStore,
    OrderRole,
    OrderSide,
    OrderState,
    OrderType,
    PositionState,
    ProtectPositionCommand,
    ReplaceOrderCommand,
    RiskApproval,
    RiskApprovedTradeIntent,
    SubmitOrderCommand,
    account_state_fingerprint,
    make_execution_event,
    risk_approve_trade_intent,
)
from shares.core.model import (
    AccountState,
    Direction,
    LiquidityLevel,
    StructuralLevel,
    Timeframe,
)
from brain.core.signal_policy import (
    CancelCondition,
    CancelConditionKind,
    SetupFamily,
)
from execution.core.trade_intent import EntryMethod, TimeInForce, TradeIntent


T0 = pd.Timestamp("2026-08-21 09:30:00", tz="America/New_York")
EXPIRY = T0 + pd.Timedelta(hours=1)
RISK_FINGERPRINT = "a" * 64


def _intent() -> TradeIntent:
    invalidation = StructuralLevel(
        price=95.0,
        side="below",
        source_level_id="structure:invalidation",
        observed_at=T0 - pd.Timedelta(minutes=1),
        rationale="frozen structural invalidation",
    )
    target = LiquidityLevel(
        level_id="liquidity:target",
        timeframe=Timeframe.H1,
        side="above",
        price=110.0,
        formed_at=T0 - pd.Timedelta(hours=2),
        confirmed_at=T0 - pd.Timedelta(hours=1),
        touches=1,
    )
    cancel = CancelCondition(
        kind=CancelConditionKind.SIGNAL_EXPIRY_REACHED,
        reference_id="signal:test",
        operator=">=",
        source_ids=("event:signal",),
        trigger_at=EXPIRY,
    )
    return TradeIntent(
        created_at=T0,
        expires_at=EXPIRY,
        signal_id="signal:test",
        candidate_id="candidate:test",
        episode_id="episode:test",
        setup_id="setup:test",
        setup_family=SetupFamily.DFP,
        competition_set_id="competition:test",
        path_hypothesis_id="path:test",
        dol_ranking_id="dol-ranking:test",
        dol_candidate_id="dol:test",
        source_event_ids=("event:path", "event:setup"),
        source_identity_ids=("identity:path", "identity:setup"),
        policy_protocol_id="signal-policy:test",
        policy_protocol_version="signal-policy-test-v1",
        policy_protocol_fingerprint="f" * 64,
        path_likelihood_artifact_id="path-artifact:test",
        path_model_id="path-model:test",
        path_model_version="path-model-test-v1",
        path_calibration_id="path-calibration:test",
        dol_calibration_artifact_id="dol-artifact:test",
        dol_model_id="dol-model:test",
        dol_model_version="dol-model-test-v1",
        dol_calibration_id="dol-calibration:test",
        outcome_model_artifact_id="outcome-artifact:test",
        outcome_model_id="outcome-model:test",
        outcome_model_version="outcome-model-test-v1",
        outcome_calibration_id="outcome-calibration:test",
        symbol="NQU6",
        instrument_id="NQ:front",
        side=Direction.LONG,
        account_snapshot_id="account:test",
        risk_budget_id="risk-budget:test",
        quantity=2,
        point_value=20.0,
        risk_budget_fraction=0.005,
        risk_budget_amount=500.0,
        position_risk_amount=200.0,
        entry_method_preferences=(
            EntryMethod.FVG_50_LIMIT,
            EntryMethod.MARKET_ENTRY,
        ),
        planned_entry=100.0,
        invalidation=invalidation,
        targets=(target,),
        trade_plan_id="trade-plan:test",
        max_wait_seconds=3600.0,
        time_in_force=TimeInForce.GOOD_TIL_TIME,
        cancel_conditions=(cancel,),
    )


def _account(**changes) -> AccountState:
    values = {
        "equity": 100_000.0,
        "open_risk_fraction": 0.0,
        "requested_risk_fraction": 0.005,
        "quantity": 2,
        "point_value": 20.0,
    }
    values.update(changes)
    return AccountState(**values)


def _approved() -> RiskApprovedTradeIntent:
    return risk_approve_trade_intent(
        _intent(),
        _account(),
        approved_at=T0,
        risk_protocol_id="risk:test",
        risk_protocol_version="risk-test-v1",
        risk_protocol_fingerprint=RISK_FINGERPRINT,
        account_snapshot_fingerprint=account_state_fingerprint(_account()),
    )


def _book(
    sequence: int = 0,
    *,
    bid: float = 99.75,
    ask: float = 100.0,
) -> TopOfBook:
    return TopOfBook(
        observed_at=T0 + pd.Timedelta(seconds=sequence),
        bid=bid,
        ask=ask,
        bid_size=10.0,
        ask_size=10.0,
    )


def _submit(
    fsm: ExecutionFSM,
    *,
    sequence: int,
    order_id: str,
    role: OrderRole,
    side: OrderSide,
    order_type: OrderType,
    quantity: int,
    entry_method: EntryMethod | None = None,
    parent_order_id: str | None = None,
    oco_group_id: str | None = None,
    limit_price: float | None = None,
    stop_price: float | None = None,
    book_bid: float = 99.75,
    book_ask: float = 100.0,
) -> None:
    if role is OrderRole.ENTRY and entry_method is None:
        entry_method = (
            EntryMethod.MARKET_ENTRY
            if order_type is OrderType.MARKET
            else EntryMethod.FVG_50_LIMIT
        )
    command = SubmitOrderCommand(
        command_id=f"command:submit:{order_id}",
        created_at=T0 + pd.Timedelta(seconds=sequence),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:command-source",),
        order_id=order_id,
        role=role,
        side=side,
        order_type=order_type,
        quantity=quantity,
        entry_method=entry_method,
        parent_order_id=parent_order_id,
        oco_group_id=oco_group_id,
        limit_price=limit_price,
        stop_price=stop_price,
    )
    fsm.accept_command(
        command,
        known_at=command.created_at,
        vendor_sequence=sequence,
        top_of_book=_book(sequence, bid=book_bid, ask=book_ask),
    )


def _record(
    fsm: ExecutionFSM,
    kind: ExecutionFactKind,
    *,
    sequence: int,
    order_id: str,
    **payload,
) -> None:
    event_source_ids = payload.pop(
        "event_source_ids",
        (f"venue-message:{sequence}",),
    )
    fact = ExecutionFact(
        kind=kind,
        approved_intent_id=fsm.approved.approved_intent_id,
        order_id=order_id,
        **payload,
    )
    clock = T0 + pd.Timedelta(seconds=sequence)
    fsm.record(
        make_execution_event(
            fact,
            event_time=clock,
            known_at=clock,
            source_event_ids=event_source_ids,
            vendor_sequence=sequence,
        )
    )


def test_risk_approval_binds_exact_trade_intent_and_has_no_live_authority() -> None:
    approved = _approved()
    assert approved.approval.trade_intent_id == approved.intent.intent_id
    assert approved.approval.policy_protocol_fingerprint == (
        approved.intent.policy_protocol_fingerprint
    )
    assert approved.approval.risk_protocol_fingerprint == RISK_FINGERPRINT
    assert approved.approval.account_snapshot_fingerprint == (
        account_state_fingerprint(_account())
    )
    assert approved.authority == EXECUTION_AUTHORITY
    assert not approved.submission_allowed
    assert not approved.approval.submission_allowed
    with pytest.raises(FrozenInstanceError):
        approved.approval.account_equity = 1.0

    wrong_policy = RiskApproval(
        **{
            **{
                field: getattr(approved.approval, field)
                for field in approved.approval.__dataclass_fields__
                if field != "approval_id"
            },
            "policy_protocol_fingerprint": "c" * 64,
        }
    )
    with pytest.raises(ExecutionContractError, match="exact TradeIntent"):
        RiskApprovedTradeIntent(intent=approved.intent, approval=wrong_policy)

    with pytest.raises(ExecutionContractError, match="account snapshot"):
        risk_approve_trade_intent(
            _intent(),
            _account(quantity=1),
            approved_at=T0,
            risk_protocol_id="risk:test",
            risk_protocol_version="risk-test-v1",
            risk_protocol_fingerprint=RISK_FINGERPRINT,
            account_snapshot_fingerprint=account_state_fingerprint(
                _account(quantity=1)
            ),
        )
    with pytest.raises(TypeError, match="RiskApprovedTradeIntent"):
        ExecutionFSM(_intent())


def test_partial_fill_then_cancel_preserves_open_quantity_and_cash() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:1",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
    )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=2, order_id="entry:1")
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=3,
        order_id="entry:1",
        quantity=1,
        price=100.0,
    )
    cancel = CancelOrderCommand(
        command_id="command:cancel:entry:1",
        created_at=T0 + pd.Timedelta(seconds=4),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:risk-cancel",),
        order_id="entry:1",
    )
    fsm.accept_command(cancel, known_at=cancel.created_at, vendor_sequence=4)
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=5,
        order_id="entry:1",
    )

    order = fsm.state.order("entry:1")
    assert order is not None
    assert order.state is OrderState.CANCELLED
    assert order.filled_quantity == 1
    assert order.remaining_quantity == 1
    assert fsm.state.position.state is PositionState.OPEN
    assert fsm.state.position.quantity == 1
    assert fsm.state.position.cash_balance == 100_000.0

    count = len(fsm.store)
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=6,
        order_id="entry:1",
        quantity=1,
        price=100.0,
    )
    assert len(fsm.store) == count + 1
    assert fsm.state.order("entry:1").state is OrderState.FILLED
    assert fsm.state.position.quantity == 2
    assert any(
        value.startswith("fill_after_cancel_ack_reconciliation_required:")
        for value in fsm.state.execution_anomalies
    )


def test_cancel_fill_race_is_resolved_by_vendor_sequence() -> None:
    fill_wins = ExecutionFSM(_approved())
    _submit(
        fill_wins,
        sequence=1,
        order_id="entry:race",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(fill_wins, ExecutionFactKind.ACKNOWLEDGED, sequence=2, order_id="entry:race")
    cancel = CancelOrderCommand(
        command_id="command:cancel:race",
        created_at=T0 + pd.Timedelta(seconds=3),
        approved_intent_id=fill_wins.approved.approved_intent_id,
        source_event_ids=("event:cancel-race",),
        order_id="entry:race",
    )
    fill_wins.accept_command(cancel, known_at=cancel.created_at, vendor_sequence=3)
    _record(
        fill_wins,
        ExecutionFactKind.FILLED,
        sequence=4,
        order_id="entry:race",
        quantity=2,
        price=100.0,
    )
    assert fill_wins.state.order("entry:race").state is OrderState.FILLED
    assert fill_wins.state.position.quantity == 2
    _record(
        fill_wins,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=5,
        order_id="entry:race",
    )
    assert fill_wins.state.order("entry:race").state is OrderState.FILLED
    assert any(
        value.startswith("cancel_ack_after_fill:")
        for value in fill_wins.state.execution_anomalies
    )

    cancel_wins = ExecutionFSM(_approved())
    _submit(
        cancel_wins,
        sequence=1,
        order_id="entry:race",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(cancel_wins, ExecutionFactKind.ACKNOWLEDGED, sequence=2, order_id="entry:race")
    cancel_wins.accept_command(cancel, known_at=cancel.created_at, vendor_sequence=3)
    _record(
        cancel_wins,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=4,
        order_id="entry:race",
    )
    _record(
        cancel_wins,
        ExecutionFactKind.FILLED,
        sequence=5,
        order_id="entry:race",
        quantity=2,
        price=100.0,
    )
    assert cancel_wins.state.order("entry:race").state is OrderState.FILLED
    assert cancel_wins.state.position.quantity == 2
    assert any(
        value.startswith("fill_after_cancel_ack_reconciliation_required:")
        for value in cancel_wins.state.execution_anomalies
    )


def test_bracket_replace_protect_and_exit_conserve_position_and_cash() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:parent",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
    )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=2, order_id="entry:parent")
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:parent",
        quantity=2,
        price=100.0,
    )
    _submit(
        fsm,
        sequence=4,
        order_id="stop:old",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:parent",
        stop_price=95.0,
    )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=5, order_id="stop:old")
    replacement = ReplaceOrderCommand(
        command_id="command:replace:stop",
        created_at=T0 + pd.Timedelta(seconds=6),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:protection",),
        order_id="stop:old",
        replacement_order_id="stop:new",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:parent",
        stop_price=96.0,
    )
    fsm.accept_command(replacement, known_at=replacement.created_at, vendor_sequence=6)
    _record(
        fsm,
        ExecutionFactKind.REPLACED,
        sequence=7,
        order_id="stop:new",
        related_order_id="stop:old",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:parent",
        stop_price=96.0,
    )
    protect = ProtectPositionCommand(
        command_id="command:protect:position",
        created_at=T0 + pd.Timedelta(seconds=8),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:confirmed-structure",),
        stop_order_id="stop:new",
        new_stop_price=96.0,
        source_level_event_id="event:confirmed-structure",
    )
    fsm.accept_command(protect, known_at=protect.created_at, vendor_sequence=8)
    _record(
        fsm,
        ExecutionFactKind.PROTECTED,
        sequence=9,
        order_id="stop:new",
        stop_price=96.0,
        source_level_event_id="event:confirmed-structure",
        event_source_ids=(
            "event:confirmed-structure",
            "venue-message:9",
        ),
    )
    assert fsm.state.position.state is PositionState.MANAGED
    assert fsm.state.position.current_stop == 96.0
    assert fsm.state.order("stop:old").state is OrderState.CANCELLED
    assert fsm.state.order("stop:new").replaces_order_id == "stop:old"

    cancel_stop = CancelOrderCommand(
        command_id="command:cancel:stop:new",
        created_at=T0 + pd.Timedelta(seconds=10),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy",),
        order_id="stop:new",
    )
    fsm.accept_command(
        cancel_stop,
        known_at=cancel_stop.created_at,
        vendor_sequence=10,
    )
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=11,
        order_id="stop:new",
    )
    assert fsm.state.position.state is PositionState.OPEN
    assert fsm.state.position.current_stop is None

    _submit(
        fsm,
        sequence=12,
        order_id="target:1",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1,
        parent_order_id="entry:parent",
        limit_price=110.0,
    )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=13, order_id="target:1")
    exit_request = ExitPositionCommand(
        command_id="command:exit:1",
        created_at=T0 + pd.Timedelta(seconds=14),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy",),
        exit_order_id="exit:1",
        quantity=1,
    )
    fsm.accept_command(exit_request, known_at=exit_request.created_at, vendor_sequence=14)
    _submit(
        fsm,
        sequence=15,
        order_id="exit:1",
        role=OrderRole.EXIT,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=1,
        parent_order_id="entry:parent",
        book_bid=105.0,
        book_ask=105.25,
    )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=16, order_id="exit:1")
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=17,
        order_id="exit:1",
        quantity=1,
        price=105.0,
    )
    assert fsm.state.position.quantity == 1
    assert fsm.state.position.state is PositionState.OPEN
    assert fsm.state.position.current_stop is None
    assert fsm.state.position.pending_exit_order_id is None
    assert fsm.state.order("stop:new").state is OrderState.CANCELLED
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=18,
        order_id="target:1",
        quantity=1,
        price=110.0,
    )
    position = fsm.state.position
    assert position.state is PositionState.EXITED
    assert position.quantity == 0
    assert position.realized_pnl == 300.0
    assert position.cash_balance == 100_300.0


def test_parent_identity_wrong_side_and_illegal_transitions_fail_closed() -> None:
    fsm = ExecutionFSM(_approved())
    count = len(fsm.store)
    with pytest.raises(ExecutionContractError, match="entry submission"):
        _submit(
            fsm,
            sequence=1,
            order_id="entry:wrong-side",
            role=OrderRole.ENTRY,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1,
        )
    assert len(fsm.store) == count

    _submit(
        fsm,
        sequence=1,
        order_id="entry:valid",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1,
    )
    with pytest.raises(ExecutionContractError, match="entry parent"):
        _submit(
            fsm,
            sequence=2,
            order_id="target:orphan",
            role=OrderRole.TARGET,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1,
            parent_order_id="entry:missing",
            limit_price=110.0,
        )
    assert len(fsm.store) == 1
    with pytest.raises(ExecutionContractError, match="fill requires working"):
        _record(
            fsm,
            ExecutionFactKind.FILLED,
            sequence=2,
            order_id="entry:valid",
            quantity=1,
            price=100.0,
        )
    assert len(fsm.store) == 1


@pytest.mark.parametrize(
    ("terminal_kind", "expected"),
    [
        (ExecutionFactKind.REJECTED, OrderState.REJECTED),
        (ExecutionFactKind.EXPIRED, OrderState.EXPIRED),
    ],
)
def test_reject_and_expire_are_explicit_terminal_states(terminal_kind, expected) -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:terminal",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1,
    )
    if terminal_kind is ExecutionFactKind.EXPIRED:
        _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=2, order_id="entry:terminal")
        terminal_sequence = 3
    else:
        terminal_sequence = 2
    _record(
        fsm,
        terminal_kind,
        sequence=terminal_sequence,
        order_id="entry:terminal",
        reason="venue terminal fact",
    )
    assert fsm.state.order("entry:terminal").state is expected


def test_aggregate_entry_and_close_reservations_require_explicit_oco_cancel() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:reserved",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    with pytest.raises(ExecutionContractError, match="entry submission"):
        _submit(
            fsm,
            sequence=2,
            order_id="entry:over-reserved",
            role=OrderRole.ENTRY,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
        )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:reserved",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:reserved",
        quantity=2,
        price=100.0,
    )
    _submit(
        fsm,
        sequence=4,
        order_id="stop:reserved",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:reserved",
        oco_group_id="oco:reserved",
        stop_price=95.0,
    )
    with pytest.raises(ExecutionContractError, match="one active stop"):
        _submit(
            fsm,
            sequence=5,
            order_id="stop:over-reserved",
            role=OrderRole.STOP,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=1,
            parent_order_id="entry:reserved",
            oco_group_id="oco:reserved",
            stop_price=95.0,
        )
    _submit(
        fsm,
        sequence=5,
        order_id="target:reserved",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=2,
        parent_order_id="entry:reserved",
        oco_group_id="oco:reserved",
        limit_price=110.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=6,
        order_id="target:reserved",
    )
    exit_request = ExitPositionCommand(
        command_id="command:exit:over-reserved",
        created_at=T0 + pd.Timedelta(seconds=7),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy",),
        exit_order_id="exit:over-reserved",
        quantity=1,
    )
    with pytest.raises(ExecutionContractError, match="exit request"):
        fsm.accept_command(
            exit_request,
            known_at=exit_request.created_at,
            vendor_sequence=7,
        )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=7,
        order_id="target:reserved",
        quantity=2,
        price=110.0,
    )
    assert fsm.state.position.state is PositionState.EXITED
    assert fsm.state.order("stop:reserved").state is OrderState.CREATED
    assert fsm.state.reconciliation_required_order_ids == ("stop:reserved",)
    cancel = CancelOrderCommand(
        command_id="command:cancel:oco-stop",
        created_at=T0 + pd.Timedelta(seconds=8),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:oco-reconciliation",),
        order_id="stop:reserved",
    )
    fsm.accept_command(cancel, known_at=cancel.created_at, vendor_sequence=8)
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=9,
        order_id="stop:reserved",
    )
    assert fsm.state.order("stop:reserved").state is OrderState.CANCELLED
    assert fsm.state.reconciliation_required_order_ids == ()


def test_late_oco_sibling_fill_is_recorded_as_overfill_anomaly() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:oco-race",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=2, order_id="entry:oco-race")
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:oco-race",
        quantity=2,
        price=100.0,
    )
    _submit(
        fsm,
        sequence=4,
        order_id="stop:oco-race",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:oco-race",
        oco_group_id="oco:race",
        stop_price=95.0,
    )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=5, order_id="stop:oco-race")
    _submit(
        fsm,
        sequence=6,
        order_id="target:oco-race",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=2,
        parent_order_id="entry:oco-race",
        oco_group_id="oco:race",
        limit_price=110.0,
    )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=7, order_id="target:oco-race")
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=8,
        order_id="target:oco-race",
        quantity=2,
        price=110.0,
    )
    assert fsm.state.order("stop:oco-race").state is OrderState.WORKING
    assert fsm.state.reconciliation_required_order_ids == ("stop:oco-race",)
    before = len(fsm.store)
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=9,
        order_id="stop:oco-race",
        quantity=1,
        price=95.0,
    )
    assert len(fsm.store) == before + 1
    assert fsm.state.order("stop:oco-race").state is OrderState.PARTIALLY_FILLED
    assert fsm.state.position.overfill_quantity == 1
    assert fsm.state.reconciliation_required_order_ids == ("stop:oco-race",)
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=10,
        order_id="stop:oco-race",
        quantity=1,
        price=95.0,
    )
    assert fsm.state.order("stop:oco-race").state is OrderState.FILLED
    assert fsm.state.position.state is PositionState.EXITED
    assert fsm.state.position.overfill_quantity == 2
    assert fsm.state.reconciliation_required_order_ids == ()
    assert any(
        value.startswith("oco_sibling_filled:")
        for value in fsm.state.execution_anomalies
    )
    assert any(
        value.startswith("overfill_reconciliation_required:")
        for value in fsm.state.execution_anomalies
    )


def test_v1_single_entry_parent_and_terminal_parent_gate_close_orders() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:single-root",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
    )
    with pytest.raises(ExecutionContractError, match="entry submission"):
        _submit(
            fsm,
            sequence=2,
            order_id="entry:second-root",
            role=OrderRole.ENTRY,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1,
            limit_price=100.0,
        )
    _record(fsm, ExecutionFactKind.ACKNOWLEDGED, sequence=2, order_id="entry:single-root")
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=3,
        order_id="entry:single-root",
        quantity=1,
        price=100.0,
    )
    with pytest.raises(ExecutionContractError, match="terminal, actually-filled"):
        _submit(
            fsm,
            sequence=4,
            order_id="target:before-entry-terminal",
            role=OrderRole.TARGET,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1,
            parent_order_id="entry:single-root",
            limit_price=110.0,
        )
    cancel = CancelOrderCommand(
        command_id="command:cancel:single-root",
        created_at=T0 + pd.Timedelta(seconds=4),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:entry-reconciliation",),
        order_id="entry:single-root",
    )
    fsm.accept_command(cancel, known_at=cancel.created_at, vendor_sequence=4)
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=5,
        order_id="entry:single-root",
    )
    _submit(
        fsm,
        sequence=6,
        order_id="target:after-entry-terminal",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1,
        parent_order_id="entry:single-root",
        limit_price=110.0,
    )
    assert fsm.state.order("target:after-entry-terminal") is not None


def test_command_lineage_known_at_expiry_and_fill_price_are_fail_closed() -> None:
    approved = _approved()
    submitted = ExecutionFact(
        kind=ExecutionFactKind.SUBMITTED,
        approved_intent_id=approved.approved_intent_id,
        order_id="entry:forged-request",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
        top_of_book=_book(0),
    )
    with pytest.raises(ExecutionContractError, match="source command lineage"):
        make_execution_event(
            submitted,
            event_time=T0,
            known_at=T0,
            source_event_ids=("forged:source",),
            vendor_sequence=1,
        )

    stale = ExecutionFSM(approved)
    created_at = EXPIRY - pd.Timedelta(seconds=1)
    stale_command = SubmitOrderCommand(
        command_id="command:stale-known-at",
        created_at=created_at,
        approved_intent_id=approved.approved_intent_id,
        source_event_ids=("event:stale-submit",),
        order_id="entry:stale-known-at",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
    )
    with pytest.raises(ExecutionContractError, match="entry submission"):
        stale.accept_command(
            stale_command,
            known_at=EXPIRY,
            vendor_sequence=1,
            top_of_book=TopOfBook(
                observed_at=created_at,
                bid=99.75,
                ask=100.0,
                bid_size=10.0,
                ask_size=10.0,
            ),
        )
    assert len(stale.store) == 0

    price = ExecutionFSM(_approved())
    _submit(
        price,
        sequence=1,
        order_id="entry:limit-price",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        limit_price=100.0,
    )
    _record(price, ExecutionFactKind.ACKNOWLEDGED, sequence=2, order_id="entry:limit-price")
    with pytest.raises(ExecutionContractError, match="fill price"):
        _record(
            price,
            ExecutionFactKind.FILLED,
            sequence=3,
            order_id="entry:limit-price",
            quantity=1,
            price=200.0,
        )
    assert len(price.store) == 2


def test_terminal_partial_exit_order_releases_reservation_for_continuation() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:partial-exit",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:partial-exit",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:partial-exit",
        quantity=2,
        price=100.0,
    )
    first_exit = ExitPositionCommand(
        command_id="command:exit:first",
        created_at=T0 + pd.Timedelta(seconds=4),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy:first",),
        exit_order_id="exit:first",
        quantity=1,
    )
    fsm.accept_command(first_exit, known_at=first_exit.created_at, vendor_sequence=4)
    _submit(
        fsm,
        sequence=5,
        order_id="exit:first",
        role=OrderRole.EXIT,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=1,
        parent_order_id="entry:partial-exit",
        book_bid=105.0,
        book_ask=105.25,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=6,
        order_id="exit:first",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=7,
        order_id="exit:first",
        quantity=1,
        price=105.0,
    )
    assert fsm.state.position.quantity == 1
    assert fsm.state.order("exit:first").state is OrderState.FILLED
    assert fsm.state.position.pending_exit_order_id is None

    continuation = ExitPositionCommand(
        command_id="command:exit:continuation",
        created_at=T0 + pd.Timedelta(seconds=8),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy:continuation",),
        exit_order_id="exit:continuation",
        quantity=1,
    )
    fsm.accept_command(
        continuation,
        known_at=continuation.created_at,
        vendor_sequence=8,
    )
    assert fsm.state.position.pending_exit_order_id == "exit:continuation"
    assert fsm.state.position.pending_exit_quantity == 1


def test_replace_ack_after_intervening_fill_cannot_overreserve_entry() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:replace-race",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:replace-race",
    )
    replacement = ReplaceOrderCommand(
        command_id="command:replace:entry-race",
        created_at=T0 + pd.Timedelta(seconds=3),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:replace-policy",),
        order_id="entry:replace-race",
        replacement_order_id="entry:replacement-stale",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
    )
    fsm.accept_command(replacement, known_at=replacement.created_at, vendor_sequence=3)
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=4,
        order_id="entry:replace-race",
        quantity=1,
        price=100.0,
    )
    count = len(fsm.store)
    with pytest.raises(ExecutionContractError, match="replace/fill race"):
        _record(
            fsm,
            ExecutionFactKind.REPLACED,
            sequence=5,
            order_id="entry:replacement-stale",
            related_order_id="entry:replace-race",
            role=OrderRole.ENTRY,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=2,
            entry_method=EntryMethod.FVG_50_LIMIT,
            limit_price=100.0,
        )
    assert len(fsm.store) == count
    assert fsm.state.position.quantity == 1
    assert fsm.state.order("entry:replace-race").remaining_quantity == 1
    assert fsm.state.order("entry:replacement-stale") is None


def test_event_dedupe_conflict_atomic_batch_checkpoint_and_replay() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:checkpoint",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:checkpoint",
    )
    original = fsm.store.events[-1]
    assert not fsm.record(original)
    conflict = replace(
        original,
        fact=replace(original.fact, reason="different immutable content"),
    )
    with pytest.raises(ExecutionContractError, match="conflicts"):
        fsm.record(conflict)

    checkpoint = fsm.store.checkpoint()
    replay = ImmutableExecutionEventStore.replay(
        fsm.approved,
        reversed(fsm.store.events),
        checkpoint=checkpoint,
    )
    assert replay.state == fsm.state
    assert replay.event_fingerprint == fsm.store.event_fingerprint
    assert replay.state_fingerprint == fsm.store.state_fingerprint
    with pytest.raises(ExecutionContractError, match="checkpoint"):
        replay.require_checkpoint(replace(checkpoint, state_fingerprint="0" * 64))

    valid_fill = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.PARTIALLY_FILLED,
            approved_intent_id=fsm.approved.approved_intent_id,
            order_id="entry:checkpoint",
            quantity=1,
            price=100.0,
        ),
        event_time=T0 + pd.Timedelta(seconds=3),
        known_at=T0 + pd.Timedelta(seconds=3),
        source_event_ids=("venue-message:3",),
        vendor_sequence=3,
    )
    illegal_second_ack = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.ACKNOWLEDGED,
            approved_intent_id=fsm.approved.approved_intent_id,
            order_id="entry:checkpoint",
        ),
        event_time=T0 + pd.Timedelta(seconds=4),
        known_at=T0 + pd.Timedelta(seconds=4),
        source_event_ids=("venue-message:4",),
        vendor_sequence=4,
    )
    before = fsm.store.checkpoint()
    with pytest.raises(ExecutionContractError, match="acknowledgement"):
        fsm.store.append_batch((valid_fill, illegal_second_ack))
    assert fsm.store.checkpoint() == before


def test_event_time_known_at_sequence_and_bbo_are_causal() -> None:
    fsm = ExecutionFSM(_approved())
    command = SubmitOrderCommand(
        command_id="command:future-book",
        created_at=T0 + pd.Timedelta(seconds=1),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:source",),
        order_id="entry:future-book",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1,
        entry_method=EntryMethod.MARKET_ENTRY,
    )
    with pytest.raises(ExecutionContractError, match="future TopOfBook"):
        fsm.accept_command(
            command,
            known_at=command.created_at,
            vendor_sequence=1,
            top_of_book=_book(2),
        )

    _submit(
        fsm,
        sequence=2,
        order_id="entry:causal",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1,
    )
    out_of_sequence = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.ACKNOWLEDGED,
            approved_intent_id=fsm.approved.approved_intent_id,
            order_id="entry:causal",
        ),
        event_time=T0 + pd.Timedelta(seconds=3),
        known_at=T0 + pd.Timedelta(seconds=3),
        source_event_ids=("venue:old-sequence",),
        vendor_sequence=1,
    )
    with pytest.raises(ExecutionContractError, match="vendor sequence"):
        fsm.record(out_of_sequence)

    with pytest.raises(ExecutionContractError, match="known_at"):
        make_execution_event(
            ExecutionFact(
                kind=ExecutionFactKind.ACKNOWLEDGED,
                approved_intent_id=fsm.approved.approved_intent_id,
                order_id="entry:causal",
            ),
            event_time=T0 + pd.Timedelta(seconds=4),
            known_at=T0 + pd.Timedelta(seconds=3),
            source_event_ids=("venue:time-travel",),
            vendor_sequence=3,
        )


def test_entry_method_is_explicit_typed_and_bound_to_approved_intent() -> None:
    restrictive_intent = replace(
        _intent(),
        entry_method_preferences=(EntryMethod.FVG_50_LIMIT,),
    )
    restrictive = risk_approve_trade_intent(
        restrictive_intent,
        _account(),
        approved_at=T0,
        risk_protocol_id="risk:test",
        risk_protocol_version="risk-test-v1",
        risk_protocol_fingerprint=RISK_FINGERPRINT,
        account_snapshot_fingerprint=account_state_fingerprint(_account()),
    )
    fsm = ExecutionFSM(restrictive)
    market = SubmitOrderCommand(
        command_id="command:market-method-not-approved",
        created_at=T0 + pd.Timedelta(seconds=1),
        approved_intent_id=restrictive.approved_intent_id,
        source_event_ids=("event:method-policy",),
        order_id="entry:market-method-not-approved",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1,
        entry_method=EntryMethod.MARKET_ENTRY,
    )
    with pytest.raises(ExecutionContractError, match="entry method"):
        fsm.accept_command(
            market,
            known_at=market.created_at,
            vendor_sequence=1,
            top_of_book=_book(1),
        )
    assert len(fsm.store) == 0

    with pytest.raises(ExecutionContractError, match="method and order type"):
        SubmitOrderCommand(
            command_id="command:wrong-method-shape",
            created_at=T0,
            approved_intent_id=restrictive.approved_intent_id,
            source_event_ids=("event:method-policy",),
            order_id="entry:wrong-method-shape",
            role=OrderRole.ENTRY,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=1,
            entry_method=EntryMethod.FVG_50_LIMIT,
            stop_price=100.0,
        )


def test_market_entry_risk_gate_and_real_slippage_fact_are_conserved() -> None:
    projected = ExecutionFSM(_approved())
    with pytest.raises(ExecutionContractError, match="risk-approved price geometry"):
        _submit(
            projected,
            sequence=1,
            order_id="entry:market-risk-breach",
            role=OrderRole.ENTRY,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2,
            book_bid=199.75,
            book_ask=200.0,
        )
    assert len(projected.store) == 0

    actual = ExecutionFSM(_approved())
    _submit(
        actual,
        sequence=1,
        order_id="entry:market-slippage",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        actual,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:market-slippage",
    )
    _record(
        actual,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:market-slippage",
        quantity=2,
        price=100.25,
    )
    assert actual.state.position.average_entry_price == 100.25
    assert actual.state.position.quantity == 2
    assert any(
        value.startswith("entry_risk_reconciliation_required:")
        for value in actual.state.execution_anomalies
    )


def test_entry_commands_use_approval_and_intent_expiry_but_cancel_does_not() -> None:
    base = _approved()
    short_approval = replace(
        base.approval,
        expires_at=T0 + pd.Timedelta(seconds=10),
        approval_id="",
    )
    approved = RiskApprovedTradeIntent(
        intent=base.intent,
        approval=short_approval,
    )

    stale_submit = ExecutionFSM(approved)
    with pytest.raises(ExecutionContractError, match="entry submission"):
        _submit(
            stale_submit,
            sequence=20,
            order_id="entry:approval-expired",
            role=OrderRole.ENTRY,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1,
            limit_price=100.0,
        )
    assert len(stale_submit.store) == 0

    replace_then_cancel = ExecutionFSM(approved)
    _submit(
        replace_then_cancel,
        sequence=1,
        order_id="entry:before-approval-expiry",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
    )
    _record(
        replace_then_cancel,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:before-approval-expiry",
    )
    stale_replace = ReplaceOrderCommand(
        command_id="command:replace-after-approval-expiry",
        created_at=T0 + pd.Timedelta(seconds=20),
        approved_intent_id=approved.approved_intent_id,
        source_event_ids=("event:replace-policy",),
        order_id="entry:before-approval-expiry",
        replacement_order_id="entry:expired-replacement",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
    )
    with pytest.raises(ExecutionContractError, match="entry replacement is stale"):
        replace_then_cancel.accept_command(
            stale_replace,
            known_at=stale_replace.created_at,
            vendor_sequence=3,
        )
    cancel = CancelOrderCommand(
        command_id="command:cancel-after-approval-expiry",
        created_at=T0 + pd.Timedelta(seconds=21),
        approved_intent_id=approved.approved_intent_id,
        source_event_ids=("event:risk-reduction",),
        order_id="entry:before-approval-expiry",
    )
    replace_then_cancel.accept_command(
        cancel,
        known_at=cancel.created_at,
        vendor_sequence=4,
    )
    assert replace_then_cancel.state.order(
        "entry:before-approval-expiry"
    ).cancel_requested


def test_command_id_has_one_store_wide_canonical_fingerprint() -> None:
    fsm = ExecutionFSM(_approved())
    submit = SubmitOrderCommand(
        command_id="command:reused",
        created_at=T0 + pd.Timedelta(seconds=1),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:submit-policy",),
        order_id="entry:command-id",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
    )
    fsm.accept_command(
        submit,
        known_at=submit.created_at,
        vendor_sequence=1,
        top_of_book=_book(1),
    )
    checkpoint = fsm.store.checkpoint()
    cancel = CancelOrderCommand(
        command_id="command:reused",
        created_at=T0 + pd.Timedelta(seconds=2),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:cancel-policy",),
        order_id="entry:command-id",
    )
    cancel_event = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.CANCEL_REQUESTED,
            approved_intent_id=fsm.approved.approved_intent_id,
            order_id="entry:command-id",
        ),
        event_time=cancel.created_at,
        known_at=cancel.created_at,
        source_event_ids=(
            "event:cancel-policy",
            cancel.command_id,
            fsm.approved.approval.approval_id,
            fsm.approved.approved_intent_id,
        ),
        vendor_sequence=2,
        source_command=cancel,
    )
    with pytest.raises(ExecutionContractError, match="command ID conflicts"):
        fsm.record(cancel_event)
    assert fsm.store.checkpoint() == checkpoint

    replay = ImmutableExecutionEventStore.replay(
        fsm.approved,
        fsm.store.events,
        checkpoint=checkpoint,
    )
    with pytest.raises(ExecutionContractError, match="command ID conflicts"):
        replay.append(cancel_event)


def test_managed_stop_replace_rebinds_only_after_explicit_protection() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:managed-replace",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:managed-replace",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:managed-replace",
        quantity=2,
        price=100.0,
    )
    _submit(
        fsm,
        sequence=4,
        order_id="stop:managed-old",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:managed-replace",
        stop_price=96.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=5,
        order_id="stop:managed-old",
    )
    protect_old = ProtectPositionCommand(
        command_id="command:protect-managed-old",
        created_at=T0 + pd.Timedelta(seconds=6),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:structure-96",),
        stop_order_id="stop:managed-old",
        new_stop_price=96.0,
        source_level_event_id="event:structure-96",
    )
    fsm.accept_command(
        protect_old,
        known_at=protect_old.created_at,
        vendor_sequence=6,
    )
    _record(
        fsm,
        ExecutionFactKind.PROTECTED,
        sequence=7,
        order_id="stop:managed-old",
        stop_price=96.0,
        source_level_event_id="event:structure-96",
        event_source_ids=("event:structure-96", "venue-message:7"),
    )
    assert fsm.state.position.state is PositionState.MANAGED

    replacement = ReplaceOrderCommand(
        command_id="command:replace-managed-stop",
        created_at=T0 + pd.Timedelta(seconds=8),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:structure-97",),
        order_id="stop:managed-old",
        replacement_order_id="stop:managed-new",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:managed-replace",
        stop_price=97.0,
    )
    fsm.accept_command(
        replacement,
        known_at=replacement.created_at,
        vendor_sequence=8,
    )
    _record(
        fsm,
        ExecutionFactKind.REPLACED,
        sequence=9,
        order_id="stop:managed-new",
        related_order_id="stop:managed-old",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:managed-replace",
        stop_price=97.0,
    )
    assert fsm.state.order("stop:managed-old").state is OrderState.CANCELLED
    assert fsm.state.order("stop:managed-new").state is OrderState.WORKING
    assert fsm.state.position.state is PositionState.OPEN
    assert fsm.state.position.current_stop is None

    protect_new = ProtectPositionCommand(
        command_id="command:protect-managed-new",
        created_at=T0 + pd.Timedelta(seconds=10),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:structure-97",),
        stop_order_id="stop:managed-new",
        new_stop_price=97.0,
        source_level_event_id="event:structure-97",
    )
    fsm.accept_command(
        protect_new,
        known_at=protect_new.created_at,
        vendor_sequence=10,
    )
    _record(
        fsm,
        ExecutionFactKind.PROTECTED,
        sequence=11,
        order_id="stop:managed-new",
        stop_price=97.0,
        source_level_event_id="event:structure-97",
        event_source_ids=("event:structure-97", "venue-message:11"),
    )
    assert fsm.state.position.state is PositionState.MANAGED
    assert fsm.state.position.current_stop == 97.0


def test_cancel_and_replace_rejections_unlock_exact_pending_request() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:request-reject",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:request-reject",
    )
    replacement = ReplaceOrderCommand(
        command_id="command:replace-will-reject",
        created_at=T0 + pd.Timedelta(seconds=3),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:replace-policy",),
        order_id="entry:request-reject",
        replacement_order_id="entry:replace-rejected",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
    )
    fsm.accept_command(
        replacement,
        known_at=replacement.created_at,
        vendor_sequence=3,
    )
    _record(
        fsm,
        ExecutionFactKind.REPLACE_REJECTED,
        sequence=4,
        order_id="entry:request-reject",
        related_order_id="entry:replace-rejected",
        reason="venue rejected replace",
    )
    order = fsm.state.order("entry:request-reject")
    assert order.pending_replacement_order_id is None
    assert order.pending_replacement_fingerprint is None
    assert order.state is OrderState.WORKING

    cancel = CancelOrderCommand(
        command_id="command:cancel-will-reject",
        created_at=T0 + pd.Timedelta(seconds=5),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:cancel-policy",),
        order_id="entry:request-reject",
    )
    fsm.accept_command(
        cancel,
        known_at=cancel.created_at,
        vendor_sequence=5,
    )
    _record(
        fsm,
        ExecutionFactKind.CANCEL_REJECTED,
        sequence=6,
        order_id="entry:request-reject",
        reason="venue rejected cancel",
    )
    order = fsm.state.order("entry:request-reject")
    assert order.state is OrderState.WORKING
    assert not order.cancel_requested


def test_oco_fill_after_cancel_ack_keeps_partial_and_full_venue_facts() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:oco-after-cancel",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:oco-after-cancel",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:oco-after-cancel",
        quantity=2,
        price=100.0,
    )
    _submit(
        fsm,
        sequence=4,
        order_id="stop:oco-after-cancel",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:oco-after-cancel",
        oco_group_id="oco:after-cancel",
        stop_price=95.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=5,
        order_id="stop:oco-after-cancel",
    )
    _submit(
        fsm,
        sequence=6,
        order_id="target:oco-after-cancel",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=2,
        parent_order_id="entry:oco-after-cancel",
        oco_group_id="oco:after-cancel",
        limit_price=110.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=7,
        order_id="target:oco-after-cancel",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=8,
        order_id="target:oco-after-cancel",
        quantity=2,
        price=110.0,
    )
    cancel = CancelOrderCommand(
        command_id="command:cancel-oco-after-fill",
        created_at=T0 + pd.Timedelta(seconds=9),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:oco-reconciliation",),
        order_id="stop:oco-after-cancel",
    )
    fsm.accept_command(
        cancel,
        known_at=cancel.created_at,
        vendor_sequence=9,
    )
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=10,
        order_id="stop:oco-after-cancel",
    )
    before = len(fsm.store)
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=11,
        order_id="stop:oco-after-cancel",
        quantity=1,
        price=95.0,
    )
    assert len(fsm.store) == before + 1
    assert fsm.state.position.overfill_quantity == 1
    assert fsm.state.order("stop:oco-after-cancel").state is OrderState.CANCELLED
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=12,
        order_id="stop:oco-after-cancel",
        quantity=1,
        price=95.0,
    )
    assert fsm.state.position.overfill_quantity == 2
    assert fsm.state.order("stop:oco-after-cancel").state is OrderState.FILLED
    assert any(
        value.startswith("fill_after_cancel_ack_reconciliation_required:")
        for value in fsm.state.execution_anomalies
    )


def test_market_exit_is_not_blocked_by_best_level_entry_depth_gate() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:exit-depth",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:exit-depth",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:exit-depth",
        quantity=2,
        price=100.0,
    )
    exit_request = ExitPositionCommand(
        command_id="command:exit-beyond-best-depth",
        created_at=T0 + pd.Timedelta(seconds=4),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy",),
        exit_order_id="exit:beyond-best-depth",
        quantity=2,
    )
    fsm.accept_command(
        exit_request,
        known_at=exit_request.created_at,
        vendor_sequence=4,
    )
    submit_exit = SubmitOrderCommand(
        command_id="command:submit-exit-beyond-best-depth",
        created_at=T0 + pd.Timedelta(seconds=5),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy",),
        order_id="exit:beyond-best-depth",
        role=OrderRole.EXIT,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=2,
        parent_order_id="entry:exit-depth",
    )
    shallow_book = TopOfBook(
        observed_at=submit_exit.created_at,
        bid=99.75,
        ask=100.0,
        bid_size=1.0,
        ask_size=1.0,
    )
    fsm.accept_command(
        submit_exit,
        known_at=submit_exit.created_at,
        vendor_sequence=5,
        top_of_book=shallow_book,
    )
    assert fsm.state.order("exit:beyond-best-depth").state is OrderState.CREATED


def test_protocol_identity_is_pinned_and_old_replay_contract_fails_closed() -> None:
    assert EXECUTION_PROTOCOL_VERSION == "phase8_execution_fsm_v1.5"
    assert EXECUTION_PROTOCOL_FINGERPRINT == (
        "cc42f67d9a98ea702e957b23e68c9306109f37a36386f58503d3dae1418d5d9b"
    )
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:protocol",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        limit_price=100.0,
    )
    with pytest.raises(ExecutionContractError, match="envelope is invalid"):
        replace(
            fsm.store.events[0],
            protocol_version="phase8_execution_fsm_v1.4",
        )
    with pytest.raises(ExecutionContractError, match="checkpoint"):
        fsm.store.require_checkpoint(
            replace(
                fsm.store.checkpoint(),
                protocol_version="phase8_execution_fsm_v1.4",
            )
        )


def test_venue_fact_time_cannot_predate_order_or_exact_request() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:causal-prerequisite",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
    )
    early_ack = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.ACKNOWLEDGED,
            approved_intent_id=fsm.approved.approved_intent_id,
            order_id="entry:causal-prerequisite",
        ),
        event_time=T0 - pd.Timedelta(minutes=5),
        known_at=T0 + pd.Timedelta(seconds=2),
        source_event_ids=("venue:early-ack",),
        vendor_sequence=2,
    )
    with pytest.raises(ExecutionContractError, match="predates order submission"):
        fsm.record(early_ack)
    assert len(fsm.store) == 1

    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:causal-prerequisite",
    )
    early_fill = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.FILLED,
            approved_intent_id=fsm.approved.approved_intent_id,
            order_id="entry:causal-prerequisite",
            quantity=2,
            price=100.0,
        ),
        event_time=T0,
        known_at=T0 + pd.Timedelta(seconds=3),
        source_event_ids=("venue:early-fill",),
        vendor_sequence=3,
    )
    with pytest.raises(ExecutionContractError, match="predates order submission"):
        fsm.record(early_fill)
    assert fsm.state.position.quantity == 0

    cancel = CancelOrderCommand(
        command_id="command:causal-cancel",
        created_at=T0 + pd.Timedelta(seconds=3),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:cancel-policy",),
        order_id="entry:causal-prerequisite",
    )
    fsm.accept_command(
        cancel,
        known_at=cancel.created_at,
        vendor_sequence=3,
    )
    early_cancel_ack = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.CANCEL_ACKNOWLEDGED,
            approved_intent_id=fsm.approved.approved_intent_id,
            order_id="entry:causal-prerequisite",
        ),
        event_time=T0 + pd.Timedelta(seconds=2, milliseconds=500),
        known_at=T0 + pd.Timedelta(seconds=4),
        source_event_ids=("venue:early-cancel-ack",),
        vendor_sequence=4,
    )
    with pytest.raises(ExecutionContractError, match="predates cancel request"):
        fsm.record(early_cancel_ack)
    assert fsm.state.order("entry:causal-prerequisite").cancel_requested


@pytest.mark.parametrize("terminal_race", ("cancel", "expire", "replace"))
def test_pending_protection_is_released_by_exact_stop_terminal_race(
    terminal_race: str,
) -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:pending-protection-race",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:pending-protection-race",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:pending-protection-race",
        quantity=2,
        price=100.0,
    )
    _submit(
        fsm,
        sequence=4,
        order_id="stop:pending-old",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:pending-protection-race",
        stop_price=96.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=5,
        order_id="stop:pending-old",
    )
    pending = ProtectPositionCommand(
        command_id=f"command:pending-protect:{terminal_race}",
        created_at=T0 + pd.Timedelta(seconds=6),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:structure-96",),
        stop_order_id="stop:pending-old",
        new_stop_price=96.0,
        source_level_event_id="event:structure-96",
    )
    fsm.accept_command(
        pending,
        known_at=pending.created_at,
        vendor_sequence=6,
    )
    assert fsm.state.position.pending_protection_order_id == "stop:pending-old"

    next_sequence = 7
    if terminal_race == "cancel":
        cancel = CancelOrderCommand(
            command_id="command:cancel-pending-protect",
            created_at=T0 + pd.Timedelta(seconds=7),
            approved_intent_id=fsm.approved.approved_intent_id,
            source_event_ids=("event:cancel-policy",),
            order_id="stop:pending-old",
        )
        fsm.accept_command(
            cancel,
            known_at=cancel.created_at,
            vendor_sequence=7,
        )
        _record(
            fsm,
            ExecutionFactKind.CANCEL_ACKNOWLEDGED,
            sequence=8,
            order_id="stop:pending-old",
        )
        next_sequence = 9
    elif terminal_race == "expire":
        _record(
            fsm,
            ExecutionFactKind.EXPIRED,
            sequence=7,
            order_id="stop:pending-old",
        )
        next_sequence = 8
    else:
        replacement = ReplaceOrderCommand(
            command_id="command:replace-pending-protect",
            created_at=T0 + pd.Timedelta(seconds=7),
            approved_intent_id=fsm.approved.approved_intent_id,
            source_event_ids=("event:structure-97",),
            order_id="stop:pending-old",
            replacement_order_id="stop:pending-new",
            role=OrderRole.STOP,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=2,
            parent_order_id="entry:pending-protection-race",
            stop_price=97.0,
        )
        fsm.accept_command(
            replacement,
            known_at=replacement.created_at,
            vendor_sequence=7,
        )
        _record(
            fsm,
            ExecutionFactKind.REPLACED,
            sequence=8,
            order_id="stop:pending-new",
            related_order_id="stop:pending-old",
            role=OrderRole.STOP,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=2,
            parent_order_id="entry:pending-protection-race",
            stop_price=97.0,
        )
        next_sequence = 9

    position = fsm.state.position
    assert position.pending_protection_order_id is None
    assert position.pending_protection_fingerprint is None
    assert position.pending_protection_requested_at is None

    if terminal_race != "replace":
        _submit(
            fsm,
            sequence=next_sequence,
            order_id="stop:pending-new",
            role=OrderRole.STOP,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=2,
            parent_order_id="entry:pending-protection-race",
            stop_price=97.0,
        )
        _record(
            fsm,
            ExecutionFactKind.ACKNOWLEDGED,
            sequence=next_sequence + 1,
            order_id="stop:pending-new",
        )
        next_sequence += 2

    protect_new = ProtectPositionCommand(
        command_id=f"command:protect-new:{terminal_race}",
        created_at=T0 + pd.Timedelta(seconds=next_sequence),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:structure-97",),
        stop_order_id="stop:pending-new",
        new_stop_price=97.0,
        source_level_event_id="event:structure-97",
    )
    fsm.accept_command(
        protect_new,
        known_at=protect_new.created_at,
        vendor_sequence=next_sequence,
    )
    _record(
        fsm,
        ExecutionFactKind.PROTECTED,
        sequence=next_sequence + 1,
        order_id="stop:pending-new",
        stop_price=97.0,
        source_level_event_id="event:structure-97",
        event_source_ids=(
            "event:structure-97",
            f"venue-message:{next_sequence + 1}",
        ),
    )
    assert fsm.state.position.state is PositionState.MANAGED
    assert fsm.state.position.current_stop == 97.0


def test_same_oco_stop_partial_fill_reconciles_pending_exit_without_losing_fact() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:exit-stop-oco",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:exit-stop-oco",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:exit-stop-oco",
        quantity=2,
        price=100.0,
    )
    exit_request = ExitPositionCommand(
        command_id="command:exit-stop-oco",
        created_at=T0 + pd.Timedelta(seconds=4),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy",),
        exit_order_id="exit:stop-oco",
        quantity=2,
    )
    fsm.accept_command(
        exit_request,
        known_at=exit_request.created_at,
        vendor_sequence=4,
    )
    _submit(
        fsm,
        sequence=5,
        order_id="exit:stop-oco",
        role=OrderRole.EXIT,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=2,
        parent_order_id="entry:exit-stop-oco",
        oco_group_id="oco:exit-stop",
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=6,
        order_id="exit:stop-oco",
    )
    _submit(
        fsm,
        sequence=7,
        order_id="stop:exit-oco",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=2,
        parent_order_id="entry:exit-stop-oco",
        oco_group_id="oco:exit-stop",
        stop_price=95.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=8,
        order_id="stop:exit-oco",
    )
    before = len(fsm.store)
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=9,
        order_id="stop:exit-oco",
        quantity=1,
        price=95.0,
    )
    assert len(fsm.store) == before + 1
    assert fsm.state.position.quantity == 1
    assert fsm.state.position.pending_exit_order_id == "exit:stop-oco"
    assert fsm.state.position.pending_exit_quantity == 2
    assert fsm.state.reconciliation_required_order_ids == ("exit:stop-oco",)

    cancel_exit = CancelOrderCommand(
        command_id="command:cancel-exit-stop-oco",
        created_at=T0 + pd.Timedelta(seconds=10),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:oco-reconciliation",),
        order_id="exit:stop-oco",
    )
    fsm.accept_command(
        cancel_exit,
        known_at=cancel_exit.created_at,
        vendor_sequence=10,
    )
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=11,
        order_id="exit:stop-oco",
    )
    assert fsm.state.position.pending_exit_order_id is None
    assert fsm.state.reconciliation_required_order_ids == ()


def test_late_cancelled_entry_fill_after_exit_reopens_truth_with_anomaly() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:late-reopen",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:late-reopen",
    )
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=3,
        order_id="entry:late-reopen",
        quantity=1,
        price=100.0,
    )
    cancel = CancelOrderCommand(
        command_id="command:cancel-late-reopen",
        created_at=T0 + pd.Timedelta(seconds=4),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:entry-reconciliation",),
        order_id="entry:late-reopen",
    )
    fsm.accept_command(
        cancel,
        known_at=cancel.created_at,
        vendor_sequence=4,
    )
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=5,
        order_id="entry:late-reopen",
    )
    _submit(
        fsm,
        sequence=6,
        order_id="target:late-reopen",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1,
        parent_order_id="entry:late-reopen",
        limit_price=110.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=7,
        order_id="target:late-reopen",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=8,
        order_id="target:late-reopen",
        quantity=1,
        price=110.0,
    )
    assert fsm.state.position.state is PositionState.EXITED
    before = len(fsm.store)
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=9,
        order_id="entry:late-reopen",
        quantity=1,
        price=100.0,
    )
    assert len(fsm.store) == before + 1
    assert fsm.state.position.state is PositionState.OPEN
    assert fsm.state.position.quantity == 1
    assert any(
        value.startswith("fill_after_cancel_ack_reconciliation_required:")
        for value in fsm.state.execution_anomalies
    )


def test_replacement_order_identity_is_reserved_globally() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:replacement-identity",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:replacement-identity",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=3,
        order_id="entry:replacement-identity",
        quantity=2,
        price=100.0,
    )
    for sequence, order_id in ((4, "target:replace-a"), (6, "target:replace-b")):
        _submit(
            fsm,
            sequence=sequence,
            order_id=order_id,
            role=OrderRole.TARGET,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1,
            parent_order_id="entry:replacement-identity",
            oco_group_id="oco:replacement-identity",
            limit_price=110.0,
        )
        _record(
            fsm,
            ExecutionFactKind.ACKNOWLEDGED,
            sequence=sequence + 1,
            order_id=order_id,
        )
    first = ReplaceOrderCommand(
        command_id="command:replace-a-shared-id",
        created_at=T0 + pd.Timedelta(seconds=8),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:replace-policy-a",),
        order_id="target:replace-a",
        replacement_order_id="target:new-shared",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1,
        parent_order_id="entry:replacement-identity",
        oco_group_id="oco:replacement-identity",
        limit_price=110.0,
    )
    fsm.accept_command(
        first,
        known_at=first.created_at,
        vendor_sequence=8,
    )
    second = ReplaceOrderCommand(
        command_id="command:replace-b-shared-id",
        created_at=T0 + pd.Timedelta(seconds=9),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:replace-policy-b",),
        order_id="target:replace-b",
        replacement_order_id="target:new-shared",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1,
        parent_order_id="entry:replacement-identity",
        oco_group_id="oco:replacement-identity",
        limit_price=110.0,
    )
    before = fsm.store.checkpoint()
    with pytest.raises(ExecutionContractError, match="replace request"):
        fsm.accept_command(
            second,
            known_at=second.created_at,
            vendor_sequence=9,
        )
    assert fsm.store.checkpoint() == before
    assert fsm.state.order(
        "target:replace-a"
    ).pending_replacement_order_id == "target:new-shared"
    assert fsm.state.order("target:replace-b").pending_replacement_order_id is None

    conflicting_exit = ExitPositionCommand(
        command_id="command:exit-shared-replacement-id",
        created_at=T0 + pd.Timedelta(seconds=9),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy",),
        exit_order_id="target:new-shared",
        quantity=1,
    )
    with pytest.raises(ExecutionContractError, match="exit request is invalid"):
        fsm.accept_command(
            conflicting_exit,
            known_at=conflicting_exit.created_at,
            vendor_sequence=9,
        )

    reserved_exit = ExitPositionCommand(
        command_id="command:reserve-exit-id",
        created_at=T0 + pd.Timedelta(seconds=9),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:exit-policy",),
        exit_order_id="exit:reserved-for-submit",
        quantity=1,
    )
    fsm.accept_command(
        reserved_exit,
        known_at=reserved_exit.created_at,
        vendor_sequence=9,
    )
    steals_exit_id = ReplaceOrderCommand(
        command_id="command:replace-steals-exit-id",
        created_at=T0 + pd.Timedelta(seconds=10),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:replace-policy-b",),
        order_id="target:replace-b",
        replacement_order_id="exit:reserved-for-submit",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1,
        parent_order_id="entry:replacement-identity",
        oco_group_id="oco:replacement-identity",
        limit_price=110.0,
    )
    with pytest.raises(ExecutionContractError, match="replace request"):
        fsm.accept_command(
            steals_exit_id,
            known_at=steals_exit_id.created_at,
            vendor_sequence=10,
        )


def test_late_entry_fill_nets_prior_close_overfill_and_replays_exactly() -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:three-way-late-race",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:three-way-late-race",
    )
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=3,
        order_id="entry:three-way-late-race",
        quantity=1,
        price=100.0,
    )
    cancel_entry = CancelOrderCommand(
        command_id="command:cancel-three-way-entry",
        created_at=T0 + pd.Timedelta(seconds=4),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:entry-reconciliation",),
        order_id="entry:three-way-late-race",
    )
    fsm.accept_command(
        cancel_entry,
        known_at=cancel_entry.created_at,
        vendor_sequence=4,
    )
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=5,
        order_id="entry:three-way-late-race",
    )
    _submit(
        fsm,
        sequence=6,
        order_id="stop:three-way-late-race",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=1,
        parent_order_id="entry:three-way-late-race",
        oco_group_id="oco:three-way-late-race",
        stop_price=95.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=7,
        order_id="stop:three-way-late-race",
    )
    _submit(
        fsm,
        sequence=8,
        order_id="target:three-way-late-race",
        role=OrderRole.TARGET,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1,
        parent_order_id="entry:three-way-late-race",
        oco_group_id="oco:three-way-late-race",
        limit_price=110.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=9,
        order_id="target:three-way-late-race",
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=10,
        order_id="target:three-way-late-race",
        quantity=1,
        price=110.0,
    )
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=11,
        order_id="stop:three-way-late-race",
        quantity=1,
        price=95.0,
    )
    assert fsm.state.position.quantity == 0
    assert fsm.state.position.overfill_quantity == 1
    before = len(fsm.store)
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=12,
        order_id="entry:three-way-late-race",
        quantity=1,
        price=100.0,
    )
    assert len(fsm.store) == before + 1
    assert fsm.state.position.state is PositionState.EXITED
    assert fsm.state.position.quantity == 0
    assert fsm.state.position.overfill_quantity == 0
    assert any(
        value.startswith("reverse_overfill_netted_by_entry:")
        for value in fsm.state.execution_anomalies
    )
    checkpoint = fsm.store.checkpoint()
    replay = ImmutableExecutionEventStore.replay(
        fsm.approved,
        reversed(fsm.store.events),
        checkpoint=checkpoint,
    )
    assert replay.state == fsm.state
    assert replay.event_fingerprint == fsm.store.event_fingerprint


@pytest.mark.parametrize("protection_completed", (False, True))
def test_late_cancelled_entry_fill_invalidates_undersized_protection(
    protection_completed: bool,
) -> None:
    fsm = ExecutionFSM(_approved())
    _submit(
        fsm,
        sequence=1,
        order_id="entry:late-protection",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=2,
        order_id="entry:late-protection",
    )
    _record(
        fsm,
        ExecutionFactKind.PARTIALLY_FILLED,
        sequence=3,
        order_id="entry:late-protection",
        quantity=1,
        price=100.0,
    )
    cancel_entry = CancelOrderCommand(
        command_id=f"command:cancel-late-protection:{protection_completed}",
        created_at=T0 + pd.Timedelta(seconds=4),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:entry-reconciliation",),
        order_id="entry:late-protection",
    )
    fsm.accept_command(
        cancel_entry,
        known_at=cancel_entry.created_at,
        vendor_sequence=4,
    )
    _record(
        fsm,
        ExecutionFactKind.CANCEL_ACKNOWLEDGED,
        sequence=5,
        order_id="entry:late-protection",
    )
    _submit(
        fsm,
        sequence=6,
        order_id="stop:late-protection",
        role=OrderRole.STOP,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=1,
        parent_order_id="entry:late-protection",
        stop_price=96.0,
    )
    _record(
        fsm,
        ExecutionFactKind.ACKNOWLEDGED,
        sequence=7,
        order_id="stop:late-protection",
    )
    protect = ProtectPositionCommand(
        command_id=f"command:late-protection:{protection_completed}",
        created_at=T0 + pd.Timedelta(seconds=8),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("event:structure-96",),
        stop_order_id="stop:late-protection",
        new_stop_price=96.0,
        source_level_event_id="event:structure-96",
    )
    fsm.accept_command(
        protect,
        known_at=protect.created_at,
        vendor_sequence=8,
    )
    late_fill_sequence = 9
    if protection_completed:
        _record(
            fsm,
            ExecutionFactKind.PROTECTED,
            sequence=9,
            order_id="stop:late-protection",
            stop_price=96.0,
            source_level_event_id="event:structure-96",
            event_source_ids=("event:structure-96", "venue-message:9"),
        )
        assert fsm.state.position.state is PositionState.MANAGED
        late_fill_sequence = 10
    else:
        assert fsm.state.position.pending_protection_order_id == (
            "stop:late-protection"
        )

    before = len(fsm.store)
    _record(
        fsm,
        ExecutionFactKind.FILLED,
        sequence=late_fill_sequence,
        order_id="entry:late-protection",
        quantity=1,
        price=100.0,
    )
    assert len(fsm.store) == before + 1
    assert fsm.state.position.quantity == 2
    assert fsm.state.position.state is PositionState.OPEN
    assert fsm.state.position.current_stop is None
    assert fsm.state.position.pending_protection_order_id is None
    assert fsm.state.position.pending_protection_fingerprint is None
    assert fsm.state.order("stop:late-protection").state is OrderState.WORKING
    assert any(
        value.startswith("protection_coverage_reconciliation_required:")
        for value in fsm.state.execution_anomalies
    )


def test_request_envelope_cannot_omit_exact_command_sources() -> None:
    fsm = ExecutionFSM(_approved())
    command = SubmitOrderCommand(
        command_id="command:source-closure",
        created_at=T0 + pd.Timedelta(seconds=1),
        approved_intent_id=fsm.approved.approved_intent_id,
        source_event_ids=("critical:policy-source",),
        order_id="entry:source-closure",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
    )
    fact = ExecutionFact(
        kind=ExecutionFactKind.SUBMITTED,
        approved_intent_id=fsm.approved.approved_intent_id,
        order_id="entry:source-closure",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        entry_method=EntryMethod.FVG_50_LIMIT,
        limit_price=100.0,
        top_of_book=_book(1),
    )
    checkpoint = fsm.store.checkpoint()
    with pytest.raises(ExecutionContractError, match="source command lineage"):
        make_execution_event(
            fact,
            event_time=command.created_at,
            known_at=command.created_at,
            source_event_ids=(
                command.command_id,
                fsm.approved.approval.approval_id,
                fsm.approved.approved_intent_id,
            ),
            vendor_sequence=1,
            source_command=command,
        )
    assert fsm.store.checkpoint() == checkpoint

    accepted = fsm.accept_command(
        command,
        known_at=command.created_at,
        vendor_sequence=1,
        top_of_book=_book(1),
    )
    assert "critical:policy-source" in accepted.source_event_ids
