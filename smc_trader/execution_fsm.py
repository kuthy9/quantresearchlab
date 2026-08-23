"""Immutable shadow execution facts and an explicit order/position FSM.

The module is deliberately a narrow Phase 8 boundary.  It accepts only a
risk-approved Phase 7 :class:`TradeIntent`; it never imports an engine
snapshot, a playbook, a broker client, or the legacy sequential portfolio.
Commands are immutable requests.  Facts are separate immutable observations
and are the *only* values allowed to mutate the replayed state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import pandas as pd

from .execution import TopOfBook
from .model import AccountState, Direction
from .trade_intent import EntryMethod, TradeIntent


# Any reducer, risk-gate, race, or reconciliation semantic change must update
# both this version and the canonical payload.  Phase 9 may additionally bind
# the runtime module digest, but replay identity must already fail closed here.
EXECUTION_PROTOCOL_VERSION = "phase8_execution_fsm_v1.5"
_EXECUTION_PROTOCOL_CANONICAL_PAYLOAD = (
    b"phase8_execution_fsm_v1.5|"
    b"shadow_authority_only|commands_separate_from_venue_facts|"
    b"exact_approval_command_and_source_lineage|command_id_unique_fingerprint|"
    b"exact_entry_method_to_order_type_permission|entry_dual_expiry_gate|"
    b"entry_market_bbo_depth_and_projected_risk_gate|"
    b"market_actual_fill_append_with_risk_anomaly|"
    b"explicit_oco_reconciliation_no_synthetic_cancel|"
    b"oco_pending_exit_reconciliation|"
    b"cancel_fill_race_append_with_anomaly_and_truthful_reopen|"
    b"late_entry_nets_reverse_overfill_and_invalidates_undersized_protection|"
    b"cancel_replace_reject_unlock_and_managed_stop_rebind|"
    b"global_prospective_order_identity_reservation|"
    b"exact_pending_protection_order_fingerprint_time_binding|"
    b"causal_submit_cancel_replace_protect_exit_event_times|"
    b"quantity_and_gross_shadow_equity_conservation_no_fees"
)
EXECUTION_PROTOCOL_FINGERPRINT = hashlib.sha256(
    _EXECUTION_PROTOCOL_CANONICAL_PAYLOAD
).hexdigest()
EXECUTION_AUTHORITY = "shadow_research_only"


class ExecutionContractError(ValueError):
    """Raised when execution history cannot be interpreted safely."""


class OrderState(str, Enum):
    CREATED = "created"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class PositionState(str, Enum):
    FLAT = "flat"
    OPEN = "open"
    MANAGED = "managed"
    EXITED = "exited"


class OrderRole(str, Enum):
    ENTRY = "entry"
    STOP = "stop"
    TARGET = "target"
    EXIT = "exit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "OrderSide":
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class ExecutionFactKind(str, Enum):
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_REQUESTED = "cancel_requested"
    CANCEL_ACKNOWLEDGED = "cancel_acknowledged"
    CANCEL_REJECTED = "cancel_rejected"
    EXPIRED = "expired"
    REPLACE_REQUESTED = "replace_requested"
    REPLACED = "replaced"
    REPLACE_REJECTED = "replace_rejected"
    PROTECT_REQUESTED = "protect_requested"
    PROTECTED = "protected"
    EXIT_REQUESTED = "exit_requested"


_TERMINAL_ORDER_STATES = {
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.EXPIRED,
    OrderState.REJECTED,
}
_CLOSE_ROLES = {OrderRole.STOP, OrderRole.TARGET, OrderRole.EXIT}
_COMMAND_FACT_KINDS = {
    ExecutionFactKind.SUBMITTED,
    ExecutionFactKind.CANCEL_REQUESTED,
    ExecutionFactKind.REPLACE_REQUESTED,
    ExecutionFactKind.PROTECT_REQUESTED,
    ExecutionFactKind.EXIT_REQUESTED,
}


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ExecutionContractError(f"{name} must be timezone aware")
    return result


def _identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExecutionContractError(f"{name} must be non-empty text")
    return value


def _fingerprint(value: Any, *, name: str) -> str:
    value = _identity(value, name=name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ExecutionContractError(f"{name} must be a lowercase sha256 digest")
    return value


def _ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(not isinstance(value, str) or not value for value in result)
    ):
        raise ExecutionContractError(
            f"{name} must contain unique non-empty identities"
        )
    return tuple(sorted(result))


def _positive(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ExecutionContractError(f"{name} must be finite and positive")
    return result


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if is_dataclass(value):
        return {
            item.name: _canonical(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entry_side(direction: Direction) -> OrderSide:
    return OrderSide.BUY if direction is Direction.LONG else OrderSide.SELL


def account_state_fingerprint(account: AccountState) -> str:
    """Content fingerprint for the exact typed account snapshot."""

    if not isinstance(account, AccountState):
        raise TypeError("account fingerprint requires AccountState")
    return _digest(account)


@dataclass(frozen=True)
class RiskApproval:
    """A content-addressed risk approval, not a general risk engine result."""

    approved_at: pd.Timestamp
    expires_at: pd.Timestamp
    trade_intent_id: str
    policy_protocol_fingerprint: str
    risk_protocol_id: str
    risk_protocol_version: str
    risk_protocol_fingerprint: str
    account_snapshot_id: str
    account_snapshot_fingerprint: str
    risk_budget_id: str
    approved_quantity: int
    account_equity: float
    approved_position_risk_amount: float
    authority: str = EXECUTION_AUTHORITY
    submission_allowed: bool = False
    approval_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "approved_at", _aware(self.approved_at, name="approval approved_at"))
        object.__setattr__(self, "expires_at", _aware(self.expires_at, name="approval expires_at"))
        for name in (
            "trade_intent_id",
            "risk_protocol_id",
            "risk_protocol_version",
            "account_snapshot_id",
            "risk_budget_id",
        ):
            _identity(getattr(self, name), name=f"approval {name}")
        for name in (
            "policy_protocol_fingerprint",
            "risk_protocol_fingerprint",
            "account_snapshot_fingerprint",
        ):
            _fingerprint(getattr(self, name), name=f"approval {name}")
        if (
            self.expires_at <= self.approved_at
            or type(self.approved_quantity) is not int
            or self.approved_quantity <= 0
            or self.authority != EXECUTION_AUTHORITY
            or type(self.submission_allowed) is not bool
            or self.submission_allowed
        ):
            raise ExecutionContractError("risk approval contract is invalid")
        _positive(self.account_equity, name="approval account_equity")
        _positive(
            self.approved_position_risk_amount,
            name="approval approved_position_risk_amount",
        )
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "approval_id"
        }
        expected = f"risk-approval:{_digest(payload)[:32]}"
        if self.approval_id and self.approval_id != expected:
            raise ExecutionContractError("risk approval identity conflicts with content")
        object.__setattr__(self, "approval_id", expected)


@dataclass(frozen=True)
class RiskApprovedTradeIntent:
    """Exact Phase 7 intent plus the exact independent risk approval."""

    intent: TradeIntent
    approval: RiskApproval
    authority: str = EXECUTION_AUTHORITY
    submission_allowed: bool = False
    approved_intent_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.intent, TradeIntent):
            raise TypeError("RiskApprovedTradeIntent requires TradeIntent")
        if not isinstance(self.approval, RiskApproval):
            raise TypeError("RiskApprovedTradeIntent requires RiskApproval")
        intent = self.intent
        approval = self.approval
        if (
            self.authority != EXECUTION_AUTHORITY
            or type(self.submission_allowed) is not bool
            or self.submission_allowed
            or approval.trade_intent_id != intent.intent_id
            or approval.policy_protocol_fingerprint
            != intent.policy_protocol_fingerprint
            or approval.account_snapshot_id != intent.account_snapshot_id
            or approval.risk_budget_id != intent.risk_budget_id
            or approval.approved_quantity != intent.quantity
            or not math.isclose(
                approval.approved_position_risk_amount,
                intent.position_risk_amount,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or approval.approved_at < intent.created_at
            or approval.approved_at >= intent.expires_at
            or approval.expires_at > intent.expires_at
        ):
            raise ExecutionContractError(
                "risk approval does not bind the exact TradeIntent"
            )
        expected = (
            "risk-approved-intent:"
            + _digest(
                {
                    "trade_intent_id": intent.intent_id,
                    "approval_id": approval.approval_id,
                    "risk_protocol_fingerprint": approval.risk_protocol_fingerprint,
                    "account_snapshot_fingerprint": approval.account_snapshot_fingerprint,
                }
            )[:32]
        )
        if self.approved_intent_id and self.approved_intent_id != expected:
            raise ExecutionContractError(
                "risk-approved intent identity conflicts with content"
            )
        object.__setattr__(self, "approved_intent_id", expected)


def risk_approve_trade_intent(
    intent: TradeIntent,
    account: AccountState,
    *,
    approved_at: pd.Timestamp,
    risk_protocol_id: str,
    risk_protocol_version: str,
    risk_protocol_fingerprint: str,
    account_snapshot_fingerprint: str,
) -> RiskApprovedTradeIntent:
    """Bind an already-approved account snapshot to one exact TradeIntent.

    The function checks arithmetic and identity conservation; it intentionally
    does not recreate the Phase 7 signal policy or the legacy Decision risk
    review.  Calling it represents the independent risk boundary's approval.
    """

    if not isinstance(intent, TradeIntent):
        raise TypeError("risk approval requires TradeIntent")
    if not isinstance(account, AccountState):
        raise TypeError("risk approval requires AccountState")
    clock = _aware(approved_at, name="risk approval clock")
    expected_risk = (
        abs(float(intent.planned_entry) - float(intent.invalidation.price))
        * float(account.point_value)
        * int(account.quantity)
    )
    expected_budget = float(account.equity) * float(account.requested_risk_fraction)
    exact_account_fingerprint = account_state_fingerprint(account)
    if (
        account.position is not None
        or account.quantity != intent.quantity
        or not math.isclose(account.point_value, intent.point_value, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            account.requested_risk_fraction,
            intent.risk_budget_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(expected_risk, intent.position_risk_amount, rel_tol=1e-12, abs_tol=1e-9)
        or not math.isclose(expected_budget, intent.risk_budget_amount, rel_tol=1e-12, abs_tol=1e-9)
        or account_snapshot_fingerprint != exact_account_fingerprint
        or account.open_risk_fraction + intent.risk_budget_fraction > 1.0 + 1e-12
        or clock < intent.created_at
        or clock >= intent.expires_at
    ):
        raise ExecutionContractError(
            "account snapshot does not exactly conserve TradeIntent risk"
        )
    approval = RiskApproval(
        approved_at=clock,
        expires_at=intent.expires_at,
        trade_intent_id=intent.intent_id,
        policy_protocol_fingerprint=intent.policy_protocol_fingerprint,
        risk_protocol_id=risk_protocol_id,
        risk_protocol_version=risk_protocol_version,
        risk_protocol_fingerprint=risk_protocol_fingerprint,
        account_snapshot_id=intent.account_snapshot_id,
        account_snapshot_fingerprint=exact_account_fingerprint,
        risk_budget_id=intent.risk_budget_id,
        approved_quantity=intent.quantity,
        account_equity=float(account.equity),
        approved_position_risk_amount=float(intent.position_risk_amount),
    )
    return RiskApprovedTradeIntent(intent=intent, approval=approval)


@dataclass(frozen=True, kw_only=True)
class ExecutionCommand:
    command_id: str
    created_at: pd.Timestamp
    approved_intent_id: str
    source_event_ids: tuple[str, ...]
    authority: str = EXECUTION_AUTHORITY
    transmission_allowed: bool = False

    def __post_init__(self) -> None:
        _identity(self.command_id, name="command_id")
        _identity(self.approved_intent_id, name="command approved_intent_id")
        object.__setattr__(self, "created_at", _aware(self.created_at, name="command created_at"))
        object.__setattr__(
            self,
            "source_event_ids",
            _ids(self.source_event_ids, name="command source_event_ids"),
        )
        if (
            self.authority != EXECUTION_AUTHORITY
            or type(self.transmission_allowed) is not bool
            or self.transmission_allowed
        ):
            raise ExecutionContractError("execution commands have no live authority")


def _validate_order_spec(
    *,
    role: OrderRole,
    side: OrderSide,
    order_type: OrderType,
    quantity: int,
    parent_order_id: str | None,
    limit_price: float | None,
    stop_price: float | None,
) -> None:
    if type(quantity) is not int or quantity <= 0:
        raise ExecutionContractError("order quantity must be a positive integer")
    if role is OrderRole.ENTRY:
        if parent_order_id is not None:
            raise ExecutionContractError("entry order cannot have a parent")
    elif not parent_order_id:
        raise ExecutionContractError("stop/target/exit order requires entry parent")
    if order_type is OrderType.MARKET:
        valid_price_shape = limit_price is None and stop_price is None
    elif order_type is OrderType.LIMIT:
        valid_price_shape = limit_price is not None and stop_price is None
        if valid_price_shape:
            _positive(limit_price, name="limit price")
    else:
        valid_price_shape = stop_price is not None and limit_price is None
        if valid_price_shape:
            _positive(stop_price, name="stop price")
    if not valid_price_shape:
        raise ExecutionContractError("order type and price fields disagree")
    if role is OrderRole.STOP and order_type is not OrderType.STOP:
        raise ExecutionContractError("protective stop role requires stop order type")
    if role is OrderRole.TARGET and order_type is not OrderType.LIMIT:
        raise ExecutionContractError("target role requires limit order type")
    if not isinstance(side, OrderSide):
        raise ExecutionContractError("order side is invalid")


def _validate_entry_method(
    *,
    role: OrderRole,
    order_type: OrderType,
    entry_method: EntryMethod | None,
) -> None:
    if role is OrderRole.ENTRY:
        if entry_method is None:
            raise ExecutionContractError("entry order requires explicit EntryMethod")
        expected_order_type = {
            EntryMethod.FVG_50_LIMIT: OrderType.LIMIT,
            EntryMethod.OB_50_LIMIT: OrderType.LIMIT,
            EntryMethod.RECLAIM_ENTRY: OrderType.LIMIT,
            EntryMethod.MARKET_ENTRY: OrderType.MARKET,
        }[entry_method]
        if order_type is not expected_order_type:
            raise ExecutionContractError(
                "entry method and order type disagree"
            )
    elif entry_method is not None:
        raise ExecutionContractError("close order cannot carry EntryMethod")


@dataclass(frozen=True, kw_only=True)
class SubmitOrderCommand(ExecutionCommand):
    order_id: str
    role: OrderRole
    side: OrderSide
    order_type: OrderType
    quantity: int
    entry_method: EntryMethod | None = None
    parent_order_id: str | None = None
    oco_group_id: str | None = None
    limit_price: float | None = None
    stop_price: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _identity(self.order_id, name="submit order_id")
        object.__setattr__(self, "role", OrderRole(self.role))
        object.__setattr__(self, "side", OrderSide(self.side))
        object.__setattr__(self, "order_type", OrderType(self.order_type))
        if self.entry_method is not None:
            object.__setattr__(self, "entry_method", EntryMethod(self.entry_method))
        _validate_order_spec(
            role=self.role,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            parent_order_id=self.parent_order_id,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
        )
        _validate_entry_method(
            role=self.role,
            order_type=self.order_type,
            entry_method=self.entry_method,
        )
        if self.oco_group_id is not None:
            _identity(self.oco_group_id, name="submit oco_group_id")
        if self.role is OrderRole.ENTRY and self.oco_group_id is not None:
            raise ExecutionContractError("entry order cannot belong to OCO group")


@dataclass(frozen=True, kw_only=True)
class CancelOrderCommand(ExecutionCommand):
    order_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _identity(self.order_id, name="cancel order_id")


@dataclass(frozen=True, kw_only=True)
class ReplaceOrderCommand(ExecutionCommand):
    order_id: str
    replacement_order_id: str
    role: OrderRole
    side: OrderSide
    order_type: OrderType
    quantity: int
    entry_method: EntryMethod | None = None
    parent_order_id: str | None = None
    oco_group_id: str | None = None
    limit_price: float | None = None
    stop_price: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _identity(self.order_id, name="replace order_id")
        _identity(self.replacement_order_id, name="replacement_order_id")
        if self.order_id == self.replacement_order_id:
            raise ExecutionContractError("replacement requires a new order identity")
        object.__setattr__(self, "role", OrderRole(self.role))
        object.__setattr__(self, "side", OrderSide(self.side))
        object.__setattr__(self, "order_type", OrderType(self.order_type))
        if self.entry_method is not None:
            object.__setattr__(self, "entry_method", EntryMethod(self.entry_method))
        _validate_order_spec(
            role=self.role,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            parent_order_id=self.parent_order_id,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
        )
        _validate_entry_method(
            role=self.role,
            order_type=self.order_type,
            entry_method=self.entry_method,
        )
        if self.oco_group_id is not None:
            _identity(self.oco_group_id, name="replace oco_group_id")
        if self.role is OrderRole.ENTRY and self.oco_group_id is not None:
            raise ExecutionContractError("entry order cannot belong to OCO group")
        if self.order_type is OrderType.MARKET:
            raise ExecutionContractError(
                "market replacement lacks a newly frozen TopOfBook"
            )


@dataclass(frozen=True, kw_only=True)
class ProtectPositionCommand(ExecutionCommand):
    stop_order_id: str
    new_stop_price: float
    source_level_event_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _identity(self.stop_order_id, name="protect stop_order_id")
        _identity(self.source_level_event_id, name="protect source_level_event_id")
        _positive(self.new_stop_price, name="protect stop price")


@dataclass(frozen=True, kw_only=True)
class ExitPositionCommand(ExecutionCommand):
    exit_order_id: str
    quantity: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _identity(self.exit_order_id, name="exit order_id")
        if type(self.quantity) is not int or self.quantity <= 0:
            raise ExecutionContractError("exit quantity must be a positive integer")


@dataclass(frozen=True)
class ExecutionFact:
    """Domain fact carried inside an immutable event envelope."""

    kind: ExecutionFactKind
    approved_intent_id: str
    order_id: str
    role: OrderRole | None = None
    side: OrderSide | None = None
    order_type: OrderType | None = None
    quantity: int | None = None
    entry_method: EntryMethod | None = None
    price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    parent_order_id: str | None = None
    oco_group_id: str | None = None
    related_order_id: str | None = None
    source_level_event_id: str | None = None
    reason: str | None = None
    top_of_book: TopOfBook | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ExecutionFactKind(self.kind))
        _identity(self.approved_intent_id, name="fact approved_intent_id")
        _identity(self.order_id, name="fact order_id")
        if self.role is not None:
            object.__setattr__(self, "role", OrderRole(self.role))
        if self.side is not None:
            object.__setattr__(self, "side", OrderSide(self.side))
        if self.order_type is not None:
            object.__setattr__(self, "order_type", OrderType(self.order_type))
        if self.entry_method is not None:
            object.__setattr__(self, "entry_method", EntryMethod(self.entry_method))
        if self.quantity is not None and (
            type(self.quantity) is not int or self.quantity <= 0
        ):
            raise ExecutionContractError("fact quantity must be a positive integer")
        for name in ("price", "limit_price", "stop_price"):
            value = getattr(self, name)
            if value is not None:
                _positive(value, name=f"fact {name}")
        if self.reason is not None and not self.reason:
            raise ExecutionContractError("fact reason cannot be empty")
        if self.top_of_book is not None and not isinstance(self.top_of_book, TopOfBook):
            raise ExecutionContractError("fact top_of_book must be TopOfBook")
        if self.oco_group_id is not None:
            _identity(self.oco_group_id, name="fact oco_group_id")
        if self.role is OrderRole.ENTRY and self.oco_group_id is not None:
            raise ExecutionContractError("entry fact cannot carry OCO group")

        spec_kinds = {
            ExecutionFactKind.SUBMITTED,
            ExecutionFactKind.REPLACE_REQUESTED,
            ExecutionFactKind.REPLACED,
        }
        if self.kind in spec_kinds:
            if self.role is None or self.side is None or self.order_type is None or self.quantity is None:
                raise ExecutionContractError("order specification fact is incomplete")
            _validate_order_spec(
                role=self.role,
                side=self.side,
                order_type=self.order_type,
                quantity=self.quantity,
                parent_order_id=self.parent_order_id,
                limit_price=self.limit_price,
                stop_price=self.stop_price,
            )
            _validate_entry_method(
                role=self.role,
                order_type=self.order_type,
                entry_method=self.entry_method,
            )
        if self.kind is ExecutionFactKind.SUBMITTED:
            if self.top_of_book is None or self.related_order_id is not None:
                raise ExecutionContractError("submitted fact requires BBO and no related order")
        elif self.top_of_book is not None:
            raise ExecutionContractError("only submitted facts may carry top of book")
        if self.kind in {
            ExecutionFactKind.PARTIALLY_FILLED,
            ExecutionFactKind.FILLED,
        } and (self.quantity is None or self.price is None):
            raise ExecutionContractError("fill fact requires quantity and price")
        if self.kind in {
            ExecutionFactKind.REPLACE_REQUESTED,
            ExecutionFactKind.REPLACED,
            ExecutionFactKind.REPLACE_REJECTED,
        } and not self.related_order_id:
            raise ExecutionContractError("replace fact requires related order identity")
        if self.kind in {
            ExecutionFactKind.PROTECT_REQUESTED,
            ExecutionFactKind.PROTECTED,
        } and (self.stop_price is None or not self.source_level_event_id):
            raise ExecutionContractError("protect fact requires stop and source identity")
        if self.kind is ExecutionFactKind.EXIT_REQUESTED and self.quantity is None:
            raise ExecutionContractError("exit request requires quantity")
        payload_fields = {
            "role",
            "side",
            "order_type",
            "quantity",
            "entry_method",
            "price",
            "limit_price",
            "stop_price",
            "parent_order_id",
            "oco_group_id",
            "related_order_id",
            "source_level_event_id",
            "top_of_book",
        }
        allowed_payload = {
            ExecutionFactKind.SUBMITTED: {
                "role",
                "side",
                "order_type",
                "quantity",
                "entry_method",
                "limit_price",
                "stop_price",
                "parent_order_id",
                "oco_group_id",
                "top_of_book",
            },
            ExecutionFactKind.ACKNOWLEDGED: set(),
            ExecutionFactKind.REJECTED: set(),
            ExecutionFactKind.PARTIALLY_FILLED: {"quantity", "price"},
            ExecutionFactKind.FILLED: {"quantity", "price"},
            ExecutionFactKind.CANCEL_REQUESTED: set(),
            ExecutionFactKind.CANCEL_ACKNOWLEDGED: set(),
            ExecutionFactKind.CANCEL_REJECTED: set(),
            ExecutionFactKind.EXPIRED: set(),
            ExecutionFactKind.REPLACE_REQUESTED: {
                "role",
                "side",
                "order_type",
                "quantity",
                "entry_method",
                "limit_price",
                "stop_price",
                "parent_order_id",
                "oco_group_id",
                "related_order_id",
            },
            ExecutionFactKind.REPLACED: {
                "role",
                "side",
                "order_type",
                "quantity",
                "entry_method",
                "limit_price",
                "stop_price",
                "parent_order_id",
                "oco_group_id",
                "related_order_id",
            },
            ExecutionFactKind.REPLACE_REJECTED: {"related_order_id"},
            ExecutionFactKind.PROTECT_REQUESTED: {
                "stop_price",
                "source_level_event_id",
            },
            ExecutionFactKind.PROTECTED: {
                "stop_price",
                "source_level_event_id",
            },
            ExecutionFactKind.EXIT_REQUESTED: {"quantity"},
        }[self.kind]
        provided_payload = {
            name for name in payload_fields if getattr(self, name) is not None
        }
        if not provided_payload.issubset(allowed_payload):
            raise ExecutionContractError(
                f"{self.kind.value} fact carries ambiguous payload fields"
            )


def _command_matches_fact(
    command: ExecutionCommand,
    fact: ExecutionFact,
) -> bool:
    if command.approved_intent_id != fact.approved_intent_id:
        return False
    if isinstance(command, SubmitOrderCommand):
        return bool(
            fact.kind is ExecutionFactKind.SUBMITTED
            and fact.order_id == command.order_id
            and fact.role is command.role
            and fact.side is command.side
            and fact.order_type is command.order_type
            and fact.quantity == command.quantity
            and fact.entry_method is command.entry_method
            and fact.parent_order_id == command.parent_order_id
            and fact.oco_group_id == command.oco_group_id
            and fact.limit_price == command.limit_price
            and fact.stop_price == command.stop_price
        )
    if isinstance(command, CancelOrderCommand):
        return bool(
            fact.kind is ExecutionFactKind.CANCEL_REQUESTED
            and fact.order_id == command.order_id
        )
    if isinstance(command, ReplaceOrderCommand):
        return bool(
            fact.kind is ExecutionFactKind.REPLACE_REQUESTED
            and fact.order_id == command.order_id
            and fact.related_order_id == command.replacement_order_id
            and fact.role is command.role
            and fact.side is command.side
            and fact.order_type is command.order_type
            and fact.quantity == command.quantity
            and fact.entry_method is command.entry_method
            and fact.parent_order_id == command.parent_order_id
            and fact.oco_group_id == command.oco_group_id
            and fact.limit_price == command.limit_price
            and fact.stop_price == command.stop_price
        )
    if isinstance(command, ProtectPositionCommand):
        return bool(
            fact.kind is ExecutionFactKind.PROTECT_REQUESTED
            and fact.order_id == command.stop_order_id
            and fact.stop_price == command.new_stop_price
            and fact.source_level_event_id == command.source_level_event_id
        )
    if isinstance(command, ExitPositionCommand):
        return bool(
            fact.kind is ExecutionFactKind.EXIT_REQUESTED
            and fact.order_id == command.exit_order_id
            and fact.quantity == command.quantity
        )
    return False


@dataclass(frozen=True)
class ExecutionEventEnvelope:
    event_id: str
    event_time: pd.Timestamp
    known_at: pd.Timestamp
    source_event_ids: tuple[str, ...]
    vendor_sequence: int
    fact: ExecutionFact
    source_command: ExecutionCommand | None = None
    protocol_version: str = EXECUTION_PROTOCOL_VERSION
    protocol_fingerprint: str = EXECUTION_PROTOCOL_FINGERPRINT
    authority: str = EXECUTION_AUTHORITY
    live_authority: bool = False

    def __post_init__(self) -> None:
        _identity(self.event_id, name="execution event_id")
        object.__setattr__(self, "event_time", _aware(self.event_time, name="execution event_time"))
        object.__setattr__(self, "known_at", _aware(self.known_at, name="execution known_at"))
        object.__setattr__(
            self,
            "source_event_ids",
            _ids(self.source_event_ids, name="execution source_event_ids"),
        )
        if not isinstance(self.fact, ExecutionFact):
            raise TypeError("execution envelope requires ExecutionFact")
        if self.fact.kind in _COMMAND_FACT_KINDS:
            if (
                not isinstance(self.source_command, ExecutionCommand)
                or not _command_matches_fact(self.source_command, self.fact)
                or self.source_command.created_at != self.event_time
                or self.source_command.command_id not in self.source_event_ids
                or not set(self.source_command.source_event_ids).issubset(
                    set(self.source_event_ids)
                )
            ):
                raise ExecutionContractError(
                    "request fact lacks exact source command lineage"
                )
        elif self.source_command is not None:
            raise ExecutionContractError(
                "venue fact cannot masquerade as a command request"
            )
        if self.known_at < self.event_time:
            raise ExecutionContractError("execution known_at precedes event_time")
        if (
            type(self.vendor_sequence) is not int
            or self.vendor_sequence < 0
            or self.protocol_version != EXECUTION_PROTOCOL_VERSION
            or self.protocol_fingerprint != EXECUTION_PROTOCOL_FINGERPRINT
            or self.authority != EXECUTION_AUTHORITY
            or type(self.live_authority) is not bool
            or self.live_authority
        ):
            raise ExecutionContractError("execution event envelope is invalid")
        if (
            self.fact.kind is ExecutionFactKind.SUBMITTED
            and self.fact.top_of_book is not None
            and self.fact.top_of_book.observed_at > self.event_time
        ):
            raise ExecutionContractError("submitted fact uses future top of book")

    @property
    def command_id(self) -> str | None:
        return (
            None if self.source_command is None else self.source_command.command_id
        )

    @property
    def command_fingerprint(self) -> str | None:
        return None if self.source_command is None else _digest(self.source_command)


# Concise alias for callers that prefer the domain term.
ExecutionEvent = ExecutionEventEnvelope


def make_execution_event(
    fact: ExecutionFact,
    *,
    event_time: pd.Timestamp,
    known_at: pd.Timestamp,
    source_event_ids: Sequence[str],
    vendor_sequence: int,
    event_id: str | None = None,
    source_command: ExecutionCommand | None = None,
) -> ExecutionEventEnvelope:
    """Create a deterministic event identity unless an upstream ID is supplied."""

    clock = _aware(event_time, name="execution event_time")
    available = _aware(known_at, name="execution known_at")
    sources = _ids(source_event_ids, name="execution source_event_ids")
    payload = {
        "event_time": clock,
        "known_at": available,
        "source_event_ids": sources,
        "vendor_sequence": vendor_sequence,
        "fact": fact,
        "source_command": source_command,
        "protocol_version": EXECUTION_PROTOCOL_VERSION,
        "protocol_fingerprint": EXECUTION_PROTOCOL_FINGERPRINT,
    }
    identity = event_id or f"execution-event:{_digest(payload)[:32]}"
    return ExecutionEventEnvelope(
        event_id=identity,
        event_time=clock,
        known_at=available,
        source_event_ids=sources,
        vendor_sequence=vendor_sequence,
        fact=fact,
        source_command=source_command,
    )


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str
    submitted_at: pd.Timestamp
    role: OrderRole
    side: OrderSide
    order_type: OrderType
    state: OrderState
    original_quantity: int
    filled_quantity: int
    average_fill_price: float | None
    entry_method: EntryMethod | None
    limit_price: float | None
    stop_price: float | None
    parent_order_id: str | None
    oco_group_id: str | None
    market_reference_price: float | None
    replaces_order_id: str | None = None
    replaced_by_order_id: str | None = None
    cancel_requested: bool = False
    cancel_requested_at: pd.Timestamp | None = None
    pending_replacement_order_id: str | None = None
    pending_replacement_fingerprint: str | None = None
    pending_replacement_requested_at: pd.Timestamp | None = None
    last_event_id: str | None = None
    last_vendor_sequence: int | None = None

    @property
    def remaining_quantity(self) -> int:
        return self.original_quantity - self.filled_quantity


@dataclass(frozen=True)
class ExecutionPositionSnapshot:
    """Gross research ledger, not broker cash or net account equity.

    ``cash_balance`` is the frozen approval equity plus realized price PnL.
    Commissions, fees, and any reverse exposure represented by ``overfill_quantity``
    are deliberately left for execution research/reconciliation; callers must
    not interpret this field as venue cash.
    """

    state: PositionState
    direction: Direction
    quantity: int
    average_entry_price: float | None
    current_stop: float | None
    realized_pnl: float
    cash_balance: float
    overfill_quantity: int = 0
    entry_order_ids: tuple[str, ...] = ()
    exit_order_ids: tuple[str, ...] = ()
    pending_protection_order_id: str | None = None
    pending_protection_fingerprint: str | None = None
    pending_protection_requested_at: pd.Timestamp | None = None
    pending_exit_order_id: str | None = None
    pending_exit_quantity: int | None = None
    pending_exit_requested_at: pd.Timestamp | None = None


@dataclass(frozen=True)
class ExecutionState:
    approved_intent_id: str
    orders: tuple[OrderSnapshot, ...]
    position: ExecutionPositionSnapshot
    reconciliation_required_order_ids: tuple[str, ...] = ()
    execution_anomalies: tuple[str, ...] = ()
    events_applied: int = 0
    last_event_id: str | None = None
    last_vendor_sequence: int | None = None
    last_known_at: pd.Timestamp | None = None

    def order(self, order_id: str) -> OrderSnapshot | None:
        return next((item for item in self.orders if item.order_id == order_id), None)


def _initial_state(approved: RiskApprovedTradeIntent) -> ExecutionState:
    return ExecutionState(
        approved_intent_id=approved.approved_intent_id,
        orders=(),
        position=ExecutionPositionSnapshot(
            state=PositionState.FLAT,
            direction=approved.intent.side,
            quantity=0,
            average_entry_price=None,
            current_stop=None,
            realized_pnl=0.0,
            cash_balance=approved.approval.account_equity,
        ),
    )


def _replacement_fingerprint(
    *,
    old_order_id: str,
    new_order_id: str,
    fact: ExecutionFact,
) -> str:
    return _digest(
        {
            "old_order_id": old_order_id,
            "new_order_id": new_order_id,
            "role": fact.role,
            "side": fact.side,
            "order_type": fact.order_type,
            "quantity": fact.quantity,
            "entry_method": fact.entry_method,
            "parent_order_id": fact.parent_order_id,
            "oco_group_id": fact.oco_group_id,
            "limit_price": fact.limit_price,
            "stop_price": fact.stop_price,
        }
    )


def _protection_fingerprint(fact: ExecutionFact) -> str:
    return _digest(
        {
            "order_id": fact.order_id,
            "stop_price": fact.stop_price,
            "source_level_event_id": fact.source_level_event_id,
        }
    )


def _orders(state: ExecutionState) -> dict[str, OrderSnapshot]:
    return {item.order_id: item for item in state.orders}


def _ordered(values: Iterable[OrderSnapshot]) -> tuple[OrderSnapshot, ...]:
    return tuple(sorted(values, key=lambda item: item.order_id))


def _require_order(
    orders: dict[str, OrderSnapshot], order_id: str
) -> OrderSnapshot:
    order = orders.get(order_id)
    if order is None:
        raise ExecutionContractError(f"unknown order identity: {order_id}")
    return order


def _active(order: OrderSnapshot) -> bool:
    return order.state not in _TERMINAL_ORDER_STATES


def _active_reservations(
    orders: Iterable[OrderSnapshot],
    *,
    roles: set[OrderRole],
) -> int:
    return sum(
        item.remaining_quantity
        for item in orders
        if item.role in roles and _active(item)
    )


def _close_reserved_quantity(
    orders: Iterable[OrderSnapshot],
    *,
    additional: tuple[OrderRole, int, str | None] | None = None,
    excluded_order_ids: set[str] | None = None,
) -> int:
    """Reserve max quantity within explicit OCO groups, sum otherwise."""

    excluded = excluded_order_ids or set()
    ungrouped = 0
    grouped: dict[str, int] = {}
    for item in orders:
        if (
            item.order_id in excluded
            or item.role not in _CLOSE_ROLES
            or not _active(item)
        ):
            continue
        if item.oco_group_id is None:
            ungrouped += item.remaining_quantity
        else:
            grouped[item.oco_group_id] = max(
                grouped.get(item.oco_group_id, 0),
                item.remaining_quantity,
            )
    if additional is not None:
        role, quantity, group_id = additional
        if role not in _CLOSE_ROLES:
            raise ExecutionContractError("entry cannot reserve close quantity")
        if group_id is None:
            ungrouped += quantity
        else:
            grouped[group_id] = max(grouped.get(group_id, 0), quantity)
    return ungrouped + sum(grouped.values())


def _event_order_update(
    order: OrderSnapshot,
    event: ExecutionEventEnvelope,
    **changes: Any,
) -> OrderSnapshot:
    return replace(
        order,
        last_event_id=event.event_id,
        last_vendor_sequence=event.vendor_sequence,
        **changes,
    )


def _validate_intent_order_prices(
    fact: ExecutionFact,
    approved: RiskApprovedTradeIntent,
) -> None:
    """Keep executable prices inside the exact risk-approved TradeIntent."""

    intent = approved.intent
    if fact.role is OrderRole.ENTRY:
        if fact.entry_method not in intent.entry_method_preferences:
            raise ExecutionContractError(
                "entry method is absent from risk-approved TradeIntent"
            )
        requested = (
            fact.limit_price
            if fact.order_type is OrderType.LIMIT
            else fact.stop_price
            if fact.order_type is OrderType.STOP
            else None
        )
        if requested is not None and not math.isclose(
            requested,
            intent.planned_entry,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ExecutionContractError(
                "entry price disagrees with risk-approved TradeIntent"
            )
    elif fact.role is OrderRole.STOP:
        assert fact.stop_price is not None
        valid = (
            fact.stop_price >= intent.invalidation.price - 1e-9
            if intent.side is Direction.LONG
            else fact.stop_price <= intent.invalidation.price + 1e-9
        )
        if not valid:
            raise ExecutionContractError(
                "protective stop loosens risk-approved invalidation"
            )
    elif fact.role is OrderRole.TARGET:
        assert fact.limit_price is not None
        if not any(
            math.isclose(
                fact.limit_price,
                target.price,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for target in intent.targets
        ):
            raise ExecutionContractError(
                "target price is absent from risk-approved TradeIntent"
            )


def _validate_fill_price(order: OrderSnapshot, fill_price: float) -> None:
    if order.order_type is OrderType.LIMIT:
        assert order.limit_price is not None
        valid = (
            fill_price <= order.limit_price + 1e-9
            if order.side is OrderSide.BUY
            else fill_price >= order.limit_price - 1e-9
        )
    elif order.order_type is OrderType.STOP:
        assert order.stop_price is not None
        valid = (
            fill_price >= order.stop_price - 1e-9
            if order.side is OrderSide.BUY
            else fill_price <= order.stop_price + 1e-9
        )
    else:
        # The frozen BBO is a causal submission input, not a promise about the
        # venue's eventual execution price.  Real market slippage must remain
        # an append-only fact; risk drift is audited after applying the fill.
        valid = order.market_reference_price is not None
    if not valid:
        raise ExecutionContractError(
            "fill price violates frozen side/order-type constraints"
        )


def _entry_geometry_is_valid(
    price: float,
    approved: RiskApprovedTradeIntent,
) -> bool:
    invalidation = float(approved.intent.invalidation.price)
    return (
        price > invalidation
        if approved.intent.side is Direction.LONG
        else price < invalidation
    )


def _entry_risk_amount(
    price: float,
    quantity: int,
    approved: RiskApprovedTradeIntent,
) -> float:
    return (
        abs(price - float(approved.intent.invalidation.price))
        * float(approved.intent.point_value)
        * quantity
    )


def _entry_risk_is_approved(
    price: float,
    quantity: int,
    approved: RiskApprovedTradeIntent,
) -> bool:
    return bool(
        _entry_geometry_is_valid(price, approved)
        and _entry_risk_amount(price, quantity, approved)
        <= approved.approval.approved_position_risk_amount + 1e-9
    )


def _apply_position_fill(
    position: ExecutionPositionSnapshot,
    order: OrderSnapshot,
    *,
    fill_quantity: int,
    fill_price: float,
    approved: RiskApprovedTradeIntent,
    allow_cancelled_entry_fill: bool = False,
) -> ExecutionPositionSnapshot:
    intent = approved.intent
    entry_side = _entry_side(intent.side)
    if order.role is OrderRole.ENTRY:
        if order.side is not entry_side:
            raise ExecutionContractError("entry fill side opposes TradeIntent")
        if (
            position.state is PositionState.EXITED
            and not allow_cancelled_entry_fill
        ):
            raise ExecutionContractError("exited lifecycle cannot reopen")
        if (
            position.state is PositionState.MANAGED
            and not allow_cancelled_entry_fill
        ):
            raise ExecutionContractError("protected position cannot add exposure")
        overfill_offset = min(position.overfill_quantity, fill_quantity)
        net_entry_quantity = fill_quantity - overfill_offset
        new_quantity = position.quantity + net_entry_quantity
        if new_quantity > approved.approval.approved_quantity:
            raise ExecutionContractError("entry fills exceed risk-approved quantity")
        previous_notional = (
            0.0
            if position.average_entry_price is None
            else position.average_entry_price * position.quantity
        )
        average = (
            None
            if new_quantity == 0
            else (
                previous_notional + fill_price * net_entry_quantity
            ) / new_quantity
        )
        return replace(
            position,
            state=(PositionState.OPEN if new_quantity > 0 else position.state),
            quantity=new_quantity,
            average_entry_price=(None if average is None else float(average)),
            overfill_quantity=position.overfill_quantity - overfill_offset,
            entry_order_ids=tuple(
                dict.fromkeys((*position.entry_order_ids, order.order_id))
            ),
        )

    if order.role not in _CLOSE_ROLES or order.side is not entry_side.opposite:
        raise ExecutionContractError("close fill side or role is invalid")
    closable_quantity = min(position.quantity, fill_quantity)
    overfill_quantity = fill_quantity - closable_quantity
    if closable_quantity > 0 and (
        position.average_entry_price is None
        or position.state not in {PositionState.OPEN, PositionState.MANAGED}
    ):
        raise ExecutionContractError("close fill has no open position basis")
    pnl = (
        intent.side.sign
        * (fill_price - float(position.average_entry_price or fill_price))
        * closable_quantity
        * intent.point_value
    )
    remaining = position.quantity - closable_quantity
    realized = position.realized_pnl + pnl
    return replace(
        position,
        state=(
            PositionState.EXITED
            if remaining == 0
            else PositionState.MANAGED
            if position.current_stop is not None
            else PositionState.OPEN
        ),
        quantity=remaining,
        average_entry_price=(None if remaining == 0 else position.average_entry_price),
        current_stop=(None if remaining == 0 else position.current_stop),
        realized_pnl=float(realized),
        cash_balance=float(approved.approval.account_equity + realized),
        overfill_quantity=position.overfill_quantity + overfill_quantity,
        exit_order_ids=tuple(
            dict.fromkeys((*position.exit_order_ids, order.order_id))
        ),
        pending_exit_order_id=(
            None if remaining == 0 else position.pending_exit_order_id
        ),
        pending_exit_quantity=(
            None if remaining == 0 else position.pending_exit_quantity
        ),
        pending_exit_requested_at=(
            None if remaining == 0 else position.pending_exit_requested_at
        ),
    )


def _reconcile_close_orders_after_fill(
    orders: dict[str, OrderSnapshot],
    position: ExecutionPositionSnapshot,
    *,
    filled_order_id: str,
    event: ExecutionEventEnvelope,
    reconciliation_required: set[str],
    anomalies: list[str],
) -> tuple[
    dict[str, OrderSnapshot],
    ExecutionPositionSnapshot,
    set[str],
    list[str],
]:
    """Request explicit reconciliation without inventing venue cancellation."""

    filled = _require_order(orders, filled_order_id)
    if filled.role not in _CLOSE_ROLES:
        return orders, position, reconciliation_required, anomalies
    was_reconciliation_required = filled_order_id in reconciliation_required
    if was_reconciliation_required:
        reconciliation_required.remove(filled_order_id)
        anomalies.append(f"oco_sibling_filled:{event.event_id}:{filled_order_id}")
    if filled.oco_group_id is not None:
        reconciliation_required.update(
            item.order_id
            for item in orders.values()
            if item.order_id != filled_order_id
            and item.oco_group_id == filled.oco_group_id
            and item.role in _CLOSE_ROLES
            and _active(item)
        )
    if was_reconciliation_required and _active(filled):
        # A late partial sibling fill is still a live venue order.  Keep it out
        # of normal close reservations until an explicit terminal venue fact
        # arrives; otherwise the real partial fill would be rolled back by the
        # post-reduction reservation invariant.
        reconciliation_required.add(filled_order_id)
    if position.quantity == 0:
        position = replace(
            position,
            state=PositionState.EXITED,
            current_stop=None,
            pending_protection_order_id=None,
            pending_protection_fingerprint=None,
            pending_protection_requested_at=None,
            pending_exit_order_id=None,
            pending_exit_quantity=None,
            pending_exit_requested_at=None,
        )
    active_stop_covers_position = any(
        item.role is OrderRole.STOP
        and _active(item)
        and item.order_id not in reconciliation_required
        and item.remaining_quantity >= position.quantity
        and position.current_stop is not None
        and item.stop_price is not None
        and math.isclose(
            item.stop_price,
            position.current_stop,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for item in orders.values()
    )
    if position.state is PositionState.MANAGED and not active_stop_covers_position:
        position = replace(
            position,
            state=PositionState.OPEN,
            current_stop=None,
            pending_protection_order_id=None,
            pending_protection_fingerprint=None,
            pending_protection_requested_at=None,
        )
    return orders, position, reconciliation_required, anomalies


def _validate_state(
    state: ExecutionState, approved: RiskApprovedTradeIntent
) -> None:
    orders = {item.order_id: item for item in state.orders}
    if len(orders) != len(state.orders):
        raise ExecutionContractError("order identities are not unique")
    entry_fills = 0
    close_fills = 0
    for order in state.orders:
        if (
            order.original_quantity <= 0
            or order.filled_quantity < 0
            or order.filled_quantity > order.original_quantity
            or (order.filled_quantity == 0) != (order.average_fill_price is None)
        ):
            raise ExecutionContractError("order quantity/fill conservation failed")
        _aware(order.submitted_at, name="order submitted_at")
        if (order.cancel_requested_at is not None) != order.cancel_requested:
            raise ExecutionContractError("cancel request timestamp diverged")
        pending_replacement_fields = (
            order.pending_replacement_order_id,
            order.pending_replacement_fingerprint,
            order.pending_replacement_requested_at,
        )
        if any(value is None for value in pending_replacement_fields) != all(
            value is None for value in pending_replacement_fields
        ):
            raise ExecutionContractError("replacement request timestamps diverged")
        if order.parent_order_id is not None:
            parent = orders.get(order.parent_order_id)
            if parent is None or parent.role is not OrderRole.ENTRY:
                raise ExecutionContractError("child order has no exact entry parent")
        if order.role is OrderRole.ENTRY:
            if order.entry_method not in approved.intent.entry_method_preferences:
                raise ExecutionContractError(
                    "entry order method diverges from approved intent"
                )
            _validate_entry_method(
                role=order.role,
                order_type=order.order_type,
                entry_method=order.entry_method,
            )
            entry_fills += order.filled_quantity
        else:
            if order.entry_method is not None:
                raise ExecutionContractError("close order carries EntryMethod")
            close_fills += order.filled_quantity
    pending_replacement_ids = tuple(
        order.pending_replacement_order_id
        for order in state.orders
        if order.pending_replacement_order_id is not None
    )
    if (
        len(pending_replacement_ids) != len(set(pending_replacement_ids))
        or any(order_id in orders for order_id in pending_replacement_ids)
        or (
            state.position.pending_exit_order_id is not None
            and state.position.pending_exit_order_id not in orders
            and state.position.pending_exit_order_id in pending_replacement_ids
        )
    ):
        raise ExecutionContractError(
            "replacement order identity is not globally reserved"
        )
    net_quantity = entry_fills - close_fills
    expected_quantity = max(0, net_quantity)
    expected_overfill = max(0, -net_quantity)
    position = state.position
    if (
        expected_quantity != position.quantity
        or expected_overfill != position.overfill_quantity
    ):
        raise ExecutionContractError("order fills and position quantity diverged")
    if expected_quantity > approved.approval.approved_quantity:
        raise ExecutionContractError("position exceeds approved quantity")
    active_entry_reservation = _active_reservations(
        state.orders,
        roles={OrderRole.ENTRY},
    )
    if position.quantity + active_entry_reservation > approved.approval.approved_quantity:
        raise ExecutionContractError("working entry exposure exceeds approval")
    active_entries = tuple(
        item
        for item in state.orders
        if item.role is OrderRole.ENTRY and _active(item)
    )
    if len(active_entries) > 1:
        raise ExecutionContractError("v1 permits one active entry parent")
    active_stops = tuple(
        item
        for item in state.orders
        if item.role is OrderRole.STOP and _active(item)
    )
    if len(active_stops) > 1:
        raise ExecutionContractError("v1 permits one active stop")
    pending_unsubmitted_exit = 0
    if (
        position.pending_exit_order_id is not None
        and position.pending_exit_order_id not in orders
    ):
        pending_unsubmitted_exit = int(position.pending_exit_quantity or 0)
    reconciliation_required = set(state.reconciliation_required_order_ids)
    active_close_reservation = _close_reserved_quantity(
        state.orders,
        excluded_order_ids=reconciliation_required,
    )
    if active_close_reservation + pending_unsubmitted_exit > position.quantity:
        raise ExecutionContractError("close reservation exceeds open position")
    if (
        len(reconciliation_required)
        != len(state.reconciliation_required_order_ids)
        or any(
            (order := orders.get(order_id)) is None
            or order.role not in _CLOSE_ROLES
            or not _active(order)
            for order_id in reconciliation_required
        )
    ):
        raise ExecutionContractError("close reconciliation identity is invalid")
    if len(state.execution_anomalies) != len(set(state.execution_anomalies)):
        raise ExecutionContractError("execution anomaly identities are duplicated")
    if not math.isclose(
        position.cash_balance,
        approved.approval.account_equity + position.realized_pnl,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ExecutionContractError(
            "gross shadow equity and realized PnL are not conserved"
        )
    if position.quantity == 0:
        if position.state not in {PositionState.FLAT, PositionState.EXITED} or position.average_entry_price is not None:
            raise ExecutionContractError("flat position state is inconsistent")
    elif position.state not in {PositionState.OPEN, PositionState.MANAGED} or position.average_entry_price is None:
        raise ExecutionContractError("open position state is inconsistent")
    if position.state is PositionState.MANAGED:
        if position.current_stop is None or not any(
            item.role is OrderRole.STOP
            and _active(item)
            and item.order_id not in reconciliation_required
            and item.remaining_quantity >= position.quantity
            and item.stop_price is not None
            and math.isclose(
                item.stop_price,
                position.current_stop,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for item in state.orders
        ):
            raise ExecutionContractError(
                "managed position lacks exact active protection"
            )
    pending_id = position.pending_exit_order_id
    if pending_id is not None:
        if (
            position.pending_exit_quantity is None
            or position.pending_exit_requested_at is None
            or position.pending_exit_quantity <= 0
            or (
                position.pending_exit_quantity > position.quantity
                and pending_id not in reconciliation_required
            )
        ):
            raise ExecutionContractError("pending exit reservation is invalid")
        pending_order = orders.get(pending_id)
        if pending_order is not None and (
            pending_order.role is not OrderRole.EXIT
            or not _active(pending_order)
            or pending_order.remaining_quantity != position.pending_exit_quantity
        ):
            raise ExecutionContractError("pending exit and order state diverged")
    elif (
        position.pending_exit_quantity is not None
        or position.pending_exit_requested_at is not None
    ):
        raise ExecutionContractError("pending exit quantity has no identity")
    pending_protection_fields = (
        position.pending_protection_order_id,
        position.pending_protection_fingerprint,
        position.pending_protection_requested_at,
    )
    if any(value is None for value in pending_protection_fields) != all(
        value is None for value in pending_protection_fields
    ):
        raise ExecutionContractError("protection request timestamps diverged")


def reduce_execution_event(
    state: ExecutionState,
    event: ExecutionEventEnvelope,
    approved: RiskApprovedTradeIntent,
) -> ExecutionState:
    """Pure fail-closed reducer for one immutable fact."""

    if not isinstance(state, ExecutionState) or not isinstance(event, ExecutionEventEnvelope):
        raise TypeError("execution reducer requires typed state and event")
    if not isinstance(approved, RiskApprovedTradeIntent):
        raise TypeError("execution reducer requires risk-approved intent")
    fact = event.fact
    if (
        state.approved_intent_id != approved.approved_intent_id
        or fact.approved_intent_id != approved.approved_intent_id
    ):
        raise ExecutionContractError("event belongs to a different approved intent")
    orders = _orders(state)
    position = state.position
    kind = fact.kind
    reconciliation_required = set(state.reconciliation_required_order_ids)
    anomalies = list(state.execution_anomalies)
    existing_order = orders.get(fact.order_id)
    if (
        existing_order is not None
        and kind is not ExecutionFactKind.SUBMITTED
        and event.event_time < existing_order.submitted_at
    ):
        raise ExecutionContractError(
            "execution fact predates order submission"
        )
    if (
        kind in _COMMAND_FACT_KINDS
        and state.last_known_at is not None
        and event.event_time < state.last_known_at
    ):
        raise ExecutionContractError(
            "execution command predates already-known order state"
        )
    if reconciliation_required and kind in {
        ExecutionFactKind.SUBMITTED,
        ExecutionFactKind.REPLACE_REQUESTED,
        ExecutionFactKind.PROTECT_REQUESTED,
        ExecutionFactKind.EXIT_REQUESTED,
    }:
        raise ExecutionContractError(
            "close reconciliation must finish before new execution commands"
        )

    if kind is ExecutionFactKind.SUBMITTED:
        if fact.order_id in orders:
            raise ExecutionContractError("order identity already exists")
        if any(
            item.pending_replacement_order_id == fact.order_id
            for item in orders.values()
        ):
            raise ExecutionContractError(
                "submitted order identity is reserved for replacement"
            )
        assert fact.role is not None and fact.side is not None
        assert fact.order_type is not None and fact.quantity is not None
        if fact.quantity > approved.approval.approved_quantity:
            raise ExecutionContractError("submitted quantity exceeds approval")
        entry_side = _entry_side(approved.intent.side)
        _validate_intent_order_prices(fact, approved)
        assert fact.top_of_book is not None
        displayed_market_size = (
            fact.top_of_book.ask_size
            if fact.side is OrderSide.BUY
            else fact.top_of_book.bid_size
        )
        market_reference_price = (
            fact.top_of_book.ask
            if fact.order_type is OrderType.MARKET
            and fact.side is OrderSide.BUY
            else fact.top_of_book.bid
            if fact.order_type is OrderType.MARKET
            and fact.side is OrderSide.SELL
            else None
        )
        if (
            fact.order_type is OrderType.MARKET
            and fact.role is OrderRole.ENTRY
            and displayed_market_size < fact.quantity
        ):
            raise ExecutionContractError(
                "market order exceeds frozen top-of-book size"
            )
        if fact.role is OrderRole.ENTRY:
            active_entry_remaining = sum(
                item.remaining_quantity
                for item in orders.values()
                if item.role is OrderRole.ENTRY
                and item.state not in _TERMINAL_ORDER_STATES
            )
            if (
                fact.side is not entry_side
                or event.event_time < approved.approval.approved_at
                or event.event_time >= approved.approval.expires_at
                or event.known_at >= approved.approval.expires_at
                or event.event_time >= approved.intent.expires_at
                or event.known_at >= approved.intent.expires_at
                or fact.top_of_book.observed_at < approved.approval.approved_at
                or position.state is PositionState.EXITED
                or any(
                    item.role is OrderRole.ENTRY
                    for item in orders.values()
                )
                or active_entry_remaining > 0
                or position.quantity + active_entry_remaining + fact.quantity
                > approved.approval.approved_quantity
            ):
                raise ExecutionContractError("entry submission is stale or wrong-side")
            if (
                market_reference_price is not None
                and not _entry_risk_is_approved(
                    market_reference_price,
                    fact.quantity,
                    approved,
                )
            ):
                raise ExecutionContractError(
                    "market entry exceeds risk-approved price geometry"
                )
        else:
            parent = orders.get(fact.parent_order_id or "")
            if parent is None:
                raise ExecutionContractError(
                    "bracket child has no exact entry parent"
                )
            if parent.role is not OrderRole.ENTRY or fact.side is not entry_side.opposite or fact.quantity > parent.original_quantity:
                raise ExecutionContractError("bracket child identity/specification is invalid")
            if (
                fact.parent_order_id not in position.entry_order_ids
                or _active_reservations(
                    orders.values(),
                    roles={OrderRole.ENTRY},
                )
                > 0
            ):
                raise ExecutionContractError(
                    "close requires a terminal, actually-filled entry parent"
                )
            if position.quantity <= 0 or fact.quantity > position.quantity:
                raise ExecutionContractError(
                    "close order exceeds the current open position"
                )
            if fact.role is OrderRole.STOP and any(
                item.role is OrderRole.STOP and _active(item)
                for item in orders.values()
            ):
                raise ExecutionContractError("v1 permits one active stop")
            reserved = _close_reserved_quantity(
                orders.values(),
                additional=(fact.role, fact.quantity, fact.oco_group_id),
            )
            if (
                position.pending_exit_order_id is not None
                and position.pending_exit_order_id not in orders
                and position.pending_exit_order_id != fact.order_id
            ):
                reserved += int(position.pending_exit_quantity or 0)
            if reserved > position.quantity:
                raise ExecutionContractError(
                    "close reservation exceeds the current position"
                )
            if fact.role is OrderRole.EXIT:
                if (
                    position.quantity <= 0
                    or position.pending_exit_order_id != fact.order_id
                    or position.pending_exit_quantity != fact.quantity
                    or position.pending_exit_requested_at is None
                    or event.event_time < position.pending_exit_requested_at
                ):
                    raise ExecutionContractError("exit order lacks exact exit request")
        orders[fact.order_id] = OrderSnapshot(
            order_id=fact.order_id,
            submitted_at=event.event_time,
            role=fact.role,
            side=fact.side,
            order_type=fact.order_type,
            state=OrderState.CREATED,
            original_quantity=fact.quantity,
            filled_quantity=0,
            average_fill_price=None,
            entry_method=fact.entry_method,
            limit_price=fact.limit_price,
            stop_price=fact.stop_price,
            parent_order_id=fact.parent_order_id,
            oco_group_id=fact.oco_group_id,
            market_reference_price=market_reference_price,
            last_event_id=event.event_id,
            last_vendor_sequence=event.vendor_sequence,
        )

    elif kind is ExecutionFactKind.ACKNOWLEDGED:
        order = _require_order(orders, fact.order_id)
        if order.state is not OrderState.CREATED:
            raise ExecutionContractError("acknowledgement requires created order")
        orders[order.order_id] = _event_order_update(order, event, state=OrderState.WORKING)

    elif kind is ExecutionFactKind.REJECTED:
        order = _require_order(orders, fact.order_id)
        if order.state is not OrderState.CREATED:
            raise ExecutionContractError("rejection requires created order")
        orders[order.order_id] = _event_order_update(order, event, state=OrderState.REJECTED)
        reconciliation_required.discard(order.order_id)
        if order.role is OrderRole.EXIT and position.pending_exit_order_id == order.order_id:
            position = replace(
                position,
                pending_exit_order_id=None,
                pending_exit_quantity=None,
                pending_exit_requested_at=None,
            )

    elif kind in {ExecutionFactKind.PARTIALLY_FILLED, ExecutionFactKind.FILLED}:
        order = _require_order(orders, fact.order_id)
        fill_after_cancel = order.state is OrderState.CANCELLED and order.cancel_requested
        if order.state not in {OrderState.WORKING, OrderState.PARTIALLY_FILLED} and not fill_after_cancel:
            raise ExecutionContractError("fill requires working order")
        assert fact.quantity is not None and fact.price is not None
        _validate_fill_price(order, fact.price)
        if (
            order.role in _CLOSE_ROLES
            and fact.quantity > position.quantity
            and order.order_id not in reconciliation_required
            and not fill_after_cancel
        ):
            raise ExecutionContractError(
                "close fill exceeds position outside OCO reconciliation"
            )
        remaining = order.remaining_quantity
        if (
            fact.quantity > remaining
            or (kind is ExecutionFactKind.PARTIALLY_FILLED and fact.quantity >= remaining)
            or (kind is ExecutionFactKind.FILLED and fact.quantity != remaining)
        ):
            raise ExecutionContractError("incremental fill quantity is inconsistent")
        total_filled = order.filled_quantity + fact.quantity
        average = (
            (order.average_fill_price or 0.0) * order.filled_quantity
            + fact.price * fact.quantity
        ) / total_filled
        new_state = (
            OrderState.FILLED
            if kind is ExecutionFactKind.FILLED
            else OrderState.CANCELLED
            if fill_after_cancel
            else OrderState.PARTIALLY_FILLED
        )
        orders[order.order_id] = _event_order_update(
            order,
            event,
            state=new_state,
            filled_quantity=total_filled,
            average_fill_price=float(average),
        )
        if fill_after_cancel:
            anomalies.append(
                f"fill_after_cancel_ack_reconciliation_required:"
                f"{event.event_id}:{order.order_id}"
            )
        prior_overfill = position.overfill_quantity
        position = _apply_position_fill(
            position,
            order,
            fill_quantity=fact.quantity,
            fill_price=fact.price,
            approved=approved,
            allow_cancelled_entry_fill=fill_after_cancel,
        )
        if (
            order.role is OrderRole.ENTRY
            and fill_after_cancel
            and (
                position.current_stop is not None
                or position.pending_protection_order_id is not None
            )
        ):
            position = replace(
                position,
                state=PositionState.OPEN,
                current_stop=None,
                pending_protection_order_id=None,
                pending_protection_fingerprint=None,
                pending_protection_requested_at=None,
            )
            anomalies.append(
                f"protection_coverage_reconciliation_required:"
                f"{event.event_id}:{order.order_id}"
            )
        if (
            order.role is OrderRole.ENTRY
            and position.average_entry_price is not None
            and not _entry_risk_is_approved(
                position.average_entry_price,
                position.quantity,
                approved,
            )
        ):
            actual_risk = _entry_risk_amount(
                position.average_entry_price,
                position.quantity,
                approved,
            )
            anomalies.append(
                f"entry_risk_reconciliation_required:{event.event_id}:"
                f"{actual_risk:.12g}"
            )
        filled_order = orders[order.order_id]
        if filled_order.role is OrderRole.EXIT:
            if filled_order.state is OrderState.FILLED:
                position = replace(
                    position,
                    pending_exit_order_id=None,
                    pending_exit_quantity=None,
                    pending_exit_requested_at=None,
                )
            elif position.pending_exit_order_id == filled_order.order_id:
                position = replace(
                    position,
                    pending_exit_quantity=filled_order.remaining_quantity,
                )
        (
            orders,
            position,
            reconciliation_required,
            anomalies,
        ) = _reconcile_close_orders_after_fill(
            orders,
            position,
            filled_order_id=order.order_id,
            event=event,
            reconciliation_required=reconciliation_required,
            anomalies=anomalies,
        )
        if position.overfill_quantity > prior_overfill:
            anomalies.append(
                f"overfill_reconciliation_required:{event.event_id}:"
                f"{position.overfill_quantity - prior_overfill}"
            )
        elif position.overfill_quantity < prior_overfill:
            anomalies.append(
                f"reverse_overfill_netted_by_entry:"
                f"{event.event_id}:{prior_overfill - position.overfill_quantity}"
            )

    elif kind is ExecutionFactKind.CANCEL_REQUESTED:
        order = _require_order(orders, fact.order_id)
        if (
            order.state in _TERMINAL_ORDER_STATES
            or order.cancel_requested
            or order.pending_replacement_order_id is not None
        ):
            raise ExecutionContractError("cancel request is illegal or duplicated")
        orders[order.order_id] = _event_order_update(
            order,
            event,
            cancel_requested=True,
            cancel_requested_at=event.event_time,
        )

    elif kind is ExecutionFactKind.CANCEL_ACKNOWLEDGED:
        order = _require_order(orders, fact.order_id)
        if (
            order.cancel_requested_at is None
            or event.event_time < order.cancel_requested_at
        ):
            raise ExecutionContractError(
                "cancel acknowledgement predates cancel request"
            )
        cancel_ack_after_fill = (
            order.state is OrderState.FILLED and order.cancel_requested
        )
        if (
            order.state in _TERMINAL_ORDER_STATES
            and not cancel_ack_after_fill
        ) or not order.cancel_requested:
            raise ExecutionContractError("cancel acknowledgement lacks live request")
        orders[order.order_id] = _event_order_update(
            order,
            event,
            state=(OrderState.FILLED if cancel_ack_after_fill else OrderState.CANCELLED),
            cancel_requested=(False if cancel_ack_after_fill else True),
            cancel_requested_at=(
                None if cancel_ack_after_fill else order.cancel_requested_at
            ),
        )
        if cancel_ack_after_fill:
            anomalies.append(
                f"cancel_ack_after_fill:{event.event_id}:{order.order_id}"
            )
        reconciliation_required.discard(order.order_id)
        if (
            order.role is OrderRole.STOP
            and position.current_stop is not None
            and order.stop_price == position.current_stop
        ):
            position = replace(
                position,
                state=PositionState.OPEN,
                current_stop=None,
                pending_protection_order_id=None,
                pending_protection_fingerprint=None,
                pending_protection_requested_at=None,
            )
        elif (
            order.role is OrderRole.STOP
            and position.pending_protection_order_id == order.order_id
        ):
            position = replace(
                position,
                pending_protection_order_id=None,
                pending_protection_fingerprint=None,
                pending_protection_requested_at=None,
            )
        if order.role is OrderRole.EXIT and position.pending_exit_order_id == order.order_id:
            position = replace(
                position,
                pending_exit_order_id=None,
                pending_exit_quantity=None,
                pending_exit_requested_at=None,
            )

    elif kind is ExecutionFactKind.CANCEL_REJECTED:
        order = _require_order(orders, fact.order_id)
        if not order.cancel_requested:
            raise ExecutionContractError("cancel rejection lacks live request")
        if (
            order.cancel_requested_at is None
            or event.event_time < order.cancel_requested_at
        ):
            raise ExecutionContractError("cancel rejection predates cancel request")
        if order.state in _TERMINAL_ORDER_STATES:
            orders[order.order_id] = _event_order_update(order, event)
            anomalies.append(
                f"cancel_reject_after_terminal:{event.event_id}:{order.order_id}"
            )
        else:
            orders[order.order_id] = _event_order_update(
                order,
                event,
                cancel_requested=False,
                cancel_requested_at=None,
            )

    elif kind is ExecutionFactKind.EXPIRED:
        order = _require_order(orders, fact.order_id)
        if order.state in _TERMINAL_ORDER_STATES:
            raise ExecutionContractError("terminal order cannot expire")
        orders[order.order_id] = _event_order_update(order, event, state=OrderState.EXPIRED)
        reconciliation_required.discard(order.order_id)
        if (
            order.role is OrderRole.STOP
            and position.current_stop is not None
            and order.stop_price == position.current_stop
        ):
            position = replace(
                position,
                state=PositionState.OPEN,
                current_stop=None,
                pending_protection_order_id=None,
                pending_protection_fingerprint=None,
                pending_protection_requested_at=None,
            )
        elif (
            order.role is OrderRole.STOP
            and position.pending_protection_order_id == order.order_id
        ):
            position = replace(
                position,
                pending_protection_order_id=None,
                pending_protection_fingerprint=None,
                pending_protection_requested_at=None,
            )
        if order.role is OrderRole.EXIT and position.pending_exit_order_id == order.order_id:
            position = replace(
                position,
                pending_exit_order_id=None,
                pending_exit_quantity=None,
                pending_exit_requested_at=None,
            )

    elif kind is ExecutionFactKind.REPLACE_REQUESTED:
        old = _require_order(orders, fact.order_id)
        assert fact.related_order_id is not None and fact.quantity is not None
        if (
            old.state in _TERMINAL_ORDER_STATES
            or fact.related_order_id in orders
            or old.pending_replacement_order_id is not None
            or any(
                item.pending_replacement_order_id == fact.related_order_id
                for item in orders.values()
            )
            or position.pending_exit_order_id == fact.related_order_id
        ):
            raise ExecutionContractError("replace request is illegal or duplicated")
        if old.cancel_requested:
            raise ExecutionContractError("cancel-pending order cannot be replaced")
        if old.role is OrderRole.ENTRY and (
            event.event_time >= approved.approval.expires_at
            or event.known_at >= approved.approval.expires_at
            or event.event_time >= approved.intent.expires_at
            or event.known_at >= approved.intent.expires_at
        ):
            raise ExecutionContractError("entry replacement is stale")
        if (
            fact.role is not old.role
            or fact.side is not old.side
            or fact.parent_order_id != old.parent_order_id
            or fact.oco_group_id != old.oco_group_id
            or fact.quantity > old.remaining_quantity
        ):
            raise ExecutionContractError("replacement changes immutable order lineage")
        _validate_intent_order_prices(fact, approved)
        if old.role is OrderRole.STOP and position.current_stop is not None:
            assert fact.stop_price is not None
            loosens = (
                fact.stop_price < position.current_stop - 1e-9
                if approved.intent.side is Direction.LONG
                else fact.stop_price > position.current_stop + 1e-9
            )
            if loosens:
                raise ExecutionContractError(
                    "replacement cannot loosen managed protection"
                )
        replacement_digest = _replacement_fingerprint(
            old_order_id=old.order_id,
            new_order_id=fact.related_order_id,
            fact=fact,
        )
        orders[old.order_id] = _event_order_update(
            old,
            event,
            pending_replacement_order_id=fact.related_order_id,
            pending_replacement_fingerprint=replacement_digest,
            pending_replacement_requested_at=event.event_time,
        )

    elif kind is ExecutionFactKind.REPLACED:
        assert fact.related_order_id is not None and fact.quantity is not None
        old = _require_order(orders, fact.related_order_id)
        if fact.order_id in orders or old.state in _TERMINAL_ORDER_STATES:
            raise ExecutionContractError("replacement acknowledgement is illegal")
        if old.cancel_requested or fact.quantity > old.remaining_quantity:
            raise ExecutionContractError(
                "replacement lost a replace/fill race"
            )
        if (
            old.pending_replacement_requested_at is None
            or event.event_time < old.pending_replacement_requested_at
        ):
            raise ExecutionContractError(
                "replacement acknowledgement predates replace request"
            )
        replacement_digest = _replacement_fingerprint(
            old_order_id=old.order_id,
            new_order_id=fact.order_id,
            fact=fact,
        )
        if (
            old.pending_replacement_order_id != fact.order_id
            or old.pending_replacement_fingerprint != replacement_digest
        ):
            raise ExecutionContractError("replacement fact disagrees with command")
        orders[old.order_id] = _event_order_update(
            old,
            event,
            state=OrderState.CANCELLED,
            replaced_by_order_id=fact.order_id,
            pending_replacement_order_id=None,
            pending_replacement_fingerprint=None,
            pending_replacement_requested_at=None,
        )
        assert fact.role is not None and fact.side is not None and fact.order_type is not None
        orders[fact.order_id] = OrderSnapshot(
            order_id=fact.order_id,
            submitted_at=event.event_time,
            role=fact.role,
            side=fact.side,
            order_type=fact.order_type,
            state=OrderState.WORKING,
            original_quantity=fact.quantity,
            filled_quantity=0,
            average_fill_price=None,
            entry_method=fact.entry_method,
            limit_price=fact.limit_price,
            stop_price=fact.stop_price,
            parent_order_id=fact.parent_order_id,
            oco_group_id=fact.oco_group_id,
            market_reference_price=None,
            replaces_order_id=old.order_id,
            last_event_id=event.event_id,
            last_vendor_sequence=event.vendor_sequence,
        )
        if old.order_id in reconciliation_required:
            reconciliation_required.remove(old.order_id)
            reconciliation_required.add(fact.order_id)
        if (
            old.role is OrderRole.STOP
            and position.current_stop is not None
            and old.stop_price is not None
            and math.isclose(
                position.current_stop,
                old.stop_price,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            # The replacement acknowledgement proves the new order is live,
            # but it carries no structural-source fact authorizing protection.
            # Clear the old managed binding atomically; a new explicit
            # Protect/Protected pair can bind the tighter stop.
            position = replace(
                position,
                state=PositionState.OPEN,
                current_stop=None,
                pending_protection_order_id=None,
                pending_protection_fingerprint=None,
                pending_protection_requested_at=None,
            )
        elif (
            old.role is OrderRole.STOP
            and position.pending_protection_order_id == old.order_id
        ):
            position = replace(
                position,
                pending_protection_order_id=None,
                pending_protection_fingerprint=None,
                pending_protection_requested_at=None,
            )
        if old.role is OrderRole.EXIT and position.pending_exit_order_id == old.order_id:
            position = replace(
                position,
                pending_exit_order_id=fact.order_id,
                pending_exit_quantity=fact.quantity,
            )

    elif kind is ExecutionFactKind.REPLACE_REJECTED:
        old = _require_order(orders, fact.order_id)
        if (
            old.pending_replacement_order_id is None
            or fact.related_order_id != old.pending_replacement_order_id
        ):
            raise ExecutionContractError("replace rejection lacks exact request")
        if (
            old.pending_replacement_requested_at is None
            or event.event_time < old.pending_replacement_requested_at
        ):
            raise ExecutionContractError("replace rejection predates replace request")
        orders[old.order_id] = _event_order_update(
            old,
            event,
            pending_replacement_order_id=None,
            pending_replacement_fingerprint=None,
            pending_replacement_requested_at=None,
        )

    elif kind is ExecutionFactKind.PROTECT_REQUESTED:
        order = _require_order(orders, fact.order_id)
        if order.role is not OrderRole.STOP or order.state not in {OrderState.WORKING, OrderState.PARTIALLY_FILLED} or position.quantity <= 0:
            raise ExecutionContractError("protection request requires working stop and position")
        if _active_reservations(
            orders.values(),
            roles={OrderRole.ENTRY},
        ) > 0:
            raise ExecutionContractError(
                "protection requires terminal entry parent"
            )
        if order.remaining_quantity < position.quantity or fact.stop_price != order.stop_price:
            raise ExecutionContractError("protective stop does not cover exact position")
        if fact.source_level_event_id not in event.source_event_ids:
            raise ExecutionContractError(
                "protection source is absent from immutable provenance"
            )
        if position.pending_protection_fingerprint is not None:
            raise ExecutionContractError("protection request already pending")
        position = replace(
            position,
            pending_protection_order_id=order.order_id,
            pending_protection_fingerprint=_protection_fingerprint(fact),
            pending_protection_requested_at=event.event_time,
        )

    elif kind is ExecutionFactKind.PROTECTED:
        order = _require_order(orders, fact.order_id)
        expected = _protection_fingerprint(fact)
        if (
            position.pending_protection_order_id != order.order_id
            or position.pending_protection_fingerprint != expected
            or order.role is not OrderRole.STOP
            or position.quantity <= 0
        ):
            raise ExecutionContractError("protection fact lacks exact request")
        if (
            position.pending_protection_requested_at is None
            or event.event_time < position.pending_protection_requested_at
        ):
            raise ExecutionContractError("protection fact predates protect request")
        if fact.source_level_event_id not in event.source_event_ids:
            raise ExecutionContractError(
                "protection source is absent from immutable provenance"
            )
        assert fact.stop_price is not None
        previous = (
            approved.intent.invalidation.price
            if position.current_stop is None
            else position.current_stop
        )
        tightens = (
            fact.stop_price > previous
            if approved.intent.side is Direction.LONG
            else fact.stop_price < previous
        )
        if not tightens:
            raise ExecutionContractError("protection can only tighten structural risk")
        position = replace(
            position,
            state=PositionState.MANAGED,
            current_stop=float(fact.stop_price),
            pending_protection_order_id=None,
            pending_protection_fingerprint=None,
            pending_protection_requested_at=None,
        )

    elif kind is ExecutionFactKind.EXIT_REQUESTED:
        assert fact.quantity is not None
        if (
            position.state not in {PositionState.OPEN, PositionState.MANAGED}
            or fact.quantity > position.quantity
            or position.pending_exit_order_id is not None
            or fact.order_id in orders
            or any(
                item.pending_replacement_order_id == fact.order_id
                for item in orders.values()
            )
            or _active_reservations(
                orders.values(),
                roles={OrderRole.ENTRY},
            )
            > 0
            or _close_reserved_quantity(
                orders.values(),
                additional=(OrderRole.EXIT, fact.quantity, None),
            )
            > position.quantity
        ):
            raise ExecutionContractError("exit request is invalid or duplicated")
        position = replace(
            position,
            pending_exit_order_id=fact.order_id,
            pending_exit_quantity=fact.quantity,
            pending_exit_requested_at=event.event_time,
        )

    else:  # pragma: no cover - exhaustive enum guard
        raise ExecutionContractError(f"unsupported execution fact: {kind}")

    result = ExecutionState(
        approved_intent_id=state.approved_intent_id,
        orders=_ordered(orders.values()),
        position=position,
        reconciliation_required_order_ids=tuple(
            sorted(reconciliation_required)
        ),
        execution_anomalies=tuple(anomalies),
        events_applied=state.events_applied + 1,
        last_event_id=event.event_id,
        last_vendor_sequence=event.vendor_sequence,
        last_known_at=event.known_at,
    )
    _validate_state(result, approved)
    return result


def event_order_key(event: ExecutionEventEnvelope) -> tuple[int, pd.Timestamp, str]:
    """Vendor sequence resolves races; known-at and ID are deterministic ties."""

    return (event.vendor_sequence, event.known_at, event.event_id)


@dataclass(frozen=True)
class ExecutionCheckpoint:
    protocol_version: str
    protocol_fingerprint: str
    approved_intent_id: str
    event_count: int
    last_event_id: str | None
    last_vendor_sequence: int | None
    last_known_at: pd.Timestamp | None
    event_fingerprint: str
    state_fingerprint: str


class ImmutableExecutionEventStore:
    """Append-only, approval-bound execution log with atomic reduction."""

    def __init__(self, approved: RiskApprovedTradeIntent) -> None:
        if not isinstance(approved, RiskApprovedTradeIntent):
            raise TypeError("execution store requires RiskApprovedTradeIntent")
        self.approved = approved
        self._events: list[ExecutionEventEnvelope] = []
        self._by_id: dict[str, ExecutionEventEnvelope] = {}
        self._digests: dict[str, str] = {}
        self._by_sequence: dict[int, str] = {}
        self._command_fingerprints: dict[str, str] = {}
        self._state = _initial_state(approved)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> tuple[ExecutionEventEnvelope, ...]:
        return tuple(self._events)

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def event_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"smc-execution-event-store-v1\0")
        digest.update(EXECUTION_PROTOCOL_VERSION.encode("utf-8"))
        digest.update(b"\0")
        digest.update(EXECUTION_PROTOCOL_FINGERPRINT.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.approved.approved_intent_id.encode("utf-8"))
        for event in self._events:
            digest.update(b"\0")
            digest.update(self._digests[event.event_id].encode("ascii"))
        return digest.hexdigest()

    @property
    def state_fingerprint(self) -> str:
        return _digest(self._state)

    def get(self, event_id: str) -> ExecutionEventEnvelope | None:
        return self._by_id.get(event_id)

    def append(self, event: ExecutionEventEnvelope) -> bool:
        if not isinstance(event, ExecutionEventEnvelope):
            raise TypeError("execution store accepts only ExecutionEventEnvelope")
        digest = _digest(event)
        previous = self._by_id.get(event.event_id)
        if previous is not None:
            if self._digests[event.event_id] != digest:
                raise ExecutionContractError(
                    "execution event ID conflicts with immutable history"
                )
            return False
        if event.fact.approved_intent_id != self.approved.approved_intent_id:
            raise ExecutionContractError("event is bound to another approval")
        command_id = event.command_id
        command_fingerprint = event.command_fingerprint
        if command_id is not None:
            assert command_fingerprint is not None
            previous_command_fingerprint = self._command_fingerprints.get(
                command_id
            )
            if (
                previous_command_fingerprint is not None
                and previous_command_fingerprint != command_fingerprint
            ):
                raise ExecutionContractError(
                    "command ID conflicts with immutable command fingerprint"
                )
        if event.fact.kind in _COMMAND_FACT_KINDS and not {
            self.approved.approved_intent_id,
            self.approved.approval.approval_id,
            event.command_id,
        }.issubset(set(event.source_event_ids)):
            raise ExecutionContractError(
                "request event omits approval or command provenance"
            )
        if event.vendor_sequence in self._by_sequence:
            raise ExecutionContractError("vendor sequence is reused by another event")
        if self._events:
            prior = self._events[-1]
            if event.vendor_sequence <= prior.vendor_sequence:
                raise ExecutionContractError("execution event is out of vendor sequence")
            if event.known_at < prior.known_at:
                raise ExecutionContractError("execution event is out of known-at order")
        # Reduce before committing so an illegal fact cannot partially mutate history.
        next_state = reduce_execution_event(self._state, event, self.approved)
        self._events.append(event)
        self._by_id[event.event_id] = event
        self._digests[event.event_id] = digest
        self._by_sequence[event.vendor_sequence] = event.event_id
        if command_id is not None:
            assert command_fingerprint is not None
            self._command_fingerprints[command_id] = command_fingerprint
        self._state = next_state
        return True

    def append_batch(self, events: Iterable[ExecutionEventEnvelope]) -> int:
        incoming = tuple(events)
        clone = ImmutableExecutionEventStore(self.approved)
        for prior in self._events:
            clone.append(prior)
        appended = 0
        for event in incoming:
            appended += int(clone.append(event))
        self._events = list(clone._events)
        self._by_id = dict(clone._by_id)
        self._digests = dict(clone._digests)
        self._by_sequence = dict(clone._by_sequence)
        self._command_fingerprints = dict(clone._command_fingerprints)
        self._state = clone._state
        return appended

    def checkpoint(self) -> ExecutionCheckpoint:
        return ExecutionCheckpoint(
            protocol_version=EXECUTION_PROTOCOL_VERSION,
            protocol_fingerprint=EXECUTION_PROTOCOL_FINGERPRINT,
            approved_intent_id=self.approved.approved_intent_id,
            event_count=len(self),
            last_event_id=self._state.last_event_id,
            last_vendor_sequence=self._state.last_vendor_sequence,
            last_known_at=self._state.last_known_at,
            event_fingerprint=self.event_fingerprint,
            state_fingerprint=self.state_fingerprint,
        )

    def require_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        if not isinstance(checkpoint, ExecutionCheckpoint) or checkpoint != self.checkpoint():
            raise ExecutionContractError("execution checkpoint fingerprint drifted")

    @classmethod
    def replay(
        cls,
        approved: RiskApprovedTradeIntent,
        events: Iterable[ExecutionEventEnvelope],
        *,
        checkpoint: ExecutionCheckpoint | None = None,
    ) -> "ImmutableExecutionEventStore":
        store = cls(approved)
        for event in sorted(tuple(events), key=event_order_key):
            store.append(event)
        if checkpoint is not None:
            store.require_checkpoint(checkpoint)
        return store


class ExecutionFSM:
    """Small shadow adapter: commands become requests; facts drive state."""

    def __init__(self, approved: RiskApprovedTradeIntent) -> None:
        self.store = ImmutableExecutionEventStore(approved)

    @property
    def approved(self) -> RiskApprovedTradeIntent:
        return self.store.approved

    @property
    def state(self) -> ExecutionState:
        return self.store.state

    def record(self, event: ExecutionEventEnvelope) -> bool:
        return self.store.append(event)

    def accept_command(
        self,
        command: ExecutionCommand,
        *,
        known_at: pd.Timestamp,
        vendor_sequence: int,
        top_of_book: TopOfBook | None = None,
    ) -> ExecutionEventEnvelope:
        """Audit a shadow command without transmitting it anywhere."""

        if not isinstance(command, ExecutionCommand):
            raise TypeError("ExecutionFSM accepts only typed execution commands")
        if command.approved_intent_id != self.approved.approved_intent_id:
            raise ExecutionContractError("command belongs to another approved intent")
        if isinstance(command, SubmitOrderCommand):
            if top_of_book is None:
                raise ExecutionContractError("submission requires causal TopOfBook")
            if top_of_book.observed_at > command.created_at:
                raise ExecutionContractError("submission uses future TopOfBook")
            fact = ExecutionFact(
                kind=ExecutionFactKind.SUBMITTED,
                approved_intent_id=command.approved_intent_id,
                order_id=command.order_id,
                role=command.role,
                side=command.side,
                order_type=command.order_type,
                quantity=command.quantity,
                entry_method=command.entry_method,
                limit_price=command.limit_price,
                stop_price=command.stop_price,
                parent_order_id=command.parent_order_id,
                oco_group_id=command.oco_group_id,
                top_of_book=top_of_book,
            )
        elif isinstance(command, CancelOrderCommand):
            if top_of_book is not None:
                raise ExecutionContractError("cancel command cannot carry TopOfBook")
            fact = ExecutionFact(
                kind=ExecutionFactKind.CANCEL_REQUESTED,
                approved_intent_id=command.approved_intent_id,
                order_id=command.order_id,
            )
        elif isinstance(command, ReplaceOrderCommand):
            if top_of_book is not None:
                raise ExecutionContractError("replace request cannot carry TopOfBook")
            fact = ExecutionFact(
                kind=ExecutionFactKind.REPLACE_REQUESTED,
                approved_intent_id=command.approved_intent_id,
                order_id=command.order_id,
                related_order_id=command.replacement_order_id,
                role=command.role,
                side=command.side,
                order_type=command.order_type,
                quantity=command.quantity,
                entry_method=command.entry_method,
                parent_order_id=command.parent_order_id,
                oco_group_id=command.oco_group_id,
                limit_price=command.limit_price,
                stop_price=command.stop_price,
            )
        elif isinstance(command, ProtectPositionCommand):
            if top_of_book is not None:
                raise ExecutionContractError("protect request cannot carry TopOfBook")
            fact = ExecutionFact(
                kind=ExecutionFactKind.PROTECT_REQUESTED,
                approved_intent_id=command.approved_intent_id,
                order_id=command.stop_order_id,
                stop_price=command.new_stop_price,
                source_level_event_id=command.source_level_event_id,
            )
        elif isinstance(command, ExitPositionCommand):
            if top_of_book is not None:
                raise ExecutionContractError("exit request cannot carry TopOfBook")
            fact = ExecutionFact(
                kind=ExecutionFactKind.EXIT_REQUESTED,
                approved_intent_id=command.approved_intent_id,
                order_id=command.exit_order_id,
                quantity=command.quantity,
            )
        else:  # pragma: no cover - subclass exhaustiveness guard
            raise TypeError("unsupported execution command")
        sources = tuple(
            sorted(
                set(
                    (
                        *command.source_event_ids,
                        command.command_id,
                        self.approved.approval.approval_id,
                        self.approved.approved_intent_id,
                    )
                )
            )
        )
        event = make_execution_event(
            fact,
            event_time=command.created_at,
            known_at=known_at,
            source_event_ids=sources,
            vendor_sequence=vendor_sequence,
            source_command=command,
        )
        self.store.append(event)
        return event


__all__ = [
    "CancelOrderCommand",
    "EXECUTION_AUTHORITY",
    "EXECUTION_PROTOCOL_FINGERPRINT",
    "EXECUTION_PROTOCOL_VERSION",
    "ExecutionCheckpoint",
    "ExecutionCommand",
    "ExecutionContractError",
    "ExecutionEvent",
    "ExecutionEventEnvelope",
    "ExecutionFSM",
    "ExecutionFact",
    "ExecutionFactKind",
    "ExecutionPositionSnapshot",
    "ExecutionState",
    "ExitPositionCommand",
    "ImmutableExecutionEventStore",
    "OrderRole",
    "OrderSide",
    "OrderSnapshot",
    "OrderState",
    "OrderType",
    "PositionState",
    "ProtectPositionCommand",
    "ReplaceOrderCommand",
    "RiskApproval",
    "RiskApprovedTradeIntent",
    "SubmitOrderCommand",
    "account_state_fingerprint",
    "event_order_key",
    "make_execution_event",
    "reduce_execution_event",
    "risk_approve_trade_intent",
]
