"""The order machine: several intents per episode, one bracket per intent.

```text
per intent:  WORKING ──entry fill──► IN_POSITION ──stop / target / flatten──► done
             WORKING ──partial fill──► PARTIAL ──rest fills──► IN_POSITION
             WORKING ──cancel / expire / reject──► done
machine:     WORKING if an entry works, else IN_POSITION if a position is open, else IDLE
```

The identity of an intent is the plan's ``signature`` (direction and the
three Eye entity ids).  One entry works at a time: a plan whose signature
matches a working entry is a no-op, a different one, a dropped plan, an
entry object the Eye no longer shows, or ``order_ttl_bars`` without a fill
cancels it.  A plan whose signature matches an open position describes it;
a new signature while positions are open goes to the ``ThesisBook`` and the
Risk gate (which counts the machine's positions, up to
``max_open_positions``, all in one direction).  A signature that expired,
was rejected, or whose position closed is not resubmitted in the episode.
After any position closes nothing is submitted until the Brain has been
called again (``llm_called``), so no order is placed on a plan chosen
before the outcome was known.

The ``ThesisBook`` refuses a re-expression of a closed thesis, a second
expression while one works, more than ``max_expressions`` per thesis and
any expression during the ``stop_cooldown_bars`` after a stop-out; every
refusal is journaled ``thesis_refused`` on the first bar of a (thesis,
reason) pair and on every bar on which the LLM proposed it again.  A
refused plan at the gate is journaled ``veto`` the same way.

Two exits happen at market through ``Broker.flatten``: the close-beyond
exit (a position whose plan is ``CLOSE_BEYOND`` when a bar of the
invalidation object's scale closes beyond the object's far edge;
``invalidation_close``, then ``position_closed`` with ``exit_role``
``invalidation``) and the halt (the gate's drawdown stop: every working
entry and every exit leg cancelled, every position flattened, ``halted``
journaled, nothing ever submitted again).  A stop or target the broker
cancels or rejects on its own is journaled ``exit_leg_lost``.

Every transition is a journal ``trade`` record; ``ExecutionLedger`` is what
the Brain reads: engaged (a position or a working order) means it cannot
sleep, and ``execution_view()`` is what the LLM is told."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from brain.core.journal import BrainJournal
from brain.core.position_ledger import IDLE_VIEW, PositionRecord
from contract.brain.state import InvalidationMode, TradeDirection, isoformat_utc
from contract.execution import BracketIntent, BrokerEvent, OrderRole, OrderState
from contract.market.primitives import Bar
from contract.risk import RiskVerdict, TradePlan
from execution.core.broker import Broker
from execution.core.thesis import ThesisBook
from risk.core.gate import RiskGate

STAT_KINDS: tuple[str, ...] = (
    "veto_bars", "veto", "thesis_refused", "submitted", "working", "partial", "filled", "position_opened", "position_closed",
    "cancel_requested", "cancelled", "expired", "rejected", "exit_leg_lost", "invalidation_close", "flattened", "halted",
)
EXIT_INVALIDATION = "invalidation"
EXIT_HALT = "flatten"


class MachineState(str, Enum):
    IDLE = "IDLE"
    WORKING = "WORKING"
    PARTIAL = "PARTIAL"
    IN_POSITION = "IN_POSITION"


@dataclass
class _Intent:
    plan: TradePlan
    verdict: RiskVerdict
    entry: OrderState
    submitted_at: pd.Timestamp
    bars_since_submit: int = 0
    cancel_reason: str | None = None
    position: PositionRecord | None = None
    lost_legs: set[str] = field(default_factory=set)
    exits: dict[str, OrderState] = field(default_factory=dict)  # "stop" / "target" → latest state
    exit_requested: str | None = None  # EXIT_INVALIDATION / EXIT_HALT once a flatten was sent

    @property
    def working(self) -> bool:
        return self.entry.is_open

    @property
    def open_quantity(self) -> int:
        return self.entry.filled_quantity


@dataclass
class _VetoMemory:
    """The last refused plan of the episode, as the LLM is told about it."""

    signature: str
    plan: TradePlan
    vetoes: tuple[str, ...]
    reasons: tuple[str, ...]
    first_at: pd.Timestamp
    last_at: pd.Timestamp
    bars: int = 0
    proposals: int = 0

    def to_dict(self) -> dict[str, Any]:
        plan = self.plan
        return {
            "direction": plan.direction.value,
            "entry_object_id": plan.entry.alias,
            "invalidation_object_id": plan.invalidation.alias,
            "target_object_id": plan.target.alias,
            "thesis_id": plan.thesis_id,
            "vetoes": list(self.vetoes),
            "reasons": list(self.reasons),
            "reward_risk": round(float(plan.geometry.reward_risk), 2),
            "first_vetoed_at": isoformat_utc(self.first_at),
            "last_vetoed_at": isoformat_utc(self.last_at),
            "bars_vetoed": self.bars,
            "proposals_vetoed": self.proposals,
        }


class ExecutionLedger:
    """The Brain's ``PositionLedger``, read from the machine's state."""

    def __init__(self, machine: "OrderMachine") -> None:
        self._machine = machine

    def has_open_position(self) -> bool:
        return bool(self._machine.positions())

    def has_working_order(self) -> bool:
        return self._machine.state in (MachineState.WORKING, MachineState.PARTIAL)

    def open_positions(self) -> tuple[PositionRecord, ...]:
        return self._machine.positions()

    def execution_view(self) -> Mapping[str, Any]:
        return self._machine.execution_view()


class OrderMachine:
    def __init__(self, broker: Broker, gate: RiskGate, *, journal: BrainJournal | None = None) -> None:
        self._broker = broker
        self._gate = gate
        self._journal = journal
        self._intents: dict[str, _Intent] = {}  # by signature
        self._working: str | None = None  # the signature whose entry works
        # (episode id, signature) of the intents that expired, were rejected or
        # closed their position: not resubmitted in that episode.
        self._blocked: set[tuple[str, str]] = set()
        # Per-episode memory: the (signature, vetoes) and (thesis, refusal) pairs
        # already journaled, the last veto and the last terminal outcome.
        self._episode: str | None = None
        self._bar_index = 0
        self._asof: pd.Timestamp | None = None
        self._vetoed: set[tuple[str, tuple[str, ...]]] = set()
        self._refused: set[tuple[str, str]] = set()
        self._veto: _VetoMemory | None = None
        self._last_outcome: dict[str, Any] | None = None
        # Set when a position closes: nothing is submitted until the Brain has
        # been called with the outcome in front of it (``llm_called``).
        self._await_brain = False
        self._halted: dict[str, Any] | None = None
        self._book = ThesisBook(gate.config.thesis)
        self.stats: dict[str, int] = {kind: 0 for kind in STAT_KINDS}
        self.ledger = ExecutionLedger(self)

    # ------------------------------------------------------------ state

    @property
    def state(self) -> MachineState:
        working = self._intents.get(self._working or "")
        if working is not None and working.working:
            return MachineState.PARTIAL if working.position is not None else MachineState.WORKING
        if any(intent.position is not None for intent in self._intents.values()):
            return MachineState.IN_POSITION
        return MachineState.IDLE

    @property
    def working_signature(self) -> str | None:
        working = self._intents.get(self._working or "")
        return None if working is None or not working.working else working.plan.signature

    @property
    def halted(self) -> bool:
        return self._halted is not None

    @property
    def halt_record(self) -> dict[str, Any] | None:
        return None if self._halted is None else dict(self._halted)

    def positions(self) -> tuple[PositionRecord, ...]:
        return tuple(intent.position for intent in self._intents.values() if intent.position is not None)

    # ------------------------------------------------------------ the view

    def execution_view(self) -> Mapping[str, Any]:
        book = self._book.view(self._bar_index)
        daily_stop = self._asof is not None and self._gate.daily_stopped(self._asof)
        if (
            not self._intents and self._veto is None and self._last_outcome is None and not book["theses"]
            and book["cooldown_bars_left"] == 0 and not daily_stop and self._halted is None
        ):
            return IDLE_VIEW
        order = None
        working = self._intents.get(self._working or "")
        if working is not None and working.working:
            plan = working.plan
            order = {
                "direction": plan.direction.value,
                "entry_object_id": plan.entry.alias,
                "invalidation_object_id": plan.invalidation.alias,
                "target_object_id": plan.target.alias,
                "thesis_id": plan.thesis_id,
                "submitted_at": isoformat_utc(working.submitted_at),
                "bars_working": working.bars_since_submit,
                "quantity": working.entry.quantity,
                "filled_quantity": working.entry.filled_quantity,
                "ttl_bars": self._gate.config.order_ttl_bars,
            }
        positions = [
            {
                "direction": intent.position.direction.value,
                "entry_object_id": intent.position.entry_object_id,
                "opened_at": isoformat_utc(intent.position.opened_at),
                "quantity": intent.entry.filled_quantity,
                "thesis_id": intent.plan.thesis_id,
                "invalidation_mode": intent.plan.invalidation_mode.value,
            }
            for intent in self._intents.values()
            if intent.position is not None
        ]
        return {
            "status": self.state.value,
            "order": order,
            "positions": positions,
            "theses": book["theses"],
            "cooldown_bars_left": book["cooldown_bars_left"],
            "daily_stop": bool(daily_stop),
            "halted": self._halted is not None,
            "last_outcome": None if self._last_outcome is None else dict(self._last_outcome),
            "last_veto": None if self._veto is None else self._veto.to_dict(),
        }

    def _start_episode(self, episode_id: str | None) -> None:
        if episode_id == self._episode:
            return
        self._episode = episode_id
        self._bar_index = 0
        self._vetoed = set()
        self._refused = set()
        self._veto = None
        self._last_outcome = None
        self._await_brain = False
        self._book.start_episode(episode_id)

    # ------------------------------------------------------------ journal

    def _record(self, kinds: list[str], kind: str, episode_id: str | None, asof: pd.Timestamp, payload: dict[str, Any]) -> None:
        kinds.append(kind)
        if kind in self.stats:
            self.stats[kind] += 1
        if self._journal is None or episode_id is None:
            return
        self._journal.write("trade", episode_id=episode_id, known_at=asof, payload={"kind": kind, "machine": self.state.value, **payload})

    def _outcome(self, kind: str, asof: pd.Timestamp, *, reason: str | None = None, exit_role: str | None = None, thesis_id: str = "") -> None:
        self._last_outcome = {"kind": kind, "at": isoformat_utc(asof), "reason": reason, "exit_role": exit_role, "thesis_id": thesis_id}

    # ------------------------------------------------------------ events

    def _intent_for(self, client_ref: str) -> _Intent | None:
        parts = client_ref.split(":")
        return self._intents.get(parts[1]) if len(parts) >= 2 else None

    def _apply(
        self, event: BrokerEvent, kinds: list[str], episode_id: str | None, asof: pd.Timestamp, lost: list[OrderState]
    ) -> None:
        intent = self._intent_for(event.order.client_ref)
        if intent is None:
            return  # not ours (another session's order at the broker)
        order = event.order
        base = {"signature": intent.plan.signature, "thesis_id": intent.plan.thesis_id, "order": order.to_dict(), "fill": None if event.fill is None else event.fill.to_dict()}
        if order.role is OrderRole.ENTRY:
            intent.entry = order
            if event.kind == "working":
                self._record(kinds, "working", episode_id, asof, base)
            elif event.kind == "partial":
                intent.position = self._position(intent, order, asof)
                self._record(kinds, "partial", episode_id, asof, base)
            elif event.kind == "filled":
                intent.position = self._position(intent, order, asof)
                if self._working == intent.plan.signature:
                    self._working = None
                self._record(kinds, "filled", episode_id, asof, base)
                self._record(kinds, "position_opened", episode_id, asof, {**base, "position": intent.position.to_dict()})
            elif event.kind in ("cancelled", "expired", "rejected"):
                kind = "expired" if intent.cancel_reason == "ttl" and event.kind == "cancelled" else event.kind
                if self._working == intent.plan.signature:
                    self._working = None
                if order.filled_quantity > 0:
                    self._record(kinds, kind, episode_id, asof, {**base, "reason": intent.cancel_reason})  # the filled part lives on
                else:
                    self._record(kinds, kind, episode_id, asof, {**base, "reason": intent.cancel_reason})
                    self._outcome(kind, asof, reason=intent.cancel_reason, thesis_id=intent.plan.thesis_id)
                    if kind in ("expired", "rejected"):
                        self._blocked.add((intent.plan.episode_id, intent.plan.signature))
                    self._book.outcome(intent.plan, kind, exit_role=None, bar_index=self._bar_index)
                    del self._intents[intent.plan.signature]
            return
        if order.role is OrderRole.FLATTEN:
            if event.kind == "filled":
                exit_role = EXIT_INVALIDATION if intent.exit_requested == EXIT_INVALIDATION else EXIT_HALT
                self._close(intent, order, event, kinds, episode_id, asof, exit_role=exit_role, base=base)
                self._record(kinds, "flattened", episode_id, asof, {**base, "reason": intent.exit_requested})
            return
        # a stop or a target
        intent.exits[order.role.value] = order
        if event.kind in ("cancelled", "expired", "rejected"):
            if intent.exit_requested is None:
                lost.append(order)  # journaled after the poll unless this batch also closed the position
            return
        if event.kind == "filled":
            self._close(intent, order, event, kinds, episode_id, asof, exit_role=order.role.value, base=base)

    def _close(
        self, intent: _Intent, order: OrderState, event: BrokerEvent, kinds: list[str], episode_id: str | None, asof: pd.Timestamp,
        *, exit_role: str, base: dict[str, Any],
    ) -> None:
        self._record(kinds, "position_closed", episode_id, asof, {
            **base, "exit_role": exit_role, "exit_price": None if event.fill is None else event.fill.price,
            "reason": intent.exit_requested,
            "position": None if intent.position is None else intent.position.to_dict(),
        })
        self._outcome("position_closed", asof, exit_role=exit_role, reason=intent.exit_requested, thesis_id=intent.plan.thesis_id)
        # The plan the Brain still holds is not re-entered on its own: the
        # same signature waits for the episode to end, and any plan waits for
        # the Brain's next call.
        self._blocked.add((intent.plan.episode_id, intent.plan.signature))
        self._book.outcome(intent.plan, "position_closed", exit_role=exit_role, bar_index=self._bar_index)
        self._await_brain = True
        if self._working == intent.plan.signature:
            self._working = None
        del self._intents[intent.plan.signature]

    def _position(self, intent: _Intent, order: OrderState, asof: pd.Timestamp) -> PositionRecord:
        if intent.position is not None:
            return intent.position
        return PositionRecord(
            order.order_id, intent.plan.direction, asof, intent.plan.entry.alias,
            thesis_id=intent.plan.thesis_id, invalidation_mode=intent.plan.invalidation_mode.value,
        )

    # ------------------------------------------------------------ exits at market

    def _flatten(self, intent: _Intent, asof: pd.Timestamp, reason: str, kinds: list[str], episode_id: str | None) -> None:
        """Cancel the exits (and a still-working entry) and send a market
        order for the filled quantity."""
        intent.exit_requested = reason
        if intent.working and intent.cancel_reason is None:
            intent.cancel_reason = reason
            self._broker.cancel(intent.entry.order_id, asof)
            self._record(kinds, "cancel_requested", episode_id, asof, {"signature": intent.plan.signature, "reason": reason, "order_id": intent.entry.order_id})
        for leg in ("stop", "target"):
            order = intent.exits.get(leg)
            if order is not None and order.is_open:
                self._broker.cancel(order.order_id, asof)
                self._record(kinds, "cancel_requested", episode_id, asof, {"signature": intent.plan.signature, "reason": reason, "order_id": order.order_id})
        quantity = intent.open_quantity
        if quantity < 1:
            return
        side = "SELL" if intent.plan.direction is TradeDirection.LONG else "BUY"
        symbol = self._gate.config.contract.symbol
        self._broker.flatten(symbol, quantity, side, asof, f"{intent.entry.client_ref}:{reason}")

    def _halt(self, asof: pd.Timestamp, kinds: list[str], episode_id: str | None) -> None:
        record = self._gate.halt_record or {}
        self._halted = {**record, "positions_flattened": sum(1 for intent in self._intents.values() if intent.open_quantity > 0)}
        for intent in list(self._intents.values()):
            if intent.exit_requested is None:
                self._flatten(intent, asof, EXIT_HALT, kinds, episode_id)
        self._record(kinds, "halted", episode_id, asof, dict(self._halted))

    def _close_beyond(self, bar: Bar, closed_timeframes: frozenset[str], asof: pd.Timestamp, kinds: list[str], episode_id: str | None) -> None:
        for intent in list(self._intents.values()):
            plan = intent.plan
            if (
                intent.position is None or intent.exit_requested is not None or plan.invalidation_mode is not InvalidationMode.CLOSE_BEYOND
                or plan.invalidation_level is None or plan.invalidation.timeframe not in closed_timeframes
            ):
                continue
            close = float(bar.close)
            beyond = close < plan.invalidation_level if plan.direction is TradeDirection.LONG else close > plan.invalidation_level
            if not beyond:
                continue
            self._record(kinds, "invalidation_close", episode_id, asof, {
                "signature": plan.signature, "thesis_id": plan.thesis_id, "timeframe": plan.invalidation.timeframe,
                "close": close, "invalidation_level": plan.invalidation_level, "position": intent.position.to_dict(),
            })
            self._flatten(intent, asof, EXIT_INVALIDATION, kinds, episode_id)

    # ------------------------------------------------------------ step

    def on_bar(
        self,
        asof: pd.Timestamp,
        bar: Bar | None,
        plan: TradePlan | None,
        *,
        episode_id: str | None,
        visible: Callable[[str], bool],
        llm_called: bool = False,
        closed_timeframes: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        asof = pd.Timestamp(asof).tz_convert("UTC")
        self._start_episode(episode_id)
        self._bar_index += 1
        self._asof = asof
        if llm_called:
            self._await_brain = False  # the Brain reasoned on this bar, before the poll below
        kinds: list[str] = []
        lost: list[OrderState] = []
        for event in self._broker.poll(asof, bar):
            self._apply(event, kinds, episode_id, asof, lost)
        for intent in self._intents.values():
            for order in lost:
                if order.client_ref == intent.entry.client_ref and order.order_id not in intent.lost_legs:
                    intent.lost_legs.add(order.order_id)
                    self._record(kinds, "exit_leg_lost", episode_id, asof, {
                        "signature": intent.plan.signature, "exit_role": order.role.value, "order": order.to_dict(),
                        "position": None if intent.position is None else intent.position.to_dict(),
                    })
        account = self._broker.snapshot(asof)
        self._gate.observe(account, asof)
        if self._halted is not None:
            return tuple(kinds)
        if self._gate.halted:
            self._halt(asof, kinds, episode_id)
            return tuple(kinds)
        working = self._intents.get(self._working or "")
        if working is not None and working.working:
            working.bars_since_submit += 1
            if working.cancel_reason is None:
                reason = None
                if plan is None:
                    reason = "plan_dropped"
                elif plan.signature != working.plan.signature:
                    reason = "signature_changed"
                elif not visible(working.plan.entry.alias):
                    reason = "entry_object_not_visible"
                elif working.bars_since_submit >= self._gate.config.order_ttl_bars:
                    reason = "ttl"
                if reason is not None:
                    working.cancel_reason = reason
                    self._broker.cancel(working.entry.order_id, asof)
                    self._record(kinds, "cancel_requested", episode_id, asof, {"signature": working.plan.signature, "reason": reason, "order_id": working.entry.order_id})
        if bar is not None and closed_timeframes:
            self._close_beyond(bar, closed_timeframes, asof, kinds, episode_id)
        if working is not None and working.working:
            return tuple(kinds)
        if plan is None or self._await_brain or plan.signature in self._intents:
            return tuple(kinds)
        if (plan.episode_id, plan.signature) in self._blocked:
            return tuple(kinds)
        if not visible(plan.entry.alias):
            return tuple(kinds)
        refusal = self._book.admit(plan, self._bar_index)
        if refusal is not None:
            self._refuse(plan, refusal, kinds, episode_id, asof, llm_called=llm_called)
            return tuple(kinds)
        verdict = self._gate.assess(plan, account, asof=asof, positions=self.positions())
        if not verdict.passed:
            self._veto_bar(plan, verdict, kinds, episode_id, asof, account_asof=account.asof, llm_called=llm_called)
            return tuple(kinds)
        contract = self._gate.config.contract
        intent_spec = BracketIntent(
            client_ref=f"{plan.episode_id}:{plan.signature}", symbol=contract.symbol,
            side="BUY" if plan.direction is TradeDirection.LONG else "SELL", quantity=verdict.quantity,
            limit_price=float(verdict.limit_price), stop_price=float(verdict.stop_price), target_price=float(verdict.target_price),
            signature=plan.signature,
        )
        entry = self._broker.submit_bracket(intent_spec, asof)
        self._intents[plan.signature] = _Intent(plan, verdict, entry, asof)
        self._working = plan.signature
        self._book.expressed(plan)
        self._veto = None
        self._last_outcome = None
        self._record(kinds, "submitted", episode_id, asof, {
            "signature": plan.signature, "thesis_id": plan.thesis_id, "plan": plan.to_dict(), "verdict": verdict.to_dict(), "intent": intent_spec.to_dict(),
            "order": entry.to_dict(), "account": {"equity": account.equity, "asof": isoformat_utc(account.asof), "account_id": account.account_id},
            "open_positions": len(self.positions()),
        })
        return tuple(kinds)

    def _refuse(self, plan: TradePlan, reason: str, kinds: list[str], episode_id: str | None, asof: pd.Timestamp, *, llm_called: bool) -> None:
        key = (plan.thesis_id or plan.signature, reason)
        if key in self._refused and not llm_called:
            return
        self._refused.add(key)
        self._record(kinds, "thesis_refused", episode_id, asof, {
            "signature": plan.signature, "thesis_id": plan.thesis_id, "reason": reason, "plan": plan.to_dict(),
            "cooldown_bars_left": self._book.cooldown_bars_left(self._bar_index),
        })

    def _veto_bar(
        self, plan: TradePlan, verdict: RiskVerdict, kinds: list[str], episode_id: str | None, asof: pd.Timestamp,
        *, account_asof: pd.Timestamp, llm_called: bool,
    ) -> None:
        self.stats["veto_bars"] += 1
        vetoes = tuple(item.value for item in verdict.vetoes)
        memory = self._veto
        if memory is None or memory.signature != plan.signature or memory.vetoes != vetoes:
            memory = _VetoMemory(plan.signature, plan, vetoes, tuple(verdict.reasons), asof, asof)
            self._veto = memory
        memory.plan = plan
        memory.reasons = tuple(verdict.reasons)
        memory.last_at = asof
        memory.bars += 1
        key = (plan.signature, vetoes)
        if key in self._vetoed and not llm_called:
            return
        self._vetoed.add(key)
        memory.proposals += 1
        self._record(kinds, "veto", episode_id, asof, {
            "signature": plan.signature, "thesis_id": plan.thesis_id, "plan": plan.to_dict(), "verdict": verdict.to_dict(),
            "account_asof": isoformat_utc(account_asof), "proposals_vetoed": memory.proposals, "bars_vetoed": memory.bars,
        })


__all__ = ["EXIT_HALT", "EXIT_INVALIDATION", "STAT_KINDS", "ExecutionLedger", "MachineState", "OrderMachine"]
