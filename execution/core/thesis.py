"""The ``ThesisBook``: one record per thesis id per episode, and the rules
that stop one reading from being re-expressed through swapped objects.

A thesis (the Brain's ``thesis_id``) is OPEN from its first proposal.  One
expression (a working entry or an open position) at a time; at most
``max_expressions`` orders in the episode; a stop-out or a reached target
closes it for the episode (``stopped`` / ``achieved``), a flipped direction
closes it (``direction_changed``), a flatten because the Brain's bias
turned against the position closes it (``bias_reversed``, no cooldown),
and every stop-out — whatever the thesis — holds every new expression for
``stop_cooldown_bars`` 1m bars.  An expiry or a cancel leaves the thesis
open (the entry was never reached); an expiry gives its expression back
(2026-09-19), and so does a cancel that replaced the entry object or lost
it to the Eye (2026-09-20); a dropped plan spends it.
The book resets with the episode; ``view`` is what the LLM reads in
``prior_state.execution``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from contract.brain.state import TradeDirection
from contract.risk import TradePlan
from risk.core.gate import ThesisConfig

REFUSALS: tuple[str, ...] = ("direction_changed", "thesis_closed", "expressions_exhausted", "stop_cooldown", "thesis_engaged", "entry_marketable")
# Cancel reasons that replace an expression rather than spend it (2026-09-20): the Brain moved the entry to
# another object, or the Eye retired the object.  A dropped plan (``plan_dropped``) is the churn the cap counts.
REPLACEMENT_REASONS: frozenset[str] = frozenset({"signature_changed", "entry_object_not_visible"})


@dataclass
class ThesisRecord:
    thesis_id: str
    direction: TradeDirection
    governing_timeframe: str
    opened_at: pd.Timestamp
    expressions: int = 0
    closed_reason: str | None = None
    last_outcome: str | None = None

    @property
    def status(self) -> str:
        return "OPEN" if self.closed_reason is None else "CLOSED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis_id": self.thesis_id,
            "direction": self.direction.value,
            "governing_timeframe": self.governing_timeframe,
            "status": self.status,
            "closed_reason": self.closed_reason,
            "expressions": self.expressions,
            "last_outcome": self.last_outcome,
        }


class ThesisBook:
    def __init__(self, config: ThesisConfig) -> None:
        self._config = config
        self._episode: str | None = None
        self._records: dict[str, ThesisRecord] = {}
        self._engaged: dict[str, str] = {}  # signature → thesis id of a working or open expression
        self._last_stop_bar: int | None = None

    def start_episode(self, episode_id: str | None) -> None:
        if episode_id == self._episode:
            return
        self._episode = episode_id
        self._records = {}
        self._engaged = {}
        self._last_stop_bar = None

    # ------------------------------------------------------------ rules

    def cooldown_bars_left(self, bar_index: int) -> int:
        if self._last_stop_bar is None:
            return 0
        return max(0, self._config.stop_cooldown_bars - (bar_index - self._last_stop_bar))

    def admit(self, plan: TradePlan, bar_index: int) -> str | None:
        """``None`` when the plan may be expressed now, else the refusal, in
        the order of the thesis's own state first and the cooldown last.  A
        plan without a thesis id is only held by the cooldown."""
        record = self._records.get(plan.thesis_id) if plan.thesis_id else None
        if plan.thesis_id and record is None:
            record = ThesisRecord(plan.thesis_id, plan.direction, plan.governing_timeframe, plan.known_at)
            self._records[plan.thesis_id] = record
        if record is not None:
            if record.direction is not plan.direction:
                if record.closed_reason is None:
                    record.closed_reason = "direction_changed"
                return "direction_changed"
            if record.closed_reason is not None:
                return "thesis_closed"
            if record.expressions >= self._config.max_expressions:
                record.closed_reason = "expressions_exhausted"
                return "expressions_exhausted"
            if plan.thesis_id in self._engaged.values():
                return "thesis_engaged"
        if self.cooldown_bars_left(bar_index) > 0:
            return "stop_cooldown"
        return None

    def expressed(self, plan: TradePlan) -> None:
        """An order was submitted for the plan: count it and mark the
        thesis engaged until ``engaged(signature, False)`` or ``outcome``."""
        self._engaged[plan.signature] = plan.thesis_id
        record = self._records.get(plan.thesis_id) if plan.thesis_id else None
        if record is not None:
            record.expressions += 1

    def engaged(self, signature: str, engaged: bool) -> None:
        """Release (or re-affirm) a signature's engagement."""
        if not engaged:
            self._engaged.pop(signature, None)

    def outcome(self, plan: TradePlan, kind: str, *, exit_role: str | None, bar_index: int, reason: str | None = None) -> None:
        """A terminal event of an expression: a stop (or the close-beyond
        ``invalidation`` exit) closes the thesis and starts the cooldown, a
        target closes it as achieved, the bias-reversal flatten closes it
        without a cooldown, anything else (expired, cancelled, rejected, the
        halt's flatten) leaves it open.  An expiry gives its expression back
        (the entry was never reached), and so does a cancel whose ``reason``
        is a replacement (``REPLACEMENT_REASONS``); a dropped plan keeps it
        (the Brain changed its mind)."""
        self._engaged.pop(plan.signature, None)
        record = self._records.get(plan.thesis_id) if plan.thesis_id else None
        if record is not None:
            record.last_outcome = kind if exit_role is None else f"{kind}:{exit_role}"
            if kind == "expired" or (kind == "cancelled" and reason in REPLACEMENT_REASONS):
                record.expressions = max(0, record.expressions - 1)
        if exit_role in ("stop", "invalidation"):
            self._last_stop_bar = bar_index
            if record is not None and record.closed_reason is None:
                record.closed_reason = "stopped"
        elif exit_role == "target" and record is not None and record.closed_reason is None:
            record.closed_reason = "achieved"
        elif exit_role == "bias_reversed" and record is not None and record.closed_reason is None:
            record.closed_reason = "bias_reversed"

    # ------------------------------------------------------------ the view

    def view(self, bar_index: int) -> dict[str, Any]:
        return {
            "theses": [self._records[key].to_dict() for key in self._records],
            "cooldown_bars_left": self.cooldown_bars_left(bar_index),
        }


__all__ = ["REFUSALS", "REPLACEMENT_REASONS", "ThesisBook", "ThesisRecord"]
