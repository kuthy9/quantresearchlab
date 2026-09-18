"""The Brain's view of open positions — the boundary Execution will own.

The reducer reads one fact through it: is the Brain engaged — a position
open, or an entry order working?  Either forbids sleep.  The Main Brain
reads ``execution_view()`` — what the executor did with the last opportunity
(the working order, the position, the last outcome, the last Risk veto) —
and copies it into ``prior_state.execution`` for the LLM.
``InMemoryPositionLedger`` holds what a caller records and nothing else — no
sizing, no stops, no fills, an IDLE view; ``execution.core.order_fsm.ExecutionLedger``
is the real one, read from the order machine."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from contract.brain.state import TradeDirection
from contract.market.primitives import FrozenDict, aware_timestamp


@dataclass(frozen=True)
class PositionRecord:
    position_id: str
    direction: TradeDirection
    opened_at: pd.Timestamp
    entry_object_id: str
    # The thesis the position expresses and how its invalidation object
    # falsifies it (2026-09-17); empty / TOUCH for records before then.
    thesis_id: str = ""
    invalidation_mode: str = "TOUCH"

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", TradeDirection(self.direction))
        object.__setattr__(self, "opened_at", aware_timestamp(self.opened_at, name="position.opened_at"))

    def to_dict(self) -> dict:
        from contract.brain.state import isoformat_utc

        return {
            "position_id": self.position_id,
            "direction": self.direction.value,
            "opened_at": isoformat_utc(self.opened_at),
            "entry_object_id": self.entry_object_id,
            "thesis_id": self.thesis_id,
            "invalidation_mode": self.invalidation_mode,
        }


# What ``execution_view`` reports while nothing is working, open or remembered.
IDLE_VIEW: Mapping[str, Any] = FrozenDict(
    {
        "status": "IDLE", "order": None, "positions": (), "theses": (), "cooldown_bars_left": 0,
        "daily_stop": False, "halted": False, "last_outcome": None, "last_veto": None,
    }
)


class PositionLedger(Protocol):
    def has_open_position(self) -> bool: ...

    def has_working_order(self) -> bool: ...

    def open_positions(self) -> tuple[PositionRecord, ...]: ...

    def execution_view(self) -> Mapping[str, Any]:
        """The executor's state for the LLM: ``status`` (IDLE / WORKING /
        PARTIAL / IN_POSITION), ``order``, ``positions``, ``theses``,
        ``cooldown_bars_left``, ``daily_stop``, ``halted``, ``last_outcome``
        and ``last_veto`` — aliases and counts, never a price the LLM could copy."""
        ...


def engaged(ledger: "PositionLedger") -> bool:
    """A position or a working entry order: the Brain may not sleep."""
    return ledger.has_open_position() or ledger.has_working_order()


class InMemoryPositionLedger:
    def __init__(self, positions: tuple[PositionRecord, ...] = ()) -> None:
        self._open: dict[str, PositionRecord] = {record.position_id: record for record in positions}

    def has_open_position(self) -> bool:
        return bool(self._open)

    def has_working_order(self) -> bool:
        return False

    def open_positions(self) -> tuple[PositionRecord, ...]:
        return tuple(self._open[key] for key in sorted(self._open))

    def execution_view(self) -> Mapping[str, Any]:
        return IDLE_VIEW

    def open(self, record: PositionRecord) -> None:
        if record.position_id in self._open:
            raise ValueError(f"position {record.position_id!r} is already open")
        self._open[record.position_id] = record

    def close(self, position_id: str) -> PositionRecord:
        try:
            return self._open.pop(position_id)
        except KeyError:
            raise ValueError(f"position {position_id!r} is not open") from None


__all__ = ["IDLE_VIEW", "InMemoryPositionLedger", "PositionLedger", "PositionRecord", "engaged"]
