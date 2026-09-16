"""The Brain's view of open positions — the boundary Execution will own.

The reducer reads one fact through it: is anything open?  An open position
forbids sleep.  ``InMemoryPositionLedger`` is the only implementation until
Execution provides the real one; it holds what a caller records and nothing
else — no sizing, no stops, no fills."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from contract.brain.state import TradeDirection
from contract.market.primitives import aware_timestamp


@dataclass(frozen=True)
class PositionRecord:
    position_id: str
    direction: TradeDirection
    opened_at: pd.Timestamp
    entry_object_id: str

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
        }


class PositionLedger(Protocol):
    def has_open_position(self) -> bool: ...

    def open_positions(self) -> tuple[PositionRecord, ...]: ...


class InMemoryPositionLedger:
    def __init__(self, positions: tuple[PositionRecord, ...] = ()) -> None:
        self._open: dict[str, PositionRecord] = {record.position_id: record for record in positions}

    def has_open_position(self) -> bool:
        return bool(self._open)

    def open_positions(self) -> tuple[PositionRecord, ...]:
        return tuple(self._open[key] for key in sorted(self._open))

    def open(self, record: PositionRecord) -> None:
        if record.position_id in self._open:
            raise ValueError(f"position {record.position_id!r} is already open")
        self._open[record.position_id] = record

    def close(self, position_id: str) -> PositionRecord:
        try:
            return self._open.pop(position_id)
        except KeyError:
            raise ValueError(f"position {position_id!r} is not open") from None


__all__ = ["InMemoryPositionLedger", "PositionLedger", "PositionRecord"]
