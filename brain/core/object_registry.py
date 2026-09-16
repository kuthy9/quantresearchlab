"""Aliases for the Eye's objects, stable inside one episode.

The Eye identifies entities by hash.  The LLM names them by alias —
``FVG_5m_3`` — assigned on first appearance and never reused, so a state, a
journal and a reply all speak the same names.  Aliases are numbered per prefix
and timeframe in the order ``alias_for`` is called, which ``eye_view`` keeps
deterministic; a replay therefore reproduces them."""
from __future__ import annotations

from collections.abc import Mapping

from contract.brain.state import RegisteredObject
from contract.market.primitives import FrozenDict

ALIAS_PREFIX: Mapping[str, str] = FrozenDict(
    {
        "fvg": "FVG",
        "ob": "OB",
        "bsl": "BSL",
        "ssl": "SSL",
        "range": "DR",
        "swing_high": "SWING_H",
        "swing_low": "SWING_L",
    }
)


class ObjectRegistry:
    def __init__(self, entries: Mapping[str, RegisteredObject] | None = None) -> None:
        self._by_alias: dict[str, RegisteredObject] = {}
        self._by_entity: dict[str, str] = {}
        self._counters: dict[tuple[str, str], int] = {}
        for alias, entry in (entries or {}).items():
            entry = entry if isinstance(entry, RegisteredObject) else RegisteredObject(**entry)
            self._by_alias[alias] = entry
            self._by_entity[entry.entity_id] = alias
            prefix = ALIAS_PREFIX[entry.kind]
            number = int(alias.rsplit("_", 1)[1])
            key = (prefix, entry.timeframe)
            self._counters[key] = max(self._counters.get(key, 0), number)

    def alias_for(self, entity_id: str, *, kind: str, timeframe: str) -> str:
        existing = self._by_entity.get(entity_id)
        if existing is not None:
            return existing
        prefix = ALIAS_PREFIX[kind]
        key = (prefix, timeframe)
        number = self._counters.get(key, 0) + 1
        self._counters[key] = number
        alias = f"{prefix}_{timeframe}_{number}"
        self._by_alias[alias] = RegisteredObject(entity_id, kind, timeframe)
        self._by_entity[entity_id] = alias
        return alias

    def get(self, alias: str) -> RegisteredObject | None:
        return self._by_alias.get(alias)

    def alias_of(self, entity_id: str | None) -> str | None:
        if entity_id is None:
            return None
        return self._by_entity.get(entity_id)

    def snapshot(self) -> Mapping[str, RegisteredObject]:
        return FrozenDict(dict(self._by_alias))

    def __len__(self) -> int:
        return len(self._by_alias)


__all__ = ["ALIAS_PREFIX", "ObjectRegistry"]
