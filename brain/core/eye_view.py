"""What the Brain reads from one Eye observation.

The only Brain module that touches Eye types."""
from __future__ import annotations

from contract.eye import LiquidityInventoryLifecycle, MarketObservation


def visible_liquidity_ids(observation: MarketObservation) -> set[str]:
    """Inventory items a downstream consumer may act on: VISIBLE, confirmed by
    ``asof``, and not published twice with different identity or geometry."""
    seen: dict[str, tuple] = {}
    ambiguous: set[str] = set()
    for item in observation.liquidity_inventory:
        if (
            item.lifecycle is not LiquidityInventoryLifecycle.VISIBLE
            or item.confirmed_at > observation.asof
        ):
            continue
        if item.item_id in ambiguous:
            continue
        identity = (
            item.timeframe,
            item.side,
            round(float(item.price), 9),
            item.formed_at,
            item.confirmed_at,
        )
        prior = seen.get(item.item_id)
        if prior is None:
            seen[item.item_id] = identity
        elif prior != identity:
            del seen[item.item_id]
            ambiguous.add(item.item_id)
    return set(seen)


__all__ = ["visible_liquidity_ids"]
