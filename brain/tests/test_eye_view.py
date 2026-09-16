from __future__ import annotations

import pandas as pd

from brain.core.eye_view import visible_liquidity_ids
from contract.eye import LiquidityInventoryItem, LiquidityInventoryLifecycle
from contract.market import Timeframe


def _item(
    item_id: str,
    *,
    lifecycle=LiquidityInventoryLifecycle.VISIBLE,
    confirmed="2022-01-04T15:00:00Z",
    price=100.0,
):
    consumed = lifecycle is LiquidityInventoryLifecycle.CONSUMED
    return LiquidityInventoryItem(
        item_id=item_id, timeframe=Timeframe.M5, side="above", kind="swing",
        price=price, lower_bound=price - 0.25, upper_bound=price + 0.25,
        formed_at=pd.Timestamp("2022-01-04T14:00:00Z"), confirmed_at=pd.Timestamp(confirmed),
        lifecycle=lifecycle, source_ids=("s1",), age_bars=3, strength=0.5,
        consumed_at=pd.Timestamp("2022-01-04T15:02:00Z") if consumed else None,
        lifecycle_reason="swing_swept" if consumed else None,
    )


class _Obs:
    def __init__(self, items, asof="2022-01-04T15:05:00Z"):
        self.liquidity_inventory = tuple(items)
        self.asof = pd.Timestamp(asof)


def test_only_visible_confirmed_unambiguous_items_count() -> None:
    obs = _Obs([
        _item("a"),
        _item("b", lifecycle=LiquidityInventoryLifecycle.CONSUMED),
        _item("c", confirmed="2022-01-04T15:10:00Z"),
        _item("d", price=100.0), _item("d", price=101.0),  # ambiguous duplicate
        _item("e"), _item("e"),  # an exact repeat is not ambiguous
    ])
    assert visible_liquidity_ids(obs) == {"a", "e"}
