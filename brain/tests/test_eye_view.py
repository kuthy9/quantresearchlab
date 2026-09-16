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


# --- the full context ------------------------------------------------------

from pathlib import Path

import pytest

from brain.core.eye_view import CausalityError, EvidenceRule, assert_causal, build_eye_context, price_relation
from brain.core.object_registry import ObjectRegistry
from shares.core.eye_factory import build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
RULE = EvidenceRule(frozenset({"5m", "15m", "1H", "4H"}), frozenset({"bar_completed", "market_epoch_reset"}), "_state")


@pytest.fixture(scope="module")
def synthetic_observations():
    reader, observer = build_eye(ROOT / "configs" / "model.json", root=ROOT, audit_journal_dir=None)
    out = []
    for bar in session_bars(1):
        obs = observer.observe(reader.on_bar(bar))
        if obs.market_snapshot is not None:
            out.append(obs)
    return out


def test_price_relation() -> None:
    assert price_relation(103.0, 100.0, 102.0, 0.5) == ("above", 2.0)
    assert price_relation(99.0, 100.0, 102.0, 0.5) == ("below", -2.0)
    assert price_relation(101.0, 100.0, 102.0, 0.5) == ("inside", 0.0)
    assert price_relation(101.0, 100.0, 102.0, None) == ("inside", None)


def test_context_is_causal_and_aliases_every_object(synthetic_observations) -> None:
    registry = ObjectRegistry()
    seen_alias = False
    seen_evidence = False
    for obs in synthetic_observations[-120:]:
        ctx = build_eye_context(obs, registry, rule=RULE)
        assert ctx.known_at == obs.asof
        assert_causal({"scales": ctx.scales, "session": ctx.session, "events": [e.to_dict() for e in ctx.events]}, ctx.known_at)
        for view in ctx.objects:
            assert registry.get(view.alias) is not None
            assert view.timeframe != "1m"
            seen_alias = True
        for item in ctx.events:
            assert item.timeframe in RULE.timeframes and item.verdict is None
            assert item.evidence_id.startswith("ev_")
            seen_evidence = True
        assert {rel["object_id"] for rel in ctx.price_relations} == ctx.visible_aliases()
        for rel in ctx.price_relations:
            assert rel["relation"] in {"above", "inside", "below"}
        assert set(ctx.scales) <= {"4H", "1H", "15m", "5m", "1m"}
        assert "structure" in ctx.scales["1m"] and "zones" not in ctx.scales["1m"]
    assert seen_alias and seen_evidence


def test_aliases_are_reproduced_by_a_fresh_registry(synthetic_observations) -> None:
    a, b = ObjectRegistry(), ObjectRegistry()
    for obs in synthetic_observations[-30:]:
        ca = build_eye_context(obs, a, rule=RULE)
        cb = build_eye_context(obs, b, rule=RULE)
        assert [v.to_dict() for v in ca.objects] == [v.to_dict() for v in cb.objects]
    assert a.snapshot() == b.snapshot()


def test_assert_causal_rejects_future_timestamps() -> None:
    import pandas as pd
    with pytest.raises(CausalityError):
        assert_causal({"a": [{"t": "2022-01-04T15:00:00Z"}]}, pd.Timestamp("2022-01-04T14:59:00Z"))
    assert_causal({"a": [{"t": "2022-01-04T14:59:00Z"}], "n": 3, "s": "not a date"}, pd.Timestamp("2022-01-04T14:59:00Z"))
