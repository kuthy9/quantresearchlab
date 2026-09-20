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
        objects = ctx.object_map()
        for rel in ctx.price_relations:
            # the row says where the OBJECT lies relative to price, signed the same way
            assert set(rel) == {"object_id", "position", "offset_atr"}
            view = objects[rel["object_id"]]
            if rel["position"] == "above_price":
                assert view.lower > ctx.close and (rel["offset_atr"] is None or rel["offset_atr"] > 0.0)
            elif rel["position"] == "below_price":
                assert view.upper < ctx.close and (rel["offset_atr"] is None or rel["offset_atr"] < 0.0)
            else:
                assert rel["position"] == "contains_price" and view.lower <= ctx.close <= view.upper and rel["offset_atr"] in (0.0, None)
            assert ctx.relation_of(rel["object_id"]) == rel["position"]
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


def _step(step_id: str, kind: str, at: str, source: str, *, prev: str | None = None) -> "PathSequenceStep":
    from contract.eye import PathSequenceStep
    from contract.market import Direction

    return PathSequenceStep(
        step_id=step_id, kind=kind, observed_at=pd.Timestamp(at), source_event_id=None, source_entity_id=source,
        predecessor_step_ids=() if prev is None else (prev,), same_clock_relation="origin" if prev is None else "strictly_after",
        direction=Direction.LONG, strength=0.5, reason="test",
    )


def _path(context_kind: str, context_id: str, steps, *, active: bool = True) -> "PathSequenceState":
    from contract.eye import PathSequenceLifecycle, PathSequenceState
    from contract.market import Direction

    last = max(step.observed_at for step in steps)
    return PathSequenceState(
        sequence_id=f"seq-{context_id}", protocol_hash="p", symbol="NQ", instrument_id=1,
        context_kind=context_kind, context_id=context_id, direction=Direction.LONG,
        lifecycle=PathSequenceLifecycle.ACTIVE if active else PathSequenceLifecycle.CLOSED,
        formed_at=steps[0].observed_at, state_started_at=steps[0].observed_at, last_updated_at=last,
        age_real_1m_bars=0, state_duration_real_1m_bars=0, steps=tuple(steps),
        ended_at=None if active else last, transition_reason="context_registered" if active else "closed",
    )


def test_interaction_rows_alias_their_source_object_and_say_when_they_last_stepped() -> None:
    from brain.core.eye_view import interaction_rows
    from brain.core.object_registry import ObjectRegistry

    registry = ObjectRegistry()
    fvg = registry.alias_for("zone-1", kind="fvg", timeframe="5m")
    pool = registry.alias_for("inv-7", kind="bsl", timeframe="15m")
    known_at = pd.Timestamp("2022-01-03T14:05:00Z")
    zone_path = _path("zone_return", "loc-1", [
        _step("s1", "zone_visible", "2022-01-03T13:50:00Z", "zone-1"),
        _step("s2", "departure_confirmed", "2022-01-03T14:05:00Z", "zone-1", prev="s1"),
    ])
    pool_path = _path("pool_reversal", "man-1", [
        _step("s3", "pool_swept", "2022-01-03T13:31:00Z", "man-1"),
        _step("s4", "reacceptance_held", "2022-01-03T13:40:00Z", "man-1", prev="s3"),
    ])
    orphan = _path("pool_reversal", "man-2", [_step("s5", "pool_swept", "2022-01-03T14:00:00Z", "man-9")])
    closed = _path("zone_return", "loc-2", [_step("s6", "zone_visible", "2022-01-03T14:05:00Z", "zone-1")], active=False)

    rows = interaction_rows(
        (zone_path, pool_path, orphan, closed), manipulation_sources={"man-1": "inv-7"},
        registry=registry, known_at=known_at, since=pd.Timestamp("2022-01-03T14:02:00Z"),
    )
    assert [row["context_kind"] for row in rows] == ["zone_return", "pool_reversal", "pool_reversal"]
    assert [row["object_id"] for row in rows] == [fvg, pool, None]
    assert [row["last_step"] for row in rows] == ["departure_confirmed", "reacceptance_held", "pool_swept"]
    assert [row["last_step_at"] for row in rows] == ["2022-01-03T14:05:00Z", "2022-01-03T13:40:00Z", "2022-01-03T14:00:00Z"]
    assert [row["stepped_since_last_call"] for row in rows] == [True, False, False]

    # No previous call (a wake): only a step on this very bar counts as open.
    rows = interaction_rows((pool_path, orphan), manipulation_sources={}, registry=registry, known_at=known_at, since=None)
    assert [row["stepped_since_last_call"] for row in rows] == [False, False]
    rows = interaction_rows((zone_path,), manipulation_sources={}, registry=registry, known_at=known_at, since=None)
    assert rows[0]["stepped_since_last_call"] is True


def test_scales_publish_the_forming_leg_the_displacement_age_and_the_reset(synthetic_observations) -> None:
    registry = ObjectRegistry()
    seen_forming = False
    for observation in synthetic_observations:
        context = build_eye_context(observation, registry, rule=RULE)
        for name, scale in context.scales.items():
            delivery = scale["delivery"]
            for key in ("phase", "active_leg_direction", "last_leg_direction", "forming_leg_atr", "displacement_score", "displacement_direction", "displacement_age_bars"):
                assert key in delivery, (name, key)
            assert "reset" in scale["structure"]
            if delivery["forming_leg_atr"] not in (None, 0):
                seen_forming = True
                assert (delivery["forming_leg_atr"] > 0) == (delivery["active_leg_direction"] == "long")
            if delivery["displacement_age_bars"] is not None:
                assert delivery["displacement_age_bars"] >= 0 and delivery["displacement_direction"] in ("long", "short")
    assert seen_forming


def test_session_drift_is_the_close_against_the_session_open_in_1m_atrs(synthetic_observations) -> None:
    observation = synthetic_observations[-1]
    context = build_eye_context(observation, ObjectRegistry(), rule=RULE)
    drift = context.session["drift_atr"]
    if context.atr_1m is None or context.session["session_open"] is None:
        assert drift is None
    else:
        assert drift == round((context.close - context.session["session_open"]) / context.atr_1m, 6)
