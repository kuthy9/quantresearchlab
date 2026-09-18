from __future__ import annotations

import pytest

from dataclasses import replace

from brain.core.opportunity_geometry import CLOSE_BEYOND_BUFFER_ATR, ObjectGeometry, coherence_error, resolve_geometry
from contract.brain.state import Opportunity, OpportunityState, TradeDirection
from contract.decision import GeometryError

OBJ = {
    "FVG_5m_3": ObjectGeometry("FVG_5m_3", "fvg", "5m", 100.0, 102.0, 101.0),
    "SSL_5m_2": ObjectGeometry("SSL_5m_2", "ssl", "5m", 97.75, 98.25, 98.0),
    "BSL_1H_1": ObjectGeometry("BSL_1H_1", "bsl", "1H", 109.75, 110.25, 110.0),
    "OB_15m_1": ObjectGeometry("OB_15m_1", "ob", "15m", 104.0, 106.0, 105.0),
    "DR_15m_1": ObjectGeometry("DR_15m_1", "range", "15m", 96.0, 112.0, 104.0),
    "SWING_L_5m_1": ObjectGeometry("SWING_L_5m_1", "swing_low", "5m", 99.0, 99.0, 99.0),
}


def test_long_from_fvg_to_pool_with_pool_stop() -> None:
    g = resolve_geometry(
        Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1"),
        OBJ, close=103.0, tick=0.25,
    )
    assert (g.entry_price, g.stop_price, g.target_price) == (102.0, 97.5, 110.0)
    assert g.reward_risk == pytest.approx(8.0 / 4.5)
    assert g.rule_ids == ("entry.zone.near_edge", "stop.pool.far_edge", "target.pool.midpoint")


def test_short_mirrors() -> None:
    g = resolve_geometry(
        Opportunity(OpportunityState.DEVELOPING, TradeDirection.SHORT, "OB_15m_1", "BSL_1H_1", "SSL_5m_2"),
        OBJ, close=103.0, tick=0.25,
    )
    assert (g.entry_price, g.stop_price, g.target_price) == (104.0, 110.5, 98.0)
    assert g.rule_ids == ("entry.zone.near_edge", "stop.pool.far_edge", "target.pool.midpoint")


def test_swing_stop_and_range_entry() -> None:
    g = resolve_geometry(
        Opportunity(OpportunityState.DEVELOPING, TradeDirection.LONG, "DR_15m_1", "SWING_L_5m_1", "BSL_1H_1"),
        OBJ, close=103.0, tick=0.25,
    )
    assert (g.entry_price, g.stop_price, g.target_price) == (104.0, 98.75, 110.0)
    assert g.rule_ids == ("entry.range.value", "stop.swing.price", "target.pool.midpoint")


def test_incoherent_direction_fails() -> None:
    long_with_target_below = Opportunity(
        OpportunityState.ACTIONABLE, TradeDirection.LONG, "OB_15m_1", "BSL_1H_1", "SSL_5m_2"
    )
    assert coherence_error(long_with_target_below, OBJ, close=103.0, tick=0.25) is not None
    with pytest.raises(GeometryError):
        resolve_geometry(long_with_target_below, OBJ, close=103.0, tick=0.25)


def test_none_opportunity_and_unknown_alias() -> None:
    with pytest.raises(GeometryError, match="NONE"):
        resolve_geometry(Opportunity(), OBJ, close=103.0, tick=0.25)
    with pytest.raises(GeometryError, match="GHOST"):
        resolve_geometry(
            Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "GHOST_1", "SSL_5m_2", "BSL_1H_1"),
            OBJ, close=103.0, tick=0.25,
        )
    assert coherence_error(Opportunity(), OBJ, close=103.0, tick=0.25) is None


def test_close_beyond_puts_the_hard_stop_one_scaled_atr_past_the_edge_rounded_away() -> None:
    objects = {
        "FVG_15m_1": ObjectGeometry("FVG_15m_1", "fvg", "15m", 100.0, 102.0, 101.0),
        "BSL_15m_1": ObjectGeometry("BSL_15m_1", "bsl", "15m", 104.0, 104.0, 104.0),
        "SSL_5m_1": ObjectGeometry("SSL_5m_1", "ssl", "5m", 80.0, 80.0, 80.0),
    }
    opp = Opportunity("ACTIONABLE", "SHORT", "FVG_15m_1", "BSL_15m_1", "SSL_5m_1", thesis_id="T1", governing_timeframe="1H", invalidation_mode="CLOSE_BEYOND")
    assert CLOSE_BEYOND_BUFFER_ATR == 1.0
    g = resolve_geometry(opp, objects, close=99.0, tick=0.25, atr_1m=2.0)
    assert g.stop_price == 111.75  # 104 + 1.0 × 2.0 × √15 = 111.746, rounded away from the entry
    assert g.rule_ids[1] == "stop.pool.close_beyond" and g.reward_risk == pytest.approx((100.0 - 80.0) / (111.75 - 100.0))
    touch = resolve_geometry(replace(opp, invalidation_mode="TOUCH"), objects, close=99.0, tick=0.25, atr_1m=2.0)
    assert touch.stop_price == 104.25 and touch.rule_ids[1] == "stop.pool.far_edge"
    long_ = Opportunity("ACTIONABLE", "LONG", "SSL_5m_1", "FVG_15m_1", "BSL_15m_1", thesis_id="T1", governing_timeframe="15m", invalidation_mode="CLOSE_BEYOND")
    g = resolve_geometry(long_, {**objects, "SSL_5m_1": ObjectGeometry("SSL_5m_1", "ssl", "5m", 103.0, 103.0, 103.0)}, close=103.5, tick=0.25, atr_1m=1.0)
    assert g.stop_price == 96.0 and g.rule_ids[1] == "stop.zone.close_beyond"  # 100 − 1.0 × 1.0 × √15 = 96.127, floored away to 96.0
    with pytest.raises(GeometryError, match="atr"):
        resolve_geometry(opp, objects, close=99.0, tick=0.25, atr_1m=None)
    assert coherence_error(opp, objects, close=99.0, tick=0.25, atr_1m=None) is not None
