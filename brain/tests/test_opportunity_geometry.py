from __future__ import annotations

import pytest

from brain.core.opportunity_geometry import ObjectGeometry, coherence_error, resolve_geometry
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
