from brain.core.object_registry import ObjectRegistry
from contract.brain.state import RegisteredObject


def test_aliases_are_stable_and_numbered_per_prefix_and_timeframe() -> None:
    reg = ObjectRegistry()
    assert reg.alias_for("e1", kind="fvg", timeframe="5m") == "FVG_5m_1"
    assert reg.alias_for("e2", kind="fvg", timeframe="5m") == "FVG_5m_2"
    assert reg.alias_for("e3", kind="fvg", timeframe="15m") == "FVG_15m_1"
    assert reg.alias_for("e1", kind="fvg", timeframe="5m") == "FVG_5m_1"
    assert reg.alias_for("e4", kind="bsl", timeframe="1H") == "BSL_1H_1"
    assert reg.alias_for("e5", kind="swing_high", timeframe="1H") == "SWING_H_1H_1"
    assert reg.alias_of("e2") == "FVG_5m_2"
    assert reg.get("FVG_5m_2") == RegisteredObject("e2", "fvg", "5m")
    assert reg.alias_of(None) is None and reg.get("GHOST") is None
    assert len(reg) == 5


def test_snapshot_restores_counters() -> None:
    reg = ObjectRegistry()
    reg.alias_for("e1", kind="fvg", timeframe="5m")
    reg.alias_for("e2", kind="fvg", timeframe="5m")
    restored = ObjectRegistry(reg.snapshot())
    assert restored.alias_for("e4", kind="fvg", timeframe="5m") == "FVG_5m_3"
    assert restored.alias_of("e1") == "FVG_5m_1"
