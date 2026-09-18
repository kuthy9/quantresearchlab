from pathlib import Path

import pandas as pd
import pytest

from brain.core.sleep_controller import ControllerConfig, Decision, decide
from contract.eye import EventKind, MarketEvent
from contract.market import Timeframe

ROOT = Path(__file__).resolve().parents[2]
CFG = ControllerConfig.from_json(ROOT / "brain" / "configs" / "sleep_controller.json")
T = pd.Timestamp("2022-01-04T15:00:00Z")


def ev(kind: EventKind, tf: Timeframe) -> MarketEvent:
    return MarketEvent(
        event_id=f"{kind.value}:{tf.value}", kind=kind, observed_at=T, timeframe=tf,
        side=None, price=None, strength=0.5,
    )


@pytest.mark.parametrize("kind,tf,expected", [
    (EventKind.SWEEP_CONFIRMED, Timeframe.M1, Decision.STAY_ASLEEP),
    (EventKind.SWING_CONFIRMED, Timeframe.M15, Decision.STAY_ASLEEP),
    (EventKind.FVG_CREATED, Timeframe.H1, Decision.STAY_ASLEEP),
    (EventKind.SWEEP_CONFIRMED, Timeframe.M15, Decision.WAKE),
    (EventKind.QUALIFIED_BOS, Timeframe.M5, Decision.WAKE),
    (EventKind.FVG_CREATED, Timeframe.M5, Decision.STAY_ASLEEP),
    (EventKind.LEVEL_REACHED, Timeframe.M5, Decision.STAY_ASLEEP),
    (EventKind.DEALING_RANGE_INVALIDATED, Timeframe.H4, Decision.WAKE),
    (EventKind.FVG_STATE, Timeframe.H4, Decision.STAY_ASLEEP),
    (EventKind.BAR_COMPLETED, Timeframe.H4, Decision.STAY_ASLEEP),
    (EventKind.DISPLACEMENT_OBSERVED, Timeframe.M15, Decision.WAKE),
    (EventKind.DISPLACEMENT_OBSERVED, Timeframe.M5, Decision.WAKE),
])
def test_wake_rule(kind, tf, expected) -> None:
    d = decide([ev(kind, tf)], active=False, config=CFG)
    assert d.decision is expected
    if expected is Decision.WAKE:
        assert d.reasons == (f"ev_{kind.value}:{tf.value}",)


def test_active_updates_on_a_5m_plus_reaction_or_relation_change_not_on_bookkeeping() -> None:
    # Bookkeeping kinds (formation, touches, level creation) are evidence but do not wake the LLM.
    assert decide([ev(EventKind.FVG_CREATED, Timeframe.M5)], active=True, config=CFG).decision is Decision.TICK
    assert decide([ev(EventKind.LEVEL_TOUCHED, Timeframe.M15)], active=True, config=CFG).decision is Decision.TICK
    assert decide([ev(EventKind.SWEEP_CONFIRMED, Timeframe.M5)], active=True, config=CFG).decision is Decision.UPDATE
    assert decide([ev(EventKind.DISPLACEMENT_OBSERVED, Timeframe.M5)], active=True, config=CFG).decision is Decision.UPDATE
    assert decide([ev(EventKind.ACCEPTANCE_CONFIRMED, Timeframe.M5)], active=True, config=CFG).decision is Decision.UPDATE
    assert decide([ev(EventKind.SWEEP_CONFIRMED, Timeframe.M1)], active=True, config=CFG).decision is Decision.TICK
    assert decide([ev(EventKind.FVG_STATE, Timeframe.M5)], active=True, config=CFG).decision is Decision.TICK
    assert decide([], active=True, config=CFG).decision is Decision.TICK
    d = decide([], active=True, config=CFG, relation_changes=["FVG_5m_3"])
    assert d.decision is Decision.UPDATE and d.reasons == ("FVG_5m_3",)


def test_tape_rule_and_hash() -> None:
    assert CFG.is_tape_event(ev(EventKind.SWEEP_CONFIRMED, Timeframe.M1))
    assert not CFG.is_tape_event(ev(EventKind.SWEEP_CONFIRMED, Timeframe.M5))
    assert not CFG.is_tape_event(ev(EventKind.LEVEL_TOUCHED, Timeframe.M1))
    assert len(CFG.sha256) == 64 and CFG.tape_recent_limit == 8


def test_idle_archive_threshold_is_configured() -> None:
    assert CFG.idle_archive_after_updates == 6
    assert "fvg_created" in CFG.bookkeeping_kinds and "displacement_observed" not in CFG.bookkeeping_kinds


def test_relation_changes_count_only_on_the_configured_scales(tmp_path: Path) -> None:
    import json

    assert CFG.relation_change_timeframes == frozenset({"15m", "1H", "4H"})
    payload = json.loads((ROOT / "brain" / "configs" / "sleep_controller.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    del payload["relation_change_timeframes"]
    old = tmp_path / "v2.json"
    old.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        ControllerConfig.from_json(old)
