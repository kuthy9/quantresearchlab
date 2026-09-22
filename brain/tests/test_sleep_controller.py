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
    assert CFG.relation_change_debounce_bars == 15  # schema 4 (2026-09-18)
    payload = json.loads((ROOT / "brain" / "configs" / "sleep_controller.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 4
    del payload["events"]
    old = tmp_path / "v4.json"
    old.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        ControllerConfig.from_json(old)


def test_phase_transitions_are_bookkeeping_not_reactions() -> None:
    # A delivery phase is a derived label the scales block already carries
    # (2026-09-18); its entered/exited events are evidence, never a call.
    for kind in (EventKind.DELIVERY_PHASE_ENTERED, EventKind.DELIVERY_PHASE_EXITED):
        for tf in (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4):
            assert not CFG.is_update_event(ev(kind, tf)), (kind, tf)
            assert not CFG.is_wake_event(ev(kind, tf)), (kind, tf)


def test_schema_5_carries_the_event_filter_and_the_calendar_enters_the_hash() -> None:
    import hashlib

    raw = (ROOT / "brain" / "configs" / "sleep_controller.json").read_bytes()
    assert CFG.sha256 != hashlib.sha256(raw).hexdigest()  # the calendar bytes are part of the identity
    kinds = {event.kind for event in CFG.events.events}
    assert kinds == {"CPI", "NFP", "FOMC"}
    cpi = CFG.events.active(pd.Timestamp("2022-10-13T12:31:00Z"))
    assert cpi is not None and cpi.kind == "CPI" and cpi.start == pd.Timestamp("2022-10-13T11:30:00Z") and cpi.end == pd.Timestamp("2022-10-13T13:00:00Z")
    fomc = CFG.events.active(pd.Timestamp("2022-07-27T17:00:00Z"))
    assert fomc is not None and fomc.kind == "FOMC" and fomc.end == pd.Timestamp("2022-07-27T19:30:00Z")
    nfp = CFG.events.active(pd.Timestamp("2022-01-07T12:45:00Z"))
    assert nfp is not None and nfp.kind == "NFP"
    assert CFG.events.active(pd.Timestamp("2022-01-03T14:30:00Z")) is None  # the frozen window has no release


def test_inside_an_event_window_the_brain_stays_or_goes_to_sleep_and_wakes_when_it_ends() -> None:
    inside = pd.Timestamp("2022-10-13T12:31:00Z")  # 08:31 New York, one minute after the CPI print
    wake = [ev(EventKind.DISPLACEMENT_OBSERVED, Timeframe.M15)]
    asleep = decide(wake, active=False, config=CFG, known_at=inside, previous_known_at=inside - pd.Timedelta(minutes=1))
    assert asleep.decision is Decision.STAY_ASLEEP and asleep.reasons == ("event:CPI:2022-10-13T12:30:00Z",)
    active = decide(wake, active=True, config=CFG, known_at=inside, previous_known_at=inside - pd.Timedelta(minutes=1))
    assert active.decision is Decision.EVENT_SLEEP and active.reasons == ("event:CPI:2022-10-13T12:30:00Z",)
    # the window's first bar puts an active episode to sleep even without an Eye event
    first = pd.Timestamp("2022-10-13T11:30:00Z")
    assert decide([], active=True, config=CFG, known_at=first, previous_known_at=first - pd.Timedelta(minutes=1)).decision is Decision.EVENT_SLEEP
    # the first bar at the window's end wakes the Brain, Eye event or not
    end = pd.Timestamp("2022-10-13T13:00:00Z")
    woken = decide([], active=False, config=CFG, known_at=end, previous_known_at=end - pd.Timedelta(minutes=1))
    assert woken.decision is Decision.WAKE and woken.reasons == ("event_ended:CPI:2022-10-13T12:30:00Z",)
    later = decide([], active=False, config=CFG, known_at=end + pd.Timedelta(minutes=1), previous_known_at=end)
    assert later.decision is Decision.STAY_ASLEEP
    # outside any window the calendar changes nothing, and callers without a clock get the old rule
    outside = pd.Timestamp("2022-10-13T15:00:00Z")
    assert decide(wake, active=False, config=CFG, known_at=outside, previous_known_at=outside - pd.Timedelta(minutes=1)).decision is Decision.WAKE
    assert decide(wake, active=True, config=CFG, known_at=outside, previous_known_at=outside - pd.Timedelta(minutes=1)).decision is Decision.UPDATE
    assert decide(wake, active=False, config=CFG).decision is Decision.WAKE
    assert decide(wake, active=True, config=CFG, known_at=inside).decision is Decision.EVENT_SLEEP  # no previous bar needed to sleep
