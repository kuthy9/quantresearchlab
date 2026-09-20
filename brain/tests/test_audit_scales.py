"""``audit_scales.py`` prints the change points of the per-scale facts the
Brain reads (2026-09-18)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from brain.core.sleep_controller import ControllerConfig
from brain.scripts.audit_scales import change_points, replay_triggers, scale_facts
from brain.tests.test_eye_view import synthetic_observations  # noqa: F401 — the fixture
from contract.market import Timeframe


def test_change_points_keep_only_rows_whose_discrete_facts_moved() -> None:
    rows = [
        ("t0", 1.0, {"a": 1, "forming_leg_atr": 0.1}),
        ("t1", 2.0, {"a": 1, "forming_leg_atr": 0.7}),
        ("t2", 3.0, {"a": 2, "forming_leg_atr": 0.7}),
    ]
    assert [row[0] for row in change_points(rows)] == ["t0", "t2"]


def test_scale_facts_read_the_published_state(synthetic_observations) -> None:
    snapshot = synthetic_observations[-1].market_snapshot
    facts = scale_facts(snapshot, Timeframe.M5, known_at=snapshot.asof)
    assert set(facts) >= {"external", "internal", "active_leg", "last_leg", "phase", "forming_leg_atr", "reset", "displacement", "displacement_age_bars"}
    assert facts["phase"] in ("expansion", "retracement", "reversal_attempt", "balance", "transition")


CFG = ControllerConfig.from_json(Path(__file__).resolve().parents[2] / "brain" / "configs" / "sleep_controller.json")


def test_replay_triggers_keeps_wakes_reactions_and_undebounced_relations() -> None:
    t = pd.Timestamp("2022-01-03T15:00:00Z")
    kinds = {"ev_a": "sweep_confirmed", "ev_b": "delivery_phase_entered"}
    calls = [
        (t, "WAKE", ["ev_a"], kinds),
        (t + pd.Timedelta(minutes=1), "UPDATE", ["ev_b"], kinds),          # bookkeeping only: dropped
        (t + pd.Timedelta(minutes=2), "UPDATE", ["FVG_15m_9"], kinds),     # first relation flip: kept
        (t + pd.Timedelta(minutes=4), "UPDATE", ["FVG_15m_9"], kinds),     # within the window: dropped
        (t + pd.Timedelta(minutes=20), "UPDATE", ["FVG_15m_9"], kinds),    # after it: kept
        (t + pd.Timedelta(minutes=21), "UPDATE", ["ev_a", "FVG_15m_9"], kinds),  # a reaction: kept
    ]
    kept = replay_triggers(calls, config=CFG)
    assert kept == [t, t + pd.Timedelta(minutes=2), t + pd.Timedelta(minutes=20), t + pd.Timedelta(minutes=21)]
