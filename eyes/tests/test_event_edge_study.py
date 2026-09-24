from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from eyes.scripts.event_edge_study import (
    Tape,
    candidate_edge,
    classify_day,
    clustered_mean,
    excess_over_baseline,
    leg_flips,
    outcomes,
    sweep_then_mss,
    trade_date,
)

TZ = "America/New_York"


def _tape(highs, lows, closes, *, gap_after=None, start="2023-03-01 10:00"):
    t0 = pd.Timestamp(start, tz=TZ)
    minutes = []
    for i in range(len(closes)):
        shift = 0 if gap_after is None or i <= gap_after else 60
        minutes.append(int((t0 + pd.Timedelta(minutes=i + shift)).value // 60_000_000_000))
    return Tape(np.array(minutes, dtype=np.int64), np.array(highs, float), np.array(lows, float), np.array(closes, float))


def test_long_and_short_returns_mfe_and_mae_over_the_horizon() -> None:
    tape = _tape([101, 103, 104, 102], [99, 100, 101, 98], [100, 102, 103, 101])
    long = outcomes(tape, 0, +1, unit=2.0, horizons=(2, 3), brackets=())
    assert long["ret_2"] == 3.0 and long["mfe_2"] == 4.0 and long["mae_2"] == 0.0
    assert long["ret_3"] == 1.0 and long["mfe_3"] == 4.0 and long["mae_3"] == 2.0
    short = outcomes(tape, 0, -1, unit=2.0, horizons=(3,), brackets=())
    assert short["ret_3"] == -1.0 and short["mfe_3"] == 2.0 and short["mae_3"] == 4.0


def test_a_horizon_crossing_a_gap_is_dropped_and_the_bracket_stops_at_the_gap() -> None:
    tape = _tape([100.5, 100.5, 100.5, 110, 110], [99.5, 99.5, 99.5, 99, 99], [100, 100, 100, 105, 105], gap_after=2)
    out = outcomes(tape, 0, +1, unit=1.0, horizons=(2, 3), brackets=((2, 1),), bracket_horizon=4)
    assert out["ret_2"] == 0.0 and out["ret_3"] is None and out["mfe_3"] is None
    assert out["br_2_1"] == "none" and out["br_2_1_min"] is None


def test_brackets_resolve_first_touch_and_a_bar_touching_both_is_the_stop() -> None:
    tape = _tape([100, 101, 103, 100], [100, 99.5, 99, 98], [100, 100.5, 102, 99])
    out = outcomes(tape, 0, +1, unit=1.0, horizons=(), brackets=((1, 1), (2, 1)), bracket_horizon=3)
    assert out["br_1_1"] == "target" and out["br_1_1_min"] == 1
    assert out["br_2_1"] == "stop" and out["br_2_1_min"] == 2
    short = outcomes(tape, 0, -1, unit=1.0, horizons=(), brackets=((1, 1),), bracket_horizon=3)
    assert short["br_1_1"] == "stop" and short["br_1_1_min"] == 1


def test_day_types_follow_the_preregistered_rule() -> None:
    at = pd.Timestamp("2023-03-01 10:00", tz=TZ)
    later = at + pd.Timedelta(hours=3)
    assert classify_day(100, 110, 99, 109, t_high=later, t_low=at) == "trend"
    assert classify_day(100, 108, 96, 97, t_high=at, t_low=later) == "reversal"
    assert classify_day(100, 104, 96, 101, t_high=at, t_low=later) == "chop"
    assert classify_day(100, 100, 100, 100, t_high=at, t_low=at) == "chop"


def test_the_trade_date_is_the_date_of_the_17h_close() -> None:
    assert trade_date(pd.Timestamp("2023-03-01 18:30", tz=TZ)) == pd.Timestamp("2023-03-02").date()
    assert trade_date(pd.Timestamp("2023-03-01 10:00", tz=TZ)) == pd.Timestamp("2023-03-01").date()
    assert trade_date(pd.Timestamp("2023-03-05 18:00", tz=TZ)) == pd.Timestamp("2023-03-06").date()


def test_the_mean_carries_a_standard_error_clustered_by_day() -> None:
    mean, se, t, n = clustered_mean(np.array([2.0, 4.0, 6.0, 8.0]), np.array(["a", "a", "b", "b"]))
    assert mean == 5.0 and se == pytest.approx(2.0) and t == pytest.approx(2.5) and n == 4
    mean, se, t, n = clustered_mean(np.array([1.0, np.nan]), np.array(["a", "b"]))
    assert n == 1 and math.isnan(se) and math.isnan(t)


def test_the_excess_is_measured_against_the_same_hour_and_direction() -> None:
    baseline = pd.DataFrame({"hour": [10, 10, 11], "direction": ["long", "short", "long"], "ret_60_R": [0.5, -0.2, 0.1]})
    events = pd.DataFrame({"hour": [10, 10, 12], "direction": ["long", "short", "long"], "ret_60_R": [1.5, 0.3, 9.0]})
    ex = excess_over_baseline(events, baseline, "ret_60_R")
    assert ex.iloc[0] == pytest.approx(1.0) and ex.iloc[1] == pytest.approx(0.5) and math.isnan(ex.iloc[2])


def test_leg_flips_are_changes_between_consecutive_closes() -> None:
    bars = pd.DataFrame({"known_at": pd.date_range("2023-03-01 10:00", periods=6, freq="15min", tz=TZ),
                         "leg": ["long", "long", "short", None, "short", "long"]})
    flips = leg_flips(bars, "leg")
    assert list(flips["direction"]) == ["short", "long"] and list(flips.index) == [2, 5]


def test_sweep_then_mss_keeps_an_mss_that_follows_a_same_direction_sweep_within_the_window() -> None:
    t = pd.Timestamp("2023-03-01 10:00", tz=TZ)
    sweeps = pd.DataFrame({"known_at": [t], "direction": ["long"]})
    mss = pd.DataFrame({"known_at": [t + pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=90), t + pd.Timedelta(minutes=30), t],
                        "direction": ["long", "long", "short", "long"]})
    assert list(sweep_then_mss(sweeps, mss, minutes=60)) == [True, False, False, False]


def _stats(**over):
    base = dict(n=400, n_h1=180, n_h2=220, t=3.5, mean=0.2, t_h1=2.0, mean_h1=0.15, t_h2=1.8, mean_h2=0.25,
                months_positive=9, months=12, mean_pts=2.0)
    base.update(over)
    return base


def test_the_candidate_rule_needs_size_significance_both_halves_months_and_cost() -> None:
    assert candidate_edge(_stats()) is True
    assert candidate_edge(_stats(n=120)) is False
    assert candidate_edge(_stats(n_h1=50)) is False
    assert candidate_edge(_stats(t=2.9)) is False
    assert candidate_edge(_stats(mean_h2=-0.1, t_h2=-1.8)) is False
    assert candidate_edge(_stats(t_h1=1.2)) is False
    assert candidate_edge(_stats(months_positive=7)) is False
    assert candidate_edge(_stats(mean_pts=0.8)) is False
    assert candidate_edge(_stats(mean_pts=None), require_cost=False) is True


def test_a_fade_is_the_same_rule_on_the_negated_statistics() -> None:
    from eyes.scripts.event_edge_study import negated

    fade = negated(_stats(mean=0.2, t=3.5, mean_h1=0.15, t_h1=2.0, mean_h2=0.25, t_h2=1.8, months_positive=9, mean_pts=2.0))
    assert fade["mean"] == -0.2 and fade["t"] == -3.5 and fade["months_positive"] == 3 and fade["mean_pts"] == -2.0
    assert candidate_edge(fade) is False
    strong_negative = _stats(mean=-0.2, t=-3.5, mean_h1=-0.15, t_h1=-2.0, mean_h2=-0.25, t_h2=-1.8, months_positive=2, mean_pts=-2.0)
    assert candidate_edge(negated(strong_negative)) is True
