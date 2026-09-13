"""The Setup gate on synthetic instances: a planted Setup effect passes on its
cell, no effect fails, thin or one-sided cells are refused not judged."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from brain.research.setup_labels import CENSORED, FAILURE_FIRST, TARGET_FIRST
from brain.scripts.setup_gate import clock_instances, run_setup_gate


def _instances(*, sessions: int = 14, per_session: int = 160, effect: float = 0.0, seed: int = 3,
               censor: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    day = pd.Timestamp("2022-01-03T09:30", tz="America/New_York")
    n = 0
    for _ in range(sessions):
        while day.weekday() >= 5:
            day += pd.Timedelta(days=1)
        for i in range(per_session):
            at = (day + pd.Timedelta(minutes=2 * i)).tz_convert("UTC")
            d_t, d_f = rng.uniform(0.3, 2.0), rng.uniform(0.3, 2.0)
            hold = rng.normal()
            logit = np.log(d_f / d_t) + effect * hold
            p_target = 1.0 / (1.0 + np.exp(-logit))
            if rng.uniform() < censor:
                label = CENSORED
            else:
                label = TARGET_FIRST if rng.uniform() < p_target else FAILURE_FIRST
            kind = "zone_return" if i % 2 == 0 else "pool_reversal"
            step = "reacceptance_held" if i % 3 else "zone_visible"
            row = {
                "known_at": at, "sequence_id": f"s{n}", "context_kind": kind, "step_kind": step,
                "path_formed_at": at - pd.Timedelta(minutes=20), "direction": 1.0 if i % 4 else -1.0,
                "source_zone_kind": "fvg" if kind == "zone_return" else None, "entry_mode": None,
                "source_timeframe": "5m" if kind == "pool_reversal" else None,
                "steps_so_far": json.dumps([["zone_visible", "registered", 0.5], [step, "held", 0.7]]),
                "lower_bound": 99.0, "upper_bound": 99.5, "atr_1m": 1.0, "first_penetration_fraction": 0.1,
                "penetration_atr": np.nan, "reclaim_margin_atr": np.nan, "hold_margin_atr": hold,
                "d_target_atr": d_t, "d_failure_atr": d_f, "minutes_to_session_end": 400 - 2 * i,
                "rv_30": rng.uniform(0.5, 1.5), "rv_60": rng.uniform(0.5, 1.5), "minutes_since_open": 930 + 2 * i,
                "label": label, "time_to_resolve": int(rng.integers(1, 240)), "mae_atr": rng.uniform(0, 1),
                "same_bar": False, "drop_reason": "",
            }
            for scale in ("5m", "15m", "1h"):
                row.update({f"ext_dir_{scale}": 1.0, f"int_dir_{scale}": 0.0, f"last_bos_dir_{scale}": -1.0})
            row.update({name: rng.normal() for name in FEATURE_NAMES})
            rows.append(row)
            n += 1
        day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_clock_instances_take_the_first_matching_step_per_path() -> None:
    frame = _instances(sessions=1, per_session=12)
    frame.loc[1, "sequence_id"] = frame.loc[0, "sequence_id"]  # same path, two held steps
    frame.loc[0, "step_kind"] = "reacceptance_held"
    frame.loc[2, "drop_reason"] = "no_target"
    out = clock_instances(frame, ("reacceptance_held",))
    assert out["sequence_id"].is_unique
    assert frame.loc[0, "known_at"] in set(out["known_at"]) and frame.loc[1, "known_at"] not in set(out["known_at"])
    assert frame.loc[2, "sequence_id"] not in set(out["sequence_id"])
    assert out["known_at"].is_monotonic_increasing


def test_planted_setup_effect_passes_and_writes_every_table(tmp_path: Path) -> None:
    verdict = run_setup_gate(
        _instances(effect=2.5), out_dir=tmp_path, primary=(9, 2), rolling=(6, 2, 2),
        models=("logistic",), minimum_oos_rows=50,
    )
    assert (tmp_path / "results.csv").exists() and (tmp_path / "verdict.csv").exists()
    assert (tmp_path / "descriptives.csv").exists()
    results = pd.read_csv(tmp_path / "results.csv")
    assert {"m0_loss", "m1_loss", "m2_loss", "m0_accuracy", "m1_accuracy", "analytic_loss", "delta_logloss"} <= set(results.columns)
    assert set(results["target"]) == {"target_first"}
    assert set(verdict["clock"]) == {f"{k}:{c}" for k in ("K0", "K1", "K2") for c in ("zone_return", "pool_reversal")}
    judged = verdict[verdict["judged"]]
    assert set(judged["clock"]) == {f"{k}:{c}" for k in ("K0", "K1") for c in ("zone_return", "pool_reversal")}
    assert not verdict.loc[verdict["clock"].str.startswith("K2"), "judged"].any()  # no K2 rows at all
    assert (judged["primary_delta"] < 0).all()
    assert judged["PASS"].any()
    descriptives = pd.read_csv(tmp_path / "descriptives.csv")
    assert {"clock", "rows", "share_target", "share_failure", "share_censored", "analytic_loss",
            "time_to_resolve_p50_target", "mae_atr_p50_failure"} <= set(descriptives.columns)


def test_no_effect_fails(tmp_path: Path) -> None:
    verdict = run_setup_gate(
        _instances(effect=0.0), out_dir=tmp_path, primary=(9, 2), rolling=(6, 2, 2),
        models=("logistic",), minimum_oos_rows=50,
    )
    assert not verdict["PASS"].any()


def test_thin_and_one_sided_cells_are_refused_not_judged(tmp_path: Path) -> None:
    frame = _instances(effect=2.5)
    frame.loc[frame["context_kind"] == "pool_reversal", "label"] = TARGET_FIRST  # one-sided
    verdict = run_setup_gate(
        frame, out_dir=tmp_path, primary=(9, 2), rolling=(6, 2, 2), models=("logistic",), minimum_oos_rows=100_000,
    )
    assert not verdict["judged"].any()
    populated = verdict[~verdict["clock"].str.startswith("K2")]  # K2 has no synthetic instances at all
    assert set(populated.loc[populated["clock"].str.endswith("pool_reversal"), "refusal"]) == {"class_share"}
    assert set(populated.loc[populated["clock"].str.endswith("zone_return"), "refusal"]) == {"oos_rows"}
    assert set(verdict.loc[verdict["clock"].str.startswith("K2"), "refusal"]) == {"no_rows"}
    assert not verdict["PASS"].any()
