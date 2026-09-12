"""The gate runs end to end on a synthetic tape and writes every table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from brain.research.event_log import EVENT_COLUMNS
from brain.scripts.information_gain_gate import run_gate


def _synthetic_data(sessions: int = 14, per_session: int = 240, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    rows = sessions * per_session
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2022-01-03T09:30", tz="America/New_York")
    for _ in range(sessions):
        while day.weekday() >= 5:
            day += pd.Timedelta(days=1)
        stamps += [day + pd.Timedelta(minutes=i) for i in range(per_session)]
        day += pd.Timedelta(days=1)
    index = pd.DatetimeIndex(stamps).tz_convert("UTC")
    closes = 100 + np.cumsum(rng.normal(0, 0.3, rows))
    prices = np.column_stack([closes, closes + 0.2, closes - 0.2, np.full(rows, 1.0)])
    steps = rng.normal(0, 0.3, (rows, 60)).cumsum(axis=1)
    future_closes = closes[:, None] + steps
    kinds = rng.choice(["sweep_confirmed", "qualified_bos", "fvg_created"], rows // 5)
    scales = rng.choice(["1m", "5m", "15m"], rows // 5, p=[0.6, 0.3, 0.1])
    events = pd.DataFrame(
        [
            {
                "known_at": index[i * 5], "kind": kinds[i], "timeframe": scales[i],
                "direction": "long" if i % 2 else "short", "side": None, "strength": 1.0,
                "price": None, "entity_id": f"e{i}", "lifecycle": None, "event_id": f"ev{i}",
            }
            for i in range(rows // 5)
        ],
        columns=list(EVENT_COLUMNS),
    )
    return {
        "index": index,
        "features": rng.normal(size=(rows, 150)),
        "prices": prices,
        "future_closes": future_closes,
        "future_highs": future_closes + 0.3,
        "future_lows": future_closes - 0.3,
        "volumes": np.ones(rows),
        "events": events,
    }


def test_run_gate_writes_results_ablation_and_verdict(tmp_path: Path) -> None:
    verdict = run_gate(
        _synthetic_data(),
        out_dir=tmp_path,
        clocks={"C1": 1, "C5": 5},
        descriptive={"ALL": 0},
        primary=(9, 2),
        rolling=(6, 2, 2),
        models=("logistic",),
        ablation_kinds=("sweep_confirmed",),
    )
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "verdict.csv").exists()
    assert (tmp_path / "ablation.csv").exists()
    results = pd.read_csv(tmp_path / "results.csv")
    assert set(results["clock"]) == {"C1", "C5", "ALL"}
    assert set(results.loc[results["clock"] == "ALL", "primary"]) == {True}
    assert {"fp_1.0_1.0", "fp_1.0_0.5", "fp_0.5_1.0", "asymmetry_60", "range_60"} <= set(results["target"])
    assert set(verdict["clock"]) == {"C1", "C5"}
    assert set(verdict["target"]) == {"fp_1.0_1.0", "fp_1.0_0.5", "fp_0.5_1.0"}
    assert len(verdict) == 6
    assert verdict["PASS"].dtype == bool
    ablation = pd.read_csv(tmp_path / "ablation.csv")
    assert set(ablation["kind"]) == {"sweep_confirmed"}
    assert set(ablation["clock"]) == {"C1", "C5"}
